"""Common functionality for Wikipedia data processing."""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Set

import mwclient
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Constants
CUT_OFF = "2008-01-04T00:00:00Z"
MIN_EXPECTED_BOTS = 15
FALLBACK_BOTS = {
    "ClueBot",
    "VoABot II",
    "SineBot",
    "COIBot",
    "SpellBot",
    "AvicBot",
    "CmdrObot",
    "TawkerBot",
    "SieBot",
    "SmackBot",
    "YurikBot",
    "BetacommandBot",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Cache:
    """Cache handler for storing and retrieving data."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize cache with directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load(self, filename: str, default=None):
        """Load data from cache file."""
        try:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Cache load error ({filename}): {e}")
            return default

    def save(self, filename: str, data) -> bool:
        """Save data to cache file."""
        try:
            with open(self.cache_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Cache save error ({filename}): {e}")
            return False


class WikiAPI:
    """Wikipedia API connector and data fetcher."""

    def __init__(self, cache: Cache):
        """Initialize with cache handler."""
        self.cache = cache
        self.site = None
        self._bots = None
        self._username_changes = None

    def connect(self) -> Optional[mwclient.Site]:
        """Connect to Wikipedia API with retries."""
        if self.site:
            return self.site

        retries = 3
        for attempt in range(retries):
            try:
                self.site = mwclient.Site(
                    "en.wikipedia.org", clients_useragent="WikiGraphBuilder/1.0"
                )
                return self.site
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt+1} failed: {e}")
        return None

    @property
    def bots(self) -> Set[str]:
        """Get bot list (cached)."""
        if self._bots is None:
            self._bots = self._get_bots()
        return self._bots

    def _get_bots(self) -> Set[str]:
        """Load or fetch bot list."""
        cached_bots = self.cache.load("bots.json", [])
        if cached_bots and len(cached_bots) >= MIN_EXPECTED_BOTS:
            logger.info(f"Using {len(cached_bots)} bots from cache")
            return set(cached_bots)

        try:
            site = self.connect()
            bots = set()

            # Current bots
            members = site.allusers(group="bot", api_chunk_size=500)
            bots.update(user["name"] for user in members)

            # Deflagged bots from wiki pages
            for page in [
                "Wikipedia:Bots/Status/inactive_bots_1",
                "Wikipedia:Bots/Status/inactive_bots_2",
            ]:
                try:
                    response = requests.get(f"https://en.wikipedia.org/wiki/{page}")
                    soup = BeautifulSoup(response.text, "html.parser")

                    for table in soup.find_all("table", class_="wikitable"):
                        for row in table.find_all("tr")[1:]:  # Skip header
                            if link := row.find("td").find("a"):
                                bots.add(link.text.strip())
                except Exception as e:
                    logger.error(f"Error fetching bots from {page}: {e}")

            if len(bots) >= MIN_EXPECTED_BOTS:
                self.cache.save("bots.json", list(bots))
                return bots

            logger.warning(f"Bot list too small ({len(bots)}), using fallback")
            return FALLBACK_BOTS

        except Exception as e:
            logger.error(f"Error fetching bots: {e}")
            return FALLBACK_BOTS

    @property
    def username_changes(self) -> Dict[str, str]:
        """Get username changes (cached)."""
        if self._username_changes is None:
            self._username_changes = self._get_username_changes()
        return self._username_changes

    def _get_username_changes(self) -> Dict[str, str]:
        """Load or fetch username changes."""
        cached_changes = self.cache.load("username_changes.json", {})
        if cached_changes:
            logger.info(f"Using {len(cached_changes)} username changes from cache")
            return cached_changes

        try:
            site = self.connect()
            events = []

            # Fetch rename events from logs
            for ev in tqdm(
                site.logevents(
                    type="renameuser",
                    start=None,
                    end=CUT_OFF,
                    dir="older",
                    prop="ids|timestamp|user|details",
                    api_chunk_size=500,
                ),
                desc="Fetching rename logs",
            ):
                events.append(
                    {
                        "old": ev["params"]["olduser"],
                        "new": ev["params"]["newuser"],
                        "timestamp": ev["timestamp"],
                    }
                )

            # Create mapping of old -> new usernames
            events.sort(key=lambda e: e["timestamp"], reverse=True)
            mapping = {}
            for ev in events:
                latest = mapping.get(ev["new"], ev["new"])
                mapping[ev["old"]] = latest

            self.cache.save("username_changes.json", mapping)
            return mapping

        except Exception as e:
            logger.error(f"Error fetching username changes: {e}")
            return {}


class UsernameHandler:
    """Handles username normalization and bot detection."""

    def __init__(self, wiki_api: WikiAPI):
        """Initialize with WikiAPI instance."""
        self.wiki_api = wiki_api
        self.bot_cache = {}

    def normalize(self, username: str) -> str:
        """Normalize a username removing IP prefix and applying renames."""
        if not username:
            return ""

        # Remove prefix
        lower_username = username.lower()
        if lower_username.startswith("ip:"):
            username = username[3:]
        elif lower_username.startswith("user:"):
            username = username[5:]

        # Apply username changes
        return self.wiki_api.username_changes.get(username, username)

    @lru_cache(maxsize=10000)
    def is_bot(self, username: str) -> bool:
        """Check if username belongs to a bot."""
        if not username:
            return False

        # Regular expression pattern
        # Matches:
        # - "bot" (case insensitive)
        # - Followed by any combination of numbers and special characters (or none)
        # - At the end of the string
        pattern = r"bot[\d\W]*$"

        # Compile the pattern with case insensitivity flag
        bot_regex = re.compile(pattern, re.IGNORECASE)

        normalized = self.normalize(username)
        return normalized in self.wiki_api.bots or bool(bot_regex.search(normalized))
