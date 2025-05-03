#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO

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


@dataclass
class RevisionEntry:
    """Data class representing a single revision entry"""

    source: str = ""
    target: str = ""
    timestamp: str = ""
    minor: int = 0
    textdata: int = 0

    def to_row(self) -> List:
        """Convert the entry to a CSV row"""
        return [
            self.source.lower(),
            self.target.lower(),
            self.timestamp,
            self.minor,
            self.textdata,
        ]

    def is_valid(self) -> bool:
        """Check if the entry has valid source and target"""
        return bool(self.source and self.target)


class Cache:
    """Cache handler for storing and retrieving data"""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load(self, filename: str, default=None):
        """Load data from cache file"""
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
        """Save data to cache file"""
        try:
            with open(self.cache_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Cache save error ({filename}): {e}")
            return False


class WikiAPI:
    """Wikipedia API connector and data fetcher"""

    def __init__(self, cache: Cache):
        self.cache = cache
        self.site = None
        self._bots = None
        self._username_changes = None

    def connect(self) -> mwclient.Site:
        """Connect to Wikipedia API with retries"""
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

        return None  # Should not reach here

    @property
    def bots(self) -> Set[str]:
        """Get bot list (cached)"""
        if self._bots is None:
            self._bots = self._get_bots()
        return self._bots

    def _get_bots(self) -> Set[str]:
        """Load or fetch bot list"""
        cached_bots = self.cache.load("bots.json", [])
        if cached_bots and len(cached_bots) >= MIN_EXPECTED_BOTS:
            logger.info(f"Using {len(cached_bots)} bots from cache")
            return set(cached_bots)

        # Fetch from API
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
        """Get username changes (cached)"""
        if self._username_changes is None:
            self._username_changes = self._get_username_changes()
        return self._username_changes

    def _get_username_changes(self) -> Dict[str, str]:
        """Load or fetch username changes"""
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
    """Handles username normalization and bot detection"""

    def __init__(self, wiki_api: WikiAPI):
        self.wiki_api = wiki_api
        self.bot_cache = {}

    def normalize(self, username: str) -> str:
        """Normalize a username removing IP prefix and applying renames"""
        if not username:
            return ""

        # Remove IP prefix
        if username.startswith("ip:"):
            username = username[3:]

        # Apply username changes
        return self.wiki_api.username_changes.get(username, username)

    @lru_cache(maxsize=10000)
    def is_bot(self, username: str) -> bool:
        """Check if username belongs to a bot"""
        if not username:
            return False

        normalized = self.normalize(username)
        return normalized in self.wiki_api.bots


class RevisionParser:
    """Parses revision entries from Wikipedia dump"""

    def __init__(self, username_handler: UsernameHandler):
        self.username_handler = username_handler

    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse a REVISION line"""
        fields = line.split(" ")
        if len(fields) < 6:
            return None

        # Extract article title and target username
        article_title = fields[3]
        target_username = ""

        if ":" in article_title:
            parts = article_title.split(":")
            if len(parts) > 1:
                if "/" in parts[1]:
                    target_username = parts[1].split("/")[0]  # Remove subpages
                else:
                    target_username = parts[1]

        # Extract timestamp and source username
        timestamp = fields[4]
        source_username = " ".join(fields[5:-1]) if len(fields) > 6 else fields[5]

        return {
            "target_username": target_username,
            "timestamp": timestamp,
            "source_username": source_username,
        }

    def parse_entry(self, lines: List[str]) -> RevisionEntry:
        """Parse a complete revision entry"""
        entry = RevisionEntry()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("REVISION"):
                if revision_data := self.parse_line(line):
                    entry.source = revision_data["source_username"]
                    entry.target = revision_data["target_username"]
                    entry.timestamp = revision_data["timestamp"]

            elif line.startswith("MINOR"):
                try:
                    entry.minor = int(line.split(" ")[1])
                except (IndexError, ValueError):
                    pass

            elif line.startswith("TEXTDATA"):
                try:
                    entry.textdata = int(line.split(" ")[1])
                except (IndexError, ValueError):
                    pass

        # Normalize usernames
        entry.source = self.username_handler.normalize(entry.source)
        entry.target = self.username_handler.normalize(entry.target)

        return entry


class WikiDumpProcessor:
    """Process Wikipedia dumps and generate interaction graph"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache = Cache(cache_dir)
        self.wiki_api = WikiAPI(self.cache)
        self.username_handler = UsernameHandler(self.wiki_api)
        self.parser = RevisionParser(self.username_handler)

    def _open_file(self, filename: str) -> TextIO:
        """Open a file with appropriate handler based on extension"""
        if filename.endswith(".gz"):
            return gzip.open(filename, "rt", encoding="utf-8", errors="replace")
        return open(filename, "r", encoding="utf-8", errors="replace")

    def _create_csv(self, filename: str) -> None:
        """Create a CSV file with header row"""
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["source", "target", "timestamp", "minor", "textdata"])

    def _write_entries(self, filename: str, entries: List[RevisionEntry]) -> None:
        """Write entries to CSV file"""
        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for entry in entries:
                writer.writerow(entry.to_row())

    def process_file(
        self, input_file: str, output_file: str, chunk_size: int = 10000
    ) -> None:
        """Process dump file and generate interaction graph"""
        logger.info(f"Processing file: {input_file}")

        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return

        # Initialize counters and buffers
        processed = 0
        skipped = 0
        invalid = 0
        current_chunk = []
        current_lines = []

        # Create output file
        self._create_csv(output_file)

        try:
            with self._open_file(input_file) as file:
                for line in tqdm(file, desc="Processing dump"):
                    line = line.strip()

                    # Process completed entry when new REVISION line is found
                    if line.startswith("REVISION") and current_lines:
                        entry = self.parser.parse_entry(current_lines)

                        if not entry.is_valid():
                            invalid += 1
                        elif not self.username_handler.is_bot(
                            entry.source
                        ) and not self.username_handler.is_bot(entry.target):
                            current_chunk.append(entry)
                            processed += 1
                        else:
                            skipped += 1

                        # Start new entry
                        current_lines = [line]

                        # Write chunk if full
                        if len(current_chunk) >= chunk_size:
                            self._write_entries(output_file, current_chunk)
                            current_chunk = []

                            # Log progress
                            if processed % (chunk_size * 10) == 0:
                                logger.info(
                                    f"Progress: {processed} processed, {skipped} skipped"
                                )
                    else:
                        current_lines.append(line)

                # Process final entry
                if current_lines:
                    entry = self.parser.parse_entry(current_lines)

                    if not entry.is_valid():
                        invalid += 1
                    elif not self.username_handler.is_bot(
                        entry.source
                    ) and not self.username_handler.is_bot(entry.target):
                        current_chunk.append(entry)
                        processed += 1
                    else:
                        skipped += 1

                # Write remaining entries
                if current_chunk:
                    self._write_entries(output_file, current_chunk)

            logger.info(
                f"Complete: {processed} processed, {skipped} skipped, {invalid} invalid"
            )

        except Exception as e:
            logger.error(f"Error processing file: {e}")

    def build_graph(
        self, input_file: str, output_file: str, chunk_size: int = 10000
    ) -> None:
        """Main entry point to build interaction graph"""
        try:
            # Process will automatically initialize the API and caches
            self.process_file(input_file, output_file, chunk_size)
            logger.info(f"Graph data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error building graph: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Wikipedia user interaction graph from User Talk pages dump"
    )
    parser.add_argument(
        "input_file", help="Path to the Wikipedia User Talk pages dump file"
    )
    parser.add_argument("output_file", help="Path to save the output CSV file")
    parser.add_argument(
        "--cache-dir", default="./cache", help="Directory to store cache files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of entries to process in each chunk",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    # Create and run the processor
    processor = WikiDumpProcessor(cache_dir=args.cache_dir)
    processor.build_graph(args.input_file, args.output_file, args.chunk_size)


if __name__ == "__main__":
    main()
