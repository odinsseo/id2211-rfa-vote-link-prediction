"""Wikipedia revision parsing functionality."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from wiki_base_parser import CSVWriter, DumpParser, DumpProcessor
from wiki_common import UsernameHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RevisionEntry:
    """Data class representing a single revision entry."""

    source: str = ""
    target: str = ""
    timestamp: str = ""
    minor: int = 0
    textdata: int = 0

    def to_row(self) -> List:
        """Convert the entry to a CSV row."""
        return [
            self.source.lower(),
            self.target.lower(),
            self.timestamp,
            self.minor,
            self.textdata,
        ]

    def is_valid(self) -> bool:
        """Check if the entry has valid source and target."""
        return bool(self.source and self.target)


class RevisionParser(DumpParser):
    """Parses revision entries from Wikipedia dump."""

    def __init__(self, username_handler: UsernameHandler):
        """Initialize with username handler."""
        self.username_handler = username_handler

    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse a REVISION line.

        Args:
            line: A single line from the dump file

        Returns:
            Dictionary with parsed data or None if parsing failed
        """
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
        """Parse a complete revision entry.

        Args:
            lines: List of lines making up a revision entry

        Returns:
            RevisionEntry object with parsed data
        """
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


class RevisionProcessor(DumpProcessor):
    """Process Wikipedia dumps and extract revision data."""

    def __init__(self, parser: RevisionParser, writer: CSVWriter):
        """Initialize with parser and writer instances."""
        super().__init__(parser, writer)
        self.revision_parser = parser

    def should_process_entry(self, entry: RevisionEntry) -> bool:
        """Check if a revision entry should be processed.

        Filters out bot-generated entries.

        Args:
            entry: The revision entry to check

        Returns:
            True if neither source nor target is a bot
        """
        return not (
            self.revision_parser.username_handler.is_bot(entry.source)
            or self.revision_parser.username_handler.is_bot(entry.target)
        )
