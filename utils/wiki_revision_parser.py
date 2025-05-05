"""Wikipedia revision parsing functionality."""

import csv
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from tqdm import tqdm

from .wiki_common import UsernameHandler

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


class RevisionParser:
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


class CSVWriter:
    """Handles writing revision data to CSV files."""

    def __init__(self, output_file: str):
        """Initialize with output file path."""
        self.output_file = Path(output_file)

    def create_csv(self) -> None:
        """Create a CSV file with header row."""
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["source", "target", "timestamp", "minor", "textdata"])

    def write_entries(self, entries: List[RevisionEntry]) -> None:
        """Write entries to CSV file.

        Args:
            entries: List of RevisionEntry objects to write
        """
        with open(self.output_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for entry in entries:
                writer.writerow(entry.to_row())


class DumpProcessor:
    """Process Wikipedia dumps and extract revision data."""

    def __init__(self, parser: RevisionParser, writer: CSVWriter):
        """Initialize with parser and writer instances."""
        self.parser = parser
        self.writer = writer

    def _open_file(self, filename: str) -> TextIO:
        """Open a file with appropriate handler based on extension."""
        if filename.endswith(".gz"):
            return gzip.open(filename, "rt", encoding="utf-8", errors="replace")
        return open(filename, "r", encoding="utf-8", errors="replace")

    def process_file(self, input_file: str, chunk_size: int = 10000) -> None:
        """Process dump file and generate interaction data.

        Args:
            input_file: Path to input dump file
            chunk_size: Number of entries to process in each chunk
        """
        logger.info(f"Processing file: {input_file}")

        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return

        # Initialize counters and buffers
        processed = skipped = invalid = 0
        current_chunk = []
        current_lines = []

        # Create output file
        self.writer.create_csv()

        try:
            with self._open_file(input_file) as file:
                for line in tqdm(file, desc="Processing dump"):
                    line = line.strip()

                    # Process completed entry when new REVISION line is found
                    if line.startswith("REVISION") and current_lines:
                        entry = self.parser.parse_entry(current_lines)

                        if not entry.is_valid():
                            invalid += 1
                        elif not self.parser.username_handler.is_bot(
                            entry.source
                        ) and not self.parser.username_handler.is_bot(entry.target):
                            current_chunk.append(entry)
                            processed += 1
                        else:
                            skipped += 1

                        # Start new entry
                        current_lines = [line]

                        # Write chunk if full
                        if len(current_chunk) >= chunk_size:
                            self.writer.write_entries(current_chunk)
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
                    elif not self.parser.username_handler.is_bot(
                        entry.source
                    ) and not self.parser.username_handler.is_bot(entry.target):
                        current_chunk.append(entry)
                        processed += 1
                    else:
                        skipped += 1

                # Write remaining entries
                if current_chunk:
                    self.writer.write_entries(current_chunk)

            logger.info(
                f"Complete: {processed} processed, {skipped} skipped, {invalid} invalid"
            )

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
