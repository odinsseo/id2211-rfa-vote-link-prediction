"""Base functionality for Wikipedia dump processing."""

import bz2
import csv
import gzip
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, TextIO, TypeVar

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


class DumpEntry(Protocol):
    """Protocol for dump entries."""

    def to_row(self) -> List:
        """Convert the entry to a CSV row."""
        ...

    def is_valid(self) -> bool:
        """Check if the entry is valid."""
        ...


class CSVWriter:
    """Handles writing data entries to CSV files."""

    def __init__(self, output_file: str, headers: List[str]):
        """Initialize with output file path."""
        self.output_file = Path(output_file)
        self.headers = headers

    def create_csv(self) -> None:
        """Create a CSV file with header row."""
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)

    def write_entries(self, entries: List[DumpEntry]) -> None:
        """Write entries to CSV file.

        Args:
            entries: List of entry objects to write
        """
        # Create the file in binary mode to handle Unicode properly
        with open(self.output_file, "a", newline="", encoding="utf-8", errors="ignore") as csvfile:
            writer = csv.writer(csvfile)
            for entry in entries:
                try:
                    # Clean any problematic characters from the row data
                    row = []
                    for field in entry.to_row():
                        if isinstance(field, str):
                            # Remove any non-UTF8 characters
                            field = field.encode('utf-8', 'ignore').decode('utf-8')
                        row.append(field)
                    writer.writerow(row)
                except Exception as e:
                    logger.warning(f"Error writing entry: {e}")
                    continue


class DumpParser(ABC):
    """Base class for dump file parsers."""

    @abstractmethod
    def parse_line(self, line: str) -> Optional[dict]:
        """Parse a single line from the dump file.

        Args:
            line: A line from the dump file

        Returns:
            Parsed data dictionary or None if parsing failed
        """
        pass

    @abstractmethod
    def parse_entry(self, lines: List[str]) -> DumpEntry:
        """Parse a complete entry from multiple lines.

        Args:
            lines: List of lines making up an entry

        Returns:
            Parsed entry object
        """
        pass


class DumpProcessor:
    """Base class for dump file processors."""

    def __init__(self, parser: DumpParser, writer: CSVWriter):
        """Initialize with parser and writer instances."""
        self.parser = parser
        self.writer = writer

    def _open_file(self, filename: str) -> TextIO:
        """Open a file with appropriate handler based on extension."""
        if filename.endswith(".gz"):
            return gzip.open(filename, "rt", encoding="utf-8", errors="replace")
        elif filename.endswith(".bz2"):
            return bz2.open(filename, "rt", encoding="utf-8", errors="replace")
        return open(filename, "r", encoding="utf-8", errors="replace")

    def process_file(self, input_file: str, chunk_size: int = 10000) -> None:
        """Process dump file and generate data.

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
                        elif self.should_process_entry(entry):
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

                # Process final entry and write remaining chunk
                if current_lines:
                    entry = self.parser.parse_entry(current_lines)
                    if not entry.is_valid():
                        invalid += 1
                    elif self.should_process_entry(entry):
                        current_chunk.append(entry)
                        processed += 1
                    else:
                        skipped += 1

                if current_chunk:
                    self.writer.write_entries(current_chunk)

            logger.info(
                f"Complete: {processed} processed, {skipped} skipped, {invalid} invalid"
            )

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise

    def should_process_entry(self, entry: DumpEntry) -> bool:
        """Determine if an entry should be processed.

        Can be overridden by subclasses to implement custom filtering.

        Args:
            entry: The entry to check

        Returns:
            True if the entry should be processed, False otherwise
        """
        return True
