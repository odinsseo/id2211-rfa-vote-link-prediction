#!/usr/bin/env python3
"""Parser for Wikipedia admin election data."""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class NominationEntry:
    """Wrapper for nomination entries to implement DumpEntry protocol."""
    nominator: str
    nominee: str
    close_time: str
    outcome: int

    def to_row(self) -> List:
        """Convert to CSV row format."""
        return [self.nominator, self.nominee, self.close_time, self.outcome]

    def is_valid(self) -> bool:
        """All fields are required."""
        return all([self.nominator, self.nominee, self.close_time, self.outcome is not None])

@dataclass
class VoteEntry:
    """Wrapper for vote entries to implement DumpEntry protocol."""
    voter: str
    candidate: str
    vote: int
    vote_time: str
    close_time: str

    def to_row(self) -> List:
        """Convert to CSV row format."""
        return [self.voter, self.candidate, self.vote, self.vote_time, self.close_time]

    def is_valid(self) -> bool:
        """All fields are required."""
        return all([self.voter, self.candidate, self.vote_time, self.close_time])

from wiki_base_parser import CSVWriter, DumpParser, DumpProcessor
from wiki_common import Cache, UsernameHandler, WikiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Vote:
    """Single vote in an election."""

    voter: str
    value: int
    timestamp: str

    @property
    def is_support(self) -> bool:
        """Check if this is a supporting vote."""
        return self.value > 0

    @property
    def is_oppose(self) -> bool:
        """Check if this is an opposing vote."""
        return self.value < 0

    def to_row(self) -> List:
        """Convert to CSV row format."""
        return [self.voter, None, self.value, self.timestamp, None]  # nominee and close_time set later


@dataclass
class Election:
    """Wikipedia admin election data."""

    outcome: Optional[int] = None
    close_time: Optional[str] = None
    nominee: Optional[str] = None
    nominator: Optional[str] = None
    votes: List[Vote] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the election has all required fields."""
        return all(
            [
                self.outcome is not None,
                self.close_time is not None,
                self.nominee is not None,
                self.nominator is not None,
            ]
        )

    @property
    def support_count(self) -> int:
        """Count supporting votes."""
        return sum(1 for vote in self.votes if vote.is_support)

    @property
    def oppose_count(self) -> int:
        """Count opposing votes."""
        return sum(1 for vote in self.votes if vote.is_oppose)

    def add_vote(self, vote: Vote) -> None:
        """Add a vote to this election."""
        self.votes.append(vote)

    def to_row(self) -> List:
        """Convert to CSV row format."""
        return [self.nominator, self.nominee, self.close_time, self.outcome]


class ElectionParser(DumpParser):
    """Parser for Wikipedia election data."""

    def __init__(self, username_handler: UsernameHandler):
        """Initialize with username handler."""
        self.username_handler = username_handler

    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse a line from the election data.

        Args:
            line: A single line from the election file

        Returns:
            Dictionary with parsed data or None if parsing failed
        """
        parts = line.split("\t")
        if len(parts) < 2:
            return None

        code = parts[0]
        data = {"code": code}

        if code == "E":
            data["outcome"] = int(parts[1])
        elif code == "T":
            data["close_time"] = parts[1]
        elif code == "U" and len(parts) >= 3:
            data["nominee"] = self.username_handler.normalize(parts[2])
        elif code == "N" and len(parts) >= 3:
            data["nominator"] = self.username_handler.normalize(parts[2])
        elif code == "V" and len(parts) >= 5:
            data.update(
                {
                    "voter": self.username_handler.normalize(parts[4]),
                    "value": int(parts[1]),
                    "timestamp": parts[3],
                }
            )

        return data

    def parse_entry(self, lines: List[str]) -> Election:
        """Parse a complete election entry.

        Args:
            lines: List of lines making up an election entry

        Returns:
            Election object with parsed data
        """
        election = Election()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if data := self.parse_line(line):
                code = data["code"]

                if code == "E":
                    election.outcome = data["outcome"]
                elif code == "T":
                    election.close_time = data["close_time"]
                elif code == "U":
                    election.nominee = data["nominee"]
                elif code == "N":
                    election.nominator = data["nominator"]
                elif code == "V":
                    voter = data["voter"]
                    if not self.username_handler.is_bot(voter):
                        vote = Vote(
                            voter=voter,
                            value=data["value"],
                            timestamp=data["timestamp"],
                        )
                        election.add_vote(vote)

        return election


class ElectionProcessor(DumpProcessor):
    """Process Wikipedia election dumps."""

    def __init__(
        self,
        parser: ElectionParser,
        nominations_writer: CSVWriter,
        votes_writer: CSVWriter,
    ):
        """Initialize with parser and writers."""
        super().__init__(parser, nominations_writer)
        self.votes_writer = votes_writer
        self.election_parser = parser

    def _split_entries(self, content: str) -> List[List[str]]:
        """Split content into election entries."""
        content = self._remove_comments(content)
        entries = []
        current_entry = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                if current_entry:
                    entries.append(current_entry)
                    current_entry = []
            else:
                current_entry.append(line)

        if current_entry:
            entries.append(current_entry)

        return entries

    def _remove_comments(self, content: str) -> str:
        """Remove comment lines from content."""
        return "\n".join(
            line for line in content.split("\n") if not line.strip().startswith("#")
        )

    def process_file(self, input_file: str, chunk_size: int = 10000) -> None:
        """Process election file and generate data.

        Args:
            input_file: Path to input file
            chunk_size: Number of entries to process in each chunk
        """
        logger.info(f"Processing file: {input_file}")

        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            with open(input_file, "r", encoding="utf-8", errors="surrogateescape") as f:
                content = f.read()

            # Split into election entries
            entries = self._split_entries(content)

            # Initialize counters
            processed = invalid = 0
            nominations_chunk = []
            votes_chunk = []

            # Create output files
            self.writer.create_csv()  # Nominations file
            self.votes_writer.create_csv()

            # Process entries
            for entry_lines in tqdm(entries, desc="Processing elections"):
                election = self.parser.parse_entry(entry_lines)

                if not election.is_valid:
                    invalid += 1
                    continue

                # Add to nominations chunk
                nomination = NominationEntry(
                    nominator=election.nominator,
                    nominee=election.nominee,
                    close_time=election.close_time,
                    outcome=election.outcome
                )
                nominations_chunk.append(nomination)

                # Add to votes chunk
                for vote in election.votes:
                    vote_entry = VoteEntry(
                        voter=vote.voter,
                        candidate=election.nominee,
                        vote=vote.value,
                        vote_time=vote.timestamp,
                        close_time=election.close_time
                    )
                    votes_chunk.append(vote_entry)

                processed += 1

            # Write chunks if full
            if len(nominations_chunk) >= chunk_size:
                try:
                    self.writer.write_entries(nominations_chunk)
                    self.votes_writer.write_entries(votes_chunk)
                except UnicodeEncodeError as e:
                    # Filter out problematic characters
                    nominations_chunk = [[str(field).encode('ascii', 'ignore').decode() for field in entry.to_row()] for entry in nominations_chunk]
                    votes_chunk = [[str(field).encode('ascii', 'ignore').decode() for field in entry.to_row()] for entry in votes_chunk]
                    self.writer.write_entries(nominations_chunk)
                    self.votes_writer.write_entries(votes_chunk)
                nominations_chunk = []
                votes_chunk = []

            # Write remaining chunks
            if nominations_chunk:
                try:
                    self.writer.write_entries(nominations_chunk)
                    self.votes_writer.write_entries(votes_chunk)
                except UnicodeEncodeError as e:
                    # Filter out problematic characters
                    nominations_chunk = [[str(field).encode('ascii', 'ignore').decode() for field in entry.to_row()] for entry in nominations_chunk]
                    votes_chunk = [[str(field).encode('ascii', 'ignore').decode() for field in entry.to_row()] for entry in votes_chunk]
                    self.writer.write_entries(nominations_chunk)
                    self.votes_writer.write_entries(votes_chunk)

            logger.info(f"Complete: {processed} processed, {invalid} invalid")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


def parse_elections(
    input_file: str, nominations_file: str, votes_file: str, cache_dir: str = "./cache"
) -> bool:
    """Parse Wikipedia election data and export to CSV files.

    Args:
        input_file: Path to the input file
        nominations_file: Path for the nominations CSV
        votes_file: Path for the votes CSV
        cache_dir: Directory for caching

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize components
        cache = Cache(cache_dir)
        wiki_api = WikiAPI(cache)
        username_handler = UsernameHandler(wiki_api)

        # Create parser and writers
        parser = ElectionParser(username_handler)
        nominations_writer = CSVWriter(
            nominations_file, headers=["nominator", "nominee", "close_time", "outcome"]
        )
        votes_writer = CSVWriter(
            votes_file,
            headers=["voter", "candidate", "vote", "vote_time", "close_time"],
        )

        # Process elections
        processor = ElectionProcessor(parser, nominations_writer, votes_writer)
        processor.process_file(input_file)

        return True

    except Exception as e:
        logger.error(f"Error processing elections: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Wikipedia admin election data"
    )
    parser.add_argument(
        "input_file", help="Path to the Wikipedia Admin Election dump file"
    )
    parser.add_argument("nominations_file", help="Path to save nominations CSV file")
    parser.add_argument("votes_file", help="Path to save individual votes CSV file")
    parser.add_argument(
        "--cache-dir", default="./cache", help="Directory to store cache files"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    # Process elections
    success = parse_elections(
        args.input_file, args.nominations_file, args.votes_file, args.cache_dir
    )

    if success:
        print("Processing completed successfully.")
        print(f"Files created:")
        print(f"  - {args.nominations_file} (nominations)")
        print(f"  - {args.votes_file} (votes)")
    else:
        print("Processing failed. Check the logs for details.")
        exit(1)


if __name__ == "__main__":
    main()
