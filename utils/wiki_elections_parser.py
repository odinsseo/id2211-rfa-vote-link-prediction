#!/usr/bin/env python3
"""Parser for Wikipedia admin election data."""

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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


class ElectionParser:
    """Parser for Wikipedia election data."""

    def __init__(self, username_handler: UsernameHandler):
        """Initialize with username handler."""
        self.username_handler = username_handler

    def parse_file(self, input_path: Path) -> List[Election]:
        """Parse elections from input file.

        Args:
            input_path: Path to the input file

        Returns:
            List of parsed Election objects
        """
        logger.info(f"Parsing elections from {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with open(
                input_path, "r", encoding="utf-8", errors="surrogateescape"
            ) as file:
                content = file.read()

            # Remove comments and split into election entries
            content = self._remove_comments(content)
            election_entries = self._split_entries(content)

            # Parse each election entry
            elections = []
            for entry in election_entries:
                if not entry.strip():
                    continue

                try:
                    election = self._parse_entry(entry)
                    if election.is_valid:
                        elections.append(election)
                except Exception as e:
                    logger.warning(f"Error parsing election entry: {e}")

            logger.info(f"Successfully parsed {len(elections)} elections")
            return elections

        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def _remove_comments(self, content: str) -> str:
        """Remove comment lines from content."""
        return "\n".join(
            line for line in content.split("\n") if not line.strip().startswith("#")
        )

    def _split_entries(self, content: str) -> List[str]:
        """Split content into individual election entries."""
        return re.split(r"\n\s*\n", content)

    def _parse_entry(self, entry: str) -> Election:
        """Parse a single election entry."""
        lines = entry.strip().split("\n")
        election = Election()

        for line in lines:
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            code = parts[0]

            if code == "E":
                election.outcome = int(parts[1])
            elif code == "T":
                election.close_time = parts[1]
            elif code == "U" and len(parts) >= 3:
                election.nominee = self.username_handler.normalize(parts[2])
            elif code == "N" and len(parts) >= 3:
                election.nominator = self.username_handler.normalize(parts[2])
            elif code == "V" and len(parts) >= 5:
                voter = self.username_handler.normalize(parts[4])
                if not self.username_handler.is_bot(voter):
                    vote = Vote(voter=voter, value=int(parts[1]), timestamp=parts[3])
                    election.add_vote(vote)

        return election


class ElectionExporter:
    """Exports election data to CSV files."""

    def __init__(self, nominations_path: Path, votes_path: Path):
        """Initialize with output paths."""
        self.nominations_path = nominations_path
        self.votes_path = votes_path

    def export(self, elections: List[Election]) -> None:
        """Export elections to CSV files.

        Args:
            elections: List of Election objects to export
        """
        self._export_nominations(elections)
        self._export_votes(elections)

    def _export_nominations(self, elections: List[Election]) -> None:
        """Export nominations data."""
        with open(self.nominations_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nominator", "nominee", "close_time", "outcome"])

            for election in elections:
                writer.writerow(
                    [
                        election.nominator,
                        election.nominee,
                        election.close_time,
                        election.outcome,
                    ]
                )

        logger.info(f"Exported {len(elections)} nominations to {self.nominations_path}")

    def _export_votes(self, elections: List[Election]) -> None:
        """Export individual votes data."""
        vote_count = 0
        with open(self.votes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["voter", "candidate", "vote", "vote_time", "close_time"])

            for election in elections:
                for vote in election.votes:
                    writer.writerow(
                        [
                            vote.voter,
                            election.nominee,
                            vote.value,
                            vote.timestamp,
                            election.close_time,
                        ]
                    )
                    vote_count += 1

        logger.info(f"Exported {vote_count} votes to {self.votes_path}")


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

        # Parse elections
        parser = ElectionParser(username_handler)
        elections = parser.parse_file(Path(input_file))

        # Export data
        exporter = ElectionExporter(Path(nominations_file), Path(votes_file))
        exporter.export(elections)

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
