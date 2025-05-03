import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from wiki_interactions_parser import Cache, WikiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

USERNAME_CHANGES = {
    k.lower(): v.lower() for k, v in WikiAPI(Cache()).username_changes.items()
}


@dataclass
class Vote:
    """Data class for a vote record"""

    voter: str
    value: int
    timestamp: str


@dataclass
class Election:
    """Data class for an election record"""

    outcome: Optional[int] = None
    close_time: Optional[str] = None
    nominee: Optional[str] = None
    nominator: Optional[str] = None
    votes: List[Vote] = None

    def __post_init__(self):
        if self.votes is None:
            self.votes = []

    @property
    def is_valid(self) -> bool:
        """Check if the election has all required fields"""
        return all(
            [
                self.outcome is not None,
                self.close_time is not None,
                self.nominee is not None,
                self.nominator is not None,
            ]
        )

    def add_vote(self, vote: Vote):
        """Add a vote to this election"""
        self.votes.append(vote)


class ElectionParser:
    """Parser for Wikipedia election data"""

    def __init__(self, input_path: str):
        """
        Initialize the parser with the input file path

        Args:
            input_path: Path to the input file
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    def parse_elections(self) -> List[Election]:
        """
        Parse elections from the input file

        Returns:
            List of Election objects
        """
        logger.info(f"Parsing elections from {self.input_path}")

        try:
            with open(
                self.input_path, "r", encoding="utf-8", errors="surrogateescape"
            ) as file:
                content = file.read()

            # Remove comments
            content = "\n".join(
                line for line in content.split("\n") if not line.startswith("#")
            )

            # Split by empty lines to get election entries
            election_entries = re.split(r"\n\s*\n", content)

            # Parse each election entry
            elections = []
            for entry in election_entries:
                if not entry.strip():
                    continue

                try:
                    election = self._parse_election_entry(entry)
                    if election.is_valid:
                        elections.append(election)
                except Exception as e:
                    logger.warning(f"Error parsing election entry: {e}")

            logger.info(f"Successfully parsed {len(elections)} elections")
            return elections

        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def _parse_election_entry(self, entry: str) -> Election:
        """
        Parse a single election entry

        Args:
            entry: String containing an election entry

        Returns:
            Election object
        """
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
                election.nominee = parts[2].lower()
            elif code == "N" and len(parts) >= 3:
                election.nominator = parts[2].lower()
            elif code == "V" and len(parts) >= 5:
                voter = parts[4].lower()
                voter = USERNAME_CHANGES.get(voter, voter)
                vote = Vote(voter=voter, value=int(parts[1]), timestamp=parts[3])
                election.add_vote(vote)

        return election


class CSVExporter:
    """Exports election data to CSV files"""

    def __init__(self, nominations_path: str, votes_path: str):
        """
        Initialize the exporter with output file paths

        Args:
            nominations_path: Path for the nominations CSV file
            votes_path: Path for the votes CSV file
        """
        self.nominations_path = Path(nominations_path)
        self.votes_path = Path(votes_path)

    def export(self, elections: List[Election]) -> Tuple[int, int]:
        """
        Export elections to CSV files

        Args:
            elections: List of Election objects

        Returns:
            Tuple of (nomination_count, vote_count)
        """
        nomination_count = 0
        vote_count = 0

        # Export nominations
        with open(
            self.nominations_path,
            "w",
            newline="",
            encoding="utf-8",
            errors="surrogateescape",
        ) as f:
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
                nomination_count += 1

        # Export votes
        with open(
            self.votes_path, "w", newline="", encoding="utf-8", errors="surrogateescape"
        ) as f:
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

        logger.info(
            f"Exported {nomination_count} nominations to {self.nominations_path}"
        )
        logger.info(f"Exported {vote_count} votes to {self.votes_path}")

        return nomination_count, vote_count


def parse_wiki_elections(
    input_file: str, nominations_file: str, votes_file: str
) -> bool:
    """
    Parse Wikipedia election data and export to CSV files

    Args:
        input_file: Path to the input file
        nominations_file: Path for the nominations CSV
        votes_file: Path for the votes CSV

    Returns:
        True if successful, False otherwise
    """
    try:
        parser = ElectionParser(input_file)
        elections = parser.parse_elections()

        exporter = CSVExporter(nominations_file, votes_file)
        exporter.export(elections)

        return True
    except Exception as e:
        logger.error(f"Error processing elections: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build Wikipedia admin voting interaction graph from admin election dump"
    )
    parser.add_argument(
        "input_file",
        help="Path to the Wikipedia Admin Election dump file",
    )
    parser.add_argument(
        "nominations_file",
        help="Path to save the output CSV file containing nominations and election results",
    )
    parser.add_argument(
        "votes_file",
        help="Path to save the output CSV file containing individual votes",
    )
    parser.add_argument(
        "--log-level",
        help="Set the logging level (default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    try:
        success = parse_wiki_elections(
            args.input_file, args.nominations_file, args.votes_file
        )

        if success:
            logger.info("Processing completed successfully.")
            print(f"Successfully created:")
            print(f"  - {args.nominations_file} (nominations and election results)")
            print(f"  - {args.votes_file} (individual votes)")
        else:
            logger.error("Processing failed.")
            print("Processing failed. Check the logs for details.")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
