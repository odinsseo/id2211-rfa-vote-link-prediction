#!/usr/bin/env python3
"""Wikipedia Admin Extractor - Retrieves all admins promoted before a specified date."""

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from time import mktime
from typing import Dict, List

from wiki_common import Cache, UsernameHandler, WikiAPI
from wiki_elections_parser import ElectionParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AdminPromotion:
    """Represents a Wikipedia admin promotion event."""

    username: str
    timestamp: datetime

    @property
    def date_string(self) -> str:
        """Returns the promotion date in YYYY-MM-DD format."""
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def timestamp_string(self) -> str:
        """Returns the promotion timestamp in ISO format."""
        return self.timestamp.isoformat()


class AdminExtractor:
    """Extracts admin information from Wikipedia log entries."""

    def __init__(self, cutoff_date: datetime, wiki_api: WikiAPI):
        """Initialize the AdminExtractor.

        Args:
            cutoff_date: Only include promotions before this date.
            wiki_api: WikiAPI instance for data fetching.
        """
        self.cutoff_date = cutoff_date
        self.wiki_api = wiki_api

    def is_admin_promotion(self, log_event: dict) -> bool:
        """Determine if a log event represents an admin promotion.

        Args:
            log_event: A single log entry from Wikipedia.

        Returns:
            True if the event is an admin promotion, False otherwise.
        """
        params = log_event.get("params", {})
        old_groups = params.get("oldgroups", [])
        new_groups = params.get("newgroups", [])

        # Check if 'sysop' was added and wasn't already present
        return "sysop" in new_groups and "sysop" not in old_groups

    def extract_promotions(self, log_entries: List[dict]) -> Dict[str, AdminPromotion]:
        """Extract admin promotions from log entries.

        Args:
            log_entries: List of log entries from Wikipedia.

        Returns:
            Dictionary mapping usernames to their earliest admin promotion.
        """
        admins = {}

        for event in log_entries:
            if not self.is_admin_promotion(event):
                continue

            username = event["title"].lower().replace(" ", "_")
            timestamp = datetime.fromtimestamp(mktime(event["timestamp"]))

            # Only include entries before the cutoff date
            if timestamp > self.cutoff_date:
                continue

            # Keep the earliest promotion date for each admin
            if username not in admins or timestamp < admins[username].timestamp:
                admins[username] = AdminPromotion(
                    username=username, timestamp=timestamp
                )

        return admins


class CSVExporter:
    """Exports data to CSV files."""

    def __init__(self, wiki_api: WikiAPI):
        """Initialize with WikiAPI instance."""
        self.username_handler = UsernameHandler(wiki_api)

    def export_admin_data(
        self, admins: Dict[str, AdminPromotion], filename: str
    ) -> None:
        """Export admin data to a CSV file.

        Args:
            admins: Dict mapping usernames to AdminPromotion objects.
            filename: Output CSV filename.
        """
        # Sort admins by promotion date
        sorted_admins = sorted(admins.values(), key=lambda admin: admin.timestamp)

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["username", "promotion_date", "promotion_timestamp"])

            for admin in sorted_admins:
                normalized_username = self.username_handler.normalize(admin.username)
                writer.writerow(
                    [normalized_username, admin.date_string, admin.timestamp_string]
                )


def parse_wiki_elections(
    input_file: str, nominations_file: str, votes_file: str
) -> bool:
    """Parse Wikipedia election data and export to CSV files.

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

        exporter = CSVExporter()
        exporter.export(elections)

        return True
    except Exception as e:
        logger.error(f"Error processing elections: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Wikipedia admin voting interaction graph from admin election dump"
    )
    parser.add_argument(
        "cutoff_date",
        help="Only include admins promoted before this date (YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
    )
    parser.add_argument(
        "output_file",
        help="Path to save the output CSV file containing admin promotions",
    )
    parser.add_argument(
        "--cache-dir", help="Directory to store cache files", default="./cache"
    )
    parser.add_argument(
        "--log-level",
        help="Set the logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    args = parser.parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Initialize components
        cache = Cache(args.cache_dir)
        wiki_api = WikiAPI(cache)
        wiki_api.connect()

        # Get log entries
        logger.info("Retrieving user rights log entries...")
        log_entries = wiki_api.site.logevents(
            "rights",
            dir="newer",
            end=args.cutoff_date.isoformat(),
            api_chunk_size=500,
        )

        # Extract admin promotions
        logger.info("Extracting admin promotions...")
        extractor = AdminExtractor(args.cutoff_date, wiki_api)
        admins = extractor.extract_promotions(log_entries)

        # Export to CSV
        logger.info("Exporting data to CSV...")
        exporter = CSVExporter(wiki_api)
        exporter.export_admin_data(admins, args.output_file)

        logger.info(f"Data exported to {args.output_file}")
        logger.info(f"Total admins found before {args.cutoff_date}: {len(admins)}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
