#!/usr/bin/env python3
"""Main entry point for Wikipedia article talk page interactions parser."""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List

from wiki_base_parser import CSVWriter, DumpParser, DumpProcessor
from wiki_common import Cache, UsernameHandler, WikiAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ArticleTalkRevision:
    """Data class representing a single article talk page revision."""

    user: str = ""
    namespace: str = ""
    timestamp: str = ""
    minor: int = 0
    textdata: int = 0

    def to_row(self) -> List:
        """Convert the entry to a CSV row."""
        return [
            self.user.lower(),
            self.namespace.lower(),
            self.timestamp,
            self.minor,
            self.textdata,
        ]

    def is_valid(self) -> bool:
        """Check if the revision has valid user and namespace."""
        return bool(self.user and self.namespace)


class ArticleTalkParser(DumpParser):
    """Parser for article talk page revisions."""

    def __init__(self, username_handler: UsernameHandler):
        """Initialize with username handler."""
        self.username_handler = username_handler

    def parse_line(self, line: str) -> Dict:
        """Parse a REVISION line.

        Args:
            line: A single line from the dump file

        Returns:
            Dictionary with parsed data
        """
        fields = line.split(" ")
        if len(fields) < 6:
            return {}

        # Extract article title
        article_title = fields[3]
        namespace = article_title

        # Handle "Talk:" prefix and extract actual article namespace
        if article_title.startswith("Talk:"):
            namespace = article_title[5:]  # Remove "Talk:" prefix

        # Extract timestamp and username
        timestamp = fields[4]
        username = " ".join(fields[5:-1]) if len(fields) > 6 else fields[5]

        return {
            "namespace": namespace,
            "timestamp": timestamp,
            "username": username,
        }

    def parse_entry(self, lines: List[str]) -> ArticleTalkRevision:
        """Parse a complete article talk revision entry.

        Args:
            lines: List of lines making up a revision entry

        Returns:
            ArticleTalkRevision object with parsed data
        """
        entry = ArticleTalkRevision()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("REVISION"):
                if revision_data := self.parse_line(line):
                    entry.user = revision_data["username"]
                    entry.namespace = revision_data["namespace"]
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

        # Normalize username
        entry.user = self.username_handler.normalize(entry.user)

        return entry


class ArticleTalkProcessor(DumpProcessor):
    """Process Wikipedia dumps and extract article talk revision data."""

    def __init__(self, parser: ArticleTalkParser, writer: CSVWriter):
        """Initialize with parser and writer instances."""
        super().__init__(parser, writer)
        self.article_parser = parser

    def should_process_entry(self, entry: ArticleTalkRevision) -> bool:
        """Check if an article talk revision should be processed.

        Filters out bot-generated entries.

        Args:
            entry: The revision entry to check

        Returns:
            True if the user is not a bot
        """
        return not self.article_parser.username_handler.is_bot(entry.user)


def extract_article_talk_data(
    input_file: str,
    output_file: str,
    cache_dir: str = "./cache",
    chunk_size: int = 10000,
) -> None:
    """Extract article talk interactions from Wikipedia dump.

    Args:
        input_file: Path to input dump file
        output_file: Path to output CSV file
        cache_dir: Directory for caching
        chunk_size: Number of entries to process in each chunk
    """
    try:
        # Initialize components with dependency injection
        cache = Cache(cache_dir)
        wiki_api = WikiAPI(cache)
        username_handler = UsernameHandler(wiki_api)

        # Create parser and writer with headers
        parser = ArticleTalkParser(username_handler)
        writer = CSVWriter(
            output_file, headers=["user", "namespace", "timestamp", "minor", "textdata"]
        )
        processor = ArticleTalkProcessor(parser, writer)

        # Process the dump file
        processor.process_file(input_file, chunk_size)
        logger.info(f"Article talk data saved to {output_file}")

    except Exception as e:
        logger.error(f"Error extracting article talk data: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract article talk page interactions from Wikipedia Talk pages dump"
    )
    parser.add_argument("input_file", help="Path to the Wikipedia Talk pages dump file")
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

    # Extract the data
    extract_article_talk_data(
        args.input_file, args.output_file, args.cache_dir, args.chunk_size
    )


if __name__ == "__main__":
    main()
