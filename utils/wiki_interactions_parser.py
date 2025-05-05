#!/usr/bin/env python3
"""Main entry point for Wikipedia interactions parser."""

import argparse
import logging

from wiki_common import Cache, UsernameHandler, WikiAPI
from wiki_revision_parser import CSVWriter, DumpProcessor, RevisionParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_graph(
    input_file: str,
    output_file: str,
    cache_dir: str = "./cache",
    chunk_size: int = 10000,
) -> None:
    """Build interaction graph from Wikipedia dump.

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
        parser = RevisionParser(username_handler)
        writer = CSVWriter(output_file)
        processor = DumpProcessor(parser, writer)

        # Process the dump file
        processor.process_file(input_file, chunk_size)
        logger.info(f"Graph data saved to {output_file}")

    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise


def main():
    """Main entry point."""
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

    # Build the graph
    build_graph(args.input_file, args.output_file, args.cache_dir, args.chunk_size)


if __name__ == "__main__":
    main()
