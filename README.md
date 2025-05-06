# Network Analysis Project

This project investigates user behavior patterns in social networks, focusing on Wikipedia admin elections and user interactions.

## Project Structure

```text
├── data/                      # Data files (not included in repo)
│   ├── enwiki-*.bz2          # Wikipedia dump files
│   │   ├── user_talk.bz2     # User talk page interactions
│   │   └── talk.bz2          # Article talk page interactions
│   ├── wikiElec.*.txt.gz     # Wikipedia election data
│   ├── admins.csv            # Processed admin promotions
│   ├── article_talk.csv      # Article talk page interactions
│   ├── individual_votes.csv  # Election votes
│   ├── nominations.csv       # Election nominations
│   └── user_talk.csv        # User talk interactions
├── utils/                    # Utility modules for data processing
│   ├── wiki_base_parser.py  # Base classes for dump processing
│   ├── wiki_common.py       # Shared utilities (caching, API, etc.)
│   ├── wiki_admin_fetcher.py # Extract admin promotions
│   ├── wiki_elections_parser.py # Parse election data
│   ├── wiki_revision_parser.py # Parse user talk revisions
│   ├── wiki_article_talk_parser.py # Parse article talk data
│   └── wiki_interactions_parser.py # Process user interactions
└── preliminary_analysis.ipynb # Data analysis notebook

```

## Architecture

The project follows a modular architecture with shared base classes and utilities:

- `DumpEntry` protocol: Defines interface for all dump entries
- `CSVWriter`: Handles CSV file output
- `DumpParser`: Base class for parsing different dump formats
- `DumpProcessor`: Base class for processing dumps with chunking and filtering
- `Cache`: Caches API results and bot lists
- `WikiAPI`: Handles Wikipedia API interactions
- `UsernameHandler`: Normalizes usernames and detects bots

## Setup

1. Clone the repository
1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install required packages:

```bash
pip install -r requirements.txt

# Additional dependencies for analysis
pip install pandas networkx jupyter
```

## Dependencies

### Core Dependencies (requirements.txt)

- beautifulsoup4==4.13.4 - HTML parsing for bot list extraction
- mwclient==0.11.0 - Wikipedia API client
- requests==2.32.3 - HTTP requests for API access
- tqdm==4.66.1 - Progress bars for long operations

### Analysis Dependencies

- pandas - Data manipulation and analysis
- networkx - Network/graph analysis
- jupyter - Interactive analysis environment

## Data Processing Pipeline

### 1. Admin Data Collection

```bash
python -m utils.wiki_admin_fetcher 2008-01-04 data/admins.csv
```

Fetches all admin promotions before the cutoff date, with username normalization.

### 2. Election Data Processing

```bash
python -m utils.wiki_elections_parser data/wikiElec.ElecBs3.txt.gz data/nominations.csv data/votes.csv
```

Processes election data into:

- Nominations: nominator, nominee, timestamp, outcome
- Votes: voter, candidate, vote value, timestamps

### 3. User Interactions Processing

```bash
# Process user talk interactions
python -m utils.wiki_interactions_parser data/enwiki-20080103.user_talk.bz2 data/user_talk.csv

# Process article talk interactions
python -m utils.wiki_article_talk_parser data/enwiki-20080103.talk.bz2 data/article_talk.csv
```

All processors handle:

- Compressed dump reading (bz2, gz)
- Bot filtering
- Username normalization
- Chunk-based processing for memory efficiency
- Progress tracking and logging

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
