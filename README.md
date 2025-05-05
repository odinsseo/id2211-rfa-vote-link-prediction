# Network Analysis Project

This project investigates user behavior patterns in social networks, particularly focusing on Wikipedia admin elections and user interactions.

## Project Structure

```text
├── data/                      # Data files (not included in repo)
│   ├── enwiki-*.txt          # Wikipedia user talk page dumps
│   ├── wikiElec.*.txt        # Wikipedia election data
│   ├── individual_votes.csv   # Processed election votes
│   └── user_talk.csv         # Processed user interactions
├── utils/                    # Utility modules for data processing
│   ├── wiki_admin_fetcher.py # Extract admin promotions
│   ├── wiki_common.py        # Shared Wikipedia processing utils
│   ├── wiki_elections_parser.py # Parse election data
│   ├── wiki_interactions_parser.py # Process user interactions
│   └── wiki_revision_parser.py # Parse revision history
└── preliminary_analysis.ipynb # Initial data analysis notebook

## Setup

1. Clone the repository
2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install required packages:

```bash
pip install -r requirements.txt

# Additional dependencies for analysis
pip install pandas networkx
```

## Dependencies

### Core Dependencies (requirements.txt)

- beautifulsoup4==4.13.4
- mwclient==0.11.0
- requests==2.32.3
- tqdm==4.66.1

### Additional Dependencies for Analysis

- pandas
- networkx

## Usage

### Wikipedia Data Processing

1. Process election data:

```bash
python -m utils.wiki_elections_parser input_file nominations.csv votes.csv
```

1. Extract user interactions:

```bash
python -m utils.wiki_interactions_parser input_file output.csv
```

1. Fetch admin information:

```bash
python -m utils.wiki_admin_fetcher YYYY-MM-DD admins.csv
```

### Analysis

The `preliminary_analysis_stefano.ipynb` notebook contains initial analysis of:

- Network construction from interaction data
- Graph metrics computation
- Activity patterns analysis

## Data Description

The project uses two main data sources:

1. Wikipedia admin elections and user interactions
2. Twitter Higgs boson announcement cascade data (following/follower networks and user activities)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
