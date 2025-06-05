# Network Analysis Project

This project investigates user behavior patterns in social networks, focusing on Wikipedia admin elections and user interactions. It uses machine learning to predict voting behavior in admin elections based on prior article edit interactions between users.

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
│   ├── nominations_elections.csv # Election nominations
│   ├── votes_with_election_info.csv # Processed election votes
│   ├── user_talk.csv        # User talk interactions
│   └── features/            # Generated interaction features
│       └── article_edits.*.csv # Article edit interactions per election
├── analysis/                 # Analysis notebooks
│   └── article_edit_analysis.ipynb # Analysis of article edit patterns
├── modeling/                 # Machine learning models
│   └── article_edit_modeling.ipynb # Prediction model for voting behavior
├── scripts/                  # Processing scripts
│   └── article_edit_features.py # Feature generation for article edits
├── utils/                    # Utility modules for data processing
│   ├── wiki_base_parser.py  # Base classes for dump processing
│   ├── wiki_common.py       # Shared utilities (caching, API, etc.)
│   ├── wiki_admin_fetcher.py # Extract admin promotions
│   ├── wiki_elections_parser.py # Parse election data
│   ├── wiki_revision_parser.py # Parse user talk revisions
│   ├── wiki_article_talk_parser.py # Parse article talk data
│   ├── wiki_interactions_parser.py # Process user interactions
│   └── feature_extraction/  # Feature extraction utilities
│       └── extract_user_talk_graph.py # User talk interaction features
```

## Architecture

The project follows a modular architecture with shared base classes and utilities organized into two main components:

### Data Processing Utilities (`utils/`)

Base classes and shared utilities for working with Wikipedia data:

- `DumpEntry` protocol: Defines interface for all dump entries
- `CSVWriter`: Handles CSV file output
- `DumpParser`: Base class for parsing different dump formats
- `DumpProcessor`: Base class for processing dumps with chunking and filtering
- `Cache`: Caches API results and bot lists
- `WikiAPI`: Handles Wikipedia API interactions
- `UsernameHandler`: Normalizes usernames and detects bots

### Feature Engineering (`scripts/`)

Components for generating and processing features:

- `article_edit_features.py`: Extracts interaction features from edit histories

  - Processes article edit data into interaction features per election
  - Computes temporal and behavioral metrics
  - Handles feature aggregation and normalization
  - Outputs features in a format ready for modeling

## Setup

1. Clone the repository

1. Create and activate a Python virtual environment:

```bash
# Install dependencies
pip install -r requirements.txt
```

## Dependencies

### Core Dependencies (requirements.txt)

Data Processing:

- beautifulsoup4==4.13.4 - HTML parsing for bot list extraction
- dask==2024.12.0 - Parallel computing and large dataset handling
- mwclient==0.11.0 - Wikipedia API client
- numpy==1.26.4 - Numerical computing foundation
- pandas==2.2.3 - Data manipulation and analysis
- requests==2.32.3 - HTTP requests for API access
- tqdm==4.66.1 - Progress bars for long operations

Analysis and Visualization:

- matplotlib==3.10.3 - Basic plotting library
- networkx==3.4.2 - Network/graph analysis
- pyvis==0.3.2 - Interactive network visualization
- seaborn==0.13.2 - Statistical data visualization

Machine Learning:

- scikit-learn==1.5.0 - Machine learning algorithms and tools

### Additional Development Dependencies

These are not in requirements.txt but needed for development:

- jupyter - Interactive notebook environment
- black - Code formatting
- pylint - Code linting
- pytest - Unit testing

## Data Processing Pipeline

### 1. Admin Data Collection

```bash
python -m utils.wiki_admin_fetcher 2008-01-04 data/admins.csv
```

Fetches all admin promotions before the cutoff date, with username normalization.

### 2. Election Data Processing

```bash
python -m utils.wiki_elections_parser data/wikiElec.ElecBs3.txt.gz data/nominations_elections.csv data/votes_with_election_info.csv
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

### 4. Feature Generation and Modeling

```bash
# Generate article edit features
python -m scripts.article_edit_features --time-window 30 --weight-threshold 1.0 --start-date 2005-08-31 --article-edits data/article_edits.csv --output-dir data/features
```

Features are generated for each election, capturing interaction patterns between voters and candidates in article edits. The modeling pipeline includes:

1. Feature generation from article edit data
2. Data preprocessing and normalization
3. Training a logistic regression model with L1 regularization
4. Model evaluation using ROC-AUC metric

The complete analysis and modeling process is documented in:

- `analysis/article_edit_analysis.ipynb` - Exploratory analysis of edit patterns
- `modeling/article_edit_modeling.ipynb` - Predictive modeling of voting behavior

## License

MIT License - See LICENSE file for details
