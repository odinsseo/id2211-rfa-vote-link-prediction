# Network Analysis Project

This project investigates user behavior patterns in social networks, focusing on Wikipedia admin elections and user interactions. It uses machine learning to predict voting behavior in admin elections based on prior article edit interactions and user talk page communications between users.

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
│   ├── article_edits.csv     # Article edit interactions
│   ├── user_talk.csv        # User talk interactions
│   └── features/            # Generated interaction features
│       └── article_edits.*.csv # Article edit interactions per election
├── analysis/                 # Analysis notebooks
│   ├── article_edit_analysis.ipynb # Analysis of article edit patterns
├── modeling/                 # Machine learning models and visualization
│   ├── modeling.ipynb       # Core modeling and feature evaluation
│   └── imgs/                # Generated plots and visualizations
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
│   ├── vote_graph.py        # Vote network construction and analysis
│   └── feature_extraction/  # Feature extraction utilities
├── vote-graph/              # Vote graph analysis components
│   ├── vote_feature_pipeline_modelling.py # Vote network feature pipeline
│   ├── feature_extraction/  # Vote graph feature generation
│   └── graph_construction/  # Graph building utilities
└── tests/                   # Test data and unit tests
```

## Architecture

The project follows a modular architecture with shared base classes and utilities organized into three main components:

### Data Processing Utilities (`utils/`)

Base classes and shared utilities for working with Wikipedia data:

- `DumpEntry` protocol: Defines interface for all dump entries
- `CSVWriter`: Handles CSV file output
- `DumpParser`: Base class for parsing different dump formats
- `DumpProcessor`: Base class for processing dumps with chunking and filtering
- `Cache`: Caches API results and bot lists
- `WikiAPI`: Handles Wikipedia API interactions
- `UsernameHandler`: Normalizes usernames and detects bots

### Feature Engineering (`scripts/`, `vote-graph/`)

Components for generating and processing features:

1. Article Edit Features (`scripts/article_edit_features.py`):
   - Processes article edit data into interaction features per election
   - Computes temporal and behavioral metrics
   - Handles feature aggregation and normalization
   - Outputs features in a format ready for modeling

2. Vote Network Features (`vote-graph/`):
   - Constructs and analyzes vote networks from election data
   - Extracts network-based features (centrality, influence)
   - Computes temporal voting patterns and user relationships
   - Maintains temporal causality in feature generation

### Analysis and Modeling (`analysis/`, `modeling/`)

Components for analyzing data and building predictive models:

- Article edit pattern analysis
- Vote network visualization and analysis
- Feature importance evaluation
- Model training and evaluation (Logistic Regression, Random Forest, XGBoost)
- Performance metrics and comparison across model types

## Setup

1. Clone the repository

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
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

Fetches all admin promotions before the cutoff date, with username normalization and bot filtering.

### 2. Election Data Processing

```bash
python -m utils.wiki_elections_parser data/wikiElec.ElecBs3.txt.gz data/nominations_elections.csv data/votes_with_election_info.csv
```

Processes election data into two main outputs:

1. Nominations containing:

- Nominator
- Nominee
- Timestamp
- Outcome

2. Votes containing:

- Voter
- Candidate
- Vote value
- Timestamps
- Election metadata

### 3. User Interactions Processing

```bash
# Process user talk interactions
python -m utils.wiki_interactions_parser data/enwiki-20080103.user_talk.bz2 data/user_talk.csv

# Process article talk interactions
python -m utils.wiki_article_talk_parser data/enwiki-20080103.talk.bz2 data/article_talk.csv
```

All processors implement the following features:

- Compressed dump reading (bz2, gz)
- Bot filtering and username normalization
- Chunk-based processing for memory efficiency
- Progress tracking and logging

### 4. Feature Generation

The project uses two main feature generation pipelines:

#### 4.1 Article Edit Features

```bash
python -m scripts.article_edit_features --time-window 30 --weight-threshold 1.0 --start-date 2005-08-31 --article-edits data/article_edits.csv --output-dir data/features
```

Generates election-specific features capturing:

- Article edit interactions between voters and candidates
- Temporal editing patterns
- Collaboration metrics
- User activity statistics

#### 4.2 Vote Network Features

```bash
python -m vote-graph.vote_feature_pipeline_modelling --input data/votes_with_election_info.csv --output data/features/vote_features.csv
```

Extracts network-based features including:

- Network centrality metrics
- Temporal voting patterns
- User influence measures
- Vote history analysis

See `FEATURE_PIPELINE_EXPLAINED.md` for detailed documentation of the feature engineering process.

## Modeling and Analysis

The project implements several machine learning approaches to predict voting behavior:

### Feature Sets

The analysis incorporates multiple feature types:

#### Article Edit Features

- User collaboration patterns
- Edit timing and frequency
- Article overlap metrics

#### Vote Network Features

- User influence metrics
- Historical voting patterns
- Network position features

#### Combined Features

- Integration of both feature sets
- Feature importance analysis
- Cross-validation evaluation

### Models Implemented

The project evaluates three main model types:

#### Logistic Regression

- L1 and L2 regularization
- Feature selection capability
- Interpretable coefficients

#### Random Forest

- Non-linear pattern capture
- Feature importance ranking
- Robust to outliers

#### XGBoost

- Gradient boosting approach
- Advanced feature interactions
- High predictive performance

### Model Evaluation

The complete analysis and modeling process is documented in:

- `modeling/modeling.ipynb` - Core modeling pipeline and evaluation
- `analysis/article_edit_analysis.ipynb` - Analysis of edit patterns

Key metrics tracked:

- ROC-AUC scores
- Feature importance rankings
- Model comparison statistics
- Temporal performance analysis

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated documentation
   - Added tests if needed

## License

MIT License - See LICENSE file for details
