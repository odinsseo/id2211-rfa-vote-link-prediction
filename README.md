# Vote Feature Pipeline

A Python library for extracting temporal network features from vote data.

## Overview

This library processes vote data to extract network-based features, including:
- User voting patterns
- Network centrality metrics
- Temporal activity patterns

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from utils.vote_feature_pipeline import VoteFeaturePipeline

# Initialize pipeline
pipeline = VoteFeaturePipeline(
    votes_file="data/votes.csv",
    elections_file="data/elections.csv"
)

# Extract features
features_df = pipeline.process_votes()
```

## Features Generated

### Voter Features
- `voter_support_ratio`: Ratio of positive votes
- `voter_out_degree`: Number of votes cast
- `voter_clustering`: Local network density
- `voter_balanced_triads`: Positive path patterns
- `voter_pagerank`: Network influence score

### Dynamic Features
- `voter_window_votes`: Recent activity count
- `voter_velocity`: Vote rate
- `voter_acceleration`: Activity change

### Candidate Features
- `cand_support_ratio`: Received support ratio
- `cand_total_votes`: Total votes received
- `cand_time_since_last`: Time since last vote

## Testing

```bash
# Run tests
python -m pytest tests/test_pipeline.py
```

## Project Structure

```
.
├── data/                # Data directory
│   ├── votes.csv       # Vote data
│   └── elections.csv   # Election timestamps
│
├── utils/              # Core implementation
│   ├── vote_feature_pipeline.py  # Main pipeline
│   └── graph_metrics.py         # Network metrics
│
├── tests/             # Test suite
│   └── test_pipeline.py        # Pipeline tests
│
└── requirements.txt   # Dependencies
```

## Input Data Format

### votes.csv
```csv
voter,candidate,vote,vote_time
UserA,UserB,1,2005-01-01 12:00:00
UserC,UserB,-1,2005-01-01 13:00:00
```

### elections.csv
```csv
election_date
2005-01-15
2005-02-15
```

## License

MIT License - See LICENSE file for details
