# Vote Network Feature Engineering Pipeline Explained

## Overview

The feature engineering pipeline processes votes chronologically to maintain temporal causality. For each vote, we extract features based on the network state BEFORE adding that vote to the graph.

## Pipeline Steps

1. **Load Data**
   ```python
   df_votes = pd.read_csv(votes_file, parse_dates=['vote_time'])
   df_votes = df_votes.sort_values('vote_time')  # Ensure temporal order
   ```

2. **For Each Vote**:
   ```python
   for vote in chronological_votes:
       # 1. Extract features before updating graph
       features = extract_features(vote)
       
       # 2. Update graph and state
       update_state(vote)
   ```

## Feature Categories

### 1. Voter Features (Network Position)
```python
voter_features = {
    'support_ratio': positive_votes / total_votes,
    'out_degree': total_votes_cast,
    'pagerank': influence_in_network,
    'clustering_coeff': local_network_density,
    'betweenness': bridge_role_importance
}
```

### 2. Candidate Features (Vote History)
```python
candidate_features = {
    'total_votes': support_count + oppose_count,
    'support_ratio': support_count / total_votes,
    'time_since_last': seconds_since_last_vote,
    'vote_rank': position_in_sequence,
    'decayed_support': time_weighted_support
}
```

### 3. Dynamic Features (Recent Activity)
```python
dynamic_features = {
    'window_votes': votes_in_last_30_days,
    'velocity': votes_per_day,
    'acceleration': velocity_change,
    'window_support_ratio': recent_support_ratio
}
```

### 4. Temporal Context
```python
temporal_features = {
    'time_active': days_since_first_vote,
    'activity_regularity': std_time_between_votes,
    'elections_before': prior_election_count
}
```

## Key Optimizations

1. **Caching**
   - PageRank computed every N votes
   - Clustering coefficients cached
   - Network metrics reused

2. **Efficient Updates**
   - Sliding window with deque
   - Incremental state updates
   - Decay factor maintenance

3. **Memory Management**
   - Clear old cache entries
   - Prune expired window data
   - Batch processing for large graphs

## Feature Matrix Output

The final output is a CSV with rows representing votes and columns for all features:

```
timestamp | voter | candidate | support_ratio | pagerank | ... | label
----------|--------|-----------|---------------|----------|-----|-------
2005-01-01| UserA | UserB     | 0.8          | 0.02     | ... | 1
2005-01-02| UserC | UserD     | 0.6          | 0.01     | ... | 0
```

## Usage Example

```python
# Initialize pipeline
pipeline = VoteFeaturePipeline(
    votes_file="votes.csv",
    elections_file="elections.csv",
    window_days=30
)

# Extract features
features_df = pipeline.process_votes()

# Save result
features_df.to_csv("vote_features.csv")
```

## Performance Considerations

1. **Computation Scheduling**
   - PageRank: Every 5000 votes
   - Window cleanup: Every vote
   - Cache clearing: When memory high

2. **Feature Dependencies**
   - Network metrics → Need graph structure
   - Temporal features → Need time series
   - Dynamic features → Need sliding window

3. **Memory vs Speed**
   - Cache frequently used metrics
   - Clear old window data
   - Batch process when possible

The pipeline maintains temporal causality while efficiently computing a comprehensive set of features that capture network structure, temporal patterns, and voting behavior.
