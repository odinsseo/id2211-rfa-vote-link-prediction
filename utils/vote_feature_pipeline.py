"""Vote feature engineering pipeline with comprehensive metrics."""
import pandas as pd
import networkx as nx
import numpy as np
import math
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkFeatureExtractor:
    """Extract features from temporal vote network."""
    
    def __init__(self, window_days: int = 30, pagerank_batch: int = 5000, decay_halflife: float = 30.0):
        self.window_days = window_days
        self.pagerank_batch = pagerank_batch
        self.decay_halflife = decay_halflife
        self.decay_factor = math.log(2) / decay_halflife
        
        # Network state
        self.G = nx.DiGraph()
        self.elections_count = defaultdict(int)
        
        # Caches
        self.pagerank_cache = {}
        self.last_pagerank_update = 0
        self.clustering_cache = {}
        
        # Vote tracking
        self.vote_counts = defaultdict(int)
        self.positive_votes = defaultdict(int)
        
        # Candidate tracking
        self.cand_support = defaultdict(int)
        self.cand_oppose = defaultdict(int)
        self.cand_last_vote = {}
        self.decayed_support = defaultdict(float)
        self.decayed_oppose = defaultdict(float)
        self.last_decay_update = {}
        
        # Window tracking
        self.window_queue = deque()
        self.window_votes = defaultdict(int)
        self.window_support = defaultdict(int)
        self.prev_velocity = defaultdict(float)
        
    def update_decay(self, candidate: str, timestamp: datetime):
        """Update decayed vote counts."""
        if candidate in self.last_decay_update:
            delta = (timestamp - self.last_decay_update[candidate]).total_seconds() / (24 * 3600)  # days
            decay = math.exp(-self.decay_factor * delta)
            self.decayed_support[candidate] *= decay
            self.decayed_oppose[candidate] *= decay
        self.last_decay_update[candidate] = timestamp
        
    def update_graph(self, voter: str, candidate: str, sign: int, timestamp: datetime):
        """Add vote to graph with timestamp."""
        self.G.add_edge(voter, candidate, sign=sign, timestamp=timestamp)
        
        # Update voter stats
        self.vote_counts[voter] += 1
        if sign > 0:
            self.positive_votes[voter] += 1
            
        # Update candidate stats
        self.update_decay(candidate, timestamp)
        if sign > 0:
            self.cand_support[candidate] += 1
            self.decayed_support[candidate] += 1
        else:
            self.cand_oppose[candidate] += 1
            self.decayed_oppose[candidate] += 1
        self.cand_last_vote[candidate] = timestamp
        
        # Update window
        self.window_queue.append((timestamp, voter, sign))
        self.window_votes[voter] += 1
        self.window_support[voter] += (1 if sign > 0 else 0)
        
        # Evict old window entries
        cutoff = timestamp - timedelta(days=self.window_days)
        while (self.window_queue and 
               self.window_queue[0][0] < cutoff):
            old_t, old_u, old_sign = self.window_queue.popleft()
            self.window_votes[old_u] -= 1
            self.window_support[old_u] -= (1 if old_sign > 0 else 0)
            
    def compute_voter_features(self, voter: str, timestamp: datetime, idx: int, elections_before: int) -> dict:
        """Extract voter-centric features."""
        # Update PageRank periodically
        self.update_pagerank(idx)
        
        # Basic metrics
        out_deg = self.vote_counts[voter]
        if out_deg == 0:
            return {
                'support_ratio': 0.0,
                'out_degree': 0,
                'scaled_out_degree': 0.0,
                'clustering_coeff': 0.0,
                'balanced_triad_count': 0,
                'pagerank': 0.0,
                'window_out_degree': 0,
                'window_support_ratio': 0.0,
                'velocity': 0.0,
                'acceleration': 0.0
            }
            
        # Static metrics
        support_ratio = self.positive_votes[voter] / out_deg
        scaled_degree = out_deg / max(1, elections_before)
        clustering = self.compute_clustering(voter) if out_deg > 1 else 0.0
        balanced_triads = self.compute_balanced_triads(voter)
        pagerank = self.pagerank_cache.get(voter, 0.0)
        
        # Dynamic metrics
        window_votes = self.window_votes[voter]
        window_ratio = (self.window_support[voter] / window_votes 
                       if window_votes > 0 else 0.0)
        velocity = window_votes / self.window_days
        acceleration = velocity - self.prev_velocity[voter]
        self.prev_velocity[voter] = velocity
        
        return {
            'support_ratio': support_ratio,
            'out_degree': out_deg,
            'scaled_out_degree': scaled_degree,
            'clustering_coeff': clustering,
            'balanced_triad_count': balanced_triads,
            'pagerank': pagerank,
            'window_out_degree': window_votes,
            'window_support_ratio': window_ratio,
            'velocity': velocity,
            'acceleration': acceleration
        }
        
    def compute_candidate_features(self, candidate: str, timestamp: datetime) -> dict:
        """Extract candidate-centric features."""
        # Update decay
        self.update_decay(candidate, timestamp)
        
        support = self.cand_support[candidate]
        oppose = self.cand_oppose[candidate]
        total = support + oppose
        
        if total == 0:
            return {
                'cand_support_count': 0,
                'cand_oppose_count': 0,
                'cand_total_votes': 0,
                'cand_support_ratio': 0.0,
                'vote_rank_on_candidate': 0,
                'time_since_last_vote_on_candidate': None,
                'decayed_support': 0.0,
                'decayed_oppose': 0.0,
                'decay_support_ratio': 0.0
            }
            
        # Basic stats
        support_ratio = support / total
        
        # Temporal features
        time_since_last = None
        if candidate in self.cand_last_vote:
            delta = timestamp - self.cand_last_vote[candidate]
            time_since_last = delta.total_seconds()
            
        # Decay features
        decayed_support = self.decayed_support[candidate]
        decayed_oppose = self.decayed_oppose[candidate]
        decayed_total = decayed_support + decayed_oppose
        decay_support_ratio = (decayed_support / decayed_total 
                             if decayed_total > 0 else 0.0)
            
        return {
            'cand_support_count': support,
            'cand_oppose_count': oppose,
            'cand_total_votes': total,
            'cand_support_ratio': support_ratio,
            'vote_rank_on_candidate': total,
            'time_since_last_vote_on_candidate': time_since_last,
            'decayed_support': decayed_support,
            'decayed_oppose': decayed_oppose,
            'decay_support_ratio': decay_support_ratio
        }
        
    def compute_clustering(self, node: str) -> float:
        """Get cached clustering coefficient."""
        if node not in self.clustering_cache:
            try:
                G_undir = self.G.to_undirected()
                self.clustering_cache[node] = nx.clustering(G_undir, node)
            except:
                self.clustering_cache[node] = 0.0
        return self.clustering_cache[node]
        
    def compute_balanced_triads(self, node: str) -> int:
        """Count balanced triads (positive 2-step paths)."""
        try:
            # Get positive neighbors
            pos_neighbors = {v for _, v, d in self.G.out_edges(node, data=True)
                           if d.get('sign', 0) > 0}
            
            # Count their positive out-edges
            triad_count = 0
            for neighbor in pos_neighbors:
                triad_count += sum(1 for _, _, d in self.G.out_edges(neighbor, data=True)
                                 if d.get('sign', 0) > 0)
            return triad_count
        except:
            return 0
            
    def update_pagerank(self, current_idx: int):
        """Recompute PageRank if needed."""
        if current_idx - self.last_pagerank_update >= self.pagerank_batch:
            try:
                self.pagerank_cache = nx.pagerank(self.G, weight='sign', alpha=0.85)
                self.last_pagerank_update = current_idx
                self.clustering_cache = {}  # Clear clustering cache
            except:
                pass

class VoteFeaturePipeline:
    """Complete feature extraction pipeline."""
    
    def __init__(self, 
                 votes_file: str,
                 elections_file: str,
                 window_days: int = 30):
        self.votes_file = Path(votes_file)
        self.elections_file = Path(elections_file)
        self.window_days = window_days
        
        # Initialize feature extractor
        self.extractor = NetworkFeatureExtractor(window_days)
        
    def load_elections(self) -> pd.DataFrame:
        """Load and process election dates."""
        logger.info("Loading election data...")
        df_elections = pd.read_csv(self.elections_file, sep=';', parse_dates=['election_start'])
        df_elections = df_elections.sort_values('election_start')
        
        # Count elections before each date
        for date in df_elections['election_start']:
            self.extractor.elections_count[date.date()] += 1
            
        return df_elections
            
    def process_votes(self) -> pd.DataFrame:
        """Process votes and extract features."""
        # Load data
        logger.info("Loading votes...")
        df_votes = pd.read_csv(self.votes_file, parse_dates=['vote_time'])
        df_votes = df_votes.sort_values('vote_time')
        
        # Load elections
        df_elections = self.load_elections()
        
        features = []
        total_votes = len(df_votes)
        
        # Process each vote chronologically
        logger.info("Extracting features...")
        with tqdm(total=total_votes, desc="Processing votes") as pbar:
            for idx, row in df_votes.iterrows():
                voter = row['voter']
                candidate = row['candidate']
                sign = int(row['vote'])
                timestamp = row['vote_time']
                
                # Count elections before this vote
                elections_before = sum(1 for date in df_elections['election_start']
                                    if date.date() < timestamp.date())
                
                # Extract features before updating graph
                voter_features = self.extractor.compute_voter_features(
                    voter, timestamp, idx, elections_before
                )
                candidate_features = self.extractor.compute_candidate_features(
                    candidate, timestamp
                )
                
                # Record features
                features.append({
                    'timestamp': timestamp,
                    'voter': voter,
                    'candidate': candidate,
                    'elections_before': elections_before,
                    **voter_features,
                    **candidate_features,
                    'label': 1 if sign > 0 else 0
                })
                
                # Update state
                self.extractor.update_graph(voter, candidate, sign, timestamp)
                
                # Update progress
                pbar.update(1)
                if (idx + 1) % 1000 == 0:
                    pbar.set_description(f"Processed {idx + 1}/{total_votes} votes")
                    
        return pd.DataFrame(features)

def main():
    """Run feature extraction pipeline."""
    pipeline = VoteFeaturePipeline(
        votes_file="data/votes.csv",
        elections_file="data/elections.csv"
    )
    
    logger.info("Starting feature extraction...")
    df_features = pipeline.process_votes()
    
    output_file = Path("data") / "features.csv"
    logger.info(f"Saving {len(df_features)} feature vectors...")
    df_features.to_csv(output_file, index=False)
    
    # Print summary statistics
    logger.info("\nFeature Statistics:")
    logger.info(f"Total samples: {len(df_features)}")
    logger.info(f"Support ratio: {df_features['label'].mean():.2%}")
    logger.info(f"Unique voters: {df_features['voter'].nunique()}")
    logger.info(f"Unique candidates: {df_features['candidate'].nunique()}")
    logger.info(f"Average clustering: {df_features['clustering_coeff'].mean():.3f}")
    logger.info(f"Average window activity: {df_features['window_out_degree'].mean():.1f}")
    logger.info(f"Average velocity: {df_features['velocity'].mean():.2f}")

if __name__ == "__main__":
    main()
