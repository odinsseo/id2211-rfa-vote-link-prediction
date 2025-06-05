#!/usr/bin/env python3
"""Core graph construction functionality with cutoff times."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphConstructor:
    """Base class for constructing temporal graphs with cutoff times."""
    
    def __init__(self, data_file: str):
        """Initialize graph constructor.
        
        Args:
            data_file: Path to data file
        """
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        self.votes = self._load_votes()
        
    def _load_votes(self) -> List[dict]:
        """Load and preprocess votes."""
        votes = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.strptime(row['vote_time'], 
                                               '%Y-%m-%d %H:%M:%S')
                    if timestamp.year > 2025:  # Skip invalid dates
                        continue
                    votes.append({
                        'voter': row['voter'],
                        'candidate': row['candidate'],
                        'vote': int(row['vote']),
                        'timestamp': timestamp
                    })
                except ValueError:
                    continue
                    
        return sorted(votes, key=lambda x: x['timestamp'])
    
    def build_vote_graph(self, 
                        cutoff_time: Optional[datetime] = None,
                        start_time: Optional[datetime] = None) -> nx.DiGraph:
        """Build vote graph up to cutoff time.
        
        Args:
            cutoff_time: Only include votes up to this time
            start_time: Only include votes after this time
            
        Returns:
            Directed graph Gᵥ(t) with signed edges
        """
        G = nx.DiGraph()
        
        # Filter votes by time range
        filtered_votes = self.votes
        
        if start_time:
            filtered_votes = [v for v in filtered_votes 
                            if v['timestamp'] >= start_time]
        
        if cutoff_time:
            filtered_votes = [v for v in filtered_votes 
                            if v['timestamp'] <= cutoff_time]
        
        # Add filtered votes to graph
        for vote in filtered_votes:
            if vote['vote'] != 0:  # Skip neutral votes
                G.add_edge(vote['voter'],
                          vote['candidate'],
                          sign=vote['vote'],
                          timestamp=vote['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
        
        logger.info(f"Built vote graph with {G.number_of_nodes()} nodes "
                   f"and {G.number_of_edges()} edges"
                   f"{f' from {start_time} to {cutoff_time}' if start_time and cutoff_time else ''}")
        
        return G
    
    def get_graph_sequence(self, 
                          start_time: datetime,
                          end_time: datetime,
                          num_steps: int) -> List[Tuple[datetime, nx.DiGraph]]:
        """Get sequence of graph snapshots between times.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            num_steps: Number of snapshots to generate
            
        Returns:
            List of (timestamp, graph) pairs
        """
        from datetime import timedelta
        
        time_delta = (end_time - start_time) / num_steps
        sequence = []
        
        for i in range(num_steps + 1):
            timestamp = start_time + (time_delta * i)
            G = self.build_vote_graph(cutoff_time=timestamp)
            sequence.append((timestamp, G))
            
        return sequence
