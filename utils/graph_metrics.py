"""Additional graph metrics for feature engineering."""

import networkx as nx
import numpy as np
from typing import Dict, Set, Tuple

def compute_centrality_metrics(G: nx.DiGraph, node: str) -> Dict[str, float]:
    """Compute various centrality metrics for a node."""
    try:
        # Degree centralities
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        total_degree = in_degree + out_degree
        
        # Local clustering
        clustering = nx.clustering(G.to_undirected(), node) if total_degree > 1 else 0.0
        
        # PageRank and eigenvector centrality
        pagerank = nx.pagerank(G, alpha=0.85).get(node, 0.0)
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G).get(node, 0.0)
        except:
            eigenvector = 0.0
            
        # Betweenness centrality (local approximation)
        try:
            between = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes())).get(node, 0.0)
        except:
            between = 0.0
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': total_degree,
            'clustering_coeff': clustering,
            'pagerank': pagerank,
            'eigenvector_centrality': eigenvector,
            'betweenness_centrality': between
        }
    except:
        return {
            'in_degree': 0,
            'out_degree': 0,
            'total_degree': 0,
            'clustering_coeff': 0.0,
            'pagerank': 0.0,
            'eigenvector_centrality': 0.0,
            'betweenness_centrality': 0.0
        }

def compute_structural_metrics(G: nx.DiGraph, node: str) -> Dict[str, float]:
    """Compute structural metrics around a node."""
    try:
        # Get node neighborhoods
        successors = set(G.successors(node))
        predecessors = set(G.predecessors(node))
        
        # Reciprocity
        reciprocal = len(successors.intersection(predecessors))
        reciprocity = reciprocal / len(successors) if successors else 0.0
        
        # Local density
        neighborhood = successors.union(predecessors)
        if len(neighborhood) > 1:
            local_edges = G.subgraph(neighborhood).number_of_edges()
            max_edges = len(neighborhood) * (len(neighborhood) - 1)
            density = local_edges / max_edges if max_edges > 0 else 0.0
        else:
            density = 0.0
            
        # Support ratio in neighborhood
        pos_votes = sum(1 for _, _, d in G.edges(node, data=True) 
                       if d.get('sign', 0) > 0)
        total_votes = G.out_degree(node)
        support_ratio = pos_votes / total_votes if total_votes > 0 else 0.0
        
        return {
            'reciprocity': reciprocity,
            'local_density': density,
            'neighborhood_size': len(neighborhood),
            'support_ratio': support_ratio,
            'reciprocal_edges': reciprocal
        }
    except:
        return {
            'reciprocity': 0.0,
            'local_density': 0.0,
            'neighborhood_size': 0,
            'support_ratio': 0.0,
            'reciprocal_edges': 0
        }

def compute_temporal_metrics(timestamps: Dict[Tuple[str, str], float], 
                           node: str, current_time: float) -> Dict[str, float]:
    """Compute temporal metrics for a node."""
    try:
        # Get all timestamps involving this node
        node_times = [t for (u, v), t in timestamps.items() 
                     if u == node or v == node]
        
        if not node_times:
            return {
                'time_active': 0.0,
                'avg_time_between': 0.0,
                'vote_frequency': 0.0
            }
            
        # Activity duration
        first_time = min(node_times)
        time_active = current_time - first_time
        
        # Average time between votes
        if len(node_times) > 1:
            sorted_times = sorted(node_times)
            time_diffs = np.diff(sorted_times)
            avg_time_between = np.mean(time_diffs)
        else:
            avg_time_between = 0.0
            
        # Vote frequency (votes per day)
        vote_frequency = len(node_times) / max(1.0, time_active / 86400)  # seconds to days
        
        return {
            'time_active': time_active,
            'avg_time_between': avg_time_between,
            'vote_frequency': vote_frequency
        }
    except:
        return {
            'time_active': 0.0,
            'avg_time_between': 0.0,
            'vote_frequency': 0.0
        }

def compute_network_evolution_metrics(G: nx.DiGraph, 
                                   node: str,
                                   prev_metrics: Dict[str, float]) -> Dict[str, float]:
    """Compute metrics showing node's network evolution."""
    try:
        # Current metrics
        current = compute_centrality_metrics(G, node)
        
        # Calculate changes
        changes = {
            f'{k}_change': current[k] - prev_metrics.get(k, 0.0)
            for k in current.keys()
        }
        
        # Calculate rates of change (per time unit)
        time_diff = 1.0  # Use appropriate time difference
        rates = {
            f'{k}_rate': v / time_diff
            for k, v in changes.items()
        }
        
        return {**changes, **rates}
    except:
        return {}
