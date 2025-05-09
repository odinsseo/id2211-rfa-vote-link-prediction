#!/usr/bin/env python3
"""Create directed, signed vote graph from Wikipedia election data."""

import csv
import sys
import logging
from pathlib import Path
from datetime import datetime

import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_votes(votes_file: str) -> nx.DiGraph:
    """Load votes from CSV and create directed graph.
    
    Args:
        votes_file: Path to votes CSV file
        
    Returns:
        NetworkX directed graph with vote edges
    """
    votes_path = Path(votes_file)
    if not votes_path.exists():
        raise FileNotFoundError(f"Votes file not found: {votes_file}")
        
    # Create directed graph
    G = nx.DiGraph()
    
    try:
        with open(votes_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                voter = row['voter']
                candidate = row['candidate']
                vote = int(row['vote'])
                timestamp = row['vote_time']
                
                # Skip neutral votes (0)
                if vote == 0:
                    continue
                    
                # Validate timestamp
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    # Skip if year is invalid (e.g., beyond 2025)
                    if dt.year > 2025:
                        logger.warning(f"Skipping vote with invalid timestamp: {timestamp}")
                        continue
                    
                    # Add edge with vote and timestamp attributes
                    G.add_edge(voter, candidate, 
                              sign=vote,
                              timestamp=timestamp)
                except ValueError:
                    logger.warning(f"Skipping vote with invalid timestamp format: {timestamp}")
                    continue
                
    except Exception as e:
        logger.error(f"Error loading votes: {e}")
        raise
        
    return G

def analyze_graph(G: nx.DiGraph):
    """Print basic analysis of the vote graph.
    
    Args:
        G: NetworkX directed graph with vote edges
    """
    logger.info(f"Number of nodes (users): {G.number_of_nodes()}")
    logger.info(f"Number of edges (votes): {G.number_of_edges()}")
    
    # Count positive vs negative votes
    pos_votes = sum(1 for u,v,d in G.edges(data=True) if d['sign'] > 0)
    neg_votes = sum(1 for u,v,d in G.edges(data=True) if d['sign'] < 0)
    
    logger.info(f"Positive votes: {pos_votes}")
    logger.info(f"Negative votes: {neg_votes}")
    
    # Get time range
    times = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') 
             for _,_,d in G.edges(data=True)]
    if times:
        start = min(times)
        end = max(times)
        logger.info(f"Time range: {start} to {end}")
    else:
        logger.warning("No timestamped edges found in graph")

def export_graph(G: nx.DiGraph, output_file: str):
    """Export graph to GraphML format.
    
    Args:
        G: NetworkX directed graph
        output_file: Path to output GraphML file
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_graphml(G, output_file)
        logger.info(f"Graph exported to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        raise

def main():
    """Main entry point."""
    # Get project root directory (parent of utils/)
    root_dir = Path(__file__).parent.parent
    
    # Set file paths relative to root
    votes_file = root_dir / "data" / "votes.csv"
    graph_file = root_dir / "data" / "vote_graph.graphml"
    
    try:
        logger.info("Loading votes and creating graph...")
        G = load_votes(str(votes_file))
        
        logger.info("\nGraph analysis:")
        analyze_graph(G)
        
        logger.info(f"\nExporting graph...")
        export_graph(G, str(graph_file))
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
