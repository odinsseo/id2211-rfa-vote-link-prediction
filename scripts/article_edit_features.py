#!/usr/bin/env python3
"""
Script to generate feature files for Wikipedia admin elections.
This script processes article edit data to create features based on user interactions
and graph metrics for admin election analysis.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx import bipartite
from tqdm import tqdm


class BipartiteMetricsCalculator:
    """Calculates metrics based on bipartite graph of user-article interactions."""

    @staticmethod
    def compute_basic_weight(common_neighbors: Set[str], nbr_weights: Dict) -> float:
        """Compute the basic weight between two nodes based on common neighbors."""
        if not common_neighbors:
            return 0.0

        weights = [
            (nbr_weights[nbr]["u"] + nbr_weights[nbr]["v"], nbr_weights[nbr]["total"])
            for nbr in common_neighbors
        ]
        weights = np.array(weights)
        return (
            np.sum(weights[:, 0]) / np.sum(weights[:, 1]) if weights.size > 0 else 0.0
        )

    @staticmethod
    def compute_weighted_jaccard(all_neighbors: Set[str], nbr_weights: Dict) -> float:
        """Compute the weighted Jaccard similarity between two nodes."""
        if not all_neighbors:
            return 0.0

        weights = [
            (nbr_weights[nbr]["u"], nbr_weights[nbr]["v"]) for nbr in all_neighbors
        ]
        weights = np.array(weights)
        if weights.size == 0:
            return 0.0

        intersection = np.minimum(weights[:, 0], weights[:, 1]).sum()
        union = np.maximum(weights[:, 0], weights[:, 1]).sum()

        return intersection / union if union != 0 else 0.0

    @staticmethod
    def compute_entropy_and_mutual_info(
        all_neighbors: Set[str], nbr_weights: Dict, total_weight: float
    ) -> Tuple[float, float]:
        """Compute both participation entropy and mutual information."""
        joint_entropy = 0.0
        mutual_info = 0.0

        for nbr in all_neighbors:
            nbr_total = nbr_weights[nbr]["total"]
            p_u = nbr_weights[nbr]["u"] / nbr_total if nbr_total > 0 else 0
            p_v = nbr_weights[nbr]["v"] / nbr_total if nbr_total > 0 else 0

            # Compute entropy
            if p_u > 0:
                joint_entropy += -p_u * np.log2(p_u)
            if p_v > 0:
                joint_entropy += -p_v * np.log2(p_v)

            # Compute mutual information
            if total_weight > 0:
                joint_prob = (
                    nbr_weights[nbr]["u"] + nbr_weights[nbr]["v"]
                ) / total_weight
                if p_u > 0 and p_v > 0 and joint_prob > 0:
                    mutual_info += joint_prob * np.log2(joint_prob / (p_u * p_v))

        return joint_entropy, mutual_info

    def weight_function(
        self, B: nx.DiGraph, u: str, v: str
    ) -> Tuple[float, float, float, float]:
        """Calculate multiple metrics between two users in a user-article bipartite graph."""
        # Get neighborhoods once
        unbrs = set(B.predecessors(u))
        vnbrs = set(B.predecessors(v))
        common_neighbors = unbrs & vnbrs
        all_neighbors = unbrs | vnbrs

        # Pre-compute all weights in a single dictionary
        nbr_weights = {
            nbr: {
                "u": B.edges[nbr, u]["weight"] if B.has_edge(nbr, u) else 0.0,
                "v": B.edges[nbr, v]["weight"] if B.has_edge(nbr, v) else 0.0,
                "total": B.out_degree(nbr, weight="weight"),
            }
            for nbr in all_neighbors
        }

        # Calculate total weight once
        total_weight = B.in_degree(u, weight="weight") + B.in_degree(v, weight="weight")

        # Compute all metrics using the pre-computed weights
        breadth_u = B.in_degree(u)
        breadth_v = B.in_degree(v)
        basic_weight = self.compute_basic_weight(common_neighbors, nbr_weights)
        weighted_jaccard = self.compute_weighted_jaccard(all_neighbors, nbr_weights)
        participation_entropy, mutual_info = self.compute_entropy_and_mutual_info(
            all_neighbors, nbr_weights, total_weight
        )

        return (
            breadth_u,
            breadth_v,
            basic_weight,
            weighted_jaccard,
            participation_entropy,
            mutual_info,
        )


class GraphFeatureExtractor:
    """Extracts features from both bipartite and monopartite graphs for user interactions."""

    def __init__(self):
        self.bipartite_calculator = BipartiteMetricsCalculator()

    def compute_user_article_metrics(
        self, B: nx.DiGraph, voters: List[str], candidate: str
    ) -> pd.DataFrame:
        """Compute weight metrics for all pairs of users using numpy vectorization."""
        # Filter valid users that exist in the graph
        valid_voters = [u for u in sorted(voters) if u in B]
        valid_candidate = candidate if candidate in B else None

        if valid_candidate is None:
            return pd.DataFrame(
                columns=[
                    "voter",
                    "candidate",
                    "collaboration",
                    "pairwise_jaccard",
                    "participation_entropy",
                    "mutual_information",
                ]
            )

        # Create user pairs using combinations
        pairs = list(itertools.product(valid_voters, [valid_candidate]))

        if not pairs:
            return pd.DataFrame(
                columns=[
                    "voter",
                    "candidate",
                    "collaboration",
                    "pairwise_jaccard",
                    "participation_entropy",
                    "mutual_information",
                ]
            )

        # Pre-allocate metrics list
        metrics = []
        for voter, valid_candidate in pairs:
            # Compute metrics for each pair
            (
                breadth_voter,
                breadth_candidate,
                basic_w,
                jaccard_w,
                entropy,
                mutual_info,
            ) = self.bipartite_calculator.weight_function(B, voter, valid_candidate)

            metrics.append(
                {
                    "voter": voter,
                    "candidate": valid_candidate,
                    "breadth_voter": breadth_voter,
                    "breadth_candidate": breadth_candidate,
                    "collaboration": basic_w,
                    "pairwise_jaccard": jaccard_w,
                    "participation_entropy": entropy,
                    "mutual_information": mutual_info,
                }
            )

        # Create DataFrame and sort
        return pd.DataFrame(metrics).sort_values("collaboration", ascending=False)

    @staticmethod
    def compute_user_user_metrics(
        G: nx.Graph, voters: List[str], candidate: str
    ) -> pd.DataFrame:
        """Compute graph metrics for pairs of users in a social network."""
        # Filter valid users that exist in the graph
        valid_voters = [u for u in sorted(voters) if u in G]
        valid_candidate = candidate if candidate in G else None

        if valid_candidate is None:
            return pd.DataFrame(
                columns=[
                    "voter",
                    "candidate",
                    "jaccard",
                    "adamic_adar",
                    "pref_attachment",
                    "pagerank_voter",
                    "pagerank_candidate",
                ]
            )

        # Create user pairs using combinations
        pairs = list(itertools.product(valid_voters, [valid_candidate]))

        if not pairs:
            return pd.DataFrame(
                columns=[
                    "voter",
                    "candidate",
                    "jaccard",
                    "adamic_adar",
                    "pref_attachment",
                    "pagerank_voter",
                    "pagerank_candidate",
                ]
            )

        # Pre-compute centrality measures
        pagerank = nx.pagerank(G)
        jaccard = {(u, v): p for u, v, p in nx.jaccard_coefficient(G, ebunch=pairs)}
        adamic_adar = {(u, v): p for u, v, p in nx.adamic_adar_index(G, ebunch=pairs)}
        pref_attach = {
            (u, v): p for u, v, p in nx.preferential_attachment(G, ebunch=pairs)
        }

        metrics = []
        for u, v in pairs:
            metrics.append(
                {
                    "voter": u,
                    "candidate": v,
                    "jaccard": jaccard[(u, v)],
                    "adamic_adar": adamic_adar[(u, v)],
                    "pref_attachment": pref_attach[(u, v)],
                    "pagerank_voter": pagerank[u],
                    "pagerank_candidate": pagerank[v],
                }
            )

        return pd.DataFrame(metrics)


class GraphDataProcessor:
    """Processes article edit data to create interaction graphs."""

    def __init__(self, time_window: float = 30):
        self.time_window = time_window

    def create_graph_data(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp,
        users: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create graph data from article talk data up to a cutoff date."""
        required_cols = {"user", "namespace", "timestamp", "minor", "textdata"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Filter data up to cutoff date
        filtered_data = df[df["timestamp"] <= cutoff_date].copy()

        if users is not None:
            filtered_data = filtered_data[filtered_data["user"].isin(users)]

        # Calculate weights using exponential decay
        lamb = np.log(2) / self.time_window
        filtered_data["weight"] = (
            np.log(filtered_data["textdata"] + 1)
            * (1 - 0.25 * filtered_data["minor"])
            * np.exp(-lamb * (filtered_data["timestamp"].rsub(cutoff_date).dt.days))
        )

        return (
            filtered_data.groupby(["user", "namespace"], observed=True)["weight"]
            .sum()
            .reset_index(name="weight")
            .sort_values("weight", ascending=False)
        )

    @staticmethod
    def bipartite_graph_from_edgelist(
        edgelist: pd.DataFrame,
        source: str = "namespace",
        target: str = "user",
        weight: str = "weight",
        B: Optional[nx.DiGraph] = None,
    ) -> nx.DiGraph:
        """Create a bipartite graph from an edge list DataFrame efficiently."""
        if B is None:
            B = nx.DiGraph()

        # Add nodes in bulk operations
        B.add_nodes_from(edgelist[source].unique(), bipartite=0)
        B.add_nodes_from(edgelist[target].unique(), bipartite=1)

        # Add edges in bulk operation
        edges = list(zip(edgelist[source], edgelist[target], edgelist[weight]))
        B.add_weighted_edges_from(edges)

        return B


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate feature files for Wikipedia admin elections"
    )
    parser.add_argument(
        "--time-window",
        type=float,
        default=30,
        help="Time window parameter (delta_t) in days for exponential decay (default: 30)",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=1.0,
        help="Threshold for filtering graph data weights (default: 1.0)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2005-08-31",
        help="Starting date for processing (format: YYYY-MM-DD, default: 2005-08-31)",
    )
    parser.add_argument(
        "--article-edits",
        type=str,
        default="../data/article_edits.csv",
        help="Path to article edits CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/features",
        help="Directory for output feature CSV files",
    )
    return parser


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main function to run the feature generation process."""
    parser = setup_argparse()
    args = parser.parse_args()
    setup_logging()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processors
    graph_processor = GraphDataProcessor(time_window=args.time_window)
    feature_extractor = GraphFeatureExtractor()

    # Load data
    logging.info("Loading article edits data in chunks...")
    article_data = []
    for chunk in tqdm(pd.read_csv(args.article_edits, chunksize=1000000)):
        chunk["timestamp"] = (
            pd.to_datetime(chunk["timestamp"]).values.astype(np.int64) // 10**9
        )
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s")
        chunk["minor"] = chunk["minor"].astype(bool)
        article_data.append(chunk)

    article_edits = pd.concat(article_data, ignore_index=True)
    chunk["user"] = chunk["user"].astype("category")
    chunk["namespace"] = chunk["namespace"].astype("category")
    logging.info(f"Loaded {len(article_edits):,} article edits")

    logging.info("Loading voting data...")
    voting_data = pd.read_csv(
        "../data/votes_with_election_info.csv",
        parse_dates=["start_time"],
    )
    voting_data.sort_values(by=["start_time"], inplace=True)

    # Filter elections
    missing_elections = voting_data[
        voting_data["start_time"] > pd.Timestamp(args.start_date)
    ]

    # Process each election date
    for date in tqdm(missing_elections["start_time"].unique()):
        try:
            # Create graph data
            graph_data = graph_processor.create_graph_data(
                article_edits, cutoff_date=date
            )
            graph_data = graph_data[graph_data["weight"] >= args.weight_threshold]

            # Get election information
            rows = voting_data[voting_data["start_time"] == date]
            voters = voting_data[voting_data["start_time"] == date]["voter"].unique()
            candidate = rows["candidate"].unique()[0]

            # Create and analyze graphs
            bipartite_graph = graph_processor.bipartite_graph_from_edgelist(graph_data)
            bipartite_metrics = feature_extractor.compute_user_article_metrics(
                bipartite_graph, voters, candidate
            )

            user_graph = bipartite.projected_graph(
                bipartite_graph, graph_data["user"].unique()
            ).to_undirected()
            monopartite_metrics = feature_extractor.compute_user_user_metrics(
                user_graph, voters, candidate
            )

            # Merge metrics and save
            features = pd.merge(
                bipartite_metrics, monopartite_metrics, on=["voter", "candidate"]
            )

            edit_counts = (
                article_edits[article_edits["timestamp"] <= date]
                .groupby("user", observed=True)
                .size()
            )
            features["voter_edits"] = features["voter"].map(edit_counts)
            features["candidate_edits"] = features["candidate"].map(edit_counts)

            if not features.empty:
                output_file = (
                    output_dir
                    / f"article_edits.{date.strftime('%Y-%m-%d')}.{candidate}.csv"
                )
                features.to_csv(output_file, index=False)
                logging.info(
                    f"Saved features for {candidate} ({date.strftime('%Y-%m-%d')})"
                )

        except Exception as e:
            logging.error(f"Error processing election on {date}: {str(e)}")


if __name__ == "__main__":
    import itertools

    main()
