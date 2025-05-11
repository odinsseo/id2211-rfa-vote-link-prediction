#!/usr/bin/env python3
import csv
import argparse
import logging
import itertools

import pandas as pd
import networkx as nx


def build_user_talk_graph(data: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed user-talk graph from the DataFrame,
    accumulating:
     - weight: total messages
     - minor_count: total minor interactions
     - texts: list of text-lengths (in chars)
    Assumes data has columns: source, target, minor, textdata
    """
    G = nx.DiGraph()
    for _, row in data.iterrows():
        u = row["source"]
        v = row["target"]
        is_minor = bool(row.get("minor", False))
        text = row.get("textdata", "")
        length = len(text)

        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
            if is_minor:
                G[u][v]["minor_count"] += 1
            G[u][v]["texts"].append(length)
        else:
            G.add_edge(u, v,
                       weight=1,
                       minor_count=1 if is_minor else 0,
                       texts=[length])
    return G


def resume_index(output_file):
    """Count existing data rows (minus header)."""
    try:
        with open(output_file, newline='', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        return 0


def admin_user_pairs(G, admins, nbrs):
    """Yield (admin, user) with at least one shared neighbor."""
    for a in admins:
        for b in G.nodes():
            if a == b:
                continue
            if nbrs[a] & nbrs[b]:
                yield a, b


def batched(iterator, batch_size):
    """Yield successive batches of size batch_size."""
    it = iter(iterator)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger()

    # 1) Load interaction data and build graph
    log.info("Loading data from %s", args.data_csv)
    df = pd.read_csv(args.data_csv, dtype=str)
    pattern = r'(?i)bot(?:_?(?:\d+|[ivxlcdm]+))?$'
    is_bot = df["source"].str.contains(pattern, regex=True)  | df["target"].str.contains(pattern, regex=True)
    df = df[~is_bot]
    G_dir = build_user_talk_graph(df)
    log.info("Built directed graph: %d nodes, %d edges",
             G_dir.number_of_nodes(), G_dir.number_of_edges())

    # Sanitize nodes
    bad = [n for n in G_dir.nodes() if not isinstance(n, str) or (isinstance(n, float) and n != n)]
    if bad:
        log.warning("Removing %d invalid nodes", len(bad))
        G_dir.remove_nodes_from(bad)

    # Undirected for metrics
    G = G_dir.to_undirected()
    log.info("Undirected graph: %d nodes, %d edges",
             G.number_of_nodes(), G.number_of_edges())

    # 2) Load admins
    log.info("Loading admin list from %s", args.admin_csv)
    df_admin = pd.read_csv(args.admin_csv, dtype=str)
    admins = [u for u in df_admin['username'] if G.has_node(u)]
    dropped = set(df_admin['username']) - set(admins)
    if dropped:
        log.warning("Dropped %d admins not in graph", len(dropped))
    if not admins:
        log.error("No admins found")
        return

    # 3) Pre-cache neighbor sets
    nbrs = {u: set(G[u]) for u in G.nodes()}

    # 4) Build pairs
    all_pairs = list(admin_user_pairs(G, admins, nbrs))
    total = len(all_pairs)
    log.info("Total candidate pairs: %d", total)

    start = resume_index(args.output)
    log.info("Resuming at %d", start)

    with open(args.output, 'a', newline='', encoding='utf-8') as outcsv:
        writer = csv.writer(outcsv)
        if start == 0:
            writer.writerow([
                'admin','user',
                'jaccard','adamic_adar','pref_attachment',
                'admin_to_user_talk_count','user_to_admin_talk_count',
                'avg_textlen_admin_to_user','avg_textlen_user_to_admin'
            ])
            log.info("Wrote header")

        processed = start
        pairs_iter = iter(all_pairs[start:])
        batch_num = start // args.checkpoint

        for batch in batched(pairs_iter, args.checkpoint):
            batch_num += 1
            log.info("Batch %d/%d: %d", batch_num, (total // args.checkpoint)+1, len(batch))

            jacc = { (u,v):s for u,v,s in nx.jaccard_coefficient(G, ebunch=batch) }
            aa   = { (u,v):s for u,v,s in nx.adamic_adar_index(G, ebunch=batch) }
            pa   = { (u,v):s for u,v,s in nx.preferential_attachment(G, ebunch=batch) }

            for u,v in batch:
                sj = jacc.get((u,v),0.0)
                sa = aa.get((u,v),0.0)
                sp = pa.get((u,v),0.0)
                wt_uv = G_dir[u][v]['weight'] if G_dir.has_edge(u,v) else 0
                wt_vu = G_dir[v][u]['weight'] if G_dir.has_edge(v,u) else 0
                texts_uv = G_dir[u][v].get('texts',[]) if G_dir.has_edge(u,v) else []
                texts_vu = G_dir[v][u].get('texts',[]) if G_dir.has_edge(v,u) else []
                avg_uv = sum(texts_uv)/len(texts_uv) if texts_uv else 0.0
                avg_vu = sum(texts_vu)/len(texts_vu) if texts_vu else 0.0
                writer.writerow([u,v,f"{sj:.6f}",f"{sa:.6f}",f"{sp:.6f}",
                                 wt_uv,wt_vu,f"{avg_uv:.2f}",f"{avg_vu:.2f}"])

            outcsv.flush()
            processed += len(batch)
            pct = processed/total*100
            log.info("Processed %d/%d (%.2f%%)", processed, total, pct)

    log.info("Completed")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv",  required=True, help="Interaction data CSV")
    p.add_argument("--admin-csv", required=True, help="Admins CSV")
    p.add_argument("--output",    required=True, help="Output CSV")
    p.add_argument("--checkpoint",type=int, default=500, help="Batch size")
    args = p.parse_args()
    main(args)
