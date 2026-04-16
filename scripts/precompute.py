"""
Precompute and cache all inputs needed for RL training.

Caches per graph:
  - neato layout positions       -> cache/neato/{graph_id}.pt
  - Diff GD warm-start positions -> cache/diffgd/{graph_id}.pt
  - hard crossing count at warm-start -> cache/diffgd/{graph_id}_xing.pt
  - NetworkX graph object        -> cache/graphs/{graph_id}.pkl
  - GNN edge index tensor        -> cache/edge_index/{graph_id}.pt

Summary CSV: cache/init_crossings.csv

Run once before any RL training. Subsequent runs skip already-cached graphs.
Use --force to recompute everything.
"""

import os
import glob
import argparse
import pickle
import csv
import concurrent.futures
import sys

import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.xing import XingLoss
from src.stress import StressLoss

NEATO_DIR = "neato"
DIFFGD_DIR = "diffgd"
GRAPH_DIR = "graphs"
EDGE_INDEX_DIR = "edge_index"
INIT_CROSSINGS_CSV = "init_crossings.csv"


def graph_id(path):
    return os.path.basename(path).replace(".graphml", "")


def graph_number(gid):
    try:
        return int(gid.split(".")[0].replace("grafo", ""))
    except ValueError:
        return -1


def diffgd_warmstart(G, coords_neato, max_epochs=1000, patience=5, check_interval=10, lr=2.0):
    """
    Run Adam on soft XingLoss + 0.2 * StressLoss from neato init.
    Stops when hard crossing count hasn't improved for `patience` checks.
    Returns (best_coords, best_hard_crossings).
    """
    coords = coords_neato.clone().detach().requires_grad_(True)
    xing_soft = XingLoss(G, soft=True)
    xing_hard = XingLoss(G, soft=False)
    stress = StressLoss(G)
    optimizer = optim.Adam([coords], lr=lr)

    with torch.no_grad():
        best_hard = int(xing_hard(coords).item())
    best_coords = coords.detach().clone()
    no_improve = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = xing_soft(coords) + 0.2 * stress(coords)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % check_interval == 0:
            with torch.no_grad():
                hard = int(xing_hard(coords).item())
            if hard < best_hard:
                best_hard = hard
                best_coords = coords.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

    return best_coords, best_hard


def process_graph(args):
    graph_path, cache_dir, force = args
    gid = graph_id(graph_path)

    paths = {
        "neato":      os.path.join(cache_dir, NEATO_DIR, f"{gid}.pt"),
        "diffgd":     os.path.join(cache_dir, DIFFGD_DIR, f"{gid}.pt"),
        "xing":       os.path.join(cache_dir, DIFFGD_DIR, f"{gid}_xing.pt"),
        "graph":      os.path.join(cache_dir, GRAPH_DIR, f"{gid}.pkl"),
        "edge_index": os.path.join(cache_dir, EDGE_INDEX_DIR, f"{gid}.pt"),
    }

    if not force and all(os.path.exists(p) for p in paths.values()):
        init_xing = int(torch.load(paths["xing"], weights_only=True).item())
        return {"graph_id": gid, "init_crossings": init_xing, "status": "cached"}

    try:
        G = nx.read_graphml(graph_path)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")

        # neato layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
        coords_neato = torch.tensor(
            [[pos[v][0], pos[v][1]] for v in G.nodes()],
            dtype=torch.float32,
        )

        # Diff GD warm-start with convergence stopping
        warmstart_coords, init_xing = diffgd_warmstart(G, coords_neato)

        # GNN edge index [2, num_edges], both directions for undirected graphs
        edges = list(G.edges())
        if edges:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([ei, ei.flip(0)], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        torch.save(coords_neato, paths["neato"])
        torch.save(warmstart_coords, paths["diffgd"])
        torch.save(torch.tensor(init_xing), paths["xing"])
        torch.save(edge_index, paths["edge_index"])
        with open(paths["graph"], "wb") as f:
            pickle.dump(G, f)

        return {"graph_id": gid, "init_crossings": init_xing, "status": "success"}

    except Exception as e:
        return {"graph_id": gid, "init_crossings": None, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Precompute RL training cache")
    parser.add_argument("--rome_dir", default="rome")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="train=graphs<=9999, test=graphs 10000-10100, all=everything")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true", help="Recompute even if already cached")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    for d in [NEATO_DIR, DIFFGD_DIR, GRAPH_DIR, EDGE_INDEX_DIR]:
        os.makedirs(os.path.join(args.cache_dir, d), exist_ok=True)

    graph_files = sorted(glob.glob(os.path.join(args.rome_dir, "*.graphml")))
    if not graph_files:
        print(f"No .graphml files found in {args.rome_dir}")
        return
    print(f"Found {len(graph_files)} graphs.")

    if args.split == "train":
        graph_files = [f for f in graph_files if graph_number(graph_id(f)) <= 9999]
    elif args.split == "test":
        graph_files = [f for f in graph_files if 10000 <= graph_number(graph_id(f)) <= 10100]

    if args.shard is not None:
        graph_files = graph_files[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(graph_files)} graphs.")
    else:
        print(f"Processing {len(graph_files)} graphs (split={args.split}).")

    tasks = [(f, args.cache_dir, args.force) for f in graph_files]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(process_graph, tasks), total=len(tasks)):
            results.append(res)

    successful = [r for r in results if r["status"] in ("success", "cached")]
    failed = [r for r in results if r["status"] == "failed"]

    csv_path = os.path.join(args.cache_dir, INIT_CROSSINGS_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_id", "init_crossings"])
        writer.writeheader()
        for r in successful:
            if r["init_crossings"] is not None:
                writer.writerow({"graph_id": r["graph_id"], "init_crossings": r["init_crossings"]})

    print(f"\nDone. {len(successful)} succeeded, {len(failed)} failed.")
    if failed:
        print("Sample failures:")
        for r in failed[:5]:
            print(f"  {r['graph_id']}: {r['error']}")
    print(f"Cache: {args.cache_dir}/")
    print(f"Init crossings: {csv_path}")


if __name__ == "__main__":
    main()
