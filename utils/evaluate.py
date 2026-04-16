"""
Evaluate a trained policy on the test set (graphs 10000-10100).

Runs per-graph RL: loads warm-start from cache, runs the policy for max_steps,
reports final crossing count and SPC vs neato.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --cache_dir cache --out results.csv
"""

import os
import glob
import argparse
import csv
import sys

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GraphLayoutEnv
from src.policy import GNNPolicy
from utils.metrics import spc


def graph_number(gid):
    try:
        return int(gid.split(".")[0].replace("grafo", ""))
    except ValueError:
        return -1


def evaluate(args):
    # Load test graph IDs
    cache_graphs = glob.glob(os.path.join(args.cache_dir, "graphs", "*.pkl"))
    all_ids = [os.path.basename(p).replace(".pkl", "") for p in cache_graphs]
    test_ids = [g for g in all_ids if 10000 <= graph_number(g) <= 10100]
    if not test_ids:
        raise RuntimeError(f"No test graphs found in {args.cache_dir}. Run precompute.py --split test first.")
    print(f"Evaluating on {len(test_ids)} test graphs.")

    # Load policy
    policy = GNNPolicy(hidden_dim=args.hidden_dim)
    policy.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    policy.eval()

    # Load neato baseline crossings from cache CSV
    neato_xings = {}
    neato_csv = os.path.join(args.cache_dir, "neato_crossings.csv")
    if os.path.exists(neato_csv):
        with open(neato_csv) as f:
            for row in csv.DictReader(f):
                neato_xings[row["graph_id"]] = int(row["neato_crossings"])

    results = []
    env = GraphLayoutEnv(test_ids, cache_dir=args.cache_dir, patience=args.patience)

    for gid in tqdm(test_ids):
        # Reset to this specific graph
        env.graph_ids = [gid]
        obs, _ = env.reset(seed=0)
        done = False

        while not done:
            coords     = torch.tensor(obs["coords"],     dtype=torch.float32)
            edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
            with torch.no_grad():
                node_idx, dx, dy, _, _ = policy.act(coords, edge_index)
            obs, _, terminated, truncated, _ = env.step_with_node(node_idx, dx, dy)
            done = terminated or truncated

        final_xing = env.current_crossings
        # Load initial (warm-start) crossings for this graph
        xing_path = os.path.join(args.cache_dir, "diffgd", f"{gid}_xing.pt")
        init_xing = int(torch.load(xing_path, weights_only=True).item())

        results.append({
            "graph_id":      gid,
            "init_xing":     init_xing,
            "final_xing":    final_xing,
            "neato_xing":    neato_xings.get(gid, None),
            "improvement":   init_xing - final_xing,
        })

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_id", "init_xing", "final_xing", "neato_xing", "improvement"])
        writer.writeheader()
        writer.writerows(results)

    # Compute SPC vs warm-start (Diff GD)
    init_arr  = np.array([r["init_xing"]  for r in results], dtype=float)
    final_arr = np.array([r["final_xing"] for r in results], dtype=float)
    spc_vs_diffgd = spc(final_arr, init_arr)

    print(f"\nResults saved to {args.out}")
    print(f"Mean init crossings (Diff GD warm-start): {init_arr.mean():.2f}")
    print(f"Mean final crossings (RL agent):           {final_arr.mean():.2f}")
    print(f"SPC vs Diff GD warm-start:                 {spc_vs_diffgd:.2f}%")

    # SPC vs neato if available
    if any(r["neato_xing"] is not None for r in results):
        neato_arr = np.array([r["neato_xing"] for r in results if r["neato_xing"] is not None], dtype=float)
        rl_arr    = np.array([r["final_xing"]  for r in results if r["neato_xing"] is not None], dtype=float)
        spc_vs_neato = spc(rl_arr, neato_arr)
        print(f"SPC vs Neato:                              {spc_vs_neato:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy on test graphs")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--cache_dir",  default="cache")
    parser.add_argument("--patience",   type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out",        default="eval_results.csv")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
