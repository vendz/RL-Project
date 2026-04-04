"""
Evaluate a trained DQN agent on the Rome test set and report SPC metrics.

Usage:
    python evaluate_dqn.py --checkpoint checkpoints/dqn_final.pt

Output:
    dqn_eval_results.csv  — per-graph crossing counts
    Prints SPC vs neato baseline to stdout
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from dqn_agent import DQNAgent
from env import GraphLayoutEnv
from metrics import spc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_graphs(rome_dir: str, test_graphs_file: str):
    """Load the held-out test graphs listed in *test_graphs_file*."""
    rome_path = Path(rome_dir)
    with open(test_graphs_file) as f:
        filenames = [line.strip() for line in f if line.strip()]
    graphs = []
    for name in filenames:
        path = rome_path / name
        G = nx.read_graphml(path)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        if G.number_of_nodes() >= 3 and G.number_of_edges() >= 2:
            graphs.append((path.stem, G))
    return graphs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # Load agent
    agent = DQNAgent(feat_dim=7, num_dirs=8, hidden_dim=args.hidden_dim, device=device)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}  (epsilon={agent.epsilon:.3f})")

    # Load test graphs
    test_graphs = load_test_graphs(args.rome_dir, args.test_graphs_file)
    print(f"Evaluating on {len(test_graphs)} test graphs (from '{args.test_graphs_file}') ...")

    env = GraphLayoutEnv(step_size=args.step_size, max_steps=args.max_steps)

    results = []
    for graph_id, G in test_graphs:
        state = env.reset(G)
        initial_xings = env.initial_crossings

        for _ in range(args.max_steps):
            action = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)
            if done:
                break

        results.append({
            "graph_id":       graph_id,
            "neato_xing":     initial_xings,     # neato initialisation
            "dqn_xing":       env.current_crossings,
        })

    df = pd.DataFrame(results)

    mean_neato = df["neato_xing"].mean()
    mean_dqn   = df["dqn_xing"].mean()
    spc_score  = spc(df["dqn_xing"], df["neato_xing"])

    print(f"\n{'='*50}")
    print(f"Test graphs:            {len(df)}")
    print(f"Mean neato crossings:   {mean_neato:.2f}")
    print(f"Mean DQN crossings:     {mean_dqn:.2f}")
    print(f"Mean improvement:       {mean_neato - mean_dqn:+.2f}")
    print(f"SPC vs neato:           {spc_score:+.2f}%  (negative = better)")
    print(f"{'='*50}\n")

    df.to_csv(args.output, index=False)
    print(f"Per-graph results saved to {args.output}")

    # If baseline CSV exists, compute SPC against all baselines
    baseline_csv = Path(args.rome_dir).parent / "baseline_metrics.csv"
    if baseline_csv.exists():
        bl = pd.read_csv(baseline_csv)
        merged = bl.merge(
            df[["graph_id", "dqn_xing"]].rename(columns={"graph_id": "graph_id"}),
            on="graph_id", how="inner"
        )
        if len(merged) > 0:
            print("SPC against all baselines (on matched graphs):")
            for col in ["neato_xing", "sfdp_xing", "smartgd_xing", "diff_gd_xing"]:
                if col in merged.columns:
                    s = spc(merged["dqn_xing"], merged[col])
                    marker = " ← better" if s < 0 else ""
                    print(f"  vs {col:20s}: {s:+.2f}%{marker}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Evaluate DQN on Rome test graphs")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--rome-dir",         default="rome")
    p.add_argument("--test-graphs-file", default="test_graphs.txt",
                   help="File produced by train_dqn.py listing held-out test graphs")
    p.add_argument("--max-steps",   type=int, default=200)
    p.add_argument("--step-size",   type=float, default=5.0)
    p.add_argument("--hidden-dim",  type=int,   default=128)
    p.add_argument("--output",      default="dqn_eval_results.csv")
    p.add_argument("--cpu",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(get_args())
