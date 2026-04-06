"""
Evaluate the hybrid GD + DQN pipeline on the Rome test set.

Two-phase evaluation per graph:
  Phase 1 — run Adam gradient descent (XingLoss soft + StressLoss) from the
             neato initialisation for --gd-iterations steps.
  Phase 2 — run the DQN agent from the GD-optimised layout for --max-steps.

Three crossing counts are recorded per graph:
  neato_xing  — original Graphviz neato layout (baseline)
  gd_xing     — after Phase 1 (GD only)
  hybrid_xing — after Phase 2 (GD + DQN)

SPC is reported for GD vs neato, hybrid vs neato, and hybrid vs GD.

Usage:
    python evaluate_hybrid.py --checkpoint checkpoints_hybrid/hybrid_final.pt
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
from stress import StressLoss
from xing import XingLoss


# ---------------------------------------------------------------------------
# Graph loading
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
# Gradient-descent layout phase (Phase 1)
# ---------------------------------------------------------------------------

def gd_layout(G: nx.Graph, num_iterations: int, lr: float) -> np.ndarray:
    """
    Run Adam on XingLoss (soft=True) + StressLoss starting from the neato
    (or random fallback) layout.

    Returns:
        coords_np: numpy [n, 2] of optimised positions
        neato_crossings: float — crossing count of the raw neato seed layout
    """
    n = G.number_of_nodes()

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
        nodes = sorted(G.nodes())
        init = torch.tensor(
            [[pos[v][0], pos[v][1]] for v in nodes], dtype=torch.float32
        )
    except Exception:
        init = torch.rand(n, 2, dtype=torch.float32) * 100.0

    # Record neato crossing count before GD
    xing_hard = XingLoss(G, device=None, soft=False)
    with torch.no_grad():
        neato_crossings = xing_hard(init).item()

    coords = init.clone().detach().requires_grad_(True)

    xing_soft   = XingLoss(G, device=None, soft=True)
    stress_loss = StressLoss(G, device=None)

    optimizer = torch.optim.Adam([coords], lr=lr)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = xing_soft(coords) + stress_loss(coords)
        loss.backward()
        optimizer.step()

    # Evaluate hard crossing count after GD
    optimised = coords.detach()
    with torch.no_grad():
        gd_crossings = xing_hard(optimised).item()

    return optimised.numpy(), neato_crossings, gd_crossings


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
    print(
        f"Evaluating on {len(test_graphs)} test graphs "
        f"(from '{args.test_graphs_file}') ..."
    )
    print(
        f"GD: {args.gd_iterations} iterations @ lr={args.gd_lr}  |  "
        f"DQN: {args.max_steps} steps"
    )

    env = GraphLayoutEnv(step_size=args.step_size, max_steps=args.max_steps)

    results = []
    for i, (graph_id, G) in enumerate(test_graphs):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test_graphs)}] {graph_id} ...", flush=True)

        # Phase 1: gradient-descent layout
        init_coords, neato_xing, gd_xing = gd_layout(
            G, args.gd_iterations, args.gd_lr
        )

        # Phase 2: DQN fine-tuning from GD layout (greedy, no exploration)
        state = env.reset(G, init_coords=init_coords)
        for _ in range(args.max_steps):
            action = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)
            if done:
                break

        results.append({
            "graph_id":    graph_id,
            "neato_xing":  neato_xing,
            "gd_xing":     gd_xing,
            "hybrid_xing": env.current_crossings,
        })

    df = pd.DataFrame(results)

    mean_neato  = df["neato_xing"].mean()
    mean_gd     = df["gd_xing"].mean()
    mean_hybrid = df["hybrid_xing"].mean()

    spc_gd     = spc(df["gd_xing"],     df["neato_xing"])
    spc_hybrid = spc(df["hybrid_xing"], df["neato_xing"])
    spc_dqn_on_top = spc(df["hybrid_xing"], df["gd_xing"])

    print(f"\n{'='*55}")
    print(f"Test graphs:                {len(df)}")
    print(f"Mean neato crossings:       {mean_neato:.2f}")
    print(f"Mean GD crossings:          {mean_gd:.2f}")
    print(f"Mean hybrid crossings:      {mean_hybrid:.2f}")
    print(f"")
    print(f"SPC  GD vs neato:           {spc_gd:+.2f}%  (negative = better)")
    print(f"SPC  hybrid vs neato:       {spc_hybrid:+.2f}%  (negative = better)")
    print(f"SPC  DQN on top of GD:      {spc_dqn_on_top:+.2f}%  (negative = better)")
    print(f"{'='*55}\n")

    df.to_csv(args.output, index=False)
    print(f"Per-graph results saved to {args.output}")

    # Compare against all baselines if a baseline CSV exists
    baseline_csv = Path(args.rome_dir).parent / "baseline_metrics.csv"
    if baseline_csv.exists():
        bl = pd.read_csv(baseline_csv)
        merged = bl.merge(
            df[["graph_id", "gd_xing", "hybrid_xing"]],
            on="graph_id",
            how="inner",
        )
        if len(merged) > 0:
            print(f"SPC against all baselines (on {len(merged)} matched graphs):")
            for col in ["neato_xing", "sfdp_xing", "smartgd_xing", "diff_gd_xing"]:
                if col in merged.columns:
                    s_gd  = spc(merged["gd_xing"],     merged[col])
                    s_hyb = spc(merged["hybrid_xing"], merged[col])
                    print(
                        f"  vs {col:20s}:  "
                        f"GD {s_gd:+.2f}%  |  hybrid {s_hyb:+.2f}%"
                    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate hybrid GD + DQN pipeline on Rome test graphs"
    )
    p.add_argument("--checkpoint",       required=True, help="Path to .pt checkpoint")
    p.add_argument("--rome-dir",         default="rome")
    p.add_argument("--test-graphs-file", default="test_graphs.txt")
    p.add_argument("--max-steps",        type=int,   default=200,
                   help="DQN steps per graph (Phase 2)")
    p.add_argument("--step-size",        type=float, default=5.0)
    p.add_argument("--hidden-dim",       type=int,   default=128)
    p.add_argument("--gd-iterations",    type=int,   default=100,
                   help="Adam GD iterations per graph (Phase 1)")
    p.add_argument("--gd-lr",            type=float, default=0.01,
                   help="Adam learning rate for GD phase")
    p.add_argument("--output",           default="hybrid_eval_results.csv")
    p.add_argument("--cpu",              action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(get_args())
