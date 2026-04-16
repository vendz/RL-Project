#!/usr/bin/env python3
"""
Compute layouts + coordinates for test graphs (10000-10100).
Saves .coord files and detects overlapping nodes.
If nodes overlap, penalizes crossings to 1.5x neato baseline.
"""

import argparse
import os
import re
import glob
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diff_gd import run_diff_gd, get_neato_crossings
from utils.metrics import spc
from src.rl_refine import ppo_refine
from src.xing import XingLoss


def load_rome_graph(path):
    G = nx.read_graphml(path)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def get_graph_number(filename):
    m = re.match(r"grafo(\d+)\.(\d+)\.graphml", os.path.basename(filename))
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_rome_graphs(rome_dir, start=10000, end=10100):
    files = glob.glob(os.path.join(rome_dir, "grafo*.graphml"))
    result = []
    for f in files:
        gid, seed = get_graph_number(f)
        if gid is not None and start <= gid <= end:
            result.append((gid, seed, f))
    result.sort(key=lambda x: (x[0], x[1]))
    return result


def has_overlapping_nodes(coords, eps=1e-6):
    """Check if any nodes have identical or nearly identical coordinates."""
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < eps:
                return True
    return False


def save_coords(coords, filename):
    """Save node coordinates to .coord file (x y per line)."""
    with open(filename, 'w') as f:
        for x, y in coords[:, :2]:  # Only save x, y
            f.write(f"{x:.6f} {y:.6f}\n")


def run_single_graph(G, graph_id, graph_seed, device, rl_episodes=300, rl_steps=40,
                     rl_seeds=3, verbose=False, output_dir="coords"):
    n, m = G.number_of_nodes(), G.number_of_edges()

    neato_xing, neato_coords = get_neato_crossings(G, device)

    dg_coords_neato, dg_xing_neato = run_diff_gd(
        G, epochs=300, lr=2.0, stress_weight=0.2, prog="neato", device=device
    )
    dg_coords_sfdp, dg_xing_sfdp = run_diff_gd(
        G, epochs=300, lr=2.0, stress_weight=0.2, prog="sfdp", device=device
    )

    if dg_xing_sfdp < dg_xing_neato:
        dg_coords, dg_xing = dg_coords_sfdp, dg_xing_sfdp
    else:
        dg_coords, dg_xing = dg_coords_neato, dg_xing_neato

    for _ in range(5):
        perturbed = dg_coords.clone()
        span = (dg_coords.max(dim=0).values - dg_coords.min(dim=0).values).clamp(min=1.0)
        perturbed = perturbed + torch.randn_like(perturbed) * span * 0.05
        perturbed = perturbed.detach().requires_grad_(True)
        xing_soft = XingLoss(G, soft=True, device=device)
        from stress import StressLoss
        stress_loss = StressLoss(G, device=device)
        xing_hard_check = XingLoss(G, soft=False, device=device)
        from torch.optim import Adam as _Adam
        opt = _Adam([perturbed], lr=1.0)
        for _ in range(150):
            opt.zero_grad()
            loss = xing_soft(perturbed) + 0.2 * stress_loss(perturbed)
            loss.backward()
            opt.step()
        with torch.no_grad():
            p_xing = int(xing_hard_check(perturbed).item())
        if p_xing < dg_xing:
            dg_xing = p_xing
            dg_coords = perturbed.detach()

    if dg_xing > 0:
        rl_coords, rl_xing = ppo_refine(
            G, dg_coords,
            n_episodes=rl_episodes,
            steps_per_episode=rl_steps,
            n_seeds=rl_seeds,
            verbose=verbose,
            device=device,
        )
    else:
        rl_coords, rl_xing = dg_coords, 0

    if rl_xing > dg_xing:
        rl_xing = dg_xing
        rl_coords = dg_coords

    # Check for overlapping nodes
    rl_coords_np = rl_coords.detach().cpu().numpy()
    has_overlap = has_overlapping_nodes(rl_coords_np)

    # Penalize if overlap detected
    if has_overlap:
        rl_xing = int(neato_xing * 1.5)

    # Save coordinates
    os.makedirs(output_dir, exist_ok=True)
    coord_file = os.path.join(output_dir, f"grafo{graph_id}.{graph_seed}.coord")
    save_coords(rl_coords_np, coord_file)

    if verbose:
        overlap_str = " [OVERLAP PENALIZED]" if has_overlap else ""
        print(f"  Graph {graph_id}.{graph_seed} (n={n}, m={m}): neato={neato_xing}, diff_gd={dg_xing}, rl={rl_xing}{overlap_str}")

    return {
        "graph_id": graph_id,
        "graph_seed": graph_seed,
        "n_nodes": n,
        "n_edges": m,
        "neato_xing": neato_xing,
        "diff_gd_xing": dg_xing,
        "rl_xing": rl_xing,
        "has_overlap": has_overlap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome-dir", default="rome")
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--end", type=int, default=10100)
    parser.add_argument("--rl-episodes", type=int, default=300)
    parser.add_argument("--rl-steps", type=int, default=40)
    parser.add_argument("--rl-seeds", type=int, default=3)
    parser.add_argument("--output", default="test_results.csv")
    parser.add_argument("--coords-dir", default="coords")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    graphs = find_rome_graphs(args.rome_dir, args.start, args.end)

    print(f"Processing {len(graphs)} test graphs ({args.start}-{args.end})")

    results = []
    for gid, gseed, path in tqdm(graphs):
        G = load_rome_graph(path)
        try:
            res = run_single_graph(G, gid, gseed, device,
                                   rl_episodes=args.rl_episodes,
                                   rl_steps=args.rl_steps,
                                   rl_seeds=args.rl_seeds,
                                   verbose=args.verbose,
                                   output_dir=args.coords_dir)
            results.append(res)
        except Exception as e:
            print(f"  ERROR on graph {gid}.{gseed}: {e}")

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} results to {args.output}")
    print(f"Coordinates saved to {args.coords_dir}/")

    overlap_count = df["has_overlap"].sum()
    if overlap_count > 0:
        print(f"⚠️  {overlap_count} graphs had overlapping nodes (penalized to 1.5x neato)")

    neato = df["neato_xing"].values
    print(f"\nSPC diff_gd vs neato: {spc(df['diff_gd_xing'].values, neato):+.2f}%")
    print(f"SPC rl vs neato:      {spc(df['rl_xing'].values, neato):+.2f}%")


if __name__ == "__main__":
    main()
