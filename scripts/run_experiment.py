#!/usr/bin/env python3
"""
End-to-end experiment with per-graph checkpointing.
Writes results after each graph to avoid losing work on timeout.
"""

import argparse
import glob
import os
import re
import sys

import networkx as nx
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
    m = re.match(r"grafo(\d+)\.", os.path.basename(filename))
    return int(m.group(1)) if m else None


def find_rome_graphs(rome_dir, start=10000, end=10100):
    files = glob.glob(os.path.join(rome_dir, "grafo*.graphml"))
    result = []
    for f in files:
        gid = get_graph_number(f)
        if gid is not None and start <= gid <= end:
            result.append((gid, f))
    result.sort(key=lambda x: x[0])
    return result


def run_single_graph(G, graph_id, device, rl_episodes=200, rl_steps=40,
                     rl_seeds=3, verbose=False):
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
        from src.stress import StressLoss
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

    if verbose:
        print(f"  Graph {graph_id} (n={n}, m={m}): neato={neato_xing}, diff_gd={dg_xing}, rl={rl_xing}")

    return {
        "graph_id": graph_id,
        "n_nodes": n,
        "n_edges": m,
        "neato_xing": neato_xing,
        "diff_gd_xing": dg_xing,
        "rl_xing": rl_xing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome-dir", default="rome")
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--end", type=int, default=10100)
    parser.add_argument("--rl-episodes", type=int, default=300)
    parser.add_argument("--rl-steps", type=int, default=40)
    parser.add_argument("--rl-seeds", type=int, default=3)
    parser.add_argument("--output", default="rl_results.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    graphs = find_rome_graphs(args.rome_dir, args.start, args.end)

    if args.shard is not None:
        graphs = graphs[args.shard::args.num_shards]

    print(f"Processing {len(graphs)} graphs (shard={args.shard}/{args.num_shards})")

    # Load existing results if available
    done_ids = set()
    if os.path.exists(args.output):
        df_existing = pd.read_csv(args.output)
        done_ids = set(df_existing["graph_id"])
        print(f"Resuming: {len(done_ids)} already done")

    # Open file in append mode for per-graph checkpointing
    output_file = open(args.output, 'a')
    write_header = len(done_ids) == 0

    for gid, path in tqdm(graphs):
        if gid in done_ids:
            continue

        G = load_rome_graph(path)
        try:
            res = run_single_graph(G, gid, device,
                                   rl_episodes=args.rl_episodes,
                                   rl_steps=args.rl_steps,
                                   rl_seeds=args.rl_seeds,
                                   verbose=args.verbose)

            # Write per-graph
            df_row = pd.DataFrame([res])
            df_row.to_csv(output_file, mode='a', header=write_header, index=False)
            output_file.flush()
            write_header = False

        except Exception as e:
            print(f"  ERROR on graph {gid}: {e}")

    output_file.close()

    # Final summary
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        print(f"Saved {len(df)} results to {args.output}")
        neato = df["neato_xing"].values
        print(f"SPC diff_gd vs neato: {spc(df['diff_gd_xing'].values, neato):+.2f}%")
        print(f"SPC rl vs neato:      {spc(df['rl_xing'].values, neato):+.2f}%")


if __name__ == "__main__":
    main()
