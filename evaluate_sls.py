import argparse, numpy as np, networkx as nx, torch
from pathlib import Path
import pandas as pd
from xing import XingLoss
from stress import StressLoss
from env import GraphLayoutEnv
from dqn_agent import DQNAgent
from metrics import spc

# ---------------------------------------------------------------------------
# Fast numpy crossing counter (avoids torch overhead in SLS loop)
# ---------------------------------------------------------------------------

def _make_pairs(edge_arr):
    """Precompute all non-adjacent edge pair indices for a graph."""
    E = len(edge_arr)
    ii, jj = np.triu_indices(E, k=1)
    ei, ej = edge_arr[ii], edge_arr[jj]
    valid = ~((ei[:,0]==ej[:,0])|(ei[:,0]==ej[:,1])|(ei[:,1]==ej[:,0])|(ei[:,1]==ej[:,1]))
    return ii[valid].astype(np.int32), jj[valid].astype(np.int32)

def _count_crossings(coords, edge_arr, pi, pj):
    """Count intersecting edge pairs from precomputed pair indices."""
    if len(pi) == 0:
        return 0
    p1, p2 = coords[edge_arr[pi, 0]], coords[edge_arr[pi, 1]]
    p3, p4 = coords[edge_arr[pj, 0]], coords[edge_arr[pj, 1]]
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0]
    par = np.abs(cross) < 1e-10
    sc = np.where(par, 1.0, cross)
    qp = p3 - p1
    t = (qp[:,0]*d2[:,1] - qp[:,1]*d2[:,0]) / sc
    u = (qp[:,0]*d1[:,1] - qp[:,1]*d1[:,0]) / sc
    return int((~par & (t > 0) & (t < 1) & (u > 0) & (u < 1)).sum())

def _incident_mask(node, edge_arr, pi, pj):
    """Boolean mask of pairs that involve at least one edge incident to node."""
    inc = (edge_arr[:,0] == node) | (edge_arr[:,1] == node)
    return inc[pi] | inc[pj]

# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = None
    if args.dqn:
        agent = DQNAgent(feat_dim=7, num_dirs=8, hidden_dim=128, device=device)
        agent.load(args.checkpoint)
        env = GraphLayoutEnv(step_size=5.0, max_steps=200)

    with open(args.test_graphs_file) as f:
        names = [l.strip() for l in f if l.strip()]
    results = []
    for name in names:
        G = nx.read_graphml(Path(args.rome_dir) / name)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
            continue
        nodes = sorted(G.nodes())
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
            coords = np.array([[pos[v][0], pos[v][1]] for v in nodes], dtype=np.float32)
        except Exception:
            coords = np.random.rand(G.number_of_nodes(), 2).astype(np.float32) * 100.0

        edge_arr = np.array(list(G.edges()), dtype=np.int32)
        pi, pj = _make_pairs(edge_arr)
        neato_xing = _count_crossings(coords, edge_arr, pi, pj)

        gd_xing = None
        if args.gd:
            xing_soft = XingLoss(G, soft=True)
            stress = StressLoss(G)
            t_coords = torch.tensor(coords).requires_grad_(True)
            opt = torch.optim.Adam([t_coords], lr=2.0)
            for _ in range(300):
                opt.zero_grad()
                (xing_soft(t_coords) + 0.2 * stress(t_coords)).backward()
                opt.step()
            coords = t_coords.detach().numpy()
            gd_xing = _count_crossings(coords, edge_arr, pi, pj)

        # SLS — maintain running crossing count; only recount incident-edge pairs per move
        degs = np.array([G.degree(v) for v in nodes], dtype=np.float32)
        weights = degs / degs.sum()
        rng = np.random.default_rng()
        cur = _count_crossings(coords, edge_arr, pi, pj)
        for _ in range(args.sls_iters):
            if cur == 0:
                break
            node = rng.choice(len(nodes), p=weights)
            mask = _incident_mask(node, edge_arr, pi, pj)
            pi_m, pj_m = pi[mask], pj[mask]
            old_inc = _count_crossings(coords, edge_arr, pi_m, pj_m)
            for _ in range(16):
                new_coords = coords.copy()
                new_coords[node] += rng.uniform(-5, 5, size=2).astype(np.float32)
                new_inc = _count_crossings(new_coords, edge_arr, pi_m, pj_m)
                if new_inc < old_inc:
                    coords = new_coords
                    cur = cur - old_inc + new_inc
                    break
        sls_xing = cur

        dqn_xing = None
        if args.dqn:
            state = env.reset(G, init_coords=coords)
            for _ in range(200):
                action = agent.select_action(state, training=False)
                state, _, done, _ = env.step(action)
                if done:
                    break
            dqn_xing = env.current_crossings

        row = {"graph_id": Path(name).stem, "neato_xing": neato_xing, "sls_xing": sls_xing}
        if gd_xing is not None:
            row["gd_xing"] = gd_xing
        if dqn_xing is not None:
            row["dqn_xing"] = dqn_xing
        results.append(row)
        stages = f"neato={neato_xing:.0f}"
        if gd_xing is not None:
            stages += f" gd={gd_xing:.0f}"
        stages += f" sls={sls_xing:.0f}"
        if dqn_xing is not None:
            stages += f" dqn={dqn_xing:.0f}"
        print(f"{Path(name).stem}: {stages} (done {len(results)}/{len(names)})", flush=True)

    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"Graphs evaluated: {len(df)}")
    print(f"Mean neato:  {df['neato_xing'].mean():.2f}")
    if "gd_xing" in df:
        print(f"Mean GD:     {df['gd_xing'].mean():.2f}  SPC: {spc(df['gd_xing'],  df['neato_xing']):+.2f}%")
    print(f"Mean SLS:    {df['sls_xing'].mean():.2f}  SPC: {spc(df['sls_xing'], df['neato_xing']):+.2f}%")
    if "dqn_xing" in df:
        print(f"Mean DQN:    {df['dqn_xing'].mean():.2f}  SPC: {spc(df['dqn_xing'], df['neato_xing']):+.2f}%")
    print(f"{'='*50}\n")
    df.to_csv(args.output, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--rome-dir", default="rome")
parser.add_argument("--test-graphs-file", default="test_graphs.txt")
parser.add_argument("--sls-iters", type=int, default=500)
parser.add_argument("--output", default="sls_eval_results.csv")
parser.add_argument("--gd", action="store_true")
parser.add_argument("--dqn", action="store_true")
parser.add_argument("--checkpoint", default="checkpoints/dqn_final.pt")
parser.add_argument("--cpu", action="store_true")
evaluate(parser.parse_args())
