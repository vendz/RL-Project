import argparse, numpy as np, networkx as nx, torch
from pathlib import Path
import pandas as pd
from xing import XingLoss
from stress import StressLoss
from env import GraphLayoutEnv
from dqn_agent import DQNAgent
from metrics import spc

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
            coords = torch.tensor([[pos[v][0], pos[v][1]] for v in nodes], dtype=torch.float32)
        except Exception:
            coords = torch.rand(G.number_of_nodes(), 2) * 100.0
        xing = XingLoss(G, soft=False)
        with torch.no_grad():
            neato_xing = xing(coords).item()

        gd_xing = None
        if args.gd:
            xing_soft = XingLoss(G, soft=True)
            stress = StressLoss(G)
            coords = coords.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([coords], lr=2.0)
            for _ in range(300):
                opt.zero_grad()
                (xing_soft(coords) + 0.2 * stress(coords)).backward()
                opt.step()
            coords = coords.detach()
            with torch.no_grad():
                gd_xing = xing(coords).item()

        # SLS
        degs = np.array([G.degree(v) for v in nodes], dtype=np.float32)
        weights = degs / degs.sum()
        rng = np.random.default_rng()
        for _ in range(args.sls_iters):
            with torch.no_grad():
                cur = xing(coords).item()
            if cur == 0:
                break
            node = rng.choice(len(nodes), p=weights)
            for _ in range(16):
                dx, dy = rng.uniform(-5, 5), rng.uniform(-5, 5)
                new_coords = coords.clone()
                new_coords[node] += torch.tensor([dx, dy])
                with torch.no_grad():
                    new_xing = xing(new_coords).item()
                if new_xing < cur:
                    coords = new_coords
                    break
        with torch.no_grad():
            sls_xing = xing(coords).item()

        dqn_xing = None
        if args.dqn:
            state = env.reset(G, init_coords=coords.numpy())
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

    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"Graphs evaluated: {len(df)}")
    print(f"Mean neato:  {df['neato_xing'].mean():.2f}")
    if "gd_xing" in df:
        print(f"Mean GD:     {df['gd_xing'].mean():.2f}  SPC: {spc(df['gd_xing'], df['neato_xing']):+.2f}%")
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
