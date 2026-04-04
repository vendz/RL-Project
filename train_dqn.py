"""
Training script for DQN graph layout agent.

Usage:
    python train_dqn.py [options]

Key hyperparameters (with defaults):
    --num-episodes 10000  total training episodes
    --max-steps    50     environment steps per episode
    --num-train    10000  graphs reserved for training (rest held out as test)
    --batch-size   32
    --step-size    5.0    pixels to move a node per action
"""

import argparse
import os
import random
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from dqn_agent import DQNAgent
from env import GraphLayoutEnv
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_split_rome_graphs(rome_dir: str, num_train: int, test_graphs_file: str):
    """
    Load all Rome graphs, shuffle with seed=42, split into train/test.
    Writes test filenames to *test_graphs_file* for reproducible evaluation.
    Returns only the train graphs (as NetworkX objects).
    """
    all_files = sorted(Path(rome_dir).glob("*.graphml"))
    print(f"  Found {len(all_files)} total .graphml files in '{rome_dir}'")

    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    train_files = shuffled[:num_train]
    test_files  = shuffled[num_train:]

    # Persist test split so evaluate_dqn.py uses the exact same graphs
    with open(test_graphs_file, "w") as f:
        for p in test_files:
            f.write(p.name + "\n")
    print(f"  Test split: {len(test_files)} files → saved to '{test_graphs_file}'")

    print(f"  Loading {len(train_files)} training files ...")
    graphs = []
    for i, path in enumerate(train_files):
        if i % 500 == 0:
            print(f"    [{i}/{len(train_files)}] loading {path.name} ...")
        G = nx.read_graphml(path)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        if G.number_of_nodes() >= 3 and G.number_of_edges() >= 2:
            graphs.append(G)
    return graphs


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    print(f"\n[1/4] Loading Rome graphs (train={args.num_train}, seed=42) ...")
    train_graphs = load_and_split_rome_graphs(
        args.rome_dir, args.num_train, args.test_graphs_file
    )
    print(f"  Done — {len(train_graphs)} usable training graphs loaded")

    print(f"\n[2/4] Building environment and agent ...")
    env    = GraphLayoutEnv(step_size=args.step_size, max_steps=args.max_steps)
    agent  = DQNAgent(
        feat_dim=7,
        num_dirs=8,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        device=device,
    )
    buffer = ReplayBuffer(capacity=args.buffer_size)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"  Replay buffer capacity: {args.buffer_size}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Log dir:        {args.log_dir}")

    print(f"\n[3/4] Warming up replay buffer (need {args.batch_size} transitions before training) ...")

    log_rows = []
    recent_improvements = deque(maxlen=args.log_interval)
    recent_rewards      = deque(maxlen=args.log_interval)
    total_steps = 0
    training_started = False

    print(f"\n[4/4] Starting training loop ({args.num_episodes} episodes) ...")
    for ep in range(args.num_episodes):
        G     = random.choice(train_graphs)
        if ep < 5:
            print(f"  Ep {ep}: resetting env (graph: {G.number_of_nodes()}n {G.number_of_edges()}e) ...", flush=True)
        state = env.reset(G)
        if ep < 5:
            print(f"  Ep {ep}: env reset done — initial crossings: {env.initial_crossings:.0f}", flush=True)
        ep_reward = 0.0
        loss_acc  = 0.0
        loss_cnt  = 0

        for _step in range(args.max_steps):
            action     = agent.select_action(state, training=True)
            node_idx   = action // env.num_dirs
            dir_idx    = action % env.num_dirs

            next_state, reward, done, info = env.step(action)

            if total_steps % 50 == 0:
                print(f"    step {_step+1}/{args.max_steps} | crossings: {info['crossings']:.0f} | reward: {reward:+.4f} | buffer: {len(buffer)}", flush=True)

            buffer.push(state, node_idx, dir_idx, reward, next_state, done)
            state      = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= args.batch_size:
                if not training_started:
                    print(f"  Buffer full — gradient updates starting at ep {ep}, step {total_steps}", flush=True)
                    training_started = True
                batch = buffer.sample(args.batch_size)
                loss  = agent.train_step(batch)
                loss_acc += loss
                loss_cnt += 1

            if done:
                break

        improvement = env.initial_crossings - env.current_crossings
        recent_improvements.append(improvement)
        recent_rewards.append(ep_reward)
        agent.decay_epsilon()

        log_rows.append({
            "episode":           ep,
            "initial_crossings": env.initial_crossings,
            "final_crossings":   env.current_crossings,
            "improvement":       improvement,
            "episode_reward":    ep_reward,
            "mean_loss":         loss_acc / max(1, loss_cnt),
            "epsilon":           agent.epsilon,
            "total_steps":       total_steps,
        })

        if (ep + 1) % args.log_interval == 0:
            avg_imp = np.mean(recent_improvements)
            avg_rew = np.mean(recent_rewards)
            print(
                f"Ep {ep+1:5d}/{args.num_episodes} | "
                f"improvement {avg_imp:+6.2f} | "
                f"reward {avg_rew:+7.3f} | "
                f"ε {agent.epsilon:.3f} | "
                f"steps {total_steps:7d}"
            )

        if (ep + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"dqn_ep{ep+1}.pt")
            agent.save(ckpt)
            print(f"  Saved checkpoint → {ckpt}")

    # Final save
    final_ckpt = os.path.join(args.checkpoint_dir, "dqn_final.pt")
    agent.save(final_ckpt)
    print(f"\nTraining complete. Final model → {final_ckpt}")

    pd.DataFrame(log_rows).to_csv(
        os.path.join(args.log_dir, "training_log.csv"), index=False
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train DQN for graph layout optimisation")

    # Data
    p.add_argument("--rome-dir",         default="rome")
    p.add_argument("--num-train",        type=int, default=10_000,
                   help="Number of graphs reserved for training (rest = test)")
    p.add_argument("--test-graphs-file", default="test_graphs.txt",
                   help="File to write held-out test graph filenames")

    # Environment
    p.add_argument("--step-size",    type=float, default=5.0,
                   help="Pixels to move a node per action")
    p.add_argument("--max-steps",    type=int,   default=50,
                   help="Max environment steps per episode")

    # Network
    p.add_argument("--hidden-dim",   type=int,   default=128)

    # RL hyperparameters
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--epsilon-start",    type=float, default=1.0)
    p.add_argument("--epsilon-end",      type=float, default=0.01)
    # 0.99954 ≈ 0.01^(1/10000): epsilon reaches 0.01 by episode 10000
    p.add_argument("--epsilon-decay",    type=float, default=0.99954)
    p.add_argument("--target-update-freq", type=int, default=200)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--buffer-size",      type=int,   default=50_000)

    # Training schedule
    p.add_argument("--num-episodes",  type=int, default=10_000)
    p.add_argument("--log-interval",  type=int, default=100)
    p.add_argument("--save-interval", type=int, default=1000)

    # Paths
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir",        default="logs")

    # Compute
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")

    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
