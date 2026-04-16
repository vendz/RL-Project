"""
PPO training loop for graph layout optimization.

Usage:
    python train.py --cache_dir cache --split train --episodes 50000 --out checkpoints/

On HPC, set --workers to number of available CPU cores.
"""

import os
import glob
import argparse
import random
from collections import deque
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GraphLayoutEnv
from src.policy import GNNPolicy


# ------------------------------------------------------------------
# PPO hyperparameters (defaults tuned for this problem)
# ------------------------------------------------------------------
DEFAULTS = dict(
    hidden_dim=128,
    lr=3e-4,
    gamma=0.99,          # discount factor
    gae_lambda=0.95,     # GAE lambda
    clip_eps=0.2,        # PPO clip epsilon
    entropy_coef=0.01,   # entropy bonus
    value_coef=0.5,      # value loss weight
    ppo_epochs=4,        # gradient updates per batch
    batch_size=256,      # transitions per update
    max_grad_norm=0.5,
)


# ------------------------------------------------------------------
# Rollout buffer
# ------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.coords      = []
        self.edge_index  = []
        self.node_idxs   = []
        self.disps       = []   # [dx, dy]
        self.log_probs   = []
        self.values      = []
        self.rewards     = []
        self.dones       = []

    def add(self, coords, edge_index, node_idx, disp, log_prob, value, reward, done):
        self.coords.append(coords.clone())
        self.edge_index.append(edge_index.clone())
        self.node_idxs.append(node_idx)
        self.disps.append(disp)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae = delta + gamma * gae_lambda * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ------------------------------------------------------------------
# PPO update
# ------------------------------------------------------------------

def ppo_update(policy, optimizer, buffer, args):
    advantages, returns = compute_gae(
        buffer.rewards, [v.item() for v in buffer.values],
        buffer.dones, args.gamma, args.gae_lambda
    )
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.tensor(returns, dtype=torch.float32)
    old_log_probs = torch.stack(buffer.log_probs).detach()

    total_loss = 0.0
    for _ in range(args.ppo_epochs):
        idx = list(range(len(buffer)))
        random.shuffle(idx)
        for start in range(0, len(idx), args.batch_size):
            batch = idx[start:start + args.batch_size]
            batch_loss = torch.tensor(0.0)
            for i in batch:
                coords     = buffer.coords[i]
                edge_index = buffer.edge_index[i]
                node_idx   = buffer.node_idxs[i]
                disp       = buffer.disps[i]  # [2]

                node_logits, mu, log_std, value = policy(coords, edge_index)

                # Node log prob
                node_dist = torch.distributions.Categorical(logits=node_logits)
                node_lp = node_dist.log_prob(torch.tensor(node_idx))
                node_entropy = node_dist.entropy()

                # Displacement log prob
                std = log_std[node_idx].exp()
                disp_dist = torch.distributions.Normal(mu[node_idx], std)
                disp_lp = disp_dist.log_prob(disp).sum()
                disp_entropy = disp_dist.entropy().sum()

                new_log_prob = node_lp + disp_lp

                # PPO clipped objective
                ratio = (new_log_prob - old_log_probs[i]).exp()
                adv = advantages[i]
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                )

                value_loss = (value - returns[i]).pow(2)
                entropy = node_entropy + disp_entropy

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += batch_loss.item()

    return total_loss


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(args):
    # Load graph IDs for the requested split
    cache_graphs = glob.glob(os.path.join(args.cache_dir, "graphs", "*.pkl"))
    all_ids = [os.path.basename(p).replace(".pkl", "") for p in cache_graphs]

    def graph_number(gid):
        try:
            return int(gid.split(".")[0].replace("grafo", ""))
        except ValueError:
            return -1

    if args.split == "train":
        graph_ids = [g for g in all_ids if graph_number(g) <= 9999]
    elif args.split == "test":
        graph_ids = [g for g in all_ids if 10000 <= graph_number(g) <= 10100]
    else:
        graph_ids = all_ids

    if not graph_ids:
        raise RuntimeError(f"No graphs found in {args.cache_dir} for split={args.split}. Run precompute.py first.")
    print(f"Training on {len(graph_ids)} graphs.")

    env = GraphLayoutEnv(graph_ids, cache_dir=args.cache_dir, patience=args.patience)
    policy = GNNPolicy(hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_avg_reward = -float("inf")
    episode_rewards = deque(maxlen=100)
    episode_crossings = deque(maxlen=100)
    buffer = RolloutBuffer()

    obs, _ = env.reset(seed=0)

    for episode in trange(1, args.episodes + 1):
        coords     = torch.tensor(obs["coords"], dtype=torch.float32)
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        ep_reward  = 0.0
        done = False

        while not done:
            with torch.no_grad():
                node_idx, dx, dy, log_prob, value = policy.act(coords, edge_index)

            disp = torch.tensor([dx, dy], dtype=torch.float32)
            obs, reward, terminated, truncated, _ = env.step_with_node(node_idx, dx, dy)
            done = terminated or truncated

            buffer.add(coords, edge_index, node_idx, disp, log_prob, value, reward, float(done))
            ep_reward += reward

            coords     = torch.tensor(obs["coords"], dtype=torch.float32)
            edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)

            if len(buffer) >= args.batch_size:
                ppo_update(policy, optimizer, buffer, args)
                buffer.clear()

        episode_rewards.append(ep_reward)
        episode_crossings.append(env.current_crossings)
        obs, _ = env.reset()

        if episode % args.log_interval == 0:
            avg_r  = np.mean(episode_rewards)
            avg_xing = np.mean(episode_crossings)
            print(f"Ep {episode:6d} | avg_reward={avg_r:+.2f} | avg_crossings={avg_xing:.1f}")

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                torch.save(policy.state_dict(), os.path.join(args.out, "best.pt"))

    torch.save(policy.state_dict(), os.path.join(args.out, "final.pt"))
    print(f"Training complete. Checkpoints saved to {args.out}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for graph layout")
    parser.add_argument("--cache_dir",    default="cache")
    parser.add_argument("--split",        default="train", choices=["train", "test", "all"])
    parser.add_argument("--episodes",     type=int,   default=50000)
    parser.add_argument("--patience",     type=int,   default=50,   help="Steps without improvement before episode ends")
    parser.add_argument("--out",          default="checkpoints")
    parser.add_argument("--log_interval", type=int,   default=500)
    # Policy
    parser.add_argument("--hidden_dim",   type=int,   default=DEFAULTS["hidden_dim"])
    # PPO
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--gamma",        type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--gae_lambda",   type=float, default=DEFAULTS["gae_lambda"])
    parser.add_argument("--clip_eps",     type=float, default=DEFAULTS["clip_eps"])
    parser.add_argument("--entropy_coef", type=float, default=DEFAULTS["entropy_coef"])
    parser.add_argument("--value_coef",   type=float, default=DEFAULTS["value_coef"])
    parser.add_argument("--ppo_epochs",   type=int,   default=DEFAULTS["ppo_epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--max_grad_norm",type=float, default=DEFAULTS["max_grad_norm"])

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
