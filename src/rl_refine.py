"""
PPO-based refinement of graph layouts to minimize edge crossings.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import networkx as nx
from src.xing import XingLoss


def compute_node_features(G, coords, xing_hard, adj_t=None, edge_src=None, edge_dst=None):
    n = coords.shape[0]
    device = coords.device

    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    span = (cmax - cmin).clamp(min=1e-6)
    norm_coords = (coords - cmin) / span

    if adj_t is None:
        adj = nx.to_numpy_array(G)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    degrees = adj_t.sum(dim=1)
    max_deg = degrees.max().clamp(min=1.0)
    norm_deg = degrees / max_deg

    node_crossings = xing_hard.per_node_crossings(coords)
    max_xing = node_crossings.max().clamp(min=1.0)
    norm_xing = node_crossings / max_xing

    neighbor_count = degrees.clamp(min=1.0)
    neighbor_sum = adj_t @ norm_coords
    neighbor_mean = neighbor_sum / neighbor_count.unsqueeze(1) - norm_coords

    diff = norm_coords.unsqueeze(0) - norm_coords.unsqueeze(1)
    sq_diff = (diff ** 2) * adj_t.unsqueeze(2)
    neighbor_var = sq_diff.sum(dim=1) / neighbor_count.unsqueeze(1)
    neighbor_std = neighbor_var.sqrt()

    if edge_src is None or edge_dst is None:
        edges = list(G.edges())
        edge_src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
        edge_dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

    edge_vecs = norm_coords[edge_dst] - norm_coords[edge_src]
    edge_lens = edge_vecs.norm(dim=1)

    edge_len_sum = torch.zeros(n, device=device)
    edge_len_sum.scatter_add_(0, edge_src, edge_lens)
    edge_len_sum.scatter_add_(0, edge_dst, edge_lens)
    edge_count = torch.zeros(n, device=device)
    edge_count.scatter_add_(0, edge_src, torch.ones_like(edge_lens))
    edge_count.scatter_add_(0, edge_dst, torch.ones_like(edge_lens))
    edge_count = edge_count.clamp(min=1.0)
    mean_edge_len = edge_len_sum / edge_count

    neg_lens = -edge_lens
    neg_min_src = torch.full((n,), float('inf'), device=device)
    neg_min_src.scatter_reduce_(0, edge_src, neg_lens, reduce='amin')
    neg_min_dst = torch.full((n,), float('inf'), device=device)
    neg_min_dst.scatter_reduce_(0, edge_dst, neg_lens, reduce='amin')
    min_edge_len = -torch.min(neg_min_src, neg_min_dst)
    min_edge_len = torch.where(min_edge_len == float('-inf'), torch.zeros_like(min_edge_len), min_edge_len)

    max_mel = mean_edge_len.max().clamp(min=1e-6)
    mean_edge_len = mean_edge_len / max_mel
    min_edge_len = min_edge_len / max_mel

    crossing_density = node_crossings / degrees.clamp(min=1.0)
    max_cd = crossing_density.max().clamp(min=1e-6)
    crossing_density = crossing_density / max_cd

    neighbor_xing_sum = adj_t @ node_crossings
    neighbor_xing_mean = neighbor_xing_sum / neighbor_count
    max_nxm = neighbor_xing_mean.max().clamp(min=1e-6)
    neighbor_xing_mean = neighbor_xing_mean / max_nxm

    features = torch.cat([
        norm_coords,
        norm_deg.unsqueeze(1),
        norm_xing.unsqueeze(1),
        neighbor_mean,
        neighbor_std,
        mean_edge_len.unsqueeze(1),
        min_edge_len.unsqueeze(1),
        crossing_density.unsqueeze(1),
        neighbor_xing_mean.unsqueeze(1),
    ], dim=1)

    return features, node_crossings


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        return torch.tanh(self.linear(adj_norm @ x))


OBS_DIM = 12


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.gcn1 = GCNLayer(obs_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), init_log_std))

    def forward(self, obs, adj_norm):
        g1 = self.gcn1(obs, adj_norm)
        g2 = self.gcn2(g1, adj_norm)
        combined = torch.cat([g2, obs], dim=1)
        mean = self.mlp(combined)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128):
        super().__init__()
        self.gcn1 = GCNLayer(obs_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, adj_norm):
        g1 = self.gcn1(obs, adj_norm)
        g2 = self.gcn2(g1, adj_norm)
        combined = torch.cat([g2, obs], dim=1)
        pooled = combined.mean(dim=0, keepdim=True)
        return self.head(pooled).squeeze()


class PPORefiner:
    def __init__(self, G, init_coords, *, obs_dim=OBS_DIM, hidden=128, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef_start=0.05, entropy_coef_end=0.005,
                 value_coef=0.5, max_grad_norm=0.5, ppo_epochs=6,
                 step_scale_frac=0.01, log_std_start=-0.5, log_std_end=-2.0, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.xing_hard = XingLoss(G, soft=False, device=device)
        self.xing_soft = XingLoss(G, soft=True, device=device)
        self.init_coords = init_coords.clone().to(device)

        span = (init_coords.max(dim=0).values - init_coords.min(dim=0).values).clamp(min=1.0)
        self.step_scale = span.to(device) * step_scale_frac

        adj = nx.to_numpy_array(G)
        self.adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
        adj_hat = self.adj_t + torch.eye(self.n_nodes, device=device)
        deg_inv = 1.0 / adj_hat.sum(dim=1).clamp(min=1.0)
        self.adj_norm = deg_inv.unsqueeze(1) * adj_hat

        edges = list(G.edges())
        self.edge_src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
        self.edge_dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

        self.policy = PolicyNetwork(obs_dim, hidden, init_log_std=log_std_start).to(device)
        self.value_net = ValueNetwork(obs_dim, hidden).to(device)
        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.log_std_start = log_std_start
        self.log_std_end = log_std_end
        self.base_lr = lr

    def refine(self, n_episodes=200, steps_per_episode=40, verbose=False):
        best_coords = self.init_coords.clone()
        best_crossings = int(self.xing_hard(self.init_coords).item())
        init_crossings = best_crossings

        if best_crossings == 0:
            return best_coords, 0

        current_best_coords = self.init_coords.clone()
        stagnation = 0

        for ep in range(n_episodes):
            progress = ep / max(n_episodes - 1, 1)
            target_log_std = self.log_std_start + (self.log_std_end - self.log_std_start) * progress
            with torch.no_grad():
                self.policy.log_std.fill_(target_log_std)

            entropy_coef = self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * max(lr_mult, 0.1)

            step_mult = 1.0 + 1.0 * (1.0 - progress)
            current_step_scale = self.step_scale * step_mult

            trajectory = self._collect_episode(current_best_coords, best_crossings,
                                                steps_per_episode, step_scale=current_step_scale)

            if trajectory["best_crossings"] < best_crossings:
                best_crossings = trajectory["best_crossings"]
                best_coords = trajectory["best_coords"].clone()
                current_best_coords = best_coords.clone()
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > 0 and stagnation % 30 == 0:
                noise_scale = 0.02 * (1.0 - 0.5 * progress)
                span = (best_coords.max(dim=0).values - best_coords.min(dim=0).values).clamp(min=1.0)
                current_best_coords = best_coords.clone() + torch.randn_like(best_coords) * span * noise_scale

            self._ppo_update(trajectory, entropy_coef)

            if best_crossings == 0:
                break

        return best_coords, best_crossings

    def _collect_episode(self, start_coords, start_crossings, max_steps, step_scale=None):
        if step_scale is None:
            step_scale = self.step_scale
        coords = start_coords.clone()
        crossings = start_crossings
        obs_list, act_list, logp_list, rew_list, val_list = [], [], [], [], []
        best_crossings = crossings
        best_coords = coords.clone()

        for t in range(max_steps):
            obs, node_xings = compute_node_features(
                self.G, coords, self.xing_hard, self.adj_t, self.edge_src, self.edge_dst)
            with torch.no_grad():
                dist = self.policy(obs, self.adj_norm)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = self.value_net(obs, self.adj_norm)

            crossing_mask = (node_xings > 0).float().unsqueeze(1)
            masked_action = action * crossing_mask
            new_coords = coords + masked_action * step_scale.unsqueeze(0)
            new_crossings = int(self.xing_hard(new_coords).item())

            reward = float(crossings - new_crossings)
            with torch.no_grad():
                soft_old = self.xing_soft(coords).item()
                soft_new = self.xing_soft(new_coords).item()
                reward += 0.2 * max(-2.0, min(2.0, soft_old - soft_new))

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(log_prob)
            rew_list.append(reward)
            val_list.append(value)
            coords = new_coords
            crossings = new_crossings

            if crossings < best_crossings:
                best_crossings = crossings
                best_coords = coords.clone()
            if crossings == 0:
                break

        with torch.no_grad():
            last_obs, _ = compute_node_features(
                self.G, coords, self.xing_hard, self.adj_t, self.edge_src, self.edge_dst)
            last_val = self.value_net(last_obs, self.adj_norm)

        return {
            "obs": obs_list, "actions": act_list,
            "log_probs": torch.stack(logp_list) if logp_list else torch.tensor([], device=self.device),
            "rewards": torch.tensor(rew_list, device=self.device),
            "values": torch.stack(val_list) if val_list else torch.tensor([], device=self.device),
            "last_value": last_val,
            "best_crossings": best_crossings, "best_coords": best_coords,
            "final_crossings": crossings,
        }

    def _ppo_update(self, traj, entropy_coef):
        T = len(traj["obs"])
        if T == 0:
            return
        rewards = traj["rewards"]
        values = traj["values"].detach()
        old_log_probs = traj["log_probs"].detach()

        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = traj["last_value"].detach() if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            total_loss = 0.0
            for t in range(T):
                dist = self.policy(traj["obs"][t], self.adj_norm)
                new_log_prob = dist.log_prob(traj["actions"][t]).sum()
                entropy = dist.entropy().sum()
                new_value = self.value_net(traj["obs"][t], self.adj_norm)
                ratio = (new_log_prob - old_log_probs[t]).exp()
                surr1 = ratio * advantages[t]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[t]
                loss = -torch.min(surr1, surr2) + self.value_coef * (new_value - returns[t])**2 - entropy_coef * entropy
                total_loss = total_loss + loss
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm)
            self.optimizer.step()


def stochastic_local_search(G, coords, xing_hard, n_iters=200, n_candidates=8, device=None):
    if device is None:
        device = torch.device("cpu")
    best_coords = coords.clone()
    best_crossings = int(xing_hard(coords).item())
    if best_crossings == 0:
        return best_coords, 0

    xing_soft = XingLoss(G, soft=True, device=device)
    span = (coords.max(dim=0).values - coords.min(dim=0).values).clamp(min=1.0)
    current_coords = coords.clone()
    current_crossings = best_crossings

    for it in range(n_iters):
        node_xings = xing_hard.per_node_crossings(current_coords)
        crossing_nodes = torch.where(node_xings > 0)[0]
        if len(crossing_nodes) == 0:
            break

        weights = node_xings[crossing_nodes]
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, 1).item()
        node = crossing_nodes[idx].item()

        scale = span * 0.02 * (1.0 - 0.5 * it / n_iters)
        for _ in range(n_candidates):
            delta = torch.randn(2, device=device) * scale
            trial_coords = current_coords.clone()
            trial_coords[node] += delta
            trial_crossings = int(xing_hard(trial_coords).item())
            if trial_crossings < current_crossings:
                current_coords = trial_coords
                current_crossings = trial_crossings
                break
            elif trial_crossings == current_crossings:
                with torch.no_grad():
                    if xing_soft(trial_coords).item() < xing_soft(current_coords).item():
                        current_coords = trial_coords
                        break

        if current_crossings < best_crossings:
            best_crossings = current_crossings
            best_coords = current_coords.clone()

    return best_coords, best_crossings


def ppo_refine(G, init_coords, n_episodes=200, steps_per_episode=40,
               n_seeds=3, verbose=False, device=None):
    if device is None:
        device = torch.device("cpu")

    xing_hard = XingLoss(G, soft=False, device=device)
    init_crossings = int(xing_hard(init_coords.to(device)).item())
    if init_crossings == 0:
        return init_coords, 0

    sls_coords, sls_crossings = stochastic_local_search(
        G, init_coords.to(device), xing_hard, n_iters=500, n_candidates=16, device=device)

    if verbose:
        print(f"  SLS phase 1: {init_crossings} -> {sls_crossings}")
    if sls_crossings == 0:
        return sls_coords, 0

    if sls_crossings <= 3:
        final_coords, final_crossings = stochastic_local_search(
            G, sls_coords, xing_hard, n_iters=500, n_candidates=20, device=device)
        best = min((sls_crossings, sls_coords), (final_crossings, final_coords), key=lambda x: x[0])
        return best[1], best[0]

    adjusted_episodes = int(n_episodes * 1.5) if sls_crossings >= 30 else n_episodes
    best_ppo_coords = sls_coords
    best_ppo_crossings = sls_crossings
    episodes_per_seed = adjusted_episodes // n_seeds

    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + 42)
        refiner = PPORefiner(G, sls_coords, device=device)
        ppo_coords, ppo_crossings = refiner.refine(episodes_per_seed, steps_per_episode, verbose=verbose)
        if ppo_crossings < best_ppo_crossings:
            best_ppo_crossings = ppo_crossings
            best_ppo_coords = ppo_coords
        if verbose:
            print(f"  PPO seed {seed}: {sls_crossings} -> {ppo_crossings}")
        if best_ppo_crossings == 0:
            break

    final_coords, final_crossings = stochastic_local_search(
        G, best_ppo_coords, xing_hard, n_iters=400, n_candidates=16, device=device)

    best = min(
        (sls_crossings, sls_coords),
        (best_ppo_crossings, best_ppo_coords),
        (final_crossings, final_coords),
        key=lambda x: x[0],
    )
    return best[1], best[0]
