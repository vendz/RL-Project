"""
DQN Agent for graph layout optimisation.

Key design choices:
  - Shared per-node Q-network: same MLP applied to every node's features.
    This naturally handles variable-size graphs.
  - Action selection: argmax over all (node, direction) Q-values.
  - Target network updated every `target_update_freq` gradient steps.
  - Epsilon-greedy exploration with multiplicative decay.
  - Gradient updates process each transition in a batch independently
    (graphs differ in size), then average the losses before stepping.
"""

import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from q_network import QNetwork
from replay_buffer import Transition


class DQNAgent:
    def __init__(
        self,
        feat_dim: int = 7,
        num_dirs: int = 8,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        target_update_freq: int = 200,
        device: torch.device = None,
    ):
        self.num_dirs = num_dirs
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device or torch.device("cpu")
        self._update_count = 0

        self.q_net    = QNetwork(feat_dim, num_dirs, hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, features: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            features: numpy [n, feat_dim]
            training: if False, greedy (epsilon = 0)

        Returns:
            action int = node_idx * num_dirs + dir_idx
        """
        n = features.shape[0]
        if training and np.random.rand() < self.epsilon:
            node_idx = np.random.randint(n)
            dir_idx  = np.random.randint(self.num_dirs)
        else:
            feat_t = torch.tensor(features, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_vals = self.q_net(feat_t)          # [n, num_dirs]
            flat_idx = q_vals.view(-1).argmax().item()
            node_idx = flat_idx // self.num_dirs
            dir_idx  = flat_idx % self.num_dirs

        return node_idx * self.num_dirs + dir_idx

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: List[Transition]) -> float:
        """
        One gradient update from a sampled batch.

        For each transition we compute:
            current_Q = Q(features[node_idx], dir_idx)
            target_Q  = reward + gamma * max_{n,d} Q_target(next_features)

        Because transitions may come from different graphs (variable n),
        we accumulate loss over the batch and step once.

        Returns:
            mean loss (float)
        """
        self.q_net.train()
        total_loss = torch.zeros(1, device=self.device)

        for features_np, node_idx, dir_idx, reward, next_features_np, done in batch:
            feat      = torch.tensor(features_np,      dtype=torch.float32, device=self.device)
            next_feat = torch.tensor(next_features_np, dtype=torch.float32, device=self.device)

            # Q-value for the action taken
            q_vals      = self.q_net(feat)            # [n, num_dirs]
            current_q   = q_vals[node_idx, dir_idx]   # scalar

            # Bellman target
            with torch.no_grad():
                next_q_vals = self.q_target(next_feat) # [n, num_dirs]
                max_next_q  = next_q_vals.max()
                target_q    = reward + (1.0 - float(done)) * self.gamma * max_next_q

            total_loss = total_loss + F.smooth_l1_loss(current_q, target_q)

        loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net":        self.q_net.state_dict(),
                "q_target":     self.q_target.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "epsilon":      self.epsilon,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon      = ckpt["epsilon"]
        self._update_count = ckpt["update_count"]
