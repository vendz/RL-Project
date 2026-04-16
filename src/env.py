"""
Gymnasium environment for graph layout optimization.

One episode = one graph. The agent iteratively moves nodes to reduce edge crossings.

Observation:
    Dict with:
      - "coords":     float32 [N, 2]  current node positions
      - "edge_index": int64   [2, E]  graph connectivity (both directions)
      - "num_nodes":  int64   scalar

Action:
    float32 [3] = [node_score_index (unused at env level), dx, dy]
    The policy selects a node externally via GNN scores; the env receives
    (node_idx, dx, dy) via step().

Reward:
    crossings_before - crossings_after  (positive = improvement)

Done:
    Hard crossings unchanged for `patience` consecutive steps, or max_steps reached.
"""

import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.xing import XingLoss

CACHE_DIR = "cache"


class GraphLayoutEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graph_ids, cache_dir=CACHE_DIR, max_steps=None, patience=50):
        """
        graph_ids: list of graph id strings to sample episodes from
        cache_dir: path to precomputed cache
        max_steps: max steps per episode; defaults to 10 * num_nodes
        patience:  stop if crossings don't improve for this many steps
        """
        super().__init__()
        self.graph_ids = graph_ids
        self.cache_dir = cache_dir
        self.max_steps = max_steps
        self.patience = patience

        # Placeholders — set on reset()
        self.G = None
        self.xing_loss = None
        self.coords = None          # [N, 2] torch tensor (current layout)
        self.edge_index = None      # [2, E] torch tensor
        self.current_crossings = 0
        self.steps = 0
        self.no_improve_steps = 0
        self._episode_max_steps = 0

        # Observation and action spaces are defined dynamically per graph.
        # We use a large fixed bound here for compatibility with SB3.
        # Actual shapes depend on num_nodes — handled in reset().
        self.observation_space = spaces.Dict({
            "coords":     spaces.Box(-np.inf, np.inf, shape=(1, 2), dtype=np.float32),
            "edge_index": spaces.Box(0, 10000,        shape=(2, 1), dtype=np.int64),
            "num_nodes":  spaces.Box(0, 10000,        shape=(1,),   dtype=np.int64),
        })
        # Action: (node_idx as float, dx, dy)
        # node_idx is clipped to valid range in step()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -50.0, -50.0], dtype=np.float32),
            high=np.array([1.0,  50.0,  50.0], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        graph_id = self.np_random.choice(self.graph_ids)
        self._load_graph(graph_id)

        self.steps = 0
        self.no_improve_steps = 0
        self._episode_max_steps = (
            self.max_steps if self.max_steps is not None
            else 10 * self.G.number_of_nodes()
        )
        with torch.no_grad():
            self.current_crossings = int(self.xing_loss(self.coords).item())

        return self._obs(), {}

    def step(self, action):
        """
        action: array [3] = [node_frac, dx, dy]
          node_frac in [-1, 1] is mapped to a node index in [0, N-1]
        """
        node_frac, dx, dy = float(action[0]), float(action[1]), float(action[2])
        n = self.coords.shape[0]
        node_idx = int(((node_frac + 1) / 2) * n) % n  # map [-1,1] -> [0, N-1]

        prev_crossings = self.current_crossings
        self.coords[node_idx, 0] += dx
        self.coords[node_idx, 1] += dy

        with torch.no_grad():
            self.current_crossings = int(self.xing_loss(self.coords).item())

        reward = float(prev_crossings - self.current_crossings)

        if self.current_crossings < prev_crossings:
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        self.steps += 1
        terminated = self.no_improve_steps >= self.patience
        truncated = self.steps >= self._episode_max_steps

        return self._obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def step_with_node(self, node_idx: int, dx: float, dy: float):
        """
        Convenience method for GNN-based policies that select node directly.
        Used during manual testing / custom training loops.
        """
        action = np.array([
            (node_idx / max(self.coords.shape[0] - 1, 1)) * 2 - 1,
            dx,
            dy,
        ], dtype=np.float32)
        return self.step(action)

    def _load_graph(self, graph_id):
        graph_path = os.path.join(self.cache_dir, "graphs", f"{graph_id}.pkl")
        diffgd_path = os.path.join(self.cache_dir, "diffgd", f"{graph_id}.pt")
        edge_index_path = os.path.join(self.cache_dir, "edge_index", f"{graph_id}.pt")

        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)
        self.coords = torch.load(diffgd_path, weights_only=True).clone()
        self.edge_index = torch.load(edge_index_path, weights_only=True)
        self.xing_loss = XingLoss(self.G, soft=False)

    def _obs(self):
        return {
            "coords":     self.coords.numpy(),
            "edge_index": self.edge_index.numpy(),
            "num_nodes":  np.array([self.coords.shape[0]], dtype=np.int64),
        }
