"""
GraphLayoutEnv: RL environment for graph layout optimization.

State:  per-node feature matrix [n, NODE_FEAT_DIM]
Action: integer  node_idx * NUM_DIRS + dir_idx
Reward: reduction in edge crossings after moving a node one step
"""

import numpy as np
import torch
import networkx as nx

from xing import XingLoss

NUM_DIRS = 8
# Unit-length direction vectors (E, W, N, S, NE, NW, SE, SW)
_dirs = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, 1], [1, -1], [-1, -1],
], dtype=np.float32)
DIRECTIONS = _dirs / np.linalg.norm(_dirs, axis=1, keepdims=True)  # [8, 2]

NODE_FEAT_DIM = 7  # [norm_x, norm_y, neigh_mean_x, neigh_mean_y, deg_norm, bb_x, bb_y]


class GraphLayoutEnv:
    """
    Discrete-action RL environment for minimising edge crossings.

    Action space: NUM_DIRS directions × num_nodes nodes
                  encoded as a single int: node_idx * NUM_DIRS + dir_idx
    State:        numpy array [n, NODE_FEAT_DIM]
    Reward:       (crossings_before – crossings_after) / max(1, initial_crossings)
                  shaped so improvements are +, regressions are -.
    """

    def __init__(self, step_size: float = 5.0, max_steps: int = 200):
        self.step_size = step_size
        self.max_steps = max_steps
        self.num_dirs = NUM_DIRS
        self.directions = DIRECTIONS  # [8, 2] numpy

        # Set in reset()
        self.G = None
        self.coords = None          # [n, 2] torch float32, always CPU
        self.xing_loss = None
        self.adj = None             # [n, n] dense torch float32
        self.degree = None          # [n, 1]
        self.max_degree = None      # scalar
        self.current_crossings = 0.0
        self.initial_crossings = 0.0
        self.current_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, G: nx.Graph, init_coords: np.ndarray = None) -> np.ndarray:
        """
        Reset with a new graph.

        Args:
            G:           NetworkX graph with integer node labels 0..n-1.
            init_coords: Optional [n, 2] numpy array of initial positions.
                         If None, uses Graphviz neato layout.

        Returns:
            state: numpy [n, NODE_FEAT_DIM]
        """
        self.G = G
        self.current_step = 0
        n = G.number_of_nodes()

        # Crossing loss (always CPU — xing.py stores edges as CPU tensors)
        self.xing_loss = XingLoss(G, device=None, soft=False)

        # Precompute dense adjacency & degree
        adj = torch.zeros(n, n, dtype=torch.float32)
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        self.adj = adj
        self.degree = adj.sum(dim=1, keepdim=True)          # [n, 1]
        self.max_degree = self.degree.max().clamp(min=1.0)  # scalar

        # Initialise coordinates
        if init_coords is not None:
            self.coords = torch.tensor(init_coords, dtype=torch.float32)
        else:
            pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
            nodes = sorted(G.nodes())
            self.coords = torch.tensor(
                [[pos[v][0], pos[v][1]] for v in nodes], dtype=torch.float32
            )

        with torch.no_grad():
            self.current_crossings = self.xing_loss(self.coords).item()
        self.initial_crossings = self.current_crossings

        return self._compute_features(self.coords)

    def step(self, action: int):
        """
        Execute one action.

        Args:
            action: int in [0, num_nodes * NUM_DIRS)

        Returns:
            next_state: numpy [n, NODE_FEAT_DIM]
            reward:     float
            done:       bool
            info:       dict with 'crossings'
        """
        n = self.coords.shape[0]
        node_idx = action // self.num_dirs
        dir_idx  = action % self.num_dirs

        delta = torch.tensor(self.directions[dir_idx] * self.step_size, dtype=torch.float32)
        new_coords = self.coords.clone()
        new_coords[node_idx] += delta

        with torch.no_grad():
            new_crossings = self.xing_loss(new_coords).item()

        # Normalised reward: improvement relative to initial crossing count
        denom = max(1.0, self.initial_crossings)
        reward = (self.current_crossings - new_crossings) / denom

        self.coords = new_coords
        self.current_crossings = new_crossings
        self.current_step += 1

        done = self.current_step >= self.max_steps
        next_state = self._compute_features(self.coords)
        return next_state, reward, done, {"crossings": new_crossings}

    @property
    def num_nodes(self) -> int:
        return self.coords.shape[0] if self.coords is not None else 0

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, coords: torch.Tensor) -> np.ndarray:
        """
        Compute per-node feature matrix.

        Features per node (NODE_FEAT_DIM = 7):
            [0-1] norm_x, norm_y          — z-score normalised position
            [2-3] neigh_mean_x, neigh_y   — mean z-score pos of graph-neighbours
            [4]   deg_norm                — degree / max_degree
            [5-6] bb_x, bb_y              — position in bounding box [0, 1]
        """
        # z-score normalise
        mean_c = coords.mean(dim=0, keepdim=True)
        std_c  = coords.std(dim=0, keepdim=True).clamp(min=1e-6)
        norm   = (coords - mean_c) / std_c             # [n, 2]

        # Mean neighbour z-score position
        neigh_sum  = self.adj @ norm                   # [n, 2]
        neigh_mean = neigh_sum / self.degree.clamp(min=1.0)  # [n, 2]

        # Normalised degree
        deg_norm = self.degree / self.max_degree       # [n, 1]

        # Bounding-box position ∈ [0, 1]
        cmin   = coords.min(dim=0).values
        cmax   = coords.max(dim=0).values
        crange = (cmax - cmin).clamp(min=1e-6)
        bb     = (coords - cmin) / crange              # [n, 2]

        features = torch.cat([norm, neigh_mean, deg_norm, bb], dim=1)  # [n, 7]
        return features.numpy()
