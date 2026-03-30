"""
Q-Network for graph layout optimization.

Architecture: shared per-node MLP
  Input:  node feature vector  [feat_dim]
  Output: Q-values for each direction  [num_dirs]

Because the network operates on individual node features, it handles
graphs of any size without modification.
"""

import torch
import torch.nn as nn

from env import NODE_FEAT_DIM, NUM_DIRS


class QNetwork(nn.Module):
    """
    Per-node MLP that maps node features → Q-values for each move direction.

    Usage:
        q_vals = net(features)   # features: [batch, feat_dim] → [batch, num_dirs]
    """

    def __init__(
        self,
        feat_dim: int = NODE_FEAT_DIM,
        num_dirs: int = NUM_DIRS,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_dirs = num_dirs

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_dirs),
        )

        # Initialise output layer near zero so early Q-values are small
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, feat_dim]  →  [*, num_dirs]"""
        return self.net(x)
