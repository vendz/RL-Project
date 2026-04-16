"""
GNN policy network for graph layout RL.

Architecture:
  - 2-layer GCN encoder: node features [x, y, degree] -> 128-dim embeddings
  - Score head:          per-node scalar -> softmax -> node selection distribution
  - Displacement head:   per-node (mu_x, mu_y, log_std_x, log_std_y) -> Gaussian action
  - Value network:       global mean-pool of embeddings -> MLP -> scalar

Forward returns a named tuple with:
  node_logits   [N]        log-probabilities over nodes
  mu            [N, 2]     displacement mean per node
  log_std       [N, 2]     displacement log-std per node
  value         scalar     state value estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNNPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Encoder: node features (x, y, degree) -> hidden_dim
        self.conv1 = GCNConv(3, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Score head: selects which node to move
        self.score_head = nn.Linear(hidden_dim, 1)

        # Displacement head: predicts (mu, log_std) for (dx, dy)
        self.mu_head     = nn.Linear(hidden_dim, 2)
        self.log_std_head = nn.Linear(hidden_dim, 2)

        # Value network: global mean-pool -> scalar
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        coords:     [N, 2]  node (x, y) positions
        edge_index: [2, E]  graph edges (both directions)
        Returns:    [N, hidden_dim] node embeddings
        """
        # Node features: x, y, degree (normalized)
        degree = torch.zeros(coords.shape[0], dtype=torch.float32, device=coords.device)
        if edge_index.shape[1] > 0:
            degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=coords.device))
        degree = degree / (degree.max().clamp(min=1.0))  # normalize to [0, 1]
        x = torch.cat([coords, degree.unsqueeze(1)], dim=1)  # [N, 3]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x  # [N, hidden_dim]

    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns node_logits [N], mu [N,2], log_std [N,2], value scalar.
        """
        emb = self.encode(coords, edge_index)          # [N, hidden_dim]

        node_logits = self.score_head(emb).squeeze(-1) # [N]
        node_logits = F.log_softmax(node_logits, dim=0)

        mu      = self.mu_head(emb)                    # [N, 2]
        log_std = self.log_std_head(emb).clamp(-4, 2)  # [N, 2]

        global_emb = emb.mean(dim=0)                   # [hidden_dim]
        value = self.value_mlp(global_emb).squeeze()   # scalar

        return node_logits, mu, log_std, value

    def act(self, coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Sample an action: select a node, then sample (dx, dy).
        Returns:
          node_idx    int
          dx, dy      float
          log_prob    tensor  (for PPO)
          value       tensor  (for PPO)
        """
        node_logits, mu, log_std, value = self.forward(coords, edge_index)

        # Sample node
        node_dist = torch.distributions.Categorical(logits=node_logits)
        node_idx = node_dist.sample()

        # Sample displacement for selected node
        std = log_std[node_idx].exp()
        disp_dist = torch.distributions.Normal(mu[node_idx], std)
        displacement = disp_dist.sample()

        log_prob = node_dist.log_prob(node_idx) + disp_dist.log_prob(displacement).sum()

        return node_idx.item(), displacement[0].item(), displacement[1].item(), log_prob, value
