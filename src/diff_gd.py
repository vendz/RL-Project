"""
Differentiable Crossing Loss + Gradient Descent baseline.
Reusable function extracted from notebook.
"""

import networkx as nx
import torch
from torch.optim import Adam
from src.xing import XingLoss
from src.stress import StressLoss


def run_diff_gd(G, epochs=300, lr=2.0, stress_weight=0.2, prog="neato", device=None):
    if device is None:
        device = torch.device("cpu")

    pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()],
        dtype=torch.float32, device=device, requires_grad=True,
    )

    xing_soft = XingLoss(G, soft=True, device=device)
    stress = StressLoss(G, device=device)
    xing_hard = XingLoss(G, soft=False, device=device)
    optimizer = Adam([coords], lr=lr)

    best_coords = coords.detach().clone()
    best_crossings = int(xing_hard(coords).item())

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = xing_soft(coords) + stress_weight * stress(coords)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                hx = int(xing_hard(coords).item())
                if hx < best_crossings:
                    best_crossings = hx
                    best_coords = coords.detach().clone()

    return best_coords, best_crossings


def get_neato_crossings(G, device=None):
    if device is None:
        device = torch.device("cpu")
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()],
        dtype=torch.float32, device=device,
    )
    xing_hard = XingLoss(G, soft=False, device=device)
    return int(xing_hard(coords).item()), coords
