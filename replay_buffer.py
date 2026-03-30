"""
Replay buffer for DQN.

Each transition stores:
    features      – numpy [n, feat_dim]   (current state, all nodes)
    node_idx      – int                   (which node was moved)
    dir_idx       – int                   (which direction was chosen)
    reward        – float
    next_features – numpy [n, feat_dim]   (next state, all nodes)
    done          – bool

Because graphs can have different sizes, transitions are stored as a list
rather than pre-allocated arrays.  Sampling returns raw tuples; the agent
is responsible for moving data to the appropriate device.
"""

import random
from collections import deque
from typing import List, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        features: np.ndarray,
        node_idx: int,
        dir_idx: int,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((features, node_idx, dir_idx, reward, next_features, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
