# Design

## Problem

Minimize edge crossings in 2D graph layouts. Edge crossing count is discrete and non-differentiable, so gradient descent cannot directly optimize it. We use RL to optimize it via reward signals.

**Dataset:** Rome graph collection. Train on graphs ≤9999, test on graphs 10000–10100.

---

## Pipeline

```
Rome Graph → neato → Multi-restart Diff GD → SLS → PPO Refinement → SLS → Final Layout
```

**Multi-restart Diff GD warm-start:** Run Adam on soft `XingLoss + 0.2 * StressLoss` for 300 epochs from 7 starting points (neato, sfdp, 5 random perturbations). Take the best hard crossing count across all restarts.

**Rationale:** Diff GD is the best deterministic baseline (−37.18% SPC vs neato). Starting RL from its output means the agent only handles what gradient descent can't.

See [`approach.md`](approach.md) for full pipeline documentation.

---

## Reward

$$R_{t+1} = (\text{crossings}_t - \text{crossings}_{t+1}) + 0.2 \cdot (\text{SoftLoss}_t - \text{SoftLoss}_{t+1})$$

Hard crossing improvement is primary; soft loss provides dense signal when hard crossings don't change.

---

## Evaluation

Metric: **SPC vs neato** on test set (graphs 10000–10100). See [`metrics.md`](metrics.md).

| Method | Mean Crossings | SPC vs Neato |
|--------|---------------|-------------|
| Neato | 29.1 | 0% (baseline) |
| SFDP | — | −3.14% |
| SmartGD | — | −3.60% |
| Diff GD | 18.0 | −37.18% |
| **PPO Refinement (ours)** | **15.9** | **−44.04%** |

---

## Submission Approach: Per-graph PPO + SLS

The submitted approach uses `rl_refine.py` — per-graph RL with no generalization requirement. For each test graph:
1. Multi-restart Diff GD warm-start
2. Stochastic Local Search (SLS) — fast greedy refinement
3. Multi-seed PPO with GCN policy
4. Final SLS pass

Results: `rl_results.csv` (99 test graphs).

---

## Future Work: Amortized GCN Policy

A generalizable GCN policy (train once, infer on new graphs) was explored but not completed due to training speed constraints. The architecture (`policy.py`, `env.py`, `train.py`) is implemented but requires a batched PPO update loop for viable training time (~40h unoptimized). Parked for future work.

---

## Related Work

| Paper | Relevance |
|-------|-----------|
| [arxiv:2509.06108](https://arxiv.org/abs/2509.06108) — RL for Crossing Number (GD'25) | Same problem. Uses PPO, 16 discrete actions per node, neato init. Competitive only on *local* crossing number — global remains unsolved. Our gap to exploit. |
| [arxiv:2206.06434](https://arxiv.org/abs/2206.06434) — SmartGD | GAN-based baseline in our evaluation. |
| [arxiv:2106.15347](https://arxiv.org/abs/2106.15347) — DeepGD | GNN-based graph drawing; shows GNNs generalize across graphs. |
| [NeuLay](https://www.nature.com/articles/s41467-023-37189-2) — Nature Comms 2023 | 2-layer GCN for direct per-node position prediction. |
| [Google Chip Placement](https://www.nature.com/articles/s41586-021-03544-w) — Nature 2021 | Edge-GCN + PPO for node placement. Most validated real-world GNN+PPO example. |

**Key differentiator vs arxiv:2509.06108:** Diff GD warm-start, multi-restart strategy, SLS+PPO combination, hybrid reward.
