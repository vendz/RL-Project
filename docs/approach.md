# RL for Graph Layout: Approach & Results

## Problem

Given a graph $G = (V, E)$, assign 2D coordinates to each node to **minimize edge crossings**.

Edge crossing count is **discrete and non-differentiable** — gradient descent cannot optimize it directly. We use RL to bridge this gap.

**Dataset:** Rome graph collection (~11,500 graphs). Test set: graphs 10000–10100 (99 graphs).

---

## Pipeline Overview

```
Graph
  ↓
neato layout (initial positions)
  ↓
Multi-restart Diff GD  ──────── differentiable warm-start
  ↓
Stochastic Local Search (SLS)  ─ fast greedy refinement
  ↓
PPO Refinement (GCN policy)  ── RL for hard crossing reduction
  ↓
Final SLS pass  ───────────────  squeeze out last crossings
  ↓
Best layout across all phases
```

---

## Stage 1: Multi-restart Diff GD

**What:** Optimize node coordinates via gradient descent on a differentiable proxy loss:

$$\mathcal{L} = \text{SoftXingLoss}(X) + 0.2 \cdot \text{StressLoss}(X)$$

- **SoftXingLoss:** sigmoid-smoothed approximation of edge crossing count
- **StressLoss:** penalizes deviation from graph-theoretic distances (preserves layout aesthetics)

**Multi-restart strategy:** Run 300 epochs of Adam (lr=2.0) from 7 different starting points:
- neato layout
- sfdp layout
- 5 random perturbations of the best result so far

Take the best hard crossing count across all restarts.

**Why:** Diff GD is cheap, reliable, and gets close to optimal under the differentiable proxy. The multi-restart combats local minima.

---

## Stage 2: Stochastic Local Search (SLS)

**What:** A fast, greedy local search that directly optimizes hard crossing count.

For each iteration:
1. Compute per-node crossing count (how many crossings involve each node's edges)
2. Sample a node proportional to its crossing count — nodes causing more crossings are picked more often
3. Try $k=16$ random displacement candidates for that node
4. Accept the first move that reduces crossings; on ties, prefer the one with lower soft loss

Run for 500 iterations before PPO, 400 iterations after.

**Why:** SLS is extremely effective for this problem. It focuses effort on problematic nodes and makes fast, targeted improvements. Much of the gain over Diff GD comes from this stage.

---

## Stage 3: PPO Refinement

Applied only when SLS leaves $\geq 4$ crossings remaining (easy graphs skip PPO).

### Policy Network (GCN)

Input: 12 features per node:

| Feature | Description |
|---------|-------------|
| x, y | Normalized coordinates |
| degree | Normalized node degree |
| crossing count | Normalized per-node crossings |
| neighbor mean | Mean relative neighbor position (x, y) |
| neighbor std | Std of neighbor positions (x, y) |
| mean edge length | Mean length of incident edges |
| min edge length | Min length of incident edges |
| crossing density | Crossings per degree |
| neighbor crossing mean | Avg crossing count of neighbors |

Architecture:
```
Node features [N, 12]
      ↓
GCNLayer(12 → 128) + tanh   ← sees graph structure
      ↓
GCNLayer(128 → 128) + tanh
      ↓
Concat [GCN output, raw features] [N, 140]
      ↓
MLP(140 → 128 → 2)  → displacement (dx, dy) per node
```

Value network: same GCN encoder → global mean-pool → MLP → scalar.

### Action

At each step, the policy outputs a displacement $(dx, dy)$ for **every node simultaneously**. Displacements are then **masked** — only nodes with at least one crossing are allowed to move. This focuses the agent on nodes that actually matter.

### Reward

$$R_{t+1} = (\text{crossings}_t - \text{crossings}_{t+1}) + 0.2 \cdot (\text{SoftLoss}_t - \text{SoftLoss}_{t+1})$$

Hard crossing improvement is primary; soft loss improvement provides a dense signal to guide learning even when hard crossings don't change.

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | PPO with GAE (λ=0.95) |
| Clip ε | 0.2 |
| Entropy coef | 0.05 → 0.005 (annealed) |
| Learning rate | 3e-4 → 3e-5 (cosine decay) |
| Episodes | 200 per graph (split across 3 seeds) |
| Steps per episode | 40 |

**Per-graph training:** PPO is re-trained from scratch for each test graph. No generalization across graphs — the agent specializes fully to each graph.

**Multi-seed:** Run PPO 3 times with different random seeds, take the best result. Reduces sensitivity to unlucky initializations.

**Exploration annealing:** `log_std` of the displacement distribution decreases over training (−0.5 → −2.0), tightening the policy from broad exploration to precise local refinement.

**Adaptive budget:**
- 0 crossings after SLS → skip PPO
- 1–3 crossings → extra SLS only (PPO overhead not worth it)
- 4–29 crossings → standard 200 episodes
- 30+ crossings → 300 episodes (hard graphs need more exploration)

---

## Results (Test set: 99 graphs)

| Method | Mean Crossings | SPC vs Neato |
|--------|---------------|-------------|
| Neato | 29.1 | 0% (baseline) |
| SFDP | — | −3.14% |
| SmartGD | — | −3.60% |
| Diff GD (warm-start) | 18.0 | −37.18% |
| **PPO Refinement (ours)** | **15.9** | **−44.04%** |

SPC = Symmetric Percent Change. Negative = better than neato.

**RL improves over Diff GD by 6.86 percentage points** (15.9 vs 18.0 mean crossings).

---

## Key Design Choices & Why

| Choice | Rationale |
|--------|-----------|
| Diff GD warm-start | Reaches near-optimal differentiable solution before RL starts |
| Multi-restart Diff GD | Combats local minima in the soft proxy loss |
| SLS before PPO | Fast and effective; handles easy crossings cheaply |
| Action masking | Focuses agent on nodes causing crossings; ignores irrelevant nodes |
| Hybrid reward | Hard crossings are sparse signal; soft loss provides dense gradient |
| Per-graph PPO | No generalization needed; agent fully specializes |
| Multi-seed PPO | Cheap on HPC; guards against bad random initialization |
| Annealing | Broad exploration early, precise refinement late |

---

## What Didn't Work / Limitations

- Generalization across graphs (amortized policy) not yet explored
- PPO alone without SLS performs worse — the combination is key
- Very large graphs (high crossing count) still hard to solve fully
- Per-graph training is slow at test time (~minutes per graph)
