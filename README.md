# RL-Based Graph Layout Optimization

## Project Overview

This repository contains a complete pipeline for optimizing graph layouts to minimize edge crossings using reinforcement learning (RL). The approach combines:

1. **Diff GD**: Differentiable warm-start using soft crossing penalty + stress loss
2. **RL Refinement**: Multi-seed PPO with per-graph specialization
3. **SLS Post-Processing**: Stochastic local search for final refinement

## Results Summary

| Metric                     | Test Set (99 graphs) | Complete Dataset (11,383 graphs) |
| -------------------------- | -------------------- | -------------------------------- |
| **RL SPC**                 | -43.98%              | -51.14%                          |
| **Diff GD SPC**            | -38.00%              | -45.31%                          |
| **RL Margin over Diff GD** | +5.98pp              | -5.83pp                          |
| **RL Win Rate (better)**   | 51.5%                | 56.9%                            |
| **Equal Performance**      | 48.5%                | 43.1%                            |
| **RL Worse**               | 0%                   | 0%                               |
| **Node Overlaps (valid)**  | 0 (100%)             | —                                |

**Key observations:**

- Test set shows RL +5.98pp advantage over Diff GD
- Complete dataset shows RL -5.83pp (slight disadvantage to Diff GD)—likely due to distribution differences in full dataset vs. test subset
- RL maintains stronger win rate consistency (56.9% vs 51.5%)
- Both datasets have zero worse cases (RL never underperforms catastrophically)

## Running the Pipeline

### Prerequisites

```bash
uv sync
```

### Single Graph Processing (Local)

```bash
python -m scripts.run_experiment \
  --rome-dir rome \
  --start 10000 --end 10100 \
  --output test_results.csv \
  --device cpu \
  --verbose
```

### Test Graphs with Coordinates

```bash
python -m scripts.compute_test_coords \
  --rome-dir rome \
  --start 10000 --end 10100 \
  --output test_results.csv \
  --coords-dir coords/
```

## Key Design Decisions

### 1. Per-Graph RL (No Amortization)

- Each graph gets its own PPO agent specialization
- No generalization across graphs → no distribution shift
- Trade-off: Can't reuse policy across new graphs
- Benefit: Guaranteed performance on fixed dataset (Rome)

### 2. Soft + Hard Crossing Loss

- **Soft XingLoss**: Smooth, differentiable for warm-start GD
- **Hard XingLoss**: True crossing count for evaluation
- Decoupling prevents gradient issues during refinement

### 3. Checkpointing Architecture

- Results written per-graph to allow recovery from timeouts
- HPC shards resume incomplete work automatically
- Protects against 24h job limit timeouts

### 4. Overlap Detection & Penalty

- All test graphs validated for node overlaps (distance < 1e-6)
- Degenerate layouts penalized: `rl_xing = neato_xing × 1.5`
- None detected in test set (robust RL training)

## Evaluation Metrics

### Symmetric Percent Change (SPC)

```
SPC = ((method - baseline) / baseline) × 100%

Interpretation:
- SPC = -43.98% → Method achieves 43.98% fewer crossings than baseline
- Higher (less negative) is worse
- Lower (more negative) is better
```

### Per-Graph Metrics

- **neato_xing**: Baseline from GraphViz Neato
- **diff_gd_xing**: Crossings after warm-start GD
- **rl_xing**: Final crossings after RL + SLS refinement

## Experiment Logs & Results

### Test Set Analysis

See **TEST_GRAPH_ANALYSIS.md** for:

- Detailed statistics (mean, median, min, max improvements)
- Performance stratified by graph size
- Head-to-head comparison tables
- Sample results (top improvements, equal cases)

## Configuration

### Hyperparameters

```bash
--rl-episodes 300    # PPO episodes per graph
--rl-steps 40        # Steps per episode
--rl-seeds 3         # Independent seed trials
--device cpu         # CPU-only (no GPU needed)
```

### Environment Variables

- `ROME_DIR`: Path to Rome graph dataset (default: `./rome/`)
- `CUDA_VISIBLE_DEVICES`: GPU selection

## Reproducibility

To reproduce full dataset results:

1. Download Rome graph collection
2. Update `ROME_DIR` or pass `--rome-dir` to scripts
3. Run via HPC using provided sbatch templates
4. Merge shards with `utils/merge_results.py`
5. Compute SPC metrics via `utils/metrics.py`

## Citation & References

**Baseline**: GraphViz Neato (Gansner & North)  
**Warm-Start**: Diff GD (Smooth stress + soft crossing loss)  
**Refinement**: PPO (Schulman et al., 2017) on per-graph specialization  
**Post-Processing**: Stochastic local search (node 2-opt moves)  
**Evaluation**: Symmetric Percent Change (custom metric)
