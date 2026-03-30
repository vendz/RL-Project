#!/bin/bash
#SBATCH --job-name=rl_graph_dqn
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# -----------------------------------------------------------------------
# DQN training for graph layout optimisation
# Run from project root: sbatch scripts/train_dqn.sh
# -----------------------------------------------------------------------

set -e

PROJECT_DIR="/scratch/vasa.v/RL-Project"
cd "$PROJECT_DIR"

mkdir -p logs checkpoints

module load anaconda3/2024.06
source activate /scratch/vasa.v/conda_envs/rl_graph

echo "=== Environment ==="
python --version
python -c "import torch; print('PyTorch', torch.__version__)"

echo "=== Starting training ==="
python train_dqn.py \
    --rome-dir      rome \
    --train-start   0 \
    --train-end     500 \
    --num-episodes  5000 \
    --max-steps     50 \
    --step-size     5.0 \
    --hidden-dim    128 \
    --lr            1e-3 \
    --gamma         0.99 \
    --epsilon-start 1.0 \
    --epsilon-end   0.05 \
    --epsilon-decay 0.998 \
    --target-update-freq 200 \
    --batch-size    32 \
    --buffer-size   50000 \
    --log-interval  100 \
    --save-interval 1000 \
    --checkpoint-dir checkpoints \
    --log-dir        logs \
    --cpu

echo "=== Training done ==="

echo "=== Running evaluation ==="
python evaluate_dqn.py \
    --checkpoint  checkpoints/dqn_final.pt \
    --rome-dir    rome \
    --eval-start  10000 \
    --eval-end    10100 \
    --max-steps   200 \
    --step-size   5.0 \
    --hidden-dim  128 \
    --output      dqn_eval_results.csv \
    --cpu

echo "=== Done ==="
