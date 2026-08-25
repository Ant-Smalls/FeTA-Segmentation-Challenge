#!/bin/bash -l
#SBATCH --job-name=feta_refine_control
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00              # baseline and MC-Dropout inference per case
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucdconnect.ie
#SBATCH --output=logs/feta_refine_control_%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Working dir: $(pwd)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

# module load python/3.10 lacks LD_LIBRARY_PATH and SSL support; use anaconda3 instead.
module load anaconda3/2024.10-1

source venv/bin/activate

# Fail before any inference rather than partway through the allocation.
for checkpoint in src/training/checkpoints/best_model.pt \
                  src/training/checkpoints_comparative/best_model_comp.pt; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
  echo "Found $checkpoint ($(stat -c %s "$checkpoint") bytes)"
done

echo ""
nvidia-smi
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Separates the uncertainty gate from the smoothing filter it gates.
PYTHONUNBUFFERED=1 python3 -m src.explainability.refinement_control \
  --baseline-checkpoint src/training/checkpoints/best_model.pt \
  --comparative-checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
  --split test \
  --out-dir src/explainability/analysis

echo "Done. Report in src/explainability/analysis/refinement_control.json."
