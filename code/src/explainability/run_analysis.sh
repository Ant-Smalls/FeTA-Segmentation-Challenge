#!/bin/bash -l
#SBATCH --job-name=feta_unc_analysis
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00              # MC-Dropout inference over the full split, twice
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucdconnect.ie
#SBATCH --output=logs/feta_unc_analysis_%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Working dir: $(pwd)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

# module load python/3.10 lacks LD_LIBRARY_PATH and SSL support; use anaconda3 instead.
module load anaconda3/2024.10-1

source venv/bin/activate

echo ""
nvidia-smi
echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Unrefined, so the error statistics describe the model's raw output.
PYTHONUNBUFFERED=1 python3 -m src.explainability.uncertainty_analysis \
  --checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
  --split test \
  --out-dir src/explainability/analysis

# Separates the uncertainty gate from the smoothing filter it gates.
PYTHONUNBUFFERED=1 python3 -m src.explainability.refinement_control \
  --baseline-checkpoint src/training/checkpoints/best_model.pt \
  --comparative-checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
  --split test \
  --out-dir src/explainability/analysis

echo "Done. Reports in src/explainability/analysis/."
