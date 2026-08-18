#!/bin/bash -l
#SBATCH --job-name=feta_train_comparative
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@ucdconnect.ie
#SBATCH --output=logs/feta_train_comparative_%j.log
#SBATCH --error=logs/feta_train_comparative_%j.err

set -euo pipefail

CODE_DIR="/home/people/ucd_number/scratch/segmentation-assignment/code"
cd "$CODE_DIR"
mkdir -p logs
mkdir -p src/training/checkpoints_comparative

echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
date
source venv/bin/activate

# Confirm GPU is visible to this job before spending wall time on training.
python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

# --- comparative (uncertainty) model training ---
# Trains UncertaintyUNet from scratch (does NOT load the baseline checkpoint).
# Writes best_model.pt + comparative_history.{json,png} under
# src/training/checkpoints_comparative/
# NOTE: -m needs package dots (src.training.train), not path slashes.
python3 -u -m src.training.train --config src/training/train_config_comparative.yaml

echo "Comparative training done. Check:"
echo "  src/training/checkpoints_comparative/best_model.pt"
echo "  src/training/checkpoints_comparative/comparative_history.png"
echo "  src/training/tuning_results/refinement_threshold_sweep.json"
echo "  src/training/tuning_results/refinement_threshold_sweep.png"
date
