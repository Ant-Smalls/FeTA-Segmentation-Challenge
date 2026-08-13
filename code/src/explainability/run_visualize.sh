#!/bin/bash -l
#SBATCH --job-name=feta_visualize
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00              # inference-only
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucdconnect.ie
#SBATCH --output=logs/feta_visualize_%j.log
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

# Generates comparative-model figures (T2w / GT / refined prediction / uncertainty)
# across several test cases. Requires src/training/checkpoints_comparative/best_model_comp.pt.
for i in 0 1 2 3 4; do
  PYTHONUNBUFFERED=1 python3 -m src.explainability.visualize \
    --checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
    --model-type comparative --refine \
    --split test --case-index $i \
    --out-dir src/explainability/figures
done

echo "Done. Figures in src/explainability/figures/."
