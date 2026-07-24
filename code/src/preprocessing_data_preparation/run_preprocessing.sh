#!/bin/bash -l
#SBATCH --job-name=feta_preprocess_validate
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@ucdconnect.ie
#SBATCH --output=logs/feta_preprocess_validate_%j.log
set -euo pipefail

CODE_DIR="/home/people/ucd_number/scratch/segmentation-assignment/code"
cd "$CODE_DIR"
mkdir -p logs

echo "Working dir: $(pwd)"
source venv/bin/activate

# --- full-scale preprocessing validation across every case in
# split_v1.json (requires run_qc_stratify.sh) ---
python3 -m src.preprocessing_data_preparation.validate_preprocessing

echo "Preprocessing validation done. Review failures/timing summary above."
