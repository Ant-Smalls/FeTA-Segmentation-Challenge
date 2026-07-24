#!/bin/bash -l
#SBATCH --job-name=feta_qc_preprocess
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@ucdconnect.ie
#SBATCH --output=logs/feta_qc_preprocess_%j.log

set -euo pipefail

CODE_DIR="/home/people/ucd_number/scratch/segmentation-assignment/code"
cd "$CODE_DIR"
mkdir -p logs

echo "Working dir: $(pwd)"
source venv/bin/activate

# ---  QC + stratified split ---
python3 -m src.preprocessing_data_preparation.qc_and_split

echo "QC/split done. Review exclusions above, then check:"
echo "  src/data/splits/split_v1.json"

