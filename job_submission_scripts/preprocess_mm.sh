#!/bin/bash
#SBATCH --job-name=brats_mm_preproc
#SBATCH --output=model_scripts/ddpm_25d_mm/logs/preprocess/%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# ==================================================================
# Preprocess BraTS NIfTI volumes into numpy memmap files for the
# memmap-backed 2.5D DDPM.
#
# Usage (from project root):
#   sbatch job_submission_scripts/preprocess_mm.sh
# ==================================================================

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Env setup
source "$HOME/.bashrc"
[ -f .env ] && source .env
[ -d venv ] && source venv/bin/activate

# Create output and log directories
mkdir -p datasets/memmap
mkdir -p model_scripts/ddpm_25d_mm/logs/preprocess

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "CWD:       $(pwd)"
echo "Python:    $(command -v python)"
echo "Date:      $(date)"
echo "============================================"

# --- 1) Train set ---
echo ""
echo "=== Preprocessing TRAIN set ==="
python -m model_scripts.ddpm_25d_mm.prep_all \
    --root_dir datasets/train \
    --output_file datasets/memmap/train_flair_256.npy \
    --image_size 256

# --- 2) Val set ---
echo ""
echo "=== Preprocessing VAL set ==="
python -m model_scripts.ddpm_25d_mm.prep_all \
    --root_dir datasets/val \
    --output_file datasets/memmap/val_flair_256.npy \
    --image_size 256

echo ""
echo "=== All done ==="
echo "Train memmap: datasets/memmap/train_flair_256.npy"
echo "Val memmap:   datasets/memmap/val_flair_256.npy"
ls -lh datasets/memmap/
