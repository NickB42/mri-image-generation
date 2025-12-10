#!/bin/bash
#SBATCH --job-name=brats_preproc
#SBATCH --output=logs/brats_preproc_%j.out
#SBATCH --error=logs/brats_preproc_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Require first two positional args: ROOT_DIR and OUT_DIR
if [ $# -lt 2 ]; then
    echo "Usage: $0  OUT_DIR [--image_size 128] [additional args...]"
    exit 1
fi

ROOT_DIR="$1"
OUT_DIR="$2"
shift 2

# Ensure logs directory exists (for SBATCH output paths)
mkdir -p logs

# Move to the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Env setup
source "$HOME/.bashrc"
[ -f .env ] && source .env
[ -d venv ] && source venv/bin/activate

IMAGE_SIZE="${IMAGE_SIZE:-128}"

echo "Running module: ${MODULE:-<none>}"
echo "CWD: $(pwd)"
echo "Python: $(command -v python)"
echo "Command: python model_scripts/slice_cond_2d_ddpm/preprocess_data.py --root_dir \"$ROOT_DIR\" --output_dir \"$OUT_DIR\" --image_size \"$IMAGE_SIZE\" $*"
echo

python model_scripts/slice_cond_2d_ddpm/preprocess_data.py \
    --root_dir "$ROOT_DIR" \
    --output_dir "$OUT_DIR" \
    --image_size "$IMAGE_SIZE" \
    "$@"
