#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:full:3
#SBATCH --job-name=ddpm_25d_mm_train
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8

# ==================================================================
# Train the memmap-backed 2.5D DDPM with Accelerate.
#
# Prerequisites:
#   1. Run preprocessing first:
#      sbatch job_submission_scripts/preprocess_mm.sh
#
# Usage (from project root):
#   sbatch job_submission_scripts/training_job_mm.sh
# ==================================================================

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Env setup
source "$HOME/.bashrc"
[ -f .env ] && source .env
[ -d venv ] && source venv/bin/activate

# Logging
LOG_DIR="model_scripts/ddpm_25d_mm/logs/train/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SLURM_JOB_ID}.out"

exec >"$LOG_FILE" 2>&1

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "CWD:       $(pwd)"
echo "Python:    $(command -v python)"
echo "GPUs:      $(echo $CUDA_VISIBLE_DEVICES)"
echo "Date:      $(date)"
echo "============================================"

# Verify memmap files exist
if [ ! -f datasets/memmap/train_flair_256.npy ] || [ ! -f datasets/memmap/val_flair_256.npy ]; then
    echo "ERROR: Memmap files not found. Run preprocess_mm.sh first."
    exit 1
fi

echo "Memmap files:"
ls -lh datasets/memmap/

## GPU MONITOR: start periodic logging in background
# GPU_LOG="${LOG_DIR}/gpu_usage_${SLURM_JOB_ID}.csv"
# echo "Logging GPU usage to: $GPU_LOG"
# echo "timestamp,util.gpu,util.mem,mem.used,mem.total" > "$GPU_LOG"

# nvidia-smi \
#   --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
#   --format=csv,noheader,nounits \
#   -l 2 >> "$GPU_LOG" &
# GPU_MONITOR_PID=$!

# Cleanup GPU monitor
# trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

echo ""
echo "=== Starting training ==="
accelerate launch -m model_scripts.ddpm_25d_mm.train
