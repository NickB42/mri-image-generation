#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4g.20gb:2
#SBATCH --job-name=ddpm_3d_ldm
#SBATCH --output=/dev/null
#SBATCH --signal=SIGUSR1@600

cd "$SLURM_SUBMIT_DIR"

# Require an argument (module or path)
if [ -z "$1" ]; then
  echo "Error: no target specified." >&2
  echo "Usage: sbatch job_submission.sh <module_or_path> [args...]" >&2
  echo "Examples:" >&2
  echo "  sbatch job_submission.sh model_scripts.DDPM.model" >&2
  echo "  sbatch job_submission.sh model_scripts/DDPM/model" >&2
  exit 1
fi

TARGET="$1"
shift

# Determine module and script path
if [[ "$TARGET" == *"/"* ]]; then
  SCRIPT_PATH="$TARGET"
  [[ "$SCRIPT_PATH" != *.py ]] && SCRIPT_PATH="${SCRIPT_PATH}.py"

  MODULE="${SCRIPT_PATH%.py}"
  MODULE="${MODULE//\//.}"
else
  MODULE="$TARGET"
  SCRIPT_PATH="${MODULE//./\/}.py"
fi

# Script dir & name for logging
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH" .py)"

# Logs: <script_dir>/logs/<script_name>/<jobid>.out
LOG_DIR="${SCRIPT_DIR}/logs/${SCRIPT_NAME}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SLURM_JOB_ID}.out"

# Redirect output
exec >"$LOG_FILE" 2>&1

echo "Logging to: $LOG_FILE"
echo "Module: $MODULE"
echo "Script path (derived): $SCRIPT_PATH"
echo

echo "SLURM job id:  $SLURM_JOB_ID"
echo "Node(s):       $SLURM_JOB_NODELIST"
echo "SLURM_JOB_GPUS:  $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo

echo "==== scontrol (Gres/Tres info) ===="
scontrol show job "$SLURM_JOB_ID" | egrep -i "gres|tres"
echo

nvidia-smi
echo 

source "$HOME"/.bashrc
source .env
source venv/bin/activate

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "Running module with torchrun: $MODULE"
echo "CWD: $(pwd)"
echo "NUM_GPUS: $NUM_GPUS"
echo "Command: torchrun --standalone --nproc_per_node=${NUM_GPUS} -m $MODULE $*"
echo

### GPU MONITOR: start periodic logging in background
GPU_LOG="${LOG_DIR}/gpu_usage_${SLURM_JOB_ID}.csv"
echo "Logging GPU usage to: $GPU_LOG"
echo "timestamp,util.gpu,util.mem,mem.used,mem.total" > "$GPU_LOG"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits \
  -l 30 >> "$GPU_LOG" &
GPU_MONITOR_PID=$!

# Run training
torchrun --standalone --nproc_per_node=${NUM_GPUS} -m "$MODULE" "$@"
TRAIN_EXIT_CODE=$?

kill "$GPU_MONITOR_PID" 2>/dev/null || true

exit "$TRAIN_EXIT_CODE"