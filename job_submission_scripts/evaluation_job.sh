#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:full:4
#SBATCH --job-name=ddpm_eval
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8

cd "$SLURM_SUBMIT_DIR"

if [ -z "$1" ]; then
  echo "Error: no target specified." >&2
  echo "Usage: sbatch job_eval.sh <module_or_path> [args...]" >&2
  echo "Examples:" >&2
  echo "  sbatch job_eval.sh model_scripts.slice_cond_2d_ddpm.metrics" >&2
  echo "  sbatch job_eval.sh model_scripts/slice_cond_2d_ddpm/metrics.py" >&2
  exit 1
fi

TARGET="$1"
shift

MODULE="${TARGET%.py}"
MODULE="${MODULE//\//.}"
SCRIPT_PATH="${MODULE//./\/}.py"

SCRIPT_NAME="$(basename "$SCRIPT_PATH" .py)"
PACKAGE_PATH="${SCRIPT_PATH%/*}"
PACKAGE_DIR="$(basename "$PACKAGE_PATH")"

LOG_DIR="${PACKAGE_PATH}/logs/${SCRIPT_NAME}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SLURM_JOB_ID}.out"

# Redirect output
exec >"$LOG_FILE" 2>&1

echo "Logging to: $LOG_FILE"
echo "Target: $TARGET"
echo "Module (normalized): $MODULE"
echo "Script path (derived): $SCRIPT_PATH"
echo
echo "SLURM job id:        $SLURM_JOB_ID"
echo "Node(s):             $SLURM_JOB_NODELIST"
echo "SLURM_JOB_GPUS:      $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES"
echo

echo "==== scontrol (Gres/Tres info) ===="
scontrol show job "$SLURM_JOB_ID" | egrep -i "gres|tres"
echo

nvidia-smi
echo

# Env / venv
source "$HOME"/.bashrc
source .env
source venv/bin/activate


NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sed '/^\s*$/d' | wc -l)
[ -z "$NUM_GPUS" ] && NUM_GPUS=1

echo "CWD: $(pwd)"
echo "NUM_GPUS: $NUM_GPUS"
echo
echo "Command: accelerate launch --multi_gpu --num_processes $NUM_GPUS -m $MODULE $*"
echo

accelerate launch --multi_gpu --num_processes "$NUM_GPUS" -m "$MODULE" "$@"
EXIT_CODE=$?

exit "$EXIT_CODE"