#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:full:3
#SBATCH --job-name=ddpm_25d_eval
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=8

cd "$SLURM_SUBMIT_DIR"

# Require an argument (module or path)
if [ -z "$1" ]; then
  echo "Error: no target specified." >&2
  echo "Usage: sbatch job_submission_scripts/evaluation_job_accelerate.sh <module_or_path> [args...]" >&2
  echo "Examples:" >&2
  echo "  sbatch job_submission_scripts/evaluation_job_accelerate.sh model_scripts.ddpm_25d_mm.eval_metrics --ckpt /path/to/ddpm_25d_mm_best.pt" >&2
  echo "  sbatch job_submission_scripts/evaluation_job_accelerate.sh model_scripts/ddpm_25d_mm/eval_metrics.py --ckpt /path/to/ddpm_25d_mm_best.pt" >&2
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

# -----------------------------
# Decide how many processes to launch
# -----------------------------
NUM_VISIBLE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sed '/^\s*$/d' | wc -l | tr -d ' ')
if [[ -z "$NUM_VISIBLE" || "$NUM_VISIBLE" -lt 1 ]]; then
  NUM_VISIBLE=1
fi

NUM_PROCS="$NUM_VISIBLE"
CUDA_VISIBLE_DEVICES_FINAL="$CUDA_VISIBLE_DEVICES"

# If MIG UUIDs are present, run at most one process per physical GPU
# to avoid NCCL duplicate-GPU issues.
if echo "$CUDA_VISIBLE_DEVICES" | grep -q "MIG-"; then
  echo "Detected MIG UUIDs in CUDA_VISIBLE_DEVICES. Collapsing to one MIG per physical GPU."

  declare -A GPU_BUS
  while IFS=',' read -r idx bus uuid; do
    idx=$(echo "$idx" | xargs)
    bus=$(echo "$bus" | xargs)
    GPU_BUS["$idx"]="$bus"
  done < <(nvidia-smi --query-gpu=index,pci.bus_id,uuid --format=csv,noheader)

  declare -A MIG_PARENT_GPU
  current_gpu=""
  while IFS= read -r line; do
    if [[ "$line" =~ ^GPU[[:space:]]+([0-9]+): ]]; then
      current_gpu="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ UUID:[[:space:]]*(MIG-[A-Za-z0-9\-]+)\) ]]; then
      mig_uuid="${BASH_REMATCH[1]}"
      MIG_PARENT_GPU["$mig_uuid"]="$current_gpu"
    fi
  done < <(nvidia-smi -L)

  declare -A SEEN_GPU
  UNIQUE_DEVICES=()

  IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
  for dev in "${DEV_ARR[@]}"; do
    dev="$(echo "$dev" | xargs)"
    parent="${MIG_PARENT_GPU[$dev]}"

    if [[ -z "$parent" ]]; then
      UNIQUE_DEVICES+=("$dev")
      continue
    fi

    if [[ -z "${SEEN_GPU[$parent]}" ]]; then
      SEEN_GPU["$parent"]=1
      UNIQUE_DEVICES+=("$dev")
      echo "Using MIG $dev on physical GPU $parent (PCI ${GPU_BUS[$parent]})"
    else
      echo "Skipping MIG $dev (same physical GPU $parent / PCI ${GPU_BUS[$parent]})"
    fi
  done

  CUDA_VISIBLE_DEVICES_FINAL="$(IFS=,; echo "${UNIQUE_DEVICES[*]}")"
  NUM_PROCS="${#UNIQUE_DEVICES[@]}"
fi

echo "CUDA_VISIBLE_DEVICES (original): $CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES (used):     $CUDA_VISIBLE_DEVICES_FINAL"
echo "NUM_VISIBLE: $NUM_VISIBLE | NUM_PROCS: $NUM_PROCS"

echo "CWD: $(pwd)"
echo

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_FINAL"

# -----------------------------
# Launch
# -----------------------------
if [[ "$NUM_PROCS" -le 1 ]]; then
  echo "Launching SINGLE-GPU / SINGLE-PROCESS (no --multi_gpu)"
  echo "Command: accelerate launch --num_processes 1 -m $MODULE $*"
  accelerate launch --num_processes 1 -m "$MODULE" "$@"
else
  echo "Launching MULTI-GPU with $NUM_PROCS processes (--multi_gpu)"
  echo "Command: accelerate launch --multi_gpu --num_processes $NUM_PROCS -m $MODULE $*"
  accelerate launch --multi_gpu --num_processes "$NUM_PROCS" -m "$MODULE" "$@"
fi

EXIT_CODE=$?
exit "$EXIT_CODE"
