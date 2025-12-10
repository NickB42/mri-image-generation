#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --job-name=ddpm_gen
#SBATCH --output=/dev/null

cd "$SLURM_SUBMIT_DIR"

# Require an argument (module or path)
if [ -z "$1" ]; then
  echo "Error: no target specified." >&2
  echo "Usage: sbatch job_generation.sh <module_or_path> [args...]" >&2
  echo "Examples:" >&2
  echo "  sbatch job_generation.sh model_scripts.generate_pseudo3d_volume" >&2
  echo "  sbatch job_generation.sh model_scripts/generate_pseudo3d_volume" >&2
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
echo

# Env / venv
source "$HOME"/.bashrc
source .env
source venv/bin/activate

echo "Running module: $MODULE"
echo "CWD: $(pwd)"
echo "Command: python -m $MODULE $*"
echo

# Run generation
python -m "$MODULE" "$@"
EXIT_CODE=$?

exit "$EXIT_CODE"