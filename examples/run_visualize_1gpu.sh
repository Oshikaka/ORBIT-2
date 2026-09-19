#!/bin/bash
# ---------------------------------------------------------------------------
# Single-GPU ORBIT-2 downscaling inference + figures.
#
# Runs examples/visualize.py on ONE AMD GPU without going through Slurm, so it
# works on a Frontier login node that has a free GPU as well as inside an
# salloc/sbatch allocation.  visualize.py insists on reading SLURM_* variables,
# so we synthesise a 1-rank "allocation" when we are not already inside one.
#
# Usage:
#   ./examples/run_visualize_1gpu.sh [config.yaml] [extra args for visualize.py]
#
# Example:
#   ./examples/run_visualize_1gpu.sh configs/infer_us_9.5m_precip.yaml \
#       --index 200 --variable total_precipitation_24hr --output-dir outputs/us95m
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/infer_us_9.5m_precip.yaml}"
shift || true

# visualize.py imports its sibling utils.py, so it has to run from examples/.
# Resolve the config (and any --output-dir) first so that paths given relative
# to the current directory keep working after the cd.
[ -f "$CONFIG" ] && CONFIG="$(readlink -f "$CONFIG")"
ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--output-dir" ] && [ -n "${2:-}" ]; then
        mkdir -p "$2"
        ARGS+=("$1" "$(readlink -f "$2")")
        shift 2
    else
        ARGS+=("$1")
        shift
    fi
done

cd "$(dirname "$0")"

module load miniforge3/23.11.0-0 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/csc662/proj-shared/xinru/envs/orbit

export PYTHONPATH="$PWD/../src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export ORBIT_USE_DDSTORE=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/miopen-$USER-$$}"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Synthesise a single-rank allocation when not launched through srun.
export HOSTNAME="${HOSTNAME:-$(hostname)}"
export SLURM_NTASKS="${SLURM_NTASKS:-1}"
export SLURM_PROCID="${SLURM_PROCID:-0}"
export SLURM_LOCALID="${SLURM_LOCALID:-0}"
# Pick a per-user port so two people on the same login node do not collide.
PORT="${MASTER_PORT:-$((29500 + UID % 1000))}"

echo "config=$CONFIG  ntasks=$SLURM_NTASKS  master_port=$PORT"
exec python ./visualize.py "$CONFIG" --master-port "$PORT" ${ARGS[@]+"${ARGS[@]}"}
