#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ERA5_DIR [training options]" >&2
  echo "Example: $0 /path/to/era5 --max-epochs 30 --batch-size 16" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/train.py" "$@"
