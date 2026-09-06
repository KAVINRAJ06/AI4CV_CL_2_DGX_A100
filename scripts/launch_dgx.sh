#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" -m tqsi.train --config "${1:-configs/dgx.yaml}" "${@:2}"
