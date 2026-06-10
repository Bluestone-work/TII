#!/bin/bash

set -euo pipefail

TARGET_SCRIPT="/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/run_curriculum.sh"

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "[ERROR] 未找到目标脚本: $TARGET_SCRIPT" >&2
    exit 1
fi

exec /bin/bash "$TARGET_SCRIPT" "$@"
