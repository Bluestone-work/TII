#!/bin/bash
# 直接运行训练脚本的包装器

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Source ROS2环境
source "$WORKSPACE_DIR/install/setup.bash"

# 添加Python路径
export PYTHONPATH="$WORKSPACE_DIR/src/sb3_training:$PYTHONPATH"

# 运行训练脚本
python3 "$WORKSPACE_DIR/src/sb3_training/sb3_training/train_ppo.py" "$@"
