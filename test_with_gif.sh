#!/bin/bash
# GIF测试快捷方式 - 自动调用DGTA_nobuffer版本的run_test.sh
# 用法: ./test_with_gif.sh -c <checkpoint> [其他参数]

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGTA_DIR="$WORKSPACE/src/gnn_marl_training_DGTA_nobuffer"

if [[ ! -f "$DGTA_DIR/run_test.sh" ]]; then
    echo "错误: 找不到 $DGTA_DIR/run_test.sh"
    exit 1
fi

cd "$DGTA_DIR"
exec ./run_test.sh "$@"
