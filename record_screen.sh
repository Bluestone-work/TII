#!/bin/bash
# 屏幕录制辅助脚本 - 录制Gazebo/RViz为GIF
# 用法: ./record_screen.sh <output.gif> <duration_sec> [x y width height]

set -euo pipefail

OUTPUT_FILE="$1"
DURATION="${2:-60}"  # 默认60秒
X="${3:-0}"
Y="${4:-0}"
WIDTH="${5:-1920}"
HEIGHT="${6:-1080}"

# 检查依赖
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg未安装,运行: sudo apt install ffmpeg"
    exit 1
fi

echo "🎬 开始录制屏幕..."
echo "   区域: ${WIDTH}x${HEIGHT} 从 (${X},${Y})"
echo "   时长: ${DURATION}秒"
echo "   输出: $OUTPUT_FILE"

# 方案1: 录制指定区域 (如果xdotool可用,自动找窗口)
if command -v xdotool &> /dev/null; then
    # 尝试找Gazebo窗口
    GZCLIENT_WIN=$(xdotool search --name "Gazebo" | head -1 || echo "")
    if [[ -n "$GZCLIENT_WIN" ]]; then
        eval $(xdotool getwindowgeometry --shell "$GZCLIENT_WIN")
        X=$X
        Y=$Y
        WIDTH=$WIDTH
        HEIGHT=$HEIGHT
        echo "✓ 自动定位Gazebo窗口: ${WIDTH}x${HEIGHT}+${X}+${Y}"
    fi
fi

# 录制为mp4(更高效),然后转GIF
TMP_MP4="${OUTPUT_FILE%.gif}_tmp.mp4"

# 使用x11grab录制屏幕
ffmpeg -f x11grab -r 10 -s "${WIDTH}x${HEIGHT}" -i "${DISPLAY}+${X},${Y}" \
    -t "$DURATION" -vcodec libx264 -preset ultrafast -y "$TMP_MP4" 2>&1 | \
    grep -E "frame=|error|Error" || true

# 转换为GIF(压缩)
if [[ -f "$TMP_MP4" ]]; then
    ffmpeg -i "$TMP_MP4" -vf "fps=5,scale=640:-1:flags=lanczos" \
        -y "$OUTPUT_FILE" 2>&1 | grep -E "frame=|error|Error" || true
    rm -f "$TMP_MP4"

    if [[ -f "$OUTPUT_FILE" ]]; then
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "✅ 录制完成: $OUTPUT_FILE ($SIZE)"
    else
        echo "❌ GIF生成失败"
        exit 1
    fi
else
    echo "❌ 屏幕录制失败"
    exit 1
fi
