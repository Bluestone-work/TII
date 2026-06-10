#!/bin/bash
# 一键清理所有 ROS2 / Gazebo 相关进程，并兜底释放目标 Gazebo 端口

set -euo pipefail

KILL_ALL_ROS_SCOPE="${KILL_ALL_ROS_SCOPE:-global}"

extract_gazebo_port() {
    local uri="${GAZEBO_MASTER_URI:-}"
    if [[ -n "${GAZEBO_PORT:-}" ]]; then
        echo "${GAZEBO_PORT}"
        return 0
    fi
    if [[ "$uri" =~ :([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    echo "11345"
}

kill_port_listeners() {
    local port="${1:-}"
    [[ -n "$port" ]] || return 0

    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN -Pn 2>/dev/null | tr '\n' ' ' || true)"
    fi
    if [[ -z "$pids" ]] && command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$port" 2>/dev/null | tr '\n' ' ' || true)"
    fi

    if [[ -n "$pids" ]]; then
        echo "🎯 检测到 Gazebo 端口 ${port} 残留监听进程: ${pids}"
        kill $pids 2>/dev/null || true
        sleep 1
        kill -9 $pids 2>/dev/null || true
    fi
}

kill_process_patterns() {
    local sig="${1:-TERM}"
    shift || true
    local pattern
    for pattern in "$@"; do
        pkill "-${sig}" -f "$pattern" 2>/dev/null || true
    done
}

echo "🛑 正在清理 ROS2 进程..."
echo "🧭 清理范围: ${KILL_ALL_ROS_SCOPE}"

TARGET_GAZEBO_PORT="$(extract_gazebo_port)"
echo "🎯 目标 Gazebo 端口: ${TARGET_GAZEBO_PORT}"

if [[ "$KILL_ALL_ROS_SCOPE" == "port_only" ]]; then
    kill_port_listeners "$TARGET_GAZEBO_PORT"
    sleep 1
    kill_port_listeners "$TARGET_GAZEBO_PORT"
    remaining=$(lsof -tiTCP:"$TARGET_GAZEBO_PORT" -sTCP:LISTEN -Pn 2>/dev/null | wc -l || true)
    if [ "${remaining:-0}" -eq 0 ]; then
        echo "✅ 目标 Gazebo 端口 ${TARGET_GAZEBO_PORT} 已清理完毕"
    else
        echo "⚠️  端口 ${TARGET_GAZEBO_PORT} 仍有残留监听，已尽力清理"
    fi
else
    # 方法1: Kill launch进程（会自动清理子进程）
    kill_process_patterns TERM \
        "ros2 launch" \
        "gzserver" \
        "gzclient" \
        "gazebo.*world" \
        "robot_state_publisher" \
        "static_transform_publisher" \
        "lifecycle_manager" \
        "nav2_map_server" \
        "map_server" \
        "obstacle_mover.py" \
        "spawn_entity.py" \
        "rviz2" \
        "python3.*ros2" \
        "python3.*/home/wj/work/multi-robot-exploration-rl"

    kill_port_listeners "$TARGET_GAZEBO_PORT"
    sleep 2
    kill_port_listeners "$TARGET_GAZEBO_PORT"

    # 检查是否还有残留
    remaining=$(ps aux | grep -E "(ros2|gazebo|gzserver|gzclient|spawn_entity|robot_state_publisher|rviz2)" | grep -v grep | wc -l)

    if [ "$remaining" -eq 0 ]; then
        echo "✅ 所有 ROS2 进程已清理完毕"
    else
        echo "⚠️  还有 $remaining 个进程残留，尝试强制清理..."
        kill_process_patterns KILL \
            "ros2 launch" \
            "gzserver" \
            "gzclient" \
            "spawn_entity.py" \
            "obstacle_mover.py"
        kill_port_listeners "$TARGET_GAZEBO_PORT"
        sleep 1
        echo "✅ 强制清理完成"
    fi

    echo ""
    echo "当前 ROS2 相关进程："
    ps aux | grep -E "(ros2|gazebo|gzserver|gzclient|spawn_entity|robot_state_publisher|rviz2)" | grep -v grep || echo "无残留进程"
fi
