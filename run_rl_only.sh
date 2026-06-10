#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail

WORKSPACE="/home/wj/work/multi-robot-exploration-rl"
RUN_SCRIPT="${WORKSPACE}/run_curriculum.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_SUFFIX="rl_only_${STAMP}"

usage() {
    cat <<'EOF'
Pure RL launcher for the current GNN-MAPPO mainline.

This wrapper intentionally avoids any BC / imitation-learning warm start.
It forwards to `run_curriculum.sh` with defaults that match the current
`option_mode`-only training entry.

Defaults:
  --model_type gat
  --gat_actor_graph neighbor
  --gat_critic_mode mlp
  --action_mode option_mode
  --ppo_profile auto
  --option_policy_source rl
  --enable_replan_option 1
  --num_agents 4
  --num_workers 0
  --start_stage 1
  --end_stage 4
  --train_steps 500000
  --train_batch_size 5000
  --checkpoint_freq 20
  --rollout_fragment_length 1000
  --high_conflict_mode mixed
  --high_conflict_prob 0.75
  --rolling_lookahead_dist 0.8
  --obstacle_filter_range 1.2
  --obstacle_filter_fov_deg 360
  --obstacle_top_k 9
  visualization enabled
  --run_suffix rl_only_<timestamp>

Examples:
  ./run_rl_only.sh
  ./run_rl_only.sh --train_steps 300000
  ./run_rl_only.sh --start_stage 1 --end_stage 4
  ./run_rl_only.sh --resume /abs/path/checkpoint --start_stage 2 --end_stage 2

Notes:
  1. Extra arguments are forwarded to `run_curriculum.sh`.
  2. Forwarded arguments come last, so they can override the defaults above.
  3. BC-related flags are intentionally not added here.
  4. Set `RL_ONLY_LOCAL_TRANSPORT=1` to force localhost-only ROS + SHM Fast DDS.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "[ERROR] 未找到训练入口: ${RUN_SCRIPT}" >&2
    exit 1
fi

# Keep Gazebo on loopback by default to reduce multicast / interface issues.
export GAZEBO_IP="${GAZEBO_IP:-127.0.0.1}"

# Optional fallback for restricted environments where UDP interface discovery fails.
if [[ "${RL_ONLY_LOCAL_TRANSPORT:-0}" == "1" ]]; then
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-SHM}"
fi

exec "${RUN_SCRIPT}" \
    --model_type gat \
    --gat_actor_graph neighbor \
    --gat_critic_mode mlp \
    --action_mode option_mode \
    --ppo_profile auto \
    --option_policy_source rl \
    --enable_replan_option 1 \
    --num_agents 4 \
    --num_workers 0 \
    --start_stage 1 \
    --end_stage 1 \
    --train_steps 500000 \
    --train_batch_size 5000 \
    --checkpoint_freq 20 \
    --rollout_fragment_length 1000 \
    --high_conflict_mode mixed \
    --high_conflict_prob 0.75 \
    --rolling_lookahead_dist 0.8 \
    --obstacle_filter_range 1.2 \
    --obstacle_filter_fov_deg 360 \
    --obstacle_top_k 9 \
    --run_suffix "${DEFAULT_SUFFIX}" \
    "$@"
