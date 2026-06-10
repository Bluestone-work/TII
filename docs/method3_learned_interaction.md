# Method 3: Learned High-Level Interaction + Classical Low-Level Tracking

## Goal

This variant is the third comparison method in the project:

1. `method1_pure_rl`: no high-level controller, RL learns everything
2. `method2_rule_high_rl_low`: rule-based high-level interaction + RL low-level control
3. `method3_learned_high_classical_low`: RL learns high-level interaction decisions, low-level uses classical path tracking
4. `method4_full_rl`: both high-level and low-level are learned

Current `gnn_marl_training` has now been extended to support method 3.

## What Changes in Method 3

- `action_mode=interaction_mode`
- Policy output is no longer continuous `[v, w]`
- Policy outputs one of four high-level modes:
  - `go`
  - `yield`
  - `wait`
  - `backoff`
- Low-level control is classical tracking control
- The environment converts the selected mode into a tracking behavior and then into `cmd_vel`
- High-level mode now has `mode_hold_steps`, so the selected mode is not allowed to switch every 0.1s
- Observation for `interaction_mode` now uses a dedicated high-level feature layout
- Reward for `interaction_mode` now uses a dedicated high-level decision reward branch
- `replan` is no longer a learned interaction action; it stays as an environment-side recovery/navigation mechanism

So in method 3:

- RL learns **interaction protocol selection**
- classical controller executes **motion tracking**

## Training Entry

Main training entry:

- [train_gnn_mappo_full.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py)

Relevant option:

- `--action_mode interaction_mode`

When `--ppo_profile auto` is used, this mode will automatically use the discrete PPO profile.

## Core Code Paths

### High-level learned action space

- action mode parser:
  - [train_gnn_mappo_full.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py:1161)
- PPO auto profile dispatch:
  - [train_gnn_mappo_full.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py:961)
- run tag:
  - [train_gnn_mappo_full.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py:1397)

### Environment side

- learned interaction modes:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:1888)
- action space becomes discrete:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:2018)
- high-level mode hold:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:2905)
- observation branch for learned interaction:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:3005)
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:5345)
- learned mode to target/controller:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:3655)
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:4309)
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:4344)
- action execution bypasses old rule-based interaction shield in method 3:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:4428)
- dedicated high-level reward branch:
  - [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:4782)

### Metrics

New metrics for method 3:

- `policy_interaction_mode_id`
- `executed_behavior_mode_id`

Logged in:

- [train_gnn_mappo_full.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py:76)
- [gnn_marl_env.py](/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training/gnn_marl_training/gnn_marl_env.py:4784)

## Recommended Training Command

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training && \
./run_curriculum.sh \
  --model_type gat \
  --gat_actor_graph local_risk \
  --gat_critic_mode mlp \
  --num_agents 4 \
  --num_workers 1 \
  --start_stage 2 \
  --end_stage 2 \
  --train_steps 250000 \
  --train_batch_size 5000 \
  --checkpoint_freq 5000 \
  --action_mode interaction_mode \
  --ppo_profile auto \
  --interaction_mode_hold_steps 6 \
  --counterfactual_advantage_coef 0.08 \
  --progress_reward_scale 1.2 \
  --path_progress_reward_scale 0.35 \
  --goal_progress_reward_scale 3.0 \
  --goal_reward 24.0 \
  --collision_penalty 25.0 \
  --time_penalty 0.001 \
  --close_obstacle_penalty_scale 0.12 \
  --close_obstacle_dist 0.50 \
  --predictive_social_penalty_scale 0.08 \
  --predictive_front_penalty_scale 0.10 \
  --social_proximity_risk_scale 0.18 \
  --risk_aware_forward_penalty_scale 0.10 \
  --head_on_avoidance_reward_scale 0.25 \
  --team_reward_lambda 0.65 \
  --risk_gate_soft 0.18 \
  --risk_gate_hard 0.65 \
  --avoidance_low_risk_scale 0.12 \
  --navigation_high_risk_scale 0.92 \
  --time_penalty_risk_relax 0.85 \
  --high_conflict_mode mixed \
  --high_conflict_prob 0.35 \
  --headless_sim \
  --disable_rviz \
  --run_suffix method3_interaction_stage2
```

## Recommended Test Command

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training && \
./run_test.sh \
  -c /home/wj/work/multi-robot-exploration-rl/ray_results/method3_interaction_stage2/GNN_MAPPO_Stage1_Interact_method3_interaction_stage2_EnvStage2/best \
  --num_episodes 5
```

## What to Watch

Compared with method 2, method 3 should be evaluated using:

- success rate
- collision rate
- deadlock rate
- average episode length
- `policy_interaction_mode_id`
- `executed_behavior_mode_id`
- `interaction_mode_reward`
- `interaction_mode_penalty`

The key question is not only whether reward increases, but whether the learned high-level policy can produce better:

- yielding timing
- passing priority decisions
- backoff decisions
- interaction mode stability

than the current rule-based high-level controller.
