# gnn_bc_tools

Utilities for ORCA/DWA expert rollout collection, behavior cloning, and RL fine-tuning baselines.

## Build

```bash
colcon build --packages-select gnn_bc_tools
source install/setup.bash
```

## Examples

Collect expert rollouts:

```bash
ros2 run gnn_bc_tools collect_orca_dwa_bc \
  --env_stage 4 \
  --map_number 5 \
  --episodes 120
```

Behavior cloning pretraining:

```bash
ros2 run gnn_bc_tools pretrain_mappo_bc \
  --dataset_path ./artifacts/bc_datasets/example.npz \
  --epochs 30 \
  --device auto
```

Pipeline wrapper:

```bash
ros2 run gnn_bc_tools run_orca_dwa_bc_pipeline \
  --env_stage 4 \
  --map_number 5 \
  --episodes 120 \
  --rl_train_steps 300000
```
