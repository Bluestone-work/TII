# gnn_bc_tools

独立于原有 `gnn_marl_training` 的 BC 工具包，提供：

- `collect_orca_dwa_bc`：用 ORCA/DWA teacher 采集 BC 数据。
- `pretrain_mappo_bc`：对 `MAPPOMLPModel` 做离线 BC 预训练。
- `run_orca_dwa_bc_pipeline`：自动执行「采集 -> BC 预训练 -> RL 微调」。

> 该包不会覆盖原有训练包，只新增在 `src/gnn_bc_tools`。

## Build

```bash
source /home/wj/anaconda3/etc/profile.d/conda.sh
conda activate ros2
source /opt/ros/humble/setup.bash
cd ~/work/multi-robot-exploration-rl
colcon build --packages-select gnn_bc_tools
source install/setup.bash
```

## 一键全流程（采集 -> BC -> RL）

```bash
ros2 run gnn_bc_tools run_orca_dwa_bc_pipeline \
  --env_stage 4 \
  --map_number 5 \
  --episodes 120 \
  --enable_visualization_collect \
  --enable_visualization_rl \
  --rl_train_steps 300000 \
  --rl_num_workers 2 \
  --rl_sample_timeout_s 1200 \
  --rl_rollout_fragment_length 200
```

## 分步执行

1) 采集 ORCA/DWA 专家数据

```bash
ros2 run gnn_bc_tools collect_orca_dwa_bc \
  --env_stage 4 \
  --map_number 5 \
  --episodes 120 \
  --enable_visualization
```

2) BC 预训练

```bash
ros2 run gnn_bc_tools pretrain_mappo_bc \
  --dataset_path ~/work/multi-robot-exploration-rl/bc_datasets/your_dataset.npz \
  --epochs 30 \
  --device auto
```

3) 跳过采集/BC，直接 RL 微调

```bash
ros2 run gnn_bc_tools run_orca_dwa_bc_pipeline \
  --skip_collect \
  --dataset_path ~/work/multi-robot-exploration-rl/bc_datasets/your_dataset.npz \
  --skip_bc \
  --bc_weights_path ~/work/multi-robot-exploration-rl/bc_models/your_weights.pt \
  --enable_visualization_rl
```

## 小地图参数搜索（Map1）

```bash
ros2 run gnn_bc_tools tune_orca_dwa_map1 \
  --episodes 20 \
  --num_trials 16 \
  --num_agents 4
```

仓库根目录还提供了一键脚本（自动启动 map1 环境）：

```bash
./run_bc_orca_dwa_tune_map1.sh --episodes 20 --num_trials 16 --num_agents 4
```
