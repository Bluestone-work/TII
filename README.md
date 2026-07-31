# DGR-VDS Multi-Robot Navigation

Code-only release for multi-robot navigation training, evaluation, and baseline comparison.

This repository does not include trained models, RLlib checkpoints, raw datasets, generated figures, experiment logs, or paper result tables.

## Contents

```text
src/
  gnn_marl_training_DGTA_nobuffer/   # DGR-VDS/MAPPO training and evaluation
  start_rl_environment_tb3/          # ROS2/Gazebo TurtleBot3 simulation
  start_orca_nav/                    # ORCA baseline controller
  gnn_bc_tools/                      # ORCA/DWA collection and BC baseline tools
scripts/                            # evaluation, aggregation, diagnostics
experiments/                         # example launchers
docs/                               # reproducibility notes
```

## Requirements

- Ubuntu 22.04
- ROS2 Humble
- Gazebo Classic
- Python 3.10
- TurtleBot3 ROS packages
- PyTorch and Ray/RLlib

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build the workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Training example

Headless Map4-to-Map9 curriculum:

```bash
bash experiments/train_map4_map9_headless.sh
```

Defaults:

- Map 4 for up to 300k environment steps.
- Resume the latest Map 4 checkpoint.
- Map 9 for up to 300k environment steps.
- Safety shield disabled during training.
- Outputs under `artifacts/ray_results/`.

## Evaluation example

Pass an external RLlib checkpoint:

```bash
CKPT=/path/to/checkpoint \
NUM_EPISODES=100 \
MAPS="4 9" \
bash scripts/run_cross_map_paper_eval.sh
```

Results are written under `artifacts/eval/`.

## License

The ROS packages declare Apache-2.0 in package metadata. Confirm the final repository-level license file before public release.
