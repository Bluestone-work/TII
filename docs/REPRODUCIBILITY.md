# Reproducibility Notes

## Included

- DGR-VDS/MAPPO training code.
- Policy and counterfactual/VDS components.
- ROS2/Gazebo TurtleBot3 simulation environment.
- Static maps and Gazebo worlds needed by the launch files.
- ORCA and ORCA/DWA baseline tooling.
- Evaluation and aggregation scripts.

## Not included

- Trained checkpoints or exported policy weights.
- `ray_results`, `paper_results`, `curriculum_logs`, `experiment_logs`.
- Raw behavior-cloning datasets.
- Rendered figures, videos, trajectory dumps, or generated paper artifacts.

## Recommended protocol

1. Build and source the ROS2 workspace.
2. Use headless Gazebo for training and large-scale evaluation.
3. Keep `--safety_filter_enable 0` during PPO training unless explicitly running a shield ablation.
4. Use fixed seeds for training and evaluation.
5. Store generated files under `artifacts/`.

## Metrics

- `SR`: success rate.
- `TSR`: team success rate.
- `CR`: collision rate. Check the table script for agent-level vs episode-level aggregation.
- `TTG`: time to goal.
- `minimum_clearance`: minimum observed clearance.
- `shield_intervention_rate`: fraction of control steps modified by the safety shield.
