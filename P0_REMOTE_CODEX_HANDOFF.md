# P0 Remote Server Codex Handoff

Date: 2026-07-19

## Objective

Use this remote server to accelerate the paper's P0 architecture ablation. This
server is assigned **training seed 3** and must train all three methods under the
same frozen protocol:

1. `p0_graph_off`: MAPPO-LSTM with the graph residual disabled.
2. `p0_unified_graph`: robots and dynamic obstacles encoded by one
   parameter-matched Unified GAT.
3. `p0_dual_graph`: separate social and dynamic-obstacle GAT encoders.

For all methods, Value-difference shaping is disabled (`lambda_om = 0`). The
purpose is to isolate architecture effects, not shaping effects.

The local workstation is currently training seeds 1 and 2. Once this server has
successfully started seed 3, report that fact to the user so the local pending
seed-3 jobs can be removed. Do not intentionally run duplicate seed-3 chains.

## Frozen Training Protocol

- Seed assigned to this server: `3`.
- Four robots with a shared recurrent MAPPO actor.
- Map order: Map 4 then Map 9.
- Budget: 400,000 environment steps per map.
- Map 9 must exact-resume the Map 4 policy, optimizer, and training state.
- Early stopping disabled.
- Training safety filter disabled.
- `counterfactual_advantage_coef=0` for every method.
- No legacy checkpoint or historical output may be loaded.
- Use a fresh output tag and output root.
- Do not reduce steps, change rewards, change PPO hyperparameters, or patch
  frozen source merely to meet a deadline.
- Evaluation is not this server's primary assignment. Return final Map 9
  checkpoints and the training manifest to the main evaluation host.

## Input Files

The user will place these files on the remote server:

```text
p0_server_bundle_20260719.tar.gz
p0_server_bundle_20260719.tar.gz.sha256
```

Expected archive SHA-256:

```text
fd7a22bf6186a431e447cb57cb89f1c29f0e14218c4a19293319ddee689a2e4e
```

The bundle contains source and scenario assets only. It intentionally excludes
legacy checkpoints, historical results, `build/`, `install/`, and active logs.

## Remote Codex Checklist

### 1. Inspect the host before changing it

Record and report:

```bash
uname -a
cat /etc/os-release
free -h
df -h
nvidia-smi
loginctl show-user "$USER" -p Linger
```

Expected baseline:

- Ubuntu 22.04.
- NVIDIA GPU and working driver.
- At least 50 GB free disk.
- Prefer at least 16 GB available RAM for two concurrent chains.
- ROS 2 Humble and Gazebo Classic 11, or sudo access to install them.
- Anaconda/Miniconda, or permission to install a Python 3.10 conda environment.

Do not select parallelism from GPU memory alone. Gazebo and Ray consume
substantial host RAM. Use two-way parallelism unless the host has enough RAM and
the first two workers leave a large safety margin.

### 2. Verify and unpack the bundle

Run from the directory containing both files:

```bash
sha256sum --check p0_server_bundle_20260719.tar.gz.sha256
tar -xzf p0_server_bundle_20260719.tar.gz
cd p0_server_bundle_20260719
```

Stop and report if the archive checksum fails.

### 3. Install host dependencies when necessary

If ROS 2 Humble/Gazebo dependencies are absent and the ROS apt repository is
already configured:

```bash
bash install_system_deps.sh
```

This step may use sudo. Do not silently switch ROS distributions or Ubuntu
versions if installation fails.

### 4. Deploy and build

If a compatible conda environment named `ros2` already exists:

```bash
bash deploy.sh
```

If it does not exist:

```bash
CREATE_CONDA_ENV=1 bash deploy.sh
```

The default target is:

```text
$HOME/work/multi-robot-exploration-rl
```

`TARGET_ROOT=/another/path` is supported. The installer relocates path-only
constants, regenerates the frozen hash for the relocated copy, builds
`gnn_marl_training` and `start_rl_environment_tb3`, and runs a no-training
preflight.

Deployment acceptance criteria:

- Both ROS packages build successfully.
- Python is 3.10.
- Ray, Torch, Gymnasium, NumPy, and SciPy import successfully.
- `verify_p0_core_protocol.sh` passes.
- Unified/Dual active graph parameter gap is below 1%.
- Initial model hashes are produced for seeds 1, 2, and 3.
- Pipeline prints `training remains stopped` during preflight.

### 5. Ensure jobs survive SSH logout

The training launcher uses a systemd user service. Check user lingering:

```bash
loginctl show-user "$USER" -p Linger
```

If it reports `Linger=no`, enable it before starting the long run:

```bash
sudo loginctl enable-linger "$USER"
```

If sudo is unavailable, use a persistent `tmux` session and the launcher's
`--foreground` option instead.

### 6. Dry-run the assigned work

```bash
cd "$HOME/work/multi-robot-exploration-rl"
./server_tools/run_training.sh \
  --seeds 3 \
  --parallel 2 \
  --tag p0_remote_seed3_20260719 \
  --dry-run
```

The plan must state:

```text
seeds=[3]
maps=[4 9]
steps_per_map=400000
resume_across_maps=1
```

### 7. Start seed 3

```bash
./server_tools/run_training.sh \
  --seeds 3 \
  --parallel 2 \
  --tag p0_remote_seed3_20260719
```

The command prints the systemd unit name and master-log path. Do not launch a
second copy with a different tag.

### 8. Validate the first PPO iterations

Within the first 10-20 minutes, report:

- systemd unit state;
- active method/seed/map chains;
- latest environment steps for each chain;
- `training_fps`;
- available RAM and GPU memory;
- any Ray worker recovery, OOM, NaN, traceback, or repeated zero-progress rows.

Useful commands:

```bash
systemctl --user status 'p0-p0_remote_seed3_20260719.service'
journalctl --user -u 'p0-p0_remote_seed3_20260719.service' -n 100 --no-pager
tail -f curriculum_logs/p0_remote_seed3_20260719/master.log
find ray_results/p0_remote_seed3_20260719 -name training_monitor.csv -print
free -h
nvidia-smi
```

Reset-layout rejection warnings may occur while the environment resamples an
unsafe start. A traceback, worker death, OOM, NaN, or repeated lack of training
progress is not acceptable and must be reported.

### 9. Failure handling

- The runner supports checkpoint-based retries and exact partial resume.
- Do not delete a partial run without inspecting its checkpoint and logs.
- Do not import a checkpoint from another method, seed, or legacy experiment.
- Do not edit frozen model, environment, reward, map, or protocol files after a
  chain starts.
- If a source change appears necessary, stop and ask the user. All methods and
  affected seeds may need a clean restart for fairness.

### 10. Completion and result collection

The seed-3 assignment is complete only when all six training phases are marked
`trained` in the manifest:

```text
p0_graph_off seed3 Map4
p0_graph_off seed3 Map9
p0_unified_graph seed3 Map4
p0_unified_graph seed3 Map9
p0_dual_graph seed3 Map4
p0_dual_graph seed3 Map9
```

Then package results:

```bash
./server_tools/collect_results.sh p0_remote_seed3_20260719
```

Return both generated files:

```text
p0_remote_seed3_20260719_results_<timestamp>.tar.gz
p0_remote_seed3_20260719_results_<timestamp>.tar.gz.sha256
```

Before transfer, verify the result archive checksum and report the final six
manifest rows, checkpoint paths, completion timestamps, and source-hash file.

## Required Status Message Back to the User

After a healthy remote start, send a concise status containing:

```text
Remote seed 3 started successfully.
Methods: Graph-off, Unified Graph, Dual Graph.
Protocol: Map4 -> Map9, 400k steps per map, lambda_om=0, safety filter off.
Unit: <systemd unit>
Current chains: <method/seed/map and steps>
Training FPS: <per chain>
Available RAM/GPU memory: <values>
Errors: none / details
```

That confirmation is the handoff signal for removing pending seed-3 work from
the local workstation queue.
