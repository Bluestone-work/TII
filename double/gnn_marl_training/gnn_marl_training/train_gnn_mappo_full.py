#!/usr/bin/env python3
"""
MAPPO 训练脚本（逐步构建版本）

--model_type mlp  : 基础 MAPPO-MLP-LSTM（无 GNN，带 LSTM）— 用于先跑通 MAPPO
--model_type gat  : GNN-MAPPO（GAT + LSTM）+ 课程学习
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import argparse
import os
import sys
import warnings
from pathlib import Path
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']  = '0'   # 消除 Ray GPU FutureWarning
os.environ['RAY_DISABLE_METRICS_COLLECTION']       = '1'   # 彻底关闭 metrics 采集（解决 rpc_code:14）
os.environ['RAY_DISABLE_IMPORT_METRICS_REPORTER']  = '1'   # 禁用 Prometheus metrics exporter
os.environ['RAY_metrics_export_port']              = '0'   # 不绑定 metrics 端口
warnings.filterwarnings('ignore', category=FutureWarning, module='ray')


def _inject_workspace_paths():
    repo_root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        repo_root / "src" / "gnn_marl_training",
        repo_root / "build" / "gnn_marl_training",
        repo_root / "build" / "gnn_marl_training" / "build" / "lib",
    ]
    inserted_paths = []
    for path in reversed(candidate_paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted_paths.insert(0, path_str)

    py_path_entries = inserted_paths + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    deduped = []
    for entry in py_path_entries:
        if entry not in deduped:
            deduped.append(entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(deduped)

    # ── 修复：将本地工作空间 install/ 下所有包前缀加入 AMENT_PREFIX_PATH ──
    # Ray RolloutWorker 是独立子进程，不会继承 `source install/setup.bash`
    # 必须显式注入，否则 get_package_share_directory() 只能搜索 /opt/ros/humble
    install_dir = repo_root / "install"
    if install_dir.exists():
        ws_prefixes = [
            str(p) for p in install_dir.iterdir()
            if p.is_dir() and not p.name.startswith('_') and p.name != 'COLCON_IGNORE'
        ]
        existing_ament = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
        ament_entries = ws_prefixes + [p for p in existing_ament if p not in ws_prefixes]
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(ament_entries)

    return os.environ["PYTHONPATH"], os.environ.get("AMENT_PREFIX_PATH", "")


WORKSPACE_PYTHONPATH, WORKSPACE_AMENT_PATH = _inject_workspace_paths()

from gnn_marl_training.gnn_marl_rllib_env import env_creator
from gnn_marl_training.gat_rllib_model   import GATRLlibModel,  MODEL_NAME       as MODEL_NAME_GAT
from gnn_marl_training.mappo_mlp_model   import MAPPOMLPModel,  MODEL_NAME_MLP


ENV_CURRICULUM = {
    1: {
        "name": "Stage 1 · 静态入门",
        "map_number": 3,          # corridor_swap，与 run_curriculum.sh STAGE_MAP_NUM 保持同步
        "max_episode_steps": 1000,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "description": "corridor_swap + 无动态障碍，先学会朝 waypoint 平滑前进。",
    },
    2: {
        "name": "Stage 2 · 静态变长",
        "map_number": 3,          # corridor_swap，与 run_curriculum.sh STAGE_MAP_NUM 保持同步
        "max_episode_steps": 1000,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "description": "corridor_swap 更长路径与更多转弯，强化 waypoint 跟踪。",
    },
    3: {
        "name": "Stage 3 · 慢速动态障碍",
        "map_number": 3,
        "max_episode_steps": 1000,
        "num_obstacles": 4,
        "obs_speed_scale": 0.5,
        "description": "走廊交换地图 + 少量慢速动态障碍。",
    },
    4: {
        "name": "Stage 4 · 完整任务",
        "map_number": 3,
        "max_episode_steps": 1000,
        "num_obstacles": 8,
        "obs_speed_scale": 1.0,
        "description": "完整 corridor_swap 难度，对齐最终训练环境。",
    },
}


def _extract_reward(result):
    for val in [
        result.get("episode_reward_mean"),
        result.get("env_runners", {}).get("episode_reward_mean"),
        result.get("sampler_results", {}).get("episode_reward_mean"),
    ]:
        if val is not None:
            return val
    return float("nan")


def _extract_timesteps(result):
    for val in [
        result.get("timesteps_total"),
        result.get("num_env_steps_sampled_lifetime"),
        result.get("info", {}).get("num_env_steps_sampled"),
    ]:
        if val is not None:
            return val
    return 0


def _build_launch_command(stage_cfg, num_agents):
    return (
        "ros2 launch start_rl_environment_tb3 main.launch.py "
        f"map_number:={stage_cfg['map_number']} robot_number:={num_agents} "
        f"num_obstacles:={stage_cfg['num_obstacles']} "
        f"obs_speed_scale:={stage_cfg['obs_speed_scale']:.1f}"
    )


def _run_training(config, run_name, train_steps, checkpoint_freq, storage_path, init_checkpoint=None):
    run_dir = os.path.join(storage_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    algo = config.build()
    if init_checkpoint:
        print(f"🔄 从 checkpoint 迁移权重: {init_checkpoint}")
        donor = config.build()
        donor.restore(init_checkpoint)
        algo.set_weights(donor.get_weights())
        donor.stop()

    best_reward = float("-inf")
    best_ckpt = None
    last_ckpt = None
    start_time = __import__('time').time()
    iteration = 0

    while True:
        iteration += 1
        result = algo.train()
        done_steps = _extract_timesteps(result)
        reward = _extract_reward(result)
        pct = min(100.0, done_steps / max(train_steps, 1) * 100)
        elapsed = __import__('time').time() - start_time
        if done_steps > 0:
            eta_s = elapsed / done_steps * max(train_steps - done_steps, 0)
            eta_str = f"  ETA {int(eta_s//3600):02d}:{int(eta_s%3600//60):02d}:{int(eta_s%60):02d}"
        else:
            eta_str = ""
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  {done_steps:>8,} / {train_steps:,} 步"
            f"  回报 {reward:7.2f}{eta_str}",
            end="",
            flush=True,
        )

        if reward > best_reward:
            best_reward = reward
            best_ckpt = algo.save(os.path.join(run_dir, "best"))

        if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
            last_ckpt = algo.save(run_dir)

        if done_steps >= train_steps:
            break

    print()
    last_ckpt = algo.save(run_dir)
    algo.stop()

    ckpt = best_ckpt or last_ckpt
    print(f"  平均回报: {best_reward:.2f}")
    print(f"  训练步数: {train_steps:,}")
    print(f"  最佳 Checkpoint: {ckpt}")
    return ckpt


def _build_ppo_config(args, env_config, model_name, model_cfg):
    """根据参数构建 PPOConfig。"""
    policy_name = "shared_policy"
    config = (
        PPOConfig()
        .environment(
            env="gnn_marl",
            env_config=env_config,
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            sample_timeout_s=600,
            batch_mode="complete_episodes",
        )
        .training(
            lr=args.lr,
            gamma=0.99,
            lambda_=0.95,
            train_batch_size=args.train_batch_size,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=50.0,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            minibatch_size=256,
            num_epochs=10,
            model=model_cfg,
        )
        .multi_agent(
            policies={policy_name: (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            policies_to_train=[policy_name],
        )
        .resources(
            num_gpus=int(__import__('torch').cuda.is_available())
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    return config, policy_name


def _run_tune(config, run_name, train_steps, checkpoint_freq, storage_path, init_checkpoint=None):
    """兼容旧调用名；内部改为支持权重迁移的手动训练循环。"""
    return _run_training(config, run_name, train_steps, checkpoint_freq, storage_path, init_checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MAPPO 训练脚本（逐步构建版）\n"
            "  --model_type mlp : 基础 MAPPO-MLP-LSTM，先跑通再加 GNN\n"
            "  --model_type gat : GNN-MAPPO + 课程学习（需先用 mlp 验证）"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # ── 通用参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--model_type",       type=str,   default="mlp",
                        choices=["mlp", "gat"],
                        help="mlp = 纯MLP+LSTM基础MAPPO（推荐先用这个）\n"
                             "gat = GAT+LSTM的GNN-MAPPO（验证完mlp再用）")
    parser.add_argument("--num_agents",       type=int,   default=3)
    parser.add_argument("--communication_range", type=float, default=3.5,
                        help="通信范围(米)，仅 gat 模式下有意义")
    parser.add_argument("--num_workers",      type=int,   default=2)
    parser.add_argument("--train_steps",      type=int,   default=500000)
    parser.add_argument("--checkpoint_freq",  type=int,   default=20)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--train_batch_size", type=int,   default=4000)
    parser.add_argument("--hidden_dim",       type=int,   default=256,
                        help="MLP/LSTM 隐藏层维度（mlp 模式）")
    parser.add_argument("--env_stage",        type=int,   default=1,
                        choices=sorted(ENV_CURRICULUM.keys()),
                        help="环境课程学习阶段；切换阶段后需按提示重新 ros2 launch 环境")

    # ── GAT 专用：课程学习 ────────────────────────────────────────────────
    parser.add_argument("--curriculum_stage", type=int,   default=1,
                        choices=[1, 2],
                        help="[仅 gat 模式]\n"
                             "  1 = 独立导航阶段 (comm_dropout=1.0)\n"
                             "  2 = 协作通信阶段 (从阶段1 checkpoint 加载)")
    parser.add_argument("--resume_checkpoint", type=str,  default=None,
                        help="[仅 gat 模式阶段2] 阶段1 checkpoint 路径")
    parser.add_argument("--comm_mode",          type=str,   default="decentralized",
                        choices=["decentralized", "centralized_oracle", "ros2_bridge"])
    parser.add_argument("--comm_dropout_prob",  type=float, default=0.05)
    parser.add_argument("--comm_latency_steps", type=int,   default=1)
    parser.add_argument("--comm_jitter_steps",  type=int,   default=1)
    parser.add_argument("--comm_noise_std",     type=float, default=0.05)
    args = parser.parse_args()

    stage_cfg = ENV_CURRICULUM[args.env_stage]
    launch_cmd = _build_launch_command(stage_cfg, args.num_agents)
    resume_path = None
    if args.resume_checkpoint:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_checkpoint))
        if not os.path.exists(resume_path):
            parser.error(f"--resume_checkpoint 路径不存在: {resume_path}")

    # ── 初始化 Ray ────────────────────────────────────────────────────────
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": WORKSPACE_PYTHONPATH,
                "AMENT_PREFIX_PATH": WORKSPACE_AMENT_PATH,
            }
        },
        _system_config={
            "metrics_report_interval_ms": 999999999,  # 禁用定时 metrics 上报
        },
    )
    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)

    # ════════════════════════════════════════════════════════════════════
    #  MLP 模式：基础 MAPPO-LSTM，单阶段训练，无通信
    # ════════════════════════════════════════════════════════════════════
    if args.model_type == "mlp":
        print(f"{'='*80}")
        print(f"🧱 MAPPO-MLP-LSTM 基础训练（无 GNN，带 LSTM）")
        print(f"{'='*80}")
        print(f"  机器人数量:  {args.num_agents}")
        print(f"  环境阶段:    {args.env_stage} · {stage_cfg['name']}")
        print(f"  并行Workers: {args.num_workers}")
        print(f"  训练步数:    {args.train_steps:,}")
        print(f"  学习率:      {args.lr}")
        print(f"  隐藏维度:    {args.hidden_dim}")
        print(f"  Actor 输入:  148  (本体 lidar(36×4帧)+target(2)+vel(2))")
        print(f"  Critic 输入: {args.num_agents * 148}  ({args.num_agents} 个机器人全局状态)")
        print(f"  阶段描述:    {stage_cfg['description']}")
        print(f"  需要环境:    {launch_cmd}")
        print(f"{'='*80}\n")

        env_config = {
            "num_agents":          args.num_agents,
            "map_number":          stage_cfg['map_number'],
            "max_episode_steps":   stage_cfg['max_episode_steps'],
            "communication_range": args.communication_range,
            # MLP 模式：关闭邻居观测，obs = [local(40)] + [reset_flag(1)] + [global(N*40)]
            "enable_neighbor_obs": False,
            "enable_local_map":    False,
            "comm_mode":           "decentralized",
            "comm_dropout_prob":   1.0,   # 无意义，但保持接口一致
            "comm_latency_steps":  1,
            "comm_jitter_steps":   0,
            "comm_noise_std":      0.0,
            # 统一语义：单个机器人终止后立刻局部 reset，团队 episode 仅在超时结束
            "auto_reset_agents":   True,
            # 动态障碍物参数（当前由 Gazebo Actor 驱动，保持接口一致）
            "num_dynamic_obstacles": stage_cfg['num_obstacles'],
            "obs_speed":           0.3 * stage_cfg['obs_speed_scale'],
        }
        model_cfg = {
            "custom_model": MODEL_NAME_MLP,
            "custom_model_config": {
                "num_agents": args.num_agents,
                "hidden_dim": args.hidden_dim,
            },
            # 策略级 LSTM：每条轨迹被切成长度 20 的序列送入训练
            # 20 步 × ~0.1s/步 ≈ 2 秒，足以覆盖动态障碍物一次穿越过程
            "max_seq_len": 20,
        }
        config, _ = _build_ppo_config(args, env_config, MODEL_NAME_MLP, model_cfg)
        storage_path = os.path.abspath("./ray_results")
        if args.env_stage > 1 and not resume_path:
            print("⚠️ 当前为高阶段课程学习，但未提供 --resume_checkpoint，将从随机初始化开始。\n")
        ckpt = _run_tune(config, f"MAPPO_MLP_LSTM_Stage{args.env_stage}", args.train_steps,
                         args.checkpoint_freq, storage_path, init_checkpoint=resume_path)
        print(f"\n{'='*80}")
        print(f"✅ MLP-LSTM 训练完成！最佳 Checkpoint: {ckpt}")
        if args.env_stage < max(ENV_CURRICULUM):
            next_stage = args.env_stage + 1
            next_cfg = ENV_CURRICULUM[next_stage]
            next_launch = _build_launch_command(next_cfg, args.num_agents)
            ckpt_path = ckpt.path if hasattr(ckpt, "path") else str(ckpt)
            print("   下一阶段请先重启环境：")
            print(f"   {next_launch}")
            print("   然后继续训练：")
            print(f"   python build/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py --model_type mlp \\")
            print(f"     --env_stage {next_stage} --resume_checkpoint {ckpt_path} \\")
            print(f"     --num_agents {args.num_agents} --num_workers {args.num_workers}")
        else:
            print(f"   验证成功后可切换到 GNN 模式：--model_type gat")
        print(f"{'='*80}\n")

    # ════════════════════════════════════════════════════════════════════
    #  GAT 模式：GNN-MAPPO + 课程学习
    # ════════════════════════════════════════════════════════════════════
    else:
        # 参数校验
        if args.curriculum_stage == 2:
            if not resume_path:
                parser.error("--curriculum_stage 2 需要指定 --resume_checkpoint")

        effective_dropout = 1.0 if args.curriculum_stage == 1 else args.comm_dropout_prob
        stage_name = "阶段1 (独立导航)" if args.curriculum_stage == 1 else "阶段2 (协作通信)"
        print(f"{'='*80}")
        print(f"🚀 GNN-MAPPO 课程学习训练  [{stage_name}]")
        print(f"{'='*80}")
        print(f"  环境阶段:     {args.env_stage} · {stage_cfg['name']}")
        print(f"  机器人数量:   {args.num_agents}")
        print(f"  通信范围:     {args.communication_range} m")
        print(f"  并行Workers:  {args.num_workers}")
        print(f"  训练步数:     {args.train_steps:,}")
        print(f"  需要环境:     {launch_cmd}")
        print(f"  comm_dropout: {effective_dropout:.0%}"
              f"  {'← 强制100%丢包，等同单机' if args.curriculum_stage == 1 else ''}")
        if resume_path:
            print(f"  恢复自:       {resume_path}")
        print(f"{'='*80}\n")

        env_config = {
            "num_agents":          args.num_agents,
            "map_number":          stage_cfg['map_number'],
            "max_episode_steps":   stage_cfg['max_episode_steps'],
            "communication_range": args.communication_range,
            "enable_neighbor_obs": True,   # 保留邻居槽位，保证两阶段 obs 维度一致
            "enable_local_map":    False,
            "comm_mode":           args.comm_mode,
            "comm_dropout_prob":   effective_dropout,
            "comm_latency_steps":  args.comm_latency_steps,
            "comm_jitter_steps":   args.comm_jitter_steps,
            "comm_noise_std":      args.comm_noise_std,
            # 连续任务流：agent done 后在当前 episode 内立刻重置，适合课程学习
            "auto_reset_agents":   True,
            # 动态障碍物参数（当前由 Gazebo Actor 驱动，保持接口一致）
            "num_dynamic_obstacles": stage_cfg['num_obstacles'],
            "obs_speed":           0.3 * stage_cfg['obs_speed_scale'],
        }
        model_cfg = {
            "custom_model": MODEL_NAME_GAT,
            "custom_model_config": {
                "num_agents":    args.num_agents,
                "max_neighbors": min(args.num_agents - 1, 5),
                "hidden_dim":    128,
                "gat_hidden_dim": 128,
                "lstm_hidden_dim": 256,
                "n_gat_heads":   4,
            },
            "max_seq_len": 20,
        }
        config, _ = _build_ppo_config(args, env_config, MODEL_NAME_GAT, model_cfg)
        storage_path = os.path.abspath("./ray_results")
        run_name = f"GNN_MAPPO_Stage{args.curriculum_stage}"

        if args.curriculum_stage == 1:
            print("📚 阶段1：邻居消息全丢弃，先学会独立导航...\n")
            ckpt = _run_tune(config, f"{run_name}_EnvStage{args.env_stage}", args.train_steps,
                             args.checkpoint_freq, storage_path)
            ckpt_path = ckpt.path if hasattr(ckpt, "path") else str(ckpt)
            print(f"\n{'='*80}")
            print(f"✅ 阶段1 完成！下一步：")
            if args.env_stage < max(ENV_CURRICULUM):
                next_stage = args.env_stage + 1
                next_cfg = ENV_CURRICULUM[next_stage]
                next_launch = _build_launch_command(next_cfg, args.num_agents)
                print(f"   先重启环境：{next_launch}")
            print(f"   python train_gnn_mappo_full.py --model_type gat \\")
            print(f"     --env_stage {min(args.env_stage + 1, max(ENV_CURRICULUM))} --curriculum_stage 2 \\")
            print(f"     --resume_checkpoint {ckpt_path} \\")
            print(f"     --num_agents {args.num_agents} --num_workers {args.num_workers}")
            print(f"{'='*80}\n")
        else:
            print(f"🤝 阶段2：从阶段1 checkpoint 恢复，开放通信...\n")
            ckpt = _run_tune(config, f"{run_name}_EnvStage{args.env_stage}", args.train_steps,
                             args.checkpoint_freq, storage_path, init_checkpoint=resume_path)
            print(f"\n{'='*80}")
            print(f"✅ 阶段2 完成！最佳 Checkpoint: {ckpt}")
            print(f"{'='*80}\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
