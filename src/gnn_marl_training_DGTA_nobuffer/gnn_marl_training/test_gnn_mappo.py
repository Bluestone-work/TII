# #!/usr/bin/env python3
# """
# GNN-MAPPO 测试脚本
# """
# import ray
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.tune.registry import register_env
# from ray.rllib.models import ModelCatalog
# import argparse
# import os
# import numpy as np
# import time

# from gnn_marl_training.gnn_marl_env import env_creator, GNNMARLEnv
# from gnn_marl_training.gat_rllib_model import GATRLlibModel, MODEL_NAME


# def main():
#     parser = argparse.ArgumentParser(description="GNN-MAPPO Testing")
#     parser.add_argument("--checkpoint_path", type=str, required=True, help="模型Checkpoint路径")
#     parser.add_argument("--num_agents", type=int, default=2, help="机器人数量")
#     parser.add_argument("--num_episodes", type=int, default=5, help="测试回合数")
#     parser.add_argument("--communication_range", type=float, default=3.5, help="通信范围(米), 与LDS-01一致")
#     parser.add_argument("--enable_neighbor_obs", type=bool, default=True, help="启用邻居观测")
#     parser.add_argument("--explore", action="store_true", default=False,
#                         help="测试时加入探索噪声（模型训练不足时可用）")
#     parser.add_argument("--diag_steps", type=int, default=5,
#                         help="运行前 N 步打印动作/速度详细诊断信息，0=关闭")
#     args = parser.parse_args()
    
#     # 转换为绝对路径
#     checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    
#     if not os.path.exists(checkpoint_path):
#         print(f"❌ 找不到checkpoint: {checkpoint_path}", flush=True)
#         return
    
#     print(f"{'='*80}", flush=True)
#     print(f"🧪 GNN-MAPPO 测试", flush=True)
#     print(f"{'='*80}", flush=True)
#     print(f"Checkpoint: {checkpoint_path}", flush=True)
#     print(f"机器人数量: {args.num_agents}", flush=True)
#     print(f"测试回合: {args.num_episodes}", flush=True)
#     print(f"{'='*80}\n", flush=True)
    
#     # 初始化 Ray
#     print("🔧 初始化 Ray...", flush=True)
#     if ray.is_initialized():
#         ray.shutdown()
#     ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
#     print("✅ Ray 初始化完成", flush=True)
    
#     # 注册环境和模型
#     print("📦 注册环境和模型...", flush=True)
#     register_env("gnn_marl", env_creator)
#     ModelCatalog.register_custom_model(MODEL_NAME, GATRLlibModel)
#     print("✅ 注册完成", flush=True)
    
#     # 重建配置
#     policy_name = "shared_policy"
#     env_config = {
#         "num_agents": args.num_agents,
#         "map_number": 3,
#         "max_episode_steps": 1000,
#         "communication_range": args.communication_range,
#         "enable_neighbor_obs": args.enable_neighbor_obs,
#         "enable_local_map": False,
#         # 与训练侧对齐的避碰参数（减少 train/test 行为漂移）
#         "collision_penalty": 35.0,
#         "near_collision_dist": 0.45,
#         "near_collision_penalty_scale": 1.2,
#         "front_safety_dist": 0.55,
#         "front_safety_penalty_scale": 0.8,
#         "neighbor_safety_dist": 0.45,
#         "neighbor_safety_penalty_scale": 1.0,
#         "shield_enable": True,
#         "shield_front_slow_dist": 0.60,
#         "shield_front_stop_dist": 0.26,
#         "shield_neighbor_slow_dist": 0.45,
#         "turn_in_place_front_dist": 0.40,
#         "env_log_level": "WARNING",
#     }
    
#     config = (
#         PPOConfig()
#         .environment(
#             env="gnn_marl",
#             env_config=env_config,
#             disable_env_checking=True
#         )
#         .framework("torch")
#         .resources(num_gpus=0)
#         .env_runners(
#             num_env_runners=0,
#             batch_mode="truncate_episodes"  # 纯推理模式下对结果无影响
#         )
#         .multi_agent(
#             policies={policy_name: (None, None, None, {})},
#             policy_mapping_fn=lambda agent_id, **kwargs: policy_name,
#             policies_to_train=[policy_name],
#         )
#         .training(
#             model={
#                 "custom_model": MODEL_NAME,
#                 "custom_model_config": {
#                     "num_agents": args.num_agents,
#                     "max_neighbors": min(args.num_agents - 1, 5),
#                     "hidden_dim": 128,
#                     "gat_hidden_dim": 128,
#                     "lstm_hidden_dim": 256,
#                     "n_gat_heads": 4,
#                 },
#                 "use_lstm": False,
#                 "max_seq_len": 20,
#             }
#         )
#         .api_stack(
#             enable_rl_module_and_learner=False,
#             enable_env_runner_and_connector_v2=False
#         )
#     )
    
#     # 加载模型
#     print("📥 构建算法配置...", flush=True)
#     algo = config.build()
#     print(f"📥 从 checkpoint 加载模型...", flush=True)
#     print(f"   路径: {checkpoint_path}", flush=True)
#     algo.restore(checkpoint_path)
#     print("✅ 模型加载成功\n", flush=True)
#     print("🌍 创建测试环境...", flush=True)
#     env = GNNMARLEnv(env_config)
#     print("✅ 环境创建成功\n", flush=True)
    
#     # 测试
#     episode_stats = []
    
#     try:
#         for ep in range(args.num_episodes):
#             print(f"{'='*80}", flush=True)
#             print(f"🎬 Episode {ep + 1}/{args.num_episodes}", flush=True)
            
#             obs_dict, _ = env.reset()
#             dones = {"__all__": False}
            
#             total_rewards = {f"agent_{i}": 0.0 for i in range(args.num_agents)}
#             successes     = {f"agent_{i}": 0   for i in range(args.num_agents)}  # 本 episode 到达目标次数
#             collisions    = {f"agent_{i}": 0   for i in range(args.num_agents)}  # 本 episode 碰撞次数
#             step_count = 0
            
#             # 【重要】初始化LSTM状态
#             states = {aid: algo.get_policy(policy_name).get_initial_state() 
#                      for aid in [f"agent_{i}" for i in range(args.num_agents)]}
            
#             while not dones["__all__"] and step_count < 1000:
#                 action_dict = {}

#                 for aid in [f"agent_{i}" for i in range(args.num_agents)]:
#                     if aid in obs_dict:
#                         action, state_out, _ = algo.compute_single_action(
#                             observation=obs_dict[aid],
#                             state=states[aid],
#                             policy_id=policy_name,
#                             explore=args.explore,
#                         )
#                         action_dict[aid] = action
#                         states[aid] = state_out

#                         # 诊断输出：前 N 步打印动作值与对应的实际速度，确认令牌是否正常
#                         if args.diag_steps > 0 and step_count < args.diag_steps:
#                             lin_vel = (float(action[0]) + 1.0) / 2.0 * 0.22
#                             ang_vel = float(action[1]) * 1.0
#                             print(
#                                 f"  [diag ep={ep+1} step={step_count+1}] {aid}"
#                                 f"  raw_action=[{action[0]:.3f}, {action[1]:.3f}]"
#                                 f"  → linear={lin_vel:.3f}m/s  angular={ang_vel:.3f}rad/s",
#                                 flush=True,
#                             )
                
#                 obs_dict, rewards, dones, truncateds, infos = env.step(action_dict)
#                 step_count += 1
                
#                 for aid, r in rewards.items():
#                     total_rewards[aid] += r

#                     if aid in infos:
#                         event = infos[aid].get('event', '')
#                         if event == 'goal':
#                             successes[aid] += 1
#                             print(f"\u2705 {aid} 到达目标 #第{successes[aid]}次 (step={step_count})", flush=True)
#                         elif event == 'collision':
#                             collisions[aid] += 1
#                             print(f"💥 {aid} 碰撞 #第{collisions[aid]}次 (step={step_count})", flush=True)
                
#                 if step_count % 50 == 0:
#                     print(f"Step {step_count}...", end="\r")
            
#             # 统计
#             total_success   = sum(successes.values())
#             total_collision = sum(collisions.values())

#             print(f"\n{'='*80}", flush=True)
#             print(f"📊 Episode {ep + 1} 结果:", flush=True)
#             print(f"   总步数: {step_count}", flush=True)
#             print(f"   到达目标: {total_success} 次", flush=True)
#             print(f"   碰撞: {total_collision} 次", flush=True)
#             for aid in [f"agent_{i}" for i in range(args.num_agents)]:
#                 print(f"   {aid}: 成功 {successes[aid]}次 / 碰撞 {collisions[aid]}次", flush=True)
#             print(f"   平均奖励/agent: {sum(total_rewards.values())/args.num_agents:.2f}", flush=True)
#             print(f"{'='*80}\n", flush=True)

#             episode_stats.append({
#                 'steps':           step_count,
#                 'total_success':   total_success,
#                 'total_collision': total_collision,
#                 'avg_reward':      sum(total_rewards.values()) / args.num_agents,
#             })
            
#             time.sleep(1.0)
        
#         # 总体统计
#         print(f"\n{'='*80}", flush=True)
#         print(f"📈 总体测试结果 ({args.num_episodes} Episodes)", flush=True)
#         print(f"{'='*80}", flush=True)
#         avg_steps    = sum(s['steps']           for s in episode_stats) / len(episode_stats)
#         avg_reward   = sum(s['avg_reward']      for s in episode_stats) / len(episode_stats)
#         all_success  = sum(s['total_success']   for s in episode_stats)
#         all_collision= sum(s['total_collision'] for s in episode_stats)
#         nav_total    = args.num_episodes * args.num_agents * (args.max_episode_steps if hasattr(args, 'max_episode_steps') else 1000)
#         print(f"   到达目标次数: {all_success}  ({all_success/args.num_episodes:.1f} 次/episode)", flush=True)
#         print(f"   碰撞次数:     {all_collision}  ({all_collision/args.num_episodes:.1f} 次/episode)", flush=True)
#         print(f"   平均 episode 步数: {avg_steps:.1f}", flush=True)
#         print(f"   平均奖励/agent:  {avg_reward:.2f}", flush=True)
#         print(f"{'='*80}\n", flush=True)
        
#     except KeyboardInterrupt:
#         print("\n🛑 测试中断", flush=True)
#     finally:
#         env.close()
#         ray.shutdown()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
GNN-MAPPO 终极测试脚本 (全自动克隆训练配置，杜绝权重失效)
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=UserWarning, module="ray")

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import argparse
import os
import numpy as np
import time

from gnn_marl_training.gnn_marl_env import env_creator, GNNMARLEnv
from gnn_marl_training.gat_rllib_model import GATRLlibModel, MODEL_NAME as MODEL_NAME_GAT
from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel, MODEL_NAME_MLP
from gnn_marl_training.counterfactual_ppo_policy import (
    CounterfactualPPOTorchPolicy,  # noqa: F401
    register_counterfactual_policy,
)
register_counterfactual_policy()


def _decode_action_for_diag(agent_env, action):
    try:
        return agent_env._decode_action_to_cmd_vel(action)
    except Exception:
        if getattr(agent_env, "action_mode", "") == "continuous":
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            a_lin = float(np.clip(arr[0], -1.0, 1.0))
            a_ang = float(np.clip(arr[1], -1.0, 1.0))
            if a_lin >= 0.0:
                linear_vel = a_lin * float(agent_env.max_forward_vel)
            else:
                linear_vel = a_lin * float(agent_env.max_reverse_vel)
            angular_vel = a_ang * float(agent_env.max_angular_vel)
            return float(linear_vel), float(angular_vel)
        action_id = int(np.asarray(action).reshape(-1)[0]) if isinstance(action, np.ndarray) else int(action)
        primitive = agent_env.discrete_action_primitives[int(np.clip(action_id, 0, len(agent_env.discrete_action_primitives) - 1))]
        return float(primitive[0]), float(primitive[1])


def main():
    parser = argparse.ArgumentParser(description="GNN-MAPPO Auto-Test")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型Checkpoint路径")
    parser.add_argument("--num_episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--explore", action="store_true", default=False, help="开启探索噪声(打破狭窄走廊对称死锁)")
    parser.add_argument("--diag_steps", type=int, default=15, help="打印前N步诊断信息")
    parser.add_argument("--map_number", type=int, default=None, help="覆盖 checkpoint 中的 map_number，使全局规划与当前测试地图一致")
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None, help="覆盖动态障碍物数量")
    parser.add_argument("--obs_speed_scale", type=float, default=None, help="覆盖动态障碍物速度系数，实际 obs_speed=0.3*scale")
    parser.add_argument("--shield_enable", type=int, default=None, help="可选：覆盖 checkpoint 中的 shield 开关 (0/1)")
    parser.add_argument("--shield_front_slow_dist", type=float, default=None)
    parser.add_argument("--shield_front_stop_dist", type=float, default=None)
    parser.add_argument("--shield_neighbor_slow_dist", type=float, default=None)
    parser.add_argument("--shield_linear_slow", type=float, default=None)
    parser.add_argument("--shield_linear_stop", type=float, default=None)
    parser.add_argument("--shield_turn_bias", type=float, default=None)
    parser.add_argument("--turn_in_place_front_dist", type=float, default=None)
    parser.add_argument("--turn_in_place_angle_thresh", type=float, default=None)
    parser.add_argument("--turn_in_place_w", type=float, default=None)
    parser.add_argument("--auto_reset_agents", type=int, default=None,
                        help="可选：覆盖 checkpoint 中的 auto_reset_agents (0/1)")
    args = parser.parse_args()
    
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到 checkpoint: {checkpoint_path}")
        return
    
    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)

    # 注册环境与模型
    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)

    print(f"{'='*80}")
    print("正在读取 Checkpoint 配置...")
    print(f"{'='*80}")

    # 直接从 algorithm_state.pkl 读 config dict（避免触发 Algorithm.from_checkpoint
    # 创建 env_runner actor 与 ROS2 daemon 冲突），再用 Policy.from_checkpoint
    # 单独加载 policy 权重。
    import pickle as _pickle
    algo_state_pkl = os.path.join(checkpoint_path, 'algorithm_state.pkl')
    with open(algo_state_pkl, 'rb') as _f:
        _algo_state = _pickle.load(_f)
    trained_config = _algo_state['config']
    env_config = dict(trained_config.get('env_config', {}))
    model_cfg = trained_config.get('model', {})
    custom_model_cfg = dict(model_cfg.get('custom_model_config', {}))

    policy_path = os.path.join(checkpoint_path, 'policies', 'shared_policy')
    policy = Policy.from_checkpoint(policy_path)

    num_agents = env_config.get("num_agents", 2)
    action_mode = env_config.get("action_mode", "discrete_primitive")
    model_type_used = model_cfg.get("custom_model", "<unknown>")

    print(f"✅ 成功克隆训练架构:")
    print(f"   - 模型结构: {model_type_used}")
    print(f"   - 机器人数量: {num_agents}")
    print(f"   - 动作模式: {action_mode}")
    graph_ablation = custom_model_cfg.get("graph_ablation", "<unset>")
    obstacle_token_dim = custom_model_cfg.get("obstacle_motion_dim", "<unset>")
    social_token_dim = custom_model_cfg.get("neighbor_prediction_dim", "<unset>")
    print(f"   - 图消融: {graph_ablation}")
    print(f"   - social token dim: {social_token_dim}")
    print(f"   - obstacle token dim: {obstacle_token_dim}")
    print(f"{'='*80}\n")
    
    # 默认保留 checkpoint 中的环境语义，只在显式传参时覆盖。
    env_config["env_log_level"] = "WARNING"
    if args.auto_reset_agents is not None:
        env_config["auto_reset_agents"] = bool(int(args.auto_reset_agents))
    if args.shield_enable is not None:
        env_config["shield_enable"] = bool(int(args.shield_enable))
    if args.shield_front_slow_dist is not None:
        env_config["shield_front_slow_dist"] = float(args.shield_front_slow_dist)
    if args.shield_front_stop_dist is not None:
        env_config["shield_front_stop_dist"] = float(args.shield_front_stop_dist)
    if args.shield_neighbor_slow_dist is not None:
        env_config["shield_neighbor_slow_dist"] = float(args.shield_neighbor_slow_dist)
    if args.shield_linear_slow is not None:
        env_config["shield_linear_slow"] = float(args.shield_linear_slow)
    if args.shield_linear_stop is not None:
        env_config["shield_linear_stop"] = float(args.shield_linear_stop)
    if args.shield_turn_bias is not None:
        env_config["shield_turn_bias"] = float(args.shield_turn_bias)
    if args.turn_in_place_front_dist is not None:
        env_config["turn_in_place_front_dist"] = float(args.turn_in_place_front_dist)
    if args.turn_in_place_angle_thresh is not None:
        env_config["turn_in_place_angle_thresh"] = float(args.turn_in_place_angle_thresh)
    if args.turn_in_place_w is not None:
        env_config["turn_in_place_w"] = float(args.turn_in_place_w)

    # 关键修复：测试时以 run_test.sh 当前阶段参数为准，避免 checkpoint 历史 map_number 造成规划-仿真不一致
    if args.map_number is not None:
        env_config["map_number"] = int(args.map_number)
    if args.num_dynamic_obstacles is not None:
        env_config["num_dynamic_obstacles"] = int(args.num_dynamic_obstacles)
    if args.obs_speed_scale is not None:
        scale = float(args.obs_speed_scale)
        env_config["obs_speed"] = 0.3 * scale

    print("测试环境关键参数:")
    print(f"   - map_number: {env_config.get('map_number')}")
    print(f"   - num_dynamic_obstacles: {env_config.get('num_dynamic_obstacles')}")
    print(f"   - obs_speed: {env_config.get('obs_speed')}")
    print(f"   - auto_reset_agents: {env_config.get('auto_reset_agents')}")
    print(f"   - shield_enable: {env_config.get('shield_enable')}")
    
    print("创建测试环境...")
    env = GNNMARLEnv(env_config)
    
    discrete_primitives = [
        (0.00, 0.00), (0.08, 0.00), (0.14, 0.00), (0.20, 0.00),
        (0.08, 0.55), (0.08, -0.55), (0.00, 1.00), (0.00, -1.00), (-0.06, 0.00)
    ]

    episode_stats = []
    
    try:
        for ep in range(args.num_episodes):
            print(f"{'='*80}")
            print(f"Episode {ep + 1}/{args.num_episodes}")
            
            obs_dict, _ = env.reset()
            dones = {"__all__": False}
            
            total_rewards = {f"agent_{i}": 0.0 for i in range(num_agents)}
            successes     = {f"agent_{i}": 0   for i in range(num_agents)}  
            collisions    = {f"agent_{i}": 0   for i in range(num_agents)}  
            step_count = 0
            min_dists = []
            social_risks = []
            front_risks = []
            reached_agents = set()
            collided_agents = set()
            truncated_episode = False
            
            # 安全初始化 LSTM 状态
            states = {aid: policy.get_initial_state() for aid in [f"agent_{i}" for i in range(num_agents)]}
            
            while not dones["__all__"] and step_count < env_config.get("max_episode_steps", 1000):
                action_dict = {}

                for aid in [f"agent_{i}" for i in range(num_agents)]:
                    if aid in obs_dict and aid not in env.dones:
                        # 执行推理
                        action, state_out, _ = policy.compute_single_action(
                            obs=obs_dict[aid],
                            state=states[aid],
                            explore=args.explore,
                        )
                        action_dict[aid] = action
                        states[aid] = state_out

                        # 诊断输出
                        if args.diag_steps > 0 and step_count < args.diag_steps:
                            agent_env = env.agents[aid]
                            sector_metrics = agent_env._scan_sector_metrics()
                            front_min = float(sector_metrics.get("front_min", agent_env.scan_max_range))
                            left_min = float(sector_metrics.get("left_min", agent_env.scan_max_range))
                            right_min = float(sector_metrics.get("right_min", agent_env.scan_max_range))
                            pred = getattr(agent_env, "_last_predictive_metrics", {}) or {}
                            lin_vel, ang_vel = _decode_action_for_diag(agent_env, action)
                            if action_mode == 'continuous':
                                act_str = f"[{float(action[0]):.3f}, {float(action[1]):.3f}]"
                            else:
                                action_id = int(action)
                                act_str = f"{action_id}"

                            print(
                                f"  [diag] {aid} front={front_min:.2f} left={left_min:.2f} right={right_min:.2f}"
                                f" social_risk={float(pred.get('social_risk', 0.0)):.3f}"
                                f" front_risk={float(pred.get('front_risk', 0.0)):.3f}"
                                f" | raw={act_str} -> v={lin_vel:.3f}, w={ang_vel:.3f}"
                            )
                
                obs_dict, rewards, dones, truncateds, infos = env.step(action_dict)
                step_count += 1
                
                for aid, r in rewards.items():
                    total_rewards[aid] += r
                    if aid in infos:
                        event = infos[aid].get('event', '')
                        if 'min_dist' in infos[aid]:
                            try:
                                min_dists.append(float(infos[aid]['min_dist']))
                            except Exception:
                                pass
                        if 'predictive_social_risk' in infos[aid]:
                            try:
                                social_risks.append(float(infos[aid]['predictive_social_risk']))
                            except Exception:
                                pass
                        if 'predictive_front_risk' in infos[aid]:
                            try:
                                front_risks.append(float(infos[aid]['predictive_front_risk']))
                            except Exception:
                                pass
                        if event == 'goal':
                            successes[aid] += 1
                            reached_agents.add(aid)
                            print(f"✅ {aid} 到达目标 (step={step_count})")
                        elif event == 'collision':
                            collisions[aid] += 1
                            collided_agents.add(aid)
                            print(f"碰撞: {aid} (step={step_count})")
                if any(bool(v) for v in truncateds.values()):
                    truncated_episode = True
                
                if step_count % 50 == 0:
                    print(f"Step {step_count}...", end="\r")
            
            total_success   = sum(successes.values())
            total_collision = sum(collisions.values())
            ep_summary = {
                'steps': step_count, 'total_success': total_success,
                'total_collision': total_collision, 'avg_reward': sum(total_rewards.values()) / num_agents,
                'min_dist': min(min_dists) if min_dists else float("nan"),
                'avg_social_risk': float(np.mean(social_risks)) if social_risks else 0.0,
                'avg_front_risk': float(np.mean(front_risks)) if front_risks else 0.0,
                'reached_agents': len(reached_agents),
                'collided_agents': len(collided_agents),
                'truncated': truncated_episode,
            }
            episode_stats.append(ep_summary)
            print(
                f"  [episode-summary] steps={ep_summary['steps']} "
                f"avg_reward={ep_summary['avg_reward']:.2f} "
                f"success={ep_summary['total_success']} collision={ep_summary['total_collision']} "
                f"reached_agents={ep_summary['reached_agents']} collided_agents={ep_summary['collided_agents']} "
                f"min_dist={ep_summary['min_dist']:.3f} "
                f"avg_social_risk={ep_summary['avg_social_risk']:.3f} "
                f"avg_front_risk={ep_summary['avg_front_risk']:.3f} "
                f"truncated={ep_summary['truncated']}"
            )
            
        print(f"\n{'='*80}")
        print(f"总体测试结果 ({args.num_episodes} Episodes)")
        print(f"{'='*80}")
        avg_steps    = sum(s['steps'] for s in episode_stats) / len(episode_stats)
        all_success  = sum(s['total_success'] for s in episode_stats)
        all_collision= sum(s['total_collision'] for s in episode_stats)
        avg_reward = sum(s['avg_reward'] for s in episode_stats) / len(episode_stats)
        valid_min_dists = [s['min_dist'] for s in episode_stats if not np.isnan(s['min_dist'])]
        avg_min_dist = float(np.mean(valid_min_dists)) if valid_min_dists else float("nan")
        avg_social_risk = sum(s['avg_social_risk'] for s in episode_stats) / len(episode_stats)
        avg_front_risk = sum(s['avg_front_risk'] for s in episode_stats) / len(episode_stats)
        total_reached_agents = sum(s['reached_agents'] for s in episode_stats)
        total_collided_agents = sum(s['collided_agents'] for s in episode_stats)
        truncated_eps = sum(1 for s in episode_stats if s['truncated'])
        print(f"   总计到达: {all_success} 次  |  总计碰撞: {all_collision} 次")
        print(f"   平均回报: {avg_reward:.2f}  |  平均步数: {avg_steps:.1f}")
        print(f"   平均最小间距: {avg_min_dist:.3f} m")
        print(f"   平均社交风险: {avg_social_risk:.3f}  |  平均前向风险: {avg_front_risk:.3f}")
        print(f"   到达过目标的 agent 数: {total_reached_agents}  |  发生过碰撞的 agent 数: {total_collided_agents}")
        print(f"   被时间截断的 Episode: {truncated_eps}/{len(episode_stats)}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n测试中断")
    finally:
        env.close()
        ray.shutdown()

if __name__ == "__main__":
    main()
