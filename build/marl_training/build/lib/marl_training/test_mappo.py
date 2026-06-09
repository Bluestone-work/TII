#!/usr/bin/env python3
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import rclpy
import os
import argparse
import numpy as np
import time

# 导入你的环境
from marl_training.marl_env import ROS2MAPPEnv

def env_creator(env_config):
    return ROS2MAPPEnv(env_config)

def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description="ROS2 MAPPO Testing")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型 Checkpoint 路径 (例如: ~/ray_results/.../checkpoint_000100)")
    parser.add_argument("--num_agents", type=int, default=3, help="机器人数量")
    parser.add_argument("--num_episodes", type=int, default=5, help="测试多少个回合")
    args = parser.parse_args()

    # 【修复】将相对路径转换为绝对路径
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    
    # 检查路径
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型路径: {checkpoint_path}")
        print(f"   原始输入: {args.checkpoint_path}")
        return

    # 2. 初始化 Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=True) # 测试时建议用 local_mode (单进程)，调试方便且避免 ROS 通信延迟

    # 3. 注册环境
    register_env("ros2_mappo", env_creator)
    
    print(f"🔄 正在重建算法配置...")
    
    # 4. 重建配置 (必须与训练时完全一致!)
    # 我们只需要构建一个结构相同的空算法，然后把权重加载进去
    policy_name = "shared_policy"
    
    config = (
        PPOConfig()
        .environment(
            env="ros2_mappo",
            env_config={
                "num_agents": args.num_agents,
                "map_number": 3,
                "max_episode_steps": 1000 # 测试时可以给长一点时间
            },
            disable_env_checking=True
        )
        .framework("torch")
        # 资源设置 (测试时只需要 1 个 CPU 即可，不需要 GPU)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0) # 0 表示在主进程中运行环境
        .multi_agent(
            # 【关键】必须与训练时的格式完全一致
            policies={
                policy_name: (None, None, None, {})  # (policy_cls, obs_space, act_space, config)
            },
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            policies_to_train=[policy_name],
        )
        # === 关键：网络结构必须与训练一致 ===
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,        
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            }
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )

    # 5. 构建算法并加载权重
    print(f"📥 正在加载模型权重...")
    print(f"   路径: {checkpoint_path}")
    algo = config.build()
    algo.restore(checkpoint_path)
    print("✅ 模型加载成功！")

    # 6. 创建测试环境
    # 注意：必须使用与训练相同的环境配置
    env = ROS2MAPPEnv({
        "num_agents": args.num_agents,
        "map_number": 3,
        "max_episode_steps": 1000
    })
    
    print(f"\n{'='*60}")
    print(f"🎮 测试配置:")
    print(f"   机器人数量: {args.num_agents}")
    print(f"   测试回合数: {args.num_episodes}")
    print(f"   最大步数: 1000")
    print(f"{'='*60}\n")

    # 统计变量
    episode_stats = []
    
    try:
        for ep in range(args.num_episodes):
            print(f"\n{'='*60}")
            print(f"🎬 开始测试 Episode {ep + 1}/{args.num_episodes}")
            print(f"{'='*60}")
            
            # 重置环境
            obs_dict, _ = env.reset()
            dones = {"__all__": False}
            truncateds = {"__all__": False}
            
            # === LSTM 状态初始化 ===
            # 每个机器人的 LSTM 状态是独立的
            # 状态形状: [c_state, h_state]，每个大小为 lstm_cell_size
            lstm_cell_size = 256
            state_dict = {}
            for i in range(args.num_agents):
                aid = f"agent_{i}"
                # 初始化为全 0
                state_dict[aid] = [
                    np.zeros(lstm_cell_size, dtype=np.float32), 
                    np.zeros(lstm_cell_size, dtype=np.float32)
                ]
            
            # Episode 统计
            total_rewards = {f"agent_{i}": 0.0 for i in range(args.num_agents)}
            agent_completed = {f"agent_{i}": False for i in range(args.num_agents)}
            agent_collided = {f"agent_{i}": False for i in range(args.num_agents)}
            step_count = 0
            
            while not dones["__all__"]:
                action_dict = {}
                
                # 为每个智能体计算动作
                for aid in [f"agent_{i}" for i in range(args.num_agents)]:
                    if aid not in obs_dict:
                        continue
                        
                    # 计算单个动作
                    # compute_single_action 会处理观测归一化（如果有）并返回新动作和新状态
                    action, state_out, _ = algo.compute_single_action(
                        observation=obs_dict[aid],
                        state=state_dict[aid],  # 传入上一帧的 LSTM 状态
                        policy_id=policy_name,
                        explore=False           # <--- 测试时关闭探索 (Deterministic)
                    )
                    
                    action_dict[aid] = action
                    state_dict[aid] = state_out # 更新 LSTM 状态
                
                # 环境步进
                obs_dict, rewards, dones, truncateds, infos = env.step(action_dict)
                step_count += 1
                
                # 统计奖励和事件
                for aid, r in rewards.items():
                    total_rewards[aid] += r
                    
                    # 检查是否完成目标或碰撞
                    if aid in infos:
                        info = infos[aid]
                        if info.get('episode_complete'):
                            agent_completed[aid] = True
                            print(f"\n✅ {aid} 到达目标! (步数: {step_count})")
                        elif info.get('episode_incomplete'):
                            if info.get('event') == 'collision':
                                agent_collided[aid] = True
                                print(f"\n💥 {aid} 发生碰撞! (步数: {step_count})")
                    
                # 打印进度 (每 50 步)
                if step_count % 50 == 0:
                    active = len([aid for aid in obs_dict if aid.startswith('agent')])
                    print(f"Step {step_count}: 活跃智能体={active}/{args.num_agents}", end="\r")

            # Episode 结束统计
            print(f"\n{'='*60}")
            print(f"🏁 Episode {ep + 1} 结束!")
            print(f"   总步数: {step_count}")
            print(f"   结束原因: {'超时 (Truncated)' if truncateds.get('__all__') else '全部完成'}")
            print(f"{'='*60}")
            
            # 详细统计
            completed_count = sum(agent_completed.values())
            collided_count = sum(agent_collided.values())
            
            print(f"\n📊 智能体表现:")
            for i in range(args.num_agents):
                aid = f"agent_{i}"
                status = "✅ 完成" if agent_completed[aid] else ("💥 碰撞" if agent_collided[aid] else "⏱️ 超时")
                print(f"   {aid}: {status} | 总奖励 = {total_rewards[aid]:.2f}")
            
            print(f"\n📈 汇总:")
            print(f"   成功率: {completed_count}/{args.num_agents} ({100*completed_count/args.num_agents:.1f}%)")
            print(f"   碰撞数: {collided_count}")
            print(f"   平均奖励: {sum(total_rewards.values())/args.num_agents:.2f}")
            
            # 保存统计
            episode_stats.append({
                'episode': ep + 1,
                'steps': step_count,
                'completed': completed_count,
                'collided': collided_count,
                'avg_reward': sum(total_rewards.values())/args.num_agents,
                'total_rewards': total_rewards.copy()
            })
            
            time.sleep(2.0) # 休息一下方便观察
        
        # 打印总体统计
        print(f"\n{'='*60}")
        print(f"📊 总体测试结果 ({args.num_episodes} Episodes)")
        print(f"{'='*60}")
        avg_success = sum(s['completed'] for s in episode_stats) / len(episode_stats)
        avg_steps = sum(s['steps'] for s in episode_stats) / len(episode_stats)
        avg_reward = sum(s['avg_reward'] for s in episode_stats) / len(episode_stats)
        
        print(f"   平均成功率: {avg_success:.2f}/{args.num_agents} ({100*avg_success/args.num_agents:.1f}%)")
        print(f"   平均步数: {avg_steps:.1f}")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    finally:
        env.close()
        ray.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()