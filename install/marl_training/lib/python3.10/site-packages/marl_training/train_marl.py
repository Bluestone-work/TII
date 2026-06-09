#!/usr/bin/env python3
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import rclpy
import os
import argparse

# 导入上面写的环境
from marl_training.marl_env import ROS2MAPPEnv

def env_creator(env_config):
    return ROS2MAPPEnv(env_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=3, help="并行采样的 Worker 数量")
    parser.add_argument("--num_agents", type=int, default=3)
    args = parser.parse_args()

    # 1. 初始化 Ray
    # local_mode=True 对于调试 ROS 非常有用，因为它不创建新进程，避免通讯死锁
    ray.init(local_mode=False) 

    # 2. 注册环境
    register_env("ros2_marl_env", env_creator)

    # 3. 配置算法 (以 MAPPO 为例，这里使用 RLLib 的 PPO 实现多智能体)
    # RLLib 的 PPO 天然支持多智能体（通过 Parameter Sharing）
    config = (
        PPOConfig()
        .environment(
            env="ros2_marl_env",
            env_config={
                "num_agents": args.num_agents,
                "map_number": 3,
                "max_episode_steps": 500
            }
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=args.num_workers, # 并行 Worker 数量
            num_envs_per_worker=1,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=2000,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=100.0,
            entropy_coeff=0.01,
        )
        # 多智能体设置：参数共享
        # 将所有 agent 映射到同一个 policy "shared_policy"
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        .resources(num_gpus=1) # 如果有 GPU
    )

    print("🚀 开始 MARL 训练...")

    # 4. 运行训练
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 1000},
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=10),
            storage_path=os.path.abspath("./marl_results")
        ),
    )
    
    results = tuner.fit()
    print("✅ 训练完成")
    ray.shutdown()

if __name__ == "__main__":
    main()