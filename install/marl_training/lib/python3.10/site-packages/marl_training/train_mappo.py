#!/usr/bin/env python3
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import rclpy
import os
import argparse
from marl_training.marl_env import ROS2MAPPEnv

def env_creator(env_config):
    return ROS2MAPPEnv(env_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=1000000)
    parser.add_argument("--debug", action="store_true", help="启用调试输出")
    args = parser.parse_args()

    if ray.is_initialized(): ray.shutdown()
    ray.init() # 启动 Ray

    register_env("ros2_mappo", env_creator)
    
    # === MAPPO 核心配置 ===
    # 这里的关键是 multi_agent 配置，将所有 agent 映射到同一个 policy
    policy_name = "shared_policy" 
    
    config = (
        PPOConfig()
        .environment(
            env="ros2_mappo",
            env_config={
                "num_agents": args.num_agents,
                "map_number": 3,
                "max_episode_steps": 1000  # 从2000改为1000，与PPO对齐
            },
            disable_env_checking=True
        )
        .framework("torch")
        # 资源设置 (Ray 2.10+ 新版API)
        .env_runners(
            num_env_runners=args.num_workers, 
            num_envs_per_env_runner=1,
            sample_timeout_s=600
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            train_batch_size=4000,
            clip_param=0.2,
            entropy_coeff=0.01,
            # 模型网络结构 (LSTM)
            model={
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            }
        )
        # === 参数共享 (Parameter Sharing) ===
        # 这就是 MAPPO 的实现方式：所有机器人共用一个大脑
        .multi_agent(
            # 【修复】正确的 policies 配置格式
            policies={
                policy_name: (None, None, None, {})  # (policy_cls, obs_space, act_space, config)
            },
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            # 【关键】明确告诉 RLLib 所有可能的智能体
            policies_to_train=[policy_name],
        )
        # 兼容性设置：使用旧版 API 栈以获得最稳定的 PPO 表现
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        # 资源分配 (根据你电脑实际情况，没显卡写0)
        .resources(num_gpus=1)
    )
    
    # 显式设置 PPO 训练参数 (绕过新版API检查)
    config.sgd_minibatch_size = 256
    config.num_sgd_iter = 10

    print(f"🚀 开始 MAPPO 训练...")
    
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"timesteps_total": args.train_steps},
            storage_path=os.path.abspath("./ray_results"),
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True
            )
        ),
    )

    tuner.fit()
    ray.shutdown()

if __name__ == "__main__":
    main()