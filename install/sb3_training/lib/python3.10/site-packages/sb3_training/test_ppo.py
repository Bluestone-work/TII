#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np

import rclpy
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from sb3_training.independent_env import IndependentRobotEnv

def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description='SB3 RecurrentPPO 多智能体测试')
    parser.add_argument('--robot_number', type=int, default=3, help='机器人数量 (默认: 3)')
    parser.add_argument('--map_number', type=int, default=3, help='地图编号 (默认: 3)')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.zip)')
    parser.add_argument('--random_mode', action='store_true', help='是否使用随机位置 (默认: 固定起点)')
    
    # 使用 parse_known_args 避免与 ROS2 参数 (--ros-args) 冲突
    args, unknown = parser.parse_known_args()
    
    # 2. 初始化 ROS2
    if not rclpy.ok():
        rclpy.init()
    
    print(f"\n{'='*80}")
    print(f"🔥 启动多智能体测试")
    print(f"{'='*80}")
    print(f"🤖 机器人数量: {args.robot_number}")
    print(f"🗺️ 地图编号: {args.map_number}")
    print(f"📂 模型路径: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 找不到模型文件 {args.model_path}")
        return

    # 3. 创建多智能体环境 (DummyVecEnv)
    # 这里的逻辑与 train_ppo.py 保持一致
    env_fns = []
    for i in range(args.robot_number):
        def make_env(rank=i):
            return IndependentRobotEnv(
                robot_id=rank,
                map_number=args.map_number,
                max_episode_steps=1000, # 测试时给更多时间观察
                use_random_mode=args.random_mode
            )
        env_fns.append(make_env)

    # 创建向量化环境
    # 这会为每个机器人创建一个 IndependentRobotEnv 实例
    # 并且会自动应用我们刚才修改的颜色可视化和路径清除逻辑
    env = DummyVecEnv(env_fns)
    
    # 4. 加载模型
    try:
        # 注意：不需要手动指定 device，SB3 会自动处理
        model = RecurrentPPO.load(args.model_path, env=env)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        env.close()
        return

    print("\n🚀 开始测试... (按 Ctrl+C 停止)")
    
    # 5. 测试主循环
    obs = env.reset()
    
    # LSTM 状态初始化
    # state 初始为 None，RecurrentPPO 会自动处理
    lstm_states = None
    
    # 记录每个环境是否刚开始 (用于重置 LSTM 内部状态)
    # 初始时所有环境都是刚开始
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    try:
        while rclpy.ok():
            # 获取动作
            # deterministic=True: 测试时通常使用确定性策略（不加噪声），表现更稳定
            # predict 会一次性返回所有机器人的动作
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            
            # 执行动作
            # DummyVecEnv 会自动调用所有子环境的 step，并返回组合好的数据
            obs, rewards, dones, infos = env.step(action)
            
            # 更新 episode_start 标记
            # 如果某个环境 done 了，VecEnv 会自动 reset 它，
            # 下一次 predict 时对应的 LSTM 状态需要被重置，所以 episode_starts = dones
            episode_starts = dones
            
            # 打印完成信息 (可选)
            for i, done in enumerate(dones):
                if done:
                    # 获取结束时的信息
                    # VecEnv 会把 done 时的 info 放在 infos[i]['terminal_observation'] 里
                    # 但通常 infos[i] 也会包含最后一步的信息
                    event = infos[i].get('event', 'unknown')
                    print(f"🤖 Robot {i} Episode 结束 ({event}).")
            
            # 仿真环境下通常不需要 sleep，因为 env.step 内部有速率控制
            # 但如果你觉得 print 刷屏太快，可以取消下面的注释
            # time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n🛑 测试停止")
    finally:
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()