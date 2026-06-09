#!/usr/bin/env python3
"""
使用Stable-Baselines3的RecurrentPPO训练单个机器人
每个机器人独立训练，将其他机器人视为动态障碍物
"""
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# ── 自动注入工作空间路径（无需 source install/setup.bash）──────────────────
def _inject_paths():
    repo_root = Path(__file__).resolve().parents[3]
    # Python 包路径
    for p in [
        repo_root / "src"  / "sb3_training",
        repo_root / "build"/ "sb3_training",
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    # AMENT_PREFIX_PATH（让 get_package_share_directory 能找到本地包）
    install_dir = repo_root / "install"
    if install_dir.exists():
        ws_prefixes = [
            str(p) for p in install_dir.iterdir()
            if p.is_dir() and not p.name.startswith('_') and p.name != 'COLCON_IGNORE'
        ]
        existing = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
        merged = ws_prefixes + [p for p in existing if p not in ws_prefixes]
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(merged)

_inject_paths()

import rclpy
from rclpy.node import Node

# Stable-Baselines3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 导入自定义环境
from sb3_training.independent_env import IndependentRobotEnv


class TensorboardCallback(BaseCallback):
    """
    自定义回调：记录训练指标到TensorBoard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 检查episode是否结束
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    # 记录episode奖励和长度
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        self.logger.record('rollout/ep_reward', ep_reward)
                        self.logger.record('rollout/ep_length', ep_length)
                        
                        # 如果有奖励分量，也记录
                        if 'reward_components' in info:
                            comp = info['reward_components']
                            for key, value in comp.items():
                                if key != 'total' and not key.startswith('obs_'):
                                    self.logger.record(f'reward/{key}', value)
        
        return True


class MultiRobotTrainingNode(Node):
    """
    多机器人训练节点
    使用RecurrentPPO为每个机器人训练独立策略
    """
    
    def __init__(self, args):
        super().__init__('sb3_ppo_training')
        
        self.args = args
        self.robot_number = args.robot_number
        self.map_number = args.map_number
        self.declare_parameter('max_episode_steps', 500)
        # 创建保存目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            '/home/wj/work/multi-robot-exploration-rl/sb3_models',
            f'ppo_map{self.map_number}_robots{self.robot_number}_{self.timestamp}'
        )
        self.max_episode_steps = self.get_parameter('max_episode_steps').value
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"模型保存路径: {self.save_dir}")
        
        # 创建日志目录
        self.log_dir = os.path.join(
            '/home/wj/work/multi-robot-exploration-rl/sb3_logs',
            f'ppo_map{self.map_number}_robots{self.robot_number}_{self.timestamp}'
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.get_logger().info(f"日志保存路径: {self.log_dir}")
        
        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"🚀 启动SB3 RecurrentPPO训练")
        self.get_logger().info(f"{'='*80}")
        self.get_logger().info(f"地图: {self.map_number}")
        self.get_logger().info(f"机器人数量: {self.robot_number}")
        self.get_logger().info(f"总训练步数: {args.total_timesteps:,}")
        self.get_logger().info(f"{'='*80}\n")
    
    def create_env(self):
        """创建训练环境"""
        # 每个机器人有自己的环境实例（但共享底层ROS环境）
        # env = IndependentRobotEnv(
        #     robot_id=0,  # 第一个机器人
        #     total_robots=self.robot_number,
        #     map_number=self.map_number,
        #     use_random_mode=self.args.random_mode,
        #     max_episode_steps=self.args.max_steps
        # )
        # env = IndependentRobotEnv(
        #     robot_id=0,
        #     map_number=self.map_number,
        #     max_episode_steps=self.max_episode_steps,
        #     use_random_mode=True  # 确保开启随机模式
        # )
        """创建多机器人并行的向量化环境"""
        env_fns = []

        # 为每个机器人创建一个独立的环境函数
        for i in range(self.robot_number):
            # 使用 lambda 捕获当前的 robot_id (i)
            # 注意：必须使用 i=i 默认参数，否则所有环境都会变成最后一个 ID
            def make_env(rank=i):
                return IndependentRobotEnv(
                    robot_id=rank,
                    map_number=self.map_number,
                    max_episode_steps=self.max_episode_steps,
                    use_random_mode=True
                )
            env_fns.append(make_env)

        # 使用 DummyVecEnv 串行运行所有环境 (调试最稳定)
        # 如果想要更快，可以使用 SubprocVecEnv 并行运行 (但在 ROS2 中处理节点销毁较复杂)
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        
        # 推荐先用 DummyVecEnv，虽然是串行执行 step，但逻辑简单不容易报错
        env = DummyVecEnv(env_fns)

        return env
    
    def train(self):
        """训练主循环"""
        # 创建环境
        # self.get_logger().info("创建训练环境...")
        # env = self.create_env()
        self.get_logger().info(f"正在为 {self.robot_number} 个机器人创建并行环境...")
        env = self.create_env()
        
        # 创建或加载模型
        # if self.args.load_path and os.path.exists(self.args.load_path):
        #     self.get_logger().info(f"加载已有模型: {self.args.load_path}")
        #     model = RecurrentPPO.load(
        #         self.args.load_path,
        #         env=env,
        #         device=self.args.device
        #     )
        # else:
        #     self.get_logger().info("创建新的RecurrentPPO模型...")
        #     model = RecurrentPPO(
        #         "MlpLstmPolicy",
        #         env,
        #         learning_rate=self.args.learning_rate,
        #         n_steps=self.args.n_steps,
        #         batch_size=self.args.batch_size,
        #         n_epochs=self.args.n_epochs,
        #         gamma=self.args.gamma,
        #         gae_lambda=self.args.gae_lambda,
        #         clip_range=self.args.clip_range,
        #         ent_coef=self.args.ent_coef,
        #         vf_coef=self.args.vf_coef,
        #         max_grad_norm=self.args.max_grad_norm,
        #         verbose=1,
        #         tensorboard_log=self.log_dir,
        #         device=self.args.device
        #     )
        if self.args.load_path and os.path.exists(self.args.load_path):
            self.get_logger().info(f"加载已有模型: {self.args.load_path}")
            model = RecurrentPPO.load(
                self.args.load_path,
                env=env,
                device=self.args.device
            )
        else:
            self.get_logger().info("创建新的 PPO 模型 (Parameter Sharing)...")
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                learning_rate=self.args.learning_rate,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                n_epochs=self.args.n_epochs,
                gamma=self.args.gamma,
                gae_lambda=self.args.gae_lambda,
                clip_range=self.args.clip_range,
                ent_coef=self.args.ent_coef,
                vf_coef=self.args.vf_coef,
                max_grad_norm=self.args.max_grad_norm,
                verbose=1,
                tensorboard_log=self.log_dir,
                device=self.args.device
            )
        
        # 创建回调
        callbacks = []
        
        # 检查点回调：定期保存模型
        checkpoint_callback = CheckpointCallback(
            save_freq=self.args.save_freq,
            save_path=self.save_dir,
            name_prefix='ppo_model',
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        callbacks.append(checkpoint_callback)
        
        # TensorBoard回调
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)
        
        callback = CallbackList(callbacks)
        
        # 开始训练
        self.get_logger().info("\n" + "="*80)
        self.get_logger().info("🎯 开始训练...")
        self.get_logger().info("="*80 + "\n")
        
        try:
            model.learn(
                total_timesteps=self.args.total_timesteps,
                callback=callback,
                log_interval=10,
                reset_num_timesteps=not self.args.load_path
            )
            
            # 保存最终模型
            final_model_path = os.path.join(self.save_dir, 'final_model')
            model.save(final_model_path)
            self.get_logger().info(f"\n✅ 训练完成！最终模型保存至: {final_model_path}")
            
        except KeyboardInterrupt:
            self.get_logger().info("\n⚠️ 训练被中断")
            # 保存中断时的模型
            interrupt_model_path = os.path.join(self.save_dir, 'interrupted_model')
            model.save(interrupt_model_path)
            self.get_logger().info(f"模型已保存至: {interrupt_model_path}")
        
        finally:
            env.close()
            self.get_logger().info("环境已关闭")


def main(args=None):
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SB3 RecurrentPPO训练')
    
    # 环境参数
    parser.add_argument('--robot_number', '--num_agents', dest='robot_number',
                       type=int, default=1, 
                       help='机器人数量 (默认: 1)')
    parser.add_argument('--map_number', '--env_stage', dest='map_number',
                       type=int, default=3, 
                       help='地图编号 (默认: 3)')
    parser.add_argument('--random_mode', action='store_true', 
                       help='使用随机起始位置')
    parser.add_argument('--max_steps', type=int, default=1000, 
                       help='每个episode最大步数 (默认: 1000)')
    
    # 训练参数
    parser.add_argument('--total_timesteps', type=int, default=1000000, 
                       help='总训练步数 (默认: 1,000,000)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, 
                       help='学习率 (默认: 3e-4)')
    parser.add_argument('--n_steps', type=int, default=2048, 
                       help='每次更新的步数 (默认: 2048)')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='批次大小 (默认: 64)')
    parser.add_argument('--n_epochs', type=int, default=10, 
                       help='优化轮数 (默认: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='折扣因子 (默认: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, 
                       help='GAE lambda (默认: 0.95)')
    parser.add_argument('--clip_range', type=float, default=0.2, 
                       help='PPO裁剪范围 (默认: 0.2)')
    parser.add_argument('--ent_coef', type=float, default=0.01, 
                       help='熵系数 (默认: 0.01)')
    parser.add_argument('--vf_coef', type=float, default=0.5, 
                       help='价值函数系数 (默认: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, 
                       help='最大梯度范数 (默认: 0.5)')
    
    # 模型保存/加载
    parser.add_argument('--save_freq', type=int, default=10000, 
                       help='保存检查点频率 (默认: 10000)')
    parser.add_argument('--load_path', type=str, default=None, 
                       help='加载已有模型路径')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='训练设备')
    
    # 过滤ROS2参数（--ros-args之后的所有参数）
    import sys
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--ros-args':
            break
        if skip_next:
            skip_next = False
            continue
        filtered_args.append(arg)
    
    args = parser.parse_args(filtered_args)
    
    # 初始化ROS2
    rclpy.init(args=None)
    
    # 创建训练节点
    training_node = MultiRobotTrainingNode(args)
    
    try:
        # 开始训练
        training_node.train()
    finally:
        # 清理
        training_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
