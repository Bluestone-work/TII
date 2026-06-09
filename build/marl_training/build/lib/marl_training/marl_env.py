import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import rclpy

# === 关键：直接复用你现有的环境逻辑 ===
# 确保你已经编译过 marl_training (colcon build)
from marl_training.independent_env import IndependentRobotEnv

class ROS2MAPPEnv(MultiAgentEnv):
    """
    适配 MAPPO 的多智能体环境包装器
    核心逻辑：实例化 N 个 IndependentRobotEnv，并统一管理
    """
    def __init__(self, env_config):
        super().__init__()
        
        # 1. 解析参数
        # 注意：RLLib 新版要求 num_agents 是只读属性，所以我们用 num_robots 存变量
        self.num_robots = env_config.get("num_agents", 3)
        self.map_number = env_config.get("map_number", 3)
        self.max_steps = env_config.get("max_episode_steps", 500)
        self._env_config = env_config  # 保存完整配置，供创建子环境使用
        
        # 2. 初始化 ROS2
        if not rclpy.ok():
            rclpy.init()
            
        # 3. 创建底层单体环境列表
        self.agents = {}
        self.agent_ids = []
        
        for i in range(self.num_robots):
            aid = f"agent_{i}"
            self.agent_ids.append(aid)
            # 这里直接使用了你原有的 IndependentRobotEnv
            # 所有的奖励计算(reward)、观测(obs)、动作(action)定义都来自它
            # 【关键】多智能体环境中collision_ends_episode=False避免轨迹中断
            # 单个agent碰撞不应结束整个episode，否则LSTM会报错
            self.agents[aid] = IndependentRobotEnv(
                robot_id=i,
                map_number=self.map_number,
                max_episode_steps=self.max_steps,
                use_random_mode=True,
                collision_ends_episode=False,  # 多智能体：碰撞只扣分不结束
                num_dynamic_obstacles=self._env_config.get('num_dynamic_obstacles', 8),
                obs_speed=self._env_config.get('obs_speed', 0.3),
            )
            
        # 4. 定义空间（假设所有机器人同构）
        # RLLib 需要知道 observation_space 和 action_space
        ref_env = self.agents[self.agent_ids[0]]
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space
        
        # 【修复弃用警告】使用新版 API
        self._agent_ids = set(self.agent_ids)  # 旧版兼容
        self.possible_agents = self.agent_ids.copy()  # 新版 API (list)
        self._possible_agents = set(self.agent_ids)  # 新版 API (set)
        
        # 状态记录
        self.dones = set()
        self.current_step_count = 0

    def get_agent_ids(self):
        """【兼容旧版 API】返回智能体 ID 列表"""
        return self.possible_agents
    
    def reset(self, *, seed=None, options=None):
        # 【关键】清空已完成集合
        self.dones = set()
        self.current_step_count = 0
        obs_dict = {}
        infos = {}
        
        print(f"\n{'='*60}")
        print(f"🔄 环境重置 - {self.num_robots} 个机器人")
        print(f"   已完成集合已清空: {self.dones}")
        print(f"{'='*60}")
        
        # 重置所有机器人
        for aid, env in self.agents.items():
            obs, info = env.reset(seed=seed)
            obs_dict[aid] = obs
            infos[aid] = info
            print(f"✅ {aid} 已重置, current_step={env.current_step}")
        
        # 【验证】确保 dones 仍然为空
        assert len(self.dones) == 0, f"Reset 后 dones 应该为空，但包含: {self.dones}"
        print(f"✓ 验证通过: 所有机器人都是活跃状态\n")

        # 随机化动态障碍物（每 episode 不同，增强避障训练）
        master_env = self.agents[self.agent_ids[0]]
        if hasattr(master_env, 'randomize_obstacles'):
            all_positions = [
                [self.agents[aid].current_pose['x'], self.agents[aid].current_pose['y']]
                for aid in self.agent_ids
            ]
            master_env.randomize_obstacles(all_positions)

        # 【修复】RLlib 要求 reset 也返回空的 truncateds（虽然不常用）
        # 某些版本的 RLlib 会检查这个，即使是在 reset 时
        return obs_dict, infos

    def step(self, action_dict):
        self.current_step_count += 1
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        truncated_dict = {}
        info_dict = {}
        
        # 【调试】检查哪些智能体收到了动作
        # 只在有缺失或有完成时打印
        if len(action_dict) != self.num_robots or len(self.dones) > 0:
            missing = set(self.agent_ids) - set(action_dict.keys())
            received = list(action_dict.keys())
            print(f"⚠️  Step {self.current_step_count}: 收到 {len(action_dict)}/{self.num_robots} 动作, 已完成: {len(self.dones)}/{self.num_robots}")
            if missing:
                print(f"   缺失动作: {missing}")
            if self.dones:
                print(f"   已完成: {self.dones}")
        
        # 1. 【群发指令】所有活跃机器人同时发布速度
        # 【关键修复】必须遍历所有智能体，而不只是 action_dict 中的
        for aid in self.agent_ids:
            if aid not in self.dones:
                if aid in action_dict:
                    # 收到动作：执行该动作
                    self.agents[aid].apply_action(action_dict[aid])
                else:
                    # 【修复】没收到动作：发布停止命令，避免机器人"卡住"
                    self.agents[aid]._publish_vel(0.0, 0.0)
        
        # 2. 【并行等待】统一推进时间，并刷新所有节点的传感器数据
        # 必须确保所有机器人节点都 spin，否则拿不到最新的 Scan/Odom
        self._wait_and_spin_all(0.1)

        # 3. 【群收数据】计算所有机器人的奖励和状态
        # 【关键修复】必须为所有智能体返回观测，不管它们是否收到动作
        for aid in self.agent_ids:
            if aid not in self.dones:
                if aid in action_dict:
                    # 正常情况：收到动作的智能体，计算完整的奖励和状态
                    obs, rew, done, truncated, info = self.agents[aid].get_step_result()
                    
                    obs_dict[aid] = obs
                    rew_dict[aid] = rew
                    info_dict[aid] = info
                    
                    # 【关键修复】检测是否需要单独重置（碰撞或到达目标）
                    if info.get('need_reset', False) or done:
                        # 碰撞或到达目标：记录到dones集合，但不返回done=True给RLlib
                        # 这样避免LSTM trajectory中断问题
                        if aid not in self.dones:
                            self.dones.add(aid)
                            event = info.get('event', 'collision' if info.get('need_reset') else 'goal')
                            print(f"\n🏁 {aid} 达到终止条件: {event}")
                            print(f"   步数: {self.current_step_count}/{self.max_steps}")
                            print(f"   奖励: {rew:.2f}")
                            print(f"   碰撞距离: {info.get('min_dist', 'N/A')}")
                            print(f"   目标距离: {info.get('final_dist', 'N/A')}\n")
                        
                        # 【关键】不返回done=True，而是继续返回观测
                        # 这样LSTM trajectory保持连续，直到整个episode结束
                        done_dict[aid] = False
                        truncated_dict[aid] = False
                        
                        # 立即重置该机器人（生成新目标）
                        reset_obs, reset_info = self.agents[aid].reset()
                        obs_dict[aid] = reset_obs
                        # 奖励仍然给予（碰撞惩罚或到达奖励）
                    else:
                        done_dict[aid] = False
                        truncated_dict[aid] = False
                else:
                    # 【关键修复】没收到动作的智能体，必须返回完整的状态
                    # 否则 RLLib 会认为它已经 done，下次就不给动作了！
                    obs_dict[aid] = self.agents[aid]._get_obs()
                    rew_dict[aid] = 0.0
                    done_dict[aid] = False  # 明确标记为未完成！
                    truncated_dict[aid] = False
                    info_dict[aid] = {'warning': 'no_action_received'}
            else:
                # 【修复】已完成的智能体：继续返回观测但不标记为done
                # 避免LSTM trajectory中断
                obs_dict[aid] = self.agents[aid]._get_obs()
                rew_dict[aid] = 0.0
                done_dict[aid] = False  # 改为False，保持trajectory连续
                truncated_dict[aid] = False
                info_dict[aid] = {'status': 'waiting'}

        # 4. 【静止逻辑】已完成的机器人维持静止
        for aid in self.agent_ids:
            if aid in self.dones:
                self.agents[aid]._publish_vel(0.0, 0.0)

        # 5. 全局终止
        all_done = (len(self.dones) == self.num_robots)
        timeout = (self.current_step_count >= self.max_steps)
        
        done_dict["__all__"] = all_done or timeout
        truncated_dict["__all__"] = timeout  # 【修复】必须包含 __all__ 键
        
        # 【关键】只有在整个episode结束时，才标记所有智能体为done
        # 这样确保LSTM trajectory完整且一致
        if done_dict["__all__"]:
            print(f"\n{'='*60}")
            print(f"🏁 Episode 结束!")
            print(f"   原因: {'全部完成' if all_done else '超时'}")
            print(f"   步数: {self.current_step_count}/{self.max_steps}")
            print(f"   完成智能体: {len(self.dones)}/{self.num_robots}")
            print(f"{'='*60}\n")
            
            for aid in self.agent_ids:
                # 只标记真正完成的 agent 为 done
                # 未完成的仍然是 False，但 episode 因为超时而结束
                if aid in self.dones:
                    done_dict[aid] = True
                    info_dict[aid]['episode_complete'] = True
                else:
                    # 未完成的 agent：episode 因超时结束，但它自己未 done
                    done_dict[aid] = False
                    info_dict[aid]['episode_incomplete'] = True
                
                # truncated 用于标记超时截断
                if timeout:
                    truncated_dict[aid] = True
        
        # 【最终检查】确保所有字典都包含所有智能体
        for aid in self.agent_ids:
            assert aid in obs_dict, f"obs_dict 缺少 {aid}"
            assert aid in rew_dict, f"rew_dict 缺少 {aid}"
            assert aid in done_dict, f"done_dict 缺少 {aid}"
            assert aid in truncated_dict, f"truncated_dict 缺少 {aid}"
            assert aid in info_dict, f"info_dict 缺少 {aid}"
        
        return obs_dict, rew_dict, done_dict, truncated_dict, info_dict

    def _wait_and_spin_all(self, seconds):
        """
        自定义的等待函数：推进时间的同时，刷新所有机器人的回调
        """
        # 使用第0个机器人的时钟作为基准
        ref_node = self.agents[self.agent_ids[0]].node
        start_time = ref_node.get_clock().now().nanoseconds
        delta_ns = seconds * 1e9
        
        while rclpy.ok():
            # 1. 检查时间是否到达
            now = ref_node.get_clock().now().nanoseconds
            if now - start_time >= delta_ns:
                break
            
            # 2. 轮询所有机器人节点 (关键！)
            # 这样 Agent 1, 2, 3... 才能收到最新的 Scan 和 Odom
            for agent in self.agents.values():
                rclpy.spin_once(agent.node, timeout_sec=0.001)

    # def step(self, action_dict):
    #     self.current_step_count += 1
    #     obs_dict = {}
    #     rew_dict = {}
    #     done_dict = {}
    #     truncated_dict = {}
    #     info_dict = {}
        
    #     # 1. 遍历收到动作的机器人并执行
    #     for aid, action in action_dict.items():
    #         if aid not in self.dones:
    #             # 调用你原有环境的 step
    #             obs, rew, done, truncated, info = self.agents[aid].step(action)
                
    #             obs_dict[aid] = obs
    #             rew_dict[aid] = rew
    #             info_dict[aid] = info
                
    #             # 记录完成状态
    #             if done:
    #                 self.dones.add(aid)
    #                 done_dict[aid] = True
    #                 # 可以在这里额外加一个团队协作奖励（可选）
    #             else:
    #                 done_dict[aid] = False

    #     # 2. 对于已经完成的机器人，强制发送静止指令
    #     for aid in self.agent_ids:
    #         if aid in self.dones:
    #             # 调用原有环境的 _publish_vel 方法让其停下
    #             self.agents[aid]._publish_vel(0.0, 0.0)

    #     # 3. 全局结束条件：所有人都到达或超时
    #     all_done = (len(self.dones) == self.num_robots)
    #     timeout = (self.current_step_count >= self.max_steps)
        
    #     done_dict["__all__"] = all_done or timeout
    #     truncated_dict["__all__"] = timeout
    #     if timeout:
    #         for aid in self.agent_ids:
    #             truncated_dict[aid] = True
        
    #     return obs_dict, rew_dict, done_dict, truncated_dict, info_dict

    def close(self):
        for env in self.agents.values():
            env.close()