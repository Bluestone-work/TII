import os
import csv
from datetime import datetime
import rclpy
from rclpy.node import Node
import numpy as np

from start_reinforcement_learning.env_logic.logic import Env
from start_reinforcement_learning.matd3_algorithm.matd3 import MATD3
from start_reinforcement_learning.matd3_algorithm.buffer import MultiAgentReplayBuffer
import torch as T
import gc
from collections import deque
from ament_index_python.packages import get_package_share_directory

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Main function that runs the MATD3 algorithm
class MATD3Node(Node):
    def __init__(self, map_number=None, robot_number=None, use_random_mode=None):
        super().__init__('matd3_node')

        # Access the parameters passed from the launch file or use provided values
        if map_number is None:
            map_number = self.declare_parameter('map_number', 1).get_parameter_value().integer_value
        if robot_number is None:
            robot_number = self.declare_parameter('robot_number', 3).get_parameter_value().integer_value
        if use_random_mode is None:
            use_random_mode = self.declare_parameter('use_random_mode', False).get_parameter_value().bool_value

        # 训练稳定性相关参数（可选）
        goal_termination_mode = self.declare_parameter('goal_termination_mode', 'any').get_parameter_value().string_value
        stuck_enabled = self.declare_parameter('stuck_enabled', True).get_parameter_value().bool_value
        stuck_min_progress = self.declare_parameter('stuck_min_progress', 0.02).get_parameter_value().double_value
        stuck_max_steps = self.declare_parameter('stuck_max_steps', 40).get_parameter_value().integer_value
        stuck_check_after_steps = self.declare_parameter('stuck_check_after_steps', 20).get_parameter_value().integer_value
        stuck_penalty = self.declare_parameter('stuck_penalty', -10.0).get_parameter_value().double_value

        map_names = ['map1', 'map2', 'corridor_swap', 'intersection', 'warehouse_aisles']
        map_name = map_names[map_number - 1] if 1 <= map_number <= 5 else 'map1'
        mode_str = 'Random Mode' if use_random_mode else 'Fixed Mode'
        
        self.get_logger().info(f"Map number: {map_number} ({map_name})")
        self.get_logger().info(f"Robot number: {robot_number}")
        self.get_logger().info(f"Position mode: {mode_str}")
        #T.cuda.empty_cache()
        #gc.collect()

        # Set environment with action size
        env = Env(
            robot_number,
            map_number,
            use_random_mode,
            goal_termination_mode=goal_termination_mode,
            stuck_enabled=stuck_enabled,
            stuck_min_progress=stuck_min_progress,
            stuck_max_steps=stuck_max_steps,
            stuck_check_after_steps=stuck_check_after_steps,
            stuck_penalty=stuck_penalty,
        )
        n_agents = env.number_of_robots
        
        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # Action space is discrete, one of 4 actions,  look in env
        n_actions = env.action_space()

        chkpt_dir_var = os.path.join(get_package_share_directory('start_reinforcement_learning'),
                                    'start_reinforcement_learning','deep_learning_weights','matd3')
        
        # Initialize MATD3 algorithm
        matd3_agents = MATD3(actor_dims, critic_dims, n_agents, n_actions, 
                              fc1=512, fc2=512, tau=0.00025,
                              alpha=1e-4, beta=1e-3, scenario='robot',
                              chkpt_dir=chkpt_dir_var, node_logger=self, 
                              policy_freq=2)  # TD3: 每2步更新一次policy

        # Initialize memory
        memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                    n_actions, n_agents, batch_size=256)

        PRINT_INTERVAL = 10
        WARMUP_STEPS = 2000  # 先收集一定样本再开始学习
        LEARN_EVERY = 1      # 每步学习一次（在warmup后）
        N_GAMES = 5000
        total_steps = 0
        score_history = []
        evaluate = False
        best_score = 0
        event_window = deque(maxlen=100)
        event_counts = {'goal_reached': 0, 'collision': 0, 'stuck': 0, 'timeout': 0, 'unknown': 0}

        # CSV日志
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        log_dir = os.path.join(repo_root, 'train_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'matd3_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        log_file = open(log_path, 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([
            'episode', 'score_per_robot', 'avg_score_100', 'total_steps', 'noise',
            'last_event', 'win_rate_100', 'collision_rate_100', 'stuck_rate_100', 'timeout_rate_100'
        ])

        step_log_path = os.path.join(log_dir, f'matd3_step_rewards_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        step_log_file = open(step_log_path, 'w', newline='')
        step_csv_writer = csv.writer(step_log_file)
        step_csv_writer.writerow([
            'episode', 'step_in_episode', 'total_steps', 'robot_id', 'event',
            'reward_total', 'r_action', 'r_heading', 'r_obstacle', 'r_goal', 'r_time',
            'tf_ok', 'tf_frame', 'tf_tx', 'tf_ty', 'tf_yaw'
        ])

        # Test network, remember decentralised centralised network,  training include critic + actor, testing only includes actor
        if evaluate:
            matd3_agents.load_checkpoint()

        # Currently 5000 episodes
        print(f"\n[INFO] 🚀 开始训练: {N_GAMES} episodes\n")
        try:
            for i in range(N_GAMES):
                # reset to get initial observation
                if i == 0:
                    print(f"[DEBUG] Episode {i}: 调用 env.reset()...")
                obs = env.reset()
                if i == 0:
                    print(f"[DEBUG] Episode {i}: reset完成，obs维度: {list(obs.values())[0].shape}")
                # Convert dict -> list of arrays to go into 'obs_list_to_state_vector' function
                list_obs = list(obs.values())
                score = 0
                done = [False]*n_agents
                terminal = [False] * n_agents
                episode_step = 0
                run_ep = True
                last_event = 'unknown'
                # Truncated means episode has reached max number of steps, done means collided or reached goal
                if i == 0:
                    print(f"[DEBUG] Episode {i}: 进入while循环...")
                while not any(terminal):
                    # Get the actions that the algorithm thinks are best in given observation
                    if i == 0 and episode_step == 0:
                        print(f"[DEBUG] Episode {i} Step {episode_step}: 调用 choose_action()...")
                    actions = matd3_agents.choose_action(obs)
                    if i == 0 and episode_step == 0:
                        print(f"[DEBUG] Episode {i} Step {episode_step}: 动作生成完成，调用 env.step()...")
                    # use step function to get next state and reward info as well as if the episode is 'done'
                    obs_, reward, done, truncated, info = env.step(actions)
                    if i == 0 and episode_step == 0:
                        print(f"[DEBUG] Episode {i} Step {episode_step}: step完成!")
                    if isinstance(info, dict) and 'event' in info:
                        last_event = info.get('event', 'unknown')

                    # 每步奖励分解写入CSV
                    if isinstance(info, dict) and 'reward_components' in info:
                        comps = info.get('reward_components') or []
                        tf_info = info.get('tf_info') if isinstance(info.get('tf_info'), dict) else {}
                        tf_ok = tf_info.get('ok', False)
                        tf_frame = tf_info.get('frame', '')
                        tf_tx = tf_info.get('tx', 0.0)
                        tf_ty = tf_info.get('ty', 0.0)
                        tf_yaw = tf_info.get('yaw', 0.0)
                        for ridx, comp in enumerate(comps):
                            try:
                                step_csv_writer.writerow([
                                    i, episode_step, total_steps, ridx, last_event,
                                    comp.get('total', 0.0), comp.get('r_action', 0.0),
                                    comp.get('r_heading', 0.0), comp.get('r_obstacle', 0.0),
                                    comp.get('r_goal', 0.0), comp.get('r_time', 0.0),
                                    tf_ok, tf_frame, tf_tx, tf_ty, tf_yaw
                                ])
                            except Exception:
                                pass
                    
                    
                    # Convert dict -> list of arrays to go into 'obs_list_to_state_vector' function
                    list_done = list(done.values())
                    list_reward = list(reward.values())
                    list_actions = list(actions.values())
                    list_obs_ = list(obs_.values())
                    list_trunc = list(truncated.values())

                    # Convert list of arrays to one flat array of observations
                    state = obs_list_to_state_vector(list_obs)
                    state_ = obs_list_to_state_vector(list_obs_)
                    
                    terminal = [d or t for d, t in zip(list_done, list_trunc)]

                    # Store raw observation as well as list of each agent's observation, reward, and terminal value together
                    memory.store_transition(list_obs, state, list_actions, list_reward, list_obs_, state_, terminal)

                    # 更频繁的学习：warmup后每步学习一次（当memory有足够样本后）
                    if total_steps >= WARMUP_STEPS and (total_steps % LEARN_EVERY == 0) and not evaluate:
                        matd3_agents.learn(memory)

                    # Set new obs to current obs
                    obs = obs_
                    score += sum(list_reward)
                    total_steps += 1
                    episode_step += 1
                # Calcualte the average score per robot
                score_history.append(score/robot_number)
                # 统计回合结束原因
                event_window.append(last_event)
                if last_event in event_counts:
                    event_counts[last_event] += 1
                else:
                    event_counts['unknown'] += 1
                
                # 🔧 新增：每个episode结束后衰减探索噪声
                for agent in matd3_agents.agents:
                    agent.decay_noise()
                
                # Average the last 100 recent scores
                avg_score = np.mean(score_history[-100:])
                if not evaluate:
                    if avg_score > best_score:
                        matd3_agents.save_checkpoint()
                        best_score = avg_score
                if i % PRINT_INTERVAL == 0 and i > 0:
                    avg_noise = np.mean([agent.noise_std for agent in matd3_agents.agents])
                    window_size = len(event_window)
                    if window_size > 0:
                        goal_rate = event_window.count('goal_reached') / window_size
                        collision_rate = event_window.count('collision') / window_size
                        stuck_rate = event_window.count('stuck') / window_size
                        timeout_rate = event_window.count('timeout') / window_size
                        self.get_logger().info(
                            'Episode: {}, Avg score: {:.1f}, Noise: {:.4f}, Steps: {}, '
                            'Win: {:.2f}, Coll: {:.2f}, Stuck: {:.2f}, Timeout: {:.2f}'.format(
                                i, avg_score, avg_noise, total_steps,
                                goal_rate, collision_rate, stuck_rate, timeout_rate))
                    else:
                        self.get_logger().info('Episode: {}, Avg score: {:.1f}, Noise: {:.4f}, Steps: {}'.format(
                            i, avg_score, avg_noise, total_steps))

                # 写入CSV日志（每回合）
                window_size = len(event_window)
                if window_size > 0:
                    goal_rate = event_window.count('goal_reached') / window_size
                    collision_rate = event_window.count('collision') / window_size
                    stuck_rate = event_window.count('stuck') / window_size
                    timeout_rate = event_window.count('timeout') / window_size
                else:
                    goal_rate = collision_rate = stuck_rate = timeout_rate = 0.0

                avg_noise = np.mean([agent.noise_std for agent in matd3_agents.agents])
                csv_writer.writerow([
                    i, score/robot_number, avg_score, total_steps, avg_noise,
                    last_event, goal_rate, collision_rate, stuck_rate, timeout_rate
                ])
                
                # 定期清理内存，避免内存泄漏
                if i % 50 == 0 and i > 0:
                    T.cuda.empty_cache()
                    gc.collect()
        except KeyboardInterrupt:
            self.get_logger().info('\n[INFO] 收到中断信号 (Ctrl+C)，正在停止机器人...')
        finally:
            # 清理：停止所有机器人
            env.cleanup()
            self.get_logger().info('[INFO] 所有机器人已停止')
            if log_file:
                log_file.flush()
                log_file.close()
            if step_log_file:
                step_log_file.flush()
                step_log_file.close()


def main(args=None):
    rclpy.init(args=args)
    
    # 不再从环境变量读取，而是让节点从ROS2参数中读取
    # 这样launch文件传递的参数才能生效
    node = MATD3Node()
    #rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()