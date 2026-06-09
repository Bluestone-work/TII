import os
import torch as T
import torch.nn.functional as F
from start_reinforcement_learning.matd3_algorithm.agent import Agent
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class MATD3:  # MADDPG升级为MATD3
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='robot',  alpha=0.01, beta=0.01, fc1=512, 
                 fc2=512, gamma=0.99, tau=0.01, chkpt_dir='tmp/matd3/', 
                 node_logger = None, policy_freq=2):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.logger = node_logger
        self.started_learning = False
        
        # TD3: Delayed policy updates
        self.policy_freq = policy_freq  # 每policy_freq次更新critic才更新一次actor
        self.learn_step_cnt = 0  # 记录学习步数
        
        chkpt_dir += scenario
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(chkpt_dir, exist_ok=True)
        
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2,gamma=gamma,tau=tau,
                            noise_std_dev=0.4))  # 增大初始噪声，增强探索

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    # 将[-1,1]的连续动作映射到速度控制
    def action_to_velocity(self, continuous_actions):
        # continuous_actions: [linear_action, angular_action] in [-1, 1]
        # 映射到实际速度范围（避免过快导致Gazebo不稳定）
        # 前进速度：[-0.1, 0.22]（倒车更慢）
        if continuous_actions[0] >= 0:
            linear_vel = continuous_actions[0] * 0.22  # 前进 [0, 0.22]
        else:
            linear_vel = continuous_actions[0] * 0.10  # 倒车 [-0.1, 0]
        
        angular_vel = continuous_actions[1] * 1.0  # [-1.0, 1.0] 降低旋转速度
        return np.array([linear_vel, angular_vel])
                
    # returns dict of each agents chosen action for linear and velocity
    def choose_action(self, raw_obs):        
        actions = {}
        for agent_idx, agent in enumerate(self.agents):
            agent_id = f'robot{agent_idx}'
            if agent_id in raw_obs:
                continuous_actions = agent.choose_action(raw_obs[agent_id])

                # velocity_actions = self.action_to_velocity(continuous_actions)
                actions[agent_id] = continuous_actions
            else:
                print(f"[WARNING] Agent {agent_id} not found in observations!")
            
        return actions

    # Adjusts actor and critic weights - MATD3 implementation
    def learn(self, memory):
        # If memory is not the size of a batch size (1024) then return
        if not memory.ready():
            return
        
        # Samples algorithms central memory
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        # Makes sure each tensor is working on the same device, should be gpu (cuda:0)
        device = self.agents[0].actor.device
        
        # converts sampled memory list of arrays to Tensors
        states = T.tensor(np.array(states), dtype=T.float).to(device)
        rewards = T.tensor(np.array(rewards), dtype=T.float).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        states_ = T.tensor(np.array(states_), dtype=T.float).to(device)
        dones = T.tensor(np.array(dones)).to(device)

        # ============ MATD3: Twin Critics + Target Policy Smoothing ============
        with T.no_grad():
            # Target policy smoothing: 添加裁剪噪声
            all_agents_new_actions = []
            for agent_idx, agent in enumerate(self.agents):
                new_states = T.tensor(actor_new_states[agent_idx], 
                                     dtype=T.float).to(device)
                new_pi = agent.target_actor.forward(new_states).to(device)
                
                # TD3: 添加高斯噪声并裁剪
                noise = T.randn_like(new_pi) * 0.2
                noise = T.clamp(noise, -0.5, 0.5)
                new_pi = T.clamp(new_pi + noise, -1., 1.)
                all_agents_new_actions.append(new_pi)
            
            new_actions = T.cat([a for a in all_agents_new_actions], dim=1)
        
        # Update Critics (both Q1 and Q2 for each agent)
        old_actions = T.cat([actions[i] for i in range(self.n_agents)],dim=1)
        
        for agent_idx, agent in enumerate(self.agents):
            # Twin critics: 取两个Q值的最小值（减少过估计）
            q1_ = agent.target_critic_1.forward(states_, new_actions).squeeze()
            q2_ = agent.target_critic_2.forward(states_, new_actions).squeeze()
            q_next = T.min(q1_, q2_)  # TD3核心：取最小值
            
            q_next[dones[:, agent_idx]] = 0.0
            q_target = rewards[:, agent_idx] + agent.gamma * q_next
            
            # 更新两个critic
            q1 = agent.critic_1.forward(states, old_actions).squeeze()
            q2 = agent.critic_2.forward(states, old_actions).squeeze()
            
            critic_1_loss = F.mse_loss(q1, q_target)
            critic_2_loss = F.mse_loss(q2, q_target)
            
            agent.critic_1.optimizer.zero_grad()
            critic_1_loss.backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(agent.critic_1.parameters(), 10.0)
            agent.critic_1.optimizer.step()
            
            agent.critic_2.optimizer.zero_grad()
            critic_2_loss.backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(agent.critic_2.parameters(), 10.0)
            agent.critic_2.optimizer.step()
        
        # ============ MATD3: Delayed Policy Updates ============
        self.learn_step_cnt += 1
        if self.learn_step_cnt % self.policy_freq != 0:
            # 不更新actor和target networks
            return
        
        # 更新Actor（每policy_freq步更新一次）
        all_agents_new_mu_actions = []
        for agent_idx, agent in enumerate(self.agents):
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states).to(device)
            all_agents_new_mu_actions.append(pi)
        
        for agent_idx, agent in enumerate(self.agents):
            # 构造混合动作：当前agent用带梯度的pi，其他agent用detached的
            mu_for_agent_i = []
            for j in range(self.n_agents):
                if j == agent_idx:
                    mu_for_agent_i.append(all_agents_new_mu_actions[j])
                else:
                    mu_for_agent_i.append(all_agents_new_mu_actions[j].detach())
            
            mu_i = T.cat(mu_for_agent_i, dim=1)
            # Actor loss: 最大化Q1（只用一个critic计算actor loss）
            actor_loss = -agent.critic_1.forward(states, mu_i).flatten()
            actor_loss = T.mean(actor_loss)
            
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.actor.optimizer.step()
            
            # 软更新target networks
            agent.update_network_parameters()
        
        # 衰减所有agent的探索噪声
        for agent in self.agents:
            agent.decay_noise()
