import torch as T
from start_reinforcement_learning.matd3_algorithm.networks import ActorNetwork, CriticNetwork
import numpy as np

# [新增] OU噪声类：这是解决“原地抖动”的核心，给动作增加惯性
class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01, noise_std_dev=0.2):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        
        # 使用高斯噪声（TD3风格，比OU噪声更简单有效）
        self.noise_std = noise_std_dev
        self.min_noise_std = 0.05  # 最小噪声，保持一定探索
        self.noise_decay = 0.9995  # 噪声衰减率
        self.noise_clip = 0.5  # TD3: 限制噪声范围
        
        # Actor网络
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        
        self.critic_1 = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic_1')
        self.critic_2 = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic_2')
        self.target_critic_1 = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic_2')
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, explore=True):
        state = T.tensor(observation[np.newaxis, :], dtype=T.float,
                         device=self.actor.device)
        actions = self.actor.forward(state)
        actions_np = actions.detach().cpu().numpy()[0]
        
        if explore:
            # TD3风格：高斯噪声 + 裁剪
            noise = np.random.normal(0, self.noise_std, size=self.n_actions)
            action = np.clip(actions_np + noise, -1., 1.)
        else:
            action = actions_np

        return action

    def decay_noise(self):
        """衰减探索噪声，从高探索逐渐过渡到低探索"""
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)
    
    def update_network_parameters(self, tau=None):
        tau = tau or self.tau
        # Update target actor
        for param, target in zip(self.actor.parameters(), self.target_actor.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)
        # Update target critic 1
        for param, target in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)
        # Update target critic 2
        for param, target in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()