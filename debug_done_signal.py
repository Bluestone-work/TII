#!/usr/bin/env python3
"""
直接测试GNNMARLEnv的done信号
"""
import sys
import os
sys.path.insert(0, '/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer')

# 测试配置
test_config = {
    'num_agents': 2,
    'map_number': 1,
    'max_episode_steps': 500,
    'auto_reset_agents': False,  # 测试模式：关闭auto_reset
    'min_active_agents_to_continue': 0,  # 所有agent done后立即结束
    'max_failed_agents_before_cutoff': 0,
}

print("测试配置:")
for k, v in test_config.items():
    print(f"  {k}: {v}")
print()

# 模拟环境状态
class MockEnv:
    def __init__(self, config):
        self._num_agents = config['num_agents']
        self.auto_reset_agents = config['auto_reset_agents']
        self.min_active_agents_to_continue = config['min_active_agents_to_continue']
        self.max_failed_agents_before_cutoff = config['max_failed_agents_before_cutoff']
        self.max_steps = config['max_episode_steps']
        self.current_step_count = 0
        self.dones = set()
        self.failed_agents = set()

    def check_episode_over(self):
        timeout = (self.current_step_count >= self.max_steps)
        all_done = (len(self.dones) == self._num_agents)
        active_remaining = self._num_agents - len(self.dones)
        failed_count = len(self.failed_agents)

        print(f"\n[Step {self.current_step_count}]")
        print(f"  dones集合: {self.dones} (len={len(self.dones)})")
        print(f"  failed_agents: {self.failed_agents}")
        print(f"  timeout: {timeout}")
        print(f"  all_done: {all_done} ({len(self.dones)} == {self._num_agents})")
        print(f"  active_remaining: {active_remaining}")

        if self.auto_reset_agents:
            episode_over = timeout
            reason = 'timeout' if timeout else ''
        else:
            cutoff_few_active = (
                self.min_active_agents_to_continue > 0
                and active_remaining < self.min_active_agents_to_continue
            )
            cutoff_too_many_failed = (
                self.max_failed_agents_before_cutoff > 0
                and failed_count >= self.max_failed_agents_before_cutoff
            )
            episode_over = all_done or timeout or cutoff_few_active or cutoff_too_many_failed

            print(f"  cutoff_few_active: {cutoff_few_active}")
            print(f"  cutoff_too_many_failed: {cutoff_too_many_failed}")

            if timeout:
                reason = 'timeout'
            elif all_done:
                reason = 'all_done'
            elif cutoff_too_many_failed:
                reason = 'too_many_failed'
            elif cutoff_few_active:
                reason = 'too_few_active'
            else:
                reason = ''

        print(f"  => episode_over: {episode_over}, reason: '{reason}'")
        return episode_over, reason

# 测试场景
env = MockEnv(test_config)

print("\n" + "="*60)
print("场景1: 初始状态（没有agent done）")
print("="*60)
env.current_step_count = 10
env.dones = set()
env.failed_agents = set()
env.check_episode_over()

print("\n" + "="*60)
print("场景2: 一个agent到达目标")
print("="*60)
env.current_step_count = 50
env.dones = {'agent_0'}
env.failed_agents = set()
env.check_episode_over()

print("\n" + "="*60)
print("场景3: 一个到达目标，一个碰撞（两个都done）")
print("="*60)
env.current_step_count = 100
env.dones = {'agent_0', 'agent_1'}
env.failed_agents = {'agent_1'}
over, reason = env.check_episode_over()

print("\n" + "="*60)
if over:
    print(f"✅ Episode正确结束！原因: {reason}")
else:
    print(f"❌ Episode没有结束！这是BUG！")
print("="*60)
