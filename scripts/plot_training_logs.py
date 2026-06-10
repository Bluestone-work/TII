#!/usr/bin/env python3
import os
import glob
import csv
from datetime import datetime
import matplotlib.pyplot as plt

def _latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_episode_log(path):
    episodes = []
    avg_scores = []
    win_rates = []
    coll_rates = []
    stuck_rates = []
    timeout_rates = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            avg_scores.append(float(row['avg_score_100']))
            win_rates.append(float(row['win_rate_100']))
            coll_rates.append(float(row['collision_rate_100']))
            stuck_rates.append(float(row['stuck_rate_100']))
            timeout_rates.append(float(row['timeout_rate_100']))
    return episodes, avg_scores, win_rates, coll_rates, stuck_rates, timeout_rates


def load_step_log(path):
    # Aggregate reward components by episode (mean over steps and robots)
    agg = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['episode'])
            if ep not in agg:
                agg[ep] = {'count': 0, 'r_action': 0.0, 'r_heading': 0.0, 'r_obstacle': 0.0, 'r_goal': 0.0, 'r_time': 0.0, 'total': 0.0}
            agg[ep]['count'] += 1
            agg[ep]['r_action'] += float(row['r_action'])
            agg[ep]['r_heading'] += float(row['r_heading'])
            agg[ep]['r_obstacle'] += float(row['r_obstacle'])
            agg[ep]['r_goal'] += float(row['r_goal'])
            agg[ep]['r_time'] += float(row['r_time'])
            agg[ep]['total'] += float(row['reward_total'])

    episodes = sorted(agg.keys())
    def mean(key):
        return [agg[ep][key] / max(1, agg[ep]['count']) for ep in episodes]

    return episodes, mean('total'), mean('r_action'), mean('r_heading'), mean('r_obstacle'), mean('r_goal'), mean('r_time')


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir = os.path.join(repo_root, 'train_logs')

    episode_log = _latest_file(os.path.join(log_dir, 'matd3_train_*.csv'))
    step_log = _latest_file(os.path.join(log_dir, 'matd3_step_rewards_*.csv'))

    if not episode_log or not step_log:
        print('No log files found in train_logs.')
        return

    print(f'Using episode log: {episode_log}')
    print(f'Using step log: {step_log}')

    ep, avg_scores, win_rates, coll_rates, stuck_rates, timeout_rates = load_episode_log(episode_log)
    ep2, total, r_action, r_heading, r_obstacle, r_goal, r_time = load_step_log(step_log)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(ep, avg_scores, label='Avg score (100ep)')
    axes[0].set_ylabel('Score')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, win_rates, label='Win')
    axes[1].plot(ep, coll_rates, label='Collision')
    axes[1].plot(ep, stuck_rates, label='Stuck')
    axes[1].plot(ep, timeout_rates, label='Timeout')
    axes[1].set_ylabel('Rate (100ep)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(ep2, total, label='Total')
    axes[2].plot(ep2, r_action, label='r_action')
    axes[2].plot(ep2, r_heading, label='r_heading')
    axes[2].plot(ep2, r_obstacle, label='r_obstacle')
    axes[2].plot(ep2, r_goal, label='r_goal')
    axes[2].plot(ep2, r_time, label='r_time')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Mean reward components')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3)

    plt.tight_layout()
    out_path = os.path.join(log_dir, f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved plot to: {out_path}')


if __name__ == '__main__':
    main()
