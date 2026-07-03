#!/usr/bin/env python3
"""
试点实验结果聚合脚本。
读取各 pilot 实验的 training_monitor.csv, 汇总末期性能指标成对比表。
用法: python3 aggregate_pilot_results.py
"""
import csv
import os
import glob
import statistics

RAY_RESULTS = "/home/wj/work/multi-robot-exploration-rl/ray_results"


def load_csv(path):
    """读取 training_monitor.csv, 返回行列表(dict)。"""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows, tail_frac=0.1):
    """计算末期(末 tail_frac)的均值, 以及峰值/收敛信息。"""
    if not rows:
        return None
    rew = [float(r["episode_reward_mean"]) for r in rows]
    ent = [float(r["entropy"]) for r in rows]
    vf = [float(r["vf_loss"]) for r in rows]
    n = len(rew)
    tail = max(1, int(n * tail_frac))
    return {
        "iters": n,
        "reward_start": rew[0],
        "reward_peak": max(rew),
        "reward_final": statistics.mean(rew[-tail:]),
        "entropy_final": statistics.mean(ent[-tail:]),
        "vf_loss_final": statistics.mean(vf[-tail:]),
    }


def find_run_csv(suffix_pattern):
    """根据 run_suffix 模式找到对应的 training_monitor.csv。"""
    # ray_results/<suffix>/GNN_MAPPO_*/training_monitor.csv
    matches = glob.glob(
        os.path.join(RAY_RESULTS, suffix_pattern, "**", "training_monitor.csv"),
        recursive=True,
    )
    return matches[0] if matches else None


def print_table(title, rows_dict):
    """打印对比表。rows_dict: {标签: summary_dict}"""
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    hdr = f"{'变体':28} {'iters':>6} {'起始':>8} {'峰值':>8} {'末尾':>8} {'熵末':>6} {'vf末':>6}"
    print(hdr)
    print("-" * 78)
    for label, s in rows_dict.items():
        if s is None:
            print(f"{label:28} {'(无数据)':>6}")
            continue
        print(f"{label:28} {s['iters']:>6} {s['reward_start']:>8.1f} "
              f"{s['reward_peak']:>8.1f} {s['reward_final']:>8.1f} "
              f"{s['entropy_final']:>6.2f} {s['vf_loss_final']:>6.2f}")


def main():
    # P0 对比实验
    cmp_runs = {
        "MLP baseline": "p0_cmp_mlp_seed42",
        "Full GAT (dual)": "pilot_smoke_gat",  # 复用冒烟运行(全长度 dual_graph)
    }
    cmp_res = {}
    for label, suffix in cmp_runs.items():
        csv_path = find_run_csv(suffix)
        cmp_res[label] = summarize(load_csv(csv_path)) if csv_path else None
    print_table("P0-1: 对比实验 (MLP vs Full GAT)", cmp_res)

    # P0 图消融
    abl_runs = {
        "dual_graph (Full)": "pilot_smoke_gat",  # 复用冒烟运行
        "social_only": "p0_abl_social_only_seed42",
        "obstacle_only": "p0_abl_obstacle_only_seed42",
    }
    abl_res = {}
    for label, suffix in abl_runs.items():
        csv_path = find_run_csv(suffix)
        abl_res[label] = summarize(load_csv(csv_path)) if csv_path else None
    print_table("P0-2: 图结构消融", abl_res)

    # P0 参数敏感性
    sweep_runs = {
        "comm_range=2.0": "p0_comm_2.0_seed42",
        "comm_range=3.5 (default)": "pilot_smoke_gat",  # 复用冒烟运行(comm=3.5)
        "comm_range=6.0": "p0_comm_6.0_seed42",
    }
    sweep_res = {}
    for label, suffix in sweep_runs.items():
        csv_path = find_run_csv(suffix)
        sweep_res[label] = summarize(load_csv(csv_path)) if csv_path else None
    print_table("P0-3: 参数敏感性 (communication_range)", sweep_res)

    print(f"\n{'='*78}")
    print("说明: P0 全长度单种子(seed=42)。dual_graph/comm=3.5 复用 pilot_smoke_gat 运行。")
    print("      最终定稿前需补 3 种子(42/123/456)均值±方差。")


if __name__ == "__main__":
    main()
