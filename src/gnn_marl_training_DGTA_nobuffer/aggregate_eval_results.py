#!/usr/bin/env python3
"""
论文评估结果聚合脚本 (2026-06-29)

输入: run_paper_evaluation.sh 产出的 episodes.jsonl (每行一个 episode)
输出:
  1. table1_main_results.tex   — 论文 Table 1 (LaTeX)
  2. aggregated_metrics.csv    — 各 method 的 mean±std 指标
  3. figure7_ablation.pdf/.png — 消融柱状图 (如有 matplotlib)

用法:
  python3 aggregate_eval_results.py <results_dir>/episodes.jsonl
  python3 aggregate_eval_results.py <results_dir>/episodes.jsonl --num_agents 4
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


# method 标签 → 论文里显示的名字
METHOD_DISPLAY = {
    "ours_dual":     "Ours (Dual-Graph)",
    "ours_dual_cf":  "Ours (Dual-Graph+CF)",
    "social_only":   "Social-Only",
    "obstacle_only": "Obstacle-Only",
    "no_gate":       "No-Gate",
    "mlp_lstm":      "MLP-LSTM",
    "commgraph_gat": "CommGraph-GAT",
    "orca":          "ORCA",
}
# 论文里的方法排序（baseline 在前，Ours 在后高亮）
METHOD_ORDER = [
    "mlp_lstm", "commgraph_gat", "orca",
    "social_only", "obstacle_only", "no_gate",
    "ours_dual", "ours_dual_cf",
]


def load_episodes(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records, num_agents_hint=None):
    """按 (method, scenario) 聚合，计算每个指标的 mean±std。"""
    groups = defaultdict(list)
    for r in records:
        method = r.get("method", "unknown")
        scenario = r.get("scenario", "default")
        groups[(method, scenario)].append(r)

    agg = {}
    for (method, scenario), eps in groups.items():
        n_eps = len(eps)
        # 每个 episode 的 agent 数（优先用记录里的，其次用 hint）
        na = eps[0].get("num_agents") or num_agents_hint or 1

        # success rate (%) = 到达的 agent 数 / 总 agent 数
        success_rates = [
            100.0 * e.get("reached_agents", 0) / max(1, e.get("num_agents", na))
            for e in eps
        ]
        # collision rate (%)
        collision_rates = [
            100.0 * e.get("collided_agents", 0) / max(1, e.get("num_agents", na))
            for e in eps
        ]
        ep_lengths = [e.get("steps", 0) for e in eps]
        rewards = [e.get("avg_reward", 0.0) for e in eps]
        # min_dist 可能是 None
        min_dists = [e.get("min_dist") for e in eps if e.get("min_dist") is not None]
        trunc_rate = 100.0 * sum(1 for e in eps if e.get("truncated")) / max(1, n_eps)

        def ms(x):
            if len(x) == 0:
                return (float("nan"), float("nan"))
            return (float(np.mean(x)), float(np.std(x)))

        agg[(method, scenario)] = {
            "n_episodes": n_eps,
            "num_agents": na,
            "success_rate": ms(success_rates),
            "collision_rate": ms(collision_rates),
            "episode_length": ms(ep_lengths),
            "avg_reward": ms(rewards),
            "min_dist": ms(min_dists),
            "truncated_rate": trunc_rate,
        }
    return agg


def write_csv(agg, out_csv):
    import csv
    rows = []
    for (method, scenario), m in agg.items():
        rows.append({
            "method": method,
            "scenario": scenario,
            "n_episodes": m["n_episodes"],
            "num_agents": m["num_agents"],
            "success_rate_mean": round(m["success_rate"][0], 2),
            "success_rate_std": round(m["success_rate"][1], 2),
            "collision_rate_mean": round(m["collision_rate"][0], 2),
            "collision_rate_std": round(m["collision_rate"][1], 2),
            "episode_length_mean": round(m["episode_length"][0], 1),
            "episode_length_std": round(m["episode_length"][1], 1),
            "avg_reward_mean": round(m["avg_reward"][0], 2),
            "avg_reward_std": round(m["avg_reward"][1], 2),
            "min_dist_mean": round(m["min_dist"][0], 3),
            "min_dist_std": round(m["min_dist"][1], 3),
            "truncated_rate": round(m["truncated_rate"], 1),
        })
    # 按 METHOD_ORDER 排序
    rows.sort(key=lambda r: (r["scenario"],
                             METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99))
    if not rows:
        print("⚠️ 无数据可写 CSV")
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV 已写入: {out_csv}")


def write_latex_table(agg, out_tex):
    """生成论文 Table 1 (LaTeX)。每个 scenario 一个 block。"""
    scenarios = sorted(set(s for (_, s) in agg.keys()))
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison on multi-AGV navigation. "
                 r"Success and collision rates are per-agent percentages; "
                 r"values are mean$\pm$std over evaluation episodes. "
                 r"Best results in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{@{}llccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Method & Success (\%) $\uparrow$ & Collision (\%) $\downarrow$ "
                 r"& Episode Len. $\downarrow$ & Min Sep. (m) $\uparrow$ & Reward $\uparrow$ \\")
    lines.append(r"\midrule")

    for scenario in scenarios:
        methods_here = [(mth, sc) for (mth, sc) in agg.keys() if sc == scenario]
        methods_here.sort(key=lambda x: METHOD_ORDER.index(x[0]) if x[0] in METHOD_ORDER else 99)

        # 找每列最优值用于加粗
        succ_vals = {m: agg[(m, sc)]["success_rate"][0] for (m, sc) in methods_here}
        coll_vals = {m: agg[(m, sc)]["collision_rate"][0] for (m, sc) in methods_here}
        len_vals  = {m: agg[(m, sc)]["episode_length"][0] for (m, sc) in methods_here}
        sep_vals  = {m: agg[(m, sc)]["min_dist"][0] for (m, sc) in methods_here}
        rew_vals  = {m: agg[(m, sc)]["avg_reward"][0] for (m, sc) in methods_here}
        best_succ = max(succ_vals.values()) if succ_vals else None
        best_coll = min(coll_vals.values()) if coll_vals else None
        best_len  = min(len_vals.values()) if len_vals else None
        best_sep  = max(sep_vals.values()) if sep_vals else None
        best_rew  = max(rew_vals.values()) if rew_vals else None

        first = True
        for (method, sc) in methods_here:
            m = agg[(method, sc)]
            disp = METHOD_DISPLAY.get(method, method)
            scen_cell = scenario.replace("_", r"\_") if first else ""
            first = False

            def fmt(val_ms, best, fmtspec="{:.1f}", lower=False):
                v, s = val_ms
                if np.isnan(v):
                    return "--"
                txt = (fmtspec + r"$\pm$" + fmtspec).format(v, s)
                is_best = (best is not None and abs(v - best) < 1e-6)
                return r"\textbf{" + txt + "}" if is_best else txt

            row = " & ".join([
                scen_cell,
                disp,
                fmt(m["success_rate"], best_succ),
                fmt(m["collision_rate"], best_coll),
                fmt(m["episode_length"], best_len, "{:.0f}"),
                fmt(m["min_dist"], best_sep, "{:.2f}"),
                fmt(m["avg_reward"], best_rew, "{:.1f}"),
            ])
            lines.append(row + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ LaTeX 表格已写入: {out_tex}")


def plot_ablation(agg, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用，跳过 Figure 7 绘制")
        return

    # 只画第一个 scenario
    scenarios = sorted(set(s for (_, s) in agg.keys()))
    if not scenarios:
        return
    scenario = scenarios[0]
    methods_here = [m for (m, s) in agg.keys() if s == scenario]
    methods_here.sort(key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)

    metrics = [
        ("success_rate", "Success Rate (%)", False),
        ("collision_rate", "Collision Rate (%)", True),
        ("episode_length", "Episode Length", True),
        ("min_dist", "Min Separation (m)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_here)))

    for ax, (key, label, _lower_better) in zip(axes.flat, metrics):
        means = [agg[(m, scenario)][key][0] for m in methods_here]
        stds  = [agg[(m, scenario)][key][1] for m in methods_here]
        disp  = [METHOD_DISPLAY.get(m, m) for m in methods_here]
        bars = ax.bar(range(len(methods_here)), means, yerr=stds,
                      capsize=4, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(range(len(methods_here)))
        ax.set_xticklabels(disp, rotation=35, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Ablation & Comparison ({scenario})", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_path + ".png", dpi=300, bbox_inches="tight")
    print(f"✅ Figure 7 已写入: {out_path}.pdf / .png")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="episodes.jsonl 路径")
    ap.add_argument("--num_agents", type=int, default=None, help="若 JSONL 缺 num_agents 字段则用此")
    ap.add_argument("--outdir", default=None, help="输出目录 (默认与 jsonl 同目录)")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"❌ 找不到: {args.jsonl}")
        sys.exit(1)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.jsonl))
    os.makedirs(outdir, exist_ok=True)

    records = load_episodes(args.jsonl)
    print(f"📥 读取 {len(records)} 条 episode 记录")
    if not records:
        print("❌ 无有效记录")
        sys.exit(1)

    agg = aggregate(records, num_agents_hint=args.num_agents)

    # 控制台预览
    print("\n=== 聚合结果预览 ===")
    for (method, scenario), m in sorted(agg.items()):
        print(f"  [{method} @ {scenario}] n={m['n_episodes']}  "
              f"success={m['success_rate'][0]:.1f}±{m['success_rate'][1]:.1f}%  "
              f"collision={m['collision_rate'][0]:.1f}±{m['collision_rate'][1]:.1f}%  "
              f"len={m['episode_length'][0]:.0f}  "
              f"min_sep={m['min_dist'][0]:.2f}m")

    write_csv(agg, os.path.join(outdir, "aggregated_metrics.csv"))
    write_latex_table(agg, os.path.join(outdir, "table1_main_results.tex"))
    plot_ablation(agg, os.path.join(outdir, "figure7_ablation"))

    print(f"\n✅ 全部完成，输出目录: {outdir}")


if __name__ == "__main__":
    main()
