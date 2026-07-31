#!/usr/bin/env python3
"""
Aggregate communication robustness results for Reviewer Q6
"""

import pandas as pd
import sys
from pathlib import Path
import re

def extract_comm_params(dirname):
    """Extract dropout and noise from directory name like 'dropout0.1_noise0.02'"""
    match = re.match(r'dropout([0-9.]+)_noise([0-9.]+)', dirname)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def aggregate_comms_robustness(sweep_dir):
    """Aggregate results from all communication configurations"""
    sweep_path = Path(sweep_dir)

    results = []

    for subdir in sorted(sweep_path.iterdir()):
        if not subdir.is_dir():
            continue

        dropout, noise = extract_comm_params(subdir.name)
        if dropout is None:
            continue

        episodes_csv = subdir / "episodes.csv"
        if not episodes_csv.exists():
            print(f"WARNING: Missing {episodes_csv}", file=sys.stderr)
            continue

        df = pd.read_csv(episodes_csv)

        # Calculate metrics
        agent_sr = (df['agent_success'] == 1).mean() * 100
        team_sr = (df['team_success'] == 1).mean() * 100
        collision_rate = (df['collision'] == 1).mean() * 100
        timeout_rate = (df['timeout'] == 1).mean() * 100

        results.append({
            'dropout_prob': dropout,
            'noise_std': noise,
            'agent_SR_%': round(agent_sr, 1),
            'team_SR_%': round(team_sr, 1),
            'collision_%': round(collision_rate, 1),
            'timeout_%': round(timeout_rate, 1),
            'n_episodes': len(df),
        })

    if not results:
        print("ERROR: No valid results found", file=sys.stderr)
        return None

    results_df = pd.DataFrame(results)

    # Sort: dropout sweep first (noise=0), then noise sweep (dropout=0)
    results_df = results_df.sort_values(['noise_std', 'dropout_prob'])

    return results_df

def print_summary(df):
    """Print formatted summary"""
    print("\n" + "="*80)
    print("COMMUNICATION ROBUSTNESS ANALYSIS RESULTS")
    print("="*80 + "\n")

    # Dropout sweep (noise=0)
    dropout_sweep = df[df['noise_std'] == 0.0].copy()
    if not dropout_sweep.empty:
        print("Dropout Robustness (noise_std=0.0):")
        print("-" * 80)
        print(dropout_sweep.to_string(index=False))
        print()

        # Calculate degradation
        baseline_sr = dropout_sweep[dropout_sweep['dropout_prob'] == 0.0]['agent_SR_%'].values[0]
        print(f"Baseline (dropout=0): {baseline_sr:.1f}% Agent SR")
        for _, row in dropout_sweep.iterrows():
            if row['dropout_prob'] > 0:
                degradation = baseline_sr - row['agent_SR_%']
                print(f"  dropout={row['dropout_prob']:.2f}: {row['agent_SR_%']:.1f}% SR "
                      f"({degradation:+.1f}% degradation)")
        print()

    # Noise sweep (dropout=0)
    noise_sweep = df[df['dropout_prob'] == 0.0].copy()
    if not noise_sweep.empty:
        print("Noise Robustness (dropout_prob=0.0):")
        print("-" * 80)
        print(noise_sweep.to_string(index=False))
        print()

        # Calculate degradation
        baseline_sr = noise_sweep[noise_sweep['noise_std'] == 0.0]['agent_SR_%'].values[0]
        print(f"Baseline (noise=0): {baseline_sr:.1f}% Agent SR")
        for _, row in noise_sweep.iterrows():
            if row['noise_std'] > 0:
                degradation = baseline_sr - row['agent_SR_%']
                print(f"  noise_std={row['noise_std']:.2f}: {row['agent_SR_%']:.1f}% SR "
                      f"({degradation:+.1f}% degradation)")
        print()

    print("="*80 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 aggregate_comms_robustness_results.py <comms_robustness_dir>")
        sys.exit(1)

    sweep_dir = sys.argv[1]

    print(f"Aggregating results from: {sweep_dir}")

    results_df = aggregate_comms_robustness(sweep_dir)

    if results_df is None:
        sys.exit(1)

    # Save results
    output_csv = Path(sweep_dir) / "comms_robustness_summary.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Print summary
    print_summary(results_df)

    # Generate LaTeX tables
    output_tex = Path(sweep_dir) / "comms_robustness_table.tex"
    with open(output_tex, 'w') as f:
        f.write("% Communication robustness tables\n\n")

        # Dropout table
        dropout_sweep = results_df[results_df['noise_std'] == 0.0].copy()
        if not dropout_sweep.empty:
            f.write("% Dropout robustness\n")
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Communication Dropout Robustness}\n")
            f.write("\\label{tab:dropout_robustness}\n")
            f.write("\\begin{tabular}{c|cc|c}\n")
            f.write("\\toprule\n")
            f.write("Dropout Rate & Agent SR (\\%) & Team SR (\\%) & Collision (\\%) \\\\\n")
            f.write("\\midrule\n")

            for _, row in dropout_sweep.iterrows():
                f.write(f"{row['dropout_prob']:.2f} & {row['agent_SR_%']:.1f} & "
                       f"{row['team_SR_%']:.1f} & {row['collision_%']:.1f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

        # Noise table
        noise_sweep = results_df[results_df['dropout_prob'] == 0.0].copy()
        if not noise_sweep.empty:
            f.write("% Noise robustness\n")
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Communication Noise Robustness}\n")
            f.write("\\label{tab:noise_robustness}\n")
            f.write("\\begin{tabular}{c|cc|c}\n")
            f.write("\\toprule\n")
            f.write("Noise Std & Agent SR (\\%) & Team SR (\\%) & Collision (\\%) \\\\\n")
            f.write("\\midrule\n")

            for _, row in noise_sweep.iterrows():
                f.write(f"{row['noise_std']:.2f} & {row['agent_SR_%']:.1f} & "
                       f"{row['team_SR_%']:.1f} & {row['collision_%']:.1f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

    print(f"Saved: {output_tex}\n")

if __name__ == "__main__":
    main()
