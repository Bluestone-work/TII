#!/usr/bin/env python3
"""Paired Wilcoxon analysis for VDS and no-CF episode CSV files."""
import argparse

import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vds-csv", required=True, help="episodes.csv for the VDS run")
    parser.add_argument("--nocf-csv", required=True, help="episodes.csv for the no-CF run")
    parser.add_argument("--baseline-csv", default="", help="optional baseline episodes.csv")
    parser.add_argument("--expected-episodes", type=int, default=0, help="optional exact row-count check")
    args = parser.parse_args()

    df_vds = pd.read_csv(args.vds_csv)
    df_nocf = pd.read_csv(args.nocf_csv)
    df_baseline = pd.read_csv(args.baseline_csv) if args.baseline_csv else None

    print("=== Paired comparison: VDS vs no-CF ===\n")
    for map_num in sorted(df_vds["map_number"].unique()):
        vds_map = df_vds[df_vds["map_number"] == map_num]
        nocf_map = df_nocf[df_nocf["map_number"] == map_num]

        print(f"Map {map_num}:")
        print(f"  VDS:      SR={vds_map['SR'].mean():.3f}, TSR={vds_map['TSR'].mean():.3f}, CR={vds_map['CR'].mean():.3f}")
        print(f"  no_cf:    SR={nocf_map['SR'].mean():.3f}, TSR={nocf_map['TSR'].mean():.3f}, CR={nocf_map['CR'].mean():.3f}")
        if df_baseline is not None:
            baseline_map = df_baseline[df_baseline["map_number"] == map_num]
            print(f"  baseline: SR={baseline_map['SR'].mean():.3f}, TSR={baseline_map['TSR'].mean():.3f}, CR={baseline_map['CR'].mean():.3f}")
        print()

    if args.expected_episodes:
        assert len(df_vds) == len(df_nocf) == args.expected_episodes, "Episode counts mismatch"
    else:
        assert len(df_vds) == len(df_nocf), "Episode counts mismatch"

    tsr_vds = df_vds["TSR"].values
    tsr_nocf = df_nocf["TSR"].values
    _, p_tsr = wilcoxon(tsr_vds, tsr_nocf, alternative="greater")

    cr_vds = df_vds["CR"].values
    cr_nocf = df_nocf["CR"].values
    _, p_cr = wilcoxon(cr_vds, cr_nocf, alternative="less")

    print("=== Paired Wilcoxon test ===")
    print(f"TSR: VDS mean={tsr_vds.mean():.3f}, no_cf mean={tsr_nocf.mean():.3f}, diff={tsr_vds.mean() - tsr_nocf.mean():.3f}, p={p_tsr:.4f}")
    print(f"CR:  VDS mean={cr_vds.mean():.3f}, no_cf mean={cr_nocf.mean():.3f}, diff={cr_vds.mean() - cr_nocf.mean():.3f}, p={p_cr:.4f}")
    print(f"\nTSR one-sided p-value: {p_tsr:.4f} (alternative: VDS > no_cf)")
    print(f"CR one-sided p-value:  {p_cr:.4f} (alternative: VDS < no_cf)")


if __name__ == "__main__":
    main()
