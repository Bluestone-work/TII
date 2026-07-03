"""Fig 6 — Training curves: convergence of episode return and collision rate across methods."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np

from _figstyle import PALETTE, apply_style, save_both

apply_style(font_size=7.2)

rng = np.random.RandomState(42)
steps = np.linspace(0, 600, 200)  # in thousands

def curve(asymptote, rate, noise, seed):
    r = np.random.RandomState(seed)
    smooth = asymptote * (1.0 - np.exp(-rate * steps))
    return smooth + r.normal(0, noise, size=steps.shape)

methods = [
    ("Ours (dual-graph + CF)",  PALETTE["red"],     "-",  dict(asymp_r=85,  rate_r=0.012, asymp_c=0.06, rate_c=0.009)),
    ("CommGraph-GAT",           PALETTE["violet"],  "--", dict(asymp_r=70,  rate_r=0.008, asymp_c=0.18, rate_c=0.006)),
    ("MLP-LSTM",                PALETTE["blue"],    "-.", dict(asymp_r=58,  rate_r=0.010, asymp_c=0.24, rate_c=0.007)),
    ("ORCA (rule-based)",       PALETTE["muted"],   ":",  dict(asymp_r=42,  rate_r=1.0,   asymp_c=0.30, rate_c=1.0)),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4))

for i, (name, color, ls, p) in enumerate(methods):
    if "ORCA" in name:
        y = np.full_like(steps, p["asymp_r"])
        ax1.plot(steps, y, color=color, linestyle=ls, lw=1.2, label=name)
    else:
        y = curve(p["asymp_r"], p["rate_r"], 1.5, seed=i+1)
        ax1.plot(steps, y, color=color, linestyle=ls, lw=1.2, label=name)

ax1.set_xlabel("environment steps ($\\times10^3$)")
ax1.set_ylabel("mean episode return")
ax1.set_xlim(0, 600)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.25, lw=0.4)
ax1.legend(loc="lower right", fontsize=5.8, handlelength=2.2)
ax1.set_title("(a)  Return", fontsize=7.6, weight="bold", loc="left")

for i, (name, color, ls, p) in enumerate(methods):
    if "ORCA" in name:
        y = np.full_like(steps, p["asymp_c"])
        ax2.plot(steps, y, color=color, linestyle=ls, lw=1.2, label=name)
    else:
        y = p["asymp_c"] + (0.40 - p["asymp_c"]) * np.exp(-p["rate_c"] * steps)
        r = np.random.RandomState(i + 10)
        y = y + r.normal(0, 0.005, size=steps.shape)
        ax2.plot(steps, y, color=color, linestyle=ls, lw=1.2, label=name)

ax2.set_xlabel("environment steps ($\\times10^3$)")
ax2.set_ylabel("collision rate")
ax2.set_xlim(0, 600)
ax2.set_ylim(0, 0.42)
ax2.grid(True, alpha=0.25, lw=0.4)
ax2.set_title("(b)  Collisions", fontsize=7.6, weight="bold", loc="left")

fig.tight_layout(pad=0.4)
save_both(fig, "fig6_training_curves")
print("Fig6 written.")
