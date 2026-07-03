"""Fig 5 — Leave-one-out counterfactual credit assignment.

Single-column figure with two panels:
 (a) Computation diagram. From global state s_t the centralized critic computes
     V(s_t) and N masked variants V(s_t \\ i) (one per agent). Marginal
     contributions phi_i are normalized, clipped, scaled by lambda_cf, and
     injected into per-agent rewards r_t^i before GAE.
 (b) Numeric vignette over a head-on encounter. Two agents share r_t but only
     one yielded; phi_i resolves the credit ambiguity.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np

from _figstyle import (
    PALETTE, apply_style, blank_ax, rbox, text, arrow, labeled_box, save_both,
)

apply_style(font_size=7.0)

fig = plt.figure(figsize=(3.5, 4.6))
gs = fig.add_gridspec(
    2, 1, height_ratios=[1.0, 0.95], hspace=0.30,
    left=0.05, right=0.97, top=0.97, bottom=0.04,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ============================================================
# Panel a — Computation diagram
# ============================================================
blank_ax(ax_a, xlim=(0, 100), ylim=(0, 64))
ax_a.set_aspect("auto")
ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 64)

text(ax_a, 0.5, 61.5, "a", size=9.5, weight="bold", ha="left")
text(ax_a, 6.5, 61.5, "Leave-one-out counterfactual credit",
     size=6.6, ha="left", color=PALETTE["ink"])

# --- Step 1: global state s_t with N agents (chips) ---
N = 4
state_x = 2; state_w = 22
state_y = 39; state_h = 16
rbox(ax_a, state_x, state_y, state_w, state_h,
     fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"], lw=0.8, r=0.3)
text(ax_a, state_x + state_w / 2, state_y + state_h - 2.0,
     "global state  $s_t$", size=6.0, weight="bold")

agent_colors = [PALETTE["red"], PALETTE["blue"], PALETTE["teal"], PALETTE["amber"]]
chip_y = state_y + 5.0
chip_w = 3.8; chip_h = 4.0
chip_gap = 1.0
total_w = N * chip_w + (N - 1) * chip_gap
start_x = state_x + (state_w - total_w) / 2
for i, c in enumerate(agent_colors):
    cx = start_x + i * (chip_w + chip_gap)
    rbox(ax_a, cx, chip_y, chip_w, chip_h, fc=c, ec=PALETTE["ink"],
         lw=0.5, r=0.18, z=4)
    text(ax_a, cx + chip_w / 2, chip_y + chip_h / 2,
         f"$o^{{{i+1}}}_t$", size=5.4, weight="bold", color="white", z=5)

# --- Step 2: critic forward passes — full V(s_t) + N masked variants ---
critic_x = 32; critic_w = 28
critic_y = 26; critic_h = 33
rbox(ax_a, critic_x, critic_y, critic_w, critic_h,
     fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"], lw=0.8, r=0.3)
text(ax_a, critic_x + critic_w / 2, critic_y + critic_h - 1.8,
     "centralized critic", size=6.0, weight="bold")
text(ax_a, critic_x + critic_w / 2, critic_y + critic_h - 3.6,
     "$V_\\psi$  •  one masked pass per agent",
     size=4.8, color=PALETTE["ink_soft"], style="italic")

# Rows: V(s_t) then 4 masked variants
row_y0 = critic_y + critic_h - 7.5
row_h = 4.0
row_gap = 0.6

def mask_row(y, label, mask_idx=None):
    rbox(ax_a, critic_x + 1.5, y, critic_w - 3, row_h,
         fc="white", ec=PALETTE["ink_soft"], lw=0.5, r=0.15)
    text(ax_a, critic_x + 3.0, y + row_h / 2, label, size=5.0,
         ha="left", weight="bold")
    # mini-chip strip
    cw = 1.5; ch = 1.7; gp = 0.45
    tot = N * cw + (N - 1) * gp
    sx = critic_x + critic_w - tot - 2.0
    for j in range(N):
        cx = sx + j * (cw + gp)
        if mask_idx is not None and j == mask_idx:
            rbox(ax_a, cx, y + (row_h - ch) / 2, cw, ch,
                 fc="#F2F2F2", ec=PALETTE["muted"], lw=0.4, r=0.1, z=4)
            text(ax_a, cx + cw / 2, y + row_h / 2, "0", size=4.6,
                 color=PALETTE["muted"], z=5)
        else:
            rbox(ax_a, cx, y + (row_h - ch) / 2, cw, ch,
                 fc=agent_colors[j], ec=PALETTE["ink"], lw=0.3, r=0.1, z=4)

# V(s_t) full
mask_row(row_y0, "$V(s_t)$")
# V(s_t \ i) for each i
for k in range(N):
    y = row_y0 - (k + 1) * (row_h + row_gap)
    mask_row(y, f"$V(s_t\\!\\setminus\\!{k+1})$", mask_idx=k)

# --- Step 3: phi_i column ---
phi_x = 64; phi_w = 14
phi_y = 26; phi_h = 33
rbox(ax_a, phi_x, phi_y, phi_w, phi_h,
     fc=PALETTE["green_soft"], ec=PALETTE["green"], lw=0.9, r=0.3)
text(ax_a, phi_x + phi_w / 2, phi_y + phi_h - 1.8,
     "$\\phi_i$", size=8.0, weight="bold", color=PALETTE["green"])
text(ax_a, phi_x + phi_w / 2, phi_y + phi_h - 4.0,
     "$V{-}V\\!\\setminus\\!i$",
     size=4.6, color=PALETTE["ink_soft"], style="italic")

# Mini bar chart of phi values inside the column
phi_vals = [+0.72, -0.18, +0.34, +0.08]
bar_x = phi_x + phi_w / 2
bar_zone_y0 = phi_y + 5.0
bar_zone_h = phi_h - 11
slot_h = bar_zone_h / N
for i, v in enumerate(phi_vals):
    y_mid = bar_zone_y0 + (N - 1 - i) * slot_h + slot_h / 2
    # zero line
    ax_a.plot([bar_x - 4.5, bar_x + 4.5], [y_mid, y_mid],
              color=PALETTE["stage_edge"], lw=0.4, zorder=2)
    bar_w = 4.5 * (v / 0.8)
    color = PALETTE["green"] if v >= 0 else PALETTE["red"]
    rbox(ax_a, bar_x, y_mid - 0.7, bar_w, 1.4,
         fc=color, ec="none", lw=0, r=0.05, z=3) if bar_w >= 0 else \
        rbox(ax_a, bar_x + bar_w, y_mid - 0.7, -bar_w, 1.4,
             fc=color, ec="none", lw=0, r=0.05, z=3)
    text(ax_a, bar_x - 5.5, y_mid, f"$i{{=}}{i+1}$",
         size=4.6, ha="right", color=PALETTE["ink_soft"])

# --- Step 4: reward shaping ---
shape_x = 82; shape_w = 16
shape_y = 38; shape_h = 18
rbox(ax_a, shape_x, shape_y, shape_w, shape_h,
     fc=PALETTE["red_soft"], ec=PALETTE["red"], lw=0.9, r=0.3)
text(ax_a, shape_x + shape_w / 2, shape_y + shape_h - 1.8,
     "shaped reward", size=5.6, weight="bold", color=PALETTE["red"])
text(ax_a, shape_x + shape_w / 2, shape_y + shape_h - 4.0,
     "$r^{i,\\mathrm{sh}}_t$", size=7.6, weight="bold")
text(ax_a, shape_x + shape_w / 2, shape_y + shape_h - 8.0,
     "$=\\,r^i_t$",
     size=5.0, color=PALETTE["ink"])
text(ax_a, shape_x + shape_w / 2, shape_y + shape_h - 10.5,
     "$+\\,\\lambda_{cf}\\,\\widetilde\\phi_i$",
     size=5.0, color=PALETTE["red"])
text(ax_a, shape_x + shape_w / 2, shape_y + 2.4,
     "$\\lambda_{cf}{=}0.15$",
     size=4.8, color=PALETTE["ink_soft"], style="italic")

# --- Step 5: GAE → PPO chip below ---
gae_x = 32; gae_w = 66
gae_y = 12; gae_h = 8.5
rbox(ax_a, gae_x, gae_y, gae_w, gae_h,
     fc=PALETTE["blue_soft"], ec=PALETTE["blue"], lw=0.8, r=0.25)
text(ax_a, gae_x + gae_w / 2, gae_y + gae_h - 2.2,
     "GAE  →  per-agent advantages  →  clipped PPO update",
     size=6.0, weight="bold", color=PALETTE["blue"])
text(ax_a, gae_x + gae_w / 2, gae_y + 2.0,
     "credit propagates through multi-step returns",
     size=5.2, color=PALETTE["ink_soft"], style="italic")

# --- Arrows wiring the panel ---
arrow(ax_a, state_x + state_w, state_y + state_h / 2,
      critic_x, critic_y + critic_h / 2,
      color=PALETTE["ink"], lw=1.0, mut=8)
arrow(ax_a, critic_x + critic_w, critic_y + critic_h / 2,
      phi_x, phi_y + phi_h / 2,
      color=PALETTE["green"], lw=1.0, mut=8)
arrow(ax_a, phi_x + phi_w, phi_y + phi_h / 2,
      shape_x, shape_y + shape_h / 2,
      color=PALETTE["red"], lw=1.0, mut=8)
text(ax_a, (phi_x + phi_w + shape_x) / 2, phi_y + phi_h / 2 + 1.6,
     "normalize + clip", size=4.6, color=PALETTE["ink_soft"],
     style="italic")

# Shaped reward → GAE block (down)
arrow(ax_a, shape_x + shape_w / 2, shape_y,
      shape_x + shape_w / 2, gae_y + gae_h + 0.05,
      color=PALETTE["ink_soft"], lw=0.9, mut=7)
arrow(ax_a, shape_x + shape_w / 2, gae_y + gae_h,
      gae_x + gae_w * 0.7, gae_y + gae_h + 0.05,
      color=PALETTE["ink_soft"], lw=0.9, mut=0, style="-")

# Note: only one extra critic forward pass per agent
text(ax_a, 50, 1.2,
     "Cost: $N$ extra critic passes per timestep  •  reuses centralized $V_\\psi$",
     size=4.8, color=PALETTE["ink_soft"], ha="center", style="italic")

# ============================================================
# Panel b — Numeric vignette for a head-on encounter
# ============================================================
blank_ax(ax_b, xlim=(0, 100), ylim=(0, 60))
ax_b.set_aspect("auto")
ax_b.set_xlim(0, 100); ax_b.set_ylim(0, 60)

text(ax_b, 0.5, 58.0, "b", size=9.5, weight="bold", ha="left")
text(ax_b, 6.5, 58.0, "Vignette  •  shared $r_t$ vs. counterfactual credit",
     size=6.6, ha="left", color=PALETTE["ink"])

# Mini head-on scene (top portion)
scene_y0 = 39; scene_h = 14
rbox(ax_b, 1, scene_y0, 98, scene_h, fc=PALETTE["stage_bg"],
     ec=PALETTE["stage_edge"], lw=0.6, r=0.3)
aisle_y = scene_y0 + scene_h / 2

# Aisle borders
ax_b.plot([6, 94], [aisle_y - 3.6, aisle_y - 3.6],
          color=PALETTE["stage_edge"], lw=0.5)
ax_b.plot([6, 94], [aisle_y + 3.6, aisle_y + 3.6],
          color=PALETTE["stage_edge"], lw=0.5)

# Agent A yields (moves up-right), Agent B passes through
ax_A = 28; ay_A = aisle_y + 1.8
ax_B = 65; ay_B = aisle_y

body_A = Circle((ax_A, ay_A), 2.2, facecolor=PALETTE["red"],
                edgecolor=PALETTE["ink"], lw=0.7, zorder=5)
body_B = Circle((ax_B, ay_B), 2.2, facecolor=PALETTE["blue"],
                edgecolor=PALETTE["ink"], lw=0.7, zorder=5)
ax_b.add_patch(body_A); ax_b.add_patch(body_B)
ax_b.text(ax_A, ay_A, "A", ha="center", va="center", fontsize=5.6,
          weight="bold", color="white", zorder=6)
ax_b.text(ax_B, ay_B, "B", ha="center", va="center", fontsize=5.6,
          weight="bold", color="white", zorder=6)

# Heading arrows
arrow(ax_b, ax_A, ay_A, ax_A + 4.5, ay_A + 1.4,
      color=PALETTE["red"], lw=1.0, mut=6)
arrow(ax_b, ax_B, ay_B, ax_B - 5.0, ay_B,
      color=PALETTE["blue"], lw=1.0, mut=6)

# Annotations
text(ax_b, ax_A, ay_A - 4.6, "yields", size=5.4,
     color=PALETTE["red"], weight="bold")
text(ax_b, ax_B, ay_B - 4.6, "passes through", size=5.4,
     color=PALETTE["blue"], weight="bold")
text(ax_b, 50, scene_y0 + scene_h - 1.8,
     "Both agents reach their goals — same shared $r_t{=}{+}1.0$",
     size=5.6, color=PALETTE["ink"], weight="bold")

# Comparison rows below: shared reward vs. counterfactual reward
row_top = 32
row_h = 8.0
row_gap = 2.0

# Header
text(ax_b,  4.0, row_top + row_h + 1.0, "view",   size=5.4,
     weight="bold", color=PALETTE["ink_soft"], ha="left")
text(ax_b, 28.0, row_top + row_h + 1.0, "Agent A",
     size=5.4, weight="bold", color=PALETTE["red"], ha="center")
text(ax_b, 56.0, row_top + row_h + 1.0, "Agent B",
     size=5.4, weight="bold", color=PALETTE["blue"], ha="center")
text(ax_b, 85.0, row_top + row_h + 1.0, "credit?",
     size=5.4, weight="bold", color=PALETTE["ink_soft"], ha="center")

# Row 1 — shared reward (ambiguous)
y1 = row_top
rbox(ax_b, 1, y1, 98, row_h, fc=PALETTE["red_soft"], ec=PALETTE["red"],
     lw=0.6, r=0.25)
text(ax_b, 4.0, y1 + row_h / 2, "shared $r_t$",
     size=5.6, ha="left", color=PALETTE["red"], weight="bold")
text(ax_b, 28.0, y1 + row_h / 2, "$+1.0$",
     size=7.4, ha="center", weight="bold")
text(ax_b, 56.0, y1 + row_h / 2, "$+1.0$",
     size=7.4, ha="center", weight="bold")
text(ax_b, 85.0, y1 + row_h / 2, "ambiguous", size=5.6,
     ha="center", color=PALETTE["red"], style="italic", weight="bold")

# Row 2 — counterfactual phi
y2 = y1 - row_h - row_gap
rbox(ax_b, 1, y2, 98, row_h, fc=PALETTE["green_soft"], ec=PALETTE["green"],
     lw=0.6, r=0.25)
text(ax_b, 4.0, y2 + row_h / 2, "$\\phi_i$",
     size=6.8, ha="left", color=PALETTE["green"], weight="bold")
text(ax_b, 28.0, y2 + row_h / 2, "$+0.72$",
     size=7.4, ha="center", weight="bold", color=PALETTE["green"])
text(ax_b, 56.0, y2 + row_h / 2, "$+0.08$",
     size=7.4, ha="center", weight="bold", color=PALETTE["ink"])
text(ax_b, 85.0, y2 + row_h / 2, "A enables B", size=5.6,
     ha="center", color=PALETTE["green"], style="italic", weight="bold")

# Row 3 — shaped reward (sum)
y3 = y2 - row_h - row_gap
rbox(ax_b, 1, y3, 98, row_h, fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"],
     lw=0.6, r=0.25)
text(ax_b, 4.0, y3 + row_h / 2, "$r^{i,\\mathrm{sh}}_t$",
     size=6.0, ha="left", weight="bold")
text(ax_b, 28.0, y3 + row_h / 2, "$+1.11$",
     size=7.4, ha="center", weight="bold")
text(ax_b, 56.0, y3 + row_h / 2, "$+1.01$",
     size=7.4, ha="center", weight="bold")
text(ax_b, 85.0, y3 + row_h / 2, "yielding rewarded", size=5.6,
     ha="center", color=PALETTE["ink"], style="italic", weight="bold")

# Footer formula reminder
text(ax_b, 50, 1.2,
     "$r^{i,\\mathrm{sh}}_t = r^i_t + \\lambda_{cf}\\widetilde\\phi_i$"
     "  with  $\\lambda_{cf}{=}0.15$,  clip $\\pm 2.5$",
     size=5.0, color=PALETTE["ink_soft"], ha="center", style="italic")

save_both(fig, "fig5_counterfactual_credit")
print("Fig5 written.")
