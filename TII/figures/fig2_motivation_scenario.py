"""Fig 2 — Motivation scenario for dual-graph counterfactual MAPPO.

Two panels (stacked) on a single-column canvas:
 (a) Warehouse top-down view showing strongly coupled multi-AGV interactions
     (head-on encounter, intersection conflict, dynamic obstacle proximity).
 (b) Shared-reward credit ambiguity — both agents receive the same r_t, but
     only one yielded.  Counterfactual signal phi_i resolves credit.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Wedge,
    Ellipse,
)
import numpy as np

from _figstyle import (
    PALETTE, apply_style, blank_ax, rbox, text, arrow, labeled_box, save_both,
)

apply_style(font_size=7.0)

fig = plt.figure(figsize=(3.5, 4.2))
gs = fig.add_gridspec(
    2, 1, height_ratios=[1.32, 1.0], hspace=0.22,
    left=0.02, right=0.98, top=0.97, bottom=0.04,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ============================================================
# Panel a — Warehouse top-down view
# ============================================================
blank_ax(ax_a, xlim=(0, 100), ylim=(0, 75))
ax_a.set_aspect("equal")
ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 75)

# Panel label
text(ax_a, 0.5, 73.5, "a", size=9.5, weight="bold", ha="left")
text(ax_a, 6.5, 73.5, "Strongly coupled interactions in a warehouse aisle network",
     size=6.6, ha="left", color=PALETTE["ink"])

# Aisle layout — rectangular shelves leaving narrow aisles
shelf_color = PALETTE["stage_bg"]
shelf_edge  = PALETTE["stage_edge"]

shelves = [
    # x, y, w, h
    (6, 50, 26, 12),
    (38, 50, 26, 12),
    (70, 50, 24, 12),
    (6, 14, 26, 26),
    (38, 14, 26, 12),
    (38, 32, 26,  8),
    (70, 14, 24, 26),
]
for x, y, w, h in shelves:
    rect = Rectangle((x, y), w, h, facecolor=shelf_color, edgecolor=shelf_edge,
                     linewidth=0.6, zorder=1)
    ax_a.add_patch(rect)
    # stripe lines on shelf
    for sy in np.linspace(y + 2.0, y + h - 2.0, 4):
        ax_a.plot([x + 1.2, x + w - 1.2], [sy, sy],
                  color=PALETTE["stage_edge"], lw=0.4, zorder=1.1)

# A* path lines (faint, dashed) — show two robots routed through aisles
def path_line(pts, color, lw=0.8):
    pts = np.asarray(pts)
    ax_a.plot(pts[:, 0], pts[:, 1], color=color, lw=lw,
              linestyle=(0, (3.0, 1.6)), zorder=2)

# Robot helper
def robot(ax, x, y, hx, hy, color, label, label_dx=0, label_dy=0):
    """Draw an AGV body (circle) with a small heading triangle."""
    body = Circle((x, y), 1.55, facecolor=color, edgecolor=PALETTE["ink"],
                  lw=0.7, zorder=5)
    ax.add_patch(body)
    h = np.array([hx - x, hy - y], dtype=float)
    norm = np.linalg.norm(h)
    if norm > 1e-6:
        h /= norm
        perp = np.array([-h[1], h[0]])
        tip  = np.array([x, y]) + 1.55 * h
        base_l = np.array([x, y]) + 0.55 * perp - 0.25 * h
        base_r = np.array([x, y]) - 0.55 * perp - 0.25 * h
        tri = Polygon(np.vstack([tip, base_l, base_r]), facecolor="white",
                      edgecolor=PALETTE["ink"], lw=0.5, zorder=6)
        ax.add_patch(tri)
    ax.text(x + label_dx, y + label_dy, label, ha="center", va="center",
            fontsize=5.4, weight="bold", color="white", zorder=7)

# Head-on encounter in horizontal aisle (between top shelves and middle shelves)
# Aisle band y ≈ 41..49.  Two robots converge.
path_line([(8, 45), (35, 45), (60, 45)], PALETTE["red"])      # robot A path
path_line([(94, 45), (60, 45), (35, 45)], PALETTE["blue"])    # robot B path

robot(ax_a, 22, 45, 28, 45, PALETTE["red"],   "A")
robot(ax_a, 48, 45, 42, 45, PALETTE["blue"],  "B")

# Conflict zone marker
conflict = Ellipse((35, 45), 10, 4.6, facecolor=PALETTE["red_soft"],
                   edgecolor=PALETTE["red"], lw=0.7, alpha=0.55, zorder=3)
ax_a.add_patch(conflict)
text(ax_a, 35, 49.6, "head-on", size=5.6, color=PALETTE["red"], weight="bold")

# Intersection conflict (vertical + horizontal aisles meet)
# Vertical aisle between left shelves and middle shelves at x≈33..37, going down
path_line([(35, 64), (35, 45), (35, 12)], PALETTE["teal"])
robot(ax_a, 35, 30, 35, 26, PALETTE["teal"], "C")

# Crossing horizontal aisle (between middle bottom shelves and bottom)
path_line([(94, 28), (66, 28), (35, 28), (10, 28)], PALETTE["amber"])
robot(ax_a, 66, 28, 60, 28, PALETTE["amber"], "D")

# Intersection conflict zone
inter = Ellipse((35, 28), 6.0, 6.0, facecolor=PALETTE["amber_soft"],
                edgecolor=PALETTE["amber"], lw=0.7, alpha=0.55, zorder=3)
ax_a.add_patch(inter)
text(ax_a, 41.5, 31.8, "merge", size=5.6, color=PALETTE["amber"], weight="bold")

# Dynamic obstacle (forklift / human) — diamond marker with velocity arrow
dyn_x, dyn_y = 80, 28
diamond = Polygon([(dyn_x, dyn_y + 1.7), (dyn_x + 1.7, dyn_y),
                   (dyn_x, dyn_y - 1.7), (dyn_x - 1.7, dyn_y)],
                  facecolor=PALETTE["violet"], edgecolor=PALETTE["ink"],
                  lw=0.6, zorder=5)
ax_a.add_patch(diamond)
arrow(ax_a, dyn_x, dyn_y, dyn_x - 7, dyn_y, color=PALETTE["violet"], lw=0.9, mut=6)
text(ax_a, 80, 24.0, "dynamic obs.", size=5.4, color=PALETTE["violet"], ha="center")

# Goal pins
def pin(ax, x, y, color, label):
    star = Polygon(
        [(x, y + 1.6), (x + 0.55, y + 0.5), (x + 1.6, y + 0.3),
         (x + 0.8, y - 0.4), (x + 1.0, y - 1.5), (x, y - 0.9),
         (x - 1.0, y - 1.5), (x - 0.8, y - 0.4), (x - 1.6, y + 0.3),
         (x - 0.55, y + 0.5)],
        facecolor=color, edgecolor=PALETTE["ink"], lw=0.4, zorder=6)
    ax.add_patch(star)
    ax.text(x, y - 2.7, label, ha="center", va="center", fontsize=5.0,
            color=color, weight="bold")

pin(ax_a, 92, 45, PALETTE["red"],   "$g_A$")
pin(ax_a, 12, 45, PALETTE["blue"],  "$g_B$")
pin(ax_a, 35, 8,  PALETTE["teal"],  "$g_C$")
pin(ax_a, 8,  28, PALETTE["amber"], "$g_D$")

# Compact legend on bottom-left over the aisle
text(ax_a, 7, 5.0, "Robots A–D • dashed lines: A* plan • shaded zones: conflict",
     size=5.0, color=PALETTE["ink_soft"], ha="left")

# Pain-point callout (top-right, soft box)
callout_x, callout_y, callout_w, callout_h = 67, 64, 30, 9.5
rbox(ax_a, callout_x, callout_y, callout_w, callout_h,
     fc="white", ec=PALETTE["red"], lw=0.7, r=0.4, z=4)
text(ax_a, callout_x + callout_w / 2, callout_y + callout_h - 1.7,
     "Challenge", size=6.0, color=PALETTE["red"], weight="bold")
text(ax_a, callout_x + 1.2, callout_y + callout_h - 3.7,
     "tightly coupled head-on, merge,",
     size=5.4, ha="left", color=PALETTE["ink"])
text(ax_a, callout_x + 1.2, callout_y + callout_h - 5.5,
     "and congestion under partial",
     size=5.4, ha="left", color=PALETTE["ink"])
text(ax_a, callout_x + 1.2, callout_y + callout_h - 7.3,
     "observation",
     size=5.4, ha="left", color=PALETTE["ink"])

# ============================================================
# Panel b — Shared reward → credit ambiguity
# ============================================================
blank_ax(ax_b, xlim=(0, 100), ylim=(0, 56))
ax_b.set_aspect("auto")
ax_b.set_xlim(0, 100); ax_b.set_ylim(0, 56)

text(ax_b, 0.5, 54.5, "b", size=9.5, weight="bold", ha="left")
text(ax_b, 6.5, 54.5, "Shared reward yields ambiguous individual credit",
     size=6.6, ha="left", color=PALETTE["ink"])

# Two-mini-scene strip showing yield vs. pass-through
def mini_scene(x0, w, y0, h, title, color_yield, color_pass, label_y, label_p, sub_text):
    """Small head-on scene inside a rounded panel."""
    rbox(ax_b, x0, y0, w, h, fc=PALETTE["stage_bg"], ec=PALETTE["stage_edge"],
         lw=0.6, r=0.3)
    text(ax_b, x0 + w / 2, y0 + h - 2.5, title, size=6.0, weight="bold")
    # aisle band
    aisle_y = y0 + 8.0
    ax_b.plot([x0 + 3, x0 + w - 3], [aisle_y - 2.6, aisle_y - 2.6],
              color=PALETTE["stage_edge"], lw=0.5)
    ax_b.plot([x0 + 3, x0 + w - 3], [aisle_y + 2.6, aisle_y + 2.6],
              color=PALETTE["stage_edge"], lw=0.5)
    # robots
    rA_x, rB_x = x0 + 7, x0 + w - 7
    if "yields" in sub_text:
        # left agent steps aside; right passes
        body_A = Circle((rA_x, aisle_y + 1.8), 1.4, facecolor=color_yield,
                        edgecolor=PALETTE["ink"], lw=0.6, zorder=5)
        body_B = Circle((rB_x, aisle_y), 1.4, facecolor=color_pass,
                        edgecolor=PALETTE["ink"], lw=0.6, zorder=5)
        ax_b.add_patch(body_A); ax_b.add_patch(body_B)
        arrow(ax_b, rA_x, aisle_y, rA_x, aisle_y + 1.5, color=color_yield,
              lw=0.9, mut=5)
        arrow(ax_b, rB_x, aisle_y, rB_x - 4, aisle_y, color=color_pass,
              lw=0.9, mut=5)
        ax_b.text(rA_x, aisle_y + 1.8, label_y, ha="center", va="center",
                  fontsize=5.0, weight="bold", color="white", zorder=6)
        ax_b.text(rB_x, aisle_y, label_p, ha="center", va="center",
                  fontsize=5.0, weight="bold", color="white", zorder=6)
        ax_b.text(rA_x - 1, aisle_y + 4.4, "yields", fontsize=5.0,
                  color=color_yield, ha="center")
        ax_b.text(rB_x, aisle_y - 4.0, "passes", fontsize=5.0,
                  color=color_pass, ha="center")
    else:
        # both stubborn — collision marker
        body_A = Circle((rA_x, aisle_y), 1.4, facecolor=color_yield,
                        edgecolor=PALETTE["ink"], lw=0.6, zorder=5)
        body_B = Circle((rB_x, aisle_y), 1.4, facecolor=color_pass,
                        edgecolor=PALETTE["ink"], lw=0.6, zorder=5)
        ax_b.add_patch(body_A); ax_b.add_patch(body_B)
        arrow(ax_b, rA_x, aisle_y, rA_x + 4, aisle_y, color=color_yield,
              lw=0.9, mut=5)
        arrow(ax_b, rB_x, aisle_y, rB_x - 4, aisle_y, color=color_pass,
              lw=0.9, mut=5)
        ax_b.text(rA_x, aisle_y, label_y, ha="center", va="center",
                  fontsize=5.0, weight="bold", color="white", zorder=6)
        ax_b.text(rB_x, aisle_y, label_p, ha="center", va="center",
                  fontsize=5.0, weight="bold", color="white", zorder=6)
        mid_x = (rA_x + rB_x) / 2
        bolt = Polygon([(mid_x - 1.4, aisle_y + 1.4), (mid_x + 0.2, aisle_y + 0.3),
                        (mid_x - 0.5, aisle_y + 0.0), (mid_x + 1.4, aisle_y - 1.6),
                        (mid_x - 0.2, aisle_y - 0.3), (mid_x + 0.5, aisle_y + 0.1)],
                       facecolor=PALETTE["red"], edgecolor=PALETTE["red"],
                       lw=0.3, zorder=7)
        ax_b.add_patch(bolt)

    text(ax_b, x0 + w / 2, y0 + 3.0, sub_text, size=5.4,
         color=PALETTE["ink_soft"])

# Two scenes side by side
mini_scene( 1, 47, 16, 36, "Episode rollout — Agent A yields",
           PALETTE["red"], PALETTE["blue"], "A", "B",
           "Both reach goals  •  shared $r_t$ rewards both")
mini_scene(52, 47, 16, 36, "Counterfactual probe — mask A",
           PALETTE["red"], PALETTE["blue"], "A", "B",
           "Without A's yield  →  conflict, low team value")

# Reward annotations under each scene
def reward_chip(ax, x, y, w, h, text_top, value, color, soft):
    rbox(ax, x, y, w, h, fc=soft, ec=color, lw=0.7, r=0.2)
    ax.text(x + w / 2, y + h * 0.65, text_top, ha="center", va="center",
            fontsize=5.4, color=color, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, value, ha="center", va="center",
            fontsize=6.4, color=PALETTE["ink"], weight="bold")

reward_chip(ax_b, 8,  6, 14, 6, "shared $r_t$", "$+1.0$",
            PALETTE["ink_soft"], PALETTE["stage_bg"])
reward_chip(ax_b, 27, 6, 14, 6, "$V(s_t)$", "high",
            PALETTE["green"], PALETTE["green_soft"])

reward_chip(ax_b, 59, 6, 14, 6, "$V(s_t \\setminus A)$", "low",
            PALETTE["red"], PALETTE["red_soft"])
reward_chip(ax_b, 78, 6, 14, 6, "$\\phi_A = V{-}V\\!\\setminus\\!A$",
            "large $+$", PALETTE["green"], PALETTE["green_soft"])

# Arrow linking the two
arrow(ax_b, 49, 9, 53, 9, color=PALETTE["ink"], lw=1.0, mut=8, z=8)

save_both(fig, "fig2_motivation_scenario")
print("Fig2 written.")
