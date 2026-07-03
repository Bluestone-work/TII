"""Fig 3 — Dual-graph attention architecture.

Three stacked panels on a single-column canvas:
 (a) Social-risk graph — ego + top-K_s=2 TTC-ranked neighbors; faded neighbors
     are pruned; edge thickness ∝ attention weight; bias term β·r_social.
 (b) Dynamic-obstacle graph — ego + top-K_o=3 risk-ranked LiDAR sectors; LiDAR
     fan illustrates obstacle motion tokens.
 (c) One-head GAT computation — risk-biased attention from raw features to
     updated ego embedding.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle, Polygon, Rectangle, Wedge, FancyArrowPatch, FancyBboxPatch,
)
import numpy as np

from _figstyle import (
    PALETTE, apply_style, blank_ax, rbox, text, arrow, labeled_box, save_both,
)

apply_style(font_size=7.0)

fig = plt.figure(figsize=(3.5, 5.0))
gs = fig.add_gridspec(
    3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.30,
    left=0.02, right=0.98, top=0.97, bottom=0.04,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ============================================================
# Panel a — Social-risk graph (ego + top-K_s=2 neighbors)
# ============================================================
blank_ax(ax_a, xlim=(0, 100), ylim=(0, 60))
ax_a.set_aspect("auto")
ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 60)

text(ax_a, 0.5, 57.5, "a", size=9.5, weight="bold", ha="left")
text(ax_a, 6.5, 57.5, "Social-risk graph  •  top-$K_s{=}2$ by predicted TTC",
     size=6.6, ha="left", color=PALETTE["ink"])

# Subpanel: graph (left, ~55%) + ranking column (right, ~45%)
GRAPH_X0, GRAPH_X1 = 1, 54
graph_cx = (GRAPH_X0 + GRAPH_X1) / 2
graph_cy = 28

# Ego node
ego_color = PALETTE["red"]
ego_r = 3.6
eg = Circle((graph_cx, graph_cy), ego_r, facecolor=ego_color,
            edgecolor=PALETTE["ink"], lw=0.8, zorder=6)
ax_a.add_patch(eg)
ax_a.text(graph_cx, graph_cy, "ego $i$", ha="center", va="center",
          fontsize=6.0, color="white", weight="bold", zorder=7)

# Five candidate neighbors at varying distances; rank by risk
# (label, angle_deg, distance, ttc, selected)
neighbors = [
    ("$j_1$", 150, 16.0, 1.2, True),
    ("$j_2$",  35, 14.5, 1.6, True),
    ("$j_3$",  -55, 20.0, 3.8, False),
    ("$j_4$",  100, 22.0, 4.2, False),
    ("$j_5$",  -150, 19.0, 5.0, False),
]

def attn_lw(ttc, selected):
    if not selected:
        return 0.5
    return 0.6 + 3.4 * (1.6 - min(ttc, 1.6)) / 1.6 + 1.4

for lbl, deg, r, ttc, sel in neighbors:
    a = np.deg2rad(deg)
    nx, ny = graph_cx + r * np.cos(a), graph_cy + r * np.sin(a) * 0.85
    fc = PALETTE["red_soft"] if sel else "#F2F2F2"
    ec = PALETTE["red"] if sel else PALETTE["muted"]
    nb = Circle((nx, ny), 2.4, facecolor=fc, edgecolor=ec, lw=0.7, zorder=5)
    ax_a.add_patch(nb)
    ax_a.text(nx, ny, lbl, ha="center", va="center", fontsize=5.4,
              color=PALETTE["ink"] if sel else PALETTE["muted"],
              weight="bold", zorder=6)
    # Edge — pruned (dashed faint) or selected (solid weighted)
    if sel:
        # Edge thickness ∝ attention weight
        lw = attn_lw(ttc, True)
        arrow(ax_a, nx, ny, graph_cx, graph_cy, color=PALETTE["red"],
              lw=lw, mut=6, z=4)
    else:
        # Pruned: dotted faded
        ax_a.plot([nx, graph_cx], [ny, graph_cy], color=PALETTE["muted"],
                  lw=0.5, linestyle=(0, (1.0, 1.8)), zorder=3)

# Risk-rank column on the right
rank_x = 56
rank_w = 42
text(ax_a, rank_x + rank_w / 2, 53.5, "risk score  $r_{\\mathrm{soc}}$",
     size=5.8, weight="bold", color=PALETTE["red"])
ranked = sorted(neighbors, key=lambda v: v[3])
bar_top = 50
bar_h = 4.6
gap = 1.2
for i, (lbl, deg, r, ttc, sel) in enumerate(ranked):
    y = bar_top - i * (bar_h + gap) - bar_h
    fc = PALETTE["red_soft"] if sel else "#F2F2F2"
    ec = PALETTE["red"] if sel else PALETTE["muted"]
    rbox(ax_a, rank_x, y, rank_w, bar_h, fc=fc, ec=ec, lw=0.6, r=0.2)
    score_norm = 1.0 / (ttc + 0.4)
    bar_w_max = rank_w - 12
    sb_w = score_norm * bar_w_max * 0.85
    rbox(ax_a, rank_x + 9.0, y + 0.9, sb_w, bar_h - 1.8,
         fc=PALETTE["red"] if sel else PALETTE["muted"],
         ec="none", lw=0, r=0.05, z=3)
    text(ax_a, rank_x + 1.2, y + bar_h / 2, lbl, size=5.4,
         ha="left", weight="bold",
         color=PALETTE["ink"] if sel else PALETTE["muted"])
    text(ax_a, rank_x + 7.5, y + bar_h / 2, "TTC=" + f"{ttc:.1f}s",
         size=4.6, ha="left", color=PALETTE["ink_soft"])

text(ax_a, rank_x + rank_w / 2, 1.2,
     "selected (top-$K_s$)  •  pruned",
     size=5.0, color=PALETTE["ink_soft"], ha="center")

# Selection bracket on graph side
text(ax_a, 27, 5.5, "edge thickness $\\propto$ attention $\\alpha_{ij}$",
     size=5.2, color=PALETTE["ink_soft"], ha="center", style="italic")

# ============================================================
# Panel b — Dynamic-obstacle graph (ego + top-K_o=3 sectors)
# ============================================================
blank_ax(ax_b, xlim=(0, 100), ylim=(0, 60))
ax_b.set_aspect("auto")
ax_b.set_xlim(0, 100); ax_b.set_ylim(0, 60)

text(ax_b, 0.5, 57.5, "b", size=9.5, weight="bold", ha="left")
text(ax_b, 6.5, 57.5, "Dynamic-obstacle graph  •  top-$K_o{=}3$ by predicted risk",
     size=6.6, ha="left", color=PALETTE["ink"])

# Ego with LiDAR fan on the left
ego_cx, ego_cy = 22, 27
fan_r = 16
fan_color = PALETTE["amber_soft"]
# 9 sectors covering 180°  (front fan)
sector_specs = [
    # (angle deg from horizontal, risk: high/med/low, selected)
    (-80, "low",  False),
    (-60, "low",  False),
    (-40, "med",  False),
    (-20, "high", True),   # top-1
    (  0, "high", True),   # top-2
    ( 20, "med",  False),
    ( 40, "high", True),   # top-3
    ( 60, "low",  False),
    ( 80, "low",  False),
]
sector_width = 20  # degrees
for ang, level, sel in sector_specs:
    color = {
        "high": PALETTE["amber"],
        "med":  PALETTE["amber_soft"],
        "low":  "#F2F2F2",
    }[level]
    a0 = ang - sector_width / 2
    a1 = ang + sector_width / 2
    w = Wedge((ego_cx, ego_cy), fan_r, a0, a1, width=fan_r - 0.6,
              facecolor=color, edgecolor=PALETTE["stage_edge"],
              lw=0.4, zorder=2, alpha=0.85 if sel else 0.55)
    ax_b.add_patch(w)
    if sel:
        # outline the selected sector more boldly
        w2 = Wedge((ego_cx, ego_cy), fan_r, a0, a1, width=fan_r - 0.6,
                   facecolor="none", edgecolor=PALETTE["amber"],
                   lw=1.2, zorder=3)
        ax_b.add_patch(w2)

# Ego node on top of fan origin
eg = Circle((ego_cx, ego_cy), 2.4, facecolor=PALETTE["amber"],
            edgecolor=PALETTE["ink"], lw=0.7, zorder=6)
ax_b.add_patch(eg)
ax_b.text(ego_cx, ego_cy, "ego", ha="center", va="center", fontsize=5.2,
          color="white", weight="bold", zorder=7)

# Obstacle markers placed inside the top-3 sectors
obstacles = [
    (-20, 11.0, "$o_1$", True),
    (  0, 12.5, "$o_2$", True),
    ( 40,  9.0, "$o_3$", True),
    ( 60, 13.0, "",      False),
]
for ang, r, lbl, sel in obstacles:
    a = np.deg2rad(ang)
    ox = ego_cx + r * np.cos(a)
    oy = ego_cy + r * np.sin(a)
    if sel:
        diamond = Polygon([(ox, oy + 1.4), (ox + 1.4, oy),
                           (ox, oy - 1.4), (ox - 1.4, oy)],
                          facecolor=PALETTE["violet"],
                          edgecolor=PALETTE["ink"], lw=0.5, zorder=5)
        ax_b.add_patch(diamond)
        ax_b.text(ox + 1.9, oy, lbl, fontsize=5.2, color=PALETTE["violet"],
                  weight="bold", ha="left", va="center", zorder=6)
        # velocity arrow
        arrow(ax_b, ox, oy, ox - 3 * np.cos(a), oy - 3 * np.sin(a),
              color=PALETTE["violet"], lw=0.7, mut=5)
        # edge from selected obstacle to ego — risk-weighted thickness
        arrow(ax_b, ox, oy, ego_cx, ego_cy, color=PALETTE["amber"],
              lw=0.6 + 2.0 * (1.0 / (r * 0.18)), mut=5, z=4)
    else:
        dm = Polygon([(ox, oy + 0.9), (ox + 0.9, oy),
                      (ox, oy - 0.9), (ox - 0.9, oy)],
                     facecolor=PALETTE["muted"], edgecolor=PALETTE["muted"],
                     lw=0.3, zorder=5)
        ax_b.add_patch(dm)

# Heading indicator (ego front direction)
arrow(ax_b, ego_cx, ego_cy, ego_cx + 4.0, ego_cy, color=PALETTE["ink"],
      lw=0.7, mut=5, z=7)

# Right side — sector ranking column
rank_x = 56; rank_w = 42
text(ax_b, rank_x + rank_w / 2, 53.5,
     "risk score  $r_{\\mathrm{obs}}$",
     size=5.8, weight="bold", color=PALETTE["amber"])
sector_ranking = [
    ("$s_1{:}\\;{+}40^\\circ$", 0.92, True),
    ("$s_2{:}\\;{-}20^\\circ$", 0.84, True),
    ("$s_3{:}\\;0^\\circ$",      0.74, True),
    ("$s_4{:}\\;{-}40^\\circ$", 0.42, False),
    ("$s_5{:}\\;{+}60^\\circ$", 0.30, False),
]
bar_top = 50
bar_h = 4.6
gap = 1.2
for i, (lbl, score, sel) in enumerate(sector_ranking):
    y = bar_top - i * (bar_h + gap) - bar_h
    fc = PALETTE["amber_soft"] if sel else "#F2F2F2"
    ec = PALETTE["amber"] if sel else PALETTE["muted"]
    rbox(ax_b, rank_x, y, rank_w, bar_h, fc=fc, ec=ec, lw=0.6, r=0.2)
    bar_w_max = rank_w - 12
    rbox(ax_b, rank_x + 9.0, y + 0.9, score * bar_w_max * 0.85, bar_h - 1.8,
         fc=PALETTE["amber"] if sel else PALETTE["muted"],
         ec="none", lw=0, r=0.05, z=3)
    text(ax_b, rank_x + 1.2, y + bar_h / 2, lbl, size=5.0,
         ha="left", weight="bold",
         color=PALETTE["ink"] if sel else PALETTE["muted"])

text(ax_b, 22, 5.5, "9 LiDAR sectors, fan radius $\\sim$3.5 m",
     size=5.0, color=PALETTE["ink_soft"], ha="center", style="italic")

# ============================================================
# Panel c — Risk-biased attention computation (single head)
# ============================================================
blank_ax(ax_c, xlim=(0, 100), ylim=(0, 38))
ax_c.set_aspect("auto")
ax_c.set_xlim(0, 100); ax_c.set_ylim(0, 38)

text(ax_c, 0.5, 35.5, "c", size=9.5, weight="bold", ha="left")
text(ax_c, 6.5, 35.5, "Risk-biased single-head attention",
     size=6.6, ha="left", color=PALETTE["ink"])

# Five stages: features → Wh → concat+LReLU → +β r_j → softmax → aggregate
stage_y = 12
stage_h = 14

def stage_box(x, w, ttl, sub=None, fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"]):
    rbox(ax_c, x, stage_y, w, stage_h, fc=fc, ec=ec, lw=0.7, r=0.25)
    text(ax_c, x + w / 2, stage_y + stage_h - 2.2, ttl, size=5.6,
         weight="bold")
    if sub:
        text(ax_c, x + w / 2, stage_y + stage_h - 5.0, sub, size=5.4,
             color=PALETTE["ink_soft"])

w1 = 16; w2 = 17; w3 = 17; w4 = 13; w5 = 17
gap = 1.3
x1 = 2
x2 = x1 + w1 + gap
x3 = x2 + w2 + gap
x4 = x3 + w3 + gap
x5 = x4 + w4 + gap

# 1. Node features
stage_box(x1, w1, "node feat.",
          sub="$h_i,\\,h_j\\in\\mathbb{R}^{128}$",
          fc=PALETTE["red_soft"], ec=PALETTE["red"])
# 2. Linear proj
stage_box(x2, w2, "linear",
          sub="$\\mathbf{W}h_i,\\;\\mathbf{W}h_j$")
# 3. Concat + LeakyReLU
stage_box(x3, w3, "concat + LReLU",
          sub="$\\mathbf{a}^\\top[\\mathbf{W}h_i\\Vert\\mathbf{W}h_j]$")
# 4. +β r_j bias  (small chip)
stage_box(x4, w4, "risk bias",
          sub="$+\\beta\\,r_j$",
          fc=PALETTE["amber_soft"], ec=PALETTE["amber"])
# 5. softmax → aggregate
stage_box(x5, w5, "softmax + agg.",
          sub="$h^i_{\\mathrm{out}}=\\sum_j\\alpha_{ij}\\mathbf{W}h_j$",
          fc=PALETTE["red_soft"], ec=PALETTE["red"])

# Arrows between stages
for xa, wa, xb in [(x1, w1, x2), (x2, w2, x3), (x3, w3, x4), (x4, w4, x5)]:
    arrow(ax_c, xa + wa, stage_y + stage_h / 2,
          xb, stage_y + stage_h / 2,
          color=PALETTE["ink_soft"], lw=0.9, mut=7)

# Footnote about heads + symmetry
text(ax_c, 50, 5.0,
     "Repeated for 4 heads (128-D each), then aggregated to single-head output",
     size=5.0, color=PALETTE["ink_soft"], ha="center", style="italic")
text(ax_c, 50, 1.8,
     "$\\beta{=}2.5$ scales the risk-aware attention bias",
     size=5.0, color=PALETTE["ink_soft"], ha="center", style="italic")

save_both(fig, "fig3_dual_graph_attention")
print("Fig3 written.")
