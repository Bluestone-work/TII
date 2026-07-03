"""Fig 1 — System architecture of the dual-graph counterfactual MAPPO framework."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

from _figstyle import (
    PALETTE, apply_style, blank_ax, rbox, stage_panel,
    text, arrow, labeled_box, save_both,
)

apply_style(font_size=7.0)

fig, ax = plt.subplots(figsize=(7.16, 4.0))
blank_ax(ax, xlim=(0, 100), ylim=(0, 56))

# Stage panels
S_W, S_H = 22.0, 46.0
S_Y = 2.5
GAP = 2.6
S_X = [2.1 + i * (S_W + GAP) for i in range(4)]
ACCENTS = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"], PALETTE["violet"]]
TITLES = [
    "1. Observation",
    "2. Dual-graph encoder",
    "3. Policy network",
    "4. Centralized training",
]
for x, t, c in zip(S_X, TITLES, ACCENTS):
    stage_panel(ax, x, S_Y, S_W, S_H, t, c, title_fs=7.4, bar_h=2.6)

# Inter-stage anchor y
MID_Y = S_Y + S_H * 0.50

# ============================================================
# Stage 1 — Observation construction
# ============================================================
x0 = S_X[0]
inner_w = S_W - 3.4
inner_x = x0 + 1.7

items_s1 = [
    ("LiDAR history",      "4 × 9 sectors  (36-D)",      PALETTE["blue_soft"]),
    ("Rolling subgoal",    "body-frame target (6-D)",    PALETTE["blue_soft"]),
    ("Ego-motion",         "$v,\\;\\omega$  (2-D)",      PALETTE["blue_soft"]),
    ("Base safety",        "min-range + flags (7-D)",    PALETTE["blue_soft"]),
    ("Social-risk tokens", "top-2 neighbors (12-D)",     PALETTE["red_soft"]),
    ("Obstacle-motion",    "top-3 sectors (21-D)",       PALETTE["amber_soft"]),
]
n = len(items_s1)
top_y = S_Y + S_H - 6.0
bot_y = S_Y + 9.0
row_h = 2.5
gap_h = ((top_y - bot_y) - n * row_h) / (n - 1)
for i, (ttl, sub, fc) in enumerate(items_s1):
    y = top_y - i * (row_h + gap_h) - row_h
    rbox(ax, inner_x, y, inner_w, row_h, fc=fc, ec=PALETTE["ink_soft"], lw=0.6, r=0.18)
    text(ax, inner_x + 0.6, y + row_h * 0.64, ttl, size=6.4, ha="left", weight="bold")
    text(ax, inner_x + 0.6, y + row_h * 0.22, sub, size=5.8, ha="left",
         color=PALETTE["ink_soft"])

# Output chip — local observation
chip_h = 2.8
out_y = S_Y + 4.6
rbox(ax, inner_x, out_y, inner_w, chip_h, fc=PALETTE["ink"], ec=PALETTE["ink"], lw=0, r=0.2)
text(ax, inner_x + inner_w / 2, out_y + chip_h / 2, "$o_t^i$  (84-D local obs.)",
     size=7.0, color="white", weight="bold")

# Vertical bracket from items into obs chip
arrow(ax, inner_x + inner_w / 2, bot_y - 0.4,
      inner_x + inner_w / 2, out_y + chip_h + 0.05,
      color=PALETTE["ink_soft"], lw=0.9, mut=7)

# ============================================================
# Stage 2 — Dual-graph encoding
# ============================================================
x1 = S_X[1]
gx = x1 + 1.5
gw = S_W - 3.0

# Slot top half (social) and bottom half (obstacle)
g1_y_title = S_Y + S_H - 6.8
g2_y_title = S_Y + 16.5

def draw_minigraph(cx, cy, n_neigh, ego_color, neigh_color, seed):
    r = 0.95
    e = Circle((cx, cy), r, facecolor=ego_color, edgecolor=PALETTE["ink"], lw=0.7, zorder=5)
    ax.add_patch(e)
    text(ax, cx, cy, "ego", size=5.4, weight="bold", color="white")
    R = 3.2
    if n_neigh == 2:
        angles = np.deg2rad([135, 45])
    else:
        angles = np.deg2rad([155, 90, 25])
    rng = np.random.RandomState(seed)
    weights = 0.5 + 1.8 * rng.rand(len(angles))
    for a, lw in zip(angles, weights):
        nx, ny = cx + R * np.cos(a), cy + R * np.sin(a)
        nc = Circle((nx, ny), r * 0.85, facecolor=neigh_color,
                    edgecolor=PALETTE["ink"], lw=0.6, zorder=5)
        ax.add_patch(nc)
        arrow(ax, nx, ny, cx, cy, color=PALETTE["ink_soft"], lw=lw, mut=5, z=4)

g_cx = gx + gw / 2

text(ax, g_cx, g1_y_title, "Social-risk graph", size=6.8, weight="bold",
     color=PALETTE["red"])
text(ax, g_cx, g1_y_title - 1.5, "ego $+\\;K_s{=}2$ top-TTC neighbors",
     size=5.9, color=PALETTE["ink_soft"])
draw_minigraph(g_cx, g1_y_title - 6.0, 2, PALETTE["red"], PALETTE["red_soft"], seed=3)

text(ax, g_cx, g2_y_title, "Dynamic-obstacle graph", size=6.8, weight="bold",
     color=PALETTE["amber"])
text(ax, g_cx, g2_y_title - 1.5, "ego $+\\;K_o{=}3$ top-risk sectors",
     size=5.9, color=PALETTE["ink_soft"])
draw_minigraph(g_cx, g2_y_title - 6.0, 3, PALETTE["amber"], PALETTE["amber_soft"], seed=7)

# Divider + GAT equation between the two
mid_div_y = (g1_y_title - 12 + g2_y_title) / 2 + 0.5
ax.plot([gx, gx + gw], [mid_div_y, mid_div_y], color=PALETTE["stage_edge"],
        lw=0.6, linestyle=(0, (3, 2)), zorder=2)
text(ax, g_cx, mid_div_y - 0.0 + 1.2,
     "GAT  $e_{ij}=\\mathrm{LReLU}(\\mathbf{a}^\\top[\\mathbf{W}h_i\\Vert\\mathbf{W}h_j])+\\beta r_j$",
     size=5.5, color=PALETTE["ink_soft"])

# h_soc / h_obs output chips at right edge of stage 2
chip_y_soc = g1_y_title - 6.0
chip_y_obs = g2_y_title - 6.0
chip_x = x1 + S_W - 3.6
rbox(ax, chip_x, chip_y_soc - 1.1, 3.0, 2.2, fc="white",
     ec=PALETTE["red"], lw=0.9, r=0.22, z=6)
text(ax, chip_x + 1.5, chip_y_soc, "$h^i_{\\mathrm{soc}}$", size=6.6, weight="bold",
     color=PALETTE["red"], z=7)
rbox(ax, chip_x, chip_y_obs - 1.1, 3.0, 2.2, fc="white",
     ec=PALETTE["amber"], lw=0.9, r=0.22, z=6)
text(ax, chip_x + 1.5, chip_y_obs, "$h^i_{\\mathrm{obs}}$", size=6.6, weight="bold",
     color=PALETTE["amber"], z=7)

# ============================================================
# Stage 3 — Policy network with residual gate (compact)
# ============================================================
x2 = S_X[2]
px = x2 + 1.6
pw = S_W - 3.2

# Backbone: MLP-LSTM
y_lstm = S_Y + S_H - 9.0
labeled_box(ax, px, y_lstm, pw, 3.4, "MLP-LSTM backbone",
            fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"], lw=0.7, ts=6.6,
            sub="$h^i_{\\mathrm{lstm}}\\in\\mathbb{R}^{256}$")

# Fusion MLP
y_fuse = S_Y + S_H - 16.0
labeled_box(ax, px, y_fuse, pw, 3.4, "Fusion MLP",
            fc=PALETTE["violet_soft"], ec=PALETTE["violet"], lw=0.8, ts=6.6,
            sub="$h^i_{\\mathrm{graph}}=\\mathrm{MLP}[h^i_{\\mathrm{soc}};h^i_{\\mathrm{obs}}]$")

# Residual risk gate (compact, single block; see Fig 4 for internals)
y_gate = S_Y + S_H - 25.5
gh = 6.6
rbox(ax, px, y_gate, pw, gh, fc=PALETTE["amber_soft"], ec=PALETTE["amber"],
     lw=0.9, r=0.3)
text(ax, px + pw / 2, y_gate + gh - 1.4, "Residual risk gate",
     size=6.8, weight="bold")
text(ax, px + pw / 2, y_gate + gh - 3.2,
     "$g^i=\\sigma(W_g\\,[h^i_{\\mathrm{lstm}};h^i_{\\mathrm{graph}}])$",
     size=5.8, color=PALETTE["ink_soft"])
text(ax, px + pw / 2, y_gate + gh - 4.7,
     "$\\Delta h^i=\\mathrm{MLP}_\\Delta(h^i_{\\mathrm{graph}})$",
     size=5.8, color=PALETTE["ink_soft"])
text(ax, px + pw / 2, y_gate + 0.9,
     "$h^i_{\\mathrm{policy}}=\\mathrm{LN}(h^i_{\\mathrm{lstm}}+g^i\\odot\\Delta h^i)$",
     size=6.0, weight="bold", color=PALETTE["amber"])

# Action head
y_pi = S_Y + 3.4
labeled_box(ax, px, y_pi, pw, 3.4, "Action head",
            fc=PALETTE["blue_soft"], ec=PALETTE["blue"], lw=0.8, ts=6.4,
            sub="$\\pi_\\theta(a^i_t\\mid o^i_t,h^i_t)\\sim\\mathcal{N}(\\mu^i,\\sigma)$")

# Vertical arrows inside stage 3 (clean centerline)
arrow(ax, px + pw / 2, y_lstm, px + pw / 2, y_fuse + 3.4 + 0.05,
      color=PALETTE["ink_soft"], lw=0.9, mut=7)
arrow(ax, px + pw / 2, y_fuse, px + pw / 2, y_gate + gh + 0.05,
      color=PALETTE["violet"], lw=0.9, mut=7)
arrow(ax, px + pw / 2, y_gate, px + pw / 2, y_pi + 3.4 + 0.05,
      color=PALETTE["amber"], lw=0.9, mut=7)

# Residual skip on the right side (h_lstm bypasses Fusion into gate)
res_x = px + pw - 0.8
arrow(ax, px + pw, y_lstm + 1.7, res_x, y_lstm + 1.7,
      color=PALETTE["blue"], lw=0.9, mut=0, style="-")
ax.plot([res_x, res_x], [y_lstm + 1.7, y_gate + gh * 0.5],
        color=PALETTE["blue"], lw=1.0, linestyle=(0, (2.5, 1.5)), zorder=4)
arrow(ax, res_x, y_gate + gh * 0.5, px + pw - 0.05, y_gate + gh * 0.5,
      color=PALETTE["blue"], lw=1.0, mut=7, ls=(0, (2.5, 1.5)))
text(ax, res_x + 0.4, (y_lstm + y_gate + gh / 2) / 2, "residual",
     size=5.6, color=PALETTE["blue"], ha="left", weight="bold", z=6)

# ============================================================
# Stage 4 — Centralized training
# ============================================================
x3 = S_X[3]
tx = x3 + 1.6
tw = S_W - 3.2

y_c1 = S_Y + S_H - 9.0
labeled_box(ax, tx, y_c1, tw, 3.4, "Centralized critic",
            fc=PALETTE["stage_bg"], ec=PALETTE["ink_soft"], lw=0.7, ts=6.6,
            sub="$V_\\psi(s_t)\\;\\;\\;V_\\psi(s_t\\setminus i)$")

y_phi = S_Y + S_H - 16.0
labeled_box(ax, tx, y_phi, tw, 3.4, "Marginal contribution",
            fc=PALETTE["green_soft"], ec=PALETTE["green"], lw=0.8, ts=6.6,
            sub="$\\phi_i=V(s_t)-V(s_t\\setminus i)$")

y_rs = S_Y + S_H - 23.0
labeled_box(ax, tx, y_rs, tw, 3.4, "Counterfactual reward",
            fc=PALETTE["red_soft"], ec=PALETTE["red"], lw=0.8, ts=6.4,
            sub="$r^{i,\\mathrm{sh}}_t=r^i_t+\\lambda_{cf}\\widetilde\\phi_i$")

y_ppo = S_Y + S_H - 30.0
labeled_box(ax, tx, y_ppo, tw, 3.4, "GAE  +  PPO update",
            fc=PALETTE["blue_soft"], ec=PALETTE["blue"], lw=0.8, ts=6.6,
            sub="clip ratio  $\\epsilon=0.05$")

y_pd = S_Y + 3.4
labeled_box(ax, tx, y_pd, tw, 3.4, "Partial-done protocol",
            fc=PALETTE["amber_soft"], ec=PALETTE["amber"], lw=0.8, ts=6.6,
            sub="early cutoff + $\\rho=\\sqrt{N/4}$")

for ya, yb in [(y_c1, y_phi), (y_phi, y_rs), (y_rs, y_ppo), (y_ppo, y_pd)]:
    arrow(ax, tx + tw / 2, ya, tx + tw / 2, yb + 3.4 + 0.05,
          color=PALETTE["ink_soft"], lw=0.9, mut=7)

# ============================================================
# Inter-stage arrows
# ============================================================
# Stage 1 -> Stage 2  (obs into graphs)
arrow(ax, S_X[0] + S_W, MID_Y, S_X[1], MID_Y,
      color=PALETTE["ink"], lw=1.3, mut=10, z=7)
text(ax, (S_X[0] + S_W + S_X[1]) / 2, MID_Y + 1.4, "$o^i_t$", size=6.4,
     color=PALETTE["ink_soft"])

# Stage 1 obs also feeds LSTM in Stage 3 directly (routed over the top)
y_top_route = S_Y + S_H + 0.9
ax.plot([S_X[0] + S_W * 0.5, S_X[0] + S_W * 0.5], [S_Y + S_H, y_top_route],
        color=PALETTE["ink_soft"], lw=0.7, linestyle=(0, (2.5, 1.5)), zorder=6)
ax.plot([S_X[0] + S_W * 0.5, px + pw / 2], [y_top_route, y_top_route],
        color=PALETTE["ink_soft"], lw=0.7, linestyle=(0, (2.5, 1.5)), zorder=6)
arrow(ax, px + pw / 2, y_top_route, px + pw / 2, y_lstm + 3.4 + 0.05,
      color=PALETTE["ink_soft"], lw=0.7, mut=6, ls=(0, (2.5, 1.5)), z=6)

# Stage 2 -> Stage 3  (two graph embeddings -> Fusion)
arrow(ax, chip_x + 3.0, chip_y_soc, S_X[2], y_fuse + 3.4 * 0.75,
      color=PALETTE["red"], lw=1.1, mut=8, z=7)
arrow(ax, chip_x + 3.0, chip_y_obs, S_X[2], y_fuse + 3.4 * 0.25,
      color=PALETTE["amber"], lw=1.1, mut=8, z=7)

# Stage 3 -> Stage 4  (rollout)
arrow(ax, S_X[2] + S_W, MID_Y, S_X[3], MID_Y,
      color=PALETTE["ink"], lw=1.3, mut=10, z=7)
text(ax, (S_X[2] + S_W + S_X[3]) / 2, MID_Y + 1.4, "rollout $(s,a,r)$",
     size=6.0, color=PALETTE["ink_soft"])

# Stage 4 -> Stage 3 feedback (policy update) over the top
y_fb = S_Y + S_H + 2.6
arrow(ax, S_X[3] + S_W * 0.5, S_Y + S_H, S_X[3] + S_W * 0.5, y_fb,
      color=PALETTE["ink"], lw=1.0, mut=0, style="-")
ax.plot([S_X[3] + S_W * 0.5, S_X[2] + S_W * 0.5], [y_fb, y_fb],
        color=PALETTE["ink"], lw=1.0, zorder=7)
arrow(ax, S_X[2] + S_W * 0.5, y_fb, S_X[2] + S_W * 0.5, S_Y + S_H + 0.05,
      color=PALETTE["ink"], lw=1.0, mut=7, z=7)
text(ax, (S_X[2] + S_X[3] + S_W) / 2, y_fb + 0.7, "policy update",
     size=6.0, color=PALETTE["ink"])

# Footer note
text(ax, 50, 0.6,
     "CTDE — decentralized execution  •  centralized training",
     size=5.8, color=PALETTE["muted"])

save_both(fig, "fig1_system_architecture")
print("Fig1 written.")
