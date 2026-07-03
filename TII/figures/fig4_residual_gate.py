"""Fig 4 — Residual risk-gated fusion.

Single-column figure with two panels:
 (a) Data-flow diagram. h_lstm (backbone) bypasses the graph encoder via a
     residual skip; h_graph is transformed by MLP_Delta and modulated by a
     learned sigmoid gate g^i; the gated delta is added to h_lstm and
     LayerNormed to produce h_policy.
 (b) Gate behavior across two regimes. In open corridors (low TTC risk) the
     gate stays near zero so the backbone dominates; in high-risk pockets the
     gate opens and graph reasoning is mixed in.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon
import numpy as np

from _figstyle import (
    PALETTE, apply_style, blank_ax, rbox, text, arrow, labeled_box, save_both,
)

apply_style(font_size=7.0)

fig = plt.figure(figsize=(3.5, 4.0))
gs = fig.add_gridspec(
    2, 1, height_ratios=[1.0, 0.78], hspace=0.32,
    left=0.07, right=0.97, top=0.97, bottom=0.10,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ============================================================
# Panel a — Data flow through the residual risk gate
# ============================================================
blank_ax(ax_a, xlim=(0, 100), ylim=(0, 60))
ax_a.set_aspect("auto")
ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 60)

text(ax_a, 0.5, 57.5, "a", size=9.5, weight="bold", ha="left")
text(ax_a, 6.5, 57.5, "Residual risk-gated fusion of backbone and graph features",
     size=6.4, ha="left", color=PALETTE["ink"])

# Left-side inputs
in_w, in_h = 26, 7
y_lstm = 41
y_graph = 25
labeled_box(ax_a, 2, y_lstm, in_w, in_h, "$h^i_{\\mathrm{lstm}}\\in\\mathbb{R}^{256}$",
            fc=PALETTE["blue_soft"], ec=PALETTE["blue"], lw=0.9, ts=6.4,
            sub="MLP-LSTM backbone")
labeled_box(ax_a, 2, y_graph, in_w, in_h, "$h^i_{\\mathrm{graph}}\\in\\mathbb{R}^{256}$",
            fc=PALETTE["violet_soft"], ec=PALETTE["violet"], lw=0.9, ts=6.4,
            sub="Fusion of $h^i_{\\mathrm{soc}}, h^i_{\\mathrm{obs}}$")

# Middle column — gate + Δh
gate_x = 38; gate_w = 28
y_gate = 40; gate_h = 8.0
y_delta = 24; delta_h = 8.0

rbox(ax_a, gate_x, y_gate, gate_w, gate_h, fc=PALETTE["amber_soft"],
     ec=PALETTE["amber"], lw=0.9, r=0.25)
text(ax_a, gate_x + gate_w / 2, y_gate + gate_h - 1.8,
     "Risk gate", size=6.4, weight="bold", color=PALETTE["amber"])
text(ax_a, gate_x + gate_w / 2, y_gate + gate_h - 4.2,
     "$g^i=\\sigma(W_g[h_{\\mathrm{lstm}};h_{\\mathrm{graph}}]+b_g)$",
     size=5.4, color=PALETTE["ink_soft"])
text(ax_a, gate_x + gate_w / 2, y_gate + 1.2,
     "$b_g{=}{-}1.0$  (closed at init.)",
     size=4.8, color=PALETTE["ink_soft"], style="italic")

rbox(ax_a, gate_x, y_delta, gate_w, delta_h, fc=PALETTE["violet_soft"],
     ec=PALETTE["violet"], lw=0.9, r=0.25)
text(ax_a, gate_x + gate_w / 2, y_delta + delta_h - 1.8,
     "$\\Delta h^i$", size=7.2, weight="bold", color=PALETTE["violet"])
text(ax_a, gate_x + gate_w / 2, y_delta + delta_h - 4.2,
     "$\\mathrm{MLP}_\\Delta(h^i_{\\mathrm{graph}})$",
     size=5.4, color=PALETTE["ink_soft"])
text(ax_a, gate_x + gate_w / 2, y_delta + 1.2,
     "2-layer  +  GELU",
     size=4.8, color=PALETTE["ink_soft"], style="italic")

# Arrows: inputs → middle column
# h_lstm and h_graph both feed the GATE (they are concatenated inside)
arrow(ax_a, 2 + in_w, y_lstm + in_h / 2,
      gate_x, y_gate + gate_h - 2.0,
      color=PALETTE["blue"], lw=0.9, mut=7)
arrow(ax_a, 2 + in_w, y_graph + in_h / 2,
      gate_x, y_gate + 2.0,
      color=PALETTE["violet"], lw=0.9, mut=7)
# h_graph → Δh
arrow(ax_a, 2 + in_w, y_graph + in_h / 2,
      gate_x, y_delta + delta_h / 2,
      color=PALETTE["violet"], lw=0.9, mut=7)

# Right column — ⊙, ⊕, LN, output
odot_x = 76
odot_y = (y_gate + y_delta + delta_h) / 2 + 1.0
oc = Circle((odot_x, odot_y), 2.0, facecolor="white",
            edgecolor=PALETTE["ink"], lw=0.9, zorder=6)
ax_a.add_patch(oc)
text(ax_a, odot_x, odot_y, "$\\odot$", size=10.0, weight="bold", z=7)
text(ax_a, odot_x, odot_y + 3.2, "element-wise", size=4.8,
     color=PALETTE["ink_soft"], style="italic")

plus_x = 88
plus_y = odot_y - 8.5
pc = Circle((plus_x, plus_y), 2.0, facecolor="white",
            edgecolor=PALETTE["ink"], lw=0.9, zorder=6)
ax_a.add_patch(pc)
text(ax_a, plus_x, plus_y, "$\\oplus$", size=11.0, weight="bold", z=7)
text(ax_a, plus_x, plus_y + 3.2, "residual sum", size=4.8,
     color=PALETTE["ink_soft"], style="italic")

# Gate → ⊙ and Δh → ⊙
arrow(ax_a, gate_x + gate_w, y_gate + gate_h / 2, odot_x - 2.0, odot_y + 0.4,
      color=PALETTE["amber"], lw=1.0, mut=7)
arrow(ax_a, gate_x + gate_w, y_delta + delta_h / 2, odot_x - 2.0, odot_y - 0.4,
      color=PALETTE["violet"], lw=1.0, mut=7)
text(ax_a, gate_x + gate_w + 4.8, y_gate + gate_h / 2 - 0.4, "$g^i$",
     size=6.0, color=PALETTE["amber"], weight="bold", ha="left")
text(ax_a, gate_x + gate_w + 4.8, y_delta + delta_h / 2 - 0.4, "$\\Delta h^i$",
     size=6.0, color=PALETTE["violet"], weight="bold", ha="left")

# ⊙ → ⊕
arrow(ax_a, odot_x, odot_y - 2.0, plus_x - 1.6, plus_y + 1.2,
      color=PALETTE["ink_soft"], lw=1.0, mut=7)
text(ax_a, odot_x + 5.0, (odot_y + plus_y) / 2 + 0.6,
     "$g^i\\odot\\Delta h^i$", size=5.4, color=PALETTE["ink"], ha="left")

# Residual skip from h_lstm → ⊕ (curves over the top)
skip_y = 55
arrow(ax_a, 2 + in_w * 0.5, y_lstm + in_h, 2 + in_w * 0.5, skip_y,
      color=PALETTE["blue"], lw=1.0, mut=0, style="-")
ax_a.plot([2 + in_w * 0.5, plus_x], [skip_y, skip_y],
          color=PALETTE["blue"], lw=1.0, linestyle=(0, (2.5, 1.5)), zorder=4)
arrow(ax_a, plus_x, skip_y, plus_x, plus_y + 2.0,
      color=PALETTE["blue"], lw=1.0, mut=7, ls=(0, (2.5, 1.5)))
text(ax_a, 70, skip_y + 1.5, "residual skip ($h^i_{\\mathrm{lstm}}$)",
     size=5.4, color=PALETTE["blue"], weight="bold", ha="center")

# Output: LN + h_policy
out_y = plus_y - 9.5
labeled_box(ax_a, plus_x - 14, out_y, 28, 7.5, "$h^i_{\\mathrm{policy}}$",
            fc=PALETTE["stage_bg"], ec=PALETTE["ink"], lw=0.9, ts=7.2,
            sub="$\\mathrm{LN}(h^i_{\\mathrm{lstm}}+g^i\\odot\\Delta h^i)$")
arrow(ax_a, plus_x, plus_y - 2.0, plus_x, out_y + 7.5,
      color=PALETTE["ink_soft"], lw=1.0, mut=7)

# Side note: gate range
text(ax_a, 50, 1.5,
     "$g^i\\in[0,1]^{256}$  •  graph influence vanishes when $g^i\\!\\to\\!0$",
     size=5.0, color=PALETTE["ink_soft"], ha="center", style="italic")

# ============================================================
# Panel b — Gate behavior across regimes
# ============================================================
ax_b.set_aspect("auto")
# Use real plot axes here (not blank) so x/y ticks are meaningful

# Two regimes: synthetic gate response to a single "risk" scalar
risk = np.linspace(0, 1, 200)
# Mean activation of 256-D gate (illustrative): sigmoid centered around r=0.45
mean_gate = 1.0 / (1.0 + np.exp(-12 * (risk - 0.45)))
# Show distribution band: spread tightens near both extremes
spread = 0.18 * np.exp(-((risk - 0.45) ** 2) / 0.05)
upper = np.clip(mean_gate + spread, 0, 1)
lower = np.clip(mean_gate - spread, 0, 1)

ax_b.fill_between(risk, lower, upper, color=PALETTE["amber_soft"],
                  alpha=0.7, label="gate dist. (256-D)")
ax_b.plot(risk, mean_gate, color=PALETTE["amber"], lw=1.6,
          label="$\\bar g^i$ (mean)")

# Shade regimes
ax_b.axvspan(0, 0.30, color=PALETTE["blue_soft"], alpha=0.45, zorder=0)
ax_b.axvspan(0.65, 1.0, color=PALETTE["red_soft"], alpha=0.45, zorder=0)

# Regime labels
ax_b.text(0.15, 0.94, "open corridor", ha="center", va="center",
          fontsize=6.0, color=PALETTE["blue"], weight="bold",
          transform=ax_b.transData)
ax_b.text(0.15, 0.86, "backbone dominates", ha="center", va="center",
          fontsize=5.2, color=PALETTE["ink_soft"], style="italic",
          transform=ax_b.transData)

ax_b.text(0.83, 0.94, "high-risk pocket", ha="center", va="center",
          fontsize=6.0, color=PALETTE["red"], weight="bold",
          transform=ax_b.transData)
ax_b.text(0.83, 0.86, "graph reasoning mixed in", ha="center", va="center",
          fontsize=5.2, color=PALETTE["ink_soft"], style="italic",
          transform=ax_b.transData)

# Axis cosmetics
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1.0)
ax_b.set_xlabel("normalized interaction risk", fontsize=6.6, labelpad=2)
ax_b.set_ylabel("gate activation  $g^i$", fontsize=6.6, labelpad=2)
ax_b.tick_params(axis="both", labelsize=5.6, length=2.0, pad=1.5)
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.spines["left"].set_linewidth(0.7)
ax_b.spines["bottom"].set_linewidth(0.7)

# Panel label (offset to top-left of axes)
ax_b.text(-0.10, 1.04, "b", transform=ax_b.transAxes,
          fontsize=9.5, fontweight="bold", ha="left", va="bottom",
          color=PALETTE["ink"])
ax_b.text(0.0, 1.04, "Learned gate opens under elevated risk",
          transform=ax_b.transAxes, fontsize=6.4,
          ha="left", va="bottom", color=PALETTE["ink"])

# Legend
ax_b.legend(loc="lower right", fontsize=5.4, frameon=False,
            handlelength=1.4, handletextpad=0.5, labelspacing=0.3)

save_both(fig, "fig4_residual_gate")
print("Fig4 written.")
