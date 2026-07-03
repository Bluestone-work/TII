"""Shared style + drawing helpers for TII paper figures."""
from __future__ import annotations
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patches import Circle, Wedge, Polygon, PathPatch
from matplotlib.path import Path
import numpy as np

PALETTE = {
    "ink":        "#1F2A36",
    "ink_soft":   "#3A4756",
    "rule":       "#5C6B7A",
    "muted":      "#94A3B0",
    "panel_bg":   "#FFFFFF",
    "stage_bg":   "#F3F5F8",
    "stage_edge": "#C9D1DB",
    "blue":       "#2F6FB5",
    "blue_soft":  "#D6E3F3",
    "teal":       "#2F8F8A",
    "teal_soft":  "#D3ECEA",
    "amber":      "#C57B19",
    "amber_soft": "#F3E1C3",
    "red":        "#B64342",
    "red_soft":   "#F2D5D2",
    "green":      "#2E8B4B",
    "green_soft": "#D6ECDB",
    "violet":     "#7E5BB0",
    "violet_soft":"#E3D8EF",
}

def apply_style(font_size: float = 7.5) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.titlesize": font_size + 0.5,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def blank_ax(ax, xlim=(0, 100), ylim=(0, 60)) -> None:
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def rbox(ax, x, y, w, h, *, fc="#FFFFFF", ec=None, lw=0.9, r=0.08, z=2):
    ec = ec or PALETTE["ink"]
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(patch)
    return patch

def stage_panel(ax, x, y, w, h, title, accent, *, title_fs=7.2, bar_h=2.2):
    rbox(ax, x, y, w, h, fc=PALETTE["stage_bg"], ec=PALETTE["stage_edge"], lw=0.8, r=0.5, z=1)
    rbox(ax, x, y + h - bar_h, w, bar_h, fc=accent, ec=accent, lw=0, r=0.5, z=1.5)
    ax.text(x + w / 2, y + h - bar_h / 2, title, ha="center", va="center",
            fontsize=title_fs, color="white", fontweight="bold", zorder=3)

def text(ax, x, y, s, *, size=7.2, ha="center", va="center", color=None,
         weight="normal", style="normal", z=4):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or PALETTE["ink"],
            fontweight=weight, fontstyle=style, zorder=z)

def arrow(ax, x0, y0, x1, y1, *, color=None, lw=1.1, style="->", mut=8, z=3, ls="-"):
    color = color or PALETTE["rule"]
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=mut,
                        linewidth=lw, color=color, zorder=z, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a

def labeled_box(ax, x, y, w, h, title, *, fc="#FFFFFF", ec=None, lw=0.9, ts=7.2, sub=None):
    rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if sub is None:
        text(ax, x + w / 2, y + h / 2, title, size=ts)
    else:
        text(ax, x + w / 2, y + h * 0.62, title, size=ts, weight="bold")
        text(ax, x + w / 2, y + h * 0.30, sub, size=ts - 0.6, color=PALETTE["ink_soft"])

def save_both(fig, stem: str, out_dir: str = "figures") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    return pdf, png
