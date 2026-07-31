#!/usr/bin/env python3
"""Plot GNN (GAT) ego->token attention heatmaps captured during evaluation.

Reads the .npz written by test_gnn_mappo.py --capture_attention and renders:
  1. time x token attention (how ego attends to each token over the episode)
  2. episode-mean ego->token attention (per branch: social/neighbor + obstacle)
  3. per-head comparison of layer-1 multi-head attention

Node 0 in every graph is the ego robot; nodes 1..K are social/obstacle tokens.
We plot the ego row's attention to tokens 1..K (self-attention column 0 dropped).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    attn = json.loads(str(data["attention_records"])) if "attention_records" in data else []
    meta = {
        "map_number": int(data["map_number"]) if "map_number" in data else -1,
        "num_agents": int(data["num_agents"]) if "num_agents" in data else 0,
        "scenario": str(data["scenario"]) if "scenario" in data else "",
    }
    return attn, meta


def _ego_to_tokens(row) -> np.ndarray:
    """Drop the ego self-attention (col 0), return attention to tokens 1..K."""
    arr = np.asarray(row, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    return arr[1:]


def group_by(attn, ep, agent, branch):
    """Return records for one (episode, agent, branch), sorted by step."""
    rows = [r for r in attn if r["episode"] == ep and r["agent"] == agent and r["branch"] == branch]
    rows.sort(key=lambda r: r["step"])
    return rows


def plot_time_token(rows, out_path: Path, *, title: str, token_kind: str) -> None:
    seq = [_ego_to_tokens(r["layer2"]) for r in rows if r.get("layer2") is not None]
    seq = [s for s in seq if s.size > 0]
    if not seq:
        return
    k = max(s.shape[0] for s in seq)
    mat = np.zeros((len(seq), k), dtype=np.float32)
    for i, s in enumerate(seq):
        mat[i, : s.shape[0]] = s
    steps = [r["step"] for r in rows if r.get("layer2") is not None][: len(seq)]

    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * k + 2.5), 5.0), dpi=200)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=(0.5, k + 0.5, steps[0] if steps else 0, steps[-1] if steps else len(seq)))
    ax.set_xlabel(f"{token_kind} token index")
    ax.set_ylabel("episode step")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(1, k + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ego attention weight")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mean_bar(rows, out_path: Path, *, title: str, token_kind: str) -> None:
    seq = [_ego_to_tokens(r["layer2"]) for r in rows if r.get("layer2") is not None]
    seq = [s for s in seq if s.size > 0]
    if not seq:
        return
    k = max(s.shape[0] for s in seq)
    mat = np.zeros((len(seq), k), dtype=np.float32)
    for i, s in enumerate(seq):
        mat[i, : s.shape[0]] = s
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * k + 2.0), 3.6), dpi=200)
    ax.bar(range(1, k + 1), mean, yerr=std, capsize=3, color="#4c78a8", edgecolor="black", linewidth=0.7)
    ax.set_xlabel(f"{token_kind} token index")
    ax.set_ylabel("mean ego attention")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(1, k + 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_heads(rows, out_path: Path, *, title: str, token_kind: str) -> None:
    """Layer-1 multi-head: episode-mean ego->token attention per head."""
    head_seqs: list[list[np.ndarray]] = []
    for r in rows:
        heads = r.get("heads") or []
        for hi, h in enumerate(heads):
            while len(head_seqs) <= hi:
                head_seqs.append([])
            tok = _ego_to_tokens(h)
            if tok.size > 0:
                head_seqs[hi].append(tok)
    head_seqs = [hs for hs in head_seqs if hs]
    if not head_seqs:
        return
    k = max(s.shape[0] for hs in head_seqs for s in hs)
    mat = np.zeros((len(head_seqs), k), dtype=np.float32)
    for hi, hs in enumerate(head_seqs):
        acc = np.zeros(k, dtype=np.float32)
        for s in hs:
            acc[: s.shape[0]] += s
        mat[hi] = acc / max(1, len(hs))

    fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * k + 2.5), 0.6 * len(head_seqs) + 2.2), dpi=200)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel(f"{token_kind} token index")
    ax.set_ylabel("attention head")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(k))
    ax.set_xticklabels(range(1, k + 1))
    ax.set_yticks(range(len(head_seqs)))
    ax.set_yticklabels([f"head {i}" for i in range(len(head_seqs))])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean ego attention")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="attention .npz from test_gnn_mappo.py --capture_attention")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: npz's parent)")
    parser.add_argument("--prefix", type=str, default="", help="filename prefix (default: scenario)")
    parser.add_argument("--episode", type=int, default=0, help="which episode to render time-token for")
    parser.add_argument("--agent", type=str, default="agent_0", help="which agent to render")
    args = parser.parse_args()

    attn, meta = load_records(args.npz)
    if not attn:
        print(f"[plot_gat_attention] no attention records in {args.npz}")
        return
    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (meta["scenario"] or f"map{meta['map_number']}")

    branches = sorted({r["branch"] for r in attn})
    made = 0
    for branch in branches:
        rows = group_by(attn, args.episode, args.agent, branch)
        if not rows:
            continue
        kind = rows[0].get("token_kind", branch)
        tag = f"{prefix}_{args.agent}_{branch}"
        plot_time_token(
            rows, out_dir / f"{tag}_attn_time_token.png",
            title=f"{args.agent} {branch} ego->{kind} attention over episode {args.episode}",
            token_kind=kind,
        )
        plot_mean_bar(
            rows, out_dir / f"{tag}_attn_mean.png",
            title=f"{args.agent} {branch} episode-mean ego->{kind} attention",
            token_kind=kind,
        )
        plot_heads(
            rows, out_dir / f"{tag}_attn_heads.png",
            title=f"{args.agent} {branch} layer-1 per-head ego->{kind} attention",
            token_kind=kind,
        )
        made += 1
        print(f"[OK] {branch}: wrote time-token / mean / heads for {args.agent} ep{args.episode}")
    if made == 0:
        print(f"[plot_gat_attention] no records for episode={args.episode} agent={args.agent}")


if __name__ == "__main__":
    main()
