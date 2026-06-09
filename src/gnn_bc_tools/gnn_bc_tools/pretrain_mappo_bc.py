#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gnn_bc_tools.path_utils import ensure_runtime_modules, inject_workspace_paths


@dataclass
class DatasetChunks:
    obs: np.ndarray
    actions: np.ndarray
    seq_lens: np.ndarray
    chunks: List[Tuple[int, int]]  # (start_index_in_flat, chunk_len)
    obs_dim: int
    action_dim: int
    num_agents: int
    base_obs_dim: int
    neighbor_dim: int
    reset_flag_dim: int


def _scalar(npz, key: str, default: int = 0) -> int:
    if key not in npz:
        return int(default)
    val = npz[key]
    if np.isscalar(val):
        return int(val)
    return int(np.asarray(val).reshape(-1)[0])


def load_dataset(dataset_path: Path, chunk_len: int) -> DatasetChunks:
    data = np.load(dataset_path, allow_pickle=False)

    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    seq_lens = np.asarray(data["seq_lens"], dtype=np.int32)

    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"obs/actions 样本数不一致: {obs.shape[0]} vs {actions.shape[0]}")
    if seq_lens.sum() != obs.shape[0]:
        raise ValueError(f"seq_lens 累计长度({seq_lens.sum()}) != 样本数({obs.shape[0]})")

    chunks: List[Tuple[int, int]] = []
    offset = 0
    for seq_len in seq_lens.tolist():
        if seq_len <= 0:
            continue
        for st in range(0, seq_len, chunk_len):
            ln = min(chunk_len, seq_len - st)
            chunks.append((offset + st, ln))
        offset += seq_len

    return DatasetChunks(
        obs=obs,
        actions=actions,
        seq_lens=seq_lens,
        chunks=chunks,
        obs_dim=int(obs.shape[1]),
        action_dim=int(actions.shape[1]),
        num_agents=_scalar(data, "num_agents", 2),
        base_obs_dim=_scalar(data, "base_obs_dim", 155),
        neighbor_dim=_scalar(data, "neighbor_dim", 0),
        reset_flag_dim=_scalar(data, "reset_flag_dim", 1),
    )


def _build_batch(
    dataset: DatasetChunks,
    chunk_ids: List[int],
    reset_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = [dataset.chunks[cid][1] for cid in chunk_ids]
    bsz = len(chunk_ids)
    tmax = max(lengths)

    obs_b = np.zeros((bsz, tmax, dataset.obs_dim), dtype=np.float32)
    act_b = np.zeros((bsz, tmax, dataset.action_dim), dtype=np.float32)
    mask = np.zeros((bsz, tmax), dtype=np.float32)

    for bi, cid in enumerate(chunk_ids):
        st, ln = dataset.chunks[cid]
        obs_slice = dataset.obs[st:st + ln]
        act_slice = dataset.actions[st:st + ln]
        obs_b[bi, :ln] = obs_slice
        act_b[bi, :ln] = act_slice
        mask[bi, :ln] = 1.0
        if ln < tmax:
            obs_b[bi, ln:, reset_index] = 1.0  # pad steps reset hidden

    return obs_b, act_b, mask


def pretrain_bc(args) -> Path:
    inject_workspace_paths()

    from gymnasium import spaces
    from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_path, chunk_len=int(args.chunk_len))
    if ds.action_dim != 2:
        raise ValueError(f"当前只支持2维连续动作，数据集 action_dim={ds.action_dim}")

    max_neighbors = max(0, ds.neighbor_dim // 5)
    use_neighbor_obs = ds.neighbor_dim > 0
    reset_index = ds.base_obs_dim + ds.neighbor_dim

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ds.obs_dim,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    model_config = {
        "custom_model_config": {
            "num_agents": ds.num_agents,
            "max_neighbors": max_neighbors,
            "neighbor_feature_dim": 5,
            "use_neighbor_obs": use_neighbor_obs,
            "hidden_dim": int(args.hidden_dim),
            "lstm_hidden_dim": int(args.lstm_hidden_dim or args.hidden_dim),
        }
    }

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = MAPPOMLPModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=2,
        model_config=model_config,
        name="bc_pretrain_model",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    indices = np.arange(len(ds.chunks), dtype=np.int32)
    np.random.shuffle(indices)
    val_count = int(len(indices) * float(args.val_ratio))
    val_count = max(1, val_count) if len(indices) >= 10 else 0
    val_ids = indices[:val_count].tolist() if val_count > 0 else []
    train_ids = indices[val_count:].tolist() if val_count > 0 else indices.tolist()

    print("=" * 80)
    print("MAPPO-MLP BC 预训练")
    print(f"dataset:       {dataset_path}")
    print(f"obs_dim:       {ds.obs_dim}")
    print(f"samples:       {ds.obs.shape[0]}")
    print(f"sequences:     {len(ds.seq_lens)}")
    print(f"chunks:        {len(ds.chunks)}  (chunk_len={args.chunk_len})")
    print(f"train/val:     {len(train_ids)}/{len(val_ids)}")
    print(f"device:        {device}")
    print("=" * 80)

    best_val = math.inf
    best_state = None
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        np.random.shuffle(train_ids)
        model.train()
        train_loss_sum = 0.0
        train_weight = 0.0

        for bi in range(0, len(train_ids), int(args.batch_sequences)):
            batch_chunk_ids = train_ids[bi:bi + int(args.batch_sequences)]
            if not batch_chunk_ids:
                continue

            obs_b, act_b, mask_b = _build_batch(ds, batch_chunk_ids, reset_index)
            bsz, tmax, _ = obs_b.shape

            obs_t = torch.from_numpy(obs_b).to(device)
            act_t = torch.from_numpy(act_b).to(device)
            mask_t = torch.from_numpy(mask_b).to(device)

            flat_obs = obs_t.reshape(bsz * tmax, ds.obs_dim)
            seq_lens = torch.full((bsz,), tmax, dtype=torch.int32, device=device)
            h0 = torch.zeros((bsz, model.lstm_hidden_dim), dtype=torch.float32, device=device)
            c0 = torch.zeros((bsz, model.lstm_hidden_dim), dtype=torch.float32, device=device)

            pred, _ = model({"obs_flat": flat_obs}, [h0, c0], seq_lens)
            pred = pred.view(bsz, tmax, ds.action_dim)

            per_step = F.smooth_l1_loss(pred, act_t, reduction="none").mean(dim=-1)
            loss = (per_step * mask_t).sum() / torch.clamp(mask_t.sum(), min=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            w = float(mask_b.sum())
            train_loss_sum += float(loss.item()) * w
            train_weight += w

        train_loss = train_loss_sum / max(train_weight, 1.0)

        val_loss = float("nan")
        if val_ids:
            model.eval()
            val_loss_sum = 0.0
            val_weight = 0.0
            with torch.no_grad():
                for bi in range(0, len(val_ids), int(args.batch_sequences)):
                    batch_chunk_ids = val_ids[bi:bi + int(args.batch_sequences)]
                    obs_b, act_b, mask_b = _build_batch(ds, batch_chunk_ids, reset_index)
                    bsz, tmax, _ = obs_b.shape

                    obs_t = torch.from_numpy(obs_b).to(device)
                    act_t = torch.from_numpy(act_b).to(device)
                    mask_t = torch.from_numpy(mask_b).to(device)

                    flat_obs = obs_t.reshape(bsz * tmax, ds.obs_dim)
                    seq_lens = torch.full((bsz,), tmax, dtype=torch.int32, device=device)
                    h0 = torch.zeros((bsz, model.lstm_hidden_dim), dtype=torch.float32, device=device)
                    c0 = torch.zeros((bsz, model.lstm_hidden_dim), dtype=torch.float32, device=device)

                    pred, _ = model({"obs_flat": flat_obs}, [h0, c0], seq_lens)
                    pred = pred.view(bsz, tmax, ds.action_dim)
                    per_step = F.smooth_l1_loss(pred, act_t, reduction="none").mean(dim=-1)
                    loss = (per_step * mask_t).sum() / torch.clamp(mask_t.sum(), min=1.0)

                    w = float(mask_b.sum())
                    val_loss_sum += float(loss.item()) * w
                    val_weight += w
            val_loss = val_loss_sum / max(val_weight, 1.0)

        score = val_loss if val_ids else train_loss
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[bc] epoch={epoch:03d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mappo_mlp_bc_init_{stamp}.pt"
    log_path = out_dir / f"mappo_mlp_bc_init_{stamp}.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_custom_config": model_config["custom_model_config"],
            "dataset_path": str(dataset_path),
            "history": history,
            "best_score": float(best_val),
            "obs_dim": int(ds.obs_dim),
            "action_dim": int(ds.action_dim),
            "num_agents": int(ds.num_agents),
            "base_obs_dim": int(ds.base_obs_dim),
            "neighbor_dim": int(ds.neighbor_dim),
        },
        out_path,
    )

    log_path.write_text(
        json.dumps(
            {
                "weights_path": str(out_path),
                "dataset_path": str(dataset_path),
                "best_score": float(best_val),
                "epochs": int(args.epochs),
                "history": history,
                "args": vars(args),
                "created_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("BC 预训练完成")
    print(f"weights:      {out_path}")
    print(f"training log: {log_path}")
    print(f"best score:   {best_val:.6f}")
    print("=" * 80)

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain MAPPO-MLP actor with BC dataset.")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="~/work/multi-robot-exploration-rl/bc_models")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_sequences", type=int, default=32)
    p.add_argument("--chunk_len", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_ratio", type=float, default=0.05)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lstm_hidden_dim", type=int, default=256)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main() -> None:
    ensure_runtime_modules(
        required_modules=["numpy", "torch", "gymnasium"],
        runner_module="gnn_bc_tools.pretrain_mappo_bc",
    )

    parser = build_arg_parser()
    args = parser.parse_args()
    pretrain_bc(args)


if __name__ == "__main__":
    main()
