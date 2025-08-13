
#!/usr/bin/env python3
# finetune_players.py
# Fine‑tune a ViT chess move‑prediction model per‑player by looping over .pt datasets.
#
# Usage (example):
#   python finetune_players.py --data_dir ./data --base_ckpt ./best_vit_amd.pth --out_dir ./outputs_players
#
# Notes
# - Expects each file in --data_dir to be a single‑player dataset saved as a .pt (list of dicts like your sample).
# - Loads your original TokenViT from a module you provide (see IMPORT MODEL section).
# - Applies *legal‑move masking* and optimizes a masked cross‑entropy objective over from/to squares.
# - Uses modern fine‑tuning practices: AdamW + cosine LR with warmup, mixed precision, grad clipping,
#   optional layer‑wise LR decay (LLRD), optional partial freezing, EMA of weights, early stopping.
#
# Research references (why these choices):
# - Human‑move imitation (Maia / Maia‑2): train to match human moves; use legal move set when scoring. (McIlroy‑Young et al.; Tang et al.) 
# - Legal‑move masking is standard in policy heads for board games (e.g., AlphaZero). 
# - ViT fine‑tuning recipes: AdamW + cosine LR + (optionally) LLRD and stochastic depth / dropout. (DeiT; Steiner et al.)
#
# IMPORTANT
# - Ensure the TokenViT class used here is *the same implementation* as the one used to produce best_vit_amd.pth.
#   Put that class in model.py (or adjust the import below). The checkpoint is expected to have a `model_state_dict`
#   and a `config` with keys like EMBED_DIM, DEPTH, N_HEADS, MAX_TOKENS.
#
# - Each .pt file should contain dictionaries with keys at least:
#     'position': LongTensor [T, 10]
#     'move':     FloatTensor [2, 8, 8]  (one‑hot from/to)
#   Optional (preferred to avoid recomputing):
#     'legal_mask_from': uint8/bool [8,8]
#     'legal_mask_dest': uint8/bool [8,8]
#
# - If legal masks are missing and you *really* need them computed from tokens, you can extend `compute_legal_masks_from_tokens`
#   using python‑chess, mirroring your training notebook. For speed and reproducibility, it's better to store masks in the .pt.

import os
import math
import json
import glob
import time
import copy
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# -------------------------------
# IMPORT MODEL (TokenViT)
# -------------------------------
# Provide your original model class in model.py (same as you used for best_vit_amd.pth).
# It must instantiate from a config dict: TokenViT(embed_dim=..., depth=..., n_heads=..., max_tokens=...)
try:
    from model import TokenViT  # You may rename this import to match your codebase
except Exception as e:
    TokenViT = None
    print("[WARN] Could not import TokenViT from model.py. Please ensure your model class is available. Error:", e)

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

def onehot_to_index(moves_8x8: torch.Tensor) -> torch.Tensor:
    # moves_8x8: [B, 8, 8] one‑hot -> [B] index in [0..63]
    B = moves_8x8.shape[0]
    flat = moves_8x8.reshape(B, 64)
    idx = flat.argmax(dim=1)
    return idx

def mask_logits_with_legal(logits_64: torch.Tensor, legal_mask_8x8: torch.Tensor) -> torch.Tensor:
    # logits_64: [B, 64]; legal_mask_8x8: [B, 8, 8] (bool/byte)
    legal_flat = legal_mask_8x8.reshape(logits_64.shape[0], 64).bool()
    # set illegal to -inf so CE ignores them
    neg_inf = torch.finfo(logits_64.dtype).min
    masked = logits_64.masked_fill(~legal_flat, neg_inf)
    return masked

# -------------------------------
# Dataset
# -------------------------------
class SinglePTDataset(Dataset):
    """
    Loads a single player's dataset from a .pt file containing a list of dicts.
    Each dict matches the sample structure you provided.
    """
    def __init__(self, pt_path: str):
        super().__init__()
        self.path = pt_path
        print(f"[INFO] Loading dataset: {pt_path}")
        self.records: List[Dict[str, Any]] = torch.load(pt_path, map_location="cpu")
        assert isinstance(self.records, (list, tuple)), "Expected the .pt file to contain a list of examples"
        # Light validation
        r0 = self.records[0]
        for k in ["position", "move"]:
            assert k in r0, f"Missing key '{k}' in first record"
        self.has_masks = ("legal_mask_from" in r0) and ("legal_mask_dest" in r0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pos = torch.as_tensor(rec["position"], dtype=torch.long)      # [T, 10]
        move = torch.as_tensor(rec["move"], dtype=torch.float32)      # [2, 8, 8]
        if self.has_masks:
            mask_from = torch.as_tensor(rec["legal_mask_from"]).bool()  # [8,8]
            mask_dest = torch.as_tensor(rec["legal_mask_dest"]).bool()  # [8,8]
            return pos, move, mask_from, mask_dest
        else:
            return pos, move

def collate_batch(batch, max_tokens: Optional[int] = None):
    """
    Pads variable‑length token sequences to the longest in batch (or max_tokens if provided).
    Returns:
      x_tokens: Long [B, T, 10], pad_mask: Bool [B, T]
      y_from: Float [B, 8, 8], y_to: Float [B, 8, 8]
      legal_from (optional): Bool [B, 8, 8], legal_to (optional): Bool [B, 8, 8]
    """
    with_masks = (len(batch[0]) == 4)
    if with_masks:
        positions, moves, masks_from, masks_to = zip(*batch)
    else:
        positions, moves = zip(*batch)

    B = len(positions)
    lengths = [p.shape[0] for p in positions]
    T = max(lengths) if max_tokens is None else max(max(lengths), max_tokens)
    T = int(T)

    x = torch.zeros((B, T, 10), dtype=torch.long)
    pad = torch.zeros((B, T), dtype=torch.bool)
    for i, p in enumerate(positions):
        n = min(p.shape[0], T)
        x[i, :n] = p[:n]
        pad[i, :n] = True

    moves = torch.stack(moves, dim=0)  # [B, 2, 8, 8]
    y_from = moves[:, 0]
    y_to   = moves[:, 1]

    if with_masks:
        legal_from = torch.stack(masks_from, dim=0).bool()
        legal_to   = torch.stack(masks_to, dim=0).bool()
        return x, pad, y_from, y_to, legal_from, legal_to
    else:
        return x, pad, y_from, y_to

# -------------------------------
# Optimizer param‑groups with optional LLRD and selective freezing
# -------------------------------
def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                       llrd_gamma: float = 0.8, freeze_prefixes: Optional[List[str]] = None):
    """
    Splits parameters into groups with (optionally) decreasing LR for lower layers.
    Attempts to detect transformer blocks named like 'blocks.N' or 'transformer.blocks.N'.
    If grouping fails, falls back to a single group.
    """
    no_wd = []
    groups_by_block: Dict[str, List[nn.Parameter]] = {}
    head_params = []
    other_params = []

    # Identify blocks
    block_name_regexes = [
        r'^blocks\.(\d+)\.',
        r'^transformer\.blocks\.(\d+)\.',
        r'^encoder\.layers\.(\d+)\.',
    ]

    def find_block_idx(name: str) -> Optional[int]:
        import re
        for rgx in block_name_regexes:
            m = re.match(rgx, name)
            if m:
                return int(m.group(1))
        return None

    # Freezing
    freeze_prefixes = freeze_prefixes or []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in freeze_prefixes):
            p.requires_grad = False
            continue
        # weight decay exemptions
        if name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_wd.append(p)
            continue
        # head or block grouping
        if name.startswith("head") or "head" in name:
            head_params.append(p)
        else:
            blk = find_block_idx(name)
            if blk is None:
                other_params.append(p)
            else:
                groups_by_block.setdefault(blk, []).append(p)

    # Sort blocks from input -> output
    block_ids = sorted(groups_by_block.keys())
    param_groups = []

    # earlier blocks get smaller lr (layer‑wise decay)
    for i, blk in enumerate(block_ids):
        scale = llrd_gamma ** (len(block_ids) - 1 - i)  # last block scale=1.0
        lr = base_lr * scale
        param_groups.append({"params": groups_by_block[blk], "lr": lr, "weight_decay": weight_decay})

    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * (llrd_gamma ** len(block_ids)), "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})
    if no_wd:
        param_groups.append({"params": no_wd, "lr": base_lr, "weight_decay": 0.0})

    if not param_groups:
        # fallback
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": base_lr, "weight_decay": weight_decay}]

    return param_groups

# -------------------------------
# EMA of weights
# -------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))

# -------------------------------
# Training / Evaluation
# -------------------------------
def forward_model(model: nn.Module, x_tokens: torch.Tensor, pad_mask: torch.Tensor):
    """
    Calls your TokenViT. Must return logits_from [B,64], logits_to [B,64].
    We support a few common return shapes to be robust.
    """
    out = model(x_tokens, pad_mask) if pad_mask is not None else model(x_tokens)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        logits_from, logits_to = out
    elif isinstance(out, dict) and "logits_from" in out and "logits_to" in out:
        logits_from, logits_to = out["logits_from"], out["logits_to"]
    else:
        # fallback: expect [B, 2, 64]
        if out.dim() == 3 and out.shape[1] == 2 and out.shape[2] == 64:
            logits_from, logits_to = out[:,0], out[:,1]
        else:
            raise RuntimeError("Model forward must return (logits_from[B,64], logits_to[B,64]) or a dict with those tensors.")
    return logits_from, logits_to

def train_one_epoch(model, ema, loader, opt, scaler, device, label_smoothing: float = 0.0,
                    grad_clip: float = 1.0, warmup_steps: int = 0, step_idx_start: int = 0):
    model.train()
    total_loss = 0.0
    total_batches = 0
    ce_kwargs = {"label_smoothing": label_smoothing} if label_smoothing > 0 else {}

    step_idx = step_idx_start
    for batch in loader:
        step_idx += 1
        if len(batch) == 6:
            x, pad, y_from, y_to, legal_from, legal_to = batch
        else:
            x, pad, y_from, y_to = batch
            raise RuntimeError("This script expects legal masks in the dataset to compute masked CE. Add legal_mask_* to your .pt files.")

        x, pad, y_from, y_to = to_device((x, pad, y_from, y_to), device)
        legal_from, legal_to = to_device((legal_from, legal_to), device)

        tgt_from = onehot_to_index(y_from)
        tgt_to   = onehot_to_index(y_to)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            lf, lt = forward_model(model, x, pad)  # [B,64] each
            lf = mask_logits_with_legal(lf, legal_from)
            lt = mask_logits_with_legal(lt, legal_to)

            loss_from = F.cross_entropy(lf, tgt_from, **ce_kwargs)
            loss_to   = F.cross_entropy(lt, tgt_to, **ce_kwargs)
            loss = loss_from + loss_to

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # simple linear warmup on lr
        if warmup_steps and step_idx <= warmup_steps:
            for pg in opt.param_groups:
                pg['lr'] = pg['lr'] * step_idx / warmup_steps

        scaler.step(opt)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1), step_idx

@torch.no_grad()
def evaluate(model, loader, device, use_ema: Optional[ModelEMA] = None):
    model_to_eval = use_ema.ema if use_ema is not None else model
    model_to_eval.eval()

    total_loss = 0.0
    total_batches = 0
    top1_match = 0
    denom = 0

    for batch in loader:
        if len(batch) == 6:
            x, pad, y_from, y_to, legal_from, legal_to = batch
        else:
            x, pad, y_from, y_to = batch
            raise RuntimeError("This script expects legal masks in the dataset to compute masked CE.")

        x, pad, y_from, y_to = to_device((x, pad, y_from, y_to), device)
        legal_from, legal_to = to_device((legal_from, legal_to), device)

        tgt_from = onehot_to_index(y_from)
        tgt_to   = onehot_to_index(y_to)

        lf, lt = forward_model(model_to_eval, x, pad)
        lf = mask_logits_with_legal(lf, legal_from)
        lt = mask_logits_with_legal(lt, legal_to)

        loss = F.cross_entropy(lf, tgt_from) + F.cross_entropy(lt, tgt_to)
        total_loss += loss.item()
        total_batches += 1

        # Move‑matching accuracy: both from & to predicted correctly (top‑1 under legal mask)
        pred_from = lf.argmax(dim=1)
        pred_to   = lt.argmax(dim=1)
        top1_match += ((pred_from == tgt_from) & (pred_to == tgt_to)).sum().item()
        denom += x.size(0)

    return {
        "loss": total_loss / max(total_batches, 1),
        "move_match_top1": top1_match / max(denom, 1)
    }

# -------------------------------
# Main fine‑tuning loop
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine‑tune per‑player ViT models from a base checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing per‑player .pt files")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Path to best_vit_amd.pth")
    parser.add_argument("--out_dir", type=str, default="./outputs_players", help="Where to save fine‑tuned models")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs per player (early stopping enabled)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (adjust per VRAM)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--holdout_last_n", type=int, default=20000, help="Number of final examples to hold out entirely (no train/val usage) for later testing")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate for head/last layers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (AdamW)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for CE on from/to")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (global norm)")
    parser.add_argument("--llrd_gamma", type=float, default=0.8, help="Layer‑wise LR decay factor (lower->smaller LR). Set 1.0 to disable.")
    parser.add_argument("--freeze_prefix", type=str, default="emb,patch_embed,cls_token,pos_embed,stem", 
                        help="Comma‑separated module prefixes to freeze (best to keep embeddings frozen initially)")
    parser.add_argument("--unfreeze_last_n", type=int, default=4, help="Number of final transformer blocks to keep trainable (if detected)")
    parser.add_argument("--ema", action="store_true", help="Enable EMA of weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max tokens per position (else uses longest in batch)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base checkpoint to get config & weights
    base = torch.load(args.base_ckpt, map_location="cpu")
    if isinstance(base, dict) and "model_state_dict" in base:
        base_state = base["model_state_dict"]
        base_cfg = base.get("config", {})
    else:
        raise RuntimeError("Checkpoint must be a dict with keys: model_state_dict and (optionally) config.")

    # Instantiate model
    if TokenViT is None:
        raise RuntimeError("TokenViT class is not importable. Provide the same TokenViT implementation used for the base model.")
    # Try common constructor signatures
    try:
        model = TokenViT(**{
            "embed_dim": base_cfg.get("EMBED_DIM"),
            "depth": base_cfg.get("DEPTH"),
            "n_heads": base_cfg.get("N_HEADS"),
            "max_tokens": base_cfg.get("MAX_TOKENS")
        })
    except TypeError:
        # Fallback to positional or different kw names; user may need to adjust
        model = TokenViT()

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"[INFO] Loaded base weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print("  Missing (first 10):", missing[:10])
    if unexpected:
        print("  Unexpected (first 10):", unexpected[:10])

    model.to(device)

    # Discover transformer blocks to optionally freeze
    block_param_names = [n for n, _ in model.named_parameters() if ".blocks." in n or "blocks." in n or ".layers." in n]
    block_ids = sorted(set(int(n.split(".")[n.split(".").index("blocks")+1]) 
                           for n in block_param_names if "blocks" in n and n.split(".")[n.split(".").index("blocks")+1].isdigit())) if block_param_names else []
    max_block = max(block_ids) if block_ids else None
    freeze_prefixes = [p.strip() for p in args.freeze_prefix.split(",") if p.strip()]

    # Loop over player datasets
    pt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    assert pt_files, f"No .pt files found in {args.data_dir}"

    for pt_path in pt_files:
        player_tag = Path(pt_path).stem
        run_dir = out_dir / player_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        # Fresh copy of the base model per player
        model.load_state_dict(base_state, strict=False)
        model.to(device)

        # Freeze prefixes & (optionally) early blocks
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in freeze_prefixes):
                p.requires_grad_(False)
        if max_block is not None and args.unfreeze_last_n > 0:
            keep_from = max_block - args.unfreeze_last_n + 1
            for name, p in model.named_parameters():
                # if it's a block param and block idx < keep_from: freeze
                if ".blocks." in name or "blocks." in name:
                    try:
                        parts = name.split(".")
                        bidx = int(parts[parts.index("blocks")+1])
                        if bidx < keep_from:
                            p.requires_grad_(False)
                    except Exception:
                        pass

        # Data splitting
        #   Reserve the LAST holdout_last_n examples as a test set (NOT used for train or validation)
        #   Perform train/val split only on the remaining (chronologically earlier) examples.
        ds_full = SinglePTDataset(pt_path)
        N = len(ds_full)
        holdout_n = args.holdout_last_n
        if N > holdout_n and holdout_n > 0:
            test_start = N - holdout_n
            test_indices = list(range(test_start, N))
            usable_indices = list(range(0, test_start))
            usable_ds = torch.utils.data.Subset(ds_full, usable_indices)
            test_ds = torch.utils.data.Subset(ds_full, test_indices)  # kept for future evaluation, unused here
            N_usable = len(usable_indices)
            n_val = max(1, int(math.ceil(N_usable * args.val_split)))
            n_train = max(1, N_usable - n_val)
            train_ds, val_ds = random_split(usable_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
            print(f"[SPLIT] Holdout last {holdout_n} examples (train={n_train}, val={n_val}, test/holdout={holdout_n}).")
            # Persist simple metadata for later testing convenience
            split_meta = {
                "total": N,
                "train": n_train,
                "val": n_val,
                "holdout_test": holdout_n,
                "holdout_range": [test_start, N-1]
            }
            with open(run_dir / "split_meta.json", "w") as f:
                json.dump(split_meta, f)
        else:
            # Not enough examples to carve out full holdout; revert to original random fraction split
            ds = ds_full
            n_val = max(1, int(math.ceil(N * args.val_split)))
            n_train = max(1, N - n_val)
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
            print(f"[SPLIT] Dataset smaller than holdout target ({holdout_n}); using fraction split (train={n_train}, val={n_val}).")

        collate = (lambda b: collate_batch(b, args.max_tokens))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, collate_fn=collate, drop_last=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                pin_memory=True, collate_fn=collate)

        # Optimizer with LLRD param groups
        param_groups = build_param_groups(model, base_lr=args.lr, weight_decay=args.weight_decay, llrd_gamma=args.llrd_gamma,
                                          freeze_prefixes=freeze_prefixes)
        opt = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.95))

        # Scheduler: cosine to a small fraction of lr
        # We implement simple cosine per‑step over epochs*steps; warmup handled in train loop.
        total_steps = args.epochs * max(1, len(train_loader))
        def lr_lambda(step):
            if total_steps <= 1: return 1.0
            # Cosine from 1.0 to 0.1
            cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            return 0.1 + 0.9 * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        ema = ModelEMA(model, decay=0.999) if args.ema else None

        # Train
        best_val = float("inf")
        best_acc = 0.0
        patience = 3
        bad_epochs = 0
        global_step = 0

        metrics_path = run_dir / "metrics.jsonl"
        ckpt_best_path = run_dir / f"{player_tag}_best.pth"
        ckpt_last_path = run_dir / f"{player_tag}_last.pth"

        print(f"\n===== Fine‑tuning on {player_tag}  (N={N}) =====")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, global_step = train_one_epoch(model, ema, train_loader, opt, scaler, device,
                                                      label_smoothing=args.label_smoothing,
                                                      grad_clip=args.grad_clip, warmup_steps=args.warmup_steps,
                                                      step_idx_start=global_step)
            scheduler.step()
            val_stats = evaluate(model, val_loader, device, use_ema=ema)

            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_stats["loss"],
                "val_move_match_top1": val_stats["move_match_top1"],
                "lr_head": opt.param_groups[0]["lr"]
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(log) + "\n")

            print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_stats['loss']:.4f} | top1 {val_stats['move_match_top1']:.4f} | time {time.time()-t0:.1f}s")

            # Save last
            torch.save({
                "model_state_dict": model.state_dict(),
                "ema_state_dict": (ema.ema.state_dict() if ema is not None else None),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "config": base_cfg,
                "player_tag": player_tag
            }, ckpt_last_path)

            # Track best by val loss; tie‑break by accuracy
            improved = (val_stats["loss"] < best_val - 1e-5) or \
                       (abs(val_stats["loss"] - best_val) < 1e-5 and val_stats["move_match_top1"] > best_acc + 1e-4)
            if improved:
                best_val = val_stats["loss"]
                best_acc = val_stats["move_match_top1"]
                torch.save({
                    "model_state_dict": (ema.ema.state_dict() if ema is not None else model.state_dict()),
                    "config": base_cfg,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "best_val_top1": best_acc,
                    "player_tag": player_tag
                }, ckpt_best_path)
                bad_epochs = 0
                print(f"  ✔ Saved BEST to {ckpt_best_path}")
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    print("\nAll players done.")

if __name__ == "__main__":
    main()
