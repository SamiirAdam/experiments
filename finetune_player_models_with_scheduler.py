
#!/usr/bin/env python3
# finetune_player_models.py — with gradual unfreeze schedule
# Adds: --unfreeze_schedule like "0:2,2:4" (epoch:start -> unfreeze last N blocks)
# Rebuilds optimizer when unfreeze changes, transferring optimizer state.

# python finetune_player_models.py \
#   --data_dir ./data \
#   --base_ckpt ./best_vit_amd.pth \
#   --max_tokens 256 \
#   --holdout_last_n 20000 \
#   --freeze_prefix emb,cls,pos \
#   --unfreeze_last_n 2 \
#   --unfreeze_schedule "2:4" \
#   --epochs 8 --ema

import os, math, json, glob, time, copy, argparse, random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.serialization import add_safe_globals, safe_globals

try:
    from model import TokenViT
except Exception as e:
    TokenViT = None
    print("[WARN] Could not import TokenViT from model.py. Please ensure your model class is available. Error:", e)

def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, (list, tuple)): return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict): return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

def onehot_to_index(moves_8x8: torch.Tensor) -> torch.Tensor:
    flat = moves_8x8.reshape(moves_8x8.shape[0], 64)
    return flat.argmax(dim=1)

def mask_logits_with_legal(logits_64: torch.Tensor, legal_mask_8x8: torch.Tensor) -> torch.Tensor:
    legal_flat = legal_mask_8x8.reshape(logits_64.shape[0], 64).bool()
    neg_inf = torch.finfo(logits_64.dtype).min
    return logits_64.masked_fill(~legal_flat, neg_inf)

# ---------- Dataset (safe torch.load for PyTorch 2.6+) ----------
class SinglePTDataset(Dataset):
    def __init__(self, pt_path: str):
        super().__init__()
        self.path = pt_path
        self.records = self._safe_load_pt(pt_path)
        assert isinstance(self.records, (list, tuple)), f"Expected a list of examples in {pt_path}"
        r0 = self.records[0]
        for k in ["position", "move"]:
            assert k in r0, f"Missing key '{k}' in first record"
        self.has_masks = ("legal_mask_from" in r0) and ("legal_mask_dest" in r0)

    def _safe_load_pt(self, path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            try:
                reconstruct = getattr(np.core.multiarray, "_reconstruct")
                add_safe_globals([reconstruct])
                with safe_globals([reconstruct]):
                    return torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e2:
                raise e2

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        pos = torch.as_tensor(rec["position"], dtype=torch.long)
        move = torch.as_tensor(rec["move"], dtype=torch.float32)
        if self.has_masks:
            mf = torch.as_tensor(rec["legal_mask_from"]).bool()
            mt = torch.as_tensor(rec["legal_mask_dest"]).bool()
            return pos, move, mf, mt
        else:
            return pos, move

def collate_batch(batch, max_tokens: Optional[int] = None):
    with_masks = (len(batch[0]) == 4)
    if with_masks: positions, moves, masks_from, masks_to = zip(*batch)
    else: positions, moves = zip(*batch)

    B = len(positions)
    lengths = [p.shape[0] for p in positions]
    T = max(lengths) if max_tokens is None else max(max(lengths), max_tokens)
    T = int(T)

    x = torch.zeros((B, T, 10), dtype=torch.long)
    pad = torch.zeros((B, T), dtype=torch.bool)
    for i, p in enumerate(positions):
        n = min(p.shape[0], T)
        x[i, :n] = p[:n]; pad[i, :n] = True

    moves = torch.stack(moves, dim=0)  # [B,2,8,8]
    y_from, y_to = moves[:,0], moves[:,1]

    if with_masks:
        legal_from = torch.stack(masks_from, dim=0).bool()
        legal_to   = torch.stack(masks_to, dim=0).bool()
        return x, pad, y_from, y_to, legal_from, legal_to
    else:
        return x, pad, y_from, y_to

# ---------- Optimizer groups ----------
def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                       llrd_gamma: float = 0.8, freeze_prefixes: Optional[List[str]] = None):
    no_wd = []
    groups_by_block: Dict[str, List[nn.Parameter]] = {}
    head_params, other_params = [], []
    import re
    def find_block_idx(name: str):
        for rgx in [r'^blocks\.(\d+)\.', r'^transformer\.blocks\.(\d+)\.', r'^enc\.layers\.(\d+)\.']:
            m = re.match(rgx, name)
            if m: return int(m.group(1))
        return None
    freeze_prefixes = freeze_prefixes or []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(name.startswith(pref) for pref in freeze_prefixes):
            p.requires_grad = False; continue
        if name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_wd.append(p); continue
        if name.startswith("head") or ".head" in name:
            head_params.append(p)
        else:
            blk = find_block_idx(name)
            if blk is None: other_params.append(p)
            else: groups_by_block.setdefault(blk, []).append(p)

    block_ids = sorted(groups_by_block.keys())
    param_groups = []
    for i, blk in enumerate(block_ids):
        scale = llrd_gamma ** (len(block_ids) - 1 - i)
        param_groups.append({"params": groups_by_block[blk], "lr": base_lr * scale, "weight_decay": weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * (llrd_gamma ** len(block_ids)), "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})
    if no_wd:
        param_groups.append({"params": no_wd, "lr": base_lr, "weight_decay": 0.0})
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": base_lr, "weight_decay": weight_decay}]
    return param_groups

def rebuild_optimizer(model, old_opt, base_lr, weight_decay, llrd_gamma, freeze_prefixes):
    # Build new optimizer with current requires_grad flags; transfer state where possible
    new_groups = build_param_groups(model, base_lr=base_lr, weight_decay=weight_decay,
                                    llrd_gamma=llrd_gamma, freeze_prefixes=freeze_prefixes)
    new_opt = torch.optim.AdamW(new_groups, eps=1e-8, betas=(0.9, 0.95))
    if old_opt is None:
        return new_opt
    # Transfer state
    old_state = old_opt.state
    param_map = {}
    for group in new_opt.param_groups:
        for p in group['params']:
            if p in old_state:
                new_opt.state[p] = old_state[p]
    return new_opt

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay; msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd: v.copy_(v * d + msd[k] * (1.0 - d))

def forward_model(model: nn.Module, x_tokens: torch.Tensor, pad_mask: torch.Tensor):
    out = model(x_tokens, pad_mask) if pad_mask is not None else model(x_tokens)
    if isinstance(out, (list, tuple)) and len(out) == 2: return out
    if isinstance(out, dict) and "logits_from" in out and "logits_to" in out:
        return out["logits_from"], out["logits_to"]
    if torch.is_tensor(out) and out.dim() == 3 and out.shape[1] == 2 and out.shape[2] == 64:
        return out[:,0], out[:,1]
    raise RuntimeError("Model forward must return (logits_from[B,64], logits_to[B,64])")

# ---------- Remapping helpers ----------
def _alias_keys(base_state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    ms = model.state_dict()
    new = dict(base_state)
    def move_key(src, dst):
        if src in new:
            new[dst] = new[src]; del new[src]
    if any(k.startswith("enc.") for k in ms.keys()):
        for k in list(new.keys()):
            if k.startswith("encoder."):
                move_key(k, "enc." + k.split("encoder.",1)[1])
    elif any(k.startswith("encoder.") for k in ms.keys()):
        for k in list(new.keys()):
            if k.startswith("enc."):
                move_key(k, "encoder." + k.split("enc.",1)[1])
    if "pos" in ms and "pos_embed" in new: move_key("pos_embed", "pos")
    if "pos_embed" in ms and "pos" in new: move_key("pos", "pos_embed")
    if "cls" in ms and "cls_token" in new: move_key("cls_token", "cls")
    if "cls_token" in ms and "cls" in new: move_key("cls", "cls_token")
    return new

def _remap_heads(base_state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    ms = model.state_dict()
    has_dual_in_model = any(k.startswith("head_from.") for k in ms.keys()) and any(k.startswith("head_to.") for k in ms.keys())
    has_single_in_model = ("head.weight" in ms) and ("head.bias" in ms)
    new = dict(base_state)

    if has_dual_in_model and ("head.weight" in new and "head.bias" in new):
        W = new["head.weight"]; b = new["head.bias"]
        out_dim = W.shape[0]
        if out_dim == 128:
            new["head_from.weight"] = W[:64].clone(); new["head_to.weight"] = W[64:128].clone()
            new["head_from.bias"] = b[:64].clone();   new["head_to.bias"]  = b[64:128].clone()
        elif out_dim == 64:
            new["head_from.weight"] = W.clone(); new["head_to.weight"] = W.clone()
            new["head_from.bias"] = b.clone();   new["head_to.bias"]  = b.clone()
        else:
            raise RuntimeError(f"Cannot split head of out_dim={out_dim}; expected 64 or 128")
        del new["head.weight"]; del new["head.bias"]
        print("[INFO] Remapped single 'head' from ckpt -> dual heads (head_from/head_to).")
        return new

    if has_single_in_model and ("head_from.weight" in new and "head_to.weight" in new):
        Wf, bf = new["head_from.weight"], new["head_from.bias"]
        Wt, bt = new["head_to.weight"], new["head_to.bias"]
        new["head.weight"] = torch.cat([Wf, Wt], dim=0)
        new["head.bias"]   = torch.cat([bf, bt], dim=0)
        for k in ["head_from.weight","head_from.bias","head_to.weight","head_to.bias"]:
            del new[k]
        print("[INFO] Remapped dual heads from ckpt -> single 'head'.")
        return new
    return new

def _resize_positional_embed(base_state: Dict[str, torch.Tensor], model: nn.Module, key_model: str = "pos") -> Dict[str, torch.Tensor]:
    new = dict(base_state)
    ms = model.state_dict()
    key_ckpt = key_model if key_model in new else ("pos_embed" if "pos_embed" in new else None)
    key_target = key_model if key_model in ms else ("pos_embed" if "pos_embed" in ms else None)
    if key_ckpt is None or key_target is None: return new
    W_old = new[key_ckpt]; W_tgt = ms[key_target]
    if W_old.shape != W_tgt.shape:
        with torch.no_grad():
            B_old, L_old, C_old = W_old.shape
            B_tgt, L_tgt, C_tgt = W_tgt.shape
            assert B_old == 1 and B_tgt == 1 and C_old == C_tgt, f"pos channels differ: {C_old} vs {C_tgt}"
            old = W_old.permute(0,2,1)  # [1,C,L_old]
            new_interp = F.interpolate(old, size=L_tgt, mode="linear", align_corners=False)  # [1,C,L_tgt]
            new_pos = new_interp.permute(0,2,1).contiguous()
            print(f"[INFO] Resized positional embedding: {L_old} -> {L_tgt}")
            new[key_ckpt] = new_pos
    return new

def load_base_checkpoint(base_ckpt_path: str, model: nn.Module) -> Dict[str, Any]:
    base = torch.load(base_ckpt_path, map_location="cpu")
    assert isinstance(base, dict) and "model_state_dict" in base, "Checkpoint must be a dict with model_state_dict"
    state = base["model_state_dict"]
    state = _alias_keys(state, model)
    state = _remap_heads(state, model)
    state = _resize_positional_embed(state, model, key_model="pos")
    return {"state": state, "config": base.get("config", {})}

def train_one_epoch(model, ema, loader, opt, scaler, device, label_smoothing: float = 0.0,
                    grad_clip: float = 1.0, warmup_steps: int = 0, step_idx_start: int = 0):
    model.train()
    total_loss, total_batches = 0.0, 0
    ce_kwargs = {"label_smoothing": label_smoothing} if label_smoothing > 0 else {}
    step_idx = step_idx_start
    t0 = time.perf_counter()
    ema_dt = 0.0
    for batch in loader:
        step_idx += 1
        if len(batch) == 6:
            x, pad, y_from, y_to, legal_from, legal_to = batch
        else:
            raise RuntimeError("Dataset must include legal_mask_from/dest for masked CE.")
        x, pad, y_from, y_to = to_device((x, pad, y_from, y_to), device)
        legal_from, legal_to = to_device((legal_from, legal_to), device)

        tgt_from = onehot_to_index(y_from)
        tgt_to   = onehot_to_index(y_to)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            lf, lt = forward_model(model, x, pad)
            lf = mask_logits_with_legal(lf, legal_from)
            lt = mask_logits_with_legal(lt, legal_to)
            loss = F.cross_entropy(lf, tgt_from, **ce_kwargs) + F.cross_entropy(lt, tgt_to, **ce_kwargs)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip: 
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if warmup_steps and step_idx <= warmup_steps:
            for pg in opt.param_groups: pg['lr'] = pg['lr'] * step_idx / warmup_steps
        scaler.step(opt); scaler.update()
        if ema is not None: ema.update(model)

        total_loss += loss.item(); total_batches += 1

        # Light ETA print
        dt = time.perf_counter() - t0
        ema_dt = 0.9 * ema_dt + 0.1 * dt
        if total_batches % 50 == 0:
            remaining = (len(loader) - total_batches) * ema_dt
            print(f"~{ema_dt:.3f}s/step | ETA epoch ~{remaining/60:.1f} min", flush=True)
        t0 = time.perf_counter()

    return total_loss / max(total_batches, 1), step_idx

@torch.no_grad()
def evaluate(model, loader, device, use_ema=None):
    m = use_ema.ema if use_ema is not None else model
    m.eval()
    total_loss, total_batches, top1_match, denom = 0.0, 0, 0, 0
    for batch in loader:
        if len(batch) == 6: x, pad, y_from, y_to, legal_from, legal_to = batch
        else: raise RuntimeError("Dataset must include legal masks.")
        x, pad, y_from, y_to = to_device((x, pad, y_from, y_to), device)
        legal_from, legal_to = to_device((legal_from, legal_to), device)
        tgt_from = onehot_to_index(y_from); tgt_to = onehot_to_index(y_to)
        lf, lt = forward_model(m, x, pad)
        lf = mask_logits_with_legal(lf, legal_from); lt = mask_logits_with_legal(lt, legal_to)
        loss = F.cross_entropy(lf, tgt_from) + F.cross_entropy(lt, tgt_to)
        total_loss += loss.item(); total_batches += 1
        pred_from, pred_to = lf.argmax(1), lt.argmax(1)
        top1_match += ((pred_from == tgt_from) & (pred_to == tgt_to)).sum().item()
        denom += x.size(0)
    return {"loss": total_loss / max(total_batches, 1), "move_match_top1": top1_match / max(denom, 1)}

# ---------- Unfreeze utilities ----------
def detect_max_block(model: nn.Module) -> Optional[int]:
    block_param_names = [n for n,_ in model.named_parameters() if ".blocks." in n or ".layers." in n or ".enc.layers." in n]
    block_ids = []
    for n in block_param_names:
        parts = n.split(".")
        if "blocks" in parts:
            i = parts.index("blocks")
            if i+1 < len(parts) and parts[i+1].isdigit(): block_ids.append(int(parts[i+1]))
        if "layers" in parts:
            i = parts.index("layers")
            if i+1 < len(parts) and parts[i+1].isdigit(): block_ids.append(int(parts[i+1]))
    return max(block_ids) if block_ids else None

def apply_unfreeze(model: nn.Module, last_n: int, max_block: Optional[int], freeze_prefixes: List[str]):
    # Always keep prefixes frozen
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in freeze_prefixes):
            p.requires_grad_(False)

    if max_block is None or last_n <= 0:
        return

    keep_from = max_block - last_n + 1
    for name, p in model.named_parameters():
        is_block_param = (".blocks." in name) or (".layers." in name)
        if is_block_param:
            try:
                parts = name.split(".")
                if "blocks" in parts: bidx = int(parts[parts.index("blocks")+1])
                elif "layers" in parts: bidx = int(parts[parts.index("layers")+1])
                else: continue
                if bidx < keep_from:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
            except Exception:
                pass

def parse_unfreeze_schedule(s: str) -> List[Tuple[int,int]]:
    """
    Parse schedule like "0:2,2:4" => [(0,2),(2,4)] meaning:
      from epoch >=0 use last_n=2; from epoch >=2 use last_n=4.
    """
    if not s: return []
    out = []
    for part in s.split(","):
        ep, ln = part.split(":")
        out.append((int(ep), int(ln)))
    out.sort(key=lambda x: x[0])
    return out

def current_last_n(schedule: List[Tuple[int,int]], epoch: int, default_last_n: int) -> int:
    ln = default_last_n
    for ep, val in schedule:
        if epoch >= ep:
            ln = val
    return ln

def main():
    parser = argparse.ArgumentParser(description="Fine‑tune per‑player ViT models from a base checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_players")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--llrd_gamma", type=float, default=0.8)
    parser.add_argument("--freeze_prefix", type=str, default="emb,patch_embed,cls,pos,stem")
    parser.add_argument("--unfreeze_last_n", type=int, default=2, help="Default number of final blocks to train")
    parser.add_argument("--unfreeze_schedule", type=str, default="", help='Epoch→last_n list, e.g., "0:2,2:4"')
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=256, help="Target sequence length for model pos embeddings")
    parser.add_argument("--holdout_last_n", type=int, default=0, help="Leave last N examples untouched per player")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if TokenViT is None:
        raise RuntimeError("TokenViT class is not importable. Provide TokenViT in model.py.")

    base_raw = torch.load(args.base_ckpt, map_location="cpu")
    base_cfg = base_raw.get("config", {})
    embed_dim  = base_cfg.get("EMBED_DIM", 384)
    depth      = base_cfg.get("DEPTH", 8)
    n_heads    = base_cfg.get("N_HEADS", 6)
    vocab_size = base_cfg.get("VOCAB_SIZE", 512)
    use_cls    = base_cfg.get("USE_CLS_TOKEN", True)

    model = TokenViT(embed_dim=embed_dim, depth=depth, n_heads=n_heads,
                     max_tokens=args.max_tokens, vocab_size=vocab_size, use_cls_token=use_cls)
    model.to(device)

    # Load base checkpoint (remaps heads, aliases names, resizes pos)
    def _alias_keys(base_state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        ms = model.state_dict()
        new = dict(base_state)
        def move_key(src, dst):
            if src in new:
                new[dst] = new[src]; del new[src]
        if any(k.startswith("enc.") for k in ms.keys()):
            for k in list(new.keys()):
                if k.startswith("encoder."):
                    move_key(k, "enc." + k.split("encoder.",1)[1])
        elif any(k.startswith("encoder.") for k in ms.keys()):
            for k in list(new.keys()):
                if k.startswith("enc."):
                    move_key(k, "encoder." + k.split("enc.",1)[1])
        if "pos" in ms and "pos_embed" in new: move_key("pos_embed", "pos")
        if "pos_embed" in ms and "pos" in new: move_key("pos", "pos_embed")
        if "cls" in ms and "cls_token" in new: move_key("cls_token", "cls")
        if "cls_token" in ms and "cls" in new: move_key("cls", "cls_token")
        return new

    def _remap_heads(base_state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        ms = model.state_dict()
        has_dual_in_model = any(k.startswith("head_from.") for k in ms.keys()) and any(k.startswith("head_to.") for k in ms.keys())
        has_single_in_model = ("head.weight" in ms) and ("head.bias" in ms)
        new = dict(base_state)
        if has_dual_in_model and ("head.weight" in new and "head.bias" in new):
            W = new["head.weight"]; b = new["head.bias"]
            out_dim = W.shape[0]
            if out_dim == 128:
                new["head_from.weight"] = W[:64].clone(); new["head_to.weight"] = W[64:128].clone()
                new["head_from.bias"] = b[:64].clone();   new["head_to.bias"]  = b[64:128].clone()
            elif out_dim == 64:
                new["head_from.weight"] = W.clone(); new["head_to.weight"] = W.clone()
                new["head_from.bias"] = b.clone();   new["head_to.bias"]  = b.clone()
            else:
                raise RuntimeError(f"Cannot split head of out_dim={out_dim}; expected 64 or 128")
            del new["head.weight"]; del new["head.bias"]
            print("[INFO] Remapped single 'head' from ckpt -> dual heads (head_from/head_to).")
            return new
        if has_single_in_model and ("head_from.weight" in new and "head_to.weight" in new):
            Wf, bf = new["head_from.weight"], new["head_from.bias"]
            Wt, bt = new["head_to.weight"], new["head_to.bias"]
            new["head.weight"] = torch.cat([Wf, Wt], dim=0)
            new["head.bias"]   = torch.cat([bf, bt], dim=0)
            for k in ["head_from.weight","head_from.bias","head_to.weight","head_to.bias"]:
                del new[k]
            print("[INFO] Remapped dual heads from ckpt -> single 'head'.")
            return new
        return new

    def _resize_positional_embed(base_state: Dict[str, torch.Tensor], model: nn.Module, key_model: str = "pos") -> Dict[str, torch.Tensor]:
        new = dict(base_state)
        ms = model.state_dict()
        key_ckpt = key_model if key_model in new else ("pos_embed" if "pos_embed" in new else None)
        key_target = key_model if key_model in ms else ("pos_embed" if "pos_embed" in ms else None)
        if key_ckpt is None or key_target is None: return new
        W_old = new[key_ckpt]; W_tgt = ms[key_target]
        if W_old.shape != W_tgt.shape:
            with torch.no_grad():
                B_old, L_old, C_old = W_old.shape
                B_tgt, L_tgt, C_tgt = W_tgt.shape
                assert B_old == 1 and B_tgt == 1 and C_old == C_tgt, f"pos channels differ: {C_old} vs {C_tgt}"
                old = W_old.permute(0,2,1)  # [1,C,L_old]
                new_interp = F.interpolate(old, size=L_tgt, mode="linear", align_corners=False)  # [1,C,L_tgt]
                new_pos = new_interp.permute(0,2,1).contiguous()
                print(f"[INFO] Resized positional embedding: {L_old} -> {L_tgt}")
                new[key_ckpt] = new_pos
        return new

    base = torch.load(args.base_ckpt, map_location="cpu")
    base_state = base["model_state_dict"]
    base_state = _alias_keys(base_state, model)
    base_state = _remap_heads(base_state, model)
    base_state = _resize_positional_embed(base_state, model, key_model="pos")
    base_cfg = base.get("config", {})

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"[INFO] Loaded base weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:   print("  Missing (first 10):", missing[:10])
    if unexpected:print("  Unexpected (first 10):", unexpected[:10])

    max_block = detect_max_block(model)
    freeze_prefixes = [p.strip() for p in args.freeze_prefix.split(",") if p.strip()]

    pt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    assert pt_files, f"No .pt files found in {args.data_dir}"

    schedule = parse_unfreeze_schedule(args.unfreeze_schedule)

    for pt_path in pt_files:
        player_tag = Path(pt_path).stem
        run_dir = out_dir / player_tag; run_dir.mkdir(parents=True, exist_ok=True)

        model.load_state_dict(base_state, strict=False); model.to(device)

        # Initial freeze according to default/schedule at epoch 0
        last_n_now = current_last_n(schedule, epoch=0, default_last_n=args.unfreeze_last_n)
        apply_unfreeze(model, last_n_now, max_block=max_block, freeze_prefixes=freeze_prefixes)

        # Data + holdout
        ds = SinglePTDataset(pt_path)
        N = len(ds)
        holdout = min(args.holdout_last_n, N) if args.holdout_last_n > 0 else 0

        if holdout > 0:
            n_pool = N - holdout
            n_val = max(1, int(math.ceil(n_pool * args.val_split)))
            n_train = max(1, n_pool - n_val)
            train_ds = torch.utils.data.Subset(ds, range(0, n_train))
            val_ds   = torch.utils.data.Subset(ds, range(n_train, n_train + n_val))
            with open(run_dir / "holdout_info.txt", "w") as f:
                f.write(f"Total: {N}, Holdout_last: {holdout}, Used_for_train: {n_train}, Used_for_val: {n_val}\n")
        else:
            n_val = max(1, int(math.ceil(N * args.val_split)))
            n_train = max(1, N - n_val)
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        collate = (lambda b: collate_batch(b, args.max_tokens))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, collate_fn=collate, drop_last=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                pin_memory=True, collate_fn=collate)

        # Build optimizer after initial freeze
        opt = rebuild_optimizer(model, old_opt=None, base_lr=args.lr, weight_decay=args.weight_decay,
                                llrd_gamma=args.llrd_gamma, freeze_prefixes=freeze_prefixes)

        total_steps = args.epochs * max(1, len(train_loader))
        def lr_lambda(step):
            if total_steps <= 1: return 1.0
            cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            return 0.1 + 0.9 * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        ema = ModelEMA(model, decay=0.999) if args.ema else None

        best_val, best_acc, patience, bad_epochs, global_step = float("inf"), 0.0, 3, 0, 0
        metrics_path = run_dir / "metrics.jsonl"
        ckpt_best_path = run_dir / f"{player_tag}_best.pth"
        ckpt_last_path = run_dir / f"{player_tag}_last.pth"

        print(f"\\n===== Fine‑tuning on {player_tag}  (N={N} | holdout={holdout}) =====")
        print(f"[INFO] Unfreeze at epoch 0: last_n={last_n_now} (freeze_prefix={freeze_prefixes})")

        for epoch in range(1, args.epochs + 1):
            # Apply scheduled unfreeze at START of this epoch if changed
            want_last_n = current_last_n(schedule, epoch=epoch-1, default_last_n=args.unfreeze_last_n)
            if want_last_n != last_n_now:
                print(f"[INFO] Updating unfreeze: last_n {last_n_now} -> {want_last_n} at epoch {epoch-1}")
                last_n_now = want_last_n
                apply_unfreeze(model, last_n_now, max_block=max_block, freeze_prefixes=freeze_prefixes)
                # Rebuild optimizer to include newly trainable params
                opt = rebuild_optimizer(model, old_opt=opt, base_lr=args.lr, weight_decay=args.weight_decay,
                                        llrd_gamma=args.llrd_gamma, freeze_prefixes=freeze_prefixes)

            t0 = time.time()
            train_loss, global_step = train_one_epoch(model, ema, train_loader, opt, scaler, device,
                                                      label_smoothing=args.label_smoothing,
                                                      grad_clip=args.grad_clip, warmup_steps=args.warmup_steps,
                                                      step_idx_start=global_step)
            scheduler.step()
            val_stats = evaluate(model, val_loader, device, use_ema=ema)

            log = {"epoch": epoch,
                   "train_loss": train_loss,
                   "val_loss": val_stats["loss"],
                   "val_move_match_top1": val_stats["move_match_top1"],
                   "lr_head": opt.param_groups[0]["lr"],
                   "unfreeze_last_n": last_n_now}
            with open(metrics_path, "a") as f: f.write(json.dumps(log) + "\\n")
            print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_stats['loss']:.4f} | top1 {val_stats['move_match_top1']:.4f} | last_n {last_n_now} | time {time.time()-t0:.1f}s")

            torch.save({"model_state_dict": model.state_dict(),
                        "ema_state_dict": (ema.ema.state_dict() if ema is not None else None),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch, "config": base_cfg, "player_tag": player_tag}, ckpt_last_path)

            improved = (val_stats["loss"] < best_val - 1e-5) or (abs(val_stats["loss"] - best_val) < 1e-5 and val_stats["move_match_top1"] > best_acc + 1e-4)
            if improved:
                best_val, best_acc, bad_epochs = val_stats["loss"], val_stats["move_match_top1"], 0
                torch.save({"model_state_dict": (ema.ema.state_dict() if ema is not None else model.state_dict()),
                            "config": base_cfg, "epoch": epoch, "best_val_loss": best_val,
                            "best_val_top1": best_acc, "player_tag": player_tag}, ckpt_best_path)
                print(f"  ✔ Saved BEST to {ckpt_best_path}")
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered."); break
    print("\\nAll players done.")

if __name__ == "__main__":
    main()
