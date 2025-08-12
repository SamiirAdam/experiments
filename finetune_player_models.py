
#!/usr/bin/env python3
# finetune_player_models.py (patched for PyTorch 2.6+ dataset loading)
# - Adds robust torch.load handling for .pt datasets created with older numpy/pickle:
#   * Prefer weights_only=False for trusted local data
#   * Fall back to safe allowlisting of numpy._core.multiarray._reconstruct
# - Keeps positional-embed resize + head remap + holdout support from previous version

# python finetune_player_models.py \                                                                                                                                   thesis 18:54:57
#   --data_dir ./data \
#   --base_ckpt ./best_vit_amd.pth \
#   --max_tokens 256 \
#   --holdout_last_n 20000 --ema

# pip install --upgrade pip
# pip install --upgrade "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu121
# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

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

# ---------- Dataset (patched torch.load) ----------
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
        # Preferred path for trusted local datasets
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch without weights_only kw
            return torch.load(path, map_location="cpu")
        except Exception as e:
            # Allowlist numpy reconstruct if needed (PyTorch 2.6 safe-serialization)
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

# def train_one_epoch(model, ema, loader, opt, scaler, device, label_smoothing: float = 0.0,
#                     grad_clip: float = 1.0, warmup_steps: int = 0, step_idx_start: int = 0):
#     model.train()
#     total_loss, total_batches = 0.0, 0
#     ce_kwargs = {"label_smoothing": label_smoothing} if label_smoothing > 0 else {}
#     step_idx = step_idx_start
#     for batch in loader:
#         step_idx += 1
#         if len(batch) == 6:
#             x, pad, y_from, y_to, legal_from, legal_to = batch
#         else:
#             raise RuntimeError("Dataset must include legal_mask_from/dest for masked CE.")
#         x, pad, y_from, y_to = to_device((x, pad, y_from, y_to), device)
#         legal_from, legal_to = to_device((legal_from, legal_to), device)

#         tgt_from = onehot_to_index(y_from)
#         tgt_to   = onehot_to_index(y_to)

#         with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
#             lf, lt = forward_model(model, x, pad)
#             lf = mask_logits_with_legal(lf, legal_from)
#             lt = mask_logits_with_legal(lt, legal_to)
#             loss = F.cross_entropy(lf, tgt_from, **ce_kwargs) + F.cross_entropy(lt, tgt_to, **ce_kwargs)

#         opt.zero_grad(set_to_none=True)
#         scaler.scale(loss).backward()
#         if grad_clip: 
#             scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#         if warmup_steps and step_idx <= warmup_steps:
#             for pg in opt.param_groups: pg['lr'] = pg['lr'] * step_idx / warmup_steps
#         scaler.step(opt); scaler.update()
#         if ema is not None: ema.update(model)

#         total_loss += loss.item(); total_batches += 1
#     return total_loss / max(total_batches, 1), step_idx

def train_one_epoch(model, ema, loader, opt, scaler, device, label_smoothing: float = 0.0,
                    grad_clip: float = 1.0, warmup_steps: int = 0, step_idx_start: int = 0,
                    grad_accum_steps: int = 2):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss, total_updates = 0.0, 0
    ce_kwargs = {"label_smoothing": label_smoothing} if label_smoothing > 0 else {}
    step_idx = step_idx_start  # counts optimizer update steps
    micro = 0
    for micro, batch in enumerate(loader):
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
            loss = (F.cross_entropy(lf, tgt_from, **ce_kwargs) +
                    F.cross_entropy(lt, tgt_to, **ce_kwargs))
            loss = loss / grad_accum_steps  # scale for accumulation

        scaler.scale(loss).backward()

        do_update = ((micro + 1) % grad_accum_steps == 0) or (micro + 1 == len(loader))
        if do_update:
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Warmup based on update steps
            if warmup_steps and (step_idx + 1) <= warmup_steps:
                warm_scale = (step_idx + 1) / warmup_steps
                for pg in opt.param_groups:
                    pg['lr'] = pg['lr_initial'] * warm_scale if 'lr_initial' in pg else pg['lr']
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            step_idx += 1
            total_loss += loss.item() * grad_accum_steps
            total_updates += 1
    avg_loss = total_loss / max(total_updates, 1)
    return avg_loss, step_idx

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

def main():
    parser = argparse.ArgumentParser(description="Fine‑tune per‑player ViT models from a base checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs_players")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--llrd_gamma", type=float, default=0.8)
    parser.add_argument("--freeze_prefix", type=str, default="emb,patch_embed,cls,pos,stem")
    parser.add_argument("--unfreeze_last_n", type=int, default=4)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=256, help="Target sequence length for model pos embeddings")
    parser.add_argument("--holdout_last_n", type=int, default=0, help="Leave last N examples untouched per player")
    args = parser.parse_args()
    
    os.environ['PYTORCH_DISABLE_CUDNN_BATCH_NORM'] = '1'
    os.environ['TORCH_COMPILE_DEBUG'] = '0' 

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True 

    if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name():
        # Enable AMD-specific optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # AMD memory management
        torch.cuda.empty_cache()
        
        # Set memory fraction to use most of the 24GB
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable AMD's optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except:
            pass

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

    ckpt = load_base_checkpoint(args.base_ckpt, model)
    base_state = ckpt["state"]

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"[INFO] Loaded base weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:   print("  Missing (first 10):", missing[:10])
    if unexpected:print("  Unexpected (first 10):", unexpected[:10])

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
    max_block = max(block_ids) if block_ids else None
    freeze_prefixes = [p.strip() for p in args.freeze_prefix.split(",") if p.strip()]

    pt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    assert pt_files, f"No .pt files found in {args.data_dir}"

    for pt_path in pt_files:
        player_tag = Path(pt_path).stem
        run_dir = out_dir / player_tag; run_dir.mkdir(parents=True, exist_ok=True)

        model.load_state_dict(base_state, strict=False); model.to(device)

        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in freeze_prefixes): p.requires_grad_(False)
        if max_block is not None and args.unfreeze_last_n > 0:
            keep_from = max_block - args.unfreeze_last_n + 1
            for name, p in model.named_parameters():
                if ".blocks." in name or ".layers." in name:
                    try:
                        parts = name.split(".")
                        if "blocks" in parts: bidx = int(parts[parts.index("blocks")+1])
                        elif "layers" in parts: bidx = int(parts[parts.index("layers")+1])
                        else: continue
                        if bidx < keep_from: p.requires_grad_(False)
                    except: pass

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

        param_groups = build_param_groups(model, base_lr=args.lr, weight_decay=args.weight_decay, llrd_gamma=args.llrd_gamma,
                                          freeze_prefixes=freeze_prefixes)
        opt = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.95))
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
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, global_step = train_one_epoch(model, ema, train_loader, opt, scaler, device,
                                                      label_smoothing=args.label_smoothing,
                                                      grad_clip=args.grad_clip, warmup_steps=args.warmup_steps,
                                                      step_idx_start=global_step, grad_accum_steps=2)
            scheduler.step()
            val_stats = evaluate(model, val_loader, device, use_ema=ema)

            log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_stats["loss"],
                   "val_move_match_top1": val_stats["move_match_top1"], "lr_head": opt.param_groups[0]["lr"]}
            with open(metrics_path, "a") as f: f.write(json.dumps(log) + "\\n")
            print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_stats['loss']:.4f} | top1 {val_stats['move_match_top1']:.4f} | time {time.time()-t0:.1f}s")

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
