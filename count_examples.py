#!/usr/bin/env python3
"""Count number of examples stored in each .pt file inside a data directory.

Assumptions:
- Each .pt file was saved via torch.save(obj) where obj is an indexable/len()-able
  collection (e.g. list of dict samples, list of tensors, etc.).
- Loading the file returns an object that supports len().

If files are huge and you worry about RAM, use --max-files to sample a subset or
--size-only for a raw byte size report (no loading) and optionally --estimate
which attempts to infer average sample size from a small head of one file.

Usage:
  python count_examples.py                # scan ./data/*.pt
  python count_examples.py --data-dir data --pattern "*.pt"
  python count_examples.py --size-only    # just file sizes
  python count_examples.py --estimate     # approximate counts without full loads
"""
from __future__ import annotations
import argparse
import os
import sys
import glob
import time
import json
from typing import List, Dict, Any

try:
    import torch  # Only needed for actual counting
    import torch.serialization as torch_serial  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    torch_serial = None  # type: ignore

# Pre-register common numpy globals often needed for dataset object unpickling in safe mode.
try:  # pragma: no cover - best effort
    import numpy as _np
    if torch_serial and hasattr(torch_serial, "add_safe_globals"):
        g_list = []
        try:
            g_list.append(_np.ndarray)
        except Exception:
            pass
        try:
            from numpy._core import multiarray as _np_multiarray  # type: ignore
            if hasattr(_np_multiarray, "_reconstruct"):
                g_list.append(getattr(_np_multiarray, "_reconstruct"))
        except Exception:
            pass
        if g_list:
            try:
                torch_serial.add_safe_globals(g_list)  # type: ignore[attr-defined]
            except Exception:
                pass
except Exception:
    pass

# -------- Safe load helper for PyTorch 2.6+ change (weights_only default True) -------- #

def torch_safe_load(path: str, map_location="cpu", allow_unpickled: bool = False, allow_globals: bool = True):
    """Load a .pt file robustly across PyTorch versions.

    PyTorch 2.6 switched default weights_only=True which breaks loading arbitrary
    Python objects (lists/dicts of samples). We explicitly request weights_only=False
    when the caller indicates the source is trusted (allow_unpickled=True).

    If not allowing unpickled execution but complex globals are needed, we can
    temporarily extend safe globals list (allow_globals=True) for common numpy
    reconstruction symbols.
    """
    if torch is None:
        raise RuntimeError("torch not available")

    # If user explicitly trusts file, bypass weights_only guard.
    if allow_unpickled:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older versions without weights_only param
            return torch.load(path, map_location=map_location)

    # Try safe loading first (weights_only default True). This will usually
    # fail for dataset object files; we then optionally expand safe globals.
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        needs_globals = 'Unsupported global:' in msg and allow_globals and torch_serial is not None
        if needs_globals:
            # Add commonly needed numpy reconstruct symbol to safe globals and retry.
            try:
                from numpy._core import multiarray as _np_multiarray  # type: ignore
                gsym = getattr(_np_multiarray, '_reconstruct', None)
                if gsym is not None and torch_serial and hasattr(torch_serial, "add_safe_globals"):
                    torch_serial.add_safe_globals([gsym])  # type: ignore[attr-defined]
                return torch.load(path, map_location=map_location)
            except Exception:
                raise
        # If still failing, instruct user to rerun with --allow-unpickled
        raise RuntimeError(
            f"Failed safe load for {path}: {e}. If this file is a dataset and trusted, rerun with --allow-unpickled" )

# -------- Helpers -------- #

def human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B","KiB","MiB","GiB","TiB"]
    i = 0
    f = float(n)
    while f >= step and i < len(units)-1:
        f /= step
        i += 1
    return f"{f:.2f} {units[i]}"

def sample_average_item_size(path: str, sample: int = 256, allow_unpickled: bool = False, allow_globals: bool = True) -> float:
    if torch is None:
        return 0.0
    obj = torch_safe_load(path, map_location="cpu", allow_unpickled=allow_unpickled, allow_globals=allow_globals)
    if not hasattr(obj, '__len__'):
        raise TypeError(f"Object in {path} has no __len__ for estimation")
    L = len(obj)
    if L == 0:
        return 0.0
    take = min(sample, L)
    # Serialize a slice to approximate average serialized size
    import io, pickle
    buf = io.BytesIO()
    # if sliceable
    if isinstance(obj, (list, tuple)):
        subset = obj[:take]
    else:
        # Fallback: sample first N iteration items
        subset = []
        for idx, item in enumerate(obj):
            subset.append(item)
            if idx+1 >= take:
                break
    pickle.dump(subset, buf, protocol=pickle.HIGHEST_PROTOCOL)
    subset_bytes = buf.tell()
    avg_item_serialized = subset_bytes / take if take else 0.0
    return avg_item_serialized

# -------- Main counting logic -------- #

def count_examples(paths: List[str], lazy: bool = False, allow_unpickled: bool = False, allow_globals: bool = True) -> Dict[str, Any]:
    results = []
    total = 0
    for p in paths:
        start = time.time()
        try:
            if not lazy:
                if torch is None:
                    raise RuntimeError("torch not available to load file; install torch or use --size-only/--estimate")
                obj = torch_safe_load(p, map_location='cpu', allow_unpickled=allow_unpickled, allow_globals=allow_globals)
            else:
                obj = None
            n = len(obj) if obj is not None and hasattr(obj, '__len__') else None
            elapsed = time.time() - start
            results.append({
                'file': os.path.basename(p),
                'path': p,
                'count': n,
                'size_bytes': os.path.getsize(p),
                'load_time_s': round(elapsed, 3)
            })
            if n is not None:
                total += n
        except Exception as e:
            results.append({
                'file': os.path.basename(p),
                'path': p,
                'error': str(e),
                'size_bytes': os.path.getsize(p)
            })
    return {'files': results, 'total': total}

def estimate_counts(paths: List[str], allow_unpickled: bool = False, allow_globals: bool = True) -> Dict[str, Any]:
    """Estimate counts without fully loading each file.

    Uses average serialized size of a sample subset from the first file.
    """
    if not paths:
        return {'files': [], 'total_estimated': 0}
    first = paths[0]
    try:
        avg_item_bytes = sample_average_item_size(first, allow_unpickled=allow_unpickled, allow_globals=allow_globals)
    except Exception as e:  # pragma: no cover
        # Retry once with unpickled if not already and globals allowed
        if not allow_unpickled:
            try:
                avg_item_bytes = sample_average_item_size(first, allow_unpickled=True, allow_globals=allow_globals)
            except Exception as e2:
                return {'error': f'Failed estimation on {first}: {e2}'}
        else:
            return {'error': f'Failed estimation on {first}: {e}'}
    results = []
    total_est = 0.0
    for p in paths:
        size_b = os.path.getsize(p)
        est = int(size_b / avg_item_bytes) if avg_item_bytes > 0 else 0
        results.append({
            'file': os.path.basename(p),
            'size_bytes': size_b,
            'estimated_count': est
        })
        total_est += est
    return {
        'files': results,
        'avg_item_bytes': avg_item_bytes,
        'total_estimated': int(total_est)
    }

# -------- CLI -------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Count examples in .pt dataset files")
    ap.add_argument('--data-dir', default='data', help='Directory containing .pt files')
    ap.add_argument('--pattern', default='*.pt', help='Glob pattern for dataset files')
    ap.add_argument('--max-files', type=int, default=None, help='Limit number of files processed')
    ap.add_argument('--json', action='store_true', help='Output JSON')
    ap.add_argument('--size-only', action='store_true', help='Do not load, only show sizes')
    ap.add_argument('--estimate', action='store_true', help='Estimate counts from serialized size (loads small sample)')
    ap.add_argument('--sort-by', choices=['name','count','size'], default='name')
    ap.add_argument('--allow-unpickled', action='store_true', help='Trust files; load with weights_only=False (enables arbitrary code execution)')
    ap.add_argument('--no-globals', action='store_true', help='Do not extend safe globals when safe loading fails')
    return ap.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.data_dir, args.pattern)
    paths = sorted(glob.glob(pattern))
    if args.max_files:
        paths = paths[:args.max_files]
    if not paths:
        print(f"No files matched pattern: {pattern}")
        return 1

    if args.estimate:
        data = estimate_counts(paths, allow_unpickled=args.allow_unpickled, allow_globals=not args.no_globals)
        if args.json:
            print(json.dumps(data, indent=2))
            return 0
        if 'error' in data:
            print(data['error'])
            return 1
        print(f"Estimation based on avg serialized item size: {data['avg_item_bytes']:.1f} bytes")
        print()
        rows = data['files']
        if args.sort_by == 'size':
            rows.sort(key=lambda r: r['size_bytes'], reverse=True)
        elif args.sort_by == 'name':
            rows.sort(key=lambda r: r['file'])
        elif args.sort_by == 'count':
            rows.sort(key=lambda r: r['estimated_count'], reverse=True)
        width = max(len(r['file']) for r in rows)
        print(f"{'FILE'.ljust(width)}  SIZE        EST_COUNT")
        for r in rows:
            print(f"{r['file'].ljust(width)}  {human_bytes(r['size_bytes']).rjust(10)}  {r['estimated_count']:>10}")
        print(f"\nEstimated total examples: {data['total_estimated']:,}")
        return 0

    if args.size_only:
        rows = []
        for p in paths:
            rows.append({'file': os.path.basename(p), 'size_bytes': os.path.getsize(p)})
        if args.json:
            print(json.dumps({'files': rows}, indent=2))
            return 0
        width = max(len(r['file']) for r in rows)
        print(f"{'FILE'.ljust(width)}  SIZE")
        for r in rows:
            print(f"{r['file'].ljust(width)}  {human_bytes(r['size_bytes']).rjust(10)}")
        total_size = sum(r['size_bytes'] for r in rows)
        print(f"\nTotal size: {human_bytes(total_size)}")
        return 0

    if torch is None:
        print("torch not installed; cannot count examples without --size-only or --estimate")
        return 1

    data = count_examples(paths, allow_unpickled=args.allow_unpickled, allow_globals=not args.no_globals)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    rows = data['files']
    # Sorting
    if args.sort_by == 'count':
        rows.sort(key=lambda r: (r.get('count') is None, r.get('count', 0)), reverse=True)
    elif args.sort_by == 'size':
        rows.sort(key=lambda r: r['size_bytes'], reverse=True)
    else:  # name
        rows.sort(key=lambda r: r['file'])

    width = max(len(r['file']) for r in rows)
    print(f"{'FILE'.ljust(width)}  COUNT        SIZE       LOAD_S  NOTES")
    for r in rows:
        note = r.get('error', '')
        count_str = 'N/A' if r.get('count') is None else f"{r['count']:,}"
        print(f"{r['file'].ljust(width)}  {count_str:>10}  {human_bytes(r['size_bytes']).rjust(9)}  {r.get('load_time_s',''):>6}  {note}")
    print(f"\nTotal examples: {data['total']:,}")
    total_bytes = sum(r['size_bytes'] for r in rows)
    print(f"Aggregate size: {human_bytes(total_bytes)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
