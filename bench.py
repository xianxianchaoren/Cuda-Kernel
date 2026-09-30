import argparse
import sys

import torch

from harness.loader import discover_variants, load_variant
from harness.reference import compare, reference
from harness.report import Row, metrics, print_table, write_csv
from harness.timing import cuda_time, kernel_time


def short_error(exc):
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    for ln in lines:
        if "error:" in ln.lower():
            return ln[:160]
    if not lines:
        return type(exc).__name__
    return lines[-1][:160]


def parse_args():
    p = argparse.ArgumentParser(description="Unified FlashAttention benchmark")
    p.add_argument("--variants", nargs="+", default=None, help="variants to run (default: all)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    available = discover_variants()
    names = args.variants or available
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; available: {available}")

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.seq, args.dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = reference(q, k, v)

    rows = []
    for name in names:
        print(f"[{name}] building ...", flush=True)
        row = Row(variant=name)
        try:
            module, entry = load_variant(name)
            fn = getattr(module, entry)

            out = fn(q, k, v)
            torch.cuda.synchronize()
            ok, max_diff = compare(out, ref, atol=args.atol)
            row.max_diff = max_diff

            row.gpu_ms = cuda_time(lambda: fn(q, k, v), args.warmup, args.iters)
            row.kernel_ms = kernel_time(lambda: fn(q, k, v), args.warmup, args.iters)
            row.tflops, row.gbps = metrics(q, row.kernel_ms)
            row.status = "OK" if ok else "MISMATCH"
        except Exception as exc:
            row.error = short_error(exc)
        rows.append(row)

    print()
    print_table(rows, q)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nwrote {args.csv}")

    return 0 if all(r.status == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
