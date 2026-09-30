import csv
from dataclasses import dataclass


@dataclass
class Row:
    variant: str
    status: str = "FAIL"
    max_diff: float = float("nan")
    gpu_ms: float = float("nan")
    kernel_ms: float = float("nan")
    tflops: float = float("nan")
    gbps: float = float("nan")
    error: str = ""


def metrics(q, kernel_ms):
    B, nh, N, d = q.shape
    if kernel_ms != kernel_ms or kernel_ms <= 0:
        return float("nan"), float("nan")
    flops = 4.0 * B * nh * N * N * d
    bytes_moved = 4.0 * B * nh * N * d * 4
    tflops = flops / (kernel_ms * 1e-3) / 1e12
    gbps = bytes_moved / (kernel_ms * 1e-3) / 1e9
    return tflops, gbps


def _fmt(x):
    return "nan" if x != x else f"{x:.4f}"


def print_table(rows, q):
    B, nh, N, d = q.shape
    print(f"shape: B={B} nh={nh} N={N} d={d}")
    header = (f"{'variant':<12}{'status':<10}{'max_diff':>10}"
              f"{'gpu_ms':>10}{'kernel_ms':>11}{'TFLOPS':>9}{'GB/s':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (f"{r.variant:<12}{r.status:<10}{_fmt(r.max_diff):>10}"
                f"{_fmt(r.gpu_ms):>10}{_fmt(r.kernel_ms):>11}"
                f"{_fmt(r.tflops):>9}{_fmt(r.gbps):>9}")
        if r.error:
            line += f"  <- {r.error}"
        print(line)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "status", "max_diff", "gpu_ms", "kernel_ms",
                    "tflops", "gbps", "error"])
        for r in rows:
            w.writerow([r.variant, r.status, r.max_diff, r.gpu_ms, r.kernel_ms,
                        r.tflops, r.gbps, r.error])
