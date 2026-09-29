import math
from pathlib import Path

import matplotlib.pyplot as plt

M = N = K = 4096
BANDWIDTH_GBPS = 936.2
FP32_PEAK_GFLOPS = 35580.0

results = {
    "naive": 2304.6,
    "tiled": 2987.5,
    "tiled16": 3081.6,
    "regblock": 9690.9,
    "vec4": 10660.3,
    "dbuf": 8892.4,
    "cpasync": 11820.5,
    "cutlass_simt": 21410.3,
    "tc_tf32": 3898.2,
}

def arithmetic_intensity(tile_m, tile_n):
    flops = 2.0 * M * N * K
    bytes_moved = 4.0 * (M * N * K / tile_n + M * N * K / tile_m + M * N)
    return flops / bytes_moved

intensities = {
    "naive": 2.0 * M * N * K / (4.0 * (2.0 * M * N * K + M * N)),
    "tiled": arithmetic_intensity(32, 32),
    "tiled16": arithmetic_intensity(16, 16),
    "regblock": arithmetic_intensity(64, 64),
    "vec4": arithmetic_intensity(64, 64),
    "dbuf": arithmetic_intensity(64, 64),
    "cpasync": arithmetic_intensity(64, 64),
    "cutlass_simt": arithmetic_intensity(128, 128),
    "tc_tf32": arithmetic_intensity(16, 16),
}

x_values = [10 ** (-1 + i * 3 / 500) for i in range(501)]
memory_roof = [BANDWIDTH_GBPS * x for x in x_values]
compute_roof = [FP32_PEAK_GFLOPS] * len(x_values)
roof = [min(memory, compute) for memory, compute in zip(memory_roof, compute_roof)]
ridge = FP32_PEAK_GFLOPS / BANDWIDTH_GBPS

plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
ax.loglog(x_values, memory_roof, "--", color="#4C78A8", linewidth=2, label="DRAM roof: 936.2 GB/s")
ax.loglog(x_values, compute_roof, "--", color="#E45756", linewidth=2, label="FP32 compute roof: 35.58 TFLOPS")
ax.loglog(x_values, roof, color="#222222", linewidth=2.5, label="FP32 Roofline")
ax.axvline(ridge, color="#888888", linestyle=":", linewidth=1.5)
ax.text(ridge * 1.06, 350, f"Ridge point: {ridge:.1f} FLOP/B", color="#666666")

colors = {
    "naive": "#888888",
    "tiled": "#F2CF5B",
    "tiled16": "#B279A2",
    "regblock": "#59A14F",
    "vec4": "#76B7B2",
    "dbuf": "#EDC948",
    "cpasync": "#E15759",
    "cutlass_simt": "#FF9D00",
    "tc_tf32": "#4C78A8",
}

offsets = {
    "naive": (6, 14),
    "tiled": (6, 12),
    "tiled16": (8, 8),
    "regblock": (8, 12),
    "vec4": (8, 2),
    "dbuf": (8, -12),
    "cpasync": (8, 14),
    "cutlass_simt": (8, 8),
    "tc_tf32": (8, 14),
}

for name, performance in results.items():
    intensity = intensities[name]
    ax.scatter(intensity, performance, s=90, color=colors[name], edgecolor="black", linewidth=0.6, zorder=4)
    ax.annotate(name, (intensity, performance), xytext=offsets[name], textcoords="offset points", fontsize=8)

ax.set_title("RTX 3090 GEMM Roofline (4096³, analytical arithmetic intensity)", pad=14)
ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
ax.set_ylabel("Performance [GFLOP/s]")
ax.set_xlim(0.1, 100)
ax.set_ylim(100, 100000)
ax.legend(loc="upper left")
fig.text(0.5, 0.01, "AI counts explicit global A/B loads and C stores; cache reuse, control overhead, and edge effects are excluded.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))

output = Path(__file__).resolve().parent.parent / "roofline_4096.png"
fig.savefig(output, bbox_inches="tight")

for name in results:
    print(f"{name:10s} AI={intensities[name]:.4f} FLOP/B  performance={results[name]:.1f} GFLOPS")
