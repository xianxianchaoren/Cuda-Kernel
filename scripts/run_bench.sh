#!/usr/bin/env bash
# 一键 benchmark：自动构建 + 多组维度跑测
#
# 用法:
#   ./scripts/run_bench.sh                     # 全部默认
#   ./scripts/run_bench.sh --kernels naive     # 透传参数给 gemm_bench
#
# 环境变量:
#   BUILD_DIR  构建目录 (default: build)
#   BIN        gemm_bench 可执行文件路径 (default: $BUILD_DIR/gemm_bench)
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BIN:-$BUILD_DIR/gemm_bench}"

if [[ ! -x "$BIN" ]]; then
    echo "[run_bench] 未找到 $BIN，开始构建..."
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j
fi

# 覆盖多种形状：小/中/大 + 非对齐（边界）维度
SHAPES=(
    "256 256 256"
    "1024 1024 1024"
    #"4096 4096 4096"
    #"8192 8192 8192"
    "1000 999 1001"
)

for shape in "${SHAPES[@]}"; do
    read -r M N K <<< "$shape"
    echo ""
    echo "===================== M=$M N=$N K=$K ====================="
    "$BIN" -M "$M" -N "$N" -K "$K" "$@"
done
