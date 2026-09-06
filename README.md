# Cuda-Kernel — GEMM 内核测试框架

面向 **矩阵乘法 CUDA kernel 学习与优化** 的测试基准框架：

- **专注写 kernel**：只需实现 `__global__` 函数 + launch 逻辑 + 一行注册宏
- **正确性校验**：与 CPU double 累加参考实现对比，报告最大相对/绝对误差
- **性能基准**：CUDA Event 计时 + warmup + 多轮取最优/平均 + GFLOPS 报告
- **多内核同台竞技**：naive / tiled / vectorized / tensor core… 注册后自动对比
- **参数化**：M/N/K、迭代次数、seed、误差阈值全部命令行可配

> 本地（macOS）只负责写代码，编译运行在 Linux + NVIDIA GPU 服务器上进行。

## 目录结构

```
├── CMakeLists.txt          # 构建脚本（自动收集 kernels/*.cu）
├── include/gemm/           # 框架头文件（无需修改）
│   ├── common.h            # CUDA_CHECK 错误检查宏
│   ├── registry.h          # 内核注册机制（核心契约）
│   ├── data.h              # 随机数据生成
│   ├── reference.h         # CPU 参考实现声明
│   ├── checker.h           # 正确性校验
│   └── timer.h             # CUDA Event 计时器
├── src/                    # 框架实现 + main.cpp
├── kernels/                # ★ 你的主战场
│   ├── gemm_naive.cu       # 示例 naive（baseline）
│   └── gemm_tiled.cu       # 工作区 stub（已注释注册，实现后取消注释）
└── scripts/run_bench.sh    # 一键 benchmark
```

## 服务器部署

```bash
git clone <repo> && cd Cuda-Kernel

# 可选：指定 GPU 架构（不指定则自动探测）
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 40 系
cmake --build build -j
```

常用构建参数：

| 参数 | 说明 |
|------|------|
| `-DCMAKE_CUDA_ARCHITECTURES=89` | 指定 GPU 架构（80=A100, 89=RTX40, 90=H100） |
| `-DGEMM_USE_FAST_MATH=ON` | 打开 `--use_fast_math`（更快但可能失精度） |
| `-DCMAKE_BUILD_TYPE=Release` | Release 构建 |

## 使用

```bash
./build/gemm_bench --list                          # 列出已注册 kernel

./build/gemm_bench -M 4096 -N 4096 -K 4096         # 全部 kernel 跑 4096³
./build/gemm_bench --kernels naive,tiled -M 4096 -N 4096 -K 4096
./build/gemm_bench -M 1000 -N 999 -K 1001 --iters 5 --seed 42

./scripts/run_bench.sh                             # 一键：多形状自动跑
```

输出示例：

```
GPU      : NVIDIA GeForce RTX 4090 (sm_89)
Problem  : C(4096x4096) = A(4096x4096) x B(4096x4096), 137.439 GFLOP
Kernels  : naive
Config   : iters=10 warmup=3 seed=42 rel_tol=1.0e-02 abs_tol=1.0e-04 l2flush=0MB

kernel     result    best(ms)    avg(ms)     max-rel    max-abs    GFLOPS
----------------------------------------------------------------------------------
naive      PASS      25.3120     25.4100     1.2e-06    5.4e-05    5429.2
```

> 结果中的 `max-rel` / `max-abs` 是相对 CPU 参考的误差；`GFLOPS` 按最优耗时计算。
> 若结果为 `FAIL`，会额外打印 `rms_err` 辅助排查。

完整参数见 `./build/gemm_bench --help`。

## 添加你自己的 kernel（3 步）

以 `kernels/` 下新建 `gemm_mine.cu` 为例：

```cpp
// 1. 实现 __global__ kernel
__global__ void gemm_mine_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    // ... 你的实现
}

// 2. 实现 host launcher（签名必须与契约一致）
void launch_gemm_mine(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K, cudaStream_t stream) {
    // ... 配置 grid/block 并 launch，kernel 必须跑在传入的 stream 上
}

// 3. 注册（一行宏，框架自动发现）
REGISTER_GEMM_KERNEL("mine", launch_gemm_mine);
```

**CMake 会自动收集 `kernels/*.cu`**，新增文件无需改构建脚本，重新构建即可：

```bash
cmake --build build -j
./build/gemm_bench --kernels mine -M 4096 -N 4096 -K 4096
```

### 内核契约

```
C(MxN) = A(MxK) * B(KxN)     # row-major, float32, 数据范围 [-1, 1]
void launcher(const float* d_A, const float* d_B, float* d_C,
              int M, int N, int K, cudaStream_t stream)
```

框架负责：内存分配、H2D/D2H、CPU 参考计算、计时、校验。launcher 只负责 launch。
**注意**：M/N/K 不一定是 tile 的整数倍，kernel 必须处理边界越界（参考 `gemm_naive.cu`）。

## GEMM 优化路线图（参考）

| 阶段 | 技术 | 预期收益 |
|------|------|---------|
| v0 | naive（每线程一个元素） | baseline |
| v1 | shared memory tiling（`__syncthreads`） | 消除全局重复访存 |
| v2 | float4 向量化访存 + 边界处理 | 减少访存指令 |
| v3 | 双缓冲 / 数据预取 | 隐藏 global→shared 延迟 |
| v4 | `cp.async`（Ampere+） | 异步拷贝，进一步隐藏延迟 |
| v5 | 1D block tile / swizzle 优化 L2 命中 | 大 K 场景提升 |
| v6 | tensor core（`mma` / cutlass） | 数量级提升 |
| v7 | split-K / 多 stream / 流水线 | 大 K 或超大矩阵 |

每个阶段都建议：**先过正确性，再看 GFLOPS，用 `--kernels` 与上一版同台对比**。

## 调试工具

```bash
# 内存访问越界检测（强烈建议 kernel 报错时使用）
compute-sanitizer ./build/gemm_bench -M 256 -N 256 -K 256

# 性能剖析（按函数耗时排序）
ncu --set full -o profile ./build/gemm_bench -M 4096 -N 4096 -K 4096
```

## 常见问题

- **`no kernel selected`**：先 `./build/gemm_bench --list` 确认 kernel 已注册。
- **`FAIL`**：检查边界处理、`__syncthreads` 是否配对、寄存器/共享内存溢出（看 `compute-sanitizer`）。
- **启动报 CUDA error**：`CUDA_CHECK` 会打印出错文件行号；grid/block 配置错误会在 `CUDA_CHECK_LAST` 处暴露。
- **新增 .cu 未被编译**：CMake 用了 `CONFIGURE_DEPENDS` 自动扫描，若仍不生效，删除 `build/` 重新配置。
