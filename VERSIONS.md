# FlashAttention 各版本更新记录

本仓库用 **统一的 bench** 横向对比各个 FlashAttention 实现版本。

## 测试框架

```
Cuda-Kernel/
├── bench.py                 # 统一入口：自动发现并测试所有 variant
├── harness/                 # loader / reference / timing / report
├── FA-minimal/              # v1
├── FA-minimal-v2/           # v2
├── FA-minimal-v3/           # v3
├── FA-minimal-v4/           # v4
└── FA-minimal-v5/           # v5
```

**约定**：每个目录放 `main.cpp`（pybind 暴露 `forward(q,k,v)`）、kernel 源文件，以及清单 `variant.py`：

```python
NAME       = "FA-minimal-v4"
SOURCES    = ["main.cpp", "flash.cu"]
CUDA_FLAGS = ["-O2"]
ENTRY      = "forward"
```

新增算法 → 新建目录 + 上述三个文件即可被自动收录并编译（编译产物落在 `.build/<NAME>/`）。

**用法**

```bash
python bench.py                              # 全部版本，默认 B=16 nh=12 N=512 d=64
python bench.py --variants FA-minimal-v4     # 指定版本
python bench.py --seq 64 --iters 20          # 指定尺寸/迭代
python bench.py --csv out.csv                # 导出结果
```

**输出指标**

| 指标 | 含义 |
|---|---|
| `status` / `max_diff` | 与参考实现（torch manual attention）比对，是否 allclose 及最大绝对误差 |
| `gpu_ms` | CUDA Event 计的 GPU 时间线时长 / iters（含 kernel 之间的空隙） |
| `kernel_ms` | kineto profiler 汇总的**纯 kernel 执行时间** / iters |
| `TFLOPS` | `4·B·nh·N²·d / kernel_ms`，按理论 FLOPs 估算 |
| `GB/s` | 按「读 QKV + 写 O」的最小访存量估算的等效带宽 |

> 跨版本比较请用 **`kernel_ms`**：它只统计 kernel 本体，重复运行稳定。
> `gpu_ms` 包含 kernel 之间的空隙（launch 间隙、`zeros_like` 的 memset 等），
> 实测同一版本反复跑可以在 2.99 ~ 3.44 ms 之间飘动，不要用它做结论。

---

## v1 —— FA-minimal（FlashAttention-1 式）

- **并行粒度**：一个 block 负责一个 `(batch, head)`；`grid = (B, nh)`，`block = 32`
- **循环结构**：外层遍历 K/V tile（`j`），内层遍历 Q tile（`i`）
- **状态存放**：`l`、`m` 存 HBM；`O` 在 HBM 上反复 read-modify-write
- **smem**：`3·Bc·d + Bc·Br` = **28672 B** → 3 blocks/SM，`blockDim=32` → 3 warps/SM

问题：全局访存未合并（每线程连续读整行，warp 内地址间隔 256B）；smem 全部 32 路 bank conflict。

## v2 —— FA-minimal-v2（FlashAttention-2 式循环顺序）

- **并行粒度**：同 v1，一个 block 负责一个 `(batch, head)`
- **循环结构**：改为外层 `i`（Q tile）、内层 `j`（K/V tile）
- **状态存放**：`l`/`m` 放寄存器；`Oi` 在 shared memory 上在线累加，每个 `i` 只写回 HBM 一次
- **smem**：`2·Bc·d + 2·Br·d + 2·Bc·Br` = **40960 B** → 2 blocks/SM

问题：smem 变大导致每 SM 只能驻留 2 个 block；默认 `grid = B·nh = 192` 时凑不满整波，产生「尾波」空转，反而比 v1 慢。

## v3 —— FA-minimal-v3（序列维度并行）

- **核心改动**：`grid` 按 Q tile 展开 —— `dim3(Tr, B, nh)`，**一个 block 只算一个 Q tile**（论文 FA2 的做法）
- `blockIdx.x` = Q tile 序号，`blockIdx.y` = batch，`blockIdx.z` = head
- Q tile 只加载一次，扫 `j` 时把 `Oi` 在线累加在 smem，最后一次性写回
- **smem**：`2·Br·d + 2·Bc·d + Bc·Br` = **36864 B**（S 只占一份）→ 2 blocks/SM

收益：并行度从 `B·nh` 提升到 `Tr·B·nh`，消除了 grid 欠占用/尾波问题；**N 越大收益越明显**（N=512 时成为最快版本）。

## v4 —— FA-minimal-v4（合并访存 + 削减 smem + 提高占用率）

本次做了三件事：

**Step A —— 消除 bank conflict**
smem 每行 padding 1 个 float（行 stride 改为 `d+1`），彻底消除原先 S / Qi / Kj 的 32 路 bank conflict。

**Step B —— 合并访存**
tile 拷贝从「每线程搬自己一整行」改为「整个 block 线性协作」：
```cuda
for (int idx = tid; idx < BR * D; idx += NTHREADS)
    Qi[(idx / D) * DS + (idx % D)] = Q[qkv_offset + q_off + idx];
```
连续线程访问连续全局地址 → 完全合并；smem 侧也连续 → 无 bank conflict。
（进一步的 `float4` 向量化见 v5）

**Step C —— 削减 smem + 提高线程数**
- `blockDim` 从 32 提到 **128**（32 行 × 4 线程协作一行，每线程负责 `d/4` 个元素）
- `rowmax` / `rowsum` 用 `__shfl_xor_sync` 在 4 个协作线程间归约
- `P` 用 `__shfl_sync` 在协作线程间凑齐，`Oi` 和 `P` **全部放寄存器**
- smem 只剩 `Qi + Kj + Vj`（含 padding）= **24960 B** → **4 blocks/SM**

结果：`d=64` 时 **72 寄存器、0 spill**；每 SM 占用从 2 warps 提升到 **16 warps**。

其他：kernel 按 `d` 做模板特化（支持 32/64/128），并加了 `N` 可整除性检查。

### 为什么这里必须用 shuffle

v4 把线程映射从「1 线程 = 1 行」改成了「**4 线程协作 1 行**」（`NW=4`，4 个线程是 warp 内连续的 lane）：

| 角色 | 每线程负责的范围 |
|---|---|
| 算 `S`（`QK^T`） | 本行的 **8 个 y 列**（`y = yi*NW + g`） |
| 算 `O`（`PV`） | 本行的 **16 个 x 列**（`x = t*NW + g`） |

于是出现两个**跨线程**的需求：

1. **softmax 是整行的操作**：`rowmax` 和 `rowsum` 必须覆盖全部 `BC=32` 个 y，但每个线程手里只有 8 个。4 个线程必须交换部分结果，且交换后要保证 4 个线程拿到**一致的** `m`、`l`（否则在线 softmax 的 `alpha/beta` 会对不上，结果直接错）。
2. **`P` 的分布与消费方式不一致**：`P[row][y]` 按 y 切分给了 4 个线程（各 8 个），但 `PV` 需要整行 32 个 `P`。所以必须做一次**warp 内的 8×4 转置/广播**。

`__shfl_xor_sync` / `__shfl_sync` 正好是这两件事的**唯一零开销原语**：
- 4 个协作线程是**连续的 lane**，所以 `__shfl_xor_sync(mask, v, 1/2)` 两轮就精确归约到这一组 4 个线程（`reduce_max_n` / `reduce_sum_n`），不越界到别的行；
- `__shfl_sync(mask, p[yi], src0 + r)` 直接从第 `r` 个协作线程取值，32 条指令就把 `Pall[32]` 凑齐，**全程在寄存器里**。

对比其它做法：

| 做法 | 代价 |
|---|---|
| **shuffle（现方案）** | 纯寄存器操作，0 字节 smem，0 次 `__syncthreads` |
| smem 归约 | 需要额外 smem + 每个 `j` 迭代多 2~3 次 `__syncthreads`，且又引入 bank conflict 风险 |
| 每线程自己算全部 32 个 y | `QK^T` 计算量 ×4（冗余），S 需要 32 个寄存器 |
| 用 `atomicAdd` | 慢几个数量级 |

关键前提是 **NW 必须整除 warp 大小且协作线程在 warp 内连续**（`NW ∈ {4,8,16}`），这样归约和广播都不需要跨 warp，才能完全避开 `__syncthreads`。

---

## v5 —— FA-minimal-v5（float4 向量化全局加载）

在 v4 的基础上，只改 **tile 搬运**这一块，把每条指令的搬运量从 4B 提到 16B：

```cuda
const float4* Qv = reinterpret_cast<const float4*>(Q + qkv_offset + q_off);
for (int idx = tid; idx < BR * DV; idx += NTHREADS) {   // DV = D / 4
    const int r = idx / DV, c = (idx % DV) * 4;
    const float4 v = Qv[idx];
    Qi[r * DS + c + 0] = v.x;  Qi[r * DS + c + 1] = v.y;
    Qi[r * DS + c + 2] = v.z;  Qi[r * DS + c + 3] = v.w;
}
```

K / V 的 tile 拷贝同样改成 `float4` 读。配套改动：
- `extern __shared__ __align__(16) float sram[]` 保证 16B 对齐
- `static_assert(D % 4 == 0)`（当前支持 32/64/128 都满足）

**为什么 smem 侧还是 4 个标量写**
行 stride 是 `D+1`（v4 为消 bank conflict 做的 padding），行首地址不是 16B 对齐，无法直接 `float4` 写入。
如果把 padding 改成 `D+4` 来对齐：`(BR + 2·BC)·(D+4)·4 = 26112 B > 25600 B`，
每 SM 可驻留的 block 数会从 **4 掉到 3**，占用率反而下降，得不偿失。

**实测（同进程对照，B=16 nh=12 d=64）**

| N | v4 kernel_ms | v5 kernel_ms | 加速 |
|---|---|---|---|
| 64 | 0.0731 | **0.0699** | 4.6% |
| 512 | 3.5047 | **3.4425** | 1.8% |
| 2048 | 47.21 | **46.20** | 2.1% |

寄存器与 smem 均未变化（d=64 仍 72 寄存器、0 spill、24960 B、4 blocks/SM）。

---

---

## 已尝试但未采纳

### tile 形状调整（BR 32→64 / BC 32→16）

**动机**：v5 每 SM 驻留 4 个 block × 4 warps = 16 warps（33% 占用率），
想通过「行数加倍、K/V tile 减半」在 smem 总量不变的前提下把 warps/SM 提到 32。

**实测（B=16 nh=12 d=64，交错 4 轮取最小）**：

| 配置 | N=64 | N=512 | N=2048 |
|---|---|---|---|
| v5 `BR32 BC32` | **0.070** | 2.974 | 46.137 |
| `BR64 BC32` | 0.085 | **2.931** | **45.265** |
| `BR32 BC16` | 0.076 | 3.162 | 48.692 |
| `BR64 BC16` | 0.087 | 3.108 | 47.990 |

`BR64 BC32` 的规模扫描：

```
     N    Tr   BR32 ms   BR64 ms   比值
    64     1     0.070     0.085  0.833x
   128     2     0.245     0.246  0.994x
   256     4     0.882     0.916  0.963x
   512     8     3.011     2.951  1.020x
  1024    16    11.626    11.472  1.013x
  2048    32    45.942    45.065  1.019x
```

**结论**：
- `BC` 改 16 **全线变差**：`NY = BC/NW` 从 8 降到 4，S 阶段的独立累加链减半（ILP 损失），
  叠加 `Tc` 翻倍（`__syncthreads` 次数翻倍），把占用率收益全吃掉。
- `BR` 改 64 只在大 N 有 ~2% 收益，N=64 时 **-17%**：此时 `Tr=1`，
  grid 仅 `1×16×12=192` 个 block × 256 线程 = 49152 线程，而 GPU 可容纳 125952 线程 → **只填了 39%**。
  `BR=32` 时是 384 个 block，装得更满。

整体不构成改进，未采纳；验证代码已删除。

### 修正后的性能诊断

「warps/SM 16→32 就能提速」的假设被证伪（`BR64 BC32` 把 warps/SM 提到 24，性能只动 2%）。
真正的天花板是 **smem 读 / MAC 的比值**：

```
loads per MAC = (R + NY) / (R × NY) = 1/NY + 1/R
```

当前 `R=1`（每线程 1 行）、`NY=8` → **1.125 次 smem 读 / MAC**。
smem 吞吐 128 B/cycle = 32 float/cycle → **MACs/cycle ≤ 32/1.125 ≈ 28**，
而 FP32 FMA 峰值是 **128 MACs/cycle/SM** → **天花板只有峰值的 ~22%**，实测 10.5% TFLOPS 已贴着它。

所以继续调 tile 形状 / 占用率都无效，**下一步应该加大 `R`（每线程算多行）来提升算术强度**：

| R | NY | loads/MAC | MACs/cycle 上限 | 占峰值 |
|---|---|---|---|---|
| 1 | 8 | 1.125 | 28 | 22%（现状） |
| 2 | 8 | 0.625 | 51 | 40% |
| 4 | 8 | 0.375 | 85 | 66% |

代价：累加器寄存器 `R×NY` 增长，且要重新设计线程映射保持 bank conflict-free。

---

## 各版本对比

### B=16, nh=12, N=512, d=64（iters=20）

| 版本 | grid | block | smem | blocks/SM | warps/SM | N=64 kernel_ms | N=512 kernel_ms |
|---|---|---|---|---|---|---|---|
| v1 FA-minimal | `(B, nh)` = 192 | 32 | 28672 B | 3 | 3 | 1.110 | 59.41 |
| v2 FA-minimal-v2 | `(B, nh)` = 192 | 32 | 40960 B | 2 | 2 | 1.690 | 90.67 |
| v3 FA-minimal-v3 | `(Tr, B, nh)` | 32 | 36864 B | 2 | 2 | 1.357 | 56.75 |
| v4 FA-minimal-v4 | `(Tr, B, nh)` | 128 | 24960 B | 4 | 16 | 0.073 | 3.50 |
| **v5 FA-minimal-v5** | `(Tr, B, nh)` | 128 | 24960 B | 4 | 16 | **0.070** | **3.44** |

**v4 相比 v3**：N=64 快 **18.6x**，N=512 快 **16.3x**；且实测在 `grid = 82k` 各档位都保持线性扩展，没有占用率台阶。
**v5 相比 v4**：主要收益来自 float4 向量化加载，N 越大收益越稳（N=64 快 4.6%，N=2048 快 2.1%）。

### 已知限制

- v4 的模板支持 `d = 32/64/128`，但注意：`d=128` 时 smem = 49536 B **超过 48KB 默认动态 smem 上限**，kernel 会启动失败并静默返回全 0（bench 里表现为 `MISMATCH`）。要用 d=128 必须调 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`，且此时每 SM 只能放 2 个 block。**d ≥ 128 目前没有可用的实现。**
- 所有版本都要求 `N` 能被 32 整除（v4 会直接报错，v1~v3 会静默越界）
- `TFLOPS` / `GB/s` 是基于理论最小量的估算值，不是 ncu 实测的硬件计数器，只适合版本间相对比较
- ncu 在当前环境无权限（`ERR_NVGPUCTRPERM`），占用率数据均由「smem/寄存器上限 + grid 阶梯扫描」推算
