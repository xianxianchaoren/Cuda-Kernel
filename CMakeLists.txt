cmake_minimum_required(VERSION 3.18)

project(CudaKernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# GPU 架构
# ---------------------------------------------------------------------------
# 不显式设置 CMAKE_CUDA_ARCHITECTURES 时，CMake 会在配置阶段自动探测
# 构建机（即服务器）上的 GPU 架构。探测不到时才按 toolkit 支持全集编译。
#
# 也可以手动指定，例如：
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=89        # RTX 40 系
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80        # A100
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=90        # H100
# ---------------------------------------------------------------------------

# 通用 CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# --use_fast_math 会牺牲精度换取性能，默认关闭；benchmark 时可打开对比：
#   cmake .. -DGEMM_USE_FAST_MATH=ON
option(GEMM_USE_FAST_MATH "Compile kernels with --use_fast_math" OFF)
if(GEMM_USE_FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()

# ---------------------------------------------------------------------------
# 框架静态库（与 kernel 无关，你不需要改）
# ---------------------------------------------------------------------------
add_library(gemm_framework STATIC
    src/reference.cpp
    src/data.cpp
    src/checker.cpp
    src/timer.cpp
)
target_include_directories(gemm_framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gemm_framework PUBLIC CUDA::cudart)

# ---------------------------------------------------------------------------
# benchmark 主程序
# 自动收集 kernels/ 下所有 *.cu，新增 kernel 文件后无需改这里
# （CONFIGURE_DEPENDS 会在构建时自动重新扫描目录）
# ---------------------------------------------------------------------------
file(GLOB GEMM_KERNEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu)

add_executable(gemm_bench
    src/main.cpp
    ${GEMM_KERNEL_SOURCES}
)
target_link_libraries(gemm_bench PRIVATE gemm_framework)
