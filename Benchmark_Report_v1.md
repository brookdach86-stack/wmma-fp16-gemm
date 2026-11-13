<!--
@Author: Wain (brookdach86-stack@users.noreply.github.com)
@GitHub: https://github.com/brookdach86-stack
@Note: 本内容为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
-->

# WMMA FP16 GEMM Benchmark Report

学习笔记里的一次完整测量。不是生产 GEMM，也不是峰值效率。

## 环境

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- CC: sm_120，26 SMs，L2 32.0 MB
- CUDA 13.3
- 编译: `nvcc -arch=sm_120 -O3 -lcublas`
- 矩阵: 1024³，行主序，FP16 输入 / FP32 累加 / FP32 写回
- 输入: `srand(1)`，`(rand() % 200) / 100.0f`，范围 [0, 1.99]
- 体积: A+B 4.00 MB，C 4.00 MB。三块加起来小于 L2，1024³ 看不出 HBM 瓶颈

## Tile

`BM=64`，`BN=64`，`BK=16`，`PAD=8`，fragment 16×16×16。

- k1: 每个 block 1 个 warp，直接 global load
- k2–k4: 16 个 warp / block，每个 warp 一个 16×16 输出
- k5: 4 个 warp / block。每个 warp 持有 4 个 accumulator，输出 16×64，`fa` 在 N 方向复用

k5 不是 128×128 那种 M、N 两边一起展开的 register tile。

## 各版本实际做了什么

- k1 naive: 没有 shared memory
- k2 smem: 把 tile 放进 shared memory，行宽等于 `BK` / `BN`，有 bank conflict
- k3 pad: 行宽加 8 个 half。这只是 padding，不是 XOR swizzle
- k4 db: 两块 buffer。当前 buffer 做 MMA 时，用 `__pipeline_memcpy_async` 把下一块以 `__half2`（4 字节）拷进来。`__pipeline_wait_prior(0)` 紧跟在这次 MMA 后面，重叠只在本轮内部
- k5 rb: 在 k4 的加载方式上，沿 N 把 accumulator 增到 4 个
- cuBLAS: `cublasGemmEx`，`CUBLAS_COMPUTE_32F`，`CUBLAS_GEMM_DEFAULT`

cuBLAS 按列主序。这里用的等价形式是 `C^T = B^T * A^T`：

```c
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K,
             &alpha,
             dB, CUDA_R_16F, N,
             dA, CUDA_R_16F, K,
             &beta,
             dC, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT);
```

`alpha` 和 `beta` 是 `float`。不要用 `cublasHgemm`：那是 FP16 输出，而且旧调用按列主序直接吃了行主序指针，1024³ 时算量碰巧相同，乘积不是同一个。

## 正确性

CPU 参考把同一批 FP16 先转成 FP32，再做 IEEE FP32 点积。门槛是 max abs ≤ 0.05，k2–k5 相对 k1 的 max abs ≤ 1e-3。

| Kernel | avg vs CPU | max vs CPU | max vs k1 | 结果 |
| --- | --- | --- | --- | --- |
| k1 | 0.005432 | 0.008179 | 0 | PASS |
| k2 | 0.005432 | 0.008179 | 0 | PASS |
| k3 | 0.005432 | 0.008179 | 0 | PASS |
| k4 | 0.005432 | 0.008179 | 0 | PASS |
| k5 | 0.005432 | 0.008179 | 0 | PASS |
| cuBLAS | 0.005432 | 0.008179 | 0 | PASS |

cuBLAS 的判定只用 vs CPU。它和 k1 的 max abs 也是 0，用来确认布局变换没有算成另一个乘积。

## 性能

warmup 10 + 计时 100 次，cudaEvent，默认流。一次运行：

| Kernel | us | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 254.24 | 8.4 | 25.7% |
| k2 smem | 310.95 | 6.9 | 21.0% |
| k3 pad | 220.52 | 9.7 | 29.6% |
| k4 db | 159.14 | 13.5 | 41.0% |
| k5 rb | 88.16 | 24.4 | 74.0% |
| cuBLAS | 65.25 | 32.9 | 100% |

另一次紧挨着的运行：k5 24.4 TFLOPS，cuBLAS 32.8。刚编译完的那次：k5 23.8，cuBLAS 30.5。下面的相对变化用上表，只保留到大约的百分比。

## 这组数说明什么

- k2 比 k1 慢大约 22%。1024³ 已经在 L2 里，多一次 shared memory 搬运，又没有去掉 bank conflict，是亏的
- k2 到 k3 大约 +41%。bank conflict 在这个 tile 上是实打实的
- k3 到 k4 大约 +39%。双缓冲有收益，但拷贝只有 4 字节，而且等待点很近
- k4 到 k5 大约 +81%。主要是每个 warp 少 load 了 3 次 `fa`，并且 block 从 16 个 warp 减到 4 个。不要把它写成完整的 register blocking
- k5 大约是这条同精度 cuBLAS 的 74%。绝对速度仍约 24 TFLOPS。旧笔记的 45.7%（对照 52.9 TFLOPS 的 `cublasHgemm`）作废

## 还没做

- 16 字节向量化拷贝
- shared memory swizzle
- 多于两级的 pipeline
- 4096³，以及任何明显超出 L2 的尺寸
- `mma.sync` 或这张卡上更新的 MMA 形状
- `BM/BN=128` 以前比 64 慢，这次没有重测

## 踩坑

1. `#define N` 和 CUDA 头冲突，改成 `constexpr`
2. `__pipeline_memcpy_async` 拷 2 字节，在 sm_120 上死锁，随后 TDR。改成 4 字节 `__half2`
3. `-arch=sm_100` 时 `cublasCreate` 返回 1。这张卡要 `sm_120`
