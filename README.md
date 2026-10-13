# WMMA FP16 GEMM 笔记

在 RTX 5060 Laptop（sm_120）上，用 `nvcuda::wmma` 把 1024×1024×1024 的 FP16 GEMM 从 naive 做到沿 N 的 register blocking，并和同精度 cuBLAS 对比。

## 环境

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU，sm_120，26 SMs，L2 32 MB
- CUDA 13.3
- 编译运行: `build.bat`（`nvcc -arch=sm_120 -O3 -lcublas`）
- 校验失败时 `lab.exe` 返回非 0

## 算什么

行主序 `C = A * B`。FP16 输入，FP32 累加，FP32 写回。`srand(1)`，元素是 `(rand() % 200) / 100`，范围 [0, 1.99]。

A+B 4 MB，C 4 MB，小于 32 MB 的 L2。1024³ 不能代表大矩阵、受显存带宽限制时的性能。

## 版本

| Kernel | 做了什么 |
| --- | --- |
| k1 naive | 直接从 global memory 取 fragment |
| k2 smem | shared memory tile，没有 padding，有 bank conflict |
| k3 pad | `PAD=8`，只把行拉开，不是 swizzle |
| k4 db | 双缓冲。本轮 MMA 和下一 tile 的 4 字节异步拷贝重叠，MMA 之后就 `wait_prior(0)` |
| k5 rb | 每个 warp 算 16×64，复用 `fa`，`fc` 有 4 个。不是 128×128 的完整 register tile |
| cuBLAS | `cublasGemmEx`，`CUBLAS_COMPUTE_32F`，FP16 进 / FP32 出 |

cuBLAS 是列主序。行主序 `C = A*B` 写成列主序就是 `C^T = B^T * A^T`，所以调用维度是 `(N, M, K)`，矩阵顺序是 B、A。

## 结果

warmup 10，计时 100 次，cudaEvent，默认流。下表是一次运行。紧挨着的另一次几乎一样；刚编译完的那次 cuBLAS 掉到 30.5 TFLOPS，k5 掉到 23.8。笔记本频率会漂，74% 不要看成稳定小数。

| Kernel | 延迟 (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 254.24 | 8.4 | 25.7% |
| k2 smem | 310.95 | 6.9 | 21.0% |
| k3 pad | 220.52 | 9.7 | 29.6% |
| k4 db | 159.14 | 13.5 | 41.0% |
| k5 rb | 88.16 | 24.4 | 74.0% |
| cuBLAS | 65.25 | 32.9 | 100% |

k5 的绝对速度仍是大约 24 TFLOPS。比例比旧笔记里的 45.7% 高，是因为基线换了，不是 kernel 变快。旧的 52.9 TFLOPS 来自 `cublasHgemm`：FP16 输出，而且那个调用没有算同一个行主序乘积。那个比例作废。

## 正确性

每个 kernel 都对 CPU 参考（先把 FP16 转成 FP32，再做 IEEE FP32 点积）算平均绝对误差和最大绝对误差。k2–k5 再和 k1 比。

- vs CPU：avg 0.005432，max 0.008179，门槛 0.05
- vs k1：k2、k3、k4、k5、cuBLAS 的 max abs 都是 0

cuBLAS 和 k1 完全一致，说明转置调用算的是同一个乘积。cuBLAS 只按 vs CPU 判定；`max_vs_k1` 那一列是对照，不是它的门槛。

## 现在的上限

- 只测了 1024³
- shared memory 拷贝是 4 字节 `__half2`，不是 16 字节
- 没有 swizzle，只有 `PAD=8`
- 只有两级缓冲，而且本轮 MMA 结束后立刻等待
- k5 只沿 N 展开了 4 个 accumulator，tile 仍是 64
- 没有 `mma.sync`
- 之前试过 `BM/BN=128`，比 64 慢；这次没有重测

## 踩坑

1. `#define N` 和 CUDA 头冲突，改成 `constexpr`
2. `__pipeline_memcpy_async` 拷 2 字节，在 sm_120 上死锁。改成 `__half2` 的 4 字节
3. `-arch=sm_100` 会让 `cublasCreate` 返回 1。这张卡是 sm_120
4. 坑 2 触发过 TDR，GPU 被复位

## 文件

- `wmma_lab.cu`：k1–k5、cuBLAS、fragment 布局、校验和计时
- `Benchmark_Report_v1.md`：同一次测量的记录
- `build.bat`
