# WMMA FP16 GEMM

> k1–k5 源码已合并到 [cuda-kernels](https://github.com/brookdach86-stack/cuda-kernels) 的 `矩阵乘法/wmma_gemm.cu`。本仓库保留独立 benchmark 数据。

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

warmup 10，计时 100 次，cudaEvent，默认流。1024³ 是在满血状态下测的，4096/8192³ 是连续测试时测的（笔记本降频，cuBLAS 从 32.9 TFLOPS 掉到 30.3/32.3，但比例稳定）。

### 1024³（L2 装得下，非带宽受限）

| Kernel | 延迟 (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 254.24 | 8.4 | 25.7% |
| k2 smem | 310.95 | 6.9 | 21.0% |
| k3 pad | 220.52 | 9.7 | 29.6% |
| k4 db | 159.14 | 13.5 | 41.0% |
| k5 rb | 88.16 | 24.4 | 74.0% |
| cuBLAS | 65.25 | 32.9 | 100% |

### 4096³（HBM 带宽受限）

| Kernel | 延迟 (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 22323 | 6.2 | 20.3% |
| k2 smem | 25400 | 5.4 | 17.8% |
| k3 pad | 22351 | 6.1 | 20.3% |
| k4 db | 12648 | 10.9 | 35.8% |
| k5 rb | 7059 | 19.5 | 64.2% |
| cuBLAS | 4534 | 30.3 | 100% |

### 8192³（HBM 带宽受限）

| Kernel | 延迟 (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 581278 | 1.9 | 5.8% |
| k2 smem | 233343 | 4.7 | 14.6% |
| k3 pad | 200644 | 5.5 | 16.9% |
| k4 db | 113983 | 9.6 | 29.8% |
| k5 rb | 58799 | 18.7 | 57.8% |
| cuBLAS | 33997 | 32.3 | 100% |

### 观察

1. **k1 在大尺寸下崩了**：8192³ 时 k1 只有 1.9 TFLOPS（5.8%），因为它从 global memory 直接读 fragment，HBM 带宽完全顶不住。1024³ 时 k1 还有 8.4 TFLOPS，因为 A+B 才 4 MB，L2 全装下了。
2. **双缓冲 k4 在大尺寸下才真正发力**：8192³ 时 k4 比 k3 快 1.76 倍（29.8% vs 16.9%），但 1024³ 时只快 1.38 倍。大尺寸下访存延迟高，双缓冲藏延迟的效果才明显。
3. **k5 的 TFLOPS 随尺寸增大反而略降**（19.3 → 19.5 → 18.7），说明 register blocking 的复用收益被访存开销抵消了一部分。
4. **cuBLAS 在大尺寸下稳定在 30-32 TFLOPS**，接近 RTX 5060 Laptop 的 FP16 Tensor Core 峰值，说明大尺寸下它能跑满算力。
5. **k5 vs cuBLAS 的比例随尺寸下降**（74% → 64% → 58%），因为 cuBLAS 的多级 pipeline 和 swizzle 在带宽受限时优势更大。

编译大尺寸：

```powershell
nvcc -o lab_4096.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=4096 -DGEMM_N=4096 -DGEMM_K=4096 -DSKIP_CPU=1
nvcc -o lab_8192.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=8192 -DGEMM_N=8192 -DGEMM_K=8192 -DSKIP_CPU=1
```

`SKIP_CPU=1` 跳过 CPU reference（8192³ 的 CPU 矩阵乘法要好几分钟），正确性改为 kernel 之间对比（1024³ 已验证过 vs CPU）。

## 正确性

每个 kernel 都对 CPU 参考（先把 FP16 转成 FP32，再做 IEEE FP32 点积）算平均绝对误差和最大绝对误差。k2–k5 再和 k1 比。

- vs CPU：avg 0.005432，max 0.008179，门槛 0.05
- vs k1：k2、k3、k4、k5、cuBLAS 的 max abs 都是 0

cuBLAS 和 k1 完全一致，说明转置调用算的是同一个乘积。cuBLAS 只按 vs CPU 判定；`max_vs_k1` 那一列是对照，不是它的门槛。

## 现在的上限

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
