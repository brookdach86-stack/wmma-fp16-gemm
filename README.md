![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM)-brightgreen)

---

# WMMA FP16 GEMM

> 代码已合并到 [cuda-kernels](https://github.com/brookdach86-stack/cuda-kernels) 的 `矩阵乘法/wmma_gemm.cu`，本仓库保留benchmark数据。

WMMA（Warp Matrix Multiply-Accumulate）是Tensor Core的CUDA API——每个warp的32个thread协同做16×16×16的matmul，比普通FP16快2-3x。

踩过的坑：

- WMMA fragment布局和NVIDIA文档写的不一样——K/N维度的stride搞反了，编译过了但结果是错的，花了一天才定位到是`n_frag_stride`和`k_frag_stride`搞反
- `wmma::fill_fragment`和`wmma::load_matrix_sync`的matrix_layout参数不能瞎猜——row_major和col_major搞反也会产生静默的错误结果，零warning
- accumulator fragment必须显式zero——我以为CUDA会自动清零，结果上一次计算的残留值混进来，测试数据里突然出现随机大值

这些坑直接导致后来FlashAttention的`tl.dot()`（底层就是WMMA）没有踩同样的布局错——因为我已经手动写过WMMA，知道fragment到底是怎么排的。

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, sm_120, 26 SMs, 32 MB L2
- CUDA 13.3
- Build: `build.bat` (`nvcc -arch=sm_120 -O3 -lcublas`)
- `lab.exe`验证失败时返回非零

## 计算内容

Row-major `C = A * B`。FP16输入、FP32累加、FP32写回。`srand(1)`初始化，元素值`(rand() % 200) / 100`，范围[0, 1.99]。

A+B总共4MB，C也是4MB——L2有32MB，放得下。1024³的测的不是大矩阵带宽受限的情况。

## 各版本说明

| Kernel | 做了什么 |
| --- | --- |
| k1 naive | 直接从global memory读fragment |
| k2 smem | Shared memory tile，没padding所以有bank conflict |
| k3 pad | `PAD=8`只加了行间距，没做swizzle |
| k4 db | Double buffering。当前tile做MMA和下一tile的4字节async copy重叠；MMA之后调`wait_prior(0)` |
| k5 rb | 每个warp算16×64，复用`fa`、`fc`有4份拷贝。还不是完整的128×128 register tile |
| cuBLAS | `cublasGemmEx`，`CUBLAS_COMPUTE_32F`，FP16 in / FP32 out |

cuBLAS是column-major的。Row-major的`C = A*B`在column-major下等价于`C^T = B^T * A^T`，所以调用维度传`(N, M, K)`，矩阵顺序B, A。

## 结果

Warmup 10次，计时100次，cudaEvent，default stream。1024³在全boost频率下测；4096/8192³连续测（笔记本供电限制cuBLAS从32.9 TFLOPS掉到30.3/32.3，但比值稳定）。

### 1024³（放L2里，不是带宽受限）

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 254.24 | 8.4 | 25.7% |
| k2 smem | 310.95 | 6.9 | 21.0% |
| k3 pad | 220.52 | 9.7 | 29.6% |
| k4 db | 159.14 | 13.5 | 41.0% |
| k5 rb | 88.16 | 24.4 | 74.0% |
| cuBLAS | 65.25 | 32.9 | 100% |

### 4096³（HBM带宽受限）

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 22323 | 6.2 | 20.3% |
| k2 smem | 25400 | 5.4 | 17.8% |
| k3 pad | 22351 | 6.1 | 20.3% |
| k4 db | 12648 | 10.9 | 35.8% |
| k5 rb | 7059 | 19.5 | 64.2% |
| cuBLAS | 4534 | 30.3 | 100% |

### 8192³（HBM带宽受限）

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 581278 | 1.9 | 5.8% |
| k2 smem | 233343 | 4.7 | 14.6% |
| k3 pad | 200644 | 5.5 | 16.9% |
| k4 db | 113983 | 9.6 | 29.8% |
| k5 rb | 58799 | 18.7 | 57.8% |
| cuBLAS | 33997 | 32.3 | 100% |

### 观察

1. **k1大矩阵直接崩**：8192³时k1只有1.9 TFLOPS（5.8%）——fragment直接从global memory读，HBM带宽跟不上。1024³时A+B才4MB放L2里，k1还能跑8.4 TFLOPS
2. **Double buffering k4只有大矩阵才出效果**：8192³时k4比k3快1.76倍（29.8% vs 16.9%），但1024³时只有1.38倍。大矩阵memory latency高，hide latency才有意义
3. **k5 TFLOPS随size略降**（19.3 → 19.5 → 18.7）——register blocking的复用收益被memory开销部分抵消
4. **cuBLAS大矩阵稳定在30-32 TFLOPS**——接近RTX 5060 Laptop FP16 Tensor Core峰值，说明它在大矩阵下已经榨满算力
5. **k5 vs cuBLAS比值随size下降**（74% → 64% → 58%）——带宽受限的时候，cuBLAS的多级pipeline和swizzling优势更大

大矩阵编译：

```powershell
nvcc -o lab_4096.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=4096 -DGEMM_N=4096 -DGEMM_K=4096 -DSKIP_CPU=1
nvcc -o lab_8192.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=8192 -DGEMM_N=8192 -DGEMM_K=8192 -DSKIP_CPU=1
```

`SKIP_CPU=1`跳过CPU参考实现（8192³ CPU matmul要跑几分钟）；正确性通过kernel间对比验证（1024³已经和CPU验证过）。

## 正确性验证

每个kernel和CPU参考（FP16→FP32转换后做IEEE FP32点积）比avg abs error和max abs error。k2–k5额外和k1比。

- vs CPU：avg 0.005432, max 0.008179, 阈值0.05
- vs k1：k2、k3、k4、k5、cuBLAS全部max abs = 0

cuBLAS和k1结果一模一样——确认转置调用算的是同一个积。cuBLAS只跟CPU比；`max_vs_k1`那列是参考用的不是它的阈值。

## 还有哪些没做

- Shared memory copy是4字节`__half2`，不是16字节
- 没有swizzle，只有`PAD=8`
- 只有两级buffering，而且当前MMA之后立刻wait
- k5只沿着N tile unroll了4个accumulator，tile还是64
- 没用到`mma.sync`
- 试过`BM/BN=128`，比64慢，这次没重测

## 额外踩过的坑

1. `#define N`和CUDA头文件冲突——改成`constexpr`
2. `__pipeline_memcpy_async` copy 2字节在sm_120上会deadlock——改成4字节`__half2`
3. `-arch=sm_100`导致`cublasCreate`返回1——这张卡是sm_120
4. 上面第2条触发了TDR GPU reset

## 文件

- `wmma_lab.cu`：k1–k5、cuBLAS、fragment layout验证和计时
- `Benchmark_Report_v1.md`：测量数据笔记
- `build.bat`
