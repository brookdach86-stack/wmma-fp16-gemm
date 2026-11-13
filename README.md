![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)

---

<!--
@Author: Wain (brookdach86-stack@users.noreply.github.com)
@GitHub: https://github.com/brookdach86-stack
@Note: 本内容为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
-->

# WMMA FP16 GEMM

> k1–k5 source has been merged into [cuda-kernels](https://github.com/brookdach86-stack/cuda-kernels) at `矩阵乘法/wmma_gemm.cu`. This repo retains standalone benchmark data.

Implementing 1024×1024×1024 FP16 GEMM from naive to register-blocked-along-N using `nvcuda::wmma` on RTX 5060 Laptop (sm_120), compared against same-precision cuBLAS.

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, sm_120, 26 SMs, 32 MB L2
- CUDA 13.3
- Build: `build.bat` (`nvcc -arch=sm_120 -O3 -lcublas`)
- `lab.exe` returns non-zero on validation failure

## What It Computes

Row-major `C = A * B`. FP16 inputs, FP32 accumulation, FP32 writeback. `srand(1)`, elements are `(rand() % 200) / 100`, range [0, 1.99].

A+B total 4 MB, C 4 MB — fits in L2's 32 MB. 1024³ does not represent large-matrix, bandwidth-limited performance.

## Versions

| Kernel | What It Does |
| --- | --- |
| k1 naive | Direct fragment from global memory |
| k2 smem | Shared memory tile, no padding, has bank conflict |
| k3 pad | `PAD=8`, only row spacing, not swizzle |
| k4 db | Double buffering. Current tile MMA overlaps with next tile's 4-byte async copy; `wait_prior(0)` after MMA |
| k5 rb | Each warp computes 16×64, reuses `fa`, `fc` has 4 copies. Not a complete 128×128 register tile |
| cuBLAS | `cublasGemmEx`, `CUBLAS_COMPUTE_32F`, FP16 in / FP32 out |

cuBLAS is column-major. Row-major `C = A*B` in column-major becomes `C^T = B^T * A^T`, so call dimensions are `(N, M, K)` with matrix order B, A.

## Results

Warmup 10, timed 100 iterations, cudaEvent, default stream. 1024³ measured at full boost; 4096/8192³ measured back-to-back (laptop power-limits, cuBLAS drops from 32.9 TFLOPS to 30.3/32.3 but ratio stays stable).

### 1024³ (fits in L2, not bandwidth-limited)

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 254.24 | 8.4 | 25.7% |
| k2 smem | 310.95 | 6.9 | 21.0% |
| k3 pad | 220.52 | 9.7 | 29.6% |
| k4 db | 159.14 | 13.5 | 41.0% |
| k5 rb | 88.16 | 24.4 | 74.0% |
| cuBLAS | 65.25 | 32.9 | 100% |

### 4096³ (HBM bandwidth-limited)

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 22323 | 6.2 | 20.3% |
| k2 smem | 25400 | 5.4 | 17.8% |
| k3 pad | 22351 | 6.1 | 20.3% |
| k4 db | 12648 | 10.9 | 35.8% |
| k5 rb | 7059 | 19.5 | 64.2% |
| cuBLAS | 4534 | 30.3 | 100% |

### 8192³ (HBM bandwidth-limited)

| Kernel | Latency (us) | TFLOPS | vs cuBLAS |
| --- | --- | --- | --- |
| k1 naive | 581278 | 1.9 | 5.8% |
| k2 smem | 233343 | 4.7 | 14.6% |
| k3 pad | 200644 | 5.5 | 16.9% |
| k4 db | 113983 | 9.6 | 29.8% |
| k5 rb | 58799 | 18.7 | 57.8% |
| cuBLAS | 33997 | 32.3 | 100% |

### Observations

1. **k1 collapses at large sizes**: at 8192³ k1 only gets 1.9 TFLOPS (5.8%) — reading fragments directly from global memory, HBM bandwidth can't keep up. At 1024³ k1 still has 8.4 TFLOPS because A+B is only 4 MB, fits in L2.
2. **Double buffering k4 only shows real gains at large sizes**: at 8192³ k4 is 1.76× faster than k3 (29.8% vs 16.9%), but at 1024³ only 1.38×. At large sizes memory latency is high, hiding it matters.
3. **k5 TFLOPS slightly decreases with size** (19.3 → 19.5 → 18.7) — register blocking reuse gains are partially offset by memory overhead.
4. **cuBLAS stabilizes at 30-32 TFLOPS at large sizes**, near the RTX 5060 Laptop FP16 Tensor Core peak — showing it saturates compute at large sizes.
5. **k5 vs cuBLAS ratio drops with size** (74% → 64% → 58%) because cuBLAS's multi-level pipeline and swizzling give bigger advantage when bandwidth-limited.

Build for large sizes:

```powershell
nvcc -o lab_4096.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=4096 -DGEMM_N=4096 -DGEMM_K=4096 -DSKIP_CPU=1
nvcc -o lab_8192.exe wmma_lab.cu -arch=sm_120 -O3 -lcublas -DGEMM_M=8192 -DGEMM_N=8192 -DGEMM_K=8192 -DSKIP_CPU=1
```

`SKIP_CPU=1` skips CPU reference (8192³ CPU matmul takes minutes); correctness is verified by kernel-to-kernel comparison (1024³ has been validated vs CPU).

## Correctness

Each kernel is checked against CPU reference (FP16 → FP32 first, then IEEE FP32 dot product) for avg abs error and max abs error. k2–k5 additionally compared against k1.

- vs CPU: avg 0.005432, max 0.008179, threshold 0.05
- vs k1: k2, k3, k4, k5, cuBLAS all have max abs 0

cuBLAS and k1 are identical — confirms the transposed call computes the same product. cuBLAS judged only vs CPU; `max_vs_k1` column is reference, not its threshold.

## Current Ceiling

- Shared memory copies are 4-byte `__half2`, not 16-byte
- No swizzle, only `PAD=8`
- Only two-level buffering, and we wait immediately after current MMA
- k5 only unrolled 4 accumulators along N, tile still 64
- No `mma.sync`
- Tried `BM/BN=128` before, was slower than 64; haven't retested this time

## Bugs Hit Along the Way

1. `#define N` conflicts with CUDA headers; changed to `constexpr`
2. `__pipeline_memcpy_async` copying 2 bytes deadlocks on sm_120; changed to 4-byte `__half2`
3. `-arch=sm_100` causes `cublasCreate` to return 1; this card is sm_120
4. Bug #2 triggered TDR, GPU reset

## Files

- `wmma_lab.cu`: k1–k5, cuBLAS, fragment layout, validation and timing
- `Benchmark_Report_v1.md`: Same-measurement notes
- `build.bat`
