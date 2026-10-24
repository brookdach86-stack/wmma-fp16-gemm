#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define CHECK_CUDA(expr)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t _st = (expr);                                                            \
        if (_st != cudaSuccess)                                                              \
        {                                                                                    \
            fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(_st)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

constexpr int WARMUP = 10;
constexpr int REPS = 100;
// 固定种子下实测 max|gpu-cpu| = 0.008。0.05 留余量，同时能抓住算错的结果。
constexpr float CPU_MAX_ABS_LIMIT = 0.05f;
constexpr float VS_K1_MAX_ABS_LIMIT = 1.0e-3f;

#ifndef GEMM_M
#define GEMM_M 1024
#endif
#ifndef GEMM_N
#define GEMM_N 1024
#endif
#ifndef GEMM_K
#define GEMM_K 1024
#endif
constexpr int M = GEMM_M;
constexpr int N = GEMM_N;
constexpr int K = GEMM_K;
// 大尺寸（4096/8192）下 CPU reference 太慢，跳过 CPU 校验，
// 只对比 kernel 之间一致性（1024^3 已验证过正确性）
#ifndef SKIP_CPU
#define SKIP_CPU 0
#endif
constexpr int PAD = 8;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int BM2 = 64;
constexpr int BN2 = 64;

__global__ void k1_naive(const half *A, const half *B, float *C)
{
    int wx = blockIdx.x; // 每个block只有1个warp，所以blockIdx.x就是warp编号
    int wy = blockIdx.y;
    int tx = wx, ty = wy;
    if (tx * 16 >= N)
        return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    for (int k = 0; k < K; k += 16)
    {
        wmma::load_matrix_sync(fa, A + ty * 16 * K + k, K);
        wmma::load_matrix_sync(fb, B + k * N + tx * 16, N);
        wmma::mma_sync(fc, fa, fb, fc);
    }
    wmma::store_matrix_sync(C + ty * 16 * N + tx * 16, fc, N, wmma::mem_row_major);
}

__global__ void k2_smem(const half *A, const half *B, float *C)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_cols = BN / 16;
    int wr = warp_id / warp_cols, wc = warp_id % warp_cols;
    int br = blockIdx.y * BM, bc = blockIdx.x * BN;

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        for (int i = tid; i < BM * BK; i += blockDim.x)
            As[i] = A[(br + i / BK) * K + k0 + i % BK];
        for (int i = tid; i < BK * BN; i += blockDim.x)
            Bs[i] = B[(k0 + i / BN) * N + bc + i % BN];
        __syncthreads();

        wmma::load_matrix_sync(fa, As + wr * 16 * BK, BK);
        wmma::load_matrix_sync(fb, Bs + wc * 16, BN);
        wmma::mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + (br + wr * 16) * N + bc + wc * 16, fc, N, wmma::mem_row_major);
}

__global__ void k3_pad(const half *A, const half *B, float *C)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_cols = BN / 16;
    int wr = warp_id / warp_cols, wc = warp_id % warp_cols;
    int br = blockIdx.y * BM, bc = blockIdx.x * BN;

    __shared__ half As[BM * (BK + PAD)];
    __shared__ half Bs[BK * (BN + PAD)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        for (int i = tid; i < BM * BK; i += blockDim.x)
            As[(i / BK) * (BK + PAD) + i % BK] = A[(br + i / BK) * K + k0 + i % BK];
        for (int i = tid; i < BK * BN; i += blockDim.x)
            Bs[(i / BN) * (BN + PAD) + i % BN] = B[(k0 + i / BN) * N + bc + i % BN];
        __syncthreads();

        wmma::load_matrix_sync(fa, As + wr * 16 * (BK + PAD), BK + PAD);
        wmma::load_matrix_sync(fb, Bs + wc * 16, BN + PAD);
        wmma::mma_sync(fc, fa, fb, fc);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + (br + wr * 16) * N + bc + wc * 16, fc, N, wmma::mem_row_major);
}

//==================== Double Buffering ====================
// 两块 shared memory ping-pong：本轮 MMA 读 curr，同时把 next tile 异步拷进来。
// __pipeline_wait_prior(0) 紧跟在本轮 MMA 后面，重叠只发生在这一轮内部。
// 拷贝是 4 字节 __half2，不是 16 字节。

__global__ void k4_db(const half *A, const half *B, float *C)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_cols = BN / 16;
    int wr = warp_id / warp_cols, wc = warp_id % warp_cols;
    int br = blockIdx.y * BM, bc = blockIdx.x * BN;

    // 两套 shared memory：buf[0] 和 buf[1] 交替使用
    // 用 __half2 确保 pipeline memcpy 的 4 字节对齐要求
    __shared__ half As[2][BM * (BK + PAD)];
    __shared__ half Bs[2][BK * (BN + PAD)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    // 预加载 tile 0 到 buf[0]（普通赋值，无对齐问题）
    for (int i = tid; i < BM * BK; i += blockDim.x)
    {
        int r = i / BK, c = i % BK;
        As[0][r * (BK + PAD) + c] = A[(br + r) * K + 0 + c];
    }
    for (int i = tid; i < BK * BN; i += blockDim.x)
    {
        int r = i / BN, c = i % BN;
        Bs[0][r * (BN + PAD) + c] = B[(0 + r) * N + bc + c];
    }
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        int curr = (k0 / BK) % 2; // 当前计算用的 buffer
        int next = 1 - curr;      // 后台加载用的 buffer

        // --- Phase 1: 异步加载下一个 tile 到 next buffer ---
        // 一次拷 2 个 half (= __half2 = 4 字节)，保证 pipeline 4 字节对齐
        if (k0 + BK < K)
        {
            for (int i = tid; i < BM * BK / 2; i += blockDim.x)
            {
                int elem = i * 2;
                int r = elem / BK, c = elem % BK; // c 总是偶数
                __pipeline_memcpy_async(&As[next][r * (BK + PAD) + c],
                                        &A[(br + r) * K + k0 + BK + c],
                                        sizeof(__half2));
            }
            for (int i = tid; i < BK * BN / 2; i += blockDim.x)
            {
                int elem = i * 2;
                int r = elem / BN, c = elem % BN; // c 总是偶数
                __pipeline_memcpy_async(&Bs[next][r * (BN + PAD) + c],
                                        &B[(k0 + BK + r) * N + bc + c],
                                        sizeof(__half2));
            }
            __pipeline_commit();
        }

        // --- Phase 2: 用 curr buffer 做 mma 计算 ---
        wmma::load_matrix_sync(fa, As[curr] + wr * 16 * (BK + PAD), BK + PAD);
        wmma::load_matrix_sync(fb, Bs[curr] + wc * 16, BN + PAD);
        wmma::mma_sync(fc, fa, fb, fc);

        // --- Phase 3: 等后台加载完成，准备下一轮 ---
        if (k0 + BK < K)
        {
            __pipeline_wait_prior(0); // 等 pipeline batch 完成
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(C + (br + wr * 16) * N + bc + wc * 16, fc, N, wmma::mem_row_major);
}

//==================== 双缓冲 + 沿 N 的 register blocking ====================
// 每个 warp 算 16x64：fa 只 load 一次，fc 有 BN/16 = 4 个。
// 不是 128x128 那种在 M、N 两边都展开的 register tile。
__global__ void k5_rb(const half *A, const half *B, float *C)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wr = warp_id;
    int br = blockIdx.y * BM2, bc = blockIdx.x * BN2;

    __shared__ half As[2][BM2 * (BK + PAD)];
    __shared__ half Bs[2][BK * (BN2 + PAD)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[BN2 / 16];

#pragma unroll
    for (int j = 0; j < BN2 / 16; j++)
        wmma::fill_fragment(fc[j], 0.0f);

    for (int i = tid; i < BM2 * BK; i += blockDim.x)
    {
        int r = i / BK, c = i % BK;
        As[0][r * (BK + PAD) + c] = A[(br + r) * K + 0 + c];
    }
    for (int i = tid; i < BK * BN2; i += blockDim.x)
    {
        int r = i / BN2, c = i % BN2;
        Bs[0][r * (BN2 + PAD) + c] = B[(0 + r) * N + bc + c];
    }
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        int curr = (k0 / BK) % 2;
        int next = 1 - curr;

        if (k0 + BK < K)
        {
            for (int i = tid; i < BM2 * BK / 2; i += blockDim.x)
            {
                int elem = i * 2;
                int r = elem / BK, c = elem % BK;
                __pipeline_memcpy_async(&As[next][r * (BK + PAD) + c],
                                        &A[(br + r) * K + k0 + BK + c],
                                        sizeof(__half2));
            }
            for (int i = tid; i < BK * BN2 / 2; i += blockDim.x)
            {
                int elem = i * 2;
                int r = elem / BN2, c = elem % BN2;
                __pipeline_memcpy_async(&Bs[next][r * (BN2 + PAD) + c],
                                        &B[(k0 + BK + r) * N + bc + c],
                                        sizeof(__half2));
            }
            __pipeline_commit();
        }

        wmma::load_matrix_sync(fa, As[curr] + wr * 16 * (BK + PAD), BK + PAD);
#pragma unroll
        for (int j = 0; j < BN2 / 16; j++)
        {
            wmma::load_matrix_sync(fb, Bs[curr] + j * 16, BN2 + PAD);
            wmma::mma_sync(fc[j], fa, fb, fc[j]);
        }

        if (k0 + BK < K)
            __pipeline_wait_prior(0);
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < BN2 / 16; j++)
        wmma::store_matrix_sync(C + (br + wr * 16) * N + bc + j * 16, fc[j], N, wmma::mem_row_major);
}

//==================== Fragment 数据分布探索 ====================
// 每个 lane 把自己的 8 个 half 写到 global memory 的对应位置
__global__ void k0_explore_fragment(const half *A, float *out)
{
    int laneId = threadIdx.x % 32;
    if (blockIdx.x != 0 || threadIdx.x >= 32)
        return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, A, 16);

    const half *ptr = reinterpret_cast<const half *>(&fa);
    for (int i = 0; i < 8; i++) // 每个线程固定8个元素，不能用fa.num_elements(256)
        out[laneId * 8 + i] = __half2float(ptr[i]);
}

void run_fragment_explore(const half *dA)
{
    float *dOut, *hOut = new float[32 * 8];
    cudaMalloc(&dOut, 32 * 8 * sizeof(float));

    printf("\n==== Fragment Data Distribution (matrix_a, row_major, 16x16) ====\n");
    printf("Rule: 32 lanes form a 4-row x 8-col grid\n");
    printf("  grid_row gr = lane/8 ,  grid_col gc = lane%%8\n");
    printf("  Each lane holds 2 rows x 4 cols = 8 half values\n\n");
    printf("  Row mapping:  gc<4 ? rows(gr*2, gr*2+8) : rows(gr*2+1, gr*2+9)\n");
    printf("  Col mapping:  c = gc%%4,  col set = {c*2, c*2+1, c*2+8, c*2+9}\n");
    printf("  8-value order: [r1][c*2], [r1][c*2+1], [r2][c*2], [r2][c*2+1],\n");
    printf("                 [r1][c*2+8], [r1][c*2+9], [r2][c*2+8], [r2][c*2+9]\n\n");

    k0_explore_fragment<<<1, 32>>>(dA, dOut);
    cudaMemcpy(hOut, dOut, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("lane  grid_pos    holds          8 values\n");
    printf("--------------------------------------------\n");
    for (int l = 0; l < 32; l++)
    {
        int gr = l / 8; // grid row 0~3
        int gc = l % 8; // grid col 0~7
        int r1 = gr * 2 + (gc < 4 ? 0 : 1);
        int r2 = r1 + 8;
        int c = gc % 4;
        int cols[4] = {c * 2, c * 2 + 1, c * 2 + 8, c * 2 + 9};
        printf("lane%2d  (gr=%d,gc=%d)  rows(%d,%d) cols", l, gr, gc, r1, r2);
        printf("%d,%d,%d,%d: ", cols[0], cols[1], cols[2], cols[3]);
        for (int i = 0; i < 8; i++)
            printf("%5.2f ", hOut[l * 8 + i]);
        printf("\n");
    }
    printf("\n");

    cudaFree(dOut);
    delete[] hOut;
}

//==================== 验证 & 计时 & 主程序 ====================
// 正确性有两层：
//   vs CPU：同一批 FP16 输入先转 FP32，再按 IEEE FP32 做点积
//   vs k1 ：k2~k5 必须复现 naive WMMA 的结果，用来抓住 padding / pipeline / blocking 写错
// cuBLAS 走列主序。行主序 C = A*B 等价于列主序 C^T = B^T * A^T，
// 所以 GemmEx 的维度是 (N, M, K)，矩阵参数顺序是 B, A，ldc = N。

struct ErrStats
{
    double avg;
    float max_abs;
};

const char *cublas_status_str(cublasStatus_t st)
{
    switch (st)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "ARCH_MISMATCH";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "EXECUTION_FAILED";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "NOT_SUPPORTED";
    default:
        return "OTHER";
    }
}

void cpu_ref(const half *A, const half *B, float *C)
{
    float *Af = new float[M * K];
    float *Bf = new float[K * N];
    for (int i = 0; i < M * K; i++)
        Af[i] = __half2float(A[i]);
    for (int i = 0; i < K * N; i++)
        Bf[i] = __half2float(B[i]);
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            float a = Af[i * K + k];
            const float *brow = Bf + k * N;
            float *crow = C + i * N;
            for (int j = 0; j < N; j++)
                crow[j] += a * brow[j];
        }
    }
    delete[] Af;
    delete[] Bf;
}

ErrStats err_stats(const float *got, const float *ref, int n)
{
    double sum = 0.0;
    float mx = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float d = fabsf(got[i] - ref[i]);
        sum += d;
        if (d > mx)
            mx = d;
    }
    ErrStats s;
    s.avg = sum / n;
    s.max_abs = mx;
    return s;
}

void run_k1(const half *a, const half *b, float *c) { k1_naive<<<dim3(N / 16, M / 16), 32>>>(a, b, c); }
void run_k2(const half *a, const half *b, float *c) { k2_smem<<<dim3(N / BN, M / BM), (BM / 16) * (BN / 16) * 32>>>(a, b, c); }
void run_k3(const half *a, const half *b, float *c) { k3_pad<<<dim3(N / BN, M / BM), (BM / 16) * (BN / 16) * 32>>>(a, b, c); }
void run_k4(const half *a, const half *b, float *c) { k4_db<<<dim3(N / BN, M / BM), (BM / 16) * (BN / 16) * 32>>>(a, b, c); }
void run_k5(const half *a, const half *b, float *c) { k5_rb<<<dim3(N / BN2, M / BM2), (BM2 / 16) * 32>>>(a, b, c); }

// alpha/beta 放在静态存储里。默认 host pointer mode 下 cuBLAS 会在调用时读取它们；
// 用静态地址避免异步 GEMM 还在跑时栈上的标量已经失效。
static const float kAlpha = 1.0f;
static const float kBeta = 0.0f;

void gemm_cublas(cublasHandle_t handle, const half *dA, const half *dB, float *dC)
{
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &kAlpha,
        dB, CUDA_R_16F, N,
        dA, CUDA_R_16F, K,
        &kBeta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasGemmEx failed: %s (%d)\n", cublas_status_str(st), (int)st);
        exit(1);
    }
}

float bench(void (*launch)(const half *, const half *, float *),
            const half *dA, const half *dB, float *dC)
{
    for (int i = 0; i < WARMUP; i++)
        launch(dA, dB, dC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < REPS; i++)
        launch(dA, dB, dC);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / REPS;
}

float bench_cublas(cublasHandle_t handle, const half *dA, const half *dB, float *dC)
{
    for (int i = 0; i < WARMUP; i++)
        gemm_cublas(handle, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < REPS; i++)
        gemm_cublas(handle, dA, dB, dC);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / REPS;
}

bool check_kernel(const char *name,
                  void (*launch)(const half *, const half *, float *),
                  const half *dA, const half *dB, float *dC,
                  float *hC, const float *hCpu, const float *hK1,
                  bool is_k1)
{
    CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(float)));
    launch(dA, dB, dC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

#if SKIP_CPU
    ErrStats vs_cpu = {0.0f, 0.0f};
#else
    ErrStats vs_cpu = err_stats(hC, hCpu, M * N);
#endif
    float vs_k1 = is_k1 ? 0.0f : err_stats(hC, hK1, M * N).max_abs;
    bool pass =
#if SKIP_CPU
        vs_k1 <= VS_K1_MAX_ABS_LIMIT;
#else
        vs_cpu.max_abs <= CPU_MAX_ABS_LIMIT && vs_k1 <= VS_K1_MAX_ABS_LIMIT;
#endif
    printf("%-8s %12.6f %12.6f %12.6f   %s\n",
           name, vs_cpu.avg, vs_cpu.max_abs, vs_k1, pass ? "PASS" : "FAIL");
    return pass;
}

int main()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    const double bytes_ab = (double)(M * K + K * N) * sizeof(half);
    const double bytes_c = (double)(M * N) * sizeof(float);
    const double mib = 1024.0 * 1024.0;
    printf("GPU: %s\n", prop.name);
    printf("CC: sm_%d%d  SMs: %d  L2: %.1f MB\n",
           prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize / mib);
    printf("Problem: %d x %d x %d  row-major  FP16 in / FP32 acc / FP32 out\n", M, N, K);
    printf("Footprint: A+B %.2f MB, C %.2f MB\n", bytes_ab / mib, bytes_c / mib);
    if (bytes_ab + bytes_c < (double)prop.l2CacheSize)
        printf("A+B+C fits in L2. 1024^3 is not an HBM-bound large-GEMM result.\n");
    else
        printf("A+B+C exceeds L2. This is an HBM-bound large-GEMM result.\n");
    printf("Timing: warmup %d + reps %d, cudaEvent, default stream\n", WARMUP, REPS);
#if SKIP_CPU
    printf("SKIP_CPU=1: skipping CPU reference, correctness checked kernel-vs-k1 only\n");
#endif

    srand(1);
    half *hA = new half[M * K];
    half *hB = new half[K * N];
    float *hC = new float[M * N];
    float *hK1 = new float[M * N];
    float *hCpu = new float[M * N];
    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    for (int i = 0; i < M * K; i++)
        hA[i] = __float2half((rand() % 200) / 100.0f);
    for (int i = 0; i < K * N; i++)
        hB[i] = __float2half((rand() % 200) / 100.0f);
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice));

#if !SKIP_CPU
    half *hA_test = new half[256];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            hA_test[i * 16 + j] = __float2half((float)(i * 16 + j));
    half *dA_test = nullptr;
    CHECK_CUDA(cudaMalloc(&dA_test, 256 * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(dA_test, hA_test, 256 * sizeof(half), cudaMemcpyHostToDevice));

    printf("\n==== Matrix A (labeled 0~255) ====\n");
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
            printf("%3.0f ", __half2float(hA_test[i * 16 + j]));
        printf("\n");
    }
    run_fragment_explore(dA_test);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(dA_test));
    delete[] hA_test;

    printf("CPU reference...\n");
    cpu_ref(hA, hB, hCpu);
#endif

    cublasHandle_t handle;
    cublasStatus_t cst = cublasCreate(&handle);
    if (cst != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasCreate failed: %s (%d)\n", cublas_status_str(cst), (int)cst);
        return 1;
    }

    printf("\n==== Correctness ====\n");
#if SKIP_CPU
    printf("SKIP_CPU: only checking max|kernel-k1| <= %.1e (CPU ref skipped)\n", VS_K1_MAX_ABS_LIMIT);
#else
    printf("limits: max|gpu-cpu| <= %.2f,  max|kernel-k1| <= %.1e\n",
           CPU_MAX_ABS_LIMIT, VS_K1_MAX_ABS_LIMIT);
    printf("cuBLAS is gated on max_vs_cpu only; its max_vs_k1 column is informational\n");
#endif
    printf("%-8s %12s %12s %12s   %s\n", "kernel", "avg_vs_cpu", "max_vs_cpu", "max_vs_k1", "status");

    bool ok = true;
    ok = check_kernel("k1", run_k1, dA, dB, dC, hC, hCpu, hK1, true) && ok;
    memcpy(hK1, hC, M * N * sizeof(float));
    ok = check_kernel("k2", run_k2, dA, dB, dC, hC, hCpu, hK1, false) && ok;
    ok = check_kernel("k3", run_k3, dA, dB, dC, hC, hCpu, hK1, false) && ok;
    ok = check_kernel("k4", run_k4, dA, dB, dC, hC, hCpu, hK1, false) && ok;
    ok = check_kernel("k5", run_k5, dA, dB, dC, hC, hCpu, hK1, false) && ok;

    CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(float)));
    gemm_cublas(handle, dA, dB, dC);
    CHECK_CUDA(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
#if SKIP_CPU
    ErrStats cublas_cpu = {0.0f, 0.0f};
    bool cublas_pass = true;
#else
    ErrStats cublas_cpu = err_stats(hC, hCpu, M * N);
    bool cublas_pass = cublas_cpu.max_abs <= CPU_MAX_ABS_LIMIT;
#endif
    float cublas_k1 = err_stats(hC, hK1, M * N).max_abs;
    printf("%-8s %12.6f %12.6f %12.6f   %s\n",
           "cuBLAS", cublas_cpu.avg, cublas_cpu.max_abs, cublas_k1, cublas_pass ? "PASS" : "FAIL");
    ok = cublas_pass && ok;
    printf("correctness: %s\n", ok ? "PASS" : "FAIL");

    float ms1 = bench(run_k1, dA, dB, dC);
    float ms2 = bench(run_k2, dA, dB, dC);
    float ms3 = bench(run_k3, dA, dB, dC);
    float ms4 = bench(run_k4, dA, dB, dC);
    float ms5 = bench(run_k5, dA, dB, dC);
    float msc = bench_cublas(handle, dA, dB, dC);
    double flops = 2.0 * M * N * K;
    auto report = [&](const char *name, float ms)
    {
        double tflops = flops / ms / 1e9;
        double pct = msc > 0.0f ? (double)msc / ms * 100.0 : 0.0;
        printf("%-8s %8.2f us  %6.1f TFLOPS  %5.1f%% of cuBLAS\n", name, ms * 1000.0, tflops, pct);
    };
    printf("\n==== Performance ====\n");
    report("k1", ms1);
    report("k2", ms2);
    report("k3", ms3);
    report("k4", ms4);
    report("k5", ms5);
    report("cuBLAS", msc);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hK1;
    delete[] hCpu;
    return ok ? 0 : 1;
}