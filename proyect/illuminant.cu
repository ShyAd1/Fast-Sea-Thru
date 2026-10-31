/**
 * illuminant.cu  -  CUDA-accelerated illuminant map for Sea-Thru
 *
 * Replaces the BFS-based LSAC neighbourhood computation with a fixed
 * spatial window + depth-threshold approach that maps naturally to CUDA
 * (one thread per pixel).
 *
 * Changes vs. the previous version (robustness):
 *   - Context is never left half-initialised: on any failure all device
 *     buffers + the stream are released, so the next call starts clean
 *     (previously a failed cudaMalloc left capacityN > 0 with null pointers
 *     -> illegal memory access on the following image).
 *   - Every error path frees what it allocated (no leaks on failure).
 *   - Launch configuration is validated (block_x * block_y <= 1024, > 0).
 *   - Work is synchronised every few iterations, so one long queue of
 *     kernels cannot trip the Windows WDDM watchdog (TDR) on a display GPU.
 *   - Threads are mapped x -> column, y -> row, so neighbouring threads read
 *     neighbouring memory (coalesced loads; much faster than before).
 *   - The centre pixel is always part of its own neighbourhood, so NaN or
 *     degenerate depths can no longer produce a 0 illuminant.
 *   - Entry points are serialised with a mutex and never let a C++ exception
 *     cross the extern "C" boundary.
 *   - The legacy single-channel entry point now returns an error code and
 *     checks every CUDA call.
 *
 * Compile (Windows, CUDA 12.x/13.x, x64 Native Tools prompt):
 *   nvcc -O3 --shared -arch=native -o illuminant_cuda.dll illuminant.cu
 *
 * Compile (Linux):
 *   nvcc -O3 --shared -arch=native -Xcompiler -fPIC -o illuminant_cuda.so illuminant.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <mutex>
#include <new>
#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* Synchronise (and check for async errors) every this many iterations. */
#define SYNC_EVERY 8

/* ── error helpers ───────────────────────────────────────────────────────── */

static inline int check_cuda(cudaError_t err, const char *expr, const char *file, int line)
{
    if (err == cudaSuccess)
        return 0;
    fprintf(stderr, "[cuda] %s failed at %s:%d -> %s\n", expr, file, line, cudaGetErrorString(err));
    return -1;
}

#define CUDA_CHECK(expr)                                        \
    do                                                          \
    {                                                           \
        if (check_cuda((expr), #expr, __FILE__, __LINE__) != 0) \
            return -1;                                          \
    } while (0)

/* ── persistent context (reused between calls) ───────────────────────────── */

struct GpuIlluContext
{
    float *dDcRGB = nullptr;  /* N*3 */
    float *dDepths = nullptr; /* N   */
    float *dAc0RGB = nullptr; /* N*3 */
    float *dAc1RGB = nullptr; /* N*3 */
    size_t capacityN = 0;
    cudaStream_t stream = nullptr;
    bool streamReady = false;
};

static GpuIlluContext g_ctx;
static std::mutex g_mutex;

static void free_context_buffers()
{
    if (g_ctx.dDcRGB)
        cudaFree(g_ctx.dDcRGB);
    if (g_ctx.dDepths)
        cudaFree(g_ctx.dDepths);
    if (g_ctx.dAc0RGB)
        cudaFree(g_ctx.dAc0RGB);
    if (g_ctx.dAc1RGB)
        cudaFree(g_ctx.dAc1RGB);
    g_ctx.dDcRGB = nullptr;
    g_ctx.dDepths = nullptr;
    g_ctx.dAc0RGB = nullptr;
    g_ctx.dAc1RGB = nullptr;
    g_ctx.capacityN = 0;
}

static void reset_context()
{
    free_context_buffers();
    if (g_ctx.streamReady)
    {
        cudaStreamDestroy(g_ctx.stream);
        g_ctx.streamReady = false;
        g_ctx.stream = nullptr;
    }
    cudaGetLastError(); /* clear any sticky non-fatal error state */
}

static int ensure_context_capacity(size_t N)
{
    if (!g_ctx.streamReady)
    {
        int ndev = 0;
        if (check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount", __FILE__, __LINE__) != 0)
            return -1;
        if (ndev <= 0)
        {
            fprintf(stderr, "[cuda] no CUDA device available\n");
            return -1;
        }
        if (check_cuda(cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags", __FILE__, __LINE__) != 0)
            return -1;
        g_ctx.streamReady = true;
    }

    if (N <= g_ctx.capacityN && g_ctx.dDcRGB && g_ctx.dDepths && g_ctx.dAc0RGB && g_ctx.dAc1RGB)
        return 0;

    free_context_buffers();

    const size_t b1 = N * sizeof(float);
    const size_t b3 = N * 3 * sizeof(float);
    if (check_cuda(cudaMalloc(&g_ctx.dDcRGB, b3), "cudaMalloc(dDcRGB)", __FILE__, __LINE__) != 0 ||
        check_cuda(cudaMalloc(&g_ctx.dDepths, b1), "cudaMalloc(dDepths)", __FILE__, __LINE__) != 0 ||
        check_cuda(cudaMalloc(&g_ctx.dAc0RGB, b3), "cudaMalloc(dAc0RGB)", __FILE__, __LINE__) != 0 ||
        check_cuda(cudaMalloc(&g_ctx.dAc1RGB, b3), "cudaMalloc(dAc1RGB)", __FILE__, __LINE__) != 0)
    {
        free_context_buffers(); /* never keep a half-allocated context */
        return -1;
    }

    g_ctx.capacityN = N;
    return 0;
}

/* Tiny RAII wrapper for the one-shot legacy entry point. */
struct DevBuf
{
    float *p = nullptr;
    ~DevBuf()
    {
        if (p)
            cudaFree(p);
    }
    int alloc(size_t n)
    {
        return check_cuda(cudaMalloc(&p, n * sizeof(float)), "cudaMalloc(DevBuf)", __FILE__, __LINE__);
    }
};

/* ── kernels ─────────────────────────────────────────────────────────────── */

/**
 * One thread per pixel; x -> column, y -> row (coalesced access).
 * For pixel (row, col) we average ac_in over the (2*half_k+1)^2 window, keeping
 * only neighbours whose depth differs from the centre by less than `eps`.
 * Then:  ac_out = Dc*p + avg_neighbour*(1-p)
 */
__global__ void illuminant_step(
    const float *__restrict__ Dc,
    const float *__restrict__ depths,
    const float *__restrict__ ac_in,
    float *__restrict__ ac_out,
    int H, int W,
    float p, float eps, int half_k)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= W || row >= H)
        return;

    const size_t pix = (size_t)row * W + col;
    const float z_c = depths[pix];
    const int r0 = max(row - half_k, 0), r1 = min(row + half_k, H - 1);
    const int c0 = max(col - half_k, 0), c1 = min(col + half_k, W - 1);

    float sum = 0.0f;
    int cnt = 0;
    for (int r = r0; r <= r1; ++r)
    {
        const size_t rbase = (size_t)r * W;
        for (int c = c0; c <= c1; ++c)
        {
            const size_t npix = rbase + c;
            if (fabsf(depths[npix] - z_c) < eps)
            {
                sum += ac_in[npix];
                ++cnt;
            }
        }
    }
    if (cnt == 0) /* NaN depth / eps <= 0: the pixel is its own neighbourhood */
    {
        sum = ac_in[pix];
        cnt = 1;
    }

    ac_out[pix] = Dc[pix] * p + (sum / (float)cnt) * (1.0f - p);
}

/**
 * RGB-interleaved variant: Dc/ac are H*W*3 laid out as [pix0.r, pix0.g, pix0.b, pix1.r, ...].
 */
__global__ void illuminant_step_rgb(
    const float *__restrict__ Dc_rgb,
    const float *__restrict__ depths,
    const float *__restrict__ ac_in_rgb,
    float *__restrict__ ac_out_rgb,
    int H, int W,
    float p, float eps, int half_k)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= W || row >= H)
        return;

    const size_t pix = (size_t)row * W + col;
    const float z_c = depths[pix];
    const int r0 = max(row - half_k, 0), r1 = min(row + half_k, H - 1);
    const int c0 = max(col - half_k, 0), c1 = min(col + half_k, W - 1);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    int cnt = 0;
    for (int r = r0; r <= r1; ++r)
    {
        const size_t rbase = (size_t)r * W;
        for (int c = c0; c <= c1; ++c)
        {
            const size_t npix = rbase + c;
            if (fabsf(depths[npix] - z_c) < eps)
            {
                const size_t base = npix * 3;
                sum0 += ac_in_rgb[base + 0];
                sum1 += ac_in_rgb[base + 1];
                sum2 += ac_in_rgb[base + 2];
                ++cnt;
            }
        }
    }
    const size_t base = pix * 3;
    if (cnt == 0) /* NaN depth / eps <= 0: the pixel is its own neighbourhood */
    {
        sum0 = ac_in_rgb[base + 0];
        sum1 = ac_in_rgb[base + 1];
        sum2 = ac_in_rgb[base + 2];
        cnt = 1;
    }

    const float inv_cnt = 1.0f / (float)cnt;
    ac_out_rgb[base + 0] = Dc_rgb[base + 0] * p + (sum0 * inv_cnt) * (1.0f - p);
    ac_out_rgb[base + 1] = Dc_rgb[base + 1] * p + (sum1 * inv_cnt) * (1.0f - p);
    ac_out_rgb[base + 2] = Dc_rgb[base + 2] * p + (sum2 * inv_cnt) * (1.0f - p);
}

/* Scale buffer in-place by f */
__global__ void scale_kernel(float *__restrict__ buf, float f, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] *= f;
}

/* ── argument validation ─────────────────────────────────────────────────── */

static bool grid_too_big(int H, int W, int bx, int by)
{
    const long long gx = ((long long)W + bx - 1) / bx;
    const long long gy = ((long long)H + by - 1) / by;
    return gx > 2147483647LL || gy > 65535LL;
}

static int validate_args(double p, double f, double eps, int H, int W,
                         int iterations, int window_size, int block_x, int block_y)
{
    if (H <= 0 || W <= 0 || iterations <= 0 || window_size <= 0 ||
        block_x <= 0 || block_y <= 0)
    {
        fprintf(stderr, "[cuda] invalid arguments (H=%d W=%d iters=%d window=%d block=%dx%d)\n",
                H, W, iterations, window_size, block_x, block_y);
        return -1;
    }
    if ((long long)block_x * (long long)block_y > 1024)
    {
        fprintf(stderr, "[cuda] block %dx%d exceeds 1024 threads per block\n", block_x, block_y);
        return -1;
    }
    if (!std::isfinite(p) || !std::isfinite(f) || !std::isfinite(eps))
    {
        fprintf(stderr, "[cuda] non-finite p/f/eps\n");
        return -1;
    }
    if (grid_too_big(H, W, block_x, block_y))
    {
        fprintf(stderr, "[cuda] image too large for launch grid\n");
        return -1;
    }
    return 0;
}

/* ── RGB entry point (preferred) ─────────────────────────────────────────── */

static int run_illuminant_rgb(
    const double *Dc_rgb_d64, const double *depths_d64, double *illu_rgb_d64,
    double p, double f, double eps, int H, int W,
    int iterations, int window_size, int block_x, int block_y)
{
    if (!Dc_rgb_d64 || !depths_d64 || !illu_rgb_d64)
    {
        fprintf(stderr, "[cuda] null pointer argument\n");
        return -1;
    }
    if (validate_args(p, f, eps, H, W, iterations, window_size, block_x, block_y) != 0)
        return -1;

    const size_t N = (size_t)H * (size_t)W;
    const size_t N3 = N * 3;
    const int half = window_size / 2;

    if (ensure_context_capacity(N) != 0)
        return -1;

    std::vector<float> hDc(N3);
    std::vector<float> hDepths(N);
    for (size_t i = 0; i < N3; ++i)
        hDc[i] = (float)Dc_rgb_d64[i];
    for (size_t i = 0; i < N; ++i)
        hDepths[i] = (float)depths_d64[i];

    CUDA_CHECK(cudaMemcpyAsync(g_ctx.dDcRGB, hDc.data(), N3 * sizeof(float), cudaMemcpyHostToDevice, g_ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(g_ctx.dDepths, hDepths.data(), N * sizeof(float), cudaMemcpyHostToDevice, g_ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(g_ctx.dAc0RGB, 0, N3 * sizeof(float), g_ctx.stream));
    /* hDc/hDepths must stay alive until the async copies finished. */
    CUDA_CHECK(cudaStreamSynchronize(g_ctx.stream));

    dim3 block(block_x, block_y);
    dim3 grid((W + block_x - 1) / block_x, (H + block_y - 1) / block_y);

    float *src = g_ctx.dAc0RGB;
    float *dst = g_ctx.dAc1RGB;
    for (int k = 0; k < iterations; ++k)
    {
        illuminant_step_rgb<<<grid, block, 0, g_ctx.stream>>>(
            g_ctx.dDcRGB, g_ctx.dDepths, src, dst,
            H, W, (float)p, (float)eps, half);
        if (check_cuda(cudaGetLastError(), "illuminant_step_rgb<<<>>>", __FILE__, __LINE__) != 0)
            return -1;
        float *tmp = src;
        src = dst;
        dst = tmp;

        if ((k + 1) % SYNC_EVERY == 0)
            CUDA_CHECK(cudaStreamSynchronize(g_ctx.stream)); /* WDDM/TDR-friendly + early error detection */
    }

    {
        const int thr = 256;
        const size_t blk = (N3 + thr - 1) / thr;
        scale_kernel<<<(unsigned int)blk, thr, 0, g_ctx.stream>>>(src, (float)f, N3);
        if (check_cuda(cudaGetLastError(), "scale_kernel<<<>>>", __FILE__, __LINE__) != 0)
            return -1;
    }

    std::vector<float> hResult(N3);
    CUDA_CHECK(cudaMemcpyAsync(hResult.data(), src, N3 * sizeof(float), cudaMemcpyDeviceToHost, g_ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(g_ctx.stream));

    for (size_t i = 0; i < N3; ++i)
        illu_rgb_d64[i] = (double)hResult[i];

    return 0;
}

/* ── Public C interface ──────────────────────────────────────────────────── */

extern "C"
{

    /**
     * compute_illuminant_cuda_rgb
     *
     *   Dc_rgb_d64   direct signal, interleaved RGB          [H x W x 3] (host, double)
     *   depths_d64   depth map                               [H x W]
     *   illu_rgb_d64 output illuminant map (written here)    [H x W x 3]
     *   p            direct-signal weight (Sea-Thru uses 0.4)
     *   f            illuminant scale     (Sea-Thru uses 2.0)
     *   eps          depth tolerance      (Sea-Thru uses mean_depth / 50)
     *   window_size  neighbourhood side length (7 works well; even values behave like +1)
     *   block_x/y    thread block (x covers columns, y rows); block_x*block_y <= 1024
     *
     * Returns 0 on success, non-zero on error (device buffers are reset on error).
     */
    EXPORT int compute_illuminant_cuda_rgb(
        const double *Dc_rgb_d64,
        const double *depths_d64,
        double *illu_rgb_d64,
        double p, double f, double eps,
        int H, int W,
        int iterations, int window_size,
        int block_x, int block_y)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        int rc;
        try
        {
            rc = run_illuminant_rgb(Dc_rgb_d64, depths_d64, illu_rgb_d64, p, f, eps,
                                    H, W, iterations, window_size, block_x, block_y);
        }
        catch (const std::bad_alloc &)
        {
            fprintf(stderr, "[cuda] host out of memory in compute_illuminant_cuda_rgb\n");
            rc = -2;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "[cuda] exception in compute_illuminant_cuda_rgb: %s\n", e.what());
            rc = -3;
        }
        catch (...)
        {
            fprintf(stderr, "[cuda] unknown exception in compute_illuminant_cuda_rgb\n");
            rc = -3;
        }
        if (rc != 0)
            reset_context();
        return rc;
    }

    EXPORT void release_illuminant_cuda_buffers()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        reset_context();
    }

    /**
     * compute_illuminant_cuda  (legacy, one channel per call)
     *
     *   Dc_d64 / depths_d64 / illu_d64 : [H x W] host, double, row-major.
     * Returns 0 on success, non-zero on error (the old void signature is
     * ABI-compatible for callers that ignore the result).
     */
    EXPORT int compute_illuminant_cuda(
        const double *Dc_d64,
        const double *depths_d64,
        double *illu_d64,
        double p, double f, double eps,
        int H, int W,
        int iterations, int window_size)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        try
        {
            if (!Dc_d64 || !depths_d64 || !illu_d64)
            {
                fprintf(stderr, "[cuda] null pointer argument\n");
                return -1;
            }
            if (validate_args(p, f, eps, H, W, iterations, window_size, 16, 16) != 0)
                return -1;

            const size_t N = (size_t)H * (size_t)W;
            const int half = window_size / 2;

            std::vector<float> hDc(N), hDepths(N), hResult(N);
            for (size_t i = 0; i < N; ++i)
            {
                hDc[i] = (float)Dc_d64[i];
                hDepths[i] = (float)depths_d64[i];
            }

            DevBuf dDc, dDepths, dAc0, dAc1;
            if (dDc.alloc(N) || dDepths.alloc(N) || dAc0.alloc(N) || dAc1.alloc(N))
                return -1;

            CUDA_CHECK(cudaMemcpy(dDc.p, hDc.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dDepths.p, hDepths.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(dAc0.p, 0, N * sizeof(float)));

            dim3 block(16, 16);
            dim3 grid((W + 15) / 16, (H + 15) / 16);

            float *src = dAc0.p, *dst = dAc1.p;
            for (int k = 0; k < iterations; ++k)
            {
                illuminant_step<<<grid, block>>>(dDc.p, dDepths.p, src, dst,
                                                 H, W, (float)p, (float)eps, half);
                if (check_cuda(cudaGetLastError(), "illuminant_step<<<>>>", __FILE__, __LINE__) != 0)
                    return -1;
                float *tmp = src;
                src = dst;
                dst = tmp;
                if ((k + 1) % SYNC_EVERY == 0)
                    CUDA_CHECK(cudaDeviceSynchronize());
            }

            {
                const int thr = 256;
                const size_t blk = (N + thr - 1) / thr;
                scale_kernel<<<(unsigned int)blk, thr>>>(src, (float)f, N);
                if (check_cuda(cudaGetLastError(), "scale_kernel<<<>>>", __FILE__, __LINE__) != 0)
                    return -1;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hResult.data(), src, N * sizeof(float), cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < N; ++i)
                illu_d64[i] = (double)hResult[i];
            return 0;
        }
        catch (...)
        {
            fprintf(stderr, "[cuda] exception in compute_illuminant_cuda\n");
            return -3;
        }
    }

} /* extern "C" */
