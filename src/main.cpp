// Recover from a sticky CUDA fault IN-PROCESS -- no new process.
//
//   load -> fault (sticky CUDA_ERROR_ILLEGAL_ADDRESS) -> cuDeinit() -> reload
//   -> create a fresh context -> run a real kernel and verify the result.
//
// The recovery hinges on cuDeinit() (see cuda_evict.h): it evicts both CUDA
// driver DLLs from the loader so the reload brings up a brand-new, clean driver.
#include "cuda_evict.h"
#include <cstdio>
#include <vector>

static CUcontext makeContext(CudaApi& cu, const char* tag)
{
    CUdevice dev = 0;
    CUresult r = cu.DeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) { std::printf("[%s] cuDeviceGet -> %s\n", tag, cudaErr(cu, r)); return nullptr; }
    char name[256] = {0};
    cu.DeviceGetName(name, sizeof(name), dev);
    CUcontext ctx = nullptr;
    r = cu.CtxCreate(&ctx, nullptr, 0, dev);
    std::printf("[%s] %s ; cuCtxCreate -> %s\n", tag, name, cudaErr(cu, r));
    return r == CUDA_SUCCESS ? ctx : nullptr;
}

// Launch oob_write through a pointer 1 GiB past a real allocation. The in-kernel
// store dereferences memory the driver never validated -> hardware MMU fault ->
// sticky CUDA_ERROR_ILLEGAL_ADDRESS. (Validated memory ops like cuMemsetD8 are
// rejected host-side and can't fault the device, so a kernel is required.)
static void faultDevice(CudaApi& cu, CUcontext ctx, const char* ptx)
{
    CUmodule m = nullptr;
    CUresult r = cu.ModuleLoad(&m, ptx);
    if (r != CUDA_SUCCESS) { std::printf("cuModuleLoad(%s) -> %s\n", ptx, cudaErr(cu, r)); return; }
    CUfunction f = nullptr;
    cu.ModuleGetFunction(&f, m, "oob_write");
    CUdeviceptr buf = 0;
    cu.MemAlloc(&buf, 256 * sizeof(int));
    CUdeviceptr wild = buf + (CUdeviceptr)(1ull << 30);
    void* args[] = { &wild };
    cu.LaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
    std::printf("oob_write launched; cuCtxSynchronize -> %s\n", cudaErr(cu, cu.CtxSynchronize(ctx)));
}

// Prove the recovered context truly works -- not just that cuCtxCreate returned
// OK, or that a single launch slipped through. We give it a real workout:
//   - 256 alloc/free cycles (exercises the device allocator),
//   - 64 iterations of vec_add over 1M elements with fresh data each time,
//     every launch synced and every result element verified.
// Any latent stickiness would surface as a non-SUCCESS sync or a wrong value.
static bool validate(CudaApi& cu, CUcontext ctx, const char* ptx)
{
    CUmodule m = nullptr;
    if (cu.ModuleLoad(&m, ptx) != CUDA_SUCCESS) return false;
    CUfunction f = nullptr;
    if (cu.ModuleGetFunction(&f, m, "vec_add") != CUDA_SUCCESS) return false;

    // Allocator churn: many independent alloc/free cycles must all succeed.
    for (int i = 0; i < 256; ++i) {
        CUdeviceptr p = 0;
        CUresult r = cu.MemAlloc(&p, 1u << 20);   // 1 MiB
        if (r != CUDA_SUCCESS) { std::printf("alloc churn #%d -> %s\n", i, cudaErr(cu, r)); return false; }
        cu.MemFree(p);
    }
    std::printf("allocator churn: 256 alloc/free cycles -> OK\n");

    // Sustained compute: vec_add over 1M elements, 64 times, verifying each.
    const int N = 1 << 20;                        // 1M elements -> 4 MiB / buffer
    const size_t bytes = N * sizeof(float);
    std::vector<float> a(N), b(N), c(N);

    CUdeviceptr da = 0, db = 0, dc = 0;
    if (cu.MemAlloc(&da, bytes) != CUDA_SUCCESS) return false;
    cu.MemAlloc(&db, bytes); cu.MemAlloc(&dc, bytes);

    const int ITERS = 64;
    long long verified = 0;
    for (int it = 0; it < ITERS; ++it) {
        for (int i = 0; i < N; ++i) { a[i] = float(i + it); b[i] = float(2 * i - it); }
        if (cu.MemcpyHtoD(da, a.data(), bytes) != CUDA_SUCCESS) return false;
        cu.MemcpyHtoD(db, b.data(), bytes);

        int n = N;
        void* args[] = { &da, &db, &dc, &n };
        CUresult r = cu.LaunchKernel(f, (N + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
        if (r != CUDA_SUCCESS) { std::printf("iter %d launch -> %s\n", it, cudaErr(cu, r)); return false; }
        r = cu.CtxSynchronize(ctx);
        if (r != CUDA_SUCCESS) { std::printf("iter %d sync -> %s\n", it, cudaErr(cu, r)); return false; }
        if (cu.MemcpyDtoH(c.data(), dc, bytes) != CUDA_SUCCESS) return false;
        for (int i = 0; i < N; ++i)
            if (c[i] != a[i] + b[i]) { std::printf("iter %d mismatch @ %d\n", it, i); return false; }
        verified += N;
    }
    cu.MemFree(da); cu.MemFree(db); cu.MemFree(dc);
    std::printf("compute: %d x vec_add(%d) -> all CORRECT (%lld values verified)\n", ITERS, N, verified);
    return true;
}

int main(int argc, char** argv)
{
    const char* ptx = (argc > 1) ? argv[1] : "kernel.ptx";
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: survive a crash mid-run

    std::printf("== load ==\n");
    CudaApi cu;
    if (!cudaLoad(cu)) return 2;
    CUcontext ctx = makeContext(cu, "init");
    if (!ctx) return 3;

    std::printf("\n== fault ==\n");
    faultDevice(cu, ctx, ptx);
    CUdeviceptr probe = 0;
    CUresult pr = cu.MemAlloc(&probe, 256);
    std::printf("probe cuMemAlloc -> %s %s\n", cudaErr(cu, pr),
                pr == CUDA_SUCCESS ? "(not sticky?!)" : "(context poisoned -- sticky confirmed)");
    cu.CtxDestroy(ctx);

    std::printf("\n== cuDeinit (evict driver DLLs) ==\n");
    int evicted = cuDeinit();
    std::printf("evicted %d driver module(s)\n", evicted);

    std::printf("\n== reload + validate ==\n");
    CudaApi cu2;
    if (!cudaLoad(cu2)) return 1;
    std::printf("reloaded nvcuda.dll -> %p %s\n", (void*)cu2.mod,
                (cu2.mod == cu.mod) ? "(SAME base -- eviction failed)" : "(NEW base -- fresh driver)");
    CUcontext ctx2 = makeContext(cu2, "reload");
    bool ok = ctx2 && validate(cu2, ctx2, ptx);
    if (ctx2) cu2.CtxDestroy(ctx2);

    std::printf("\n== RESULT: %s ==\n",
                ok ? "RECOVERED in-process (fresh context runs kernels correctly)"
                   : "STILL BROKEN");
    return ok ? 0 : 1;
}
