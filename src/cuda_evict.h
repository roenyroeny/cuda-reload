// Minimal dynamically-loaded CUDA Driver API + the in-process driver eviction
// that "cuDeinit" should have been.
//
// No cuda.lib is linked. We bootstrap one symbol (cuGetProcAddress_v2) with the
// OS loader and resolve the rest through CUDA's own version-negotiating loader,
// so the negotiated _vN ABIs always match the installed driver.
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cuda.h>   // types/enums only

typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate)(CUcontext*, CUctxCreateParams*, unsigned, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *PFN_cuModuleLoad)(CUmodule*, const char*);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                               unsigned, unsigned, unsigned,
                                               unsigned, CUstream, void**, void**);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName)(CUresult, const char**);

// The slice of the Driver API this project needs.
struct CudaApi {
    HMODULE                 mod = nullptr;
    PFN_cuInit              Init = nullptr;
    PFN_cuDeviceGet         DeviceGet = nullptr;
    PFN_cuDeviceGetName     DeviceGetName = nullptr;
    PFN_cuCtxCreate         CtxCreate = nullptr;
    PFN_cuCtxDestroy        CtxDestroy = nullptr;
    PFN_cuCtxSynchronize    CtxSynchronize = nullptr;
    PFN_cuMemAlloc          MemAlloc = nullptr;
    PFN_cuMemFree           MemFree = nullptr;
    PFN_cuMemcpyHtoD        MemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH        MemcpyDtoH = nullptr;
    PFN_cuModuleLoad        ModuleLoad = nullptr;
    PFN_cuModuleGetFunction ModuleGetFunction = nullptr;
    PFN_cuLaunchKernel      LaunchKernel = nullptr;
    PFN_cuGetErrorName      GetErrorName = nullptr;
};

// LoadLibrary("nvcuda.dll"), resolve the table, and call cuInit(0). Returns
// false on any failure. Call it again after cuDeinit() to get a clean driver.
bool cudaLoad(CudaApi& api);

// cuGetErrorName wrapper; returns "?" if unavailable.
const char* cudaErr(const CudaApi& api, CUresult r);

// The teardown NVIDIA's driver doesn't provide.
//
// A sticky fault (CUDA_ERROR_ILLEGAL_ADDRESS) is latched in the per-process
// driver *instance* -- nvcuda64.dll's state plus the kernel-mode GPU context it
// owns -- not permanently in the hardware. The cure is a brand-new driver
// instance, which needs a fresh user-mode image of BOTH the nvcuda.dll shim and
// the real driver nvcuda64.dll. They are pinned (FreeLibrary can't unload them),
// so instead we make them invisible to the loader: rename each in the PEB (so
// name lookup misses) and unlink each from the loader's RB-tree indexes (so the
// section/mapping dedup misses). The next cudaLoad() then maps fresh copies and
// cuInit brings up a clean driver + clean context. The old instances are
// orphaned but stay resident until the process exits.
//
// Returns the number of driver modules evicted (normally 2).
int cuDeinit();
