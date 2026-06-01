// Compiled to PTX (not linked into the host binary).
// The Driver API loads this module at runtime via cuModuleLoad.
//
// extern "C" prevents C++ name mangling so cuModuleGetFunction can find
// the function by the plain name "vec_add".
extern "C" __global__ void vec_add(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// Deliberately faults: the host passes a pointer that points far outside any
// valid allocation, so every thread performs an illegal global write. This
// raises a *sticky* CUDA_ERROR_ILLEGAL_ADDRESS that corrupts the context.
extern "C" __global__ void oob_write(int* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    p[i] = 42;
}
