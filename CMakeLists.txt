cmake_minimum_required(VERSION 3.18)
project(cuda_recover LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emit the executable (and the kernel.ptx copied next to it) into <root>/bin.
# Multi-config generators (Visual Studio) otherwise append a per-config subdir
# (bin/Release, bin/Debug), so override the per-config variables too to keep
# every config in the same bin/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

# We only need the CUDA headers for cu* types/enums. We do NOT link cuda.lib
# and do NOT enable the CUDA language: nvcuda.dll is loaded dynamically and
# every entry point is resolved through cuGetProcAddress at runtime.
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# In-process sticky-fault recovery demo.
#   load -> fault (oob_write kernel) -> cuDeinit() (evict driver DLLs) ->
#   reload -> create a fresh context -> validate with vec_add.
# ---------------------------------------------------------------------------
add_executable(cuda_recover src/main.cpp src/cuda_evict.cpp)
target_include_directories(cuda_recover PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Compile the kernels (oob_write fault + vec_add validation) to PTX with nvcc
# directly (no CUDA-language / VS integration needed). compute_75 PTX is
# forward-compatible: the driver JITs it for newer GPUs (Blackwell) at
# cuModuleLoad time. A kernel is required for the fault: validated memory ops
# (cuMemsetD8 etc.) reject bad pointers host-side and never reach the GPU, so
# only actual GPU code can raise a sticky CUDA_ERROR_ILLEGAL_ADDRESS.
# nvcc needs MSVC's cl.exe as host compiler even for -ptx; it isn't on PATH in
# the MSBuild step, so point -ccbin at the cl.exe CMake already found.
# This VS 2026 / CUDA 13.1 combo is newer than nvcc and its STL guards expect,
# so we override both version checks (the kernels are trivial device code):
#   -allow-unsupported-compiler            : nvcc's MSVC-version gate
#   /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH : MSVC STL's CUDA-version gate
add_custom_command(TARGET cuda_recover POST_BUILD
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -ptx -arch=compute_75
            -ccbin "${CMAKE_CXX_COMPILER}" -allow-unsupported-compiler
            -Xcompiler "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cu"
            -o "$<TARGET_FILE_DIR:cuda_recover>/kernel.ptx"
    COMMENT "Compiling src/kernel.cu -> kernel.ptx (next to cuda_recover.exe)")
