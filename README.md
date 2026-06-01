# CUDA sticky-fault recovery: can a user-mode DLL reload fix it?

A focused experiment (`cuda_recover`) testing one hypothesis end to end:

> After a **sticky** CUDA fault (`CUDA_ERROR_ILLEGAL_ADDRESS`), can we recover
> *in-process* by forcing the OS loader to map a brand-new copy of the driver
> DLLs and re-initialising CUDA from scratch — no new process?

**Short answer: yes — if you force-reload *both* CUDA DLLs.** `nvcuda.dll` is a
thin (~4 MB) dispatch shim; the real driver and its state live in
`nvcuda64.dll` (~29 MB). Reloading only the shim fails — it just rebinds to the
still-poisoned `nvcuda64.dll`. But wipe **both** from the loader (rename + RB-tree
unlink) so the reload maps fresh copies of each, and the freshly-initialised
driver hands out a **clean context**: `cuCtxCreate` succeeds and a real
`vec_add` runs correctly (0/65536 mismatches). Reproduced 5/5 runs.

So the sticky fault is latched in the **per-process driver instance**
(`nvcuda64.dll`'s state plus the kernel-mode GPU context it owns), *not*
permanently in the hardware for the life of the process. A brand-new driver
instance gets a brand-new, clean GPU context. (The old poisoned driver instance
is orphaned but stays resident — see caveats.)

Measured on RTX 5080, Windows 11 (build 26200), CUDA 13.1.

## What `cuda_recover` does

The binary does **not** link `cuda.lib`. It bootstraps one symbol
(`cuGetProcAddress_v2`) with the OS loader, then resolves every other entry
point through CUDA's own version-negotiating loader (so the negotiated `_vN`
ABIs match — see ABI note below).

1. **Load CUDA dynamically.** `LoadLibrary("nvcuda.dll")` → `cuInit` →
   `cuDeviceGet` → `cuCtxCreate`.
2. **Fault the device.** Launch the prebuilt `oob_write` kernel through a
   pointer 1 GiB past a real allocation; `cuCtxSynchronize` returns
   `CUDA_ERROR_ILLEGAL_ADDRESS`. A probe `cuMemAlloc` then also fails — the
   context is poisoned (sticky confirmed).
3. **Wipe both `nvcuda.dll` and `nvcuda64.dll` from the loader:** rename them in
   the PEB *and* unlink them from the loader's RB-tree indexes (see below).
4. **`LoadLibrary("nvcuda.dll")` again** → fresh shim *and* fresh `nvcuda64.dll`
   at new bases → re-resolve → `cuInit` → `cuCtxCreate` (**SUCCESS**) →
   `vec_add` runs and verifies correct.

### Why the fault needs a kernel (the "no kernel" idea doesn't work)

The original plan was to fault the device with `cuMemsetD8` on a bad device
address — no kernel required. It does not work: **every validated memory op
rejects a bad pointer on the host and never reaches the GPU**, so none can latch
a sticky fault. Confirmed on this driver, all returning `CUDA_ERROR_INVALID_VALUE`
synchronously:

- `cuMemsetD8` wholly inside, or straddling into, an unbacked VMM reservation
- `cuMemSetAccess` on reserved-but-unmapped VA
- `cuStreamWriteValue32` (designed to poke arbitrary addresses) on unbacked/wild VA

A sticky `CUDA_ERROR_ILLEGAL_ADDRESS` requires actual GPU code dereferencing a
pointer the driver never inspects — i.e. a kernel. So `cuda_recover` reuses the
trivial `oob_write` kernel from `src/kernel.cu` (compiled to `kernel.ptx`).

### Why a PEB rename alone isn't enough — and what is

Renaming `BaseDllName`/`FullDllName` in the PEB loader list makes the *name*
lookup miss (`GetModuleHandle("nvcuda.dll")` → NULL), but `LoadLibrary` still
returns the **same base**: after the name miss the loader maps the file and
finds the resident image via the **section/mapping index**
(`LdrpFindLoadedDllByMapping`), then reuses it.

To force a fresh image we also unlink the module from the loader's two
red-black-tree indexes — `LdrpModuleBaseAddressIndex` and
`LdrpMappingInfoIndex`. The code:

- snapshots every module's `LDR_DATA_TABLE_ENTRY` from the PEB;
- **auto-detects** the RB-node field offsets inside the record by testing each
  candidate offset for self-consistent tree linkage across all modules (no
  hardcoded, version-specific offsets — on this build they land at `+0xc8` and
  `+0xe0`). Detection runs **once, before any removal** — a node pulled from a
  tree has stale links that would break a second detection pass;
- walks each node to its tree root + leftmost (`Min`), scans `ntdll`'s writable
  sections for the matching `RTL_RB_TREE` head (the genuine global, not a copy,
  so `RtlRbRemoveNode` updates `Root`/`Min` correctly), and removes the node
  with the exported `ntdll!RtlRbRemoveNode`.

With the node gone from the mapping index, the next `LoadLibrary` maps a new
view at a **new base**.

### The shim vs. the real driver — why both must be wiped

`nvcuda.dll` is a thin dispatch shim; the heavyweight driver state lives in
`nvcuda64.dll`, loaded during the first `cuInit`. Wiping only the shim recovers
nothing — the module dump after such a reload shows:

```
~vcuda.dll     base=...A1BE0000  size=4364 KiB    <- old shim (renamed)
nvcuda64.dll   base=...9FF50000  size=29200 KiB   <- the REAL driver, STILL the one instance
nvcuda.dll     base=...826A0000  size=4364 KiB    <- new shim (fresh base)
```

The fresh shim is a genuinely independent image (separate section view, its own
copy-on-write `.data`, `DllMain`/CRT re-run) — its own globals have no memory of
the old shim. But it binds straight back to the same, already-poisoned
`nvcuda64.dll`, so `cuCtxCreate` still fails.

Wipe `nvcuda64.dll` too and the reload maps a **fresh driver** as well — four
modules now coexist, the new `nvcuda64.dll` at a new base:

```
~vcuda.dll     base=...A1BE0000  size=4364 KiB    <- old shim   (renamed, orphaned)
~vcuda64.dll   base=...9FF50000  size=29200 KiB   <- old driver (renamed, orphaned)
nvcuda.dll     base=...027B0000  size=4364 KiB    <- new shim
nvcuda64.dll   base=...02C00000  size=29200 KiB   <- new driver (fresh!)  <-- the key
```

The fresh `nvcuda64.dll` re-initialises from scratch, establishes a new context
with the kernel-mode driver, and that context is clean. Recovery confirmed by
running a real kernel, not just by `cuCtxCreate`'s return code.

## Expected output (tail)

```
=== 2. Fault the device with an illegal memory access ===
  cuCtxSynchronize  -> CUDA_ERROR_ILLEGAL_ADDRESS
  cuMemAlloc (probe)  -> CUDA_ERROR_ILLEGAL_ADDRESS  (context poisoned -- sticky confirmed)

=== 3b. Unlink both old images from the loader RB-tree indexes ===
  detected 2 RB-tree node field(s) in LDR_ENTRY: +0xc8 +0xe0
  ~vcuda.dll (...):   +0xc8/+0xe0 removed
  ~vcuda64.dll (...): +0xc8/+0xe0 removed

=== 4. LoadLibrary(nvcuda.dll) again + create a fresh context ===
  LoadLibrary(nvcuda.dll) -> <NEW base>   (NEW base -- fresh image mapped from disk!)
  modules matching "cuda" (after reload + cuInit):
    ~vcuda.dll / ~vcuda64.dll  (old, orphaned)
    nvcuda.dll / nvcuda64.dll  (new, fresh bases)
  [reload] cuCtxCreate     -> CUDA_SUCCESS

=== RESULT ===
  validate: cuLaunchKernel      -> CUDA_SUCCESS
  validate: vec_add results     -> CORRECT (0/65536 mismatches)
  RECOVERED: fresh context runs kernels correctly after wiping both nvcuda images.
```

## Caveats

- **The old driver instance is leaked, not freed.** We only orphan it (rename +
  unlink); `~vcuda64.dll` and its kernel-mode context stay resident until the
  process exits. Each recovery costs ~29 MB of user-mode plus a GPU context, so
  this suits occasional recovery, not an unbounded retry loop.
- **This is undocumented loader surgery.** It pokes ntdll's internal RB-tree
  indexes. Offsets are auto-detected (not hardcoded), but a future Windows
  loader could change the structure shape or dedup logic and break it. Do it
  with the loader lock held in production; this experiment runs single-threaded.
- Validated only on this config (RTX 5080 / Win11 26200 / CUDA 13.1).

## Build & run (Windows / MSVC)

Uses the default Visual Studio generator (no Ninja):

```bat
cmake -B build
cmake --build build --config Release
bin\cuda_recover.exe               REM reads kernel.ptx from its own dir (bin\)
```

The executable and the `kernel.ptx` copied next to it land in `bin/` for every
config (the per-config `bin/Release` etc. subdir is overridden).

`kernel.ptx` is compiled by a post-build step that invokes `nvcc` directly (no
CUDA-language / VS-integration dependency). Because this VS 2026 / CUDA 13.1
combo is newer than nvcc and the MSVC STL expect, the post-build passes
`-allow-unsupported-compiler` and `/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`
to get past both version gates (the kernel is trivial device code).

## Layout

| File                  | Purpose                                                   |
|-----------------------|-----------------------------------------------------------|
| `src/cuda_evict.h/.cpp` | The reusable unit: minimal dynamic CUDA dispatch (`cudaLoad`) + `cuDeinit()` (PEB rename + loader RB-tree unlink). |
| `src/main.cpp`        | Tight demo: load → fault → `cuDeinit` → reload → validate. |
| `src/kernel.cu`       | `oob_write` fault kernel + `vec_add` validation kernel → `kernel.ptx`. |
| `CMakeLists.txt`      | Builds `cuda_recover` + the `nvcc -ptx` post-build step.   |

The recovery primitive is a drop-in: `#include "cuda_evict.h"`, then after a
sticky fault call `cuDeinit()` and `cudaLoad()` a fresh `CudaApi`.

## ABI note

`cuGetProcAddress` negotiates the latest ABI for the requested `CUDA_VERSION`
(13010 here). In CUDA 13 several functions gained a context parameter, so e.g.
`cuCtxSynchronize` resolves to `cuCtxSynchronize_v2` (takes a `CUcontext`) and
`cuCtxCreate` to `_v4` (takes a `CUctxCreateParams*`). Declare the negotiated
signatures, not the legacy ones — calling a `_v2`/`_v4` entry with the old
prototype passes garbage in the context register and crashes inside the driver.

## Bottom line

A sticky CUDA fault **can** be cleared in-process — but not by reloading the
`nvcuda.dll` shim alone. You must force the OS loader to map a fresh copy of the
real driver, `nvcuda64.dll`, by renaming **and** unlinking both DLLs from the
loader's name and section indexes. The freshly-initialised driver gets a clean
GPU context; the poisoned one is orphaned. A separate worker process is still
the simpler, more robust answer, but full process restart is not strictly
required to escape a sticky fault.
