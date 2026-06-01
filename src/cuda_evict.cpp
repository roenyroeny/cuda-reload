#include "cuda_evict.h"
#include <winternl.h>   // PEB, PEB_LDR_DATA, UNICODE_STRING
#include <intrin.h>     // __readgsqword
#include <cstdio>
#include <cwchar>

// ===========================================================================
// Dynamic load
// ===========================================================================

typedef CUresult (CUDAAPI *PFN_GetProcAddr)(const char*, void**, int, cuuint64_t,
                                            CUdriverProcAddressQueryResult*);
static PFN_GetProcAddr g_getProc;   // re-bootstrapped from each loaded image

static bool through(const char* name, void** slot)
{
    void* fn = nullptr;
    CUdriverProcAddressQueryResult st = CU_GET_PROC_ADDRESS_SUCCESS;
    CUresult r = g_getProc(name, &fn, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &st);
    if (r != CUDA_SUCCESS || !fn) { std::printf("  resolve %s failed (r=%d)\n", name, (int)r); return false; }
    *slot = fn;
    return true;
}

bool cudaLoad(CudaApi& a)
{
    a = CudaApi{};
    a.mod = LoadLibraryA("nvcuda.dll");
    if (!a.mod) { std::printf("LoadLibrary(nvcuda.dll) failed (%lu)\n", GetLastError()); return false; }

    g_getProc = (PFN_GetProcAddr)GetProcAddress(a.mod, "cuGetProcAddress_v2");
    if (!g_getProc) { std::printf("cuGetProcAddress_v2 not found\n"); return false; }

    if (!through("cuInit", (void**)&a.Init)) return false;
    CUresult ri = a.Init(0);
    if (ri != CUDA_SUCCESS) { std::printf("cuInit -> %d\n", (int)ri); return false; }

    return through("cuGetErrorName",      (void**)&a.GetErrorName)
        && through("cuDeviceGet",         (void**)&a.DeviceGet)
        && through("cuDeviceGetName",     (void**)&a.DeviceGetName)
        && through("cuCtxCreate",         (void**)&a.CtxCreate)
        && through("cuCtxDestroy",        (void**)&a.CtxDestroy)
        && through("cuCtxSynchronize",    (void**)&a.CtxSynchronize)
        && through("cuMemAlloc",          (void**)&a.MemAlloc)
        && through("cuMemFree",           (void**)&a.MemFree)
        && through("cuMemcpyHtoD",        (void**)&a.MemcpyHtoD)
        && through("cuMemcpyDtoH",        (void**)&a.MemcpyDtoH)
        && through("cuModuleLoad",        (void**)&a.ModuleLoad)
        && through("cuModuleGetFunction", (void**)&a.ModuleGetFunction)
        && through("cuLaunchKernel",      (void**)&a.LaunchKernel);
}

const char* cudaErr(const CudaApi& a, CUresult r)
{
    const char* n = nullptr;
    if (a.GetErrorName) a.GetErrorName(r, &n);
    return n ? n : "?";
}

// ===========================================================================
// cuDeinit -- evict the driver DLLs from the Windows loader
// ===========================================================================

// The loader's per-module record. winternl.h hides most of it (and BaseDllName),
// so we spell out the prefix; the RB-tree node fields live further in and are
// auto-detected below rather than hardcoded.
typedef struct _LDR_ENTRY {
    LIST_ENTRY     InLoadOrderLinks;
    LIST_ENTRY     InMemoryOrderLinks;
    LIST_ENTRY     InInitializationOrderLinks;
    PVOID          DllBase;
    PVOID          EntryPoint;
    ULONG          SizeOfImage;
    UNICODE_STRING FullDllName;
    UNICODE_STRING BaseDllName;
} LDR_ENTRY;

// ntdll red-black tree node + head, and the exported remove routine.
typedef struct _RB_NODE { struct _RB_NODE* Left; struct _RB_NODE* Right; ULONG_PTR ParentValue; } RB_NODE;
typedef struct _RB_TREE { RB_NODE* Root; RB_NODE* Min; } RB_TREE;
typedef void (NTAPI *PFN_RtlRbRemoveNode)(RB_TREE*, RB_NODE*);

static RB_NODE* rbParent(RB_NODE* n) { return (RB_NODE*)(n->ParentValue & ~(ULONG_PTR)0x7); }

static int snapshotEntries(LDR_ENTRY** out, int cap)
{
    PPEB peb = (PPEB)__readgsqword(0x60);
    LIST_ENTRY* head = &peb->Ldr->InMemoryOrderModuleList;
    int n = 0;
    for (LIST_ENTRY* e = head->Flink; e != head && n < cap; e = e->Flink)
        out[n++] = CONTAINING_RECORD(e, LDR_ENTRY, InMemoryOrderLinks);
    return n;
}

static bool nameEq(const UNICODE_STRING* u, const wchar_t* name)
{
    size_t n = wcslen(name);
    return u->Buffer && u->Length == n * sizeof(wchar_t) && _wcsnicmp(u->Buffer, name, n) == 0;
}

// Find the module named `name`, mangle its first char in BaseDllName (and the
// FullDllName filename) so the loader's NAME lookup can no longer match it, and
// return its DllBase (for the index unlink). Same-length edit, in place.
static void* evictName(const wchar_t* name, wchar_t mark)
{
    LDR_ENTRY* es[1024];
    int n = snapshotEntries(es, 1024);
    for (int i = 0; i < n; ++i) {
        if (!nameEq(&es[i]->BaseDllName, name)) continue;
        void* base = es[i]->DllBase;
        es[i]->BaseDllName.Buffer[0] = mark;
        UNICODE_STRING* f = &es[i]->FullDllName;
        if (f->Buffer) {
            int chars = f->Length / (int)sizeof(wchar_t), slash = -1;
            for (int j = 0; j < chars; ++j) if (f->Buffer[j] == L'\\' || f->Buffer[j] == L'/') slash = j;
            if (slash + 1 < chars) f->Buffer[slash + 1] = mark;
        }
        return base;
    }
    return nullptr;
}

static bool isEntryNodeAt(LDR_ENTRY** es, int n, size_t off, RB_NODE* node)
{
    for (int i = 0; i < n; ++i) if ((RB_NODE*)((char*)es[i] + off) == node) return true;
    return false;
}

// Auto-detect the byte offsets of the RB-tree node fields in LDR_ENTRY by
// testing each candidate for self-consistent tree linkage across all modules:
// every non-null child is itself an entry-node at the same offset and points
// back to its parent, with exactly one root. Avoids version-specific offsets.
static int detectRbNodeOffsets(LDR_ENTRY** es, int n, size_t* offs, int maxOffs)
{
    int found = 0;
    for (size_t off = 0x60; off + sizeof(RB_NODE) <= 0x200 && found < maxOffs; off += sizeof(void*)) {
        bool ok = true; int roots = 0;
        for (int i = 0; i < n && ok; ++i) {
            RB_NODE* node = (RB_NODE*)((char*)es[i] + off);
            RB_NODE* kids[2] = { node->Left, node->Right };
            for (int k = 0; k < 2; ++k)
                if (kids[k] && (!isEntryNodeAt(es, n, off, kids[k]) || rbParent(kids[k]) != node)) { ok = false; break; }
            RB_NODE* par = rbParent(node);
            if (!par) ++roots;
            else if (!isEntryNodeAt(es, n, off, par)) ok = false;
        }
        if (ok && roots == 1) offs[found++] = off;
    }
    return found;
}

// Walk a node to its tree root + leftmost, then scan ntdll's writable sections
// for the matching RTL_RB_TREE head -- the genuine global, so RtlRbRemoveNode
// keeps Root/Min correct even when we remove the current root or min.
static RB_TREE* findTreeHead(RB_NODE* node)
{
    RB_NODE* root = node; while (rbParent(root)) root = rbParent(root);
    RB_NODE* mn = root;   while (mn->Left)      mn = mn->Left;

    BYTE* base = (BYTE*)GetModuleHandleW(L"ntdll.dll");
    if (!base) return nullptr;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
    IMAGE_NT_HEADERS* nt  = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
    IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
    for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (!(sec[i].Characteristics & IMAGE_SCN_MEM_WRITE)) continue;
        BYTE* s = base + sec[i].VirtualAddress;
        SIZE_T sz = sec[i].Misc.VirtualSize;
        for (SIZE_T o = 0; o + sizeof(RB_TREE) <= sz; o += sizeof(void*)) {
            RB_TREE* cand = (RB_TREE*)(s + o);
            if (cand->Root == root && cand->Min == mn) return cand;
        }
    }
    return nullptr;
}

int cuDeinit()
{
    static const wchar_t* kDriverDlls[] = { L"nvcuda.dll", L"nvcuda64.dll" };
    const int N = (int)(sizeof(kDriverDlls) / sizeof(kDriverDlls[0]));

    // Rename each (defeats the loader's name lookup) and capture its base.
    void* bases[N]; int nb = 0;
    for (int i = 0; i < N; ++i) {
        void* b = evictName(kDriverDlls[i], L'~');
        std::printf("  rename %-13ls -> %s\n", kDriverDlls[i], b ? "ok" : "not loaded");
        if (b) bases[nb++] = b;
    }
    if (nb == 0) return 0;

    PFN_RtlRbRemoveNode RtlRbRemoveNode =
        (PFN_RtlRbRemoveNode)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlRbRemoveNode");
    if (!RtlRbRemoveNode) { std::printf("  RtlRbRemoveNode missing -- name rename only\n"); return nb; }

    // Detect node offsets ONCE on the pristine list: a node already removed
    // from a tree has stale links that would break a second detection pass.
    LDR_ENTRY* es[1024];
    int n = snapshotEntries(es, 1024);
    size_t offs[4];
    int nOff = detectRbNodeOffsets(es, n, offs, 4);
    std::printf("  RB-tree node offsets:");
    for (int i = 0; i < nOff; ++i) std::printf(" +0x%zx", offs[i]);
    std::printf("\n");

    // Unlink each driver module from every loader index tree (base-address +
    // section/mapping). With the section index entry gone, the next LoadLibrary
    // maps a fresh image instead of reusing the resident one.
    for (int b = 0; b < nb; ++b) {
        LDR_ENTRY* t = nullptr;
        for (int i = 0; i < n; ++i) if (es[i]->DllBase == bases[b]) { t = es[i]; break; }
        if (!t) continue;
        for (int i = 0; i < nOff; ++i) {
            RB_NODE* node = (RB_NODE*)((char*)t + offs[i]);
            RB_TREE* tree = findTreeHead(node);
            if (tree) RtlRbRemoveNode(tree, node);
        }
    }
    return nb;
}
