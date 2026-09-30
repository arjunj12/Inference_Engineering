Course.module({
  id: "m11-cuda-basics",
  title: "CUDA from scratch: write, compile and benchmark your own kernels",
  short: "CUDA from scratch",
  tagline: "Write real CUDA kernels on a free Colab T4 — vector add → reduction → softmax → tiled matmul — and measure them against the hardware limits and cuBLAS.",
  hours: 20,
  level: "core",
  runsOn: ["colab", "browser"],
  tags: ["cuda", "kernels", "matmul", "shared-memory", "roofline"],

  goal: MD(function () {/*
By the end of this module a single Colab notebook cell prints a table like this — every row is a kernel **you** wrote in CUDA C++, compiled with `nvcc`, timed with CUDA events and checked against a reference:

```text
GPU: Tesla T4 (sm_75, 40 SMs, 320 GB/s, 8.1 TFLOPS FP32)
kernel                        time        achieved              % of limit
vector_add  (2^26 floats)     3.05 ms     264 GB/s              83% of HBM
reduce_sum  warp-shuffle      1.10 ms     244 GB/s              76% of HBM
softmax     4096 x 4096       0.52 ms     258 GB/s  (ideal IO)  81% of HBM
matmul      naive   4096^3    272 ms      0.51 TFLOPS           12% of cuBLAS
matmul      smem tiled 32     112 ms      1.23 TFLOPS           30% of cuBLAS
matmul      1D reg-blocked     56 ms      2.45 TFLOPS           60% of cuBLAS
cuBLAS sgemm (reference)       34 ms      4.07 TFLOPS          100%
```

(Your exact numbers will differ — T4s throttle at 70 W — but the *shape* of this table is the point.) More importantly, you’ll be able to explain every row: why vector add can never beat ~320 GB/s, why the naive matmul is stuck at a few % of peak even though the GPU is “busy”, and why shared-memory tiling fixes it. That explanation is the core skill of a kernel/performance engineer.

::viz cuda-grid
  */}),
  demo: { viz: "tiled-matmul", params: {} },

  why: MD(function () {/*
Everything in an inference engine eventually becomes a **kernel launch**: matmuls (cuBLAS/CUTLASS), attention (FlashAttention/FlashInfer), RMSNorm, RoPE, sampling. Most inference engineers never write a production matmul — but they constantly **read, select, fuse, debug and profile** kernels (Inference Engineering (Baseten) ch. 4.1: “most engineers never write kernels, but they do select them”). You can’t reason about “this op is memory-bound, fuse it” or “this kernel only hits 30% of bandwidth” until you’ve written a few kernels and watched the numbers move.

Kernel and performance roles (NVIDIA, frontier labs, Together, Fireworks, Baseten, Modal, PyTorch/vLLM/SGLang teams) screen for exactly this: *explain the CUDA thread hierarchy, write a reduction, explain coalescing and shared memory, estimate the roofline of a kernel.* This module gets you there; M12 (Triton + FlashAttention) and M13 (profiling) build on it.

> [!NOTE] Mac users
> CUDA only runs on NVIDIA GPUs. Your Mac can’t run anything in this module locally — you’ll use **Google Colab’s free T4** (or [LeetGPU](https://leetgpu.com), which runs CUDA in the browser). There’s an optional side-quest writing the same kernels for your Apple GPU with **MLX `mx.fast.metal_kernel`**, and the Muser book if you want to go deep on Metal.
  */}),

  prereqs: [
    {
      title: "Pointers and arrays in C, in five minutes (for a Python engineer)",
      skipIf: "you’ve written C or C++ with raw pointers",
      md: MD(function () {/*
CUDA is C++ with a few extensions. You need exactly this much C:

```cpp
float* a;                          // a pointer: the address of a float in memory
a = (float*)malloc(n * sizeof(float));   // allocate n floats (4 bytes each) on the CPU
a[i] = 3.0f;                       // same as *(a + i) = 3.0f  (pointer arithmetic)
const float* x;                    // pointer to floats you promise not to modify
free(a);
```

- A 2-D matrix is stored as **one flat array**, row after row (“row-major”): element `(row, col)` of an `M x K` matrix lives at `A[row * K + col]`. PyTorch tensors default to the same layout.
- `int` is 32-bit; indices past ~2.1 billion overflow — use `size_t` for byte counts.
- There is no bounds checking. Reading `a[n]` doesn’t raise; it silently reads garbage (or crashes the GPU context). That is why every kernel you write starts with `if (i < n)`.
      */}),
    },
    {
      title: "Colab GPU runtime + nvcc in a notebook",
      skipIf: "you already compile CUDA on a machine with an NVIDIA GPU",
      md: MD(function () {/*
1. Open [colab.research.google.com](https://colab.research.google.com) → *Runtime → Change runtime type → T4 GPU*.
2. Colab images ship the CUDA toolkit, so `nvcc` is already there:

```bash
!nvidia-smi                 # Tesla T4, driver + CUDA version, 15 GB free
!nvcc --version             # the compiler
```

3. The workflow for every exercise: write the file with the `%%writefile` cell magic, then compile and run with `!` shell commands:

```cpp
%%writefile hello.cu
#include <cstdio>
__global__ void hello() { printf("block %d thread %d\n", blockIdx.x, threadIdx.x); }
int main() { hello<<<2, 4>>>(); cudaDeviceSynchronize(); return 0; }
```

```bash
!nvcc -O3 -arch=sm_75 hello.cu -o hello && ./hello
```

`-arch=sm_75` targets the T4’s architecture (Turing). An L4 is `sm_89`, A100 `sm_80`, H100 `sm_90`.

> [!TIP] No Colab quota left?
> [LeetGPU](https://leetgpu.com) runs CUDA (and Triton) in the browser with graded problems — vector add, reduction, softmax and matmul are all on it.
      */}),
    },
    {
      title: "Roofline recap (from M07)",
      skipIf: "you can compute arithmetic intensity and say whether a kernel is memory- or compute-bound",
      math: true,
      md: MD(function () {/*
- **Arithmetic intensity (AI)** = FLOPs performed ÷ bytes moved to/from memory.
- A GPU has two ceilings: peak **FLOP/s** and peak **bytes/s**. Their ratio is the *ridge point*. T4: $8.1 \times 10^{12} / 320 \times 10^{9} \approx 25$ FLOP/byte in FP32 (and ~200 with FP16 tensor cores).
- If AI < ridge → **memory-bound**: time ≈ bytes ÷ bandwidth. If AI > ridge → **compute-bound**: time ≈ FLOPs ÷ peak FLOP/s.
- Vector add does 1 FLOP per 12 bytes (AI ≈ 0.08) — hopelessly memory-bound. A big matmul can reach AI in the hundreds — compute-bound *if* the kernel is written well.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: your first kernel, timed against PyTorch",
      kind: "demo",
      minutes: 60,
      runsOn: ["colab", "browser"],
      md: MD(function () {/*
Top-down as always: get a real kernel running and measured in the first 20 minutes, then take it apart.

## Step 1 — vector add in CUDA

Paste this into a Colab cell (T4 runtime):

```cpp
%%writefile vadd.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// __global__ = runs on the GPU, launched from the CPU. One thread computes one element.
__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // my global index
  if (i < n) c[i] = a[i] + b[i];                    // guard: the last block may overhang
}

int main() {
  const int n = 1 << 26;                            // 67M floats = 256 MB per array
  size_t bytes = (size_t)n * sizeof(float);
  float* ha = (float*)malloc(bytes);
  float* hb = (float*)malloc(bytes);
  float* hc = (float*)malloc(bytes);
  for (int i = 0; i < n; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes)); CHECK(cudaMalloc(&db, bytes)); CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;         // ceil(n / threads)
  vadd<<<blocks, threads>>>(da, db, dc, n);         // warm-up launch
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  const int iters = 20;
  cudaEventRecord(start);
  for (int r = 0; r < iters; r++) vadd<<<blocks, threads>>>(da, db, dc, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0; cudaEventElapsedTime(&ms, start, stop); ms /= iters;

  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  bool ok = hc[0] == 3.0f && hc[n - 1] == 3.0f;
  double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;    // read a, read b, write c
  printf("vadd n=%d  %.3f ms  %.1f GB/s  correct=%s\n", n, ms, gbps, ok ? "yes" : "NO");
  return 0;
}
```

```bash
!nvcc -O3 -arch=sm_75 vadd.cu -o vadd && ./vadd
```

Expected on a T4: roughly `3.0–3.3 ms` and `240–270 GB/s`.

## Step 2 — the same thing in PyTorch

```python
import torch
n = 1 << 26
a = torch.ones(n, device="cuda"); b = torch.full((n,), 2.0, device="cuda")
for _ in range(3): c = a + b                     # warm-up
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20): c = a + b
end.record(); torch.cuda.synchronize()
ms = start.elapsed_time(end) / 20
print(f"torch a+b: {ms:.3f} ms  {3 * n * 4 / (ms * 1e-3) / 1e9:.1f} GB/s")
```

You should see **almost the same number**. Your 5-line kernel is as fast as PyTorch’s — because both are limited by the same thing: the T4 can move at most **320 GB/s** between HBM (actually GDDR6 on T4) and the chip, and vector add moves 12 bytes per 1 FLOP.

> [!INTUITION] The lesson hiding in this demo
> For memory-bound kernels, “fast” means *close to peak bandwidth*, and the only way to go faster is to **move fewer bytes** (fusion, lower precision). You can’t out-code physics. Keep this in mind for the whole phase: roughly 80% of the kernels in an LLM decode step are like this one.

## Step 3 — break it on purpose

- [ ] Ran `vadd` and the PyTorch version; recorded both GB/s numbers
- [ ] Removed the `if (i < n)` guard and set `n = (1 << 26) + 1`. Run with `!compute-sanitizer ./vadd` — see the out-of-bounds write reported
- [ ] Changed `threads` to 32, 64, 128, 512, 1024. Which hurt? (Hint: 1025 fails to launch — check the error from `cudaGetLastError`)
- [ ] Timed a single launch with tiny `n = 1024`. It takes ~5–10 µs no matter how small the work — that’s **launch overhead**, the villain of M13
      */}),
      resources: [
        { title: "LeetGPU", url: "https://leetgpu.com/", type: "practice", note: "CUDA/Triton in the browser with graded problems — no GPU needed" },
        { title: "NVIDIA blog — How to Implement Performance Metrics in CUDA C/C++", url: "https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/", type: "article", note: "cudaEvent timing and effective bandwidth, the canonical short intro" },
      ],
    },

    {
      id: "programming-model",
      title: "The CUDA programming model: grids, blocks, threads, warps",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
## The mental model

A GPU is a **throughput machine**: thousands of simple threads running the *same program* on different data. CUDA gives you a two-level hierarchy for organizing them:

| Level | What it is | Size limits | Cooperates via |
|---|---|---|---|
| **Thread** | one instance of your kernel function | — | its own registers |
| **Block** (thread block, CTA) | up to 1024 threads scheduled on **one SM** | ≤ 1024 threads | `__shared__` memory + `__syncthreads()` |
| **Grid** | all blocks of one launch | up to $2^{31}-1$ blocks in x | only global memory (and kernel boundaries) |

You launch a kernel with `kernel<<<gridDim, blockDim>>>(args)`. Inside it, built-in variables tell each thread who it is: `blockIdx`, `blockDim`, `threadIdx`, `gridDim` (each has `.x .y .z`). The formula you’ll write a thousand times:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;   // global 1-D index
```

Hover over the elements below to see each one’s `blockIdx`, `threadIdx` and global index:

::viz cuda-grid

## What the hardware actually does

::viz gpu-anatomy

- The GPU is made of **SMs** (streaming multiprocessors). T4: 40 SMs. A100: 108. H100 SXM: 132.
- The block scheduler assigns whole blocks to SMs. A block never migrates; several blocks can share an SM if resources (registers, shared memory, thread slots) allow.
- Inside an SM, threads execute in groups of **32 called warps**. A warp issues one instruction at a time for all 32 threads — **SIMT** (single instruction, multiple threads).
- When a warp waits on memory (~400–800 cycles for HBM/DRAM), the SM’s scheduler switches to another ready warp **in one cycle**. This is how GPUs hide latency: not with big caches like CPUs, but with **lots of warps in flight** (“occupancy”).

> [!IMPORTANT] Three consequences you must internalize
> 1. **Block size should be a multiple of 32.** A 100-thread block wastes 28 lanes of its 4th warp.
> 2. **Divergence costs.** If threads in one warp take different branches of an `if`, the warp executes *both* paths with lanes masked off. Branch on data per-warp, not per-thread, when you can.
> 3. **Blocks are independent.** There’s no guaranteed order and no barrier across blocks inside one kernel. If you need a global result (e.g. a sum), either use atomics or launch a second kernel.

## 2-D grids for 2-D problems

For matrices, use 2-D blocks so the index math is natural:

```cpp
dim3 block(32, 32);                                   // 1024 threads
dim3 grid((N + 31) / 32, (M + 31) / 32);              // cover N columns, M rows
kernel<<<grid, block>>>(...);
// inside:
int col = blockIdx.x * blockDim.x + threadIdx.x;      // x is the fast-changing dimension
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Threads are numbered with **x fastest**: the 32 threads of a warp are `threadIdx.x = 0..31` with the same `threadIdx.y`. Remember that — it decides whether your memory accesses are coalesced (next lesson).

## The grid-stride loop

Instead of one thread per element, launch “enough” blocks to fill the GPU and let each thread loop:

```cpp
__global__ void scale(float* x, float s, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    x[i] *= s;
}
// launch with e.g. 40 SMs x 8 blocks: scale<<<320, 256>>>(x, 2.f, n);
```

This decouples problem size from launch size and is how most production elementwise kernels are written.

<details><summary>Same ideas on your Mac (Metal)</summary>

Apple GPUs use the same model with different names. The Muser book’s *Ch. 2 The Metal compute model* and *Ch. 29 CUDA vs Metal* walk through this mapping in detail.

| CUDA | Metal |
|---|---|
| thread | thread |
| block | threadgroup |
| warp (32) | SIMD-group (32 on Apple GPUs) |
| `__shared__` memory | `threadgroup` memory |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__shfl_down_sync` | `simd_shuffle_down`, `simd_sum` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `thread_position_in_grid` |
| global memory (HBM) | device memory (unified with the CPU) |

</details>

- [ ] Explain to yourself why a warp is the unit of scheduling, and what happens on divergence
- [ ] Wrote the grid-stride version of `vadd` and confirmed the same bandwidth with `<<<320, 256>>>`
- [ ] Read `!nvidia-smi -q | grep -i -A3 "clocks"` and looked up the T4’s SM count with `torch.cuda.get_device_properties(0)`
      */}),
      resources: [
        { title: "CUDA Programming Guide — programming model", url: "https://docs.nvidia.com/cuda/cuda-programming-guide/", type: "docs", note: "the normative reference for threads, blocks, grids and warps" },
        { title: "Modal GPU Glossary", url: "https://modal.com/gpu-glossary", type: "docs", note: "plain-English definitions of SM, warp, occupancy, SIMT" },
      ],
    },

    {
      id: "memory-model",
      title: "The memory model: global, shared, registers — and coalescing",
      kind: "concept",
      minutes: 90,
      runsOn: ["colab"],
      md: MD(function () {/*
Almost all kernel optimization is **memory optimization**. Compute is cheap; getting data to the compute units is expensive.

::viz memory-hierarchy

| Memory | Scope | Size (T4 / H100) | Bandwidth | Latency |
|---|---|---|---|---|
| **Registers** | one thread | 64K × 32-bit per SM (256 KB) | ~tens of TB/s aggregate | ~1 cycle |
| **Shared memory** (`__shared__`, SMEM) | one block | up to 64 KB/SM (T4) · 228 KB/SM (H100) | ~10+ TB/s aggregate | ~20–30 cycles |
| **L2 cache** | whole GPU | 4 MB (T4) · 50 MB (H100) | a few TB/s | ~200 cycles |
| **Global memory** (HBM / GDDR) | whole GPU + host copies | 16 GB (T4) · 80 GB (H100) | 320 GB/s (T4) · 3.35 TB/s (H100) | ~400–800 cycles |

The game: **load each byte from global memory as few times as possible**, keep reused data in shared memory / registers, and make every global load **coalesced**.

## Coalescing — the #1 rule

Global memory is read in **32-byte sectors** (grouped into 128-byte lines). When the 32 threads of a warp issue a load, the hardware merges their addresses into as few transactions as possible:

- Thread `t` reads `x[base + t]` (consecutive floats) → 32 × 4 B = 128 B = **4 sectors, one line**. Perfect.
- Thread `t` reads `x[base + t * 32]` (stride 32) → 32 different lines → **32× more traffic** for the same useful bytes.

::viz coalescing

## Measure it yourself

```cpp
%%writefile stride.cu
#include <cstdio>
#include <cuda_runtime.h>
__global__ void copy_stride(const float* in, float* out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (int)(((long long)i * stride) % n);        // scatter reads across memory
  if (i < n) out[i] = in[j];
}
int main() {
  int n = 1 << 25; size_t bytes = (size_t)n * sizeof(float);
  float *in, *out; cudaMalloc(&in, bytes); cudaMalloc(&out, bytes);
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  for (int stride : {1, 2, 4, 8, 16, 32, 64}) {
    copy_stride<<<n / 256, 256>>>(in, out, n, stride);        // warm-up
    cudaEventRecord(s);
    for (int r = 0; r < 10; r++) copy_stride<<<n / 256, 256>>>(in, out, n, stride);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= 10;
    printf("stride %2d: %7.3f ms  useful bandwidth %6.1f GB/s\n", stride, ms, 2.0 * bytes / (ms * 1e-3) / 1e9);
  }
}
```

```bash
!nvcc -O3 -arch=sm_75 -std=c++17 stride.cu -o stride && ./stride
```

You’ll see useful bandwidth collapse from ~250 GB/s at stride 1 to a small fraction by stride 8–32. Same FLOPs, same bytes *requested* — wildly different bytes *moved*.

## Shared memory: a programmer-managed cache

`__shared__ float tile[32][32];` declares memory that all threads in a block share. The standard pattern:

```cpp
__shared__ float tile[256];
tile[threadIdx.x] = x[blockIdx.x * 256 + threadIdx.x];   // 1. coalesced load from global
__syncthreads();                                         // 2. wait until everyone has loaded
float v = tile[255 - threadIdx.x];                       // 3. reuse / reorder cheaply
```

Two things to know:

- **`__syncthreads()` is a barrier for the whole block.** Never put it inside an `if` that only some threads take — the kernel hangs.
- **Bank conflicts.** SMEM is split into 32 banks (4-byte words, bank = word index mod 32). If two lanes of a warp hit *different addresses in the same bank*, the accesses serialize. Column-wise access of a `[32][32]` float tile is a 32-way conflict; the classic fix is padding: `__shared__ float tile[32][33];`.

## Registers and occupancy

Local variables live in registers — the fastest storage. But the SM has a fixed register file (64K registers). If each thread uses 128 registers, only 512 threads fit per SM → fewer warps to hide latency. Check usage with:

```bash
!nvcc -O3 -arch=sm_75 --ptxas-options=-v stride.cu -o stride 2>&1 | grep registers
```

> [!REAL] Why this matters for LLMs
> A decode step is dominated by reading weights (GEMV) and KV cache (attention). Engines live or die on coalesced, vectorized loads (`float4` / 16-byte loads), because at batch 1 they are *pure* memory streaming. The FlashAttention trick (M12) is 100% a shared-memory reuse trick.

- [ ] Ran `stride.cu`; plotted GB/s vs stride
- [ ] Wrote a matrix **transpose** kernel two ways: naive (reads coalesced, writes strided) and via a `[32][33]` shared-memory tile (both coalesced). Measured the speedup
- [ ] Removed the `+1` padding (`[32][32]`) and measured the bank-conflict penalty
      */}),
      resources: [
        { title: "NVIDIA blog — How to Access Global Memory Efficiently", url: "https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/", type: "article", note: "coalescing with measured stride/offset experiments" },
        { title: "NVIDIA blog — Using Shared Memory in CUDA C/C++", url: "https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/", type: "article", note: "shared memory, __syncthreads and bank conflicts" },
        { title: "Efficient Matrix Transpose in CUDA C/C++", url: "https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/", type: "article", note: "the transpose exercise, with padding trick" },
      ],
    },

    {
      id: "reduction-softmax",
      title: "Build: reductions with shared memory & warp shuffles, then row softmax",
      kind: "build",
      minutes: 180,
      runsOn: ["colab"],
      md: MD(function () {/*
Vector add is embarrassingly parallel. The next step up is a **reduction** — many inputs, one output — which forces threads to *cooperate*. Reductions are inside every LLM layer: RMSNorm (sum of squares), softmax (max and sum), sampling (argmax / top-k), loss.

## Version 1 — shared-memory tree reduction

Each block sums its chunk in shared memory with a tree: 256 → 128 → 64 → … → 1, then one thread adds the block’s partial sum into the global result with an atomic.

```cpp
__global__ void reduce_smem(const float* x, float* out, int n) {
  __shared__ float s[256];                         // blockDim.x must be 256
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;       // each thread first adds 2 elements
  float v = 0.f;
  if (i < n) v += x[i];
  if (i + blockDim.x < n) v += x[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];   // sequential addressing: no bank conflicts
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, s[0]);
}
// launch: reduce_smem<<<(n + 511) / 512, 256>>>(x, out, n);   (out zeroed with cudaMemset first)
```

## Version 2 — warp shuffles

Within a warp, threads can read each other’s **registers** directly with `__shfl_down_sync` — no shared memory, no `__syncthreads()`:

```cpp
__inline__ __device__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);   // lane i adds lane i+off
  return v;                                      // lane 0 holds the warp's total
}

__global__ void reduce_warp(const float* x, float* out, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    v += x[i];                                   // grid-stride: many elements per thread
  v = warp_sum(v);
  __shared__ float warp_sums[32];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) atomicAdd(out, v);
  }
}
// launch: reduce_warp<<<40 * 16, 256>>>(x, out, n);
```

Why it’s faster: fewer `__syncthreads()`, less SMEM traffic, and each thread accumulates many elements in a register before any cooperation (more work per thread = better latency hiding with fewer blocks).

**Harness task:** put both kernels in `reduce.cu` with a `main` like `vadd.cu`: fill `x` with `rand() / (float)RAND_MAX`, compute a `double` sum on the CPU, time each kernel with CUDA events (remember `cudaMemset(out, 0, 4)` before each launch), and print ms, GB/s (`n * 4` bytes read) and relative error.

> [!WARNING] Floating-point sums aren’t associative
> The GPU adds in a different order than the CPU and `atomicAdd` order is non-deterministic, so the last digits differ run to run. Check relative error `< 1e-4`, not equality. (This is also why batch-invariant, deterministic inference is a real engineering topic.)

## Row softmax — one block per row

Softmax over the last dimension (e.g. attention scores, logits) of an `R x C` matrix:

$$\text{softmax}(x)_j = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}, \qquad m = \max_k x_k$$

Subtracting the max doesn’t change the result but prevents `exp` overflow (M12’s math lesson covers why). That needs **two reductions** per row (max, then sum) — so build a reusable block-wide reduction:

```cpp
#include <cmath>

__inline__ __device__ float warp_max(float v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;                                        // xor-shuffle: EVERY lane ends with the result
}
__inline__ __device__ float warp_sum_all(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

template <bool IsMax>
__device__ float block_reduce(float v) {           // every thread of the block gets the result
  __shared__ float buf[32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = IsMax ? warp_max(v) : warp_sum_all(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  int nwarps = blockDim.x >> 5;
  v = (lane < nwarps) ? buf[lane] : (IsMax ? -INFINITY : 0.f);
  v = IsMax ? warp_max(v) : warp_sum_all(v);
  __syncthreads();                                 // buf may be reused by the next call
  return v;
}

__global__ void softmax_rows(const float* x, float* y, int cols) {
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, row[j]);   // pass 1: max
  m = block_reduce<true>(m);
  float s = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) s += __expf(row[j] - m); // pass 2: sum
  s = block_reduce<false>(s);
  float inv = 1.f / s;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) out[j] = __expf(row[j] - m) * inv;  // pass 3
}
// launch: softmax_rows<<<rows, 256>>>(x, y, cols);
```

Notice the loops: consecutive threads read consecutive `j` → coalesced. The row is read three times, but for rows up to a few thousand floats passes 2–3 mostly hit L1/L2, so HBM traffic ≈ **read once + write once**. That’s the number to compare against:

$$\text{effective bandwidth} = \frac{2 \cdot R \cdot C \cdot 4\ \text{bytes}}{t}$$

> [!INTUITION] How this connects
> Passes 1 and 2 can be merged into **one** pass with the *online softmax* trick (running max + rescaled running sum). That trick, applied across tiles of the attention matrix, *is* FlashAttention. You’ll derive it in M12.

- [ ] `reduce.cu`: both reductions correct (rel. error `< 1e-4`) and ≥ 70% of peak bandwidth for the warp version
- [ ] `softmax.cu`: matches a CPU reference (`max abs err < 1e-6`) for `4096 x 4096` and an awkward shape like `1000 x 1537`
- [ ] Reported softmax effective bandwidth; compared with `torch.softmax(x, dim=-1)` timed with CUDA events
- [ ] Tried `blockDim = 128, 512, 1024` for softmax and explained the result
      */}),
      resources: [
        { title: "Mark Harris — Optimizing Parallel Reduction in CUDA", url: "https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf", type: "article", note: "the classic 7-step reduction optimization deck" },
        { title: "NVIDIA blog — Using CUDA Warp-Level Primitives", url: "https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/", type: "article", note: "__shfl_*_sync, masks and warp-synchronous programming done right" },
      ],
    },

    {
      id: "matmul-math",
      title: "Math: matmul FLOPs (2MNK) and why tiling raises arithmetic intensity",
      kind: "math",
      minutes: 75,
      runsOn: ["any"],
      md: MD(function () {/*
> [!PREREQ] Matrix shapes in one line
> $A$ is $M \times K$ (M rows, K columns), $B$ is $K \times N$. $C = AB$ is $M \times N$, and each entry is a dot product: $C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$. The inner dimensions ($K$) must match.

::viz matmul {"m":4,"k":3,"n":5}

## Counting FLOPs

Each $C_{ij}$ needs $K$ multiplies and $K$ adds (strictly $K-1$, but everyone rounds) → $2K$ FLOPs. There are $M \cdot N$ entries, so:

$$\text{FLOPs}(C = AB) = 2MNK$$

**Worked example:** $M = N = K = 4096$ → $2 \cdot 4096^3 \approx 1.37 \times 10^{11}$ FLOPs = 137 GFLOP. On a T4 at a realistic ~4 TFLOPS FP32 that’s ~34 ms; at the 65 TFLOPS FP16 tensor-core peak it would be ~2 ms.

> [!REAL] The same formula runs the whole course
> A linear layer applied to $T$ tokens with a $d_{in} \times d_{out}$ weight costs $2 T d_{in} d_{out}$ FLOPs — that’s where “≈ 2 × params FLOPs per token” comes from (M06/M07).

## Bytes — and the difference between *minimum* and *actual*

The **minimum** traffic is reading $A$ and $B$ once and writing $C$ once: $4(MK + KN + MN)$ bytes in FP32. For $4096^3$ that’s ~201 MB, so the *ideal* intensity is

$$\text{AI}_{\text{ideal}} = \frac{2 \cdot 4096^3}{201 \times 10^6} \approx 683 \ \text{FLOP/byte}$$

— far above the T4’s ridge (~25). So matmul *can* be compute-bound. Whether it *is* depends on your kernel.

**Naive kernel:** each thread computes one $C_{ij}$ and reads a full row of $A$ ($K$ floats) and a full column of $B$ ($K$ floats) straight from global memory. Per output: $2K$ FLOPs, $8K$ bytes requested →

$$\text{AI}_{\text{naive}} = \frac{2K}{8K} = 0.25\ \text{FLOP/byte}$$

Caches recover some of that, but you’re fundamentally hammering memory — which is why the naive kernel sits at a few % of peak.

## Tiling: reuse each loaded value $T$ times

Split $C$ into $T \times T$ tiles, one per block. To compute a tile, march along $K$ in steps of $T$: load a $T \times T$ tile of $A$ and a $T \times T$ tile of $B$ into shared memory (**$2T^2$ loads**), then every thread does $T$ multiply-adds out of SMEM. Per step, the block does $2T^3$ FLOPs for $2T^2 \times 4$ bytes of global traffic:

$$\text{AI}_{\text{tiled}} = \frac{2T^3}{8T^2} = \frac{T}{4}\ \text{FLOP/byte}$$

| Tile $T$ | Global loads per output vs naive | AI (FLOP/byte) | vs T4 ridge ≈ 25 |
|---|---|---|---|
| 1 (naive) | 1× | 0.25 | 100× too low |
| 16 | 1/16 | 4 | memory-bound |
| 32 | 1/32 | 8 | memory-bound (but L2 helps) |
| 64 × 64 block + register tiles | 1/64 | 16+ | approaching compute-bound |
| 128 × 128 (cuBLAS-class) | 1/128 | 32+ | compute-bound |

Drag the tile size here and watch global-memory loads drop:

::viz tiled-matmul

> [!INTUITION] Why not just use one gigantic tile?
> Shared memory is small (≤ 64 KB/SM on T4: two $64 \times 64$ FP32 tiles = 32 KB) and each thread only has ~255 registers. Bigger tiles also mean fewer blocks, so fewer SMs busy on small matrices. Real kernels use **two levels**: block tiles in SMEM and smaller per-thread tiles in **registers** (register blocking) — that’s the path from 30% to 90% of cuBLAS in Simon Boehm’s article.

## Check yourself

- [ ] Computed FLOPs and minimum bytes for a Llama-3-8B MLP up-projection at batch 1 ($1 \times 4096$ by $4096 \times 14336$) and at batch 256. Which is memory-bound on an H100 (ridge ≈ 295 in BF16)?
- [ ] Derived $\text{AI}_{\text{tiled}} = T/4$ yourself on paper for $T = 16$
      */}),
    },

    {
      id: "tiled-matmul",
      title: "Build: naive → shared-memory tiled → register-blocked matmul vs cuBLAS",
      kind: "build",
      minutes: 240,
      runsOn: ["colab"],
      md: MD(function () {/*
The canonical kernel-engineering exercise. You’ll write three SGEMM kernels ($C = AB$, FP32, row-major) and benchmark them against **cuBLAS** in one harness.

## The harness

```cpp
%%writefile mm.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------- kernel 1: naive, one thread per C element ----------
__global__ void mm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> column: a warp reads 32 consecutive B/C columns
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------- kernel 2: shared-memory tiling ----------
#define TILE 32
__global__ void mm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;   // coalesced along tx
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();                               // don't overwrite tiles others still read
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------- kernel 3: 1-D register blocking (each thread computes TM outputs) ----------
// Assumes M % BM == 0, N % BN == 0, K % BK == 0 (true for 4096).
template <int BM, int BN, int BK, int TM>
__global__ void mm_reg(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;          // each thread owns rows threadRow*TM .. +TM-1
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;
  const int aCol = threadIdx.x % BK, aRow = threadIdx.x / BK;   // BM*BK == blockDim.x
  const int bCol = threadIdx.x % BN, bRow = threadIdx.x / BN;   // BK*BN == blockDim.x
  float acc[TM] = {0.f};
  for (int bk = 0; bk < K; bk += BK) {
    As[aRow * BK + aCol] = A[aRow * K + aCol];
    Bs[bRow * BN + bCol] = B[bRow * N + bCol];
    __syncthreads();
    A += BK;
    B += BK * N;
    #pragma unroll
    for (int d = 0; d < BK; d++) {
      float b = Bs[d * BN + threadCol];            // loaded once into a register...
      #pragma unroll
      for (int r = 0; r < TM; r++) acc[r] += As[(threadRow * TM + r) * BK + d] * b;  // ...reused TM times
    }
    __syncthreads();
  }
  for (int r = 0; r < TM; r++) C[(threadRow * TM + r) * N + threadCol] = acc[r];
}

// ---------- timing + checking ----------
template <typename F> float time_ms(F launch, int iters = 10) {
  launch(); cudaDeviceSynchronize();               // warm-up
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < iters; i++) launch();
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  return ms / iters;
}
float max_err(const float* d1, const float* d2, size_t n) {
  std::vector<float> a(n), b(n);
  cudaMemcpy(a.data(), d1, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), d2, n * sizeof(float), cudaMemcpyDeviceToHost);
  float m = 0; for (size_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

int main() {
  const int M = 4096, N = 4096, K = 4096;
  size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
  std::vector<float> hA(nA), hB(nB);
  for (auto& v : hA) v = rand() / (float)RAND_MAX - 0.5f;
  for (auto& v : hB) v = rand() / (float)RAND_MAX - 0.5f;
  float *A, *B, *C, *Ref;
  cudaMalloc(&A, nA * 4); cudaMalloc(&B, nB * 4); cudaMalloc(&C, nC * 4); cudaMalloc(&Ref, nC * 4);
  cudaMemcpy(A, hA.data(), nA * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB.data(), nB * 4, cudaMemcpyHostToDevice);
  double flops = 2.0 * M * N * K;

  // cuBLAS is column-major. Row-major C = A B  <=>  column-major C^T = B^T A^T, so swap A and B.
  cublasHandle_t h; cublasCreate(&h);
  float alpha = 1.f, beta = 0.f;
  float t_ref = time_ms([&] { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, Ref, N); });
  auto report = [&](const char* name, float ms, float err) {
    printf("%-18s %8.2f ms %6.2f TFLOPS %6.1f%% of cuBLAS  max_err=%.2e\n",
           name, ms, flops / (ms * 1e-3) / 1e12, 100.0 * t_ref / ms, err);
  };
  report("cuBLAS", t_ref, 0.f);

  dim3 b2(32, 32), g2((N + 31) / 32, (M + 31) / 32);
  float t1 = time_ms([&] { mm_naive<<<g2, b2>>>(A, B, C, M, N, K); });
  report("naive", t1, max_err(C, Ref, nC));
  float t2 = time_ms([&] { mm_tiled<<<g2, b2>>>(A, B, C, M, N, K); });
  report("smem tiled 32", t2, max_err(C, Ref, nC));

  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;   // 64*64/8 = 512 threads per block
  dim3 g3(N / BN, M / BM);
  float t3 = time_ms([&] { mm_reg<BM, BN, BK, TM><<<g3, BM * BN / TM>>>(A, B, C, M, N, K); });
  report("1D reg-blocked", t3, max_err(C, Ref, nC));
  return 0;
}
```

```bash
!nvcc -O3 -arch=sm_75 -std=c++17 mm.cu -lcublas -o mm && ./mm
```

`max_err` should be around `1e-4` or below (FP32 sums of 4096 terms in different orders). If it’s `1e+00` or `nan`, you have an indexing bug — debug on a `64 x 64` problem first.

## What to observe and why

1. **Naive** — ~10–15% of cuBLAS. Each thread streams $2K$ floats; L1/L2 catch some reuse, but the SMs mostly wait on memory.
   - Try swapping the index mapping (`row` from `threadIdx.x`, `col` from `threadIdx.y`). Now a warp reads 32 *different rows* of $A$ with stride $K$ — uncoalesced — and it gets several times slower. This is Boehm’s kernel 1 vs kernel 2.
2. **SMEM tiled** — ~2–3× faster than naive. Global loads dropped 32×, but now the inner loop does **2 shared-memory loads per FMA**; you’re bound by SMEM bandwidth and instruction issue.
3. **Register blocked** — another ~2×. Each `Bs` value is loaded once into a register and used `TM = 8` times, so the ratio of FMAs to SMEM loads goes from 1:2 to 8:9. This is the same reuse idea as tiling, one level down the hierarchy.
4. **cuBLAS** — uses 2-D register tiles, vectorized `float4` loads, double-buffered SMEM, and per-GPU autotuned tile shapes. On FP16 it would also use **tensor cores** (a separate ~8× jump — try `cublasGemmEx` with `CUDA_R_16F` as a stretch).

> [!WARNING] Benchmarking pitfalls you just avoided
> Warm-up launch (first launch pays module loading), averaging many iterations, timing with **events on the GPU** rather than `time.time()` on the CPU, and checking correctness *every* time you change a kernel. A fast wrong kernel is worth nothing.

## Step-by-step tasks

- [ ] All three kernels pass `max_err < 1e-3` at 4096 and at a non-multiple size (naive + tiled) like `1000 x 1000 x 1000`
- [ ] Filled a results table: kernel · ms · TFLOPS · % of cuBLAS
- [ ] Measured `mm_tiled` with `TILE = 16` vs `32` and explained the difference with the math lesson
- [ ] Added a **2-D register-blocked** kernel (each thread computes a `TM x TN = 8 x 8` sub-tile) following Boehm’s kernel 5 — this is the path to the challenge target
      */}),
      resources: [
        { title: "Simon Boehm — How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance", url: "https://siboehm.com/articles/22/CUDA-MMM", type: "article", note: "THE worklog: 10 kernels from naive to ~94% of cuBLAS, with the reasoning" },
        { title: "Aleksa Gordić — Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels", url: "https://www.aleksagordic.com/blog/matmul", type: "article", note: "the modern (Hopper) sequel: tensor cores, TMA, PTX/SASS" },
      ],
    },

    {
      id: "measure-roofline",
      title: "Lab: measure like an engineer — bandwidth, TFLOPS, and the roofline",
      kind: "lab",
      minutes: 120,
      runsOn: ["colab"],
      md: MD(function () {/*
A number without a reference point is meaningless. “My kernel takes 1.1 ms” — is that good? You only know once you compare it to **what the hardware could possibly do**.

## The two metrics

$$\text{effective bandwidth (GB/s)} = \frac{\text{bytes read} + \text{bytes written}}{t \cdot 10^9}$$

$$\text{throughput (TFLOP/s)} = \frac{\text{FLOPs}}{t \cdot 10^{12}}$$

Always count **minimum necessary** bytes/FLOPs for the operation (what an ideal kernel must do), not what your kernel happens to do — that way inefficiency shows up as a low percentage.

| Kernel | FLOPs | Min. bytes (FP32) | AI | Bound on T4 | Ceiling |
|---|---|---|---|---|---|
| vector add, $n$ | $n$ | $12n$ | 0.08 | memory | 320 GB/s |
| sum reduction, $n$ | $n$ | $4n$ | 0.25 | memory | 320 GB/s |
| row softmax $R \times C$ | $\approx 5RC$ | $8RC$ | ~0.6 | memory | 320 GB/s |
| SGEMM $4096^3$ | $1.37 \times 10^{11}$ | $2.0 \times 10^{8}$ | ~680 | compute | 8.1 TFLOPS (FP32, CUDA cores) |

## Plot your kernels on the roofline

::viz roofline

Pick the T4 preset and place each of your kernels: x = arithmetic intensity (from the table), y = your measured FLOP/s. Memory-bound kernels should sit *on or just under the diagonal*; your matmuls sit far below the flat roof — the gap is your optimization headroom.

> [!TIP] Realistic ceilings
> Nobody hits 100%. Well-written streaming kernels reach **80–90%** of datasheet bandwidth (use a `cudaMemcpy` device-to-device of 1 GB as your practical ceiling). T4 FP32 matmul rarely beats ~4–5 TFLOPS because the card power-throttles at 70 W below its boost clock. Measure `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -l 1` while benchmarking to see it.

## A reusable Python timing helper

Once kernels are callable from PyTorch (next lesson), use the same method Triton uses:

```python
import torch

def bench(fn, *args, warmup=5, iters=50):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record(); fn(*args); end[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start, end))
    return times[len(times) // 2]            # median ms

# or simply: from triton.testing import do_bench; ms = do_bench(lambda: fn(*args))
```

## First look at Nsight Compute

`ncu` profiles a single kernel launch in depth: achieved memory throughput, SM utilization, occupancy, and a built-in roofline chart.

```bash
!ncu --set full -k regex:mm_tiled -c 1 -o tiled_report ./mm
!ncu --import tiled_report.ncu-rep --page details | head -80
```

Look for **Memory Throughput %** vs **Compute (SM) Throughput %** in the *GPU Speed Of Light* section: whichever is near 100% is your bottleneck. (On some hosted runtimes `ncu` fails with `ERR_NVGPUCTRPERM` because counter access is disabled by the host; if so, do this on a rented cloud GPU — M13 covers profiling tools in depth.)

- [ ] Measured `cudaMemcpy` D2D bandwidth as your practical ceiling
- [ ] Table of all your kernels with % of practical ceiling (bandwidth for memory-bound, cuBLAS/peak for matmul)
- [ ] Placed each kernel on the roofline and wrote one sentence per kernel: what bounds it and what would move it
- [ ] (If `ncu` works) Compared *Memory Throughput* for `mm_naive` vs `mm_reg`
      */}),
      resources: [
        { title: "CUDA C++ Best Practices Guide", url: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/", type: "docs", note: "APOD workflow, effective bandwidth, occupancy — the official perf manual" },
        { title: "Nsight Compute Profiling Guide", url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/", type: "docs", note: "what each ncu section/metric means, including the roofline view" },
      ],
    },

    {
      id: "pytorch-extension",
      title: "Build: call your CUDA kernels from PyTorch (and Metal kernels from MLX on your Mac)",
      kind: "build",
      minutes: 120,
      runsOn: ["colab", "mac"],
      md: MD(function () {/*
Standalone `.cu` programs are great for learning, but real engines call kernels from Python on PyTorch tensors. vLLM’s `csrc/` and SGLang’s `sgl-kernel` are exactly this: CUDA kernels bound to PyTorch as custom ops.

## `load_inline`: compile CUDA from a Python string

```python
# Colab, T4 runtime.  !pip install ninja   (if the build complains)
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__inline__ __device__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__inline__ __device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
template <bool IsMax> __device__ float block_reduce(float v) {
  __shared__ float buf[32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = IsMax ? warp_max(v) : warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  v = (lane < (int)(blockDim.x >> 5)) ? buf[lane] : (IsMax ? -INFINITY : 0.f);
  v = IsMax ? warp_max(v) : warp_sum(v);
  __syncthreads();
  return v;
}
__global__ void softmax_rows_kernel(const float* x, float* y, int cols) {
  const float* row = x + (size_t)blockIdx.x * cols;
  float* out = y + (size_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, row[j]);
  m = block_reduce<true>(m);
  float s = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) s += __expf(row[j] - m);
  s = block_reduce<false>(s);
  float inv = 1.f / s;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) out[j] = __expf(row[j] - m) * inv;
}

torch::Tensor softmax_rows(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2 && x.is_contiguous(),
              "expected a contiguous 2-D float32 CUDA tensor");
  auto y = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();          // play nicely with PyTorch streams
  softmax_rows_kernel<<<x.size(0), 256, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), x.size(1));
  return y;
}
"""
cpp_src = "torch::Tensor softmax_rows(torch::Tensor x);"

ext = load_inline(name="my_kernels", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["softmax_rows"], extra_cuda_cflags=["-O3"], verbose=True)

x = torch.randn(4096, 4096, device="cuda")
torch.testing.assert_close(ext.softmax_rows(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-4)

from triton.testing import do_bench
for name, fn in [("mine", ext.softmax_rows), ("torch", lambda t: torch.softmax(t, dim=-1))]:
    ms = do_bench(lambda: fn(x))
    print(f"{name:6s} {ms:.3f} ms  {2 * x.numel() * 4 / (ms * 1e-3) / 1e9:.0f} GB/s")
```

The first call compiles (~1 minute) and caches the build under `~/.cache/torch_extensions`. Your kernel is now a normal Python function on tensors — you can drop it into your mini-engine from M10.

> [!NOTE] The production path
> `load_inline` is for experiments. Real projects build ahead of time with `setup.py`/`CUDAExtension` and register ops with `TORCH_LIBRARY` so `torch.compile` can see them — see the PyTorch *Custom C++ and CUDA Operators* tutorial. That’s the same mechanism vLLM uses for its `_C` ops.

- [ ] Your softmax passes `assert_close` and is within ~20% of `torch.softmax`
- [ ] Added your `mm_reg` kernel as a second function `matmul(A, B)` and compared with `A @ B`
- [ ] Deliberately passed a non-contiguous tensor (`x.t()`) and saw your `TORCH_CHECK` fire

## On your Mac: the same kernel in Metal with MLX (optional)

MLX lets you JIT a Metal kernel body from Python. You write only the body; MLX generates the signature from your inputs/outputs:

```python
import mlx.core as mx, time

source = """
    uint i = thread_position_in_grid.x;
    out[i] = a[i] + b[i];
"""
vadd_kernel = mx.fast.metal_kernel(name="vadd", input_names=["a", "b"],
                                   output_names=["out"], source=source)

def vadd(a, b):
    return vadd_kernel(inputs=[a, b], grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[a.shape], output_dtypes=[a.dtype])[0]

n = 1 << 26
a, b = mx.ones((n,)), mx.full((n,), 2.0)
mx.eval(vadd(a, b))                               # compile + warm-up
t0 = time.perf_counter()
for _ in range(20): mx.eval(vadd(a, b))
ms = (time.perf_counter() - t0) / 20 * 1e3
print(f"metal vadd: {ms:.2f} ms  {3 * n * 4 / (ms * 1e-3) / 1e9:.0f} GB/s")   # compare to your chip's bandwidth
```

`grid` is the *total* number of threads (Metal’s `dispatchThreads`), so no bounds check is needed when it equals `a.size`. Compare the GB/s to your chip’s unified-memory bandwidth (M2 ~100 GB/s, M3 Max ~400 GB/s). Porting the softmax kernel is a great exercise: `threadgroup float buf[32];`, `simd_max`/`simd_sum`, `threadgroup_barrier(mem_flags::mem_threadgroup)`.

- [ ] (Mac, optional) Ran the Metal vector add and computed % of your chip’s bandwidth
      */}),
      resources: [
        { title: "torch.utils.cpp_extension", url: "https://docs.pytorch.org/docs/stable/cpp_extension.html", type: "docs", note: "load_inline / CUDAExtension reference" },
        { title: "PyTorch — Custom C++ and CUDA Operators", url: "https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html", type: "docs", note: "the production way to register ops that work with torch.compile" },
        { title: "MLX — Custom Metal Kernels", url: "https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html", type: "docs", note: "mx.fast.metal_kernel for writing GPU kernels on your Mac" },
      ],
    },

    {
      id: "deep-dive",
      title: "Deep dive: Boehm’s matmul worklog, PMPP, GPU MODE — and Metal via the Muser book",
      kind: "deep",
      optional: true,
      minutes: 360,
      runsOn: ["colab", "mac"],
      md: MD(function () {/*
You now know enough to read the serious material productively. Pick the track that matches your goal.

## Track A — finish the matmul (recommended)

Work through **Simon Boehm’s “How to Optimize a CUDA Matmul Kernel”** and implement kernels 4–6 yourself before reading his code:

| Boehm kernel | Idea | What it teaches |
|---|---|---|
| 1 → 2 | global memory coalescing | warp-level access patterns |
| 3 | shared-memory cache blocking | the tiling you just did |
| 4 → 5 | 1-D then 2-D register blocktiling | reuse at the register level; arithmetic intensity per thread |
| 6 | vectorized `float4` loads + transposed `As` | fewer, wider memory instructions |
| 9 | autotuning tile sizes | why cuBLAS ships many kernels per GPU |
| 10 | warptiling | explicit warp-level hierarchy — the structure CUTLASS formalizes |

Then read **Aleksa Gordić’s “Inside NVIDIA GPUs”** for the modern picture: tensor cores (`mma`/`wgmma`), TMA, and why Hopper kernels look nothing like your T4 kernels.

## Track B — the textbook and the lectures

- **Programming Massively Parallel Processors (PMPP)** by Hwu, Kirk & El Hajj — the standard textbook. Chapters on memory, tiling, reduction, scan, convolution and sparse matmul map one-to-one to what you did here.
- **GPU MODE lectures** — a community lecture series (YouTube + GitHub) that follows PMPP and then goes further: Triton, CUTLASS, FlashAttention, quantization kernels, profiling. Watch lectures 1–8 alongside the book.
- **karpathy/llm.c** — GPT-2 training in raw C/CUDA. Read `train_gpt2.cu`: layernorm, softmax, attention, and matmul kernels written by people optimizing for real — an ideal “read production-grade CUDA” exercise.

## Track C — practice problems

- **LeetGPU** and **Tensara**: graded GPU problems with leaderboards. Do reduction, softmax, matmul, and then attention.

## Track D — the Mac/Metal path (optional)

The **Muser book (“How to Write an Inference Engine”)** builds a complete LLM inference engine on Apple Metal from scratch. Relevant chapters now:

- *Ch. 1 Why inference is a memory problem* — the bandwidth argument you’ve been measuring, done by hand.
- *Ch. 2 The Metal compute model* — threadgroups and SIMD-groups vs blocks and warps.
- *Ch. 16 Attention decode kernels* — a ladder of attention kernels, a great preview of M12.
- *Ch. 29 CUDA vs Metal: the differences that mattered.*

> [!REAL] What “good” looks like in industry
> Production GEMMs come from cuBLAS/cuBLASLt, CUTLASS, or DeepGEMM (FP8), and engines *select* among them. People who write kernels professionally spend most of their time on the non-GEMM ops (attention, MoE routing, quantized GEMV, fused norms, sampling) — where library coverage is thinner and fusion wins are large.

- [ ] Implemented Boehm kernel 5 (2-D blocktiling) and reached ≥ 60% of cuBLAS on the T4
- [ ] Watched GPU MODE lectures 1–4
- [ ] Read one CUDA kernel in `llm.c` and wrote a paragraph explaining its launch configuration and memory access pattern
      */}),
      resources: [
        { title: "GPU MODE lectures", url: "https://github.com/gpu-mode/lectures", type: "course", note: "best structured community course on CUDA/Triton/perf" },
        { title: "Programming Massively Parallel Processors (PMPP)", url: "https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0", type: "book", note: "the CUDA textbook; GPU MODE follows it" },
        { title: "karpathy/llm.c", url: "https://github.com/karpathy/llm.c", type: "repo", note: "readable production-grade CUDA kernels for a real transformer" },
        { title: "Muser book — Ch. 2 The Metal compute model", url: "https://highperformanceailab.com/muser-book/chapters/02-metal-compute-model.html", type: "book", note: "the Apple-GPU equivalent of this module" },
        { title: "Muser book — Ch. 29 CUDA vs Metal", url: "https://highperformanceailab.com/muser-book/chapters/29-cuda-versus-metal.html", type: "book", note: "explicit Metal ↔ CUDA mapping from people who shipped both" },
        { title: "Tensara", url: "https://tensara.org/", type: "practice", note: "GPU kernel challenges with head-to-head benchmarks" },
      ],
    },
  ],

  challenge: {
    title: "Kernel report: a tiled matmul at ≥ 50% of cuBLAS + a fused softmax bandwidth report",
    md: MD(function () {/*
In `course-work/m11/`, deliver a notebook (or `.cu` + `.py`) and a `REPORT.md`:

1. **Matmul.** Your best FP32 SGEMM kernel on a Colab T4 at $M = N = K = 4096$, reaching **≥ 50% of cuBLAS SGEMM** measured in the *same* process (2-D register blocking + `float4` loads usually gets you there). Include the full progression table (naive → coalesced → SMEM → 1-D → 2-D …) with ms, TFLOPS, % of cuBLAS, and one sentence per step explaining *why* it got faster.
2. **Softmax.** Your row-softmax kernel for FP32 shapes `(4096, 1024)`, `(4096, 4096)`, `(1024, 32768)`: correctness vs `torch.softmax`, time, effective bandwidth, and % of your measured D2D-copy ceiling, next to `torch.softmax` on the same shapes.
3. **Roofline.** One roofline figure (matplotlib or a screenshot of the viz) with every kernel placed on it, and a paragraph on what bounds each.
4. **PyTorch integration.** Both kernels callable via `load_inline` and verified with `torch.testing.assert_close`.
    */}),
    checklist: [
      "Best matmul ≥ 50% of cuBLAS SGEMM at 4096³ on a T4 (same process, median of ≥ 10 runs)",
      "Every kernel verified against a reference (`max_err` reported) including a non-multiple-of-tile shape where supported",
      "Softmax effective bandwidth reported for 3 shapes, with % of measured copy bandwidth",
      "Each optimization step explained in terms of memory traffic or reuse, not just “it got faster”",
      "Roofline plot with all kernels placed and bound identified",
      "Kernels callable from PyTorch via `load_inline`",
    ],
    stretch: "Write an FP16 tensor-core matmul with the WMMA API (`#include <mma.h>`, `wmma::mma_sync` on 16×16×16 fragments) and compare against `cublasGemmEx`/`torch.matmul` in FP16 — then explain the jump using the T4’s 65 TFLOPS FP16 tensor-core peak. Or: implement a single-pass online softmax and show the bandwidth gain for very wide rows.",
  },

  connects: MD(function () {/*
You can now read a CUDA kernel and predict its performance from its memory pattern. **M12** raises the abstraction: **Triton** lets you write the same tiled kernels in Python by thinking in blocks instead of threads, and you’ll use it to build the most important kernel in LLM inference — **FlashAttention**, which is just tiling + shared memory + the online softmax you previewed here. **M13** zooms out from single kernels to whole programs: when the GPU sits idle between tiny kernels (launch overhead), CUDA graphs and `torch.compile` matter more than any single kernel. Later, **M14** (quantization) is largely about GEMV/GEMM kernels that read fewer bytes — exactly the memory-bound regime you measured in the first demo.
  */}),

  interview: [
    "Explain the CUDA thread hierarchy (thread, warp, block, grid) and how it maps onto SMs. Why must block sizes be multiples of 32?",
    "What is memory coalescing? Given a kernel where thread `t` reads `x[t * stride]`, how does performance change as `stride` goes from 1 to 32?",
    "Walk me through a parallel sum reduction. Why are warp shuffles faster than a shared-memory tree for the last 32 elements?",
    "A matmul of $M = N = K = 4096$ takes 40 ms on a GPU. What TFLOP/s is that? Is it good for a T4? For an H100?",
    "Why does shared-memory tiling speed up matrix multiplication? Derive the arithmetic intensity as a function of tile size.",
    "What is a shared-memory bank conflict and how do you avoid one when transposing a tile?",
    "Your elementwise kernel runs at 90% of peak bandwidth. Your manager wants it 2× faster. What do you tell them, and what could actually help?",
    "What does `__syncthreads()` do, and what happens if you call it inside a branch only some threads take?",
  ],

  resources: [
    { title: "Simon Boehm — How to Optimize a CUDA Matmul Kernel", url: "https://siboehm.com/articles/22/CUDA-MMM", type: "article", note: "the single best kernel-optimization tutorial; follow it kernel by kernel" },
    { title: "GPU MODE lectures", url: "https://github.com/gpu-mode/lectures", type: "course", note: "structured video course from CUDA basics to FlashAttention" },
    { title: "Programming Massively Parallel Processors (PMPP)", url: "https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0", type: "book", note: "the textbook behind GPU MODE" },
    { title: "CUDA Programming Guide", url: "https://docs.nvidia.com/cuda/cuda-programming-guide/", type: "docs", note: "official reference for the execution and memory model" },
    { title: "CUDA C++ Best Practices Guide", url: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/", type: "docs", note: "coalescing, shared memory, occupancy — official optimization guide" },
    { title: "LeetGPU", url: "https://leetgpu.com/", type: "practice", note: "write and grade CUDA kernels in the browser — no GPU needed" },
    { title: "Modal GPU Glossary", url: "https://modal.com/gpu-glossary", type: "docs", note: "quick plain-English lookups for GPU terms" },
    { title: "wafer-ai — AI Performance Engineering resources", url: "https://github.com/wafer-ai/gpu-perf-engineering-resources", type: "repo", note: "curated primary-source reading list, request → GPU → kernels → engines" },
    { title: "Inference Engineering (Baseten) — Ch. 3–4.1", url: "Inference%20Engineering.pdf", type: "book", note: "GPU architecture, CUDA, kernel sources/selection/fusion" },
    { title: "MLX — Custom Metal Kernels", url: "https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html", type: "docs", note: "Mac equivalent: write Metal kernels from Python" },
    { title: "Muser book — How to Write an Inference Engine", url: "https://highperformanceailab.com/muser-book/", type: "book", note: "optional: a full inference engine on Apple Metal, byte by byte" },
  ],
});
