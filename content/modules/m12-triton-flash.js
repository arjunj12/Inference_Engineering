Course.module({
  id: "m12-triton-flash",
  title: "Triton & FlashAttention: fused kernels in Python, then the kernel that made long context possible",
  short: "Triton & FlashAttention",
  tagline: "Write fused GPU kernels in Triton, derive online softmax, and build a FlashAttention-style forward kernel that beats naive PyTorch attention in speed and memory.",
  hours: 18,
  level: "core",
  runsOn: ["colab"],
  tags: ["triton", "flashattention", "online-softmax", "fusion", "attention-kernels"],

  goal: MD(function () {/*
You finish with a ~60-line **Triton** kernel — your own FlashAttention forward pass with a causal mask — and a benchmark on a free Colab T4 that looks like this:

```text
causal attention fwd, B=1 H=16 D=64 fp16, Tesla T4
  N       naive torch            your Triton FA         SDPA (mem-efficient)
          ms      peak mem       ms      peak mem       ms      peak mem
  1024    1.9     0.17 GB        0.9     0.01 GB        0.8     0.01 GB
  4096   27.0     2.7  GB        9.8     0.03 GB        8.9     0.03 GB
  8192  108      10.7  GB       37       0.07 GB       34       0.07 GB
 16384    OOM                  146       0.13 GB      133       0.13 GB
max |out - SDPA| = 9.8e-04   ✓
```

(Illustrative — your numbers will differ.) Naive attention materializes an $N \times N$ score matrix per head and runs out of memory at 16k tokens; your kernel never materializes it, so memory grows **linearly** with $N$ and it’s several times faster — within shouting distance of PyTorch’s built-in kernel. You’ll be able to derive, on a whiteboard, *why* it works (online softmax) and *why* it’s fast (IO-awareness).

::viz online-softmax
  */}),
  demo: { viz: "kernel-fusion", params: {} },

  why: MD(function () {/*
**FlashAttention is the most consequential kernel of the LLM era.** Long context windows (128k–1M tokens), fast prefill, and cheap training all depend on attention kernels that never write the $N \times N$ matrix to memory. Every serving engine (vLLM, SGLang, TensorRT-LLM) is built around a family of these kernels — FlashAttention 2/3/4, FlashInfer, and paged/decode variants — and choosing and configuring them is daily work (Inference Engineering (Baseten) ch. 2.5 and 4.1: “FA3 on Hopper, FA4 on Blackwell”).

**Triton** is how most people write custom kernels today: `torch.compile` generates Triton code for you, vLLM and SGLang ship many Triton kernels (MoE, fused norms, attention fallbacks), and it’s the language of LLM kernel interview take-homes. Knowing Triton + the online-softmax derivation is the most direct path from “understands transformers” to “can make them fast”.
  */}),

  prereqs: [
    {
      title: "M11 in one paragraph",
      skipIf: "you finished M11 (CUDA from scratch)",
      md: MD(function () {/*
GPUs run thousands of threads grouped into blocks; each block has fast on-chip **shared memory**; everything else lives in slow, big **HBM**. Kernels that do little math per byte (elementwise, softmax, norms) are **memory-bound** — their speed is bytes ÷ bandwidth — so the win is **moving fewer bytes**, e.g. by **fusing** several ops into one kernel. Matmuls become compute-bound only if you **tile**: load a block of data into shared memory once and reuse it many times. Triton automates the tiling mechanics; FlashAttention applies tiling to attention.
      */}),
    },
    {
      title: "Attention in 6 lines (from M03)",
      skipIf: "you can write causal self-attention in PyTorch from memory",
      math: true,
      md: MD(function () {/*
For one head with queries $Q$, keys $K$, values $V$ (each $N \times d$):

$$S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \qquad P = \text{softmax}_{\text{row}}(S + \text{mask}), \qquad O = PV \in \mathbb{R}^{N \times d}$$

```python
def attention(q, k, v, causal=True):              # (B, H, N, D)
    N, D = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-2, -1)) / D ** 0.5      # (B, H, N, N)  <- the N^2 problem
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), 1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v
```

FLOPs: two matmuls of $2N^2d$ each → $4N^2d$ per head (half that with a causal mask, if the kernel skips masked blocks).
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: the attention memory wall, and Triton’s fused softmax",
      kind: "demo",
      minutes: 60,
      runsOn: ["colab"],
      md: MD(function () {/*
## Part 1 — hit the wall

On a Colab T4 runtime, compare naive attention with PyTorch’s fused `scaled_dot_product_attention` (SDPA) as sequence length grows:

```python
import torch, torch.nn.functional as F
from triton.testing import do_bench

def naive(q, k, v):
    N, D = q.shape[-2:]
    s = (q @ k.transpose(-2, -1)) * D ** -0.5
    s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device="cuda"), 1), float("-inf"))
    return torch.softmax(s.float(), dim=-1).half() @ v

def measure(fn, q, k, v):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        ms = do_bench(lambda: fn(q, k, v))
    except torch.cuda.OutOfMemoryError:
        return "OOM", "-"
    return f"{ms:7.1f}", f"{(torch.cuda.max_memory_allocated() - base) / 1e9:5.2f} GB"

B, H, D = 1, 16, 64
for N in [1024, 2048, 4096, 8192, 16384]:
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(N, "naive:", *measure(naive, q, k, v), "| sdpa:", *measure(sdpa, q, k, v))
```

Watch naive attention’s extra memory grow **4× every time $N$ doubles** (it stores $B \cdot H \cdot N^2$ scores, in FP16 *and* FP32) until it OOMs, while SDPA’s stays tiny. At $N = 16384$, the score matrix alone is $16 \times 16384^2 \times 2$ bytes ≈ **8.6 GB** per layer.

> [!NOTE] Which kernel does SDPA pick on a T4?
> SDPA dispatches to one of several backends. The FlashAttention-2 backend needs Ampere (sm_80) or newer, so on a T4 (sm_75) PyTorch uses the **memory-efficient** backend (a CUTLASS kernel based on the same tiling + online-softmax idea). On an L4/A100/H100 it uses FlashAttention. You can force one with `torch.nn.attention.sdpa_kernel(...)`.

## Part 2 — run the official Triton fused softmax tutorial

Triton ships with PyTorch on Colab. Fetch the tutorial matching your installed version and run it:

```python
import triton; print(triton.__version__)          # e.g. 3.4.0  ->  release/3.4.x
```

```bash
!wget -q https://raw.githubusercontent.com/triton-lang/triton/release/3.4.x/python/tutorials/02-fused-softmax.py
!python 02-fused-softmax.py
```

(Replace `3.4.x` with your version’s branch.) It benchmarks a Triton softmax against `torch.softmax` and a naive PyTorch softmax, printing GB/s across row widths. The Triton kernel is ~20 lines of *Python* and matches or beats the hand-written CUDA in PyTorch. By the end of lesson 3 you’ll have written it yourself.

> [!WARNING] Triton on a T4
> Triton officially targets compute capability 8.0+; on the T4 (7.5) the kernels in this module (FP16 inputs, modest block sizes) work in practice, but newest-feature tutorials (TMA, FP8, tensor descriptors) will not. If you hit a compiler error, switch to an L4 or A100 runtime or run on [LeetGPU](https://leetgpu.com) (Triton supported). Never use BF16 on a T4 — it has no BF16 tensor cores; use FP16.

- [ ] Ran Part 1 and recorded the $N$ at which naive attention OOMs on your runtime
- [ ] Computed by hand the score-matrix bytes at that $N$ — does it explain the OOM?
- [ ] Ran the fused softmax tutorial and noted peak GB/s for Triton vs `torch.softmax` (compare to the T4’s 320 GB/s)
      */}),
      resources: [
        { title: "Triton tutorial — Fused Softmax", url: "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html", type: "docs", note: "the official version of what you’ll build in lesson 3" },
        { title: "torch.nn.functional.scaled_dot_product_attention", url: "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", type: "docs", note: "PyTorch’s fused attention entry point and its backends" },
      ],
    },

    {
      id: "triton-model",
      title: "The Triton programming model: programs over blocks, not threads",
      kind: "concept",
      minutes: 90,
      runsOn: ["colab"],
      md: MD(function () {/*
In CUDA you write what **one thread** does and hand-manage shared memory, warps and synchronization. In Triton you write what **one program instance** does to a **block of data** (a small tensor), and the compiler decides how to spread that across threads, vectorize loads, use shared memory and tensor cores.

| | CUDA | Triton |
|---|---|---|
| Unit you write | one thread | one *program* operating on blocks (tiles) |
| Index | `blockIdx.x * blockDim.x + threadIdx.x` | `tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)` |
| Memory ops | per-element pointers | `tl.load(ptrs, mask=...)` of a whole block of pointers |
| Shared memory, sync, coalescing | manual | automatic |
| Matmul on tensor cores | WMMA/MMA intrinsics | `tl.dot(a, b)` |
| Tuning knobs | block dims, SMEM size | `BLOCK_*` constexprs, `num_warps`, `num_stages`, `@triton.autotune` |
| Language | C++ | a Python subset, JIT-compiled to PTX via MLIR |

## Vector add, the Triton way

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                    # which block am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)       # a vector of BLOCK indices
    mask = offs < n                                # guard the ragged last block
    x = tl.load(x_ptr + offs, mask=mask)           # load a whole block at once
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)   # number of program instances
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

x, y = torch.rand(1 << 26, device="cuda"), torch.rand(1 << 26, device="cuda")
torch.testing.assert_close(add(x, y), x + y)
from triton.testing import do_bench
ms = do_bench(lambda: add(x, y)); print(f"{ms:.3f} ms  {3 * x.numel() * 4 / (ms * 1e-3) / 1e9:.0f} GB/s")
```

Key ideas:

- **Pointers are tensors.** `x_ptr + offs` is a block of 1024 addresses; `tl.load` fetches them all. For 2-D tiles you build a 2-D block of pointers by broadcasting: `ptr + rows[:, None] * stride + cols[None, :]`.
- **Masks replace `if (i < n)`.** Out-of-range lanes are skipped on store; on load they take the `other=` value (use `other=float("-inf")` before a max, `0.0` before a sum).
- **`tl.constexpr`** arguments are compile-time constants. Block sizes must be constexpr (and powers of two), so each distinct value compiles a new specialized kernel.
- **Kernels are JIT-compiled and cached** per signature (dtypes, constexprs, divisibility-by-16 of int args).

## Autotuning

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (256, 1024, 4096) for w in (2, 4, 8)],
    key=["n"],                        # re-tune when n changes
)
@triton.jit
def add_kernel_tuned(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...                                # same body; call it WITHOUT passing BLOCK
```

The first call benchmarks every config and caches the winner — exactly how Inductor and vLLM pick Triton kernel configs per GPU.

## Seeing the machine code

```python
k = add_kernel[(1,)](x, y, torch.empty_like(x), x.numel(), BLOCK=1024)
print(k.asm.keys())                # ttir, ttgir, llir, ptx, cubin
print(k.asm["ptx"][:2000])         # look for ld.global.v4.f32: 16-byte vectorized loads, for free
```

> [!INTUITION] What Triton gives up
> Less control: you can’t pick exactly which thread does what, specialize warps (FA3-style), or hand-schedule async copies (newer Triton versions and Gluon are adding some of this). For ~80% of kernels you lose little and gain 10× productivity; the last 10–20% on Hopper/Blackwell is where CUDA/CUTLASS/ThunderKittens still win.

- [ ] Vector add in Triton reaches the same GB/s as your CUDA version from M11
- [ ] Added `@triton.autotune`, printed `add_kernel_tuned.best_config`
- [ ] Found the vectorized loads (`ld.global.v4`) in the PTX
- [ ] (Optional, fun) Did the first 4 puzzles of Sasha Rush’s **Triton-Puzzles**
      */}),
      resources: [
        { title: "Triton tutorial — Vector Addition", url: "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html", type: "docs", note: "official first kernel, with benchmark" },
        { title: "Triton programming guide — introduction", url: "https://triton-lang.org/main/programming-guide/chapter-1/introduction.html", type: "docs", note: "the block-level programming model and why it exists" },
        { title: "Triton-Puzzles (Sasha Rush)", url: "https://github.com/srush/Triton-Puzzles", type: "practice", note: "puzzles that drill masks, broadcasting and tl.dot — runs in Colab" },
      ],
    },

    {
      id: "fused-softmax",
      title: "Build: fused softmax in Triton — kernel fusion, measured",
      kind: "build",
      minutes: 120,
      runsOn: ["colab"],
      md: MD(function () {/*
## Why fusion matters

Here is softmax written as separate PyTorch ops. Each line is (at least) one kernel that reads its inputs from HBM and writes its output back:

```python
def naive_softmax(x):                   # x: (M, N)
    x_max = x.max(dim=1)[0]             # read MN,  write M
    z = x - x_max[:, None]              # read MN + M, write MN
    num = torch.exp(z)                  # read MN,  write MN
    den = num.sum(dim=1)                # read MN,  write M
    return num / den[:, None]           # read MN + M, write MN
```

That’s ~$5MN$ reads + $3MN$ writes. A **fused** kernel reads each row once into on-chip memory, does everything there, and writes once: $MN$ reads + $MN$ writes → **~4× less traffic → ~4× faster** for a memory-bound op.

::viz kernel-fusion

## The kernel: one program per row

```python
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)                          # BLOCK >= n_cols (power of two)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                           # numerically-safe shift
    num = tl.exp(x)                                     # masked lanes: exp(-inf) = 0
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

def softmax(x):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    y = torch.empty_like(x)
    softmax_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK=BLOCK, num_warps=num_warps)
    return y
```

The whole row lives in registers for the duration of the program — that’s the fusion. (For rows too large for registers, you’d loop over column chunks — and to do it in one pass you need the online softmax of lesson 4.)

## Test and benchmark

```python
from triton.testing import do_bench
torch.manual_seed(0)
for N in [256, 1024, 4096, 16384]:
    x = torch.randn(4096, N, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9   # ideal IO: read + write once
    t_tri, t_torch, t_naive = (do_bench(lambda f=f: f(x)) for f in (softmax, lambda t: torch.softmax(t, 1), naive_softmax))
    print(f"N={N:6d}  triton {gbps(t_tri):5.0f} GB/s | torch {gbps(t_torch):5.0f} | naive {gbps(t_naive):5.0f}")
```

Expect Triton ≈ `torch.softmax` (both near the T4’s practical ~250–280 GB/s) and the naive version ~3–4× worse.

## The compiler can do this too

`torch.compile` generates Triton kernels for exactly this pattern. See for yourself:

```python
import os; os.environ["TORCH_LOGS"] = "output_code"   # set BEFORE importing torch in a fresh runtime
compiled = torch.compile(naive_softmax)
compiled(torch.randn(4096, 1024, device="cuda"))     # prints the generated @triton.jit kernel
```

You’ll see a single fused kernel with a `tl.max` and `tl.sum` reduction — structurally like yours. This is why M13 treats `torch.compile` as the first thing to try, and hand-written kernels as what you write when the compiler can’t see the pattern (FlashAttention is the canonical example: compilers don’t invent online softmax).

- [ ] Triton softmax passes `assert_close` for all widths (including a non-power-of-two like `N = 1000`)
- [ ] Table of GB/s for Triton / torch / naive across widths; explained naive’s gap with the traffic count above
- [ ] Wrote a fused **scale + causal-mask + softmax** kernel (inputs: scores and a scale; mask where `col > row`) and verified vs PyTorch — this is the “unfused attention” building block
- [ ] Read the Inductor-generated kernel for `naive_softmax` and matched each line to your kernel
      */}),
      resources: [
        { title: "Triton tutorial — Fused Softmax", url: "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html", type: "docs", note: "official version with occupancy-aware launch" },
        { title: "Horace He — Making Deep Learning Go Brrrr From First Principles", url: "https://horace.io/brrr_intro.html", type: "article", note: "why fusion wins: memory-bound vs compute-bound vs overhead" },
      ],
    },

    {
      id: "online-softmax-math",
      title: "Math: deriving online softmax, step by step",
      kind: "math",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
FlashAttention rests on one small algebraic trick. Let’s build it from the ground up.

> [!PREREQ] Exponential rules you need
> - $e^{a+b} = e^a \cdot e^b$ and therefore $e^{a-b} = e^a / e^b$.
> - $e^0 = 1$, $e^x > 0$ always, $e^{-\infty} = 0$.
> - $e^x$ explodes fast: $e^{89}$ overflows FP32 (max ≈ $3.4 \times 10^{38}$); $e^{12}$ already overflows FP16 (max 65504).
> - $\ln$ undoes $\exp$: $\ln(e^x) = x$. Also $e^x = 2^{x \log_2 e}$ — GPUs compute `exp2` natively, which is why kernels multiply by $\log_2 e \approx 1.4427$.

::viz exp-log

## Step 1 — the max trick (safe softmax)

$$\text{softmax}(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}$$

Multiply top and bottom by $e^{-m}$ for any constant $m$:

$$\frac{e^{x_j} e^{-m}}{\sum_k e^{x_k} e^{-m}} = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}$$

Same value. Choosing $m = \max_k x_k$ makes every exponent $\le 0$, so every term is in $(0, 1]$ — no overflow. Cost: an extra pass over the data to find $m$ first. Standard (“safe”) softmax is therefore **3 passes**: max, sum, normalize.

## Step 2 — can we find max and sum in one pass?

Scan left to right, keeping a running max $m$ and a running sum $\ell$ of exponentials *relative to the current max*. When a new element $x_j$ arrives:

$$m_{\text{new}} = \max(m, x_j)$$

The old sum was computed relative to the old max: $\ell = \sum_{\text{seen}} e^{x_k - m}$. To express it relative to the new max, use the exponent rule:

$$e^{x_k - m_{\text{new}}} = e^{x_k - m} \cdot e^{m - m_{\text{new}}}$$

so the whole old sum just gets **multiplied by one correction factor**:

$$\ell_{\text{new}} = \ell \cdot e^{\,m - m_{\text{new}}} + e^{\,x_j - m_{\text{new}}}$$

If the max didn’t change, the factor is $e^0 = 1$. If it grew, the factor shrinks the old sum appropriately. Initialize $m = -\infty$, $\ell = 0$. That’s **online softmax** (Milakov & Gimelshein, 2018).

## Step 3 — do it per block, not per element

GPUs process tiles. For a block $B$ of new scores, compute the block max, then apply the same rescaling once per block:

$$m_{\text{new}} = \max\!\left(m, \max_{i \in B} x_i\right), \qquad \ell_{\text{new}} = \ell \cdot e^{\,m - m_{\text{new}}} + \sum_{i \in B} e^{\,x_i - m_{\text{new}}}$$

## Worked example

Scores $x = [2, 4, 1, 5]$ in two blocks $[2, 4]$ and $[1, 5]$:

| step | block | $m$ | correction $e^{m_{\text{old}} - m_{\text{new}}}$ | $\ell$ |
|---|---|---|---|---|
| start | — | $-\infty$ | — | 0 |
| 1 | [2, 4] | 4 | $e^{-\infty} = 0$ | $e^{-2} + e^{0} = 0.1353 + 1 = 1.1353$ |
| 2 | [1, 5] | 5 | $e^{4-5} = 0.3679$ | $1.1353 \times 0.3679 + e^{-4} + e^{0} = 0.4177 + 0.0183 + 1 = 1.4360$ |

Check directly: $e^{2-5} + e^{4-5} + e^{1-5} + e^{5-5} = 0.0498 + 0.3679 + 0.0183 + 1 = 1.4360$ ✓. So $\text{softmax}(x)_4 = 1/1.4360 = 0.696$.

Step through it interactively — watch the running max jump and the running sum get rescaled:

::viz online-softmax

## Step 4 — carry the output along too (this *is* FlashAttention)

In attention we don’t want the probabilities themselves, we want $O = \sum_j p_j v_j$. Keep an **unnormalized** output accumulator $o = \sum_{\text{seen}} e^{x_k - m} v_k$. It was computed relative to the same max, so it gets the **same correction**:

$$o_{\text{new}} = o \cdot e^{\,m - m_{\text{new}}} + \sum_{i \in B} e^{\,x_i - m_{\text{new}}} \, v_i$$

and at the very end, divide once: $O = o / \ell$. We never needed all scores at the same time — only one block of them plus three running quantities ($m$, $\ell$, $o$) per query row. Memory: $O(N)$ instead of $O(N^2)$.

## Verify it in 15 lines of PyTorch

```python
import torch
torch.manual_seed(0)
N, d, Bk = 1000, 64, 128
q, K, V = torch.randn(d), torch.randn(N, d), torch.randn(N, d)
ref = torch.softmax(K @ q / d ** 0.5, dim=0) @ V

m, l, o = torch.tensor(float("-inf")), torch.tensor(0.0), torch.zeros(d)
for s in range(0, N, Bk):
    x = K[s:s + Bk] @ q / d ** 0.5               # scores for this block only
    m_new = torch.maximum(m, x.max())
    corr = torch.exp(m - m_new)
    p = torch.exp(x - m_new)
    l = l * corr + p.sum()
    o = o * corr + p @ V[s:s + Bk]
    m = m_new
print(torch.allclose(o / l, ref, atol=1e-5))     # True
```

> [!MATH] Two more identities you’ll meet in kernels
> - **Log-sum-exp**: $\text{LSE} = m + \ln \ell$. FlashAttention saves this one number per row so the backward pass can recompute $P = e^{S - \text{LSE}}$ without storing $P$.
> - **Merging two partial results** $(m_1, \ell_1, o_1)$ and $(m_2, \ell_2, o_2)$ computed on *different* key ranges: $m = \max(m_1, m_2)$, $\ell = \ell_1 e^{m_1 - m} + \ell_2 e^{m_2 - m}$, $o = o_1 e^{m_1 - m} + o_2 e^{m_2 - m}$. This is how split-K / FlashDecoding (lesson 7) combines work done in parallel.

- [ ] Reproduced the worked example by hand, then with the PyTorch snippet
- [ ] Explained in one sentence why the correction factor is $e^{m_{\text{old}} - m_{\text{new}}}$ and why it’s $\le 1$
- [ ] Modified the snippet to split keys into 2 halves processed independently, then merged with the formula above
      */}),
      resources: [
        { title: "Milakov & Gimelshein — Online normalizer calculation for softmax", url: "https://arxiv.org/abs/1805.02867", type: "paper", note: "the original online softmax paper (4 pages)" },
      ],
    },

    {
      id: "flash-attention",
      title: "FlashAttention: tiling + online softmax + recomputation (and FA2, FA3, FA4)",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
## The problem: attention is IO-bound, not FLOP-bound

Standard attention runs as separate kernels, each round-tripping an $N \times N$ matrix through HBM:

1. $S = QK^\top$: read $Q, K$, **write $S$** ($N^2$)
2. $P = \text{softmax}(S)$: **read $S$, write $P$** ($2N^2$)
3. $O = PV$: **read $P$**, read $V$, write $O$

Inference Engineering (Baseten) ch. 2.4 works this out for $N = 4096$, $d = 128$, FP16: ~138 MB moved for ~8.6 GFLOP → **arithmetic intensity ≈ 62**, far below the H100’s ridge of ~295. The GPU spends its time moving $S$ and $P$, which nobody even wants as output. And memory is $O(N^2)$ per head, which is what killed naive attention in lesson 1.

## The FlashAttention algorithm (forward)

Tile $Q$ into row-blocks of size $B_r$ and $K, V$ into column-blocks of size $B_c$ that fit in SRAM (shared memory/registers). For each query block (one GPU thread block each, in FA2):

```text
load Q_i (B_r x d) into SRAM
m = -inf, l = 0, acc = 0                     # per query row, in registers
for each key/value block j:                  # causal: only j <= i
    load K_j, V_j into SRAM
    S_ij = Q_i @ K_j^T * scale               # B_r x B_c, stays on chip
    (apply causal mask on the diagonal block)
    m_new = max(m, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l   = l * exp(m - m_new) + rowsum(P_ij)
    acc = acc * exp(m - m_new) + P_ij @ V_j  # tensor-core matmul
    m = m_new
O_i = acc / l ;  write O_i  (and LSE_i = m + log l for backward)
```

That’s lesson 4’s math, one tile at a time. $S$ and $P$ exist only as small tiles on chip. HBM traffic is essentially: read $Q$ once, read $K, V$ once per query block (and concurrent query blocks of the same head share them through L2), write $O$ once.

| | Standard attention | FlashAttention |
|---|---|---|
| Extra memory | $O(N^2)$ per head | $O(N)$ (just $m$, $\ell$ / LSE) |
| HBM traffic (paper’s IO analysis) | $\Theta(Nd + N^2)$ | $\Theta(N^2 d^2 / M)$, $M$ = SRAM size |
| FLOPs | $4N^2 d$ | same (forward) — it’s *exact*, not an approximation |
| Kernels | 3+ | 1 |

With realistic $d$ and $M$, $d^2/M \ll 1$, so FlashAttention moves many times fewer bytes — the forward becomes compute-bound and runs near matmul speed.

## Recomputation in the backward pass

Training needs $P$ for gradients. Instead of storing it ($O(N^2)$), FlashAttention stores only $O$ and the per-row LSE, and **recomputes** $S$ and $P$ tile by tile in the backward pass. More FLOPs, far fewer bytes — and since the kernel is IO-bound, it’s a net win. “Trade compute for memory traffic” is the general lesson.

## The family tree

| Version | Year / target | Key ideas |
|---|---|---|
| **FlashAttention** (Dao et al.) | 2022, A100 | tiling + online softmax + recomputation; IO-aware analysis; 2–4× faster, linear memory |
| **FlashAttention-2** | 2023, A100 | parallelize over sequence length too (one block per $Q$ tile), fewer non-matmul FLOPs (rescale once at the end), split work across warps by $Q$ rows to avoid SMEM communication → ~2× FA1, 50–73% of A100 peak |
| **FlashAttention-3** | 2024, **H100** | Hopper-specific: asynchronous **wgmma** tensor-core instructions, **TMA** bulk copies, **warp specialization** (producer warps load, consumer warps compute), ping-pong scheduling to overlap softmax with matmuls, FP8 support → ~75% of H100 peak in FP16 |
| **FlashAttention-4** | 2025, **Blackwell** | rewritten for B200-class GPUs (tcgen05 tensor cores, tensor memory); different code again — kernels are hardware-specific |

> [!REAL] What engines actually do
> vLLM and SGLang choose an **attention backend** per GPU and workload: FlashAttention (2/3), FlashInfer, Triton fallbacks, or TRT-LLM kernels on Blackwell. Prefill uses FA-style kernels (compute-bound); decode uses paged, split-K kernels (memory-bound) — next lessons. As the Baseten book puts it: use FA3 on Hopper and FA4 on Blackwell; kernels are not portable across generations.

- [ ] Explained to yourself why FA does the *same* FLOPs yet runs faster
- [ ] Computed standard-attention bytes for $N = 32768$, $d = 128$, 32 heads in FP16 — would $S$ even fit in an H100’s 80 GB?
- [ ] Read Sections 1–3 of the FlashAttention paper (Algorithm 1 should now read like your pseudocode)
      */}),
      resources: [
        { title: "FlashAttention (Dao et al., 2022)", url: "https://arxiv.org/abs/2205.14135", type: "paper", note: "IO-aware exact attention — the original" },
        { title: "FlashAttention-2 (Dao, 2023)", url: "https://arxiv.org/abs/2307.08691", type: "paper", note: "better parallelism and work partitioning" },
        { title: "FlashAttention-3 (Shah et al., 2024)", url: "https://arxiv.org/abs/2407.08608", type: "paper", note: "Hopper: async wgmma/TMA, warp specialization, FP8" },
        { title: "Dao-AILab/flash-attention", url: "https://github.com/Dao-AILab/flash-attention", type: "repo", note: "the production kernels (FA2/FA3/FA4)" },
      ],
    },

    {
      id: "build-flash",
      title: "Build: a FlashAttention forward kernel in Triton (causal)",
      kind: "build",
      minutes: 240,
      runsOn: ["colab"],
      md: MD(function () {/*
This is a simplified version of Triton’s official `06-fused-attention` tutorial (which now carries Hopper/Blackwell-specific machinery). It implements exactly the pseudocode from the previous lesson. FP16 inputs, FP32 accumulation, head dim ∈ {16, 32, 64, 128}.

```python
import torch, triton, triton.language as tl

@triton.jit
def flash_fwd(Q, K, V, O, sm_scale,
              stride_qh, stride_qm, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_om,
              N_CTX,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)                 # which block of queries
    bh = tl.program_id(1)                      # which (batch * head)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # load the Q tile once; it stays on chip for the whole loop
    q = tl.load(Q + bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # unnormalized output
    qk_scale = sm_scale * 1.44269504                             # fold in log2(e): use exp2

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)          # skip blocks entirely above the diagonal
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        # K tile loaded transposed: (HEAD_DIM, BLOCK_N)
        k = tl.load(K + bh * stride_kh + cols[None, :] * stride_kn + offs_d[:, None],
                    mask=cols[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k) * qk_scale                              # (BLOCK_M, BLOCK_N) on tensor cores
        valid = cols[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        # online softmax update (lesson 4, in base 2)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)                              # correction for old sum/output
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + bh * stride_vh + cols[:, None] * stride_vn + offs_d[None, :],
                    mask=cols[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)                           # P_ij @ V_j
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attn(q, k, v, causal=True, BLOCK_M=64, BLOCK_N=64):
    B, H, N, D = q.shape                                          # (B, H, N, D) fp16
    assert D in (16, 32, 64, 128) and q.dtype == torch.float16
    q, k, v = (t.contiguous().view(B * H, N, D) for t in (q, k, v))
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    flash_fwd[grid](q, k, v, o, D ** -0.5,
                    q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                    v.stride(0), v.stride(1), o.stride(0), o.stride(1),
                    N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
                    num_warps=4, num_stages=2)
    return o.view(B, H, N, D)
```

Map it to the pseudocode: `q` = $Q_i$ in SRAM; the loop walks $K_j, V_j$; `m_i`, `l_i`, `acc` are $m$, $\ell$, $o$; `alpha` is the correction factor $2^{m_{\text{old}} - m_{\text{new}}}$ (base 2 because we pre-multiplied by $\log_2 e$).

## Correctness first

```python
import torch.nn.functional as F
torch.manual_seed(0)
for (B, H, N, D) in [(1, 2, 128, 64), (2, 8, 1000, 64), (1, 4, 2048, 128)]:
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    for causal in (False, True):
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = flash_attn(q, k, v, causal=causal)
        err = (out - ref).abs().max().item()
        print(B, H, N, D, "causal" if causal else "full", f"max_err={err:.2e}")
        assert err < 2e-2, "mismatch!"
```

Include $N = 1000$ (not a multiple of the block size) — masking bugs hide at ragged edges. FP16 outputs typically differ from SDPA by ~$10^{-3}$.

## Then speed and memory

```python
from triton.testing import do_bench

def naive(q, k, v):
    N, D = q.shape[-2:]
    s = (q @ k.transpose(-2, -1)) * D ** -0.5
    s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device="cuda"), 1), float("-inf"))
    return torch.softmax(s.float(), dim=-1).half() @ v

B, H, D = 1, 16, 64
print(f"{'N':>6} | {'naive ms':>9} {'GB':>5} | {'triton ms':>9} {'TFLOPS':>6} | {'sdpa ms':>8}")
for N in [1024, 2048, 4096, 8192, 16384]:
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    flops = 4 * B * H * N * N * D * 0.5                       # causal: half the work
    row = []
    for fn in (naive, flash_attn, lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            ms = do_bench(lambda: fn(q, k, v)); gb = torch.cuda.max_memory_allocated() / 1e9
        except torch.cuda.OutOfMemoryError:
            ms, gb = float("nan"), float("nan")
        row.append((ms, gb))
    (tn, gn), (tt, _), (ts, _) = row
    print(f"{N:6d} | {tn:9.2f} {gn:5.2f} | {tt:9.2f} {flops / (tt * 1e-3) / 1e12:6.1f} | {ts:8.2f}")
```

> [!TIP] Tuning knobs to try
> `BLOCK_M, BLOCK_N ∈ {32, 64, 128}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {1, 2, 3}` (software pipelining of the K/V loads). On a T4, shared memory (64 KB) limits how big you can go with $D = 128$. Wrap the kernel in `@triton.autotune(configs=[...], key=["N_CTX", "HEAD_DIM"])` once it’s correct.

> [!INTUITION] Where the official kernel goes further
> It splits the loop into two stages: fully-visible blocks **without** masking (cheaper) and the diagonal block **with** masking; it stores `LSE` for the backward pass; and on Hopper/Blackwell it uses tensor descriptors (TMA) and warp specialization. Read its source now — you’ll recognize every line of the core loop.

- [ ] All correctness cases pass for causal and non-causal
- [ ] Benchmark table from $N = 1024$ to $16384$: your kernel beats naive at every $N \ge 2048$ and runs where naive OOMs
- [ ] Reported TFLOPS vs the T4’s 65 TFLOPS FP16 tensor-core peak — what fraction do you reach, and what bounds it?
- [ ] Tried 3+ block/warp/stage configs and kept the best
      */}),
      resources: [
        { title: "Triton tutorial — Fused Attention", url: "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html", type: "docs", note: "the official (FA2-style, now Hopper/Blackwell-aware) Triton attention" },
        { title: "Triton tutorial — Matrix Multiplication", url: "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html", type: "docs", note: "tl.dot, block pointers and L2-friendly program ordering" },
      ],
    },

    {
      id: "decode-attention",
      title: "Decode attention: FlashDecoding (split-K), paged KV and FlashInfer",
      kind: "concept",
      minutes: 90,
      runsOn: ["colab"],
      md: MD(function () {/*
Everything so far was **prefill**: many queries at once, compute-bound. **Decode** is different: each sequence has **one** new query token that attends to its entire KV cache.

## Why the prefill kernel is bad at decode

- **Too little parallelism.** FA2 launches one thread block per (query block, batch, head). At decode with batch 1 and 32 heads that’s **32 blocks** — on a 132-SM H100, 75% of the GPU idles while each block scans a 32k-token KV cache sequentially.
- **It’s memory-bound.** One query row × $N$ keys: $\sim 4Nd$ FLOPs to read $\sim 4Nd$ bytes of K and V (FP16) → AI ≈ 1 (times the GQA group size, since query heads sharing a KV head reuse it). Decode attention speed = KV-cache bytes ÷ bandwidth, just like decode GEMVs are weight bytes ÷ bandwidth.

## FlashDecoding = split-K

Split the KV sequence into $S$ chunks, process chunks **in parallel** (more thread blocks → all SMs busy), each producing a partial $(m_s, \ell_s, o_s)$, then a tiny second kernel **merges** them with the formula from lesson 4:

$$m = \max_s m_s, \qquad \ell = \sum_s \ell_s\, e^{m_s - m}, \qquad O = \frac{\sum_s o_s\, e^{m_s - m}}{\ell}$$

Verify the math in PyTorch before writing any kernel:

```python
import torch, math
def attn_ref(q, K, V):                              # q: (d,), K, V: (N, d)
    return torch.softmax(K @ q / math.sqrt(q.numel()), dim=0) @ V

def attn_split_k(q, K, V, n_splits=8):
    parts = []
    for Kc, Vc in zip(K.chunk(n_splits), V.chunk(n_splits)):   # a kernel runs these in parallel
        s = Kc @ q / math.sqrt(q.numel())
        m = s.max(); p = torch.exp(s - m)
        parts.append((m, p.sum(), p @ Vc))           # (max, sum, unnormalized output)
    m_all = torch.stack([m for m, _, _ in parts]).max()
    l = sum(l_s * torch.exp(m_s - m_all) for m_s, l_s, _ in parts)
    o = sum(o_s * torch.exp(m_s - m_all) for m_s, _, o_s in parts)
    return o / l

q, K, V = torch.randn(128), torch.randn(32768, 128), torch.randn(32768, 128)
print((attn_ref(q, K, V) - attn_split_k(q, K, V)).abs().max())   # ~1e-7
```

The same merge is used for **cascade / shared-prefix attention** (attend to a shared system-prompt KV once, to each request’s suffix separately, then merge) and for **context parallelism** across GPUs (M16–M17).

## Paged KV cache kernels

In your mini engine (M08–M10) the KV cache lives in fixed-size **pages/blocks** scattered through a pool, with a per-sequence **block table**. A paged attention kernel therefore can’t assume K and V are contiguous: in its inner loop it looks up `block_table[seq, j // block_size]` to find the physical page, then loads the tile from there. That indirection is cheap if pages are large enough (16–64 tokens × head_dim) to keep loads coalesced.

> [!REAL] Who implements this in production
> - **vLLM** V1 defaults to FlashAttention’s varlen kernels with a `block_table` argument (paged KV in FA2/FA3), with FlashInfer, Triton and other backends selectable per GPU/model.
> - **FlashInfer** is a kernel library specifically for serving: paged and ragged KV layouts, decode/prefill/append kernels, split-K load balancing, cascade attention, fused sampling, and a *plan/run* API that precomputes the work split on the CPU so the GPU kernel (and CUDA graph) stays static. SGLang uses it heavily.
> - **TensorRT-LLM** has its own XQA/MMHA decode kernels (Baseten book ch. 6.2).

## Optional: call FlashInfer

```python
# pip install flashinfer-python   (pick the wheel matching your torch/CUDA per the docs)
import torch, flashinfer
kv_len, num_kv_heads, num_qo_heads, head_dim = 8192, 8, 32, 128     # GQA: 4 query heads per KV head
k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
o = flashinfer.single_decode_with_kv_cache(q, k, v)                 # (num_qo_heads, head_dim)
```

Measure it and compute achieved bandwidth: bytes = $2 \times \text{kv\_len} \times \text{num\_kv\_heads} \times \text{head\_dim} \times 2$ (K and V, FP16). A good decode kernel should approach your GPU’s peak bandwidth.

- [ ] Ran `attn_split_k` and confirmed the merge formula
- [ ] Computed the decode-attention time lower bound for Llama-3-8B (32 layers, 8 KV heads, $d=128$) at 32k context, batch 1, on an H100 (3.35 TB/s). Compare to the weight-read time (16 GB)
- [ ] Explained why split-K helps at batch 1 but matters less at batch 64
      */}),
      resources: [
        { title: "Flash-Decoding for long-context inference (Stanford CRFM)", url: "https://crfm.stanford.edu/2023/10/12/flashdecoding.html", type: "article", note: "the split-K decode idea, with animations and numbers" },
        { title: "FlashInfer (repo)", url: "https://github.com/flashinfer-ai/flashinfer", type: "repo", note: "serving-oriented attention/sampling kernels: paged, ragged, cascade" },
        { title: "FlashInfer paper (Ye et al., 2025)", url: "https://arxiv.org/abs/2501.01005", type: "paper", note: "design of a customizable attention engine for LLM serving" },
        { title: "FlashInfer docs", url: "https://docs.flashinfer.ai/", type: "docs", note: "API for decode/prefill wrappers and page tables" },
      ],
    },

    {
      id: "deep-kernels",
      title: "Deep dive: CUTLASS/CuTe, ThunderKittens, Hopper TMA/wgmma, Blackwell, FlexAttention",
      kind: "deep",
      optional: true,
      minutes: 240,
      runsOn: ["cloud"],
      md: MD(function () {/*
Triton gets you ~80–90% of the way on Ampere. The frontier kernels on Hopper and Blackwell use hardware features Triton only partially exposes. You don’t need to write these on day one — but you should be able to *read* them and discuss the trade-offs.

## What changed in Hopper (H100) and Blackwell (B200)

| Feature | Gen | What it does | Why kernels care |
|---|---|---|---|
| **TMA** (Tensor Memory Accelerator) | Hopper | hardware unit that copies whole multi-dim tiles global ↔ shared asynchronously | one thread issues a tile copy; others keep computing |
| **wgmma** | Hopper | *warpgroup* (4 warps) async matrix-multiply reading operands straight from SMEM | bigger, asynchronous MMAs; overlap with softmax |
| **Thread-block clusters** + distributed SMEM | Hopper | blocks on neighbouring SMs can read each other’s shared memory | multicast tiles to several SMs |
| **Warp specialization** | technique | producer warps only load (TMA), consumer warpgroups only compute | classic pipelining, the core of FA3 and modern GEMMs |
| **tcgen05 MMA + Tensor Memory (TMEM)** | Blackwell | new tensor-core instructions with a dedicated accumulator memory; 2-SM MMAs | accumulators leave the register file; FA4/Blackwell GEMMs are rewritten around it |
| **FP8 / FP4 (NVFP4, MXFP4)** | Hopper / Blackwell | 2× / 4× FLOPs vs FP16 | quantized attention/GEMM (M14) |

## The libraries and DSLs

- **CUTLASS** — NVIDIA’s C++ template library for GEMM/conv/attention; **CuTe** is its layout algebra for describing how tiles map onto threads and memory. FlashAttention-3 is built on CUTLASS/CuTe. The CuTe **Python DSL** now lets you write these kernels without C++ templates.
- **ThunderKittens** (Hazy Research) — a small C++ embedded DSL of 16×16 register/shared tiles with operations like `mma`, `exp`, `row_max`; FlashAttention-class kernels in a few hundred readable lines. Great for *reading*.
- **FlexAttention** (PyTorch) — write attention *variants* (causal, sliding window, ALiBi, document masking, soft-capping) as tiny Python functions; `torch.compile` generates a fused FlashAttention-style Triton kernel for them:

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def sliding_causal(b, h, q_idx, kv_idx):                 # a mask_mod: which (q, kv) pairs are allowed
    return (q_idx >= kv_idx) & (q_idx - kv_idx < 1024)

N = 8192
q, k, v = (torch.randn(1, 16, N, 64, device="cuda", dtype=torch.float16) for _ in range(3))
block_mask = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)   # skips fully-masked blocks
```

(Best on Ampere or newer; the `block_mask` lets the kernel skip entire tiles, which is how sliding-window attention becomes $O(Nw)$.)

## Suggested reading order

1. Tri Dao’s **FlashAttention-3 blog post** — the clearest explanation of warp specialization and ping-pong scheduling.
2. **ThunderKittens** blog post (“GPUs Go Brrr”) and its attention kernel.
3. **Colfax’s CUTLASS TMA tutorial**, then the **CuTe GEMM tutorial**.
4. **“Outperforming cuBLAS on H100: a worklog”** — Boehm’s journey, redone for Hopper.
5. The FA3/FA4 sources in `Dao-AILab/flash-attention`.

> [!REAL] Career note
> Few people write Hopper/Blackwell attention from scratch, and they’re among the best-paid engineers in the industry. Many more are valuable because they can *integrate* these kernels: pick the right backend, add a new mask or head dimension, fix a numerical issue, or port a kernel to a new layout (paged, ragged, FP8 KV). That’s the realistic target after this module.

- [ ] Read the FA3 blog post and explained warp specialization in your own words
- [ ] Implemented one FlexAttention variant (e.g. document masking) and checked it against a dense masked reference
- [ ] Skimmed a ThunderKittens attention kernel and mapped its loop to your Triton kernel
      */}),
      resources: [
        { title: "Tri Dao — FlashAttention-3 blog", url: "https://tridao.me/blog/2024/flash3/", type: "article", note: "warp specialization, ping-pong scheduling, FP8 on Hopper" },
        { title: "ThunderKittens", url: "https://github.com/HazyResearch/ThunderKittens", type: "repo", note: "tile DSL for fast, readable attention/GEMM kernels" },
        { title: "ThunderKittens — GPUs Go Brrr", url: "https://hazyresearch.stanford.edu/blog/2024-05-12-tk", type: "article", note: "what H100s actually need, explained with kernels" },
        { title: "CUTLASS", url: "https://github.com/NVIDIA/cutlass", type: "repo", note: "NVIDIA’s GEMM/attention template library and CuTe" },
        { title: "Colfax — CUTLASS Tutorial: Mastering TMA", url: "https://research.colfax-intl.com/tutorial-hopper-tma/", type: "article", note: "working Hopper kernels built around TMA" },
        { title: "Outperforming cuBLAS on H100: a worklog", url: "https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog", type: "article", note: "Hopper GEMM optimization diary (tensor cores, async copies)" },
        { title: "FlexAttention docs", url: "https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html", type: "docs", note: "attention variants compiled to fused kernels" },
        { title: "attention-gym", url: "https://github.com/pytorch-labs/attention-gym", type: "repo", note: "FlexAttention recipes for real masks and score mods" },
      ],
    },
  ],

  challenge: {
    title: "Your FlashAttention: causal Triton forward, verified and benchmarked",
    md: MD(function () {/*
In `course-work/m12/`, deliver a notebook + `REPORT.md`:

1. **Kernel.** A Triton FlashAttention-style forward kernel supporting causal and non-causal attention, FP16 in / FP32 accumulate, head dims 64 and 128, arbitrary $N$ (not just multiples of the block size). Autotuned over at least 4 configs.
2. **Correctness.** A test suite against `torch.nn.functional.scaled_dot_product_attention` over a grid of shapes: $B \in \{1, 4\}$, $H \in \{8, 16\}$, $N \in \{128, 1000, 4096\}$, $D \in \{64, 128\}$, causal on/off. Report max abs error per case.
3. **Benchmark.** Causal forward for $N$ = 512 → 16384 (B=1, H=16, D=64): time, TFLOPS (count $2BHN^2D$ for causal) and peak memory for **naive PyTorch**, **your kernel**, and **SDPA**. Plot time vs $N$ on log-log axes and peak memory vs $N$.
4. **Analysis.** Explain (a) why naive memory grows as $N^2$ and yours as $N$; (b) what fraction of the T4’s FP16 tensor-core peak you reach and what limits it; (c) what the official tutorial / FA2 do that you don’t.
    */}),
    checklist: [
      "All correctness cases pass (max abs error < 2e-2 in FP16), including non-multiple-of-block $N$",
      "Your kernel runs at $N = 16384$ where naive attention OOMs (or would exceed memory) on a T4",
      "Your kernel beats naive PyTorch at every $N \\ge 2048$, and you report the ratio to SDPA",
      "Log-log plots of time and memory vs $N$, with slopes explained",
      "Written derivation of online softmax (with the correction factor) in your own words",
      "At least 4 autotune configs benchmarked, best one justified",
    ],
    stretch: "Implement a split-K **decode** kernel in Triton (one query per sequence, KV split across programs + a merge kernel) and show it beats your prefill kernel at batch 1, $N = 32768$ — then add a `block_table` argument so it reads a **paged** KV cache from your M10 engine. Or: implement the backward pass (recompute $P$ from saved LSE) and check gradients with `torch.autograd.gradcheck` on small FP64 inputs.",
  },

  connects: MD(function () {/*
You’ve now built the two pillars of LLM kernels: **fusion** (softmax) and **IO-aware tiling with online softmax** (FlashAttention). **M13** steps back to the whole program: even with perfect kernels, a decode step made of hundreds of tiny launches can be **CPU-bound** — you’ll use profilers, `torch.compile` (which writes Triton kernels like yours) and CUDA graphs to fix it. **M14** (quantization) brings FP8/INT4 attention and GEMMs, and **M17** uses the split-K merge you derived for context parallelism and disaggregated prefill. Your paged split-K decode kernel (stretch) is the exact kernel your mini-engine from M10 has been missing.
  */}),

  interview: [
    "Derive online softmax. Why is the correction factor $e^{m_{old} - m_{new}}$, and how does it extend to the attention output accumulator?",
    "FlashAttention does the same (or more) FLOPs as standard attention. Why is it faster? Give the memory complexity of both.",
    "Why does FlashAttention recompute attention in the backward pass instead of storing $P$? What does it store instead?",
    "What changed from FlashAttention-2 to FlashAttention-3, and why were those changes Hopper-specific?",
    "Why is a prefill attention kernel a poor fit for decode at batch size 1, and how does FlashDecoding fix it?",
    "How does a paged attention kernel find the keys for token position $t$? What’s the cost of that indirection?",
    "Compare writing a kernel in Triton vs CUDA. What does the Triton compiler handle for you, and when would you drop down to CUDA/CUTLASS?",
    "Softmax is memory-bound. Estimate the time for softmax over a $4096 \\times 32000$ FP32 logits tensor on an H100 (3.35 TB/s) with a fused vs a naive 5-kernel implementation.",
  ],

  resources: [
    { title: "FlashAttention (Dao et al., 2022)", url: "https://arxiv.org/abs/2205.14135", type: "paper", note: "the foundational IO-aware attention paper" },
    { title: "FlashAttention-2", url: "https://arxiv.org/abs/2307.08691", type: "paper", note: "work partitioning that production kernels follow" },
    { title: "FlashAttention-3", url: "https://arxiv.org/abs/2407.08608", type: "paper", note: "Hopper-specific async + FP8 attention" },
    { title: "Triton tutorials", url: "https://triton-lang.org/main/getting-started/tutorials/index.html", type: "docs", note: "vector add → softmax → matmul → fused attention, officially maintained" },
    { title: "Triton-Puzzles", url: "https://github.com/srush/Triton-Puzzles", type: "practice", note: "hands-on drills for the Triton programming model" },
    { title: "Online normalizer calculation for softmax", url: "https://arxiv.org/abs/1805.02867", type: "paper", note: "the math behind FlashAttention’s one-pass softmax" },
    { title: "Flash-Decoding (Stanford CRFM)", url: "https://crfm.stanford.edu/2023/10/12/flashdecoding.html", type: "article", note: "split-K attention for long-context decode" },
    { title: "FlashInfer", url: "https://github.com/flashinfer-ai/flashinfer", type: "repo", note: "the serving-kernel library behind SGLang and a vLLM backend" },
    { title: "GPU MODE lectures", url: "https://github.com/gpu-mode/lectures", type: "course", note: "lectures on Triton, FlashAttention and CUTLASS" },
    { title: "Inference Engineering (Baseten) — Ch. 2.4–2.5, 4.1", url: "Inference%20Engineering.pdf", type: "book", note: "attention arithmetic intensity, FlashAttention and kernel selection" },
    { title: "Muser book — Ch. 16 Attention decode kernels", url: "https://highperformanceailab.com/muser-book/chapters/16-attention-decode-kernels.html", type: "book", note: "optional Mac/Metal track: a ladder of decode attention kernels" },
  ],
});
