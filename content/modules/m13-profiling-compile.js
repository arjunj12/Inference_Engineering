Course.module({
  id: "m13-profiling-compile",
  title: "Profiling & compilers: torch.compile, CUDA graphs, Nsight",
  short: "Profiling & torch.compile",
  tagline: "Take a decode loop from slow Python to near bandwidth-bound — one measured, profiled, explained change at a time.",
  hours: 18,
  level: "core",
  runsOn: ["colab", "cloud"],
  tags: ["profiling", "torch.compile", "cuda-graphs", "nsight", "perfetto", "launch-overhead"],

  goal: MD(function () {/*
You finish with a **profiling report** for a Llama-shaped decoder (your M10 engine’s model, or the self-contained `tinyllama.py` from lesson 1, or gpt-fast) that looks like this — every row backed by a trace:

```text
decode, batch 1, 220M-param Llama-shaped model, fp16, Tesla T4 (320 GB/s)
bytes read per token ≈ 0.42 GB  ->  bandwidth ceiling ≈ 762 tok/s (1.31 ms/token)

 variant                                   ms/token   tok/s   kernels/token   MBU    what the trace showed
 v0 eager, torch.cat cache, .item()/token   11.9        84        ~540        11%    GPU idle 85%: CPU launch-bound + a sync per token
 v1 static KV cache, no per-token sync       9.1       110        ~530        14%    syncs gone, still CPU-bound (gaps between tiny kernels)
 v2 torch.compile (default mode)             3.2       312        ~110        41%    norms/RoPE fused into Triton kernels; smaller gaps
 v3 + mode="reduce-overhead" (CUDA graphs)   1.85      540       1 graph      71%    one cudaGraphLaunch; GPU kernels back-to-back
 v4 + max-autotune                           1.75      571       1 graph      75%    GEMV-heavy step now bandwidth-bound; ncu confirms
```

(Illustrative — your numbers will differ.) The speedup is **~7×** with **zero changes to the math** — the same weights, the same kernels in spirit. The point isn’t the number; it’s that you can open a trace, say *why* the GPU is idle, fix it, and prove the fix with the next trace. MBU (model bandwidth utilization) = achieved bytes/s ÷ peak bytes/s.

::viz launch-overhead
  */}),
  demo: { viz: "launch-overhead", params: {} },

  why: MD(function () {/*
**Most “slow model” bugs aren’t slow kernels — they’re an idle GPU.** At batch size 1–8, a decode step reads a few GB of weights in a millisecond or two, but eager PyTorch spends several milliseconds of *CPU* time dispatching hundreds of small kernels. gpt-fast showed a Llama-7B going from **25 tok/s to 107 tok/s** on an A100 purely from `torch.compile` + a static KV cache + CUDA graphs — before any quantization.

Every production engine is built around this: **vLLM** compiles the model with `torch.compile` and captures **CUDA graphs** for a set of batch sizes; **SGLang** and **TensorRT-LLM** do the same; Inference Engineering (Baseten) ch. 4.2 and 4.5 treat compilation and profiling as core skills (and warn about compile-time cold starts). In interviews and on the job, “here is the trace, here is the bottleneck, here is the fix and its measured impact” is *the* inference-engineering deliverable. This module teaches the method and the tools: `torch.profiler` + Perfetto, Nsight Systems / Nsight Compute, `torch.compile` and CUDA graphs.
  */}),

  prereqs: [
    {
      title: "Roofline and memory-bound decode (M07, M11)",
      skipIf: "you can compute a decode step’s bandwidth-bound time from parameter count and GPU bandwidth",
      math: true,
      md: MD(function () {/*
At batch 1, each decode step must read **every weight once** (plus the KV cache). Matrix-vector products do ~2 FLOPs per 2-byte FP16 weight → arithmetic intensity ≈ 1 FLOP/byte, far below any GPU’s ridge point (T4 ≈ 200, H100 ≈ 295 FLOP/byte). So the **floor** on step time is

$$t_{\text{step}} \ge \frac{\text{bytes read per token}}{\text{memory bandwidth}}$$

Example: 7B params × 2 bytes = 14 GB on an H100 (3.35 TB/s) → ≥ 4.2 ms → ≤ ~240 tok/s. Anything much slower than this floor means time is being lost *somewhere else* — that “somewhere else” is what this module hunts.
      */}),
    },
    {
      title: "GPU execution is asynchronous",
      skipIf: "you know why `time.time()` around a CUDA op without `torch.cuda.synchronize()` is wrong",
      md: MD(function () {/*
When Python calls `y = x @ w` on a CUDA tensor, PyTorch **enqueues** a kernel on a CUDA stream and returns immediately; the GPU runs it later. The CPU can run ahead, queueing more work. The CPU only waits when you ask for a result on the host (`.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, `if tensor:`) or call `torch.cuda.synchronize()`. Two consequences drive this whole module: (1) timing needs explicit synchronization, and (2) if the CPU takes longer to *enqueue* a kernel than the GPU takes to *run* it, the GPU sits idle — you’re **CPU-bound**.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: the same model, 5× faster, with one line of torch.compile",
      kind: "demo",
      minutes: 75,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
## Part 1 — a Llama-shaped model you can abuse (Colab T4)

We need a model that is *real-shaped* (RMSNorm, RoPE, SwiGLU, KV cache, 32k vocab) but loads instantly and fits anywhere. Random weights are fine: performance doesn’t care what the weights *mean*. Create it once — later lessons import it:

```python
%%writefile tinyllama.py
# tinyllama.py - a Llama-shaped decoder with random weights, for performance work only
import time, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Cfg:
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    hidden: int = 2816
    vocab: int = 32000
    max_seq: int = 1024

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w

def rope(x, cos, sin):                          # x: (B, H, T, D); cos, sin: (T, D/2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class DynamicCache(nn.Module):                  # grows with torch.cat: a new shape every step
    def __init__(self):
        super().__init__(); self.k = self.v = None
    def update(self, pos, k, v):
        if self.k is None: self.k, self.v = k, v
        else: self.k, self.v = torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)
        return self.k, self.v

class StaticCache(nn.Module):                   # preallocated: the same shapes every step
    def __init__(self, c, B, dtype, device):
        super().__init__()
        shape = (B, c.n_heads, c.max_seq, c.dim // c.n_heads)
        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device))
    def update(self, pos, k, v):                # pos: (T,) int64 positions on the GPU
        self.k.index_copy_(2, pos, k); self.v.index_copy_(2, pos, v)
        return self.k, self.v

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.hd = c.n_heads, c.dim // c.n_heads
        self.wqkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)
        self.cache = None
    def forward(self, x, cos, sin, pos, mask):
        B, T, _ = x.shape
        q, k, v = self.wqkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        k, v = self.cache.update(pos, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and T > 1))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2, self.attn = RMSNorm(c.dim), RMSNorm(c.dim), Attention(c)
        self.w13 = nn.Linear(c.dim, 2 * c.hidden, bias=False)
        self.w2 = nn.Linear(c.hidden, c.dim, bias=False)
    def forward(self, x, cos, sin, pos, mask):
        x = x + self.attn(self.n1(x), cos, sin, pos, mask)
        a, b = self.w13(self.n2(x)).chunk(2, dim=-1)
        return x + self.w2(F.silu(a) * b)

class Model(nn.Module):
    def __init__(self, c=Cfg()):
        super().__init__()
        self.c, self.static = c, False
        self.emb = nn.Embedding(c.vocab, c.dim)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm, self.out = RMSNorm(c.dim), nn.Linear(c.dim, c.vocab, bias=False)
        hd = c.dim // c.n_heads
        ang = torch.outer(torch.arange(c.max_seq).float(), 1.0 / 10000 ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.register_buffer("kv_idx", torch.arange(c.max_seq), persistent=False)

    def setup_cache(self, B=1, static=True):
        if static and self.static:              # reuse buffers -> same tensors -> no recompiles
            for l in self.layers: l.attn.cache.k.zero_(); l.attn.cache.v.zero_()
            return
        w = self.out.weight
        self.static = static
        for l in self.layers:
            l.attn.cache = StaticCache(self.c, B, w.dtype, w.device) if static else DynamicCache()

    def forward(self, idx, pos):                # idx: (B, T) token ids; pos: (T,) positions
        x = self.emb(idx)
        cos, sin = self.cos[pos], self.sin[pos]
        mask = (self.kv_idx[None, :] <= pos[:, None]) if self.static else None
        for l in self.layers:
            x = l(x, cos, sin, pos, mask)
        return self.out(self.norm(x[:, -1:]))   # logits for the last position only

@torch.no_grad()
def generate_eager(model, prompt, n_new, step=None):      # v0: the "obvious" loop
    model.setup_cache(static=False)
    T = prompt.shape[1]
    tok = model(prompt, torch.arange(T, device=prompt.device)).argmax(-1)
    out = []
    for i in range(n_new):
        out.append(tok.item())                             # GPU -> CPU sync every token
        tok = model(tok, torch.tensor([T + i], device=prompt.device)).argmax(-1)
    return out

def decode_one(model, tok, pos):
    return model(tok, pos).argmax(-1)

@torch.no_grad()
def generate_static(model, prompt, n_new, step=decode_one):   # v1+: static cache, no per-token sync
    model.setup_cache(static=True)
    T = prompt.shape[1]
    tok = model(prompt, torch.arange(T, device=prompt.device)).argmax(-1)
    pos = torch.tensor([T], device=prompt.device)
    out = torch.empty(n_new, dtype=torch.long, device=prompt.device)
    for i in range(n_new):
        out[i] = tok[0, 0]
        tok = step(model, tok, pos).clone()   # clone: CUDA-graph outputs are overwritten by the next replay
        pos += 1
    return out.tolist()                       # one sync, at the end

def bench(model, prompt, gen, n_new=256, **kw):
    gen(model, prompt, 32, **kw)              # warmup: compilation, cuBLAS init, CUDA-graph capture
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen(model, prompt, n_new, **kw)
    torch.cuda.synchronize()
    return n_new / (time.perf_counter() - t0)

def bytes_per_token(model):
    w = sum(p.numel() * p.element_size() for n, p in model.named_parameters() if not n.startswith("emb"))
    kv = sum(b.numel() * b.element_size() for n, b in model.named_buffers() if n.endswith((".k", ".v")))
    return w + kv                             # embedding lookup reads 1 row, so it's excluded
```

Now the demo (Runtime → Change runtime type → T4 GPU):

```python
import torch
from tinyllama import Model, Cfg, generate_eager, generate_static, decode_one, bench, bytes_per_token
torch.manual_seed(0)
with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")

print(f"eager   : {bench(model, prompt, generate_eager):6.0f} tok/s")
fast = torch.compile(decode_one, mode="reduce-overhead", fullgraph=True)
print(f"compiled: {bench(model, prompt, generate_static, step=fast):6.0f} tok/s   (first call compiles: ~1 min)")
bpt = bytes_per_token(model)
print(f"bytes/token {bpt/1e9:.2f} GB -> T4 ceiling {320e9/bpt:.0f} tok/s")
```

```text
eager   :     84 tok/s
compiled:    540 tok/s   (first call compiles: ~1 min)
bytes/token 0.42 GB -> T4 ceiling 762 tok/s
```

(Illustrative.) Same weights, same math, **~6× faster**. Eager decode ran at ~11% of what the memory system allows. Where did the other 89% go? That question *is* this module.

## Part 2 — gpt-fast on a real 7B (optional, L4/A100/H100)

gpt-fast is PyTorch’s ~1000-line reference for fast LLM decode: a plain model file, a static KV cache and `torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)`. On a rented GPU with ≥ 24 GB:

```bash
git clone https://github.com/meta-pytorch/gpt-fast && cd gpt-fast
pip install -r requirements.txt
export MODEL_REPO=openlm-research/open_llama_7b      # ungated; Llama-2/3 need an approved HF token
./scripts/prepare.sh $MODEL_REPO
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"            # eager
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"  # compiled
```

It prints tokens/sec and **bandwidth achieved (GB/s)** — compare the latter to your GPU’s spec sheet. The PyTorch team’s published A100 progression for Llama-7B:

| Step | tok/s |
|---|---|
| eager baseline | 25.5 |
| + `torch.compile` + static KV cache (+ CUDA graphs) | 107.0 |
| + int8 weight-only quantization | 157.4 |
| + int4 weight-only quantization | 202.1 |
| + int4 + speculative decoding | 244.7 |

> [!NOTE] gpt-fast on a T4
> gpt-fast loads weights in BF16 and uses FlexAttention; the T4 has no BF16 tensor cores and is below Triton’s official target, so on a T4 stick with `tinyllama.py` (FP16). It also has a `--profile` flag that writes a Chrome trace — you’ll use traces like that from lesson 4 on.

- [ ] Ran Part 1 and wrote down eager tok/s, compiled tok/s and the bandwidth ceiling for your GPU
- [ ] Computed MBU for both runs: `tok_per_s * bytes_per_token / peak_bandwidth`
- [ ] (Optional) Ran gpt-fast eager vs `--compile` on a cloud GPU and compared its reported GB/s to the spec sheet
      */}),
      resources: [
        { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "the minimal reference for compiled, CUDA-graphed LLM decode" },
        { title: "Accelerating Generative AI with PyTorch II: GPT, Fast", url: "https://pytorch.org/blog/accelerating-generative-ai-2/", type: "article", note: "the step-by-step story (with traces) behind the 25 → 244 tok/s table" },
      ],
    },

    {
      id: "methodology",
      title: "Methodology: measure → hypothesis → change → measure",
      kind: "concept",
      minutes: 60,
      runsOn: ["colab"],
      md: MD(function () {/*
Performance work is the scientific method with a stopwatch. The loop:

1. **Measure** end-to-end with a trustworthy benchmark (the number your users feel: tok/s, TTFT, ms/step).
2. **Profile** to see *where* the time goes (a trace, not a guess).
3. **Hypothesize** one cause, and predict the gain *before* you change anything (“removing the per-token sync should save ≈ the sync gap, ~0.3 ms”).
4. **Change one thing.** Re-check correctness (same tokens/logits within tolerance).
5. **Measure again** with the same harness; record it in a results table, with the trace.

If the gain differs from your prediction, your mental model was wrong — that’s the most valuable moment of the loop.

## Timing that doesn’t lie

```python
import torch, statistics

def time_cuda(fn, iters=50, warmup=10):
    for _ in range(warmup):                     # compile, autotune, cuBLAS handles, allocator warm-up
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()                    # wait for the GPU before reading events
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))   # milliseconds
    return {"median": ts[len(ts) // 2], "p10": ts[len(ts) // 10], "p90": ts[int(len(ts) * 0.9)],
            "cv": statistics.stdev(ts) / statistics.mean(ts)}
```

CUDA events are timestamps written *by the GPU* into the stream. The interval between two events includes any idle gaps while the GPU waited for the CPU — which is exactly what you want when measuring a CPU-bound step. For single kernels, `triton.testing.do_bench` is a good default (it also flushes L2 between runs).

## The classic mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No `torch.cuda.synchronize()` | impossibly fast numbers (you timed the *enqueue*) | sync before reading the clock, or use CUDA events |
| Timing the first call | compile/autotune/`cudnn`/`cuBLAS` init dominates | warm up; report compile time separately |
| One sample | noise looks like a speedup | median of ≥ 20 runs; report p10/p90; check CV < ~3% |
| Timing under the profiler | profiler overhead inflates CPU time | profile to *understand*, benchmark to *quote* |
| Different work in A vs B | “faster” because it generated fewer tokens or used a shorter cache | fix inputs, lengths, seeds; assert outputs match |
| Ignoring clocks/thermals | numbers drift during a run (the 70 W T4 throttles) | check `nvidia-smi -q -d CLOCK,PERFORMANCE`; interleave A/B runs |
| Hidden syncs in the timed region | `.item()`, `print(t)`, `if t.any():` | move them out; the profiler shows them (lesson 4) |

## Latency is a distribution

Decode steps have a *distribution* of latencies — occasional allocator calls, Python GC pauses, or a recompile show up as a long tail. Report percentiles, not just the mean (you’ll do this for real in M19).

::viz percentiles

## The results table

Keep this in your notebook from now on and fill a row per experiment:

```text
| # | change (one!)              | hypothesis / predicted gain          | ms/token (median, p90) | tok/s | MBU | trace file   | correct? |
|---|----------------------------|--------------------------------------|------------------------|-------|-----|--------------|----------|
| 0 | baseline                   | -                                    |                        |       |     | v0.json      | ✓        |
| 1 | static cache, no .item()   | removes sync + cat; ~15% faster      |                        |       |     | v1.json      |          |
```

> [!REAL] How this looks at work
> A perf PR at an inference company is exactly this table plus two traces (before/after) and a correctness check. Reviewers ask “what did you predict, and why did it differ?” Get used to writing the prediction *first*.

- [ ] Timed `generate_eager` three ways: wall clock without sync, wall clock with sync, CUDA events — and explained the differences
- [ ] Ran the same benchmark 10 times and computed its CV; decided what size of improvement you can trust
- [ ] Created your results table with the v0 row filled in
      */}),
      resources: [
        { title: "PyTorch — CUDA semantics", url: "https://docs.pytorch.org/docs/stable/notes/cuda.html", type: "docs", note: "asynchronous execution, streams, events and the caching allocator" },
        { title: "Horace He — Making Deep Learning Go Brrrr From First Principles", url: "https://horace.io/brrr_intro.html", type: "article", note: "compute vs memory vs overhead: the three regimes to diagnose" },
      ],
    },

    {
      id: "launch-overhead",
      title: "Math: CPU-bound vs GPU-bound — why eager decode leaves the GPU idle",
      kind: "math",
      minutes: 75,
      runsOn: ["colab"],
      md: MD(function () {/*
> [!PREREQ] A producer and a consumer
> The CPU (producer) pushes kernels into a queue (the CUDA stream); the GPU (consumer) pops and runs them. If the producer is slower than the consumer, the queue is usually empty and the consumer waits. Throughput is set by the **slower** of the two.

## The model

Let a decode step issue $n$ kernels. Each costs the CPU $c$ µs to issue (Python + PyTorch dispatcher + `cudaLaunchKernel`) and the GPU $g_i$ µs to run. If the CPU never waits for the GPU (no syncs), the two overlap and

$$t_{\text{step}} \approx \max\left(n \cdot c,\; \sum_i g_i\right)$$

With a sync at the end of the step (e.g. `.item()`), the pipeline drains each step, and you pay closer to the *sum* of the CPU-only and GPU-only parts.

> [!INTUITION] Tiny kernels are the enemy
> A GEMV of a 1024 × 1024 FP16 weight reads 2 MB — at 320 GB/s that’s ~6.5 µs of GPU time. The eager PyTorch CPU cost of launching it is ~10–20 µs on a Colab CPU. Every such op is *CPU-bound on its own*, and a decode step is hundreds of them.

## Worked example: `tinyllama.py` on a T4

Count kernels per layer in eager mode (roughly): RMSNorm ≈ 8 elementwise/reduction kernels × 2 norms, RoPE ≈ 7 kernels × 2 (q and k), 4 GEMVs (`wqkv`, `wo`, `w13`, `w2`), 2 cache writes, attention (1–3), SiLU, multiply, 2 residual adds, a transpose copy → **~45 kernels/layer**, × 12 layers + embedding, final norm, LM head, argmax ≈ **540 kernels per token**.

- CPU: $540 \times 15\ \mu s \approx 8.1$ ms
- GPU: 0.42 GB ÷ 320 GB/s ≈ 1.3 ms of useful traffic, plus ~1 µs+ of launch latency and tail per tiny kernel ≈ **2 ms**

$$t_{\text{step}} \approx \max(8.1, 2.0) = 8.1 \text{ ms} \;\Rightarrow\; \approx 120 \text{ tok/s}, \quad \text{GPU busy} \approx \frac{2.0}{8.1} \approx 25\%$$

Now the three levers, in the order `torch.compile` pulls them:

| Lever | Effect on the model | Example |
|---|---|---|
| **Fusion** (Inductor) | shrinks $n$ *and* GPU bytes (fewer intermediates) | 540 → ~110 kernels; CPU ≈ 110 × ~8 µs ≈ 0.9 ms (plus guards/wrapper) |
| **CUDA graphs** | shrinks $c$ to ~0 per kernel: one `cudaGraphLaunch` per step | CPU ≈ 20–50 µs per step |
| **Better kernels** (autotune) | shrinks $\sum g_i$ toward bytes ÷ bandwidth | 2.0 → ~1.5 ms |

Once CPU time is below GPU time, *only* the GPU side matters and you’re on the roofline problem from M11 again — now the right next steps are fewer bytes (quantization, M14) or more tokens per byte (batching, M09; speculative decoding, M15).

::viz launch-overhead

Drag the controls: with small kernels, the CPU launch gaps dominate the timeline; with graph replay, kernels pack back-to-back. Note that **batching** is the other cure: at batch 64 each kernel does 64× more useful work for the same launch cost — one reason CPU overhead is mostly a *low-batch* (latency) problem.

## Measure your own $n$ and $c$

```python
import torch
from torch.profiler import profile, ProfilerActivity
from tinyllama import Model, Cfg, generate_eager
with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")
generate_eager(model, prompt, 8)                             # warm up
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    generate_eager(model, prompt, 20)
ev = {e.key: e for e in prof.key_averages()}
launches = sum(ev[k].count for k in ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch") if k in ev)
gpu_us = sum(e.self_device_time_total for e in prof.key_averages()) / 20
print(f"kernel launches/token ≈ {launches / 20:.0f}, GPU busy time/token ≈ {gpu_us / 1e3:.2f} ms")
```

(On older PyTorch versions the attribute is `self_cuda_time_total`.) Divide measured ms/token by launches/token to estimate $c$.

- [ ] Measured launches/token and GPU-busy time/token for v0; computed GPU utilization = busy ÷ step time
- [ ] Predicted the tok/s if CPU cost dropped to zero (GPU-busy time only) — this is your v3 target
- [ ] Re-ran at batch size 16 (change `prompt` to shape `(16, 16)`): did tok/s per sequence drop? Did total tok/s rise? Explain with the formula
      */}),
      resources: [
        { title: "NVIDIA — Getting started with CUDA Graphs", url: "https://developer.nvidia.com/blog/cuda-graphs/", type: "article", note: "measures per-launch overhead and how graphs remove it" },
        { title: "Inference Engineering (Baseten) — Ch. 4.2, 4.5", url: "Inference%20Engineering.pdf", type: "book", note: "compilation, CUDA graphs and profiling in production engines" },
      ],
    },

    {
      id: "torch-profiler",
      title: "torch.profiler + Perfetto: reading a trace like a pro",
      kind: "build",
      minutes: 120,
      runsOn: ["colab"],
      md: MD(function () {/*
## Capture a trace

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from tinyllama import Model, Cfg, generate_eager, generate_static

with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")

def trace(gen, name, n_new=16, **kw):
    gen(model, prompt, 8, **kw)                                # warm up outside the profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=False) as prof:
        with record_function(f"decode_{name}"):                # a named span in the trace
            gen(model, prompt, n_new, **kw)
    prof.export_chrome_trace(f"{name}.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))   # newer: "device_time_total"

trace(generate_eager, "v0")
trace(generate_static, "v1")
```

Download `v0.json` (Colab file browser) and drag it into **https://ui.perfetto.dev** (or `chrome://tracing`). Keyboard: **W/S** zoom, **A/D** pan, click a slice for its details, and select an area to get aggregate stats. Use `record_function` (or `torch.cuda.nvtx.range` for Nsight) to label *your* phases — prefill, decode step, sampling — so the trace reads like your code.

For long runs, use a schedule so you only record a few steps: `profile(..., schedule=torch.profiler.schedule(wait=1, warmup=1, active=3), on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"))` and call `prof.step()` each iteration.

## Reading the table

Sort by device time to see which kernels dominate the GPU; sort by `cpu_time_total` to see which ops dominate the CPU. For decode on v0 you should see GEMV kernels (cuBLAS `gemv`/`gemm` names) at the top of GPU time, but `aten::` ops with CPU times far larger than their CUDA times — the overhead signature.

## Reading a trace like a pro — a pattern gallery

A trace has (at least) a **CPU thread row** (Python → `aten::` ops → `cudaLaunchKernel`) and a **GPU stream row** (the kernels). Flow arrows connect each launch to its kernel. Learn to recognize these shapes:

| What you see | Diagnosis | Typical fix |
|---|---|---|
| GPU row: thin kernels with **gaps** between them; CPU row: dense `aten::` ops | CPU/launch-bound | fusion (`torch.compile`), CUDA graphs, larger batch |
| Many kernels **< 5 µs** (elementwise, `copy_`, `cat`) | unfused small ops; each pays launch + a round trip to HBM | fuse; avoid `.contiguous()`/`cat` in the hot loop |
| `cudaMemcpyAsync` **DtoH** + `cudaStreamSynchronize` on the CPU row, then a GPU gap | host sync: `.item()`, `.tolist()`, `print(t)`, `if t:`, `torch.nonzero`, boolean-mask indexing | keep values on the GPU; sync once at the end |
| `cudaMemcpyAsync` **HtoD** each step | `torch.tensor([...], device="cuda")` from Python data (pageable memory) | keep state (positions, tokens) on the GPU; pinned memory + `non_blocking=True` |
| `cudaMalloc`/`cudaFree` in steady state | shapes change every step (growing cache) → allocator churn | static shapes, preallocated buffers |
| “Torch-Compiled Region” with Dynamo frames or long CPU gaps every N steps | recompilation (guard failures) | find it with `TORCH_LOGS=recompiles` (lesson 7) |
| One kernel dominates the GPU row | it’s a kernel problem, not an overhead problem | Nsight Compute on that kernel; compare to the roofline |
| A single `cudaGraphLaunch`, then kernels back-to-back | CUDA graphs working | now optimize kernels/bytes |

> [!TIP] Always check the flow arrows
> Click a kernel and follow its arrow to the CPU launch. If the launch happened *long before* the kernel started, the GPU was busy (good: GPU-bound). If the kernel started *immediately* after its launch and the previous kernel finished long ago, the GPU was waiting for the CPU (bad: CPU-bound).

## Exercise: annotate v0 vs v1

In `v0.json` find (1) the `aten::item` → `cudaStreamSynchronize` per token, (2) the HtoD copy for `torch.tensor([T + i])`, (3) growing `aten::cat` kernels from the dynamic cache. In `v1.json` confirm they’re gone — and notice that v1 is **still** mostly gaps. Removing syncs let the CPU run ahead, but it can’t run ahead faster than it can launch.

- [ ] Exported `v0.json` and `v1.json`, opened both in Perfetto, and took one screenshot of a single decode step in each
- [ ] Measured (with Perfetto’s area select) the GPU-busy fraction of one decode step in each trace
- [ ] Found and labeled every host sync in v0; wrote each one’s cost in µs into your results table
- [ ] Added `record_function` labels for prefill vs decode and confirmed they appear in the trace
      */}),
      resources: [
        { title: "PyTorch Profiler recipe", url: "https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", type: "docs", note: "profile(), key_averages, record_function, schedules, trace export" },
        { title: "torch.profiler API", url: "https://docs.pytorch.org/docs/stable/profiler.html", type: "docs", note: "every option: activities, record_shapes, with_stack, schedule" },
        { title: "Perfetto UI", url: "https://ui.perfetto.dev/", type: "tool", note: "open Chrome/Kineto JSON traces in the browser, locally" },
        { title: "Kineto", url: "https://github.com/pytorch/kineto", type: "repo", note: "the library under torch.profiler that collects GPU activity via CUPTI" },
      ],
    },

    {
      id: "nsight-instruments",
      title: "Nsight Systems, Nsight Compute — and Metal tools on the Mac",
      kind: "lab",
      minutes: 120,
      runsOn: ["colab", "cloud", "mac"],
      md: MD(function () {/*
`torch.profiler` sees PyTorch ops. **Nsight Systems** (`nsys`) sees the *whole process* timeline (CPU threads, CUDA API, kernels, memcpys, NVTX ranges, OS runtime, NCCL) with lower overhead. **Nsight Compute** (`ncu`) zooms into *one kernel*: achieved bandwidth, occupancy, stalls — the kernel-level roofline.

## A benchmark script to profile

```python
%%writefile decode_bench.py
import sys, torch
from tinyllama import Model, Cfg, generate_eager, generate_static, decode_one
variant = sys.argv[1]
with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")
gen, kw = (generate_eager, {}) if variant == "v0" else (generate_static, {})
if variant == "v3":
    kw["step"] = torch.compile(decode_one, mode="reduce-overhead", fullgraph=True)
gen(model, prompt, 32, **kw)                                   # warmup
torch.cuda.synchronize()
with torch.cuda.nvtx.range(f"decode_{variant}"):
    gen(model, prompt, 64, **kw)
torch.cuda.synchronize()
```

## Nsight Systems

```bash
!which nsys || ls /usr/local/cuda/bin | grep -i nsys
!nsys profile -o v0 --trace=cuda,nvtx,osrt --force-overwrite true python decode_bench.py v0
!nsys profile -o v3 --trace=cuda,nvtx,osrt --cuda-graph-trace=node --force-overwrite true python decode_bench.py v3
!nsys stats --report cuda_api_sum,cuda_gpu_kern_sum v0.nsys-rep
```

`nsys stats` gives text summaries (time in `cudaLaunchKernel` vs `cudaStreamSynchronize`, top kernels) that work even without a GUI; download the `.nsys-rep` and open it in the free Nsight Systems desktop app (Windows/Linux/macOS host) for the timeline. `--cuda-graph-trace=node` shows the individual kernels inside a CUDA graph instead of one opaque block.

> [!WARNING] Tool availability on hosted GPUs
> `nsys` is not always installed on Colab images; if `which nsys` finds nothing, install the CLI following the Nsight Systems installation guide, or use a cloud GPU image that ships the full CUDA toolkit. `ncu` needs access to GPU performance counters; on many hosted/containerized runtimes it fails with `ERR_NVGPUCTRPERM`. If so, do the `ncu` part on a rented GPU where you control the container, and rely on `nsys` + `torch.profiler` on Colab.

## Nsight Compute on one kernel

Profile only a few launches of the kernels you care about, after warmup:

```bash
!ncu --set full -k regex:"gemv|gemm" --launch-skip 200 --launch-count 3 -o gemv python decode_bench.py v1
!ncu --import gemv.ncu-rep --page details --section SpeedOfLight --section MemoryWorkloadAnalysis
```

For a batch-1 GEMV, **Speed Of Light** should show DRAM throughput as the high bar (memory-bound); a good GEMV reaches ~80–90% of peak DRAM bandwidth. If DRAM % is low *and* compute % is low, the kernel is latency-bound (too few blocks/too little work) — common for tiny decode kernels, and a reason fused kernels help. The roofline chart in the `full` set places the kernel on the same plot you built in M11.

## On the Mac (optional)

CUDA tools don’t exist on Apple Silicon, but the method is identical:

- **Xcode Instruments → Metal System Trace**: CPU encoding vs GPU execution timelines — the Metal equivalent of an nsys trace. Look for the same gaps.
- **MLX GPU capture** for a kernel-level view in Xcode’s Metal debugger:

```python
# run with: MTL_CAPTURE_ENABLED=1 python capture.py
import mlx.core as mx
a, b = mx.random.normal((4096, 4096)), mx.random.normal((4096,))
mx.eval(a, b)
mx.metal.start_capture("gemv.gputrace")
for _ in range(10):
    mx.eval(a @ b)
mx.metal.stop_capture()                 # open gemv.gputrace in Xcode
```

- **PyTorch MPS**: wrap code in `with torch.mps.profiler.profile():` and record with Instruments to see MPS op signposts.
- `mx.compile` is MLX’s analog of `torch.compile` (fuses elementwise graphs); the Muser book’s chapter on the **dispatch gap** walks through the same CPU-encoding-vs-GPU-execution problem on Metal.

- [ ] Captured `nsys` traces for v0 and v3 and pasted the `cuda_api_sum` top rows into your report: how much CPU time is `cudaLaunchKernel` vs `cudaGraphLaunch`?
- [ ] Profiled one decode GEMV with `ncu` (or recorded why you couldn’t, and did it on a cloud GPU) and wrote down its % of peak DRAM bandwidth
- [ ] (Mac, optional) Captured a Metal System Trace of an MLX or MPS decode loop and identified CPU encode gaps
      */}),
      resources: [
        { title: "Nsight Systems User Guide", url: "https://docs.nvidia.com/nsight-systems/UserGuide/", type: "docs", note: "CLI flags (--trace, --cuda-graph-trace), stats reports, NVTX" },
        { title: "Nsight Compute Profiling Guide", url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/", type: "docs", note: "what Speed Of Light, memory workload and roofline sections mean" },
        { title: "Nsight Systems", url: "https://developer.nvidia.com/nsight-systems", type: "tool", note: "download the free desktop app to open .nsys-rep files" },
        { title: "MLX — Metal debugger / GPU capture", url: "https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html", type: "docs", note: "capture MLX GPU work into an Xcode .gputrace" },
        { title: "Xcode Metal debugger", url: "https://developer.apple.com/documentation/xcode/metal-debugger", type: "docs", note: "Apple’s tools for inspecting Metal GPU workloads" },
        { title: "Muser book — Ch. 35 Ordering hazards and the dispatch gap", url: "https://highperformanceailab.com/muser-book/chapters/35-ordering-hazards-and-the-dispatch-gap.html", type: "book", note: "optional Mac track: CPU dispatch overhead on Metal" },
      ],
    },

    {
      id: "cuda-graphs",
      title: "Build: CUDA graphs by hand — capture once, replay every token",
      kind: "build",
      minutes: 120,
      runsOn: ["colab"],
      md: MD(function () {/*
A **CUDA graph** records a sequence of kernel launches (with their arguments — including *pointers*) once, then replays the whole sequence with a single `cudaGraphLaunch`. The GPU-side scheduling is precomputed, so per-kernel CPU cost disappears. The price: everything is **frozen** — the same kernels, the same shapes, the same memory addresses on every replay.

That’s why a static KV cache matters: the dynamic cache allocates a new, bigger tensor every step (new addresses, new shapes) — uncapturable. The static cache writes into the same buffers with `index_copy_` at a position that lives in a GPU tensor, and masks the unused tail.

## Capture decode by hand

```python
import torch, time
from tinyllama import Model, Cfg, generate_static

with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")

@torch.no_grad()
def generate_graphed(model, prompt, n_new):
    model.setup_cache(static=True)
    T = prompt.shape[1]
    static_tok = model(prompt, torch.arange(T, device="cuda")).argmax(-1)   # (1, 1) input buffer
    static_pos = torch.tensor([T], device="cuda")                             # (1,)  input buffer

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                        # warm up on a side stream (cuBLAS workspaces, etc.)
        for _ in range(3):
            model(static_tok, static_pos)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                         # record, don't run
        static_out = model(static_tok, static_pos).argmax(-1)

    out = torch.empty(n_new, dtype=torch.long, device="cuda")
    for i in range(n_new):
        out[i] = static_tok[0, 0]
        g.replay()                                    # ~540 kernels, one launch
        static_tok.copy_(static_out)                  # feed the output back into the input buffer
        static_pos += 1                               # the graph reads the position from this buffer
    return out.tolist()

ref = generate_static(model, prompt, 32)
got = generate_graphed(model, prompt, 32)
print("tokens match:", sum(a == b for a, b in zip(ref, got)), "/ 32")

for fn in (generate_static, generate_graphed):
    fn(model, prompt, 32); torch.cuda.synchronize(); t = time.perf_counter()
    fn(model, prompt, 256); torch.cuda.synchronize()
    print(f"{fn.__name__:18s} {256 / (time.perf_counter() - t):6.0f} tok/s")
```

```text
tokens match: 32 / 32
generate_static       110 tok/s
generate_graphed      360 tok/s
```

(Illustrative.) Graphs alone remove the CPU cost, but the GPU still runs ~540 *unfused* kernels — so this lands between eager and compiled. Fusion + graphs together (lesson 7) is what gets you to ~70% MBU.

## The rules of capture

- **No host syncs inside capture** (`.item()`, `print(t)`, `.cpu()`): capture fails with an error.
- **No new allocations that must persist across replays** outside the graph’s private memory pool; inputs/outputs are *fixed buffers* you copy into/out of.
- **Shapes are fixed.** A different batch size or sequence length needs a different graph.
- **CPU-side logic is baked in.** `if` statements on Python values run once, at capture time.
- **Randomness** needs care (PyTorch registers generator state for graph-safe RNG, but custom sampling code should be checked).

## Why vLLM captures graphs for a list of batch sizes

A serving engine’s decode batch size changes every step (requests arrive and finish). vLLM therefore captures one graph per size in `cudagraph_capture_sizes` (e.g. 1, 2, 4, 8, 16, … up to a max), and at runtime **pads** the batch up to the nearest captured size — wasting a little compute to avoid the launch overhead. Attention with variable sequence lengths is the hard part: vLLM supports *piecewise* graphs (capture everything except attention, which runs eagerly between graph pieces) and *full* graphs for decode when the attention backend is graph-compatible. Capture costs startup time and GPU memory; `--enforce-eager` turns it off for debugging. Large prefills typically run without graphs (they’re GPU-bound anyway).

> [!NOTE] Compare with your M10 engine
> If your mini engine batches decode across requests, the same design applies: pick a few batch sizes, preallocate inputs per size, capture, pad. Try it in the challenge.

- [ ] Ran the hand-captured graph; confirmed identical tokens vs `generate_static`
- [ ] Put a `print(static_tok)` inside the captured region and read the error message — then explained it
- [ ] Captured graphs for batch sizes 1, 2, 4, 8 and wrote a `pick_graph(bs)` that pads a batch of 3 to 4
- [ ] Traced `generate_graphed` and found the single `cudaGraphLaunch` per token on the CPU row
      */}),
      resources: [
        { title: "Accelerating PyTorch with CUDA Graphs", url: "https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/", type: "article", note: "torch.cuda.graph API, capture rules and real speedups" },
        { title: "PyTorch — CUDA graphs (CUDA semantics notes)", url: "https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs", type: "docs", note: "capture constraints, memory pools, partial-network capture" },
        { title: "vLLM — CUDA Graphs design", url: "https://docs.vllm.ai/en/latest/design/cuda_graphs.html", type: "docs", note: "capture sizes, padding, piecewise vs full graphs in a real engine" },
      ],
    },

    {
      id: "torch-compile",
      title: "torch.compile: Dynamo, Inductor, graph breaks, modes and static shapes",
      kind: "concept",
      minutes: 120,
      runsOn: ["colab"],
      md: MD(function () {/*
## What happens when you call a compiled function

1. **TorchDynamo** intercepts Python bytecode, traces the tensor operations into an **FX graph**, and installs **guards** (checks on input shapes, dtypes, module attributes, Python values). If guards pass on the next call, the cached compiled code runs; if not, it **recompiles**.
2. **AOTAutograd** (for training) traces the backward too; for inference it functionalizes the graph.
3. **TorchInductor** lowers the graph, **fuses** pointwise/reduction ops, and generates **Triton kernels** for GPUs (the kind you wrote in M12) plus calls to cuBLAS for matmuls — wrapped in generated Python (or C++) “wrapper” code.
4. With `mode="reduce-overhead"`, the wrapper is captured into **CUDA graphs** (via “CUDA graph trees”).

When Dynamo meets something it can’t trace (a `.item()` whose value decides control flow, a `print`, an unsupported library call), it inserts a **graph break**: it compiles the part before, runs the problematic code in plain Python, and starts a new graph after. Each break costs Python overhead and blocks fusion and CUDA graph capture across the break. `fullgraph=True` turns breaks into errors — use it for hot loops.

## See it: graph breaks, recompiles and generated code

```python
import os, torch
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"      # set before compiling (or run from the shell)
from tinyllama import Model, Cfg, decode_one

with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")

def decode_bad(model, tok, pos):
    logits = model(tok, pos)
    print("max logit", logits.max().item())                  # graph break: a host sync + side effect
    return logits.argmax(-1)

model.setup_cache(static=True)
with torch.no_grad():
    tok = model(prompt, torch.arange(16, device="cuda")).argmax(-1)
    pos = torch.tensor([16], device="cuda")
    print(torch._dynamo.explain(decode_bad)(model, tok, pos))   # graph count, break count and reasons
```

Then try the **dynamic cache** with `torch.compile(decode_one)`: each step the KV length grows, so the first compile specializes on a length, the next call fails the shape guard, and Dynamo recompiles with a **dynamic** (symbolic) length. With `mode="reduce-overhead"` you’ll get a warning that a new CUDA graph is being recorded for each distinct input size — hundreds of graphs, each used once. **Static shapes** (a preallocated cache + a mask) are what make decode compile to one graph.

To see what Inductor generated — the Triton kernels and the wrapper — run with `TORCH_LOGS="output_code"` and search for `@triton.jit`. You’ll find kernels named after what they fused (e.g. RMSNorm’s pow/mean/rsqrt/mul in one kernel; RoPE and the residual add fused into neighbors). Count them: this is the ~540 → ~110 kernel reduction.

## Modes and knobs

| Option | What it does | When |
|---|---|---|
| `mode="default"` | fusion + Triton codegen; no CUDA graphs | good first step; fewest surprises |
| `mode="reduce-overhead"` | + CUDA graphs (static input copies, graph trees) | small batches, launch-bound decode |
| `mode="max-autotune"` | + benchmark Triton GEMM templates/configs vs cuBLAS; CUDA graphs | when kernels themselves matter; long compile |
| `mode="max-autotune-no-cudagraphs"` | autotuning without graphs | when graphs are impossible/undesired |
| `fullgraph=True` | error on any graph break | hot paths you intend to be one graph |
| `dynamic=False` / `torch._dynamo.mark_dynamic(t, dim)` | force static / declare a dynamic dim up front | avoid recompiles, or avoid over-specialization |

> [!WARNING] Compile time is a production cost
> Compiling a 7B model can take minutes. Serving engines cache compiled artifacts (Inductor’s FX graph cache is on by default; PyTorch also supports saving/loading “mega-cache” artifacts) so a new replica doesn’t pay the cost at scale-up (Inference Engineering (Baseten) ch. 4.2 discusses this cold-start trade-off). Also: `torch.compile` can’t fuse *through* opaque custom kernels (e.g. a hand-written CUDA attention kernel) — fusion stops at their boundaries.

## Other compilers you’ll meet

**TensorRT / TensorRT-LLM** (NVIDIA’s ahead-of-time engine builder), **XLA** (JAX/TPU), **MLX `mx.compile`** (Mac), and vLLM’s own `torch.compile` integration with custom fusion passes. The concepts — graph capture, fusion, static shapes, autotuning, graph replay — transfer directly.

- [ ] Ran `torch._dynamo.explain` on `decode_bad` and on `decode_one`; recorded graph/break counts for both
- [ ] Compiled the dynamic-cache loop, observed recompiles in the logs, and counted how many graphs were produced for 64 tokens
- [ ] Dumped `output_code` for the static-cache model and identified the fused RMSNorm kernel; counted Triton kernels per layer
- [ ] Measured compile time (first call) for `default`, `reduce-overhead` and `max-autotune` and added it to your results table
      */}),
      resources: [
        { title: "Introduction to torch.compile", url: "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", type: "docs", note: "official tutorial: usage, modes, graph breaks" },
        { title: "torch.compiler docs", url: "https://docs.pytorch.org/docs/stable/torch.compiler.html", type: "docs", note: "Dynamo, Inductor, dynamic shapes, CUDA graph trees" },
        { title: "Dynamo overview", url: "https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html", type: "docs", note: "how bytecode tracing, guards and graph breaks work" },
        { title: "torch.compile troubleshooting", url: "https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html", type: "docs", note: "TORCH_LOGS, explain(), recompilation and graph-break debugging" },
        { title: "CUDA graph trees", url: "https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html", type: "docs", note: "how reduce-overhead manages graphs, and why outputs get overwritten" },
        { title: "Compile-time caching", url: "https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html", type: "docs", note: "reuse compiled artifacts to avoid cold-start compile" },
      ],
    },

    {
      id: "lab-decode",
      title: "Lab: optimize a decode loop to near bandwidth-bound, with a results table",
      kind: "lab",
      minutes: 240,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
Now run the full loop end to end, one change per row. Use `tinyllama.py` on a T4 (or your M10 engine’s model; or gpt-fast on a cloud GPU — the variants map one-to-one).

## The variants

```python
import torch, time
from tinyllama import Model, Cfg, generate_eager, generate_static, decode_one, bench, bytes_per_token

torch.manual_seed(0)
with torch.device("cuda"):
    model = Model(Cfg()).half().eval()
prompt = torch.randint(0, 32000, (1, 16), device="cuda")

variants = {
    "v0 eager, dynamic cache, .item()": (generate_eager, {}),
    "v1 static cache, no per-token sync": (generate_static, {}),
    "v2 compile default": (generate_static, {"step": torch.compile(decode_one, fullgraph=True)}),
    "v3 compile reduce-overhead": (generate_static, {"step": torch.compile(decode_one, mode="reduce-overhead", fullgraph=True)}),
    "v4 compile max-autotune": (generate_static, {"step": torch.compile(decode_one, mode="max-autotune", fullgraph=True)}),
}

ref = generate_static(model, prompt, 32)
model.setup_cache(static=True); bpt = bytes_per_token(model); peak = 320e9       # T4; set for your GPU
for name, (gen, kw) in variants.items():
    torch._dynamo.reset()                          # fresh compile caches: variants share decode_one's code object
    t0 = time.perf_counter(); gen(model, prompt, 8, **kw); first = time.perf_counter() - t0   # includes compile
    runs = sorted(bench(model, prompt, gen, **kw) for _ in range(5))
    tps = runs[2]                                                                              # median of 5
    same = sum(a == b for a, b in zip(ref, gen(model, prompt, 32, **kw)))
    print(f"{name:38s} {1e3 / tps:6.2f} ms/tok {tps:6.0f} tok/s  MBU {tps * bpt / peak:4.0%}  "
          f"first call {first:5.1f} s  tokens match {same}/32")
```

(With FP16 and random weights, compiled kernels may reorder floating-point sums; an occasional mismatch late in the sequence is expected. If early tokens differ, compare the first step’s logits: max abs difference should be ~1e-2 or less.) `torch.compile` is lazy, so each variant compiles on its first call, right after `torch._dynamo.reset()`; if you see recompile-limit warnings or odd results, restart the runtime and run one variant per session.

## Place each variant on the roofline

Each decode step is ~1 FLOP/byte. Open the T4 preset and drag the intensity slider to ~1: the ceiling is the bandwidth line. Your variants all sit at (nearly) the **same intensity** — they differ only in *how close to the roof* they get. Before v3, the distance to the roof is overhead (idle GPU); after v3 it’s kernel efficiency. That’s the story your report must tell.

::viz roofline

## Profile every row

For each variant, save a trace (lesson 4’s `trace()` helper or `nsys`) and write one sentence for the “what the trace showed” column. Expected story on a T4:

1. **v0 → v1**: syncs and HtoD copies disappear; `cat`/allocator churn disappears. Modest gain — the step is still launch-bound.
2. **v1 → v2**: kernel count per token drops ~5× (fusion). Gaps shrink but remain (Python wrapper + guards launch every kernel).
3. **v2 → v3**: the CPU row collapses to one graph launch per token; GPU kernels are back-to-back. Biggest jump.
4. **v3 → v4**: small gain or none — the step is now dominated by GEMVs already near the bandwidth roof; autotuning helps the long tail.

## Go further (pick at least one)

- **The LM head.** At 32k vocab × 1024 dim, the output projection is 65 MB — ~15% of bytes per token. Check its share of GPU time in the trace.
- **Attention over the whole static cache.** v1+ reads all `max_seq` positions every step (masked). Halve `max_seq` and predict the change in bytes/token and tok/s; measure.
- **Batch size.** Rerun v3 at batch 1, 4, 16, 64. Plot total tok/s and per-sequence tok/s — you’re now on the latency–throughput curve of M09.
- **Your M10 engine.** Apply the same steps to your engine’s decode loop (static cache or paged cache with fixed-size block tables; graphs per batch size).

- [ ] Filled the results table (v0–v4) with median ms/token, tok/s, MBU, first-call time and a trace per row
- [ ] Every row has a *prediction* written before the measurement, and a sentence on why it differed
- [ ] Explained the v3 → ceiling gap: which kernels are below the roof, and by how much (use `ncu` or kernel times ÷ bytes)
- [ ] Completed one “go further” experiment with a before/after trace
      */}),
      resources: [
        { title: "Accelerating Generative AI with PyTorch II: GPT, Fast", url: "https://pytorch.org/blog/accelerating-generative-ai-2/", type: "article", note: "the same lab on Llama-7B, with traces for each step" },
        { title: "Hugging Face — LLM inference optimization", url: "https://huggingface.co/docs/transformers/main/en/llm_optims", type: "docs", note: "static KV cache + torch.compile in transformers’ generate()" },
      ],
    },

    {
      id: "deep-dive",
      title: "Deep dive: autotuning, TunableOp, NCCL traces and multi-GPU timelines",
      kind: "deep",
      optional: true,
      minutes: 180,
      runsOn: ["cloud", "multi-gpu"],
      md: MD(function () {/*
## Kernel autotuning

The best tile sizes, warps and pipeline stages depend on the shape *and* the GPU. Three layers of autotuning you’ll meet:

- **Triton `@triton.autotune`** (M12): benchmark a list of configs per key (e.g. `N`), cache the winner.
- **Inductor**: `mode="max-autotune"` benchmarks Triton GEMM templates against cuBLAS; `torch._inductor.config.coordinate_descent_tuning = True` further tunes generated kernels’ block sizes (gpt-fast enables it, along with `triton.unique_kernel_names` for readable traces and `fx_graph_cache`). Browse the knobs in `torch/_inductor/config.py`.
- **PyTorch TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`): benchmarks cuBLAS/hipBLAS algorithm choices for each GEMM shape your program runs and writes the winners to a CSV you can ship with a deployment.

Autotuning has a cost (compile/startup time) and a trap: tuning on one shape and serving another. Tune on production shapes (the batch sizes you capture graphs for).

## NCCL and multi-GPU traces (preview of M16)

With tensor parallelism, each layer ends in an all-reduce. In a `torch.profiler` or `nsys` trace you’ll see `ncclDevKernel_AllReduce_...` kernels on each rank’s stream. Questions to ask of a multi-GPU trace:

- Is communication **overlapped** with compute, or does compute stall waiting for NCCL?
- Are ranks **skewed** (one GPU starts the all-reduce late — a straggler — so all others wait)?
- Is the achieved bus bandwidth close to NVLink/PCIe spec? (`nccl-tests` gives the reference.)

Useful switches: `NCCL_DEBUG=INFO` (topology, algorithm and protocol chosen), `NCCL_DEBUG_SUBSYS=INIT,COLL`, and `nsys profile --trace=cuda,nvtx,osrt` per rank. **Holistic Trace Analysis (HTA)** loads per-rank `torch.profiler` traces and computes temporal breakdowns (compute / communication / idle), comm–compute overlap and kernel stats across ranks.

## How production engines apply this

- **vLLM** runs the model through `torch.compile` with custom passes (e.g. fusing norms and quantization ops), caches the compiled artifacts, and captures CUDA graphs per batch size; read its torch.compile design doc next to the CUDA graphs one.
- **SGLang / TensorRT-LLM** capture decode graphs too, and rely on hand-written kernels (FlashInfer, CUTLASS) where fusion across library boundaries isn’t possible.
- For the Mac track, Muser’s chapter on **measuring against llama.cpp** shows the same discipline: a fixed benchmark, a reference implementation, and traces to explain every gap.

- [ ] Enabled `coordinate_descent_tuning` for v3 and measured the change in compile time and tok/s
- [ ] Ran a GEMM benchmark with `PYTORCH_TUNABLEOP_ENABLED=1` and inspected the result CSV
- [ ] (Multi-GPU) Traced a 2-GPU tensor-parallel forward, loaded it in HTA, and reported the comm/compute overlap
      */}),
      resources: [
        { title: "Inductor config.py", url: "https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py", type: "repo", note: "every Inductor knob, with comments" },
        { title: "PyTorch TunableOp", url: "https://docs.pytorch.org/docs/stable/cuda.tunable.html", type: "docs", note: "autotune GEMM algorithm selection per shape" },
        { title: "Holistic Trace Analysis", url: "https://github.com/facebookresearch/HolisticTraceAnalysis", type: "tool", note: "multi-rank trace analysis: idle time, overlap, kernel breakdowns" },
        { title: "NCCL environment variables", url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html", type: "docs", note: "NCCL_DEBUG and friends for communication debugging" },
        { title: "vLLM — torch.compile integration", url: "https://docs.vllm.ai/en/latest/design/torch_compile.html", type: "docs", note: "how a production engine compiles, caches and graphs its model" },
        { title: "GPU MODE lectures", url: "https://github.com/gpu-mode/lectures", type: "course", note: "lectures on profiling, torch.compile and Nsight" },
      ],
    },
  ],

  challenge: {
    title: "Profiling report: from slow Python to bandwidth-bound, with evidence",
    md: MD(function () {/*
In `course-work/m13/`, deliver `REPORT.md` + notebook + trace files for **your M10 mini engine’s decode loop** (preferred), `tinyllama.py`, or gpt-fast:

1. **Baseline.** Model size, bytes per token, GPU, peak bandwidth, bandwidth-bound ceiling (tok/s), baseline median ms/token and MBU, with a trace screenshot of one decode step annotated (CPU row, GPU row, gaps, syncs).
2. **At least three optimizations**, applied one at a time (e.g. remove host syncs, static KV cache, `torch.compile`, CUDA graphs, fusing a custom op, reducing bytes per token). For each: hypothesis + predicted gain, the change (diff), measured result (median + p90 of ≥ 5 runs), correctness check, and a before/after trace pair.
3. **Roofline explanation.** Put baseline and final on the roofline for your GPU; explain what fraction of the gap to the roof was *overhead* vs *kernel efficiency*, and what the next lever would be (quantization, batching, speculative decoding).
4. **Costs.** Compile time, CUDA-graph capture time and memory, and any flexibility you gave up (fixed shapes, max sequence length).
    */}),
    checklist: [
      "Baseline and final numbers are medians of ≥ 5 runs with p90, measured with synchronization and after warmup",
      "At least 3 optimizations, each with a predicted gain written *before* measuring, and a measured impact",
      "Before/after trace (Perfetto or Nsight screenshot) for every optimization, with the relevant feature circled or labeled",
      "Correctness check for every variant (tokens match or logits within tolerance)",
      "Final MBU ≥ 60% at batch 1 (or a trace-backed explanation of what prevents it on your hardware)",
      "Roofline figure with baseline and final points and a paragraph on the remaining gap",
      "Compile/capture time and memory overhead reported",
    ],
    stretch: "Add CUDA graphs to your **M10 engine’s batched decode**: capture graphs for batch sizes 1, 2, 4, 8, 16, 32, pad the live batch to the nearest size, and show a latency–throughput plot with and without graphs. Or: use `ncu` to find the slowest non-GEMM kernel in the compiled step and replace it with a Triton kernel from M12, measuring the end-to-end impact.",
  },

  connects: MD(function () {/*
M11 and M12 made individual kernels fast; this module made the **program** fast — and gave you the tools to tell those two problems apart. With decode now close to the bandwidth roof, the only remaining ways up are to **read fewer bytes per token** (**M14**, quantization: int8/int4 weights and FP8 KV), to **produce more tokens per byte read** (**M15**, speculative decoding; **M09**, batching), or to **spread the bytes over more GPUs** (**M16**, where NCCL traces become part of every profile). In **M18–M19** you’ll profile a production server (vLLM/SGLang) under load, where the same patterns — gaps, syncs, graph coverage — explain p99 latency.
  */}),

  interview: [
    "A batch-1 decode step on an H100 takes 25 ms for a 7B FP16 model. What’s the bandwidth-bound floor, and how would you find where the rest goes?",
    "What does `mode=\"reduce-overhead\"` do in `torch.compile`, and why does it require a static KV cache for decode?",
    "Name five things in a PyTorch decode loop that cause a host–device synchronization, and how each shows up in a trace.",
    "What is a graph break in `torch.compile`? How do you find them, and why do they hurt performance?",
    "Why does vLLM capture CUDA graphs for a list of batch sizes instead of one? What are the costs?",
    "You profile a kernel with Nsight Compute: DRAM throughput 20% of peak, compute throughput 10% of peak. What might be wrong, and what would you try?",
    "How do you benchmark a GPU function correctly? List the pitfalls you control for.",
    "Explain the difference between Nsight Systems and Nsight Compute, and when you’d reach for `torch.profiler` instead.",
  ],

  resources: [
    { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "the reference fast decode loop in ~1000 lines of PyTorch" },
    { title: "Accelerating Generative AI with PyTorch II: GPT, Fast", url: "https://pytorch.org/blog/accelerating-generative-ai-2/", type: "article", note: "profiling-driven optimization of Llama-7B decode, step by step" },
    { title: "Horace He — Making Deep Learning Go Brrrr", url: "https://horace.io/brrr_intro.html", type: "article", note: "the compute / bandwidth / overhead mental model" },
    { title: "PyTorch Profiler recipe", url: "https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", type: "docs", note: "capturing and reading PyTorch traces" },
    { title: "Introduction to torch.compile", url: "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", type: "docs", note: "Dynamo + Inductor basics and modes" },
    { title: "Accelerating PyTorch with CUDA Graphs", url: "https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/", type: "article", note: "CUDA graph capture in PyTorch, with rules and results" },
    { title: "Nsight Systems User Guide", url: "https://docs.nvidia.com/nsight-systems/UserGuide/", type: "docs", note: "whole-system timelines, NVTX, CUDA graph tracing" },
    { title: "Nsight Compute Profiling Guide", url: "https://docs.nvidia.com/nsight-compute/ProfilingGuide/", type: "docs", note: "kernel-level metrics and roofline analysis" },
    { title: "vLLM — CUDA Graphs design", url: "https://docs.vllm.ai/en/latest/design/cuda_graphs.html", type: "docs", note: "how a production engine captures and pads decode graphs" },
    { title: "Inference Engineering (Baseten) — Ch. 4.2, 4.5", url: "Inference%20Engineering.pdf", type: "book", note: "compilation, CUDA graphs, cold starts and profiling tools" },
    { title: "Muser book — Ch. 35 Ordering hazards and the dispatch gap", url: "https://highperformanceailab.com/muser-book/chapters/35-ordering-hazards-and-the-dispatch-gap.html", type: "book", note: "optional Mac/Metal track: the dispatch gap on Apple GPUs" },
  ],
});
