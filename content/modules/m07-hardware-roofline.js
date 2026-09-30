Course.module({
  id: "m07-hardware-roofline",
  title: "GPUs & the roofline: why decode is memory-bound",
  short: "GPUs & roofline",
  tagline: "Predict the tokens/sec of any model on any GPU before you run it — then verify the prediction on your Mac and a Colab T4. The roofline model is the single most useful piece of napkin math in inference.",
  hours: 14,
  runsOn: ["mac", "colab"],
  tags: ["gpu", "roofline", "arithmetic-intensity", "memory-bandwidth", "mfu", "mbu", "hardware"],

  goal: MD(function () {/*
By the end of this module you will be able to do this in your head, **before** renting anything:

> Llama-3-8B in BF16 is ~16 GB of weights. Every decode step at batch 1 must read all of them. An H100 SXM streams 3.35 TB/s. So one token takes at least 16 GB ÷ 3.35 TB/s ≈ **4.8 ms → at most ~200 tokens/sec per user.** The H100's 989 TFLOPS are irrelevant here — the step needs only 16 GFLOP, which the tensor cores finish in 0.016 ms.

…and then prove it on hardware you own, with a script you wrote:

```text
$ python napkin.py --model qwen2.5-1.5b --gpu m2pro --prompt 128 --output 256
qwen2.5-1.5b on m2pro, prompt=128, output=256, MFU=0.6, MBU=0.75
batch  mem GB  TTFT ms  TPOT ms   bound tok/s/user tok/s total
    1     3.1     97.0    20.58  memory         49          48

$ mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 --max-tokens 256 --prompt "..."
Prompt: 38 tokens, 812.4 tokens-per-sec
Generation: 256 tokens, 52.7 tokens-per-sec          <-- within 10% of prediction
```

(Illustrative Mac output — your chip and numbers will differ.) You'll also measure your own chip's **memory bandwidth** and **matmul TFLOPS**, and plot your own **roofline** with real operations on it.

::viz roofline
  */}),
  demo: { viz: "roofline", params: {} },

  why: MD(function () {/*
"Which GPU should we buy/rent for this model?", "Why is our H100 only at 5% utilization?", "Will FP8 make it faster?", "How big a batch before we become compute-bound?" — these are the everyday questions of an inference engineer, and they all have the same answer: **compare the arithmetic intensity of the work with the ops:byte ratio of the hardware.** Baseten's *Inference Engineering* (ch. 2.4 and 3) builds its whole hardware chapter on this idea. Interviewers love it because it separates people who've memorized "decode is memory-bound" from people who can *derive* it, put numbers on it, and predict where it stops being true. This module turns the slogan into a calculator you trust.
  */}),

  prereqs: [
    {
      title: "Module 06: prefill/decode, KV cache, TTFT/TPOT",
      skipIf: "you can compute KV bytes per token and explain TTFT vs TPOT",
      md: MD(function () {/*
This module quantifies what Module 06 showed empirically. You should know: prefill processes all prompt tokens in one pass (TTFT); decode generates one token per pass (TPOT); decode reads **all weights + the KV cache** every step; KV bytes per token $= 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b$. If any of that is fuzzy, reread Module 06 lessons *prefill-decode* and *flops-bytes* first.
      */}),
    },
    {
      title: "Ratios, units and log scales",
      skipIf: "you're comfortable with TB/s vs GB/s, TFLOPS, and log-log plots",
      math: true,
      md: MD(function () {/*
**Units.** Tera = $10^{12}$, Giga = $10^{9}$. 1 TFLOPS = $10^{12}$ floating-point operations per second; one multiply-add counts as **2 FLOPs**. 1 TB/s = 1,000 GB/s. Careful: networking uses **bits** (Gb/s) — 400 Gb/s InfiniBand is only 50 GB/s.

**Ratios.** Time for a job limited by one resource = amount ÷ rate. 16 GB at 3.35 TB/s: $\frac{16 \times 10^9}{3.35 \times 10^{12}} = 4.8 \times 10^{-3}$ s. Keep powers of ten separate and you'll never be off by 1000×.

**Log scales.** On a log axis, equal distances mean equal *ratios*: 1 → 10 → 100 are evenly spaced. A line $y = c \cdot x$ becomes a straight line of slope 1 on a log-log plot, shifted up or down by $\log c$. That's why the roofline's bandwidth ceiling is a 45° diagonal: achievable FLOPS = bandwidth × intensity. Log scales let you see a 0.5 FLOPs/byte decode step and a 1,000 FLOPs/byte prefill on the same chart.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: predict tok/s, then measure it on your Mac and a T4",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
The prediction rule for batch-1 decode is one line:

$$\text{max tok/s} \approx \frac{\text{memory bandwidth}}{\text{bytes of weights}}$$

Let's test it before understanding it. Look up your chip's memory bandwidth (Apple lists it on the tech-specs page), then fill in the prediction **before** running anything.

| Chip | Bandwidth | Qwen2.5-1.5B bf16 (3.09 GB) | Qwen2.5-0.5B bf16 (0.99 GB) |
|---|---|---|---|
| M1 / M2 | 68 / 100 GB/s | ~22 / ~32 tok/s | ~69 / ~101 tok/s |
| M2 Pro / M1 Pro | 200 GB/s | ~65 tok/s | ~202 tok/s |
| M4 Pro | 273 GB/s | ~88 tok/s | ~276 tok/s |
| M3 Max / M2 Max | 300–400 GB/s | ~97–129 tok/s | ~300–400 tok/s |
| M4 Max | 410–546 GB/s | ~133–177 tok/s | ~414–550 tok/s |
| NVIDIA T4 (Colab) | 320 GB/s | ~104 tok/s | ~323 tok/s |

These are **ceilings**. Real engines reach 60–90% of them (that fraction is **MBU**, Lesson 7).

**On your Mac** (MLX is the fastest local engine on Apple Silicon):

```bash
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \
  --prompt "Explain how a lighthouse works, in detail." --max-tokens 256
mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --prompt "Explain how a lighthouse works, in detail." --max-tokens 256
```

It prints `Prompt: … tokens-per-sec` (prefill) and `Generation: … tokens-per-sec` (decode). Compute measured ÷ predicted for both models.

**On Colab (Runtime → T4 GPU)** with plain Hugging Face:

```python
# t4_predict.py
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).cuda().eval()  # T4 has no bf16
w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"weights {w_bytes / 1e9:.2f} GB -> ceiling {320e9 / w_bytes:.0f} tok/s at 320 GB/s")

inputs = tok("Explain how a lighthouse works, in detail.", return_tensors="pt").to("cuda")
model.generate(**inputs, max_new_tokens=16)          # warm-up

def run(n):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    torch.cuda.synchronize(); return time.perf_counter() - t0

t1, t257 = run(1), run(257)
tpot = (t257 - t1) / 256                              # subtract prefill, keep decode
print(f"TPOT {1e3 * tpot:.1f} ms -> {1 / tpot:.0f} tok/s")
```

> [!WARNING] Don't be surprised if the T4 misses badly
> Eager Hugging Face `generate()` launches hundreds of small GPU kernels per token from Python, and Colab's CPU is slow. You'll often measure 30–50% of the ceiling — the step is **overhead-bound**, not memory-bound (Lesson 3). Try `model.generation_config.cache_implementation = "static"` plus `model.forward = torch.compile(model.forward, mode="reduce-overhead")` and watch it climb toward the ceiling. MLX on the Mac usually lands much closer, because its per-token overhead is small.

Now try a few what-ifs with the same rule: 4-bit weights (`mlx-community/Qwen2.5-1.5B-Instruct-4bit`, ~0.9 GB) should be roughly 3× faster. Batch 1 vs a batch of 8 prompts should cost almost the same time per step. A 7B model should be ~5× slower than 1.5B. Every one of those predictions comes from dividing bytes by bandwidth.

- [ ] Wrote down predictions for two models on your Mac *before* running
- [ ] Measured decode tok/s with `mlx_lm.generate`; computed measured ÷ predicted
- [ ] Ran `t4_predict.py` on Colab; noted how far eager HF is from the ceiling
- [ ] Tried a 4-bit model and checked the speedup matches the byte ratio
      */}),
      resources: [
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "`mlx_lm.generate` prints prefill and decode tok/s" },
        { title: "Qwen2.5-1.5B-Instruct (MLX bf16)", url: "https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-bf16", type: "tool", note: "model used in the Mac measurements" },
      ],
    },
    {
      id: "gpu-anatomy",
      title: "GPU architecture for software engineers",
      kind: "concept",
      minutes: 90,
      md: MD(function () {/*
A GPU is a **throughput machine**: it applies one operation to thousands of independent data elements at once, and hides the latency of any single element by always having other work ready (Inference Engineering (Baseten) ch. 3.1). Here's the H100 SXM, top-down.

::viz gpu-anatomy

**Compute: SMs, warps, tensor cores.**
- The chip is **132 Streaming Multiprocessors (SMs)**. Think of an SM as an independent core with its own scheduler, registers and fast scratch memory.
- Each SM runs up to 64 **warps**; a warp is **32 threads** that execute the same instruction in lockstep (SIMT). A kernel launch is a grid of *thread blocks*; each block lands on one SM and is split into warps. When one warp waits on memory (hundreds of cycles), the SM switches to another in one clock — that's how GPUs hide latency.
- **CUDA cores** do scalar FP32/INT math. **Tensor Cores** (4 per SM, 528 total) do small matrix multiply-accumulates, $D = A \times B + C$, in one instruction (MMA). Virtually all inference FLOPs are tensor-core FLOPs, so **always quote tensor-core FLOPS** at the right precision: 989 TFLOPS dense BF16, 1,979 dense FP8.
- **SFUs** (special function units) handle `exp`, `sin`, `log` — they matter for softmax.

**Memory: a hierarchy of speed vs size.**

::viz memory-hierarchy

| Level | H100 size | Rough bandwidth | Who manages it |
|---|---|---|---|
| Registers | 256 KB per SM | fastest | compiler |
| L1 / shared memory (SMEM, SRAM) | 256 KB per SM (~33 MB total) | tens of TB/s aggregate | **you** (kernel author) |
| L2 cache (SRAM) | 50 MB | ~10 TB/s-class | hardware |
| HBM3 (VRAM, DRAM) | 80 GB | **3.35 TB/s** | allocator (PyTorch, vLLM) |
| NVLink to other GPUs | — | 900 GB/s (bidirectional) | NCCL |
| PCIe Gen5 to CPU | — | ~64 GB/s each way | driver |
| Network (InfiniBand) | — | 400 Gb/s = 50 GB/s per NIC | NCCL / RDMA |

(SMEM/L2 figures are order-of-magnitude; vendor docs rarely state them.) Each step down is ~10× bigger and ~3–10× slower. Model weights and the KV cache live in **HBM**, and every forward pass drags them through the SMs. A fast kernel (FlashAttention, a good GEMM) is one that loads a tile from HBM into SMEM **once** and reuses it as many times as possible before writing results back.

**Interconnects.** Inside one node, 8 GPUs connect via **NVLink/NVSwitch** (900 GB/s on Hopper, 1,800 GB/s on Blackwell). Between nodes, **InfiniBand** at 400 Gb/s per NIC — about an order of magnitude slower (Baseten ch. 3.3). That gap is why tensor parallelism stays inside a node (Module 13).

> [!INTUITION] Apple Silicon is the same story, simpler
> An M-series chip has GPU cores (Apple's equivalent of SMs), on-chip caches, and **unified memory** shared by CPU and GPU — LPDDR5 at 100–819 GB/s depending on the chip. No PCIe copy between host and device, and far more capacity (up to 512 GB on an M3 Ultra), but less bandwidth than HBM. It's the same trade-off: capacity vs speed (Baseten ch. 3.5 compares an RTX 5090 at 1,792 GB/s/32 GB with an M3 Ultra at 819 GB/s/512 GB).

> [!REAL] What this means for inference
> - Decode performance ≈ how fast you can stream HBM → SMs. More HBM bandwidth (H200 over H100) = more tok/s (Baseten: "Bandwidth bounds decode at low-to-medium batch").
> - Prefill performance ≈ tensor-core FLOPS.
> - VRAM capacity decides whether the model fits and how much room is left for KV cache (Baseten's rule of thumb: weights **plus at least 50% headroom**).

- [ ] Can explain SM, warp, tensor core, SMEM, HBM in one sentence each
- [ ] Can say which memory level holds weights, KV cache and a FlashAttention tile
- [ ] Knows NVLink vs InfiniBand vs PCIe bandwidths to within 2×
      */}),
      resources: [
        { title: "Modal GPU Glossary", url: "https://modal.com/gpu-glossary", type: "docs", note: "the best plain-English reference for every GPU term" },
        { title: "Aleksa Gordić — Inside NVIDIA GPUs", url: "https://www.aleksagordic.com/blog/matmul", type: "article", note: "Hopper architecture + how fast matmul kernels are built" },
      ],
    },
    {
      id: "cpu-vs-gpu",
      title: "CPU vs GPU: async launches, timing correctly, and the overhead regime",
      kind: "concept",
      minutes: 60,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
**Mental model.** A CPU has a few big, latency-optimized cores with huge caches and branch predictors — great at one complicated thing at a time. A GPU has thousands of small lanes optimized for **throughput** — terrible at one thing, amazing at a million identical things. In PyTorch, the CPU (your Python process) is the *conductor* and the GPU is the *orchestra*: Python decides what to run and **enqueues kernels**; the GPU executes them asynchronously.

**Consequence 1: timing without synchronizing is a lie.** `y = x @ W` returns as soon as the kernel is *queued*, not when it's done.

```python
import time, torch
dev = "cuda" if torch.cuda.is_available() else "mps"
sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize
x = torch.randn(8192, 8192, device=dev, dtype=torch.float16)

t0 = time.perf_counter(); y = x @ x; t1 = time.perf_counter()
print(f"no sync: {1e3 * (t1 - t0):.2f} ms")          # suspiciously fast
sync(); t0 = time.perf_counter(); y = x @ x; sync(); t1 = time.perf_counter()
print(f"synced:  {1e3 * (t1 - t0):.2f} ms")          # the real time

if dev == "cuda":                                     # CUDA events time on the GPU itself
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); y = x @ x; end.record(); torch.cuda.synchronize()
    print(f"events:  {start.elapsed_time(end):.2f} ms")
```

Always: warm up first (the first call compiles/allocates), sync before starting the clock, sync before stopping it, repeat and take a median. `torch.utils.benchmark.Timer` does all of this for you.

**Consequence 2: the overhead regime.** Every kernel launch costs a few microseconds of CPU time (Python + PyTorch dispatch + driver). A decode step of a 28-layer model launches hundreds of kernels. If each kernel's GPU work is *shorter* than its launch cost, the GPU sits idle waiting for Python. Horace He calls this the third regime, after compute-bound and memory-bound: **overhead-bound**.

```python
# overhead.py — time per op vs tensor size: flat = overhead, rising = bandwidth
import time, torch
dev = "cuda" if torch.cuda.is_available() else "mps"
sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize
for p in range(10, 29, 2):
    x = torch.zeros(2**p, device=dev)
    for _ in range(10): x.add_(1)
    sync(); t0 = time.perf_counter()
    for _ in range(200): x.add_(1)
    sync(); us = (time.perf_counter() - t0) / 200 * 1e6
    print(f"{2**p:>10} elems  {us:8.1f} us/op  {2 * x.nbytes / us / 1e3:8.1f} GB/s")
```

Up to ~$10^5$–$10^6$ elements the time per op is flat (a few µs): pure overhead. Past that it grows linearly and the GB/s column plateaus near your memory bandwidth. Small models at batch 1 live in the flat region — which is why your Colab T4 missed the ceiling in Lesson 1.

> [!REAL] How engines kill overhead
> - **CUDA graphs**: record the whole decode step's kernel sequence once, replay it with one launch. vLLM captures graphs for a set of batch sizes at startup (that's the "Capturing CUDA graphs" line in its log).
> - **torch.compile** fuses elementwise ops into fewer kernels; `mode="reduce-overhead"` adds CUDA graphs. gpt-fast reaches near-roofline decode this way in < 1,000 lines.
> - **Fused kernels** (RMSNorm+residual, fused QKV, SwiGLU) reduce both launches and HBM round-trips.
> - A C++/Rust scheduler loop around the model (SGLang, TensorRT-LLM, llama.cpp) keeps Python off the hot path.

- [ ] Reproduced the "no sync" timing lie
- [ ] Ran `overhead.py`; found the size where your device leaves the overhead regime
- [ ] Can name three techniques engines use to remove launch overhead
      */}),
      resources: [
        { title: "Horace He — Making Deep Learning Go Brrrr From First Principles", url: "https://horace.io/brrr_intro.html", type: "article", note: "compute vs memory vs overhead — read it twice" },
        { title: "PyTorch — CUDA semantics (asynchronous execution)", url: "https://docs.pytorch.org/docs/stable/notes/cuda.html", type: "docs", note: "streams, events, why sync matters" },
        { title: "gpt-fast", url: "https://github.com/pytorch-labs/gpt-fast", type: "repo", note: "torch.compile + CUDA graphs decode near the bandwidth ceiling" },
      ],
    },
    {
      id: "roofline-math",
      title: "Math: arithmetic intensity & the roofline",
      kind: "math",
      minutes: 90,
      md: MD(function () {/*
Two numbers decide everything.

**Hardware: ops:byte ratio** = peak FLOPS ÷ memory bandwidth. It says how many FLOPs the chip can do in the time it takes to fetch one byte.

$$\text{ops:byte}_{H100} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{B/s}} \approx 295\ \text{FLOPs/byte}$$

(Inference Engineering (Baseten) ch. 2.4 — dense BF16.)

**Algorithm: arithmetic intensity (AI)** = FLOPs performed ÷ bytes moved to/from memory.

**The roofline**: attainable FLOPS $= \min(\text{peak FLOPS},\ \text{bandwidth} \times \text{AI})$. On a log-log plot it's a 45° ramp (memory-bound) meeting a flat roof (compute-bound) at the **ridge point** AI = ops:byte.

::viz roofline

- AI **below** the ridge → **memory-bound**: time ≈ bytes ÷ bandwidth. Extra FLOPs are free.
- AI **above** the ridge → **compute-bound**: time ≈ FLOPs ÷ peak. Extra bytes are free.

> [!MATH] Intensity of a matmul
> For $Y = X W$ with $X \in \mathbb{R}^{M \times K}$, $W \in \mathbb{R}^{K \times N}$ in 2-byte precision:
> FLOPs $= 2MKN$. Bytes $= 2(MK + KN + MN)$. So
> $$\text{AI} = \frac{MKN}{MK + KN + MN}$$
> - **Decode, batch 1** ($M=1$): $\text{AI} = \frac{KN}{K + KN + N} \approx 1$. Deep in memory-bound territory (295× below the H100 ridge).
> - **Decode, batch B** ($M=B \ll K, N$): $\text{AI} \approx B$. Each weight byte fetched is used $B$ times.
> - **Prefill, P = 2,048 tokens** with $K = N = 4096$: $\text{AI} = \frac{2048 \cdot 4096 \cdot 4096}{2048 \cdot 4096 + 4096^2 + 2048 \cdot 4096} = 1024$. Well above 295 → compute-bound.
> - **Square matmul** $n \times n$: $\text{AI} = n/3$. That's why benchmark GEMMs use $n \ge 4096$.

**Worked example — Llama-3-8B decode, batch 1, H100.** One step: FLOPs $\approx 2N = 16.1$ GFLOP; bytes $\approx 16.1$ GB (every weight, once). AI $\approx 1$.

- Memory time: $\frac{16.1 \times 10^9}{3.35 \times 10^{12}} = 4.8$ ms.
- Compute time: $\frac{16.1 \times 10^9}{989 \times 10^{12}} = 0.016$ ms.

The step takes the **max** of the two (they overlap) → 4.8 ms → ≤ ~205 tok/s. Compute is 300× underused.

**Worked example — prefill of 1,024 tokens.** FLOPs $= 2 \cdot 8.03 \times 10^9 \cdot 1024 \approx 16.4$ TFLOP. Bytes ≈ still ~16 GB. Compute time 16.6 ms, memory time 4.8 ms → compute-bound; TTFT ≈ 17 ms at 100% of peak, ~25–30 ms realistically.

**Worked example — attention (Baseten).** Unfused standard attention with $N = 4096$, $d = 128$, FP16 writes and re-reads the $N \times N$ score matrices: ~8.6 GFLOP over ~138 MB → **AI ≈ 62**, well under 295 → memory-bound. FlashAttention keeps those tiles in SMEM, removing the $N^2$ HBM traffic and pushing the kernel toward the compute roof.

**Decode attention never escapes.** Batching raises the intensity of the *weight* matmuls, but each sequence's KV cache is read by only that sequence's query: for MHA, AI ≈ 1 no matter the batch size. GQA with group size $g$ gives AI ≈ $g$ (Llama-3-8B: $g = 4$). This is why long-context decode stays memory-bound even at big batches — and why Modules 08 and 16 attack KV bytes directly.

> [!WARNING] Lower precision raises the ridge
> FP8 doubles peak FLOPS, so H100's ridge moves from ~295 to ~590 (Baseten). If you quantize only the FLOPs (not the bytes), memory-bound work becomes *more* memory-bound. For decode, the win from FP8/INT4 comes from **fewer bytes**, not faster math.

**Practice.**
1. A100 80GB: 312 TFLOPS, 2.04 TB/s. Ridge? *(≈153)*
2. T4: 65 TFLOPS FP16, 320 GB/s. Ridge? At what batch does a 4096×4096 FP16 layer become compute-bound? *(≈203; batch ≈ 200+)*
3. Your Mac: fill in from the lab in Lesson 8. Why is its ridge so much lower than an H100's? *(far fewer FLOPS per byte of bandwidth — Apple GPUs are relatively bandwidth-rich)*

- [ ] Can derive AI ≈ 1 for batch-1 decode and AI ≈ B for batch B
- [ ] Computed ridge points for three GPUs
- [ ] Can explain why decode attention stays memory-bound as batch grows
      */}),
      resources: [
        { title: "Williams, Waterman, Patterson — Roofline (2009)", url: "https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf", type: "paper", note: "the original roofline paper" },
        { title: "JAX Scaling Book — All About Rooflines", url: "https://jax-ml.github.io/scaling-book/roofline/", type: "book", note: "the clearest modern treatment, with exercises" },
      ],
    },
    {
      id: "batching-intensity",
      title: "Batching changes intensity: the latency–throughput trade-off",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
If decode at batch 1 uses 0.3% of the H100's compute, the obvious fix is to decode **many sequences in one forward pass**. Weights are read once per step no matter how many sequences share it; what grows with batch is FLOPs (cheap, we have spare) and KV-cache reads (not cheap).

::viz latency-throughput

Llama-3-8B BF16 on one H100, each sequence at 2,048 tokens of context, ideal bandwidth (100% MBU):

| Batch | Bytes read per step | TPOT | Per-user tok/s | Total tok/s |
|---|---|---|---|---|
| 1 | 16.3 GB | 4.9 ms | 205 | 205 |
| 8 | 18.2 GB | 5.4 ms | 184 | 1,472 |
| 32 | 24.6 GB | 7.4 ms | 136 | 4,349 |
| 64 | 33.2 GB | 9.9 ms | 101 | 6,450 |
| 128 | 50.4 GB | 15.1 ms | 66 | 8,505 |
| 192 | 67.6 GB | 20.2 ms | 50 | 9,515 |
| 256 | — | doesn't fit in 80 GB | — | — |

(Bytes = 16.06 GB of weights + B × 2,048 × 128 KiB of KV. Compute at B = 128 is only $128 \times 16.1\ \text{GFLOP} / 989\ \text{TF} \approx 2.1$ ms — still far below the 15 ms memory time.)

Read the table as a product manager would:
- **Going from 1 to 8 users costs each user 10% speed and multiplies total throughput by 7×.** That's nearly free money — the most important optimization in serving, and why continuous batching (Module 09) matters so much.
- Beyond ~64, per-user speed falls fast while total throughput saturates. Somewhere there is an **SLO line** (e.g., "each user must see ≥ 50 tok/s") that sets your max batch.
- The **KV cache** is what bends the curve: at B = 128 the KV bytes (34 GB) exceed the weights (16 GB). Decode never reaches the compute roof here — memory capacity runs out first. That's why KV-cache size, paging and compression (Module 08) and quantized KV (Module 16) are throughput features, not just memory features.

> [!INTUITION] The bus
> Batch 1 is a taxi: fast for one person, expensive per seat. A big batch is a bus: each rider is a bit slower (more stops = more KV to read), but cost per rider collapses. Inference engineering is choosing the bus size that still meets everyone's arrival-time promise.

> [!REAL] Where the knee is in practice
> Real engines hit this trade-off via `max_num_seqs` (vLLM) / `--max-running-requests` (SGLang) and the scheduler's token budget. Baseten (ch. 7) frames it the same way: raise batch size for cost efficiency until latency SLOs break. For short-context chat on an H100 with an 8B model, the throughput knee is typically in the 64–256 range; for 32K-context requests it can be under 16, because KV bytes dominate.

- [ ] Recomputed the B = 64 row from the formula yourself
- [ ] Can explain why total throughput saturates before compute does
- [ ] Can pick a max batch given an SLO of "≥ 100 tok/s per user"
      */}),
      resources: [
        { title: "Databricks — LLM Inference Performance Engineering: Best Practices", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "batch size vs latency/throughput curves, MBU" },
      ],
    },
    {
      id: "gpu-specs",
      title: "GPU generations & spec sheets: which chip for which job",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Engineers work with the 3–5 most recent generations (Baseten ch. 3.2). Names are an architecture letter + model number: **T**uring, **A**mpere, **L** = Ada **L**ovelace, **H**opper, **B**lackwell; Rubin is next (2026).

| GPU | Arch | Memory | Bandwidth | Dense BF16/FP16 TFLOPS | Ridge (ops:byte) | Notes |
|---|---|---|---|---|---|---|
| **T4** | Turing (2018) | 16 GB GDDR6 | 320 GB/s | 65 (FP16; no BF16) | ~203 | free on Colab; 70 W; fine for ≤ 3B models |
| **L4** | Ada (2023) | 24 GB GDDR6 | 300 GB/s | ~121 (242 FP8) | ~403 | cheap; embeddings, small models; no NVLink |
| **A10G** | Ampere | 24 GB GDDR6 | 600 GB/s | ~70–125 (figures vary) | ~120–210 | AWS g5; common for 7B models |
| **A100 80GB SXM** | Ampere (2020) | 80 GB HBM2e | 2.0 TB/s | 312 | ~153 | legacy workhorse; no FP8 |
| **H100 SXM** | Hopper (2022) | 80 GB HBM3 | 3.35 TB/s | 989 (1,979 FP8) | ~295 | today's standard; FP8; NVLink 900 GB/s |
| **H100 PCIe** | Hopper | 80 GB HBM2e | 2.0 TB/s | ~756 | ~378 | lower power, less bandwidth |
| **H200** | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | ~206 | same compute, 43% more bandwidth → faster decode |
| **B200** | Blackwell (2024) | 180–192 GB HBM3e | 8 TB/s | ~2,250 (~4,500 FP8; FP4 too) | ~281 | new gold standard; NVLink 1.8 TB/s |
| **GB200** | Grace + 2× B200 | per GPU as B200 | — | — | — | NVL72: 72 GPUs in one NVLink domain |
| **AMD MI300X** | CDNA3 | 192 GB HBM3 | 5.3 TB/s | ~1,307 | ~247 | most memory per GPU in its generation; ROCm |
| **Apple M-series** | unified | 16–512 GB | 100–819 GB/s | ~4–30 (GPU) | ~30–40 | local dev; capacity-rich, bandwidth-modest |
| **Google TPU v5e / v6e** | TPU | 16 / 32 GB HBM | 819 / 1,640 GB/s | 197 / 918 | ~240 / ~560 | JAX / vLLM-TPU; pods with ICI links |

Numbers are vendor dense-tensor figures (Apple and A10G approximate). Always check the datasheet for your exact SKU.

> [!WARNING] Reading spec sheets without getting fooled
> 1. **Sparse vs dense.** Headline numbers often assume 2:4 structured sparsity (~2× dense). Inference is dense — halve them (Baseten ch. 3.1).
> 2. **Precision.** FLOPS double with each halving of precision. Compare BF16 to BF16, FP8 to FP8.
> 3. **Form factor.** SXM vs PCIe versions of the "same" GPU differ in power, bandwidth and NVLink.
> 4. **GB vs Gb.** Network links are quoted in bits; divide by 8.
> 5. **Memory size vs bandwidth.** H200 vs H100 is *the same compute* — the upgrade is purely memory, which is exactly what decode wants.

**Which GPU for which job (first-order rules).**
- **Decode-heavy chat at low/medium batch** → maximize **bandwidth** per dollar (H200, B200, MI300X).
- **Prefill-heavy** (RAG with long prompts, embeddings, image/video generation) → maximize **FLOPS**.
- **Model doesn't fit** → you need capacity (H200/B200/MI300X) or multiple GPUs with NVLink (Module 13), or quantization (Module 16).
- **Small models (≤ 8B)** → an L4, A10G, or a **MIG slice** of an H100 can be more cost-efficient than a full GPU (Baseten ch. 3.3).
- Price matters as much as speed: compute tokens/sec **per dollar**. A rough H100 on-demand price is \$2–3/GPU-hour at neoclouds (prices move fast; check).

> [!NOTE] Beyond NVIDIA
> AMD MI300X/MI325X/MI355X (ROCm; supported by vLLM and SGLang), Google TPUs (JAX; vLLM has a TPU backend), AWS Trainium/Inferentia, and specialized chips: Groq (SRAM-based), Cerebras (wafer-scale), Etched (transformer ASIC). Each bets on one edge — usually memory bandwidth — and fights the same battle: software maturity (Baseten ch. 3.4).

- [ ] Can quote bandwidth and dense BF16 FLOPS for T4, A100, H100, H200, B200 from memory (within 20%)
- [ ] Can explain why H200 beats H100 for decode but not for prefill
- [ ] Can spot a sparse-FLOPS number on a spec sheet
      */}),
      resources: [
        { title: "NVIDIA H100", url: "https://www.nvidia.com/en-us/data-center/h100/", type: "docs", note: "datasheet — note the sparsity footnotes" },
        { title: "NVIDIA H200", url: "https://www.nvidia.com/en-us/data-center/h200/", type: "docs", note: "same compute, more bandwidth" },
        { title: "NVIDIA T4", url: "https://www.nvidia.com/en-us/data-center/tesla-t4/", type: "docs", note: "the Colab GPU" },
        { title: "AMD Instinct MI300X", url: "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html", type: "docs", note: "192 GB HBM3, 5.3 TB/s" },
        { title: "Google Cloud TPU v6e", url: "https://cloud.google.com/tpu/docs/v6e", type: "docs", note: "TPU specs for comparison" },
      ],
    },
    {
      id: "mfu-mbu",
      title: "Math: MFU and MBU — how close are you to the roof?",
      kind: "math",
      minutes: 60,
      md: MD(function () {/*
A roofline tells you the ceiling. **Utilization** tells you how close you got — and therefore how much is left to win.

> [!MATH] The two utilizations
> **MFU (Model FLOPs Utilization)** — for compute-bound work (prefill, big batches):
> $$\text{MFU} = \frac{\text{useful model FLOPs per second}}{\text{peak FLOPS}} = \frac{2N \cdot \text{tokens/s}}{\text{peak}}$$
> **MBU (Model Bandwidth Utilization)** — for memory-bound work (decode):
> $$\text{MBU} = \frac{\text{bytes that must be read per step} / \text{TPOT}}{\text{peak bandwidth}} = \frac{(\text{weight bytes} + \text{KV bytes}) / \text{TPOT}}{\text{peak BW}}$$
> "Useful" matters: count the model's FLOPs/bytes, not what your inefficient kernel actually did. Recomputation (e.g., no KV cache) doesn't count.

**Worked example 1 — decode MBU.** Llama-3-8B BF16 on H100, batch 1, short context, you measure **150 tok/s** (TPOT 6.67 ms).

$$\text{MBU} = \frac{16.06\ \text{GB} \times 150}{3350\ \text{GB/s}} = \frac{2409}{3350} \approx 72\%$$

Good, not great. Well-tuned kernels reach 80–90%. At 72% there's maybe 15% left from better kernels; beyond that you need **fewer bytes** (quantization, Module 16) or **more tokens per weight read** (batching, speculative decoding — Module 15).

**Worked example 2 — prefill MFU.** Same model, a 1,024-token prompt, TTFT 30 ms.

$$\text{MFU} = \frac{2 \times 8.03 \times 10^{9} \times 1024 / 0.030}{989 \times 10^{12}} = \frac{548\ \text{TFLOPS}}{989\ \text{TFLOPS}} \approx 55\%$$

Typical for prefill with good kernels: 40–70%. Short prompts get lower MFU (not enough work to fill 132 SMs); very long prompts are dragged down by attention's $O(P^2)$ cost unless you count those FLOPs too.

**Worked example 3 — batched decode.** Batch 64, 2,048-token contexts, TPOT 12 ms. Bytes per step = 16.06 + 64 × 2048 × 131,072 B ≈ 33.2 GB. MBU = 33.2 ÷ 0.012 ÷ 3350 ≈ 83%. MFU = 64 × 16.1 GFLOP ÷ 0.012 ÷ 989 TF ≈ 9%. Both numbers are "right": the step is memory-bound, so MBU is the one to optimize.

> [!IMPORTANT] Which one to report
> - Memory-bound regime (decode, small batch) → **MBU**. Low MFU is expected and harmless.
> - Compute-bound regime (prefill, very large batch, training) → **MFU**.
> - If **both** are low → you are overhead-bound (Lesson 3), or your kernels are bad, or the GPU is idle waiting on the scheduler/network. Profile.

> [!REAL] Industry numbers
> Databricks' inference guide popularized MBU for LLM serving and shows MBU falling as batch size grows on real hardware; training teams report MFU (PaLM/Llama papers quote ~40–60%). If a vendor claims 95% MBU, ask about context length and batch size.

**Practice.** Your Mac: from Lesson 1's measurement, compute MBU for both Qwen models. Which is closer to the roof, and why might the smaller model be *further* from it? *(fixed per-token overhead is a larger fraction of a shorter step)*

- [ ] Computed MBU for your Mac measurements
- [ ] Computed MFU for a prefill measurement (use `Prompt: … tokens-per-sec` from mlx-lm)
- [ ] Can explain why reporting MFU for batch-1 decode is misleading
      */}),
      resources: [
        { title: "Databricks — LLM Inference Performance Engineering", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "where MBU was popularized for serving" },
        { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "the FLOPs/bytes accounting behind MFU/MBU" },
      ],
    },
    {
      id: "lab-roofline",
      title: "Lab: measure bandwidth & TFLOPS, plot your own roofline (MPS + T4)",
      kind: "lab",
      minutes: 120,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Spec sheets are promises. Measure what your hardware actually delivers, then put real operations on the chart.

**Step 1 — the benchmark.** Three experiments in one script: a big `copy_` (pure bandwidth: read N bytes, write N bytes), square matmuls of growing size (peak TFLOPS), and a **batch sweep** of `x @ W` with `W` 8192×8192 — batch 1 is exactly a decode matvec; batch 4096 looks like prefill.

```python
# roofline_lab.py — measure bandwidth, matmul TFLOPS and a batch sweep; plot your roofline.
import argparse, json, time
import torch

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()
    elif dev == "mps": torch.mps.synchronize()

def bench(fn, dev, iters=20, warmup=5):
    for _ in range(warmup): fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters

def copy_bandwidth(dev, mib=1024, dtype=torch.float16):
    n = mib * 2**20 // (torch.finfo(dtype).bits // 8)  # elements in a `mib`-MiB buffer
    a = torch.ones(n, dtype=dtype, device=dev)
    b = torch.empty_like(a)
    s = bench(lambda: b.copy_(a), dev)
    moved = 2 * a.numel() * a.element_size()          # read a + write b
    return moved / s / 1e9                              # GB/s

def matmul_tflops(dev, n, dtype):
    a = torch.randn(n, n, dtype=dtype, device=dev)
    b = torch.randn(n, n, dtype=dtype, device=dev)
    s = bench(lambda: a @ b, dev, iters=10)
    return 2 * n**3 / s / 1e12

def batch_sweep(dev, d, dtype, batches):
    """y = x @ W with W [d, d]: B=1 is decode (matvec); big B looks like prefill."""
    W = torch.randn(d, d, dtype=dtype, device=dev)
    es = W.element_size()
    rows = []
    for B in batches:
        x = torch.randn(B, d, dtype=dtype, device=dev)
        s = bench(lambda: x @ W, dev)
        flops = 2 * B * d * d
        bytes_ = (d * d + 2 * B * d) * es               # read W and x, write y
        rows.append(dict(B=B, ai=flops / bytes_, tflops=flops / s / 1e12,
                         gbs=bytes_ / s / 1e9, us=s * 1e6))
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny sizes (CPU smoke test)")
    args = p.parse_args()
    dev = pick_device()
    dtype = torch.float32 if dev == "cpu" else torch.float16
    mib, d = (64, 1024) if args.quick else (1024, 8192)
    sizes = [256, 512, 1024] if args.quick else [512, 1024, 2048, 4096, 8192]
    batches = [1, 4, 16, 64, 256] if args.quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    bw = copy_bandwidth(dev, mib, dtype)
    mm = {n: matmul_tflops(dev, n, dtype) for n in sizes}
    peak = max(mm.values())
    sweep = batch_sweep(dev, d, dtype, batches)

    print(f"device={dev} dtype={dtype}")
    print(f"copy bandwidth: {bw:8.1f} GB/s")
    for n, t in mm.items(): print(f"matmul {n:5d}^3: {t:7.2f} TFLOPS")
    print(f"measured ridge point: {peak * 1e12 / (bw * 1e9):.0f} FLOPs/byte")
    print(f"{'B':>5} {'AI':>7} {'TFLOPS':>8} {'GB/s':>8} {'us':>9}")
    for r in sweep:
        print(f"{r['B']:5d} {r['ai']:7.1f} {r['tflops']:8.2f} {r['gbs']:8.1f} {r['us']:9.1f}")
    json.dump(dict(device=dev, bw_gbs=bw, peak_tflops=peak, sweep=sweep),
              open(f"roofline_{dev}.json", "w"), indent=1)
```

**Step 2 — the plot.**

```python
# plot_roofline.py — draw ceilings from measured numbers and overlay your sweep.
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1]))                     # e.g. roofline_mps.json
bw, peak = data["bw_gbs"] * 1e9, data["peak_tflops"] * 1e12
ridge = peak / bw

ai = np.logspace(-1, 4, 200)
plt.loglog(ai, np.minimum(peak, bw * ai) / 1e12, "k-", label=f"measured roof (ridge ≈ {ridge:.0f})")
xs = [r["ai"] for r in data["sweep"]]
ys = [r["tflops"] for r in data["sweep"]]
plt.loglog(xs, ys, "o-", label="x @ W, batch sweep")
for r in data["sweep"]:
    plt.annotate(f"B={r['B']}", (r["ai"], r["tflops"]), fontsize=7)
plt.axvline(ridge, ls=":", c="gray")
plt.xlabel("arithmetic intensity (FLOPs / byte)")
plt.ylabel("achieved TFLOPS")
plt.title(f"Roofline — {data['device']}")
plt.legend(); plt.grid(True, which="both", alpha=.3)
plt.savefig(f"roofline_{data['device']}.png", dpi=150)
print("wrote", f"roofline_{data['device']}.png")
```

Run `python roofline_lab.py && python plot_roofline.py roofline_mps.json` on your Mac, and the same on a Colab T4 (`roofline_cuda.json`).

**What to expect and explain.**
- **Bandwidth**: copy typically reaches 70–90% of spec (a T4 often shows ~220–260 GB/s of its 320; M-series usually gets closer to spec).
- **TFLOPS**: small matmuls are far below peak (not enough work, overhead); large ones plateau. A T4 in FP16 often reaches 25–45 of its "65" — it throttles at 70 W, and 65 is a peak tensor-core figure.
- **Batch sweep**: the points should climb the 45° ramp (time nearly flat as B grows — the weight read dominates), then bend over near the ridge and flatten under the roof. **The B at which the curve bends is your hardware's "free batching" limit.** Compare it with the measured ridge.

> [!TIP] Make the chart tell the inference story
> Add three labeled points: a batch-1 decode step of Qwen2.5-1.5B (AI ≈ 1, achieved FLOPS = 2N × measured tok/s from Lesson 1), a prefill of 512 tokens (AI ≈ a few hundred), and the Baseten attention example (AI = 62). You now have the one chart that explains most inference optimizations.

> [!NOTE] Numerical footnote
> The copy benchmark counts read + write. Some tools (e.g., STREAM "copy") report the same convention; others report only one direction. When comparing to someone else's number, check which.

- [ ] Measured bandwidth and peak TFLOPS on MPS and on a T4; listed % of spec
- [ ] Plotted both rooflines with the batch sweep overlaid
- [ ] Marked where decode, prefill and unfused attention sit on your chart
- [ ] Wrote two sentences explaining the bend in the batch sweep
      */}),
      resources: [
        { title: "PyTorch benchmark utilities", url: "https://docs.pytorch.org/docs/stable/benchmark_utils.html", type: "docs", note: "`torch.utils.benchmark.Timer` handles warm-up and sync" },
        { title: "JAX Scaling Book — How to think about GPUs", url: "https://jax-ml.github.io/scaling-book/gpus/", type: "book", note: "GPU rooflines and their numbers" },
      ],
    },
    {
      id: "deep-matmul",
      title: "Deep dive: how a matmul kernel reaches the roof",
      kind: "deep",
      optional: true,
      minutes: 120,
      md: MD(function () {/*
You don't need to write CUDA to be an inference engineer, but you should know **why** a naïve kernel is 10–50× slower than cuBLAS — because the same ideas (tiling, reuse, keeping data in SMEM) explain FlashAttention, PagedAttention kernels and fused MoE.

Read, in order:
1. **Simon Boehm — How to Optimize a CUDA Matmul Kernel.** Ten steps from a naïve kernel (~1% of cuBLAS) to ~94%: coalescing, shared-memory tiling, 1D/2D blocktiling, vectorized loads, warptiling. Each step is a move *up* the roofline by raising arithmetic intensity at some memory level.
2. **Aleksa Gordić — Inside NVIDIA GPUs.** The Hopper-era story: tensor cores, TMA (async bulk copies into SMEM), warp specialization, and why modern kernels are *pipelines* of producer/consumer warps.
3. **Horace He — Go Brrrr** (if you skipped it in Lesson 3): the compute/memory/overhead triad applied to real PyTorch code.

> [!INTUITION] The one idea
> A GEMM tile of size $T \times T$ loaded into SMEM costs $2T^2$ element loads and enables $2T^3$ FLOPs: intensity ∝ $T$. Bigger tiles → higher intensity → closer to the compute roof, until you run out of SMEM/registers. Every fast kernel is a negotiation between tile size and on-chip memory.

**Try it (Colab):** write a Triton matmul from the official tutorial, run it at sizes 512–8192, and plot achieved TFLOPS on your T4 roofline next to `torch.matmul`.

- [ ] Can explain shared-memory tiling and why it raises intensity
- [ ] Can name what TMA and warp specialization do on Hopper
      */}),
      resources: [
        { title: "Simon Boehm — How to Optimize a CUDA Matmul Kernel", url: "https://siboehm.com/articles/22/CUDA-MMM", type: "article", note: "the classic step-by-step" },
        { title: "Aleksa Gordić — Inside NVIDIA GPUs", url: "https://www.aleksagordic.com/blog/matmul", type: "article", note: "Hopper tensor cores, TMA, warp specialization" },
      ],
    },
  ],

  challenge: {
    title: "The napkin calculator: predict TTFT, TPOT and throughput within 30%",
    md: MD(function () {/*
Build `course-work/m07/napkin.py`: given a model config, a GPU, batch size, prompt length and output length, predict **TTFT, TPOT, per-user and total tok/s**, whether it fits in memory, and whether decode is memory- or compute-bound. Then measure and compare. Here's a starting point to extend — every line is a formula from this module:

```python
# napkin.py — predict TTFT, TPOT and throughput from first principles.
import argparse

GPUS = {  # dense bf16/fp16 TFLOPS, memory bandwidth GB/s, memory GB  (spec sheets; Apple = approximate)
    "t4":     dict(tflops=65,   gbs=320,  mem=16),
    "l4":     dict(tflops=121,  gbs=300,  mem=24),
    "a10g":   dict(tflops=70,   gbs=600,  mem=24),
    "a100":   dict(tflops=312,  gbs=2039, mem=80),
    "h100":   dict(tflops=989,  gbs=3350, mem=80),
    "h200":   dict(tflops=989,  gbs=4800, mem=141),
    "b200":   dict(tflops=2250, gbs=8000, mem=180),
    "mi300x": dict(tflops=1307, gbs=5300, mem=192),
    "m2pro":  dict(tflops=6.8,  gbs=200,  mem=32),
    "m3max":  dict(tflops=14,   gbs=400,  mem=64),
}
MODELS = {  # params, layers, attention heads, kv heads, head dim
    "qwen2.5-0.5b": dict(params=0.494e9, layers=24, heads=14, kv_heads=2, head_dim=64),
    "qwen2.5-1.5b": dict(params=1.54e9,  layers=28, heads=12, kv_heads=2, head_dim=128),
    "qwen2.5-7b":   dict(params=7.62e9,  layers=28, heads=28, kv_heads=4, head_dim=128),
    "llama3-8b":    dict(params=8.03e9,  layers=32, heads=32, kv_heads=8, head_dim=128),
    "llama3-70b":   dict(params=70.6e9,  layers=80, heads=64, kv_heads=8, head_dim=128),
}

def predict(m, g, batch, prompt, output, w_bytes=2, kv_bytes=2, mfu=0.6, mbu=0.75, overhead_ms=0.0):
    flops, bw = g["tflops"] * 1e12 * mfu, g["gbs"] * 1e9 * mbu
    weights = m["params"] * w_bytes
    kv_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes
    d_attn = m["heads"] * m["head_dim"]

    need = weights + batch * (prompt + output) * kv_tok
    fits = need <= g["mem"] * 1e9 * 0.9

    # Prefill all `batch` prompts together: linear layers + causal attention (QK^T and PV).
    pf_flops = batch * (2 * m["params"] * prompt + 2 * m["layers"] * prompt**2 * d_attn)
    pf_bytes = weights + batch * prompt * kv_tok
    ttft = max(pf_flops / flops, pf_bytes / bw) + overhead_ms / 1e3

    # One decode step at the *average* context length.
    ctx = prompt + output / 2
    dc_flops = batch * (2 * m["params"] + 4 * m["layers"] * ctx * d_attn)
    dc_bytes = weights + batch * ctx * kv_tok
    t_mem, t_cmp = dc_bytes / bw, dc_flops / flops
    tpot = max(t_mem, t_cmp) + overhead_ms / 1e3

    e2e = ttft + (output - 1) * tpot
    return dict(fits=fits, need_gb=need / 1e9, ttft_ms=ttft * 1e3, tpot_ms=tpot * 1e3,
                bound="memory" if t_mem >= t_cmp else "compute",
                per_user_tps=1 / tpot, total_tps=batch * output / e2e, e2e_s=e2e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3-8b", choices=MODELS)
    p.add_argument("--gpu", default="h100", choices=GPUS)
    p.add_argument("--batch", type=int, nargs="+", default=[1])
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--output", type=int, default=256)
    p.add_argument("--w-bytes", type=float, default=2)
    p.add_argument("--kv-bytes", type=float, default=2)
    p.add_argument("--mfu", type=float, default=0.6)
    p.add_argument("--mbu", type=float, default=0.75)
    p.add_argument("--overhead-ms", type=float, default=0.0)
    a = p.parse_args()
    print(f"{a.model} on {a.gpu}, prompt={a.prompt}, output={a.output}, MFU={a.mfu}, MBU={a.mbu}")
    print(f"{'batch':>5} {'mem GB':>7} {'TTFT ms':>8} {'TPOT ms':>8} {'bound':>7} {'tok/s/user':>10} {'tok/s total':>11}")
    for B in a.batch:
        r = predict(MODELS[a.model], GPUS[a.gpu], B, a.prompt, a.output,
                    a.w_bytes, a.kv_bytes, a.mfu, a.mbu, a.overhead_ms)
        flag = "" if r["fits"] else "  <-- does not fit"
        print(f"{B:5d} {r['need_gb']:7.1f} {r['ttft_ms']:8.1f} {r['tpot_ms']:8.2f} {r['bound']:>7}"
              f" {r['per_user_tps']:10.0f} {r['total_tps']:11.0f}{flag}")
```

`python napkin.py --batch 1 8 32 64 128 256` for Llama-3-8B on an H100 predicts ~155 tok/s at batch 1 (at 75% MBU) and total throughput flattening past batch ~128 — compare with the ideal table in Lesson 5.

**Your job:**
1. Replace the MFU/MBU defaults with **your measured values** from the lab (per device).
2. Add a `--measure` mode (or a separate script) that runs the same config through a real engine — `mlx_lm` on the Mac; HF `generate` or vLLM on a T4 — and prints predicted vs measured side by side.
3. Run ≥ 6 configurations across 2 devices (vary model size, batch, prompt length, output length, and one 4-bit/8-bit run using `--w-bytes 0.5` / `1`).
4. Get TTFT and TPOT within **30%** for at least 5 of them. Where you miss, explain *why* (overhead-bound? attention cost at long context? dequantization cost? prefill chunking?) and, if you can, add a term that fixes it (e.g., a fitted `--overhead-ms`).
    */}),
    checklist: [
      "`napkin.py` predicts TTFT, TPOT, per-user and total tok/s, memory fit and bound type",
      "Uses measured bandwidth/TFLOPS/efficiency for your Mac and T4",
      "Prediction vs measurement table for ≥ 6 configs on 2 devices",
      "≥ 5 configs within 30% on both TTFT and TPOT",
      "Each miss has a written explanation (overhead, attention, quantization, etc.)",
      "One chart: predicted vs measured TPOT with a y = x line",
    ],
    stretch: "Add tensor parallelism (weights and KV split across G GPUs, plus an all-reduce term: 2 × (G−1)/G × hidden × batch × 2 bytes per layer over NVLink bandwidth) and predict Llama-3-70B on 4× H100. You'll reuse this in Module 13 and check it against vLLM.",
  },

  connects: MD(function () {/*
Module 06 showed decode is slow; this module showed *why* (bytes, not FLOPs) and gave you a calculator. Everything next is a way to beat the memory roof: **Module 08** manages the KV cache so more sequences fit (bigger batch → higher intensity); **Module 09** batches continuously so the GPU always has a full bus; **Module 13** splits weights across GPUs so each reads fewer bytes; **Module 15** (speculative decoding) verifies several tokens per weight read; **Module 16** (quantization) makes every byte carry more parameters. When a later module claims "2× faster", your first question will be: *which roof did it move?*
  */}),

  interview: [
    "Estimate the maximum batch-1 decode speed of Llama-3-70B in FP8 on 4× H100 with tensor parallelism. What did you ignore?",
    "Define arithmetic intensity and ops:byte. What's the H100's ridge point, and what batch size makes a decode linear layer compute-bound?",
    "Why doesn't batching help attention's arithmetic intensity during decode, while it does help the MLP?",
    "Your H100 shows 8% utilization in nvidia-smi-style metrics during decode. Is that a problem? What metric would you look at instead?",
    "H100 vs H200: same FLOPS, more bandwidth and memory. Which workloads get faster, and by roughly how much?",
    "What's the difference between MFU and MBU? Give a measurement where MFU is 5% but the system is well optimized.",
    "Why can FP8 compute make a memory-bound kernel *more* memory-bound, and what actually speeds up decode when you quantize?",
    "A small model on a fast GPU gets far less than the bandwidth-predicted tok/s. List three causes and how to confirm each.",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 2.4 & 3", url: "Inference%20Engineering.pdf", type: "book", note: "ops:byte (H100 ≈ 295), attention AI = 62, GPU generations, MIG, interconnects" },
    { title: "Horace He — Making Deep Learning Go Brrrr", url: "https://horace.io/brrr_intro.html", type: "article", note: "compute / memory / overhead regimes" },
    { title: "JAX Scaling Book — All About Rooflines", url: "https://jax-ml.github.io/scaling-book/roofline/", type: "book", note: "rigorous roofline with worked problems" },
    { title: "JAX Scaling Book — How to think about GPUs", url: "https://jax-ml.github.io/scaling-book/gpus/", type: "book", note: "GPU-specific rooflines and networking" },
    { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "napkin math for latency and batch size" },
    { title: "Databricks — LLM Inference Performance Engineering", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "MBU, batch-size trade-offs, hardware comparison" },
    { title: "Modal GPU Glossary", url: "https://modal.com/gpu-glossary", type: "docs", note: "every GPU term, explained" },
    { title: "Williams et al. — Roofline (2009)", url: "https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf", type: "paper", note: "the original model" },
    { title: "Stas Bekman — ML Engineering", url: "https://github.com/stas00/ml-engineering", type: "book", note: "practical hardware, networking and benchmarking chapters" },
    { title: "Simon Boehm — CUDA matmul optimization", url: "https://siboehm.com/articles/22/CUDA-MMM", type: "article", note: "climbing the roofline one kernel trick at a time" },
  ],
});
