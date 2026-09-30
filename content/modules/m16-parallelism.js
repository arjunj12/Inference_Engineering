Course.module({
  id: "m16-parallelism",
  title: "Distributed inference: tensor, pipeline, data & expert parallelism",
  short: "Parallelism (TP / PP / DP / EP)",
  tagline: "Split one model across many GPUs without drowning in communication: implement tensor parallelism yourself, then serve a 70B model on a multi-GPU box and measure how it scales.",
  hours: 16,
  level: "core",
  runsOn: ["mac", "multi-gpu"],
  tags: ["tensor-parallelism", "pipeline-parallelism", "collectives", "nccl", "nvlink", "vllm"],

  goal: MD(function () {/*
Two things, one on your laptop and one on a rented multi-GPU box.

**1 — On your Mac:** you write Megatron-style tensor-parallel attention and MLP layers with `torch.distributed` (gloo backend, 2–4 CPU processes launched by `torchrun`) and prove they compute *exactly* what the single-process model computes:

```text
$ torchrun --standalone --nproc_per_node=4 tp_layers.py
TP=4 attention max err 1.4e-06  OK
TP=4 mlp       max err 1.1e-06  OK
```

**2 — On a 4–8× H100 node:** you serve Llama-3.1-8B at TP=1/2/4 and Llama-3.1-70B at TP=4 (and TP=8) with `vllm serve --tensor-parallel-size N`, benchmark them, and fill a table like this (your numbers will differ — that is the point of measuring):

```text
model            TP  GPUs  ITL p50 @c=1   out tok/s @c=64   tok/s per GPU
Llama-3.1-8B      1    1      ~6.5 ms          ~4,500            ~4,500
Llama-3.1-8B      2    2      ~4.5 ms          ~6,500            ~3,250
Llama-3.1-8B      4    4      ~3.5 ms          ~8,000            ~2,000
Llama-3.1-70B     4    4     ~14 ms            ~2,600              ~650
Llama-3.1-70B     8    8      ~9 ms            ~3,800              ~475
```

…and you can explain every trend in it: *why latency improves sub-linearly, why throughput per GPU drops, and why none of this works over PCIe the way it works over NVLink.*

::viz parallelism
  */}),
  demo: { viz: "parallelism", params: {} },

  why: MD(function () {/*
Every frontier-scale model you'll serve professionally — Llama-3.1-70B/405B, Qwen3-235B, DeepSeek-V3, Kimi K2 — is **too big for one GPU**, or too slow on one GPU for its latency SLO. So every inference team needs someone who can answer, with numbers: *"How many GPUs, in what layout (TP? PP? EP?), on what interconnect, and what latency/throughput will we get?"*

The Baseten book lists **parallelism** as one of its six core runtime techniques (Inference Engineering ch. 5.4). Interviews for inference roles routinely ask you to size a deployment on a whiteboard, explain ring all-reduce, or say why TP stops at the node boundary. After this module you can do all three — and you'll have written the code that vLLM, SGLang and TensorRT-LLM run under the hood.
  */}),

  prereqs: [
    {
      title: "Matrix shapes and splitting a matmul",
      skipIf: "you can say what shape X @ W has for X:(T, C) and W:(C, 4C) without thinking",
      math: true,
      md: MD(function () {/*
A matmul $Y = XW$ with $X$ of shape $(T, C)$ and $W$ of shape $(C, H)$ gives $Y$ of shape $(T, H)$. Each output entry is a **dot product** of one row of $X$ with one column of $W$. Two facts you'll use all module:

1. **Columns of $W$ are independent.** Column $j$ of $Y$ only needs column $j$ of $W$. So you can hand different columns to different GPUs.
2. **A dot product is a sum**, and sums can be split: $\sum_{i=1}^{C} x_i w_i = \sum_{i \le C/2} x_i w_i + \sum_{i > C/2} x_i w_i$. So you can also split along the inner dimension — as long as someone **adds the partial results** at the end.

::viz matmul {"m":2,"k":4,"n":4}

That "someone adds the partial results" is the **all-reduce** you'll meet in lesson 3.
      */}),
    },
    {
      title: "Processes, ranks and torchrun",
      skipIf: "you have used torch.distributed / DDP before",
      md: MD(function () {/*
Distributed PyTorch runs **one Python process per device**. Each process has a **rank** (0…N−1) and knows the **world size** N. `torchrun` launches the N processes and sets `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` in their environment; `dist.init_process_group(backend)` reads them and connects everyone.

| Backend | Runs on | Used for |
|---|---|---|
| `gloo` | CPU (works on your Mac) | learning, CPU tensors, control messages |
| `nccl` | NVIDIA GPUs | everything real — vLLM, SGLang, Megatron |
| `mpi` | CPU/GPU with an MPI install | HPC clusters |

```bash
uv pip install torch            # already in your ~/ai env from M00
torchrun --standalone --nproc_per_node=2 my_script.py
# macOS: if it hangs at init, pin gloo to loopback:
export GLOO_SOCKET_IFNAME=lo0
```

Every process runs the **same script** — the code branches on `rank`. That's "SPMD": single program, multiple data.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: why one GPU isn't enough",
      kind: "demo",
      minutes: 45,
      runsOn: ["browser", "any"],
      md: MD(function () {/*
## The 140 GB problem

Pick the **Llama-3-70B** preset below and look at the memory line.

::viz param-counter

Llama-3.1-70B has **70.6 B parameters**. In BF16 (2 bytes each) that's **≈ 141 GB of weights**. An H100 has **80 GB**. It doesn't fit — not "it's slow", it simply cannot load. And weights aren't the only tenant:

::viz kv-calculator

With GQA (8 KV heads × 128 dims × 80 layers), every token in every conversation costs

$$
2 \times 80 \times 8 \times 128 \times 2\ \text{bytes} = 327{,}680\ \text{bytes} \approx 320\ \text{KiB}.
$$

32 concurrent users × 8K tokens of context = 262K tokens ≈ **86 GB of KV cache**. So a useful 70B deployment wants roughly $141 + 86 + \text{overhead} \approx 240$ GB → **4× H100** (320 GB) at minimum.

| Model | BF16 weights | FP8 weights | Minimum sensible H100-80GB setup |
|---|---|---|---|
| Llama-3.1-8B | 16 GB | 8 GB | 1 GPU (replicate for throughput) |
| Llama-3.1-70B | 141 GB | 71 GB | 4 GPUs BF16, or 2 GPUs FP8 |
| Llama-3.1-405B | 810 GB | 405 GB | 8 GPUs FP8, or 16 GPUs (2 nodes) BF16 |
| DeepSeek-V3 (671B MoE) | — | ~690 GB (ships in FP8) | 8× H200 (141 GB each) or 16× H100 |

## It's not only about capacity — it's about bandwidth

Recall the M00 napkin law: batch-1 decode time ≈ **bytes read ÷ memory bandwidth**. Even if you squeezed 70B into one GPU with FP8 (71 GB):

$$
\frac{71\ \text{GB}}{3.35\ \text{TB/s}} \approx 21\ \text{ms/token} \Rightarrow \le 47\ \text{tok/s}.
$$

Split the weights over **4 GPUs** and each GPU reads only a quarter per token, **in parallel** — ideally ~5 ms/token. Parallelism buys you **aggregate memory bandwidth**, not just capacity. That's why teams run 70B at TP=8 even when TP=4 fits: lower per-user latency.

> [!INTUITION] The whole module in one sentence
> We split the model so each GPU stores and reads less — then pay for it with **communication** between GPUs. Every design choice in this module is about making that communication cheap, rare, or hidden.

> [!REAL] Inference Engineering (Baseten) ch. 5.4
> The book's rule of thumb: **FP8 ≈ 1 GB per billion params**, then `GPUs = ceil(weights × (1 + KV allowance) ÷ VRAM per GPU)`, rounded **up** to an instance size (1, 2, 4, 8). Its worked example: DeepSeek-V3.1 (671B) in FP8 on B200s → 4 GPUs hold the weights but leave no KV room → a full 8-GPU node in production.

- [ ] Used `param-counter` to confirm 70B BF16 ≈ 141 GB
- [ ] Used `kv-calculator` to find max concurrency for 70B on 4× H100 at 8K context
- [ ] Computed the batch-1 decode upper bound for 70B FP8 on 1 vs 4 H100s
      */}),
    },
    {
      id: "parallelism-zoo",
      title: "The parallelism zoo: DP, TP, PP, EP, CP, SP",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
There are only a few ways to cut a transformer. Click through each mode below and watch **what lives on each GPU** and **what crosses the wire**.

::viz parallelism

| Name | What is split | What each GPU holds | Communication | Typical inference use |
|---|---|---|---|---|
| **DP** — data parallel | the **requests** | a full copy of the model | none between replicas (a load balancer in front) | the default way to scale throughput: "replicas" |
| **TP** — tensor parallel | each **weight matrix** (inside every layer) | $\tfrac{1}{N}$ of every layer | **2 all-reduces per layer**, every token | lower latency and fit big models within one NVLink node |
| **PP** — pipeline parallel | the **layers** (stage 1 = layers 0–39, stage 2 = 40–79) | a contiguous block of layers | point-to-point activations between stages | span nodes over slow links; GPUs without NVLink |
| **EP** — expert parallel | the **experts** of an MoE layer | a subset of experts | **all-to-all**: send tokens to their experts and back | MoE models (DeepSeek, Qwen3-MoE, Kimi) |
| **CP** — context parallel | the **sequence** (tokens 0–64K on GPU 0, 64K–128K on GPU 1…) | full weights, part of the context | ring-pass KV blocks (ring attention) | million-token prefill |
| **SP** — sequence parallel (Megatron) | the **activations outside attention/MLP** along sequence | same as TP | swaps all-reduce for reduce-scatter + all-gather | mostly training memory; appears in some inference engines |

Real deployments **compose** them. A few you'll see written down:

- `TP8` — one 8-GPU node, every layer split 8 ways. The default for 70B–405B.
- `TP8 PP2` — two nodes; TP inside each node over NVLink, 2 pipeline stages across the slower InfiniBand link.
- `DP8 EP8` (a.k.a. "DP-attention + EP") — each GPU runs attention for its own requests, and experts are spread across all 8. vLLM: `--data-parallel-size 8 --enable-expert-parallel`.
- `EP32 / EP144` — DeepSeek's production prefill / decode layouts (M17).

> [!INTUITION] Inference ≠ training
> Training cares about gradients and optimizer states (ZeRO/FSDP shard those). **Inference has no gradients** — what matters is weights, the **KV cache**, and latency per token. So FSDP-style sharding (gather the weights before every layer) is rarely used for serving: re-gathering 140 GB of weights for every decode step would be absurd. TP, EP and replicas dominate.

## The one-line trade-off of each

- **DP:** zero communication, but every replica must fit the whole model. Latency unchanged.
- **TP:** best latency (every GPU works on every token), but communicates **every layer** → needs NVLink.
- **PP:** communicates rarely (once per stage boundary), but a single request still walks through the stages **sequentially** → no latency win, and "bubbles" of idle GPUs.
- **EP:** great throughput for MoE, communication proportional to tokens (not weights) → scales across nodes; load imbalance is the enemy.

> [!CHECK] Self-check
> For each mode, answer: *if I double the GPUs, does per-user latency go down? does max model size go up? does total throughput go up?* Write a 6×3 table before moving on.

- [ ] Explored all modes in the `parallelism` viz
- [ ] Filled the 6×3 "what does doubling GPUs buy" table
      */}),
      resources: [
        { title: "HF Ultrascale Playbook", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook", type: "course", note: "the best single visual reference for every parallelism strategy (training-focused, but the pictures transfer)" },
        { title: "JAX Scaling Book — Sharded matrices", url: "https://jax-ml.github.io/scaling-book/sharding/", type: "book", note: "rigorous notation for 'which axis is split where' and the communication each split costs" },
      ],
    },
    {
      id: "collectives",
      title: "Collectives: broadcast, all-reduce, all-gather, reduce-scatter, all-to-all",
      kind: "concept",
      minutes: 75,
      runsOn: ["mac"],
      md: MD(function () {/*
Distributed inference is built from a handful of **collective operations** — group operations every rank calls at the same time. Learn these five and you can read any distributed engine.

| Collective | Before (4 ranks) | After | Where you'll see it in inference |
|---|---|---|---|
| **broadcast** | rank 0 has `x` | everyone has `x` | vLLM's driver broadcasting the scheduled batch to TP workers |
| **all-reduce** (sum) | rank r has $x_r$ | everyone has $\sum_r x_r$ | **TP**: combining partial matmul results, twice per layer |
| **all-gather** | rank r has chunk $c_r$ | everyone has $[c_0, c_1, c_2, c_3]$ | vocab-parallel LM head (gather logits), SP |
| **reduce-scatter** | rank r has $[x_{r,0}, …, x_{r,3}]$ | rank r has $\sum_{r'} x_{r',r}$ | SP; first half of ring all-reduce |
| **all-to-all** | rank r has chunks addressed to each rank | rank r has every chunk addressed to it | **EP**: dispatch tokens to experts, combine back; Ulysses CP |
| send / recv (point-to-point) | — | — | **PP** activations; KV-cache transfer (M17) |

> [!INTUITION] all-reduce = reduce-scatter + all-gather
> This identity is how the fastest algorithm works: first everyone ends up owning the **sum of one slice** (reduce-scatter), then everyone shares their slice (all-gather).

## Run them on your Mac

```python
# collectives.py — run: torchrun --standalone --nproc_per_node=4 collectives.py
import torch, torch.distributed as dist

dist.init_process_group(backend="gloo")
rank, world = dist.get_rank(), dist.get_world_size()

def show(name, t):
    for r in range(world):            # print in rank order so output is readable
        dist.barrier()
        if r == rank:
            print(f"[{name:14s}] rank {rank}: {t.tolist()}", flush=True)

# 1) broadcast: rank 0's tensor is copied to everyone
t = torch.tensor([42.0, 7.0]) if rank == 0 else torch.zeros(2)
dist.broadcast(t, src=0);                          show("broadcast", t)

# 2) all_reduce: everyone ends with the element-wise SUM
t = torch.tensor([float(rank), 10.0 * rank])
dist.all_reduce(t, op=dist.ReduceOp.SUM);          show("all_reduce", t)

# 3) all_gather: everyone ends with everyone's tensor
t = torch.tensor([float(rank)])
out = [torch.zeros(1) for _ in range(world)]
dist.all_gather(out, t);                           show("all_gather", torch.cat(out))

# 4) reduce_scatter: sum, but rank r keeps only slice r
inp = [torch.full((1,), float(rank * 10 + i)) for i in range(world)]
out = torch.zeros(1)
dist.reduce_scatter(out, inp, op=dist.ReduceOp.SUM); show("reduce_scatter", out)

# 5) all_to_all: rank r sends chunk i to rank i (the MoE "dispatch" pattern)
inp = [torch.tensor([float(rank * 10 + i)]) for i in range(world)]
out = [torch.zeros(1) for _ in range(world)]
dist.all_to_all(out, inp);                         show("all_to_all", torch.cat(out))

dist.destroy_process_group()
```

Expected (4 ranks): all-reduce gives `[6.0, 60.0]` everywhere; all-to-all gives rank 1 `[1.0, 11.0, 21.0, 31.0]` — it's a **transpose** of who-holds-what.

> [!NOTE] Backend support
> Recent PyTorch (2.x) gloo supports all five on CPU. If an older build raises "not supported" for `reduce_scatter` or `all_to_all`, upgrade PyTorch — on GPUs NCCL supports everything.

## How all-reduce is actually done: the ring

Naively, everyone sends everything to rank 0, which sums and broadcasts — rank 0's link becomes the bottleneck. The **ring algorithm** arranges GPUs in a circle and splits the tensor into N chunks:

1. **Reduce-scatter phase** (N−1 steps): each step, every GPU sends one chunk to its right neighbour and adds the chunk it receives from the left. After N−1 steps, each GPU owns the **full sum of one chunk**.
2. **All-gather phase** (N−1 steps): pass the finished chunks around the ring until everyone has all of them.

::viz ring-allreduce

Every GPU sends and receives at every step — **all links busy all the time**. Each GPU sends $2(N-1)$ chunks of size $S/N$ in total, i.e. $\approx 2S$ bytes no matter how many GPUs. That's why ring all-reduce scales. Next lesson we turn this into a time formula.

- [ ] Ran `collectives.py` with 2 and 4 processes; predicted each output before reading it
- [ ] Wrote all-reduce yourself as `reduce_scatter` followed by `all_gather` and checked it equals `dist.all_reduce`
- [ ] Stepped through `ring-allreduce` for N=4 and counted the steps: $2(N-1) = 6$
      */}),
      resources: [
        { title: "PyTorch — Writing distributed applications", url: "https://pytorch.org/tutorials/intermediate/dist_tuto.html", type: "docs", note: "official tutorial: point-to-point, collectives, and a hand-written ring all-reduce" },
        { title: "torch.distributed docs", url: "https://docs.pytorch.org/docs/stable/distributed.html", type: "docs", note: "API reference + backend support table" },
      ],
    },
    {
      id: "comm-math",
      title: "Math: the α–β communication model and ring all-reduce cost",
      kind: "math",
      minutes: 75,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!PREREQ] Units refresher — the #1 source of wrong answers
> - **bits vs bytes:** network links are quoted in **bits per second**, memory in **bytes per second**. 1 byte = 8 bits. InfiniBand NDR "400 Gb/s" = **50 GB/s**. Always divide by 8.
> - **per direction vs bidirectional:** NVLink 4 on H100 is quoted as **900 GB/s** — that's both directions added. One direction ≈ **450 GB/s**. PCIe 5.0 x16 ≈ 64 GB/s per direction (≈ 50 in practice).
> - **latency units:** 1 µs (microsecond) = $10^{-6}$ s; 1 ms = 1000 µs. GPU kernels take µs; network round-trips take µs to ms.
> - **bandwidth vs latency:** bandwidth = how many bytes per second once data flows (the width of the pipe); latency = how long before the first byte arrives (the length of the pipe). Small messages feel latency; big messages feel bandwidth.

## The α–β model

The time to send a message of $S$ bytes over a link:

$$
T(S) = \alpha + \frac{S}{B}
$$

- $\alpha$ = fixed **startup latency** (software, kernel launch, link latency) — µs.
- $B$ = **bandwidth** (bytes/s). (Papers often write $\beta = 1/B$, "seconds per byte" — same thing.)

Tiny messages: $T \approx \alpha$ (latency-bound). Huge messages: $T \approx S/B$ (bandwidth-bound). The crossover is at $S^\ast = \alpha B$. For NVLink with $\alpha \approx 5\,\mu s$, $B \approx 450$ GB/s: $S^\ast \approx 2.3$ MB. **Anything smaller than a couple of MB is latency-bound on NVLink.** Remember that — decode all-reduces are much smaller.

::viz comm-cost

## From one message to ring all-reduce

The ring does $2(N-1)$ steps; in each step every GPU sends a chunk of $S/N$ bytes (all links in parallel). Applying the α–β model per step:

$$
T_{\text{ring}} = 2(N-1)\left(\alpha + \frac{S/N}{B}\right) = \underbrace{2(N-1)\,\alpha}_{\text{latency term}} + \underbrace{\frac{2(N-1)}{N}\cdot\frac{S}{B}}_{\text{bandwidth term}}
$$

As $N$ grows, $\frac{2(N-1)}{N} \to 2$: the bandwidth term stops growing (good), but the **latency term grows linearly with N** (bad for small messages). That's why NCCL switches to **tree** algorithms and special low-latency protocols for small tensors.

> [!NOTE] algbw vs busbw
> `nccl-tests` prints **algbw** $= S / T$ and **busbw** $= \text{algbw} \times \frac{2(N-1)}{N}$. busbw is comparable to the raw link speed; if busbw ≈ 80–90% of NVLink's per-direction number, your node is healthy.

## Worked example 1 — TP decode on 8× H100 (Llama-3.1-70B)

Decode, batch 32. Each all-reduce carries the hidden state for 32 tokens: $S = 32 \times 8192 \times 2\text{ B} = 524{,}288$ B ≈ 0.52 MB.

- Bandwidth term: $\frac{2 \cdot 7}{8} \cdot \frac{0.52 \times 10^6}{450 \times 10^9} \approx 2\,\mu s$.
- Latency term: realistically **5–20 µs** per all-reduce on NVLink depending on the implementation.
- Per token: 80 layers × 2 all-reduces = **160 all-reduces** → roughly **1–3 ms**.
- Compute/memory per token at TP8: $141\text{ GB}/8 \div 3.35\text{ TB/s} \approx 5.3$ ms.

So communication is a **20–40% tax** on decode — and it's almost all **latency**, not bandwidth. This is exactly why vLLM and TensorRT-LLM ship **custom all-reduce kernels** for small messages and capture them in CUDA graphs.

## Worked example 2 — TP prefill, NVLink vs PCIe

Prefill an 8K-token prompt: $S = 8192 \times 8192 \times 2 \approx 134$ MB per all-reduce, 160 of them.

| Link (per direction) | Per all-reduce ($\frac{2 \cdot 7}{8}\frac{S}{B}$) | × 160 layers·2 | Prefill compute (≈ 0.2 s at TP8) |
|---|---|---|---|
| NVLink 4, 450 GB/s | 0.52 ms | **83 ms** | comm ≈ 40% — must overlap |
| PCIe 4.0, ~25 GB/s | 9.4 ms | **1.5 s** | comm ≈ 7× compute — TP is pointless |

Compute estimate: $2 \times 70\text{B} \times 8192 \approx 1.15 \times 10^{15}$ FLOPs ÷ (8 × 989 TFLOPS × 0.6 MFU) ≈ 0.19 s.

> [!INTUITION] The rule this gives you
> **TP needs NVLink.** Over PCIe or the network, TP's per-layer all-reduces dominate. That's the math behind "TP within a node, PP/EP across nodes."

## Measure α and B on your Mac

```python
# allreduce_bench.py — run: torchrun --standalone --nproc_per_node=4 allreduce_bench.py
import time, torch, torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
for exp in range(2, 25, 2):                     # 4 floats ... 16M floats (16 B ... 64 MB)
    n = 2 ** exp
    t = torch.ones(n)
    for _ in range(3): dist.all_reduce(t)       # warm-up
    dist.barrier()
    iters = 20 if n < 2**20 else 5
    t0 = time.perf_counter()
    for _ in range(iters): dist.all_reduce(t)
    dist.barrier()
    dt = (time.perf_counter() - t0) / iters
    if rank == 0:
        busbw = 2 * (world - 1) / world * n * 4 / dt / 1e9    # same definition as nccl-tests
        print(f"{n*4:>12,d} B  {dt*1e6:>10.1f} us   busbw {busbw:6.2f} GB/s")
dist.destroy_process_group()
```

Plot time vs bytes on log-log axes. You'll see a **flat floor** (that's $\alpha$ — tens to hundreds of µs for gloo over loopback) and then a **straight line** (slope gives $B$ — a few GB/s through CPU memory). Fit $\alpha$ and $B$ with `numpy.polyfit` on the linear part. The same script with `backend="nccl"` and CUDA tensors on an H100 node shows α of a few µs and busbw in the hundreds of GB/s.

- [ ] Converted: 400 Gb/s IB, 900 GB/s bidirectional NVLink, 100 GbE → one-direction GB/s
- [ ] Reproduced worked example 1 for batch 1 and batch 256 — at which batch does bandwidth start to matter?
- [ ] Ran `allreduce_bench.py` on your Mac, fitted $\alpha$ and $B$, and found your crossover size $S^\ast = \alpha B$
      */}),
      resources: [
        { title: "NVIDIA nccl-tests", url: "https://github.com/NVIDIA/nccl-tests", type: "tool", note: "the standard all-reduce benchmark; its PERFORMANCE.md explains algbw vs busbw" },
        { title: "JAX Scaling Book", url: "https://jax-ml.github.io/scaling-book/", type: "book", note: "same α–β reasoning applied to TPUs/GPUs, with worked exercises" },
      ],
    },
    {
      id: "tensor-parallel",
      title: "Tensor parallelism: Megatron's column / row split",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Megatron-LM (Shoeybi et al., 2019) found the trick that made TP practical: arrange the splits so each transformer sub-block needs **exactly one all-reduce**.

## The math of splitting a matmul (with 2 GPUs)

**Column split** — cut $A$ into column blocks $A = [A_1 \mid A_2]$:

$$
XA = X[A_1 \mid A_2] = [XA_1 \mid XA_2]
$$

GPU $i$ computes $XA_i$. No communication: the output is simply **sharded** along its last dimension. Elementwise functions (GELU, SiLU) can be applied to each shard independently: $\text{GELU}([XA_1 \mid XA_2]) = [\text{GELU}(XA_1) \mid \text{GELU}(XA_2)]$.

**Row split** — cut $B$ into row blocks and the input into matching column blocks:

$$
YB = [Y_1 \mid Y_2]\begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = Y_1B_1 + Y_2B_2
$$

GPU $i$ computes the **partial sum** $Y_iB_i$ (full shape!). An **all-reduce (sum)** produces the answer on every GPU.

**Chain them:** column-split output is *already* the row-split input. So the MLP $Z = \text{GELU}(XA)\,B$ becomes

$$
Z = \underbrace{\text{GELU}(XA_1)B_1}_{\text{GPU 1}} + \underbrace{\text{GELU}(XA_2)B_2}_{\text{GPU 2}} \quad\Rightarrow\quad \text{one all-reduce}.
$$

> [!WARNING] Why not row-split the first matmul?
> Then each GPU holds a partial sum of $XA$, and $\text{GELU}(P_1 + P_2) \neq \text{GELU}(P_1) + \text{GELU}(P_2)$. You'd need an all-reduce **before** the nonlinearity *and* after the second matmul — twice the communication. Column-then-row is the whole trick.

::viz tp-split

### Tiny numeric check

$X = [1, 2]$, $A = \begin{bmatrix}1 & 0 & 2 & 1\\ 0 & 1 & 1 & 0\end{bmatrix}$ → $XA = [1, 2, 4, 1]$. Split columns: GPU 1 gets $[1, 2]$, GPU 2 gets $[4, 1]$. Let $B$ be 4×1 with entries $[1, 1, 1, 1]^\top$ and skip GELU: GPU 1 computes $1+2 = 3$, GPU 2 computes $4+1=5$, all-reduce → $8 = 1+2+4+1$. ✓

## Attention: heads are the natural split

Multi-head attention is *already* parallel: head $h$ only uses its own slice of $W_Q, W_K, W_V$ (columns $h \cdot d_h$ to $(h+1)\cdot d_h$). So:

- $W_Q, W_K, W_V$: **column-parallel** → GPU $i$ gets $n_{\text{heads}}/N$ heads and computes attention for them entirely locally (including its **share of the KV cache**).
- $W_O$: **row-parallel** → partial sums → **all-reduce**.

With GQA (Llama-3-70B: 64 query heads, 8 KV heads), TP=8 gives each GPU 8 query heads and **1 KV head**. TP=16 would need to **replicate** KV heads — one reason TP rarely goes beyond 8 for these models.

## What a full layer costs

| Piece | Split | Communication |
|---|---|---|
| QKV projection | column | none |
| attention (per head) | heads | none |
| output projection | row | **all-reduce #1** |
| MLP up/gate | column | none |
| MLP down | row | **all-reduce #2** |
| LayerNorm/RMSNorm, residual | replicated | none (inputs identical on all ranks) |
| embedding / LM head | vocab-parallel | all-reduce / all-gather of logits |

Per GPU, **weights and KV cache both shrink by N** — so TP adds capacity *and* bandwidth. The price: 2 all-reduces × layers × every forward pass.

> [!REAL] What engines do
> vLLM's `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` and `VocabParallelEmbedding` (in `vllm/model_executor/layers/`) are exactly this. PyTorch has a native version too: `torch.distributed.tensor.parallel` with `ColwiseParallel` / `RowwiseParallel` and `parallelize_module`. TensorRT-LLM fuses the all-reduce with the following RMSNorm to save a kernel.

- [ ] Worked the numeric example with 4 columns split over 2 GPUs, including GELU
- [ ] Explained in one sentence why column-then-row needs 1 all-reduce and row-then-column needs 2
- [ ] For Llama-3.1-70B at TP=4, wrote down per-GPU shapes of $W_Q$, $W_K$, $W_O$, $W_{\text{up}}$, $W_{\text{down}}$ (hidden 8192, FFN 28672, 64 Q heads, 8 KV heads, head dim 128)
      */}),
      resources: [
        { title: "Megatron-LM paper (Shoeybi et al., 2019)", url: "https://arxiv.org/abs/1909.08053", type: "paper", note: "section 3 is the column/row split — short and very readable" },
        { title: "PyTorch Tensor Parallel tutorial", url: "https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html", type: "docs", note: "the DTensor ColwiseParallel/RowwiseParallel API applied to Llama" },
      ],
    },
    {
      id: "build-tp",
      title: "Build: tensor-parallel MLP & attention with torch.distributed on your Mac",
      kind: "build",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] What you're building
> `tp_mlp.py` (30 lines) and then `tp_layers.py`: reusable `ColumnParallelLinear`, `RowParallelLinear`, `TPAttention`, `TPMLP` — verified against a single-process reference at TP = 1, 2, 4.

## Step 1 — the smallest possible TP: an MLP

```python
# tp_mlp.py — run: torchrun --standalone --nproc_per_node=2 tp_mlp.py
import torch, torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group(backend="gloo")
rank, world = dist.get_rank(), dist.get_world_size()

# Every rank builds the SAME full weights (same seed). A real engine loads only its
# shard from disk — we keep the full copy so we can check correctness.
torch.manual_seed(0)
B, T, C = 2, 16, 512
H = 4 * C                                   # MLP hidden size
x  = torch.randn(B, T, C)
W1 = torch.randn(C, H) / C**0.5             # up-projection   C -> H
W2 = torch.randn(H, C) / H**0.5             # down-projection H -> C

ref = F.gelu(x @ W1) @ W2                   # the single-device answer

assert H % world == 0, "hidden size must divide by TP degree"
s = H // world
W1_shard = W1[:, rank * s:(rank + 1) * s].contiguous()   # column-parallel: split OUTPUT features
W2_shard = W2[rank * s:(rank + 1) * s, :].contiguous()   # row-parallel:    split INPUT features

h = F.gelu(x @ W1_shard)                    # (B,T,H/N) — GELU is elementwise: no comm needed
y = h @ W2_shard                            # (B,T,C)   — a PARTIAL sum
dist.all_reduce(y, op=dist.ReduceOp.SUM)    # the ONE communication of the MLP

err = (y - ref).abs().max().item()
print(f"rank {rank}/{world}: W1 shard {tuple(W1_shard.shape)}, W2 shard {tuple(W2_shard.shape)}, "
      f"max |tp - ref| = {err:.2e}", flush=True)
assert torch.allclose(y, ref, atol=1e-5, rtol=1e-4)
dist.destroy_process_group()
```

```text
rank 0/2: W1 shard (512, 1024), W2 shard (1024, 512), max |tp - ref| = 2.38e-07
rank 1/2: W1 shard (512, 1024), W2 shard (1024, 512), max |tp - ref| = 2.38e-07
```

> [!NOTE] Why the error isn't exactly 0
> Floating-point addition isn't associative: summing 2 partial results of 1024 terms each rounds differently from summing 2048 terms in one go. ~1e-7 in FP32 is "identical". In BF16 on GPUs expect ~1e-2 — which is why TP=1 and TP=8 can produce **slightly different greedy outputs**. Real engines accept this.

## Step 2 — reusable layers + attention

```python
# tp_layers.py — run: torchrun --standalone --nproc_per_node=4 tp_layers.py
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """y_local = x @ W[:, my columns]. Output stays sharded (no comm)."""
    def __init__(self, full_weight):                      # full_weight: (in, out)
        super().__init__()
        r, n = dist.get_rank(), dist.get_world_size()
        out = full_weight.shape[1]; assert out % n == 0
        s = out // n
        self.weight = nn.Parameter(full_weight[:, r * s:(r + 1) * s].clone())
    def forward(self, x):
        return x @ self.weight

class RowParallelLinear(nn.Module):
    """Input arrives sharded on its last dim; partial results are summed with all-reduce."""
    def __init__(self, full_weight):                      # full_weight: (in, out)
        super().__init__()
        r, n = dist.get_rank(), dist.get_world_size()
        inp = full_weight.shape[0]; assert inp % n == 0
        s = inp // n
        self.weight = nn.Parameter(full_weight[r * s:(r + 1) * s, :].clone())
    def forward(self, x_shard):
        y = x_shard @ self.weight
        dist.all_reduce(y)                                # SUM is the default op
        return y

class TPAttention(nn.Module):
    """Heads are split across ranks: Q/K/V column-parallel, output projection row-parallel."""
    def __init__(self, Wq, Wk, Wv, Wo, n_heads):
        super().__init__()
        n = dist.get_world_size(); assert n_heads % n == 0
        self.h_local, self.hd = n_heads // n, Wq.shape[0] // n_heads
        self.q, self.k, self.v = ColumnParallelLinear(Wq), ColumnParallelLinear(Wk), ColumnParallelLinear(Wv)
        self.o = RowParallelLinear(Wo)
    def forward(self, x):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h_local, self.hd).transpose(1, 2)   # (B, h_local, T, hd)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # each rank: its own heads
        a = a.transpose(1, 2).reshape(B, T, self.h_local * self.hd)          # (B, T, C/N)
        return self.o(a)                                                     # all-reduce #1

class TPMLP(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()
        self.up, self.down = ColumnParallelLinear(W1), RowParallelLinear(W2)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))                                 # all-reduce #2

def reference_attention(x, Wq, Wk, Wv, Wo, n_heads):
    B, T, C = x.shape; hd = C // n_heads
    split = lambda t: t.view(B, T, n_heads, hd).transpose(1, 2)
    a = F.scaled_dot_product_attention(split(x @ Wq), split(x @ Wk), split(x @ Wv), is_causal=True)
    return a.transpose(1, 2).reshape(B, T, C) @ Wo

if __name__ == "__main__":
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    B, T, C, n_heads = 2, 32, 256, 8
    x = torch.randn(B, T, C)
    Wq, Wk, Wv, Wo = (torch.randn(C, C) / C**0.5 for _ in range(4))
    W1, W2 = torch.randn(C, 4 * C) / C**0.5, torch.randn(4 * C, C) / (4 * C)**0.5

    with torch.no_grad():
        attn, mlp = TPAttention(Wq, Wk, Wv, Wo, n_heads), TPMLP(W1, W2)
        y_attn, y_mlp = attn(x), mlp(x)
        ref_attn = reference_attention(x, Wq, Wk, Wv, Wo, n_heads)
        ref_mlp = F.gelu(x @ W1) @ W2
    for name, a, b in [("attention", y_attn, ref_attn), ("mlp", y_mlp, ref_mlp)]:
        ok = torch.allclose(a, b, atol=1e-5, rtol=1e-4)
        if rank == 0:
            print(f"TP={world} {name:9s} max err {(a - b).abs().max():.1e}  {'OK' if ok else 'MISMATCH'}")
    dist.destroy_process_group()
```

Note the column split of $W_Q$ lines up with heads *because* heads are laid out contiguously in the columns: columns `[r*h_local*hd, (r+1)*h_local*hd)` are exactly heads `r*h_local … (r+1)*h_local − 1`.

## Step 3 — instrument it

Wrap `dist.all_reduce` to count calls and bytes:

```python
import functools
stats = {"calls": 0, "bytes": 0}
_orig = dist.all_reduce
@functools.wraps(_orig)
def counting_all_reduce(t, *a, **kw):
    stats["calls"] += 1; stats["bytes"] += t.numel() * t.element_size()
    return _orig(t, *a, **kw)
dist.all_reduce = counting_all_reduce      # patch before running the forward pass
```

Predict first: one attention + one MLP forward on `(B=2, T=32, C=256)` FP32 should be **2 calls × 2·32·256·4 B = 64 KiB each**.

> [!TIP] Moving to real GPUs later
> Change `"gloo"` → `"nccl"`, put tensors on `torch.device("cuda", rank)` (call `torch.cuda.set_device(rank)` first), and the same code runs on an 8-GPU node. That is essentially how gpt-fast's `tp.py` works.

- [ ] `tp_mlp.py` passes at TP = 1, 2, 4
- [ ] `tp_layers.py` passes at TP = 1, 2, 4 (and fails loudly at TP = 3 — why?)
- [ ] Counted all-reduce calls/bytes and matched your prediction
- [ ] Converted `TPMLP` to Llama's **SwiGLU**: `down(silu(gate(x)) * up(x))` — which of gate/up/down are column vs row?
- [ ] Added GQA: `n_kv_heads < n_heads`; handled `world > n_kv_heads` by replicating KV heads
- [ ] Timed TP=1 vs TP=2 vs TP=4 on your Mac. It probably got **slower** — explain using $\alpha$ from the previous lesson
      */}),
      resources: [
        { title: "gpt-fast (tp.py)", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "~150 lines of real TP for Llama on NCCL — read after you finish this build" },
        { title: "Picotron", url: "https://github.com/huggingface/picotron", type: "repo", note: "minimal, hackable 4D-parallel reference (TP/PP/DP/CP) to compare your layers against" },
      ],
    },
    {
      id: "pipeline-parallel",
      title: "Pipeline parallelism and bubbles",
      kind: "concept",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
Pipeline parallelism cuts the model **by depth**: GPU 0 gets layers 0–39, GPU 1 gets layers 40–79. Only the **activations at the boundary** cross the wire — for a 70B model that's $T \times 8192 \times 2$ bytes per stage boundary, once per forward pass, instead of 160 all-reduces. That is *far* less communication, which is why PP works over slow links.

The catch: a request must go through stage 0, **then** stage 1. While stage 1 works, stage 0 is idle — unless another batch is ready to enter. Split the batch into $m$ **micro-batches** and stream them through $p$ stages:

::viz pipeline-bubble

The idle fraction ("bubble") with $p$ stages and $m$ micro-batches is

$$
\text{bubble} = \frac{p-1}{m+p-1}.
$$

Example: $p=4$, $m=4$ → $\tfrac{3}{7} \approx 43\%$ idle. $m = 32$ → $\tfrac{3}{35} \approx 9\%$.

> [!MATH] Where the formula comes from
> Draw the schedule as a grid: time slots on x, stages on y. The last micro-batch leaves the last stage after $m + p - 1$ slots. Each stage is busy for exactly $m$ of those slots. Idle fraction $= 1 - \frac{m}{m+p-1} = \frac{p-1}{m+p-1}$.

## PP in inference is different from training

- **Latency doesn't improve.** One request still runs all 80 layers in sequence, plus a hop between stages. TP=4 makes a single token ~3–4× faster; PP=4 makes it *no faster* (slightly slower).
- **Throughput can be fine.** An engine with continuous batching always has many sequences in flight, so it keeps $m$ large: different micro-batches occupy different stages. vLLM does this with `--pipeline-parallel-size`.
- **Memory per GPU shrinks** — weights *and* KV for your layers only. That's the reason to use it.

| | TP=4 | PP=4 |
|---|---|---|
| per-token latency | ≈ ¼ (+ comm) | ≈ 1× (+ hops) |
| comm per forward | 2 × layers all-reduces | $p-1$ point-to-point sends |
| link it needs | NVLink | anything (IB, Ethernet, PCIe) |
| failure mode | all-reduce latency | bubbles, stage imbalance |

## Try it: a 2-stage pipeline with send/recv

```python
# pp_toy.py — run: torchrun --standalone --nproc_per_node=2 pp_toy.py
import time, torch, torch.nn as nn, torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)
C, L, B, M = 512, 8, 32, 4                        # width, total layers, batch, micro-batches
layers = [nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C)) for _ in range(L)]
x = torch.randn(B, C)

per = L // world                                   # this rank's stage = a contiguous slice of layers
stage = nn.Sequential(*layers[rank * per:(rank + 1) * per])

with torch.no_grad():
    outs = []
    t0 = time.perf_counter()
    for mb in x.chunk(M):                          # stream micro-batches through the pipe
        if rank == 0:
            dist.send(stage(mb), dst=1)            # point-to-point: activations to next stage
        else:
            h = torch.empty(mb.shape[0], C)
            dist.recv(h, src=0)
            outs.append(stage(h))
    dt = time.perf_counter() - t0
    if rank == world - 1:
        ref = nn.Sequential(*layers)(x)
        print(f"PP={world} M={M}: max err {(torch.cat(outs) - ref).abs().max():.1e}, {1e3*dt:.1f} ms")
dist.destroy_process_group()
```

- [ ] Ran `pp_toy.py`; confirmed error ≈ 0
- [ ] Varied `M` = 1, 2, 4, 8, 16 and plotted time — does it follow the bubble formula? (On CPU, both processes share cores, so be skeptical of the numbers.)
- [ ] Computed the PP boundary traffic for Llama-3.1-70B, 8K prompt, PP=2 — compare to TP=2's total all-reduce bytes
      */}),
      resources: [
        { title: "GPipe (Huang et al., 2018)", url: "https://arxiv.org/abs/1811.06965", type: "paper", note: "micro-batch pipelining and the bubble" },
        { title: "Megatron-LM at scale (Narayanan et al., 2021)", url: "https://arxiv.org/abs/2104.04473", type: "paper", note: "how TP + PP + DP compose; interleaved schedules" },
      ],
    },
    {
      id: "interconnects",
      title: "Interconnects: NVLink, NVSwitch, PCIe, InfiniBand, RoCE, NVL72",
      kind: "concept",
      minutes: 50,
      md: MD(function () {/*
Your parallelism layout is decided by the **wires**. Memorize the orders of magnitude:

::viz memory-hierarchy

| Link | Connects | Bandwidth (per GPU, one direction) | Latency | Notes |
|---|---|---|---|---|
| **HBM** (for comparison) | GPU ↔ its memory | 3.35 TB/s (H100), 4.8 TB/s (H200), 8 TB/s (B200) | ~ns | the thing we're trying not to be slower than |
| **NVLink 4** (Hopper) | GPU ↔ GPU in a node | ~450 GB/s (900 GB/s bidirectional) | ~µs | 18 links per H100 |
| **NVLink 5** (Blackwell) | GPU ↔ GPU | ~900 GB/s (1.8 TB/s bidirectional) | ~µs | |
| **NVSwitch** | all 8 GPUs of an HGX node | full NVLink speed between **any** pair | | makes the node a crossbar; NVLS can even do reductions *in the switch* |
| **PCIe 5.0 x16** | GPU ↔ CPU / NIC / other GPU | ~64 GB/s theoretical, ~50 real | µs | L4 / L40S / RTX cards have **only** this |
| **InfiniBand NDR** | node ↔ node | 400 Gb/s = **50 GB/s** per NIC; typically 1 NIC per GPU (8/node) | a few µs (RDMA) | NVIDIA (Mellanox); GPUDirect RDMA skips the CPU |
| **RoCE v2** | node ↔ node | 200–400 Gb/s per NIC | a few µs | RDMA over Converged Ethernet — what many clouds use instead of IB |
| **Plain Ethernet / TCP** | node ↔ node | 25–100 Gb/s | tens of µs | fine for PP or KV transfer, bad for TP |

> [!INTUITION] The ratios that matter
> HBM : NVLink : IB ≈ **70 : 9 : 1** per GPU. NVLink is about **an order of magnitude** faster than a node's network — so anything that talks **every layer** (TP) stays inside the NVLink domain, and anything that talks **rarely or in bulk** (PP, DP, KV transfer, EP with overlap) may cross the network.

## NVL72: making the "node" bigger

**GB200 NVL72** connects **72 Blackwell GPUs** (and 36 Grace CPUs) in one rack with NVLink switches — one NVLink domain of 72 GPUs with ~13.5 TB of HBM. For inference this changes the rules: TP or wide EP across 72 GPUs **without touching InfiniBand**. It's why SGLang/TensorRT-LLM/Dynamo showcase DeepSeek-class MoE serving on NVL72 (M17).

## How to check what you rented

```bash
nvidia-smi topo -m          # NV18 = 18 NVLinks between that pair; PIX/PHB/SYS = PCIe paths
nvidia-smi nvlink -s        # per-link speed
ibstat                      # InfiniBand ports (if any) and their rate
```

If `topo -m` shows `SYS` or `PHB` between GPUs, you're on PCIe: **don't use TP > 2** — prefer replicas or PP.

> [!WARNING] Cloud gotchas
> "8× H100" can mean **SXM** (NVLink + NVSwitch, what you want) or **PCIe/NVL** cards (pairs bridged, rest over PCIe). Multi-node instances may have fewer NICs than GPUs, or plain Ethernet. Always run `topo -m` and a quick all-reduce benchmark before you trust a node.

> [!REAL] Inference Engineering (Baseten) ch. 3 & 5.4
> NVLink 900 GB/s on Hopper and 1,800 GB/s on Blackwell; InfiniBand up to 400 Gb/s per NIC; **"IB is in Gb/s, not GB/s."** Ada cards (L4, L40S) have no NVLink, so multi-GPU falls back to pipeline parallelism.

- [ ] Converted every row to one-direction GB/s and computed HBM:NVLink:IB ratios yourself
- [ ] For a 134 MB all-reduce on 8 GPUs, estimated the time over NVLink 4, PCIe 5, and IB NDR (using the ring formula)
      */}),
      resources: [
        { title: "NVIDIA NVLink & NVSwitch", url: "https://www.nvidia.com/en-us/data-center/nvlink/", type: "docs", note: "official bandwidth numbers per generation" },
        { title: "NVIDIA GB200 NVL72", url: "https://www.nvidia.com/en-us/data-center/gb200-nvl72/", type: "docs", note: "the 72-GPU NVLink domain that reshapes MoE serving" },
      ],
    },
    {
      id: "choosing-layout",
      title: "Choosing a layout: TP within a node, PP/EP/DP across nodes",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
You now have all the pieces. Here's the decision procedure inference engineers actually use.

## Rules of thumb

1. **Does it fit on one GPU with enough KV headroom for your target concurrency?** → Use **1 GPU per replica** and scale with **DP** (more replicas). Zero communication. This is right for most ≤ 8–14B models.
2. **Doesn't fit, or need lower per-user latency?** → **TP** inside one NVLink domain. Pick the **smallest** power of two that fits weights + KV (TP=2, 4, 8)… then consider going **one step higher** if your ITL SLO needs it.
3. **Doesn't fit in one node?** First ask: can **FP8/FP4** make it fit? (405B FP8 = 405 GB fits 8× H100; 810 GB BF16 does not.) If not → **TP within node × PP across nodes** (e.g. `TP8 PP2`).
4. **MoE model?** → **EP** for the expert FFNs (+ DP or TP for attention). Across nodes, EP beats PP for throughput; TP8·PP2 gives lower per-user latency (Baseten ch. 5.4).
5. **No NVLink** (L4, L40S, consumer cards)? → replicas, or **PP**; TP=2 at most.
6. **Divisibility:** TP must divide the number of attention heads (and ideally KV heads) and the FFN dim. Llama-3-70B: 64 Q heads, 8 KV heads → TP ∈ {1, 2, 4, 8}.
7. **Extra nodes that you don't strictly need** → spend them on **replicas or disaggregation** (M17), not on more parallelism.

> [!INTUITION] Latency vs throughput per GPU
> More TP → lower latency per user, but **lower throughput per GPU** (communication overhead + smaller per-GPU matmuls). If you're cost-optimizing a batch job, prefer the smallest TP that fits and add replicas. If you're selling a fast chat endpoint, pay for more TP.

## Worked sizing: Llama-3.1-70B on H100-80GB

Memory per GPU at TP=N: weights $\tfrac{141}{N}$ GB + activations/CUDA graphs (~4–6 GB) + KV. vLLM uses `--gpu-memory-utilization 0.9` → ~72 GB usable.

| TP | Weights/GPU | KV room/GPU | KV tokens total (320 KiB/token ÷ N per GPU) | Concurrency @ 8K ctx |
|---|---|---|---|---|
| 2 | 70.5 GB | ~0 | — | doesn't really fit |
| 4 | 35 GB | ~32 GB | $4 \times 32\text{ GB} / 320\text{ KiB} \approx$ 390K | ~48 |
| 8 | 17.6 GB | ~50 GB | $\approx$ 1.2M | ~150 |

Note: TP=8 has **3× the KV capacity** of TP=4 with only 2× the GPUs. Bigger TP can *raise* throughput when you were KV-bound.

## Worked sizing: Llama-3.1-405B

- KV per token (126 layers, 8 KV heads, head dim 128, BF16): $2 \times 126 \times 8 \times 128 \times 2 = 516{,}096$ B ≈ 504 KiB.
- BF16: 810 GB → does not fit 8× H100 (640 GB). Options: **2 nodes, TP8 PP2** (IB between them), **8× H200** (1,128 GB → roughly 170–250 GB left for KV depending on memory-utilization settings ≈ 330K–480K tokens), or **8× B200**.
- FP8: 405 GB → 8× H100 at 90% utilization leaves ~140 GB for KV ≈ 270K tokens (double that with an FP8 KV cache). Most production 405B deployments look like this.

## In vLLM terms

```bash
# 70B, one node, 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --max-model-len 16384

# 405B BF16 across 2 nodes (Ray cluster already up): TP inside nodes, PP across
vllm serve meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2

# MoE: attention data-parallel, experts expert-parallel over 8 GPUs
vllm serve Qwen/Qwen3-30B-A3B --data-parallel-size 8 --enable-expert-parallel
```

> [!REAL] What the book recommends (Inference Engineering ch. 5.4)
> Dense multi-node: **TP8 PP2**. MoE multi-node: **EP16** or TP8 PP2. "TP8PP2 gives lower per-user latency; EP16 gives higher system throughput." Multi-node is for huge models at high precision, multi-million-token contexts, or maximum speed.

- [ ] Reproduced the 70B table for H200 (141 GB) — what's the smallest TP now?
- [ ] Wrote a one-paragraph layout recommendation for Qwen2.5-32B on (a) 2× L40S (48 GB, PCIe) and (b) 2× H100 SXM
- [ ] Wrote a tiny `size.py`: given params, bytes/param, KV bytes/token, GPU memory, target concurrency × context → minimum TP
      */}),
      resources: [
        { title: "vLLM — Parallelism & scaling", url: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", type: "docs", note: "official guidance on TP/PP/DP choices and multi-node setup with Ray" },
        { title: "Efficiently Scaling Transformer Inference (Pope et al., 2022)", url: "https://arxiv.org/abs/2211.05102", type: "paper", note: "the canonical analysis of partitioning layouts for inference latency vs cost" },
      ],
    },
    {
      id: "lab-vllm-tp",
      title: "Lab: vLLM at TP = 1 / 2 / 4 (/ 8) on a rented multi-GPU node",
      kind: "lab",
      minutes: 180,
      runsOn: ["multi-gpu", "cloud"],
      md: MD(function () {/*
> [!GOAL] Output of this lab
> A table + two plots (ITL vs TP, throughput-per-GPU vs TP) for Llama-3.1-8B and Llama-3.1-70B, with your explanation of each trend using this module's math.

## 0 — Rent and check the box

Rent **4× or 8× H100 SXM** (a single node). Cost note: roughly **\$2–3 per GPU-hour** on-demand, so 4× H100 for 2–3 hours ≈ **\$25–35**; 8× for 2 hours ≈ \$40–50. Budget option: do the 8B part on **2–4× A100/L40S** and compare NVLink vs PCIe. Download weights **before** starting the clock if the provider offers a cheap CPU instance with shared storage.

```bash
nvidia-smi topo -m                      # expect NV18 between every GPU pair (SXM + NVSwitch)
pip install vllm                        # brings a matching PyTorch + NCCL
export HF_TOKEN=...                     # Llama weights are gated: accept the license on HF first
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct   # ~140 GB, start early
```

Optional but educational — measure the wire first:

```bash
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make -j
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 4     # note busbw at 1 GB and latency at 8 B
```

## 1 — Latency: one request, varying TP

```bash
for TP in 1 2 4; do
  vllm bench latency --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size $TP --input-len 1024 --output-len 128 --batch-size 1 \
    --num-iters 10 2>&1 | tail -n 3
done
```

## 2 — Throughput under load

```bash
# terminal 1
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --max-model-len 8192

# terminal 2 — sweep concurrency
for C in 1 16 64; do
  vllm bench serve --backend vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 300 --max-concurrency $C \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99
done
```

Repeat for TP = 1, 2, 4. Then **70B at TP=4** (and TP=8 if you have 8 GPUs), with `--max-model-len 8192`.

## 3 — Record and explain

| model | TP | ITL p50 @c=1 | TTFT p50 @c=16 | output tok/s @c=64 | tok/s per GPU | KV tokens (from vLLM startup log) |
|---|---|---|---|---|---|---|
| 8B | 1 | | | | | |
| 8B | 2 | | | | | |
| 8B | 4 | | | | | |
| 70B | 4 | | | | | |
| 70B | 8 | | | | | |

The startup log prints the KV capacity (look for "GPU KV cache size" / "Maximum concurrency"). Compare it to your `choosing-layout` prediction.

> [!CHECK] Explain these (you should see them)
> 1. 8B: TP=1→2 improves ITL by much less than 2× — model is small, per-GPU matmuls get tiny, all-reduce latency and kernel launch overhead dominate.
> 2. Throughput **per GPU** falls as TP grows for 8B — replicas (DP) would beat TP for throughput.
> 3. 70B TP=8 vs TP=4: better ITL *and* much more KV capacity → higher throughput at high concurrency.
> 4. Predicted ITL (weights/N ÷ 3.35 TB/s + 160 × α) vs measured — within 30%?

> [!TIP] Profile one step (optional)
> `vllm bench latency ... --profile` (or `VLLM_TORCH_PROFILER_DIR=./traces`) writes a trace; open it in Perfetto and find the `all_reduce` / `cross_device_reduce` kernels between the GEMMs. Their share of the step time is your "communication tax".

- [ ] Verified NVLink topology and (optionally) ran nccl-tests
- [ ] Filled the table for 8B at TP 1/2/4 and 70B at TP 4 (and 8)
- [ ] Plotted ITL and per-GPU throughput vs TP
- [ ] Wrote a half-page explanation of each trend using the α–β model
- [ ] Shut the instance down (check the billing page!)
      */}),
      resources: [
        { title: "vLLM — Parallelism & scaling", url: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", type: "docs", note: "flags for TP/PP/DP and multi-node" },
        { title: "NVIDIA nccl-tests", url: "https://github.com/NVIDIA/nccl-tests", type: "tool", note: "verify the interconnect before blaming the engine" },
      ],
    },
    {
      id: "deep-nccl",
      title: "Deep dive: NCCL and fast all-reduce in real engines",
      kind: "deep",
      optional: true,
      minutes: 60,
      runsOn: ["multi-gpu"],
      md: MD(function () {/*
**NCCL** (NVIDIA Collective Communications Library, "nickel") is what `backend="nccl"` calls. Every engine sits on it, and when multi-GPU inference is slow, it's often NCCL misconfiguration.

## What NCCL decides for you

- **Algorithm:** *Ring* (bandwidth-optimal, latency $\propto N$), *Tree* (latency $\propto \log N$, good for small messages and many nodes), **NVLS** (NVLink SHARP — the NVSwitch itself performs the reduction; Hopper+), *CollNet* (in-network reduction on InfiniBand SHARP switches).
- **Protocol:** *Simple* (big messages, max bandwidth), *LL* / *LL128* (low-latency: flags packed with data so the receiver doesn't need a separate sync — lower α, less bandwidth).
- **Transport:** NVLink P2P inside a node, GPUDirect RDMA over IB/RoCE between nodes, sockets as last resort.

Useful knobs when debugging:

```bash
NCCL_DEBUG=INFO            # prints chosen transports/rings at init — read it once
NCCL_ALGO=Ring|Tree|NVLS   # force an algorithm to compare
NCCL_PROTO=LL|LL128|Simple
NCCL_P2P_DISABLE=1         # simulate "no NVLink/P2P" — watch TP collapse
NCCL_IB_HCA=mlx5_0,mlx5_1  # which IB NICs to use
NCCL_SOCKET_IFNAME=eth0    # which interface for bootstrap/sockets
```

## Why engines don't *only* use NCCL

Decode all-reduces are **small** (KB–MB) and happen **160× per token** for a 70B model: pure α territory. So:

- **vLLM custom all-reduce:** for small tensors inside one NVLink node, each GPU writes directly into peers' buffers through CUDA IPC and reduces in one kernel ("one-shot"), or a reduce-scatter + all-gather in two ("two-shot"). Lower latency than NCCL for small sizes; falls back to NCCL for large ones. Disable with `--disable-custom-all-reduce` to measure the difference.
- **PyTorch SymmetricMemory** provides similar building blocks natively.
- **TensorRT-LLM** fuses all-reduce with the next RMSNorm (+ residual add, + FP8 quantize) to delete a kernel and a memory round-trip.
- **CUDA graphs:** collectives get captured into the decode graph so there's no CPU launch overhead per all-reduce (M13).
- **Overlap:** split the batch in two and compute micro-batch B while micro-batch A's all-reduce is in flight (the same "dual-batch overlap" DeepSeek uses for EP — M17).

> [!REAL] How to spot it in a profile
> In an Nsight Systems or PyTorch-profiler trace of a TP=8 decode step you'll see GEMM → `cross_device_reduce_1stage` (vLLM custom) or `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` → GEMM. If the all-reduce bars are as wide as the GEMMs, you're latency-bound: try a lower TP, CUDA graphs, or larger batches.

- [ ] Ran vLLM 70B TP=8 with and without `--disable-custom-all-reduce`; compared ITL at c=1
- [ ] Ran nccl-tests with `NCCL_ALGO=Ring` vs `Tree` vs `NVLS`; plotted latency at 8 KB–1 MB
- [ ] Read the NCCL user guide section on environment variables
      */}),
      resources: [
        { title: "NCCL user guide", url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html", type: "docs", note: "algorithms, protocols, env vars, troubleshooting" },
        { title: "NVIDIA/nccl", url: "https://github.com/NVIDIA/nccl", type: "repo", note: "source; issues are a goldmine for real-world debugging" },
      ],
    },
    {
      id: "deep-sp-cp",
      title: "Deep dive: sequence & context parallelism for long context (ring attention)",
      kind: "deep",
      optional: true,
      minutes: 60,
      md: MD(function () {/*
TP and PP split **weights**. When the problem is a **1M-token prompt**, the bottleneck is attention over a gigantic sequence — so split the **sequence**.

## Megatron sequence parallelism (SP)

In a TP layer, LayerNorm, dropout and residual adds run **replicated** on every rank on the full $(T, C)$ activations — wasted memory. SP shards those regions along $T$. The all-reduce around each TP block becomes a **reduce-scatter** (leaving the block, shard by sequence) + **all-gather** (entering the next block). Same bytes on the wire (all-reduce = reduce-scatter + all-gather!), $\tfrac{1}{N}$ the activation memory. Mostly a training win; some engines use it for large-batch prefill.

## Context parallelism (CP) and ring attention

Give each of $N$ GPUs a contiguous chunk of the sequence (full weights on each). Linear layers are embarrassingly parallel over tokens. Only **attention** needs other chunks' K and V. **Ring attention**:

1. GPU $i$ holds $Q_i, K_i, V_i$ for its chunk.
2. For $N$ steps: compute attention of $Q_i$ against the KV block you currently hold, **accumulate with online softmax** (running max + running sum — exactly the FlashAttention trick from M12), then pass your KV block to the next GPU in the ring while receiving one from the previous.
3. Communication of step $s+1$ overlaps compute of step $s$.

It's **exact** (not an approximation). It hides communication if computing one block takes longer than sending it:

$$
\underbrace{\frac{4\, T_b^2\, d}{\text{FLOP/s}}}_{\text{attention for one block pair}} \;\ge\; \underbrace{\frac{2\, T_b\, d_{kv} \cdot \text{bytes}}{B}}_{\text{send one KV block}}
$$

Compute grows with $T_b^2$, comms with $T_b$ → big enough blocks always win. That's why CP works even over ordinary data-center networks.

**Ulysses** (DeepSpeed) is the alternative: an **all-to-all** switches from sequence-sharded to **head-sharded** before attention (each GPU gets all tokens for $H/N$ heads), then back. Cheaper comm, but $N$ is capped by the number of heads.

> [!REAL] Numbers from Meta
> "Context Parallelism for Scalable Million-Token Inference" reports **1M-token prefill of Llama 3 405B in 77 s on 128 H100s (16 nodes)** at 93% parallel efficiency, and 128K prefill in 3.8 s — over both RDMA and TCP networks. Decode is a different story: one new token per step gives nothing to split, so CP is a **prefill** technique (pair it with disaggregation — M17).

- [ ] Explained why online softmax makes ring attention exact
- [ ] Computed the block size $T_b$ at which ring-attention compute hides communication for H100s over 50 GB/s IB (use $d = 128$ per head, 64 heads, BF16)
- [ ] Stretch: simulate ring attention with `torch.distributed` send/recv on your Mac (4 ranks, verify against full attention)
      */}),
      resources: [
        { title: "Ring Attention (Liu et al., 2023)", url: "https://arxiv.org/abs/2310.01889", type: "paper", note: "blockwise attention passed around a ring for near-infinite context" },
        { title: "Context Parallelism for Million-Token Inference", url: "https://arxiv.org/abs/2411.01783", type: "paper", note: "pass-KV vs pass-Q ring attention for inference, with production numbers" },
        { title: "DeepSpeed Ulysses", url: "https://arxiv.org/abs/2309.14509", type: "paper", note: "all-to-all sequence parallelism over heads" },
        { title: "Reducing Activation Recomputation (Megatron SP)", url: "https://arxiv.org/abs/2205.05198", type: "paper", note: "introduces Megatron sequence parallelism" },
      ],
    },
  ],

  challenge: {
    title: "Tensor-parallel transformer block + a sizing doc for 70B and 405B",
    md: MD(function () {/*
**Part A — code (Mac).** Build `tp_block.py`: a full **Llama-style** transformer block — RMSNorm → attention with **RoPE and GQA** → residual → RMSNorm → **SwiGLU** MLP → residual — in both a reference single-process version and a tensor-parallel version using your `ColumnParallelLinear` / `RowParallelLinear`. Add a `pytest` suite launched with `torchrun` (or `torch.multiprocessing.spawn`) that checks:

- outputs match the reference for TP ∈ {1, 2, 4} with `n_heads=8, n_kv_heads=2` (so TP=4 forces KV-head replication), tolerance 1e-5 in FP32;
- a **decode step with KV cache** (1 new token after a 32-token prefill) also matches — each rank caches only its own KV heads;
- exactly **2 all-reduces** per block per forward (use your counting wrapper).

**Part B — sizing doc (`SIZING.md`, 1–2 pages).** For **Llama-3.1-70B** and **Llama-3.1-405B**, target: 8K average context, 64 concurrent users, ITL p50 < 30 ms. For each model:

1. Weights (BF16 and FP8), KV bytes/token, KV for the target load.
2. Candidate layouts on H100-80GB and H200-141GB (e.g. 70B: TP4, TP8, 2×TP4 replicas; 405B: FP8 TP8, BF16 TP8·PP2, H200 BF16 TP8).
3. Predicted ITL per layout (weights/N ÷ bandwidth + all-reduce latency × 2 × layers) and KV capacity.
4. Your pick, the \$/hr, and what you'd measure first to validate it. Use your lab numbers to calibrate.
    */}),
    checklist: [
      "`tp_block.py` reference and TP versions match at TP = 1, 2, 4 (prefill and a KV-cached decode step)",
      "GQA with `n_kv_heads < TP` handled by replicating KV heads, with a test for it",
      "Test asserts exactly 2 all-reduces per block forward",
      "`SIZING.md` shows memory math for 70B and 405B in BF16 and FP8, including KV at the target load",
      "Each candidate layout has a predicted ITL and KV capacity, and one is recommended with a \\$/hr figure",
      "At least one prediction is checked against a number you measured in the vLLM lab",
    ],
    stretch: "Port `tp_block.py` to NCCL on a 2–4 GPU cloud box, stack 4 blocks, and compare your per-step latency against vLLM running the same shapes. Where does the gap come from (CUDA graphs? custom all-reduce? fused kernels)?",
  },

  connects: MD(function () {/*
You can now split a dense model across GPUs and predict what it costs. **M17** takes the next two steps that frontier deployments take: splitting the **phases** of inference (prefill vs decode) across different GPU pools — which turns the KV cache into something you ship over the network — and serving **MoE** models, where expert parallelism and all-to-all replace TP's all-reduce. In **M18–M19** these layouts become deployment configs you autoscale and benchmark; in **M23** the same collectives (broadcast, all-gather) are how RL systems sync fresh weights into inference engines.
  */}),

  interview: [
    "Llama-3.1-70B in BF16: how many H100-80GB GPUs do you need, and why might you choose more than the minimum?",
    "Explain Megatron tensor parallelism for an MLP. Why column-split the first matmul and row-split the second? How many all-reduces per transformer layer?",
    "Derive the cost of ring all-reduce. Why is the bandwidth term almost independent of the number of GPUs, and which term dominates for decode?",
    "Why is TP generally kept within a node, while PP or EP is used across nodes?",
    "What is the pipeline bubble? Why does pipeline parallelism not reduce single-request latency in inference?",
    "Convert 400 Gb/s InfiniBand and 900 GB/s NVLink into comparable one-direction GB/s numbers. What does the ratio imply?",
    "You moved a model from TP=4 to TP=8 and throughput per GPU dropped but total throughput rose. Explain both.",
    "How would you serve Llama-3.1-405B on H100s? Compare FP8 TP8 against BF16 TP8·PP2.",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 5.4 Model parallelism", url: "Inference%20Engineering.pdf", type: "book", note: "TP vs EP vs PP trade-offs, multi-node options, GPU-count formula" },
    { title: "Megatron-LM paper", url: "https://arxiv.org/abs/1909.08053", type: "paper", note: "the origin of column/row tensor parallelism" },
    { title: "Megatron-LM repo", url: "https://github.com/NVIDIA/Megatron-LM", type: "repo", note: "reference implementations of TP, PP, SP and CP" },
    { title: "HF Ultrascale Playbook", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook", type: "course", note: "visual, end-to-end tour of every parallelism strategy" },
    { title: "JAX Scaling Book — Inference chapter", url: "https://jax-ml.github.io/scaling-book/inference/", type: "book", note: "roofline + communication math for serving, with exercises" },
    { title: "Efficiently Scaling Transformer Inference", url: "https://arxiv.org/abs/2211.05102", type: "paper", note: "partitioning layouts for inference, latency vs cost Pareto" },
    { title: "PyTorch — Writing distributed applications", url: "https://pytorch.org/tutorials/intermediate/dist_tuto.html", type: "docs", note: "collectives from first principles, including a hand-written ring all-reduce" },
    { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "compact, real tensor-parallel Llama inference" },
    { title: "vLLM — Parallelism & scaling", url: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", type: "docs", note: "the production knobs you'll use daily" },
    { title: "NCCL user guide", url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html", type: "docs", note: "debugging and tuning collectives" },
  ],
});
