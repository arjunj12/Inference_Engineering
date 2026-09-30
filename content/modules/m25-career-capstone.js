Course.module({
  id: "m25-career-capstone",
  title: "Career: become a hireable inference engineer",
  short: "Career capstone",
  tagline: "Turn everything you built into a portfolio, rehearse the interviews inference teams actually run, land an open-source PR, and run a 90-day job search.",
  hours: 30,
  level: "core",
  capstone: true,
  runsOn: ["any"],
  tags: ["career", "interviews", "system-design", "portfolio", "open-source"],

  goal: MD(function () {/*
At the end of this module you have a **job-search kit** you can send tomorrow:

```text
github.com/<you>/inference-portfolio
├── README.md                ← 1-page pitch: who you are, 5 projects, 5 numbers
├── mini-vllm/               ← M10: paged KV cache, continuous batching, OpenAI API  (+ benchmark charts)
├── kernels/                 ← M11–M12: tiled matmul, fused softmax, flash-attention (Triton)  (+ roofline plot)
├── benchmark-report/        ← M19: vLLM vs SGLang, FP16 vs FP8, $/1M tokens, p99 under load
├── grpo-run/                ← M22–M23: GRPO on a small model + actor–learner rollout system
├── continual-learning/      ← M24: forgetting matrices + LLM forgetting study
├── system-design/           ← 2 written mock designs (this module)
└── oss.md                   ← links to your merged/open PRs in vLLM / SGLang / llm-d / Dynamo
```

…plus the ability to do, **out loud and without notes**: size a KV cache, bound decode tok/s from bandwidth, estimate GPUs for a QPS target, design a serving system with SLOs, write a top-p sampler and a block manager, and debug a p99 spike. That's what the 2025–2026 postings screen for.
  */}),
  demo: { viz: "inference-stack", params: {} },

  why: MD(function () {/*
Inference engineering became its own job family in 2024–2026: Baseten, Together, Fireworks, Modal, Perplexity, Cerebras, CoreWeave, NVIDIA, Anthropic, OpenAI, Google DeepMind, xAI and others post roles that name **vLLM/SGLang/TensorRT-LLM internals, KV-cache management, CUDA/Triton, quantization, distributed inference and RL infrastructure** as requirements. Postings and hiring guidance repeatedly say the same thing: **show work people can inspect** — open-source PRs, benchmarked projects, write-ups with numbers. You've built that work; this module packages it and trains the interview muscles.
  */}),

  prereqs: [
    {
      title: "Which course work you should have finished",
      skipIf: "you completed the challenges of M10, M12, M19 and M22",
      md: MD(function () {/*
You can start this module any time (reading postings early helps you focus), but the portfolio lessons assume these challenges are done or nearly done:

| Flagship project | Module | Minimum to be "portfolio-ready" |
|---|---|---|
| mini-vLLM engine | M10 | runs, streams, has a benchmark vs. naive HF generate |
| GPU kernels | M11–M12 | correctness tests + speed vs. PyTorch on one GPU |
| Benchmark report | M19 | load-test curves (throughput vs p99), \$/1M tokens |
| GRPO / RL system | M22–M23 | reward curve + a rollout/trainer timing breakdown |
| Continual-learning report | M24 | accuracy matrices + LLM forgetting table |

If one is missing, the 90-day plan (last lesson) schedules it.
      */}),
    },
  ],

  lessons: [
    {
      id: "archetypes",
      title: "The five role archetypes — and which one you're aiming at",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
Across ~70 postings from ~22 companies (researched for this course, 2025–2026), inference jobs cluster into four archetypes plus a custom-silicon variant. Titles vary wildly ("MTS", "Performance Engineer", "Software Engineer, Model Inference") — **read the responsibilities, not the title.**

## A. Kernel / GPU performance engineer
**Day to day:** write and tune CUDA/Triton/CUTLASS kernels (GEMMs, attention, MoE routing, quantized matmuls, fused epilogues); read PTX/SASS; live in Nsight Compute; chase 10–30% wins on the hot path; co-design kernels with model architecture.
**Screened on:** GPU architecture, CUDA/Triton, profiling, C++, roofline reasoning. **Modules that matter most:** M07, M11, M12, M13, M14.
**Example postings:** Baseten *GPU Kernels*, Anthropic *Performance Engineer, GPU*, xAI *Kernels/CUDA*, CoreWeave *GPU Kernel Authoring*.

## B. Serving-runtime / inference-systems engineer
**Day to day:** build the engine — schedulers, continuous batching, KV-cache and prefix caching, speculative decoding, disaggregated prefill/decode, quantization integration, multi-LoRA — often directly inside vLLM/SGLang/TRT-LLM/Dynamo or a Rust/C++ in-house engine.
**Screened on:** engine internals, distributed inference (TP/PP/EP), quantization, spec decode, Python + C++/Rust. **Modules:** M06, M08, M09, M10, M14, M15, M16, M17.
**Example postings:** Perplexity *AI Inference Engineer*, Anthropic *Performance Engineer, Inference Engine*, Modal *MTS Research, Inference*, Cerebras *Staff SWE GPU Inference*.

## C. Inference platform / infrastructure engineer
**Day to day:** Kubernetes control planes and operators for GPU fleets, autoscaling, routing/load balancing, cold starts (weight loading, caching), multi-cloud capacity, observability, RDMA networking, reliability — usually engine-agnostic.
**Screened on:** Kubernetes internals, Go/Python/Rust, networking, SLOs, autoscaling design. **Modules:** M18, M19, M17, M16, M13.
**Example postings:** Together *Inference/Compute Infra*, Baseten *Inference Stack* and *GPU Networking*, Cohere *Inference Infrastructure*, Google Cloud *GKE Platform for AI Inference* (llm-d).

## D. RL / post-training infrastructure engineer
**Day to day:** rollout generation at scale with inference engines, trainer→rollout **weight sync**, reward and eval pipelines, sandboxed agent environments, co-optimizing the engine for RL (async rollouts, low precision, long-tail generations).
**Screened on:** RL fundamentals (PPO/GRPO), distributed training (FSDP/Megatron), engine internals, systems glue. **Modules:** M21, M22, M23, M24, M16.
**Example postings:** Together *Research Engineer, Post-Training Inference*, xAI *MTS RL Inference*, Scale *ML Systems (RLXF)*, Google DeepMind *Gemini Post-training*.

## E. Custom silicon variant
The same four jobs on **non-NVIDIA hardware**: Cerebras (CSL kernels), TPUs (Pallas/JAX/XLA), Trainium, AMD (ROCm/HIP). Same fundamentals — roofline, memory hierarchy, scheduling — different ISA and compiler. **Modules:** M07, M11–M13 fundamentals + the vendor's docs.

::viz inference-stack

## Pick a primary and a secondary

| If you enjoy… | Primary | Natural secondary |
|---|---|---|
| making one op 2× faster | A. Kernel | B. Runtime |
| schedulers, caches, data structures | B. Runtime | C. Platform or D. RL infra |
| distributed systems, k8s, reliability | C. Platform | B. Runtime |
| training + inference glue, experiments | D. RL infra | B. Runtime |

> [!TIP] For a strong backend engineer (you)
> **B (runtime)** and **C (platform)** are the shortest path: your systems/backend skills transfer directly, and your mini-vLLM + benchmark report are exactly their portfolio. **D** is the fastest-growing category. **A** needs the most deliberate practice (kernels) — do it as a secondary unless you love it.

The full requirements-by-archetype map lives in the **[Job skills map](#/x/skills)** tab — it links every skill to the modules that teach it and shows your progress.

- [ ] Read 10 live postings across at least 3 archetypes; highlight every skill you *can't* yet demonstrate
- [ ] Chose a primary and a secondary archetype and wrote one sentence why
- [ ] Opened the [Job skills map](#/x/skills) and listed your 5 weakest "Must-have" skills for your archetype
      */}),
      resources: [
        { title: "Baseten — Inference Engineering (book)", url: "https://www.baseten.co/inference-engineering/", type: "book", note: "company-authored overview of exactly this job" },
        { title: "sizief/llmway — inference career plan", url: "https://github.com/sizief/llmway/blob/main/inference_career_plan.md", type: "article", note: "a concrete hands-on project plan for switching into inference" },
      ],
    },
    {
      id: "napkin-drills",
      title: "Napkin-math drills with worked answers (KV cache, tok/s, GPUs, $/token)",
      kind: "math",
      minutes: 120,
      md: MD(function () {/*
Every inference interview has a back-of-envelope section. You must do these in ~2 minutes each, out loud, with round numbers. Cover the answer, try it, then check.

> [!PREREQ] Refresher: units and the four constants to memorize
> KB/MB/GB = $10^3/10^6/10^9$ bytes (KiB/MiB/GiB = $2^{10}/2^{20}/2^{30}$; the ~7% difference never matters on a napkin). BF16 = 2 bytes, FP8 = 1, INT4 ≈ 0.5.
> **H100 SXM:** 80 GB HBM3, **3.35 TB/s**, **989 TFLOPS** dense BF16 (~1,979 FP8). **A100 80GB:** 2.0 TB/s, 312 TFLOPS. Forward pass ≈ **2 × params FLOPs per token**. Decode at small batch ≈ **bytes read ÷ bandwidth**.

## Drill 1 — KV cache per token (Llama-3-8B)

32 layers, 8 KV heads (GQA), head dim 128, BF16.

<details><summary>Answer</summary>

$$
\text{KV bytes/token} = 2_{(K,V)} \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ B} \approx 128\text{ KiB}
$$

An 8K-token sequence → **1 GiB**. Without GQA (32 KV heads) it would be 4× bigger — that's *why* GQA exists.

</details>

## Drill 2 — how many concurrent 8K sequences fit? (Llama-3-8B BF16 on one H100)

<details><summary>Answer</summary>

Weights: 8B × 2 B = 16 GB. Reserve ~10% for activations, CUDA graphs, fragmentation → ~72 GB usable, ~56 GB left for KV. At 1 GiB per 8K sequence → **~52 sequences** (~430K tokens of KV). Real engines report the KV capacity at startup — compare with vLLM's "GPU KV cache size" log line.

</details>

::viz kv-calculator

## Drill 3 — decode speed upper bound at batch 1

Llama-3-70B in FP8 on one H100. Then on 2× H100 with tensor parallelism.

<details><summary>Answer</summary>

70B × 1 B = 70 GB read per token. $70 \text{ GB} / 3.35 \text{ TB/s} \approx 21$ ms → **≤ ~48 tok/s**. With TP=2 each GPU reads 35 GB → ~10.5 ms → **≤ ~95 tok/s** minus all-reduce overhead (M16). Also: 70 GB on an 80 GB card leaves ~0 room for KV, so one GPU is a *bad* deployment even if it "fits".

</details>

## Drill 4 — prefill time (TTFT floor)

Llama-3-8B, 4,000-token prompt, H100, assume 50% MFU.

<details><summary>Answer</summary>

FLOPs ≈ 2 × 8e9 × 4,000 = 6.4e13 = 64 TFLOP. Effective 0.5 × 989 ≈ 495 TFLOPS → **~0.13 s**. Prefill is compute-bound, decode is memory-bound — say that sentence in every interview.

</details>

## Drill 5 — GPUs needed for a QPS target

100 requests/s, 1,000 input + 250 output tokens average, Llama-3-8B BF16, H100s, p99 TPOT ≤ 50 ms.

<details><summary>Answer</summary>

- **Prefill load:** 100 × 1,000 = 100K tok/s × 16 GFLOP/token = 1.6 PFLOP/s. At ~500 TFLOPS effective → **~3.2 GPUs** of prefill work.
- **Decode load:** 100 × 250 = 25K tok/s. At batch 64 a decode step reads 16 GB weights + KV (64 seqs × ~1.1K ctx × 128 KiB ≈ 9 GB) = ~25 GB → ~7.5 ms/step → 64 / 7.5 ms ≈ 8.5K tok/s ideal, ~5K tok/s realistic → **~5 GPUs**. Step time ~7.5–15 ms is well under the 50 ms TPOT SLO.
- Sum ≈ 8 GPUs of work; add ~1.5× headroom for peaks and p99 → **~12 H100s** (e.g. 12 replicas of TP=1). Validate with a load test (M19) — napkin math gets you within 2×, benchmarks get you the real number.

</details>

## Drill 6 — cost per million tokens

H100 at \$2.50/hr sustaining 5,000 output tok/s (from Drill 5).

<details><summary>Answer</summary>

5,000 × 3,600 = 18M tokens/hour → \$2.50 / 18 ≈ **\$0.14 per 1M output tokens** at 100% utilization. At a realistic 40% average utilization (diurnal traffic): ~**\$0.35**. Compare with API prices — and remember prefill GPU time isn't free.

</details>

::viz cost-calc

## Drill 7 — speculative decoding payoff

Draft acceptance rate $\alpha = 0.7$, draft length $k = 4$, draft cost 10% of a target step.

<details><summary>Answer</summary>

Expected tokens per target step $= \frac{1 - \alpha^{k+1}}{1 - \alpha} = \frac{1 - 0.7^5}{0.3} \approx 2.77$. Cost per step ≈ 1 + 4 × 0.1 = 1.4 target-steps → speedup ≈ 2.77 / 1.4 ≈ **~2×** at low batch. At high batch the target step is no longer memory-bound, so the win shrinks (M15).

</details>

## Drill 8 — the 70B deployment question

"Serve Llama-3-70B with 32K context for 20 concurrent users. How many H100s, what precision, what parallelism?"

<details><summary>Answer</summary>

KV/token = 2 × 80 layers × 8 × 128 × 2 B = 320 KiB. 32K × 20 users = 640K tokens → ~210 GB of KV (BF16). Weights 140 GB BF16 / 70 GB FP8. FP8 weights + FP8 KV (~105 GB) ≈ 175 GB + overhead → **4× H100 (320 GB), TP=4**; BF16 everything (~350 GB + overhead) → **8× H100, TP=8**. Mention prefix caching if users share system prompts, and that 20 × 32K is a worst case — size for p95 context, not max.

</details>

- [ ] Did all 8 drills with the answers hidden; redo the ones you missed 2 days later
- [ ] Made a one-page "constants cheat sheet" (GPU specs, model shapes of Llama-3-8B/70B, Qwen2.5-7B, DeepSeek-V3)
- [ ] Timed yourself: each drill in under 3 minutes, explained aloud
      */}),
      resources: [
        { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "the classic napkin-math reference" },
        { title: "How To Scale Your Model (JAX scaling book)", url: "https://jax-ml.github.io/scaling-book/", type: "book", note: "rooflines, sharding and inference math, beautifully explained" },
      ],
    },
    {
      id: "system-design",
      title: "System design: three worked designs (serving with SLOs, KV-aware router, RL rollout infra)",
      kind: "concept",
      minutes: 150,
      md: MD(function () {/*
## The 6-step framework (45-minute interview)

1. **Requirements (5 min)** — model(s), QPS and its shape (diurnal? bursty?), input/output length distributions, SLOs (TTFT p99, TPOT p99, availability), cost target, multi-tenancy, regions. *Ask*, don't assume.
2. **Napkin math (5–10 min)** — memory per replica (weights + KV), tok/s per GPU, GPUs needed, cost. (Previous lesson.)
3. **High-level architecture (10 min)** — client → gateway → router → engine replicas → GPUs; autoscaler; model/adapter store; observability.
4. **Deep dive (15 min)** — the interviewer picks: batching/scheduling, KV cache, routing, autoscaling, disaggregation, weight loading…
5. **Failure modes & trade-offs (5 min)** — overload, GPU failure, cold start, noisy tenants, bad deploys.
6. **Metrics & rollout** — what you'd dashboard and alert on; how you'd ship changes (canary).

::viz inference-stack

## Design 1 — "Serve a chat model at 500 QPS with p99 TTFT < 1 s and p99 TPOT < 50 ms"

**Clarify:** Llama-3-8B-class model, 1.5K input / 300 output tokens average, 3× peak-to-trough daily, single region, cost matters.

**Napkin:** prefill 500 × 1.5K = 750K tok/s × 16 GFLOP ≈ 12 PFLOP/s → ~24 H100s of prefill; decode 500 × 300 = 150K tok/s at ~5K/GPU → ~30 H100s. ~55 GPUs of work at peak; plan ~70 with headroom, scaling down to ~25 at night.

**Architecture:**
- **Gateway:** auth, per-tenant rate limits and token budgets, request validation (max context), streaming SSE.
- **Router:** **prefix-cache-aware** load balancing (system prompts are shared → big TTFT win), falling back to least-outstanding-tokens.
- **Engine replicas:** vLLM or SGLang, TP=1, continuous batching + **chunked prefill** (protects TPOT from long prompts), FP8 weights and KV (M14) to double KV capacity; CUDA graphs on.
- **Autoscaler:** scale on **queue depth / KV-cache utilization**, not CPU; min replicas sized for the trough; pre-pulled images and weights on local NVMe or a peer cache to cut cold start (M18).
- **Optional:** disaggregate prefill/decode (M17) if long prompts make TTFT and TPOT fight each other.

**Deep-dive talking points:** why chunked prefill; how the scheduler admits requests when KV is full (preemption vs queueing); why p99 TTFT is dominated by **queueing** near saturation (M18 queueing viz); goodput = requests meeting *both* SLOs.

**Failure modes:** traffic spike → queue grows → shed load with 429s per tenant before latency collapses; replica OOM → cap `max_num_seqs`/context; bad model deploy → canary with automatic rollback on SLO + quality alarms.

**Dashboards:** TTFT/TPOT p50/p99, queue depth, KV utilization, prefix-cache hit rate, batch size, GPU SM utilization vs memory utilization, tokens/s, \$/1M tokens.

## Design 2 — "Design a KV-cache-aware router"

**Why:** a request whose prefix is already cached on replica R costs almost no prefill on R but full prefill elsewhere. Random routing wastes that cache.

**Core idea:** maintain an approximate map `prefix-block-hash → replicas that hold it`. For a new request, hash its prompt in block-sized chunks (e.g. 16 tokens, chained hashes like vLLM's block hashing), find the replica with the **longest cached prefix**, and trade it off against load:

$$
\text{score}(r) = w_1 \cdot \text{cached\_prefix\_tokens}(r) - w_2 \cdot \text{queued\_tokens}(r) - w_3 \cdot \text{kv\_utilization}(r)
$$

**Design choices to discuss:**
- **Where does cache state come from?** Engines publish KV events (block stored / evicted) — this is what NVIDIA Dynamo and llm-d do — vs. the router *predicting* cache contents from its own routing history (cheaper, drifts under eviction).
- **Consistency:** eventual is fine — a wrong guess only costs a prefill, never correctness.
- **Hot prefixes:** one very popular system prompt shouldn't pin all traffic to one replica → replicate hot prefixes, cap per-replica share.
- **Scale:** a radix tree / hash map per replica in the router; shard routers by tenant; keep it off the critical path (< 1 ms).
- **Evaluation:** cache hit rate, prefill tokens saved, TTFT p50/p99 vs. round-robin under a replayed production trace.

::viz load-balancing

> [!REAL] It's in production
> Dynamo's KV-aware router and llm-d's prefix-cache-aware scheduling implement exactly this; Baseten reported large TTFT/TPOT improvements from Dynamo's KV-aware routing on a large coding model. Name them — it shows you read beyond papers.

## Design 3 — "Design the rollout infrastructure for GRPO training of a 32B model"

**Clarify:** 8 samples per prompt, 512 prompts per step, responses up to 16K tokens (long tail!), reward = unit tests in sandboxes, trainer on 64 GPUs (FSDP).

**Key facts to state:** rollouts (inference) dominate wall-clock; response lengths are long-tailed so synchronous batches wait for the slowest sample; policy weights change every step.

**Architecture:**
- **Rollout fleet:** vLLM/SGLang engines (TP=2–4 each) behind a scheduler that streams prompts in and completions out; continuous batching keeps GPUs busy despite the long tail.
- **Reward service:** sandboxed executors (containers / gVisor / Firecracker), autoscaled on CPU, with timeouts; results keyed by sample id.
- **Trainer:** FSDP/Megatron consumes completed groups, computes group-normalized advantages, updates.
- **Weight sync:** after each step, broadcast new weights trainer → engines (NCCL/RDMA over a dedicated process group, or via shared storage for simplicity); engines pause, load, **flush prefix cache**, resume.
- **Async option:** let rollouts use weights up to *k* steps stale (off-policy by ≤ k) to overlap generation with training; correct with importance ratios / clipping (M23).

**Trade-offs to discuss:** colocated (train and generate on the same GPUs, time-sliced; simpler, idle memory) vs. disaggregated (separate pools; better utilization, needs fast weight transfer); sync vs. async; how to cap the long tail (max length, partial rollouts); numerical mismatch between inference and training kernels (log-prob drift).

**Metrics:** rollout tokens/s, GPU idle % on each side, weight-sync time, staleness, reward curve, KL to reference.

::viz rl-systems-timeline

- [ ] Wrote Design 1 as a 2-page doc with a diagram and a napkin table
- [ ] Wrote Design 2 or 3 as a 2-page doc (these two docs are part of the challenge)
- [ ] Did one **mock interview** out loud (a friend, or record yourself) for 45 minutes on a design you haven't written
- [ ] Prepared 30-second answers for "why chunked prefill?", "why disaggregate?", "why not always use the biggest batch?"
      */}),
      resources: [
        { title: "Aleksa Gordić — Inside vLLM", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "the engine internals you'll be asked to reason about" },
        { title: "ombharatiya/ai-system-design-guide — interview prep", url: "https://github.com/ombharatiya/ai-system-design-guide/blob/main/00-interview-prep/06-job-market-trends-2026.md", type: "article", note: "community-collected AI system-design prompts and process notes" },
        { title: "NVIDIA Dynamo", url: "https://github.com/ai-dynamo/dynamo", type: "repo", note: "KV-aware routing, disaggregation, SLA-based planner" },
        { title: "llm-d", url: "https://github.com/llm-d/llm-d", type: "repo", note: "Kubernetes-native distributed inference with prefix-aware scheduling" },
      ],
    },
    {
      id: "coding-rounds",
      title: "Coding rounds: top-p sampler, block manager with LRU prefix cache, a kernel",
      kind: "build",
      minutes: 180,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Inference coding rounds are rarely LeetCode puzzles; they're **"implement a piece of an engine, cleanly, with tests, under time pressure"**. Practice these three until each takes < 30 minutes.

## 1. Sampler: temperature, top-k, top-p (batched)

```python
import torch

def sample(logits: torch.Tensor, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """logits: (B, V) -> token ids (B,)"""
    if temperature == 0:
        return logits.argmax(dim=-1)                       # greedy
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]   # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) > top_p                     # mass BEFORE this token already >= p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

# tests you should write in the interview
g = torch.Generator().manual_seed(0)
x = torch.tensor([[2.0, 1.0, 0.1, -1.0]])
assert sample(x, temperature=0).item() == 0
assert all(sample(x, top_k=1, generator=g).item() == 0 for _ in range(20))
assert all(sample(x, top_p=0.5, generator=g).item() == 0 for _ in range(20))   # p(token0) ≈ 0.64 ≥ 0.5
assert set(sample(x.repeat(2000, 1), top_p=0.9, generator=g).tolist()) <= {0, 1, 2}
```

**Follow-ups they'll ask:** why `(cum - probs) > p` (always keeps at least one token); per-request parameters in one batch (make `temperature`/`top_p` tensors of shape `(B, 1)`); min-p; repetition penalty; why sorting the full vocab is slow on GPU and how engines avoid it (top-k first, or specialized kernels as in FlashInfer / vLLM's sampler).

::viz softmax-temp

## 2. KV block manager with reference counts and an LRU prefix cache

The data structure at the heart of vLLM (M08). Blocks hold `block_size` tokens; sequences own a **block table**; full blocks are hashed so identical prefixes are shared; freed blocks stay cached until evicted **LRU**.

```python
from collections import OrderedDict

class OutOfBlocks(Exception): pass

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.bs = block_size
        self.free = list(range(num_blocks))        # never-used or evicted blocks
        self.ref = [0] * num_blocks
        self.hash_of = {}                          # block -> prefix hash (full blocks only)
        self.cached = {}                           # prefix hash -> block
        self.evictable = OrderedDict()             # ref==0 but cached, LRU order
        self.tables = {}                           # seq_id -> [block ids]
        self.lens = {}                             # seq_id -> #tokens

    def _grab(self) -> int:
        if self.free:
            return self.free.pop()
        if self.evictable:                         # evict least-recently-used cached block
            b, _ = self.evictable.popitem(last=False)
            del self.cached[self.hash_of.pop(b)]
            return b
        raise OutOfBlocks()

    def allocate(self, seq_id, tokens: list[int]):
        """Allocate blocks for a new prompt, reusing cached full-block prefixes."""
        table, h = [], None
        for start in range(0, len(tokens), self.bs):
            chunk = tuple(tokens[start:start + self.bs])
            if len(chunk) == self.bs:
                h = hash((h, chunk))               # chained hash = whole prefix identity
                b = self.cached.get(h)
                if b is not None:                  # prefix-cache hit
                    self.evictable.pop(b, None)
                    self.ref[b] += 1
                    table.append(b)
                    continue
            b = self._grab()
            self.ref[b] = 1
            if len(chunk) == self.bs:
                self.hash_of[b], self.cached[h] = h, b
            table.append(b)
        self.tables[seq_id], self.lens[seq_id] = table, len(tokens)

    def append_slot(self, seq_id):
        """Reserve room for one more generated token (call before each decode step)."""
        if self.lens[seq_id] % self.bs == 0:        # last block is full -> need a new one
            b = self._grab(); self.ref[b] = 1
            self.tables[seq_id].append(b)
        self.lens[seq_id] += 1

    def free_seq(self, seq_id):
        for b in self.tables.pop(seq_id):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if b in self.hash_of:
                    self.evictable[b] = True       # keep for reuse, evict later (LRU)
                else:
                    self.free.append(b)
        del self.lens[seq_id]

    def num_free(self): return len(self.free) + len(self.evictable)

# tests
bm = BlockManager(num_blocks=8, block_size=4)
bm.allocate("a", list(range(10)))                 # 3 blocks (2 full + 1 partial)
bm.allocate("b", list(range(8)) + [99])           # shares the 2 full blocks with "a"
assert bm.tables["a"][:2] == bm.tables["b"][:2] and bm.num_free() == 4
bm.free_seq("a"); bm.free_seq("b")
assert bm.num_free() == 8                         # 2 cached-evictable + 6 free
bm.allocate("c", list(range(8)))                  # full prefix hit, no new blocks
assert bm.num_free() == 6
```

**Follow-ups:** copy-on-write when two sequences share a *partial* last block (parallel sampling / beam search); what to do on `OutOfBlocks` (preempt the newest sequence: free its blocks and recompute later, or swap to CPU); hash collisions (vLLM uses strong hashes); why only **full** blocks are cached.

## 3. A kernel (for kernel/runtime roles)

Expect: write a row-wise softmax, a reduction, a vector add with correct indexing, or discuss how you'd optimize a matmul. Know this one cold (Triton, from M12):

```python
import torch, triton, triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                               # one program per row
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                            # numerical stability
    num = tl.exp(x)
    tl.store(out_ptr + row * out_stride + offs, num / tl.sum(num, axis=0), mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
```

Be ready to discuss: why this is memory-bound (read once, write once), what happens when `n_cols` doesn't fit in one block (→ online softmax, M12), coalescing, occupancy.

> [!TIP] Practice platforms
> [LeetGPU](https://leetgpu.com/) and [Tensara](https://tensara.org/) have GPU-kernel problems you can run in the browser — a good daily warm-up. For the Python parts, time-box yourself: 25 minutes, tests included.

- [ ] Implemented the sampler from scratch (no peeking) with 4 tests, in < 30 min
- [ ] Implemented the block manager with prefix caching and preemption-on-OOM, in < 45 min
- [ ] Wrote the Triton softmax from memory on Colab and verified vs `torch.softmax`
- [ ] Solved 5 problems on LeetGPU or Tensara
      */}),
      resources: [
        { title: "LeetGPU", url: "https://leetgpu.com/", type: "practice", note: "browser-run CUDA/Triton problems" },
        { title: "Tensara", url: "https://tensara.org/", type: "practice", note: "GPU kernel challenges with leaderboards" },
        { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "a compact engine whose block manager and scheduler are interview-sized" },
      ],
    },
    {
      id: "debugging",
      title: "Debugging scenarios: p99 spike, OOM, low GPU utilization, hangs",
      kind: "concept",
      minutes: 90,
      md: MD(function () {/*
Debugging questions test whether you have a **mental model of where time and memory go**. Answer with a structure: *symptom → hypotheses ranked by likelihood → the one metric that distinguishes them → fix*. Postings explicitly list tail-latency (p95/p99) reduction, and one career guide calls failing to distinguish **GPU memory utilization from SM utilization** a no-hire signal.

## Scenario 1 — "p99 TTFT tripled since yesterday; p50 is fine"

| Hypothesis | Check | Fix |
|---|---|---|
| **Queueing** near saturation (traffic grew) | request rate vs capacity, queue depth, waiting requests | scale out; autoscale on queue depth; admission control |
| **Long prompts** arriving (new customer / feature) | input-length histogram over time | chunked prefill; separate pool or disaggregate long-context traffic |
| **Prefix-cache hit rate dropped** (system prompt changed, router change) | cache hit rate metric | fix routing affinity; stabilize prompts |
| **Preemptions** (KV full → recompute) | preemption counter, KV utilization ≈ 100% | lower max concurrency, FP8 KV, more replicas |
| **A bad node** (thermal throttling, degraded NVLink) | per-replica latency breakdown | drain the node; health checks |

The key move: **break p99 down by replica and by request size** before theorizing.

::viz percentiles

## Scenario 2 — "CUDA out of memory after we raised max context to 128K"

- Weights fit, so it's **KV cache or activations**. Engines pre-allocate KV from a memory fraction; the OOM is usually **prefill activations** for a huge prompt (attention workspace, logits of size seq × vocab if you're returning all logprobs) or a CUDA-graph capture size.
- Fixes: chunked prefill (bounded activation memory), lower `max_num_batched_tokens`, reduce `gpu_memory_utilization` to leave headroom, don't compute full-sequence logits, FP8 KV, TP=2.
- Know the formula: **free memory for KV = total × utilization − weights − activation peak − graphs/overhead** (Drill 2).

## Scenario 3 — "GPU utilization shows 100% but throughput is low" / "GPU util is 30%"

- `nvidia-smi` "utilization" = % of time **any** kernel was running — not how busy the SMs are. 100% can mean one tiny kernel at a time. Look at **SM activity / occupancy / tensor-core activity** (DCGM, Nsight Systems).
- Low throughput at small batch = **memory-bound decode** (expected!) or **CPU overhead** (Python scheduler, tokenization, sampling, kernel launch gaps → CUDA graphs, async scheduling; M13).
- Low utilization with a queue building up = the **scheduler** isn't batching (max batch too small, KV full, a lock in the API server, detokenization on the hot path).

## Scenario 4 — "Our multi-GPU job hangs with no error"

Classic NCCL hang: ranks disagree on the collective sequence (one rank took a different code path), a rank died, or network/topology issues. Steps: `NCCL_DEBUG=INFO`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`/timeouts, py-spy dump on every rank to find who is stuck where, check that all ranks run the same collectives with the same shapes, test NCCL alone (nccl-tests) to isolate hardware.

## Scenario 5 — "After upgrading the engine, quality dropped slightly"

Different kernels/precision (FP8 path now enabled?), a changed default (sampling params, chat template, max tokens), tokenizer mismatch, or batch-dependent numerics. Run the eval suite at temperature 0 on both versions; diff outputs; bisect config flags. This is why eval gates exist (M24).

- [ ] Wrote a one-page runbook for each scenario in your own words
- [ ] Reproduced Scenario 1 locally: overload your M19 setup and watch p99 TTFT vs queue depth
- [ ] Reproduced Scenario 3 on Colab: profile a batch-1 decode with Nsight Systems or `torch.profiler` and find the launch gaps
      */}),
      resources: [
        { title: "LecoMV/alexmayhew.dev — SRE → AI infrastructure career research", url: "https://github.com/LecoMV/alexmayhew.dev/blob/main/docs/research/sre-to-ai-infrastructure-career-2026.md", type: "article", note: "debugging-style interview questions and hire/no-hire signals (secondary source)" },
        { title: "Stas Bekman — Machine Learning Engineering Open Book", url: "https://github.com/stas00/ml-engineering", type: "book", note: "practical debugging of GPUs, NCCL, hangs, OOMs" },
      ],
    },
    {
      id: "portfolio",
      title: "Portfolio: present your flagship projects so a hiring manager gets it in 60 seconds",
      kind: "build",
      minutes: 240,
      md: MD(function () {/*
Hiring guidance from inference teams is blunt: put **independent work, write-ups and open-source contributions at the top** of your resume. Postings ask for "work you can point to" and "a story about boosting GPU performance". Your course projects are exactly that — if they're packaged well.

## The five flagship projects

| Project | Module | The headline number to show | The chart |
|---|---|---|---|
| **mini-vLLM** | M10 | throughput vs HF `generate` at N concurrent users | tok/s vs concurrency; p99 TPOT vs concurrency |
| **Kernels** (matmul, softmax, flash-attn) | M11–M12 | % of cuBLAS / speed-up vs PyTorch | roofline plot with your kernels as points |
| **Benchmark report** | M19 | \$/1M tokens and max QPS under a p99 SLO, engine × precision | throughput–latency curves |
| **GRPO run + rollout system** | M22–M23 | reward/accuracy gain; rollout vs train time split | reward curve; timeline of GPU idle |
| **Continual-learning report** | M24 | BWT naive vs replay vs EWC; LLM general-score drop | accuracy-matrix heatmaps |

## README template (each project)

```text
# mini-vLLM — a 1,500-line LLM inference engine with paged KV cache and continuous batching

**Result:** 7.4× the throughput of HF generate at 32 concurrent users on a T4 (Qwen2.5-0.5B), p99 TPOT 38 ms.
[chart: throughput vs concurrency]

## What it does          (3 bullets)
## Architecture          (diagram: API → scheduler → block manager → model runner)
## Key design decisions  (why block size 16, why FCFS + preemption, what I'd do next)
## Benchmarks            (setup, command to reproduce, table, caveats)
## What I learned        (2–3 honest bullets, incl. a bug that took a day)
```

Rules: **number in the first line**, a chart above the fold, a one-command reproduction, honest caveats (hardware, model size). A reviewer spends ~60 seconds.

## One blog post (the multiplier)

Write **one** deep post from your best project, e.g. *"I built a paged-attention engine from scratch — here's where the time actually goes"*. Outline: the problem → a surprising measurement → how you found it (profiler screenshots) → the fix → before/after numbers → what production engines do differently. Post it on your site/GitHub and share it in communities (next lessons). This is also your best interview story.

## Resume bullets (quantified, archetype-tuned)

- *Built a paged-KV-cache inference engine with continuous batching and prefix caching (Python/PyTorch); 7.4× throughput vs. HF baseline at 32 concurrent users; OpenAI-compatible streaming API.*
- *Wrote Triton fused-softmax and FlashAttention-style kernels; 2.1× faster than eager PyTorch at 8K sequence length on A100; analysis with Nsight Compute.*
- *Benchmarked vLLM vs SGLang with FP16/FP8 on H100 under open-loop load; identified max QPS at p99 TTFT < 1 s; cost model in \$/1M tokens.*
- Your backend experience counts: *"Ran services at N QPS with p99 SLOs"* is directly relevant to platform roles — keep those bullets and connect them.

(Replace every number with your real ones — never inflate. Interviewers will ask how you measured it.)

- [ ] Created the `inference-portfolio` repo with the top-level README (who you are, 5 projects, 5 numbers)
- [ ] Each flagship project has a README following the template, with at least one chart
- [ ] Every benchmark has a one-command reproduction and a hardware line
- [ ] Wrote and published one technical blog post
- [ ] Rewrote your resume: projects + OSS section at the top, 3–5 quantified bullets tuned to your primary archetype
      */}),
    },
    {
      id: "oss-contribution",
      title: "Open source: read the codebases and land your first PR in vLLM / SGLang / llm-d / Dynamo",
      kind: "lab",
      minutes: 600,
      runsOn: ["mac", "colab", "cloud"],
      md: MD(function () {/*
Several postings list open-source contributions to inference engines or kernel libraries as a standout qualification, and some (e.g. Hugging Face) treat a public record of merged PRs as a filter. It's also the fastest way to learn how production engines really work.

## Where to find a first issue

| Project | Good first issues | Contributing guide | Good for archetype |
|---|---|---|---|
| **vLLM** | [label: good first issue](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22) | [docs](https://docs.vllm.ai/en/latest/contributing/index.html) | B, A, D |
| **SGLang** | [label: good first issue](https://github.com/sgl-project/sglang/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22) | [guide](https://docs.sglang.ai/developer_guide/contribution_guide.html) | B, D |
| **llm-d** | [label: good first issue](https://github.com/llm-d/llm-d/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22) | [CONTRIBUTING.md](https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md) | C |
| **NVIDIA Dynamo** | [label: good first issue](https://github.com/ai-dynamo/dynamo/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22) | [CONTRIBUTING.md](https://github.com/ai-dynamo/dynamo/blob/main/CONTRIBUTING.md) | B, C |
| **mlx-lm** (Mac-friendly) | [label: good first issue](https://github.com/ml-explore/mlx-lm/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22) | repo README | warm-up |

Good first contributions besides labeled issues: **reproduce a bug report** and post a minimal repro; add a **missing test**; fix **docs** that are wrong (you'll find some while doing this course); add a **benchmark** or a model config; improve an **error message** you hit.

## How to read an engine codebase (vLLM V1 as the example)

Follow **one request** end to end — the same path you built in M10:

1. **Entry:** `vllm/entrypoints/openai/api_server.py` — HTTP → engine client.
2. **Engine core loop:** `vllm/v1/engine/core.py` — `step()`: schedule → execute → update.
3. **Scheduler:** `vllm/v1/core/sched/scheduler.py` — token budget, running/waiting queues, preemption, chunked prefill.
4. **KV cache:** `vllm/v1/core/kv_cache_manager.py` and `vllm/v1/core/block_pool.py` — your block manager from the coding lesson, production-grade (prefix hashing, LRU eviction).
5. **Model runner:** `vllm/v1/worker/gpu_model_runner.py` — input preparation, CUDA graphs, attention metadata.
6. **Sampler:** `vllm/v1/sample/sampler.py` — compare with your sampler.

For SGLang, the equivalents are `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/mem_cache/radix_cache.py` (RadixAttention — M08's radix tree) and `python/sglang/srt/model_executor/model_runner.py`.

> [!TIP] Reading technique
> Run the engine with a tiny model under a debugger (or add logging), set breakpoints at each file above, send one request, and write a 1-page "life of a request" note with file:line references. Then read the tests for that component — tests are the best documentation. Aleksa Gordić's *Inside vLLM* post is the ideal companion.

## A 4-week first-PR plan

| Week | Do |
|---|---|
| 1 | Build from source; run the test suite for one component; join the project Slack/Discord; write your "life of a request" note |
| 2 | Pick 3 candidate issues; comment on one ("I'd like to take this; my plan is…"); reproduce it |
| 3 | Implement with tests; follow the contributing guide (pre-commit, DCO sign-off if required, PR title tags like `[Core]`, `[Kernel]`, `[Doc]` in vLLM) |
| 4 | Respond to review within 24 h; iterate; once merged, pick a slightly bigger second issue in the same area |

> [!WARNING] Etiquette that gets PRs merged
> Small and focused beats big and ambitious. Don't open a PR for an issue someone else claimed. Include before/after numbers for anything performance-related, and the exact command to reproduce. Don't ping maintainers repeatedly; a polite bump after a week is fine.

- [ ] Built vLLM or SGLang (or mlx-lm) from source and ran one component's tests
- [ ] Wrote a "life of a request" note with file references
- [ ] Commented on an issue with a plan, and got a response
- [ ] Opened a PR (link it in your portfolio's `oss.md`)
- [ ] PR merged — or feedback addressed and a second PR opened
      */}),
      resources: [
        { title: "vLLM — Contributing", url: "https://docs.vllm.ai/en/latest/contributing/index.html", type: "docs", note: "setup, tests, PR conventions" },
        { title: "vLLM — Architecture overview", url: "https://docs.vllm.ai/en/latest/design/arch_overview.html", type: "docs", note: "official map of the codebase" },
        { title: "SGLang — Contribution guide", url: "https://docs.sglang.ai/developer_guide/contribution_guide.html", type: "docs", note: "how to set up and submit to SGLang" },
        { title: "mini-sglang", url: "https://github.com/sgl-project/mini-sglang", type: "repo", note: "a compact SGLang to read before the real one" },
      ],
    },
    {
      id: "stay-current",
      title: "Staying current: papers, blogs and communities worth your time",
      kind: "read",
      minutes: 60,
      md: MD(function () {/*
The field moves monthly (Blackwell and FP4, disaggregation going mainstream, MoE expert parallelism, RL infra). You don't need to read everything — you need a **weekly habit** with a few high-signal sources.

## Communities (where the practitioners are)

| Community | Why |
|---|---|
| [GPU MODE Discord](https://discord.gg/gpumode) · [lectures](https://github.com/gpu-mode/lectures) · [YouTube](https://www.youtube.com/@GPUMODE/videos) | the best community for CUDA/Triton/kernels; weekly lectures; kernel competitions |
| [vLLM Slack](https://slack.vllm.ai) · [vLLM meetups](https://docs.vllm.ai/en/latest/community/meetups.html) | talk to maintainers, find issues, see roadmap talks |
| [SGLang docs](https://docs.sglang.ai/) & GitHub discussions | the other major engine's community |
| [GPU MODE resource-stream](https://github.com/gpu-mode/resource-stream) | curated firehose of GPU papers/posts |

## Blogs & newsletters (weekly skim)

- [vLLM blog](https://blog.vllm.ai/) and [LMSYS / SGLang blog](https://lmsys.org/blog/) — engine features with benchmarks (e.g. large-scale expert parallelism, GB200 results).
- [Baseten blog](https://www.baseten.co/blog/) and [Anyscale blog](https://www.anyscale.com/blog) — production inference engineering.
- [PyTorch blog](https://pytorch.org/blog/) — compiler, kernels, distributed.
- [SemiAnalysis](https://semianalysis.com/) — hardware, supply and inference economics (partly paywalled).
- [Lilian Weng's blog](https://lilianweng.github.io/) and [Chip Huyen's blog](https://huyenchip.com/blog/) — deep, well-sourced overviews.

## Papers

- Skim [Hugging Face Daily Papers](https://huggingface.co/papers) and [arXiv cs.DC recent](https://arxiv.org/list/cs.DC/recent); read one systems paper a week **with its code**.
- Venues where inference systems papers land: **OSDI, SOSP, NSDI, MLSys, EuroSys, ASPLOS** (postings list these as differentiators for senior roles).
- A reading method that sticks: for each paper write *problem → key idea → the number that matters → how vLLM/SGLang does it today*.

## Trends to be able to talk about (2025–2026)

1. **Disaggregated prefill/decode** going mainstream (Dynamo, llm-d, SGLang).
2. **KV cache as a distributed resource**: hierarchical offload (GPU → CPU → disk), KV-aware routing.
3. **MoE + expert parallelism** serving (DeepSeek-style), with dedicated kernels.
4. **FP8 → FP4 (NVFP4/MXFP4)** on Blackwell.
5. **RL/post-training infra** as its own discipline (async rollouts, weight sync, sandboxes).
6. **Beyond CUDA**: TPUs, Trainium, AMD ROCm, Cerebras.

- [ ] Joined GPU MODE and vLLM Slack; introduced yourself with a link to your portfolio
- [ ] Set up a weekly 2-hour "reading block" in your calendar
- [ ] Wrote a 5-bullet summary of one recent vLLM/LMSYS blog post
- [ ] For each of the 6 trends, can say one sentence + one project/paper name
      */}),
    },
    {
      id: "job-search-90",
      title: "The 90-day job-search plan",
      kind: "read",
      minutes: 60,
      md: MD(function () {/*
A plan with weekly outputs. Adjust to your pace (10–15 h/week); the order matters more than the dates.

## Days 1–30: package and fill gaps

- [ ] **Week 1:** choose archetypes; gap list from the [skills map](#/x/skills); portfolio repo skeleton + top-level README
- [ ] **Week 2:** polish 2 flagship projects (READMEs, charts, reproduction commands)
- [ ] **Week 3:** polish the remaining projects; draft the blog post
- [ ] **Week 4:** publish the blog post; rewrite resume + LinkedIn/GitHub profile; pick your first OSS issue
- [ ] Napkin drills: all 8, twice

## Days 31–60: prove it publicly, start conversations

- [ ] Open your first OSS PR (and keep iterating)
- [ ] Write both mock system-design docs; do 2 recorded mock interviews
- [ ] Coding: sampler, block manager, a kernel — each under time pressure, twice
- [ ] Build a target list of **30 companies** across tiers: inference platforms (Baseten, Together, Fireworks, Modal…), labs, hardware vendors (NVIDIA, Cerebras…), clouds (CoreWeave…), and companies running big inference fleets in-house
- [ ] **Warm outreach:** 10 people who wrote a blog post or PR you learned from — thank them specifically, share your related project, ask one sharp question. No "can you refer me" in message one.
- [ ] Apply to the first 10 roles (tailor the top 3 resume bullets per archetype)

## Days 61–90: interview loop

- [ ] Apply to 20 more; track everything in a spreadsheet (company, role, archetype, stage, next step, date)
- [ ] Before each interview: read the company's engineering blog and 2 posts about their stack; prepare "why here" with a technical hook
- [ ] After each interview: write down every question within 1 hour; turn misses into drills
- [ ] Second OSS PR merged or in review
- [ ] Negotiate with data: postings in pay-transparency states (CA, NY, WA) often list ranges — use them

## Weekly scorecard

| Metric | Target / week |
|---|---|
| Deep-work hours on portfolio/OSS | 6–8 |
| Applications (after day 45) | 5–8, tailored |
| Outreach messages | 3–5 |
| Mock interviews / drills | 2 |
| Public artifacts (PR, post, README) | 1 |

> [!INTUITION] Why this order works
> Hiring for this role is **evidence-driven**. A merged vLLM PR plus a benchmarked engine and one sharp blog post outweighs a long list of courses. Applying before the evidence exists wastes your best leads — so package first, apply second.

> [!IMPORTANT] How this connects
> This is the end of the map you saw in M00: every layer of the stack now has a project with your name on it. The job is the same loop you've practised 25 times — **measure, explain where the time goes, change the system, measure again** — just with someone else's GPUs.
      */}),
    },
  ],

  challenge: {
    title: "Ship the job-search kit",
    md: MD(function () {/*
Final capstone. Everything public (or shareable) and linked from one README:

1. **Portfolio checklist complete** — the `inference-portfolio` repo with the five flagship projects, each with a headline number, a chart, and a reproduction command.
2. **Two mock system-design write-ups** (2–3 pages each, with a diagram, napkin table, deep dive, failure modes and metrics): Design 1 (serving at a QPS with SLOs) and one of Design 2 (KV-aware router) or Design 3 (RL rollout infra) — or a design prompt you got in a real interview.
3. **One open-source PR** to vLLM, SGLang, llm-d, Dynamo (or another inference/kernel project such as mlx-lm, FlashInfer, Triton) — **merged or open with review activity**.
    */}),
    checklist: [
      "Portfolio README: who you are, primary archetype, 5 projects with 5 headline numbers",
      "Every flagship project has a chart, a hardware line and a one-command reproduction",
      "Two system-design documents written and each rehearsed aloud in a 45-minute mock",
      "All 8 napkin drills done from memory in under 3 minutes each",
      "Sampler and block manager implemented from scratch with tests under time limits",
      "At least one OSS PR merged or open with maintainer review, linked in `oss.md`",
      "90-day plan started: target list of 30 companies and a weekly scorecard",
    ],
    stretch: "Give a 20-minute talk on one of your projects (a GPU MODE community session, a local meetup, or a recorded video) — and turn the questions you get into your next blog post.",
  },

  connects: MD(function () {/*
This module closes the loop opened in **M00**: you started by measuring one model on a laptop and learning the map; you finish with a project at every layer of that map and a plan to get paid to keep going. Keep the [Job skills map](#/x/skills) and the [Glossary](#/x/glossary) handy while interviewing — and revisit M06–M10 and M18–M19 right before loops for runtime/platform roles, M11–M13 for kernel roles, M21–M24 for RL infra.
  */}),

  interview: [
    "Walk me through what happens to a request in vLLM from the HTTP call to the last streamed token.",
    "Estimate the KV cache per token for Llama-3-70B and how many 8K-token sequences fit on 4× H100 with FP8 weights.",
    "Design an LLM serving system for 1,000 QPS with p99 TTFT < 800 ms. Where does your design break first as traffic doubles?",
    "Design a KV-cache-aware router. How do you keep its view of the caches fresh, and what happens with a very hot prefix?",
    "Our p99 latency tripled but p50 is unchanged. How do you debug it?",
    "GPU utilization is 100% but throughput is low. What's going on and how do you prove it?",
    "Implement top-p sampling for a batch where each request has its own temperature and top_p.",
    "Tell me about a performance problem you found and fixed — how did you measure it and what was the before/after?",
  ],

  resources: [
    { title: "Baseten — Inference Engineering (book)", url: "https://www.baseten.co/inference-engineering/", type: "book", note: "the job, described by a company that hires for it" },
    { title: "sizief/llmway — inference career plan", url: "https://github.com/sizief/llmway/blob/main/inference_career_plan.md", type: "article", note: "hands-on project sequence for switching into inference" },
    { title: "LecoMV — SRE to AI infrastructure career research", url: "https://github.com/LecoMV/alexmayhew.dev/blob/main/docs/research/sre-to-ai-infrastructure-career-2026.md", type: "article", note: "portfolio ladder, interview questions, hire signals (secondary)" },
    { title: "ombharatiya/ai-system-design-guide", url: "https://github.com/ombharatiya/ai-system-design-guide", type: "course", note: "AI system-design interview prep" },
    { title: "Aleksa Gordić — Inside vLLM", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "read before any runtime-role interview" },
    { title: "vLLM — Contributing", url: "https://docs.vllm.ai/en/latest/contributing/index.html", type: "docs", note: "your first PR starts here" },
    { title: "GPU MODE Discord", url: "https://discord.gg/gpumode", type: "practice", note: "the kernel/perf community; lectures and competitions" },
    { title: "vLLM Slack", url: "https://slack.vllm.ai", type: "practice", note: "maintainers, contributors, roadmap discussions" },
    { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "napkin math refresher" },
    { title: "How To Scale Your Model", url: "https://jax-ml.github.io/scaling-book/", type: "book", note: "systems math for training and inference at scale" },
    { title: "SemiAnalysis", url: "https://semianalysis.com/", type: "article", note: "hardware and inference economics context for interviews" },
  ],
});
