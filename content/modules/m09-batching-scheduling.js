Course.module({
  id: "m09-batching-scheduling",
  title: "Batching & scheduling: continuous batching, chunked prefill",
  short: "Batching & scheduling",
  tagline: "Write the scheduler at the heart of every inference engine: many requests share one forward pass, joining and leaving every step — several times the throughput of one-at-a-time generation, with plots to prove it.",
  hours: 16,
  runsOn: ["mac", "colab"],
  tags: ["continuous-batching", "chunked-prefill", "scheduler", "orca", "sarathi", "littles-law", "preemption"],

  goal: MD(function () {/*
By the end you have a `Scheduler` (waiting/running queues, a per-step **token budget**, chunked prefill, preemption) driving your **m08 `BlockManager`** and a real Qwen2.5-0.5B forward pass on your Mac. You replay a synthetic traffic trace and print a table like this (illustrative numbers from an M2 Pro — yours will differ):

```text
trace: 64 requests, Poisson 2 req/s, prompts 64–512 tok, outputs 32–128 tok   model: Qwen2.5-0.5B fp16 (MPS)
config                       out tok/s   TTFT p50   TTFT p99   ITL p50   ITL p99
sequential (HF generate)          29       41.2 s     83.0 s    34 ms     37 ms
continuous, 32 seqs              168        0.6 s      2.4 s    61 ms    390 ms
continuous + chunked (256)       163        0.7 s      2.6 s    60 ms    150 ms
```

…plus three plots (throughput, TTFT p99 and ITL p99 against arrival rate) and one paragraph explaining something that surprised you. That table is the whole story of LLM serving in miniature: **batching buys throughput, scheduling decides who pays for it in latency.**

::viz batching-sim
  */}),
  demo: { viz: "batching-sim", params: {} },

  why: MD(function () {/*
The scheduler is where an inference engine's economics are decided. The same GPU and the same model can serve 1 user or 100, depending on how well you pack work into each forward pass — that's a 10–50× swing in **\$ per million tokens**. Every serious engine (vLLM, SGLang, TensorRT-LLM "in-flight batching", TGI) is built around an iteration-level scheduler, and "tune `max_num_batched_tokens` / `max_num_seqs` for this SLO" is a weekly task on inference teams. Interviewers love this topic because it mixes systems (queues, memory, preemption) with performance modelling (roofline, Little's law). It is also the core of the Baseten book's production chapter: *set the autoscaler's concurrency target equal to the replica batch size* (Inference Engineering (Baseten) ch. 7.2.1).
  */}),

  prereqs: [
    {
      title: "Your m08 BlockManager, in one screen",
      skipIf: "you finished m08 and your BlockManager can answer “can this sequence grow to N tokens?”",
      md: MD(function () {/*
The scheduler only needs five things from the KV-cache manager. If your m08 class uses other names, write a 10-line adapter — don't rewrite it. If you skipped m08, this minimal version (no prefix caching, no ref counts) is enough for this module:

```python
# block_manager.py
import math

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free_blocks = list(range(num_blocks - 1, -1, -1))   # stack of free block ids
        self.tables: dict[str, list[int]] = {}                    # seq_id -> block table

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def blocks_needed(self, seq_id: str, num_tokens: int) -> int:
        have = len(self.tables.get(seq_id, ()))
        return max(0, math.ceil(num_tokens / self.block_size) - have)

    def can_grow(self, seq_id: str, num_tokens: int, reserve: int = 0) -> bool:
        return self.blocks_needed(seq_id, num_tokens) + reserve <= len(self.free_blocks)

    def grow(self, seq_id: str, num_tokens: int) -> None:
        n = self.blocks_needed(seq_id, num_tokens)
        assert n <= len(self.free_blocks), "call can_grow() first"
        table = self.tables.setdefault(seq_id, [])
        for _ in range(n):
            table.append(self.free_blocks.pop())

    def block_table(self, seq_id: str) -> list[int]:
        return self.tables.get(seq_id, [])

    def free(self, seq_id: str) -> None:
        self.free_blocks.extend(reversed(self.tables.pop(seq_id, [])))
```

Physical slot of logical token position `p` for a sequence: `table[p // block_size] * block_size + p % block_size`. That one line is what "paged" means.
      */}),
    },
    {
      title: "Rates, averages and random arrivals (5-minute version)",
      skipIf: "you know what a Poisson process and a percentile are",
      math: true,
      md: MD(function () {/*
- A **rate** is "how many per second". $\lambda = 4$ req/s means on average 4 requests arrive each second, i.e. one every $\tfrac{1}{\lambda} = 0.25$ s *on average*.
- Real users don't arrive on a metronome. The standard model is a **Poisson process**: gaps between arrivals are random, drawn from an **exponential distribution** with mean $\tfrac{1}{\lambda}$. Many short gaps, a few long ones — so requests come in **bursts**. In Python: `random.expovariate(rate)`.
- A **percentile**: p99 latency = the value 99% of requests beat. Averages hide the unlucky users; p99 is what they complain about (you met this in m00/m06).
- **Open loop vs closed loop** load: open loop sends at a fixed rate no matter what (like real traffic — if the server is slow, a queue builds). Closed loop keeps exactly N requests in flight (each user waits for their answer before asking again). `vllm bench serve --request-rate` is open loop; `--max-concurrency` caps it into a closed-ish loop.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: 1 vs 32 concurrent requests on a real engine",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Before writing a scheduler, feel what it does. You'll fire **N concurrent streaming requests** at a real engine and watch two numbers: **total tokens/s** (throughput, what the operator pays for) and **per-user tokens/s** (latency, what the user feels).

## The client (works against any OpenAI-compatible server)

```python
# concurrency.py — usage: python concurrency.py URL MODEL
import asyncio, json, statistics, sys, time
import httpx

URL, MODEL = sys.argv[1], sys.argv[2]
PROMPT = "Write a long, detailed story about a lighthouse keeper who finds a map."

async def one(client, max_tokens=128):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    t0, stamps = time.perf_counter(), []
    async with client.stream("POST", URL, json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    return t0, stamps

async def run(conc, total=64):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=None) as client:
        async def guarded():
            async with sem:
                return await one(client)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[guarded() for _ in range(total)])
        wall = time.perf_counter() - t0
    chunks = sum(len(s) for _, s in res)
    ttft = [s[0] - t for t, s in res if s]
    itl = [b - a for _, s in res for a, b in zip(s, s[1:])]
    print(f"conc={conc:3d}  total≈{chunks / wall:7.1f} tok/s  "
          f"per-user≈{1 / statistics.median(itl):6.1f} tok/s  TTFT p50={statistics.median(ttft):.2f}s")

for c in [1, 2, 4, 8, 16, 32]:
    asyncio.run(run(c))
```

(Each streamed chunk is ~one token for these servers; good enough for a demo. You'll count tokens exactly in the lab.)

## Option A — on your Mac with Ollama (llama.cpp has continuous batching via "parallel slots")

```bash
OLLAMA_NUM_PARALLEL=1  ollama serve        # terminal 1, run the client, then Ctrl-C and restart with:
OLLAMA_NUM_PARALLEL=16 ollama serve
python concurrency.py http://localhost:11434/v1/completions qwen2.5:0.5b
```

## Option B — on Colab (T4) with vLLM, the reference engine

```bash
pip install vllm
nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 \
      --gpu-memory-utilization 0.85 > vllm.log 2>&1 &
python concurrency.py http://localhost:8000/v1/completions Qwen/Qwen2.5-0.5B-Instruct
# vLLM's own benchmark, closed loop at 1 vs 32 in flight:
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random \
  --random-input-len 256 --random-output-len 128 --num-prompts 200 --max-concurrency 1
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random \
  --random-input-len 256 --random-output-len 128 --num-prompts 200 --max-concurrency 32
```

(`--dtype half` because the T4 has no BF16. If your vLLM version misbehaves on a T4, switch the Colab runtime to an L4.)

## What you should see

| concurrency | total tok/s | per-user tok/s | TTFT |
|---|---|---|---|
| 1 | $x$ | $x$ | lowest |
| 8 | ~5–7 $x$ | slightly lower | a bit higher |
| 32 | ~10–20 $x$ on vLLM | noticeably lower | higher |

Total throughput grows almost linearly at first while per-user speed barely moves. That's m00's law in action: decode is limited by **reading the weights**, and one weight read can serve 32 users as cheaply as 1. Eventually the curve bends — compute, KV-cache reads or memory capacity catch up.

::viz latency-throughput

> [!INTUITION] The two numbers are in tension
> Each extra request in the batch makes *every* step slightly slower (more KV to read, more FLOPs) but adds a whole user's worth of tokens. Operators want the right side of the curve, users want the left. A scheduler is the machine that picks the point — and keeps it stable as traffic changes.

- [ ] Ran `concurrency.py` at 1…32 against Ollama or vLLM and saved the output
- [ ] Plotted total tok/s and per-user tok/s against concurrency on one chart (two y-axes)
- [ ] Wrote down the concurrency where per-user speed drops below 20 tok/s (≈ comfortable reading speed)
      */}),
      resources: [
        { title: "vLLM — Optimization and Tuning (chunked prefill, preemption)", url: "https://docs.vllm.ai/en/latest/configuration/optimization.html", type: "docs", note: "the knobs you'll re-implement in this module" },
        { title: "Anyscale — How continuous batching enables 23x throughput", url: "https://www.anyscale.com/blog/continuous-batching-llm-inference", type: "article", note: "the classic benchmark post that made continuous batching famous" },
      ],
    },
    {
      id: "batching-strategies",
      title: "Static vs dynamic vs continuous batching (Orca)",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
## Three ways to fill a batch

::viz batching-sim

| Strategy | When does a batch start? | When can a new request join? | Waste | Who uses it |
|---|---|---|---|---|
| **Static** | when `B` requests have queued | only after the **whole batch** finishes | early finishers sit idle until the longest one ends; early arrivals wait to fill the batch | offline jobs, naive `model.generate(batch)` |
| **Dynamic** | when `B` requests queued **or** a timeout (e.g. 10–15 ms) fires | after the batch finishes | same end-of-batch waste; timeout bounds the wait | classic model servers (Triton), fixed-size models: embeddings, ASR encoders, audio decoders |
| **Continuous** (in-flight, iteration-level) | every model step | **every step**; finished requests leave every step | almost none — slots are refilled immediately | vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp |

For fixed-cost models (an embedding model, an image classifier) dynamic batching is fine: every request costs the same. LLMs break that assumption: one request wants 5 output tokens, another 2,000. With static batching the short one is **held hostage**.

**Worked example.** A static batch of 4 requests needs 20, 50, 200 and 30 output tokens. The batch runs for 200 decode steps. Useful token-steps: $20+50+200+30 = 300$. Paid-for token-steps: $4 \times 200 = 800$. **Utilization 37.5%** — and the 20-token user waited for 200 steps to get their answer back (if not streaming).

## Orca: schedule iterations, not requests

The Orca paper (OSDI '22) made two moves that every engine now copies:

1. **Iteration-level scheduling.** The scheduler runs *between every forward pass*. After each step it removes finished sequences and admits new ones. A request's latency now depends on its own length, not on its batch-mates'.
2. **Selective batching.** Requests in a step have different lengths, so you can't stack them into a rectangular `[B, T]` tensor. Orca's trick: for everything except attention (QKV projections, MLP, norms — the parts that read the weights), **flatten all tokens of all requests into one `[total_tokens, hidden]` matrix**. Only attention is done per-request, because each request attends to its own history. You'll build exactly this "packed batch" in lesson *variable-lengths* and in the lab.

Orca reported a **36.9× throughput improvement** over NVIDIA FasterTransformer at the same latency on GPT-3 175B. vLLM (2023) added paged KV memory on top, so far more sequences fit in the batch; together they are the "23×" of the Anyscale post.

> [!INTUITION] One sentence
> Continuous batching turns "a batch" from a *group of requests* into a *stream of forward passes*, each containing whatever work is ready right now.

## What goes into one step?

In a modern engine one forward pass may contain, for different requests:

```text
step 1041:  [req 7: decode 1 tok] [req 9: decode 1 tok] [req 12: prefill tokens 512..1023] [req 13: decode 1 tok]
step 1042:  [req 7: decode]  [req 9: decode — hits EOS, leaves] [req 12: prefill 1024..1300 → first token!] [req 13: decode] [req 14: prefill 0..211 (new)]
```

Prefill tokens and decode tokens are processed **together** (vLLM V1 calls this a unified token budget — lesson *deep-vllm-sglang*). The scheduler's job each step is to pick that list.

> [!REAL] Terminology you'll hear
> TensorRT-LLM says **in-flight batching**, Hugging Face TGI says **continuous batching**, Orca says **iteration-level scheduling**. Same idea. The Baseten book (ch. 7.2.1) calls it the lowest-latency-penalty option and uses it in all three engines it deploys.

- [ ] Compute utilization for a static batch with outputs [10, 10, 10, 500]. What does continuous batching do for the three short ones?
- [ ] In `batching-sim`, find settings where static batching's p50 latency is >5× continuous batching's
- [ ] Explain "selective batching" to a colleague in two sentences
      */}),
      resources: [
        { title: "Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22)", url: "https://www.usenix.org/conference/osdi22/presentation/yu", type: "paper", note: "iteration-level scheduling + selective batching; read §3–4" },
      ],
    },
    {
      id: "throughput-latency-math",
      title: "Math: the throughput–latency trade-off, Little's law and why queues explode",
      kind: "math",
      minutes: 90,
      md: MD(function () {/*
Two pieces of math explain almost every serving graph you'll ever see. Part A is about **one step** (what batching does to speed). Part B is about **the queue in front of the engine** (what traffic does to latency).

> [!PREREQ] Refresher: max, ratios, and "per"
> $\max(a, b)$ is just the larger of the two. "tokens per second" = tokens ÷ seconds. If one step takes 10 ms and produces 32 tokens, throughput is $\frac{32}{0.010} = 3{,}200$ tok/s and each user gets $\frac{1}{0.010} = 100$ tok/s. That's all the algebra you need.

## Part A — step time vs batch size

A decode step with $B$ sequences must (1) read all weights once ($W$ bytes), (2) read each sequence's KV cache, and (3) do $2P$ FLOPs per token ($P$ = parameters). Memory and compute overlap, so the slower one wins:

$$
t_{\text{step}}(B) \approx t_{\text{overhead}} + \max\!\Big(\underbrace{\frac{W + B \cdot \text{ctx} \cdot kv}{BW_{\text{mem}}}}_{\text{memory}},\ \underbrace{\frac{2 P B}{F}}_{\text{compute}}\Big)
$$

$$
\text{throughput} = \frac{B}{t_{\text{step}}(B)} \qquad \text{per-user speed} = \frac{1}{t_{\text{step}}(B)}
$$

Ignoring KV for a moment, compute catches up with the weight read when $\frac{2PB}{F} = \frac{W}{BW}$. With FP16 weights $W = 2P$, so

$$
B_{\text{sat}} \approx \frac{F}{BW_{\text{mem}}}
$$

— exactly the **ops:byte ratio** from m07. H100: $989\text{T} / 3.35\text{T} \approx 295$. T4: $65\text{T}/0.32\text{T} \approx 203$. M2 GPU: $\approx 3.6\text{T}/0.1\text{T} \approx 36$. Below $B_{\text{sat}}$, adding users is almost free.

**Worked example — Qwen2.5-7B, FP16, one H100, 2k-token contexts.** Weights $W \approx 15.2$ GB; KV per token $= 28 \text{ layers} \times 2 \times 4 \text{ KV heads} \times 128 \times 2 \text{ B} \approx 57$ KB.

| B | weights read | KV read ($B \times 2048 \times 57$ KB) | compute ($2 \cdot 7.6\text{e}9 \cdot B$ at ~60% of 989 TFLOPS) | step | per-user tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 1 | 4.5 ms | 0.03 ms | 0.03 ms | ≈4.6 ms | ≈217 | ≈217 |
| 64 | 4.5 ms | 2.2 ms | 1.6 ms | ≈6.7 ms | ≈150 | ≈9,500 |
| 256 | 4.5 ms | 8.9 ms | 6.6 ms | ≈13.4 ms | ≈75 | ≈19,000 |

From B=1 to B=64: **44× the throughput for 1.45× the per-token latency.** Notice that at long contexts it's the **KV read**, not compute, that eventually slows each step — which is why m08's KV tricks (GQA, quantized KV) also buy batch size.

::viz latency-throughput

## Part B — Little's law and utilization

**Little's law** (true for any stable system, no assumptions about distributions):

$$
L = \lambda \, W
$$

$L$ = average number of requests *in the system*, $\lambda$ = arrival rate, $W$ = average time each spends inside.

**Worked example.** A chat product gets $\lambda = 5$ req/s, and each request takes $W = 8$ s end-to-end (TTFT + 400 tokens at 50 tok/s). Then $L = 5 \times 8 = 40$ requests are in flight at any moment. So your engine must hold **40 sequences in its batch** — and 40 KV caches: $40 \times 2{,}300 \text{ tokens} \times 57 \text{ KB} \approx 5.2$ GB for the 7B model above. Little's law turns a traffic number into a **batch size and a memory budget**. (It's also why the Baseten book says to set the autoscaler's concurrency target to the batch size.)

**Utilization** $\rho = \lambda / \mu$, where $\mu$ is the service rate (requests/s the system can finish). For the simplest queue (one server, random arrivals and service times — "M/M/1"):

$$
W = \frac{1/\mu}{1 - \rho}
$$

| ρ (load) | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| time in system ÷ service time | 2× | 5× | 10× | 20× | 100× |

The $\frac{1}{1-\rho}$ term is the **hockey stick**: latency is flat, flat, flat… then explodes as you approach capacity. Play with it:

::viz queueing

> [!INTUITION] What's different for an LLM engine?
> A batching engine's $\mu$ **grows with load** (bigger batches → more tokens per step), so the knee comes later than M/M/1 predicts. But there is still a hard wall: when KV memory is full, the scheduler stops admitting, the **waiting queue** grows without bound, and **TTFT explodes while ITL looks perfectly healthy** (the running requests are fine; the new ones can't get in). You'll reproduce this signature in the lab. Rule of thumb: run replicas at ~60–80% of measured max capacity and autoscale on queue depth.

<details><summary>Where does 1/(1−ρ) come from? (optional)</summary>

For M/M/1 the number in the system is geometric: $P(n) = (1-\rho)\rho^n$. Its mean is $L = \rho/(1-\rho)$. Little's law then gives $W = L/\lambda = \frac{\rho}{\lambda(1-\rho)} = \frac{1/\mu}{1-\rho}$. You don't need to derive it — just recognize the shape.

</details>

- [ ] Using Part A, predict per-user and total tok/s for B = 1, 8, 32 on your Mac with Qwen2.5-0.5B (W ≈ 1 GB, your chip's bandwidth); compare with your *see-it* numbers
- [ ] With Little's law: 12 req/s, 6 s average latency — how many concurrent sequences? How much KV for 1,500-token contexts on Qwen2.5-7B?
- [ ] In the `queueing` viz, find the load where p99 latency is 10× the unloaded latency
      */}),
      resources: [
        { title: "Little's law (Wikipedia)", url: "https://en.wikipedia.org/wiki/Little%27s_law", type: "article", note: "statement, intuition and examples" },
        { title: "Databricks — LLM Inference Performance Engineering: Best Practices", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "batch size vs latency measurements on real GPUs" },
      ],
    },
    {
      id: "variable-lengths",
      title: "Batching sequences of different lengths: padding vs packed batches, masks, varlen attention",
      kind: "concept",
      minutes: 75,
      runsOn: ["mac"],
      md: MD(function () {/*
Four requests with prompts of 50, 400, 120 and 30 tokens. How do you put them through one forward pass?

## Option 1 — padding (what `model.generate(batch)` does)

Stack into a rectangle `[B=4, T=400]`, filling short rows with a pad token and an **attention mask** that says "ignore these":

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
batch = tok(["hi", "a much longer prompt ...", "medium one", "short"], padding=True, return_tensors="pt")
print(batch["input_ids"].shape, batch["attention_mask"])   # 0 = padding
```

Decoder-only models pad on the **left** for generation, so every row's *next* token lands in the same column. Positions must skip the pads (`position_ids = attention_mask.cumsum(-1) - 1`) or RoPE sees wrong positions.

**Cost:** $4 \times 400 = 1{,}600$ token slots for $600$ real tokens — **62.5% of the prefill FLOPs are wasted**. During decode the mask grows too, and a finished row keeps burning a slot until the batch ends.

## Option 2 — packed ("ragged") batches: what engines do

Concatenate all tokens into **one row** and describe the boundaries with metadata:

```text
tokens:      [ A0 A1 … A49 | B0 … B399 | C0 … C119 | D0 … D29 ]     T = 600, no padding
positions:   [ 0  1  … 49  | 0  … 399  | 0  … 119  | 0  … 29  ]     restart per sequence (RoPE!)
cu_seqlens:  [0, 50, 450, 570, 600]                                 cumulative lengths
slot_mapping:[ where each token's K/V goes in the paged pool (m08) ]
```

Linear layers and MLPs don't care about sequences at all — they see a `[600, hidden]` matrix (Orca's *selective batching*). Only **attention** must respect boundaries: token A7 may attend to A0…A7, never to B-anything. The mask is **block-diagonal and causal**:

```text
      A A A B B B B C C
  A   ■ · · · · · · · ·
  A   ■ ■ · · · · · · ·
  A   ■ ■ ■ · · · · · ·
  B   · · · ■ · · · · ·
  B   · · · ■ ■ · · · ·
  …
```

Decode rows fit the same scheme: a decoding request contributes **1 query token** but attends to its whole cached history (`k_len = context length`, read through its block table).

## Three ways to compute packed attention

| Approach | Where | Notes |
|---|---|---|
| **Loop per sequence** with `F.scaled_dot_product_attention` | anywhere (MPS, CPU, CUDA) | simple, exact; Python overhead per sequence — batch the decodes (pad only the tiny `[B, 1]` query side) |
| **FlexAttention** with a document mask | CUDA (compiled via Triton) | one call; you write the mask as a Python function |
| **Varlen / paged kernels**: FlashAttention `flash_attn_varlen_func`, `flash_attn_with_kvcache(block_table=…)`, FlashInfer | CUDA | what nano-vllm, vLLM and SGLang use; take `cu_seqlens` + block tables directly |

```python
# FlashAttention varlen (CUDA, Ampere+): q/k/v are [total_tokens, heads, head_dim]
from flash_attn import flash_attn_varlen_func
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                             max_seqlen_q=400, max_seqlen_k=400, causal=True)
```

```python
# FlexAttention document masking (CUDA): doc_id[i] = which sequence packed token i belongs to
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
def doc_causal(b, h, q_idx, kv_idx):
    return (doc_id[q_idx] == doc_id[kv_idx]) & (q_idx >= kv_idx)
mask = create_block_mask(doc_causal, B=None, H=None, Q_LEN=T, KV_LEN=T)
out = flex_attention(q, k, v, block_mask=mask)          # q: [1, heads, T, head_dim]
```

On your Mac you'll use the first approach (the lab's `PagedRunner` does exactly this): **one packed forward for everything except attention**, one batched SDPA call for all decode rows, and one causal SDPA per prefill chunk. That keeps the weight reads shared — the part that matters for throughput.

> [!REAL] Production
> vLLM's model runner flattens every scheduled token into a single "super-sequence" and builds `positions`, `slot_mapping` and attention metadata for the chosen backend (FlashAttention, FlashInfer, Triton, FlexAttention…). nano-vllm uses `flash_attn_varlen_func` for prefill and `flash_attn_with_kvcache` for decode, plus a tiny Triton kernel that scatters new K/V into the pool by `slot_mapping`.

- [ ] Compute the padding waste for prompts [30, 30, 30, 2000]. Why does sorting requests by length ("bucketing") help static batching but not fix it?
- [ ] Build `input_ids`, `positions` and `cu_seqlens` for three toy prompts by hand in Python, then verify with a per-sequence SDPA loop that packed attention equals running each sequence alone (`torch.allclose`)
- [ ] Explain why positions must restart at 0 for each packed sequence
      */}),
      resources: [
        { title: "FlashAttention (Dao-AILab)", url: "https://github.com/Dao-AILab/flash-attention", type: "repo", note: "see flash_attn_varlen_func and flash_attn_with_kvcache (paged KV)" },
        { title: "PyTorch — scaled_dot_product_attention", url: "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", type: "docs", note: "the portable kernel you'll use on MPS; boolean masks: True = attend" },
      ],
    },
    {
      id: "chunked-prefill",
      title: "Chunked prefill and token budgets (Sarathi-Serve)",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
## The problem: one long prompt freezes everyone

Continuous batching lets a new request join at the next step. But its **prefill** is a big step: an 8k-token prompt for an 8B model is $2 \times 8\text{e}9 \times 8192 \approx 131$ TFLOPs — about **200+ ms on an H100** at realistic efficiency. Every decoding user in the batch waits for that step, so their inter-token latency jumps from ~10 ms to ~230 ms. In a stream this is a visible **stutter**; for a voice agent it's a glitch.

::viz chunked-prefill

## The fix: a token budget per step

**Chunked prefill** (Sarathi, then **Sarathi-Serve**, OSDI '24) caps how many tokens any single step may process — the **token budget** — and splits long prompts into chunks that ride along with the decodes:

$$
\underbrace{n_{\text{decode}}}_{\text{1 token each}} + \sum_{\text{prefilling}} \text{chunk}_i \;\le\; \text{max\_num\_batched\_tokens}
$$

Each step now costs roughly the same ("uniform batches"), decodes never stall ("**stall-free scheduling**"), and the idle compute of memory-bound decode steps gets filled with compute-bound prefill work — *piggybacking*. Sarathi-Serve reports **2.6× higher serving capacity** for Mistral-7B on one A100 under tail-latency SLOs versus the vLLM of the time.

**Worked example.** Budget 512, 40 requests decoding, one 3,000-token prompt arrives:

| step | decode tokens | prefill chunk | step tokens |
|---|---|---|---|
| 1 | 40 | 472 (tokens 0–471) | 512 |
| 2 | 40 | 472 (472–943) | 512 |
| … | … | … | … |
| 7 | 40 | 168 (2832–2999) → **first token sampled** | 208 |

The new user's TTFT is now ~7 steps instead of 1 big one — *slightly worse* for them — but 40 other users never see a 200 ms hiccup.

## Choosing the budget

| smaller budget (e.g. 256–2048) | larger budget (e.g. 8192+) |
|---|---|
| better **ITL** (less prefill work in each step) | better **TTFT** (prompts finish in fewer steps) |
| more steps → more per-step overhead | better GPU efficiency, higher throughput |
| each chunk re-reads all earlier chunks' KV → attention cost grows | risk of decode stalls again |

This matches vLLM's own tuning guidance: smaller `max_num_batched_tokens` → better ITL; larger → better TTFT; > 8192 for maximum throughput on big GPUs with small models. On a GPU, keep chunks multiples of the kernel tile (e.g. 256) so matmuls stay efficient.

> [!REAL] The knobs in real engines
> **vLLM V1:** chunked prefill is on by default; `--max-num-batched-tokens` is the budget, `--max-num-seqs` caps running sequences, `--long-prefill-token-threshold` caps a single request's chunk, `--max-num-partial-prefills` limits how many prompts prefill concurrently. **SGLang:** `--chunked-prefill-size`, `--max-running-requests`, `--enable-mixed-chunk` (mix a prefill chunk with decodes). **TensorRT-LLM:** chunked context in the executor config.

> [!NOTE] The other answer: disaggregation
> Chunking shares one GPU between prefill and decode. **Disaggregation** (DistServe, Splitwise; Baseten ch. 5.5) runs them on *different* GPUs and ships the KV cache across. You'll meet it in m17; chunked prefill is the single-GPU version of the same insight — prefill and decode interfere.

- [ ] In `chunked-prefill`, find the chunk size where ITL p99 stops improving but TTFT still gets worse — that's your sweet spot for that workload
- [ ] With budget 1024 and 100 decoding sequences, how many prefill tokens fit per step? What happens at 1,000 decoding sequences?
- [ ] Why can't a decode step itself be "chunked"?
      */}),
      resources: [
        { title: "Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference", url: "https://arxiv.org/abs/2403.02310", type: "paper", note: "chunked prefill + stall-free batching; §3–4 are the scheduler" },
      ],
    },
    {
      id: "policies",
      title: "Preemption, priorities, fairness and scheduling policies",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
The scheduler admits requests optimistically: it allocates KV blocks **as sequences grow**, not their worst case up front (that's what makes paging efficient). The price: sometimes a running sequence needs a new block and **the pool is empty**. Something has to give.

## Preemption: recompute vs swap

| | **Recompute** (vLLM V1 default) | **Swap** (vLLM V0, PageServe) |
|---|---|---|
| what happens | free the victim's blocks; later re-prefill prompt + already-generated tokens | copy the victim's KV blocks to CPU RAM; copy back later |
| cost | prefill FLOPs for all its tokens (cheap-ish: prefill is compute-efficient, and freed blocks often survive in the **prefix cache** → recompute becomes a cache hit) | 2 × KV bytes over PCIe (e.g. 2,000 tok × 57 KB ≈ 114 MB ≈ 5 ms each way at ~25 GB/s) + CPU memory + complexity |
| on a Mac | natural (unified memory, no PCIe) | pointless — "CPU RAM" is the same memory |

**Victim choice.** vLLM preempts the **most recently admitted** running request (LIFO) in FCFS mode, or the lowest-priority one in priority mode — it has the least sunk work and protects the oldest users. Your scheduler will do the same. Two more guards:

- **Don't admit new work in a step that preempted** — otherwise you evict and re-admit in a loop (thrashing).
- **Watermark:** only admit a new request if a few blocks stay free afterwards, so running sequences can grow for a while.

> [!WARNING] Thrashing
> With a KV pool too small for the load, the scheduler spends its life preempting and recomputing. In the simulator you'll build, shrinking the pool from 4,096 to 200 blocks turns 0 preemptions into ~800 and cuts throughput 4×. When you see preemption warnings in vLLM logs, the fix is more KV memory (`--gpu-memory-utilization`, TP, KV quantization) or lower `--max-num-seqs`.

## Scheduling policies: who goes next?

| Policy | Rule | Good at | Bad at |
|---|---|---|---|
| **FCFS** | arrival order | predictable, fair-ish, simple | head-of-line blocking: a huge prompt at the front delays all |
| **Priority** | lowest `priority` value first (vLLM `--scheduling-policy priority`; tie-break by arrival) | paid tiers, interactive before batch jobs | **starvation** of low priority → add *aging* |
| **Shortest-job-first** (SJF/SRPT) | smallest (prompt + expected output) first | minimizes **mean/median** latency | output length is unknown (use `max_tokens` or a learned predictor); long jobs starve → worse **p99** |
| **SLO-aware / EDF** | earliest deadline first, deadline = arrival + TTFT SLO; shed load you can't serve in time | maximizing **goodput** (requests meeting SLO) | needs accurate cost models |
| **Cache-aware** (SGLang `lpm`) | longest shared-prefix-with-cache first | prefix-cache hit rate, throughput | can reorder unfairly; needs a radix tree |
| **Fair share** (VTC) | serve the client with the fewest weighted tokens served so far | multi-tenant fairness without hard rate limits | per-client bookkeeping |

A concrete surprise from the simulator in the build lesson (16 seats, overloaded): **SJF cut median TTFT from 9.1 s to 0.6 s — and doubled p99 TTFT** (20 s → 42 s), because long requests kept getting pushed back. Policies move latency between users; they rarely create capacity.

**Fairness in practice.** Most APIs use per-key rate limits, which waste capacity when the system is idle. The **Virtual Token Counter** (VTC, "Fairness in Serving LLMs") keeps a counter per client of (weighted) input + output tokens served, always admits from the client with the smallest counter, and proves the service difference between two backlogged clients stays within a 2× bound — while remaining work-conserving.

> [!REAL] Admission control is a policy too
> When the queue is so long that a request can't possibly meet its SLO, rejecting it quickly (HTTP 429/503) is kinder than serving it late — and lets the load balancer send it elsewhere. PageServe caps its queue (`MAX_QUEUE_SIZE`) and times requests out; production gateways (m18) do this at fleet level.

- [ ] Write the victim-selection rule for priority mode in one line of Python
- [ ] Sketch how you'd add aging to the priority policy (hint: effective priority = priority − α·waiting_time)
- [ ] Why does vLLM refuse to admit new requests in a step that preempted?
      */}),
      resources: [
        { title: "Fairness in Serving Large Language Models (VTC)", url: "https://arxiv.org/abs/2401.00588", type: "paper", note: "token-based fairness with continuous batching; short and readable" },
      ],
    },
    {
      id: "build-scheduler",
      title: "Build: a Scheduler with queues, token budget and a step() loop",
      kind: "build",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
You'll build the scheduler **against a simulated GPU first** (runs in milliseconds, deterministic, no model needed), then plug in the real model in the lab. Separating *policy* from *execution* is exactly how vLLM is structured (Scheduler vs ModelRunner), and it makes the scheduler unit-testable.

## The one idea: `num_computed`

Every request tracks `num_computed` — how many of its tokens already have K/V in the cache. Each step it processes some of the rest:

- **prefill**: many tokens (capped by the budget → a *chunk*),
- **decode**: exactly 1 (the token sampled last step),
- **after recompute-preemption**: `num_computed = 0`, so it re-prefills prompt + outputs.

When a step reaches the **last** known token, its logits predict a **new** token. Prefill, chunked prefill, decode and recompute are the same operation — which is why one forward pass can mix them.

## `scheduler.py`

```python
import time
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    ignore_eos: bool = False

@dataclass
class Request:
    req_id: str
    prompt_ids: list[int]
    params: SamplingParams
    arrival: float = 0.0
    priority: int = 0                      # lower = more important (vLLM convention)
    output_ids: list[int] = field(default_factory=list)
    num_computed: int = 0                  # tokens whose K/V already sit in the cache
    status: str = "waiting"                # waiting | running | finished
    first_token_t: float | None = None
    token_ts: list[float] = field(default_factory=list)
    finish_t: float | None = None
    num_preemptions: int = 0

    @property
    def all_ids(self): return self.prompt_ids + self.output_ids
    @property
    def num_tokens(self): return len(self.prompt_ids) + len(self.output_ids)
    @property
    def num_uncomputed(self): return self.num_tokens - self.num_computed

@dataclass
class ScheduledReq:
    req: Request
    start: int            # first position computed this step
    n: int                # tokens computed this step
    @property
    def samples(self):    # reaching the last known token => logits predict a NEW token
        return self.start + self.n == self.req.num_tokens

@dataclass
class SchedulerOutput:
    items: list[ScheduledReq]
    num_tokens: int
    num_preempted: int = 0

class Scheduler:
    def __init__(self, bm, max_num_batched_tokens=512, max_num_seqs=32,
                 long_prefill_threshold=256, policy="fcfs", eos_id=None,
                 watermark_blocks=1, clock=time.perf_counter):
        self.bm, self.budget, self.max_num_seqs = bm, max_num_batched_tokens, max_num_seqs
        self.chunk, self.policy, self.eos_id = long_prefill_threshold, policy, eos_id
        self.watermark, self.clock = watermark_blocks, clock
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add(self, req):
        req.status = "waiting"
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def _order_waiting(self):
        if self.policy == "fcfs":
            return                                         # deque order = arrival order
        keys = {"priority": lambda r: (r.priority, r.arrival),
                "sjf": lambda r: (len(r.prompt_ids) + r.params.max_tokens, r.arrival)}
        self.waiting = deque(sorted(self.waiting, key=keys[self.policy]))

    def _preempt(self, req):
        # recompute-style: drop the KV, re-prefill prompt + outputs later
        self.running.remove(req)
        self.bm.free(req.req_id)
        req.num_computed, req.status = 0, "waiting"
        req.num_preemptions += 1
        self._n_preempted += 1
        self.waiting.appendleft(req)

    def _make_room(self, req, num_tokens) -> bool:
        """Grow req's blocks to num_tokens, evicting the newest running requests if needed.
        Returns False if req itself had to be evicted."""
        while not self.bm.can_grow(req.req_id, num_tokens):
            victim = self.running[-1]                      # LIFO: newest arrival loses
            self._preempt(victim)
            if victim is req:
                return False
        self.bm.grow(req.req_id, num_tokens)
        return True

    def schedule(self) -> SchedulerOutput:
        budget, items = self.budget, []
        self._n_preempted = 0
        # 1) running requests first: decodes (1 token) and unfinished prefill chunks
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_uncomputed, budget, self.chunk)
            if not self._make_room(req, req.num_computed + n):
                break                                      # req itself was preempted
            items.append(ScheduledReq(req, req.num_computed, n))
            budget -= n
            i += 1
        # 2) admit waiting requests while budget, seats and memory allow
        self._order_waiting()
        while (self.waiting and budget > 0 and not self._n_preempted
               and len(self.running) < self.max_num_seqs):
            req = self.waiting[0]
            n = min(req.num_uncomputed, budget, self.chunk)
            if not self.bm.can_grow(req.req_id, req.num_computed + n, reserve=self.watermark):
                break                                      # head of line doesn't fit: stop
            self.waiting.popleft()
            self.bm.grow(req.req_id, req.num_computed + n)
            req.status = "running"
            self.running.append(req)
            items.append(ScheduledReq(req, req.num_computed, n))
            budget -= n
        return SchedulerOutput(items, self.budget - budget, self._n_preempted)

    def update(self, out, sampled: dict[str, int]) -> list[Request]:
        now, done = self.clock(), []
        for it in out.items:
            req = it.req
            samples = it.samples                           # evaluate before appending
            req.num_computed += it.n
            if not samples:
                continue                                   # mid-prompt chunk: no new token
            tok = sampled[req.req_id]
            req.output_ids.append(tok)
            req.token_ts.append(now)
            if req.first_token_t is None:
                req.first_token_t = now
            if (tok == self.eos_id and not req.params.ignore_eos) or \
               len(req.output_ids) >= req.params.max_tokens:
                req.status, req.finish_t = "finished", now
                self.bm.free(req.req_id)
                self.running.remove(req)
                self.finished.append(req)
                done.append(req)
        return done
```

Why is it safe to preempt `self.running[-1]` while iterating by index? Victims come from the **tail**, and everything before index `i` was already scheduled — so a victim is never something already in `items`, except `req` itself, which we handle.

## The engine step and a simulated GPU (`sim.py`)

```python
import random
from collections import deque
import numpy as np
from block_manager import BlockManager
from scheduler import Request, SamplingParams, Scheduler

class SimRunner:
    """Pretends to be a GPU: advances a virtual clock by a roofline-style step cost (Part A of the math lesson)."""
    def __init__(self, params=0.5e9, bytes_per_param=2, bw=100e9, flops=3.6e12,
                 kv_bytes_per_token=12_288, overhead_s=0.004, vocab=32_000, seed=0):
        self.P, self.bpp, self.bw, self.flops = params, bytes_per_param, bw, flops
        self.kvb, self.overhead, self.vocab = kv_bytes_per_token, overhead_s, vocab
        self.rng, self.now = random.Random(seed), 0.0
    def clock(self):
        return self.now
    def execute(self, out, bm):
        ctx = sum(it.start + it.n for it in out.items)            # KV entries attention reads
        t_mem = (self.P * self.bpp + ctx * self.kvb) / self.bw
        t_cmp = 2 * self.P * out.num_tokens / self.flops
        self.now += self.overhead + max(t_mem, t_cmp)
        return {it.req.req_id: self.rng.randrange(1, self.vocab) for it in out.items if it.samples}

def step(sched, runner):                     # THE engine loop body: schedule → execute → update
    out = sched.schedule()
    if not out.items:
        raise RuntimeError("a single request is larger than the KV pool")
    return sched.update(out, runner.execute(out, sched.bm))

def make_trace(n=200, rate=4.0, prompt=(64, 1024), output=(32, 256), seed=0):
    rng, t, reqs = random.Random(seed), 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)                               # Poisson arrivals
        p = rng.randint(*prompt)
        reqs.append(Request(f"r{i}", [rng.randrange(1, 32_000) for _ in range(p)],
                            SamplingParams(max_tokens=rng.randint(*output), ignore_eos=True),
                            arrival=t))
    return reqs

def run_trace(trace, sched, runner):
    pending, t0 = deque(sorted(trace, key=lambda r: r.arrival)), runner.clock()
    while pending or sched.has_work():
        while pending and pending[0].arrival <= runner.clock():
            sched.add(pending.popleft())
        if not sched.has_work():
            runner.now = pending[0].arrival                       # idle: jump to next arrival
            continue
        step(sched, runner)
    return sched.finished, runner.clock() - t0

def summarize(reqs, wall_s):
    pct = lambda xs, p: float(np.percentile(xs, p))
    ttft = [r.first_token_t - r.arrival for r in reqs]
    itl = [b - a for r in reqs for a, b in zip(r.token_ts, r.token_ts[1:])]
    return {"ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
            "itl_p50_ms": 1e3 * pct(itl, 50), "itl_p99_ms": 1e3 * pct(itl, 99),
            "e2e_p50_s": pct([r.finish_t - r.arrival for r in reqs], 50),
            "out_tok_s": sum(len(r.output_ids) for r in reqs) / wall_s,
            "preemptions": sum(r.num_preemptions for r in reqs)}

if __name__ == "__main__":
    big = 10**9
    for name, kw in {"sequential":  dict(max_num_seqs=1,  max_num_batched_tokens=big, long_prefill_threshold=big),
                     "continuous":  dict(max_num_seqs=64, max_num_batched_tokens=big, long_prefill_threshold=big),
                     "cont+chunked": dict(max_num_seqs=64, max_num_batched_tokens=512, long_prefill_threshold=512)}.items():
        runner = SimRunner()
        sched = Scheduler(BlockManager(4096, 16), clock=runner.clock, **kw)
        done, wall = run_trace(make_trace(), sched, runner)
        print(f"{name:13s}", {k: round(v, 3) for k, v in summarize(done, wall).items()})
```

Running it (200 requests at 4 req/s, a "0.5B model on an M2" cost model) prints something like:

```text
sequential    ttft_p50 196 s   itl_p99  14 ms   out_tok_s   67
continuous    ttft_p50 0.25 s  itl_p99 336 ms   out_tok_s  547
cont+chunked  ttft_p50 0.28 s  itl_p99 146 ms   out_tok_s  548
```

Read it: sequential can't keep up (TTFT = minutes, the queue explodes — Part B of the math lesson). Continuous batching serves everything with **8× the throughput**; chunking at 512 **halves ITL p99** at a tiny TTFT cost.

## Tests you should write (pytest)

- [ ] **Conservation:** after any trace, `bm.num_free_blocks() == num_blocks` and every request has exactly `max_tokens` outputs (with `ignore_eos`)
- [ ] **Budget:** every `SchedulerOutput.num_tokens <= max_num_batched_tokens`, and `len(running) <= max_num_seqs`
- [ ] **Chunking:** a 1,000-token prompt with threshold 256 is scheduled as 256+256+256+232 and samples only on the last chunk
- [ ] **Preemption:** with a tiny pool (e.g. 200 blocks) the trace still completes, and `num_preemptions > 0`
- [ ] **Policy:** with `policy="sjf"`, median TTFT drops and p99 TTFT rises versus FCFS on an overloaded trace (16 seats, 6 req/s)
- [ ] Swap in your m08 `BlockManager` (with prefix caching) — all tests still pass
      */}),
      resources: [
        { title: "nano-vllm — scheduler.py", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "compare with nanovllm/engine/scheduler.py: ~100 lines, prefill-first, recompute preemption" },
        { title: "PageServe (vLLM_Inference_Engine)", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "same num_computed_tokens design; see inference_engine/engine/scheduler.py" },
      ],
    },
    {
      id: "lab-sweep",
      title: "Lab: real model on your Mac + throughput/latency sweeps and plots",
      kind: "lab",
      minutes: 180,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Two parts: **(A)** replace `SimRunner` with a real Qwen2.5-0.5B forward pass over a **packed batch + paged KV pool** and measure actual speedups on your Mac; **(B)** use the fast simulator to sweep arrival rate and token budget and draw the plots.

## Part A — `PagedRunner`: the real forward pass

It reuses the Hugging Face module *weights* but replaces attention and KV handling (your m04 model works the same way — swap the attribute names). Everything except attention runs on one `[T, hidden]` packed matrix.

```python
# runner.py
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM

def rope(x, cos, sin):                          # x: [T, heads, D], HF "rotate_half" convention
    d = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat((-x[..., d:], x[..., :d]), -1) * sin[:, None]

class PagedRunner:
    def __init__(self, model_or_id, num_blocks=2048, block_size=16, device="mps", dtype=torch.float16):
        m = model_or_id
        if isinstance(m, str):
            m = AutoModelForCausalLM.from_pretrained(m, dtype=dtype)   # older transformers: torch_dtype=
        self.model = m.to(device=device, dtype=dtype).eval()
        cfg = self.model.config
        self.H, self.KVH = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.D, 2, device=device).float() / self.D))
        self.bs, self.device, self.dtype = block_size, device, dtype
        L, S = cfg.num_hidden_layers, num_blocks * block_size
        self.k = torch.zeros(L, S, self.KVH, self.D, device=device, dtype=dtype)   # the paged pool
        self.v = torch.zeros_like(self.k)

    def _slots(self, table, positions):
        return [table[p // self.bs] * self.bs + p % self.bs for p in positions]

    @torch.inference_mode()
    def execute(self, out, bm):
        ids, pos, slots, last, who, dec, pre = [], [], [], [], [], [], []
        for it in out.items:
            r, table = it.req, bm.block_table(it.req.req_id)
            p = list(range(it.start, it.start + it.n))
            q0 = len(ids)
            ids += r.all_ids[it.start: it.start + it.n]; pos += p
            slots += self._slots(table, p)
            ctx = self._slots(table, range(it.start + it.n))      # whole history incl. new tokens
            (dec if it.n == 1 else pre).append((q0, it.start, it.n, ctx))
            if it.samples:
                last.append(q0 + it.n - 1); who.append(r.req_id)
        dev, T, g = self.device, len(ids), self.H // self.KVH
        ids_t, pos_t, slot_t = (torch.tensor(a, device=dev) for a in (ids, pos, slots))
        dec_meta = None
        if dec:                                    # all decode rows -> one padded SDPA call
            Lmax = max(len(c) for *_, c in dec)
            idx = torch.zeros(len(dec), Lmax, dtype=torch.long)
            valid = torch.zeros(len(dec), Lmax, dtype=torch.bool)
            for b, (*_, c) in enumerate(dec):
                idx[b, :len(c)] = torch.tensor(c); valid[b, :len(c)] = True
            dec_meta = (torch.tensor([d[0] for d in dec], device=dev), idx.to(dev), valid.to(dev))
        pre_meta = []                              # each prefill chunk -> one causal SDPA call
        for q0, start, n, c in pre:
            mask = torch.arange(len(c))[None, :] <= torch.arange(start, start + n)[:, None]
            pre_meta.append((q0, n, torch.tensor(c, device=dev), mask.to(dev)))

        m = self.model.model
        freqs = pos_t.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().to(self.dtype), emb.sin().to(self.dtype)
        x = m.embed_tokens(ids_t)                                   # [T, C] — the packed batch
        for i, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = rope(a.q_proj(h).view(T, self.H, self.D), cos, sin)
            k = rope(a.k_proj(h).view(T, self.KVH, self.D), cos, sin)
            v = a.v_proj(h).view(T, self.KVH, self.D)
            self.k[i][slot_t] = k; self.v[i][slot_t] = v             # scatter new K/V into the pool
            o = torch.empty_like(q)
            if dec_meta is not None:
                qi, idx, valid = dec_meta
                K = self.k[i][idx].transpose(1, 2).repeat_interleave(g, 1)   # [B, H, Lmax, D]
                V = self.v[i][idx].transpose(1, 2).repeat_interleave(g, 1)
                o[qi] = F.scaled_dot_product_attention(q[qi].unsqueeze(2), K, V,
                            attn_mask=valid[:, None, None, :]).squeeze(2)
            for q0, n, c, mask in pre_meta:
                K = self.k[i][c].transpose(0, 1).repeat_interleave(g, 0)     # [H, ctx, D]
                V = self.v[i][c].transpose(0, 1).repeat_interleave(g, 0)
                o[q0:q0 + n] = F.scaled_dot_product_attention(
                    q[q0:q0 + n].transpose(0, 1), K, V, attn_mask=mask).transpose(0, 1)
            x = x + a.o_proj(o.reshape(T, self.H * self.D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if not last:
            return {}
        logits = self.model.lm_head(m.norm(x[torch.tensor(last, device=dev)])).float()
        return dict(zip(who, logits.argmax(-1).tolist()))           # greedy; m10 adds a real sampler
```

**Correctness first — greedy parity.** Build a tiny random model (`Qwen2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=500)`), run it in **float32 on CPU** through your scheduler with a small budget and tiny pool (chunking + preemption happen), and assert the tokens equal `hf.generate(do_sample=False)` for every prompt. Only then trust any speed number.

**Then measure on MPS.** Replay the same trace three ways with a *wall clock* driver (like `run_trace`, but `time.perf_counter()` and `time.sleep` until the next arrival): (1) sequential HF `model.generate` per request, (2) your engine with `max_num_seqs=32`, (3) plus chunking at 256. Fill in the goal table.

> [!WARNING] MPS gotchas
> Use `index_put_`-style assignment (`pool[slots] = k`) — `index_copy_` has been reported to be pathologically slow on MPS. Keep dtypes consistent (fp16 pool, fp16 activations). `.tolist()` synchronizes; don't time without it.

## Part B — sweeps and plots (simulator, seconds to run)

```python
# sweep.py
import matplotlib.pyplot as plt
from sim import SimRunner, Scheduler, BlockManager, make_trace, run_trace, summarize

configs = {"no chunking": dict(max_num_batched_tokens=10**9, long_prefill_threshold=10**9),
           "budget 2048": dict(max_num_batched_tokens=2048, long_prefill_threshold=2048),
           "budget 512":  dict(max_num_batched_tokens=512,  long_prefill_threshold=512)}
rates = [0.5, 1, 2, 3, 4, 5, 6, 8]
rows = []
for name, kw in configs.items():
    for rate in rates:
        runner = SimRunner()
        sched = Scheduler(BlockManager(4096, 16), clock=runner.clock, max_num_seqs=64, **kw)
        done, wall = run_trace(make_trace(n=300, rate=rate, seed=1), sched, runner)
        rows.append(dict(config=name, rate=rate, **summarize(done, wall)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for name in configs:
    rs = [r for r in rows if r["config"] == name]
    x = [r["rate"] for r in rs]
    ax[0].plot(x, [r["out_tok_s"] for r in rs], "o-", label=name)
    ax[1].plot(x, [r["ttft_p99_s"] for r in rs], "o-", label=name)
    ax[2].plot(x, [r["itl_p99_ms"] for r in rs], "o-", label=name)
for a, t in zip(ax, ["output tok/s", "TTFT p99 (s)", "ITL p99 (ms)"]):
    a.set_xlabel("arrival rate (req/s)"); a.set_title(t); a.grid(alpha=.3)
ax[1].set_yscale("log"); ax[0].legend()
plt.tight_layout(); plt.savefig("sweep_rate.png", dpi=150)
```

Then a second figure: fix the rate near the knee and sweep `max_num_batched_tokens` over [128, 256, 512, 1024, 2048, 8192]; plot **TTFT p99 (x) vs ITL p99 (y)** with points labelled by budget — a **Pareto curve**. Every engine config you'll ever tune lives on a curve like that.

- [ ] Greedy-parity test passes (float32, CPU, tiny model) with chunking **and** preemption active
- [ ] Measured sequential vs continuous vs continuous+chunked on MPS; speedup ≥ 3× (target: 5×+)
- [ ] `sweep_rate.png`: identify the knee rate for each config; confirm TTFT explodes while ITL stays flat
- [ ] Budget Pareto plot; picked a budget for "ITL p99 < 100 ms" and wrote down its TTFT cost
- [ ] Re-ran one sweep with a pool of 300 blocks; plotted preemptions vs rate
      */}),
    },
    {
      id: "deep-vllm-sglang",
      title: "Deep dive: vLLM V1's unified scheduler and SGLang's overlap scheduler",
      kind: "deep",
      optional: true,
      minutes: 120,
      md: MD(function () {/*
Now read how the pros do what you just built. Use Aleksa Gordić's walkthrough as a map (it pins vLLM commit `42172ad`, Aug 2025) and open the source next to it.

## vLLM V1: no "prefill phase" and "decode phase" — just tokens

vLLM V0 ran *either* a prefill step *or* a decode step. V1 dropped the distinction: the scheduler outputs a dict `{request_id: num_new_tokens}` and every request is just `num_computed_tokens` catching up to `num_tokens` (+ speculative tokens). Simplified:

```python
# vllm/v1/core/sched/scheduler.py — Scheduler.schedule(), heavily simplified
token_budget = max_num_batched_tokens
num_scheduled_tokens = {}                                # req_id -> n   (the key output)
for req in running:                                      # 1) running first (decodes + partial prefills)
    n = min(req.num_tokens_with_spec - req.num_computed_tokens,
            long_prefill_token_threshold or float("inf"), token_budget)
    while (new_blocks := kv_cache_manager.allocate_slots(req, n)) is None:
        victim = running.pop()                           # lowest priority / most recent
        kv_cache_manager.free(victim); victim.num_computed_tokens = 0
        waiting.appendleft(victim); preempted.append(victim)
        if victim is req: break
    else:
        num_scheduled_tokens[req.request_id] = n; token_budget -= n
if not preempted:                                        # 2) then waiting requests
    while waiting and token_budget > 0 and len(running) < max_num_seqs:
        req = waiting.peek()
        blocks, num_cached = kv_cache_manager.get_computed_blocks(req)   # prefix-cache hit
        n = min(req.num_tokens - num_cached, token_budget, long_prefill_token_threshold)
        if kv_cache_manager.allocate_slots(req, n, blocks) is None: break
        running.append(waiting.pop()); num_scheduled_tokens[req.request_id] = n; token_budget -= n
return SchedulerOutput(scheduled_new_reqs=..., scheduled_cached_reqs=...,
                       num_scheduled_tokens=num_scheduled_tokens, ...)
```

Things to notice (and compare with yours):

- **Decodes are prioritized** (running first), matching the docs' "chunked prefill prioritizes decode".
- **Preemption is recompute**, and freed blocks stay in the prefix cache — so the "recompute" is often a cheap cache hit.
- **`SchedulerOutput` is a diff.** New requests are sent in full; already-running ones send only new block ids / tokens. The worker keeps a **persistent batch** (`InputBatch`) so it doesn't rebuild tensors from scratch every step — this matters because the CPU work around each step can rival GPU time for small models.
- The same budget also covers speculative tokens, and there's a separate encoder budget for multimodal inputs.

## nano-vllm: the minimalist contrast

`nanovllm/engine/scheduler.py` (~100 lines) is **prefill-first and unmixed**: if anything is waiting, the step is prefill-only (only the first sequence may be chunked); otherwise it's a decode-only step. Simpler to make fast with CUDA graphs (decode shapes are uniform), worse ITL under bursty arrivals. A great exercise: point your simulator's cost model at both policies.

## SGLang: separate processes and overlap scheduling

SGLang splits the server into processes connected by ZMQ: **TokenizerManager** (HTTP + tokenization) → **Scheduler** (one per GPU rank; owns the radix-tree prefix cache and drives the model worker) → **DetokenizerManager**. The scheduler loop is roughly: receive requests → pick the next batch (by default it tries to build a **prefill** batch from the waiting queue first, else continues **decoding** the running batch; `--enable-mixed-chunk` mixes them) → run → process results.

Its signature trick is the **overlap ("zero-overhead") scheduler**, on by default since v0.4: the CPU schedules batch $N{+}1$ **while the GPU runs batch $N$**. The catch — batch $N{+}1$ needs the tokens batch $N$ hasn't produced yet — is solved with *future token* placeholders resolved on the GPU. Why bother? An unoptimized engine can spend up to half its time on CPU overhead (scheduling, radix-cache matching, building metadata); overlapping hides it. vLLM has since added its own async scheduling for the same reason.

Policies: `--schedule-policy` supports `lpm` (longest prefix match — cache-aware), `fcfs`, `lof`, `dfs-weight`, `random`.

> [!TIP] How to read engine source efficiently
> Put a breakpoint in `Scheduler.schedule()` and run the offline `LLM.generate` example with 3 prompts (vLLM: set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so everything is in one debuggable process). Step through one full engine step.

- [ ] Read Aleksa's "Scheduler" and "Chunked prefill" sections; map each step to a line in *your* `schedule()`
- [ ] Find in vLLM's scheduler where the watermark / "no admission after preemption" rules live
- [ ] Implement nano-vllm's prefill-first policy as `policy="prefill_first"` in your scheduler and compare ITL p99 in the simulator
- [ ] Sketch (don't build) how you'd overlap `schedule()` for step N+1 with `execute()` for step N in your engine — what state is unsafe to touch?
      */}),
      resources: [
        { title: "Aleksa Gordić — Inside vLLM: Anatomy of a High-Throughput LLM Inference System", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "the best tour of the V1 engine core, scheduler, chunked prefill and serving layer" },
        { title: "SGLang v0.4 — zero-overhead batch scheduler", url: "https://lmsys.org/blog/2024-12-04-sglang-v0-4/", type: "article", note: "overlap scheduling explained with Nsight traces" },
        { title: "vLLM source at the commit Aleksa analyzed", url: "https://github.com/vllm-project/vllm/tree/42172ad", type: "repo", note: "vllm/v1/core/sched/scheduler.py and kv_cache_manager.py" },
      ],
    },
  ],

  challenge: {
    title: "Continuous batching + chunked prefill, measured",
    md: MD(function () {/*
Build and measure a continuous-batching scheduler with chunked prefill on top of your m08 `BlockManager`, and write it up in `course-work/m09/REPORT.md`.

1. `scheduler.py` with waiting/running queues, a token budget, a per-request chunk cap, `max_num_seqs`, recompute preemption (LIFO victim, no admission after preemption) and at least **two policies** (FCFS + one of priority / SJF / fair-share).
2. A **synthetic trace generator** (Poisson arrivals; configurable prompt/output length distributions — include a "long prompt" mix where 10% of prompts are 8× longer).
3. Run it with the **simulator** for sweeps and with the **real model on MPS** (or Colab) for at least one point to calibrate the simulator's cost model (adjust `overhead_s`, `bw`, `flops` until simulated step times are within ~30% of measured).
4. Plots: **TTFT (p50/p99), ITL (p50/p99) and throughput vs arrival rate** for sequential / continuous / continuous+chunked; and a **budget Pareto** plot.
5. **One surprising result, explained.** Examples: SJF improving the median but hurting p99; chunking *lowering* throughput at low load; TTFT exploding while ITL is flat; preemption storms when the pool is 10% too small.

Hints: start from the build lesson's code; test with the tiny random model before touching real weights; keep every experiment's config in the JSON you save next to the plot.
    */}),
    checklist: [
      "Greedy parity with HF `generate` in float32 with chunking and preemption active",
      "Continuous batching reaches **≥3×** the throughput of sequential generation on your Mac (measured, not simulated)",
      "Plots of TTFT p50/p99, ITL p50/p99 and output tok/s vs arrival rate for 3 configs",
      "A budget sweep showing the TTFT-vs-ITL trade-off, with your chosen setting for an SLO of ITL p99 < 100 ms",
      "Simulator calibrated against at least one real measurement (step time within ~30%)",
      "`REPORT.md` explains one surprising result with a mechanism, not just a description",
    ],
    stretch: "Implement VTC fair-share scheduling for two tenants (one sends 10× more traffic) and show per-tenant token throughput with and without it; or implement SGLang-style overlap scheduling in your real engine (schedule step N+1 on a CPU thread while MPS runs step N) and measure the gain.",
  },

  connects: MD(function () {/*
You now own the **policy** half of an inference engine. In **m10** you wrap it with a sampler, streaming detokenizer, OpenAI-compatible API and metrics to get a complete mini-vLLM — the scheduler you wrote here is its heart. Later: **m15** (speculative decoding) adds multiple tokens per decode step to the same budget — and you'll see why *large batches starve speculation*; **m17** replaces chunked prefill with **disaggregation** for large deployments; **m18** turns Little's law into autoscaling ("concurrency target = batch size"); **m19** measures all of this under real load with `vllm bench serve` and proper SLO/goodput reporting.
  */}),

  interview: [
    "Explain static, dynamic and continuous batching. Why does continuous batching matter far more for LLMs than for, say, an image classifier?",
    "What problem does chunked prefill solve, and what does it cost? How would you choose `max_num_batched_tokens` for a chat workload with a strict ITL SLO?",
    "Your service handles 8 req/s with an average end-to-end latency of 6 s. How many sequences are in flight? Roughly how much KV cache do you need for 2k-token contexts on an 8B GQA model?",
    "Under load, p99 TTFT explodes but ITL looks perfectly healthy. What's happening inside the engine, and what would you change?",
    "Recompute vs swap preemption: when is each cheaper? Why did vLLM V1 default to recompute?",
    "Why does batching increase decode throughput almost linearly up to a point, but barely help prefill?",
    "Design a scheduler for a multi-tenant API where one customer sends 90% of the traffic. How do you keep the others' latency acceptable without hard rate limits?",
    "What is the CPU-side overhead in an inference engine's step, and how does SGLang's overlap scheduler hide it?",
  ],

  resources: [
    { title: "Orca (OSDI '22) — iteration-level scheduling", url: "https://www.usenix.org/conference/osdi22/presentation/yu", type: "paper", note: "the origin of continuous batching" },
    { title: "Sarathi-Serve — chunked prefill, stall-free batching", url: "https://arxiv.org/abs/2403.02310", type: "paper", note: "the origin of token budgets and piggybacking" },
    { title: "Anyscale — continuous batching (23× throughput)", url: "https://www.anyscale.com/blog/continuous-batching-llm-inference", type: "article", note: "accessible intro with benchmarks" },
    { title: "Aleksa Gordić — Inside vLLM", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "V1 scheduler, chunked prefill, preemption, benchmarks" },
    { title: "vLLM — Optimization and Tuning", url: "https://docs.vllm.ai/en/latest/configuration/optimization.html", type: "docs", note: "official guidance for max_num_batched_tokens, preemption" },
    { title: "SGLang v0.4 blog — zero-overhead scheduler", url: "https://lmsys.org/blog/2024-12-04-sglang-v0-4/", type: "article", note: "overlap scheduling and cache-aware load balancing" },
    { title: "Fairness in Serving LLMs (VTC)", url: "https://arxiv.org/abs/2401.00588", type: "paper", note: "fair scheduling under continuous batching" },
    { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "~1,200-line vLLM; read engine/scheduler.py and block_manager.py" },
    { title: "PageServe (vLLM_Inference_Engine)", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "readable PyTorch engine that runs on MPS; continuous batching + chunked prefill + preemption" },
    { title: "DistServe — disaggregating prefill and decode", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "the alternative answer to prefill/decode interference (m17 preview)" },
    { title: "Inference Engineering (Baseten) — local PDF, ch. 7.2", url: "Inference%20Engineering.pdf", type: "book", note: "batching types and concurrency targets in production" },
  ],
});
