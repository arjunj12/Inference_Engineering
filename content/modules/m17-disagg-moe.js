Course.module({
  id: "m17-disagg-moe",
  title: "Disaggregated serving & MoE at scale",
  short: "Disaggregation & MoE",
  tagline: "Split prefill and decode onto different workers, ship the KV cache between them, and reason about serving DeepSeek-V3-class MoE models with wide expert parallelism.",
  hours: 16,
  level: "core",
  runsOn: ["mac", "multi-gpu"],
  tags: ["disaggregation", "kv-transfer", "moe", "expert-parallelism", "dynamo", "deepseek"],

  goal: MD(function () {/*
On your Mac you run **two processes** — a *prefill worker* and a *decode worker* — built on a GPT-2-sized toy model with an explicit KV cache. The prefill worker processes the prompt, serializes the KV cache, ships it over a socket, and the decode worker continues generating **the exact same tokens** a single process would have produced. Then you show the reason anyone bothers: in a colocated engine, a long prefill **freezes** everyone else's token stream.

```text
$ python disagg_toy.py prefill --prompt-len 1024
prompt  1024 | prefill   412.3 ms | KV   75.5 MB float32 | pack  38.0 ms | send  61.2 ms | unpack  20.4 ms | decode  610.5 ms / 32 tok
  disaggregated tokens == colocated tokens: True

$ python interference.py
decode-only (disaggregated)  ITL p50   18.9 ms   p95   21.0 ms   max   24.8 ms
colocated prefill+decode     ITL p50   19.4 ms   p95  905.7 ms   max  930.2 ms
```

*(Illustrative numbers — yours depend on your chip. The shape of the result is what matters: same p50, catastrophic tail.)*

Then you move up to real systems: you'll size the KV transfer for Llama-3.1-70B over NVLink vs InfiniBand vs Ethernet, implement **expert parallelism with all-to-all** for a toy MoE layer, dissect **DeepSeek-V3's production serving architecture** (prefill EP32, decode EP144), and write a **capacity plan for DeepSeek-V3 on H200 nodes** — including the case where disaggregation is *not* worth it.

::viz disaggregation
  */}),
  demo: { viz: "disaggregation", params: {} },

  why: MD(function () {/*
Disaggregation and large-scale MoE serving are *the* frontier of inference infrastructure in 2025–26. DeepSeek, Moonshot (Kimi), and every big API provider serve their largest models this way; NVIDIA built **Dynamo** around it; vLLM, SGLang and TensorRT-LLM all ship prefill/decode (PD) disaggregation and wide expert parallelism; Red Hat/Google/IBM built **llm-d** to run it on Kubernetes.

The Baseten book calls disaggregation one of the six core runtime techniques (Inference Engineering ch. 5.5) — and is blunt that it's only worth it at scale. Being able to say **when** it pays off, **what** it costs (KV bytes over which link), and **how** MoE changes the picture is exactly the judgement senior inference roles hire for.
  */}),

  prereqs: [
    {
      title: "Prefill vs decode, TTFT vs ITL (1-minute recap)",
      skipIf: "you did M06 and M09",
      md: MD(function () {/*
- **Prefill** processes all prompt tokens in one big forward pass → **compute-bound**, sets **TTFT**.
- **Decode** generates one token per sequence per step → **memory-bandwidth-bound**, sets **ITL/TPOT**.
- A **continuous-batching** engine mixes both in the same GPU steps. **Chunked prefill** (M09) caps how many prompt tokens a step may contain, to protect ITL.
- The **KV cache** is the state that links them: prefill *creates* it, decode *reads and extends* it.

::viz prefill-decode
      */}),
    },
    {
      title: "KV bytes per token",
      skipIf: "you can compute the KV size of Llama-3-70B from its config",
      math: true,
      md: MD(function () {/*
For every token, every layer stores one **K** and one **V** vector per KV head:

$$
\text{KV bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes/elem}
$$

Llama-3.1-8B: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = **128 KiB**. Llama-3.1-70B: $2 \times 80 \times 8 \times 128 \times 2$ = **320 KiB**. Multiply by the number of tokens to get the bytes you'd have to ship from a prefill GPU to a decode GPU.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: a long prefill stalls everyone's decode",
      kind: "demo",
      minutes: 45,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
## The interference problem

Imagine 30 users happily streaming tokens at 20 ms each. A new user pastes a **30,000-token codebase** into their prompt. In a colocated engine, the next GPU step must include that prefill (or a chunk of it). Everyone else's next token waits.

::viz disaggregation

Play with it: turn on long prompts in the colocated timeline and watch the **inter-token latency (ITL)** spikes on the decode rows. Switch to disaggregated: prefill runs on its own GPUs, the KV cache is transferred, and the decode GPUs' rhythm never breaks.

Compare with the fix you already know — **chunked prefill**:

::viz chunked-prefill

Chunking shrinks each stall, but prefill and decode still **share the same GPU steps**. A decode step that carries a 2,048-token prefill chunk is still several times slower than a pure decode step. Under heavy load with long prompts, you can't tune the chunk size to satisfy both a tight TTFT and a tight ITL SLO at once.

## The two phases want different things

| | Prefill | Decode |
|---|---|---|
| bottleneck | compute (FLOPs) | memory bandwidth (weights + KV reads) |
| ideal batch | a few long prompts fill the GPU | as many sequences as KV memory allows |
| ideal parallelism | lower TP (less comm per FLOP), or even replicas | higher TP / wide EP (more aggregate bandwidth) |
| sets which SLO | TTFT | ITL / TPOT |
| hardware preference | max FLOPS | max HBM bandwidth & capacity |

**Disaggregation** = run each phase on its own pool of GPUs, tuned for that phase, and ship the KV cache in between:

1. Router sends the request to a **prefill worker**, which computes the KV cache and the first token.
2. The KV cache is **transferred** (NVLink / RDMA) to a **decode worker**.
3. The decode worker streams the remaining tokens.

> [!REAL] Inference Engineering (Baseten) ch. 5.5
> The book builds disaggregation on three ideas: prefill is compute-bound and sets TTFT while decode is memory-bound and sets TPS; **specialization helps**; and multi-GPU is fine *if you avoid interconnect bottlenecks*. It also warns that prefill and decode **interfere** under heavy load with large batches.

- [ ] In the `disaggregation` viz, found a load level where colocated ITL p99 is > 5× its p50
- [ ] In `chunked-prefill`, found the chunk size that keeps the decode stall under 2× — what did it do to TTFT?
- [ ] Filled the table above from memory
      */}),
    },
    {
      id: "distserve-splitwise",
      title: "DistServe, Splitwise and the idea of goodput",
      kind: "read",
      minutes: 75,
      md: MD(function () {/*
Two 2024 papers made disaggregation mainstream. Read both abstracts, intros and evaluation figures (≈ 1.5 hours total).

## DistServe (Zhong et al., OSDI 2024)

Key argument: colocation (1) causes prefill–decode **interference** and (2) **couples the parallelism and resource allocation** of both phases, so you over-provision to meet both SLOs. DistServe puts the phases on different GPUs, **searches parallelism per phase** (e.g. prefill TP=2, decode TP=4) and **places** them according to cluster bandwidth so KV transfers stay on fast links.

Its metric is the important idea — **goodput**:

$$
\text{goodput} = \max\ \text{request rate such that} \ge 90\%\ \text{of requests meet both the TTFT and the TPOT SLO}
$$

usually reported **per GPU**. Throughput counts tokens even when users were miserable; goodput only counts requests that were served *well*. DistServe reports up to **7.4× more requests** or **12.6× tighter SLOs** than colocated vLLM at the time, while keeping ≥ 90% of requests within SLO.

> [!INTUITION] Why goodput changes the answer
> A colocated engine can post a higher raw tokens/s number while failing the ITL SLO on 30% of requests. Once you measure goodput, a "slower" disaggregated system can win by a wide margin. Always ask: *throughput at what SLO?*

## Splitwise (Patel et al., ISCA 2024, Microsoft)

Same split, different emphasis: **hardware and cost**. Characterizing production traces from Azure, they found decode leaves compute idle and doesn't need the newest GPU. Splitwise designs clusters with separate **prompt machines** and **token machines** — possibly *different GPU types* — and optimizes for throughput, cost *and power*. Headline: **1.4× throughput at 20% lower cost**, or **2.35× throughput at the same cost and power**. They transfer KV **layer by layer**, overlapping transfer with the remaining prefill compute.

## The vocabulary you'll hear

- **xPyD** — x prefill workers, y decode workers (e.g. **5P3D**). The ratio depends on prompt/output lengths and SLOs, and it **drifts** during the day.
- **Conditional disaggregation** — send each request to decode first; if its prompt is short or mostly **prefix-cached**, decode does the prefill locally; otherwise forward to the prefill pool. Baseten and Dynamo both recommend this for real traffic.
- **KV connector / transfer engine** — the component that moves KV blocks between engines (NIXL, Mooncake Transfer Engine, LMCache).

> [!REAL] Where the ratio comes from
> If prefill of one request takes $t_p$ on a prefill worker and decode of one request takes $t_d$ of a decode worker's time (at its target batch), then at steady state $x \cdot \frac{1}{t_p} = y \cdot \frac{1}{t_d}$ requests/s must match. Long prompts + short outputs (RAG, summarization, code editing) → more P. Short prompts + long outputs (reasoning, chat) → more D.

- [ ] Read the DistServe abstract, §1–2 and the goodput figures
- [ ] Read the Splitwise abstract, §1–3 (phase characterization)
- [ ] Wrote down, in your own words, the difference between throughput and goodput, with a numeric example
- [ ] Estimated a sensible xPyD for: (a) RAG with 8K prompts / 200 output tokens, (b) reasoning with 500-token prompts / 4K output tokens
      */}),
      resources: [
        { title: "DistServe (Zhong et al., 2024)", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "disaggregation for goodput; per-phase parallelism search" },
        { title: "Splitwise (Patel et al., 2024)", url: "https://arxiv.org/abs/2311.18677", type: "paper", note: "phase splitting for throughput, cost and power; heterogeneous hardware" },
      ],
    },
    {
      id: "kv-transfer-math",
      title: "Math: KV transfer — size, bandwidth, and when it's cheap enough",
      kind: "math",
      minutes: 75,
      md: MD(function () {/*
> [!PREREQ] Two reminders from M16
> Network links are quoted in **bits per second** — divide by 8 for bytes (400 Gb/s IB = 50 GB/s). And the α–β model: sending $S$ bytes takes $T = \alpha + S/B$.

## The core formula

$$
T_{\text{transfer}} \approx \alpha + \frac{\text{KV bytes/token} \times \text{prompt tokens}}{\text{link bandwidth}}
$$

::viz kv-calculator

## Worked example — Llama-3.1-70B, 8K-token prompt

KV = 327,680 B/token × 8,192 tokens ≈ **2.68 GB**.

| Link | Bandwidth (one direction) | Transfer time | vs. prefill (~0.2–0.3 s on 8× H100) |
|---|---|---|---|
| NVLink 4 (same node) | 450 GB/s | **6 ms** | negligible |
| InfiniBand NDR, 1 NIC | 50 GB/s | **54 ms** | noticeable |
| IB NDR, 8 NICs (each TP rank ships its own shard over its own NIC) | 400 GB/s | **7 ms** | negligible |
| 100 GbE, TCP | 12.5 GB/s | **215 ms** | ≈ doubles TTFT |

::viz comm-cost

## The rate view (more useful for capacity planning)

Per request is one thing; the fleet question is: *can the link keep up with the prefill pool's output rate?* A prefill worker produces KV at

$$
\text{KV rate} = \text{prefill tokens/s} \times \text{KV bytes/token}.
$$

Prefill tokens/s for a dense model ≈ $\frac{N_{\text{GPU}} \times \text{peak FLOP/s} \times \text{MFU}}{2 \times \text{params}}$. For 70B on 8× H100 at 50% MFU: $\frac{8 \times 989\text{T} \times 0.5}{2 \times 70.6\text{B}} \approx 28\text{K tok/s}$ → $28{,}000 \times 327{,}680 \approx$ **9.2 GB/s** of KV. That's 2% of a node's 8×IB capacity — comfortable — but ~75% of one 100 GbE link.

> [!INTUITION] Why this is roughly independent of prompt length
> Both prefill compute and KV bytes grow **linearly** with prompt length (attention adds a quadratic compute term, which only makes transfer *relatively* cheaper for long prompts). So the ratio *transfer time ÷ prefill time* is set mostly by **model shape and link speed**, not by the prompt. What prompt length changes is **how much prefill interferes with decode** — which is why long prompts are where disaggregation wins.

## Architecture changes the answer by 10–70×

| Model | KV per token (BF16) | 8K prompt | Why |
|---|---|---|---|
| hypothetical 70B with full MHA (64 KV heads) | 2.5 MiB | 21 GB | no KV sharing — disaggregation is painful |
| Llama-3.1-70B (GQA, 8 KV heads) | 320 KiB | 2.7 GB | GQA: 8× less |
| DeepSeek-V3 (MLA: 512-dim latent + 64 RoPE dims, 61 layers) | $61 \times 576 \times 2 \approx$ **69 KiB** | 0.58 GB | MLA compresses KV into a latent |
| same, FP8 KV | ~34 KiB | 0.29 GB | KV quantization (M14) |

That's why the Baseten book notes that **KV-cache quantization helps disaggregation**, and why MLA models are such good candidates for it.

## Hiding the transfer

Engines don't wait for the whole prefill to finish: they send **layer by layer** (Splitwise) or **block by block** as each layer's KV is written, overlapping transfer with the remaining layers' compute. With overlap, the visible cost is roughly $\max(0,\ T_{\text{transfer}} - T_{\text{prefill}}) + $ one layer's worth.

- [ ] Recomputed the 70B table for a 32K prompt and for FP8 KV
- [ ] Computed the KV rate for Llama-3.1-8B prefill on 1× H100 (~50% MFU): does 100 GbE keep up?
- [ ] Used `kv-calculator` to verify DeepSeek-V3 MLA's bytes/token vs a GQA model of similar size
- [ ] Explained why a prefix-cache hit reduces **both** prefill compute and KV transfer
      */}),
    },
    {
      id: "systems",
      title: "Systems: Dynamo, llm-d, SGLang PD, Mooncake, NIXL, LMCache",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
A production disaggregated stack has four pieces. Map each system onto them:

| Piece | Job | Examples |
|---|---|---|
| **Engine** | runs prefill or decode | vLLM, SGLang, TensorRT-LLM |
| **KV transfer library** | move KV blocks GPU→GPU (or via CPU/SSD) fast, async | **NIXL**, Mooncake Transfer Engine, UCX, NCCL p2p |
| **KV cache layer / store** | offload, share and reuse KV across instances and tiers | **LMCache**, **Mooncake Store**, Dynamo KVBM |
| **Router / orchestrator** | choose P and D workers, conditional disaggregation, rebalance xPyD | **Dynamo**, **llm-d**, SGLang router, vLLM proxy examples |

## NVIDIA Dynamo

Open-source, datacenter-scale serving framework that **orchestrates** vLLM, SGLang or TRT-LLM workers. Features (Baseten ch. 4.4 & 5.5):

- **Disaggregated serving** with a **prefill queue** and **conditional disaggregation**: thresholds on *prompt length after prefix-cache hits* and on *prefill queue size* decide whether decode does prefill locally.
- **KV-aware router**: sends requests where their prefix is already cached (M18).
- **KVBM (KV Block Manager)**: moves KV blocks across GPU HBM → CPU DRAM → SSD → remote storage.
- **Planner**: changes the **xPyD ratio at runtime** (e.g. 5P3D → 4P4D) as traffic shifts.
- **NIXL-based KV transfer**, including a kernel that **transposes KV layouts** when prefill and decode use different TP.

## NIXL

NVIDIA Inference Xfer Library: one API for point-to-point transfers between GPU memory, CPU memory and storage, over pluggable backends (UCX, GPUDirect Storage, libfabric…). Used by Dynamo, vLLM's `NixlConnector` and SGLang.

## Mooncake (Moonshot AI / Kimi)

A **KVCache-centric** architecture: separate prefill and decode clusters plus a **distributed KV cache pool** built from the cluster's otherwise-idle CPU DRAM and SSDs, moved with an RDMA **Transfer Engine**. The scheduler treats KV placement as a first-class decision and uses early rejection under overload. Its transfer engine now backs connectors in vLLM and SGLang.

## LMCache

A KV cache layer for vLLM (and SGLang): offloads KV to CPU/disk/remote, **reuses** it across requests and instances (not only prefixes), and supports PD transfer via NIXL. In vLLM it's `LMCacheConnectorV1`.

## llm-d

Kubernetes-native distributed inference built on vLLM and the Gateway API Inference Extension. Offers "well-lit paths": **prefix/KV-aware scheduling**, **PD disaggregation**, and **wide expert parallelism** for big MoE models — each as Helm-deployable recipes.

## What it looks like in vLLM and SGLang

```bash
# vLLM: prefill instance (producer) and decode instance (consumer) with NIXL
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve Qwen/Qwen3-0.6B --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
# + a proxy that sends each request to prefill first, then to decode (lab below)

# SGLang: same idea, with its router
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill --port 30000
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --port 8000
```

> [!NOTE] Flags move fast
> These connector names and flags are current as of late 2025–2026 but change often; check the vLLM "Disaggregated Prefilling" page and SGLang "PD Disaggregation" page before running. vLLM's own docs note that disaggregated prefill **does not improve throughput by itself** — it lets you tune TTFT and ITL independently and control **tail ITL**.

- [ ] Drew the four-piece diagram and placed Dynamo, llm-d, NIXL, Mooncake, LMCache on it
- [ ] Read vLLM's disaggregated prefilling page and listed its connectors
- [ ] Explained why Dynamo needs a *layout-transpose* kernel when prefill TP ≠ decode TP
      */}),
      resources: [
        { title: "NVIDIA Dynamo (repo)", url: "https://github.com/ai-dynamo/dynamo", type: "repo", note: "disaggregation, KV-aware routing, KVBM, planner" },
        { title: "NIXL", url: "https://github.com/ai-dynamo/nixl", type: "repo", note: "the transfer library under Dynamo and vLLM's NixlConnector" },
        { title: "Mooncake paper", url: "https://arxiv.org/abs/2407.00079", type: "paper", note: "KVCache-centric disaggregated architecture behind Kimi" },
        { title: "Mooncake (repo)", url: "https://github.com/kvcache-ai/Mooncake", type: "repo", note: "Transfer Engine + Mooncake Store" },
        { title: "LMCache", url: "https://github.com/LMCache/LMCache", type: "repo", note: "KV cache offload/reuse layer for vLLM and SGLang" },
        { title: "llm-d", url: "https://github.com/llm-d/llm-d", type: "repo", note: "Kubernetes-native PD disaggregation and wide-EP recipes" },
        { title: "vLLM — Disaggregated prefilling", url: "https://docs.vllm.ai/en/latest/features/disagg_prefill.html", type: "docs", note: "connectors and how vLLM implements PD" },
        { title: "SGLang — PD disaggregation", url: "https://docs.sglang.ai/advanced_features/pd_disaggregation.html", type: "docs", note: "Mooncake/NIXL backends, router, DeepSeek multi-node recipes" },
      ],
    },
    {
      id: "build-toy-disagg",
      title: "Build: a toy disaggregated engine on your Mac (two processes, KV over a socket)",
      kind: "build",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] What you're building
> `toy_model.py` (GPT-2-small-shaped model, random weights, explicit KV cache), `disagg_toy.py` (prefill worker → socket → decode worker, with timing and a correctness check) and `interference.py` (measures ITL with and without prefill interference). If you finished your mini-engine in M10, port these ideas into it afterwards — the toy keeps everything visible.

## Step 1 — a model whose KV cache you can hold in your hand

```python
# toy_model.py — a GPT-2-small-shaped model with an explicit KV cache (random weights).
import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.qkv, self.o = nn.Linear(C, 3 * C, bias=False), nn.Linear(C, C, bias=False)
        self.up, self.down = nn.Linear(C, 4 * C, bias=False), nn.Linear(4 * C, C, bias=False)

    def forward(self, x, kv):                        # kv: None or (k, v), each (B, H, T_past, hd)
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.H, C // self.H).permute(2, 0, 3, 1, 4)
        if kv is not None:
            k, v = torch.cat([kv[0], k], dim=2), torch.cat([kv[1], v], dim=2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=kv is None)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        x = x + self.down(F.gelu(self.up(self.ln2(x))))
        return x, (k, v)

class ToyLM(nn.Module):
    def __init__(self, V=32000, C=768, L=12, H=12, max_T=8192):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, C), nn.Embedding(max_T, C)
        self.blocks = nn.ModuleList(Block(C, H) for _ in range(L))
        self.ln, self.head = nn.LayerNorm(C), nn.Linear(C, V, bias=False)

    @torch.no_grad()
    def forward(self, idx, cache=None):              # returns last-position logits + new cache
        past = 0 if cache is None else cache[0][0].shape[2]
        x = self.emb(idx) + self.pos(torch.arange(past, past + idx.shape[1]))
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.head(self.ln(x))[:, -1], new_cache

def load_model():
    torch.manual_seed(0)                             # same seed => identical weights in every process
    return ToyLM().eval()

def pack(cache):                                     # list[(k,v)] -> one contiguous tensor (L, 2, B, H, T, hd)
    return torch.stack([torch.stack(kv) for kv in cache]).contiguous()

def unpack(t):
    return [(t[i, 0], t[i, 1]) for i in range(t.shape[0])]

def greedy_decode(model, first_token, cache, n):
    out, tok = [first_token], torch.tensor([[first_token]])
    for _ in range(n - 1):
        logits, cache = model(tok, cache)
        tok = logits.argmax(-1, keepdim=True)
        out.append(tok.item())
    return out
```

Predict before running: KV bytes/token $= 2 \times 12 \times 12 \times 64 \times 4$ (FP32) = **73,728 B**, so a 1,024-token prompt is **75.5 MB**.

## Step 2 — prefill worker, decode worker, a socket in between

```python
# disagg_toy.py — prefill in one process, ship the KV cache over a socket, decode in another.
#   terminal 1:  python disagg_toy.py decode
#   terminal 2:  python disagg_toy.py prefill --prompt-len 2048
import argparse, time, numpy as np, torch
from multiprocessing.connection import Listener, Client
from toy_model import load_model, pack, unpack, greedy_decode

ADDR, KEY = ("127.0.0.1", 6001), b"toy-disagg"
torch.set_num_threads(4)

def decode_server():
    model = load_model()
    with Listener(ADDR, authkey=KEY) as ln:
        print("decode worker ready on", ADDR, flush=True)
        while True:
            with ln.accept() as conn:
                meta = conn.recv()                                  # small pickled header
                t0 = time.perf_counter()
                buf = conn.recv_bytes()                             # the KV payload
                t1 = time.perf_counter()
                kv = torch.from_numpy(np.frombuffer(buf, dtype=meta["np_dtype"]).copy())
                cache = unpack(kv.view(meta["shape"]).float())      # compute in fp32 on CPU
                t2 = time.perf_counter()
                toks = greedy_decode(model, meta["first_token"], cache, meta["max_new"])
                t3 = time.perf_counter()
                conn.send({"tokens": toks, "recv_ms": 1e3 * (t1 - t0),
                           "deser_ms": 1e3 * (t2 - t1), "decode_ms": 1e3 * (t3 - t2)})

def prefill_client(prompt_len, max_new, wire_dtype):
    model = load_model()
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 32000, (1, prompt_len), generator=g)
    t0 = time.perf_counter()
    logits, cache = model(prompt)                                   # PREFILL: whole prompt at once
    first = logits.argmax(-1).item()
    t1 = time.perf_counter()
    kv = pack(cache).to(getattr(torch, wire_dtype))                 # cast to the wire dtype
    payload = kv.numpy().tobytes()
    t2 = time.perf_counter()
    with Client(ADDR, authkey=KEY) as conn:
        conn.send({"shape": tuple(kv.shape), "np_dtype": wire_dtype,
                   "first_token": first, "max_new": max_new})
        conn.send_bytes(payload)
        t3 = time.perf_counter()
        reply = conn.recv()
    print(f"prompt {prompt_len:5d} | prefill {1e3*(t1-t0):7.1f} ms | KV {len(payload)/1e6:6.1f} MB {wire_dtype} | "
          f"pack {1e3*(t2-t1):5.1f} ms | send {1e3*(t3-t2):5.1f} ms | unpack {reply['deser_ms']:5.1f} ms | "
          f"decode {reply['decode_ms']:6.1f} ms / {max_new} tok")
    ref = greedy_decode(model, first, cache, max_new)               # colocated reference
    print("  disaggregated tokens == colocated tokens:", ref == reply["tokens"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["prefill", "decode"])
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    a = ap.parse_args()
    decode_server() if a.role == "decode" else prefill_client(a.prompt_len, a.max_new, a.dtype)
```

`multiprocessing.connection` gives you length-prefixed messages over TCP and an auth handshake for free, so you can focus on the KV. Run the decode worker in one terminal, then sweep prompt lengths in another:

```bash
python disagg_toy.py decode
for n in 128 512 1024 2048 4096; do python disagg_toy.py prefill --prompt-len $n; done
python disagg_toy.py prefill --prompt-len 2048 --dtype float16    # half the bytes on the wire
```

## Step 3 — measure the interference you're trying to remove

```python
# interference.py — does a long prefill stall ongoing decodes? (colocated vs decode-only worker)
import time, statistics, torch
from toy_model import load_model

torch.set_num_threads(4)
model = load_model()
B, CTX, STEPS = 8, 256, 60            # 8 users mid-conversation, 256 tokens of context each
PROMPT_LEN, EVERY = 2048, 15          # a new 2048-token prompt arrives every 15 decode steps

def run(colocated):
    _, cache = model(torch.randint(0, 32000, (B, CTX)))      # equal-length batch keeps the toy simple
    tok = torch.zeros(B, 1, dtype=torch.long)
    itl, last = [], time.perf_counter()
    for step in range(STEPS):
        if colocated and step % EVERY == 0 and step > 0:
            model(torch.randint(0, 32000, (1, PROMPT_LEN)))   # new request's prefill runs HERE
        logits, cache = model(tok, cache)                     # one decode step for all B users
        tok = logits.argmax(-1, keepdim=True)
        now = time.perf_counter(); itl.append(1e3 * (now - last)); last = now
    itl = sorted(itl[3:])                                     # drop warm-up steps
    return statistics.median(itl), itl[int(0.95 * len(itl))], itl[-1]

for name, col in [("decode-only (disaggregated)", False), ("colocated prefill+decode", True)]:
    p50, p95, mx = run(col)
    print(f"{name:28s} ITL p50 {p50:6.1f} ms   p95 {p95:7.1f} ms   max {mx:7.1f} ms")
```

> [!WARNING] Be honest about what the toy measures
> Both processes share your Mac's CPU cores and memory bandwidth, so the "separate hardware" part of disaggregation is simulated. That's fine: the point is to see **where the time goes** (prefill vs pack vs wire vs unpack) and **the shape of the ITL tail**. Pin threads (`torch.set_num_threads`) so the two workers don't fight.

## Step 4 — make it less toy

- [ ] Ran the sweep; made a table of prompt length vs prefill ms, KV MB, pack+send+unpack ms
- [ ] Plotted *transfer overhead ÷ prefill time* vs prompt length — roughly flat, as the math lesson predicted?
- [ ] Switched the wire dtype to float16: bytes halve — did the tokens still match? (If not, why can rounding flip a greedy argmax?)
- [ ] Replaced the socket with **shared memory** (`multiprocessing.shared_memory.SharedMemory`, then `np.ndarray(..., buffer=shm.buf)`), sending only the name + shape — how much faster than the socket?
- [ ] **Layer-wise streaming:** send each layer's KV as soon as that layer finishes prefill (a hook in `ToyLM.forward`), with the decode side receiving in a background thread — measure how much transfer time you hide
- [ ] Added a tiny **router** that does *conditional disaggregation*: prompts shorter than a threshold are prefilled on the decode worker itself
      */}),
    },
    {
      id: "lab-vllm-pd",
      title: "Lab (cloud, optional): vLLM 1P1D with the NIXL connector vs 2 colocated replicas",
      kind: "lab",
      optional: true,
      minutes: 150,
      runsOn: ["multi-gpu", "cloud"],
      md: MD(function () {/*
> [!GOAL] Output
> Goodput-style comparison of **1 prefill + 1 decode** (disaggregated) vs **2 independent replicas** (colocated, DP=2) on the same 2 GPUs, for a prefill-heavy and a decode-heavy workload.

Cost note: a 2× H100 (or 2× A100/L40S) instance for ~2 hours ≈ **\$5–12**. Any 2-GPU box works; NVLink helps but isn't required for a small model.

## Setup

```bash
pip install vllm nixl                       # NIXL wheel for NVIDIA; see vLLM's NixlConnector guide
git clone --depth 1 https://github.com/vllm-project/vllm    # for the toy proxy script
MODEL=meta-llama/Llama-3.1-8B-Instruct      # or Qwen/Qwen3-0.6B to start

# prefill (producer) on GPU 0, decode (consumer) on GPU 1
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve $MODEL --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' &
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  vllm serve $MODEL --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &

# proxy: sends each request to the prefiller (max_tokens=1), then to the decoder
python vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200
```

(If a path or flag has moved, the vLLM "NixlConnector Usage Guide" has the current version.)

## Benchmark both setups

```bash
# prefill-heavy (RAG-like) and decode-heavy (chat/reasoning-like)
for IO in "4096 128" "256 1024"; do set -- $IO
  vllm bench serve --backend vllm --model $MODEL --base-url http://127.0.0.1:8192 \
    --dataset-name random --random-input-len $1 --random-output-len $2 \
    --num-prompts 300 --request-rate 4 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99
done
```

Then kill everything and run the **baseline**: two plain `vllm serve` instances (one per GPU, *no* `--enforce-eager` for fairness, or with it on both) behind a round-robin proxy — or simply `--data-parallel-size 2`. Sweep `--request-rate` (2, 4, 8, 16) for both setups.

## Record

| setup | workload | rate | TTFT p99 | ITL p99 | % requests meeting (TTFT < 2 s and ITL < 50 ms) |
|---|---|---|---|---|---|
| 1P1D | 4096/128 | | | | |
| 2× colocated | 4096/128 | | | | |
| 1P1D | 256/1024 | | | | |
| 2× colocated | 256/1024 | | | | |

> [!CHECK] What you should find
> Disaggregated should win on **ITL p99** for the prefill-heavy workload at higher rates. For the decode-heavy workload, 1P1D wastes most of the prefill GPU — 2 colocated replicas usually win. Small model + only 2 GPUs is *not* the regime where disaggregation shines; you are measuring the mechanism, not the production payoff.

- [ ] Both setups running; KV transfer confirmed in the logs
- [ ] Filled the table for at least 3 request rates
- [ ] Wrote 5 sentences on where 1P1D won, where it lost, and why
- [ ] Shut down the instance
      */}),
      resources: [
        { title: "vLLM — NixlConnector usage guide", url: "https://docs.vllm.ai/en/latest/features/nixl_connector_usage/", type: "docs", note: "current commands, env vars and proxy for 1P1D" },
      ],
    },
    {
      id: "moe-inference",
      title: "MoE inference: routing, expert parallelism, all-to-all, load balancing",
      kind: "concept",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
A **mixture-of-experts** layer replaces one big FFN with $E$ smaller expert FFNs and a tiny **router** that picks the top-$k$ experts for each token. DeepSeek-V3: **256 routed experts + 1 shared expert, top-8**, in 58 of its 61 layers → **671B total, 37B active** per token.

::viz moe-routing

## Why MoE is great and awkward for serving

- **Compute per token ≈ active params** (37B) → prefill FLOPs like a 37B model.
- **Memory ≈ total params** (671B) → must hold all experts somewhere.
- **Decode bandwidth:** at batch 1 you read only ~37B params per token. But at batch 256 with top-8 of 256, almost **every** expert is hit every step (Baseten ch. 2.2.4: "in batched production, expect almost all experts to be active"). So at serving batch sizes you read nearly all 671B per step — unless the experts are **spread across many GPUs**, each reading only its own.
- **Per-expert batch is small:** 256 tokens × 8 choices ÷ 256 experts = **8 tokens per expert** — tiny matmuls, memory-bound. To get efficient expert GEMMs you need a **huge global batch**. That's the core reason for large-scale EP.

## Expert parallelism, step by step

Each GPU owns $E/N$ experts; the router is **replicated**. For each MoE layer:

1. **Route:** each GPU computes top-$k$ for *its own* tokens.
2. **Dispatch (all-to-all #1):** send each token's hidden state to the GPUs owning its chosen experts.
3. **Expert compute:** each GPU runs its experts on the tokens it received (grouped GEMM).
4. **Combine (all-to-all #2):** send results back; each token sums its $k$ expert outputs weighted by the gates.

Communication scales with **tokens × hidden × k**, not with weights, and there's no per-layer all-reduce of the whole hidden state across the EP group — which is why EP stretches across nodes where TP can't. Attention is usually run **data-parallel** (each GPU handles its own requests — "DP attention") or with small TP.

::viz parallelism

## Build it: EP with `all_to_all_single` on your Mac

```python
# ep_moe.py — expert parallelism with all-to-all. run: torchrun --standalone --nproc_per_node=4 ep_moe.py
import torch, torch.nn.functional as F, torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                   # identical router + experts on every rank
C, E, K = 64, 8, 2                                     # hidden, experts, top-k
assert E % world == 0
E_local = E // world                                   # experts owned by this rank
router = torch.randn(C, E) / C**0.5                    # replicated on every rank
W_up = torch.randn(E, C, 4 * C) / C**0.5               # full copies only for the reference check
W_dn = torch.randn(E, 4 * C, C) / (4 * C)**0.5
my_experts = range(rank * E_local, (rank + 1) * E_local)

def expert(e, x):
    return F.gelu(x @ W_up[e]) @ W_dn[e]

# Each rank has its OWN tokens (think: its own requests, DP attention)
x = torch.randn(16, C, generator=torch.Generator().manual_seed(100 + rank))

# 1) route: top-k experts per token, softmax over the chosen scores
scores = x @ router
topv, topi = scores.topk(K, dim=-1)                    # (T, K)
gates = topv.softmax(-1)
tok_idx = torch.arange(x.shape[0]).repeat_interleave(K)   # which token each (token, k) slot is
exp_idx = topi.reshape(-1)
dest = exp_idx // E_local                              # which rank owns that expert
order = dest.argsort(stable=True)                      # group slots by destination rank
send_x, send_e = x[tok_idx[order]], exp_idx[order]
send_counts = torch.bincount(dest, minlength=world)

# 2) exchange counts, then DISPATCH tokens (all-to-all #1)
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)
recv_x = torch.empty(int(recv_counts.sum()), C)
dist.all_to_all_single(recv_x, send_x, recv_counts.tolist(), send_counts.tolist())
recv_e = torch.empty(int(recv_counts.sum()), dtype=torch.long)
dist.all_to_all_single(recv_e, send_e, recv_counts.tolist(), send_counts.tolist())

# 3) run MY experts on the tokens I received
out = torch.empty_like(recv_x)
for e in my_experts:
    m = recv_e == e
    out[m] = expert(e, recv_x[m])
load = torch.tensor([int((recv_e == e).sum()) for e in my_experts])

# 4) COMBINE: send results back (all-to-all #2), then weight by gates and sum over k
back = torch.empty_like(send_x)
dist.all_to_all_single(back, out, send_counts.tolist(), recv_counts.tolist())
y_slots = torch.empty_like(back)
y_slots[order] = back                                  # undo the sort
y = (y_slots.view(-1, K, C) * gates.unsqueeze(-1)).sum(1)

# reference: every expert computed locally
ref = torch.zeros_like(x)
for t in range(x.shape[0]):
    for k in range(K):
        ref[t] += gates[t, k] * expert(topi[t, k].item(), x[t:t + 1])[0]
print(f"rank {rank}: tokens-per-local-expert {load.tolist()}  max err {(y - ref).abs().max():.1e}", flush=True)
dist.destroy_process_group()
```

```text
rank 0: tokens-per-local-expert [5, 21]  max err 7.7e-07
rank 3: tokens-per-local-expert [15, 23]  max err 5.4e-07
```

Look at the loads: **5 vs 23** tokens. The slowest GPU sets the step time — that's **load imbalance**, the central problem of EP.

## Load balancing and capacity

- **Capacity factor** (GShard/Switch): each expert accepts at most $\text{cf} \times \frac{T \cdot k}{E}$ tokens per step; overflow tokens are **dropped** (skip the expert) or re-routed. Training uses cf ≈ 1.0–1.25. **Inference engines generally avoid dropping** (it changes outputs) and instead size buffers for the worst case — which costs memory and makes imbalance a *latency* problem.
- **Auxiliary-loss / bias balancing** at training time makes routing more even (DeepSeek-V3 uses an auxiliary-loss-free bias method), but real traffic still has **hot experts**.
- **EPLB** (Expert-Parallel Load Balancer, DeepSeek): **replicate** hot experts ("redundant experts") and re-pack experts onto GPUs from observed load statistics, periodically.
- **DeepEP** (DeepSeek): all-to-all kernels built for MoE — *normal* kernels (high throughput, NVLink + RDMA forwarding) for prefill and *low-latency* kernels (pure RDMA, CUDA-graph friendly) for decode, with FP8 dispatch. vLLM: `--all2all-backend deepep_high_throughput | deepep_low_latency`.

> [!REAL] Hybrid layouts you'll see
> **TP for attention + EP for experts** (Baseten ch. 5.4), or **DP attention + EP** (vLLM `--data-parallel-size 8 --enable-expert-parallel`; SGLang `--enable-dp-attention`). Example from the book: 128 experts at EP8 → 16 experts per GPU.

- [ ] Ran `ep_moe.py` at 2 and 4 ranks; errors ≈ 1e-7
- [ ] Printed the bytes sent in each all-to-all; derived the formula (tokens × k × C × bytes, both directions)
- [ ] Added a **capacity factor**: drop overflow tokens and measure how the output error vs reference grows as cf shrinks
- [ ] Added a **redundant expert**: replicate the hottest expert on two ranks, split its tokens, and show the max load drops
      */}),
      resources: [
        { title: "DeepEP", url: "https://github.com/deepseek-ai/DeepEP", type: "repo", note: "the MoE dispatch/combine kernels everyone now uses" },
        { title: "EPLB", url: "https://github.com/deepseek-ai/EPLB", type: "repo", note: "redundant experts + placement from load statistics" },
        { title: "Switch Transformers (Fedus et al., 2021)", url: "https://arxiv.org/abs/2101.03961", type: "paper", note: "top-1 routing and capacity factor" },
        { title: "GShard (Lepikhin et al., 2020)", url: "https://arxiv.org/abs/2006.16668", type: "paper", note: "expert parallelism with all-to-all at scale" },
        { title: "Hugging Face — Mixture of Experts explained", url: "https://huggingface.co/blog/moe", type: "article", note: "friendly overview of routing, load balancing and serving" },
      ],
    },
    {
      id: "deepseek-case-study",
      title: "Case study: how DeepSeek serves V3/R1 (MLA, MTP, large-scale EP, PD)",
      kind: "read",
      minutes: 90,
      md: MD(function () {/*
DeepSeek published its production inference architecture in February 2025 (Open Infra Index, "day 6"). It's the best public blueprint of frontier-scale serving. Read it now; this lesson is your annotated guide.

## The model features that shape serving

| Feature | What it is | Serving consequence |
|---|---|---|
| **MoE** 256 routed + 1 shared, top-8 | 671B total / 37B active | needs a huge batch per expert → **wide EP** |
| **MLA** (multi-head latent attention) | K,V compressed into a 512-dim latent (+64 RoPE dims) per token per layer | ~69 KiB/token KV in BF16 → huge contexts, cheap KV transfer. But one latent "head" **can't be split by TP** → use **DP attention** |
| **MTP** (multi-token prediction) | extra module trained to predict token $t+2$ | reused as a **speculative-decoding draft** (M15) with high acceptance |
| **FP8** training | weights and GEMMs in FP8 | serve in FP8 with **DeepGEMM**; no post-hoc quantization needed |

::viz gqa

## The deployment (H800 nodes, 8 GPUs each)

- **PD disaggregation.**
- **Prefill unit: 4 nodes (32 GPUs)** — routed experts **EP32**, MLA and shared expert **DP32**; each GPU holds **9 routed experts + 1 shared**; **32 redundant experts** for load balance.
- **Decode unit: 18 nodes (144 GPUs)** — routed experts **EP144**, MLA/shared **DP144**; each GPU holds **2 routed experts + 1 shared**; 32 redundant experts.
- **Computation–communication overlap:** prefill uses **dual micro-batch overlap** (compute one micro-batch while the other's all-to-all is in flight); decode splits attention into two steps and runs a **5-stage pipeline** to hide communication.
- **Three load balancers:** prefill (balance attention compute and tokens per GPU), decode (balance KV usage and request counts), and **expert-parallel** (EPLB: minimize the max dispatch-receive load).
- **Precision:** FP8 for GEMMs and dispatch, BF16 for core MLA and combine.

## The numbers (24 h, late February 2025)

- Peak **278 nodes**, average **226.75 nodes** (≈ 1,800 H800s).
- 608B input tokens/day, **56.3% hit the on-disk KV cache**; 168B output tokens/day.
- Per H800 node: **~73.7K input tok/s** in prefill (incl. cache hits), **~14.8K output tok/s** in decode; users see ~20–22 tok/s.
- At an assumed \$2/GPU-hour, cost ≈ \$87K/day; theoretical revenue at R1 prices ≈ \$562K/day ("545% cost-profit margin").

> [!INTUITION] Why 144 GPUs for decode?
> With 2 experts per GPU, each GPU reads only ~2 experts' weights per layer per step — so per-GPU weight traffic is tiny, leaving HBM bandwidth and capacity for **KV cache** and **big batches**. And with EP144, the global batch is huge, so each expert gets enough tokens for efficient GEMMs. Wide EP converts a bandwidth-bound problem into a batch-size problem.

> [!REAL] Open reproductions
> SGLang reproduced this on **12 nodes × 8 H100** (PD + large-scale EP with DeepEP, DeepGEMM, EPLB, two-batch overlap), reaching **52.3K input and 22.3K output tok/s per node** for 2,000-token inputs — up to 5× the output throughput of vanilla TP on the same hardware.

- [ ] Read the DeepSeek day-6 inference overview end to end
- [ ] Computed: tokens per expert per decode step if each of 144 GPUs runs a batch of 128 sequences with top-8 routing over 256 experts (+32 redundant)
- [ ] Explained why MLA pushes you to DP attention instead of TP attention (hint: what happens to KV capacity if 8 TP ranks each store the same latent?)
- [ ] Read the LMSYS large-scale EP post and listed its three biggest optimizations
      */}),
      resources: [
        { title: "DeepSeek-V3/R1 inference system overview", url: "https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md", type: "article", note: "the production blueprint: PD, EP32/EP144, overlap, load balancing, real numbers" },
        { title: "DeepSeek-V3 technical report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "MoE, MLA, MTP, FP8 — and the deployment section" },
        { title: "DeepSeek-V2 (introduces MLA)", url: "https://arxiv.org/abs/2405.04434", type: "paper", note: "where multi-head latent attention comes from" },
        { title: "LMSYS — Large-scale EP with SGLang", url: "https://lmsys.org/blog/2025-05-05-large-scale-ep/", type: "article", note: "open reproduction on 96 H100s with measured throughput" },
        { title: "DeepSeek profile data", url: "https://github.com/deepseek-ai/profile-data", type: "repo", note: "real traces of the overlap schedules" },
      ],
    },
    {
      id: "wide-ep",
      title: "Deep dive: wide-EP — DP attention, NVL72, and what breaks",
      kind: "deep",
      optional: true,
      minutes: 60,
      runsOn: ["multi-gpu"],
      md: MD(function () {/*
**Wide EP** means spreading experts over *many* GPUs (32–144+) so each holds only a few. It's how DeepSeek-class models are served at scale, and it's what vLLM, SGLang, TRT-LLM, Dynamo and llm-d all now advertise.

## The memory math that forces DP attention with MLA

Take one **8× H200** node (141 GB each, 1,128 GB total) serving DeepSeek-V3 in FP8 (~690 GB of weights, of which ~650 GB are routed experts).

| Layout | Weights per GPU | Free for KV per GPU (at ~90% util) | KV tokens per node (69 KiB/token) |
|---|---|---|---|
| **TP8** (attention + experts split 8 ways) | ~86 GB | ~35 GB | ~500K — the MLA latent is **replicated** on all 8 ranks, so the node holds one copy |
| **DP8 attention + EP8** | experts ~82 GB + replicated attention/shared/embeddings ~20 GB ≈ 100 GB | ~20 GB | ~2.3M — each GPU caches only its own requests |

*(Approximate, for reasoning — check against the "KV cache size" line in your engine's startup log.)* Even though DP attention **replicates** the non-expert weights, it wins ~4× on KV capacity because MLA's KV can't be sharded by heads. That's why vLLM's single-node recipe for DeepSeek-V3 on H200 is:

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
```

## Going wider

Across nodes, add `--data-parallel-size 16 --data-parallel-size-local 8` (one command per node, the others headless) and a DeepEP backend. On **GB200 NVL72** the whole EP group (72 GPUs) is one NVLink domain, so the all-to-all doesn't touch InfiniBand at all — the reason MoE serving benchmarks increasingly headline NVL72 numbers.

## What breaks at scale

- **Stragglers:** one hot expert or one DP rank with long sequences stalls the whole EP group every layer. → EPLB, redundant experts, KV-aware DP load balancing.
- **DP ranks must step together:** every rank must join every all-to-all, even with no requests (dummy batches). Idle ranks still burn time.
- **Prefill and decode want different EP layouts** (high-throughput vs low-latency all-to-all kernels, different batch shapes) → one more reason large MoE is almost always **disaggregated**.
- **Failure blast radius:** a 144-GPU decode unit is one failure domain. Plan for restarts, health checks and spare capacity.
- **Weight loading and startup:** hundreds of GB across dozens of nodes; cold starts take minutes (M18).

- [ ] Recomputed the table for 16× H100 (2 nodes, DP16 + EP16)
- [ ] Listed three metrics you'd put on a dashboard for a wide-EP deployment (hint: per-expert load, all-to-all time, DP-rank skew)
      */}),
      resources: [
        { title: "vLLM — Expert parallel deployment", url: "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html", type: "docs", note: "DP+EP flags, DeepEP backends, multi-node recipe" },
        { title: "SGLang v0.4 — DP attention", url: "https://lmsys.org/blog/2024-12-04-sglang-v0-4/", type: "article", note: "why data-parallel attention for MLA models" },
      ],
    },
    {
      id: "when-not",
      title: "When disaggregation is not worth it",
      kind: "concept",
      minutes: 50,
      md: MD(function () {/*
Disaggregation is fashionable. It is also more GPUs, more moving parts, a new failure mode (KV transfer), and a scheduling problem (xPyD balance). Default to **not** doing it, and make the data argue you into it.

## The Baseten checklist (Inference Engineering ch. 5.5)

Use disaggregation only when **all** hold:

1. **Volume:** roughly **100M–1B+ tokens/day**, depending on model size.
2. **Model size:** at least ~**100B parameters**.
3. **Prefill-heavy traffic:** long inputs (ISL), e.g. a frontier model inside a code editor.

If (1) or (2) fails you waste hardware; if (3) fails, **add replicas** instead.

## Why each condition matters

- **Low volume:** with one or two replicas you can't split into P and D pools without leaving one mostly idle. Minimum unit is now 2× what it was.
- **Small models:** a 8B model's prefill is so fast that interference is small; chunked prefill handles it. The transfer and routing overheads eat the gain.
- **Short prompts / high prefix-cache hit rate:** prefill is cheap already → **conditional disaggregation** routes these to decode locally anyway, so a static split would just waste the P pool.
- **Slow network:** a 100 GbE link between pools can add hundreds of ms to TTFT for large GQA models (see the math lesson).
- **Changing traffic mix:** the ideal xPyD drifts through the day; without a planner (Dynamo) you end up over-provisioned on one side.

## Alternatives to try first

| Problem | Try first |
|---|---|
| ITL spikes when long prompts arrive | **chunked prefill** with a tuned token budget (M09) |
| TTFT high due to repeated context | **prefix caching** + cache-aware routing (M08, M18) |
| Need more throughput | **replicas** + a better load balancer |
| Decode KV memory exhausted | KV quantization, offload (LMCache/KVBM), higher TP/EP |
| Mixed workloads | separate **deployments** per workload class (a "manual" disaggregation) |

## New bottlenecks once you do it

- **Prefill queue growth** → raise the local-prefill threshold or rebalance xPyD.
- **Decode KV exhaustion** (decode now holds *all* the long contexts) → KV quantization and offload.
- **Transfer failures / timeouts** → retries or local recompute; vLLM's NIXL connector has a KV **lease** so abandoned blocks get freed.

> [!REAL] A pragmatic take
> "Beyond the Buzz: A Pragmatic Take on Inference Disaggregation" (2025) runs large design-space sweeps and concludes, roughly, that disaggregation pays off mainly for **prefill-heavy traffic and larger models**, and that **dynamic rate matching and elastic scaling** of the P/D pools are essential to get the benefit.

- [ ] For your toy engine, found the prompt length below which colocated beats disaggregated on ITL p95 at your load (this is your challenge's first data point)
- [ ] Wrote a 5-bullet "should we disaggregate?" checklist for a team you're advising
      */}),
      resources: [
        { title: "Beyond the Buzz: A Pragmatic Take on Inference Disaggregation", url: "https://arxiv.org/abs/2506.05508", type: "paper", note: "when disaggregation actually pays off, from large sweeps" },
        { title: "Sarathi-Serve (chunked prefill)", url: "https://arxiv.org/abs/2403.02310", type: "paper", note: "the main alternative: stall-free batching with chunked prefills" },
      ],
    },
  ],

  challenge: {
    title: "When does disaggregation win? — a measured answer + a DeepSeek-V3 capacity plan",
    md: MD(function () {/*
**Part A — prototype measurement (Mac).** Extend your toy into a small **load test**: an arrival process (Poisson, rate $\lambda$) of requests with prompt length $P$ and output length $O$, served either by

- **colocated:** 2 workers, each doing its own prefills and decodes (chunked prefill optional), or
- **disaggregated:** 1 prefill worker + 1 decode worker with KV shipped over your socket/shared-memory path.

Sweep $P \in \{128, 512, 2048, 4096\}$ and $\lambda$ from light to overload. Report **TTFT p50/p99, ITL p50/p99 and goodput** (fraction of requests meeting TTFT < X and ITL < Y — pick SLOs that make sense on your machine). Produce a 2-D "who wins" map over (prompt length, load) and explain its boundary using your measured prefill time, transfer time and interference.

**Part B — design doc (`DESIGN.md`, 2–3 pages): DeepSeek-V3 on H200 nodes.** Workload: **2B input tokens/day** (50% prefix-cache hit), **300M output tokens/day**, peak-to-average **2.5×**, SLOs **TTFT p99 < 2 s, ITL p99 < 60 ms**.

1. Memory per node for DP8+EP8 vs TP8 (weights, KV capacity at MLA's ~69 KiB/token).
2. Peak rates: input (cache-miss) tok/s and output tok/s.
3. Throughput per node assumptions — cite DeepSeek's (73.7K in / 14.8K out per H800 node at EP32/EP144) and SGLang's (52.3K / 22.3K per H100 node) numbers, and explain why a **single-node** deployment will be well below them.
4. Two designs: **(a)** colocated single-node DP8+EP8 replicas, **(b)** 1–2 prefill nodes + decode nodes (xPyD). Node counts, \$/day at an assumed H200 price, headroom and failure plan.
5. Your recommendation, and the traffic level at which you'd switch from (a) to (b) — and to multi-node wide EP.
    */}),
    checklist: [
      "Load generator with Poisson arrivals drives both colocated and disaggregated toy setups",
      "TTFT/ITL p50/p99 and goodput reported for ≥ 4 prompt lengths × ≥ 3 load levels",
      "A (prompt length × load) map shows where disaggregation wins, with an explanation grounded in measured prefill/transfer times",
      "`DESIGN.md` computes DeepSeek-V3 memory and KV capacity per H200 node for DP8+EP8 and TP8",
      "Both designs have node counts, peak-rate math and a \\$/day estimate",
      "The doc states the traffic threshold where disaggregation and then wide EP become worth it",
    ],
    stretch: "On a 2-GPU cloud box, reproduce Part A with vLLM (1P1D via NixlConnector vs 2 colocated replicas) using Llama-3.1-8B and the same (prompt length × load) sweep — does the boundary move in the direction your toy predicted?",
  },

  connects: MD(function () {/*
M16 split the **model**; this module split the **request** (prefill vs decode) and the **experts**. Together they're how frontier models are served. Next, **M18** wraps these deployments in production machinery — routing (KV/prefix-aware, which is what makes conditional disaggregation work), autoscaling separate P and D pools, and cold starts for 700 GB models — and **M19** gives you the benchmarking and observability discipline (goodput, percentiles, per-expert load dashboards) to prove a layout is better. In **M23**, RL systems reuse the same ideas: rollout workers are inference engines, and weight sync is a broadcast/all-gather problem.
  */}),

  interview: [
    "What is prefill/decode disaggregation, and what problem does it solve that chunked prefill doesn't fully solve?",
    "Estimate the KV-cache transfer time for an 8K-token prompt on Llama-3.1-70B over InfiniBand NDR and over 100 GbE. Is it a problem?",
    "Define goodput. Why can a system with higher throughput have lower goodput?",
    "When would you NOT use disaggregated serving? Give concrete thresholds.",
    "Walk through one MoE layer under expert parallelism: which collectives, what data, and what dominates latency?",
    "Why does DeepSeek serve decode with EP144 (2 experts per GPU) rather than on a single 8-GPU node?",
    "Why do MLA models favour data-parallel attention over tensor-parallel attention?",
    "Your EP deployment's step time is dominated by one GPU. How do you diagnose and fix it?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 5.5 Disaggregation, ch. 2.2.4 MoE, ch. 4.4 Dynamo", url: "Inference%20Engineering.pdf", type: "book", note: "when to disaggregate, conditional disaggregation, xPyD, Dynamo features" },
    { title: "DistServe", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "disaggregation for goodput" },
    { title: "Splitwise", url: "https://arxiv.org/abs/2311.18677", type: "paper", note: "phase splitting for cost and power" },
    { title: "Mooncake", url: "https://arxiv.org/abs/2407.00079", type: "paper", note: "KVCache-centric disaggregated serving at Kimi" },
    { title: "DeepSeek-V3/R1 inference system overview", url: "https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md", type: "article", note: "the best public blueprint of large-scale PD + EP" },
    { title: "DeepSeek-V3 technical report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "MoE, MLA, MTP, FP8 details" },
    { title: "NVIDIA Dynamo docs", url: "https://docs.nvidia.com/dynamo/latest/index.html", type: "docs", note: "disaggregated serving, KV routing, planner" },
    { title: "DeepEP", url: "https://github.com/deepseek-ai/DeepEP", type: "repo", note: "MoE all-to-all kernels" },
    { title: "LMSYS — Large-scale EP with SGLang", url: "https://lmsys.org/blog/2025-05-05-large-scale-ep/", type: "article", note: "open reproduction of DeepSeek-style serving with numbers" },
    { title: "vLLM — Disaggregated prefilling", url: "https://docs.vllm.ai/en/latest/features/disagg_prefill.html", type: "docs", note: "connectors: NIXL, Mooncake, LMCache, offloading" },
    { title: "llm-d", url: "https://llm-d.ai/", type: "tool", note: "Kubernetes-native PD disaggregation and wide EP" },
  ],
});
