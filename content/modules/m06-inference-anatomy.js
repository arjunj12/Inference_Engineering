Course.module({
  id: "m06-inference-anatomy",
  title: "Anatomy of LLM inference: prefill, decode, KV cache",
  short: "Inference anatomy (KV cache)",
  tagline: "Add a KV cache to the GPT you built, watch generation get 5–20× faster at long lengths, and learn to measure inference like a professional: TTFT, TPOT, ITL, throughput, percentiles.",
  hours: 16,
  runsOn: ["mac", "colab"],
  tags: ["kv-cache", "prefill", "decode", "metrics", "sampling", "streaming", "benchmarking"],

  goal: MD(function () {/*
You will take the GPT you trained in Module 03 (and the Qwen-style model from Module 04), add a **KV cache**, and produce a benchmark table like this one — with a script *you* wrote:

```text
$ python bench_gen.py --model m03-gpt --prompt-len 64 --device mps
new_tokens  mode      TTFT(ms)  TPOT(ms)  E2E(s)   tok/s   speedup
   128      no-cache     4.0      5.1      0.66    194
   128      kv-cache     4.2      2.3      0.30    427     2.2x
   512      no-cache     4.0     14.9      7.61     67
   512      kv-cache     4.2      2.4      1.23    416     6.2x
  1984      no-cache     4.1     52.8    104.70     19
  1984      kv-cache     4.3      2.6      5.16    384    20.3x
```

(Illustrative numbers for a 6-layer, 384-wide GPT with `block_size=2048` on an M2 Pro — yours will differ. The *shape* is what matters: without a cache the per-token cost grows with length; with a cache it stays almost flat.)

Then you'll explain that curve in one sentence — *without a cache every new token re-processes the whole sequence, so total work is quadratic; with a cache each step only processes one token* — and extend the same benchmark to report the metrics every inference team lives by: **TTFT, TPOT/ITL, end-to-end latency, throughput**, with percentiles.

::viz generation-loop
  */}),
  demo: { viz: "generation-loop", params: {} },

  why: MD(function () {/*
Every serving engine — vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX — is organized around the two phases you meet here: **prefill** (process the prompt, fill the KV cache, emit the first token) and **decode** (one token per forward pass, reading the whole cache). Their schedulers, memory managers, kernels and metrics all come from this split. Interviewers for inference roles routinely ask you to derive the KV-cache size of a model, explain why TTFT and TPOT respond to different optimizations, or reason about p99 vs mean latency. After this module you can do all three from first principles — and you'll have code that proves it.
  */}),

  prereqs: [
    {
      title: "Your Module 03 GPT (and Module 04 Qwen) code",
      skipIf: "you have a trained m03 checkpoint and a working m04 Llama/Qwen implementation",
      md: MD(function () {/*
This module edits code you already own. You need:

- **m03**: Karpathy-style GPT (`token_embedding_table`, `position_embedding_table`, `blocks`, `ln_f`, `lm_head`) and ideally a saved `state_dict` (`torch.save(model.state_dict(), "gpt.pt")`).
- **m04**: your Llama/Qwen-style model with RMSNorm, RoPE, GQA and SwiGLU that matches Hugging Face logits for `Qwen/Qwen2.5-0.5B`.

If you skipped them, grab [`karpathy/ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) `gpt.py` and train it for 5 minutes on Shakespeare — the KV-cache lessons work with random weights too, you just won't get pretty text.
      */}),
    },
    {
      title: "Why 1 + 2 + … + T grows like T²",
      skipIf: "you know the triangle-number formula",
      math: true,
      md: MD(function () {/*
Without a cache, generating token number $t$ means running the model over all $t$ tokens seen so far. Total tokens processed for $T$ new tokens:

$$1 + 2 + 3 + \dots + T = \frac{T(T+1)}{2} \approx \frac{T^2}{2}$$

Pair the first and last terms ($1 + T$), the second and second-to-last ($2 + (T-1)$) … every pair sums to $T+1$ and there are $T/2$ pairs. Double $T$ → roughly **4×** the work. With a cache you process each token once: $T$ tokens total, so doubling $T$ doubles the work. For $T = 1000$ that's 500,500 token-forwards vs 1,000 — a 500× difference in token work. (Measured speedups are smaller, because at short lengths the chip is mostly waiting on memory and Python overhead, not doing that work — Lesson 2 explains why.)
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: time generation with and without a cache",
      kind: "demo",
      minutes: 45,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Before building anything, feel the problem with a real model. Hugging Face `generate()` has a `use_cache` switch, so you can switch the KV cache off and time it.

```python
# see_it.py — Qwen2.5-0.5B with and without the KV cache
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "mps" if torch.backends.mps.is_available() else "cuda"
sync = torch.mps.synchronize if dev == "mps" else torch.cuda.synchronize
mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()  # older transformers: torch_dtype=
inputs = tok("Write a very long story about a lighthouse keeper.", return_tensors="pt").to(dev)

model.generate(**inputs, max_new_tokens=8)            # warm-up (kernel compilation, allocator)
for n in [32, 128, 512, 1024]:
    for use_cache in [True, False]:
        sync(); t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=use_cache)
        sync(); dt = time.perf_counter() - t0
        print(f"{n:5d} tokens  cache={str(use_cache):5}  {dt:7.2f}s  {n / dt:6.1f} tok/s")
```

What you should see (shape, not exact numbers): with the cache, tok/s stays roughly **constant** as `n` grows. Without it, tok/s **falls** as `n` grows — at 1,024 tokens it's several times slower, and the gap keeps widening. (The 1,024-token no-cache run can take a couple of minutes. That *is* the lesson.)

> [!INTUITION] What's being wasted
> At step $t$ the no-cache model recomputes keys and values for tokens $1..t-1$ — **exactly the same numbers** it computed at step $t-1$. Causal attention means old tokens never look at new ones, so their K and V never change. Recomputing them is pure waste; caching them is free correctness-wise.

Now do the same with your **own** GPT. Your m03 `generate()` probably looks like Karpathy's:

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]     # the WHOLE context, every step
        logits, _ = self(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return idx
```

Two things are wrong for inference: (1) it re-runs the whole context every step, and (2) it computes logits for **every** position although only the last one is used. Time it for 64, 256 and 1,024 new tokens (build a fresh model with `block_size=2048` — random weights are fine for timing) and write the numbers down; you'll beat them in Lesson 3.

- [ ] Ran `see_it.py`; recorded tok/s with and without cache at 4 lengths
- [ ] Timed your m03 `generate()` at 64 / 256 / 1,024 tokens
- [ ] Can explain in one sentence why the no-cache curve bends downward
      */}),
      resources: [
        { title: "HF Transformers — KV cache strategies", url: "https://huggingface.co/docs/transformers/kv_cache", type: "docs", note: "DynamicCache, StaticCache, offloaded caches — what `use_cache` actually does" },
      ],
    },
    {
      id: "prefill-decode",
      title: "Prefill vs decode: two very different workloads",
      kind: "concept",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
Every request has two phases, and they stress the hardware in opposite ways.

::viz prefill-decode

| | **Prefill** | **Decode** |
|---|---|---|
| Input per forward pass | all $P$ prompt tokens at once | **1** new token per sequence |
| Output | KV cache for the prompt + first token | one more token + one more KV entry |
| Core math | matrix × matrix ($P \times d$ times $d \times d$) | vector × matrix ($1 \times d$ times $d \times d$) |
| Bottleneck | **compute** (FLOPs) | **memory bandwidth** (bytes of weights + KV read) |
| User-visible metric | **TTFT** (time to first token) | **TPOT / ITL** (time per output token) |
| Scales with | prompt length | output length × (weights + context) |

> [!INTUITION] The restaurant analogy
> The model weights are a giant cookbook in the basement (HBM / unified memory). **Prefill** is cooking 2,000 dishes: you fetch each page once and cook 2,000 dishes from it — the kitchen (compute) is the bottleneck. **Decode** is cooking 1 dish: you still have to fetch *every page* of the cookbook, but you cook just one dish per trip. The staircase (memory bandwidth) is the bottleneck, and the kitchen sits idle.

That's the key fact of LLM inference (Inference Engineering (Baseten) ch. 2.4): **prefill is compute-bound, decode is memory-bandwidth-bound.** You'll make it quantitative with the roofline in Module 07. For now, *see* it:

```python
# free_tokens.py — a forward pass over T tokens costs ~the same as over 1 token (until it doesn't)
import time, torch
from transformers import AutoModelForCausalLM
dev = "mps"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16).to(dev).eval()

@torch.no_grad()
def t_forward(T, iters=10):
    x = torch.randint(0, 1000, (1, T), device=dev)
    model(x); torch.mps.synchronize()                 # warm-up
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, use_cache=False)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters

for T in [1, 4, 16, 64, 256, 1024]:
    dt = t_forward(T)
    print(f"T={T:5d}  {1e3 * dt:7.1f} ms/pass   {1e3 * dt / T:7.3f} ms/token")
```

You'll see the time per pass barely moves from $T=1$ to $T\approx 32$–$128$, then starts growing roughly linearly. The flat region is the memory-bound regime: the chip spends the time streaming weights, and extra tokens ride along "for free". The linear region is compute-bound: now the arithmetic is the cost.

> [!IMPORTANT] Two consequences you'll exploit for the rest of the course
> 1. **Decode wastes the chip.** At batch 1 the tensor cores sit mostly idle. The fix is **batching** many users' decode steps into one forward pass (Module 09) — nearly free throughput.
> 2. **Prefill and decode interfere.** A long prefill in the same batch as 50 decoding users stalls all of them (ITL spikes). Fixes: **chunked prefill** (Module 09) and **disaggregated prefill/decode** on separate GPUs (Module 17).

> [!REAL] What engines print
> Ollama's `--verbose` shows `prompt eval rate` (prefill tok/s, often thousands) vs `eval rate` (decode tok/s, tens). mlx-lm prints `Prompt: … tokens-per-sec` vs `Generation: … tokens-per-sec`. vLLM's logs report `Avg prompt throughput` and `Avg generation throughput` separately. Prefill tok/s is typically 10–100× decode tok/s on the same hardware.

- [ ] Ran `free_tokens.py`; found where the ms/pass curve starts rising on your Mac
- [ ] Can explain why prefill is compute-bound and decode memory-bound using the words "weights read once per pass"
      */}),
    },
    {
      id: "kv-cache",
      title: "Build: a KV cache for your GPT (and your Qwen)",
      kind: "build",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
## What gets cached

In each attention layer every token produces a query $q$, key $k$ and value $v$. The new token's output is $\text{softmax}(q K^\top / \sqrt{d})\,V$ over **all** keys/values so far. Old tokens' $k, v$ never change (causal mask), so store them per layer:

$$\text{cache shape per layer: } [B,\; n_{kv\_heads},\; T_{max},\; d_{head}] \text{ for K, and the same for V}$$

We **preallocate** the buffers once (like `StaticCache` in HF and every production engine) instead of `torch.cat`-ing a growing tensor every step — concatenation copies the whole cache each step, which is quadratic again.

## Step 1 — refactor attention and add the cache

If your m03 attention is a list of `Head` modules, fuse them first: one `query`/`key`/`value` Linear of size `n_embd → n_embd` whose output is reshaped into heads. Mathematically identical, far fewer kernel launches.

```python
# gpt_kv.py
import torch, torch.nn as nn, torch.nn.functional as F

class KVCache:
    """Preallocated K/V buffers, one pair per layer, shape [B, n_head, max_len, head_dim]."""
    def __init__(self, n_layer, B, n_head, max_len, head_dim, device, dtype):
        shape = (B, n_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.pos = 0                                   # how many tokens are already stored

    def update(self, layer, k_new, v_new):
        T = k_new.size(2)
        end = self.pos + T
        self.k[layer][:, :, self.pos:end] = k_new       # write the new tokens' K/V in place
        self.v[layer][:, :, self.pos:end] = v_new
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]   # everything so far

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.n_head, self.layer_idx = n_head, layer_idx
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        hs = C // self.n_head
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)                  # now (B, nh, S, hs), S >= T
        S = k.size(2)
        if T == 1:
            mask = None                        # a single new query may attend to every cached key
        else:
            # query i sits at absolute position S-T+i and may see keys 0..S-T+i
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, layer_idx)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x, cache=None):
        x = x + self.sa(self.ln1(x), cache)
        return x + self.ffwd(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256):
        super().__init__()
        self.block_size, self.n_head, self.n_layer = block_size, n_head, n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, cache=None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        pos = torch.arange(start, start + T, device=idx.device)   # positions continue where the cache ends
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for blk in self.blocks:
            x = blk(x, cache)
        if cache is not None:
            cache.pos += T
        return self.lm_head(self.ln_f(x))
```

> [!WARNING] The three classic KV-cache bugs
> 1. **Positions**: the new token is at position `cache.pos`, not 0. Forgetting the offset gives fluent-looking garbage.
> 2. **The mask**: `is_causal=True` in SDPA assumes a *square* mask aligned top-left. As soon as the query length $T$ is smaller than the key length $S$ (decode, or chunked prefill) it's wrong. Hence `tril(diagonal=S-T)`.
> 3. **Advancing `pos` once per forward**, not once per layer.

To load your trained m03 checkpoint (per-head `Head` modules), concatenate the heads' weights — head $h$'s output columns are `h*hs:(h+1)*hs`, matching the `view(B, T, nh, hs)`:

```python
def load_m03_weights(model, sd):
    new = {k.replace(".ffwd.net.", ".ffwd."): v for k, v in sd.items()
           if ".sa.heads." not in k and not k.endswith(".tril")}
    for i in range(model.n_layer):
        for w in ("query", "key", "value"):
            new[f"blocks.{i}.sa.{w}.weight"] = torch.cat(
                [sd[f"blocks.{i}.sa.heads.{h}.{w}.weight"] for h in range(model.n_head)], dim=0)
    model.load_state_dict(new)
```

## Step 2 — prefill, then decode

```python
@torch.no_grad()
def generate_cached(model, idx, n_new):
    B, T = idx.shape
    assert T + n_new <= model.block_size, "learned position table is only block_size long"
    C = model.token_embedding_table.embedding_dim
    cache = KVCache(model.n_layer, B, model.n_head, T + n_new, C // model.n_head,
                    idx.device, model.lm_head.weight.dtype)
    logits = model(idx, cache)                        # PREFILL: whole prompt in one pass
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    for _ in range(n_new - 1):
        logits = model(nxt, cache)                    # DECODE: one token in, one token out
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat([idx] + out, dim=1)
```

## Step 3 — prove it's correct before timing it

A fast wrong cache is worse than no cache. Test that cached and uncached generation match exactly (greedy), and that prefilling in chunks gives the same logits as one pass:

```python
@torch.no_grad()
def generate_nocache(model, idx, n_new):             # Karpathy's loop, greedy
    for _ in range(n_new):
        logits = model(idx[:, -model.block_size:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx

def test_kv_cache():
    torch.manual_seed(0)
    m = GPT(vocab_size=65, block_size=512).eval()
    idx = torch.randint(0, 65, (2, 17))
    assert torch.equal(generate_nocache(m, idx, 100), generate_cached(m, idx, 100))
    seq = torch.randint(0, 65, (2, 60))
    full = m(seq)
    c = KVCache(m.n_layer, 2, m.n_head, 60, 64, "cpu", torch.float32)
    chunked = torch.cat([m(seq[:, :40], c), m(seq[:, 40:59], c), m(seq[:, 59:], c)], dim=1)
    assert (chunked - full).abs().max() < 1e-4       # ~1e-6 in fp32
```

Then time both on MPS with `block_size=2048` at 128/512/1,984 new tokens, calling `torch.mps.synchronize()` before reading the clock.

## Step 4 — the same for your m04 Qwen: RoPE offsets + GQA

Two differences for Llama/Qwen-style models:

```python
# inside your m04 attention forward(x, cos, sin, cache=None)
q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # cos/sin = table[start : start + T]
if cache is not None:
    k, v = cache.update(self.layer_idx, k, v)             # cache holds n_kv_heads, POST-RoPE keys
y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)  # PyTorch >= 2.5
# older PyTorch: k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1) (same for v)
```

- **RoPE offset**: slice the cos/sin tables at `start = cache.pos`. Keys are cached *after* rotation, so they never need re-rotating.
- **GQA**: the cache stores only `n_kv_heads` (2 for Qwen2.5-0.5B vs 14 query heads) — that's a 7× smaller cache, the whole point of GQA. Expand at attention time, never in the cache.

Validate against Hugging Face: run HF with `use_cache=True`, feeding `past_key_values=out.past_key_values` one token at a time, and compare your per-step logits (`atol≈1e-2` in fp16).

- [ ] `test_kv_cache` passes (exact greedy match + chunked-prefill logits match)
- [ ] m03 GPT: measured speedup at 128 / 512 / 1,984 new tokens
- [ ] m04 Qwen: KV-cached generation matches HF token-for-token for 100 greedy tokens
- [ ] Your cache for Qwen stores `n_kv_heads`, not `n_heads`
      */}),
      resources: [
        { title: "PyTorch scaled_dot_product_attention", url: "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", type: "docs", note: "attn_mask semantics, is_causal alignment, enable_gqa" },
        { title: "karpathy/ng-video-lecture", url: "https://github.com/karpathy/ng-video-lecture", type: "repo", note: "the m03 baseline you're modifying" },
      ],
    },
    {
      id: "flops-bytes",
      title: "Math: FLOPs and bytes per token, and the KV-cache size formula",
      kind: "math",
      minutes: 75,
      md: MD(function () {/*
> [!PREREQ] Refresher: counting FLOPs in a matmul
> Multiplying an $(m \times k)$ matrix by a $(k \times n)$ matrix produces $m \cdot n$ outputs, each a dot product of length $k$: $k$ multiplies and $k$ adds. So **FLOPs $= 2mkn$**. A single token (a $1 \times k$ vector) through a $k \times n$ weight matrix costs $2kn$ FLOPs — **2 FLOPs per weight**.

::viz matmul {"m":1,"k":6,"n":4}

## 1. Compute per token: ≈ 2N FLOPs

Every weight in every linear layer is used once per token (one multiply, one add). So for a model with $N$ parameters:

$$\text{FLOPs per token} \approx 2N \qquad(\text{+ attention: } \approx 4 \cdot L \cdot d_{model} \cdot T_{ctx})$$

The attention term is $QK^\top$ and $PV$ against the $T_{ctx}$ cached tokens: $2 \cdot T_{ctx} \cdot d$ each, per layer. Llama-3-8B ($N = 8.0$B, $L=32$, $d_{model}=4096$): $2N = 16$ GFLOP/token; at an 8K context attention adds $4 \times 32 \times 4096 \times 8192 \approx 4.3$ GFLOP (~27%). Prefill of $P$ tokens costs $\approx 2NP$: a 1,024-token prompt is **16.4 TFLOP** — about 17 ms on an H100 at its 989 TFLOPS BF16 peak. That's the floor on TTFT.

## 2. Bytes per decode step: all the weights, every step

A decode step at batch 1 must stream **every weight** from memory (plus the KV cache):

$$\text{bytes per decode step} \approx N \cdot b_{w} \;+\; T_{ctx} \cdot (\text{KV bytes per token})$$

Llama-3-8B in BF16 ($b_w = 2$): 16 GB per step. On an H100 (3.35 TB/s): $\frac{16}{3350} \approx 4.8$ ms → **≈ 200 tok/s upper bound** for one user, no matter how many TFLOPS the chip has. Arithmetic per step: 16 GFLOP in 4.8 ms ≈ 3.3 TFLOPS — ~0.3% of the H100's compute. That's what "memory-bound" means.

## 3. KV-cache size: the formula you must know cold

Per token, per layer, you store one key vector and one value vector for each **KV head**:

$$\boxed{\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_{head} \times b_{kv}}$$

(2 = K and V; $L$ layers; $H_{kv}$ KV heads; $d_{head}$ head dim; $b_{kv}$ bytes per element: 2 for FP16/BF16, 1 for FP8.) Multiply by tokens and by concurrent sequences for totals.

**Worked example — Llama-3-8B** ($L=32$, $H_{kv}=8$ with GQA, $d_{head}=128$, BF16):

| Quantity | Calculation | Result |
|---|---|---|
| per token | $2 \times 32 \times 8 \times 128 \times 2$ B | **131,072 B = 128 KiB** |
| one 8K-token conversation | $128\text{ KiB} \times 8192$ | **1 GiB** |
| one 128K-token context | $128\text{ KiB} \times 131072$ | **16 GiB** |
| 32 users × 8K | $32 \times 1$ GiB | **32 GiB** — twice the weights! |
| same model if it used full MHA ($H_{kv}=32$) | $4\times$ | 512 KiB/token |

Other models: Llama-3-70B (80 layers, 8 KV heads, 128) = **320 KiB/token** (a 128K context is 40 GiB); Qwen2.5-0.5B (24 layers, 2 KV heads, 64) = **12 KiB/token**; your m03 GPT (6 layers, 6 heads, 64, fp32) = 18 KiB/token.

::viz kv-calculator

> [!INTUITION] Why this formula runs the industry
> The KV cache is what limits **how many users fit on a GPU** (concurrency → throughput → cost), and at long context it's also a big chunk of the **bytes read per decode step** (latency). GQA, MLA, FP8 KV caches, sliding windows, paging and prefix caching (Module 08) all attack one factor of this formula.

## 4. Putting it together (napkin template)

$$\text{TTFT} \gtrsim \frac{2NP}{\text{FLOPS}}\qquad \text{TPOT} \gtrsim \frac{N b_w + T_{ctx}\cdot \text{KV}_{tok}}{\text{bandwidth}}$$

Real engines reach ~50–70% of peak FLOPS in prefill and ~60–85% of peak bandwidth in decode; Module 07 names these **MFU** and **MBU**.

- [ ] Computed KV bytes/token for Llama-3-8B, Qwen2.5-0.5B and your m03 GPT by hand; checked with the calculator
- [ ] Predicted the TPOT floor of Qwen2.5-0.5B (fp16, ~1 GB) on your Mac's bandwidth and compared to Lesson 1's measurement
- [ ] Computed how many 8K-token Llama-3-8B conversations fit in the 64 GB left on an H100 after weights (ignoring activations) — and noticed the answer is "fewer than you'd hope"
      */}),
      resources: [
        { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "the classic derivation of 2N FLOPs, KV size, and memory- vs compute-bound decode" },
      ],
    },
    {
      id: "metrics",
      title: "Math: TTFT, TPOT, ITL, E2E, throughput, goodput, percentiles & Little's law",
      kind: "math",
      minutes: 90,
      md: MD(function () {/*
## The per-request timeline

A streamed request has timestamps: sent $t_0$, first token $t_1$, then tokens $t_2 \dots t_n$.

| Metric | Definition | Driven by |
|---|---|---|
| **TTFT** | $t_1 - t_0$ | queueing + tokenization + prefill |
| **ITL** (inter-token latency) | each gap $t_{i+1} - t_i$ — a *distribution* | decode step time, batch interference |
| **TPOT** (time per output token) | $\dfrac{t_n - t_1}{n - 1}$ — one number per request | average decode step time |
| **E2E latency** | $t_n - t_0 = \text{TTFT} + (n-1)\cdot\text{TPOT}$ | everything |
| **Per-user TPS** | $\frac{1}{\text{TPOT}}$ (10 ms TPOT = 100 tok/s) | decode speed |
| **Throughput** | total output tokens/s across all requests (also: total tok/s incl. prompt, requests/s) | batching, hardware |
| **Goodput** | requests/s that **meet the SLO** (e.g. TTFT ≤ 500 ms *and* TPOT ≤ 50 ms) | the metric you actually sell |

TPOT and ITL are often used interchangeably; the difference matters when a stall happens. A single 800 ms pause mid-stream barely moves TPOT (it's averaged) but shows up as a huge ITL p99 — which is what a voice agent's user hears. The Baseten book (ch. 1.4) talks in TTFT, ITL and TPS; vLLM's benchmark prints TTFT, TPOT and ITL separately. Know all of them.

**Goodput** (from DistServe) exists because throughput lies: an engine can post huge tok/s by running giant batches where every request misses its latency target.

## Refresher: mean vs median vs p99

> [!PREREQ] Percentiles in 60 seconds
> Sort your $n$ measurements. The **p50 (median)** is the middle one — half the requests are slower. **p90**: 1 in 10 is slower. **p99**: 1 in 100 is slower. The **mean** is the sum divided by $n$ and gets dragged up by outliers.

Ten TTFTs in ms: `100 105 110 110 115 120 125 130 400 2000`.

- mean = 3315 / 10 = **331.5 ms** — describes *no* actual request
- median = (115 + 120) / 2 = **117.5 ms** — the typical user
- p90 ≈ **400 ms**, and "p99" ≈ 2000 ms — but with 10 samples p99 is just the max. You need **hundreds** of requests for a meaningful p99.

Latency distributions are **right-skewed** (long tail: GC pauses, a long prompt ahead in the queue, a preemption), so mean > median. At scale the tail *is* the user experience: a page that makes 10 LLM calls hits some call's p90 most of the time.

::viz percentiles

## Little's law: concurrency = throughput × latency

> [!MATH] The one queueing law you need
> In any stable system: **average number in the system $L$ = arrival rate $\lambda$ × average time in system $W$.** No assumptions about distributions. It's why a restaurant seating 40 with 1-hour meals serves at most 40 parties/hour.

For inference: if 4 requests/s arrive and each takes 12 s end to end (300 ms TTFT + 500 tokens × 23 ms), then on average **$4 \times 12 = 48$ requests are in flight**. Your engine must hold ~48 sequences in a batch — and 48 KV caches in memory. Flip it: if your GPU fits a batch of 32 at that latency, the most it can sustain is $\frac{32}{12} \approx 2.7$ req/s per replica. The tokens version: throughput (tok/s) ≈ concurrency × per-user tok/s.

> [!REAL] How production uses this
> Baseten's autoscaling advice (ch. 7.2) is to **set the replica's concurrency target equal to its batch size** — Little's law in disguise. When latency rises (longer outputs), the same traffic needs more concurrency → more replicas.

## Throughput vs latency is a trade-off, not a bug

Bigger batches raise total tok/s but slow each user (every decode step reads more KV and does more math). You'll plot this curve in Module 07 and pick the point that maximizes goodput under your SLO.

- [ ] Computed mean/median/p90 by hand for the 10-sample example; then with `numpy.percentile`
- [ ] Wrote `summarize(t0, stamps)` returning TTFT, TPOT, ITL p50/p99, E2E
- [ ] Used Little's law to size concurrency for 10 req/s, 800-token outputs at 25 ms TPOT
      */}),
      resources: [
        { title: "Etalon: holistic performance evaluation of LLM inference", url: "https://arxiv.org/abs/2407.07000", type: "paper", note: "why TTFT/TPOT averages hide stalls; fluidity-index metric" },
        { title: "DistServe (goodput)", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "defines goodput under TTFT/TPOT SLOs" },
      ],
    },
    {
      id: "sampling",
      title: "Sampling inside an engine: greedy, temperature, top-k/p, min-p, penalties, stops, logprobs",
      kind: "concept",
      minutes: 75,
      runsOn: ["mac"],
      md: MD(function () {/*
After each forward pass the engine has a logits vector of size $V$ (~150K for Qwen) **per sequence**, and must turn it into one token. Real engines run a pipeline of *logits processors*, then sample:

```text
raw logits → penalties (repetition/presence/frequency) → logit bias / grammar mask
           → temperature → top-k → top-p / min-p → softmax → sample → stop checks → detokenize
```

::viz softmax-temp

| Knob | What it does | Typical |
|---|---|---|
| **Greedy** (temperature 0) | argmax; deterministic* | evals, extraction, code |
| **Temperature** $\tau$ | $\text{softmax}(z/\tau)$: $\tau<1$ sharpens, $\tau>1$ flattens | 0.6–1.0 chat |
| **Top-k** | keep only the $k$ highest logits | 20–50 |
| **Top-p (nucleus)** | keep the smallest set with cumulative prob ≥ p | 0.9–0.95 |
| **Min-p** | drop tokens with prob < min_p × (top prob) — adapts to confidence | 0.05–0.1 |
| **Repetition penalty** | divide positive / multiply negative logits of seen tokens | 1.05–1.2 |
| **Stop** | stop token ids (EOS, `<|im_end|>`) and stop *strings* | per chat template |
| **Logprobs** | return log-probabilities of chosen (and top-n alternative) tokens | evals, RL, routing |

*Greedy on GPUs is not bit-for-bit reproducible across batch sizes: different batch shapes pick different kernels whose floating-point summation order differs, so near-ties can flip.

## Implement it (batched, in PyTorch)

```python
# sampler.py
import torch

def sample(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           repetition_penalty=1.0, prev_ids=None, generator=None):
    """logits: [B, V] scores for the NEXT token. Returns [B] token ids."""
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and prev_ids is not None:        # CTRL-style penalty
        seen = logits.gather(1, prev_ids)
        seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
        logits.scatter_(1, prev_ids, seen)
    if temperature == 0.0:
        return logits.argmax(dim=-1)                               # greedy
    logits = logits / temperature
    if top_k > 0:
        kth_best = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if min_p > 0.0:
        probs = logits.softmax(dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(dim=-1, keepdim=True).values, float("-inf"))
    if top_p < 1.0:
        probs = logits.softmax(dim=-1)
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        mass_before = sorted_p.cumsum(dim=-1) - sorted_p           # keep a token if the mass BEFORE it < p
        drop_sorted = mass_before >= top_p
        drop = drop_sorted.scatter(1, sorted_idx, drop_sorted)     # un-sort the mask
        logits = logits.masked_fill(drop, float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(-1)
```

Sanity tests: `top_k=1` and `top_p=1e-6` must both equal argmax; with `temperature=1` and no filters, the empirical frequencies over 10,000 samples should match `softmax(logits)`.

> [!IMPORTANT] Engine realities
> - **Per-request parameters in one batch.** Request A wants greedy, B wants $\tau=0.8$, top-p 0.9. Engines keep parameter *tensors* of shape `[B]` and apply them vectorized; sorting a 150K vocab for top-p every step is expensive enough that vLLM and FlashInfer ship dedicated sampling kernels.
> - **Stop strings are text, not tokens.** `"\n\nUser:"` can span several tokens and tokens can split UTF-8 characters. Engines **detokenize incrementally** and, when streaming, **hold back** text that might be the start of a stop string until it's resolved.
> - **Logprobs** are cheap (`log_softmax` of the logits you already have) but engines differ on whether they're reported before or after temperature/penalties — vLLM makes this configurable. RL training (Module 22) depends on getting this exactly right.
> - **Seeds**: per-request `generator` objects give reproducible sampling per request even inside a shared batch.

```python
# the same knobs in vLLM (Module 10 builds your own version of this API)
from vllm import SamplingParams
SamplingParams(temperature=0.7, top_p=0.9, top_k=40, min_p=0.05, repetition_penalty=1.1,
               max_tokens=256, stop=["\n\nUser:"], logprobs=5, seed=42)
```

- [ ] `sampler.py` passes the three sanity tests above
- [ ] Plugged `sample()` into `generate_cached`; generated Shakespeare at τ = 0.5, 1.0, 1.5 and compared
- [ ] Implemented a stop-string check with hold-back for streaming
      */}),
      resources: [
        { title: "The Curious Case of Neural Text Degeneration (top-p)", url: "https://arxiv.org/abs/1904.09751", type: "paper", note: "why pure sampling and greedy both degrade; nucleus sampling" },
        { title: "Min-p sampling", url: "https://arxiv.org/abs/2407.01082", type: "paper", note: "confidence-scaled truncation; now default-available in most engines" },
        { title: "HF — Generation strategies", url: "https://huggingface.co/docs/transformers/generation_strategies", type: "docs", note: "reference semantics for every knob" },
      ],
    },
    {
      id: "streaming",
      title: "Build: stream tokens over SSE (OpenAI-style) and measure from the client",
      kind: "build",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
Users perceive **TTFT and ITL**, not E2E — as long as you **stream**. The de-facto protocol for text is HTTP streaming with **Server-Sent Events (SSE)**: a normal HTTP response with `Content-Type: text/event-stream` that the server keeps writing lines to:

```text
data: {"choices":[{"index":0,"text":"To"}]}

data: {"choices":[{"index":0,"text":" be"}]}

data: [DONE]

```

Each event is `data: <json>` followed by a **blank line**. OpenAI, vLLM, SGLang, TRT-LLM and llama-server all speak this format on `/v1/completions` and `/v1/chat/completions`.

## Step 1 — make generation a Python generator

```python
@torch.no_grad()
def generate_stream(model, idx, n_new):
    C = model.token_embedding_table.embedding_dim
    cache = KVCache(model.n_layer, 1, model.n_head, idx.size(1) + n_new, C // model.n_head,
                    idx.device, model.lm_head.weight.dtype)
    logits = model(idx, cache)                          # prefill
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        yield nxt.item()                                # .item() syncs GPU -> CPU: the token really exists now
        logits = model(nxt, cache)                      # decode
```

## Step 2 — a tiny OpenAI-compatible streaming server

```python
# server.py   (pip install fastapi uvicorn)   run: uvicorn server:app --port 8000
import json, torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gpt_kv import GPT, load_m03_weights, generate_stream   # your module
from data import encode, decode                            # your m03 char tokenizer

model = GPT(vocab_size=65).to("mps").eval()
load_m03_weights(model, torch.load("gpt.pt", map_location="mps"))
app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    stream: bool = True

def sse(req):
    idx = torch.tensor([encode(req.prompt)], device="mps")
    for tok in generate_stream(model, idx, req.max_tokens):
        chunk = {"object": "text_completion", "choices": [{"index": 0, "text": decode([tok])}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    return StreamingResponse(sse(req), media_type="text/event-stream")
```

Try it: `curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"ROMEO:","max_tokens":100}'` (`-N` disables curl's buffering).

## Step 3 — measure from the client, like a user would

```python
# client_bench.py
import json, time, statistics, httpx

def one(prompt, max_tokens=200, url="http://localhost:8000/v1/completions"):
    t0, stamps = time.perf_counter(), []
    with httpx.stream("POST", url, json={"prompt": prompt, "max_tokens": max_tokens}, timeout=None) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            stamps.append(time.perf_counter())
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft_ms": 1e3 * (stamps[0] - t0),
            "tpot_ms": 1e3 * (stamps[-1] - stamps[0]) / (len(stamps) - 1),
            "itl_p99_ms": 1e3 * sorted(gaps)[int(0.99 * (len(gaps) - 1))],
            "e2e_s": stamps[-1] - t0}

runs = [one("ROMEO:") for _ in range(20)]
for k in runs[0]:
    print(k, "p50 =", round(statistics.median(r[k] for r in runs), 2))
```

The same client works against vLLM/llama-server/OpenAI if you switch to `/v1/chat/completions` and read `choices[0].delta.content`. Pass `"stream_options": {"include_usage": true}` to get exact token counts in the final chunk instead of counting chunks.

> [!WARNING] Streaming gotchas in production
> - **Buffering proxies** (nginx, some load balancers) collect the whole response before forwarding: TTFT silently becomes E2E. Disable buffering (e.g. nginx `proxy_buffering off` or the `X-Accel-Buffering: no` header).
> - **Detokenize incrementally.** Decoding one token id at a time breaks multi-byte characters and word-piece spacing for BPE tokenizers; decode the running sequence and emit only the new suffix.
> - **Client disconnects** must abort generation and free the KV cache — otherwise abandoned requests keep burning GPU (Module 10).
> - Chat is fine over SSE; bidirectional audio uses WebSockets, service-to-service often gRPC (Inference Engineering (Baseten) ch. 7.5).

- [ ] `curl -N` shows tokens arriving one by one
- [ ] `client_bench.py` reports TTFT/TPOT/ITL p99 for 20 requests against your server
- [ ] Same client, pointed at Ollama's OpenAI endpoint (`http://localhost:11434/v1/chat/completions`), works unchanged except for the payload
      */}),
      resources: [
        { title: "MDN — Server-sent events", url: "https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events", type: "docs", note: "the wire format and its rules" },
      ],
    },
    {
      id: "bench-lab",
      title: "Lab: benchmark HF transformers vs llama.cpp/Ollama vs MLX on your Mac",
      kind: "lab",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
Same model, same prompt, four engines, one script. You'll see how much of "speed" is the engine and how much is the precision.

**Setup** (model: Qwen2.5-1.5B-Instruct):

```bash
uv pip install torch transformers accelerate mlx-lm requests
ollama pull qwen2.5:1.5b-instruct-fp16            # also try :1.5b-instruct-q8_0 and the default Q4_K_M
brew install llama.cpp
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0 --port 8080 -c 4096   # OpenAI-compatible server
```

```python
# bench_engines.py
import json, statistics, time, requests

PROMPT = "Write a detailed 600-word story about a lighthouse keeper who discovers a message in a bottle."
N_OUT, RUNS = 256, 5

def summarize(engine, t0, stamps, n_prompt):
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    e2e, ttft = stamps[-1] - t0, stamps[0] - t0
    return {"engine": engine, "prompt_toks": n_prompt, "out_toks": len(stamps),
            "ttft_ms": 1e3 * ttft, "tpot_ms": 1e3 * (e2e - ttft) / max(1, len(stamps) - 1),
            "itl_p99_ms": 1e3 * sorted(gaps)[int(0.99 * (len(gaps) - 1))], "tok_s": len(stamps) / e2e}

def bench_hf(model_id="Qwen/Qwen2.5-1.5B-Instruct", device="mps"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.streamers import BaseStreamer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    class Stamps(BaseStreamer):                       # generate() calls put() once per new token
        def __init__(self): self.t, self.first = [], True
        def put(self, value):
            if self.first: self.first = False; return  # the first call carries the prompt
            self.t.append(time.perf_counter())
        def end(self): pass

    def once():
        s, t0 = Stamps(), time.perf_counter()
        model.generate(ids, max_new_tokens=N_OUT, min_new_tokens=N_OUT, do_sample=False, streamer=s)
        return summarize("hf-transformers fp16", t0, s.t, ids.shape[1])
    once()                                            # warm-up
    return [once() for _ in range(RUNS)]

def bench_mlx(model_id="Qwen/Qwen2.5-1.5B-Instruct", label="mlx bf16"):
    from mlx_lm import load, stream_generate
    model, tok = load(model_id)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    def once():
        t0, stamps = time.perf_counter(), []
        for resp in stream_generate(model, tok, prompt, max_tokens=N_OUT):   # greedy by default
            stamps.append(time.perf_counter())
        return summarize(label, t0, stamps, resp.prompt_tokens)
    once()
    return [once() for _ in range(RUNS)]

def bench_openai(base_url, model, label):
    """Any OpenAI-compatible streaming server: llama-server, Ollama (/v1), vLLM, SGLang."""
    def once():
        t0, stamps, usage = time.perf_counter(), [], {}
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": N_OUT,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        with requests.post(f"{base_url}/v1/chat/completions", json=body, stream=True) as r:
            for line in r.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"): usage = chunk["usage"]
                if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                    stamps.append(time.perf_counter())
        return summarize(label, t0, stamps, usage.get("prompt_tokens", -1))
    once()
    return [once() for _ in range(RUNS)]

if __name__ == "__main__":
    results = []
    results += bench_openai("http://localhost:11434", "qwen2.5:1.5b-instruct-fp16", "ollama fp16")
    results += bench_openai("http://localhost:11434", "qwen2.5:1.5b-instruct-q8_0", "ollama q8_0")
    results += bench_openai("http://localhost:8080", "any", "llama-server q8_0")
    results += bench_mlx()
    results += bench_mlx("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "mlx 4bit")
    results += bench_hf()
    by = {}
    for r in results: by.setdefault(r["engine"], []).append(r)
    print(f"{'engine':24} {'TTFT ms':>8} {'TPOT ms':>8} {'ITLp99':>8} {'tok/s':>7}")
    for eng, rs in by.items():
        med = lambda k: statistics.median(r[k] for r in rs)
        print(f"{eng:24} {med('ttft_ms'):8.1f} {med('tpot_ms'):8.2f} {med('itl_p99_ms'):8.2f} {med('tok_s'):7.1f}")
```

> [!NOTE] Counting tokens honestly
> Most servers send one token per SSE chunk, but not all — some batch tokens, and a chunk can be empty text for a partial UTF-8 character. `include_usage` gives exact counts; compare `out_toks` with `usage.completion_tokens` once. Also: Ollama and llama-server may stop at EOS before 256 tokens (hence the "600-word" prompt) — always report `out_toks`.

## What to look for

1. **Precision dominates decode speed.** fp16 → q8 → 4-bit should give roughly 1× → 1.7× → 2.5–3× decode tok/s. Bytes per token, again (Module 00's napkin law; Module 14).
2. **Engine overhead matters for small models.** HF `generate()` in eager PyTorch on MPS is usually the slowest at equal precision: per-token Python and kernel-launch overhead is comparable to the actual work for a 1.5B model. llama.cpp and MLX are lean C++/Metal loops. (Module 13 fixes this for PyTorch with `torch.compile` and CUDA graphs.)
3. **TTFT ≠ decode speed.** Prefill throughput (thousands of tok/s) ranks engines differently from decode.

> [!REAL] Apples to apples
> A benchmark comparing engines at *different precisions* is marketing, not engineering. State model, precision, prompt/output lengths, runs, and hardware — the same checklist the Baseten book (ch. 4.5) gives for load tests.

- [ ] Table with ≥ 5 engine/precision rows, median of 5 runs each
- [ ] For each row: % of your Mac's peak bandwidth achieved in decode (weight bytes × tok/s ÷ bandwidth)
- [ ] One paragraph explaining the ranking
      */}),
      resources: [
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "`stream_generate`, prompt/generation tps, quantized community models" },
        { title: "llama.cpp server", url: "https://github.com/ggml-org/llama.cpp/tree/master/tools/server", type: "docs", note: "llama-server flags and its OpenAI-compatible API" },
        { title: "Ollama qwen2.5 tags", url: "https://ollama.com/library/qwen2.5/tags", type: "tool", note: "pick fp16 / q8_0 / q4 variants of the same model" },
      ],
    },
    {
      id: "real-caches",
      title: "Deep dive: how real engines lay out the KV cache",
      kind: "deep",
      optional: true,
      minutes: 60,
      md: MD(function () {/*
Your `KVCache` is one contiguous preallocated buffer per sequence. That's the simplest design; production engines pick different trade-offs:

| Engine | Layout | Trade-off |
|---|---|---|
| **HF `DynamicCache`** (default) | per-layer tensors grown with `torch.cat` each step | flexible; copies the cache every step; shapes change → no CUDA graphs |
| **HF `StaticCache`** (`cache_implementation="static"`) | preallocated `[B, H_kv, max_len, d]` + `cache_position` indices | fixed shapes → works with `torch.compile` / CUDA graphs; wastes memory up to `max_len` |
| **MLX `KVCache`** | grows in chunks of 256 tokens | amortized growth, few copies; `RotatingKVCache` for sliding windows |
| **llama.cpp** | one KV buffer sized by `-c/--ctx-size`, shared by parallel slots | simple, predictable; context is carved up among `--parallel` sequences |
| **vLLM / SGLang / TRT-LLM** | a global pool of fixed-size **blocks/pages**; per-sequence **block tables** | near-zero fragmentation, sharing across requests (prefix caching); needs paged-attention kernels |

The last row is the whole of Module 08. The reason it wins: with contiguous per-sequence buffers you must reserve `max_len` for each request up front because you don't know how long it will generate. Most of that reservation is never used, so far fewer requests fit — and concurrency is throughput (Little's law).

<details><summary>Why `torch.cat` caches are secretly quadratic</summary>

Appending one token to a `[1, 8, T, 128]` tensor with `torch.cat` allocates a new `[1, 8, T+1, 128]` tensor and copies all $T$ old entries. Over $T$ steps that's $\sum t \approx T^2/2$ copied entries per layer — the same triangle number you started this module with, just moved from FLOPs to memory traffic. It's usually hidden at short lengths because copies are fast, but it shows up in profiles at 32K+ contexts.

</details>

**Reading exercise**: open `mlx_lm/models/cache.py` in the mlx-lm repo and find `KVCache.update_and_fetch`. Compare it with your `update()`. Then open HF's `cache_utils.py` and find how `StaticCache` uses `cache_position` to write in place (`index_copy_`).

- [ ] Found the 256-token `step` in MLX's `KVCache` and explained why it exists
- [ ] Ran HF generation with `cache_implementation="static"` and compared speed to the default on your Mac
      */}),
      resources: [
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "mlx_lm/models/cache.py — compact, readable cache implementations" },
      ],
    },
  ],

  challenge: {
    title: "KV-cached generation for your GPT + a benchmark report that explains the curve",
    md: MD(function () {/*
Put it together in `course-work/m06/`:

1. `gpt_kv.py`: your m03 GPT with a preallocated KV cache, loading your trained weights, plus `test_kv_cache.py` (exact greedy match vs no-cache; chunked-prefill logits match).
2. The same for your m04 Qwen (RoPE offsets, GQA cache), validated token-for-token against HF.
3. `bench_gen.py`: for output lengths **{32, 64, 128, 256, 512, 1024, 1984}**, both modes, ≥ 5 runs each, report **TTFT, TPOT, ITL p50/p99, E2E, tok/s** and speedup. Use `torch.mps.synchronize()` (or CUDA sync) correctly.
4. A plot: E2E latency vs output length, both modes, on one chart. Add your prediction curves: no-cache ≈ $a \cdot T^2$, cache ≈ $b \cdot T$ (fit $a, b$ from two points).
5. `REPORT.md`: *why* is the no-cache curve quadratic? Why is the speedup small at short lengths (hint: at small $T$ both modes are memory/overhead-bound, and a 64-token forward costs about the same as a 1-token one)? What would change on an H100? Include the KV-cache size of both models at their max context.
    */}),
    checklist: [
      "KV-cached and uncached greedy outputs are identical (test passes)",
      "Qwen implementation caches only `n_kv_heads` and matches HF for 100 greedy tokens",
      "Benchmark table covers ≥ 7 output lengths × 2 modes with TTFT, TPOT, ITL p99, E2E, tok/s",
      "≥ 5× speedup demonstrated at the longest length on your hardware",
      "Plot shows measured points and fitted T² / T curves",
      "REPORT.md explains the curve and the short-length regime in your own words",
    ],
    stretch: "Batch it: run `generate_cached` with B = 1, 4, 16, 64 prompts at once and plot total tok/s and per-user tok/s vs B. You've just discovered why batching is the #1 throughput lever — Module 09 builds the scheduler that makes it work for requests of different lengths.",
  },

  connects: MD(function () {/*
You now own the three core objects of inference: the **prefill/decode split**, the **KV cache**, and the **metrics**. Module 07 explains *why* decode is memory-bound with the roofline model, so you can predict tok/s for any GPU before renting it. Module 08 replaces your contiguous cache with a **paged** one (blocks, block tables, prefix caching) — the core of vLLM. Module 09 batches many sequences through one forward pass with a scheduler, and Module 10 wraps it all in your own OpenAI-compatible engine using the sampler and SSE server from this module.
  */}),

  interview: [
    "Derive the KV-cache size per token for Llama-3-8B in BF16. How much memory do 64 concurrent 4K-token conversations need?",
    "Why is prefill compute-bound and decode memory-bound? What happens to each as batch size grows?",
    "A user complains that the model 'freezes' mid-answer, but your average TPOT dashboard looks fine. What metric would reveal the problem, and what might cause it?",
    "Your service receives 20 req/s, and requests average 6 s end to end. How many concurrent sequences must the fleet hold? What if outputs get 2× longer?",
    "Explain TTFT vs TPOT vs E2E and which optimizations move each (e.g. quantization, prefix caching, speculative decoding, bigger batches).",
    "Why can't you use `is_causal=True` in SDPA during decode or chunked prefill with a KV cache?",
    "How do top-p and min-p differ, and why do engines implement sampling with custom kernels?",
    "Why does greedy decoding sometimes give different outputs at batch size 1 vs 32 on the same GPU?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 1.4 & 2.2–2.4", url: "Inference%20Engineering.pdf", type: "book", note: "metrics, prefill/decode mechanics, bottleneck math" },
    { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "FLOPs, bytes and KV-cache napkin math" },
    { title: "JAX Scaling Book — Inference", url: "https://jax-ml.github.io/scaling-book/inference/", type: "book", note: "rigorous treatment of prefill vs decode costs" },
    { title: "Databricks — LLM Inference Performance Engineering", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "TTFT/TPOT, MBU, batching trade-offs" },
    { title: "HF Transformers — KV cache strategies", url: "https://huggingface.co/docs/transformers/kv_cache", type: "docs", note: "Dynamic/Static/offloaded caches" },
    { title: "vLLM benchmark CLI", url: "https://docs.vllm.ai/en/latest/benchmarking/cli/", type: "docs", note: "`vllm bench serve` reports TTFT/TPOT/ITL percentiles — the industry-standard output format" },
    { title: "Etalon (LLM inference evaluation)", url: "https://arxiv.org/abs/2407.07000", type: "paper", note: "metrics beyond averages" },
    { title: "Min-p sampling", url: "https://arxiv.org/abs/2407.01082", type: "paper", note: "modern truncation sampling" },
    { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "your fastest local engine on Apple Silicon" },
    { title: "llama.cpp", url: "https://github.com/ggml-org/llama.cpp", type: "repo", note: "C++ engine with Metal backend; llama-server" },
  ],
});
