Course.module({
  id: "m15-speculative-decoding",
  title: "Speculative decoding: guessing ahead",
  short: "Speculative decoding",
  tagline: "Let a cheap model guess several tokens and the big model check them all in one pass — implement it yourself with exact rejection sampling, prove it doesn’t change the output distribution, and measure when it pays off.",
  hours: 18,
  level: "core",
  runsOn: ["mac", "colab", "cloud"],
  tags: ["speculative-decoding", "eagle", "mtp", "n-gram", "rejection-sampling", "latency"],

  goal: MD(function () {/*
You write `specdec.py` — ~150 lines of PyTorch — where `Qwen2.5-0.5B-Instruct` drafts tokens for `Qwen2.5-3B-Instruct` on your Mac. It prints something like:

```text
target: Qwen2.5-3B-Instruct (bf16, MPS)   draft: Qwen2.5-0.5B-Instruct   task: "refactor this function"   T=0
k   accepted/drafted   tokens/step   tok/s    speedup   output == plain greedy?
0        —               1.00        21.4     1.00×     —
2      0.78              2.39        34.7     1.62×     ✓
4      0.74              2.99        33.8     1.58×     ✓
8      0.71              3.29        25.9     1.21×     ✓
sampling test (T=1, 200k draws of the first token): chi² p = 0.62 → indistinguishable from the target ✓
(illustrative numbers; yours will differ)
```

Then you run vLLM’s built-in speculators (n-gram, a draft model, EAGLE-3) on a GPU, watch the speedup **shrink as concurrency grows**, and write down exactly when you would turn speculation off.

The interactive below is the whole idea in one picture: acceptance rate $\alpha$, draft length $k$ and draft cost decide how many tokens each expensive target pass produces.
  */}),
  demo: { viz: "spec-decode" },

  why: MD(function () {/*
Decode is memory-bound: at low batch the GPU spends most of each step waiting on weight reads while its compute units idle. Speculative decoding turns that idle compute into extra tokens per step — typically **1.5–3× lower inter-token latency** with **mathematically identical output** (Inference Engineering (Baseten) ch. 5.2). It’s in every serious engine (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX), frontier labs ship models with built-in drafters (DeepSeek-V3’s multi-token prediction), and latency-critical products — code completion, voice agents, agentic loops — lean on it. It’s also a favourite interview topic because it mixes systems (rooflines, KV caches, batching) with a clean probability proof. You’ll own both halves.
  */}),

  prereqs: [
    {
      title: "KV cache mechanics (recap)",
      skipIf: "you built the KV cache in M06 and the paged block manager in M08",
      md: MD(function () {/*
- The KV cache stores each layer’s keys and values for every token already processed, so a decode step only computes the **new** token’s Q/K/V and attends to the cached ones.
- Feeding a model **several** new tokens at once (like a mini-prefill) returns logits for **each** of them — position $i$’s logits are the prediction for the token *after* position $i$. Speculative decoding verification is exactly this.
- **Rolling back** = forgetting the last $n$ cached positions. In Hugging Face this is `DynamicCache.crop(...)`; in a paged engine it’s decrementing the sequence length and freeing now-empty blocks.

```python
from transformers import DynamicCache
cache = DynamicCache()
model(input_ids=prompt, past_key_values=cache, use_cache=True)   # cache now holds len(prompt) positions
print(cache.get_seq_length())
cache.crop(-3)                                                     # drop the last 3 positions
```
      */}),
    },
    {
      title: "Decode is memory-bound (recap)",
      skipIf: "you did the roofline module (M07) and M14’s why-faster lesson",
      math: true,
      md: MD(function () {/*
One decode step for one sequence reads **all** weights (e.g. 7.6B params × 2 bytes ≈ 15 GB for Qwen2.5-7B in BF16) but does only ~2 FLOPs per weight. On an H100 that’s $15\text{ GB} \div 3.35\text{ TB/s} \approx 4.5$ ms of memory time vs $15\text{ GFLOP} \div 989\text{ TFLOPS} \approx 0.015$ ms of math. The compute units are >99% idle. Processing 5 tokens instead of 1 reads the **same** weights and does 5× the (still tiny) math — so it costs almost the same time. Hold on to that; it’s the whole trick.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: --draft-model speedups on your Mac (MLX and llama.cpp)",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
## MLX-LM: one flag

```bash
uv pip install -U mlx-lm
M=mlx-community/Qwen2.5-7B-Instruct-4bit        # target (~4.3 GB)
D=mlx-community/Qwen2.5-0.5B-Instruct-4bit      # draft  (~0.3 GB), same tokenizer family
cat > code_prompt.txt <<'EOF'
Rename the variable `data` to `records` everywhere and add type hints. Return the full file.

import json
def load(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for row in data:
        if row.get("active"):
            out.append({"id": row["id"], "name": row["name"].strip()})
    return out
EOF
mlx_lm.generate --model $M --prompt "$(cat code_prompt.txt)" --max-tokens 300 --temp 0
mlx_lm.generate --model $M --draft-model $D --num-draft-tokens 3 --prompt "$(cat code_prompt.txt)" --max-tokens 300 --temp 0
```

Compare the **Generation tok/s** lines. Then repeat with an open-ended prompt (“Write a short story about a lighthouse keeper”) and with `--temp 1.0`. Record:

| Prompt | Temp | Draft? | num-draft-tokens | Gen tok/s | Speedup |
|---|---|---|---|---|---|
| code edit | 0 | no | — | | 1.0× |
| code edit | 0 | yes | 3 | | |
| code edit | 0 | yes | 6 | | |
| story | 0 | yes | 3 | | |
| story | 1.0 | yes | 3 | | |

Expect: the code-edit prompt speeds up the most (the output largely copies the input, so the small model guesses right), the story less, and temperature 1.0 least.

## llama.cpp: the same idea, via the server

```bash
brew install llama.cpp
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8081 &                      # baseline
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M \
             -hfd bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --draft-max 8 --port 8082 &     # with draft
```

(`--draft-max` is also spelled `--spec-draft-n-max` in newer builds; run `llama-server --help | grep -i draft`.) Send the same prompt to both ports (your M00 `bench.py` works against the OpenAI-compatible `/v1/chat/completions`), compare decode tok/s, and read the server log: it reports how many drafted tokens were accepted.

> [!INTUITION] What’s happening
> The 0.5B model runs ~6–10× faster than the 7B. It guesses the next few tokens; the 7B then checks all of them in **one** forward pass (costing about the same as generating one token), keeps the longest correct prefix, and adds one token of its own. If the guesses are usually right, you get several tokens for the price of one big step.

> [!NOTE] Two ways to verify
> MLX-LM verifies by sampling the target’s own token at each position and checking whether it **equals** the draft token. llama.cpp and vLLM use **rejection sampling** (next lessons), which accepts more often at temperature > 0. Both are exact — the output distribution is the target’s.

- [ ] Filled the table for MLX; best speedup ≥ 1.5× on the code prompt
- [ ] Same comparison through llama-server; noted the acceptance stats from its log
- [ ] One sentence: why does temperature hurt the speedup?
      */}),
      resources: [
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "`--draft-model` / `--num-draft-tokens`; read speculative_generate_step in generate.py" },
        { title: "llama.cpp", url: "https://github.com/ggml-org/llama.cpp", type: "repo", note: "server `-md/-hfd` drafts, plus examples/speculative, lookup and lookahead" },
      ],
    },
    {
      id: "why-it-works",
      title: "Why it works: verifying k tokens costs about one step",
      kind: "concept",
      minutes: 60,
      runsOn: ["any"],
      md: MD(function () {/*
## The asymmetry: checking is cheaper than solving

Baseten’s analogy: verifying a sudoku is easy, solving one is hard. For an LLM, *generating* $k$ tokens takes $k$ sequential forward passes (each depends on the previous token). *Checking* $k$ proposed tokens takes **one** pass, because all $k$ positions are known up front — just like prefill.

```text
plain decode       [T]→t1 [T]→t2 [T]→t3 [T]→t4                          4 big steps → 4 tokens
speculative        [d][d][d][d] → guesses g1..g4
                   [T verify g1..g4 in one pass] → accept g1 g2, reject g3, + target's own t3'
                                                                         1 big step + 4 tiny → 3 tokens
```

The target always contributes **one** token per verify step (a correction at the first rejection, or a free “bonus” token if all $k$ were accepted), so speculation is never worse *in tokens per target pass*: $\text{tokens/pass} = N_{\text{accepted}} + 1$.

## Why the verify pass is (almost) free: the roofline

Verification runs the target on $k+1$ tokens. For its Linear layers (most of the time):

$$ \text{arithmetic intensity} \approx \frac{2 \times (k+1) \times N}{N \times \text{bytes/weight}} = \frac{2(k+1)}{\text{bytes/weight}} $$

| Setup | Ridge point (FLOP/byte) | Intensity at $k{=}4$ | Still memory-bound? |
|---|---|---|---|
| H100, BF16 weights (989 TFLOPS / 3.35 TB/s) | ~295 | 5 | yes, massively → verify ≈ free |
| H100, batch 64 × $(k{+}1){=}5$ tokens | ~295 | 320 | **no** → verify costs real compute |
| M2 Pro (~7 TFLOPS / 200 GB/s), 4-bit weights | ~35 | ~18 | yes, but only ~2× headroom |

::viz roofline

Worked numbers for Qwen2.5-7B BF16 on an H100: a step reads 15 GB → **4.5 ms**. Verifying 5 tokens needs $5 \times 15 = 76$ GFLOP → **0.08 ms** of math. The step still takes ~4.5 ms. On your Mac the compute headroom is much smaller (low FLOPs relative to bandwidth), so verification of large $k$ starts to cost real time — one reason Mac speedups are more modest than GPU speedups.

## Latency vs throughput

Speculation spends **extra compute** (draft passes + verifying tokens that get rejected) to cut **latency**. That’s a great deal when compute is idle (low batch) and a bad one when compute is the bottleneck (high batch): rejected tokens become wasted work that could have served another user.

::viz latency-throughput

## What it does and doesn’t improve

| Metric | Effect | Why |
|---|---|---|
| ITL / TPOT (time per output token) | ✅ down 1.5–3× | more tokens per memory-bound step |
| TTFT | ❌ unchanged (slightly worse) | prefill is compute-bound; a draft model adds its own prefill |
| Throughput at low batch | ✅ up | same as ITL |
| Throughput at high batch | ⚠️ flat or **down** | compute saturated; rejected tokens are waste |
| Output quality | ✅ identical (exact methods) | rejection sampling preserves the target’s distribution |

- [ ] Using the roofline viz, find the batch size at which verifying $k=4$ tokens per sequence makes a BF16 8B model compute-bound on an H100
- [ ] Explain to a colleague why speculative decoding doesn’t help a summarization workload with 8k-token prompts and 100-token outputs
      */}),
    },
    {
      id: "the-math",
      title: "The math: acceptance rule, residual distribution, proof, expected speedup",
      kind: "math",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
> [!PREREQ] Probability in 2 minutes
> A **distribution** over the vocabulary is a list of non-negative numbers summing to 1: $p(v)$ = probability the next token is $v$. **Sampling** picks $v$ with probability $p(v)$. If you do A with probability $a$ and then, independently, B with probability $b$, both happen with probability $a \cdot b$. If an outcome can happen in two *mutually exclusive* ways, add their probabilities. The **expected value** of a count is the probability-weighted average. A **geometric series**: $1 + \alpha + \alpha^2 + \dots + \alpha^{k} = \dfrac{1 - \alpha^{k+1}}{1 - \alpha}$.

## The setup

At one position, the **target** model says the next token has distribution $p$; the **draft** model says $q$ (both after the *same* temperature / top-p processing). The draft has already sampled a token $x \sim q$. We want to output a token whose distribution is **exactly $p$** — while keeping $x$ as often as possible.

## The rule (Leviathan et al. 2022; Chen et al. 2023)

1. **Accept** $x$ with probability $\min\!\Big(1, \dfrac{p(x)}{q(x)}\Big)$.
2. If rejected, sample a replacement from the **residual distribution**
$$ r(v) = \frac{\max\big(0,\; p(v) - q(v)\big)}{\sum_{u} \max\big(0,\; p(u) - q(u)\big)} $$

Intuition: where the draft **under**-estimates ($q(x) \le p(x)$) always accept. Where it **over**-estimates, accept only the fraction $p/q$. The rejections then “refill” exactly the tokens the draft under-covered.

## Worked example (3-token vocabulary)

$p = [0.6,\ 0.3,\ 0.1]$ for tokens A, B, C; $q = [0.3,\ 0.5,\ 0.2]$.

| Token | $q$ | accept prob $\min(1,p/q)$ | $q \times$ accept $= \min(p,q)$ | $\max(0, p-q)$ |
|---|---|---|---|---|
| A | 0.3 | 1 | 0.3 | 0.3 |
| B | 0.5 | 0.6 | 0.3 | 0 |
| C | 0.2 | 0.5 | 0.1 | 0 |
| **sum** | 1 | | **0.7** = P(accept) | 0.3 = P(reject) |

Residual $r = [1, 0, 0]$. Final probabilities: A $= 0.3 + 0.3 \times 1 = 0.6$ ✓, B $= 0.3$ ✓, C $= 0.1$ ✓. Exactly $p$.

## Proof sketch (it’s three lines)

A token $v$ can be output two mutually exclusive ways: **drafted and accepted**, or **rejected then drawn from $r$**.

$$ P(\text{drafted \& accepted } v) = q(v)\cdot\min\!\Big(1, \tfrac{p(v)}{q(v)}\Big) = \min\big(p(v), q(v)\big) $$

$$ P(\text{reject}) = 1 - \sum_u \min(p(u), q(u)) = \sum_u \big(p(u) - \min(p(u),q(u))\big) = \sum_u \max\big(0, p(u) - q(u)\big) $$

$$ P(\text{output } v) = \min(p(v), q(v)) + P(\text{reject}) \cdot \frac{\max(0, p(v)-q(v))}{\sum_u \max(0,p(u)-q(u))} = \min(p,q) + \max(0, p-q) = p(v) \;\blacksquare $$

The normalizer cancels with $P(\text{reject})$ — that’s the whole trick. It works position by position: after an accepted token, the next position’s $p$ and $q$ are both conditioned on the same (correct) prefix, so the argument repeats. After the first rejection everything later is discarded, because it was conditioned on a token we didn’t output.

**Acceptance rate:** $\alpha = \sum_v \min(p(v), q(v)) = 1 - \text{TV}(p, q)$ — one minus the total-variation distance. The closer the draft imitates the target, the higher $\alpha$.

**Greedy special case ($T=0$):** $p$ and $q$ are one-hot. Accept iff the draft token equals the target’s argmax; on rejection, $r$ is the target’s argmax. Simple string matching.

**Deterministic drafts (n-gram lookup):** $q$ is one-hot at $x$, so accept with probability $p(x)$; on rejection, $r$ = $p$ with $x$ removed and renormalized.

## How many tokens per step?

Assume each drafted token is accepted independently with probability $\alpha$. The step emits $N_{\text{acc}} + 1$ tokens, and $P(N_{\text{acc}} \ge i) = \alpha^i$ for $i \le k$. Summing:

$$ \mathbb{E}[\text{tokens per step}] = \sum_{i=0}^{k} \alpha^{i} = \frac{1 - \alpha^{k+1}}{1 - \alpha} $$

If one draft step costs $c$ target steps (e.g. $c = 0.1$ for a 10× cheaper draft) and verifying costs ~1 target step:

$$ \text{speedup} \approx \frac{1 - \alpha^{k+1}}{(1 - \alpha)(1 + k c)} $$

$\alpha = 0.8$, $c = 0.1$:

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| tokens/step | 1.80 | 2.44 | 2.95 | 3.36 | 3.69 | 3.95 | 4.33 |
| speedup | 1.64 | 2.03 | 2.27 | 2.40 | 2.46 | **2.47** | 2.41 |

Diminishing returns: each extra draft token is accepted only if *all previous ones were* ($\alpha^i$ shrinks), while its cost is paid every time. Baseten: “aim for short, high-acceptance drafts.” Lower $\alpha$ (say 0.6) moves the optimum down to $k \approx 3$ and caps speedup near 1.7×.

::viz spec-decode

## Check it numerically

```python
import numpy as np
rng = np.random.default_rng(0)
p = np.array([0.6, 0.3, 0.1]); q = np.array([0.3, 0.5, 0.2])

def spec_one(p, q):
    x = rng.choice(len(q), p=q)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, True
    r = np.maximum(p - q, 0)
    return rng.choice(len(p), p=r / r.sum()), False

draws = [spec_one(p, q) for _ in range(100_000)]
print(np.bincount([x for x, _ in draws], minlength=3) / len(draws))   # ≈ [0.6 0.3 0.1]
print(np.mean([a for _, a in draws]))                                  # ≈ 0.7
```

- [ ] Verify the worked example by hand, then with the simulation
- [ ] Break it on purpose: “accept if $p(x) > 0.2$” or “on rejection sample from $p$ instead of $r$”. Show the output is no longer $p$
- [ ] Reproduce the $\alpha=0.8$ table in 5 lines of Python; find the best $k$ for $\alpha = 0.6$ and $\alpha = 0.9$ with $c = 0.1$ and $c = 0.3$
- [ ] Prove $\alpha = 1 - \text{TV}(p,q)$ using $\min(a,b) = a - \max(0, a-b)$
      */}),
      resources: [
        { title: "Fast Inference from Transformers via Speculative Decoding (Leviathan et al.)", url: "https://arxiv.org/abs/2211.17192", type: "paper", note: "the rule, the proof, the expected-speedup analysis" },
        { title: "Accelerating LLM Decoding with Speculative Sampling (Chen et al., DeepMind)", url: "https://arxiv.org/abs/2302.01318", type: "paper", note: "concurrent derivation; clean algorithm box" },
      ],
    },
    {
      id: "build",
      title: "Build: greedy, then sampled speculative decoding with KV-cache rollback",
      kind: "build",
      minutes: 240,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] What you’ll have
> `specdec.py` with (1) a plain KV-cached generator, (2) greedy speculative decoding that provably matches it, (3) sampled speculative decoding with exact rejection sampling, (4) a statistical test that the sampled version preserves the target distribution, and (5) acceptance / tokens-per-step / tok/s stats.

## Step 0 — models and helpers

```python
# specdec.py
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

dev = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16            # use torch.float16 if your macOS/PyTorch lacks bf16 on MPS
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
def load(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to(dev).eval()  # older transformers: torch_dtype=
target = load("Qwen/Qwen2.5-3B-Instruct")
draft = load("Qwen/Qwen2.5-0.5B-Instruct")
V = min(target.config.vocab_size, draft.config.vocab_size)   # 7B pads its vocab to 152064; 0.5B–3B use 151936

@torch.no_grad()
def forward(model, cache, tokens):
    """Append `tokens` (list of ids) to `cache`; return float32 logits, one row per token: (len(tokens), V)."""
    x = torch.tensor([tokens], device=dev)
    return model(input_ids=x, past_key_values=cache, use_cache=True).logits[0, :, :V].float()

def rollback(cache, keep):
    extra = cache.get_seq_length() - keep
    if extra > 0:
        cache.crop(-extra)          # negative = drop that many trailing positions

def encode(user_msg):
    text = tok.apply_chat_template([{"role": "user", "content": user_msg}], add_generation_prompt=True, tokenize=False)
    return tok(text).input_ids
```

**Invariant used everywhere:** the cache holds every token of `out` **except the last one**; the last token is fed at the start of the next step. Keep this in your head and the rollback arithmetic becomes easy.

## Step 1 — the baseline

```python
@torch.no_grad()
def generate_plain(prompt_ids, max_new):
    cache = DynamicCache()
    forward(target, cache, prompt_ids[:-1])
    out = list(prompt_ids)
    for _ in range(max_new):
        out.append(int(forward(target, cache, [out[-1]])[-1].argmax()))
    return out[len(prompt_ids):]
```

## Step 2 — greedy speculative decoding

```python
@torch.no_grad()
def generate_spec_greedy(prompt_ids, max_new, k=4):
    t_cache, d_cache = DynamicCache(), DynamicCache()
    forward(target, t_cache, prompt_ids[:-1]); forward(draft, d_cache, prompt_ids[:-1])
    out, n0 = list(prompt_ids), len(prompt_ids)
    steps = acc_total = 0
    while len(out) - n0 < max_new:
        # 1) draft k tokens autoregressively with the small model
        drafts, last = [], out[-1]
        for _ in range(k):
            last = int(forward(draft, d_cache, [last])[-1].argmax())
            drafts.append(last)
        # 2) verify all k in ONE target pass: row i predicts the token after (out[-1], d_1..d_i)
        preds = forward(target, t_cache, [out[-1]] + drafts).argmax(-1).tolist()   # k+1 predictions
        # 3) longest agreeing prefix + the target's own next token (correction or bonus)
        n_acc = 0
        while n_acc < k and drafts[n_acc] == preds[n_acc]:
            n_acc += 1
        out += drafts[:n_acc] + [preds[n_acc]]
        steps += 1; acc_total += n_acc
        # 4) restore the invariant for both caches
        keep = len(out) - 1
        rollback(t_cache, keep); rollback(d_cache, keep)
        if d_cache.get_seq_length() < keep:          # all k accepted: the draft never saw d_k
            forward(draft, d_cache, out[d_cache.get_seq_length():keep])
    new = out[n0:][:max_new]
    return new, {"steps": steps, "accept_rate": acc_total / (steps * k), "tokens_per_step": len(new) / steps}
```

Walk through the cache lengths once with pencil and paper ($n$ = `len(out)` at the start of a step): target cache $n-1$ → after verify $n+k$ → keep $n+n_{acc}$. Draft cache $n-1$ → after drafting $n-1+k$ → keep $n+n_{acc}$, which is one *more* than it has when all $k$ were accepted — hence the catch-up forward. (MLX-LM’s `speculative_generate_step` does the same thing: “if we accepted all the draft tokens, include the last draft token in the next draft step”.)

**Test:** greedy speculation must reproduce plain greedy.

```python
ids = encode("Rewrite this function with type hints and a docstring:\n\ndef add(a, b):\n    return a + b\n")
ref = generate_plain(ids, 200)
for k in (1, 2, 4, 8):
    got, st = generate_spec_greedy(ids, 200, k)
    same = sum(a == b for a, b in zip(ref, got))
    print(k, st, f"{same}/{len(ref)} tokens identical")
```

> [!WARNING] “Almost identical” can be correct
> Verifying 5 tokens in one pass uses different matmul shapes than 1-token decoding, so logits differ in the last bits (especially in bf16). At a near-tie the argmax can flip and the texts diverge from there. Check that the **first** divergence is at a position where the target’s top-2 logits are within ~0.1 of each other; anything else is a bug (almost always the cache invariant). Running both in float32 on CPU should give 100% identical output.

## Step 3 — sampled speculative decoding (exact)

```python
def probs(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)

@torch.no_grad()
def generate_spec_sampled(prompt_ids, max_new, k=4, temperature=1.0):
    t_cache, d_cache = DynamicCache(), DynamicCache()
    forward(target, t_cache, prompt_ids[:-1]); forward(draft, d_cache, prompt_ids[:-1])
    out, n0 = list(prompt_ids), len(prompt_ids)
    steps = acc_total = 0
    while len(out) - n0 < max_new:
        drafts, qs, last = [], [], out[-1]
        for _ in range(k):
            q = probs(forward(draft, d_cache, [last])[-1], temperature)
            last = int(torch.multinomial(q, 1))
            drafts.append(last); qs.append(q)
        ps = probs(forward(target, t_cache, [out[-1]] + drafts), temperature)   # (k+1, V)
        n_acc = 0
        for i, x in enumerate(drafts):
            if torch.rand(()).item() < min(1.0, (ps[i, x] / qs[i][x]).item()):
                n_acc += 1
            else:
                break
        if n_acc < k:                                          # rejected: sample the residual
            r = torch.clamp(ps[n_acc] - qs[n_acc], min=0)
            nxt = int(torch.multinomial(r / r.sum(), 1)) if r.sum() > 0 else int(torch.multinomial(ps[n_acc], 1))
        else:                                                  # all accepted: free bonus token from p
            nxt = int(torch.multinomial(ps[k], 1))
        out += drafts[:n_acc] + [nxt]
        steps += 1; acc_total += n_acc
        keep = len(out) - 1
        rollback(t_cache, keep); rollback(d_cache, keep)
        if d_cache.get_seq_length() < keep:
            forward(draft, d_cache, out[d_cache.get_seq_length():keep])
    new = out[n0:][:max_new]
    return new, {"steps": steps, "accept_rate": acc_total / (steps * k), "tokens_per_step": len(new) / steps}
```

If you add top-p/top-k, apply the **same** filtering to both $p$ and $q$ before the rule — the guarantee is about the processed distributions.

## Step 4 — a statistical test that the distribution is preserved

Take the real $p$ and $q$ at the first generated position of a prompt and run the accept/residual rule 200k times, vectorized:

```python
from scipy.stats import chisquare
ids = encode("Name a random animal.")
p = probs(forward(target, DynamicCache(), ids)[-1], 1.0).cpu().double()
q = probs(forward(draft, DynamicCache(), ids)[-1], 1.0).cpu().double()

def spec_first_token(p, q, n):
    x = torch.multinomial(q, n, replacement=True)
    accept = torch.rand(n, dtype=torch.double) < (p[x] / q[x]).clamp(max=1)
    r = (p - q).clamp(min=0)
    y = torch.multinomial(r / r.sum(), n, replacement=True)
    return torch.where(accept, x, y)

def chi2_vs(p, samples, top=20):
    idx = p.topk(top).indices
    obs = torch.stack([(samples == t).sum() for t in idx]).double()
    exp = p[idx] * len(samples)
    obs = torch.cat([obs, (len(samples) - obs.sum()).view(1)]); exp = torch.cat([exp, (len(samples) - exp.sum()).view(1)])
    return chisquare(obs.numpy(), exp.numpy()).pvalue

s = spec_first_token(p, q, 200_000)
print("accept rate", ((p.minimum(q)).sum()).item(), " p-value (correct rule):", chi2_vs(p, s))
wrong = torch.multinomial(q, 200_000, replacement=True)                # "just use the draft"
print("p-value (draft only):", chi2_vs(p, wrong))                        # ≈ 0 → caught
```

A p-value that is *not* tiny means “can’t distinguish from $p$” (run it a few times; p-values are uniform on [0, 1] when the null is true). The draft-only sampler should get a p-value ≈ 0. The challenge extends this to multi-token sequences through your full `generate_spec_sampled`.

## Step 5 — measure

```python
def timed(fn, *a, **kw):
    t = time.perf_counter(); r = fn(*a, **kw); return r, time.perf_counter() - t
(ref, dt0) = timed(generate_plain, ids, 256)
print(f"plain: {256/dt0:.1f} tok/s")
for k in (1, 2, 3, 4, 6, 8):
    (new, st), dt = timed(generate_spec_greedy, ids, 256, k)
    print(f"k={k}: {len(new)/dt:.1f} tok/s  speedup {dt0/dt:.2f}×  {st}")
```

> [!NOTE] Why your HF version is slower than MLX
> Each `forward` call here pays Python + dispatch overhead that is large relative to a 0.5B model’s actual work, so the effective draft cost $c$ is much worse than the 6× parameter ratio suggests. Measure $c$ directly (time one draft step vs one target step) and plug it into the speedup formula — your measured speedup should match the formula with *measured* $\alpha$ and $c$. That’s the real lesson: **draft cost matters as much as acceptance.**

- [ ] Greedy speculation matches plain greedy (100% in float32 on CPU; explain any bf16 divergence)
- [ ] Sampled version runs; accept rate at $T=1$ lower than at $T=0$
- [ ] Chi-square test: correct rule not rejected, draft-only sampler rejected
- [ ] Measured $c$ and $\alpha$; predicted speedup from the formula within ~20% of measured
- [ ] Printed a table of tok/s and speedup for $k \in \{1,2,3,4,6,8\}$
      */}),
      resources: [
        { title: "Hugging Face — Assisted generation", url: "https://huggingface.co/blog/assisted-generation", type: "article", note: "the same algorithm inside `model.generate(assistant_model=...)`" },
        { title: "Transformers generation strategies", url: "https://huggingface.co/docs/transformers/main/en/generation_strategies", type: "docs", note: "assisted decoding and prompt lookup built into generate()" },
        { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "compact, fast speculative decoding in pure PyTorch (see generate.py)" },
      ],
    },
    {
      id: "variants",
      title: "Variants: n-gram / prompt lookup, Medusa, EAGLE-1/2/3, MTP, lookahead",
      kind: "concept",
      minutes: 90,
      runsOn: ["mac", "any"],
      md: MD(function () {/*
A separate draft model is the simplest speculator and the one with the **most overhead** (Baseten ch. 5.2.1): its own weights, KV cache, prefill, and CPU round-trips. Every variant below attacks that overhead or the acceptance rate.

## N-gram / prompt lookup — no model at all

If the output copies spans of the input (code edits, RAG answers quoting documents, JSON rewrites, summaries with names), just **look up** the last few tokens earlier in the context and propose what followed them:

```python
def prompt_lookup(tokens, k, max_ngram=3):
    for n in range(max_ngram, 0, -1):                     # prefer longer matches
        tail = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):  # most recent earlier occurrence first
            if tokens[start:start + n] == tail:
                cont = tokens[start + n : start + n + k]
                if cont:
                    return cont
    return []

@torch.no_grad()
def generate_lookup_greedy(prompt_ids, max_new, k=8):
    cache = DynamicCache(); forward(target, cache, prompt_ids[:-1])
    out, n0, steps = list(prompt_ids), len(prompt_ids), 0
    while len(out) - n0 < max_new:
        drafts = prompt_lookup(out, k)
        preds = forward(target, cache, [out[-1]] + drafts).argmax(-1).tolist()
        n_acc = 0
        while n_acc < len(drafts) and drafts[n_acc] == preds[n_acc]:
            n_acc += 1
        out += drafts[:n_acc] + [preds[n_acc]]
        rollback(cache, len(out) - 1); steps += 1
    return out[n0:][:max_new], (len(out) - n0) / steps
```

Drop this into your `specdec.py` and try the code-edit prompt: tokens/step can exceed 3–5 with *zero* draft cost. On open-ended prose it collapses to ~1. Baseten: n-gram drafts can be longer than 10 tokens and “easily outperform EAGLE” for code completion and revision. vLLM’s **suffix decoding** extends the idea with a suffix tree over previous *outputs* too (great for agents that repeat themselves).

## Medusa — extra heads on the target

Add 2–4 small **extra LM heads** to the target’s last hidden state; head $i$ predicts token $t+i+1$ directly. Candidates from all heads form a **tree** that is verified in one pass with a tree-shaped attention mask. No separate model, but heads predict far-ahead tokens *without* seeing the tokens in between, so acceptance drops quickly. Rare in production today; it inspired EAGLE.

## EAGLE-1/2/3 — a tiny autoregressive drafter on the target’s features

- **EAGLE** (2024): the drafter is ~one transformer layer that autoregresses over the target’s **top-layer hidden features** (plus the sampled token embeddings), reusing the target’s LM head. Features are far more predictable than tokens, so a tiny network drafts well. It runs *inside* the target’s process — no CPU round-trips.
- **EAGLE-2**: builds a **dynamic draft tree**, expanding branches where the drafter is confident — more accepted tokens per verify pass.
- **EAGLE-3**: fuses **low, mid and high-layer** features and trains with “training-time test” (the drafter sees its own predictions during training, like at inference). Up to ~8 draft tokens with high acceptance, and it keeps improving with more training data. Baseten: “the go-to general algorithm if you can train EAGLE heads.” Pre-trained EAGLE-3 heads exist for popular models.

## MTP — multi-token prediction built into the model

DeepSeek-V3 trains an extra **MTP module** alongside the main model to predict token $t+2$ from the main model’s hidden state and token $t+1$. At inference that module is a free, perfectly-matched drafter: DeepSeek reports **85–90% acceptance** for the second token and ~**1.8× tokens/s**. Newer model families increasingly ship MTP heads; engines expose them as `method: "mtp"`.

## Lookahead decoding — draft by Jacobi iteration

Guess a window of future tokens, repeatedly refine all of them in parallel (Jacobi iteration), and harvest the n-grams that appear along the way into a pool; verify candidates from the pool. No draft model or training, more general than prompt lookup — but it needs spare compute (Baseten ch. 5.2.4).

## Tree attention — verify many guesses at once

Instead of one chain of $k$ tokens, propose a tree (e.g. top-2 at each of 3 depths). Flatten it into one sequence and use an attention mask where each node attends only to its ancestors. One verify pass checks all branches; accept the longest valid path. More tokens accepted per pass, more compute per pass — exactly the trade you can afford at batch 1 and can’t at batch 128 (SpecInfer, Medusa, EAGLE-2).

## Comparison

| Method | Extra model? | Training? | Draft cost | Typical tokens/step | Best for |
|---|---|---|---|---|---|
| Draft model | yes (≥10× smaller, same tokenizer) | optional (distill) | medium–high | 1.8–3 | quick win when a small sibling exists |
| N-gram / prompt lookup | no | no | ~0 | 1–5+ (domain!) | code edits, RAG, rewriting |
| Suffix decoding | no | no | ~0 | varies | agents, repetitive outputs |
| Medusa | heads | yes | low | 1.8–2.5 | historical |
| EAGLE-3 | tiny head | yes | low | 3–4.5 | general default |
| MTP | built in | by the model creator | low | ~1.8–2.5 | models that ship it (DeepSeek-V3…) |
| Lookahead | no | no | uses spare compute | 1.5–2 | no drafter available |

- [ ] Implemented `prompt_lookup`; measured tokens/step on a code-edit prompt vs a story prompt
- [ ] Explained in 3 sentences why EAGLE drafts from features instead of tokens
- [ ] Drew (on paper) the attention mask for a tree with 2 candidates at depth 1 and 2 at depth 2
      */}),
      resources: [
        { title: "Prompt lookup decoding", url: "https://github.com/apoorvumang/prompt-lookup-decoding", type: "repo", note: "the original n-gram-from-prompt trick, in a few lines" },
        { title: "Medusa", url: "https://arxiv.org/abs/2401.10774", type: "paper", note: "multiple decoding heads + tree attention" },
        { title: "EAGLE", url: "https://arxiv.org/abs/2401.15077", type: "paper", note: "feature-level drafting" },
        { title: "EAGLE-2", url: "https://arxiv.org/abs/2406.16858", type: "paper", note: "dynamic draft trees" },
        { title: "EAGLE-3", url: "https://arxiv.org/abs/2503.01840", type: "paper", note: "multi-layer features, training-time test" },
        { title: "DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "MTP module reused for speculative decoding (85–90% acceptance)" },
        { title: "Lookahead Decoding", url: "https://arxiv.org/abs/2402.02057", type: "paper", note: "Jacobi-iteration n-gram drafting without a drafter" },
        { title: "SpecInfer (tree-based verification)", url: "https://arxiv.org/abs/2305.09781", type: "paper", note: "token trees + tree attention" },
      ],
    },
    {
      id: "production",
      title: "Production concerns: batch size, domains, memory, and when to turn it off",
      kind: "concept",
      minutes: 60,
      runsOn: ["any"],
      md: MD(function () {/*
## 1. Batch size eats the free compute

At batch $B$ with $k$ drafts per sequence, each verify pass processes $B(k+1)$ tokens. Once that pushes the Linear layers past the roofline ridge, verification is **no longer free**, and every rejected token is compute stolen from other users.

Napkin: 8B BF16 on an H100, ridge ≈ 295 tokens per pass (ignoring attention). With $k=4$: $B(k+1) = 295 \Rightarrow B \approx 60$. In practice attention, the drafter and kernel efficiency pull the crossover lower — often somewhere in the 16–64 range.

Baseten ch. 5.2: *only useful at low batch sizes; engines must dynamically disable it at high batch; it trades throughput and cost for latency.* vLLM describes its speculation as for “medium-to-low QPS, memory-bound workloads” and offers **dynamic speculative decoding** that adapts to load. In the lab you’ll plot this crossover yourself.

## 2. Acceptance depends on the traffic, not just the models

| Factor | Effect on $\alpha$ |
|---|---|
| Temperature ↑ | ↓ (flatter $p$ and $q$ disagree more) |
| Domain mismatch (drafter trained on chat, traffic is SQL) | ↓↓ |
| Output copies input (edits, RAG, extraction, JSON) | ↑↑ (n-gram shines) |
| Different chat template / system prompt in drafter | ↓ |
| Structured output (grammar masks) | ↑ usually (fewer valid tokens) — but masks must apply to verification too |
| Position within the draft | falls with depth: the $i$-th token needs all earlier ones accepted |

So monitor **mean acceptance length per route / customer**, not a global average. A new customer with different traffic can silently halve your speedup.

## 3. Memory and TTFT costs

- A draft model’s **weights and KV cache** take GPU memory that would otherwise hold more target KV → fewer concurrent sequences. EAGLE/MTP heads are small; a 0.5B draft for a 7B target is ~7% extra weights plus its own KV.
- A separate drafter needs its **own prefill** → slightly worse TTFT. Feature-based drafters (EAGLE, MTP) reuse the target’s prefill.
- CPU orchestration between two models costs latency; TensorRT-LLM and vLLM work hard to keep the loop on the GPU.

## 4. Engine plumbing you now understand

- **KV rollback in a paged cache:** rejected tokens occupied slots; the engine lowers the sequence length and frees trailing blocks (M08) — exactly your `crop`.
- **CUDA graphs** want fixed shapes; verify passes of $k+1$ tokens per sequence need their own captured graphs.
- **Sampling parameters** (temperature, top-p, penalties) must be applied identically to draft and target distributions; logprob outputs must come from the target.
- **Streaming:** a single step may emit several tokens, so streamed chunks can contain multiple tokens — clients that count chunks as tokens will mis-measure (your lab client uses the server’s `usage` instead).

## 5. When to turn it off

| Turn it **off** (or shrink $k$) when… | Because |
|---|---|
| Batch / concurrency is high | no idle compute; rejections waste throughput |
| Mean acceptance length < ~1.3–1.5 | overhead exceeds gains |
| High temperature / creative traffic | low $\alpha$ |
| Prefill-dominated workloads | speculation only helps decode |
| Memory is the bottleneck | drafter weights/KV reduce batch size |
| Cost-per-token (offline batch) is the goal | you pay compute for latency you don’t need |

> [!REAL] Interaction with other techniques
> Baseten ch. 5: “large batches starve speculation.” Quantization (M14) makes each target step cheaper *and* shrinks the drafter; disaggregated decode pools (M17) run at controlled batch sizes where speculation is predictable; RL rollout engines (M22–M23) use it at low concurrency tails when a few long generations hold up the batch.

- [ ] For your own product idea, write the three metrics you’d alert on to catch speculation regressions
- [ ] Estimate the crossover batch size for a 70B FP8 model on H100 with $k=3$ (state your assumptions)
      */}),
      resources: [
        { title: "vLLM — Speculative decoding", url: "https://docs.vllm.ai/en/latest/features/speculative_decoding/", type: "docs", note: "methods, `--speculative-config`, lossless guarantees, method-selection table" },
      ],
    },
    {
      id: "lab-sweep",
      title: "Lab: sweep k on your Mac, then vLLM n-gram / draft / EAGLE-3 vs concurrency",
      kind: "lab",
      minutes: 300,
      runsOn: ["mac", "colab", "cloud"],
      md: MD(function () {/*
## Part A — sweep $k$ on your Mac

Use your `specdec.py`. Build two prompt sets of ~10 prompts each: **copy-heavy** (code edits: paste a 30-line function and ask for a rename + type hints) and **open-ended** (explanations, stories). For each set, $T \in \{0, 1\}$ and $k \in \{1, 2, 3, 4, 6, 8\}$, record accept rate $\alpha$, tokens/step and tok/s (256 new tokens, median over prompts). Also measure the draft cost ratio $c$ once:

```python
import time, statistics
def step_time(model, n=30):
    cache = DynamicCache(); forward(model, cache, ids[:-1]); ts = []
    for _ in range(n):
        t = time.perf_counter(); forward(model, cache, [ids[-1]]); ts.append(time.perf_counter() - t)
    return statistics.median(ts[3:])
c = step_time(draft) / step_time(target); print("draft cost ratio c =", round(c, 3))
```

Plot measured speedup vs $k$ next to the formula $\frac{1-\alpha^{k+1}}{(1-\alpha)(1+kc)}$ using your measured $\alpha$ and $c$ (one line per prompt set × temperature). Where do they disagree, and why? (Hint: acceptance isn’t independent per position, and verify cost grows a bit with $k$.)

Add a third curve for **prompt lookup** (`generate_lookup_greedy`) on the copy-heavy set.

## Part B — vLLM on a GPU (Colab T4 is enough for n-gram and draft-model)

```bash
pip install -q vllm openai
# baseline (T4 has no BF16 → --dtype half)
nohup vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 4096 --port 8000 > base.log 2>&1 &
# n-gram / prompt lookup
nohup vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 4096 --port 8000 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 4}' > ngram.log 2>&1 &
# separate draft model (same tokenizer)
nohup vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 4096 --port 8000 \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4}' > draft.log 2>&1 &
```

Run one server at a time. Method names and keys evolve — check `vllm serve --help=speculative-config` or the docs for your installed version.

A load generator that measures per-user decode speed and total throughput at a given concurrency:

```python
# load.py — python load.py --model Qwen/Qwen2.5-3B-Instruct --conc 1 4 16 64
import argparse, asyncio, json, statistics, time
from openai import AsyncOpenAI

async def one(c, model, prompt):
    stamps, usage = [], None
    s = await c.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=256, temperature=0, stream=True, stream_options={"include_usage": True})
    async for ch in s:
        if ch.choices and ch.choices[0].delta.content: stamps.append(time.perf_counter())
        if ch.usage: usage = ch.usage
    return usage.completion_tokens, stamps[0], stamps[-1]     # count tokens from usage: chunks may hold several

async def run(model, conc, prompts, n_req):
    c = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x"); sem = asyncio.Semaphore(conc)
    async def guarded(p):
        async with sem: return await one(c, model, p)
    t0 = time.perf_counter()
    res = await asyncio.gather(*[guarded(prompts[i % len(prompts)]) for i in range(n_req)])
    wall = time.perf_counter() - t0
    per_user = statistics.median((n - 1) / (b - a) for n, a, b in res if b > a)
    return per_user, sum(n for n, _, _ in res) / wall

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--model"); a.add_argument("--conc", type=int, nargs="+")
    a.add_argument("--prompts", default="prompts.jsonl"); args = a.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    for conc in args.conc:
        pu, tot = asyncio.run(run(args.model, conc, prompts, n_req=max(4 * conc, 16)))
        print(f"conc={conc:3d}  per-user {pu:6.1f} tok/s   total {tot:7.1f} tok/s")
```

Acceptance from the server’s Prometheus counters (take the difference before/after a run):

```python
import requests
def counter(name):
    total = 0.0
    for line in requests.get("http://localhost:8000/metrics").text.splitlines():
        if not line.startswith("#") and line.split("{")[0].split(" ")[0] == name:
            total += float(line.rsplit(" ", 1)[1])
    return total
drafts = counter("vllm:spec_decode_num_drafts_total")
accepted = counter("vllm:spec_decode_num_accepted_tokens_total")
drafted = counter("vllm:spec_decode_num_draft_tokens_total")
print("mean acceptance length", 1 + accepted / drafts, " draft acceptance rate", accepted / drafted)
```

Fill in, for both prompt sets:

| Method | conc 1 per-user | conc 1 total | conc 16 per-user | conc 16 total | conc 64 total | mean accept. length |
|---|---|---|---|---|---|---|
| none | | | | | | — |
| ngram k=4 | | | | | | |
| draft 0.5B k=4 | | | | | | |
| EAGLE-3 (Part C) | | | | | | |

## Part C — EAGLE-3 on a rented GPU (optional, ~1 hour on an L4/A100/H100)

```bash
# Llama 3.1 is gated: accept the license on Hugging Face and `hf auth login` first
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --port 8000 \
  --speculative-config '{"method": "eagle3", "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", "num_speculative_tokens": 3}'
```

Run `load.py` against it and a no-speculation baseline of the same model. Try `num_speculative_tokens` 2, 3, 5.

- [ ] Part A: speedup-vs-$k$ plot with measured points and formula curves for ≥2 prompt sets × 2 temperatures
- [ ] Part B: table filled for none / n-gram / draft model at concurrency 1, 4, 16, 64 on both prompt sets
- [ ] Identified the concurrency at which each method stops helping total throughput
- [ ] (Part C) EAGLE-3 vs baseline at concurrency 1 and 32, with mean acceptance length
      */}),
      resources: [
        { title: "vLLM — EAGLE draft models", url: "https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/", type: "docs", note: "EAGLE / EAGLE-3 configs and pre-trained heads" },
      ],
    },
    {
      id: "train-a-drafter",
      title: "Deep dive: training and distilling your own drafter",
      kind: "deep",
      optional: true,
      minutes: 180,
      runsOn: ["colab", "cloud", "mac"],
      md: MD(function () {/*
Acceptance rate is $\alpha = 1 - \text{TV}(p, q)$: the closer the drafter’s distribution is to the **target’s** (not to “good text”), the faster you go. So the best training data for a drafter is **the target’s own outputs on your traffic** — this is distillation.

## Recipe: distill a 0.5B drafter toward the 3B target

1. **Collect prompts** that look like production traffic (a few thousand; anonymized logs, or a public set in the same domain).
2. **Generate responses with the target** (vLLM, $T$ matching production). This is *self-distillation data*: the drafter will learn the target’s phrasing, not the dataset’s.
3. **Train the drafter** to match the target’s full next-token distribution (logits), not just the sampled token — forward KL gives far more signal per token:

```python
import torch, torch.nn.functional as F
# batch["input_ids"]: prompt + target-generated answer, padded; batch["mask"]: 1 on answer tokens
def distill_step(batch, target, draft, opt):
    ids, mask = batch["input_ids"], batch["mask"][:, 1:].reshape(-1).bool()
    with torch.no_grad():
        t_logits = target(ids).logits[:, :-1].float()
    d_logits = draft(ids).logits[:, :-1].float()
    V = min(t_logits.shape[-1], d_logits.shape[-1])
    t_logp = F.log_softmax(t_logits[..., :V], -1).flatten(0, 1)[mask]     # (answer tokens, V)
    d_logp = F.log_softmax(d_logits[..., :V], -1).flatten(0, 1)[mask]
    loss = F.kl_div(d_logp, t_logp, log_target=True, reduction="batchmean")   # KL(p_target || q_draft)
    loss.backward(); opt.step(); opt.zero_grad()
    return loss.item()
```

4. **Measure $\alpha$ before and after** on held-out prompts with your `specdec.py`. Gains of 5–15 points of acceptance are common for in-domain traffic — worth more than any kernel tweak.

Practical notes: full fine-tuning 0.5B with AdamW needs ~8 GB for weights + optimizer states (fp32) plus the frozen target — fine on a Colab T4/L4 or a 24 GB+ Mac; LoRA (M05) makes it lighter. Keep the drafter’s chat template identical to the target’s. The DistillSpec paper studies which divergence (forward KL, reverse KL, TV) and which data (target-generated vs fixed) maximize acceptance.

## EAGLE-3 and MTP heads

EAGLE-style heads are trained on the target’s **hidden states** from low/mid/high layers, so the data pipeline runs the target over your corpus and saves features. Don’t build that from scratch — use:

- **vLLM `speculators`** — train EAGLE-3-style speculators and load them directly in vLLM;
- **SpecForge** (SGLang) — EAGLE-3 training for SGLang;
- the original **EAGLE** repo for reference implementations.

> [!REAL] Why teams bother
> Off-the-shelf heads are trained on generic chat. A coding assistant, a SQL agent or a support bot has a very different token distribution; a domain-trained drafter can take mean acceptance length from ~2 to 3+. It’s a common, high-leverage project for an inference team — and a nice portfolio piece.

- [ ] Generated ≥1,000 target responses on a domain of your choice
- [ ] Distilled the 0.5B drafter for ≥1 epoch; logged the KL loss
- [ ] Measured $\alpha$ and tokens/step before vs after on held-out prompts
      */}),
      resources: [
        { title: "DistillSpec", url: "https://arxiv.org/abs/2310.08461", type: "paper", note: "knowledge distillation to align drafters with targets" },
        { title: "vLLM speculators", url: "https://github.com/vllm-project/speculators", type: "repo", note: "train and package EAGLE-3-style speculators for vLLM" },
        { title: "SpecForge", url: "https://github.com/sgl-project/SpecForge", type: "repo", note: "EAGLE-3 training framework for SGLang" },
        { title: "EAGLE (official code)", url: "https://github.com/SafeAILab/EAGLE", type: "repo", note: "reference implementation for EAGLE-1/2/3" },
      ],
    },
  ],

  challenge: {
    title: "Speculative decoding: correct, measured, and with an off-switch policy",
    md: MD(function () {/*
Ship `course-work/m15/` containing your implementation, tests, plots and a `REPORT.md`.

1. **Implementation:** greedy and sampled speculative decoding with a separate draft model and KV-cache rollback, plus a prompt-lookup drafter behind the same interface (`propose(tokens) -> (drafts, q_probs or None)`).
2. **Correctness (greedy):** on 20 prompts, speculative greedy output equals plain greedy output (float32), or every divergence is explained by a near-tie.
3. **Correctness (sampled) — a real statistical test:** with $T = 1$, generate the first **3** tokens 3,000 times with your full `generate_spec_sampled` and 3,000 times with plain target sampling. Compare the distributions of the 3-token sequences (two-sample chi-square over the most frequent sequences + an “other” bucket, or a permutation test on total-variation distance). Report the p-value, and show that a deliberately broken variant (e.g. no residual — sample from $p$ on rejection) **fails** the same test.
4. **Speedup vs $k$ plot:** measured speedup and the formula prediction (measured $\alpha$, $c$) for $k = 1\ldots8$, for at least two prompt domains and two temperatures, plus prompt lookup.
5. **When to turn it off:** using your vLLM concurrency data, write a policy an engine could implement — e.g. “disable when running sequences > N or rolling mean acceptance length < L; use k = f(α)”. Justify N and L with your numbers and the roofline.

```python
# sketch of the sequence-level test
from collections import Counter
from scipy.stats import chi2_contingency
A = Counter(tuple(generate_spec_sampled(ids, 3, k=4)[0]) for _ in range(3000))
B = Counter(tuple(generate_plain_sampled(ids, 3)) for _ in range(3000))   # write this: plain sampling at T=1
top = [s for s, _ in (A + B).most_common(30)]
table = [[A[s] for s in top] + [3000 - sum(A[s] for s in top)],
         [B[s] for s in top] + [3000 - sum(B[s] for s in top)]]
print(chi2_contingency(table).pvalue)
```
    */}),
    checklist: [
      "Greedy speculative output matches plain greedy on 20 prompts (or every divergence is a documented near-tie)",
      "Sequence-level statistical test passes for the correct sampler and clearly fails for a broken one",
      "Speedup-vs-k plot with measured points and the formula curve, ≥2 domains × 2 temperatures, plus prompt lookup",
      "Measured draft cost ratio c and acceptance α are reported and used in the predictions",
      "vLLM results at ≥3 concurrency levels for ≥2 methods",
      "A concrete turn-it-off policy with thresholds justified by your data and the roofline",
    ],
    stretch: "Add tree speculation: take the draft’s top-2 tokens at the first two depths (4 leaves), verify all branches in one target pass with a custom 4D attention mask, and measure tokens/step vs chain speculation at equal verify cost.",
  },

  connects: MD(function () {/*
- **Back to M07 and M14:** speculation exists because decode is memory-bound; quantization makes each memory-bound step cheaper, speculation makes each step produce more tokens. Use both: a 4-bit target with a 4-bit drafter is the standard local setup.
- **M08–M10:** your mini-engine’s KV manager needs a “truncate sequence to length L” operation and a scheduler that can decide per step whether to speculate — add both.
- **M16–M17:** at scale, speculation runs on decode pools with controlled batch sizes (disaggregation), and EAGLE/MTP heads must be sharded alongside tensor-parallel targets.
- **M18–M19:** acceptance length becomes a production metric you dashboard and alert on, per route.
- **M22–M23 (RL):** rollout generation is decode-heavy with long tails of a few slow sequences at low concurrency — exactly where speculative decoding helps RL systems.
  */}),

  interview: [
    "Walk through the accept/reject rule and prove that the output distribution equals the target model’s.",
    "With per-token acceptance α = 0.8 and k = 4, how many tokens do you expect per verification step? If a draft step costs 10% of a target step, what is the speedup?",
    "Why does speculative decoding improve inter-token latency but not TTFT? Why does its benefit shrink (or reverse) at large batch sizes?",
    "Compare draft-model, n-gram/prompt-lookup, Medusa, EAGLE-3 and MTP speculation. Which would you deploy for a code-editing product, and why?",
    "Your mean acceptance length dropped from 3.1 to 1.6 after onboarding a new customer. How do you debug it and what are your options?",
    "How do you roll back the KV cache after a rejection in a paged-attention engine?",
    "Greedy speculative decoding output differs from non-speculative greedy output on 0.3% of tokens. Bug or expected? How would you tell?",
    "Why must top-p / temperature be applied identically to the draft and target distributions?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 5.2 Speculative Decoding", url: "Inference%20Engineering.pdf", type: "book", note: "draft-target, Medusa, EAGLE, n-gram; batch-size limits" },
    { title: "Leviathan et al. — Fast Inference via Speculative Decoding", url: "https://arxiv.org/abs/2211.17192", type: "paper", note: "the original exact algorithm and analysis" },
    { title: "Chen et al. — Speculative Sampling", url: "https://arxiv.org/abs/2302.01318", type: "paper", note: "DeepMind’s concurrent formulation" },
    { title: "EAGLE-3", url: "https://arxiv.org/abs/2503.01840", type: "paper", note: "current go-to learned drafter" },
    { title: "Medusa", url: "https://github.com/FasterDecoding/Medusa", type: "repo", note: "multi-head drafting + tree attention code" },
    { title: "DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "multi-token prediction reused as a drafter" },
    { title: "Lookahead Decoding", url: "https://github.com/hao-ai-lab/LookaheadDecoding", type: "repo", note: "drafter-free parallel decoding" },
    { title: "vLLM speculative decoding docs", url: "https://docs.vllm.ai/en/latest/features/speculative_decoding/", type: "docs", note: "every method you’ll configure in production" },
    { title: "Hugging Face — Assisted generation", url: "https://huggingface.co/blog/assisted-generation", type: "article", note: "approachable explanation with latency plots" },
    { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "speculative decoding + quantization in <1000 lines of PyTorch" },
  ],
});
