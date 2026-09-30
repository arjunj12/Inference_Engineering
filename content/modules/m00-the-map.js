Course.module({
  id: "m00-the-map",
  title: "Run an LLM on your laptop — and see the whole map",
  short: "The map (run an LLM locally)",
  tagline: "In one evening: run a real model on your Mac, measure how fast it is, and learn the vocabulary of everything between a prompt and a GPU.",
  hours: 8,
  runsOn: ["mac"],
  tags: ["orientation", "ollama", "mlx", "metrics"],

  goal: MD(function () {/*
You type a prompt into a model running **on your own Mac** and get back numbers like these, measured by a script *you* wrote:

```text
model: qwen2.5:1.5b (Q4_K_M)   prompt: 212 tokens   output: 256 tokens
time-to-first-token (TTFT):   0.21 s
decode speed:                 71.4 tokens/s   (≈ 14 ms per token)
end-to-end latency:           3.80 s
```

Then you’ll explain — with a back-of-envelope calculation — **why** the 7B model is ~4× slower than the 1.5B one, and why the 4-bit version is faster than the 16-bit one. That single explanation (“decode is limited by how fast you can read the weights from memory”) is the seed of this entire curriculum.

::viz generation-loop
  */}),

  why: MD(function () {/*
Every inference-engineering job boils down to three numbers — **latency, throughput, cost** — under a **quality** constraint. Companies like Baseten, Together, Fireworks and every frontier lab hire people who can (1) measure these properly, (2) explain where the time goes, and (3) change the system to move them. Today you do a miniature version of all three.
  */}),

  prereqs: [
    {
      title: "Terminal + Python environment on macOS",
      skipIf: "you already use uv/conda/venv daily",
      md: MD(function () {/*
Install [Homebrew](https://brew.sh), then **uv** (a fast Python package manager — think `npm` for Python):

```bash
brew install uv
uv venv ~/ai && source ~/ai/bin/activate   # one env for the whole course
uv pip install torch numpy matplotlib jupyter requests
python -c "import torch; print(torch.backends.mps.is_available())"   # True on Apple Silicon
```

`mps` is PyTorch’s backend for the Apple GPU. When something says `device="cuda"`, you’ll usually write `device="mps"` on your Mac.
      */}),
    },
    {
      title: "What “parameters”, “tokens” and “bytes” mean (60-second version)",
      math: true,
      md: MD(function () {/*
- A **parameter** (weight) is one number the model learned. A “7B” model has 7 billion of them.
- Each number is stored with some **precision**: 32-bit float = 4 bytes, 16-bit (FP16/BF16) = 2 bytes, 8-bit = 1 byte, 4-bit = ½ byte.
- So **model size in memory ≈ parameters × bytes per parameter**. 7B × 2 bytes = 14 GB. 7B × 0.5 byte ≈ 3.5 GB (+ a bit of overhead).
- A **token** is a chunk of text (~¾ of an English word). Models read and write tokens, not characters.
- Units: 1 GB = $10^9$ bytes. Bandwidth in GB/s = how many of those bytes per second the chip can move.
      */}),
    },
  ],

  lessons: [
    {
      id: "run-it",
      title: "Run a model locally with Ollama (and MLX)",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
## Step 1 — the fastest path: Ollama

[Ollama](https://ollama.com) wraps **llama.cpp** (a C++ inference engine with a Metal GPU backend) behind a friendly CLI + HTTP server.

```bash
brew install ollama
ollama serve &                       # starts an HTTP server on localhost:11434
ollama run qwen2.5:1.5b "Explain a KV cache in one sentence."
ollama run qwen2.5:1.5b --verbose "Write a haiku about GPUs"   # prints timing stats
```

Look at the `--verbose` output. You’ll see `prompt eval rate` and `eval rate`. Those are **prefill** speed and **decode** speed — the two phases of every LLM request. Remember those words.

## Step 2 — Apple’s own framework: MLX

[MLX](https://github.com/ml-explore/mlx) is Apple’s NumPy/PyTorch-like array library that runs natively on Apple GPUs. `mlx-lm` runs Hugging Face models on it:

```bash
uv pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "What is inference?" --max-tokens 200
```

It prints `Prompt: N tokens, X tokens-per-sec` and `Generation: M tokens, Y tokens-per-sec` and peak memory. Same two phases again.

> [!INTUITION] What just happened
> 1. Your text was **tokenized** into integers. 2. The model processed **all prompt tokens at once** (prefill) and produced the first new token. 3. It then produced tokens **one at a time**, each time feeding the new token back in (decode). 4. Tokens were **detokenized** and streamed to your terminal.

- [ ] Ran a model with Ollama and noted prompt-eval and eval rates
- [ ] Ran the same model family with mlx-lm and compared speeds
      */}),
      resources: [
        { title: "Ollama", url: "https://github.com/ollama/ollama", type: "repo", note: "zero-friction local serving (llama.cpp under the hood)" },
        { title: "MLX", url: "https://github.com/ml-explore/mlx", type: "repo", note: "Apple’s array framework — your local lab bench" },
        { title: "llama.cpp", url: "https://github.com/ggml-org/llama.cpp", type: "repo", note: "the C++ engine behind Ollama; native Metal backend" },
      ],
    },
    {
      id: "measure",
      title: "Write your own benchmark script: TTFT, tokens/s, end-to-end latency",
      kind: "build",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
Tools print numbers; engineers **measure them independently**. Ollama exposes an HTTP API that **streams** tokens. The time until the first streamed chunk is TTFT; the gaps between chunks are inter-token latency.

```python
# bench.py — measure one streaming request against Ollama
import json, time, requests, statistics

def run(model, prompt, max_tokens=256):
    t0 = time.perf_counter()
    r = requests.post("http://localhost:11434/api/generate", stream=True, json={
        "model": model, "prompt": prompt, "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0}})
    stamps = []
    for line in r.iter_lines():
        if not line: continue
        chunk = json.loads(line)
        if chunk.get("response"): stamps.append(time.perf_counter())
        if chunk.get("done"):
            final = chunk; break
    ttft = stamps[0] - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": final["prompt_eval_count"], "output_tokens": final["eval_count"],
        "ttft_s": ttft, "e2e_s": stamps[-1] - t0,
        "itl_p50_ms": 1000 * statistics.median(gaps),
        "decode_tok_s": len(gaps) / (stamps[-1] - stamps[0]),
    }

if __name__ == "__main__":
    prompt = open("prompt.txt").read()          # paste ~200 words of any text
    for m in ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b"]:
        runs = [run(m, prompt) for _ in range(3)]  # first run includes model load: discard it
        print(m, runs[-1])
```

Definitions you just implemented (we’ll formalize them in Module 06):

| Metric | Meaning | Who cares |
|---|---|---|
| **TTFT** — time to first token | queueing + prefill of the whole prompt | chat UX (“is it thinking?”) |
| **ITL / TPOT** — inter-token latency / time per output token | one decode step | streaming feel, voice agents |
| **E2E latency** | TTFT + (output tokens × TPOT) | API callers, agents |
| **Throughput** | tokens/s across *all* concurrent users | cost per token |

::viz percentiles

> [!WARNING] Measurement hygiene
> Discard the first (cold) run, fix `temperature=0`, keep prompt/output lengths constant, and report **percentiles over many runs**, not one number. Plug in your laptop — macOS throttles on battery.

- [ ] `bench.py` prints TTFT, ITL p50, decode tok/s for three model sizes
- [ ] Plotted decode tok/s vs model size (matplotlib)
      */}),
    },
    {
      id: "napkin",
      title: "Napkin math: predict your decode speed before measuring it",
      kind: "math",
      minutes: 60,
      md: MD(function () {/*
Here is the most important back-of-envelope idea in inference. To generate **one** token at batch size 1, the chip must read **every weight** of the model from memory once (each weight takes part in a multiply). Reading is slow; the arithmetic is fast. So:

$$
\text{time per token} \approx \frac{\text{bytes of weights}}{\text{memory bandwidth}}
\qquad\Rightarrow\qquad
\text{tokens/s} \approx \frac{\text{bandwidth}}{\text{params} \times \text{bytes/param}}
$$

> [!PREREQ] Units refresher
> Bandwidth like “120 GB/s” means $120 \times 10^9$ bytes per second. Dividing bytes by bytes-per-second gives seconds. That’s all the math here.

**Worked example (M2 Pro, ~200 GB/s):** Qwen2.5-7B in 4-bit ≈ 7.6B × 0.56 bytes (4-bit + scales) ≈ 4.3 GB.
$\;200 / 4.3 \approx 46$ tokens/s **upper bound**. Real engines reach ~60–80% of peak bandwidth → expect ~30–37 tok/s.

Look up your chip’s memory bandwidth (M1 ≈ 68 GB/s, M1/M2/M3 Pro ≈ 150–200, M-Max ≈ 400–546, Ultra ≈ 800; an **H100 ≈ 3,350 GB/s**) and fill in:

| Model | Params | Bytes/param | Weight GB | Predicted tok/s | Measured | % of peak |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b Q4 | 1.5B | ~0.56 | | | | |
| qwen2.5:7b Q4 | 7.6B | ~0.56 | | | | |
| qwen2.5:7b FP16 (mlx) | 7.6B | 2 | | | | |

> [!INTUITION] Why this matters so much
> This is why **quantization** (fewer bytes per weight) speeds up decode, why **batching** many users is magic (read weights once, use them for 64 users), and why data-center GPUs obsess over **HBM bandwidth**. You’ll formalize it as the **roofline model** in Module 07.

- [ ] Filled the table; predictions within ~2× of measurements
      */}),
    },
    {
      id: "map",
      title: "The map: what sits between a prompt and a GPU",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
Every production LLM service — ChatGPT, Claude, a Baseten deployment — is some version of this stack. Each layer is a module in this curriculum:

::viz inference-stack

| Layer | What it does | Where you learn it |
|---|---|---|
| **Client & API** | OpenAI-compatible HTTP, streaming (SSE), retries | M10, M18 |
| **Gateway / router** | auth, rate limits, picks a replica (load / KV-cache aware) | M18 |
| **Autoscaler** | adds/removes GPU replicas with traffic; fights cold starts | M18 |
| **Inference engine** (vLLM, SGLang, TensorRT-LLM) | scheduler, KV cache manager, batching, sampling | M06–M10 |
| **Model runtime** | the transformer forward pass, CUDA graphs, torch.compile | M03–M04, M13 |
| **Kernels** | matmul, attention (FlashAttention), fused ops — CUDA/Triton | M11–M12 |
| **Hardware** | GPU compute, HBM bandwidth, NVLink between GPUs | M07, M16 |
| **Optimizations across layers** | quantization, speculative decoding, parallelism, disaggregation | M14–M17 |

And the model itself has to come from somewhere: **pretraining → fine-tuning → RL post-training** (M05, M21–M23). RL training is now *dominated by inference* (generating rollouts), which is why RL infrastructure and inference engineering have merged into one skill set.

> [!REAL] Role archetypes you’ll hear about
> **Kernel / performance engineer** (CUDA, Triton, profiling) · **Inference runtime engineer** (vLLM/SGLang internals, scheduling, KV cache) · **Inference platform engineer** (Kubernetes, autoscaling, routing, multi-cloud) · **RL / post-training infra engineer** (rollout engines, weight sync). This curriculum covers all four; the Job skills map tab shows which modules build which.
      */}),
      resources: [
        { title: "Inference Engineering (Baseten) — Ch. 0–1", url: "https://www.baseten.co/inference-engineering/", type: "book", note: "your local PDF; read Ch. 0–1 now" },
        { title: "LLM Visualization (bbycroft)", url: "https://bbycroft.net/llm", type: "tool", note: "3D walk through every matrix in a GPT" },
      ],
    },
    {
      id: "read-book",
      title: "Read: Baseten “Inference Engineering” Ch. 0–1",
      kind: "read",
      minutes: 90,
      md: MD(function () {/*
Open `Inference Engineering.pdf` (in this folder). Read **Chapter 0 (Inference)** and **Chapter 1 (Prerequisites)** — ~25 pages. Take notes on:

- The three layers the book uses: **runtime**, **infrastructure**, **tooling**.
- Online vs offline, consumer vs B2B workloads — how they change which metric matters.
- Why you choose the **model** and set a **quality baseline with evals** *before* optimizing.
- Latency **percentiles** (P50/P90/P99) and end-to-end metrics.

> [!CHECK] Self-check
> Could you explain to a colleague why a batch-summarization job (offline) and a voice agent (online) would be optimized completely differently — even with the same model?

The book is the reference text for Phases 2–6; each later module tells you which chapter to read.
      */}),
    },
  ],

  challenge: {
    title: "Your first inference report",
    md: MD(function () {/*
Write a one-page `REPORT.md` (keep it in a `course-work/m00` folder — this folder becomes your portfolio):

1. Hardware: chip, unified memory, memory bandwidth.
2. A table of 3 models × 2 precisions (e.g. 4-bit vs 8-bit or FP16 via MLX): TTFT, decode tok/s p50, peak memory.
3. Your **napkin prediction** next to each measurement, and the % of peak bandwidth achieved.
4. Two paragraphs: *Why does decode speed track model bytes? Why does TTFT grow with prompt length but decode speed barely does?*
    */}),
    checklist: [
      "Benchmarks use ≥5 runs per config and report the median",
      "Every measured decode speed has a napkin prediction next to it",
      "You can explain prefill vs decode in your own words without notes",
    ],
    stretch: "Run 4 concurrent requests against Ollama (`OLLAMA_NUM_PARALLEL=4`) and measure total throughput vs single-user speed. What happened, and why? (Preview of batching.)",
  },

  connects: MD(function () {/*
You now have the **map** and one crucial law (*decode ≈ bytes ÷ bandwidth*). Phase 1 opens the black box: you’ll build the model itself so that later, when we talk about KV caches, attention kernels and quantized weights, you know exactly which tensor we mean.
  */}),

  interview: [
    "What are TTFT and TPOT, and which one does prompt length mostly affect?",
    "Estimate the maximum batch-1 decode speed of a 70B FP16 model on one H100 (3.35 TB/s). Why can’t it even fit?",
    "Why is a P99 latency SLO more meaningful than a mean latency target?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — local PDF", url: "Inference%20Engineering.pdf", type: "book", note: "reference text for the course" },
    { title: "Horace He — Making Deep Learning Go Brrrr", url: "https://horace.io/brrr_intro.html", type: "article", note: "compute vs memory vs overhead bound" },
    { title: "Databricks — LLM Inference Performance Engineering", url: "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", type: "article", note: "concise metrics & trade-offs" },
  ],
});
