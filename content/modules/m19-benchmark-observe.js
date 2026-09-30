Course.module({
  id: "m19-benchmark-observe",
  title: "Capstone: benchmark, observe & cost an LLM service",
  short: "Capstone: benchmark & cost",
  tagline: "Produce the report an inference team actually ships: throughput–latency curves, SLO goodput, Grafana dashboards of engine and GPU metrics, and a defensible dollars-per-million-tokens — comparing two configurations.",
  hours: 24,
  level: "core",
  capstone: true,
  runsOn: ["colab", "cloud", "mac"],
  tags: ["benchmarking", "goodput", "prometheus", "grafana", "dcgm", "cost", "capacity-planning", "capstone"],

  goal: MD(function () {/*
You publish a blog-style benchmark report for one **model + engine + GPU** combination, comparing two configurations (e.g. **FP8 vs BF16** on an L4/H100, or **vLLM vs SGLang**, or **AWQ vs FP16** on a Colab T4). Its headline table looks like this:

```text
Qwen2.5-7B-Instruct on 1× H100 80GB · workload: chat (ISL ~1000, OSL ~250, Poisson arrivals)
SLO: TTFT p99 ≤ 500 ms and TPOT p99 ≤ 40 ms

config        max RPS @ SLO   goodput (req/s)   out tok/s @ SLO   TTFT p99   TPOT p99   $/1M out tokens*
BF16               7.5             7.3               1,810          462 ms      37 ms       0.64
FP8 (W8A8)        11.0            10.8               2,690          471 ms      38 ms       0.43   ← 1.47× goodput
* $2.50/GPU-hr on-demand, 60% average utilization, output tokens only (lesson 7 shows why comparing this to API prices is subtle)
```

…with throughput–latency curves across concurrency, a Grafana dashboard screenshot showing *why* the knee happens (queue depth, KV-cache usage, GPU SM activity), a cost model, a capacity plan for a hypothetical product, and a section called **“How this benchmark could be wrong.”** (All numbers above are illustrative — yours will differ, and that is the point.)

::viz latency-throughput
  */}),
  demo: { viz: "latency-throughput", params: {} },

  why: MD(function () {/*
Benchmarks are how inference teams make every decision — which GPU to buy, which engine to run, whether FP8 is worth it, what to charge. They are also where most public claims are wrong: mismatched input/output lengths, closed-loop tests reported as “max throughput”, no warmup, averages instead of p99, cost at 100% utilization. The Baseten book (ch. 4.5) insists on benchmarking **your** traffic shape against **your** SLO, changing one variable at a time. Hiring managers read a candidate’s benchmark report the way they read code: it shows whether you understand the system. This capstone is the portfolio piece that proves you do.
  */}),

  prereqs: [
    {
      title: "Latency vocabulary refresher",
      skipIf: "you can define TTFT, TPOT/ITL, E2E latency and throughput without notes",
      md: MD(function () {/*
- **TTFT** (time to first token) = queueing + prefill + first decode step. Dominated by prompt length and queue.
- **TPOT** (time per output token, after the first) ≈ **ITL** (inter-token latency, measured per gap). $\text{E2E} \approx \text{TTFT} + (\text{OSL}-1) \times \text{TPOT}$.
- **Throughput**: requests/s, output tokens/s, or total (input + output) tokens/s — always say which.
- **ISL / OSL**: input / output sequence length in tokens. These two numbers change results more than anything else.
- Per-user speed = $\frac{1}{\text{TPOT}}$ tokens/s; system throughput ≈ concurrency × per-user speed (until the GPU saturates).
      */}),
    },
    {
      title: "Prometheus in 10 minutes",
      skipIf: "you have written PromQL with rate() and histogram_quantile()",
      md: MD(function () {/*
- A **target** (your engine) exposes plain-text metrics at `/metrics`; Prometheus **scrapes** them every few seconds and stores time series.
- **Counter** only goes up (`vllm:generation_tokens_total`) → use `rate(x[1m])` to get per-second. **Gauge** goes up and down (`vllm:num_requests_waiting`). **Histogram** = cumulative buckets `_bucket{le="0.5"}` plus `_sum` and `_count`.
- Percentile from a histogram: `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` — an estimate, only as precise as the bucket boundaries.
- **Grafana** draws dashboards from PromQL queries. Both run in Docker in one command (lesson 6).
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: benchmark a live server in five minutes",
      kind: "demo",
      minutes: 60,
      runsOn: ["colab", "cloud", "mac"],
      md: MD(function () {/*
## Start a server and point a benchmark at it

On a Colab **T4** (free) a 1.5B model is plenty to see every effect; on an L4/A100/H100 use a 7–8B model.

```bash
pip install -U vllm
# T4 has no BF16 → --dtype half. Run in the background:
nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 --port 8000 > server.log 2>&1 &
until curl -sf localhost:8000/health; do sleep 5; done; echo READY
```

```bash
vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 200 --max-concurrency 16 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:500 tpot:50 --save-result --result-dir results
```

The tail of the output (numbers illustrative, T4):

```text
============ Serving Benchmark Result ============
Successful requests:                     200
Maximum request concurrency:             16
Benchmark duration (s):                  41.7
Request throughput (req/s):              4.80
Request goodput (req/s):                 4.62
Output token throughput (tok/s):         613.9
Total Token throughput (tok/s):          3069.4
---------------Time to First Token----------------
Mean TTFT (ms):                          151.2
P99 TTFT (ms):                           488.0
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.1
P99 TPOT (ms):                           31.5
```

Now run it three more times with `--max-concurrency 1`, `4`, `64`. Plot output tok/s (x) against p99 TPOT (y). You just drew your first **throughput–latency curve**: throughput climbs almost linearly with concurrency while per-token latency barely moves (decode is memory-bound, M07), then the curve bends — throughput flattens and latency shoots up. That bend is the **knee**, and every decision in this module is about finding it, explaining it, and pricing it.

::viz latency-throughput

## Same tool, your own engine

Point the benchmark at the mini engine you built in M10. If it exposes an OpenAI-style `/v1/completions` endpoint with streaming, use `--backend openai --endpoint /v1/completions`; if not, wrap it in a ~30-line FastAPI shim that streams server-sent events. Run concurrency 1, 4, 16 with the same ISL/OSL. On a Mac, also try it against `llama-server` or `mlx_lm.server` (both speak the OpenAI API):

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8000 -np 4 --metrics &
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 50 --max-concurrency 4
```

(`vllm bench` is a client; on macOS you can `pip install vllm` for the CLI even though you will not serve with it there. If installation fails, use GuideLLM or AIPerf from lesson 3 — they are pure clients.)

> [!NOTE] What to notice
> Your mini engine probably has a *much* earlier knee than vLLM. Write down why (no paged KV? no continuous batching? no CUDA graphs?). That comparison is a perfect paragraph for your report — and an interview story.

- [ ] Benchmarked a server at 4 concurrency levels and drew the throughput–latency curve
- [ ] Identified the knee and wrote one sentence on what saturates there
- [ ] Ran the same benchmark against your M10 engine (or llama.cpp on Mac) and compared the knee
      */}),
      resources: [
        { title: "vLLM — bench serve CLI", url: "https://docs.vllm.ai/en/latest/cli/bench/serve.html", type: "docs", note: "every flag: datasets, rates, goodput, saving results" },
        { title: "vLLM — Benchmark suites", url: "https://docs.vllm.ai/en/latest/contributing/benchmarks.html", type: "docs", note: "dataset recipes and examples" },
      ],
    },
    {
      id: "methodology",
      title: "Benchmarking methodology: how not to lie with numbers",
      kind: "concept",
      minutes: 120,
      md: MD(function () {/*
## The workload *is* the benchmark

The same model on the same GPU can show 3× different “throughput” depending on the traffic you replay. Decide these four things first and print them at the top of every result:

| Choice | Options | Why it matters |
|---|---|---|
| **Sequence lengths** | Fixed (ISL 1000 / OSL 250); distributions from a dataset (ShareGPT-like chat); your production histogram | Prefill-heavy (RAG, 8K in / 200 out) and decode-heavy (reasoning, 200 in / 4K out) stress different parts of the GPU |
| **Content** | Synthetic random tokens; real prompts (ShareGPT, your logs); shared prefixes (`prefix_repetition` dataset) | Random tokens get **zero** prefix-cache hits and odd speculative acceptance rates; real traffic often shares long system prompts |
| **Arrival process** | Closed loop (N users, each sends when the previous finishes) vs **open loop** (requests arrive at a rate, e.g. Poisson, regardless of completions) | See below — the most common source of misleading numbers |
| **Output control** | `--ignore-eos` (force exactly OSL tokens) vs natural stopping | Natural stopping makes OSL depend on the model/quantization — a confound when comparing configs |

**Baseten ch. 4.5:** match the ISL/OSL distribution of production; ideally replay **shadow traffic** (a copy of real requests) against the candidate; change **one variable at a time**; and benchmark for your SLO, not for a vanity peak.

## Open loop vs closed loop

- **Closed loop / fixed concurrency** (`--max-concurrency 32`): the load generator waits for a response before sending the next request. If the server slows down, the load slows down too — the system *can never be overloaded*, so latency looks politely bounded. Good for “how does latency scale with batch size?” and for finding the per-replica concurrency target (M18).
- **Open loop / request rate** (`--request-rate 8`, Poisson arrivals with `--burstiness 1`): requests keep arriving like real users. Past capacity, queues grow without bound and TTFT explodes — the honest picture of what users would see. Use it for “what RPS can one replica sustain within SLO?”.
- `--request-rate inf` with no concurrency cap = send everything at once: a **max-throughput** test, not a latency test. Never publish latency from it.

Report both: a concurrency sweep (the throughput–latency curve) and a rate sweep (latency vs offered load, and the max rate that meets the SLO).

## Warmup, repetition, and the usual pitfalls

1. **Warmup**: the first requests pay CUDA-graph capture, torch.compile, JIT kernel tuning, grammar compilation, cold prefix cache. Discard a warmup run (e.g. 50 requests you do not record) or run the whole benchmark twice and keep the second.
2. **Enough requests**: a p99 from 100 requests is the 1–2 worst requests — pure noise. Use ≥ 1,000 requests for p99, or ≥ 3 runs and report the spread.
3. **The client is the bottleneck**: a Python client tokenizing and parsing SSE at 5,000 tok/s can saturate a CPU core. Watch client CPU; run the client on a separate machine for big GPUs.
4. **Tokenizer mismatch**: counting tokens with the wrong tokenizer silently changes every per-token metric.
5. **Network**: benchmark from the same region; report whether TLS and a gateway were in the path.
6. **Measure what you claim**: TPOT excludes the first token; ITL is per-gap; “throughput” must say input+output or output only.
7. **Change one thing**: comparing FP8 on engine version A with BF16 on version B measures two things at once.
8. **Quality**: a faster config that is dumber is not faster. Pair every perf comparison with a quick eval (M14: lm-evaluation-harness subset, or task-specific checks).

> [!INTUITION] Synthetic vs production traces
> Synthetic fixed-length loads are for **engineering** (clean, repeatable, isolate one effect). Traces (BurstGPT, your own logs with timestamps) are for **decisions** (burstiness, length mixes, prefix sharing). A report should include at least one of each.

- [ ] Wrote a one-paragraph “workload spec” for your capstone: ISL/OSL, dataset, arrival process, SLO
- [ ] Ran the same config closed-loop at concurrency 64 and open-loop at an overload rate; explained why their TTFT p99 differ so much
- [ ] Measured the warmup effect: first 50 requests vs the rest
- [ ] Checked client CPU during the heaviest run
      */}),
      resources: [
        { title: "Inference Engineering (Baseten) — ch. 4.5 Benchmarking", url: "Inference%20Engineering.pdf", type: "book", note: "shadow traffic, matching ISL/OSL, one variable at a time" },
        { title: "Anyscale — How continuous batching enables 23× throughput", url: "https://www.anyscale.com/blog/continuous-batching-llm-inference", type: "article", note: "a classic, careful serving benchmark write-up to emulate" },
      ],
    },
    {
      id: "tools",
      title: "The benchmarking toolbox: vllm bench, AIPerf, GuideLLM and friends",
      kind: "lab",
      minutes: 120,
      runsOn: ["colab", "cloud", "mac"],
      md: MD(function () {/*
## Pick a tool, but know what it measures

All of these are **clients**: they send OpenAI-compatible (or engine-native) streaming requests and time the chunks. They differ in workload generators, arrival models, reporting and polish.

| Tool | From | Strengths | Watch out for |
|---|---|---|---|
| `vllm bench serve` | vLLM | Many datasets (random, sharegpt, sonnet, burstgpt, prefix_repetition, hf, custom), rate + burstiness + concurrency, `--goodput`, ramp-up, JSON results; also `vllm bench throughput` (offline) and `vllm bench latency` | Installs all of vLLM; defaults tuned for vLLM |
| `sglang.bench_serving` | SGLang | Same idea, supports many backends (`--backend sglang/vllm/...`), great for engine A/B | Similar flags, different defaults |
| **AIPerf** | NVIDIA (successor to GenAI-Perf) | Concurrency or request-rate modes, synthetic or trace inputs, rich per-request records, `aiperf plot` charts, many endpoint types (chat, embeddings, rankings) | Newer tool; flags still evolving |
| **GuideLLM** | vLLM project (ex Neural Magic) | **Sweep** profile that finds sync and max throughput then tests rates in between — produces a curve in one command | CLI changed between versions — check `guidellm --help` |
| **genai-bench** | SGLang project | Traffic scenarios, Excel/plot reports, used for vendor comparisons | |
| **LLMPerf** | Anyscale/Ray | Historical baseline, simple | Largely unmaintained; not for new work |
| k6 / Locust | General load testing | Realistic user scripts, spikes, auth, multi-step flows | Do not parse tokens by default — measure TTFT yourself |

The Baseten book (ch. 4.5) mentions GenAI-Perf and genai-bench for performance and Locust for load testing; public cross-vendor comparisons like SemiAnalysis’ InferenceMAX run nightly sweeps across GPUs and engines — read them to calibrate your own numbers.

## The same workload in three tools

```bash
# 1) vLLM
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --num-prompts 500 --max-concurrency 16 --save-result --result-dir r/vllm

# 2) AIPerf
pip install aiperf
aiperf profile --model Qwen/Qwen2.5-1.5B-Instruct --url http://localhost:8000 \
  --endpoint-type chat --streaming --concurrency 16 --request-count 500 \
  --synthetic-input-tokens-mean 512 --output-tokens-mean 128 \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct

# 3) GuideLLM (sweep: sync → max throughput → rates in between)
pip install guidellm
guidellm benchmark --target http://localhost:8000 --rate-type sweep --max-seconds 30 \
  --data "prompt_tokens=512,output_tokens=128"
# newer releases: guidellm run --backend kind=openai_http,target=http://localhost:8000 \
#   --profile kind=sweep --data kind=synthetic_text,prompt_tokens=512,output_tokens=128
```

Then build a comparison table: output tok/s, TTFT p50/p99, TPOT/ITL p50/p99 per tool.

> [!WARNING] Why the tools disagree
> Typical reasons for 5–20% differences: (1) **TTFT definition** — some clients count the first *non-empty content* chunk, some the first byte (a role-only chunk); (2) **token counting** — client-side re-tokenization vs the server’s `usage` field; (3) **output length** — `--ignore-eos` or `min_tokens` in one tool but not another; (4) **arrival** — concurrency vs Poisson rate; (5) **client overhead**. When you cite a number, cite the tool, version and flags.

## Offline vs online

`vllm bench throughput` (or `llm.generate` on a big list) measures the engine **without** HTTP, tokenization-in-the-loop or arrivals: an upper bound, useful for kernel/quantization work. `vllm bench serve` measures the service. Publish serving numbers; use offline numbers to explain them.

- [ ] Ran the identical workload through at least two tools and tabulated the differences
- [ ] Explained at least one discrepancy by reading the tool’s docs or source (e.g. how it defines TTFT)
- [ ] Produced a GuideLLM sweep or AIPerf plot and saved it for the report
- [ ] Wrote down exact tool versions and commands in a `REPRODUCE.md`
      */}),
      resources: [
        { title: "NVIDIA AIPerf", url: "https://github.com/ai-dynamo/aiperf", type: "tool", note: "successor to GenAI-Perf; profile + plot" },
        { title: "GuideLLM", url: "https://github.com/vllm-project/guidellm", type: "tool", note: "sweep-based rate finding and reports" },
        { title: "genai-bench", url: "https://github.com/sgl-project/genai-bench", type: "tool", note: "scenario-based serving benchmarks" },
        { title: "InferenceMAX (SemiAnalysis)", url: "https://github.com/InferenceMAX/InferenceMAX", type: "repo", note: "open, nightly cross-GPU/engine sweeps: learn from their methodology" },
      ],
    },
    {
      id: "latency-math",
      title: "The math of latency: percentiles, goodput, Little’s law, queues",
      kind: "math",
      minutes: 120,
      md: MD(function () {/*
## Percentiles, and why averages lie

Latency distributions are **skewed** with long right tails (a request that lands behind a long prefill, a preemption, a GC pause). The mean hides that; percentiles show it.

- **p50** (median): half the requests are faster. **p99**: 1 in 100 is slower — at 100 requests/s that is **one unhappy user every second**.
- Computing p99 from $n$ samples: sort, take the value at rank $\lceil 0.99\,n \rceil$ (nearest-rank; libraries interpolate slightly differently). With $n = 200$ that is the 2nd-worst request — noisy. Rule of thumb: $n \ge 10 / (1-q)$, i.e. ≥ 1,000 samples for p99.
- **Never average percentiles** across replicas or time windows (the mean of two p99s is not a p99). Merge the raw samples or the histogram buckets instead — which is exactly what `histogram_quantile(0.99, sum by (le) (rate(..._bucket[5m])))` does.

::viz percentiles

## Tail at scale

If one request fans out to $k$ independent calls (an agent calling the LLM 10 times, a RAG pipeline), the chance that *all* are fast is $0.99^k$. For $k = 10$: $0.99^{10} \approx 0.904$ — so **~10% of user requests hit at least one p99-slow call**. The per-call p99 becomes roughly the user-level p90. This is why agent products need much tighter per-call tails than chat products.

## Goodput: throughput that counts

**Goodput** (DistServe, 2024) = the rate of requests that meet **all** their SLOs. A server doing 10 req/s where 30% of requests miss the TTFT SLO has a goodput of 7 req/s. Two ways to report it:

1. **Per-run**: `vllm bench serve ... --goodput ttft:500 tpot:40` prints `Request goodput (req/s)`.
2. **Max rate at SLO**: sweep request rate; the highest rate where p99 TTFT ≤ 500 ms and p99 TPOT ≤ 40 ms. This single number is what capacity planning needs.

Computing it yourself from a detailed result (`--save-detailed`; key names may vary by version — inspect the JSON):

```python
import json, numpy as np
r = json.load(open("results/run.json"))
ttft = np.array(r["ttfts"]) * 1000                          # s → ms
tpot = np.array([np.mean(x) * 1000 if len(x) else 0 for x in r["itls"]])
ok = (ttft <= 500) & (tpot <= 40)
print("goodput req/s:", ok.sum() / r["duration"], " SLO attainment:", ok.mean())
```

## Little’s law: the capacity equation

For any stable system, averaged over time:

$$L = \lambda W$$

$L$ = requests in the system (in flight), $\lambda$ = arrival rate, $W$ = average time in the system. No assumptions about distributions. Example: 12 req/s, each taking 8 s end-to-end (TTFT 0.3 s + 250 tokens × 30 ms) → $L = 96$ requests in flight. If one replica holds 48 concurrent requests within SLO, you need **2 replicas** at average load — more for peaks. The same law explains closed-loop tests: with fixed concurrency $L$, throughput $\lambda = L / W$, so when latency rises throughput *must* flatten.

## Queues: why the knee is sharp

The simplest queue (M/M/1: Poisson arrivals, one server with exponential service time $\frac{1}{\mu}$) has mean time in system

$$W = \frac{1}{\mu - \lambda} = \frac{1/\mu}{1-\rho}, \quad \rho = \lambda/\mu$$

At $\rho = 0.5$, $W$ is 2× the service time; at 0.9, 10×; at 0.95, 20×. An LLM replica is not M/M/1 (it batches, so service rate *rises* with load until KV or compute saturates) but the qualitative lesson holds exactly: **near saturation, small increases in load cause huge increases in waiting time**. TTFT (which includes queueing) shows the knee first; TPOT follows once the batch is so large that each decode step slows.

::viz queueing

::viz latency-throughput

- [ ] Computed p50/p90/p99 from raw samples and from a histogram; explained the difference
- [ ] Showed numerically that averaging two replicas’ p99s gives the wrong answer
- [ ] Computed goodput for two configs at the same offered rate
- [ ] Verified Little’s law on your own benchmark: mean in-flight (from `vllm:num_requests_running` + waiting) ≈ rate × mean E2E
      */}),
      resources: [
        { title: "DistServe (goodput-optimized serving)", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "defines goodput under TTFT/TPOT SLOs" },
        { title: "Little’s law", url: "https://en.wikipedia.org/wiki/Little%27s_law", type: "article", note: "statement, intuition and examples" },
        { title: "Prometheus — Histograms and summaries", url: "https://prometheus.io/docs/practices/histograms/", type: "docs", note: "why you aggregate buckets, not quantiles" },
      ],
    },
    {
      id: "sweep",
      title: "Build: sweeps, curves and an A/B between two configs",
      kind: "build",
      minutes: 180,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
## Choose your comparison (by hardware)

| Hardware | Config A | Config B | What you expect to learn |
|---|---|---|---|
| H100 / L40S / L4 (FP8-capable) | Qwen2.5-7B BF16 | same model `--quantization fp8` (dynamic) or an FP8 checkpoint | Decode speedup from half the weight bytes, more KV room; check accuracy |
| Colab T4 (no BF16/FP8) | `Qwen/Qwen2.5-1.5B-Instruct --dtype half` in vLLM | same model in SGLang (`python -m sglang.launch_server --model-path ... --dtype half`) | Engine scheduling and overhead differences at small scale |
| Colab T4 alternative | `--enable-prefix-caching` on a shared-prefix workload | `--no-enable-prefix-caching` | Cache hit rate → TTFT and throughput |
| Any 2 GPUs | TP=1 × 2 replicas | TP=2 × 1 replica | Latency vs throughput trade of parallelism (M16) |

Keep **everything else identical**: engine version, `max-model-len`, `max-num-seqs`, dataset, seed, `--ignore-eos`, client machine.

## The sweep script

```python
# sweep.py — concurrency sweep + rate sweep for one running server; writes JSON per point
import subprocess, sys, json, pathlib
BASE, MODEL, TAG = "http://localhost:8000", sys.argv[1], sys.argv[2]   # e.g. Qwen/... bf16
out = pathlib.Path("results") / TAG; out.mkdir(parents=True, exist_ok=True)
common = ["vllm", "bench", "serve", "--backend", "vllm", "--base-url", BASE, "--model", MODEL,
          "--dataset-name", "random", "--random-input-len", "1000", "--random-output-len", "250",
          "--random-range-ratio", "0.2", "--ignore-eos", "--seed", "0",
          "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
          "--goodput", "ttft:500", "tpot:40", "--save-result", "--save-detailed", "--result-dir", str(out)]
def run(extra, name):
    subprocess.run(["vllm", "bench", "serve", *common[3:], *extra, "--result-filename", name], check=True)
run(["--num-prompts", "100", "--max-concurrency", "8"], "warmup.json")        # discard
for c in [1, 2, 4, 8, 16, 32, 64, 128]:
    run(["--num-prompts", str(max(200, 10 * c)), "--max-concurrency", str(c)], f"conc_{c}.json")
for rps in [1, 2, 4, 6, 8, 10, 12, 16]:
    run(["--num-prompts", str(int(rps * 120)), "--request-rate", str(rps), "--burstiness", "1"], f"rate_{rps}.json")
```

Run it once per config (restart the server between configs; save `server.log` and `pip freeze` next to the results).

## From JSON to curves

```python
# plot.py
import json, glob, re, matplotlib.pyplot as plt
def load(tag, kind):
    pts = []
    for f in glob.glob(f"results/{tag}/{kind}_*.json"):
        r = json.load(open(f)); x = float(re.findall(r"_(\d+)", f)[-1])
        pts.append((x, r["output_throughput"], r["p99_ttft_ms"], r["p99_tpot_ms"], r.get("request_goodput")))
    return sorted(pts)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for tag in ["bf16", "fp8"]:
    c = load(tag, "conc")
    ax[0].plot([p[1] for p in c], [p[3] for p in c], "o-", label=tag)          # tok/s vs p99 TPOT
    for p in c: ax[0].annotate(int(p[0]), (p[1], p[3]), fontsize=7)
    r = load(tag, "rate")
    ax[1].plot([p[0] for p in r], [p[2] for p in r], "o-", label=tag)          # offered RPS vs p99 TTFT
ax[0].set(xlabel="output tokens/s", ylabel="p99 TPOT (ms)", title="Throughput–latency (labels = concurrency)")
ax[1].set(xlabel="offered load (req/s)", ylabel="p99 TTFT (ms)", yscale="log", title="Latency vs load")
ax[0].axhline(40, ls="--", c="gray"); ax[1].axhline(500, ls="--", c="gray")
for a in ax: a.legend()
plt.tight_layout(); plt.savefig("curves.png", dpi=150)
```

Read the charts like an engineer:
- **Left**: the config whose curve is further **right at the SLO line** wins (more throughput at equal latency). If curves cross, the answer depends on the SLO — say so.
- **Right**: the rate where p99 TTFT crosses 500 ms is the **max RPS at SLO**; the gap between configs is your headline number.
- Then explain the *why* with the dashboard from the next lesson: at the knee, is `num_requests_waiting` rising (compute/scheduler-bound), is `kv_cache_usage_perc` near 1 with preemptions (memory-bound), or is the GPU idle while the client struggles (benchmark-bound)?

> [!NOTE] Quality gate
> Before you celebrate FP8: run a small eval on both configs (e.g. `lm_eval --model local-chat-completions` with a GSM8K or MMLU subset against each server). Report accuracy next to throughput. A 1.5× speedup with −0.3 points is a win; with −5 points it is a bug report.

- [ ] Both configs swept with identical workloads; environment captured (`pip freeze`, `nvidia-smi`, flags)
- [ ] `curves.png` with the SLO lines drawn and concurrency labels
- [ ] Max RPS at SLO and goodput for each config, and the ratio
- [ ] Accuracy check on both configs
- [ ] Two-sentence explanation of *why* the winner wins, backed by a metric
      */}),
      resources: [
        { title: "vLLM — FP8 quantization", url: "https://docs.vllm.ai/en/latest/features/quantization/fp8.html", type: "docs", note: "dynamic FP8 vs FP8 checkpoints, hardware support" },
        { title: "lm-evaluation-harness", url: "https://github.com/EleutherAI/lm-evaluation-harness", type: "tool", note: "quick accuracy checks against a running OpenAI-compatible server" },
      ],
    },
    {
      id: "observability",
      title: "Observability: vLLM metrics, Grafana, GPU telemetry, traces",
      kind: "build",
      minutes: 180,
      runsOn: ["cloud", "mac"],
      md: MD(function () {/*
## Benchmarks tell you *what*; metrics tell you *why*

A benchmark client only sees the outside. The engine and the GPU export what happens inside, and in production you have no benchmark — only these metrics. Baseten (ch. 7.4) lists the minimum: request volume, ISL/OSL distributions, response codes, TTFT / TPS / E2E at p50/p90/p99, replica count, utilization and **queue depth**.

## The vLLM metrics that matter

| Metric (Prometheus name) | Type | PromQL you will actually use | Tells you |
|---|---|---|---|
| `vllm:num_requests_running` / `vllm:num_requests_waiting` | gauge | `sum(vllm:num_requests_waiting)` | Batch size and **queue** — the autoscaling signal (M18) |
| `vllm:kv_cache_usage_perc` | gauge (0–1) | `max(vllm:kv_cache_usage_perc)` | Memory pressure; near 1 → preemptions coming |
| `vllm:num_preemptions_total` | counter | `rate(vllm:num_preemptions_total[5m])` | KV exhausted, work being recomputed |
| `vllm:time_to_first_token_seconds` | histogram | `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` | User-visible responsiveness |
| `vllm:inter_token_latency_seconds` | histogram | same pattern | Streaming smoothness (TPOT) |
| `vllm:e2e_request_latency_seconds` | histogram | same pattern | Total latency |
| `vllm:request_queue_time_seconds` | histogram | same pattern | How much of TTFT is waiting, not prefill |
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | counter | `sum(rate(vllm:generation_tokens_total[1m]))` | Throughput in tokens/s, input vs output |
| `vllm:request_success_total` | counter | `sum by (finished_reason) (rate(vllm:request_success_total[5m]))` | Requests/s; stop vs length |
| `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | counter | ratio of the two `rate()`s | Prefix cache effectiveness |

Names change occasionally between versions — `curl localhost:8000/metrics | grep '^vllm:'` is the source of truth.

## Build: Prometheus + Grafana in one compose file

On a GPU VM next to vLLM, or on your Mac against the **llm-d-inference-sim** (it exports the same metric names), so you can build dashboards without a GPU:

```yaml
# docker-compose.yml
services:
  sim:            # remove this service when scraping a real vLLM
    image: ghcr.io/llm-d/llm-d-inference-sim:v0.11.2
    command: ["--model", "qwen-sim", "--port", "8000", "--max-num-seqs", "8",
              "--time-to-first-token", "200ms", "--inter-token-latency", "20ms"]
    ports: ["8000:8000"]
  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    environment: ["GF_AUTH_ANONYMOUS_ENABLED=true", "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin"]
```

```yaml
# prometheus.yml
global: { scrape_interval: 5s }
scrape_configs:
  - job_name: vllm
    static_configs: [{ targets: ["sim:8000"] }]   # real vLLM on the host: "host.docker.internal:8000"
  - job_name: dcgm
    static_configs: [{ targets: ["host.docker.internal:9400"] }]
```

`docker compose up -d`, open Grafana at `localhost:3000`, add Prometheus (`http://prometheus:9090`) as a data source, and either import the dashboard JSON from vLLM’s `examples/observability/prometheus_grafana` or build these rows yourself: **Traffic** (req/s by finish reason, input/output tok/s) · **Latency** (TTFT, ITL, E2E p50/p99, queue time p99) · **Saturation** (running, waiting, KV usage, preemptions) · **GPU** (below). Run your benchmark sweep and watch the knee happen live.

## GPU metrics with DCGM exporter

`nvidia-smi`’s “utilization” only means *a kernel was running*. The NVIDIA **DCGM exporter** exposes better counters for Prometheus (on Kubernetes it ships with the GPU Operator; on a VM run its container with `--gpus all -p 9400:9400`, using the image tag from its README):

| DCGM field | Meaning | Use |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | % time any kernel ran | Coarse “is it busy” |
| `DCGM_FI_PROF_SM_ACTIVE` | Fraction of SMs with work | Real compute occupancy |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor-core pipe activity | Prefill/compute-bound phases |
| `DCGM_FI_PROF_DRAM_ACTIVE` | Memory bandwidth activity | Decode is memory-bound → this should be high |
| `DCGM_FI_DEV_FB_USED` | Framebuffer (VRAM) used | vLLM preallocates, so mostly flat |
| `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP` | Watts, °C | Throttling, energy per token |
| `DCGM_FI_DEV_XID_ERRORS` | Last Xid error code | Hardware/driver faults → drain the node (M18) |

Profiling (`PROF`) fields may need enabling in the exporter’s counters CSV. With them, you can say things like “at concurrency 64, DRAM active was 0.85 and tensor active 0.30 — still memory-bound; FP8 halved bytes and moved the knee right.”

## Traces and logs

- **Tracing**: vLLM can emit OpenTelemetry spans per request (`--otlp-traces-endpoint grpc://localhost:4317`, with the OTel packages installed; run Jaeger all-in-one to view). Propagate the trace context from your router/app so one trace shows gateway → router → engine (queue, prefill, decode). Great for “why was *this* request slow?”.
- **Logs**: structured JSON per request (id, tenant, ISL, OSL, TTFT, E2E, status, replica) — **not** the prompt text by default (M18 security). Metrics for dashboards and alerts, logs for per-request forensics, traces for cross-service latency.
- **Alerts** (Google SRE golden signals: latency, traffic, errors, saturation): p99 TTFT above SLO for 5 min; `num_requests_waiting` growing for 3 min; KV usage > 0.95; any Xid; 5xx rate > 1%.

- [ ] Grafana dashboard with the four rows, screenshot taken during a sweep showing the knee
- [ ] One PromQL panel computing p99 TTFT from histogram buckets, compared against the benchmark client’s p99
- [ ] DCGM panels on a real GPU (or a written explanation of what you would expect on your hardware)
- [ ] One end-to-end trace (router → engine) viewed in Jaeger, or a written plan for it
- [ ] Three alert rules written as PromQL
      */}),
      resources: [
        { title: "vLLM — Production metrics", url: "https://docs.vllm.ai/en/latest/usage/metrics.html", type: "docs", note: "the full list of exported metrics" },
        { title: "vLLM — Prometheus and Grafana example", url: "https://docs.vllm.ai/en/latest/examples/observability/prometheus_grafana.html", type: "docs", note: "compose file and importable dashboard" },
        { title: "NVIDIA DCGM exporter", url: "https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html", type: "docs", note: "GPU telemetry fields and deployment" },
        { title: "vLLM — OpenTelemetry example", url: "https://docs.vllm.ai/en/latest/examples/observability/opentelemetry.html", type: "docs", note: "tracing setup with Jaeger" },
      ],
    },
    {
      id: "cost",
      title: "Cost modeling: from GPU-hours to dollars per million tokens",
      kind: "math",
      minutes: 120,
      md: MD(function () {/*
## The one formula

$$\text{cost per 1M output tokens} = \frac{\text{GPU price per hour} \times N_\text{GPU}}{\text{output tok/s at SLO} \times 3600 \times U} \times 10^6$$

$U$ is **average utilization** of the capacity you pay for (0–1). Worked with the goal’s numbers — H100 at \$2.50/hr, BF16 goodput-limited throughput 1,810 output tok/s:

- At $U = 1$: $1{,}810 \times 3600 = 6.5$M tokens/hour → \$2.50 / 6.5 = **\$0.38 per 1M**.
- At $U = 0.6$ (realistic for diurnal traffic with autoscaling): **\$0.64 per 1M**.
- FP8 at 2,690 tok/s, $U = 0.6$: **\$0.43 per 1M** — the throughput gain *is* the cost gain.

Three things move cost, in order of typical impact: **utilization** (idle GPUs are the #1 waste), **throughput at SLO** (everything in M09–M17), **GPU price** (reserved vs on-demand vs spot, provider).

::viz cost-calc

## Input tokens are not free

The formula above charges all GPU time to output tokens. A RAG request with 8,000 input tokens spends real time in prefill. Two honest options: report **cost per request** for your workload ($\text{GPU price per second} / (\text{goodput req/s} \times U)$), or split cost between input and output by measured prefill vs decode time. For the goal’s workload (1,000 in / 250 out, 7.3 req/s, $U = 0.6$): \$2.50/3600 per second ÷ (7.3 × 0.6) ≈ **\$0.00016 per request**.

## Dedicated GPUs vs a per-token API

An API at \$0.10 per 1M input and \$0.30 per 1M output tokens would charge $1000 \times 0.10/10^6 + 250 \times 0.30/10^6 \approx$ **\$0.000175 per request** — about the same as the dedicated deployment above at 60% utilization. So when do dedicated GPUs win? Baseten (ch. 7.4) is clear-eyed:

- **Volume and steady utilization**: cost per token falls as $U \to$ high; APIs price in their own idle capacity and margin.
- **Custom or fine-tuned models**, specific quantization, latency SLOs or data residency that shared APIs do not offer.
- **Compare over at least a week** of real traffic (weekends, peaks), not a single benchmark hour.
- **Don’t reverse-engineer** a provider’s per-token price into their GPU cost; pricing includes strategy, batching across customers and discounts.

## Reserved vs on-demand vs spot

| Capacity | Price (illustrative H100) | Good for | Risk |
|---|---|---|---|
| Reserved (1–3 yr) | lowest per hour | Baseline load you always have | You pay 24/7 even at 3 a.m. |
| On-demand | ~1.5–2× reserved | Peaks, experiments | May not be available when you need it (M18 multi-cloud) |
| Spot / preemptible | cheapest, volatile | Batch/offline jobs, evals | Can vanish with minutes of notice |

A common shape: reserve roughly the **trough-to-average** level, burst to on-demand for peaks, send batch work to spot or to idle reserved hours. The blended price and the blended $U$ go into the formula.

## A cost script for your report

```python
# cost.py — $/1M output tokens and $/request for each config
def cost(gpu_hr, n_gpu, out_tps, goodput_rps, util):
    per_s = gpu_hr * n_gpu / 3600
    return {"usd_per_1M_out": per_s / (out_tps * util) * 1e6,
            "usd_per_request": per_s / (goodput_rps * util)}
for name, tps, rps in [("bf16", 1810, 7.3), ("fp8", 2690, 10.8)]:
    for util in (0.4, 0.6, 0.8):
        print(name, util, {k: round(v, 6) for k, v in cost(2.50, 1, tps, rps, util).items()})
```

Add energy if you want a sustainability section: tokens per joule = output tok/s ÷ `DCGM_FI_DEV_POWER_USAGE` watts.

- [ ] Cost per 1M output tokens and per request for both configs at 3 utilization levels
- [ ] Sensitivity table: which input (price, throughput, utilization) moves cost most for your case
- [ ] A comparison with at least one real API’s list price for a comparable model, with the caveats written out
- [ ] One paragraph: at what daily volume dedicated beats the API for your workload
      */}),
      resources: [
        { title: "Inference Engineering (Baseten) — ch. 7.4 cost and observability", url: "Inference%20Engineering.pdf", type: "book", note: "API vs dedicated, what to compare and for how long" },
        { title: "Artificial Analysis", url: "https://artificialanalysis.ai", type: "tool", note: "public API prices, throughput and TTFT by provider" },
      ],
    },
    {
      id: "capacity-planning",
      title: "Capacity planning: users → RPS → tokens/s → GPUs",
      kind: "concept",
      minutes: 90,
      md: MD(function () {/*
## The chain

Product people talk in users; GPUs speak tokens per second. The chain between them:

1. **Users → requests/day**: daily active users × requests per user per day.
2. **Requests/day → average RPS**: ÷ 86,400.
3. **Average → peak RPS**: × peak factor (2–4× for consumer diurnal traffic; measure yours; launches and demos spike higher).
4. **Peak RPS → replicas**: ÷ (max RPS at SLO per replica × headroom), where headroom (0.7–0.8) covers bursts during scale-up cold starts.
5. Cross-check with **Little’s law**: in-flight = peak RPS × mean E2E; ÷ per-replica concurrency target.
6. Add **redundancy**: N+1 per region, or active-active across regions (M18).
7. **Cost**: replica-hours per day from the autoscaled traffic curve × price.

## Worked example: a support-chat product

Assume 50,000 DAU × 8 requests each; ISL ~1,500 (history + system prompt), OSL ~300; SLO TTFT p99 ≤ 500 ms, TPOT p99 ≤ 40 ms; one replica = 1 H100 running the FP8 config, which your sweep (re-run with ISL 1,500 / OSL 300!) says sustains **9 req/s** at SLO with concurrency ≈ 80.

| Step | Calculation | Result |
|---|---|---|
| Requests/day | 50,000 × 8 | 400,000 |
| Average RPS | 400,000 / 86,400 | 4.6 |
| Peak RPS (×3) | 4.6 × 3 | 13.9 |
| Replicas at peak | 13.9 / (9 × 0.75) | 2.06 → **3** |
| Little’s law check | 13.9 × (0.3 + 300 × 0.03) s ≈ 13.9 × 9.3 = 129 in flight; / 80 | 1.6 → consistent |
| Redundancy | N+1 | **4 at peak**, 2 at night |
| Output tokens/day | 400,000 × 300 | 120M |
| GPU-hours/day | ~3 average replicas × 24 | 72 → \$180/day at \$2.50 |

Result: about **\$5,400/month**, or \$1.50 per 1M output tokens (all input processing included). A per-token API for a comparable model might charge less than a third of that at this volume — the honest recommendation for *this* product might be: **start on an API, move to dedicated** once volume grows ~5×, you need a fine-tuned model, or latency/residency requirements force it. The numbers let you say *exactly* when.

> [!INTUITION] The Baseten example
> The book’s rule of thumb is to set the per-replica concurrency target to the batch size where your SLO still holds, then let the autoscaler add replicas as concurrency grows. E.g. 35 peak RPS × 7.8 s E2E ≈ 273 requests in flight; at 48 per replica that is **6 replicas** at peak — the same Little’s law you just used.

## What changes the plan

- **Longer contexts** (agents, RAG): prefill dominates; max RPS per replica falls; prefix caching and cache-aware routing (M18) matter a lot.
- **Reasoning models**: OSL of thousands of tokens → E2E minutes → in-flight counts explode (Little’s law); KV capacity becomes the limit.
- **Mixed tenants**: plan per SLO class; batch traffic fills troughs for free.
- **Growth**: re-run the plan at 2× and 10× volume — where does the architecture break (single region? single GPU type?).

- [ ] Filled the table for a product idea of your own with numbers from **your** benchmark (re-run at matching ISL/OSL)
- [ ] Little’s-law cross-check agrees within ~2× with the RPS-based estimate (if not, explain)
- [ ] Monthly cost with reserved/on-demand split
- [ ] The break-even volume vs a per-token API
      */}),
      resources: [
        { title: "Google SRE workbook — Implementing SLOs", url: "https://sre.google/workbook/implementing-slos/", type: "book", note: "turning user expectations into SLOs and error budgets" },
        { title: "Hamel Husain — Optimizing LLM inference latency", url: "https://hamel.dev/notes/llm/inference/03_inference.html", type: "article", note: "a practitioner’s engine comparison with methodology notes" },
      ],
    },
    {
      id: "report",
      title: "Writing the report people actually read",
      kind: "build",
      minutes: 120,
      md: MD(function () {/*
## The structure

A benchmark report is an argument: *under this workload and SLO, config B beats config A by X, because Y, and it costs Z.* Write it like a blog post from an inference company, not a lab notebook.

1. **TL;DR (3 bullets)** — headline ratio (goodput or max RPS at SLO), cost per 1M tokens for both, recommendation.
2. **Setup** — model (exact revision), engine + version, GPU + driver + CUDA, flags for both configs, client tool + version, where the client ran.
3. **Workload and SLO** — ISL/OSL distributions (histogram), dataset, arrival process, number of requests, warmup, SLO definition and why.
4. **Results** — the two charts from the sweep lesson; a table of max RPS at SLO, goodput, p50/p99 TTFT and TPOT at 3 representative loads.
5. **Why** — dashboard screenshots at the knee (queue, KV usage, preemptions, DRAM/tensor activity) and the mechanism in two paragraphs (link M07/M08/M14 concepts).
6. **Quality** — eval results for both configs.
7. **Cost & capacity** — formulas, assumptions, sensitivity, a capacity plan example.
8. **How this benchmark could be wrong** — limitations: synthetic data, single run spread, client on same host, no prefix sharing, engine defaults, one GPU type.
9. **Reproduce** — exact commands, a `make bench` target, results JSON in the repo.

## Charts that earn their place

| Chart | Axes | Message |
|---|---|---|
| Throughput–latency | output tok/s vs p99 TPOT, points labeled by concurrency, SLO line | Who wins at equal latency |
| Latency vs load | offered req/s vs p99 TTFT (log y), SLO line | Max RPS at SLO |
| Goodput bars | config vs goodput at 2–3 load levels | Headline ratio |
| Latency CDF | TTFT CDF for both configs at one load | Tail shape, not just p99 |
| Dashboard panel | time vs queue/KV usage during the sweep | The mechanism |
| Cost bars | config × utilization → \$/1M tokens | The business answer |

Rules: label units; start y at zero unless using log scale (say so); same colors per config everywhere; every chart has one sentence below it saying what to notice.

## Tone and honesty

- Prefer **ratios with conditions** (“1.47× goodput at TTFT p99 ≤ 500 ms”) over absolutes (“2,690 tok/s”).
- Show **error bars** or the min–max of 3 runs.
- State **negative results** (“SGLang’s TTFT was worse below concurrency 8”).
- Never claim more than you measured: one GPU type is not “FP8 is 1.5× faster”.

> [!NOTE] Read two real ones first
> Skim a Baseten, Modal or Fireworks engineering blog benchmark post and the InferenceMAX methodology page. Note what they disclose (and what they omit). Your report should disclose more.

- [ ] Draft README with all 9 sections, even if some are stubs
- [ ] Every chart has a caption sentence
- [ ] Someone else ran your `REPRODUCE` steps (or you did, in a fresh environment) and got numbers within 10%
      */}),
      resources: [
        { title: "Baseten blog", url: "https://www.baseten.co/blog/", type: "article", note: "examples of production benchmark write-ups" },
        { title: "Modal — stopwatch", url: "https://github.com/modal-labs/stopwatch", type: "repo", note: "an open benchmarking harness and how it reports results" },
        { title: "InferenceMAX dashboard", url: "https://inferencemax.semianalysis.com", type: "tool", note: "public cross-GPU results to sanity-check yours" },
      ],
    },
  ],

  challenge: {
    title: "The benchmark report (blog-style README with charts)",
    md: MD(function () {/*
Publish `course-work/m19-benchmark/` as a public-quality repository whose `README.md` is the report from the last lesson, comparing two configurations of one model on one GPU type (FP8 vs BF16, vLLM vs SGLang, AWQ vs FP16, prefix caching on/off, or TP=2 vs 2×TP=1).

```text
m19-benchmark/
  README.md                # the blog-style report
  REPRODUCE.md             # exact commands, versions, hardware
  bench/sweep.py, plot.py, cost.py, goodput.py
  results/<config>/*.json  # raw benchmark outputs
  observability/docker-compose.yml, prometheus.yml, grafana-dashboard.json
  charts/*.png
```

Must include: concurrency sweep and rate sweep for both configs; goodput and max RPS at a stated SLO; Grafana screenshots (engine metrics, plus DCGM if on a real GPU) explaining the knee; an accuracy check; cost per 1M tokens and per request at several utilizations; a capacity plan for a hypothetical product; a limitations section. Bonus points for also benchmarking your M10 mini engine on the same workload as a “from scratch vs production” appendix.
    */}),
    checklist: [
      "Workload, SLO, versions and hardware are stated before any result",
      "Both configs swept identically (concurrency and rate), with ≥ 1,000 requests behind every p99",
      "Throughput–latency and latency-vs-load charts with SLO lines; goodput/max-RPS table with the headline ratio",
      "Dashboard screenshots and a metric-backed explanation of why the winner wins",
      "Cost per 1M tokens and per request with utilization sensitivity, plus an API comparison with caveats",
      "“How this benchmark could be wrong” section and a working REPRODUCE.md",
    ],
    stretch: "Replay a realistic trace (BurstGPT via `--dataset-name burstgpt`, or timestamps from your own logs) against both configs behind your M18 autoscaled deployment, and report SLO attainment and replica-hours (i.e. real cost) instead of steady-state numbers.",
  },

  connects: MD(function () {/*
This capstone closes the serving arc: M06–M17 made one replica fast, M18 made it a service, and M19 proves — with numbers, dashboards and dollars — what that is worth. The same methodology applies to every modality in M20 (with different metrics), to the rollout engines of M22–M23 (where throughput *is* RL training speed), and it is the centerpiece of your portfolio in M25.
  */}),

  interview: [
    "What is the difference between open-loop and closed-loop load testing, and which would you use to find the max RPS a replica can serve within SLO?",
    "Define goodput. Why is it a better headline number than throughput?",
    "Your p99 TTFT doubles at 80% of max throughput while p99 TPOT is flat. What is happening and which metric proves it?",
    "Why can’t you average p99 latencies from ten replicas? How do you compute a fleet p99 correctly?",
    "Estimate the cost per 1M output tokens for a 7B model on one H100 at \\$2.50/hr doing 2,000 output tok/s at 50% utilization. When is an API cheaper?",
    "A product expects 100K DAU with 10 requests each. Walk me through sizing the GPU fleet.",
    "Two benchmark tools report TTFT 15% apart on the same server. List reasons.",
    "Which GPU metrics would tell you decode is memory-bandwidth-bound rather than compute-bound?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — local PDF", url: "Inference%20Engineering.pdf", type: "book", note: "ch. 1.4 metrics, 4.5 benchmarking, 7.4 cost and observability" },
    { title: "vLLM — bench serve CLI", url: "https://docs.vllm.ai/en/latest/cli/bench/serve.html", type: "docs", note: "the primary tool for this capstone" },
    { title: "NVIDIA AIPerf", url: "https://github.com/ai-dynamo/aiperf", type: "tool", note: "second opinion on every number" },
    { title: "GuideLLM", url: "https://github.com/vllm-project/guidellm", type: "tool", note: "sweeps and rate finding" },
    { title: "DistServe", url: "https://arxiv.org/abs/2401.09670", type: "paper", note: "goodput under TTFT/TPOT SLOs" },
    { title: "vLLM — Production metrics", url: "https://docs.vllm.ai/en/latest/usage/metrics.html", type: "docs", note: "what each Prometheus metric means" },
    { title: "NVIDIA DCGM exporter", url: "https://github.com/NVIDIA/dcgm-exporter", type: "repo", note: "GPU metrics for Prometheus" },
    { title: "Google SRE book — Monitoring distributed systems", url: "https://sre.google/sre-book/monitoring-distributed-systems/", type: "book", note: "golden signals and alerting philosophy" },
    { title: "Artificial Analysis", url: "https://artificialanalysis.ai", type: "tool", note: "API price/performance to compare against" },
  ],
});
