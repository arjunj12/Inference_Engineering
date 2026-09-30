Course.module({
  id: "m10-mini-engine",
  title: "Capstone: build your own mini-vLLM",
  short: "Capstone: mini-vLLM",
  tagline: "Assemble everything from M04–M09 into a ~1,500-line inference engine with paged KV, prefix caching, continuous batching, streaming and an OpenAI-compatible API — then benchmark it against HF generate and vLLM.",
  hours: 40,
  level: "core",
  capstone: true,
  runsOn: ["mac", "colab"],
  tags: ["capstone", "inference-engine", "vllm", "paged-attention", "continuous-batching", "openai-api", "sse", "portfolio"],

  goal: MD(function () {/*
At the end of this module you own **mini-vllm**: a GitHub repo with an inference engine you wrote yourself, a design doc, and benchmark charts. It serves a real model on your Mac (MPS) and on Colab (CUDA) behind the same API your future employer's clients use:

```bash
python -m minivllm.serve --model Qwen/Qwen2.5-0.5B-Instruct --device mps --port 8000

curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "One sentence: what is paged attention?"}],
  "stream": true, "max_tokens": 40, "temperature": 0.7, "top_p": 0.9}'
```

```text
data: {"id":"chatcmpl-3f9c…","object":"chat.completion.chunk","model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-3f9c…","object":"chat.completion.chunk","model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":"Paged"},"finish_reason":null}]}

data: {"id":"chatcmpl-3f9c…","object":"chat.completion.chunk","model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{"content":" attention"},"finish_reason":null}]}
…
data: {"id":"chatcmpl-3f9c…","object":"chat.completion.chunk","model":"Qwen2.5-0.5B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

…and a `BENCHMARKS.md` with a table like this one (illustrative shape — your numbers will differ, and that is the point of measuring):

```text
Qwen2.5-0.5B-Instruct, 256 requests, prompt 256 tok, output 128 tok (ignore_eos), greedy
engine                                   device        out tok/s   TTFT p50   ITL p50
HF generate, one request at a time       M2 Pro MPS          30        —        33 ms
HF generate, static batch 16 (padded)    M2 Pro MPS         240        —        —
mini-vllm, 64 seqs, chunked 512          M2 Pro MPS         420     0.9 s       80 ms
mini-vllm, eager                         Colab L4          1,900     0.4 s       30 ms
mini-vllm + CUDA graphs                  Colab L4          3,500     0.4 s       16 ms
vLLM (same model, defaults)              Colab L4          9,000     0.1 s        7 ms
```

You will not beat vLLM — and you'll be able to explain, line by line, *where* the remaining gap lives. That explanation is the portfolio piece.

::viz inference-stack
  */}),
  demo: { viz: "inference-stack", params: {} },

  why: MD(function () {/*
"Have you built an inference engine?" is the single most discriminating question in inference-engineering interviews, because an engine forces you to get **everything** right at once: model math, memory management, scheduling, numerics, async I/O and measurement. Runtime teams (vLLM, SGLang, TensorRT-LLM, TGI) and the platform teams around them (Baseten, Fireworks, Together, Modal, every big lab's serving org) all work on exactly this loop: *schedule → prepare inputs → forward → sample → update → stream*. The Baseten book's chapter on runtimes (Inference Engineering (Baseten) ch. 4.3) describes vLLM, SGLang and TensorRT-LLM in terms of the same components you will build here — after this module those descriptions stop being vocabulary and start being code you've debugged.

Two public projects prove the scope is realistic: **nano-vllm** (~1,200 lines of Python, offline only, matches vLLM's throughput on an RTX 4070 laptop GPU) and **PageServe** (a readable PyTorch engine built in 12 documented phases, runs on CUDA *and* Apple MPS, OpenAI-compatible SSE API, 28× over HF generate on a T4). You'll build something between the two — and then read both, and vLLM itself, with the eyes of someone who has made the same decisions.
  */}),

  prereqs: [
    {
      title: "asyncio, async generators and Server-Sent Events in 5 minutes",
      skipIf: "you have written a FastAPI endpoint that streams with StreamingResponse",
      md: MD(function () {/*
**asyncio** runs many tasks on one thread. A task gives up the thread at every `await`, so one slow request doesn't block the others — as long as nobody does heavy CPU work (like a model forward!) on the event loop. That's why the engine lives in *another process*.

An **async generator** is a function with both `async def` and `yield`. You consume it with `async for`:

```python
import asyncio

async def ticker(n):
    for i in range(n):
        await asyncio.sleep(0.1)     # gives other tasks a turn
        yield f"tick {i}"

async def main():
    async for t in ticker(3):
        print(t)

asyncio.run(main())
```

**Server-Sent Events (SSE)** is the streaming format OpenAI-compatible APIs use: a normal HTTP response with `Content-Type: text/event-stream` that is never "finished" until the server closes it. Each event is a line `data: <payload>` followed by a **blank line**. OpenAI's convention: payload is a JSON chunk; the final event is the literal `data: [DONE]`.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.get("/stream")
async def stream():
    async def events():
        async for t in ticker(5):
            yield f"data: {json.dumps({'text': t})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
```

Run it with `uvicorn app:app` and watch with `curl -N localhost:8000/stream` (`-N` turns off curl's buffering). When the client disconnects, Starlette **cancels** the task iterating your generator — your `finally:` block runs. You'll use that to abort requests in the engine.
      */}),
    },
    {
      title: "Processes, spawn and queues (why the engine gets its own process)",
      skipIf: "you know why `fork` + CUDA is a bad idea and have used multiprocessing.Queue",
      md: MD(function () {/*
Python threads share one interpreter lock (the GIL), so a CPU-heavy loop in one thread slows every other thread. vLLM V1 therefore runs the **engine core** (scheduler + model) in its own **process**, and keeps the HTTP server, tokenization and detokenization in the **API process**. They talk through queues (vLLM uses ZeroMQ sockets; `multiprocessing.Queue` is fine for us).

```python
import multiprocessing as mp

def worker(in_q, out_q):              # runs in the child process
    while (msg := in_q.get()) != "stop":
        out_q.put(msg.upper())

if __name__ == "__main__":            # required with spawn: the child re-imports this file
    ctx = mp.get_context("spawn")
    in_q, out_q = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=worker, args=(in_q, out_q), daemon=True)
    p.start()
    in_q.put("hello"); print(out_q.get())   # HELLO
    in_q.put("stop"); p.join()
```

Rules that will save you an evening:

- Use the **`spawn`** start method. `fork` copies a process that may already hold CUDA/MPS state, which ends in hangs or "cannot re-initialize CUDA in forked subprocess".
- Everything you pass to `Process(args=...)` or put on a queue is **pickled**. Send token ids and small dataclasses, never tensors or models. Load the model *inside* the child.
- `queue.get()` blocks; `queue.get_nowait()` raises `queue.Empty`. An idle engine should block; a busy engine should drain without blocking.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: two tiny engines and one big one, on the wire",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Before designing, run the thing you're about to build — three times, at three sizes.

## 1. PageServe on your Mac (the "readable" end)

PageServe is a from-scratch PyTorch engine that runs on MPS. Clone it and follow its README quick-start (install, run the tests, start the server):

```bash
git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine pageserve
cd pageserve
# follow README: create a venv, pip install the requirements, run its tests,
# then start the OpenAI-compatible server it ships (server/app_v2.py)
```

Things to notice while it runs: its `tests/` compare token-for-token against HF `generate` (you'll do the same); its README has a **12-phase build log** (sequential baseline → scheduler → request queue → chunked prefill → KV tracking → block allocator → paged KV → paged attention → CPU swap → metrics → load testing → prefix cache/spec decode/streaming/OpenAI API) — that's a proven milestone order, and ours below is a compressed version of it; and its MPS notes (non-blocking copies racing, `index_copy_` being slow) are bugs you won't have to rediscover.

## 2. nano-vllm on Colab (the "fast and small" end)

nano-vllm is ~1,200 lines and reaches vLLM-level offline throughput. It uses `flash-attn`, which needs an **Ampere or newer** GPU: pick **L4 or A100** in Colab (the free T4 won't work).

```python
!pip -q install git+https://github.com/GeeeekExplorer/nano-vllm.git
!huggingface-cli download Qwen/Qwen3-0.6B --local-dir /content/Qwen3-0.6B

from nanovllm import LLM, SamplingParams
llm = LLM("/content/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
out = llm.generate(["Explain paged attention in one sentence."],
                   SamplingParams(temperature=0.6, max_tokens=64))
print(out[0]["text"])
```

Then run its `bench.py` with and without `enforce_eager=True`. The difference is **CUDA graphs** — one of the optional CUDA speed-ups at the end of this module.

## 3. vLLM on Colab — look at the bytes

```bash
pip -q install vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096 &
# wait for "Application startup complete", then:
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"The KV cache","max_tokens":8,"stream":true}'
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests|kv_cache_usage|prefix_cache)" | head
```

Copy the raw SSE lines into your notes: `object`, `choices[0].text`, `finish_reason`, the final `data: [DONE]`. Your server must produce the same shapes — that's what lets **any** OpenAI client (and `vllm bench serve`) talk to it. Note the metric names too: running/waiting requests, KV-cache usage, prefix-cache hits. Your `/metrics` will expose the same signals.

> [!REAL]
> "OpenAI-compatible" is the industry's lingua franca: vLLM, SGLang, TGI, Ollama, llama.cpp's server, and every inference provider speak it. Matching the wire format exactly — including the details like `stream_options.include_usage` — is what makes an engine drop-in.

## Exercises

- [ ] Run PageServe's tests on your Mac; find the test that checks parity with HF `generate`.
- [ ] Run nano-vllm's bench on an L4/A100 with and without `enforce_eager`; record both tok/s.
- [ ] Save 10 raw SSE lines from vLLM for `/v1/completions` and `/v1/chat/completions`; they are your API test fixtures.
      */}),
      resources: [
        { title: "PageServe (vLLM_Inference_Engine)", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "12-phase build log; runs on MPS; token-exact tests" },
        { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "~1,200-line engine; example.py and bench.py" },
        { title: "vLLM — OpenAI-compatible server", url: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html", type: "docs", note: "the API surface you're cloning" },
      ],
    },
    {
      id: "architecture",
      title: "Architecture: the engine loop, two processes, interfaces and a test plan",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
Every modern engine — vLLM V1, SGLang, nano-vllm, PageServe — has the same skeleton. Learn it once and every codebase becomes a variation.

## The engine core loop

```text
                    ┌───────────────────────── EngineCore.step() ─────────────────────────┐
 add_request ──►    │ 1 schedule      which seqs, how many tokens each (token budget,      │
 abort       ──►    │                 chunked prefill, prefix-cache hits, preemption)      │
                    │ 2 prepare       flatten chunks into one packed batch: input_ids,     │
                    │   inputs        positions, slot_mapping, block tables, sampling params│
                    │ 3 forward       your Qwen2 over [T] tokens; attention reads/writes   │
                    │                 the paged KV pool                                    │
                    │ 4 sample        logits at each sampling row -> next token (per-      │
                    │                 request temperature / top-k / top-p / seed)          │
                    │ 5 update        append tokens, advance num_computed, detect EOS /    │
                    │                 max_tokens, free blocks, publish prefix hashes       │
                    │ 6 emit          EngineCoreOutput(request_id, new_token_ids, finish)  │
                    └───────────────────────────────────────────────────────────────────────┘
```

Aleksa Gordić's walkthrough of vLLM V1 describes exactly this: the scheduler decides, the model runner "prepares inputs" (positions, `slot_mapping`, attention metadata), **flattens all sequences into one super-sequence**, runs the forward, gathers the hidden state at each sequence's last token and samples. Your M09 scheduler is step 1; M08's block manager is inside steps 1 and 5.

## Two processes, like vLLM V1

```text
  API process (asyncio, one thread)                         Engine-core process (busy loop)
 ┌───────────────────────────────────────┐   in_q: add/abort  ┌──────────────────────────────┐
 │ FastAPI: /v1/completions, /v1/chat/…  │ ─────────────────► │ drain in_q                   │
 │ tokenize, chat template               │                    │ step(): schedule → forward → │
 │ AsyncEngine: one asyncio.Queue / req  │ ◄───────────────── │         sample → update       │
 │ detokenize, stop strings, SSE, metrics│  out_q: outputs    │ block when idle              │
 └───────────────────────────────────────┘                    └──────────────────────────────┘
```

Why split? The model forward holds the CPU (and the GIL) for tens of milliseconds; if it ran on the event loop, HTTP handling would stall and streaming would stutter. vLLM V1 puts the engine core in its own process, talks over **ZeroMQ**, and does detokenization in the front-end's output processor so the GPU loop never waits on string work. For debugging, vLLM lets you collapse both into one process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); your engine should allow the same (the offline `LLM` class below does it by construction).

## Recommended repo layout

```text
mini-vllm/
├── README.md                 # what, results chart, quick-start, design summary
├── docs/DESIGN.md            # the design doc (challenge)
├── BENCHMARKS.md             # tables + charts, exact commands, hardware
├── pyproject.toml
├── minivllm/
│   ├── request.py            # SamplingParams, Request, Sequence, SchedulerOutput, EngineCoreOutput
│   ├── block_manager.py      # M08 + prefix caching (ref counts, chain hashes, LRU)
│   ├── scheduler.py          # M09 + prefix cache + abort + stop tokens
│   ├── qwen2.py              # YOUR model (M04), packed-batch forward, HF weight names
│   ├── attention.py          # TorchPagedAttention (Mac); FlashAttn/FlashInfer backends (CUDA)
│   ├── sampler.py            # batched per-request sampling
│   ├── model_runner.py       # prepare_inputs + forward + sample
│   ├── llm_engine.py         # EngineCore.step(), offline LLM.generate()
│   ├── detokenizer.py        # incremental detokenizer + stop strings
│   ├── async_engine.py       # engine-core process + AsyncEngine client
│   ├── server/api.py         # FastAPI app: completions, chat, models, health, metrics
│   └── serve.py              # CLI: python -m minivllm.serve --model … --device …
├── tests/                    # see "testing strategy"
├── bench/                    # bench_offline.py, bench_serve.sh, plot.py, results/*.json
└── notebooks/colab.ipynb     # CUDA run + vLLM comparison
```

## The interfaces (write these first)

```python
# minivllm/request.py
import enum, time
from dataclasses import dataclass, field

@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0          # 0 = greedy
    top_p: float = 1.0
    top_k: int = -1                   # -1 = off
    seed: int | None = None
    stop: list[str] = field(default_factory=list)            # text-level: API process
    stop_token_ids: list[int] = field(default_factory=list)  # token-level: engine
    ignore_eos: bool = False

@dataclass
class Request:
    """What the client asked for, already tokenized. Immutable."""
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0                 # lower = more important (vLLM convention)

class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"         # EOS / stop token
    FINISHED_LENGTH = "length"        # max_tokens
    FINISHED_ABORTED = "abort"        # client disconnected / stop string matched

    @property
    def finished(self) -> bool:
        return self.value in ("stop", "length", "abort")

class Sequence:
    """The engine's mutable state for one Request."""
    def __init__(self, req: Request):
        self.seq_id, self.req, self.params = req.request_id, req, req.sampling_params
        self.token_ids = list(req.prompt_token_ids)
        self.num_prompt_tokens = len(req.prompt_token_ids)
        self.num_computed = 0         # tokens whose K/V are in the cache
        self.num_cached = 0           # of those, served by the prefix cache
        self.status = SeqStatus.WAITING
        self.num_preemptions = 0

    @property
    def num_tokens(self): return len(self.token_ids)
    @property
    def output_ids(self): return self.token_ids[self.num_prompt_tokens:]
    @property
    def num_uncomputed(self): return self.num_tokens - self.num_computed

@dataclass
class ScheduledChunk:
    seq: Sequence
    start: int                        # first position computed this step
    n: int                            # tokens computed this step

    @property
    def samples(self) -> bool:        # reaches the last known token -> produces a new one
        return self.start + self.n == self.seq.num_tokens

@dataclass
class SchedulerOutput:
    chunks: list[ScheduledChunk]
    num_batched_tokens: int
    preempted_ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self): return not self.chunks

@dataclass
class EngineCoreOutput:
    """The only thing that crosses the process boundary, per request per step."""
    request_id: str
    new_token_ids: list[int]
    finish_reason: str | None = None  # "stop" | "length" | "abort" | None
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0

# and the contract the runner must meet:
#   class ModelRunner:
#       def execute(self, so: SchedulerOutput) -> dict[str, int]   # seq_id -> next token
```

The single most useful idea (from M09, and it's how vLLM V1 thinks): there is no "prefill phase" and "decode phase" in the scheduler's data model — only **`num_computed` catching up with `num_tokens`**. A fresh prompt, a chunk, a decode step, a recompute after preemption and a prefix-cache hit are all the same operation with different numbers. vLLM V1's `SchedulerOutput` similarly carries `num_scheduled_tokens` per request.

## Testing strategy

| layer | test | why it catches real bugs |
|---|---|---|
| model | **greedy parity with HF** `generate` on a tiny random Qwen2 in **fp32 on CPU** (exact match), and on the real 0.5B for the first ~32 tokens | RoPE convention, GQA head mapping, norm dtype, off-by-one positions |
| engine | same parity with **chunking on, preemption forced (tiny pool), prefix caching on**, and a second pass that hits the cache | any slot-mapping or `num_computed` bug changes tokens |
| golden outputs | commit `tests/golden/*.json` (prompt → greedy tokens) generated once with HF; compare on every change | catches regressions when you optimize |
| block manager | conservation: after every request finishes, `num_free_blocks == num_blocks`; ref counts never negative | leaks show up only under load otherwise |
| sampler | frequency tests on a fixed 4-token distribution (top-k, top-p, temperature); seeded requests reproduce **regardless of batch composition** | silent sampling bias |
| detokenizer | roundtrip on emoji/CJK/accents; stop string never leaks into output | U+FFFD garbage in streams |
| API | a **fake engine core** (no model) behind the real server; the official `openai` client for stream/non-stream/stop/usage; disconnect ⇒ abort reaches the engine | protocol bugs in milliseconds, no GPU needed |
| performance | `bench/` scripts with fixed seeds, warmup, JSON results, plotted | "faster" claims that are actually noise |

> [!WARNING]
> Exact token parity only holds in **fp32** (and on the same device). In fp16/bf16, a different batch shape changes the reduction order, the logits change in the 3rd decimal, and greedy decoding eventually diverges. That's normal — vLLM isn't batch-invariant by default either. Test exactness in fp32; test fp16 with "first N tokens match on most prompts" plus perplexity checks.

## Milestones (each ends with an acceptance test)

| # | milestone | acceptance test |
|---|---|---|
| M0 | repo, interfaces, HF-`generate` sequential baseline + timing | baseline tok/s recorded |
| M1 | your Qwen2 on a packed batch + paged KV (`ModelRunner`) | greedy parity (fp32 tiny + real model) |
| M2 | scheduler + prefix caching + `EngineCore.step()` + offline `LLM.generate` | parity under chunking/preemption/prefix; blocks conserved; ≥3× baseline offline |
| M3 | sampler + incremental detokenizer + stop strings | distribution + seed + roundtrip tests |
| M4 | engine-core process + FastAPI + SSE + metrics | `openai` client tests pass against fake and real engine; disconnect aborts |
| M5 | benchmarks vs HF (Mac + Colab) and vLLM (Colab), charts | `BENCHMARKS.md` with reproducible commands |
| M6 (opt.) | CUDA graphs, FlashAttention/FlashInfer, `torch.compile` | each change: parity + measured speed-up |
| M7 | design doc, README, compare with nano-vllm / PageServe / vLLM | challenge checklist |

- [ ] Draw the loop and the two-process diagram from memory.
- [ ] Create the repo with the layout above, empty modules, and `request.py` interfaces. Commit.
- [ ] Write `tests/test_parity.py` **before** the runner: it should fail with `NotImplementedError`.
      */}),
      resources: [
        { title: "Aleksa Gordić — Inside vLLM: Anatomy of a High-Throughput LLM Inference System", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "engine core, scheduler, model runner, AsyncLLM + ZMQ serving layer" },
        { title: "vLLM V1 alpha announcement", url: "https://blog.vllm.ai/2025/01/27/v1-alpha-release.html", type: "article", note: "why V1 moved to a unified scheduler and a separate engine-core process" },
      ],
    },
    {
      id: "ms-model-runner",
      title: "Milestones M0–M1: baseline, your model on a packed batch, paged attention",
      kind: "build",
      minutes: 300,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
## M0 — the number to beat

Every claim in your README will be "N× faster than X", so X must be honest. `bench/hf_baseline.py`:

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "Qwen/Qwen2.5-0.5B-Instruct"
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(dev).eval()
sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)

prompts = [f"Write a short story about robot number {i}." for i in range(16)]
model.generate(**tok(prompts[0], return_tensors="pt").to(dev), max_new_tokens=8)   # warmup
sync(); t0, n = time.perf_counter(), 0
for p in prompts:                                     # one request at a time
    x = tok(p, return_tensors="pt").to(dev)
    out = model.generate(**x, max_new_tokens=128, min_new_tokens=128, do_sample=False)
    n += out.shape[1] - x.input_ids.shape[1]
sync(); print(f"HF sequential: {n / (time.perf_counter() - t0):.1f} out tok/s")
```

(`dtype=` is the current name of the old `torch_dtype=` argument.) Record the number with your hardware, torch version and date. Also try a **static padded batch** (`tok.padding_side = "left"`, all 16 prompts in one `generate` call) — that's the stronger baseline an interviewer will ask about.

## M1 — your model, rewritten for a packed batch

Take your M04 Qwen2/Llama and change **one thing**: the forward takes a flat `[T]` vector of tokens from *many* sequences plus their `positions`, and attention is delegated to a backend object that knows the paged KV layout. Keep HF parameter names so safetensors load with no renaming. The full file is ~120 lines; the only new idea is the `attn` argument threaded through every layer.

<details><summary>minivllm/qwen2.py — the full model + safetensors loader</summary>

```python
# minivllm/qwen2.py
import torch
import torch.nn.functional as F
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps
    def forward(self, x):
        dt = x.dtype
        x = x.float()                                        # HF normalizes in fp32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)

def apply_rope(x, cos, sin):                  # x [T, heads, D]; HF rotate_half convention
    d = x.shape[-1] // 2
    rot = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]

class Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.H, self.KVH = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        bias = getattr(cfg, "attention_bias", True)          # Qwen2: True, Llama: False
        self.q_proj = nn.Linear(cfg.hidden_size, self.H * self.D, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.KVH * self.D, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.KVH * self.D, bias=bias)
        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)
        self.layer_idx = layer_idx
    def forward(self, x, cos, sin, attn):
        T = x.shape[0]
        q = apply_rope(self.q_proj(x).view(T, self.H, self.D), cos, sin)
        k = apply_rope(self.k_proj(x).view(T, self.KVH, self.D), cos, sin)
        v = self.v_proj(x).view(T, self.KVH, self.D)
        o = attn.forward(self.layer_idx, q, k, v)            # writes KV, reads context
        return self.o_proj(o.reshape(T, self.H * self.D))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, i)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
    def forward(self, x, cos, sin, attn):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn)
        return x + self.mlp(self.post_attention_layernorm(x))

class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
        self._inv_freq = None             # fp32 on purpose: not a buffer, so .to(fp16) can't touch it

    def inv_freq(self, device):
        if self._inv_freq is None or self._inv_freq.device != device:
            D = self.head_dim
            self._inv_freq = 1.0 / self.theta ** (torch.arange(0, D, 2, device=device).float() / D)
        return self._inv_freq

    def forward(self, input_ids, positions, attn):
        """input_ids, positions: [T] (all sequences packed). Returns hidden states [T, C]."""
        freqs = positions.float()[:, None] * self.inv_freq(positions.device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        x = self.model.embed_tokens(input_ids)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cos, sin, attn)
        return self.model.norm(x)

    def compute_logits(self, h):
        return self.lm_head(h).float()

def load_qwen2(model_id: str, device, dtype):
    """Build the model from an HF repo id (or local dir) using config.json + *.safetensors."""
    import json, glob, os
    from types import SimpleNamespace
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = model_id if os.path.isdir(model_id) else snapshot_download(
        model_id, allow_patterns=["*.json", "*.safetensors"])
    cfg = SimpleNamespace(**json.load(open(os.path.join(path, "config.json"))))
    with torch.device("meta"):                               # no RAM for random init
        model = Qwen2ForCausalLM(cfg)
    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(f))
    model.load_state_dict(state, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.to(device=device, dtype=dtype).eval(), cfg
```

</details>

> [!WARNING]
> Two bugs that pass casual testing: (1) keeping `inv_freq` as a registered buffer — `model.to(float16)` silently rounds it and positions beyond a few hundred drift; (2) Llama 3.x uses `rope_scaling` ("llama3" frequency scaling) — this loader ignores it, so stick to Qwen2.5 (or implement the scaling and test parity).

## The attention backend: paged KV in plain PyTorch

Metadata is built **once per step** (`begin_step`), then reused by all layers. Decode rows (1 query token) are batched into one padded SDPA call; prefill chunks each get a causal mask with an offset (query position `start + i` may see keys `0 … start + i`). The block table → slot arithmetic is the same picture as M08:

::viz paged-attention

```python
# minivllm/attention.py
import torch
import torch.nn.functional as F

class TorchPagedAttention:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, num_heads,
                 device, dtype):
        S = num_blocks * block_size
        self.k = torch.zeros(num_layers, S, num_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.g = num_heads // num_kv_heads                   # GQA group size
        self.device = device

    def begin_step(self, slot_mapping, decode, prefill):
        """slot_mapping: [T] where each new token's K/V goes.
        decode:  list of (row, ctx_slots)              -- 1 query token each
        prefill: list of (row0, start, n, ctx_slots)   -- a chunk of n query tokens"""
        dev = self.device
        self.slots, self.dec = slot_mapping, None
        if decode:
            Lmax = max(len(c) for _, c in decode)
            idx = torch.zeros(len(decode), Lmax, dtype=torch.long)
            valid = torch.zeros(len(decode), Lmax, dtype=torch.bool)
            for b, (_, c) in enumerate(decode):
                idx[b, :len(c)] = torch.tensor(c)
                valid[b, :len(c)] = True
            rows = torch.tensor([r for r, _ in decode])
            self.dec = (rows.to(dev), idx.to(dev), valid.to(dev)[:, None, None, :])
        self.pre = []
        for r0, start, n, c in prefill:
            qpos = torch.arange(start, start + n)
            mask = torch.arange(len(c))[None, :] <= qpos[:, None]        # causal with offset
            self.pre.append((r0, n, torch.tensor(c, device=dev), mask.to(dev)))

    def forward(self, layer, q, k, v):
        K_pool, V_pool = self.k[layer], self.v[layer]
        K_pool[self.slots] = k                               # index_copy_ is slow on MPS
        V_pool[self.slots] = v
        o = torch.empty_like(q)
        if self.dec is not None:
            rows, idx, valid = self.dec
            K = K_pool[idx].transpose(1, 2).repeat_interleave(self.g, 1)   # [B, H, L, D]
            V = V_pool[idx].transpose(1, 2).repeat_interleave(self.g, 1)
            o[rows] = F.scaled_dot_product_attention(q[rows].unsqueeze(2), K, V,
                                                     attn_mask=valid).squeeze(2)
        for r0, n, c, mask in self.pre:
            K = K_pool[c].transpose(0, 1).repeat_interleave(self.g, 0)     # [H, ctx, D]
            V = V_pool[c].transpose(0, 1).repeat_interleave(self.g, 0)
            o[r0:r0 + n] = F.scaled_dot_product_attention(
                q[r0:r0 + n].transpose(0, 1), K, V, attn_mask=mask).transpose(0, 1)
        return o
```

This **gathers** each sequence's context out of the pool every layer — correct and portable, but it copies the whole KV once per step. Real paged kernels (vLLM's PagedAttention, FlashAttention's `block_table` argument, FlashInfer) read blocks in place. That gather is the #1 line to point at when explaining your gap to vLLM.

## The model runner: prepare inputs → forward → sample

```python
# minivllm/model_runner.py
import torch
from .attention import TorchPagedAttention
from .sampler import Sampler

class ModelRunner:
    def __init__(self, model, cfg, bm, device, dtype):
        self.model, self.bm, self.device = model, bm, device
        H, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        self.attn = TorchPagedAttention(cfg.num_hidden_layers, bm.num_blocks, bm.block_size,
                                        KVH, D, H, device, dtype)
        self.sampler = Sampler()
        self.generators: dict[str, torch.Generator] = {}

    def _slots(self, table, positions):                     # logical position -> pool slot
        bs = self.bm.block_size
        return [table[p // bs] * bs + p % bs for p in positions]

    def prepare_inputs(self, so):
        ids, pos, slots, decode, prefill, last, samp = [], [], [], [], [], [], []
        for c in so.chunks:
            seq, table = c.seq, self.bm.block_table(c.seq.seq_id)
            p = range(c.start, c.start + c.n)
            row0 = len(ids)
            ids += seq.token_ids[c.start:c.start + c.n]
            pos += p
            slots += self._slots(table, p)
            ctx = self._slots(table, range(c.start + c.n))
            if c.n == 1:
                decode.append((row0, ctx))
            else:
                prefill.append((row0, c.start, c.n, ctx))
            if c.samples:                                    # last row of this chunk -> logits
                last.append(row0 + c.n - 1)
                samp.append(seq)
        t = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        self.attn.begin_step(t(slots), decode, prefill)
        return t(ids), t(pos), t(last), samp

    def _sampling_tensors(self, seqs):
        gens = []
        for s in seqs:
            if s.params.seed is not None and s.seq_id not in self.generators:
                self.generators[s.seq_id] = torch.Generator().manual_seed(s.params.seed)
            gens.append(self.generators.get(s.seq_id))
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=self.device)
        return (f([s.params.temperature for s in seqs]),
                torch.tensor([s.params.top_k for s in seqs], device=self.device),
                f([s.params.top_p for s in seqs]), gens)

    @torch.inference_mode()
    def execute(self, so) -> dict[str, int]:
        ids, pos, last, seqs = self.prepare_inputs(so)
        hidden = self.model(ids, pos, self.attn)
        if not seqs:
            return {}
        logits = self.model.compute_logits(hidden[last])     # only rows that sample
        toks = self.sampler(logits, *self._sampling_tensors(seqs)).tolist()
        return {s.seq_id: t for s, t in zip(seqs, toks)}

    def release(self, seq_id):
        self.generators.pop(seq_id, None)
```

Note `compute_logits(hidden[last])`: the LM head (a `[C × 151,936]` matmul for Qwen2.5) runs only on rows that sample, not on every prompt token — vLLM does the same "gather last-token hidden states" step.

## Acceptance tests (M1)

`tests/test_parity.py` builds a tiny random Qwen2 with HF (`Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)`), loads its `state_dict()` into your class, and compares 20 greedy tokens for 12 prompts against `hf.generate(..., do_sample=False)` in **fp32 on CPU**. Set `hf.generation_config.eos_token_id = None` so HF doesn't stop early.

- [ ] Tiny model: exact greedy parity for all prompts (runs in seconds, no download).
- [ ] Real Qwen2.5-0.5B-Instruct in fp32 on CPU/MPS: first 32 greedy tokens match HF on 5 prompts.
- [ ] `load_qwen2` works from a local dir and from a hub id; missing keys are only `lm_head.weight` when embeddings are tied.
- [ ] Commit `tests/golden/qwen2.5-0.5b.json` (prompt → HF greedy tokens) for regression testing.
      */}),
      resources: [
        { title: "Hugging Face — Qwen2 modeling code", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py", type: "repo", note: "the reference your parity test compares against" },
        { title: "PagedAttention paper (vLLM, SOSP '23)", url: "https://arxiv.org/abs/2309.06180", type: "paper", note: "block tables and slot mapping, the idea your backend implements" },
      ],
    },
    {
      id: "ms-scheduler",
      title: "Milestone M2: scheduler + prefix cache + the engine step",
      kind: "build",
      minutes: 300,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
M2 wires your M08 block manager and M09 scheduler to the runner. Three upgrades on the way: **prefix caching**, **abort**, and **stop tokens**.

## Prefix caching in the block manager

Idea (vLLM V1's design, also nano-vllm's): every **full, computed** block gets a *chain hash* $h_i = \text{hash}(h_{i-1}, \text{tokens of block } i)$, so equal hashes imply equal *prefixes*, not just equal blocks. A `hash → block_id` table finds them. Freed blocks **keep their hash** and wait in an LRU free list: if a new request shares the prefix, the block is revived with no compute; if memory is needed, the least-recently-used one is evicted and its hash forgotten.

::viz prefix-cache

```python
# minivllm/block_manager.py
import math
from collections import OrderedDict

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.prefix_caching = enable_prefix_caching
        self.ref = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        self.free_q: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.cached: dict[int, int] = {}                # hash -> block id
        self.tables: dict[str, list[int]] = {}          # seq_id -> block table
        self.hashes: dict[str, list[int]] = {}          # seq_id -> chain hashes of its full blocks
        self.hits = self.queries = 0                    # prefix-cache stats, in tokens

    def num_free_blocks(self) -> int:
        return len(self.free_q)

    def blocks_needed(self, seq_id, num_tokens) -> int:
        have = len(self.tables.get(seq_id, ()))
        return max(0, math.ceil(num_tokens / self.block_size) - have)

    def can_grow(self, seq_id, num_tokens, reserve=0) -> bool:
        return self.blocks_needed(seq_id, num_tokens) + reserve <= len(self.free_q)

    def grow(self, seq_id, num_tokens) -> None:
        n = self.blocks_needed(seq_id, num_tokens)
        assert n <= len(self.free_q), "call can_grow() first"
        table = self.tables.setdefault(seq_id, [])
        for _ in range(n):
            bid, _ = self.free_q.popitem(last=False)    # evict least-recently-used
            h = self.block_hash[bid]
            if h is not None and self.cached.get(h) == bid:
                del self.cached[h]
            self.block_hash[bid], self.ref[bid] = None, 1
            table.append(bid)

    def block_table(self, seq_id) -> list[int]:
        return self.tables.get(seq_id, [])

    def free(self, seq_id) -> None:
        for bid in reversed(self.tables.pop(seq_id, [])):
            self.ref[bid] -= 1
            if self.ref[bid] == 0:
                self.free_q[bid] = None                 # most-recently-used end
                if self.block_hash[bid] is None:        # nothing worth keeping: evict first
                    self.free_q.move_to_end(bid, last=False)
        self.hashes.pop(seq_id, None)

    def match_prefix(self, seq_id, token_ids) -> int:
        """On admission: attach cached full blocks; return #cached tokens.
        Never matches the block holding the last token -- we need its logits."""
        assert seq_id not in self.tables
        table, hashes, h = [], [], None
        if self.prefix_caching:
            bs = self.block_size
            for i in range((len(token_ids) - 1) // bs):
                h = hash((h, tuple(token_ids[i * bs:(i + 1) * bs])))
                bid = self.cached.get(h)
                if bid is None:
                    break
                if self.ref[bid] == 0:
                    del self.free_q[bid]                # revive from the free list
                self.ref[bid] += 1
                table.append(bid)
                hashes.append(h)
        self.tables[seq_id], self.hashes[seq_id] = table, hashes
        self.queries += len(token_ids)
        self.hits += len(table) * self.block_size
        return len(table) * self.block_size

    def commit(self, seq_id, token_ids, num_computed) -> None:
        """After a step: publish hashes of blocks that just became full and computed."""
        if not self.prefix_caching:
            return
        bs, table = self.block_size, self.tables[seq_id]
        hashes = self.hashes.setdefault(seq_id, [])
        while len(hashes) < num_computed // bs:
            i = len(hashes)
            h = hash((hashes[-1] if hashes else None, tuple(token_ids[i * bs:(i + 1) * bs])))
            hashes.append(h)
            bid = table[i]
            if self.block_hash[bid] is None and h not in self.cached:
                self.block_hash[bid] = h
                self.cached[h] = bid
```

Why is sharing safe without copy-on-write? A sequence only ever **writes to its own tail block**, and prefix matching only shares **full** blocks that end *before* the last prompt token. No shared block is ever written again.

> [!TIP]
> Python's `hash()` of a tuple of ints is deterministic across runs but only 64 bits; production engines use SHA-256 (vLLM) or xxhash (nano-vllm) so a collision can't silently serve someone else's KV. Also note that decode tokens are committed too — so multi-turn chats hit the cache on the *previous answer*, not just the system prompt.

## The scheduler (M09 + three changes)

```python
# minivllm/scheduler.py
from collections import deque
from .request import ScheduledChunk, SchedulerOutput, Sequence, SeqStatus

class Scheduler:
    def __init__(self, bm, max_num_batched_tokens=512, max_num_seqs=32,
                 long_prefill_threshold=256, eos_token_id=None, watermark_blocks=1):
        self.bm, self.budget, self.max_num_seqs = bm, max_num_batched_tokens, max_num_seqs
        self.chunk, self.eos, self.watermark = long_prefill_threshold, eos_token_id, watermark_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.seqs: dict[str, Sequence] = {}
        self.num_preemptions = 0

    def add_request(self, req):
        seq = Sequence(req)
        self.seqs[seq.seq_id] = seq
        self.waiting.append(seq)

    def abort(self, seq_id):                                # NEW: client went away
        seq = self.seqs.pop(seq_id, None)
        if seq is None:
            return None
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self.bm.free(seq_id)
        seq.status = SeqStatus.FINISHED_ABORTED
        return seq

    def has_work(self):
        return bool(self.waiting or self.running)

    def _preempt(self, seq, preempted):                     # recompute-style, as in M09
        self.running.remove(seq)
        self.bm.free(seq.seq_id)
        seq.num_computed = seq.num_cached = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        preempted.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def schedule(self) -> SchedulerOutput:
        budget, chunks, preempted = self.budget, [], []
        i = 0                                               # 1) running first
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_uncomputed, budget, self.chunk)
            while not self.bm.can_grow(seq.seq_id, seq.num_computed + n):
                self._preempt(self.running[-1], preempted)  # LIFO victim
                if seq.status is SeqStatus.WAITING:
                    break
            if seq.status is SeqStatus.WAITING:
                break
            self.bm.grow(seq.seq_id, seq.num_computed + n)
            chunks.append(ScheduledChunk(seq, seq.num_computed, n))
            budget -= n
            i += 1
        while (self.waiting and budget > 0 and not preempted  # 2) admit
               and len(self.running) < self.max_num_seqs):
            seq = self.waiting[0]
            if seq.num_computed == 0:                       # NEW: prefix-cache lookup
                seq.num_computed = seq.num_cached = self.bm.match_prefix(seq.seq_id, seq.token_ids)
            n = min(seq.num_uncomputed, budget, self.chunk)
            if not self.bm.can_grow(seq.seq_id, seq.num_computed + n, reserve=self.watermark):
                self.bm.free(seq.seq_id)                    # undo the match; retry next step
                seq.num_computed = seq.num_cached = 0
                break
            self.waiting.popleft()
            self.bm.grow(seq.seq_id, seq.num_computed + n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            chunks.append(ScheduledChunk(seq, seq.num_computed, n))
            budget -= n
        return SchedulerOutput(chunks, self.budget - budget, preempted)

    def update(self, out, sampled: dict[str, int]) -> list[Sequence]:
        touched = []
        for c in out.chunks:
            seq = c.seq
            if seq.status is not SeqStatus.RUNNING:         # aborted mid-step
                continue
            samples = c.samples                             # evaluate before appending
            seq.num_computed += c.n
            self.bm.commit(seq.seq_id, seq.token_ids, seq.num_computed)   # NEW
            if not samples:
                continue
            tok = sampled[seq.seq_id]
            seq.token_ids.append(tok)
            p = seq.params
            if (tok == self.eos and not p.ignore_eos) or tok in p.stop_token_ids:   # NEW
                seq.status = SeqStatus.FINISHED_STOPPED
            elif len(seq.output_ids) >= p.max_tokens:
                seq.status = SeqStatus.FINISHED_LENGTH
            if seq.status.finished:
                self.running.remove(seq)
                self.bm.free(seq.seq_id)
                del self.seqs[seq.seq_id]
            touched.append(seq)
        return touched
```

## The engine step and an offline API

```python
# minivllm/llm_engine.py
import itertools, time
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .request import EngineCoreOutput, Request
from .scheduler import Scheduler

class EngineCore:
    def __init__(self, model, cfg, *, num_blocks=1024, block_size=16, max_num_batched_tokens=512,
                 max_num_seqs=32, long_prefill_threshold=256, enable_prefix_caching=True,
                 eos_token_id=None, device="cpu", dtype=None):
        self.bm = BlockManager(num_blocks, block_size, enable_prefix_caching)
        self.scheduler = Scheduler(self.bm, max_num_batched_tokens, max_num_seqs,
                                   long_prefill_threshold, eos_token_id)
        self.runner = ModelRunner(model, cfg, self.bm, device, dtype)
        self.stats = {"steps": 0, "batched_tokens": 0, "step_time_s": 0.0}

    def add_request(self, req): self.scheduler.add_request(req)
    def has_work(self): return self.scheduler.has_work()
    def abort(self, request_id):
        if self.scheduler.abort(request_id) is not None:
            self.runner.release(request_id)

    def step(self) -> list[EngineCoreOutput]:
        t0 = time.perf_counter()
        so = self.scheduler.schedule()                      # 1 schedule
        if so.is_empty:
            return []
        sampled = self.runner.execute(so)                   # 2-4 prepare, forward, sample
        outs = []
        for seq in self.scheduler.update(so, sampled):      # 5 update
            fin = seq.status.value if seq.status.finished else None
            if fin:
                self.runner.release(seq.seq_id)
            outs.append(EngineCoreOutput(seq.seq_id, [seq.token_ids[-1]], fin,
                                         seq.num_prompt_tokens, seq.num_cached))
        self.stats["steps"] += 1
        self.stats["batched_tokens"] += so.num_batched_tokens
        self.stats["step_time_s"] += time.perf_counter() - t0
        return outs                                         # 6 emit

class LLM:
    """Offline, single-process: LLM(core).generate(prompt_ids_list, params)."""
    def __init__(self, core):
        self.core, self._ids = core, itertools.count()
    def generate(self, prompts, params):
        params = params if isinstance(params, list) else [params] * len(prompts)
        rids = [f"offline-{next(self._ids)}" for _ in prompts]
        for rid, p, sp in zip(rids, prompts, params):
            self.core.add_request(Request(rid, p, sp))
        out = {rid: [] for rid in rids}
        while self.core.has_work():
            for o in self.core.step():
                out[o.request_id] += o.new_token_ids
        return [out[r] for r in rids]
```

## Acceptance tests (M2)

Run the tiny-model parity test through `LLM.generate` in four configurations. A reference implementation of this module produced:

```text
budget=1000000 chunk=1000000 blocks=64 prefix=False parity=True preempt=2  hit=0/718
budget=16      chunk=5       blocks=64 prefix=True  parity=True preempt=2  hit=100/448
budget=16      chunk=5       blocks=30 prefix=True  parity=True preempt=16 hit=864/1900
   second pass on the same engine: parity=True, prefix hits keep climbing
```

- [ ] Exact greedy parity (fp32, CPU) with chunking, forced preemption (tiny pool) and prefix caching on — and again on a second pass that hits the cache.
- [ ] Block conservation: after every run, `bm.num_free_blocks() == bm.num_blocks`.
- [ ] Prefix-cache unit test: two prompts sharing 3 full blocks share the same physical block ids; after both finish, a third revives them from the free list; filling the pool evicts them.
- [ ] Offline throughput on the real model (MPS, fp16, 64+ prompts): **≥3× the M0 sequential baseline**. Save the JSON.
      */}),
      resources: [
        { title: "vLLM docs — Automatic Prefix Caching design", url: "https://docs.vllm.ai/en/latest/design/prefix_caching.html", type: "docs", note: "chain hashing, LRU eviction of free cached blocks" },
        { title: "nano-vllm — engine/block_manager.py", url: "https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py", type: "repo", note: "the same idea with xxhash in ~120 lines" },
      ],
    },
    {
      id: "ms-sampler-stream",
      title: "Milestone M3: batched sampler, incremental detokenizer, stop strings",
      kind: "build",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
Every request in the batch has its **own** temperature, top-k, top-p and seed, and the batch changes every step. The sampler must handle all of that in a few tensor ops — a Python loop over requests would cost more than the forward pass for a small model.

## The sampler

```python
# minivllm/sampler.py
import torch

class Sampler:
    def __call__(self, logits, temperature, top_k, top_p, generators):
        """logits [B, V] fp32; temperature/top_p float [B]; top_k long [B] (<=0: off);
        generators: per-row torch.Generator (seeded request) or None."""
        B, V = logits.shape
        greedy_tok = logits.argmax(-1)
        is_greedy = temperature <= 0
        logits = logits / torch.where(is_greedy, 1.0, temperature).unsqueeze(1)

        # sort once; apply top-k then top-p in sorted space
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        k = torch.where(top_k <= 0, V, top_k).clamp(1, V)
        ranks = torch.arange(V, device=logits.device).expand(B, V)
        sorted_logits = sorted_logits.masked_fill(ranks >= k.unsqueeze(1), float("-inf"))
        probs = sorted_logits.softmax(-1)
        cum_before = probs.cumsum(-1) - probs          # mass strictly above this token
        sorted_logits = sorted_logits.masked_fill(cum_before >= top_p.unsqueeze(1), float("-inf"))
        probs = sorted_logits.softmax(-1)              # renormalised over the kept set

        # exponential race: argmax(p_i / E_i) with E_i ~ Exp(1) is a sample from p
        noise = torch.empty_like(probs).exponential_()
        for i, g in enumerate(generators):
            if g is not None:                          # seeded rows: reproducible noise
                noise[i] = torch.empty(V).exponential_(generator=g).to(noise.device)
        choice = (probs / noise.clamp_min(1e-10)).argmax(-1)   # clamp: 0/0 would be NaN
        sampled = sorted_idx.gather(1, choice.unsqueeze(1)).squeeze(1)
        return torch.where(is_greedy, greedy_tok, sampled)
```

> [!MATH]
> **Top-p** keeps the smallest set of tokens whose probabilities sum to at least $p$. Sorted descending, token $i$ is kept iff the mass *strictly before it* is below $p$: $\sum_{j<i} p_j < p$ — so the top token always survives. **The exponential race**: if $E_i \sim \text{Exp}(1)$ independently, then $\arg\max_i p_i / E_i$ is distributed exactly as $p$ (equivalently, it's the Gumbel-max trick, since $-\log E_i$ is Gumbel). It replaces `torch.multinomial` with an argmax, which batches cleanly and lets you inject per-request seeded noise. vLLM and nano-vllm both use it.

**Per-request seeds.** A seeded request keeps its own `torch.Generator` for its lifetime (created on first use in `ModelRunner`, released on finish). Its noise depends only on its seed and step count — **not** on who else is in the batch. That's the property to test.

## The incremental detokenizer

Why not `tok.decode([new_token])`? Because tokens are not characters: BPE tokens can be **half a UTF-8 character** (emoji, CJK), and SentencePiece/byte-level tokenizers encode the leading space in a way that decodes differently in isolation. The fix used by TGI and vLLM: keep two offsets and decode a short **window**.

```python
# minivllm/detokenizer.py
class IncrementalDetokenizer:
    def __init__(self, tokenizer, prompt_ids, stop=(), skip_special_tokens=True):
        self.tok, self.skip = tokenizer, skip_special_tokens
        self.ids = list(prompt_ids[-5:])     # a little context so spacing decodes correctly
        self.prefix, self.read = 0, len(self.ids)
        self.stop = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.text, self.emitted = "", 0

    def _decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=self.skip)

    def add(self, new_ids) -> tuple[str, bool]:
        """Returns (text safe to send now, stop_string_hit)."""
        for t in new_ids:
            self.ids.append(t)
            prefix_text = self._decode(self.ids[self.prefix:self.read])
            new_text = self._decode(self.ids[self.prefix:])
            if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
                self.text += new_text[len(prefix_text):]
                self.prefix, self.read = self.read, len(self.ids)
        for s in self.stop:
            i = self.text.find(s, self.emitted)
            if i != -1:                          # truncate before the stop string
                self.text = self.text[:i]
                return self.flush(), True
        safe = max(self.emitted, len(self.text) - self.hold)   # hold back a possible stop prefix
        out, self.emitted = self.text[self.emitted:safe], safe
        return out, False

    def flush(self) -> str:
        out, self.emitted = self.text[self.emitted:], len(self.text)
        return out
```

`\ufffd` (the replacement character "�") at the end means "incomplete UTF-8 sequence" — wait for the next token. **Stop strings** can span tokens (`"EN"` + `"D"`), so we hold back the last `len(longest_stop) − 1` characters until we know they aren't the start of a stop string; when one matches, we truncate before it and tell the caller to **abort** the request in the engine (it's still generating!). This work happens in the **API process**: the engine only knows token ids, which keeps the GPU loop free of string work — the same split vLLM V1 makes.

## Acceptance tests (M3)

The reference implementation passes these with the Qwen2.5 tokenizer:

```text
top_p=0.8 freq [0.625, 0.375, 0.0, 0.0]        expected [0.625, 0.375, 0, 0]
top_k=3   freq [0.525, 0.317, 0.158, 0.0]      expected [0.526, 0.316, 0.158, 0]
greedy    [20000, 0, 0, 0]
seeded reproducible across batches: True
detok roundtrip True 'Hello, wörld! 你好 🙂🎉 naïve café — done. STOP here'
stop True 'Hello, wörld! 你好 🙂🎉 naïve café — done. '
```

- [ ] **Distribution tests**: logits $\log[0.5, 0.3, 0.15, 0.05]$ repeated 20,000 rows; top-p 0.8 gives ≈ [0.625, 0.375, 0, 0]; top-k 3 gives ≈ [0.526, 0.316, 0.158, 0]; temperature 0 is argmax.
- [ ] **Seed test**: the same seeded request produces identical tokens alone and inside a batch with other unseeded/greedy requests (fp32).
- [ ] **Detokenizer**: feeding the ids of an emoji/CJK/accented string one at a time reproduces `tok.decode(ids)` exactly; no chunk ever contains "�".
- [ ] **Stop strings**: the stop string never appears in the output, even when it spans tokens.
      */}),
      resources: [
        { title: "Hugging Face — Generation strategies (top-k, top-p, temperature)", url: "https://huggingface.co/docs/transformers/generation_strategies", type: "docs", note: "reference semantics for the sampling knobs" },
        { title: "vLLM — v1/sample/sampler.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py", type: "repo", note: "production sampler: penalties, logprobs, the exponential trick" },
      ],
    },
    {
      id: "ms-api",
      title: "Milestone M4: engine-core process, OpenAI-compatible API with SSE, metrics",
      kind: "build",
      minutes: 300,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Now the engine becomes a **server**. Three pieces: the engine-core process loop, an `AsyncEngine` client in the API process, and the FastAPI app.

## Engine-core process + AsyncEngine

```python
# minivllm/async_engine.py
import asyncio, multiprocessing as mp, queue, threading, time
from .detokenizer import IncrementalDetokenizer
from .request import Request

def run_engine_core(make_core, in_q, out_q):
    core = make_core()                                   # load weights *inside* the child
    out_q.put(("ready", [], {}))
    last_stats = 0.0
    while True:
        msgs = [] if core.has_work() else [in_q.get()]   # idle: block instead of spinning
        while True:
            try:
                msgs.append(in_q.get_nowait())
            except queue.Empty:
                break
        for kind, payload in msgs:
            if kind == "add":
                core.add_request(payload)
            elif kind == "abort":
                core.abort(payload)
            elif kind == "shutdown":
                return
        outs = core.step()
        now = time.monotonic()
        if outs or now - last_stats > 1.0:
            out_q.put(("outputs", outs, snapshot(core)))
            last_stats = now

def snapshot(core) -> dict:
    s, bm = core.scheduler, core.bm
    return {"num_running": len(s.running), "num_waiting": len(s.waiting),
            "kv_usage": 1 - bm.num_free_blocks() / bm.num_blocks,
            "prefix_hit_rate": bm.hits / max(bm.queries, 1),
            "num_preemptions": s.num_preemptions, **core.stats}

class AsyncEngine:
    """Lives in the API process. One asyncio.Queue per in-flight request."""
    def __init__(self, make_core, tokenizer):
        ctx = mp.get_context("spawn")                    # never fork after touching MPS/CUDA
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=run_engine_core, args=(make_core, self.in_q, self.out_q),
                                daemon=True)
        self.tok, self.streams, self.stats = tokenizer, {}, {}

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.proc.start()
        await self.loop.run_in_executor(None, self.out_q.get)       # wait for "ready"
        threading.Thread(target=self._pump, daemon=True).start()

    def shutdown(self):
        self.in_q.put(("shutdown", None))
        self.proc.join(timeout=5)

    def _pump(self):                                     # thread: blocking queue -> event loop
        while True:
            _, outs, stats = self.out_q.get()
            self.loop.call_soon_threadsafe(self._dispatch, outs, stats)

    def _dispatch(self, outs, stats):
        self.stats = stats
        for o in outs:
            if (q := self.streams.get(o.request_id)) is not None:
                q.put_nowait(o)

    def abort(self, request_id):
        self.in_q.put(("abort", request_id))

    async def generate(self, request_id, prompt_ids, params):
        """Async iterator of (text_delta, token_ids, finish_reason, core_output)."""
        q = self.streams[request_id] = asyncio.Queue()
        detok = IncrementalDetokenizer(self.tok, prompt_ids, params.stop)
        self.in_q.put(("add", Request(request_id, list(prompt_ids), params)))
        finished = False
        try:
            while not finished:
                o = await q.get()
                text, stop_hit = detok.add(o.new_token_ids)
                reason = "stop" if stop_hit else o.finish_reason
                if stop_hit:
                    self.abort(request_id)               # engine is still generating: stop it
                elif reason:
                    text += detok.flush()
                finished = reason is not None
                yield text, o.new_token_ids, reason, o
        finally:
            self.streams.pop(request_id, None)
            if not finished:                             # client disconnected mid-stream
                self.abort(request_id)
```

The `finally:` is the whole abort story: when an HTTP client disconnects, Starlette cancels the streaming task, `await q.get()` raises `CancelledError`, and the engine frees that request's blocks on its next loop iteration. Without it, a closed browser tab keeps burning GPU until `max_tokens`.

## The HTTP layer

```python
# minivllm/server/api.py
import json, time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from ..request import SamplingParams

TTFT = Histogram("minivllm_time_to_first_token_seconds", "TTFT",
                 buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30))
ITL = Histogram("minivllm_inter_token_latency_seconds", "ITL",
                buckets=(.005, .01, .02, .04, .08, .16, .32, .64, 1.28))
E2E = Histogram("minivllm_e2e_request_latency_seconds", "E2E",
                buckets=(.1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120))
PROMPT_TOKENS = Counter("minivllm_prompt_tokens", "prompt tokens")
GEN_TOKENS = Counter("minivllm_generation_tokens", "generated tokens")
FINISHED = Counter("minivllm_requests_finished", "finished requests", ["reason"])
ENGINE = {k: Gauge(f"minivllm_{k}", k) for k in
          ("num_running", "num_waiting", "kv_usage", "prefix_hit_rate", "num_preemptions")}

class CompletionBody(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1                       # extension; vLLM accepts it too
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    ignore_eos: bool = False

class ChatBody(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None

def to_params(b, max_tokens) -> SamplingParams:
    stop = [b.stop] if isinstance(b.stop, str) else (b.stop or [])
    return SamplingParams(max_tokens=max_tokens, temperature=b.temperature, top_p=b.top_p,
                          top_k=b.top_k, seed=b.seed, stop=stop,
                          ignore_eos=getattr(b, "ignore_eos", False))

def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"

def build_app(engine, tokenizer, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await engine.start()
        yield
        engine.shutdown()

    app = FastAPI(lifespan=lifespan)

    async def run(rid, prompt_ids, params):
        """engine.generate + metrics. Yields (delta, reason, n_prompt, n_generated)."""
        t0 = last = time.perf_counter()
        n_gen = 0
        PROMPT_TOKENS.inc(len(prompt_ids))
        async for text, toks, reason, _ in engine.generate(rid, prompt_ids, params):
            now = time.perf_counter()
            (TTFT if n_gen == 0 else ITL).observe(now - (t0 if n_gen == 0 else last))
            last, n_gen = now, n_gen + len(toks)
            GEN_TOKENS.inc(len(toks))
            if reason:
                E2E.observe(now - t0)
                FINISHED.labels(reason).inc()
            yield text, reason, len(prompt_ids), n_gen

    def usage(n_p, n_g):
        return {"prompt_tokens": n_p, "completion_tokens": n_g, "total_tokens": n_p + n_g}

    @app.post("/v1/completions")
    async def completions(body: CompletionBody):
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        gen = run(rid, tokenizer.encode(body.prompt), to_params(body, body.max_tokens))
        head = {"id": rid, "object": "text_completion", "created": created, "model": model_name}
        if not body.stream:
            parts, reason, u = [], None, None
            async for text, reason, n_p, n_g in gen:
                parts.append(text)
                u = usage(n_p, n_g)
            return {**head, "choices": [{"index": 0, "text": "".join(parts), "logprobs": None,
                                         "finish_reason": reason}], "usage": u}
        async def stream():
            u = None
            async for text, reason, n_p, n_g in gen:
                u = usage(n_p, n_g)
                if text or reason:
                    yield sse({**head, "choices": [{"index": 0, "text": text, "logprobs": None,
                                                    "finish_reason": reason}]})
            if (body.stream_options or {}).get("include_usage"):
                yield sse({**head, "choices": [], "usage": u})
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(body: ChatBody):
        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        text = tokenizer.apply_chat_template(body.messages, tokenize=False,
                                             add_generation_prompt=True)
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)   # template has them
        gen = run(rid, prompt_ids,
                  to_params(body, body.max_completion_tokens or body.max_tokens or 512))
        head = {"id": rid, "object": "chat.completion.chunk", "created": created,
                "model": model_name}
        if not body.stream:
            parts, reason, u = [], None, None
            async for t, reason, n_p, n_g in gen:
                parts.append(t)
                u = usage(n_p, n_g)
            return {**head, "object": "chat.completion", "usage": u, "choices": [{
                "index": 0, "finish_reason": reason,
                "message": {"role": "assistant", "content": "".join(parts)}}]}
        async def stream():
            yield sse({**head, "choices": [{"index": 0, "finish_reason": None,
                                            "delta": {"role": "assistant", "content": ""}}]})
            u = None
            async for t, reason, n_p, n_g in gen:
                u = usage(n_p, n_g)
                if t:
                    yield sse({**head, "choices": [{"index": 0, "delta": {"content": t},
                                                    "finish_reason": None}]})
                if reason:
                    yield sse({**head, "choices": [{"index": 0, "delta": {},
                                                    "finish_reason": reason}]})
            if (body.stream_options or {}).get("include_usage"):
                yield sse({**head, "choices": [], "usage": u})
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model",
                                            "owned_by": "mini-vllm"}]}

    @app.get("/health")
    async def health():
        ok = engine.proc.is_alive()
        return JSONResponse({"ok": ok}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics():
        for k, g in ENGINE.items():
            g.set(engine.stats.get(k, 0))
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
```

Details that matter for compatibility: chat prompts go through the model's **chat template** (`apply_chat_template(..., tokenize=False, add_generation_prompt=True)`) and are then encoded **without** adding special tokens again (a doubled BOS silently degrades Llama outputs); chat streams open with a `delta.role` chunk and end with an empty delta carrying `finish_reason`; `stream_options.include_usage` adds a final chunk with `choices: []` and `usage` — benchmark tools rely on it to count tokens.

## The CLI

```python
# minivllm/serve.py
import argparse, functools, torch, uvicorn
from transformers import AutoTokenizer
from .async_engine import AsyncEngine
from .llm_engine import EngineCore
from .qwen2 import load_qwen2
from .server.api import build_app

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

def make_core(model_id, device, dtype, eos_token_id, **kw):   # runs in the engine process
    model, cfg = load_qwen2(model_id, device, DTYPES[dtype])
    return EngineCore(model, cfg, device=device, dtype=DTYPES[dtype],
                      eos_token_id=eos_token_id, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=DTYPES)
    ap.add_argument("--num-blocks", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--long-prefill-threshold", type=int, default=256)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    make = functools.partial(
        make_core, a.model, a.device, a.dtype, tok.eos_token_id,
        num_blocks=a.num_blocks, block_size=a.block_size, max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens,
        long_prefill_threshold=a.long_prefill_threshold,
        enable_prefix_caching=not a.no_prefix_caching)
    app = build_app(AsyncEngine(make, tok), tok, a.model.split("/")[-1])
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")

if __name__ == "__main__":
    main()
```

`make_core` must be a **top-level** function (spawn pickles it by name). Size `--num-blocks` from M08's KV calculator: Qwen2.5-0.5B has 24 layers × 2 KV heads × 64 dims, so one 16-token block in fp16 is 2 (K,V) × 24 × 2 × 64 × 16 × 2 bytes ≈ 196 KB, and 2,048 blocks ≈ 400 MB. (Note: the Instruct tokenizer's `eos_token` is `<|im_end|>`, which is what chat turns end with.)

## Test the server without a model

A **fake engine core** (same `add_request/abort/has_work/step` methods, emits scripted token ids with a 10 ms sleep) lets you test all of HTTP in seconds. Pass `functools.partial(make_fake_core, script_ids)` instead of the real `make_core`, then drive it with the official client:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for ch in c.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}],
                                    max_tokens=10, stream=True,
                                    stream_options={"include_usage": True}):
    print(ch.choices[0].delta.content if ch.choices else ch.usage)
```

## Acceptance tests (M4)

- [ ] With the fake core: the `openai` client works for completions and chat, streaming and not; `usage` is correct; a stop string truncates output and reports `finish_reason="stop"`.
- [ ] Disconnect test: open a stream with `max_tokens=100000`, read 5 events, close — the engine logs an abort and `minivllm_num_running` returns to 0.
- [ ] With the real model on MPS: 64 concurrent streaming clients (M09's `concurrency.py`) all complete; `/health` stays responsive during load (it would not if the forward ran on the event loop).
- [ ] `/metrics` shows TTFT/ITL histograms, token counters, KV usage and prefix hit rate; a Prometheus scrape works.
      */}),
      resources: [
        { title: "OpenAI API reference — Chat Completions (streaming)", url: "https://platform.openai.com/docs/api-reference/chat/create", type: "docs", note: "chunk object shape, stream_options" },
        { title: "FastAPI — StreamingResponse", url: "https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse", type: "docs", note: "async generators as HTTP bodies" },
        { title: "prometheus_client (Python)", url: "https://github.com/prometheus/client_python", type: "repo", note: "Histogram/Counter/Gauge + generate_latest()" },
      ],
    },
    {
      id: "benchmark",
      title: "Milestone M5 (lab): benchmark against HF generate and vLLM, and find the gap",
      kind: "lab",
      minutes: 300,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
A benchmark is only portfolio-grade if someone else can rerun it and get the same shape. Rules:

1. **Same work**: same model, dtype, prompts, and *fixed* output lengths (`ignore_eos` + `max_tokens`) — otherwise an engine "wins" by stopping early.
2. **Warm up** (first MPS/CUDA calls compile kernels), then measure; run each point 3× and report the median.
3. **Measure from the scheduled send time**, not from when the client got around to sending (PageServe's load tester does this; otherwise you under-report latency under overload — "coordinated omission").
4. Record hardware, OS, torch/vLLM versions, commit hash. Save **raw JSON**; plots are generated from JSON, never by hand.

## Offline: throughput vs HF (Mac and Colab)

```python
# bench/bench_offline.py
import json, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minivllm.llm_engine import EngineCore, LLM
from minivllm.qwen2 import load_qwen2
from minivllm.request import SamplingParams

MID, N, IN, OUT = "Qwen/Qwen2.5-0.5B-Instruct", 128, 256, 128
dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
dt = torch.float16
sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)
rng = random.Random(0)
prompts = [[rng.randrange(1000, 20000) for _ in range(IN)] for _ in range(N)]

def timed(fn):
    sync(); t0 = time.perf_counter(); fn(); sync()
    return N * OUT / (time.perf_counter() - t0)

res = {"device": dev, "torch": torch.__version__, "N": N, "in": IN, "out": OUT}
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=dt).to(dev).eval()
gen = dict(max_new_tokens=OUT, min_new_tokens=OUT, do_sample=False, pad_token_id=0)
hf.generate(torch.tensor([prompts[0]], device=dev), max_new_tokens=4, pad_token_id=0)
res["hf_sequential"] = timed(lambda: [hf.generate(torch.tensor([p], device=dev), **gen)
                                      for p in prompts[:16]]) * 16 / N   # 16 is enough
res["hf_static_16"] = timed(lambda: [hf.generate(torch.tensor(prompts[i:i + 16], device=dev), **gen)
                                     for i in range(0, N, 16)])
del hf

model, cfg = load_qwen2(MID, dev, dt)
for seqs in (16, 64):
    core = EngineCore(model, cfg, num_blocks=4096, max_num_seqs=seqs,
                      max_num_batched_tokens=1024, device=dev, dtype=dt)
    llm, sp = LLM(core), SamplingParams(max_tokens=OUT, temperature=0, ignore_eos=True)
    llm.generate(prompts[:2], SamplingParams(max_tokens=4, temperature=0))     # warmup
    res[f"minivllm_{seqs}"] = timed(lambda: llm.generate(prompts, sp))
print(json.dumps(res, indent=1))
json.dump(res, open(f"bench/results/offline_{dev}.json", "w"), indent=1)
```

(The sequential line scales 16 requests' time up to N to save your afternoon — say so in the README.) All prompts are the same length here on purpose: it makes the HF static batch a *fair* opponent (no padding waste). Rerun with mixed lengths (64–1024) and watch the static batch fall behind — that's the M09 lesson in one table.

## Online: `vllm bench serve` against your server and vLLM (Colab)

`vllm bench serve` speaks the OpenAI API, so the **same client** can measure your engine and vLLM — the only fair way to compare. On a Colab **L4** (bf16-capable; on a T4 use `--dtype float16`):

```bash
pip -q install vllm
git clone https://github.com/<you>/mini-vllm && pip -q install -e mini-vllm

# 1) your engine
python -m minivllm.serve --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype bfloat16 \
       --num-blocks 8192 --max-num-seqs 128 --port 8000 &
for rate in 1 2 4 8 16 inf; do
  vllm bench serve --backend openai --base-url http://localhost:8000 --endpoint /v1/completions \
    --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random \
    --random-input-len 256 --random-output-len 128 --num-prompts 200 --ignore-eos \
    --request-rate $rate --save-result --result-dir results --result-filename mini_$rate.json
done
kill %1

# 2) vLLM itself, same client, same flags
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 &
# …repeat the loop with --result-filename vllm_$rate.json
```

On the Mac (no vLLM), drive your server with M09's `concurrency.py` / Poisson client and report TTFT/ITL percentiles the same way.

```python
# bench/plot.py — one figure per metric, one line per engine
import glob, json, re
import matplotlib.pyplot as plt

def load(prefix):
    pts = []
    for f in glob.glob(f"results/{prefix}_*.json"):
        r = json.load(open(f))
        rate = re.search(rf"{prefix}_(.+)\.json", f).group(1)
        pts.append((float(rate) if rate != "inf" else 32.0, r))
    return sorted(pts, key=lambda p: p[0])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for eng in ("mini", "vllm"):
    pts = load(eng)
    x = [p for p, _ in pts]
    for ax, key, label in zip(axes, ("output_throughput", "p99_ttft_ms", "median_itl_ms"),
                              ("output tok/s", "TTFT p99 (ms)", "ITL p50 (ms)")):
        ax.plot(x, [r[key] for _, r in pts], marker="o", label=eng)
        ax.set_xlabel("request rate (req/s; 32 = unbounded)"); ax.set_title(label)
axes[1].set_yscale("log"); axes[0].legend(); plt.tight_layout(); plt.savefig("bench/online.png", dpi=150)
```

## Find the gap: a step-time breakdown

Instrument `EngineCore.step()` with `time.perf_counter()` around schedule / prepare_inputs / forward / sample / update (call `torch.cuda.synchronize()` or `torch.mps.synchronize()` before each read, for measurement only) and plot a stacked bar per batch size. Typical findings on a 0.5B model — each one is a talking point:

- **Sampling costs more than you think**: sorting `[B, 151,936]` logits every step for top-p. Fast path: skip the sort when every row is greedy or has `top_k ≤ 0` and `top_p = 1` (vLLM has the same fast path).
- **Python `prepare_inputs`** grows linearly with batch size and context length (building slot lists in Python). vLLM keeps persistent numpy/pinned buffers and updates them incrementally.
- **The KV gather** in `TorchPagedAttention` copies every sequence's whole context every layer; a real paged kernel reads it in place.
- **Launch overhead** on CUDA: hundreds of small kernels per step for a 24-layer model — the CUDA-graphs lesson fixes it.

> [!REAL]
> This is what inference engineers actually do all day: profile the step, attribute time, fix the biggest bar, re-measure. Baseten's book frames it as working from the roofline — know what the step *should* cost (M07), then explain the difference (Inference Engineering (Baseten) ch. 4).

## Exercises

- [ ] `bench/results/offline_mps.json` and `offline_cuda.json` with HF sequential, HF static-16 and mini-vllm at 16/64 seqs.
- [ ] Online sweep on Colab for mini-vllm and vLLM with the same `vllm bench serve` flags; `online.png`.
- [ ] Step-time breakdown chart; implement the greedy/no-top-p sampling fast path and show the bar shrink.
- [ ] A 5-sentence "where the gap to vLLM lives" paragraph backed by your breakdown.
      */}),
      resources: [
        { title: "vLLM — benchmark CLI (vllm bench serve)", url: "https://docs.vllm.ai/en/latest/cli/bench/serve.html", type: "docs", note: "all flags, result JSON fields" },
        { title: "PageServe — benchmarks and load_test", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "a load tester that measures from scheduled send time; T4 results to compare against" },
      ],
    },
    {
      id: "cuda-speedups",
      title: "Optional (CUDA): CUDA graphs, FlashAttention / FlashInfer kernels, torch.compile",
      kind: "deep",
      minutes: 360,
      runsOn: ["colab"],
      optional: true,
      md: MD(function () {/*
Your Mac engine is done. On a CUDA GPU three upgrades close most of the gap to vLLM for a small model. **Do them in this order, one at a time, with parity and a benchmark after each** — and profile first (`torch.profiler` with `ProfilerActivity.CUDA`, or Nsight Systems from M13) so you know which bar you're shrinking.

## 1. CUDA graphs for decode steps

For a 0.5B model at small batch, a decode step is ~400 tiny kernels (24 layers × ~15 ops + sampling). At ~5–10 µs of CPU launch cost each, the CPU needs 2–4 ms to *issue* work the GPU finishes in ~1–2 ms: the GPU idles. A **CUDA graph** records the whole sequence of launches once and replays it with a single call.

The rules: every tensor the graph touches must live at a **fixed address with a fixed shape**, and nothing inside may sync with the CPU (`.item()`, `.tolist()`, Python branching on tensor values, fresh allocations). So:

```python
# sketch — nano-vllm's ModelRunner.capture_cudagraph is the complete reference
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]           # pad each decode batch up to a bucket
static = dict(ids=torch.zeros(128, dtype=torch.long, device="cuda"),
              pos=torch.zeros(128, dtype=torch.long, device="cuda"),
              slots=torch.zeros(128, dtype=torch.long, device="cuda"),
              ctx_lens=torch.zeros(128, dtype=torch.int32, device="cuda"),
              block_tables=torch.zeros(128, max_blocks_per_seq, dtype=torch.int32, device="cuda"))
graphs, outs, pool = {}, {}, None
for bs in reversed(BUCKETS):                       # largest first, then share its memory pool
    attn = GraphSafeAttention(static, bs)          # reads only the static tensors
    model(static["ids"][:bs], static["pos"][:bs], attn)                  # warmup
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool):
        outs[bs] = model(static["ids"][:bs], static["pos"][:bs], attn)
    pool = pool or g.pool()
    graphs[bs] = g

# at run time, for a pure-decode step with n sequences:
bs = next(b for b in BUCKETS if b >= n)
static["ids"][:n].copy_(ids); static["pos"][:n].copy_(pos)      # ...and slots, ctx_lens, tables
static["slots"][n:bs] = DUMMY_SLOT     # padded rows write into a reserved scratch block
                                       # (nano-vllm's Triton store kernel skips slot -1 instead)
graphs[bs].replay()
hidden = outs[bs][:n]
```

Mixed prefill/decode steps run eagerly (vLLM goes further with *piecewise* graphs that break around attention). Our `TorchPagedAttention` is **not** graph-safe — its gather has a data-dependent `Lmax` and builds masks in Python — so graphs force you to adopt a real paged kernel, which is upgrade #2 anyway.

## 2. Real paged-attention kernels

**FlashAttention 2** (`pip install flash-attn`; **Ampere+ only** — L4/A100/H100, not T4) exposes exactly what an engine needs, as nano-vllm shows:

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
# KV pool layout per layer: [num_blocks, block_size, kv_heads, head_dim];
# paged KV in flash-attn 2 needs block_size divisible by 256 (nano-vllm uses 256)

# prefill / chunked prefill / prefix hits: packed queries, keys read through the block table
o = flash_attn_varlen_func(q, k_cache, v_cache,
                           cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                           max_seqlen_q=max_q, max_seqlen_k=max_k,
                           causal=True, block_table=block_tables)
# decode: one query per sequence
o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=ctx_lens, block_table=block_tables, causal=True)
```

(When there's no prefix hit, nano-vllm passes the freshly computed `k`, `v` instead of the cache for prefill.) `cu_seqlens` are the cumulative-length offsets from M09's varlen lesson; `causal=True` with `seqlen_q < seqlen_k` aligns the causal mask to the **bottom-right**, which is exactly "a chunk at offset `start`". Writing new K/V into the pool is one scatter (`k_cache.view(-1, kvh, d)[slot_mapping] = k`) or nano-vllm's small Triton `store_kvcache` kernel.

**FlashInfer** (`pip install flashinfer-python`; check its wheel matrix for your CUDA/torch; it supports older GPUs than flash-attn and any page size) splits each call into `plan()` (CPU-side scheduling metadata, once per step, *outside* the CUDA graph) and `run()` (the kernel, capturable):

```python
import flashinfer
ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
decode.plan(kv_indptr, kv_indices, kv_last_page_len,           # CSR form of the block tables
            num_qo_heads, num_kv_heads, head_dim, page_size, q_data_type=torch.bfloat16)
o = decode.run(q, (k_cache, v_cache))                          # per layer
```

`BatchPrefillWithPagedKVCacheWrapper` is the prefill/chunk counterpart (adds `qo_indptr`). FlashInfer's API has changed between releases (`begin_forward/forward` → `plan/run`); pin the version in `pyproject.toml`.

## 3. `torch.compile`

With attention hidden behind a kernel call, the rest of the layer (RMSNorm, residual adds, SiLU·mul, RoPE) is a chain of memory-bound elementwise ops that Inductor fuses into a few Triton kernels (M12/M13). Compile the **layer body**, not the attention call:

```python
@torch.compiler.disable
def attention_call(attn, layer, q, k, v):      # keep the paged kernel out of the graph
    return attn.forward(layer, q, k, v)
# then in Attention.forward use attention_call(...), and:
model = torch.compile(model, dynamic=True)
```

vLLM does the same thing at larger scale: it compiles the model with Inductor, splits the graph at attention, and captures *piecewise* CUDA graphs for the pieces.

## Acceptance tests (each upgrade)

- [ ] Correctness: in bf16 you can't demand exact parity, so compare against your eager torch backend — max abs logit difference < 1e-2 on 16 prompts, and ≥ 95% of the first 64 greedy tokens identical; the fp32 tiny-model parity suite still passes with the eager backend.
- [ ] CUDA graphs: decode ITL at batch 1–8 drops substantially (often ~2×) on an L4; record before/after.
- [ ] Kernels: throughput at batch 64 and context 2k improves; the step-time breakdown's attention bar shrinks.
- [ ] `torch.compile`: measure the *warm* gain and the cold-start compile time; decide whether it's worth it and write down why.
      */}),
      resources: [
        { title: "FlashAttention (Dao-AILab)", url: "https://github.com/Dao-AILab/flash-attention", type: "repo", note: "flash_attn_varlen_func and flash_attn_with_kvcache with block_table" },
        { title: "FlashInfer", url: "https://github.com/flashinfer-ai/flashinfer", type: "repo", note: "paged prefill/decode wrappers with plan()/run(), CUDA-graph friendly" },
        { title: "PyTorch — CUDA Graphs", url: "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs", type: "docs", note: "capture rules, memory pools, warmup" },
        { title: "nano-vllm — engine/model_runner.py", url: "https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py", type: "repo", note: "capture_cudagraph and prepare_prefill/prepare_decode in ~250 lines" },
      ],
    },
    {
      id: "compare-and-read",
      title: "Compare with nano-vllm and PageServe, then read vLLM with a map",
      kind: "read",
      minutes: 240,
      runsOn: ["any"],
      md: MD(function () {/*
You've made ~30 design decisions. Now see how others made them — this is the section of your design doc interviewers read most closely.

## Three small engines side by side

| decision | your mini-vllm | nano-vllm | PageServe |
|---|---|---|---|
| size | ~1,500 lines | ~1,200 lines | larger, heavily documented |
| hardware | MPS + CUDA (+ CPU) | CUDA, Ampere+ (flash-attn) | CUDA + MPS + CPU |
| API | OpenAI completions + chat, SSE, `/metrics` | offline `LLM.generate` only | `/generate`, OpenAI `/v1/completions` SSE, `/metrics`, `/health` |
| scheduler | unified token budget, mixed prefill+decode, chunked prefill (M09) | **prefill-first**: a step is all-prefill or all-decode; only chunks when a prompt alone exceeds the budget | FCFS admission, chunked prefill, prefill/decode separation |
| preemption | recompute, LIFO victim | recompute | LIFO, plus **CPU swap** of KV blocks |
| prefix cache | chain hash + ref counts + LRU free list | xxhash chain hash, ref counts | yes (phase 12) |
| attention | torch SDPA gather (portable) | flash-attn varlen + `with_kvcache`, block size 256 | paged attention wrapper, torch-based |
| CUDA graphs | optional lesson | yes, per batch-size bucket | — |
| parallelism | none | tensor parallel via spawned processes + shared memory | none |
| process model | API process + engine-core process | single process (+ TP workers) | server + engine |
| testing | fp32 parity + fake-core API tests | benchmarks | token-exact parity vs HF in float64 on tiny models |

Two lessons hide in that table. **nano-vllm's prefill-first scheduler** is simpler and gives excellent *offline* throughput, but under online load a long prefill stalls every running decode — exactly the ITL spikes M09 measured; that's why vLLM V1 unified the budget. **PageServe's CPU swap** is the alternative to recompute you studied in M09; read its `cpu_swap_manager` and decide when it would have won on your hardware.

## Reading vLLM with a map

Aleksa Gordić's post analyzes vLLM at commit `42172ad`; browse that exact tree so line references match (later versions move files around — search by class name if a path 404s):

| your file | vLLM counterpart (at 42172ad) | read for |
|---|---|---|
| `server/api.py` | `vllm/entrypoints/openai/api_server.py`, `serving_completion.py`, `serving_chat.py` | request validation, chat templates, SSE chunking, usage |
| `async_engine.py` | `vllm/v1/engine/async_llm.py`, `core_client.py` | ZMQ client, per-request output queues |
| `detokenizer.py` | `vllm/v1/engine/output_processor.py`, `detokenizer.py` | incremental detokenization and stop strings in the front-end |
| engine-core loop | `vllm/v1/engine/core.py` (`EngineCore`, `EngineCoreProc`) | busy loop, input/output threads, `step()` |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py`, `sched/output.py` | `schedule()`, `num_scheduled_tokens`, preemption, `update_from_output()` |
| `block_manager.py` | `vllm/v1/core/kv_cache_manager.py`, `block_pool.py` | `free_block_queue`, block hashing, prefix hits |
| `model_runner.py` | `vllm/v1/worker/gpu_model_runner.py`, `gpu_input_batch.py`, `gpu_worker.py` | persistent input batch, `_prepare_inputs`, CUDA graph capture |
| `attention.py` | `vllm/v1/attention/backends/flash_attn.py` | attention metadata, `slot_mapping`, cascade attention |
| `sampler.py` | `vllm/v1/sample/sampler.py` and `sample/ops/` | penalties, top-k/top-p, logprobs |
| `qwen2.py` | `vllm/model_executor/models/qwen2.py` | fused QKV, `RMSNorm` with residual, tensor-parallel linear layers |

> [!TIP]
> Run vLLM in a single process while reading (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and set breakpoints in `Scheduler.schedule` and `GPUModelRunner.execute_model`. Stepping through one request's life with your own engine's names in your head takes an afternoon and teaches more than a week of reading.

## What production adds (and you consciously skipped)

Write this list into your design doc's "non-goals", each with the module that covers it: tensor/pipeline parallelism (M16), speculative decoding (M15), quantized weights and KV (M14), disaggregated prefill/decode and MoE (M17), overlap/async scheduling (M09 deep dive), structured outputs and tool-call parsing, logprobs and penalties, LoRA adapters, multimodal inputs (M20), KV offload to CPU/SSD, data-parallel load balancing and autoscaling (M18), and observability at fleet scale (M19).

## Exercises

- [ ] Run nano-vllm and your engine on the same L4 with the same offline workload; one table, one paragraph.
- [ ] Read PageServe's scheduler and swap manager; write down one idea you'd steal and one you'd reject, with reasons.
- [ ] Trace one request through vLLM (single-process mode) and fill in the right-hand column of the map with the functions you actually hit.
      */}),
      resources: [
        { title: "vLLM source at the commit Aleksa analyzed", url: "https://github.com/vllm-project/vllm/tree/42172ad", type: "repo", note: "paths in the map match this tree" },
        { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "read scheduler.py, block_manager.py, model_runner.py, layers/attention.py" },
        { title: "PageServe (vLLM_Inference_Engine)", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "12-phase build log; CPU swap; MPS support" },
      ],
    },
  ],

  challenge: {
    title: "Ship mini-vllm: engine, API, benchmarks and a design doc on GitHub",
    md: MD(function () {/*
Build the full engine and publish it as a public GitHub repo you'd be proud to link from your CV. It must run on **Mac (MPS)** and **Colab (CUDA)** from the README's quick-start with no edits.

**Scope** (all required): your own Qwen2/Llama forward; paged KV cache + block manager; prefix caching; continuous batching with chunked prefill and recompute preemption; batched sampler with per-request temperature/top-k/top-p/seed; incremental detokenization with stop strings; OpenAI-compatible `/v1/completions` and `/v1/chat/completions` with SSE streaming; `/metrics`, `/health`, `/v1/models`; separate engine-core process.

**Write-up** (`README.md` + `docs/DESIGN.md`):

1. **README**: one-paragraph pitch, the headline benchmark chart, a 3-command quick-start (install, serve, curl), feature list, test instructions, and an honest "limitations" section.
2. **DESIGN.md** (2–4 pages): architecture diagram (loop + processes); the interfaces; scheduler policy and why (token budget, chunk cap, admission, victim choice); KV layout and block size choice with the M08 arithmetic; prefix-cache design and its invariants; the testing strategy; three decisions you'd change with more time; a "where the gap to vLLM lives" section backed by your step-time breakdown.
3. **BENCHMARKS.md**: offline vs HF (sequential and static batch) on Mac and Colab; online sweep vs vLLM on Colab with the same `vllm bench serve` command; TTFT/ITL/throughput charts; exact commands, versions, hardware.

Hints: keep a `CHANGELOG`-style build log per milestone like PageServe's — it doubles as the story you tell in interviews. Tag a release (`v0.1`) when all acceptance tests pass. Keep generated charts in the repo so the README renders on GitHub.
    */}),
    checklist: [
      "Greedy parity with HF `generate` (fp32 tiny model: exact; real model: first 32 tokens) under chunking, preemption and prefix caching — in CI (GitHub Actions on CPU with the tiny model)",
      "Offline throughput **≥3×** HF sequential on your Mac, and a Colab number vs HF on CUDA",
      "OpenAI SDK works against your server for completions and chat, streaming and non-streaming; client disconnect aborts the request in the engine",
      "`/metrics` exposes TTFT/ITL/E2E histograms, token counters, running/waiting, KV usage and prefix-hit rate",
      "Same-client online benchmark vs vLLM on Colab with TTFT/ITL/throughput charts committed to the repo",
      "`docs/DESIGN.md` with architecture diagram, key decisions, testing strategy and a measured gap analysis",
      "Public GitHub repo with README quick-start that works on Mac and Colab, tagged `v0.1`",
    ],
    stretch: "Pick one and measure it: CUDA graphs + FlashInfer/flash-attn decode on Colab (target: within 2× of vLLM's ITL at batch 1–8); SGLang-style overlap scheduling (prepare step N+1 on a CPU thread while the GPU runs step N); n-gram speculative decoding (preview of M15) inside your token budget; or a second model family (Llama 3.2 1B with rope_scaling) with parity tests.",
  },

  connects: MD(function () {/*
This engine is your **workbench for the rest of the course**. M11–M13 give you the tools to replace its slowest parts with your own CUDA/Triton kernels (M12's paged split-K decode kernel drops straight into `attention.py`) and to profile/compile the step. M14 adds INT8/FP8/INT4 weights and a quantized KV cache; M15 adds speculative decoding inside your token budget; M16 splits the model across GPUs; M17 separates prefill and decode into different engines; M18 puts replicas behind a load balancer and autoscaler that reads your `/metrics`; M19 benchmarks it all like a production team. And in M25 it's the first line of your portfolio: "mini-vLLM — runs, streams, has a benchmark vs. naive HF generate".
  */}),

  interview: [
    "Walk me through one iteration of your engine's loop, from the scheduler's decision to the SSE bytes a client receives. Where does each piece run (process/thread), and why?",
    "Why does vLLM V1 run the engine core in a separate process, and why is detokenization done in the front-end rather than next to the model?",
    "How does your prefix cache decide two blocks are shareable? Why hash the whole prefix chain instead of each block's tokens? What happens to cached blocks when memory runs out?",
    "How do you sample a batch where every request has different temperature, top-k, top-p and seed — without a Python loop? How do you keep a seeded request reproducible when its batch-mates change?",
    "Streaming text token-by-token produced '�' characters and leaked half of a stop string. What went wrong and how did you fix it?",
    "Your engine is 3× slower than vLLM on the same GPU. How do you find out why, and what are the top three causes you'd expect for a 0.5B model?",
    "What does a CUDA graph require of your code, and why did adopting one force you to change your attention implementation?",
    "How did you test correctness of an engine whose outputs depend on batch composition? What exactly can and can't you assert in fp16?",
  ],

  resources: [
    { title: "Aleksa Gordić — Inside vLLM: Anatomy of a High-Throughput LLM Inference System", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "the architecture you're re-implementing, explained end to end" },
    { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "~1,200-line vLLM clone; the performance bar for a small engine" },
    { title: "PageServe (vLLM_Inference_Engine)", url: "https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine", type: "repo", note: "12-phase build log, MPS support, OpenAI SSE server, token-exact tests" },
    { title: "vLLM source (commit 42172ad)", url: "https://github.com/vllm-project/vllm/tree/42172ad", type: "repo", note: "read with the map in the last lesson" },
    { title: "Efficient Memory Management for LLM Serving with PagedAttention", url: "https://arxiv.org/abs/2309.06180", type: "paper", note: "the vLLM paper: block tables, sharing, preemption" },
    { title: "Orca (OSDI '22) — iteration-level scheduling", url: "https://www.usenix.org/conference/osdi22/presentation/yu", type: "paper", note: "continuous batching, the loop at the heart of your engine" },
    { title: "vLLM — OpenAI-compatible server docs", url: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html", type: "docs", note: "the API surface and extra sampling params" },
    { title: "SGLang paper (RadixAttention)", url: "https://arxiv.org/abs/2312.07104", type: "paper", note: "a different prefix-cache design (radix tree) to compare with your hash chain" },
    { title: "FlashInfer", url: "https://github.com/flashinfer-ai/flashinfer", type: "repo", note: "paged attention kernels for your CUDA backend" },
    { title: "Inference Engineering (Baseten) — local PDF, ch. 4", url: "Inference%20Engineering.pdf", type: "book", note: "runtimes (vLLM, SGLang, TensorRT-LLM) and what they optimize" },
  ],
});
