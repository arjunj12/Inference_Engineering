Course.module({
  id: "m18-serving-production",
  title: "Production serving: engines, containers, autoscaling, routing",
  short: "Production serving",
  tagline: "Take a model from `vllm serve` on one GPU to a containerized, autoscaled, routed, load-tested service on Kubernetes — with a runbook you could hand to an on-call engineer.",
  hours: 22,
  level: "core",
  runsOn: ["mac", "colab", "cloud"],
  tags: ["vllm", "sglang", "docker", "kubernetes", "autoscaling", "routing", "keda", "reliability"],

  goal: MD(function () {/*
By the end you have a **deployment repo** that stands up an OpenAI-compatible LLM service and survives a load test. On your laptop it runs on **kind** (Kubernetes-in-Docker) with a vLLM *simulator*; on a rented GPU the same manifests run real vLLM. The load test output looks like this:

```text
$ python loadtest.py        # stages: 5 → 40 → 40 → 5 concurrent users, through the router on :8080
stage  users  rps   ttft_p50  ttft_p99  e2e_p99  errors  replicas
  1      5    1.9    0.21s     0.34s     4.1s     0       1
  2     40   14.8    0.24s     2.90s     9.8s     0       1→3   ← queue grew, KEDA scaled on vllm:num_requests_waiting
  3     40   15.1    0.22s     0.41s     5.2s     0       3     ← steady state, SLO met again
  4      5    1.9    0.20s     0.30s     4.0s     0       3→1   ← scale-down after 5-min cooldown
summary: 7,412 requests, 0 5xx, 0 dropped streams during scale-down (graceful drain worked)
```

…plus a `RUNBOOK.md` that says what to do when p99 TTFT spikes, when a pod is stuck loading weights, and how to roll out a new model version with a canary.

The picture you’ll be able to reason about:
  */}),
  demo: { viz: "autoscaling-sim", params: {} },

  why: MD(function () {/*
Everything so far made **one replica** fast. Production is about the other 90%: getting weights onto GPUs quickly, adding replicas before users notice, sending each request to the replica that can serve it cheapest, and failing gracefully. The Baseten book calls this the **infrastructure layer** (ch. 7) and it is where most “inference engineer” and “ML platform engineer” job postings live: *“build the Kubernetes-native control plane that provisions and runs our GPU inference fleet”*, *“design multi-tenant traffic shaping and SLO-based routing”*. In interviews you will be asked to design exactly this system on a whiteboard — this module makes you someone who has actually run it.
  */}),

  prereqs: [
    {
      title: "Docker & Kubernetes in 15 minutes",
      skipIf: "you have deployed a service on Kubernetes with probes and an HPA",
      md: MD(function () {/*
- **Image** = a frozen filesystem + start command (built from a `Dockerfile`). **Container** = a running image. Containers share the host’s Linux kernel; GPUs are passed in with `--gpus all` (NVIDIA Container Toolkit).
- **Pod** = one or more containers scheduled together on a node. **Deployment** = “keep N identical pods running”. **Service** = stable virtual IP/DNS name that load-balances to healthy pods.
- **Probes**: `startupProbe` (is it done booting?), `readinessProbe` (send it traffic?), `livenessProbe` (restart it?). LLM pods boot slowly (minutes), so the startup probe matters most.
- **HPA** (HorizontalPodAutoscaler) changes a Deployment’s replica count from a metric. **KEDA** is an add-on that feeds HPA with metrics from Prometheus, queues, etc.
- Local clusters: **kind** (`brew install kind`) runs a whole cluster inside Docker on your Mac. No GPUs — which is why we use a vLLM *simulator* locally.

```bash
brew install kind kubectl helm
kind create cluster --name llm && kubectl get nodes
```
      */}),
    },
    {
      title: "Queues and utilization (3-minute math)",
      skipIf: "you know Little’s law and why latency explodes near 100% utilization",
      math: true,
      md: MD(function () {/*
> [!PREREQ] Rates
> A **rate** is “things per second”. If requests arrive at $\lambda = 10$/s and one replica can finish $\mu = 12$/s, its **utilization** is $\rho = \lambda / \mu = 0.83$.

- **Little’s law**: average number of requests in the system $L = \lambda \times W$ (arrival rate × average time in system). 10 req/s × 6 s per request = **60 requests in flight**. That number is what your autoscaler should watch.
- **Queueing blow-up**: for a simple queue, waiting time grows like $\frac{1}{1-\rho}$. At $\rho=0.5$ you wait ~1× the service time, at $0.9$ ~9×, at $0.99$ ~99×. So you never plan to run replicas at 100% — you pick a **concurrency target** below saturation and scale out *before* the queue forms.

Module 19 turns this into a full math lesson; here you only need the intuition.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: an OpenAI-compatible LLM server in one command",
      kind: "demo",
      minutes: 75,
      runsOn: ["cloud", "colab", "mac"],
      md: MD(function () {/*
## The finished thing first

Every production LLM service you will touch exposes the **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming via Server-Sent Events) plus two operational endpoints: `/health` and `/metrics`. Get one running now; the rest of the module is about running *many* of them well.

### Path A — the production path: Docker on a cloud GPU (H100 / L4 / A10G)

```bash
# NVIDIA driver + NVIDIA Container Toolkit are preinstalled on most GPU cloud images
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct --max-model-len 8192
```

- `--ipc=host` gives PyTorch enough shared memory (on Kubernetes you mount an in-memory `/dev/shm` instead).
- The `-v` mount is your **weight cache** — the second start skips the 15 GB download. Remember this; it is half of the cold-start lesson.
- In production you **pin a version tag** (`vllm/vllm-openai:v0.x.y`), never `latest`.

### Path B — free: Colab T4

```bash
!pip install -q vllm
!nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 > vllm.log 2>&1 &
!sleep 90 && tail -n 20 vllm.log
```

T4 (Turing) has no BF16, hence `--dtype half`. If your vLLM version refuses the GPU, switch the runtime to an L4 or use Path A/C.

### Path C — your Mac: llama.cpp’s server speaks the same API

```bash
brew install llama.cpp
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8000 -np 4 --metrics
```

`-np 4` = 4 parallel slots (llama.cpp’s continuous batching); `--metrics` exposes Prometheus metrics. Different engine, **same contract** — that is the whole point of the OpenAI-compatible API.

## Poke it

```bash
curl -s localhost:8000/v1/models | python -m json.tool
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/health          # 200 when ready
curl -N localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "Name three GPU memory types."}],
  "stream": true, "max_tokens": 64}'
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)"
```

And from Python, with the client your users will actually use:

```python
# client.py — pip install openai
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
model = client.models.list().data[0].id

t0 = time.perf_counter(); first = None; n = 0
stream = client.chat.completions.create(
    model=model, stream=True, max_tokens=200, temperature=0,
    messages=[{"role": "user", "content": "Explain continuous batching to a backend engineer."}])
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.perf_counter(); n += 1
t1 = time.perf_counter()
print(f"TTFT {1000*(first-t0):.0f} ms | {n/(t1-first):.1f} chunks/s | e2e {t1-t0:.2f} s")
```

## Read the startup log like an SRE

Restart the server and timestamp each phase from the log. For vLLM you will see, in order: weight download → `Loading weights took … GB / … s` → `torch.compile` / CUDA-graph capture → `GPU KV cache size: N tokens` → `Maximum concurrency for 8192 tokens per request: X` → `Application startup complete`.

| Phase | Your seconds (cold) | Your seconds (warm cache) |
|---|---|---|
| Weight download | | |
| Weight load to GPU | | |
| Compile + CUDA graphs | | |
| KV cache allocation | | |
| **Total until `/health` = 200** | | |

That total is your **cold start** — the number that decides how fast you can autoscale. The two KV lines tell you the **maximum concurrency** one replica can hold, which becomes your autoscaler’s target.

::viz inference-stack

> [!INTUITION] Where we are on the map
> Modules 06–17 lived inside the “engine” and “kernels” boxes. This module is the boxes *above* them: gateway, router, autoscaler, and the container/cluster that hosts the engine.

- [ ] Server running on at least one path; streamed a response with curl and with the OpenAI SDK
- [ ] Filled the cold-start table for a cold and a warm start
- [ ] Wrote down “GPU KV cache size” and “Maximum concurrency” from the log
- [ ] Fired 32 concurrent requests (asyncio + the SDK’s `AsyncOpenAI`) and watched `vllm:num_requests_running` climb, then `num_requests_waiting` appear
      */}),
      resources: [
        { title: "vLLM — Using Docker", url: "https://docs.vllm.ai/en/latest/deployment/docker.html", type: "docs", note: "official image flags (--ipc=host, cache mounts, building your own)" },
        { title: "vLLM — OpenAI-compatible server", url: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html", type: "docs", note: "every endpoint and extra parameter the server accepts" },
        { title: "llama.cpp server", url: "https://github.com/ggml-org/llama.cpp/tree/master/tools/server", type: "repo", note: "the Mac path: OpenAI API, slots, /metrics" },
      ],
    },
    {
      id: "engines",
      title: "The engine landscape and how to choose",
      kind: "concept",
      minutes: 90,
      md: MD(function () {/*
## One contract, many engines

All serious engines now share the same core (you built it in M10): **continuous batching, paged KV cache, prefix caching, chunked prefill, quantized kernels, speculative decoding, tensor parallelism**, and an **OpenAI-compatible server**. They differ in *peak performance*, *model coverage*, *hardware*, and *how much engineering they cost you*.

| Engine | From | Sweet spot | Hardware | Serve command | Watch out for |
|---|---|---|---|---|---|
| **vLLM** | UC Berkeley → PyTorch Foundation | Default choice: almost any open model, day-0 support, huge community | NVIDIA, AMD, Intel, TPU, CPU | `vllm serve <model>` | Broadness costs some peak perf; fast-moving flags — pin versions |
| **SGLang** | LMSYS | High-throughput MoE (DeepSeek, Qwen, Kimi), RadixAttention prefix reuse, structured programs; large multi-node EP | NVIDIA, AMD | `python -m sglang.launch_server --model-path <model>` | Fewer exotic architectures than vLLM |
| **TensorRT-LLM** | NVIDIA | Peak latency/throughput on Hopper/Blackwell, FP8/NVFP4, deepest Dynamo integration | NVIDIA only | `trtllm-serve <model>` + YAML config | More engineering; V0 (TensorRT plugin, compiled engines) vs V1 (PyTorch-based) — know which you run |
| **Triton Inference Server** | NVIDIA | Multi-framework model server (TRT-LLM backend, ONNX, PyTorch), ensembles/pipelines | NVIDIA (mostly) | `tritonserver --model-repository …` | A *server* around engines, not an engine itself |
| **llama.cpp / Ollama** | ggml community | Laptops, edge, CPU+GPU offload, GGUF quants; Apple Metal | Everything, esp. Apple Silicon | `llama-server -hf <repo>` / `ollama serve` | Not built for high-concurrency datacenter serving |
| **MLX (mlx-lm)** | Apple | Fastest path on Apple Silicon; research on Mac | Apple Silicon | `mlx_lm.server --model <repo>` | Single-machine, modest concurrency |
| **TGI** | Hugging Face | Historic default; now in **maintenance mode** — HF recommends vLLM/SGLang | NVIDIA, AMD, Gaudi | `text-generation-launcher` | Don’t start new projects on it |
| **LMDeploy** | InternLM / Shanghai AI Lab | TurboMind C++ engine: strong 4-bit AWQ, InternVL/Qwen VLMs | NVIDIA (+ Ascend) | `lmdeploy serve api_server <model>` | Smaller community outside China |

> [!REAL] What production teams actually do
> Baseten uses all three big engines and picks per deployment — **mostly TensorRT-LLM** for peak performance (Inference Engineering, ch. 4.3). Many teams start on vLLM, move hot, stable models to TRT-LLM or a tuned SGLang once traffic justifies the engineering, and put a **distributed layer** (NVIDIA Dynamo, llm-d) above the engines when they need KV-aware routing or disaggregation. xAI runs SGLang; frontier labs run custom engines that look a lot like these.

## How to choose (decision checklist)

1. **Is the architecture supported?** New model this week → vLLM or SGLang nightly (Baseten ch. 4: “day-zero support often needs nightly builds”). Check the engine’s supported-models list *before* anything else.
2. **Which hardware?** Apple → MLX/llama.cpp. AMD/TPU → vLLM (or SGLang on AMD). Hopper/Blackwell and you need the last 20–40% → TRT-LLM.
3. **What shape of traffic?** Huge MoE with high throughput → SGLang or vLLM with expert parallelism (M17). Many shared prefixes (agents, RAG) → both do automatic prefix caching; SGLang’s radix tree was designed for it.
4. **How much engineering can you spend?** vLLM/SGLang: `pip install`, minutes. TRT-LLM: days of config tuning, build caches per GPU type.
5. **Features you can’t live without**: multi-LoRA, structured outputs, specific quant format (NVFP4, AWQ, GGUF), multimodal inputs, disaggregation.

## The rules that keep you out of trouble

- **Benchmark on *your* workload**, not the engine’s blog post. Engine A beating B by 30% on ShareGPT tells you little about 20K-token RAG prompts (M19).
- **Compare tuned vs tuned.** Default `max-num-seqs`, memory fractions and chunk sizes differ between engines; a default-vs-tuned comparison is meaningless.
- **Pin everything**: engine version, CUDA, driver, model revision (HF commit hash). A minor engine bump can change default kernels and your p99.
- **Check quality after switching engines.** Chat templates, tokenizer settings and sampling defaults differ subtly; run your evals (Baseten ch. 1.3.1) before celebrating a speedup.

<details><summary>Go deeper: why TRT-LLM is faster and when it isn’t</summary>

TRT-LLM ships NVIDIA-written kernels (some closed-source) tuned per architecture (e.g. XQA attention, FP8/NVFP4 GEMMs), aggressive fusion, and an in-flight batching C++ runtime. The gain is largest on Hopper/Blackwell with well-supported architectures and quantized weights; on an older/smaller GPU or an unusual architecture, vLLM or SGLang often match it with far less effort. Always measure.

</details>

- [ ] Wrote a one-paragraph engine choice for three scenarios: (a) a fine-tuned 8B chat model on 2×L4, (b) DeepSeek-V3-class MoE on 8×H200, (c) an internal tool on Mac minis
- [ ] Served the same model with two engines (e.g. vLLM and SGLang on a cloud GPU, or llama.cpp and mlx-lm on Mac) and compared TTFT/decode speed at concurrency 1 and 16 with your M00 benchmark script
      */}),
      resources: [
        { title: "SGLang docs", url: "https://docs.sglang.ai/", type: "docs", note: "launch_server flags, router, RadixAttention" },
        { title: "TensorRT-LLM", url: "https://github.com/NVIDIA/TensorRT-LLM", type: "repo", note: "trtllm-serve, supported models, performance docs" },
        { title: "Triton Inference Server", url: "https://github.com/triton-inference-server/server", type: "repo", note: "the model server that hosts TRT-LLM and other backends" },
        { title: "LMDeploy", url: "https://github.com/InternLM/lmdeploy", type: "repo", note: "TurboMind engine; strong 4-bit and VLM support" },
        { title: "TGI (maintenance mode notice)", url: "https://github.com/huggingface/text-generation-inference", type: "repo", note: "read the README banner — why the ecosystem consolidated" },
      ],
    },
    {
      id: "engine-knobs",
      title: "Engine config knobs that actually matter",
      kind: "lab",
      minutes: 150,
      runsOn: ["cloud", "colab"],
      md: MD(function () {/*
## Why a handful of flags decide your cost

An engine is a **memory budget** plus a **scheduler**. Almost every important flag either changes how GPU memory is split (weights vs activations vs KV cache) or how much work the scheduler packs into one step. Get these right and you often gain 1.5–2× before touching a kernel. (Baseten ch. 5: an engineer scripted **77 configurations** and found a non-obvious one that **doubled** throughput for a code-completion model.)

| vLLM flag | SGLang | TRT-LLM (YAML) | What it trades |
|---|---|---|---|
| `--gpu-memory-utilization 0.9` | `--mem-fraction-static` | `kv_cache_config.free_gpu_memory_fraction` | Fraction of GPU memory the engine may use; the part not taken by weights/activations becomes **KV cache** → max concurrency. Too high → OOM when something else needs VRAM |
| `--max-model-len` | `--context-length` | `max_seq_len` | Longest request (prompt + output). Lower = fewer surprises, smaller worst-case KV per sequence |
| `--max-num-seqs` | `--max-running-requests` | `max_batch_size` | Max sequences decoding together = the **batch size ceiling**. Higher = throughput, worse per-user tok/s |
| `--max-num-batched-tokens` | `--chunked-prefill-size` | `max_num_tokens` | Token budget per engine step. Controls **chunked prefill**: small = smooth ITL, big = faster TTFT for long prompts (M09) |
| `--enable-prefix-caching` (on by default in V1) | on by default (RadixAttention) | `enable_block_reuse` | Reuse KV for shared prefixes. Free win for agents/RAG/multi-turn |
| `--quantization fp8` / pre-quantized checkpoint | `--quantization fp8` | quantized checkpoint | Weight bytes ↓ → decode faster, more KV room (M14). Lossy: eval it |
| `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e4m3` | `kv_cache_config.dtype` | Halves KV bytes → ~2× concurrency at long context. Small quality risk |
| `--tensor-parallel-size N` | `--tp N` | `tensor_parallel_size` | Split each layer across N GPUs (M16): lower latency, more memory, all-reduce cost |
| `--enforce-eager` | `--disable-cuda-graph` | `cuda_graph_config: null` | Skip CUDA graphs: faster startup, slower decode at small batch (M13) |

## The memory budget, worked

Take **Llama-3.1-8B in BF16 on one H100 80 GB** with `--gpu-memory-utilization 0.9`:

- Budget: $0.9 \times 80 = 72$ GB. Weights: $8.0\text{B} \times 2 = 16$ GB. Activations, CUDA graphs, workspace: ~2–4 GB. **KV pool ≈ 52 GB.**
- KV per token (32 layers, 8 KV heads, head dim 128, BF16): $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = **128 KB**.
- KV capacity: $52\text{ GB} / 128\text{ KB} \approx 400\text{K}$ tokens. At `--max-model-len 8192` worst case that is **~50 full-length sequences**; at a realistic 2K tokens per request, **~200**.

That is exactly what vLLM prints as `GPU KV cache size` and `Maximum concurrency for 8192 tokens per request`. FP8 weights free 8 GB (+~60K tokens); FP8 KV doubles the token count.

::viz kv-calculator

> [!WARNING] `max-num-seqs` above KV capacity is a lie
> If `max-num-seqs=512` but the KV pool only fits 200 average requests, the scheduler admits them, runs out of blocks, and **preempts** (evicts and later recomputes) sequences — watch `vllm:num_preemptions`. Throughput drops and p99 explodes. Size the batch cap to what the KV pool and your latency SLO allow.

## Lab: a tuning sweep (the 77-configs habit)

On a cloud GPU (or Colab with a 1.5B model), script it — never tune by hand:

```bash
#!/usr/bin/env bash
# sweep.sh — one server per config, same benchmark each time
MODEL=Qwen/Qwen2.5-7B-Instruct
for SEQS in 32 64 128 256; do
 for TOK in 2048 8192; do
  vllm serve $MODEL --max-model-len 8192 --max-num-seqs $SEQS \
       --max-num-batched-tokens $TOK --port 8000 > server_${SEQS}_${TOK}.log 2>&1 &
  PID=$!
  until curl -sf localhost:8000/health > /dev/null; do sleep 5; done
  vllm bench serve --model $MODEL --dataset-name random \
       --random-input-len 1024 --random-output-len 256 --num-prompts 400 \
       --max-concurrency 128 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
       --save-result --result-dir sweep --result-filename seqs${SEQS}_tok${TOK}.json
  kill $PID; wait $PID 2>/dev/null
 done
done
```

Then load every JSON (`output_throughput`, `p99_ttft_ms`, `p99_tpot_ms` are keys in the saved result) into a pandas table and pick the config with the **highest throughput whose p99 TPOT meets your SLO** (e.g. ≤ 50 ms). That selection rule — *max throughput subject to latency* — is the whole game, and M19 formalizes it as **goodput**.

- [ ] Predicted KV capacity for your model/GPU on paper; compared with the logged `GPU KV cache size` (within ~10%)
- [ ] Ran the sweep (≥ 6 configs) and produced a table: config → output tok/s, p99 TTFT, p99 TPOT
- [ ] Found a config where raising `max-num-seqs` made p99 *worse* without raising throughput — and explained why
- [ ] Tried `--kv-cache-dtype fp8` (Ada/Hopper) or a smaller `--max-model-len` and recorded the change in “Maximum concurrency”
      */}),
      resources: [
        { title: "vLLM — Optimization and tuning", url: "https://docs.vllm.ai/en/latest/configuration/optimization.html", type: "docs", note: "preemption, chunked prefill, batching knobs explained by the maintainers" },
        { title: "vLLM — Engine arguments", url: "https://docs.vllm.ai/en/latest/configuration/engine_args.html", type: "docs", note: "the full flag reference; check defaults for your version" },
      ],
    },
    {
      id: "containers-cold-start",
      title: "Containers, model loading and the anatomy of a cold start",
      kind: "build",
      minutes: 150,
      runsOn: ["cloud", "mac"],
      md: MD(function () {/*
## Cold start = the tax on every scale-up

When traffic rises, a new replica is useless until it answers `/health`. Baseten (ch. 7.2.2) splits the cold start into four parts — optimize each separately:

| Phase | Typical (8B BF16, 16 GB weights) | Main levers |
|---|---|---|
| **GPU procurement** (new node) | 0 s (warm pool) → 2–10 min (cloud VM) | warm/reserved pools, node start time is negotiable in contracts |
| **Image pull** | 1–5 min for a many-GB engine image | pre-pull on GPU nodes (DaemonSet), keep images lean, same-region registry |
| **Weight download + load** | 16 GB ÷ 150 MB/s ≈ **107 s** from the internet; ÷ 2 GB/s ≈ **8 s** from nearby storage | never bake weights into the image; cache near the GPU; stream straight to GPU |
| **Engine startup** | 20–120 s: torch.compile, CUDA-graph capture, KV profiling (TRT-LLM engine *build*: minutes) | persist compile caches; cache built engines per GPU type |

> [!INTUITION] Napkin math beats guessing
> Cold start ≈ image pull + $\frac{\text{weight bytes}}{\text{storage bandwidth}}$ + engine init. Quantized weights load faster simply because there are fewer bytes (FP8 8B ≈ 8 GB). Hugging Face and S3 egress are **slow** for this purpose; the book’s rule is that you need **GB/s from a cache physically near the GPUs**.

## A production-shaped Dockerfile

```dockerfile
# Dockerfile — thin layer on the official image; weights are NOT inside
FROM vllm/vllm-openai:v0.21.0          # pin the release you benchmarked

# Only what you need; every GB here is paid on every node pull
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]"

ENV HF_HOME=/models/hf \
    VLLM_CACHE_ROOT=/models/vllm-cache \
    MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
    MODEL_REVISION=main

COPY serve.sh /serve.sh
RUN chmod +x /serve.sh
ENTRYPOINT ["/serve.sh"]
```

```bash
#!/usr/bin/env bash
# serve.sh — download once into the mounted cache, then exec the engine
set -euo pipefail
LOCAL=/models/weights/${MODEL_ID//\//--}
if [ ! -f "$LOCAL/config.json" ]; then
  hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$LOCAL"
fi
exec vllm serve "$LOCAL" --served-model-name "$MODEL_ID" --host 0.0.0.0 --port 8000 "$@"
```

Why each choice:
- **Pinned base image** — the image *is* your CUDA + PyTorch + engine version. Baseten ch. 7.1.1: “pack light, pin exact versions”, and rebuild on a stable release a few weeks after a nightly you needed for a new model.
- **Weights on a volume** (`/models`, a PersistentVolume or node-local NVMe) — the image stays small and the same image serves any model.
- **`VLLM_CACHE_ROOT` on the volume** — vLLM stores torch.compile artifacts there; the second boot skips compilation.
- **`--served-model-name`** — clients keep calling `Qwen/Qwen2.5-7B-Instruct` even though you load from a local path.
- **`exec`** — the engine becomes PID 1 and receives `SIGTERM` directly (needed for graceful draining later).

## Loading faster: safetensors and streaming

- **safetensors** (Hugging Face) is a flat, memory-mappable tensor file: no pickle, so **no code runs on load** (a security win vs `.bin` pickles) and the OS can page it straight in. Big models are sharded (`model-00001-of-00004.safetensors`).
- **Run:ai Model Streamer** reads many tensors concurrently from local disk or object storage (S3, GCS, Azure) and streams them to GPU memory while still reading:

```bash
pip install "vllm[runai]"
vllm serve s3://my-bucket/qwen2.5-7b-instruct --load-format runai_streamer
```

- **Keep the process warm instead of cold-starting**: vLLM’s sleep mode (`--enable-sleep-mode`) can offload weights and drop the KV cache while keeping the process and compiled graphs alive — useful for swapping models on one GPU and for RL (M23).
- **Scale-to-zero** is only viable with fast cold starts *and* a durable queue in front (Baseten ch. 7.2.4). For latency-sensitive apps with light, unscheduled traffic, the book’s advice is blunt: use a pay-per-token API instead.

## Lab: measure your own cold start

On a cloud GPU (or with llama.cpp on your Mac, same idea with GGUF files):

1. Build the image, run it with an **empty** `/models` volume: time download, load, compile, ready.
2. Run again with the volume warm: only load + init remain.
3. Run with `--enforce-eager`: faster init, then benchmark decode to see what you paid.
4. Copy weights to local NVMe and try `--load-format runai_streamer`; compare load time.

```bash
docker build -t my-llm:0.1 .
time docker run --rm --gpus all --ipc=host -p 8000:8000 -v $PWD/models:/models my-llm:0.1 &
until curl -sf localhost:8000/health; do sleep 1; done; echo READY
```

- [ ] Cold-start table for 4 variants (cold, warm weights, warm + compile cache, eager) with the dominant phase highlighted
- [ ] Image size recorded (`docker images`) and one change that made it smaller
- [ ] Napkin prediction of weight-load time vs measurement, with the storage bandwidth you assumed
- [ ] Wrote two sentences for your runbook: “a pod stuck in startup for > N minutes probably means …”
      */}),
      resources: [
        { title: "Run:ai Model Streamer in vLLM", url: "https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html", type: "docs", note: "streaming safetensors from S3/GCS/local disk" },
        { title: "runai-model-streamer", url: "https://github.com/run-ai/runai-model-streamer", type: "repo", note: "the library itself; tuning concurrency and memory limits" },
        { title: "safetensors", url: "https://huggingface.co/docs/safetensors/index", type: "docs", note: "why the format is safe and fast to load" },
        { title: "vLLM sleep mode", url: "https://docs.vllm.ai/en/latest/features/sleep_mode.html", type: "docs", note: "keep a warm process without holding GPU memory" },
      ],
    },
    {
      id: "kubernetes",
      title: "Kubernetes for inference: probes, GPUs, and the platform zoo",
      kind: "build",
      minutes: 180,
      runsOn: ["mac", "cloud"],
      md: MD(function () {/*
## The plan: real manifests, fake GPU

You can learn 90% of inference-on-Kubernetes on a laptop. **kind** runs a Kubernetes cluster inside Docker, and **llm-d-inference-sim** is a tiny server that speaks the vLLM OpenAI API, fakes TTFT/ITL, caps concurrency like `--max-num-seqs`, and exports vLLM-named Prometheus metrics. Your manifests, probes, autoscaler and router behave exactly as they would against a real engine; only the GPU spec differs.

```bash
brew install kind kubectl helm        # or the Linux/Windows equivalents
kind create cluster --name llm
kubectl get nodes
```

## Deployment + Service with the three probes

```yaml
# k8s/sim.yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: llm, labels: { app: llm } }
spec:
  replicas: 1
  selector: { matchLabels: { app: llm } }
  template:
    metadata: { labels: { app: llm } }
    spec:
      terminationGracePeriodSeconds: 120      # >= your longest request
      containers:
      - name: engine
        image: ghcr.io/llm-d/llm-d-inference-sim:v0.11.2
        args: ["--model", "qwen-sim", "--port", "8000",
               "--max-num-seqs", "8",
               "--time-to-first-token", "200ms", "--inter-token-latency", "20ms",
               "--startup-duration", "30s"]      # pretend to load weights
        ports: [{ name: http, containerPort: 8000 }]
        startupProbe:                            # allow a long cold start...
          httpGet: { path: /health/ready, port: http }
          periodSeconds: 5
          failureThreshold: 120                  # ...up to 10 minutes
        readinessProbe:                          # only route when ready
          httpGet: { path: /health/ready, port: http }
          periodSeconds: 5
        livenessProbe:                           # restart if wedged
          httpGet: { path: /health, port: http }
          periodSeconds: 10
          failureThreshold: 3
        lifecycle:
          preStop: { exec: { command: ["sleep", "15"] } }   # let the LB notice
        resources:
          requests: { cpu: "100m", memory: "128Mi" }
---
apiVersion: v1
kind: Service
metadata: { name: llm, labels: { app: llm } }
spec:
  selector: { app: llm }
  ports: [{ name: http, port: 8000, targetPort: http }]
```

```bash
kubectl apply -f k8s/sim.yaml
kubectl get pods -w                       # 0/1 Running for ~30 s, then 1/1
kubectl port-forward svc/llm 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-sim","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
curl -s localhost:8000/metrics | grep -E 'vllm:num_requests_(running|waiting)'
```

> [!WARNING] The #1 inference-on-K8s outage
> A **liveness** probe with a short timeout on a pod that takes 4 minutes to load weights → Kubernetes kills it at 30 s → restart loop forever (`CrashLoopBackOff`). Use a **startupProbe** with a generous budget; liveness only kicks in after startup succeeds. And never point liveness at an endpoint that can be slow under load (a saturated engine is *busy*, not *dead*).

## The GPU version (cloud cluster)

Differences only; everything above carries over:

```yaml
    spec:
      nodeSelector: { nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3 }  # labels from GPU feature discovery
      tolerations: [{ key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }]
      containers:
      - name: engine
        image: my-registry/my-llm:0.1                      # the image from the last lesson
        args: ["--max-model-len", "8192", "--max-num-seqs", "64"]
        resources:
          limits: { nvidia.com/gpu: 1 }                    # device plugin / GPU Operator
        volumeMounts:
        - { name: models, mountPath: /models }
        - { name: dshm, mountPath: /dev/shm }              # NCCL/PyTorch need shared memory for TP
      volumes:
      - { name: models, persistentVolumeClaim: { claimName: model-cache } }
      - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
```

Plus a **PodDisruptionBudget** (`maxUnavailable: 1`) so a node upgrade never evicts every replica at once. The **NVIDIA GPU Operator** installs the driver, container toolkit, device plugin (which advertises `nvidia.com/gpu`) and the DCGM exporter you will scrape in M19.

## The platform zoo (what to use when)

Raw Deployments get you far. Past that, pick a layer that already knows about LLMs:

| Project | What it adds | Reach for it when |
|---|---|---|
| **KServe** (CNCF) | `InferenceService` / `LLMInferenceService` CRDs, autoscaling, canary rollouts, storage initializers; llm-d integration | You want a standard model-serving API across many models and teams |
| **llm-d** (Red Hat, Google, IBM, NVIDIA…) | vLLM + Gateway API Inference Extension: KV-cache/prefix-aware scheduling, prefill/decode disaggregation, variant autoscaling | Large vLLM fleets where routing and P/D split pay off |
| **NVIDIA Dynamo** | Engine-agnostic (vLLM, SGLang, TRT-LLM) distributed layer: KV-aware router, disaggregated serving, KV offload (KVBM), NIXL transfers, SLA planner | Multi-node, disaggregated, NVIDIA-heavy fleets (Baseten ch. 4.4) |
| **Ray Serve LLM** | Python-first: `LLMConfig` + `build_openai_app`, autoscaling on ongoing requests, multi-model, composes with Ray Data | Your team already lives in Ray, or you need custom Python pipelines around the model |
| **vLLM production-stack** | Helm chart: vLLM pods + router (round-robin/session/prefix-aware) + Prometheus/Grafana + LMCache | Fastest “reference architecture” for vLLM on K8s |
| **AIBrix** (vLLM project) | Gateway, LoRA management, LLM-specific autoscaler, distributed KV cache | vLLM fleets with many LoRA adapters |
| **LeaderWorkerSet (LWS)** | One logical replica spanning several pods (multi-node TP/PP) | Models too large for one node (M16) |

Most of these converge on the same ideas you are about to build by hand: **autoscale on concurrency/queue**, **route on cache locality**, **health-check the engine, not the pod**.

- [ ] kind cluster running the sim; `kubectl get pods -w` shows the 30 s not-ready window caused by `--startup-duration`
- [ ] Broke it on purpose: removed the startupProbe and set liveness `initialDelaySeconds: 5` with `--startup-duration 60s`; observed the restart loop, then fixed it
- [ ] Wrote the GPU overlay (nodeSelector, tolerations, `nvidia.com/gpu`, `/dev/shm`, PVC) even if you cannot apply it
- [ ] One paragraph: which platform from the table you would pick for a 3-model, 20-GPU product and why
      */}),
      resources: [
        { title: "llm-d-inference-sim", url: "https://github.com/llm-d/llm-d-inference-sim", type: "tool", note: "vLLM-compatible simulator for testing K8s plumbing without GPUs" },
        { title: "kind", url: "https://kind.sigs.k8s.io", type: "tool", note: "local Kubernetes in Docker" },
        { title: "Kubernetes — liveness, readiness and startup probes", url: "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/", type: "docs", note: "startupProbe semantics in detail" },
        { title: "vLLM — Deploying with Kubernetes", url: "https://docs.vllm.ai/en/latest/deployment/k8s.html", type: "docs", note: "reference manifests for the real GPU deployment" },
        { title: "KServe", url: "https://kserve.github.io/website/", type: "docs", note: "LLMInferenceService and generative inference docs" },
        { title: "llm-d", url: "https://llm-d.ai", type: "docs", note: "well-lit paths for vLLM at scale on Kubernetes" },
      ],
    },
    {
      id: "autoscaling",
      title: "Autoscaling on the right signal",
      kind: "build",
      minutes: 180,
      runsOn: ["mac", "cloud"],
      md: MD(function () {/*
## GPU utilization is the wrong default

The classic Kubernetes HPA scales on CPU. The tempting LLM equivalent is “scale on GPU utilization” — but `nvidia-smi` utilization only says a kernel was running, not how full the batch is; a GPU decoding 2 sequences shows ~100% just like one decoding 64. Baseten (ch. 7.2.1) distinguishes:

- **Utilization signals** (GPU busy, KV-cache usage) — *lagging*: they rise after the damage (queueing) has started.
- **Traffic signals** (requests in flight, queue depth, RPS) — *proactive*: they rise the moment load arrives.

The book’s recommendation: use both, but drive scaling primarily from **concurrency**: `num_requests_running + num_requests_waiting` per replica, compared with a **target concurrency** equal to the batch size at which your latency SLO still holds (found by the M19 sweep).

::viz queueing

> [!INTUITION] Why queues explode near 100%
> For a simple queue, wait time grows like $\frac{\rho}{1-\rho}$ where $\rho$ is utilization. At $\rho = 0.5$ the factor is 1; at 0.9 it is 9; at 0.95 it is 19. Aim replicas at ~70–80% of their SLO-safe concurrency so bursts have headroom while new pods cold-start.

## Baseten’s five parameters → Kubernetes

| Baseten (ch. 7.2.1) | Meaning | KEDA / HPA equivalent |
|---|---|---|
| Min replicas | Floor; 0 = scale to zero | `minReplicaCount` |
| Max replicas | Cost ceiling, capacity cap | `maxReplicaCount` |
| Concurrency target | Requests per replica before adding one | trigger `threshold` with `metricType: AverageValue` |
| Autoscaling window | How long to average the signal | PromQL range (`avg_over_time(...[1m])`) + `stabilizationWindowSeconds` for scale-up |
| Scale-down delay | Wait before removing replicas (avoid thrash) | `behavior.scaleDown.stabilizationWindowSeconds` |

::viz autoscaling-sim

## Build: Prometheus + KEDA on kind

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm install kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
helm install keda kedacore/keda -n keda --create-namespace
kubectl get svc -n monitoring          # note the Prometheus service name/port
```

Tell Prometheus to scrape the engine pods:

```yaml
# k8s/podmonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata: { name: llm, labels: { release: kps } }   # label must match the chart's selector
spec:
  selector: { matchLabels: { app: llm } }
  podMetricsEndpoints: [{ port: http, path: /metrics, interval: 5s }]
```

Scale on requests in flight per replica:

```yaml
# k8s/scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: llm }
spec:
  scaleTargetRef: { name: llm }
  minReplicaCount: 1
  maxReplicaCount: 6
  pollingInterval: 5
  cooldownPeriod: 300
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:   { stabilizationWindowSeconds: 0,   policies: [{ type: Pods, value: 2, periodSeconds: 15 }] }
        scaleDown: { stabilizationWindowSeconds: 300, policies: [{ type: Pods, value: 1, periodSeconds: 60 }] }
  triggers:
  - type: prometheus
    metricType: AverageValue          # total / replicas compared to threshold
    metadata:
      serverAddress: http://kps-kube-prometheus-prometheus.monitoring:9090   # check the svc name
      query: sum(vllm:num_requests_running{namespace="default"}) + sum(vllm:num_requests_waiting{namespace="default"})
      threshold: "6"                  # sim max-num-seqs is 8 → target 75%
```

The query returns *total* in-flight requests; with `AverageValue` the HPA computes desired replicas ≈ $\lceil \text{total} / 6 \rceil$. (If the sim labels metrics differently in your version, run the query in the Prometheus UI first — `kubectl port-forward -n monitoring svc/<prometheus-svc> 9090`.)

## A load test with stages

```python
# loadtest.py — ramp concurrency, record latency and errors per stage
import asyncio, time, statistics, httpx
URL, MODEL = "http://localhost:8000/v1/chat/completions", "qwen-sim"
STAGES = [(4, 60), (24, 120), (48, 120), (4, 120)]     # (concurrent users, seconds)

async def user(client, stop, lat, errs):
    while time.time() < stop:
        t0 = time.perf_counter()
        try:
            r = await client.post(URL, json={"model": MODEL, "max_tokens": 100,
                "messages": [{"role": "user", "content": "Tell me about GPUs"}]}, timeout=60)
            r.raise_for_status(); lat.append(time.perf_counter() - t0)
        except Exception:
            errs.append(1)

async def main():
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as c:
        for users, secs in STAGES:
            lat, errs = [], []
            stop = time.time() + secs
            await asyncio.gather(*[user(c, stop, lat, errs) for _ in range(users)])
            q = statistics.quantiles(lat, n=100) if len(lat) > 2 else [0] * 99
            print(f"users={users:3d} ok={len(lat):5d} err={len(errs):3d} p50={q[49]:.2f}s p99={q[98]:.2f}s")
asyncio.run(main())
```

Run `kubectl get hpa,pods -w` in another terminal. Expect: at 24 users p99 climbs (queueing at 8 per pod), KEDA adds pods, new pods sit **not-ready for 30 s** (`--startup-duration`), then p99 recovers; at the final stage replicas hold for 5 minutes, then step down one at a time.

- [ ] Prometheus shows `vllm:num_requests_running` for every pod
- [ ] Scale-up observed: a timeline (replicas vs time vs p99) from the load test
- [ ] Measured *reaction time*: first queueing → new pod ready; explained which part is the startup duration
- [ ] Changed the threshold to 8 (100%) and showed the p99 penalty; changed scale-down to 0 s and showed thrashing
- [ ] Wrote down the target-concurrency number you would use for a real engine and where it comes from
      */}),
      resources: [
        { title: "KEDA — Prometheus scaler", url: "https://keda.sh/docs/latest/scalers/prometheus/", type: "docs", note: "trigger fields, AverageValue vs Value" },
        { title: "Kubernetes — Horizontal Pod Autoscaling", url: "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/", type: "docs", note: "the behavior field: stabilization windows and policies" },
        { title: "kube-prometheus-stack chart", url: "https://github.com/prometheus-community/helm-charts", type: "repo", note: "Prometheus, Grafana, operator CRDs in one install" },
      ],
    },
    {
      id: "routing",
      title: "Routing and load balancing: cache-aware beats round-robin",
      kind: "build",
      minutes: 150,
      runsOn: ["mac", "any"],
      md: MD(function () {/*
## A load balancer spreads; a router decides

A Kubernetes Service does connection-level round-robin: fine for stateless web apps, wasteful for LLMs, because every replica holds **state** that makes some requests cheaper there: prefix KV cache (M09), loaded LoRA adapters, a current queue. Baseten (ch. 7.2.3) separates the **load balancer** (spread traffic, health) from the **router** (pick the *best* replica for this request), and both from a **queue** (hold work when every replica is full).

| Policy | Picks | Good at | Bad at |
|---|---|---|---|
| Round-robin / random | Next replica | Uniform, short, cache-free requests | Long-tailed request sizes; wastes prefix cache |
| Least outstanding requests | Replica with fewest in flight | Uneven request lengths | Ignores cache locality |
| **Session affinity** (hash of user/conversation id) | Same replica per session | Multi-turn chat reuse | Hot users overload one replica |
| **Prefix / KV-cache-aware** | Replica with the longest cached prefix, penalized by load | Agents, RAG with shared system prompts, multi-turn | Needs cache state (approximate or event-driven) |
| **LoRA-aware** | Replica that already has the adapter loaded | Multi-tenant fine-tunes | Adapter hot-spots |

The sweet spot almost everyone converges on: **score = cache overlap − λ · load**, or equivalently **consistent hashing with bounded loads** (hash the prefix, but spill to the next replica if the chosen one is above, say, 1.25× the mean load).

::viz load-balancing

::viz prefix-cache

Who ships this: the **Gateway API Inference Extension** (an Envoy-based gateway plus an *endpoint picker* that reads vLLM metrics: queue, KV usage, loaded LoRAs, prefix match) — the routing core of llm-d and KServe; **NVIDIA Dynamo’s KV router** (tracks KV blocks per worker via events and scores overlap vs load); **vLLM production-stack’s router** (round-robin, session, prefix-aware); **SGLang’s router** (cache-aware, originally from the SGLang team). Mooncake (Kimi) showed cache-aware scheduling plus a shared KV store as the core of a large production service.

## Build: a 100-line cache-aware router

```python
# router.py — FastAPI proxy: prefix-affinity with bounded load, streaming passthrough
import hashlib, itertools, httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

BACKENDS = ["http://llm-0:8000", "http://llm-1:8000", "http://llm-2:8000"]
inflight = {b: 0 for b in BACKENDS}
MAX_PER_BACKEND, SLACK = 8, 1.25
PREFIX_CHARS = 512                         # "prefix" = start of the prompt
client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=2))
app = FastAPI()

def ring(key: str):
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    start = h % len(BACKENDS)
    return [BACKENDS[(start + i) % len(BACKENDS)] for i in range(len(BACKENDS))]

def pick(body: dict):
    msgs = body.get("messages", [])
    key = "".join(m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msgs)[:PREFIX_CHARS]
    mean = max(1.0, sum(inflight.values()) / len(BACKENDS))
    for b in ring(key):                    # preferred replica first, then spill
        if inflight[b] < min(MAX_PER_BACKEND, SLACK * mean + 1):
            return b
    return None                            # everyone full -> backpressure

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    b = pick(body)
    if b is None:
        return JSONResponse({"error": "overloaded"}, status_code=429, headers={"Retry-After": "2"})
    inflight[b] += 1
    upstream = client.build_request("POST", b + "/v1/chat/completions", json=body)
    r = await client.send(upstream, stream=True)
    async def relay():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose(); inflight[b] -= 1
    return StreamingResponse(relay(), status_code=r.status_code,
                             media_type=r.headers.get("content-type", "application/json"))
```

On kind, run the sim as a **StatefulSet** of 3 (stable names `llm-0`, `llm-1`, `llm-2` behind a headless Service) so the router can address pods individually. On a real cluster, discover endpoints from the Kubernetes API or use the Inference Extension instead of a static list.

## Lab: measure the cache win (on real engines)

Point the router at 2–3 real vLLM replicas (Colab can only host one, so use a cloud box with 2 small GPUs, or 3 `llama-server` processes on a Mac). Workload: 50 “users”, each with a 2,000-token system prompt of its own and 5 turns.

1. Round-robin: record TTFT p50/p99 and `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` per replica.
2. Prefix-affinity router: same numbers.
3. Make 5 users 10× more active than the rest (hot keys) and compare pure hashing vs bounded load.

- [ ] Router proxies streaming responses correctly (tokens arrive incrementally through it)
- [ ] Returns 429 + `Retry-After` when every backend is at its cap
- [ ] Table: policy → cache hit rate, TTFT p50/p99 (hit rate should jump with affinity)
- [ ] Showed the hot-key failure of pure hashing and that bounded load fixes it
      */}),
      resources: [
        { title: "Gateway API Inference Extension", url: "https://gateway-api-inference-extension.sigs.k8s.io", type: "docs", note: "the Kubernetes-standard model-aware routing layer (endpoint picker)" },
        { title: "Mooncake: a KVCache-centric architecture for serving LLM chatbot", url: "https://arxiv.org/abs/2407.00079", type: "paper", note: "cache-aware global scheduling in a large production system" },
        { title: "vLLM production-stack", url: "https://github.com/vllm-project/production-stack", type: "repo", note: "router implementations you can read in an afternoon" },
      ],
    },
    {
      id: "serving-features",
      title: "Multi-LoRA, structured outputs and tool calling in production",
      kind: "concept",
      minutes: 120,
      runsOn: ["cloud", "colab"],
      md: MD(function () {/*
## Multi-LoRA: one base model, hundreds of fine-tunes

A LoRA adapter is a few tens of MB of low-rank deltas on top of a base model (M22). Serving each fine-tune on its own GPU wastes 99% of the hardware; **multi-LoRA serving** keeps one copy of the base weights and applies each request’s adapter inside the same batch (S-LoRA and Punica introduced the batched kernels; vLLM, SGLang, TRT-LLM and LoRAX all do it).

::viz lora

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules support=./adapters/support sql=./adapters/sql \
  --max-loras 4 --max-lora-rank 16
# the adapter name is the "model" field:
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sql","messages":[{"role":"user","content":"orders by month"}]}'
```

Production notes:
- `--max-loras` = adapters active **in one batch** (GPU slots); others wait in CPU memory (`--max-cpu-loras`). Many distinct adapters in flight → lower throughput; route by adapter (previous lesson) to keep each replica’s set small.
- Adding adapters at runtime (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` + `POST /v1/load_lora_adapter`) is powerful but is a code-deployment path: guard it.
- Baseten (ch. 5) notes LoRA serving costs some throughput vs a merged model; if one adapter serves most of the traffic, **merge it** into its own deployment.

## Structured outputs: grammar-constrained decoding

Asking nicely for JSON fails a few percent of the time; at a million requests per day that is thousands of broken responses. **Constrained decoding** compiles a JSON Schema / regex / grammar into a state machine and, at every step, masks the logits of tokens that would violate it — so the output is valid by construction. **XGrammar** (default in vLLM and SGLang) precomputes most of the mask per grammar state so overhead is near zero; **llguidance** (guidance) is the other common backend.

```python
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

class Ticket(BaseModel):
    category: str
    priority: int
    summary: str

r = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Printer on floor 3 is on fire. Classify."}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "ticket", "schema": Ticket.model_json_schema()}},
)
print(Ticket.model_validate_json(r.choices[0].message.content))
# vLLM extras (v0.12+): extra_body={"structured_outputs": {"choice": ["low", "high"]}}  (also regex, grammar)
```

Gotchas: the first request with a new schema pays compile time (cache warms); constraining does not make the *content* correct, only the *shape*; very deep or recursive schemas can be slow; and forcing JSON before the model “thinks” can hurt reasoning models — let them reason, then constrain the final answer.

## Tool calling

The engine must parse the model’s native tool-call format into OpenAI `tool_calls`:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```

The parser is **model-specific** (hermes for Qwen2.5, `llama3_json`, `mistral`, …); the wrong parser silently returns tool calls as plain text. `tool_choice="required"` or a named function uses structured outputs underneath, guaranteeing a parseable call. Measure tool-call validity as an SLI (M19) — it regresses quietly when you swap models or quantize.

- [ ] Served two LoRA adapters from one base; showed both names in `/v1/models`
- [ ] Measured throughput with 1 vs 4 distinct adapters in the same load test
- [ ] 1,000 requests with `response_format` JSON Schema: 100% parse; same prompt without it: counted failures
- [ ] Tool-calling round trip with a wrong parser (observe failure) and the right one
      */}),
      resources: [
        { title: "vLLM — Structured outputs", url: "https://docs.vllm.ai/en/latest/features/structured_outputs.html", type: "docs", note: "response_format, structured_outputs extra body, backends" },
        { title: "XGrammar", url: "https://arxiv.org/abs/2411.15100", type: "paper", note: "how near-zero-overhead grammar masking works" },
        { title: "vLLM — LoRA adapters", url: "https://docs.vllm.ai/en/latest/features/lora.html", type: "docs", note: "flags, runtime loading, limits" },
        { title: "vLLM — Tool calling", url: "https://docs.vllm.ai/en/latest/features/tool_calling.html", type: "docs", note: "parser per model family" },
      ],
    },
    {
      id: "reliability",
      title: "Reliability, multi-cloud capacity and security basics",
      kind: "concept",
      minutes: 120,
      md: MD(function () {/*
## Failure modes are different for LLMs

Requests last seconds to minutes, stream, and cost wildly different amounts (10 tokens vs 10,000). Classic web defaults — 30 s timeouts, 3 blind retries, per-request rate limits — all misbehave.

| Mechanism | LLM-specific rule |
|---|---|
| **Timeouts** | Separate *connect* (1–2 s), *time-to-first-token* (e.g. 10–30 s) and *idle between chunks* (e.g. 30 s) instead of one total timeout; a 4,000-token answer at 30 tok/s legitimately takes 2+ minutes |
| **Retries** | Only retry **before the first token** was sent to the user (after that, you would duplicate output), only on retryable errors (connect failure, 429, 503), with **exponential backoff + jitter** and a retry *budget* (e.g. ≤ 10% extra load) to avoid retry storms |
| **Backpressure** | When every replica is at its concurrency cap, reject fast with `429` + `Retry-After` (or hold in a bounded queue with a deadline) — never let queue time silently exceed the SLO |
| **Rate limits** | Limit **tokens per minute** per tenant, not just requests; estimate input tokens up front and reconcile output tokens after (LiteLLM, Envoy AI Gateway do this) |
| **Priorities** | Interactive traffic ahead of batch; batch/async jobs soak up idle capacity (Baseten ch. 7.5: async inference + webhooks) |
| **Graceful draining** | On scale-down or deploy: pod leaves readiness → `preStop` sleep so load balancers stop sending → `SIGTERM` → engine finishes in-flight streams → exit before `terminationGracePeriodSeconds`. Test your engine version’s SIGTERM behavior; do not assume |
| **Load shedding** | Under overload, drop lowest-priority or longest requests first; a 30% error rate on batch beats a 100% timeout rate on everyone |

> [!WARNING] Streaming hides failures
> Once a `200 OK` header is sent, a mid-stream crash cannot change the status code. Clients must check that the final chunk has a `finish_reason`; your metrics must count truncated streams as errors.

## Deploying new versions safely

Baseten (ch. 7.4.1): a new engine version, quantization or checkpoint is a risky deploy. **Canary** sends 1–5% of traffic to the new version and ramps while comparing latency, error rate and output quality (pre-scale the canary so it does not cold-start under its first slice). **Blue-green** spins up a full second fleet and flips — clean rollback, but briefly **2× the GPUs**, which may not even exist in your region. Canary is the usual default for GPU fleets.

## Capacity is the hard part: multi-region, multi-cloud

The book’s chapter 7.3 thesis: at scale, the bottleneck is not software but **getting GPUs**. No single region of a single cloud reliably has the H100s/B200s you need, when you need them. Baseten’s architecture:

- One **global control plane** (deployments, routing, autoscaling decisions) and many **workload planes** (clusters in different clouds/regions/neoclouds) that run the models.
- A mix of **reserved** (cheap, committed), **on-demand** (flexible) and occasionally **spot** capacity; autoscaling spills across clusters.
- Latency geography: roughly **5 ms per time zone** of distance, New York ↔ San Francisco ≈ 15 ms — small next to a 1 s response, so routing to a farther region with free GPUs beats queueing locally.
- Hardware fails: roughly **one GPU failure per ~50,000 GPU-hours**. A 1,000-GPU fleet runs 1,000 GPU-hours per hour → expect a GPU failure every ~2 days. Plan for **active-active** redundancy across regions (or active-passive for cost), health checks that detect Xid errors (M19), and automatic node replacement.
- Multi-step pipelines (e.g. ASR → LLM → TTS) belong in **one cluster**: 5 hops at 10 ms vs 50 ms is 50 ms vs 250 ms — a quarter of a 1 s budget.

## Security basics for a model endpoint

- **Auth** on every endpoint (`vllm serve --api-key` for a start; a gateway with per-tenant keys in production). Never expose `/v1/load_lora_adapter`, sleep/wake or other admin routes publicly.
- **Supply chain**: load `safetensors`, avoid `--trust-remote-code` unless you have read the code, pin model *revisions*, scan images.
- **Data**: do not log prompts/outputs by default; Baseten’s default is to **not store inputs or outputs at all**. Customers will ask for SOC 2, HIPAA, and **data residency** (EU traffic stays on EU GPUs — which constrains your multi-region routing).
- **TLS** everywhere; terminating it close to users matters for latency (the book notes TLS setup can eat ≥10% of a 300 ms SLA without connection reuse).
- **Isolation**: separate namespaces/node pools per tenant for dedicated deployments; network policies so engine pods are reachable only from the router.

- [ ] Wrote a client with the three timeouts, first-token-only retries, backoff + jitter, and a retry budget
- [ ] Router/gateway rejects with 429 + `Retry-After` at saturation; load test shows bounded p99 instead of timeouts
- [ ] Rolled the kind Deployment while a load test ran; zero failed requests with preStop + grace period (and a nonzero count without)
- [ ] One-page capacity plan: regions, reserved vs on-demand split, failure math for your fleet size
      */}),
      resources: [
        { title: "AWS Builders’ Library — Timeouts, retries and backoff with jitter", url: "https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/", type: "article", note: "retry budgets and jitter, the canonical write-up" },
        { title: "Google SRE book — Handling overload", url: "https://sre.google/sre-book/handling-overload/", type: "book", note: "load shedding, criticality, client-side throttling" },
        { title: "Kubernetes — Container lifecycle hooks", url: "https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/", type: "docs", note: "preStop and termination ordering" },
        { title: "LiteLLM", url: "https://github.com/BerriAI/litellm", type: "tool", note: "OpenAI-compatible gateway: keys, TPM limits, fallbacks" },
      ],
    },
  ],

  challenge: {
    title: "A deployment repo that survives a load test",
    md: MD(function () {/*
Build `course-work/m18-deploy/`, a repository someone else could use to run your model in production:

```text
m18-deploy/
  Dockerfile, serve.sh            # pinned engine image, weights on a volume
  k8s/base/                       # Deployment (probes, preStop, grace), Service, PDB, PodMonitor
  k8s/overlays/kind-sim/          # llm-d-inference-sim, runs on a laptop
  k8s/overlays/gpu/               # real vLLM/SGLang: nvidia.com/gpu, /dev/shm, PVC, tolerations
  k8s/scaledobject.yaml           # KEDA on running+waiting per replica
  router/                         # your cache-aware router (or Inference Extension config)
  loadtest/loadtest.py            # staged ramp + spike, CSV output
  RUNBOOK.md
```

`RUNBOOK.md` must cover: how to deploy and roll back; the autoscaling policy and **why** each number (target concurrency from your sweep, windows from your cold-start measurement); what each alert means (queue growing, KV usage > 90%, preemptions, 5xx, pod stuck in startup) and the first three commands to run; how to drain a node; a capacity table (replicas → max sustained RPS at your SLO).

Final demonstration: run the load test (steady → 5× spike → steady) against kind-sim, and — if you have credits — once against the GPU overlay. Include the replicas/p99/error-rate timeline chart.
    */}),
    checklist: [
      "`kubectl apply -k k8s/overlays/kind-sim` brings up a working, autoscaled, OpenAI-compatible endpoint from scratch",
      "Probes are correct: no restart loop during a 60 s simulated cold start",
      "Load test with a 5× spike: autoscaler adds replicas, p99 recovers, error rate < 1% (429s with Retry-After count as handled)",
      "A rolling update during the load test drops zero streams",
      "Every autoscaling number in the runbook is justified by a measurement",
      "Runbook has an alert → diagnosis → action table",
    ],
    stretch: "Deploy the GPU overlay on a managed Kubernetes cluster with two GPU nodes, replace your router with the Gateway API Inference Extension (or llm-d), and compare TTFT p99 and prefix-cache hit rate against your own router under a multi-turn workload.",
  },

  connects: MD(function () {/*
You can now put an engine behind a real endpoint and keep it alive under load. M19 turns that into **numbers people trust**: throughput–latency curves, goodput under an SLO, Grafana dashboards of the very metrics you autoscaled on, and \$ per million tokens. M17’s disaggregation and M15’s speculative decoding plug in here as engine flags and routing policies; M23 reuses the same serving stack as the rollout engine for RL.
  */}),

  interview: [
    "Why is GPU utilization a poor autoscaling signal for LLM serving, and what would you scale on instead? How do you pick the target value?",
    "Walk through the cold start of a new replica for a 70B model. Where does the time go and how would you cut it by 10×?",
    "Design a router for a multi-turn chat product on 20 replicas. How do you balance prefix-cache locality against load?",
    "A pod keeps restarting while loading weights. What is the likely cause and fix?",
    "When is it safe to retry a failed LLM request, and how do you avoid retry storms?",
    "How do `max-num-seqs`, `gpu-memory-utilization` and `max-model-len` interact? What happens when you set max-num-seqs too high?",
    "Your provider has no H100s in your region for a traffic spike. What architecture would have prevented the outage?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — local PDF", url: "Inference%20Engineering.pdf", type: "book", note: "ch. 4 engines, ch. 7 production: autoscaling, cold starts, multi-cloud, canaries" },
    { title: "vLLM documentation", url: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html", type: "docs", note: "the OpenAI-compatible server you deploy all module" },
    { title: "SGLang documentation", url: "https://docs.sglang.ai", type: "docs", note: "the main alternative engine; server arguments and router" },
    { title: "NVIDIA Dynamo", url: "https://docs.nvidia.com/dynamo/latest/index.html", type: "docs", note: "KV-aware routing, disaggregation, planner" },
    { title: "Ray Serve LLM", url: "https://docs.ray.io/en/latest/serve/llm/index.html", type: "docs", note: "Python-native serving with autoscaling on ongoing requests" },
    { title: "KEDA — Scaling deployments", url: "https://keda.sh/docs/latest/concepts/scaling-deployments/", type: "docs", note: "ScaledObject concepts" },
    { title: "Baseten blog", url: "https://www.baseten.co/blog/", type: "article", note: "production write-ups on cold starts, autoscaling, multi-cloud" },
    { title: "Google SRE book — Monitoring distributed systems", url: "https://sre.google/sre-book/monitoring-distributed-systems/", type: "book", note: "the four golden signals you will alert on" },
  ],
});
