# Inference Engineering — a build-first curriculum

A static, offline website. **Open `index.html` by double-clicking** (no server, no build).

**What's inside:** 26 modules in 9 phases (≈230 lessons, ≈450–500 h, ~42 weeks at 12 h/week), 65 interactive visualizations, a just-in-time Math track, a Projects view (every build/lab + module challenge), a job-market Skills map (2026 postings), a 177-term glossary, and a de-duplicated resource library. Progress checkboxes persist in your browser.

```
AI Course/
├── index.html                  ← open this
├── Inference Engineering.pdf   ← Baseten book (reference text)
├── content/
│   ├── course.js               ← THE MAP: phases, module order, final goal  (edit to restructure)
│   ├── modules/<id>.js         ← one file per module (lessons, challenge, resources)
│   ├── modules/_template.js    ← copy to create a new module
│   ├── skills.js               ← job-market skills map view
│   └── glossary.js             ← glossary view
├── assets/
│   ├── css/style.css
│   ├── js/core.js              ← registry + markdown/math renderer
│   ├── js/app.js               ← views, routing, progress, search
│   ├── js/viz/_kit.js          ← tiny chart/control toolkit for visualizations
│   ├── js/viz/*.js             ← interactive explainers (registered by name)
│   └── vendor/                 ← marked, KaTeX, highlight.js (local copies → works offline)
└── tools/validate.js           ← `node tools/validate.js` checks all content
```

## Changing the curriculum

| Change | How |
|---|---|
| Reorder / move / remove modules | edit `phases[].modules` in `content/course.js` |
| Add a module | copy `content/modules/_template.js` → `content/modules/<id>.js`, set `id`, add id to a phase |
| Edit a lesson | edit its `md: MD(function(){/* markdown */})` block |
| Add a visualization | `Viz.register("name", {...})` in any `assets/js/viz/*.js` (list new files in `course.js → vizFiles`), then write `::viz name {"param":1}` on its own line in markdown |
| Validate | `node tools/validate.js` (or `node tools/validate.js m03-build-gpt`) |

Progress lives in browser `localStorage` (Guide tab → export/import).

## Authoring rules (markdown inside `MD(function(){/* … */})`)

- Everything inside the comment is literal: backticks, `${}`, LaTeX backslashes are all fine. **Never write the two characters `*` `/` adjacent** (it ends the comment) — in C/CUDA code use `//` comments; write multiplication-then-division with spaces (`a * / b` never occurs naturally, but `/* */` does).
- Math: inline `$...$`, display `$$...$$` (KaTeX). A literal dollar sign must be written `\$` (e.g. `\$2.50/hr`).
- Callouts: `> [!TIP]`, `[!NOTE]`, `[!WARNING]`, `[!IMPORTANT]`, `[!MATH]`, `[!PREREQ]`, `[!INTUITION]`, `[!REAL]`, `[!WHY]`, `[!GOAL]`, `[!BUILD]`, `[!CHECK]` — optional custom title after the tag: `> [!TIP] My title`.
- `- [ ] task` → persistent checkbox (exercises).
- `<details><summary>Title</summary>` … `</details>` → collapsible aside (blank line after `<summary>` line and before `</details>`).
- Code fences: ```` ```python ````, ```` ```bash ````, ```` ```cpp ```` (use cpp for CUDA), ```` ```text ````.
- Lesson kinds: `demo` (see it working) · `concept` · `math` · `build` · `lab` · `read` · `deep` (optional deep dive).
- Hardware tags (`runsOn`): `mac` · `colab` · `cloud` · `multi-gpu` · `browser` · `any`.
- Resource types: `video` · `paper` · `repo` · `article` · `book` · `course` · `docs` · `tool` · `practice`.

## Visualization catalog

Use in markdown as `::viz <name>` or `::viz <name> {"json":"params"}`. All params are optional; each viz reads its supported params at the top of its `render()` function in `assets/js/viz/<file>.js` (e.g. `{"model":"llama31-8b","gpu":"H100"}`). A module's `demo` viz is shown in the goal box unless the goal markdown already embeds the same viz.

### math.js
| name | what it shows | params |
|---|---|---|
| `softmax-temp` | logits → softmax probs with temperature, top-k, top-p, sampling counts | `tokens`, `logits` |
| `derivative` | function + tangent line + finite-difference secant | `fn` ("x²","sin(x)","eˣ","sigmoid(x)","ReLU(x)","x³ − 3x") |
| `gradient-descent` | 1-D loss curve with two minima; learning-rate, step/run | `lr`, `x0` |
| `vectors-dot` | draggable 2-D vectors; dot product, cosine similarity, projection | — |
| `matmul` | matrix multiply cell-by-cell, shape rule, 2mnk FLOPs | `m`,`k`,`n` |
| `activations` | ReLU/GELU/SiLU/Sigmoid/Tanh plots | — |
| `neg-log` | −log p curve = cross-entropy; ln(V) init loss; perplexity | `vocab` |
| `backprop-graph` | micrograd-style graph, forward values, step-through backward | — |
| `normalize` | LayerNorm vs RMSNorm on a vector | — |
| `exp-log` | exp and ln as inverses; log rules | — |
| `percentiles` | lognormal latency histogram with p50/mean/p90/p99 | — |

### llm.js
| name | what it shows |
|---|---|
| `bigram-live` | char-level bigram model trained on editable text in-browser: count heatmap, NLL loss, sampled text |
| `tokenizer-bpe` | BPE merges step-by-step on editable text; tokens colored; vocab/sequence length trade-off |
| `embeddings-2d` | toy word embeddings in 2-D; nearest neighbours; analogy arithmetic (king − man + woman) |
| `causal-average` | Karpathy's "matmul trick": lower-triangular weight matrix averaging past tokens → softmax weights |
| `attention` | tokens of a sentence; Q·Kᵀ scores, ÷√d, causal mask toggle, softmax heatmap; hover a query to see arcs; multiple heads |
| `transformer-block` | clickable GPT block diagram with tensor shapes (B,T,C) at each stage |
| `param-counter` | model config sliders/presets (GPT-2, Llama-3-8B, Qwen2.5-7B, Llama-3-70B) → parameter & memory breakdown |
| `rope` | rotary position embedding: rotating 2-D pairs by position; dot product depends on relative distance |
| `gqa` | MHA vs GQA vs MQA (vs MLA) head sharing diagram + KV-cache size per token |
| `moe-routing` | tokens → router → top-k experts; load imbalance; active vs total params |
| `overfitting` | train vs val loss curves as model/data size change; early stopping |
| `scaling-laws` | C ≈ 6·N·D calculator, Chinchilla-optimal tokens, training days for N GPUs at given MFU |
| `training-memory` | memory for weights, grads, Adam states, activations in fp32/bf16/mixed; vs GPU sizes |
| `lora` | W vs W + BA: parameter counts vs rank r; which weights are trainable |

### inference.js
| name | what it shows |
|---|---|
| `generation-loop` | autoregressive token-by-token generation animation; compute per step with vs without KV cache |
| `inference-stack` | layered diagram client → gateway → router → engine (scheduler, KV mgr) → runtime → kernels → GPU; click for details |
| `prefill-decode` | request timeline: queue, prefill (TTFT), decode steps (TPOT); prompt/output length sliders; compute- vs memory-bound bars |
| `kv-calculator` | model presets + dtype + context + batch → KV bytes/token, per sequence, GPU memory breakdown, max concurrency |
| `paged-attention` | sequences growing, blocks allocated from a free pool, block tables, fragmentation vs contiguous; prefix sharing |
| `prefix-cache` | radix tree of prompts sharing prefixes; cache hit rate; tokens saved |
| `batching-sim` | Gantt of requests under static vs continuous batching; utilization, throughput, latency |
| `chunked-prefill` | decode stalls when a long prefill arrives vs chunked prefill with a token budget |
| `latency-throughput` | batch size vs per-user tok/s and total tok/s (roofline-based); SLO line |

### gpu.js
| name | what it shows |
|---|---|
| `roofline` | log-log roofline for GPU presets (T4, A100, H100, B200, M-series); operations as points; arithmetic intensity slider |
| `memory-hierarchy` | registers / SMEM / L2 / HBM / NVLink / PCIe / network: bandwidth, latency, capacity (log bars) |
| `gpu-anatomy` | clickable GPU → GPCs → SMs → warps / tensor cores / SMEM diagram with H100 numbers |
| `cuda-grid` | map an array to grid/blocks/threads; hover shows blockIdx, threadIdx, global index formula |
| `coalescing` | a warp's 32 threads accessing memory with stride → number of memory transactions |
| `tiled-matmul` | tiled matmul: global-memory loads with/without shared-memory tiling vs tile size |
| `online-softmax` | step through online softmax over blocks: running max, running sum, rescaling |
| `kernel-fusion` | unfused vs fused ops: memory traffic and time bars |
| `launch-overhead` | CPU launch gaps between small kernels vs CUDA graph replay timeline |

### optimize.js
| name | what it shows |
|---|---|
| `number-formats` | bit layouts FP32/BF16/FP16/FP8(E4M3,E5M2)/INT8/INT4/FP4(E2M1); encode a value; range & precision |
| `quantize-weights` | weight histogram with outliers; quantize with bits, per-tensor / per-channel / per-group; error |
| `spec-decode` | acceptance rate α, draft length k, draft cost → expected tokens/step & speedup; draft/verify animation |

### distributed.js
| name | what it shows |
|---|---|
| `parallelism` | DP / TP / PP / EP (and CP) across N GPUs: what lives where, what is communicated |
| `tp-split` | column-parallel then row-parallel matmul split across GPUs and the all-reduce |
| `ring-allreduce` | animated ring all-reduce; cost 2(N−1)/N · size / bandwidth |
| `pipeline-bubble` | pipeline schedule with micro-batches; bubble fraction (p−1)/(m+p−1) |
| `comm-cost` | α–β model: message size vs time over NVLink / PCIe / InfiniBand / Ethernet |
| `disaggregation` | colocated vs disaggregated prefill/decode timelines; ITL interference; KV transfer |

### production.js
| name | what it shows |
|---|---|
| `queueing` | arrival rate vs capacity → latency blow-up near 100% utilization; Little's law |
| `autoscaling-sim` | daily traffic, replicas scaling with cold-start delay; SLO violations vs cost |
| `cost-calc` | GPU \$/hr, throughput, utilization → \$/1M tokens; compare with API price |
| `load-balancing` | random / round-robin / least-loaded / KV-cache-aware routing: latency & cache hit rate |

### rl.js
| name | what it shows |
|---|---|
| `gridworld` | Q-learning agent training live: values heatmap, policy arrows, reward curve; ε, α, γ sliders |
| `bandit` | multi-armed bandit: ε-greedy vs greedy vs UCB; regret |
| `discount` | γ → weight of future rewards; effective horizon 1/(1−γ) |
| `policy-gradient` | REINFORCE on a 3-action softmax policy; baseline toggle shows variance reduction |
| `ppo-clip` | PPO clipped objective vs probability ratio for positive/negative advantage |
| `grpo-group` | a group of sampled answers with rewards → group-normalized advantages; KL penalty |
| `rl-systems-timeline` | synchronous vs asynchronous RL: rollout (inference) / train / weight-sync timelines, long-tail idle time |

### continual.js
| name | what it shows |
|---|---|
| `forgetting` | train a small classifier on task A then task B in-browser; accuracy on A collapses; naive vs replay vs EWC |
| `accuracy-matrix` | tasks × evaluation-after-each-task heatmap for naive / replay / EWC / adapter-per-task |
