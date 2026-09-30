/* course.js — THE curriculum map.
 *
 * To change the curriculum:
 *   • Reorder / move modules  → edit the `modules` arrays below.
 *   • Add a module            → create content/modules/<id>.js (copy content/modules/_template.js)
 *                               and add its id to a phase here.
 *   • Remove a module         → delete its id here (the file can stay).
 *   • Add a visualization     → create assets/js/viz/<file>.js and add <file> to `vizFiles`.
 * No build step: refresh index.html.
 */
window.COURSE_CONFIG = {
  title: "Inference Engineering",
  subtitle: "A build-first path from “I’ve never used PyTorch” to serving LLMs at scale — with the math, GPUs, RL and continual learning you need along the way.",
  version: "2026.09",
  hoursPerWeek: 12,

  // Visualization bundles loaded from assets/js/viz/<name>.js
  vizFiles: ["math", "llm", "inference", "gpu", "optimize", "distributed", "production", "rl", "continual"],

  // Extra data files loaded from content/<name>.js
  extraFiles: ["skills", "glossary"],

  finalGoal: MD(function () {/*
By the end you will have **built**, not just read about, the whole stack:

1. **A GPT from scratch** that writes Shakespeare — then a **Llama/Qwen-style model** you load real weights into and match Hugging Face logits.
2. **Your own mini-vLLM**: KV cache, paged block manager, prefix cache, continuous batching scheduler, sampler, streaming OpenAI-compatible API.
3. **GPU kernels** in CUDA and Triton — a tiled matmul, fused softmax, and a FlashAttention-style kernel — profiled with Nsight.
4. **An optimized engine**: INT8/INT4 quantization and speculative decoding with measured speed/quality trade-offs.
5. **A distributed, production deployment**: tensor-parallel serving on multiple GPUs, autoscaling, dashboards, a benchmark report with $/1M tokens.
6. **A GRPO-trained reasoning model** and a mini **actor–learner RL system** that uses an inference engine for rollouts.
7. **A continual-learning experiment** that measures (and fixes) catastrophic forgetting.

Every module starts with the finished thing working, then takes it apart.
  */}),

  howToLearn: MD(function () {/*
- **Top-down loop:** *See it work → poke it → break it into parts → rebuild each part → finish with a challenge.*
- **Math is just-in-time.** Purple **Math** lessons appear exactly when a concept needs them, each with a tiny prerequisite refresher. Use the **Math track** tab to review them all in order.
- **Hardware:** 🍎 = runs on your Mac (PyTorch MPS / MLX). ☁️ = needs an NVIDIA GPU (free Colab T4 is enough unless stated; a few labs rent an H100 for ~1–2 hours).
- **Checkboxes persist** in your browser, so the progress bars are real.
- **Pace:** ~10–15 h/week → roughly 9–10 months end to end (week ranges on the roadmap are computed from the lesson estimates at 12 h/week — expect capstones to run long). Optional deep-dives are marked and can be skipped on a first pass.
  */}),

  phases: [
    {
      id: "p0", title: "Orientation", weeks: "Week 1",
      blurb: "Run a real LLM on your laptop, measure it, and get the map of everything that sits between a prompt and a GPU.",
      modules: ["m00-the-map"],
    },
    {
      id: "p1", title: "Build an LLM from scratch", weeks: "Weeks 2–10",
      blurb: "Neural nets → language models → GPT → modern Llama-style architectures → training & fine-tuning. You can’t optimize what you don’t understand.",
      modules: ["m01-neural-nets", "m02-language-models", "m03-build-gpt", "m04-modern-llms", "m05-training-finetuning"],
    },
    {
      id: "p2", title: "Inference fundamentals", weeks: "Weeks 11–18",
      blurb: "Prefill vs decode, the roofline, KV cache paging, continuous batching — then assemble them into your own inference engine.",
      modules: ["m06-inference-anatomy", "m07-hardware-roofline", "m08-kv-cache-paging", "m09-batching-scheduling", "m10-mini-engine"],
    },
    {
      id: "p3", title: "GPU programming & performance", weeks: "Weeks 19–23",
      blurb: "Write CUDA and Triton kernels, derive FlashAttention, and learn to profile like a performance engineer.",
      modules: ["m11-cuda-basics", "m12-triton-flash", "m13-profiling-compile"],
    },
    {
      id: "p4", title: "Optimization techniques", weeks: "Weeks 24–26",
      blurb: "Quantization and speculative decoding: the two biggest levers after batching.",
      modules: ["m14-quantization", "m15-speculative-decoding"],
    },
    {
      id: "p5", title: "Distributed & large-scale inference", weeks: "Weeks 27–29",
      blurb: "Tensor / pipeline / expert parallelism, collectives, disaggregated prefill-decode, MoE serving.",
      modules: ["m16-parallelism", "m17-disagg-moe"],
    },
    {
      id: "p6", title: "Production inference", weeks: "Weeks 30–34",
      blurb: "Engines, containers, autoscaling, routing, benchmarking, observability and cost — the job most inference teams actually do.",
      modules: ["m18-serving-production", "m19-benchmark-observe", "m20-modalities"],
    },
    {
      id: "p7", title: "Reinforcement learning", weeks: "Weeks 35–38",
      blurb: "From a gridworld agent to GRPO reasoning models to the distributed rollout systems that train frontier models.",
      modules: ["m21-rl-fundamentals", "m22-rl-for-llms", "m23-distributed-rl"],
    },
    {
      id: "p8", title: "Continual learning & career", weeks: "Weeks 39–42",
      blurb: "Models that keep learning without forgetting — and turning everything you built into a job-ready portfolio.",
      modules: ["m24-continual-learning", "m25-career-capstone"],
    },
  ],
};
