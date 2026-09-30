/* skills.js — the job-market skills map (route: #/x/skills).
 *
 * Edit the data arrays below; the render function at the bottom turns them into the page.
 * Source: ~70 live postings from ~22 companies (2025–2026) — see m25-career-capstone.
 *   demand:     "must" | "common" | "nice"
 *   archetypes: letters from ARCHETYPES (A–E)
 *   modules:    module ids where the skill is taught (links + progress are computed)
 */
(function () {
  "use strict";

  var ARCHETYPES = [
    {
      key: "A", name: "Kernel / GPU performance",
      does: "Writes and tunes CUDA / Triton / CUTLASS kernels for GEMMs, attention, MoE routing and quantized ops. Reads PTX/SASS, reasons about occupancy, coalescing and tensor cores, and chases 10–30% wins on the hot path with Nsight.",
      modules: ["m07-hardware-roofline", "m11-cuda-basics", "m12-triton-flash", "m13-profiling-compile", "m14-quantization"],
      postings: [["Baseten – GPU Kernels", "https://jobs.ashbyhq.com/baseten/ddb5bc98-6116-49a2-802e-1c05398663f1"], ["Anthropic – Performance Engineer, GPU", "https://job-boards.greenhouse.io/anthropic/jobs/4926227008"]],
    },
    {
      key: "B", name: "Serving runtime / inference systems",
      does: "Builds the engine itself: schedulers, continuous batching, paged KV cache, prefix caching, speculative decoding, disaggregated prefill/decode, multi-LoRA — usually inside or next to vLLM, SGLang, TensorRT-LLM or Dynamo.",
      modules: ["m06-inference-anatomy", "m08-kv-cache-paging", "m09-batching-scheduling", "m10-mini-engine", "m15-speculative-decoding", "m16-parallelism", "m17-disagg-moe"],
      postings: [["Perplexity – MTS AI Inference", "https://jobs.ashbyhq.com/perplexity/8a976851-9bef-4b07-8d36-567fa9540aef"], ["Modal – MTS Research, Inference", "https://jobs.ashbyhq.com/modal/73c97bbc-8e27-4c5d-b38b-90b3afdb0d93"]],
    },
    {
      key: "C", name: "Inference platform / infra",
      does: "Owns the fleet: Kubernetes control planes, GPU provisioning, autoscaling, KV-aware routing, observability, RDMA networking, model storage and cold starts, SLOs and on-call — often agnostic to which engine runs underneath.",
      modules: ["m16-parallelism", "m17-disagg-moe", "m18-serving-production", "m19-benchmark-observe"],
      postings: [["Together AI – Compute Infra", "https://job-boards.greenhouse.io/togetherai/jobs/5213325007"], ["Anyscale – Distributed LLM Inference", "https://jobs.ashbyhq.com/anyscale/1cf38233-8aa0-47f8-9d85-65ce27bc3047"]],
    },
    {
      key: "D", name: "RL / post-training infra",
      does: "Bridges training and serving: rollout generation with inference engines, trainer→rollout weight sync, reward and eval pipelines, sandboxed agent environments, and engine features tuned for RL workloads.",
      modules: ["m05-training-finetuning", "m21-rl-fundamentals", "m22-rl-for-llms", "m23-distributed-rl", "m24-continual-learning"],
      postings: [["Together AI – Post-Training Inference", "https://job-boards.greenhouse.io/togetherai/jobs/5179372007"], ["xAI – MTS, RL Inference", "https://job-boards.greenhouse.io/xai/jobs/5180223007"]],
    },
    {
      key: "E", name: "Custom silicon",
      does: "Any of A–D, but targeting a non-CUDA stack: TPU (Pallas/XLA), Trainium (Neuron), Cerebras CSL, AMD ROCm/HIP. Same roofline thinking, different ISA and compiler.",
      modules: ["m07-hardware-roofline", "m11-cuda-basics", "m12-triton-flash", "m13-profiling-compile"],
      postings: [["Cerebras – Staff Kernel Optimization", "https://jobs.ashbyhq.com/cerebras/270407eb-1452-4a36-8f8c-7d64263eaa0a"], ["Anthropic – TPU Kernel Engineer", "https://job-boards.greenhouse.io/anthropic/jobs/4720576008"]],
    },
  ];

  var SKILLS = [
    { skill: "PyTorch internals, torch.compile, custom ops", demand: "must", arch: "ABDE", modules: ["m01-neural-nets", "m03-build-gpt", "m13-profiling-compile"] },
    { skill: "CUDA / Triton / CUTLASS kernel programming", demand: "must", arch: "AE", modules: ["m11-cuda-basics", "m12-triton-flash"] },
    { skill: "vLLM / SGLang / TensorRT-LLM internals", demand: "must", note: "serving roles", arch: "BCD", modules: ["m06-inference-anatomy", "m10-mini-engine", "m18-serving-production"] },
    { skill: "KV cache, paged & prefix attention", demand: "must", arch: "ABC", modules: ["m08-kv-cache-paging", "m10-mini-engine"] },
    { skill: "Continuous batching & request scheduling", demand: "must", arch: "BC", modules: ["m09-batching-scheduling", "m10-mini-engine"] },
    { skill: "Quantization (FP8 / FP4 / INT8 / AWQ / GPTQ)", demand: "must", arch: "AB", modules: ["m14-quantization"] },
    { skill: "Distributed inference (TP / PP / EP)", demand: "must", arch: "BCD", modules: ["m16-parallelism", "m17-disagg-moe"] },
    { skill: "NCCL / RDMA / InfiniBand / NVLink", demand: "must", note: "systems roles", arch: "CD", modules: ["m16-parallelism", "m23-distributed-rl"] },
    { skill: "Kubernetes, autoscaling, Ray", demand: "must", note: "platform roles", arch: "CD", modules: ["m18-serving-production", "m23-distributed-rl"] },
    { skill: "Transformer & LLM architecture fundamentals", demand: "must", note: "assumed everywhere", arch: "ABCDE", modules: ["m02-language-models", "m03-build-gpt", "m04-modern-llms"] },
    { skill: "Roofline / memory- vs compute-bound reasoning", demand: "must", note: "assumed everywhere", arch: "ABCE", modules: ["m07-hardware-roofline", "m06-inference-anatomy"] },
    { skill: "Speculative decoding", demand: "common", arch: "B", modules: ["m15-speculative-decoding"] },
    { skill: "Disaggregated prefill / decode", demand: "common", growing: true, arch: "BC", modules: ["m17-disagg-moe"] },
    { skill: "MoE routing & expert-parallel serving", demand: "common", growing: true, arch: "AB", modules: ["m04-modern-llms", "m17-disagg-moe"] },
    { skill: "Profiling (Nsight Systems / Compute, CUPTI)", demand: "common", arch: "ABE", modules: ["m13-profiling-compile", "m11-cuda-basics"] },
    { skill: "Benchmarking & observability (TTFT, TPOT, goodput)", demand: "common", arch: "BC", modules: ["m06-inference-anatomy", "m19-benchmark-observe"] },
    { skill: "Rust / C++ / Go systems programming", demand: "common", growing: true, arch: "ABC", modules: ["m10-mini-engine", "m11-cuda-basics"] },
    { skill: "RL / post-training infrastructure", demand: "common", growing: true, arch: "D", modules: ["m21-rl-fundamentals", "m22-rl-for-llms", "m23-distributed-rl"] },
    { skill: "Model storage, caching & cold starts", demand: "common", arch: "C", modules: ["m18-serving-production"] },
    { skill: "Multi-LoRA serving & fine-tuning", demand: "common", arch: "BD", modules: ["m05-training-finetuning", "m18-serving-production", "m24-continual-learning"] },
    { skill: "Multimodal serving (vision, audio, diffusion)", demand: "nice", arch: "B", modules: ["m20-modalities"] },
    { skill: "Continual learning & safe weight updates", demand: "nice", arch: "D", modules: ["m24-continual-learning"] },
    { skill: "Custom silicon / ISA (TPU, Trainium, Cerebras CSL, ROCm)", demand: "nice", note: "must-have at those shops", arch: "E", modules: ["m07-hardware-roofline", "m12-triton-flash"] },
  ];

  var TRENDS = [
    { t: "Disaggregated prefill/decode goes mainstream", d: "NVIDIA Dynamo and llm-d sit above vLLM/SGLang and add KV-aware routing and prefill/decode split.", modules: ["m17-disagg-moe"], url: "https://github.com/ai-dynamo/dynamo" },
    { t: "llm-d as the Kubernetes-native standard", d: "Prefix-cache-aware routing, hierarchical KV offload and wide-EP reference setups, backed by Red Hat, Google, IBM, CoreWeave and NVIDIA.", modules: ["m18-serving-production"], url: "https://github.com/llm-d/llm-d" },
    { t: "MoE expert-parallel serving is its own specialty", d: "Serving at DeepSeek scale combines PD disaggregation with large expert parallelism.", modules: ["m17-disagg-moe", "m16-parallelism"], url: "https://lmsys.org/blog/2025-05-05-large-scale-ep/" },
    { t: "FP4 / NVFP4 on Blackwell", d: "Micro-block scaled 4-bit formats are becoming the default quantization skill for new hardware.", modules: ["m14-quantization"], url: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/" },
    { t: "RL infra as an inference discipline", d: "Rollout engines, trainer→rollout weight sync (e.g. RDMA P2P transfer) and RL-aware engine features.", modules: ["m22-rl-for-llms", "m23-distributed-rl"], url: "https://github.com/volcengine/verl" },
    { t: "Long-context and agents put the KV cache at the centre", d: "Hierarchical GPU→CPU→disk KV offload and cache-aware routing decouple context length from GPU memory.", modules: ["m08-kv-cache-paging", "m17-disagg-moe"], url: "https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale" },
    { t: "Beyond CUDA", d: "TPU, Trainium, AMD ROCm and wafer-scale stacks are hiring for the same skills on different ISAs.", modules: ["m07-hardware-roofline", "m12-triton-flash"], url: "https://github.com/kvcache-ai/Mooncake" },
  ];

  var INTERVIEW = [
    { area: "System design", items: [
      ["Design an LLM serving system for X QPS with TTFT/TPOT SLOs", ["m25-career-capstone", "m18-serving-production", "m09-batching-scheduling"]],
      ["Design a KV-cache-aware router / multi-tenant platform", ["m17-disagg-moe", "m18-serving-production", "m08-kv-cache-paging"]],
      ["Design disaggregated prefill/decode and justify the trade-offs", ["m17-disagg-moe"]],
      ["Design RL rollout infrastructure with weight sync", ["m23-distributed-rl", "m22-rl-for-llms"]],
      ["Walk me through PagedAttention", ["m08-kv-cache-paging"]],
    ] },
    { area: "Napkin math", items: [
      ["KV-cache size for model × context × batch × precision", ["m08-kv-cache-paging", "m25-career-capstone"]],
      ["Upper bound on decode tok/s from memory bandwidth", ["m07-hardware-roofline", "m06-inference-anatomy"]],
      ["GPUs needed and $/1M tokens for a traffic target", ["m19-benchmark-observe", "m25-career-capstone"]],
      ["GPTQ vs AWQ vs FP8 — which would you deploy?", ["m14-quantization"]],
    ] },
    { area: "Coding", items: [
      ["Write / optimize a softmax or attention kernel (CUDA or Triton)", ["m11-cuda-basics", "m12-triton-flash"]],
      ["Implement a top-p / top-k sampler", ["m02-language-models", "m25-career-capstone"]],
      ["Implement a block manager or LRU prefix cache", ["m08-kv-cache-paging", "m10-mini-engine"]],
      ["Transformer internals from memory (attention, RoPE, GQA)", ["m03-build-gpt", "m04-modern-llms"]],
    ] },
    { area: "Debugging", items: [
      ["p99 latency spiked after a deploy — what do you check?", ["m19-benchmark-observe", "m25-career-capstone"]],
      ["OOM under load / preemption storms", ["m08-kv-cache-paging", "m09-batching-scheduling"]],
      ["Low GPU utilization: memory-util vs SM-util", ["m13-profiling-compile", "m07-hardware-roofline"]],
      ["NCCL job hangs — how do you debug it?", ["m16-parallelism"]],
    ] },
  ];

  // Fallback names when a module file isn't present yet.
  var NAMES = {
    "m00-the-map": "The map", "m01-neural-nets": "Neural nets", "m02-language-models": "Language models", "m03-build-gpt": "Build a GPT",
    "m04-modern-llms": "Modern LLMs", "m05-training-finetuning": "Training & fine-tuning", "m06-inference-anatomy": "Inference anatomy",
    "m07-hardware-roofline": "Hardware & roofline", "m08-kv-cache-paging": "KV cache & paging", "m09-batching-scheduling": "Batching & scheduling",
    "m10-mini-engine": "Mini engine", "m11-cuda-basics": "CUDA", "m12-triton-flash": "Triton & Flash", "m13-profiling-compile": "Profiling & compile",
    "m14-quantization": "Quantization", "m15-speculative-decoding": "Speculative decoding", "m16-parallelism": "Parallelism",
    "m17-disagg-moe": "Disaggregation & MoE", "m18-serving-production": "Serving in production", "m19-benchmark-observe": "Benchmark & observe",
    "m20-modalities": "Modalities", "m21-rl-fundamentals": "RL fundamentals", "m22-rl-for-llms": "RL for LLMs", "m23-distributed-rl": "Distributed RL",
    "m24-continual-learning": "Continual learning", "m25-career-capstone": "Career",
  };

  var DEMAND = {
    must: { label: "Must-have", style: { background: "var(--green-bg)", color: "var(--green)", border: "1px solid color-mix(in srgb, var(--green) 35%, transparent)" } },
    common: { label: "Common", style: { background: "var(--amber-bg)", color: "var(--amber)", border: "1px solid color-mix(in srgb, var(--amber) 35%, transparent)" } },
    nice: { label: "Nice-to-have", cls: "pill-opt" },
  };

  Course.extra("skills", {
    eyebrow: "Job market",
    title: "Skills map",
    moduleNames: NAMES,
    lead: "What inference-engineering employers actually ask for in 2026, and where in this course you learn each skill. Built from ~70 live postings at ~22 companies.",
    render: function (ctx) {
      var h = ctx.h;

      function modName(id) { var m = ctx.mod(id); return m ? (m.short || m.title) : (NAMES[id] || id); }
      function modProgress(id) {
        var m = ctx.mod(id);
        if (!m) return null;
        var total = m.lessons.length + (m.challenge ? 1 : 0);
        if (!total) return null;
        var done = m.lessons.filter(function (l) { return ctx.store.get("done:" + (l.key || m.id + "/" + l.id)); }).length +
          (m.challenge && ctx.store.get("done:" + m.id + "/challenge") ? 1 : 0);
        return done / total;
      }
      function modLink(id) {
        var num = ctx.ORDER.indexOf(id) >= 0 ? ctx.moduleNum(id) : "··";
        return h("a", { href: "#/m/" + id, style: { whiteSpace: "nowrap", display: "inline-block", marginRight: "10px" } },
          h("span", { class: "mod-num sm", text: num }), modName(id));
      }
      function modLinks(ids) { return ids.map(modLink); }
      function pill(demand) {
        var d = DEMAND[demand];
        return h("span", { class: "pill " + (d.cls || ""), style: d.style || null, text: d.label });
      }
      function section(title, sub) {
        return h("h2", { style: { fontSize: "22px", letterSpacing: "-.02em", margin: "34px 0 6px" } }, title,
          sub ? h("span", { class: "muted", style: { fontSize: "14px", fontWeight: "400", marginLeft: "10px" }, text: sub }) : null);
      }
      function progressCell(ids) {
        var vals = ids.map(modProgress).filter(function (v) { return v !== null; });
        if (!vals.length) return h("span", { class: "muted small", text: "—" });
        var p = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
        return h("div", { style: { minWidth: "90px" } },
          h("div", { class: "bar" }, h("i", { style: { width: Math.round(p * 100) + "%" } })),
          h("span", { class: "muted small", text: Math.round(p * 100) + "%" }));
      }

      var out = [];

      // (a) intro
      out.push(ctx.md("div", "prose", [
        "> **Snapshot: 2026.** Sourced from postings at Baseten, Together AI, Fireworks, Modal, Anyscale, Anthropic, OpenAI, NVIDIA, Meta, Google DeepMind, Mistral, Perplexity, Character.ai, Cerebras, Databricks, CoreWeave, Cohere, xAI, Lambda, Hugging Face, Red Hat and Scale AI. The job titles vary a lot, but the work falls into **five archetypes**. **Must-have** means it was a hard requirement in most relevant postings.",
        "",
        "Use this page to pick an archetype, then look at the **Must-have** rows for it. The progress column fills in as you tick lessons off. For interview prep and a 90-day plan, see [Career: become a hireable inference engineer](#/m/m25-career-capstone).",
      ].join("\n")));

      // (b) archetypes
      out.push(section("Role archetypes", "what the job is day to day"));
      out.push(h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(270px, 1fr))", gap: "12px", margin: "10px 0" } },
        ARCHETYPES.map(function (a) {
          return h("div", { class: "mod-card", style: { margin: "0", padding: "14px 16px" } },
            h("div", { class: "mod-title", style: { fontSize: "16px" } }, h("span", { class: "mod-num", text: a.key }), a.name),
            h("p", { style: { fontSize: "14px", color: "var(--fg2)", margin: "8px 0" }, text: a.does }),
            h("div", { class: "sub-h", text: "Modules that matter most" }),
            h("div", { style: { fontSize: "13.5px", lineHeight: "1.9" } }, modLinks(a.modules)),
            h("div", { class: "sub-h", text: "Example postings" }),
            h("ul", { style: { margin: "0", paddingLeft: "18px", fontSize: "13px" } },
              a.postings.map(function (p) { return h("li", null, h("a", { href: p[1], target: "_blank", rel: "noopener", text: p[0] })); })));
        })));

      // (c) skills matrix
      out.push(section("Skills matrix", SKILLS.length + " skills"));
      var legend = h("div", { class: "muted small", style: { display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap", margin: "4px 0 8px" } },
        pill("must"), "hard requirement in most relevant postings", pill("common"), "frequent / strong differentiator", pill("nice"), "occasional bonus", h("span", { text: "↑ = growing fast" }));
      out.push(legend);
      var rows = SKILLS.map(function (s) {
        return h("tr", null,
          h("td", null, h("b", { text: s.skill }), s.note ? h("div", { class: "muted small", text: s.note }) : null),
          h("td", { style: { whiteSpace: "nowrap" } }, pill(s.demand), s.growing ? h("span", { title: "growing fast", style: { color: "var(--green)", fontWeight: "700", marginLeft: "4px" }, text: "↑" }) : null),
          h("td", { style: { fontFamily: "var(--mono)", whiteSpace: "nowrap" }, title: s.arch.split("").map(function (k) { return ARCHETYPES.filter(function (a) { return a.key === k; })[0].name; }).join(", "), text: s.arch.split("").join(" ") }),
          h("td", { style: { fontSize: "13.5px" } }, modLinks(s.modules)),
          h("td", null, progressCell(s.modules)));
      });
      out.push(h("div", { class: "prose" }, h("div", { class: "table-wrap" },
        h("table", null,
          h("thead", null, h("tr", null, ["Skill", "Demand", "Archetypes", "Where you learn it", "Your progress"].map(function (t) { return h("th", { text: t }); }))),
          h("tbody", null, rows)))));

      // (d) trends
      out.push(section("Trending in 2025–2026", "what is new in postings"));
      out.push(h("ol", { class: "prose", style: { paddingLeft: "22px" } }, TRENDS.map(function (t) {
        return h("li", { style: { margin: "0 0 10px" } },
          h("b", null, h("a", { href: t.url, target: "_blank", rel: "noopener", text: t.t })), " — ", t.d,
          h("div", { style: { fontSize: "13px", marginTop: "3px" } }, modLinks(t.modules)));
      })));

      // (e) interview topics
      out.push(section("Interview topics", "and where to prepare"));
      out.push(h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "12px", margin: "10px 0" } },
        INTERVIEW.map(function (g) {
          return h("div", { class: "mod-card", style: { margin: "0", padding: "12px 16px" } },
            h("div", { class: "box-label", text: g.area }),
            h("ul", { style: { margin: "0", paddingLeft: "18px" } }, g.items.map(function (it) {
              return h("li", { style: { margin: "0 0 8px", fontSize: "14px" } }, it[0],
                h("div", { style: { fontSize: "12.5px" } }, modLinks(it[1])));
            })));
        })));
      out.push(ctx.md("div", "prose", "Practice drills with worked answers, three full system designs and the coding rounds are in [module " + ctx.moduleNum("m25-career-capstone") + "](#/m/m25-career-capstone)."));
      return out;
    },
  });
})();
