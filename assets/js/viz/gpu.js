/* gpu.js — interactive explainers for GPU hardware, CUDA and kernels. */
(function () {
  "use strict";
  var V = window.Viz;

  // ---------------------------------------------------------------- shared data & helpers
  // Dense (no sparsity) tensor-core peaks. peak8 = FP8 (Hopper/Ada/Blackwell) or INT8 (Turing/Ampere).
  var GPUS = {
    T4: { name: "T4", peak: 65, peak8: 130, bw: 0.32, mem: 16, p8: "INT8" },
    L4: { name: "L4", peak: 121, peak8: 242, bw: 0.3, mem: 24, p8: "FP8" },
    A100: { name: "A100 80GB", peak: 312, peak8: 624, bw: 2.04, mem: 80, p8: "INT8" },
    H100: { name: "H100 SXM", peak: 989, peak8: 1979, bw: 3.35, mem: 80, p8: "FP8" },
    B200: { name: "B200", peak: 2250, peak8: 4500, bw: 8, mem: 192, p8: "FP8" },
    M3Max: { name: "M3 Max (approx)", peak: 28, peak8: 28, bw: 0.4, mem: 128, p8: "none (same as FP16)" },
  };
  function gpuOptions() { return Object.keys(GPUS).map(function (k) { return { value: k, label: GPUS[k].name }; }); }
  function svgPt(svg, ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    var m = svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 };
  }
  // drag helper: getSvg() returns the *current* svg (it may be rebuilt while dragging)
  function onDrag(el, getSvg, move) {
    el.addEventListener("pointerdown", function (e) {
      e.preventDefault();
      var mv = function (ev) { move(svgPt(getSvg(), ev)); };
      var up = function () { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
      mv(e);
    });
  }
  // interval timer that stops itself when the card leaves the DOM
  function ticker(c, ms, tick, onState) {
    var id = null;
    function stop() { if (id) clearInterval(id); id = null; if (onState) onState(false); }
    function start() {
      if (id) return;
      id = setInterval(function () { if (!c.root.isConnected) { stop(); return; } if (tick() === false) stop(); }, ms);
      if (onState) onState(true);
    }
    return { start: start, stop: stop, running: function () { return !!id; }, toggle: function () { if (id) stop(); else start(); } };
  }
  function playButton(K, tm) {
    var b = K.button("▶ Play", function () { tm.toggle(); }, "primary");
    return b;
  }
  function T(K, x, y, text, cls, extra) {
    var a = { x: x, y: y, class: cls || "lbl-sm", text: text };
    if (extra) Object.keys(extra).forEach(function (k) { a[k] = extra[k]; });
    return K.s("text", a);
  }
  function col(n) { return { fill: "var(--c" + n + ")" }; }
  function stat(label, val) { return "<div class='stat'><small>" + label + "</small><b>" + val + "</b></div>"; }
  function grid(items) { return "<div class='stat-grid' style='margin-bottom:8px'>" + items.join("") + "</div>"; }
  function fmtT(s) {
    var a = Math.abs(s);
    if (a === 0) return "0";
    if (a < 1e-6) return +(s * 1e9).toPrecision(3) + " ns";
    if (a < 1e-3) return +(s * 1e6).toPrecision(3) + " µs";
    if (a < 1) return +(s * 1e3).toPrecision(3) + " ms";
    if (a < 120) return +s.toPrecision(3) + " s";
    return +(s / 60).toPrecision(3) + " min";
  }
  function fmtAI(v) { return v >= 10 ? Math.round(v).toLocaleString() : +v.toPrecision(2) + ""; }
  function table(head, rows, hlIdx) {
    var td = "padding:3px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap";
    var h = "<table style='border-collapse:collapse;font-size:12.5px;margin-top:6px'><tr>" +
      head.map(function (x, i) { return "<th style='" + td + (i ? "" : ";text-align:left") + ";color:var(--muted);font-weight:600'>" + x + "</th>"; }).join("") + "</tr>";
    rows.forEach(function (r, ri) {
      h += "<tr style='" + (ri === hlIdx ? "background:var(--accent-bg)" : "") + "'>" +
        r.map(function (x, i) { return "<td style='" + td + (i ? "" : ";text-align:left") + "'>" + x + "</td>"; }).join("") + "</tr>";
    });
    return h + "</table>";
  }

  // ---------------------------------------------------------------- roofline
  function gemmAI(M, N, Kd, b) { return 2 * M * N * Kd / (b * (M * Kd + Kd * N + M * N)); }
  var ROOF_OPS = [
    { id: "gemv", name: "Decode GEMV (batch 1)", cls: "f5", ai: function (b) { return gemmAI(1, 4096, 4096, b); }, lx: -8, ly: -10, anchor: "end",
      why: "y = W·x for one token and a 4096×4096 weight: 2·4096² FLOPs but every weight byte (4096²·b bytes) is read for a single multiply-add → I ≈ 2/b. This is why batch-1 decode is hopelessly memory-bound." },
    { id: "b64", name: "Decode GEMM (batch 64)", cls: "f2", ai: function (b) { return gemmAI(64, 4096, 4096, b); }, lx: -8, ly: -10, anchor: "end",
      why: "Same weight, 64 sequences batched: each weight byte is now reused 64× (I ≈ 2·64/b). Batching is the #1 lever of inference throughput — the time per step barely changes, tokens per step grow 64×." },
    { id: "prefill", name: "Prefill GEMM (4k tokens)", cls: "f3", ai: function (b) { return gemmAI(4096, 4096, 4096, b); }, lx: -8, ly: -12, anchor: "end",
      why: "4096 prompt tokens × a 4096×4096 weight: I = 2MNK / b(MK+KN+MN) ≈ 1365 (bf16). Prefill is compute-bound on every GPU: it is limited by tensor-core FLOPS, not HBM." },
    { id: "attn", name: "Attention decode (GQA, 4 q-heads/KV-head)", cls: "f4", ai: function (b) { return 2 * 4 / b; }, lx: 8, ly: 16, anchor: "start",
      why: "Each decode step reads the whole KV cache; each K/V element is used by the 4 query heads that share it (GQA) → I ≈ 2g/b = 4 (bf16). Batching does NOT help here: every sequence has its own KV cache." },
    { id: "softmax", name: "Softmax (row)", cls: "f6", ai: function (b) { return 5 / (2 * b); }, lx: 8, ly: 16, anchor: "start",
      why: "Read x, write y (2 elements) for ~5 FLOPs each (max, sub, exp, sum, div) → I ≈ 1.25. Pure bandwidth: fuse it into the neighbour kernel (FlashAttention) instead of optimizing it." },
    { id: "ew", name: "Elementwise add", cls: "f1", ai: function (b) { return 1 / (3 * b); }, lx: 8, ly: -10, anchor: "start",
      why: "c = a + b: 1 FLOP per 3 elements moved → I ≈ 0.17. Residual adds, activations, casts, norms: all far left of the ridge — kernel fusion is the only fix." },
  ];
  V.register("roofline", {
    title: "The roofline: is this operation limited by memory bandwidth or by compute?",
    desc: "Arithmetic intensity I = FLOPs done per byte moved to/from memory. Left of the <b>ridge point</b> you are <b>memory-bound</b> (slanted roof = bandwidth × I); right of it you are <b>compute-bound</b> (flat roof = peak FLOPS). Drag the orange marker, or click an operation point.",
    render: function (c, p, K) {
      var gk = GPUS[p.gpu] ? p.gpu : "H100";
      var b = (p.precision === "fp8" || p.precision === "8" || p.precision === 8) ? 1 : 2;
      var ai = +p.intensity || 1, showAll = true, curOp = null, svg = null;
      c.controls.appendChild(K.select({ label: "GPU", options: gpuOptions(), value: gk, onChange: function (v) { gk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "precision", options: [{ value: "2", label: "BF16/FP16 (2 B)" }, { value: "1", label: "FP8/INT8 (1 B)" }], value: String(b), onChange: function (v) { b = +v; if (curOp) { ai = curOp.ai(b); sl.set(Math.log10(ai)); } draw(); } }));
      var sl = K.slider({ label: "intensity I", min: -1.3, max: 4, step: 0.01, value: Math.log10(ai), fmt: function (v) { return fmtAI(Math.pow(10, v)) + " FLOP/B"; }, onInput: function (v) { ai = Math.pow(10, v); curOp = null; draw(); } });
      c.controls.appendChild(sl);
      c.controls.appendChild(K.toggle({ label: "show other GPUs", value: showAll, onChange: function (v) { showAll = v; draw(); } }));
      function peakOf(g) { return b === 1 ? g.peak8 : g.peak; }
      function setAI(v) { ai = K.clamp(v, 0.05, 1e4); sl.set(Math.log10(ai)); draw(); }
      function draw() {
        var g = GPUS[gk], pk = peakOf(g), ridge = pk / g.bw;
        var ch = K.chart({ w: 680, h: 360, x: [0.05, 1e4], y: [0.01, 1e4], xLog: true, yLog: true, pad: { l: 58, r: 16, t: 12, b: 44 },
          xLabel: "arithmetic intensity I (FLOPs per byte of memory traffic, log scale)", yLabel: "attainable TFLOPS (log)" });
        svg = ch.svg;
        var sx = ch.sx, sy = ch.sy, x0 = 0.05, x1 = 1e4;
        // drag capture area
        var ov = K.s("rect", { x: ch.pad.l, y: ch.pad.t, width: ch.W - ch.pad.l - ch.pad.r, height: ch.H - ch.pad.t - ch.pad.b, fill: "transparent", style: { cursor: "ew-resize" } });
        ch.g.appendChild(ov);
        onDrag(ov, function () { return svg; }, function (pt) { curOp = null; setAI(sx.invert(pt.x)); });
        if (showAll) Object.keys(GPUS).forEach(function (k) {
          if (k === gk) return;
          var o = GPUS[k], op = peakOf(o), r = op / o.bw;
          ch.g.appendChild(K.s("path", { d: K.path([[x0, o.bw * x0], [r, op], [x1, op]], sx, sy), class: "ln-thin smuted dash" }));
          ch.g.appendChild(T(K, sx(x1) - 4, sy(op) - 3, o.name.replace(" (approx)", "") + " " + op, "lbl-sm", { "text-anchor": "end" }));
        });
        // shaded regions
        ch.g.appendChild(K.s("path", { d: K.path([[x0, g.bw * x0], [ridge, pk], [ridge, 0.01], [x0, 0.01]], sx, sy) + "Z", class: "f5", opacity: 0.05 }));
        ch.g.appendChild(K.s("path", { d: K.path([[ridge, pk], [x1, pk], [x1, 0.01], [ridge, 0.01]], sx, sy) + "Z", class: "f3", opacity: 0.05 }));
        ch.g.appendChild(K.s("path", { d: K.path([[x0, g.bw * x0], [ridge, pk], [x1, pk]], sx, sy), class: "ln s1", style: { strokeWidth: "3px" } }));
        ch.g.appendChild(K.s("line", { x1: sx(ridge), x2: sx(ridge), y1: sy(pk), y2: sy(0.01), class: "ln-thin s1 dash" }));
        ch.g.appendChild(T(K, sx(ridge) + 5, sy(0.01) - 6, "ridge = " + fmtAI(ridge) + " FLOP/B", "lbl", { style: col(1) }));
        ch.g.appendChild(T(K, sx(x1) - 4, sy(pk) - 4, g.name + " " + pk + " TFLOPS", "lbl", { "text-anchor": "end" }));
        var mx = Math.sqrt(x0 * ridge) / 3;
        ch.g.appendChild(T(K, sx(mx), sy(g.bw * mx) - 18, "memory-bound: " + g.bw + " TB/s × I", "lbl-sm", { transform: "rotate(-" + (Math.atan2(sy(1) - sy(10), sx(10) - sx(1)) * 180 / Math.PI).toFixed(1) + " " + sx(mx) + " " + (sy(g.bw * mx) - 18) + ")" }));
        ch.g.appendChild(T(K, sx(Math.min(ridge * 1.3, 800)), sy(pk) + 16, "compute-bound", "lbl-sm"));
        // operations
        ROOF_OPS.forEach(function (o) {
          var a = o.ai(b), pf = Math.min(pk, g.bw * a);
          var gg = K.s("g", { style: { cursor: "pointer" } });
          gg.appendChild(K.s("circle", { cx: sx(a), cy: sy(pf), r: curOp === o ? 8 : 6, class: o.cls, style: { stroke: curOp === o ? "var(--fg)" : "none", strokeWidth: "2px" } }));
          gg.appendChild(T(K, sx(a) + o.lx, sy(pf) + o.ly, o.name, "lbl-sm", { "text-anchor": o.anchor }));
          gg.addEventListener("click", function () { curOp = o; setAI(o.ai(b)); });
          ch.g.appendChild(gg);
        });
        // marker
        var perf = Math.min(pk, g.bw * ai);
        ch.g.appendChild(K.s("line", { x1: sx(ai), x2: sx(ai), y1: sy(0.01), y2: sy(perf), class: "ln-thin s2 dash" }));
        ch.g.appendChild(K.s("line", { x1: ch.pad.l, x2: sx(ai), y1: sy(perf), y2: sy(perf), class: "ln-thin s2 dash" }));
        var mk = K.s("circle", { cx: sx(ai), cy: sy(perf), r: 8, class: "f2", style: { cursor: "grab" } });
        ch.g.appendChild(mk);
        onDrag(mk, function () { return svg; }, function (pt) { curOp = null; setAI(sx.invert(pt.x)); });
        ch.g.appendChild(T(K, ch.pad.l + 4, sy(perf) + 15, (perf >= 10 ? Math.round(perf) : +perf.toPrecision(2)) + " TFLOPS", "lbl", { style: col(2) }));
        K.clear(c.stage).appendChild(svg);

        var memB = ai < ridge, pct = perf / pk * 100;
        var gemmBatch = ridge * b / 2;
        c.readout.innerHTML = grid([
          stat("peak compute π", pk + " TFLOPS"), stat("memory bandwidth β", g.bw + " TB/s"), stat("ridge π/β", fmtAI(ridge) + " FLOP/B"),
          stat("your intensity I", fmtAI(ai) + " FLOP/B"), stat("attainable", (perf >= 10 ? Math.round(perf) : +perf.toPrecision(3)) + " TFLOPS"),
          stat("verdict", "<span style='color:var(--c" + (memB ? 5 : 3) + ")'>" + (memB ? "memory-bound" : "compute-bound") + "</span>"),
        ]) +
          K.tex("P = \\min(\\pi,\\ \\beta\\cdot I) = \\min(" + pk + ",\\ " + g.bw + "\\times" + fmtAI(ai) + ") = " + (+perf.toPrecision(3))) + " TFLOPS → " + pct.toFixed(pct < 1 ? 2 : 0) + "% of peak. " +
          (memB ? "The tensor cores sit idle waiting for memory; faster math won't help — move fewer bytes (quantize, batch, fuse). "
            : "Memory keeps up; you are limited by FLOPS — only fewer FLOPs, lower precision (FP8) or a bigger GPU help. ") +
          "For a GEMM with M tokens against a big weight, " + K.tex("I \\approx 2M/b") + ", so you need about <b>" + Math.round(gemmBatch) + " tokens</b> in a batch to reach the ridge on " + g.name + "." +
          (curOp ? "<br><b>" + curOp.name + ":</b> " + curOp.why : "");
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- memory hierarchy
  var LEVELS = [
    { name: "Registers", sub: "per thread, inside SM", bw: 3e14, lat: 0.5e-9, cap: 256e3 * 132, capN: "256 KB per SM × 132 SMs",
      d: "Fastest storage: per-thread scalars, MMA fragments and accumulators. 65,536 × 32-bit registers per SM. Using many registers per thread lowers occupancy (fewer warps resident to hide latency). Bandwidth is an aggregate estimate." },
    { name: "Shared memory / L1", sub: "per SM, programmer-managed", bw: 132 * 128 * 1.98e9, lat: 1.5e-8, cap: 228e3 * 132, capN: "up to 228 KB per SM (256 KB with L1)",
      d: "On-chip scratchpad you control (__shared__). Tiled matmul and FlashAttention stage tiles of A/B or Q/K/V here so each HBM byte is reused dozens of times. ≈132 SMs × 128 B/clk × 1.98 GHz." },
    { name: "L2 cache", sub: "shared by all SMs", bw: 7e12, lat: 1.5e-7, cap: 50e6, capN: "50 MB",
      d: "Hardware-managed, shared by every SM. Helps when many thread blocks read the same data (e.g. the same weight tile). Bandwidth/latency are approximate microbenchmark figures." },
    { name: "HBM3 (GPU memory)", sub: "off-chip, on package", bw: 3.35e12, lat: 5e-7, cap: 80e9, capN: "80 GB",
      d: "Where weights, KV cache and activations live. Decode reads ALL weights every step: 16 GB (Llama-3.1-8B bf16) ÷ 3.35 TB/s ≈ 4.8 ms → at most ~200 tokens/s per step at batch 1. This is the number that rules LLM inference." },
    { name: "NVLink 4 (GPU↔GPU)", sub: "inside a node, via NVSwitch", bw: 9e11, lat: 2e-6, cap: 7 * 80e9, capN: "7 peer GPUs × 80 GB",
      d: "900 GB/s total (450 GB/s each direction) per H100. Tensor-parallel all-reduces run here every layer — that's why TP stays inside one 8-GPU node." },
    { name: "PCIe Gen5 x16 (GPU↔CPU)", sub: "host link", bw: 64e9, lat: 5e-6, cap: 2e12, capN: "host DRAM, ~0.5–2 TB",
      d: "64 GB/s per direction: ~50× slower than HBM. Used for loading weights from host RAM, CPU offload of KV cache (swapping), and host↔device copies. Latency includes driver overhead." },
    { name: "InfiniBand NDR (node↔node)", sub: "400 Gb/s per NIC", bw: 50e9, lat: 3e-6, cap: 640e12, capN: "a cluster: ~1000 nodes × 640 GB",
      d: "400 Gb/s = 50 GB/s per NIC (nodes usually have 8). Multi-node pipeline/expert parallelism and disaggregated prefill→decode KV-cache transfer travel here." },
    { name: "NVMe SSD", sub: "local disk", bw: 14e9, lat: 8e-5, cap: 30e12, capN: "~30 TB per node",
      d: "Model checkpoints at rest. Cold start: a 140 GB (70B bf16) checkpoint at ~7–14 GB/s takes 10–20 s just to read — a big part of autoscaling latency." },
  ];
  V.register("memory-hierarchy", {
    title: "The memory hierarchy: every level is ~10× bigger and slower than the one above",
    desc: "H100-class numbers (≈ = approximate). Pick a metric; bars are on a log scale. Choose “time to move” and a size to see why weights must live in HBM and why data should be reused on-chip. Click a row for details.",
    render: function (c, p, K) {
      var metric = p.metric || "bw", lg = Math.log10(p.bytes || 16e9), sel = 3;
      var msel = K.select({ label: "metric", options: [{ value: "bw", label: "bandwidth" }, { value: "lat", label: "latency" }, { value: "cap", label: "capacity" }, { value: "time", label: "time to move N bytes" }], value: metric, onChange: function (v) { metric = v; draw(); } });
      c.controls.appendChild(msel);
      c.controls.appendChild(K.slider({ label: "N bytes", min: 3, max: 11.3, step: 0.05, value: lg, fmt: function (v) { return K.fmtBytes(Math.pow(10, v)); }, onInput: function (v) { lg = v; if (metric !== "time") { metric = "time"; msel.input.value = "time"; } draw(); } }));
      function val(L) {
        var n = Math.pow(10, lg);
        return metric === "bw" ? L.bw : metric === "lat" ? L.lat : metric === "cap" ? L.cap : L.lat + n / L.bw;
      }
      function fmtV(v) { return metric === "bw" ? K.fmtBytes(v) + "/s" : metric === "cap" ? K.fmtBytes(v) : fmtT(v); }
      function draw() {
        var W = 680, rh = 38, top = 30, H = top + LEVELS.length * rh + 12;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var vs = LEVELS.map(val), lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs);
        var l0 = Math.floor(Math.log10(lo)) - 0.3, l1 = Math.ceil(Math.log10(hi));
        var bx = K.scale(Math.pow(10, l0), Math.pow(10, l1), 225, 575, true);
        var good = metric === "bw" || metric === "cap" ? "longer bar = more" : "longer bar = slower";
        svg.appendChild(T(K, 225, 16, ["bandwidth", "latency", "capacity", "time to move " + K.fmtBytes(Math.pow(10, lg))][["bw", "lat", "cap", "time"].indexOf(metric)] + " (log scale; " + good + ")", "lbl"));
        for (var e = Math.ceil(l0); e <= l1; e++) {
          svg.appendChild(K.s("line", { x1: bx(Math.pow(10, e)), x2: bx(Math.pow(10, e)), y1: top - 4, y2: H - 8, class: "grid" }));
        }
        LEVELS.forEach(function (L, i) {
          var y = top + i * rh, v = vs[i], g = K.s("g", { style: { cursor: "pointer" } });
          g.appendChild(K.s("rect", { x: 2, y: y, width: W - 4, height: rh - 4, rx: 6, class: i === sel ? "box-hl" : "box", opacity: i === sel ? 1 : 0.5 }));
          g.appendChild(T(K, 10, y + 15, L.name, "lbl"));
          g.appendChild(T(K, 10, y + 28, L.sub, "lbl-sm"));
          var wv = Math.max(2, bx(v) - bx(Math.pow(10, l0)));
          g.appendChild(K.s("rect", { x: 225, y: y + 7, width: wv, height: rh - 18, rx: 3, class: "f" + (i < 3 ? 3 : i < 4 ? 1 : i < 7 ? 4 : 2), opacity: 0.8 }));
          g.appendChild(T(K, 225 + wv + 6, y + 22, fmtV(v), "lbl mono"));
          g.addEventListener("click", function () { sel = i; draw(); });
          svg.appendChild(g);
        });
        K.clear(c.stage).appendChild(svg);
        var L = LEVELS[sel], n = Math.pow(10, lg), hbm = LEVELS[3];
        c.readout.innerHTML = grid([stat(L.name + " bandwidth", "≈ " + K.fmtBytes(L.bw) + "/s"), stat("latency", "≈ " + fmtT(L.lat)), stat("capacity", K.fmtBytes(L.cap)),
          stat("move " + K.fmtBytes(n), fmtT(L.lat + n / L.bw))]) +
          "<b>" + L.name + "</b> (" + L.capN + "): " + L.d + "<br><span class='muted'>" + K.tex("t \\approx \\text{latency} + \\frac{\\text{bytes}}{\\text{bandwidth}}") +
          " — latency dominates tiny transfers, bandwidth dominates big ones. Moving " + K.fmtBytes(n) + " over PCIe is " + Math.round((LEVELS[5].lat + n / LEVELS[5].bw) / (hbm.lat + n / hbm.bw)) + "× slower than reading it from HBM.</span>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- GPU anatomy
  var PARTS = {
    gpu: ["H100 SXM5 (GH100)", "132 SMs in 8 GPCs · 528 tensor cores · 16,896 FP32 lanes · 50 MB L2 · 80 GB HBM3 at 3.35 TB/s · ~1.8–1.98 GHz. Peak dense BF16: 132 SMs × 4 tensor cores × 1024 FLOP/clk × 1.83 GHz ≈ 989 TFLOPS. Click a GPC or an SM to zoom in."],
    gpc: ["GPC (Graphics Processing Cluster)", "A group of 9 TPCs = 18 SMs on the full die (some disabled for manufacturing yield: H100 SXM ships 132 of 144). On Hopper, a <i>thread-block cluster</i> spans SMs of one GPC, which can read each other's shared memory (distributed shared memory)."],
    tpc: ["TPC (Texture Processing Cluster)", "Two SMs sharing some front-end hardware. For CUDA programming you can mostly ignore TPCs — the SM is the unit that matters."],
    sm: ["SM (Streaming Multiprocessor)", "The GPU's 'core'. A thread block runs entirely on one SM. Each SM has 4 sub-partitions, 256 KB of L1/shared memory, 64K registers, and can keep up to 64 warps (2048 threads) resident. More resident warps = more latency hiding."],
    l2: ["L2 cache — 50 MB", "Shared by all SMs (two partitions connected by a crossbar). Every HBM read goes through L2. Weights are far too big to stay in L2 (16 GB vs 50 MB), so decode streams them from HBM each step."],
    hbm: ["HBM3 stacks", "5 active stacks × 16 GB = 80 GB, 5120-bit bus, 3.35 TB/s total. Stacked DRAM on the same package as the die. Holds weights + KV cache + activations. (The 6th stack site is unused on H100 SXM.)"],
    nvlink: ["NVLink 4", "18 links × 50 GB/s = 900 GB/s total (450 GB/s each way) to other GPUs through NVSwitch. Tensor-parallel all-reduces use this."],
    pcie: ["PCIe Gen5 x16", "64 GB/s each way to the CPU / host memory / NIC. ~50× slower than HBM — avoid it on the hot path."],
    sched: ["Warp scheduler + dispatch", "Each sub-partition picks one <i>ready</i> warp per clock and issues its next instruction to all 32 threads at once (SIMT). A warp waiting on memory (~500 ns = ~1000 cycles for HBM) is skipped — switching warps is free, which is how GPUs hide latency. Also here: LD/ST units and SFUs (exp, rsqrt…)."],
    warps: ["Warps", "A warp = 32 threads executing the same instruction in lockstep. Up to 16 warps per sub-partition (64 per SM) can be resident. Press “Schedule warps” and lower “resident warps” to see the scheduler starve when there are too few warps to cover memory stalls (low occupancy)."],
    regs: ["Register file — 64 KB per sub-partition", "16,384 × 32-bit registers per sub-partition (256 KB per SM). Split among resident threads: 2048 threads × 32 regs fills it; a kernel using 128 regs/thread fits only 512 threads (16 warps) per SM."],
    int32: ["INT32 units ×16", "Integer math: address/index calculations (i = blockIdx.x*blockDim.x + threadIdx.x)."],
    fp32: ["FP32 lanes ×32", "Classic 'CUDA cores': one fp32 FMA per lane per clock. 128 per SM × 132 SMs × 2 FLOP × 1.98 GHz ≈ 67 TFLOPS — softmax, norms, activations run here, not on tensor cores."],
    fp64: ["FP64 units ×16", "Double precision for HPC. Irrelevant for LLM inference."],
    tc: ["Tensor core (4th gen)", "Does a small matrix multiply-accumulate (e.g. 8×4×16 per instruction group) every clock: 1024 dense BF16 FLOP/clk, 2048 FP8. 4 per SM = where ~95% of the GPU's FLOPS live. Matmuls must be tiled into these shapes to use them."],
    smem: ["L1 data cache / shared memory — 256 KB", "Configurable split; up to 228 KB usable as __shared__ memory per SM (≤227 KB per block). ~33 TB/s aggregate. This is where tiled matmul and FlashAttention keep their tiles."],
    tma: ["TMA (Tensor Memory Accelerator)", "Hopper hardware that copies whole tiles between HBM and shared memory asynchronously, so warps can compute while the next tile streams in (used by CUTLASS / FlashAttention-3)."],
  };
  V.register("gpu-anatomy", {
    title: "Inside an H100: GPU → GPC → SM → warps & tensor cores",
    desc: "Click to zoom: the die (8 GPCs, L2, HBM), a GPC (TPCs and SMs), and one SM (4 sub-partitions with warp schedulers, registers, FP32 lanes and tensor cores). Hover/click any part for numbers.",
    render: function (c, p, K) {
      var level = p.level === "sm" || p.level === "gpc" ? p.level : "gpu", gpcIdx = 0, part = level, occ = 64;
      var r = K.rng(5), warps = [], issued = [], ticks = 0, issuedTotal = 0;
      function initWarps() { warps = []; for (var i = 0; i < 64; i++) warps.push(0); ticks = 0; issuedTotal = 0; issued = []; }
      initWarps();
      var nav = K.h("span", { class: "ctl" });
      c.controls.appendChild(nav);
      var tm = ticker(c, 350, function () { tick(); draw(); }, function (on) { pb.textContent = on ? "⏸ Pause warps" : "▶ Schedule warps"; });
      var pb = K.button("▶ Schedule warps", function () { if (level !== "sm") { level = "sm"; } part = "warps"; tm.toggle(); draw(); }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.slider({ label: "resident warps / SM", min: 4, max: 64, step: 4, value: occ, onInput: function (v) { occ = v; initWarps(); draw(); } }));
      function tick() {
        ticks++; issued = [];
        for (var q = 0; q < 4; q++) {
          var per = occ / 4, cand = [];
          for (var j = 0; j < per; j++) { var w = q * 16 + j; if (warps[w] === 0) cand.push(w); }
          if (cand.length) {
            var w2 = cand[Math.floor(r() * cand.length)];
            issued.push(w2); issuedTotal++;
            var u = r();
            warps[w2] = u < 0.3 ? 6 + Math.floor(r() * 10) : u < 0.6 ? 1 + Math.floor(r() * 2) : 0;
            warps[w2] = warps[w2] ? warps[w2] + 1 : 0;
          }
        }
        for (var i = 0; i < 64; i++) if (warps[i] > 0) warps[i]--;
      }
      function go(l, pt) { level = l; part = pt || l; draw(); }
      function hov(el, key) { el.style.cursor = "pointer"; el.addEventListener("mouseenter", function () { part = key; info(); }); }
      function info() {
        var d = PARTS[part] || PARTS[level];
        var extra = "";
        if (level === "sm" && ticks) extra = "<br><span class='muted'>Scheduler: " + ticks + " cycles, " + issuedTotal + " of " + ticks * 4 + " issue slots used (" + Math.round(issuedTotal / ticks / 4 * 100) + "%) with " + occ + " resident warps. Orange = issued this cycle, green = ready, grey = stalled on memory/dependency.</span>";
        c.readout.innerHTML = "<b>" + d[0] + "</b> — " + d[1] + extra;
      }
      function crumbs() {
        K.clear(nav);
        nav.appendChild(K.button("GPU", function () { go("gpu"); }, level === "gpu" ? "primary" : ""));
        nav.appendChild(K.button("GPC " + gpcIdx, function () { go("gpc"); }, level === "gpc" ? "primary" : ""));
        nav.appendChild(K.button("SM", function () { go("sm"); }, level === "sm" ? "primary" : ""));
      }
      var DIS = [2, 1, 2, 1, 2, 1, 2, 1];
      function drawGPU(svg) {
        svg.appendChild(T(K, 340, 16, "GH100 die — H100 SXM5 enables 132 of 144 SMs", "lbl", { "text-anchor": "middle" }));
        svg.appendChild(K.s("rect", { x: 88, y: 24, width: 504, height: 328, rx: 10, class: "box" }));
        for (var hI = 0; hI < 6; hI++) {
          var hx = hI < 3 ? 18 : 602, hy = 40 + (hI % 3) * 105, act = hI !== 5;
          var hg = K.s("g");
          hg.appendChild(K.s("rect", { x: hx, y: hy, width: 60, height: 90, rx: 6, class: act ? "f4" : "fmuted", opacity: act ? 0.75 : 0.25 }));
          hg.appendChild(T(K, hx + 30, hy + 42, act ? "HBM3" : "unused", act ? "lbl t-white" : "lbl-sm", { "text-anchor": "middle" }));
          if (act) hg.appendChild(T(K, hx + 30, hy + 58, "16 GB", "lbl-sm t-white", { "text-anchor": "middle" }));
          hov(hg, "hbm"); svg.appendChild(hg);
        }
        var l2 = K.s("g");
        l2.appendChild(K.s("rect", { x: 98, y: 170, width: 484, height: 36, rx: 6, class: "f6", opacity: 0.7 }));
        l2.appendChild(T(K, 340, 193, "L2 cache 50 MB (2 partitions + crossbar)", "lbl t-white", { "text-anchor": "middle" }));
        hov(l2, "l2"); svg.appendChild(l2);
        for (var g = 0; g < 8; g++) {
          var gx = 98 + (g % 4) * 122, gy = g < 4 ? 34 : 212;
          var gg = K.s("g");
          gg.appendChild(K.s("rect", { x: gx, y: gy, width: 116, height: 130, rx: 7, class: "box-hl", opacity: 0.9 }));
          gg.appendChild(T(K, gx + 6, gy + 14, "GPC " + g, "lbl-sm"));
          (function (g) { hov(gg, "gpc"); gg.addEventListener("click", function () { gpcIdx = g; go("gpc"); }); })(g);
          svg.appendChild(gg);
          for (var s = 0; s < 18; s++) {
            var on = s < 18 - DIS[g];
            var sm = K.s("rect", { x: gx + 5 + (s % 6) * 18, y: gy + 22 + Math.floor(s / 6) * 35, width: 15, height: 31, rx: 2, class: on ? "f1" : "fmuted", opacity: on ? 0.8 : 0.3 });
            (function (g) { hov(sm, "sm"); sm.addEventListener("click", function (e) { e.stopPropagation(); gpcIdx = g; go("sm"); }); })(g);
            svg.appendChild(sm);
          }
        }
        var nv = K.s("g");
        nv.appendChild(K.s("rect", { x: 88, y: 358, width: 300, height: 26, rx: 6, class: "f5", opacity: 0.7 }));
        nv.appendChild(T(K, 238, 375, "NVLink 4 · 18 links · 900 GB/s", "lbl t-white", { "text-anchor": "middle" }));
        hov(nv, "nvlink"); svg.appendChild(nv);
        var pc = K.s("g");
        pc.appendChild(K.s("rect", { x: 396, y: 358, width: 196, height: 26, rx: 6, class: "f2", opacity: 0.7 }));
        pc.appendChild(T(K, 494, 375, "PCIe Gen5 x16 → CPU", "lbl t-white", { "text-anchor": "middle" }));
        hov(pc, "pcie"); svg.appendChild(pc);
      }
      function drawGPC(svg) {
        svg.appendChild(T(K, 340, 16, "GPC " + gpcIdx + ": 9 TPCs × 2 SMs (" + (18 - DIS[gpcIdx]) + " enabled) — click an SM", "lbl", { "text-anchor": "middle" }));
        svg.appendChild(K.s("rect", { x: 40, y: 26, width: 600, height: 330, rx: 10, class: "box" }));
        for (var t = 0; t < 9; t++) {
          var tx = 56 + (t % 3) * 194, ty = 38 + Math.floor(t / 3) * 106;
          var tg = K.s("g");
          tg.appendChild(K.s("rect", { x: tx, y: ty, width: 182, height: 96, rx: 8, class: "box-hl", opacity: 0.6 }));
          tg.appendChild(T(K, tx + 6, ty + 13, "TPC " + t, "lbl-sm"));
          hov(tg, "tpc"); svg.appendChild(tg);
          for (var s = 0; s < 2; s++) {
            var idx = t * 2 + s, on = idx < 18 - DIS[gpcIdx];
            var sg = K.s("g");
            sg.appendChild(K.s("rect", { x: tx + 8 + s * 88, y: ty + 20, width: 80, height: 68, rx: 6, class: on ? "f1" : "fmuted", opacity: on ? 0.85 : 0.3 }));
            sg.appendChild(T(K, tx + 48 + s * 88, ty + 50, on ? "SM" : "disabled", on ? "lbl t-white" : "lbl-sm", { "text-anchor": "middle" }));
            if (on) {
              sg.appendChild(T(K, tx + 48 + s * 88, ty + 66, "4 TC · 128 FP32", "lbl-sm t-white", { "text-anchor": "middle" }));
              hov(sg, "sm"); sg.addEventListener("click", function () { go("sm"); });
            }
            svg.appendChild(sg);
          }
        }
        svg.appendChild(T(K, 340, 374, "↑ up: shared L2 (50 MB) and HBM (80 GB) are outside the GPC", "lbl-sm", { "text-anchor": "middle" }));
      }
      function drawSM(svg) {
        svg.appendChild(T(K, 340, 16, "One SM: 4 sub-partitions (each: 1 scheduler, 64 KB registers, 32 FP32 lanes, 1 tensor core)", "lbl", { "text-anchor": "middle" }));
        for (var q = 0; q < 4; q++) {
          var qx = 12 + (q % 2) * 336, qy = 26 + Math.floor(q / 2) * 136;
          svg.appendChild(K.s("rect", { x: qx, y: qy, width: 320, height: 128, rx: 8, class: "box" }));
          var sc = K.s("g");
          sc.appendChild(K.s("rect", { x: qx + 8, y: qy + 7, width: 304, height: 20, rx: 4, class: "f2", opacity: 0.8 }));
          sc.appendChild(T(K, qx + 160, qy + 21, "Warp scheduler + dispatch (1 instr / clk)", "lbl-sm t-white", { "text-anchor": "middle" }));
          hov(sc, "sched"); svg.appendChild(sc);
          var wg = K.s("g");
          for (var j = 0; j < 16; j++) {
            var w = q * 16 + j, res = j < occ / 4;
            var cls = !res ? "fline" : issued.indexOf(w) >= 0 ? "f2" : warps[w] > 0 ? "fmuted" : "f3";
            wg.appendChild(K.s("rect", { x: qx + 8 + j * 19, y: qy + 31, width: 16, height: 11, rx: 3, class: cls, opacity: res ? 0.9 : 0.5 }));
          }
          hov(wg, "warps"); svg.appendChild(wg);
          var rg = K.s("g");
          rg.appendChild(K.s("rect", { x: qx + 8, y: qy + 47, width: 304, height: 20, rx: 4, class: "f4", opacity: 0.75 }));
          rg.appendChild(T(K, qx + 160, qy + 61, "Register file 64 KB (16,384 × 32-bit)", "lbl-sm t-white", { "text-anchor": "middle" }));
          hov(rg, "regs"); svg.appendChild(rg);
          var units = [["int32", "INT32", "×16", 58, "f6"], ["fp32", "FP32", "×32", 84, "f1"], ["fp64", "FP64", "×16", 58, "fmuted"], ["tc", "Tensor", "Core", 88, "f3"]];
          var ux = qx + 8;
          units.forEach(function (u) {
            var ug = K.s("g");
            ug.appendChild(K.s("rect", { x: ux, y: qy + 72, width: u[3], height: 48, rx: 5, class: u[4], opacity: 0.8 }));
            if (u[0] === "fp32") for (var k = 0; k < 32; k++) ug.appendChild(K.s("rect", { x: ux + 4 + (k % 8) * 9.6, y: qy + 76 + Math.floor(k / 8) * 7, width: 7, height: 5, class: "fbg", opacity: 0.6 }));
            ug.appendChild(T(K, ux + u[3] / 2, qy + (u[0] === "fp32" ? 114 : 94), u[1] + " " + (u[0] === "fp32" ? u[2] : ""), "lbl-sm t-white", { "text-anchor": "middle" }));
            if (u[0] !== "fp32") ug.appendChild(T(K, ux + u[3] / 2, qy + 108, u[2], "lbl-sm t-white", { "text-anchor": "middle" }));
            hov(ug, u[0]); svg.appendChild(ug);
            ux += u[3] + 4;
          });
        }
        var sm = K.s("g");
        sm.appendChild(K.s("rect", { x: 12, y: 300, width: 520, height: 32, rx: 6, class: "f6", opacity: 0.75 }));
        sm.appendChild(T(K, 272, 321, "L1 data cache / shared memory — 256 KB (≤ 228 KB shared)", "lbl t-white", { "text-anchor": "middle" }));
        hov(sm, "smem"); svg.appendChild(sm);
        var tma = K.s("g");
        tma.appendChild(K.s("rect", { x: 540, y: 300, width: 128, height: 32, rx: 6, class: "f5", opacity: 0.75 }));
        tma.appendChild(T(K, 604, 321, "TMA (async copy)", "lbl t-white", { "text-anchor": "middle" }));
        hov(tma, "tma"); svg.appendChild(tma);
        svg.appendChild(T(K, 340, 356, "↕ L2 cache (shared by all 132 SMs) ↔ HBM3 80 GB", "lbl-sm", { "text-anchor": "middle" }));
        svg.appendChild(T(K, 340, 374, "max 64 warps = 2048 threads resident · max 32 blocks · 1024 threads per block", "lbl-sm", { "text-anchor": "middle" }));
      }
      function draw() {
        crumbs();
        var svg = K.s("svg", { viewBox: "0 0 680 390", class: "viz-svg" });
        if (level === "gpu") drawGPU(svg); else if (level === "gpc") drawGPC(svg); else drawSM(svg);
        if (level !== "sm" && tm.running()) tm.stop();
        K.clear(c.stage).appendChild(svg);
        info();
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- CUDA grid / blocks / threads
  var BCOL = ["f1", "f2", "f3", "f4", "f5", "f6"];
  V.register("cuda-grid", {
    title: "CUDA indexing: grid → blocks → threads → your array element",
    desc: "A kernel launch <code>kernel&lt;&lt;&lt;blocks, threadsPerBlock&gt;&gt;&gt;</code> creates many threads; each computes its own global index and handles one element. Hover a thread or an element. Threads past the end are masked off by <code>if (i &lt; n)</code>.",
    render: function (c, p, K) {
      var mode = p.mode === "2d" ? "2d" : "1d", n = p.n || 20, B = p.block || 8;
      var Wm = p.width || 13, Hm = p.height || 10, bd = p.block2d || "4x4", hover = null;
      var c1 = K.h("span", { class: "ctl", style: { gap: "18px", flexWrap: "wrap" } }), c2 = K.h("span", { class: "ctl", style: { gap: "18px", flexWrap: "wrap" } });
      c.controls.appendChild(K.select({ label: "layout", options: [{ value: "1d", label: "1-D array" }, { value: "2d", label: "2-D matrix" }], value: mode, onChange: function (v) { mode = v; hover = null; draw(); } }));
      c1.appendChild(K.slider({ label: "n elements", min: 1, max: 100, value: n, onInput: function (v) { n = v; hover = null; draw(); } }));
      c1.appendChild(K.select({ label: "blockDim.x", options: ["4", "8", "16", "32"], value: String(B), onChange: function (v) { B = +v; hover = null; draw(); } }));
      c2.appendChild(K.slider({ label: "width (cols)", min: 2, max: 24, value: Wm, onInput: function (v) { Wm = v; hover = null; draw(); } }));
      c2.appendChild(K.slider({ label: "height (rows)", min: 2, max: 14, value: Hm, onInput: function (v) { Hm = v; hover = null; draw(); } }));
      c2.appendChild(K.select({ label: "blockDim (x×y)", options: ["2x2", "4x4", "8x4", "4x8", "8x8"], value: bd, onChange: function (v) { bd = v; hover = null; draw(); } }));
      c.controls.appendChild(c1); c.controls.appendChild(c2);
      function cell(svg, x, y, w, h, cls, op, text, key, hl) {
        var g = K.s("g", { style: { cursor: "pointer" } });
        g.appendChild(K.s("rect", { x: x, y: y, width: w - 2, height: h - 2, rx: 3, class: cls, opacity: op }));
        if (hl) g.appendChild(K.s("rect", { x: x - 1, y: y - 1, width: w, height: h, rx: 3, fill: "none", class: "sfg", "stroke-width": 2 }));
        if (text !== null) g.appendChild(T(K, x + w / 2 - 1, y + h / 2 + 3, text, "lbl-sm mono", { "text-anchor": "middle" }));
        g.addEventListener("mouseenter", function () { if (JSON.stringify(hover) === JSON.stringify(key)) return; hover = key; draw(); });
        svg.appendChild(g);
      }
      function draw1d() {
        var nb = Math.ceil(n / B), cs = 18, W = 680, perRow = Math.max(1, Math.floor((W - 20) / (B * cs + 10)));
        var gridRows = Math.ceil(nb / perRow), gy0 = 22, bh = 46;
        var ay0 = gy0 + gridRows * bh + 30, arrRows = Math.ceil(n / 32), H = ay0 + arrRows * 22 + 10;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        svg.appendChild(T(K, 10, 14, "Grid: " + nb + " blocks × " + B + " threads = " + nb * B + " threads (numbers = threadIdx.x)", "lbl"));
        var hi = hover ? hover.i : -1, pos = {};
        for (var b = 0; b < nb; b++) {
          var bx = 10 + (b % perRow) * (B * cs + 10), by = gy0 + Math.floor(b / perRow) * bh;
          svg.appendChild(K.s("rect", { x: bx - 3, y: by + 10, width: B * cs + 4, height: cs + 8, rx: 5, fill: "none", class: "smuted" }));
          svg.appendChild(T(K, bx, by + 7, "blockIdx.x = " + b, "lbl-sm"));
          for (var t = 0; t < B; t++) {
            var i = b * B + t, act = i < n;
            pos["t" + i] = [bx + t * cs + cs / 2, by + 14 + cs];
            cell(svg, bx + t * cs, by + 14, cs, cs, act ? BCOL[b % 6] : "fmuted", act ? 0.35 : 0.15, act ? t : "×", { i: i, b: b, t: t }, i === hi);
          }
        }
        svg.appendChild(T(K, 10, ay0 - 8, "Array a[0.." + (n - 1) + "] in global memory (numbers = element index i, colour = block that owns it)", "lbl"));
        for (var j = 0; j < n; j++) {
          var ax = 10 + (j % 32) * 20.5, ay = ay0 + Math.floor(j / 32) * 22;
          pos["a" + j] = [ax + 10, ay];
          cell(svg, ax, ay, 20.5, 20, BCOL[Math.floor(j / B) % 6], 0.35, j, { i: j, b: Math.floor(j / B), t: j % B }, j === hi);
        }
        if (hover && hover.i < n) svg.appendChild(K.s("line", { x1: pos["t" + hi][0], y1: pos["t" + hi][1], x2: pos["a" + hi][0], y2: pos["a" + hi][1], class: "ln s2" }));
        K.clear(c.stage).appendChild(svg);
        var h = hover || { i: 0, b: 0, t: 0 };
        var code = "<code>__global__ void add(float *a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i &lt; n) a[i] += 1.0f; }</code><br><code>add&lt;&lt;&lt;" + nb + ", " + B + "&gt;&gt;&gt;(a, " + n + ");</code> &nbsp;<span class='muted'>(blocks = ceil(n / blockDim) = ceil(" + n + "/" + B + "))</span>";
        c.readout.innerHTML = K.tex("i = \\text{blockIdx.x}\\times\\text{blockDim.x} + \\text{threadIdx.x} = " + h.b + "\\times" + B + " + " + h.t + " = " + h.i) +
          (h.i >= n ? " &nbsp;<b style='color:var(--c5)'>≥ n → this thread does nothing</b> (" + (nb * B - n) + " idle threads)" : " → handles <b>a[" + h.i + "]</b>") +
          "<br>" + code + "<br><span class='muted'>Real kernels use 128–1024 threads per block (multiples of 32 = one warp) and thousands of blocks; the GPU schedules blocks onto SMs in any order.</span>";
      }
      function draw2d() {
        var bxd = +bd.split("x")[0], byd = +bd.split("x")[1];
        var gx = Math.ceil(Wm / bxd), gy = Math.ceil(Hm / byd), PW = gx * bxd, PH = gy * byd;
        var cs = Math.min(26, Math.floor(560 / PW), Math.floor(300 / PH)), x0 = 60, y0 = 34;
        var W = 680, H = y0 + PH * cs + 16;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        svg.appendChild(T(K, 10, 14, "gridDim = (" + gx + ", " + gy + ") blocks, blockDim = (" + bxd + ", " + byd + ") threads → covers " + PW + "×" + PH + " for a " + Wm + "×" + Hm + " matrix", "lbl"));
        for (var cc = 0; cc < PW; cc++) svg.appendChild(T(K, x0 + cc * cs + cs / 2, y0 - 4, cc, "tick", { "text-anchor": "middle" }));
        svg.appendChild(T(K, x0 - 30, y0 - 4, "col →", "tick", { "text-anchor": "middle" }));
        for (var r0 = 0; r0 < PH; r0++) svg.appendChild(T(K, x0 - 6, y0 + r0 * cs + cs / 2 + 4, "row " + r0, "tick", { "text-anchor": "end" }));
        for (var row = 0; row < PH; row++) for (var col2 = 0; col2 < PW; col2++) {
          var bX = Math.floor(col2 / bxd), bY = Math.floor(row / byd), inM = row < Hm && col2 < Wm;
          var key = { row: row, col: col2, bX: bX, bY: bY, tX: col2 % bxd, tY: row % byd, inM: inM };
          var hl = hover && hover.row === row && hover.col === col2;
          cell(svg, x0 + col2 * cs, y0 + row * cs, cs, cs, inM ? BCOL[(bX + bY * 3) % 6] : "fmuted", inM ? 0.4 : 0.12, cs >= 22 && inM ? row * Wm + col2 : null, key, hl);
        }
        for (var by2 = 0; by2 < gy; by2++) for (var bx2 = 0; bx2 < gx; bx2++)
          svg.appendChild(K.s("rect", { x: x0 + bx2 * bxd * cs - 1, y: y0 + by2 * byd * cs - 1, width: bxd * cs, height: byd * cs, fill: "none", class: "sfg", "stroke-width": 1.5, opacity: 0.6 }));
        K.clear(c.stage).appendChild(svg);
        var h = hover || { row: 0, col: 0, bX: 0, bY: 0, tX: 0, tY: 0, inM: true };
        c.readout.innerHTML = "blockIdx = (" + h.bX + ", " + h.bY + "), threadIdx = (" + h.tX + ", " + h.tY + ") → " +
          K.tex("\\text{col} = " + h.bX + "\\cdot" + bxd + " + " + h.tX + " = " + h.col + ",\\ \\ \\text{row} = " + h.bY + "\\cdot" + byd + " + " + h.tY + " = " + h.row) +
          (h.inM ? " → element " + K.tex("A[\\text{row}\\cdot W + \\text{col}] = A[" + (h.row * Wm + h.col) + "]") + " (row-major)" : " <b style='color:var(--c5)'>outside the matrix → masked by <code>if (row &lt; H &amp;&amp; col &lt; W)</code></b>") +
          "<br><code>dim3 block(" + bxd + ", " + byd + "); dim3 grid(" + gx + ", " + gy + "); // grid = (ceil(W/bx), ceil(H/by))</code><br><span class='muted'>Note threadIdx.x varies fastest and walks along a row: consecutive threads touch consecutive addresses (coalesced). Swap row/col and you get strided access.</span>";
      }
      function draw() {
        c1.style.display = mode === "1d" ? "" : "none"; c2.style.display = mode === "2d" ? "" : "none";
        if (mode === "1d") draw1d(); else draw2d();
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- memory coalescing
  V.register("coalescing", {
    title: "Memory coalescing: how a warp's 32 loads become memory transactions",
    desc: "The 32 threads of a warp issue their loads together. Hardware fetches memory in 32-byte <b>sectors</b> (4 per 128-byte line). Neighbouring threads reading neighbouring addresses → few sectors; strided or random access → up to 32 sectors, most bytes wasted. Hover a thread.",
    render: function (c, p, K) {
      var pat = p.pattern || "stride", stride = p.stride || 2, off = p.offset || 0, es = p.elem || 4, seed = 3, hov = -1;
      var patSel = K.select({ label: "pattern", options: [
        { value: "contig", label: "contiguous a[tid]" }, { value: "stride", label: "strided a[tid*stride]" }, { value: "offset", label: "misaligned a[tid+offset]" },
        { value: "random", label: "random gather a[idx[tid]]" }, { value: "bcast", label: "broadcast a[0]" }], value: pat, onChange: function (v) { pat = v; draw(); } });
      c.controls.appendChild(patSel);
      c.controls.appendChild(K.select({ label: "element", options: [{ value: "2", label: "2 B (bf16)" }, { value: "4", label: "4 B (fp32)" }, { value: "8", label: "8 B" }, { value: "16", label: "16 B (float4 / 8×bf16)" }], value: String(es), onChange: function (v) { es = +v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "stride", min: 1, max: 64, value: stride, onInput: function (v) { stride = v; if (pat !== "stride") { pat = "stride"; patSel.input.value = pat; } draw(); } }));
      c.controls.appendChild(K.slider({ label: "offset", min: 0, max: 31, value: off, onInput: function (v) { off = v; if (pat !== "offset") { pat = "offset"; patSel.input.value = pat; } draw(); } }));
      c.controls.appendChild(K.button("New random", function () { seed++; pat = "random"; patSel.input.value = pat; draw(); }));
      function idxOf(t, rnd) {
        return pat === "contig" ? t : pat === "stride" ? t * stride : pat === "offset" ? t + off : pat === "random" ? rnd[t] : 0;
      }
      function draw() {
        var r = K.rng(seed), rnd = []; for (var q = 0; q < 32; q++) rnd.push(Math.floor(r() * 4096 / es * 4));
        var addr = [], sectors = {}, lines = {}, uniq = {};
        for (var t = 0; t < 32; t++) {
          var a = idxOf(t, rnd) * es; addr.push(a);
          sectors[Math.floor(a / 32)] = 1; lines[Math.floor(a / 128)] = 1;
          for (var bb = 0; bb < es; bb++) uniq[a + bb] = 1;
        }
        var nS = Object.keys(sectors).length, lineIds = Object.keys(lines).map(Number).sort(function (x, y) { return x - y; });
        var nU = Object.keys(uniq).length, eff = nU / (nS * 32);
        var W = 680, x0 = 60, pxB = 608 / 128, y0 = 92, nR = lineIds.length;
        var rowH = Math.min(16, Math.max(7, 250 / nR)), rowY = {}, y = y0;
        lineIds.forEach(function (L, i) { if (i && L !== lineIds[i - 1] + 1) y += 6; rowY[L] = y; y += rowH; });
        var H = y + 26;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        svg.appendChild(T(K, x0, 14, "one warp = 32 threads (tid)", "lbl"));
        svg.appendChild(T(K, x0, y0 - 12, "memory: touched 128-byte lines only (4 × 32 B sectors each)" + (nR > 1 && lineIds[nR - 1] - lineIds[0] + 1 > nR ? " — gaps skipped" : ""), "lbl"));
        lineIds.forEach(function (L) {
          var ry = rowY[L];
          svg.appendChild(T(K, x0 - 4, ry + rowH / 2 + 3, "0x" + (L * 128).toString(16), "tick mono", { "text-anchor": "end", style: { fontSize: rowH < 10 ? "8px" : "10px" } }));
          for (var s = 0; s < 4; s++) {
            var sid = L * 4 + s;
            svg.appendChild(K.s("rect", { x: x0 + s * 32 * pxB, y: ry, width: 32 * pxB - 1, height: rowH - 1.5, class: sectors[sid] ? "f2" : "fbg3", opacity: sectors[sid] ? 0.25 : 0.6 }));
          }
        });
        for (var t2 = 0; t2 < 32; t2++) {
          var a2 = addr[t2], L2 = Math.floor(a2 / 128), ex = x0 + (a2 % 128) * pxB, ey = rowY[L2];
          svg.appendChild(K.s("rect", { x: ex, y: ey + 1, width: Math.max(1.5, es * pxB - 0.5), height: rowH - 3.5, class: "f1", opacity: 0.85 }));
          var tx = x0 + t2 * 19 + 9;
          svg.appendChild(K.s("line", { x1: tx, y1: 42, x2: ex + es * pxB / 2, y2: ey, class: t2 === hov ? "ln s2" : "ln-thin s1", opacity: t2 === hov ? 1 : 0.28 }));
        }
        for (var t3 = 0; t3 < 32; t3++) {
          var g = K.s("g", { style: { cursor: "pointer" } });
          g.appendChild(K.s("rect", { x: x0 + t3 * 19, y: 22, width: 17, height: 20, rx: 3, class: t3 === hov ? "f2" : "f1", opacity: t3 === hov ? 1 : 0.8 }));
          g.appendChild(T(K, x0 + t3 * 19 + 8.5, 36, t3, "lbl-sm t-white", { "text-anchor": "middle", style: { fontSize: "9px" } }));
          (function (t3) { g.addEventListener("mouseenter", function () { if (hov !== t3) { hov = t3; draw(); } }); })(t3);
          svg.appendChild(g);
        }
        svg.appendChild(T(K, x0, H - 8, "■ blue = bytes a thread asked for   ■ orange sector = 32 B actually fetched from memory", "lbl-sm"));
        K.clear(c.stage).appendChild(svg);
        var ideal = Math.ceil(32 * es / 32);
        var h = hov >= 0 ? hov : 1;
        c.readout.innerHTML = grid([stat("sectors fetched (32 B)", nS + " <span class='muted'>(ideal " + ideal + ")</span>"), stat("128 B lines touched", nR),
          stat("bytes fetched / used", nS * 32 + " / " + nU), stat("efficiency", Math.round(eff * 100) + "%"), stat("effective H100 BW", (3.35 * eff).toFixed(2) + " TB/s")]) +
          "thread " + h + " reads element " + idxOf(h, rnd) + " → byte address " + addr[h] + " = sector " + Math.floor(addr[h] / 32) + ". " +
          (pat === "bcast" ? "All 32 threads read the same address: <b>1 sector serves the whole warp</b> (broadcast) — cheap." :
            eff > 0.99 ? "<b>Perfectly coalesced</b>: every fetched byte is used. This is what <code>a[blockIdx.x*blockDim.x + threadIdx.x]</code> gives you." :
              "Only " + Math.round(eff * 100) + "% of fetched bytes are useful, so a memory-bound kernel runs ~" + (1 / eff).toFixed(1) + "× slower than it could. " +
              (pat === "stride" ? "Typical cause: reading a column of a row-major matrix, or array-of-structs layouts. Fix: transpose via shared memory, or use struct-of-arrays." :
                pat === "offset" ? "Misalignment costs at most one extra sector per line — minor. Strides are the real killer." : "Gathers (e.g. embedding lookups, paged KV blocks) are inherently scattered; make each gathered chunk ≥ 32–128 contiguous bytes (that's why KV blocks store 16 tokens × head_dim contiguously)."));
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- tiled matmul
  V.register("tiled-matmul", {
    title: "Tiled matmul: reuse each loaded value T times via shared memory",
    desc: "C = A·B with 16×16 matrices. Naively every multiply-add loads one A and one B value from global memory (HBM). With tiling, a thread block loads a T×T tile of A and of B into shared memory once and every thread in the block reuses them → global traffic ÷ T. Step through it.",
    render: function (c, p, K) {
      var N = 16, Tt = [1, 2, 4, 8, 16].indexOf(+p.tile) >= 0 ? +p.tile : 4, s = 0, Nr = p.n || 4096;
      var tm = ticker(c, 160, function () { adv(Tt === 1 ? 16 : Tt === 2 ? 4 : 1); draw(); if (s >= total()) return false; }, function (on) { pb.textContent = on ? "⏸ Pause" : "▶ Play"; });
      c.controls.appendChild(K.select({ label: "tile T", options: [{ value: "1", label: "1 (naive)" }, "2", "4", "8", "16"], value: String(Tt), onChange: function (v) { Tt = +v; s = 0; tm.stop(); draw(); } }));
      c.controls.appendChild(K.button("Step", function () { adv(1); draw(); }));
      var pb = K.button("▶ Play", function () { if (s >= total()) s = 0; tm.toggle(); }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Finish", function () { tm.stop(); s = total(); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { tm.stop(); s = 0; draw(); }));
      c.controls.appendChild(K.select({ label: "real N (table)", options: ["1024", "4096", "8192", "16384"], value: String(Nr), onChange: function (v) { Nr = +v; draw(); } }));
      function nt() { return N / Tt; }
      function total() { return nt() * nt() * nt(); }
      function adv(k) { s = Math.min(total(), s + k); }
      function draw() {
        var W = 680, H = 372, cs = 10, n = nt();
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var A = [30, 200], B = [210, 22], C = [210, 200];
        var cur = s > 0 ? s - 1 : -1, tIdx = cur >= 0 ? Math.floor(cur / n) : -1, kk = cur >= 0 ? cur % n : -1;
        var ti = tIdx >= 0 ? Math.floor(tIdx / n) : -1, tj = tIdx >= 0 ? tIdx % n : -1, doneTiles = Math.floor(s / n);
        function mat(o, label) {
          svg.appendChild(K.s("rect", { x: o[0], y: o[1], width: N * cs, height: N * cs, class: "box" }));
          for (var i = 1; i < N; i++) {
            svg.appendChild(K.s("line", { x1: o[0] + i * cs, x2: o[0] + i * cs, y1: o[1], y2: o[1] + N * cs, class: "grid", opacity: i % Tt === 0 ? 1 : 0.35, style: { strokeWidth: i % Tt === 0 ? "1.6px" : "1px" } }));
            svg.appendChild(K.s("line", { y1: o[1] + i * cs, y2: o[1] + i * cs, x1: o[0], x2: o[0] + N * cs, class: "grid", opacity: i % Tt === 0 ? 1 : 0.35, style: { strokeWidth: i % Tt === 0 ? "1.6px" : "1px" } }));
          }
          svg.appendChild(T(K, o[0], o[1] - 5, label, "lbl"));
        }
        function rect(o, r0, c0, h, w, cls, op, strokeCls) {
          svg.appendChild(K.s("rect", { x: o[0] + c0 * cs, y: o[1] + r0 * cs, width: w * cs, height: h * cs, class: cls + (strokeCls ? " " + strokeCls : ""), opacity: op, "stroke-width": strokeCls ? 2 : 0 }));
        }
        mat(A, "A (16×16)"); mat(B, "B (16×16)"); mat(C, "C = A·B");
        for (var d = 0; d < Math.min(doneTiles, n * n); d++) rect(C, Math.floor(d / n) * Tt, (d % n) * Tt, Tt, Tt, "f3", 0.45);
        if (cur >= 0) {
          rect(A, ti * Tt, 0, Tt, N, "f1", 0.1); rect(B, 0, tj * Tt, N, Tt, "f2", 0.1);
          rect(A, ti * Tt, kk * Tt, Tt, Tt, "f1", 0.75); rect(B, kk * Tt, tj * Tt, Tt, Tt, "f2", 0.75);
          rect(C, ti * Tt, tj * Tt, Tt, Tt, "f3", 0.2 + 0.6 * (kk + 1) / n);
          svg.appendChild(K.s("rect", { x: C[0] + tj * Tt * cs - 1, y: C[1] + ti * Tt * cs - 1, width: Tt * cs + 2, height: Tt * cs + 2, fill: "none", class: "sfg", "stroke-width": 2 }));
        }
        // shared memory panel
        var px = 400;
        svg.appendChild(K.s("rect", { x: px, y: 22, width: 270, height: 150, rx: 8, class: "box" }));
        svg.appendChild(T(K, px + 8, 38, "Shared memory of the block computing this C tile", "lbl-sm"));
        var tsz = 80, tc = tsz / Tt;
        [["As (A tile)", "f1", 20], ["Bs (B tile)", "f2", 150]].forEach(function (tl) {
          svg.appendChild(T(K, px + tl[2], 58, tl[0], "lbl-sm"));
          for (var i = 0; i < Tt; i++) for (var j = 0; j < Tt; j++)
            svg.appendChild(K.s("rect", { x: px + tl[2] + j * tc, y: 64 + i * tc, width: tc - (Tt > 8 ? 0.5 : 1.5), height: tc - (Tt > 8 ? 0.5 : 1.5), class: cur >= 0 ? tl[1] : "fline", opacity: cur >= 0 ? 0.7 : 0.6 }));
        });
        svg.appendChild(T(K, px + 8, 162, Tt === 1 ? "T = 1: no reuse — every value comes from HBM" : "each loaded value is used by " + Tt + " threads (" + Tt + "× reuse)", "lbl-sm"));
        // traffic bars
        var tiled = s * 2 * Tt * Tt, naiveEq = s * 2 * Tt * Tt * Tt, naiveTot = 2 * N * N * N;
        var bx = K.scale(0, naiveTot, px + 70, px + 262);
        svg.appendChild(T(K, px, 200, "global-memory loads so far (elements)", "lbl"));
        [["naive", naiveEq, "f5"], ["tiled T=" + Tt, tiled, "f3"]].forEach(function (b, i) {
          var yy = 212 + i * 34;
          svg.appendChild(T(K, px, yy + 16, b[0], "lbl-sm"));
          svg.appendChild(K.s("rect", { x: px + 70, y: yy, width: 192, height: 24, rx: 4, class: "fbg3" }));
          svg.appendChild(K.s("rect", { x: px + 70, y: yy, width: Math.max(0, bx(b[1]) - px - 70), height: 24, rx: 4, class: b[2], opacity: 0.75 }));
          var be = px + 70 + Math.max(0, bx(b[1]) - px - 70), inside = be > px + 70 + 192 - 56;
          svg.appendChild(T(K, inside ? be - 5 : be + 5, yy + 16, b[1].toLocaleString(), "lbl mono" + (inside ? " t-white" : ""), { "text-anchor": inside ? "end" : "start" }));
        });
        svg.appendChild(T(K, px, 300, "step " + s + " / " + total() + " (one step = load 1 A tile + 1 B tile,", "lbl-sm"));
        svg.appendChild(T(K, px, 314, "then " + Tt + "×" + Tt + " threads each do " + Tt + " multiply-adds)", "lbl-sm"));
        svg.appendChild(T(K, px, 340, "total loads for the whole matmul: naive " + naiveTot.toLocaleString() + ", tiled " + (naiveTot / Tt).toLocaleString(), "lbl-sm"));
        K.clear(c.stage).appendChild(svg);
        // real-size table
        var bb = 2, fl = 2 * Math.pow(Nr, 3), rows = [], hl = -1;
        [1, 16, 32, 64, 128, 256].forEach(function (t, i) {
          var traffic = fl * bb / t + Nr * Nr * bb, smem = 2 * t * t * bb, fits = smem <= 227e3;
          var tMem = traffic / 3.35e12, tCmp = fl / 989e12;
          if (t === 128) hl = i;
          rows.push([t === 1 ? "1 (naive)" : t, K.fmtBytes(traffic), fmtAI(fl / traffic), K.fmtBytes(smem) + (fits ? "" : " ✗"), fits ? fmtT(Math.max(tMem, tCmp)) + (tMem > tCmp ? " (memory)" : " (compute)") : "doesn't fit"]);
        });
        c.readout.innerHTML = "Naive: " + K.tex("2N^3") + " loads. Tiled: " + K.tex("2N^3/T") + " — arithmetic intensity grows linearly with T. Limit: tiles must fit in shared memory (" + K.tex("2T^2 b") + " bytes ≤ 227 KB/block on H100)." +
          table(["tile T", "HBM traffic", "FLOP/B", "SMEM/block", "H100 lower bound"], rows, hl) +
          "<span class='muted'>N = " + Nr + ", bf16, " + K.fmtShort(fl) + "FLOP total; ignores L2 caching. Real kernels (cuBLAS/CUTLASS) use ~128×128 block tiles plus per-warp register tiling, double-buffered with async copies (TMA) — that's how they reach 70–80% of 989 TFLOPS.</span>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- online softmax
  V.register("online-softmax", {
    title: "Online softmax: one pass over blocks with a running max and a rescaled running sum",
    desc: "FlashAttention can't see a whole row of scores at once — it streams blocks. Keep a running max <b>m</b> and running sum <b>ℓ</b> of e<sup>x−m</sup>. When a block brings a bigger max, multiply the old sum by e<sup>m_old − m_new</sup> to re-base it. The final result equals the exact softmax.",
    render: function (c, p, K) {
      var xs = (p.values || [1.2, -0.5, 0.3, 2.1, 0.8, 3.5, -1.0, 1.7, 4.2, 0.1, 2.9, -0.3]).slice();
      var vs = [0.5, -1, 2, 0.3, 1.5, -0.4, 0.8, 1.1, -0.7, 0.2, 1.9, -1.3];
      var Bsz = [1, 2, 3, 4, 6].indexOf(+p.block) >= 0 ? +p.block : 4, s = 0, useV = false, seed = 1;
      var tm = ticker(c, 900, function () { if (s >= nb()) return false; s++; draw(); if (s >= nb()) return false; }, function (on) { pb.textContent = on ? "⏸ Pause" : "▶ Play"; });
      c.controls.appendChild(K.select({ label: "block size", options: ["1", "2", "3", "4", "6"], value: String(Bsz), onChange: function (v) { Bsz = +v; s = 0; tm.stop(); draw(); } }));
      c.controls.appendChild(K.button("Step block ▶", function () { if (s < nb()) s++; draw(); }));
      var pb = K.button("▶ Play", function () { if (s >= nb()) s = 0; tm.toggle(); }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Reset", function () { tm.stop(); s = 0; draw(); }));
      c.controls.appendChild(K.button("New vector", function () {
        seed++; var r = K.rng(seed * 17);
        xs = xs.map(function () { return +(K.randn(r) * 1.8 + 0.8).toFixed(1); }); vs = vs.map(function () { return +(K.randn(r)).toFixed(1); });
        tm.stop(); s = 0; draw();
      }));
      c.controls.appendChild(K.toggle({ label: "also accumulate output o = Σ p·v (attention)", value: useV, onChange: function (v) { useV = v; draw(); } }));
      function nb() { return Math.ceil(xs.length / Bsz); }
      function run() {
        var hist = [], m = -Infinity, l = 0, o = 0;
        for (var b0 = 0; b0 < xs.length; b0 += Bsz) {
          var blk = xs.slice(b0, b0 + Bsz), mb = Math.max.apply(null, blk), mn = Math.max(m, mb);
          var sc = m === -Infinity ? 0 : Math.exp(m - mn), bs = 0, bo = 0;
          blk.forEach(function (x, j) { var e = Math.exp(x - mn); bs += e; bo += e * vs[b0 + j]; });
          hist.push({ s: b0, e: b0 + blk.length, mOld: m, mBlk: mb, mNew: mn, sc: sc, lOld: l, bs: bs, lNew: l * sc + bs, oOld: o, bo: bo, oNew: o * sc + bo });
          m = mn; l = l * sc + bs; o = o * sc + bo;
        }
        return hist;
      }
      function f2(v) { return v === -Infinity ? "−∞" : v.toFixed(2); }
      function draw() {
        var hist = run(), n = xs.length, W = 680, H = 236, x0 = 60, bw = (W - x0 - 10) / n;
        var lo = Math.min(0, Math.min.apply(null, xs)) - 0.4, hi = Math.max.apply(null, xs) + 0.9;
        var sy = K.scale(lo, hi, 170, 26);
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        svg.appendChild(K.s("line", { x1: x0, x2: W - 10, y1: sy(0), y2: sy(0), class: "axis" }));
        svg.appendChild(T(K, 4, sy(0) + 4, "scores x", "lbl-sm"));
        var cur = s > 0 ? hist[s - 1] : null, st = s > 0 ? hist[s - 1] : null;
        for (var b = 0; b < nb(); b++) {
          var h = hist[b], bx = x0 + h.s * bw;
          if (b) svg.appendChild(K.s("line", { x1: bx, x2: bx, y1: 18, y2: 176, class: "ln-thin smuted dash" }));
          svg.appendChild(T(K, bx + (h.e - h.s) * bw / 2, 16, "block " + b, b === s - 1 ? "lbl" : "lbl-sm", { "text-anchor": "middle" }));
        }
        xs.forEach(function (x, i) {
          var bi = Math.floor(i / Bsz), cls = bi < s - 1 ? "f1" : bi === s - 1 ? "f2" : "fmuted";
          var y0 = sy(0), y1 = sy(x);
          svg.appendChild(K.s("rect", { x: x0 + i * bw + 4, y: Math.min(y0, y1), width: bw - 8, height: Math.max(1, Math.abs(y1 - y0)), class: cls, opacity: bi < s ? 0.8 : 0.35 }));
          svg.appendChild(T(K, x0 + i * bw + bw / 2, (x >= 0 ? y1 - 4 : y1 + 12), x.toFixed(1), "lbl-sm mono", { "text-anchor": "middle" }));
        });
        if (st) {
          var xe = x0 + st.e * bw;
          if (st.mOld !== -Infinity && st.mOld < st.mNew) {
            svg.appendChild(K.s("line", { x1: x0, x2: x0 + st.s * bw, y1: sy(st.mOld), y2: sy(st.mOld), class: "ln-thin s4 dash" }));
            var yo = sy(st.mOld);
            if (Math.abs(yo - sy(0)) < 24) yo = sy(0) - 24;
            svg.appendChild(T(K, 4, yo - 2, "old m", "lbl-sm", { style: col(4) }));
            svg.appendChild(T(K, 4, yo + 10, st.mOld.toFixed(2), "lbl-sm mono", { style: col(4) }));
          }
          svg.appendChild(K.s("line", { x1: x0, x2: xe, y1: sy(st.mNew), y2: sy(st.mNew), class: "ln s5 dash" }));
          svg.appendChild(T(K, xe + 4, sy(st.mNew) + 4, "m = " + st.mNew.toFixed(2), "lbl", { style: col(5) }));
        }
        svg.appendChild(T(K, 4, 196, "e^(x−m)", "lbl-sm"));
        var done = s >= nb(), fin = hist[hist.length - 1], trueP = K.softmax(xs);
        if (done) svg.appendChild(T(K, 4, 222, "p online", "lbl-sm"));
        xs.forEach(function (x, i) {
          if (Math.floor(i / Bsz) >= s) return;
          svg.appendChild(T(K, x0 + i * bw + bw / 2, 196, Math.exp(x - st.mNew).toFixed(3), "lbl-sm mono", { "text-anchor": "middle" }));
          if (done) svg.appendChild(T(K, x0 + i * bw + bw / 2, 222, (Math.exp(x - fin.mNew) / fin.lNew).toFixed(3), "lbl mono", { "text-anchor": "middle" }));
        });
        K.clear(c.stage).appendChild(svg);
        var head = ["block", "block max", "m: old → new", "rescale e<sup>m_old − m_new</sup>", "ℓ = ℓ_old·rescale + Σ e<sup>x−m</sup>"];
        if (useV) head.push("o = o_old·rescale + Σ e<sup>x−m</sup>·v");
        var rows = hist.slice(0, s).map(function (h, i) {
          var r = [i, h.mBlk.toFixed(2), f2(h.mOld) + " → " + h.mNew.toFixed(2),
            h.mOld === -Infinity ? "— (first block)" : h.sc.toFixed(4) + (h.sc < 1 ? " ⚠ re-base" : ""),
            h.lOld.toFixed(3) + "·" + (h.mOld === -Infinity ? "0" : h.sc.toFixed(3)) + " + " + h.bs.toFixed(3) + " = <b>" + h.lNew.toFixed(3) + "</b>"];
          if (useV) r.push(h.oOld.toFixed(3) + "·" + (h.mOld === -Infinity ? "0" : h.sc.toFixed(3)) + " + " + h.bo.toFixed(3) + " = <b>" + h.oNew.toFixed(3) + "</b>");
          return r;
        });
        var tb = K.h("div", { style: { overflowX: "auto" }, html: s ? table(head, rows, s - 1) : "<span class='muted' style='font-size:13px'>Press “Step block” to process the first block. State starts at m = −∞, ℓ = 0.</span>" });
        c.stage.appendChild(tb);
        var err = 0; if (done) xs.forEach(function (x, i) { err = Math.max(err, Math.abs(Math.exp(x - fin.mNew) / fin.lNew - trueP[i])); });
        var exactO = trueP.reduce(function (a, pp, i) { return a + pp * vs[i]; }, 0);
        c.readout.innerHTML = K.tex("m_{new} = \\max(m_{old}, \\max_j x_j),\\quad \\ell_{new} = \\ell_{old}\\, e^{m_{old}-m_{new}} + \\sum_j e^{x_j - m_{new}}") + "<br>" +
          (done ? "<b>Done.</b> " + K.tex("p_i = e^{x_i - m}/\\ell") + " — max difference from the exact (two-pass) softmax: <b>" + err.toExponential(1) + "</b> (floating-point noise). " +
            (useV ? "Output " + K.tex("o/\\ell = " + (fin.oNew / fin.lNew).toFixed(4)) + " vs exact " + K.tex("\\sum_i p_i v_i = " + exactO.toFixed(4)) + ". " : "") +
            "We touched each score once and never stored the whole row — that is what lets FlashAttention keep everything in SRAM."
            : "After " + s + "/" + nb() + " blocks: m = " + (st ? st.mNew.toFixed(2) : "−∞") + ", ℓ = " + (st ? st.lNew.toFixed(3) : "0") + ". Subtracting the max keeps every exponent ≤ 0, so " + K.tex("e^{x-m}") + " never overflows (fp16 overflows at " + K.tex("e^{11}") + "). " +
            (useV ? "The output accumulator is rescaled by the same factor, so no score is ever revisited." : "Toggle “accumulate output” to see how FlashAttention also rescales its output accumulator."));
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- kernel fusion
  var LL8 = { d: 4096, ffn: 14336, h: 32, hd: 128 };
  var CHAINS = {
    norm: { name: "residual add + RMSNorm + ×γ", ops: function (t) { var S = t * LL8.d; return [
      { n: "add x + r", r: 2 * S, w: S, f: S }, { n: "mean(x²)", r: S, w: t, f: 2 * S }, { n: "x · rsqrt(ms+ε)", r: S + t, w: S, f: 2 * S }, { n: "× γ", r: S + LL8.d, w: S, f: S }]; },
      fused: function (t) { var S = t * LL8.d; return { n: "fused_add_rms_norm", r: 2 * S + LL8.d, w: 2 * S, f: 6 * S }; },
      note: "The fused kernel writes the normalized output and the updated residual (needed by the next layer). vLLM ships exactly this as fused_add_rms_norm." },
    swiglu: { name: "SwiGLU: silu(gate) × up", ops: function (t) { var S = t * LL8.ffn; return [
      { n: "silu(gate)", r: S, w: S, f: 4 * S }, { n: "× up", r: 2 * S, w: S, f: S }]; },
      fused: function (t) { var S = t * LL8.ffn; return { n: "silu_and_mul", r: 2 * S, w: S, f: 5 * S }; },
      note: "ffn = 14336 for Llama-3.1-8B. Fusing saves writing and re-reading silu(gate). vLLM: silu_and_mul; torch.compile finds this automatically." },
    attn: { name: "attention: naive vs FlashAttention", ops: function (t) { var q = t * LL8.h * LL8.hd, P = t * t * LL8.h; return [
      { n: "S = Q·Kᵀ", r: 2 * q, w: P, f: 2 * t * t * LL8.h * LL8.hd }, { n: "scale + mask", r: P, w: P, f: 2 * P }, { n: "softmax", r: P, w: P, f: 5 * P }, { n: "O = P·V", r: P + q, w: q, f: 2 * t * t * LL8.h * LL8.hd }]; },
      fused: function (t) { var q = t * LL8.h * LL8.hd, P = t * t * LL8.h; return { n: "FlashAttention", r: 3 * q, w: q, f: 4 * t * t * LL8.h * LL8.hd + 7 * P }; },
      note: "Naive attention materializes the T×T score matrix per head in HBM (written and read 3×). FlashAttention tiles Q/K/V through SRAM with online softmax and never writes it. (MHA, 32 heads × 128, prefill of T tokens.)" },
  };
  V.register("kernel-fusion", {
    title: "Kernel fusion: stop round-tripping intermediates through HBM",
    desc: "Each separate kernel reads its inputs from HBM and writes its output back. For memory-bound ops that traffic <i>is</i> the runtime. A fused kernel reads the inputs once, keeps intermediates in registers/shared memory, and writes once. Llama-3.1-8B shapes, bf16.",
    render: function (c, p, K) {
      var ck = CHAINS[p.chain] ? p.chain : "norm", lt = p.tokens ? Math.log2(p.tokens) : 12, gk = GPUS[p.gpu] ? p.gpu : "H100", launch = true, LAUNCH = 4e-6;
      c.controls.appendChild(K.select({ label: "ops", options: Object.keys(CHAINS).map(function (k) { return { value: k, label: CHAINS[k].name }; }), value: ck, onChange: function (v) { ck = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "tokens T", min: 0, max: 14, step: 1, value: lt, fmt: function (v) { return Math.pow(2, v).toLocaleString(); }, onInput: function (v) { lt = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "GPU", options: gpuOptions(), value: gk, onChange: function (v) { gk = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "+ ~4 µs launch per kernel", value: launch, onChange: function (v) { launch = v; draw(); } }));
      function time(o, g) { var mem = (o.r + o.w) * 2 / (g.bw * 1e12), cmp = o.f / (g.peak * 1e12); return { t: Math.max(mem, cmp) + (launch ? LAUNCH : 0), mem: mem >= cmp }; }
      function draw() {
        var t = Math.pow(2, lt), ch = CHAINS[ck], ops = ch.ops(t), fu = ch.fused(t), g = GPUS[gk];
        var W = 680, H = 338, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var n = ops.length, gap = 14, bw = (640 - (n - 1) * gap) / n;
        svg.appendChild(T(K, 20, 13, "Unfused: " + n + " kernels, every intermediate goes to HBM and back", "lbl"));
        var HY = 88;
        svg.appendChild(K.s("rect", { x: 20, y: HY, width: 640, height: 20, rx: 5, class: "f4", opacity: 0.75 }));
        svg.appendChild(T(K, 340, HY + 14, "HBM — " + g.bw + " TB/s", "lbl t-white", { "text-anchor": "middle" }));
        function arrows(x, yTop, yBot, r, w, up) {
          // read: HBM → kernel, write: kernel → HBM
          var yk = up ? yBot : yTop, yh = up ? yTop : yBot;
          svg.appendChild(K.s("line", { x1: x - 14, x2: x - 14, y1: yh, y2: yk, class: "ln s1" }));
          svg.appendChild(K.s("path", { d: "M" + (x - 18) + " " + (yk + (up ? -5 : 5)) + " L" + (x - 14) + " " + yk + " L" + (x - 10) + " " + (yk + (up ? -5 : 5)), class: "ln-thin s1" }));
          svg.appendChild(K.s("line", { x1: x + 14, x2: x + 14, y1: yk, y2: yh, class: "ln s2" }));
          svg.appendChild(K.s("path", { d: "M" + (x + 10) + " " + (yh + (up ? 5 : -5)) + " L" + (x + 14) + " " + yh + " L" + (x + 18) + " " + (yh + (up ? 5 : -5)), class: "ln-thin s2" }));
          svg.appendChild(T(K, x - 18, (yTop + yBot) / 2 + 4, "R " + K.fmtBytes(r * 2), "lbl-sm", { "text-anchor": "end" }));
          svg.appendChild(T(K, x + 18, (yTop + yBot) / 2 + 4, "W " + K.fmtBytes(w * 2), "lbl-sm"));
        }
        ops.forEach(function (o, i) {
          var x = 20 + i * (bw + gap);
          svg.appendChild(K.s("rect", { x: x, y: 20, width: bw, height: 32, rx: 6, class: "f" + (i + 1), opacity: 0.8 }));
          svg.appendChild(T(K, x + bw / 2, 40, o.n, "lbl t-white", { "text-anchor": "middle" }));
          arrows(x + bw / 2, 52, HY, o.r, o.w, false);
        });
        svg.appendChild(K.s("rect", { x: 20, y: 146, width: 640, height: 34, rx: 6, class: "box-hl" }));
        svg.appendChild(T(K, 30, 160, "Fused: 1 kernel (" + fu.n + ")", "lbl"));
        svg.appendChild(T(K, 30, 174, "intermediates stay in registers / shared memory", "lbl-sm"));
        var px = 290, pw = (360 - (n - 1) * 6) / n;
        ops.forEach(function (o, i) {
          svg.appendChild(K.s("rect", { x: px + i * (pw + 6), y: 152, width: pw, height: 22, rx: 4, class: "f" + (i + 1), opacity: 0.7 }));
          svg.appendChild(T(K, px + i * (pw + 6) + pw / 2, 167, o.n, "lbl-sm t-white", { "text-anchor": "middle" }));
        });
        arrows(200, HY + 20, 146, fu.r, fu.w, true);
        // bars
        var tu = 0, bu = 0, segs = ops.map(function (o) { var tt = time(o, g); tu += tt.t; bu += (o.r + o.w) * 2; return { t: tt.t, b: (o.r + o.w) * 2 }; });
        var tf = time(fu, g).t, bf = (fu.r + fu.w) * 2;
        var xs = K.scale(0, 1, 150, 560), y0 = 202;
        [["HBM traffic", "unfused", bu, bf, "b", function (v) { return K.fmtBytes(v); }], ["time", "unfused", tu, tf, "t", fmtT]].forEach(function (row, ri) {
          var yy = y0 + ri * 68, mx = row[2];
          svg.appendChild(T(K, 20, yy + 15, row[0] + " unfused", "lbl-sm"));
          var acc = 0;
          segs.forEach(function (sg, i) {
            var v = sg[row[4]] / mx;
            svg.appendChild(K.s("rect", { x: xs(acc), y: yy, width: Math.max(0.5, xs(acc + v) - xs(acc) - 1), height: 24, class: "f" + (i + 1), opacity: 0.8 }));
            acc += v;
          });
          svg.appendChild(T(K, xs(1) + 6, yy + 16, row[5](row[2]), "lbl mono"));
          svg.appendChild(T(K, 20, yy + 45, row[0] + " fused", "lbl-sm"));
          svg.appendChild(K.s("rect", { x: xs(0), y: yy + 30, width: Math.max(1, xs(row[3] / mx) - xs(0)), height: 24, class: "f3", opacity: 0.85 }));
          svg.appendChild(T(K, xs(row[3] / mx) + 6, yy + 46, row[5](row[3]) + "  (" + (row[2] / row[3]).toFixed(1) + "× less)", "lbl mono"));
        });
        K.clear(c.stage).appendChild(svg);
        var extra = "";
        if (ck === "attn") {
          var scoreB = t * t * LL8.h * 2;
          extra = " Score matrix alone: <b>" + K.fmtBytes(scoreB) + "</b> per layer" + (scoreB > g.mem * 1e9 * 0.5 ? " — <b style='color:var(--c5)'>doesn't even fit next to the model on " + g.name + "</b>" : "") + "; FlashAttention's extra memory is O(T).";
        }
        c.readout.innerHTML = grid([stat("kernels", n + " → 1"), stat("HBM traffic", K.fmtBytes(bu) + " → " + K.fmtBytes(bf)), stat("est. time", fmtT(tu) + " → " + fmtT(tf)), stat("speedup", (tu / tf).toFixed(1) + "×")]) +
          "Model: " + K.tex("t_{kernel} \\approx \\max\\left(\\frac{\\text{bytes}}{\\beta}, \\frac{\\text{FLOPs}}{\\pi}\\right) + t_{launch}") + " per kernel on " + g.name + ". " + ch.note + extra +
          (t <= 16 && launch && ck !== "attn" ? " <span class='muted'>At tiny T (decode) the ~4 µs launch cost dominates — see launch-overhead / CUDA graphs.</span>" : "");
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- launch overhead & CUDA graphs
  V.register("launch-overhead", {
    title: "Launch overhead: small kernels starve the GPU — CUDA graphs fix it",
    desc: "In eager mode the CPU (Python → PyTorch → CUDA driver) spends several µs launching each kernel. If a kernel runs for less time than it takes to launch the next one, the GPU sits idle between kernels. A CUDA graph records the whole sequence once and replays it with a single launch.",
    render: function (c, p, K) {
      var N = p.kernels || 12, g = p.kernel_us || 3, L = p.launch_us || 8, G = 6, GAP = 0.5, t = Infinity;
      var tm = ticker(c, 30, function () { var tot = totals(); t += tot.max / 150; if (t >= tot.max) { t = Infinity; draw(); return false; } draw(); }, function (on) { pb.textContent = on ? "⏸ Pause" : "▶ Play"; });
      c.controls.appendChild(K.slider({ label: "kernels", min: 3, max: 40, value: N, onInput: function (v) { N = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "GPU time / kernel", min: 1, max: 40, step: 0.5, value: g, fmt: function (v) { return v + " µs"; }, onInput: function (v) { g = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "CPU launch cost", min: 1, max: 30, step: 0.5, value: L, fmt: function (v) { return v + " µs"; }, onInput: function (v) { L = v; draw(); } }));
      var pb = K.button("▶ Play", function () { if (!tm.running() && t === Infinity) t = 0; tm.toggle(); }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Reset", function () { tm.stop(); t = Infinity; draw(); }));
      function sim() {
        var ek = [], end = 0;
        for (var i = 0; i < N; i++) { var st = Math.max((i + 1) * L, end); ek.push([st, st + g, i * L]); end = st + g; }
        var gk = []; for (var j = 0; j < N; j++) { var s2 = G + j * (g + GAP); gk.push([s2, s2 + g]); }
        return { ek: ek, gk: gk, eager: end, graph: G + N * g + (N - 1) * GAP };
      }
      function totals() { var s = sim(); return { max: Math.max(s.eager, s.graph) * 1.03 }; }
      function draw() {
        var S = sim(), tmax = Math.max(S.eager, S.graph) * 1.03, now = Math.min(t, tmax);
        var W = 680, H = 214, x0 = 112, sx = K.scale(0, tmax, x0, W - 12);
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        function box(s, e, y, h, cls, op, label, alt) {
          if (now <= s) return;
          var ee = Math.min(e, now);
          svg.appendChild(K.s("rect", { x: sx(s), y: y, width: Math.max(0.8, sx(ee) - sx(s) - 0.6), height: h, rx: 2, class: cls, opacity: op }));
          if (label != null && sx(ee) - sx(s) < String(label).length * 5.2 + 4) label = alt;
          if (label != null && sx(ee) - sx(s) > 14) svg.appendChild(T(K, (sx(s) + sx(ee)) / 2, y + h / 2 + 4, label, "lbl-sm t-white", { "text-anchor": "middle" }));
        }
        [["Eager", 8, S.ek], ["CUDA graph", 100, S.gk]].forEach(function (lane, li) {
          var y = lane[1];
          svg.appendChild(T(K, 4, y + 12, lane[0], "lbl"));
          svg.appendChild(T(K, 4, y + 34, "CPU", "lbl-sm")); svg.appendChild(T(K, 4, y + 62, "GPU", "lbl-sm"));
          svg.appendChild(K.s("rect", { x: x0, y: y + 22, width: W - 12 - x0, height: 20, class: "fbg3" }));
          svg.appendChild(K.s("rect", { x: x0, y: y + 50, width: W - 12 - x0, height: 20, class: "fbg3" }));
          if (li === 0) {
            S.ek.forEach(function (k, i) {
              box(k[2], k[2] + L, y + 22, 20, i % 2 ? "f4" : "f6", 0.8, "L");
              var prevEnd = i ? S.ek[i - 1][1] : 0;
              if (k[0] > prevEnd) box(prevEnd, k[0], y + 50, 20, "f5", 0.18);
              box(k[0], k[1], y + 50, 20, "f1", 0.85, i);
            });
          } else {
            box(0, G, y + 22, 20, "f4", 0.8, "graph launch", "G");
            box(0, G, y + 50, 20, "f5", 0.18);
            S.gk.forEach(function (k, i) { box(k[0], k[1], y + 50, 20, "f3", 0.85, i); });
          }
          var tot = li === 0 ? S.eager : S.graph;
          if (now >= tot) svg.appendChild(T(K, Math.min(sx(tot) + 4, W - 70), y + 12, fmtT(tot * 1e-6), "lbl mono"));
        });
        K.niceTicks(0, tmax, 8).forEach(function (v) {
          svg.appendChild(K.s("line", { x1: sx(v), x2: sx(v), y1: 186, y2: 191, class: "axis" }));
          svg.appendChild(T(K, sx(v), 204, v + " µs", "tick", { "text-anchor": "middle" }));
        });
        svg.appendChild(K.s("line", { x1: x0, x2: W - 12, y1: 186, y2: 186, class: "axis" }));
        if (t !== Infinity) svg.appendChild(K.s("line", { x1: sx(now), x2: sx(now), y1: 6, y2: 190, class: "ln s2" }));
        K.clear(c.stage).appendChild(svg);
        var busyE = N * g / S.eager, busyG = N * g / S.graph;
        var K8 = 32 * 15, eStep = Math.max(K8 * L, K8 * g), gStep = G + K8 * (g + GAP);
        c.readout.innerHTML = grid([stat("eager total", fmtT(S.eager * 1e-6)), stat("eager GPU busy", Math.round(busyE * 100) + "%"), stat("graph total", fmtT(S.graph * 1e-6)), stat("graph GPU busy", Math.round(busyG * 100) + "%"), stat("speedup", (S.eager / S.graph).toFixed(2) + "×")]) +
          (L > g ? "<b>CPU-bound:</b> launch cost (" + L + " µs) > kernel time (" + g + " µs), so the GPU waits " + (L - g).toFixed(1) + " µs after every kernel (pink gaps). Faster kernels would change nothing. "
            : "<b>GPU-bound:</b> kernels are longer than launches, so the CPU runs ahead and queues work — overhead is hidden. ") +
          "Real scale: a Llama-3.1-8B decode step is ≈ 32 layers × ~15 kernels ≈ " + K8 + " launches → eager ≈ " + fmtT(eStep * 1e-6) + " vs graph ≈ " + fmtT(gStep * 1e-6) + " per token with these settings. " +
          "<span class='muted'>That is why vLLM/SGLang capture decode steps as CUDA graphs (per batch size) and why <code>torch.compile(mode=\"reduce-overhead\")</code> exists. Prefill kernels are long, so it matters much less there.</span>";
      }
      draw();
    },
  });
})();
