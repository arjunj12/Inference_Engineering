/* distributed.js — multi-GPU parallelism, collectives, pipeline bubbles, α–β comm cost, disaggregation. */
(function () {
  "use strict";
  var V = window.Viz;
  var uid = 0;

  // Model presets (params P, layers L, hidden d, heads H, KV heads kvh, head_dim hd, ffn).
  // MoE: E experts, topk, moeL = MoE layers, Pexp = params in routed experts, Pa = active params. mla = latent KV width per layer.
  var MODELS = {
    "Llama-3.1-8B": { P: 8.03e9, L: 32, d: 4096, H: 32, kvh: 8, hd: 128, ffn: 14336 },
    "Mistral-7B": { P: 7.24e9, L: 32, d: 4096, H: 32, kvh: 8, hd: 128, ffn: 14336 },
    "Qwen2.5-7B": { P: 7.62e9, L: 28, d: 3584, H: 28, kvh: 4, hd: 128, ffn: 18944 },
    "Llama-3.1-70B": { P: 70.6e9, L: 80, d: 8192, H: 64, kvh: 8, hd: 128, ffn: 28672 },
    "Mixtral-8x7B": { P: 46.7e9, Pa: 12.9e9, L: 32, d: 4096, H: 32, kvh: 8, hd: 128, ffn: 14336, E: 8, topk: 2, moeL: 32, Pexp: 45.1e9 },
    "DeepSeek-V3": { P: 671e9, Pa: 37e9, L: 61, d: 7168, H: 128, mla: 576, E: 256, topk: 8, moeL: 58, Pexp: 654e9 },
  };
  // One-way bandwidth (bytes/s) and per-message latency α (s). Approximate, for intuition.
  var LINKS = [
    { id: "nvlink", name: "NVLink 4 (H100)", bw: 450e9, a: 3e-6, note: "900 GB/s total = 450 GB/s each way" },
    { id: "pcie", name: "PCIe Gen5 x16", bw: 64e9, a: 5e-6, note: "64 GB/s each way" },
    { id: "ib", name: "InfiniBand NDR", bw: 50e9, a: 8e-6, note: "400 Gb/s = 50 GB/s per NIC" },
    { id: "roce", name: "Ethernet 100 GbE", bw: 12.5e9, a: 15e-6, note: "100 Gb/s = 12.5 GB/s (RoCE)" },
    { id: "eth", name: "Ethernet 10 GbE", bw: 1.25e9, a: 50e-6, note: "10 Gb/s TCP" },
  ];
  var FILL = ["f1", "f2", "f3", "f4", "f5", "f6", "ffg", "fmuted"];

  function linkBy(id) { for (var i = 0; i < LINKS.length; i++) if (LINKS[i].id === id) return LINKS[i]; return LINKS[0]; }
  function kvLayer(m, b) { return m.mla ? m.mla * b : 2 * m.kvh * m.hd * b; }
  function fmtT(s) {
    var a = Math.abs(s);
    if (a === 0) return "0";
    if (a < 1e-3) return +(s * 1e6).toPrecision(3) + " µs";
    if (a < 1) return +(s * 1e3).toPrecision(3) + " ms";
    if (a < 120) return +s.toPrecision(3) + " s";
    return +(s / 60).toPrecision(3) + " min";
  }
  function fmtGB(b) { return (b / 1e9).toFixed(b < 10e9 ? 1 : 0) + " GB"; }
  function stats(items) {
    return "<div class='stat-grid'>" + items.map(function (it) {
      return "<div class='stat'><small>" + it[0] + "</small><b" + (it[2] ? " style='color:" + it[2] + "'" : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  function txt(K, x, y, t, cls, anchor, extra) {
    var a = { x: x, y: y, class: cls || "lbl-sm", text: t };
    if (anchor) a["text-anchor"] = anchor;
    if (extra) Object.keys(extra).forEach(function (k) { a[k] = extra[k]; });
    return K.s("text", a);
  }
  function arrowMarker(K, svg, cls) {
    var id = "dz-arrow-" + (++uid);
    var defs = K.s("defs");
    var mk = K.s("marker", { id: id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
    mk.appendChild(K.s("path", { d: "M0 0 L10 5 L0 10 z", class: cls || "fmuted" }));
    defs.appendChild(mk);
    svg.appendChild(defs);
    return "url(#" + id + ")";
  }
  function fmtTok(n) { return n >= 1024 ? (n / 1024).toFixed(n % 1024 ? 1 : 0).replace(/\.0$/, "") + "k" : String(n); }

  // ---------------------------------------------------------------- parallelism
  V.register("parallelism", {
    title: "Five ways to split a model across GPUs: DP · TP · PP · EP · CP",
    desc: "Pick a strategy and a GPU count. Each box shows what a GPU holds (layers, weight slices, experts, sequence chunks); arrows show what must be communicated. The readout gives memory per GPU and bytes sent per token for a real model.",
    render: function (c, p, K) {
      var strat = String(p.strategy || "TP").toUpperCase(), N = +p.gpus || 4;
      var mname = MODELS[p.model] ? p.model : "Llama-3.1-70B";
      var bytes = 2, batch = p.batch || 16, ctx = p.ctx || 8192, cap = 80;
      if (["DP", "TP", "PP", "EP", "CP"].indexOf(strat) < 0) strat = "TP";
      if ([2, 4, 8].indexOf(N) < 0) N = 4;
      var C = c.controls;
      C.appendChild(K.select({ label: "strategy", value: strat, options: [
        { value: "DP", label: "DP — data parallel (replicas)" },
        { value: "TP", label: "TP — tensor parallel" },
        { value: "PP", label: "PP — pipeline parallel" },
        { value: "EP", label: "EP — expert parallel (MoE)" },
        { value: "CP", label: "CP — context parallel" }], onChange: function (v) { strat = v; draw(); } }));
      C.appendChild(K.select({ label: "GPUs", value: String(N), options: ["2", "4", "8"], onChange: function (v) { N = +v; draw(); } }));
      C.appendChild(K.select({ label: "model", value: mname, options: Object.keys(MODELS), onChange: function (v) { mname = v; draw(); } }));
      C.appendChild(K.select({ label: "weights", value: "2", options: [{ value: "2", label: "BF16" }, { value: "1", label: "FP8" }], onChange: function (v) { bytes = +v; draw(); } }));
      C.appendChild(K.select({ label: "GPU memory", value: "80", options: [{ value: "80", label: "80 GB (H100/A100)" }, { value: "141", label: "141 GB (H200)" }, { value: "192", label: "192 GB (B200)" }], onChange: function (v) { cap = +v; draw(); } }));
      C.appendChild(K.slider({ label: "concurrent seqs", min: 1, max: 128, value: batch, onInput: function (v) { batch = v; draw(); } }));
      C.appendChild(K.select({ label: "context", value: String(ctx), options: [{ value: "2048", label: "2k" }, { value: "8192", label: "8k" }, { value: "32768", label: "32k" }, { value: "131072", label: "128k" }], onChange: function (v) { ctx = +v; draw(); } }));

      function compute() {
        var m = MODELS[mname], moe = !!m.E;
        var Wt = m.P * bytes, kvTok = kvLayer(m, 2) * m.L; // KV cache kept in BF16
        var KV = batch * ctx * kvTok, r = { m: m, moe: moe, KV: KV, kvTok: kvTok, kvRep: 1 };
        var nvl = linkBy("nvlink"), ib = linkBy("ib"), act = m.d * 2;
        if (strat === "DP") { r.w = Wt; r.kv = KV / N; r.tok = 0; r.nColl = 0; r.coll = "none"; }
        else if (strat === "TP") {
          r.w = Wt / N;
          r.kvRep = m.mla ? N : Math.max(1, N / m.kvh);
          r.kv = KV / N * r.kvRep;
          r.tok = m.L * 2 * 2 * (N - 1) / N * act; // 2 ring all-reduces per layer
          r.nColl = 2 * m.L; r.coll = "all-reduce";
        } else if (strat === "PP") { r.w = Wt / N; r.kv = KV / N; r.tok = act; r.nColl = N - 1; r.coll = "send/recv"; }
        else if (strat === "EP") {
          if (moe) { r.w = (m.P - m.Pexp) * bytes + m.Pexp * bytes / N; r.tok = m.moeL * 2 * m.topk * act * (N - 1) / N; r.nColl = 2 * m.moeL; }
          else { r.w = Wt; r.tok = 0; r.nColl = 0; }
          r.kv = KV / N; r.coll = "all-to-all";
        } else { r.w = Wt; r.kv = KV / N; r.tok = kvTok * (N - 1) / N; r.nColl = m.L * (N - 1); r.coll = "ring send/recv"; }
        r.total = r.w + r.kv;
        r.stepTokens = strat === "CP" ? ctx : batch;
        if (strat === "PP") {
          r.tNv = (N - 1) * (nvl.a + batch * act / nvl.bw); r.tIb = (N - 1) * (ib.a + batch * act / ib.bw);
        } else {
          var sb = r.tok * r.stepTokens;
          r.tNv = r.nColl ? r.nColl * nvl.a + sb / nvl.bw : 0; r.tIb = r.nColl ? r.nColl * ib.a + sb / ib.bw : 0;
        }
        return r;
      }

      function content(svg, i, x, y, w, r) {
        var m = r.m, cap8 = "";
        function band(k, n, bh, cls, extra) {
          var a = { x: x, y: y + k * (bh + 2), width: w, height: bh, rx: 2, class: cls };
          if (extra) Object.keys(extra).forEach(function (kk) { a[kk] = extra[kk]; });
          svg.appendChild(K.s("rect", a));
        }
        if (strat === "DP" || (strat === "EP" && !r.moe)) {
          for (var k = 0; k < 8; k++) band(k, 8, 10, "f1", { opacity: 0.85 });
          cap8 = strat === "EP" ? "dense: no experts" : "full copy";
        } else if (strat === "TP") {
          for (var k2 = 0; k2 < 8; k2++) {
            band(k2, 8, 10, "fline");
            svg.appendChild(K.s("rect", { x: x + w * i / N, y: y + k2 * 12, width: w / N, height: 10, class: "f1" }));
          }
          cap8 = "1/" + N + " of every W";
        } else if (strat === "PP") {
          var lo = Math.round(i * m.L / N), hi = Math.round((i + 1) * m.L / N) - 1;
          for (var k3 = 0; k3 < 8; k3++) {
            if (Math.floor(k3 * N / 8) === i) band(k3, 8, 10, "f1", { opacity: 0.85 });
            else band(k3, 8, 10, "box", { "stroke-dasharray": "3 3", opacity: 0.6 });
          }
          cap8 = "layers " + lo + "–" + hi;
        } else if (strat === "EP") {
          for (var k4 = 0; k4 < 3; k4++) band(k4, 3, 10, "f1", { opacity: 0.85 });
          svg.appendChild(txt(K, x + w / 2, y + 44, "experts", "lbl-sm", "middle"));
          var cells = Math.min(m.E, 16), cols = 4, rows = Math.ceil(cells / cols), cw = (w - (cols - 1) * 3) / cols, chh = Math.min(10, (46 - (rows - 1) * 3) / rows);
          for (var e = 0; e < cells; e++) {
            var own = Math.floor(e * N / cells) === i;
            svg.appendChild(K.s("rect", { x: x + (e % cols) * (cw + 3), y: y + 50 + Math.floor(e / cols) * (chh + 3), width: cw, height: chh, rx: 2, class: own ? "f3" : "fline" }));
          }
          var per = m.E / N;
          cap8 = per === 1 ? "expert E" + i : "E" + Math.round(i * per) + "–" + Math.round((i + 1) * per - 1) + " (" + per + ")";
        } else {
          for (var k5 = 0; k5 < 5; k5++) band(k5, 5, 10, "f1", { opacity: 0.85 });
          svg.appendChild(txt(K, x + w / 2, y + 72, "sequence", "lbl-sm", "middle"));
          var tw = w / 16;
          for (var t = 0; t < 16; t++) {
            var mine = Math.floor(t * N / 16) === i;
            svg.appendChild(K.s("rect", { x: x + t * tw + 0.5, y: y + 78, width: tw - 1, height: 16, class: mine ? FILL[i % 6] : "fline" }));
          }
          cap8 = "tok " + fmtTok(Math.round(i * ctx / N)) + "–" + fmtTok(Math.round((i + 1) * ctx / N));
        }
        return cap8;
      }

      function draw() {
        var r = compute(), m = r.m;
        var W = 720, H = 360, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var mk = arrowMarker(K, svg, "fmuted");
        var gap = 14, x0 = 20, gw = (W - 2 * x0 - (N - 1) * gap) / N, y0 = 84, gh = 180, capB = cap * 1e9;
        function gx(i) { return x0 + i * (gw + gap); }
        function cx(i) { return gx(i) + gw / 2; }
        var line = function (x1, y1, x2, y2, both) { return K.s("line", { x1: x1, y1: y1, x2: x2, y2: y2, class: "ln-thin smuted", "marker-end": mk, "marker-start": both ? mk : null }); };
        // communication pattern
        if (strat === "DP" || strat === "TP") {
          var bw = 330, bx = W / 2 - bw / 2;
          svg.appendChild(K.s("rect", { x: bx, y: 8, width: bw, height: 26, rx: 6, class: "box-hl" }));
          svg.appendChild(txt(K, W / 2, 25, strat === "DP" ? "load balancer: different requests → different replicas" : "same input activations x broadcast to every GPU", "lbl", "middle"));
          for (var i = 0; i < N; i++) svg.appendChild(line(W / 2, 34, cx(i), y0 - 2));
          if (strat === "TP") {
            var by = y0 + gh + 26;
            svg.appendChild(K.s("line", { x1: cx(0), x2: cx(N - 1), y1: by, y2: by, class: "s2", "stroke-width": 3 }));
            for (var j = 0; j < N; j++) svg.appendChild(K.s("line", { x1: cx(j), x2: cx(j), y1: y0 + gh + 2, y2: by, class: "ln-thin s2", "marker-start": mk }));
            svg.appendChild(txt(K, W / 2, by + 20, "all-reduce (sum of partial outputs) — 2× per layer: after attention and after the MLP", "lbl", "middle", { style: "fill:var(--c2)" }));
          } else {
            svg.appendChild(txt(K, W / 2, y0 + gh + 30, "No GPU↔GPU traffic during inference — replicas are independent.", "lbl", "middle"));
            svg.appendChild(txt(K, W / 2, y0 + gh + 48, "(Training with DP adds one all-reduce of the gradients per optimizer step.)", "lbl-sm", "middle"));
          }
        } else if (strat === "PP") {
          svg.appendChild(txt(K, x0, 20, "tokens enter GPU 0 → activations hop stage → stage → logits leave GPU " + (N - 1), "lbl"));
          for (var k = 0; k < N - 1; k++) {
            var mid = (cx(k) + cx(k + 1)) / 2;
            svg.appendChild(K.s("path", { d: "M" + cx(k) + " " + (y0 - 2) + " Q" + mid + " " + (y0 - 48) + " " + cx(k + 1) + " " + (y0 - 2), class: "ln-thin s2", "marker-end": mk }));
            svg.appendChild(txt(K, mid, y0 - 30, "send/recv", "lbl-sm", "middle", { style: "fill:var(--c2)" }));
          }
          svg.appendChild(txt(K, W / 2, y0 + gh + 30, "Only one activation vector (d = " + m.d + ") crosses each stage boundary per token.", "lbl", "middle"));
          svg.appendChild(txt(K, W / 2, y0 + gh + 48, "But a request visits stages one at a time: without many micro-batches, most GPUs sit idle (the bubble).", "lbl-sm", "middle"));
        } else if (strat === "EP") {
          if (r.moe) {
            for (var a = 0; a < N; a++) for (var b = a + 1; b < N; b++) {
              var mx = (cx(a) + cx(b)) / 2, hgt = 14 + 9 * (b - a);
              svg.appendChild(K.s("path", { d: "M" + cx(a) + " " + (y0 - 2) + " Q" + mx + " " + (y0 - 2 - hgt * 1.4) + " " + cx(b) + " " + (y0 - 2), class: "ln-thin s3", opacity: 0.65 }));
            }
            svg.appendChild(txt(K, W / 2, y0 + gh + 30, "all-to-all: each token is dispatched to the GPUs holding its top-" + m.topk + " experts, then combined back (2× per MoE layer)", "lbl", "middle", { style: "fill:var(--c3)" }));
            svg.appendChild(txt(K, W / 2, y0 + gh + 48, "Blue = router + attention (replicated, often data-parallel attention). Green = routed experts owned by this GPU.", "lbl-sm", "middle"));
          } else {
            svg.appendChild(txt(K, W / 2, 30, "⚠ " + mname + " is dense: there are no experts to distribute. Pick Mixtral-8x7B or DeepSeek-V3.", "lbl", "middle", { style: "fill:var(--red)" }));
            svg.appendChild(txt(K, W / 2, y0 + gh + 30, "Shown as plain replicas (same as DP).", "lbl", "middle"));
          }
        } else {
          for (var q = 0; q < N - 1; q++) {
            var md = (cx(q) + cx(q + 1)) / 2;
            svg.appendChild(K.s("path", { d: "M" + cx(q) + " " + (y0 - 2) + " Q" + md + " " + (y0 - 44) + " " + cx(q + 1) + " " + (y0 - 2), class: "ln-thin s4", "marker-end": mk }));
            svg.appendChild(txt(K, md, y0 - 28, "K/V", "lbl-sm", "middle", { style: "fill:var(--c4)" }));
          }
          svg.appendChild(K.s("path", { d: "M" + cx(N - 1) + " " + (y0 + gh + 2) + " Q" + W / 2 + " " + (y0 + gh + 50) + " " + cx(0) + " " + (y0 + gh + 2), class: "ln-thin s4", "marker-end": mk }));
          svg.appendChild(txt(K, W / 2, y0 + gh + 48, "ring: each GPU passes K/V blocks to its neighbour — N−1 hops per layer, overlapped with attention compute", "lbl", "middle", { style: "fill:var(--c4)" }));
          svg.appendChild(txt(K, x0, 20, "Weights replicated (usually combined with TP); the long sequence is split into " + N + " chunks.", "lbl"));
        }
        // GPUs
        for (var g = 0; g < N; g++) {
          var x = gx(g);
          svg.appendChild(K.s("rect", { x: x, y: y0, width: gw, height: gh, rx: 8, class: "box" }));
          svg.appendChild(txt(K, x + gw / 2, y0 + 15, "GPU " + g, "lbl", "middle"));
          var cp = content(svg, g, x + 7, y0 + 24, gw - 14, r);
          svg.appendChild(txt(K, x + gw / 2, y0 + 133, cp, "lbl-sm", "middle"));
          var by2 = y0 + 142, bw2 = gw - 14, wf = Math.min(1, r.w / capB), kf = Math.min(1 - wf, r.kv / capB);
          svg.appendChild(K.s("rect", { x: x + 7, y: by2, width: bw2, height: 10, rx: 3, class: "fline" }));
          svg.appendChild(K.s("rect", { x: x + 7, y: by2, width: bw2 * wf, height: 10, class: "f1" }));
          svg.appendChild(K.s("rect", { x: x + 7 + bw2 * wf, y: by2, width: bw2 * kf, height: 10, class: "f3" }));
          var over = r.total > capB;
          svg.appendChild(txt(K, x + gw / 2, by2 + 25, fmtGB(r.total) + (over ? " ✗" : ""), "lbl mono", "middle", over ? { style: "fill:var(--red)" } : null));
        }
        svg.appendChild(txt(K, W - x0, H - 4, "memory bar: blue = weights, green = KV cache, of " + cap + " GB", "lbl-sm", "end"));
        K.clear(c.stage).appendChild(svg);

        var fits = r.total <= capB;
        var EXPL = {
          DP: "Each GPU holds a <b>full copy</b> and serves different requests. Zero communication and linear throughput scaling — but the whole model must fit on one GPU and per-request latency does not improve. This is just “add replicas”.",
          TP: "Every weight matrix is split " + N + " ways (column-parallel, then row-parallel), so each GPU reads only 1/" + N + " of the weights per token → <b>lower latency</b>. The price: <b>" + r.nColl + " all-reduces per forward pass</b> on the critical path, so TP lives inside one NVLink node (TP ≤ 8)." +
            (r.kvRep > 1 ? " Note: with only " + (m.mla ? "1 latent KV" : m.kvh + " KV heads") + ", KV heads get <b>replicated ×" + r.kvRep + "</b> across GPUs — wasted KV memory." : ""),
          PP: "Layers are split into " + N + " stages. Only a small activation vector crosses each boundary, so PP tolerates slow links (use it <b>across nodes</b>). But a single request walks through the stages one after another: no latency win, and bubbles unless micro-batches keep every stage busy.",
          EP: r.moe ? "Each GPU owns " + (m.E / N) + " of " + m.E + " experts; router + attention are replicated. Tokens travel to whichever GPUs hold their top-" + m.topk + " experts and back (<b>all-to-all</b>). No per-layer all-reduce, so EP scales across nodes and gives the <b>best throughput for MoE</b> (" + (m.Pa / 1e9).toFixed(0) + "B of " + (m.P / 1e9).toFixed(0) + "B params active per token)." :
            "EP only applies to Mixture-of-Experts models. For a dense model use TP (latency) or DP (throughput).",
          CP: "The sequence is cut into " + N + " chunks; each GPU does attention for its chunk while K/V blocks circulate around a ring (ring attention). Weights are not split, but the KV cache and attention FLOPs are — this is how you prefill <b>100k+ token prompts</b>. Rare for chat serving; key for long context and video.",
        };
        c.readout.innerHTML = stats([
          ["weights / GPU", fmtGB(r.w)],
          ["KV cache / GPU (" + batch + " × " + fmtTok(ctx) + " tok)", fmtGB(r.kv)],
          ["total / GPU vs " + cap + " GB", fmtGB(r.total) + (fits ? " ✓ fits" : " ✗ OOM"), fits ? "var(--green)" : "var(--red)"],
          ["sent per token, per GPU", r.tok ? K.fmtBytes(r.tok) : "0"],
          ["collectives / forward", r.nColl ? r.nColl + " × " + r.coll : "none"],
          [strat === "CP" ? "comm for one " + fmtTok(ctx) + " prefill" : "comm per decode step (" + batch + " seqs)", r.tNv ? "NVLink " + fmtT(r.tNv) + " · IB " + fmtT(r.tIb) : "—"],
        ]) + "<div style='margin-top:8px'>" + EXPL[strat] + "</div>" +
          "<div class='muted' style='font-size:12.5px;margin-top:4px'>KV bytes/token = " + (m.mla ? "MLA latent " + m.mla + " × 2 B × " + m.L + " layers" : "2 × " + m.L + " layers × " + m.kvh + " KV heads × " + m.hd + " × 2 B") + " = " + K.fmtBytes(r.kvTok) + ". Comm times use α–β estimates (NVLink 450 GB/s, IB NDR 50 GB/s per direction) and ignore overlap with compute.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- tp-split
  V.register("tp-split", {
    title: "Tensor parallelism inside one MLP: column-parallel → GeLU → row-parallel → all-reduce",
    desc: "Step through how Z = GeLU(X·A)·B is split across GPUs. A is cut by <b>columns</b>, B by <b>rows</b>, so GeLU runs locally and only the final partial sums need one all-reduce. The numbers are real: the sum of the partials equals the single-GPU result exactly (hover a cell for its value).",
    render: function (c, p, K) {
      var N = +p.gpus === 4 ? 4 : 2, step = 0, timer = null, mname = MODELS[p.model] ? p.model : "Llama-3.1-70B", tokens = p.tokens || 1;
      var f = 8, d = 4, rows = 3, X, A, B, seed = 11;
      var STEPS = ["① shard the weights", "② column-parallel matmul Yᵢ = X·Aᵢ", "③ GeLU locally — no communication", "④ row-parallel matmul → partial Zᵢ", "⑤ all-reduce: Z = Σ Zᵢ on every GPU"];
      function init() {
        var r = K.rng(seed);
        var mat = function (a, b, s) { var M = []; for (var i = 0; i < a; i++) { M.push([]); for (var j = 0; j < b; j++) M[i].push(+((r() * 2 - 1) * s).toFixed(2)); } return M; };
        X = mat(rows, d, 1); A = mat(d, f, 1.2); B = mat(f, d, 1);
      }
      function mm(P, Q) { return P.map(function (row) { return Q[0].map(function (_, j) { return row.reduce(function (s, v, k) { return s + v * Q[k][j]; }, 0); }); }); }
      function gelu(x) { return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
      function cols(M, a, b) { return M.map(function (r) { return r.slice(a, b); }); }
      function map(M, fn) { return M.map(function (r) { return r.map(fn); }); }
      function add(P, Q) { return P.map(function (r, i) { return r.map(function (v, j) { return v + Q[i][j]; }); }); }
      init();
      var gs = K.select({ label: "GPUs", value: String(N), options: ["2", "4"], onChange: function (v) { N = +v; draw(); } });
      c.controls.appendChild(gs);
      c.controls.appendChild(K.button("◀ Prev", function () { stop(); step = Math.max(0, step - 1); draw(); }));
      c.controls.appendChild(K.button("Next ▶", function () { stop(); step = Math.min(4, step + 1); draw(); }));
      var play = K.button("▶ Play", function () {
        if (timer) return stop();
        play.textContent = "⏸ Pause";
        timer = setInterval(function () { if (!c.root.isConnected) return stop(); step = (step + 1) % 5; draw(); }, 1400);
      }, "primary");
      c.controls.appendChild(play);
      c.controls.appendChild(K.button("New numbers", function () { seed++; init(); draw(); }));
      c.controls.appendChild(K.select({ label: "real model", value: mname, options: ["Llama-3.1-8B", "Qwen2.5-7B", "Llama-3.1-70B"], onChange: function (v) { mname = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "tokens in flight", value: String(tokens), options: [{ value: "1", label: "1 (one decode step)" }, { value: "64", label: "64 (batch of 64 decodes)" }, { value: "4096", label: "4096 (a 4k prefill)" }], onChange: function (v) { tokens = +v; draw(); } }));
      function stop() { clearInterval(timer); timer = null; play.textContent = "▶ Play"; }

      function draw() {
        var W = 720, cs = N === 2 ? 14 : 11, fn = f / N, laneH = 4 * cs + 36, top = 40;
        var H = top + N * (laneH + 8) + 6;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var widths = [4 * cs, 16, fn * cs, 16, fn * cs, 56, fn * cs, 16, 4 * cs, 16, 4 * cs, 64, 4 * cs];
        var tot = 50 + widths.reduce(function (a, b) { return a + b; }, 0), xs = [], acc = (W - tot) / 2 + 50;
        widths.forEach(function (w) { xs.push(acc); acc += w; });
        var heads = [[0, "X"], [2, "Aᵢ"], [4, "Yᵢ"], [6, "GeLU(Yᵢ)"], [8, "Bᵢ"], [10, "Zᵢ"], [12, "Z = ΣZᵢ"]];
        heads.forEach(function (hd) { svg.appendChild(txt(K, xs[hd[0]] + widths[hd[0]] / 2, 22, hd[1], "lbl-sm", "middle")); });
        var Zfull = mm(map(mm(X, A), gelu), B), Zsum = null;
        function matrix(M, x, y, cls, show, hl) {
          var R = M.length, Cn = M[0].length;
          if (!show) { svg.appendChild(K.s("rect", { x: x, y: y, width: Cn * cs, height: R * cs, rx: 3, class: "box", "stroke-dasharray": "3 3", opacity: 0.6 })); return; }
          var mx = 0; M.forEach(function (r) { r.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); });
          M.forEach(function (r, i) {
            r.forEach(function (v, j) {
              var rc = K.s("rect", { x: x + j * cs + 0.5, y: y + i * cs + 0.5, width: cs - 1, height: cs - 1, rx: 2, class: cls, opacity: (0.18 + 0.82 * Math.abs(v) / (mx || 1)).toFixed(2) });
              rc.appendChild(K.s("title", { text: v.toFixed(3) }));
              svg.appendChild(rc);
            });
          });
          if (hl) svg.appendChild(K.s("rect", { x: x - 3, y: y - 3, width: Cn * cs + 6, height: R * cs + 6, rx: 4, class: "box-hl", "fill-opacity": 0 }));
        }
        for (var i = 0; i < N; i++) {
          var y = top + i * (laneH + 8), my = y + 14, gcls = FILL[i];
          svg.appendChild(K.s("rect", { x: xs[0] - 50, y: y, width: xs[11] - xs[0] + 56, height: laneH, rx: 8, class: "fbg3" }));
          svg.appendChild(txt(K, xs[0] - 44, y + 16, "GPU " + i, "lbl"));
          var Ai = cols(A, i * fn, (i + 1) * fn), Bi = B.slice(i * fn, (i + 1) * fn);
          var Yi = mm(X, Ai), Gi = map(Yi, gelu), Zi = mm(Gi, Bi);
          Zsum = Zsum ? add(Zsum, Zi) : Zi;
          matrix(X, xs[0], my, "f6", true);
          svg.appendChild(txt(K, xs[1] + 8, my + 1.5 * cs + 4, "·", "lbl", "middle"));
          matrix(Ai, xs[2], my, gcls, true, step === 0);
          svg.appendChild(txt(K, xs[3] + 8, my + 1.5 * cs + 4, "=", "lbl", "middle"));
          matrix(Yi, xs[4], my, gcls, step >= 1, step === 1);
          svg.appendChild(txt(K, xs[5] + 28, my + 1.5 * cs + 4, "→ GeLU →", "lbl-sm", "middle"));
          matrix(Gi, xs[6], my, gcls, step >= 2, step === 2);
          svg.appendChild(txt(K, xs[7] + 8, my + 1.5 * cs + 4, "·", "lbl", "middle"));
          matrix(Bi, xs[8], my, gcls, true, step === 0);
          svg.appendChild(txt(K, xs[9] + 8, my + 1.5 * cs + 4, "=", "lbl", "middle"));
          matrix(Zi, xs[10], my, gcls, step >= 3, step === 3);
          matrix(Zfull, xs[12], my, "f5", step >= 4, step === 4);
          svg.appendChild(txt(K, xs[2], y + laneH - 6, "A[:, " + i * fn + ":" + (i + 1) * fn + "]  ·  B[" + i * fn + ":" + (i + 1) * fn + ", :]", "lbl-sm mono"));
        }
        // all-reduce bracket
        var bx = xs[11] + 22, y1 = top + 14 + 1.5 * cs, y2 = top + (N - 1) * (laneH + 8) + 14 + 1.5 * cs;
        if (step >= 4) {
          svg.appendChild(K.s("line", { x1: bx, x2: bx, y1: y1, y2: y2, class: "s2", "stroke-width": 3 }));
          for (var j = 0; j < N; j++) {
            var yy = top + j * (laneH + 8) + 14 + 1.5 * cs;
            svg.appendChild(K.s("line", { x1: xs[10] + 4 * cs + 3, x2: xs[12] - 4, y1: yy, y2: yy, class: "ln-thin s2" }));
          }
          svg.appendChild(txt(K, bx - 6, (y1 + y2) / 2, "Σ all-reduce", "lbl", "middle", { transform: "rotate(-90 " + (bx - 6) + " " + (y1 + y2) / 2 + ")", style: "fill:var(--c2)" }));
        }
        K.clear(c.stage).appendChild(svg);

        // numbers
        var err = 0;
        Zfull.forEach(function (r, a) { r.forEach(function (v, b) { err = Math.max(err, Math.abs(v - Zsum[a][b])); }); });
        var Pfull = mm(X, A), gsum = null;
        for (var k = 0; k < N; k++) {
          var Xi = cols(X, k * d / N, (k + 1) * d / N), Ar = A.slice(k * d / N, (k + 1) * d / N);
          var gp = map(mm(Xi, Ar), gelu);
          gsum = gsum ? add(gsum, gp) : gp;
        }
        var wrong = 0;
        Pfull.forEach(function (r, a) { r.forEach(function (v, b) { wrong = Math.max(wrong, Math.abs(gelu(v) - gsum[a][b])); }); });
        var m = MODELS[mname], tp = N, payload = tokens * m.d * 2, ring = 2 * (tp - 1) / tp * payload, nv = linkBy("nvlink");
        var DESC = [
          "Each GPU gets a column slice Aᵢ (d × f/N) and the matching row slice Bᵢ (f/N × d). The input X is replicated on every GPU.",
          "Yᵢ = X·Aᵢ. Each GPU computes f/N of the hidden units — a complete, correct slice of Y. No communication.",
          "GeLU is element-wise, so GeLU(Yᵢ) is exactly the i-th slice of GeLU(Y). Still no communication — <b>this is why A is split by columns</b>.",
          "Zᵢ = GeLU(Yᵢ)·Bᵢ. Each GPU only has part of the inner sum over the f hidden units, so Zᵢ has the full output shape but is a <i>partial</i> sum.",
          "All-reduce adds the partials so every GPU ends up with the full Z, ready for the next layer: <b>one all-reduce for the whole MLP</b> (attention gets one too → 2 per layer).",
        ];
        c.readout.innerHTML = "<b>" + STEPS[step] + "</b> — " + DESC[step] +
          "<br>" + K.tex("Z = \\mathrm{GeLU}(XA)\\,B = \\sum_i \\mathrm{GeLU}(XA_i)\\,B_i") + " &nbsp; check: max |Z<sub>1 GPU</sub> − Σ Zᵢ| = <b>" + err.toExponential(1) + "</b> (exact up to rounding)." +
          "<br><span class='muted'>Wrong split: cutting A by <i>rows</i> gives partial pre-activations, and GeLU(a+b) ≠ GeLU(a)+GeLU(b): error = <b style='color:var(--red)'>" + wrong.toFixed(3) + "</b> → you would need an extra all-reduce <i>before</i> GeLU.</span>" +
          stats([
            [mname + " MLP weights / GPU", K.fmtBytes(3 * m.d * m.ffn * 2 / tp) + " (of " + K.fmtBytes(3 * m.d * m.ffn * 2) + ")"],
            ["all-reduce payload (" + tokens + " tok × d=" + m.d + " × 2 B)", K.fmtBytes(payload)],
            ["ring traffic per GPU", K.fmtBytes(ring)],
            ["per layer on NVLink (2 all-reduces)", fmtT(2 * (2 * (tp - 1) * nv.a + ring / nv.bw))],
          ]) + "<div class='muted' style='font-size:12.5px;margin-top:4px'>Real SwiGLU MLPs have gate + up projections (both column-parallel) and a down projection (row-parallel): same pattern, 3 matrices. For one decode token the payload is tiny, so latency α dominates — that's why TP wants NVLink.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- ring all-reduce
  V.register("ring-allreduce", {
    title: "Ring all-reduce: reduce-scatter, then all-gather",
    desc: "Every GPU starts with its own vector (its own colour), cut into N chunks. In N−1 reduce-scatter steps each chunk travels round the ring collecting everyone's contribution; in N−1 all-gather steps the finished chunks are copied round. Each stripe in a cell = one GPU's contribution.",
    render: function (c, p, K) {
      var N = K.clamp(+p.gpus || 4, 2, 8), size = p.size || 1e9, link = "nvlink", k = 0, has, anim = null, playing = false, dur = 750;
      c.controls.appendChild(K.slider({ label: "GPUs N", min: 2, max: 8, value: N, onInput: function (v) { N = v; reset(); } }));
      c.controls.appendChild(K.slider({ label: "data size S", min: 5, max: 10.5, step: 0.05, value: Math.log10(size), fmt: function (v) { return K.fmtBytes(Math.pow(10, v)); }, onInput: function (v) { size = Math.pow(10, v); text(); } }));
      c.controls.appendChild(K.select({ label: "link", value: link, options: LINKS.map(function (l) { return { value: l.id, label: l.name }; }), onChange: function (v) { link = v; text(); } }));
      c.controls.appendChild(K.button("Step", function () { setPlaying(false); go(); }));
      var pb = K.button("▶ Play", function () { if (playing) setPlaying(false); else { if (k >= 2 * (N - 1)) reset(); setPlaying(true); go(); } }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Reset", function () { setPlaying(false); reset(); }));
      function setPlaying(v) { playing = v; pb.textContent = v ? "⏸ Pause" : "▶ Play"; }
      function mod(a) { return ((a % N) + N) % N; }
      function reset() {
        anim = null; k = 0; has = [];
        for (var g = 0; g < N; g++) { has.push([]); for (var ch = 0; ch < N; ch++) has[g].push(1 << g); }
        draw(0); text();
      }
      function transfers(kk) {
        var rs = kk < N - 1, s = rs ? kk : kk - (N - 1), out = [];
        for (var g = 0; g < N; g++) out.push({ from: g, to: mod(g + 1), chunk: rs ? mod(g - s) : mod(g + 1 - s), rs: rs, bits: has[g][rs ? mod(g - s) : mod(g + 1 - s)] });
        return out;
      }
      function apply(tr) {
        tr.forEach(function (t) { has[t.to][t.chunk] = t.rs ? (has[t.to][t.chunk] | t.bits) : t.bits; });
      }
      function go() {
        if (anim || k >= 2 * (N - 1)) return;
        anim = { t0: performance.now(), tr: transfers(k) };
        requestAnimationFrame(frame);
      }
      function frame(now) {
        if (!c.root.isConnected) { setPlaying(false); return; }
        if (!anim) return;
        var t = (now - anim.t0) / dur;
        if (t >= 1) {
          apply(anim.tr); k++; anim = null; draw(0); text();
          if (k >= 2 * (N - 1)) setPlaying(false);
          else if (playing) setTimeout(function () { if (playing && c.root.isConnected) go(); }, 300);
          return;
        }
        draw(t);
        requestAnimationFrame(frame);
      }
      function draw(t) {
        var W = 720, gx0 = 86, colW = (W - gx0 - 16) / N, cw = Math.min(colW - 14, 96), chH = Math.min(30, 230 / N), gy = 86;
        var H = gy + N * (chH + 5) + 44;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var mk = arrowMarker(K, svg, "fmuted");
        function cx(g) { return gx0 + g * colW + colW / 2; }
        function cellXY(g, ch) { return [gx0 + g * colW + (colW - cw) / 2, gy + ch * (chH + 5)]; }
        for (var g = 0; g < N; g++) {
          svg.appendChild(K.s("rect", { x: cx(g) - 34, y: 50, width: 68, height: 22, rx: 6, class: "box" }));
          svg.appendChild(K.s("rect", { x: cx(g) - 28, y: 56, width: 10, height: 10, rx: 2, class: FILL[g] }));
          svg.appendChild(txt(K, cx(g) + 6, 65, "GPU " + g, "lbl", "middle"));
          if (g < N - 1) svg.appendChild(K.s("path", { d: "M" + (cx(g) + 10) + " 48 Q" + (cx(g) + colW / 2) + " 26 " + (cx(g + 1) - 10) + " 48", class: "ln-thin smuted", "marker-end": mk }));
        }
        var ybot = gy + N * (chH + 5) + 4;
        svg.appendChild(K.s("path", { d: "M" + cx(N - 1) + " " + ybot + " Q" + (W + gx0) / 2 + " " + (ybot + 36) + " " + cx(0) + " " + ybot, class: "ln-thin smuted", "marker-end": mk }));
        svg.appendChild(txt(K, (W + gx0) / 2, ybot + 30, "ring wraps around: GPU " + (N - 1) + " → GPU 0", "lbl-sm", "middle"));
        var phase = k < N - 1 ? "reduce-scatter" : k < 2 * (N - 1) ? "all-gather" : "done";
        svg.appendChild(txt(K, 8, 20, phase === "done" ? "✓ done — every GPU holds the full sum" : (phase === "reduce-scatter" ? "Phase 1 · reduce-scatter" : "Phase 2 · all-gather") + " · step " + (k + 1) + " of " + 2 * (N - 1), "lbl"));
        var full = (1 << N) - 1, sending = {};
        if (anim) anim.tr.forEach(function (tr) { sending[tr.from + "," + tr.chunk] = 1; });
        function cell(x, y, w, hgt, bits, strokeCls) {
          svg.appendChild(K.s("rect", { x: x, y: y, width: w, height: hgt, rx: 4, class: "fbg3" }));
          for (var j = 0; j < N; j++) if (bits & (1 << j)) svg.appendChild(K.s("rect", { x: x + j * w / N, y: y, width: w / N, height: hgt, class: FILL[j], opacity: 0.9 }));
          svg.appendChild(K.s("rect", { x: x, y: y, width: w, height: hgt, rx: 4, fill: "none", class: strokeCls || "smuted", "stroke-width": strokeCls ? 2.5 : 0.8 }));
        }
        for (var ch = 0; ch < N; ch++) {
          svg.appendChild(txt(K, 8, gy + ch * (chH + 5) + chH / 2 + 4, "chunk " + ch, "lbl-sm"));
          for (var g2 = 0; g2 < N; g2++) {
            var xy = cellXY(g2, ch), b = has[g2][ch];
            cell(xy[0], xy[1], cw, chH, b, sending[g2 + "," + ch] ? "s2" : (b === full ? "s3" : null));
          }
        }
        if (anim) anim.tr.forEach(function (tr) {
          var a = cellXY(tr.from, tr.chunk), bb = cellXY(tr.to, tr.chunk), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          var wrap = tr.to < tr.from, lift = wrap ? 26 : 14;
          var x = K.lerp(a[0], bb[0], e) + cw * 0.1, y = K.lerp(a[1], bb[1], e) - Math.sin(Math.PI * e) * lift + chH * 0.15;
          cell(x, y, cw * 0.8, chH * 0.7, tr.bits, "sfg");
        });
        K.clear(c.stage).appendChild(svg);
      }
      function text() {
        var L = linkBy(link), steps = 2 * (N - 1), chunk = size / N;
        var tRing = steps * (L.a + chunk / L.bw), tNaive = 2 * (N - 1) * (L.a + size / L.bw);
        var ph = k >= steps ? "<b>Done.</b> Every GPU now holds the element-wise sum of all " + N + " vectors (green border = fully reduced chunk)." :
          k < N - 1 ? "<b>Reduce-scatter</b>: each GPU sends one chunk to its right neighbour, which <b>adds</b> it to its own copy. After N−1 steps GPU g owns the complete sum of chunk g+1." :
            "<b>All-gather</b>: each GPU forwards a finished chunk; the receiver simply <b>overwrites</b> its copy.";
        c.readout.innerHTML = ph + stats([
          ["chunk size S/N", K.fmtBytes(chunk)],
          ["sent per GPU so far", K.fmtBytes(k * chunk) + " / " + K.fmtBytes(steps * chunk)],
          ["ring all-reduce on " + L.name, fmtT(tRing)],
          ["naive: reduce to GPU 0 + broadcast", fmtT(tNaive)],
        ]) + "<div style='margin-top:6px'>" + K.tex("T_{ring} = 2(N-1)\\,\\alpha + \\frac{2(N-1)}{N}\\cdot\\frac{S}{\\beta}") +
          " — each GPU sends ≈ 2S no matter how many GPUs, so the ring is <b>bandwidth-optimal</b>; but the 2(N−1) latency terms grow with N (NCCL switches to tree algorithms for small messages / many nodes). β = " + L.note + ".</div>";
      }
      reset();
    },
  });

  // ---------------------------------------------------------------- pipeline bubble
  V.register("pipeline-bubble", {
    title: "Pipeline parallelism: micro-batches and the bubble",
    desc: "Each row is a GPU holding a slice of the layers (a <i>stage</i>). The batch is cut into m micro-batches (numbered, coloured) that flow down the pipeline. Red gaps = idle GPU time — the <b>bubble</b>. More micro-batches → smaller bubble.",
    render: function (c, p, K) {
      var P = K.clamp(+p.stages || 4, 2, 8), M = K.clamp(+p.micro || 4, 1, 16), mode = ["infer", "gpipe", "1f1b"].indexOf(p.mode) >= 0 ? p.mode : "infer";
      var cursor = null, t0 = 0;
      c.controls.appendChild(K.slider({ label: "stages p (GPUs)", min: 2, max: 8, value: P, onInput: function (v) { P = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "micro-batches m", min: 1, max: 16, value: M, onInput: function (v) { M = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "schedule", value: mode, options: [
        { value: "infer", label: "inference (forward only)" },
        { value: "gpipe", label: "training — GPipe (all F, then all B)" },
        { value: "1f1b", label: "training — 1F1B" }], onChange: function (v) { mode = v; draw(); } }));
      var ab = K.button("▶ Animate", function () {
        if (cursor !== null) { cursor = null; ab.textContent = "▶ Animate"; draw(); return; }
        cursor = 0; t0 = performance.now(); ab.textContent = "⏸ Stop"; requestAnimationFrame(tick);
      }, "primary");
      c.controls.appendChild(ab);
      function tick(now) {
        if (cursor === null) return;
        if (!c.root.isConnected) { cursor = null; return; }
        var S = sched();
        cursor = (now - t0) / 1000 * Math.max(3, S.T / 5);
        if (cursor >= S.T) { cursor = null; ab.textContent = "▶ Animate"; draw(); return; }
        draw(); requestAnimationFrame(tick);
      }
      function sched() {
        var F = 1, B = 2, lists = [], fEnd = [], bEnd = [], ops = [];
        for (var s = 0; s < P; s++) {
          var L = [];
          if (mode !== "1f1b") {
            for (var j = 0; j < M; j++) L.push(["F", j]);
            if (mode === "gpipe") for (var j2 = 0; j2 < M; j2++) L.push(["B", j2]);
          } else {
            var warm = Math.min(P - s - 1, M), fi = 0, bi = 0;
            for (; fi < warm; fi++) L.push(["F", fi]);
            while (fi < M) { L.push(["F", fi++]); L.push(["B", bi++]); }
            while (bi < M) L.push(["B", bi++]);
          }
          lists.push(L); fEnd.push([]); bEnd.push([]);
        }
        var ptr = lists.map(function () { return 0; }), free = lists.map(function () { return 0; }), progress = true;
        while (progress) {
          progress = false;
          for (var st = 0; st < P; st++) {
            while (ptr[st] < lists[st].length) {
              var op = lists[st][ptr[st]], dep;
              if (op[0] === "F") dep = st === 0 ? 0 : fEnd[st - 1][op[1]];
              else dep = st === P - 1 ? fEnd[st][op[1]] : bEnd[st + 1][op[1]];
              if (dep === undefined) break;
              var start = Math.max(free[st], dep), end = start + (op[0] === "F" ? F : B);
              (op[0] === "F" ? fEnd : bEnd)[st][op[1]] = end;
              free[st] = end; ops.push({ s: st, j: op[1], type: op[0], start: start, end: end });
              ptr[st]++; progress = true;
            }
          }
        }
        var T = Math.max.apply(null, free), busy = ops.reduce(function (a, o) { return a + o.end - o.start; }, 0);
        return { ops: ops, T: T, busy: busy };
      }
      function draw() {
        var S = sched(), W = 720, l = 70, r = 16, top = 34, rowH = Math.min(34, 250 / P), H = top + P * (rowH + 6) + 40;
        var u = (W - l - r) / S.T, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        for (var s = 0; s < P; s++) {
          var y = top + s * (rowH + 6);
          svg.appendChild(K.s("rect", { x: l, y: y, width: W - l - r, height: rowH, rx: 3, style: "fill:var(--red)", opacity: 0.13 }));
          svg.appendChild(txt(K, l - 8, y + rowH / 2 + 4, "GPU " + s, "lbl", "end"));
        }
        S.ops.forEach(function (o) {
          var x = l + o.start * u, w = (o.end - o.start) * u, y = top + o.s * (rowH + 6);
          var fut = cursor !== null && o.start >= cursor;
          svg.appendChild(K.s("rect", { x: x + 0.75, y: y, width: Math.max(1, w - 1.5), height: rowH, rx: 3, class: FILL[o.j % 6], opacity: fut ? 0.08 : (o.type === "F" ? 0.92 : 0.5) }));
          if (w >= 15 && !fut) svg.appendChild(txt(K, x + w / 2, y + rowH / 2 + 4, (o.type === "B" ? "b" : "") + o.j, o.type === "F" ? "lbl-sm t-white" : "lbl-sm", "middle", { style: "font-weight:700" + (o.type === "F" ? ";fill:#fff" : ";fill:var(--fg)") }));
        });
        if (cursor !== null) svg.appendChild(K.s("line", { x1: l + cursor * u, x2: l + cursor * u, y1: top - 6, y2: top + P * (rowH + 6), class: "sfg", "stroke-width": 1.5 }));
        var yAx = top + P * (rowH + 6) + 4;
        K.niceTicks(0, S.T, 10).forEach(function (t) {
          svg.appendChild(K.s("line", { x1: l + t * u, x2: l + t * u, y1: yAx, y2: yAx + 4, class: "axis" }));
          svg.appendChild(txt(K, l + t * u, yAx + 16, String(t), "tick", "middle"));
        });
        svg.appendChild(txt(K, (l + W - r) / 2, yAx + 32, "time (units of one micro-batch forward pass" + (mode === "infer" ? ")" : "; backward = 2 units)"), "axis-label", "middle"));
        svg.appendChild(txt(K, l, 18, mode === "infer" ? "solid = forward of micro-batch j" : "solid = forward Fj · faded bj = backward of micro-batch j", "lbl-sm"));
        K.clear(c.stage).appendChild(svg);

        var bubble = 1 - S.busy / (P * S.T), formula = (P - 1) / (M + P - 1);
        var inflight = 0, cur = 0, ev = [];
        S.ops.forEach(function (o) { if (o.s === 0) { if (o.type === "F") ev.push([o.start, 1]); else ev.push([o.end, -1]); } });
        ev.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; }).forEach(function (e) { cur += e[1]; inflight = Math.max(inflight, cur); });
        c.readout.innerHTML = stats([
          ["bubble (measured)", (bubble * 100).toFixed(1) + "%", bubble > 0.3 ? "var(--red)" : null],
          ["formula (p−1)/(m+p−1)", (formula * 100).toFixed(1) + "%"],
          ["GPU utilization", ((1 - bubble) * 100).toFixed(1) + "%"],
          [mode === "infer" ? "latency of one micro-batch" : "peak activations held on GPU 0", mode === "infer" ? P + " units (" + P + " stages in series)" : inflight + " micro-batches"],
        ]) + "<div style='margin-top:6px'>" + K.tex("\\text{bubble} = \\frac{p-1}{m+p-1}") + " — the pipeline needs p−1 steps to fill and p−1 to drain. " +
          (mode === "infer" ? "In inference serving the “micro-batches” are just different requests: continuous batching keeps stages busy, but each request still walks through all " + P + " stages in series, so <b>PP never lowers latency</b> — use it only when the model doesn't fit in one node." :
            mode === "gpipe" ? "GPipe keeps <b>all m micro-batches' activations</b> alive until their backward passes → memory grows with m." :
              "1F1B has the same bubble as GPipe but interleaves backwards early, so at most ~p micro-batches of activations are alive → <b>much less memory</b>, letting you raise m.") + "</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- comm cost (alpha-beta)
  V.register("comm-cost", {
    title: "How long does it take to send n bytes? The α–β model",
    desc: "Every transfer costs a fixed latency α plus size ÷ bandwidth β. Small messages are <b>latency-bound</b> (flat left part), big ones <b>bandwidth-bound</b> (the slope). Slide the size or pick a real message from LLM serving; diamonds mark n½ = α·β, where half the time is latency.",
    render: function (c, p, K) {
      var lg = Math.log10(p.bytes || 16384), op = p.op === "allreduce" ? "allreduce" : "p2p", N = 8;
      var PRE = [
        { label: "TP all-reduce, 1 decode token (70B: d=8192, BF16)", v: 8192 * 2 },
        { label: "TP all-reduce, batch of 64 decode tokens (70B)", v: 64 * 8192 * 2 },
        { label: "TP all-reduce, 4k-token prefill (70B)", v: 4096 * 8192 * 2 },
        { label: "PP activation hop, 1 token (8B: d=4096)", v: 4096 * 2 },
        { label: "KV cache of a 2k prompt (Llama-3.1-8B)", v: 2048 * 131072 },
        { label: "KV cache of an 8k prompt (Llama-3.1-70B)", v: 8192 * 327680 },
        { label: "gradients of an 8B model (BF16)", v: 16.06e9 },
        { label: "weights of a 70B model (BF16)", v: 141e9 },
      ];
      var sl = K.slider({ label: "message size n", min: 2, max: 12, step: 0.02, value: lg, fmt: function (v) { return K.fmtBytes(Math.pow(10, v)); }, onInput: function (v) { lg = v; draw(); } });
      c.controls.appendChild(K.select({ label: "real message", value: "", options: [{ value: "", label: "— pick one —" }].concat(PRE.map(function (q, i) { return { value: String(i), label: q.label }; })),
        onChange: function (v) { if (v === "") return; lg = Math.log10(PRE[+v].v); sl.set(lg); draw(); } }));
      c.controls.appendChild(sl);
      c.controls.appendChild(K.select({ label: "operation", value: op, options: [{ value: "p2p", label: "point-to-point send" }, { value: "allreduce", label: "ring all-reduce over N GPUs" }], onChange: function (v) { op = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "N (all-reduce)", min: 2, max: 16, value: N, onInput: function (v) { N = v; if (op === "allreduce") draw(); } }));
      function tm(L, n) { return op === "p2p" ? L.a + n / L.bw : 2 * (N - 1) * L.a + 2 * (N - 1) / N * n / L.bw; }
      function draw() {
        var ch = K.chart({ w: 720, h: 340, x: [1e2, 1e12], y: [1e-6, 1e3], xLog: true, yLog: true, xLabel: "message size n (bytes)", yLabel: "time", xFmt: K.fmtBytes, yFmt: fmtT, pad: { l: 64 } });
        var n = Math.pow(10, lg);
        ch.g.appendChild(K.s("line", { x1: ch.sx(n), x2: ch.sx(n), y1: ch.pad.t, y2: ch.H - ch.pad.b, class: "ln-thin sfg dash" }));
                ch.g.appendChild(K.s("rect", { x: 74, y: 14, width: 262, height: LINKS.length * 17 + 8, rx: 4, class: "fbg", opacity: 0.9 }));
        LINKS.forEach(function (L, i) {
          var pts = [];
          for (var e = 2; e <= 12.001; e += 0.05) pts.push([Math.pow(10, e), tm(L, Math.pow(10, e))]);
          ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln s" + (i + 1) }));
          var nh = op === "p2p" ? L.a * L.bw : N * L.a * L.bw;
          var dx = ch.sx(nh), dy = ch.sy(tm(L, nh));
          ch.g.appendChild(K.s("path", { d: "M" + dx + " " + (dy - 5) + " L" + (dx + 5) + " " + dy + " L" + dx + " " + (dy + 5) + " L" + (dx - 5) + " " + dy + "z", class: "f" + (i + 1) }));
          ch.g.appendChild(K.s("circle", { cx: ch.sx(n), cy: ch.sy(tm(L, n)), r: 5, class: "f" + (i + 1), style: "stroke:var(--bg);stroke-width:1.5" }));
          ch.g.appendChild(K.s("rect", { x: 80, y: 22 + i * 17, width: 14, height: 4, class: "f" + (i + 1) }));
          ch.g.appendChild(txt(K, 100, 28 + i * 17, L.name + " — β " + K.fmtBytes(L.bw) + "/s, α " + fmtT(L.a), "lbl-sm"));
        });
        K.clear(c.stage).appendChild(ch.svg);
        var items = LINKS.map(function (L, i) {
          var t = tm(L, n), lat = (op === "p2p" ? L.a : 2 * (N - 1) * L.a) / t;
          return [L.name, fmtT(t) + " <span style='font-size:11px;color:var(--muted)'>(" + (lat * 100).toFixed(0) + "% latency)</span>", "var(--c" + (i + 1) + ")"];
        });
        c.readout.innerHTML = "<b>" + K.fmtBytes(n) + "</b> " + (op === "p2p" ? "sent point-to-point" : "all-reduced over " + N + " GPUs") + ":" + stats(items) +
          "<div style='margin-top:6px'>" + (op === "p2p" ? K.tex("t(n) = \\alpha + \\frac{n}{\\beta}") : K.tex("t(n) = 2(N-1)\\,\\alpha + \\frac{2(N-1)}{N}\\cdot\\frac{n}{\\beta}")) +
          " &nbsp; Below n½ = α·β (NVLink ≈ " + K.fmtBytes(LINKS[0].a * LINKS[0].bw) + ") you pay mostly latency — batching many small messages into one big one is the fix. A decode-time TP all-reduce is only ~16 KB, so its cost is almost pure α × (number of all-reduces): that's why TP stays on NVLink and why kernels fuse communication. Moving a whole KV cache or weights is bandwidth-bound: pick the fattest pipe.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- disaggregation
  V.register("disaggregation", {
    title: "Colocated vs disaggregated prefill / decode",
    desc: "Top: one engine does both — every new prompt's prefill pauses all running decodes, so users see <b>inter-token latency (ITL) spikes</b>. Bottom: a prefill pool builds the KV cache and ships it over the interconnect to a decode pool that never stalls. Orange = prefill, blue = decode steps, purple = KV transfer.",
    render: function (c, p, K) {
      var DM = { "Llama-3.1-8B": 1, "Qwen2.5-7B": 1, "Mistral-7B": 1, "Llama-3.1-70B": 4 };
      var mname = DM[p.model] ? p.model : "Llama-3.1-8B", plen = p.prompt || 4096, rate = p.rate || 4, link = p.link || "ib", kvb = 2, chunked = false, fair = true, seed = 5;
      c.controls.appendChild(K.select({ label: "model", value: mname, options: Object.keys(DM).map(function (k) { return { value: k, label: k + (DM[k] > 1 ? " (TP4 per engine)" : " (1 GPU per engine)") }; }), onChange: function (v) { mname = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "prompt length", min: 8, max: 15, step: 0.5, value: Math.log2(plen), fmt: function (v) { return fmtTok(Math.round(Math.pow(2, v))) + " tok"; }, onInput: function (v) { plen = Math.round(Math.pow(2, v)); draw(); } }));
      c.controls.appendChild(K.slider({ label: "arrivals", min: 0.2, max: 8, step: 0.1, value: rate, fmt: function (v) { return v.toFixed(1) + " req/s"; }, onInput: function (v) { rate = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "KV link", value: link, options: LINKS.map(function (l) { return { value: l.id, label: l.name }; }), onChange: function (v) { link = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "KV dtype", value: "2", options: [{ value: "2", label: "BF16" }, { value: "1", label: "FP8" }], onChange: function (v) { kvb = +v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "chunked prefill (colocated)", value: chunked, onChange: function (v) { chunked = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "equal GPUs (2 colocated replicas)", value: fair, onChange: function (v) { fair = v; draw(); } }));
      c.controls.appendChild(K.button("New arrivals", function () { seed++; draw(); }));

      function pct(a, q) { if (!a.length) return 0; var s = a.slice().sort(function (x, y) { return x - y; }); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; }
      function mean(a) { return a.length ? a.reduce(function (x, y) { return x + y; }, 0) / a.length : 0; }

      function draw() {
        var m = MODELS[mname], tp = DM[mname], L = linkBy(link);
        var td = m.P * 2 / tp / (3.35e12 * 0.7) + (tp > 1 ? 0.0015 : 0) + 0.001;
        var flopTok = 2 * m.P + 2 * m.L * plen * m.d;
        var prefRate = tp * 989e12 * 0.5 * (tp > 1 ? 0.85 : 1) / flopTok, tPre = plen / prefRate;
        var kvTok = kvLayer(m, kvb) * m.L, kvBytes = kvTok * plen, tX = L.a + kvBytes / (tp * L.bw);
        var Wv = Math.min(20, Math.max(3, 4 * tPre)), Tsim = Math.max(60, 20 * Wv);
        var r = K.rng(seed), arr = [], t = 0;
        while (true) { t += -Math.log(1 - r()) / rate; if (t > Tsim) break; arr.push(t); }
        // --- colocated (one replica; with "equal GPUs" it gets every other request)
        var carr = fair ? arr.filter(function (_, i) { return i % 2 === 0; }) : arr;
        var steps = [], citl = [], cttft = [], q = [], ai = 0, stall = 0, cur = null, CH = 512;
        t = 0;
        while (t < Tsim && steps.length < 400000) {
          while (ai < carr.length && carr[ai] <= t) q.push({ arr: carr[ai++], left: plen });
          if (!chunked) {
            if (q.length) { var rq = q.shift(); steps.push({ t: t, dur: tPre, k: "p" }); t += tPre; stall += tPre; cttft.push(t + td - rq.arr); continue; }
            steps.push({ t: t, dur: td, k: "d", itl: td + stall }); citl.push(td + stall); stall = 0; t += td;
          } else {
            if (!cur && q.length) cur = q.shift();
            var pre = 0;
            if (cur) { pre = Math.min(CH, cur.left); cur.left -= pre; }
            var dur = td + pre / prefRate;
            steps.push({ t: t, dur: dur, k: pre ? "m" : "d", pre: pre / prefRate, itl: dur }); citl.push(dur); t += dur;
            if (cur && cur.left <= 0) { cttft.push(t + td - cur.arr); cur = null; }
          }
        }
        // --- disaggregated
        var free = 0, blocks = [], dttft = [];
        arr.forEach(function (a) { var s = Math.max(a, free), e = s + tPre; free = e; blocks.push({ s: s, e: e, x: e + tX }); dttft.push(e + tX + td - a); });
        var dItl = td;

        // --- draw
        var W = 720, l = 96, rr = 14, H = 336, sx = K.scale(0, Wv, l, W - rr);
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var mk = arrowMarker(K, svg, "f4");
        var vis = steps.filter(function (s) { return s.t < Wv; });
        var ymax = Math.max(td * 3, Math.max.apply(null, vis.map(function (s) { return s.itl || 0; }).concat([0]))) * 1.1;
        function clipW(a, b) { return Math.max(0, sx(Math.min(b, Wv)) - sx(a)); }
        function spark(y0, h, pts, cls) {
          var sy = K.scale(0, ymax, y0 + h, y0);
          svg.appendChild(K.s("rect", { x: l, y: y0, width: W - l - rr, height: h, class: "fbg3" }));
          svg.appendChild(K.s("line", { x1: l, x2: W - rr, y1: sy(td), y2: sy(td), class: "ln-thin smuted dash" }));
          svg.appendChild(txt(K, l - 6, y0 + 9, fmtT(ymax), "tick", "end"));
          svg.appendChild(txt(K, l - 6, sy(td) + 4, fmtT(td), "tick", "end"));
          svg.appendChild(txt(K, 8, y0 + h / 2 + 4, "ITL", "lbl"));
          var d = "";
          pts.forEach(function (pt, i) { d += (i ? "L" : "M") + sx(pt[0]).toFixed(1) + " " + sy(pt[1]).toFixed(1); });
          if (d) svg.appendChild(K.s("path", { d: d, class: "ln " + cls }));
        }
        svg.appendChild(txt(K, 8, 14, "Colocated: one engine does prefill + decode" + (fair ? " (replica 1 of 2 — each gets half the traffic)" : ""), "lbl"));
        svg.appendChild(txt(K, 8, 40, "engine", "lbl-sm"));
        var run = null;
        function flush() { if (run) { svg.appendChild(K.s("rect", { x: sx(run[0]), y: 26, width: clipW(run[0], run[1]), height: 22, class: "f1", opacity: 0.55 })); run = null; } }
        vis.forEach(function (s) {
          if (s.k === "d") { if (run && Math.abs(run[1] - s.t) < 1e-9) run[1] = s.t + s.dur; else { flush(); run = [s.t, s.t + s.dur]; } }
          else {
            flush();
            if (s.k === "p") svg.appendChild(K.s("rect", { x: sx(s.t), y: 26, width: clipW(s.t, s.t + s.dur), height: 22, class: "f2", rx: 2 }));
            else {
              svg.appendChild(K.s("rect", { x: sx(s.t), y: 26, width: clipW(s.t, s.t + s.dur), height: 22, class: "f1", opacity: 0.55 }));
              svg.appendChild(K.s("rect", { x: sx(s.t), y: 26, width: clipW(s.t, s.t + s.pre), height: 22, class: "f2" }));
            }
          }
        });
        flush();
        var cpts = [];
        vis.forEach(function (s) { var v = s.itl !== undefined ? s.itl : td; cpts.push([Math.min(Wv, s.t), v]); cpts.push([Math.min(Wv, s.t + s.dur), v]); });
        spark(56, 58, cpts, "s2");

        var y2 = 142;
        svg.appendChild(txt(K, 8, y2, "Disaggregated: prefill pool → KV transfer (" + K.fmtBytes(kvBytes) + " in " + fmtT(tX) + " per prompt) → decode pool", "lbl"));
        svg.appendChild(txt(K, 8, y2 + 24, "prefill", "lbl-sm"));
        svg.appendChild(txt(K, 8, y2 + 50, "KV xfer", "lbl-sm"));
        svg.appendChild(txt(K, 8, y2 + 78, "decode", "lbl-sm"));
        svg.appendChild(K.s("rect", { x: l, y: y2 + 64, width: W - l - rr, height: 22, class: "f1", opacity: 0.55 }));
        blocks.forEach(function (b) {
          if (b.s >= Wv) return;
          svg.appendChild(K.s("rect", { x: sx(b.s), y: y2 + 10, width: clipW(b.s, b.e), height: 22, class: "f2", rx: 2 }));
          if (b.e < Wv) {
            var xe = Math.min(b.x, Wv);
            svg.appendChild(K.s("path", { d: "M" + sx(b.e) + " " + (y2 + 33) + " L" + sx(xe) + " " + (y2 + 62), class: "s4", "stroke-width": 2.5, fill: "none", "marker-end": mk }));
            if (b.x < Wv) svg.appendChild(K.s("line", { x1: sx(b.x), x2: sx(b.x), y1: y2 + 64, y2: y2 + 86, class: "sfg", "stroke-width": 1.5 }));
            
          }
        });
        spark(y2 + 94, 58, [[0, dItl], [Wv, dItl]], "s3");
        var yAx = y2 + 158;
        K.niceTicks(0, Wv, 8).forEach(function (v) {
          svg.appendChild(K.s("line", { x1: sx(v), x2: sx(v), y1: yAx, y2: yAx + 4, class: "axis" }));
          svg.appendChild(txt(K, sx(v), yAx + 16, v + " s", "tick", "middle"));
        });
        svg.appendChild(txt(K, (l + W) / 2, yAx + 32, "time — first " + (+Wv.toPrecision(2)) + " s of a " + Math.round(Tsim) + " s simulation (stats below use all of it)", "axis-label", "middle"));
        K.clear(c.stage).appendChild(svg);

        var load = rate * tPre / (fair ? 2 : 1), dload = rate * tPre;
        var cp99 = pct(citl, 0.99);
        c.readout.innerHTML = stats([
          ["ITL p99 · max — colocated", fmtT(cp99) + " · " + fmtT(citl.reduce(function (a, b) { return Math.max(a, b); }, 0)), cp99 > 2 * td ? "var(--red)" : null],
          ["ITL p99 · max — disaggregated", fmtT(dItl) + " · " + fmtT(dItl), "var(--green)"],
          ["TTFT mean — colocated", fmtT(mean(cttft))],
          ["TTFT mean — disaggregated", fmtT(mean(dttft))],
          ["prefill of one prompt", fmtT(tPre)],
          ["KV transfer size", K.fmtBytes(kvBytes)],
          ["KV transfer on " + L.name, fmtT(tX)],
          ["prefill engine busy", (Math.min(dload, 9.99) * 100).toFixed(0) + "%" + (dload >= 1 ? " ⚠ overloaded" : ""), dload >= 1 ? "var(--red)" : null],
        ]) + "<div style='margin-top:6px'>" + K.tex("\\text{KV size} = \\underbrace{" + K.fmtBytes(kvTok).replace(" ", "\\,\\text{") + "}}_{\\text{bytes/token}} \\times " + plen + "\\ \\text{tokens}") +
          " — decode ITL ≈ " + fmtT(td) + " (reading " + K.fmtBytes(m.P * 2 / tp) + " of weights per GPU per step). " +
          (chunked ? "Chunked prefill (" + CH + " tokens per step) replaces the big spikes with many slightly slower steps — the cheap fix before disaggregating. " :
            "In the colocated engine each " + fmtT(tPre) + " prefill freezes every running stream: users see a stutter. ") +
          (tX > tPre ? "<b style='color:var(--red)'>Transfer is slower than prefill — the link is the bottleneck (try NVLink/IB or FP8 KV).</b> " : "") +
          (load >= 1 ? "<b style='color:var(--red)'>Colocated replicas are overloaded by prefill alone.</b> " : "") +
          "Disaggregation pays off at high volume with long prompts; it costs a KV transfer on the TTFT path and more moving parts.</div>";
      }
      draw();
    },
  });
})();
