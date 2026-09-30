/* inference.js — interactive explainers for the inference track:
 * generation loop, serving stack, prefill/decode, KV-cache sizing, paging, prefix caching,
 * batching, chunked prefill, latency vs throughput. */
(function () {
  "use strict";
  var V = window.Viz;

  // ---------------------------------------------------------------- shared data & helpers
  // Achievable fractions of peak used by the simple roofline models below.
  var BW_EFF = 0.8, FLOP_EFF = 0.6, STEP_OVH = 1e-3;
  var GPUS = {
    T4: { label: "NVIDIA T4 · 16 GB", mem: 16, bw: 320, tf: 65, tf8: 65, util: 0.9 },
    L4: { label: "NVIDIA L4 · 24 GB", mem: 24, bw: 300, tf: 121, tf8: 242, util: 0.9 },
    A100: { label: "A100 · 80 GB", mem: 80, bw: 2039, tf: 312, tf8: 312, util: 0.9 },
    H100: { label: "H100 SXM · 80 GB", mem: 80, bw: 3350, tf: 989, tf8: 1979, util: 0.9 },
    H200: { label: "H200 · 141 GB", mem: 141, bw: 4800, tf: 989, tf8: 1979, util: 0.9 },
    B200: { label: "B200 · 192 GB", mem: 192, bw: 8000, tf: 2250, tf8: 4500, util: 0.9 },
    M36: { label: "Mac M3 Max · 36 GB", mem: 36, bw: 300, tf: 21, tf8: 21, util: 0.75, mac: true },
    M128: { label: "Mac M3 Max · 128 GB", mem: 128, bw: 400, tf: 28, tf8: 28, util: 0.75, mac: true },
  };
  var MODELS = {
    gpt2: { label: "GPT-2 small (124M)", P: 124e6, L: 12, H: 12, kvh: 12, hd: 64, d: 768 },
    llama8: { label: "Llama-3.1-8B", P: 8.03e9, L: 32, H: 32, kvh: 8, hd: 128, d: 4096 },
    qwen7: { label: "Qwen2.5-7B", P: 7.62e9, L: 28, H: 28, kvh: 4, hd: 128, d: 3584 },
    mistral7: { label: "Mistral-7B", P: 7.24e9, L: 32, H: 32, kvh: 8, hd: 128, d: 4096 },
    llama70: { label: "Llama-3.1-70B", P: 70.6e9, L: 80, H: 64, kvh: 8, hd: 128, d: 8192 },
    dsv3: { label: "DeepSeek-V3 (671B MoE, 37B active, MLA)", P: 671e9, Pact: 37e9, L: 61, H: 128, mla: 576, d: 7168,
      moe: { shared: 17e9, routed: 654e9, k: 8, E: 256 } },
  };
  var WDT = { bf16: { label: "BF16", b: 2 }, fp8: { label: "FP8", b: 1 }, int4: { label: "INT4 (~4.5 bits w/ scales)", b: 0.5625 } };
  var KVDT = { bf16: { label: "FP16 / BF16", b: 2 }, fp8: { label: "FP8", b: 1 }, int4: { label: "INT4", b: 0.5 } };

  function opts(d, skip) {
    return Object.keys(d).filter(function (k) { return !skip || skip.indexOf(k) < 0; }).map(function (k) { return { value: k, label: d[k].label }; });
  }
  function pick(d, v, def) {
    if (v === undefined || v === null) return def;
    if (d[v]) return v;
    var lv = String(v).toLowerCase(), ks = Object.keys(d);
    for (var i = 0; i < ks.length; i++) if (ks[i].toLowerCase() === lv || d[ks[i]].label.toLowerCase().indexOf(lv) >= 0) return ks[i];
    return def;
  }
  function kvElems(m) { return m.mla ? m.L * m.mla : 2 * m.L * m.kvh * m.hd; }
  // Weight bytes read in one step with b tokens in the batch. MoE: only the experts that are hit get read.
  function wRead(m, wb, b) {
    if (!m.moe) return m.P * wb;
    var f = 1 - Math.pow(1 - m.moe.k / m.moe.E, Math.max(1, b));
    return (m.moe.shared + m.moe.routed * f) * wb;
  }
  function autoGpus(m, g, wb) {
    var need = m.P * wb / 1e9;
    for (var n = 1; n <= 8; n *= 2) if (need < g.mem * g.util * n * 0.85) return n;
    return 8;
  }
  function peakFlops(g, wk) { return (wk === "fp8" ? g.tf8 : g.tf) * 1e12; }
  function fmtMs(ms) {
    if (ms >= 1000) return (ms / 1000).toFixed(ms >= 10000 ? 1 : 2) + " s";
    if (ms >= 10) return ms.toFixed(0) + " ms";
    if (ms >= 1) return ms.toFixed(1) + " ms";
    return ms.toFixed(2) + " ms";
  }
  function fmtTok(v) { return v >= 1024 && v % 1024 === 0 ? v / 1024 + "k" : String(v); }
  function stats(items) {
    return '<div class="stat-grid">' + items.map(function (it) {
      return '<div class="stat"><small>' + it[0] + "</small><b" + (it[2] ? ' style="color:' + it[2] + '"' : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  function col(i) { return "var(--c" + ((i % 6) + 1) + ")"; }
  function fcls(i) { return "f" + ((i % 6) + 1); }
  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function mean(a) { return a.length ? a.reduce(function (x, y) { return x + y; }, 0) / a.length : 0; }
  function pctl(a, q) { var s = a.slice().sort(function (x, y) { return x - y; }); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : 0; }
  function idxSlider(K, o) {
    var vals = o.values, i0 = 0;
    vals.forEach(function (v, i) { if (Math.abs(v - o.value) < Math.abs(vals[i0] - o.value)) i0 = i; });
    return K.slider({ label: o.label, min: 0, max: vals.length - 1, step: 1, value: i0,
      fmt: function (i) { return o.fmt ? o.fmt(vals[i]) : String(vals[i]); }, onInput: function (i) { o.onInput(vals[i]); } });
  }
  function nearest(vals, v) { var b = vals[0]; vals.forEach(function (x) { if (Math.abs(x - v) < Math.abs(b - v)) b = x; }); return b; }
  function svgPt(svg, ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  function arrow(K, x1, y1, x2, y2, cls) {
    var g = K.s("g", { "pointer-events": "none" });
    g.appendChild(K.s("line", { x1: x1, y1: y1, x2: x2, y2: y2, class: "ln-thin " + (cls || "smuted") }));
    var a = Math.atan2(y2 - y1, x2 - x1), L = 6;
    var p1 = [x2 - L * Math.cos(a - 0.45), y2 - L * Math.sin(a - 0.45)], p2 = [x2 - L * Math.cos(a + 0.45), y2 - L * Math.sin(a + 0.45)];
    g.appendChild(K.s("path", { d: "M" + x2 + " " + y2 + "L" + p1[0] + " " + p1[1] + "L" + p2[0] + " " + p2[1] + "Z", class: (cls || "smuted").replace(/^s/, "f") }));
    return g;
  }
  var patId = 0;
  function hatch(K, svg) {
    var id = "inf-hatch-" + (++patId);
    var defs = K.s("defs"), pat = K.s("pattern", { id: id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.appendChild(K.s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "smuted", "stroke-width": 2 }));
    defs.appendChild(pat); svg.appendChild(defs);
    return "url(#" + id + ")";
  }
  function ticker(c, fn, ms, onStop) {
    var id = null;
    var o = {
      running: function () { return !!id; },
      start: function () { if (id) return; id = setInterval(function () { if (!c.root.isConnected) { o.stop(); return; } fn(); }, ms); },
      stop: function () { if (id) clearInterval(id); id = null; if (onStop) onStop(); },
    };
    return o;
  }

  // ---------------------------------------------------------------- generation loop
  var GEN_SCRIPTS = [
    { label: "The capital of France is", prompt: ["The", " capital", " of", " France", " is"], steps: [
      [[" Paris", 0.82], [" a", 0.06], [" the", 0.04], [" located", 0.03], [" known", 0.02]],
      [[".", 0.58], [",", 0.27], [" and", 0.07], ["!", 0.03], [" (", 0.02]],
      [[" It", 0.41], [" Paris", 0.22], [" The", 0.17], ["\n", 0.1], [" Its", 0.04]],
      [[" is", 0.72], [" has", 0.12], [" was", 0.07], ["'s", 0.05], [" lies", 0.02]],
      [[" also", 0.39], [" known", 0.26], [" the", 0.18], [" famous", 0.09], [" home", 0.05]],
      [[" the", 0.55], [" known", 0.28], [" home", 0.08], [" a", 0.05], [" one", 0.02]],
      [[" largest", 0.46], [" most", 0.23], [" country", 0.12], [" home", 0.1], [" center", 0.05]],
      [[" city", 0.89], [" metropolitan", 0.05], [" and", 0.03], [" urban", 0.02], [" French", 0.01]],
      [[" in", 0.8], [" of", 0.13], [",", 0.04], [" and", 0.02], [" on", 0.01]],
      [[" France", 0.93], [" the", 0.04], [" Europe", 0.02], [" Western", 0.005], [" French", 0.004]],
      [[".", 0.86], [",", 0.08], [" and", 0.04], ["!", 0.01], [";", 0.01]],
      [["<eos>", 0.63], [" It", 0.15], ["\n", 0.12], [" The", 0.06], [" Its", 0.02]],
    ] },
    { label: "Once upon a time", prompt: ["Once", " upon", " a", " time"], steps: [
      [[",", 0.88], [" there", 0.07], [" in", 0.03], [" lived", 0.01], ["...", 0.01]],
      [[" there", 0.62], [" in", 0.21], [" a", 0.08], [" the", 0.05], [" long", 0.03]],
      [[" was", 0.71], [" lived", 0.22], [" were", 0.04], [" once", 0.02], [" stood", 0.01]],
      [[" a", 0.83], [" an", 0.09], [" no", 0.03], [" only", 0.02], [" one", 0.02]],
      [[" little", 0.34], [" young", 0.21], [" small", 0.16], [" curious", 0.12], [" robot", 0.08]],
      [[" robot", 0.29], [" girl", 0.27], [" boy", 0.19], [" fox", 0.13], [" dragon", 0.07]],
      [[" who", 0.36], [" named", 0.33], [" that", 0.14], [".", 0.1], [" called", 0.05]],
      [[" loved", 0.44], [" wanted", 0.25], [" lived", 0.14], [" could", 0.1], [" dreamed", 0.05]],
      [[" to", 0.58], [" books", 0.17], [" the", 0.12], [" stars", 0.08], [" music", 0.04]],
      [[" read", 0.31], [" dance", 0.24], [" paint", 0.19], [" explore", 0.15], [" sing", 0.08]],
      [[".", 0.52], [" books", 0.28], [" stories", 0.12], [" at", 0.05], [" about", 0.02]],
      [["<eos>", 0.4], [" Every", 0.25], [" One", 0.2], [" She", 0.1], ["\n", 0.05]],
    ] },
    { label: "def fib(n):  (code)", prompt: ["def", " fib", "(n", "):"], steps: [
      [["\n   ", 0.93], ["\n", 0.04], [" return", 0.02], [" pass", 0.005], [" #", 0.005]],
      [[" if", 0.78], [' """', 0.12], [" return", 0.06], [" #", 0.03], [" a", 0.01]],
      [[" n", 0.94], [" not", 0.03], [" (", 0.02], [" type", 0.005], [" len", 0.005]],
      [[" <", 0.56], [" <=", 0.31], [" ==", 0.11], [" in", 0.01], [" >", 0.01]],
      [[" 2", 0.83], [" 1", 0.14], [" 0", 0.02], [" 3", 0.005], [" n", 0.005]],
      [[":", 0.97], [" or", 0.01], [" and", 0.01], [")", 0.005], [";", 0.005]],
      [[" return", 0.64], ["\n       ", 0.35], [" pass", 0.005], [" print", 0.003], [" yield", 0.002]],
      [[" n", 0.88], [" 1", 0.09], [" 0", 0.02], [" [", 0.005], [" None", 0.005]],
      [["\n   ", 0.95], ["\n", 0.04], [";", 0.005], [" #", 0.003], [" +", 0.002]],
      [[" return", 0.9], [" else", 0.07], [" a", 0.02], [" memo", 0.005], [" #", 0.005]],
      [[" fib", 0.93], [" n", 0.04], [" (", 0.02], [" self", 0.005], [" sum", 0.005]],
      [["(n", 0.98], ["(", 0.015], ["(i", 0.003], ["_", 0.001], [" (", 0.001]],
      [[" -", 0.97], ["-", 0.02], [" +", 0.005], [" //", 0.003], [" *", 0.002]],
      [[" 1", 0.96], [" 2", 0.03], [" n", 0.005], [" i", 0.003], [" k", 0.002]],
      [[")", 0.98], ["):", 0.01], ["]", 0.005], [",", 0.003], [" )", 0.002]],
      [[" +", 0.97], ["\n", 0.015], [" *", 0.01], [" -", 0.003], [" or", 0.002]],
      [[" fib", 0.99], [" n", 0.005], [" (", 0.003], [" f", 0.001], [" fibonacci", 0.001]],
      [["(n", 0.99], ["(", 0.007], ["(i", 0.001], ["_", 0.001], [" (", 0.001]],
      [[" -", 0.99], ["-", 0.007], [" +", 0.001], [" //", 0.001], [" *", 0.001]],
      [[" 2", 0.97], [" 1", 0.02], [" 3", 0.005], [" n", 0.003], [" i", 0.002]],
      [[")", 0.98], ["):", 0.01], ["]", 0.005], [",", 0.003], [" )", 0.002]],
      [["<eos>", 0.72], ["\n", 0.2], ["\n\n", 0.05], [" #", 0.02], [";", 0.01]],
    ] },
  ];
  function dispTok(t) { return t === "<eos>" ? "<eos>" : t.replace(/\n/g, "⏎").replace(/ /g, "\u00a0"); }
  function tokW(t) { return 12 + dispTok(t).length * 7.3; }
  function layoutTok(toks, x0, y0, maxW, lh) {
    var x = x0, y = y0, out = [];
    toks.forEach(function (t) {
      var w = tokW(t);
      if (x + w > x0 + maxW) { x = x0; y += lh; }
      out.push({ x: x, y: y, w: w }); x += w + 4;
    });
    return out;
  }

  V.register("generation-loop", {
    title: "An LLM writes one token at a time — and the KV cache saves the repeated work",
    desc: "Each step the model reads the context, scores every token in its vocabulary, picks one (here the most likely: <i>greedy</i>), appends it and repeats. Toggle the KV cache to see which tokens must go through the model each step — and watch the two work counters diverge.",
    render: function (c, p, K) {
      var si = K.clamp(+(p.script || 0) || 0, 0, GEN_SCRIPTS.length - 1);
      var S = GEN_SCRIPTS[si], gen = 0, ph = 0, playing = false, one = false, raf = null, last = 0, speed = +p.speed || 1, kv = p.kvCache !== false;
      c.controls.appendChild(K.select({ label: "Prompt", options: GEN_SCRIPTS.map(function (s, i) { return { value: String(i), label: s.label }; }), value: String(si), onChange: function (v) { si = +v; reset(); } }));
      var playB = K.button("▶ Play", function () { if (playing) pause(); else play(false); }, "primary");
      c.controls.appendChild(playB);
      c.controls.appendChild(K.button("Step", function () { play(true); }));
      c.controls.appendChild(K.button("Reset", function () { reset(); }));
      c.controls.appendChild(K.slider({ label: "speed", min: 0.25, max: 3, step: 0.25, value: speed, fmt: function (v) { return v + "×"; }, onInput: function (v) { speed = v; } }));
      c.controls.appendChild(K.toggle({ label: "use KV cache", value: kv, onChange: function (v) { kv = v; draw(); info(); } }));

      function loop(ts) {
        raf = null;
        if (!playing) return;
        if (!c.root.isConnected) { playing = false; return; }
        var dt = last ? Math.min(100, ts - last) : 16; last = ts;
        ph += dt * speed / 1700;
        if (ph >= 1) {
          ph = 0; gen++; info();
          if (gen >= S.steps.length || one) { pause(); return; }
        }
        draw();
        raf = requestAnimationFrame(loop);
      }
      function play(o) {
        if (gen >= S.steps.length) reset();
        if (playing && one && o) { gen++; ph = 0.0001; info(); if (gen >= S.steps.length) pause(); else draw(); return; }
        if (playing) { one = one || o; return; }
        one = o; playing = true; last = 0; playB.textContent = "⏸ Pause";
        if (ph === 0) ph = 0.0001;
        raf = requestAnimationFrame(loop);
      }
      function pause() {
        playing = false; one = false;
        if (raf) cancelAnimationFrame(raf); raf = null;
        playB.textContent = gen >= S.steps.length ? "↻ Replay" : "▶ Play";
        draw();
      }
      function reset() {
        playing = false; one = false; if (raf) cancelAnimationFrame(raf); raf = null;
        S = GEN_SCRIPTS[si]; gen = 0; ph = 0; playB.textContent = "▶ Play"; draw(); info();
      }
      function work(k) {
        var P = S.prompt.length, no = 0, w = 0;
        for (var i = 0; i < k; i++) { no += P + i; w += i === 0 ? P : 1; }
        return [no, w];
      }

      function draw() {
        var fullSeq = S.prompt.concat(S.steps.map(function (st) { return st[0][0]; })), layF = layoutTok(fullSeq, 14, 20, 652, 34);
        var my = Math.max(80, layF[layF.length - 1].y + 62), rowY0 = my + 22, by = my + 152;
        var W = 680, H = by + 120, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var P = S.prompt.length, N = S.steps.length;
        var ctx = S.prompt.concat(S.steps.slice(0, gen).map(function (st) { return st[0][0]; }));
        var active = ph > 0 && gen < N;
        var fwd = active && ph < 0.45, fwdT = active ? Math.min(1, ph / 0.45) : 0;
        var barT = active ? (ph < 0.45 ? 0 : ease(Math.min(1, (ph - 0.45) / 0.3))) : (gen > 0 ? 1 : 0);
        var cand = active ? S.steps[gen] : gen > 0 ? S.steps[gen - 1] : null;
        var nextTok = active ? S.steps[gen][0][0] : null;
        var procFrom = kv ? (gen === 0 ? 0 : ctx.length - 1) : 0;
        var nProc = ctx.length - procFrom;
        var cachedNow = kv ? (active && !fwd ? ctx.length : gen === 0 ? 0 : P + gen - 1) : 0;

        // --- context strip
        svg.appendChild(K.s("text", { x: 14, y: 12, class: "lbl-sm", text: "context = prompt (grey) + generated tokens (green)" + (fwd ? "  ·  blue = going through the model now" : "") }));
        if (kv) {
          svg.appendChild(K.s("rect", { x: 520, y: 5, width: 10, height: 6, rx: 1.5, class: "f4" }));
          svg.appendChild(K.s("rect", { x: 532, y: 5, width: 10, height: 6, rx: 1.5, class: "f6" }));
          svg.appendChild(K.s("text", { x: 547, y: 12, class: "lbl-sm", text: "= cached K and V" }));
        }
        var lay = layoutTok(nextTok !== null ? ctx.concat([nextTok]) : ctx, 14, 20, 652, 34);
        ctx.forEach(function (t, i) {
          var L = lay[i], isP = i < P, proc = fwd && i >= procFrom;
          var r = K.s("rect", { x: L.x, y: L.y, width: L.w, height: 22, rx: 5, class: proc ? "box-hl" : (isP || t === "<eos>") ? "box" : "f3 s3" });
          if (!proc && !isP && t !== "<eos>") { r.setAttribute("fill-opacity", 0.16); r.setAttribute("stroke-width", 1.2); }
          svg.appendChild(r);
          svg.appendChild(K.s("text", { x: L.x + L.w / 2, y: L.y + 15, "text-anchor": "middle", class: "lbl mono", style: t === "<eos>" ? { fill: "var(--muted)" } : null, text: dispTok(t) }));
          if (kv) {
            var fill = i < cachedNow ? 1 : proc ? fwdT : 0, cw = (L.w - 6) / 2;
            [["f4", 0], ["f6", 1]].forEach(function (q) {
              svg.appendChild(K.s("rect", { x: L.x + 2 + q[1] * (cw + 2), y: L.y + 25, width: cw, height: 5, rx: 1.5, class: fill > 0 ? q[0] : "fline", opacity: fill > 0 ? 0.25 + 0.75 * fill : 1 }));
            });
          }
        });
        if (nextTok !== null) {
          var Lt = lay[ctx.length];
          if (ph < 0.75) {
            svg.appendChild(K.s("rect", { x: Lt.x, y: Lt.y, width: Lt.w, height: 22, rx: 5, class: "box", "stroke-dasharray": "3 3" }));
            svg.appendChild(K.s("text", { x: Lt.x + Lt.w / 2, y: Lt.y + 15, "text-anchor": "middle", class: "lbl-sm", text: "?" }));
          } else {
            var f = ease((ph - 0.75) / 0.25), fx = K.lerp(398, Lt.x, f), fy = K.lerp(rowY0 - 4, Lt.y, f);
            svg.appendChild(K.s("rect", { x: fx, y: fy, width: Lt.w, height: 22, rx: 5, class: "f3 s3", "fill-opacity": 0.3, "stroke-width": 1.5 }));
            svg.appendChild(K.s("text", { x: fx + Lt.w / 2, y: fy + 15, "text-anchor": "middle", class: "lbl mono", text: dispTok(nextTok) }));
          }
        }

        // --- model box
        svg.appendChild(arrow(K, 139, my - 14, 139, my - 2));
        svg.appendChild(K.s("rect", { x: 14, y: my, width: 250, height: 126, rx: 10, class: fwd ? "box-hl" : "box" }));
        svg.appendChild(K.s("text", { x: 26, y: my + 18, class: "lbl", text: "LLM forward pass" }));
        svg.appendChild(K.s("text", { x: 252, y: my + 18, "text-anchor": "end", class: "lbl-sm", text: "transformer layers" }));
        var lit = fwd ? Math.floor(fwdT * 6.99) : -1;
        for (var j = 0; j < 6; j++) {
          svg.appendChild(K.s("rect", { x: 30, y: my + 96 - j * 12, width: 218, height: 8, rx: 3, class: j <= lit ? "f1" : "fline", opacity: j <= lit ? 0.4 + 0.6 * (j === lit ? 1 : 0.6) : 1 }));
        }
        var mtxt = fwd ? "processing " + nProc + " token" + (nProc > 1 ? "s" : "") + (kv && gen > 0 ? " (rest read from KV cache)" : kv ? " (prefill: fills the cache)" : " (no cache: all of them, again)")
          : active ? "→ one score per vocabulary token" : gen >= N ? "finished: model emitted <eos>" : gen ? "paused" : "press ▶ Play";
        svg.appendChild(K.s("text", { x: 26, y: my + 118, class: "lbl-sm", text: mtxt }));
        svg.appendChild(arrow(K, 266, my + 63, 294, my + 63));

        // --- probability bars
        svg.appendChild(K.s("text", { x: 300, y: my + 8, class: "lbl-sm", text: "next-token probabilities (top 5 of the whole vocabulary)" }));
        if (cand) {
          var other = 1;
          cand.forEach(function (cd, i) {
            var y = rowY0 + i * 19, w = Math.max(1, cd[1] * 215 * barT); other -= cd[1];
            svg.appendChild(K.s("text", { x: 392, y: y + 11, "text-anchor": "end", class: "mono", style: i === 0 ? { fill: "var(--fg)", fontWeight: 700 } : null, text: dispTok(cd[0]) }));
            svg.appendChild(K.s("rect", { x: 398, y: y, width: w, height: 14, rx: 3, class: i === 0 ? "f3" : "f1", opacity: i === 0 ? 0.9 : 0.35 }));
            if (barT > 0.3) svg.appendChild(K.s("text", { x: 398 + w + 5, y: y + 11, class: "lbl-sm mono", text: (cd[1] * 100).toFixed(cd[1] < 0.01 ? 1 : 0) + "%" + (i === 0 && barT >= 1 ? "  ← picked" : "") }));
          });
          if (barT > 0.3) svg.appendChild(K.s("text", { x: 300, y: rowY0 + 5 * 19 + 10, class: "lbl-sm", text: "all other tokens together: " + Math.max(0, other * 100).toFixed(1) + "%" }));
        } else {
          svg.appendChild(K.s("text", { x: 480, y: my + 70, "text-anchor": "middle", class: "lbl-sm", text: "(the model hasn't predicted anything yet)" }));
        }

        // --- work counters
        var wk = work(gen), wN = work(N);
        svg.appendChild(K.s("line", { x1: 14, x2: 666, y1: by - 12, y2: by - 12, class: "grid" }));
        svg.appendChild(K.s("text", { x: 14, y: by + 2, class: "lbl", text: "Total work so far (token-passes through the model)" }));
        var xs = K.scale(0, N, 60, 380), ys = K.scale(0, wN[0], by + 96, by + 22);
        svg.appendChild(K.s("line", { x1: 60, x2: 380, y1: by + 96, y2: by + 96, class: "axis" }));
        svg.appendChild(K.s("line", { x1: 60, x2: 60, y1: by + 18, y2: by + 96, class: "axis" }));
        svg.appendChild(K.s("text", { x: 56, y: by + 26, "text-anchor": "end", class: "tick", text: wN[0] }));
        svg.appendChild(K.s("text", { x: 56, y: by + 99, "text-anchor": "end", class: "tick", text: "0" }));
        svg.appendChild(K.s("text", { x: 380, y: by + 108, "text-anchor": "end", class: "tick", text: N + " tokens generated" }));
        [[0, "s2"], [1, "s3"]].forEach(function (q) {
          var full = [], done = [];
          for (var k = 0; k <= N; k++) { var v = work(k)[q[0]]; full.push([k, v]); if (k <= gen) done.push([k, v]); }
          svg.appendChild(K.s("path", { d: K.path(full, xs, ys), class: "ln-thin dash " + q[1], opacity: 0.45 }));
          if (done.length > 1) svg.appendChild(K.s("path", { d: K.path(done, xs, ys), class: "ln " + q[1] }));
          var e = done[done.length - 1];
          svg.appendChild(K.s("circle", { cx: xs(e[0]), cy: ys(e[1]), r: 4, class: q[1].replace("s", "f") }));
        });
        var bx = 412, bw = 250, mx = Math.max(1, wN[0]);
        [["without KV cache: re-process whole context every step", wk[0], "f2", 0], ["with KV cache: only the new token (after prefill)", wk[1], "f3", 1]].forEach(function (q) {
          var y = by + 20 + q[3] * 40;
          svg.appendChild(K.s("text", { x: bx, y: y, class: "lbl-sm", text: q[0] }));
          svg.appendChild(K.s("rect", { x: bx, y: y + 5, width: bw, height: 14, rx: 3, class: "fline" }));
          svg.appendChild(K.s("rect", { x: bx, y: y + 5, width: Math.max(2, q[1] / mx * bw), height: 14, rx: 3, class: q[2] }));
          svg.appendChild(K.s("text", { x: bx + bw, y: y + 16, "text-anchor": "end", class: "lbl mono", text: q[1] }));
        });
        var ratio = wk[1] ? wk[0] / wk[1] : 1;
        svg.appendChild(K.s("text", { x: bx, y: by + 100, class: "lbl", style: { fill: "var(--c2)" }, text: gen ? ratio.toFixed(1) + "× more work without the cache" : "" }));
        if (gen) svg.appendChild(K.s("text", { x: bx, y: by + 112, class: "lbl-sm", text: "…and the gap keeps growing with every token" }));
        K.clear(c.stage).appendChild(svg);
      }

      function info() {
        var P = S.prompt.length, wk = work(gen), P2 = 1000, N2 = 1000;
        var no2 = N2 * P2 + N2 * (N2 - 1) / 2, w2 = P2 + N2 - 1;
        c.readout.innerHTML = stats([
          ["prompt tokens P", P], ["generated N", gen], ["context now", P + gen],
          ["work w/o cache", wk[0], "var(--c2)"], ["work with cache", wk[1], "var(--c3)"],
        ]) + "<div style='margin-top:8px'>Without a cache, step <i>i</i> re-runs all " + K.tex("P+i") + " tokens: " +
          K.tex("\\sum_{i=0}^{N-1}(P+i) = NP + \\tfrac{N(N-1)}{2}") + " — quadratic, " + K.tex("O(N^2)") + ". With a KV cache, the prompt is processed once (<b>prefill</b>) and each later step runs just the 1 new token (<b>decode</b>): " +
          K.tex("P + (N-1)") + " — linear. For P = N = 1000 that is <b>" + K.fmtNum(no2, 0) + "</b> vs <b>" + K.fmtNum(w2, 0) + "</b> token-passes (" + Math.round(no2 / w2) + "×)." +
          " <span class='muted'>Honest footnote: with the cache, attention still <i>reads</i> every cached K,V each step (" + K.tex("O(n)") + " memory traffic per token) — that is why long contexts slow decode and why KV memory is the #1 resource in serving.</span></div>";
      }
      draw(); info();
      if (p.autoplay !== false) setTimeout(function () { if (c.root.isConnected && !playing && gen === 0 && ph === 0) play(false); }, 700);
    },
  });

  // ---------------------------------------------------------------- inference stack
  var STACK = [
    { id: "client", name: "Client & API", sub: "OpenAI-compatible HTTP · streaming (SSE)", h: 42 },
    { id: "gateway", name: "Gateway", sub: "auth · rate limits · quotas · metering", h: 42 },
    { id: "router", name: "Cluster: router & autoscaler", h: 58, kids: [["router", "Router (load / KV-cache aware)"], ["autoscaler", "Autoscaler (replicas)"]] },
    { id: "engine", name: "Inference engine (one replica)", h: 84, kids: [["scheduler", "Scheduler", "continuous batching"], ["kvmgr", "KV cache manager", "paged blocks · prefix cache"], ["sampler", "Tokenizer · sampler", "logits → token → text"]] },
    { id: "runtime", name: "Model runtime", sub: "forward pass · weights · CUDA graphs · torch.compile", h: 42 },
    { id: "kernels", name: "Kernels", sub: "GEMM · FlashAttention · fused norms/RoPE · MoE · all-reduce", h: 42 },
    { id: "gpu", name: "GPU hardware", sub: "SMs + tensor cores · HBM · NVLink · network", h: 42 },
  ];
  var DETAIL = {
    client: { t: "Client & API", what: "Your app (or the OpenAI SDK) sends <code>POST /v1/chat/completions</code> with messages and sampling params (temperature, max_tokens). With <code>stream: true</code> the server replies with Server-Sent Events — one small JSON chunk per generated token.",
      tech: ["OpenAI API / SDK", "LiteLLM (SDK)", "curl + SSE", "Anthropic / Gemini APIs"], nums: "User-visible latency: <b>TTFT</b> (time to first token) and tokens/s while streaming. Typical chat targets: TTFT &lt; 0.5–1 s, ≥ 20–50 tok/s per user.", ana: "A normal REST client — the only twist is a long-lived streaming response." },
    gateway: { t: "Gateway", what: "The front door: TLS, API keys, per-tenant rate limits and <i>token</i> quotas, request logging and metering for billing, mapping a model name to a backend pool, fallbacks between providers.",
      tech: ["Envoy / Envoy AI Gateway", "LiteLLM proxy", "Kong / NGINX", "cloud API gateways"], nums: "Adds ~1–10 ms. Limits are usually tokens/min, not just requests/min.", ana: "Exactly the API gateway / reverse proxy you already know." },
    router: { t: "Router (cluster load balancer)", what: "Chooses which model replica serves the request. Round-robin wastes the KV cache; smart routers send requests sharing a prefix (system prompt, chat history) to the replica that already has it cached, and balance by queue depth and KV usage. With prefill/decode disaggregation it also pairs a prefill worker with a decode worker.",
      tech: ["llm-d inference scheduler", "NVIDIA Dynamo router", "SGLang router", "Gateway API Inference Extension", "KServe"], nums: "Random routing over N replicas gives a prefix-cache hit only ~1/N of the time; cache-aware routing can hit most of the time.", ana: "An L7 load balancer with sticky sessions — stickiness by prompt prefix, and the 'session state' is GBs of KV cache on a GPU." },
    autoscaler: { t: "Autoscaler", what: "Adds and removes GPU replicas as traffic changes. Cold start is the enemy: provisioning a node, pulling the image and loading 16–140 GB of weights takes tens of seconds to minutes, so you scale on queue depth / KV utilization, keep warm capacity and cache weights near the GPUs.",
      tech: ["Kubernetes HPA / KEDA", "KServe / Knative", "Ray Serve", "Baseten / Modal platforms"], nums: "Scale-up delay = node + image pull + weight load + CUDA-graph capture.", ana: "Autoscaling web servers — except each instance costs $2–10 per hour and boots in minutes, not seconds." },
    engine: { t: "Inference engine", what: "The heart of serving — this is what vLLM, SGLang and TensorRT-LLM are. A loop runs every few milliseconds: the scheduler picks which requests run, the KV cache manager gives them memory, the runtime does one forward pass for the whole batch, the sampler picks next tokens, finished requests stream out.",
      tech: ["vLLM", "SGLang", "TensorRT-LLM", "TGI", "llama.cpp / Ollama", "MLX-LM"], nums: "Watch: throughput (tok/s), running batch size, KV cache usage %, queue length, TTFT / TPOT percentiles.", ana: "An event loop / job scheduler whose scarce resource is GPU memory rather than CPU time." },
    scheduler: { t: "Scheduler", what: "Every iteration decides the batch: which waiting requests to admit, how many prefill tokens to mix in with the decodes (chunked-prefill token budget), and whom to preempt when KV memory runs out. <b>Continuous batching</b> lets requests join and leave the batch at every step instead of waiting for the slowest one.",
      tech: ["vLLM V1 scheduler", "SGLang scheduler", "TensorRT-LLM in-flight batching"], nums: "Knobs: <code>max_num_seqs</code>, <code>max_num_batched_tokens</code>, FCFS vs priority policy.", ana: "An OS scheduler time-slicing processes — here all requests share every GPU step." },
    kvmgr: { t: "KV cache manager", what: "Owns the GPU memory for keys/values of every token of every running request. Splits it into fixed-size blocks (<b>PagedAttention</b>), keeps a block table per request, frees blocks when requests finish, and reuses blocks for shared prefixes (prefix caching / RadixAttention). Can offload to CPU RAM or SSD.",
      tech: ["vLLM PagedAttention + prefix caching", "SGLang RadixAttention", "LMCache", "TRT-LLM KV reuse"], nums: "Llama-3.1-8B in BF16 needs 128 KiB of KV per token. Knob: <code>gpu_memory_utilization</code>.", ana: "Virtual memory: pages, page tables, a free list, copy-on-write sharing, swapping." },
    sampler: { t: "Tokenizer & sampler", what: "Turns text into token IDs before the model runs; after each step turns the final logits into the next token (temperature, top-k/top-p, penalties, JSON-grammar masks for structured output, stop sequences) and detokenizes incrementally for streaming.",
      tech: ["HF tokenizers / tiktoken", "xgrammar / outlines", "FlashInfer sampling kernels"], nums: "Runs every step for every sequence in the batch — must be cheap.", ana: "Serialization / deserialization at the edges of the model." },
    runtime: { t: "Model runtime", what: "Executes the model for a batch of tokens: embedding → N transformer blocks → LM head. Loads weights (maybe quantized), shards them across GPUs (tensor / pipeline parallel) and removes Python/CPU overhead with CUDA graphs and torch.compile.",
      tech: ["PyTorch", "CUDA graphs", "torch.compile", "TensorRT", "MLX (Mac)"], nums: "Every decode step must read every weight once: 16 GB for an 8B model in BF16.", ana: "The interpreter / JIT that runs your program — the program is the model's forward pass." },
    kernels: { t: "Kernels", what: "The GPU functions that do the math: GEMMs (matrix multiplies) for linear layers, attention (FlashAttention streams K/V in tiles without materializing the T×T matrix), fused RMSNorm/RoPE/activations, MoE expert kernels, all-reduce between GPUs. Fusing ops avoids round-trips to slow memory.",
      tech: ["FlashAttention / FlashInfer", "cuBLAS / CUTLASS", "Triton", "NCCL", "custom CUDA"], nums: "Good GEMMs hit 60–80% of peak FLOPS; decode attention should approach peak memory bandwidth.", ana: "Hand-tuned SIMD hot loops — where most of the raw speed lives." },
    gpu: { t: "GPU hardware", what: "Streaming multiprocessors (SMs) with tensor cores do the math; HBM holds weights + KV cache; NVLink connects GPUs in a node, InfiniBand/Ethernet connects nodes. Decode is usually limited by HBM <i>bandwidth</i>, prefill by tensor-core <i>FLOPS</i>.",
      tech: ["H100: 132 SMs · 80 GB HBM3 · 3.35 TB/s · 989 TFLOPS BF16", "NVLink 900 GB/s", "B200: 192 GB · 8 TB/s", "Apple unified memory"], nums: "H100 ridge point ≈ 989e12 / 3.35e12 ≈ 295 FLOPs per byte — do less math per byte than that and you are memory-bound.", ana: "The CPU + RAM of this world — ~50× the memory bandwidth, but tiny, precious memory." },
  };
  var TRACE = [
    ["client", "① Client sends POST /v1/chat/completions {model, messages, stream: true}"],
    ["gateway", "② Gateway checks the API key, the tenant's token quota and rate limit, logs the request"],
    ["router", "③ Router sees replica 3 already caches this system prompt → routes there (KV-cache-aware)"],
    ["scheduler", "④ Scheduler queues the request and admits it into the next iteration's batch"],
    ["kvmgr", "⑤ KV manager maps 12 cached prefix blocks and allocates 3 new blocks for the rest of the prompt"],
    ["runtime", "⑥ Runtime runs prefill: one forward pass over the new prompt tokens (compute-bound)"],
    ["kernels", "⑦ Kernels: GEMMs for the linear layers + FlashAttention over the cached K/V"],
    ["gpu", "⑧ GPU: tensor cores crunch the prefill; every later decode step re-reads all weights from HBM"],
    ["sampler", "⑨ Sampler picks the next token from the logits → detokenized to text"],
    ["client", "⑩ Token streamed back as an SSE chunk; steps ⑥–⑨ repeat per token until stop / max_tokens"],
  ];

  V.register("inference-stack", {
    title: "The inference stack: everything between a prompt and a GPU",
    desc: "Click any layer (or sub-box) for what lives there and the real tech used. <b>▶ Trace a request</b> follows one chat request down the stack and its tokens back up.",
    render: function (c, p, K) {
      var sel = DETAIL[p.layer] ? p.layer : "engine", refs = {}, centers = {}, trace = -1, timer = null, raf = null;
      var W = 420, y = 8, svg = K.s("svg", { class: "viz-svg" });
      var nt = { "pointer-events": "none" };
      STACK.forEach(function (L, li) {
        var r = K.s("rect", { x: 10, y: y, width: 400, height: L.h, rx: 9, class: "box", style: { cursor: "pointer" } });
        r.addEventListener("click", function () { select(L.id); });
        svg.appendChild(r); refs[L.id] = r; centers[L.id] = [396, y + 14];
        svg.appendChild(K.s("rect", Object.assign({ x: 15, y: y + 6, width: 5, height: L.h - 12, rx: 2.5, class: fcls(li) }, nt)));
        svg.appendChild(K.s("text", Object.assign({ x: 28, y: y + 17, class: "lbl", text: L.name }, nt)));
        if (L.sub) svg.appendChild(K.s("text", Object.assign({ x: 28, y: y + 33, class: "lbl-sm", text: L.sub }, nt)));
        if (L.kids) {
          var kw = (400 - 30 - (L.kids.length - 1) * 8) / L.kids.length;
          L.kids.forEach(function (k, ki) {
            var kx = 28 + ki * (kw + 8), ky = y + 24, kh = L.h - 30;
            var kr = K.s("rect", { x: kx, y: ky, width: kw, height: kh, rx: 7, class: "box", style: { cursor: "pointer" } });
            kr.addEventListener("click", function () { select(k[0]); });
            svg.appendChild(kr); refs[k[0]] = kr; centers[k[0]] = [kx + kw / 2, ky + kh / 2];
            svg.appendChild(K.s("text", Object.assign({ x: kx + kw / 2, y: ky + (k[2] ? kh / 2 - 2 : kh / 2 + 4), "text-anchor": "middle", class: "lbl", style: { fontSize: "11.5px" }, text: k[1] }, nt)));
            if (k[2]) svg.appendChild(K.s("text", Object.assign({ x: kx + kw / 2, y: ky + kh / 2 + 12, "text-anchor": "middle", class: "lbl-sm", text: k[2] }, nt)));
          });
        }
        y += L.h;
        if (li < STACK.length - 1) {
          svg.appendChild(arrow(K, 60, y + 1, 60, y + 13, "s1"));
          svg.appendChild(arrow(K, 360, y + 13, 360, y + 1, "s3"));
          if (li === 0) {
            svg.appendChild(K.s("text", Object.assign({ x: 68, y: y + 11, class: "lbl-sm", text: "request" }, nt)));
            svg.appendChild(K.s("text", Object.assign({ x: 352, y: y + 11, "text-anchor": "end", class: "lbl-sm", text: "tokens (stream)" }, nt)));
          }
          y += 14;
        }
      });
      svg.setAttribute("viewBox", "0 0 " + W + " " + (y + 8));
      var dot = K.s("circle", { r: 7, class: "f2", opacity: 0, "pointer-events": "none" });
      svg.appendChild(dot);
      var detail = K.h("div", { style: { border: "1px solid var(--line)", borderRadius: "10px", padding: "10px 14px", background: "var(--bg2)", fontSize: "13.5px", lineHeight: "1.55", alignSelf: "flex-start" } });
      var row = K.h("div", { class: "viz-row", style: { alignItems: "flex-start" } }, K.h("div", { style: { flex: "1.15", minWidth: "280px" } }, svg), detail);
      c.stage.appendChild(row);

      var tb = K.button("▶ Trace a request", function () { if (timer) stopTrace(); else startTrace(); }, "primary");
      c.controls.appendChild(tb);
      c.controls.appendChild(K.button("Step ▸", function () { stopTrace(); advance(); }));

      function select(id) {
        sel = id;
        Object.keys(refs).forEach(function (k) { refs[k].setAttribute("class", k === id ? "box-hl" : "box"); });
        var d = DETAIL[id];
        detail.innerHTML = "<div style='font-weight:650;font-size:15px;margin-bottom:4px'>" + d.t + "</div>" +
          "<div>" + d.what + "</div>" +
          "<div style='margin-top:8px'><small class='muted'>Real-world tech</small><br>" + d.tech.map(function (t) { return "<span class='tok'>" + t + "</span>"; }).join("") + "</div>" +
          "<div style='margin-top:8px'><b>Numbers & knobs:</b> " + d.nums + "</div>" +
          "<div class='muted' style='margin-top:6px'><b>Backend analogy:</b> " + d.ana + "</div>";
      }
      function moveDot(id) {
        var from = [+dot.getAttribute("cx") || centers[id][0], +dot.getAttribute("cy") || centers[id][1]], to = centers[id], t0 = null;
        dot.setAttribute("opacity", 1);
        if (raf) cancelAnimationFrame(raf);
        function fr(ts) {
          if (!c.root.isConnected) return;
          if (t0 === null) t0 = ts;
          var f = ease(Math.min(1, (ts - t0) / 450));
          dot.setAttribute("cx", K.lerp(from[0], to[0], f)); dot.setAttribute("cy", K.lerp(from[1], to[1], f));
          if (f < 1) raf = requestAnimationFrame(fr);
        }
        raf = requestAnimationFrame(fr);
      }
      function advance() {
        trace = (trace + 1) % TRACE.length;
        var tr = TRACE[trace];
        select(tr[0]); moveDot(tr[0]);
        c.readout.innerHTML = "<b>" + tr[1] + "</b><br><span class='muted'>Step " + (trace + 1) + " of " + TRACE.length + ". Each decode step (⑥–⑨) takes ~5–50 ms; the whole loop repeats once per output token.</span>";
      }
      function startTrace() {
        if (trace >= TRACE.length - 1) trace = -1;
        tb.textContent = "⏸ Pause"; advance();
        timer = setInterval(function () {
          if (!c.root.isConnected) { stopTrace(); return; }
          if (trace >= TRACE.length - 1) { stopTrace(); return; }
          advance();
        }, 1900);
      }
      function stopTrace() { if (timer) clearInterval(timer); timer = null; tb.textContent = "▶ Trace a request"; }
      select(sel);
      c.readout.innerHTML = "Click a layer. Top half = normal backend engineering (HTTP, gateways, load balancing). Bottom half = where inference engineering gets special: batching, KV memory, kernels, bandwidth.";
    },
  });

  // ---------------------------------------------------------------- prefill vs decode timeline
  V.register("prefill-decode", {
    title: "Anatomy of one request: queue → prefill (TTFT) → decode steps (TPOT)",
    desc: "Prefill processes the whole prompt in one big parallel pass (lots of math per byte → <b>compute-bound</b>). Decode makes one token per step and must re-read all weights each time (little math per byte → <b>memory-bound</b>). Numbers come from a simple roofline model of real GPUs.",
    render: function (c, p, K) {
      var mk = pick(MODELS, p.model, "llama8"), gk = pick(GPUS, p.gpu, "H100");
      if (mk === "dsv3") mk = "llama8";
      var PR = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], OUT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], BT = [1, 2, 4, 8, 16, 32, 64, 128, 256];
      var P = nearest(PR, +p.prompt || 1024), N = nearest(OUT, +p.output || 256), B = nearest(BT, +p.batch || 1), Q = +p.queue || 0;
      c.controls.appendChild(K.select({ label: "model", options: opts(MODELS, ["dsv3"]), value: mk, onChange: function (v) { mk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "GPU", options: opts(GPUS), value: gk, onChange: function (v) { gk = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "prompt tokens", values: PR, value: P, onInput: function (v) { P = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "output tokens", values: OUT, value: N, onInput: function (v) { N = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "batch (concurrent users)", values: BT, value: B, onInput: function (v) { B = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "queue wait", min: 0, max: 2000, step: 50, value: Q, fmt: function (v) { return v + " ms"; }, onInput: function (v) { Q = v; draw(); } }));

      function calc() {
        var m = MODELS[mk], g = GPUS[gk], n = autoGpus(m, g, 2);
        var bw = g.bw * 1e9 * n * BW_EFF, fl = g.tf * 1e12 * n * FLOP_EFF, W = m.P * 2, kvT = kvElems(m) * 2;
        var fP = 2 * m.P * P + 2 * m.L * m.d * P * P, bP = W + kvT * P;
        var pc = fP / fl * 1e3, pm = bP / bw * 1e3, ttft = Math.max(pc, pm) + STEP_OVH * 1e3;
        var steps = [], sum = 0, dMid = null;
        for (var i = 1; i < N; i++) {
          var cx = P + i, by = W + B * kvT * cx, fd = B * (2 * m.P + 4 * m.L * m.d * cx);
          var t = Math.max(by / bw, fd / fl) * 1e3 + STEP_OVH * 1e3;
          steps.push(t); sum += t;
          if (i === Math.max(1, Math.floor(N / 2))) dMid = { c: fd / fl * 1e3, m: by / bw * 1e3, ai: fd / by };
        }
        if (!dMid) { var cx0 = P + 1, by0 = W + B * kvT * cx0, fd0 = B * (2 * m.P + 4 * m.L * m.d * cx0); dMid = { c: fd0 / fl * 1e3, m: by0 / bw * 1e3, ai: fd0 / by0 }; }
        var memNeed = W + B * kvT * (P + N), memHave = g.mem * 1e9 * g.util * n;
        return { m: m, g: g, n: n, ttft: ttft, pc: pc, pm: pm, aiP: fP / bP, steps: steps, dec: sum, tpot: steps.length ? sum / steps.length : 0, d: dMid,
          e2e: Q + ttft + sum, ridge: fl / bw, fits: memNeed <= memHave, memNeed: memNeed, memHave: memHave };
      }
      function draw() {
        var r = calc(), Wd = 680, svg = K.s("svg", { viewBox: "0 0 " + Wd + " 350", class: "viz-svg" });
        var X0 = 20, X1 = 660;
        function lane(y, t0, t1, title, nSteps) {
          var sx = K.scale(t0, t1, X0, X1);
          svg.appendChild(K.s("text", { x: X0, y: y - 14, class: "lbl", text: title }));
          svg.appendChild(K.s("rect", { x: X0, y: y, width: X1 - X0, height: 30, rx: 4, class: "fbg3" }));
          var q0 = sx(Math.max(t0, 0)), q1 = sx(Math.min(Q, t1));
          if (Q > t0 && q1 > q0) svg.appendChild(K.s("rect", { x: q0, y: y, width: q1 - q0, height: 30, class: "fmuted", opacity: 0.35 }));
          var p0 = sx(Math.max(Q, t0)), p1 = sx(Math.min(Q + r.ttft, t1));
          if (p1 > p0) svg.appendChild(K.s("rect", { x: p0, y: y, width: Math.max(1.5, p1 - p0), height: 30, class: "f2" }));
          var t = Q + r.ttft, k = Math.min(nSteps, r.steps.length), px = (sx(t + r.tpot) - sx(t));
          if (px >= 2.5 || k <= 12) {
            for (var i = 0; i < k && t < t1; i++) {
              var a = sx(t), b = sx(Math.min(t1, t + r.steps[i]));
              svg.appendChild(K.s("rect", { x: a, y: y, width: Math.max(0.8, b - a - (px > 4 ? 1 : 0.3)), height: 30, class: "f1", opacity: i % 2 ? 0.6 : 0.85 }));
              if (b - a > 24) svg.appendChild(K.s("text", { x: (a + b) / 2, y: y + 19, "text-anchor": "middle", class: "t-white", style: { fontSize: "10.5px" }, text: "t" + (i + 2) }));
              t += r.steps[i];
            }
          } else {
            var e = Math.min(t1, t + r.dec);
            svg.appendChild(K.s("rect", { x: sx(t), y: y, width: Math.max(1, sx(e) - sx(t)), height: 30, class: "f1", opacity: 0.8 }));
          }
          if (p1 - p0 > 44) svg.appendChild(K.s("text", { x: (p0 + p1) / 2, y: y + 19, "text-anchor": "middle", class: "t-white", style: { fontSize: "10.5px" }, text: "prefill" }));
          if (Q > t0 && q1 - q0 > 44) svg.appendChild(K.s("text", { x: (q0 + q1) / 2, y: y + 19, "text-anchor": "middle", class: "lbl-sm", text: "queue" }));
          return sx;
        }
        // full request
        var sx = lane(34, 0, r.e2e * 1.0001, "Full request · end-to-end " + fmtMs(r.e2e) + (r.steps.length ? " · " + r.steps.length + " decode steps" : ""), 1e9);
        var tx = sx(Q + r.ttft);
        svg.appendChild(K.s("line", { x1: tx, x2: tx, y1: 28, y2: 72, class: "ln-thin s5" }));
        svg.appendChild(K.s("text", { x: Math.min(tx + 4, 560), y: 84, class: "lbl-sm", style: { fill: "var(--c5)" }, text: "first token (TTFT " + fmtMs(Q + r.ttft) + ")" }));
        svg.appendChild(K.s("text", { x: X1, y: 84, "text-anchor": "end", class: "lbl-sm", text: "last token" }));
        // zoom
        var kz = Math.min(8, r.steps.length), zEnd = Q + r.ttft + r.steps.slice(0, kz).reduce(function (a, b) { return a + b; }, 0);
        var zs = lane(122, Q, zEnd * 1.0001, "Zoom: prefill + first " + kz + " decode steps (" + fmtMs(zEnd - Q) + ")", kz);
        if (kz) {
          var a0 = zs(Q + r.ttft), a1 = zs(Q + r.ttft + r.steps[0]);
          svg.appendChild(K.s("line", { x1: a0, x2: a1, y1: 160, y2: 160, class: "ln-thin s1" }));
          svg.appendChild(K.s("text", { x: (a0 + a1) / 2, y: 172, "text-anchor": "middle", class: "lbl-sm", style: { fill: "var(--c1)" }, text: "TPOT " + fmtMs(r.steps[0]) }));
        }
        var z0 = zs(Q), z1 = zs(Q + r.ttft);
        svg.appendChild(K.s("line", { x1: z0, x2: z1, y1: 160, y2: 160, class: "ln-thin s2" }));
        svg.appendChild(K.s("text", { x: Math.max(z0 + 2, Math.min((z0 + z1) / 2, 600)), y: 172, "text-anchor": z1 - z0 > 100 ? "middle" : "start", class: "lbl-sm", style: { fill: "var(--c2)" }, text: "prefill " + fmtMs(r.ttft) + " → token 1" }));

        // bound bars
        var by0 = 204;
        svg.appendChild(K.s("text", { x: X0, y: by0 - 8, class: "lbl", text: "Why: time one step would take if limited only by compute vs only by memory (log scale)" }));
        var vals = [r.pc, r.pm, r.d.c, r.d.m], lo = Math.min.apply(null, vals) / 3, hi = Math.max.apply(null, vals) * 1.5;
        var ls = K.scale(lo, hi, 170, 530, true);
        [["prefill · compute", r.pc, r.pc >= r.pm], ["prefill · memory", r.pm, r.pm > r.pc], ["decode step · compute", r.d.c, r.d.c >= r.d.m], ["decode step · memory", r.d.m, r.d.m > r.d.c]].forEach(function (q, i) {
          var y = by0 + i * 26 + (i >= 2 ? 14 : 0);
          svg.appendChild(K.s("text", { x: 162, y: y + 12, "text-anchor": "end", class: "lbl-sm", text: q[0] }));
          svg.appendChild(K.s("rect", { x: 170, y: y, width: Math.max(2, ls(q[1]) - 170), height: 16, rx: 3, class: i % 2 ? "f1" : "f2", opacity: q[2] ? 0.95 : 0.3 }));
          svg.appendChild(K.s("text", { x: ls(q[1]) + 6, y: y + 12, class: "lbl-sm mono", text: fmtMs(q[1]) + (q[2] ? (i % 2 ? "  ← memory-bound" : "  ← compute-bound") : "") }));
        });
        K.clear(c.stage).appendChild(svg);

        c.readout.innerHTML = stats([
          ["TTFT (queue + prefill)", fmtMs(Q + r.ttft)], ["TPOT (avg per token)", fmtMs(r.tpot || 0)], ["per-user speed", (r.tpot ? (1000 / r.tpot).toFixed(0) : "—") + " tok/s"],
          ["all " + B + " users", (r.tpot ? (B * 1000 / r.tpot).toFixed(0) : "—") + " tok/s"], ["end-to-end", fmtMs(r.e2e)], ["GPUs (tensor parallel)", r.n + "× " + gk],
        ]) + "<div style='margin-top:8px'>" + K.tex("E2E = \\text{queue} + \\text{TTFT} + (N-1)\\cdot\\text{TPOT}") +
          ". &nbsp;Arithmetic intensity (FLOPs per byte moved): prefill ≈ <b>" + K.fmtNum(r.aiP, 0) + "</b>, decode ≈ <b>" + K.fmtNum(r.d.ai, 1) + "</b>, vs this GPU's ridge point ≈ <b>" + K.fmtNum(r.ridge, 0) +
          "</b>. Above the ridge you wait on math, below it you wait on memory. Decode's intensity ≈ batch size — which is exactly why batching users is nearly free until the ridge. " +
          (r.fits ? "" : "<b style='color:var(--red)'>Warning: weights + KV (" + K.fmtBytes(r.memNeed) + ") exceed usable memory (" + K.fmtBytes(r.memHave) + ") — lower batch/context.</b> ") +
          "<span class='muted'>Model: " + Math.round(BW_EFF * 100) + "% of peak bandwidth, " + Math.round(FLOP_EFF * 100) + "% of peak BF16 FLOPS, +1 ms/step overhead.</span></div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- KV cache calculator
  V.register("kv-calculator", {
    title: "KV cache calculator: how many users fit on this GPU?",
    desc: "Every token of every running request keeps its keys and values in GPU memory. Pick a model, precision, context and concurrency and see where the memory goes.",
    render: function (c, p, K) {
      var CTX = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], BT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
      var mk = pick(MODELS, p.model, "llama8"), kk = pick(KVDT, p.kvDtype, "bf16"), wk = pick(WDT, p.weightsDtype, "bf16"), gk = pick(GPUS, p.gpu, "H100");
      var ctx = nearest(CTX, +p.context || 8192), B = nearest(BT, +p.batch || 32), ng = [1, 2, 4, 8].indexOf(+p.gpus) >= 0 ? +p.gpus : 1;
      c.controls.appendChild(K.select({ label: "model", options: opts(MODELS), value: mk, onChange: function (v) { mk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "weights", options: opts(WDT), value: wk, onChange: function (v) { wk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "KV cache", options: opts(KVDT), value: kk, onChange: function (v) { kk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "GPU", options: opts(GPUS), value: gk, onChange: function (v) { gk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "# GPUs", options: ["1", "2", "4", "8"], value: String(ng), onChange: function (v) { ng = +v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "context / request", values: CTX, value: ctx, fmt: fmtTok, onInput: function (v) { ctx = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "concurrent requests", values: BT, value: B, onInput: function (v) { B = v; draw(); } }));

      function calc(kvb, cx) {
        var m = MODELS[mk], g = GPUS[gk];
        var kvTok = kvElems(m) * kvb, perSeq = kvTok * cx, W = m.P * WDT[wk].b;
        var total = g.mem * 1e9 * ng, usable = total * g.util, reserve = (g.mac ? 1 : 1.5) * 1e9 * ng, freeKV = usable - W - reserve;
        return { kvTok: kvTok, perSeq: perSeq, W: W, total: total, usable: usable, reserve: reserve, freeKV: freeKV,
          maxSeq: freeKV > 0 ? Math.floor(freeKV / perSeq) : 0, maxTok: freeKV > 0 ? Math.floor(freeKV / kvTok) : 0 };
      }
      function draw() {
        var m = MODELS[mk], g = GPUS[gk], kb = KVDT[kk].b, r = calc(kb, ctx), need = r.perSeq * B;
        K.clear(c.stage);
        // ---- memory bar
        var svg = K.s("svg", { viewBox: "0 0 680 114", class: "viz-svg" }), X0 = 14, BW = 652, sc = function (b) { return b / r.total * BW; };
        svg.appendChild(K.s("text", { x: X0, y: 14, class: "lbl", text: "GPU memory: " + (ng > 1 ? ng + " × " : "") + g.label + " = " + K.fmtBytes(r.total) + (ng > 1 ? " (weights & KV sharded, tensor parallel)" : "") }));
        var segs = [], x = X0, over = 0;
        var wUse = Math.min(r.W, r.usable);
        segs.push(["weights", wUse, "f1", K.fmtBytes(r.W)]);
        var left = r.usable - wUse, res = Math.min(r.reserve, left); left -= res;
        segs.push(["activations / runtime", res, "fmuted", K.fmtBytes(r.reserve)]);
        var kvUse = Math.min(need, Math.max(0, left));
        segs.push(["KV cache for your " + B + " requests", kvUse, "f2", K.fmtBytes(need)]);
        segs.push(["free KV space", Math.max(0, left - kvUse), "fline", K.fmtBytes(Math.max(0, left - kvUse))]);
        segs.push([g.mac ? "reserved by macOS (~25%)" : "headroom (gpu_memory_utilization = 0.9)", r.total - r.usable, "fbg3", K.fmtBytes(r.total - r.usable)]);
        if (r.W > r.usable) over = r.W - r.usable; else if (need > Math.max(0, left)) over = need - Math.max(0, left);
        segs.forEach(function (sg, i) {
          var w = sc(sg[1]); if (w <= 0) return;
          var rc = K.s("rect", { x: x, y: 24, width: w, height: 34, class: sg[2], opacity: sg[2] === "fmuted" ? 0.55 : 1 });
          if (sg[2] === "fbg3" || sg[2] === "fline") { rc.setAttribute("stroke", "var(--line2)"); rc.setAttribute("stroke-width", 0.8); }
          rc.appendChild(K.s("title", { text: sg[0] + ": " + K.fmtBytes(sg[1]) }));
          svg.appendChild(rc);
          if (w > 70 && (sg[2] === "f1" || sg[2] === "f2")) svg.appendChild(K.s("text", { x: x + w / 2, y: 45, "text-anchor": "middle", class: "t-white", style: { fontSize: "11.5px", fontWeight: 600 }, text: (sg[2] === "f1" ? "weights " + sg[3] : "KV " + K.fmtBytes(sg[1])) }));
          x += w;
        });
        var lx = X0;
        segs.forEach(function (sg) {
          var t = sg[0] + " " + sg[3], w = 16 + t.length * 5.6;
          if (lx + w > 670) return;
          svg.appendChild(K.s("rect", { x: lx, y: 68, width: 10, height: 10, rx: 2, class: sg[2], opacity: sg[2] === "fmuted" ? 0.55 : 1, stroke: "var(--line2)", "stroke-width": 0.6 }));
          svg.appendChild(K.s("text", { x: lx + 14, y: 77, class: "lbl-sm", text: t }));
          lx += w + 10;
        });
        var msg = r.W > r.usable ? "✗ The weights alone (" + K.fmtBytes(r.W) + ") don't fit in usable memory — quantize, or add GPUs." :
          over > 0 ? "✗ Short by " + K.fmtBytes(over) + ": only " + r.maxSeq + " requests of " + fmtTok(ctx) + " tokens fit (the engine would queue or preempt the rest)." :
          "✓ Fits. Room for up to " + K.fmtNum(r.maxSeq, 0) + " concurrent requests at " + fmtTok(ctx) + " tokens each.";
        svg.appendChild(K.s("text", { x: X0, y: 104, class: "lbl", style: { fill: over > 0 || r.W > r.usable ? "var(--red)" : "var(--green)" }, text: msg }));
        c.stage.appendChild(svg);

        // ---- chart: max concurrency vs context for each KV dtype
        var series = Object.keys(KVDT).map(function (k) { return { k: k, pts: CTX.map(function (cx) { return [cx, calc(KVDT[k].b, cx).maxSeq]; }) }; });
        var top = Math.max(10, B * 2, series[2].pts[0][1] * 1.5);
        var ch = K.chart({ w: 680, h: 250, x: [512, 131072], y: [1, Math.pow(10, Math.ceil(Math.log10(top)))], xLog: true, yLog: true,
          xTicks: CTX, xFmt: fmtTok, xLabel: "context length per request (tokens)", yLabel: "max concurrent requests" });
        if (r.freeKV > 0) {
          series.forEach(function (sr, i) {
            var pts = sr.pts.filter(function (q) { return q[1] >= 1; });
            if (pts.length > 1) ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: (sr.k === kk ? "ln " : "ln-thin dash ") + ["s2", "s4", "s6"][i], opacity: sr.k === kk ? 1 : 0.7 }));
            var lx2 = ch.W - ch.pad.r - 330 + i * 110, ly2 = ch.pad.t + 12;
            ch.g.appendChild(K.s("line", { x1: lx2, x2: lx2 + 18, y1: ly2 - 4, y2: ly2 - 4, class: (sr.k === kk ? "ln " : "ln-thin dash ") + ["s2", "s4", "s6"][i] }));
            ch.g.appendChild(K.s("text", { x: lx2 + 22, y: ly2, class: "lbl-sm", style: { fill: "var(--c" + [2, 4, 6][i] + ")", fontWeight: sr.k === kk ? 700 : 400 }, text: "KV " + KVDT[sr.k].label }));
          });
          ch.g.appendChild(K.s("line", { x1: ch.sx(512), x2: ch.sx(131072), y1: ch.sy(B), y2: ch.sy(B), class: "ln-thin s5 dash" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(131072) - 4, y: ch.sy(B) - 5, "text-anchor": "end", class: "lbl-sm", style: { fill: "var(--c5)" }, text: "your concurrency: " + B }));
          if (r.maxSeq >= 1) ch.g.appendChild(K.s("circle", { cx: ch.sx(ctx), cy: ch.sy(r.maxSeq), r: 6, class: "f2" }));
        } else {
          ch.g.appendChild(K.s("text", { x: 340, y: 120, "text-anchor": "middle", class: "lbl", text: "No memory left for KV cache on this configuration." }));
        }
        c.stage.appendChild(ch.svg);

        var f = m.mla ? K.tex("\\underbrace{" + m.L + "}_{\\text{layers}}\\times\\underbrace{(512+64)}_{\\text{MLA latent + RoPE key}}\\times\\underbrace{" + kb + "}_{\\text{bytes}} = " + K.fmtNum(r.kvTok, 0) + "\\text{ B}")
          : K.tex("\\underbrace{2}_{K,V}\\times\\underbrace{" + m.L + "}_{\\text{layers}}\\times\\underbrace{" + m.kvh + "}_{\\text{KV heads}}\\times\\underbrace{" + m.hd + "}_{d_{head}}\\times\\underbrace{" + kb + "}_{\\text{bytes}} = " + K.fmtNum(r.kvTok, 0) + "\\text{ B}");
        var note = m.mla ? "MLA caches one compressed 576-number latent per layer instead of per-head K and V — that's how a 671B model gets a tiny KV cache."
          : m.kvh < m.H ? "GQA: " + m.H + " query heads share " + m.kvh + " KV heads, so the cache is <b>" + (m.H / m.kvh) + "× smaller</b> than full multi-head attention would need."
          : "Full multi-head attention (every head keeps its own K,V) — modern models use GQA to shrink this.";
        c.readout.innerHTML = "<div>KV bytes per token = " + f + " (" + (r.kvTok / 1024).toFixed(r.kvTok < 10240 ? 1 : 0) + " KiB). " + note + "</div>" + stats([
          ["KV / token", K.fmtBytes(r.kvTok)], ["KV / request (" + fmtTok(ctx) + ")", K.fmtBytes(r.perSeq)], ["KV for " + B + " requests", K.fmtBytes(need), need > Math.max(0, r.freeKV) ? "var(--red)" : null],
          ["weights", K.fmtBytes(r.W)], ["free for KV", K.fmtBytes(Math.max(0, r.freeKV))], ["max concurrent @ " + fmtTok(ctx), K.fmtNum(r.maxSeq, 0)], ["max tokens in cache", K.fmtNum(r.maxTok, 0)],
        ]) + "<div class='muted' style='margin-top:6px'>Usable = " + Math.round(g.util * 100) + "% of memory, minus ~" + (g.mac ? 1 : 1.5) + " GB/GPU for activations & CUDA graphs. Real engines report this as “# GPU blocks” / “maximum concurrency” at startup.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- paged attention
  V.register("paged-attention", {
    title: "PagedAttention: KV cache in fixed-size blocks, like virtual memory",
    desc: "Press <b>Step</b>: every running sequence generates one token. When its last block fills up, it grabs any free block from the pool — no need for contiguous memory. Finished sequences return their blocks. Compare with reserving max_len contiguous slots per request.",
    render: function (c, p, K) {
      var SEQ = [
        { id: "A", prompt: 20, out: 18, sh: 1 }, { id: "B", prompt: 18, out: 10, sh: 1 }, { id: "C", prompt: 7, out: 26 }, { id: "D", prompt: 12, out: 6 },
        { id: "E", prompt: 17, out: 14, sh: 1 }, { id: "F", prompt: 9, out: 20 }, { id: "G", prompt: 5, out: 12 }, { id: "H", prompt: 16, out: 9, sh: 1 },
      ];
      var SLOTS = 160, PL = 16;
      var bs = [4, 8, 16].indexOf(+p.blockSize) >= 0 ? +p.blockSize : 4, share = p.prefixSharing !== false, maxLen = +p.maxLen || 48, st;
      var tk = ticker(c, function () { if (!step()) tk.stop(); }, 650, function () { playB.textContent = "▶ Play"; });
      c.controls.appendChild(K.button("Step ▸", function () { tk.stop(); step(); }));
      var playB = K.button("▶ Play", function () { if (tk.running()) tk.stop(); else { if (finished()) reset(); tk.start(); playB.textContent = "⏸ Pause"; } }, "primary");
      c.controls.appendChild(playB);
      c.controls.appendChild(K.button("Reset", function () { tk.stop(); reset(); }));
      c.controls.appendChild(K.select({ label: "block size (tokens)", options: ["4", "8", "16"], value: String(bs), onChange: function (v) { bs = +v; tk.stop(); reset(); } }));
      c.controls.appendChild(K.toggle({ label: "share system-prompt blocks (A, B, E, H)", value: share, onChange: function (v) { share = v; tk.stop(); reset(); } }));
      c.controls.appendChild(K.slider({ label: "contiguous max_len", min: 32, max: 160, step: 8, value: maxLen, onInput: function (v) { maxLen = v; draw(); } }));

      function reset() {
        var NB = SLOTS / bs, r = K.rng(42), free = [];
        for (var i = 0; i < NB; i++) free.push(i);
        for (var j = NB - 1; j > 0; j--) { var q = Math.floor(r() * (j + 1)), t = free[j]; free[j] = free[q]; free[q] = t; }
        st = { t: 0, NB: NB, free: free, ref: new Array(NB).fill(0), run: [], wait: SEQ.map(function (s, i) { return { id: s.id, ci: i, prompt: s.prompt, total: s.prompt + s.out, sh: s.sh && share, len: 0, blocks: [] }; }),
          done: [], prefix: null, log: [], preempt: 0 };
        admit(); draw();
      }
      function finished() { return !st.run.length && !st.wait.length; }
      function log(s) { st.log.unshift("t=" + st.t + ": " + s); st.log = st.log.slice(0, 4); }
      function alloc() { if (!st.free.length) return -1; var b = st.free.shift(); st.ref[b] = 1; return b; }
      function release(s) {
        s.blocks.forEach(function (b) { st.ref[b]--; if (st.ref[b] <= 0) { st.ref[b] = 0; st.free.push(b); } });
        s.blocks = [];
        if (st.prefix && st.ref[st.prefix[0]] === 0) st.prefix = null;
      }
      function admit() {
        while (st.wait.length) {
          var s = st.wait[0], len = s.len || s.prompt, nb = Math.ceil(len / bs), pb = Math.floor(PL / bs);
          var shareN = s.sh && st.prefix ? st.prefix.length : 0;
          if (nb - shareN > st.free.length) break;
          var blocks = shareN ? st.prefix.slice() : [];
          blocks.forEach(function (b) { st.ref[b]++; });
          while (blocks.length < nb) blocks.push(alloc());
          s.blocks = blocks; s.len = len;
          if (s.sh && !st.prefix && pb >= 1) st.prefix = blocks.slice(0, pb);
          st.run.push(s); st.wait.shift();
          log(s.id + " admitted: prefill " + len + " tokens → " + nb + (nb === 1 ? " block" : " blocks") + (shareN ? " (" + shareN + " shared with the system prompt)" : ""));
        }
      }
      function step() {
        if (finished()) { draw(); return false; }
        st.t++;
        st.run = st.run.filter(function (s) {
          if (s.len >= s.total) { var n = s.blocks.length; release(s); st.done.push(s); log(s.id + " finished → " + n + (n === 1 ? " block" : " blocks") + " back to the free pool"); return false; }
          return true;
        });
        admit();
        for (var i = 0; i < st.run.length; i++) {
          var s = st.run[i];
          if (s.len >= s.total) continue;
          if (s.len === s.blocks.length * bs) {
            var b = alloc();
            if (b < 0) {
              var v = st.run[st.run.length - 1];
              release(v); v.pre = true; st.run.pop(); st.wait.unshift(v); st.preempt++;
              log("pool empty! preempted " + v.id + " (blocks freed; it will be recomputed later)");
              if (v === s) continue;
              b = alloc(); if (b < 0) continue;
            }
            s.blocks.push(b);
          }
          s.len++;
        }
        draw();
        return !finished();
      }
      function blockMap() {
        var own = {};
        st.run.forEach(function (s) {
          s.blocks.forEach(function (b, j) {
            var fill = Math.min(bs, s.len - j * bs);
            if (!own[b]) own[b] = { ci: s.ci, id: s.id, fill: fill, ids: [s.id] };
            else { own[b].ids.push(s.id); own[b].fill = Math.max(own[b].fill, fill); }
          });
        });
        return own;
      }
      function draw() {
        var W = 680, H = 410, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" }), own = blockMap();
        // left: logical view
        svg.appendChild(K.s("text", { x: 10, y: 14, class: "lbl", text: "Running sequences (logical tokens) → block table" }));
        var maxTot = 38, nbMax = Math.ceil(maxTot / bs), tw = Math.min(6, (290 - nbMax * 3) / maxTot);
        st.run.slice(0, 6).forEach(function (s, r) {
          var y0 = 28 + r * 44;
          svg.appendChild(K.s("circle", { cx: 18, cy: y0 + 7, r: 9, class: fcls(s.ci) }));
          svg.appendChild(K.s("text", { x: 18, y: y0 + 11, "text-anchor": "middle", class: "t-white", style: { fontSize: "11px", fontWeight: 700 }, text: s.id }));
          s.blocks.forEach(function (b, j) {
            var bx = 34 + j * (bs * tw + 3), shared = st.ref[b] > 1;
            svg.appendChild(K.s("rect", { x: bx - 1, y: y0 - 1, width: bs * tw + 1, height: 16, rx: 2, class: "box", "stroke-dasharray": shared ? "2 2" : null }));
            for (var k = 0; k < bs; k++) {
              var ti = j * bs + k;
              if (ti < s.len) svg.appendChild(K.s("rect", { x: bx + k * tw, y: y0 + 1, width: Math.max(1, tw - 1), height: 12, class: fcls(s.ci), opacity: ti < s.prompt ? 0.9 : 0.5 }));
            }
          });
          var tbl = s.blocks.map(function (b) { return st.ref[b] > 1 ? b + "*" : b; }).join(", ");
          svg.appendChild(K.s("text", { x: 34, y: y0 + 29, class: "lbl-sm mono", text: "len " + s.len + "/" + s.total + "   table [" + tbl + "]" }));
        });
        if (st.run.length > 6) svg.appendChild(K.s("text", { x: 34, y: 296, class: "lbl-sm", text: "+" + (st.run.length - 6) + " more running" }));
        if (!st.run.length) svg.appendChild(K.s("text", { x: 34, y: 60, class: "lbl-sm", text: finished() ? "All sequences finished." : "nothing running" }));
        // waiting
        svg.appendChild(K.s("text", { x: 10, y: 308, class: "lbl-sm", text: "waiting:" }));
        st.wait.forEach(function (s, i) {
          svg.appendChild(K.s("circle", { cx: 66 + i * 22, cy: 304, r: 8, class: fcls(s.ci), opacity: 0.6 }));
          svg.appendChild(K.s("text", { x: 66 + i * 22, y: 308, "text-anchor": "middle", class: "t-white", style: { fontSize: "10px", fontWeight: 700 }, text: s.id + (s.pre ? "↺" : "") }));
        });
        if (!st.wait.length) svg.appendChild(K.s("text", { x: 60, y: 308, class: "lbl-sm", text: "—" }));
        // right: physical pool
        var PX = 350, PW = 322, cols = bs === 4 ? 8 : 5, rows = st.NB / cols, gap = 6;
        var cw = (PW - (cols - 1) * gap) / cols, chh = bs === 16 ? 58 : bs === 8 ? 46 : 34;
        svg.appendChild(K.s("text", { x: PX, y: 14, class: "lbl", text: "Physical KV blocks in GPU memory (" + st.NB + " × " + bs + " slots)" }));
        for (var b = 0; b < st.NB; b++) {
          var cx = PX + (b % cols) * (cw + gap), cy = 26 + Math.floor(b / cols) * (chh + gap), o = own[b];
          var rc = K.s("rect", { x: cx, y: cy, width: cw, height: chh, rx: 5, class: "box", "stroke-dasharray": o && o.ids.length > 1 ? "3 2" : null });
          if (o) { rc.setAttribute("stroke", col(o.ci)); rc.setAttribute("stroke-width", 1.6); }
          rc.appendChild(K.s("title", { text: "block " + b + (o ? " · owner " + o.ids.join(" & ") + " · " + o.fill + "/" + bs + " slots used" : " · free") }));
          svg.appendChild(rc);
          svg.appendChild(K.s("text", { x: cx + 4, y: cy + 11, class: "lbl-sm mono", text: "#" + b }));
          if (o) {
            svg.appendChild(K.s("text", { x: cx + cw - 4, y: cy + 11, "text-anchor": "end", class: "lbl-sm", style: { fill: col(o.ci), fontWeight: 700 }, text: o.ids.length > 1 ? "×" + o.ids.length : o.id }));
            var per = bs === 16 ? 8 : bs, sw = (cw - 8) / per, sh = bs === 16 ? (chh - 22) / 2 : chh - 20;
            for (var k2 = 0; k2 < bs; k2++) {
              var rr = Math.floor(k2 / per), cc = k2 % per;
              svg.appendChild(K.s("rect", { x: cx + 4 + cc * sw, y: cy + 15 + rr * (sh + 2), width: Math.max(1, sw - 1.2), height: sh, rx: 1, class: k2 < o.fill ? fcls(o.ci) : "fline", opacity: k2 < o.fill ? 0.8 : 1 }));
            }
          }
        }
        // bottom: contiguous vs paged strips
        var SX = 160, SW = 340, slot = SW / SLOTS, y1 = 334, y2 = 374;
        svg.appendChild(K.s("line", { x1: 10, x2: 670, y1: 318, y2: 318, class: "grid" }));
        svg.appendChild(K.s("text", { x: 10, y: y1 + 7, class: "lbl", text: "Contiguous" }));
        svg.appendChild(K.s("text", { x: 10, y: y1 + 19, class: "lbl-sm", text: "reserve max_len=" + maxLen + " each" }));
        svg.appendChild(K.s("rect", { x: SX, y: y1, width: SW, height: 16, class: "fline" }));
        var fitN = Math.floor(SLOTS / maxLen), placed = st.run.slice(0, fitN), used = 0;
        placed.forEach(function (s, i) {
          var x0 = SX + i * maxLen * slot;
          svg.appendChild(K.s("rect", { x: x0, y: y1, width: maxLen * slot - 1, height: 16, class: fcls(s.ci), opacity: 0.18 }));
          svg.appendChild(K.s("rect", { x: x0, y: y1, width: Math.min(s.len, maxLen) * slot, height: 16, class: fcls(s.ci), opacity: 0.85 }));
          used += Math.min(s.len, maxLen);
        });
        var resv = placed.length * maxLen, cWaste = resv ? 1 - used / resv : 0, cant = Math.max(0, st.run.length - fitN);
        svg.appendChild(K.s("text", { x: SX + SW + 8, y: y1 + 8, class: "lbl-sm", text: Math.round(cWaste * 100) + "% of reserved slots wasted" }));
        svg.appendChild(K.s("text", { x: SX + SW + 8, y: y1 + 20, class: "lbl-sm", style: cant ? { fill: "var(--red)" } : null, text: "fits " + fitN + " at once" + (cant ? " → " + cant + " must wait" : "") }));
        svg.appendChild(K.s("text", { x: 10, y: y2 + 7, class: "lbl", text: "Paged" }));
        svg.appendChild(K.s("text", { x: 10, y: y2 + 19, class: "lbl-sm", text: "block size " + bs + ", any free block" }));
        var alloc_ = 0, usedP = 0;
        for (var b2 = 0; b2 < st.NB; b2++) {
          var o2 = own[b2], x2 = SX + b2 * bs * slot;
          svg.appendChild(K.s("rect", { x: x2, y: y2, width: bs * slot - 0.6, height: 16, class: o2 ? fcls(o2.ci) : "fline", opacity: o2 ? 0.18 : 1 }));
          if (o2) { svg.appendChild(K.s("rect", { x: x2, y: y2, width: o2.fill * slot, height: 16, class: fcls(o2.ci), opacity: 0.85 })); alloc_ += bs; usedP += o2.fill; }
        }
        var pWaste = alloc_ ? 1 - usedP / alloc_ : 0;
        var logical = st.run.reduce(function (a, s) { return a + s.blocks.length; }, 0), physical = Object.keys(own).length;
        svg.appendChild(K.s("text", { x: SX + SW + 8, y: y2 + 8, class: "lbl-sm", text: Math.round(pWaste * 100) + "% of allocated slots wasted" }));
        svg.appendChild(K.s("text", { x: SX + SW + 8, y: y2 + 20, class: "lbl-sm", text: "all " + st.run.length + " fit" + (logical > physical ? " · " + (logical - physical) + (logical - physical === 1 ? " block" : " blocks") + " deduped" : "") }));
        K.clear(c.stage).appendChild(svg);

        c.readout.innerHTML = stats([
          ["step", st.t], ["running / waiting / done", st.run.length + " / " + st.wait.length + " / " + st.done.length], ["free blocks", st.free.length + " of " + st.NB],
          ["paged waste", Math.round(pWaste * 100) + "%", "var(--c3)"], ["contiguous waste", Math.round(cWaste * 100) + "%", "var(--c2)"], ["preemptions", st.preempt],
        ]) + "<div style='margin-top:6px' class='mono'>" + (st.log.length ? st.log.join("<br>") : "") + "</div>" +
          "<div class='muted' style='margin-top:6px'>Only the <i>last</i> block of each sequence can be partly empty, so waste stays below one block per sequence (the vLLM paper measured under 4%, vs 60–80% for contiguous preallocation). " +
          "Bigger blocks → fewer table entries but more waste; shared blocks (dashed, “*”) are reference-counted and copied only if written (copy-on-write).</div>";
      }
      reset();
    },
  });

  // ---------------------------------------------------------------- prefix cache (radix tree)
  var PC_SYS = "You are a helpful support assistant for Acme. Answer briefly and link the docs.";
  var PC_SQL = "You are a SQL expert. Reply with one SQL query only.";
  var PC_FS = "Classify the sentiment. Review: Great battery life -> positive. Review: Broke after a day -> negative. Review: Does the job -> neutral. Review:";
  var PC_WL = {
    chat: { label: "Chat app: shared system prompt + multi-turn", reqs: [
      PC_SYS + " User: How do I reset my password?", PC_SYS + " User: How do I change my email?",
      PC_SYS + " User: How do I reset my password? Assistant: Open Settings then Security. User: The link expired.",
      PC_SQL + " User: count users by country", PC_SYS + " User: How do I delete my account?", PC_SQL + " User: top 5 products by revenue",
      PC_SYS + " User: How do I change my email? Assistant: Settings then Profile. User: And my phone number?", PC_SYS + " User: How do I reset my password?"] },
    fewshot: { label: "Few-shot classification (long shared examples)", reqs: [
      PC_FS + " Screen is gorgeous ->", PC_FS + " Shipping took forever ->", PC_FS + " Love it, would buy again ->",
      PC_FS + " Meh, it is fine ->", PC_FS + " Stopped charging in a week ->", PC_FS + " Five stars ->"] },
    unique: { label: "Unrelated prompts (nothing to share)", reqs: [
      "Translate to French: the cat sleeps on the sofa", "Write a haiku about GPUs and heat", "Summarize: Rust ownership rules prevent data races",
      "What is 17 times 23?", "List three uses of Redis in production", "Explain TCP slow start in one line"] },
  };

  V.register("prefix-cache", {
    title: "Prefix caching: a radix tree of prompts that share beginnings",
    desc: "Send requests one by one. Tokens already in the tree (same prefix as an earlier request) are <b>cache hits</b>: their KV is reused and prefill skips them. Only the new suffix is computed. (Here 1 word ≈ 1 token.) Type your own prompt too.",
    render: function (c, p, K) {
      var wl = PC_WL[p.workload] ? p.workload : "chat", cap = +p.capacity || 300, root, nid, clock, size, sent, hist, totTok, totHit, evicted, lastReq;
      c.controls.appendChild(K.select({ label: "workload", options: Object.keys(PC_WL).map(function (k) { return { value: k, label: PC_WL[k].label }; }), value: wl, onChange: function (v) { wl = v; reset(); } }));
      var nextB = K.button("Send next request ▸", function () { sendNext(); }, "primary");
      c.controls.appendChild(nextB);
      c.controls.appendChild(K.button("Send all", function () { while (sent < PC_WL[wl].reqs.length) sendNext(true); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { reset(); }));
      c.controls.appendChild(K.slider({ label: "cache capacity (tokens)", min: 20, max: 300, step: 10, value: cap, onInput: function (v) { cap = v; evict(); draw(); } }));
      var inp = K.h("input", { type: "text", placeholder: "type a prompt, e.g. " + PC_SYS.slice(0, 22) + "…", style: { padding: "4px 8px", borderRadius: "7px", border: "1px solid var(--line2)", background: "var(--bg2)", color: "var(--fg)", fontSize: "13px", width: "260px" } });
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") sendCustom(); });
      c.controls.appendChild(K.h("span", { class: "ctl" }, inp, K.button("Send", sendCustom)));

      function node(edge, parent) { return { edge: edge, kids: [], parent: parent, last: clock, id: ++nid, mark: "" }; }
      function reset() { nid = 0; clock = 0; root = node([], null); size = 0; sent = 0; hist = []; totTok = 0; totHit = 0; evicted = 0; lastReq = null; nextB.disabled = false; draw(); }
      function all(n, f) { f(n); n.kids.forEach(function (k) { all(k, f); }); }
      function insert(tokens) {
        clock++;
        all(root, function (n) { n.mark = ""; });
        var n = root, i = 0, hit = 0; root.last = clock;
        while (i < tokens.length) {
          var ch = null;
          for (var k = 0; k < n.kids.length; k++) if (n.kids[k].edge[0] === tokens[i]) ch = n.kids[k];
          if (!ch) { var leaf = node(tokens.slice(i), n); leaf.mark = "new"; n.kids.push(leaf); size += leaf.edge.length; break; }
          var l = 0;
          while (l < ch.edge.length && i + l < tokens.length && ch.edge[l] === tokens[i + l]) l++;
          if (l < ch.edge.length) {
            var mid = node(ch.edge.slice(0, l), n); mid.kids = [ch];
            ch.edge = ch.edge.slice(l); ch.parent = mid; n.kids[n.kids.indexOf(ch)] = mid; ch = mid;
          }
          ch.mark = "hit"; ch.last = clock; hit += l; i += l; n = ch;
        }
        return hit;
      }
      function evict() {
        while (size > cap) {
          var best = null;
          all(root, function (n) { if (n !== root && !n.kids.length && n.last < clock && (!best || n.last < best.last)) best = n; });
          if (!best) break;
          best.parent.kids.splice(best.parent.kids.indexOf(best), 1); size -= best.edge.length; evicted += best.edge.length;
        }
      }
      function send(text) {
        var toks = text.split(/\s+/).filter(Boolean);
        if (!toks.length) return;
        var hit = insert(toks); evict();
        totTok += toks.length; totHit += hit;
        lastReq = { n: toks.length, hit: hit, text: text };
        hist.push(Math.round(hit / toks.length * 100));
      }
      function sendNext(quiet) {
        var R = PC_WL[wl].reqs;
        if (sent >= R.length) return;
        send(R[sent++]);
        nextB.disabled = sent >= R.length;
        if (!quiet) draw();
      }
      function sendCustom() { if (inp.value.trim()) { send(inp.value.trim()); inp.value = ""; draw(); } }

      function draw() {
        var leaves = 0, maxD = 0;
        (function lay(n, d) {
          n.depth = d; maxD = Math.max(maxD, d);
          if (!n.kids.length) n.y = leaves++;
          else { n.kids.forEach(function (k) { lay(k, d + 1); }); n.y = (n.kids[0].y + n.kids[n.kids.length - 1].y) / 2; }
        })(root, 0);
        var rowH = Math.min(44, 330 / Math.max(1, leaves)), H = Math.max(root.kids.length ? 120 : 64, 20 + leaves * rowH + 10), W = 680;
        var colW = Math.min(150, (W - 70) / Math.max(1, maxD)), bw = colW - 16, bh = Math.min(30, rowH - 5);
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        function pos(n) { return n === root ? [8, 20 + n.y * rowH + rowH / 2] : [60 + (n.depth - 1) * colW, 20 + n.y * rowH + rowH / 2]; }
        svg.appendChild(K.s("text", { x: 8, y: 12, class: "lbl-sm", text: "green = reused from cache this request · orange = newly computed (prefill) · grey = cached from earlier requests" }));
        all(root, function (n) {
          if (n === root) return;
          var a = pos(n.parent), b = pos(n), ax = n.parent === root ? a[0] + 40 : a[0] + bw;
          var mx = (ax + b[0]) / 2;
          svg.appendChild(K.s("path", { d: "M" + ax + " " + a[1] + "C" + mx + " " + a[1] + " " + mx + " " + b[1] + " " + b[0] + " " + b[1], class: "ln-thin " + (n.mark ? (n.mark === "hit" ? "s3" : "s2") : "smuted") }));
        });
        var rp = pos(root);
        svg.appendChild(K.s("rect", { x: rp[0], y: rp[1] - 12, width: 40, height: 24, rx: 6, class: "box" }));
        svg.appendChild(K.s("text", { x: rp[0] + 20, y: rp[1] + 4, "text-anchor": "middle", class: "lbl-sm", text: "root" }));
        all(root, function (n) {
          if (n === root) return;
          var q = pos(n), cls = n.mark === "hit" ? "f3 s3" : n.mark === "new" ? "f2 s2" : "box";
          var r = K.s("rect", { x: q[0], y: q[1] - bh / 2, width: bw, height: bh, rx: 6, class: cls });
          if (n.mark) { r.setAttribute("fill-opacity", 0.16); r.setAttribute("stroke-width", 1.5); }
          r.appendChild(K.s("title", { text: n.edge.join(" ") + "  (" + n.edge.length + " tokens)" }));
          svg.appendChild(r);
          var chars = Math.max(4, Math.floor((bw - 10) / 5.9)), txt = n.edge.join(" ");
          if (txt.length > chars) txt = txt.slice(0, chars - 1) + "…";
          var two = bh >= 26;
          svg.appendChild(K.s("text", { x: q[0] + 5, y: q[1] + (two ? -2 : 4), class: "lbl-sm", style: { fill: "var(--fg)" }, text: txt }));
          if (two) svg.appendChild(K.s("text", { x: q[0] + 5, y: q[1] + 10, class: "lbl-sm mono", text: n.edge.length + " tok" }));
        });
        K.clear(c.stage).appendChild(svg);
        var rate = totTok ? totHit / totTok : 0;
        c.readout.innerHTML = (lastReq ? "<div>Last request: <b>" + lastReq.n + "</b> tokens, <b style='color:var(--c3)'>" + lastReq.hit + " cached</b>, <b style='color:var(--c2)'>" + (lastReq.n - lastReq.hit) + " prefilled</b> (" + Math.round(lastReq.hit / lastReq.n * 100) + "% hit) — its prefill (and TTFT) shrinks by about that fraction.</div>" : "<div>Press <b>Send next request</b>.</div>") +
          stats([["requests", hist.length], ["prompt tokens", totTok], ["served from cache", totHit, "var(--c3)"], ["hit rate", Math.round(rate * 100) + "%"], ["cache size", size + " / " + cap + " tok"], ["evicted (LRU)", evicted]]) +
          "<div style='margin-top:6px'>" + hist.map(function (h, i) { return "<span class='tok" + (h >= 50 ? " new" : "") + "'>R" + (i + 1) + " " + h + "%</span>"; }).join("") + "</div>" +
          "<div class='muted' style='margin-top:4px'>A radix tree stores each shared prefix once; nodes split where prompts diverge. SGLang's RadixAttention works like this (evicting least-recently-used leaves when full); vLLM hashes fixed-size KV blocks instead. Keep system prompts identical and put variable content <i>last</i> to get hits.</div>";
      }
      reset();
    },
  });

  // ---------------------------------------------------------------- static vs continuous batching
  V.register("batching-sim", {
    title: "Static vs continuous batching (same requests, same GPU)",
    desc: "Each bar is a request occupying one batch slot (dark head = prefill step, lighter = decode steps). <b>Static</b> batching waits for the longest request before starting a new batch — hatched = idle slots. <b>Continuous</b> batching refills a slot the moment it frees up.",
    render: function (c, p, K) {
      var seed = +p.seed || 3, n = +p.requests || 16, B = +p.slots || 4, rate = p.rate !== undefined ? +p.rate : 3, spread = 0.8, cur = null, raf = null;
      var seedV = K.h("span", { class: "ctl-val", text: "seed " + seed });
      c.controls.appendChild(K.button("🎲 New workload", function () { seed++; seedV.textContent = "seed " + seed; cur = null; draw(); }));
      c.controls.appendChild(seedV);
      c.controls.appendChild(K.slider({ label: "requests", min: 6, max: 40, value: n, onInput: function (v) { n = v; cur = null; draw(); } }));
      c.controls.appendChild(K.slider({ label: "batch slots", min: 2, max: 8, value: B, onInput: function (v) { B = v; cur = null; draw(); } }));
      c.controls.appendChild(K.slider({ label: "arrivals / 10 steps", min: 0, max: 10, step: 0.5, value: rate, fmt: function (v) { return v === 0 ? "all at t=0" : v.toFixed(1); }, onInput: function (v) { rate = v; cur = null; draw(); } }));
      c.controls.appendChild(K.slider({ label: "output-length spread", min: 0.1, max: 1.3, step: 0.1, value: spread, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { spread = v; cur = null; draw(); } }));
      var anim = K.button("▶ Animate", function () { if (raf) stopA(); else startA(); }, "primary");
      c.controls.appendChild(anim);

      function gen() {
        var r = K.rng(seed * 7919), t = 0, out = [];
        for (var i = 0; i < n; i++) {
          if (rate > 0 && i > 0) t += -Math.log(1 - r()) * (10 / rate);
          var len = Math.round(K.clamp(Math.exp(Math.log(16) + spread * K.randn(r)), 2, 120));
          out.push({ id: i, arr: Math.round(t), out: len });
        }
        return out;
      }
      function simStatic(q) {
        var t = 0, i = 0, rows = [], batches = [];
        while (i < q.length) {
          if (q[i].arr > t) t = q[i].arr;
          var batch = [];
          while (i < q.length && q[i].arr <= t && batch.length < B) batch.push(q[i++]);
          var dur = Math.max.apply(null, batch.map(function (r) { return r.out; }));
          batch.forEach(function (r, s) { rows.push({ r: r, slot: s, start: t, end: t + r.out }); });
          batches.push({ start: t, end: t + dur, rows: rows.slice(rows.length - batch.length) });
          t += dur;
        }
        return { rows: rows, batches: batches, span: t };
      }
      function simCont(q) {
        var t = 0, i = 0, rows = [], slots = new Array(B).fill(null), guard = 0;
        while ((i < q.length || slots.some(function (s) { return s; })) && guard++ < 100000) {
          for (var s = 0; s < B; s++) if (slots[s] && slots[s].end <= t) slots[s] = null;
          for (var s2 = 0; s2 < B; s2++) if (!slots[s2] && i < q.length && q[i].arr <= t) { var r = q[i++], row = { r: r, slot: s2, start: t, end: t + r.out }; rows.push(row); slots[s2] = row; }
          if (!slots.some(function (x) { return x; }) && i < q.length) { t = Math.max(t, q[i].arr); continue; }
          t++;
        }
        var span = rows.reduce(function (a, r) { return Math.max(a, r.end); }, 0);
        return { rows: rows, span: span };
      }
      function metrics(sim) {
        var tok = sim.rows.reduce(function (a, r) { return a + r.r.out; }, 0);
        var lat = sim.rows.map(function (r) { return r.end - r.r.arr; }), ttft = sim.rows.map(function (r) { return r.start + 1 - r.r.arr; });
        return { span: sim.span, util: tok / (B * sim.span), thr: tok / sim.span, lat: mean(lat), p90: pctl(lat, 0.9), ttft: mean(ttft) };
      }
      function startA() { cur = 0; anim.textContent = "⏸ Stop"; var t0 = null;
        function fr(ts) {
          if (!c.root.isConnected) { raf = null; return; }
          if (t0 === null) t0 = ts;
          cur = (ts - t0) / 7000 * spanMax;
          if (cur >= spanMax) { cur = null; stopA(); draw(); return; }
          draw(); raf = requestAnimationFrame(fr);
        }
        raf = requestAnimationFrame(fr);
      }
      function stopA() { if (raf) cancelAnimationFrame(raf); raf = null; anim.textContent = "▶ Animate"; }
      var spanMax = 1;
      function draw() {
        var q = gen(), S = simStatic(q), C = simCont(q), ms = metrics(S), mc = metrics(C);
        spanMax = Math.max(S.span, C.span);
        var laneH = K.clamp(Math.floor(118 / B), 12, 22), panelH = 34 + B * laneH, W = 680, H = 2 * panelH + 54;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" }), pat = hatch(K, svg);
        var X0 = 52, X1 = 668, sx = K.scale(0, spanMax, X0, X1), tc = cur === null ? Infinity : cur;
        function panel(y0, title, sim, m, isStatic) {
          svg.appendChild(K.s("text", { x: 8, y: y0 + 10, class: "lbl", text: title }));
          svg.appendChild(K.s("text", { x: X1, y: y0 + 10, "text-anchor": "end", class: "lbl-sm", text: "done at step " + m.span + " · slot utilization " + Math.round(m.util * 100) + "% · " + m.thr.toFixed(2) + " tok/step" }));
          q.forEach(function (r) { if (r.arr <= tc) svg.appendChild(K.s("path", { d: "M" + sx(r.arr) + " " + (y0 + 22) + "l-3 -6h6z", class: fcls(r.id) })); });
          var ly = y0 + 26;
          for (var s = 0; s < B; s++) {
            svg.appendChild(K.s("rect", { x: X0, y: ly + s * laneH, width: X1 - X0, height: laneH - 2, class: "fbg3" }));
            svg.appendChild(K.s("text", { x: X0 - 6, y: ly + s * laneH + laneH / 2 + 3, "text-anchor": "end", class: "lbl-sm", text: "slot " + (s + 1) }));
          }
          if (isStatic) sim.batches.forEach(function (bt) {
            for (var s = 0; s < B; s++) {
              var rw = bt.rows.filter(function (r) { return r.slot === s; })[0], a = rw ? rw.end : bt.start, e = Math.min(bt.end, tc);
              if (e > a) svg.appendChild(K.s("rect", { x: sx(a), y: ly + s * laneH, width: sx(e) - sx(a), height: laneH - 2, fill: pat, opacity: 0.9 }));
            }
            svg.appendChild(K.s("line", { x1: sx(bt.start), x2: sx(bt.start), y1: ly - 2, y2: ly + B * laneH, class: "ln-thin smuted dash" }));
          });
          sim.rows.forEach(function (rw) {
            if (rw.start >= tc) return;
            var e = Math.min(rw.end, tc), y = ly + rw.slot * laneH, g = K.s("g");
            g.appendChild(K.s("rect", { x: sx(rw.start), y: y, width: Math.max(1, sx(e) - sx(rw.start) - 0.8), height: laneH - 2, rx: 2, class: fcls(rw.r.id), opacity: 0.55 }));
            g.appendChild(K.s("rect", { x: sx(rw.start), y: y, width: Math.max(1, sx(Math.min(e, rw.start + 1)) - sx(rw.start)), height: laneH - 2, class: fcls(rw.r.id) }));
            if (sx(e) - sx(rw.start) > 16 && laneH >= 14) g.appendChild(K.s("text", { x: sx(rw.start) + 4, y: y + laneH / 2 + 3, class: "t-white", style: { fontSize: "9.5px", fontWeight: 700 }, text: "r" + rw.r.id }));
            g.appendChild(K.s("title", { text: "request r" + rw.r.id + ": arrived t=" + rw.r.arr + ", started " + rw.start + ", finished " + rw.end + " (" + rw.r.out + " tokens, waited " + (rw.start - rw.r.arr) + ")" }));
            svg.appendChild(g);
          });
        }
        panel(4, "Static batching", S, ms, true);
        panel(4 + panelH + 8, "Continuous batching", C, mc, false);
        var ay = H - 30;
        svg.appendChild(K.s("line", { x1: X0, x2: X1, y1: ay, y2: ay, class: "axis" }));
        K.niceTicks(0, spanMax, 8).forEach(function (v) { svg.appendChild(K.s("text", { x: sx(v), y: ay + 13, "text-anchor": "middle", class: "tick", text: v })); });
        svg.appendChild(K.s("text", { x: X0, y: H - 1, class: "lbl-sm", text: "time (decode steps; ▼ = request arrives)" }));
        if (cur !== null) svg.appendChild(K.s("line", { x1: sx(cur), x2: sx(cur), y1: 14, y2: ay, class: "ln-thin s5" }));
        K.clear(c.stage).appendChild(svg);

        function rowH(label, a, b, better, fmt) {
          var wa = better === "lo" ? a <= b : a >= b;
          return "<tr><td>" + label + "</td><td style='text-align:right'>" + (wa ? "<b>" + fmt(a) + "</b>" : fmt(a)) + "</td><td style='text-align:right'>" + (!wa ? "<b>" + fmt(b) + "</b>" : fmt(b)) + "</td></tr>";
        }
        var f0 = function (v) { return v.toFixed(0); }, f1 = function (v) { return v.toFixed(1); }, fp = function (v) { return Math.round(v * 100) + "%"; };
        c.readout.innerHTML = "<table style='border-collapse:collapse;font-size:13px;min-width:380px'><tr class='muted'><td></td><td style='text-align:right;padding-left:18px'>static</td><td style='text-align:right;padding-left:18px'>continuous</td></tr>" +
          rowH("time to finish all (steps)", ms.span, mc.span, "lo", f0) + rowH("slot utilization", ms.util, mc.util, "hi", fp) + rowH("throughput (tokens / step)", ms.thr, mc.thr, "hi", function (v) { return v.toFixed(2); }) +
          rowH("mean time to first token", ms.ttft, mc.ttft, "lo", f1) + rowH("mean latency", ms.lat, mc.lat, "lo", f1) + rowH("p90 latency", ms.p90, mc.p90, "lo", f0) + "</table>" +
          "<div style='margin-top:6px'>Continuous batching is <b>" + (ms.span / mc.span).toFixed(2) + "×</b> faster to drain this workload. Because one decode step costs about the same whether the batch has 1 or " + B + " requests (it's memory-bound), idle slots are pure lost throughput. Crank up the length spread: the longer the tail, the worse static batching gets. <span class='muted'>This is iteration-level scheduling from the Orca paper; every modern engine (vLLM, SGLang, TensorRT-LLM) does it.</span></div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- chunked prefill
  V.register("chunked-prefill", {
    title: "Chunked prefill: stop a long prompt from freezing everyone else's stream",
    desc: "Several users are streaming tokens (blue steps). A long prompt arrives (orange). Without chunking the engine runs the whole prefill as one giant step and every stream stalls. With chunked prefill each step has a token budget: decodes first, then a slice of the prompt.",
    render: function (c, p, K) {
      var PR = [1024, 2048, 4096, 8192, 16384, 32768], BU = [256, 512, 1024, 2048, 4096, 8192];
      var P = nearest(PR, +p.prompt || 8192), bud = nearest(BU, +p.budget || 512), D = +p.decodes || 16, slo = +p.slo || 50;
      c.controls.appendChild(idxSlider(K, { label: "long prompt", values: PR, value: P, fmt: fmtTok, onInput: function (v) { P = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "token budget / step", values: BU, value: bud, onInput: function (v) { bud = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "streaming users", min: 1, max: 64, value: D, onInput: function (v) { D = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "ITL SLO", min: 10, max: 200, step: 5, value: slo, fmt: function (v) { return v + " ms"; }, onInput: function (v) { slo = v; draw(); } }));
      var m = MODELS.llama8, g = GPUS.H100, bw = g.bw * 1e9 * BW_EFF, fl = g.tf * 1e12 * FLOP_EFF, W = m.P * 2, kvT = kvElems(m) * 2, CTXD = 2000, NT = 24, ARR = 4;
      function stepMs(dec, pre, pre0) {
        var bytes = W + dec * kvT * CTXD + (pre ? kvT * (pre0 + pre) : 0);
        var flops = 2 * m.P * (dec + pre) + 4 * m.L * m.d * dec * CTXD + 4 * m.L * m.d * pre * (pre0 + pre / 2);
        return (Math.max(bytes / bw, flops / fl) + STEP_OVH) * 1e3;
      }
      function sim(chunked) {
        var t = 0, steps = [], itl = [], lastEmit = 0, rem = P, done = 0, arrT = null, ttft = null, emitted = 0, guard = 0, starve = false;
        while ((emitted < NT || (ttft === null && !starve)) && guard++ < 2000) {
          var i = steps.length, pre = 0, dec = D + (ttft !== null ? 1 : 0), emits = true;
          if (i === ARR) arrT = t;
          if (arrT !== null && rem > 0) {
            if (chunked) { pre = Math.min(rem, Math.max(0, bud - dec)); if (!pre) starve = true; }
            else { pre = rem; dec = 0; emits = false; }
          }
          var dur = stepMs(dec, pre, P - rem);
          steps.push({ t0: t, dur: dur, dec: dec, pre: pre });
          t += dur; rem -= pre;
          if (pre && rem === 0) ttft = t - arrT;
          if (emits) { itl.push(t - lastEmit); lastEmit = t; emitted++; }
        }
        return { steps: steps, itl: itl.slice(1), total: t, arrT: arrT, ttft: ttft, starve: starve };
      }
      function draw() {
        var A = sim(false), Bc = sim(true), tmax = Math.max(A.total, Bc.total), X0 = 16, X1 = 664, sx = K.scale(0, tmax, X0, X1);
        var svg = K.s("svg", { viewBox: "0 0 680 196", class: "viz-svg" });
        function lane(y, s, title) {
          svg.appendChild(K.s("text", { x: X0, y: y - 14, class: "lbl", text: title }));
          s.steps.forEach(function (st) {
            var x = sx(st.t0), w = Math.max(0.6, sx(st.t0 + st.dur) - x - 0.6), gg = K.s("g");
            if (st.pre && st.dec) { gg.appendChild(K.s("rect", { x: x, y: y, width: w, height: 40, class: "f2", opacity: 0.9 })); gg.appendChild(K.s("rect", { x: x, y: y + 30, width: w, height: 10, class: "f1" })); }
            else gg.appendChild(K.s("rect", { x: x, y: y, width: w, height: 40, class: st.pre ? "f2" : "f1", opacity: st.pre ? 0.9 : 0.7 }));
            if (st.pre && w > 60) gg.appendChild(K.s("text", { x: x + w / 2, y: y + 20, "text-anchor": "middle", class: "t-white", style: { fontSize: "10.5px" }, text: "prefill " + fmtTok(st.pre) + " tok" }));
            gg.appendChild(K.s("title", { text: "step: " + fmtMs(st.dur) + " · " + st.dec + " decode tokens" + (st.pre ? " + " + st.pre + " prefill tokens" : "") }));
            svg.appendChild(gg);
          });
          var ax = sx(s.arrT);
          svg.appendChild(K.s("path", { d: "M" + ax + " " + (y - 2) + "l-5 -8h10z", class: "f5" }));
          if (s.ttft !== null) {
            var tx = sx(s.arrT + s.ttft);
            svg.appendChild(K.s("line", { x1: tx, x2: tx, y1: y - 4, y2: y + 46, class: "ln s5" }));
            svg.appendChild(K.s("text", { x: Math.min(tx + 4, 600), y: y + 54, class: "lbl-sm", style: { fill: "var(--c5)" }, text: "long request's first token: TTFT " + fmtMs(s.ttft) }));
          }
        }
        lane(30, A, "Without chunked prefill (prefill runs as its own step; streams pause)");
        lane(130, Bc, "Chunked prefill, budget " + bud + " tokens/step (decodes + a slice of the prompt)");
        K.clear(c.stage).appendChild(svg);
        var ymax = Math.max(slo * 1.3, Math.max.apply(null, A.itl.concat(Bc.itl)) * 1.1);
        var NX = Math.max(A.itl.length, Bc.itl.length), ch = K.chart({ w: 680, h: 210, x: [1, NX], y: [0, ymax], xLabel: "token # of each streaming user", yLabel: "inter-token latency (ms)" });
        [[A.itl, "s2", "no chunking"], [Bc.itl, "s3", "chunked"]].forEach(function (q) {
          var pts = q[0].map(function (v, i) { return [i + 1, v]; });
          ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln " + q[1] }));
          pts.forEach(function (pt) { ch.g.appendChild(K.s("circle", { cx: ch.sx(pt[0]), cy: ch.sy(pt[1]), r: 2.6, class: q[1].replace("s", "f") })); });
        });
        ch.g.appendChild(K.s("line", { x1: ch.sx(1), x2: ch.sx(NX), y1: ch.sy(slo), y2: ch.sy(slo), class: "ln-thin s5 dash" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(NX) - 4, y: ch.sy(slo) - 5, "text-anchor": "end", class: "lbl-sm", style: { fill: "var(--c5)" }, text: "SLO " + slo + " ms" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(NX) - 170, y: ch.sy(ymax) + 12, class: "lbl-sm", style: { fill: "var(--c2)" }, text: "— no chunking" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(NX) - 70, y: ch.sy(ymax) + 12, class: "lbl-sm", style: { fill: "var(--c3)" }, text: "— chunked" }));
        c.stage.appendChild(ch.svg);
        var mA = Math.max.apply(null, A.itl), mB = Math.max.apply(null, Bc.itl);
        var vA = A.itl.filter(function (v) { return v > slo; }).length, vB = Bc.itl.filter(function (v) { return v > slo; }).length;
        c.readout.innerHTML = stats([
          ["worst ITL · no chunking", fmtMs(mA), "var(--c2)"], ["worst ITL · chunked", fmtMs(mB), "var(--c3)"],
          ["long TTFT · no chunking", A.ttft !== null ? fmtMs(A.ttft) : "—"], ["long TTFT · chunked", Bc.ttft !== null ? fmtMs(Bc.ttft) : "not yet"],
          ["SLO misses (no chunk / chunked)", vA + " / " + vB],
        ]) + "<div style='margin-top:6px'>Trade-off: chunking bounds every step at ≈ budget tokens, so streams stay smooth; the long request's TTFT gets a little <i>worse</i> (its prefill is spread over " +
          Bc.steps.filter(function (s) { return s.pre; }).length + " steps and it re-reads earlier chunks' KV). Smaller budget → smoother ITL, slower prefill. " + (Bc.starve ? "<b style='color:var(--red)'>Budget ≤ decode tokens: the prefill starves!</b> " : "") +
          "<span class='muted'>Model: Llama-3.1-8B BF16 on one H100, roofline step time, streams at ~2k context. vLLM V1 enables chunked prefill by default (<code>max_num_batched_tokens</code> = budget).</span></div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- latency vs throughput
  V.register("latency-throughput", {
    title: "The fundamental trade-off: batch size vs per-user speed vs total throughput",
    desc: "One decode step reads all the weights once, whether it serves 1 user or 100. So batching multiplies total throughput almost for free — until compute (or KV-cache reads) catch up. Hover the chart. The SLO line is the per-user speed you promised.",
    render: function (c, p, K) {
      var mk = pick(MODELS, p.model, "llama8"), gk = pick(GPUS, p.gpu, "H100"), wk = pick(WDT, p.weightsDtype, "bf16"), ngSel = p.gpus ? String(p.gpus) : "auto";
      var CTX = [512, 1024, 2048, 4096, 8192, 16384, 32768], ctx = nearest(CTX, +p.context || 2048), slo = +p.slo || 30, ovh = 1, cur = +p.batch || 32;
      c.controls.appendChild(K.select({ label: "model", options: opts(MODELS), value: mk, onChange: function (v) { mk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "weights", options: opts(WDT), value: wk, onChange: function (v) { wk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "GPU", options: opts(GPUS), value: gk, onChange: function (v) { gk = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "# GPUs", options: [{ value: "auto", label: "auto (fit weights)" }, "1", "2", "4", "8"], value: ngSel, onChange: function (v) { ngSel = v; draw(); } }));
      c.controls.appendChild(idxSlider(K, { label: "avg context", values: CTX, value: ctx, fmt: fmtTok, onInput: function (v) { ctx = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "SLO per user", min: 5, max: 150, step: 5, value: slo, fmt: function (v) { return v + " tok/s"; }, onInput: function (v) { slo = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "overhead / step", min: 0, max: 5, step: 0.25, value: ovh, fmt: function (v) { return v + " ms"; }, onInput: function (v) { ovh = v; draw(); } }));

      function model() {
        var m = MODELS[mk], g = GPUS[gk], wb = WDT[wk].b, n = ngSel === "auto" ? autoGpus(m, g, wb) : +ngSel;
        var bw = g.bw * 1e9 * n * BW_EFF, fl = peakFlops(g, wk) * n * FLOP_EFF, kvT = kvElems(m) * 2, act = m.Pact || m.P;
        var comm = n > 1 ? 2 * m.L * 15e-6 : 0, free = g.mem * 1e9 * g.util * n - n * 1.5e9 - m.P * wb, bmax = free > 0 ? Math.floor(free / (kvT * ctx)) : 0;
        function st(b) {
          var mem = (wRead(m, wb, b) + b * kvT * ctx) / bw, cmp = b * (2 * act + 4 * m.L * m.d * ctx) / fl;
          return { t: Math.max(mem, cmp) + ovh / 1e3 + comm, mem: mem, cmp: cmp };
        }
        return { m: m, g: g, n: n, st: st, bmax: bmax, free: free, comm: comm };
      }
      function draw(hoverOnly) {
        var M = model(), BMAX = 1024, pts = [], knee = null, bslo = 0;
        for (var i = 0; i <= 90; i++) {
          var b = Math.pow(BMAX, i / 90), s = M.st(b);
          pts.push([b, 1 / s.t, b / s.t]);
        }
        for (var b2 = 1; b2 <= BMAX; b2++) {
          var s2 = M.st(b2);
          if (knee === null && s2.cmp >= s2.mem) knee = b2;
          if (1 / s2.t >= slo && b2 <= M.bmax) bslo = b2;
        }
        var ymin = Math.pow(10, Math.floor(Math.log10(Math.min(slo, pts[pts.length - 1][1]) * 0.8)));
        var ymax = Math.pow(10, Math.ceil(Math.log10(pts[pts.length - 1][2] * 1.3)));
        var ch = K.chart({ w: 680, h: 330, x: [1, BMAX], y: [ymin, ymax], xLog: true, yLog: true, xTicks: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
          xLabel: "batch size = concurrent users (log scale)", yLabel: "tokens / second (log scale)", pad: { r: 20 } });
        var gx = ch.g;
        if (M.bmax < BMAX) {
          var xm = ch.sx(Math.max(1, M.bmax));
          gx.appendChild(K.s("rect", { x: xm, y: ch.pad.t, width: ch.W - ch.pad.r - xm, height: ch.H - ch.pad.b - ch.pad.t, class: "fmuted", opacity: 0.14 }));
          gx.appendChild(K.s("text", { x: Math.min(xm + 6, ch.W - 150), y: ch.pad.t + 14, class: "lbl-sm", style: { fill: "var(--red)" }, text: M.bmax < 1 ? "doesn't fit in memory at all" : "KV cache won't fit beyond " + M.bmax }));
        }
        if (knee) {
          gx.appendChild(K.s("line", { x1: ch.sx(knee), x2: ch.sx(knee), y1: ch.pad.t, y2: ch.H - ch.pad.b, class: "ln-thin s4 dash" }));
          gx.appendChild(K.s("text", { x: ch.sx(knee) + 4, y: ch.H - ch.pad.b - 8, class: "lbl-sm", style: { fill: "var(--c4)" }, text: "compute-bound →" }));
          gx.appendChild(K.s("text", { x: ch.sx(knee) - 4, y: ch.H - ch.pad.b - 8, "text-anchor": "end", class: "lbl-sm", style: { fill: "var(--c4)" }, text: "← memory-bound" }));
        }
        gx.appendChild(K.s("path", { d: K.path(pts.map(function (q) { return [q[0], q[1]]; }), ch.sx, ch.sy), class: "ln s1" }));
        gx.appendChild(K.s("path", { d: K.path(pts.map(function (q) { return [q[0], q[2]]; }), ch.sx, ch.sy), class: "ln s3" }));
        gx.appendChild(K.s("text", { x: ch.sx(1) + 6, y: ch.sy(pts[0][1]) + 18, class: "lbl-sm", style: { fill: "var(--c1)", fontWeight: 700 }, text: "per-user tok/s" }));
        var lp = pts[30];
        gx.appendChild(K.s("text", { x: ch.sx(lp[0]) - 4, y: ch.sy(lp[2]) - 10, "text-anchor": "end", class: "lbl-sm", style: { fill: "var(--c3)", fontWeight: 700 }, text: "total tok/s (all users)" }));
        gx.appendChild(K.s("line", { x1: ch.sx(1), x2: ch.sx(BMAX), y1: ch.sy(slo), y2: ch.sy(slo), class: "ln-thin s5 dash" }));
        gx.appendChild(K.s("text", { x: ch.sx(1) + 6, y: ch.sy(slo) + 13, class: "lbl-sm", style: { fill: "var(--c5)" }, text: "SLO " + slo + " tok/s per user" }));
        if (bslo) {
          var ss = M.st(bslo);
          gx.appendChild(K.s("circle", { cx: ch.sx(bslo), cy: ch.sy(bslo / ss.t), r: 6, class: "f5" }));
          gx.appendChild(K.s("circle", { cx: ch.sx(bslo), cy: ch.sy(1 / ss.t), r: 4, class: "f5" }));
        }
        var cb = K.clamp(Math.round(cur), 1, BMAX), cs = M.st(cb);
        gx.appendChild(K.s("line", { x1: ch.sx(cb), x2: ch.sx(cb), y1: ch.pad.t, y2: ch.H - ch.pad.b, class: "ln-thin sfg", opacity: 0.35 }));
        gx.appendChild(K.s("circle", { cx: ch.sx(cb), cy: ch.sy(1 / cs.t), r: 4, class: "f1" }));
        gx.appendChild(K.s("circle", { cx: ch.sx(cb), cy: ch.sy(cb / cs.t), r: 4, class: "f3" }));
        var bxl = ch.sx(cb) > 470 ? ch.sx(cb) - 178 : ch.sx(cb) + 8;
        gx.appendChild(K.s("rect", { x: bxl, y: ch.pad.t + 22, width: 170, height: 50, rx: 6, class: "box", opacity: 0.95 }));
        [["batch " + cb + " · step " + fmtMs(cs.t * 1e3), "var(--fg)"], ["per user " + K.fmtNum(1 / cs.t, 0) + " tok/s", "var(--c1)"], ["total " + K.fmtNum(cb / cs.t, 0) + " tok/s", "var(--c3)"]].forEach(function (t, i) {
          gx.appendChild(K.s("text", { x: bxl + 8, y: ch.pad.t + 37 + i * 14, class: "lbl-sm", style: { fill: t[1] }, text: t[0] }));
        });
        ch.svg.addEventListener("pointermove", function (ev) {
          var q = svgPt(ch.svg, ev);
          if (q.x < ch.pad.l || q.x > ch.W - ch.pad.r) return;
          var nb = Math.round(K.clamp(ch.sx.invert(q.x), 1, BMAX));
          if (nb !== Math.round(cur)) { cur = nb; draw(true); }
        });
        K.clear(c.stage).appendChild(ch.svg);
        if (hoverOnly) return;

        var agg = bslo ? bslo / M.st(bslo).t : 0, perHr = agg * 3600, usd = 2.5 * M.n;
        c.readout.innerHTML = "<div>" + K.tex("t_{step}(b) = \\max\\!\\Big(\\underbrace{\\tfrac{\\text{weight bytes} + b\\cdot \\text{KV bytes/seq}}{\\text{bandwidth}}}_{\\text{memory}},\\ \\underbrace{\\tfrac{2\\cdot N_{active}\\cdot b}{\\text{FLOPS}}}_{\\text{compute}}\\Big) + \\text{overhead}") +
          " &nbsp; per-user = " + K.tex("1/t_{step}") + ", total = " + K.tex("b/t_{step}") + ".</div>" +
          stats([
            ["setup", M.n + "× " + gk + " · " + WDT[wk].label], ["batch 1: per user", K.fmtNum(1 / M.st(1).t, 0) + " tok/s"], ["best batch within SLO", bslo ? bslo : "none", "var(--c5)"],
            ["total tok/s at that batch", bslo ? K.fmtNum(agg, 0) : "—", "var(--c3)"], ["max batch (KV memory)", K.fmtNum(Math.max(0, M.bmax), 0)], ["≈ $/1M tokens @ $2.50/GPU-h", bslo ? "$" + (usd / (perHr / 1e6)).toFixed(2) : "—"],
          ]) + "<div class='muted' style='margin-top:6px'>Left of the knee, adding users barely slows anyone (weights are read once per step regardless) — free throughput. Right of it, compute is the limit and per-user speed falls ~1/b. Long contexts add per-user KV reads that erode the free region; MoE models read more experts as the batch grows. " + (M.comm ? "Sharded over " + M.n + " GPUs: adds ≈" + fmtMs(M.comm * 1e3) + "/step for 2 all-reduces per layer. " : "") + "Real engines also lose some speed to attention kernels and scheduling.</div>";
      }
      draw();
    },
  });
})();
