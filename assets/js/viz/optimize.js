/* optimize.js — interactive explainers for number formats, quantization and speculative decoding. */
(function () {
  "use strict";
  var V = window.Viz;

  // ---------------------------------------------------------------- helpers
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
  function T(K, x, y, text, cls, extra) {
    var a = { x: x, y: y, class: cls || "lbl-sm", text: text };
    if (extra) Object.keys(extra).forEach(function (k) { a[k] = extra[k]; });
    return K.s("text", a);
  }
  function col(n) { return { fill: "var(--c" + n + ")" }; }
  var MUTED = { fill: "var(--muted)" };
  function stat(label, val) { return "<div class='stat'><small>" + label + "</small><b>" + val + "</b></div>"; }
  function grid(items) { return "<div class='stat-grid' style='margin-bottom:8px'>" + items.join("") + "</div>"; }
  function table(head, rows, hlIdx, wrapLast) {
    var td = "padding:3px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap";
    var h = "<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:12.5px;margin-top:6px'><tr>" +
      head.map(function (x, i) { return "<th style='" + td + (i ? "" : ";text-align:left") + ";color:var(--muted);font-weight:600'>" + x + "</th>"; }).join("") + "</tr>";
    rows.forEach(function (r, ri) {
      h += "<tr style='" + (ri === hlIdx ? "background:var(--accent-bg)" : "") + "'>" +
        r.map(function (x, i) { return "<td style='" + td + (i ? "" : ";text-align:left") + (wrapLast && i === r.length - 1 ? ";white-space:normal;text-align:left;min-width:180px" : "") + "'>" + x + "</td>"; }).join("") + "</tr>";
    });
    return h + "</table></div>";
  }
  // round half to even (what hardware does)
  function rhe(v) { var f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); }
  function fmtV(v, prec) {
    if (v !== v) return "NaN";
    if (v === Infinity) return "+inf";
    if (v === -Infinity) return "−inf";
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (Number.isInteger(v) && a < 1e6) return String(v);
    if (a >= 1e6 || a < 1e-4) return v.toExponential(Math.max(0, (prec || 4) - 1)).replace("e+", "e");
    return String(+v.toPrecision(prec || 4));
  }

  // ---------------------------------------------------------------- number-formats
  var FMTS = [
    { id: "fp32", name: "FP32", E: 8, M: 23, bias: 127, kind: "ieee", use: "master weights, optimizer state, matmul accumulators" },
    { id: "bf16", name: "BF16", E: 8, M: 7, bias: 127, kind: "ieee", use: "default training & inference dtype (FP32's range, less precision)" },
    { id: "fp16", name: "FP16", E: 5, M: 10, bias: 15, kind: "ieee", use: "older inference default; overflows above 65504" },
    { id: "e4m3", name: "FP8 E4M3", E: 4, M: 3, bias: 7, kind: "fn", use: "FP8 weights/activations on H100/B200 (no inf, max 448)" },
    { id: "e5m2", name: "FP8 E5M2", E: 5, M: 2, bias: 15, kind: "ieee", use: "FP8 gradients (more range, less precision)" },
    { id: "e2m1", name: "FP4 E2M1", E: 2, M: 1, bias: 1, kind: "fin", use: "MXFP4 / NVFP4 elements, always with a shared block scale" },
    { id: "int8", name: "INT8", bits: 8, int: true, use: "W8A8 / KV cache, with per-channel or per-token scale" },
    { id: "int4", name: "INT4", bits: 4, int: true, use: "weight-only quant (GPTQ, AWQ) with a scale per group of 128" },
  ];
  function fmax(f) {
    var eTop = f.kind === "ieee" ? Math.pow(2, f.E) - 2 : Math.pow(2, f.E) - 1;
    var mTop = f.kind === "fn" ? Math.pow(2, f.M) - 2 : Math.pow(2, f.M) - 1;
    return { e: eTop, m: mTop, v: (1 + mTop / Math.pow(2, f.M)) * Math.pow(2, eTop - f.bias) };
  }
  function fdec(f, s, e, m) {
    var two = Math.pow(2, f.M), sg = s ? -1 : 1;
    if (f.kind === "ieee" && e === Math.pow(2, f.E) - 1) return m === 0 ? sg * Infinity : NaN;
    if (f.kind === "fn" && e === Math.pow(2, f.E) - 1 && m === two - 1) return NaN;
    if (e === 0) return sg * (m / two) * Math.pow(2, 1 - f.bias);
    return sg * (1 + m / two) * Math.pow(2, e - f.bias);
  }
  function fpack(f, s, e, m, note) {
    var v = fdec(f, s, e, m);
    var bits = [{ b: String(s), cls: "f5" }];
    e.toString(2).padStart(f.E, "0").split("").forEach(function (b) { bits.push({ b: b, cls: "f1" }); });
    m.toString(2).padStart(f.M, "0").split("").forEach(function (b) { bits.push({ b: b, cls: "f3" }); });
    return { s: s, e: e, m: m, val: v, note: note || "", bits: bits, ulp: isFinite(v) ? Math.pow(2, Math.max(e, 1) - f.bias - f.M) : NaN };
  }
  function fenc(f, x) {
    var s = (x < 0 || (x === 0 && 1 / x < 0)) ? 1 : 0, a = Math.abs(x), two = Math.pow(2, f.M), mx = fmax(f), e, m, note = "";
    var minN = Math.pow(2, 1 - f.bias);
    if (a === Infinity) {
      if (f.kind === "ieee") return fpack(f, s, Math.pow(2, f.E) - 1, 0, "infinity");
      return fpack(f, s, mx.e, mx.m, "no inf here → saturated to max");
    }
    if (a < minN) {
      var mi = rhe(a / Math.pow(2, 1 - f.bias - f.M));
      if (mi >= two) { e = 1; m = 0; }
      else { e = 0; m = mi; note = mi === 0 ? (a > 0 ? "underflow → 0" : "") : "subnormal: fewer significant bits"; }
    } else {
      var ex = Math.floor(Math.log2(a));
      if (Math.pow(2, ex) > a) ex--;
      if (Math.pow(2, ex + 1) <= a) ex++;
      var mi2 = rhe(a / Math.pow(2, ex - f.M));
      if (mi2 >= 2 * two) { ex++; mi2 = two; }
      e = ex + f.bias; m = mi2 - two;
    }
    if (e > mx.e || (e === mx.e && m > mx.m)) {
      if (f.kind === "ieee") return fpack(f, s, Math.pow(2, f.E) - 1, 0, "overflow → ±inf");
      return fpack(f, s, mx.e, mx.m, "overflow → saturated to ±max");
    }
    return fpack(f, s, e, m, note);
  }
  function ienc(f, x, R) {
    var qm = Math.pow(2, f.bits - 1) - 1, sc = R / qm, q = Math.round(x / sc), note = "";
    if (!isFinite(q)) q = x > 0 ? qm : -qm;
    if (q > qm) { q = qm; note = "clipped to +R"; }
    if (q < -qm) { q = -qm; note = "clipped to −R"; }
    if (q === 0 && x !== 0 && !note) note = "rounds to 0";
    var tc = (q + Math.pow(2, f.bits)) % Math.pow(2, f.bits);
    var bits = tc.toString(2).padStart(f.bits, "0").split("").map(function (b) { return { b: b, cls: "f6" }; });
    return { q: q, val: q * sc, ulp: sc, note: note, bits: bits, sc: sc, qm: qm };
  }

  V.register("number-formats", {
    title: "Number formats: how FP32, BF16, FP16, FP8, FP4 and INT8/INT4 store a value",
    desc: "A float is (−1)<sup>sign</sup> × 2<sup>exponent − bias</sup> × 1.mantissa. <b>Exponent</b> bits buy range, <b>mantissa</b> bits buy precision. Type a value to see the nearest representable number in every format and the rounding error. Integers store q with a shared scale: value ≈ q × s. Click a row to decode it.",
    render: function (c, p, K) {
      var x = p.value !== undefined && isFinite(+p.value) ? +p.value : Math.PI;
      var R = +p.range > 0 ? +p.range : 4;
      var list = FMTS;
      if (p.formats) {
        var want = String(p.formats).toLowerCase().split(/[\s,]+/);
        var l2 = FMTS.filter(function (f) { return want.indexOf(f.id) >= 0; });
        if (l2.length) list = l2;
      }
      var sel = Math.max(0, list.findIndex(function (f) { return f.id === "bf16"; }));
      var num = K.number({ label: "value", value: x, onInput: function (v) { if (isFinite(v) && String(num.input.value).trim() !== "") { x = v; draw(); } } });
      num.input = num.querySelector("input");
      num.input.style.width = "130px";
      c.controls.appendChild(num);
      [["π", Math.PI], ["0.1", 0.1], ["1/3", 1 / 3], ["−2.5", -2.5], ["300", 300], ["70000", 70000], ["1e-5", 1e-5], ["1e-8", 1e-8]].forEach(function (pr) {
        c.controls.appendChild(K.button(pr[0], function () { x = pr[1]; num.set(+pr[1].toPrecision(12)); draw(); }));
      });
      var rs = K.slider({ label: "INT range ±R (scale s = R/qmax)", min: -1, max: 3, step: 0.001, value: Math.log10(R), fmt: function (v) { return "±" + fmtV(Math.pow(10, v), 3); }, onInput: function (v) { R = +Math.pow(10, v).toPrecision(3); draw(); } });
      c.controls.appendChild(rs);

      function enc(f) { return f.int ? ienc(f, x, R) : fenc(f, x); }
      function draw() {
        var rows = list.map(enc);
        var W = 680, top = 30, rh = 38, H = top + rows.length * rh + 2, bw = 12;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var lx = 112;
        [["f5", "sign"], ["f1", "exponent"], ["f3", "mantissa"], ["f6", "integer q (two's complement)"]].forEach(function (lg) {
          svg.appendChild(K.s("rect", { x: lx, y: 8, width: 11, height: 11, rx: 2, class: lg[0] }));
          svg.appendChild(T(K, lx + 15, 18, lg[1], "lbl-sm"));
          lx += 30 + lg[1].length * 6.2;
        });
        svg.appendChild(T(K, 520, 18, "stored value · error", "lbl-sm", { style: MUTED }));
        rows.forEach(function (r, i) {
          var f = list[i], y = top + i * rh;
          var g = K.s("g", { style: { cursor: "pointer" }, onclick: function () { sel = i; draw(); } });
          g.appendChild(K.s("rect", { x: 0, y: y, width: W, height: rh - 3, rx: 6, class: i === sel ? "box-hl" : "fbg", opacity: i === sel ? 1 : 0.01 }));
          g.appendChild(T(K, 8, y + 15, f.name, "lbl"));
          g.appendChild(T(K, 8, y + 29, f.int ? "s = R/" + r.qm + " = " + fmtV(r.sc, 3) : "E" + f.E + "M" + f.M + " · bias " + f.bias, "lbl-sm", { style: MUTED }));
          var bx = 112;
          r.bits.forEach(function (bt, j) {
            if (j > 0 && bt.cls !== r.bits[j - 1].cls) bx += 4;
            g.appendChild(K.s("rect", { x: bx, y: y + 6, width: bw - 1.5, height: 21, rx: 2, class: bt.cls }));
            g.appendChild(T(K, bx + (bw - 1.5) / 2, y + 21, bt.b, "lbl-sm t-white mono", { "text-anchor": "middle", style: { fontSize: "10px" } }));
            bx += bw;
          });
          g.appendChild(T(K, 520, y + 15, fmtV(r.val, f.id === "fp32" ? 9 : 7), "lbl mono"));
          var err = Math.abs(r.val - x), rel = x !== 0 ? err / Math.abs(x) : (err === 0 ? 0 : Infinity);
          var bad = /overflow|underflow|clipped|saturated|→ 0/.test(r.note) || rel > 0.05;
          var line = r.note && !/^subnormal/.test(r.note) ? r.note : "err " + fmtV(err, 2) + (x !== 0 && isFinite(rel) ? " (" + fmtV(rel * 100, 2) + "%)" : "") + (r.note ? " · subnormal" : "");
          g.appendChild(T(K, 520, y + 30, line, "lbl-sm", { style: bad ? col(5) : MUTED }));
          svg.appendChild(g);
        });
        K.clear(c.stage).appendChild(svg);

        // ---- readout
        var f = list[sel], r = rows[sel], h = "";
        if (f.int) {
          h += "<p><b>" + f.name + "</b>: s = R / " + r.qm + " = " + fmtV(r.sc, 4) + "; q = round(" + fmtV(x, 6) + " / s) = <b>" + r.q + "</b> → stored bits " +
            r.bits.map(function (b) { return b.b; }).join("") + " (two's complement) → value = q × s = <b>" + fmtV(r.val, 7) + "</b>. The grid is uniform: every value in ±R gets the same absolute step s, so small values lose relative precision.</p>";
        } else if (!isFinite(r.val)) {
          h += "<p><b>" + f.name + "</b>: exponent field all ones with mantissa " + (r.val !== r.val ? "≠ 0 → NaN" : "0 → infinity") + ". The largest finite " + f.name + " is " + fmtV(fmax(f).v, 5) + ".</p>";
        } else {
          var eb = r.e.toString(2).padStart(f.E, "0"), mb = r.m.toString(2).padStart(f.M, "0"), two = Math.pow(2, f.M);
          h += "<p><b>" + f.name + "</b>: sign " + r.s + " · exponent <span class='mono'>" + eb + "</span>₂ = " + r.e + " · mantissa <span class='mono'>" + mb + "</span>₂ = " + r.m + " → ";
          if (r.e === 0) h += "subnormal: (−1)<sup>" + r.s + "</sup> × 2<sup>1−" + f.bias + "</sup> × (0 + " + r.m + "/" + two + ")";
          else h += "(−1)<sup>" + r.s + "</sup> × 2<sup>" + r.e + "−" + f.bias + "</sup> × (1 + " + r.m + "/" + two + ") = " + (r.s ? "−" : "") + fmtV(Math.pow(2, r.e - f.bias), 6) + " × " + fmtV(1 + r.m / two, 8);
          h += " = <b>" + fmtV(r.val, 9) + "</b>. Neighbouring values here are " + fmtV(r.ulp, 3) + " apart.</p>";
        }
        h += table(["format", "stored value", "abs error", "rel error", "step (ulp) here", "note"], rows.map(function (rr, i) {
          var err = Math.abs(rr.val - x);
          return [list[i].name, "<span class='mono'>" + fmtV(rr.val, list[i].id === "fp32" ? 9 : 7) + "</span>", fmtV(err, 2),
            x !== 0 ? (isFinite(err / Math.abs(x)) ? fmtV(100 * err / Math.abs(x), 2) + "%" : "∞") : "—", isFinite(rr.ulp) ? fmtV(rr.ulp, 3) : "—", rr.note || ""];
        }), sel);
        h += table(["format", "bits", "max finite", "min normal", "min subnormal", "ε = 2<sup>−M</sup>", "typical use"], list.map(function (ff) {
          if (ff.int) {
            var qm = Math.pow(2, ff.bits - 1) - 1;
            return [ff.name, ff.bits, "R = " + fmtV(R, 3), "—", "s = " + fmtV(R / qm, 3), "uniform step s", ff.use];
          }
          return [ff.name, (1 + ff.E + ff.M) + " (1+" + ff.E + "+" + ff.M + ")", fmtV(fmax(ff).v, 5), fmtV(Math.pow(2, 1 - ff.bias), 3),
            fmtV(Math.pow(2, 1 - ff.bias - ff.M), 3), fmtV(Math.pow(2, -ff.M), 3), ff.use];
                    }), sel, true);
        h += "<p class='muted'>BF16 keeps FP32's 8 exponent bits (same range, ~3 significant digits), which is why it replaced FP16: no overflow, no loss scaling. " +
          "FP8/FP4 have so little range that real kernels multiply by a scale per tensor, per block of 32 (MXFP) or 16 (NVFP4) values. Try 70000 (FP16 overflows), 300 (E4M3 near its max) and 1e-8 (underflow).</p>";
        c.readout.innerHTML = h;
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- quantize-weights
  var GRANS = [
    { value: "tensor", label: "per-tensor (1 scale)" },
    { value: "channel", label: "per-channel (1 scale / row)" },
    { value: "group", label: "per-group (1 scale / G weights)" },
  ];
  V.register("quantize-weights", {
    title: "Quantizing weights: bits, scale granularity and outliers",
    desc: "Symmetric absmax quantization: s = max|w| / (2<sup>b−1</sup>−1), q = round(w / s), ŵ = q·s. One scale for the whole tensor, one per output channel (row), or one per group of G consecutive weights. A few large outliers stretch s, so the grid (vertical lines) gets too coarse for the ordinary weights and many of them round to 0.",
    render: function (c, p, K) {
      var bits = K.clamp(Math.round(+p.bits || 4), 2, 8);
      var gran = ["tensor", "channel", "group"].indexOf(p.granularity) >= 0 ? p.granularity : "channel";
      var G = [32, 64, 128, 256].indexOf(+p.group) >= 0 ? +p.group : 128;
      var outl = p.outliers !== false && p.outliers !== "false", logY = !!p.log, insp = 0;
      var C = 32, N = 512, n = C * N, r = K.rng(+p.seed || 42);
      var std = [], base = new Float64Array(n), outs = [];
      for (var ch = 0; ch < C; ch++) {
        std.push(0.02 * Math.exp(0.45 * K.randn(r)));
        for (var j = 0; j < N; j++) base[ch * N + j] = K.randn(r) * std[ch];
      }
      for (var o = 0; o < Math.round(n * 0.002); o++) {
        var idx = Math.floor(r() * n);
        outs.push([idx, (8 + 12 * r()) * std[Math.floor(idx / N)] * (r() < 0.5 ? -1 : 1)]);
      }
      var w;
      function weights() { w = Float64Array.from(base); if (outl) outs.forEach(function (q) { w[q[0]] = q[1]; }); }
      function gsize(g) { return g === "tensor" ? n : g === "channel" ? N : G; }
      function quant(b, size) {
        var qm = Math.pow(2, b - 1) - 1, wq = new Float64Array(n), qs = new Int16Array(n), sc = [], se = 0, ss = 0, z = 0;
        for (var g0 = 0; g0 < n; g0 += size) {
          var am = 0, i;
          for (i = g0; i < g0 + size; i++) am = Math.max(am, Math.abs(w[i]));
          var s = am / qm || 1;
          sc.push(s);
          for (i = g0; i < g0 + size; i++) {
            var q = K.clamp(Math.round(w[i] / s), -qm, qm), e = w[i] - q * s;
            qs[i] = q; wq[i] = q * s; if (q === 0) z++; se += e * e; ss += w[i] * w[i];
          }
        }
        return { wq: wq, q: qs, sc: sc, mse: se / n, snr: 10 * Math.log10(ss / se), zeros: z / n, qm: qm, size: size };
      }
      var snrTab;
      function computeTab() {
        snrTab = {};
        ["tensor", "channel", "group"].forEach(function (g) { snrTab[g] = []; for (var b = 2; b <= 8; b++) snrTab[g].push(quant(b, gsize(g))); });
      }
      function outlierGroup() { var sz = gsize(gran); return outl && outs.length ? Math.floor(outs[0][0] / sz) : 0; }

      c.controls.appendChild(K.slider({ label: "bits b", min: 2, max: 8, step: 1, value: bits, fmt: function (v) { return v + " bits (" + (Math.pow(2, v) - 1) + " levels)"; }, onInput: function (v) { bits = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "granularity", options: GRANS, value: gran, onChange: function (v) { gran = v; resetInsp(); draw(); } }));
      var gsel = K.select({ label: "group size G", options: [32, 64, 128, 256].map(function (v) { return { value: String(v), label: String(v) }; }), value: String(G), onChange: function (v) { G = +v; computeTab(); resetInsp(); draw(); } });
      c.controls.appendChild(gsel);
      c.controls.appendChild(K.toggle({ label: "outliers (0.2% of weights at 8–20σ)", value: outl, onChange: function (v) { outl = v; weights(); computeTab(); resetInsp(); draw(); } }));
      c.controls.appendChild(K.toggle({ label: "log count axis", value: logY, onChange: function (v) { logY = v; draw(); } }));
      var isl = K.slider({ label: "inspect group #", min: 0, max: 1, step: 1, value: 0, onInput: function (v) { insp = v; draw(); } });
      c.controls.appendChild(isl);
      function resetInsp() {
        var ng = n / gsize(gran);
        isl.input.max = Math.max(0, ng - 1);
        insp = Math.min(outlierGroup(), ng - 1);
        isl.set(insp);
        isl.style.display = ng > 1 ? "" : "none";
        gsel.style.display = gran === "group" ? "" : "none";
      }
      weights(); computeTab(); resetInsp();

      function draw() {
        var Q = snrTab[gran][bits - 2], sz = Q.size, g0 = insp * sz, s = Q.sc[insp], am = s * Q.qm;
        var lo = -am * 1.06, hi = am * 1.06, NB = 90, bwid = (hi - lo) / NB, cnt = new Array(NB).fill(0), lvl = {}, nOut = 0;
        for (var i = g0; i < g0 + sz; i++) {
          cnt[K.clamp(Math.floor((w[i] - lo) / bwid), 0, NB - 1)]++;
          lvl[Qq(i)] = (lvl[Qq(i)] || 0) + 1;
          if (Math.abs(w[i]) > 6 * std[Math.floor(i / N)]) nOut++;
        }
        function Qq(i) { return Q.q[i]; }
        var cmax = Math.max.apply(null, cnt), lmax = 0;
        Object.keys(lvl).forEach(function (k) { lmax = Math.max(lmax, lvl[k]); });
        var ch2 = K.chart({ w: 680, h: 270, x: [lo, hi], y: logY ? [0.8, cmax * 1.6] : [0, cmax * 1.12], yLog: logY, pad: { l: 52, r: 14, t: 26, b: 40 },
          xLabel: "weight value (" + (gran === "tensor" ? "whole tensor" : gran === "channel" ? "channel " + insp : "group " + insp + " = channel " + Math.floor(g0 / N) + ", cols " + (g0 % N) + "–" + (g0 % N + sz - 1)) + ")",
          yLabel: "count", xFmt: function (v) { return Math.abs(v) < am * 1e-6 ? "0" : fmtV(v, 2); }, yFmt: function (v) { return K.fmtShort(v); } });
        var sx = ch2.sx, sy = ch2.sy, y0 = sy(logY ? 0.8 : 0), gg = ch2.g;
        cnt.forEach(function (v, b) {
          if (!v) return;
          gg.appendChild(K.s("rect", { x: sx(lo + b * bwid) + 0.3, y: sy(v), width: Math.max(0.5, sx(lo + (b + 1) * bwid) - sx(lo + b * bwid) - 0.6), height: y0 - sy(v), class: "f1", opacity: 0.5 }));
        });
        var px = sx(s) - sx(0);
        if (px >= 3) {
          for (var q = -Q.qm; q <= Q.qm; q++) gg.appendChild(K.s("line", { x1: sx(q * s), x2: sx(q * s), y1: ch2.pad.t, y2: y0, class: "ln-thin s5", opacity: q === 0 ? 0.9 : 0.45 }));
        }
        Object.keys(lvl).forEach(function (k) {
          var v = lvl[k], hgt = logY ? Math.pow(10, Math.log10(0.8) + (Math.log10(v) + 0.3) / (Math.log10(lmax) + 0.3) * (Math.log10(cmax * 1.6) - Math.log10(0.8)) * 0.92) : v / lmax * cmax * 1.05;
          var xx = sx(+k * s);
          gg.appendChild(K.s("line", { x1: xx, x2: xx, y1: y0, y2: sy(hgt), class: "ln s5", opacity: 0.8 }));
          gg.appendChild(K.s("circle", { cx: xx, cy: sy(hgt), r: 2.6, class: "f5" }));
        });
        [-am, am].forEach(function (v) {
          gg.appendChild(K.s("line", { x1: sx(v), x2: sx(v), y1: ch2.pad.t - 4, y2: y0, class: "ln s2 dash" }));
        });
        gg.appendChild(T(K, sx(am) - 4, ch2.pad.t - 10, "absmax = " + fmtV(am, 3), "lbl-sm", { "text-anchor": "end", style: col(2) }));
        gg.appendChild(T(K, 60, 14, "bars: original weights · orange lines: the " + (2 * Q.qm + 1) + " levels q·s" + (px >= 3 ? "" : " (too dense to draw: " + fmtV(s, 2) + " apart)") + " · dots: weights per level (scaled)", "lbl-sm", { style: MUTED }));
        K.clear(c.stage).appendChild(ch2.svg);

        // SNR vs bits
        var smax = 0;
        ["tensor", "channel", "group"].forEach(function (g) { snrTab[g].forEach(function (qq) { smax = Math.max(smax, qq.snr); }); });
        var ch3 = K.chart({ w: 680, h: 200, x: [2, 8], y: [Math.min(0, Math.floor(snrTab.tensor[0].snr / 5) * 5), Math.ceil(smax / 10) * 10 + 2], pad: { l: 52, r: 150, t: 12, b: 38 },
          xTicks: [2, 3, 4, 5, 6, 7, 8], xLabel: "bits b", yLabel: "SNR (dB, higher = better)", xFmt: function (v) { return v; } });
        [["tensor", "s6", "per-tensor"], ["channel", "s2", "per-channel"], ["group", "s3", "per-group G=" + G]].forEach(function (L, li) {
          var pts = snrTab[L[0]].map(function (qq, bi) { return [bi + 2, qq.snr]; });
          ch3.g.appendChild(K.s("path", { d: K.path(pts, ch3.sx, ch3.sy), class: "ln " + L[1], fill: "none", style: { strokeWidth: L[0] === gran ? "3px" : "1.5px" }, opacity: L[0] === gran ? 1 : 0.6 }));
          var last = pts[pts.length - 1];
          ch3.g.appendChild(T(K, ch3.sx(8) + 8, ch3.sy(last[1]) + 4 + (li - 1) * 3, L[2] + " " + last[1].toFixed(0) + " dB", "lbl-sm", { style: col(L[1].slice(1)) }));
        });
        ch3.g.appendChild(K.s("circle", { cx: ch3.sx(bits), cy: ch3.sy(Q.snr), r: 5.5, class: "f5", style: { stroke: "var(--bg)", strokeWidth: "1.5px" } }));
        ch3.g.appendChild(T(K, ch3.sx(bits) + 8, ch3.sy(Q.snr) + 16, Q.snr.toFixed(1) + " dB", "lbl mono"));
        c.stage.appendChild(ch3.svg);

        // readout
        var bpw = function (qq) { return bits + 16 / qq.size; };
        var h = grid([
          stat("scale for this " + (gran === "tensor" ? "tensor" : gran === "channel" ? "channel" : "group"), fmtV(s, 3)),
          stat("outliers (&gt;6σ) here", nOut),
          stat("MSE · SNR", Q.mse.toExponential(1) + " · " + Q.snr.toFixed(1) + " dB"),
          stat("weights → 0", (100 * Q.zeros).toFixed(1) + "%"),
          stat("bits / weight (fp16 scales)", +bpw(Q).toFixed(3) + ""),
        ]);
        var gi = ["tensor", "channel", "group"].indexOf(gran);
        h += table(["granularity (" + bits + "-bit)", "# scales", "bits / weight", "MSE", "SNR", "→ 0", "8B-param model"], ["tensor", "channel", "group"].map(function (g, k) {
          var qq = snrTab[g][bits - 2];
          return [GRANS[k].label.replace("G ", G + " "), qq.sc.length, +bpw(qq).toFixed(3), qq.mse.toExponential(2), qq.snr.toFixed(1) + " dB", (100 * qq.zeros).toFixed(1) + "%", K.fmtBytes(8e9 * bpw(qq) / 8)];
        }), gi);
        h += "<p class='muted'>" + (gran === "tensor" && outl
          ? "One scale for everything: the single largest outlier decides s, so at low bits most ordinary weights fall inside the zero bucket. "
          : gran === "group" ? "Groups of " + G + " isolate an outlier: only its own group gets a coarse grid. Cost: one fp16 scale per " + G + " weights = +" + +(16 / G).toFixed(3) + " bits/weight. "
            : "Per-channel scales adapt to each row's spread (rows differ ~2× here) but one outlier still ruins its whole row. ") +
          "Each extra bit adds ≈ 6 dB SNR. 32×512 synthetic weights (seeded), σ≈0.02 per row. Real methods go further: GPTQ corrects rounding error with second-order info, AWQ rescales salient channels, SmoothQuant moves activation outliers into the weights.</p>";
        c.readout.innerHTML = h;
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- spec-decode
  var SD_TEXT = ("the big model reads every weight once per step so checking a few extra tokens in the same forward pass is almost free " +
    "when the small draft model guesses well and that is the whole trick behind speculative decoding").split(" ");
  var SD_WRONG = ["a", "it", "big", "ran", "dog", "but", "very", "blue", "so", "in", "cat", "not"];
  function sdE(a, k) { return (1 - Math.pow(a, k + 1)) / (1 - a); }
  V.register("spec-decode", {
    title: "Speculative decoding: a cheap draft guesses k tokens, the big model verifies them in one pass",
    desc: "A small draft model proposes k tokens; the target model scores all k+1 positions in <b>one</b> forward pass, keeps the longest accepted prefix and always adds one token of its own (a correction, or a bonus if all k were accepted). The output distribution is exactly the target model's.",
    render: function (c, p, K) {
      var a = K.clamp(+p.alpha || 0.8, 0.3, 0.95), k = K.clamp(Math.round(+p.k || 4), 1, 10), cc = K.clamp(+p.c || 0.05, 0.01, 0.5);
      function speed(al, kk) { return sdE(al, kk) / (1 + kk * cc); }
      function bestK(al) { var bk = 1; for (var kk = 1; kk <= 10; kk++) if (speed(al, kk) > speed(al, bk)) bk = kk; return bk; }
      c.controls.appendChild(K.slider({ label: "acceptance α", min: 0.3, max: 0.95, step: 0.01, value: a, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { a = v; reset(); draw(); } }));
      c.controls.appendChild(K.slider({ label: "draft length k", min: 1, max: 10, step: 1, value: k, onInput: function (v) { k = v; reset(); draw(); } }));
      c.controls.appendChild(K.slider({ label: "draft cost c (× target step)", min: 0.01, max: 0.5, step: 0.01, value: cc, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { cc = v; draw(); } }));
      var tm = ticker(c, 420, function () { step(); draw(); }, function (on) { pb.textContent = on ? "❚❚ Pause" : "▶ Play"; });
      var pb = K.button("▶ Play", function () { tm.toggle(); }, "primary");
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Step", function () { tm.stop(); step(); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { tm.stop(); reset(); draw(); }));

      var rng, pos, cur, hist, st;
      function newRound() {
        var acc = 0;
        while (acc < k && rng() < a) acc++;
        var toks = [];
        for (var i = 0; i < k; i++) {
          toks.push({ w: i < acc ? SD_TEXT[(pos + i) % SD_TEXT.length] : SD_WRONG[Math.floor(rng() * SD_WRONG.length)], st: i < acc ? "acc" : i === acc ? "rej" : "drop" });
        }
        toks.push({ w: SD_TEXT[(pos + acc) % SD_TEXT.length], st: acc === k ? "bonus" : "fix" });
        pos += acc + 1;
        return { toks: toks, acc: acc, shown: 0, verified: false, n: st.rounds + 1 };
      }
      function reset() { rng = K.rng(+p.seed || 7); pos = 0; hist = []; st = { rounds: 0, tokens: 0, acc: 0 }; cur = newRound(); }
      function step() {
        if (cur.shown < k) cur.shown++;
        else if (!cur.verified) { cur.verified = true; st.rounds++; st.tokens += cur.acc + 1; st.acc += cur.acc; }
        else { hist.unshift(cur); if (hist.length > 4) hist.pop(); cur = newRound(); }
      }
      reset();

      function draw() {
        // ---- chart: speedup vs k
        var E = sdE(a, k), S = speed(a, k), kb = bestK(a);
        var ymax = Math.max(2, Math.ceil(sdE(a, 10) * 1.12));
        var ch = K.chart({ w: 680, h: 260, x: [1, 10], y: [0, ymax], pad: { l: 50, r: 150, t: 14, b: 40 }, xTicks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          xLabel: "k = draft tokens per round", yLabel: "× tokens per target-step time", xFmt: function (v) { return v; } });
        var ks = []; for (var i = 1; i <= 10; i++) ks.push(i);
        var g = ch.g, sx = ch.sx, sy = ch.sy;
        g.appendChild(K.s("line", { x1: sx(1), x2: sx(10), y1: sy(1), y2: sy(1), class: "ln-thin smuted dash" }));
        var rl = [[sy(1), "1× = no gain", MUTED]];
        [0.5, 0.7, 0.9].forEach(function (al) {
          if (Math.abs(al - a) < 0.015) return;
          var pts = ks.map(function (kk) { return [kk, Math.min(speed(al, kk), ymax)]; });
          g.appendChild(K.s("path", { d: K.path(pts, sx, sy), class: "ln-thin smuted", fill: "none" }));
          rl.push([sy(pts[9][1]), "α=" + al, MUTED]);
        });
        var ep = ks.map(function (kk) { return [kk, Math.min(sdE(a, kk), ymax)]; });
        g.appendChild(K.s("path", { d: K.path(ep, sx, sy), class: "ln s2 dash", fill: "none" }));
        rl.push([sy(ep[9][1]), "E = tokens/round", col(2)]);
        var sp = ks.map(function (kk) { return [kk, speed(a, kk)]; });
        g.appendChild(K.s("path", { d: K.path(sp, sx, sy), class: "ln s1", fill: "none", style: { strokeWidth: "3px" } }));
        rl.push([sy(sp[9][1]), "speedup α=" + a.toFixed(2), col(1)]);
        rl.sort(function (u, v) { return u[0] - v[0]; });
        rl.forEach(function (L, li) { if (li && L[0] < rl[li - 1][0] + 13) L[0] = rl[li - 1][0] + 13; });
        rl.forEach(function (L) { g.appendChild(T(K, sx(10) + 6, L[0] + 4, L[1], "lbl-sm", { style: L[2] })); });
        sp.forEach(function (pt) { g.appendChild(K.s("circle", { cx: sx(pt[0]), cy: sy(pt[1]), r: 2.5, class: "f1" })); });
        g.appendChild(K.s("circle", { cx: sx(kb), cy: sy(speed(a, kb)), r: 8, fill: "none", class: "ln s3" }));
        g.appendChild(T(K, sx(kb), sy(speed(a, kb)) - 13, "best k = " + kb, "lbl-sm", { "text-anchor": "middle", style: col(3) }));
        g.appendChild(K.s("line", { x1: sx(k), x2: sx(k), y1: sy(0), y2: sy(E), class: "ln-thin s5 dash" }));
        g.appendChild(K.s("circle", { cx: sx(k), cy: sy(S), r: 5.5, class: "f5", style: { stroke: "var(--bg)", strokeWidth: "1.5px" } }));
        g.appendChild(K.s("circle", { cx: sx(k), cy: sy(E), r: 4, class: "f2" }));
        var svg = ch.svg;
        K.clear(c.stage).appendChild(svg);

        // ---- animation of rounds
        var W = 680, bw = 50, gap = 4, x0 = 88, rh = 36, rows = [cur].concat(hist), H = 34 + rows.length * rh;
        var an = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var lx = x0;
        [["f6", "drafted (unverified)"], ["f3", "accepted"], ["f5", "rejected"], ["fbg3", "discarded"], ["f1", "target's own token"]].forEach(function (lg) {
          an.appendChild(K.s("rect", { x: lx, y: 6, width: 11, height: 11, rx: 2, class: lg[0] }));
          an.appendChild(T(K, lx + 15, 16, lg[1], "lbl-sm"));
          lx += 30 + lg[1].length * 6;
        });
        rows.forEach(function (R, ri) {
          var y = 26 + ri * rh;
          an.appendChild(T(K, 4, y + 17, "round " + R.n, "lbl-sm", { style: ri ? MUTED : null }));
          an.appendChild(T(K, 4, y + 30, R.verified ? R.acc + " ok + 1 = " + (R.acc + 1) : "drafting " + R.shown + "/" + k, "lbl-sm", { style: R.verified ? col(3) : MUTED }));
          R.toks.forEach(function (t, ti) {
            var last = ti === k;
            if (last && !R.verified) return;
            if (!last && ti >= R.shown) return;
            var bx = x0 + ti * (bw + gap) + (last ? 6 : 0);
            var cls = !R.verified ? "f6" : t.st === "acc" ? "f3" : t.st === "rej" ? "f5" : t.st === "drop" ? "fbg3" : "f1";
            an.appendChild(K.s("rect", { x: bx, y: y + 6, width: bw, height: 24, rx: 4, class: cls, opacity: ri ? 0.7 : 1 }));
            an.appendChild(T(K, bx + bw / 2, y + 22, t.w, "lbl-sm mono" + (cls === "fbg3" ? "" : " t-white"), { "text-anchor": "middle", style: Object.assign(t.w.length > 7 ? { fontSize: "9.5px" } : {}, cls === "fbg3" ? MUTED : {}) }));
            if (R.verified && t.st === "rej") an.appendChild(K.s("line", { x1: bx + 5, x2: bx + bw - 5, y1: y + 18, y2: y + 18, class: "ln", style: { stroke: "#fff", strokeWidth: "1.5px" } }));
            if (last) an.appendChild(T(K, bx + bw / 2, y + 4, t.st === "bonus" ? "bonus" : "fix", "lbl-sm", { "text-anchor": "middle", style: Object.assign({ fontSize: "9px" }, col(1)) }));
          });
        });
        c.stage.appendChild(an);

        // ---- readout
        var emp = st.rounds ? st.tokens / st.rounds : 0;
        var h = grid([
          stat("E tokens / round", E.toFixed(2)),
          stat("cost / round = 1 + k·c", (1 + k * cc).toFixed(2)),
          stat("speedup", S.toFixed(2) + "×"),
          stat("best k for α=" + a.toFixed(2), kb + " → " + speed(a, kb).toFixed(2) + "×"),
          stat("P(all k accepted)", (100 * Math.pow(a, k)).toFixed(0) + "%"),
          stat("simulated rounds · tokens", st.rounds + " · " + st.tokens),
          stat("simulated tokens/round", st.rounds ? emp.toFixed(2) + " → " + (emp / (1 + k * cc)).toFixed(2) + "×" : "press Play"),
        ]);
        h += "<p>" + K.tex("E[\\text{tokens/round}] = 1 + \\alpha + \\dots + \\alpha^k = \\frac{1-\\alpha^{k+1}}{1-\\alpha}") + " &nbsp; " +
          K.tex("\\text{speedup} = \\frac{E}{1 + k\\,c}") + "</p>";
        h += "<p class='muted'><b>Assumptions:</b> verifying k+1 tokens costs the same as one normal decode step (decode is memory-bound: the weights are read once either way, so a few extra tokens are nearly free); " +
          "each draft token costs c target-steps; each token is accepted independently with probability α. Real gains are lower at large batch sizes, where verification becomes compute-bound, and α depends on the text (code and repetitive text accept more). " +
          "Larger k helps only while α<sup>k</sup> is still sizable — past the best k you pay for drafts that get thrown away.</p>";
        c.readout.innerHTML = h;
      }
      draw();
    },
  });
})();
