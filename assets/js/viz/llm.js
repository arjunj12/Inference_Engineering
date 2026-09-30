/* llm.js — interactive explainers for GPT internals and modern LLM architecture. */
(function () {
  "use strict";
  var V = window.Viz;

  // ================================================================ shared helpers
  function svgPt(svg, ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  function onDrag(el, move) {
    el.addEventListener("pointerdown", function (e) {
      e.preventDefault();
      var mv = function (ev) { move(ev); };
      var up = function () { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
    });
  }
  function textArea(K, value, rows) {
    var ta = K.h("textarea", {
      rows: rows || 5, spellcheck: "false",
      style: {
        width: "100%", boxSizing: "border-box", fontFamily: "var(--mono)", fontSize: "12.5px", lineHeight: "1.45",
        background: "var(--bg2)", color: "var(--fg)", border: "1px solid var(--line2)", borderRadius: "8px",
        padding: "8px 10px", resize: "vertical", display: "block", margin: "4px 0 8px",
      },
    });
    ta.value = value;
    return ta;
  }
  function statGrid(items) {
    return '<div class="stat-grid" style="margin-bottom:8px">' + items.map(function (it) {
      return '<div class="stat"><small>' + it[0] + "</small><b" + (it[2] ? ' style="color:var(' + it[2] + ')"' : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  function codeBlock(src) {
    var esc = src.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    return '<pre style="font-family:var(--mono);font-size:12px;background:var(--bg3);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0;white-space:pre-wrap;overflow-x:auto">' + esc + "</pre>";
  }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function debounce(fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function newSvg(K, W, H, maxW) {
    return K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg", preserveAspectRatio: "xMidYMid meet", style: maxW ? { maxWidth: maxW + "px", margin: "0 auto" } : null });
  }
  // Play/pause helper: calls step() every ms until step returns false or the card is detached.
  function player(c, K, step, ms, label) {
    var timer = null;
    var btn = K.button("▶ " + (label || "Play"), function () { if (timer) stop(); else start(); }, "primary");
    function start() {
      timer = setInterval(function () {
        if (!c.root.isConnected) { stop(); return; }
        if (step() === false) stop();
      }, ms);
      btn.textContent = "⏸ Pause";
    }
    function stop() { clearInterval(timer); timer = null; btn.textContent = "▶ " + (label || "Play"); }
    return { btn: btn, stop: stop, start: start, running: function () { return !!timer; } };
  }
  function arrow(K, g, x1, y1, x2, y2, sCls, fCls, w) {
    var ang = Math.atan2(y2 - y1, x2 - x1), L = 9, a = 0.42;
    var bx = x2 - Math.cos(ang) * L, by = y2 - Math.sin(ang) * L;
    g.appendChild(K.s("line", { x1: x1, y1: y1, x2: bx, y2: by, class: sCls, "stroke-width": w || 2.2 }));
    var p1 = [x2 - Math.cos(ang - a) * L, y2 - Math.sin(ang - a) * L], p2 = [x2 - Math.cos(ang + a) * L, y2 - Math.sin(ang + a) * L];
    g.appendChild(K.s("polygon", { points: x2 + "," + y2 + " " + p1.join(",") + " " + p2.join(","), class: fCls }));
  }
  function fmtTime(sec) {
    if (!isFinite(sec)) return "∞";
    if (sec < 60) return sec.toFixed(1) + " s";
    if (sec < 3600) return (sec / 60).toFixed(1) + " min";
    if (sec < 86400 * 2) return (sec / 3600).toFixed(1) + " h";
    if (sec < 86400 * 365 * 2) return (sec / 86400).toFixed(1) + " days";
    return (sec / 86400 / 365).toFixed(1) + " years";
  }
  function fmtGB(b) { var g = b / 1e9; return g >= 100 ? g.toFixed(0) + " GB" : g >= 10 ? g.toFixed(1) + " GB" : g >= 1 ? g.toFixed(2) + " GB" : (b / 1e6).toFixed(g >= 0.1 ? 0 : 1) + " MB"; }
  function fmtP(n) { return n >= 1e12 ? (n / 1e12).toFixed(1) + "T" : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e11 ? 0 : 2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(Math.round(n)); }
  function showCh(ch) { return ch === "\n" ? "↵" : ch === " " ? "␣" : ch === "\t" ? "⇥" : ch; }
  function tokStyle(i) {
    var c = "var(--c" + (1 + (i % 6)) + ")";
    return "background:color-mix(in srgb," + c + " 17%, var(--bg2));border-color:color-mix(in srgb," + c + " 55%, transparent)";
  }

  // Shared model configs (public configs from Hugging Face).
  var MODELS = {
    "gpt2": { name: "GPT-2 small (124M)", style: "gpt2", L: 12, d: 768, h: 12, kv: 12, hd: 64, ffn: 3072, V: 50257, ctx: 1024, tied: true },
    "llama32-1b": { name: "Llama-3.2-1B", style: "llama", L: 16, d: 2048, h: 32, kv: 8, hd: 64, ffn: 8192, V: 128256, tied: true },
    "qwen25-7b": { name: "Qwen2.5-7B", style: "qwen", L: 28, d: 3584, h: 28, kv: 4, hd: 128, ffn: 18944, V: 152064, tied: false },
    "mistral-7b": { name: "Mistral-7B", style: "llama", L: 32, d: 4096, h: 32, kv: 8, hd: 128, ffn: 14336, V: 32000, tied: false },
    "llama31-8b": { name: "Llama-3.1-8B", style: "llama", L: 32, d: 4096, h: 32, kv: 8, hd: 128, ffn: 14336, V: 128256, tied: false },
    "llama31-70b": { name: "Llama-3.1-70B", style: "llama", L: 80, d: 8192, h: 64, kv: 8, hd: 128, ffn: 28672, V: 128256, tied: false },
  };
  function countParams(m) {
    var hd = m.hd || m.d / m.h;
    var emb = m.V * m.d + (m.style === "gpt2" ? (m.ctx || 1024) * m.d : 0);
    var attn = m.d * m.h * hd + 2 * m.d * m.kv * hd + m.h * hd * m.d;
    if (m.style === "gpt2") attn += m.h * hd + 2 * m.kv * hd + m.d;
    if (m.style === "qwen") attn += m.h * hd + 2 * m.kv * hd;
    var mlp = m.style === "gpt2" ? 2 * m.d * m.ffn + m.ffn + m.d : 3 * m.d * m.ffn;
    var norm = m.style === "gpt2" ? 4 * m.d : 2 * m.d;
    var head = m.tied ? 0 : m.V * m.d;
    var fnorm = m.style === "gpt2" ? 2 * m.d : m.d;
    var r = { emb: emb, attn: attn * m.L, mlp: mlp * m.L, norm: norm * m.L + fnorm, head: head, perLayer: attn + mlp + norm, attnL: attn, mlpL: mlp };
    r.total = r.emb + r.attn + r.mlp + r.norm + r.head;
    return r;
  }

  // ================================================================ bigram-live
  var SHAKES = [
    "HAMLET:",
    "To be, or not to be, that is the question:",
    "Whether 'tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune,",
    "Or to take arms against a sea of troubles",
    "And by opposing end them. To die, to sleep,",
    "No more; and by a sleep to say we end",
    "The heart-ache and the thousand natural shocks",
    "That flesh is heir to: 'tis a consummation",
    "Devoutly to be wish'd.",
    "",
    "MACBETH:",
    "To-morrow, and to-morrow, and to-morrow,",
    "Creeps in this petty pace from day to day,",
    "To the last syllable of recorded time;",
    "And all our yesterdays have lighted fools",
    "The way to dusty death. Out, out, brief candle!",
    "Life's but a walking shadow, a poor player,",
    "That struts and frets his hour upon the stage,",
    "And then is heard no more. It is a tale",
    "Told by an idiot, full of sound and fury,",
    "Signifying nothing.",
    "",
    "SONNET:",
    "Shall I compare thee to a summer's day?",
    "Thou art more lovely and more temperate:",
    "Rough winds do shake the darling buds of May,",
    "And summer's lease hath all too short a date;",
    "Sometime too hot the eye of heaven shines,",
    "And often is his gold complexion dimm'd;",
    "And every fair from fair sometime declines,",
    "By chance or nature's changing course untrimm'd;",
    "But thy eternal summer shall not fade,",
    "Nor lose possession of that fair thou ow'st;",
    "Nor shall death brag thou wander'st in his shade,",
    "When in eternal lines to time thou grow'st:",
    "So long as men can breathe or eyes can see,",
    "So long lives this, and this gives life to thee.",
    "",
  ].join("\n");

  V.register("bigram-live", {
    title: "A bigram language model, trained live on your text",
    desc: "Edit the text: the “model” just counts which character follows which. The count table <i>is</i> the model — normalize each row to get P(next | current), then sample from it. Hover the heatmap.",
    render: function (c, p, K) {
      var lower = p.lowercase !== undefined ? !!p.lowercase : false, smooth = p.smoothing !== undefined ? +p.smoothing : 0.1;
      var topN = p.topN || 27, seed = p.seed || 42, len = p.length || 260, showProb = false, hover = null, M = null;
      var ta = textArea(K, p.text || SHAKES, 6);
      var heatWrap = K.h("div", { style: { flex: "1.25", minWidth: "280px" } });
      var sideWrap = K.h("div", { style: { minWidth: "240px" } });
      c.stage.appendChild(ta);
      c.stage.appendChild(K.h("div", { class: "viz-row" }, heatWrap, sideWrap));
      ta.addEventListener("input", debounce(function () { train(); draw(); }, 250));

      c.controls.appendChild(K.toggle({ label: "lowercase", value: lower, onChange: function (v) { lower = v; train(); draw(); } }));
      c.controls.appendChild(K.slider({ label: "smoothing k", min: 0, max: 3, step: 0.1, value: smooth, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { smooth = v; train(); draw(); } }));
      c.controls.appendChild(K.slider({ label: "heatmap chars", min: 8, max: 40, value: topN, onInput: function (v) { topN = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "cells show", options: [{ value: "count", label: "counts N[a,b]" }, { value: "prob", label: "row probs P(b|a)" }], value: "count", onChange: function (v) { showProb = v === "prob"; draw(); } }));
      var seedS = K.slider({ label: "sample seed", min: 1, max: 999, value: seed, onInput: function (v) { seed = v; drawSample(); } });
      c.controls.appendChild(seedS);
      c.controls.appendChild(K.slider({ label: "sample length", min: 40, max: 800, step: 20, value: len, onInput: function (v) { len = v; drawSample(); } }));
      c.controls.appendChild(K.button("🎲 New sample", function () { seed = 1 + Math.floor(Math.random() * 998); seedS.set(seed); drawSample(); }));

      function train() {
        var text = ta.value.slice(0, 20000);
        if (lower) text = text.toLowerCase();
        var freq = {};
        for (var i = 0; i < text.length; i++) freq[text[i]] = (freq[text[i]] || 0) + 1;
        var vocab = Object.keys(freq).sort(function (a, b) { return a.charCodeAt(0) - b.charCodeAt(0); });
        var ix = {}; vocab.forEach(function (ch, i) { ix[ch] = i; });
        var n = vocab.length, N = [];
        for (var a = 0; a < n; a++) { N.push(new Array(n).fill(0)); }
        for (var j = 0; j + 1 < text.length; j++) N[ix[text[j]]][ix[text[j + 1]]]++;
        var rows = N.map(sum);
        var P = N.map(function (row, a) { return row.map(function (v) { return rows[a] + smooth * n > 0 ? (v + smooth) / (rows[a] + smooth * n) : 1 / n; }); });
        var nll = 0, nb = Math.max(0, text.length - 1);
        for (var t = 0; t + 1 < text.length; t++) nll -= Math.log(P[ix[text[t]]][ix[text[t + 1]]]);
        nll = nb ? nll / nb : 0;
        var uni = 0; vocab.forEach(function (ch) { var q = freq[ch] / text.length; uni -= q * Math.log(q); });
        M = { text: text, vocab: vocab, ix: ix, N: N, P: P, rows: rows, freq: freq, nll: nll, nb: nb, uni: uni };
      }

      function drawSample() {
        K.clear(sideWrap);
        if (!M || M.vocab.length < 2) { sideWrap.appendChild(K.h("div", { class: "muted", text: "Type at least a few characters of text." })); return; }
        var r = K.rng(seed), cur = M.ix["\n"] !== undefined ? M.ix["\n"] : M.ix[M.text[0]], out = "";
        for (var s = 0; s < len; s++) {
          var row = M.P[cur], u = r(), acc = 0, nx = row.length - 1;
          for (var i = 0; i < row.length; i++) { acc += row[i]; if (u <= acc) { nx = i; break; } }
          out += M.vocab[nx]; cur = nx;
        }
        sideWrap.appendChild(K.h("div", { class: "muted", style: { fontSize: "12.5px", margin: "2px 0 4px" }, text: "Sampled from the model (seed " + seed + "):" }));
        sideWrap.appendChild(K.h("pre", { style: { fontFamily: "var(--mono)", fontSize: "12px", background: "var(--bg3)", border: "1px solid var(--line)", borderRadius: "8px", padding: "8px 10px", margin: 0, whiteSpace: "pre-wrap", maxHeight: "360px", overflow: "auto" }, text: out }));
        sideWrap.appendChild(K.h("div", { class: "muted", style: { fontSize: "12px", marginTop: "4px" }, text: "Locally plausible letter pairs, globally nonsense: one character of context is all it has." }));
      }

      function draw() {
        K.clear(heatWrap);
        if (!M || M.vocab.length < 2) { drawSample(); c.readout.innerHTML = "Need more text."; return; }
        var chars = M.vocab.slice().sort(function (a, b) { return M.freq[b] - M.freq[a]; }).slice(0, topN);
        chars.sort(function (a, b) { return a.charCodeAt(0) - b.charCodeAt(0); });
        var n = chars.length, cs = Math.min(16, 400 / n), L = 24, T = 36;
        var W = L + n * cs + 6, H = T + n * cs + 6;
        var svg = newSvg(K, W, H, 470);
        var mx = 0;
        chars.forEach(function (a) { chars.forEach(function (b) { var v = showProb ? M.P[M.ix[a]][M.ix[b]] : M.N[M.ix[a]][M.ix[b]]; if (v > mx) mx = v; }); });
        var fs = Math.max(7, Math.min(11, cs * 0.72));
        chars.forEach(function (ch, i) {
          svg.appendChild(K.s("text", { x: L - 4, y: T + i * cs + cs * 0.72, "text-anchor": "end", class: "mono", style: { fontSize: fs + "px" }, text: showCh(ch) }));
          svg.appendChild(K.s("text", { x: L + i * cs + cs / 2, y: T - 6, "text-anchor": "middle", class: "mono", style: { fontSize: fs + "px" }, text: showCh(ch) }));
        });
        var hl = K.s("rect", { x: 0, y: 0, width: 0, height: 0, fill: "none", class: "sfg", "stroke-width": 1.5, "pointer-events": "none" });
        var rowHl = K.s("rect", { x: L, y: 0, width: n * cs, height: cs, fill: "none", class: "s2", "stroke-width": 1, "pointer-events": "none", opacity: 0 });
        chars.forEach(function (a, i) {
          chars.forEach(function (b, j) {
            var v = showProb ? M.P[M.ix[a]][M.ix[b]] : M.N[M.ix[a]][M.ix[b]];
            var op = mx > 0 ? Math.sqrt(v / mx) : 0;
            var rc = K.s("rect", { x: L + j * cs, y: T + i * cs, width: cs - 0.6, height: cs - 0.6, class: v > 0 || showProb ? "f1" : "fbg3", "fill-opacity": v > 0 || showProb ? Math.max(0.04, op) : 1 });
            rc.addEventListener("mouseenter", function () {
              hover = [a, b];
              hl.setAttribute("x", L + j * cs); hl.setAttribute("y", T + i * cs); hl.setAttribute("width", cs); hl.setAttribute("height", cs);
              rowHl.setAttribute("y", T + i * cs); rowHl.setAttribute("opacity", 1);
              readout();
            });
            svg.appendChild(rc);
          });
        });
        svg.appendChild(K.s("text", { x: 2, y: 11, class: "lbl-sm", text: "row = current, col = next" }));
        svg.appendChild(rowHl); svg.appendChild(hl);
        heatWrap.appendChild(svg);
        drawSample();
        readout();
      }

      function readout() {
        var V_ = M.vocab.length;
        var html = statGrid([
          ["characters in text", K.fmtNum(M.text.length, 0)],
          ["vocab size V", V_],
          ["avg NLL (bigram)", M.nll.toFixed(3) + " nats", "--c3"],
          ["perplexity e^NLL", Math.exp(M.nll).toFixed(1)],
          ["unigram NLL (no context)", M.uni.toFixed(3)],
          ["uniform guess ln V", Math.log(V_).toFixed(3)],
        ]);
        if (hover && (M.ix[hover[0]] === undefined || M.ix[hover[1]] === undefined)) hover = null;
        if (hover) {
          var a = M.ix[hover[0]], b = M.ix[hover[1]], row = M.N[a];
          var top = row.map(function (v, i) { return [v, i]; }).sort(function (x, y) { return y[0] - x[0]; }).slice(0, 5);
          html += "After <b>“" + esc(showCh(hover[0])) + "”</b> (seen " + M.rows[a] + "×), <b>“" + esc(showCh(hover[1])) + "”</b> followed <b>" + row[b] + "</b> times → " +
            K.tex("P = \\frac{N+k}{\\sum N + kV} = \\frac{" + row[b] + "+" + smooth + "}{" + M.rows[a] + "+" + smooth + "\\cdot" + V_ + "} = " + M.P[a][b].toFixed(3)) +
            ". Most likely next: " + top.map(function (t) { return "“" + esc(showCh(M.vocab[t[1]])) + "” " + (M.P[a][t[1]] * 100).toFixed(0) + "%"; }).join(", ") + ".<br>";
        } else html += "<span class='muted'>Hover a cell to see the count and the probability it becomes.</span><br>";
        html += "Loss " + K.tex("\\text{NLL} = -\\frac{1}{n}\\sum_{i} \\ln P(c_{i+1}\\mid c_i)") + " over all " + M.nb + " pairs in the text. " +
          "Bigram &lt; unigram &lt; ln V: each bit of context lowers the loss. A trained GPT on Shakespeare reaches ≈1.5 because it sees hundreds of characters of context. " +
          "Smoothing k = 0 gives unseen pairs probability 0 (→ infinite loss on new text); k &gt; 0 is “fake counts”.";
        c.readout.innerHTML = html;
      }
      train(); draw();
    },
  });

  // ================================================================ tokenizer-bpe
  var BPE_TEXT = "the cat sat on the mat. the other cat sat on the hat.\nlow lower lowest; new newer newest; wide wider widest.\nthe mother gathered the feathers together in the weather.";

  V.register("tokenizer-bpe", {
    title: "Byte-Pair Encoding: learning a tokenizer by merging the most frequent pair",
    desc: "Start from single characters. Repeatedly find the most common adjacent pair and fuse it into a new token. Slide the merge count: vocabulary grows by one each step, the sequence gets shorter.",
    render: function (c, p, K) {
      var split = p.split !== undefined ? !!p.split : true, maxM = p.maxMerges || 150, m = p.merges !== undefined ? p.merges : 12, R = null, hoverTok = null;
      var ta = textArea(K, p.text || BPE_TEXT, 4);
      var tokWrap = K.h("div", { style: { lineHeight: "2", padding: "4px 0", maxHeight: "210px", overflow: "auto" } });
      var chartWrap = K.h("div");
      c.stage.appendChild(ta); c.stage.appendChild(tokWrap); c.stage.appendChild(chartWrap);
      ta.addEventListener("input", debounce(function () { train(); draw(); }, 250));

      var ms = K.slider({ label: "merges", min: 0, max: maxM, value: m, onInput: function (v) { m = v; draw(); } });
      c.controls.appendChild(ms);
      c.controls.appendChild(K.button("◀", function () { if (m > 0) { m--; ms.set(m); draw(); } }));
      c.controls.appendChild(K.button("Merge ▶", function () { if (m < R.merges.length) { m++; ms.set(m); draw(); } }));
      var pl = player(c, K, function () { if (m >= R.merges.length) return false; m++; ms.set(m); draw(); }, 350);
      c.controls.appendChild(pl.btn);
      c.controls.appendChild(K.button("Reset", function () { pl.stop(); m = 0; ms.set(0); draw(); }));
      c.controls.appendChild(K.toggle({ label: "pre-split into words (GPT-2 style)", value: split, onChange: function (v) { split = v; train(); draw(); } }));

      function pretok(text) {
        if (!split) return [text];
        return text.match(/ ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g) || [];
      }
      function train() {
        var text = ta.value.slice(0, 4000), chunks = pretok(text);
        var wmap = {}, words = [];
        chunks.forEach(function (ch) { if (!wmap[ch]) { wmap[ch] = { syms: Array.from(ch), n: 0 }; words.push(wmap[ch]); } wmap[ch].n++; });
        var baseSet = {}; Array.from(text).forEach(function (ch) { baseSet[ch] = 1; });
        var base = Object.keys(baseSet).sort();
        var total = function () { return words.reduce(function (a, w) { return a + w.syms.length * w.n; }, 0); };
        var merges = [], lengths = [total()];
        for (var it = 0; it < maxM; it++) {
          var counts = {}, order = [];
          words.forEach(function (w) {
            for (var i = 0; i + 1 < w.syms.length; i++) {
              var key = w.syms[i] + "\u0000" + w.syms[i + 1];
              if (counts[key] === undefined) { counts[key] = 0; order.push(key); }
              counts[key] += w.n;
            }
          });
          var best = null, bc = 1;
          order.forEach(function (k) { if (counts[k] > bc) { bc = counts[k]; best = k; } });
          if (!best) break;
          var pr = best.split("\u0000"), tok = pr[0] + pr[1];
          words.forEach(function (w) { w.syms = mergeSyms(w.syms, pr[0], pr[1]); });
          merges.push({ a: pr[0], b: pr[1], tok: tok, count: bc });
          lengths.push(total());
        }
        R = { text: text, chunks: chunks, base: base, merges: merges, lengths: lengths };
        ms.input.max = merges.length;
        if (m > merges.length) { m = merges.length; }
        ms.set(m);
      }
      function mergeSyms(s, a, b) {
        var out = [];
        for (var i = 0; i < s.length; i++) {
          if (i + 1 < s.length && s[i] === a && s[i + 1] === b) { out.push(a + b); i++; } else out.push(s[i]);
        }
        return out;
      }
      function encode(mm) {
        var memo = {}, toks = [];
        R.chunks.forEach(function (ch) {
          if (!memo[ch]) {
            var s = Array.from(ch);
            for (var j = 0; j < mm; j++) s = mergeSyms(s, R.merges[j].a, R.merges[j].b);
            memo[ch] = s;
          }
          memo[ch].forEach(function (t) { toks.push(t); });
        });
        return toks;
      }
      function draw() {
        m = Math.max(0, Math.min(m, R.merges.length));
        var toks = encode(m), id = {};
        R.base.forEach(function (ch, i) { id[ch] = i; });
        for (var j = 0; j < m; j++) id[R.merges[j].tok] = R.base.length + j;
        var last = m > 0 ? R.merges[m - 1].tok : null;
        K.clear(tokWrap);
        toks.forEach(function (t) {
          var disp = t.replace(/ /g, "·").replace(/\n/g, "↵");
          var span = K.h("span", { class: "tok" + (t === last ? " new" : ""), text: disp, title: "token id " + id[t] });
          if (t !== last) span.setAttribute("style", tokStyle(id[t] * 7));
          if (hoverTok === t) span.style.outline = "2px solid var(--fg)";
          span.addEventListener("mouseenter", function () { hoverTok = t; readout(toks, id); });
          tokWrap.appendChild(span);
          if (t.indexOf("\n") >= 0) tokWrap.appendChild(K.h("br"));
        });
        // trade-off chart
        var n = R.merges.length, base = R.base.length;
        var ch = K.chart({ w: 660, h: 170, x: [base, base + Math.max(1, n)], y: [0, R.lengths[0] * 1.05], xLabel: "vocabulary size (base chars + merges)", yLabel: "# tokens", pad: { l: 56, b: 38, t: 10 } });
        ch.g.appendChild(K.s("path", { d: K.path(R.lengths.map(function (L, i) { return [base + i, L]; }), ch.sx, ch.sy), class: "ln s1" }));
        ch.g.appendChild(K.s("circle", { cx: ch.sx(base + m), cy: ch.sy(R.lengths[m]), r: 6, class: "f2" }));
        K.clear(chartWrap).appendChild(ch.svg);
        readout(toks, id);
      }
      function readout(toks, id) {
        var nChars = Array.from(R.text).length;
        var html = statGrid([
          ["base vocab (chars)", R.base.length],
          ["merges learned", m + " / " + R.merges.length],
          ["vocab size", R.base.length + m, "--c1"],
          ["sequence length", toks.length + " tokens", "--c2"],
          ["chars per token", (nChars / Math.max(1, toks.length)).toFixed(2)],
        ]);
        if (m > 0) {
          var mg = R.merges[m - 1];
          html += "Merge #" + m + ": most frequent pair <b>(“" + esc(mg.a.replace(/ /g, "·")) + "”, “" + esc(mg.b.replace(/ /g, "·")) + "”)</b> seen <b>" + mg.count + "×</b> → new token <b style='color:var(--green)'>“" + esc(mg.tok.replace(/ /g, "·")) + "”</b> (id " + (R.base.length + m - 1) + "). ";
          var recent = R.merges.slice(Math.max(0, m - 8), m).map(function (x) { return "<code>" + esc(x.tok.replace(/ /g, "·")) + "</code>"; });
          html += "Recent merges: " + recent.join(" ") + "<br>";
        } else html += "No merges yet: every character is its own token.<br>";
        if (hoverTok) html += "Hovered token <code>" + esc(hoverTok.replace(/ /g, "·").replace(/\n/g, "↵")) + "</code> → id <b>" + id[hoverTok] + "</b>. ";
        html += "<span class='muted'>Trade-off: bigger vocab ⇒ shorter sequences (cheaper attention, more text per context window) but a bigger embedding table and softmax. GPT-2 uses 50,257 tokens (byte-level BPE, 50k merges); Llama-3 uses 128,256. “·” marks a space — GPT-2 attaches the space to the <i>next</i> word, which is why “ the” and “the” are different tokens.</span>";
        c.readout.innerHTML = html;
      }
      train(); draw();
    },
  });

  // ================================================================ embeddings-2d
  var EMB = {
    man: [1.0, 1.0, 1], woman: [2.2, 1.05, 1], boy: [0.9, 0.05, 1], girl: [2.15, -0.05, 1],
    king: [1.05, 3.0, 1], queen: [2.3, 3.1, 1], prince: [0.95, 2.1, 1], princess: [2.25, 2.0, 1],
    dog: [-3.0, -1.0, 2], cat: [-1.8, -1.2, 2], puppy: [-3.25, -2.1, 2], kitten: [-2.0, -2.35, 2], wolf: [-3.7, -0.3, 2], lion: [-1.3, -0.35, 2],
    apple: [-3.0, 3.0, 3], banana: [-2.1, 3.6, 3], orange: [-3.7, 2.4, 3], grape: [-2.4, 2.45, 3],
    france: [3.0, -1.6, 4], paris: [3.8, -2.6, 4], japan: [4.2, -1.1, 4], tokyo: [5.05, -2.15, 4], germany: [3.55, -0.45, 4], berlin: [4.45, -1.5, 4],
  };
  var ANALOGIES = [["king", "man", "woman"], ["queen", "woman", "man"], ["prince", "boy", "girl"], ["puppy", "dog", "cat"], ["paris", "france", "japan"], ["berlin", "germany", "france"], ["kitten", "cat", "dog"]];

  V.register("embeddings-2d", {
    title: "Word embeddings: meaning as position, relationships as directions",
    desc: "Each word is a vector (here 2-D; real models use 768–8192 dims). Similar words sit close together, and consistent relationships become consistent <i>offsets</i>. Pick an analogy, drag words, hover to see neighbours.",
    render: function (c, p, K) {
      var pos = {}; function reset() { Object.keys(EMB).forEach(function (w) { pos[w] = [EMB[w][0], EMB[w][1]]; }); }
      reset();
      var words = Object.keys(EMB), A = "king", B = "man", C = "woman", metric = "euclid", hover = null;
      var selP = K.select({ label: "analogy", options: ANALOGIES.map(function (a, i) { return { value: i, label: a[0] + " − " + a[1] + " + " + a[2] }; }), value: 0, onChange: function (v) { var a = ANALOGIES[+v]; A = a[0]; B = a[1]; C = a[2]; sA.input.value = A; sB.input.value = B; sC.input.value = C; draw(); } });
      var sA = K.select({ label: "A", options: words, value: A, onChange: function (v) { A = v; draw(); } });
      var sB = K.select({ label: "− B", options: words, value: B, onChange: function (v) { B = v; draw(); } });
      var sC = K.select({ label: "+ C", options: words, value: C, onChange: function (v) { C = v; draw(); } });
      [selP, sA, sB, sC].forEach(function (x) { c.controls.appendChild(x); });
      c.controls.appendChild(K.select({ label: "similarity", options: [{ value: "euclid", label: "Euclidean distance" }, { value: "cos", label: "cosine similarity" }], value: metric, onChange: function (v) { metric = v; draw(); } }));
      c.controls.appendChild(K.button("Reset positions", function () { reset(); draw(); }));

      function sim(u, v) {
        if (metric === "cos") return (u[0] * v[0] + u[1] * v[1]) / (Math.hypot(u[0], u[1]) * Math.hypot(v[0], v[1]) || 1);
        return -Math.hypot(u[0] - v[0], u[1] - v[1]);
      }
      function nearest(vec, excl, n) {
        return words.filter(function (w) { return excl.indexOf(w) < 0; }).map(function (w) { return [w, sim(vec, pos[w])]; })
          .sort(function (a, b) { return b[1] - a[1]; }).slice(0, n);
      }
      function fmtSim(s) { return metric === "cos" ? "cos " + s.toFixed(3) : "dist " + (-s).toFixed(2); }
      function draw() {
        var ch = K.chart({ w: 680, h: 400, x: [-4.6, 5.9], y: [-3.2, 4.2], xLabel: "embedding dim 1", yLabel: "embedding dim 2", xN: 8 });
        var g = ch.g, sx = ch.sx, sy = ch.sy;
        var res = [pos[A][0] - pos[B][0] + pos[C][0], pos[A][1] - pos[B][1] + pos[C][1]];
        var nn = nearest(res, [A, B, C], 3);
        if (hover) {
          nearest(pos[hover], [hover], 3).forEach(function (q) {
            g.appendChild(K.s("line", { x1: sx(pos[hover][0]), y1: sy(pos[hover][1]), x2: sx(pos[q[0]][0]), y2: sy(pos[q[0]][1]), class: "ln-thin sfg dash", opacity: 0.6 }));
          });
        }
        if (metric === "cos") g.appendChild(K.s("line", { x1: sx(0), y1: sy(0), x2: sx(res[0]), y2: sy(res[1]), class: "ln-thin smuted dash" }));
        // offset B→A, then same offset applied at C
        arrow(K, g, sx(pos[B][0]), sy(pos[B][1]), sx(pos[A][0]), sy(pos[A][1]), "s2", "f2", 2);
        arrow(K, g, sx(pos[C][0]), sy(pos[C][1]), sx(res[0]), sy(res[1]), "s2", "f2", 2);
        g.appendChild(K.s("line", { x1: sx(res[0]), y1: sy(res[1]), x2: sx(pos[nn[0][0]][0]), y2: sy(pos[nn[0][0]][1]), class: "ln-thin s3", "stroke-width": 2 }));
        words.forEach(function (w) {
          var hl = w === A || w === B || w === C, isNN = w === nn[0][0];
          var dot = K.s("circle", { cx: sx(pos[w][0]), cy: sy(pos[w][1]), r: hl || isNN ? 7 : 5.5, class: "f" + EMB[w][2], opacity: hl || isNN ? 1 : 0.75, style: { cursor: "grab" } });
          if (isNN) g.appendChild(K.s("circle", { cx: sx(pos[w][0]), cy: sy(pos[w][1]), r: 11, fill: "none", class: "s3", "stroke-width": 2 }));
          g.appendChild(dot);
          g.appendChild(K.s("text", { x: sx(pos[w][0]) + 8, y: sy(pos[w][1]) - 6, class: hl || isNN ? "lbl" : "lbl-sm", text: w }));
          dot.addEventListener("mouseenter", function () { if (hover !== w) { hover = w; draw(); } });
          onDrag(dot, function (ev) {
            var q = svgPt(ch.svg, ev);
            pos[w] = [+K.clamp(sx.invert(q.x), -4.5, 5.8).toFixed(2), +K.clamp(sy.invert(q.y), -3.1, 4.1).toFixed(2)]; draw();
          });
        });
        g.appendChild(K.s("circle", { cx: sx(res[0]), cy: sy(res[1]), r: 6, fill: "none", class: "s2", "stroke-width": 2.5 }));
        g.appendChild(K.s("text", { x: sx(res[0]) + 9, y: sy(res[1]) + 14, class: "lbl-sm", text: "A − B + C" }));
        K.clear(c.stage).appendChild(ch.svg);
        var f = function (v) { return "[" + v[0].toFixed(2) + ", " + v[1].toFixed(2) + "]"; };
        c.readout.innerHTML =
          "<b>" + A + "</b> − <b>" + B + "</b> + <b>" + C + "</b> = " + f(pos[A]) + " − " + f(pos[B]) + " + " + f(pos[C]) + " = <b style='color:var(--c2)'>" + f(res) + "</b>" +
          " → nearest word (excluding the inputs): <b style='color:var(--c3)'>" + nn[0][0] + "</b> (" + fmtSim(nn[0][1]) + "), then " + nn.slice(1).map(function (q) { return q[0] + " (" + fmtSim(q[1]) + ")"; }).join(", ") + ".<br>" +
          (hover ? "Neighbours of <b>" + hover + "</b>: " + nearest(pos[hover], [hover], 3).map(function (q) { return q[0] + " (" + fmtSim(q[1]) + ")"; }).join(", ") + ". " : "") +
          "Cosine similarity " + K.tex("\\cos\\theta = \\frac{u\\cdot v}{\\|u\\|\\|v\\|}") + " compares <i>direction</i> only; with high-dimensional embeddings it's the standard measure. " +
          "<span class='muted'>In a GPT the embedding table is just a learned (V × C) matrix — row i is token i's vector. Nobody designs these directions; they emerge because they make next-token prediction easier. Drag “woman” and watch the analogy break.</span>";
      }
      draw();
    },
  });

  // ================================================================ causal-average
  V.register("causal-average", {
    title: "The matmul trick: averaging the past with a lower-triangular matrix",
    desc: "Each token should only see tokens before it. Multiply the sequence x by a lower-triangular weight matrix and every row becomes a weighted average of the past. Replace uniform weights with data-dependent ones and you have self-attention. Hover an output row.",
    render: function (c, p, K) {
      var T = p.T || 8, mode = p.mode || "tril", causal = true, strength = 1.5, seed = 1337, hov = null;
      c.controls.appendChild(K.select({ label: "weights", value: mode, options: [
        { value: "tril", label: "v2: tril ÷ row-sum (uniform average)" },
        { value: "softmax0", label: "v3: softmax(masked zeros)" },
        { value: "attn", label: "v4: softmax(masked q·k) = self-attention" },
      ], onChange: function (v) { mode = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "T (tokens)", min: 2, max: 8, value: T, onInput: function (v) { T = v; hov = null; draw(); } }));
      c.controls.appendChild(K.slider({ label: "affinity strength", min: 0, max: 3, step: 0.1, value: strength, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { strength = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "causal mask", value: causal, onChange: function (v) { causal = v; draw(); } }));
      c.controls.appendChild(K.button("New data", function () { seed++; draw(); }));

      function draw() {
        var r = K.rng(seed), x = [], aff = [];
        for (var i = 0; i < 8; i++) x.push([K.randn(r), K.randn(r)]);
        for (var i2 = 0; i2 < 8; i2++) { aff.push([]); for (var j2 = 0; j2 < 8; j2++) aff[i2].push(K.randn(r)); }
        x = x.slice(0, T);
        var scores = [], wei = [];
        for (var a = 0; a < T; a++) {
          scores.push([]);
          for (var b = 0; b < T; b++) {
            var masked = causal && b > a;
            var sc = mode === "tril" ? (masked ? 0 : 1) : masked ? -Infinity : mode === "attn" ? strength * aff[a][b] : 0;
            scores[a].push(sc);
          }
          if (mode === "tril") { var rs = sum(scores[a]); wei.push(scores[a].map(function (v) { return v / rs; })); }
          else wei.push(K.softmax(scores[a]));
        }
        var out = wei.map(function (row) { return [0, 1].map(function (k) { var s = 0; for (var j = 0; j < T; j++) s += row[j] * x[j][k]; return s; }); });

        var cs = 26, top = 34, W = 700, H = top + 8 * cs + 150;
        var svg = newSvg(K, W, H);
        var x0 = 20, x1 = x0 + T * cs + 60, x2 = x1 + T * cs + 36, x3 = x2 + 2 * cs + 36;
        function cellsM(M, ox, oy, cols, kind, label) {
          svg.appendChild(K.s("text", { x: ox, y: oy - 10, class: "lbl-sm", text: label }));
          for (var i = 0; i < M.length; i++) for (var j = 0; j < cols; j++) {
            var v = M[i][j], fin = isFinite(v), rowHL = hov !== null && (kind === "x" ? false : i === hov), colHL = hov !== null && kind === "x" && wei[hov][i] > 0.001;
            var cls = !fin ? "fbg3" : kind === "w" ? "f1" : v >= 0 ? "f1" : "f2";
            var op = !fin ? 1 : kind === "w" ? Math.max(0.05, Math.min(1, v)) : Math.min(1, Math.abs(v) / 2.2) * 0.85 + 0.05;
            if (kind === "s" && mode !== "tril") { cls = fin ? (v >= 0 ? "f1" : "f2") : "fbg3"; op = fin ? Math.min(1, Math.abs(v) / 3) * 0.8 + 0.08 : 1; }
            if (kind === "x" && hov !== null) op *= colHL ? 1 : 0.35;
            svg.appendChild(K.s("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 2, height: cs - 2, rx: 3, class: cls, "fill-opacity": op }));
            if (rowHL || colHL) svg.appendChild(K.s("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 2, height: cs - 2, rx: 3, fill: "none", class: "sfg", "stroke-width": 1.2 }));
            var txt = !fin ? "−∞" : kind === "w" ? (v === 0 ? "0" : v.toFixed(2).replace(/^0/, "")) : v.toFixed(1);
            svg.appendChild(K.s("text", { x: ox + j * cs + cs / 2 - 1, y: oy + i * cs + cs / 2 + 3, "text-anchor": "middle", class: "mono" + ((kind === "w" && op > 0.6) ? " t-white" : ""), style: { fontSize: "8.5px" }, text: txt }));
            if (kind === "o") (function (i) {
              var hit = K.s("rect", { x: ox, y: oy + i * cs, width: cols * cs, height: cs, fill: "transparent", style: { cursor: "pointer" } });
              hit.addEventListener("mouseenter", function () { if (hov !== i) { hov = i; draw(); } });
              svg.appendChild(hit);
            })(i);
          }
        }
        var sLabel = mode === "tril" ? "tril(ones(T,T))" : mode === "softmax0" ? "zeros, masked −∞" : "q·kᵀ affinities, masked";
        cellsM(scores, x0, top, T, "s", sLabel);
        svg.appendChild(K.s("text", { x: x0 + T * cs + 30, y: top + T * cs / 2, "text-anchor": "middle", class: "lbl-sm", text: mode === "tril" ? "÷ row sum" : "softmax" }));
        arrow(K, svg, x0 + T * cs + 8, top + T * cs / 2 + 8, x1 - 8, top + T * cs / 2 + 8, "smuted", "fmuted", 1.4);
        cellsM(wei, x1, top, T, "w", "wei (T×T)");
        svg.appendChild(K.s("text", { x: x1 + T * cs + 16, y: top + T * cs / 2 + 4, "text-anchor": "middle", class: "lbl", text: "@" }));
        cellsM(x, x2, top, 2, "x", "x (T×C)");
        svg.appendChild(K.s("text", { x: x2 + 2 * cs + 16, y: top + T * cs / 2 + 4, "text-anchor": "middle", class: "lbl", text: "=" }));
        cellsM(out, x3, top, 2, "o", "out (T×C)");
        for (var t = 0; t < T; t++) svg.appendChild(K.s("text", { x: x3 + 2 * cs + 6, y: top + t * cs + cs / 2 + 3, class: "lbl-sm", text: "t=" + t }));
        // line chart channel 0: x vs out
        var cy0 = top + 8 * cs + 22, chH = 100, cx0 = 60, cxW = 600;
        var sxL = K.scale(0, 7, cx0, cx0 + cxW), syL = K.scale(-2.5, 2.5, cy0 + chH, cy0);
        svg.appendChild(K.s("line", { x1: cx0, x2: cx0 + cxW, y1: syL(0), y2: syL(0), class: "axis" }));
        svg.appendChild(K.s("text", { x: 4, y: cy0 + 12, class: "lbl-sm", text: "channel 0" }));
        var px = x.map(function (v, i) { return [i, K.clamp(v[0], -2.5, 2.5)]; }), po = out.map(function (v, i) { return [i, K.clamp(v[0], -2.5, 2.5)]; });
        svg.appendChild(K.s("path", { d: K.path(px, sxL, syL), class: "ln-thin smuted dash" }));
        svg.appendChild(K.s("path", { d: K.path(po, sxL, syL), class: "ln s3" }));
        px.forEach(function (q) { svg.appendChild(K.s("circle", { cx: sxL(q[0]), cy: syL(q[1]), r: 3.5, class: "fmuted" })); });
        po.forEach(function (q, i) { svg.appendChild(K.s("circle", { cx: sxL(q[0]), cy: syL(q[1]), r: i === hov ? 6 : 4, class: "f3" })); svg.appendChild(K.s("text", { x: sxL(q[0]), y: cy0 + chH + 14, "text-anchor": "middle", class: "tick", text: "t=" + i })); });
        svg.appendChild(K.s("text", { x: cx0 + cxW - 150, y: cy0 + 4, class: "lbl-sm", text: "grey: x   green: out" }));
        K.clear(c.stage).appendChild(svg);

        var codes = {
          tril: "wei = torch.tril(torch.ones(T, T))\nwei = wei / wei.sum(1, keepdim=True)\nout = wei @ x          # (T,T) @ (B,T,C) -> (B,T,C)",
          softmax0: "tril = torch.tril(torch.ones(T, T))\nwei = torch.zeros((T, T))\nwei = wei.masked_fill(tril == 0, float('-inf'))\nwei = F.softmax(wei, dim=-1)   # e^-inf = 0 -> future gets weight 0\nout = wei @ x",
          attn: "q, k, v = query(x), key(x), value(x)          # (B,T,hs)\nwei = q @ k.transpose(-2, -1) * hs**-0.5       # (B,T,T) affinities\nwei = wei.masked_fill(tril == 0, float('-inf'))\nwei = F.softmax(wei, dim=-1)\nout = wei @ v                                   # here v = x for simplicity",
        };
        var html = codeBlock(codes[mode]);
        if (hov !== null) {
          html += "out[" + hov + "] = " + wei[hov].map(function (w, j) { return w > 0.0005 ? w.toFixed(2) + "·x[" + j + "]" : null; }).filter(Boolean).join(" + ") +
            " = [" + out[hov][0].toFixed(2) + ", " + out[hov][1].toFixed(2) + "]. ";
        } else html += "<span class='muted'>Hover an output row to see which inputs it mixes.</span> ";
        html += causal ? "Every row of wei sums to 1 and is zero above the diagonal, so token t only mixes x[0..t] — it can't peek at the future it's trying to predict. " :
          "<b style='color:var(--red)'>Mask off:</b> every token averages the <i>whole</i> sequence, including future tokens — fine for an encoder (BERT), but a GPT trained like this would cheat. ";
        if (mode === "attn") html += "Uniform weights treat all past tokens equally; data-dependent affinities let a token pick <i>which</i> past tokens matter.";
        c.readout.innerHTML = html;
      }
      draw();
    },
  });

  // ================================================================ attention
  V.register("attention", {
    title: "Self-attention, one query at a time",
    desc: "Each token emits a query q (“what am I looking for?”) and a key k (“what do I contain?”). Scores = q·k for every pair, scaled by 1/√d, masked so tokens can't see the future, then softmaxed into weights. Hover a token or a row.",
    render: function (c, p, K) {
      var ending = "tired", head = p.head !== undefined ? +p.head : 0, scaled = true, causal = p.causal !== undefined ? !!p.causal : true, seed = p.seed || 7, sel = 7, d = 16;
      var HEADS = ["Head 1 · coreference (who is “it”?)", "Head 2 · previous token", "Head 3 · syntax (verb ↔ object)"];
      c.controls.appendChild(K.select({ label: "sentence ends", options: [{ value: "tired", label: "…because it was too tired" }, { value: "wide", label: "…because it was too wide" }], value: ending, onChange: function (v) { ending = v; draw(); } }));
      var hs = K.select({ label: "head", options: HEADS.map(function (h, i) { return { value: i, label: h }; }), value: head, onChange: function (v) { head = +v; draw(); } });
      c.controls.appendChild(hs);
      c.controls.appendChild(K.toggle({ label: "scale by 1/√d", value: scaled, onChange: function (v) { scaled = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "causal mask", value: causal, onChange: function (v) { causal = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "seed", min: 1, max: 50, value: seed, onInput: function (v) { seed = v; draw(); } }));

      function target(h, toks) {
        var T = toks.length, A = [], I = function (w) { return toks.indexOf(w); };
        for (var i = 0; i < T; i++) { A.push(new Array(T).fill(0)); A[i][0] += 1.0; A[i][i] += 0.6; }
        function set(q, k, v) { if (q >= 0 && k >= 0) A[q][k] = v; }
        var an = 1, st = 5, it = 7, was = 8, too = 9, last = 10;
        if (h === 0) {
          set(an, 0, 1.6); set(an, an, 1.2); set(an, it, 1.8); set(an, last, ending === "tired" ? 1.6 : 0.3);
          set(st, 4, 1.4); set(st, st, 1.0); set(st, it, ending === "wide" ? 1.8 : 0.4);
          set(it, an, 3.2); set(it, st, 2.0); set(it, it, 0.6); set(it, last, 1.8);
          set(was, it, 2.6); set(was, an, 1.6);
          set(too, it, 1.4); set(too, was, 1.2);
          if (ending === "tired") { set(last, an, 3.3); set(last, it, 2.6); set(last, st, 0.6); }
          else { set(last, st, 3.3); set(last, it, 2.6); set(last, an, 0.6); }
        } else if (h === 1) {
          for (var q = 0; q < T; q++) { A[q] = new Array(T).fill(0); A[q][q] = 1.0; if (q > 0) A[q][q - 1] = 3.6; if (q + 1 < T) A[q][q + 1] = 0.8; }
          A[0][0] = 3;
        } else {
          set(2, an, 2.4); set(2, 3, 2.0);
          set(3, an, 2.2); set(3, st, 2.9); set(3, 2, 1.4);
          set(4, st, 2.4); set(4, 3, 1.5);
          set(st, 3, 3.0); set(st, 4, 1.2);
          set(6, 3, 2.2); set(6, last, 1.5);
          set(it, 6, 1.6); set(it, was, 1.4);
          set(was, it, 2.8); set(was, last, 1.8);
          set(too, last, 3.0); set(too, was, 1.6);
          set(last, too, 2.8); set(last, was, 2.4);
        }
        return A;
      }
      function orth(r) { // random orthogonal d×d via Gram-Schmidt
        var Q = [];
        for (var i = 0; i < d; i++) {
          var v = []; for (var j = 0; j < d; j++) v.push(K.randn(r));
          Q.forEach(function (u) { var dp = 0; for (var k = 0; k < d; k++) dp += u[k] * v[k]; for (var k2 = 0; k2 < d; k2++) v[k2] -= dp * u[k2]; });
          var n = Math.hypot.apply(null, v); Q.push(v.map(function (x) { return x / n; }));
        }
        return Q;
      }
      function qk(h, toks) {
        var A = target(h, toks), T = toks.length, r = K.rng(seed * 31 + h * 7 + 1), R = orth(r), s = 2;
        var Q = [], Kk = [];
        for (var i = 0; i < T; i++) {
          var kv = new Array(d).fill(0), qv = new Array(d).fill(0);
          kv[i] = s;
          for (var j = 0; j < T; j++) qv[j] = A[i][j] * Math.sqrt(d) / s;
          for (var z = 0; z < d; z++) { kv[z] += 0.12 * K.randn(r); qv[z] += 0.35 * K.randn(r); }
          var rot = function (v) { return R.map(function (row) { var t = 0; for (var k = 0; k < d; k++) t += row[k] * v[k]; return t; }); };
          Q.push(rot(qv)); Kk.push(rot(kv));
        }
        return { Q: Q, K: Kk };
      }
      function weights(h, toks) {
        var m = qk(h, toks), T = toks.length, raw = [], S = [], W = [];
        for (var i = 0; i < T; i++) {
          raw.push([]); S.push([]);
          for (var j = 0; j < T; j++) {
            var dp = 0; for (var z = 0; z < d; z++) dp += m.Q[i][z] * m.K[j][z];
            raw[i].push(dp);
            S[i].push(causal && j > i ? -Infinity : scaled ? dp / Math.sqrt(d) : dp);
          }
          W.push(K.softmax(S[i]));
        }
        return { raw: raw, S: S, W: W, Q: m.Q, K: m.K };
      }

      function draw() {
        var toks = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", ending];
        var T = toks.length, res = weights(head, toks);
        var Wd = 720, H = 432, svg = newSvg(K, Wd, H);
        // token strip + arcs
        var tx = function (i) { return 40 + i * ((Wd - 80) / (T - 1)); }, ty = 96;
        res.W[sel].forEach(function (w, j) {
          if (w < 0.01) return;
          var x1 = tx(sel), x2 = tx(j), hgt = 10 + Math.abs(x2 - x1) * 0.1;
          var dpath = j === sel ? "M" + (x1 - 8) + " " + (ty - 14) + " C" + (x1 - 16) + " " + (ty - 44) + " " + (x1 + 16) + " " + (ty - 44) + " " + (x1 + 8) + " " + (ty - 14) :
            "M" + x1 + " " + (ty - 14) + " Q" + (x1 + x2) / 2 + " " + (ty - 14 - hgt * 2) + " " + x2 + " " + (ty - 14);
          svg.appendChild(K.s("path", { d: dpath, class: "s1", fill: "none", "stroke-width": 1 + 9 * w, opacity: 0.25 + 0.75 * w, "stroke-linecap": "round" }));
        });
        toks.forEach(function (t, i) {
          var w = res.W[sel][i];
          var g = K.s("g", { style: { cursor: "pointer" } });
          g.appendChild(K.s("rect", { x: tx(i) - 29, y: ty - 13, width: 58, height: 24, rx: 6, class: i === sel ? "box-hl" : "box" }));
          if (i !== sel && w > 0.005) g.appendChild(K.s("rect", { x: tx(i) - 29, y: ty - 13, width: 58, height: 24, rx: 6, class: "f1", "fill-opacity": Math.min(0.85, w * 1.2) }));
          g.appendChild(K.s("text", { x: tx(i), y: ty + 3, "text-anchor": "middle", class: "lbl" + (i !== sel && w > 0.55 ? " t-white" : ""), text: t }));
          g.appendChild(K.s("text", { x: tx(i), y: ty + 24, "text-anchor": "middle", class: "lbl-sm mono", text: causal && i > sel ? "masked" : (w * 100).toFixed(0) + "%" }));
          g.addEventListener("mouseenter", function () { if (sel !== i) { sel = i; draw(); } });
          svg.appendChild(g);
        });
        // matrices
        var cs = 20, my = 204, mx1 = 78, mx2 = 382;
        function matrix(M, ox, isW, label) {
          svg.appendChild(K.s("text", { x: ox, y: my - 58, class: "lbl", text: label }));
          var mxAbs = 0; M.forEach(function (row) { row.forEach(function (v) { if (isFinite(v) && Math.abs(v) > mxAbs) mxAbs = Math.abs(v); }); });
          toks.forEach(function (t, j) { svg.appendChild(K.s("text", { x: ox + j * cs + cs / 2 + 3, y: my - 5, class: "lbl-sm", transform: "rotate(-55 " + (ox + j * cs + cs / 2 + 3) + " " + (my - 5) + ")", text: t })); });
          toks.forEach(function (t, i) {
            var lab = K.s("text", { x: ox - 4, y: my + i * cs + cs / 2 + 4, "text-anchor": "end", class: i === sel ? "lbl" : "lbl-sm", text: t });
            svg.appendChild(lab);
            for (var j = 0; j < T; j++) {
              var v = M[i][j], fin = isFinite(v);
              var cls = !fin ? "fbg3" : isW ? "f1" : v >= 0 ? "f1" : "f2";
              var op = !fin ? 1 : isW ? Math.max(0.03, v) : 0.06 + 0.9 * Math.abs(v) / (mxAbs || 1);
              svg.appendChild(K.s("rect", { x: ox + j * cs, y: my + i * cs, width: cs - 1.5, height: cs - 1.5, rx: 2, class: cls, "fill-opacity": op }));
              if (i === sel) {
                var txt = !fin ? "−∞" : isW ? Math.round(v * 100) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
                svg.appendChild(K.s("text", { x: ox + j * cs + cs / 2 - 0.5, y: my + i * cs + cs / 2 + 3, "text-anchor": "middle", class: "mono" + (fin && op > 0.6 ? " t-white" : ""), style: { fontSize: "8px" }, text: txt }));
              }
            }
            var hit = K.s("rect", { x: ox - 60, y: my + i * cs, width: T * cs + 60, height: cs, fill: "transparent" });
            (function (i) { hit.addEventListener("mouseenter", function () { if (sel !== i) { sel = i; draw(); } }); })(i);
            svg.appendChild(hit);
          });
          svg.appendChild(K.s("rect", { x: ox - 1, y: my + sel * cs - 1, width: T * cs + 1, height: cs + 1, fill: "none", class: "sfg", "stroke-width": 1.4, "pointer-events": "none" }));
        }
        matrix(res.S, mx1, false, scaled ? "scores = q·kᵀ / √d" : "scores = q·kᵀ (unscaled)");
        matrix(res.W, mx2, true, "weights = softmax(row)");
        // head thumbnails
        var thX = 624, ts = 7;
        svg.appendChild(K.s("text", { x: thX, y: my - 58, class: "lbl-sm", text: "all heads (click)" }));
        [0, 1, 2].forEach(function (h) {
          var w = weights(h, toks).W, oy = my - 40 + h * 86;
          var g = K.s("g", { style: { cursor: "pointer" } });
          g.appendChild(K.s("rect", { x: thX - 3, y: oy - 3, width: T * ts + 6, height: T * ts + 6, rx: 4, class: h === head ? "box-hl" : "box" }));
          for (var i = 0; i < T; i++) for (var j = 0; j < T; j++) g.appendChild(K.s("rect", { x: thX + j * ts, y: oy + i * ts, width: ts - 0.5, height: ts - 0.5, class: "f1", "fill-opacity": Math.max(0.03, w[i][j]) }));
          g.appendChild(K.s("text", { x: thX, y: oy + T * ts + 14, class: "lbl-sm", text: "head " + (h + 1) }));
          g.addEventListener("click", function () { head = h; hs.input.value = h; draw(); });
          svg.appendChild(g);
        });
        K.clear(c.stage).appendChild(svg);

        var row = res.W[sel], top = row.map(function (w, j) { return [w, j]; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, 3);
        var flat = []; res.raw.forEach(function (r) { r.forEach(function (v) { flat.push(v); }); });
        var mean = sum(flat) / flat.length, sd = Math.sqrt(flat.reduce(function (a, v) { return a + (v - mean) * (v - mean); }, 0) / flat.length);
        var ent = -row.reduce(function (a, w) { return a + (w > 0 ? w * Math.log(w) : 0); }, 0);
        var html = K.tex("\\text{Attention}(Q,K,V) = \\operatorname{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\right)V") + " &nbsp; (d<sub>k</sub> = " + d + ", M = 0 or −∞)<br>" +
          "Query <b>“" + toks[sel] + "”</b>: q = [" + res.Q[sel].slice(0, 4).map(function (v) { return v.toFixed(2); }).join(", ") + ", …] attends to " +
          top.map(function (t) { return "<b>" + toks[t[1]] + "</b> " + (t[0] * 100).toFixed(0) + "%"; }).join(", ") + " (entropy " + ent.toFixed(2) + " nats). ";
        html += "Raw q·k scores have std ≈ <b>" + sd.toFixed(1) + "</b>; ÷√" + d + " brings it to ≈ " + (sd / Math.sqrt(d)).toFixed(1) + ". " +
          (scaled ? "" : "<b style='color:var(--red)'>Unscaled:</b> softmax saturates into near one-hot rows → tiny gradients, the model stops learning. ") + "<br>";
        if (head === 0) html += causal ? "<span class='muted'>With the causal mask “it” can't see “" + ending + "” yet, so it hedges between animal/street; the <i>later</i> token “" + ending + "” resolves it (switch the ending and hover “" + ending + "”). Mask off (encoder/BERT-style) lets “it” use both sides.</span>"
          : "<span class='muted'>Bidirectional (no mask): “it” can look ahead at “" + ending + "”. This is what an encoder like BERT does; a GPT can't, because it must predict the future.</span>";
        else if (head === 1) html += "<span class='muted'>A “previous-token head”: each query matches the key of the token before it. Real models learn heads like this early; they're a building block of induction heads (copying patterns).</span>";
        else html += "<span class='muted'>Real heads specialise in fuzzy, overlapping ways; different heads run in parallel on the same tokens, and their outputs are concatenated. The Q/K vectors here are designed so the patterns are plausible — a trained model learns them from data.</span>";
        c.readout.innerHTML = html;
      }
      draw();
    },
  });

  // ================================================================ transformer-block
  V.register("transformer-block", {
    title: "Inside a GPT: follow the tensor shapes (B, T, C)",
    desc: "Click any box to see what it does, its tensor shape and parameter count. B = batch, T = tokens (time), C = channels (d_model), nh = heads, hs = head size = C/nh.",
    render: function (c, p, K) {
      var PRE = {
        nano: { name: "nanoGPT (Karpathy, Shakespeare)", style: "gpt2", B: 64, T: 256, C: 384, nh: 6, nkv: 6, L: 6, V: 65, ffn: 1536, ctx: 256 },
        gpt2: { name: "GPT-2 small", style: "gpt2", B: 8, T: 1024, C: 768, nh: 12, nkv: 12, L: 12, V: 50257, ffn: 3072, ctx: 1024 },
        llama: { name: "Llama-3.1-8B", style: "llama", B: 1, T: 8192, C: 4096, nh: 32, nkv: 8, L: 32, V: 128256, ffn: 14336, ctx: 131072 },
      };
      var key = p.preset || "nano", cfg = Object.assign({}, PRE[key]), selId = "scores";
      var TV = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
      var bS, tS;
      c.controls.appendChild(K.select({ label: "model", options: Object.keys(PRE).map(function (k) { return { value: k, label: PRE[k].name }; }), value: key, onChange: function (v) { key = v; cfg = Object.assign({}, PRE[v]); bS.set(cfg.B); tS.set(TV.indexOf(cfg.T)); draw(); } }));
      bS = K.slider({ label: "B (batch)", min: 1, max: 64, value: cfg.B, onInput: function (v) { cfg.B = v; draw(); } });
      tS = K.slider({ label: "T (tokens)", min: 0, max: TV.length - 1, value: TV.indexOf(cfg.T), fmt: function (v) { return TV[v]; }, onInput: function (v) { cfg.T = TV[v]; draw(); } });
      c.controls.appendChild(bS); c.controls.appendChild(tS);

      function nodes() {
        var B = cfg.B, T = cfg.T, C = cfg.C, nh = cfg.nh, nkv = cfg.nkv, hs = C / nh, Vv = cfg.V, F = cfg.ffn, ll = cfg.style === "llama";
        var sh = function (a, b) { return { sym: a, num: "(" + b.join(", ") + ")", n: b.reduce(function (x, y) { return x * y; }, 1) }; };
        var O = [
          { id: "idx", label: "token ids", s: sh("(B, T)", [B, T]), params: 0, desc: "Integers: which vocabulary entry each position holds. The tokenizer produced these.", code: "idx = tokenizer.encode(text)   # LongTensor (B, T)" },
          { id: "wte", label: "token embedding", s: sh("(B, T, C)", [B, T, C]), params: Vv * C, desc: "A lookup table of V rows × C columns. Row i is token i's learned vector; indexing it turns ids into vectors.", code: "tok_emb = self.wte(idx)        # nn.Embedding(V, C)" },
          ll ? { id: "pos", label: "(no position table)", s: sh("—", [B, T, C]), params: 0, desc: "Llama has no learned position embedding: position is injected inside attention by rotating q and k (RoPE).", code: "x = tok_emb" }
            : { id: "pos", label: "+ position embedding", s: sh("(B, T, C)", [B, T, C]), params: cfg.ctx * C, desc: "Attention itself is order-blind, so GPT-2 adds a learned vector per position (up to the max context).", code: "x = tok_emb + self.wpe(torch.arange(T))" },
          { id: "blocks", label: "Block × " + cfg.L, s: sh("(B, T, C)", [B, T, C]), params: 0, desc: "The same block design repeated L times; each reads and writes the residual stream x of shape (B,T,C). Shape in = shape out, which is what makes stacking trivial.", code: "for block in self.blocks: x = block(x)" },
          { id: "lnf", label: ll ? "final RMSNorm" : "final LayerNorm", s: sh("(B, T, C)", [B, T, C]), params: ll ? C : 2 * C, desc: "Normalise before the output projection.", code: "x = self.ln_f(x)" },
          { id: "head", label: "lm_head", s: sh("(B, T, V)", [B, T, Vv]), params: Vv * C, desc: "Project each position's C-dim vector to V logits: one score per possible next token. GPT-2 ties this matrix to the token embedding (shares weights).", code: "logits = self.lm_head(x)       # nn.Linear(C, V, bias=False)" },
          { id: "probs", label: "softmax → next token", s: sh("(B, V) at inference", [B, Vv]), params: 0, desc: "Training uses all T positions (cross-entropy vs the shifted targets). Generation only needs the last position: logits[:, -1, :] → softmax → sample.", code: "probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)\nnext_id = torch.multinomial(probs, 1)" },
        ];
        var Bk = [
          { id: "ln1", label: ll ? "RMSNorm 1" : "LayerNorm 1", s: sh("(B, T, C)", [B, T, C]), params: ll ? C : 2 * C, desc: "Pre-norm: normalise a copy of x before attention; the residual stream itself is left untouched.", code: "h = self.ln_1(x)" },
          { id: "qkv", label: ll ? "Q, K, V projections (GQA)" : "Q, K, V projections", s: sh("q (B,nh,T,hs) k,v (B," + (ll ? "nkv" : "nh") + ",T,hs)", [B, nh, T, hs]), params: C * C + 2 * C * nkv * hs + (ll ? 0 : 3 * C), desc: "Three linear maps produce queries, keys, values, split into " + nh + " heads of size hs = " + hs + (ll ? ". Llama uses grouped-query attention: only " + nkv + " K/V heads shared by " + nh + " query heads (smaller KV cache)." : "."), code: "q, k, v = self.c_attn(h).split(C, dim=2)\nq = q.view(B, T, nh, hs).transpose(1, 2)   # (B, nh, T, hs)" },
          ll ? { id: "rope", label: "RoPE: rotate q, k", s: sh("(B, nh, T, hs)", [B, nh, T, hs]), params: 0, desc: "Rotary position embedding: rotate pairs of q/k dimensions by an angle proportional to position, so q·k depends on relative distance.", code: "q, k = apply_rotary_emb(q, k, freqs_cis)" } : null,
          { id: "scores", label: "scores = q·kᵀ/√hs + mask", s: sh("(B, nh, T, T)", [B, nh, T, T]), params: 0, desc: "Every query against every key: a T×T matrix per head. This is the part that grows quadratically with context. FlashAttention never materialises it in memory.", code: "att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))\natt = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))" },
          { id: "softmax", label: "softmax (per row)", s: sh("(B, nh, T, T)", [B, nh, T, T]), params: 0, desc: "Turn each row of scores into weights that sum to 1.", code: "att = F.softmax(att, dim=-1)" },
          { id: "av", label: "weights @ v", s: sh("(B, nh, T, hs)", [B, nh, T, hs]), params: 0, desc: "Each query position gets a weighted average of the value vectors it attends to.", code: "y = att @ v   # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)" },
          { id: "proj", label: "concat heads + out proj", s: sh("(B, T, C)", [B, T, C]), params: C * C + (ll ? 0 : C), desc: "Glue the heads back together (nh × hs = C) and mix them with one more linear layer.", code: "y = y.transpose(1, 2).contiguous().view(B, T, C)\ny = self.c_proj(y)" },
          { id: "res1", label: "⊕ residual add", s: sh("(B, T, C)", [B, T, C]), params: 0, desc: "Add attention's output back onto x. The residual stream is a highway: gradients flow straight through, and each sublayer only has to learn a small update.", code: "x = x + self.attn(self.ln_1(x))" },
          { id: "ln2", label: ll ? "RMSNorm 2" : "LayerNorm 2", s: sh("(B, T, C)", [B, T, C]), params: ll ? C : 2 * C, desc: "Normalise again before the MLP.", code: "h = self.ln_2(x)" },
          { id: "up", label: ll ? "gate & up: C → ffn" : "MLP up: C → 4C", s: sh(ll ? "2 × (B, T, ffn)" : "(B, T, 4C)", [B, T, F]), params: ll ? 2 * C * F : C * F + F, desc: "Expand each token's vector independently (no mixing across tokens here). This is where most parameters — and much of the model's “knowledge” — live.", code: ll ? "g, u = self.w1(h), self.w3(h)   # (B,T,ffn) each" : "h = self.c_fc(h)                 # (B,T,4C)" },
          { id: "act", label: ll ? "SiLU(gate) ⊙ up (SwiGLU)" : "GELU", s: sh(ll ? "(B, T, ffn)" : "(B, T, 4C)", [B, T, F]), params: 0, desc: "The non-linearity. Llama uses a gated variant: one branch decides how much of the other passes through.", code: ll ? "h = F.silu(g) * u" : "h = F.gelu(h)" },
          { id: "down", label: ll ? "down: ffn → C" : "MLP down: 4C → C", s: sh("(B, T, C)", [B, T, C]), params: F * C + (ll ? 0 : C), desc: "Project back to C so the result can be added to the residual stream.", code: ll ? "h = self.w2(h)" : "h = self.c_proj(h)" },
          { id: "res2", label: "⊕ residual add", s: sh("(B, T, C)", [B, T, C]), params: 0, desc: "x = x + mlp(ln_2(x)). One block done — the output has the same shape as the input.", code: "x = x + self.mlp(self.ln_2(x))" },
        ].filter(Boolean);
        return { O: O, Bk: Bk };
      }
      function draw() {
        var N = nodes(), W = 700, H = 404, svg = newSvg(K, W, H), all = N.O.concat(N.Bk), cur = all.filter(function (n) { return n.id === selId; })[0] || all[0];
        function box(n, x, y, w, h) {
          var g = K.s("g", { style: { cursor: "pointer" } });
          var special = n.id === "blocks";
          g.appendChild(K.s("rect", { x: x, y: y, width: w, height: h, rx: 7, class: n.id === selId ? "box-hl" : "box", "stroke-dasharray": special ? "5 3" : null }));
          g.appendChild(K.s("text", { x: x + 9, y: y + h / 2 + 4, class: "lbl", style: { fontSize: "11.5px" }, text: n.label }));
          g.appendChild(K.s("text", { x: x + w - 8, y: y + h / 2 + 4, "text-anchor": "end", class: "lbl-sm mono", text: n.s.num }));
          g.addEventListener("click", function () { selId = n.id; draw(); });
          svg.appendChild(g);
        }
        // outer column
        var ox = 8, ow = 262, oy0 = 20, ostep = 53, bh = 30;
        svg.appendChild(K.s("text", { x: ox, y: 12, class: "lbl-sm", text: "whole model" }));
        N.O.forEach(function (n, i) {
          var y = oy0 + i * ostep;
          if (i) arrow(K, svg, ox + ow / 2, y - ostep + bh, ox + ow / 2, y - 1, "smuted", "fmuted", 1.3);
          box(n, ox, y, ow, bh);
        });
        // block column
        var bx = 316, bw = 346, n = N.Bk.length, by0 = 20, bstep = (H - by0 - 10) / n, bbh = bstep - 7;
        svg.appendChild(K.s("text", { x: bx, y: 12, class: "lbl-sm", text: "one transformer block (repeated " + cfg.L + "×)" }));
        var bIdx = 3, byOuter = oy0 + bIdx * ostep;
        svg.appendChild(K.s("path", { d: "M" + (ox + ow) + " " + byOuter + " L" + (bx - 8) + " " + by0 + " M" + (ox + ow) + " " + (byOuter + bh) + " L" + (bx - 8) + " " + (H - 12), class: "ln-thin smuted dash" }));
        var ri = {};
        N.Bk.forEach(function (nd, i) {
          var y = by0 + i * bstep;
          ri[nd.id] = y + bbh / 2;
          if (i) arrow(K, svg, bx + bw / 2, y - 7, bx + bw / 2, y - 0.5, "smuted", "fmuted", 1.1);
          box(nd, bx, y, bw, bbh);
        });
        // residual stream lines
        var rx = bx + bw + 16;
        [["ln1", "res1"], ["ln2", "res2"]].forEach(function (pr) {
          var y1 = ri[pr[0]] - bbh / 2 - 3, y2 = ri[pr[1]];
          svg.appendChild(K.s("path", { d: "M" + (bx + bw) + " " + y1 + " H" + rx + " V" + y2 + " H" + (bx + bw + 9), class: "ln-thin s3", "stroke-width": 2, fill: "none" }));
          svg.appendChild(K.s("polygon", { points: (bx + bw + 2) + "," + y2 + " " + (bx + bw + 9) + "," + (y2 - 4) + " " + (bx + bw + 9) + "," + (y2 + 4), class: "f3" }));
        });
        svg.appendChild(K.s("text", { x: rx + 6, y: (ri.ln1 + ri.res1) / 2, class: "lbl-sm", transform: "rotate(90 " + (rx + 6) + " " + (ri.ln1 + ri.res1) / 2 + ")", "text-anchor": "middle", text: "residual" }));
        K.clear(c.stage).appendChild(svg);
        var pc = countParams({ style: cfg.style === "llama" ? "llama" : "gpt2", L: cfg.L, d: cfg.C, h: cfg.nh, kv: cfg.nkv, ffn: cfg.ffn, V: cfg.V, ctx: cfg.ctx, tied: cfg.style !== "llama" });
        c.readout.innerHTML = "<b>" + cur.label + "</b> — shape <b class='mono'>" + esc(cur.s.sym) + "</b> = <span class='mono'>" + cur.s.num + "</span> → " + K.fmtNum(cur.s.n, 0) + " numbers (" + K.fmtBytes(cur.s.n * 2) + " in bf16)" +
          (cur.params ? " · params: <b>" + fmtP(cur.params) + "</b>" + (N.Bk.indexOf(cur) >= 0 ? " per block × " + cfg.L : "") : "") + "<br>" + cur.desc + codeBlock(cur.code) +
          "<span class='muted'>Whole model ≈ <b>" + fmtP(pc.total) + "</b> parameters (C=" + cfg.C + ", nh=" + cfg.nh + ", hs=" + cfg.C / cfg.nh + ", L=" + cfg.L + ", V=" + cfg.V + "). Try T = 8192 and click “scores”: the (B,nh,T,T) tensor is why long context is expensive.</span>";
      }
      draw();
    },
  });

  // ================================================================ param-counter
  V.register("param-counter", {
    title: "Where do the parameters live? (and how much memory do they need)",
    desc: "Pick a real model or drag the config sliders. Parameters = embeddings + L × (attention + MLP + norms) + output head. Memory for the weights = params × bytes per parameter.",
    render: function (c, p, K) {
      var key = p.preset || "llama31-8b", cfg = Object.assign({}, MODELS[key]), S = {};
      var presetSel = K.select({ label: "preset", options: Object.keys(MODELS).map(function (k) { return { value: k, label: MODELS[k].name }; }).concat([{ value: "custom", label: "Custom" }]), value: key, onChange: function (v) { if (v === "custom") return; key = v; cfg = Object.assign({}, MODELS[v]); syncSliders(); draw(); } });
      c.controls.appendChild(presetSel);
      var styleSel = K.select({ label: "style", options: [{ value: "gpt2", label: "GPT-2 (LayerNorm, GELU, biases)" }, { value: "llama", label: "Llama (RMSNorm, SwiGLU)" }, { value: "qwen", label: "Qwen2 (Llama + QKV bias)" }], value: cfg.style, onChange: function (v) { cfg.style = v; custom(); } });
      c.controls.appendChild(styleSel);
      function mk(k, label, min, max, step) {
        S[k] = K.slider({ label: label, min: min, max: max, step: step || 1, value: cfg[k], onInput: function (v) { cfg[k] = v; if (k === "h") { cfg.hd = cfg.d / cfg.h; if (cfg.kv > cfg.h) cfg.kv = cfg.h; S.kv.input.max = cfg.h; S.kv.set(cfg.kv); } if (k === "d") cfg.hd = cfg.d / cfg.h; custom(); } });
        c.controls.appendChild(S[k]);
      }
      mk("L", "layers L", 1, 128); mk("d", "d_model", 64, 16384, 64); mk("h", "heads", 1, 128); mk("kv", "KV heads", 1, 128); mk("ffn", "ffn dim", 256, 65536, 256); mk("V", "vocab", 1000, 262144, 1);
      var tie = K.toggle({ label: "tied embeddings", value: cfg.tied, onChange: function (v) { cfg.tied = v; custom(); } });
      c.controls.appendChild(tie);
      function custom() { presetSel.input.value = "custom"; draw(); }
      function syncSliders() { styleSel.input.value = cfg.style; S.kv.input.max = cfg.h; ["L", "d", "h", "kv", "ffn", "V"].forEach(function (k) { S[k].set(cfg[k]); }); tie.input.checked = cfg.tied; }
      S.kv.input.max = cfg.h;

      function draw() {
        cfg.hd = cfg.hd || cfg.d / cfg.h;
        var r = countParams(cfg), T = r.total;
        var parts = [["embedding", r.emb, "f1"], ["attention", r.attn, "f2"], ["MLP", r.mlp, "f3"], ["norms", r.norm, "f4"], ["lm_head", r.head, "f5"]];
        var W = 680, svg = newSvg(K, W, 262);
        svg.appendChild(K.s("text", { x: 0, y: 14, class: "lbl", text: "Parameter breakdown — total " + fmtP(T) }));
        var x = 0, bwid = W;
        parts.forEach(function (pt) {
          var w = pt[1] / T * bwid;
          if (w <= 0) return;
          svg.appendChild(K.s("rect", { x: x, y: 24, width: Math.max(0.5, w - 1), height: 34, rx: 3, class: pt[2], opacity: 0.85 }));
          if (w > 60) svg.appendChild(K.s("text", { x: x + 6, y: 45, class: "t-white", style: { fontSize: "11px", fontWeight: 600 }, text: pt[0] + " " + (pt[1] / T * 100).toFixed(0) + "%" }));
          x += w;
        });
        parts.forEach(function (pt, i) {
          svg.appendChild(K.s("rect", { x: i * 136, y: 68, width: 10, height: 10, class: pt[2] }));
          svg.appendChild(K.s("text", { x: i * 136 + 14, y: 77, class: "lbl-sm", text: pt[0] + " " + fmtP(pt[1]) }));
        });
        // memory bars (log scale)
        var dts = [["fp32", 4], ["bf16 / fp16", 2], ["int8 / fp8", 1], ["int4", 0.5]];
        var gb = dts.map(function (d) { return T * d[1] / 1e9; });
        var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, gb) / 1.5))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(Math.max.apply(null, gb) * 1.3, 200))));
        var sx = K.scale(lo, hi, 90, W - 30, true), y0 = 110;
        svg.appendChild(K.s("text", { x: 0, y: y0 - 12, class: "lbl", text: "Weight memory (log scale)" }));
        K.logTicks(lo, hi).forEach(function (v) {
          svg.appendChild(K.s("line", { x1: sx(v), x2: sx(v), y1: y0 - 4, y2: y0 + 4 * 26, class: "grid" }));
          svg.appendChild(K.s("text", { x: sx(v), y: y0 + 4 * 26 + 14, "text-anchor": "middle", class: "tick", text: v >= 1 ? v + " GB" : v * 1000 + " MB" }));
        });
        dts.forEach(function (d, i) {
          var y = y0 + i * 26;
          svg.appendChild(K.s("text", { x: 84, y: y + 14, "text-anchor": "end", class: "lbl-sm", text: d[0] }));
          svg.appendChild(K.s("rect", { x: 90, y: y + 2, width: Math.max(1, sx(gb[i]) - 90), height: 18, rx: 3, class: "f" + (i + 1), opacity: 0.8 }));
          svg.appendChild(K.s("text", { x: sx(gb[i]) + 5, y: y + 15, class: "lbl-sm mono", text: fmtGB(gb[i] * 1e9) }));
        });
        [[16, "T4 / Mac 16"], [24, "L4 24"], [80, "A100/H100 80"], [192, "B200 192"]].forEach(function (g, i) {
          if (g[0] < lo || g[0] > hi) return;
          svg.appendChild(K.s("line", { x1: sx(g[0]), x2: sx(g[0]), y1: y0 - 6, y2: y0 + 4 * 26, class: "ln-thin s5 dash" }));
          svg.appendChild(K.s("text", { x: sx(g[0]) + 3, y: y0 + 4 * 26 + 28 + (i % 2) * 13, class: "lbl-sm", style: { fill: "var(--c5)" }, text: g[1] }));
        });
        K.clear(c.stage).appendChild(svg);
        var hd = cfg.d / cfg.h, ll = cfg.style !== "gpt2";
        var kvB = 2 * cfg.L * cfg.kv * hd * 2;
        c.readout.innerHTML = statGrid([
          ["total params", fmtP(T), "--c1"],
          ["non-embedding", fmtP(r.attn + r.mlp + r.norm)],
          ["per layer", fmtP(r.perLayer)],
          ["attn : MLP per layer", fmtP(r.attnL) + " : " + fmtP(r.mlpL)],
          ["head_dim", Number.isInteger(hd) ? hd : hd.toFixed(1) + " ⚠"],
          ["KV cache / token (bf16)", K.fmtBytes(kvB)],
        ]) +
          "Per layer: attention " + K.tex("= d\\cdot(n_h + 2n_{kv})\\,d_h + n_h d_h\\cdot d") + ", MLP " + K.tex(ll ? "= 3\\cdot d\\cdot d_{ffn}" : "= 2\\cdot d\\cdot d_{ffn}") +
          (ll ? " (gate, up, down)" : "") + ". Rule of thumb for GPT-style models: " + K.tex("N \\approx 12\\,L\\,d^2") + " = " + fmtP(12 * cfg.L * cfg.d * cfg.d) + " (ignores embeddings). " +
          "<span class='muted'>Inference needs ≈ " + fmtGB(T * 2) + " just for bf16 weights — before any KV cache or activations. That single number decides which GPU you need and why quantization matters." +
          (cfg.tied ? " Tied embeddings reuse the input table as the output head, saving " + fmtP(cfg.V * cfg.d) + "." : "") + "</span>";
      }
      draw();
    },
  });

  // ================================================================ rope
  V.register("rope", {
    title: "RoPE: encode position by rotating q and k",
    desc: "Split each q/k vector into pairs of numbers; rotate pair i by angle position × θᵢ. Fast-spinning pairs track nearby positions, slow ones track far positions. Because both q and k are rotated, their dot product only depends on the <i>distance</i> m − n.",
    render: function (c, p, K) {
      var m = 5, n = 2, pi = p.pair !== undefined ? p.pair : 2, base = p.base || 10000, d = 128;
      var r = K.rng(21), qv = [], kv = [];
      for (var i = 0; i < d; i++) { qv.push(K.randn(r)); kv.push(qv[i] * 0.7 + 0.7 * K.randn(r)); }
      var mS = K.slider({ label: "query position m", min: 0, max: 128, value: m, onInput: function (v) { m = v; draw(); } });
      var nS = K.slider({ label: "key position n", min: 0, max: 128, value: n, onInput: function (v) { n = v; draw(); } });
      c.controls.appendChild(mS); c.controls.appendChild(nS);
      c.controls.appendChild(K.slider({ label: "pair i", min: 0, max: d / 2 - 1, value: pi, onInput: function (v) { pi = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "base", options: [{ value: 10000, label: "10,000 (RoPE paper, Llama-2)" }, { value: 500000, label: "500,000 (Llama-3)" }], value: base, onChange: function (v) { base = +v; draw(); } }));
      c.controls.appendChild(K.button("Shift both +10", function () { if (m <= 118 && n <= 118) { m += 10; n += 10; } else { m -= Math.min(m, n); n -= Math.min(m, n); } mS.set(m); nS.set(n); draw(); }));

      function theta(i) { return Math.pow(base, -2 * i / d); }
      function rot(v, i, pos) { var a = pos * theta(i), cs = Math.cos(a), sn = Math.sin(a); return [v[2 * i] * cs - v[2 * i + 1] * sn, v[2 * i] * sn + v[2 * i + 1] * cs]; }
      function score(mm, nn) { var s = 0; for (var i = 0; i < d / 2; i++) { var a = rot(qv, i, mm), b = rot(kv, i, nn); s += a[0] * b[0] + a[1] * b[1]; } return s / Math.sqrt(d); }
      function draw() {
        var W = 700, H = 330, svg = newSvg(K, W, H);
        // circle panel for pair pi
        var cx = 150, cy = 150, R = 110;
        svg.appendChild(K.s("circle", { cx: cx, cy: cy, r: R, fill: "none", class: "grid" }));
        svg.appendChild(K.s("line", { x1: cx - R - 8, x2: cx + R + 8, y1: cy, y2: cy, class: "axis" }));
        svg.appendChild(K.s("line", { y1: cy - R - 8, y2: cy + R + 8, x1: cx, x2: cx, class: "axis" }));
        var q0 = [qv[2 * pi], qv[2 * pi + 1]], k0 = [kv[2 * pi], kv[2 * pi + 1]], sc = R / Math.max(Math.hypot(q0[0], q0[1]), Math.hypot(k0[0], k0[1]), 0.01) * 0.9;
        var qr = rot(qv, pi, m), kr = rot(kv, pi, n);
        svg.appendChild(K.s("line", { x1: cx, y1: cy, x2: cx + q0[0] * sc, y2: cy - q0[1] * sc, class: "ln-thin s1 dash", opacity: 0.5 }));
        svg.appendChild(K.s("line", { x1: cx, y1: cy, x2: cx + k0[0] * sc, y2: cy - k0[1] * sc, class: "ln-thin s2 dash", opacity: 0.5 }));
        arrow(K, svg, cx, cy, cx + qr[0] * sc, cy - qr[1] * sc, "s1", "f1", 3);
        arrow(K, svg, cx, cy, cx + kr[0] * sc, cy - kr[1] * sc, "s2", "f2", 3);
        svg.appendChild(K.s("text", { x: cx + qr[0] * sc * 1.08 + 4, y: cy - qr[1] * sc * 1.08, class: "lbl", style: { fill: "var(--c1)" }, text: "q at m=" + m }));
        svg.appendChild(K.s("text", { x: cx + kr[0] * sc * 1.08 + 4, y: cy - kr[1] * sc * 1.08 + 12, class: "lbl", style: { fill: "var(--c2)" }, text: "k at n=" + n }));
        svg.appendChild(K.s("text", { x: 8, y: 16, class: "lbl-sm", text: "pair " + pi + " (dims " + 2 * pi + "," + (2 * pi + 1) + "); dashed = before rotation" }));
        // dials: rotation angle of several pairs at position m
        var dialIdx = [0, 4, 8, 16, 24, 32, 48, 63], dx0 = 330, dy = 44;
        svg.appendChild(K.s("text", { x: dx0, y: 14, class: "lbl-sm", text: "rotation of q's pairs at position m = " + m + " (like clock hands at different speeds)" }));
        dialIdx.forEach(function (i, j) {
          var x = dx0 + 18 + j * 46, a = m * theta(i);
          svg.appendChild(K.s("circle", { cx: x, cy: dy, r: 17, class: i === pi ? "box-hl" : "box" }));
          svg.appendChild(K.s("line", { x1: x, y1: dy, x2: x + 15 * Math.cos(a), y2: dy - 15 * Math.sin(a), class: "s1", "stroke-width": 2.2 }));
          svg.appendChild(K.s("text", { x: x, y: dy + 30, "text-anchor": "middle", class: "lbl-sm", text: "i=" + i }));
        });
        // score vs relative distance
        var ch0 = 330, cw = 360, ct = 100, chh = 190, maxD = 128;
        var ss = [], env = [];
        for (var t = -maxD; t <= maxD; t++) {
          ss.push([t, t >= 0 ? score(t, 0) : score(0, -t)]);
          var e = 0; for (var i2 = 0; i2 < d / 2; i2++) e += Math.cos(t * theta(i2)); env.push([t, e / (d / 2)]);
        }
        var mx = Math.max.apply(null, ss.map(function (q) { return Math.abs(q[1]); })) * 1.1;
        var sx = K.scale(-maxD, maxD, ch0 + 30, ch0 + cw), sy = K.scale(-mx, mx, ct + chh, ct), sy2 = K.scale(-1, 1, ct + chh, ct);
        svg.appendChild(K.s("line", { x1: ch0 + 30, x2: ch0 + cw, y1: sy(0), y2: sy(0), class: "axis" }));
        svg.appendChild(K.s("text", { x: ch0 + 26, y: sy(0) + 4, "text-anchor": "end", class: "tick", text: "0" }));
        [-128, -64, 0, 64, 128].forEach(function (v) { svg.appendChild(K.s("text", { x: sx(v), y: ct + chh + 14, "text-anchor": "middle", class: "tick", text: v })); });
        svg.appendChild(K.s("text", { x: ch0 + 30 + cw / 2, y: ct + chh + 30, "text-anchor": "middle", class: "axis-label", text: "relative position m − n" }));
        svg.appendChild(K.s("text", { x: ch0 + 34, y: ct - 4, class: "lbl-sm", text: "q·k vs m − n (blue) · decay envelope for q = k (grey)" }));
        svg.appendChild(K.s("path", { d: K.path(env, sx, sy2), class: "ln-thin smuted" }));
        svg.appendChild(K.s("path", { d: K.path(ss, sx, sy), class: "ln s1" }));
        var dd = m - n;
        svg.appendChild(K.s("line", { x1: sx(dd), x2: sx(dd), y1: ct, y2: ct + chh, class: "ln-thin s2 dash" }));
        svg.appendChild(K.s("circle", { cx: sx(dd), cy: sy(score(m, n)), r: 5, class: "f2" }));
        K.clear(c.stage).appendChild(svg);
        var th = theta(pi), ang = (m * th * 180 / Math.PI) % 360, s1 = score(m, n), s2 = score(m + 7, n + 7);
        c.readout.innerHTML = K.tex("\\begin{pmatrix}q'_{2i}\\\\ q'_{2i+1}\\end{pmatrix} = \\begin{pmatrix}\\cos m\\theta_i & -\\sin m\\theta_i\\\\ \\sin m\\theta_i & \\cos m\\theta_i\\end{pmatrix}\\begin{pmatrix}q_{2i}\\\\ q_{2i+1}\\end{pmatrix},\\quad \\theta_i = b^{-2i/d}") + "<br>" +
          "Pair " + pi + ": θ = " + th.toExponential(2) + " rad/token → wavelength " + K.fmtNum(2 * Math.PI / th, 0) + " tokens; at m = " + m + " it's rotated " + ang.toFixed(0) + "°. " +
          "Score q<sub>m</sub>·k<sub>n</sub>/√d = <b>" + s1.toFixed(3) + "</b>; shift both by 7 → <b>" + s2.toFixed(3) + "</b> (identical: only m − n = " + (m - n) + " matters). " +
          "<span class='muted'>No position vectors are added to x at all — RoPE is applied to q and k inside every attention layer. A larger base (Llama-3's 500k) slows the slow pairs further, which helps long context.</span>";
      }
      draw();
    },
  });

  // ================================================================ gqa
  V.register("gqa", {
    title: "MHA vs GQA vs MQA vs MLA: sharing keys & values to shrink the KV cache",
    desc: "Every query head needs keys/values. Multi-head (MHA) gives each its own; grouped-query (GQA) lets a group of query heads share one K/V head; multi-query (MQA) shares one for all; DeepSeek's MLA caches a small compressed latent instead.",
    render: function (c, p, K) {
      var PR = {
        "llama31-8b": { name: "Llama-3.1-8B", L: 32, h: 32, kv: 8, hd: 128 },
        "llama31-70b": { name: "Llama-3.1-70B", L: 80, h: 64, kv: 8, hd: 128 },
        "qwen25-7b": { name: "Qwen2.5-7B", L: 28, h: 28, kv: 4, hd: 128 },
        "mistral-7b": { name: "Mistral-7B", L: 32, h: 32, kv: 8, hd: 128 },
        "dsv3": { name: "DeepSeek-V3 (MLA)", L: 61, h: 128, kv: 128, hd: 128, mla: true },
      };
      var key = p.preset || "llama31-8b", cfg = PR[key], mode = p.mode || (cfg.mla ? "mla" : "gqa"), kv = cfg.kv, bytes = 2;
      var CT = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], ci = 5, users = 16;
      function divisors(h) { var o = []; for (var i = 1; i <= h; i++) if (h % i === 0) o.push(i); return o; }
      var kvS;
      c.controls.appendChild(K.select({ label: "model", options: Object.keys(PR).map(function (k) { return { value: k, label: PR[k].name }; }), value: key, onChange: function (v) { key = v; cfg = PR[v]; kv = cfg.kv === cfg.h ? cfg.h / 8 : cfg.kv; mode = cfg.mla ? "mla" : "gqa"; modeSel.input.value = mode; resetKv(); draw(); } }));
      var modeSel = K.select({ label: "attention", options: [{ value: "mha", label: "MHA" }, { value: "gqa", label: "GQA" }, { value: "mqa", label: "MQA" }, { value: "mla", label: "MLA" }], value: mode, onChange: function (v) { mode = v; draw(); } });
      c.controls.appendChild(modeSel);
      kvS = K.slider({ label: "KV heads (GQA)", min: 0, max: 1, value: 0, fmt: function (v) { return divisors(cfg.h)[Math.round(v)]; }, onInput: function (v) { kv = divisors(cfg.h)[Math.round(v)]; mode = "gqa"; modeSel.input.value = "gqa"; draw(); } });
      c.controls.appendChild(kvS);
      function resetKv() { var dv = divisors(cfg.h); if (cfg.mla && kv === cfg.h) kv = 16; kvS.input.max = dv.length - 1; var ix = dv.indexOf(kv); if (ix < 0) { ix = 0; kv = dv[0]; } kvS.set(ix); }
      if (cfg.mla) kv = 16;
      resetKv();
      c.controls.appendChild(K.select({ label: "KV dtype", options: [{ value: 2, label: "bf16" }, { value: 1, label: "fp8" }], value: 2, onChange: function (v) { bytes = +v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "context", min: 0, max: CT.length - 1, value: ci, fmt: function (v) { return CT[v] >= 1024 ? CT[v] / 1024 + "k" : CT[v]; }, onInput: function (v) { ci = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "concurrent users", min: 1, max: 256, value: users, onInput: function (v) { users = v; draw(); } }));

      function perTok(md) {
        if (md === "mla") return cfg.L * (512 + 64) * bytes;
        var k = md === "mha" ? cfg.h : md === "mqa" ? 1 : kv;
        return 2 * cfg.L * k * cfg.hd * bytes;
      }
      function draw() {
        var W = 700, H = 330, svg = newSvg(K, W, H);
        var nq = Math.min(cfg.h, 32), bw = (W - 40) / nq, kvn = mode === "mha" ? cfg.h : mode === "mqa" ? 1 : kv, g = cfg.h / kvn;
        svg.appendChild(K.s("text", { x: 20, y: 14, class: "lbl-sm", text: "query heads" + (cfg.h > 32 ? " (showing 32 of " + cfg.h + ")" : "") }));
        for (var i = 0; i < nq; i++) {
          svg.appendChild(K.s("rect", { x: 20 + i * bw + 1, y: 20, width: bw - 2, height: 22, rx: 3, class: "f1", opacity: 0.8 }));
        }
        if (mode === "mla") {
          var lx = W / 2 - 120;
          for (var j = 0; j < nq; j++) svg.appendChild(K.s("line", { x1: 20 + j * bw + bw / 2, y1: 42, x2: 20 + j * bw + bw / 2, y2: 70, class: "ln-thin s2" }));
          svg.appendChild(K.s("rect", { x: 20, y: 70, width: W - 40, height: 18, rx: 3, class: "f2", opacity: 0.35 }));
          svg.appendChild(K.s("text", { x: W / 2, y: 83, "text-anchor": "middle", class: "lbl-sm", text: "per-head K,V reconstructed on the fly (up-projection, can be absorbed into the weights)" }));
          for (var j2 = 0; j2 < 12; j2++) svg.appendChild(K.s("line", { x1: 40 + j2 * (W - 80) / 11, y1: 88, x2: W / 2, y2: 110, class: "ln-thin smuted" }));
          svg.appendChild(K.s("rect", { x: lx, y: 110, width: 240, height: 26, rx: 5, class: "f4", opacity: 0.9 }));
          svg.appendChild(K.s("text", { x: W / 2, y: 127, "text-anchor": "middle", class: "t-white", style: { fontSize: "11.5px", fontWeight: 600 }, text: "cached latent c_KV: 512 + 64 (RoPE key)" }));
        } else {
          var nk = Math.max(1, Math.round(nq / g)), kw = (W - 40) / nk;
          for (var k = 0; k < nk; k++) {
            svg.appendChild(K.s("rect", { x: 20 + k * kw + 2, y: 104, width: kw - 4, height: 26, rx: 3, class: "f2", opacity: 0.85 }));
            if (kw > 34) svg.appendChild(K.s("text", { x: 20 + k * kw + kw / 2, y: 121, "text-anchor": "middle", class: "t-white", style: { fontSize: "10.5px", fontWeight: 600 }, text: kw > 60 ? "K,V " + k : k }));
          }
          for (var q = 0; q < nq; q++) {
            var kk = Math.min(nk - 1, Math.floor(q / g));
            svg.appendChild(K.s("line", { x1: 20 + q * bw + bw / 2, y1: 42, x2: 20 + kk * kw + kw / 2, y2: 104, class: "ln-thin s2", opacity: 0.6 }));
          }
          svg.appendChild(K.s("text", { x: 20, y: 146, class: "lbl-sm", text: "KV heads: " + kvn + " (each shared by " + g + " query head" + (g > 1 ? "s" : "") + ")" }));
        }
        // bars
        var modes = [["mha", "MHA (" + cfg.h + " KV heads)"], ["gqa", "GQA (" + kv + " KV heads)"], ["mqa", "MQA (1 KV head)"], ["mla", "MLA (latent 576)"]];
        var vals = modes.map(function (md) { return perTok(md[0]); }), mx = Math.max.apply(null, vals);
        var by = 170, bx = 150, bwid = W - bx - 170;
        svg.appendChild(K.s("text", { x: 0, y: by - 6, class: "lbl", text: "KV cache per token (all " + cfg.L + " layers)" }));
        modes.forEach(function (md, i) {
          var y = by + i * 34, w = vals[i] / mx * bwid, on = md[0] === mode;
          svg.appendChild(K.s("text", { x: bx - 6, y: y + 17, "text-anchor": "end", class: on ? "lbl" : "lbl-sm", text: md[1] }));
          svg.appendChild(K.s("rect", { x: bx, y: y + 4, width: Math.max(2, w), height: 20, rx: 3, class: on ? "f2" : "fmuted", opacity: on ? 0.9 : 0.45 }));
          svg.appendChild(K.s("text", { x: bx + Math.max(2, w) + 6, y: y + 18, class: "lbl-sm mono", text: K.fmtBytes(vals[i]) + (i === 0 ? " · baseline" : " · " + (vals[0] / vals[i]).toFixed(1) + "× smaller") }));
        });
        K.clear(c.stage).appendChild(svg);
        var pt = perTok(mode), ctx = CT[ci], tot = pt * ctx * users;
        c.readout.innerHTML = statGrid([
          ["KV bytes / token", K.fmtBytes(pt), "--c2"],
          ["per sequence @ " + (ctx / 1024) + "k", fmtGB(pt * ctx)],
          ["× " + users + " users", fmtGB(tot), tot > 80e9 ? "--red" : "--green"],
          ["H100s' worth (80 GB)", (tot / 80e9).toFixed(2)],
        ]) +
          (mode === "mla" ? K.tex("\\text{bytes/token} = L\\times(d_c + d_{rope})\\times\\text{bytes} = " + cfg.L + "\\times 576 \\times " + bytes) :
            K.tex("\\text{bytes/token} = 2 \\times L \\times n_{kv} \\times d_{head} \\times \\text{bytes} = 2\\times" + cfg.L + "\\times" + (mode === "mha" ? cfg.h : mode === "mqa" ? 1 : kv) + "\\times" + cfg.hd + "\\times" + bytes)) +
          " (the 2 is K and V). <span class='muted'>Fewer KV heads ⇒ smaller cache ⇒ more concurrent users and cheaper decoding (decode is memory-bandwidth-bound: every step re-reads the whole cache). Quality cost: MQA hurts noticeably, GQA with 8 groups is nearly free — why Llama-3, Qwen2.5 and Mistral all use it. MLA gets an even smaller cache with MHA-level quality at the price of extra projection math." +
          (cfg.mla ? " DeepSeek-V3 with plain MHA (128 heads) would need " + K.fmtBytes(perTok("mha")) + "/token." : "") + "</span>";
      }
      draw();
    },
  });

  // ================================================================ moe-routing
  V.register("moe-routing", {
    title: "Mixture of Experts: a router sends each token to its top-k experts",
    desc: "The MLP is replaced by E expert MLPs plus a tiny router. Each token only runs k of them, so compute tracks <i>active</i> params while memory must hold <i>all</i> params. Watch the load: popular experts overflow.",
    render: function (c, p, K) {
      var TOK = [["The", "fn"], ["cat", "noun"], ["sat", "verb"], ["on", "fn"], ["the", "fn"], ["mat", "noun"], [".", "p"], ["def", "code"], ["f", "code"], ["(", "p"], ["x", "code"], [")", "p"], [":", "p"],
        ["return", "code"], ["x", "code"], ["*", "p"], ["2", "num"], ["+", "p"], ["17", "num"], ["=", "p"], ["19", "num"]];
      var MP = {
        mixtral: { name: "Mixtral 8x7B", E: 8, k: 2, total: 46.7e9, shared: 1.6e9, shared_note: "attention + embeddings" },
        dsv3: { name: "DeepSeek-V3", E: 256, k: 8, total: 671e9, shared: 17e9, shared_note: "attention (MLA), 1 shared expert, 3 dense layers, embeddings" },
        qwen3: { name: "Qwen3-30B-A3B", E: 128, k: 8, total: 30.5e9, shared: 1.5e9, shared_note: "attention + embeddings" },
      };
      var E = p.experts || 8, k = p.k || 2, imb = 1.2, capF = 1.25, balance = false, seed = 3, hov = null, mkey = "mixtral";
      c.controls.appendChild(K.slider({ label: "experts E", min: 2, max: 16, value: E, onInput: function (v) { E = v; if (k > E) k = E; draw(); } }));
      c.controls.appendChild(K.slider({ label: "top-k", min: 1, max: 4, value: k, onInput: function (v) { k = Math.min(v, E); draw(); } }));
      c.controls.appendChild(K.slider({ label: "router skew", min: 0, max: 3, step: 0.1, value: imb, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { imb = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "capacity factor", min: 1, max: 3.1, step: 0.05, value: capF, fmt: function (v) { return v > 3 ? "∞ (dropless)" : v.toFixed(2); }, onInput: function (v) { capF = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "load balancing (bias)", value: balance, onChange: function (v) { balance = v; draw(); } }));
      c.controls.appendChild(K.button("New router", function () { seed++; draw(); }));
      c.controls.appendChild(K.select({ label: "real model", options: Object.keys(MP).map(function (q) { return { value: q, label: MP[q].name }; }), value: mkey, onChange: function (v) { mkey = v; draw(); } }));

      function route() {
        var r = K.rng(seed * 101 + E), types = ["fn", "noun", "verb", "p", "code", "num"], pref = {};
        types.forEach(function (t) { pref[t] = []; for (var e = 0; e < E; e++) pref[t].push(0); pref[t][Math.floor(r() * E)] += 1.6; pref[t][Math.floor(r() * E)] += 1.0; });
        var logits = TOK.map(function (tk) { var row = []; for (var e = 0; e < E; e++) row.push(pref[tk[1]][e] + 0.7 * K.randn(r) + imb * (e === 0 ? 1 : e === 1 ? 0.5 : 0)); return row; });
        var bias = new Array(E).fill(0);
        function select(b) {
          return logits.map(function (row) {
            var idx = row.map(function (v, e) { return e; }).sort(function (x, y) { return (row[y] - b[y]) - (row[x] - b[x]); }).slice(0, k);
            var g = K.softmax(idx.map(function (e) { return row[e]; }));
            return idx.map(function (e, j) { return [e, g[j]]; });
          });
        }
        var sel = select(bias);
        if (balance) {
          for (var it = 0; it < 200; it++) {
            var load = new Array(E).fill(0); sel.forEach(function (s) { s.forEach(function (q) { load[q[0]]++; }); });
            var mean = TOK.length * k / E;
            for (var e2 = 0; e2 < E; e2++) bias[e2] += 0.02 * (load[e2] - mean);
            sel = select(bias);
          }
        }
        var probs = logits.map(function (row) { return K.softmax(row); });
        var cap = capF > 3 ? Infinity : Math.ceil(capF * TOK.length * k / E), used = new Array(E).fill(0), loads = new Array(E).fill(0), dropped = 0;
        sel.forEach(function (s) { s.forEach(function (q) { loads[q[0]]++; if (used[q[0]] < cap) { used[q[0]]++; q.push(true); } else { q.push(false); dropped++; } }); });
        var f = loads.map(function (l) { return l / (TOK.length * k); }), P = [];
        for (var e3 = 0; e3 < E; e3++) P.push(sum(probs.map(function (pr) { return pr[e3]; })) / TOK.length);
        var aux = E * sum(f.map(function (fi, i) { return fi * P[i]; }));
        return { sel: sel, loads: loads, cap: cap, dropped: dropped, aux: aux, bias: bias };
      }
      function draw() {
        var R = route(), W = 700, H = 410, svg = newSvg(K, W, H), n = TOK.length;
        var th = (H - 30) / n, ty = function (i) { return 22 + i * th + th / 2; };
        var eh = (H - 30) / E, ey = function (e) { return 22 + e * eh + eh / 2; }, ex = 430;
        svg.appendChild(K.s("text", { x: 10, y: 12, class: "lbl-sm", text: "tokens" }));
        svg.appendChild(K.s("text", { x: 245, y: 12, "text-anchor": "middle", class: "lbl-sm", text: "router: top-" + k + " of softmax(x·W_r)" }));
        svg.appendChild(K.s("text", { x: ex, y: 12, class: "lbl-sm", text: "experts (FFNs)" }));
        svg.appendChild(K.s("text", { x: 548, y: 12, class: "lbl-sm", text: "load (tokens)" }));
        R.sel.forEach(function (s, i) {
          s.forEach(function (q) {
            var on = hov === null || hov === i;
            svg.appendChild(K.s("path", { d: "M100 " + ty(i) + " C 260 " + ty(i) + " 280 " + ey(q[0]) + " " + ex + " " + ey(q[0]), fill: "none", class: q[2] ? "s1" : "s5", "stroke-width": 0.8 + 3.5 * q[1], opacity: on ? (hov === i ? 0.95 : 0.35) : 0.06, "stroke-dasharray": q[2] ? null : "4 3" }));
          });
        });
        TOK.forEach(function (tk, i) {
          var g = K.s("g", { style: { cursor: "pointer" } });
          g.appendChild(K.s("rect", { x: 10, y: ty(i) - th / 2 + 1, width: 90, height: th - 2, rx: 4, class: hov === i ? "box-hl" : "box" }));
          g.appendChild(K.s("text", { x: 16, y: ty(i) + 4, class: "mono", style: { fontSize: "11px" }, text: tk[0] }));
          g.appendChild(K.s("text", { x: 96, y: ty(i) + 4, "text-anchor": "end", class: "lbl-sm", text: tk[1] }));
          g.addEventListener("mouseenter", function () { if (hov !== i) { hov = i; draw(); } });
          svg.appendChild(g);
        });
        var mxL = Math.max(Math.max.apply(null, R.loads), isFinite(R.cap) ? R.cap : 0, 1), lw = 130, lx = 548;
        for (var e = 0; e < E; e++) {
          svg.appendChild(K.s("rect", { x: ex, y: ey(e) - eh / 2 + 2, width: 100, height: eh - 4, rx: 5, class: "f" + (1 + (e % 6)), opacity: 0.85 }));
          svg.appendChild(K.s("text", { x: ex + 50, y: ey(e) + 4, "text-anchor": "middle", class: "t-white", style: { fontSize: "11px", fontWeight: 600 }, text: "expert " + e }));
          var l = R.loads[e], kept = Math.min(l, R.cap);
          svg.appendChild(K.s("rect", { x: lx, y: ey(e) - 7, width: kept / mxL * lw, height: 14, rx: 2, class: "f3", opacity: 0.8 }));
          if (l > kept) svg.appendChild(K.s("rect", { x: lx + kept / mxL * lw, y: ey(e) - 7, width: (l - kept) / mxL * lw, height: 14, rx: 2, class: "f5", opacity: 0.8 }));
          svg.appendChild(K.s("text", { x: lx + l / mxL * lw + 4, y: ey(e) + 4, class: "lbl-sm mono", text: l }));
        }
        if (isFinite(R.cap)) {
          svg.appendChild(K.s("line", { x1: lx + R.cap / mxL * lw, x2: lx + R.cap / mxL * lw, y1: 18, y2: H - 6, class: "ln-thin s5 dash" }));
          svg.appendChild(K.s("text", { x: lx + R.cap / mxL * lw + 3, y: H - 2, class: "lbl-sm", text: "capacity " + R.cap }));
        }
        K.clear(c.stage).appendChild(svg);
        var mean = n * k / E, mx = Math.max.apply(null, R.loads), M = MP[mkey], per = (M.total - M.shared) / M.E, act = M.shared + M.k * per;
        var html = statGrid([
          ["max / mean load", (mx / mean).toFixed(2) + "×", mx / mean > 1.5 ? "--red" : "--green"],
          ["dropped assignments", R.dropped + " / " + n * k, R.dropped ? "--red" : null],
          ["aux balance loss", R.aux.toFixed(3) + " (1.0 = perfect)"],
          [M.name + " total", fmtP(M.total)],
          [M.name + " active / token", fmtP(act), "--c3"],
          ["bf16 weights to hold", fmtGB(M.total * 2)],
        ]);
        if (hov !== null) html += "Token <b>“" + esc(TOK[hov][0]) + "”</b> → " + R.sel[hov].map(function (q) { return "expert " + q[0] + " (gate " + q[1].toFixed(2) + (q[2] ? "" : ", <b style='color:var(--c5)'>dropped</b>") + ")"; }).join(" + ") + ". Output = Σ gate × expert(x).<br>";
        html += K.tex("y = \\sum_{e\\in\\text{top-}k} g_e(x)\\,\\text{FFN}_e(x)") + ". " + M.name + ": " + M.E + " experts, " + M.k + " active; every token pays compute for only " + fmtP(act) + " params (≈ " + fmtP(2 * act) + " FLOPs/token) but the GPU(s) must store all " + fmtP(M.total) + ". " +
          "<span class='muted'>Skewed routers collapse onto a few experts (the rest never learn); fixes are an auxiliary balance loss (Switch/Mixtral) or DeepSeek-V3's bias trick (toggle it: a per-expert bias nudges <i>selection</i> only). With a capacity limit, overflow tokens are dropped (pink); modern inference engines run dropless. At serving time, expert parallelism spreads experts across GPUs — imbalance then means one GPU is the straggler.</span>";
        c.readout.innerHTML = html;
      }
      draw();
    },
  });

  // ================================================================ overfitting
  V.register("overfitting", {
    title: "Overfitting: when training loss keeps falling but validation loss turns up",
    desc: "Simulated char-level training runs (tokens/step = 64 × 256 like Karpathy's nanoGPT). Change model size, data size and dropout. Validation loss is what you actually care about; the gap tells you if the model is memorising.",
    render: function (c, p, K) {
      var NV = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8], DV = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 1e9];
      var ni = 4, di = 2, drop = 0.2, early = true, patience = 5, cur = 5000, seed = 1, STEPS = 5000, TPS = 16384;
      var fmtN = function (v) { return v >= 1e9 ? v / 1e9 + "B" : v >= 1e6 ? v / 1e6 + "M" : v / 1e3 + "K"; };
      c.controls.appendChild(K.slider({ label: "model params", min: 0, max: NV.length - 1, value: ni, fmt: function (v) { return fmtN(NV[v]); }, onInput: function (v) { ni = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "dataset tokens", min: 0, max: DV.length - 1, value: di, fmt: function (v) { return fmtN(DV[v]); }, onInput: function (v) { di = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "dropout", min: 0, max: 0.5, step: 0.05, value: drop, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { drop = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "early stopping (patience 5 evals)", value: early, onChange: function (v) { early = v; draw(); } }));
      var pl = player(c, K, function () { if (cur >= STEPS) return false; cur = Math.min(STEPS, cur + 100); draw(); }, 60, "Train");
      c.controls.appendChild(K.button("Reset", function () { pl.stop(); cur = 0; draw(); }));
      c.controls.appendChild(pl.btn);

      function curves() {
        var N = NV[ni], D = DV[di], r = K.rng(seed + ni * 13 + di * 7);
        var Linit = Math.log(65), Lbest = 1.0 + 0.9 * Math.pow(1e6 / N, 0.35), tau = 450 * Math.pow(N / 1e7, -0.15);
        var kap = Math.max(0.03, Math.sqrt(N / D) * (1 - 1.6 * drop)), dataPen = 0.05 * Math.pow(Math.max(0, Math.log10(N / D)), 1.5);
        var tr = [], va = [];
        for (var t = 0; t <= STEPS; t += 100) {
          var ep = t * TPS / D, base = Lbest + (Linit - Lbest) * Math.exp(-t / tau);
          tr.push([t, Math.max(0.05, base - kap * 0.18 * (1 - Math.exp(-ep / 60))) + 0.012 * K.randn(r)]);
          va.push([t, base + (dataPen * (1 - Math.exp(-t / tau))) + kap * 1.7e-5 * ep * ep + 0.015 * K.randn(r)]);
        }
        return { tr: tr, va: va, N: N, D: D };
      }
      function draw() {
        var C = curves(), shownN = Math.floor(cur / 100) + 1;
        var best = 0, stopAt = null, since = 0;
        for (var i = 1; i < C.va.length; i++) {
          if (C.va[i][1] < C.va[best][1]) { best = i; since = 0; } else since++;
          if (early && since >= patience && stopAt === null) { stopAt = i; break; }
        }
        var lastIdx = Math.min(shownN, stopAt !== null ? stopAt + 1 : C.va.length);
        var ch = K.chart({ w: 680, h: 320, x: [0, STEPS], y: [0, 4.4], xLabel: "training step (× 16,384 tokens)", yLabel: "loss (nats / char)" });
        var tr = C.tr.slice(0, lastIdx), va = C.va.slice(0, lastIdx);
        ch.g.appendChild(K.s("line", { x1: ch.sx(0), x2: ch.sx(STEPS), y1: ch.sy(Math.log(65)), y2: ch.sy(Math.log(65)), class: "ln-thin smuted dash" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(STEPS) - 4, y: ch.sy(Math.log(65)) - 4, "text-anchor": "end", class: "lbl-sm", text: "random init: ln 65 = 4.17" }));
        if (tr.length > 1) {
          ch.g.appendChild(K.s("path", { d: K.path(tr, ch.sx, ch.sy), class: "ln s1" }));
          ch.g.appendChild(K.s("path", { d: K.path(va, ch.sx, ch.sy), class: "ln s2" }));
        }
        var bestShown = 0; va.forEach(function (q, i) { if (q[1] < va[bestShown][1]) bestShown = i; });
        if (va.length > 1) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(va[bestShown][0]), x2: ch.sx(va[bestShown][0]), y1: ch.sy(0), y2: ch.sy(4.4), class: "ln-thin s3 dash" }));
          ch.g.appendChild(K.s("circle", { cx: ch.sx(va[bestShown][0]), cy: ch.sy(va[bestShown][1]), r: 5, class: "f3" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(va[bestShown][0]) + 5, y: ch.sy(0) - 6, class: "lbl-sm", text: "best checkpoint" }));
        }
        if (stopAt !== null && shownN > stopAt) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(C.va[stopAt][0]), x2: ch.sx(C.va[stopAt][0]), y1: ch.sy(0), y2: ch.sy(4.4), class: "ln s5" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(C.va[stopAt][0]) + 5, y: ch.sy(3.6), class: "lbl-sm", text: "early stop" }));
        }
        ch.g.appendChild(K.s("text", { x: ch.sx(STEPS * 0.62), y: ch.sy(3.95), class: "lbl", style: { fill: "var(--c1)" }, text: "— train" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(STEPS * 0.75), y: ch.sy(3.95), class: "lbl", style: { fill: "var(--c2)" }, text: "— validation" }));
        K.clear(c.stage).appendChild(ch.svg);
        var lt = tr[tr.length - 1][1], lv = va[va.length - 1][1], gap = lv - lt, epochs = tr[tr.length - 1][0] * TPS / C.D;
        var diag = va.length < 5 ? "Training…" :
          gap > 0.25 ? "<b style='color:var(--red)'>Overfitting:</b> the model is memorising the training text. Fixes: more data, more dropout/weight decay, a smaller model, or stop earlier (use the best checkpoint)." :
          lv > 2.2 ? "<b style='color:var(--amber)'>Underfitting:</b> both losses are high — the model is too small (or training too short). Make it bigger." :
          "<b style='color:var(--green)'>Healthy:</b> train ≈ val. With this much data relative to model size you could afford a bigger model.";
        c.readout.innerHTML = statGrid([
          ["step", tr[tr.length - 1][0]], ["epochs over data", epochs.toFixed(epochs < 10 ? 2 : 0)], ["train loss", lt.toFixed(3), "--c1"], ["val loss", lv.toFixed(3), "--c2"],
          ["gap val − train", gap.toFixed(3)], ["best val", va[bestShown][1].toFixed(3) + " @ " + va[bestShown][0]],
        ]) + diag + " <span class='muted'>params / tokens = " + (C.N / C.D).toFixed(C.N / C.D < 1 ? 3 : 1) + ". Karpathy's 10M-param model on ~1M characters of Shakespeare (the defaults) reaches val ≈ 1.47 and then starts to overfit; dropout 0.2 is what keeps it in check. LLM pre-training usually sees each token ≈ once, so overfitting is rarely the problem there — it bites in fine-tuning on small datasets. Curves are a stylised simulation, not a real run.</span>";
      }
      draw();
    },
  });

  // ================================================================ scaling-laws
  V.register("scaling-laws", {
    title: "Scaling laws: compute C ≈ 6·N·D, and how to spend it",
    desc: "Training FLOPs ≈ 6 × parameters × tokens (2 for the forward pass, 4 for backward). For a fixed compute budget there's a best model size — Chinchilla found ≈ 20 tokens per parameter. The curves use the Chinchilla loss fit.",
    render: function (c, p, K) {
      var PRE = {
        nanogpt: ["GPT-2 124M repro (10B tok)", 124e6, 10e9], gpt3: ["GPT-3 175B (300B tok)", 175e9, 300e9], chin: ["Chinchilla 70B (1.4T)", 70e9, 1.4e12],
        l2: ["Llama-2-7B (2T)", 7e9, 2e12], l3: ["Llama-3-8B (15T)", 8e9, 15e12], l405: ["Llama-3.1-405B (15.6T)", 405e9, 15.6e12],
      };
      var GPU = { h100: ["H100 (989 TFLOPS bf16)", 989e12, 2.5], a100: ["A100 (312 TFLOPS)", 312e12, 1.3], b200: ["B200 (~2250 TFLOPS)", 2250e12, 5], l4: ["L4 (121 TFLOPS)", 121e12, 0.7], t4: ["T4 (65 TFLOPS fp16)", 65e12, 0.35], m3: ["M3 Max (~28 TFLOPS)", 28e12, 0] };
      var key = p.preset || "l3", N = PRE[key][1], D = PRE[key][2], gpu = "h100", ng = 10, mfu = 0.4, price = 2.5;
      var nS, dS;
      c.controls.appendChild(K.select({ label: "preset", options: Object.keys(PRE).map(function (k) { return { value: k, label: PRE[k][0] }; }), value: key, onChange: function (v) { N = PRE[v][1]; D = PRE[v][2]; nS.set(Math.log10(N)); dS.set(Math.log10(D)); draw(); } }));
      nS = K.slider({ label: "params N", min: 7, max: 12, step: 0.05, value: Math.log10(N), fmt: function (v) { return fmtP(Math.pow(10, v)); }, onInput: function (v) { N = Math.pow(10, v); draw(); } });
      dS = K.slider({ label: "tokens D", min: 8, max: 14, step: 0.05, value: Math.log10(D), fmt: function (v) { return fmtP(Math.pow(10, v)); }, onInput: function (v) { D = Math.pow(10, v); draw(); } });
      c.controls.appendChild(nS); c.controls.appendChild(dS);
      c.controls.appendChild(K.button("Chinchilla: D = 20·N", function () { D = 20 * N; dS.set(Math.log10(D)); draw(); }));
      c.controls.appendChild(K.select({ label: "GPU", options: Object.keys(GPU).map(function (k) { return { value: k, label: GPU[k][0] }; }), value: gpu, onChange: function (v) { gpu = v; price = GPU[v][2]; pr.set(price); draw(); } }));
      c.controls.appendChild(K.slider({ label: "# GPUs", min: 0, max: 14, value: ng, fmt: function (v) { return Math.pow(2, v); }, onInput: function (v) { ng = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "MFU", min: 0.1, max: 0.7, step: 0.01, value: mfu, fmt: function (v) { return Math.round(v * 100) + "%"; }, onInput: function (v) { mfu = v; draw(); } }));
      var pr = K.number({ label: "$ / GPU-hour", value: price, min: 0, step: 0.1, onInput: function (v) { price = v; draw(); } });
      c.controls.appendChild(pr);

      var E0 = 1.69, A = 406.4, B = 410.7, al = 0.34, be = 0.28;
      function loss(n, d) { return E0 + A / Math.pow(n, al) + B / Math.pow(d, be); }
      function optN(C) { var best = null, bl = 1e9; for (var e = 6; e <= 13; e += 0.01) { var n = Math.pow(10, e), l = loss(n, C / (6 * n)); if (l < bl) { bl = l; best = n; } } return best; }
      function draw() {
        var C = 6 * N * D;
        var ch = K.chart({ w: 680, h: 320, x: [1e7, 1e12], y: [1.8, 4.2], xLog: true, xLabel: "model size N (parameters) — each curve is one compute budget", yLabel: "predicted loss", xFmt: fmtP, pad: { r: 24 } });
        [1e18, 1e20, 1e22, 1e24, 1e26].forEach(function (Cc) {
          var pts = []; for (var e = 7; e <= 12.001; e += 0.02) { var n = Math.pow(10, e), d = Cc / (6 * n); if (d < 1e6) continue; pts.push([n, loss(n, d)]); }
          pts = pts.filter(function (q) { return q[1] < 4.2; });
          if (pts.length > 1) {
            ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln-thin smuted" }));
            var lo = pts.reduce(function (a, b) { return b[1] < a[1] ? b : a; });
            ch.g.appendChild(K.s("text", { x: ch.sx(lo[0]), y: ch.sy(lo[1]) + 14, "text-anchor": "middle", class: "tick", text: "1e" + Math.round(Math.log10(Cc)) }));
          }
        });
        var front = []; for (var e2 = 17; e2 <= 27; e2 += 0.25) { var Cf = Math.pow(10, e2), no = optN(Cf); front.push([no, loss(no, Cf / (6 * no))]); }
        front = front.filter(function (q) { return q[0] >= 1e7 && q[0] <= 1e12 && q[1] < 4.2; });
        ch.g.appendChild(K.s("path", { d: K.path(front, ch.sx, ch.sy), class: "ln-thin s3 dash" }));
        var pts2 = []; for (var e3 = 7; e3 <= 12.001; e3 += 0.02) { var n3 = Math.pow(10, e3), d3 = C / (6 * n3); if (d3 < 1e6) continue; var l3 = loss(n3, d3); if (l3 < 4.2) pts2.push([n3, l3]); }
        if (pts2.length > 1) ch.g.appendChild(K.s("path", { d: K.path(pts2, ch.sx, ch.sy), class: "ln s1" }));
        var No = optN(C), L = loss(N, D);
        ch.g.appendChild(K.s("circle", { cx: ch.sx(No), cy: ch.sy(loss(No, C / (6 * No))), r: 5, class: "f3" }));
        if (L < 4.2) { ch.g.appendChild(K.s("circle", { cx: ch.sx(N), cy: ch.sy(L), r: 7, class: "f2" })); ch.g.appendChild(K.s("text", { x: ch.sx(N), y: ch.sy(L) + 22, "text-anchor": "middle", class: "lbl", text: "you" })); }
        ch.g.appendChild(K.s("text", { x: ch.sx(1.2e7), y: ch.sy(4.1), class: "lbl-sm", text: "green dashed = compute-optimal; grey = other FLOP budgets" }));
        K.clear(c.stage).appendChild(ch.svg);
        var G = GPU[gpu], n = Math.pow(2, ng), secs = C / (n * G[1] * mfu), gh = secs / 3600 * n;
        c.readout.innerHTML = statGrid([
          ["compute C = 6ND", C.toExponential(2) + " FLOPs", "--c1"],
          ["tokens / param", (D / N).toFixed(0) + (D / N > 25 ? " (over-trained)" : D / N < 15 ? " (under-trained)" : " (≈ optimal)")],
          ["optimal N for this C", fmtP(No) + " on " + fmtP(C / (6 * No)) + " tok", "--c3"],
          ["predicted loss", L.toFixed(3)],
          ["wall-clock on " + n + " GPU" + (n > 1 ? "s" : ""), fmtTime(secs), "--c2"],
          ["GPU-hours · cost", K.fmtShort(gh) + " · " + (price > 0 ? "$" + K.fmtShort(gh * price) : "—")],
        ]) +
          K.tex("C \\approx 6ND,\\quad t = \\frac{C}{n_{GPU}\\times \\text{peak FLOPS}\\times \\text{MFU}}") + " &nbsp; " + K.tex("L(N,D) = 1.69 + \\frac{406.4}{N^{0.34}} + \\frac{410.7}{D^{0.28}}") +
          " <span class='muted'>(Hoffmann et al. 2022). MFU (model FLOPs utilisation) of 35–50% is typical for well-tuned large runs. Llama-3-8B trains on ~1,900 tokens/param — far past compute-optimal — because a smaller model trained longer is <i>cheaper to serve</i>, and inference cost dominates over a model's lifetime.</span>";
      }
      draw();
    },
  });

  // ================================================================ training-memory
  V.register("training-memory", {
    title: "Why training needs ~16 bytes per parameter (and inference only 2)",
    desc: "Training keeps much more than the weights: gradients, Adam's two running averages (m, v), often an fp32 master copy, plus activations saved for the backward pass. Compare against real GPU / Mac memory.",
    render: function (c, p, K) {
      var MODES = {
        fp32: { name: "fp32 + Adam", w: 4, g: 4, ms: 0, m: 4, v: 4, act: 4 },
        mixed: { name: "mixed bf16 + fp32 master + Adam (standard)", w: 2, g: 2, ms: 4, m: 4, v: 4, act: 2 },
        adam8: { name: "bf16 + 8-bit Adam", w: 2, g: 2, ms: 4, m: 1, v: 1, act: 2 },
        purebf16: { name: "pure bf16 (bf16 Adam states, no master)", w: 2, g: 2, ms: 0, m: 2, v: 2, act: 2 },
        lora: { name: "LoRA r=16 (frozen bf16 base)", w: 2, g: 0, ms: 0, m: 0, v: 0, act: 2, lora: true },
        qlora: { name: "QLoRA r=16 (frozen 4-bit base)", w: 0.52, g: 0, ms: 0, m: 0, v: 0, act: 2, lora: true },
        infer: { name: "inference only (bf16 weights)", w: 2, g: 0, ms: 0, m: 0, v: 0, act: 0 },
      };
      var key = p.model || "llama31-8b", mode = p.mode || "mixed", b = 1, SL = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768], si = 3, flash = true, ckpt = false, shards = 1;
      c.controls.appendChild(K.select({ label: "model", options: Object.keys(MODELS).map(function (k) { return { value: k, label: MODELS[k].name }; }), value: key, onChange: function (v) { key = v; draw(); } }));
      c.controls.appendChild(K.select({ label: "recipe", options: Object.keys(MODES).map(function (k) { return { value: k, label: MODES[k].name }; }), value: mode, onChange: function (v) { mode = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "micro-batch", min: 1, max: 64, value: b, onInput: function (v) { b = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "seq len", min: 0, max: SL.length - 1, value: si, fmt: function (v) { return SL[v]; }, onInput: function (v) { si = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "FlashAttention", value: flash, onChange: function (v) { flash = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "activation checkpointing", value: ckpt, onChange: function (v) { ckpt = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "FSDP / ZeRO-3 GPUs", min: 1, max: 64, value: shards, onInput: function (v) { shards = v; draw(); } }));

      function loraParams(m, r) {
        var hd = m.hd, per = r * (m.d + m.h * hd) + 2 * r * (m.d + m.kv * hd) + r * (m.h * hd + m.d) + (m.style === "gpt2" ? 2 : 3) * r * (m.d + m.ffn);
        return per * m.L;
      }
      function draw() {
        var m = MODELS[key], M = MODES[mode], P = countParams(m).total, s = SL[si], h = m.d, a = m.h, L = m.L;
        var tp = M.lora ? loraParams(m, 16) : 0;
        var sh = function (x) { return x / shards; };
        var parts = [
          ["weights", sh(P * M.w + tp * 2), "f1"],
          ["gradients", sh(M.lora ? tp * 2 : P * M.g), "f2"],
          ["fp32 master", sh(M.lora ? tp * 4 : P * M.ms), "f4"],
          ["Adam m", sh(M.lora ? tp * 4 : P * M.m), "f3"],
          ["Adam v", sh(M.lora ? tp * 4 : P * M.v), "f6"],
        ];
        var act = 0;
        if (M.act) {
          var perLayer = s * b * h * (34 + (flash ? 0 : 5 * a * s / h)) * (M.act / 2);
          act = ckpt ? (2 * s * b * h * L * (M.act / 2) + perLayer) : perLayer * L;
          act += s * b * m.V * 4; // fp32 logits for the loss
        }
        parts.push(["activations", act, "f5"]);
        var tot = sum(parts.map(function (q) { return q[1]; }));
        var GP = [[16e9, "T4 · 16 GB Mac"], [24e9, "L4 24"], [36e9, "M3 Pro 36"], [80e9, "A100 / H100 80"], [128e9, "M-series 128"]];
        var mx = Math.max(tot, 128e9) * 1.08, W = 700, H = 170, svg = newSvg(K, W, H), sx = K.scale(0, mx, 10, W - 10);
        svg.appendChild(K.s("text", { x: 10, y: 14, class: "lbl", text: "Memory per GPU: " + fmtGB(tot) + (shards > 1 ? " (states sharded over " + shards + " GPUs)" : "") }));
        var x = 10;
        parts.forEach(function (q) {
          var w = sx(q[1]) - 10; if (w <= 0) return;
          svg.appendChild(K.s("rect", { x: x, y: 26, width: Math.max(0.5, w - 0.8), height: 40, rx: 3, class: q[2], opacity: 0.85 }));
          if (w > 70) svg.appendChild(K.s("text", { x: x + 5, y: 50, class: "t-white", style: { fontSize: "11px", fontWeight: 600 }, text: q[0] }));
          x += w;
        });
        GP.forEach(function (g, i) {
          var gx = sx(g[0]);
          svg.appendChild(K.s("line", { x1: gx, x2: gx, y1: 20, y2: 88 + (i % 2) * 16, class: "ln-thin sfg dash", opacity: 0.55 }));
          svg.appendChild(K.s("text", { x: gx, y: 99 + (i % 2) * 16, "text-anchor": "middle", class: "lbl-sm", text: g[1] }));
        });
        parts.forEach(function (q, i) {
          svg.appendChild(K.s("rect", { x: 10 + i * 114, y: 138, width: 10, height: 10, class: q[2] }));
          svg.appendChild(K.s("text", { x: 24 + i * 114, y: 147, class: "lbl-sm", text: q[0] + " " + fmtGB(q[1]) }));
        });
        K.clear(c.stage).appendChild(svg);
        var bpp = (M.w + M.g + M.ms + M.m + M.v);
        var fits = [["T4 16 GB", 16e9], ["Mac 16 GB", 16e9 * 0.7], ["Mac 36 GB", 36e9 * 0.7], ["A100 80 GB", 80e9], ["H100 80 GB", 80e9], ["Mac 128 GB", 128e9 * 0.75]].map(function (g) {
          return "<span class='tok' style='" + (tot <= g[1] ? "border-color:var(--green);color:var(--green)" : "border-color:var(--red);color:var(--red)") + "'>" + (tot <= g[1] ? "✓ " : "✗ ") + g[0] + "</span>";
        }).join(" ");
        c.readout.innerHTML = fits + "<br>" +
          (M.lora ? "LoRA: base " + fmtP(P) + " params frozen at " + M.w + " B/param; only <b>" + fmtP(tp) + "</b> adapter params (" + (tp / P * 100).toFixed(2) + "%) get grads + Adam states (16 B each). " :
            "Model/optimizer state: <b>" + bpp + " bytes/param</b> × " + fmtP(P) + " = " + fmtGB(P * bpp) + (bpp === 16 ? " — the classic rule: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m 4 + Adam v 4. " : ". ")) +
          (M.act ? "Activations ≈ " + K.tex("L\\cdot s\\cdot b\\cdot h\\,(34 + " + (flash ? "0" : "5as/h") + ")") + " bytes" + (ckpt ? " → with checkpointing only each layer's input is kept (≈2sbh per layer) and recomputed in backward (~33% more compute)" : "") + ", plus fp32 logits b·s·V. " : "") +
          "<span class='muted'>Mac unified memory: the GPU can use roughly 65–75% by default. Rules of thumb: inference ≈ 2 B/param, full fine-tune ≈ 16 B/param + activations, so an 8B model needs ~130 GB to fully fine-tune (multiple GPUs + FSDP) but ~6 GB to QLoRA. Activation estimate from Korthikanti et al. 2022 (approximate).</span>";
      }
      draw();
    },
  });

  // ================================================================ lora
  V.register("lora", {
    title: "LoRA: freeze W, learn a low-rank update B·A",
    desc: "Instead of updating a d×k weight matrix, learn two thin matrices B (d×r) and A (r×k); the update ΔW = B·A has rank ≤ r. For r = 16 on a 4096×4096 matrix that's 0.8% of the parameters.",
    render: function (c, p, K) {
      var RS = [1, 2, 4, 8, 16, 32, 64, 128, 256], ri = 4, alpha = 32, key = p.model || "llama31-8b", seed = 4;
      var tg = { q: true, k: true, v: true, o: true, gate: true, up: true, down: true };
      if (p.r) { var ix = RS.indexOf(p.r); if (ix >= 0) ri = ix; }
      c.controls.appendChild(K.select({ label: "model", options: Object.keys(MODELS).map(function (k) { return { value: k, label: MODELS[k].name }; }), value: key, onChange: function (v) { key = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "rank r", min: 0, max: RS.length - 1, value: ri, fmt: function (v) { return RS[v]; }, onInput: function (v) { ri = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "alpha α", min: 1, max: 128, value: alpha, onInput: function (v) { alpha = v; draw(); } }));
      Object.keys(tg).forEach(function (k) { c.controls.appendChild(K.toggle({ label: k + "_proj", value: tg[k], onChange: function (v) { tg[k] = v; draw(); } })); });
      c.controls.appendChild(K.button("New toy matrices", function () { seed++; draw(); }));

      function mats(m) {
        var hd = m.hd, g2 = m.style === "gpt2";
        return [["q", m.d, m.h * hd], ["k", m.d, m.kv * hd], ["v", m.d, m.kv * hd], ["o", m.h * hd, m.d], ["gate", g2 ? 0 : m.d, m.ffn], ["up", m.d, m.ffn], ["down", m.ffn, m.d]];
      }
      function trainable(m, r) { return m.L * sum(mats(m).map(function (x) { return tg[x[0]] && x[1] ? r * (x[1] + x[2]) : 0; })); }
      function draw() {
        var m = MODELS[key], r = RS[ri], P = countParams(m).total, tp = trainable(m, r);
        var W = 700, H = 400, svg = newSvg(K, W, H), n = 12, cs = 11, rt = Math.min(r, 8);
        var rg = K.rng(seed), Wm = [], Bm = [], Am = [];
        for (var i = 0; i < n; i++) { Wm.push([]); for (var j = 0; j < n; j++) Wm[i].push(K.randn(rg) * 0.6); }
        for (var i2 = 0; i2 < n; i2++) { Bm.push([]); for (var j2 = 0; j2 < rt; j2++) Bm[i2].push(K.randn(rg)); }
        for (var i3 = 0; i3 < rt; i3++) { Am.push([]); for (var j3 = 0; j3 < n; j3++) Am[i3].push(K.randn(rg)); }
        var dW = []; for (var a = 0; a < n; a++) { dW.push([]); for (var b = 0; b < n; b++) { var s = 0; for (var t = 0; t < rt; t++) s += Bm[a][t] * Am[t][b]; dW[a].push(s / Math.sqrt(rt)); } }
        function mat(M, ox, oy, frozen, label) {
          var rows = M.length, cols = M[0].length;
          for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) {
            var v = M[i][j];
            svg.appendChild(K.s("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 1, height: cs - 1, class: frozen ? "fmuted" : v >= 0 ? "f1" : "f2", "fill-opacity": frozen ? 0.15 + 0.5 * Math.min(1, Math.abs(v)) : 0.1 + 0.8 * Math.min(1, Math.abs(v) / 1.8) }));
          }
          svg.appendChild(K.s("text", { x: ox, y: oy - 8, class: "lbl-sm", text: label }));
        }
        var y0 = 40;
        mat(Wm, 10, y0, true, "W (frozen ❄)  d×k");
        svg.appendChild(K.s("text", { x: 10 + n * cs + 14, y: y0 + n * cs / 2 + 5, "text-anchor": "middle", class: "lbl", text: "+" }));
        var bxp = 10 + n * cs + 28;
        mat(Bm, bxp, y0, false, "B  d×" + rt);
        svg.appendChild(K.s("text", { x: bxp + rt * cs + 10, y: y0 + n * cs / 2 + 5, "text-anchor": "middle", class: "lbl", text: "·" }));
        var axp = bxp + rt * cs + 20;
        mat(Am, axp, y0, false, "A  " + rt + "×k");
        svg.appendChild(K.s("text", { x: axp + n * cs + 16, y: y0 + n * cs / 2 + 5, "text-anchor": "middle", class: "lbl", text: "=" }));
        var dxp = axp + n * cs + 32;
        mat(dW, dxp, y0, false, "ΔW = B·A (rank ≤ " + rt + ")");
        svg.appendChild(K.s("text", { x: 10, y: y0 + n * cs + 22, class: "lbl-sm", text: "toy 12×12 matrix" + (r > 8 ? " (showing rank " + rt + ")" : "") + ": trainable " + 2 * n * rt + " numbers instead of " + n * n }));
        // chart: trainable vs rank
        var ch0 = 40, cw = W - ch0 - 20, ctop = 236, chh = 120;
        var lo = Math.max(1e3, trainable(m, 1) / 2), hi = P * 2;
        var sx = K.scale(1, 256, ch0 + 34, ch0 + cw, true), sy = K.scale(lo, hi, ctop + chh, ctop, true);
        svg.appendChild(K.s("line", { x1: ch0 + 34, x2: ch0 + cw, y1: ctop + chh, y2: ctop + chh, class: "axis" }));
        svg.appendChild(K.s("line", { x1: ch0 + 34, x2: ch0 + 34, y1: ctop, y2: ctop + chh, class: "axis" }));
        K.logTicks(lo, hi).forEach(function (v) { svg.appendChild(K.s("text", { x: ch0 + 30, y: sy(v) + 3, "text-anchor": "end", class: "tick", text: fmtP(v) })); svg.appendChild(K.s("line", { x1: ch0 + 34, x2: ch0 + cw, y1: sy(v), y2: sy(v), class: "grid" })); });
        [1, 4, 16, 64, 256].forEach(function (v) { svg.appendChild(K.s("text", { x: sx(v), y: ctop + chh + 14, "text-anchor": "middle", class: "tick", text: v })); });
        svg.appendChild(K.s("text", { x: ch0 + 34 + (cw - 34) / 2, y: ctop + chh + 30, "text-anchor": "middle", class: "axis-label", text: "rank r" }));
        svg.appendChild(K.s("line", { x1: ch0 + 34, x2: ch0 + cw, y1: sy(P), y2: sy(P), class: "ln-thin s5 dash" }));
        svg.appendChild(K.s("text", { x: ch0 + cw - 4, y: sy(P) - 4, "text-anchor": "end", class: "lbl-sm", text: "full model " + fmtP(P) }));
        svg.appendChild(K.s("path", { d: K.path(RS.map(function (q) { return [q, Math.max(lo, trainable(m, q))]; }), sx, sy), class: "ln s1" }));
        if (tp > 0) svg.appendChild(K.s("circle", { cx: sx(r), cy: sy(tp), r: 5, class: "f2" }));
        svg.appendChild(K.s("text", { x: ch0 + 38, y: ctop - 8, class: "lbl-sm", text: "trainable params (log–log)" }));
        K.clear(c.stage).appendChild(svg);
        var ex = mats(m)[0];
        c.readout.innerHTML = statGrid([
          ["trainable params", fmtP(tp) + " (" + (tp / P * 100).toFixed(2) + "%)", "--c1"],
          ["adapter file (bf16)", fmtGB(tp * 2)],
          ["full fine-tune memory", "≈ " + fmtGB(P * 16)],
          ["LoRA memory", "≈ " + fmtGB(P * 2 + tp * 16)],
          ["QLoRA memory", "≈ " + fmtGB(P * 0.52 + tp * 16), "--c3"],
          ["scale α / r", (alpha / r).toFixed(2)],
        ]) +
          K.tex("h = Wx + \\frac{\\alpha}{r}\\,B A x") + ". One q_proj of " + m.name + ": full " + K.tex(ex[1] + "\\times" + ex[2] + " = " + fmtP(ex[1] * ex[2]).replace(/([KMB])/, "\\text{$1}")) +
          " vs LoRA " + K.tex("r(d+k) = " + r + "\\times" + (ex[1] + ex[2]) + " = " + fmtP(r * (ex[1] + ex[2])).replace(/([KMB])/, "\\text{$1}")) + ". " +
          "<span class='muted'>B starts at zero, so training begins exactly at the base model. After training, merge W' = W + (α/r)BA for zero extra inference cost — or keep adapters separate and serve many of them on one base model (multi-LoRA serving in vLLM/SGLang). Memory figures exclude activations.</span>";
      }
      draw();
    },
  });
})();
