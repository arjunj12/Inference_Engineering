/* math.js — interactive explainers for the math track. */
(function () {
  "use strict";
  var V = window.Viz;

  // ---------------------------------------------------------------- softmax + temperature + top-k/top-p
  V.register("softmax-temp", {
    title: "From scores (logits) to probabilities: softmax, temperature, top-k, top-p",
    desc: "A language model outputs one score per possible next token. Softmax turns scores into probabilities. Temperature sharpens or flattens them; top-k / top-p cut the tail before sampling.",
    render: function (c, p, K) {
      var toks = p.tokens || ["cat", "dog", "mat", "hat", "car", "sky", "the", "ran"];
      var logits = (p.logits || [3.1, 2.4, 2.0, 1.1, 0.4, -0.3, -0.8, -1.5]).slice();
      var T = 1, topk = toks.length, topp = 1, counts = toks.map(function () { return 0; }), nS = 0;
      var r = K.rng(7);
      c.controls.appendChild(K.slider({ label: "Temperature T", min: 0.05, max: 3, step: 0.05, value: 1, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { T = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "top-k", min: 1, max: toks.length, value: toks.length, onInput: function (v) { topk = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "top-p", min: 0.05, max: 1, step: 0.05, value: 1, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { topp = v; draw(); } }));
      c.controls.appendChild(K.button("Sample ×100", function () { sample(100); }));
      c.controls.appendChild(K.button("Reset counts", function () { counts = counts.map(function () { return 0; }); nS = 0; draw(); }));
      function probs() {
        var pr = K.softmax(logits, T);
        var idx = pr.map(function (_, i) { return i; }).sort(function (a, b) { return pr[b] - pr[a]; });
        var keep = {}, cum = 0;
        for (var j = 0; j < idx.length; j++) {
          if (j >= topk) break;
          keep[idx[j]] = 1; cum += pr[idx[j]];
          if (cum >= topp) break;
        }
        var z = 0; pr.forEach(function (v, i) { if (keep[i]) z += v; });
        return { raw: pr, fin: pr.map(function (v, i) { return keep[i] ? v / z : 0; }) };
      }
      function sample(n) {
        var f = probs().fin;
        for (var s = 0; s < n; s++) {
          var u = r(), acc = 0;
          for (var i = 0; i < f.length; i++) { acc += f[i]; if (u <= acc) { counts[i]++; break; } }
        }
        nS += n; draw();
      }
      function draw() {
        K.clear(c.stage);
        var pr = probs(), W = 660, H = 290;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var bw = (W - 80) / toks.length;
        var sl = K.scale(-3, 5, 110, 10), sp = K.scale(0, 1, 270, 140);
        svg.appendChild(K.s("text", { x: 4, y: 20, class: "lbl-sm", text: "logits (drag)" }));
        svg.appendChild(K.s("text", { x: 4, y: 150, class: "lbl-sm", text: "probability" }));
        svg.appendChild(K.s("line", { x1: 70, x2: W, y1: sl(0), y2: sl(0), class: "axis" }));
        toks.forEach(function (t, i) {
          var x = 76 + i * bw;
          // logit bar (draggable)
          var y0 = sl(0), y1 = sl(logits[i]);
          var bar = K.s("rect", { x: x + 6, y: Math.min(y0, y1), width: bw - 12, height: Math.max(1, Math.abs(y1 - y0)), class: "f1", opacity: 0.8, style: { cursor: "ns-resize" } });
          svg.appendChild(bar);
          svg.appendChild(K.s("text", { x: x + bw / 2, y: Math.min(y0, y1) - 3, "text-anchor": "middle", class: "lbl-sm mono", text: logits[i].toFixed(1) }));
          (function (i) {
            var drag = function (ev) {
              var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
              var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
              logits[i] = +K.clamp(sl.invert(loc.y), -3, 5).toFixed(1); draw();
            };
            bar.addEventListener("pointerdown", function (e) {
              e.preventDefault();
              var mv = function (ev) { drag(ev); }, up = function () { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
              window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
            });
          })(i);
          // prob bars
          var pf = pr.fin[i], praw = pr.raw[i];
          svg.appendChild(K.s("rect", { x: x + 6, y: sp(praw), width: bw - 12, height: 270 - sp(praw), class: "fline" }));
          svg.appendChild(K.s("rect", { x: x + 6, y: sp(pf), width: bw - 12, height: 270 - sp(pf), class: pf > 0 ? "f3" : "fmuted", opacity: 0.85 }));
          if (nS) {
            var emp = counts[i] / nS;
            svg.appendChild(K.s("line", { x1: x + 4, x2: x + bw - 4, y1: sp(emp), y2: sp(emp), class: "s2", "stroke-width": 3 }));
          }
          svg.appendChild(K.s("text", { x: x + bw / 2, y: sp(pf) - 4, "text-anchor": "middle", class: "lbl-sm mono", text: (pf * 100).toFixed(0) + "%" }));
          svg.appendChild(K.s("text", { x: x + bw / 2, y: 286, "text-anchor": "middle", class: "lbl", text: t }));
        });
        c.stage.appendChild(svg);
        var ent = -pr.fin.reduce(function (a, v) { return a + (v > 0 ? v * Math.log(v) : 0); }, 0);
        c.readout.innerHTML = K.tex("p_i = \\dfrac{e^{z_i/T}}{\\sum_j e^{z_j/T}}") +
          " &nbsp; Green = final sampling probabilities (grey ghost = before top-k/top-p). " +
          (nS ? "<span style='color:var(--c2)'>Orange ticks</span> = empirical frequency from " + nS + " samples. " : "") +
          "Entropy (uncertainty) = <b>" + ent.toFixed(2) + " nats</b>. Try T→0 (greedy/argmax) and T=2 (chaotic).";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- derivative / tangent
  V.register("derivative", {
    title: "A derivative is just the slope of the tangent line",
    desc: "Drag the point (or use the slider). The derivative tells you: if I nudge x a tiny bit, how much does f(x) change? Gradient descent only ever needs this number.",
    render: function (c, p, K) {
      var F = {
        "x²": { f: function (x) { return x * x; }, d: function (x) { return 2 * x; }, y: [-1, 9], d2: "2x" },
        "sin(x)": { f: Math.sin, d: Math.cos, y: [-1.5, 1.5], d2: "cos x" },
        "eˣ": { f: Math.exp, d: Math.exp, y: [-0.5, 12], d2: "e^x" },
        "sigmoid(x)": { f: function (x) { return 1 / (1 + Math.exp(-x)); }, d: function (x) { var s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, y: [-0.2, 1.2], d2: "\\sigma(x)(1-\\sigma(x))" },
        "ReLU(x)": { f: function (x) { return Math.max(0, x); }, d: function (x) { return x > 0 ? 1 : 0; }, y: [-0.5, 3], d2: "1 \\text{ if } x>0 \\text{ else } 0" },
        "x³ − 3x": { f: function (x) { return x * x * x - 3 * x; }, d: function (x) { return 3 * x * x - 3; }, y: [-5, 5], d2: "3x^2-3" },
      };
      var name = p.fn || "x²", x0 = 1, h = 0.5;
      c.controls.appendChild(K.select({ label: "f(x) =", options: Object.keys(F), value: name, onChange: function (v) { name = v; draw(); } }));
      var sx = K.slider({ label: "x", min: -3, max: 3, step: 0.01, value: x0, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { x0 = v; draw(); } });
      c.controls.appendChild(sx);
      c.controls.appendChild(K.slider({ label: "nudge h", min: 0.01, max: 1.5, step: 0.01, value: h, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { h = v; draw(); } }));
      function draw() {
        var fn = F[name];
        var ch = K.chart({ w: 640, h: 300, x: [-3, 3], y: fn.y, xLabel: "x", yLabel: "f(x)" });
        var pts = []; for (var x = -3; x <= 3.001; x += 0.02) pts.push([x, fn.f(x)]);
        ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln s1" }));
        var y0 = fn.f(x0), m = fn.d(x0);
        var tl = [[-3, y0 + m * (-3 - x0)], [3, y0 + m * (3 - x0)]];
        ch.g.appendChild(K.s("path", { d: K.path(tl, ch.sx, ch.sy), class: "ln s2" }));
        // secant (finite difference)
        var y1 = fn.f(x0 + h), ms = (y1 - y0) / h;
        var sl = [[-3, y0 + ms * (-3 - x0)], [3, y0 + ms * (3 - x0)]];
        ch.g.appendChild(K.s("path", { d: K.path(sl, ch.sx, ch.sy), class: "ln-thin s4 dash" }));
        ch.g.appendChild(K.s("circle", { cx: ch.sx(x0 + h), cy: ch.sy(y1), r: 4, class: "f4" }));
        var dot = K.s("circle", { cx: ch.sx(x0), cy: ch.sy(y0), r: 7, class: "f2", style: { cursor: "grab" } });
        ch.g.appendChild(dot);
        dot.addEventListener("pointerdown", function (e) {
          e.preventDefault();
          var svg = ch.svg;
          var mv = function (ev) {
            var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
            var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
            x0 = K.clamp(ch.sx.invert(loc.x), -3, 3); sx.set(x0); draw();
          };
          var up = function () { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
          window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
        });
        K.clear(c.stage).appendChild(ch.svg);
        c.readout.innerHTML = "At x = <b>" + x0.toFixed(2) + "</b>: slope (derivative) " + K.tex("f'(x)=" + fn.d2) + " = <b style='color:var(--c2)'>" + m.toFixed(3) +
          "</b>. &nbsp;Finite-difference estimate " + K.tex("\\frac{f(x+h)-f(x)}{h}") + " = <b style='color:var(--c4)'>" + ms.toFixed(3) + "</b> — shrink h and watch it converge. " +
          (m > 0 ? "Positive slope ⇒ to <i>decrease</i> f, move x <b>left</b>." : m < 0 ? "Negative slope ⇒ to decrease f, move x <b>right</b>." : "Slope 0 ⇒ flat: a minimum, maximum or plateau.");
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- gradient descent 1D
  V.register("gradient-descent", {
    title: "Gradient descent: walk downhill using only the slope",
    desc: "The loss is how wrong the model is. Each step: compute slope, move a little in the opposite direction. The learning rate is the step size — too small is slow, too big explodes.",
    render: function (c, p, K) {
      var f = function (x) { return 0.25 * x * x * x * x - 1.2 * x * x + 0.35 * x + 2; };
      var df = function (x) { return x * x * x - 2.4 * x + 0.35; };
      var lr = p.lr || 0.1, x = p.x0 !== undefined ? p.x0 : 2.6, hist = [x], timer = null;
      c.controls.appendChild(K.slider({ label: "learning rate η", min: 0.005, max: 0.9, step: 0.005, value: lr, fmt: function (v) { return v.toFixed(3); }, onInput: function (v) { lr = v; } }));
      var start = K.slider({ label: "start x", min: -2.8, max: 2.8, step: 0.1, value: x, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { stop(); x = v; hist = [x]; draw(); } });
      c.controls.appendChild(start);
      c.controls.appendChild(K.button("Step", function () { step(); }));
      var play = K.button("▶ Run", function () { if (timer) stop(); else { timer = setInterval(function () { step(); if (hist.length > 80) stop(); }, 140); play.textContent = "⏸ Pause"; } }, "primary");
      c.controls.appendChild(play);
      c.controls.appendChild(K.button("Reset", function () { stop(); x = start.get(); hist = [x]; draw(); }));
      function stop() { clearInterval(timer); timer = null; play.textContent = "▶ Run"; }
      function step() { x = x - lr * df(x); if (!isFinite(x) || Math.abs(x) > 50) { x = Math.sign(x) * 50; stop(); } hist.push(x); draw(); }
      function draw() {
        var ch = K.chart({ w: 640, h: 300, x: [-3, 3], y: [0, 6], xLabel: "parameter w", yLabel: "loss L(w)" });
        var pts = []; for (var t = -3; t <= 3.001; t += 0.02) pts.push([t, f(t)]);
        ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln s1" }));
        var vis = hist.filter(function (v) { return Math.abs(v) <= 3; });
        if (vis.length > 1) ch.g.appendChild(K.s("path", { d: K.path(vis.map(function (v) { return [v, f(v)]; }), ch.sx, ch.sy), class: "ln-thin s2" }));
        vis.forEach(function (v, i) { ch.g.appendChild(K.s("circle", { cx: ch.sx(v), cy: ch.sy(f(v)), r: i === vis.length - 1 ? 7 : 3, class: "f2", opacity: i === vis.length - 1 ? 1 : 0.45 })); });
        K.clear(c.stage).appendChild(ch.svg);
        var out = Math.abs(x) > 3;
        c.readout.innerHTML = "Update rule " + K.tex("w \\leftarrow w - \\eta\\, \\frac{dL}{dw}") + " &nbsp; step <b>" + (hist.length - 1) + "</b>, w = <b>" + x.toFixed(3) + "</b>, loss = <b>" + (out ? "∞ (diverged!)" : f(x).toFixed(3)) +
          "</b>, slope = " + (out ? "—" : df(x).toFixed(3)) + ". Notice two valleys: where you start decides which minimum you find (a <i>local</i> vs <i>global</i> minimum). Try η = 0.6+.";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- vectors & dot product
  V.register("vectors-dot", {
    title: "Vectors & the dot product: “how much do these point the same way?”",
    desc: "Drag the arrow tips. The dot product is big when vectors align, zero when perpendicular, negative when opposed. Attention uses exactly this to score how relevant one token is to another.",
    render: function (c, p, K) {
      var a = [2, 1], b = [1, 2.2];
      var S = 300, sc = K.scale(-3.2, 3.2, 20, S - 20), scy = K.scale(-3.2, 3.2, S - 20, 20);
      function draw() {
        var svg = K.s("svg", { viewBox: "0 0 " + S + " " + S, class: "viz-svg", style: { maxWidth: "360px" } });
        for (var i = -3; i <= 3; i++) {
          svg.appendChild(K.s("line", { x1: sc(i), x2: sc(i), y1: 20, y2: S - 20, class: "grid" }));
          svg.appendChild(K.s("line", { y1: scy(i), y2: scy(i), x1: 20, x2: S - 20, class: "grid" }));
        }
        svg.appendChild(K.s("line", { x1: sc(0), x2: sc(0), y1: 20, y2: S - 20, class: "axis" }));
        svg.appendChild(K.s("line", { y1: scy(0), y2: scy(0), x1: 20, x2: S - 20, class: "axis" }));
        // projection of b onto a
        var na = Math.hypot(a[0], a[1]), nb = Math.hypot(b[0], b[1]);
        var dot = a[0] * b[0] + a[1] * b[1];
        if (na > 0.01) {
          var k = dot / (na * na), pr = [a[0] * k, a[1] * k];
          svg.appendChild(K.s("line", { x1: sc(b[0]), y1: scy(b[1]), x2: sc(pr[0]), y2: scy(pr[1]), class: "ln-thin smuted dash" }));
          svg.appendChild(K.s("line", { x1: sc(0), y1: scy(0), x2: sc(pr[0]), y2: scy(pr[1]), class: "s3", "stroke-width": 6, opacity: 0.45 }));
        }
        [[a, "s1", "f1", "a"], [b, "s2", "f2", "b"]].forEach(function (v, idx) {
          svg.appendChild(K.s("line", { x1: sc(0), y1: scy(0), x2: sc(v[0][0]), y2: scy(v[0][1]), class: v[1], "stroke-width": 3 }));
          var tip = K.s("circle", { cx: sc(v[0][0]), cy: scy(v[0][1]), r: 8, class: v[2], style: { cursor: "grab" } });
          svg.appendChild(tip);
          svg.appendChild(K.s("text", { x: sc(v[0][0]) + 10, y: scy(v[0][1]) - 8, class: "lbl", text: v[3] + " = [" + v[0][0].toFixed(1) + ", " + v[0][1].toFixed(1) + "]" }));
          tip.addEventListener("pointerdown", function (e) {
            e.preventDefault();
            var mv = function (ev) {
              var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
              var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
              var tgt = idx === 0 ? a : b;
              tgt[0] = +K.clamp(sc.invert(loc.x), -3, 3).toFixed(1); tgt[1] = +K.clamp(scy.invert(loc.y), -3, 3).toFixed(1); draw();
            };
            var up = function () { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
            window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
          });
        });
        K.clear(c.stage).appendChild(svg);
        var cos = dot / (na * nb || 1), ang = Math.acos(K.clamp(cos, -1, 1)) * 180 / Math.PI;
        c.readout.innerHTML =
          K.tex("a\\cdot b = a_1b_1 + a_2b_2 = " + a[0].toFixed(1) + "\\times" + b[0].toFixed(1) + " + " + a[1].toFixed(1) + "\\times" + b[1].toFixed(1) + " = \\mathbf{" + dot.toFixed(2) + "}") +
          "<br>" + K.tex("\\|a\\|\\,\\|b\\|\\cos\\theta = " + na.toFixed(2) + "\\times" + nb.toFixed(2) + "\\times\\cos(" + ang.toFixed(0) + "^\\circ)") +
          " &nbsp; cosine similarity = <b>" + cos.toFixed(3) + "</b> " + (cos > 0.9 ? "(nearly same direction)" : cos < -0.9 ? "(opposite)" : Math.abs(cos) < 0.1 ? "(perpendicular: unrelated)" : "") +
          "<br><span class='muted'>Green bar = projection of b onto a (“how much of b lies along a”).</span>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- matmul
  V.register("matmul", {
    title: "Matrix multiplication = many dot products at once",
    desc: "Each output cell is the dot product of a row of A with a column of B. Hover (or step) through output cells. Inner dimensions must match: (m×k)·(k×n) → (m×n).",
    render: function (c, p, K) {
      var m = p.m || 2, k = p.k || 3, n = p.n || 4;
      var r = K.rng(3);
      function rnd() { return Math.round((r() * 4 - 2)); }
      var A, B;
      function init() {
        A = []; B = [];
        for (var i = 0; i < m; i++) { A.push([]); for (var j = 0; j < k; j++) A[i].push(rnd()); }
        for (var i2 = 0; i2 < k; i2++) { B.push([]); for (var j2 = 0; j2 < n; j2++) B[i2].push(rnd()); }
      }
      init();
      var sel = [0, 0];
      c.controls.appendChild(K.slider({ label: "m (rows of A)", min: 1, max: 5, value: m, onInput: function (v) { m = v; init(); sel = [0, 0]; draw(); } }));
      c.controls.appendChild(K.slider({ label: "k (shared)", min: 1, max: 6, value: k, onInput: function (v) { k = v; init(); sel = [0, 0]; draw(); } }));
      c.controls.appendChild(K.slider({ label: "n (cols of B)", min: 1, max: 6, value: n, onInput: function (v) { n = v; init(); sel = [0, 0]; draw(); } }));
      c.controls.appendChild(K.button("Next cell →", function () { sel[1]++; if (sel[1] >= n) { sel[1] = 0; sel[0] = (sel[0] + 1) % m; } draw(); }));
      function draw() {
        var cs = 34, gap = 40;
        var ax = 10, ay = 20 + k * cs + 10, bx = ax + k * cs + gap, by = 20, cx = bx, cy = ay;
        var W = cx + n * cs + 20, H = cy + m * cs + 20;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg", style: { maxWidth: Math.min(W * 1.5, 640) + "px" } });
        function cell(x, y, v, hl, onEnter) {
          var g = K.s("g");
          var rc = K.s("rect", { x: x, y: y, width: cs - 3, height: cs - 3, rx: 5, class: hl ? "box-hl" : "box" });
          g.appendChild(rc);
          g.appendChild(K.s("text", { x: x + cs / 2 - 1.5, y: y + cs / 2 + 3, "text-anchor": "middle", class: "lbl mono", text: v }));
          if (onEnter) { g.style.cursor = "pointer"; g.addEventListener("mouseenter", onEnter); g.addEventListener("click", onEnter); }
          svg.appendChild(g);
        }
        for (var i = 0; i < m; i++) for (var j = 0; j < k; j++) cell(ax + j * cs, ay + i * cs, A[i][j], i === sel[0]);
        for (var i2 = 0; i2 < k; i2++) for (var j2 = 0; j2 < n; j2++) cell(bx + j2 * cs, by + i2 * cs, B[i2][j2], j2 === sel[1]);
        for (var i3 = 0; i3 < m; i3++) for (var j3 = 0; j3 < n; j3++) {
          var v = 0; for (var t = 0; t < k; t++) v += A[i3][t] * B[t][j3];
          (function (i3, j3) { cell(cx + j3 * cs, cy + i3 * cs, v, i3 === sel[0] && j3 === sel[1], function () { sel = [i3, j3]; draw(); }); })(i3, j3);
        }
        svg.appendChild(K.s("text", { x: ax, y: ay - 6, class: "lbl-sm", text: "A (" + m + "×" + k + ")" }));
        svg.appendChild(K.s("text", { x: bx, y: by - 6, class: "lbl-sm", text: "B (" + k + "×" + n + ")" }));
        svg.appendChild(K.s("text", { x: cx + n * cs + 4, y: cy + m * cs / 2, class: "lbl-sm", text: "C" }));
        K.clear(c.stage).appendChild(svg);
        var terms = [], sum = 0;
        for (var q = 0; q < k; q++) { terms.push("(" + A[sel[0]][q] + ")(" + B[q][sel[1]] + ")"); sum += A[sel[0]][q] * B[q][sel[1]]; }
        c.readout.innerHTML = "C[" + sel[0] + "][" + sel[1] + "] = row " + sel[0] + " of A · column " + sel[1] + " of B = " + terms.join(" + ") + " = <b>" + sum + "</b><br>" +
          "Work: " + K.tex("m\\times n") + " outputs × " + K.tex("k") + " multiply-adds each = " + K.tex("2mnk = " + 2 * m * n * k) + " FLOPs. " +
          "<span class='muted'>A 7B-parameter model does ≈ 2 × 7B = 14 GFLOPs of this per generated token.</span>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- activation functions
  V.register("activations", {
    title: "Activation functions: the bend that makes networks powerful",
    desc: "Without a non-linear bend between layers, stacking layers collapses into one big linear function. Toggle functions to compare shapes.",
    render: function (c, p, K) {
      var fns = {
        ReLU: function (x) { return Math.max(0, x); },
        GELU: function (x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); },
        "SiLU / Swish": function (x) { return x / (1 + Math.exp(-x)); },
        Sigmoid: function (x) { return 1 / (1 + Math.exp(-x)); },
        Tanh: Math.tanh,
      };
      var on = { ReLU: true, GELU: true, "SiLU / Swish": false, Sigmoid: false, Tanh: false };
      var cls = ["s1", "s2", "s3", "s4", "s5"];
      Object.keys(fns).forEach(function (k) { c.controls.appendChild(K.toggle({ label: k, value: on[k], onChange: function (v) { on[k] = v; draw(); } })); });
      function draw() {
        var ch = K.chart({ w: 640, h: 280, x: [-4, 4], y: [-1.5, 4], xLabel: "input", yLabel: "output" });
        Object.keys(fns).forEach(function (k, i) {
          if (!on[k]) return;
          var pts = []; for (var x = -4; x <= 4.001; x += 0.04) pts.push([x, fns[k](x)]);
          ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln " + cls[i] }));
          ch.g.appendChild(K.s("text", { x: ch.sx(3.2), y: ch.sy(fns[k](3.2)) - 6, class: "lbl-sm", text: k }));
        });
        K.clear(c.stage).appendChild(ch.svg);
        c.readout.innerHTML = "GPT-2 uses <b>GELU</b>; Llama/Qwen use <b>SiLU</b> inside a gated “SwiGLU” MLP. ReLU is the simplest (just clip negatives). Sigmoid/Tanh squash into a range and were common in older nets.";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- log / exp / cross-entropy
  V.register("neg-log", {
    title: "Why −log(p) is the perfect “surprise” score (cross-entropy loss)",
    desc: "The loss for one prediction is −log(probability the model gave to the correct answer). Drag p: confident & right → ~0 loss; confident & wrong → huge loss.",
    render: function (c, p, K) {
      var pr = 0.3, V_ = p.vocab || 65;
      c.controls.appendChild(K.slider({ label: "p(correct token)", min: 0.001, max: 1, step: 0.001, value: pr, fmt: function (v) { return v.toFixed(3); }, onInput: function (v) { pr = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "vocab size", min: 2, max: 50000, step: 1, value: V_, onInput: function (v) { V_ = v; draw(); } }));
      function draw() {
        var ch = K.chart({ w: 640, h: 280, x: [0, 1], y: [0, 7], xLabel: "probability assigned to the correct answer", yLabel: "loss = −ln p" });
        var pts = []; for (var x = 0.001; x <= 1.0001; x += 0.002) pts.push([x, Math.min(7, -Math.log(x))]);
        ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln s1" }));
        var u = 1 / V_;
        ch.g.appendChild(K.s("line", { x1: ch.sx(u), x2: ch.sx(u), y1: ch.sy(0), y2: ch.sy(Math.min(7, -Math.log(u))), class: "ln-thin s4 dash" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(u) + 6, y: ch.sy(Math.min(6.6, -Math.log(u))), class: "lbl-sm", text: "random guess 1/V" }));
        ch.g.appendChild(K.s("circle", { cx: ch.sx(pr), cy: ch.sy(Math.min(7, -Math.log(pr))), r: 7, class: "f2" }));
        K.clear(c.stage).appendChild(ch.svg);
        c.readout.innerHTML = "loss = " + K.tex("-\\ln(" + pr.toFixed(3) + ") = " + (-Math.log(pr)).toFixed(3)) +
          ". A freshly initialised model guesses uniformly, so its loss should start near " + K.tex("\\ln V = \\ln " + V_ + " \\approx " + Math.log(V_).toFixed(2)) +
          " — Karpathy uses this as a sanity check. Perplexity " + K.tex("= e^{\\text{loss}}") + " = <b>" + Math.exp(-Math.log(pr)).toFixed(1) + "</b> ≈ “how many tokens the model is torn between”.";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- chain rule / backprop on a tiny graph
  V.register("backprop-graph", {
    title: "Backpropagation on a tiny graph (micrograd-style)",
    desc: "Forward pass computes values left→right. Backward pass computes gradients right→left using the chain rule: each node multiplies the gradient coming in by its local derivative.",
    render: function (c, p, K) {
      var vals = { a: 2, b: -3, c: 10, f: -2 };
      var stage = 0; // 0 = forward only, 1..5 backward steps
      ["a", "b", "c", "f"].forEach(function (k) {
        c.controls.appendChild(K.slider({ label: k, min: -5, max: 10, step: 0.5, value: vals[k], fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { vals[k] = v; draw(); } }));
      });
      c.controls.appendChild(K.button("◀ Back", function () { stage = Math.max(0, stage - 1); draw(); }));
      c.controls.appendChild(K.button("Backward step ▶", function () { stage = Math.min(4, stage + 1); draw(); }, "primary"));
      function draw() {
        var a = vals.a, b = vals.b, cc = vals.c, f = vals.f;
        var e = a * b, d = e + cc, L = d * f;
        var g = { L: 1, d: f, f: d, e: f, c: f, a: f * b, b: f * a };
        var shown = [["L"], ["L", "d", "f"], ["L", "d", "f", "e", "c"], ["L", "d", "f", "e", "c", "a", "b"], ["L", "d", "f", "e", "c", "a", "b"]][Math.max(0, stage - 1)] || [];
        if (stage === 0) shown = [];
        var N = {
          a: [40, 50, "a", a], b: [40, 150, "b", b], e: [220, 100, "e = a·b", e], c: [220, 200, "c", cc],
          d: [400, 150, "d = e + c", d], f: [400, 250, "f", f], L: [580, 200, "L = d·f", L],
        };
        var edges = [["a", "e"], ["b", "e"], ["e", "d"], ["c", "d"], ["d", "L"], ["f", "L"]];
        var svg = K.s("svg", { viewBox: "0 0 680 290", class: "viz-svg" });
        edges.forEach(function (ed) {
          var s = N[ed[0]], t = N[ed[1]];
          svg.appendChild(K.s("line", { x1: s[0] + 55, y1: s[1], x2: t[0] - 55, y2: t[1], class: "ln-thin sfg", opacity: 0.35 }));
        });
        Object.keys(N).forEach(function (k) {
          var n = N[k], has = shown.indexOf(k) >= 0;
          svg.appendChild(K.s("rect", { x: n[0] - 55, y: n[1] - 26, width: 110, height: 52, rx: 9, class: has ? "box-hl" : "box" }));
          svg.appendChild(K.s("text", { x: n[0], y: n[1] - 8, "text-anchor": "middle", class: "lbl-sm", text: n[2] }));
          svg.appendChild(K.s("text", { x: n[0], y: n[1] + 8, "text-anchor": "middle", class: "lbl mono", text: "val " + (+n[3].toFixed(2)) }));
          if (has) svg.appendChild(K.s("text", { x: n[0], y: n[1] + 21, "text-anchor": "middle", class: "lbl-sm mono f2", text: "grad " + (+g[k].toFixed(2)) }));
        });
        K.clear(c.stage).appendChild(svg);
        var msgs = [
          "Forward pass done: <b>L = " + L.toFixed(2) + "</b>. Press <i>Backward step</i> to compute how L changes when each input is nudged.",
          "Start at the end: " + K.tex("\\frac{\\partial L}{\\partial L} = 1") + ".",
          "L = d·f is a product, so " + K.tex("\\frac{\\partial L}{\\partial d} = f = " + f) + " and " + K.tex("\\frac{\\partial L}{\\partial f} = d = " + d.toFixed(2)) + ".",
          "d = e + c is a sum: a plus node just <b>copies</b> the gradient through: " + K.tex("\\frac{\\partial L}{\\partial e} = \\frac{\\partial L}{\\partial d}\\cdot 1 = " + f) + ".",
          "e = a·b: chain rule multiplies local slope × incoming gradient: " + K.tex("\\frac{\\partial L}{\\partial a} = b \\cdot \\frac{\\partial L}{\\partial e} = " + (b * f).toFixed(2)) + ". That's it — this is <i>all</i> of backprop, repeated for millions of nodes.",
        ];
        c.readout.innerHTML = msgs[stage];
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- normal distribution / normalization
  V.register("normalize", {
    title: "Normalization: rescale numbers to mean 0, spread 1",
    desc: "LayerNorm / RMSNorm do this to every token’s vector so activations don’t blow up or vanish as they pass through dozens of layers.",
    render: function (c, p, K) {
      var mu = 3, sd = 2.5, n = 24, r = K.rng(11), base = [];
      for (var i = 0; i < n; i++) base.push(K.randn(r));
      c.controls.appendChild(K.slider({ label: "raw mean", min: -5, max: 8, step: 0.1, value: mu, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { mu = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "raw spread", min: 0.2, max: 6, step: 0.1, value: sd, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { sd = v; draw(); } }));
      function draw() {
        var raw = base.map(function (z) { return mu + sd * z; });
        var m = raw.reduce(function (a, b) { return a + b; }, 0) / n;
        var v = raw.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / n;
        var ln = raw.map(function (x) { return (x - m) / Math.sqrt(v + 1e-5); });
        var rms = Math.sqrt(raw.reduce(function (a, b) { return a + b * b; }, 0) / n);
        var rn = raw.map(function (x) { return x / rms; });
        var svg = K.s("svg", { viewBox: "0 0 660 230", class: "viz-svg" });
        var rows = [["raw activations", raw, "f1"], ["LayerNorm", ln, "f3"], ["RMSNorm", rn, "f4"]];
        var sx = K.scale(-12, 16, 120, 650);
        rows.forEach(function (row, ri) {
          var y = 40 + ri * 70;
          svg.appendChild(K.s("line", { x1: 120, x2: 650, y1: y, y2: y, class: "grid" }));
          svg.appendChild(K.s("line", { x1: sx(0), x2: sx(0), y1: y - 18, y2: y + 18, class: "axis" }));
          svg.appendChild(K.s("text", { x: 6, y: y + 4, class: "lbl", text: row[0] }));
          row[1].forEach(function (x) { svg.appendChild(K.s("circle", { cx: sx(K.clamp(x, -12, 16)), cy: y, r: 5, class: row[2], opacity: 0.6 })); });
        });
        [-10, -5, 0, 5, 10, 15].forEach(function (t) { svg.appendChild(K.s("text", { x: sx(t), y: 224, "text-anchor": "middle", class: "tick", text: t })); });
        K.clear(c.stage).appendChild(svg);
        c.readout.innerHTML = "mean " + K.tex("\\mu=" + m.toFixed(2)) + ", std " + K.tex("\\sigma=" + Math.sqrt(v).toFixed(2)) + ". &nbsp;LayerNorm: " + K.tex("\\hat x = \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}") +
          " &nbsp;RMSNorm (Llama): " + K.tex("\\hat x = \\frac{x}{\\sqrt{\\tfrac1n\\sum x_i^2}}") + " — cheaper (no mean), works just as well. Both then multiply by a learned scale γ.";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- exponents & logs refresher
  V.register("exp-log", {
    title: "Exponentials and logs are inverses",
    desc: "exp turns sums into products; log turns products into sums. That’s why we add log-probabilities instead of multiplying tiny probabilities (which underflow to 0).",
    render: function (c, p, K) {
      var x = 1;
      c.controls.appendChild(K.slider({ label: "x", min: -3, max: 2.5, step: 0.01, value: x, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { x = v; draw(); } }));
      function draw() {
        var ch = K.chart({ w: 640, h: 300, x: [-3, 12], y: [-3, 12], xLabel: "", yLabel: "" });
        var e = [], l = [], id = [[-3, -3], [12, 12]];
        for (var t = -3; t <= 2.5; t += 0.02) e.push([t, Math.exp(t)]);
        for (var u = 0.05; u <= 12; u += 0.02) l.push([u, Math.log(u)]);
        ch.g.appendChild(K.s("path", { d: K.path(id, ch.sx, ch.sy), class: "ln-thin smuted dash" }));
        ch.g.appendChild(K.s("path", { d: K.path(e, ch.sx, ch.sy), class: "ln s1" }));
        ch.g.appendChild(K.s("path", { d: K.path(l, ch.sx, ch.sy), class: "ln s2" }));
        var ex = Math.exp(x);
        ch.g.appendChild(K.s("circle", { cx: ch.sx(x), cy: ch.sy(ex), r: 6, class: "f1" }));
        ch.g.appendChild(K.s("circle", { cx: ch.sx(ex), cy: ch.sy(x), r: 6, class: "f2" }));
        ch.g.appendChild(K.s("line", { x1: ch.sx(x), y1: ch.sy(ex), x2: ch.sx(ex), y2: ch.sy(x), class: "ln-thin sfg dash", opacity: 0.4 }));
        ch.g.appendChild(K.s("text", { x: ch.sx(2.1), y: ch.sy(10.5), class: "lbl", text: "eˣ" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(10.5), y: ch.sy(2.7), class: "lbl", text: "ln x" }));
        K.clear(c.stage).appendChild(ch.svg);
        c.readout.innerHTML = K.tex("e^{" + x.toFixed(2) + "} = " + ex.toFixed(3)) + " &nbsp;⇔&nbsp; " + K.tex("\\ln(" + ex.toFixed(3) + ") = " + x.toFixed(2)) +
          ". &nbsp;Rules you’ll use: " + K.tex("\\ln(ab)=\\ln a+\\ln b") + ", " + K.tex("e^{a+b}=e^a e^b") + ", " + K.tex("\\ln 1 = 0") + ", " + K.tex("\\ln(p)<0") + " for p<1.";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- percentiles / tail latency
  V.register("percentiles", {
    title: "Averages lie: latency percentiles (p50 / p90 / p99)",
    desc: "Real latencies have a long right tail. p99 = the latency that 99% of requests beat. SLOs are written on percentiles because users remember the slow ones.",
    render: function (c, p, K) {
      var tail = 0.5, med = 400, n = 4000, seed = 5;
      c.controls.appendChild(K.slider({ label: "median (ms)", min: 50, max: 1500, step: 10, value: med, onInput: function (v) { med = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "tail heaviness", min: 0.1, max: 1.2, step: 0.05, value: tail, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { tail = v; draw(); } }));
      c.controls.appendChild(K.button("New sample", function () { seed++; draw(); }));
      function draw() {
        var r = K.rng(seed), xs = [];
        for (var i = 0; i < n; i++) xs.push(med * Math.exp(tail * K.randn(r)));
        xs.sort(function (a, b) { return a - b; });
        var q = function (pp) { return xs[Math.min(n - 1, Math.floor(pp * n))]; };
        var mean = xs.reduce(function (a, b) { return a + b; }, 0) / n;
        var max = q(0.995) * 1.1, bins = 60, hist = new Array(bins).fill(0);
        xs.forEach(function (x) { var b = Math.floor(x / max * bins); if (b < bins) hist[b]++; });
        var hm = Math.max.apply(null, hist);
        var ch = K.chart({ w: 640, h: 280, x: [0, max], y: [0, hm * 1.15], xLabel: "latency (ms)", yLabel: "# requests", yFmt: function () { return ""; } });
        hist.forEach(function (hc, i) {
          ch.g.appendChild(K.s("rect", { x: ch.sx(i * max / bins), y: ch.sy(hc), width: Math.max(1, ch.sx(max / bins) - ch.sx(0) - 1), height: ch.sy(0) - ch.sy(hc), class: "f1", opacity: 0.55 }));
        });
        [["p50", q(0.5), "s3"], ["mean", mean, "sfg"], ["p90", q(0.9), "s2"], ["p99", q(0.99), "s5"]].forEach(function (m, i) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(m[1]), x2: ch.sx(m[1]), y1: ch.sy(0), y2: ch.sy(hm * 1.1), class: "ln " + m[2] + (m[0] === "mean" ? " dash" : "") }));
          ch.g.appendChild(K.s("text", { x: ch.sx(m[1]) + 4, y: ch.sy(hm * (1.08 - i * 0.09)), class: "lbl-sm", text: m[0] + " " + Math.round(m[1]) }));
        });
        K.clear(c.stage).appendChild(ch.svg);
        c.readout.innerHTML = "p50 <b>" + Math.round(q(0.5)) + " ms</b> · mean <b>" + Math.round(mean) + "</b> · p90 <b>" + Math.round(q(0.9)) + "</b> · p99 <b>" + Math.round(q(0.99)) + " ms</b> (" + (q(0.99) / q(0.5)).toFixed(1) +
          "× the median). If a page makes 20 calls, the chance at least one hits your p99 is " + K.tex("1-0.99^{20}\\approx 18\\%") + ".";
      }
      draw();
    },
  });
})();
