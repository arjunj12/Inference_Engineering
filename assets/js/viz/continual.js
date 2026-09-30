/* continual.js — interactive explainers for continual learning / catastrophic forgetting. */
(function () {
  "use strict";
  var V = window.Viz;
  function num(v, d) { return typeof v === "number" && isFinite(v) ? v : d; }
  function stats(items) {
    return '<div class="stat-grid">' + items.map(function (it) {
      return '<div class="stat"><small>' + it[0] + "</small><b" + (it[2] ? ' style="color:' + it[2] + '"' : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  var STRAT_LABEL = { naive: "naive fine-tuning", replay: "replay buffer", ewc: "EWC", adapter: "adapter per task", joint: "joint retrain (upper bound)" };

  // ---------------------------------------------------------------- live catastrophic forgetting demo
  V.register("forgetting", {
    title: "Catastrophic forgetting, live: train on task A, then task B",
    desc: "A real 2→16→1 neural net trains in your browser. Task A (circles, top) and task B (squares, bottom) are 2-class problems. Train A, then train B: with naive fine-tuning the net overwrites what it knew and task-A accuracy collapses. Try replay (mix in a few stored A examples) or EWC (penalise moving weights that mattered for A).",
    render: function (c, p, K) {
      var HID = 16, NP = 4 * HID + 1, LR = 0.05, MOM = 0.9, STEPS_A = 150, STEPS_B = 150;
      var strategy = ["naive", "replay", "ewc"].indexOf(p.strategy) >= 0 ? p.strategy : "naive";
      var bufSize = num(p.buffer, 4), logLam = Math.log10(num(p.lambda, 100));
      // ----- data
      var dr = K.rng(6);
      function blobs(centers, n) {
        var d = [];
        centers.forEach(function (cc) { for (var i = 0; i < n; i++) d.push([cc[0] + 0.45 * K.randn(dr), cc[1] + 0.45 * K.randn(dr), cc[2]]); });
        return d;
      }
      var CA = [[-1.3, 1.1, 0], [1.3, 1.1, 1]], CB = [[0, -0.3, 1], [0, -2.2, 0]];
      var A = blobs(CA, 40), B = blobs(CB, 40), At = blobs(CA, 60), Bt = blobs(CB, 60);
      // ----- model
      var P, Vel, act = new Float64Array(HID), G = new Float64Array(NP);
      var phase, stepN, stepsInA, stepsInB, hist, switchAt, F, star, replay, snap, results = {};
      function initParams() {
        var r = K.rng(3); P = new Float64Array(NP); Vel = new Float64Array(NP);
        for (var h = 0; h < HID; h++) { P[2 * h] = K.randn(r) * 0.9; P[2 * h + 1] = K.randn(r) * 0.9; P[2 * HID + h] = K.randn(r) * 0.3; P[3 * HID + h] = K.randn(r) * 0.3; }
      }
      function fwd(x, y, a) {
        var o = P[4 * HID];
        for (var h = 0; h < HID; h++) {
          var t = Math.tanh(P[2 * h] * x + P[2 * h + 1] * y + P[2 * HID + h]);
          if (a) a[h] = t; o += P[3 * HID + h] * t;
        }
        return 1 / (1 + Math.exp(-o));
      }
      function gradPt(pt, g, w) {
        var pr = fwd(pt[0], pt[1], act), d = (pr - pt[2]) * w;
        g[4 * HID] += d;
        for (var h = 0; h < HID; h++) {
          g[3 * HID + h] += d * act[h];
          var dz = d * P[3 * HID + h] * (1 - act[h] * act[h]);
          g[2 * h] += dz * pt[0]; g[2 * h + 1] += dz * pt[1]; g[2 * HID + h] += dz;
        }
      }
      function acc(D) { var k = 0; D.forEach(function (pt) { if ((fwd(pt[0], pt[1]) > 0.5 ? 1 : 0) === pt[2]) k++; }); return k / D.length; }
      function lam() { return Math.pow(10, logLam); }
      function sgd(data) {
        G.fill(0);
        data.forEach(function (pt) { gradPt(pt, G, 1 / data.length); });
        var useEwc = phase === "B" && strategy === "ewc" && F;
        for (var i = 0; i < NP; i++) {
          Vel[i] = MOM * Vel[i] - LR * G[i]; P[i] += Vel[i];
          // EWC penalty (λ/2)·F·(θ−θ*)² applied as a proximal step (unconditionally stable)
          if (useEwc) { var k = LR * lam() * F[i]; P[i] = (P[i] + k * star[i]) / (1 + k); }
        }
      }
      function record() { hist.push([stepN, acc(At), acc(Bt)]); }
      function reset() {
        initParams(); phase = "A"; stepN = 0; stepsInA = 0; stepsInB = 0; hist = []; switchAt = null; F = null; star = null; replay = []; snap = null;
        record();
      }
      function setupB() {
        // Fisher diagonal on task A (empirical, per-sample squared gradients), normalised to max 1.
        F = new Float64Array(NP); var g = new Float64Array(NP);
        A.forEach(function (pt) { g.fill(0); gradPt(pt, g, 1); for (var i = 0; i < NP; i++) F[i] += g[i] * g[i] / A.length; });
        var mx = 0; for (var i = 0; i < NP; i++) mx = Math.max(mx, F[i]);
        for (var j = 0; j < NP; j++) F[j] /= mx || 1;
        star = Float64Array.from(P);
        pickReplay();
        Vel = new Float64Array(NP);
      }
      function pickReplay() {
        var r = K.rng(9), idx = A.map(function (_, i) { return i; });
        for (var i = idx.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        replay = idx.slice(0, bufSize).map(function (k) { return A[k]; });
      }
      function train(task, n) {
        for (var s = 0; s < n; s++) {
          if (task === "A") {
            if (phase !== "A") return;
            sgd(A); stepsInA++;
          } else {
            if (phase === "A") {
              phase = "B"; switchAt = stepN;
              snap = { P: Float64Array.from(P), stepN: stepN, hist: hist.slice(), stepsInA: stepsInA };
              setupB();
            }
            sgd(strategy === "replay" ? B.concat(replay) : B); stepsInB++;
          }
          stepN++; record();
        }
        if (phase === "B" && stepsInB >= STEPS_B) {
          var last = hist[hist.length - 1];
          results[strategy + (strategy === "replay" ? " (" + bufSize + ")" : strategy === "ewc" ? " (λ=" + K.fmtShort(lam()) + ")" : "")] = [last[1], last[2]];
        }
      }
      function restartB() {
        if (!snap) return;
        P = Float64Array.from(snap.P); stepN = snap.stepN; hist = snap.hist.slice(); stepsInA = snap.stepsInA; stepsInB = 0;
        phase = "B"; setupB();
      }

      var timer = null;
      var runBtn = K.button("▶ Run A → B", function () { if (timer) stopRun(); else startRun(); }, "primary");
      function startRun() {
        if (phase === "B" && stepsInB >= STEPS_B) restartB();
        timer = setInterval(function () {
          if (!c.root.isConnected) { stopRun(); return; }
          if (phase === "A" && stepsInA < STEPS_A) train("A", 3);
          else if (stepsInB < STEPS_B) train("B", 3);
          else { stopRun(); }
          draw();
        }, 40);
        runBtn.textContent = "⏸ Pause";
      }
      function stopRun() { clearInterval(timer); timer = null; runBtn.textContent = "▶ Run A → B"; }
      c.controls.appendChild(runBtn);
      var btnA = K.button("Train A +25", function () { train("A", 25); draw(); });
      c.controls.appendChild(btnA);
      c.controls.appendChild(K.button("Train B +25", function () { train("B", 25); draw(); }));
      c.controls.appendChild(K.button("↺ Redo B from A-model", function () { stopRun(); restartB(); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { stopRun(); reset(); draw(); }));
      c.controls.appendChild(K.select({ label: "strategy", value: strategy, options: [{ value: "naive", label: "naive fine-tune" }, { value: "replay", label: "replay buffer" }, { value: "ewc", label: "EWC" }], onChange: function (v) { strategy = v; if (phase === "B") { stopRun(); restartB(); } draw(); } }));
      c.controls.appendChild(K.slider({ label: "replay buffer (A samples)", min: 1, max: 40, value: bufSize, onInput: function (v) { bufSize = v; if (phase === "B") { pickReplay(); } draw(); } }));
      c.controls.appendChild(K.slider({ label: "EWC λ", min: 0, max: 3.5, step: 0.1, value: logLam, fmt: function (v) { return K.fmtShort(Math.pow(10, v)); }, onInput: function (v) { logLam = v; draw(); } }));

      // ----- drawing
      var GX = [-3, 3], GY = [-3.3, 2.7], NC = 30;
      function draw() {
        var S = 330, pad = 6;
        var sx = K.scale(GX[0], GX[1], pad, S - pad), sy = K.scale(GY[0], GY[1], S - pad, pad);
        var svg = K.s("svg", { viewBox: "0 0 " + S + " " + S, class: "viz-svg", style: { maxWidth: "400px", margin: "0 auto" } });
        var cw = (S - 2 * pad) / NC;
        for (var i = 0; i < NC; i++) for (var j = 0; j < NC; j++) {
          var x = GX[0] + (i + 0.5) * (GX[1] - GX[0]) / NC, y = GY[1] - (j + 0.5) * (GY[1] - GY[0]) / NC;
          var pr = fwd(x, y);
          svg.appendChild(K.s("rect", { x: pad + i * cw, y: pad + j * cw, width: cw + 0.3, height: cw + 0.3, class: pr > 0.5 ? "f1" : "f2", opacity: (0.06 + Math.abs(pr - 0.5) * 0.55).toFixed(3) }));
        }
        svg.appendChild(K.s("line", { x1: pad, x2: S - pad, y1: sy(0.4), y2: sy(0.4), class: "ln-thin smuted dash" }));
        svg.appendChild(K.s("text", { x: pad + 4, y: sy(0.4) - 5, class: "lbl-sm", text: "A ↑" }));
        svg.appendChild(K.s("text", { x: pad + 4, y: sy(0.4) + 13, class: "lbl-sm", text: "B ↓" }));
        var inBuf = {};
        if (phase === "B" && strategy === "replay") replay.forEach(function (pt) { inBuf[A.indexOf(pt)] = 1; });
        A.forEach(function (pt, k) {
          svg.appendChild(K.s("circle", { cx: sx(pt[0]), cy: sy(pt[1]), r: 4, class: (pt[2] ? "f1" : "f2") + " sfg", "stroke-width": 0.8 }));
          if (inBuf[k]) svg.appendChild(K.s("circle", { cx: sx(pt[0]), cy: sy(pt[1]), r: 7.5, class: "ln-thin s4", "stroke-width": 2 }));
        });
        B.forEach(function (pt) {
          svg.appendChild(K.s("rect", { x: sx(pt[0]) - 3.5, y: sy(pt[1]) - 3.5, width: 7, height: 7, class: (pt[2] ? "f1" : "f2") + " sfg", "stroke-width": 0.8 }));
        });
        var ch = K.chart({ w: 420, h: S, x: [0, Math.max(STEPS_A + STEPS_B, stepN)], y: [0, 1], xLabel: "training step", yLabel: "test accuracy", yFmt: function (v) { return Math.round(v * 100) + "%"; }, pad: { l: 50, r: 14, t: 26 } });
        ch.g.appendChild(K.s("line", { x1: ch.sx(0), x2: ch.sx(Math.max(STEPS_A + STEPS_B, stepN)), y1: ch.sy(0.5), y2: ch.sy(0.5), class: "ln-thin smuted dash" }));
        ch.g.appendChild(K.s("text", { x: ch.sx(0) + 4, y: ch.sy(0.5) - 4, class: "lbl-sm", text: "chance" }));
        if (switchAt !== null) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(switchAt), x2: ch.sx(switchAt), y1: ch.sy(0), y2: ch.sy(1), class: "ln-thin sfg dash" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(switchAt) + 4, y: ch.sy(0.08), class: "lbl-sm", text: "switch to task B" }));
        }
        var pa = hist.map(function (h) { return [h[0], h[1]]; }), pb = hist.map(function (h) { return [h[0], h[2]]; });
        if (hist.length > 1) {
          ch.g.appendChild(K.s("path", { d: K.path(pa, ch.sx, ch.sy), class: "ln s4" }));
          ch.g.appendChild(K.s("path", { d: K.path(pb, ch.sx, ch.sy), class: "ln s6" }));
        }
        ch.svg.appendChild(K.s("circle", { cx: 60, cy: 12, r: 5, class: "f4" }));
        ch.svg.appendChild(K.s("text", { x: 69, y: 16, class: "lbl-sm", text: "accuracy on task A" }));
        ch.svg.appendChild(K.s("rect", { x: 190, y: 7, width: 10, height: 10, class: "f6" }));
        ch.svg.appendChild(K.s("text", { x: 205, y: 16, class: "lbl-sm", text: "accuracy on task B" }));
        K.clear(c.stage).appendChild(K.h("div", { class: "viz-row" }, K.h("div", {}, svg), K.h("div", {}, ch.svg)));
        btnA.disabled = phase !== "A";

        var last = hist[hist.length - 1];
        var peakA = snap ? snap.hist[snap.hist.length - 1][1] : last[1];
        var how = strategy === "naive" ? "Loss = " + K.tex("L_B(\\theta)") + " only. Nothing protects the weights task A relied on." :
          strategy === "replay" ? "Loss = " + K.tex("L_B(\\theta) + L_{\\text{buffer}}(\\theta)") + " — every B step also rehearses <b>" + bufSize + "</b> stored A examples (purple rings)." :
            "Loss = " + K.tex("L_B(\\theta) + \\tfrac{\\lambda}{2}\\sum_i F_i(\\theta_i-\\theta^*_{A,i})^2") + ", λ = <b>" + K.fmtShort(lam()) + "</b>. " + K.tex("F_i") + " (Fisher diagonal, computed on task A) = how much A's loss cares about weight i. Too big a λ and B can't be learned (stability vs plasticity).";
        var res = Object.keys(results);
        c.readout.innerHTML = stats([
          ["step / phase", stepN + " · training " + phase], ["task A accuracy", Math.round(last[1] * 100) + "%", "var(--c4)"], ["task B accuracy", Math.round(last[2] * 100) + "%", "var(--c6)"],
          ["forgetting on A", snap ? Math.round((peakA - last[1]) * 100) + " pts" : "—"],
        ]) + "<div style='margin-top:6px'>" + how + "</div>" +
          (res.length ? "<div style='margin-top:4px'>Finished runs (A / B): " + res.map(function (k) { return "<b>" + k + "</b> " + Math.round(results[k][0] * 100) + "% / " + Math.round(results[k][1] * 100) + "%"; }).join(" · ") + "</div>" : "") +
          "<div class='muted' style='margin-top:4px'>Background colour = the net's prediction (blue = class 1, orange = class 0). B's rule (“upper = 1”) says nothing about task A's area, so gradient descent happily repaints it. Fine-tuning an LLM on a narrow dataset does the same thing to skills it isn't rehearsing. Use “Redo B from A-model” to compare strategies from the same starting point.</div>";
      }
      reset(); draw();
    },
  });

  // ---------------------------------------------------------------- accuracy matrix (stylised)
  V.register("accuracy-matrix", {
    title: "The continual-learning accuracy matrix: R[i][j] after training task i, tested on task j",
    desc: "Rows = “after finishing task i”, columns = “evaluated on task j”. The diagonal is how well each task was learned; below it is how well old tasks are remembered. Numbers come from a stylised model (illustrative, not measured) that mimics typical published behaviour. Hover a cell.",
    render: function (c, p, K) {
      var T = K.clamp(num(p.tasks, 5), 3, 6), strategy = STRAT_LABEL[p.strategy] ? p.strategy : "naive", dis = 0.6, buf = 10, seed = 4, hover = null;
      var CH = 0.1; // chance for 10-way tasks
      c.controls.appendChild(K.select({ label: "strategy", value: strategy, options: Object.keys(STRAT_LABEL).map(function (k) { return { value: k, label: STRAT_LABEL[k] }; }), onChange: function (v) { strategy = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "tasks T", min: 3, max: 6, value: T, onInput: function (v) { T = v; hover = null; draw(); } }));
      c.controls.appendChild(K.slider({ label: "task dissimilarity", min: 0, max: 1, step: 0.05, value: dis, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { dis = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "replay buffer", min: 0, max: 50, value: buf, fmt: function (v) { return v + "%"; }, onInput: function (v) { buf = v; draw(); } }));
      c.controls.appendChild(K.button("🎲 New noise", function () { seed++; draw(); }));

      function matrix(st) {
        var r = K.rng(seed * 31 + T), base = [];
        for (var t = 0; t < T; t++) base.push(0.9 + (r() - 0.5) * 0.08);
        var noise = []; for (var q = 0; q < T * T; q++) noise.push((r() - 0.5) * 0.03);
        var cov = 1 - Math.exp(-buf / 6);
        var rho = { naive: 1 - 0.8 * dis, replay: 1 - 0.8 * dis * (1 - cov), ewc: 1 - 0.8 * dis * 0.4, adapter: 1, joint: 1 }[st];
        var R = [];
        for (var i = 0; i < T; i++) {
          R.push([]);
          for (var j = 0; j < T; j++) {
            var diag = base[j] - (st === "ewc" ? 0.03 * j * (0.5 + dis) : st === "adapter" ? 0.02 : st === "replay" ? 0.005 * j : 0);
            var v;
            if (j > i) v = st === "adapter" ? CH : CH + 0.12 * (1 - dis) * (1 - 0.3 * (j - i - 1));
            else if (j === i) v = diag;
            else v = st === "joint" ? diag + 0.01 * (i - j) : CH + (diag - CH) * Math.pow(rho, i - j);
            R[i].push(K.clamp(v + (j === i && st === "adapter" ? 0 : noise[i * T + j]), 0, 1));
          }
        }
        return R;
      }
      function metrics(R) {
        var accv = 0, bwt = 0;
        for (var j = 0; j < T; j++) accv += R[T - 1][j] / T;
        for (var k = 0; k < T - 1; k++) bwt += (R[T - 1][k] - R[k][k]) / (T - 1);
        return { acc: accv, bwt: bwt };
      }
      function draw() {
        var R = matrix(strategy), m = metrics(R);
        var cs = Math.min(62, 330 / T), x0 = 120, y0 = 50, W = x0 + T * cs + 170, H = y0 + T * cs + 36;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg", style: { maxWidth: Math.round(W * 1.15) + "px", margin: "0 auto" } });
        svg.appendChild(K.s("text", { x: x0 + T * cs / 2, y: 16, "text-anchor": "middle", class: "lbl", text: "evaluated on task j →" }));
        svg.appendChild(K.s("text", { x: 8, y: y0 - 10, class: "lbl-sm", text: "after training ↓" }));
        for (var j = 0; j < T; j++) svg.appendChild(K.s("text", { x: x0 + j * cs + cs / 2, y: y0 - 8, "text-anchor": "middle", class: "lbl-sm", text: "task " + (j + 1) }));
        for (var i = 0; i < T; i++) {
          svg.appendChild(K.s("text", { x: x0 - 8, y: y0 + i * cs + cs / 2 + 4, "text-anchor": "end", class: "lbl-sm", text: "task " + (i + 1) }));
          for (var j2 = 0; j2 < T; j2++) {
            (function (i, j) {
              var v = R[i][j], op = K.clamp((v - CH) / (1 - CH), 0, 1), future = j > i;
              var g = K.s("g", { style: { cursor: "pointer" } });
              g.appendChild(K.s("rect", { x: x0 + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, rx: 4, class: "box" }));
              g.appendChild(K.s("rect", { x: x0 + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, rx: 4, class: future ? "fmuted" : "f3", opacity: (future ? 0.15 : 0.1 + 0.85 * op).toFixed(3) }));
              if (i === j) g.appendChild(K.s("rect", { x: x0 + j * cs + 1, y: y0 + i * cs + 1, width: cs - 4, height: cs - 4, rx: 4, class: "ln-thin sfg", "stroke-width": 1.5 }));
              if (hover && hover[0] === i && hover[1] === j) g.appendChild(K.s("rect", { x: x0 + j * cs - 1, y: y0 + i * cs - 1, width: cs, height: cs, rx: 5, class: "ln-thin s2", "stroke-width": 2.5 }));
              g.appendChild(K.s("text", { x: x0 + j * cs + cs / 2 - 1, y: y0 + i * cs + cs / 2 + 4, "text-anchor": "middle", class: "mono " + (!future && op > 0.55 ? "t-white" : ""), style: { fontSize: cs < 50 ? "10.5px" : "12px" }, text: Math.round(v * 100) }));
              g.addEventListener("mouseenter", function () { if (!hover || hover[0] !== i || hover[1] !== j) { hover = [i, j]; draw(); } });
              g.addEventListener("click", function () { hover = [i, j]; draw(); });
              svg.appendChild(g);
            })(i, j2);
          }
        }
        var lx = x0 + T * cs + 16;
        [["diagonal = just learned", "sfg"], ["below = remembered", "f3"], ["above = not trained yet", "fmuted"]].forEach(function (lg, k) {
          svg.appendChild(K.s("rect", { x: lx, y: y0 + k * 20, width: 12, height: 12, rx: 2, class: lg[1] === "sfg" ? "ln-thin sfg" : lg[1], opacity: lg[1] === "fmuted" ? 0.3 : 0.8, fill: lg[1] === "sfg" ? "none" : null }));
          svg.appendChild(K.s("text", { x: lx + 17, y: y0 + k * 20 + 10, class: "lbl-sm", text: lg[0] }));
        });
        svg.appendChild(K.s("text", { x: lx, y: y0 + 80, class: "lbl-sm", text: "values = accuracy %" }));
        svg.appendChild(K.s("text", { x: lx, y: y0 + 96, class: "lbl-sm", text: "chance = 10%" }));
        svg.appendChild(K.s("text", { x: 8, y: H - 8, class: "lbl-sm", style: { fontStyle: "italic" }, text: "Illustrative: stylised generative model, not real measurements." }));
        K.clear(c.stage).appendChild(svg);

        var cmp = Object.keys(STRAT_LABEL).map(function (k) { var mm = metrics(matrix(k)); return [STRAT_LABEL[k] + ": ACC · BWT", Math.round(mm.acc * 100) + "% · " + (mm.bwt >= 0 ? "+" : "−") + Math.abs(Math.round(mm.bwt * 100)), k === strategy ? "var(--accent)" : null]; });
        var hv = hover ? "Cell R[" + (hover[0] + 1) + "][" + (hover[1] + 1) + "] = <b>" + Math.round(R[hover[0]][hover[1]] * 100) + "%</b>: after training through task " + (hover[0] + 1) + ", accuracy on task " + (hover[1] + 1) +
          (hover[1] > hover[0] ? " (not trained yet — anything above chance is forward transfer)." : hover[1] === hover[0] ? " (just learned it: plasticity)." : " (learned " + (hover[0] - hover[1]) + " task(s) ago: retention).") : "Hover a cell to read it.";
        var notes = {
          naive: "Each new task overwrites the last: rows fade to chance below the diagonal.",
          replay: "Rehearsing a small buffer of old examples keeps most of the old accuracy; bigger buffer → less forgetting (more memory/compute, and data you must be allowed to keep).",
          ewc: "Old tasks decay slowly, but the diagonal drifts down: the growing penalty makes new tasks harder to learn (intransigence).",
          adapter: "Frozen base + one small adapter (e.g. LoRA) per task: zero forgetting by construction, but you need to know which adapter to use at inference, and no forward transfer.",
          joint: "Retraining on all data so far: the upper bound everyone compares against — expensive, and needs all old data.",
        };
        c.readout.innerHTML = stats([["average accuracy ACC", Math.round(m.acc * 100) + "%"], ["backward transfer BWT", (m.bwt >= 0 ? "+" : "") + (m.bwt * 100).toFixed(1) + " pts", m.bwt < -0.05 ? "var(--red)" : "var(--c3)"]]) +
          "<div style='margin-top:6px'>" + hv + "</div>" +
          "<div style='margin-top:4px'>" + K.tex("\\text{ACC} = \\tfrac{1}{T}\\sum_{j} R_{T,j}") + " (last row), " + K.tex("\\text{BWT} = \\tfrac{1}{T-1}\\sum_{j<T} (R_{T,j} - R_{j,j})") + " — negative BWT = forgetting. " + notes[strategy] + "</div>" +
          "<div style='margin-top:6px'>" + stats(cmp) + "</div>";
      }
      draw();
    },
  });
})();
