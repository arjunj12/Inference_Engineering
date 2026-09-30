/* _template.js — copy this file to create a new module.
 * 1) Save as content/modules/<your-id>.js  (id below must match the filename)
 * 2) Add "<your-id>" to a phase's `modules` array in content/course.js
 *
 * Markdown lives inside MD(function(){/* ... *\/}) — write backticks, $math$, \LaTeX freely.
 * Extras inside markdown:
 *   ::viz <name> {"param": 1}     interactive chart on its own line (see assets/js/viz/*.js)
 *   > [!TIP] / [!MATH] / [!WARNING] / [!INTUITION] / [!REAL] / [!PREREQ] / [!CHECK] / [!BUILD]
 *   - [ ] task                    persistent checkbox
 *   <details><summary>…</summary> … </details>   collapsible aside (leave blank lines around markdown inside)
 *   \$                            literal dollar sign
 */
Course.module({
  id: "_template",
  title: "Module title",
  short: "Short sidebar title",           // optional
  tagline: "One sentence: what you'll be able to do after this.",
  hours: 12,                               // total estimate (optional; else summed from lessons)
  level: "core",                           // "core" | "optional"
  capstone: false,
  runsOn: ["mac", "colab"],                // mac | colab | cloud | multi-gpu | browser | any
  tags: ["tag1", "tag2"],

  // TOP-DOWN: show the finished result first.
  goal: MD(function () {/*
What you'll have running at the end, with a concrete output sample.
  */}),
  demo: { viz: "softmax-temp", params: {} }, // optional interactive demo shown inside the goal box (skipped if goal already has ::viz of the same name)

  why: MD(function () {/*
Where this shows up in real systems / jobs.
  */}),

  prereqs: [
    // math: true -> also listed in the Math track
    { title: "Refresher title", skipIf: "you already know X", math: true, md: MD(function () {/*
Short, friendly refresher.
    */}) },
  ],

  lessons: [
    {
      id: "big-picture",                    // stable id (used for progress + deep links)
      title: "Lesson title",
      kind: "concept",                      // demo | concept | math | build | lab | read | deep
      minutes: 60,
      runsOn: ["mac"],
      optional: false,
      md: MD(function () {/*
## Heading
Text, $x^2$, code, and ::viz lines.
      */}),
      resources: [{ title: "Name", url: "https://…", type: "video", note: "why" }],
    },
  ],

  challenge: {
    title: "Challenge title",
    md: MD(function () {/* What to build, constraints, hints. */}),
    checklist: ["Acceptance criterion 1", "Acceptance criterion 2"],
    stretch: "Optional harder variant.",
  },

  connects: MD(function () {/* How this links to the next modules and to real-world engineering. */}),
  interview: ["Question 1?", "Question 2?"],
  resources: [{ title: "Name", url: "https://…", type: "repo", note: "" }],
  // types: video | paper | repo | article | book | course | docs | tool | practice
});
