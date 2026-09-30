Course.module({
  id: "m24-continual-learning",
  title: "Continual learning: models that keep learning",
  short: "Continual learning",
  tagline: "Watch a network forget everything it knew, measure it with an accuracy matrix, then fix it with replay, EWC and adapters — on MNIST and on a real LLM.",
  hours: 16,
  level: "core",
  runsOn: ["mac", "colab"],
  tags: ["continual-learning", "forgetting", "ewc", "replay", "lora", "model-merging", "multi-lora"],

  goal: MD(function () {/*
You train one small network on five tasks **in sequence** (digits 0/1, then 2/3, … then 8/9) and print its **accuracy matrix** — row *i* = "after training task *i*", column *j* = "accuracy on task *j*". First with naive fine-tuning, then with the fixes:

```text
naive/class   after task 4: 0.0%  0.0%  0.0%  0.0%  97.9%     ACC=19.6%   BWT=-99.3%
replay/class  after task 4: 94.8% 83.3% 86.8% 88.8% 96.3%     ACC=90.0%   BWT=-10.8%
ewc/task      after task 4: 98.9% 99.2% 99.2% 98.8% 98.9%     ACC=99.0%   BWT=-0.7%
```

The first row is **catastrophic forgetting**: 97.9% on the newest task, *zero* on everything else. Then you repeat the experiment on a real LLM — LoRA-fine-tune **Qwen2.5-0.5B** on SQL with `mlx-lm` on your Mac and measure how much general knowledge (ARC-Easy) it loses, with and without replay. You finish with a design note on how to **continuously update a production model safely** — which, it turns out, is mostly an inference-engineering problem.

(Your exact numbers will differ with seed and hardware; the *shape* of the matrix won't.)
  */}),
  demo: { viz: "forgetting", params: {} },

  why: MD(function () {/*
Every deployed model is frozen at a **knowledge cutoff**, while the world, your product and your users keep changing. Labs refresh models with **continued pretraining**; product teams ship **fine-tunes and LoRA adapters** weekly; RL post-training is literally a model learning from its own experience. Each of these can silently break something the model used to do well — and the people who own the serving stack own the blast radius: **adapter hot-swapping, weight updates in a live engine, versioning, rollback and eval gates**. Continual learning is where training research meets inference operations.
  */}),

  prereqs: [
    {
      title: "A PyTorch training loop (forward → loss → backward → step)",
      skipIf: "you finished M01 and M05",
      md: MD(function () {/*
Everything here reuses the loop from M01:

```python
logits = model(x)                       # forward
loss = F.cross_entropy(logits, y)       # how wrong are we?
opt.zero_grad(); loss.backward()        # gradients of loss w.r.t. every weight
opt.step()                              # nudge weights downhill
```

Continual learning changes **what data** flows into that loop and **what extra terms** get added to `loss`. The loop itself never changes.
      */}),
    },
    {
      title: "LoRA in one paragraph",
      skipIf: "you did the LoRA lab in M05",
      md: MD(function () {/*
LoRA freezes a weight matrix $W$ (shape $d \times k$) and learns a low-rank update $\Delta W = BA$ with $B$ of shape $d \times r$ and $A$ of shape $r \times k$, $r \ll d$ (e.g. $r = 8$). The layer computes $Wx + BAx$. You train ~0.1–1% of the parameters, and the adapter is a small file (MBs) you can **swap** at serving time. That swap-ability is exactly why LoRA is the most practical continual-learning tool for LLMs today.

::viz lora
      */}),
    },
  ],

  lessons: [
    {
      id: "see-forgetting",
      title: "See it: a network forgets task A the moment it learns task B",
      kind: "demo",
      minutes: 45,
      runsOn: ["mac", "browser"],
      md: MD(function () {/*
## Step 1 — watch it in the browser

Train on task A, then switch to task B, and watch the task-A accuracy line. Then flip the strategy to **replay** and **EWC** and run it again.

::viz forgetting

## Step 2 — reproduce it on real data (2 minutes on your Mac)

**Split-MNIST** is the "hello world" of continual learning: MNIST's 10 digits chopped into 5 tasks of 2 digits each. We train one 2-layer MLP on the tasks **in order**, never revisiting old data — just like a model that is fine-tuned on this month's data and then next month's.

```python
# forget.py — the smallest possible catastrophic-forgetting demo
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

dev = "mps" if torch.backends.mps.is_available() else "cpu"
def load(train):
    ds = datasets.MNIST("data", train=train, download=True)
    return ds.data.float().div(255).view(-1, 784).to(dev), ds.targets.to(dev)
Xtr, Ytr = load(True); Xte, Yte = load(False)
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
def subset(X, Y, t):
    m = (Y == TASKS[t][0]) | (Y == TASKS[t][1]); return X[m], Y[m]

model = nn.Sequential(nn.Linear(784, 400), nn.ReLU(), nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 10)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(5):
    x, y = subset(Xtr, Ytr, t)
    for epoch in range(2):
        perm = torch.randperm(len(x), device=dev)
        for s in range(0, len(x), 128):
            b = perm[s:s + 128]
            loss = F.cross_entropy(model(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        accs = [(model(subset(Xte, Yte, j)[0]).argmax(1) == subset(Xte, Yte, j)[1]).float().mean().item() for j in range(5)]
    print(f"after task {t} {TASKS[t]}: " + "  ".join(f"{a:6.1%}" for a in accs))
```

```text
after task 0 (0, 1): 100.0%    0.0%    0.0%    0.0%    0.0%
after task 1 (2, 3):   0.0%   97.9%    0.0%    0.0%    0.0%
after task 2 (4, 5):   0.0%    0.0%   99.7%    0.0%    0.0%
after task 3 (6, 7):   0.0%    0.0%    0.0%   99.6%    0.0%
after task 4 (8, 9):   0.0%    0.0%    0.0%    0.0%   98.4%
```

> [!INTUITION] Why *zero*, not "a bit worse"?
> During task B the loss only ever says "output 2 or 3". The cheapest way for gradient descent to satisfy that is to **push the logits of 0 and 1 down** for every input — including images of 0s and 1s. Nothing in the loss protects old behaviour, so it's overwritten. The weights that encoded "this is a 0" are still *mostly* there (you'll see that task-incremental evaluation recovers a lot) — it's the **output layer's bias toward the newest classes** that makes the collapse total.

That's the entire problem of this module: **gradient descent is greedy about the current data.** Every fix you'll learn either (1) shows the model old data again (**replay**), (2) penalizes moving important weights (**regularization / EWC**), or (3) gives each task its own parameters (**isolation / adapters**).

- [ ] Played with the `forgetting` viz: naive vs replay vs EWC
- [ ] Ran `forget.py` and got a (near-)diagonal accuracy table
- [ ] Wrote one sentence explaining why accuracy on old tasks drops to ~0 rather than ~50%
      */}),
    },
    {
      id: "why-continual",
      title: "Why continual learning matters (and what it is not)",
      kind: "concept",
      minutes: 45,
      md: MD(function () {/*
## Three reasons you'll run into this at work

1. **Knowledge cutoff.** A model trained on data up to date *D* knows nothing after *D*. Retraining from scratch costs millions of GPU-hours; labs instead do **continued pretraining** on fresh data and must not lose what the model already knew. Same for a company that wants its 8B model to learn its internal codebase every month.
2. **Personalization & specialization.** Per-customer or per-domain fine-tunes (legal, medical, SQL, a customer's tone of voice). Often served as **one base model + thousands of LoRA adapters** — each adapter is a tiny "continual learning" step that must not break the base behaviour users rely on.
3. **Agents that learn from experience.** RL post-training (M22–M23) *is* learning from experience: the model acts, gets rewarded, updates. The long-term dream is agents that keep improving from deployment feedback. Every update risks regressions elsewhere — "fixed the coding benchmark, broke the chit-chat".

## What continual learning is *not*

Several techniques make a model "know new things" **without changing weights**. Be precise about the difference — interviewers ask.

| Approach | Where new knowledge lives | Changes weights? | Persists across sessions? | Typical use |
|---|---|---|---|---|
| **In-context learning** | the prompt | no | no | few-shot examples, instructions |
| **RAG** (retrieval) | an external index, pasted into the prompt | no | yes (the index) | fresh facts, citations, private docs |
| **Agent memory** | notes/summaries/skills the agent writes and re-reads | no | yes | long-running assistants, "reflection" |
| **Fine-tuning / LoRA** | the weights (or an adapter) | yes | yes | new skills, formats, styles, domains |
| **Continual learning** | the weights, **over a sequence** of updates, with old skills preserved | yes | yes | model refresh, lifelong agents |

> [!TIP] Rule of thumb used in practice
> **Facts** that change → RAG. **Behaviour/skills/format** → fine-tune (LoRA). **Many small updates over time** → you now have a continual-learning problem: measure forgetting, keep replay data, version everything.

> [!REAL] The inference engineer's angle
> Every one of these lands on the serving stack: RAG adds prefill tokens (TTFT, prefix caching — M08); agent memory adds long contexts (KV cache — M08, M17); LoRA adds adapter management (multi-LoRA serving — last lesson); weight updates need **hot reload, cache invalidation, canaries and rollback** (M18, M23).

## A short history (so the papers make sense)

- **1989–1999:** "catastrophic interference" noticed in early neural nets (McCloskey & Cohen; French).
- **2017:** **EWC** (Kirkpatrick et al., DeepMind) — regularize important weights. **GEM** (Lopez-Paz & Ranzato) — defines the accuracy-matrix metrics (ACC, BWT, FWT) we use. **Deep Generative Replay** (Shin et al.).
- **2019:** van de Ven & Tolias split the field into **task / domain / class-incremental** scenarios — and show many methods only work in the easy one.
- **2022–2025 (LLM era):** model merging (task arithmetic, TIES), LoRA-as-continual-learning ("LoRA Learns Less and Forgets Less"), continued-pretraining recipes, test-time training, and results that **on-policy RL forgets less than SFT**.

- [ ] For your own work (or a product you use), name one case each for RAG, fine-tuning, and continual learning
- [ ] Explain to a friend why RAG does not cause catastrophic forgetting
      */}),
      resources: [
        { title: "Wang et al. — A Comprehensive Survey of Continual Learning (2023)", url: "https://arxiv.org/abs/2302.00487", type: "paper", note: "the map of the field; read §1–3 and skim the method taxonomy" },
      ],
    },
    {
      id: "formalize-measure",
      title: "The problem, formalized: stability–plasticity, three scenarios, and the accuracy matrix",
      kind: "math",
      minutes: 75,
      md: MD(function () {/*
## Stability vs plasticity

A continual learner has two jobs that pull in opposite directions:

- **Plasticity** — learn the new task well (move the weights).
- **Stability** — keep old tasks working (don't move the weights).

Freeze everything → perfect stability, zero plasticity. Train freely → perfect plasticity, catastrophic forgetting. Every method is a different point on this dial, and every *hyperparameter* in those methods (EWC's $\lambda$, replay buffer size, LoRA rank) is literally the dial.

## Three scenarios (van de Ven & Tolias)

Same data, three different questions at test time — and wildly different difficulty:

| Scenario | At test time the model… | Split-MNIST example | Difficulty |
|---|---|---|---|
| **Task-incremental** | is *told* which task (e.g. "is this 2 or 3?") | separate "head" per task | easiest |
| **Domain-incremental** | must solve the same *kind* of problem, task unknown | "is it the first or second digit of its pair?" | medium |
| **Class-incremental** | must pick among **all classes seen so far** | "which of 0–9 is this?" | hardest |

Most real LLM updates are closest to **domain-incremental** (same next-token task, new data distribution). Many classic methods (EWC included) look great in task-IL and collapse in class-IL — so always report which scenario you ran.

## The accuracy matrix

> [!PREREQ] Refresher: indexing a matrix and taking averages
> A matrix is a grid of numbers; $R_{i,j}$ means "row $i$, column $j$". An **average** of $n$ numbers is their sum divided by $n$: $\frac{1}{n}\sum_{j=1}^{n} x_j$. The $\sum$ ("sigma") just means "add these up for $j = 1$ to $n$". That's all the math in this lesson.

Train on tasks $1 \dots T$ in order. After finishing task $i$, evaluate on **every** task $j$ and write the accuracy into $R_{i,j}$:

::viz accuracy-matrix

- The **diagonal** $R_{i,i}$ = how well you learned each task right after training on it (**plasticity**).
- **Below the diagonal** ($i > j$) = what happened to old task $j$ later on (**stability**).
- **Above the diagonal** ($i < j$) = accuracy on tasks you haven't trained on yet (**forward transfer**).

From this one matrix come the three standard metrics (Lopez-Paz & Ranzato, GEM, 2017):

$$
\text{ACC} = \frac{1}{T}\sum_{j=1}^{T} R_{T,j}
\qquad
\text{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\left(R_{T,j} - R_{j,j}\right)
\qquad
\text{FWT} = \frac{1}{T-1}\sum_{j=2}^{T}\left(R_{j-1,j} - b_j\right)
$$

- **ACC (average accuracy)** — average of the **last row**: how good is the final model on everything?
- **BWT (backward transfer)** — for each old task, "final accuracy minus accuracy right after learning it", averaged. **Negative = forgetting.** −40% means old tasks lost 40 points on average. Positive (rare) means learning later tasks *helped* old ones.
- **FWT (forward transfer)** — did earlier tasks help a task *before* you trained on it? $b_j$ is the accuracy of a randomly-initialized model on task $j$ (the baseline).

## Worked example (3 tasks)

| after training ↓ / eval on → | task 1 | task 2 | task 3 |
|---|---|---|---|
| task 1 | **99** | 12 | 8 |
| task 2 | 60 | **98** | 10 |
| task 3 | 40 | 70 | **99** |

- ACC = (40 + 70 + 99) / 3 = **69.7%**
- BWT = [(40 − 99) + (70 − 98)] / 2 = (−59 − 28) / 2 = **−43.5 points** → heavy forgetting
- FWT with $b_2 = b_3 = 10$: [(12 − 10) + (10 − 10)] / 2 = **+1 point** → essentially none

> [!INTUITION] Why not just report ACC?
> A model that learns nothing new (frozen) and a model that forgets everything old can have similar ACC. BWT tells you *which failure* you have. In LLM land the same idea appears as "new-domain score vs regression on general benchmarks" — two numbers, never one.

Two other numbers people report: **average forgetting** (for each old task, *best* accuracy ever minus final accuracy, averaged) and **intransigence** (how much worse you learn a new task than a model trained on it alone — plasticity loss).

- [ ] Compute ACC and BWT by hand for the naive Split-MNIST table in the first lesson (answer: ACC ≈ 20%, BWT ≈ −99%)
- [ ] Write `metrics(R)` in Python returning ACC, BWT, FWT (with a `baseline` list argument)
- [ ] In one sentence: why can FWT be ~0 in class-incremental Split-MNIST no matter what you do?
      */}),
      resources: [
        { title: "van de Ven & Tolias — Three scenarios for continual learning", url: "https://arxiv.org/abs/1904.07734", type: "paper", note: "task/domain/class-incremental; why class-IL is hard" },
        { title: "Lopez-Paz & Ranzato — Gradient Episodic Memory (GEM)", url: "https://arxiv.org/abs/1706.08840", type: "paper", note: "defines ACC / BWT / FWT from the accuracy matrix" },
      ],
    },
    {
      id: "method-families",
      title: "The four families of fixes: regularize, replay, isolate, merge",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Every continual-learning method is one of four ideas (or a mix). Learn the idea, not the zoo of acronyms.

## 1. Regularization — "don't move the weights that matter"

Add a penalty to the loss that grows when you change parameters that were important for old tasks.

- **EWC** (Elastic Weight Consolidation): importance = **Fisher information** per weight; penalty is a weighted squared distance to the old weights. Math in the next lesson.
- **SI** (Synaptic Intelligence) / **MAS**: same idea, different importance estimates (path integral of the loss change; gradient of output magnitude).
- **LwF** (Learning without Forgetting): regularize **outputs** instead of weights — distill the old model's predictions on new inputs (a KL term). This is the ancestor of the **KL penalty to a reference model** in RLHF/GRPO (M22)!

✅ no stored data (privacy-friendly) · ❌ weak in class-incremental settings; one $\lambda$ to tune.

## 2. Replay (rehearsal) — "keep showing it the old stuff"

Keep a small **memory buffer** of old examples and mix them into every new batch.

- **Experience replay**: store real examples (reservoir sampling keeps a uniform sample of an unbounded stream). Surprisingly strong: "On Tiny Episodic Memories" showed even 1 example per class helps a lot.
- **Generative replay**: train a generator (VAE/GAN — or, for LLMs, *the LLM itself*) to produce pseudo-old-data, so you don't need to store the originals.
- **DER / dark experience replay**: store old **logits** too and distill them.

✅ the most reliable baseline, works in class-IL · ❌ storage and privacy; compute grows with buffer usage.

> [!REAL] Replay is how LLM labs do it
> Continued pretraining recipes mix a few percent of the **original pretraining distribution** back into the new data. Fine-tuning pipelines mix in general instruction data. It's replay, just with a less academic name: **data mixing**.

## 3. Parameter isolation — "give each task its own weights"

- **Adapters / LoRA per task**: freeze the base, train a small adapter per task, pick the adapter at inference. **Zero forgetting of the base** by construction — the base weights never change.
- **Progressive networks**: add a new column of layers per task with lateral connections to frozen old columns.
- **PackNet / masks**: prune and freeze a subset of weights per task.

✅ no forgetting; trivial rollback (delete the adapter) · ❌ needs a task/adapter ID at inference (→ **routing**), parameters grow with tasks, less knowledge sharing.

::viz lora

## 4. Model merging — "train separately, combine the weights"

Fine-tune copies of the base on different tasks, then combine the **weights** arithmetically — no extra training.

- **Model soups**: average weights of several fine-tunes of the same base (often *better* than any single one).
- **Task arithmetic**: a **task vector** is $\tau = \theta_{\text{fine-tuned}} - \theta_{\text{base}}$. Add several: $\theta = \theta_{\text{base}} + \lambda \sum_k \tau_k$. You can even **subtract** one to remove a behaviour.
- **TIES-merging**: before adding task vectors, **T**rim small changes, **E**lect a sign per parameter by majority, and merge only agreeing values — fixes interference between tasks.

```yaml
# ties.yml — merge two fine-tunes of the same base with mergekit
models:
  - model: ./qwen-sql            # fused fine-tune #1
    parameters: { weight: 1.0, density: 0.5 }
  - model: ./qwen-medical        # fused fine-tune #2
    parameters: { weight: 1.0, density: 0.5 }
merge_method: ties
base_model: Qwen/Qwen2.5-0.5B-Instruct
parameters: { normalize: true }
dtype: bfloat16
```

```bash
pip install mergekit
mergekit-yaml ties.yml ./qwen-merged      # CPU is fine for a 0.5B model
```

✅ no training, cheap, great for combining skills · ❌ only works for fine-tunes of the **same base**; quality needs evals.

## Cheat sheet

| Family | Needs old data? | Forgetting | Extra cost | Serving impact |
|---|---|---|---|---|
| Regularization (EWC, LwF) | no | reduced | Fisher / old model | none |
| Replay / data mixing | yes (a little) | strongly reduced | more tokens per step | none |
| Isolation (LoRA per task) | no | none on base | adapter per task | **multi-LoRA serving, routing** |
| Merging (soups, TIES) | no | depends | one merge job | one model to serve |

> [!IMPORTANT] How this connects
> Families 3 and 4 move the continual-learning problem **out of training and into serving**: which adapter, which merged checkpoint, which version is live. That's the last lesson.

- [ ] For each family, write one sentence on when you'd *not* use it
- [ ] Explain why LwF's distillation term and GRPO's KL-to-reference term are the same idea
      */}),
      resources: [
        { title: "Ilharco et al. — Editing Models with Task Arithmetic", url: "https://arxiv.org/abs/2212.04089", type: "paper", note: "task vectors: add, subtract, combine fine-tunes" },
        { title: "Yadav et al. — TIES-Merging", url: "https://arxiv.org/abs/2306.01708", type: "paper", note: "trim, elect sign, merge — reduces interference" },
        { title: "Wortsman et al. — Model soups", url: "https://arxiv.org/abs/2203.05482", type: "paper", note: "averaging fine-tuned weights improves accuracy" },
        { title: "Shin et al. — Continual Learning with Deep Generative Replay", url: "https://arxiv.org/abs/1705.08690", type: "paper", note: "replay without storing data" },
        { title: "Rusu et al. — Progressive Neural Networks", url: "https://arxiv.org/abs/1606.04671", type: "paper", note: "the classic parameter-isolation architecture" },
        { title: "Chaudhry et al. — On Tiny Episodic Memories in Continual Learning", url: "https://arxiv.org/abs/1902.10486", type: "paper", note: "why a tiny replay buffer is such a strong baseline" },
        { title: "mergekit", url: "https://github.com/arcee-ai/mergekit", type: "tool", note: "the standard tool for task-arithmetic / TIES / soups merges" },
      ],
    },
    {
      id: "ewc-math",
      title: "EWC math: Fisher information as “how important is each weight?”",
      kind: "math",
      minutes: 60,
      md: MD(function () {/*
> [!PREREQ] Refresher 1: a gradient, and why we square it
> The gradient $\frac{\partial L}{\partial \theta_i}$ says how much the loss changes if you nudge weight $\theta_i$ a tiny bit. Big magnitude = the loss is **sensitive** to that weight. Its sign (up/down) varies between examples, so averaging raw gradients cancels out. **Squaring** makes every contribution positive, so the average measures *how sensitive*, not *which direction*.

> [!PREREQ] Refresher 2: a weighted squared distance
> The usual squared distance between two vectors is $\sum_i (a_i - b_i)^2$. A **weighted** one multiplies each term by a weight $w_i \ge 0$: $\sum_i w_i (a_i - b_i)^2$. Big $w_i$ → moving along coordinate $i$ is expensive; $w_i = 0$ → free.

## The idea in one picture

After task A, weights sit at $\theta^{A}$. Some weights matter a lot for A (a small change wrecks accuracy); many barely matter. EWC ties each weight back to $\theta^{A}$ with a **spring** whose stiffness is that weight's importance $F_i$:

$$
L_{\text{total}}(\theta) = L_B(\theta) + \frac{\lambda}{2} \sum_i F_i \,\big(\theta_i - \theta^{A}_i\big)^2
$$

- $L_B$ — the normal loss on the new task B.
- $F_i$ — importance of weight $i$ for task A (the **Fisher information**, below).
- $\lambda$ — global stiffness: the stability–plasticity dial.

Each penalty term is a parabola in $\theta_i$ — the same $x^2$ you can play with here. Its slope $\lambda F_i(\theta_i - \theta_i^{A})$ is the "spring force" pulling the weight back:

::viz derivative {"fn":"x²"}

## Where $F_i$ comes from

The (diagonal, empirical) **Fisher information** for weight $i$ is the average squared gradient of the log-likelihood over task-A data:

$$
F_i = \frac{1}{N}\sum_{n=1}^{N} \left(\frac{\partial \log p_\theta(y_n \mid x_n)}{\partial \theta_i}\right)^2 \Bigg|_{\theta = \theta^{A}}
$$

Since cross-entropy loss is $-\log p(y \mid x)$, that's just: **run backward on single task-A examples, square each weight's gradient, average.** Ten lines of PyTorch.

**Worked example (2 weights).** Three task-A examples give per-weight gradients $(0.9, 0.01)$, $(-1.1, 0.02)$, $(1.0, -0.01)$. Then

- $F_1 = (0.81 + 1.21 + 1.00)/3 \approx 1.01$ — weight 1 is **important**
- $F_2 = (0.0001 + 0.0004 + 0.0001)/3 = 0.0002$ — weight 2 is **free**

With $\lambda = 100$, moving weight 1 by 0.1 costs $\frac{100}{2}\cdot 1.01 \cdot 0.01 \approx 0.5$ (a lot, compared to a typical loss of ~0.1); moving weight 2 by 0.1 costs $0.0001$. So task B is learned mostly with "free" weights. That's the whole trick.

<details><summary>Why is it called Fisher information? (Bayesian view, optional)</summary>

Treat training as finding the most probable weights given the data. After task A the posterior over weights is roughly a Gaussian centred at $\theta^{A}$; its **precision** (1/variance) along each weight is the curvature of the loss there. Under mild assumptions that curvature equals the Fisher information. Learning B while keeping the A-posterior as a prior gives exactly the quadratic penalty above. The "true" Fisher samples $y$ from the model's own predictions; the "empirical" Fisher uses the real labels — we use the empirical one for simplicity.

</details>

## Multiple tasks and known limitations

- For tasks A, B, C…: keep one $(F, \theta^{*})$ pair per task and sum the penalties, or keep a running average ("**online EWC**") to save memory.
- Memory: EWC stores **two extra copies** of the parameters (Fisher + anchor). For an 8B LLM in BF16 that's +32 GB — why LLM practitioners prefer replay or LoRA.
- The diagonal ignores interactions between weights, and in class-incremental settings EWC barely helps (you'll see this in the build).

> [!INTUITION] Connect it to RL post-training
> GRPO/PPO's "KL penalty to the reference model" (M22) keeps the **outputs** near the old model; EWC keeps the **weights** near the old model. Both are springs that trade plasticity for stability.

- [ ] Compute $F_1, F_2$ for the example yourself and the penalty for moving both weights by 0.1 with $\lambda = 1000$
- [ ] Explain in one sentence why we square the gradients instead of averaging them directly
- [ ] Estimate EWC's extra memory for Qwen2.5-0.5B in FP32
      */}),
      resources: [
        { title: "Kirkpatrick et al. — Overcoming catastrophic forgetting in neural networks (EWC)", url: "https://arxiv.org/abs/1612.00796", type: "paper", note: "the EWC paper; Fig. 1 is the whole idea" },
      ],
    },
    {
      id: "build-split-mnist",
      title: "Build: Split-MNIST with naive, replay and EWC — and the accuracy matrix",
      kind: "build",
      minutes: 150,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
One script, four strategies, two scenarios. It keeps the whole dataset on the GPU as tensors (no `DataLoader`), so each run takes ~1–2 minutes on an M-series Mac.

```bash
uv pip install torch torchvision
```

```python
# split_mnist.py — catastrophic forgetting on Split-MNIST, and three fixes.
#   python split_mnist.py --strategy naive
#   python split_mnist.py --strategy replay --buffer 1000
#   python split_mnist.py --strategy ewc --lam 10000
#   python split_mnist.py --strategy joint            # upper bound: retrain on everything seen so far
#   add --scenario task for task-incremental (multi-head) instead of class-incremental
import argparse, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

p = argparse.ArgumentParser()
p.add_argument("--strategy", choices=["naive", "replay", "ewc", "joint"], default="naive")
p.add_argument("--scenario", choices=["class", "task"], default="class")
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lam", type=float, default=10000.0)   # EWC strength
p.add_argument("--buffer", type=int, default=1000)     # replay memory size (examples, all tasks)
p.add_argument("--fisher-samples", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)                  # save the accuracy matrix as JSON
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ---------- data: 5 tasks of 2 digits each ----------
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
T = len(TASKS)
def load(train):
    ds = datasets.MNIST("data", train=train, download=True)
    return ds.data.float().div(255).view(-1, 784), ds.targets.clone()
Xtr, Ytr = load(True)
Xte, Yte = load(False)
def split(X, Y, t):
    m = (Y == TASKS[t][0]) | (Y == TASKS[t][1])
    return X[m].to(dev), Y[m].to(dev)
train_sets = [split(Xtr, Ytr, t) for t in range(T)]
test_sets = [split(Xte, Yte, t) for t in range(T)]

# task-incremental = each task only competes among its own 2 output units ("multi-head")
TASK_MASK = torch.full((T, 10), float("-inf"))
for t, (a, b) in enumerate(TASKS):
    TASK_MASK[t, [a, b]] = 0.0
TASK_MASK = TASK_MASK.to(dev)

model = nn.Sequential(nn.Linear(784, 400), nn.ReLU(),
                      nn.Linear(400, 400), nn.ReLU(),
                      nn.Linear(400, 10)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def logits(x, tid):
    out = model(x)
    if args.scenario == "task":
        out = out + TASK_MASK[tid]          # tid: int or LongTensor of task ids
    return out

@torch.no_grad()
def evaluate(t):
    model.eval()
    x, y = test_sets[t]
    acc = (logits(x, t).argmax(1) == y).float().mean().item()
    model.train()
    return acc

# ---------- EWC: remember (Fisher, weights) after each task ----------
ewc_memory = []                             # list of (fisher_dict, theta_star_dict)
def fisher_diagonal(t):
    x, y = train_sets[t]
    idx = torch.randperm(len(x), device=dev)[: args.fisher_samples]
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for i in idx:                           # one example at a time: F_i = E[(d log p / d theta_i)^2]
        model.zero_grad()
        F.cross_entropy(logits(x[i : i + 1], t), y[i : i + 1]).backward()
        for n, p in model.named_parameters():
            fisher[n] += p.grad.detach() ** 2
    model.train()
    return {n: f / len(idx) for n, f in fisher.items()}

def ewc_penalty():
    total = 0.0
    for fisher, star in ewc_memory:
        for n, p in model.named_parameters():
            total = total + (fisher[n] * (p - star[n]) ** 2).sum()
    return args.lam / 2 * total

# ---------- Replay: a small memory of past examples ----------
buf_x = torch.empty(0, 784, device=dev)
buf_y = torch.empty(0, dtype=torch.long, device=dev)
buf_t = torch.empty(0, dtype=torch.long, device=dev)

# ---------- the continual-learning loop ----------
R = [[0.0] * T for _ in range(T)]           # R[i][j] = accuracy on task j after training task i
t0 = time.time()
for t in range(T):
    x, y = train_sets[t]
    tid = torch.full((len(x),), t, dtype=torch.long, device=dev)
    if args.strategy == "joint":            # cheat/upper bound: all data seen so far
        x = torch.cat([train_sets[k][0] for k in range(t + 1)])
        y = torch.cat([train_sets[k][1] for k in range(t + 1)])
        tid = torch.cat([torch.full((len(train_sets[k][0]),), k, dtype=torch.long, device=dev) for k in range(t + 1)])
    for ep in range(args.epochs):
        perm = torch.randperm(len(x), device=dev)
        for s in range(0, len(x), 128):
            b = perm[s : s + 128]
            loss = F.cross_entropy(logits(x[b], tid[b]), y[b])
            if args.strategy == "replay" and len(buf_x):
                r = torch.randint(len(buf_x), (128,), device=dev)
                loss = loss + F.cross_entropy(logits(buf_x[r], buf_t[r]), buf_y[r])
            if args.strategy == "ewc" and ewc_memory:
                loss = loss + ewc_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    # after finishing task t: consolidate
    if args.strategy == "ewc":
        star = {n: p.detach().clone() for n, p in model.named_parameters()}
        ewc_memory.append((fisher_diagonal(t), star))
    if args.strategy == "replay":
        k = args.buffer // T                # equal slots per task
        keep = torch.randperm(len(x), device=dev)[:k]
        buf_x = torch.cat([buf_x, x[keep]])
        buf_y = torch.cat([buf_y, y[keep]])
        buf_t = torch.cat([buf_t, tid[keep]])
    R[t] = [evaluate(j) for j in range(T)]
    print(f"after task {t} {TASKS[t]}: " + " ".join(f"{a:5.1%}" for a in R[t]))

# ---------- metrics ----------
acc = sum(R[T - 1]) / T
bwt = sum(R[T - 1][j] - R[j][j] for j in range(T - 1)) / (T - 1)
print(f"\n{args.strategy}/{args.scenario}: ACC={acc:.1%}  BWT={bwt:+.1%}  ({time.time() - t0:.0f}s on {dev})")
if args.out:
    json.dump({"strategy": args.strategy, "scenario": args.scenario, "R": R, "ACC": acc, "BWT": bwt}, open(args.out, "w"))
```

## Walk through the three strategies

| Strategy | Lines that matter | What they do |
|---|---|---|
| **naive** | the inner loop only | plain fine-tuning, task after task |
| **replay** | `buf_x/buf_y/buf_t` + the second `cross_entropy` | after each task, keep 200 random examples; every step adds a loss on 128 random memories |
| **ewc** | `fisher_diagonal`, `ewc_penalty` | after each task, snapshot weights + Fisher; add $\frac{\lambda}{2}\sum F_i(\theta_i-\theta^*_i)^2$ |
| **joint** | the `if args.strategy == "joint"` block | retrain on all data so far — not continual, but the **upper bound** you compare against |

`TASK_MASK` implements **task-incremental** evaluation: adding $-\infty$ to the logits of other tasks' classes means only the current pair competes — the "multi-head" setting. Without it (the default) you're in the hard **class-incremental** setting.

## Run the grid

```bash
for s in naive replay ewc joint; do
  for sc in class task; do
    python split_mnist.py --strategy $s --scenario $sc --out results_${s}_${sc}.json
  done
done
```

What you should see (numbers from one CPU run, seed 0; yours will vary by a point or two):

| strategy | class-IL ACC | class-IL BWT | task-IL ACC | task-IL BWT |
|---|---|---|---|---|
| naive | 19.6% | −99.3% | 96.7% | −3.5% |
| EWC ($\lambda$=1e4) | 19.7% | −99.4% | 99.0% | −0.7% |
| replay (1,000) | 90.0% | −10.8% | 99.1% | −0.6% |
| joint (upper bound) | 97.6% | −0.2% | — | — |

> [!IMPORTANT] The lesson hiding in that table
> EWC **does not help** in class-incremental Split-MNIST — the problem there is the output layer choosing between classes that were never trained together, which no per-weight penalty can fix. Replay fixes it because old and new classes appear **in the same batch**. This is why "which scenario?" is the first question to ask about any continual-learning claim.

## Plot the matrices

```python
# plot_matrices.py
import json, glob, matplotlib.pyplot as plt
files = sorted(glob.glob("results_*_class.json"))
fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 3.6))
for ax, f in zip(axes, files):
    r = json.load(open(f))
    ax.imshow(r["R"], vmin=0, vmax=1, cmap="viridis")
    for i, row in enumerate(r["R"]):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(f"{r['strategy']}  ACC={r['ACC']:.0%}  BWT={r['BWT']:+.0%}")
    ax.set_xlabel("evaluated on task"); ax.set_ylabel("after training task")
plt.tight_layout(); plt.savefig("accuracy_matrices.png", dpi=150)
```

## Experiments (do at least three)

- [ ] Ran all 8 configurations and saved `accuracy_matrices.png`
- [ ] **Buffer sweep:** replay with `--buffer` 50, 200, 1000, 5000 → plot ACC vs buffer size. Where are diminishing returns?
- [ ] **λ sweep:** EWC with `--lam` 10, 1e2, 1e3, 1e4, 1e5 in task-IL → plot ACC and BWT. Find where plasticity starts to suffer (diagonal drops)
- [ ] **Epochs:** does training *longer* on each task make forgetting worse? (`--epochs 1` vs `5`)
- [ ] Add **EWC + replay** together (both terms in the loss). Better than either alone?
- [ ] Add a 4th strategy — **LwF**: keep a frozen copy of the model after each task and add a KL term between old and new logits on the *current* batch
- [ ] Repeat with 3 seeds and report mean ± std (a single seed is not a result)

> [!TIP] Going further: Avalanche
> [Avalanche](https://github.com/ContinualAI/avalanche) (ContinualAI) implements these benchmarks and ~20 strategies (EWC, LwF, GEM, iCaRL, GDumb, replay variants) with standard metrics. Once your own implementation works, reproduce one number with Avalanche to check yourself.
      */}),
      resources: [
        { title: "Avalanche (ContinualAI)", url: "https://github.com/ContinualAI/avalanche", type: "repo", note: "reference implementations and benchmarks in PyTorch" },
      ],
    },
    {
      id: "llm-continual",
      title: "Continual learning for LLMs: continued pretraining, SFT vs RL, test-time training, memory",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Everything above transfers to LLMs, but the scale changes which tools are practical: storing a Fisher copy of a 70B model is absurd, while **mixing in old data** and **training adapters** are cheap. Here's the LLM-era landscape.

## 1. Continued pretraining & data mixing

Labs refresh a model's knowledge by continuing next-token pretraining on new data (newer web crawls, code, a new language). The well-documented recipe (Ibrahim et al., 2024, "Simple and Scalable Strategies to Continually Pre-train LLMs"):

- **Re-warm and re-decay the learning rate** for the new phase (a cosine schedule that ends at ~0 can't learn more without warming back up).
- **Replay** a small fraction of the previous pretraining data mixed into the new data to limit forgetting.
- With those two, continued pretraining can roughly match re-training from scratch on the union of data — at a fraction of the compute.

> [!REAL] This is also what "domain adaptation" means in industry
> A company adapting an open model to its codebase or medical notes runs a mini version of this: new-domain tokens + a slice of general data + a low LR, evaluated on both domain and general benchmarks.

## 2. Forgetting in SFT, LoRA and RL

Three findings worth knowing (and citing correctly in interviews):

- **Fine-tuning forgets, and it's measurable.** Continual instruction tuning of 1B–7B models shows clear forgetting of general knowledge and reasoning (Luo et al., 2023).
- **LoRA learns less and forgets less** (Biderman et al., 2024): on code and math, standard low-rank LoRA underperforms full fine-tuning on the *target* domain but better preserves performance *outside* it — more than weight decay or dropout do. Full fine-tuning learns weight perturbations of much higher rank than typical LoRA configs.
- **RL's Razor** (Shenfeld, Pari & Agrawal, 2025): at *matched* new-task performance, **on-policy RL forgets significantly less than SFT**. The amount of forgetting is predicted by the **KL divergence between the fine-tuned and base policy measured on the new task**; on-policy RL is implicitly biased toward KL-minimal solutions, while SFT can move arbitrarily far from the base model.

> [!INTUITION] Why would RL forget less?
> SFT pushes the model toward *someone else's* outputs (the dataset), which may be far from what the model would naturally say. On-policy RL only reinforces or suppresses things the model **already samples** — so it changes the distribution as little as needed. Same stability–plasticity dial, different default setting. It's also why the KL term in GRPO/PPO matters.

## 3. Test-time training / adaptation (TTT)

Instead of updating once and serving forever, **update per input at inference time**:

- **TTT** (Sun et al., 2019): before predicting on a test example, take a few gradient steps on a self-supervised loss on that example.
- **TTT for LLM few-shot reasoning** (Akyürek et al., 2024): train a temporary LoRA on the few-shot examples of *each* problem (with augmentations), then answer — big gains on ARC-style abstract reasoning.

Inference-engineering consequence: a "request" now contains a **mini training job** plus a **per-request adapter** — compute, memory and latency budgets change completely. Throw the adapter away afterwards and there's no forgetting at all.

## 4. Memory-augmented agents

Most "agents that learn" in production today don't touch weights: they write **memories** (facts, summaries, reflections, reusable skills/code) to a store and retrieve them into the context later. Cheap, reversible and inspectable — but it lives in the context window, so it costs **prefill tokens and KV cache** on every call. The frontier combines both: memories now, periodic consolidation into weights (a LoRA) later — which brings you straight back to replay, evals and versioning.

## Decision table

| You need… | Reach for | Forgetting risk |
|---|---|---|
| fresh facts, citations | RAG | none (weights unchanged) |
| new format/skill for one customer | LoRA adapter per customer | none on base |
| a domain the model should "just know" | continued pretraining / full FT + **replay** + low LR | medium — measure it |
| better reasoning from rewards | on-policy RL with KL control | lower than SFT (RL's Razor) |
| to combine several fine-tunes | model merging (TIES / task arithmetic) | depends — eval gate |
| per-problem adaptation | test-time training | none (ephemeral) |

- [ ] Pick one row and describe a real product that would use it
- [ ] Explain RL's Razor to a colleague in two sentences, including what KL is measured on
- [ ] Why is EWC rarely used for LLMs but replay is used everywhere?
      */}),
      resources: [
        { title: "Ibrahim et al. — Simple and Scalable Strategies to Continually Pre-train LLMs", url: "https://arxiv.org/abs/2403.08763", type: "paper", note: "LR re-warming/re-decaying + replay recipe" },
        { title: "Biderman et al. — LoRA Learns Less and Forgets Less", url: "https://arxiv.org/abs/2405.09673", type: "paper", note: "LoRA vs full FT: target gains vs forgetting" },
        { title: "Shenfeld, Pari & Agrawal — RL's Razor: Why Online RL Forgets Less", url: "https://arxiv.org/abs/2509.04259", type: "paper", note: "forgetting predicted by KL on the new task; RL vs SFT" },
        { title: "Luo et al. — Catastrophic Forgetting in LLMs During Continual Fine-tuning", url: "https://arxiv.org/abs/2308.08747", type: "paper", note: "empirical forgetting in 1B–7B instruction tuning" },
        { title: "Sun et al. — Test-Time Training with Self-Supervision", url: "https://arxiv.org/abs/1909.13231", type: "paper", note: "the original test-time training idea" },
        { title: "Akyürek et al. — The Surprising Effectiveness of Test-Time Training for Few-Shot Learning", url: "https://arxiv.org/abs/2411.07279", type: "paper", note: "per-problem LoRA at inference time for LLMs" },
      ],
    },
    {
      id: "lab-llm-forgetting",
      title: "Lab: measure an LLM forgetting (LoRA vs full fine-tune vs replay) with mlx-lm",
      kind: "lab",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
**Goal:** fine-tune **Qwen2.5-0.5B-Instruct** on a narrow new domain (text-to-SQL), then measure (a) how well it learned SQL and (b) how much general knowledge it lost. Four runs, one 2×4 accuracy matrix.

```bash
uv pip install "mlx-lm[train]" datasets
```

## Step 1 — build the data (domain + general + replay mix)

`mlx-community/wikisql` is already in mlx-lm's `{"text": ...}` format. Alpaca gives us a slice of "general" instruction data for replay and for a general held-out perplexity set.

```python
# make_data.py
import json, os, random
from datasets import load_dataset
random.seed(0)
sql = load_dataset("mlx-community/wikisql")
gen = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)

def dump(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

sql_train = [r["text"] for r in sql["train"]]
sql_valid = [r["text"] for r in sql["valid"]]
sql_test = [r["text"] for r in sql["test"]]
general = [r["text"] for r in gen.select(range(3000))]

dump("data/sql/train.jsonl", sql_train)
dump("data/sql/valid.jsonl", sql_valid)
dump("data/sql/test.jsonl", sql_test)

mix = sql_train + general[: len(sql_train) // 5]      # ~20% replay of general data
random.shuffle(mix)
dump("data/sql_replay/train.jsonl", mix)
dump("data/sql_replay/valid.jsonl", sql_valid)

dump("data/general/test.jsonl", general[2000:2500])  # held-out general text
print(len(sql_train), "sql train,", len(mix), "mixed")
```

## Step 2 — a general-knowledge probe (ARC-Easy)

Next-token scoring of the answer letter — no generation, fast, deterministic. It works with or without an adapter.

```python
# eval_arc.py — general-knowledge probe: ARC-Easy multiple choice
import argparse
import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter", default=None)
p.add_argument("--n", type=int, default=500)
args = p.parse_args()

model, tok = load(args.model, adapter_path=args.adapter)
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(args.n))
LET = ["A", "B", "C", "D", "E"]
let_ids = [tok.encode(" " + L)[-1] for L in LET]

correct = 0
for ex in ds:
    opts = ex["choices"]["text"]
    gold = ex["choices"]["label"].index(ex["answerKey"])
    prompt = ("Question: " + ex["question"] + "\n"
              + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer:")
    logits = model(mx.array(tok.encode(prompt))[None])[0, -1]   # next-token logits
    scores = [logits[let_ids[i]].item() for i in range(len(opts))]
    correct += int(max(range(len(opts)), key=lambda i: scores[i]) == gold)
print(f"ARC-Easy accuracy: {correct / len(ds):.1%} (n={len(ds)})")
```

## Step 3 — baseline, then four fine-tunes

```bash
M=Qwen/Qwen2.5-0.5B-Instruct
python eval_arc.py --model $M                                      # baseline general score
mlx_lm.lora --model $M --data data/sql --test                      # baseline SQL loss/ppl (no adapter yet)

# (1) LoRA (default rank, 16 layers)
mlx_lm.lora --model $M --train --data data/sql --iters 600 --adapter-path ad/lora
# (2) full fine-tune — expect more learning AND more forgetting
mlx_lm.lora --model $M --train --data data/sql --iters 600 --fine-tune-type full \
            --learning-rate 1e-5 --batch-size 2 --adapter-path ad/full
# (3) full fine-tune + replay of general data
mlx_lm.lora --model $M --train --data data/sql_replay --iters 600 --fine-tune-type full \
            --learning-rate 1e-5 --batch-size 2 --adapter-path ad/full_replay
# (4) LoRA + replay
mlx_lm.lora --model $M --train --data data/sql_replay --iters 600 --adapter-path ad/lora_replay

for a in lora full full_replay lora_replay; do
  python eval_arc.py --model $M --adapter ad/$a
  mlx_lm.lora --model $M --adapter-path ad/$a --data data/sql --test       # domain: SQL test ppl
  mlx_lm.lora --model $M --adapter-path ad/$a --data data/general --test   # general text ppl
done
```

> [!WARNING] Memory & time
> LoRA on a 0.5B model fits comfortably in 8 GB. The full fine-tune needs optimizer state for all weights — if you hit memory pressure, lower `--batch-size` to 1 or add `--grad-checkpoint`. Each run is ~5–20 minutes on an M-series Mac. To make forgetting **more visible**, try `--iters 2000` or `--learning-rate 5e-5` for the full run.

## Step 4 — fill in the matrix

| checkpoint | SQL test ppl ↓ (new task) | ARC-Easy acc ↑ (old skill) | general ppl ↓ (old distribution) | Δ ARC vs base |
|---|---|---|---|---|
| base | | | | 0 |
| LoRA | | | | |
| full FT | | | | |
| full FT + replay | | | | |
| LoRA + replay | | | | |

Also do a **qualitative check** — ask each checkpoint a general question:

```bash
mlx_lm.generate --model $M --adapter-path ad/full --prompt "Give me three tips for a job interview." --max-tokens 120
```

A classic symptom of forgetting in narrow fine-tunes is **format creep**: the model starts answering everything in the fine-tuning format (here, `SELECT ...`). Benchmarks can miss it; users won't.

> [!INTUITION] What you're likely to find
> The *direction* is predictable even if the magnitudes aren't: full fine-tuning gets the lowest SQL perplexity and the biggest general regression; LoRA learns a bit less and regresses less (Biderman et al.); mixing in general data shrinks the regression at a small cost in SQL. If you see **no** forgetting, your fine-tune is too gentle — push iterations or LR until you can see the trade-off, then report both regimes.

- [ ] Baseline ARC-Easy and perplexities recorded
- [ ] Four fine-tunes trained, all metrics filled in
- [ ] Format-creep check done on all checkpoints; one example pasted in your report
- [ ] One chart: x = SQL test ppl (lower = learned more), y = ARC-Easy acc (higher = forgot less); one point per run — the **stability–plasticity frontier**
- [ ] Stretch: `mlx_lm.fuse` two LoRAs (SQL + another domain), merge with mergekit TIES, and check whether the merged model keeps both skills
      */}),
      resources: [
        { title: "mlx-lm — LoRA / QLoRA / full fine-tuning guide", url: "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md", type: "docs", note: "flags for --train, --test, --fine-tune-type, data formats, fuse" },
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "LLM inference + fine-tuning on Apple silicon" },
      ],
    },
    {
      id: "serving-continual",
      title: "Serving continually-updated models: multi-LoRA, hot swaps, versioning, eval gates",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Training produces a new adapter or checkpoint. **Getting it safely in front of users — and back out when it's bad — is inference engineering.** This is the part of continual learning you'll actually own.

## 1. Multi-LoRA serving: one base, thousands of adapters

If every customer/domain/task has its own LoRA (parameter isolation!), you don't want one GPU replica per adapter. You want **one base model in GPU memory** and adapters swapped per request, batched together.

- **Punica** (2023): a custom CUDA kernel (**SGMV**, segmented gather matrix-vector multiply) that computes $xW + xB_iA_i$ for a batch where **each request uses a different adapter** — so heterogeneous-adapter requests share one batch.
- **S-LoRA** (2023): serves **thousands** of adapters by keeping them in CPU memory, paging active ones into GPU, and managing adapter weights and KV cache in a **unified paged memory pool** (the PagedAttention idea from M08, applied to adapters).
- **vLLM** implements this natively:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules sql=./adapters/sql medical=./adapters/medical \
  --max-loras 8 --max-lora-rank 16 --max-cpu-loras 64
# a request picks an adapter via the "model" field
curl localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "sql", "prompt": "table: users ...", "max_tokens": 64}'
```

`--max-loras` = adapters that can be active in **one batch** (GPU slots); `--max-cpu-loras` = adapters cached in host memory. Each costs memory that would otherwise be KV cache — a real capacity trade-off.

**Hot-swapping at runtime:** with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, vLLM exposes `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter` (and resolver plugins that fetch unknown adapters on demand). The docs warn this is for **trusted, isolated environments only** — loading arbitrary weights from a request path is a security surface.

> [!REAL] Routing becomes adapter-aware
> With hundreds of adapters over many replicas, the router should send a request to a replica that **already has its adapter loaded** (and its prefix cached) — the Baseten book calls this LoRA-aware / cache-aware routing (Inference Engineering (Baseten) ch. 7.2). Same idea as KV-cache-aware routing in M18.

::viz load-balancing

## 2. Full-weight updates in a live engine

When you update the **base** weights (a new continued-pretraining checkpoint, a merged model, an RL step), you face exactly the machinery you built in M23 for **RL weight sync**:

- **Transport:** load from disk/object storage, or broadcast tensors from a trainer over NCCL/RDMA straight into the engine's GPU memory.
- **Consistency:** pause or drain in-flight requests (or finish them on the old weights) so no sequence mixes two models mid-generation.
- **Cache invalidation:** every cached KV block and prefix-cache entry was computed with the **old** weights — engines expose a reset/flush of the prefix cache for exactly this reason. Forgetting to flush serves stale activations.
- **Compiled artifacts:** CUDA graphs and `torch.compile` artifacts usually survive a same-shape weight swap; a changed architecture or quantization scheme needs a full restart.

## 3. Versioning and rollback

Treat every checkpoint and adapter like a container image:

- **Immutable IDs** (`qwen7b-sql@2026-09-12-a1b2c3`), content hashes, and a registry recording *data*, *base*, *hyperparameters* and *eval results* per version.
- Never overwrite in place: serve by version, move a **pointer** (alias) for "latest".
- **Rollback = move the pointer back** — which only works if the old version is still loadable (keep N previous adapters warm in CPU memory; keep old base checkpoints in object storage).

## 4. Eval gates: the continual-learning metrics, operationalized

Your accuracy matrix becomes a **release gate**:

| Gate | Check | Example threshold |
|---|---|---|
| New-task quality | domain eval (diagonal) | ≥ +X over current prod |
| **Regression** (BWT!) | general benchmarks + previous tasks' evals + safety evals | no metric drops > 1 point |
| Behavioural | format-creep / refusal / tone probes | 0 new failures on golden prompts |
| Performance | TTFT/TPOT p99, throughput under load (M19) | within 5% of current |
| Online | shadow traffic → canary 1% → 10% → 100% with auto-rollback on SLO or quality alarms | Baseten ch. 7.4 |

> [!IMPORTANT] How this connects
> Continual learning in production = **train (M05, M22) → evaluate like M24 → ship like M18–M19 → sync weights like M23**. The model that "keeps learning" is really a pipeline that keeps shipping, measured by an accuracy matrix that never stops growing.

- [ ] Draw the pipeline: data → train → eval gate → registry → canary → full rollout → monitor → rollback
- [ ] For a 7B model with rank-16 LoRAs on all attention + MLP projections, estimate adapter size and how many fit in 2 GB
- [ ] Write the list of caches and compiled artifacts you'd invalidate when swapping base weights
      */}),
      resources: [
        { title: "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", url: "https://arxiv.org/abs/2311.03285", type: "paper", note: "unified paging for adapters + KV cache" },
        { title: "Punica: Multi-Tenant LoRA Serving", url: "https://arxiv.org/abs/2310.18547", type: "paper", note: "SGMV kernel: different adapters in one batch" },
        { title: "vLLM docs — LoRA adapters", url: "https://docs.vllm.ai/en/latest/features/lora.html", type: "docs", note: "--enable-lora, --max-loras, runtime load/unload endpoints" },
      ],
    },
  ],

  challenge: {
    title: "Forgetting report + a safe continuous-update design",
    md: MD(function () {/*
Put everything in `course-work/m24/` as a short report with charts — this is a portfolio piece.

**Part A — accuracy matrices (Split-MNIST, 5 tasks).** Three strategies (**naive, replay, EWC**) × both scenarios (class-IL and task-IL), 3 seeds each. Show the 5×5 matrices as heatmaps, a table of ACC and BWT (mean ± std), plus the **joint** upper bound. Include one sweep (buffer size or $\lambda$) as a stability–plasticity curve.

**Part B — LLM forgetting.** The mlx-lm lab: base vs LoRA vs full FT vs replay variants, with a new-domain metric, a general-knowledge metric, and a qualitative format-creep example. One scatter plot of "learned" vs "forgot".

**Part C — design note (1–2 pages): "Continuously updating a production model safely."** Scenario: a 7B assistant serving 500 req/s gets a new domain LoRA every week and a refreshed base model every quarter. Cover: data pipeline and replay mix; eval gates (which suites, which thresholds, who can override); adapter registry and versioning; serving topology (multi-LoRA limits, adapter-aware routing); weight rollout (shadow → canary → full) and cache invalidation; rollback time objective; what you'd monitor after launch.
    */}),
    checklist: [
      "5×5 accuracy matrices for naive, replay and EWC in both class-IL and task-IL, with ACC and BWT over 3 seeds",
      "A stability–plasticity sweep (buffer size or $\\lambda$) with a chart and a one-paragraph interpretation",
      "LLM experiment table: new-domain metric and general metric for base, LoRA, full FT and at least one replay run",
      "A qualitative forgetting example (format creep or lost skill) included verbatim",
      "Design note covers eval gates, versioning, multi-LoRA serving, canary + rollback, and prefix-cache invalidation",
      "You can explain why EWC fails in class-incremental Split-MNIST while replay works",
    ],
    stretch: "Implement a tiny **multi-adapter server**: load the base with mlx-lm once, keep 3 LoRA adapters in memory, and route each request to its adapter by a `model` field — then measure the latency cost of switching adapters between requests vs batching requests per adapter.",
  },

  connects: MD(function () {/*
Continual learning ties the whole curriculum together: the **training loop** (M01, M05), **LoRA** (M05) and **RL with KL control** (M22) are the learning side; **multi-LoRA serving, routing, canaries and benchmarks** (M18–M19) and **weight sync** (M23) are the serving side. The final module turns everything you've built — including this report — into a portfolio and a job search.
  */}),

  interview: [
    "What is catastrophic forgetting and why does naive fine-tuning cause it? Give a concrete example from LLM fine-tuning.",
    "Define ACC and BWT from an accuracy matrix. What does a BWT of −30% mean?",
    "Explain EWC. What is the Fisher information intuitively, and why is EWC rarely used on large LLMs?",
    "Why does EWC work in task-incremental but not class-incremental Split-MNIST, while replay works in both?",
    "Compare RAG, LoRA fine-tuning and continued pretraining for keeping a model up to date. When would you choose each?",
    "You serve 2,000 customer LoRA adapters on one base model. How do you design memory management, batching and routing?",
    "You swap base weights in a running inference engine after an RL step. What can go wrong, and what must you invalidate?",
    "Design an eval gate for weekly model updates. What would block a release?",
  ],

  resources: [
    { title: "Kirkpatrick et al. — EWC (2017)", url: "https://arxiv.org/abs/1612.00796", type: "paper", note: "the canonical regularization method" },
    { title: "Wang et al. — A Comprehensive Survey of Continual Learning (2023)", url: "https://arxiv.org/abs/2302.00487", type: "paper", note: "theory, method families, applications" },
    { title: "Avalanche (ContinualAI)", url: "https://github.com/ContinualAI/avalanche", type: "repo", note: "benchmarks + strategies + metrics in PyTorch" },
    { title: "S-LoRA (2023)", url: "https://arxiv.org/abs/2311.03285", type: "paper", note: "serving thousands of adapters" },
    { title: "Task Arithmetic (Ilharco et al.)", url: "https://arxiv.org/abs/2212.04089", type: "paper", note: "task vectors for editing/merging models" },
    { title: "TIES-Merging (Yadav et al.)", url: "https://arxiv.org/abs/2306.01708", type: "paper", note: "interference-aware merging" },
    { title: "mergekit (arcee-ai)", url: "https://github.com/arcee-ai/mergekit", type: "tool", note: "merge fine-tunes: linear, task arithmetic, TIES, DARE, SLERP" },
    { title: "RL's Razor (Shenfeld et al., 2025)", url: "https://arxiv.org/abs/2509.04259", type: "paper", note: "why on-policy RL forgets less than SFT" },
    { title: "LoRA Learns Less and Forgets Less (Biderman et al., 2024)", url: "https://arxiv.org/abs/2405.09673", type: "paper", note: "the LoRA vs full-FT forgetting trade-off" },
    { title: "HF PEFT", url: "https://huggingface.co/docs/peft/index", type: "docs", note: "LoRA and friends in the PyTorch/Transformers world" },
  ],
});
