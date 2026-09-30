Course.module({
  id: "m05-training-finetuning",
  title: "Training & fine-tuning: how models are made",
  short: "Training & fine-tuning (LoRA)",
  tagline: "Fine-tune Qwen2.5 with LoRA on your Mac (MLX) or Colab (TRL + PEFT), prove it got better with a real eval, serve the adapter, and understand the whole pipeline from pretraining to RL with back-of-envelope compute and memory math.",
  hours: 18,
  level: "core",
  runsOn: ["mac", "colab"],
  tags: ["training", "lora", "qlora", "sft", "mlx", "trl", "scaling-laws", "evals"],

  goal: MD(function () {/*
You take **Qwen2.5-0.5B-Instruct**, which is mediocre at text-to-SQL, fine-tune a **LoRA adapter** (~4M trainable parameters, under 1% of the model) on your Mac in about 15–30 minutes, and measure the improvement on a held-out test set **you froze before training**:

```text
$ python eval_sql.py                                  # base model
exact-match  38/200 (19.0%)   valid SQL  151/200 (75.5%)   41.2 tok/s
$ mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --train --data data --iters 1000 --mask-prompt ...
Iter 1000: Val loss 0.183, Val took 3.1s
$ python eval_sql.py --adapter adapters/sql-r8        # + your adapter
exact-match 131/200 (65.5%)   valid SQL  196/200 (98.0%)   39.8 tok/s
$ mlx_lm.fuse ... && mlx_lm.convert ... -q            # merge, then 4-bit
exact-match 127/200 (63.5%)   valid SQL  195/200 (97.5%)   142 tok/s
```

(Illustrative numbers: yours will differ. The *shape* of the result is what matters: a big task gain, a small quality cost from quantization, and a big speed gain.) Along the way you learn to estimate, on a napkin, **what it costs** to train any model: FLOPs ≈ $6ND$, bytes of GPU memory per parameter, and the Chinchilla-optimal token count.

::viz lora
  */}),
  demo: { viz: "lora" },

  why: MD(function () {/*
Inference engineers don't usually pretrain models, but they live downstream of training every day:
- **Most production deployments are fine-tunes.** Customers bring LoRA adapters or fully fine-tuned checkpoints. Baseten ch. 1.3.2 gives text-to-SQL as the canonical case: a few-billion-parameter fine-tune can match models with hundreds of billions of parameters on a narrow task.
- **Multi-LoRA serving** (one base model, thousands of adapters, swapped per request) is a core engine feature in vLLM, SGLang and TRT-LLM, and a routing problem (LoRA-aware routing).
- **Quantization, distillation and speculative-decoding drafters** are all training-adjacent. QAT, distilled students and fine-tuned EAGLE heads come out of training pipelines.
- **RL post-training is mostly inference** (rollouts). Modules 21–23 build on the training-loop intuition you get here.
- In interviews you'll be asked to estimate training and inference compute and memory from scratch. This module gives you the formulas and the reflexes.
  */}),

  prereqs: [
    {
      title: "Modules 03–04: a GPT training loop and a Llama/Qwen model file",
      skipIf: "you have trained a small transformer with AdamW and loaded HF weights into PyTorch",
      md: MD(function () {/*
You should be comfortable with `loss.backward()`, `optimizer.step()`, cross-entropy on next-token prediction, and the Qwen2 architecture (Module 04). The LoRA math lesson wraps `nn.Linear` layers of **your own** `qwen2.py`.
      */}),
    },
    {
      title: "Exponents, powers of ten and log-log plots",
      skipIf: "you can compute 6 × 7e9 × 2e12 in your head and know why power laws are straight lines on log-log axes",
      math: true,
      md: MD(function () {/*
- $10^{9}$ = a billion (B), $10^{12}$ = a trillion (T). Multiply by **adding exponents**: $10^9 \times 10^{12} = 10^{21}$.
- "7e9" is how code writes $7 \times 10^9$. For example, $6 \times 7\text{e}9 \times 2\text{e}12 = 84 \times 10^{21} = 8.4 \times 10^{22}$.
- A **FLOP** is one floating-point operation. **FLOPS** (capital S) is FLOPs *per second*. An H100 does ~$989 \times 10^{12}$ dense BF16 FLOPS.
- A **power law** $y = a x^{-b}$ becomes a straight line if you take logs of both axes: $\log y = \log a - b \log x$. Scaling-law plots are log-log for exactly this reason.

::viz exp-log
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: a 15-minute LoRA fine-tune on your Mac with mlx-lm",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
Fine-tuning first, theory after. Apple's **mlx-lm** can fine-tune small models on a MacBook's GPU with one command.

```bash
uv pip install "mlx-lm[train]" datasets

# 1) Before: ask the base model to write SQL in the dataset's raw format
mlx_lm.generate --model Qwen/Qwen2.5-0.5B-Instruct --ignore-chat-template --max-tokens 60 \
  --prompt $'table: 1-10015132-16\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What is terrence ross\' nationality\nA: '

# 2) Train a LoRA adapter on WikiSQL (pre-formatted on the HF Hub)
mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --train --data mlx-community/wikisql \
  --iters 300 --batch-size 4 --num-layers 8 --adapter-path adapters/wikisql

# 3) After: same prompt, now with the adapter
mlx_lm.generate --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters/wikisql --ignore-chat-template --max-tokens 60 \
  --prompt $'table: 1-10015132-16\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: What is terrence ross\' nationality\nA: '
```

Before, you typically get a chatty explanation or a malformed query. After, you get something like `SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrence Ross'`. Watch the training log while it runs:

```text
Trainable parameters: 0.089% (0.442M/494.033M)
Iter 1: Val loss 2.721, Val took 4.2s
Iter 10: Train loss 2.311, Learning Rate 1.000e-05, It/sec 3.9, Tokens/sec 1480, Peak mem 2.1 GB
...
Iter 300: Train loss 1.102, Val loss 1.196
Saved final weights to adapters/wikisql/adapters.safetensors.
```

(Your exact numbers depend on the mlx-lm version, chip and flags.)

> [!INTUITION] What just happened
> 1. The **base weights stayed frozen**. `adapters.safetensors` is a few MB; the model is ~1 GB.
> 2. For each chosen linear layer, mlx-lm added two tiny matrices $A$ and $B$ and trained **only those**, with next-token cross-entropy on your examples, exactly the loss from Module 03.
> 3. At generation time, each adapted layer computes $Wx + BAx$. The adapter "steers" the model toward your format and task.
> 4. **Train loss** is measured on batches it learns from. **Val loss** is on held-out data. When val stops falling while train keeps falling, you're overfitting.

```bash
ls -lh adapters/wikisql/          # adapter_config.json + adapters.safetensors (a few MB)
cat adapters/wikisql/adapter_config.json
```

This module explains **why** this works (low-rank updates), **what it costs** (compute and memory math), **how it fits** into the full pipeline that produced Qwen-Instruct in the first place, and **how to prove** it helped (evals). Then you do it properly on a task of your own.

- [ ] Ran the before/after comparison and saved both outputs
- [ ] Noted trainable-parameter %, tokens/sec and peak memory from the log
- [ ] Opened `adapter_config.json` and found the rank, scale and number of layers
      */}),
      resources: [
        { title: "mlx-lm — LoRA / QLoRA guide", url: "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md", type: "docs", note: "every flag, data format, fuse and memory tips" },
        { title: "ml-explore/mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "generate, lora, fuse, convert, server" },
      ],
    },
    {
      id: "lifecycle",
      title: "The lifecycle of an LLM: pretraining → mid-training → SFT → preference tuning / RL → distillation",
      kind: "concept",
      minutes: 75,
      runsOn: ["any"],
      md: MD(function () {/*
The Qwen2.5-0.5B-**Instruct** you just fine-tuned went through several stages, each with a very different scale. Know these numbers: they show up in every conversation about cost.

| Stage | What it does | Data | Compute (share) |
|---|---|---|---|
| **Pretraining** | next-token prediction on the internet, code, books; learns language + knowledge | **10–40 trillion tokens** (Llama 3: 15T+; Qwen2.5: 18T; Qwen3: 36T; DeepSeek-V3: 14.8T) | **>95%** of total |
| **Mid-training / annealing** | continue pretraining on high-quality, code/math and **long-context** data; extend RoPE to 32K–128K | 100B–1T tokens | a few % |
| **SFT** (supervised fine-tuning, "instruction tuning") | imitate good (prompt, response) pairs in the **chat template** | 10K–1M examples (InstructGPT SFT: ~13K prompts; Tülu 3: ~940K) | <1% |
| **Preference tuning** (RLHF with PPO, or DPO) | prefer "better" answers using human/AI preference pairs | 10K–1M comparisons | <1% |
| **RL with verifiable rewards** (GRPO etc.) | reason better on math/code by trial and error with checkable answers | prompts + verifiers; rollouts dominate cost | growing fast (DeepSeek-R1, o-series) |
| **Distillation** | a big "teacher" trains a smaller "student" on its outputs/probabilities | teacher generations | cheap vs pretraining |

## Real numbers to anchor on

| Model | Params | Tokens | Training FLOPs | Hardware |
|---|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | $3.1 \times 10^{23}$ | V100 cluster |
| Chinchilla (2022) | 70B | 1.4T | $5.8 \times 10^{23}$ | TPUs |
| Llama 3 8B (2024) | 8B | 15T | $\approx 7 \times 10^{23}$ | H100s |
| Llama 3.1 405B (2024) | 405B | 15.6T | $3.8 \times 10^{25}$ | up to 16K H100s |
| DeepSeek-V3 (2024) | 671B (37B active) | 14.8T | $\approx 3.3 \times 10^{24}$ | 2.79M H800-hours (≈ \$5.6M at \$2/GPU-hr, final run only) |

You'll derive the FLOPs column yourself in the next lesson with $C \approx 6ND$.

## Base vs instruct: see the difference SFT makes

```python
from mlx_lm import load, generate
for name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]:
    model, tok = load(name)
    msgs = [{"role": "user", "content": "Give me three tips for writing SQL."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(name, "\n", generate(model, tok, prompt=prompt, max_tokens=120), "\n" + "-" * 60)
```

The base model *continues text*: it may invent a new user turn, drift, or never stop. The instruct model *answers and emits `<|im_end|>`*. Same architecture, same size. The difference is a few hundred thousand SFT examples plus preference/RL tuning. **Fine-tuning changes behaviour and format cheaply; knowledge mostly comes from pretraining.** That's why you fine-tune for narrow tasks, and why fine-tuning facts in is unreliable (use retrieval for facts).

> [!NOTE] What "RLHF" means today
> InstructGPT (2022) did SFT → train a **reward model** on human comparisons → optimize the policy with **PPO** against it. **DPO** (2023) skips the reward model and the RL loop, optimizing directly on preference pairs. It's popular because it's just a loss function. Since DeepSeek-R1 (2025), **RL with verifiable rewards** (GRPO, checking math answers or running unit tests) drives reasoning. Its cost is dominated by **generating rollouts**, which is inference. That's Modules 21–23.

> [!REAL] Who does what
> Frontier labs do all stages. Most companies do **SFT (often LoRA) + maybe DPO** on an open base/instruct model. Some distill a big model into a small one for cost (Baseten ch. 1.3.3: DeepSeek-R1 was distilled into Qwen2.5 and Llama 3 models). Inference engineers are handed the output and asked to make it fast. Knowing how it was made tells you what can go wrong: a template mismatch, a tokenizer with new special tokens, or an untied LM head.

- [ ] Ran base vs instruct on 3 prompts; wrote one sentence per prompt on the behavioural difference
- [ ] Watched Karpathy's "Deep Dive into LLMs like ChatGPT" (pretraining, SFT and RL sections)
- [ ] For one open model you care about, found the pretraining token count and post-training recipe in its tech report
      */}),
      resources: [
        { title: "Karpathy — Deep Dive into LLMs like ChatGPT", url: "https://www.youtube.com/watch?v=7xTGNNLPyMI", type: "video", note: "3.5h; the full pipeline for a general audience, very clear" },
        { title: "Training language models to follow instructions (InstructGPT)", url: "https://arxiv.org/abs/2203.02155", type: "paper", note: "the SFT → RM → PPO recipe" },
        { title: "The Llama 3 Herd of Models", url: "https://arxiv.org/abs/2407.21783", type: "paper", note: "sections 3–4: pretraining at 3.8e25 FLOPs and the post-training recipe" },
      ],
    },
    {
      id: "flops-math",
      title: "Math: counting FLOPs: 2N per token to run, 6ND to train",
      kind: "math",
      minutes: 90,
      runsOn: ["browser"],
      md: MD(function () {/*
Two formulas cover most compute conversations in this field: **inference ≈ $2N$ FLOPs per token** and **training ≈ $6ND$ FLOPs**. You'll derive both from one fact about matrix multiplication.

> [!PREREQ] Counting work in a matrix multiply
> Multiplying an $(m \times k)$ matrix by a $(k \times n)$ matrix gives an $(m \times n)$ result. Each output cell is a dot product of length $k$: $k$ multiplies and $k$ adds. So the total is $m \cdot n \cdot k$ **multiply-adds** = $2mnk$ **FLOPs**.
> Example: $(4 \times 3)\,@\,(3 \times 2)$ → 8 outputs × 3 multiply-adds = 24 multiply-adds = **48 FLOPs**.

::viz matmul {"m":4,"k":3,"n":2}

## Inference: 2 FLOPs per parameter per token

A linear layer with weight $W$ of shape $(d_\text{out} \times d_\text{in})$ applied to **one token** is a $(1 \times d_\text{in})\,@\,(d_\text{in} \times d_\text{out})$ matmul: $2 \cdot d_\text{in} \cdot d_\text{out}$ FLOPs. That's **2 × (number of weights in the layer)**. Nearly all of a transformer's parameters are in linear layers, so:

$$
\text{forward FLOPs per token} \approx 2N
$$

Llama-3-8B: $2 \times 8 \times 10^9 = 16$ GFLOPs per token. A 1,000-token prompt (prefill) is 16 TFLOPs, about 16 ms of *ideal* H100 compute (989 TFLOPS). Attention adds a term that grows with context, $\approx 2 \cdot n_\text{layers} \cdot T_\text{ctx} \cdot d_\text{model}$ per token (4 × for QKᵀ and ·V), small until contexts get long (kipply's post derives it).

## Training: ×3 for the backward pass

Backpropagation (Module 01) needs, for each linear layer, **two** matmuls of the same size as the forward one: gradient w.r.t. the input (to keep propagating) and gradient w.r.t. the weights (to update them). Forward (2) + backward (4) = **6 FLOPs per parameter per token**:

$$
C_\text{train} \approx 6 \cdot N \cdot D \qquad (N = \text{parameters},\; D = \text{training tokens})
$$

**Worked example: pretraining Llama-3-8B.** $6 \times 8\text{e}9 \times 15\text{e}12 = 7.2 \times 10^{23}$ FLOPs. One H100 at 989 TFLOPS peak, achieving **40% MFU** (model FLOPs utilization, typical for large runs), delivers ~$4 \times 10^{14}$ FLOP/s:

$$
\frac{7.2 \times 10^{23}}{4 \times 10^{14}} = 1.8 \times 10^{9}\ \text{s} \approx 57\ \text{GPU-years} \;\Rightarrow\; \text{1,024 H100s} \approx 20\ \text{days}
$$

That's ~500,000 GPU-hours, or roughly \$1–1.5M at \$2–3 per GPU-hour, for the final run alone.

**Worked example: your LoRA run.** Qwen2.5-0.5B, 4,000 examples × ~150 tokens × 2 epochs ≈ 1.2M tokens. LoRA still runs the full forward and backward pass through the frozen model (it needs input gradients to reach the adapters), but skips most weight-gradient matmuls. So it's roughly $4N$ per token: $4 \times 0.5\text{e}9 \times 1.2\text{e}6 = 2.4 \times 10^{15}$ FLOPs. A Mac GPU sustaining a few TFLOPS in practice gets through that in **minutes**. That matches what you saw.

::viz scaling-laws

## The same numbers, as an inference engineer

| Quantity | Formula | Llama-3-8B |
|---|---|---|
| Prefill FLOPs for a $P$-token prompt | $2NP$ | $P$ = 2,000 → 32 TFLOPs → ~80 ms at 40% of an H100 |
| Decode FLOPs per token | $2N$ | 16 GFLOPs, but decode is **memory-bound** (Module 00): bytes, not FLOPs, set the speed |
| Arithmetic intensity at batch $b$ | ≈ $b$ FLOPs per weight byte (BF16) | batch 1 ≈ 1 FLOP/byte, far below the H100's ~295 → idle compute (Module 07) |
| Training a 1-epoch full fine-tune on 100M tokens | $6ND$ | $4.8 \times 10^{18}$ → ~3.3 H100-hours at 40% MFU |

- [ ] Computed by hand the FLOPs of `x @ W` for `x` (8, 4096) and `W` (4096, 14336); checked with the viz
- [ ] Reproduced the training-FLOPs column of the previous lesson's table with $6ND$ (DeepSeek-V3: use **active** params)
- [ ] Estimated GPU-days and dollars to pretrain a 1B model on 1T tokens on H100s at 40% MFU
- [ ] Measured your Mac's matmul TFLOPS (time `a @ b` for 4096² BF16/FP16 on MPS or MLX) and estimated your LoRA run's time before running it
      */}),
      resources: [
        { title: "EleutherAI — Transformer Math 101", url: "https://blog.eleuther.ai/transformer-math/", type: "article", note: "C ≈ 6ND, memory per parameter, all derived clearly" },
        { title: "kipply — Transformer Inference Arithmetic", url: "https://kipply.github.io/blog/transformer-inference-arithmetic/", type: "article", note: "2N per token, the attention term, and the KV cache" },
      ],
    },
    {
      id: "scaling-laws",
      title: "Scaling laws and Chinchilla: how big, how many tokens, and why labs over-train",
      kind: "concept",
      minutes: 75,
      runsOn: ["browser"],
      md: MD(function () {/*
If you have a compute budget $C$, how should you split it between **model size** $N$ and **data** $D$ (with $C \approx 6ND$)? Scaling laws answer this empirically. The answer has shifted in a way that matters directly to inference.

## Kaplan et al. (OpenAI, 2020)

Loss falls as a smooth **power law** in $N$, $D$ and $C$ across 7+ orders of magnitude, so a straight line on log-log axes. Architecture details (depth vs width) matter far less than scale. Their fitted trade-off said: *grow the model faster than the data*. GPT-3 (175B params, 300B tokens: under 2 tokens per parameter) followed that advice.

## Chinchilla (Hoffmann et al., DeepMind, 2022)

With better-tuned learning-rate schedules, they found **parameters and tokens should scale equally**: $N_\text{opt} \propto C^{0.5}$, $D_\text{opt} \propto C^{0.5}$, which works out to about **20 tokens per parameter**. They fit:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad E = 1.69,\; A = 406.4,\; B = 410.7,\; \alpha = 0.34,\; \beta = 0.28
$$

Read it as: irreducible loss $E$ (the entropy of text) + a penalty for too-small a model + a penalty for too little data. The proof: **Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) with the same compute**. It was 4× smaller, so it was also 4× cheaper to serve.

> [!MATH] Worked example: compute-optimal for $C = 10^{21}$ FLOPs
> With $D = 20N$: $C = 6N \cdot 20N = 120N^2$, so $N = \sqrt{C/120} = \sqrt{8.3 \times 10^{18}} \approx 2.9 \times 10^{9}$ parameters and $D \approx 58 \times 10^{9}$ tokens. Check: $6 \times 2.9\text{e}9 \times 58\text{e}9 \approx 1.0 \times 10^{21}$ ✓.

::viz scaling-laws

## Why modern models are "over-trained", an inference argument

Chinchilla minimizes **training** compute for a given loss. But a deployed model's lifetime cost is training **plus inference**, and inference cost ∝ $N$ per token, forever. If you'll serve trillions of tokens, it pays to train a **smaller model on far more data** than Chinchilla-optimal:

| Model | Params | Tokens | Tokens / param | vs Chinchilla (20) |
|---|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20 | 1× |
| Llama 2 7B | 7B | 2T | 286 | 14× |
| Llama 3 8B | 8B | 15T | 1,875 | 94× |
| Qwen2.5-0.5B | 0.5B | 18T | 36,000 | 1,800× |

The loss keeps improving (slowly) past 20 tokens/param. You pay more at training time to get a model that's cheaper per token at serving time. Sardana et al. (2023) formalize this "inference-aware" scaling. **This is the single biggest reason small open models got so good.**

> [!INTUITION] What scaling laws mean for your job
> 1. **Model choice is the first optimization** (Baseten ch. 1.3). A heavily over-trained 8B can replace a 70B for many tasks at ~1/9 of the serving cost.
> 2. Power laws mean **diminishing returns**: 10× the compute buys a fixed *decrement* in loss. That's why labs also scale RL and **test-time compute** (reasoning tokens), which moves the cost into inference, your domain.
> 3. For fine-tuning, the same intuition applies in miniature. Past a point, more examples of the same kind stop helping, and **data quality and diversity** beat quantity.

- [ ] Used the viz to find the Chinchilla-optimal $N, D$ for $10^{22}$, $10^{23}$ and $10^{24}$ FLOPs
- [ ] Plugged Llama 3 8B and a Chinchilla-optimal model of the same compute into $L(N, D)$; compared predicted losses and serving cost per token
- [ ] Read the Chinchilla abstract and Figure 1; read Kaplan's Figure 1
- [ ] Wrote a 5-sentence argument for why an inference-heavy company should prefer an over-trained small model
      */}),
      resources: [
        { title: "Scaling Laws for Neural Language Models (Kaplan et al., 2020)", url: "https://arxiv.org/abs/2001.08361", type: "paper", note: "the original power laws" },
        { title: "Training Compute-Optimal LLMs (Hoffmann et al., 2022, Chinchilla)", url: "https://arxiv.org/abs/2203.15556", type: "paper", note: "~20 tokens per parameter; the L(N, D) fit" },
        { title: "Beyond Chinchilla-Optimal: Accounting for Inference (Sardana et al.)", url: "https://arxiv.org/abs/2401.00448", type: "paper", note: "why you over-train when inference demand is high" },
      ],
    },
    {
      id: "training-memory",
      title: "Memory for training: weights, grads, Adam states, activations (and optimizers SGD → AdamW)",
      kind: "concept",
      minutes: 90,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Inference needs the weights plus the KV cache. **Training needs far more**, and that's why a 7B model you can *run* on a 16 GB GPU needs ~120 GB to fully *fine-tune*. Here's where the bytes go.

## First: what the optimizer stores (brief intuition)

- **SGD**: $w \leftarrow w - \eta\, g$. Step downhill along the gradient $g$ with learning rate $\eta$. It stores nothing extra. It's noisy and sensitive to $\eta$.
- **Momentum**: keep a running average of gradients, $m \leftarrow \beta m + g$, and step along $m$. It's a heavy ball that rolls through noise. It stores **1 extra number per parameter**.
- **Adam**: keep running averages of $g$ (**m**, the direction) *and* $g^2$ (**v**, how big gradients usually are), and step $\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$. Each parameter gets its **own adaptive step size**: parameters with consistently large gradients take smaller steps. It stores **2 extra numbers per parameter**, usually in FP32.
- **AdamW**: Adam with **decoupled weight decay** ($w \leftarrow w - \eta \lambda w$ applied separately). It's the default for every LLM. Typical settings: $\beta = (0.9, 0.95)$, weight decay 0.1, warmup then cosine decay. Learning rate ~3e-4 for small pretraining, ~1e-5 for full fine-tunes, and ~1e-4 to 2e-4 for LoRA.

::viz gradient-descent

## Bytes per parameter

| Item | Full fine-tune, mixed precision (BF16 + FP32 master) | LoRA (BF16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| Weights | 2 (BF16) + 4 (FP32 master copy) | 2 (frozen) | ~0.5 (NF4, frozen) |
| Gradients | 2 | ≈ 0 (only adapters) | ≈ 0 |
| Adam m, v | 4 + 4 | ≈ 0 (only adapters) | ≈ 0 |
| **Total / param** | **≈ 16 bytes** | **≈ 2 bytes** | **≈ 0.5–0.6 bytes** |
| 7B model | ~112 GB + activations | ~14 GB + activations | ~4–5 GB + activations |
| Qwen2.5-0.5B | ~8 GB + activations | ~1 GB + activations | ~0.3 GB + activations |

(Adapters are ~1% of params, so their own weights + grads + Adam states add only a little.)

**Activations** are the other big term. Backprop needs the intermediate tensors of every layer from the forward pass. They scale with **batch × sequence length × hidden × layers**. For long sequences they can exceed the weights. Two standard fixes:
- **Gradient checkpointing** (activation recomputation): keep only each block's input and recompute the rest during backward. It costs ~30% more compute and cuts activation memory massively. Use `--grad-checkpoint` in mlx-lm and `gradient_checkpointing=True` in TRL.
- **Smaller micro-batches + gradient accumulation**: same effective batch size, less memory, more steps.

::viz training-memory

## Mixed precision in one paragraph

Matmuls run in **BF16** (2 bytes, same exponent range as FP32, so no overflow issues and no loss scaling needed). A **FP32 master copy** of the weights receives the updates, because adding a tiny update like 1e-6 to a BF16 weight of 1.0 rounds away to nothing. BF16 needs Ampere or newer (A100/H100) or Apple Silicon. The Colab **T4 has no BF16**, so use **FP16 + loss scaling** (`fp16=True`) there. Module 14 covers number formats in depth.

> [!REAL] At scale
> Full fine-tuning a 70B model needs ~1.1 TB for weights/grads/optimizer. That means sharding across GPUs with **ZeRO / FSDP**: each GPU holds 1/N of the optimizer states, gradients and even the weights (Module 16). This is why LoRA/QLoRA took over fine-tuning outside big labs. The QLoRA paper fine-tuned a 65B model on a **single 48 GB GPU**.

- [ ] Filled the bytes-per-param table for Llama-3-8B and Qwen2.5-7B for all three methods; checked against the viz
- [ ] Measured peak memory (`mlx_lm.lora` prints `Peak mem`) for batch 1, 4 and 8, with and without `--grad-checkpoint`; explained the trend
- [ ] Implemented one AdamW step by hand for a single parameter in Python, and compared it with `torch.optim.AdamW`
      */}),
      resources: [
        { title: "Decoupled Weight Decay Regularization (AdamW)", url: "https://arxiv.org/abs/1711.05101", type: "paper", note: "why decay is applied separately from Adam's update" },
        { title: "Hugging Face — The Ultra-Scale Playbook", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook", type: "article", note: "the 'memory usage in transformers' section is the best visual treatment" },
      ],
    },
    {
      id: "lora-math",
      title: "Math: LoRA and QLoRA: why ΔW = BA works",
      kind: "math",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
LoRA (Hu et al., 2021) freezes the pretrained weight $W$ and learns a **low-rank** update:

$$
h = Wx + \Delta W x = Wx + \frac{\alpha}{r}\, B A\, x, \qquad W \in \mathbb{R}^{d \times k},\; B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
$$

> [!PREREQ] What "rank" means
> An **outer product** of a column $u$ ($d \times 1$) and a row $v^\top$ ($1 \times k$) is a full $d \times k$ matrix, but every row is a multiple of $v^\top$. It holds only $d + k$ numbers of information, not $d \cdot k$. That's a **rank-1** matrix. A **rank-$r$** matrix is a sum of $r$ such outer products, which is exactly $BA$ (the columns of $B$ times the rows of $A$).
> Example: $u = (1, 2, 3, 4)^\top$, $v = (1, 0, -1, 2)$: $uv^\top$ is $4 \times 4$ = 16 numbers built from 8. Row 3 is just $3 \times$ row 1.

## Why a low-rank update is enough

Fine-tuning a pretrained model doesn't need to change *everything*. It nudges existing features toward a task. Empirically, the weight change $\Delta W$ from full fine-tuning has low **intrinsic rank** (Aghajanyan et al., 2020): most of its "energy" lives in a few directions. LoRA bets on this. For narrow tasks, $r$ = 8–16 typically matches full fine-tuning. For big distribution shifts (new language, lots of new knowledge), full fine-tuning or higher ranks win.

**Parameter count** for one $4096 \times 4096$ projection:

| | Trainable params | Fraction |
|---|---|---|
| Full | $4096^2 = 16{,}777{,}216$ | 100% |
| LoRA $r = 16$ | $16 \times (4096 + 4096) = 131{,}072$ | 0.78% |
| LoRA $r = 8$ | 65,536 | 0.39% |

For Qwen2.5-0.5B with $r = 8$ on all seven linear layers of all 24 blocks: per block $8 \times [(896{+}896) \cdot 2 + (896{+}128) \cdot 2 + (896{+}4864) \cdot 3] = 183{,}296$, so **4.4M total ≈ 0.9%** of the model.

::viz lora

## The details that make it work

- **Initialization**: $A$ is random, $B = 0$, so $\Delta W = BA = 0$ at step 0. The model starts *exactly* as the pretrained model, and training moves it gently.
- **Scaling** $\alpha / r$: it keeps the update's size roughly stable when you change $r$. PEFT's `lora_alpha` is $\alpha$. mlx-lm's `scale` is the whole multiplier (default 20.0).
- **Which layers**: the original paper did only $W_q, W_v$. Today "all linear layers" (`target_modules="all-linear"`) is the common default and usually better.
- **Zero inference overhead after merging**: $W' = W + \tfrac{\alpha}{r}BA$ is the same shape as $W$. Merge once and serve a normal model. **Unmerged**, each adapted layer adds two skinny matmuls, which is what enables **multi-LoRA serving** (one base, many adapters, chosen per request).

## Build it: LoRA on your own Module 04 model

```python
# lora.py — wrap nn.Linear layers of your qwen2.py with LoRA
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)                       # frozen W (and bias)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r)  # (r, k)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # (d, r): zero, so ΔW = 0 at start
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
    @torch.no_grad()
    def merged(self):
        self.base.weight += (self.B @ self.A) * self.scale
        return self.base

def add_lora(model, r=8, alpha=16, targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.model.layers:
        for parent in (block.self_attn, block.mlp):
            for name in targets:
                if hasattr(parent, name):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return model

from qwen2 import load
model = add_lora(load("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, device="mps"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable {trainable:,} / {sum(p.numel() for p in model.parameters()):,}")   # ~4.4M / ~498M
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
# training step: logits = model(x); loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()); ...
```

## QLoRA: LoRA on a 4-bit base

QLoRA (Dettmers et al., 2023) stores the frozen $W$ in **4-bit NF4** (a 4-bit format whose levels match normally distributed weights), dequantizes on the fly for each matmul, and trains BF16 adapters on top. It also adds double quantization of the scales and paged optimizers. Memory per base param drops from 2 bytes to ~0.5. In mlx-lm, **pointing `--model` at a quantized model automatically gives you QLoRA**.

> [!WARNING] Serving a QLoRA adapter
> The adapter was trained against the **quantized** base. Merging it into the BF16 base (or re-quantizing after merging) shifts the weights slightly. Usually that's fine, but **always re-run your eval after every merge or quantize step**. That's part of this module's challenge.

- [ ] Built the rank-1 example matrix in numpy and checked `np.linalg.matrix_rank`
- [ ] Verified the 4.4M trainable-parameter count for Qwen2.5-0.5B with `lora.py`
- [ ] Confirmed that at step 0 the LoRA model's logits equal the base model's exactly (because $B = 0$)
- [ ] Trained your `LoRALinear` model for 50 steps on MPS on 100 SQL examples and watched the loss fall; merged it and checked the logits are unchanged by merging
      */}),
      resources: [
        { title: "LoRA: Low-Rank Adaptation of Large Language Models", url: "https://arxiv.org/abs/2106.09685", type: "paper", note: "sections 4 and 7: method and the low-rank analysis of ΔW" },
        { title: "QLoRA: Efficient Finetuning of Quantized LLMs", url: "https://arxiv.org/abs/2305.14314", type: "paper", note: "NF4, double quantization, 65B on one 48 GB GPU" },
        { title: "Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA", url: "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms", type: "article", note: "rank/alpha/layers experiments, with results" },
      ],
    },
    {
      id: "build-lora",
      title: "Build lab: text-to-SQL LoRA on Mac with mlx-lm (or Colab with TRL + PEFT), then fuse, quantize and serve",
      kind: "build",
      minutes: 240,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Now do it properly: **your data, a frozen test set, a baseline, a training run, an eval, a merge, a quantization and a server.** The task is text-to-SQL, using `b-mc2/sql-create-context` (78K question + `CREATE TABLE` + SQL triples).

> [!BUILD] Steps
> 1. Build `train/valid/test.jsonl` in chat format → 2. baseline eval → 3. LoRA train → 4. eval with adapter → 5. fuse + quantize + re-eval → 6. serve with the adapter over an OpenAI-compatible API.

## 1. Data (freeze the test set *first*)

```python
# prep.py
import json, os
from datasets import load_dataset

ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=0)
SYSTEM = "You translate questions into SQLite. Reply with one SQL query and nothing else."

def row(ex):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{ex['context']}\n\nQuestion: {ex['question']}"},
        {"role": "assistant", "content": ex["answer"]},
    ]}

os.makedirs("data", exist_ok=True)
splits = {"test": range(0, 200), "valid": range(200, 400), "train": range(400, 4400)}
for name, idx in splits.items():
    with open(f"data/{name}.jsonl", "w") as f:
        for ex in ds.select(idx):
            f.write(json.dumps(row(ex)) + "\n")
```

## 2. The eval script (used before *and* after)

```python
# eval_sql.py — exact match + "does it even execute" on an in-memory SQLite DB with the schema
import argparse, json, re, sqlite3, time
from mlx_lm import load, generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", default=None)
args = ap.parse_args()
model, tok = load(args.model, adapter_path=args.adapter)

def norm(s):
    s = s.strip().strip("`").removeprefix("sql").strip().rstrip(";")
    return re.sub(r"\s+", " ", s).replace('"', "'").lower()

def executes(sql, schema):
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(schema); db.execute(sql); return True
    except Exception:
        return False

rows = [json.loads(line) for line in open("data/test.jsonl")]
em = ok = ntok = 0
t0 = time.perf_counter()
for r in rows:
    msgs, gold = r["messages"][:-1], r["messages"][-1]["content"]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pred = generate(model, tok, prompt=prompt, max_tokens=128)
    ntok += len(tok.encode(pred))
    schema = msgs[1]["content"].split("Schema:\n")[1].split("\n\nQuestion:")[0]
    em += norm(pred) == norm(gold)
    ok += executes(norm(pred), schema)
dt = time.perf_counter() - t0
print(f"exact-match {em}/{len(rows)} ({100*em/len(rows):.1f}%)   valid SQL {ok}/{len(rows)} ({100*ok/len(rows):.1f}%)   {ntok/dt:.1f} tok/s")
```

Run the baseline: `python eval_sql.py | tee baseline.txt`. Read 20 failures by eye. **Looking at your data** is the most underrated eval skill (Baseten ch. 1.3.1).

## 3. Train

```bash
mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --train --data data \
  --iters 1000 --batch-size 8 --num-layers 16 --learning-rate 1e-4 \
  --mask-prompt --steps-per-eval 100 --val-batches 25 --save-every 200 \
  --adapter-path adapters/sql-r8
```

`--mask-prompt` computes the loss **only on the assistant's SQL**, not on the schema/question. You want the model to learn to *write* SQL, not to predict schemas. To change rank or target layers, pass a YAML with `-c config.yaml`:

```yaml
# config.yaml (CLI flags override these)
lora_parameters:
  rank: 16
  scale: 20.0
  dropout: 0.05
```

Watch train and val loss. If val loss rises while train keeps falling, stop earlier (fewer iters) or add data.

## 4–5. Evaluate, fuse, quantize, re-evaluate

```bash
python eval_sql.py --adapter adapters/sql-r8 | tee lora.txt
mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters/sql-r8 --data data --test   # test loss / perplexity

mlx_lm.fuse --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters/sql-r8 --save-path qwen-sql-fused
python eval_sql.py --model qwen-sql-fused | tee fused.txt          # should match lora.txt closely

mlx_lm.convert --hf-path qwen-sql-fused --mlx-path qwen-sql-4bit -q --q-bits 4
python eval_sql.py --model qwen-sql-4bit | tee q4.txt              # small quality drop, big speed-up
```

## 6. Serve the adapter

```bash
mlx_lm.server --model Qwen/Qwen2.5-0.5B-Instruct --adapter-path adapters/sql-r8 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role":"system","content":"You translate questions into SQLite. Reply with one SQL query and nothing else."},
               {"role":"user","content":"Schema:\nCREATE TABLE users (id INTEGER, name VARCHAR, age INTEGER)\n\nQuestion: How many users are older than 30?"}],
  "max_tokens": 64, "temperature": 0}'
```

## Path B: Colab (T4) with TRL + PEFT

The same data works unchanged. TRL's `SFTTrainer` applies the chat template, and with the **prompt/completion** format it computes loss on the completion only (like `--mask-prompt`).

```python
# pip install -q trl peft datasets   (TRL >= 0.20; tested API: SFTConfig/SFTTrainer)
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/valid.jsonl"})
ds = ds.map(lambda ex: {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}, remove_columns=["messages"])

args = SFTConfig(
    output_dir="qwen-sql-lora", num_train_epochs=2, per_device_train_batch_size=8,
    gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=10, eval_strategy="steps", eval_steps=100, max_length=512,
    fp16=True,                       # T4 has no BF16; use bf16=True on A100/H100
    gradient_checkpointing=True, report_to="none",
    model_init_kwargs={"dtype": torch.float32},
)
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", args=args, peft_config=peft_config,
                     train_dataset=ds["train"], eval_dataset=ds["validation"])
trainer.train()
trainer.save_model("qwen-sql-lora")          # adapter_config.json + adapter_model.safetensors
```

Merge for deployment: `AutoPeftModelForCausalLM.from_pretrained("qwen-sql-lora").merge_and_unload().save_pretrained("qwen-sql-merged")`. Or keep it unmerged and serve with vLLM's multi-LoRA support (next lessons).

> [!WARNING] The #1 fine-tuning bug: train/serve template mismatch
> Train with the chat template and the system prompt, then evaluate/serve with **the same** template and system prompt. If you trained on raw `text` (like the WikiSQL demo), use `--ignore-chat-template` at inference. Most "my fine-tune got worse" reports are this.

- [ ] `prep.py` wrote 4,000 / 200 / 200 examples; the test set was never looked at during tuning
- [ ] Baseline, LoRA, fused and 4-bit results saved as text files with the same eval script
- [ ] Tried two ranks (8 vs 16) **or** two learning rates; recorded val loss and exact-match for each
- [ ] Served the adapter with `mlx_lm.server` and hit it with `curl` (or the `openai` Python client)
- [ ] (Colab path) Trained with TRL + PEFT on a T4 and compared exact-match with the MLX run
      */}),
      resources: [
        { title: "mlx-lm — LORA.md", url: "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md", type: "docs", note: "data formats, --mask-prompt, fuse, memory tips" },
        { title: "TRL — SFT Trainer", url: "https://huggingface.co/docs/trl/sft_trainer", type: "docs", note: "dataset formats, completion-only loss, PEFT integration" },
        { title: "PEFT docs", url: "https://huggingface.co/docs/peft/index", type: "docs", note: "LoraConfig, merge_and_unload, adapters" },
        { title: "b-mc2/sql-create-context", url: "https://huggingface.co/datasets/b-mc2/sql-create-context", type: "docs", note: "78K text-to-SQL examples with CREATE TABLE context" },
      ],
    },
    {
      id: "evaluate",
      title: "Lab: evaluation basics: held-out loss, perplexity, task evals, lm-evaluation-harness",
      kind: "lab",
      minutes: 120,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
"It looks better" isn't a result. Every fine-tune, merge, quantization and engine change you make for the rest of this course needs a **number before and a number after**. Baseten ch. 1.3.1: evals give you the *quality baseline* that every lossy optimization is measured against.

## Level 1: held-out loss and perplexity

Loss is the average cross-entropy $-\ln p(\text{correct token})$ on data the model didn't train on. **Perplexity** is $e^{\text{loss}}$: "the model is as uncertain as if it were choosing uniformly among PPL tokens".

> [!PREREQ] Quick log refresher
> $\ln$ is the inverse of $e^x$. $-\ln(1) = 0$ (certain and right), $-\ln(0.5) = 0.69$, $-\ln(0.01) = 4.6$. A random guess over a 151,936-token vocabulary has loss $\ln(151936) = 11.9$, perplexity 151,936.

::viz neg-log {"vocab": 151936}

Worked example: a val loss of 0.183 means PPL = $e^{0.183} = 1.20$. On SQL completions the model is nearly certain of each token. That's plausible because SQL is very formulaic. Loss is great for **comparing runs on the same data and tokenizer**. It says nothing about whether the query is *correct*. And it's not comparable across tokenizers.

::viz overfitting

## Level 2: task evals (what you actually care about)

Your `eval_sql.py` measures **exact match** (strict, underestimates) and **executes** (lenient). Better still is **execution accuracy**: run gold and predicted SQL on a database *with rows* and compare results. Rules for a task eval set:
- **100–500 examples, frozen before training**, drawn from the real distribution (including the hard cases), with no overlap with training data (check for near-duplicates).
- **Deterministic decoding** (temperature 0) for comparisons. Report sampling variance separately if you serve with sampling.
- **Look at failures.** Categorize 30 of them (wrong column, wrong aggregation, hallucinated table, extra prose). The categories tell you what data to add.
- For open-ended outputs, use **LLM-as-judge** with a rubric, and validate the judge against ~50 human labels first.

## Level 3: regression on general benchmarks (did you break anything?)

Fine-tuning can cause **catastrophic forgetting** (Module 24). Check that general ability didn't collapse with EleutherAI's **lm-evaluation-harness**, the standard tool behind the Open LLM Leaderboard:

```bash
uv pip install "lm_eval[hf]" peft
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float16 \
  --tasks arc_easy,hellaswag --device mps --batch_size 8 --limit 500
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,peft=qwen-sql-lora,dtype=float16 \
  --tasks arc_easy,hellaswag --device mps --batch_size 8 --limit 500
# MLX alternative (wraps lm-eval):  mlx_lm.evaluate --model qwen-sql-fused --tasks arc_easy hellaswag
```

`--limit 500` keeps it to minutes (and adds noise: ±2% on 500 examples). A LoRA fine-tune on 4K SQL examples should barely move these scores. If ARC drops 10 points, your learning rate or iteration count was too high.

| Level | Metric | Cost | Tells you |
|---|---|---|---|
| 1 | val loss / perplexity | seconds | training is healthy; compare runs |
| 2 | task accuracy on your frozen set | minutes | **is it better at the job?** |
| 3 | general benchmarks | minutes to hours | did you break something else? |
| 4 | online A/B, user feedback | days | does it matter to users? |

> [!REAL] Evals in inference work
> Every optimization in Modules 14–15 (quantization, speculative decoding, KV-cache quantization) and every engine upgrade is gated by an eval run like this. A 4-bit model that is 3× faster but drops task accuracy from 65% to 40% is a regression, not an optimization. Teams keep a **small, fast, frozen eval suite** in CI and run it on every deployment candidate (Baseten ch. 7.4).

- [ ] Computed perplexity from your val loss; explained why it's not comparable across different tokenizers
- [ ] Categorized 30 failures of the base model and 30 of the fine-tune
- [ ] Ran lm-eval (`arc_easy`, `hellaswag`, `--limit 500`) on base vs adapter; reported the deltas with ± noise
- [ ] Added a stricter execution-accuracy metric: create 3 rows per table with sqlite and compare result sets
      */}),
      resources: [
        { title: "EleutherAI lm-evaluation-harness", url: "https://github.com/EleutherAI/lm-evaluation-harness", type: "tool", note: "the standard open benchmark runner; supports HF + PEFT adapters" },
        { title: "Inference Engineering (Baseten) — Ch. 1.3", url: "Inference%20Engineering.pdf", type: "book", note: "evals, fine-tuning and distillation for model selection" },
      ],
    },
    {
      id: "distill-serve",
      title: "Distillation, synthetic data, and why inference engineers care about fine-tuning",
      kind: "concept",
      minutes: 75,
      runsOn: ["any"],
      md: MD(function () {/*
Two more ideas complete the picture: how small models inherit big-model skills, and what fine-tuned artifacts mean for the serving stack.

## Distillation

A large **teacher** trains a small **student** (Hinton et al., 2015).
- **Logit (soft-label) distillation**: train the student to match the teacher's full probability distribution over the next token (a KL-divergence loss, often at temperature $T > 1$ to expose "dark knowledge": the teacher thinks *cat* is more likely than *car*). This needs the teacher's logits, so the vocabularies must match (same model family).
- **Sequence-level distillation** (the common one): generate lots of outputs from the teacher, then **SFT** the student on them. DeepSeek-R1's reasoning traces were distilled into Qwen2.5 and Llama 3 students this way (Baseten ch. 1.3.3). The student copies the teacher's good *and* bad habits.

$$
\mathcal{L}_\text{KD} = T^2 \cdot \mathrm{KL}\!\left(\mathrm{softmax}(z_\text{teacher}/T)\;\Vert\;\mathrm{softmax}(z_\text{student}/T)\right)
$$

## Synthetic data: the modern fine-tuning pipeline

For your SQL task, a production-grade dataset is usually **generated**, not hand-written:
1. **Seed**: 50–200 real, hand-checked examples (your real schemas and question styles).
2. **Generate**: a strong teacher (a big open model on vLLM, or an API) writes thousands of new (schema, question, SQL) triples. This is an **offline batch-inference job**, and throughput per dollar is everything (that's you).
3. **Filter with a verifier**: keep only SQL that parses and executes, dedupe, drop near-duplicates of the test set.
4. **Train, eval, look at failures, generate more of what fails.** Repeat.

This is also the skeleton of RL with verifiable rewards (Module 22): generate → verify → learn.

```python
# offline teacher generation with vLLM (on a GPU box)
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, n=4)   # 4 candidates per prompt
outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
```

## Why inference engineers care

| Training artifact | What it means for serving |
|---|---|
| **LoRA adapters** (MBs each) | **Multi-LoRA serving**: one base model in GPU memory, many adapters swapped **per request** in the same batch (S-LoRA/Punica kernels). `vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules sql=./qwen-sql-lora --max-lora-rank 16`, then send `"model": "sql"`. Adapters can be cached in CPU memory and loaded on demand. Routers send requests to replicas that already have the adapter loaded (**LoRA-aware routing**, Baseten ch. 7.2) |
| **Merged fine-tunes** | a normal checkpoint: same serving path as the base, but check template, EOS and dtype |
| **Unmerged adapter cost** | extra skinny matmuls per layer: a few % slower. Merge if you only serve one adapter |
| **QLoRA / quantized bases** | serve on the same quantization it was trained on, or re-eval after merging |
| **QAT models** (GPT-OSS in MXFP4, Kimi K2 Thinking in INT4) | trained to be quantized: run them in their native low precision (Baseten ch. 5.1.2) |
| **Distilled students / drafters** | cheaper models to serve; **fine-tuned speculative-decoding drafters** raise acceptance rates (Baseten ch. 5.2, Module 15) |
| **RL post-training** | the training loop *contains* an inference engine (vLLM/SGLang for rollouts) plus weight sync (Modules 22–23) |

> [!INTUITION] The through-line
> Training decides **what** the model computes. Inference engineering decides **how cheaply and quickly** it's computed. The boundary keeps blurring: QAT, drafters, distillation and RL rollouts all need people who understand both sides. You now do.

- [ ] Wrote a 30-line script that generates 500 synthetic SQL examples from 20 seeds with a teacher (an API, or Qwen2.5-7B via mlx-lm), filters them by execution, and dedupes against your test set
- [ ] Retrained with real + synthetic data; compared exact-match
- [ ] (Colab/GPU) Served two adapters from one vLLM base with `--enable-lora` and sent requests to each
- [ ] Explained to yourself why multi-LoRA batching needs special kernels (different adapters for different rows of one batch)
      */}),
      resources: [
        { title: "Distilling the Knowledge in a Neural Network (Hinton et al.)", url: "https://arxiv.org/abs/1503.02531", type: "paper", note: "soft targets and temperature" },
        { title: "vLLM — LoRA adapters", url: "https://docs.vllm.ai/en/latest/features/lora.html", type: "docs", note: "--enable-lora, --lora-modules, dynamic loading" },
        { title: "DeepSeek-R1", url: "https://arxiv.org/abs/2501.12948", type: "paper", note: "section on distilling reasoning into Qwen/Llama students" },
      ],
    },
  ],

  challenge: {
    title: "Ship a narrow-task fine-tune with a before/after report",
    md: MD(function () {/*
Pick **one narrow task**: text-to-SQL (you have the pipeline), **JSON extraction** (e.g. pull `{name, date, amount, currency}` out of messy invoice/email text, graded by field-level exact match and valid JSON), or one from your own work. Deliver in `course-work/m05/`:

1. **Data**: ≥1,000 train examples (real, synthetic-and-filtered, or mixed), 100–200 valid, and a **frozen** 100–200 example test set written *before* training. Document how you built it.
2. **Baseline**: base model on your eval (deterministic decoding).
3. **Fine-tune**: LoRA with mlx-lm (Mac) or TRL + PEFT (Colab). At least two configurations (rank or LR or iterations). Include loss curves (train + val).
4. **Merge + quantize**: fuse the adapter, quantize to 4-bit (MLX `-q` or GGUF `Q4_K_M`), and re-run the eval after **each** step.
5. **Speed**: report decode tok/s and peak memory for base BF16, fused BF16 and fused 4-bit (`mlx_lm.generate` prints both; or your Module 00 `bench.py` against `mlx_lm.server`/Ollama).
6. **`REPORT.md`**: one table (rows: base, LoRA, fused, fused-4bit; columns: task metric, secondary metric, tok/s, peak GB, size on disk), the lm-eval regression check, 5 categorized failure examples, and a paragraph on what you'd do next.
    */}),
    checklist: [
      "Test set frozen before training; no train/test overlap (checked)",
      "Task metric improves clearly over baseline (and you can explain why, with failure analysis)",
      "Eval re-run after merge and after quantization; any quality drop quantified",
      "tok/s and peak memory reported for BF16 vs 4-bit, with a one-line napkin explanation (bytes ÷ bandwidth)",
      "lm-eval regression check on ≥2 general tasks shows no large drop (or explains it)",
      "Adapter served over an OpenAI-compatible endpoint and called from a client script",
    ],
    stretch: "Train the **same** adapter as QLoRA on the 4-bit base (`--model mlx-community/Qwen2.5-0.5B-Instruct-4bit`) and compare it with LoRA-then-quantize. Then scale up: Qwen2.5-1.5B or 3B. Is a fine-tuned 0.5B better than a base 3B on your task? What's the serving-cost ratio? That trade-off is the whole business case for fine-tuning.",
  },

  connects: MD(function () {/*
Phase 1 is done: you can **build** (M03), **load** (M04) and **adapt** (M05) a modern LLM, and you can estimate compute ($2N$, $6ND$) and memory (bytes/param, KV/token) on a napkin.

Phase 2 starts from the most obvious inefficiency you've seen twice now: your `generate()` recomputes the whole sequence every step. Module 06 splits inference into **prefill and decode**, adds a **KV cache**, and measures TTFT/TPOT properly. Later, fine-tuning comes back: **multi-LoRA** serving (M09–M10, M18), **quantization** with eval gates (M14), fine-tuned **drafters** (M15), and **RL post-training**, where the training loop you learned here wraps an inference engine (M21–M23). Forgetting, which you checked for with lm-eval, becomes a whole topic in M24.
  */}),

  interview: [
    "Derive why training costs about 6ND FLOPs and inference about 2N FLOPs per token. What does the approximation ignore?",
    "Estimate the GPU-days to pretrain an 8B model on 15T tokens on H100s at 40% MFU. How would you check whether a claimed training cost is plausible?",
    "What does Chinchilla say about the optimal tokens per parameter, and why do Llama 3 and Qwen train far past it?",
    "How much GPU memory does a full fine-tune of a 7B model with AdamW in mixed precision need? How do LoRA and QLoRA change that?",
    "Explain LoRA: what are A and B, why is B initialized to zero, and what does α/r do? Is there inference overhead?",
    "How would you serve 500 different customer LoRA adapters on the same base model? What are the bottlenecks?",
    "Your fine-tuned model scores well on validation loss but users say it's worse. What could be going on?",
    "What's the difference between logit distillation and sequence-level distillation? When can't you use the former?",
  ],

  resources: [
    { title: "mlx-lm LoRA guide", url: "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md", type: "docs", note: "the Mac fine-tuning path" },
    { title: "TRL — SFT Trainer", url: "https://huggingface.co/docs/trl/sft_trainer", type: "docs", note: "the GPU/Colab fine-tuning path" },
    { title: "LoRA paper", url: "https://arxiv.org/abs/2106.09685", type: "paper", note: "low-rank adaptation" },
    { title: "QLoRA paper", url: "https://arxiv.org/abs/2305.14314", type: "paper", note: "4-bit base + adapters" },
    { title: "Chinchilla (Hoffmann et al., 2022)", url: "https://arxiv.org/abs/2203.15556", type: "paper", note: "compute-optimal scaling" },
    { title: "Kaplan et al. — Scaling Laws (2020)", url: "https://arxiv.org/abs/2001.08361", type: "paper", note: "the original power laws" },
    { title: "EleutherAI — Transformer Math 101", url: "https://blog.eleuther.ai/transformer-math/", type: "article", note: "compute and memory formulas in one place" },
    { title: "Karpathy — Deep Dive into LLMs like ChatGPT", url: "https://www.youtube.com/watch?v=7xTGNNLPyMI", type: "video", note: "the full training pipeline, intuitively" },
    { title: "HF Ultra-Scale Playbook", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook", type: "article", note: "training memory and parallelism, visually" },
    { title: "lm-evaluation-harness", url: "https://github.com/EleutherAI/lm-evaluation-harness", type: "tool", note: "standard benchmark runner" },
    { title: "Sebastian Raschka — Practical Tips for LoRA", url: "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms", type: "article", note: "empirical LoRA hyperparameter lessons" },
  ],
});
