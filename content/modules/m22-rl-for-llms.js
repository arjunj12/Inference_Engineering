Course.module({
  id: "m22-rl-for-llms",
  title: "RL for LLMs: RLHF, DPO, GRPO & reasoning",
  short: "RL for LLMs (GRPO)",
  tagline: "GRPO-train a small Qwen model on grade-school math with a verifiable reward, watch its accuracy and answer length change, and understand the RLHF → DPO → GRPO lineage well enough to debug it.",
  hours: 20,
  level: "core",
  runsOn: ["colab", "cloud", "mac"],
  tags: ["rlhf", "grpo", "dpo", "ppo", "trl", "vllm", "reasoning", "rlvr"],

  goal: MD(function () {/*
You run **GRPO** (the algorithm behind DeepSeek-R1) with Hugging Face **TRL** on a Colab GPU, training `Qwen2.5-0.5B-Instruct` (or 1.5B) on GSM8K math with a reward **you** wrote in Python — no human labels, no reward model. Your TensorBoard ends up looking like this (typical shape; your numbers will differ):

```text
step   reward  rewards/correctness  rewards/format  completions/mean_length  frac_reward_zero_std
  10    0.21        0.18               0.03              212                     0.55
  60    0.52        0.41               0.11              176                     0.38
 150    0.68        0.51               0.17              158                     0.29
held-out GSM8K test (greedy, strict format):   before 31.8%   →   after 47.9%
```

…and you can explain every column: why the format reward saturates first, why length moves, why `frac_reward_zero_std` matters, and why **~70–90% of each step’s wall-clock was spent generating text**, not computing gradients — the fact that makes RL an inference-engineering problem (M23).

Play with a group of sampled answers below: set their rewards and watch GRPO turn them into advantages.
  */}),
  demo: { viz: "grpo-group", params: {} },

  why: MD(function () {/*
Post-training is where models become useful, and since 2024–25 **RL with verifiable rewards** is where they learn to *reason* (OpenAI o1, DeepSeek-R1, Qwen3, Kimi). Every major lab and many startups now have “post-training” or “RL infra” teams, and job posts for them ask for exactly this stack: PPO/GRPO/DPO, TRL/verl/OpenRLHF, **vLLM or SGLang as the rollout engine**, and reward/verifier pipelines. Companies selling fine-tuning (Together, Fireworks, Baseten customers) increasingly sell RL fine-tuning too.

As an inference engineer you bring the scarce half of the skill set: rollouts are generation, generation is your job, and the training–inference **mismatch** (different kernels → different log-probs → biased gradients) is a problem only someone who knows both sides can fix.
  */}),

  prereqs: [
    {
      title: "Policy gradients, advantages and PPO’s clipped ratio",
      skipIf: "you finished M21",
      md: MD(function () {/*
You need three facts from M21:

1. **Policy gradient:** increase $\log\pi(a\mid s)$ of actions in proportion to their **advantage** $A$ (how much better than expected they did).
2. **Baseline:** subtracting the expected return (a value function, or any per-state average) doesn’t bias the gradient but cuts variance.
3. **PPO:** reuse rollouts for several updates, using the ratio $r=\pi_\theta/\pi_{\text{old}}$ clipped to $[1-\epsilon, 1+\epsilon]$ as a trust region.

If these are fuzzy, do M21’s policy-gradient and PPO lessons first (≈4 hours).
      */}),
    },
    {
      title: "Chat templates, generation and LoRA in Transformers",
      skipIf: "you finished M05 (fine-tuning) and M06",
      md: MD(function () {/*
- **Chat template:** a model-specific way to wrap `[{"role": "system", ...}, {"role": "user", ...}]` into tokens (`tokenizer.apply_chat_template`). RL prompts are usually conversational.
- **Sampling:** `temperature`, `top_p` (M06). RL samples at temperature ≈ 1.0 on purpose — exploration.
- **SFT** (supervised fine-tuning on demonstrations) and **LoRA** (train small low-rank adapters instead of all weights). TRL’s `SFTTrainer` from M05 has a sibling `GRPOTrainer` with almost the same API.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: 60 steps of GRPO on GSM8K — reward climbs, answers change",
      kind: "demo",
      minutes: 90,
      runsOn: ["colab"],
      md: MD(function () {/*
Open a Colab notebook with a GPU (**Runtime → Change runtime type → T4** is enough for this minimal run; L4/A100 is ~3× faster). This version skips vLLM so it’s simple — the full version in the Build lesson adds it.

```python
!pip -q install -U trl datasets transformers accelerate
```

```python
import re, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "Think step by step, then give only the final number inside <answer></answer>."

def to_prompt(ex):
    return {"prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": ex["question"]}],
            "answer": ex["answer"].split("####")[-1].strip().replace(",", "")}

ds = load_dataset("openai/gsm8k", "main", split="train").map(to_prompt, remove_columns=["question"])

def correctness(completions, answer, **kw):
    out = []
    for c, gt in zip(completions, answer):
        m = re.search(r"<answer>\s*([-\d.,]+)\s*</answer>", c[0]["content"])
        ok = m is not None and m.group(1).replace(",", "").rstrip(".") == gt
        out.append(1.0 if ok else 0.0)
    return out

bf16 = torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if bf16 else torch.float32)
args = GRPOConfig(output_dir="grpo-demo", max_steps=60, learning_rate=3e-6,
                  per_device_train_batch_size=8, gradient_accumulation_steps=2,
                  num_generations=8, max_completion_length=256,
                  bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
                  logging_steps=5, log_completions=True, report_to="tensorboard")
trainer = GRPOTrainer(model=model, reward_funcs=correctness, args=args, train_dataset=ds,
                      processing_class=AutoTokenizer.from_pretrained(MODEL))
trainer.train()
```

```python
%load_ext tensorboard
%tensorboard --logdir grpo-demo
```

Watch `reward` (= mean correctness) climb over 60 steps — typically from ~0.2–0.3 to ~0.4–0.5. Most of the early gain is the model learning to **use the answer tag you asked for**; later gains are real accuracy. `log_completions=True` prints sample completions with their rewards to the console.

## Before vs after (illustrative)

```text
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many
   clips in May. How many clips did Natalia sell altogether in April and May?

BEFORE (step 0)                              reward 0.0
  In April, Natalia sold 48 clips. In May, she sold half as many, so 48 / 2 = 24.
  Altogether she sold 48 + 24 = 72 clips.
  **Final answer: 72**                        ← right number, wrong format → 0

AFTER (step 60)                              reward 1.0
  April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.
  <answer>72</answer>
```

> [!INTUITION] What just happened
> For each question the model wrote **8 different answers** (temperature 1.0). Your function scored them 0 or 1. GRPO made the tokens of above-average answers more likely and below-average ones less likely — **no labels, no reward model, just a checker**. That is “RL with verifiable rewards” (RLVR), and it’s how reasoning models are trained.

> [!WARNING] Time check
> Note how long each step takes, and look at the GPU: during most of it the model is **generating** 8×16 = 128 completions one token at a time. The gradient step is a small fraction. Remember this number — we fix it with vLLM in the Build lesson and it is the whole story of M23.

- [ ] Ran 60 GRPO steps and saw `reward` increase in TensorBoard
- [ ] Found one completion that earned reward 1 and one that earned 0 for the same question in the logs
- [ ] Timed one training step and estimated what fraction was generation (watch `nvidia-smi` or add timing)
      */}),
      resources: [
        { title: "TRL GRPOTrainer docs", url: "https://huggingface.co/docs/trl/main/en/grpo_trainer", type: "docs", note: "the API you just used; logged metrics explained" },
        { title: "GSM8K dataset", url: "https://huggingface.co/datasets/openai/gsm8k", type: "docs", note: "8.5k grade-school math problems with numeric answers after ####" },
      ],
    },
    {
      id: "post-training-stack",
      title: "The post-training stack: SFT → RLHF → DPO → RLVR/GRPO",
      kind: "concept",
      minutes: 75,
      runsOn: ["any"],
      md: MD(function () {/*
A pretrained model (M03–M05) is a brilliant autocomplete. **Post-training** turns it into an assistant, then into a reasoner. Four generations of technique, each fixing the previous one’s pain:

| Stage | Data | What’s trained | Key paper | Pain it removed / introduced |
|---|---|---|---|---|
| **SFT** | human-written (prompt, ideal answer) pairs | policy, cross-entropy | — (M05) | easy & stable; but capped by demo quality, teaches *imitation* not *preference* |
| **RLHF (PPO)** | humans rank 2+ answers → **reward model**; then RL against it | reward model, then policy + value with PPO + KL penalty | **InstructGPT** (Ouyang et al. 2022) | learns what humans *prefer*; but 4 models in memory, finicky, RM gets hacked |
| **DPO** | the same preference pairs | policy directly, a classification-like loss | **DPO** (Rafailov et al. 2023) | no RM, no sampling, no RL loop — cheap & stable; but offline, can’t explore beyond the dataset |
| **RLVR / GRPO** | prompts + a **programmatic checker** (math answer, unit tests) | policy with group-relative advantages, no critic | **DeepSeekMath** (GRPO, 2024), **DeepSeek-R1** (2025), Tülu 3 (named “RLVR”) | reward can’t be flattered, scales to long reasoning; but only works where answers are checkable; rollouts are very long and expensive |

## RLHF in one picture (InstructGPT)

```text
Step 1  SFT:      prompts + human demos  ──►  SFT model  (= reference policy π_ref)
Step 2  RM:       prompt → 4–9 SFT samples → humans rank → train RM:  score(prompt, answer) ∈ ℝ
Step 3  PPO:      policy samples answer → RM scores it → reward − β·KL(π ‖ π_ref) → PPO update
```

The famous result: the 1.3B InstructGPT model was **preferred over the 175B GPT-3** by labelers. Post-training is leverage.

## DPO’s shortcut

DPO noticed that the RLHF objective (maximize reward, stay close to $\pi_{\text{ref}}$) has a closed-form optimal policy, and that the reward model can be expressed *through* that policy. So you can skip steps 2–3 and train on preference pairs directly with a logistic loss. You’ll derive it in the DPO lesson.

## RLVR: the reasoning era

DeepSeekMath introduced **GRPO** (drop PPO’s value network; use the mean reward of a group of samples for the same prompt as the baseline). DeepSeek-R1-Zero then showed that running GRPO on a *base* model with only a correctness reward and a format reward makes long chain-of-thought, self-verification and “aha moments” **emerge** — response length grows from hundreds to thousands of tokens over training. R1 added a small cold-start SFT, more RL stages and distillation into small models.

> [!REAL] What frontier pipelines look like now
> Typical recipe (Tülu 3, Qwen3, DeepSeek-V3/R1, Kimi): **SFT** on curated data → **preference optimization** (DPO or RLHF) for style/helpfulness → **RLVR** on math, code, instruction-following constraints and agentic tasks, often in several stages with increasing response length. Everything past SFT needs sampling from the current model — i.e. **inference at massive scale inside the training loop**.

## Where this module goes

- The LLM as a policy and why PPO needs **four models** (lesson 4) — after the one piece of math you need, **KL divergence** (lesson 3).
- **GRPO** in detail (lesson 5) and **DPO** (lesson 6).
- **Reward design** and the zoo of GRPO fixes (lesson 7).
- **Build** a real GRPO run with vLLM (lesson 8) and **evaluate** it properly (lesson 9).

- [ ] Read the InstructGPT abstract + Figure 2 and mapped each box to the table above
- [ ] Read DeepSeek-R1 §2.2 (R1-Zero) and found the plot of response length over training
- [ ] For each of the 4 stages, wrote one sentence on what it would cost *you* to run (data, GPUs, models in memory)
      */}),
      resources: [
        { title: "InstructGPT (Ouyang et al. 2022)", url: "https://arxiv.org/abs/2203.02155", type: "paper", note: "the canonical RLHF pipeline: SFT → RM → PPO" },
        { title: "DPO (Rafailov et al. 2023)", url: "https://arxiv.org/abs/2305.18290", type: "paper", note: "preference optimization without RL" },
        { title: "DeepSeekMath (introduces GRPO)", url: "https://arxiv.org/abs/2402.03300", type: "paper", note: "§4 defines GRPO" },
        { title: "DeepSeek-R1", url: "https://arxiv.org/abs/2501.12948", type: "paper", note: "RLVR at scale; emergent long reasoning" },
        { title: "Tülu 3", url: "https://arxiv.org/abs/2411.15124", type: "paper", note: "fully open post-training recipe; coins “RLVR”" },
      ],
    },
    {
      id: "kl-divergence",
      title: "Math: KL divergence — the leash that keeps the policy near the reference",
      kind: "math",
      minutes: 60,
      runsOn: ["browser"],
      md: MD(function () {/*
Every RLHF/GRPO objective contains a term like $-\beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. Here is what it is and why it’s there.

> [!PREREQ] Log ratios
> $\log\frac{a}{b} = \log a - \log b$. If $a=b$ it’s 0. If $a$ is twice $b$: $\log 2 \approx 0.69$. If half: $\approx -0.69$. A log ratio measures “how many times more likely” on an additive scale — and LLMs hand you log-probabilities directly, so log ratios are just **subtractions of two numbers you already have**.

::viz exp-log

## Definition

For two distributions $P$ and $Q$ over the same outcomes:

$$
\mathrm{KL}(P\,\|\,Q) = \sum_x P(x)\,\log\frac{P(x)}{Q(x)} = \mathbb{E}_{x\sim P}\Big[\log P(x) - \log Q(x)\Big]
$$

Read it as: **sample from $P$, and on average how much more likely was that sample under $P$ than under $Q$?** It’s 0 iff $P=Q$, and always ≥ 0.

**Worked example** (3 “tokens”): $P = [0.7, 0.2, 0.1]$ (the policy after some training), $Q = [0.5, 0.3, 0.2]$ (the reference).

| x | P | Q | log(P/Q) | P·log(P/Q) |
|---|---|---|---|---|
| a | 0.7 | 0.5 | +0.336 | +0.235 |
| b | 0.2 | 0.3 | −0.405 | −0.081 |
| c | 0.1 | 0.2 | −0.693 | −0.069 |
| | | | **KL(P‖Q)** | **0.085 nats** |

The reverse, $\mathrm{KL}(Q\|P)$, is 0.092 — **KL is not symmetric**. $\mathrm{KL}(\pi\|\pi_{\text{ref}})$ (the direction RLHF uses) heavily penalizes the policy for putting probability where the reference puts almost none — i.e. inventing text the original model would never write.

## For sequences: sum over tokens

An LLM’s probability of a completion is a product of token probabilities, so its log is a sum. The sequence KL becomes a sum of per-token log ratios, and we estimate it from the tokens we sampled:

$$
\mathrm{KL} \approx \sum_t \big(\log\pi_\theta(y_t\mid\cdot) - \log\pi_{\text{ref}}(y_t\mid\cdot)\big)
$$

Practical estimators (John Schulman’s note “Approximating KL Divergence”), with $\rho = \pi_{\text{ref}}/\pi_\theta$ at the sampled token:

| name | formula | property |
|---|---|---|
| k1 | $-\log\rho$ | unbiased, high variance, can be negative per sample |
| k2 | $\tfrac12(\log\rho)^2$ | low variance, slightly biased |
| **k3** | $(\rho - 1) - \log\rho$ | **unbiased and always ≥ 0** — used by GRPO/TRL and CleanRL’s `approx_kl` |

```python
import torch
logp_policy = torch.tensor([-0.36, -1.61, -2.30])   # log π_θ of 3 sampled tokens
logp_ref    = torch.tensor([-0.69, -1.20, -1.61])   # log π_ref of the same tokens
log_rho = logp_ref - logp_policy
k1 = -log_rho
k3 = (log_rho.exp() - 1) - log_rho
print(k1.sum().item(), k3.sum().item())
```

## Why the KL penalty exists

1. **Reward models are only accurate near the data they were trained on.** Without a leash the policy drifts to weird text the RM happens to score highly (reward hacking). The KL term makes drifting cost reward.
2. **Keep capabilities.** The reference model is fluent and knowledgeable; staying close preserves that while shifting behaviour.
3. **Math view:** maximizing $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ has the exact solution $\pi^*(y) \propto \pi_{\text{ref}}(y)\,e^{r(y)/\beta}$ — “reweight the reference by exponentiated reward”. Small $\beta$ → chase reward; large $\beta$ → barely move. This formula is the whole basis of DPO.

> [!REAL] Current practice
> In RLHF with a learned RM, KL (β ≈ 0.01–0.1) is essential. In RLVR with a trustworthy verifier many recipes set **β = 0** (DAPO, Dr. GRPO, Open-Reasoner-Zero; TRL’s default is now `beta=0.0`), which also lets you drop the reference model from memory. PPO’s clip still limits each step. Note: PPO’s `approx_kl` measures old-vs-new policy **per update**; the KL penalty measures current-vs-**reference** over the whole run. Different KLs, same formula.

- [ ] Recomputed the table’s KL(P‖Q) and KL(Q‖P) in Python
- [ ] Ran the estimator snippet; confirmed k3 ≥ 0 per token while k1 can be negative
- [ ] Explained in one sentence why β = 0 is reasonable with a math checker but dangerous with a learned reward model
      */}),
      resources: [
        { title: "John Schulman — Approximating KL Divergence", url: "http://joschu.net/blog/kl-approx.html", type: "article", note: "the k1/k2/k3 estimators used in every RL library" },
      ],
    },
    {
      id: "llm-as-policy-ppo",
      title: "The LLM as a policy — and PPO’s four models",
      kind: "concept",
      minutes: 90,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
## Mapping RL onto text generation

| RL (M21) | LLM |
|---|---|
| state $s_t$ | prompt + tokens generated so far |
| action $a_t$ | next token (vocab ≈ 150k) |
| policy $\pi_\theta(a\mid s)$ | the LLM’s softmax over the vocab |
| episode | one completion, until EOS or max length |
| transition | deterministic: append the token |
| reward | **usually one number at the end** (RM score or checker), plus optional per-token KL penalty |
| $\gamma$ | 1 |

The log-probability of a whole completion is the sum of its token log-probs, which one forward pass gives you:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)   # CPU/MPS is fine

prompt = tok.apply_chat_template([{"role": "user", "content": "2+2?"}], tokenize=False, add_generation_prompt=True)
completion = "4"
ids = tok(prompt + completion, return_tensors="pt").input_ids
n_prompt = len(tok(prompt).input_ids)
with torch.no_grad():
    logits = model(ids).logits[:, :-1]                       # position t predicts token t+1
logp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
completion_logp = logp[:, n_prompt - 1:]                     # only the completion tokens
print(completion_logp, completion_logp.sum())                # per-token and sequence log-prob
```

This “recompute log-probs of sampled tokens” forward pass is done by **the trainer** (for the policy, the reference, and the old policy) on every batch. Getting the *same* numbers the sampler (vLLM) got is harder than it looks — different kernels, precision, batch shapes → slightly different log-probs → the **training–inference mismatch** TRL corrects with importance sampling (`vllm_importance_sampling_correction`).

## PPO for LLMs: four models

```text
               ┌────────────────── rollout ──────────────────┐
 prompts ──►  POLICY πθ (trainable)  ──► completions
               │                                           │
               ▼                                           ▼
     REFERENCE π_ref (frozen)                     REWARD MODEL (frozen)
     per-token KL penalty                          scalar score at EOS
               │                                           │
               └──► per-token rewards  rₜ = −β·KLₜ  (+ RM score on last token)
                                   │
                    VALUE / CRITIC Vφ (trainable) ── GAE advantages ──► PPO update of πθ and Vφ
```

| Model | Trainable? | Size (typical) | Role |
|---|---|---|---|
| **Policy (actor)** | yes | N | generates; updated with clipped PPO |
| **Value (critic)** | yes | ≈ N (often init from RM) | predicts expected reward **at every token** → GAE advantages |
| **Reference** | no | N | frozen SFT model for the KL penalty |
| **Reward model** | no | ≈ N | scores full completions |

**Memory napkin math (7B, bf16 mixed-precision + Adam, from M05):** a trainable model needs ~16 bytes/param → **112 GB** each for policy and critic = 224 GB. Frozen ref + RM: 2 × 14 GB = 28 GB. Plus a rollout engine copy (vLLM) with its KV cache, plus activations. **≈ 300 GB before activations → 4–8 × 80 GB GPUs for a 7B model.** This is why RLHF frameworks (OpenRLHF, verl) are distributed systems from day one.

> [!INTUITION] Why the critic is the weakest link
> The critic must predict, from a half-written answer, how the reward model will score the *finished* answer. That’s nearly as hard as the task itself, it’s a whole extra LLM to train, and a bad critic gives bad advantages. GRPO’s insight: for a prompt, just **sample several answers and use their average reward as the baseline**. No critic.

## The PPO clip, now per token

The clipped objective is identical to M21 — each generated token is an action with ratio $r_t = \pi_\theta(y_t\mid\cdot)/\pi_{\text{old}}(y_t\mid\cdot)$:

::viz ppo-clip

With ε = 0.2 and a few epochs per rollout batch. In LLM RL it is common to do only **1–2** gradient steps per batch (TRL default `num_iterations=1`): the clip is then mainly a safety net for asynchrony and training–inference mismatch (M23), not for data reuse.

> [!REAL] What the frameworks call these
> verl: `actor_rollout_ref` (policy + its vLLM/SGLang rollout + reference, often colocated), `critic`, `reward_model`. OpenRLHF: separate Ray actor groups for each of the four models plus a vLLM engine group. TRL `PPOTrainer` exists but TRL’s focus (and the community’s) moved to `GRPOTrainer`.

- [ ] Ran the log-prob snippet; changed the completion to “5” and compared the sequence log-prob
- [ ] Computed the PPO memory budget for a 1.5B and a 70B model; how many 80 GB GPUs does each need?
- [ ] Listed which of the 4 models GRPO removes, and which one β = 0 additionally removes
      */}),
      resources: [
        { title: "Nathan Lambert — RLHF Book", url: "https://rlhfbook.com/", type: "book", note: "free book; the best end-to-end reference for RLHF/PPO/DPO/RLVR" },
        { title: "Fine-Tuning LMs from Human Preferences (Ziegler et al. 2019)", url: "https://arxiv.org/abs/1909.08593", type: "paper", note: "the original PPO-for-LMs setup with per-token KL" },
      ],
    },
    {
      id: "grpo",
      title: "GRPO: group-relative advantages, no critic",
      kind: "concept",
      minutes: 90,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
## The algorithm

For each prompt $q$:

1. Sample a **group** of $G$ completions $o_1,\dots,o_G$ from the current policy (TRL `num_generations`, typically 8–16; DeepSeek used 16–64).
2. Score each with the reward function: $r_1,\dots,r_G$.
3. **Advantage** of every token in completion $i$ = its completion’s z-score within the group:

$$
\hat A_{i} = \frac{r_i - \operatorname{mean}(r_1..r_G)}{\operatorname{std}(r_1..r_G) + \varepsilon}
$$

4. PPO-style clipped update on all tokens, optionally minus $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$.

That’s it. The group mean is the **baseline** (M21) — a Monte-Carlo estimate of $V(q)$ from $G$ samples, instead of a learned critic.

::viz grpo-group

## Worked example

Group of 8 answers to one GSM8K question, correctness rewards `[1, 0, 0, 1, 0, 0, 0, 1]`:

- mean = 0.375, std = $\sqrt{0.375 \times 0.625} = 0.484$
- correct answers: $(1 - 0.375)/0.484 = +1.29$ → every token pushed **up**
- wrong answers: $(0 - 0.375)/0.484 = -0.77$ → every token pushed **down**

```python
import numpy as np
r = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=float)
adv = (r - r.mean()) / (r.std() + 1e-4)
print(adv.round(2))     # [ 1.29 -0.77 -0.77  1.29 -0.77 -0.77 -0.77  1.29]
```

Now the degenerate cases, which dominate real runs:

| group rewards | advantages | learning signal |
|---|---|---|
| all 0 (too hard) | all 0 | **none** — wasted generation |
| all 1 (too easy) | all 0 | **none** — wasted generation |
| mixed | ± | yes |

TRL logs this as `frac_reward_zero_std`. If it’s 0.8, **80% of your (expensive) rollouts taught nothing**. Fixes: filter the dataset to problems of medium difficulty for the current model; DAPO’s **dynamic sampling** (keep generating until the batch has enough mixed groups); larger $G$.

## Why every token gets the same advantage

GRPO doesn’t know *which* tokens made the answer right — it credits all of them equally (the reward is only at the end, $\gamma = 1$, no critic to localize credit). Over many samples, tokens that consistently appear in good answers (a correct arithmetic step, a “let me double-check”) get reinforced, and ones in bad answers get suppressed. Crude but it works — and it is why RLVR needs **lots** of samples.

## The loss, as TRL implements it

$$
\mathcal{L} = -\frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\Big[\min\big(r_{i,t}\hat A_i,\ \text{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big) - \beta\,\mathrm{KL}_{i,t}\Big]
$$

with $r_{i,t} = \pi_\theta(o_{i,t}\mid q, o_{i,<t}) / \pi_{\text{old}}(\cdot)$. The normalization in front (divide by total tokens vs per-sequence length vs a constant) matters more than you’d think — lesson 7.

## GRPO vs PPO for LLMs

| | PPO (RLHF) | GRPO |
|---|---|---|
| Baseline | learned critic $V_\phi$ per token | group mean per prompt |
| Models in memory | 4 | 2 (policy + ref), or 1 with β = 0 |
| Samples per prompt | 1 usually | G = 8–64 |
| Credit assignment | per token (GAE) | per sequence |
| Compute profile | more training compute | **more generation** (G× rollouts) |

That last row is the inference engineer’s punchline: GRPO trades a critic’s memory and training cost for **G times more generation**. Rollout throughput becomes *the* bottleneck.

<details><summary>Related critic-free estimators: RLOO and REINFORCE++</summary>

**RLOO** (REINFORCE Leave-One-Out, Ahmadian et al. 2024 “Back to Basics”) uses the mean of the *other* $G-1$ samples as the baseline and doesn’t divide by std — an unbiased estimator. It predates GRPO’s popularity and performs similarly. **REINFORCE++** (in OpenRLHF) normalizes advantages over the global batch instead of per group. All are “policy gradient with a sampled baseline”; the differences are normalization choices.

</details>

- [ ] In the grpo-group viz, found reward settings that give zero learning signal and explained why
- [ ] Computed advantages by hand for rewards `[1, 1, 1, 0]` — why is the single wrong answer pushed down so hard?
- [ ] Read DeepSeekMath §4.1 (GRPO) and matched each symbol in their equation to the loss above
      */}),
      resources: [
        { title: "DeepSeekMath (GRPO)", url: "https://arxiv.org/abs/2402.03300", type: "paper", note: "§4.1: GRPO objective and algorithm box" },
        { title: "Back to Basics: REINFORCE-style optimization (RLOO)", url: "https://arxiv.org/abs/2402.14740", type: "paper", note: "why simpler critic-free estimators work for LLMs" },
      ],
    },
    {
      id: "dpo",
      title: "DPO: preference learning without RL (derivation intuition)",
      kind: "math",
      minutes: 90,
      runsOn: ["colab", "mac"],
      md: MD(function () {/*
Not every task has a checker. “Which of these two emails is more polite?” needs human (or AI) **preferences**. DPO learns from them without a reward model or sampling loop.

> [!PREREQ] Sigmoid and log-odds
> $\sigma(z) = \frac{1}{1+e^{-z}}$ squashes any number into (0, 1). $\sigma(0) = 0.5$, $\sigma(2) \approx 0.88$, $\sigma(-2)\approx 0.12$. Logistic regression = “probability that A beats B is $\sigma(\text{score}_A - \text{score}_B)$”. The loss $-\log\sigma(z)$ is small when $z$ is large and positive.

## Step 1 — preference data and the Bradley–Terry model

Data: triples $(x, y_w, y_l)$ — prompt, preferred (“winner”) answer, rejected (“loser”) answer. Assume humans prefer $y_w$ with probability

$$P(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

for some hidden reward $r$. RLHF step 2 fits a reward model $r_\phi$ by maximizing this likelihood. Only reward **differences** matter.

## Step 2 — the optimal RLHF policy has a closed form

From the KL lesson: the policy maximizing $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ is

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{r(x,y)/\beta}$$

($Z(x)$ just normalizes so probabilities sum to 1 — intractable, but watch it vanish.)

## Step 3 — flip it: the reward is hidden inside the policy

Take logs and solve for $r$:

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

This is the **implicit reward**: any policy defines a reward via how much it up-weights $y$ relative to the reference.

## Step 4 — plug into Bradley–Terry; $Z$ cancels

In the difference $r(x,y_w) - r(x,y_l)$ the $\beta\log Z(x)$ terms cancel (same prompt). Replace $\pi^*$ by the policy we’re training, $\pi_\theta$, and maximize likelihood:

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\Big[\underbrace{\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{winner margin}} - \underbrace{\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{loser margin}}\Big]\Big)
$$

**In words: increase the log-prob of the winner and decrease the loser’s, relative to the reference, until the gap is comfortably positive.** Four forward passes (policy & ref × winner & loser), one backward. No generation.

**Numeric example** (β = 0.1): sequence log-probs — policy: winner −40, loser −42; reference: winner −41, loser −41. Margins: winner +1, loser −1 → $z = 0.1 \times (1-(-1)) = 0.2$ → loss $= -\log\sigma(0.2) = 0.598$. Before any training ($\pi_\theta=\pi_{\text{ref}}$), $z=0$ and loss $= \log 2 = 0.693$ — you always start there, which is a nice sanity check.

```python
import torch, torch.nn.functional as F
def dpo_loss(pw, pl, rw, rl, beta=0.1):          # sequence log-probs: policy/ref × winner/loser
    z = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(z).mean()
print(dpo_loss(torch.tensor([-40.]), torch.tensor([-42.]), torch.tensor([-41.]), torch.tensor([-41.])))
```

## Run it (TRL)

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2000]")   # columns: chosen, rejected
trainer = DPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct",
                     args=DPOConfig(output_dir="dpo-demo", beta=0.1, per_device_train_batch_size=4,
                                    max_steps=200, logging_steps=10),
                     train_dataset=ds)
trainer.train()
```

Watch `rewards/chosen`, `rewards/rejected` (the implicit rewards) and `rewards/accuracies` (fraction of pairs where chosen > rejected). On a Mac, `mlx-lm-lora` supports `--train-mode dpo` with LoRA.

## DPO vs online RL

| | DPO | PPO / GRPO |
|---|---|---|
| Data | fixed pairs (offline) | fresh samples from current policy (online) |
| Cost | ~SFT cost | dominated by generation |
| Exploration | none beyond the dataset | yes |
| Failure modes | overfits pairs, can *lower* both chosen and rejected log-probs, length exploitation | reward hacking, instability |

Common practice: DPO (or online/iterative DPO) for chat style and safety; RLVR for reasoning.

- [ ] Verified the loss starts at $\log 2 \approx 0.693$ when policy = reference
- [ ] Ran 200 DPO steps and plotted `rewards/accuracies` and `rewards/margins`
- [ ] Explained where $Z(x)$ went and why that’s the key trick
- [ ] Checked whether chosen log-probs actually went *up* during training (often they don’t — why is the loss still decreasing?)
      */}),
      resources: [
        { title: "DPO paper", url: "https://arxiv.org/abs/2305.18290", type: "paper", note: "§4 has the derivation; Appendix A.1 the closed-form optimum" },
        { title: "TRL DPOTrainer docs", url: "https://huggingface.co/docs/trl/main/en/dpo_trainer", type: "docs", note: "API, logged metrics and loss variants" },
      ],
    },
    {
      id: "reward-design",
      title: "Reward design: verifiable & format rewards, hacking, length bias — and DAPO, Dr. GRPO, GSPO",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
In RLVR, **the reward function is the spec**. The model will find every gap in it.

## Kinds of rewards

| Reward | Example | Hackable? |
|---|---|---|
| **Verifiable / outcome** | parse final number, compare to ground truth; run unit tests; check JSON schema | hard (but see below) |
| **Format** | +0.1–0.2 if output matches `<think>…</think><answer>…</answer>` | easy — keep it small |
| **Learned reward model** | RM scores helpfulness | **very** — needs KL, RM ensembles, fresh RMs |
| **LLM-as-judge** | another model grades the answer against a rubric | yes: flattery, length, confident tone |
| **Process reward** (PRM) | score each reasoning step | expensive labels; DeepSeek-R1 reported PRMs weren’t worth it at scale |

## A robust GSM8K reward (and what goes wrong)

```python
import re
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

def parse(text):
    tags = ANSWER.findall(text)
    if len(tags) != 1:                 # exactly one answer tag — no "hedging" with several
        return None
    nums = NUMBER.findall(tags[0].replace(",", ""))
    return float(nums[-1]) if len(nums) == 1 else None   # exactly one number inside

def correctness_reward(completions, answer, **kw):
    out = []
    for c, gt in zip(completions, answer):
        pred = parse(c[0]["content"])
        out.append(1.0 if pred is not None and abs(pred - float(gt)) < 1e-6 else 0.0)
    return out
```

Hacks this closes, all observed in practice: printing several `<answer>` tags hoping one matches; writing “the answer is between 10 and 20” so a lenient regex grabs the right number; the answer tag inside the thinking section; unit tests that can be special-cased; code that calls `exit(0)` before failing tests; editing the test file in an agentic sandbox.

## Length: the most important side effect

Response length is a key metric in every reasoning-RL paper. Two opposite pressures:

- **Longer is often genuinely better** for hard problems (more reasoning steps, self-checking). R1-Zero’s length grew steadily.
- **Length bias from the loss:** in the original GRPO loss each sequence’s token-loss is averaged over its own length $|o_i|$. For a **wrong** answer (negative advantage) a long sequence spreads the penalty thin — so being wrong *and long* is punished less per token. Result: wrong answers get longer and longer. Also: learned RMs and LLM judges prefer longer answers.
- **Truncation:** answers hitting `max_completion_length` usually get reward 0 even if they were on track; the model learns “long = bad” for the wrong reason. Options: `mask_truncated_completions=True` (drop them from the loss), or DAPO’s soft overlong penalty.

Always plot `completions/mean_length` and `completions/clipped_ratio` next to reward.

## The GRPO fixes zoo (2025)

| Variant | Change | Why | In TRL |
|---|---|---|---|
| **DAPO** (ByteDance) | *Clip-higher* ($\epsilon_{\text{low}}{=}0.2$, $\epsilon_{\text{high}}{=}0.28$); **dynamic sampling** (drop all-0/all-1 groups, resample); **token-level** loss normalization; overlong reward shaping | prevent entropy collapse; no wasted batches; fix length bias | `epsilon_high=0.28`, `loss_type="dapo"` (default), `mask_truncated_completions=True` |
| **Dr. GRPO** (“GRPO Done Right”) | remove per-sequence length normalization **and** std normalization | length bias and question-difficulty bias | `loss_type="dr_grpo"`, `scale_rewards=False` |
| **GSPO** (Qwen) | importance ratio and clipping at the **sequence** level (length-normalized) instead of per token | per-token ratios are noisy, especially for MoE where routing changes between old/new policy; used for Qwen3 | `importance_sampling_level="sequence"` |

> [!INTUITION] They’re all about normalization and trust regions
> None of these change the core idea (sampled baseline + clipped policy gradient). They change **what you average over** and **how far each update may move**. When you read a new “XPO” paper, ask: what’s the baseline, what’s the normalizer, what’s clipped?

- [ ] Wrote 5 adversarial completions that fool a naive `re.search(r"\d+")` reward and confirmed the robust parser rejects them
- [ ] Explained, with a 2-sequence numeric example, how per-sequence length normalization penalizes a long wrong answer less per token
- [ ] Matched each DAPO trick to a TRL config flag
      */}),
      resources: [
        { title: "DAPO", url: "https://arxiv.org/abs/2503.14476", type: "paper", note: "clip-higher, dynamic sampling, token-level loss, overlong shaping" },
        { title: "Understanding R1-Zero-Like Training (Dr. GRPO)", url: "https://arxiv.org/abs/2503.20783", type: "paper", note: "length and difficulty biases in GRPO" },
        { title: "GSPO", url: "https://arxiv.org/abs/2507.18071", type: "paper", note: "sequence-level ratios; stability for MoE" },
      ],
    },
    {
      id: "build-grpo",
      title: "Build: a real GRPO run with TRL + vLLM — and why rollouts dominate",
      kind: "build",
      minutes: 180,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
## Hardware

| Option | Model | Notes |
|---|---|---|
| Free Colab **T4** (16 GB, no bf16) | 0.5B | works with the settings below; slow (fp16 AMP, small vLLM slice) |
| Colab **L4 / A100** or rented **A100/H100** (~\$1–3/hr) | 0.5B–1.5B | recommended; 200 steps ≈ 1 hour on an A100 for 0.5B |
| **Mac** (optional) | 0.5B, LoRA | `mlx-lm-lora --train-mode grpo` (below); great for understanding, slower |

## The script

```python
# grpo_gsm8k.py   —   pip install -U "trl[vllm]" datasets transformers accelerate tensorboard
import re, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"        # or Qwen/Qwen2.5-1.5B-Instruct on an A100/H100
SYSTEM = ("You are a careful math solver. Reason step by step inside <think></think>, "
          "then write only the final number inside <answer></answer>.")

def to_prompt(ex):
    return {"prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": ex["question"]}],
            "answer": ex["answer"].split("####")[-1].strip().replace(",", "")}

train = load_dataset("openai/gsm8k", "main", split="train").map(to_prompt, remove_columns=["question"])

ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
FORMAT = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)

def parse(text):
    tags = ANSWER.findall(text)
    if len(tags) != 1: return None
    nums = NUMBER.findall(tags[0].replace(",", ""))
    return float(nums[-1]) if len(nums) == 1 else None

def correctness_reward(completions, answer, **kw):
    return [1.0 if (p := parse(c[0]["content"])) is not None and abs(p - float(gt)) < 1e-6 else 0.0
            for c, gt in zip(completions, answer)]

def format_reward(completions, **kw):
    return [0.2 if FORMAT.match(c[0]["content"]) else 0.0 for c in completions]

if __name__ == "__main__":             # so eval_gsm8k.py can import SYSTEM/parse without training
    bf16 = torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if bf16 else torch.float32)
    args = GRPOConfig(
        output_dir="qwen-grpo-gsm8k",
        max_steps=200,
        learning_rate=2e-6,
        per_device_train_batch_size=16,     # completions per micro-batch
        gradient_accumulation_steps=4,      # → 64 completions per step = 8 prompts × 8 samples
        num_generations=8,                  # G: the group size
        max_completion_length=512,
        temperature=1.0,
        beta=0.0,                           # no KL / no reference model (verifiable reward)
        loss_type="dr_grpo",                # try "dapo" (default) and compare
        mask_truncated_completions=True,
        use_vllm=True,                      # generate with vLLM instead of model.generate()
        vllm_mode="colocate",               # vLLM lives in the same process/GPU as training
        vllm_gpu_memory_utilization=0.3 if bf16 else 0.25,
        gradient_checkpointing=True,
        bf16=bf16, fp16=not bf16,
        logging_steps=5, log_completions=True, report_to="tensorboard",
        save_steps=100,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, args=args, train_dataset=train,
                          reward_funcs=[correctness_reward, format_reward])
    trainer.train()
    trainer.save_model("qwen-grpo-gsm8k/final")
    tok.save_pretrained("qwen-grpo-gsm8k/final")
```

```bash
python grpo_gsm8k.py        # or: accelerate launch grpo_gsm8k.py   (multi-GPU)
tensorboard --logdir qwen-grpo-gsm8k
```

What to watch: `rewards/correctness_reward/mean`, `rewards/format_reward/mean` (saturates first), `completions/mean_length`, `completions/clipped_ratio`, `frac_reward_zero_std`, `entropy`, `step_time`.

> [!TIP] If vLLM misbehaves on a T4
> Set `use_vllm=False`. Everything still works — just slower. Run 20 steps each way and compare `step_time`: that ratio is the lesson.

## Why rollout generation dominates — napkin math

One step above = 64 completions × up to 512 tokens.

- **Training compute:** forward + backward ≈ $6 \times N \times \text{tokens}$ = $6 \times 0.5\text{B} \times (64 \times {\sim}600) \approx 1.2\times10^{14}$ FLOPs. On an A100 (312 TFLOPS bf16) at 40% MFU: **≈ 1 s**. Add the log-prob forward passes: ~1.5 s.
- **Generation:** 512 sequential decode steps. Each is memory-bound (M07): read 1 GB of weights + the KV cache. Even a perfect engine needs ~512 × (1 GB / 2 TB/s) ≈ 0.25 s; HF `generate()` with padding and Python overhead is typically **20–40 ms/step → 10–20 s**. vLLM with continuous batching and CUDA graphs: a few seconds.
- And it gets worse with **long reasoning** (4k–32k tokens) and the **long tail**: the batch finishes only when the *longest* completion finishes.

So even for a tiny model, generation is ~70–90% of the step without an inference engine — the same ratio reported for large-scale reasoning RL. Every trick you learned in M06–M10 (**KV cache, paged attention, continuous batching, CUDA graphs, prefix caching** — all 8 samples share the same prompt!) directly speeds up RL training. That’s why TRL, verl, OpenRLHF, slime all embed vLLM or SGLang.

> [!IMPORTANT] What `use_vllm=True` costs you
> 1. **Weight sync:** after every optimizer step the new weights must be copied into vLLM (colocate: in-process copy; server mode: NCCL broadcast to another GPU). 2. **Memory:** vLLM’s KV cache competes with optimizer states (`vllm_gpu_memory_utilization`, `vllm_enable_sleep_mode`). 3. **Mismatch:** vLLM’s log-probs differ slightly from the trainer’s — TRL applies truncated importance sampling by default. All three are the core topics of M23.

For multi-GPU runs, TRL’s **server mode** (`vllm_mode="server"`) runs vLLM on dedicated GPUs and the trainer on the rest — a disaggregated placement. The server launch command has changed across TRL versions; follow the current GRPO docs.

## Optional: GRPO on a Mac with MLX

```bash
uv pip install mlx-lm-lora
mlx_lm_lora.train --model Qwen/Qwen2.5-0.5B-Instruct --train --train-mode grpo \
  --data mlx-community/gsm8k --group-size 4 --max-completion-length 256 \
  --reward-functions "accuracy_reward,format_reward" --reward-weights "[0.8, 0.2]"
```

LoRA by default; slower than a GPU, but you can watch the whole loop on your laptop and read its source.

- [ ] Ran ≥ 150 GRPO steps with vLLM; saved TensorBoard screenshots of reward, format reward, length, zero-std fraction
- [ ] Ran 20 steps with `use_vllm=False` and 20 with `use_vllm=True`; recorded `step_time` for each
- [ ] Wrote down the generation-vs-training split of one step (use `torch.cuda.synchronize()` + timers or the PyTorch profiler)
- [ ] Ran once with `loss_type="dapo"` and once with `"dr_grpo"`; compared length curves
      */}),
      resources: [
        { title: "Hugging Face TRL", url: "https://github.com/huggingface/trl", type: "repo", note: "GRPOTrainer source: read `_generate_and_score_completions`" },
        { title: "mlx-lm-lora", url: "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora", type: "repo", note: "SFT/DPO/GRPO with MLX on Apple Silicon" },
        { title: "Open R1 (Hugging Face)", url: "https://github.com/huggingface/open-r1", type: "repo", note: "open R1 reproduction built on TRL GRPO + vLLM" },
      ],
    },
    {
      id: "evaluate",
      title: "Lab: evaluating an RL-trained model — held-out accuracy, pass@k, and SFT baseline",
      kind: "lab",
      minutes: 120,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
Training reward is **not** an evaluation: it’s measured on training prompts, at temperature 1, with your (possibly hackable) reward. Evaluate on the **GSM8K test split** (1,319 problems) that training never saw.

## Two metrics

- **Greedy accuracy (pass@1, T = 0):** one answer per problem. What a user sees.
- **pass@k:** probability that at least one of $k$ samples is correct. Measures what the model *can* do. Unbiased estimator (Chen et al. 2021, Codex paper): sample $n \ge k$ answers, count $c$ correct:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

```python
import numpy as np
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))   # numerically stable form
print(pass_at_k(16, 3, 1), pass_at_k(16, 3, 8))   # 0.1875, ~0.9
```

## Fast eval with vLLM

```python
# eval_gsm8k.py  — pip install vllm datasets
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from grpo_gsm8k import SYSTEM, parse          # reuse your prompt + parser (training is behind __main__)

test = load_dataset("openai/gsm8k", "main", split="test")
msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}] for q in test["question"]]
gts = [float(a.split("####")[-1].strip().replace(",", "")) for a in test["answer"]]

def evaluate(model_path, n=8):
    llm = LLM(model=model_path, gpu_memory_utilization=0.85)
    greedy = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=768))
    sampled = llm.chat(msgs, SamplingParams(temperature=1.0, n=n, max_tokens=768, seed=0))
    ok = lambda text, gt: (p := parse(text)) is not None and abs(p - gt) < 1e-6
    acc = np.mean([ok(o.outputs[0].text, gt) for o, gt in zip(greedy, gts)])
    cs = [sum(ok(x.text, gt) for x in o.outputs) for o, gt in zip(sampled, gts)]
    lens = np.mean([len(o.outputs[0].token_ids) for o in greedy])
    return {"greedy_acc": acc, "pass@1": np.mean([pass_at_k(n, c, 1) for c in cs]),
            f"pass@{n}": np.mean([pass_at_k(n, c, n) for c in cs]), "mean_len": lens}

print("base", evaluate("Qwen/Qwen2.5-0.5B-Instruct"))
print("grpo", evaluate("qwen-grpo-gsm8k/final"))
```

(Import `pass_at_k` from the snippet above. Run each model in a fresh process if GPU memory is tight.) vLLM evaluates all 1,319 problems × 8 samples in a couple of minutes on an A100 — **the same speedup as in training**.

## Reading the results

Typical pattern for small models (your numbers will vary):

| model | greedy acc (strict format) | pass@1 (T=1) | pass@8 | mean length |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ~30% | ~20% | ~65% | ~250 |
| + GRPO 200 steps | ~45–50% | ~40% | ~70% | ~180 |

Things to notice:

1. **Much of the early gain is format**: Qwen reports ~50% GSM8K for this model with its own prompt/parser. Evaluate with a *lenient* parser too, to separate “learned to follow the format” from “learned to solve more problems”.
2. **pass@1 rises a lot, pass@k much less.** A known finding (e.g. “Does RL really incentivize reasoning capacity…”, 2025): RLVR mostly **sharpens** the distribution toward answers the base model could already sometimes produce. At very large k the base model can even win. Discuss this in your report.
3. **Length change**: did GRPO make answers shorter (efficient) or longer (more reasoning)? With a 0.5B model and short GSM8K problems, often shorter.

## Baseline: SFT-only

The fair comparison is SFT on the **same prompts** with the **reference solutions** (GSM8K’s `answer` field has worked solutions). Reformat each to `<think>{reasoning}</think><answer>{number}</answer>` and train with TRL’s `SFTTrainer` (M05) for ~1 epoch. SFT teaches imitation of human-style solutions; GRPO teaches the model to find *its own* successful paths. Which generalizes better to a different distribution (e.g. the MATH dataset or SVAMP)?

- [ ] Evaluated base and GRPO models on the full GSM8K test set: greedy accuracy, pass@1, pass@8, mean length
- [ ] Also evaluated with a lenient parser (last number anywhere) to separate format gains from accuracy gains
- [ ] Trained an SFT-only baseline on reformatted GSM8K solutions and evaluated it the same way
- [ ] Wrote 3 sentences on what RL changed that SFT didn’t (or vice versa)
      */}),
      resources: [
        { title: "Evaluating LLMs Trained on Code (Codex, pass@k)", url: "https://arxiv.org/abs/2107.03374", type: "paper", note: "§2.1 defines the unbiased pass@k estimator" },
        { title: "Understanding R1-Zero repo", url: "https://github.com/sail-sg/understand-r1-zero", type: "repo", note: "minimal, well-evaluated R1-Zero-style experiments" },
      ],
    },
  ],

  challenge: {
    title: "GRPO run report",
    md: MD(function () {/*
Write `course-work/m22/REPORT.md` for a GRPO run of **your own configuration** (Qwen2.5-0.5B or 1.5B-Instruct, GSM8K or a math dataset of your choice, ≥ 200 steps):

1. **Setup:** model, hardware, G, batch, LR, β, loss type, max length, vLLM on/off, total wall-clock and GPU cost.
2. **Curves:** correctness reward, format reward, held-out accuracy at 3+ checkpoints (0, mid, end), response length (mean + clipped ratio), `frac_reward_zero_std`, entropy.
3. **Time breakdown:** % of step time in generation vs training vs weight sync (measure it).
4. **Failure modes:** 3 concrete examples (full completions) of reward hacking, format gaming, truncation, repetitive loops, or degenerate reasoning — each with an explanation and a proposed reward/config fix.
5. **SFT comparison:** SFT-only baseline on the same data, same eval: greedy accuracy, pass@8, length. Which wins, where, and why?
6. **One ablation** of your choice: `loss_type` dapo vs dr_grpo, G = 4 vs 16, β = 0 vs 0.04, or `epsilon_high`.
    */}),
    checklist: [
      "Reward and held-out accuracy curves with at least 3 checkpoints evaluated on GSM8K test",
      "Response-length trend plotted with an explanation of why it moved",
      "Measured generation vs training time share per step",
      "Three documented failure modes with full example completions and proposed fixes",
      "SFT-only baseline evaluated identically (greedy acc, pass@8, length)",
      "One ablation with a clear conclusion",
    ],
    stretch: "Add a second verifiable task (e.g. Countdown: reach a target number with given integers — the TinyZero task) with its own reward function, train on a mix, and check whether math accuracy is preserved. Or run the same experiment with DPO on self-generated (correct, incorrect) pairs and compare to GRPO.",
  },

  connects: MD(function () {/*
You saw that **generation dominates RL training time**, that the trainer and the inference engine must **share weights after every step**, and that their **log-probs must agree**. M23 is about the systems built around exactly those three facts: how verl, OpenRLHF, slime, AReaL, prime-rl and PipelineRL place rollout engines and trainers across hundreds of GPUs, sync weights over NCCL, handle stragglers from long reasoning chains, and go asynchronous (with importance-sampling corrections — the same math as PPO’s ratio). Your inference-engine skills (M06–M10) and parallelism skills (M16–M17) are what make those systems fast.
  */}),

  interview: [
    "Walk through the RLHF pipeline from InstructGPT. What models are in memory during the PPO phase, and roughly how much memory does a 7B setup need?",
    "Derive DPO’s loss from the KL-regularized RL objective. Why does the partition function cancel?",
    "How does GRPO compute advantages? What happens when all samples in a group get the same reward, and how does DAPO handle it?",
    "What is the length bias in the original GRPO loss, and how do DAPO and Dr. GRPO address it?",
    "Why is the KL penalty essential with a learned reward model but often dropped with verifiable rewards?",
    "In a GRPO run, 80% of wall-clock is generation. List five ways to reduce it.",
    "Your vLLM rollout log-probs differ from the trainer’s recomputed log-probs. Why does this happen and why does it matter?",
    "What’s the difference between pass@1 and pass@k, and what does it mean if RL improves pass@1 but not pass@64?",
  ],

  resources: [
    { title: "Nathan Lambert — RLHF Book", url: "https://rlhfbook.com/", type: "book", note: "free, up-to-date reference for the whole module" },
    { title: "InstructGPT", url: "https://arxiv.org/abs/2203.02155", type: "paper", note: "RLHF with PPO" },
    { title: "DPO", url: "https://arxiv.org/abs/2305.18290", type: "paper", note: "preference optimization without RL" },
    { title: "DeepSeekMath (GRPO)", url: "https://arxiv.org/abs/2402.03300", type: "paper", note: "introduces GRPO" },
    { title: "DeepSeek-R1", url: "https://arxiv.org/abs/2501.12948", type: "paper", note: "reasoning via large-scale RLVR" },
    { title: "Hugging Face TRL", url: "https://github.com/huggingface/trl", type: "repo", note: "GRPOTrainer, DPOTrainer, SFTTrainer" },
    { title: "TRL GRPOTrainer docs", url: "https://huggingface.co/docs/trl/main/en/grpo_trainer", type: "docs", note: "loss types, vLLM modes, mismatch correction" },
    { title: "DAPO", url: "https://arxiv.org/abs/2503.14476", type: "paper", note: "the most practical GRPO improvements" },
    { title: "Open R1 for students (HF LLM course ch. 12)", url: "https://huggingface.co/learn/llm-course/chapter12/1", type: "course", note: "guided GRPO notebooks" },
    { title: "TinyZero (archived)", url: "https://github.com/Jiayi-Pan/TinyZero", type: "repo", note: "minimal R1-Zero reproduction on Countdown; unmaintained, points to verl" },
  ],
});
