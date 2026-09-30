Course.module({
  id: "m02-language-models",
  title: "Language models: from counting to neural nets (makemore)",
  short: "Language models (makemore)",
  tagline: "Build a character-level name generator three ways — counting, a one-layer neural net, and Bengio's MLP — and learn the loss, sampling and embedding ideas every LLM (and every inference engine's sampler) is built on.",
  hours: 18,
  level: "core",
  runsOn: ["mac", "browser"],
  tags: ["language-modeling", "makemore", "cross-entropy", "sampling", "embeddings", "softmax"],

  goal: MD(function () {/*
A model that has read 32,033 real names and invents new, plausible ones:

```text
model            params    train loss   val loss   samples
uniform guess         0      3.296        3.296    "xqzwjfa.", "bdpvz."
bigram (counts)     729      2.454        2.46     "mor.", "axx.", "minaymoryles.", "kondlaisah."
bigram (neural)     729      2.47         2.48     (same model, learned by gradient descent)
MLP (Bengio '03) 11,897      2.12         2.17     "carmah.", "amelle.", "khi.", "mili.", "taty.", "skanden.", "jazhien."
```

Every number in that table is a **negative log-likelihood** — the same cross-entropy loss GPT-4 is trained on. And every sample was drawn by the same procedure an inference engine runs millions of times a second: **logits → temperature → softmax → sample one token → append → repeat**.

Train a bigram model in your browser right now — edit the text and watch the counts, the loss and the samples change:
  */}),
  demo: { viz: "bigram-live" },

  why: MD(function () {/*
**Next-token prediction is the whole game.** GPT, Llama, Claude: all are functions from "tokens so far" to a probability distribution over the next token, trained to minimize cross-entropy. The only differences from what you'll build this week are the tokenizer (M03), the architecture (transformer, M03–M04) and scale.

For an inference engineer this module is directly practical:

- The **sampler** (temperature, top-k, top-p, min-p, seeds) is a component of every engine — vLLM's `SamplingParams`, SGLang, TensorRT-LLM — and a frequent source of production bugs ("why is output different at temperature 0 across batch sizes?").
- **Loss / perplexity** is how you check that an optimization (quantization in M14, a new kernel in M12) didn't break the model.
- **Embedding tables** are the first and (tied to the output head) often one of the largest single weights in a model.
  */}),

  prereqs: [
    {
      title: "M01: tensors, broadcasting, the training loop",
      skipIf: "you can write a PyTorch forward/backward/step loop from memory",
      md: MD(function () {/*
You need: `tensor.shape`, broadcasting with `keepdim=True`, `F.cross_entropy(logits, targets)`, `loss.backward()`, updating `p.data -= lr * p.grad`. If any of that is fuzzy, redo M01's "tensors" and "training-loop anatomy" lessons.

```bash
mkdir -p ~/course-work/m02 && cd ~/course-work/m02
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
wc -l names.txt      # 32033
```
      */}),
    },
    {
      title: "Probability in 90 seconds",
      math: true,
      skipIf: "you know what a probability distribution and conditional probability are",
      md: MD(function () {/*
- A **probability distribution** over a set of outcomes assigns each outcome a number in $[0, 1]$, and they **sum to 1**. For 27 characters: 27 non-negative numbers summing to 1.
- **Conditional probability** $P(\text{next} = b \mid \text{current} = a)$: "given the current character is *a*, how likely is *b* next?" A language model *is* a conditional distribution $P(\text{next token} \mid \text{context})$.
- **Independent events multiply:** the probability of a whole word under a bigram model is the product of its step probabilities: $P(\text{"emma"}) = P(e \mid .) \cdot P(m \mid e) \cdot P(m \mid m) \cdot P(a \mid m) \cdot P(. \mid a)$.
- **Sampling** = drawing an outcome at random *according to* the probabilities. With $P = [0.6, 0.3, 0.1]$, over 1,000 draws you get ≈ 600 / 300 / 100 of each.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it-work",
      title: "See it work: makemore generates names (browser + one command)",
      kind: "demo",
      minutes: 45,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
## 1. In the browser

Play with the demo viz above (or here) for ten minutes before writing any code:

::viz bigram-live

- [ ] Replaced the text with something different (code, another language) and watched the heatmap and samples change
- [ ] Noticed: the loss **at init** (uniform) vs after counting; which is lower and why

## 2. The real repo, on your Mac

Karpathy's [makemore](https://github.com/karpathy/makemore) is a single-file script that trains a chain of increasingly powerful character-level models: bigram → MLP → RNN → GRU → Transformer. Run two of them now; you'll rebuild them by hand in the next lessons.

```bash
cd ~/course-work/m02
git clone https://github.com/karpathy/makemore && cd makemore
python makemore.py -i names.txt -o out-bigram --type bigram --max-steps 2000 --device mps
python makemore.py -i names.txt -o out-mlp    --type mlp    --max-steps 5000 --device mps
python makemore.py -i names.txt -o out-mlp --sample-only --device mps   # sample from the saved model
```

(makemore writes checkpoints to the `-o` dir; press Ctrl-C any time — it saves the best model by test loss.) You'll see logs like `step 1000 | loss 2.46` and every so often a list of generated names. Compare the bigram's samples to the MLP's: the MLP's names are noticeably more name-like because it sees **3 previous characters**, not 1.

## 3. What a "language model" is — precisely

A language model assigns a probability to the next token given the previous ones:

$$
P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

Chain these and you get the probability of an entire sequence. **Generation** is just: compute the distribution, sample a token, append it, repeat until an end token. For names, the vocabulary is 26 letters + `.` (a special start/end token) = 27.

| Model | Context it sees | How it gets $P$ | Lesson |
|---|---|---|---|
| Bigram (counts) | 1 previous char | count pairs, normalize | next |
| Bigram (neural) | 1 previous char | one-hot → linear → softmax, trained | "bigram-neural" |
| MLP | 3 previous chars | embeddings → hidden layer → softmax | "mlp-embeddings" |
| Transformer | up to $T$ previous tokens | attention | M03 |

> [!REAL] It's the same loop in production
> When you call ChatGPT, a server runs this exact loop: forward pass → logits for the last position → sampler → append → repeat, with a KV cache so earlier tokens aren't recomputed (M06, M08). Everything you learn about sampling here maps 1:1 to vLLM's `SamplingParams`.

- [ ] Ran makemore's bigram and MLP; noted final test losses and 5 samples each
- [ ] Read `makemore.py`'s `Bigram` class (it's ~10 lines) — it's a 27×27 lookup table of logits
      */}),
      resources: [
        { title: "karpathy/makemore", url: "https://github.com/karpathy/makemore", type: "repo", note: "the single-file reference for this module" },
      ],
    },
    {
      id: "bigram-counting",
      title: "Build a bigram model by counting (and sample from it)",
      kind: "build",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] Watch-along
> Karpathy's **["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=PaCmpygFfXo)** (~1 h 57 min). This lesson covers roughly the first hour (counting, sampling, the loss); the next math lesson and the neural-bigram lesson cover the rest. Type along in a notebook.

## 1. Data → vocabulary → integer ids

```python
import torch
words = open("names.txt").read().splitlines()          # ['emma', 'olivia', 'ava', ...]
chars = sorted(set("".join(words)))                     # 26 letters
stoi = {s: i + 1 for i, s in enumerate(chars)}; stoi["."] = 0   # '.' = start/end token
itos = {i: s for s, i in stoi.items()}
```

The integer ids are a **tokenizer** — a trivial one (1 char = 1 token). GPT's tokenizer (M03) maps chunks of text to ids from a 50k–200k vocabulary, but the idea is identical.

## 2. Count every pair

```python
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    cs = ["."] + list(w) + ["."]          # ".emma." -> pairs (.e)(em)(mm)(ma)(a.)
    for c1, c2 in zip(cs, cs[1:]):
        N[stoi[c1], stoi[c2]] += 1
N[stoi["."]].topk(3)   # most common first letters: a, k, m
```

Row $a$ of `N` is a histogram of "what comes after $a$". Plot it with `plt.imshow(N)` — it's the heatmap in the viz.

## 3. Normalize rows into probability distributions

```python
P = (N + 1).float()                  # +1 = "smoothing": no pair has probability exactly 0
P /= P.sum(1, keepdim=True)          # each ROW sums to 1   (keepdim! see M01 broadcasting)
P[0].sum()                           # tensor(1.)
```

## 4. Sample names

```python
g = torch.Generator().manual_seed(2147483647)   # reproducible randomness
for _ in range(5):
    out, ix = [], 0                              # start at '.'
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        if ix == 0:                              # sampled the end token
            break
        out.append(itos[ix])
    print("".join(out))
```

Output is name-*ish*: `mor`, `axx`, `minaymoryles`, `kondlaisah`, `anchshizarie`. Bad — but **much** better than uniform random letters. Replace `P[ix]` by `torch.ones(27) / 27` to see what "no model" looks like.

## 5. How good is it? One number.

We want a single score: *how much probability did the model assign to the real data?* Multiply the probability of every real bigram (the **likelihood**). Those products underflow to 0 immediately, so we sum **logs** instead, negate (so lower = better), and average:

```python
log_likelihood, n = 0.0, 0
for w in words:
    cs = ["."] + list(w) + ["."]
    for c1, c2 in zip(cs, cs[1:]):
        log_likelihood += torch.log(P[stoi[c1], stoi[c2]])
        n += 1
print(f"{n} bigrams, avg negative log-likelihood = {-log_likelihood.item() / n:.4f}")
# 228146 bigrams, avg negative log-likelihood ≈ 2.45
```

The next lesson explains exactly why this is the right number — and why it's the same thing as the cross-entropy loss from M01.

- [ ] Built `N`, visualized it with `imshow`, found the most common bigram (it's `n.` — names ending in n)
- [ ] Sampled 10 names with a fixed seed; then with the uniform distribution for contrast
- [ ] Computed the average NLL ≈ 2.45
- [ ] Computed the NLL of your own name. Which of its bigrams is the least likely?
- [ ] Removed the `+1` smoothing and computed the NLL of `"andrejq"`. Explain the `inf`.
      */}),
      resources: [
        { title: "Karpathy — building makemore (part 1: bigram)", url: "https://www.youtube.com/watch?v=PaCmpygFfXo", type: "video", note: "the lecture this lesson and the next two follow" },
      ],
    },
    {
      id: "math-nll",
      title: "Math: likelihood → log-likelihood → negative log-likelihood = cross-entropy",
      kind: "math",
      minutes: 75,
      runsOn: ["browser"],
      md: MD(function () {/*
> [!PREREQ] Refresher: logarithms
> $\ln y$ is the power you raise $e \approx 2.718$ to, to get $y$. Three facts are all you need:
> 1. $\ln(a \cdot b) = \ln a + \ln b$ — **logs turn products into sums**.
> 2. $\ln 1 = 0$, and for $0 < p < 1$, $\ln p < 0$ (e.g. $\ln 0.5 = -0.69$, $\ln 0.01 = -4.6$).
> 3. $\ln$ is increasing: bigger $p$ → bigger $\ln p$. So maximizing $\ln(\cdot)$ is the same as maximizing $(\cdot)$.

::viz exp-log

## Step 1 — likelihood: "how probable did my model find the real data?"

For a bigram model and the word "emma":

$$
\mathcal{L} = P(e \mid .)\,P(m \mid e)\,P(m \mid m)\,P(a \mid m)\,P(. \mid a)
$$

For the whole dataset, multiply over all 228,146 bigrams. A better model gives the real data higher probability → higher likelihood. **Training = maximizing likelihood.**

Problem: each factor is ~0.1, so the product is ~$10^{-228000}$ — far below the smallest float (~$10^{-38}$ in float32). It underflows to exactly 0.

## Step 2 — log-likelihood: sum instead of multiply

$$
\ln \mathcal{L} = \sum_{\text{bigrams}} \ln P(\text{next} \mid \text{prev})
$$

Same ranking of models (fact 3), no underflow (fact 1). It's a big negative number.

## Step 3 — negative, averaged: the loss

Optimizers *minimize*, and we want a number independent of dataset size:

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \ln P(y_i \mid x_i)
$$

**Worked example.** Three predictions assign the correct next character probabilities 0.5, 0.25 and 0.1:
$-\frac{1}{3}(\ln 0.5 + \ln 0.25 + \ln 0.1) = -\frac{1}{3}(-0.693 - 1.386 - 2.303) = 1.461$.

## Step 4 — it's the cross-entropy you used in M01

The model outputs a distribution $p$; the "true" distribution for one example is one-hot on the correct token $y$. Cross-entropy $H(\text{true}, p) = -\sum_k \text{true}_k \ln p_k$ — every term with $\text{true}_k = 0$ vanishes, leaving $-\ln p_y$. **Cross-entropy with one-hot targets = negative log-likelihood.** That's why `F.cross_entropy` is *the* LLM loss.

Explore the curve: loss is 0 when $p_{\text{correct}} = 1$, and shoots to infinity as $p \to 0$:

::viz neg-log {"vocab":27}

## Three numbers to memorize

| Quantity | Formula | For names (V = 27) | For GPT-2 (V = 50,257) |
|---|---|---|---|
| Loss of a uniform (untrained) model | $\ln V$ | 3.296 | 10.82 |
| Perplexity | $e^{\text{loss}}$ | bigram: $e^{2.45} \approx 11.6$ | GPT-2 small on web text: loss ≈ 3.1 → ≈ 22 |
| Perfect model | 0 | impossible — the data is genuinely random | |

> [!INTUITION] Perplexity
> $e^{\text{loss}}$ is "the effective number of equally likely choices the model is hesitating between". The uniform model hesitates among 27; the bigram among ~11.6; the MLP (next lessons) among ~8.8. When papers report perplexity, they're reporting exponentiated cross-entropy.

> [!REAL] Loss as a regression test
> When you quantize a model (M14) or write a new attention kernel (M12), the first check is: **does the loss/perplexity on a fixed eval set stay the same** (within ~0.01)? A broken kernel that produces plausible-looking text will still show up as a perplexity jump. Keep an eval-loss script in your toolbox from now on.

- [ ] Verified the worked example (1.461) in Python with `torch.log`
- [ ] Explained why we never multiply raw probabilities in code
- [ ] Computed $\ln 27$, $\ln 65$, $\ln 50257$ — the init losses for names, Shakespeare chars (M03), GPT-2
- [ ] In the viz, read off the loss when the model gives the correct token 1%, 10%, 50% probability
      */}),
    },
    {
      id: "bigram-neural",
      title: "The same bigram, learned by gradient descent (one-hot → logits → softmax)",
      kind: "build",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
Counting only works when the context is tiny (27 rows). With 3 characters of context you'd need $27^3 \approx 20$k rows; with 1,000 tokens of context, more rows than atoms in the universe. The fix: a **neural network** that *computes* the distribution from the context, trained by minimizing the NLL. We start by re-deriving the bigram model this way — same answer, but now the method scales.

## 1. Training set: (current char → next char) pairs

```python
import torch, torch.nn.functional as F
xs, ys = [], []
for w in words:
    cs = ["."] + list(w) + ["."]
    for c1, c2 in zip(cs, cs[1:]):
        xs.append(stoi[c1]); ys.append(stoi[c2])
xs, ys = torch.tensor(xs), torch.tensor(ys)      # 228,146 examples each
```

## 2. Model: one-hot → linear layer → logits → softmax

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
    xenc = F.one_hot(xs, num_classes=27).float()      # (228146, 27)
    logits = xenc @ W                                 # (228146, 27) "log-counts"
    counts = logits.exp()                             # positive "counts"
    probs = counts / counts.sum(1, keepdim=True)     # softmax, by hand
    loss = -probs[torch.arange(len(ys)), ys].log().mean() + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad                            # big LR is fine for this convex problem
    if k % 20 == 0: print(k, loss.item())
# converges to ≈ 2.48 — the counting model's 2.45 plus the regularization term
```

## 3. What just happened (the key insight)

- `F.one_hot(x) @ W` just **selects row `x` of `W`**. So `W` *is* a table of 27×27 logits — the neural net learned the log of the count table. After training, `W.exp()` normalized by rows ≈ the smoothed `P` from counting.
- The `0.01 * (W**2).mean()` term (**L2 regularization**) pulls logits toward 0 = toward uniform — exactly what the `+1` smoothing did for counts.
- Lines 2–4 of the loop are **softmax** + **NLL**; PyTorch fuses them into `F.cross_entropy(logits, ys)` which is faster and numerically safer (it subtracts the max logit before `exp`). Use it from now on.

```python
loss = F.cross_entropy(xenc @ W, ys)          # same number, one call
loss_fast = F.cross_entropy(W[xs], ys)       # and indexing instead of one-hot: same again
```

> [!MATH] The gradient of softmax + cross-entropy
> For one example with probabilities $p$ and correct class $y$: $\frac{\partial L}{\partial \text{logit}_k} = p_k - [k = y]$. The model pushes the correct logit up by $(1 - p_y)$ and every wrong logit down by its probability. Verify it: after `loss.backward()` on a single example, compare `logits.grad` with `probs - F.one_hot(y, 27)` (divided by the batch size, because of `.mean()`).

That one-hot-times-matrix = row lookup is the seed of **embeddings** (lesson after next): `nn.Embedding` is literally a learnable table indexed by token id.

- [ ] Trained the neural bigram to ≈ 2.48; compared `F.softmax(W, 1)` against count-based `P` for row `a`
- [ ] Replaced the manual softmax/NLL with `F.cross_entropy` and `W[xs]`; same loss
- [ ] Verified the $p - \text{onehot}$ gradient numerically on one example
- [ ] Sampled 5 names from the trained `W` with the same seed as the counting model — near-identical output
      */}),
    },
    {
      id: "softmax-sampling",
      title: "Math: softmax, temperature, top-k, top-p — the sampler inside every inference engine",
      kind: "math",
      minutes: 90,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
> [!PREREQ] Refresher: $e^x$
> $e^x$ is always positive, $e^0 = 1$, and it grows fast: $e^1 = 2.7$, $e^2 = 7.4$, $e^5 = 148$. Adding a constant to $x$ multiplies $e^x$ by a constant: $e^{x + c} = e^x e^c$.

## Softmax

$$
p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

Takes any real-valued **logits** $z$ and returns a probability distribution. Two properties matter for engines:

1. **Shift-invariant:** adding the same $c$ to every logit changes nothing ($e^c$ cancels). Implementations compute $e^{z_i - \max z}$ so nothing overflows. (This max-subtraction becomes the heart of **online softmax** and FlashAttention in M12.)
2. **Only differences matter**, and they are exponentiated: a logit gap of 2.3 means 10× more probable.

## Temperature

Divide logits by $\tau$ before softmax: $p_i \propto e^{z_i / \tau}$.

**Worked example:** logits $[2, 1, 0]$.

| $\tau$ | probabilities | behavior |
|---|---|---|
| 0.5 | [0.867, 0.117, 0.016] | sharper, more deterministic |
| 1.0 | [0.665, 0.245, 0.090] | the model's actual distribution |
| 2.0 | [0.506, 0.307, 0.186] | flatter, more random |
| $\to 0$ | [1, 0, 0] | **greedy decoding** (argmax) |

## Truncation: top-k and top-p

- **Top-k:** keep only the $k$ highest-probability tokens, renormalize, sample. Removes the long tail of garbage tokens.
- **Top-p (nucleus):** sort by probability, keep the smallest set whose cumulative probability ≥ $p$ (e.g. 0.9), renormalize. Adapts: keeps 2 tokens when the model is confident, 200 when it isn't. (Holtzman et al., "The Curious Case of Neural Text Degeneration".)
- **Min-p:** keep tokens with $p_i \ge \text{min\_p} \cdot p_{\max}$. Popular in local-LLM land; supported by vLLM and llama.cpp.

Play with all of them — watch how the sample counts change:

::viz softmax-temp {"tokens":["a","e","i","n","r","l",".","q"],"logits":[2.2,1.9,1.1,1.0,0.6,0.3,-0.5,-3.0]}

## Implement it

```python
import torch, torch.nn.functional as F

def sample(logits, temperature=1.0, top_k=None, generator=None):
    """logits: (vocab,) raw scores for the next token -> int token id"""
    if temperature == 0:
        return int(logits.argmax())                       # greedy
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))   # e^-inf = 0
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
```

Top-p is left for this module's challenge (hint: `torch.sort(probs, descending=True)`, `torch.cumsum`, mask, scatter back).

> [!REAL] This is literally what vLLM does
> In vLLM you pass `SamplingParams(temperature=0.7, top_p=0.9, top_k=50, min_p=0.0, seed=42, max_tokens=256)`. After each forward pass, the engine takes the logits of the **last position of every sequence in the batch** (a `(num_seqs, vocab)` tensor), applies penalties, divides by per-request temperatures, applies top-k/top-p/min-p masks, and samples — all batched on the GPU in the V1 engine's sampler. Things you'll meet later: why `temperature=0` isn't always bit-identical across batch sizes (floating-point reduction order changes with batch shape, M09), why sampling must be fast (it runs once per token for every request), and how **speculative decoding** (M15) must preserve exactly this distribution. Inference Engineering (Baseten) ch. 2.2 covers these sampling controls.

- [ ] Reproduced the temperature table above with `F.softmax(torch.tensor([2.,1.,0.]) / t, 0)`
- [ ] Implemented `sample()`; sampled 20 names from your bigram at $\tau$ = 0.5, 1.0, 1.5 and described the difference
- [ ] Explained why top-k = 1 and temperature → 0 give the same result
- [ ] In the viz, found settings where top-p keeps only 2 tokens and settings where it keeps all 8
      */}),
      resources: [
        { title: "Holtzman et al. — The Curious Case of Neural Text Degeneration (top-p)", url: "https://arxiv.org/abs/1904.09751", type: "paper", note: "introduced nucleus (top-p) sampling; read sections 1–3" },
        { title: "vLLM — SamplingParams", url: "https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html", type: "docs", note: "the production version of the function you just wrote" },
        { title: "Hugging Face — Generation strategies", url: "https://huggingface.co/docs/transformers/generation_strategies", type: "docs", note: "greedy, sampling, beam search and friends in Transformers" },
      ],
    },
    {
      id: "mlp-embeddings",
      title: "Build the MLP language model (Bengio 2003): embeddings + hidden layer",
      kind: "build",
      minutes: 180,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] Watch-along
> Karpathy's **["Building makemore Part 2: MLP"](https://www.youtube.com/watch?v=TCH_1BHY58I)** (~1 h 15 min), implementing Bengio et al. 2003, *[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)*. Skim the paper's Figure 1 first — it's the architecture below.

## The idea: embeddings

Give every character a small learnable vector — its **embedding** — e.g. 10 numbers. Characters that behave similarly (vowels; `.` vs rare letters) end up with similar vectors because that helps the loss. The model then works on these dense vectors instead of sparse one-hots, so what it learns about "a" transfers to "e".

::viz embeddings-2d

`C[ix]` (indexing a table) is the same as `one_hot(ix) @ C` from the previous lesson — just without materializing the one-hots. In PyTorch it's `nn.Embedding(vocab, dim)`. **Every LLM starts with exactly this lookup**; in Llama-3-8B the table is 128,256 × 4,096 ≈ 525M parameters.

## Architecture

```text
context: 3 previous chars  ->  C lookup: (3, 10)  ->  concat: (30,)
   -> Linear 30->200 + tanh  ->  Linear 200->27  ->  logits  ->  softmax / cross-entropy
```

## 1. Dataset with a sliding context window + train/dev/test split

```python
import random, torch, torch.nn.functional as F
block_size = 3                                     # context length

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size                 # start with "..."
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context); Y.append(ix)        # "..." -> e, "..e" -> m, ".em" -> m, ...
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42); random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])       # 80%: fit parameters
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%: tune hyperparameters
Xte, Yte = build_dataset(words[n2:])       # 10%: report ONCE at the very end
```

## 2. Parameters

```python
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10), generator=g)           # embedding table
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(sum(p.nelement() for p in parameters))       # 11897
```

## 3. Minibatch training loop

```python
for i in range(200_000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)   # random minibatch of 32
    emb = C[Xtr[ix]]                                   # (32, 3, 10)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)         # (32, 200)
    logits = h @ W2 + b2                               # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 100_000 else 0.01                  # step decay
    for p in parameters:
        p.data += -lr * p.grad

@torch.no_grad()
def split_loss(X, Y):
    h = torch.tanh(C[X].view(-1, 30) @ W1 + b1)
    return F.cross_entropy(h @ W2 + b2, Y).item()
print("train", split_loss(Xtr, Ytr), "dev", split_loss(Xdev, Ydev))   # ≈ 2.12 / 2.17
```

Runs in a few minutes on CPU; this model is so small that CPU is often as fast as MPS (M01 lab: overhead-bound).

> [!TIP] Finding a learning rate (Karpathy's trick)
> Sweep the LR exponentially over the first 1,000 steps (`lre = torch.linspace(-3, 0, 1000); lrs = 10**lre`), record the loss at each, and plot loss vs `lre`. Pick the LR just before the loss starts to blow up (≈ 0.1 here).

## 4. Sample (with temperature from the last lesson)

```python
for _ in range(10):
    out, context = [], [0] * block_size
    while True:
        h = torch.tanh(C[torch.tensor([context])].view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        ix = torch.multinomial(F.softmax(logits / 1.0, dim=1), 1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))
```

Typical output: `carmah`, `amelle`, `khi`, `mili`, `taty`, `skanden`, `jazhien`, `delynn`. Far more name-like than the bigram's.

## 5. Look at the embeddings

Retrain with `C` of shape `(27, 2)` (and `W1` of `(6, 200)`), then scatter-plot `C[:, 0]` vs `C[:, 1]` with each point labeled by its character. Vowels cluster together; `.` sits apart; `q` is an outlier. The model discovered "vowel-ness" on its own because it reduces the loss.

> [!REAL] Embeddings at inference time
> The embedding lookup is a **gather**, not a matmul: ~0 FLOPs, but for a big vocabulary the table is large. Many models **tie** the input embedding with the output projection (`lm_head.weight = wte.weight`, GPT-2 does this) to save parameters. You'll count these bytes with the `param-counter` viz in M03.

- [ ] Built the MLP; train ≈ 2.12, dev ≈ 2.17 (±0.03)
- [ ] Ran the LR-finder sweep and plotted it
- [ ] Trained the 2-D embedding version and plotted the characters; described one cluster
- [ ] Rewrote the model as an `nn.Module` with `nn.Embedding`, `nn.Linear`, `nn.Tanh` and `torch.optim.SGD`; same losses
      */}),
      resources: [
        { title: "Karpathy — Building makemore Part 2: MLP", url: "https://www.youtube.com/watch?v=TCH_1BHY58I", type: "video", note: "the lecture this lesson follows" },
        { title: "Bengio et al. 2003 — A Neural Probabilistic Language Model", url: "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf", type: "paper", note: "the original MLP language model with learned word embeddings" },
      ],
    },
    {
      id: "overfitting-lab",
      title: "Lab: train/dev/test, overfitting, and a hyperparameter search to beat 2.17",
      kind: "lab",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
## Why three splits?

- **Train** — the model's parameters are fit to it. Train loss always goes down with more capacity/steps.
- **Dev / validation** — you look at it to choose hyperparameters (embedding size, hidden size, LR, steps). Because *you* are fitting to it, it slowly becomes optimistic too.
- **Test** — touched once, at the end, to report an honest number.

**Overfitting** = train loss keeps dropping while dev loss stalls or rises: the model memorizes training examples instead of learning general patterns. **Underfitting** = both are high and close: the model is too small or trained too little. Explore how model size and dataset size move the two curves:

::viz overfitting

With 11,897 parameters and ~182k training examples, our MLP is **underfitting** (train 2.12 vs dev 2.17 — close together). That tells you the next move: make it *bigger*.

## The experiment grid

Change one thing at a time from the baseline and record train/dev loss in a table (`course-work/m02/LAB.md`):

| Run | Embedding dim | Context | Hidden | Steps | Params | Train | Dev |
|---|---|---|---|---|---|---|---|
| baseline | 10 | 3 | 200 | 200k | 11,897 | 2.12 | 2.17 |
| A | 2 | 3 | 200 | 200k | | | |
| B | 10 | 3 | 300 | 200k | | | |
| C | 20 | 3 | 300 | 200k | | | |
| D | 10 | 5 | 300 | 200k | | | |
| E | 10 | 3 | 1000 | 200k | | | |
| F | your best | | | 300k | | | |

```python
def make_params(emb=10, ctx=3, hidden=200, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    ps = [torch.randn((27, emb), generator=g),
          torch.randn((ctx * emb, hidden), generator=g) * (5/3) / (ctx * emb) ** 0.5,  # Kaiming-ish init for tanh
          torch.randn(hidden, generator=g) * 0.01,
          torch.randn((hidden, 27), generator=g) * 0.01,     # small output layer -> init loss ≈ ln 27
          torch.zeros(27)]
    for p in ps: p.requires_grad = True
    return ps
```

> [!NOTE] Two free improvements
> The scaled init above (from makemore part 3) fixes the "hockey-stick" first steps where the loss starts at ~27 instead of $\ln 27 = 3.3$ because the initial logits are huge and confident. Check it: print the loss at step 0 for both inits. That's M01's "loss at init" check paying off.

> [!WARNING] Don't tune on test
> Pick your best config by **dev** loss. Only then compute test loss, once. If you peek at test while tuning, it stops being a test set.

## Plot the curves

Log `(step, train_loss_on_batch)` every step and `(step, dev_loss)` every 5k steps; plot both with a log-scale y (batch losses are noisy — average every 1,000 steps). Mark where the LR decays: you'll see a visible drop.

- [ ] Filled the grid; your best config reaches dev < 2.15 (Karpathy's hint: bigger embedding + hidden, more steps; ~2.10 is achievable)
- [ ] Found one config that clearly **overfits** (train ≪ dev) — e.g. hidden 1000 with context 8 and few examples
- [ ] Compared step-0 loss with naive vs scaled init
- [ ] Reported test loss exactly once for your final model
      */}),
    },
    {
      id: "deep-makemore-3-5",
      title: "Deep dive (optional): makemore parts 3–5 — activation stats, BatchNorm, backprop ninja, WaveNet",
      kind: "deep",
      optional: true,
      minutes: 360,
      runsOn: ["mac"],
      md: MD(function () {/*
Three more lectures take this exact MLP apart at the level a model-training engineer needs. They're optional for an inference career track, but part 3 in particular pays off every time you debug a model that "trains badly".

| Lecture | Length | What you learn | Why it matters later |
|---|---|---|---|
| **[Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)** | ~1 h 55 min | why init loss was ~27; tanh saturation (histograms of activations); Kaiming init $\text{gain}/\sqrt{\text{fan\_in}}$; BatchNorm; diagnostic plots (update-to-data ratio ≈ $10^{-3}$) | LayerNorm/RMSNorm in M03–M04 solve the same problem; the diagnostics are how you debug fine-tuning (M05) |
| **[Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)** | ~1 h 55 min | backprop by hand through cross-entropy, BatchNorm, tanh, embedding — at the **tensor** level | writing backward kernels (M12), understanding what `torch.compile` / autograd generates |
| **[Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)** | ~56 min | hierarchical fusion of context (dilated-convolution-like), `nn.Module`-style layers, 3-D tensors `(B, T, C)`; dev loss ≈ 1.99 | the `(B, T, C)` shapes are exactly M03's; the bridge from MLP to transformer |

> [!INTUITION] The one idea to steal from part 3
> Every layer should keep its activations at roughly unit scale (mean 0, std ~1) — not collapsing to 0, not exploding, not saturating. Good init gets you there at step 0; normalization layers (BatchNorm → LayerNorm → RMSNorm) keep you there during training. Transformers work at depth largely because of **residual connections + normalization** (M03).

Suggested exercises if you do these:

- [ ] Part 3: plot histograms of the tanh outputs at init with the naive vs scaled init; count the % of saturated (|t| > 0.97) units
- [ ] Part 4: complete Karpathy's exercise notebook (manual backward for every tensor) and check against autograd with `torch.allclose`
- [ ] Part 5: reach dev loss < 2.0 with the WaveNet-style model
      */}),
      resources: [
        { title: "makemore Part 3: Activations & Gradients, BatchNorm", url: "https://www.youtube.com/watch?v=P6sfmUTpUmc", type: "video", note: "initialization and normalization — highest-value of the three" },
        { title: "makemore Part 4: Becoming a Backprop Ninja", url: "https://www.youtube.com/watch?v=q8SA3rM6ckI", type: "video", note: "manual tensor-level backprop" },
        { title: "makemore Part 5: Building a WaveNet", url: "https://www.youtube.com/watch?v=t3YJ5hKiMQ0", type: "video", note: "deeper hierarchical model; (B,T,C) shapes" },
      ],
    },
  ],

  challenge: {
    title: "Your own language model + a real sampler",
    md: MD(function () {/*
Pick a dataset **you** care about and train a language model on it, then give it a production-style sampler.

**Dataset ideas** (≥ 200 KB of text): your own Git commit messages, Python function names from a repo you like, Pokémon names, city names, song lyrics you own, a public-domain book from Project Gutenberg. Char-level is fine; word-level (vocab = most frequent ~5,000 words + `<unk>`) is harder and more interesting.

**Requirements**

1. Tokenizer: build `stoi`/`itos` (char or word level), with a start/end token; report vocab size $V$ and the init loss $\ln V$.
2. Model: the MLP LM (embeddings → hidden → logits) as an `nn.Module`, trained with AdamW on MPS or CPU. Split 80/10/10.
3. **Target:** dev loss at least **15% below the bigram baseline** on the same data (train a counting bigram first to get that baseline). Report perplexity too.
4. Implement `sample(logits, temperature, top_k, top_p, generator)` handling all edge cases: `temperature == 0` → greedy; `top_k` larger than vocab; `top_p = 1.0` → no-op; always keep at least one token.
5. Unit-test the sampler: with fixed logits and 100k draws, empirical frequencies match the expected truncated/renormalized distribution within ~0.5%.
6. Generate 20 samples at three settings (e.g. greedy, $\tau=0.8$ + top-p 0.9, $\tau=1.2$) and write two sentences on the quality/diversity trade-off.

Hint for top-p: `sorted_p, idx = probs.sort(descending=True)`, `cum = sorted_p.cumsum(-1)`, remove tokens where `cum - sorted_p > top_p` (this keeps the token that crosses the threshold), then scatter the mask back with `idx`.
    */}),
    checklist: [
      "Dataset, tokenizer and vocab size documented; init loss ≈ $\\ln V$ verified",
      "Counting-bigram baseline and MLP dev loss reported; MLP is ≥ 15% lower",
      "`sample()` supports temperature (incl. 0), top-k and top-p, and never returns an empty distribution",
      "Sampler unit test with ≥ 100k draws passes",
      "Samples at three sampling settings + a short written comparison",
      "Test loss reported exactly once, for the final model",
    ],
    stretch: "Batch the sampler: accept `logits` of shape `(B, V)` with per-row `temperature`, `top_k`, `top_p` tensors (like a real engine serving B different requests at once) and make it fully vectorized — no Python loop over rows. Then time it for B = 256, V = 128k on MPS.",
  },

  connects: MD(function () {/*
You now have the conceptual core of every LLM: **tokens → embeddings → a network → logits → softmax → cross-entropy for training, sampling for generation.**

- **M03** keeps all of it and replaces the MLP-over-a-fixed-window with **self-attention**, which lets every position look at all previous positions with learned, data-dependent weights. Your `block_size = 3` becomes 256; your `(32, 3, 10)` tensor becomes `(B, T, C)`.
- **M06 / M10**: the sampler you wrote becomes a component of your mini inference engine, applied to a batch of requests after each forward pass.
- **M14**: dev loss / perplexity is your quality regression test when you quantize.
- **M15**: speculative decoding must reproduce exactly the distribution your sampler defines — you'll prove it with rejection sampling.
  */}),

  interview: [
    "Why do we minimize the *negative log* likelihood instead of maximizing the likelihood directly?",
    "What is the expected cross-entropy loss of an untrained model over a vocabulary of 32,000 tokens, and why?",
    "Explain temperature, top-k and top-p. What does temperature 0 correspond to, and why might two runs at temperature 0 still differ on a GPU server?",
    "What is perplexity and how does it relate to cross-entropy?",
    "What does an embedding layer compute? Why is `one_hot(x) @ W` equivalent, and why don't we implement it that way?",
    "Your model's train loss is 1.2 and validation loss is 2.3. What's happening and what would you try?",
    "Where does the sampler run in an inference engine, what are its inputs and outputs, and why must it be batched?",
  ],

  resources: [
    { title: "Karpathy — building makemore (part 1)", url: "https://www.youtube.com/watch?v=PaCmpygFfXo", type: "video", note: "bigram model, likelihood, neural bigram" },
    { title: "Karpathy — makemore Part 2: MLP", url: "https://www.youtube.com/watch?v=TCH_1BHY58I", type: "video", note: "embeddings, MLP LM, train/dev/test, LR finding" },
    { title: "karpathy/makemore", url: "https://github.com/karpathy/makemore", type: "repo", note: "single-file bigram → MLP → RNN → Transformer reference" },
    { title: "Bengio et al. 2003 — A Neural Probabilistic Language Model", url: "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf", type: "paper", note: "the MLP LM you implement" },
    { title: "Mikolov et al. 2013 — word2vec", url: "https://arxiv.org/abs/1301.3781", type: "paper", note: "optional: embeddings as a standalone idea (king − man + woman)" },
    { title: "Holtzman et al. 2019 — nucleus (top-p) sampling", url: "https://arxiv.org/abs/1904.09751", type: "paper", note: "why pure sampling and greedy both degenerate" },
    { title: "vLLM — SamplingParams", url: "https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html", type: "docs", note: "every sampling knob a production engine exposes" },
    { title: "Inference Engineering (Baseten) — Ch. 2.2 LLM Inference Mechanics", url: "Inference%20Engineering.pdf", type: "book", note: "tokens, sampling controls, prefill/decode framing" },
  ],
});
