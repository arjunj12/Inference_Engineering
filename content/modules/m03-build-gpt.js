Course.module({
  id: "m03-build-gpt",
  title: "Build GPT from scratch",
  short: "Build GPT from scratch",
  tagline: "Write a ~10M-parameter GPT yourself — tokens, batches, self-attention, transformer blocks — train it on your Mac GPU until it writes Shakespeare, and understand every tensor an inference engine will later have to move.",
  hours: 30,
  level: "core",
  runsOn: ["mac", "colab"],
  tags: ["transformer", "self-attention", "gpt", "nanogpt", "karpathy", "mps"],

  goal: MD(function () {/*
Your own `gpt.py` (~200 lines, no libraries beyond PyTorch), trained on 1 MB of Shakespeare on your Mac's GPU:

```text
10.74 M parameters   device: mps
step    0: train 4.2849, val 4.2823      (ln 65 = 4.17 ✓)
step 1000: train 1.9571, val 2.0526
step 2500: train 1.5905, val 1.7688
step 5000: train 1.4062, val 1.6121      (≈ 20–30 min on an M-series Pro/Max)

KING RICHARD III:
Why, then, the heavens be your kingdom, sir,
And well contented with the wind of Rome.
What say'st thou, gentle Clarence? speak, and tell me
Whose blood this hath been, that we should be his friend.
```

It isn't Shakespeare — but it invented the play format, character names, iambic-ish rhythm and archaic words from raw characters, with an architecture that is *the same* as GPT-2/3's, only smaller. You'll also know its **parameter count by hand (10,788,929 for Karpathy's config)**, its FLOPs per token, and why its `generate()` loop gets slower as the text gets longer — the problem the KV cache solves in M06/M08.

Hover a token to see what it attends to — this is the mechanism you'll implement:
  */}),
  demo: { viz: "attention" },

  why: MD(function () {/*
This is the centerpiece of Phase 1. **Every inference-engine concept is defined in terms of this model's tensors:**

| Later concept | What it is in *your* `gpt.py` |
|---|---|
| Prefill (M06) | one forward pass over all prompt tokens at once — `(B, T, C)` with big `T` |
| Decode (M06) | the `generate()` loop, one new token per forward pass |
| KV cache (M06, M08) | saving the `k` and `v` tensors of each `Head` instead of recomputing them |
| FlashAttention (M12) | a fused kernel for `softmax(q @ k.T / sqrt(d)) @ v` |
| Tensor parallelism (M16) | splitting the heads (and FFN columns) across GPUs |
| Quantization (M14) | storing the `nn.Linear` weights in 8/4 bits |

Hiring loops for inference and ML-systems roles very commonly include "implement attention / a transformer block / `generate()` from scratch" and "count the parameters and FLOPs of this model". After this module you can.
  */}),

  prereqs: [
    {
      title: "M01–M02: training loop, cross-entropy, embeddings, sampling",
      skipIf: "you've built makemore's MLP and can explain softmax + cross-entropy",
      md: MD(function () {/*
You need: `nn.Module`, `nn.Embedding`, `nn.Linear`, `F.cross_entropy`, `torch.multinomial`, and the train/val idea. The new model is makemore's MLP with the fixed 3-character window replaced by attention over up to 256 characters.

```bash
mkdir -p ~/course-work/m03 && cd ~/course-work/m03
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c input.txt       # 1115394 characters
```
      */}),
    },
    {
      title: "Batched matmul and transpose of the last two dims",
      math: true,
      skipIf: "you know what `(B, T, C) @ (B, C, T)` returns",
      md: MD(function () {/*
For tensors with more than 2 dims, `@` multiplies the **last two** dims and broadcasts the rest: `(B, T, C) @ (B, C, T)` → `(B, T, T)` — B independent `(T×C)(C×T)` matmuls. `k.transpose(-2, -1)` swaps the last two dims: `(B, T, C)` → `(B, C, T)`. So `q @ k.transpose(-2, -1)` gives, for every batch element, a `T × T` table of dot products between every query position and every key position. That table is the heart of attention.

```python
import torch
q = torch.randn(4, 8, 16); k = torch.randn(4, 8, 16)
(q @ k.transpose(-2, -1)).shape      # torch.Size([4, 8, 8])
```
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it-work",
      title: "See it work: train Karpathy's GPT on your Mac, then map the video",
      kind: "demo",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
## 1. Run the finished thing first

The companion repo of the lecture, [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture), has two scripts: `bigram.py` (the baseline) and `gpt.py` (the final transformer). Both hard-code `device = 'cuda' if torch.cuda.is_available() else 'cpu'` — patch them to use your Mac GPU:

```bash
cd ~/course-work/m03
git clone https://github.com/karpathy/ng-video-lecture && cd ng-video-lecture
# point the device line at MPS (macOS sed needs the '' after -i)
sed -i '' "s/device = 'cuda' if torch.cuda.is_available() else 'cpu'/device = 'mps' if torch.backends.mps.is_available() else 'cpu'/" bigram.py gpt.py
grep -n "device =" bigram.py gpt.py
python bigram.py        # ~1 min: val loss ≈ 2.5, then gibberish
```

Now the transformer. Karpathy's config (`batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2, max_iters=5000`) took ~15 min on an A100 to reach **val 1.48**. On a Mac that's 1–2+ hours, so first run a **Mac-sized** config by editing the hyperparameters at the top of `gpt.py`:

| Config | Edits to `gpt.py` | Params | Tokens seen | Rough time on M-series (fp32, MPS) | Expected val loss |
|---|---|---|---|---|---|
| **Mac-quick** | `batch_size=32, block_size=128, n_embd=256, n_head=4, n_layer=4, max_iters=3000, eval_iters=50` | ~3.2M | 12M | ~3–8 min | ~1.75 |
| **Mac-small** (the goal box) | `batch_size=32, block_size=128, max_iters=5000, eval_iters=100` (keep 384/6/6) | ~10.7M | 20M | ~15–30 min (Pro/Max); longer on base chips | ~1.6 |
| **Karpathy full** | unchanged | 10.8M | 82M | ~1–2 h (Max/Ultra faster) — or ~15 min on a Colab A100 / ~30–45 min on a T4 | ~1.48 |

These are estimates; your first job is to measure: time the first 100 iterations and extrapolate. Plug in your laptop and close heavy apps.

```bash
time python gpt.py | tee run-mac-small.log
```

> [!TIP] Why it's slower than an A100
> Training FLOPs ≈ $6 \times \text{params} \times \text{tokens}$ (lesson "param-count"). Karpathy full: $6 \times 10.8\text{M} \times 82\text{M} \approx 5.3 \times 10^{15}$ FLOPs. An A100 sustains maybe ~50–100 TFLOPS on this small model (with TF32); an M-series GPU in fp32 through MPS sustains a few TFLOPS. Same math, 10–30× less throughput.

## 2. The video: your map for this module

Karpathy's **["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (~1 h 56 min) is the spine of this module. Timestamps below are approximate; click to jump.

| Time | Chapter | Lesson here |
|---|---|---|
| [0:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=0s) | Intro: ChatGPT, Transformers, nanoGPT, Shakespeare | this one |
| [7:52](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=472s) | Reading and exploring the data | tokens-batches |
| [9:28](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=568s) | Tokenization, train/val split | tokens-batches |
| [14:27](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=867s) | Data loader: batches of chunks of data | tokens-batches |
| [22:11](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1331s) | Simplest baseline: bigram language model, loss, generation | tokens-batches |
| [34:53](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2093s) | Training the bigram model; porting to a script | tokens-batches |
| [42:13](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s) | Self-attention v1: averaging past context with for loops | averaging-trick |
| [47:11](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2831s) | **The mathematical trick**: matrix multiply as weighted aggregation | averaging-trick |
| [51:54](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3114s) | v2 matmul, v3 softmax, code cleanup | averaging-trick |
| [1:00:18](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3618s) | Positional encoding | tokens-batches / build |
| [1:02:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) | **The crux: v4 self-attention** (queries, keys, values) | math-attention |
| [1:11:38](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4298s) | Notes: attention as communication, no notion of space, no cross-batch communication, encoder vs decoder, self vs cross | math-attention |
| [1:16:56](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s) | **Scaled** attention: why divide by $\sqrt{d_k}$ | math-attention |
| [1:19:11](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4751s) | Inserting a single self-attention head into the network | build-gpt |
| [1:21:59](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s) | Multi-head self-attention | transformer-block |
| [1:24:25](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s) | Feed-forward layers | transformer-block |
| [1:26:48](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s) | Residual connections | transformer-block |
| [1:32:51](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s) | LayerNorm (and its relation to BatchNorm) | transformer-block |
| [1:37:49](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5869s) | Scaling up the model; dropout | build-gpt / lab |
| [1:42:39](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6159s) | Encoder vs decoder vs both | math-attention notes |
| [1:46:22](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6382s) | Quick walkthrough of nanoGPT; batched multi-head attention | deep-nanogpt |
| [1:48:53](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6533s) | Back to ChatGPT: pretraining vs fine-tuning, RLHF | deep-nanogpt; M05, M22 |
| [1:54:32](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6872s) | Conclusions | |

> [!IMPORTANT] How to watch
> Watch the first 42 minutes in one sitting, typing along. Then do the next lessons *interleaved* with the video: watch a chapter, do the matching lesson, write the code yourself. Budget ~2× the video length for typing and experimenting.

- [ ] `bigram.py` ran on MPS; val loss ≈ 2.5
- [ ] Mac-quick config trained; recorded time and final val loss
- [ ] Mac-small (or full) config trained; saved the log and a 500-character sample
- [ ] Watched the intro (to ~7:52) and skimmed all of `gpt.py` — mark every line you don't understand yet
      */}),
      resources: [
        { title: "Karpathy — Let's build GPT: from scratch, in code, spelled out", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", type: "video", note: "the lecture this module is built around" },
        { title: "karpathy/ng-video-lecture", url: "https://github.com/karpathy/ng-video-lecture", type: "repo", note: "`bigram.py` and `gpt.py` from the video" },
      ],
    },
    {
      id: "tokens-batches",
      title: "Tokenization, batching (B, T) and the bigram baseline",
      kind: "concept",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
Video: [7:52 → 42:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=472s). Start a fresh file `gpt.py` in `~/course-work/m03` — you'll grow it through the module.

## 1. Tokenize: characters → integers

```python
import torch, torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); vocab_size = len(chars)        # 65: \n, space, punctuation, A-Z, a-z
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(encode("hii there"), decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)       # (1115394,)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]                 # first 90% / last 10%
```

> [!NOTE] Real tokenizers
> Character-level gives a tiny vocabulary (65) but long sequences. GPT-2 uses **byte-pair encoding (BPE)** with 50,257 tokens: ~4 characters per token, so the same text is ~4× shorter — which directly means ~4× fewer decode steps at inference time and more text per context window. The trade-off: a bigger embedding table and output layer. Try it:

::viz tokenizer-bpe

```python
# optional: compare with GPT-2's tokenizer
# uv pip install tiktoken
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(len(enc.encode(text)), "GPT-2 tokens vs", len(text), "characters")   # ≈ 338k vs 1.1M
```

Karpathy has a full lecture on this: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) and repo [minbpe](https://github.com/karpathy/minbpe) — recommended before M04.

## 2. Batches: B independent chunks of length T

We never feed the whole text. We sample random windows of `block_size` (= **T**, the context length) tokens. **One window contains T training examples** — predict token 2 from token 1, token 3 from tokens 1–2, …, token T+1 from tokens 1..T:

```python
block_size, batch_size = 8, 4
x = train_data[:block_size]; y = train_data[1:block_size + 1]
for t in range(block_size):
    print(f"context {x[:t + 1].tolist()} -> target {y[t].item()}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))        # B random start offsets
    x = torch.stack([d[i:i + block_size] for i in ix])            # (B, T)
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])    # (B, T) — shifted by one
    return x, y
xb, yb = get_batch("train"); print(xb.shape, yb.shape)            # (4, 8) (4, 8)
```

The targets are the inputs **shifted left by one**. Training on all T positions at once is why transformers train efficiently — and it's also why **prefill** at inference time processes a whole prompt in one parallel forward pass (M06).

> [!IMPORTANT] Shape vocabulary for the rest of the course
> **B** = batch (independent sequences), **T** = time / sequence length (tokens), **C** = channels (embedding size, `n_embd`), **V** = vocab size. Engines use the same letters: vLLM flattens B×T into one "num_tokens" dimension, but the idea is identical.

## 3. The bigram baseline

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # row i = logits after token i
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)                            # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T)) # cross_entropy wants (N, V)
        return logits, loss
    def generate(self, idx, max_new_tokens):                                # idx: (B, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)                     # last position only
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)      # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb); print(loss.item())        # ≈ 4.8 at init (expected ln 65 = 4.17 — randn init is a bit overconfident)
```

Train it with AdamW (`lr=1e-3`, batch 32, 10k steps) and it reaches ≈ **2.5** — the same bigram model as M02, on a new dataset. Notice that `generate()` feeds the **whole** history to a model that only uses the last token: wasteful now, but it's the general interface every model will use.

- [ ] Encoded/decoded a string; printed vocab size (65) and the first 100 token ids
- [ ] Printed the 8 (context → target) examples in one window; explained why one window = T examples
- [ ] Trained the bigram to ≈ 2.5 on MPS; generated 300 characters
- [ ] (Optional) Compared char count vs GPT-2 token count for `input.txt` with tiktoken
      */}),
      resources: [
        { title: "Karpathy — Let's build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE", type: "video", note: "BPE from scratch; do it before M04" },
        { title: "karpathy/minbpe", url: "https://github.com/karpathy/minbpe", type: "repo", note: "minimal BPE implementation matching GPT-2/GPT-4 tokenizers" },
      ],
    },
    {
      id: "averaging-trick",
      title: "The mathematical trick: weighted averages of the past with one matmul",
      kind: "concept",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
Video: [42:13 → 1:02:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2533s). The bigram model predicts from one token. We want each position to **gather information from all previous positions** — but never from the future (that would be cheating: the future is the target).

The simplest aggregation: position $t$ takes the **average** of the vectors at positions $1..t$. Karpathy builds it three ways; all give the same result.

## Version 1 — for loops (clear, slow)

```python
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
xbow = torch.zeros(B, T, C)                   # "bag of words"
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, :t + 1].mean(0)     # average of positions 0..t
```

## Version 2 — a lower-triangular matrix multiply (fast)

Row $t$ of a matrix $W$ holds the weights position $t$ gives to each earlier position. For a uniform average: row $t$ = $\frac{1}{t+1}$ in columns $0..t$, 0 after.

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)          # rows sum to 1
xbow2 = wei @ x                               # (T, T) @ (B, T, C) -> (B, T, C)
torch.allclose(xbow, xbow2)                   # True
```

$$
\begin{bmatrix} 1 & 0 & 0 \\ \tfrac12 & \tfrac12 & 0 \\ \tfrac13 & \tfrac13 & \tfrac13 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
=
\begin{bmatrix} x_1 \\ \tfrac{x_1 + x_2}{2} \\ \tfrac{x_1 + x_2 + x_3}{3} \end{bmatrix}
$$

## Version 3 — masking + softmax (the form attention uses)

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)                                  # "affinities" — all equal for now
wei = wei.masked_fill(tril == 0, float("-inf"))          # the future is forbidden
wei = F.softmax(wei, dim=-1)                             # e^-inf = 0; equal scores -> uniform
xbow3 = wei @ x
torch.allclose(xbow, xbow3)                              # True
```

Why this roundabout form? Because now the zeros in `wei` can be replaced by **learned, data-dependent scores**: "how interesting is token $j$ to token $t$?" Softmax turns any scores into weights that sum to 1, and the `-inf` mask guarantees the future gets exactly zero weight. That is self-attention, minus how the scores are computed (next lesson).

Step through it — flip between uniform weights and learned scores, and watch the causal mask:

::viz causal-average

> [!REAL] The causal mask at inference time
> During **decode** there is no future to mask: the new token is the last one, and it may attend to everything before it. So engines skip the mask in decode kernels, and only need it in **prefill** (where all prompt tokens are processed in parallel). FlashAttention has a `causal=True` flag that also *skips computing* the upper triangle entirely — ~2× fewer FLOPs for long prompts.

- [ ] Implemented all three versions; `allclose` is True for both comparisons
- [ ] Printed `wei` for T=4 in version 3 and explained each row
- [ ] Replaced the zeros in version 3 with `torch.randn(T, T)` before masking; confirmed rows still sum to 1 and the upper triangle is still 0
- [ ] Explained why version 2/3 is much faster than version 1 on a GPU
      */}),
    },
    {
      id: "math-attention",
      title: "Math: self-attention — queries, keys, values, √d_k and softmax",
      kind: "math",
      minutes: 120,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Video: [1:02:00 → 1:19:11](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s).

> [!PREREQ] Refresher: dot product as similarity (M01)
> $\mathbf{q}\cdot\mathbf{k} = \sum_i q_i k_i$ is large when two vectors point the same way, ~0 when unrelated, negative when opposite. Softmax (M02) turns a list of scores into weights that sum to 1, and exaggerates differences exponentially.

## Q, K, V — the idea

Every token emits three vectors, each a learned linear projection of its embedding $x$:

- **Query** $q = xW_Q$ — "what am I looking for?"
- **Key** $k = xW_K$ — "what do I contain?"
- **Value** $v = xW_V$ — "what will I give you if you attend to me?"

Token $t$'s affinity for earlier token $j$ is $q_t \cdot k_j$. Mask the future, softmax the scores into weights, and output the weighted sum of values:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

where $M$ is the causal mask (0 on and below the diagonal, $-\infty$ above) and $d_k$ is the head size.

## Worked example (d_k = 2, 3 tokens)

$q_3 = [1, 0]$ (token 3's query); keys $k_1 = [1, 0]$, $k_2 = [0, 1]$, $k_3 = [1, 1]$; values $v_1 = [1, 0]$, $v_2 = [0, 1]$, $v_3 = [1, 1]$.

1. Scores $q_3 \cdot k_j$: $[1, 0, 1]$.
2. Divide by $\sqrt{2} = 1.414$: $[0.707, 0, 0.707]$.
3. Softmax: $e^{0.707} = 2.03$, $e^0 = 1$ → $[2.03, 1, 2.03] / 5.06 = [0.40, 0.20, 0.40]$.
4. Output: $0.40 \cdot [1, 0] + 0.20 \cdot [0, 1] + 0.40 \cdot [1, 1] = [0.80, 0.60]$.

Token 3 mostly listened to tokens 1 and 3, whose keys matched its query.

## Why divide by $\sqrt{d_k}$? (variance intuition)

> [!MATH] Variance in one line
> Variance measures spread: $\text{Var}(X) = \mathbb{E}[(X - \text{mean})^2]$; standard deviation = $\sqrt{\text{Var}}$. Two facts: variances of **independent** things **add**, and the product of two independent mean-0 variance-1 numbers has variance 1.

If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ terms each with variance 1, so $\text{Var}(q\cdot k) = d_k$ and its standard deviation is $\sqrt{d_k}$. With $d_k = 64$, raw scores are typically ±8. Softmax of scores that spread is nearly **one-hot**: e.g. $\text{softmax}([8, 0, -8]) \approx [0.9997, 0.0003, 0.0000]$. The token attends to just one other token, and the gradient through a saturated softmax is ~0 — it stops learning. Dividing by $\sqrt{d_k}$ brings the variance back to 1, so softmax starts out diffuse and trainable.

```python
import torch
hs = 64
q, k = torch.randn(4, 8, hs), torch.randn(4, 8, hs)
print((q @ k.transpose(-2, -1)).var().item())                 # ≈ 64
print((q @ k.transpose(-2, -1) * hs ** -0.5).var().item())    # ≈ 1
```

## Play with it

Toggle the causal mask and the $\sqrt{d}$ scaling, hover a query token to see where its attention goes, and switch heads:

::viz attention

## Karpathy's six notes (1:11:38 → 1:19:11), condensed

1. **Attention is communication**: a directed graph where each node aggregates info from nodes that point to it (here: all earlier nodes).
2. **No notion of space**: attention treats its inputs as a *set*. Order only exists because we add **positional embeddings** to $x$.
3. **Batch elements never talk**: `(B, T, T)` scores — each batch row has its own independent graph. (That's what lets an engine batch unrelated users together safely.)
4. **Decoder vs encoder**: the causal mask makes it a *decoder* (autoregressive). Remove the mask (e.g. sentiment classification, BERT) and all tokens talk to all → *encoder*.
5. **Self- vs cross-attention**: self = Q, K, V from the same sequence. Cross = Q from one sequence, K/V from another (e.g. text encoder → image model, translation).
6. **Scaled** attention: the $\sqrt{d_k}$ above.

## One head, in code

```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # not a parameter
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                           # x: (B, T, C)
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)         # each (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5         # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v                                              # (B, T, hs)
```

> [!REAL] Where the KV cache comes from
> Look at `k` and `v`: for token $j$ they depend only on $x_j$ (and, in deeper layers, on earlier tokens) — **never on later tokens**. So once computed, they never change. During generation, instead of recomputing `k, v` for the whole history every step, you can **cache** them and compute only the new token's `q, k, v`. That's the KV cache (M06, M08); its size is $2 \times \text{layers} \times \text{heads} \times d_k \times \text{bytes}$ per token — the number that dominates inference memory. PyTorch's fused version of this head is `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which dispatches to FlashAttention-style kernels on CUDA.

- [ ] Reproduced the worked example by hand, then in PyTorch
- [ ] Ran the variance snippet; then computed `softmax` of one row with and without scaling and compared how peaked it is
- [ ] Implemented `Head`; verified output shape `(B, T, head_size)` and that changing `x[:, -1]` doesn't change outputs at earlier positions (causality test)
- [ ] Checked your `Head` against `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with `allclose`
- [ ] Explained notes 2 and 3 in your own words
      */}),
      resources: [
        { title: "Vaswani et al. 2017 — Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", type: "paper", note: "read §3.2 (scaled dot-product and multi-head attention) now" },
        { title: "The Illustrated Transformer (Jay Alammar)", url: "https://jalammar.github.io/illustrated-transformer/", type: "article", note: "visual walk-through of Q/K/V and multi-head" },
      ],
    },
    {
      id: "transformer-block",
      title: "Multi-head + feed-forward + residual + LayerNorm = a transformer block",
      kind: "concept",
      minutes: 120,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Video: [1:19:11 → 1:42:39](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4751s). One attention head gets you from val ≈ 2.5 to ≈ 2.4. Four more ideas get you to 1.48. Click through the block and watch the shapes:

::viz transformer-block

## 1. Multi-head attention — several conversations in parallel

Instead of one head of size 384, run 6 heads of size 64 in parallel and concatenate: each head can learn a different relation (previous vowel, matching quote, the speaker's name…). Then a linear **output projection** mixes them back into the residual stream.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_head*hs) = (B, T, C)
        return self.dropout(self.proj(out))
```

Same FLOPs as one big head, more expressive. Production code doesn't loop over heads: it does **one** `(C → 3C)` matmul for Q, K, V together, then reshapes to `(B, n_head, T, hs)` so all heads run as one batched matmul (you'll see this in nanoGPT). That `n_head` dimension is exactly what **tensor parallelism** splits across GPUs (M16) and what **GQA** shrinks for a smaller KV cache (M04).

## 2. Feed-forward (MLP) — each token thinks on its own

Attention is **communication** between tokens; the FFN is **computation** per token (no mixing across T):

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
```

The 4× expansion means the FFN holds **2/3 of each block's weights** ($8C^2$ vs attention's $4C^2$). Inference Engineering (Baseten) ch. 2.2 makes the same point: FFN linear layers hold most of the weights, which is why weight-only quantization (M14) and MoE (M17) target them.

## 3. Residual connections — a gradient superhighway

```python
x = x + self.sa(self.ln1(x))      # not: x = self.sa(x)
x = x + self.ffwd(self.ln2(x))
```

Each sub-layer *adds* its contribution to a running **residual stream** instead of replacing it. Addition passes gradients through unchanged (M01: `+` routes the gradient to both inputs), so gradients reach early layers directly even in a 96-layer model. At init, blocks contribute little and the network behaves like a shallow one, then gradually "turns on" the blocks.

## 4. LayerNorm — keep every token vector at a sane scale

For each token's $C$-dim vector: subtract its mean, divide by its standard deviation, then apply a learned scale $\gamma$ and shift $\beta$:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad \mu = \frac{1}{C}\sum_i x_i,\;\; \sigma^2 = \frac{1}{C}\sum_i (x_i - \mu)^2
$$

**Worked example:** $x = [2, 4, 6, 8]$ → $\mu = 5$, $\sigma^2 = (9 + 1 + 1 + 9)/4 = 5$, $\sigma = 2.236$ → normalized $[-1.34, -0.45, 0.45, 1.34]$ (with $\gamma = 1, \beta = 0$).

Unlike BatchNorm (makemore part 3) it normalizes **across features of one token**, not across the batch — so it behaves identically at train and inference time and for batch size 1, which is exactly what serving needs. Modern LLMs use **RMSNorm** (no mean subtraction, no $\beta$ — cheaper); compare them:

::viz normalize

We use **pre-norm** (`x + sa(ln(x))`, as in GPT-2 and every modern LLM) rather than the original paper's post-norm (`ln(x + sa(x))`); pre-norm trains more stably at depth. Plus one final `ln_f` before the output head.

## 5. Dropout — regularization for training only

`nn.Dropout(0.2)` randomly zeroes 20% of activations during training (and scales the rest by 1/0.8), so the network can't rely on any single path; `model.eval()` turns it off. It matters here because 10M params on 1M characters overfits. **Inference never uses dropout**, and big pretrained LLMs often train with dropout 0.

## The block

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):                     # (B, T, C) in -> (B, T, C) out
        x = x + self.sa(self.ln1(x))          # communicate
        x = x + self.ffwd(self.ln2(x))        # compute
        return x
```

Input and output shapes are identical — so you can stack as many as you like. GPT-2 small stacks 12, Llama-3-8B 32, Llama-3-70B 80.

> [!INTUITION] What each ingredient bought in the video (tiny config, n_embd=32)
> bigram ≈ 2.49 → + 1 head ≈ 2.4 → + 4 heads ≈ 2.28 → + FFN ≈ 2.24 → + stacked blocks with residuals ≈ 2.08 → + LayerNorm ≈ 2.06 → scaled up (384/6/6, T=256) + dropout ≈ **1.48**. Approximate; you'll reproduce the trend in the build lesson.

- [ ] Computed the LayerNorm worked example with `F.layer_norm(torch.tensor([2.,4,6,8]), (4,))`
- [ ] Wrote `MultiHeadAttention`, `FeedForward`, `Block`; `Block` maps `(4, 8, 32)` → `(4, 8, 32)`
- [ ] Explained "communication vs computation" and where the residual stream is
- [ ] In the `transformer-block` viz, traced the shape at each of: embeddings, QKV, scores, FFN hidden, logits
      */}),
      resources: [
        { title: "Ba et al. 2016 — Layer Normalization", url: "https://arxiv.org/abs/1607.06450", type: "paper", note: "the normalization every transformer uses (or its RMSNorm variant)" },
        { title: "He et al. 2015 — Deep Residual Learning", url: "https://arxiv.org/abs/1512.03385", type: "paper", note: "origin of residual connections" },
      ],
    },
    {
      id: "param-count",
      title: "Math: parameter counting, shapes and FLOPs (a model card by hand)",
      kind: "math",
      minutes: 75,
      runsOn: ["browser"],
      md: MD(function () {/*
> [!PREREQ] Refresher
> `nn.Linear(a, b)` has $a \cdot b$ weights + $b$ biases. `nn.Embedding(n, d)` has $n \cdot d$. `nn.LayerNorm(d)` has $2d$ ($\gamma$ and $\beta$). A matmul $(m \times k)(k \times n)$ costs $2mkn$ FLOPs (M01).

## Count Karpathy's model exactly

$V = 65$, $C = 384$, $L = 6$ layers, $H = 6$ heads, $T = 256$.

| Component | Formula | Count |
|---|---|---|
| Token embedding | $V \cdot C$ | 24,960 |
| Position embedding | $T \cdot C$ | 98,304 |
| Per block: Q, K, V (no bias) | $3C^2$ | 442,368 |
| Per block: output proj | $C^2 + C$ | 147,840 |
| Per block: FFN | $(C \cdot 4C + 4C) + (4C \cdot C + C)$ | 1,181,568 |
| Per block: 2 LayerNorms | $4C$ | 1,536 |
| **Per block total** | $\approx 12C^2$ | **1,773,312** |
| × 6 blocks | | 10,639,872 |
| Final LayerNorm | $2C$ | 768 |
| LM head | $C \cdot V + V$ | 25,025 |
| **Total** | | **10,788,929** |

`print(sum(p.numel() for p in model.parameters()))` should print exactly this. The rule of thumb **$N \approx 12 L C^2$** (ignoring embeddings) gives $12 \cdot 6 \cdot 384^2 = 10.6$M — within 2%.

**GPT-2 small** ($L=12$, $C=768$, $V=50{,}257$, $T=1024$, head tied to the embedding): $12 \cdot 12 \cdot 768^2 = 84.9$M in blocks + $50{,}257 \cdot 768 = 38.6$M embedding + $0.8$M positions ≈ **124M**. Note how the embedding is almost a third of the model at this size — the reason big vocabularies are costly for small models.

Explore real models with presets (GPT-2, Llama-3-8B, Qwen2.5-7B, Llama-3-70B):

::viz param-counter

## FLOPs per token

Every weight participates in one multiply-add (2 FLOPs) per token in the forward pass:

$$
\text{forward FLOPs per token} \approx 2N + \underbrace{2 \cdot L \cdot T \cdot C}_{\text{attention scores + weighted sum}}
\qquad
\text{training FLOPs per token} \approx 6N
$$

(backward ≈ 2× forward, hence $6N$ — Kaplan et al. 2020.) For our model: $2N \approx 21.6$ MFLOPs/token forward; attention adds $2 \cdot 6 \cdot 256 \cdot 384 \approx 1.2$ MFLOPs at full context — ~5%. For long contexts (100k+ tokens) the attention term dominates; that's why long-context inference is an attention problem.

**Worked example — training cost:** Mac-small run = 5,000 steps × 32 × 128 tokens = 20.5M tokens × $6 \times 10.7$M ≈ $1.3 \times 10^{15}$ FLOPs. If it took 20 min (1,200 s), you sustained ~1.1 TFLOPS. Compare with the matmul TFLOPS you measured in M01 — the gap is overhead: small kernels, Python, softmax/LayerNorm/dropout which move memory but do few FLOPs.

## Memory

| What | Training (fp32 + AdamW) | Inference (bf16) |
|---|---|---|
| Weights | $4N$ bytes | $2N$ |
| Gradients | $4N$ | — |
| Adam moments | $8N$ | — |
| Activations | $\propto B \cdot T \cdot C \cdot L$ (saved for backward) | only the current layer's |
| KV cache | — | $2 \cdot L \cdot C \cdot 2\text{ bytes}$ per token per sequence |

For our model: 43 MB of fp32 weights; KV cache in bf16 = $2 \cdot 6 \cdot 384 \cdot 2 = 9{,}216$ bytes per token. For Llama-3-8B the same formula (with GQA) gives ~128 KB per token — at 8k context × 32 users, 32 GB. That's M08 in one line.

- [ ] Reproduced the 10,788,929 count on paper, then confirmed in code; also per module with `named_parameters()`
- [ ] Counted GPT-2 small by hand to within 1%
- [ ] Computed forward FLOPs/token for your Mac-small model and your sustained TFLOPS from its run time
- [ ] In `param-counter`, found what fraction of Llama-3-8B's parameters are in the FFN
      */}),
      resources: [
        { title: "Kaplan et al. 2020 — Scaling Laws for Neural Language Models", url: "https://arxiv.org/abs/2001.08361", type: "paper", note: "source of the 2N / 6N FLOP approximations (Table 1)" },
      ],
    },
    {
      id: "build-gpt",
      title: "Build: write gpt.py yourself, step by step, and train it on MPS",
      kind: "build",
      minutes: 300,
      runsOn: ["mac"],
      md: MD(function () {/*
Now assemble everything into **your own** `gpt.py`. Grow it in stages and **train after each stage** (tiny config: `n_embd=32, n_head=4, n_layer=3, block_size=8, batch_size=32`, 5k steps, `lr=1e-3` — a minute or two each) so you see what each piece buys. Only at the end switch to the Mac-small config.

| Stage | Add | Expected val (tiny config) |
|---|---|---|
| 0 | bigram (token embedding → logits) | ~2.5 |
| 1 | `n_embd` embedding + position embedding + `lm_head` + **one `Head`** | ~2.4 |
| 2 | `MultiHeadAttention` (4 heads) | ~2.3 |
| 3 | `FeedForward` | ~2.25 |
| 4 | stack `Block`s **with residuals** | ~2.1 |
| 5 | LayerNorm (pre-norm + `ln_f`) | ~2.05–2.1 |
| 6 | dropout, scale up to Mac-small | ~1.6 |

## The final file

Use `Head`, `MultiHeadAttention`, `FeedForward` and `Block` from the previous lessons (they take their sizes as arguments). Then:

```python
# gpt.py — run: python gpt.py
import time, torch, torch.nn as nn
from torch.nn import functional as F

# ---- config (Mac-small) ----
batch_size, block_size = 32, 128
max_iters, eval_interval, eval_iters = 5000, 500, 100
learning_rate = 3e-4
n_embd, n_head, n_layer, dropout = 384, 6, 6, 0.2
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)

# ---- data ----
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- model: paste Head, MultiHeadAttention, FeedForward, Block here ----

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):                 # GPT-2-style init: small weights -> init loss ≈ ln V
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):       # idx: (B, T) token ids
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))   # (B, T, C)
        x = self.ln_f(self.blocks(x))                                              # (B, T, C)
        logits = self.lm_head(x)                                                   # (B, T, V)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                 # position table only has block_size rows
            logits, _ = self(idx_cond)                      # full forward over the whole context!
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

@torch.no_grad()
def estimate_loss(model):
    model.eval(); out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model = GPT().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters   device: {device}")
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        l = estimate_loss(model)
        print(f"step {it:4d}: train {l['train']:.4f}, val {l['val']:.4f}   ({time.time() - t0:.0f}s)")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

torch.save(model.state_dict(), "gpt-shakespeare.pt")
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)     # token 0 = '\n'
print(decode(model.generate(ctx, max_new_tokens=500)[0].tolist()))
```

> [!TIP] Mac tips
> - Keep **fp32** on MPS for this module — it's the most robust path; mixed precision and `torch.compile` are CUDA-first topics for M05/M13.
> - `.item()` forces a GPU→CPU sync; calling it every step slows training. That's why we only read losses in `estimate_loss`.
> - If memory pressure is high (8 GB Macs), drop `batch_size` to 16 before shrinking the model.
> - On Colab (T4), set `device="cuda"` and Karpathy's full config; add `torch.backends.cuda.matmul.allow_tf32 = True` on A100/H100.

## Understand `generate()` — this is what inference engines optimize

Read the loop again. To produce each new token it runs the **entire** model over the **entire** context (up to `block_size` tokens) and then throws away all logits except the last position's. At step $t$ it redoes the work for all $t$ previous tokens, so generating $n$ tokens costs about $n^2/2$ token-forwards instead of $n$ — **quadratic** recomputation (until the context hits `block_size` and the window starts sliding).

```python
@torch.no_grad()
def time_generate(model, n_tokens=512):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    stamps = []
    for _ in range(n_tokens):
        idx = model.generate(idx, 1)
        if device == "mps": torch.mps.synchronize()
        stamps.append(time.perf_counter())
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    for pos in [1, 32, 64, 128, 256, 500]:
        if pos < len(gaps): print(f"token {pos:3d}: {1000 * gaps[pos]:.2f} ms")
```

On a GPU at this tiny scale, per-token time may look flat at first (the GPU is under-used, overhead dominates — M01 lab); try `block_size=1024` or run on CPU to see it clearly grow with position. Either way, the FLOPs per step grow linearly with context and total work grows quadratically.

> [!REAL] The fix, and your roadmap
> Keys and values of past tokens never change (previous lesson), so a real engine runs the prompt once (**prefill**), stores every layer's `k` and `v` in a **KV cache**, and then each **decode** step feeds *only the newest token* through the model, attending to the cached K/V. Per-step work drops from $O(t)$ token-forwards to $O(1)$ (plus an $O(t)$ attention read). You'll implement exactly this in M06 and page it like virtual memory in M08. The `generate()` you just wrote is the baseline you'll beat.

- [ ] Stage table filled in with *your* val losses at the tiny config
- [ ] Final `gpt.py` trains the Mac-small config; printed param count, loss curve and a 500-char sample saved to `course-work/m03/`
- [ ] Causality test on the full model: changing the last input token doesn't change logits at earlier positions
- [ ] Ran `time_generate`; recorded per-token latency at several positions and explained the trend
- [ ] Reloaded `gpt-shakespeare.pt` in a separate `sample.py` and generated text at temperatures 0.5, 1.0, 1.5
      */}),
      resources: [
        { title: "karpathy/ng-video-lecture — gpt.py", url: "https://github.com/karpathy/ng-video-lecture", type: "repo", note: "diff your file against it after you're done, not before" },
      ],
    },
    {
      id: "lab-experiments",
      title: "Lab: experiments — context length, heads, layers, loss curves, overfitting",
      kind: "lab",
      minutes: 240,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
Treat your `gpt.py` like a research codebase: make every hyperparameter a command-line flag, log to CSV, and run a small sweep. Use the **Mac-quick** config as the base so each run takes minutes (use Colab for the bigger ones if you like).

## 1. Make it scriptable

```python
import argparse, csv
p = argparse.ArgumentParser()
p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_embd", type=int, default=256); p.add_argument("--block_size", type=int, default=128)
p.add_argument("--dropout", type=float, default=0.1); p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--out", default="runs.csv")
args = p.parse_args()
# ...use args.* instead of constants; inside the eval branch:
# csv.writer(open(args.out, "a")).writerow([run_name, it, l["train"], l["val"], time.time() - t0])
```

## 2. The sweep

Change **one** variable at a time from the base; record final train/val loss, params, wall time and ms/iter:

| Experiment | Values | Question to answer |
|---|---|---|
| Context length `block_size` | 16, 32, 64, 128, 256 | How much does more context help a char model? Where do returns diminish? |
| Heads `n_head` (fixed `n_embd`) | 1, 2, 4, 8 | Same params — does splitting into heads help? |
| Depth `n_layer` | 1, 2, 4, 6 | Loss vs params vs time |
| Width `n_embd` | 64, 128, 256, 384 | Compare to depth at equal params |
| Dropout | 0.0, 0.1, 0.2, 0.3 | When does the train/val gap open up? |

Plot val loss vs step for each experiment family (matplotlib, one figure per family), and **val loss vs parameter count** on a log x-axis for the depth/width runs — you'll see the beginnings of a scaling law.

::viz overfitting

## 3. Things to look for (and explain in `LAB.md`)

- With `dropout=0` and a 10M model, **train loss keeps falling while val loss turns up** after a few thousand steps — 1 MB of text is small. That's overfitting; dropout and early stopping (save the best-val checkpoint) are the fixes at this scale; "more data" is the fix at LLM scale.
- Time per iteration grows with `block_size` faster than linearly once attention's $T^2$ term matters — measure it.
- Heads: 1 head of 256 vs 8 heads of 32 have **identical parameter counts** (Q/K/V are $C \times C$ total either way). Differences are purely about expressiveness.

> [!REAL] Connecting to inference
> Every knob you just turned has an inference cost. `block_size` = max context → KV cache size and attention cost. `n_layer` → sequential latency per token (layers can't run in parallel). `n_embd` → weight bytes read per decode step (the M00 law). `n_head` with fixed `n_embd` → no change in params, but it's the axis that GQA (M04) shrinks to cut KV cache memory. Architecture choices are inference choices.

- [ ] At least 12 runs logged to CSV with wall time
- [ ] Plots: val-loss curves per family, val loss vs params (log x)
- [ ] Found and plotted a clear overfitting run; identified the best-val step
- [ ] `LAB.md` with 5 findings, each tied to a number
      */}),
    },
    {
      id: "deep-nanogpt",
      title: "Deep dive (optional): nanoGPT and reproducing GPT-2 (124M)",
      kind: "deep",
      optional: true,
      minutes: 480,
      runsOn: ["colab", "cloud"],
      md: MD(function () {/*
Your `gpt.py` *is* GPT — architecturally. What separates it from GPT-2 is engineering: tokenizer, efficiency tricks, and scale. Two Karpathy resources close that gap:

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — `model.py` (~300 lines) + `train.py` (~300 lines). The video walks through it at [1:46:22](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6382s).
- **[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)** (~4 h) with repo **[build-nanogpt](https://github.com/karpathy/build-nanogpt)** — one commit per step, from an empty file to a GPT-2-quality model trained on ~10B tokens of FineWeb-Edu.

## What changes from your gpt.py

| Your `gpt.py` | nanoGPT / GPT-2 | Why |
|---|---|---|
| Char tokenizer, V = 65 | GPT-2 BPE, V = 50,257 (padded to 50,304 for speed) | ~4× shorter sequences; multiples of 64/128 suit GPU kernels |
| Loop over `Head` modules | one `c_attn` Linear (C → 3C), reshape to `(B, nh, T, hs)` | one big matmul instead of many small ones |
| Manual softmax attention | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` | FlashAttention kernel: never materializes the T×T matrix (M12) |
| ReLU FFN | GELU FFN | smoother; what GPT-2 used |
| Separate `lm_head` | **weight tying** `lm_head.weight = wte.weight` | saves 38.6M params (~30% of GPT-2 small) |
| fp32 | bf16 autocast, TF32 matmuls | 2–8× faster on A100/H100 (M05) |
| constant LR | warmup + cosine decay, grad clipping at 1.0, weight decay 0.1 | stable training at scale |
| one process | `torch.compile`, DDP across 8 GPUs, gradient accumulation to ~0.5M-token batches | throughput (M13, M16) |

## Pretraining vs fine-tuning (video 1:48:53)

What you trained is a **base model**: it continues documents. ChatGPT adds (1) supervised fine-tuning on conversations and (2) RLHF (a reward model + PPO) to make it an assistant. You'll do fine-tuning in M05 and RL post-training (PPO/GRPO) in M22 — and learn why RL training is dominated by *inference* (rollout generation).

## Cost reality check

Karpathy's GPT-2 (124M) reproduction takes roughly an hour and a half on one 8×A100 80GB node; his llm.c write-up famously priced the equivalent run at about \$20. On a single rented H100 (~\$2–3/hr) budget ~6–10 hours for the full 10B tokens — or train on 1–2B tokens in ~1–2 h and accept a slightly worse model. Not a Mac job; use Colab for the early commits (they run fine on a T4 with small batch sizes).

- [ ] Read nanoGPT's `model.py` end to end; mapped each class to yours
- [ ] Swapped your `Head` loop for the batched `c_attn` + `scaled_dot_product_attention` version; verified identical outputs (with dropout off) and measured the speedup on MPS
- [ ] (Optional, cloud) followed build-nanogpt to at least the "add DDP" commit; recorded tokens/s and MFU
- [ ] (Optional) loaded the real GPT-2 weights into nanoGPT (`GPT.from_pretrained("gpt2")`) and sampled from it
      */}),
      resources: [
        { title: "karpathy/nanoGPT", url: "https://github.com/karpathy/nanoGPT", type: "repo", note: "the production-ish version of what you built" },
        { title: "Karpathy — Let's reproduce GPT-2 (124M)", url: "https://www.youtube.com/watch?v=l8pRSuU81PU", type: "video", note: "4-hour build of a real GPT-2 training run" },
        { title: "karpathy/build-nanogpt", url: "https://github.com/karpathy/build-nanogpt", type: "repo", note: "commit-by-commit companion to the GPT-2 video" },
        { title: "Radford et al. 2019 — GPT-2 paper", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", type: "paper", note: "\"Language Models are Unsupervised Multitask Learners\"" },
      ],
    },
  ],

  challenge: {
    title: "A production-style generate(), a no-KV-cache baseline, and a model card",
    md: MD(function () {/*
Turn your trained Shakespeare GPT into something you'd hand to an inference engineer.

**1. `generate()` with a real sampler.** Signature:
`generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, seed=None, stop_ids=())`.
Batched: `prompt_ids` may be `(B, T)`. Reuse your M02 sampler logic, vectorized over the batch. Temperature 0 = greedy. Deterministic given a seed (use a `torch.Generator`).

**2. KV-cache-free baseline timing.** This is the "before" number for M06. Write `bench_generate.py` that, for your model on MPS (and CPU), measures:
- **TTFT** — time for the first forward on a prompt of 64 / 128 tokens (that's prefill);
- **per-token latency vs position** for 1,000 generated tokens (it rises until `block_size`, then plateaus as the window slides — explain why);
- **tokens/s** for batch sizes 1, 8, 32 (throughput vs latency preview of M09).

**3. Model card** (`MODEL_CARD.md`): config; exact parameter count with per-component breakdown (and the $12LC^2$ estimate); forward FLOPs/token at context 1 and at full context; training FLOPs ($6ND$) and the TFLOPS you actually sustained; weights in MB at fp32/bf16/int8; KV-cache bytes/token and for a full context; train/val loss; three samples at different sampling settings; known limitations.
    */}),
    checklist: [
      "`generate()` supports batch, temperature (incl. 0 = greedy), top-k, top-p, seed and stop tokens; same seed → identical output",
      "Unit test: with temperature 0, `generate()` equals a manual argmax loop",
      "`bench_generate.py` reports TTFT, per-token latency vs position (plotted) and tokens/s at batch 1/8/32",
      "Written explanation of why per-token latency changes with position and what a KV cache would change",
      "`MODEL_CARD.md` with exact param count (matches code), FLOPs/token, training FLOPs, sustained TFLOPS, memory at 3 precisions, KV bytes/token",
      "Val loss ≤ 1.65 (Mac-small) or ≤ 1.50 (full config on Colab)",
    ],
    stretch: "Implement a KV cache *now* as a preview of M06: make `Head.forward` accept and return cached `k`/`v`, feed only the new token each decode step (handle positions correctly), verify the logits match the no-cache version to 1e-4, and plot per-token latency for both. Bonus: replace your attention with `F.scaled_dot_product_attention` and measure the speedup.",
  },

  connects: MD(function () {/*
You built the model every later module operates on.

- **M04 (modern LLMs)** upgrades this block to Llama/Qwen style: RMSNorm (you saw it in the `normalize` viz), RoPE instead of learned positions, SwiGLU instead of ReLU, GQA to shrink the KV cache — then loads real weights and matches Hugging Face logits.
- **M05** trains and fine-tunes at scale: mixed precision, AdamW memory, LoRA.
- **M06** starts from your `generate()` — the quadratic baseline you timed — and adds the KV cache, splitting inference into **prefill** (your whole-prompt forward) and **decode** (one token at a time).
- **M07–M09**: the per-token cost you measured becomes a roofline analysis; batching many sequences' decode steps together is continuous batching.
- **M12** fuses your `softmax(q @ k.T / sqrt(d)) @ v` into a single FlashAttention kernel.
  */}),

  interview: [
    "Walk me through self-attention: what are Q, K and V, what are their shapes, and what is the output shape?",
    "Why do we scale attention scores by $\\frac{1}{\\sqrt{d_k}}$? What goes wrong without it?",
    "What does the causal mask do? Is it needed during the decode phase of inference?",
    "Count the parameters of a transformer block with hidden size $C$ and a 4× MLP. What fraction is attention vs MLP?",
    "Why do transformers need positional information? Name two ways to provide it.",
    "What's the difference between LayerNorm and BatchNorm, and why is LayerNorm preferred for LLM inference?",
    "Your `generate()` recomputes the whole context each step. What's the complexity, and how does a KV cache change it? What does the KV cache cost?",
    "Estimate the training FLOPs for a 124M-parameter model trained on 10B tokens, and how long that takes on 8 A100s at 40% MFU.",
  ],

  resources: [
    { title: "Karpathy — Let's build GPT: from scratch, in code, spelled out", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", type: "video", note: "the spine of this module" },
    { title: "karpathy/ng-video-lecture", url: "https://github.com/karpathy/ng-video-lecture", type: "repo", note: "code from the video" },
    { title: "karpathy/nanoGPT", url: "https://github.com/karpathy/nanoGPT", type: "repo", note: "the efficient, GPT-2-capable version" },
    { title: "karpathy/build-nanogpt + GPT-2 reproduction video", url: "https://github.com/karpathy/build-nanogpt", type: "repo", note: "optional deep dive; video https://www.youtube.com/watch?v=l8pRSuU81PU" },
    { title: "Vaswani et al. 2017 — Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", type: "paper", note: "the original transformer" },
    { title: "The Illustrated Transformer", url: "https://jalammar.github.io/illustrated-transformer/", type: "article", note: "visual companion" },
    { title: "The Illustrated GPT-2", url: "https://jalammar.github.io/illustrated-gpt2/", type: "article", note: "decoder-only specifics, including how generation works" },
    { title: "LLM Visualization (bbycroft)", url: "https://bbycroft.net/llm", type: "tool", note: "3-D walk through nano-GPT's exact tensors" },
    { title: "Sebastian Raschka — LLMs from Scratch", url: "https://github.com/rasbt/LLMs-from-scratch", type: "repo", note: "a second, book-length from-scratch implementation" },
    { title: "Kaplan et al. 2020 — Scaling Laws", url: "https://arxiv.org/abs/2001.08361", type: "paper", note: "FLOP accounting and loss-vs-size curves" },
    { title: "Inference Engineering (Baseten) — Ch. 2.2 (transformer blocks, attention, KV cache)", url: "Inference%20Engineering.pdf", type: "book", note: "read after this module; it frames everything you built in inference terms" },
  ],
});
