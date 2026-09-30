Course.module({
  id: "m01-neural-nets",
  title: "Neural networks from zero (micrograd → PyTorch)",
  short: "Neural nets (micrograd → PyTorch)",
  tagline: "Train a 98% MNIST classifier on your Mac GPU in 20 lines of PyTorch, then rebuild the engine underneath it — autograd included — until nothing is magic.",
  hours: 22,
  level: "core",
  runsOn: ["mac"],
  tags: ["pytorch", "autograd", "backprop", "micrograd", "mnist", "mps"],

  goal: MD(function () {/*
Two artifacts, both written by you:

**1. An MNIST digit classifier trained on your Mac's GPU (PyTorch `mps` backend):**

```text
device: mps   params: 235,146
epoch 0  train loss 0.247  test acc 96.41%   (6.8 s)
epoch 1  train loss 0.096  test acc 97.35%   (6.1 s)
epoch 2  train loss 0.064  test acc 97.72%   (6.0 s)
epoch 3  train loss 0.047  test acc 97.96%   (6.1 s)
epoch 4  train loss 0.036  test acc 98.07%   (6.0 s)
```

**2. `micrograd.py` — your own ~120-line autograd engine** that computes the same gradients PyTorch does:

```text
micrograd:  x1.grad=-1.5000  w1.grad=1.0000  x2.grad=0.5000  w2.grad=0.0000
pytorch:    x1.grad=-1.5000  w1.grad=1.0000  x2.grad=0.5000  w2.grad=0.0000
```

By the end you can explain every line of the training loop — what `loss.backward()` actually does, why `opt.zero_grad()` exists, and what the tensor shapes are at each step — which is exactly the vocabulary you need to read a transformer (M03) and, later, an inference engine's forward pass (M06+).

Step through backprop on a tiny graph right now — this *is* what `loss.backward()` does, just with billions of nodes:
  */}),
  demo: { viz: "backprop-graph" },

  why: MD(function () {/*
Inference engineering is the art of running a **forward pass** fast. You can't optimize a forward pass you can't read. Every engine you'll touch — vLLM, SGLang, TensorRT-LLM, llama.cpp — is ultimately executing `x @ W` + nonlinearity, layer after layer, on tensors with shapes like `(batch, seq, hidden)`. Interviewers for inference roles routinely ask you to reason about shapes, FLOPs and memory of a layer, and to debug a model that "trains but produces garbage". This module gives you:

- **PyTorch fluency** (tensors, devices, dtypes, broadcasting, `nn.Module`) — the lingua franca of every engine.
- **Autograd understanding** — needed for training (M05), RL (M21–M23), and to know *why inference is cheaper than training* (no backward pass, no activations to keep).
- **Debugging instincts** (loss at init, overfit one batch) that you'll reuse for every model you build or port.
  */}),

  prereqs: [
    {
      title: "Python env with PyTorch + MPS (from M00)",
      skipIf: "`python -c \"import torch; print(torch.backends.mps.is_available())\"` prints True",
      md: MD(function () {/*
```bash
source ~/ai/bin/activate                       # the uv env from M00
uv pip install torch torchvision numpy matplotlib scikit-learn jupyter
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

You want PyTorch **2.x** and `True`. If `False`: you're on an Intel Mac or an x86 Python under Rosetta — reinstall a native arm64 Python (`uv python install 3.12`).
      */}),
    },
    {
      title: "Functions, slopes and \"shape\" — the 3-minute version",
      math: true,
      skipIf: "you remember what a derivative and a matrix are",
      md: MD(function () {/*
- A **function** maps inputs to an output: $f(x) = x^2$ maps 3 → 9. A neural network is just a (very big) function with **knobs** (parameters).
- The **slope** (derivative) $f'(x)$ answers: *if I nudge $x$ up by a tiny amount $h$, how much does $f$ change?* $f'(x) \approx \frac{f(x+h) - f(x)}{h}$. For $x^2$ at $x=3$: $(9.0006 - 9)/0.0001 = 6$.
- A **vector** is a list of numbers: $[2, -1, 0.5]$ has **shape** `(3,)`. A **matrix** is a grid: shape `(rows, cols)`. A "tensor" is the same idea with any number of dimensions.
- Training = repeatedly nudging every knob in the direction that makes the error smaller. The derivative tells you which direction. That's it — the rest of this module is making that sentence precise and fast.

We'll re-teach each of these right before it's needed.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it-work",
      title: "See it work: a 20-line MNIST classifier on your Mac GPU",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
## The finished thing, first

MNIST: 70,000 grayscale 28×28 images of handwritten digits (60k train, 10k test). The task: image in, digit 0–9 out. Copy this into `mnist.py` and run it. Don't try to understand it yet — just get it working and look at the numbers.

```python
# mnist.py — the whole thing. Run: python mnist.py
import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf), batch_size=128, shuffle=True)
test = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, download=True, transform=tf), batch_size=1000)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                      nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print("device:", dev, "  params:", f"{sum(p.numel() for p in model.parameters()):,}")

for epoch in range(5):
    t0 = time.time(); model.train()
    for x, y in train:
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x), y)      # how wrong are we?
        opt.zero_grad(); loss.backward(); opt.step()   # nudge every weight
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test:
            correct += (model(x.to(dev)).argmax(1) == y.to(dev)).sum().item()
    print(f"epoch {epoch}  train loss {loss.item():.3f}  test acc {correct / 100:.2f}%   ({time.time() - t0:.1f} s)")
```

Expect ~**98%** test accuracy after 5 epochs, a few seconds per epoch on any M-series chip. The first run downloads ~12 MB into `./data`.

## Read it top-down (the map for this module)

| Line(s) | What it is | Lesson where you rebuild it |
|---|---|---|
| `dev = "mps"` | pick the Apple GPU | Tensors & devices |
| `DataLoader(...)` | yields batches `x: (128, 1, 28, 28)`, `y: (128,)` | Training-loop anatomy |
| `nn.Linear(784, 256)` | a layer: `y = x @ W.T + b` — 200,960 knobs | Vectors & matrices; Neurons & layers |
| `nn.ReLU()` | the nonlinearity that makes depth useful | Neurons, activations, loss |
| `F.cross_entropy` | turns 10 scores into one "wrongness" number | Neurons, activations, loss |
| `loss.backward()` | computes $\partial \text{loss} / \partial w$ for all 235k weights | Derivatives → Chain rule → micrograd |
| `opt.step()` | $w \leftarrow w - \text{lr} \cdot \text{grad}$ (AdamW is a smarter version) | Gradient descent |
| `model.eval()`, `torch.no_grad()` | inference mode: no graph, no gradients | Training-loop anatomy |

> [!INTUITION] The whole game in one sentence
> A neural net is a function with adjustable numbers. We measure how wrong it is (the **loss**), compute how the loss would change if we nudged each number (the **gradient**), nudge them all a little in the helpful direction, and repeat ~2,000 times.

> [!REAL] How this connects to inference
> The `with torch.no_grad(): model(x)` block at the bottom **is inference**. Everything an inference engine does is making that one line faster, cheaper, and able to serve thousands of users at once. Note it needs no `backward()`, no optimizer state and no saved activations — that's why serving a model needs far less memory than training it (you'll quantify this in M05 with the `training-memory` viz).

## Poke it

- [ ] Ran `mnist.py`; got ≥ 97.5% test accuracy
- [ ] Changed `dev` to `"cpu"` and compared seconds/epoch. (On small models, the Mac GPU may barely win or even lose — the data loader, not the math, is the bottleneck. You'll fix that in the lab.)
- [ ] Removed both `nn.ReLU()` lines. Accuracy drops to ~92%. *Why?* (Answer comes in the "Neurons" lesson.)
- [ ] Set `lr=1.0`. Watch the loss explode or stall. *Why?* (Answer: gradient descent lesson.)
- [ ] Printed `x.shape`, `y.shape`, `model(x).shape` inside the loop once (then `break`).
      */}),
      resources: [
        { title: "PyTorch — Learn the Basics", url: "https://pytorch.org/tutorials/beginner/basics/intro.html", type: "course", note: "official quickstart; skim after this lesson, it maps 1:1 to our code" },
        { title: "PyTorch MPS backend notes", url: "https://pytorch.org/docs/stable/notes/mps.html", type: "docs", note: "what the `mps` device is and how to use it" },
      ],
    },
    {
      id: "tensors",
      title: "PyTorch crash course for a Python dev: tensors, shapes, dtype, device, broadcasting, @",
      kind: "concept",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
A **tensor** is an n-dimensional array (like a NumPy `ndarray`) that (a) can live on a GPU and (b) can record the operations applied to it so gradients can be computed. Open `python` or a notebook and type along — this lesson is 80% REPL.

## 1. Creating tensors and reading their metadata

```python
import torch
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x.shape, x.dtype, x.device        # (torch.Size([2, 3]), torch.float32, device(type='cpu'))
torch.zeros(2, 3); torch.ones(4); torch.arange(6); torch.randn(3, 4)   # randn = normal(0,1)
x.ndim, x.numel()                 # 2, 6
```

Every bug you'll hit for the next month is one of three things: **wrong shape, wrong dtype, wrong device**. Print all three when confused.

| dtype | bytes | used for |
|---|---|---|
| `torch.float32` | 4 | default for training on Mac/CPU |
| `torch.float16` / `torch.bfloat16` | 2 | GPU training & inference (M05, M14) |
| `torch.int64` (`long`) | 8 | indices: token ids, class labels |
| `torch.bool` | 1 | masks (the causal mask in M03!) |

> [!WARNING] MPS quirks
> MPS has **no float64**: `torch.tensor(1.0, dtype=torch.float64, device="mps")` errors. Keep float64 on CPU (handy for gradient checking). GPU work is **asynchronous** — to time it correctly call `torch.mps.synchronize()` before reading the clock (on NVIDIA: `torch.cuda.synchronize()`).

## 2. Devices

```python
dev = "mps"
a = torch.randn(1000, 1000, device=dev)   # born on the GPU
b = torch.randn(1000, 1000).to(dev)       # copied CPU -> GPU
c = a @ b                                 # runs on the GPU
c.cpu().numpy()                           # back to CPU / NumPy
# a + torch.randn(1000, 1000)             # RuntimeError: tensors on different devices
```

Moving data between CPU and GPU costs time (on NVIDIA it crosses PCIe; on Apple Silicon memory is unified, but there's still sync overhead). Engines go to great lengths to avoid these transfers.

## 3. Indexing, views, reshape

```python
t = torch.arange(24).view(2, 3, 4)   # shape (2, 3, 4)
t[0].shape, t[:, 1].shape, t[..., -1].shape   # (3, 4), (2, 4), (2, 3)
t.view(6, 4); t.view(-1, 4)           # -1 = "infer this dim"
t.transpose(1, 2).shape               # (2, 4, 3)
t.transpose(1, 2).contiguous().view(2, 12)   # view needs contiguous memory
t.sum(dim=1).shape, t.sum(dim=1, keepdim=True).shape   # (2, 4) vs (2, 1, 4)
```

`view` never copies — it reinterprets the same memory with a new shape. You'll see `view`/`transpose` constantly in attention code (splitting heads is exactly `x.view(B, T, n_head, head_size).transpose(1, 2)`).

## 4. Broadcasting — the rule that causes silent bugs

When shapes differ, PyTorch aligns them **from the right**; a dimension of size 1 (or a missing one) is stretched to match.

```python
X = torch.randn(128, 784)   # a batch of 128 flattened images
b = torch.randn(256)
W = torch.randn(784, 256)
(X @ W + b).shape            # (128, 256): b (256,) broadcast over 128 rows

P = torch.rand(27, 27)
P / P.sum(1, keepdim=True)   # (27,27)/(27,1): each ROW sums to 1   ✅
P / P.sum(1)                 # (27,27)/(27,) -> treated as (1,27): divides COLUMNS ❌ silently wrong
```

That second bug is a famous moment in Karpathy's makemore lecture (M02). Rule of thumb: **use `keepdim=True` when you reduce and then divide.**

## 5. Matrix multiply `@` — the only operation that really matters

$(m \times k)$ `@` $(k \times n)$ → $(m \times n)$. The inner dimensions must match; they disappear. Batched: `(B, m, k) @ (B, k, n)` → `(B, m, n)` — the leading dims broadcast.

::viz matmul {"m":2,"k":3,"n":4}

```python
import time
for dev in ["cpu", "mps"]:
    a = torch.randn(4096, 4096, device=dev); b = torch.randn(4096, 4096, device=dev)
    for _ in range(3): a @ b                       # warm-up
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): c = a @ b
    if dev == "mps": torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(dev, f"{2 * 4096**3 / dt / 1e12:.2f} TFLOPS")   # a matmul is 2*m*k*n FLOPs
```

You'll typically see a few TFLOPS on the M-series GPU in fp32 (roughly 2–4 on base/Pro chips, more on Max/Ultra) vs a fraction of that on CPU. For scale: an H100 does ~989 TFLOPS in dense BF16. **The `2·m·k·n` FLOP count is the first formula of inference engineering** — you'll use it to predict prefill time in M06–M07.

## 6. Autograd in 5 lines (preview)

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (w * 2 - 10) ** 2        # PyTorch records the graph: mul -> sub -> pow
loss.backward()                  # walks it backwards
w.grad                           # tensor(-16.) = d loss / d w = 2*(2w-10)*2 at w=3
```

You'll build exactly this machinery yourself in the micrograd lesson.

- [ ] Reproduced every snippet; predicted each output **shape** before running it
- [ ] Triggered and read the error messages for: shape mismatch in `@`, device mismatch, float64 on MPS
- [ ] Ran the matmul benchmark; wrote down your CPU vs MPS TFLOPS (keep it — M07 compares it to the roofline)
- [ ] Explain in one sentence why `P / P.sum(1)` is wrong
      */}),
      resources: [
        { title: "PyTorch — Tensors tutorial", url: "https://pytorch.org/tutorials/beginner/basics/intro.html", type: "docs", note: "the \"Tensors\" chapter of Learn the Basics" },
      ],
    },
    {
      id: "math-vectors-matrices",
      title: "Math: vectors, dot products, matrices and shapes",
      kind: "math",
      minutes: 75,
      runsOn: ["browser"],
      md: MD(function () {/*
> [!PREREQ] Refresher: vectors
> A vector is an ordered list of numbers, e.g. $\mathbf{a} = [3, 1]$. Geometrically it's an arrow from the origin to the point (3, 1). Its **length** is $\lVert \mathbf{a} \rVert = \sqrt{3^2 + 1^2} \approx 3.16$. In ML, a vector is usually "a thing described by numbers": an image (784 pixel values), a word (an embedding of 384 numbers), a hidden state.

## The dot product: "how aligned are these two things?"

$$
\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \dots
\qquad\text{and also}\qquad
\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert \cos\theta
$$

**Worked example:** $[3, 1] \cdot [2, 4] = 3\cdot2 + 1\cdot4 = 10$. Lengths are $\sqrt{10}$ and $\sqrt{20}$, so $\cos\theta = 10/\sqrt{200} \approx 0.71$ → about 45° apart.

- Pointing the same way → large positive. Perpendicular → 0. Opposite → negative.
- **Cosine similarity** = dot product of the two vectors after scaling each to length 1.

Drag the arrows and watch the number:

::viz vectors-dot

> [!INTUITION] Why you care
> A **neuron** computes a dot product between its weights and the input: "how much does this input look like the pattern I'm tuned to?" In M03, **attention** is dot products between a *query* and every *key*: "how relevant is that earlier token to me?" Dot products are the atom of deep learning.

## Matrices: many dot products at once

A matrix $W$ of shape $(k \times n)$ is $n$ column-vectors of length $k$. Multiplying a row vector $\mathbf{x}$ of shape $(1 \times k)$ by $W$ gives $n$ dot products — one per column:

$$
\underbrace{\mathbf{x}}_{1\times k}\;\underbrace{W}_{k\times n} = \underbrace{\mathbf{y}}_{1\times n},
\qquad y_j = \sum_{i=1}^{k} x_i W_{ij}
$$

Stack $m$ inputs as rows of $X$ $(m \times k)$ and you get **the whole batch in one call**: $XW$ is $(m \times n)$. Output cell $(i, j)$ = row $i$ of $X$ · column $j$ of $W$.

::viz matmul {"m":3,"k":4,"n":2}

**The shape rule:** $(m \times k)(k \times n) \to (m \times n)$. Inner dims must match and vanish.

**The cost rule:** each of the $m \cdot n$ outputs needs $k$ multiplies and $k$ adds → $2mkn$ FLOPs.

## Worked example: the first MNIST layer

`nn.Linear(784, 256)` stores `weight` with shape `(256, 784)` (out × in — PyTorch's convention) and `bias` `(256,)`. It computes `x @ weight.T + bias`:

| Tensor | Shape |
|---|---|
| `x` (batch of 128 flattened images) | (128, 784) |
| `weight.T` | (784, 256) |
| `x @ weight.T` | (128, 256) |
| `+ bias` (broadcast) | (128, 256) |

Parameters: $784 \cdot 256 + 256 = 200{,}960$. FLOPs for this layer per batch: $2 \cdot 128 \cdot 784 \cdot 256 \approx 51$ MFLOPs.

```python
import torch, torch.nn as nn
lin = nn.Linear(784, 256)
x = torch.randn(128, 784)
manual = x @ lin.weight.T + lin.bias
torch.allclose(manual, lin(x), atol=1e-6)   # True — nn.Linear is just this
```

> [!REAL] From here to LLMs
> A Llama-3-8B layer has projections like `(4096 → 14336)`. During decode at batch 1, `x` is `(1, 4096)` — a **matrix-vector** product: tiny math, but every weight must be read from memory. That's the "decode is memory-bound" law from M00, now in matrix language. With batch 64, the same weights serve 64 rows — same bytes read, 64× more useful FLOPs. That is **why batching works** (M09).

- [ ] Computed $[1, 2, 3] \cdot [4, -5, 6]$ by hand (answer: 12), then with `torch.dot`
- [ ] For `nn.Linear(256, 128)` on a batch of 128: wrote the input, weight, output shapes, param count, FLOPs
- [ ] In the viz, found two vectors with dot product 0 and two with cosine similarity −1
- [ ] Explained why `(128, 784) @ (256, 784)` fails and how `.T` fixes it
      */}),
      resources: [
        { title: "3Blue1Brown — Essence of Linear Algebra", url: "https://www.3blue1brown.com/topics/linear-algebra", type: "video", note: "chapters 1–4 and 9 (dot products) are all you need now" },
      ],
    },
    {
      id: "neurons-activations-loss",
      title: "Neurons, layers, activations and the loss",
      kind: "concept",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
## A neuron

$$
\text{out} = \phi\big(\mathbf{w}\cdot\mathbf{x} + b\big)
$$

A dot product of weights and inputs (how much the input matches the neuron's pattern), plus a bias (a threshold), passed through an **activation function** $\phi$. A **layer** is many neurons reading the same input — i.e. one matmul (`nn.Linear`) followed by $\phi$ applied element-wise. A **network** is layers stacked.

## Why activations? Because stacked linear layers collapse

Two linear layers without $\phi$: $(xW_1)W_2 = x(W_1W_2) = xW'$ — still one linear function, no matter how deep. A linear model can only draw straight decision boundaries; that's why your MNIST net dropped to ~92% (≈ plain logistic regression) when you removed the ReLUs. A nonlinearity between layers lets the network bend space and compose features.

::viz activations

| Activation | Formula | Where you'll meet it |
|---|---|---|
| **ReLU** | $\max(0, x)$ | MNIST today; the original GPT FFN in M03 (Karpathy's version) |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$, range $(-1, 1)$ | micrograd, makemore MLP (M02) |
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$, range $(0, 1)$ | gates, probabilities for yes/no |
| **GELU** | smooth ReLU | GPT-2 (M03 deep dive) |
| **SiLU / Swish** | $x \cdot \text{sigmoid}(x)$ | SwiGLU in Llama/Qwen (M04) |

> [!MATH] Saturation
> Look at tanh/sigmoid far from 0: flat. Flat means slope ≈ 0, which (next lessons) means gradient ≈ 0, which means **that neuron stops learning**. ReLU is flat for $x<0$ ("dead ReLU"). Much of makemore part 3 (M02 deep dive) is about keeping pre-activations in the non-flat region.

## From 10 scores to one "wrongness" number

The last layer outputs 10 raw scores called **logits**, one per digit. Two steps turn them into a loss:

**1. Softmax** — exponentiate (makes everything positive) and normalize (makes it sum to 1):

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**2. Cross-entropy** — take the probability assigned to the **correct** class and compute $-\ln p_{\text{correct}}$.

**Worked example** (3 classes for brevity): logits $z = [2.0, 1.0, 0.1]$, correct class = 0.
$e^z = [7.39, 2.72, 1.11]$, sum $= 11.21$, so $p = [0.659, 0.242, 0.099]$. Loss $= -\ln 0.659 = 0.417$.
If the correct class were 2: loss $= -\ln 0.099 = 2.31$. Confidently wrong = big loss; confidently right → loss near 0.

```python
import torch, torch.nn.functional as F
z = torch.tensor([[2.0, 1.0, 0.1]])
F.softmax(z, dim=1)                          # tensor([[0.6590, 0.2424, 0.0986]])
F.cross_entropy(z, torch.tensor([0]))        # tensor(0.4170)
F.cross_entropy(z, torch.tensor([2]))        # tensor(2.3170)
```

> [!IMPORTANT] Loss at initialization — your first sanity check
> A freshly initialized classifier over $C$ classes should be roughly uniform: $p \approx 1/C$ for each class, so loss $\approx -\ln(1/C) = \ln C$. For MNIST: $\ln 10 = 2.303$. For a 65-character Shakespeare model (M03): $\ln 65 = 4.17$. For a 50,257-token GPT-2: $\ln 50257 = 10.8$. If your first loss is way above this, your init is broken (overconfident wrong logits). You'll use this check for the rest of your career.

We'll go deeper on *why* $-\ln p$ (likelihood, logs, probability) in M02 — for now, trust that it's a smooth number that goes down when the model gets better.

## Build it by hand

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
x = torch.randn(128, 784)                 # fake batch
W1 = torch.randn(784, 256) * 0.05; b1 = torch.zeros(256)
W2 = torch.randn(256, 10) * 0.05;  b2 = torch.zeros(10)
h = torch.relu(x @ W1 + b1)               # (128, 256)
logits = h @ W2 + b2                      # (128, 10)
y = torch.randint(0, 10, (128,))
print(F.cross_entropy(logits, y))         # ≈ 2.3 — matches ln(10)
```

- [ ] Wrote the 2-layer forward pass above and got an init loss near 2.30
- [ ] Multiplied `W2` by 10 instead of 0.05; observed the init loss balloon. Explained why (overconfident logits)
- [ ] Computed softmax + cross-entropy by hand for logits `[1, 1, 1]` (answer: $\ln 3 = 1.099$)
- [ ] In the viz, identified which activations saturate on both sides and which only on one
      */}),
      resources: [
        { title: "3Blue1Brown — But what is a neural network?", url: "https://www.youtube.com/watch?v=aircAruvnKk", type: "video", note: "the best 19-minute visual intro to neurons and layers (uses MNIST too)" },
      ],
    },
    {
      id: "math-derivatives-gd",
      title: "Math: slopes, derivatives, exp/log and gradient descent",
      kind: "math",
      minutes: 90,
      runsOn: ["browser"],
      md: MD(function () {/*
> [!PREREQ] Refresher: what a derivative is
> The derivative $f'(x)$ is the **slope of $f$ at $x$**: the rate at which the output changes per unit nudge of the input. Numerically: pick a tiny $h$ and compute $\frac{f(x+h) - f(x)}{h}$. That's it. Calculus just gives shortcuts so you don't have to nudge.

## 1. Slopes you can see

Drag $x$ and shrink $h$: the secant line (two points) converges to the tangent line (the derivative).

::viz derivative {"fn":"x²"}

The shortcuts you need — a whole deep-learning course's worth:

| $f(x)$ | $f'(x)$ | Example at $x = 2$ |
|---|---|---|
| $c$ (constant) | $0$ | 0 |
| $a x$ | $a$ | $3x \to 3$ |
| $x^n$ | $n x^{n-1}$ | $x^2 \to 4$ |
| $e^x$ | $e^x$ | 7.39 |
| $\ln x$ | $\frac{1}{x}$ | 0.5 |
| $\tanh x$ | $1 - \tanh^2 x$ | 0.071 |
| $\text{ReLU}(x)$ | 1 if $x>0$ else 0 | 1 |
| $f + g$ | $f' + g'$ | sum rule |
| $f \cdot g$ | $f' g + f g'$ | product rule |

Verify one numerically — **this is how you'll test every gradient you ever write**:

```python
import math
f = lambda x: x**3 - 3*x
x, h = 2.0, 1e-6
print((f(x + h) - f(x)) / h)   # 9.000006...
print(3 * x**2 - 3)            # 9.0  (the rule: 3x^2 - 3)
```

## 2. exp and log — they undo each other

> [!PREREQ] Refresher: logarithms
> $\ln y$ answers "$e$ to what power gives $y$?" So $\ln(e^3) = 3$ and $e^{\ln 5} = 5$. Rules: $\ln(ab) = \ln a + \ln b$ (products become sums — this is why we sum log-probabilities instead of multiplying tiny probabilities), $\ln 1 = 0$, and $\ln$ of a number in $(0,1)$ is negative.

::viz exp-log

Softmax uses $e^x$ (positive, amplifies differences); cross-entropy uses $-\ln p$. Their derivatives are beautifully simple, which is why these functions are everywhere. (Famous result you'll verify in M02: the gradient of softmax+cross-entropy w.r.t. the logits is just $p - \text{onehot}(y)$.)

## 3. More than one knob: the gradient

A loss depends on many parameters $w_1, \dots, w_n$. The **partial derivative** $\frac{\partial L}{\partial w_i}$ is the slope when you nudge only $w_i$ and hold the rest fixed. The **gradient** $\nabla L$ is the vector of all of them — it points in the direction of steepest *increase* of the loss. PyTorch stores $\frac{\partial L}{\partial w}$ in `w.grad`, same shape as `w`.

## 4. Gradient descent

To decrease the loss, step **against** the gradient:

$$
w \leftarrow w - \eta \, \frac{\partial L}{\partial w}
$$

$\eta$ (eta) is the **learning rate**. Worked example: $L(w) = (w - 3)^2$, start at $w = 0$, $\eta = 0.1$. Gradient $= 2(w-3) = -6$. New $w = 0 - 0.1 \cdot (-6) = 0.6$. Next: gradient $= -4.8$, $w = 1.08$ … converges to 3. With $\eta = 1.1$ the steps overshoot further each time and diverge.

::viz gradient-descent {"lr":0.1,"x0":2.6}

- [ ] In the viz: found a learning rate that converges smoothly, one that oscillates, one that diverges
- [ ] Started from different `x0` and landed in different minima — that's why **initialization** matters
- [ ] Ran gradient descent on $L(w) = (w-3)^2$ in 5 lines of Python for $\eta \in \{0.01, 0.1, 0.9, 1.1\}$; printed $w$ after 20 steps

```python
for lr in [0.01, 0.1, 0.9, 1.1]:
    w = 0.0
    for _ in range(20):
        w -= lr * 2 * (w - 3)
    print(lr, round(w, 4))
```

> [!INTUITION] Why not just solve for the minimum?
> For $(w-3)^2$ you could. For 235k (MNIST) or 8 billion (Llama) coupled parameters there's no closed form — but you *can* compute the gradient cheaply (next lesson: backprop costs about 2× a forward pass). So we walk downhill in tiny steps. **SGD** uses the gradient on a random mini-batch instead of the whole dataset (noisy but ~1000× cheaper per step). **Adam/AdamW** keeps running averages of gradients and their squares to adapt the step per parameter — the default for transformers.
      */}),
      resources: [
        { title: "3Blue1Brown — Essence of Calculus", url: "https://www.3blue1brown.com/topics/calculus", type: "video", note: "chapters 1–4: derivatives and the chain rule, geometrically" },
      ],
    },
    {
      id: "math-chain-rule",
      title: "Math: the chain rule on a computation graph (this IS backprop)",
      kind: "math",
      minutes: 75,
      runsOn: ["browser"],
      md: MD(function () {/*
> [!PREREQ] Refresher: composing functions
> If $y = g(x)$ and $L = f(y)$, then $L = f(g(x))$ — the output of one function feeds the next. A neural network is dozens of these compositions: matmul → add bias → ReLU → matmul → softmax → −log.

## The chain rule, intuitively

If a car goes 2× as fast as a bike, and the bike goes 4× as fast as a walker, the car goes $2 \times 4 = 8$× as fast as the walker. Rates of change **multiply** along a chain:

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
$$

## A computation graph

Write any expression as a graph of tiny operations. Example (from Karpathy's micrograd lecture):
$a = 2,\; b = -3,\; c = 10,\; e = a \cdot b = -6,\; d = e + c = 4,\; f = -2,\; L = d \cdot f = -8$.

**Backward pass** — start at the output with $\frac{dL}{dL} = 1$ and walk backwards, multiplying by each node's *local* derivative:

| Node | Local derivative | Gradient $\frac{\partial L}{\partial \cdot}$ |
|---|---|---|
| $L = d \cdot f$ | $\partial L/\partial d = f = -2$, $\partial L/\partial f = d = 4$ | $d$: −2, $f$: 4 |
| $d = e + c$ | $+$ passes the gradient through unchanged (local derivative 1) | $e$: −2, $c$: −2 |
| $e = a \cdot b$ | $\partial e/\partial a = b = -3$, $\partial e/\partial b = a = 2$ | $a$: $(-2)(-3) = 6$, $b$: $(-2)(2) = -4$ |

Sanity-check $a$ numerically: nudge $a$ to 2.001 → $L = ((2.001)(-3) + 10)(-2) = -7.994$. Change $= 0.006 / 0.001 = 6$. ✅

Step through it yourself:

::viz backprop-graph

## The two rules that make it an algorithm

1. **Each node only needs its local derivative.** `+` routes the incoming gradient to both inputs unchanged. `*` sends each input the incoming gradient times the *other* input. `tanh` multiplies by $1 - \text{out}^2$. No node needs to know about the rest of the graph.
2. **If a value is used in several places, its gradients add up** (the multivariable chain rule). E.g. in $L = a \cdot a$, both branches contribute: $\partial L/\partial a = a + a = 2a$. In code this is why we write `grad += ...`, never `grad = ...`.

Process nodes in **reverse topological order** (every node after all nodes that consume it) and one backward sweep gives the gradient for *every* parameter at a cost of roughly 2× the forward pass. That's **backpropagation** — reverse-mode automatic differentiation.

## The same thing on a neuron

$o = \tanh(x_1 w_1 + x_2 w_2 + b)$ with $x_1 = 2, w_1 = -3, x_2 = 0, w_2 = 1, b = 6.8814$:
$n = -6 + 0 + 6.8814 = 0.8814$, $o = \tanh(0.8814) = 0.7071$.
Backward: $\partial o/\partial n = 1 - 0.7071^2 = 0.5$. Then $\partial o/\partial w_1 = x_1 \cdot 0.5 = 1.0$, $\partial o/\partial x_1 = w_1 \cdot 0.5 = -1.5$, $\partial o/\partial w_2 = x_2 \cdot 0.5 = 0$, $\partial o/\partial x_2 = 0.5$.

These are exactly the numbers in this module's goal box — you're about to make code print them.

> [!INTUITION] Why $w_2$'s gradient is 0
> $x_2 = 0$, so $w_2$ multiplies nothing — nudging it can't change the output. Gradients tell you *which knobs matter right now for this input*.

- [ ] Did the $a,b,c,f$ example backward pass on paper before opening the viz; matched every number
- [ ] Computed the neuron gradients above on paper
- [ ] Explained in your own words why gradients accumulate with `+=`
- [ ] Explained why backward costs about the same order as forward (each edge visited once)
      */}),
      resources: [
        { title: "3Blue1Brown — Backpropagation, intuitively", url: "https://www.youtube.com/watch?v=Ilg3gGewQ5U", type: "video", note: "the visual version of this lesson" },
        { title: "colah — Calculus on Computational Graphs: Backpropagation", url: "https://colah.github.io/posts/2015-08-Backprop/", type: "article", note: "short classic on forward- vs reverse-mode differentiation" },
      ],
    },
    {
      id: "build-micrograd",
      title: "Build micrograd: your own autograd engine (follow Karpathy)",
      kind: "build",
      minutes: 240,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] Watch-along plan
> Watch Karpathy's **["The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0)** (~2.5 h) with an editor open. Pause and type everything — **do not copy-paste**. Reference repo: [karpathy/micrograd](https://github.com/karpathy/micrograd) (`engine.py` ≈ 100 lines, `nn.py` ≈ 60). Below is the skeleton you should end up with; use it to check yourself, not to skip the video.

## Step 1 — a `Value` that remembers where it came from

```python
# micrograd.py
import math, random

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0                   # dL/d(self), filled by backward()
        self._backward = lambda: None     # how to push my grad to my children
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad          # + routes gradient unchanged
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # * swaps the inputs
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):                  # k: plain int/float
        out = Value(self.data ** k, (self,), f"pow{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad    # d/dx e^x = e^x
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # conveniences so Python operators work in every direction
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return Value(other) * self ** -1
```

## Step 2 — `backward()`: topological sort, then apply the chain rule in reverse

```python
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)            # a node is appended after all its inputs
        build(self)
        self.grad = 1.0                   # dL/dL
        for v in reversed(topo):
            v._backward()
```

## Step 3 — check against PyTorch

```python
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)
o = (x1 * w1 + x2 * w2 + b).tanh()
o.backward()
print(f"micrograd:  x1.grad={x1.grad:.4f}  w1.grad={w1.grad:.4f}  x2.grad={x2.grad:.4f}  w2.grad={w2.grad:.4f}")

import torch
t = lambda v: torch.tensor([v], dtype=torch.float64, requires_grad=True)
X1, X2, W1, W2, B = t(2.0), t(0.0), t(-3.0), t(1.0), t(6.8813735870195432)
O = torch.tanh(X1 * W1 + X2 * W2 + B)
O.backward()
print(f"pytorch:    x1.grad={X1.grad.item():.4f}  w1.grad={W1.grad.item():.4f}  x2.grad={X2.grad.item():.4f}  w2.grad={W2.grad.item():.4f}")
```

Same numbers — because PyTorch's autograd is the same algorithm over tensors instead of scalars (plus ~2,000 hand-written derivative formulas and GPU kernels).

## Step 4 — neurons, layers, an MLP, and a training loop

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

random.seed(1337)
model = MLP(3, [4, 4, 1])                         # 41 parameters
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
for step in range(100):
    ypred = [model(x) for x in xs]                            # forward
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))   # MSE loss
    for p in model.parameters():
        p.grad = 0.0                                          # zero_grad!
    loss.backward()                                           # backward
    for p in model.parameters():
        p.data -= 0.05 * p.grad                               # SGD step
    if step % 10 == 0:
        print(step, round(loss.data, 4))
print([round(yp.data, 3) for yp in ypred])                    # ≈ [1, -1, -1, 1]
```

> [!WARNING] The bug Karpathy makes on purpose
> Delete the `p.grad = 0.0` loop and train again. Gradients from every previous step **accumulate** (remember `+=`), so each step uses a sum of stale gradients and training goes haywire. This is exactly why PyTorch makes you call `opt.zero_grad()` — and a classic interview question.

## Map micrograd → PyTorch

| micrograd | PyTorch |
|---|---|
| `Value` (a scalar) | `torch.Tensor` with `requires_grad=True` (millions of scalars at once) |
| `_prev`, `_op`, `_backward` | `tensor.grad_fn` (inspect it: `loss.grad_fn.next_functions`) |
| `loss.backward()` | `loss.backward()` — identical semantics |
| `p.grad = 0.0` | `opt.zero_grad()` |
| `p.data -= lr * p.grad` | `opt.step()` (SGD); AdamW adds momentum + per-parameter scaling |
| `Neuron/Layer/MLP` | `nn.Linear`, `nn.Sequential`, `nn.Module` |

- [ ] Watched the micrograd lecture and typed the engine yourself
- [ ] Step 3 prints identical gradients for micrograd and PyTorch
- [ ] Training loop drives the loss below 0.01; predictions ≈ `[1, -1, -1, 1]`
- [ ] Reproduced the "forgot zero_grad" bug and explained it
- [ ] (Optional) Added graph drawing with `graphviz` as in the video
      */}),
      resources: [
        { title: "Karpathy — building micrograd (video)", url: "https://www.youtube.com/watch?v=VMj-3S1tku0", type: "video", note: "the lecture this lesson follows; ~2.5 h, the best backprop explanation there is" },
        { title: "karpathy/micrograd", url: "https://github.com/karpathy/micrograd", type: "repo", note: "reference implementation + demo.ipynb (moons classifier you'll build in the challenge)" },
      ],
    },
    {
      id: "training-loop-anatomy",
      title: "Back to PyTorch: nn.Module, optim, DataLoader — anatomy of a training loop",
      kind: "build",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
Now rewrite `mnist.py` the "real" way — the structure every PyTorch codebase (including nanoGPT and vLLM's model files) uses. Build it in `mnist2.py`, step by step.

## 1. `nn.Module`: parameters + a `forward`

```python
import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

class MLP(nn.Module):
    def __init__(self, d_in=784, d_hidden=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)      # registered automatically as a submodule
        self.fc2 = nn.Linear(d_hidden, d_hidden // 2)
        self.head = nn.Linear(d_hidden // 2, d_out)

    def forward(self, x):                          # x: (B, 1, 28, 28) or (B, 28, 28)
        x = x.flatten(1)                           # (B, 784)
        x = F.relu(self.fc1(x))                    # (B, 256)
        x = F.relu(self.fc2(x))                    # (B, 128)
        return self.head(x)                        # (B, 10) logits

model = MLP()
for name, p in model.named_parameters():
    print(f"{name:12s} {tuple(p.shape)}")         # fc1.weight (256, 784) ...
```

`model(x)` calls `forward` (plus hooks). `model.parameters()` walks every registered submodule — that's how the optimizer finds all 235,146 knobs. `model.state_dict()` is an ordered dict name → tensor; it's what a checkpoint file (and later a `.safetensors` file you load into your own Llama in M04) contains.

## 2. Data: skip the slow path

The DataLoader + PIL transform pipeline in `mnist.py` runs on the CPU, one image at a time — for a model this small, the GPU mostly waits. MNIST is 47 MB as float32: put the whole thing on the GPU once.

```python
dev = "mps" if torch.backends.mps.is_available() else "cpu"
def load(train):
    ds = datasets.MNIST("data", train=train, download=True)
    X = ((ds.data.float() / 255.0) - 0.1307) / 0.3081    # (N, 28, 28), normalized
    return X.to(dev), ds.targets.to(dev)
Xtr, Ytr = load(True); Xte, Yte = load(False)
```

> [!NOTE] When to use `DataLoader`
> When data doesn't fit in memory or needs per-sample work (decoding JPEGs, tokenizing, augmentation): a `Dataset` implements `__len__` and `__getitem__`, and `DataLoader(ds, batch_size, shuffle=True, num_workers=4)` batches and parallelizes it. For LLM pretraining you'll usually memory-map one big array of token ids and slice random windows from it — exactly what M03's `get_batch` does.

## 3. The loop, annotated

```python
torch.manual_seed(0)
model = MLP().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
B = 128

@torch.no_grad()
def evaluate():
    model.eval()                                            # switch off dropout/batchnorm-train behavior
    acc = (model(Xte).argmax(1) == Yte).float().mean().item()
    model.train()
    return acc

for epoch in range(5):
    t0 = time.time()
    perm = torch.randperm(Xtr.shape[0], device=dev)         # shuffle each epoch
    for i in range(0, Xtr.shape[0], B):
        idx = perm[i:i + B]
        x, y = Xtr[idx], Ytr[idx]                           # 1. get a batch
        logits = model(x)                                   # 2. forward
        loss = F.cross_entropy(logits, y)                   # 3. loss
        opt.zero_grad(set_to_none=True)                     # 4. clear old grads
        loss.backward()                                     # 5. backward: fill p.grad
        opt.step()                                          # 6. update
    if dev == "mps": torch.mps.synchronize()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {100 * evaluate():.2f}%  ({time.time() - t0:.1f} s)")

torch.save(model.state_dict(), "mnist_mlp.pt")
```

This should be several times faster per epoch than the DataLoader version. The six numbered steps are **the** training loop; nanoGPT's is the same six steps plus gradient clipping, a learning-rate schedule, mixed precision and multi-GPU.

## 4. Swap the optimizer for plain SGD you wrote yourself

Prove there's no magic in `opt.step()`:

```python
lr = 0.1
# replace opt.zero_grad(...) / opt.step() with:
for p in model.parameters():
    p.grad = None
loss.backward()
with torch.no_grad():                    # don't record the update itself in the graph
    for p in model.parameters():
        p -= lr * p.grad
```

Plain SGD at `lr=0.1` reaches ~97–98% too, a bit slower than AdamW. AdamW keeps two extra tensors per parameter (running mean and variance of the gradient) — 2× the model size in extra memory. Remember that: it's why *training* a 7B model needs ~16 bytes/param while *serving* it needs 2 (or less).

## 5. Inference with the saved model

```python
model = MLP().to(dev)
model.load_state_dict(torch.load("mnist_mlp.pt", map_location=dev))
model.eval()
with torch.inference_mode():             # like no_grad, slightly faster
    probs = F.softmax(model(Xte[:5]), dim=-1)
print(probs.argmax(-1).tolist(), Yte[:5].tolist())
```

> [!REAL] How this connects
> `load_state_dict` → `eval()` → `inference_mode()` → forward → softmax → argmax is the skeleton of every model server. In M06 you'll see vLLM do the same, except the "argmax" becomes a **sampler**, the batch is a dynamically changing set of user requests, and the forward pass reads a KV cache.

- [ ] `mnist2.py` trains to ≥ 98% with the in-GPU data path; noted seconds/epoch vs `mnist.py`
- [ ] Printed `named_parameters()` and verified the total by hand
- [ ] Replaced AdamW with your hand-written SGD; still ≥ 97%
- [ ] Saved, reloaded, and ran inference on 5 test digits
- [ ] Explained the difference between `model.eval()` and `torch.no_grad()` (they're independent!)
      */}),
      resources: [
        { title: "PyTorch — Datasets & DataLoaders", url: "https://pytorch.org/docs/stable/data.html", type: "docs", note: "reference for Dataset/DataLoader when your data doesn't fit in memory" },
        { title: "PyTorch — Autograd mechanics", url: "https://docs.pytorch.org/docs/stable/notes/autograd.html", type: "docs", note: "how grad_fn graphs, no_grad and inference_mode work" },
      ],
    },
    {
      id: "debugging-lab",
      title: "Lab: debug like a pro — loss at init, overfit one batch, LR sweeps, timing",
      kind: "lab",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
Models fail **silently**: no exception, just a loss that's a bit worse than it should be. Karpathy's ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) is the canonical checklist. Run each experiment below on your `mnist2.py` and record results in `course-work/m01/LAB.md`.

## Experiment 1 — loss at initialization

Print the loss of the untrained model on one batch. Expect $\ln 10 \approx 2.30$ (± 0.1).

```python
with torch.no_grad():
    print(F.cross_entropy(MLP().to(dev)(Xtr[:512]), Ytr[:512]).item())
```

Now break it: multiply the final layer's weights by 20 (`model.head.weight.data *= 20`). The init loss jumps well above 2.3 — the model is **confidently wrong** at init, and the first hundreds of steps are wasted just un-learning that confidence (makemore part 3 calls this the "hockey stick" loss curve).

## Experiment 2 — overfit a single batch

Take **one** batch of 32 examples and train on it for 300 steps. A correct model + loop must drive the loss to ~0 (it can memorize 32 examples easily).

```python
x, y = Xtr[:32], Ytr[:32]
model = MLP().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(300):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item())   # should be < 0.01
```

If it can't, you have a bug (wrong labels, detached graph, missing `zero_grad`, wrong loss input shape…). **This is the single most useful debugging trick in deep learning.** Introduce each bug deliberately and watch the symptom:

| Deliberate bug | Symptom |
|---|---|
| Remove `opt.zero_grad()` | loss erratic / explodes |
| Shuffle `y` independently of `x` | train loss stuck near 2.3 on full data; single batch still memorizes (slowly) |
| `F.cross_entropy(F.softmax(logits, -1), y)` (double softmax) | trains, but slower and plateaus worse — cross_entropy already applies log-softmax |
| `x.detach()` on hidden layer output | only the head learns → accuracy ~ linear model |

## Experiment 3 — learning-rate sweep

Train 1 epoch for $\eta \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ with AdamW; plot final loss vs $\eta$ (log x-axis). You'll see a U-shape: too small = slow, too big = unstable. The right LR is the single most important hyperparameter; for AdamW on transformers it's usually around $10^{-4}$ to $10^{-3}$.

::viz gradient-descent {"lr":0.9,"x0":0.5}

## Experiment 4 — timing the GPU honestly

```python
import time
def bench(dev, B=4096, steps=50):
    m = MLP().to(dev); o = torch.optim.AdamW(m.parameters())
    x = torch.randn(B, 784, device=dev); y = torch.randint(0, 10, (B,), device=dev)
    for _ in range(5):                              # warm-up (kernel compile, allocation)
        o.zero_grad(); F.cross_entropy(m(x), y).backward(); o.step()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        o.zero_grad(); F.cross_entropy(m(x), y).backward(); o.step()
    if dev == "mps": torch.mps.synchronize()
    return steps * B / (time.perf_counter() - t0)
for dev in ["cpu", "mps"]:
    print(dev, f"{bench(dev):,.0f} samples/s")
```

Try `B` = 32, 512, 4096. At tiny batch sizes the GPU loses (kernel-launch overhead dominates); at large batch sizes it wins big. **This is the first appearance of the overhead-bound vs compute-bound distinction** that M07 and M13 formalize.

- [ ] Init loss ≈ 2.30 recorded; broke it with ×20 and recorded the new value
- [ ] Single-batch overfit reaches < 0.01; each deliberate bug reproduced with its symptom
- [ ] LR sweep plot saved
- [ ] CPU vs MPS samples/s table at three batch sizes, with one sentence explaining the crossover
      */}),
      resources: [
        { title: "Karpathy — A Recipe for Training Neural Networks", url: "http://karpathy.github.io/2019/04/25/recipe/", type: "article", note: "the debugging checklist every practitioner quotes" },
      ],
    },
  ],

  challenge: {
    title: "micrograd from memory → moons classifier, gradient-checked against PyTorch",
    md: MD(function () {/*
Close the video and the repo. In a fresh file, **rewrite micrograd from memory** — `Value` with `+ * ** exp tanh`, `backward()` with topological sort, and `Neuron / Layer / MLP`. Then extend and use it:

1. Add `relu()` to `Value` (local derivative: 1 if `out.data > 0` else 0) and let `Neuron` choose `tanh`, `relu` or linear (no activation) — the last layer should be linear.
2. Train a 2-hidden-layer MLP (e.g. `2 → 16 → 16 → 1`) on `sklearn.datasets.make_moons(n_samples=100, noise=0.1)` with labels mapped to ±1. Use a max-margin loss `relu(1 - y·score)` averaged over the batch plus a small L2 penalty $\alpha \sum w^2$, and SGD with a decaying learning rate (see micrograd's `demo.ipynb` for inspiration *after* you've tried).
3. Plot the decision boundary with matplotlib (evaluate the model on a grid).
4. **Gradient check:** build the *same* network in PyTorch (float64, copy your initial weights in), run one forward/backward on the same batch, and assert every parameter's gradient matches within `1e-6`. Also check a few gradients with finite differences $\frac{L(w+h) - L(w-h)}{2h}$.

Hints: to copy weights, build the PyTorch `nn.Linear` layers, then `with torch.no_grad(): lin.weight[j, i] = neuron_j.w[i].data`. The centered difference $\frac{f(x+h)-f(x-h)}{2h}$ is much more accurate than the one-sided one.
    */}),
    checklist: [
      "micrograd rewritten without looking; supports `+ - * / ** exp tanh relu` and reverse-mode `backward()`",
      "Moons MLP reaches ≥ 97% training accuracy; decision-boundary plot saved",
      "Every micrograd gradient matches PyTorch (float64) within `1e-6` on the same weights and batch",
      "Finite-difference check passes for at least 5 randomly chosen parameters",
      "`README.md` explains (in your words) why gradients accumulate with `+=` and why we zero them",
    ],
    stretch: "Vectorize it: write `Tensor` micrograd on top of NumPy arrays (ops: `matmul`, `add` with broadcasting, `relu`, `sum`, `log_softmax`) and train the MNIST MLP to ≥ 97% with it. You will learn why broadcasting makes backward tricky (you must `sum` gradients over broadcast dimensions).",
  },

  connects: MD(function () {/*
You now own the three primitives everything else is built from: **tensors with shapes**, **matmul + nonlinearity layers**, and **autograd + an optimizer loop**.

- **M02** reuses this loop to train *language models*: the input becomes previous characters, the 10 classes become 27 characters, and cross-entropy gets a probabilistic meaning (likelihood).
- **M03** swaps the MLP for a transformer — same six-step loop, same `nn.Module`s, bigger shapes `(B, T, C)`.
- **M05** returns to training at scale: the AdamW memory cost you saw here (2 extra copies of the weights) is why training needs ~8× the memory of inference.
- **M07 / M13**: the CPU-vs-GPU crossover you measured in the lab is the overhead → compute-bound transition, formalized with the roofline and CUDA graphs.
  */}),

  interview: [
    "What does `loss.backward()` compute, and why must you call `optimizer.zero_grad()` before it?",
    "Why do we need nonlinear activation functions? What happens if you stack 10 linear layers without them?",
    "Your 10-class classifier's loss at step 0 is 25. What does that tell you and how would you fix it?",
    "A model trains but plateaus at a bad loss. Walk me through how you'd debug it. (Expected: loss at init, overfit one batch, LR sweep, check data/labels.)",
    "What is the shape of `nn.Linear(4096, 14336).weight`, how many parameters does it have, and how many FLOPs does applying it to a `(8, 4096)` input take?",
    "What's the difference between `model.eval()`, `torch.no_grad()` and `torch.inference_mode()`?",
    "Why is reverse-mode autodiff (backprop) preferred over forward-mode for training neural networks?",
    "Roughly how much memory does training a model with AdamW in fp32 need per parameter, vs serving it in bf16?",
  ],

  resources: [
    { title: "Karpathy — building micrograd", url: "https://www.youtube.com/watch?v=VMj-3S1tku0", type: "video", note: "the core lecture of this module" },
    { title: "karpathy/micrograd", url: "https://github.com/karpathy/micrograd", type: "repo", note: "~150 lines of autograd; read all of it" },
    { title: "Neural Networks: Zero to Hero (syllabus)", url: "https://karpathy.ai/zero-to-hero.html", type: "course", note: "the series M01–M03 follow" },
    { title: "PyTorch — Learn the Basics", url: "https://pytorch.org/tutorials/beginner/basics/intro.html", type: "course", note: "official tensors → autograd → nn → optim tour" },
    { title: "3Blue1Brown — Neural Networks series", url: "https://www.3blue1brown.com/topics/neural-networks", type: "video", note: "visual intuition for gradient descent and backprop" },
    { title: "3Blue1Brown — Essence of Calculus", url: "https://www.3blue1brown.com/topics/calculus", type: "video", note: "derivatives and chain rule refresher" },
    { title: "Mathematics for Machine Learning (free book)", url: "https://mml-book.github.io/", type: "book", note: "reference for linear algebra / calculus gaps — don't read cover to cover" },
    { title: "Michael Nielsen — Neural Networks and Deep Learning", url: "http://neuralnetworksanddeeplearning.com/", type: "book", note: "free online book; ch. 2 is a gentle derivation of backprop on MNIST" },
    { title: "Karpathy — A Recipe for Training Neural Networks", url: "http://karpathy.github.io/2019/04/25/recipe/", type: "article", note: "debugging discipline" },
    { title: "Inference Engineering (Baseten) — Ch. 2.1 Neural Networks", url: "Inference%20Engineering.pdf", type: "book", note: "short book-level recap of what you just built" },
  ],
});
