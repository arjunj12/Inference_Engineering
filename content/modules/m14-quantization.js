Course.module({
  id: "m14-quantization",
  title: "Quantization: fewer bits, same answers",
  short: "Quantization",
  tagline: "Shrink one model three different ways (MLX, GGUF, vLLM), write your own INT8/INT4 quantizer, and prove with numbers — size, tok/s, perplexity, task accuracy — which format you would ship.",
  hours: 18,
  level: "core",
  runsOn: ["mac", "colab", "cloud"],
  tags: ["quantization", "fp8", "int4", "gptq", "awq", "gguf", "mlx", "evals"],

  goal: MD(function () {/*
You take **one** model — `Qwen2.5-1.5B-Instruct` — quantize it three different ways, and fill in a table like this with numbers **you measured**:

```text
model: Qwen2.5-1.5B-Instruct      (illustrative numbers: M2 Pro Mac, Colab T4, rented L4 — yours will differ)
format · engine               bits/wt   size     decode tok/s   PPL Δ vs its own 16-bit   GSM8K (250 q)
bf16 · MLX (Mac)              16        3.1 GB    46            —                         66.0% ± 3.0
4-bit g64 · MLX               4.5       0.9 GB   125            +4.5%                     63.2% ± 3.1
Q4_K_M · llama.cpp (Mac)      ~4.9      1.0 GB   112            +3.1%  (KLD 0.03)         64.4% ± 3.0
W4A16 GPTQ · vLLM (T4)        ~4.3      1.2 GB   140            +3.8%                     64.0% ± 3.0
FP8 W8A8 · vLLM (L4)          8         1.8 GB   160            +0.3%                     65.6% ± 3.0
INT8 g128 · YOUR PyTorch      8.25      1.8 GB    9 (!)         +0.2%                     —
```

Then you explain, with a roofline sketch, **why** 4-bit is ~2.5× faster at decode but barely faster (or slower) at prefill, why *your* INT8 quantizer is correct but **slower** than bf16 despite moving half the bytes, and which row you would ship for a concrete product.

The interactive below is the core idea of the whole module: squeeze a weight distribution (with outliers) into $2^b$ levels and watch the error depend on **bits** and on **how many weights share one scale**.
  */}),
  demo: { viz: "quantize-weights" },

  why: MD(function () {/*
Quantization is the most-used optimization after batching. Almost every serious deployment picks a precision on purpose: **FP8 on H100s** is the default at many providers, **NVFP4/MXFP4 on Blackwell** is arriving (GPT-OSS ships in MXFP4), and every local runtime (Ollama, LM Studio, MLX) runs **4-bit** weights. It is also the **only lossy** technique in the Baseten book’s *Techniques* chapter (Inference Engineering ch. 5.1) — which means whoever quantizes owns the quality risk. Inference-engineer job posts routinely list “quantization (FP8/INT4, GPTQ/AWQ)” and “evals”; interviewers love asking *why* 4-bit speeds up decode but not prefill. After this module you can answer from first principles **and** show your own measurements.
  */}),

  prereqs: [
    {
      title: "Bytes, bandwidth and the decode law (recap)",
      skipIf: "you did the napkin-math lesson in M00 and the roofline in M07",
      math: true,
      md: MD(function () {/*
- **Decode at batch 1 reads every weight once per token**, so time per token $\approx \dfrac{\text{weight bytes}}{\text{memory bandwidth}}$.
- Bytes = parameters × bytes per parameter. BF16 = 2 bytes, INT8/FP8 = 1 byte, 4-bit ≈ 0.5 byte (+ a little for scales).
- **Arithmetic intensity** = FLOPs ÷ bytes moved. A matrix–vector product does 2 FLOPs per weight. If the chip can do more FLOPs per byte than the operation needs (H100: $989\text{ TFLOPS} / 3.35\text{ TB/s} \approx 295$ FLOPs/byte), the operation is **memory-bound** and only fewer bytes make it faster.
- Example: Qwen2.5-1.5B in BF16 ≈ 3.1 GB. On an M2 Pro (~200 GB/s): $200 \div 3.1 \approx 65$ tok/s upper bound. In 4-bit (~0.9 GB): ~220 tok/s upper bound.
      */}),
    },
    {
      title: "Poking at a Hugging Face model in PyTorch",
      skipIf: "you finished M04 (loading real weights into your own Llama/Qwen)",
      md: MD(function () {/*
```python
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)  # older transformers: torch_dtype=
for name, mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear):
        print(name, tuple(mod.weight.shape))      # (out_features, in_features)
    if name.endswith("layers.1"): break
W = m.model.layers[0].mlp.down_proj.weight        # an nn.Parameter; .data gives the raw tensor
print(W.dtype, W.shape, W.abs().max().item(), W.std().item())
```
Every `nn.Linear` computes `y = x @ W.T + b` with `W` of shape `(out, in)`. Those Linear weights are ~85–95% of the parameters — and they are what we quantize.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: the same model in bf16 vs 4-bit on your Mac",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
## Step 1 — make two copies of one model

`mlx_lm.convert` downloads Hugging Face weights and writes an MLX copy. With `-q` it quantizes the Linear (and embedding) weights **group-wise**: every 64 consecutive weights share one scale and one offset.

```bash
uv pip install -U mlx-lm
M=Qwen/Qwen2.5-3B-Instruct
mlx_lm.convert --hf-path $M --mlx-path ~/models/q3b-bf16 --dtype bfloat16          # no quantization
mlx_lm.convert --hf-path $M --mlx-path ~/models/q3b-4bit -q --q-bits 4 --q-group-size 64
mlx_lm.convert --hf-path $M --mlx-path ~/models/q3b-8bit -q --q-bits 8 --q-group-size 64
mlx_lm.convert --hf-path $M --mlx-path ~/models/q3b-2bit -q --q-bits 2 --q-group-size 64   # for fun
du -sh ~/models/q3b-*
```

(16 GB Mac? This all fits: 3B in bf16 is ~6.2 GB. With 8 GB, use `Qwen2.5-1.5B-Instruct`.)

## Step 2 — run them on the same prompt

```bash
P="Explain, in 5 bullet points, why HTTP/2 multiplexing reduces latency."
for v in bf16 8bit 4bit 2bit; do
  echo "== $v"; mlx_lm.generate --model ~/models/q3b-$v --prompt "$P" --max-tokens 200 --temp 0
done
# cleaner speed numbers: fixed synthetic prompt/gen lengths, several trials
mlx_lm.benchmark --model ~/models/q3b-4bit -p 512 -g 128 -n 3
```

`mlx_lm.generate` prints **Prompt** tok/s (prefill), **Generation** tok/s (decode) and **Peak memory**. Fill this in:

| Variant | Disk | Peak mem | Prefill tok/s | Decode tok/s | Napkin decode bound (BW ÷ bytes) | Output still good? |
|---|---|---|---|---|---|---|
| bf16 | | | | | | |
| 8-bit | | | | | | |
| 4-bit | | | | | | |
| 2-bit | | | | | | |

## What you should notice

1. **Decode speed tracks bytes.** 4-bit is typically 2.5–3.5× faster than bf16 — close to the ratio of file sizes. That is the decode law from M00 doing exactly what it predicts.
2. **Prefill barely changes** (sometimes 4-bit is *slower*). Prefill is compute-bound, and Apple GPUs have no 4-bit math units: MLX unpacks the 4-bit weights to fp16 inside the kernel and does fp16 math. Fewer bytes don’t help a computation that wasn’t waiting on bytes.
3. **8-bit is visually identical, 4-bit is nearly identical, 2-bit falls apart** (repetition, wrong facts, broken formatting). Quality doesn’t degrade linearly — there is a cliff, and *where* the cliff is depends on the model size and the method.

> [!INTUITION] The whole module in one sentence
> Quantization trades a little numerical precision for a lot fewer bytes. Fewer bytes = faster memory-bound work, bigger batches (more room for KV cache), fewer GPUs. The engineering is in (a) losing as little quality as possible and (b) turning fewer bytes into actual speed with the right kernels.

> [!TIP] Peek inside
> Open `~/models/q3b-4bit/config.json` → `"quantization": {"group_size": 64, "bits": 4}`. Then in Python, `mx.load(path)` one of the `.safetensors` files in that folder (`import mlx.core as mx`; expand `~` with `os.path.expanduser`) and look for keys ending in `.scales` and `.biases` next to each packed `.weight` (dtype `uint32` — eight 4-bit values per word). You will build exactly this layout yourself later.

- [ ] Converted bf16 / 8-bit / 4-bit / 2-bit copies and filled the table
- [ ] Napkin bound computed for each; decode measurements within ~2× of it
- [ ] Wrote one sentence explaining why prefill didn’t speed up
      */}),
      resources: [
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "repo", note: "convert / generate / benchmark / perplexity / evaluate CLIs used throughout this module" },
      ],
    },
    {
      id: "number-formats",
      title: "Number formats: how a number fits in 4, 8 or 16 bits",
      kind: "math",
      minutes: 75,
      runsOn: ["any"],
      md: MD(function () {/*
> [!PREREQ] Binary in 90 seconds
> A bit is 0/1. With $b$ bits you can write $2^b$ different patterns: 4 bits → 16, 8 bits → 256, 16 bits → 65,536. Place values are powers of two: `1011` $= 8 + 0 + 2 + 1 = 11$. **Signed** integers use two’s complement: INT8 covers $-128 \ldots 127$, INT4 covers $-8 \ldots 7$. Every format below is just a *rule for which real number each bit pattern means*.

## Integers: an evenly spaced ruler

INT8 alone can only say 127, not 0.0137. So integer formats always come with a **scale** $s$: the stored integer $q$ means $s \cdot q$. The representable values are an **evenly spaced grid** — great when the data is evenly spread, bad when a few values are huge (outliers force a big $s$, and then small values all round to 0). Next lesson is all about choosing $s$.

## Floats: scientific notation in binary

You know $6.02 \times 10^{23}$: a sign, some significant digits (**mantissa**), and a power (**exponent**). Floats do the same in base 2:

$$
x = (-1)^{\text{sign}} \times 1.\text{mantissa}_2 \times 2^{\,\text{exponent} - \text{bias}}
$$

- **Exponent bits → range** (how big/small you can go).
- **Mantissa bits → precision** (how many values between two powers of two). With $m$ mantissa bits the step between neighbours is $2^{-m}$ *relative* to the number’s size.

So floats have a **logarithmic ruler**: just as many values between 1 and 2 as between 1000 and 2000. That is why floats handle outliers better than ints (Baseten ch. 5.1: “floats beat ints because the exponent bits preserve outliers”).

### Worked example: store 0.3

$0.3 = 1.2 \times 2^{-2}$. The exponent is $-2$; the mantissa must approximate $1.2$ with $m$ bits after the binary point:

| Format | S/E/M bits | Bias | Nearest mantissa | Stored value | Relative error |
|---|---|---|---|---|---|
| FP16 | 1/5/10 | 15 | $\tfrac{1229}{1024}$ | 0.300059 | 0.02% |
| BF16 | 1/8/7 | 127 | $\tfrac{154}{128}$ | 0.300781 | 0.26% |
| FP8 E4M3 | 1/4/3 | 7 | $1.25$ (grid: 1.125, **1.25**) | 0.3125 | 4.2% |
| FP8 E5M2 | 1/5/2 | 15 | $1.25$ (grid: 1.0, **1.25**, 1.5) | 0.3125 | 4.2% |
| FP4 E2M1 | 1/2/1 | 1 | nearest of the 8 values → 0.5 | 0.5 | 67% (needs a scale!) |

E4M3 bits for 0.3: sign `0`, exponent $-2+7=5$ → `0101`, mantissa $.25 = .010_2$ → `010`, i.e. `0 0101 010`.

## The formats you’ll meet

::viz number-formats

| Format | Layout | Max | Relative step | Typical use | First NVIDIA support |
|---|---|---|---|---|---|
| FP32 | 1/8/23 | $3.4\times10^{38}$ | $2^{-23}\approx10^{-7}$ | master weights in training; never for inference | Kepler |
| **BF16** | 1/8/7 | $3.4\times10^{38}$ | $2^{-7}\approx0.8\%$ | native training & inference (FP32’s range, less precision) | Ampere |
| FP16 | 1/5/10 | 65,504 | $2^{-10}\approx0.1\%$ | older inference; can overflow | Pascal |
| **FP8 E4M3** | 1/4/3 | 448 | $2^{-3}=12.5\%$ | weights & activations (forward pass) | Hopper |
| FP8 E5M2 | 1/5/2 | 57,344 | $2^{-2}=25\%$ | gradients (need range more than precision) | Hopper |
| INT8 | 8-bit integer | 127·s | uniform | W8A8 (SmoothQuant), weight-only | Turing/Ampere tensor cores |
| INT4 | 4-bit integer | 7·s | uniform | weight-only W4A16 (GPTQ/AWQ/GGUF/MLX) | Turing |
| FP4 E2M1 | 1/2/1 | 6 | values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 | only usable with block scales ↓ | Blackwell |
| **MXFP4** | E2M1 + shared 8-bit power-of-two scale per **32** values | — | — | GPT-OSS weights; OCP standard | Blackwell |
| **NVFP4** | E2M1 + FP8 E4M3 scale per **16** values + FP32 tensor scale | — | — | NVIDIA’s 4-bit inference format | Blackwell |

**Microscaling** (MX, NVFP4) is the big modern idea: tiny elements, but each small block gets its own scale, so the grid is re-centred every 16–32 values. Cost: MXFP4 = $4 + 8/32 = 4.25$ bits/weight; NVFP4 = $4 + 8/16 = 4.5$ bits/weight. Blackwell tensor cores apply the block scales in hardware, so the overhead is nearly free.

> [!NOTE] Your Mac
> Apple GPUs have no FP8/FP4 math units. MLX 4-bit (and `--q-mode mxfp4` / `nvfp4` in recent `mlx_lm.convert`) stores fewer bytes but computes in fp16 — so on a Mac quantization speeds up **memory-bound** work only.

## Errors compound

The Baseten book’s π example: $3.14159^3 = 31.006$, $3.14^3 = 30.959$, $3^3 = 27$. A small error per operation grows through repeated multiplication — and a transformer is dozens of layers of repeated multiplication. This is why 16-bit accumulation (the running sum inside a matmul is kept in FP32/FP16 even when inputs are FP8/INT4) and sensitive parts like softmax stay high-precision.

## Try it in PyTorch

```python
import torch
x = torch.tensor([0.3, 300.0, 500.0, 1e-4])
for dt in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
    print(dt, x.to(dt).float().tolist())
# Every one of the 256 E4M3 bit patterns, decoded:
vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
print(vals[~vals.isnan()].unique())       # note the log spacing and max = 448
```

Casting 500 to E4M3 without a scale typically gives `nan` (it’s beyond 448 and there’s no infinity in E4M3) — exactly why every real FP8 recipe first computes a **scale** so the tensor’s max maps near 448.

- [ ] By hand: encode 0.3 and 500 in E4M3 and E5M2 (which one overflows?) and check with the PyTorch snippet
- [ ] Plot the 256 E4M3 values on a log x-axis; explain why the dots are evenly spaced *on the log axis*
- [ ] In one sentence each: why BF16 replaced FP16 for training, and why E4M3 (not E5M2) is used for inference weights
      */}),
      resources: [
        { title: "FP8 Formats for Deep Learning (Micikevicius et al.)", url: "https://arxiv.org/abs/2209.05433", type: "paper", note: "defines E4M3/E5M2 and why each exists" },
        { title: "Microscaling Data Formats for Deep Learning (OCP MX)", url: "https://arxiv.org/abs/2310.10537", type: "paper", note: "MXFP8/MXFP6/MXFP4 block-scaled formats" },
        { title: "NVIDIA — Introducing NVFP4", url: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/", type: "article", note: "16-value blocks + FP8 scales + FP32 tensor scale" },
      ],
    },
    {
      id: "quant-math",
      title: "The math of quantizing: scale, zero-point, granularity, error",
      kind: "math",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
> [!PREREQ] Four tiny operations
> **round** to the nearest integer · **clamp** a value into $[lo, hi]$ · **max/min** of a list · **mean squared error** $\frac{1}{n}\sum (x - \hat{x})^2$ (average squared mistake; its square root, RMSE, is in the original units).

## The one formula

Pick a scale $s$ (a float) and a zero-point $z$ (an integer). Then:

$$
q = \operatorname{clamp}\!\Big(\operatorname{round}\big(\tfrac{x}{s}\big) + z,\; q_{\min},\; q_{\max}\Big)
\qquad\qquad
\hat{x} = s\,(q - z)
$$

$q$ is what you store (few bits); $\hat{x}$ is what you compute with after **dequantizing**. Everything else is *how to pick* $s$ and $z$ and *for how many weights* to share them.

### Symmetric (zero-point = 0)

$$ s = \frac{\max |x|}{2^{b-1} - 1}, \qquad z = 0 $$

For INT8 the denominator is 127; for INT4 it is 7. Simple and fast (no $z$ in the kernel), but wastes levels when the data is lopsided.

### Asymmetric (affine)

$$ s = \frac{\max x - \min x}{2^{b} - 1}, \qquad z = \operatorname{round}\!\Big(\frac{-\min x}{s}\Big) $$

Uses all $2^b$ levels across the actual range. (MLX stores a float “bias” $= -s\,z$ instead of an integer $z$ — same idea.)

## Worked example: 5 weights → INT4

$x = [-0.75,\; -0.1,\; 0.05,\; 0.3,\; 1.6]$

**Symmetric:** $s = 1.6 / 7 = 0.2286$. $x/s = [-3.28, -0.44, 0.22, 1.31, 7.0]$ → $q = [-3, 0, 0, 1, 7]$ → $\hat{x} = [-0.686, 0, 0, 0.229, 1.6]$. Errors: $[0.064, 0.1, 0.05, 0.071, 0]$.

**Asymmetric (UINT4, 0…15):** $s = 2.35/15 = 0.1567$, $z = \operatorname{round}(0.75/0.1567) = \operatorname{round}(4.79) = 5$. $q = \operatorname{round}(x/s) + 5 = [0, 4, 5, 7, 15]$ → $\hat{x} = s(q-5) = [-0.783, -0.157, 0, 0.313, 1.567]$. Errors: $[0.033, 0.057, 0.05, 0.013, 0.033]$.

Asymmetric wins here because the data is lopsided: symmetric reserved levels down to $-1.6$ that nothing uses.

## How big is the rounding error?

The grid step is $s$. Rounding moves each value by at most $s/2$, and — for “random-looking” data — the error is roughly uniform on $[-s/2, s/2]$, whose RMS is

$$ \text{RMSE} \approx \frac{s}{\sqrt{12}} $$

Each extra bit halves $s$, so it halves the error. And $s$ is set by the **largest** value in the group: one outlier 10× bigger than everything else makes $s$ 10× bigger, costing everyone else $\log_2 10 \approx 3.3$ bits of resolution. **Outliers are the entire difficulty of quantization.**

## Granularity: how many weights share one scale?

For a Linear weight $W$ of shape (out, in):

| Granularity | One scale per | Scales for 4096×4096 | Handles outliers | Kernel cost |
|---|---|---|---|---|
| per-tensor | whole matrix | 1 | poorly | free |
| per-channel | output row | 4,096 | ok (outlier only hurts its own row) | free (see below) |
| per-group $g$ | $g$ consecutive inputs in a row | $4096 \cdot 4096/g$ | well | scales applied inside the matmul loop |

Why is per-channel “free”? Output $j$ is $y_j = \sum_k W_{jk} x_k = \sum_k s_j q_{jk} x_k = s_j \sum_k q_{jk} x_k$ — the scale factors *out* of the sum, one multiply at the end. With per-group scales the scale changes *along* the sum ($k$), so the kernel must multiply inside the loop — that’s why group-wise formats need special kernels (Marlin, MLX `quantized_matmul`, llama.cpp’s `mul_mat_q`).

::viz quantize-weights

## Bits per weight (what the file size really is)

$$ \text{bpw} = b + \frac{\text{bits of scale} + \text{bits of zero/bias}}{g} $$

- MLX 4-bit, $g=64$, fp16 scale + fp16 bias: $4 + 32/64 = 4.5$ bpw.
- GPTQ 4-bit, $g=128$, fp16 scale (+ packed 4-bit zero): $\approx 4.16$ bpw.
- llama.cpp **Q8_0**: blocks of 32 + fp16 scale = 8.5 bpw. **Q4_K**: super-blocks of 256 with 6-bit sub-block scales/mins = 4.5 bpw. **Q4_K_M** keeps some sensitive tensors (parts of `attn_v`, `ffn_down`) in Q6_K (6.56 bpw) → ~4.8–4.9 bpw average.

So “4-bit” is never exactly 4 bits. Always compare **file size**, not the marketing number.

## Exercises

```python
import numpy as np
rng = np.random.default_rng(0)
w = rng.standard_t(df=4, size=4096) * 0.02        # heavy-ish tails like real weights
def q_sym(x, b):
    s = np.abs(x).max() / (2**(b-1) - 1)
    return np.clip(np.round(x / s), -(2**(b-1) - 1), 2**(b-1) - 1) * s, s
for b in (8, 6, 4, 3):
    xh, s = q_sym(w, b)
    print(b, "RMSE", np.sqrt(np.mean((w - xh)**2)), "predicted s/sqrt(12)", s / np.sqrt(12))
```

- [ ] Redo the 5-weight example with INT8 symmetric; how much smaller are the errors?
- [ ] Run the snippet; confirm RMSE ≈ $s/\sqrt{12}$ and that each bit roughly halves it
- [ ] Add one outlier `w[0] = 1.0` and re-run; then quantize in groups of 64 and show the damage is confined to one group
- [ ] Compute bpw for 4-bit with $g=32$ and an fp16 scale only; when is the scale overhead worth it?
      */}),
      resources: [
        { title: "A Visual Guide to Quantization (Maarten Grootendorst)", url: "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization", type: "article", note: "the best illustrated walk-through of scales, zero-points, GPTQ, GGUF, QAT" },
        { title: "llama.cpp quantize README", url: "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md", type: "docs", note: "every GGUF type with bits/weight and quality numbers" },
      ],
    },
    {
      id: "why-faster",
      title: "Why fewer bits are faster (and when they aren’t)",
      kind: "concept",
      minutes: 60,
      runsOn: ["any"],
      md: MD(function () {/*
## Decode: fewer bytes → fewer milliseconds

At batch 1, a decode step reads all weights once. Halve the bytes, roughly halve the time. Baseten (ch. 5.1) puts the realistic gain at **~30–50% per precision step** in production, not 2×, because weights aren’t the only bytes (KV cache, activations), kernels don’t hit peak bandwidth, and fixed overheads (launches, sampling, scheduling) don’t shrink.

## The roofline picture

For a Linear layer with $N$ weights processing $B$ tokens at once (batch $B$ at decode, or $B$ prompt tokens at prefill):

$$
\text{FLOPs} = 2BN, \qquad \text{bytes} \approx N \cdot \text{bytes/weight}
\qquad\Rightarrow\qquad
\text{arithmetic intensity} = \frac{2B}{\text{bytes/weight}}
$$

| Weights | bytes/weight | Intensity at $B=1$ | $B$ where H100 turns compute-bound (≈295 FLOP/B, BF16 math) |
|---|---|---|---|
| BF16 | 2 | 1 | ~295 |
| INT8 / FP8 weights, BF16 math (W8A16) | 1 | 2 | ~150 |
| INT4 weights, BF16 math (W4A16) | ~0.53 | ~3.8 | ~80 |
| FP8 weights **and** FP8 math (W8A8) | 1 | 2 | ~295 — but against a 2× higher ceiling (~1979 TFLOPS, ridge ≈ 590) |

::viz roofline

Read it like this:

1. **At low batch, everything is memory-bound** → the format with the fewest bytes wins. W4A16 is king for local, single-user, and latency-critical serving.
2. **Weight-only quantization moves the ridge *left*.** A W4A16 layer becomes compute-bound at a much smaller batch — and once compute-bound, it runs at **BF16 math speed plus dequantization work**. At large batch, W4A16 can be *slower* than plain BF16.
3. **Only low-precision *math* raises the compute ceiling.** FP8 tensor cores on Hopper do ~2× the BF16 FLOPS; FP4 on Blackwell ~2× again. That’s why high-throughput deployments on H100/B200 prefer **W8A8 FP8** or **NVFP4**: they help at *every* batch size, including prefill.

::viz latency-throughput

## Prefill: compute-bound, so weight-only doesn’t help

Prefill processes hundreds or thousands of tokens per weight read — intensity is huge, it’s compute-bound. Fewer weight bytes change nothing; only faster math (FP8/FP4 tensor cores) does. That explains your Mac result from the first lesson: MLX 4-bit decode got ~3× faster, prefill didn’t.

## Dequantization overhead is real

Weights must be turned back into BF16/FP16 before multiplying (unless the hardware multiplies low-precision directly). Two ways to do it:

- **Naive:** dequantize the whole matrix into a BF16 tensor in memory, then call a normal matmul. You just *wrote and re-read* a full BF16 matrix — **more** bytes than not quantizing. You will build this first, and it will be slow.
- **Fused:** load packed 4-bit weights into registers/shared memory, unpack and scale them right there, multiply, never materialize BF16 in HBM. This is what Marlin (vLLM), MLX’s `quantized_matmul`, and llama.cpp’s kernels do. It’s a classic M12-style Triton exercise.

## Memory → capacity → throughput → money

Even when latency barely moves, quantization frees memory:

- **Bigger batches:** 70B in BF16 = 140 GB (needs 2×H100 just for weights). In FP8 = 70 GB → fits on one H100 with a sliver of KV cache, or on 2×H100 with ~5× more KV cache than BF16 → bigger batches → more tokens/s per dollar.
- **Fewer GPUs, less communication:** fewer tensor-parallel shards means fewer all-reduces (M16).
- **KV cache quantization** (FP8 KV) doubles the tokens you can cache → more concurrency, better prefix-cache hit rates, cheaper KV transfer in disaggregated serving (M17).

## When quantization **doesn’t** help

| Situation | Why |
|---|---|
| Long-prompt, short-output workloads with weight-only quant | dominated by prefill (compute-bound) |
| Very large batches with W4A16 | compute-bound + dequant overhead |
| Tiny models on a Mac/CPU | per-token overhead (Python, launches) dominates, not bytes |
| Naive dequant in PyTorch | extra memory traffic |
| Hardware without the format (FP8 on T4/A100, FP4 on Hopper) | falls back to weight-only emulation or doesn’t run |

> [!REAL] What engines pick
> vLLM chooses a kernel per checkpoint and GPU: Marlin/Machete for INT4/INT8 weight-only, CUTLASS FP8 GEMMs on Hopper, FP4 GEMMs on Blackwell. TensorRT-LLM does the same with its own kernels; llama.cpp and MLX have hand-written Metal kernels per GGUF/MLX type. As an inference engineer you mostly **choose** formats and kernels — and occasionally write one.

- [ ] Using the roofline viz, find the batch size where a W4A16 7B model becomes compute-bound on an H100 and on an M-series Mac
- [ ] Explain to a colleague, in 3 sentences, why FP8 W8A8 beats W4A16 at batch 256 but loses at batch 1
      */}),
      resources: [
        { title: "PyTorch — Accelerating Generative AI with PyTorch II: GPT, Fast", url: "https://pytorch.org/blog/accelerating-generative-ai-2/", type: "article", note: "int8/int4 weight-only quant measured end-to-end; memory-bandwidth framing" },
      ],
    },
    {
      id: "weights-vs-activations",
      title: "Weight-only vs weight + activation (W8A8) and the outlier problem",
      kind: "concept",
      minutes: 75,
      runsOn: ["mac", "any"],
      md: MD(function () {/*
## The naming scheme

**W*x*A*y*** = weights in $x$ bits, activations (the inputs to each matmul) in $y$ bits.

| Scheme | What’s low-precision | Math done in | Helps decode | Helps prefill | Examples |
|---|---|---|---|---|---|
| W4A16, W8A16 | weights only | BF16/FP16 | ✅ | ❌ | GPTQ, AWQ, GGUF, MLX |
| W8A8-INT8 | weights + activations | INT8 tensor cores | ✅ | ✅ | SmoothQuant |
| W8A8-FP8 | weights + activations | FP8 tensor cores (Hopper+) | ✅ | ✅ | vLLM `--quantization fp8`, llm-compressor `FP8_DYNAMIC` |
| W4A4 (NVFP4/MXFP4) | both | FP4 tensor cores (Blackwell) | ✅ | ✅ | NVFP4 checkpoints |

Weight-only is safest (Baseten: “weights-only is safest with the smallest gains”). Quantizing activations too lets the **math** run in low precision — but activations are harder.

## Why activations are harder

1. **They change every request.** Weights are fixed, so you can spend an hour choosing perfect scales. Activation scales must be either **static** (measured on calibration data, fixed forever) or **dynamic** (computed on the fly per token, costing a reduction per matmul).
2. **Outlier channels.** In LLMs beyond a few billion parameters, a handful of hidden dimensions carry activation values **10–100× larger** than the rest, consistently across tokens (found by LLM.int8(), Dettmers et al. 2022). Per-tensor INT8 of such an activation wastes almost all 256 levels on those few channels.

## See the outliers yourself (Mac)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to("mps").eval()

stats = {}
def hook(mod, inp, out, key=None):
    a = inp[0].detach().abs().flatten(0, -2)                  # (tokens, in_features)
    stats[key] = torch.maximum(stats.get(key, torch.zeros_like(a[0])), a.amax(0))
for i, layer in enumerate(model.model.layers):
    layer.mlp.down_proj.register_forward_hook(lambda m, i_, o, k=f"L{i}.down": hook(m, i_, o, k))
    layer.self_attn.q_proj.register_forward_hook(lambda m, i_, o, k=f"L{i}.q": hook(m, i_, o, k))

text = open("some_text.txt").read()[:4000]                  # any few paragraphs
with torch.no_grad():
    model(**tok(text, return_tensors="pt").to("mps"))
for k, v in stats.items():
    v = v.cpu()
    print(f"{k:10s} max {v.max():8.2f}  median {v.median():6.3f}  ratio {v.max()/v.median():7.1f}  top channels {v.topk(3).indices.tolist()}")
```

You should see some layers where the ratio is in the tens or hundreds, and the **same few channel indices** show up again and again. Weights, by contrast, are well-behaved (check `W.abs().max() / W.std()` per layer).

## SmoothQuant: move the difficulty from activations to weights

A matmul $Y = XW^\top$ doesn’t change if you divide activation channel $j$ by $s_j$ and multiply the matching weight column by $s_j$:

$$ Y = \big(X \operatorname{diag}(s)^{-1}\big)\big(\operatorname{diag}(s) W^\top\big) $$

Choose $s_j$ to balance the two sides:

$$ s_j = \frac{\max |X_{:,j}|^{\alpha}}{\max |W_{:,j}|^{1-\alpha}}, \qquad \alpha \approx 0.5 $$

**Worked example:** channel $j$ has activation max 50 and weight max 0.5. With $\alpha = 0.5$: $s_j = \sqrt{50}/\sqrt{0.5} = 10$. Now the activation max is $5$ and the weight max is $5$ — both easy to quantize. The division by $s$ is folded into the preceding RMSNorm weights offline, so it’s **free at runtime**.

## Why FP8 largely replaced INT8 W8A8

E4M3 has a *logarithmic* grid with range ±448 — outliers survive without clever smoothing. With per-tensor (or per-channel weight + per-token dynamic activation) scales, FP8 W8A8 is usually near-lossless for models ≥7B. That’s why Baseten calls **FP8/MXFP8 the sweet spot** — on hardware that has it (Ada/Hopper/Blackwell; not your T4, A100 or Mac).

> [!REAL] LLM.int8()
> The original fix (bitsandbytes `load_in_8bit`) splits the matmul: outlier channels in FP16, the rest in INT8. Accurate, but the split makes it *slower* than FP16 — a good example of “fewer bits ≠ faster” without the right kernel.

- [ ] Ran the outlier script; listed the 3 layers with the largest max/median ratio and their channel indices
- [ ] Applied SmoothQuant by hand to one layer (α = 0.5): recompute max|X| and max|W| after smoothing and verify the output $Y$ is unchanged to ~1e-5
- [ ] Explain in 2 sentences why dynamic per-token activation scales cost extra work at runtime
      */}),
      resources: [
        { title: "LLM.int8() (Dettmers et al.)", url: "https://arxiv.org/abs/2208.07339", type: "paper", note: "discovered emergent outlier features in LLM activations" },
        { title: "SmoothQuant (Xiao et al.)", url: "https://arxiv.org/abs/2211.10438", type: "paper", note: "migrate quantization difficulty from activations to weights" },
      ],
    },
    {
      id: "methods",
      title: "Methods: RTN, GPTQ, AWQ, QAT, KV-cache quant, FP8 & FP4 in production",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
All methods answer the same question: **given a bit budget, which integer (or FP4/FP8) value should each weight get, and which scales?** Plain rounding is the baseline; the rest are smarter.

## RTN — round-to-nearest

What you did in the math lesson: compute scales from min/max, round each weight independently. No data needed, seconds to run. Surprisingly good at 8 bits (and for FP8), noticeably worse at 4 bits on small models. MLX `-q` and llama.cpp’s basic types are RTN-like (llama.cpp can add an **importance matrix**, `llama-imatrix`, to weight the rounding).

## GPTQ — fix rounding errors as you go

Goal per Linear layer: minimize the change in *outputs* on real inputs, $\lVert WX - \hat{W}X \rVert^2$, not the change in weights. GPTQ (Frantar et al. 2022) quantizes one input-column at a time and **pushes the error it just made onto the columns not yet quantized**, weighted by how correlated those inputs are — information captured by $H = 2XX^\top$ (the “Hessian” of that squared error; $X$ = calibration activations).

> [!INTUITION] GPTQ as error diffusion
> Like dithering an image: when one pixel rounds darker, nudge its neighbours lighter so the overall picture stays right. GPTQ does that across weight columns, using calibration data to know which neighbours matter.

Needs ~128–512 calibration samples, minutes for small models, hours for 70B. Output: INT4 (or INT8) weights + group scales (commonly $g=128$) → served with Marlin kernels in vLLM.

## AWQ — protect the weights that matter

AWQ (Lin et al. 2023) observes that ~1% of weight *channels* matter far more — the ones multiplied by large activations. Instead of keeping them in FP16 (slow, mixed precision), it **scales those input channels up** before quantizing (so they get relatively finer resolution) and scales the activations down by the same factor (folded into the previous op). It grid-searches the scaling exponent per layer. No backprop, less prone to overfit the calibration set than GPTQ. Also served as W4A16.

## QAT — train with the rounding in the loop

Quantization-aware training inserts “fake quantization” (quantize→dequantize) in the forward pass and fine-tunes, so the weights learn to sit where rounding doesn’t hurt. Gradients flow through `round` with the *straight-through estimator* (pretend its derivative is 1). Best quality at low bits, but needs training data and compute — mostly done by model *creators*: GPT-OSS ships natively in **MXFP4**, Kimi K2 Thinking in **INT4** (Baseten ch. 5.1). Close cousins: MLX’s **DWQ** (distill quantized from full precision), QLoRA’s NF4 for fine-tuning.

## KV-cache quantization

The KV cache often dominates memory at long context and high batch (M08). Storing K and V in FP8 halves it:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8        # Hopper/Ada best; check your GPU
mlx_lm.generate --model ... --kv-bits 8 --kv-group-size 64         # Mac
```

It’s more sensitive than weights (Baseten ranks: weights < activations < KV cache < attention/softmax) because errors enter every future token’s attention. Research goes further (KIVI: 2-bit, keys per-channel and values per-token). Payoff beyond memory: more cached prefixes, cheaper KV transfer for disaggregation.

## Formats by hardware (what you’d actually ship)

| Hardware | Best default | Notes |
|---|---|---|
| Mac (MLX / llama.cpp) | 4-bit g64 MLX, Q4_K_M / Q5_K_M / Q6_K GGUF | memory-bound; bytes are everything |
| T4 / consumer GPUs | W4A16 (GPTQ/AWQ, Marlin), GGUF | no FP8 hardware |
| A100 (Ampere) | W4A16 for latency; W8A8-INT8 for throughput | FP8 only as weight-only emulation |
| L4 / L40S / H100 / H200 | **FP8 W8A8** (+ FP8 KV) | Baseten’s “sweet spot” |
| B200 / GB200 | **NVFP4** (or MXFP4) W4A4, FP8 for sensitive layers | FP4 tensor cores, in-hardware block scales |

## Baseten’s production recipe (ch. 5.1)

- Stick to **floating point** for quality-sensitive work; ints lack dynamic range.
- **Moderate recipe:** FP8 (ideally MXFP8) on selected linear layers, activations and often the KV cache. Rarely quantize attention; softmax stays in original precision. Often keep the first and last layers in high precision.
- **Bigger models are less sensitive** — a 70B tolerates 4-bit much better than a 1.5B.
- Use **NVIDIA TensorRT Model Optimizer (ModelOpt)** or **llm-compressor**; both export checkpoints vLLM/SGLang/TRT-LLM load directly.
- The bar is **zero perceptible loss** — which you must *prove* with evals (last lesson).

## The tools, mapped

| Tool | Methods | Output | Runs on |
|---|---|---|---|
| `mlx_lm.convert -q`, `mlx_lm.awq`, `mlx_lm.gptq`, `mlx_lm.dwq`, `mlx_lm.dynamic_quant` | RTN, AWQ, GPTQ, DWQ, mixed-bit | MLX safetensors | Mac |
| `llama-quantize` (+ `llama-imatrix`) | k-quants, i-quants | GGUF | anywhere |
| **llm-compressor** | RTN, GPTQ, AWQ, SmoothQuant, FP8, NVFP4, KV | compressed-tensors (vLLM) | NVIDIA GPU (CPU for tiny models) |
| TensorRT Model Optimizer | FP8, NVFP4, INT4-AWQ, QAT | HF / TRT-LLM checkpoints | NVIDIA |
| torchao | int8/int4 weight-only, FP8, QAT | PyTorch tensor subclasses | NVIDIA, some CPU/MPS |
| vLLM `--quantization fp8` | on-the-fly FP8 (no calibration) | — | Ada/Hopper+ |

- [ ] For each of RTN, GPTQ, AWQ, QAT write one sentence: what data it needs and what it optimizes
- [ ] Pick a model you care about on Hugging Face and find its official FP8/INT4/GGUF variants; note which method produced each
      */}),
      resources: [
        { title: "GPTQ (Frantar et al.)", url: "https://arxiv.org/abs/2210.17323", type: "paper", note: "one-shot second-order weight quantization" },
        { title: "AWQ (Lin et al.)", url: "https://arxiv.org/abs/2306.00978", type: "paper", note: "activation-aware scaling of salient channels" },
        { title: "KIVI: 2-bit KV cache quantization", url: "https://arxiv.org/abs/2402.02750", type: "paper", note: "why keys and values want different granularity" },
        { title: "vLLM — Quantized KV cache", url: "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/", type: "docs", note: "FP8 KV in practice" },
      ],
    },
    {
      id: "build-quantizer",
      title: "Build: your own INT8/INT4 group-wise weight-only quantizer in PyTorch",
      kind: "build",
      minutes: 240,
      runsOn: ["mac"],
      md: MD(function () {/*
> [!BUILD] What you’ll have
> `myquant.py`: quantize/dequantize functions for any bits × {per-tensor, per-channel, per-group}, a drop-in `QuantLinear` module, INT4 nibble packing, a perplexity harness, and a sweep table of error, perplexity and memory. Everything on your Mac with `Qwen2.5-0.5B-Instruct`.

## Step 1 — quantize and dequantize

```python
# myquant.py
import math, time, torch, torch.nn as nn, torch.nn.functional as F

def _groups(w, group):
    out_f, in_f = w.shape
    if group == "tensor":  return w.reshape(1, 1, -1)
    if group == "channel": return w.reshape(out_f, 1, in_f)
    assert in_f % group == 0, f"in_features {in_f} not divisible by {group}"
    return w.reshape(out_f, in_f // group, group)

def quantize(w, bits=8, group=128, symmetric=True):
    """w: (out, in). Returns integer codes, scales, zero-points (all shaped for broadcasting)."""
    g = _groups(w.float(), group)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1                                   # 127 or 7
        scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        zero = torch.zeros_like(scale)
        q = torch.clamp(torch.round(g / scale), -qmax, qmax).to(torch.int8)
    else:
        qmax = 2 ** bits - 1                                         # 255 or 15
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-8) / qmax
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(g / scale) + zero, 0, qmax).to(torch.uint8)
    return q, scale, zero

def dequantize(q, scale, zero, shape):
    return ((q.float() - zero) * scale).reshape(shape)
```

Test on a real weight before going further:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
W = model.model.layers[10].mlp.down_proj.weight.data            # (896, 4864)

print(f"{'bits':>4} {'gran':>8} {'sym':>5} {'rel. error':>10}")
for bits in (8, 4, 3):
    for gran in ("tensor", "channel", 128, 32):
        for sym in (True, False):
            q, s, z = quantize(W, bits, gran, sym)
            err = (dequantize(q, s, z, W.shape) - W).norm() / W.norm()
            print(f"{bits:>4} {str(gran):>8} {str(sym):>5} {err:10.4f}")
```

Relative weight error is a proxy. What matters is the **output** error on real inputs — capture the layer’s input `X` with a forward hook (as in the outliers lesson) and report $\lVert XW^\top - X\hat{W}^\top\rVert / \lVert XW^\top\rVert$ too.

## Step 2 — a drop-in `QuantLinear`

```python
class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits=8, group=128, symmetric=True):
        super().__init__()
        self.shape = tuple(lin.weight.shape)
        q, s, z = quantize(lin.weight.data.cpu(), bits, group, symmetric)
        self.register_buffer("q", q)
        self.register_buffer("scale", s.half())                      # 16-bit scales, like real formats
        self.register_buffer("zero", z.half())
        self.bias = lin.bias                                         # Qwen q/k/v have biases

    def forward(self, x):
        w = dequantize(self.q, self.scale.float(), self.zero.float(), self.shape).to(x.dtype)
        return F.linear(x, w, self.bias)                              # naive: materializes w!

def quantize_model(model, skip=("lm_head",), **kw):
    targets = [(parent, name, child) for parent in model.modules()
               for name, child in parent.named_children()
               if isinstance(child, nn.Linear) and name not in skip]
    for parent, name, child in targets:
        setattr(parent, name, QuantLinear(child, **kw))
    return model

def model_bytes(model):
    ts = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in ts)
```

(Qwen2.5-0.5B ties `lm_head` to the embedding matrix, so we leave both alone — a common real-world choice: first and last layers stay high-precision.)

## Step 3 — a perplexity harness

Perplexity = $\exp(\text{mean negative log-likelihood per token})$ on held-out text (M02). Lower = the model is less surprised.

```python
from datasets import load_dataset

def wikitext_ids(tok, max_tokens=60_000):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tok("".join(ds["text"]), return_tensors="pt").input_ids[0][:max_tokens]

@torch.no_grad()
def perplexity(model, ids, ctx=1024, device="mps"):
    model.to(device).eval()
    nll, n = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i : i + ctx + 1].to(device)
        if len(chunk) < 2: break
        logits = model(chunk[None, :-1]).logits[0].float()
        nll += F.cross_entropy(logits, chunk[1:], reduction="sum").item()
        n += len(chunk) - 1
    return math.exp(nll / n)
```

## Step 4 — the sweep

```python
import copy
ids = wikitext_ids(tok)
base = perplexity(model, ids)
print(f"fp32: ppl {base:.3f}  size {model_bytes(model)/1e9:.2f} GB")
for bits, gran, sym in [(8, "channel", True), (8, 128, True), (4, "channel", True),
                        (4, 128, True), (4, 128, False), (4, 32, False), (3, 32, False)]:
    m = quantize_model(copy.deepcopy(model).cpu(), bits=bits, group=gran, symmetric=sym)
    p = perplexity(m, ids)
    print(f"INT{bits} {str(gran):>7} sym={sym!s:5}  ppl {p:7.3f} ({100*(p/base-1):+5.1f}%)")
```

Expect the *shape* of this result (exact numbers vary): INT8 per-channel ≈ baseline; INT4 per-channel clearly worse; INT4 $g=128$ much better; asymmetric $g=32$ better still; INT3 starts to hurt. A 0.5B model is *more* sensitive than big ones — that’s realistic.

## Step 5 — honest memory: pack two INT4 values per byte

Our `q` is stored as one int8 per weight even for 4 bits, so memory only halves. Real formats pack:

```python
def pack_int4(q):            # uint8 codes in 0..15, last dim even
    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)
def unpack_int4(p):
    return torch.stack([p & 0x0F, p >> 4], dim=-1).flatten(-2)
```

Add a `bits == 4` branch to `QuantLinear` that stores `pack_int4(q)` (use asymmetric so codes are 0…15) and unpacks in `forward`. Verify `unpack_int4(pack_int4(q)).equal(q)` and recompute `model_bytes`. (If an op is missing on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.)

## Step 6 — speed: why yours is slow

```python
def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / iters * 1e3

lin = nn.Linear(4096, 11008, bias=False)                              # a Llama-7B-sized MLP matrix
W = lin.weight.data.half().to("mps")
x = torch.randn(1, 4096, dtype=torch.float16, device="mps")          # decode: one token
ql = QuantLinear(lin, bits=8, group=128).to("mps")
print("fp16 matvec      ", bench(lambda: F.linear(x, W)), "ms")
print("naive dequant+mm ", bench(lambda: ql(x)), "ms")
```

The naive version reads the INT8 codes **and** writes + re-reads a full FP16 matrix — so it’s slower than FP16. This is the “dequant overhead” from the why-faster lesson, measured. Optional: PyTorch has a fused per-channel INT8 kernel for CPU and (in recent builds) MPS, `torch._weight_int8pack_mm(x, q_int8, scales)` (codes shape `(N, K)`, one scale per output row) — try it and compare. The real fix is a fused kernel; writing one in Triton is a great M12 follow-up.

- [ ] `quantize`/`dequantize` pass a round-trip test: INT8 per-channel relative error < 1%
- [ ] Sweep table printed: bits × granularity × symmetric vs rel. weight error, output error, perplexity, bytes
- [ ] INT4 packing implemented; model memory for INT4 g128 is ≈ ¼ of fp16 Linear bytes + scales
- [ ] Timed fp16 vs naive dequant matvec and explained the result in two sentences
- [ ] (Stretch) implement a 30-line GPTQ-lite: quantize columns left→right and add each column’s error, scaled by `H_inv`, to the remaining columns; compare perplexity with RTN at INT4 g128
      */}),
      resources: [
        { title: "gpt-fast", url: "https://github.com/meta-pytorch/gpt-fast", type: "repo", note: "read quantize.py: int8 and int4 weight-only in a few hundred lines" },
        { title: "torchao", url: "https://github.com/pytorch/ao", type: "repo", note: "PyTorch-native production quantization with fused kernels" },
        { title: "WikiText dataset", url: "https://huggingface.co/datasets/Salesforce/wikitext", type: "docs", note: "the standard perplexity corpus (wikitext-2-raw-v1 test split)" },
      ],
    },
    {
      id: "lab-three-way",
      title: "Lab: MLX vs GGUF vs vLLM (GPTQ/FP8) — size, speed, quality",
      kind: "lab",
      minutes: 300,
      runsOn: ["mac", "colab", "cloud"],
      md: MD(function () {/*
One model (`Qwen2.5-1.5B-Instruct`), three toolchains, one results table. Rule: **each quantized variant is compared against a 16-bit baseline run through the *same* tool and harness.** (Read the pitfalls in the next lesson before trusting any number.)

## A shared client for speed + task accuracy

All three engines expose an OpenAI-compatible server, so one script measures all of them:

```python
# qbench.py  —  python qbench.py --base-url http://localhost:8080/v1 --model <served-name> --n 250
import argparse, math, re, statistics, time
from openai import OpenAI
from datasets import load_dataset

def decode_speed(c, model, runs=6):
    msg = [{"role": "user", "content": "Write a detailed 400-word explanation of how CPU caches work."}]
    out = []
    for _ in range(runs):
        stamps, usage = [], None
        s = c.chat.completions.create(model=model, messages=msg, stream=True, temperature=0,
                                      max_tokens=256, stream_options={"include_usage": True})
        for ch in s:
            if ch.choices and ch.choices[0].delta.content: stamps.append(time.perf_counter())
            if getattr(ch, "usage", None): usage = ch.usage
        n = usage.completion_tokens if usage else len(stamps)
        out.append((n - 1) / (stamps[-1] - stamps[0]))
    return statistics.median(out[1:])                           # drop the warm-up run

def gsm8k(c, model, n):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    results = []
    for ex in ds:
        r = c.chat.completions.create(model=model, temperature=0, max_tokens=512, messages=[{"role": "user",
            "content": ex["question"] + "\nThink step by step, then finish with a line 'Answer: <number>'."}])
        text = r.choices[0].message.content or ""
        gold = float(ex["answer"].split("####")[-1].replace(",", ""))
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"-?\d[\d,]*\.?\d*", text)
        try: ok = abs(float(m[-1].replace(",", "").rstrip(".")) - gold) < 1e-6
        except (IndexError, ValueError): ok = False
        results.append(ok)
    return results

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--base-url"); a.add_argument("--model")
    a.add_argument("--n", type=int, default=250); a.add_argument("--tag", default="run")
    args = a.parse_args(); c = OpenAI(base_url=args.base_url, api_key="x")
    tps = decode_speed(c, args.model)
    res = gsm8k(c, args.model, args.n); p = sum(res) / len(res)
    print(f"{args.tag}: decode {tps:.1f} tok/s | GSM8K {100*p:.1f}% ± {100*math.sqrt(p*(1-p)/len(res)):.1f}")
    open(f"{args.tag}.gsm8k.txt", "w").write("".join("1" if r else "0" for r in res))  # for paired comparison
```

## Part A — MLX on the Mac

```bash
M=Qwen/Qwen2.5-1.5B-Instruct
mlx_lm.convert --hf-path $M --mlx-path ~/models/q15-bf16 --dtype bfloat16
mlx_lm.convert --hf-path $M --mlx-path ~/models/q15-4bit -q --q-bits 4 --q-group-size 64
# quality: same data, same settings for both (relative change is what matters)
mlx_lm.perplexity --model ~/models/q15-bf16 --sequence-length 512 --num-samples 256
mlx_lm.perplexity --model ~/models/q15-4bit --sequence-length 512 --num-samples 256
# speed + task accuracy through the shared client
mlx_lm.server --model ~/models/q15-4bit --port 8080 &
python qbench.py --base-url http://localhost:8080/v1 --model ~/models/q15-4bit --tag mlx-4bit
```

Repeat the server + `qbench.py` step for `q15-bf16`. Optional: try `mlx_lm.awq` or `mlx_lm.dwq` for a learned 4-bit and see how much of the gap closes.

## Part B — llama.cpp GGUF on the Mac

```bash
brew install llama.cpp                                   # llama-quantize, llama-perplexity, llama-bench, llama-server
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
uv pip install -r ~/src/llama.cpp/requirements.txt       # deps for the HF→GGUF converter
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/q15-hf   # older hub: huggingface-cli download
python ~/src/llama.cpp/convert_hf_to_gguf.py ~/models/q15-hf --outtype f16 --outfile ~/models/q15-f16.gguf
llama-quantize ~/models/q15-f16.gguf ~/models/q15-Q4_K_M.gguf Q4_K_M
llama-quantize ~/models/q15-f16.gguf ~/models/q15-Q8_0.gguf Q8_0
llama-bench -m ~/models/q15-f16.gguf -m ~/models/q15-Q8_0.gguf -m ~/models/q15-Q4_K_M.gguf -p 512 -n 128
```

Perplexity **and KL divergence** against the f16 model (KLD compares full next-token distributions — more sensitive than PPL):

```bash
python -c "from datasets import load_dataset; d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test'); open('wiki.test.raw','w').write(''.join(d['text']))"
# --chunks limits the run: the saved logits file is ~vocab × tokens × 2 bytes (152k vocab → GBs quickly)
llama-perplexity -m ~/models/q15-f16.gguf -f wiki.test.raw -c 512 --chunks 40 --kl-divergence-base q15.kld
llama-perplexity -m ~/models/q15-Q4_K_M.gguf -c 512 --kl-divergence-base q15.kld --kl-divergence
llama-server -m ~/models/q15-Q4_K_M.gguf --port 8081 &
python qbench.py --base-url http://localhost:8081/v1 --model q15 --tag gguf-q4km
```

The KLD run also prints “Same top p” — how often the quantized model’s top token matches the f16 model’s.

## Part C — GPTQ W4A16 with llm-compressor + vLLM (Colab T4)

Quantize (restart the runtime afterwards — llm-compressor and vLLM may want different library versions):

```python
!pip install -q llmcompressor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier      # import paths move between releases: copy from the repo's examples/

MID = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(MID)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])   # INT4, group 128
oneshot(model=model, dataset="perfectblend", splits="train[:256]", recipe=recipe,
        max_seq_length=1024, num_calibration_samples=256)
model.save_pretrained("q15-W4A16", save_compressed=True); tok.save_pretrained("q15-W4A16")
```

Serve and measure (T4 has no BF16 — use `--dtype half`):

```bash
pip install -q vllm openai datasets lm_eval
nohup vllm serve ./q15-W4A16 --dtype half --max-model-len 4096 --port 8000 > w4.log 2>&1 &
python qbench.py --base-url http://localhost:8000/v1 --model ./q15-W4A16 --tag vllm-w4a16
# stop the server, then perplexity with lm-eval-harness (vLLM backend); repeat both for the fp16 baseline
lm_eval --model vllm --model_args pretrained=./q15-W4A16,dtype=half,max_model_len=4096,gpu_memory_utilization=0.8 \
        --tasks wikitext --batch_size auto
```

## Part D — FP8 (needs Ada/Hopper: Colab L4, or a rented L4/H100 for ~1 hour)

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --quantization fp8 --port 8000        # on-the-fly FP8 weights, no calibration
vllm serve Qwen/Qwen2.5-1.5B-Instruct --quantization fp8 --kv-cache-dtype fp8 --port 8000   # + FP8 KV cache
```

Or produce a calibrated W8A8 checkpoint with llm-compressor: `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])` and `oneshot(model=model, recipe=recipe)` (no dataset needed — activations are scaled dynamically per token).

## The results table

| Variant | Engine / HW | File size | Peak mem | Decode tok/s (b=1) | PPL (own harness) | Δ PPL vs own 16-bit | KLD / same-top-p | GSM8K ± SE |
|---|---|---|---|---|---|---|---|---|
| bf16 | MLX / Mac | | | | | — | | |
| 4-bit g64 | MLX / Mac | | | | | | | |
| f16 | llama.cpp / Mac | | | | | — | — | |
| Q8_0 | llama.cpp / Mac | | | | | | | |
| Q4_K_M | llama.cpp / Mac | | | | | | | |
| fp16 | vLLM / T4 | | | | | — | | |
| W4A16 GPTQ | vLLM / T4 | | | | | | | |
| FP8 (+FP8 KV) | vLLM / L4 or H100 | | | | | | | |
| your INT8 g128 | PyTorch / Mac | | | | | | | |

> [!WARNING] Don’t compare across rows blindly
> Absolute perplexities from `mlx_lm.perplexity` (default dataset), `llama-perplexity` (wikitext, ctx 512) and `lm_eval wikitext` (word perplexity) are **different quantities**. Compare the **Δ vs each tool’s own 16-bit baseline**. GSM8K through the shared client *is* comparable across rows. Tok/s is only comparable on the same hardware.

- [ ] Parts A and B complete on the Mac (4 variants), with Δ PPL and GSM8K for each
- [ ] Part C complete on Colab: GPTQ W4A16 vs fp16 under vLLM
- [ ] Part D (FP8) complete on an L4/H100, or skipped with a written reason
- [ ] Table filled; every speed number has a napkin bound next to it
- [ ] For the two 4-bit variants with the closest GSM8K scores, count questions that flipped right→wrong and wrong→right (the `.gsm8k.txt` files)
      */}),
      resources: [
        { title: "llm-compressor", url: "https://github.com/vllm-project/llm-compressor", type: "repo", note: "GPTQ/AWQ/SmoothQuant/FP8/NVFP4 → vLLM checkpoints; copy recipes from examples/" },
        { title: "llm-compressor docs", url: "https://docs.vllm.ai/projects/llm-compressor/en/latest/", type: "docs", note: "choosing a scheme per model and GPU" },
        { title: "vLLM — FP8 W8A8", url: "https://docs.vllm.ai/en/latest/features/quantization/fp8/", type: "docs", note: "online vs calibrated FP8, hardware requirements" },
        { title: "llama.cpp perplexity / KLD", url: "https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md", type: "docs", note: "how to measure PPL and KL divergence for GGUF quants" },
      ],
    },
    {
      id: "measuring-quality",
      title: "Measuring quality properly: perplexity, KL, lm-eval-harness, task evals, pitfalls",
      kind: "concept",
      minutes: 90,
      runsOn: ["any"],
      md: MD(function () {/*
The Baseten standard (ch. 5.1.3) is **zero perceptible loss**: differences from the unquantized model should be indistinguishable from run-to-run noise. You can only claim that with three kinds of evidence, “apples to apples”:

1. **Perplexity** on reference text (cheap, sensitive, abstract).
2. **Standard benchmarks** (MMLU, GSM8K, ARC, HumanEval… via lm-eval-harness).
3. **Custom evals** on *your* product’s traffic (the only one your users care about).

## 1. Perplexity and its stricter sibling, KL divergence

$$ \text{PPL} = \exp\!\Big(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t \mid x_{<t})\Big) $$

It only looks at the probability of the *correct* next token. Two models can have the same PPL but very different distributions elsewhere. **KL divergence** compares the whole next-token distribution of the quantized model $q$ against the original $p$ at every position:

$$ D_{\mathrm{KL}}(p \,\|\, q) = \sum_{v} p(v)\,\log\frac{p(v)}{q(v)} $$

(0 = identical.) llama.cpp reports mean/percentile KLD and “same top token %”. The **99th-percentile KLD** catches the rare positions where the quantized model goes badly wrong — those are what users notice.

::viz neg-log

## 2. lm-eval-harness

The community standard harness. Same tasks, same prompts, same scoring, many backends:

```bash
pip install lm_eval
# HF backend (works on Mac with --device mps, slowly)
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=float16 \
        --tasks arc_easy,hellaswag,wikitext --device mps --batch_size 8 --limit 500
# vLLM backend (fast, on GPU) — also loads GPTQ/AWQ/FP8 checkpoints
lm_eval --model vllm --model_args pretrained=./q15-W4A16,dtype=half --tasks gsm8k --batch_size auto
# MLX models
mlx_lm.evaluate --model ~/models/q15-4bit --tasks arc_easy --limit 500
```

Log-likelihood tasks (ARC, HellaSwag, MMLU) score the probability of each choice — cheap and low-variance, but insensitive to generation problems. Generative tasks (GSM8K, HumanEval) sample full answers — slower, closer to real use, and much better at exposing quantization damage (long reasoning chains compound errors).

## 3. Task evals: statistics you can’t skip

With $n$ questions and accuracy $p$, the standard error is

$$ \text{SE} = \sqrt{\frac{p(1-p)}{n}} $$

At $p = 0.65$, $n = 250$: SE ≈ 3.0 points. A “2-point drop” is **noise**. Two fixes:

- **More questions** (the full GSM8K test set has 1,319).
- **Paired comparison:** both models answer the *same* questions, so compare per question. Count $b$ = right→wrong and $c$ = wrong→right flips. If the models were equivalent, flips would split ~50/50; $b = 20, c = 18$ says nothing, $b = 30, c = 8$ is a real regression (a McNemar test formalizes this).

```python
a = open("mlx-bf16.gsm8k.txt").read(); b = open("mlx-4bit.gsm8k.txt").read()
lost = sum(x == "1" and y == "0" for x, y in zip(a, b)); gained = sum(x == "0" and y == "1" for x, y in zip(a, b))
print("right→wrong", lost, " wrong→right", gained)
```

## Pitfalls that invalidate quantization comparisons

| Pitfall | Why it bites | Do instead |
|---|---|---|
| Comparing absolute PPL across tools | different chunking, context length, token vs word PPL, datasets | Δ vs each tool’s own baseline |
| Different tokenizers or context lengths | PPL is per-token; changes the denominator | same tokenizer, same ctx |
| Calibrating on the eval set | GPTQ/AWQ calibrated on wikitext then evaluated on wikitext looks better than reality | calibrate on generic chat data; evaluate on held-out tasks |
| Different chat templates / system prompts / stop tokens | dominates small differences | one client, one template, `temperature=0` |
| Evaluating a different kernel than you serve | HF fake-quant ≠ Marlin ≠ Metal kernels numerically | evaluate on the engine + hardware you’ll serve |
| Only short tasks | errors compound with length; KV-cache quant hurts long contexts | include long generations & long-context tests |
| One run, no error bars | noise looks like signal | SE, paired flips, ≥ several hundred items |
| Averages only | rare catastrophic outputs hide in the mean | look at p99 KLD, read 20 diffs by hand |
| Ignoring your own traffic | benchmarks ≠ product | a custom eval from real (anonymized) prompts |

> [!REAL] “Recovery”
> Model vendors publish **recovery** = quantized score ÷ baseline score per benchmark (e.g. “99.5% average recovery” for an FP8 checkpoint). It’s a fine headline number — but ask which benchmarks, how many samples, and whether any single task dropped more than the noise.

- [ ] Computed SE for every GSM8K number in your lab table and marked which differences are significant
- [ ] Ran `lm_eval` on one baseline + one quantized variant for `arc_easy` and `gsm8k`
- [ ] Wrote a 20-prompt custom eval from a use case you care about (e.g. JSON extraction) with an automatic checker, and ran it on bf16 vs 4-bit
      */}),
      resources: [
        { title: "lm-evaluation-harness", url: "https://github.com/EleutherAI/lm-evaluation-harness", type: "repo", note: "the standard eval harness; HF, vLLM and API backends" },
        { title: "GSM8K", url: "https://huggingface.co/datasets/openai/gsm8k", type: "docs", note: "grade-school math; generative, sensitive to quantization damage" },
      ],
    },
  ],

  challenge: {
    title: "Quantization report: a Pareto chart and a shipping recommendation",
    md: MD(function () {/*
Write `course-work/m14/REPORT.md` (+ a notebook or scripts) that a staff engineer could use to pick a format.

1. **Data:** your lab table (≥6 variants across ≥2 engines, including your own INT8/INT4 quantizer) with file size, peak memory, decode tok/s at batch 1, Δ perplexity vs its own baseline, KLD where available, and a task score with standard error.
2. **Pareto chart:** x = decode tok/s (or tokens/s per GB of memory), y = task accuracy (or −ΔPPL), marker size = memory. Draw the **Pareto frontier** (variants not beaten on both axes). Label every point.
3. **Explain the frontier** with the roofline: why each point sits where it does (bytes, kernel, compute-bound or not).
4. **Recommendation for ONE scenario** (pick one, state the assumptions):
   - (a) an offline-capable writing assistant on a 16 GB MacBook Air;
   - (b) a customer-support chatbot on one L4 (24 GB), p95 TPOT < 50 ms, 20 concurrent users;
   - (c) a nightly batch job summarizing 5M documents on H100s, minimizing \$/1M tokens.
   Say which format you would ship, what you’d keep in higher precision, what eval gate must pass before rollout, and what you’d monitor after.

```python
import matplotlib.pyplot as plt
rows = [("bf16 MLX", 46, 66.0, 3.1), ("4-bit MLX", 125, 63.2, 0.9)]   # ... your data
def pareto(rows):   # higher speed and higher accuracy are both better
    return [r for r in rows if not any(o[1] >= r[1] and o[2] >= r[2] and o != r for o in rows)]
for name, tps, acc, gb in rows:
    plt.scatter(tps, acc, s=80 * gb); plt.annotate(name, (tps, acc))
f = sorted(pareto(rows), key=lambda r: r[1]); plt.plot([r[1] for r in f], [r[2] for r in f], "--")
plt.xlabel("decode tok/s (batch 1)"); plt.ylabel("GSM8K accuracy %"); plt.savefig("pareto.png")
```
    */}),
    checklist: [
      "≥6 variants across ≥2 engines, each compared against a 16-bit baseline from the same tool",
      "Every accuracy number has a standard error; paired right→wrong / wrong→right flips reported for the two closest candidates",
      "Pareto chart with the frontier drawn and every point labeled",
      "Each speed number has a napkin/roofline prediction and a one-line explanation of the gap",
      "Your own quantizer is in the table, with an honest explanation of why its speed is what it is",
      "A clear recommendation for one scenario, with the eval gate and the monitoring plan",
    ],
    stretch: "Write a fused dequant-matmul (INT4 g128, batch 1) in Triton on Colab (M12 skills) or with `mx.fast.metal_kernel` on your Mac, and show it beating the naive PyTorch version — ideally approaching the FP16 matvec time × (4.25/16).",
  },

  connects: MD(function () {/*
- **Back to M07/M12:** quantization is roofline engineering — fewer bytes for memory-bound work, faster tensor-core math for compute-bound work — and it only pays off with fused kernels (the Triton stretch goal).
- **Next, M15 (speculative decoding):** the other big decode lever. Quantization makes each weight read cheaper; speculation makes each weight read produce *more tokens*. They stack: quantized targets, quantized drafters (your 4-bit 0.5B is a great drafter), FP8 KV caches.
- **M16–M17:** FP8/FP4 halve the bytes sent in tensor-parallel all-reduces and KV transfers between prefill and decode nodes; fewer GPUs per replica means less communication.
- **M18–M19:** in production you’ll gate every new quantized checkpoint behind the eval suite you started here, and track \$/1M tokens — the business reason quantization exists.
  */}),

  interview: [
    "Why does weight-only INT4 speed up batch-1 decode but not prefill? At what point can W4A16 be *slower* than BF16?",
    "Walk through symmetric vs asymmetric quantization. What is the effective bits-per-weight of 4-bit with group size 128 and an FP16 scale?",
    "Why is per-channel weight scaling free at runtime but per-group scaling needs special kernels?",
    "What are activation outliers, where do they come from, and how do SmoothQuant and FP8 each deal with them?",
    "Explain GPTQ and AWQ in two sentences each. Which needs calibration data, and what can go wrong with it?",
    "FP8 E4M3 vs E5M2: which do you use for inference weights and why? What do MXFP4 and NVFP4 add?",
    "A new INT4 checkpoint shows +2% perplexity and −1.5 points on MMLU. Would you ship it? What else would you measure?",
    "Your 70B model runs in BF16 on 4×H100 with TP4. What changes (memory, batch size, latency, cost) if you move to FP8, and what could go wrong?",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — ch. 5.1 Quantization", url: "Inference%20Engineering.pdf", type: "book", note: "formats, granularity, sensitivity ranking, FP8 sweet spot, measuring quality" },
    { title: "A Visual Guide to Quantization (Maarten Grootendorst)", url: "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization", type: "article", note: "illustrated end-to-end overview; read first" },
    { title: "GPTQ", url: "https://arxiv.org/abs/2210.17323", type: "paper", note: "the canonical INT4 PTQ method" },
    { title: "AWQ", url: "https://arxiv.org/abs/2306.00978", type: "paper", note: "activation-aware weight quantization (MLSys best paper)" },
    { title: "SmoothQuant", url: "https://arxiv.org/abs/2211.10438", type: "paper", note: "W8A8 by smoothing activation outliers" },
    { title: "LLM.int8()", url: "https://arxiv.org/abs/2208.07339", type: "paper", note: "emergent outlier features; mixed-precision decomposition" },
    { title: "FP8 Formats for Deep Learning", url: "https://arxiv.org/abs/2209.05433", type: "paper", note: "E4M3 / E5M2 definitions" },
    { title: "QLoRA (NF4)", url: "https://arxiv.org/abs/2305.14314", type: "paper", note: "4-bit NormalFloat and double quantization; quantization meets fine-tuning" },
    { title: "llm-compressor", url: "https://github.com/vllm-project/llm-compressor", type: "repo", note: "production PTQ toolkit for vLLM" },
    { title: "NVIDIA TensorRT Model Optimizer", url: "https://github.com/NVIDIA/TensorRT-Model-Optimizer", type: "tool", note: "FP8/NVFP4/INT4 PTQ and QAT; the tool Baseten names" },
    { title: "llama.cpp quantization types", url: "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md", type: "docs", note: "GGUF k-quants with bpw and quality tables" },
    { title: "lm-evaluation-harness", url: "https://github.com/EleutherAI/lm-evaluation-harness", type: "repo", note: "how everyone measures quantized-model quality" },
  ],
});
