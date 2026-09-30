Course.module({
  id: "m04-modern-llms",
  title: "From GPT-2 to Llama/Qwen: modern architectures",
  short: "Modern LLMs (Llama/Qwen)",
  tagline: "Load real Qwen2.5 weights from Hugging Face into your own ~200-line PyTorch model, match the official logits to 1e-3, and understand every change since GPT-2: BPE, chat templates, RoPE, RMSNorm, SwiGLU, GQA/MLA and MoE.",
  hours: 16,
  level: "core",
  runsOn: ["mac"],
  tags: ["llama", "qwen", "rope", "gqa", "moe", "tokenizers", "safetensors"],

  goal: MD(function () {/*
You download **Qwen2.5-0.5B** from Hugging Face, and a model file **you wrote** — no `transformers` in the forward pass — produces the same numbers as the official implementation:

```text
$ python test_parity.py        # real output: transformers 5.11, torch 2.12, fp32 CPU
'The capital of France is'         T=  5  max|Δ|=0.00e+00  top-1 agree=100%
'def fibonacci(n):\n    '          T=  5  max|Δ|=0.00e+00  top-1 agree=100%
'In 1905, Albert Einstein publi'   T= 10  max|Δ|=0.00e+00  top-1 agree=100%
PARITY OK

$ python chat.py --device mps     # illustrative; your text and tok/s will differ
> Explain what a KV cache is in one sentence.
A KV cache stores the key and value vectors of previous tokens so the model
doesn't recompute them at every generation step.
[our model, fp16 on MPS, no KV cache: ~30 tok/s]
```

Yes, **zero**: with the same op order and the same `scaled_dot_product_attention` kernel on CPU, your file can be bit-identical to HF's. On other versions or devices, anything under `1e-3` passes.

Then you fill a table explaining, number by number, how **Llama-3-8B, Qwen2.5-7B, Mistral-7B and DeepSeek-V3** differ — layers, heads, KV heads, vocab, context, norm, activation, MoE — and what each choice costs at inference time.

::viz param-counter
  */}),
  demo: { viz: "param-counter" },

  why: MD(function () {/*
Every model an inference engineer serves in 2026 is a **GPT-2 with five or six swaps**: a bigger byte-level BPE vocabulary, a chat template, RoPE instead of learned positions, RMSNorm instead of LayerNorm, SwiGLU instead of GELU, grouped-query (or latent) attention and, for the big ones, Mixture-of-Experts. Each swap exists **partly for inference**. GQA and MLA shrink the KV cache. MoE cuts FLOPs per token. RoPE lets context be extended after training.

On the job you'll read a new model's `config.json` on release day and predict its memory footprint, KV cache per token and parallelism needs. You'll debug "the outputs are garbage" tickets, which are almost always a wrong chat template, a RoPE-scaling mismatch or a missing bias. And when vLLM, SGLang or TensorRT-LLM adds a new architecture, someone writes exactly the model file you write in this module. Inference Engineering (Baseten) ch. 2.2 frames this: *"an optimized deployment carries over to other variants of the same architecture."*
  */}),

  prereqs: [
    {
      title: "Module 03: your GPT from scratch",
      skipIf: "you have written a decoder-only transformer (attention, MLP, LayerNorm, residuals) in PyTorch",
      md: MD(function () {/*
This module is a **diff against your GPT**. You should be able to explain what `(B, T, C)` means, why attention uses a causal mask, and what `nn.Linear(C, 3*C)` does. If `model.transformer.h[0].attn.c_attn` means nothing to you, finish Module 03 first.
      */}),
    },
    {
      title: "Rotation, angles and the dot product (2-minute refresher)",
      skipIf: "you know that rotating two vectors by the same angle leaves their dot product unchanged",
      math: true,
      md: MD(function () {/*
- An **angle** can be measured in **radians**: a full turn is $2\pi \approx 6.283$ rad, so 1 rad ≈ 57°.
- For a point $(x, y)$ on a circle, $\cos\theta$ is the x-coordinate and $\sin\theta$ is the y-coordinate of the point at angle $\theta$ on the unit circle.
- The **dot product** $a \cdot b = a_x b_x + a_y b_y = |a|\,|b|\cos(\text{angle between them})$. It only cares about lengths and the **angle between** the vectors.
- So if you spin both vectors by the same amount, the angle between them doesn't change, and neither does their dot product. RoPE is built on this one fact.

::viz vectors-dot
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: run Qwen2.5 with transformers on your Mac, then open it up",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
Before building anything, run the real thing and look inside it. You'll use **Qwen2.5-0.5B-Instruct**: small enough for any Mac (1 GB in BF16), not gated (no license click-through, unlike Llama), and architecturally identical to its 7B/72B siblings.

```bash
uv pip install torch transformers safetensors accelerate huggingface_hub tiktoken
```

```python
# see_it.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps")  # older transformers: torch_dtype=

messages = [{"role": "user", "content": "Explain rotary position embeddings in two sentences."}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(repr(text))                                    # look at the special tokens!
inputs = tok(text, return_tensors="pt").to("mps")
out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.8)
print(tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Now open it up

```python
print(model.config)                                  # the architecture, as data
print(sum(p.numel() for p in model.parameters()))    # 494,032,768
for k, v in list(model.state_dict().items())[:14]:
    print(f"{k:50s} {str(tuple(v.shape)):16s} {v.dtype}")
```

```text
model.embed_tokens.weight                          (151936, 896)    torch.bfloat16
model.layers.0.self_attn.q_proj.weight             (896, 896)       torch.bfloat16
model.layers.0.self_attn.q_proj.bias               (896,)           torch.bfloat16
model.layers.0.self_attn.k_proj.weight             (128, 896)       torch.bfloat16
model.layers.0.self_attn.k_proj.bias               (128,)           torch.bfloat16
model.layers.0.self_attn.v_proj.weight             (128, 896)       torch.bfloat16
model.layers.0.self_attn.v_proj.bias               (128,)           torch.bfloat16
model.layers.0.self_attn.o_proj.weight             (896, 896)       torch.bfloat16
model.layers.0.mlp.gate_proj.weight                (4864, 896)      torch.bfloat16
model.layers.0.mlp.up_proj.weight                  (4864, 896)      torch.bfloat16
model.layers.0.mlp.down_proj.weight                (896, 4864)      torch.bfloat16
model.layers.0.input_layernorm.weight              (896,)           torch.bfloat16
model.layers.0.post_attention_layernorm.weight     (896,)           torch.bfloat16
model.layers.1.self_attn.q_proj.weight             (896, 896)       torch.bfloat16
```

Read these shapes like a detective. Compare them with your GPT-2 from Module 03:

| Clue | What it tells you | GPT-2 had |
|---|---|---|
| `k_proj` is `(128, 896)` but `q_proj` is `(896, 896)` | 14 query heads of 64 dims, but only **2 key/value heads**: grouped-query attention | one fused `c_attn` of `(768, 2304)`: equal Q, K, V |
| `gate_proj` **and** `up_proj`, then `down_proj` | a **gated** MLP (SwiGLU): three matrices instead of two | `c_fc` + `c_proj` with GELU |
| norms have only `.weight`, no `.bias` | **RMSNorm** | LayerNorm with weight + bias |
| no `wpe` (position embedding) tensor at all | positions are injected **inside attention** by RoPE, with no learned weights | `wpe` of `(1024, 768)` |
| `q/k/v_proj` have a bias, `o_proj` doesn't | a Qwen quirk (Llama has no biases anywhere) | biases everywhere |
| vocab 151,936 vs 50,257 | a bigger byte-level BPE, multilingual + code | 50,257 |
| no `lm_head.weight` in the checkpoint | `tie_word_embeddings: true`: output layer reuses the embedding matrix | tied as well |

> [!INTUITION] The big picture
> The **skeleton didn't change**: embed → N × (norm → attention → residual → norm → MLP → residual) → norm → LM head. What changed are the parts inside each box. This module takes them one at a time. At the end you rebuild the whole thing and prove it's right by matching logits.

## Where the parameters live

$494\text{M} = \underbrace{136\text{M}}_{\text{embeddings }151936\times 896} + 24 \times \underbrace{14.9\text{M}}_{\text{per layer}}$ and per layer the MLP is $3 \times 896 \times 4864 = 13.1\text{M}$, **88%** of the layer. In small models the embedding table is a huge fraction; in a 7B model the MLPs dominate (Baseten ch. 2.2.2: "the FFN linear layers hold most of the weights"). Play with the calculator:

::viz param-counter

- [ ] Ran `see_it.py` on MPS and printed the chat-templated prompt with `repr()`
- [ ] Printed `model.config` and found `num_key_value_heads`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`
- [ ] Verified 494,032,768 parameters by hand from the shapes above (embeddings + 24 × layer + final norm)
- [ ] Wrote down three differences from your Module 03 GPT before reading on
      */}),
      resources: [
        { title: "Qwen2.5-0.5B-Instruct model card", url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct", type: "docs", note: "the model you'll load; check the Files tab" },
        { title: "Hugging Face Transformers docs", url: "https://huggingface.co/docs/transformers/index", type: "docs", note: "reference implementation library (Baseten ch. 4.2.4)" },
      ],
    },
    {
      id: "checkpoint-anatomy",
      title: "Anatomy of a checkpoint on disk: config, tokenizer, safetensors shards (and GGUF, MLX)",
      kind: "concept",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
A "model" on Hugging Face is just a Git repo of files. Knowing each file cold is a day-one inference skill: engines read them directly, and half of all deployment bugs are a mismatch between them.

```bash
hf download Qwen/Qwen2.5-7B-Instruct --exclude "*.safetensors" --local-dir q7   # metadata only
ls -la q7   # (older huggingface_hub: huggingface-cli download ...)
```

| File | What's in it | Who reads it |
|---|---|---|
| `config.json` | architecture: `architectures`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `rope_theta`, `rope_scaling`, `max_position_embeddings`, `torch_dtype` | the engine, to build the right model class and size the KV cache |
| `generation_config.json` | default sampling (`temperature`, `top_p`, `repetition_penalty`) and **`eos_token_id`** (often a list!) | `generate()`; vLLM uses it for default sampling params |
| `tokenizer.json` | the whole fast tokenizer: vocab, BPE merges, pre-tokenizer regex, special tokens | Rust `tokenizers` library |
| `tokenizer_config.json` / `chat_template.jinja` | special-token config and the **Jinja chat template** | `apply_chat_template`, the OpenAI-compatible server |
| `vocab.json` + `merges.txt` | the same BPE, older "slow tokenizer" format | fallback |
| `model-0000X-of-0000N.safetensors` | the weights, **sharded** into ~4–5 GB files | `safetensors` loader (mmap) |
| `model.safetensors.index.json` | `weight_map`: tensor name → which shard file | loaders, to open only the shards they need |

`architectures: ["Qwen2ForCausalLM"]` is the key the engine uses to pick an implementation. Parse it like Baseten ch. 2.2.1 does: family (Qwen) + version (2) + optional `Moe` + `ForCausalLM` (predicts the next token).

## The safetensors format (you can parse it in 10 lines)

A `.safetensors` file is: **8 bytes** (little-endian u64 = header length N) → **N bytes of JSON** (name → dtype, shape, byte offsets) → **raw tensor bytes**. That's it. No pickle means no code runs when you load it, unlike `torch.load` on a `.bin`/`.pt` file, which can execute arbitrary Python.

```python
import json, struct, numpy as np
path = "model.safetensors"   # e.g. from hf download Qwen/Qwen2.5-0.5B
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
meta = header.pop("__metadata__", {})
for name, info in list(header.items())[:5]:
    print(name, info["dtype"], info["shape"], info["data_offsets"])

# read one tensor by hand: BF16 is the top 16 bits of an FP32
name = "model.norm.weight"; info = header[name]
start, end = info["data_offsets"]
raw = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + start, shape=((end - start) // 2,))
print(name, (raw.astype(np.uint32) << 16).view(np.float32)[:5])
```

Because offsets are known up front, a loader can **mmap** the file and hand tensors to the GPU without copying through Python. vLLM and SGLang load safetensors directly. Baseten ch. 4.2.2 describes the industry split: hand-written PyTorch model code plus engines that read safetensors, instead of exported graphs (ONNX/TensorRT engines).

> [!WARNING] Checkpoint gotchas you will hit
> **Tied embeddings**: `lm_head.weight` is absent, so reuse `embed_tokens.weight`. **Dtype**: weights are usually BF16, and loading them into FP16 can overflow on some models. **Sharding**: you must read all shards (use the index). **Renamed keys**: Meta's original Llama checkpoints (`consolidated.00.pth`) use different names *and* a different RoPE layout (see the RoPE lesson) from the HF version.

## Other formats you'll meet

| Format | Used by | What's different |
|---|---|---|
| **safetensors (HF)** | Transformers, vLLM, SGLang, TRT-LLM (via conversion) | one tensor per name, usually BF16; config and tokenizer in separate files |
| **GGUF** | llama.cpp, Ollama, LM Studio | **one self-contained file**: metadata (architecture, context length, tokenizer vocab + merges, chat template) + tensor index + aligned data; many quantization types (`Q4_K_M`, `Q8_0`, ...). Built for local/edge (Baseten ch. 5.1.1) |
| **MLX** | mlx-lm on Apple Silicon | safetensors + `config.json`, but quantized weights are packed `uint32` with per-group `scales`/`biases` (group size 64 by default) and a `"quantization"` entry in config |

```bash
# GGUF: convert + quantize with llama.cpp
python llama.cpp/convert_hf_to_gguf.py q05 --outfile qwen05-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize qwen05-f16.gguf qwen05-q4_k_m.gguf Q4_K_M

# MLX: convert + 4-bit quantize in one step
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path qwen05-mlx-4bit -q
cat qwen05-mlx-4bit/config.json | grep -A3 quantization
```

> [!REAL] Why engineers care
> When a customer says "deploy this fine-tune", your first 10 minutes are: read `config.json` (does our engine support this `architectures` value? what's the KV cache per token?), check `generation_config.json` for EOS tokens, diff the chat template against the base model's, and check the total size in `model.safetensors.index.json` against GPU memory. You'll build that checklist in the challenge.

- [ ] Downloaded only the metadata of Qwen2.5-7B-Instruct; read every JSON file
- [ ] Parsed a safetensors header by hand and decoded one BF16 tensor with numpy
- [ ] From `model.safetensors.index.json` → `metadata.total_size`, computed the model size in GB and checked it against params × 2 bytes
- [ ] (Optional) Converted Qwen2.5-0.5B-Instruct to MLX 4-bit and to GGUF Q4_K_M; compared file sizes
      */}),
      resources: [
        { title: "safetensors docs", url: "https://huggingface.co/docs/safetensors/index", type: "docs", note: "format spec and why not pickle" },
        { title: "GGUF on the HF Hub", url: "https://huggingface.co/docs/hub/gguf", type: "docs", note: "GGUF structure and quant types, with a link to the spec" },
        { title: "llama.cpp", url: "https://github.com/ggml-org/llama.cpp", type: "repo", note: "convert_hf_to_gguf.py and llama-quantize live here" },
      ],
    },
    {
      id: "tokenizers",
      title: "Tokenizers for real: byte-level BPE, special tokens and chat templates",
      kind: "concept",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
Your Module 03 GPT used characters: 65 symbols, very long sequences. Real models use **byte-level BPE** (Byte-Pair Encoding) with 32K–260K tokens. Tokenization happens **before** the model and **outside** the GPU, so it's easy to forget. It is also the cause of many weird model behaviours: bad arithmetic, trouble spelling, higher cost for non-English text, and trailing-space bugs.

## BPE in one paragraph

Start with the UTF-8 **bytes** of the text (256 base symbols, so no "unknown" tokens ever). Count every adjacent pair. **Merge the most frequent pair** into a new token. Repeat until you reach the vocabulary size you want. Encoding new text replays the merges in the order they were learned. Step through it:

::viz tokenizer-bpe

```python
# bpe.py — a minimal trainer (the core of karpathy/minbpe)
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

text = open("input.txt", encoding="utf-8").read()     # tiny Shakespeare from Module 03
ids = list(text.encode("utf-8"))
merges = {}
for step in range(256):                                # vocab 256 -> 512
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    ids = merge(ids, pair, 256 + step)
    merges[pair] = 256 + step
print(f"{len(text.encode())} bytes -> {len(ids)} tokens ({len(text.encode()) / len(ids):.2f}x compression)")
```

Production tokenizers add two things. A **pre-tokenization regex** splits text into words, numbers and punctuation first, so merges never cross those boundaries (GPT-4's pattern also splits numbers into 1–3 digit chunks). And **special tokens** are added on top of the merges.

## Compare real tokenizers

```python
import tiktoken
from transformers import AutoTokenizer
s = "Tokenization is 12345.67 ways weird: naïve 東京 def f(x):\n    return x"
enc = tiktoken.get_encoding("o200k_base")                 # GPT-4o
print(len(enc.encode(s)), [enc.decode([t]) for t in enc.encode(s)])
for name in ["Qwen/Qwen2.5-0.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
    t = AutoTokenizer.from_pretrained(name)
    print(name, len(t(s).input_ids), t.tokenize(s))      # 'Ġ' marks a leading space
```

| Tokenizer | Vocab | Notes |
|---|---|---|
| GPT-2 | 50,257 | byte-level BPE, the classic |
| Mistral-7B v0.3 | 32,768 | SentencePiece BPE, small vocab: more tokens per word |
| Llama 3 | 128,256 | tiktoken-style BPE (100K from OpenAI's cl100k + 28K extra) |
| Qwen2.5 | 151,643 + special (embedding padded to 151,936) | byte-level BPE, strong on Chinese and code |
| GPT-4o (`o200k_base`) | ~200K | tiktoken |

> [!INTUITION] The vocabulary-size trade-off
> A bigger vocab means fewer tokens per request, so less prefill compute, fewer decode steps and a smaller KV cache. The price is a bigger embedding table and LM head (Qwen2.5-0.5B spends 28% of its parameters there!) and a larger softmax. For large models the trade is clearly worth it. That's why vocabularies grew from 32K to 128K–260K.

## Special tokens and chat templates

An instruct model was fine-tuned on conversations serialized in **one exact format**. Send it anything else and quality collapses silently. Qwen uses ChatML:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
```

Llama 3 uses a different one: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`. The template is a **Jinja program** shipped in the tokenizer config. Baseten ch. 2.2 calls applying it "step zero" of inference.

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
msgs = [{"role": "user", "content": "Hi!"}]
print(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
print(tok.convert_ids_to_tokens(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)))
print(tok.chat_template[:400])            # read the Jinja
print(tok.eos_token, tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))   # 151645, 151643
```

> [!WARNING] The classic chat-template bugs
> 1. **No template**: sending raw text to an instruct model. It "sort of works" and quietly scores 10–30% worse.
> 2. **Wrong stop token**: Qwen-Instruct ends turns with `<|im_end|>` (151645), but the base tokenizer's EOS is `<|endoftext|>` (151643). Stop on only one and the model rambles into a fake next turn. `generation_config.json` lists **both**.
> 3. **Double BOS**: the template already adds `<|begin_of_text|>` and then the tokenizer adds another.
> 4. **Special-token injection**: user text containing the literal string `<|im_start|>` must *not* be parsed as the special token.
> 5. **Fine-tune/serve mismatch**: you trained with one template and serve with another (you'll hit this in Module 05).

> [!REAL] Tokenization in serving engines
> vLLM's OpenAI-compatible server applies the chat template server-side, tokenizes on the CPU, and does **incremental detokenization**. It must hold back bytes when a token ends in the middle of a multi-byte UTF-8 character, or streams show `�`. At high throughput, tokenization and detokenization become real CPU costs. vLLM V1 runs them in a separate process from the GPU loop.

- [ ] Watched Karpathy's "Let's build the GPT Tokenizer" (at least up to the minbpe implementation)
- [ ] Trained `bpe.py` to vocab 512 on tiny Shakespeare; recorded the compression ratio; wrote `encode`/`decode` and checked `decode(encode(s)) == s`
- [ ] Counted tokens for the same English, Hindi/Chinese and Python strings under three tokenizers; noted the "language tax"
- [ ] Printed Qwen's and Llama's (or Mistral's) chat templates; generated with and without the template and compared
- [ ] Explained why `generation_config.json` has a list of EOS ids
      */}),
      resources: [
        { title: "Karpathy — Let's build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE", type: "video", note: "2h13m; the definitive BPE walkthrough" },
        { title: "karpathy/minbpe", url: "https://github.com/karpathy/minbpe", type: "repo", note: "minimal BPE + GPT-4 regex tokenizer; do the exercise.md" },
        { title: "openai/tiktoken", url: "https://github.com/openai/tiktoken", type: "repo", note: "production BPE used by GPT-4/4o and Llama 3" },
        { title: "Tiktokenizer (web)", url: "https://tiktokenizer.vercel.app/", type: "tool", note: "see tokens live for many tokenizers" },
        { title: "Transformers — Chat templates", url: "https://huggingface.co/docs/transformers/chat_templating", type: "docs", note: "apply_chat_template, generation prompts, Jinja" },
      ],
    },
    {
      id: "rope",
      title: "Math: from learned positions to RoPE (rotary position embeddings)",
      kind: "math",
      minutes: 120,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Attention by itself has **no idea of order**: shuffle the tokens and each token gets the same output, just shuffled. GPT-2 fixed this by **adding a learned vector per position** (`wpe`, 1,024 rows). Two problems follow. The model has never seen position 1,025, so it can't go longer. And "token 7 attends to token 5" and "token 1007 attends to token 1005" look completely different to it, even though the relationship (2 tokens back) is the same.

**RoPE** (Su et al., 2021) is used by Llama, Qwen, Mistral, DeepSeek, Gemma and GPT-OSS. It encodes position by **rotating** the query and key vectors. The dot product then depends only on the **relative** distance between tokens.

> [!PREREQ] Rotating a 2-D vector
> Rotating $(x, y)$ counter-clockwise by angle $\theta$ gives
> $$\begin{pmatrix}x'\\y'\end{pmatrix} = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}$$
> Example: $(1, 0)$ rotated by $90°$ ($\theta = \pi/2$, so $\cos = 0$, $\sin = 1$) gives $(0, 1)$. Rotating preserves length.

> [!PREREQ] Complex numbers, lite
> Write the point $(x, y)$ as one number $z = x + iy$, where $i^2 = -1$. Euler's formula says $e^{i\theta} = \cos\theta + i\sin\theta$, a point on the unit circle. **Multiplying by $e^{i\theta}$ rotates $z$ by $\theta$**, and angles **add** when you multiply: $e^{i\alpha}e^{i\beta} = e^{i(\alpha+\beta)}$. The dot product of two 2-D vectors $a, b$ is $\mathrm{Re}(a\,\bar b)$, where $\bar b = b_x - i b_y$ is the conjugate. That's all the complex-number machinery RoPE needs.

## The idea in 2-D

Rotate the query of the token at position $m$ by angle $m\theta$. Rotate the key at position $n$ by $n\theta$. Then:

$$
\langle R_{m\theta}\,q,\; R_{n\theta}\,k\rangle
= \mathrm{Re}\!\left(q e^{im\theta}\;\overline{k e^{in\theta}}\right)
= \mathrm{Re}\!\left(q\,\bar k\, e^{i(m-n)\theta}\right)
$$

The absolute positions $m$ and $n$ have disappeared. **Only $m - n$ is left.** Rotating both vectors by the same extra amount (shifting the whole sentence) can't change the angle between them.

**Worked example.** Take $q = k = (1, 0)$ and $\theta = 0.5$ rad per position.
- Query at $m = 3$, key at $n = 1$: angles $1.5$ and $0.5$, gap $1.0$ rad → dot $= \cos(1.0) = 0.540$.
- Query at $m = 10$, key at $n = 8$: angles $5.0$ and $4.0$, gap $1.0$ → dot $= 0.540$. **Same.**
- Query at $m = 10$, key at $n = 1$: gap $4.5$ → dot $= \cos(4.5) = -0.211$. Different distance, different score.

::viz rope

## From 2-D to a 64-dimensional head

A head vector has $d$ dimensions (64 for Qwen2.5-0.5B, 128 for most 7B+ models). RoPE splits it into $d/2$ **pairs** and rotates pair $j$ at its own speed:

$$
\theta_j = \text{base}^{-2j/d}, \qquad j = 0, 1, \dots, d/2 - 1
$$

Think of clock hands. Pair 0 turns 1 rad per token (a second hand; it resolves "which neighbour"). The last pair turns extremely slowly (an hour hand; it tells "roughly how far back"). For Qwen2.5 (`rope_theta` = base = 1,000,000, $d = 64$), the slowest pair has $\theta_{31} = 10^{6 \cdot (-62/64)} \approx 1.5\times10^{-6}$ rad per token: one full turn every ~4 million tokens. **A bigger base means slower hands, which means longer usable context.** That's why `rope_theta` grew from 10,000 (original Llama) to 500,000 (Llama 3) and 1,000,000 (Qwen2.5, Mistral v0.3).

In code, this is the half-split form used by Hugging Face. Pair $j$ is dimensions $(j, j + d/2)$:

```python
def rope_cache(head_dim, max_pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # θ_j, shape (d/2,)
    angles = torch.outer(torch.arange(max_pos).float(), inv_freq)                 # m·θ_j, shape (T, d/2)
    angles = torch.cat([angles, angles], dim=-1)                                   # (T, d)
    return angles.cos(), angles.sin()

def rotate_half(x):                     # (x1, x2) -> (-x2, x1): the "multiply by i" step
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):            # x: (B, H, T, d)
    return x * cos + rotate_half(x) * sin    # = [x1 cos - x2 sin,  x2 cos + x1 sin]: the 2-D rotation
```

Check the last line against the rotation matrix above. It is literally $x' = x\cos - y\sin$, $y' = y\cos + x\sin$, applied to all $d/2$ pairs at once.

> [!WARNING] Interleaved vs half-split, the #1 porting bug
> Meta's original Llama code pairs **adjacent** dims $(0,1), (2,3), \dots$ using complex numbers (`torch.view_as_complex`). HF pairs $(j, j + d/2)$ and **permutes the rows of `q_proj`/`k_proj`** during conversion so the two agree. Mix one convention's weights with the other's `apply_rope` and you get fluent-looking garbage. The logits-parity test in the build lesson catches this instantly.

## Where RoPE applies, and what it costs at inference

- Only to **Q and K**, never V. It's applied *after* the projection and *before* the dot product, in every layer.
- RoPE has **no parameters**. `cos`/`sin` are precomputed per position, and inference engines fuse RoPE into the QKV kernel.
- The KV cache stores keys **already rotated**, so a cached key never needs re-rotation.
- **Context extension**: Position Interpolation, NTK-aware scaling and **YaRN** rescale the $\theta_j$ so a model trained at 8K–32K works at 128K. That's the `rope_scaling` field in `config.json` (Llama 3.1: `"rope_type": "llama3"`; Qwen2.5 128K: YaRN). Longer context means a bigger KV cache (Baseten ch. 5.3). Your engine must implement the exact same scaling, or long prompts silently degrade.

<details><summary>Go deeper: why not ALiBi or no positional encoding?</summary>

ALiBi (Press et al., 2021) adds a distance-proportional penalty to attention scores. It's simple and extrapolates well, but RoPE won on quality at scale. Some recent models mix **NoPE** layers (no positional encoding) with RoPE or sliding-window layers to improve long-context behaviour. Llama 4 and a few others do this. The EleutherAI blog post below derives RoPE carefully and compares it with the alternatives.

</details>

- [ ] Reproduced the worked example numerically: implement `apply_rope` for $d = 2$ and show that the score depends only on $m - n$
- [ ] For $d = 64$ and a random $q, k$, plotted score vs $m - n$ for base 10,000 and 1,000,000; described the difference
- [ ] Computed the wavelength $2\pi/\theta_j$ of the fastest and slowest pairs for Llama 3 (base 500,000, $d = 128$)
- [ ] Explained to yourself why the cached keys don't need re-rotation
      */}),
      resources: [
        { title: "RoFormer: Enhanced Transformer with Rotary Position Embedding", url: "https://arxiv.org/abs/2104.09864", type: "paper", note: "the RoPE paper (sections 3.2–3.4)" },
        { title: "EleutherAI — Rotary Embeddings: A Relative Revolution", url: "https://blog.eleuther.ai/rotary-embeddings/", type: "article", note: "the clearest derivation, with code" },
      ],
    },
    {
      id: "rmsnorm-swiglu",
      title: "RMSNorm and SwiGLU: the cheaper norm and the gated MLP",
      kind: "concept",
      minutes: 75,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Two small swaps, both in every modern model. Neither is exotic. You'll implement each in about 5 lines.

## RMSNorm: LayerNorm minus the mean

LayerNorm (GPT-2) subtracts the mean, divides by the standard deviation, then scales by $\gamma$ and shifts by $\beta$. **RMSNorm** (Zhang & Sennrich, 2019) drops the mean subtraction and the shift:

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\qquad
\text{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

**Worked example**, $x = [2, -1, 3, 0]$: $\text{mean}(x^2) = (4 + 1 + 9 + 0)/4 = 3.5$, $\text{rms} = \sqrt{3.5} = 1.871$, so $x / \text{rms} = [1.069, -0.535, 1.604, 0]$. The vector is rescaled to unit RMS, but its direction and "centre" are kept. It matches LayerNorm's quality and is cheaper: one reduction instead of two, and no bias.

::viz normalize

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                                  # reduce in fp32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)
```

> [!IMPORTANT] Details that break parity
> Upcast to fp32 for the reduction (HF does). Use the exact `eps` from config (`rms_norm_eps`: 1e-6 for Qwen2.5, 1e-5 for Llama 3). Multiply by `weight` **after** casting back (HF's order). Gemma is different again: it multiplies by `(1 + weight)`. Always read the reference code.

**Placement.** Modern models are **pre-norm**, like GPT-2: `x = x + attn(norm(x))`. Newer twists: **QK-norm** (an RMSNorm on each head's queries and keys, used in Qwen3, OLMo 2 and Gemma 3) keeps attention logits from exploding in training. Some models also add extra post-norms (Gemma, OLMo 2).

## SwiGLU: a gated MLP

GPT-2's MLP is `c_proj(gelu(c_fc(x)))` with hidden size $4d$. Llama/Qwen use a **gated linear unit** with the SiLU (a.k.a. Swish) activation, $\text{SiLU}(z) = z \cdot \sigma(z)$:

$$
\text{MLP}(x) = W_\text{down}\big(\,\text{SiLU}(W_\text{gate}\,x) \odot W_\text{up}\,x\,\big)
$$

One branch computes a smooth "gate" per hidden unit. The other computes the value. They're multiplied element-wise. Shazeer's "GLU Variants Improve Transformer" (2020) showed it beats ReLU/GELU at equal compute. It has **three** matrices instead of two, so the hidden size is cut to about $\tfrac{8}{3}d$ to keep parameters constant. In practice it's often rounded up: Llama-3-8B uses $4096 \to 14336$ (3.5×) and Qwen2.5-0.5B uses $896 \to 4864$ (5.4×).

::viz activations

```python
class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj   = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

## What this means for inference

The MLP is where most FLOPs and bytes go. Llama-3-8B per layer: MLP $3 \times 4096 \times 14336 = 176\text{M}$ params vs attention $2 \times 4096^2 + 2 \times 4096 \times 1024 = 42\text{M}$. That's **81% MLP**. So during decode (memory-bound, see Module 00) the MLP weights are most of the bytes you stream per token.

> [!REAL] What engines actually run
> vLLM's Llama/Qwen code **merges `gate_proj` and `up_proj` into one `gate_up_proj` matmul** (one kernel launch, one big GEMM) and then calls a fused `SiluAndMul` kernel. It also merges `q_proj/k_proj/v_proj` into one `qkv_proj`. RMSNorm is fused with the residual add (`fused_add_rms_norm`): it's purely memory-bound, so fusing saves a full read+write of the hidden state. The math is identical to your 5-line versions; the kernels are not. You'll write these fusions in Modules 11–13.

- [ ] Implemented `RMSNorm` and checked it against `torch.nn.RMSNorm` (PyTorch ≥ 2.4) on random input
- [ ] Hand-computed the worked example, then verified it in PyTorch
- [ ] Counted MLP vs attention params per layer for Qwen2.5-0.5B and Llama-3-8B
- [ ] Wrote a one-line explanation of why merging `gate_proj` and `up_proj` is legal (hint: stack the weight matrices)
      */}),
      resources: [
        { title: "Root Mean Square Layer Normalization", url: "https://arxiv.org/abs/1910.07467", type: "paper", note: "the RMSNorm paper" },
        { title: "GLU Variants Improve Transformer (Shazeer)", url: "https://arxiv.org/abs/2002.05202", type: "paper", note: "4 pages; SwiGLU's origin" },
      ],
    },
    {
      id: "attention-variants",
      title: "Attention variants for inference: MHA → MQA → GQA → MLA",
      kind: "concept",
      minutes: 90,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
Here is the inference-engineering reason these variants exist. During decode, every new token must **read the entire KV cache** of its sequence, in every layer. The KV cache also decides how many users fit on a GPU. So architects have spent five years **shrinking K and V**.

$$
\text{KV bytes per token} = 2 \;(\text{K and V}) \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes per value}
$$

## The four designs

| Variant | K/V heads | Example | KV per token (BF16) |
|---|---|---|---|
| **MHA** (multi-head) | = query heads | GPT-2, Llama-2-7B (32/32) | Llama-2-7B: $2\cdot32\cdot32\cdot128\cdot2$ = **512 KB** |
| **MQA** (multi-query, Shazeer 2019) | **1** | PaLM, Falcon-7B | 32× smaller than MHA; some quality loss |
| **GQA** (grouped-query, Ainslie 2023) | a few groups | Llama-3-8B (32 Q / 8 KV), Qwen2.5-7B (28 / 4) | Llama-3-8B: $2\cdot32\cdot8\cdot128\cdot2$ = **128 KB** |
| **MLA** (multi-head latent, DeepSeek-V2/V3) | a compressed **latent** of 512 dims + 64 RoPE dims, shared by all heads | DeepSeek-V3, Kimi K2 | $(512+64)\cdot61\cdot2$ ≈ **69 KB** for a 671B model |

::viz gqa

**GQA** is the pragmatic middle. Each group of query heads shares one K/V head. Llama-3-8B's 32 query heads form 8 groups of 4. Quality is close to MHA and the cache is 4× smaller. Your Qwen2.5-0.5B has 14 query heads sharing **2** KV heads: $2 \cdot 24 \cdot 2 \cdot 64 \cdot 2 = 12$ KB per token, so a 32K-token conversation costs just 384 MB.

In code, GQA is just "repeat each K/V head before attention":

```python
rep = n_heads // n_kv_heads                          # 7 for Qwen2.5-0.5B
k = k.repeat_interleave(rep, dim=1)                  # (B, n_kv, T, d) -> (B, n_heads, T, d)
v = v.repeat_interleave(rep, dim=1)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# PyTorch >= 2.5 can skip the copy:  F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

The repeat is a training/reference convenience. Real decode kernels (FlashAttention, FlashInfer) **never materialize the repeated K/V**: each group of query heads reads the same K/V tile from memory. That's where the bandwidth saving comes from.

**MLA** goes further. Instead of caching K and V, it caches one small **latent vector** $c^{KV}_t$ (512 dims) per token per layer, plus a small decoupled RoPE key (64 dims). The per-head K and V are **up-projected from the latent** when needed. A trick called *weight absorption* folds the up-projection into the query and output matrices, so decode attention works directly in the latent space. The result: DeepSeek-V3 has 128 heads and a cache ~70× smaller than MHA would need, with quality at or above MHA. The cost is complexity: special kernels (FlashMLA) and awkward tensor parallelism. Baseten ch. 4.2–4.3 notes that MLA is hard to export and needs day-zero engine support (SGLang was an early leader for DeepSeek models).

## Play with real numbers

::viz kv-calculator

| Model | layers × kv_heads × head_dim | KV/token | 32K context, 1 sequence |
|---|---|---|---|
| Qwen2.5-0.5B | 24 × 2 × 64 | 12 KB | 0.4 GB |
| Llama-3-8B | 32 × 8 × 128 | 128 KB | 4 GB |
| Qwen2.5-7B | 28 × 4 × 128 | 56 KB | 1.8 GB |
| Llama-3-70B | 80 × 8 × 128 | 320 KB | 10 GB (Baseten ch. 5.3: 40 GB at 128K) |

> [!INTUITION] Why this matters more than it looks
> On an 80 GB H100 serving Llama-3-8B (16 GB of weights), ~60 GB is left for KV. At 128 KB/token that's ~490K tokens: 60 concurrent 8K-token chats. With MHA it would be 15. Concurrency is throughput, and throughput is cost per token. **Architecture choices are pricing decisions.** You'll feel this in Module 08 when you build a paged KV cache.

<details><summary>Other attention tricks you'll see in configs</summary>

- **Sliding-window attention** (Mistral-7B v0.1, Gemma 2/3, GPT-OSS): some or all layers attend to only the last W tokens (e.g. 4096), so their KV cache stops growing. `sliding_window` / `layer_types` in config.
- **Hybrid linear attention** (Qwen3-Next, MiniMax, Kimi Linear): most layers use a recurrent/linear-attention state of fixed size, and a few use full attention. KV grows only in the full layers.
- **Attention sinks**: the first tokens soak up attention mass. Keeping them lets sliding windows work (StreamingLLM; GPT-OSS has learned sink logits).

</details>

- [ ] Computed KV bytes/token for Qwen2.5-0.5B, Llama-3-8B, Qwen2.5-7B, Mistral-7B from their `config.json` files (write a 10-line script that reads config and prints it)
- [ ] Implemented GQA both ways (`repeat_interleave` vs `enable_gqa=True`) and checked the outputs match
- [ ] For a 24 GB GPU and Llama-3-8B in BF16, computed max concurrent 4K-token sequences with GQA vs a hypothetical MHA version
- [ ] Read the GQA paper's abstract + Figure 2, and the MLA section (2.1) of DeepSeek-V2
      */}),
      resources: [
        { title: "Fast Transformer Decoding: One Write-Head is All You Need (MQA)", url: "https://arxiv.org/abs/1911.02150", type: "paper", note: "Shazeer's MQA paper; frames attention as a memory-bandwidth problem" },
        { title: "GQA: Training Generalized Multi-Query Transformer Models", url: "https://arxiv.org/abs/2305.13245", type: "paper", note: "the GQA paper; short" },
        { title: "DeepSeek-V2 (introduces MLA)", url: "https://arxiv.org/abs/2405.04434", type: "paper", note: "section 2.1: multi-head latent attention" },
      ],
    },
    {
      id: "moe",
      title: "Mixture of Experts: 671B parameters, 37B per token",
      kind: "concept",
      minutes: 90,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
The largest open models (DeepSeek-V3/R1, Kimi K2, Qwen3-235B, GPT-OSS-120B, Llama 4) are **Mixture-of-Experts** (MoE) models. The idea: replace the one big MLP in each block with **many small MLPs ("experts")** plus a tiny **router** that picks the top-$k$ experts for each token. Parameters (knowledge) grow a lot. FLOPs per token barely move.

::viz moe-routing

## The layer in 25 lines

```python
class MoE(nn.Module):
    def __init__(self, d, hidden, n_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(d, hidden) for _ in range(n_experts))   # SwiGLU MLPs from before
        self.top_k = top_k

    def forward(self, x):                                  # x: (B, T, d)
        B, T, d = x.shape
        x = x.view(-1, d)                                  # N = B*T tokens
        probs = F.softmax(self.router(x), dim=-1)          # (N, E)
        w, idx = probs.topk(self.top_k, dim=-1)            # (N, k): which experts, what weight
        w = w / w.sum(-1, keepdim=True)                    # renormalize (Mixtral/Qwen3 style)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):          # loop over experts, not tokens
            tok, slot = (idx == e).nonzero(as_tuple=True) # tokens routed to expert e
            if tok.numel():
                out[tok] += w[tok, slot, None] * expert(x[tok])
        return out.view(B, T, d)
```

## Active vs total parameters

| Model | Total | Active / token | Experts | Notes |
|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 8, top-2 | the model that popularized open MoE |
| Qwen3-30B-A3B | 30.5B | 3.3B | 128, top-8 | runs like a 3B, knows like a 30B |
| Qwen3-235B-A22B | 235B | 22B | 128, top-8, 94 layers | Baseten ch. 2.2.4's example |
| DeepSeek-V3 / R1 | 671B | 37B | 256 routed + 1 shared, top-8; first 3 layers dense | MLA + MoE + multi-token prediction |

- **Compute** per token scales with **active** params: training FLOPs ≈ $6 \times N_\text{active}$ per token (Module 05). DeepSeek-V3 trained on 14.8T tokens for ~2.8M H800-hours, far cheaper than a dense 671B would be.
- **Memory** scales with **total** params: all 671B must sit somewhere (≈ 671 GB in FP8, so a whole 8×H200 node).
- **Decode at batch 1** reads only the chosen experts, so bytes/token ∝ active params. That's why Qwen3-30B-A3B feels fast on a Mac (if it fits in memory).
- **Decode at large batch**: different tokens pick different experts, so **the union** of chosen experts covers nearly all of them, and you read nearly all the weights anyway. Baseten ch. 2.2.4: "in batched production, expect almost all experts to be active unless you use large-scale EP" (expert parallelism: spread experts across GPUs, all-to-all tokens to them; Modules 16–17).

## Load balancing

If the router sends most tokens to a few experts, those experts become the bottleneck (and in EP, their GPUs sit hot while others idle). Training adds an **auxiliary load-balancing loss** (Switch Transformer) or, in DeepSeek-V3, an *aux-loss-free* per-expert bias that nudges routing toward under-used experts. At inference, engines add **EPLB** (expert-parallel load balancer): replicate hot experts on extra GPUs.

> [!INTUITION] When MoE is (and isn't) the answer
> MoE wins when you're **compute-bound or training-cost-bound** and have memory to spare: frontier scale, big clusters. It's awkward when **memory-bound on few GPUs** (you pay memory for all experts) and in narrow domains. Baseten: models under ~32B, especially under 8B, stay dense. Recognize MoE in `config.json` by `num_experts` / `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`, and `Moe` in the `architectures` name.

- [ ] Implemented the `MoE` layer above; routed 1,000 random tokens through 8 experts and plotted the per-expert token counts
- [ ] Computed total and active params for Qwen3-30B-A3B from its `config.json` (hint: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`)
- [ ] Estimated batch-1 decode tok/s for Qwen3-30B-A3B (4-bit) vs Qwen2.5-32B (4-bit) on your Mac with the Module 00 napkin formula, then measured one if memory allows
- [ ] Explained why batch-64 decode of an MoE reads almost all expert weights
      */}),
      resources: [
        { title: "Mixtral of Experts", url: "https://arxiv.org/abs/2401.04088", type: "paper", note: "clean description of a top-2 sparse MoE" },
        { title: "Switch Transformers", url: "https://arxiv.org/abs/2101.03961", type: "paper", note: "top-1 routing, capacity factor, load-balancing loss" },
        { title: "DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "MLA + fine-grained MoE + aux-loss-free balancing + FP8 training" },
      ],
    },
    {
      id: "build-qwen",
      title: "Build: your own Qwen2/Llama model file, safetensors loader and logits-parity test",
      kind: "build",
      minutes: 240,
      runsOn: ["mac"],
      md: MD(function () {/*
Now assemble everything into **one file, ~150 lines**, that loads the official weights and matches Hugging Face to within $10^{-3}$ in FP32. This is exactly what an engine engineer does when adding a model to vLLM or SGLang.

> [!BUILD] Plan
> 1. `Config` read from `config.json` → 2. `RMSNorm`, RoPE, `Attention` (GQA), `MLP` (SwiGLU), `Block` → 3. `CausalLM` whose **module names match the checkpoint keys** → 4. `load()` that reads every safetensors shard with `strict=True` → 5. parity test vs `transformers` → 6. sampling loop on MPS.

## Step 1–4: `qwen2.py`

```python
# qwen2.py — a from-scratch Qwen2 / Llama decoder that loads Hugging Face safetensors.
import json
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: int = 0

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text())
        c = cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw and raw[k] is not None})
        # Qwen2 hard-codes bias=True on q/k/v; its config.json doesn't say so.
        c.attention_bias = raw.get("attention_bias", raw["model_type"] == "qwen2")
        c.head_dim = c.head_dim or c.hidden_size // c.num_attention_heads
        return c

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

def rope_cache(head_dim, max_pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_pos).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=c.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # (B, nh,  T, hd)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)   # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, self.nh * self.hd))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)
    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))

class CausalLM(nn.Module):
    def __init__(self, c, max_pos=4096):
        super().__init__()
        self.config = c
        self.model = nn.Module()            # mirrors the "model." prefix in checkpoint keys
        self.model.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.model.layers = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        cos, sin = rope_cache(c.head_dim, min(max_pos, c.max_position_embeddings), c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):                 # (B, T) token ids -> (B, T, vocab) logits
        T = idx.shape[1]
        dt = self.lm_head.weight.dtype
        cos, sin = self.cos[:T].to(dt), self.sin[:T].to(dt)
        x = self.model.embed_tokens(idx)
        for layer in self.model.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.model.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :].float()   # no KV cache: recompute everything (Module 06 fixes this)
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and nxt.item() in (eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]):
                break
        return idx

def load(repo_or_dir, dtype=torch.float32, device="cpu"):
    path = Path(repo_or_dir)
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_or_dir, allow_patterns=["*.json", "*.safetensors"]))
    cfg = Config.from_json(path / "config.json")
    model = CausalLM(cfg)
    state = {}
    for shard in sorted(path.glob("*.safetensors")):
        state.update(load_file(shard))
    if cfg.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
    model.load_state_dict(state, strict=True)     # fails loudly on any missing or extra key
    return model.to(device=device, dtype=dtype).eval()
```

Notice the trick in `CausalLM`: attribute names (`model.layers.0.self_attn.q_proj`) are **chosen to match the checkpoint keys**, so `load_state_dict` needs no renaming map. vLLM model files do the same, plus a `load_weights()` that handles the few renames and merges (`q/k/v → qkv_proj`).

## Step 5: the parity test

```python
# test_parity.py — ours vs Hugging Face, FP32 on CPU
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen2 import load

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(REPO)
ours = load(REPO, dtype=torch.float32)
ref = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()

with torch.no_grad():
    for p in ["The capital of France is", "def fibonacci(n):\n    ", "In 1905, Albert Einstein published"]:
        ids = tok(p, return_tensors="pt").input_ids
        a, b = ours(ids), ref(ids).logits
        diff = (a - b).abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        print(f"{p[:30]!r:34} T={ids.shape[1]:3d}  max|Δ|={diff:.2e}  top-1 agree={agree:.0%}")
        assert diff < 1e-3, "parity failed"
print("PARITY OK")
```

Expect `max|Δ|` between exactly 0 and ~$10^{-5}$ in FP32 on CPU. We tested with transformers 5.11 and torch 2.12 and got a bit-identical `0.00e+00`, because both sides compute the same ops in the same order. A different op order (for example, casting in a different place in RMSNorm) gives ~$10^{-5}$. On MPS expect a bit more. In BF16 expect ~$10^{-1}$ on the logits, which is fine: compare the top-1 agreement instead.

> [!TIP] When parity fails, bisect layer by layer
> Register forward hooks on both models (`ref.model.layers[i].register_forward_hook(...)`) and compare hidden states after the embedding, after each layer, and after the final norm. The first layer where they diverge tells you the bug. Usual suspects: RoPE convention, a missing QKV bias, the RMSNorm `eps` or cast order, the GQA repeat order (`repeat_interleave`, not `repeat`), forgetting tied embeddings.

## Step 6: generate on MPS

```python
# chat.py
import time, torch
from transformers import AutoTokenizer
from qwen2 import load

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = load(name, dtype=torch.float16, device="mps")
msgs = [{"role": "user", "content": "Explain what a KV cache is in one sentence."}]
ids = tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids.to("mps")
t0 = time.perf_counter()
out = model.generate(ids, max_new_tokens=100, temperature=0.7, top_k=50,
                     eos_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])
n = out.shape[1] - ids.shape[1]
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
print(f"{n} tokens, {n / (time.perf_counter() - t0):.1f} tok/s (no KV cache)")
```

Try `temperature=0` (greedy), 0.7 and 1.5, and change `top_k`:

::viz softmax-temp

> [!INTUITION] Feel the missing KV cache
> Time 20 vs 200 generated tokens. Without a cache, step $t$ recomputes all $t$ positions, so total work grows **quadratically** and tok/s *falls* as the output grows. HF `generate()` (which has a cache) holds roughly steady. That gap is Module 06's entire motivation.

- [ ] `qwen2.py` written by you (type it, don't paste; each line should make sense)
- [ ] `test_parity.py` prints `PARITY OK` for Qwen2.5-0.5B **and** Qwen2.5-0.5B-Instruct
- [ ] Broke it on purpose three ways (drop the QKV bias, use `repeat` instead of `repeat_interleave`, set `eps=1e-5`) and recorded each `max|Δ|`
- [ ] Generated on MPS in FP16 with sampling; measured tok/s for 20 vs 200 new tokens
- [ ] (Stretch) Qwen2.5-1.5B passes parity unchanged (no tied embeddings? check the config)
      */}),
      resources: [
        { title: "rasbt/LLMs-from-scratch — GPT → Llama conversion", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama", type: "repo", note: "step-by-step GPT-2 → Llama 2 → Llama 3.2 notebooks, with weight loading" },
        { title: "rasbt/LLMs-from-scratch — Qwen3 from scratch", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3", type: "repo", note: "standalone Qwen3 dense + MoE notebooks; compare with your file" },
        { title: "safetensors docs", url: "https://huggingface.co/docs/safetensors/index", type: "docs", note: "load_file / safe_open API" },
      ],
    },
    {
      id: "read-hf-modeling",
      title: "Deep dive: read modeling_llama.py, modeling_qwen2.py and vLLM's qwen2.py",
      kind: "deep",
      optional: true,
      minutes: 120,
      runsOn: ["any"],
      md: MD(function () {/*
You now know what every line *should* do, so reading production code becomes fast. Read three implementations of the **same** architecture, each written for a different job.

## 1. Hugging Face `modeling_llama.py`: the reference

Read it top to bottom, mapping each class to your file:

| HF | Yours | What to notice |
|---|---|---|
| `LlamaRMSNorm` | `RMSNorm` | identical fp32 upcast |
| `LlamaRotaryEmbedding` | `rope_cache` | `rope_type` dispatch (`default`, `linear`, `dynamic`, `yarn`, `llama3`) via `ROPE_INIT_FUNCTIONS`; `attention_scaling` |
| `rotate_half`, `apply_rotary_pos_emb` | same names | takes `position_ids`, so it works with padding and caches |
| `LlamaMLP` | `MLP` | `ACT2FN[config.hidden_act]` |
| `repeat_kv` | `repeat_interleave` | expand + reshape, no copy until needed |
| `eager_attention_forward` + `ALL_ATTENTION_FUNCTIONS` | `F.scaled_dot_product_attention` | pluggable backends: `eager`, `sdpa`, `flash_attention_2`, `flex_attention` (`attn_implementation=` in `from_pretrained`) |
| `past_key_values` / `Cache` | — | the KV cache (`DynamicCache`), which you add in Module 06 |
| `LlamaForCausalLM.forward` | `CausalLM.forward` | `logits_to_keep`: during generation only compute the last position's logits (a free speed-up you should copy) |

Then open `modular_qwen2.py`. Transformers now writes most models as a small **diff** that inherits from Llama ("modular transformers"), and generates `modeling_qwen2.py` from it. The diff for Qwen2 is basically: QKV bias on, sliding-window option. That confirms what you found by inspecting weights.

## 2. vLLM `vllm/model_executor/models/qwen2.py`: the serving version

Same math, different engineering:
- `QKVParallelLinear` and `MergedColumnParallelLinear` (`gate_up_proj`) are **fused and tensor-parallel-aware**. Each GPU holds a slice of the heads (Module 16).
- `RMSNorm(x, residual)` returns the fused add+norm.
- `Attention` is a layer backed by the engine's paged KV cache and FlashAttention/FlashInfer kernels. The model code never sees the cache layout (Module 08).
- `load_weights()` maps HF names to fused parameters with a `stacked_params_mapping` like `("qkv_proj", "q_proj", "q")`.
- Inputs are **flattened**: `(num_tokens, hidden)` with no batch dimension. Sequences are concatenated, and positions are passed explicitly. That's how continuous batching mixes prefill and decode in one forward pass (Module 09).

## 3. Sebastian Raschka's standalone notebooks

`standalone-qwen3.ipynb` and `standalone-llama32.ipynb` are single-notebook, from-scratch implementations with KV cache variants. Qwen3 adds **QK-norm** and drops the QKV bias. Porting your file to Qwen3-0.6B is a great exercise: diff the configs first, then add ~6 lines.

> [!CHECK] Self-check
> Without looking: what are the three things vLLM's model file does differently from HF's, and which later module explains each one?

- [ ] Read `modeling_llama.py` end to end; annotated 10 places that differ from your file
- [ ] Ran HF with `attn_implementation="eager"` vs `"sdpa"` and compared logits and speed
- [ ] Read vLLM's `qwen2.py` `load_weights()` and explained `stacked_params_mapping`
- [ ] (Stretch) Ported your file to Qwen3-0.6B (QK-norm, no bias, explicit `head_dim`) and passed parity
      */}),
      resources: [
        { title: "transformers — modeling_llama.py", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", type: "repo", note: "the reference Llama implementation" },
        { title: "transformers — modular_qwen2.py", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modular_qwen2.py", type: "repo", note: "Qwen2 written as a diff over Llama" },
        { title: "vLLM — models/qwen2.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py", type: "repo", note: "the same model, written for serving" },
      ],
    },
  ],

  challenge: {
    title: "Parity, sampling and the architecture comparison sheet",
    md: MD(function () {/*
Put everything in `course-work/m04/`:

1. **`qwen2.py` + `test_parity.py`**: parity on Qwen2.5-0.5B-Instruct, with the CPU FP32 max abs diff < 1e-3 on at least 5 prompts, including one longer than 500 tokens (this exercises RoPE at larger positions).
2. **`chat.py`**: a CLI that applies the chat template, samples with `temperature` / `top_k` / `top_p` (implement top-p yourself: sort, cumulative sum, cut at p), stops on the correct EOS ids, and prints tok/s on MPS.
3. **`inspect_config.py`**: given any HF repo id, it fetches only `config.json` and prints layers, hidden, heads, KV heads, head_dim, vocab, max context, `rope_theta`/`rope_scaling`, dense vs MoE (experts, active), estimated total params, and **KV bytes per token** in BF16.
4. **`ARCHITECTURES.md`**: a comparison table for **Llama-3-8B, Qwen2.5-7B, Mistral-7B-v0.3, DeepSeek-V3** (rows: layers, hidden size, Q heads, KV heads / attention type, head_dim, FFN type and size, vocab, context, `rope_theta`, norm, activation, biases, MoE experts / active, total / active params, KV per token). Below it, write one paragraph per model on what its choices mean for serving it.

<details><summary>Answer key (check yourself after filling the table from the configs)</summary>

| | Llama-3-8B | Qwen2.5-7B | Mistral-7B-v0.3 | DeepSeek-V3 |
|---|---|---|---|---|
| Layers | 32 | 28 | 32 | 61 |
| Hidden | 4096 | 3584 | 4096 | 7168 |
| Q heads / KV | 32 / 8 (GQA) | 28 / 4 (GQA) | 32 / 8 (GQA) | 128 heads, **MLA** (latent 512 + 64 RoPE) |
| head_dim | 128 | 128 | 128 | 128 (+64 RoPE part for Q/K) |
| FFN | SwiGLU 14336 | SwiGLU 18944 | SwiGLU 14336 | first 3 dense (18432), then **MoE**: 1 shared + 256 routed experts (2048 each), top-8 |
| Vocab | 128,256 | 152,064 | 32,768 | 129,280 |
| Context | 8K (3.1: 128K) | 32K native, 128K with YaRN | 32K | 128K (YaRN) |
| rope_theta | 500,000 | 1,000,000 | 1,000,000 | 10,000 + YaRN |
| Norm / act | RMSNorm / SiLU | RMSNorm / SiLU | RMSNorm / SiLU | RMSNorm / SiLU |
| Biases | none | QKV bias | none | none |
| Params | 8.0B | 7.6B | 7.2B | 671B total / 37B active |
| KV / token (BF16) | 128 KB | 56 KB | 128 KB | ~69 KB (latent) |

</details>
    */}),
    checklist: [
      "Parity test passes (< 1e-3, FP32, CPU) on ≥5 prompts, including one > 500 tokens",
      "`chat.py` uses the chat template, stops on both Qwen EOS ids, and implements top-p itself",
      "`inspect_config.py` reports KV bytes/token and active vs total params correctly for a dense and an MoE model",
      "`ARCHITECTURES.md` table is complete, with one serving-implications paragraph per model",
      "You can explain, without notes, why GQA/MLA exist in terms of decode memory bandwidth",
    ],
    stretch: "Load **Llama-3.2-1B** (gated: accept the license, `hf auth login`). It needs `rope_scaling` type `llama3`: scale `inv_freq` by `factor` for low frequencies (wavelength > `original_max_position_embeddings / low_freq_factor`), keep high frequencies, and interpolate smoothly in between (copy the formula from HF's `_compute_llama3_parameters`). Pass parity. Then add a tiny KV cache to `generate()` and measure the tok/s gain (a preview of Module 06).",
  },

  connects: MD(function () {/*
You now own a readable, correct model file, the "runtime" layer from the Module 00 map. Everything in Phase 2 plugs into it. Module 06 adds a **KV cache** to your `generate()`. Module 08 turns that cache into **pages**, sized with the KV-per-token formula you just used. Module 09 **batches** many sequences through the flattened-token forward pass you saw in vLLM's `qwen2.py`. Module 14 quantizes these exact `nn.Linear` weights, and Modules 16–17 split them across GPUs (TP for attention/MLP, EP for MoE experts).

Next, Module 05 shows where the weights you just loaded **came from** (pretraining, SFT, preference tuning) and has you fine-tune this same Qwen2.5 with LoRA.
  */}),

  interview: [
    "Walk me through the differences between GPT-2 and Llama 3 at the block level. Which of them matter for inference and why?",
    "Derive why RoPE makes the attention score depend only on relative position. What does `rope_theta` control, and what is YaRN for?",
    "Compute the KV cache size per token for Llama-3-70B in BF16. How many 32K-token sequences fit on 8×H100 after weights?",
    "What problem do MQA, GQA and MLA solve? What does each give up?",
    "DeepSeek-V3 has 671B parameters but 37B active. How does that affect memory, prefill compute, and decode speed at batch 1 vs batch 256?",
    "A customer's fine-tuned Qwen model produces fluent but off-topic text and never stops. List the first four things you check.",
    "Why do checkpoints use safetensors instead of pickle? What does mmap-based loading buy an inference server?",
    "Your re-implementation's logits differ from HF by 0.5 in FP32. How do you find the bug?",
  ],

  resources: [
    { title: "Karpathy — Let's build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE", type: "video", note: "BPE from scratch; the tokenizer half of this module" },
    { title: "karpathy/minbpe", url: "https://github.com/karpathy/minbpe", type: "repo", note: "companion code + exercises" },
    { title: "rasbt/LLMs-from-scratch", url: "https://github.com/rasbt/LLMs-from-scratch", type: "repo", note: "ch05 bonus folders: GPT→Llama, Qwen3, Gemma 3 from scratch" },
    { title: "Sebastian Raschka — The Big LLM Architecture Comparison", url: "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison", type: "article", note: "DeepSeek-V3, Qwen3, Llama 4, GPT-OSS, Gemma... side by side, kept up to date" },
    { title: "RoFormer (RoPE)", url: "https://arxiv.org/abs/2104.09864", type: "paper", note: "rotary position embeddings" },
    { title: "GQA paper", url: "https://arxiv.org/abs/2305.13245", type: "paper", note: "grouped-query attention" },
    { title: "DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", type: "paper", note: "MLA + MoE at frontier scale" },
    { title: "Qwen2.5 Technical Report", url: "https://arxiv.org/abs/2412.15115", type: "paper", note: "the architecture and data of the model you loaded" },
    { title: "The Llama 3 Herd of Models", url: "https://arxiv.org/abs/2407.21783", type: "paper", note: "section 3: architecture, tokenizer, RoPE base, GQA" },
    { title: "transformers — modeling_llama.py", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", type: "repo", note: "the reference implementation" },
    { title: "Inference Engineering (Baseten) — Ch. 2.2 & 4.2", url: "Inference%20Engineering.pdf", type: "book", note: "LLM mechanics, architectures, MoE, file formats" },
  ],
});
