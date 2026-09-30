Course.module({
  id: "m20-modalities",
  title: "Beyond text: VLMs, embeddings, speech, image generation",
  short: "Beyond text (modalities)",
  tagline: "Serve a vision-language model, an embedding model, Whisper and a TTS on your Mac, discover that each one has a different bottleneck, and wire three of them into a voice agent with a measured latency budget.",
  hours: 16,
  level: "optional",
  runsOn: ["mac", "colab", "cloud"],
  tags: ["vlm", "embeddings", "rerankers", "whisper", "asr", "tts", "diffusion", "voice-agents", "mlx"],

  goal: MD(function () {/*
You run four non-chat models locally and produce a table like this (M-series Mac, numbers illustrative):

```text
model                         task              key metric                     bottleneck
bge-small-en-v1.5 (33M)       embeddings        3,900 sentences/s @ batch 128   compute (tiny model, big batches)
Qwen2.5-VL-3B 4-bit           image → text      TTFT 1.9 s for a 1024² image    prefill: ~1,300 image tokens + vision encoder
whisper-large-v3-turbo        speech → text     RTF 0.05 (60 s audio in 3 s)    encoder compute + short decoder
Kokoro-82M                    text → speech     time-to-first-audio 110 ms      tiny model; streaming and chunking matter
```

…and then a **voice agent** (speech in → Whisper → LLM → TTS → speech out) whose per-stage latencies you measured and optimized:

```text
end-of-speech → transcript      180 ms   (mlx-whisper on a 4 s utterance)
transcript → first LLM sentence 310 ms   (TTFT 120 ms + 11 tokens)
first sentence → first audio    120 ms   (Kokoro, streamed)
voice-to-voice                  ~610 ms  + VAD endpointing (~300 ms)  →  < 1 s budget ✓
```

::viz roofline
  */}),
  demo: { viz: "roofline", params: {} },

  why: MD(function () {/*
Text chat is only part of inference traffic. Baseten (ch. 6) devotes a whole chapter to modalities because customers run **embeddings** for search and RAG, **transcription** at millions of hours, **TTS** for voice products, **VLMs** for documents and screens, and **image/video generation** — and each breaks different assumptions from the LLM world: embeddings have no decode phase, Whisper is an encoder–decoder with fixed 30-second windows, TTS is an LLM whose tokens are audio, diffusion is pure compute with no KV cache. Voice agents are one of the fastest-growing inference workloads, and every millisecond of their pipeline is an engineering decision. Knowing how to reason about any model’s bottleneck — not just LLMs — is what makes an inference engineer versatile.
  */}),

  prereqs: [
    {
      title: "Roofline in one paragraph",
      skipIf: "you did M07 and can say whether a workload is compute- or memory-bound",
      md: MD(function () {/*
Every operation does some FLOPs and moves some bytes. **Arithmetic intensity** = FLOPs ÷ bytes. A GPU has a peak compute rate and a peak memory bandwidth; their ratio (the *ridge point*, ~300 FLOP/byte on an H100, ~20–40 on Apple silicon GPUs) decides the regime. Below it you are **memory-bound** (LLM decode at small batch: each weight byte is used once per token). Above it you are **compute-bound** (prefill, embeddings, vision encoders, diffusion: each weight is reused across many tokens or pixels). Different modalities sit in different places on this chart — that is the whole lesson of this module.
      */}),
    },
    {
      title: "Audio basics",
      skipIf: "you know sample rate, mel spectrograms and what a codec token is",
      md: MD(function () {/*
- Audio is a waveform sampled at e.g. **16 kHz** (speech recognition) or **24 kHz** (TTS output): 16,000 numbers per second.
- **Mel spectrogram**: slice the waveform into ~25 ms overlapping windows every 10 ms and take the energy per frequency band on a perceptual (mel) scale → an image-like 2-D array (time × 80/128 bands). Whisper’s input.
- **Neural audio codec** (SNAC, EnCodec, Mimi): compresses audio into discrete **tokens** at a few dozen frames per second, and decodes tokens back to waveform. Token-based TTS models generate these tokens with an LLM.
- **Real-time factor** (RTF): processing time ÷ audio duration (lower is better; < 1 means faster than real time). Some sources, including the Baseten book, use the inverse (“1000× real time”). Always state which.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: look, listen and speak on your Mac",
      kind: "demo",
      minutes: 60,
      runsOn: ["mac"],
      md: MD(function () {/*
## Three models, three commands

All three use **MLX** (Apple’s array framework), 4-bit or BF16 weights from `mlx-community`, and run on any M-series Mac with 16 GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U mlx-vlm mlx-whisper mlx-audio misaki sounddevice soundfile
```

**See** — a vision-language model describes an image:

```bash
time mlx_vlm.generate --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --max-tokens 100 --temperature 0.0 \
  --prompt "Describe this image in two sentences." \
  --image http://images.cocodataset.org/val2017/000000039769.jpg
```

It prints prompt tokens, prompt tok/s (prefill) and generation tok/s. Notice the **prompt token count**: your 8-word question became hundreds of tokens because the image was turned into tokens.

**Listen** — Whisper transcribes speech:

```python
# listen.py
import time, mlx_whisper
t = time.perf_counter()
out = mlx_whisper.transcribe("speech.wav", path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
dt = time.perf_counter() - t
print(out["text"]); print(f"{dt:.2f}s")
```

(Record `speech.wav` with QuickTime or `say -o speech.aiff "..."` and convert: `afconvert -f WAVE -d LEI16@16000 speech.aiff speech.wav`. Run twice — the first run downloads and compiles.)

**Speak** — Kokoro (82M parameters) turns text into audio, streamed in chunks:

```python
# speak.py
import time, numpy as np, sounddevice as sd
from mlx_audio.tts.utils import load_model
model = load_model("mlx-community/Kokoro-82M-bf16")
t = time.perf_counter(); first = None; chunks = []
for r in model.generate(text="Inference engineering is the art of making models fast and cheap.",
                        voice="af_heart", speed=1.0, lang_code="a"):
    if first is None: first = time.perf_counter() - t
    chunks.append(np.array(r.audio))
audio = np.concatenate(chunks)
print(f"time to first audio {first*1000:.0f} ms, audio {len(audio)/24000:.1f}s, total {time.perf_counter()-t:.2f}s")
sd.play(audio, 24000); sd.wait()
```

## What to notice

- The VLM spends a surprising share of its time **before the first word** — prefill over image tokens plus the vision encoder.
- Whisper processes a minute of audio in a few seconds — **far faster than real time** — but you still wait for the whole clip unless you stream.
- The TTS model is tiny; time-to-first-audio is dominated by **how much text you give it at once** and by chunking.

Three models, three different answers to “what makes this slow?” The rest of the module explains each and then combines them.

- [ ] Ran all three; recorded VLM prompt tokens and TTFT, Whisper time vs audio length, TTS time-to-first-audio
- [ ] Repeated the VLM run with a small (e.g. 320 px) and a large (e.g. 1600 px) image and compared prompt tokens
      */}),
      resources: [
        { title: "mlx-vlm", url: "https://github.com/Blaizzy/mlx-vlm", type: "tool", note: "VLM inference and server on Apple silicon" },
        { title: "mlx-audio", url: "https://github.com/Blaizzy/mlx-audio", type: "tool", note: "TTS/STT on Apple silicon, including Kokoro" },
        { title: "mlx-examples: Whisper", url: "https://github.com/ml-explore/mlx-examples/tree/main/whisper", type: "repo", note: "the mlx-whisper implementation" },
      ],
    },
    {
      id: "modality-map",
      title: "The modality map: bottlenecks and metrics",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
## One question for every model

For any model ask: **what are the tokens, how many per request, and is there a sequential decode loop?** That tells you where it sits on the roofline and which metric matters.

| Modality | Architecture | Tokens per request | Sequential loop? | Usual bottleneck | Metrics that matter |
|---|---|---|---|---|---|
| **LLM chat** | Decoder | 100s–10Ks in, 100s–1000s out | Yes (decode) | Memory bandwidth (decode), compute (prefill) | TTFT, TPOT, tok/s, goodput |
| **Embeddings / rerankers** | Encoder (or decoder used as encoder) | 10s–500 in, **0 out** | No | **Compute** at large batch; overhead at small batch | Items/s, p99 latency per batch, cost per 1M items |
| **VLM** | Vision encoder + LLM | Image → ~100s–1000s of tokens + text | Yes | **Prefill** (image tokens) + encoder compute | TTFT (per image size), TPOT, images/s |
| **ASR (Whisper)** | Encoder–decoder | 30 s window → 1,500 encoder frames; ~100s text tokens out | Yes (short decoder) | Encoder compute; decoder steps | RTF, WER, latency to final transcript, streaming partial latency |
| **TTS (token-based)** | LLM → audio codec decoder | Text in; ~80–100 audio tokens per **second of audio** out | Yes | Decode speed must exceed real-time rate | **Time to first audio**, RTF, naturalness (MOS) |
| **Image diffusion** | Transformer/U-Net denoiser, many steps | Latent patches (fixed) | Steps (not tokens) | **Compute** (big matmuls every step) | Seconds per image, steps, images/s |
| **Video diffusion** | Same, huge latents | 10Ks–100Ks of patches | Steps | Compute, **attention** dominates | Seconds per clip, GPU-seconds per clip |

::viz roofline

> [!INTUITION] Where do they land?
> Put each on the roofline: LLM decode at batch 1 has intensity ≈ 1–2 FLOP/byte (far left, memory-bound). An embedding batch of 128 × 256 tokens reuses each weight ~32K times (far right, compute-bound). Diffusion, vision encoders and prefill are also far right. So the optimization playbook flips: for decode you shrink bytes (quantization, M14) and batch; for compute-bound work you need **FLOP-efficient kernels, lower-precision compute (FP8 tensor cores), fewer steps, and caching**, and batching gives much less extra.

## Engines by modality

| Modality | Common serving choices |
|---|---|
| Embeddings / rerankers | Hugging Face **TEI**, **Infinity**, vLLM `--runner pooling`, SGLang; on Mac: sentence-transformers (MPS), mlx-embeddings |
| VLMs | vLLM, SGLang, TRT-LLM, LMDeploy; on Mac: mlx-vlm, llama.cpp (mtmd), Ollama |
| ASR | faster-whisper (CTranslate2), whisper.cpp, TRT-LLM Whisper, vLLM (Whisper supported); on Mac: mlx-whisper, whisper.cpp (Metal/Core ML), WhisperKit |
| TTS | Orpheus via vLLM/SGLang/TRT-LLM + SNAC decoder; Kokoro; on Mac: mlx-audio |
| Image/video | Diffusers, TensorRT, xDiT (multi-GPU), ComfyUI; on Mac: Diffusers (MPS), mflux, Draw Things |

- [ ] Placed each of your four see-it models on the roofline with a rough arithmetic-intensity estimate
- [ ] For one modality, listed the three metrics you would put on its dashboard and why
      */}),
      resources: [
        { title: "Inference Engineering (Baseten) — ch. 6 Modalities", url: "Inference%20Engineering.pdf", type: "book", note: "VLMs, embeddings, ASR, TTS, image and video generation in production" },
      ],
    },
    {
      id: "embeddings",
      title: "Serving embeddings and rerankers",
      kind: "build",
      minutes: 120,
      runsOn: ["mac", "colab", "cloud"],
      md: MD(function () {/*
## Why embeddings are different

An embedding model runs **one forward pass** over a short input and returns a vector — no decode loop, no KV cache to keep, nothing to stream. A reranker (cross-encoder) takes (query, document) pairs and returns a score. Consequences (Baseten ch. 6.2):

- **Compute-bound at useful batch sizes**, overhead-bound at batch 1. Throughput comes from **batching many short inputs**: dynamic batching (group requests arriving within a few ms) and sorting/bucketing by length to avoid padding waste.
- **No prefix caching, no prefill/decode disaggregation** — those LLM tricks have nothing to act on.
- **Two traffic shapes, two deployments**: *backfill* (embed 100M documents: maximize throughput, big batches, any latency) vs *lookup* (embed one user query: p99 latency matters, small batches). Serve them separately so the backfill never blocks a live query.
- **Scale horizontally** with many small GPUs or even CPUs; models are small (33M–8B).
- **Quantize carefully**: FP8/INT8 is usually fine, but measure **cosine similarity** between quantized and original embeddings (aim ≥ 0.99) and retrieval quality on your data, not just perplexity.
- **Tokenization** can become the bottleneck at high throughput: it runs on CPU. Engines like TEI (Rust) do it outside the Python GIL.

## Serving options

```bash
# Hugging Face Text Embeddings Inference (CPU image; GPU images also exist — check the README for current tags)
docker run -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-embeddings-inference:cpu-1.8 \
  --model-id BAAI/bge-small-en-v1.5
curl localhost:8080/embed -X POST -H 'Content-Type: application/json' -d '{"inputs":["hello world"]}'

# Infinity (Python, many models, OpenAI-compatible)
pip install "infinity-emb[all]" && infinity_emb v2 --model-id BAAI/bge-small-en-v1.5

# vLLM pooling runner (GPU): embeddings and reranking with the OpenAI-style API
vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling
vllm serve BAAI/bge-reranker-v2-m3 --runner pooling --port 8001
curl localhost:8001/rerank -H 'Content-Type: application/json' -d \
  '{"model":"BAAI/bge-reranker-v2-m3","query":"What is a KV cache?","documents":["It stores attention keys and values.","A cache for web pages."]}'
```

## Lab: the batching curve on your Mac

```python
# embed_bench.py — throughput and latency vs batch size (sentence-transformers on MPS)
import time, random, numpy as np
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
words = open("/usr/share/dict/words").read().split()
texts = [" ".join(random.choices(words, k=random.randint(8, 60))) for _ in range(4096)]
m.encode(texts[:256], batch_size=64)                      # warmup
for bs in [1, 4, 16, 64, 128, 256, 512]:
    n = min(len(texts), bs * 16)
    t = time.perf_counter(); m.encode(texts[:n], batch_size=bs); dt = time.perf_counter() - t
    print(f"batch {bs:4d}: {n/dt:8.0f} texts/s   per-batch latency {dt/(n/bs)*1000:7.1f} ms")
```

Expect throughput to rise steeply up to batch ~64–128 then flatten (compute saturated), while per-batch latency grows roughly linearly after that point — the embedding version of the M19 knee. Then:

1. Sort `texts` by length before batching; measure the padding win.
2. Compare `all-MiniLM-L6-v2` (22M), `bge-small` (33M) and `Qwen3-Embedding-0.6B` (600M): texts/s vs retrieval quality (check their MTEB scores).
3. On a GPU, benchmark the vLLM or TEI server with `vllm bench serve --backend openai-embeddings --endpoint /v1/embeddings ...` or AIPerf’s embeddings endpoint type.

- [ ] Batching curve (texts/s and latency vs batch size) for at least two models
- [ ] Measured the effect of length-sorting
- [ ] Cosine similarity between FP32 and FP16 (or quantized) embeddings on 1,000 texts
- [ ] A one-paragraph design: separate backfill and lookup deployments, with batch sizes for each
      */}),
      resources: [
        { title: "Text Embeddings Inference", url: "https://github.com/huggingface/text-embeddings-inference", type: "tool", note: "Rust embedding/reranker server with dynamic batching" },
        { title: "Infinity", url: "https://github.com/michaelfeil/infinity", type: "tool", note: "OpenAI-compatible embedding server" },
        { title: "vLLM — Pooling models", url: "https://docs.vllm.ai/en/latest/models/pooling_models.html", type: "docs", note: "embeddings, classification, scoring and rerank endpoints" },
        { title: "MTEB leaderboard", url: "https://huggingface.co/spaces/mteb/leaderboard", type: "tool", note: "compare embedding model quality before you optimize speed" },
      ],
    },
    {
      id: "vlm",
      title: "Vision-language models: images become (lots of) tokens",
      kind: "build",
      minutes: 120,
      runsOn: ["mac", "colab", "cloud"],
      md: MD(function () {/*
## How an image gets into an LLM

A VLM = **vision encoder** (a ViT) + **projector** + a normal **LLM**. The image is cut into patches, the ViT turns patches into vectors, the projector maps them into the LLM’s embedding space, and they are inserted into the prompt as **image tokens**. From then on it is an LLM request — with a very long prompt.

**Qwen2.5-VL** uses 14 × 14 px patches and merges each 2 × 2 group into one token, so **one token per 28 × 28 pixels** at native resolution:

| Image size | Tokens | Equivalent text |
|---|---|---|
| 448 × 448 | $16 \times 16 = 256$ | a long paragraph |
| 1024 × 1024 | $\approx 36.6^2 \approx 1{,}340$ | two pages |
| 1920 × 1080 screenshot | $\approx 68.6 \times 38.6 \approx 2{,}650$ | a short chapter |
| 4 s of video at 2 fps, 448² frames | frames are merged in pairs over time → $\approx 4 \times 256 = 1{,}024$ | |

Baseten (ch. 6.1): ~**1K tokens per image** is typical, high-resolution modes ~4× that, and a naively tokenized 4-second, 24 fps clip can reach ~**100K tokens** — which is why video inputs are sampled at low fps.

## Consequences for serving

- **Prefill-heavy**: TTFT grows with image resolution; decode is ordinary LLM decode. Chunked prefill (M09) and prefill/decode disaggregation (M17) help.
- **Vision encoder cost**: runs once per image, compute-bound; at high resolution it is a visible slice of TTFT. Engines can batch it and some run it on separate GPUs (encoder disaggregation).
- **KV cache**: image tokens occupy KV like any tokens → fewer concurrent requests. 10 images × 1,300 tokens each = 13K tokens of KV for one request.
- **Caching**: the same image sent twice (multi-turn chat about one screenshot) can reuse the encoder output (vLLM’s multimodal processor cache, `vllm:mm_cache_hits`) and prefix-cached KV.
- **Control resolution**: the cheapest optimization is sending fewer pixels. Qwen2.5-VL’s processor takes `min_pixels` / `max_pixels`; cap them per use case (OCR needs detail; “is there a cat?” does not).

::viz prefill-decode

## Serving it

```bash
# GPU (Colab L4/A100 or cloud): OpenAI-compatible multimodal chat
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 4}' \
  --mm-processor-kwargs '{"max_pixels": 802816}'      # 1024 × 28 × 28 → ≤ 1,024 image tokens

# Mac: same API via mlx-vlm
mlx_vlm.server --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit --port 8080
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
r = c.chat.completions.create(model="Qwen/Qwen2.5-VL-3B-Instruct", max_tokens=100, messages=[{
    "role": "user", "content": [
        {"type": "text", "text": "How many cats are in this image, and what else is on the couch?"},
        {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/val2017/000000039769.jpg"}}]}])
print(r.usage.prompt_tokens, r.choices[0].message.content)
```

## Lab: TTFT vs resolution

Resize one image to 224, 448, 896, 1344 and 1792 px on the long side. For each, send it 10 times (after warmup) and record `usage.prompt_tokens` and TTFT (streaming). Plot TTFT against prompt tokens: it should be close to linear with a small fixed offset (encoder + overhead), exactly like long text prompts. Then run an OCR-style question on an image with small text (a screenshot of a document) at each resolution and find the smallest size that still reads the text.

- [ ] Table: resolution → image tokens (predicted with the 28 px rule and measured) → TTFT
- [ ] TTFT vs tokens plot with a linear fit; the intercept interpreted
- [ ] Smallest resolution that keeps the task accurate, and the TTFT/KV saving vs full resolution
- [ ] (GPU) Throughput with 1 vs 4 images per request at fixed concurrency
      */}),
      resources: [
        { title: "Qwen2.5-VL technical report", url: "https://arxiv.org/abs/2502.13923", type: "paper", note: "dynamic resolution, 2×2 token merging, video handling" },
        { title: "vLLM — Multimodal inputs", url: "https://docs.vllm.ai/en/latest/features/multimodal_inputs.html", type: "docs", note: "image/video/audio inputs, limits and processor kwargs" },
        { title: "Qwen2.5-VL-3B-Instruct", url: "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct", type: "docs", note: "model card: min_pixels/max_pixels usage" },
      ],
    },
    {
      id: "asr",
      title: "Speech recognition: Whisper, real-time factor and streaming",
      kind: "build",
      minutes: 120,
      runsOn: ["mac", "colab"],
      md: MD(function () {/*
## Whisper in one picture

Whisper is an **encoder–decoder** transformer trained on 680K hours of audio. Every input is a **30-second window** (padded if shorter) → 80- or 128-band log-mel spectrogram → encoder → 1,500 audio frames. The decoder then generates text tokens autoregressively, cross-attending to those frames. Whisper large-v3 has 1.55B parameters with 32 decoder layers; **large-v3-turbo** (809M) keeps the full encoder but only **4 decoder layers**, making decoding several times faster at a small accuracy cost.

Baseten (ch. 6.3) on production transcription:
- The **decoder dominates** latency for long transcripts (it is the sequential loop); the encoder is one compute-bound pass per 30 s chunk.
- Long audio is split into chunks — ideally at silences found by a **voice activity detector** (VAD, e.g. Silero) rather than blindly every 30 s, which cuts words — and chunks are transcribed **in parallel** (batch across chunks), then stitched.
- Throughput is measured as audio-seconds per wall-second: an optimized GPU deployment can transcribe an hour in a few seconds (“~1000× real time”).
- For live use, the target is partial transcripts within ~**200 ms** of speech.
- **Hallucinations** on silence or music are real: check compression ratio (repetitive output), words-per-minute sanity, and use temperature fallback; drop segments VAD says are silent.
- **Diarization** (who spoke when, e.g. pyannote) can cost **more than transcription** itself (≥ 2×).

## Real-time factor

$$\text{RTF} = \frac{\text{processing time}}{\text{audio duration}}$$

RTF 0.05 = 20× faster than real time. For **batch** transcription you care about audio-hours per GPU-hour (= 1 / RTF at full batch). For **streaming** you care about latency from a word being spoken to it appearing — RTF < 1 is necessary but not sufficient.

## Lab: RTF across implementations (Mac)

```bash
pip install mlx-whisper
# whisper.cpp (Metal): build once
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
sh ./models/download-ggml-model.sh large-v3-turbo
cmake -B build && cmake --build build -j --config Release
time ./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin -f ../talk_60s.wav
```

```python
# rtf.py — mlx-whisper on clips of different lengths
import time, soundfile as sf, mlx_whisper
MODEL = "mlx-community/whisper-large-v3-turbo"
mlx_whisper.transcribe("talk_60s.wav", path_or_hf_repo=MODEL)            # warmup + download
for f in ["talk_5s.wav", "talk_30s.wav", "talk_60s.wav", "talk_300s.wav"]:
    dur = sf.info(f).duration
    t = time.perf_counter(); out = mlx_whisper.transcribe(f, path_or_hf_repo=MODEL)
    dt = time.perf_counter() - t
    print(f"{f}: audio {dur:.0f}s  time {dt:.2f}s  RTF {dt/dur:.3f}  words {len(out['text'].split())}")
```

(Cut clips from any podcast or a talk you recorded with `ffmpeg -ss 0 -t 60 -i talk.mp3 -ar 16000 -ac 1 talk_60s.wav`.)

What to look for: short clips have **worse RTF** (a 5 s clip still pays a full 30 s window through the encoder); long clips amortize. Compare `whisper-large-v3-turbo` vs `whisper-small` for speed and errors (count mistakes on 100 words you transcribe by hand = a mini WER).

## Streaming

Whisper was not designed for streaming. Practical approaches: VAD-segmented utterances (transcribe each as soon as the speaker pauses — what voice agents do), or sliding windows re-transcribed every ~0.5–1 s with a stable prefix (whisper.cpp’s stream example, WhisperKit). Purpose-built streaming ASR models and APIs exist for lower partial latency; the trade is accuracy vs latency vs cost.

- [ ] RTF table: 4 clip lengths × 2 implementations (mlx-whisper, whisper.cpp) × 2 model sizes
- [ ] Explained why a 5 s clip has worse RTF than a 60 s clip
- [ ] Mini-WER comparison between turbo and small
- [ ] Ran Whisper on 10 s of silence/music and observed (or failed to observe) a hallucination; proposed a guard
      */}),
      resources: [
        { title: "Whisper paper (Robust Speech Recognition via Large-Scale Weak Supervision)", url: "https://arxiv.org/abs/2212.04356", type: "paper", note: "architecture, 30 s windows, training data" },
        { title: "whisper.cpp", url: "https://github.com/ggml-org/whisper.cpp", type: "tool", note: "C/C++ Whisper with Metal, Core ML, streaming example" },
        { title: "faster-whisper", url: "https://github.com/SYSTRAN/faster-whisper", type: "tool", note: "CTranslate2 backend, batched pipeline with VAD (GPU/CPU)" },
        { title: "Silero VAD", url: "https://github.com/snakers4/silero-vad", type: "tool", note: "the standard lightweight voice activity detector" },
      ],
    },
    {
      id: "tts",
      title: "Text-to-speech: LLMs that emit audio, and time to first audio",
      kind: "build",
      minutes: 105,
      runsOn: ["mac", "cloud"],
      md: MD(function () {/*
## Two families

- **Small dedicated models** (Kokoro-82M, Piper, older FastSpeech/VITS-style): predict audio (or a spectrogram) for a whole phrase in one or a few passes. Tiny, fast, CPU-friendly; less expressive.
- **Token-based TTS on an LLM backbone** (Orpheus, Sesame CSM, many 2025-era models): an LLM generates **audio codec tokens** autoregressively, then a codec decoder turns tokens into waveform. Expressive, cloneable voices, emotions — and served with the **same engines** as text LLMs.

## Orpheus as the worked example (Baseten ch. 6.4)

- Backbone: **Llama 3.2 3B** fine-tuned to emit **SNAC** codec tokens (24 kHz audio).
- SNAC encodes each ~85 ms frame as **7 tokens** across three levels; at ~12 frames per second that is $7 \times 12 \approx 84$ tokens per second of audio. So real-time playback needs the LLM to decode at **80–100 tok/s per stream** — the TPOT SLO is set by physics: $\text{TPOT} \le 1/84 \approx 12$ ms.
- That makes TTS **the most latency-strict decode workload** you will meet. Tricks: FP8 weights (quality holds), CUDA graphs, small batches per GPU to protect per-stream speed, speculative decoding.
- The **SNAC decoder** is a separate small conv net; batch it dynamically across streams with a short timeout (the book uses ~15 ms) so it does not add latency.
- Stream audio chunks to the client (WebSockets) as soon as a few frames are decoded: **time to first byte ~150 ms** on an H100 in Baseten’s deployment.
- Quality degrades on very long single generations (> ~30 s of audio): chunk text by sentence.

::viz prefill-decode

## Metrics that matter

| Metric | Definition | Target for conversation |
|---|---|---|
| **Time to first audio** (TTFA / TTFB) | Request → first playable audio chunk | ≤ 200 ms |
| **RTF** | Synthesis time ÷ audio duration | < 1 with margin (e.g. ≤ 0.5), *per stream at your concurrency* |
| **Streams per GPU** | Concurrent real-time streams before RTF crosses the limit | Capacity planning unit |
| Naturalness | MOS or pairwise preference | Evaluate after every quantization change |

## Lab: chunking changes TTFA more than the model does

```python
# tts_bench.py — time to first audio vs how much text you give at once
import time, re, numpy as np
from mlx_audio.tts.utils import load_model
tts = load_model("mlx-community/Kokoro-82M-bf16")
para = ("Inference engineering is about latency, throughput and cost. Every millisecond counts in a voice product. "
        "Streaming lets the listener start hearing the answer while the rest is still being generated. "
        "Chunking text by sentence keeps the first chunk short.")
list(tts.generate(text="Warm up.", voice="af_heart", lang_code="a"))
def ttfa(text):
    t = time.perf_counter(); first = None; n = 0
    for r in tts.generate(text=text, voice="af_heart", lang_code="a"):
        a = np.array(r.audio); n += len(a)
        if first is None: first = time.perf_counter() - t
    total = time.perf_counter() - t
    return first, total, n / 24000
f, tot, dur = ttfa(para)
print(f"whole paragraph: TTFA {f*1000:.0f} ms, RTF {tot/dur:.3f}")
first_sentence = re.split(r"(?<=[.!?])\s", para)[0]
f, tot, dur = ttfa(first_sentence)
print(f"first sentence only: TTFA {f*1000:.0f} ms, RTF {tot/dur:.3f}")
```

Then, on a cloud GPU if you have one, serve Orpheus with vLLM (the Orpheus-TTS repo has a streaming example) and measure per-stream decode tok/s at 1, 4, 16 concurrent streams: find the concurrency where it drops below ~85 tok/s — that is your **streams-per-GPU** capacity.

- [ ] TTFA and RTF for whole-paragraph vs sentence-chunked input
- [ ] Explained the 80–100 tok/s requirement from the SNAC frame rate
- [ ] (GPU, optional) Streams-per-GPU for a token-based TTS at a real-time constraint
      */}),
      resources: [
        { title: "Orpheus-TTS", url: "https://github.com/canopyai/Orpheus-TTS", type: "repo", note: "Llama-based TTS emitting SNAC tokens; streaming inference with vLLM" },
        { title: "SNAC audio codec", url: "https://github.com/hubertsiuzdak/snac", type: "repo", note: "multi-scale codec used by Orpheus" },
        { title: "Kokoro-82M", url: "https://huggingface.co/hexgrad/Kokoro-82M", type: "docs", note: "small, fast, high-quality open TTS" },
      ],
    },
    {
      id: "diffusion",
      title: "Image and video generation: steps, guidance and caching",
      kind: "lab",
      minutes: 105,
      runsOn: ["mac", "cloud"],
      md: MD(function () {/*
## A completely different loop

Diffusion models start from noise and run a denoiser (a U-Net or, now, a **diffusion transformer** such as FLUX) for $N$ **steps**. Each step processes the *entire* latent image (thousands of patches) at once — big matmuls, high arithmetic intensity, no KV cache. So (Baseten ch. 6.5):

- **Compute-bound**: FP8/FP4 tensor cores, fused attention (FlashAttention, SageAttention) and `torch.compile` pay off directly. Batching adds throughput only until compute saturates — often already at batch 1–4 for large images.
- **Cost ∝ steps × cost per step**. Latency is roughly linear in steps.
- **Classifier-free guidance (CFG)** runs the denoiser twice per step (with and without the prompt) → 2× compute. Skipping guidance on the last steps saves real time: the book’s example — skip CFG for the last 20 of 50 steps → **80 passes instead of 100**. Guidance-distilled models (FLUX.1-schnell/dev) bake guidance in: one pass per step.
- **Few-step distillation** (LCM, adversarial diffusion distillation as in SDXL-Turbo, FLUX.1-schnell): 1–8 steps instead of 25–50 — the single biggest speedup, with some quality/diversity cost.
- **Caching across steps**: neighboring steps produce similar activations, so methods like TeaCache and other “cache-DiT” approaches reuse block outputs when the change is small — ~30–40% faster in the book’s video example.

## Video: the extreme case

Video models denoise a 3-D latent (frames × height × width): tens to hundreds of thousands of tokens. The book’s profile of a video model: **attention is 70–80%** of time (quadratic in tokens), the VAE decoder 3–5%. Production runs **batch 1 across 8 GPUs** with **context/sequence parallelism** (split the tokens, M16), plus caching and FP8 attention. Metrics are seconds (and GPU-seconds, i.e. cost) per clip.

## Lab: steps and guidance on your Mac

```python
# diffusion_bench.py — SDXL-Turbo on Apple silicon (MPS)
import time, torch
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("mps")
prompt = "a cozy reading nook, watercolor"
pipe(prompt, num_inference_steps=1, guidance_scale=0.0)          # warmup
for steps, cfg in [(1, 0.0), (2, 0.0), (4, 0.0), (4, 2.0)]:     # cfg > 1 → two passes per step
    t = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, width=512, height=512).images[0]
    print(f"steps={steps} cfg={cfg}: {time.perf_counter()-t:.2f}s"); img.save(f"out_{steps}_{cfg}.png")
```

Expected pattern: time ≈ fixed (text encoders + VAE decode) + steps × per-step; turning CFG on roughly doubles the per-step part. For a modern DiT on Mac, try **mflux** (MLX port of FLUX): `mflux-generate --model schnell --prompt "..." --steps 2 -q 8` and compare 2 vs 4 steps.

- [ ] Plot: latency vs steps (with and without CFG); fitted fixed cost and per-step cost
- [ ] Explained why batching 4 images gives far less than 4× throughput here compared with LLM decode
- [ ] Estimated GPU-seconds and cost per image for a cloud deployment of your chosen model
      */}),
      resources: [
        { title: "Diffusers — Apple silicon (MPS)", url: "https://huggingface.co/docs/diffusers/optimization/mps", type: "docs", note: "running pipelines on Mac, memory tips" },
        { title: "mflux", url: "https://github.com/filipstrand/mflux", type: "tool", note: "FLUX in MLX for Apple silicon" },
        { title: "Adversarial Diffusion Distillation (SDXL-Turbo)", url: "https://arxiv.org/abs/2311.17042", type: "paper", note: "1–4 step generation" },
        { title: "TeaCache", url: "https://github.com/ali-vilab/TeaCache", type: "repo", note: "training-free step caching for diffusion transformers" },
      ],
    },
    {
      id: "voice-agent",
      title: "Realtime voice agents: a pipeline with a latency budget",
      kind: "build",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
## The budget

In human conversation the gap between turns is a few hundred milliseconds; voice agents feel natural below roughly **1 second voice-to-voice** (end of user speech → first agent audio) and awkward above ~1.5 s. A cascaded pipeline spends that budget like this:

| Stage | What happens | Typical budget | Main levers |
|---|---|---|---|
| **Endpointing** | Decide the user has finished (VAD silence threshold or a turn-detection model) | 200–500 ms | Shorter silence threshold (risk: cutting people off), semantic turn detection |
| **ASR** | Final transcript of the utterance | 100–300 ms | Streaming ASR, small/turbo models, GPU; transcribe while the user speaks |
| **LLM** | Time to first **sentence**, not first token | 200–400 ms | Low TTFT (prefix cache the system prompt!), fast small model, short answers |
| **TTS** | First audio from first sentence | 100–200 ms | Sentence chunking, streaming TTS, warm models |
| **Transport** | Audio over WebRTC/WebSockets, both ways | 50–150 ms | Region proximity, codecs, no TLS handshakes per turn |

Baseten (ch. 6.6, 7.2): run the whole pipeline **in one cluster/region**. A same-cluster hop costs ~10 ms vs ~50 ms across regions; five hops is 50 ms vs 250 ms — a quarter of a one-second budget gone to networking. Keep every model **warm** (no scale-to-zero on the critical path), and stream between stages so they overlap. End-to-end **speech-to-speech** models exist, but most production agents still use the cascade because each stage can be chosen, tuned and debugged independently.

## The overlap trick

Don’t run stages back to back. Stream the LLM’s tokens into a sentence splitter; as soon as the first sentence is complete, send it to TTS while the LLM keeps generating; play TTS audio chunks as they arrive. Voice-to-voice latency becomes ASR + time-to-first-sentence + TTS time-to-first-audio, not the sum of full stage times. Frameworks like **Pipecat** do this (plus interruption/barge-in handling) for you — but build it once by hand to understand where the time goes.

## A measurable single turn on the Mac

```python
# agent.py — wav in → Whisper → streaming LLM → sentence-chunked TTS → audio out, timed per stage
import re, sys, time, numpy as np, soundfile as sf, sounddevice as sd, mlx_whisper
from mlx_lm import load, stream_generate
from mlx_audio.tts.utils import load_model

ASR = "mlx-community/whisper-large-v3-turbo"
llm, tok = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
tts = load_model("mlx-community/Kokoro-82M-bf16")
SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."
SENT = re.compile(r"^(.+?[.!?])(\s+|$)")

def speak(text, out, T, t0):
    for r in tts.generate(text=text, voice="af_heart", lang_code="a"):
        T.setdefault("first_audio", time.perf_counter() - t0)
        out.append(np.array(r.audio))

def turn(wav):
    audio, sr = sf.read(wav, dtype="float32")                    # 16 kHz mono
    T, out, buf = {}, [], ""
    t0 = time.perf_counter()                                     # = end of user speech
    text = mlx_whisper.transcribe(audio, path_or_hf_repo=ASR)["text"].strip()
    T["asr"] = time.perf_counter() - t0
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": text}], add_generation_prompt=True)
    for r in stream_generate(llm, tok, prompt, max_tokens=120):
        T.setdefault("llm_first_token", time.perf_counter() - t0)
        buf += r.text
        m = SENT.match(buf)
        if m:
            T.setdefault("first_sentence", time.perf_counter() - t0)
            speak(m.group(1), out, T, t0); buf = buf[m.end():]
    if buf.strip(): speak(buf, out, T, t0)
    T["done"] = time.perf_counter() - t0
    print("user:", text); print({k: f"{v*1000:.0f} ms" for k, v in T.items()})
    sd.play(np.concatenate(out), 24000); sd.wait()

turn("warmup.wav")          # first call loads/compiles everything; ignore its timings
for f in sys.argv[1:]: turn(f)
```

This version is deliberately naive: TTS runs **inside** the token loop, so the LLM pauses while each sentence is synthesized, and audio plays only at the end. Measured *first_audio* is still honest for the first sentence. Your challenge is to make it properly concurrent and to add a live microphone with VAD.

## Metrics for the whole agent

- **Voice-to-voice latency** p50/p90 over ≥ 20 turns — the one number users feel.
- Per-stage latencies (the table above), so regressions are attributable.
- **Interruption handling**: time from the user starting to speak to agent audio stopping (barge-in).
- ASR word error rate on your own voice; LLM answer quality; TTS naturalness.
- At scale: concurrent conversations per GPU for each stage — they scale differently (M19 capacity planning per stage).

- [ ] Ran `agent.py` on ≥ 5 recorded questions; per-stage timing table
- [ ] Identified the largest stage and one change that cut it (smaller ASR model, shorter system prompt, shorter first sentence…)
- [ ] Explained why voice-to-voice ≠ sum of stage totals once stages overlap
      */}),
      resources: [
        { title: "Pipecat", url: "https://github.com/pipecat-ai/pipecat", type: "tool", note: "open-source framework for realtime voice agents (VAD, streaming, interruptions)" },
        { title: "Pipecat docs", url: "https://docs.pipecat.ai", type: "docs", note: "pipeline concepts, transports and latency tips" },
        { title: "mlx-lm", url: "https://github.com/ml-explore/mlx-lm", type: "tool", note: "stream_generate for the LLM stage on Mac" },
      ],
    },
  ],

  challenge: {
    title: "A voice-agent latency budget prototype on your Mac",
    md: MD(function () {/*
Build `course-work/m20-voice-agent/`: a local voice agent (microphone → VAD → ASR → LLM → TTS → speakers) with **measured, optimized per-stage latencies**.

1. **Live input**: capture the microphone with `sounddevice`, detect end of speech with **Silero VAD** (tune the silence threshold; record the value you chose and why).
2. **Concurrency**: run LLM generation, TTS synthesis and audio playback in separate threads/tasks connected by queues, so the LLM never waits for TTS and playback starts with the first chunk.
3. **Instrumentation**: timestamp every event (speech end, transcript ready, first token, first sentence, first audio chunk played) and log each turn as one JSON line.
4. **Budget report** (`README.md`): a target budget table (≤ 1 s voice-to-voice excluding endpointing), measured p50/p90 per stage over ≥ 20 turns, a waterfall chart for a typical turn, and three optimizations you tried with before/after numbers (e.g. whisper-turbo vs small, 1.5B vs 3B LLM, cached system prompt, sentence vs clause chunking, VAD threshold).
5. **Scale-out section**: how you would deploy this for 1,000 concurrent calls on GPUs — one cluster, per-stage engines, streams per GPU per stage, and where the Baseten-style co-location matters.
    */}),
    checklist: [
      "Live mic → speaker loop works end to end on a Mac with local models only",
      "Per-turn JSON logs with timestamps for every stage boundary",
      "p50/p90 voice-to-voice and per-stage latencies over ≥ 20 turns, plus a waterfall chart",
      "LLM, TTS and playback overlap (first audio before the LLM finishes), demonstrated in the logs",
      "Three optimizations with before/after measurements",
      "A written GPU deployment plan with per-stage capacity estimates",
    ],
    stretch: "Add barge-in (stop playback and cancel generation when the user starts speaking) and measure interruption latency; or replace Kokoro with a token-based TTS (Orpheus) on a cloud GPU served by vLLM, streaming audio back over a WebSocket, and compare time to first audio.",
  },

  connects: MD(function () {/*
Every technique from the LLM track reappears with a twist: batching and roofline (M07, M09) explain embeddings and diffusion; prefill optimizations (M09, M17) are VLM optimizations; decode speed (M13–M15) sets TTS capacity; the M18 serving stack and M19 benchmarking methodology apply per stage, with modality-specific metrics (items/s, RTF, time to first audio, seconds per image). The voice agent is also a preview of M25-style systems work: a product whose quality is defined by an end-to-end latency budget across several models.
  */}),

  interview: [
    "Why don’t prefix caching and prefill/decode disaggregation help an embedding service? What does help?",
    "How many tokens does a 1024×1024 image become in Qwen2.5-VL, and what does that do to TTFT and KV cache capacity?",
    "Define real-time factor. Why does a 5-second clip have a worse RTF than a 60-second clip with Whisper?",
    "A token-based TTS model emits 7 codec tokens per ~85 ms frame. What TPOT do you need for real-time playback, and how does that constrain batch size?",
    "Where does the time go in a diffusion model, and name three ways to cut latency without a new GPU.",
    "Design a voice agent with sub-second voice-to-voice latency. Give a per-stage budget and the tricks that make stages overlap.",
    "Why run all stages of a voice pipeline in one cluster? Quantify the network cost.",
  ],

  resources: [
    { title: "Inference Engineering (Baseten) — local PDF", url: "Inference%20Engineering.pdf", type: "book", note: "ch. 6 modalities; ch. 7 same-cluster pipelines" },
    { title: "vLLM — Multimodal inputs", url: "https://docs.vllm.ai/en/latest/features/multimodal_inputs.html", type: "docs", note: "serving VLMs and audio models" },
    { title: "Text Embeddings Inference", url: "https://huggingface.co/docs/text-embeddings-inference/index", type: "docs", note: "embedding/reranker serving" },
    { title: "Whisper paper", url: "https://arxiv.org/abs/2212.04356", type: "paper", note: "encoder–decoder ASR at scale" },
    { title: "WhisperKit", url: "https://github.com/argmaxinc/WhisperKit", type: "tool", note: "on-device streaming ASR for Apple platforms" },
    { title: "Hugging Face Diffusers", url: "https://github.com/huggingface/diffusers", type: "tool", note: "image/video generation pipelines and optimizations" },
    { title: "Pipecat", url: "https://github.com/pipecat-ai/pipecat", type: "tool", note: "voice agent framework" },
    { title: "Latent Consistency Models", url: "https://arxiv.org/abs/2310.04378", type: "paper", note: "few-step generation by distillation" },
  ],
});
