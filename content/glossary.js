/* glossary.js — course-wide glossary (route: #/x/glossary).
 *
 * Add a term: append ["Term", "module-id-or-empty", "Short markdown definition."] to G below.
 * Definitions are markdown with inline math ($...$ — write backslashes doubled, e.g. "$\\sqrt{d}$",
 * and a literal dollar as "\\$"). `module` is where the term is taught; it becomes a link.
 * Terms are also indexed by the global search box.
 */
(function () {
  "use strict";

  var G = [
    // ---- ML basics ----
    ["Tensor", "m01-neural-nets", "An n-dimensional array (scalar, vector, matrix, …) with a shape and a dtype. It is the unit of data in PyTorch, and every weight, activation and KV entry is one."],
    ["Gradient", "m01-neural-nets", "The vector of partial derivatives $\\nabla_\\theta L$. It points in the direction in which the loss increases fastest, so training steps go the opposite way."],
    ["Backpropagation", "m01-neural-nets", "Applying the chain rule backwards through the computation graph to get the gradient of the loss with respect to every parameter in a single backward pass."],
    ["Autograd", "m01-neural-nets", "PyTorch's automatic differentiation. It records the operations applied to tensors that have `requires_grad` set, then `loss.backward()` fills in `.grad` for each one."],
    ["Loss function", "m01-neural-nets", "A single number that measures how wrong the model is on a batch. Training minimises it."],
    ["Cross-entropy", "m02-language-models", "The standard classification and LM loss, $-\\log p_\\theta(y)$ for the correct class $y$. Averaged over tokens, it is the log of perplexity."],
    ["Softmax", "m02-language-models", "Turns logits into probabilities: $\\text{softmax}(z)_i = e^{z_i} / \\sum_j e^{z_j}$. In practice you subtract $\\max z$ first for numerical stability."],
    ["Logits", "m02-language-models", "The raw, unnormalised scores the model outputs, one per vocabulary token. Sampling applies temperature, top-k/p and then softmax to them."],
    ["Gradient descent", "m01-neural-nets", "The update rule $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta L$. The *stochastic* version (SGD) estimates the gradient from a mini-batch."],
    ["Learning rate", "m01-neural-nets", "The step size $\\eta$ in gradient descent. It is usually warmed up and then decayed; if it is too high, training diverges."],
    ["AdamW", "m05-training-finetuning", "The default LLM optimizer. It keeps per-parameter running averages of the gradient and its square, and applies weight decay separately. That is 2 extra fp32 states per weight, which matters for memory."],
    ["MLP", "m01-neural-nets", "Multi-layer perceptron: linear layers with non-linearities between them. In a transformer it is the feed-forward block, and it holds about two thirds of the parameters."],
    ["Embedding", "m02-language-models", "A learned lookup table that maps each token id to a vector of size $d_{model}$."],
    ["Residual connection", "m03-build-gpt", "$x + f(x)$: each block adds its output to its input. This gives gradients a highway through deep stacks."],
    ["LayerNorm / RMSNorm", "m04-modern-llms", "Normalise each token's activation vector. RMSNorm, used in Llama and Qwen, divides by $\\sqrt{\\text{mean}(x^2)}$ and skips mean-centering."],

    // ---- language models ----
    ["Token", "m02-language-models", "The unit an LLM reads and writes: a word piece produced by the tokenizer. For English text, 1 token is roughly ¾ of a word."],
    ["Tokenizer / BPE", "m02-language-models", "Byte-pair encoding repeatedly merges the most frequent adjacent byte pairs into new vocabulary entries. It maps text to token ids and back."],
    ["Autoregressive", "m02-language-models", "Generates one token at a time, with each token conditioned on all previous ones. This is why decode is sequential."],
    ["Perplexity", "m02-language-models", "$e^{\\text{cross-entropy}}$: the effective number of choices the model is hesitating between. Lower is better. It is used to check that quantization or fine-tuning did not hurt quality."],
    ["Temperature", "m02-language-models", "Divides the logits by $T$ before softmax. $T<1$ sharpens the distribution, $T>1$ flattens it, and $T \\to 0$ is greedy decoding."],
    ["Greedy decoding", "m02-language-models", "Always pick the argmax token. It is deterministic but can loop and repeat itself."],
    ["Top-k sampling", "m02-language-models", "Sample only from the $k$ most likely tokens, after renormalising their probabilities."],
    ["Top-p (nucleus) sampling", "m02-language-models", "Sample from the smallest set of tokens whose cumulative probability is at least $p$. The candidate set shrinks when the model is confident."],
    ["Context window", "m04-modern-llms", "The maximum number of tokens (prompt + generated) the model can attend over. Longer contexts mean a bigger KV cache."],
    ["Chat template", "m05-training-finetuning", "The model-specific special tokens that wrap system, user and assistant turns. A wrong template silently degrades output quality."],

    // ---- transformer ----
    ["Self-attention", "m03-build-gpt", "Each token builds a weighted mix of the other tokens' values: $\\text{softmax}(QK^\\top/\\sqrt{d_k})V$."],
    ["Query, key, value", "m03-build-gpt", "Three linear projections of each token. The query asks what I am looking for, the key says what I contain, and the value is what I hand over if selected."],
    ["Multi-head attention", "m03-build-gpt", "Runs several attention heads in parallel on slices of the hidden dimension, so each head can learn a different relation."],
    ["Causal mask", "m03-build-gpt", "Sets the scores for future positions to $-\\infty$ before softmax, so token $t$ only sees tokens $\\le t$."],
    ["Transformer block", "m03-build-gpt", "Norm → attention → residual, then norm → MLP → residual. An LLM is a stack of $L$ of these."],
    ["Decoder-only", "m03-build-gpt", "A transformer with only causal self-attention and no encoder. GPT, Llama, Qwen and DeepSeek are all decoder-only."],
    ["Positional encoding", "m03-build-gpt", "Tells attention where each token sits in the sequence. Options are learned absolute, sinusoidal or rotary (RoPE) encodings."],
    ["RoPE", "m04-modern-llms", "Rotary position embedding: rotates the query and key vectors by an angle proportional to position, so the $q\\cdot k$ dot product depends on relative distance."],
    ["GQA", "m04-modern-llms", "Grouped-query attention: several query heads share one key/value head. The KV cache shrinks by a factor of $n_{heads}/n_{kv\\_heads}$; MQA is the extreme case of a single shared KV head."],
    ["MLA", "m04-modern-llms", "Multi-head latent attention (DeepSeek): caches a small compressed latent per token instead of full keys and values, and up-projects it at attention time."],
    ["SwiGLU", "m04-modern-llms", "A gated MLP, $(\\text{SiLU}(xW_1) \\odot xW_3)W_2$, used in most modern LLMs."],
    ["Mixture of Experts (MoE)", "m04-modern-llms", "Replaces the MLP with $E$ expert MLPs and a router that sends each token to the top-$k$ of them. There are many parameters in total, but few are active per token."],
    ["Router (gating network)", "m17-disagg-moe", "The small layer in an MoE block that scores the experts for each token and picks the top-$k$. Load balance across experts is a serving concern."],
    ["Sliding-window attention", "m04-modern-llms", "Each token attends only to the last $W$ tokens, which caps the KV cache per layer at $W$ entries."],

    // ---- training / fine-tuning ----
    ["Pretraining", "m05-training-finetuning", "Next-token prediction on trillions of tokens of raw text. This is where most of a model's knowledge comes from."],
    ["SFT", "m05-training-finetuning", "Supervised fine-tuning on (prompt, ideal response) pairs. It teaches format and behaviour, such as following instructions."],
    ["LoRA", "m05-training-finetuning", "Low-rank adaptation: freezes $W$ and learns $\\Delta W = BA$ with a small rank $r$. The adapter is tiny and can be swapped per request."],
    ["QLoRA", "m05-training-finetuning", "LoRA trained on top of a 4-bit quantized base model, so large models can be fine-tuned on a single GPU or a Mac."],
    ["PEFT", "m05-training-finetuning", "Parameter-efficient fine-tuning: training only a small set of parameters (LoRA, adapters, prefixes). It is also the name of the Hugging Face library."],
    ["Mixed precision (BF16)", "m05-training-finetuning", "Compute in bfloat16 while keeping fp32 master weights. BF16 has fp32's exponent range, so it rarely needs loss scaling."],

    // ---- inference anatomy & metrics ----
    ["Prefill", "m06-inference-anatomy", "Processes the whole prompt in one parallel forward pass and fills the KV cache. It is compute-bound and sets TTFT."],
    ["Decode", "m06-inference-anatomy", "Generates one token per forward pass for each sequence. It reads all the weights and the KV cache each step, so it is memory-bandwidth-bound at small batch."],
    ["TTFT", "m06-inference-anatomy", "Time to first token: queueing + prefill + first sample. This is the latency users notice."],
    ["TPOT / ITL", "m06-inference-anatomy", "Time per output token, or inter-token latency: the gap between streamed tokens after the first. It sets the reading speed."],
    ["End-to-end latency", "m06-inference-anatomy", "$\\text{TTFT} + (n_{out}-1)\\cdot\\text{TPOT}$ for a request."],
    ["Throughput", "m06-inference-anatomy", "Tokens per second across all requests on a GPU or cluster. Batching raises it, usually at some cost to latency."],
    ["Goodput", "m19-benchmark-observe", "Throughput counting only requests that met their SLO (TTFT and TPOT). This is the metric that disaggregated serving optimises."],
    ["SLO", "m18-serving-production", "Service-level objective: a target such as p99 TTFT < 500 ms. The SLA is the contractual version."],
    ["Percentiles (p50 / p99)", "m19-benchmark-observe", "p99 is the latency that 99% of requests beat. Tail percentiles, not averages, are what break SLOs."],
    ["Arithmetic intensity", "m07-hardware-roofline", "FLOPs performed per byte moved from memory. Compare it with the hardware ridge point $\\text{FLOP/s} \\div \\text{B/s}$ to see which resource limits you."],
    ["Roofline model", "m07-hardware-roofline", "Attainable FLOP/s $= \\min(\\text{peak FLOP/s},\\ \\text{intensity}\\times\\text{bandwidth})$. It tells you whether a kernel is memory- or compute-bound."],
    ["Memory-bound", "m07-hardware-roofline", "Limited by memory bandwidth, not math: the arithmetic units wait for data. Decode at small batch is the classic example."],
    ["Compute-bound", "m07-hardware-roofline", "Limited by peak FLOP/s. Prefill and large-batch GEMMs are compute-bound."],
    ["HBM", "m07-hardware-roofline", "High-bandwidth memory stacked next to the GPU die: about 80–192 GB at 3–8 TB/s. Weights and the KV cache live here."],
    ["MFU / MBU", "m07-hardware-roofline", "Model FLOPs utilisation and model bandwidth utilisation: achieved FLOP/s or bytes/s as a fraction of peak. MBU is the one that matters for decode."],
    ["Batch size (inference)", "m09-batching-scheduling", "The number of sequences decoded together in one forward pass. Weights are read once per step, so a larger batch amortises them."],

    // ---- KV cache & memory ----
    ["KV cache", "m08-kv-cache-paging", "Stored keys and values for every past token in every layer, so decode never recomputes them. Size per token is $2 \\cdot L \\cdot n_{kv} \\cdot d_{head} \\cdot \\text{bytes}$."],
    ["PagedAttention", "m08-kv-cache-paging", "vLLM's idea: store the KV cache in fixed-size blocks (pages) with a per-sequence block table, like OS virtual memory. This removes almost all fragmentation."],
    ["Block table", "m08-kv-cache-paging", "The per-sequence map from logical token positions to physical KV blocks. The attention kernel reads through it."],
    ["Block manager", "m08-kv-cache-paging", "The allocator that hands out, reference-counts and frees KV blocks. It decides whether a new request fits."],
    ["Fragmentation", "m08-kv-cache-paging", "Memory wasted by over-reserving contiguous KV space for the maximum length (internal) or by gaps between allocations (external)."],
    ["Prefix caching", "m08-kv-cache-paging", "Reuses the KV blocks of an identical prompt prefix (such as a system prompt or earlier chat turns) across requests, skipping that part of prefill."],
    ["RadixAttention", "m08-kv-cache-paging", "SGLang's prefix cache: a radix tree over token sequences with LRU eviction. It matches the longest cached prefix automatically."],
    ["KV offloading", "m17-disagg-moe", "Moves cold KV blocks from HBM to CPU RAM or SSD and brings them back on reuse, so the cache can grow beyond GPU memory."],

    // ---- batching & scheduling ----
    ["Static batching", "m09-batching-scheduling", "Waits for a full batch and runs it until the longest sequence finishes. Slots sit idle as short requests end."],
    ["Continuous batching", "m09-batching-scheduling", "Iteration-level scheduling: requests join and leave the running batch at every decode step. This is the main throughput win of modern engines."],
    ["Chunked prefill", "m09-batching-scheduling", "Splits a long prompt's prefill into chunks and mixes them with decode steps, so one long prompt doesn't stall everyone's TPOT."],
    ["Preemption", "m09-batching-scheduling", "When KV memory runs out, the scheduler pauses a running sequence and either swaps its blocks to CPU or frees them to recompute later."],
    ["Scheduler", "m09-batching-scheduling", "Decides at every step which requests run, which wait and which are preempted, under a token budget and KV memory."],
    ["Token budget", "m09-batching-scheduling", "The maximum number of tokens (prefill + decode) processed in one engine step, e.g. vLLM's `max_num_batched_tokens`."],
    ["Streaming (SSE)", "m10-mini-engine", "Sends tokens to the client as they are generated, over server-sent events. This is how OpenAI-compatible APIs stream."],
    ["OpenAI-compatible API", "m10-mini-engine", "The de facto HTTP interface (`/v1/chat/completions`) that vLLM, SGLang and most providers implement."],

    // ---- GPU & CUDA ----
    ["Kernel", "m11-cuda-basics", "A function that runs on the GPU, launched over a grid of thread blocks. Every PyTorch op is one or more kernels."],
    ["Thread / block / grid", "m11-cuda-basics", "CUDA's hierarchy: threads are grouped into blocks, which share fast shared memory, and blocks form a grid that covers the problem."],
    ["Warp", "m11-cuda-basics", "32 threads that execute in lockstep. Divergent branches inside a warp run serially."],
    ["SM", "m11-cuda-basics", "Streaming multiprocessor: a GPU core cluster with its own registers, shared memory and tensor cores. An H100 has 132."],
    ["Occupancy", "m11-cuda-basics", "Active warps per SM as a fraction of the maximum. It is limited by registers and shared memory per block. Higher occupancy hides memory latency."],
    ["Memory coalescing", "m11-cuda-basics", "When the threads of a warp read consecutive addresses, the hardware merges the reads into a few wide transactions. Uncoalesced access wastes bandwidth."],
    ["Shared memory", "m11-cuda-basics", "Fast on-chip SRAM (hundreds of KB per SM) shared by a block. It is used to stage tiles so data is reused instead of re-read from HBM."],
    ["Tiling", "m11-cuda-basics", "Splitting a matmul or attention into tiles that fit in shared memory and registers, so each byte loaded from HBM is used many times."],
    ["Tensor cores", "m11-cuda-basics", "Units that do small matrix multiply-accumulates (e.g. 16×16) per instruction in FP16, BF16, FP8 or FP4. Most of a GPU's FLOPs come from them."],
    ["Kernel fusion", "m12-triton-flash", "Combining several ops into one kernel, so intermediates stay on-chip instead of making round trips to HBM."],
    ["Triton", "m12-triton-flash", "A Python DSL for writing GPU kernels at the level of tiles (blocks). The compiler handles threads, coalescing and shared memory."],
    ["FlashAttention", "m12-triton-flash", "An exact attention kernel that tiles $Q$, $K$ and $V$ and uses online softmax, never materialising the $T\\times T$ score matrix. It is IO-aware and memory-linear."],
    ["Online softmax", "m12-triton-flash", "Computes softmax in one pass by keeping a running max and sum and rescaling earlier partial sums when the max changes."],
    ["CUTLASS", "m11-cuda-basics", "NVIDIA's C++ template library for high-performance GEMMs and convolutions. CuTe is its layout algebra."],
    ["PTX / SASS", "m11-cuda-basics", "PTX is NVIDIA's virtual ISA and SASS is the real machine code. Reading SASS shows what the compiler actually did."],

    // ---- profiling & compile ----
    ["Nsight Systems", "m13-profiling-compile", "A timeline profiler (`nsys`) for CPU, GPU kernels, memcpy and NCCL. It finds gaps, launch overhead and serialisation."],
    ["Nsight Compute", "m13-profiling-compile", "A per-kernel profiler (`ncu`) that reports achieved bandwidth, occupancy, stall reasons and roofline position."],
    ["torch.compile", "m13-profiling-compile", "Captures a PyTorch program as a graph (TorchDynamo) and generates fused Triton kernels (Inductor)."],
    ["CUDA graphs", "m13-profiling-compile", "Record a sequence of kernel launches once and replay it with a single launch. This removes CPU launch overhead, which dominates small-batch decode."],
    ["Launch overhead", "m13-profiling-compile", "CPU time spent issuing kernels, a few µs each. With hundreds of tiny kernels per token, the GPU sits idle between them."],

    // ---- quantization ----
    ["Quantization", "m14-quantization", "Storing weights and/or activations in fewer bits (INT8, FP8, INT4, FP4), with scales, to cut memory and bandwidth and speed up decode."],
    ["Scale / zero-point", "m14-quantization", "The affine map $x \\approx s\\,(q - z)$ between integer codes $q$ and real values. It is chosen per tensor, per channel or per group."],
    ["Group-wise quantization", "m14-quantization", "One scale per group of, say, 128 weights. Outliers then only hurt their own group."],
    ["Weight-only quantization", "m14-quantization", "Weights are stored in low precision (e.g. W4A16) and dequantized on the fly. Decode gets faster because fewer bytes are read."],
    ["W8A8", "m14-quantization", "Both weights and activations in 8 bits (INT8 or FP8), so the GEMM itself runs on low-precision tensor cores."],
    ["FP8 (E4M3 / E5M2)", "m14-quantization", "8-bit floating point: E4M3 has more precision (weights and activations) and E5M2 more range (gradients). Natively supported from Hopper on."],
    ["NVFP4 / MXFP4", "m14-quantization", "4-bit float formats with a shared scale per micro-block of 16 or 32 values. NVFP4 adds a second-level scale. They are native on Blackwell."],
    ["GPTQ", "m14-quantization", "Post-training weight quantization that quantizes column by column and uses second-order (Hessian) information to compensate for the error."],
    ["AWQ", "m14-quantization", "Activation-aware weight quantization: scales up the few salient weight channels (found from activation magnitudes) before quantizing, so they lose less precision."],
    ["SmoothQuant", "m14-quantization", "Moves activation outliers into the weights with a per-channel scale, making both easy to quantize for W8A8."],
    ["Outlier features", "m14-quantization", "A few activation channels with huge magnitudes that ruin naive per-tensor INT8. This is why AWQ, SmoothQuant and per-group scales exist."],

    // ---- speculative decoding ----
    ["Speculative decoding", "m15-speculative-decoding", "A cheap drafter proposes $k$ tokens and the target model verifies all of them in one forward pass. With rejection sampling, the output distribution is exactly the target's."],
    ["Draft model", "m15-speculative-decoding", "The small, fast model (or head) that proposes tokens for speculative decoding."],
    ["Acceptance rate", "m15-speculative-decoding", "The fraction of drafted tokens the target accepts. Speedup grows with the expected accepted length per verification step."],
    ["EAGLE / Medusa", "m15-speculative-decoding", "Drafting heads attached to the target model's hidden states: Medusa uses parallel heads, EAGLE an autoregressive feature-level head. No separate model is needed."],
    ["Prompt lookup (n-gram) decoding", "m15-speculative-decoding", "Drafts by copying the continuation of a matching n-gram from the prompt. It needs no model and works well for editing or RAG."],

    // ---- distributed ----
    ["Tensor parallelism (TP)", "m16-parallelism", "Splits each weight matrix across GPUs (column- or row-wise), with an all-reduce per layer. It needs fast NVLink and is used within a node."],
    ["Pipeline parallelism (PP)", "m16-parallelism", "Puts different layers on different GPUs and passes activations between stages. It works across nodes but has pipeline bubbles."],
    ["Data parallelism (DP)", "m16-parallelism", "Full replicas that each serve (or train on) different requests. In serving, DP just means more replicas."],
    ["Expert parallelism (EP)", "m17-disagg-moe", "Places different MoE experts on different GPUs and routes tokens to them with all-to-all communication."],
    ["All-reduce", "m16-parallelism", "A collective that sums a tensor across all ranks and leaves the result on each. TP needs it after every sharded layer."],
    ["All-to-all", "m17-disagg-moe", "A collective where each rank sends a different slice to every other rank. It is the dispatch/combine step in EP."],
    ["NCCL", "m16-parallelism", "NVIDIA's collective communication library (all-reduce, all-gather, …) over NVLink, PCIe and InfiniBand."],
    ["NVLink", "m16-parallelism", "NVIDIA's high-bandwidth GPU-to-GPU interconnect, hundreds of GB/s per GPU and roughly 10× PCIe. It makes TP practical."],
    ["InfiniBand / RDMA", "m16-parallelism", "Low-latency networking between nodes. RDMA lets a NIC read or write remote GPU memory without involving the CPU, which KV transfer and weight sync rely on."],
    ["FSDP / ZeRO", "m23-distributed-rl", "Training-time sharding of parameters, gradients and optimizer states across data-parallel ranks, with each layer gathered just in time."],
    ["Context parallelism", "m16-parallelism", "Splits one very long sequence across GPUs (e.g. ring attention) so the attention for million-token contexts fits."],

    // ---- disaggregation & MoE serving ----
    ["Disaggregated serving", "m17-disagg-moe", "Runs prefill and decode on separate GPU pools and ships the KV cache between them, so each phase can be sized and tuned for its own SLO."],
    ["KV transfer", "m17-disagg-moe", "Moving a request's KV blocks from the prefill worker to the decode worker, typically over RDMA/NVLink (NIXL, Mooncake)."],
    ["NVIDIA Dynamo", "m17-disagg-moe", "An orchestration layer above vLLM, SGLang and TRT-LLM that adds disaggregated serving, KV-aware routing and SLA-based planning."],
    ["llm-d", "m18-serving-production", "Kubernetes-native distributed inference on vLLM: prefix-cache-aware routing, PD disaggregation, hierarchical KV offload and wide EP."],

    // ---- production ----
    ["vLLM", "m06-inference-anatomy", "The most widely used open-source LLM serving engine: PagedAttention, continuous batching, prefix caching, many quantization and parallelism backends."],
    ["SGLang", "m06-inference-anatomy", "A high-performance serving engine and frontend language known for RadixAttention, strong structured-output support and large-scale EP/PD deployments."],
    ["TensorRT-LLM", "m06-inference-anatomy", "NVIDIA's compiled, kernel-optimised LLM inference library. It is often the fastest on NVIDIA hardware, but less flexible."],
    ["Autoscaling", "m18-serving-production", "Adding or removing replicas based on load signals such as queue depth, KV utilisation or SLO headroom. It is bounded by cold-start time."],
    ["Cold start", "m18-serving-production", "Time from 'need a replica' to 'serving tokens': node, container, weight download and load, warm-up and CUDA graph capture."],
    ["KV-aware routing", "m18-serving-production", "Sends a request to the replica that already caches its prefix, balanced against load. It raises the cache hit rate and cuts TTFT."],
    ["Multi-LoRA serving", "m18-serving-production", "Serving many LoRA adapters on one base model in the same batch, with batched LoRA kernels (Punica, S-LoRA) and per-request adapter selection."],
    ["Multi-tenancy", "m18-serving-production", "Many customers on shared GPUs. It needs isolation, fair queuing, rate limits and per-tenant SLOs."],
    ["Canary / shadow deploy", "m24-continual-learning", "A canary routes a small share of live traffic to a new model version. Shadow mode mirrors traffic to it without returning its output. Both gate rollouts on real metrics."],
    ["Little's law", "m19-benchmark-observe", "$L = \\lambda W$: requests in flight equal arrival rate times time in system. Use it to size concurrency."],
    ["Cost per 1M tokens", "m19-benchmark-observe", "$\\text{GPU \\$/hour} \\div (\\text{tokens/s} \\times 3600) \\times 10^6$, measured at the throughput you can sustain within SLO."],
    ["Vision encoder", "m20-modalities", "A ViT that turns an image into patch embeddings, which are projected into the LLM's token space. Serving it adds an encoder stage before prefill."],

    // ---- RL ----
    ["Reinforcement learning", "m21-rl-fundamentals", "An agent learns a policy by acting in an environment and receiving rewards. It optimises expected return rather than imitating labels."],
    ["MDP", "m21-rl-fundamentals", "Markov decision process: states, actions, transition probabilities, rewards and discount $\\gamma$. It is the formal RL setting."],
    ["Policy", "m21-rl-fundamentals", "$\\pi_\\theta(a\\mid s)$: the agent's action distribution. For an LLM, the policy is the model and an action is a token."],
    ["Value function", "m21-rl-fundamentals", "$V(s)$ is the expected return from state $s$ under the policy. It is used as a baseline to reduce variance."],
    ["Advantage", "m21-rl-fundamentals", "$A(s,a) = Q(s,a) - V(s)$: how much better an action was than average. Policy gradients push up actions with positive advantage."],
    ["Policy gradient / REINFORCE", "m21-rl-fundamentals", "$\\nabla J = \\mathbb{E}[\\nabla \\log \\pi_\\theta(a\\mid s)\\, A]$: increase the log-probability of actions in proportion to how good they were."],
    ["PPO", "m21-rl-fundamentals", "Proximal policy optimization: a policy gradient with a clipped probability ratio that keeps each update close to the old policy. It is the classic RLHF algorithm."],
    ["Importance sampling", "m23-distributed-rl", "Reweights samples drawn from an old policy by $\\pi_{new}/\\pi_{old}$ so you can learn from slightly stale rollouts."],
    ["RLHF", "m22-rl-for-llms", "Reinforcement learning from human feedback: train a reward model on human preference pairs, then optimise the LLM against it (e.g. with PPO)."],
    ["Reward model", "m22-rl-for-llms", "A model that scores a response. It is trained on preference comparisons and stands in for a human judge during RL."],
    ["DPO", "m22-rl-for-llms", "Direct preference optimization: an RL-free loss on preference pairs that implicitly fits the RLHF objective. There is no reward model and no rollouts."],
    ["GRPO", "m22-rl-for-llms", "Group relative policy optimization: sample $G$ answers per prompt and use each one's reward minus the group mean (divided by std) as its advantage. No value network is needed."],
    ["RLVR", "m22-rl-for-llms", "RL with verifiable rewards: correctness checked by a program (math answers, unit tests) instead of a learned reward model."],
    ["KL penalty", "m22-rl-for-llms", "A term $\\beta\\,\\text{KL}(\\pi_\\theta \\| \\pi_{ref})$ that keeps the tuned policy close to the reference model and limits reward hacking and drift."],
    ["Reward hacking", "m22-rl-for-llms", "The policy exploits flaws in the reward, for example gaming a test or a length bias, instead of solving the task."],
    ["Rollout", "m22-rl-for-llms", "Generating responses (trajectories) from the current policy to be scored. In LLM RL this is mostly inference, and it dominates the wall-clock time."],
    ["Actor–learner architecture", "m23-distributed-rl", "Actors (inference engines) generate rollouts while learners (trainers) update the weights. The two are decoupled and connected by queues and weight sync."],
    ["Weight sync", "m23-distributed-rl", "Pushing freshly trained weights from the trainer into the rollout engines (NCCL broadcast, RDMA or shared memory) without restarting them."],
    ["Off-policy staleness", "m23-distributed-rl", "In async RL, rollouts come from weights that are a few steps old. You can bound the lag, or correct for it with importance ratios."],

    // ---- continual learning ----
    ["Continual learning", "m24-continual-learning", "Learning from a sequence of tasks or data distributions over time without retraining from scratch and without forgetting the earlier ones."],
    ["Catastrophic forgetting", "m24-continual-learning", "Performance on old tasks collapses after training on new ones, because the same shared weights get overwritten."],
    ["Stability–plasticity dilemma", "m24-continual-learning", "The tension between protecting old knowledge (stability) and absorbing new knowledge (plasticity). Every continual-learning method picks a point on this trade-off."],
    ["Task- / domain- / class-incremental", "m24-continual-learning", "The three continual-learning settings. The task id is given (task-IL), the input distribution shifts but the labels don't (domain-IL), or new classes must be told apart from all old ones with no task id (class-IL, the hardest)."],
    ["Accuracy matrix", "m24-continual-learning", "$R_{i,j}$ is the accuracy on task $j$ after training through task $i$. All continual-learning metrics are averages over parts of it."],
    ["Backward transfer (BWT)", "m24-continual-learning", "$\\frac{1}{T-1}\\sum_{j<T} (R_{T,j} - R_{j,j})$: how much later training changed accuracy on earlier tasks. A negative value means forgetting."],
    ["Forward transfer (FWT)", "m24-continual-learning", "How much having learned earlier tasks helps a task before (or while) you train on it, compared with a fresh model."],
    ["EWC", "m24-continual-learning", "Elastic weight consolidation: adds $\\frac{\\lambda}{2}\\sum_i F_i(\\theta_i-\\theta_i^*)^2$, a penalty that anchors the weights that were important for old tasks."],
    ["Fisher information", "m24-continual-learning", "In EWC, the average squared gradient of the log-likelihood for each weight, $F_i = \\mathbb{E}[(\\partial_i \\log p)^2]$. It estimates how important that weight is."],
    ["Experience replay", "m24-continual-learning", "Mixing a buffer of stored old examples into new training batches. It is simple, strong and the default baseline."],
    ["Generative replay", "m24-continual-learning", "Replaying synthetic samples of old tasks produced by a generative model (or by the LLM itself) instead of stored data."],
    ["Parameter isolation", "m24-continual-learning", "Giving each task its own parameters (adapters, LoRA per task, masks, progressive columns), so new learning can't overwrite old."],
    ["Model merging", "m24-continual-learning", "Combining fine-tuned weights without further training: weight averaging (model soups), task arithmetic or TIES."],
    ["Task arithmetic", "m24-continual-learning", "A task vector $\\tau = \\theta_{ft} - \\theta_{base}$. Adding task vectors combines skills; subtracting one removes a behaviour."],
    ["TIES-merging", "m24-continual-learning", "Merges task vectors by trimming small deltas, electing a sign per parameter and averaging only the values that agree, which reduces interference."],
    ["Continued pretraining", "m24-continual-learning", "Resuming next-token training on new-domain or fresher data. Re-warming the LR and replaying some original data limit forgetting."],
    ["Test-time training", "m24-continual-learning", "Updating some parameters on the test input itself (e.g. a self-supervised loss) before predicting, to adapt to that input."],
    ["S-LoRA / Punica", "m24-continual-learning", "Systems for serving thousands of LoRA adapters: unified paging of adapter weights and batched kernels (SGMV) that apply a different adapter per request."],
  ];

  var terms = G.map(function (r) { return { term: r[0], module: r[1] || null, def: r[2] }; });

  Course.extra("glossary", {
    eyebrow: "Reference",
    title: "Glossary",
    lead: "Every term the course uses, in one or two sentences, with a link to the module that teaches it. Type to filter; the global search (`/`) also finds these.",
    terms: terms,
    render: function (ctx) {
      var h = ctx.h;
      var names = (Course.extras.skills && Course.extras.skills.moduleNames) || {};
      function modName(id) { var m = ctx.mod(id); return m ? (m.short || m.title) : (names[id] || id); }

      var sorted = terms.slice().sort(function (a, b) { return a.term.toLowerCase().localeCompare(b.term.toLowerCase()); });
      var groups = {};
      sorted.forEach(function (t) {
        var c = t.term.charAt(0).toUpperCase();
        if (!/[A-Z]/.test(c)) c = "#";
        (groups[c] = groups[c] || []).push(t);
      });
      var letters = Object.keys(groups).sort();

      var rows = [];
      var sections = letters.map(function (L) {
        var items = groups[L].map(function (t) {
          var def = ctx.md("div", "prose", t.def);
          def.style.fontSize = "14.5px";
          Array.prototype.forEach.call(def.querySelectorAll("p"), function (p) { p.style.margin = "0"; });
          var row = h("div", { class: "res-row", style: { gridTemplateColumns: "210px 1fr 190px" } },
            h("b", { text: t.term }),
            def,
            t.module ? h("a", { class: "res-mod", href: "#/m/" + t.module },
              h("span", { class: "mod-num sm", text: ctx.ORDER.indexOf(t.module) >= 0 ? ctx.moduleNum(t.module) : "··" }), modName(t.module)) : h("span"));
          rows.push({ el: row, text: (t.term + " " + t.def + " " + (t.module || "")).toLowerCase() });
          return row;
        });
        var sec = h("section", { class: "track-sec", style: { margin: "18px 0" } },
          h("h3", { style: { color: "var(--accent)", fontFamily: "var(--mono)" }, text: L }), items);
        sec._items = items;
        return sec;
      });

      var count = h("span", { class: "muted small" });
      var input = h("input", { type: "search", placeholder: "Filter " + terms.length + " terms…", "aria-label": "Filter glossary" });
      function filter() {
        var q = input.value.trim().toLowerCase(), shown = 0;
        rows.forEach(function (r) { var ok = !q || r.text.indexOf(q) >= 0; r.el.style.display = ok ? "" : "none"; if (ok) shown++; });
        sections.forEach(function (s) { s.style.display = s._items.some(function (i) { return i.style.display !== "none"; }) ? "" : "none"; });
        count.textContent = shown + " of " + terms.length + " terms";
      }
      input.addEventListener("input", filter);

      var jump = h("div", { style: { display: "flex", flexWrap: "wrap", gap: "4px" } }, letters.map(function (L, i) {
        return h("a", { href: "javascript:void(0)", class: "hw", text: L, onclick: function () { sections[i].scrollIntoView({ behavior: "smooth", block: "start" }); } });
      }));

      filter();
      return [h("div", { class: "res-filters" }, input, count), jump].concat(sections);
    },
  });
})();
