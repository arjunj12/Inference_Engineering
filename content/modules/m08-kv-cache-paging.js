Course.module({
  id: "m08-kv-cache-paging",
  title: "KV cache management: PagedAttention & prefix caching",
  short: "Paged KV & prefix caching",
  tagline: "Build the heart of vLLM: a block-based KV cache manager with block tables, a free list, ref-counting, copy-on-write and hash-chained prefix caching — and plug it into your own model.",
  hours: 16,
  runsOn: ["mac", "colab"],
  tags: ["kv-cache", "paged-attention", "prefix-caching", "vllm", "sglang", "memory-management"],

  goal: MD(function () {/*
You'll write `block_manager.py` and `prefix_cache.py` — a few hundred lines of pure Python that implement the same ideas as vLLM's `vllm/v1/core/block_pool.py` and `kv_cache_manager.py` — then make your Module 06 GPT read its KV cache **through a block table**, and prove it produces exactly the same tokens:

```text
$ pytest -q test_block_manager.py test_prefix_cache.py test_paged.py
............                                                             [100%]
12 passed

$ python simulate.py
{'hit_rate': 0.702, 'preemptions': 194, 'decode_steps': 1093,
 'paged_mean_concurrency': 112.0, 'paged_tokens_per_slot': 1.99, 'naive_max_concurrency': 37}
```

Read that last line as: on the same memory, **naive contiguous allocation fits 37 requests; paging + prefix sharing runs ~112 concurrently**, 70% of prompt tokens are served from cache, and each physical KV slot backs ~2 logical tokens because system prompts are shared. That's why vLLM exists.

::viz paged-attention
  */}),
  demo: { viz: "paged-attention", params: {} },

  why: MD(function () {/*
Module 07 showed that throughput comes from **batch size**, and batch size is limited by **KV-cache memory**. The PagedAttention paper found that serving systems of the time wasted 60–80% of that memory on fragmentation and over-reservation. Fix the memory manager and you fit several times more sequences on the same GPU — a bigger win than most kernel work. Prefix caching then turns repeated system prompts, tool definitions and multi-turn history into free prefill, which is why every API now prices cached input tokens lower (Inference Engineering (Baseten) ch. 5.3). "Design a KV cache manager" and "how does prefix caching work?" are standard inference-engineer interview questions; after this module you'll have implemented both and can point to the exact vLLM functions that do the same thing.
  */}),

  prereqs: [
    {
      title: "Module 06 KV cache + Module 07 memory math",
      skipIf: "you have `gpt_kv.py` from Module 06 and can compute KV bytes per token",
      md: MD(function () {/*
You need `gpt_kv.py` from Module 06 (the `GPT` whose `forward(idx, cache)` calls `cache.update(layer, k, v)` and reads `cache.pos`, plus `generate_cached`). Our paged store will **duck-type** that `KVCache` interface, so the model code doesn't change at all. You should also be able to compute $2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b$ bytes per token and explain why batch size drives throughput (Module 07, *batching-intensity*).
      */}),
    },
    {
      title: "Operating-system paging in five minutes",
      skipIf: "you know virtual memory, pages and page tables",
      md: MD(function () {/*
Programs see a contiguous **virtual** address space, but the OS backs it with fixed-size **physical pages** (4 KiB) scattered anywhere in RAM. A **page table** maps virtual page → physical page. Benefits: no need for a big contiguous free region (no external fragmentation), memory is allocated only when touched (no over-reservation), and two processes can map the **same** physical page (shared libraries, `fork()` with **copy-on-write**: pages are shared until one side writes, then that page alone is copied). PagedAttention applies exactly this to the KV cache: *virtual* = token positions of a sequence, *physical* = KV blocks in GPU memory, *page table* = **block table**.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: vLLM's KV cache budget and a prefix-cache hit on Colab",
      kind: "demo",
      minutes: 60,
      runsOn: ["colab"],
      md: MD(function () {/*
Start a real engine and read what it tells you about memory. On Colab (T4 or better):

```bash
pip install vllm
nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 8192 > vllm.log 2>&1 &
sleep 90; grep -E "KV cache size|Maximum concurrency|gpu_memory_utilization" vllm.log
```

You'll see two lines like:

```text
GPU KV cache size: 9xx,xxx tokens
Maximum concurrency for 8,192 tokens per request: 1xx.xx x
```

**Predict it first.** vLLM takes `gpu_memory_utilization` (default **0.92**) of the GPU, subtracts weights (~1 GB), activation peak and CUDA-graph memory (~1–2 GB), and hands the rest to the KV cache. A T4 has ~15 GB usable: $15 \times 0.92 - 1 - 1.5 \approx 11$ GB. Qwen2.5-0.5B stores $2 \times 24 \times 2 \times 64 \times 2 = 12{,}288$ bytes per token (24 layers, 2 KV heads, head dim 64, FP16). $11 \times 10^9 / 12{,}288 \approx 0.9$ M tokens. The "maximum concurrency" line is just that number ÷ `max-model-len`.

> [!NOTE] T4 compatibility
> Recent vLLM releases are steadily dropping pre-Ampere GPUs. If the server refuses to start on a T4, switch the runtime to an L4 or A100 — the log lines are the same (with a larger number).

**Now watch prefix caching work.** Automatic prefix caching is **on by default** in vLLM V1. Send two requests that share a ~3K-token system prompt and measure time-to-first-token:

```python
# prefix_demo.py
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
system = "You are ACME's support bot. " + " ".join(
    f"Policy {i}: items in category {i} can be returned within {i % 30 + 1} days." for i in range(250))

def ttft(question):
    t0, first = time.perf_counter(), None
    stream = client.chat.completions.create(
        model=MODEL, max_tokens=16, stream=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}])
    for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - t0
    return first

print(f"cold : {1e3 * ttft('Can I return a toaster?'):.0f} ms")
print(f"warm : {1e3 * ttft('Can I return a kettle?'):.0f} ms")    # same system prompt, new question
print(f"warm : {1e3 * ttft('What about socks?'):.0f} ms")
```

Then restart the server with `--no-enable-prefix-caching` and run it again. With caching, the second and third TTFTs drop sharply (only the new question is prefilled); without it, every request pays the full prefill. Check the server's counters too: `curl -s localhost:8000/metrics | grep prefix_cache` shows queried vs hit tokens.

> [!INTUITION] What just happened
> The first request's prompt was split into 16-token blocks, and each **full** block was given a hash that depends on *all tokens before it*. The second request computed the same hashes for its first ~3K tokens, found them in a hash table, and pointed its block table at the **existing** physical blocks — no recomputation, no copy. You'll build exactly that in Lesson 6.

- [ ] Recorded "GPU KV cache size" and "Maximum concurrency", and matched them to your own prediction (±20%)
- [ ] Measured TTFT cold vs warm with prefix caching on and off
- [ ] Found the prefix-cache counters in `/metrics`
      */}),
      resources: [
        { title: "vLLM — Automatic Prefix Caching (design)", url: "https://docs.vllm.ai/en/latest/design/prefix_caching/", type: "docs", note: "how vLLM hashes and reuses blocks" },
        { title: "vLLM — Optimization and tuning", url: "https://docs.vllm.ai/en/latest/configuration/optimization.html", type: "docs", note: "gpu_memory_utilization, max_num_seqs, preemption" },
      ],
    },
    {
      id: "fragmentation",
      title: "The fragmentation problem and the PagedAttention idea",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
Your Module 06 cache preallocated `[B, H, max_len, D]`. That's fine for one user. For a server it's a disaster:

1. **Over-reservation.** You don't know how long the answer will be, so you reserve `max_len` (say 8K) per request. The average request uses 1.5K. ~80% of the reservation is empty for the entire request lifetime.
2. **Internal fragmentation.** Even after the request finishes, the unused tail of its reservation was never usable by anyone else.
3. **External fragmentation.** Requests of different sizes come and go; free memory ends up in holes too small for the next big contiguous reservation, even when the total free memory is plenty.

Kwon et al. measured this in real systems: only **20.4%–38.2%** of KV-cache memory actually held token states. The rest was waste.

::viz paged-attention

**PagedAttention** (Kwon et al., SOSP 2023) fixes all three:
- Split every sequence's KV cache into fixed-size **blocks** of $B$ tokens (vLLM default: **16**).
- Keep a pool of physical blocks and a **free list**. Allocate a block only when a sequence actually needs one (every $B$ tokens).
- Give each sequence a **block table**: logical block $j$ → physical block id. Token $i$ lives in physical block `table[i // B]` at offset `i % B`.
- The attention kernel reads K/V **through the block table**, so blocks can sit anywhere in memory.

Waste is now at most one partially filled block per sequence — under 4% in the paper's measurements — and blocks become the unit of **sharing**: two sequences can point to the same physical block (parallel sampling, beam search, and — the big one — shared prefixes).

> [!MATH] How much does paging save?
> Naive: each request reserves $L_{max}$ tokens. Paged: each holds $\lceil L / B \rceil \cdot B$ tokens, where $L$ is its *current* length.
> With $L_{max} = 8192$, average current length $\bar L = 1500$, $B = 16$: naive utilization $= 1500 / 8192 \approx 18\%$; paged $\approx 1500 / 1508 \approx 99.5\%$. On the same memory the paged system holds ~5× more requests — and Module 07 says ~5× batch is several times more throughput.

> [!REAL] Results
> The paper reports **2–4×** higher throughput than FasterTransformer and Orca at the same latency, and the launch blog showed up to **24×** over naive Hugging Face serving. Memory sharing cut memory for parallel sampling and beam search by up to **55%**. Today every serious engine — vLLM, SGLang, TensorRT-LLM, LMDeploy, TGI — uses paged KV. FlashAttention and FlashInfer kernels accept a `block_table` argument directly.

- [ ] Can name the three kinds of KV waste and which part of PagedAttention removes each
- [ ] Can compute token $i$'s physical location from a block table
- [ ] Can explain why block-level sharing is the enabler for prefix caching
      */}),
      resources: [
        { title: "Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention", url: "https://arxiv.org/abs/2309.06180", type: "paper", note: "read sections 3–4; the fragmentation figure is the whole motivation" },
        { title: "vLLM launch blog (2023)", url: "https://blog.vllm.ai/2023/06/20/vllm.html", type: "article", note: "the short, visual version" },
      ],
    },
    {
      id: "block-manager-design",
      title: "Block manager design: block size, tables, free lists, eviction, preemption",
      kind: "concept",
      minutes: 60,
      md: MD(function () {/*
Before coding, make the design decisions a real engine makes.

**Data structures.**
- `blocks[id]` with a **ref_count** (how many sequences' block tables point at it) and, for prefix caching, a **hash**.
- A **free list** of blocks with `ref_count == 0`. vLLM V1 uses a doubly-linked `FreeKVCacheBlockQueue` so it can pop from the front *and* remove from the middle in O(1) (the middle case happens when a free-but-cached block gets reused by a prefix hit). We'll use Python's `OrderedDict`, which gives the same two operations.
- Per sequence: a **block table** (list of block ids) and a token count.

**Block size trade-off.** Small blocks → less internal waste and finer-grained prefix sharing, but longer block tables, more metadata per step and less efficient kernels (each block is a separate memory region to gather). Large blocks → the opposite. The PagedAttention paper found 16–32 a good middle; vLLM's default is **16** tokens on GPU. Prefix caching works on **full blocks only**, so with $B = 16$ a 1,000-token shared prompt shares 992 tokens.

**The per-step allocation API** (vLLM's `KVCacheManager.allocate_slots`): before each forward pass the scheduler asks "can sequence *s* grow by *n* tokens?". The manager computes how many new blocks that needs; if the free list is too short it says no, and the scheduler must **preempt** someone.

**Preemption: swap vs recompute.** When memory runs out mid-generation:
- **Swap**: copy the victim's blocks to CPU RAM, bring them back later. Costs PCIe bandwidth (~25–64 GB/s) both ways.
- **Recompute**: free the victim's blocks, put it back in the waiting queue with prompt + tokens-generated-so-far as its new prompt, and prefill again later. Costs FLOPs, but prefill is fast and compute-bound (Module 07). vLLM V0 supported both; **V1 uses recompute**. With prefix caching, the recompute often hits cache for most of the prompt anyway.
- Victim choice: vLLM preempts the **lowest-priority / most recently admitted** running request, so older requests keep making progress (no livelock).
- Engines also keep a small **watermark** of free blocks so admitting a new request doesn't instantly force a preemption.

**Forking and copy-on-write.** For parallel sampling ($n > 1$) or beam search, children share the parent's blocks (ref_count += 1). A child that appends into a shared, partially filled last block must **copy** it first. Full blocks are never written again, so they stay shared forever. (vLLM V1 implements parallel sampling by fanning out child requests and relying on prefix caching to share the prompt blocks; the explicit CoW version below is the textbook/V0 design and is still the clearest way to learn it.)

**Eviction (with prefix caching).** A freed block keeps its hash and contents: it's *free but cached*. When a new block is needed, take the one at the **front** of the free list — the least recently freed. Free sequences tail-first, so a sequence's deep blocks become evictable before its prefix blocks: shared prefixes survive longest. That's an LRU policy with a good tie-break — exactly vLLM's.

> [!IMPORTANT] Invariants worth unit-testing
> 1. Every block is either on the free list (ref 0) or referenced by ≥ 1 block table (ref ≥ 1) — never both, never neither.
> 2. Sum of ref_counts = total entries across all block tables.
> 3. A failed allocation leaves the state unchanged.
> 4. Two different live sequences never write to the same slot.

- [ ] Can justify a block size of 16 in terms of waste, sharing granularity and kernel efficiency
- [ ] Can compare swap vs recompute preemption and say which vLLM V1 uses
- [ ] Wrote down the four invariants you'll test
      */}),
      resources: [
        { title: "vLLM source — block_pool.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py", type: "repo", note: "BlockPool: get_new_blocks, free_blocks, cache_full_blocks, eviction" },
      ],
    },
    {
      id: "build-block-manager",
      title: "Build: a BlockManager with ref-counting, fork and copy-on-write",
      kind: "build",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
Pure bookkeeping, no tensors yet — that's how real engines are structured too (the scheduler process never touches KV memory; it just produces block tables and slot ids for the workers).

```python
# block_manager.py — a vLLM-style block manager (no tensors, just bookkeeping)
from collections import OrderedDict
from dataclasses import dataclass


class OutOfBlocks(Exception):
    pass


@dataclass
class Block:
    id: int
    ref_count: int = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        # free list: OrderedDict = O(1) pop-from-front AND O(1) remove-from-middle (needed for prefix caching)
        self.free: "OrderedDict[int, None]" = OrderedDict((i, None) for i in range(num_blocks))
        self.tables: dict[str, list[int]] = {}   # seq_id -> block table (logical block j -> physical id)
        self.lengths: dict[str, int] = {}        # seq_id -> number of tokens stored

    # ---------- helpers
    def num_free(self) -> int:
        return len(self.free)

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)   # ceil division

    def _take(self) -> int:
        if not self.free:
            raise OutOfBlocks()
        bid, _ = self.free.popitem(last=False)   # oldest free block first
        self.blocks[bid].ref_count = 1
        return bid

    def _release(self, bid: int) -> None:
        b = self.blocks[bid]
        b.ref_count -= 1
        assert b.ref_count >= 0, f"double free of block {bid}"
        if b.ref_count == 0:
            self.free[bid] = None                # goes to the BACK: most recently freed = evicted last

    # ---------- public API
    def can_allocate(self, n_tokens: int) -> bool:
        return self.blocks_needed(n_tokens) <= self.num_free()

    def allocate(self, seq_id: str, n_tokens: int) -> list[int]:
        assert seq_id not in self.tables, f"{seq_id} already allocated"
        if not self.can_allocate(n_tokens):
            raise OutOfBlocks()
        self.tables[seq_id] = [self._take() for _ in range(self.blocks_needed(n_tokens))]
        self.lengths[seq_id] = n_tokens
        return self.tables[seq_id]

    def append_slot(self, seq_id: str):
        """Reserve room for ONE more token of seq_id.
        Returns (slot, copy): slot = block_id * block_size + offset (where the new K/V goes);
        copy = (src_block, dst_block) if copy-on-write happened, else None."""
        table, n = self.tables[seq_id], self.lengths[seq_id]
        offset = n % self.block_size
        copy = None
        if offset == 0:                          # last block is full (or there is none): new block
            table.append(self._take())
        else:
            last = table[-1]
            if self.blocks[last].ref_count > 1:  # shared with a fork -> copy-on-write
                new = self._take()
                self._release(last)
                table[-1] = new
                copy = (last, new)
        self.lengths[seq_id] = n + 1
        return table[-1] * self.block_size + offset, copy

    def fork(self, parent: str, child: str) -> None:
        """Child shares every block of the parent (parallel sampling n>1, beam search)."""
        assert child not in self.tables
        self.tables[child] = list(self.tables[parent])
        self.lengths[child] = self.lengths[parent]
        for bid in self.tables[child]:
            self.blocks[bid].ref_count += 1

    def free_seq(self, seq_id: str) -> None:
        for bid in reversed(self.tables.pop(seq_id)):   # tail first -> prefix blocks stay cached longest
            self._release(bid)
        del self.lengths[seq_id]

    def slot_ids(self, seq_id: str) -> list[int]:
        bs, table = self.block_size, self.tables[seq_id]
        return [table[i // bs] * bs + i % bs for i in range(self.lengths[seq_id])]

    def stats(self) -> dict:
        fill: dict[int, int] = {}
        for sid, table in self.tables.items():
            n = self.lengths[sid]
            for j, bid in enumerate(table):
                fill[bid] = max(fill.get(bid, 0), min(self.block_size, n - j * self.block_size))
        used = len(fill)
        return {"used_blocks": used, "free_blocks": self.num_free(),
                "slot_utilization": sum(fill.values()) / max(1, used * self.block_size)}
```

Notes on the choices:
- **Slots** are the flat index `block_id * block_size + offset` — the same "slot mapping" vLLM passes to its `reshape_and_cache` kernel, which writes each new token's K/V into place.
- `append_slot` does its `_take()` **before** mutating anything, so an `OutOfBlocks` leaves the sequence untouched (invariant 3).
- On CoW we return `(src, dst)` instead of copying: the manager owns no tensors. The caller (the model runner) does the actual block copy. vLLM V0 batched these as "blocks_to_copy" per step.

**Tests.** The random test is the important one — it hammers allocate/fork/append/free and checks the invariants after every operation.

```python
# test_block_manager.py — run with:  pytest -q test_block_manager.py
import random
import pytest
from block_manager import BlockManager, OutOfBlocks


def test_allocate_and_free():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate("a", 10)                          # ceil(10/4) = 3 blocks
    assert len(bm.tables["a"]) == 3 and bm.num_free() == 5
    bm.free_seq("a")
    assert bm.num_free() == 8


def test_append_crosses_block_boundary():
    bm = BlockManager(8, 4)
    bm.allocate("a", 4)                           # exactly one full block
    slot, copy = bm.append_slot("a")              # token #5 needs a second block
    assert len(bm.tables["a"]) == 2 and copy is None
    assert slot == bm.tables["a"][1] * 4          # offset 0 of the new block


def test_fork_copy_on_write():
    bm = BlockManager(8, 4)
    bm.allocate("p", 6)                           # 2 blocks, the last one half full
    bm.fork("p", "c")
    assert all(bm.blocks[b].ref_count == 2 for b in bm.tables["p"])
    _, copy = bm.append_slot("c")                 # child writes into the shared tail -> CoW
    assert copy is not None and copy[0] == bm.tables["p"][1]
    assert bm.tables["c"][0] == bm.tables["p"][0] # the full block is still shared
    _, copy = bm.append_slot("p")                 # parent is now sole owner of its tail
    assert copy is None


def test_out_of_blocks():
    bm = BlockManager(2, 4)
    bm.allocate("a", 8)
    with pytest.raises(OutOfBlocks):
        bm.append_slot("a")
    assert bm.lengths["a"] == 8                   # failed append must not corrupt state


def test_invariants_random():
    rng, bm, live = random.Random(0), BlockManager(64, 4), []
    for step in range(3000):
        op = rng.random()
        try:
            if op < 0.2:
                sid = f"s{step}"
                bm.allocate(sid, rng.randint(1, 20))
                live.append(sid)
            elif op < 0.3 and live:
                child = f"f{step}"
                bm.fork(rng.choice(live), child)
                live.append(child)
            elif op < 0.45 and live:
                bm.free_seq(live.pop(rng.randrange(len(live))))
            elif live:
                bm.append_slot(rng.choice(live))
        except OutOfBlocks:
            pass
        # every reference is accounted for, and free == blocks with no owners
        assert sum(b.ref_count for b in bm.blocks) == sum(len(t) for t in bm.tables.values())
        assert bm.num_free() == sum(1 for b in bm.blocks if b.ref_count == 0)
```

> [!TIP] Break it on purpose
> Delete the `copy-on-write` branch and rerun: `test_fork_copy_on_write` fails, but so should an invariant-4 test if you write one (two live sequences mapping the same slot for their *last* token). Writing a test that catches a bug you planted is the fastest way to trust a test suite.

- [ ] `block_manager.py` written; all 5 tests pass
- [ ] Added a test for invariant 4 (no two live sequences' newest slots collide)
- [ ] Can explain why `free_seq` releases blocks tail-first
      */}),
      resources: [
        { title: "nano-vllm — block_manager.py", url: "https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py", type: "repo", note: "a ~100-line production-shaped block manager with prefix hashing — compare with yours" },
      ],
    },
    {
      id: "build-paged-attention",
      title: "Build: paged attention via gather, plugged into your GPT",
      kind: "build",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
Now give the block manager real memory. One big tensor per layer holds **every** sequence's K/V as a flat array of slots; each sequence's block table tells us which slots are its.

```python
# paged.py — paged KV storage + a view that looks like m06's KVCache to the model
import torch
from gpt_kv import GPT, generate_cached
from block_manager import BlockManager
from prefix_cache import PrefixCachingBlockManager   # written in Lesson 6; drop this import until then


class PagedKVStore:
    """All KV memory for all sequences: per layer a flat array of slots [num_blocks * block_size, H, D]."""
    def __init__(self, n_layer, num_blocks, block_size, n_head, head_dim, device, dtype):
        self.block_size = block_size
        self.k = torch.zeros(n_layer, num_blocks * block_size, n_head, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)

    def copy_block(self, src, dst):
        bs = self.block_size
        self.k[:, dst * bs:(dst + 1) * bs] = self.k[:, src * bs:(src + 1) * bs]
        self.v[:, dst * bs:(dst + 1) * bs] = self.v[:, src * bs:(src + 1) * bs]


class SeqView:
    """Duck-types m06's KVCache (.pos + .update) for ONE sequence, backed by paged storage."""
    def __init__(self, store, slots, pos):
        self.store, self.slots, self.pos = store, slots, pos   # slots: LongTensor, every slot of the sequence

    def update(self, layer, k_new, v_new):                     # k_new: [1, H, T, D]
        T = k_new.size(2)
        new = self.slots[-T:]                                  # the last T slots belong to the new tokens
        self.store.k[layer][new] = k_new[0].transpose(0, 1)    # scatter -> [T, H, D]
        self.store.v[layer][new] = v_new[0].transpose(0, 1)
        K = self.store.k[layer][self.slots].transpose(0, 1).unsqueeze(0)   # gather -> [1, H, S, D]
        V = self.store.v[layer][self.slots].transpose(0, 1).unsqueeze(0)
        return K, V


@torch.no_grad()
def generate_paged(model, bm, store, seq_id, prompt, n_new):
    dev = next(model.parameters()).device
    if isinstance(bm, PrefixCachingBlockManager):
        n_cached = bm.allocate(seq_id, prompt)
        append = lambda tok: bm.append_token(seq_id, tok)
    else:
        bm.allocate(seq_id, len(prompt)); n_cached = 0
        append = lambda tok: bm.append_slot(seq_id)
    slots = torch.tensor(bm.slot_ids(seq_id), device=dev)
    x = torch.tensor([prompt[n_cached:]], device=dev)        # prefill only the UNcached suffix
    logits = model(x, SeqView(store, slots, n_cached))
    out = []
    while True:
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if len(out) == n_new:
            break
        _, copy = append(nxt)
        if copy:
            store.copy_block(*copy)
        slots = torch.tensor(bm.slot_ids(seq_id), device=dev)
        logits = model(torch.tensor([[nxt]], device=dev), SeqView(store, slots, len(slots) - 1))
    return out
```

How it works: `update` **scatters** the new tokens' K/V into their slots, then **gathers** all of the sequence's slots in order into a contiguous `[1, H, S, D]` tensor that SDPA can consume. Your Module 06 model code is untouched — the mask logic (`tril(diagonal=S-T)`) and position offsets (`cache.pos`) work the same, because `pos` is set to the number of tokens already stored.

The correctness test: paged generation must produce **exactly** the tokens contiguous generation does.

```python
# test_paged.py
import torch
from gpt_kv import GPT, generate_cached
from block_manager import BlockManager
from prefix_cache import PrefixCachingBlockManager
from paged import PagedKVStore, generate_paged

torch.manual_seed(0)
model = GPT(vocab_size=65, block_size=512).eval()
HD = 384 // 6

def new_store(bm):
    return PagedKVStore(model.n_layer, len(bm.blocks), bm.block_size, model.n_head, HD, "cpu", torch.float32)

def test_paged_matches_contiguous():
    prompt = torch.randint(0, 65, (37,)).tolist()
    ref = generate_cached(model, torch.tensor([prompt]), 50)[0, 37:].tolist()
    bm = BlockManager(num_blocks=64, block_size=16)
    assert generate_paged(model, bm, new_store(bm), "a", prompt, 50) == ref

def test_prefix_cache_same_output():
    bm = PrefixCachingBlockManager(num_blocks=64, block_size=16)
    store = new_store(bm)
    system = torch.randint(0, 65, (40,)).tolist()
    generate_paged(model, bm, store, "r1", system + [1, 2, 3], 20); bm.free_seq("r1")
    p2 = system + [5, 6, 7, 8]
    got = generate_paged(model, bm, store, "r2", p2, 20)
    assert bm.hit_tokens == 32                            # 2 full blocks of the system prompt reused
    assert got == generate_cached(model, torch.tensor([p2]), 20)[0, len(p2):].tolist()

def test_fork_copy_on_write_data():
    bm = BlockManager(num_blocks=8, block_size=4)
    store = new_store(bm)
    bm.allocate("p", 6)
    slots = bm.slot_ids("p")
    store.k[:, slots] = torch.randn(model.n_layer, 6, model.n_head, HD)
    bm.fork("p", "c")
    slot, copy = bm.append_slot("c")
    assert copy is not None
    store.copy_block(*copy)
    c_slots = bm.slot_ids("c")[:6]
    assert c_slots != slots                               # the tail block moved...
    assert torch.equal(store.k[:, c_slots], store.k[:, slots])   # ...with its data
```

(`test_prefix_cache_same_output` needs Lesson 6; run the other two now.)

> [!WARNING] The gather is the slow part — on purpose
> Our gather *copies* the whole sequence's K/V into a temporary tensor every layer, every step — the extra memory traffic we were trying to avoid. Real engines never materialize it: vLLM's `paged_attention` CUDA kernels, FlashAttention's `flash_attn_with_kvcache(..., block_table=...)` and FlashInfer read K/V **directly** from the scattered blocks inside the attention kernel. Semantically it's the same gather; physically it happens in SRAM. Our version is the reference implementation you'd test such a kernel against.

> [!NOTE] Batching many sequences
> A single forward over several sequences needs one flat token batch plus per-sequence block tables and lengths (vLLM's "attention metadata"). Doing that with varlen attention is Module 09's job; for now one sequence per forward is enough to prove the memory manager.

- [ ] `test_paged_matches_contiguous` and `test_fork_copy_on_write_data` pass
- [ ] Measured tok/s of paged vs contiguous generation and explained the gap
- [ ] Can explain what `block_table` means as a FlashAttention argument
      */}),
      resources: [
        { title: "vLLM — Paged Attention kernel design", url: "https://docs.vllm.ai/en/latest/design/paged_attention.html", type: "docs", note: "how the CUDA kernel walks the block table" },
      ],
    },
    {
      id: "prefix-caching",
      title: "Build: prefix caching — hash-chained blocks vs radix trees",
      kind: "build",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
Prefix caching reuses the KV of a shared prompt prefix **across requests** (Baseten ch. 5.3). The rules follow from causal attention: a token's K/V depends on **every token before it**, so

- two prompts share cache only up to their **first differing token** ("Weather in Paris?" vs "Weather in London?" share "Weather in" — the trailing "?" does *not* count);
- if the first token differs, nothing is reusable, even if the rest is identical;
- so put stable content (system prompt, tools, documents) **first** and novel content **last** (context engineering).

::viz prefix-cache

**Two designs.**

| | **vLLM: hash-chained blocks** | **SGLang: RadixAttention** |
|---|---|---|
| Structure | hash table: `hash(parent_hash, block_tokens) → block` | radix tree over token sequences; each edge holds KV for a token run |
| Granularity | full blocks only (16 tokens) | any token boundary (page size can be 1) |
| Lookup | hash block 0, 1, 2 … until a miss | walk down the tree matching tokens |
| Eviction | LRU over free blocks (free queue order) | LRU over leaves with ref 0 |
| Strength | simple, O(1) per block, fits the block allocator perfectly | finest-grained sharing, natural for tree-shaped workloads (few-shot, agents, branching) |

The **chained hash** is the trick that makes the flat table equivalent to a tree: block $j$'s hash includes block $j-1$'s hash, so it identifies the *entire prefix* up to and including block $j$, not just 16 tokens. Two blocks with identical tokens but different histories get different hashes. vLLM adds "extra keys" to the hash (LoRA adapter id, multimodal input hashes, and an optional per-tenant `cache_salt`) and uses SHA-256 by default so a crafted prompt can't collide with another user's cache.

```python
# prefix_cache.py — hash-chained automatic prefix caching on top of BlockManager
import hashlib, pickle
from block_manager import BlockManager, OutOfBlocks


def block_hash(parent, tokens):
    """Hash of a FULL block = hash(parent block's hash, this block's tokens) -> identifies the whole prefix."""
    return hashlib.sha256(pickle.dumps((parent, tuple(tokens)))).digest()


class PrefixCachingBlockManager(BlockManager):
    def __init__(self, num_blocks, block_size=16):
        super().__init__(num_blocks, block_size)
        self.hash_to_block: dict[bytes, int] = {}
        self.block_to_hash: dict[int, bytes] = {}
        self.seq_tokens: dict[str, list[int]] = {}
        self.seq_hashes: dict[str, list[bytes]] = {}
        self.queried_tokens = self.hit_tokens = 0

    def _take(self):
        bid = super()._take()
        h = self.block_to_hash.pop(bid, None)     # recycling a cached-but-free block = evicting it
        if h is not None:
            del self.hash_to_block[h]
        return bid

    def _register(self, h, bid):
        if h not in self.hash_to_block:           # identical block already cached? keep the old one
            self.hash_to_block[h] = bid
            self.block_to_hash[bid] = h

    def _chain(self, tokens):
        bs, hashes, parent = self.block_size, [], None
        for j in range(len(tokens) // bs):
            parent = block_hash(parent, tokens[j * bs:(j + 1) * bs])
            hashes.append(parent)
        return hashes

    def allocate(self, seq_id, tokens):
        """Allocate blocks for a prompt; returns how many leading tokens are already cached."""
        assert seq_id not in self.tables
        bs, hashes, table = self.block_size, self._chain(tokens), []
        # 1) longest run of cache hits (never the WHOLE prompt: we must compute >= 1 token for logits)
        for h in hashes[: (len(tokens) - 1) // bs]:
            bid = self.hash_to_block.get(h)
            if bid is None:
                break
            if self.blocks[bid].ref_count == 0:
                del self.free[bid]                # was evictable; now in use again
            self.blocks[bid].ref_count += 1
            table.append(bid)
        n_cached = len(table) * bs
        # 2) fresh blocks for everything else
        need = self.blocks_needed(len(tokens)) - len(table)
        if need > self.num_free():
            for bid in reversed(table):
                self._release(bid)
            raise OutOfBlocks()
        table += [self._take() for _ in range(need)]
        # 3) the prefill we are about to run fills these full blocks -> make them shareable
        for j in range(len(table) - need, len(hashes)):
            self._register(hashes[j], table[j])
        self.tables[seq_id], self.lengths[seq_id] = table, len(tokens)
        self.seq_tokens[seq_id], self.seq_hashes[seq_id] = list(tokens), hashes
        self.queried_tokens += len(tokens)
        self.hit_tokens += n_cached
        return n_cached

    def append_token(self, seq_id, token_id):
        slot, copy = self.append_slot(seq_id)
        toks, bs = self.seq_tokens[seq_id], self.block_size
        toks.append(token_id)
        if len(toks) % bs == 0:                   # a block just filled up -> cache it too (multi-turn reuse)
            hs = self.seq_hashes[seq_id]
            hs.append(block_hash(hs[-1] if hs else None, toks[-bs:]))
            self._register(hs[-1], self.tables[seq_id][-1])
        return slot, copy

    def fork(self, parent, child):
        super().fork(parent, child)
        self.seq_tokens[child] = list(self.seq_tokens[parent])
        self.seq_hashes[child] = list(self.seq_hashes[parent])

    def free_seq(self, seq_id):
        super().free_seq(seq_id)                  # blocks go to the free list but KEEP their hash
        del self.seq_tokens[seq_id], self.seq_hashes[seq_id]

    def hit_rate(self):
        return self.hit_tokens / max(1, self.queried_tokens)
```

Four subtle points, each one a real bug engines have had:
1. **Never a 100% hit.** Even if the whole prompt is cached, we must run the model on at least the last token to get logits. Hence `hashes[:(len - 1) // bs]`.
2. **Resurrection.** A cache hit on a block with `ref_count == 0` must remove it from the free list, or it'll be handed out to someone else while in use.
3. **Eviction = recycling.** Taking a cached-but-free block from the free list must delete its hash entry, or future lookups would return overwritten data.
4. **Generated tokens are cacheable.** `append_token` registers blocks as they fill, so turn 2 of a conversation hits on turn 1's prompt *and answer*.

```python
# test_prefix_cache.py
from prefix_cache import PrefixCachingBlockManager


def test_prefix_hit():
    bm = PrefixCachingBlockManager(16, 4)
    sys_prompt = list(range(100, 112))                     # 12 tokens = 3 full blocks
    assert bm.allocate("a", sys_prompt + [1, 2]) == 0      # cold cache
    assert bm.allocate("b", sys_prompt + [7, 8, 9]) == 12  # 3 blocks reused
    assert bm.tables["a"][:3] == bm.tables["b"][:3]
    assert all(bm.blocks[b].ref_count == 2 for b in bm.tables["a"][:3])


def test_hit_after_free_then_eviction():
    bm = PrefixCachingBlockManager(4, 4)
    p = list(range(8)) + [99]                              # 2 full blocks + 1 partial
    bm.allocate("a", p); bm.free_seq("a")
    assert bm.allocate("b", p) == 8                        # freed, but still cached
    bm.free_seq("b")
    bm.allocate("c", list(range(50, 66))); bm.free_seq("c")  # needs all 4 blocks -> evicts
    assert bm.allocate("d", p) == 0


def test_generated_tokens_become_cacheable():
    bm = PrefixCachingBlockManager(16, 4)
    bm.allocate("t1", [1, 2, 3])
    for tok in [4, 5, 6, 7, 8]:
        bm.append_token("t1", tok)                         # sequence is now 1..8 = 2 full blocks
    bm.free_seq("t1")
    assert bm.allocate("t2", list(range(1, 11))) == 8      # next turn reuses the generated tokens


def test_first_token_differs_no_hit():
    bm = PrefixCachingBlockManager(16, 4)
    bm.allocate("a", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert bm.allocate("b", [0, 2, 3, 4, 5, 6, 7, 8, 9]) == 0   # hash chain breaks at block 0
```

Now run `test_prefix_cache_same_output` from Lesson 5: the second request prefills only 12 of its 44 tokens and still generates **identical** tokens to a cold, contiguous run. That equality is the entire correctness argument for prefix caching.

> [!REAL] Beyond one GPU
> A cache hit only helps if the request lands on the replica that has the blocks. Production systems add **cache-aware routing** (send the same conversation/user/codebase to the same replica; SGLang router, NVIDIA Dynamo's KV-aware router, llm-d) or a **shared KV store** below GPU memory (Lesson 8). Baseten ch. 5.3 covers both.

- [ ] `prefix_cache.py` written; the 4 prefix tests + `test_prefix_cache_same_output` pass
- [ ] Can explain why the hash must be chained, with a two-block counterexample
- [ ] Sketched (on paper) the radix tree SGLang would build for three prompts sharing a system prompt
      */}),
      resources: [
        { title: "vLLM — Automatic Prefix Caching (design)", url: "https://docs.vllm.ai/en/latest/design/prefix_caching/", type: "docs", note: "block hashing, extra keys, eviction order" },
        { title: "SGLang paper (RadixAttention)", url: "https://arxiv.org/abs/2312.07104", type: "paper", note: "section 3: the radix-tree cache and cache-aware scheduling" },
        { title: "LMSYS blog — Fast and Expressive LLM Inference with RadixAttention and SGLang", url: "https://lmsys.org/blog/2024-01-17-sglang/", type: "article", note: "animated radix-tree examples" },
      ],
    },
    {
      id: "memory-budget",
      title: "Math: memory budgets — how many concurrent sequences fit?",
      kind: "math",
      minutes: 60,
      md: MD(function () {/*
> [!MATH] Refresher: the budget equation
> $$\text{KV budget} = M_{GPU} \times u - W - A$$
> $M_{GPU}$ = GPU memory, $u$ = the fraction the engine may use (`gpu_memory_utilization`, vLLM default 0.92; TRT-LLM's `free_gpu_memory_fraction` is the analogous knob), $W$ = weight bytes, $A$ = activations, CUDA graphs and workspace (1–5 GB typical).
> $$\text{max tokens} = \frac{\text{KV budget}}{2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b_{kv}} \qquad \text{max sequences} \approx \frac{\text{max tokens}}{\text{average (prompt + output) length}}$$
> With paging, "average length" means *actual* length; with naive contiguous allocation it means `max_model_len`.

::viz kv-calculator

**Worked example 1 — Llama-3-8B, BF16, one H100 80 GB.** KV per token $= 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B (128 KiB).
Budget $\approx 80 \times 0.9 - 16.06 - 3 \approx 53$ GB → $53 \times 10^9 / 131{,}072 \approx 404$K tokens.
- 8K-token conversations: ~**49** concurrent. 2K-token chats: ~**197**.
- Naive allocation with `max_model_len = 8192` for 2K-token chats: still only 49 — **4× fewer**. That's the paging win in one line.

**Worked example 2 — Llama-3-70B, FP8 weights, 2× H100 (TP = 2).** Weights ≈ 70 GB across both GPUs. Budget $\approx 160 \times 0.9 - 70 - 6 = 68$ GB. KV per token (BF16 KV) $= 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ B (320 KiB; tensor parallelism splits heads across GPUs but the total is the same).
→ ~**207K** tokens with BF16 KV; ~**415K** with FP8 KV (`--kv-cache-dtype fp8`). A single 128K-token request needs ~40 GB of KV in BF16 (Baseten ch. 5.3) — more than half the budget.

**Worked example 3 — Baseten's B200 (book, ch. 5.3).** 180 GB usable, 100 GB taken by weights and buffers, engine allowed 0.8 of the remainder: $0.8 \times 80 = 64$ GB of KV cache.

**Worked example 4 — your Colab T4 with Qwen2.5-0.5B.** 12,288 B/token, ~11 GB budget → ~0.9M tokens: roughly **110** full 8K-context requests. A tiny model on a small GPU still has room for a big batch because of GQA (only 2 KV heads).

> [!IMPORTANT] Things the simple formula ignores
> - **Fragmentation**: ~half a block per sequence (8 tokens × 128 KiB ≈ 1 MiB for Llama-3-8B). Negligible with paging, huge without.
> - **Prefix sharing** makes the effective number of tokens *larger* than the physical budget (your simulation will show ~2 logical tokens per physical slot for system-prompt-heavy traffic).
> - **Preemption headroom**: running at 100% of KV means frequent preemptions; engines keep a watermark and schedulers throttle admissions.
> - **Throughput ≠ capacity**: fitting 200 sequences doesn't mean you should run 200 — Module 07's latency–throughput curve and your SLO decide.

**Practice.**
1. Qwen2.5-7B (28 layers, 4 KV heads, head dim 128) in BF16 on an L4 (24 GB). How many 4K-token requests fit? *(weights ≈ 15.2 GB; budget ≈ 24 × 0.9 − 15.2 − 2 ≈ 4.4 GB; 57,344 B/token → ~77K tokens → ~18 requests)*
2. Same model on an H100: how many? And by what factor does FP8 KV change it?
3. Why does Baseten's rule "weights + at least 50% headroom" (ch. 3.1) exist, in terms of this formula?

- [ ] Reproduced worked example 1 and the naive-vs-paged ratio
- [ ] Solved the three practice problems
- [ ] Checked one of your answers against a real `vllm serve` log line
      */}),
      resources: [
        { title: "Inference Engineering (Baseten) — ch. 5.3", url: "Inference%20Engineering.pdf", type: "book", note: "prefix caching rules, KV storage tiers, the B200 budget example" },
      ],
    },
    {
      id: "kv-compression",
      title: "Shrinking the KV cache: GQA/MLA, FP8 KV, sliding windows, offloading",
      kind: "concept",
      minutes: 75,
      md: MD(function () {/*
Paging removes waste; these techniques shrink what's left. Every byte saved per token is more batch (throughput) or more context.

**1. Fewer KV heads — architecture (trained in).**

::viz gqa

| Scheme | KV heads | KV per token (32 layers, $d_{head}$ = 128, BF16) | Example |
|---|---|---|---|
| MHA | 32 | 512 KiB | Llama-2-7B |
| GQA (group 4) | 8 | 128 KiB | Llama-3-8B, Qwen2.5 |
| MQA | 1 | 16 KiB | PaLM, Falcon-7B |
| **MLA** | latent, not heads | ~69 KiB for a 671B model | DeepSeek-V2/V3 |

**MLA** (Multi-head Latent Attention, DeepSeek-V2) caches one compressed latent vector per token per layer (512 dims + 64 for RoPE in DeepSeek-V3: $576 \times 61 \text{ layers} \times 2$ B ≈ 70 KB/token) and up-projects it into per-head K/V inside attention. A standard-MHA model of that size would need several MB per token. GQA is covered in depth in Module 04; you can't retrofit it without retraining (though up-training from MHA checkpoints works — see the GQA paper).

**2. Quantized KV — serving-time.** Store K/V in FP8 instead of BF16: half the bytes, roughly half the attention read time in decode. In vLLM: `--kv-cache-dtype fp8` (E4M3 or E5M2; scales can be calibrated). Quality loss is usually small for FP8; research goes much further — **KIVI** quantizes keys per-channel and values per-token to 2 bits. Module 16 covers the numerics.

**3. Sliding-window / local attention — architecture.** Some layers only attend to the last $w$ tokens, so their cache is capped at $w$ entries no matter how long the conversation. Mistral 7B used $w = 4096$ everywhere; **Gemma 3** interleaves 5 local layers (window 1,024) per 1 global layer; **gpt-oss** alternates full and banded-window layers. Baseten ch. 2.5 lists windows of 8K–32K as typical. Engines must then manage *different* cache lifetimes per layer — vLLM's hybrid KV-cache manager frees sliding-window blocks as they fall out of the window.

**4. Offloading — tiers below HBM.** When the GPU is full, keep blocks in slower but bigger memory instead of evicting them (Baseten ch. 5.3):

| Tier | Medium | Speed | Size |
|---|---|---|---|
| G1 | GPU HBM | TB/s | 10s–100s GB |
| G2 | CPU RAM | 10s–100s GB/s | 100s GB – TBs |
| G3 | Local SSD | 5–10 GB/s | TBs |
| G4 | Networked storage | GB/s | 10s of TBs |

Reloading a block from CPU is worth it whenever it's faster than recomputing it with prefill — true for long prefixes, especially on Grace-Hopper/Grace-Blackwell systems where the CPU–GPU link runs at 900 GB/s. Tools: **LMCache** (plugs into vLLM/SGLang via a KV connector; CPU, disk, remote backends), **NVIDIA Dynamo KVBM** (KV Block Manager across G1–G4), **Mooncake** (Moonshot AI's KV-centric disaggregated architecture for Kimi). A G4 shared cache also survives autoscaling: a brand-new replica can start warm.

> [!INTUITION] Pick by where the bytes go
> Long shared prompts, many users → prefix caching + offload. Long *unique* contexts → FP8 KV, sliding windows, MLA-style models. High concurrency short chats → paging + GQA is usually enough.

- [ ] Can compute KV per token for MHA, GQA, MQA and MLA versions of the same model
- [ ] Can explain when offloading beats recomputation (bytes ÷ link bandwidth vs prefill FLOPs ÷ peak)
- [ ] Knows the vLLM flag for FP8 KV and one research method below 8 bits
      */}),
      resources: [
        { title: "GQA paper (Ainslie et al., 2023)", url: "https://arxiv.org/abs/2305.13245", type: "paper", note: "grouped-query attention and uptraining from MHA" },
        { title: "DeepSeek-V2 (MLA)", url: "https://arxiv.org/abs/2405.04434", type: "paper", note: "multi-head latent attention" },
        { title: "KIVI — 2-bit KV cache quantization", url: "https://arxiv.org/abs/2402.02750", type: "paper", note: "per-channel keys, per-token values" },
        { title: "vLLM — Quantized KV cache", url: "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html", type: "docs", note: "`--kv-cache-dtype fp8`" },
        { title: "LMCache", url: "https://github.com/LMCache/LMCache", type: "repo", note: "KV offload and sharing for vLLM/SGLang" },
        { title: "Mooncake", url: "https://arxiv.org/abs/2407.00079", type: "paper", note: "KV-cache-centric disaggregated serving" },
      ],
    },
    {
      id: "deep-vllm-source",
      title: "Deep dive: read vLLM's KV cache manager and Aleksa Gordić's walkthrough",
      kind: "deep",
      optional: true,
      minutes: 120,
      md: MD(function () {/*
You've built the toy; now map it onto the real thing.

**1. Read Aleksa Gordić — "Inside vLLM".** Two parts matter here:
- **"LLM Engine constructor"** — how the engine profiles memory, sizes the KV cache, and builds the `KVCacheManager`, the free block queue and the block pool.
- **"Advanced Features → Prefix Caching"** — the hash-chained blocks, how `get_computed_blocks` finds hits, and how blocks are cached once full.
Skim "Scheduler" too — it's where `allocate_slots` is called and where preemption happens (Module 09).

**2. Read the source** (`vllm/v1/core/` on GitHub), with your own code open next to it:

| vLLM | Your code |
|---|---|
| `kv_cache_utils.py`: `KVCacheBlock` (`block_id`, `ref_cnt`, `block_hash`), `FreeKVCacheBlockQueue` (`popleft`, `remove`, `append`), `hash_block_tokens` | `Block`, the `OrderedDict` free list, `block_hash` |
| `block_pool.py`: `BlockPool.get_new_blocks`, `free_blocks`, `touch`, `cache_full_blocks`, `get_cached_block`, `_maybe_evict_cached_block`, `get_usage` | `_take` (+ eviction), `_release`, resurrection in `allocate`, `_register` |
| `kv_cache_manager.py`: `KVCacheManager.get_computed_blocks`, `allocate_slots`, `free`, `cache_blocks` | `allocate` (hit lookup), `append_slot`/`append_token`, `free_seq` |
| `single_type_kv_cache_manager.py`, `kv_cache_coordinator.py` | (none — these handle full-attention vs sliding-window layer groups) |

Questions to answer while reading:
1. What does `touch` do, and which line of your `allocate` is its equivalent?
2. Where does vLLM decide the number of GPU blocks, and which config fields feed it (`gpu_memory_utilization`, `block_size`, `num_gpu_blocks_override`)?
3. How does vLLM avoid ever serving a *whole* prompt from cache (your point 1 in Lesson 6)?
4. What extra keys go into a block hash besides token ids?

**3. Compare with nano-vllm** (`nanovllm/engine/block_manager.py`): a ~100-line block manager with the same hashing idea, from a ~1,200-line engine that reaches vLLM-class offline throughput. It's the closest thing to your Module 10 engine.

- [ ] Answered the four questions with file + function references
- [ ] Wrote a one-page diff: "what vLLM does that my BlockManager doesn't"
      */}),
      resources: [
        { title: "Aleksa Gordić — Inside vLLM: Anatomy of a High-Throughput LLM Inference System", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "sections: LLM Engine constructor (KV cache manager) and Advanced Features → Prefix Caching" },
        { title: "vLLM source — kv_cache_manager.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py", type: "repo", note: "get_computed_blocks, allocate_slots, free" },
        { title: "vLLM source — kv_cache_utils.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py", type: "repo", note: "KVCacheBlock, FreeKVCacheBlockQueue, hash_block_tokens" },
        { title: "SGLang source — radix_cache.py", url: "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py", type: "repo", note: "the radix-tree alternative, in code" },
      ],
    },
  ],

  challenge: {
    title: "BlockManager + prefix cache with tests, and a workload simulation vs naive allocation",
    md: MD(function () {/*
In `course-work/m08/`:

1. `block_manager.py`, `prefix_cache.py`, `paged.py` with the test suites from Lessons 4–6, plus **your own** tests for invariant 4 and for fork + prefix caching together.
2. `simulate.py`: replay a synthetic trace and compare paged + prefix caching against naive contiguous reservation. A working starting point (FCFS admission, one decode step per request per iteration, recompute preemption of the newest request):

```python
# simulate.py — replay a synthetic trace: naive contiguous reservation vs paged + prefix caching
import random
from block_manager import OutOfBlocks
from prefix_cache import PrefixCachingBlockManager

def make_trace(n=400, seed=0, vocab=32000):
    rng = random.Random(seed)
    systems = [[rng.randrange(vocab) for _ in range(rng.choice([256, 512, 1024]))] for _ in range(4)]
    return [dict(id=f"r{i}", prompt=rng.choice(systems) + [rng.randrange(vocab) for _ in range(rng.randint(20, 400))],
                 out=rng.randint(20, 600), max_tokens=1024) for i in range(n)]

def run(trace, num_blocks=4096, block_size=16, max_live=256, seed=0):
    rng = random.Random(seed)
    bm = PrefixCachingBlockManager(num_blocks, block_size)
    live, queue, steps, preemptions = {}, list(trace), [], 0
    while queue or live:
        while queue and len(live) < max_live:             # admit FCFS while blocks last
            try:
                bm.allocate(queue[0]["id"], queue[0]["prompt"])
            except OutOfBlocks:
                break
            r = queue.pop(0); live[r["id"]] = dict(r, done=0, gen=[])
        for sid in list(live):                             # one decode step for every running request
            if sid not in live:
                continue
            tok = rng.randrange(32000)
            while True:
                try:
                    bm.append_token(sid, tok); break
                except OutOfBlocks:                        # preempt the newest request (recompute later)
                    victim = list(live)[-1]
                    bm.free_seq(victim); preemptions += 1
                    v = live.pop(victim)
                    queue.insert(0, dict(v, prompt=v["prompt"] + v["gen"], out=v["out"] - v["done"]))
                    if victim == sid:
                        break
            if sid not in live:
                continue
            r = live[sid]; r["done"] += 1; r["gen"].append(tok)
            if r["done"] == r["out"]:
                bm.free_seq(sid); del live[sid]
        tokens = sum(len(r["prompt"]) + r["done"] for r in live.values())
        slots = (num_blocks - bm.num_free()) * block_size  # physical slots held by running requests
        steps.append((len(live), tokens / max(1, slots)))
    avg_prompt = sum(len(r["prompt"]) for r in trace) / len(trace)
    return dict(hit_rate=round(bm.hit_rate(), 3), preemptions=preemptions, decode_steps=len(steps),
                paged_mean_concurrency=round(sum(s[0] for s in steps) / len(steps), 1),
                paged_tokens_per_slot=round(sum(s[1] for s in steps) / len(steps), 2),
                naive_max_concurrency=int(num_blocks * block_size // (avg_prompt + trace[0]["max_tokens"])))

if __name__ == "__main__":
    print(run(make_trace()))
```

3. Extend it into a real experiment and write `REPORT.md`:
   - Implement the **naive allocator as a real simulation** too (reserve `prompt + max_tokens` contiguously per request from a fixed pool, with first-fit holes so external fragmentation shows up), and compare concurrency, total decode steps (≈ time) and memory utilization over time.
   - Sweep **block size** (8, 16, 32, 64) and report hit rate, internal waste and preemptions.
   - Sweep the **workload**: no shared prefix, 4 shared system prompts, multi-turn chats (turn $k+1$'s prompt = turn $k$'s prompt + answer + new message).
   - Note the known flaw in the starter: re-admitted preempted requests inflate the hit rate. Fix the accounting.
   - Plot memory utilization (tokens ÷ allocated slots) over time for naive vs paged.
    */}),
    checklist: [
      "BlockManager, PrefixCachingBlockManager and paged attention pass all provided tests plus ≥ 2 of your own",
      "Paged generation produces identical tokens to contiguous generation (with and without prefix hits)",
      "Naive contiguous allocator simulated on the same trace and memory pool",
      "Report table: hit rate, mean concurrency, memory utilization, preemptions — naive vs paged vs paged+prefix",
      "Block-size and workload sweeps with a short explanation of each trend",
      "Utilization-over-time plot for naive vs paged",
    ],
    stretch: "Implement a tiny RadixAttention-style cache (a trie keyed by tokens, LRU eviction of unreferenced leaves) behind the same `allocate/append_token/free_seq` interface and compare its hit rate with hash-chained blocks on the multi-turn and few-shot workloads — where does token-granularity sharing beat 16-token blocks?",
  },

  connects: MD(function () {/*
Module 06 gave each request a private contiguous cache; this module made the cache a **shared, paged, content-addressed** resource — the data structure the rest of the engine revolves around. **Module 09** builds the scheduler that calls your `allocate`/`append_slot` every step, handles `OutOfBlocks` with preemption, and batches many sequences' block tables into one forward pass (continuous batching + chunked prefill). **Module 10** assembles block manager, scheduler, model runner and your SSE server into your own mini-vLLM. **Module 16** revisits KV quantization, and **Module 17** moves KV blocks *between* machines for disaggregated prefill/decode and cache-aware routing.
  */}),

  interview: [
    "Walk me through PagedAttention. What are the three kinds of memory waste it removes, and what's the remaining overhead?",
    "Design a KV cache block manager: data structures, allocation per decode step, freeing, and what happens when you run out of blocks.",
    "Swap vs recompute preemption — trade-offs, and which would you choose on an H100 with a 900 GB/s CPU link vs PCIe Gen4?",
    "How does automatic prefix caching work in vLLM? Why must the block hash be chained, and why is a full-prompt hit impossible?",
    "Compare vLLM's hash-based prefix cache with SGLang's RadixAttention. Which workloads favor each?",
    "How many concurrent 8K-token requests fit for Llama-3-8B BF16 on one H100? What changes with FP8 KV, and with naive allocation?",
    "Your prefix-cache hit rate is 5% in production although every request shares a long system prompt. List possible causes.",
    "Explain MLA vs GQA in terms of KV bytes per token and what each costs at training/inference time.",
  ],

  resources: [
    { title: "Kwon et al. — PagedAttention (SOSP 2023)", url: "https://arxiv.org/abs/2309.06180", type: "paper", note: "the paper this module implements" },
    { title: "Aleksa Gordić — Inside vLLM", url: "https://www.aleksagordic.com/blog/vllm", type: "article", note: "KV cache manager and prefix caching in the real engine" },
    { title: "vLLM — Automatic Prefix Caching (design)", url: "https://docs.vllm.ai/en/latest/design/prefix_caching/", type: "docs", note: "hash chain, extra keys, eviction" },
    { title: "vLLM source — block_pool.py", url: "https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py", type: "repo", note: "the production version of your BlockManager" },
    { title: "SGLang — RadixAttention paper", url: "https://arxiv.org/abs/2312.07104", type: "paper", note: "radix-tree prefix cache + cache-aware scheduling" },
    { title: "nano-vllm", url: "https://github.com/GeeeekExplorer/nano-vllm", type: "repo", note: "~1,200-line vLLM re-implementation; read engine/block_manager.py" },
    { title: "Inference Engineering (Baseten) — ch. 2.5 & 5.3", url: "Inference%20Engineering.pdf", type: "book", note: "PagedAttention, prefix caching rules, KV tiers G1–G4, cache-aware routing" },
    { title: "LMCache", url: "https://github.com/LMCache/LMCache", type: "repo", note: "KV offloading and cross-engine sharing" },
    { title: "NVIDIA Dynamo docs", url: "https://docs.nvidia.com/dynamo/latest/index.html", type: "docs", note: "KV Block Manager (KVBM) and KV-aware routing" },
    { title: "Mooncake", url: "https://arxiv.org/abs/2407.00079", type: "paper", note: "KV-cache-centric architecture at Moonshot AI" },
  ],
});
