---
layout: post
title: "KV Cache Compaction and Compression: From Attention Sinks to Learned Memory"
date: 2026-06-01
author: "Shreyansh Singh"
description: "A code-first guide to KV cache compression: why the cache dominates long-context serving, how token-eviction methods work, and how Cartridges and STILL turn compact KV tensors into reusable memory."
thumbnail: /assets/img/posts_images/kv_cache_compaction/kv-cache-taxonomy.svg
tags: llms transformers kv-cache compression long-context mlsys
categories: ["LLMs", "MLSys"]
giscus_comments: true
related_posts: false
permalink: "post/2026-06-01_kv-cache-compaction-compression/"
featured: false
toc:
  sidebar: left
pretty_table: true
_styles: |
  .kv-post {
    --kv-ink: #181510;
    --kv-muted: #554c3f;
    --kv-paper: #f7f2e9;
    --kv-panel: #fffaf0;
    --kv-line: #d0bea0;
    --kv-teal: #2f6d72;
    --kv-rust: #9a5932;
    --kv-green: #496f3e;
    --kv-plum: #79506d;
    color: var(--kv-ink);
  }
  html[data-theme="dark"] .kv-post {
    --kv-ink: #f3eadc;
    --kv-muted: #c9bea9;
    --kv-paper: #1d1a16;
    --kv-panel: #25211b;
    --kv-line: #5c5142;
    --kv-teal: #75c1c8;
    --kv-rust: #f0a46e;
    --kv-green: #9bc982;
    --kv-plum: #d59bc5;
    color: var(--kv-ink);
  }
  .kv-post mjx-container {
    color: var(--kv-ink) !important;
  }
  .kv-post .kv-hero,
  .kv-post .kv-callout,
  .kv-post .kv-table-note {
    border: 1px solid var(--kv-line);
    border-radius: 0.5rem;
    background: var(--kv-panel);
  }
  .kv-post .kv-hero {
    margin: 0 0 2rem;
    padding: 1rem;
    background:
      linear-gradient(90deg, rgba(24, 21, 16, 0.035) 1px, transparent 1px),
      linear-gradient(180deg, rgba(24, 21, 16, 0.035) 1px, transparent 1px),
      var(--kv-paper);
    background-size: 3rem 3rem;
  }
  .kv-post .kv-hero,
  .kv-post .kv-figure {
    max-width: 100%;
    margin-left: auto;
    margin-right: auto;
  }
  .kv-post .kv-hero img,
  .kv-post .kv-figure img {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    margin-left: auto;
    margin-right: auto;
  }
  .kv-post .kv-caption {
    max-width: 58rem;
    margin: 0.65rem auto 1.5rem;
    color: var(--kv-muted);
    font-size: 0.92rem;
    line-height: 1.45;
    text-align: center;
  }
  .kv-post .kv-callout {
    margin: 1.25rem 0;
    padding: 1rem;
  }
  .kv-post .kv-callout strong {
    color: var(--kv-rust);
  }
  .kv-post .kv-table-note {
    margin: 1rem 0;
    padding: 0.85rem 1rem;
    color: var(--kv-muted);
    font-size: 0.95rem;
  }
  .kv-post .kv-mini-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.85rem;
    margin: 1.25rem 0;
  }
  .kv-post .kv-mini-card {
    border: 1px solid var(--kv-line);
    border-radius: 0.5rem;
    background: var(--kv-panel);
    padding: 0.9rem;
  }
  .kv-post .kv-mini-card b {
    display: block;
    margin-bottom: 0.35rem;
    color: var(--kv-teal);
  }
  @media (min-width: 760px) {
    .kv-post .kv-mini-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
---

<div class="kv-post" markdown="1">

<div class="kv-hero">
  <img src="/assets/img/posts_images/kv_cache_compaction/kv-cache-taxonomy.svg" alt="A taxonomy of KV cache compression methods." />
  <p class="kv-caption">A useful map for KV cache work: are we selecting existing token states, or synthesizing new compact states? Is the cost paid per query, per corpus, or once as reusable training?</p>
</div>

**Code:** [cartridges](https://github.com/shreyansh26/cartridges), [STILL-Towards-Infinite-Context-Windows](https://github.com/shreyansh26/STILL-Towards-Infinite-Context-Windows), [kv-cache-compression](https://github.com/shreyansh26/kv-cache-compression)

**Primary sources:** [Cartridges](https://arxiv.org/abs/2506.06266), [STILL / neural KV-cache compaction](https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/), [StreamingLLM / Attention Sinks](https://arxiv.org/abs/2309.17453), [H2O](https://arxiv.org/abs/2306.14048), [SnapKV](https://arxiv.org/abs/2404.14469), [L2 norm KV compression](https://arxiv.org/abs/2406.11430)

Long context is not only a modeling problem. It is also a memory residency problem.

During autoregressive decoding, the model does not recompute all previous tokens at every step. It stores their attention keys and values in the KV cache, then each new token attends to that cache. This is exactly what makes decoding practical. It is also what makes long prompts expensive to keep alive.

The basic tension is:

- the full KV cache is faithful, reusable, and expensive;
- a textual summary is cheap, but it has already passed through a narrow language bottleneck;
- retrieval is sparse and query-dependent;
- KV cache compression tries to keep the representation in the model's own internal coordinate system.

This post is a code-first tour through that design space. I will start with the cache accounting, then walk through four token-selection style implementations: Attention Sink, L2 norm pruning, SnapKV, and H2O. After that, we will spend most of the time on the two more interesting compaction methods I implemented in detail: **Cartridges** and **STILL**.

The important distinction is that **compression** often means "keep fewer existing KV entries", while **compaction** means "make a smaller KV object that is not just a subset of the original tokens." Cartridges and STILL are compaction methods in that stronger sense.

## Setup: The Memory Bill

Before comparing compression policies, it helps to make the cache cost explicit and name the questions each method has to answer.

### The KV Cache Baseline

In a decoder-only transformer, the query for the current token attends to keys and values from previous tokens. For one layer and one head, ignoring masks for a moment:

$$
\mathrm{Attn}(q_t, K_{\le t}, V_{\le t})
= \sum_{i \le t} \alpha_{t,i} v_i
$$

where

$$
\alpha_{t,i}
= \frac{\exp(q_t^\top k_i / \sqrt{d})}
{\sum_{j \le t}\exp(q_t^\top k_j / \sqrt{d})}.
$$

The cache stores the $k_i$ and $v_i$ tensors so the model can append one new position at a time. With grouped-query attention, the number of query heads can be larger than the number of KV heads, but the storage formula is still simple:

$$
\mathrm{KVBytes}(T)
= T \cdot L \cdot H_{kv} \cdot d_{head} \cdot 2 \cdot b.
$$

Here $T$ is the number of cached tokens, $L$ is the number of layers, $H_{kv}$ is the number of KV heads, $d_{head}$ is the head dimension, the factor $2$ is for keys plus values, and $b$ is bytes per scalar.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/kv-cache-memory.svg" alt="KV cache memory formula and linear growth." />
  <p class="kv-caption">The figure separates the growing $T$ term from the fixed per-token footprint. This post is mostly about replacing $T$ with a smaller cache budget $p$.</p>
</div>

For a Llama-3.1-8B style configuration with $L=32$, $H_{kv}=8$, $d_{head}=128$, and bf16 storage, each token costs:

$$
32 \cdot 8 \cdot 128 \cdot 2 \cdot 2 = 131{,}072 \text{ bytes}
$$

which is 128 KiB per token per request. A 128K-token prompt is therefore about 16 GiB of KV cache before allocator overheads, paging metadata, batching effects, and attention workspace. This is why "the model supports 128K" and "I can cheaply keep hundreds of 128K sessions resident" are very different statements.

The baseline cache update in the [`kv-cache-compression`](https://github.com/shreyansh26/kv-cache-compression) repo is exactly what you expect:

```python
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
```

Every compression method below replaces this `update` policy, or replaces the whole cache object that the model consumes.

### Four Questions For Any KV Compression Method

The implementation details vary, but I find the following questions more useful than method names:

1. **What is retained?** Original token positions, synthesized token positions, quantized vectors, or a mixture?
2. **When is the decision made?** During prefill, during every decode step, offline before serving, or through reusable training?
3. **What signal decides importance?** Recency, attention mass, key norms, distillation loss, or a learned encoder?
4. **What does the model see afterward?** A shorter prefix, a sliding cache, a trainable past-key-value object, or compact K/V plus attention biases?

## Token Eviction: Keep A Subset Of The Original Cache

The small methods in [`kv-cache-compression`](https://github.com/shreyansh26/kv-cache-compression) are easy to understand because they keep the model architecture fixed and swap only the cache policy. Cartridges and STILL are more ambitious: they ask whether the cache itself can become a learned memory object.

### Attention Sink: Keep The First Tokens And The Tail

[StreamingLLM](https://arxiv.org/abs/2309.17453) starts from a surprising empirical observation: pure sliding-window attention can collapse even when the recent local window is present. Keeping a small number of initial tokens fixes much of the instability. Those initial tokens act as "attention sinks": places where many later tokens can put excess attention mass.

The policy is simple. Given a maximum cache budget

$$
B = G + W
$$

keep:

$$
S_t = \{0, \ldots, G-1\} \cup \{t-W+1, \ldots, t\}.
$$

The paper's key observation is not that the first few tokens are semantically important. It is that many models use the earliest positions as attention sinks: safe places to allocate attention mass when no specific old token is needed. Once a pure sliding window evicts those positions, the attention distribution shifts in a way the model was not trained for.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/attention-sink-technique.svg" alt="Attention Sink keeps initial sink tokens plus a recent tail window." />
  <p class="kv-caption"><a href="https://arxiv.org/abs/2309.17453">StreamingLLM</a>'s cache layout: the retained cache is not a contiguous suffix; it keeps initial sink tokens plus a tail window ending at the latest token.</p>
</div>

In code, prefill with a long prompt keeps the first `global_tokens` and the last `sliding_window` tokens:

```python
if total_len > self.max_cache_size:
    global_idxs = torch.arange(self.global_tokens, device=input_pos.device)
    recent_idxs = torch.arange(total_len - self.sliding_window, total_len, device=input_pos.device)
    keep_idxs = torch.cat([global_idxs, recent_idxs])
    new_pos = input_pos[keep_idxs]

    self.pos = new_pos.unsqueeze(0).unsqueeze(0).expand_as(self.pos)
    self.k_cache = k_val.index_select(dim=2, index=keep_idxs)
    self.v_cache = v_val.index_select(dim=2, index=keep_idxs)
```

During decoding, the implementation preserves the global region and evicts one position from the tail region:

```python
idx_to_pop = (self.global_tokens + torch.argmin(self.pos[:, :, self.global_tokens :], dim=-1)).flatten()
self.pos[:, :, idx_to_pop] = input_pos.long()
self.k_cache[:, :, idx_to_pop] = k_val
self.v_cache[:, :, idx_to_pop] = v_val
```

The important part is what this method does **not** try to do. It does not identify facts. It does not reconstruct old values. It does not learn a memory. It keeps the model numerically stable in streaming use by preserving a few special early tokens and a recent local window.

That makes it a good default when the application is genuinely streaming: chat, logs, or continuous text where old details are less important than keeping the model coherent.

### L2 Norm Compression: Score Keys Before Querying Them

The L2 norm strategy is interesting because it avoids attention statistics. The [paper](https://arxiv.org/abs/2406.11430) reports a correlation between key-vector norms and later attention behavior: low-norm keys tend to receive higher attention. That means we can score cached entries by the key tensor itself, before future queries arrive.

For a key vector $k_i$, define:

$$
s_i = -\lVert k_i \rVert_2.
$$

Then keep the tokens with the largest $s_i$, equivalently the lowest key norms.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/l2-norm-technique.svg" alt="L2 norm KV compression keeps low-norm key vectors." />
  <p class="kv-caption">Key logic from the <a href="https://arxiv.org/abs/2406.11430">L2 norm KV compression paper</a>: score cached keys by norm and prune without building an observation attention matrix.</p>
</div>

In the implementation, the score is written as `max_norm - norm`, then sorted:

```python
key_norm = torch.norm(self.k_cache, p=2, dim=-1)
key_norm_diff = key_norm.max() - key_norm
scoring_priority = key_norm_diff.masked_fill(self.pos == -1, float('inf'))
scoring_sorted_idx = torch.argsort(scoring_priority, dim=-1)
num_toks_to_remove = int((1 - self.keep_ratio) * self.max_cache_size)
scoring_sorted_idx_selcted = scoring_sorted_idx[:, :, num_toks_to_remove:]
```

Because `key_norm_diff` is small for large-norm keys and large for small-norm keys, removing the first `num_toks_to_remove` sorted positions drops large-norm keys and keeps lower-norm keys. The implementation then gathers the retained K/V rows and appends empty slots for future decode tokens.

```python
self.k_cache = torch.cat([
    torch.gather(
        self.k_cache,
        dim=2,
        index=scoring_sorted_idx_selcted.unsqueeze(-1).expand(-1, -1, -1, self.k_cache.shape[-1]),
    ),
    torch.zeros(
        self.k_cache.shape[0],
        self.k_cache.shape[1],
        num_toks_to_remove,
        self.k_cache.shape[-1],
        device=self.k_cache.device,
        dtype=self.k_cache.dtype,
    ),
], dim=2)
```

The useful property is that this is query-agnostic. The cache can be pruned without running an observation window or storing attention history. That also makes it compatible with attention kernels where the implementation does not expose full attention matrices.

The tradeoff is that the score is a proxy. It can work surprisingly well, but it is still betting that the norm structure learned by the model will line up with future importance.

### SnapKV: Use Prompt Attention To Select The Cache

[SnapKV](https://arxiv.org/abs/2404.14469) is more query-aware, but it tries to make the decision before generation. The idea is that the prompt's final tokens already reveal what the model will look for during decoding. So SnapKV observes attention from a small prompt window, scores old cache positions, keeps the top positions, and also keeps the most recent window.

The implementation path is:

1. Run prefill and keep the full prompt cache temporarily.
2. Compute attention scores from the prompt.
3. Aggregate query heads that share a KV head.
4. Sum attention from the last `window_size` query tokens to older prefix tokens.
5. Smooth the score with average pooling.
6. Keep the top `compress_length - window_size` older tokens plus the latest `window_size` tokens.

The "snap" in SnapKV is this one-time compression after prefill: the model observes the prompt, predicts which prefix positions will matter, and then generates against a much smaller cache.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/snapkv-technique.svg" alt="SnapKV uses an observation window to score prefix tokens and form a compact cache." />
  <p class="kv-caption"><a href="https://arxiv.org/abs/2404.14469">SnapKV</a>'s selection step: the prompt observation window scores clustered prefix positions, then selected prefix K/V is concatenated with the observation window.</p>
</div>

The core code is:

```python
attn_scores_grouped = attn_scores.view(b, n_kv_head, n_rep, lq, lk)
attn_scores_agg = attn_scores_grouped.sum(dim=2)
attn_weights_sum = attn_scores_agg[:, :, -self.window_size:, : total_len-self.window_size].sum(dim=-2)
attn_cache = self.pool(attn_weights_sum)

indices = attn_cache.topk(self.compress_length - self.window_size, dim=-1).indices
indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

k_past_compress = self.k_cache[:, :, :total_len-self.window_size, :].gather(dim=2, index=indices_expanded)
v_past_compress = self.v_cache[:, :, :total_len-self.window_size, :].gather(dim=2, index=indices_expanded)
k_cur = self.k_cache[:, :, total_len-self.window_size:total_len, :]
v_cur = self.v_cache[:, :, total_len-self.window_size:total_len, :]

key_states = torch.cat([k_past_compress, k_cur], dim=2)
value_states = torch.cat([v_past_compress, v_cur], dim=2)
```

The method is not "top attention globally." It is "top attention from the observation window, smoothed locally, with a recent tail always retained." That distinction matters because local neighborhoods can matter even when a single token's raw attention score is not maximal.

In `model_kv_cache_compression.py`, SnapKV has to use a manual attention path during prefill so the cache object can see the scores:

```python
y, scores = self.manual_attention(
    q,
    k,
    v,
    is_causal=(seqlen > 1),
    enable_gqa=(self.n_head != self.n_local_heads),
)

if self.snapkv_enabled and seqlen > 1:
     self.kv_cache.prune_cache(input_pos, scores, n_kv_head=self.n_local_heads)
```

That is the systems cost of attention-based pruning: the cache policy needs the attention matrix, and highly optimized attention kernels usually avoid materializing it.

### H2O: Keep Heavy Hitters Under A Fixed Budget

[H2O](https://arxiv.org/abs/2306.14048) treats KV eviction as a dynamic heavy-hitter problem. Some tokens repeatedly receive a large share of attention. Those tokens should remain in cache, while low-utility tokens can be evicted.

For token $i$, maintain an attention-history score:

$$
h_i(t) =
\frac{\sum_{\tau \le t} a_{\tau,i}}
{\max(1, n_i(t))}
$$

where $a_{\tau,i}$ is the attention mass assigned to token $i$ at decode step $\tau$, and $n_i(t)$ counts how many times the token was eligible or observed. When the cache is full, evict the smallest $h_i(t)$.

H2O also keeps recency in the picture. The useful mental model is not "attention history instead of a window"; it is "heavy hitters plus recent tokens under one fixed cache budget."

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/h2o-technique.svg" alt="H2O keeps heavy hitter tokens and recent tokens under a fixed cache budget." />
  <p class="kv-caption"><a href="https://arxiv.org/abs/2306.14048">H2O</a>'s eviction rule: accumulated attention identifies persistent heavy hitters, while recent tokens stay for local continuity.</p>
</div>

The implementation stores numerator and denominator buffers:

```python
self.register_buffer("attn_history_num", torch.zeros(history_num_shape, dtype=history_num_dtype))
self.register_buffer("attn_history_denom", torch.zeros(history_denom_shape, dtype=torch.int32))
self.register_buffer("attn_counter", torch.zeros((max_batch_size, n_heads), dtype=torch.int64))
```

The eviction score is the average attention history:

```python
numerator = self.attn_history_num.sum(dim=-1).float()

if (self.history_window_size == 1):
    denominator = self.attn_history_denom.clamp_min(1)
else:
    denominator = self.attn_history_denom.clamp(min=1, max=self.history_window_size)

avg_attn = numerator / denominator
scores = avg_attn.masked_fill(self.pos == -1, float('-inf'))
```

During decode, once the cache is full, the least important token is replaced:

```python
scores = self._calculate_eviction_scores()
eviction_idx = torch.argmin(scores, dim=-1)

eviction_idx_kv = eviction_idx.view(batch_size, n_heads, 1, 1).expand(-1, -1, 1, head_dim)
self.k_cache.scatter_(2, eviction_idx_kv, k_val)
self.v_cache.scatter_(2, eviction_idx_kv, v_val)
```

H2O is a good mental bridge between simple sliding windows and learned compaction. It says that recency alone is not enough; persistent attention mass is also important. But it still preserves original K/V entries. Once a token is gone, its vector is gone.

### What Token Eviction Can And Cannot Do

Token-selection methods are useful because they are local changes to an inference stack. They keep the model frozen, keep the attention interface mostly unchanged, and usually require no offline training.

But they have a hard ceiling:

$$
C = \{(k_i, v_i) : i \in S\}.
$$

The compressed cache is a subset of original token states. It cannot store a new vector that mixes multiple facts. It cannot put "the answer to three distant questions" into one slot unless one original token state already encoded that mixture. It also tends to be query-shaped: SnapKV depends on the observed prompt; H2O depends on previous decode attention; Attention Sink depends on streaming recency.

Cartridges and STILL move to a different object:

$$
C = \{(k^c_j, v^c_j)\}_{j=1}^{p}
$$

where $p \ll T$, but $(k^c_j, v^c_j)$ need not equal any original token's K/V pair. The slots can become learned memory.

## Learned Cache Artifacts: Compaction Instead Of Eviction

The next methods do not merely choose which original token states survive. They build compact K/V artifacts whose slots can move away from individual token identity.

### Cartridges: A Trainable KV Cache Per Corpus

The [Cartridges paper](https://arxiv.org/abs/2506.06266) asks a direct question: if users repeatedly ask questions against the same long corpus, can we train a compact KV cache for that corpus and reuse it?

The paper calls the training recipe **self-study**. First, the model uses the full corpus to generate supervision conversations about that corpus. Then the compact cache is trained with a context-distillation objective so future queries can use the cartridge instead of re-prefilling the whole source.

In the local implementation, a cartridge for one corpus stores, for every layer:

$$
K_c^{(l)}, V_c^{(l)}
\in \mathbb{R}^{H_{kv} \times p \times d_{head}}.
$$

The base model is frozen. The trainable parameters are the compact K/V tensors themselves.

The object is initialized from a prefix pass:

```python
@torch.no_grad()
def initialize_from_prefix_text(
    model,
    tokenizer,
    text: str,
    num_tokens: int,
    num_frozen_tokens: int = 1,
) -> TrainableKVCartridge:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][..., :num_tokens].to(model.device)
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = _normalize_past_key_values(outputs.past_key_values)
    keys = [layer[0].detach().to(model.dtype) for layer in past_key_values]
    values = [layer[1].detach().to(model.dtype) for layer in past_key_values]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)
```

This prefix initialization is easy to misread. It does not mean the final cartridge only contains the first $p$ tokens. It means optimization starts from the KV states of the first $p$ tokens. After training, the trainable slots are just parameters. They can move away from their original token identity.

The implementation also freezes a few initial positions as attention-sink-like anchors:

```python
if num_frozen_tokens > 0:
    frozen_key = nn.Parameter(key_tensor[..., :num_frozen_tokens, :], requires_grad=False)
    frozen_value = nn.Parameter(value_tensor[..., :num_frozen_tokens, :], requires_grad=False)
    train_key = nn.Parameter(key_tensor[..., num_frozen_tokens:, :])
    train_value = nn.Parameter(value_tensor[..., num_frozen_tokens:, :])
```

The layer is materialized by concatenating frozen and trainable regions:

```python
def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    key_parts = []
    value_parts = []
    if self.num_frozen_tokens > 0:
        key_parts.append(self.frozen_keys[index])
        value_parts.append(self.frozen_values[index])
    key_parts.append(self.trainable_keys[index])
    value_parts.append(self.trainable_values[index])
    return torch.cat(key_parts, dim=-2), torch.cat(value_parts, dim=-2)
```

At inference time, the cartridge is converted into the Hugging Face cache format:

```python
def as_cache(self, model_config) -> DynamicCache:
    return DynamicCache(ddp_cache_data=self.as_legacy_past_key_values(), config=model_config)
```

So from the frozen model's point of view, a cartridge is just `past_key_values`.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/cartridges-flow.svg" alt="Cartridges training and inference flow." />
  <p class="kv-caption"><a href="https://arxiv.org/abs/2506.06266">Cartridges</a> as an offline compilation workflow: generate supervision from the full corpus, optimize a compact K/V artifact, then reuse it for follow-up queries.</p>
</div>

#### The Cartridge Objective

The training signal in the local implementation is teacher distillation. First, a teacher answers using the full context. Then the cartridge is optimized so the frozen model, when given only the user query plus the cartridge cache, matches the teacher's answer-token distribution.

For one answer position $t$, let $q_t(i)$ be the teacher distribution over a sparse set of candidate tokens. Let

$$
p_{\theta}(i \mid u, C_s)
$$

be the frozen model's probability for token $i$, given user query $u$ and cartridge $C_s$ for source corpus $s$. The loss is:

$$
\mathcal{L}_{cart}(C_s)
= - \frac{1}{N} \sum_{(u,y)} \sum_t \sum_{i \in \mathcal{S}_t}
q_t(i) \log p_{\theta}(i \mid u, y_{<t}, C_s).
$$

Concretely, `_sparse_distillation_loss` is the helper that implements the equation. It receives student logits already sliced to the assistant-token positions, then reconstructs a sparse teacher distribution from stored top-logprobs. The `supervision` argument is the per-token view of the teacher answer:

```text
supervision: list[TokenSupervision]
  length == assistant_logits.shape[0]

TokenSupervision:
  token_id: int        # teacher's sampled/generated assistant token at this position
  logprob: float       # teacher log p(token_id)
  top_logprobs:
    - token_id: int    # candidate vocabulary id from the teacher's top-k distribution
      logprob: float   # teacher log p(candidate)
    - token_id: int
      logprob: float
    ...
```

So `supervision[row_idx]` describes the teacher distribution for the same assistant position as `logits[row_idx]`. In the raw synthesized conversation this information is stored as assistant `token_ids` plus sparse `top_logprobs`; the loss helper uses the expanded row-by-row form because it makes the alignment with the student logits explicit.

For example, suppose the teacher's answer starts with two assistant tokens, roughly `" Paris"` and `" is"`. Using illustrative token ids, the supervision might look like:

```text
assistant_logits.shape = [2, vocab_size]

supervision =
  - token_id: 12041        # decoded token: " Paris"
    logprob: -0.10
    top_logprobs:
      - token_id: 12041    # " Paris"
        logprob: -0.10
      - token_id: 987      # " London"
        logprob: -2.40
      - token_id: 14321    # " Lyon"
        logprob: -3.10

  - token_id: 374          # decoded token: " is"
    logprob: -0.18
    top_logprobs:
      - token_id: 374      # " is"
        logprob: -0.18
      - token_id: 596      # "'s"
        logprob: -2.20
      - token_id: 13       # "."
        logprob: -3.50
```

The first dictionary supervises `assistant_logits[0]`; the second supervises `assistant_logits[1]`. For the first row, the helper exponentiates the teacher logprobs, renormalizes those three candidates into a small distribution over `{12041, 987, 14321}`, and penalizes the student if `logits[0]` does not put matching mass on those ids. The same operation is repeated for `logits[1]`. This is distillation rather than ordinary next-token cross-entropy: the student is not only told "the answer token was `12041`"; it is also told that the teacher considered `" London"` much more plausible than `" Lyon"`.

```python
def _sparse_distillation_loss(
    logits: torch.Tensor,
    supervision: list[dict[str, Any]],
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_losses: list[torch.Tensor] = []

    for row_idx, token_supervision in enumerate(supervision):
        candidate_ids: list[int] = []
        candidate_weights: list[float] = []

        for candidate in token_supervision["top_logprobs"]:
            token_id = candidate.get("token_id")
            if token_id is None:
                continue
            candidate_ids.append(int(token_id))
            candidate_weights.append(math.exp(float(candidate["logprob"])))

        target_token_id = int(token_supervision["token_id"])
        if target_token_id not in candidate_ids:
            candidate_ids.append(target_token_id)
            candidate_weights.append(math.exp(float(token_supervision["logprob"])))

        weights = torch.tensor(candidate_weights, device=logits.device, dtype=torch.float32)
        weights = weights / weights.sum()
        candidate_ids_tensor = torch.tensor(candidate_ids, device=logits.device)
        token_losses.append(
            -(weights * log_probs[row_idx, candidate_ids_tensor]).sum()
        )

    return torch.stack(token_losses).mean()
```

There are three small but important details in that helper. The sparse teacher mass comes from `top_logprobs`; the actual target token is inserted if it fell outside the top-k list; and the remaining candidates are renormalized before cross-entropy is computed.

The forward pass aligns the student logits with the assistant target tokens by teacher-forcing the answer prefix:

```python
outputs = model(
    input_ids=model_input,
    past_key_values=cartridge.as_cache(model.config),
    use_cache=False,
)
target_len = len(example.assistant_token_ids)
start_idx = prompt_ids.shape[-1] - 1
end_idx = start_idx + target_len
assistant_logits = outputs.logits[0, start_idx:end_idx, :]
return _sparse_distillation_loss(assistant_logits, example.assistant_supervision)
```

The conceptual point is that the cartridge is trained through the frozen LLM. If moving one value vector helps the model put probability mass on the teacher's answer token, gradient descent will move that vector. This is why a cartridge slot is not a token slot after training.

#### Cartridges Benchmarks From The Local Repo

The [`cartridges`](https://github.com/shreyansh26/cartridges) repo includes stable benchmark reports on small Wikipedia-backed QA tasks. The exact-match numbers are not the most important part here because the semantic judge is a better fit for these generated-answer runs. The systems numbers are the point: query-time prefill drops because the model no longer needs to prefill the full corpus on every question.

| Experiment | Budget | Semantic Match: Full | Semantic Match: Cartridge | Compression | Prefill Speedup | E2E Query Speedup | Build Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wikipedia_india` | 512 | 1.000 | 0.800 | 16.143x | 8.284x | 1.759x | 121.36s |
| `wikipedia_india` | 1024 | 1.000 | 1.000 | 8.072x | 8.185x | 1.732x | 125.35s |
| `wikipedia_history_us` | 512 | 1.000 | 0.850 | 47.313x | 34.498x | 5.302x | 401.31s |
| `wikipedia_history_us` | 1024 | 1.000 | 1.000 | 23.656x | 37.788x | 5.998x | 401.03s |

<div class="kv-table-note">
The break-even point depends on how many follow-up queries reuse the trained cartridge. In these reports, `wikipedia_india` needs hundreds of follow-up queries to amortize the build cost, while the longer `wikipedia_history_us` setup has a much larger per-query advantage and breaks even sooner.
</div>

Cartridges are best read as an offline compilation strategy for a stable corpus. If the corpus is a codebase, legal document collection, textbook, or project memory that will receive many queries, a per-corpus optimization cost can be reasonable. If the corpus changes every request, the cost dominates.

### STILL: Learn The Compressor Instead Of The Cache

Cartridges optimize $C_s$ directly for every source corpus $s$. [STILL](https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/) asks whether that optimization can be amortized.

Instead of solving:

$$
C_s^\star = \arg\min_C \mathcal{L}(C; s)
$$

for every new corpus, STILL learns a function:

$$
f_\phi(K_s, V_s) \rightarrow C_s.
$$

Training the compactor $f_\phi$ is expensive, but it is reusable. For a new corpus, build the full cache once, run the compactor once, save the compact cache, and answer future questions against the compact artifact.

The Baseten write-up frames this as the missing amortization step for neural KV compaction. Cartridges validate that optimized compact caches can work, but they spend optimization on every new context. STILL spends optimization once on a compactor that can be reused across contexts.

In the local implementation, each transformer layer gets its own compactor:

```python
self.layers = nn.ModuleList(
    [
        StillLayerCompactor(
            head_dim=head_dim,
            num_latents=num_latents,
            rope_theta=rope_theta,
        )
        for _ in range(num_hidden_layers)
    ]
)
```

The layer compactor takes:

$$
K_l, V_l \in \mathbb{R}^{B \times H_{kv} \times T \times d}
$$

and returns:

$$
C_l^K, C_l^V \in \mathbb{R}^{B \times H_{kv} \times p \times d}
$$

plus an additive attention bias:

$$
\beta_l \in \mathbb{R}^{B \times H_{kv} \times p}.
$$

That bias is important. The compact values carry content. The compact keys define where queries land. The bias lets the compactor adjust the prior preference over compact slots after compression.

<div class="kv-figure">
  <img src="/assets/img/posts_images/kv_cache_compaction/still-flow.svg" alt="STILL compactor flow." />
  <p class="kv-caption"><a href="https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/">STILL</a>'s compactor path: unrotate RoPE keys, let perceiver latents read the cache, project compact K/V, and add a β bias for compressed attention mass.</p>
</div>

#### RoPE Makes Compaction Subtle

A raw key in a RoPE model is position-rotated. If we compress rotated keys directly, the compactor has to learn both content and position artifacts. STILL instead unrotates keys before feeding them to the perceiver, then rerotates compact keys at their new latent positions.

The implementation exposes this explicitly:

```python
def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float,
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = _rope_cos_sin(positions, dim=x.shape[-1], theta=theta)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if inverse:
        sin = -sin
    x_float = x.to(torch.float32)
    return ((x_float * cos) + (_rotate_half(x_float) * sin)).to(x.dtype)
```

Inside `StillLayerCompactor.forward`, `latent_positions` is not just metadata. It is threaded through the perceiver blocks and then used again when the compact keys are rotated back into the coordinate system expected by the frozen LLM:

```python
token_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
latent_positions = self._latent_positions(seq_len, keys.device)

unrotated_keys = apply_rope(keys, token_positions, theta=self.rope_theta, inverse=True)
kv_input = torch.cat([unrotated_keys, values], dim=-1)
kv_input = kv_input.squeeze(0).reshape(num_heads, seq_len, head_dim * 2).to(module_dtype)
latents = self.latents.unsqueeze(0).expand(num_heads, -1, -1).to(module_dtype)

for block in self.blocks:
    latents = block(
        latents,
        kv_input,
        latent_positions=latent_positions,
        token_positions=token_positions,
    )

compact_keys = self.key_head(latents)
compact_values = self.value_head(latents)
compact_biases = self.bias_head(latents).squeeze(-1)
compact_keys = apply_rope(compact_keys, latent_positions, theta=self.rope_theta)
```

The latent positions are evenly spaced across the original sequence:

```python
def _latent_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
    if self.num_latents == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    values = torch.linspace(
        0,
        max(seq_len - 1, 0),
        steps=self.num_latents,
        device=device,
        dtype=torch.float32,
    )
    return values.round().to(torch.long)
```

This is a simple way to say: compact slot $j$ represents some position in the original timeline. It is not the only possible scheme, but it gives the frozen model position-compatible keys.

#### STILL's Perceiver Block

The compactor uses learned latents. For each KV head, those latents cross-attend into the dense cache, then self-attend with one another.

One block does:

$$
\tilde{Z}^{(m)}
= \mathrm{RMSNorm}\left(
Z^{(m-1)} + \mathrm{CrossAttn}(Z^{(m-1)}, X)
\right)
$$

$$
Z^{(m)}
= \mathrm{RMSNorm}\left(
\tilde{Z}^{(m)} + \mathrm{SelfAttn}(\tilde{Z}^{(m)})
\right)
$$

where:

$$
X_l = [\mathrm{RoPE}^{-1}(K_l); V_l].
$$

The source code mirrors this:

```python
latents = self.cross_norm(latents + cross_out)
latents = self.self_norm(latents + self.self_attn(latents))
```

The cross-attention uses learned latent queries and the dense cache as memory:

```python
q = apply_rope(self.q_proj(latents), latent_positions, theta=self.rope_theta)
k = apply_rope(self.k_proj(kv_input), token_positions, theta=self.rope_theta)
v = self.v_proj(kv_input)
scale = 1.0 / math.sqrt(self.dim)
weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
outputs = self.out_proj(torch.matmul(weights, v))
```

After two perceiver blocks, the output heads produce compact keys, values, and biases:

```python
compact_keys = self.key_head(latents)
compact_values = self.value_head(latents)
compact_biases = self.bias_head(latents).squeeze(-1)
compact_keys = apply_rope(compact_keys, latent_positions, theta=self.rope_theta)
```

The initialization is deliberately structured. The key head initially reads the first half of the latent vector, the value head reads the second half, and the bias head starts at zero:

```python
self.key_head.weight.data.zero_()
self.key_head.weight.data[:, : self.head_dim] = torch.eye(self.head_dim)
self.value_head.weight.data.zero_()
self.value_head.weight.data[:, self.head_dim :] = torch.eye(self.head_dim)
```

That makes early behavior easier to reason about: the compactor starts closer to a copy-style mapping than arbitrary noise.

#### STILL Training Objective

The teacher is the frozen model with the full cache. The student is the same frozen model with the compact cache produced by the compactor. Training updates only the compactor.

The general objective in the repo is:

$$
\mathcal{L}_{still}
= \lambda_{KL}\,\mathrm{KL}(P_{teacher} \Vert P_{student})
+ \lambda_{CE}\,\mathrm{CE}(y, P_{student}).
$$

The implementation supports both terms:

```python
def _distillation_loss(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_token_ids: list[int],
    kl_weight: float,
    exact_token_ce_weight: float,
) -> torch.Tensor:
    loss = student_logits.new_tensor(0.0, dtype=torch.float32)
    if kl_weight > 0.0:
        loss = loss + (kl_weight * _kl_loss(teacher_logits, student_logits))
    if exact_token_ce_weight > 0.0:
        ce_term = _exact_token_ce_loss(student_logits, target_token_ids)
        loss = loss + (exact_token_ce_weight * ce_term)
    return loss
```

The teacher and student paths are computed in the same function:

```python
with torch.no_grad():
    full_outputs = model(input_ids=context_ids, use_cache=True)
    teacher_outputs = model(
        input_ids=model_input,
        past_key_values=full_outputs.past_key_values,
        use_cache=False,
    )

compact_cache = compactor(full_outputs.past_key_values)
student_outputs = model(
    input_ids=model_input,
    past_key_values=compact_cache.as_cache(model.config),
    still_layer_biases=compact_cache.biases,
    use_cache=False,
)
```

This is the crucial amortization step. During training, the compactor sees many contexts and learns how to compress caches. During deployment, a new context only needs:

```python
with torch.no_grad():
    outputs = model(input_ids=context_ids, use_cache=True)
    cache = compactor(outputs.past_key_values)
```

That is one full prefill plus one compactor pass, not hundreds of gradient steps on a new cache.

#### STILL Benchmark From The Local Repo

The STILL repo includes a final MCQ benchmark using `Qwen/Qwen3-4B`, 1024 compact latents, 115 training Wikipedia articles, and 20 held-out Wikipedia articles with 10 MCQ questions per held-out article.

The benchmark compares full context, truncation, STILL, and a cartridge baseline under aligned metric definitions.

| Method | Accuracy | Compression vs Full | Mean Query Total Latency | Mean Online Query Latency | Mean Target Preparation | Mean Target Total | Reusable Training |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context` | 0.950 | 1.000x | 178.388 ms | 178.388 ms | 0.000s | 1.784s | n/a |
| `truncation_1024` | 0.775 | 2.458x | 133.736 ms | 133.736 ms | 0.000s | 1.337s | n/a |
| `still_1024_ce_only` | 0.315 | 2.721x | 102.479 ms | 84.517 ms | 0.180s | 1.025s | 274.215s |
| `cartridge_1024` | 0.885 | 2.736x | 2537.627 ms | 189.468 ms | 23.482s | 25.376s | n/a |

The result is not "STILL beats cartridges on quality." It does not in this benchmark. Cartridges is much stronger on held-out MCQ accuracy because it optimizes a compact cache for each target page.

The result is: STILL shows the systems shape of a reusable compactor. The online latency is lowest, the per-target build is small, and the one-time training cost is separated from per-target preparation. The quality gap is the central open engineering problem in this reproduction.

### Cartridges Versus STILL

The easiest way to compare them is by where the optimization lives.

<div class="kv-mini-grid">
  <div class="kv-mini-card">
    <b>Cartridges</b>
    Optimize the compact KV tensors for one corpus. High per-target preparation cost, strong corpus-specific quality, no reusable compactor.
  </div>
  <div class="kv-mini-card">
    <b>STILL</b>
    Optimize a neural function that maps full caches to compact caches. Reusable training cost, cheap per-target build, harder generalization problem.
  </div>
</div>

Mathematically:

$$
\text{Cartridge:}\quad C_s^\star = \arg\min_C \mathcal{L}(C; s)
$$

$$
\text{STILL:}\quad \phi^\star = \arg\min_\phi \mathbb{E}_{s \sim \mathcal{D}}
\left[\mathcal{L}(f_\phi(K_s,V_s); s)\right].
$$

Cartridges are like per-document prompt tuning, except the prompt lives in KV space. STILL is like learning an encoder that produces those prompts directly.

Operationally:

| Question | Token Eviction | Cartridges | STILL |
| --- | --- | --- | --- |
| Keeps original token K/V only? | Yes | No | No |
| Needs per-corpus gradient optimization? | No | Yes | No |
| Needs reusable training? | No | No | Yes |
| Query-time artifact | Shorter cache | Trained cartridge | Compactor-built cache |
| Best fit | Streaming / bounded memory | Stable corpus with many queries | Many corpora after reusable training |

## What Changes In Practice

The last piece is to translate the taxonomy back into engineering tradeoffs: what is different from text prompt compression, what the implementations taught, and where the next measurements should go.

### Why This Is Not Just Prompt Compression

Prompt compression methods usually emit text. The model then tokenizes that text and builds a normal KV cache. That is useful, but it puts the bottleneck through language. If a detail is not in the summary, it is gone.

KV compaction emits model-internal vectors. This gives it a different kind of capacity. A compact slot can store information in a way that is not a grammatical sentence. It can act as a router, a value carrier, or an attention prior.

This is also why the methods are harder to debug. When a textual summary fails, we can read it. When a compact value vector fails, we need probes: attention maps, nearest-neighbor analyses, teacher-student token deltas, ablations by layer, and task-specific recall tests.

### Implementation Lessons

The code across these repos makes a few practical lessons clear.

**1. KV accounting should be canonical.** Use the tensor formula, not allocator memory, for method comparisons:

$$
T \cdot L \cdot H_{kv} \cdot d_{head} \cdot 2 \cdot b.
$$

Allocator memory still matters for deployment, but canonical bytes make algorithmic compression ratios comparable.

**2. Attention-score methods need score visibility.** SnapKV and H2O need attention matrices or attention history. That can conflict with kernels designed to avoid materializing attention probabilities.

**3. Positional encoding is part of the cache.** STILL's unrotate/compress/rerotate path exists because a key vector is not merely content. It is content after positional rotation.

**4. "Compact tokens" do not imply equal serving cost.** In the STILL benchmark, truncation and STILL both operate around a 1024-token budget, but truncation still reprefills the retained prompt on every query. STILL pays a target cache build once, then uses a short query prompt with the compact cache already loaded.

**5. Decode length can distort latency.** The STILL report explicitly tracks generated token counts because methods can normalize to the same MCQ answer while producing different raw completions.

### A Minimal Pseudocode Summary

Attention Sink:

```text
prefill(K, V, T):
    if T <= G + W:
        keep all
    else:
        keep [0:G] and [T-W:T]

decode(k_t, v_t):
    keep [0:G]
    replace oldest / lowest-position non-global slot with (k_t, v_t)
```

L2 norm pruning:

```text
score_i = -||k_i||_2
keep top B positions by score
append future decode tokens into freed slots
```

SnapKV:

```text
run prompt prefill
observe attention from last W prompt tokens to earlier prompt tokens
score_i = pooled_sum_attention_to_i
keep top (B - W) scored older tokens plus the last W tokens
```

H2O:

```text
for each decode step:
    update average attention history for cached tokens
    if cache is full:
        evict token with lowest history score
    insert current token
```

Cartridges:

```text
for each corpus:
    C = KV cache from first p prefix tokens
    freeze base model
    repeat gradient steps:
        teacher = model(full_context, query, answer_prefix)
        student = model(query, answer_prefix, past_key_values=C)
        update C to match teacher answer distribution
    save C
```

STILL:

```text
train once:
    for each training corpus:
        full_cache = frozen_model.prefill(corpus)
        compact_cache = compactor(full_cache)
        teacher = frozen_model(query, past_key_values=full_cache)
        student = frozen_model(query, past_key_values=compact_cache)
        update compactor to match teacher / target answer

serve new corpus:
    full_cache = frozen_model.prefill(corpus)
    compact_cache = compactor(full_cache)
    answer future queries from compact_cache
```

### Where I Would Push Next

The current implementations make the core ideas inspectable. The natural next steps are less about adding another eviction heuristic and more about measurement.

For Cartridges, I would want:

- layer-wise ablations: which layers need learned values versus mostly learned keys?
- query-count break-even curves under realistic serving assumptions;
- composition tests where multiple cartridges are loaded together;
- probes that distinguish factual storage from routing behavior.

For STILL, I would want:

- stronger training distributions than small MCQ supervision;
- KL-heavy or continuation-token training once decode collapse is controlled;
- different latent-position schedules;
- ablations for beta, RoPE unrotation, and identity initialization;
- iterative compaction tests where compact memory is prepended to the next chunk and compacted again.

For token eviction, I would want:

- apples-to-apples results under the same attention kernel constraints;
- memory accounting that separates effective cache size from allocated buffer size;
- long-horizon stability tests, not just short prompt generation.

### Takeaway

KV cache compression is not one trick. There are at least three levels:

1. **Eviction:** keep a smaller subset of original tokens.
2. **Per-corpus compaction:** optimize compact K/V tensors for one source.
3. **Amortized compaction:** train a reusable compressor that builds compact K/V tensors for new sources.

Attention Sink, L2 pruning, SnapKV, and H2O are useful because they make bounded-cache inference possible with relatively small changes. Cartridges are useful because they show that a compact KV object can behave like a corpus-specific memory. STILL is useful because it points toward the systems shape we actually want: compress a new context in one forward-pass-like build step, then reuse that compact memory across future queries.

The hard part is preserving the full-context model's behavior while paying less than full-context serving cost. That is exactly why this area is interesting: it sits at the boundary between representation learning, attention mechanics, and inference systems.

</div>

---

&nbsp;

<script type="text/javascript" src="//downloads.mailchimp.com/js/signup-forms/popup/unique-methods/embed.js" data-dojo-config="usePlainJson: true, isDebug: false"></script>

<!-- <button style="background-color: #70ab17; color: #1770AB" id="openpopup">Subscribe to my posts!</button> -->
<div class="button_cont" align="center"><button id="openpopup" class="example_a">Subscribe to my posts!</button></div>

<style>
    .example_a {
  color: #fff !important;
  text-transform: uppercase;
  text-decoration: none;
  background: #3f51b5;
  padding: 20px;
  border-radius: 5px;
  cursor: pointer;
  display: inline-block;
  border: none;
  transition: all 0.4s ease 0s;
    }

    .example_a:hover {
  background: #434343;
  letter-spacing: 1px;
  -webkit-box-shadow: 0px 5px 40px -10px rgba(0,0,0,0.57);
  -moz-box-shadow: 0px 5px 40px -10px rgba(0,0,0,0.57);
  box-shadow: 5px 40px -10px rgba(0,0,0,0.57);
  transition: all 0.4s ease 0s;
    }
</style>


<script type="text/javascript">

function showMailingPopUp() {
    window.dojoRequire(["mojo/signup-forms/Loader"], function(L) { L.start({"baseUrl":"mc.us4.list-manage.com","uuid":"0b10ac14f50d7f4e7d11cf26a","lid":"667a1bb3da","uniqueMethods":true}) })

    document.cookie = "MCPopupClosed=;path=/;expires=Thu, 01 Jan 1970 00:00:00 UTC";
}

document.getElementById("openpopup").onclick = function() {showMailingPopUp()};

</script>

&nbsp;  

<script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="shreyanshsingh" data-description="Support me on Buy me a coffee!" data-message="" data-color="#FF5F5F" data-position="Right" data-x_margin="18" data-y_margin="18"></script>

Follow me on [Twitter](https://twitter.com/shreyansh_26), [Github](https://github.com/shreyansh26) or connect on [LinkedIn](https://www.linkedin.com/in/shreyansh26/).
