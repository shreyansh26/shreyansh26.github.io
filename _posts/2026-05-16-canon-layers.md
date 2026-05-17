---
layout: post
title: "Canon Layers"
date: 2026-05-16
author: "Shreyansh Singh"
description: "A deep dive into Canon Layers: why sequence models need cheap horizontal token flow, how residual causal depthwise convolution implements it, and where Canon-A/B/C/D fit inside Transformer and linear-model blocks."
thumbnail: /assets/img/posts_images/canon_layers/canon-local-flow.svg
tags: llms transformers canon-layers
categories: ["LLMs"]
giscus_comments: true
related_posts: false
permalink: "post/2026-05-16_canon-layers/"
featured: false
toc:
  sidebar: left
pretty_table: true
_styles: |
  .canon-post {
    --canon-ink: #15120d;
    --canon-muted: #625949;
    --canon-paper: #f7f0e3;
    --canon-panel: #fffaf0;
    --canon-line: #d0ba94;
    --canon-line-dark: #8e7650;
    --canon-oxide: #9b3d20;
    --canon-oxide-dark: #702611;
    --canon-blue: #0a6670;
    --canon-green: #4f6f3c;
  }

  .canon-post .canon-hero {
    margin: 0 0 2rem;
    padding: 1.1rem;
    border: 1px solid var(--canon-line);
    border-radius: 0.5rem;
    background:
      radial-gradient(circle at 12% 14%, rgba(155, 61, 32, 0.15), transparent 16rem),
      linear-gradient(90deg, rgba(21, 18, 13, 0.035) 1px, transparent 1px),
      linear-gradient(180deg, rgba(21, 18, 13, 0.035) 1px, transparent 1px),
      var(--canon-paper);
    background-size: auto, 3rem 3rem, 3rem 3rem, auto;
    box-shadow: 0 1.25rem 3.5rem rgba(58, 41, 19, 0.12);
    color: var(--canon-ink);
  }

  .canon-post .canon-metric-strip,
  .canon-post .canon-lab-band,
  .canon-post .canon-placement-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin-top: 1.25rem;
  }

  .canon-post .canon-metric,
  .canon-post .canon-lab-card,
  .canon-post .canon-placement {
    box-sizing: border-box;
    min-width: 0;
    border: 1px solid var(--canon-line);
    border-radius: 0.5rem;
    background: rgba(255, 250, 240, 0.88);
    box-shadow: 0 0.7rem 1.7rem rgba(58, 41, 19, 0.08);
  }

  .canon-post .canon-metric,
  .canon-post .canon-placement {
    padding: 1rem;
  }

  .canon-post .canon-metric b {
    display: block;
    color: var(--canon-oxide-dark);
    font-family: monospace;
    font-size: 1.35rem;
    line-height: 1;
  }

  .canon-post .canon-metric span,
  .canon-post .canon-lab-card p,
  .canon-post .canon-placement span {
    display: block;
    margin: 0.45rem 0 0;
    color: var(--canon-muted);
    font-size: 0.92rem;
  }

  .canon-post .canon-lab-card {
    padding: 1rem;
    color: var(--canon-ink);
  }

  .canon-post .canon-lab-title {
    margin: 0 0 0.35rem;
    color: var(--canon-ink);
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.25;
  }

  .canon-post .canon-control {
    display: grid;
    gap: 0.55rem;
    margin-top: 0.72rem;
    color: #39342c;
    font-family: monospace;
    font-size: 0.78rem;
  }

  .canon-post .canon-control strong {
    color: var(--canon-oxide-dark);
  }

  .canon-post input[type="range"] {
    width: 100%;
    min-height: 2.4rem;
    accent-color: var(--canon-oxide);
  }

  .canon-post .canon-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.85rem;
    color: #39342c;
    font-family: monospace;
    font-size: 0.82rem;
  }

  .canon-post .canon-readout {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.6rem;
    margin-top: 0.9rem;
  }

  .canon-post .canon-readout div {
    min-width: 0;
    padding: 0.7rem;
    border-radius: 0.4rem;
    background: #17130e;
    color: #fff4df;
    font-family: monospace;
  }

  .canon-post .canon-readout small {
    display: block;
    color: #cdbf9f;
  }

  .canon-post .canon-readout span {
    display: block;
    margin-top: 0.25rem;
    color: #fff4df;
    font-size: 1.2rem;
    line-height: 1.2;
  }

  .canon-post .canon-equation {
    display: grid;
    gap: 0.42rem;
    margin-top: 0.9rem;
    padding: 0.75rem;
    border: 1px solid rgba(142, 118, 80, 0.42);
    border-radius: 0.45rem;
    background: rgba(255, 250, 240, 0.8);
    color: var(--canon-ink);
    font-family: monospace;
    font-size: 0.82rem;
    line-height: 1.45;
    overflow-x: auto;
  }

  .canon-post .canon-eq-line {
    white-space: nowrap;
  }

  .canon-post .canon-eq-symbol {
    color: var(--canon-oxide-dark);
    font-weight: 700;
  }

  .canon-post .canon-eq-value {
    color: var(--canon-oxide-dark);
    font-weight: 700;
  }

  .canon-post .canon-placement b {
    color: var(--canon-oxide-dark);
    font-family: monospace;
  }

  .canon-post blockquote {
    border-left: 0.3rem solid var(--canon-oxide);
    background: rgba(255, 250, 240, 0.78);
  }

  .canon-post .outer {
    margin: 1.6rem auto 2rem;
  }

  .canon-post .image {
    border: 1px solid var(--canon-line-dark);
    border-radius: 0.5rem;
    overflow: hidden;
    background: var(--canon-panel);
    box-shadow: 0 1.25rem 3.5rem rgba(58, 41, 19, 0.12);
  }

  .canon-post .image figcaption {
    padding: 0.75rem 0.9rem;
    border-top: 1px solid var(--canon-line);
    color: var(--canon-muted);
    background: rgba(241, 228, 207, 0.6);
    font-family: monospace;
    font-size: 0.75rem;
    line-height: 1.45;
  }

  .canon-post .image br {
    display: none;
  }

  @media (max-width: 767.98px) {
    .canon-post .canon-hero {
      padding: 0.95rem;
    }

    .canon-post .canon-metric-strip,
    .canon-post .canon-lab-band,
    .canon-post .canon-placement-grid {
      gap: 0.65rem;
      margin-top: 0.9rem;
    }

    .canon-post .canon-metric,
    .canon-post .canon-placement,
    .canon-post .canon-lab-card {
      padding: 0.85rem;
      width: 100%;
    }

    .canon-post .canon-metric b {
      font-size: 1.15rem;
    }

    .canon-post .canon-metric span,
    .canon-post .canon-lab-card p,
    .canon-post .canon-placement span {
      font-size: 0.88rem;
      line-height: 1.38;
    }

    .canon-post .canon-lab-title {
      font-size: 0.98rem;
    }

    .canon-post .canon-control {
      gap: 0.35rem;
      margin-top: 0.5rem;
      font-size: 0.72rem;
    }

    .canon-post input[type="range"] {
      min-height: 1.8rem;
    }

    .canon-post .canon-readout {
      gap: 0.45rem;
      margin-top: 0.65rem;
    }

    .canon-post .canon-readout div {
      padding: 0.55rem;
    }

    .canon-post .canon-readout span {
      font-size: 1rem;
    }

    .canon-post .canon-equation {
      padding: 0.6rem;
      font-size: 0.74rem;
    }
  }

  @media (min-width: 768px) {
    .canon-post .canon-hero {
      padding: 1.5rem;
    }

    .canon-post .canon-metric-strip {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .canon-post .canon-lab-band,
    .canon-post .canon-placement-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (min-width: 1200px) {
    .canon-post .canon-hero {
      padding: 2rem;
    }
  }
---

<div class="canon-post" markdown="1">

<section class="canon-hero" data-toc-skip>
  <div class="canon-metric-strip" aria-label="Key numbers">
    <div class="canon-metric"><b>K = 4</b><span>the short causal window used in the main Canon implementation</span></div>
    <div class="canon-metric"><b>DK</b><span>depthwise parameter cost, instead of the full-convolution cost $D^2K$</span></div>
    <div class="canon-metric"><b>A/B/C/D</b><span>four insertion points: before attention, inside attention, before MLP, inside MLP</span></div>
  </div>
  <div class="canon-lab-band" aria-label="Interactive Canon intuition panels">
    <article class="canon-lab-card">
      <p class="canon-lab-title">Local causal mixer</p>
      <p>A Canon layer adds a small weighted mixture of nearby past token states to the current token.</p>
      <label class="canon-control"><span>$h_{t-3}$: <strong id="canon-x-label-0">0.25</strong></span><input data-canon-control id="canon-x-0" type="range" min="-2" max="2" value="0.25" step="0.05" /></label>
      <label class="canon-control"><span>$h_{t-2}$: <strong id="canon-x-label-1">0.50</strong></span><input data-canon-control id="canon-x-1" type="range" min="-2" max="2" value="0.50" step="0.05" /></label>
      <label class="canon-control"><span>$h_{t-1}$: <strong id="canon-x-label-2">0.75</strong></span><input data-canon-control id="canon-x-2" type="range" min="-2" max="2" value="0.75" step="0.05" /></label>
      <label class="canon-control"><span>$h_t$: <strong id="canon-x-label-3">1.00</strong></span><input data-canon-control id="canon-x-3" type="range" min="-2" max="2" value="1.00" step="0.05" /></label>
      <label class="canon-control"><span>$w_{t-3}$: <strong id="canon-w-label-0">0.20</strong></span><input data-canon-control id="canon-w-0" type="range" min="-1" max="1" value="0.20" step="0.05" /></label>
      <label class="canon-control"><span>$w_{t-2}$: <strong id="canon-w-label-1">0.30</strong></span><input data-canon-control id="canon-w-1" type="range" min="-1" max="1" value="0.30" step="0.05" /></label>
      <label class="canon-control"><span>$w_{t-1}$: <strong id="canon-w-label-2">0.40</strong></span><input data-canon-control id="canon-w-2" type="range" min="-1" max="1" value="0.40" step="0.05" /></label>
      <label class="canon-control"><span>$w_t$: <strong id="canon-w-label-3">0.10</strong></span><input data-canon-control id="canon-w-3" type="range" min="-1" max="1" value="0.10" step="0.05" /></label>
      <label class="canon-toggle"><input data-canon-control id="canon-residual-toggle" type="checkbox" checked /> residual add $+h_t$</label>
      <div class="canon-equation" aria-label="Live Canon mixer calculation">
        <div class="canon-eq-line"><span class="canon-eq-symbol">mixed</span> = <span id="canon-mix-formula">0.20*0.25 + 0.30*0.50 + 0.40*0.75 + 0.10*1.00 = 0.600</span></div>
        <div class="canon-eq-line"><span class="canon-eq-symbol">output</span> = <span id="canon-output-formula">0.600 + residual(1.00) = 1.600</span></div>
      </div>
      <div class="canon-readout">
        <div><small>conv mixture</small><span id="canon-mixed-out" aria-live="polite">0.600</span></div>
        <div><small>Canon output</small><span id="canon-output-out" aria-live="polite">1.600</span></div>
      </div>
    </article>
    <article class="canon-lab-card">
      <p class="canon-lab-title">Depthwise vs full local convolution</p>
      <p>Canon uses a separate short causal filter per channel. Full convolution would also mix channels and is much more expensive.</p>
      <label class="canon-control"><span>hidden width $D$: <strong id="canon-d-label">4096</strong></span><input data-canon-control id="canon-d-slider" type="range" min="512" max="8192" value="4096" step="512" /></label>
      <label class="canon-control"><span>kernel size $K$: <strong id="canon-k-label">4</strong></span><input data-canon-control id="canon-k-slider" type="range" min="2" max="8" value="4" step="1" /></label>
      <div class="canon-equation" aria-label="Live convolution parameter calculation">
        <div class="canon-eq-line"><span class="canon-eq-symbol">depthwise</span> = D*K = <span id="canon-depthwise-formula">4,096*4 = 16,384</span></div>
        <div class="canon-eq-line"><span class="canon-eq-symbol">full</span> = D^2*K = <span id="canon-full-formula">4,096^2*4 = 67,108,864</span></div>
        <div class="canon-eq-line"><span class="canon-eq-symbol">ratio</span> = full/depthwise = <span id="canon-ratio-formula">4,096x</span></div>
      </div>
      <div class="canon-readout">
        <div><small>depthwise params</small><span id="canon-depthwise-params" aria-live="polite">16,384</span></div>
        <div><small>full-conv params</small><span id="canon-full-params" aria-live="polite">67,108,864</span></div>
      </div>
      <div class="canon-readout">
        <div><small>parameter ratio</small><span id="canon-param-ratio" aria-live="polite">4,096x</span></div>
        <div><small>Canon cost</small><span>$O(BTDK)$</span></div>
      </div>
    </article>
  </div>
</section>

Canon Layers are a small architectural primitive from Zeyuan Allen-Zhu's *Physics of Language Models: Part 4.1*. The basic idea is simple: give every token a cheap causal path to nearby past token states.

That path is not meant to replace attention. It handles a different job.

> Attention should spend capacity on content-addressed routing and retrieval. It should not have to spend layers on routine neighbor-to-neighbor transport.

The mechanism is a residual causal depthwise convolution over the sequence axis. It is local, cheap, and easy to insert into existing Transformer, linear-attention, and SSM-style blocks.

## 1. The missing path in a standard Transformer

A pre-norm Transformer block usually has:

$$
x^{(\ell+\frac12)}
=
x^{(\ell)}
+
\operatorname{Attn}\left(\operatorname{Norm}(x^{(\ell)})\right),
$$

$$
x^{(\ell+1)}
=
x^{(\ell+\frac12)}
+
\operatorname{MLP}\left(\operatorname{Norm}(x^{(\ell+\frac12)})\right).
$$

This gives two strong paths:

- a vertical residual path, where token position $t$ preserves and refines its own representation across layers;
- a global attention path, where token position $t$ can retrieve content from previous positions.

But the MLP is pointwise over tokens. It mixes channels, not positions. Attention can move information from $t-1$ to $t$, but attention is a global content-routing mechanism. Using it for routine local relay is expensive and depth-inefficient.

Canon adds a third path:

$$
\text{nearby causal context}
\quad\rightarrow\quad
\text{current token state}.
$$

That is why the paper describes Canon as horizontal information flow. The ordinary residual stream is vertical across depth; Canon is local residual flow across positions.

{% include image.liquid url="/assets/img/posts_images/canon_layers/canon-local-flow.svg" description="Canon as local horizontal residual flow: the current token receives a small learned mixture of nearby causal states." %}

## 2. Associative recall shows the problem

Consider the causal sequence:

```text
[A] [B] ... [A] [?]
```

The desired next token is `[B]`. A natural mechanism is:

1. the second `[A]` attends to the first `[A]`;
2. the representation at the first `[A]` carries enough information to identify the following `[B]`;
3. the model predicts `[B]`.

The catch is causal masking. The first `[A]` cannot see its future neighbor `[B]` at the same layer. A model often needs one operation to move information locally from `[B]` into a neighboring representation, then another operation to retrieve it globally.

Canon makes that first local enrichment cheap.

## 3. The Canon operator

Let a sequence of hidden states be:

$$
H=(h_1,\ldots,h_T),
\qquad
h_t\in\mathbb{R}^{m}.
$$

A width-4 Canon layer computes:

$$
\widetilde h_t
=
w_0\odot h_t
+w_1\odot h_{t-1}
+w_2\odot h_{t-2}
+w_3\odot h_{t-3},
$$

where $w_r\in\mathbb{R}^{m}$ are learned channelwise weights, $\odot$ is elementwise multiplication, and missing past states are zero-padded.

The residual form is:

$$
h'_t
=
h_t
+
\operatorname{Conv1D}_{\mathrm{causal},K=4}
\left(h_t,h_{t-1},h_{t-2},h_{t-3}\right).
$$

Equivalently, for batch index $b$, position $t$, channel $c$, and kernel size $K$:

$$
y_{b,t,c}
=
x_{b,t,c}
+
\sum_{r=0}^{K-1}
a_{c,r}\,x_{b,t-r,c},
$$

with $x_{b,t-r,c}=0$ when $t-r<0$.

The key word is **depthwise**. Channel $c$ reads only channel $c$ over nearby positions. Canon does not perform hidden-dimension mixing; the projections and MLP still own that job.

## 4. Why the residual matters

Without the residual path:

$$
h'_t=\operatorname{Canon}(H)_t.
$$

With the residual path:

$$
h'_t=h_t+\operatorname{Canon}(H)_t.
$$

The residual version is easier to insert because it starts as a local perturbation around the existing representation. If the local signal is useful, the model can add it. If it is not useful, the model can learn small weights without destroying the vertical residual stream.

The Canon paper's ablations report that residual Canon is materially more stable and efficient than non-residual variants. The implementation also exposes `canon_residual` as a configuration flag, with the released LlamaCanon path defaulting to residual behavior.

## 5. Canon is not local attention

Local attention computes content-dependent weights:

$$
y_t
=
\sum_{j=t-w}^{t}\alpha_{t,j}v_j,
\qquad
\alpha_{t,j}
=
\operatorname{softmax}_j
\left(
\frac{q_t^\top k_j}{\sqrt{d_h}}
\right).
$$

Canon computes fixed learned local propagation:

$$
y_t
=
x_t
+
\sum_{r=0}^{K-1}a_r\odot x_{t-r}.
$$

The distinction matters:

| Mechanism | Main job | Weights depend on content? | Scope |
|---|---|---:|---|
| Full attention | global retrieval and routing | yes | all past tokens |
| Local attention | adaptive local retrieval | yes | local window |
| Canon | cheap causal transport | no, in the studied version | tiny causal window |
| MLP | channel transformation | no token mixing | one token |

<br>
Canon is closer to a short learned transport operator than to a retrieval mechanism.

## 6. Where Canon goes in a Transformer block

The paper studies four insertion points. For hidden width $d$, Canon-ABCD means:

{% include image.liquid url="/assets/img/posts_images/canon_layers/canon-abcd.svg" description="Canon-A/B/C/D insertion points in a pre-norm Transformer block." %}

<div class="canon-placement-grid" markdown="1">
  <div class="canon-placement"><b>Canon-A</b><span>after attention RMSNorm, before Q/K/V projections; width $m=d$</span></div>
  <div class="canon-placement"><b>Canon-B</b><span>after Q/K/V projections, on the concatenated projected representation; width $m=n_qd_h+2n_{kv}d_h$ for GQA</span></div>
  <div class="canon-placement"><b>Canon-C</b><span>after MLP RMSNorm, before the MLP projections; width $m=d$</span></div>
  <div class="canon-placement"><b>Canon-D</b><span>inside the MLP, before activation; for gated MLPs it acts on concatenated gate/up branches</span></div>
</div>

### Canon-A: before attention

$$
u
=
\operatorname{Canon}_A(\operatorname{Norm}(x)).
$$

Then:

$$
q=W_qu,
\qquad
k=W_ku,
\qquad
v=W_vu.
$$

Attention receives token states that already contain a short causal neighborhood.

### Canon-B: inside attention

After Q/K/V projection:

$$
z_t=[q_t;k_t;v_t].
$$

Canon-B applies local mixing to that projected representation:

$$
z'_t=\operatorname{Canon}_B(z)_t,
\qquad
[q'_t;k'_t;v'_t]=z'_t.
$$

For ordinary MHA with equal Q/K/V widths, $m=3d$. For grouped-query attention:

$$
m=n_qd_h+2n_{kv}d_h.
$$

The released LlamaCanon code computes exactly this total dimension before constructing `canonB`.

### Canon-C: before the MLP

$$
r
=
\operatorname{Canon}_C(\operatorname{Norm}(x^{(\ell+\frac12)})).
$$

The MLP receives a locally enriched representation.

### Canon-D: inside the MLP

For a gated MLP:

$$
\operatorname{MLP}(r)
=
W_{\mathrm{down}}
\left(
\phi(W_{\mathrm{gate}}r)
\odot
W_{\mathrm{up}}r
\right).
$$

LlamaCanon concatenates the gate and up projections:

$$
z_t=[g_t;u_t],
\qquad
z'_t=\operatorname{Canon}_D(z)_t,
\qquad
[g'_t;u'_t]=z'_t,
$$

then computes:

$$
W_{\mathrm{down}}\left(\phi(g'_t)\odot u'_t\right).
$$

For a Llama-style gated MLP with intermediate width $\frac{8}{3}d$, Canon-D has width:

$$
m=2\cdot\frac{8}{3}d=\frac{16}{3}d.
$$

## 7. Canon-ABCD pseudocode

The same residual local mixer appears at different internal representations:

```python
def canon_residual(x, canon_conv):
    # x: [batch, seq, channels]
    # canon_conv: causal depthwise Conv1d over the sequence dimension
    return x + canon_conv(x)
```

For a Llama-style pre-norm block with grouped-query attention and a gated MLP:

```python
def llama_block_with_canon(x, mask=None, cache=None):
    # x: [B, T, d]

    residual = x
    h = rmsnorm_attn(x)

    if canonA is not None:
        h = canon_residual(h, canonA)          # [B, T, d]

    q = q_proj(h)
    k = k_proj(h)
    v = v_proj(h)

    if canonB is not None:
        qkv = concat([q, k, v], dim=-1)
        qkv = canon_residual(qkv, canonB)
        q, k, v = split(qkv, [q_dim, k_dim, v_dim], dim=-1)

    q = apply_rope(q)
    k = apply_rope(k)
    a = causal_attention(q, k, v, mask=mask, cache=cache)
    x = residual + o_proj(a)

    residual = x
    h = rmsnorm_mlp(x)

    if canonC is not None:
        h = canon_residual(h, canonC)          # [B, T, d]

    gate = gate_proj(h)
    up = up_proj(h)

    if canonD is not None:
        z = concat([gate, up], dim=-1)
        z = canon_residual(z, canonD)
        gate, up = z.chunk(2, dim=-1)

    x = residual + down_proj(silu(gate) * up)
    return x
```

Partial variants such as Canon-AC, Canon-ACD, or Canon-ABC are also meaningful. The paper's ablations find that the benefits are cumulative, and that Canon-ACD can help even without modifying the attention projections.

## 8. Tensor shapes for the core mixer

The minimal PyTorch version for a `[B,T,D]` tensor is:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonResidualMixer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        # x: [B, T, D]
        xt = x.transpose(1, 2)                    # [B, D, T]
        xt = F.pad(xt, (self.kernel_size - 1, 0)) # [B, D, T + K - 1]
        mixed = self.conv(xt)                     # [B, D, T]
        mixed = mixed.transpose(1, 2)             # [B, T, D]
        return x + mixed
```

Shape summary:

| Step | Shape | Meaning |
|---|---:|---|
| input `x` | `[B,T,D]` | Transformer layout |
| transpose | `[B,D,T]` | Conv1d layout |
| left pad by `K-1` | `[B,D,T+K-1]` | causal boundary handling |
| depthwise conv | `[B,D,T]` | local sequence mixing |
| transpose back | `[B,T,D]` | Transformer layout |
| residual add | `[B,T,D]` | unchanged external shape |

For $K=4$, one channel computes:

$$
\operatorname{mixed}_{b,t,c}
=
a_{c,0}x_{b,t-3,c}
+a_{c,1}x_{b,t-2,c}
+a_{c,2}x_{b,t-1,c}
+a_{c,3}x_{b,t,c}.
$$

Then:

$$
y_{b,t,c}=x_{b,t,c}+\operatorname{mixed}_{b,t,c}.
$$

## 9. Why `groups=channels` matters

With depthwise convolution:

```python
nn.Conv1d(D, D, K, groups=D)
```

the parameter tensor has shape:

$$
[D,1,K],
$$

so parameters scale as:

$$
DK.
$$

With full convolution:

```python
nn.Conv1d(D, D, K, groups=1)
```

the parameter tensor has shape:

$$
[D,D,K],
$$

so parameters scale as:

$$
D^2K.
$$

For $D=4096$ and $K=4$:

$$
DK=16{,}384,
\qquad
D^2K\approx 67\text{ million}.
$$

That gap is the reason Canon isolates local sequence transport from channel mixing. Channel mixing remains in the projections and MLP, where it already exists.

{% include image.liquid url="/assets/img/posts_images/canon_layers/depthwise-vs-full-conv.svg" description="Depthwise Canon uses one short causal filter per channel. A full local convolution would mix every channel into every output channel." %}

## 10. Complexity and runtime

For batch $B$, sequence length $T$, hidden width $D$, and small kernel $K$:

$$
\operatorname{cost}_{\mathrm{Canon}}
=
O(BTDK).
$$

The attention matrix/value aggregation term is roughly:

$$
\operatorname{cost}_{\mathrm{attention}}
=
O(BT^2D),
$$

plus projection costs.

Asymptotically, Canon is tiny. Practically, it is not free: every additional operator can add memory movement and kernel-launch overhead. The Part 4.1 paper reports that Canon-ABCD adds fewer than $0.45\%$ parameters for GPT-2-small, and for a 1.3B Llama-style model it adds about $0.0063\%$ parameters. The same footnote reports nonzero naive H100 runtime overheads, with Canon-AC cheaper than Canon-ABCD.

The released code uses a `ShortConvolution` wrapper with `causal_conv1d` when available and when the kernel is in $\{2,3,4\}$. During generation, the convolution cache stores only the last $K$ states per channel:

$$
\text{cache shape}=[B,D,K].
$$

## 11. The synthetic playground

The paper argues that academic-scale real-data pretraining can be too noisy for architecture science. Perplexity mixes many skills together; benchmark swings can hide whether an architecture improved reasoning, knowledge storage, local composition, or something else.

The Part 4.1 experiments therefore use five controlled synthetic pretraining tasks:

| Task | Capability | Core requirement |
|---|---|---|
| Depo | reasoning depth | follow a directed permutation for $k$ hops |
| Brevo | reasoning breadth | process recursive dependencies in a DAG |
| Capo | knowledge capacity | store synthetic facts in parameters |
| Mano | knowledge manipulation | retrieve learned facts and compute over them |
| Lano | hierarchical structure | learn CFG-like recursive constraints |

The point is not that synthetic tasks are the final benchmark. They isolate mechanisms. If a change improves Depo but not Capo, or helps NoPE but not RoPE, the result is easier to interpret than a single mixed-corpus loss number.

### Depo: depth

Depo builds a directed permutation from key-value pairs:

```text
<bos> x1 y1 x2 y2 ... xn yn <query_k> q <ans> a <eos>
```

If the pairs define $f(x_i)=y_i$, the target is:

$$
a=f^{(k)}(q).
$$

The model must compute the $k$-hop successor internally, without writing intermediate chain-of-thought tokens. Depo2 makes each node span multiple tokens, so a 4-token Canon window cannot solve the task by direct copying. The local mixer must improve segment representations that attention can later chain globally.

### Brevo: breadth

Brevo gives the model a directed acyclic graph and asks for recursive dependencies in topological order. The hard part is not one long chain; it is parallel dependency processing across branches.

### Capo: capacity

Capo measures reliable storage of synthetic facts, often as bits per parameter. Limited-exposure regimes are important because overtraining can hide architectural differences.

### Mano: manipulation

Mano uses modular arithmetic expressions. The model must retrieve learned operation tables and compose them internally. This tests manipulation of knowledge stored in weights rather than only information present in the prompt.

### Lano: structure

Lano uses CFG-like sequences with local ambiguity. Correct prediction can require maintaining recursive global structure rather than memorizing nearby tokens.

## 12. What the results imply

For Transformer-style models, the Part 4.1 paper reports that Canon-ABCD improves reasoning depth by roughly $2$-$4\times$ in the controlled setup, reasoning breadth by about $30\%$, knowledge manipulation length by about $30\%$, and knowledge capacity in limited-exposure factual-storage regimes.

The strongest interpretation is not that Canon solves every task inside a four-token window. It is that better local representations make later global routing easier.

The NoPE result is especially interesting. NoPE means no positional embedding. Without positional encoding, a Transformer has weak order information. With Canon, NoPE becomes far stronger, often competitive with RoPE+Canon in the reported synthetic setup. A causal convolution injects order-sensitive local structure:

$$
h_t
\leftarrow
h_t+f(h_t,h_{t-1},h_{t-2},h_{t-3}).
$$

The paper also studies partial RoPE. With Canon present, reduced-RoPE variants can work well, which matters because heavy RoPE usage can hurt length generalization.

## 13. Linear models and SSMs

The paper compares Transformers, GLA, Mamba2, and GDN under the same synthetic tasks. A useful takeaway is that local convolution-like components inside some linear/SSM architectures already explain a lot of their behavior.

In the paper's terminology:

- Mamba2's internal `conv1d` resembles a partial non-residual Canon-B;
- GLA and GDN implementations also contain conv-like local components;
- adding Canon systematically makes comparisons fairer because every model receives the same local-transport primitive.

After adding Canon broadly, linear models still tend to lag full-attention Transformers on deep retrieval-heavy reasoning. The diagnosis is not only state size. The harder problem is memory dynamics: compressed recurrent state must preserve and retrieve fine-grained facts across multiple hops without compounding errors.

The Part 4.2 code release extends the story to real-world pretraining recipes and released model families, including LlamaCanon, GLA, GDN, and Mamba2 variants.

## 14. Canon versus related mechanisms

### Primer

Primer introduced squared ReLU and a depthwise convolution after Q/K/V projection. The Q/K/V convolution part is closest to Canon-B without the residual path:

$$
q'=\operatorname{DWConv}(W_qx),
\qquad
k'=\operatorname{DWConv}(W_kx),
\qquad
v'=\operatorname{DWConv}(W_vx).
$$

Canon generalizes the idea in three ways:

1. it adds an explicit residual around the local mixer;
2. it applies the primitive at A/B/C/D, not only Q/K/V;
3. it studies the primitive across Transformers, linear attention, and SSM-style models.

### Longformer-style local attention

Longformer sparsifies attention with sliding windows and task-specific global attention. Canon works on a different axis.

Local attention asks:

$$
\text{which nearby tokens should I retrieve from?}
$$

Canon asks:

$$
\text{what nearby hidden signal should be cheaply propagated?}
$$

They can coexist.

### Mamba2

Mamba2 is built around state-space duality and selective SSM computation. Its local convolution is a frontend to a recurrent/SSM memory system:

$$
x'_t=\operatorname{Conv1D}(x_{t-K+1:t}),
\qquad
h_t=A_th_{t-1}+B_tx'_t,
\qquad
y_t=C_t^\top h_t.
$$

Canon isolates the local convolutional part as a reusable residual primitive that can be applied outside a specific SSM block.

### Uniform attention

Earlier Physics of Language Models work found that uniform averaging over recent tokens could help CFG-style tasks. Canon can be viewed as a learned, channelwise, modular version of that local averaging:

$$
\text{uniform local average}
\quad\rightarrow\quad
\text{learned channelwise local residual convolution}.
$$

## 15. Implementation details from LlamaCanon

The released LlamaCanon helper uses a `ShortConvolution` module:

```python
ShortConvolution(
    hidden_size=dim,
    kernel_size=config.canon_kernel,
    bias=config.canon_bias,
    activation="silu" if config.canon_activation else None,
    use_fast_conv1d=causal_conv1d_available and config.canon_kernel in [2, 3, 4],
)
```

It is dimension-last at the interface:

$$
x\in\mathbb{R}^{B\times T\times D},
$$

then rearranges internally to Conv1d layout:

$$
x\in\mathbb{R}^{B\times D\times T}.
$$

The helper masks padded positions, uses the fast `causal_conv1d` kernel when available, and supports decode-time cache updates through a `[B,D,K]` state.

The code exposes:

- `canon_set`, selecting any subset of `A`, `B`, `C`, `D`;
- `canon_kernel`, usually $4$;
- `canon_residual`, controlling whether the output is `hidden_states + hidden_states2`;
- `canon_activation`, available but not recommended by the paper for Transformer Canon layers;
- `canon_bias`, generally avoided.

For packed or padded batches, Canon must respect the same valid-token mask as attention. Otherwise, a causal convolution can propagate padding artifacts into valid positions.

## 16. Practical choices

### Initialization

There are several reasonable options:

1. **Default initialization.** This matches the released implementation path.
2. **Zero initialization.** This makes Canon an exact identity at step zero:

   $$
   y=x+0=x.
   $$

   That is useful when retrofitting Canon into an already trained model.

3. **Past-average initialization.** For $K=4$, initialize previous offsets to $\frac13$ and current offset to $0$:

   $$
   y_t=x_t+\frac13(x_{t-1}+x_{t-2}+x_{t-3}).
   $$

   This tests the local-context hypothesis directly, but it is a design choice rather than the default released setup.

### Causal padding

Use left padding:

```python
F.pad(x, (K - 1, 0))
```

Right padding would either shift outputs incorrectly or leak future information.

### Optimized kernels

`torch.nn.Conv1d` is the generic API. It is not automatically the optimized Dao-AILab `causal-conv1d` path. The implementation must call that package explicitly, as LlamaCanon's helper does.

## 17. Open engineering questions

### Runtime overhead

Parameter overhead is tiny, but runtime overhead is still real. Multiple small convolutions can add memory traffic and kernel launches. A production implementation would likely fuse Canon with adjacent projections or batch several Canon calls together.

### Dynamic Canon

The studied operator uses fixed learned weights. A dynamic version could use input-conditioned local weights:

$$
y_t
=
x_t
+
\sum_{r=0}^{K-1}a_r(x_t)\odot x_{t-r}.
$$

That moves Canon closer to lightweight local attention. It may improve expressivity, but it also changes the clean cost and interpretation.

### MoE interaction

Canon-D inside a mixture-of-experts MLP is awkward because neighboring tokens may be routed to different experts. Canon-ABC is easier; Canon-D requires a more careful dispatch design.

### Long-range compression

Canon improves local flow. It does not remove the hard problem of preserving high-fidelity information through compressed recurrent state or across very long contexts. The paper's linear-model results still suggest that full attention remains stronger for some deep in-context reasoning tasks.

## 18. Summary

Canon Layers are lightweight residual causal convolutions over neighboring token representations:

$$
h'_t
=
h_t
+
\sum_{r=0}^{K-1}w_r\odot h_{t-r}.
$$

The architecture split is clean:

- attention handles content-addressed global routing;
- MLPs handle channelwise nonlinear transformation;
- Canon handles cheap local token-to-token propagation.

The empirical claim from the Canon paper is that this small primitive improves controlled measures of reasoning depth, reasoning breadth, knowledge manipulation, NoPE viability, and several linear/SSM architectures. The implementation claim is equally simple: Canon is depthwise causal Conv1D with a residual path, placed at selected A/B/C/D points inside a block.

Canon is not interesting because "convolution is back." It is interesting because local horizontal flow is useful enough to deserve its own architectural slot.

## References

- Zeyuan Allen-Zhu, [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://arxiv.org/abs/2512.17351), 2025.
- Zeyuan Allen-Zhu, [Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality](https://physics.allen-zhu.com/part-4-architecture-design/part-4-2), 2025.
- facebookresearch, [PhysicsLM4 code release](https://github.com/facebookresearch/PhysicsLM4).
- David R. So et al., [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668), NeurIPS 2021.
- Tri Dao and Albert Gu, [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060), ICML 2024.
- Iz Beltagy, Matthew E. Peters, and Arman Cohan, [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150), 2020.

<script src="/assets/js/canon-layers.js" defer></script>

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