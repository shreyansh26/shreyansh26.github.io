---
layout: post
title: "Paper Summary #17 - Engram"
date: 2026-05-17
author: "Shreyansh Singh"
description: "A technical explainer for DeepSeek's Engram layers: conditional memory, hashed n-gram lookup, context-aware gating, sparse-capacity allocation, and the implementation path inside Transformer blocks."
thumbnail: /assets/img/posts_images/engram_layers/engram-note-13.png
tags: llms transformers engram memory sparsity paper-summaries
categories: ["LLMs", "MLSys"]
giscus_comments: true
related_posts: false
permalink: "post/2026-05-17_engram-layers/"
featured: false
toc:
  sidebar: left
pretty_table: true
_styles: |
    .engram-post {
      --engram-ink: #171511;
      --engram-muted: #4d463a;
      --engram-paper: #f3eee3;
      --engram-paper-strong: #fffaf0;
      --engram-paper-muted: #e5dac7;
      --engram-line: rgba(34, 29, 21, 0.16);
      --engram-line-strong: rgba(34, 29, 21, 0.32);
      --engram-night: #080907;
      --engram-night-2: #111510;
      --engram-copper: #b06f3b;
      --engram-copper-dark: #8f562c;
      --engram-teal: #1c6f75;
      --engram-green: #527c44;
      --engram-plum: #5a3c57;
      --engram-gold: #d5b15d;
      --engram-blue: #315b85;
      --engram-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --engram-body: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --engram-display: Georgia, "Times New Roman", serif;
      color: var(--engram-ink);
      font-family: var(--engram-body);
      line-height: 1.65;
    }

    .engram-post *,
    .engram-post *::before,
    .engram-post *::after {
      box-sizing: border-box;
    }

    .engram-post img {
      display: block;
      max-width: 100%;
      height: auto;
    }

    .engram-post a {
      color: var(--global-theme-color);
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.18em;
    }

    .engram-post code,
    .engram-post pre,
    .engram-post .mono {
      font-family: var(--engram-mono);
    }

    .engram-post .concept-title {
      color: inherit;
      font-family: var(--engram-display);
      letter-spacing: 0;
    }

    .engram-post .phrase-buttons,
    .engram-post .token-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
    }

    .engram-post .phrase-buttons button {
      min-height: 2.5rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--engram-line-strong);
      border-radius: 999px;
      padding: 0 0.85rem;
      color: var(--engram-ink);
      background: rgba(255, 255, 255, 0.64);
      font-family: var(--engram-mono);
      font-size: 0.78rem;
      text-decoration: none;
    }

    .engram-post .phrase-buttons button:hover,
    .engram-post .phrase-buttons button.is-active {
      color: var(--engram-ink);
      border-color: rgba(213, 177, 93, 0.55);
      background: rgba(213, 177, 93, 0.16);
    }

    .engram-post .concept-strip {
      margin: 0 0 2rem;
      padding: 1rem;
      border: 1px solid var(--engram-line);
      border-radius: 0.5rem;
      background:
        radial-gradient(circle at 12% 18%, rgba(176, 111, 59, 0.18), transparent 16rem),
        linear-gradient(90deg, rgba(23, 21, 17, 0.045) 1px, transparent 1px),
        linear-gradient(180deg, rgba(23, 21, 17, 0.035) 1px, transparent 1px),
        var(--engram-paper);
      background-size: auto, 4rem 4rem, 4rem 4rem, auto;
    }

    .engram-post .concept-strip__inner,
    .engram-post .figure-pair,
    .engram-post .split,
    .engram-post .paper-grid,
    .engram-post .metrics,
    .engram-post .gallery {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .engram-post .concept,
    .engram-post .note-block,
    .engram-post .equation-block,
    .engram-post .asset-figure,
    .engram-post .memory-visual,
    .engram-post .lab-panel,
    .engram-post .allocation-demo,
    .engram-post .timeline__item,
    .engram-post .ladder-step,
    .engram-post .paper-card,
    .engram-post .thumb,
    .engram-post .metric {
      border: 1px solid var(--engram-line);
      border-radius: 0.5rem;
      background: rgba(255, 250, 240, 0.9);
      box-shadow: 0 0.8rem 2rem rgba(35, 23, 10, 0.08);
    }

    .engram-post .concept {
      padding: 1rem;
    }

    .engram-post .concept span {
      color: var(--engram-teal);
      font-family: var(--engram-mono);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .engram-post .concept-title {
      display: block;
      margin: 0.25rem 0 0.35rem;
      font-size: 1.35rem;
      line-height: 1.15;
    }

    .engram-post .concept p,
    .engram-post .story-section p,
    .engram-post .paper-card p,
    .engram-post .timeline__item p {
      color: var(--engram-muted);
    }

    .engram-post .story-section {
      margin: 2.5rem 0;
      padding-top: 0.25rem;
      scroll-margin-top: 6rem;
    }

    .engram-post .lead {
      max-width: none;
      color: #312b22;
      font-size: 1.12rem;
      line-height: 1.55;
    }

    .engram-post li {
      margin: 0.45rem 0;
      color: var(--engram-muted);
    }

    .engram-post .note-block {
      margin: 1.25rem 0;
      padding: 1rem;
      border-left: 0.35rem solid var(--engram-copper);
    }

    .engram-post .note-block p:first-child,
    .engram-post .note-block p:last-child {
      margin: 0;
    }

    .engram-post .equation-block {
      margin: 1rem 0;
      padding: 1rem;
      overflow-x: auto;
      background: #111510;
      color: #fffaf0;
    }

    .engram-post .equation-block mjx-container {
      min-width: max-content;
    }

    .engram-post .asset-figure {
      margin: 1.4rem 0;
      overflow: hidden;
      background: var(--engram-paper-strong);
    }

    .engram-post .asset-figure img {
      width: 100%;
      cursor: zoom-in;
    }

    .engram-post figcaption {
      padding: 0.75rem 0.9rem;
      border-top: 1px solid var(--engram-line);
      color: var(--engram-muted);
      background: rgba(229, 218, 199, 0.48);
      font-family: var(--engram-mono);
      font-size: 0.74rem;
      line-height: 1.45;
    }

    .engram-post figcaption a,
    .engram-post .gallery-source a,
    .engram-post .paper-card a,
    .engram-post .references a {
      color: var(--engram-copper-dark);
    }

    .engram-post .source-credit {
      display: block;
      margin-top: 0.35rem;
      color: #6f6657;
    }

    .engram-post .split--wide {
      align-items: stretch;
    }

    .engram-post .memory-visual {
      display: grid;
      min-height: 18rem;
      place-items: center;
      padding: 1rem;
      background: radial-gradient(circle at 30% 20%, rgba(28, 111, 117, 0.18), transparent 13rem), #fffaf0;
    }

    .engram-post .memory-path {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 1rem;
      width: 100%;
    }

    .engram-post .token-row {
      align-items: center;
    }

    .engram-post .token {
      display: inline-flex;
      min-height: 2.35rem;
      align-items: center;
      justify-content: center;
      padding: 0 0.75rem;
      border-radius: 0.35rem;
      color: #fffaf0;
      background: var(--engram-plum);
      font-family: var(--engram-mono);
      font-size: 0.82rem;
      font-weight: 700;
    }

    .engram-post .hash-box {
      width: 4rem;
      height: 4rem;
      display: grid;
      place-items: center;
      border: 1px solid rgba(213, 177, 93, 0.55);
      border-radius: 50%;
      color: var(--engram-gold);
      background: var(--engram-night-2);
      font-family: var(--engram-display);
      font-size: 2rem;
      box-shadow: 0 0.9rem 2rem rgba(8, 9, 7, 0.16);
    }

    .engram-post .memory-stack {
      display: grid;
      gap: 0.35rem;
    }

    .engram-post .memory-row {
      height: 0.65rem;
      width: calc(var(--scale) * 100%);
      min-width: 42%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--engram-copper), var(--engram-teal));
    }

    .engram-post .table-wrap {
      margin: 1.25rem 0;
      overflow-x: auto;
      border: 1px solid var(--engram-line);
      border-radius: 0.5rem;
      background: var(--engram-paper-strong);
    }

    .engram-post table {
      width: 100%;
      margin: 0;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    .engram-post th,
    .engram-post td {
      min-width: 10rem;
      padding: 0.8rem;
      border-bottom: 1px solid var(--engram-line);
      vertical-align: top;
    }

    .engram-post th {
      color: #fffaf0;
      background: var(--engram-night-2);
      font-family: var(--engram-mono);
      font-size: 0.74rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .engram-post .table-wrap table thead th {
      color: #fffaf0 !important;
      background: var(--engram-night-2) !important;
    }

    .engram-post .table-wrap table thead th * {
      color: inherit !important;
    }

    .engram-post tr:last-child td {
      border-bottom: 0;
    }

    .engram-post .idea-ladder,
    .engram-post .timeline {
      display: grid;
      gap: 0.5rem;
      margin: 1.25rem 0;
    }

    .engram-post .ladder-step,
    .engram-post .timeline__item,
    .engram-post .paper-card,
    .engram-post .lab-panel,
    .engram-post .allocation-demo,
    .engram-post .metric {
      padding: 1rem;
    }

    .engram-post .ladder-step strong,
    .engram-post .metric strong {
      display: block;
      color: var(--engram-copper-dark);
      font-family: var(--engram-mono);
    }

    .engram-post .ladder-step span,
    .engram-post .metric span {
      color: var(--engram-muted);
      font-size: 0.9rem;
    }

    .engram-post .timeline__item {
      padding: 0.75rem 0.9rem;
      box-shadow: none;
    }

    .engram-post .timeline-title {
      display: block;
      color: var(--engram-ink);
      font-weight: 700;
      line-height: 1.3;
    }

    .engram-post .timeline__item p {
      margin: 0.25rem 0 0;
      font-size: 0.92rem;
      line-height: 1.45;
    }

    .engram-post .paper-grid {
      margin: 1.25rem 0;
    }

    .engram-post .split--stack {
      grid-template-columns: 1fr !important;
    }

    .engram-post .paper-card h3 {
      margin: 0.25rem 0 0.6rem;
      font-size: 1.3rem;
    }

    .engram-post .paper-card p {
      margin-bottom: 0;
      font-size: 0.92rem;
    }

    .engram-post .lab-panel {
      margin: 1.25rem 0;
      overflow: hidden;
      background: linear-gradient(135deg, rgba(28, 111, 117, 0.12), rgba(176, 111, 59, 0.12)), var(--engram-paper-strong);
    }

    .engram-post .lab-panel__head h3,
    .engram-post .allocation-demo h3 {
      margin-top: 0;
    }

    .engram-post .phrase-buttons button {
      cursor: pointer;
    }

    .engram-post .hash-stage {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      margin-top: 1rem;
    }

    .engram-post .hash-stage > div {
      min-width: 0;
    }

    .engram-post .slot-list {
      display: grid;
      gap: 0.5rem;
    }

    .engram-post .slot-line {
      display: grid;
      grid-template-columns: 4.5rem 1fr 4rem;
      align-items: center;
      gap: 0.6rem;
      color: var(--engram-muted);
      font-family: var(--engram-mono);
      font-size: 0.78rem;
    }

    .engram-post .slot-bar {
      height: 0.65rem;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(8, 9, 7, 0.12);
    }

    .engram-post .slot-bar i {
      display: block;
      width: var(--w);
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--engram-teal), var(--engram-gold));
    }

    .engram-post .slider-line {
      display: grid;
      gap: 0.5rem;
      margin: 1rem 0;
    }

    .engram-post input[type="range"] {
      width: 100%;
      min-height: 2.75rem;
      accent-color: var(--engram-copper);
    }

    .engram-post .bar-track {
      display: flex;
      min-height: 2.5rem;
      overflow: hidden;
      border: 1px solid var(--engram-line-strong);
      border-radius: 999px;
      background: rgba(8, 9, 7, 0.09);
    }

    .engram-post .bar-moe,
    .engram-post .bar-engram {
      display: grid;
      place-items: center;
      color: #fffaf0;
      font-family: var(--engram-mono);
      font-size: 0.78rem;
      transition: width 160ms ease;
    }

    .engram-post .bar-moe {
      background: var(--engram-teal);
    }

    .engram-post .bar-engram {
      background: var(--engram-copper);
    }

    .engram-post .metrics {
      margin-top: 1rem;
    }

    .engram-post .quote-line {
      margin: 1.25rem 0;
      padding: 1.2rem;
      border-left: 0.35rem solid var(--engram-teal);
      border-radius: 0.5rem;
      color: var(--engram-ink);
      background: rgba(28, 111, 117, 0.1);
      font-family: var(--engram-display);
      font-size: 1.35rem;
      line-height: 1.3;
    }

    .engram-post .references {
      counter-reset: ref;
      padding-left: 0;
      list-style: none;
    }

    .engram-post .references li {
      counter-increment: ref;
      display: grid;
      grid-template-columns: 2.4rem 1fr;
      gap: 0.6rem;
    }

    .engram-post .references li::before {
      content: counter(ref, decimal-leading-zero);
      color: var(--engram-copper-dark);
      font-family: var(--engram-mono);
      font-weight: 700;
    }

    .engram-post .gallery {
      margin: 1.25rem 0;
    }

    .engram-post .thumb {
      overflow: hidden;
    }

    .engram-post .thumb img {
      width: 100%;
      aspect-ratio: 16 / 10;
      object-fit: cover;
      cursor: zoom-in;
    }

    .engram-post .thumb span {
      display: block;
      min-height: 2.4rem;
      padding: 0.55rem 0.65rem;
      color: var(--engram-muted);
      font-family: var(--engram-mono);
      font-size: 0.72rem;
    }

    .engram-post .lightbox {
      position: fixed;
      inset: 0;
      z-index: 2000;
      display: none;
      place-items: center;
      padding: 1rem;
      background: rgba(8, 9, 7, 0.9);
    }

    .engram-post .lightbox.is-open {
      display: grid;
    }

    .engram-post .lightbox img {
      max-width: min(96vw, 84rem);
      max-height: 90vh;
      border-radius: 0.35rem;
      box-shadow: 0 2rem 5rem rgba(0, 0, 0, 0.48);
    }

    .engram-post .lightbox button {
      position: absolute;
      top: 1rem;
      right: 1rem;
      width: 2.75rem;
      height: 2.75rem;
      border: 1px solid rgba(255, 250, 240, 0.32);
      border-radius: 999px;
      color: #fffaf0;
      background: rgba(255, 255, 255, 0.08);
      cursor: pointer;
      font-family: var(--engram-mono);
      font-size: 1.25rem;
    }

    .engram-post .reveal {
      opacity: 0;
      transform: translateY(1.4rem);
      transition: opacity 520ms ease, transform 520ms ease;
    }

    .engram-post .reveal.is-visible {
      opacity: 1;
      transform: translateY(0);
    }

    @media (min-width: 768px) {
      .engram-post .concept-strip__inner,
      .engram-post .metrics {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .engram-post .figure-pair,
      .engram-post .split,
      .engram-post .hash-stage {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .engram-post .paper-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .engram-post .gallery {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }

    @media (min-width: 1100px) {
      .engram-post .gallery {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
    }

    @media (prefers-reduced-motion: reduce) {
      .engram-post .reveal {
        opacity: 1;
        transform: none;
        transition: none;
      }
    }

    html[data-theme="dark"] .engram-post {
      --engram-ink: #f2eadf;
      --engram-muted: #d4c8b8;
      --engram-paper: #211f1a;
      --engram-paper-strong: #2a261f;
      --engram-paper-muted: #393126;
      --engram-line: rgba(255, 250, 240, 0.16);
      --engram-line-strong: rgba(255, 250, 240, 0.3);
      --engram-night: #080907;
      --engram-night-2: #121610;
      --engram-copper: #d3915b;
      --engram-copper-dark: #f0b47f;
      --engram-teal: #63c1c8;
      --engram-green: #9ac47d;
      --engram-plum: #6d4668;
      --engram-gold: #f0cf78;
      --engram-blue: #8fb9df;
      color: var(--engram-ink);
    }

    html[data-theme="dark"] .engram-post .concept-strip {
      border-color: var(--engram-line);
      background:
        radial-gradient(circle at 12% 18%, rgba(211, 145, 91, 0.18), transparent 16rem),
        linear-gradient(90deg, rgba(255, 250, 240, 0.045) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255, 250, 240, 0.035) 1px, transparent 1px),
        var(--engram-paper);
    }

    html[data-theme="dark"] .engram-post .concept,
    html[data-theme="dark"] .engram-post .note-block,
    html[data-theme="dark"] .engram-post .memory-visual,
    html[data-theme="dark"] .engram-post .lab-panel,
    html[data-theme="dark"] .engram-post .allocation-demo,
    html[data-theme="dark"] .engram-post .timeline__item,
    html[data-theme="dark"] .engram-post .ladder-step,
    html[data-theme="dark"] .engram-post .paper-card,
    html[data-theme="dark"] .engram-post .thumb,
    html[data-theme="dark"] .engram-post .metric {
      border-color: var(--engram-line);
      background: rgba(42, 38, 31, 0.94);
      box-shadow: 0 0.8rem 2rem rgba(0, 0, 0, 0.24);
    }

    html[data-theme="dark"] .engram-post .asset-figure,
    html[data-theme="dark"] .engram-post .table-wrap {
      border-color: var(--engram-line);
      background: var(--engram-paper-strong);
      box-shadow: 0 0.8rem 2rem rgba(0, 0, 0, 0.24);
    }

    html[data-theme="dark"] .engram-post .lead,
    html[data-theme="dark"] .engram-post .concept p,
    html[data-theme="dark"] .engram-post .story-section p,
    html[data-theme="dark"] .engram-post .paper-card p,
    html[data-theme="dark"] .engram-post .timeline__item p,
    html[data-theme="dark"] .engram-post li,
    html[data-theme="dark"] .engram-post .ladder-step span,
    html[data-theme="dark"] .engram-post .metric span,
    html[data-theme="dark"] .engram-post .slot-line,
    html[data-theme="dark"] .engram-post .thumb span {
      color: var(--engram-muted);
    }

    html[data-theme="dark"] .engram-post .timeline-title,
    html[data-theme="dark"] .engram-post .quote-line,
    html[data-theme="dark"] .engram-post .paper-card h3 {
      color: var(--engram-ink);
    }

    html[data-theme="dark"] .engram-post .phrase-buttons button {
      color: var(--engram-ink);
      border-color: var(--engram-line-strong);
      background: rgba(255, 250, 240, 0.06);
    }

    html[data-theme="dark"] .engram-post .phrase-buttons button:hover,
    html[data-theme="dark"] .engram-post .phrase-buttons button.is-active {
      color: var(--engram-ink);
      border-color: rgba(240, 207, 120, 0.62);
      background: rgba(240, 207, 120, 0.16);
    }

    html[data-theme="dark"] .engram-post .memory-visual {
      background: radial-gradient(circle at 30% 20%, rgba(99, 193, 200, 0.16), transparent 13rem), var(--engram-paper-strong);
    }

    html[data-theme="dark"] .engram-post .lab-panel {
      background: linear-gradient(135deg, rgba(99, 193, 200, 0.12), rgba(211, 145, 91, 0.12)), var(--engram-paper-strong);
    }

    html[data-theme="dark"] .engram-post .equation-block {
      border-color: rgba(255, 250, 240, 0.2);
      background: #0f130d;
      color: #fffaf0;
    }

    html[data-theme="dark"] .engram-post figcaption {
      border-top-color: var(--engram-line);
      color: var(--engram-muted);
      background: rgba(32, 28, 22, 0.92);
    }

    html[data-theme="dark"] .engram-post .source-credit {
      color: #c7b8a4;
    }

    html[data-theme="dark"] .engram-post .slot-bar,
    html[data-theme="dark"] .engram-post .bar-track {
      background: rgba(255, 250, 240, 0.12);
    }

    html[data-theme="dark"] .engram-post .quote-line {
      background: rgba(99, 193, 200, 0.12);
    }
---

<div class="engram-post" markdown="1">

**Paper:** [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)  
**Official implementation:** [DeepSeek-AI/Engram](https://github.com/deepseek-ai/Engram)

----

<section class="concept-strip" aria-label="Core primitives" data-toc-skip>
    <div class="concept-strip__inner">
<div class="concept reveal">
  <span>Attention</span>
  <strong class="concept-title">Context mixing</strong>
  <p>Self-attention links tokens inside the current sequence and carries global context forward.</p>
</div>
<div class="concept reveal">
  <span>MoE</span>
  <strong class="concept-title">Conditional computation</strong>
  <p>Experts increase transformation capacity while activating only a few FFNs per token.</p>
</div>
<div class="concept reveal">
  <span>Engram</span>
  <strong class="concept-title">Conditional memory</strong>
  <p>Hashed n-grams retrieve static vectors, then the hidden state decides whether to inject them.</p>
</div>
    </div>
  </section>

<section class="story-section reveal" id="problem" data-title="The Problem" markdown="1">
## Attention is not memory

  <p class="lead">Self-attention can resolve relationships in a sentence. It does not automatically provide a grounded representation of what the entities actually are.</p>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-01.png" alt="Harry ambiguity among several possible Harry entities" data-lightbox>
      <figcaption>Token association is not enough: "Harry" can point to many entities.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-02.png" alt="Harry Potter grounded by facts such as wizard and Hogwarts" data-lightbox>
      <figcaption>"Harry Potter" becomes useful when it retrieves a richer factual cluster.</figcaption>
    </figure>
  </div>

  <p>In a standard Transformer, this grounding is reconstructed through repeated computation. Attention composes nearby tokens. Feed-forward layers transform features. Later layers gradually turn surface strings into semantic representations.</p>

  <div class="note-block">
    <p>The Engram paper frames this as an architectural mismatch: dynamic reasoning should use computation, while common static phrases should often use lookup.</p>
  </div>
</section>

<section class="story-section reveal" id="ffn-memory" data-title="FFNs as Memory" markdown="1">
## The FFN already looks like a memory

  <p class="lead">A Transformer MLP can be read as a bank of pattern detectors and value writers.</p>

  <div class="equation-block">
    $$\operatorname{FFN}(h) = W_{\text{down}} \, \sigma(W_{\text{up}}h + b_{\text{up}}) + b_{\text{down}}.$$
  </div>

  <p>Geva et al. showed that FFNs behave like key-value memories: rows of $W_{\text{up}}$ detect patterns, while columns of $W_{\text{down}}$ write value vectors into the residual stream.</p>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-03.png" alt="FFN up projection as pattern detection" data-lightbox>
      <figcaption>Up-projection features act like soft keys.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-04.png" alt="FFN down projection writing value information" data-lightbox>
      <figcaption>Down-projection values write information back to the residual stream.</figcaption>
    </figure>
  </div>

  <p>MoE scales this by adding many FFNs and routing each token to a few experts:</p>

  <div class="equation-block">
    $$\operatorname{MoE}(h_t)=\sum_{i \in \operatorname{TopK}(r(h_t))} p_i(h_t)E_i(h_t).$$
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-05.png" alt="Mixture of experts as conditional computation" data-lightbox>
    <figcaption>MoE increases the number of possible transformations, but still performs runtime computation.</figcaption>
  </figure>
</section>

<section class="story-section reveal" id="lookup" data-title="Lookup" markdown="1">
## Static facts want tables

  <p class="lead">For a single token, lookup is simple. For phrases, the combinatorics explode.</p>

  <div class="split split--wide">
    <div>
      <p>A token embedding table maps an ID directly to a vector:</p>
      <div class="equation-block">
        $$e = E[x], \qquad E \in \mathbb{R}^{|V| \times d}.$$
      </div>
      <p>But facts are usually phrase-level. "Harry" is ambiguous; "Harry Potter" is a much more specific key.</p>
    </div>
    <div class="memory-visual" aria-hidden="true">
      <div class="memory-path">
        <div class="token-row">
          <span class="token">Harry</span>
          <span class="token">Potter</span>
        </div>
        <div class="hash-box" aria-label="Indexing function phi">&Phi;</div>
        <div class="memory-stack">
          <div class="memory-row" style="--scale: .55"></div>
          <div class="memory-row" style="--scale: .82"></div>
          <div class="memory-row" style="--scale: .65"></div>
          <div class="memory-row" style="--scale: .95"></div>
          <div class="memory-row" style="--scale: .72"></div>
          <div class="memory-row" style="--scale: .5"></div>
          <div class="memory-row" style="--scale: .88"></div>
          <div class="memory-row" style="--scale: .62"></div>
        </div>
      </div>
    </div>
  </div>

  <p>A direct bigram table with $|V|=128{,}000$ would have:</p>

  <div class="equation-block">
    $$|V|^2 = 128{,}000^2 = 16{,}384{,}000{,}000.$$
  </div>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-07.png" alt="Single token lookup table" data-lightbox>
      <figcaption>Single-token lookup is manageable.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-08.png" alt="Bigram lookup table explosion" data-lightbox>
      <figcaption>Direct bigram lookup is already huge.</figcaption>
    </figure>
  </div>
</section>

<section class="story-section reveal" id="hashing" data-title="Hashing" markdown="1">
## Hash the local phrase

  <p class="lead">Engram compresses token IDs, hashes suffix n-grams, and retrieves rows from multiple embedding tables.</p>

  <p>First, a tokenizer projection maps raw token IDs into canonical IDs:</p>
  <div class="equation-block">
    $$P: V \to V', \qquad x'_t = P(x_t).$$
  </div>

  <p>Then Engram forms suffix n-grams:</p>
  <div class="equation-block">
    $$g_{t,n} = (x'_{t-n+1}, \ldots, x'_t).$$
  </div>

  <p>Each hash head maps the compressed n-gram into a table row:</p>
  <div class="equation-block">
    $$z_{t,n,k} = \phi_{n,k}(g_{t,n}), \qquad e_{t,n,k} = E_{n,k}[z_{t,n,k}].$$
    $$e_t=\big\Vert_{n=2}^{N}\big\Vert_{k=1}^{K}e_{t,n,k}.$$
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-09.png" alt="Hash lookup overview" data-lightbox>
    <figcaption>A hash function maps the local phrase to a row in a learned memory table.</figcaption>
  </figure>
### Why multiplicative-XOR?


  <p>Addition creates structured collisions and loses order. Plain XOR also loses order because it is commutative. Engram uses position-specific multipliers before XOR:</p>

  <div class="equation-block">
    $$\phi_{n,k}(g_{t,n})=
    \left(\bigoplus_{i=0}^{n-1}m^{(\ell,k)}_i x'_{t-i}\right)\bmod M_{n,k}.$$
  </div>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-10.png" alt="Addition hash weakness" data-lightbox>
      <figcaption>Addition keeps nearby IDs nearby.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-13.png" alt="Multiplicative XOR hash" data-lightbox>
      <figcaption>Position-specific multipliers make the hash order-sensitive.</figcaption>
    </figure>
  </div>
</section>

<section class="story-section reveal" id="hash-lab" data-title="Hash Lab" markdown="1">
## A small hash lab

  <p class="lead">This toy demo is not DeepSeek's implementation. It makes the design intuition visible: one phrase produces several independent table addresses.</p>

  <div class="lab-panel">
    <div class="lab-panel__head">
      <h3>Multi-head lookup</h3>
      <p>Choose a phrase and watch eight simulated heads map it to different slots. Multi-head hashing makes a total collision across all heads much less likely.</p>
      <div class="phrase-buttons" id="phraseButtons">
        <button type="button" data-phrase="Harry Potter" class="is-active">Harry Potter</button>
        <button type="button" data-phrase="Potter Harry">Potter Harry</button>
        <button type="button" data-phrase="Diana Princess Wales">Diana Princess Wales</button>
        <button type="button" data-phrase="the Milky Way">the Milky Way</button>
      </div>
    </div>
    <div class="hash-stage">
      <div>
        <div class="token-row" id="hashTokens"></div>
        <div style="height: 1rem"></div>
        <div class="hash-box" aria-label="Indexing function phi">&Phi;</div>
      </div>
      <div>
        <div class="slot-list" id="slotList"></div>
      </div>
    </div>
  </div>

  <div class="equation-block">
    $$\Pr[\forall k,\ \phi_k(a)=\phi_k(b)] \approx \prod_{k=1}^{K}\frac{1}{M_k}.$$
  </div>
</section>

<section class="story-section reveal" id="gating" data-title="Gating" markdown="1">
## Lookup needs a gate

  <p class="lead">Static memory is useful only when the current context agrees with it.</p>

  <p>The retrieved vector $e_t$ is projected into a key and value:</p>
  <div class="equation-block">
    $$k_t = W_K e_t, \qquad v_t = W_V e_t.$$
  </div>

  <p>The hidden state is the query. The scalar gate is:</p>
  <div class="equation-block">
    $$\alpha_t=\sigma\left(
    \frac{\operatorname{RMSNorm}(h_t)^\top\operatorname{RMSNorm}(k_t)}{\sqrt{d}}
    \right).$$
    $$\tilde{v}_t=\alpha_t v_t.$$
  </div>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-20.png" alt="Key and value projections from retrieved memory" data-lightbox>
      <figcaption>Memory becomes a key for relevance and a value for content.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-21.png" alt="Context-aware scalar gate" data-lightbox>
      <figcaption>The current hidden state decides how much memory to admit.</figcaption>
    </figure>
  </div>
### Short convolution


  <p>After gating, Engram applies a short depthwise causal convolution and a residual path:</p>
  <div class="equation-block">
    $$Y=\operatorname{SiLU}\left(\operatorname{Conv1D}(\operatorname{RMSNorm}(\tilde{V}))\right)+\tilde{V}.$$
    $$H^{(\ell)} \leftarrow H^{(\ell)} + Y.$$
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-23.png" alt="Short convolution applied after gating" data-lightbox>
    <figcaption>The convolution lets nearby gated values interact before residual injection.</figcaption>
  </figure>
</section>

<section class="story-section reveal" id="architecture" data-title="Architecture" markdown="1">
## Inside the Transformer, not just at the input

  <p class="lead">Engram is inserted into selected Transformer blocks. The paper's 27B model uses layers 2 and 15.</p>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-25.png" alt="Engram inserted into transformer blocks" data-lightbox>
    <figcaption>Engram augments selected blocks while the ordinary token embedding and LM head remain intact.</figcaption>
  </figure>

  <div class="timeline">
    <div class="timeline__item">
      <strong class="timeline-title">Layer 1 is too raw</strong>
      <p>The hidden state is still close to token embeddings, so context-aware gating has little context to use.</p>
    </div>
    <div class="timeline__item">
      <strong class="timeline-title">Layer 2 is the sweet spot</strong>
      <p>One round of attention is enough to make the gate useful while still being early enough to save depth.</p>
    </div>
    <div class="timeline__item">
      <strong class="timeline-title">Middle layers refine</strong>
      <p>A later Engram module catches associations that only become clear after partial processing.</p>
    </div>
  </div>

  <p>For multi-branch mHC backbones, Engram shares the memory table and value projection, but uses branch-specific key projections:</p>

  <div class="equation-block">
    $$\alpha^{(m)}_t=
    \sigma\left(
    \frac{\operatorname{RMSNorm}(h^{(m)}_t)^\top\operatorname{RMSNorm}(W^{(m)}_K e_t)}{\sqrt{d}}
    \right),$$
    $$u^{(m)}_t=\alpha^{(m)}_t(W_Ve_t).$$
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-24.png" alt="Branch-specific gating in multi-branch architecture" data-lightbox>
    <figcaption>Different residual branches can use the same memory vector differently.</figcaption>
  </figure>
</section>

<section class="story-section reveal" id="allocation" data-title="Allocation" markdown="1">
## How much memory is enough?

  <p class="lead">Engram's strongest empirical claim is that sparse capacity should be split between MoE and memory.</p>

  <div class="equation-block">
    $$P_{\text{sparse}}=P_{\text{tot}}-P_{\text{act}}.$$
    $$P_{\text{MoE}}^{(\text{sparse})}=\rho P_{\text{sparse}}, \qquad P_{\text{Engram}}=(1-\rho)P_{\text{sparse}}.$$
  </div>

  <div class="allocation-demo" id="allocationDemo">
    <h3>Sparsity allocation</h3>
    <p>Move the slider. The paper's optimum appears around $\rho \approx 0.75$ to $0.80$, where most sparse capacity remains MoE but a meaningful chunk becomes Engram memory.</p>
    <div class="slider-line">
      <label for="rhoRange" class="mono">rho = <span id="rhoValue">0.80</span></label>
      <input id="rhoRange" type="range" min="40" max="100" value="80" step="1">
    </div>
    <div class="allocation-bars">
      <div class="bar-track">
        <div class="bar-moe" id="moeBar" style="width: 80%">MoE</div>
        <div class="bar-engram" id="engramBar" style="width: 20%">Engram</div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric">
        <span>MoE sparse share</span>
        <strong id="moeShare">80%</strong>
      </div>
      <div class="metric">
        <span>Engram sparse share</span>
        <strong id="engramShare">20%</strong>
      </div>
      <div class="metric">
        <span>Toy validation loss</span>
        <strong id="lossValue">1.711</strong>
      </div>
    </div>
  </div>

  <div class="figure-pair">
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-28.png" alt="Allocation curve with rho around 0.8" data-lightbox>
      <figcaption>The paper finds a U-shaped validation-loss curve, with the best region near rho 0.75-0.80.</figcaption>
    </figure>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-29.png" alt="Engram scaling with embedding slots" data-lightbox>
      <figcaption>Increasing memory slots keeps improving loss over the tested range.</figcaption>
    </figure>
  </div>
</section>

<section class="story-section reveal" id="results" data-title="Results" markdown="1">
## What changes at scale?

  <p class="lead">Engram-27B is iso-parameter and iso-FLOPs relative to MoE-27B. The win comes from reallocating sparse capacity, not from spending more activated compute.</p>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Total params</th>
          <th>Activated params</th>
          <th>Experts</th>
          <th>Engram params</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Dense-4B</td>
          <td>4.1B</td>
          <td>3.8B</td>
          <td>none</td>
          <td>none</td>
        </tr>
        <tr>
          <td>MoE-27B</td>
          <td>26.7B</td>
          <td>3.8B</td>
          <td>2 shared + 72 routed, top-6</td>
          <td>none</td>
        </tr>
        <tr>
          <td>Engram-27B</td>
          <td>26.7B</td>
          <td>3.8B</td>
          <td>2 shared + 55 routed, top-6</td>
          <td>5.7B</td>
        </tr>
        <tr>
          <td>Engram-40B</td>
          <td>39.5B</td>
          <td>3.8B</td>
          <td>2 shared + 55 routed, top-6</td>
          <td>18.5B</td>
        </tr>
      </tbody>
    </table>
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-30.png" alt="Benchmark gain summary over MoE baseline" data-lightbox>
    <figcaption>Gains are not limited to factual knowledge; the paper reports strong improvements in reasoning, code, and math too.</figcaption>
  </figure>

  <div class="split split--stack">
    <div>
      <h3>Effective depth</h3>
      <p>Engram helps shallow layers behave like deeper MoE layers because static local reconstruction is handled by lookup.</p>
      <div class="equation-block">
$$
\text{lookup for static facts} \Rightarrow \text{less early reconstruction} \Rightarrow \text{more effective depth}
$$
      </div>
    </div>
    <figure class="asset-figure">
      <img src="/assets/img/posts_images/engram_layers/engram-note-36.png" alt="CKA heatmaps showing Engram effective depth" data-lightbox>
      <figcaption>CKA maps show shallow Engram layers aligning with deeper MoE layers.</figcaption>
    </figure>
  </div>

  <figure class="asset-figure">
    <img src="/assets/img/posts_images/engram_layers/engram-note-37.png" alt="Retained performance when Engram is ablated" data-lightbox>
    <figcaption>Zeroing Engram during inference heavily damages factual knowledge tasks while reading comprehension largely survives.</figcaption>
  </figure>
</section>

<section class="story-section reveal" id="long-context" data-title="Long Context" markdown="1">
## Lookup frees attention

  <p class="lead">The paper argues that once local stereotyped patterns are handled by memory, attention can spend more of its capacity on global context.</p>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Multi-Query NIAH</th>
          <th>Variable Tracking</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MoE-27B, 50k pretrain steps</td>
          <td>84.2</td>
          <td>77.0</td>
        </tr>
        <tr>
          <td>Engram-27B, 46k steps, matched loss</td>
          <td>97.0</td>
          <td>87.2</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p>This does not mean Engram directly performs long-context retrieval. It means early representations are cleaner and attention has less local reconstruction work to do.</p>
</section>

<section class="story-section reveal" id="systems" data-title="Systems" markdown="1">
## Why CPU offload can work

  <p class="lead">MoE routing depends on hidden states. Engram indices depend only on token IDs.</p>

  <div class="split">
    <div class="equation-block">
      $$\text{MoE expert IDs}=r(h_t).$$
      $$\text{Engram IDs}=\phi(x_1,\ldots,x_T).$$
    </div>
    <div class="note-block">
      <p>Because Engram addresses are known before the layer executes, rows can be prefetched from host memory while earlier GPU layers are still computing.</p>
    </div>
  </div>

  <p>The active communication volume scales with retrieved rows, not total table size:</p>
  <div class="equation-block">
    $$\text{bytes per token}\approx |\mathcal{N}|K d_{\text{head}}\cdot\text{bytes-per-element}.$$
  </div>

  <p>The paper reports less than 3 percent throughput penalty when offloading a 100B-parameter Engram layer to host DRAM in their nano-vLLM-based setup.</p>
</section>

<section class="story-section reveal" id="implementation" data-title="Implementation" markdown="1">
## Implementation path

  <p class="lead">The official repository ships a demo that focuses on data flow rather than production kernels.</p>

  <p>The useful way to read the demo is as a call graph. Engram is inserted inside selected Transformer blocks before the ordinary attention and MoE sublayers. The block still receives the full token IDs because the memory address is computed from tokens, not hidden states.</p>

```python
class TransformerBlock(nn.Module):
    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            hidden_states = (
                self.engram(hidden_states=hidden_states, input_ids=input_ids)
                + hidden_states
            )

        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states
```

  <p>So the lookup path is not a sidecar after decoding. It is a residual branch inside the model's forward pass. For configured layers such as 1 and 15 in the demo, the sequence is:</p>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Step</th>
          <th>Code object</th>
          <th>Role</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Compress</td>
          <td><code>CompressedTokenizer</code></td>
          <td>Normalize equivalent token strings and map original token IDs to a smaller canonical ID space.</td>
        </tr>
        <tr>
          <td>Index</td>
          <td><code>NgramHashMapping.hash</code></td>
          <td>Call the n-gram hash routine for every Engram layer and return layer-specific row IDs.</td>
        </tr>
        <tr>
          <td>Gather</td>
          <td><code>MultiHeadEmbedding</code></td>
          <td>Use offsets so many head-specific tables can live inside one contiguous embedding table.</td>
        </tr>
        <tr>
          <td>Fuse</td>
          <td><code>Engram.forward</code></td>
          <td>Project retrieved rows into keys and values, gate with the hidden state, apply short convolution, and return a residual update.</td>
        </tr>
      </tbody>
    </table>
  </div>
### Tokenizer compression


  <p>The demo builds an array mapping each original token ID to a normalized canonical ID. The normalizer applies Unicode normalization, accent stripping, lowercasing, whitespace cleanup, and a fallback for undecodable tokens. This matters because many surface forms should share lookup rows.</p>

```python
old2new = {}
key2new = {}

for tid in range(vocab_size):
    text = tokenizer.decode([tid], skip_special_tokens=False)
    key = token_string_if_undecodable(text) or normalize(text)

    if key not in key2new:
        key2new[key] = len(key2new)

    old2new[tid] = key2new[key]

lookup = np.empty(vocab_size, dtype=np.int64)
for tid in range(vocab_size):
    lookup[tid] = old2new[tid]
```

### Where the n-gram hash is called


  <p>The demo's n-gram hash routine is named <code>_get_ngram_hashes</code>. It is called by <code>NgramHashMapping.hash</code>, which first compresses the input IDs and then computes separate hash IDs for every configured Engram layer.</p>

```python
def hash(self, input_ids):
    input_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}

    for layer_id in self.layer_ids:
        hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
            input_ids,
            layer_id=layer_id,
        )

    return hash_ids_for_all_layers
```

  <p>Inside <code>_get_ngram_hashes</code>, the implementation forms shifted token views so that each position can see its local suffix. For a trigram-capable layer, the arrays are roughly current token, previous token, and token two steps back.</p>

```python
def shift_k(k):
    if k == 0:
        return x
    shifted = np.pad(
        x,
        ((0, 0), (k, 0)),
        mode="constant",
        constant_values=self.pad_id,
    )[:, :T]
    return shifted

base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
```

  <p>The actual indexing function is multiplicative-XOR followed by a per-head modulus. Each layer receives its own random odd multipliers, seeded from the layer ID, so identical n-grams can map differently in different layers.</p>

```python
for n in range(2, self.max_ngram_size + 1):
    n_gram_index = n - 2
    tokens = base_shifts[:n]

    mix = tokens[0] * multipliers[0]
    for k in range(1, n):
        mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

    for j, mod in enumerate(head_vocab_sizes):
        head_hash = mix % int(mod)
        all_hashes.append(head_hash.astype(np.int64, copy=False))

return np.stack(all_hashes, axis=2)
```

  <p>The demo chooses distinct prime table sizes for each head. That is a small but important engineering detail: if all heads used the same modulus, collisions would be correlated; different prime moduli reduce repeated collision structure.</p>

### Gathering rows


  <p>The row IDs returned by hashing have shape <code>[B, T, H]</code>, where <code>H = (N - 1)K</code>. <code>MultiHeadEmbedding</code> stores all head tables in one embedding matrix and adds precomputed offsets so every head indexes its own region.</p>

```python
offsets = [0]
for table_size in list_of_N[:-1]:
    offsets.append(offsets[-1] + table_size)

shifted_input_ids = input_ids + self.offsets
rows = self.embedding(shifted_input_ids)
```

  <p>Then <code>Engram.forward</code> flattens the per-head vectors into a single memory vector per token:</p>

```python
hash_input_ids = torch.from_numpy(
    self.hash_mapping.hash(input_ids)[self.layer_id]
)

embeddings = self.multi_head_embedding(hash_input_ids)
embeddings = embeddings.flatten(start_dim=-2)
```

### Branch-specific gating


  <p>The hidden state decides whether the retrieved memory is relevant. For every hyper-connection branch, Engram projects the memory into a key, compares it with the branch hidden state, and uses the score as a scalar gate on the value projection.</p>

```python
gates = []
for hc_idx in range(backbone_config.hc_mult):
    key = self.key_projs[hc_idx](embeddings)
    normed_key = self.norm1[hc_idx](key)

    query = hidden_states[:, :, hc_idx, :]
    normed_query = self.norm2[hc_idx](query)

    gate = (normed_key * normed_query).sum(dim=-1)
    gate = gate / math.sqrt(backbone_config.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = gate.sigmoid().unsqueeze(-1)
    gates.append(gate)

gates = torch.stack(gates, dim=2)
value = gates * self.value_proj(embeddings).unsqueeze(2)
```

### Short convolution and residual output


  <p>After gating, the demo applies grouped depthwise causal convolution over the branch dimension, then returns the memory update. The Transformer block adds that update to the current hidden state.</p>

```python
output = value + self.short_conv(value)
return output
```

  <div class="note-block">
    <p>A production implementation still needs distributed sparse table sharding, fused row gather, fused key/value projections, asynchronous host-memory prefetch, cache management, and careful handling of CPU-to-GPU transfer overlap. The demo makes the algorithm readable; it is not meant to be the final serving kernel.</p>
  </div>
</section>

<section class="story-section reveal" id="related" data-title="Related Work" markdown="1">
## Embedding scaling around Engram

<p class="lead">Engram fits a broader shift: scale the representation interface, not fixed embedding plumbing.</p>

  <div class="idea-ladder">
    <div class="ladder-step">
      <strong>FFN</strong>
      <span>Implicit key-value memory inside ordinary Transformer blocks.</span>
    </div>
    <div class="ladder-step">
      <strong>PKM</strong>
      <span>Learned nearest-neighbor memory selected from hidden-state queries.</span>
    </div>
    <div class="ladder-step">
      <strong>SCONE</strong>
      <span>Offloaded frequent n-gram embeddings learned by an auxiliary model.</span>
    </div>
    <div class="ladder-step">
      <strong>RAG</strong>
      <span>External document retrieval, editable but slower and less tightly integrated.</span>
    </div>
    <div class="ladder-step">
      <strong>Engram</strong>
      <span>Trainable parametric memory addressed by deterministic hashed local token patterns.</span>
    </div>
  </div>

  <p>The related work falls into three families. Some methods change the tokenizer so each model step carries more text. Some add larger input-side embedding tables while keeping output softmax cost controlled. Others add sparse lookup branches inside the network, closer to Engram.</p>

  <div class="paper-grid">
    <article class="paper-card">
      <span class="paper-meta">arXiv 2502.01637</span>
      <h3><a href="https://arxiv.org/abs/2502.01637">SCONE: Scaling Embedding Layers in Language Models</a></h3>
      <p>SCONE means Scalable, Contextualized, Offloaded, N-gram Embedding. It keeps the original token vocabulary but adds frequent n-gram embeddings. During training, a separate f-gram model learns contextualized vectors; during inference, those vectors are cached as a large off-accelerator lookup table. The important contrast with Engram is training: SCONE avoids instantiating a giant train-time table, while Engram directly trains hashed memory rows inside the model.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">arXiv 2503.13423</span>
      <h3><a href="https://arxiv.org/abs/2503.13423">SuperBPE: Space Travel for Language Models</a></h3>
      <p>SuperBPE changes BPE training rather than the Transformer. It first learns ordinary subwords, then removes the whitespace boundary so later merges can create superword tokens such as common multi-word expressions. This reduces token counts and can improve downstream performance because a model step can represent a more semantic chunk. It is related to Engram because both notice that phrase-level units often behave like atomic knowledge, but SuperBPE bakes them into the tokenizer.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">arXiv 2501.16975</span>
      <h3><a href="https://arxiv.org/abs/2501.16975">Over-Tokenized Transformer / Over-Encoding</a></h3>
      <p>Over-Encoding decouples input and output vocabularies. The input side receives a much larger hierarchical n-gram vocabulary, while the output softmax can remain smaller. The paper reports OE-1.2M and OE-12.8M input vocabularies and argues that input vocabulary scaling gives nearly log-linear loss improvements. It is a direct embedding-scaling result: more input lookup capacity improves the model without paying the full cost of a huge decoder vocabulary.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">arXiv 2412.09871</span>
      <h3><a href="https://arxiv.org/abs/2412.09871">Byte Latent Transformer: Patches Scale Better Than Tokens</a></h3>
      <p>BLT removes fixed-vocabulary tokenization altogether. It groups raw bytes into dynamically sized patches, often using entropy from a small byte model to decide where the next patch should start. The expensive global Transformer runs on patches, while local byte modules encode and decode. BLT is not an n-gram table method, but it is deeply relevant: it treats granularity as a scaling axis and reallocates compute away from predictable byte regions.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">Google docs</span>
      <h3><a href="https://ai.google.dev/gemma/docs/gemma-3n">Layer Embeddings / Gemma 3n Per-Layer Embeddings</a></h3>
      <p>Gemma 3n documents Per-Layer Embedding parameters that are used during execution to enhance each model layer. The public material frames PLE as an edge-device memory technique: keep only the core model hot on accelerator and cache or load layer-specific embedding parameters as needed. I did not find a standalone PLE paper, so the primary reference here is the official Gemma 3n documentation.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">RWKV docs</span>
      <h3><a href="https://wiki.rwkv.com/basic/architecture.html">DeepEmbed in RWKV-V8</a></h3>
      <p>RWKV's DeepEmbed preview adds token-indexed learned vectors inside every FFN layer and uses them as channelwise modulation. The stated deployment motivation is similar to embedding scaling: many parameters can live in RAM, SSD, or memory-mapped storage because each token activates only a tiny slice. I did not find a standalone DeepEmbed paper; the primary source is the RWKV architecture documentation and demo code references.</p>
    </article>

    <article class="paper-card">
      <span class="paper-meta">arXiv 2601.21204</span>
      <h3><a href="https://arxiv.org/abs/2601.21204">LongCat-Flash-Lite: Scaling Embeddings Outperforms Scaling Experts</a></h3>
      <p>LongCat-Flash-Lite is the strongest production-scale neighbor: a 68.5B-parameter sparse MoE model with roughly 3B activated parameters and 31.4B parameters in n-gram embeddings. The paper argues that, in high-sparsity regimes, allocating parameters to n-gram lookup can beat adding more MoE experts. It also stresses the systems side: n-gram cache, optimized embedding lookup, kernel fusion, expert parallelism, and speculative decoding are needed to turn theoretical sparsity into real throughput.</p>
    </article>
  </div>
</section>

<section class="story-section reveal" id="limitations" data-title="Limitations" markdown="1">
## Useful, not magic

  <p class="lead">Engram is not a replacement for reasoning, external retrieval, or careful training.</p>

  <ul>
    <li>It stores parametric knowledge. Changing facts still needs fine-tuning, table editing, or another update mechanism.</li>
    <li>Hash collisions are reduced by multiple heads, not eliminated.</li>
    <li>The optimal MoE/Engram ratio is empirical and may shift with scale, data, tokenizer, and hardware.</li>
    <li>It is strongest for local stereotyped patterns: names, entities, idioms, common code fragments, and frequent phrase structures.</li>
    <li>Independent replication will matter because the systems benefits depend heavily on implementation quality.</li>
  </ul>

  <div class="quote-line">Conditional memory does not replace computation. It stops computation from pretending to be a lookup table.</div>
</section>


<section class="story-section reveal" id="references" data-title="References" markdown="1">
## Sources
  <ol class="references">
    <li><a href="https://arxiv.org/abs/2601.07372">Xin Cheng et al. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.</a></li>
    <li><a href="https://github.com/deepseek-ai/Engram">DeepSeek-AI official Engram repository.</a></li>
    <li><a href="https://www.youtube.com/watch?v=87Q8nf1XHKA">Engram video by Jia-Bin Huang.</a></li>
    <li><a href="https://arxiv.org/abs/2502.01637">Da Yu et al. Scaling Embedding Layers in Language Models.</a></li>
    <li><a href="https://arxiv.org/abs/2503.13423">Alisa Liu et al. SuperBPE: Space Travel for Language Models.</a></li>
    <li><a href="https://arxiv.org/abs/2501.16975">Hongzhi Huang et al. Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling.</a></li>
    <li><a href="https://arxiv.org/abs/2412.09871">Artidoro Pagnoni et al. Byte Latent Transformer: Patches Scale Better Than Tokens.</a></li>
    <li><a href="https://arxiv.org/abs/2601.21204">Hong Liu et al. Scaling Embeddings Outperforms Scaling Experts in Language Models.</a></li>
    <li><a href="https://ai.google.dev/gemma/docs/gemma-3n">Google AI for Developers. Gemma 3n model overview.</a></li>
    <li><a href="https://wiki.rwkv.com/basic/architecture.html">RWKV Wiki. RWKV Architecture History, DeepEmbed section.</a></li>
    <li><a href="https://arxiv.org/abs/2012.14913">Mor Geva et al. Transformer Feed-Forward Layers Are Key-Value Memories.</a></li>
    <li><a href="https://arxiv.org/abs/1907.05242">Guillaume Lample et al. Large Memory Layers with Product Keys.</a></li>
  </ol>
</section>

<div class="lightbox" id="engram-lightbox" role="dialog" aria-modal="true" aria-label="Image preview">
    <button type="button" id="engram-lightbox-close" aria-label="Close image preview">x</button>
    <img alt="">
  </div>

</div>

<script defer src="{{ '/assets/js/engram-layers.js' | relative_url }}"></script>

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
