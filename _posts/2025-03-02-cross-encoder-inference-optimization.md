---
layout: post
title: "Faster Cross-Encoder Inference: Unleashing torch.compile for speed"
date: 2025-03-02
author: "Shreyansh Singh"
description: "A quick writeup on accelerating a Jina Cross-Encoder using torch.compile"
thumbnail: /assets/img/posts_images/jina_torch_compile/image.png
tags: inference-optimization efficiency mlsys
categories: ["MLSys"]
giscus_comments: true
related_posts: false
permalink: "post/2025-03-02_cross-encoder-inference-torch-compile/"
featured: false
toc:
  sidebar: left
pretty_table: true
---

---
<!-- {% include image.liquid url="/assets/img/posts_images/jina_torch_compile/image.png" description="" %} -->

**Code** - [Github repo](https://github.com/shreyansh26/Accelerating-Cross-Encoder-Inference) 

---

When deploying large ML models in production, optimization becomes crucial for maintaining both performance and cost-effectiveness. In this post, I'll share my experience optimizing the inference of a cross-encoder (reranker) model using torch.compile and a custom batching strategy. We'll explore how combining torch.compile with careful input handling can significantly improve inference speed.

## The Setup: Cross-Encoder (Neural Reranker) Model

For this experiment, I used the Jina reranker model (`jinaai/jina-reranker-v2-base-multilingual`), which is designed for scoring the similarity between text pairs. Such type of models are used in a lot of applications like information retrieval, semantic search, recommender systems, etc. The model takes pairs of text as input and outputs similarity scores. Here's what makes this use case interesting:

1. Variable input lengths (here we assume each text contains 2-15 sentences)
2. Batch processing

While running inference at scale, even the smallest of optimizations can make a huge difference.

> **Note** - The optimizations and the techniques described in this post are not silver bullets for model inference optimization. Models may have different architectures and inference algorithms which can completely change how they can be optimized. However, the general principles described in this post would definitely hold.

## Understanding torch.compile and the Inductor Backend

PyTorch 2.0 (and onwards) comes with `torch.compile`. Although there are a <ins>[bunch](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)</ins> <ins>[of resources](https://pytorch.org/docs/stable/torch.compiler.html)</ins> to understand how it works, in short, torch.compile JIT (just in time) compiles your model and makes your Pytorch code run faster by using optimizations like operation fusion, graph capture, custom triton kernels, etc.

There are various choices of backends for torch.compile. I used the `inductor` backend in my experiments as it is also the most advanced Pytorch-native backend at the moment. Let's understand how it works:

### How Inductor Works

At its core, Inductor optimizes your PyTorch model through several key steps:

1. **Graph Capture**: (TorchDynamo) When you first run your compiled model, Inductor captures the computational graph of your operations.
2. **Operation Fusion**: (TorchDynamo) Multiple operations are combined where possible to reduce memory transfers.
3. **Hardware-Specific Optimization**: (TorchInductor) The backend generates optimized kernels specifically for your GPU.

Here's how we set up our compiled model:

```python
model_compile = DynamicCrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    device="cuda",
    config_args={"use_flash_attn": False}
)

model_compile.model.forward = torch.compile(
    model_compile.model.forward, 
    backend="inductor",
    mode="max-autotune",
    dynamic=True
)
```

The key parameters we're using:
- `backend="inductor"`
- `mode="max-autotune"`: Enables aggressive optimization
- `dynamic=True`: Handles our variable input sizes

If you're curious as to why we set `use_flash_attn = False`, I discuss it in a [later section](#but-why-set-use_flash_attn--false) after describing the optimizations and results.

## Smart Batching with Length Buckets

Having static shapes is ideal for torch.compile. If there are a variations in the sizes of the variables, then TorchDynamo will have to trace all such variations. Keeping the number of size variations minimum while still giving enough flexibility will be our goal.

One way to do it is, depending on the lengths of our sentences in the dataset, we can decide to keep a static sequence length for the model by specifying the `max_length` parameter while initializing the cross encoder. This length could be the maximum sequence length or a high enough length that covers most sequences (the ones longer would be truncated), The main issue with this approach is that for sequence lengths much smaller than the fixed length (which could be a significant portion of the dataset), we would be wasting a lot of compute on the padding tokens. 

In our experiment, we tackle this by creating sequence-length buckets for padding. Instead of padding all sequences to the maximum length in the batch, we pad to the nearest multiple of 16. Obviously this is not perfect, but in my experience of using cross encoders, I find that a max-length of 512 is enough for most practical use cases where a reranker works effectively. In case we do need longer sequence lengths, I would recommend increasing the bucket size from 16 to 32 or even higher based on the maximum length we need.

Here's our implementation:

```python
BUCKETS = list(range(16, 528, 16))

def smart_batching_collate_text_only(self, batch):
    texts = [[text.strip() for text in field] for field in zip(*batch)]
    tokenized = self.tokenizer(
        *texts,
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=self.max_length
    )
    tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

    # Pad to nearest bucket
    cur_length = tokenized["input_ids"].size(1)
    bucket_length = next((b for b in BUCKETS if b >= cur_length), cur_length)
    if bucket_length > cur_length:
        diff = bucket_length - cur_length
        for key, val in tokenized.items():
            pad_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
            tokenized[key] = torch.nn.functional.pad(val, (0, diff), value=pad_value)
    return tokenized
```

This bucketing approach helps in two ways:
1. Reduces wasted computation on padding tokens
2. Helps the compiled model optimize for specific input sizes

## Input Sorting for Better Efficiency

To further improve performance, we implemented input sorting. This groups similarly-sized inputs together, making our bucket-based padding more effective:

```python
if on_sorted_inputs:
    # Sort by max length of each pair
    lengths = [(len(model.tokenizer.encode(p[0])) + len(model.tokenizer.encode(p[1])), i) 
                for i, p in enumerate(sentence_pairs)]
    sorted_indices = [i for _, i in sorted(lengths, reverse=True)]
    sentence_pairs_sorted = [sentence_pairs[i] for i in sorted_indices]
```

## But why set `use_flash_attn = False`?
While Flash Attention is generally faster than vanilla attention implementations, there are several technical reasons why I disabled it when using torch.compile for this particular optimization:

<!-- ### 1. FlashAttention is already a compiled CUDA kernel

Flash Attention operates through highly optimized CUDA kernels that are already compiled for performance:

```python
# In FlashSelfAttention, from mha.py - showing Flash Attention's compiled nature
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/mha.py
def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
    # ...
    if unpadded:
        # Using pre-compiled CUDA kernel
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            # ...
        )
    else:
        # Using pre-compiled CUDA kernel
        return flash_attn_qkvpacked_func(
            qkv,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            # ...
        )
```

Applying torch.compile on top of an already compiled kernel was giving errors.  -->

### 1. Variable sequence lengths complicate tracing

Flash Attention operates through highly optimized CUDA kernels that are already compiled for performance:

```python
# In FlashSelfAttention, from mha.py - showing Flash Attention's compiled nature
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/mha.py
def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
    # ...
    if unpadded:
        # Using pre-compiled CUDA kernel
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            # ...
        )
    else:
        # Using pre-compiled CUDA kernel
        return flash_attn_qkvpacked_func(
            qkv,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            # ...
        )
```

The goal of our bucketing strategy was to have a consistent and a small number of tensor shapes for efficient compilation. However, when using `flash_attn_varlen_qkvpacked_func` the unpadding mechanism in the original Flash Attention implementation leads to dynamic tensor shapes that are difficult to trace:

```python
# From xlm_padding.py, and called in modeling_xlm_roberta.py
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/xlm_padding.py
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/modeling_xlm_roberta.py
def unpad_input(hidden_states, attention_mask):
    """
    Convert padded sequences to packed format for efficiency
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
```

This operation creates tensors with sizes dependent on the input data, which conflicts with our bucketing strategy where we want to pad to the nearest multiple of 16. This dynamic sizing makes it challenging for torch.compile to effectively trace and optimize the model.

### 2. Attention mask handling limitations

The alternative in the code was to use `flash_attn_qkvpacked_func` which doesn't offer the flexibility we needed for custom attention masking as it expects qkv matrices together and internally handles causal or non-causal masking.

```python
# In FlashSelfAttention, from mha.py
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/mha.py
return flash_attn_qkvpacked_func(
    qkv,
    self.drop.p if self.training else 0.0,
    softmax_scale=self.softmax_scale,
    causal=causal,
    alibi_slopes=None,
    window_size=self.window_size,
    deterministic=self.deterministic,
)
```

While there is a regular `flash_attn_func` that might have worked, integrating our attention mask to mask padding tokens was not straightforward.

## The Hybrid Approach

```python
# In SelfAttention, from mha.py
# https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/blob/main/mha.py
def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output
```

The standard PyTorch attention implementation (without Flash Attention) allowed torch.compile to see through the entire computation graph and apply optimizations like operation fusion and kernel generation tailored to our specific inputs.

By disabling Flash Attention but keeping our bucketing and sorting strategies, we created a middle ground that allowed torch.compile to shine. This approach:

1. Gives torch.compile more visibility into the computation graph
2. Maintains consistent tensor shapes through our bucketing strategy
3. Allows handling of attention mask quite simply

The results showed this hybrid approach outperformed the baseline (Flash Attention) implementation. Even without input sorting, the torch.compile version was faster or about the same as the baseline (Flash Attention) + input sorting version.
<!-- 
Sometimes, combining the right optimizations means choosing which ones work well together rather than applying all available optimizations at once. -->

## Benchmarking

Our benchmarking system provides reliable measurements through proper warm-up and synchronization:

```python
def benchmark(model, print_scores=False, num_runs=10, trace=None, seed=100, on_sorted_inputs=False):  
    sentence_pairs_warmup = load_and_sample_sentences(num_pairs=512, base_seed=seed)
    sentence_pairs = load_and_sample_sentences(num_pairs=1024, base_seed=2*seed)

    with torch.inference_mode():
        # Warmup
        print("Warming up...")
        for i in range(10):
            sentence_pairs_warmup = load_and_sample_sentences(num_pairs=2048, base_seed=seed + i)
            _ = inference(model, sentence_pairs_warmup)

        # Multiple benchmark runs
        print("Benchmarking...")
        times = []

        for i in range(num_runs):
            sentence_pairs = load_and_sample_sentences(num_pairs=1024, base_seed=2*seed + i)
            
            if on_sorted_inputs:
                # Apply sorting if enabled
                lengths = [(max(len(model.tokenizer.encode(p[0])), len(model.tokenizer.encode(p[1]))), i) 
                          for i, p in enumerate(sentence_pairs)]
                sorted_indices = [i for _, i in sorted(lengths, reverse=True)]
                sentence_pairs_sorted = [sentence_pairs[i] for i in sorted_indices]
            else:
                sentence_pairs_sorted = sentence_pairs
                sorted_indices = None

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            scores = inference(model, sentence_pairs_sorted)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
```

## Results

Here are our key findings:

| Configuration | Mean Time (s) | Std Dev (s) |
|:--------------|:--------------|:-------------|
| Baseline (Without Flash Attention) + Unsorted Inputs | 0.3566 | 0.0101 |
| Baseline (Without Flash Attention) + Sorted Inputs | 0.3245 | 0.0623 |
| Baseline (Flash Attention) + Unsorted Inputs | 0.2961 | 0.0089 |
| Baseline (Flash Attention) + Sorted Inputs | 0.2658 | 0.0119 |
| torch.compile + Unsorted Inputs | 0.2595 | 0.0077 |
| **torch.compile + Sorted Inputs** | **0.2089** | **0.0196** |

<br>

### Key observations:
1. torch.compile provides upto ~1.3x speedup over the base model
2. Input sorting improves performance by upto 1.25x
3. The combination of torch.compile and sorted inputs gives us the best performance

## Best Practices and Learnings

Through this optimization process, we discovered several important practices:

1. **Proper Warm-up**: Always run warm-up iterations before benchmarking to ensure the compiled model has optimized its execution path and seen all variations of sizes so that there are no recompilations during the actual benchmarking.

2. **Accurate Timing**: Use proper CUDA synchronization for accurate measurements:
```python
torch.cuda.synchronize()
start_time = time.perf_counter()
# ... inference ...
torch.cuda.synchronize()
end_time = time.perf_counter()
```

## Conclusion

By combining torch.compile with smart batching and input sorting, we achieved a significant speedup in our neural reranker inference. The key takeaway is that optimization often requires a multi-faceted approach - compiler optimizations alone might not give you the best results, but when combined with domain-specific optimizations like bucket-based padding and input sorting, the improvements can be substantial.

For those looking to optimize their own models, I recommend:
1. Start with torch.compile as it's relatively easy to implement
2. Add bucket-based padding if you have variable-length inputs
3. Consider input sorting if your batch sizes are large enough to benefit from it
4. Always measure and profile your specific use case, as the benefits of each optimization can vary depending on your model and data

The complete code for this optimization project is available in the snippets above. Feel free to adapt these techniques for your own use case!

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
