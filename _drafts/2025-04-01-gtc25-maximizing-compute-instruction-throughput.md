---
layout: post
title: "Notes from GTC'25: CUDA Techniques to Maximize Compute and Instruction Throughput"
date: 2025-04-01
author: "Shreyansh Singh"
description: "My notes from the talk on maximizing compute and instruction throughput at NVIDIA GTC 2025."
tags: cuda mlsys
categories: ["MLSys"]
giscus_comments: true
related_posts: false
permalink: "post/2025-04-01_gtc25-maximize-compute-instruction-throughput/"
featured: false
toc:
  sidebar: left
pretty_table: true
---

---

{% include image.liquid url="/assets/img/posts_images/gtc_compute_throughput/cover.png" description="" %}

You can watch the talk here - [link](https://register.nvidia.com/flow/nvidia/gtcs25/vap/page/vsessioncatalog/session/1727709629316001myds)

---

# GPU Basics

{% include image.liquid url="/assets/img/posts_images/gtc_compute_throughput/hopper_1.png" description="Source: Slides from the talk" %}


PCIe is used for communication between the CPU and the GPU.      
NVLink is used for communication between the GPUs.    

Each SM of Hopper architecture has -     
* 4 sub-partitions     
* 128 FP32 units     
* 64 FP64 units     
* 64 INT32 units     
* 4 mixed-precision Tensor Cores     
* 16 special function units     
* 4 warp schedulers     
* 32 load/store units    
* 64K 32-bit registers      
* 256KB unified L1 cache and shared memory (however on checking CUDA device properties, I found that the shared memory is 228 KB)     
* Tensor Memory Accelerator (TMA)

{% include image.liquid url="/assets/img/posts_images/gtc_compute_throughput/hopper_sm_small.png" description="Source: Slides from the talk" %}

In Hopper, in addition to blocks and grids, there is an optional level in the thread hierarchy called - **Thread Block Clusters**. Thread blocks in a cluster are guaranteed to be concurrently scheduled and enable efficient cooperation and data sharing for threads across multiple SMs.

GPUs follow SIMT (Single Instruction, Multiple Threads) execution.     
* Each thread has its own program counter.     
* SIMT = SIMD + Program Counters    

Since Volta, each thread has its own program counter.

## Warp Divergence

{% include image.liquid url="/assets/img/posts_images/gtc_compute_throughput/simt.png" description="Source: Slides from the talk" %}

Metrics to look at (in NCU) to detect divergence:        
* Average Predicated-On Threads Executed   
    * At this instruction, how converged is my warp on average?    
* Divergent Branches    
    * Number of times branch target differed    
* (Soon) Derivative Average Predicated-On Threads Executed    
    * E.g. if in a piece of code - At to level, it diverges slightly (but only once) and stays diverged; and in lower level code, there is frequent divergence and re-convergence although less severely. Then,     
        * Derivative metric has higher value for the top level divergence.     
        * Divergent Branch metric has higher value for the lower level code.    

---

Hope this was helpful!

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
