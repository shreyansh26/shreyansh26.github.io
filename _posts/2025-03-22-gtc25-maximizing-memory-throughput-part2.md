---
layout: post
title: "Notes from GTC'25: CUDA Techniques to Maximize Memory Bandwidth and Hide Latency - Part 2"
date: 2025-03-23
author: "Shreyansh Singh"
description: "Second part of my notes from the talk on maximizing memory bandwidth at NVIDIA GTC 2025."
tags: cuda mlsys
categories: ["MLSys"]
giscus_comments: true
related_posts: false
permalink: "post/2025-03-23_gtc25-maximize-memory-bandwidth-part-2/"
featured: false
toc:
  sidebar: left
pretty_table: true
---

---

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/cover.png" description="" %}

You can watch the talk here - [link](https://register.nvidia.com/flow/nvidia/gtcs25/vap/page/vsessioncatalog/session/1727709012449001X6PZ)

Part 1 of the talk focused on maximizing memory throughput. The notes can be found [here](/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1/).




These are the notes for the second part of the talk which focused on memory models and hiding latency.

---

# Memory Model
---

Memory model is a way to understand how memory is accessed and used in a program. It is a contract between the user and the compiler/hardware/language.


## Single-threaded

Standard memory model.    
Stores are visible to the thread that stored them.     
Loads and stores to the same address remain in order - they cannot overtake each other in the memory subsystem.


Important concept - **same-address ordering**.   
Same-address ordering does not hold always. E.g. when using constant caches. The constant caches have a link to the L2 cache but not to the L1 cache. Hence, these caches are not coherent. 

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/constant_cache.png" description="Source: Slides from the talk" %}

So, constant cached values can cause issues. You can do the store - which would go through L1 to the L2 and update it. However, during load, the constant cache is used and it returns the old value.

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/constant_cache_2.png" description="Source: Slides from the talk" %}


## Memory Ordering 

Memory order specifies how memory accesses, including regular (non-atomic) accesses, are to be ordered around an atomic operation.

Four important memory orders in multi-threaded memory model:

1. Sequentially consistent
2. Acquire
3. Release
4. Relaxed


{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/memory_ordering.png" description="Source: Slides from the talk" %}

## Multi-threaded

### CUDA C++ Scope

* Thread - `cuda::thread_scope_thread` - Only local thread can observe this thread's loads and stores
* Thread Block - `cuda::thread_scope_block` - Only threads in the same block can observe this thread's loads and stores
* GPU Device - `cuda::thread_scope_device` - All threads in the GPU can observe this thread's loads and stores
* System - `cuda::thread_scope_system` - All threads in the system (CPU, other GPUs, other nodes) can observe this thread's loads and stores

### CUDA PTX Scope

* Thread Block - `.cta` - Only threads in the same block can observe this thread's loads and stores
* GPU Device - `.gpu` - All threads in the GPU can observe this thread's loads and stores
* System - `.sys` - All threads in the system (CPU, other GPUs, other nodes) can observe this thread's loads and stores

### Thread scope - Block

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/ts_block.png" description="Source: Slides from the talk" %}

Threads in the same block execute on same SM.    
Data only has to be consistent in L1. All threads in the block see the same data.    
Release and acquire semantics are quite fast. Because data does not have to be flushed very far. We don't have to invalidate many caches.


### Thread scope - Cluster

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/ts_cluster.png" description="Source: Slides from the talk" %}

Many threads working across multiple SMs working together.    
Data has to go through L2.    
**In release, we would have to flush to L2 and in acquire, we would have to make sure that L1 is invalidated.**

### Thread scope - GPU

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/ts_system.png" description="Source: Slides from the talk" %}

Many threads working across multiple SMs of a GPU working together.    
Synchronization is as difficult as cluster.    
**In release, we would have to flush to L2 and in acquire, we would have to make sure that L1 is invalidated.**

### Thread scope - System

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/ts_system.png" description="Source: Slides from the talk" %}

Many threads working across multiple GPUs working together.    
**In release, we would have to make sure that all the stores made it to the relevant caches across GPUs and nodes.**    
**Acquire is still cheap, all L1s need to be invalidated.**


### Data transfer examples

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/relaxed_1.png" description="Source: Slides from the talk" %}

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/relaxed_2.png" description="Source: Slides from the talk" %}

Using thread scope `block` when working with same block.

Using thread scope `device` when working with different thread blocks.

But, for a not so relaxed example, where there are two variables we need to work with, simply using `device` scope is not enough.

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/not_relaxed_3.png" description="Source: Slides from the talk" %}

**We need to use a release-acquire pattern.**

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/not_relaxed_2.png" description="Source: Slides from the talk" %}

### Relaxed vs Release-Acquire

**Relaxed**     
* Faster - A single store or load to or from the cache at the point of coherency.     
* Does not provide ordering w.r.t other reads and writes.    
* Useful if two threads want to exchange one value.    

**Release-Acquire**     
* Slower - Requires flushing to point of coherency and / or invalidating caches.     
* Provides ordering w.r.t other reads and writes.
* Useful if multiple threads want to exchange multiple values.


For larger chunks of data, release-acquire is preferred.


## Async thread - Ampere

**PTX instruction `st.async`**

* Stores a value to Distributed Shared Memory of another block in the cluster/
* Once the store is complete, it updates a shared memory barrier in the shared memory of the other block.

**However, a subsequent load or store can race ahead, violating the same-address ordering.**

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/store_async.png" description="Source: Slides from the talk" %}


## Async proxy - Hopper

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_proxy.png" description="Source: Slides from the talk" %}

Proxies represent situations where there are multiple different paths from a single thread to a single physical memory location, with no coherence across paths.

**Generic Proxy** - All normal loads and stores go through the generic proxy.    
**Async Proxy** - A different path that is used by TMA units, tensor cores and several other instructions.

Between a generic proxy load/store and an async proxy load/store, there is **no same-address ordering**. Even less than earlier (async threads).

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_proxy_2.png" description="Source: Slides from the talk" %}

The normal store can overtake the TMA load.

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_proxy_code_1.png" description="Source: Slides from the talk" %}

Here, the generic proxy store to shared memory will be most likely overtaken by async proxy load from shared memory.    
This will store stale values to global memory.

### Async Proxy Fence

**Solution is to use an async proxy fence** -     
{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_proxy_2_fence.png" description="Source: Slides from the talk" %}

The fence traces the store to shared memory, and makes sure that the store is complete. Once it is complete, the fence comes back, notifies the thread and only then will the TMA load be allowed to proceed.

**Implicit Fencing**   

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_implicit_fence.png" description="Source: Slides from the talk" %}

Here we start waiting on the barrier after the copy async bulk is issued. Barrier waiting request goes to to the shared memory until the load is finished. Only when all the required updates to the shared memory are done (stores), the barrier is updated.


## Async thread and Async proxy instructions

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/async_thread_proxy_instructions.png" description="Source: Slides from the talk" %}

- `st.async` and `red.async` are in Hopper but still async thread only    
- `cp.async` - Ampere     
- If you have a normal load and store before - obeys same-address ordering   
- But normal load and store after - it will not obey    
- Async proxy fence is still needed to ensure correct ordering   

# Low-Latency Cluster Synchronization
---

**Key points**

* The point of coherency for a cluster is L2 - thread blocks can be in different SMs
* Any release-acquire pattern with cluster scope requires a round trip to L2 which is expensive
* To reduce latency - avoid these round trips


## Thread synchronization in a cluster

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/low_latency_1.png" description="Source: Slides from the talk" %}

Arrive has to be executed by all threads in a cluster but wait doesn't need to be.    
The arrive can have different memory model orderings.     
* Release - Requires flushing to L2 but gives synchronization of data  
* Relaxed - Only execution synchronization but no data synchronization

### Barrier Initialization - Simple way

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/low_latency_2.png" description="Source: Slides from the talk" %}

Initializing a shared memory barrier and making it visible to all threads in the cluster.    
A cluster sync is done to make the barrier visible to all threads.    
Nothing needs to be flushed to L2 => this is more expensive than it has to be.

### Barrier Initialization - Fast way

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/low_latency_3.png" description="Source: Slides from the talk" %}

Instead of `cluster::sync`, we use a **relaxed arrive** which does not flush anything to L2, but ensures execution synchronization.    
But to ensure correctness, we do a release fence of just the mbarrier init.     
Additionally there is a release-acquire pattern and they have to be scope clusters.   
`fence_mbarrier_init`, `arrive` and `wait` are all fairly cheap.

For kernels which are short, this type of optimization can help a lot. 
However, for long kernels, this won't help much.


## Data communication in a cluster

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/data_comm_1.png" description="Source: Slides from the talk" %}

{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/data_comm_2.png" description="Source: Slides from the talk" %}

* Arrival should be relaxed and scope_cluster. If it were a release, then it would have a flush to L2.
* Wait from other cluster should be acquire (in a loop) so that it can form a release-acquire pattern with `st_async`. As `st_async` just releases the 4 bytes it has stored and that's what we acquire in the `mbarrier_try_wait` which is also a scope cluster and you wait on the local barrier which is cheap.
* FInally, we need to make sure the other thread in the cluster got our value before we send another. This can be relaxed as we just need to ensure execution synchronization.


{% include image.liquid url="/assets/img/posts_images/gtc_memory_bandwidth/data_comm_3.png" description="Source: Slides from the talk" %}


But again, this helps only for short kernels. For long kernels, this won't help much. We can fo go for the simple code.


---

Hope this was helpful!

Notes for part 1 on maximizing memory bandwidth can be found [here](/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1/).

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
