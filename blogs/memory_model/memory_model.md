---
layout: default
---

# GPU Memory Consistency Model

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

*Disclaimer2: The content of this blog may be incorrect. It just represents my own understanding of the memory consistency model. And I am being intentionally wrong sometimes (on TMEM in particular) for better clarity and simplicity. The [ptx doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model) is the ground truth.*

## 0. Introduction

I wish I never think about the memory model.
And for a while this does seem to be the case. 
You load your data from GMEM to SMEM and then you do a [`__syncthreads()`]((https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)).
Then the data is visible to all threads in the CTA.
Life is good.

However, as the GPU architecture becomes more complex, more co-processors are introduced (e.g. TMA/Tensor Cores), and more sophisticated kernel programming techniques are introduced (e.g. warp specialization, kernel fusion).
Suddenly, you have to think about the memory model in order to just write a *functionally correct* kernel.

Even though the [ptx doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model) describes the memory consistency model in a formal way, the lack of examples makes it hard to understand and use in practice.
In this blog, I attempt to provide concrete examples on how to do synchronization that adheres to the memory consistency model on many common use cases in kernel programming.

## 1. Background

### 1.1. What is Memory Consistency Model?

The memory consistency model roughly describes when the memory operation performed by one thread is visible to other threads.

This is important because many times we want to establish a producer-consumer relationship in a kernel, where the producer threads produces some data, and the consumer threads (could be a different set of threads) consume the data.
Then how to ensure the producer's output is visible to the consumer is a memory consistency problem.

### 1.2. Execution Order Does Not Guarantee Memory Order

Instruction execution of two threads can be ordered, meaning with some synchronization mechanism (e.g. `mbarrier.arrive+try_wait`) we can ensure thread B will executes `ld.shared` after thread A issues `st.shared`.

```python
thread A:
    st.shared addr, val
    mbarrier.arrive.relaxed
thread B:
    mbarrier.try_wait.relaxed
    ld.shared reg, addr
```

But this execution order does not guarantee any memory order at all!
(I intentionally use the `relaxed` semantic here just to order execution not memory).
Thread B may not see the newly written value in addr by `ld.shared`.
A totally valid hardware implementation of this would be while the stored value `val` is still in some LSU (load store unit), the load from thread B will read out the old value from memory.
In order for this producer-consumer relationship to be correct, we need to use proper fences to guarantee the memory order on top of the execution order, i.e. ensure thread A's memory operation is visible to thread B.


### 1.3. [Scope](https://docs.nvidia.com/cuda/parallel-thread-execution/#scope)

Since memory model cares about the memory order between a set of threads, as you can imagine, the scope where these threads are all belong to may be different.
Sometimes you just want threads within a CTA to be synchronized, sometimes you want threads within a GPU to be synchronized.
We call this the `scope` and this often manifests as a `.xxx` suffix in the ptx instruction.

`.cta` means we care about the memory order between threads within a CTA, `.cluster` means we care about the memory order between threads within a cluster, `.gpu` means we care about the memory order between threads within a GPU.
Note that there isn't a `.grid` scope.

The larger the scope, the more costly the synchronization is.

The figure below also illustrates this hierarchy.
The curly boxes represent different scopes.

![scope](./figures/scope.png)


### 1.4. [State Space](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-state-spaces)

An orthogonal thing to `scope` is the `state space` of the memory operation, meaning we care about the memory operation on which storage (SMEM/GMEM/DSMEM/etc.) idiom.
The figure above also shows the state space hierarchy.
The square boxes represent different state spaces.
In ptx, this often manifests as a `.xxx::yyy` suffix in the ptx instruction.

`.shared::cta` means we care about the memory operation order on SMEM (shared memory), `shared::cluster` means we care about the memory operation order on DSMEM (distributed shared memory), `.global` means we care about the memory operation order on GMEM (global memory).

The larger/farther away the state space, the more costly the synchronization is.


### 1.5. [Generic vs Async Proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#proxies)

So far the thread we've been talking about is thread running on the CUDA core.
Modern Nvidia GPUs introduces many asynchronous co-processors (e.g. TMA/Tensor Cores) that can be viewed as an asynchronous thread other than the normal threads running on the CUDA core.
We call threads running on the CUDA cores the `generic proxy` and threads running on the asynchronous co-processors (TMA/Tensor Cores, etc.) the `async proxy`.

The memory order between the generic proxy and the async proxy needs to be specially handled.
In other words, memory order operations that guarantee memory order within the generic proxy doesn't not by default extend to the async proxy.

The figure below shows the generic proxy and async proxy on an SM.

![proxy](./figures/proxy.png)

The async proxy includes the TMA unit, the Hopper ([`wgmma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions))/Blackwell ([`tcgen05`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)) Tensor Cores, the [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) unit.
The async proxy primarily communicates with the generic proxy through different storage idiom (SMEM/TMEM/GMEM). 
Unfortunately, each unit has a different way to synchronize the memory order.
So in [Sec. 2](#2-common-memory-ordering-patterns) we will cover all the common patterns.

### 1.6. Synchronization Mechanisms

There are several ways to enforce execution order and memory order on Nvidia GPUs.
The table below lists the most common ones which are the ones we will cover in this blog.

| Mechanism | Example ptx | Enforces Execution Order | Enforces Memory Order | Scope | State Space | Proxy |
|-----------|-------------|--------------------------|------------------------|-------|-------------|-------|
| [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions), Named Barriers | [`bar.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar) | Yes | Yes | .cta | SMEM/GMEM | Generic |
| [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) | [`mbarrier.arrive`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive), [`mbarrier.try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) | Yes | Optional | .cta/.cluster | SMEM/DSMEM/GMEM | Generic/Async |
| Generic fence | [`fence.cta`, `fence.cluster`, `fence.gpu`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) | No | Yes | .cta/.cluster/.gpu | SMEM/DSMEM/GMEM | Generic |
| Async fence | [`fence.proxy.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) | No | Yes | .cta/.cluster | SMEM/DSMEM/GMEM | Generic->Async |
| `tcgen05` fence | [`tcgen05.wait::ld`, `tcgen05.wait::st`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait) | No | Yes | .cta | TMEM | Generic<->Async |

### 1.7. [Memory Ordering Semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-acquire-patterns)

Often we append a synchronization instruction (e.g. [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)) with a memory ordering semantic to enforce the memory order on top of the execution order.
The few notable ones are:

- `release`: The release pattern makes prior memory operations (before `release`) from the current thread visible to other threads.
- `acquire`: The acquire pattern makes prior (`before acquire`) memory operations from other threads visible to the current thread.
- `acq_rel`: The acq_rel pattern makes prior (`before acq_rel`) memory operations from other threads visible to the current thread and makes the current thread's prior (`before acq_rel`) memory operations visible to other threads.
- `relaxed`: No memory order is enforced.

Note that the memory ordering is kinda meaningless by itself without a proper execution order synchronization.
Because then it's hard to define the term `prior`.
It's totally possible thread B executes `acquire` way earlier than thread A executes `release` (in terms of time), there is no way thread A's value is visible to thread B.
It has to be accompanied by a proper execution order synchronization.

## 2. Common Memory Ordering Patterns

In this section, we will enumerate all the common memory ordering patterns occurred in kernel programming.
We focus on the **producer-consumer** pattern as it's the most common technique and the usual place where the memory model is needed.
Based on the proxy of the producer and consumer threads, we can categorize the memory ordering patterns into 4 categories:

| producer \ consumer | Generic Proxy | Async Proxy |
|-----------|---------------|--------------|
| Generic Proxy | [Intra-CTA (Sec. 2.1.1)](#211-intra-cta-producer-consumer)<br>[Intra-Cluster (Sec. 2.1.2)](#212-intra-cluster-producer-consumer)<br>[Intra-GPU (Sec. 2.1.3)](#213-intra-gpu-producer-consumer) | [CUDA Core -> TMA (Sec. 2.3.1)](#231-cuda-core-tma)<br>[CUDA Core -> tcgen05 (Sec. 2.3.2)](#232-cuda-core-tcgen05) |
| Async Proxy | [TMA -> CUDA Core (Sec. 2.2.1)](#221-tma-cuda-core)<br>[tcgen05 -> CUDA Core (Sec. 2.2.2)](#222-tcgen05-cuda-core) | [TMA -> tcgen05 (Sec. 2.4.1)](#241-tma-tcgen05) |

### 2.1. Generic Proxy -> Generic Proxy

| State Space \ Scope | .cta | .cluster | .gpu |
|---------------------|------|----------|------|
| SMEM/DSMEM | [`fence.cta`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) / [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) / [`__threadfence_block()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) / [`mbarrier.arrive+try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) | [`fence.cluster` / `fence.gpu`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) / [`__threadfence()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) / [`barrier.cluster.arrive+wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster) / `mbarrier.arrive+try_wait` | N/A |
| GMEM | `fence.cta` / `__syncthreads()` / `__threadfence_block()`/ `mbarrier.arrive+try_wait` | `fence.cluster` / `fence.gpu` / `__threadfence()` / `barrier.cluster.arrive+wait` / `mbarrier.arrive+try_wait` | `fence.gpu` / `__threadfence()` |
| Effective SASS | `MEMBAR.CTA` | `MEMBAR.GPU` | `MEMBAR.GPU` |

#### 2.1.1. Intra-CTA Producer-Consumer

membar.cta / syncthreads()

#### 2.1.2. Intra-Cluster Producer-Consumer

membar.gpu / cluster_sync() / st.async

#### 2.1.3. Intra-GPU Producer-Consumer

membar.gpu

### 2.2. Async Proxy -> Generic Proxy

#### 2.2.1. TMA -> CUDA Core

#### 2.2.2. tcgen05 -> CUDA Core

### 2.3. Generic Proxy -> Async Proxy

#### 2.3.1. CUDA Core -> tcgen05

### 2.4. Async Proxy -> Async Proxy

#### 2.4.1. TMA -> tcgen05

## Summary

I still wish I never think about the memory model.


