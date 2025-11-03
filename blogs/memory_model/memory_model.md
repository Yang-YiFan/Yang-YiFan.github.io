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

Unfortunately we need to define some terminologies first.
Please bear with me as I guarantee you they are actually useful.

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
| [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions), [Named Barriers](https://github.com/NVIDIA/cutlass/blob/8afb19d9047afc26816a046059afe66763e68aa5/include/cutlass/arch/barrier.h#L159) | [`bar.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar) | Yes | Yes | .cta | SMEM/GMEM | Generic |
| [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) | [`mbarrier.arrive`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive), [`mbarrier.try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) | Yes | Optional | .cta/.cluster | SMEM/DSMEM | Generic/Async |
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

In this blog for the Tensor Core part we focus on the Blackwell tensor core (`tcgen05`) but Hopper Tensor Core should be spiritually the same.

### 2.1. Generic Proxy -> Generic Proxy

Traditional CUDA SIMT programming (no TMA, no Tensor Core) would just involve threads within the generic proxy synchronizing with each other in a producer-consumer relationship.
The table below lists some of the ways to enforce memory order between threads within the generic proxy.

| State Space \ Scope | .cta | .cluster | .gpu |
|---------------------|------|----------|------|
| SMEM/DSMEM | [`fence.cta`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) / [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) / [`__threadfence_block()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) / [`mbarrier.arrive+try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) | [`fence.cluster` / `fence.gpu`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar) / [`__threadfence()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions) / [`barrier.cluster.arrive+wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster) / `mbarrier.arrive+try_wait` | N/A |
| GMEM | `__syncthreads()` / `__threadfence_block()` / `mbarrier.arrive+try_wait+fence.cta` | `fence.cluster` / `fence.gpu` / `__threadfence()` / `barrier.cluster.arrive+wait` / `mbarrier.arrive+try_wait+fence.cluster` | `fence.gpu` / `__threadfence()` |
| Effective SASS | `MEMBAR.CTA` | `MEMBAR.GPU` | `MEMBAR.GPU` |

#### 2.1.1. Intra-CTA Producer-Consumer

This means the producer and consumer threads are within the same CTA.
Irrespective of the synchronization mechanism, the equivalent SASS instruction that guarantees memory order is `MEMBAR.CTA`.
It's either built implicitly into the instruction or manually inserted by the compiler when lowering from ptx to SASS.

Two use cases come into mind:
1. Producer threads read from GMEM and write to SMEM. Then consumer threads (can be the same set of threads) read from SMEM and do some computation.
2. In [Algorithm 1 of Flash Attention v1](https://arxiv.org/pdf/2205.14135), The partial result $O_i$ of the previous iteration is written to GMEM by producer threads and then reload back into SMEM for the next iteration by consumer threads.

##### 2.1.1.1. `__syncthreads()` and `Named Barriers`

The most popular way is to use [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions) to guarantee memory order.
`__syncthreads()` implicitly carry the `acq-rel` semantic meaning the threads after `__syncthreads()` can see the memory operations' effects (on *both* SMEM and GMEM) before `__syncthreads()`.
Needless to say, `__syncthreads()` guarantees execution order too, meaning the instructions after `__syncthreads()` will be executed after all the threads in the CTA have arrived at the `__syncthreads()` point.
Combined, this ensures when the consumer threads get unblocked by `__syncthreads()`, the producer threads' finish the memory operations before `__syncthreads()` and the effects are visible.

```python
Producer threads:
    ld.global reg, addr
    st.shared addr, reg
    __syncthreads()
Consumer threads:
    __syncthreads()
    ld.shared reg, addr
    ...
```

[Named Barriers](https://github.com/NVIDIA/cutlass/blob/8afb19d9047afc26816a046059afe66763e68aa5/include/cutlass/arch/barrier.h#L159) has the exact same effect as `__syncthreads()` but can be applied to a subset of threads within the CTA while `__syncthreads()` is applied to all threads in the CTA.

##### 2.1.1.2. `mbarrier.arrive+try_wait`

Alternatively, we can use [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) to guarantee execution order and memory order.
The *default* semantic of [`mbarrier.arrive`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive) is `release` and the *default* semantic of [`mbarrier.try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait) is `acquire`.
This means when the consumer threads finish waiting on the mbarrier, the producer threads finish executing their memory operations before `mbarrier.arrive`.
And the producer threads' memory operations' effects are visible to the consumer threads.

Note that `mbarrier` only applies to SMEM/DSMEM, not GMEM.
Because the state space in the ptx doc is either `.shared::cta` or `.shared::cluster`.

```python
# initialize mbarrier arrive count to num producer threads
Producer threads:
    ld.global reg, addr
    st.shared addr, reg
    mbarrier.arrive.release.cta
Consumer threads:
    mbarrier.try_wait.acquire.cta
    ld.shared reg, addr
    ...
```

##### 2.1.1.3. `mbarrier.relaxed.arrive+try_wait` + `fence.cta`

If you don't want to use the `acquire/release` semantic built into the mbarrier but still want to guarantee memory order, you can use explicit fences + execution order synchronization (achieved by `__syncthreads()` or `mbarrier.relaxed`) to achieve the same effect.

```python
# initialize mbarrier arrive count to num producer threads
Producer threads:
    ld.global reg, addr
    st.shared addr, reg
    fence.release.cta
    mbarrier.relaxed.arrive
Consumer threads:
    mbarrier.relaxed.try_wait
    fence.acquire.cta
    ld.shared reg, addr
    ...
```

Here we use `relaxed` version of the mbarrier to only guarantee execution order.
But we insert explicit memory fences to guarantee memory order.
When the consumer threads get unblocked by `mbarrier.relaxed.try_wait`, this means all the producer threads have finished executing `fence.release.cta`.
Then after the consumer threads execute `fence.acquire.cta`, the memory operation by the producer threads are visible to the consumer threads.

Additionally, since `fence.cta` guarantees memory order on GMEM, we can use this pattern when the producer threads write to GMEM and the consumer threads read from GMEM.
This is what the `acq_rel` semantic of `mbarrier` is not capable of.

The execution order here is also necessary to make this producer-consumer relationship correct.
Imagine not having the `mbarrier` synchronization, how would the consumer threads know when the producer threads have finished executing `fence.release.cta`?
There is no way to know.


#### 2.1.2. Intra-Cluster Producer-Consumer

This means the producer and consumer threads are within the same cluster but not necessarily the same CTA.


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


