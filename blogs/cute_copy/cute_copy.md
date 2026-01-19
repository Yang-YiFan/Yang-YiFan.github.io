---
layout: default
---

# 4 Ways to Do CuTe Copy

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

## 0. Introduction

The famous Chinese writer [Lu Xun](https://en.wikipedia.org/wiki/Lu_Xun) (鲁迅) once said: 
> There are 4 ways to conduct a CuTe copy.

In this blog, I will list the 4 ways to do CuTe copy following Lu Xun's guidance.
Joke aside, through the examples, I'm hoping to convey the following messages:
- There are many equivalent ways to partition a tensor into tiles in CuTe. Depending on the specific use case, one may be preferred over the other.
- When you are writing a CuTe kernel, it is still fundamentally SIMT (you have many threads running in parallel).

All the code in this blog can be found [here](https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_copy/cute_copy.py).

## 1. Working Example

We are going to do something extremely simple: load a `8x128` GMEM tensor (row major) into RF using 128 threads and 1 CTA.
The way we partition the tensor to each thread is shown in the following figure:

![partition](./figures/partition.png)

Essentially each thread loads a `1x8` tile of the GMEM tensor.
In the following sections, we will show 4 ways to do this simple task.
And we will use [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html) to write the kernel.

## 2. Approach 1: Traditional CUDA C++ Style

Version 1 of the kernel tries to mimic what you would do in CUDA C++ as much as possible.
The only CuTe part we leverage the layout of input tensor such that we don't need to hand calculate the GMEM address.


```python
# Approach 1: Traditional CUDA C++ Style
@cute.kernel
def cute_copy_kernel_1(
    mA: cute.Tensor,  # Input tensor A
    CTA_M: cutlass.Constexpr,
    CTA_K: cutlass.Constexpr,
    NUM_VAL_PER_THREAD: cutlass.Constexpr,
):
    # 128 threads
    tidx, _, _ = cute.arch.thread_idx()

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    # starting index of the tile for this thread
    m_idx = tidx // NUM_THREAD_PER_ROW
    k_idx = (tidx % NUM_THREAD_PER_ROW) * NUM_VAL_PER_THREAD

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor((NUM_VAL_PER_THREAD), dtype=mA.element_type)

    # load the value one by one in CUDA C++ style
    for i in cutlass.range(NUM_VAL_PER_THREAD):
        # indexing like CUDA C++
        rA[i] = mA[m_idx, k_idx + i]

    if tidx == 1:
        cute.print_tensor(rA)
```

## 3. Approach 2: Using CuTe Layout Algebra

```python
# Approach 2: Using CuTe Layout Algebra
@cute.kernel
def cute_copy_kernel_2(
    mA: cute.Tensor,  # Input tensor A
    CTA_M: cutlass.Constexpr,
    CTA_K: cutlass.Constexpr,
    NUM_VAL_PER_THREAD: cutlass.Constexpr,
):
    # 128 threads
    tidx, _, _ = cute.arch.thread_idx()

    # (M, K) -> addr
    gA = cute.local_tile(mA, (CTA_M, CTA_K), (0, 0))

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    # use layout algebra to get the per thread tile of gA
    tAgA = cute.flat_divide(gA, (1, NUM_VAL_PER_THREAD)) # (TileM, TileK, RestM, RestK, L, ...)
    tAgA_layout = cute.select(tAgA.layout, [0, 1, 3, 2]) # (TileM, TileK, RestK, RestM)
    tAgA = cute.make_tensor(tAgA.iterator, tAgA_layout) # (TileM, TileK, RestK, RestM)
    tAgA = cute.group_modes(tAgA, 2, 4) # (TileM, TileK, (RestK, RestM))
    tAgA = tAgA[0, None, tidx] # (TileK)

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor((NUM_VAL_PER_THREAD), dtype=mA.element_type)

    # will iterate over each element and do the copy underneath
    # equivalent to cute::copy(tAgA, rA) in C++
    cute.basic_copy(tAgA, rA)

    if tidx == 1:
        cute.print_tensor(rA)
```

## 4. Approach 3: Using TV-Layout + Composition

```python
# Approach 3: Using TV-Layout + Composition
@cute.kernel
def cute_copy_kernel_3(
    mA: cute.Tensor,  # Input tensor A
    CTA_M: cutlass.Constexpr,
    CTA_K: cutlass.Constexpr,
    NUM_VAL_PER_THREAD: cutlass.Constexpr,
):
    # 128 threads
    tidx, _, _ = cute.arch.thread_idx()

    # (M, K) -> addr
    gA = cute.local_tile(mA, (CTA_M, CTA_K), (0, 0))

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    # create the TV-layout to represent the thread partitioning
    # (T, V) -> (M, K)
    TV_layout = cute.make_layout(((NUM_THREAD_PER_ROW, CTA_M), NUM_VAL_PER_THREAD), stride=((CTA_M * NUM_VAL_PER_THREAD, 1), CTA_M))
    tAgA = cute.composition(gA, TV_layout) # (T, V)
    tAgA = tAgA[tidx, None] # (V)

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor((NUM_VAL_PER_THREAD), dtype=mA.element_type)

    # will iterate over each element and do the copy underneath
    # equivalent to cute::copy(tAgA, rA) in C++
    cute.basic_copy(tAgA, rA)

    if tidx == 1:
        cute.print_tensor(rA)
```

## 5. Approach 4: Using TV-Layout + TiledCopy

```python
# Approach 4: Using TV-Layout + TiledCopy
@cute.kernel
def cute_copy_kernel_4(
    mA: cute.Tensor,  # Input tensor A
    CTA_M: cutlass.Constexpr,
    CTA_K: cutlass.Constexpr,
    NUM_VAL_PER_THREAD: cutlass.Constexpr,
):
    # 128 threads
    tidx, _, _ = cute.arch.thread_idx()

    # (M, K) -> addr
    gA = cute.local_tile(mA, (CTA_M, CTA_K), (0, 0))

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    # create the TV-layout to represent the thread partitioning
    # (T, V) -> (M, K)
    TV_layout = cute.make_layout(((NUM_THREAD_PER_ROW, CTA_M), NUM_VAL_PER_THREAD), stride=((CTA_M * NUM_VAL_PER_THREAD, 1), CTA_M))
    
    # create the tiled copy that takes the TV layout
    # each tiled copy will copy the entire [CTA_M, CTA_K] tile using 128 threads
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)
    tiled_copy = cute.make_tiled_copy(copy_atom, TV_layout, (CTA_M, CTA_K))
    thr_copy = tiled_copy.get_slice(tidx)

    # partition the source and destination tensors into subtiles (each subtile will be copied by 1 tiled copy operation)
    tAgA = thr_copy.partition_S(gA) # (Cpy_S, RestM, RestK)
    tArA = thr_copy.partition_D(gA) # (Cpy_D, RestM, RestK)

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor_like(tArA) # (Cpy_D, RestM, RestK)

    # will iterate over each tiled copy subtile and do the tiled copy underneath
    # equivalent to cute::copy(tiled_copy, tAgA, rA) in C++
    cute.copy(tiled_copy, tAgA, rA)

    if tidx == 1:
        cute.print_tensor(rA)
```


## 6. Summary

