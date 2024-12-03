---
layout: default
---

# Using TMA Load and Prefetch in Cute

All the code in this blog can be found [here](https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_tma/code).

The [Tensor Memory Accelerator (TMA)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) is a hardware unit introduced in the NVIDIA Hopper architecture to accelerate tensor data movement. To formally motivate TMA, we need to wind the clock back a bit to Volta.

## Why TMA?

As the tensor core goes faster, it needs enough data to keep it busy. Using little's law: $\text{throughput} = \frac{\text{buffer\_size}}{\text{latency}}$, if we increase the (tensor core) throughput and keep the latency equal, we'd need more staging buffer capacity. 

![cp_async](./cp_async.png)

The figure above shows the sequence of operation when feeding the data to the tensor core. In Volta, the load sequence is `DRAM->L2->L1->RF->shared memory(smem)->RF->tensor core`. The RF/smem/L1/L2 all serve as staging buffer area when feeding the data to the tensor core. While the L2 and smem hold the loaded data tiles, the RF and L1 purely serve the role of staging buffer. Increasing tensor core throughput adds significant staging buffer (RF/L1) usage. So Ampere introduces [asyn memory copy](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) to eliminate the extra staging buffer usage in RF/L1. As shown in the figure above, load completely bypasses RF/L1 such that the new sequence becomes `DRAM->L2->shared memory(smem)->RF->tensor core`. This frees up RF/L1 so that they can be used for other computation.

At this point, it seems like we solve the bandwidth and capacity issue in the memory subsystem. We can make the tensor core go even faster. However, the throughput of other stages need to keep up with the tensor core. At a high level, the computation of a typical gemm kernel can roughly be described as `address generation->load->tensor core (MMA)->epilog (e.g. ReLU, softmax)->address generation->store`. Now let's try to make the kernel run faster. We bump up the tensor core throughput, the MMA stage becomes faster. We bump up the memory bandwidth along with the async memory copy optimization, the load/store stage becomes faster. The throughput of all the stages need to match. Therefore, we need to bump up the throughput of the *address generation* and *epilog* stage. Notice that these two stages both use the *CUDA core* and its throughput stays largely the same. Then we hit a problem, the throughput of the CUDA cores limits the overall throughput of the kernel. This is where TMA comes in. TMA offloads the address generation from the CUDA core. It by itself can generate addresses at a high throughput, matching the throughput of the rest of the stages. With this offloading, the entire CUDA core can be dedicated to the epilog stage, achieving higher throughput of the epilog stage. With the introduction of the TMA, now every stage of the gemm kernel can run at a high throughput.

## What are the ways to use TMA?

Using the TMA can be tricky, there are several options:
1. one can directly use the [CUDA APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access) but it's low level and error-prone. 
2. [Triton](https://pytorch.org/blog/hopper-tma-unit/) adds experimental support for TMA but if you care about squeezing the last percentage of performance (I do :)), you might want to have finer grain control of the kernel.
3. [Cute](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) fortunately offers a high-level abstraction to use TMA with enough lower level control.  

In this blog, I will show how to load and prefetch a tensor using TMA in Cute. Some basic understanding of Cute is required. We leave the more advanced topics like store, multicast, reduction, and swizzle for future blogs.

## TMA Load

### Walkthrough Example

We will walk through an example on how to use TMA to [load](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) a matrix (2d tensor) from global memory (gmem) to shared memory (smem). The figure below shows the shape and layout of the matrix `gmem_tensor`. It's `[N, K]` (`[6, 8] in the example`) and `rowMajor`. A common thing one would want to do is for one CTA (aka threadblock) to load a tile of the matrix. In out example, we want to tile the `gmem_tensor` into tiles of shape `[CTA_N, CTA_K]` (`[2, 4]`) and `rowMajor`. And each CTA loads a tile into smem. Then we need `[N/CTA_N, K/CTA_K]=[gridDim.x, gridDim.y]` (`[6/2, 8/4]=[3,2]`) CTA to achieve this. 

In the example figure below, we have CTA at `(1,1)` in the grid to load the blue tile in `gmem_tensor` to smem.

You can totally imagine the tile to CTA mapping to be different for different applications/implementations. For example, CTA0 loads tile `(0, 0), (2, 1)`, CTA1 loads tile `(0, 1), (2, 0)`, CTA2 loads tile `(1, 0), (1, 1)`. Here we just showcase one example mapping and it's straightforward to modify the code for other mappings.

![grid](./grid.png)

### Host Code

Now we have all the information we need to construct the host side code.

```c++
template <typename T, int CTA_N, int CTA_K>
void cute_host_load(T* data, int N, int K) {
    using namespace cute;

    // 1. create the gmem tensor, row major
    auto gmem_layout = make_layout(make_shape(N, K), make_stride(K, 1));
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // 2. create the smem layout, row major
    // smem_layout need to use static integer
    // use dynamic integer will cause compilation error
    auto smem_layout = make_layout(make_shape(Int<CTA_N>{}, Int<CTA_K>{}), make_stride(Int<CTA_K>{}, _1{}));

    // 3. create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

    // 4. invoke the kernel
    cute_tma_load_kernel<T, CTA_N, CTA_K>
                    <<<dim3{N / CTA_N, K / CTA_K, 1}, 32>>>
                    (tma_load, gmem_tensor, smem_layout);
}
```

Let's break it down:
1. We first create the `gmem_tensor` with shape `[N, K]` and stride `[K, 1]` (i.e. `rowMajor`).
2. Then we create the `smem_layout` with shape `[CTA_N, CTA_K]` and stride `[CTA_K, 1]` (i.e. `rowMajor`). The only thing to note here is the smem related object should use static integer like `cute::_1{}`) to avoid compilation error.
3. Using the `gmem_tensor` pointer and layout along with the `smem_layout`, we create a TMA `Copy_Atom` using [make_tma_copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1290). Underneath, it creates the TMA descriptor (which the user can also use lower level [CUDA APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access) to create). The type of TMA operation (e.g. load/store/prefetch) is specified by the `CopyOp` [SM90_TMA_LOAD](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L277). The name is pretty straightforward, it's a load that uses TMA on SM90 (i.e. hopper). We will discuss the TMA `Copy_Atom` construction process in detail [below](#copy_atom-construction).
4. Finally, we invoke the kernel by passing the various object we initialized in the host function to the kernel. We also specifies the `gridDim` (`dim3{N / CTA_N, K / CTA_K, 1}`) and `blockDim` (`32` since TMA only needs 1 thread to drive).

### Device Code

![crd](./crd.png)

```c++
// assume load a [N, K] row major weight matrix
template <typename T, int CTA_N, int CTA_K, class TmaLoad, class GmemTensor, class SmemLayout>
__global__ void cute_tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor, SmemLayout smem_layout) {
    using namespace cute;
    constexpr int tma_transaction_bytes = CTA_N * CTA_K * sizeof(T);

    // 1. allocate smem for the tile and memory barrier
    __shared__ T smem_data[CTA_N * CTA_K];
    __shared__ uint64_t tma_load_mbar;

    // 2. create the smem tensor
    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout); // [CTA_N, CTA_K]

    // 3. only need 1 thread to drive TMA
    if (threadIdx.x == 0) {
        // 4. initialize the barrier
        initialize_barrier(tma_load_mbar, /* arrival count */ 1);
        set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

        // 5. gets the coordinate of the smem tile in gmem tensor
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
        auto gmem_tensor_coord_cta = local_tile( // [CTA_N, CTA_K]
            gmem_tensor_coord,
            Tile<Int<CTA_N>, Int<CTA_K>>{},
            make_coord(blockIdx.x, blockIdx.y));

        // 6. get the slice of TMA work assigned to this CTA in a threadblock cluster 
        auto tma_load_per_cta = tma_load.get_slice(0);
        // 7. issue TMA load
        copy(tma_load.with(tma_load_mbar),
            tma_load_per_cta.partition_S(gmem_tensor_coord_cta), // [[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]
            tma_load_per_cta.partition_D(smem_tensor)); // [[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]
    }
    // 8. wait for TMA to finish
    __syncthreads();
    wait_barrier(tma_load_mbar, /* phase */ 0);

    // 9. after this line, the TMA load is finished
    if (threadIdx.x == 0) {
        printf("block: (%d, %d), value: %f, %f\n", blockIdx.x, blockIdx.y, float(smem_tensor(make_coord(0, 0))), float(smem_tensor(make_coord(0, 1))));
    }
}
```

Note that the `tma_load` needs to be `__grid_constant__` since the TMA descriptor is created on the host and pass to the device. It can't be modified on device.

Let's break the code down with the help of the figure above:
1. We first allocate the smem space for the tile to be loaded (`smem_data`) and the [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) `tma_load_mbar`.
2. Then we create the `smem_tensor` using the smem pointer and layout.
3. Only 1 thread is needed to drive the TMA load.
4. Here we initialize the barrier. Because the TMA is an async unit, the CTA needs a way to get notified when the data transfer is done. We do this through [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) (refer to the PTX doc to learn more about mbarrier). We first call [initialize_barrier()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_desc.hpp#L64) with expected arrive count 1. Because we will have 1 thread (e.g. the same thread) to arrive on the barrier and set the barrier transaction count for the TMA in [set_barrier_transaction_bytes()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_desc.hpp#L78).
5. Now we obtain the `gmem_tensor_coord` through [get_tma_tensor()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L153). This will give us a tensor with the same layout as `gmem_tensor`. But each entry, instead of the value, is the *coordinate* in the tensor as shown in the figure above. Then [local_tile()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/tensor_impl.hpp#L1016) tiles `gmem_tensor_coord` and returns us a tile `gmem_tensor_coord_cta` (e.g. blue tile) that the CTA wants to load from. The second argument specifies the tile size (i.e. `[CTA_N, CTA_K]`) we want to tile the big tensor with. The third argument specifies which tile we want to get. It does that by passing the coordinate (`[blockIdx.x, blockIdx.y]`) of the tile in the tile space to `local_tile()`. The numpy way to write it would be `gmem_tensor_coord[CTA_N * blockIdx.x : CTA_N * (blockIdx.x + 1), CTA_K * blockIdx.y : CTA_K * (blockIdx.y + 1)]`. To be more concrete, to get the blue tile in the figure, we do `local_tile(gmem_tensor_coord, Tile<Int<2>, Int<4>>{}, make_coord(1, 1))`. We tile `gmem_tensor_coord` by `[2, 4]` tile size and wants to get the tile at coordinate `(1, 1)`. 
6. Now we get a slice ([ThrCopy](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L338)) of the TMA `Copy_Atom` (using function [get_slice()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L322)) that is assigned to this CTA. This is only relevant if you are in a threadblock cluster and the TMA `Copy_Atom` is responsible for the load of entire cluster (with multicasting). Here we only have 1 CTA for the TMA and no threadblock cluster so there is just 1 slice. This is similar to how you get a thread slice of MMA from a TiledMMA (and set the fragment etc.). Here we are getting a CTA slice of the TMA from a tiled TMA in a threadblock cluster. We describe this hierarchy in more details [below](#copy_atom-construction).
7. The [copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L240) function issues the TMA load. It is one of the [Cute built-in algorithms](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/04_algorithms.md) that operates on tensors (other examples are MMA and prefetch). At a high level, `with()` passes the mbarrier to the TMA `Copy_Atom`. `partition_S` and `partition_D` reshapes the smem and gmem tile into the shape the copy function expects. We will have a detailed discussion of how each argument is constructed in the section [below](#the-arguments-of-the-copy-function).
8. Now that we issued the TMA load, we will wait for it to finish. To do this, we first `__syncthreads();` so that every thread arrives at the wait point. Then all the threads [wait_barrier()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_desc.hpp#L92) on the `tma_load_mbar`. The wait is completed when the [phase](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase) of the barrier flips. The initial phase is 0 during [initialization](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init). The second argument of `wait_barrier()` is the current phase bit the barrier waiting to flip. We are waiting for phase 0 to flip, hence 0 is passed in. The TMA unit would decrement the transaction count of the barrier once the loads come back until it reaches 0, meaning all load finishes. Then the barrier flips, all threads' wait is done. We resume execution.
9. At this point, the TMA load is done and load result is in smem and fully visible to all threads. Here we simply print some elements out.

### The Life of the Copy Function

For the curious minds, this entire section is dedicated to explain how the copy function works underneath. You can skip to the [harness code](#harness-code) if you are only interested in how to use TMA.

The [copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L240) function is one of the [Cute built-in algorithms](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/04_algorithms.md) that copies a tensor from the source location to destination location. We are using the three input argument variant where we specify a `Copy_Atom`, the `src` tensor and `dst` tensor. By specifying the `Copy_Atom`, we call tell the compiler explicitly which type of copy we want (e.g. Ampere `cp.async`, Hopper TMA, etc.). There is also a [two input copy variant](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L290) where only the source and destination tensor are passed in so that the copy operation falls back to the default algorithm.

#### `Copy_Atom` Construction

[make_tma_copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1290) constructs the `Copy_Atom`. The sequence of operation is similar to creating a [TiledMMA](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/media/docs/cute/0t_mma_atom.md), i.e. from bottom up `CopyOp->Copy_Traits->Copy_Atom->TiledCopy`. The `CopyOp`, `Copy_Traits`, `Copy_Atom` are all warpers over a PTX copy operation but gradually embed more meta data information in the struct (`CopyOp` has the least and `Copy_Atom` is the most complete). And then [TiledCopy](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L138) (which confusingly is a derived class of `Copy_Atom`) stacks multiple [Copy_Atom](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L52) together to create a larger macro-operation. In the particular case of TMA, a `Copy_Atom` is responsible for the TMA operation of a CTA (in a threadblock cluster), and a `TiledCopy` is responsible for all TMA operations of the entire threadblock cluster. In our simple example, since there is only 1 CTA in a threadblock cluster, the `Copy_Atom` size (`[CTA_N, CTA_K]`) should be identical to the `TiledCopy` size.

The actual call stack is shown below:
- [make_tma_copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1290), because in our example we don't use [TMA multicast load](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L706), we use the initialization function with threadblock cluster size 1.
  - [make_tma_copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1260) is the underlying implementation where we create a [TiledCopy](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L138) object. 
    - [make_tma_copy_tiled()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1136) is broadly broken down into two steps:
      - [make_tma_copy_atom()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1081) that creates the [Copy_Atom](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L52) for a CTA. In it we first define the [Copy_Traits](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L1110) from the `CopyOp` (i.e. [SM90_TMA_LOAD](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L277)). Then we construct the `Copy_Atom` from `Copy_Traits`.
      - [TiledCopy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L138) that stacks `Copy_Atom` of each CTA into a `TiledCopy` for the entire threadblock cluster.


#### The Arguments of the Copy Function

As we presented above, the [copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L240) function takes three arguments: the `Copy_Atom`, the `src` tensor and `dst` tensor. However, we can't pass what we currently have to the copy function, there are some nuances that we explain below.

1. `Copy_Atom`: Remember the TMA unit needs the [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) to synchronize with the CTA, we have never passed the mbarrier handle to the TMA `Copy_Atom` yet! It is called an *non-executable* `Copy_Atom`. This is where `tma_load.with(tma_load_mbar)` comes into play. The [with](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L80) (which eventually got dispatched [here](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L129)) embeds the mbarrier information into the `Copy_Traits` and therefore creates an *executable* `Copy_Atom`. As you can see from the function definition, you can also embed information like L2 cache hint to the `Copy_Atom`.
2. `src` tensor: We do a *source partition* of the tensor (i.e. [partition_S()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L348)). This means we reshape the source tensor layout from `[CTA_N, CTA_K]` to `[[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]`. This layout is what the `copy()` function expects. `[ATOM_N, ATOM_K]` is the TMA `Copy_Atom` size (also commonly referred to as `[TMA]` in the code comment, and `[CTA_N/ATOM_N, CTA_K/ATOM_K]` is referred as `[TMA_N, TMA_K]` in the code comment). So in theory, the copy is broken down into `CTA_N/ATOM_N*CTA_K/ATOM_K` number of steps with each step copying `[ATOM_N, ATOM_K]`. In practice, the `Copy_Atom` size is the entire tile, i.e. `[TMA_N, TMA_K]=[CTA_N, CTA_K]`, so there is only 1 atom/step in the copy, i.e. the src tensor has been reshaped to `[[ATOM_N, ATOM_K], 1, 1]`. You can also verify this by printing out the partitioned src tensor using `cute::print()`.
3. `dst` tensor: We do a *destination partition* of the tensor (i.e. [partition_D()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L358)) similar to src tensor. This means we reshape the destination tensor layout from `[CTA_N, CTA_K]` to `[[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]`.

#### Dispatch Sequence of the Copy Function

1. [copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L240) is the top level interface.
2. [copy_if()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L158) is the predicated version of the copy function. [This](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/algorithm/copy.hpp#L171) is where we loop around `CTA_N/ATOM_N*CTA_K/ATOM_K` steps with each step finishing 1 `Copy_Atom` size work.
3. [copy_atom.call()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_atom.hpp#L94) calls the actual `Copy_Atom`.
4. [copy_unpack()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L68) unpacks the `Copy_Traits` and pass some additional traits into the copy function.
5. [CallCOPY()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/util.hpp#L154) calls the actual `CopyOp::copy` using [explode()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/util.hpp#L181) for passing in additional arguments.
6. [SM90_TMA_LOAD::copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L287) function of CopyOp [SM90_TMA_LOAD](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L277C8-L277C21) is called because we are copying a 2D tensor and provides 2 coordinates, which are the coordinates of the top left corner of the blue tile in `gmem_tensor` coordinate space, `[2, 4]` in the figure.
7. [SM90_TMA_LOAD_2D::copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L96), finally the 2D variants of the [TMA PTX instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) is called.

#### How Are All the Arguments Passed to the PTX Instruction?

Take [SM90_TMA_LOAD_2D::copy()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L96) for example, it takes in TMA descriptor, membar, etc. as arguments, which are unpacked in [copy_unpack()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L68). Here we only focus on explaining how the [crd](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/copy_sm90_tma.hpp#L98) are unpacked and leave the rest as an exercise to the reader.

The coordinate of the TMA tile is obtained from [src.data().coord_](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/atom/copy_traits_sm90_tma.hpp#L72) and [explode()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/arch/util.hpp#L181) eventually unpacks it as `(..., int32_t const& crd0, int32_t const& crd1)`. Remember the src tensor is `gmem_tensor_coord` which holds all the coordinates of the smem tile in gmem. It is an `ArithTuple` and the [Cute doc](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/media/docs/cute/0z_tma_tensors.md) explains it in greater details. 

Basically `gmem_tensor_coord` is a tensor purely for getting the coordinates of the TMA tile (e.g. the blue tile) and pass it to the TMA instruction. In our Cute code, we might want to reshape (i.e. `partition_D()`) or tile the `smem_tensor`. It is very error-prone to manually updating the gmem coordinate to keep track of the `smem_tensor` manipulation. So we simply creates gmem coordinate tensor and manipulate it in exactly the same way. Then the eventual coordinate will always be in sync with `smem_tensor`. We never materialize the coordinate tensor in gmem though, we only manipulate it (reshape/tile, etc.), hence the prefix `Arith`. 

### Harness Code

```c++
// 1. Define TMA load tile size
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 128;

int main() {
    // 2. Define problem size and tensors
    int N = 256;
    int K = 256;

    // we assume this is a [N, K] row major matrix
    cutlass::HostTensor<cutlass::float_e4m3_t, cutlass::layout::RowMajor> B({N, K});

    // 3. init some value on host for B tensor and copy it to GPU memory
    // ...

    B.sync_device();

    // 4. do TMA load to smem
    cute_host_load<cutlass::float_e4m3_t, TILE_N, TILE_K>(B.device_data(), N, K);

    // 5. wait for kernel to complete
    cudaDeviceSynchronize();

    return 0;
}
```

The harness function is pretty straightforward:
1. We first define the tile size (i.e. `CTA_N` and `CTA_K`) we want for each CTA's TMA load
2. Then we define the `gmem_tensor` layout. Here we use the slightly older [HostTensor](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/tools/util/include/cutlass/util/host_tensor.h#L65) API ([tutorial here](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/examples/01_cutlass_utilities/cutlass_utilities.cu)) to define a FP8 (e4m3) row major matrix with shape `[N, K]`.
3. We do some initialization of the tensor on the host. Then we use the HostTensor utility [sync_device()](https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/tools/util/include/cutlass/util/host_tensor.h#L403) to copy the initialized tensor from host to device.
4. Now we call the TMA host function `cute_host_load` to launch the kernel.
5. Finally we use `cudaDeviceSynchronize()` to wait for the kernel to complete.


## TMA Prefetch

TMA unit can also [prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor) the same tile from gmem to L2 cache (instead of loading the tile into smem). This can be useful in cases where the data loading latency is exposed.

The code only needs to be slightly tweaked to conduct prefetching instead of loading.

### Host Code

```c++
template <typename T, int CTA_N, int CTA_K>
void cute_host_prefetch(T* data, int N, int K) {
    using namespace cute;

    // create the GMEM tensor, row major
    auto gmem_layout = make_layout(make_shape(N, K), make_stride(K, 1));
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // create the SMEM layout, row major
    // smem_layout need to use static integer
    // use dynamic integer will cause compilation error
    auto smem_layout = make_layout(make_shape(Int<CTA_N>{}, Int<CTA_K>{}), make_stride(Int<CTA_K>{}, _1{}));

    // create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

    // invoke the kernel
    cute_tma_prefetch_kernel<T, CTA_N, CTA_K>
                    <<<dim3{N / CTA_N, K / CTA_K, 1}, 32>>>
                    (tma_load, gmem_tensor, smem_layout);
}
```

You can see the host side code is exactly the same as [TMA load](#host-code) (other than function names)! This is the power of the Cute abstraction. The tensor layout obviously is the same. Even the TMA object is constructed the same. We specify whether we want to do load or prefetch on the device side, the TMA object will get dispatched into corresponding TMA atom. 

### Device Code

```c++
// assume load a [N, K] row major weight matrix
template <typename T, int CTA_N, int CTA_K, class TmaLoad, class GmemTensor, class SmemLayout>
__global__ void cute_tma_prefetch_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor, SmemLayout smem_layout) {
    using namespace cute;

    if (threadIdx.x == 0) {
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));

        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_N>, Int<CTA_K>>{},
            make_coord(blockIdx.x, blockIdx.y));

        auto tma_load_per_cta = tma_load.get_slice(0);
        prefetch(tma_load,
                 tma_load_per_cta.partition_S(gmem_tensor_coord_cta));
    }
    __syncthreads();

    // after this line, the TMA prefetch is finished
}
```

WIP

## Summary

- TMA offloads address generation from the CUDA core, freeing up resources for other computation (e.g. epilog).
- Cute offers a clean and flexible abstraction to use TMA in CUDA programs.
- We illustrated how to load/prefetch a matrix using TMA in Cute.
- We explain the dispatch sequence from the Cute code to the PTX instruction for TMA load/prefetch.
- All the code in this blog can be found [here](https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_tma/code).

## Additional references:
- [CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Nvidia A100 GPU hot chips 2020](https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_NVIDIA_Choquette_v01.pdf)
- [Nvidia H100 GPU hot chips 2022](https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.NVIDIA.Choquette.vfinal01.pdf)