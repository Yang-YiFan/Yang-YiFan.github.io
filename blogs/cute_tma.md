---
layout: default
title: Using TMA Load and Prefetch in Cute
---

# Using TMA Load and Prefetch in Cute

WIP

```c++
// assume load a [N, K] row major weight matrix
template <typename T, int CTA_N, int CTA_K, class TmaLoad, class GmemTensor, class SmemLayout>
__global__ void cute_tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor, SmemLayout smem_layout) {
    using namespace cute;
    constexpr int tma_transaction_bytes = CTA_N * CTA_K * sizeof(T);

    __shared__ T smem_data[CTA_N * CTA_K];
    __shared__ uint64_t tma_load_mbar;

    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);

    if (threadIdx.x == 0) {
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));

        initialize_barrier(tma_load_mbar, /* arrival count */ 1);

        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_N>, Int<CTA_K>>{},
            make_coord(blockIdx.x, blockIdx.y));

        set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

        auto tma_load_per_cta = tma_load.get_slice(0);
        copy(tma_load.with(tma_load_mbar),
            tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
            tma_load_per_cta.partition_D(smem_tensor));
    }
    __syncthreads();
    wait_barrier(tma_load_mbar, /* phase */ 0);

    // after this line, the TMA load is finished
    if (threadIdx.x == 0) {
        printf("block: (%d, %d), value: %f, %f\n", blockIdx.x, blockIdx.y, float(smem_tensor(make_coord(0, 0))), float(smem_tensor(make_coord(0, 1))));
    }
}

template <typename T, int CTA_N, int CTA_K>
void cute_host_load(T* data, int N, int K) {
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
    cute_tma_load_kernel<T, CTA_N, CTA_K>
                    <<<dim3{N / CTA_N, K / CTA_K, 1}, 32>>>
                    (tma_load, gmem_tensor, smem_layout);
}

```