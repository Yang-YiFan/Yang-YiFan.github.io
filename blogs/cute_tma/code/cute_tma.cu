#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include <cassert>
#include <cstdio>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/trace.h>
#include <cute/util/print.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>

#define gpuErrChk(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

        /*if ((blockIdx.x == 0) && (blockIdx.y == 0)) {
            print(tma_load);
            printf("\n");
            print(tma_load_per_cta);
            printf("\n");
            print(tma_load_per_cta.partition_S(gmem_tensor_coord_cta));
            printf("\n");
            print(tma_load_per_cta.partition_D(smem_tensor));
            printf("\n");
        }*/
    }
    // 8. wait for TMA to finish
    __syncthreads();
    wait_barrier(tma_load_mbar, /* phase */ 0);

    // 9. after this line, the TMA load is finished
    if (threadIdx.x == 0) {
        printf("block: (%d, %d), value: %f, %f\n", blockIdx.x, blockIdx.y, float(smem_tensor(make_coord(0, 0))), float(smem_tensor(make_coord(0, 1))));
        assert((float(smem_tensor(make_coord(0, 0))) - float(blockIdx.x + blockIdx.y)) < 1e-5);
        assert((float(smem_tensor(make_coord(0, 1))) - float(blockIdx.x - blockIdx.y)) < 1e-5);
    }
}

template <typename T, int CTA_N, int CTA_K>
void cute_host_load(T* data, int N, int K) {
    using namespace cute;

    // 1. create the GMEM tensor, row major
    auto gmem_layout = make_layout(make_shape(N, K), make_stride(K, 1));
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // 2. create the SMEM layout, row major
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

// assume load a [N, K] row major weight matrix
template <typename T, int CTA_N, int CTA_K, class TmaLoad, class GmemTensor, class SmemLayout>
__global__ void cute_tma_multicast_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor, SmemLayout smem_layout) {
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
    }

    // 5. synchronize to make all barrier init in a cluster are done
    __syncthreads();
    cluster_sync();
    cutlass::arch::fence_barrier_init();

    if (threadIdx.x == 0) {
        // 6. gets the coordinate of the smem tile in gmem tensor
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
        dim3 clusterIdx = cluster_id_in_grid();
        auto gmem_tensor_coord_cta = local_tile( // [CTA_N, CTA_K]
            gmem_tensor_coord,
            Tile<Int<CTA_N>, Int<CTA_K>>{},
            make_coord(clusterIdx.x, clusterIdx.y));

        // 7. get the cluster related meta data
        uint32_t _block_rank_in_cluster = block_rank_in_cluster();
        dim3 clusterDim = cluster_shape();
        int cluster_size = clusterDim.x * clusterDim.y * clusterDim.z;
        uint16_t tma_mcast_mask = (uint16_t(1) << cluster_size) - 1;

        // 8. get the slice of TMA work assigned to this CTA in a threadblock cluster 
        auto tma_load_per_cta = tma_load.get_slice(_block_rank_in_cluster);
        // 9. issue TMA multicast
        copy(tma_load.with(tma_load_mbar, tma_mcast_mask),
            tma_load_per_cta.partition_S(gmem_tensor_coord_cta), // [[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]
            tma_load_per_cta.partition_D(smem_tensor)); // [[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]

        /*if ((blockIdx.x == 0) && (blockIdx.y == 0)) {
            print(tma_load);
            printf("\n");
            print(tma_load_per_cta);
            printf("\n");
            print(tma_load_per_cta.partition_S(gmem_tensor_coord_cta));
            printf("\n");
            print(tma_load_per_cta.partition_D(smem_tensor));
            printf("\n");
        }*/
    }
    // 10. wait for TMA to finish
    __syncthreads();
    wait_barrier(tma_load_mbar, /* phase */ 0);

    // 11. after this line, the TMA multicast is finished
    if (threadIdx.x == 0) {
        dim3 clusterIdx = cluster_id_in_grid();
        uint32_t _block_rank_in_cluster = block_rank_in_cluster();
        printf("block: (%d, %d, %d), cluster: (%d, %d, %d), cluster rank: %u, value: %f, %f\n", blockIdx.x, blockIdx.y, blockIdx.z, clusterIdx.x, clusterIdx.y, clusterIdx.z, _block_rank_in_cluster, float(smem_tensor(make_coord(0, 0))), float(smem_tensor(make_coord(0, 1))));
        assert((float(smem_tensor(make_coord(0, 0))) - float(clusterIdx.x + clusterIdx.y)) < 1e-5);
        assert((float(smem_tensor(make_coord(0, 1))) - float(clusterIdx.x - clusterIdx.y)) < 1e-5);
    }

    cluster_sync();
}

template <typename T, int CTA_N, int CTA_K, int CLUSTER_N, int CLUSTER_K>
void cute_host_multicast(T* data, int N, int K) {
    using namespace cute;

    // 1. create the GMEM tensor, row major
    auto gmem_layout = make_layout(make_shape(N, K), make_stride(K, 1));
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // 2. create the SMEM layout, row major
    // smem_layout need to use static integer
    // use dynamic integer will cause compilation error
    auto smem_layout = make_layout(make_shape(Int<CTA_N>{}, Int<CTA_K>{}), make_stride(Int<CTA_K>{}, _1{}));

    // 3. create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gmem_tensor, smem_layout, Int<CLUSTER_N * CLUSTER_K>{});

    // 4. invoke the kernel using extensible launch interface
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = dim3{N / CTA_N * CLUSTER_N, K / CTA_K * CLUSTER_K, 1};
    config.blockDim = 32;
    config.dynamicSmemBytes = 0;
    config.stream = 0x0;
    config.numAttrs = 1;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_N;
    attrs[0].val.clusterDim.y = CLUSTER_K;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;

    auto* kernel = &cute_tma_multicast_kernel<T, CTA_N, CTA_K, decltype(tma_load), decltype(gmem_tensor), decltype(smem_layout)>;

    gpuErrChk(cudaLaunchKernelEx(&config, kernel, tma_load, gmem_tensor, smem_layout));
}

// assume load a [N, K] row major weight matrix
template <typename T, int CTA_N, int CTA_K, class TmaLoad, class GmemTensor, class SmemLayout>
__global__ void cute_tma_prefetch_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor, SmemLayout smem_layout) {
    using namespace cute;

    // 1. only need 1 thread to drive TMA
    if (threadIdx.x == 0) {
        // 2. gets the coordinate of the smem tile in gmem tensor
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_N>, Int<CTA_K>>{},
            make_coord(blockIdx.x, blockIdx.y));

        // 3. get the slice of TMA work assigned to this CTA in a threadblock cluster 
        auto tma_load_per_cta = tma_load.get_slice(0);
        // 4. issue TMA load
        prefetch(tma_load,
                 tma_load_per_cta.partition_S(gmem_tensor_coord_cta)); // [[ATOM_N, ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]
    }
    // 5. no need to wait for prefetch, just make sure all threads converge
    __syncthreads();

    // 6. after this line, the TMA prefetch is finished
}

template <typename T, int CTA_N, int CTA_K>
void cute_host_prefetch(T* data, int N, int K) {
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
    cute_tma_prefetch_kernel<T, CTA_N, CTA_K>
                    <<<dim3{N / CTA_N, K / CTA_K, 1}, 32>>>
                    (tma_load, gmem_tensor, smem_layout);
}

// 1. Define TMA load/prefetch tile size
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 128;
static constexpr int CLUSTER_N = 2;
static constexpr int CLUSTER_K = 2;

int main() {

    int iter = 1;

    // 2. Define problem size and tensors
    int N = 256;
    int K = 256;

    // we assume this is a [N, K] row major matrix
    cutlass::HostTensor<cutlass::float_e4m3_t, cutlass::layout::RowMajor> B({N, K});

    // 3. init some value on host for B tensor and copy it to GPU memory
    for(int i = 0; i < N / TILE_N; i++) {
        for(int j = 0; j < K / TILE_K; j++) {
            B.at({i * TILE_N, j * TILE_K}) = cutlass::float_e4m3_t((float)i + j);
            B.at({i * TILE_N, j * TILE_K + 1}) = cutlass::float_e4m3_t((float)i - j);
        }
    }

    B.sync_device();

    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    gpuErrChk(cudaProfilerStart());
    gpuErrChk(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        // 4. do TMA load to smem
        cute_host_load<cutlass::float_e4m3_t, TILE_N, TILE_K>(B.device_data(), N, K);
        // uncomment this if you want to use multicast
        //cute_host_multicast<cutlass::float_e4m3_t, TILE_N, TILE_K, CLUSTER_N, CLUSTER_K>(B.device_data(), N, K);
        // uncomment this if you want to use prefetch
        //cute_host_prefetch<cutlass::float_e4m3_t, TILE_N, TILE_K>(B.device_data(), N, K);
    }
    gpuErrChk(cudaEventRecord(stop));
    gpuErrChk(cudaProfilerStop());

    // 5. wait for kernel to complete
    gpuErrChk(cudaDeviceSynchronize());

    float elapsed_time_ms;
    gpuErrChk(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

    printf("execution time %f ms\n", elapsed_time_ms);

    return 0;
}