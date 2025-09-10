#pragma once

#include <cassert>
#include <iostream>
#include "cuda_runtime.h"

// Cutlass includes
#include <cutlass/arch/barrier.h>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation

using namespace cute;

#define gpuErrChk(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote_f32(float value, uint32_t dsmem_addr, uint32_t remote_barrier_addr)
{
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "f"(value), "r"(remote_barrier_addr));
}


template <int NUM_SPLITS>
auto make_layout_gAcc(int M) {
    // (NUM_SPLITS, M) : (M, 1)
    return make_layout(make_shape(Int<NUM_SPLITS>{}, M), make_stride(M, Int<1>{}));
}

template <int CTA_M>
auto make_layout_sAcc() {
    // CTA_M : 1
    return make_layout(Int<CTA_M>{}, Int<1>{});
}

template <int NUM_SPLITS, int CTA_M>
auto make_layout_sMailbox() {
    // (NUM_SPLITS, CTA_M/NUM_SPLITS) : (CTA_M/NUM_SPLITS, 1)
    int constexpr col = CTA_M / NUM_SPLITS;
    return make_layout(make_shape(Int<NUM_SPLITS>{}, Int<col>{}), make_stride(Int<col>{}, Int<1>{}));
}

auto make_layout_gResult(int M) {
    // M : 1
    return make_layout(M, Int<1>{});
}

template <typename T,
          class AccSmemLayout,
          class MailboxSmemLayout>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<AccSmemLayout>> Acc;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<MailboxSmemLayout>> Mailbox;

    alignas(16) cute::uint64_t mailbox_full_barrier; // barrier indicating other CTA's STAS write to local mailbox are done

    CUTE_DEVICE constexpr auto tensor_acc() { return make_tensor(make_smem_ptr(Acc.begin()), AccSmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_mailbox() { return make_tensor(make_smem_ptr(Mailbox.begin()), MailboxSmemLayout{}); }
};

template <class SharedStorage,
          class AccTensor,
          class ResultTensor,
          int CTA_M,
          int NUM_SPLITS,
          int NUM_THREADS>
__global__ void cga_reduction_device(
    AccTensor mAcc,       // (NUM_SPLITS, M)
    ResultTensor mResult  // (M)
) {
    using T = typename AccTensor::value_type;

    // Allocate SMEM
    extern __shared__ char shared_memory[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    // cta's 1d rank in the cga
    uint32_t rank = block_rank_in_cluster();
    int warp_idx = cutlass::canonical_warp_idx_sync();

    if (threadIdx.x == 0) {
        printf("cta id (%d, %d, %d) rank in cga %d\n", blockIdx.x, blockIdx.y, blockIdx.z, rank);
    }

    // only use 1 warp to initialize barrier
    if (warp_idx == 0) {
        cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterTransactionBarrier, 1>(&shared_storage.mailbox_full_barrier, /* arrival count */ 1);
    }

    // first do the CTA level partition, i.e. get the gmem/smem tile of this cta
    auto gAcc = local_tile(mAcc(rank,_), Int<CTA_M>{}, blockIdx.y);  // (CTA_M)
    auto gResult_cga = local_tile(mResult, Int<CTA_M>{}, blockIdx.y);  // (CTA_M) per CGA
    auto gResult = local_tile(gResult_cga, Int<CTA_M/NUM_SPLITS>{}, blockIdx.x);  // (CTA_M/NUM_SPLITS) per CTA

    auto sAcc = shared_storage.tensor_acc();
    auto sMailbox = shared_storage.tensor_mailbox();

    // we first copy the acc tensor to the smem, we can do it in two ways:
    // 1. manually construct a tiledCopy atom which specifies a TV layout, partition_S and partition_D the gmem/smem tensor and do the copy
    // or 2. use cute layout algebra to directly partition the gmem/smem tensor into per thread pieces, and let copy algorithm to automatically figure out 
    //       the appropriate copy atom to do the copy
    // we use the second way here since we don't want to construct an explicit TV layout
    //
    // the partition variant we are using here is local_partition (instead of the popular local_tile for partitioning at the CTA level)
    // this means per thread's value are discontiguous in the tile, for local_tile, per thread's value are contiguous in the tile
    // with a tiler size of 2 (2 threads) and tile size of 8, the two partitioning scheme gives us:
    // local_tile: 
    //     T0V0, T0V1, T0V2, T0V3, T1V0, T1V1, T1V2, T1V3
    // local_partition: 
    //     T0V0, T1V0, T0V1, T1V1, T0V2, T1V2, T0V3, T1V3
    // the reason why we want to use local_partition is to avoid smem bank conflicts, in this interleaved layout, each thread writes to different banks
    auto thr_gAcc = local_partition(gAcc, make_layout(Int<NUM_THREADS>{}), threadIdx.x);  // (CTA_M/NUM_THREADS)
    auto thr_sAcc = local_partition(sAcc, make_layout(Int<NUM_THREADS>{}), threadIdx.x);  // (CTA_M/NUM_THREADS)
    // allocate rmem for g->r->s copy
    Tensor rAcc = make_tensor<T>(shape(thr_gAcc)); // (CTA_M/NUM_THREADS)

    // now we partition the src and mailbox (dst) tensor into per thread pieces for STAS copy, we copy subtile by subtile
    // meaning acc tensor is also first partitioned into 4 subtiles (assuming NUM_SPLITS = 4), and each subtile copy is to a different CTA
    // Acc tensor:
    //   CTA0:    CTA0Tile0, CTA0Tile1, CTA0Tile2, CTA0Tile3
    //   CTA1:    CTA1Tile0, CTA1Tile1, CTA1Tile2, CTA1Tile3
    //   CTA2:    CTA2Tile0, CTA2Tile1, CTA2Tile2, CTA2Tile3
    //   CTA3:    CTA3Tile0, CTA3Tile1, CTA3Tile2, CTA3Tile3
    // after the copy, the content in the mailbox tensor:
    //   CTA0:    CTA0Tile0, CTA1Tile0, CTA2Tile0, CTA3Tile0
    //   CTA1:    CTA0Tile1, CTA1Tile1, CTA2Tile1, CTA3Tile1
    //   CTA2:    CTA0Tile2, CTA2Tile2, CTA2Tile2, CTA3Tile2
    //   CTA3:    CTA0Tile3, CTA3Tile3, CTA3Tile3, CTA3Tile3
    // we partition each subtile across all threads in the CTA separately which results in 4 separate copy procedures for CTA0:
    // CTA0Tile0: CTA0 -> CTA0's mailbox
    // CTA0Tile1: CTA0 -> CTA1's mailbox
    // CTA0Tile2: CTA0 -> CTA2's mailbox
    // CTA0Tile3: CTA0 -> CTA3's mailbox
    //
    // first reshape the acc tensor (STAS src) into 4 subtiles
    auto sAcc_subtiles = flat_divide(sAcc, Int<CTA_M/NUM_SPLITS>{}); // (CTA_M/NUM_SPLITS, NUM_SPLITS)
    // then partition the first mode (subtile mode) into each thread
    auto thr_sAcc_stas = local_partition(sAcc_subtiles, make_layout(Int<NUM_THREADS>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS, NUM_SPLITS)

    // each entry holds a mailbox subtile (dst) for a different CTA, that is the copy destination of the subtile
    // for CTA0, this means the content of dsMailbox is: CTA0Mailbox0, CTA1Mailbox0, CTA2Mailbox0, CTA3Mailbox0
    decltype(sMailbox(0,_)) dsMailbox[NUM_SPLITS]; // (NUM_SPLITS, CTA_M/NUM_SPLITS)
    uint32_t dsBarrier[NUM_SPLITS]; // barrier addr of each subtile, so we are storing dsmem addr of each barrier of each CTA
    // do mapa and get the dsmem addr of each subtile
    CUTE_UNROLL
    for (int i = 0; i < NUM_SPLITS; i++) {
        // sMailbox(i,_).data().get() is the smem addr in the generic addr space, in the generic addr space a region is reserved for smem
        // doing ld/st to this region of the generic addr space will be converted into lds/sts to the smem addr space by the compiler
        // the mapa (and many inline ptx) instruction's input and output addr are in the smem/dsmem addr space, so we need to explicitly convert from generic to shared addr space
        uint32_t smem_addr = __cvta_generic_to_shared(sMailbox(rank,_).data().get()); // smem addr space
        // mapa to get the dsmem addr of this subtile in another CTA
        uint32_t dsmem_addr = set_block_rank(smem_addr, i); // smem addr space
        T* dsmem_ptr = (T*)__cvta_shared_to_generic(dsmem_addr); // generic addr space
        dsMailbox[i] = make_tensor(make_smem_ptr(dsmem_ptr), sMailbox(rank,_).layout());

        uint32_t smem_barrier_addr = __cvta_generic_to_shared(&shared_storage.mailbox_full_barrier);
        uint32_t dsmem_barrier_addr = set_block_rank(smem_barrier_addr, i);
        dsBarrier[i] = dsmem_barrier_addr;

        /*if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
            printf("mailbox rank: %d, smem_ptr: %p, smem_addr: %x -> dsmem_ptr: %p, dsmem_addr: %x\n", i, sMailbox(rank,_).data().get(), smem_addr, dsmem_ptr, dsmem_addr);
            printf("sMailbox[%d]:\t", i); print(sMailbox(rank,_)); printf("\n");
            printf("dsMailbox[%d]:\t", i); print(dsMailbox[i]); printf("\n");

            printf("barrier rank: %d, smem_barrier_addr: %x -> dsmem_barrier_addr: %x\n", i, smem_barrier_addr, dsmem_barrier_addr);
            printf("dsBarrier[%d]:\t", i); printf("%x\n", dsBarrier[i]);
        }*/
    }
    // now we do the STAS dst thread partition, i.e. subtiles in dsmem, we use the same local_partition scheme as src
    auto thr_dsMailbox_eg = local_partition(dsMailbox[0], make_layout(Int<NUM_THREADS>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS)
    decltype(thr_dsMailbox_eg) thr_dsMailbox_stas[NUM_SPLITS]; // (NUM_SPLITS, CTA_M/NUM_SPLITS/NUM_THREADS)
    CUTE_UNROLL
    for (int i = 0; i < NUM_SPLITS; i++) {
        thr_dsMailbox_stas[i] = local_partition(dsMailbox[i], make_layout(Int<NUM_THREADS>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS)
    }

    // now we do the thread partitioning for the reduction
    // we do the local_partition for each subtile separately, however, local_partition only works on the first mode, but we want to partition the second mode of sMailbox
    // so we create a new view of sMailbox first by using select to permute the mode order of sMailbox.layout()
    auto sMailbox_view = make_tensor(sMailbox.data(), select<1, 0>(sMailbox.layout()));
    auto thr_sMailbox = local_partition(sMailbox_view, make_layout(Int<NUM_THREADS>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS, NUM_SPLITS)
    // the other way to do this is to do have a shape of 1 on the first mode, and partition the second mode with the same shape of NUM_THREADS
    //auto thr_sMailbox = local_partition(sMailbox_view, make_layout(Shape<Int<1>, Int<NUM_THREADS>>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS, NUM_SPLITS)

    // allocate rmem for reduction accumulator
    auto thr_red_acc = make_tensor<T>(shape(thr_sMailbox(_, 0))); // (CTA_M/NUM_SPLITS/NUM_THREADS)
    // partition the result tensor into per thread pieces for store back to gmem
    auto thr_gResult = local_partition(gResult, make_layout(Int<NUM_THREADS>{}), threadIdx.x); // (CTA_M/NUM_SPLITS/NUM_THREADS)

    int transaction_bytes = sizeof(make_tensor_like(sMailbox));

    /*if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
        printf("mAcc(rank, _):\t"); print(mAcc(rank, _)); printf("\n");
        printf("mResult:\t"); print(mResult); printf("\n");

        printf("gAcc:\t"); print(gAcc); printf("\n");
        printf("gResult_cga:\t"); print(gResult_cga); printf("\n");
        printf("gResult:\t"); print(gResult); printf("\n");

        printf("sAcc:\t"); print(sAcc); printf("\n");
        printf("sMailbox:\t"); print(sMailbox); printf("\n");

        printf("thr_gAcc:\t"); print(thr_gAcc); printf("\n");
        printf("thr_sAcc:\t"); print(thr_sAcc); printf("\n");
        printf("rAcc:\t"); print(rAcc); printf("\n");

        printf("sAcc_subtiles:\t"); print(sAcc_subtiles); printf("\n");
        printf("thr_sAcc_stas:\t"); print(thr_sAcc_stas); printf("\n");
        printf("thr_dsMailbox_stas:\t"); print(thr_dsMailbox_stas[0]); printf("\n");

        printf("thr_sMailbox:\t"); print(thr_sMailbox); printf("\n");
        printf("thr_red_acc:\t"); print(thr_red_acc); printf("\n");
        printf("thr_gResult:\t"); print(thr_gResult); printf("\n");

        printf("transaction_bytes: %d\n", transaction_bytes);
    }*/

    // gmem->rmem
    copy(thr_gAcc, rAcc);
    // rmem->smem
    copy(rAcc, thr_sAcc);

    // make the store to smem visible within the entire cga
    // also make sure the barrier initialization is visible to the entire cga
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    // this will have a membar.gpu to ensure dsmem write visibility within the entire cga, because there isn't a membar.cga
    cluster_sync();

    // only use 1 thread to set transaction byte
    if (elect_one_sync() && (warp_idx == 0)) {
        set_barrier_transaction_bytes(shared_storage.mailbox_full_barrier, transaction_bytes);
    }

    // do STAS copy
    CUTE_UNROLL
    for (int i = 0; i < NUM_SPLITS; i++) {
        CUTE_UNROLL
        for (int j = 0; j < size(thr_sAcc_stas(_,i)); j++) {
            //printf("i: %d, j: %d, &thr_dsMailbox_stas[i](j): %x\n", i, j, &thr_dsMailbox_stas[i](j));
            uint32_t dsmem_addr = __cvta_generic_to_shared(&thr_dsMailbox_stas[i](j));
            store_shared_remote_f32(thr_sAcc_stas(j,i), dsmem_addr, dsBarrier[i]);
        }
    }

    // initial phase bit is 0, waiting for it to flip to 1
    int phase_bit = 0;
    // wait_barrier wait on the old phase bit
    wait_barrier(shared_storage.mailbox_full_barrier, phase_bit);

    //if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
    //    // reshape sAcc to (CTA_M/NUM_SPLITS, NUM_SPLITS)
    //    printf("sAcc:\t"); print_tensor(flat_divide(sAcc, Int<CTA_M/NUM_SPLITS>{})); printf("\n");
    //    printf("sMailbox:\t"); print_tensor(sMailbox); printf("\n");
    //}

    // zero out the reduction accumulator
    clear(thr_red_acc);

    // do the reduction
    CUTE_UNROLL
    for (int i = 0; i < NUM_SPLITS; i++) {
        CUTE_UNROLL
        for (int j = 0; j < size(thr_sMailbox(_, i)); j++) {
            thr_red_acc(j) += thr_sMailbox(j, i);
        }
    }

    // store the reduction result from rmem back to gmem
    copy(thr_red_acc, thr_gResult);
}


template <typename T,
          int CTA_M,
          int NUM_SPLITS,
          int NUM_THREADS>
void cga_reduction_host(
    T* device_ptr_acc,
    T* device_ptr_result,
    int M,
    bool fdl,
    cudaStream_t stream = 0
) {
    // TODO: handle predication later, for now we assume no remainder
    assert(M % CTA_M == 0);
    assert(CTA_M % (NUM_SPLITS * NUM_THREADS) == 0);

    Layout gAccLayout = make_layout_gAcc<NUM_SPLITS>(M);
    Layout gResultLayout = make_layout_gResult(M);

    Tensor mAcc = make_tensor(make_gmem_ptr(device_ptr_acc), gAccLayout);          // (NUM_SPLITS, M)
    Tensor mResult = make_tensor(make_gmem_ptr(device_ptr_result), gResultLayout); // (M)

    printf("gAcc: "); print(mAcc); printf("\n");
    printf("gResult: "); print(mResult); printf("\n");

    Layout sAccLayout = make_layout_sAcc<CTA_M>();
    Layout sMailboxLayout = make_layout_sMailbox<NUM_SPLITS, CTA_M>();

    printf("sAccLayout: "); print(sAccLayout); printf("\n");
    printf("sMailboxLayout: "); print(sMailboxLayout); printf("\n");

    using SMEMStorage = SharedStorage<T, decltype(sAccLayout), decltype(sMailboxLayout)>;

    int smemBytes = sizeof(SMEMStorage);

    // invoke the kernel
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[2];
    // each batch is a set of separate CTA
    config.gridDim = dim3{(uint32_t)NUM_SPLITS, (uint32_t)cutlass::ceil_div(M, CTA_M), (uint32_t)1};
    config.blockDim = NUM_THREADS; // 4 warps
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {NUM_SPLITS, 1, 1};
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = fdl ? 2 : 1;

    auto *kernel_instance = &cga_reduction_device<SMEMStorage, decltype(mAcc), decltype(mResult), CTA_M, NUM_SPLITS, NUM_THREADS>;
    gpuErrChk(cudaFuncSetAttribute(*kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    gpuErrChk(cudaLaunchKernelEx(&config, kernel_instance, mAcc, mResult));
}

