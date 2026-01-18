#include <iostream>
#include <cstdio>

#include <cutlass/gemm/collective/builders/sm100_common.inl> // mma/smem selector, umma::major
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

using namespace cute;

int main() {
    cudaFree(0);

    int M = 1024;
    int K = 1024;
    int constexpr TILE_M = 32;
    int constexpr TILE_K = 128;
    using T = cutlass::bfloat16_t;

    auto gmem_layout = make_layout(make_shape(M, K), make_stride(K, 1)); // k major
    auto gtensor = make_tensor(make_gmem_ptr((T*)nullptr), gmem_layout);
    
    auto SmemLayoutAtom = UMMA::Layout_K_SW128_Atom<T>{};
    //auto SmemLayoutAtom = UMMA::Layout_K_INTER_Atom<T>{};
    auto smem_shape = make_shape(Int<TILE_M>{}, Int<TILE_K>{});
    // with Step<_1, _2>, basically we say the swizzle atom is first stacked along M, then K
    auto smem_layout = tile_to_shape(SmemLayoutAtom, smem_shape, Step<_1, _2>{}); // (TILE_M, TILE_K)

    Copy_Atom tma_atom = make_tma_atom(
        SM90_TMA_LOAD{},                              // TMA Load Op, sm100 reuses sm90 tma atom
        gtensor,                                      // Source GMEM tensor
        smem_layout,                                  // Destination SMEM layout for 1 DMA_Stage, ((Mma_M, Mma_K), NumMma_M, NumMma_K)
        smem_shape                                    // TMA box shape, it's cosize must match the cosize of the destination smem layout
    );

    print("tma_atom:\t"); print(tma_atom); print("\n");

    return 0;
}