# tested on cute dsl 4.3.3

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

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

    # (M, K) -> addr
    gA = cute.local_tile(mA, (CTA_M, CTA_K), (0, 0))

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    # starting index of the tile for this thread
    m_idx = tidx // NUM_THREAD_PER_ROW
    k_idx = (tidx % NUM_THREAD_PER_ROW) * NUM_VAL_PER_THREAD

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor((NUM_VAL_PER_THREAD), dtype=mA.element_type)

    # load the value one by one in CUDA C++ style
    for i in cutlass.range(NUM_VAL_PER_THREAD):
        # indexing like CUDA C++
        rA[i] = gA[m_idx, k_idx + i]

    if tidx == 1:
        cute.print_tensor(rA)

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
    
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)
    tiled_copy = cute.make_tiled_copy(copy_atom, TV_layout, (CTA_M, CTA_K))
    thr_copy = tiled_copy.get_slice(tidx)
    tAgA = thr_copy.partition_S(gA) # (Cpy_S, RestM, RestK)
    tArA = thr_copy.partition_D(gA) # (Cpy_D, RestM, RestK)

    # allocate rmem tensor for this thread
    rA = cute.make_rmem_tensor_like(tArA) # (Cpy_D, RestM, RestK)

    # will iterate over each element and do the copy underneath
    # equivalent to cute::copy(tiled_copy, tAgA, rA) in C++
    cute.copy(tiled_copy, tAgA, rA)

    if tidx == 1:
        cute.print_tensor(rA)

@cute.jit
def cute_copy_host(
    mA: cute.Tensor,
    mode: cutlass.Constexpr, # which kernel to pick
):
    CTA_M = 8
    CTA_K = 128
    NUM_VAL_PER_THREAD = 8

    # Create kernel instance
    cute.printf(f"mode: {mode}")
    if cutlass.const_expr(mode == 1):
        kernel = cute_copy_kernel_1(mA, CTA_M, CTA_K, NUM_VAL_PER_THREAD)
    elif cutlass.const_expr(mode == 2):
        kernel = cute_copy_kernel_2(mA, CTA_M, CTA_K, NUM_VAL_PER_THREAD)
    elif cutlass.const_expr(mode == 3):
        kernel = cute_copy_kernel_3(mA, CTA_M, CTA_K, NUM_VAL_PER_THREAD)
    elif cutlass.const_expr(mode == 4):
        kernel = cute_copy_kernel_4(mA, CTA_M, CTA_K, NUM_VAL_PER_THREAD)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Launch kernel with calculated grid dimensions
    kernel.launch(
        grid=(1, 1, 1),
        block=(CTA_M * CTA_K // NUM_VAL_PER_THREAD, 1, 1),
    )

if __name__ == "__main__":
    mA = torch.randn(8, 128).cuda()
    print(f"reference: {mA[0, 8:16]}")
    mA_tensor = from_dlpack(mA, assumed_align=16)
    cute_copy_host(mA_tensor, 1)
    cute_copy_host(mA_tensor, 2)
    cute_copy_host(mA_tensor, 3)
    cute_copy_host(mA_tensor, 4)