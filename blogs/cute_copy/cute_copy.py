# tested on cute dsl 4.3.3

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def cute_copy_kernel_1(
    mA: cute.Tensor,  # Input tensor A
    CTA_M: cutlass.Constexpr,
    CTA_K: cutlass.Constexpr,
    NUM_VAL_PER_THREAD: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()  # Thread index within block (0 to bdim-1)

    gA = cute.local_tile(mA, (CTA_M, CTA_K), (0, 0))

    NUM_THREAD_PER_ROW = CTA_K // NUM_VAL_PER_THREAD # i.e. 128 // 8 = 16

    m_idx = tidx // NUM_THREAD_PER_ROW
    k_idx = (tidx % NUM_THREAD_PER_ROW) * NUM_VAL_PER_THREAD

    rA = cute.make_rmem_tensor((NUM_VAL_PER_THREAD), dtype=mA.element_type)

    for i in cutlass.range(NUM_VAL_PER_THREAD):
        rA[i] = gA[m_idx, k_idx + i]

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
    if cutlass.const_expr(mode == 1):
        kernel = cute_copy_kernel_1(mA, CTA_M, CTA_K, NUM_VAL_PER_THREAD)
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