import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.runtime as cute_rt

CTA_M = 64
CTA_K = 256

@cute.kernel
def cute_tma_load_kernel(tma_load: cute.CopyAtom, tma_tensor: cute.Tensor, gmem_tensor: cute.Tensor, smem_layout: cute.ComposedLayout):
    M, K = tma_tensor.shape
    bytes = cute.size_in_bytes(gmem_tensor.element_type, smem_layout)
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    
    # Create shared memory buffer
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gmem_tensor.element_type, smem_layout.outer, 16, smem_layout.inner)
    # initialize a single mbarrier (64bit)
    tma_load_mbar = smem.allocate_array(cutlass.Int64)

    # initialize the barrier and set arrival count to 1
    # the initial phase is 0
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(tma_load_mbar, 1)

    # barrier init fence to ensure barrier is visible to all threads
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    current_phase = 0
    for k in range(K // CTA_K):
        #with cute.arch.elect_one():
        #    if block_idx == 1:
        #        cute.printf("k: %d", k)

        # get the tile of gmem (coordinate) tensor of this k block, tiled from the whole gmem coordinate tensor
        gmem_tensor_coord_cta = cute.local_tile(tma_tensor, smem_layout.shape, (block_idx, k))

        # set the expected bytes to be loaded
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(tma_load_mbar, bytes)

        # cta_layout and cta_coord represents the cta coord in a cga, setting cta_layout to 1 means we use 1x1 cga
        # tma partition only partition the first mode of smem/gmem tensor into various utmaldg instructions
        # this means we need to manually group/pack the smem/gmem modes (that we want the tma to load) into mode 0
        # then tAsA and tAgA will have shape [TMA_atom, rest]
        tAsA, tAgA = cpasync.tma_partition(
            tma_load, (0), cute.make_layout((1)), 
            cute.group_modes(sA, 0, 2), 
            cute.group_modes(gmem_tensor_coord_cta, 0, 2))

        cute.copy(tma_load, tAgA, tAsA, tma_bar_ptr = tma_load_mbar, mcast_mask = None)

        # wait for the current phase to flip, i.e. the arrival of all data
        cute.arch.mbarrier_wait(tma_load_mbar, current_phase)
        #with cute.arch.elect_one():
        #    if block_idx == 1:
        #        cute.print_tensor(sA)

        # phase is flipped between 0 and 1
        current_phase = 1- current_phase

@cute.jit
def cute_host_load(a: cute.Tensor):

    # Print hello world from host code
    cute.printf("hello world")
    
    # Initialize CUDA context for launching a kernel with error checking
    # We make context initialization explicit to allow users to control the context creation 
    # and avoid potential issues with multiple contexts
    cutlass.cuda.initialize_cuda_context()

    M, K = a.shape

    gmem_layout = cute.make_layout((M, K), stride=(K, 1))
    gmem_tensor = cute.make_tensor(a.iterator, gmem_layout)
    smem_layout = cute.make_layout((CTA_M, CTA_K), stride=(CTA_K, 1))
    # smem layout need to be a composed layout, here we pass in swizzle(0,4,3)
    # which represents non swizzle
    smem_layout = cute.make_composed_layout(cute.make_swizzle(0,4,3), 0, smem_layout)
    # tma_tensor is the arithmetic tuple tracking the coordinate of the gmem tensor to load from
    tma_load, tma_tensor = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), gmem_tensor, smem_layout, smem_layout.shape)

    # need to explicitly calculate smem usage to pass in to kernel launch
    bytes = cute.size_in_bytes(gmem_tensor.element_type, smem_layout) + 8

    # Launch kernel
    cute_tma_load_kernel(tma_load, tma_tensor, gmem_tensor, smem_layout).launch(
        grid=(M // CTA_M, 1, 1),   # Single thread block
        block=(32, 1, 1),  # One warp (32 threads) per thread block
        smem = bytes
    )

if __name__ == "__main__":
    M, K = 2112 * 4, 7168 // 4
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    #for k in range(K // CTA_K):
    #    print(k, a[64:128, k * CTA_K: (k + 1) * CTA_K])
    cute_host_load(from_dlpack(a))
    torch.cuda.synchronize()