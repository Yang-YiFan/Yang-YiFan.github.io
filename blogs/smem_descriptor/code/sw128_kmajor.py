"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 3.5.

Every number is derived with REAL CuTe DSL layout algebra (cute.tile_to_shape /
zipped_divide / recast_layout / logical_divide / get / crd2idx) — no hand-typed
strides. The five steps below mirror Sec. 3.5 one-for-one:

  Step 1  SMEM tile layout            tile_to_shape(swizzle_atom, (128,128))
  Step 2  MMA partition               zipped_divide(smem, (MMA_M=64, MMA_K=16))
  Step 3  recast to uint128_t         recast_layout(128, 16, <partitioned>)
  Step 4  logical_divide -> canonical logical_divide(sub_u, (ChunkM=8, ChunkK=1))
  Step 5  descriptor advance          stride of the (Num_MMA_M, Num_MMA_K) modes

The asserts check the CuTe-derived numbers against the values quoted in Sec. 3.5.

Notes:
  * The descriptor swizzle (S<3,4,3>) only sets `layout_type` + the within-atom
    XOR; it doesn't change SBO/LBO/advance strides. So the analysis runs on the
    *plain* (un-swizzled) layout, which exposes the strides cute.get needs.
  * cute.crd2idx folds to a Python int only for Python-int coords; the DSL turns
    a `range()` loop *inside* @cute.jit into a dynamic loop. So the table loop
    lives in a plain helper (not @cute.jit), where counters stay Python ints.

Run (CuTe DSL venv; no GPU needed — layouts are static/compile-time):
    python sw128_kmajor.py
"""

import cutlass
import cutlass.cute as cute


# --- plain helpers (ordinary Python loops; called from within the jit ctx) -
def stride_at(layout, mode):
    """Scalar stride of one layout mode, as a Python int."""
    return int(cute.get(layout, mode=mode).stride)


def advance_table(adv, n_m, n_k, unit_bytes):
    """Byte offset of every MMA subtile, via cute.crd2idx on the real advance
    layout. `adv` is in u128 units here, so unit_bytes = 16."""
    return [[int(cute.crd2idx((m, k), adv)) * unit_bytes for k in range(n_k)]
            for m in range(n_m)]


def report(smem, zd, rc, sub_u, adv, canon):
    sbo = stride_at(canon, [0, 1])     # canonical ((ChunkM,RestM),(ChunkK,RestK))
    lbo = stride_at(canon, [1, 1])     #            ((8,    SBO  ),(1,     LBO  ))
    print("=== K-major Swizzle 128B, SMEM M=128 K=128 bf16, tcgen05.mma M=64 ===")
    print("  (all layouts produced by CuTe layout algebra)")
    print()
    print("Step 1  SMEM tile (bf16)      :", smem)
    print("Step 2  MMA partition (bf16)  :", zd, "   ((MMA_M,MMA_K),Num_MMA_M,Num_MMA_K)")
    print("Step 3  recast to u128        :", rc)
    print("          per-subtile sub_u   :", sub_u)
    print("          advance (u128)      :", adv, "   (Num_MMA_M, Num_MMA_K)")
    print("Step 4  logical_divide canon  :", canon,
          "   ((ChunkM,RestM),(ChunkK,RestK))")
    print()
    print("descriptor for subtile (m_tile=0, k_tile=0):")
    print("  layout_type   = 2  (SWIZZLE_128B)")
    print("  start_address = 0x00000          (= 0 B from base)")
    print(f"  SBO = stride<0,1> = {sbo:>3} u128 = {sbo*16:>5} B   (M-atom stride)")
    print(f"  LBO = stride<1,1> = {lbo:>3} u128 = {lbo*16:>5} B   (intra-K step)")
    print()
    tbl = advance_table(adv, 2, 8, 16)
    print("Step 5  byte offsets from desc[0,0] (start_address advance):")
    print("        " + "".join(f"k={k:<6}" for k in range(8)))
    for m in range(2):
        print(f"  m={m}  " + " ".join(f"{tbl[m][k]:<7}" for k in range(8)))

    # --- self-check against Sec. 3.5 --------------------------------------
    # Step 1 strides: within-M=64, M-atom=512, within-K=1, K-atom=8192
    assert (stride_at(smem, [0, 0]), stride_at(smem, [0, 1]),
            stride_at(smem, [1, 0]), stride_at(smem, [1, 1])) == (64, 512, 1, 8192)
    # Step 4 SBO/LBO
    assert sbo * 16 == 1024 and lbo * 16 == 16
    # Step 5 advance table
    assert tbl[0] == [0, 32, 64, 96, 16384, 16416, 16448, 16480]
    assert tbl[1] == [8192, 8224, 8256, 8288, 24576, 24608, 24640, 24672]
    print("\nAll asserts passed — CuTe layout algebra matches Sec. 3.5.")


@cute.jit
def analyze():
    # Step 1: K-major Swizzle 128B atom (M=8 × K=64 bf16) -> SMEM tile,
    # stacking atoms along M first (order=(0,1)) — maximizes the TMA box.
    atom = cute.make_layout((8, 64), stride=(64, 1))           # (8,64):(64,1)        bf16
    smem = cute.tile_to_shape(atom, (128, 128), order=(0, 1))
    #   smem = ((8,16),(64,2)):((64,512),(1,8192))   ((AtomM,RestM),(AtomK,RestK)) bf16

    # Step 2: MMA partition, MMA subtile = (MMA_M=64, MMA_K=16) elements.
    zd  = cute.zipped_divide(smem, (64, 16))
    #   zd  = ((64,16),(2,(4,2))):((64,1),(4096,(16,8192)))  ((MMA_M,MMA_K),Num_MMA_M,Num_MMA_K)

    # Step 3: recast the whole partitioned layout to u128 (8 bf16 per u128).
    rc = cute.recast_layout(128, 16, zd)
    #   rc    = ((64,2),(2,(4,2))):((8,1),(512,(2,1024)))    u128
    sub_u = cute.get(rc, mode=[0])   # per-MMA-subtile, u128
    #   sub_u = (64,2):(8,1)
    adv   = cute.get(rc, mode=[1])   # inter-subtile advance, u128
    #   adv   = (2,(4,2)):(512,(2,1024))                     (Num_MMA_M, Num_MMA_K)

    # Step 4: logical_divide the u128 subtile by the u128 chunk (ChunkM=8, ChunkK=1).
    # (Sec 3.5) cute.logical_divide((MMA_M=64, MMA_K=2), (ChunkM=8, ChunkK=1))
    #   template : ((ChunkM, RestM), (ChunkK, RestK)) : ((8, SBO), (1, LBO))
    #   concrete : ((8,     8    ), (1,     2    )) : ((8, 64 ), (0, 1  ))
    canon = cute.logical_divide(sub_u, (cute.make_layout(8), cute.make_layout(1)))
    #   SBO = RestM stride = stride<0,1> = 64  (= 1024 B, M-atom stride)
    #   LBO = RestK stride = stride<1,1> = 1   (=   16 B, intra-K step)
    #   ChunkM=8 = rows/atom (fixed by B128); ChunkK=1 = 1 u128 (size-1 -> CuTe
    #   emits a don't-care stride 0 in slot stride<1,0>).

    report(smem, zd, rc, sub_u, adv, canon)


if __name__ == "__main__":
    analyze()
