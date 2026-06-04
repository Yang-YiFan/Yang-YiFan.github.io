"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 4.5.

Every number is derived with REAL CuTe DSL layout algebra (cute.tile_to_shape /
zipped_divide / recast_layout / logical_divide / get / crd2idx) — no hand-typed
strides. The five steps below mirror Sec. 4.5 one-for-one (which in turn mirror
Sec. 3.5 for the K-major case):

  Step 1  SMEM tile layout            tile_to_shape(swizzle_atom, (128,128), K-first)
  Step 2  MMA partition               zipped_divide(smem, (MMA_M=64, MMA_K=16))
  Step 3  recast to uint128_t         recast_layout(128, 16, <partitioned>)
  Step 4  logical_divide -> canonical logical_divide(sub_u, (AtomM_u128=4, AtomK=8))
  Step 5  descriptor advance          stride of the (Num_MMA_M, Num_MMA_K) modes

For the (M=128, K=128) bf16 MN-major Swizzle 64B SMEM tile fed to the M=64
`tcgen05.mma`. The asserts check the CuTe-derived numbers against Sec. 4.

Two things differ from the K-major case (sw128_kmajor.py):

  1. SWIZZLE / ATOM. The MN-major Swizzle 64B atom is M=32 (contiguous) × K=8,
     so the contiguous dimension is M, not K. The canonical Major::MN reshape
     therefore tiles by the *swizzle atom* (AtomM_u128=4, AtomK=8) and puts
     SBO/LBO in different stride slots than Major::K: LBO = stride<0,1>,
     SBO = stride<1,1>.

  2. ATOM ORDERING. For MN-major the contiguous (swizzled) dimension is M, so to
     maximize the TMA box we stack atoms ALONG K FIRST (tile_to_shape with
     order=(1,0)) — K-atoms become contiguous in SMEM. The K-major script used
     order=(0,1) (M-atoms contiguous). Same rule both times: stack contiguously
     along the *non-swizzled* dimension.

Because the M=64 MMA operand spans 2 MN-atoms (M=32 each) and the K=16 operand
spans 2 K-atoms (K=8 each), BOTH SBO (K-atom stride) and LBO (M-atom stride) are
live, non-degenerate strides — directly parallel to the K-major case.

Notes (same as sw128_kmajor.py):
  * The descriptor swizzle (S<2,4,3> for 64B) only sets `layout_type` + the
    within-atom XOR; it doesn't change SBO/LBO/advance strides. So the analysis
    runs on the *plain* (un-swizzled) layout, which exposes the strides cute.get
    needs.
  * cute.crd2idx folds to a Python int only for Python-int coords; the DSL turns
    a `range()` loop *inside* @cute.jit into a dynamic loop. So the table loop
    lives in a plain helper (not @cute.jit), where counters stay Python ints.

Run (CuTe DSL venv; no GPU needed — layouts are static/compile-time):
    python sw64_mnmajor.py
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
    lbo = stride_at(canon, [0, 1])     # canonical ((AtomM,RestM),(AtomK,RestK))
    sbo = stride_at(canon, [1, 1])     #           ((4,   LBO  ),(8,    SBO  ))
    print("=== MN-major Swizzle 64B, SMEM M=128 K=128 bf16, tcgen05.mma M=64 ===")
    print("  (all layouts produced by CuTe layout algebra; K-first atom order)")
    print()
    print("Step 1  SMEM tile (bf16)      :", smem,
          "   ((AtomM,RestM),(AtomK,RestK))")
    print("Step 2  MMA partition (bf16)  :", zd, "   ((MMA_M,MMA_K),Num_MMA_M,Num_MMA_K)")
    print("Step 3  recast to u128        :", rc)
    print("          per-subtile sub_u   :", sub_u)
    print("          advance (u128)      :", adv, "   (Num_MMA_M, Num_MMA_K)")
    print("Step 4  logical_divide canon  :", canon,
          "   ((AtomM,RestM),(AtomK,RestK))")
    print()
    print("descriptor for subtile (m_tile=0, k_tile=0):")
    print("  layout_type   = 4  (SWIZZLE_64B)")
    print("  start_address = 0x00000          (= 0 B from base)")
    print(f"  SBO = stride<1,1> = {sbo:>3} u128 = {sbo*16:>5} B   (K-atom stride)")
    print(f"  LBO = stride<0,1> = {lbo:>3} u128 = {lbo*16:>5} B   (M-atom stride)")
    print()
    tbl = advance_table(adv, 2, 8, 16)
    print("Step 5  byte offsets from desc[0,0] (start_address advance):")
    print("        " + "".join(f"k={k:<7}" for k in range(8)))
    for m in range(2):
        print(f"  m={m}  " + " ".join(f"{tbl[m][k]:<8}" for k in range(8)))

    # --- self-check against Sec. 4 ----------------------------------------
    # Step 1 strides: within-M=1, M-atom=4096, within-K=32, K-atom=256
    assert (stride_at(smem, [0, 0]), stride_at(smem, [0, 1]),
            stride_at(smem, [1, 0]), stride_at(smem, [1, 1])) == (1, 4096, 32, 256)
    # Step 4 SBO/LBO (both live, non-degenerate)
    assert sbo * 16 == 512 and lbo * 16 == 8192
    # Step 5 advance table
    assert tbl[0] == [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
    assert tbl[1] == [16384, 17408, 18432, 19456, 20480, 21504, 22528, 23552]
    # affine + uniform, but M-tile stride = 16 × K-tile stride (NOT 8×), so the
    # 16 subtiles do NOT collapse into one contiguous run (cf. the B128 MN case).
    assert tbl[1][0] == 16 * tbl[0][1]
    print("\nAll asserts passed — CuTe layout algebra matches Sec. 4.")


@cute.jit
def analyze():
    # Step 1: MN-major Swizzle 64B atom (M=32 contiguous × K=8 bf16) -> SMEM
    # tile, stacking atoms along K first (order=(1,0)) — maximizes the TMA box.
    atom = cute.make_layout((32, 8), stride=(1, 32))           # (32,8):(1,32)        bf16
    smem = cute.tile_to_shape(atom, (128, 128), order=(1, 0))
    #   smem = ((32,4),(8,16)):((1,4096),(32,256))   ((AtomM,RestM),(AtomK,RestK)) bf16
    #     within-M = 1 (M contiguous); M-atom = 4096 (= 8192 B, 16 K-atoms)
    #     within-K = 32 (= 64 B, one atom M-row); K-atom = 256 (= 512 B, one atom)

    # Step 2: MMA partition, MMA subtile = (MMA_M=64, MMA_K=16) elements.
    zd  = cute.zipped_divide(smem, (64, 16))
    #   zd  = (((32,2),16),(2,8)):(((1,4096),32),(8192,512))
    #         ((MMA_M, MMA_K), Num_MMA_M, Num_MMA_K)
    #     MMA_M = (32,2):(1,4096) -> M=64 spans 2 M-atoms (hierarchical)
    #     MMA_K = 16:32           -> K=16 spans 2 K-atoms (coalesced flat)
    #     advance = (2,8):(8192,512) -> M-tile 8192 elem=16384 B, K-tile 512 elem=1024 B

    # Step 3: recast the whole partitioned layout to u128 (8 bf16 per u128).
    rc = cute.recast_layout(128, 16, zd)
    #   rc    = (((4,2),16),(2,8)):(((1,512),4),(1024,64))   u128
    sub_u = cute.get(rc, mode=[0])   # per-MMA-subtile, u128
    #   sub_u = ((4,2),16):((1,512),4)   M=(within-atom 4 u128, 2 M-atoms); K=16:4
    adv   = cute.get(rc, mode=[1])   # inter-subtile advance, u128
    #   adv   = (2,8):(1024,64)          (Num_MMA_M, Num_MMA_K)
    #           M-tile 1024 u128=16384 B, K-tile 64 u128=1024 B

    # Step 4: logical_divide the u128 subtile by the u128 SWIZZLE ATOM
    # (AtomM_u128=4, AtomK=8). For Major::MN the descriptor strides are the
    # inter-*atom* strides, so we tile by the atom (not the 8x16B chunk as in
    # the K-major case).
    #   template : ((AtomM, RestM), (AtomK, RestK)) : ((1, LBO), (4, SBO))
    #   concrete : ((4,     2    ), (8,     2    )) : ((1, 512), (4, 32 ))
    canon = cute.logical_divide(sub_u, (cute.make_layout(4), cute.make_layout(8)))
    #   LBO = RestM stride = stride<0,1> = 512 (= 8192 B, M-atom stride; 2 MN-atoms)
    #   SBO = RestK stride = stride<1,1> =  32 (=  512 B, K-atom stride; 2 K-atoms)
    #   AtomM_u128=4 = atom M-width in u128 (64 B / 16 B); AtomK=8 = atom K-rows.

    report(smem, zd, rc, sub_u, adv, canon)


if __name__ == "__main__":
    analyze()
