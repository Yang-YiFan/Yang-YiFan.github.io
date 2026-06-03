"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 4.

Like sw128_kmajor.py, every number is derived with REAL CuTe DSL layout algebra
(cute.logical_divide / zipped_divide / recast_layout / get / crd2idx / size) —
no hand-typed strides, no re-implemented layout math. For the (M=128, K=128)
bf16 MN-major Swizzle 128B SMEM tile fed to the M=64 `tcgen05.mma`.

Two things differ from the K-major case:

  1. The swizzle atom is M=64 (contiguous) × K=8, so the canonical Major::MN
     reshape puts SBO/LBO in different stride slots (LBO = stride<0,1>,
     SBO = stride<1,1>).

  2. ATOM ORDERING. For MN-major the contiguous (swizzled) dimension is M, so to
     maximize the TMA box we stack atoms ALONG K FIRST (cute.tile_to_shape with
     order=(1,0)) — K-atoms become contiguous in SMEM. The K-major script used
     order=(0,1) (M-atoms contiguous). Same rule both times: stack contiguously
     along the *non-swizzled* dimension.

A consequence the CuTe algebra makes obvious: the M=64 MMA operand is exactly
ONE MN-atom wide, so the canonical layout's M-atom-count mode has size 1 and
`LBO = stride<0,1> = 0` (degenerate / unused). The 16384 B M-tile *advance* is
carried by `start_address`, not by LBO.

Run (CuTe DSL venv; no GPU needed):
    python sw128_mnmajor.py
"""

import cutlass
import cutlass.cute as cute


# --- plain helpers (see sw128_kmajor.py for why they are not @cute.jit) -----
def stride_at(layout, mode):
    return int(cute.get(layout, mode=mode).stride)


def advance_table(adv, n_m, n_k, bpe):
    return [[int(cute.crd2idx((m, k), adv)) * bpe for k in range(n_k)]
            for m in range(n_m)]


def report(smem, u128, sub, adv, canon, bpe):
    lbo = stride_at(canon, [0, 1])     # canonical ((8,n),(8,k)):((1,LBO),(8,SBO))
    sbo = stride_at(canon, [1, 1])
    print("=== MN-major Swizzle 128B, SMEM M=128 K=128 bf16, tcgen05.mma M=64 ===")
    print("  (all layouts produced by CuTe layout algebra; K-first atom order)")
    print()
    print("SMEM layout (bf16 elements):", smem)
    print("recast to u128            :", u128)
    print("per-MMA-subtile (bf16)    :", sub)
    print("advance modes (bf16)      :", adv, " = (M-tile, K-tile)")
    print("canonical (u128)          :", canon,
          f"   [size={int(cute.size(canon))}]")
    print()
    print("descriptor for subtile (m_tile=0, k_tile=0):")
    print("  layout_type   = 2  (SWIZZLE_128B)")
    print("  start_address = 0x00000          (= 0 B from base)")
    print(f"  SBO = stride<1,1> = {sbo:>4} u128 = {sbo*16:>5} B   (K-atom stride)")
    print(f"  LBO = stride<0,1> = {lbo:>4} u128 = {lbo*16:>5} B   "
          f"(M-atom stride — 0/unused, M=64 is 1 MN-atom)")
    print()
    tbl = advance_table(adv, 2, 8, bpe)
    print("byte offsets from desc[0,0] (start_address advance):")
    print("        " + "".join(f"k={k:<7}" for k in range(8)))
    for m in range(2):
        print(f"  m={m}  " + " ".join(f"{tbl[m][k]:<8}" for k in range(8)))

    # --- self-check against the blog text (Sec. 4.3 / 4.4) -----------------
    assert (stride_at(smem, [0, 0]), stride_at(smem, [0, 1]),
            stride_at(smem, [1, 0]), stride_at(smem, [1, 1])) == (1, 8192, 64, 512)
    assert sbo * 16 == 1024
    assert lbo == 0                                # M=64 == 1 MN-atom -> LBO unused
    assert tbl[0] == [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336]
    assert tbl[1] == [16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720]
    assert tbl[1][0] == 8 * tbl[0][1]              # contiguous tiling
    print("\nAll asserts passed — CuTe layout algebra matches the blog text.")


@cute.jit
def analyze():
    BPE = 2   # bytes per bf16 element
    # As in sw128_kmajor.py: partition (zipped_divide) FIRST in bf16 element
    # units, then recast only the per-subtile slice to u128. (For MN-major the
    # *contiguous* dim is M, so the recast collapses M 64->8 u128; K stays 16.)

    # Level 1+2: MN-major Swizzle 128B atom (M=64 contiguous × K=8) -> SMEM
    # tile, stacking atoms along K first (order=(1,0)) — maximizes the TMA box.
    atom = cute.make_layout((64, 8), stride=(1, 64))           # (64,8):(1,64)        bf16
    smem = cute.tile_to_shape(atom, (128, 128), order=(1, 0))
    #   smem = ((64,2),(8,16)):((1,8192),(64,512))   [(within-M,M-atom),(within-K,K-atom)] bf16

    # Level 3: MMA partition on the ELEMENT-unit tile, MMA subtile = (M=64, K=16) elements.
    zd  = cute.zipped_divide(smem, (64, 16))
    #   zd  = ((64,16),(2,8)):((1,64),(8192,1024))             ((subtile),(M-tile,K-tile))
    sub = cute.get(zd, mode=[0])     # one MMA subtile (bf16 elements)
    #   sub = (64,16):(1,64)                                   M=64, K=16 elements
    adv = cute.get(zd, mode=[1])     # inter-subtile advance (bf16 elements)
    #   adv = (2,8):(8192,1024)                                (M-tile, K-tile) strides

    # Recast the per-subtile slice to u128 (8 bf16 per u128, along contiguous M):
    sub_u = cute.recast_layout(128, 16, sub)
    #   sub_u = (8,16):(1,8)                                   M=64 elem -> 8 u128, K=16 (flat)
    # logical_divide rule (see sw128_kmajor.py): size N ÷ tiler T -> (T, N/T) =
    # (within-tile, #tiles); the #tiles sub-mode's stride is the inter-atom
    # stride. Tiler here is (8, 8) -- note SBO/LBO sit in different slots than K:
    canon = cute.logical_divide(sub_u, (cute.make_layout(8), cute.make_layout(8)))
    #   canonical template (CuTe docs):  ((8, n    ), (8, k    )) : ((1, LBO), (8, SBO))
    #   logical_divide returns        :  ((8, RestM), (8, RestK)) : ((1, LBO), (8, SBO))
    #   concrete                      :  ((8, 1    ), (8, 2    )) : ((1, 0  ), (8, 64 ))
    #     8     = SwizzleAtomMN / SwizzleAtomK (fixed by the B128 swizzle)
    #     RestM = #M-atoms in subtile = 8/8  = 1 -> stride 0  = LBO = stride<0,1>  (1 MN-atom!)
    #     RestK = #K-atoms in subtile = 16/8 = 2 -> stride 64 = SBO = stride<1,1>
    #   LBO is "the stride of the M-rest sub-mode" — and since RestM == 1 (M=64 is
    #   exactly one MN-atom), that stride is 0: LBO is unused.

    # whole-tile recast, used ONLY for the "recast to u128" display line below:
    u128 = cute.recast_layout(128, 16, smem)
    #   u128 = ((8,2),(8,16)):((1,1024),(8,64))

    report(smem, u128, sub, adv, canon, BPE)


if __name__ == "__main__":
    analyze()
