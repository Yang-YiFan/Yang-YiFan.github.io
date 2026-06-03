"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 3.

Every number is derived with REAL CuTe DSL layout algebra — no hand-typed
strides and no re-implemented layout math (we call cute.logical_divide,
cute.zipped_divide, cute.recast_layout, cute.get, cute.crd2idx, cute.size).

For the (M=128, K=128) bf16 K-major Swizzle 128B SMEM tile fed to the M=64
`tcgen05.mma`:

  * SMEM layout                 cute.tile_to_shape(swizzle_atom, ...)
  * uint128 recast              cute.recast_layout(128, 16, ...)
  * per-MMA-subtile + advance   cute.zipped_divide(smem, (64, 16))
  * SBO / LBO                   cute.logical_divide(...) to the canonical UMMA
                                layout, then cute.get(<mode>).stride — exactly
                                what make_umma_desc<Major::K> reads internally
  * advance byte offsets        cute.crd2idx(coord, advance_layout)

The asserts check the CuTe-derived numbers against the values quoted in the blog
text, so this file is self-verifying.

NOTE on two small wrinkles:
  * The descriptor swizzle (S<3,4,3>) only sets `layout_type` + the within-atom
    XOR; it doesn't change SBO/LBO/advance strides. So we run the analysis on
    the *plain* (un-swizzled) layout, which exposes the strides cute.get needs
    (composed/swizzled layouts don't).
  * cute.crd2idx returns a Python int only for Python-int coordinates; the DSL
    turns a `range()` loop *inside* @cute.jit into a dynamic loop (non-foldable
    index). So the table/print loops live in plain helpers (not @cute.jit),
    where the loop counters stay ordinary Python ints.

Run (CuTe DSL venv; no GPU needed — layouts are static/compile-time):
    python sw128_kmajor.py
"""

import cutlass
import cutlass.cute as cute


# --- plain helpers (ordinary Python loops; called from within the jit ctx) -
def stride_at(layout, mode):
    """Scalar stride of one layout mode, as a Python int."""
    return int(cute.get(layout, mode=mode).stride)


def advance_table(adv, n_m, n_k, bpe):
    """Byte offset of every MMA subtile, via cute.crd2idx on the real advance
    layout (coords are Python ints, so crd2idx folds to a Python int)."""
    return [[int(cute.crd2idx((m, k), adv)) * bpe for k in range(n_k)]
            for m in range(n_m)]


def report(smem, u128, sub, adv, canon, bpe):
    sbo = stride_at(canon, [0, 1])     # canonical ((8,n),2):((8,SBO),1)
    lbo = stride_at(canon, [1, 0])
    print("=== K-major Swizzle 128B, SMEM M=128 K=128 bf16, tcgen05.mma M=64 ===")
    print("  (all layouts produced by CuTe layout algebra)")
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
    print(f"  SBO = stride<0,1> = {sbo:>3} u128 = {sbo*16:>5} B   (M-atom stride)")
    print(f"  LBO = stride<1,0> = {lbo:>3} u128 = {lbo*16:>5} B   (intra-K step)")
    print()
    tbl = advance_table(adv, 2, 8, bpe)
    print("byte offsets from desc[0,0] (start_address advance):")
    print("        " + "".join(f"k={k:<6}" for k in range(8)))
    for m in range(2):
        print(f"  m={m}  " + " ".join(f"{tbl[m][k]:<7}" for k in range(8)))

    # --- self-check against the blog text (Sec. 3.3 / 3.4) -----------------
    assert (stride_at(smem, [0, 0]), stride_at(smem, [0, 1]),
            stride_at(smem, [1, 0]), stride_at(smem, [1, 1])) == (64, 512, 1, 8192)
    assert sbo * 16 == 1024 and lbo * 16 == 16
    assert tbl[0] == [0, 32, 64, 96, 16384, 16416, 16448, 16480]
    assert tbl[1] == [8192, 8224, 8256, 8288, 24576, 24608, 24640, 24672]
    print("\nAll asserts passed — CuTe layout algebra matches the blog text.")


@cute.jit
def analyze():
    BPE = 2   # bytes per bf16 element
    # NOTE: the whole pipeline runs in bf16 ELEMENT units and only recasts the
    # *per-subtile* slice to u128 at the very end. We partition (zipped_divide)
    # FIRST, while MMA_K is still 16 elements; the recast (16 elem -> 2 u128)
    # happens afterwards, on the small subtile. The shapes after every step are
    # the real CuTe outputs (the script prints + asserts them).

    # Level 1+2: K-major Swizzle 128B atom (M=8 × K=64 bf16) -> SMEM tile,
    # stacking atoms along M first (order=(0,1)) — maximizes the TMA box.
    atom = cute.make_layout((8, 64), stride=(64, 1))           # (8,64):(64,1)        bf16
    smem = cute.tile_to_shape(atom, (128, 128), order=(0, 1))
    #   smem = ((8,16),(64,2)):((64,512),(1,8192))   [(within-M,M-atom),(within-K,K-atom)] bf16

    # Level 3: MMA partition on the ELEMENT-unit tile, MMA subtile = (M=64, K=16) elements.
    zd  = cute.zipped_divide(smem, (64, 16))
    #   zd  = ((64,16),(2,(4,2))):((64,1),(4096,(16,8192)))    ((subtile),(M-tile,K-tile))
    sub = cute.get(zd, mode=[0])     # one MMA subtile (bf16 elements)
    #   sub = (64,16):(64,1)                                   M=64, K=16 elements
    adv = cute.get(zd, mode=[1])     # inter-subtile advance (bf16 elements)
    #   adv = (2,(4,2)):(4096,(16,8192))                       (M-tile, K-tile) strides

    # Recast the per-subtile slice to u128 (8 bf16 per u128, along contiguous K):
    sub_u = cute.recast_layout(128, 16, sub)
    #   sub_u = (64,2):(8,1)                                   M=64 (flat), K=16 elem -> 2 u128
    # logical_divide rule: a mode of size N divided by a tiler of size T becomes
    # a nested (T, N/T) = (within-tile, #tiles); the #tiles sub-mode's stride is
    # the inter-tile (inter-atom) stride. Tiler here is (8, 2):
    canon = cute.logical_divide(sub_u, (cute.make_layout(8), cute.make_layout(2)))
    #   canonical template (CuTe docs):  ((8, n    ), 2          ) : ((8, SBO), 1      )
    #   logical_divide returns        :  ((8, RestM), (2, RestK) ) : ((8, SBO), (1, ·) )
    #   concrete                      :  ((8, 8    ), (2, 1)     ) : ((8, 64 ), (1, 0) )
    #     8     = SwizzleAtomMN (rows per atom, fixed by the B128 swizzle)
    #     RestM = #M-atoms in subtile = 64/8 = 8  -> stride 64 = SBO = stride<0,1>
    #     2     = MMA_K in u128                    -> stride 1  = LBO = stride<1,0>
    #     RestK = #K-atoms in subtile = 2/2 = 1   (stride 0, degenerate)
    #   so n == RestM, and SBO is simply "the stride of the M-rest sub-mode".

    # whole-tile recast, used ONLY for the "recast to u128" display line below:
    u128 = cute.recast_layout(128, 16, smem)
    #   u128 = ((8,16),(8,2)):((8,64),(1,1024))

    report(smem, u128, sub, adv, canon, BPE)


if __name__ == "__main__":
    analyze()
