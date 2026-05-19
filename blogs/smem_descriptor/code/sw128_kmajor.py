"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 1.

Builds the (M=128, K=128) bf16 K-major Swizzle 128B SMEM layout and analyzes
the tcgen05.mma SMEM descriptor for each of the 16 subtiles
(2 M-tiles x 8 K-tiles, MMA atom M=64, K=16).

Run:
    python sw128_kmajor.py

What the script demonstrates (mirroring Sec. 1.5):

  1. Construct the swizzled SMEM layout the same way CuTe does in
     sm100_utils.make_smem_layout_a (Sw<3,4,3> o ((8, 64) : (64, 1))).
  2. Recast it to uint128_t (1 u128 = 8 bf16) -- this is the layout that
     make_umma_desc operates on.
  3. Read out the canonical-layout strides that become SBO and LBO.
  4. For each (m_tile, k_tile) pair, compute the byte offset that CuTe's
     DescriptorIterator adds to start_address.

I write everything as plain Python on top of small layout helpers so the
script runs anywhere; the CuTe DSL APIs you would call to do this in a
real kernel are noted in comments.
"""

from dataclasses import dataclass
from typing import Tuple


# ----------------------------------------------------------------------------
# Tiny CuTe-style layout helper (host Python).
# In CuTe DSL the equivalent is cute.make_layout((shape...), stride=(...)).
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Flat or hierarchical (shape, stride) pair. All in *element* units."""
    shape: Tuple
    stride: Tuple

    def __str__(self):
        return f"{self.shape}:{self.stride}"


def coord_to_offset(layout: Layout, coord: Tuple) -> int:
    """Apply (shape, stride) to a coordinate. Coordinate is a nested tuple
    matching the hierarchical shape."""
    def go(sh, st, c):
        if isinstance(sh, tuple):
            return sum(go(s, t, x) for s, t, x in zip(sh, st, c))
        return c * st
    return go(layout.shape, layout.stride, coord)


# ----------------------------------------------------------------------------
# The SMEM tile layout, as CuTe builds it.
# ----------------------------------------------------------------------------

# Layout_K_SW128_Atom<bf16> from include/cute/atom/mma_traits_sm90_gmma.hpp:84
#   Sw<3,4,3> o Layout<Shape<_8, _64>, Stride<_64, _1>>    (in bf16 elements)
# The swizzle is a position-dependent XOR applied within each 8x128B atom;
# it does not change shape or stride. So the *non-swizzled* part of the
# layout (which is what determines SBO/LBO/start_address) is:
ATOM_M = 8
ATOM_K = 64
ATOM_SHAPE  = (ATOM_M, ATOM_K)
ATOM_STRIDE = (ATOM_K,   1)        # K-major: stride 64 along M, stride 1 along K

# CuTe DSL equivalent:
#   sw_atom = cute.make_composed_layout(
#       cute.make_swizzle(3, 4, 3), 0,
#       cute.make_layout(ATOM_SHAPE, stride=ATOM_STRIDE))

# Tile to (M=128, K=128) with column-major (M-first) atom order.
# CuTe DSL equivalent:
#   smem_layout = cute.tile_to_shape(sw_atom, (128, 128))
TILE_M = 128
TILE_K = 128
M_ATOMS = TILE_M // ATOM_M    # 16
K_ATOMS = TILE_K // ATOM_K    #  2

# Resulting hierarchical layout, in bf16 element units:
#   shape  ((8, 16), (64, 2))
#   stride ((64, 1024), (1, 16384*0+...))   wait, let me derive this directly:
# Within-atom: M stride 64, K stride 1 (in elements). Atom size = 8*64 = 512 elements
# (= 1024 B).
# Atom-grid (col-major / M-first):
#   M-atom stride = 1 * atom_size = 512 elements
#   K-atom stride = M_ATOMS * atom_size = 16 * 512 = 8192 elements
SMEM_LAYOUT = Layout(
    shape =((ATOM_M, M_ATOMS), (ATOM_K, K_ATOMS)),
    stride=((ATOM_K, ATOM_M*ATOM_K), (1,      M_ATOMS*ATOM_M*ATOM_K)),
)
# Concretely: ((8, 16), (64, 2)) : ((64, 512), (1, 8192))   [elements]
# Multiplying strides by 2 (bytes/bf16) gives bytes.


# ----------------------------------------------------------------------------
# Recast to uint128 (8 bf16 per u128 along the contiguous K dimension).
# CuTe DSL equivalent:
#   u128 = cute.recast_layout(16, sizeof(bf16), smem_layout)
# This divides the contiguous-K element count by 8.
# ----------------------------------------------------------------------------

U128_PER_ATOM_K = ATOM_K // 8     # 64 / 8 = 8

U128_LAYOUT = Layout(
    shape =((ATOM_M, M_ATOMS), (U128_PER_ATOM_K, K_ATOMS)),
    stride=((U128_PER_ATOM_K, M_ATOMS*0+ATOM_M*U128_PER_ATOM_K),
            (1,                M_ATOMS*ATOM_M*U128_PER_ATOM_K)),
)
# Concretely: ((8, 16), (8, 2)) : ((8, 64), (1, 1024))   [u128]
# Sanity: K-atom stride = 16 * 8 = 128 u128? No, 16 M-atoms * 8 u128/atom = 128 u128.
# But we said K-atom-stride in bytes = 16384 B = 1024 u128. So stride should be 1024.
# Let me recompute: M_ATOMS=16, atom_size_in_u128 = 8 (M) * 8 (K) = 64. K-atom-stride
# = 16 atoms * 64 u128/atom = 1024 u128. ✓


# ----------------------------------------------------------------------------
# Canonical-layout strides for the descriptor's *first subtile*.
#
# For tcgen05.mma M=64, K=16 K-major, the first subtile occupies:
#   M = 8 M-atoms (i.e. all of mode-0-outer? no, just 8 of 16 M-atoms)
#   K = 2 u128 within K-atom 0
#
# After cute partitions for the MMA atom, the SMEM tensor that's passed to
# make_umma_desc has *first-subtile* layout in u128 units:
#   shape  ((8, 8), 2)        (within-atom M, M-atom-idx in subtile=8, MMA_K=2)
#   stride ((8, 64), 1)
#
# Then make_umma_desc<Major::K>:
#   canonical_layout = logical_divide(layout, Tile<Layout<_8,_1>, Layout<_2,_1>>)
#   SBO = stride<0,1>(canonical_layout)   --> 64 u128
#   LBO = stride<1,0>(canonical_layout)   --> 1 u128
# ----------------------------------------------------------------------------

FIRST_SUBTILE_U128 = Layout(
    shape =((ATOM_M, TILE_M // (ATOM_M * 2)),    # (8, 8) since subtile spans 8 of 16 M-atoms
            U128_PER_ATOM_K // 4),                # 2 u128 of MMA_K
    stride=((U128_PER_ATOM_K, ATOM_M*U128_PER_ATOM_K),
            1),
)
SBO_U128 = FIRST_SUBTILE_U128.stride[0][1]    # stride<0,1>
LBO_U128 = FIRST_SUBTILE_U128.stride[1]       # stride<1,0> (mode 1 is flat here)

SBO = SBO_U128 * 16
LBO = LBO_U128 * 16


# ----------------------------------------------------------------------------
# Descriptor advance: byte offset from subtile (0, 0) to subtile (m, k).
#
# CuTe DSL equivalent: partition the SMEM tensor for the MMA atom and
# read out the stride of the outer modes (MMA_M_tile, MMA_K_tile). Then
# the DescriptorIterator does `start_address += outer_stride * tile_idx`.
# ----------------------------------------------------------------------------

MMA_M = 64
MMA_K = 16
NUM_M_TILES = TILE_M // MMA_M    # 2
NUM_K_TILES = TILE_K // MMA_K    # 8

# Outer-mode strides in u128 (from the partitioned tensor):
#   M-tile: shape 2, stride 512   (= 8 M-atoms = 8 * 64 u128/atom)
#   K-tile: shape (4, 2), stride (2, 1024)  (4 K-tiles per K-atom, then 2 K-atoms)
M_TILE_STRIDE_U128 = (MMA_M // ATOM_M) * (ATOM_M * U128_PER_ATOM_K)   # 8 * 64 = 512
K_TILE_WITHIN_ATOM_STRIDE_U128 = MMA_K // 8                            # 2
K_ATOM_STRIDE_U128 = M_ATOMS * ATOM_M * U128_PER_ATOM_K                # 1024

def subtile_offset_bytes(m_tile: int, k_tile: int) -> int:
    """Byte offset from desc[0, 0] to desc[m_tile, k_tile]."""
    k_within = k_tile % 4
    k_atom   = k_tile // 4
    off_u128 = (m_tile * M_TILE_STRIDE_U128
              + k_within * K_TILE_WITHIN_ATOM_STRIDE_U128
              + k_atom * K_ATOM_STRIDE_U128)
    return off_u128 * 16


# ----------------------------------------------------------------------------
# Print everything.
# ----------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("K-major Swizzle 128B, SMEM tile M=128, K=128, bf16, tcgen05.mma M=64")
    print("=" * 78)
    print()
    print(f"  Swizzle 128B atom (bf16):      M={ATOM_M}, K={ATOM_K}, size=1024 B")
    print(f"  SMEM tile:                     M={TILE_M}, K={TILE_K}, "
          f"= {M_ATOMS}x{K_ATOMS} atoms = {M_ATOMS*K_ATOMS} atoms = "
          f"{M_ATOMS*K_ATOMS*1024} B")
    print(f"  tcgen05.mma instruction:       M={MMA_M}, K={MMA_K}")
    print(f"  partitioning:                  "
          f"{NUM_M_TILES} M-tiles x {NUM_K_TILES} K-tiles "
          f"= {NUM_M_TILES*NUM_K_TILES} tcgen05.mma instructions")
    print()
    print(f"  SMEM layout (bf16 elements):  {SMEM_LAYOUT}")
    print(f"  recast to u128:                "
          f"{U128_LAYOUT}")
    print()
    print(f"  canonical-layout strides for *first subtile* (u128):")
    print(f"    stride<0,1> = SBO = {SBO_U128} u128 = {SBO} B")
    print(f"    stride<1,0> = LBO = {LBO_U128} u128 = {LBO} B")
    print()
    print("  descriptor for subtile (m_tile=0, k_tile=0):")
    print(f"    layout_type   = 2  (SWIZZLE_128B)")
    print(f"    start_address = 0x00000          (= 0 B from base)")
    print(f"    SBO           = {SBO_U128} u128          (= {SBO} B)")
    print(f"    LBO           = {LBO_U128:>2} u128          (= {LBO:>4} B)")
    print()
    print("  byte offsets from desc[0, 0] (i.e. value of `desc -= start_address[0]`):")
    print("            " + "".join(f"k={k:<5} " for k in range(NUM_K_TILES)))
    for m in range(NUM_M_TILES):
        row = f"   m={m}    "
        for k in range(NUM_K_TILES):
            row += f"{subtile_offset_bytes(m, k):>5} B "
        print(row)
    print()


if __name__ == "__main__":
    main()
