"""
Companion script for blogs/smem_descriptor/smem_descriptor.md, Sec. 2.

Builds the (M=128, K=128) bf16 MN-major Swizzle 128B SMEM layout and analyzes
the tcgen05.mma SMEM descriptor for each of the 16 subtiles
(2 M-tiles x 8 K-tiles, MMA atom M=64, K=16).

Run:
    python sw128_mnmajor.py

See sw128_kmajor.py for an explanation of the methodology. The only thing that
changes for MN-major is the swizzle atom shape (M=64 x K=8 instead of M=8 x K=64)
and which canonical-layout stride slots become SBO and LBO.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Layout:
    shape: Tuple
    stride: Tuple

    def __str__(self):
        return f"{self.shape}:{self.stride}"


# ----------------------------------------------------------------------------
# Layout_MN_SW128_Atom<bf16> from cute/atom/mma_traits_sm90_gmma.hpp:64
#   Sw<3,4,3> o Layout<Shape<_64, _8>, Stride<_1, _64>>    (in bf16 elements)
# Atom is M=64 (contiguous) x K=8.
# ----------------------------------------------------------------------------
ATOM_M = 64
ATOM_K = 8
ATOM_STRIDE = (1, ATOM_M)         # MN-major: stride 1 along M, stride 64 along K

# CuTe DSL equivalent (compare with sm100_utils.make_smem_layout_a for MN-major):
#   sw_atom = cute.make_composed_layout(
#       cute.make_swizzle(3, 4, 3), 0,
#       cute.make_layout((ATOM_M, ATOM_K), stride=ATOM_STRIDE))

TILE_M = 128
TILE_K = 128
M_ATOMS = TILE_M // ATOM_M    # 2
K_ATOMS = TILE_K // ATOM_K    # 16
ATOM_SIZE_EL = ATOM_M * ATOM_K  # 512 elements = 1024 B

# tile_to_shape with col-major (M-first) atom ordering:
#   M-atom stride = atom_size = 512 elements
#   K-atom stride = M_ATOMS * atom_size = 2 * 512 = 1024 elements
SMEM_LAYOUT = Layout(
    shape =((ATOM_M, M_ATOMS), (ATOM_K, K_ATOMS)),
    stride=((1,      ATOM_SIZE_EL),
            (ATOM_M, M_ATOMS*ATOM_SIZE_EL)),
)
# Concretely: ((64, 2), (8, 16)) : ((1, 512), (64, 1024))   [bf16 elements]


# ----------------------------------------------------------------------------
# Recast to u128 along the contiguous (M) direction.
# ----------------------------------------------------------------------------
U128_PER_ATOM_M = ATOM_M // 8     # 64 / 8 = 8

U128_LAYOUT = Layout(
    shape =((U128_PER_ATOM_M, M_ATOMS), (ATOM_K, K_ATOMS)),
    stride=((1,                U128_PER_ATOM_M * ATOM_K),     # M-atom stride = 8*8 = 64 u128
            (U128_PER_ATOM_M,  M_ATOMS*U128_PER_ATOM_M*ATOM_K)),
)
# Concretely: ((8, 2), (8, 16)) : ((1, 64), (8, 128))   [u128]
# Sanity:
#   M-atom stride = 8 u128 * 8 K-rows = 64 u128 = 1024 B ✓
#   K-atom stride = 2 M-atoms * 64 u128/M-atom = 128 u128 = 2048 B ✓


# ----------------------------------------------------------------------------
# Canonical-layout strides for the *first subtile* (M=64, K=16 MN-major).
#
# u128 layout of the first subtile:
#   shape  ((8, 1), (8, 2))     // ((within-atom M, M-atom-idx-in-subtile=1),
#                              //   (within-atom K, K-atom-idx-in-subtile=2))
#   stride ((1, 64), (8, 128))
#
# make_umma_desc<Major::MN>:
#   canonical_layout = logical_divide(layout, Tile<Layout<_8,_1>, Layout<_8,_1>>)
#   SBO = stride<1,1>(canonical_layout)
#   LBO = stride<0,1>(canonical_layout)
# ----------------------------------------------------------------------------
SUBTILE_M = 64
SUBTILE_K = 16
M_ATOMS_PER_SUBTILE = SUBTILE_M // ATOM_M    # 1
K_ATOMS_PER_SUBTILE = SUBTILE_K // ATOM_K    # 2

FIRST_SUBTILE_U128 = Layout(
    shape =((U128_PER_ATOM_M, M_ATOMS_PER_SUBTILE),
            (ATOM_K,          K_ATOMS_PER_SUBTILE)),
    stride=((1,                U128_PER_ATOM_M * ATOM_K),     # 64 u128
            (U128_PER_ATOM_M,  M_ATOMS * U128_PER_ATOM_M * ATOM_K)),  # 8, 128 u128
)

# Major::MN: SBO = stride<1,1>, LBO = stride<0,1>
SBO_U128 = FIRST_SUBTILE_U128.stride[1][1]
LBO_U128 = FIRST_SUBTILE_U128.stride[0][1]
SBO = SBO_U128 * 16
LBO = LBO_U128 * 16


# ----------------------------------------------------------------------------
# Descriptor advance.
#
# Outer-mode strides for the partitioned tensor (in u128):
#   M-tile: shape 2, stride = M-atom-stride = 64 u128 (= 1024 B == LBO)
#   K-tile: shape 8, stride = 2 K-atoms = 256 u128 (= 4096 B == 2*SBO)
# Both are flat affine -- no hierarchical mode like the K-major case.
# ----------------------------------------------------------------------------
NUM_M_TILES = TILE_M // SUBTILE_M    # 2
NUM_K_TILES = TILE_K // SUBTILE_K    # 8

M_TILE_STRIDE_U128 = M_ATOMS_PER_SUBTILE * (U128_PER_ATOM_M * ATOM_K)   # 64
K_TILE_STRIDE_U128 = K_ATOMS_PER_SUBTILE * (M_ATOMS * U128_PER_ATOM_M * ATOM_K)  # 256


def subtile_offset_bytes(m_tile: int, k_tile: int) -> int:
    return (m_tile * M_TILE_STRIDE_U128 + k_tile * K_TILE_STRIDE_U128) * 16


# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("MN-major Swizzle 128B, SMEM tile M=128, K=128, bf16, tcgen05.mma M=64")
    print("=" * 78)
    print()
    print(f"  Swizzle 128B atom (bf16):      M={ATOM_M}, K={ATOM_K}, size=1024 B")
    print(f"  SMEM tile:                     M={TILE_M}, K={TILE_K}, "
          f"= {M_ATOMS}x{K_ATOMS} atoms = {M_ATOMS*K_ATOMS} atoms = "
          f"{M_ATOMS*K_ATOMS*1024} B")
    print(f"  tcgen05.mma instruction:       M={SUBTILE_M}, K={SUBTILE_K} (MN-major)")
    print(f"  partitioning:                  "
          f"{NUM_M_TILES} M-tiles x {NUM_K_TILES} K-tiles "
          f"= {NUM_M_TILES*NUM_K_TILES} tcgen05.mma instructions")
    print()
    print(f"  SMEM layout (bf16 elements):  {SMEM_LAYOUT}")
    print(f"  recast to u128:                "
          f"{U128_LAYOUT}")
    print()
    print(f"  canonical-layout strides for *first subtile* (u128):")
    print(f"    stride<1,1> = SBO = {SBO_U128:>3} u128 = {SBO} B   (K-atom stride)")
    print(f"    stride<0,1> = LBO = {LBO_U128:>3} u128 = {LBO} B   (M-atom stride)")
    print()
    print("  descriptor for subtile (m_tile=0, k_tile=0):")
    print(f"    layout_type   = 2  (SWIZZLE_128B)")
    print(f"    start_address = 0x00000          (= 0 B from base)")
    print(f"    SBO           = {SBO_U128:>3} u128         (= {SBO} B)")
    print(f"    LBO           = {LBO_U128:>3} u128         (= {LBO} B)")
    print()
    print("  byte offsets from desc[0, 0]:")
    print("            " + "".join(f"k={k:<5} " for k in range(NUM_K_TILES)))
    for m in range(NUM_M_TILES):
        row = f"   m={m}    "
        for k in range(NUM_K_TILES):
            row += f"{subtile_offset_bytes(m, k):>5} B "
        print(row)
    print()


if __name__ == "__main__":
    main()
