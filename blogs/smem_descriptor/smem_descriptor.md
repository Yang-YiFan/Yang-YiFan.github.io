---
layout: default
---

# Blackwell MMA SMEM Descriptor

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

*Disclaimer2: This blog is written by Claude with heavy heavy human (me) supervision and editing. So the probability of this blog being AI slop is low.*

## 0. Introduction

In my previous blog ([MMA Swizzle Layout](../mma_swizzle/mma_swizzle.md)) I covered the SMEM swizzle layouts the Blackwell tensor core expects.
The natural follow-up question is: once the data is sitting in SMEM in the correct swizzled layout, **how does the tensor core understand the layout?**

Unlike Ampere's `mma` which sources operands from RF (via `ldmatrix`), Hopper's `wgmma` and Blackwell's `tcgen05.mma` source A/B directly from SMEM.
The hardware contract between the kernel and the tensor core is a 64-bit value called the **SMEM matrix descriptor**.
The descriptor tells the tensor core where the operand lives in SMEM and how to walk through it.
It is documented in [Sec. 9.7.16.4.1 of the PTX 9.2 doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-descriptors).

I could not imagine how a human being (even LLM tbh) could understand how to correctly setup the SMEM descriptor by reading that section of the PTX doc.
If you do, by all means please write to me, I would love to learn from you on how to conquer the PTX doc.
Fortunately, CuTe / CUTLASS build and advance this descriptor for you "magically", which is what I rely on when I write Blackwell tensor core kernels.

You may ask, since CuTe abstracts the descriptor away from the user, why bother learning how it works?
Well firstly, it is always fun to understand how it works under the hood and what design choices CuTe made to abstract it away.
Plus one day, if you are doing some custom thing that is not supported by CuTe / CUTLASS, you will need to write the descriptor by hand.
Finally, in this agentic AI era, LLM would need some learning materials to understand how the SMEM descriptor works and I'm *confident* that the PTX doc along is insufficient.


In this blog I'll walk through how the SMEM descriptor is constructed and advanced (across MMA instructions) with two concrete worked examples — an `(M=128, K=128)` bf16 K-major SMEM tile, then the same tile but with MN-major layout, both with Swizzle 128B.
We will use the `M=64` variant of the bf16 `tcgen05.mma` instruction to demonstrate how the descriptor describes the input operand in SMEM needed by each `tcgen05.mma` instruction.

## 1. The SMEM Descriptor

The descriptor bit layout is shown below ([cute/arch/mma_sm100_desc.hpp](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/include/cute/arch/mma_sm100_desc.hpp#L100)):

![descriptor bit layout](./figures/descriptor_bits.png)

The fields that we'll care about in this blog are:

- **`layout_type`** (bits 61–63) — the swizzle mode: `SWIZZLE_NONE=0`, `SWIZZLE_128B=2`, `SWIZZLE_64B=4`, `SWIZZLE_32B=6`. This is the same set of 8 swizzle layouts from the [previous blog](../mma_swizzle/mma_swizzle.md#3-mma-swizzle-layout).
- **`start_address`** (bits 0–13) — the operand's starting byte address in SMEM, divided by 16. So the granularity is **16 B** (one `uint128`) and the field can address up to `2^14 * 16 = 256 KB` of SMEM, comfortably bigger than Blackwell's 232 KB.
- **`leading_byte_offset`** / **LBO** (bits 16–29) — one of the two strides the hardware uses to walk the operand. Also stored in 16-B units. It's the stride along the contiguous dimension (K for K-major, M for MN-major).
- **`stride_byte_offset`** / **SBO** (bits 32–45) — the other stride. Also in 16-B units. It's the stride along the non-contiguous dimension (M for K-major, K for MN-major).

The remaining fields (`base_offset`, `lbo_mode`, `version`) are configuration knobs that don't change across `tcgen05.mma` instructions of a single kernel; I'll leave them alone here.

`SBO` and `LBO` are strides *inside* one `tcgen05.mma` operand; `start_address` is what moves the descriptor *between* operands. To make that concrete we first need to name the four levels of granularity a SMEM tile can be represented as — which is the next section.

## 2. The Big Picture: Four Nested Levels of Granularity

Throughout this blog, four nested levels of "tile" keep showing up. 
They are primitive building blocks (with different sizes and purposes) that a SMEM tile can be represented as.
The table below uses the K-major `(M=128, K=128)` K-major SMEM tile as an example:


| Level | Shape (K-major bf16, Swizzle 128B) | What it is |
|---|---|---|
| **SMEM tile**       | `M=128 × K=128` (= `128 × 256 B`)    | The high-level SMEM region one TMA stages copies from GMEM. |
| **Swizzle atom**    | `8 × 128 B` (= `M=8 × K=64` bf16)    | The building block of the [`Swizzle 128B` layout](../mma_swizzle/mma_swizzle.md#314-k-major-swizzle-128b). **TMA-friendly**: Each TMA instruction loads multiple swizzle atoms at a time. |
| **MMA subtile**     | `M=64 × K=16` (= `64 × 32 B`)         | The A operand of one `tcgen05.mma` instruction. |
| **8×16B chunk**     | `8 × 16 B` (= 8 rows of one `uint128`) | The primitive unit the MMA hardware accesses. Each chunk lives entirely inside one swizzle atom. |

The figure below shows all four levels for a `(M=128, K=128)` K-major Swizzle 128B SMEM tile:

![the four levels](./figures/four_levels.png)

How the four levels interact:

1. The SMEM tile (`(M=128, K=128)`) is the high-level SMEM region one DMA stage copies from GMEM.
2. **Stacking swizzle atoms is how we construct the SMEM tile for TMA.** The SMEM tile is *literally* `tile_to_shape(Layout_K_SW128_Atom, (M=128, K=128))` — a 16×2 grid of `8x128B` swizzle atoms (blue or orange boxes in the figure). A single TMA instruction (box) loads multiple swizzle atoms (stacked along M) at a time.
3. **The MMA consumes the SMEM tile through MMA subtiles**, one `tcgen05.mma` instruction per MMA subtile (red boxes in the figure). For bf16, one `tcgen05.mma` instruction processes an `M=64, K=16` A subtile. And the entire SMEM tile is partitioned into 2x8=16 MMA subtiles. This is often referred to as the MMA partitioned layout `tCsA = ((MMA_M, MMA_K), Num_MMA_M, Num_MMA_K) = ((64, 16), 2, 8)`.
4. **Each MMA subtile is in turn a grid of 8×16B chunks.** The MMA hardware doesn't see swizzle atoms — it accesses SMEM in `8×16B` chunks. One swizzle atom holds 8 chunks; one MMA subtile holds 8x2=16 chunks.

> **The SMEM descriptor (`layout_type`, `SBO`, and `LBO`) describes how the `8x16B` chunks are laid out within one MMA subtile in SMEM.**

> **Each MMA subtile is described by one SMEM descriptor. Between different MMA subtiles, the `8x16B` chunks are laid out in the same way. The only difference is the `start_address` of the subtile. So SMEM descriptor update between MMA subtiles is `start_address` update in the descriptor.**

In the figures throughout the rest of the blog, the conventions are:

- **Swizzle atom**: light fill, thick blue border, with neighboring atoms alternating shades so the grid is visible at a glance. Always labeled with its shape.
- **MMA subtile**: thick red border (no fill) overlaid on top. Always labeled with its shape.
- **8×16B chunk**: thin black border, varied pastel fills.

For MN-major the absolute shapes flip — the swizzle atom is `M=64 × K=8` (`128B × 8`), and the MMA subtile is `M=64 × K=16` (`128B × 16`) — but the four-level structure is identical.

## 3. Worked Example — `(M=128, K=128)` bf16 K-major, Swizzle 128B

### 3.1. The SMEM tile and the swizzle atom

We start with a 32 KB **SMEM tile** of shape `(M=128, K=128)` in bf16, K-major, with K-major Swizzle 128B layout.
This is the high-level SMEM region one CTA stages from GMEM — a representative A operand of a low-batch GEMM kernel (e.g. the one in [TGV GEMM](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/gemm/tgv_gemm.cuh)).

The SMEM tile is built out of **swizzle atoms** stacked together (this is the TMA-friendly view from the [previous blog](../mma_swizzle/mma_swizzle.md#7-swizzle-atom-layout)).
The K-major Swizzle 128B atom is `M=8, K=128B` (or `M=8 × K=64` bf16) — an `8×128B = 1024 B` tile.
Our `(M=128, K=128)` SMEM tile therefore decomposes into **16 M-atoms × 2 K-atoms = 32 swizzle atoms**, stored in SMEM in column-major (M-first) atom order:

![kmajor tile](./figures/kmajor_tile.png)

Atom `(m, k)` sits at SMEM byte offset `m * 1024 + k * 16384` from the base of the tile.

### 3.2. The MMA subtile

The MMA doesn't consume the SMEM tile in one shot — it consumes it one **MMA subtile** at a time, where an MMA subtile is the operand of a single `tcgen05.mma` instruction.

We feed our SMEM tile to the `M=64` variant of the bf16 `tcgen05.mma` instruction.
The CuTe MMA atom is [`SM100_MMA_F16BF16_SS<M=64, N, K=16, K-major, K-major>`](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/include/cute/arch/mma_sm100_umma.hpp), which corresponds to one [`tcgen05.mma.cta_group::1.kind::f16`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma) instruction processing an `M=64, K=16` A subtile per call.
So one MMA subtile here is `M=64 × K=16` bf16 = `64 × 32 B`.

The full SMEM tile partitions into **2 M-tiles × 8 K-tiles = 16 MMA subtiles**, so 16 `tcgen05.mma` instructions are issued per CTA tile.
Each MMA subtile spans **8 M-atoms × 1/4 of a K-atom** of the SMEM tile: along M, `M=64 = 8 × (M=8-per-atom)`; along K, `K=16 = (1/4) × (K=64-per-atom)`, so 4 K-tiles fit inside one K-atom.

![kmajor subtiles](./figures/kmajor_subtiles.png)

The figure above shows three of our four levels at once: the **SMEM tile** is the whole rectangle, the **swizzle atoms** are the small light-blue boxes (the TMA-friendly stacking), and the **MMA subtiles** are the red overlays (the MMA partitioning).
The descriptor speaks in terms of *one* MMA subtile at a time — which we'll unpack into 8×16B chunks next.

### 3.3. The descriptor for the first MMA subtile

To describe the first MMA subtile (the grey one in the figure above) to the tensor core, we build a descriptor with the following fields:

```
layout_type   = 2          (SWIZZLE_128B)
start_address = &smem >> 4 (in 16-B units)
SBO           = 64 u128 = 1024 B
LBO           =  1 u128 =   16 B
```

How do `SBO` and `LBO` describe the layout of this MMA subtile?

The MMA subtile is itself broken down by the tensor core into **8×16B chunks** ([Sec. 2.1 of the previous blog](../mma_swizzle/mma_swizzle.md#21-motivating-example-ampere-mma)) — this is the smallest granularity at which the MMA hardware addresses SMEM.
For K-major, an 8×16B chunk is 8 rows along M × 16 B (= 8 bf16) along K — exactly one `uint128` per row, repeated 8 times along M.
With `M=64, K=16` K-major, our MMA subtile contains `8 (along M) × 2 (along K) = 16` chunks arranged in an 8×2 grid:

![kmajor chunks](./figures/kmajor_chunks.png)

For the tensor core to read all 16 chunks, it needs to know how to step between adjacent chunks in both directions:

- **Stepping +1 along M** (from chunk row *m* to *m+1*): jumps by one M-atom in the underlying SMEM tile, because each M-atom is exactly 8 rows tall. That's the **SBO = 1024 B**.
- **Stepping +1 along K** (from chunk column 0 to chunk column 1): jumps by 16 B (one `uint128`) along the contiguous K dimension. That's the **LBO = 16 B**.

So the absolute SMEM address of chunk `(m, k)` of this MMA subtile is

```
addr(m, k) = start_address + m * SBO + k * LBO + swizzle_XOR(m, k)
```

where `swizzle_XOR(m, k)` is the within-atom XOR pattern that the hardware applies automatically based on `layout_type = SWIZZLE_128B`.

**This is why `start_address`, `layout_type`, `SBO`, and `LBO` are sufficient.**
The `layout_type` fully determines the relative positions of chunks *within* a swizzle atom (no field needed for that).
`SBO` and `LBO` are the only two inter-chunk strides that depend on the user's tile geometry, and they are enough because the MMA subtile is rank-2 (only M and K).
And `start_address` anchors the whole pattern in SMEM.

Note that `SBO`, `LBO`, and `layout_type` are properties of the MMA subtile **shape**; they do **not** change across the 16 `tcgen05.mma` invocations.

### 3.4. The descriptors for the other 15 MMA subtiles — advance via `start_address`

The other 15 MMA subtiles have **the same shape and the same `(SBO, LBO, layout_type)`**.
The only thing that changes is **where they start in SMEM** — i.e. only `start_address`.

So advancing the descriptor from one MMA subtile to the next is just: bump `start_address` by the byte offset between the two. The offsets from MMA subtile `(0, 0)` to MMA subtile `(m_tile, k_tile)` are:

| m_tile \ k_tile |  0 |   1 |   2 |   3 |     4 |     5 |     6 |     7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** |    0 |    32 |    64 |    96 | 16384 | 16416 | 16448 | 16480 |
| **1** | 8192 |  8224 |  8256 |  8288 | 24576 | 24608 | 24640 | 24672 |

![kmajor descriptor advance](./figures/kmajor_advance.png)

Two patterns to notice:

- **Along K** (within a row of the table) the advance is **non-affine**: four small `+32 B` steps within K-atom 0, then a big `+16288 B` jump into K-atom 1, then three more `+32 B` steps.
- **Along M** (going from row 0 to row 1) the advance is a clean `+8192 B` = 8 M-atoms — because each subtile covers 8 M-atoms in M.

Here's the elegant part: **the descriptor advance is implemented as a single `uint64` addition** — `desc += offset`, where `offset` already encodes the non-affine step.
The non-affine-ness lives in the **CuTe layout** of the SMEM tile, not in any special-case descriptor logic.

The descriptor's `start_address` field sits at bits 0–13 of the 64-bit word, so adding `offset` (in 16-B units, e.g. `16384/16 = 1024` for the K-atom jump) directly increments `start_address` without spilling into any other field — `2^14 = 16384` 16-B units is 256 KB, far larger than the worst-case offset in this 32 KB tile.

### 3.5. How CuTe Implements This

Now let's walk through, end-to-end, how the numbers in Sections 3.3 and 3.4 fall out of CuTe's layout machinery. Two CuTe operations do all the work: a **`uint128` recast** and a **`logical_divide`**.

**Step 1: the original SMEM tile layout.** `tile_to_shape(Layout_K_SW128_Atom<bf16>, Shape<_128,_128>)` produces this hierarchical layout (in **bf16 element** units):

```
shape  ((8, 16), (64, 2))
stride ((64, 1024), (1, 8192))
       ^^   ^^^^^   ^   ^^^^
       |     |      |     +-- K-atom stride: 16 atoms × 512 elements = 8192 (= 16384 B)
       |     |      +-------- within-atom K stride: 1 (K is contiguous)
       |     +--------------- M-atom stride:   512 elements          = 1024 B
       +--------------------- within-atom M stride: 64 elements      =  128 B (= one atom-row)
```

The four strides correspond exactly to "step in M inside an atom", "step from one M-atom to the next", "step in K inside an atom", "step from one K-atom to the next".

**Step 2: recast to `uint128_t`.** Each `uint128` packs 8 bf16 along the contiguous K dimension. Strides in the contiguous direction divide by 8; the others stay:

```
shape  ((8, 16), (8, 2))         // K-element count went from 64 → 8 u128
stride ((8, 64), (1, 1024))      // u128 units; strides scaled by 1/8 only along contiguous K
```

After this step, everything is in 16-B units — the same granularity the descriptor's `start_address` uses. SBO and LBO will now come out as clean integers.

**Step 3: slice off the first MMA subtile.** The MMA partition takes the first MMA subtile's worth of data — `M=64 = 8 M-atoms`, `K=2 u128` — giving:

```
shape  ((8, 8), 2)               // (within-atom M, M-atom-idx-in-subtile=8), MMA_K=2 u128
stride ((8, 64), 1)              // unchanged
```

**Step 4: extract SBO and LBO via `logical_divide` to canonical form.** For Major::K B128, CuTe's canonical hierarchical shape is `((8, n), 2) : ((8, SBO), 1)`. The layout we just computed already matches that shape — so `make_umma_desc` simply reads the field values out of specific stride slots:

| canonical slot | value (u128) | descriptor field | bytes |
|---|---:|---|---:|
| `stride<0,0>` |  8 | (implicit — set by `layout_type`) |  128 B |
| `stride<0,1>` | 64 | **SBO** | **1024 B** |
| `stride<1,0>` |  1 | **LBO** |   **16 B** |

That's where the `SBO = 1024 B`, `LBO = 16 B` from Sec. 3.3 come from. `start_address` is then just `cast_smem_ptr_to_uint(&smem) >> 4`, and `layout_type` is `SWIZZLE_128B`.

**Step 5: the descriptor advance.** After the MMA partition, the full partitioned tensor has shape `((MMA_M, MMA_K), MMA_M_tile, MMA_K_tile)`. Looking at the **outer-mode strides** (in u128):

```
MMA_M_tile mode : shape 2,         stride 512                        // 8 M-atoms × 64 u128 = 512 u128 = 8192 B
MMA_K_tile mode : shape (4, 2),    stride (2, 1024)                  // 32 B within K-atom, 16384 B across K-atoms
```

The MMA_K_tile mode is hierarchically `(4, 2):(2, 1024)` — *exactly* the source of the "+32, +32, +32, +32, +16288, +32, +32, +32" non-affine K advance you saw in Sec. 3.4. The MMA_M_tile mode is flat affine — that's the `+8192 B` per M-tile. Neither stride is special-cased; they fall straight out of the layout.

**Materializing one descriptor, not 16.** CuTe's `DescriptorIterator` ([`mma_traits_sm100.hpp:347`](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/include/cute/atom/mma_traits_sm100.hpp#L347)) is a one-liner:

```cpp
struct DescriptorIterator {
  SmemDescriptor desc_;
  template <class Index>
  DescriptorIterator operator+(Index const& offset) const {
    return { desc_ + uint64_t(offset) };       // raw 64-bit add
  }
};
```

`make_umma_desc` is called *once*, on the first MMA subtile, producing one 64-bit value. The other 15 descriptors are never materialized as state — they are produced on demand by `desc_ + offset` with `offset` lifted directly from the partitioned-tensor outer-mode strides at compile time. In practice this means one register holds the base descriptor for the entire kernel, and each `tcgen05.mma` is preceded by at most one `uint64` add of a compile-time-constant immediate.

A runnable script that prints all 16 descriptors and their bit-level breakdown for this configuration is at [`code/sw128_kmajor.py`](./code/sw128_kmajor.py).

```
=== K-major Swizzle 128B, SMEM tile M=128, K=128, bf16, tcgen05.mma M=64 ===
canonical-layout strides used for descriptor:
  SBO = stride<0,1> = 64 u128 = 1024 B
  LBO = stride<1,0> =  1 u128 =   16 B

descriptor for subtile (m_tile=0, k_tile=0):
  layout_type   = 2  (SWIZZLE_128B)
  start_address = 0x00000          (= 0 B from base)
  SBO           = 64 u128          (= 1024 B)
  LBO           =  1 u128          (=   16 B)

byte offsets from desc[0, 0]:
            k=0     k=1     k=2     k=3     k=4     k=5     k=6     k=7
   m=0      0      32      64      96   16384   16416   16448   16480
   m=1   8192    8224    8256    8288   24576   24608   24640   24672
```

## 4. Worked Example — `(M=128, K=128)` bf16 MN-major, Swizzle 128B

Now we'll repeat the exercise for the same `(M=128, K=128)` bf16 tile, but stored **MN-major** (M is the contiguous dimension).
Same four levels of granularity as Sec. 2 — **SMEM tile**, **swizzle atom**, **MMA subtile**, **8×16B chunk** — but the absolute shapes flip. (The K-major version was the worked example in Sec. 3.)

### 4.1. The SMEM tile and the swizzle atom

The MN-major Swizzle 128B atom is `M=128B, K=8` (or `M=64 × K=8` for bf16) — still an `8×128B = 1024 B` slab, but now M is the contiguous dimension within an atom ([Sec. 3.2.4 of the previous blog](../mma_swizzle/mma_swizzle.md#324-mn-major-swizzle-128b)).
Our `(M=128, K=128)` SMEM tile decomposes into **2 M-atoms × 16 K-atoms = 32 swizzle atoms**, stored in column-major (M-first) atom order:

![mnmajor tile](./figures/mnmajor_tile.png)

Atom `(m, k)` sits at SMEM byte offset `m * 1024 + k * 2048` from the base.

### 4.2. The MMA subtile

We feed this MN-major SMEM tile to `SM100_MMA_F16BF16_SS<M=64, N, K=16, MN-major, …>`.
The MMA subtile is again `M=64 × K=16` bf16 — `128B × 16` viewed MN-major (M=64 bf16 contiguous, K=16 strided).
The full SMEM tile partitions into **2 M-tiles × 8 K-tiles = 16 MMA subtiles**, the same count as the K-major case.

The key difference from Sec. 3: the MN-major swizzle atom only holds `M=64` along M and `K=8` along K. An MMA subtile (`M=64 × K=16`) fits exactly within one M-atom and **spans 2 K-atoms** of the SMEM tile:

![mnmajor subtiles](./figures/mnmajor_subtiles.png)

So while a K-major MMA subtile straddled *8 M-atoms* in the SMEM tile (and stayed within 1 K-atom), an MN-major MMA subtile sits in *1 M-atom* but straddles 2 K-atoms.

### 4.3. The descriptor for the first MMA subtile

The descriptor for the first MMA subtile has these field values:

```
layout_type   = 2          (SWIZZLE_128B)
start_address = &smem >> 4
SBO           = 128 u128 = 2048 B
LBO           =  64 u128 = 1024 B
```

For MN-major, an 8×16B chunk is 16 B (= 8 bf16) along the contiguous M direction × 8 rows along K.
With `M=64, K=16` MN-major, the MMA subtile contains `8 (along M) × 2 (along K) = 16` chunks arranged in an 8×2 grid:

![mnmajor chunks](./figures/mnmajor_chunks.png)

The hardware addresses these chunks using two strides:

- **SBO = K-atom stride = 2048 B**. The MMA subtile spans 2 K-atoms of the SMEM tile; SBO is the distance between them. **`SBO` is used *inside* one `tcgen05.mma` invocation** for MN-major — unlike the K-major case where SBO was the M-atom stride.
- **LBO = M-atom stride = 1024 B**. The MMA subtile fits within a single M-atom of the SMEM tile, so `LBO` is *not* used inside one invocation. But — and this is new compared to Sec. 3 — **`LBO` *will* be used to advance from M-tile 0 to M-tile 1**, because those two MMA subtiles live in different M-atoms of the SMEM tile.

**The roles of `SBO` and `LBO` swap between Major::K and Major::MN.**
In Major::K, `SBO` is the M-atom stride and `LBO` is the 16-B intra-K step.
In Major::MN, `LBO` is the M-atom stride and `SBO` is the K-atom stride.
The naming "stride" vs "leading" is anchored to the contiguous (= leading) dimension of the operand: `LBO` is always the inter-atom stride along the leading direction, `SBO` along the strided direction.

As in Sec. 3.3, the absolute address of every chunk in the MMA subtile is fully determined by `start_address + a * SBO + b * LBO + swizzle_XOR`, plus the implicit within-atom layout encoded by `layout_type`.

### 4.4. The descriptors for the other 15 MMA subtiles — uniform advance this time

Each MMA subtile is offset from `(0, 0)` by `m_tile × M-tile-stride + k_tile × K-tile-stride`, with:

- M-tile-stride = `1 M-atom = 1024 B` (this is exactly `LBO`).
- K-tile-stride = `2 K-atoms = 2 × 2048 = 4096 B` (this is exactly `2 × SBO`).

| m_tile \ k_tile |  0 |    1 |    2 |    3 |     4 |     5 |     6 |     7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** |    0 |  4096 |  8192 | 12288 | 16384 | 20480 | 24576 | 28672 |
| **1** | 1024 |  5120 |  9216 | 13312 | 17408 | 21504 | 25600 | 29696 |

![mnmajor descriptor advance](./figures/mnmajor_advance.png)

Contrast this with the K-major case in Sec. 3.4. The K-major advance was non-affine along K because each MMA subtile only covered K=16 = 1/4 of one K-atom, so the K-tile-stride had to "wrap" 4 times before moving to the next K-atom. In MN-major each MMA subtile already covers exactly 2 K-atoms, so the K-tile-stride is one integer number of K-atoms — affine and uniform.

The descriptor advance reduces to a constant immediate that can be hoisted into a uniform register.

### 4.5. How CuTe Implements This

Same recipe as Sec. 3.5 — `uint128` recast → `logical_divide` to a canonical shape → read SBO/LBO out of fixed stride slots, then advance via outer-mode strides. Let's walk through the concrete numbers for MN-major.

**Step 1: original layout.** `tile_to_shape(Layout_MN_SW128_Atom<bf16>, Shape<_128,_128>)` produces (in bf16 element units):

```
shape  ((64, 2), (8, 16))
stride ((1, 512), (64, 1024))
       ^   ^^^   ^^   ^^^^
       |   |     |     +-- K-atom stride: 2 M-atoms × 512 elements = 1024 (= 2048 B)
       |   |     +-------- within-atom K stride: 64 elements (one M-row of atom) = 128 B
       |   +-------------- M-atom stride: 512 elements                            = 1024 B
       +------------------ within-atom M stride: 1 (M is contiguous now)
```

**Step 2: recast to `uint128_t`.** The contiguous direction is now M, so we divide M strides by 8:

```
shape  ((8, 2), (8, 16))
stride ((1, 64), (8, 128))       // u128 units
```

**Step 3: slice off the first MMA subtile** (`M=64 = 1 M-atom`, `K=16 = 2 K-atoms`):

```
shape  ((8, 1), (8, 2))
stride ((1, 64), (8, 128))
```

**Step 4: extract SBO and LBO.** For Major::MN B128, the canonical hierarchical shape is `((8, n), (8, k)) : ((1, LBO), (8, SBO))` — note the different stride slots from Major::K. Reading them off:

| canonical slot | value (u128) | descriptor field | bytes |
|---|---:|---|---:|
| `stride<0,0>` |   1 | (implicit — `layout_type`) |    16 B |
| `stride<0,1>` |  64 | **LBO** | **1024 B**  *(M-atom stride)* |
| `stride<1,0>` |   8 | (implicit — `layout_type`) |   128 B |
| `stride<1,1>` | 128 | **SBO** | **2048 B**  *(K-atom stride)* |

That's where `SBO = 2048 B` and `LBO = 1024 B` from Sec. 4.3 come from. Notice that compared to Sec. 3.5, both `SBO` and `LBO` are now read off the **outer (atom-count)** modes — that's the structural reason for the role-swap discussed in Sec. 4.3.

**Step 5: descriptor advance.** The partitioned-tensor outer-mode strides (in u128):

```
MMA_M_tile mode : shape 2,    stride  64       // 1 M-atom = 64 u128 = 1024 B (= LBO)
MMA_K_tile mode : shape 8,    stride 256       // 2 K-atoms = 256 u128 = 4096 B (= 2 × SBO)
```

Both flat affine this time (no hierarchical mode) — that's the "uniform advance" of Sec. 4.4: `+1024 B` per M-tile, `+4096 B` per K-tile.

**Materializing one descriptor, not 16.** Same as Sec. 3.5: `make_umma_desc` is called once for `(m=0, k=0)`. The `DescriptorIterator::operator+` produces all 15 other descriptors on demand via `base_desc + offset`, with `offset` lifted from the strides above at compile time. The kernel holds exactly one descriptor in a uniform register and bumps it by a small set of compile-time immediates as it walks through the MMA subtiles.

A runnable script for this MN-major case is at [`code/sw128_mnmajor.py`](./code/sw128_mnmajor.py). Sample output:

```
=== MN-major Swizzle 128B, SMEM tile M=128, K=128, bf16, tcgen05.mma M=64 ===
canonical-layout strides used for descriptor:
  SBO = stride<1,1> = 128 u128 = 2048 B
  LBO = stride<0,1> =  64 u128 = 1024 B

descriptor for subtile (m_tile=0, k_tile=0):
  layout_type   = 2  (SWIZZLE_128B)
  start_address = 0x00000          (= 0 B from base)
  SBO           = 128 u128         (= 2048 B)
  LBO           =  64 u128         (= 1024 B)

byte offsets from desc[0, 0]:
            k=0     k=1     k=2     k=3     k=4     k=5     k=6     k=7
   m=0      0    4096    8192   12288   16384   20480   24576   28672
   m=1   1024    5120    9216   13312   17408   21504   25600   29696
```

## 5. Summary

The Blackwell SMEM matrix descriptor packs into 64 bits everything the tensor core needs to fetch one `tcgen05.mma` operand from SMEM.

Throughout this blog we worked with four nested levels of granularity, and each played a specific role in the descriptor story:

| Level | What it is | Where it shows up in the descriptor |
|---|---|---|
| **SMEM tile** (e.g. M=128 × K=128) | The high-level region one CTA stages from GMEM. | Not directly — but its CuTe layout is the input to `make_umma_desc`. |
| **Swizzle atom** (1024 B) | The TMA-friendly building block; the SMEM tile is a stack of swizzle atoms. | Determines `layout_type`; its size determines `SBO` and `LBO` numerically. |
| **MMA subtile** (M=64 × K=16) | What one `tcgen05.mma` instruction consumes. | One descriptor describes one MMA subtile. |
| **8×16B chunk** (16 B per row × 8 rows) | The primitive layout unit the MMA hardware addresses. | `SBO`/`LBO` are the strides *between chunks*; `start_address` anchors them; `layout_type` encodes the swizzle XOR within an atom. |

The four descriptor fields, mapped to which level they speak about:

- **`layout_type`** — picks the swizzle mode, which implicitly nails down the relative positions of all 8×16B chunks inside one swizzle atom. **Constant** for the whole kernel.
- **`SBO`** and **`LBO`** — the two inter-chunk strides *inside* one MMA subtile. CuTe reads them off the SMEM tensor by recasting to `uint128_t` and pulling specific `stride<i,j>` lookups out of a canonical hierarchical reshape. **Constant** across all `tcgen05.mma` invocations of the same shape.
- **`start_address`** — anchors *which* MMA subtile we're talking about. **This is the only field that changes** between `tcgen05.mma` invocations, and the change is a straight `uint64` add of a precomputed offset.

The role of `SBO` vs `LBO` flips between Major::K and Major::MN:

|              | Major::K, B128                       | Major::MN, B128             |
|--------------|---------------------------------------|-----------------------------|
| **SBO**      | M-atom stride (= 1024 B in our tile)  | K-atom stride (= 2048 B)    |
| **LBO**      | 16 B intra-K step (= 1 u128)          | M-atom stride (= 1024 B)    |
| Used **inside** one MMA subtile? | both — SBO steps 8 M-atoms, LBO steps 2 K-halves | only SBO (MMA subtile spans 2 K-atoms); LBO dormant for M=64 |
| Used **between** MMA subtiles?   | yes — both K-tile and M-tile advance via `start_address` | yes — K-tile advance via `2 × SBO`, M-tile advance via `LBO` |

For our `(M=128, K=128)` bf16 B128 SMEM tile with the M=64 `tcgen05.mma`:

- **K-major**: the 16 MMA subtiles fan out in 2 (M-tiles) × 8 (K-tiles). K-tile advance is non-affine (`+32 +32 +32 +32 +16288 +32 +32 +32` bytes); M-tile advance is `+8192` bytes.
- **MN-major**: K-tile advance is uniform `+4096` bytes; M-tile advance is uniform `+1024` bytes.

The non-affine K-major advance is not a special case in CuTe — it falls out of the hierarchical CuTe layout of the K-tile mode of the `uint128`-recasted tensor. The descriptor iterator's `operator+` is just `desc + uint64_t(offset)`. All the cleverness is in picking the canonical layout shape so the right strides pop out of `logical_divide` — once that's done, the descriptor's "advance" is the dumbest possible operation.
