---
layout: default
---

# CuTe Layout and Tensor Part 2

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

## 0. Introduction

In part 2 of CuTe layout and Tensor series, I'll cover some advanced topics of CuTe layout algebra and concepts, namely `Composition`, `Complement`, `Division`, `TV Layout`. The core of it is how CuTe does the tiling of a tensor with layout algebra. We will walk you through an example of how we use layout algebra to tile a tensor and then introduce all the rest of the concepts with that context.

The code of this blog is available at [here](https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_layout2/code). If you haven't read part 1 of this series, you can refer to [here](../cute_layout/cute_layout.md) as it's a *prerequisite* for this blog.

## Composition

Composition is the core enabler of the tiling mechanism in CuTe.
So before jumping into our tiling example, we need to understand how composition works.

## A Tiling Example in CuTe

tile + repetition1

tile + repetition2



## TV Layout

tile is V_layout

repetition is T_layout

the mma image you normally see is the inverse of the TV layout


## Complement and Division

repetition 1 is complement

division is tile + repetition1



## Additional References

- [CuTe doc 02_layout_algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
- [CuTe doc 03_tensor](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/03_tensor.md)
- [Cris's GPU mode lecture](https://www.youtube.com/watch?v=vzUhbDO_0qk)


Tile and repetition

size / A_layout = complement(A_layout)
A_layout * complement(A_layout) = size

A_layout compose B_layout = a tile of A_layout with shape of B_layout

logical_divide() reshapes tensor with (tiler, rest) layout.

zipped divide is a reorder rank of logical_divide() output, local_tile = zipped_divide + indexing

tensor indexing doesn't change the layout of sub-tile, only change its initial offset.

smem_layout compose TV_layout then index by t, is the data needed for each thread

T_layout is A_layout, V_layout is complement(A_layout)
T_layout * V_layout = size
but V_layout doesn't have to be the same as complement(A_layout), it can be another layout



is complement deterministic? why can't we have multiple complement layouts?

tiled copy do (tid, vid) -> coord

co-tile copy takes 
- (tid, vid) -> addr
- coord -> addr
and tries to compute (tid, vid) -> coord
