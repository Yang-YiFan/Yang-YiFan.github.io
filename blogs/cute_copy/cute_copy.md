---
layout: default
---

# 4 Ways to Do CuTe Copy

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

## 0. Introduction

The famous Chinese writer [Lu Xun](https://en.wikipedia.org/wiki/Lu_Xun) (鲁迅) once said: 
> There are 4 ways to conduct a CuTe copy.

In this blog, I will list the 4 ways to do CuTe copy following Lu Xun's guidance.
Joke aside, through the examples, I'm hoping to convey the following messages:
- There are many equivalent ways to partition a tensor into tiles in CuTe. Depending on the specific use case, one may be preferred over the other.
- When you are writing a CuTe kernel, it is still fundamentally SIMT (you have many threads running in parallel).

## 1. Working Example

We are going to do something extremely simple: load a `8x128` GMEM tensor (row major) into RF using 128 threads and 1 CTA.
The way we partition the tensor to each thread is shown in the following figure:

![partition](./figures/partition.png)

Essentially each thread loads a `1x8` tile of the GMEM tensor.
In the following sections, we will show 4 ways to do this simple task.
And we will use [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html) to write the kernel.

## 2. Approach 1: Traditional CUDA C++ Style

## 3. Approach 2: Using CuTe Layout Algebra

## 4. Approach 3: Using TV-Layout + Composition

## 5. Approach 4: Using TV-Layout + TiledCopy


## 6. Summary

