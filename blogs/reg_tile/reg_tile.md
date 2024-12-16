---
layout: default
---

# How to Make a Compute-bound Problem Actually Compute-bound (WIP)

One of the first thing when a computer architect would do when they want to accelerate an application is to determine whether the application is compute-bound or memory-bound. To do that, they often employ the [roofline model](https://en.wikipedia.org/wiki/Roofline_model).

If the application is deemed memory-bound, then the architect would focus on ways to 
- First achieve the peak memory bandwidth for baseline.
- Reduce memory traffic: such as quantization, sparsity, compression, operator fusion, etc.
- Increase memory bandwidth: such as putting more HBM sites, having more SRAM banks, etc.

It's not hard to achieve peak memory bandwidth for the baseline implementation as long as you maintain enough memory level parallelism (MLP). The memory system does reasonable things saturating itself. And there are numerous research proposals to alleviate the memory bottleneck.

If the application is deemed compute-bound, similarly the architect would focus on ways to
- **First achieve the peak compute throughput for baseline.**
- Reduce the amount of computation: such as sparsity, or using a more compute efficient algorithm, etc.
- Increase the compute throughput: such as increasing frequency, putting more cores, etc.

I argue without careful thinking it's not easy to achieve the peak compute throughput for baseline especially on the GPUs. Often times your can't even reach the compute roofline. The reason is that the simple machine model that we often use in the roofline model to reason about performance is not informative enough. It can't give us clear/systematic guidance on what optimizations we should do. With a more realistic machine model, we can reason about the various optimization we apply to achieve peak compute throughput from first principle.

In this blog, I'll use matrix multiplication on CUDA cores as an example to illustrate how to achieve theoretical peak compute throughput. **In short, the memory system needs to deliver operands to the compute units at a rate that matches the compute units' consumption rate, using careful tiling at every level of the memory hierarchy to achieve bandwidth amplification.** Then I'll briefly touch upon some alternative ways to achieve this bandwidth amplification.

## Example Problem: Matrix Multiplication on CUDA Cores

Many people have attempted to write a CUDA core fp32 gemm kernel that matches cuBLAS's performance. [Simon Boehm's blog](https://siboehm.com/articles/22/CUDA-MMM) is the one I followed. Though his blog is super informative, I feel like the most important optimization that gets you most of the way is buried behind lots of less important optimizations. I tried writing a CUDA core gemm on Nvidia V100 GPU myself and gets 88% of cuBLAS's performance with only two optimizations over naive implementation:
1. Shared memory (smem) tiling.
2. Register File (RF) tiling.

| algorithm           | throughput (TFLOPS) | % of cuBLAS         |
| ------------------- |:-------------------:|:-------------------:|
| Naive               | 1.9                 | 16%                 |
| smem tiling         | 3.7                 | 31%                 |
| RF tiling           | 10.7                | 88%                 |
| cuBLAS              | 12.1                | 100%                |

As you might notice, both optimizations has something to do with *tiling*, but on different levels of the memory hierarchy. Hopefully with the rest of the blog I can show you why tiling itself is enough to achieve (almost) peak compute throughput and how you should choose your tile size. All of my reasoning will be purely *analytical* without any silicon profiling and that's enough for us to achieve peak compute throughput.

First, let's define the gemm problem and some terminologies. We have two matrices `A` and `B`, and we want to compute the product `C = A * B`. The matrices are of size `[M, K]` and `[K, N]`, and we want to compute the product C of size `[M, N]`. 

## Simple Machine Model is not Informative

![simple_machine_model](./simple_model.png)

## A More Realistic Machine Model

![realistic_machine_model](./realistic_model.png)

## How to Deliver Sufficient Operands to the Compute Units

### Tiling

#### Shared Memory (smem) Tiling

![smem_tiling](./smem_tiling.png)

#### Register File (RF) Tiling

![rf_tiling](./rf_tiling.png)

#### Putting It All Together

![both_tiling](./both_tiling.png)

### Higher Memory Bandwidth

### Multicasting

## Summary

## Additional references