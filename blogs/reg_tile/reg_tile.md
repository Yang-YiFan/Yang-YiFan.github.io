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
- First achieve the peak compute throughput for baseline.
- Reduce the amount of computation: such as sparsity, or using a more compute efficient algorithm, etc.
- Increase the compute throughput: such as increasing frequency, putting more cores, etc.

I argue **without careful thinking it's not easy to achieve the peak compute throughput for a compute-bound application especially on the GPUs**. Often times your can't even reach the compute roofline. The reason is that the simple machine model that we often use in the roofline model to reason about performance is not informative enough. It can't give us clear/systematic guidance on what optimizations we should do. With a more realistic machine model, we can reason about the various optimization we apply to achieve peak compute throughput from first principle.

In this blog, I'll use matrix multiplication on CUDA cores as an example to illustrate how to achieve theoretical peak compute throughput. **In short, the memory system needs to deliver operands to the compute units at a rate that matches the compute units' consumption rate, using careful tiling at every level of the memory hierarchy to achieve bandwidth amplification.** Then I'll briefly touch upon some alternative ways to achieve this bandwidth amplification.

This blog will be centered around **how to achieve throughput matching rather than latency hiding across pipeline stages for a SIMT program**. Even though the conventional GPU SIMT programming (non tensor core) education we receive (myself included) teaches us the goal is to achieve full (memory) latency hiding. I argue it's not the best way to think about it. Memory latency in particular is a function of memory throughput (i.e. $latency = static\_latency + \frac{tile\_size}{memory\_BW}$). So throughput is the first class citizen and latency is a by-product. Therefore the first priority is throughput matching.

## Example Problem: Matrix Multiplication on CUDA Cores

### Motivation

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

### Background

First, let's define the gemm problem and some terminologies. We have two matrices `A` and `B`, and we want to compute the product `C = A * B`. The matrices are of size `[M, K]` and `[K, N]`, and we want to compute the product C of size `[M, N]`. For simplicity, we use `M=N=K=4096` as an example problem size in this blog.

Because the GPU has many SMs (Streaming Multiprocessor), we need to somehow parallelize the gemm problem across all SMs. The way we parallel is part of the `dataflow` (or compute schedule, [ref1](https://people.csail.mit.edu/emer/media/papers/2016.06.isca.eyeriss_architecture.pdf), [ref2](https://csg.csail.mit.edu/6.5930/Lectures/L11.pdf), [ref3](https://yang-yifan.github.io/papers/isca24_trapezoid.pdf)). The most common dataflow people use in mapping gemm to SMs is called `output-stationary (OS) dataflow` (or inner-product (IP) dataflow). This means each SM is responsible for producing a disjoint tile of the output matrix. The tile of output is *stationary* in the SM. Output gets maximum reuse in SM.

The reason why people tend to use OS dataflow is because inter-SM communication is expensive. Suppose we don't use OS dataflow, meaning that multiple SMs are collaboratively producing 1 output tile. They have to communicate and synchronize through global memory (cached in L2) which is extremely slow. So using OS dataflow to avoid expensive inter-SM communication is a common practice. 

This tradeoff might change with the introduction of [distributed shared memory](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) in Hopper that offers faster inter-SM communication within a threadblock cluster. I will explore this in a future blog.

## Simple Machine Model is not Informative

Now that I've defined the gemm problem, let's see how we analyze whether it's a compute-bound or a memory-bound problem on an Nvidia [A100 80GB SXM](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) GPU using the roofline model. 

![simple_machine_model](./simple_model.png)

The vanilla roofline model assumes a simple machine model where we have an off-chip memory (DRAM), an on-chip storage (SRAM) with *infinite* capacity, and the compute units (SM CUDA cores). The off-chip to on-chip memory has a limited bandwidth. On-chip memory to compute unit has infinite bandwidth. And the compute unit has a fixed throughput. Plug in the A100 number we have the simple machine model shown above. 

Note that this is a per-SM slice of the A100 GPU, meaning that the DRAM bandwidth is the bandwidth each SM (108 SMs) can get out of the total DRAM bandwidth (2039 GB/s). $\frac{2039GB/s}{108 \times 1.41GHz} = 13.3B/cycle$. And the compute unit throughput is for 1 SM (i.e. 64 FMA/cycle). Alternatively, you can create a machine model for the entire GPU by aggregating the throughput across all SMs. This is basically equivalent to multiply 108 (number of SMs) to all numbers in our machine model and it won't affect the roofline analysis at all.

Now we calculate the arithmetic intensity of the `4096x4096x4096` (M, N, K) gemm problem with the simple machine model. The compulsory traffic (i.e. infinite on-chip memory) is $4096 \times 4096 \times 3 = 48MB$. The number of FMA (fused multiply and add) is $4096 \times 4096 \times 4096 = 64G$. So the arithmetic intensity of the gemm problem is $\frac{64GFMA}{48MB} = 1365FMA/B$.

The arithmetic intensity of the A100 80GB SXM GPU is $\frac{64FMA/cycle}{13.3B/cycle} = 4.8FMA/B$. Since 1365 > 4.8, we can conclude that this gemm will be *compute-bound* when running on A100, i.e. the goal for the CUDA kernel is to saturate compute throughput. But remember, this conclusion is based on at least two idealized assumptions:
1. Infinite on-chip storage.
2. Infinite on-chip storage to compute unit bandwidth.

With the actual A100 spec that these two resources are not unlimited, are we still compute-bound? It appears so since cuBLAS reaches the compute roofline. Why is our naive implementation only 16\% the speed of cuBLAS? Not knowable. How should we write the kernel such that it fully utilizes compute units similar to cuBLAS? Not knowable. The simple machine model is not informative enough to guide us to write the most optimized kernel to achieve the compute-bound goal.

## A More Realistic Machine Model

This is where the more realistic machine model comes into play. It strikes a balance between abstraction and detail. It models the GPU memory hierarchy faithfully and at the same time is high-level enough to not distract us from hardware implementation details. The more realistic machine model gives us clear guidance on how to write a high performance CUDA core gemm kernel that achieves full compute unit utilization.

![realistic_machine_model](./realistic_model.png)

On the left I show the more realistic machine model. Notice the differences with the simple model are:
1. I explicitly show the 3-level memory hierarchy (DRAM->smem->RF).
2. Each level has limited capacity (assume we can always fit everything in DRAM).
3. Each level has limited bandwidth to the next level.

Note that from RF to CUDA core, the bandwidth is 512 B/cycle. This is the minimum *input data* bandwidth needed to fully saturate the CUDA cores. We have 64 CUDA cores, each requires two 4B input per cycle. In total, the memory hierarchy needs to deliver $64 \times 2 \times 4B/cycle = 512B/cycle$ input data to the CUDA cores in order to fully saturate it. The actual RF to CUDA core bandwidth might be higher but we don't need more than 512 B/cycle input bandwidth. So we use the min of the two numbers.

Why do we only look at input data bandwidth requirement rather than both input and output? This is because we make an assumption of our kernel implementation that we use `output-stationary (OS) dataflow` at all levels of the memory hierarchy. The output only gets to write out to the next level once all the partial results are fully accumulated at the current level. It gets *full reuse*. This means the output bandwidth requirement is minimal and minute compare to input bandwidth requirement because we need to read the same input multiple times. So we can ignore output for simplicity. If we choose not to use OS everywhere, this assumption breaks. And we need to consider both input and output bandwidth requirement.

## The Kernel Optimization Goal

With the more realistic machine model in mind, we have established that the CUDA cores need 512 B/cycle input data from the memory hierarchy to get full compute utilization, which is our ultimate goal. All the input data come from DRAM. Through our kernel optimization, we need to create an illusion that the CUDA cores directly read DRAM at 512 B/cycle bandwidth. 

From the realistic machine model, the input data delivery pipeline is `DRAM->smem->RF->CUDA core`. We want this pipeline throughput to be 512 B/cycle. This means every pipeline stage needs to be 512 B/cycle.

Unfortunately, without any optimization, the `DRAM->smem` bandwidth is only 13.3 B/cycle which limits the overall data delivery pipeline throughput. We need a way to amplify the bandwidth to match the data delivery throughput requirement of the whole pipeline. Similarly, the `smem->RF` is also not big enough (128 B/cycle) to match the overall pipeline throughput requirement (512 B/cycle). We need a way to amplify this bandwidth as well. **In summary, the overarching goal is to amplify the bandwidth between all stages to match the throughput requirement (i.e. CUDA core data delivery bandwidth).**

As I have already alluded, the bandwidth amplification problem of all the stages can be *decoupled*. In our example problem and machine, we can decompose the bandwidth amplification problem into two independent sub-problems:
1. `DRAM->smem` bandwidth amplification (red dotted box).
2. `smem->RF` bandwidth amplification (blue dotted box).

If we succeed in both, we can then deliver 512 B/cycle of input data between all stages. Thus fully utilizing the CUDA core.

## How to Achieve Bandwidth Amplification

There are many ways to achieve bandwidth amplification across the memory hierarchy. I will discuss two popular techniques tiling and multicasting. And briefly touch upon other techniques. These techniques should be agnostic to which levels of memory hierarchy we are at so that we can use different techniques at different levels. We can also combine multiple techniques together at the same level.

### Tiling

#### Shared Memory (smem) Tiling

![smem_tiling](./smem_tiling.png)

#### Register File (RF) Tiling

![rf_tiling](./rf_tiling.png)

#### Putting It All Together

![both_tiling](./both_tiling.png)

### Multicasting

### Other Techniques

Higher Memory Bandwidth

compression/sparsity/fusion

## From Theory to Practice

assume OS dataflow, timeloop

fix dataflow, search hw config

memory latency i.e. pipelining not considered

smem->rf bank conflict, axel's blog

## Summary

## Additional references