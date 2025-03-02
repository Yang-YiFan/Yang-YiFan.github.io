---
layout: default
---

# DeepSeek成本分析 中文版

最近在知乎、小红书上，关于DeepSeek推理是否能盈利有很多讨论。最近似乎变成了吃瓜为主。在这篇文章中，我还是想回到技术层面，探讨一下DeepSeek的推理成本估算，以及为什么尤老师的计算是不准确的。

## 1. 背景

一些背景资料: 
- [尤老师的成本估算](http://xhslink.com/a/eXuUY2P6VpU6)，好像被删了。。。
- [DeepSeek官方的数字](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- [Lyken的估算](https://zhuanlan.zhihu.com/p/23282743306?utm_psn=1879469595716470338)

这篇文章受到了[Lyken](https://lzhu.me/)文章的启发，我在他文章的基础上进行了一些简化和背景补充，并修改了一些问题。建议先阅读他的文章。感谢Lyken的讨论！然后我会解释为什么尤老师的估算不准确。

## 2. 系统假设和简化

这是这篇文章最重要的部分，只要你理解了我做了什么假设和简化，实际计算就很简单：
- Disaggregation：即做prefill和decode用单独的GPU系统。这简化了我们的估算，我们**只需要考虑decode阶段就可以**，可以忽略prefill阶段的干扰。
- 极致的专家并行 (EP) (比如EP 320) 加attention数据并行(DP): 这是DeepSeek的部署方案 (根据[DeepSeek V3的文章](https://arxiv.org/pdf/2412.19437)和[他们的推理系统文章](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)), 通过极致的专家并行，每个GPU的HBM大部分存储了Multi-head Latent Attention (MLA)的KV cache，权重部分很小。在decode的1个iteration中，每个GPU会加载`所有用户的KV cache`，`MLA权重`，`1个专家（expert）权重`。后两个部分很小，所以我们可以**近似认为decode阶段大部分时间都在加载KV cache**。如果假设在Mixture of Experts (MOE)或MLA中存在Tensor Parallel (TP)会改变我们的估算，为了简化，我们忽略TP。
- 没有HBM容量限制: 这意味着我们的HBM无限大，可以存储任意数量的KV cache (batch size)，使得MOE不会变成memory bound。这是前一个假设的副产品。**足够的batch size**意味着我们的MOE阶段不会memory bound，并且与KV cache加载时间相比，这部分数据搬运时间可以忽略不计。
- 在大batch size下MLA是memory bound: 因为MLA压缩KV cache会带来更多计算，所以这部分我不太确定。我采用了和Lyken一样的假设，如果有问题欢迎大家提建议。如果这个假设没问题，这意味着我们可以**近似认为decode的执行时间就是从HBM加载所有用户的KV cache到GPU的时间**。
- **NVLink/InfiniBand不是瓶颈**: 根据[这个文章](https://zhuanlan.zhihu.com/p/27292649125?utm_psn=1879469993151944398)，网络带宽实际上限制了吞吐量。为了简化（并且我也不懂网络。。。），我们假设我们有足够的网络带宽（特别是在Blackwell NVL72估计没有这个问题）。


## 3. Throughput估算

为了简化，我忽略了成本估算部分，因为一旦我们有了throughput的数字，成本估算就很简单了。让我们先尝试估算一下`tps/GPU`。

我们参考的数字是DeepSeek官方的H800 decode的throughput：`14.8k tps/node`，即`1850 tps/GPU`。

我们使用的GPU是Nvidia H800 SXM来进行估算:
- 3.35TB/s HBM带宽
- 80GB HBM容量

DeepSeek V3/R1的每个用户KV cache大小为$(d_c + d_h^r) * seq\_len * num\_layers$。我们使用bf16来存储KV cache。假设sequence length为5k。所以每个用户的KV cache大小为334MB。

生成每个token需要1次KV cache从片下加载到片上。所以每秒每个GPU可以加载`3.35TB/s * 1s / 334MB = 10030` tokens。所以speed-of-light (SOL) throughput是`10030 tps/GPU`，是DeepSeek测到的数字的5倍多.

注意到这个SOL throughput上限是达不到的，因为我在上面做了很多理想化的假设（特别是足够网络带宽的假设）。我们可以通过更现实的假设来收紧这个上限。再举个例子的话，如果MLA是compute bound，那么每个iteration的运行时间会比从加载KV cache的时间要长，所以能达到的SOL throughput会更低。

其次进一步的优化可以继续提高SOL throughput:
- 使用FP8来存储KV cache，这可以提高SOL throughput两倍.
- 在decode阶段使用MTP (Multi-Token Prediction)，这意味着每次KV cache加载，我们可以生成超过1个token。假设Acceptance Rate (AR)为N，那么SOL throughput可以乘以N.

那我们现在来看看B200的数字:
- 8TB/s HBM带宽
- 180GB HBM容量

按照同样的计算，B200的SOL throughput是`8TB/s * 1s / 334MB = 23952k tps/GPU`。

## 4. 为什么尤老师的估算不准确?

就在我写文章的同时尤老师删除了他的视频。。。根据我的记忆，他说在4-8个节点（32-64个GPU）的情况下，他大概可以实现`1000 tps/system`的吞吐量，换算为`30 tps/GPU`。

尤老师的估算和我的估算的关键区别在于他使用了不同的并行策略假设，即他没有做极致的EP。根据我的理解，SGLang当时支持DP attention + TP MOE。这会带来两个劣势：

### 4.1 HBM容量不再主要用于存储KV cache

回顾一下极致的EP的情况: 在decode的1个iteration中，每个GPU会加载`所有用户的KV cache`，`MLA权重`，`1个expert权重`。`所有用户的KV cache`的大小远大于`MLA权重` + `1个expert权重`。

如果你没有做极致的EP (用了比较小的EP或TP)，在decode的1个iteration中，每个GPU会加载:
- **`HBM容量限制下可以存储的KV cache`**
- `MLA权重`
- **`多个expert权重`**

因为我们的模型没有足够的切分，每个GPU存储的权重占比变多了，我们被迫在每个GPU的每个iteration中加载更多的权重。更糟糕的是，因为HBM需要存储更多的MOE权重，我们留给KV cache的容量被减少了。这意味着不能同时服务更多的用户，有效batch size被减少了。

现在带入具体数字:
- 假设EP8 MOE，总MOE权重 = 612GB，每个GPU = 76.5GB
- Nvidia H200有141GB HBM，所以只有一半的HBM容量用于存储KV cache!
- H200有比H800更高的带宽4.8TB/s.

所以每秒每个GPU可以加载所有的权重和KV cache `4.8TB/s * 1s / 141 GB = 34`次。这意味着可以服务`(141GB - 76.5GB) / 334MB = 193`个用户。所以每秒每个GPU可以生成`34 * 193 = 6562`个token，只有H800 (极致EP)的60%的吞吐量！如果考虑MLA权重，这个数字甚至会更低。即使使用更强大的GPU，如果你的并行策略不好，你也不能达到和弱一点的GPU一样的吞吐量。

这也解释了为什么直接比较[Nvidia TensorRT-LLM的数字](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/media/r1-perf.jpeg)和DeepSeek的数字是不完全公平的。TensorRT-LLM的数字是在8个H200上测到的，就会遇到我们上面讲的HBM容量的问题。（而且TensorRT-LLM还有很多优化还没有上线）

### 4.2 TP MOE效率不高

在MOE中做TP会比EP MOE带来更多的网络（NVLink/InfiniBand）流量。从每个token的角度来看，假设我们用TP32对比EP32。每个token会激活8个专家，并把结果累加起来。所以对于TP32，32个GPU都会发送partial result参与all-reduce。但对于EP32，最多只有8个GPU会参与all-reduce。有4倍的节省。

另外，做了TP以后，每个GPU跑的gemm problem size会变小。GPU在跑小problem size的gemm时，会损失一些效率。

## 5. 结论

- 在极致EP的假设下，DeepSeek H800的SOL throughput是`10k tps/GPU`，是DeepSeek测到的`1850 tps/GPU`的5倍多。
- 通过FP8 KV cache和MTP，SOL throughput可以进一步提高。
- 不同的并行策略之间有明显的性能差异。最公平的比较应该在相同的并行策略下进行。
- 我们已经算出了throughput数字，我们把成本估算留作课后作业。

我可能有一些计算/假设是有错误的，所以各位大神发现任何错误或有任何建议改进估算的方法，请多多指教。
