---
layout: default
---

# DeepSeek Serving Cost Estimate

Recently there are heated discussions on the Chinese internet about whether DeepSeek will make a profit serving their model with such a low price tag. Unfortunately the discussion has drifted away from technical to other things. In this blog, let's turn the discussion back to technical and I'll try to explain how the calculation is done and why it was done wrongly by some.

## 1. Background

Some background materials: 
- [Original cost estimate by Professor Yang You](http://xhslink.com/a/eXuUY2P6VpU6)
- [DeepSeek's official cost number](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- [Lyken's cost estimate](https://zhuanlan.zhihu.com/p/23282743306?utm_psn=1879469595716470338)

This blog is heavily inspired by [Lyken](https://lzhu.me/)'s estimate and my calculation builds on top of it with certain simplifications and clarifications. I highly recommend checkout his awesome blog first (use a translator if needed). Thanks Lyken for the discussion! Then I'll explain why Professor You's estimate is off.

## 2. System Assumptions and Simplifications

This is the most important part of the blog, the actual calculation is easy as long as you understand the assumptions and simplifications well:
- Disaggregation: i.e. having dedicated pools of GPUs for prefill and decode separately. This simplifies estimation by **focusing our estimation purely on decode phase**.
- Extreme Expert Parallel (EP) (e.g. EP 320) with Data Parallel (DP) attention: this is what DeepSeek is using in practice (according to [DeepSeek V3 paper](https://arxiv.org/pdf/2412.19437) and [their serving system blog](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)), by doing extreme EP, the HBM of each GPU is mostly storing KV cache of Multi-head Latent Attention (MLA), the weight portion is tiny. During 1 iteration of decode, each GPU is loading `KV cache for all user`, `MLA weights`, `1 expert weight`. The portion of the latter two is tiny so we could **approximate the majority of the decode work is loading KV cache**. If we assume there is Tensor Parallel (TP) in Mixture of Experts (MOE) or MLA, that will change the calculation, for simplicity let's ignore TP.
- Not HBM capacity limited: this means out HBM is big enough to store however many KV cache (batch size) we want in order to make MOE not memory bound. This is sort of a by-product of the previous assumption. **Enough batch size** means our MOE stage is not memory bound and is a tiny portion compared to the KV cache loading time.
- MLA with high batch size is memory bound: this is the part I'm least sure about since MLA compresses KV cache at the cost of more compute. I'll go with the same assumption as Lyken and update it later if I'm wrong. If this is true, this means we can **approximate the decode execution time as the time to load KV cache of all users from HBM to on-chip**.
- **NVLink/InfiniBand is not a bottleneck**: according to [this estimate](https://zhuanlan.zhihu.com/p/27292649125?utm_psn=1879469993151944398) the network bandwidth actually limits the throughput. For simplicity, let's assume we have enough network bandwidth (especially with Blackwell NVL72).


## 3. Throughput Estimate

For simplicity, I'll ignore the cost estimation part since it's a straightforward next step if we can obtain the throughput number. Let's just try to get the `tps/GPU` number.

The golden number we are referring to is the DeepSeek official H800 decoding throughput number: `14.8k tps/node`, which is `1850 tps/GPU`.

The spec we are using is Nvidia H800 SXM:
- 3.35TB/s HBM bandwidth
- 80GB HBM capacity

The KV cache size per user for DeepSeek V3/R1 is $(d_c + d_h^r) * seq\_len * num\_layers$. We are using bf16 for KV cache. Assuming 5k sequence length. So the total KV cache size per user is 334MB.

Each token generation is associated with 1 KV cache load. So in 1 second, each GPU can load `3.35TB/s * 1s / 334MB = 10030` tokens. So the speed-of-light (SOL) throughput is `10030 tps/GPU`, >5x of the measured number.

Note that this SOL throughput bound is not achievable in practice because I made too many idealized assumptions above (especially the sufficient network bandwidth assumption). It is possible to tighten it with more realistic assumptions.

Note note that further optimizations could raise the SOL throughput even higher:
- Use FP8 for KV cache, this can raise the SOL throughput by 2x.
- Use MTP (Multi-Token Prediction) for decoding, this means that each iteration loading of the KV cache, we can generate more than 1 token. Assuming an Acceptance Rate (AR) of N, then the SOL throughput can be multiplied by N.

For a bonus, let's plug in the B200 numbers:
- 8TB/s HBM bandwidth
- 180GB HBM capacity

Following the same calculation, the B200 SOL throughput is `8TB/s * 1s / 334MB = 23952k tps/GPU`.

## 4. Why is Professor You's estimate off?

LOL just as I'm writing this blog, Professor You deleted his video. From my recollection, he was saying something like in a 4-8 node (32-64 GPU) setup, he can achieve an aggregate of `1000 tps/system` using SGLang. This translates into `30 tps/GPU`.

The key difference is he is using a different assumption of the parallelization strategy, i.e. he did not assume there is extreme EP. My understanding is SGLang at the time supported DP attention + TP MOE. This will break two benefits of extreme EP:

### 4.1 HBM capacity is no longer mostly used for KV cache

Recall the implications of extreme EP: During 1 iteration of decode, each GPU is loading `KV cache for all user`, `MLA weights`, `1 expert weight`. And the size of `KV cache for all user` >> `MLA weights` + `1 expert weight`.

If you don't have extreme EP to begin with (small EP or TP), during 1 iteration of decode, each GPU is loading:
- **`KV cache for as many users as they can fit in HBM`**
- `MLA weights`
- **`many expert weight`**

Because you don't have enough sharding of the model, you are forced to load more weights per GPU per iteration. To make things worse, because your HBM has to store more MOE weights, your capacity for the KV cache is reduced. This means you can't serve more users in parallel, the effective batch size is reduced.

Plugging in some numbers:
- Assume EP8 MOE, total MOE weights = 612GB, per GPU = 76.5GB
- Nvidia H200 has 141GB HBM, so only half of the HBM capacity is used for KV cache!
- H200 has a higher bandwidth of 4.8TB/s.

So in 1 second, each GPU can load (all the weights + KV cache) `4.8TB/s * 1s / 141 GB = 34` times. That's for `(141GB - 76.5GB) / 334MB = 193` users worth of KV cache. So in 1 second, each GPU can generate `34 * 193 = 6562` tokens, only 60% of the H800 (extreme EP) SOL throughput! This number could be even lower if you consider the MLA weights. Even with a more powerful GPU, if your parallelization strategy is not good, you can't achieve the same throughput as a wimpier GPU.

This is also why comparing the [Nvidia TensorRT-LLM numbers](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/media/r1-perf.jpeg) directly with DeepSeek's number is not a fair game. The TensorRT-LLM numbers are obtained on 8xH200 which will face the exact same capacity issue as we explained above. (Let alone the fact that there are many optimizations Nvidia engineers are working hard on getting it implemented :))

### 4.2 TP MOE is inefficient

Doing TP in MOE will incur a lot more network (NVLink/InfiniBand) traffic compared to EP MOE. Think of it from the per token perspective. Assume we have TP32 vs EP32. Each token will activate 8 experts and reduce the result between them. So for TP32, all 32 GPU will send partial results to participate in all-reduce. But for EP32, at most 8 GPU will participate in all-reduce. A 4x saving.

Also there will be the minor engineering details of TP kernel being less efficient than the EP kernel because after TP, per GPU gemm problem size is smaller. And GPU will lose some efficiency in executing small problem size gemm.

## 5. Conclusion

- With the extreme EP setting along with the other assumptions, the DeepSeek H800 SOL throughput is `10k tps/GPU`, > 5x of the measured `1850 tps/GPU`.
- The SOL throughput can be further improved with FP8 KV cache and MTP.
- There is material performance difference between different parallelization strategies. Comparisons should be done under similar parallelization settings.
- Given the throughput number, we leave the cost estimation as an exercise to the reader.

I might have done some calculations/assumptions wrongly, so please let me know if you find any mistakes or have suggestions on how to improve the estimation.
