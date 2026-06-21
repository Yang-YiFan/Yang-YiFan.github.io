---
layout: default
---

# Multi-head Latent Attention (MLA) Absorb and Non-Absorb Comparison

*Disclaimer: The content of this blog reflects my personal experiences and opinions while learning GPU programming in my own time. All information presented is publicly available and does not represent the views or positions of NVIDIA Corporation or any of its affiliates.*

## Introduction

Multi-head Latent Attention (MLA) is the attention mechanism used in [DeepSeek V2](https://arxiv.org/abs/2405.04434), [DeepSeek V3](https://arxiv.org/abs/2412.19437v1) and [DeepSeek R1](https://arxiv.org/abs/2501.12948).
It is known for its high KV cache compression ratio for efficient LLM decode.
However, state of the art libraries like [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/551) uses two different (but mathematically equivalent) implementations of MLA during inference.
One for prefill phase (called non-absorb) and one for decode phase (called absorb).
In this blog, I'll explain why we need different implementations of MLA for prefill and decode.

The derivation is done during helpful discussion with [Jenny Huang](https://hqjenny.com/).

**TL,DR:** 
Non-absorb computes on uncompressed KV cache and has fewer flops (head dim 128 MHA). Absorb computes on compressed KV cache and has more flops (head dim 512 MQA).
The difference between non-absorb and absorb is **the order in which you compute a series of gemm in MLA**.
Prefill is compute-bound so it prefers non-absorb (fewer flops).
Decode is memory-bound so it prefers absorb (smaller KV cache).





## What is MLA?

## MLA Non-Absorb

## MLA Absorb

## Absorb vs Non-Absorb

## Summary

- Non-absorb is head dim 128 MHA. The KV cache is uncompressed.
- Absorb is head dim 512 MQA. The KV cache is compressed.

Prefill and decode have different characteristics hence prefer different implementations.

- Prefill is compute-bound. Non-absorb has lower flops than absorb. So it chooses non-absorb despite its higher memory footprint of KV cache.
- Decode is memory-bound. Absorb has smaller KV cachethan non-absorb. So it chooses absorb despite its higher flops.

## Additional References