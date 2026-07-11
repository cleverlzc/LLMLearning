# vLLM 技术博客系列

> 本系列文章深入解读 vLLM 项目的架构设计与实现细节，从宏观到微观，由浅入深。vLLM 是一个高性能 LLM 推理与服务引擎，以 PagedAttention 为核心创新，通过连续批处理、CUDA Graphs、推测解码等技术，实现了业界领先的推理吞吐与延迟表现。

基于 vLLM v0.24.0版本源码，2026.6.27 13:00从主干拉取。

### 文章目录

1. [一文读懂 vLLM V1：从请求到 Token 的千里之行](./01-architecture.md) — *架构概览篇*
2. [PagedAttention 详解：让 KV Cache 从"一维长条"变"乐高拼图"](./02-paged-attention.md) — *核心模块深潜篇*
3. [调度器与连续批处理：当请求像潮水般涌来](./03-scheduler.md) — *核心模块深潜篇*
4. [vLLM 设计模式解析：从源码中学习可复用的架构智慧](./04-design-patterns.md) — *设计模式篇*
5. [CUDA Graphs 与编译优化：用编译换速度，用融合换吞吐](./05-cuda-graphs.md) — *性能优化篇*
6. [vLLM 部署踩坑实录：那些文档没告诉你的事](./06-pitfalls.md) — *踩坑避坑篇*
7. [推测解码：让大模型学会"预判对手的下一步棋"](./07-speculative-decoding.md) — *核心概念深潜篇*
8. [分布式推理：从单卡到千卡的并行之道——一张灶台炒不完所有菜](./08-distributed-inference.md) — *核心概念深潜篇*
9. [一次推理的完整旅程：从 HTTP 请求到 Token 流出](./09-inference-journey.md) — *全景概览篇*
10. [注意力后端详解：FlashAttention vs FlashInfer 的选择之道](./10-attention-backends.md) — *核心技术详解篇*
11. [量化与精度：FP8/INT8/AWQ/GPTQ 全景对比](./11-quantization.md) — *核心技术详解篇*
12. [多模态与 LoRA：让推理引擎长出"第三只眼"](./12-multimodal-lora.md) — *核心技术详解篇*
13. [Offload Memory 详解：GPU 装不下？借 CPU 的"仓库"用用](./13-offload-memory.md) — *核心概念深潜篇*
14. [Mooncake KV Connector 详解：让 KV Cache 在 GPU 之间"瞬移"](./14-mooncake-kv-connector.md) — *核心概念深潜篇*
15. [写在理解 vLLM 量化之前，从 FP32 到 FP4：大模型精度演进的"瘦身简史"](./15-precision-evolution.md) — *核心概念深潜篇*

注解：建议先看15再看11，笔者是在看11的时候有些困惑，遂补充了15。

---

*更新时间: 2026-07-09*
