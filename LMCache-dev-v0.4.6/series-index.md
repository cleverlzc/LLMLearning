# LMCache 技术博客系列

> 本系列文章深入解读 LMCache 项目的架构设计与实现细节，从宏观到微观，由浅入深。LMCache 是一个面向 LLM 推理的 KV Cache 管理层，将 KV Cache 从临时状态转变为可持久化存储、跨引擎复用、可观测、可变换的 AI 原生知识。

基于 LMCache v0.4.6版本源码，2026.6.9 9:00从主干拉取。

### 文章目录

1. [一文读懂 LMCache：让 KV Cache 从"阅后即焚"变"永久记忆"](./01-architecture.md) — *架构概览篇*
2. [LMCache 存储引擎深潜：从 GPU 到远程的分层存储之旅](./02-storage-engine.md) — *核心模块深潜篇*
3. [LMCache GPU 连接器与内存管理：跨越 GPU 与 CPU 的数据桥梁](./03-gpu-connector-memory.md) — *核心模块深潜篇*
4. [LMCache 设计模式解析：从实战中学习可复用的架构智慧](./04-design-patterns.md) — *设计模式篇*
5. [LMCache 性能优化：用空间换延迟，用异步换吞吐](./05-performance.md) — *性能优化篇*
6. [LMCache 踩坑与避坑：那些文档里没写的暗礁](./06-pitfalls.md) — *踩坑避坑篇*
7. [D2H 与 H2D 深度解析：GPU 和 CPU 之间的数据搬运学](./07-d2h-h2d-explained.md) — *核心概念深潜篇*
8. [NVLink 与 RDMA 深度解析：LLM 推理中跨越节点的数据高速公路](./08-nvlink-rdma-explained.md) — *核心技术深潜篇*
9. [LMCache 存储与传输全景：数据从 GPU 到云端的全链路图谱](./09-storage-transport-panorama.md) — *全景概览篇*
10. [LMCache 存储介质详解：从 GPU 显存到云端 S3 的五层存储体系](./10-storage-media-deep-dive.md) — *核心技术详解篇*
11. [LMCache 传输协议详解：从 PCIe 到 RDMA 的协议栈全景](./11-transfer-protocols-deep-dive.md) — *核心技术详解篇*
12. [LMCache 物理网络详解：从节点内总线到跨机架集群的网络基础设施](./12-physical-network-deep-dive.md) — *核心技术详解篇*

---

*更新时间: 2026-06-15*
