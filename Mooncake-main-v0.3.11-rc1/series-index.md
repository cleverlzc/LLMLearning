# Mooncake 技术博客系列

> 本系列文章深入解读 Mooncake 项目的架构设计与实现细节，从宏观到微观，由浅入深。Mooncake 是一个以 KVCache 为中心的解耦架构，用于大规模 LLM 推理与服务。它通过分离 Prefill 与 Decode 集群，并利用 GPU 集群中闲置的 CPU、DRAM 和 SSD 资源构建分布式 KV Cache 池，在真实负载下实现了 75% 的请求吞吐提升。Mooncake 荣获 FAST 2025 最佳论文奖。

基于 Mooncake 主干代码，版本v0.3.11-rc1，2026.7.13 分析。

### 文章目录

0. [RoCE 演进史：RDMA 如何从"贵族专享"变成"AI 基建标配"](./00-roce-evolution.md) — *前置知识篇*
1. [一文读懂 Mooncake：让 KV Cache 在 GPU 之间"瞬移"的解耦架构](./01-architecture.md) — *架构概览篇*
2. [Transfer Engine 详解：数据高速通道的"立交桥"设计](./02-transfer-engine.md) — *核心模块深潜篇*
3. [RDMA 传输：让数据绕过 CPU 的"直达快线"](./03-rdma-transport.md) — *核心技术详解篇*
4. [拓扑感知路由：当集群有十张网卡，谁走哪条路？](./04-topology-aware.md) — *核心技术详解篇*
5. [Mooncake Store 详解：分布式 KV Cache 的"中央仓储"系统](./05-mooncake-store.md) — *核心模块深潜篇*
6. [PD 解耦服务：让 Prefill 和 Decode 各自为战](./06-pd-disaggregation.md) — *核心概念深潜篇*
7. [Expert Parallel 详解：当 MoE 模型的专家们需要"跨车间协作"](./07-expert-parallel.md) — *核心模块深潜篇*
8. [Process Group 后端：给 PyTorch 装上"弹性骨架"](./08-process-group.md) — *核心模块深潜篇*
9. [P2P Store 详解：一火传千灯的模型权重分发](./09-p2p-store.md) — *核心概念深潜篇*
10. [生态集成：Mooncake 如何融入 vLLM、SGLang 和 TensorRT-LLM](./10-ecosystem.md) — *生态实战篇*
11. [Mooncake 部署踩坑实录：那些文档没告诉你的事](./11-deployment-pitfalls.md) — *踩坑避坑篇*
12. [分层存储：KV Cache 的"热货架与冷库"经济学](./12-tiered-storage.md) — *核心技术详解篇*
13. [RDMA/RoCE 深潜：从驱动、协议到硬件的完整技术图谱](./rdma-roce-deep-dive.md) — *技术专题篇*
14. [硬件交互：KV Cache 如何在 GPU、DRAM 与 SSD 之间"搬家"](./14-hardware-interaction.md) — *核心技术详解篇*
15. [性能调优：给数据流水线"疏通河道"的实战指南](./15-performance-tuning.md) — *实战调优篇*
16. [KV Cache 全景：从 Attention 公式到 SSD 磁盘上的字节，一条数据的六次变身](./16-kv-cache-anatomy.md) — *核心概念深潜篇*
17. [安全实践：六道信任边界的攻防审视](./17-security-practice.md) — *安全防御篇*
18. [测试实践：1,900+ 测试用例背后的分层防线](./18-testing-practice.md) — *测试实践篇*
19. [部署实践：从裸机到云原生的全栈交付](./19-deployment-practice.md) — *部署实践篇*
20. [元数据与数据流：Master 不碰数据，Transfer Engine 不碰元数据](./20-metadata-and-dataflow.md) — *核心概念深潜篇*
21. [数据传输三代演进：从 CPU 搬砖到 GPU 直连](./21-three-generations-transfer.md) — *核心技术详解篇*
22. [Count-Min Sketch：16KB 内存如何管住百万 KV Cache 的"热度账本"](./22-count-min-sketch.md) — *核心技术详解篇*
23. [控制平面与数据平面分离：智能控制器决定去向，数据字节直通目的地](./23-control-data-plane.md) — *架构设计模式篇*
24. [四个工程技巧：让数据高速公路"会思考"](./24-engineering-tricks.md) — *工程模式篇*
25. [当故障发生时：水密舱室、心跳探针与弹性恢复](./25-fault-tolerance.md) — *故障容错篇*
26. [Mooncake 关键几问：架构、传输与容错的 18 个灵魂拷问](./26-key-questions.md) — *问答篇*

---

*更新时间: 2026-08-01*
