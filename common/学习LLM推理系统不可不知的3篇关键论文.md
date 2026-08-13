# 学习LLM推理系统不可不知的3篇关键论文

> **PagedAttention、FlashAttention、DistServe这三篇论文，笔者认为可以理解为 LLM 推理系统的"原理三件套"。读懂它们，再看 vLLM、SGLang、TensorRT-LLM 等框架的架构、设计及源码实现，就有了清晰的认知坐标系。毕竟，凡事要从第一性原理出发嘛。**

剥开所有复杂工程，核心主要是三个瓶颈（一台计算机不论是硬件还是软件，到最后无非就是计算、存储，加上网络）：**算得慢、存得浪费、调度低效**。


| 瓶颈 | 论文 | 核心思想 | 核心价值               |
|------|------|---------|--------------------|
| 算力 | FlashAttention | IO感知分块计算 | 慢在搬运，不在计算          |
| 显存 | PagedAttention | KV Block按需分配 | 存储利用率从20%拉到96%     |
| 调度 | DistServe | Prefill/Decode分离 | workload不同，解耦开独立优化 |

## 论文速查表

| 维度 | FlashAttention | PagedAttention | DistServe |
|------|---------------|----------------|------------|
| 会议 | NeurIPS 2022 / ICLR 2024 | SOSP 2023 | OSDI 2024 |
| 团队 | 斯坦福（Tri Dao） | UC Berkeley | 北大 + UCSD |
| 解决瓶颈 | 算力（IO 带宽） | 显存（碎片浪费） | 调度（PD 混跑） |
| 核心思想 | IO 感知分块计算 | KV Block 按需分配 | Prefill/Decode 分离 |
| 关键指标 | 显存 O(N²)→O(N) | 利用率 20%→96% | Goodput 最高 4.48x |


---

## 一、PagedAttention（vLLM）：显存利用率从 20% 到 96%，吞吐提升数倍

| 项目 | 内容 |
|------|------|
| **论文标题** | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| **arXiv链接** | https://arxiv.org/abs/2309.06180 |
| **作者** | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica（UC 伯克利团队） |
| **会议** | SOSP 2023（操作系统顶会） |
| **开源项目** | https://github.com/vllm-project/vllm |


**关键发现，使用KV cache做推理时的一些特点：**
- 随着prompt数量变多和序列变长，KV cache也变大，对GPU显存造成压力。
- 由于输出的序列长度无法预先知道，所以我们很难提前为KV cache量身定制存储空间。

因此，如何优化KV cache，节省显存，提高推理吞吐量，就成了LLM推理框架需要解决的重点问题。

### 1.1 解决的核心痛点（LLM 自回归推理场景）


**内存碎片化严重：**

- **连续预分配内存**：为每个请求一次性分配一整块连续 GPU 显存存放 KV；请求实际 token 长度短，造成**内部碎片**；请求长短不一产生**外部碎片**。
- **实测**：传统系统 KV Cache 内存真实利用率仅 **20%~38%**，大量显存闲置，无法同时调度更多并发请求。

**无法 KV 缓存共享：**

- 并行采样（Parallel Sampling）多条输出序列、束搜索（Beam Search）、多轮会话共用前缀 Prompt （Shared Prefix）时，不能复用 KV，重复存储。

**根据大模型解码方式的实际需求场景，补充说明：**
- Parallel Sampling：我给模型发送一个请求，希望它对prompt做续写，并给出三种不同的回答。我们管这个场景叫parallel sampling。在这个场景中，我们可以将prompt复制3次后拼接成1个batch喂给模型，让它做推理。但我们也需注意到，这种方式会产生prompt部分KV cache的重复存储。
- Beam Search：束搜索，这是LLM常用的decode策略之一，即在每个decode阶段，我不是只产生1个token，而是产生top k个token（这里k也被称为束宽）。top k个token必然对应着此刻的top k个序列。我把这top k个序列喂给模型，假设词表的大小为|V|，那么在下一时刻，我就要在k*|V|个候选者中再选出top k，以此类推。不难想象每一时刻我把top k序列喂给模型时，它们的前置token中有大量的KV cache是重复的。
- Shared prefix：在某些大模型中，或者当前市面上几乎全部的AI Agent CLI，所有请求可能都会共享一个前置信息（比如system prompt: “假设你是一个有帮助的AI助手...."），多轮对话场景、AI Coding长程任务场景，这些共同的前置信息天然是重复的，没有必要重复存储KV cache。

### 1.2 核心设计

1. 将 KV Cache 切分为**固定大小 KV Block**（比如每块存放 16/32 个 token 的 K、V）；
2. 一条序列逻辑上连续的 KV，物理上可以分散存放在任意不连续 Block；
3. 通过 **Block Table** 做间接寻址，注意力内核遍历块表分批读取 KV 完成计算；
4. **按需分配 Block**，消除内外碎片，KV 内存利用率提升至 **96% 以上**。

**效果：**

通过非连续分块和块表映射机制，减少内存碎片，复用KV Cache，存储空间利用率从20%提升到96%。

同等延迟约束下，相比 FasterTransformer、Orca：
- 整体吞吐提升 2～4 倍；
- 序列越长、模型越大、Beam Search 场景增益越明显。

>代价：非连续内存 + 查表带来少量 kernel 开销（单注意力内核延迟上升约 20%~26%），但允许并发批大小大幅提升，整体服务吞吐收益远高于开销。

---

## 二、FlashAttention（三代演进：FA1 / FA2 / FA3）：慢在搬运，不在计算

### 2.1 FlashAttention-1

| 项目 | 内容 |
|------|------|
| **论文标题** | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| **arXiv链接** | https://arxiv.org/abs/2205.14135 |
| **作者** | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré（Tri Dao 等为斯坦福，Atri Rudra 为布法罗分校 SUNY） |
| **会议** | NeurIPS 2022 |
| **开源仓库** | https://github.com/Dao-AILab/flash-attention |

#### 2.1.1 解决的核心痛点

标准 Self-Attention 会生成 **N×N 注意力分数矩阵**存放在 HBM（GPU 全局显存）：

- **计算复杂度和显存复杂度 O(N²)**：随着输入序列的变长，将给计算和存储带来极大的压力。
- **内存带宽瓶颈**：大量在 HBM ↔ SM（Streaming Multiprocessors，流式多处理器，可以将其理解成 GPU 的计算单元）共享内存 SRAM 来回搬运数据，GPU 被内存带宽瓶颈限制（内存受限 workload）。

> **关键发现**：计算慢的卡点不在运算能力，而是在**读写速度**上。因此通过降低对显存（HBM）的访问次数来加快整体运算速度，这种方法又被称为 **IO-Awareness**。

- 计算限制（math-bound）：大矩阵乘法（N和d都非常大）、通道数很大的卷积运算。相对而言，**读得快，算得慢**。
- 内存限制（memory-bound）：逐点运算操作。例如：激活函数、dropout、mask、softmax、BN和LN。相对而言，**算得快，读得慢**。

简单一句话：硬件是有算力上限的，硬件也有带宽上限，计算和存储谁达到上限，即拖慢一个完整请求推理输出时间的关键阶段，谁就是瓶颈，也就是限制xx-bound。我们从系统优化的角度，就要抓主要矛盾，就去优化谁（这个过程是动态的，螺旋上升的）。

- 硬件算力上限：指的是一个计算平台倾尽全力每秒钟所能完成的浮点运算数。单位是 FLOPS or FLOP/s。
- 硬件带宽上限：指的是一个计算平台倾尽全力每秒所能完成的内存交换量。单位是Byte/s。

**核心优化点：**

- Kernel 融合（Kernel Fusion）
- 尽可能利用起 SRAM，以减少数据读取时间

**补充说明：**

**Kernel融合**：由于从显存读一次数据是耗时的，因此在SRAM存储容许的情况下，**能合并的计算我们尽量合并在一起，避免重复从显存读取数据**。

举例来说，我现在要做计算A和计算B。在老方法里，我做完A后得到一个中间结果，写回显存，然后再从显存中把这个结果加载到SRAM，做计算B。但是现在我发现SRAM完全有能力存下我的中间结果，那我就可以把A和B放在一起做了，这样就能节省很多读取时间，我们管这样的操作叫kernel融合。

回顾一下**GPU的计算流程**：将数据从显存（HBM）加载至on-chip的SRAM中，然后由SM读取并进行计算。计算结果再通过SRAM返回给显存。我们知道显存的带宽相比SRAM要小的多，读一次数据是很费时的，但是SRAM存储又太小，装不下太多数据。所以我们就以SRAM的存储为上限，**尽量保证每次加载数据都把SRAM给打满，节省数据读取时间**。

#### 2.1.2 三大核心思想

**① IO 感知分块计算（Tiling）**

- 把 Q/K/V 切分成小块（tile），保证块大小能放进片上 SRAM；
- 全程不在 HBM 落地完整 N×N 注意力矩阵。

**② Online Softmax（在线 Softmax）**

- 迭代式增量计算 softmax，不需要一次性载入全部 score；
- 支持流式累加归一化。

**③ 前向保存归一化因子，反向重计算**

- 前向只存少量 logsumexp 系数；
- 反向过程在片上重算 attention 分数，避免保存巨大注意力矩阵。

**效果：**

- Memory Efficicent，节省显存：显存复杂度降到 **O(N)**
- Fast（with IO-Awareness），计算快：大幅减少 HBM 读写次数（通过降低对显存HBM的访问次数来加快整体运算速度）
- Exact Attention，精准注意力：“稀疏attention”的方法虽然能减少计算量，但算出来的结果并不完全等同于标准attention下的结果。但是Flash Attention却做到了完全等同于标准attention的实现方式。

---

### 2.2 FlashAttention-2

| 项目 | 内容 |
|------|------|
| **论文标题** | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（ICLR 标题：Improving the Efficiency of Attention in Large Language Models） |
| **作者** | Tri Dao |
| **arXiv链接** | https://tridao.me/publications/flash2/flash2.pdf |
| **会议** | ICLR 2024 |

#### 2.2.1 FA2 对比 FA1 的关键改进

| 改进点 | 说明 |
|--------|------|
| **调换循环维度，支持序列长度并行** | FA1 外层循环遍历 Q 块；FA2 外层遍历 K/V 块，可以同时并行处理多个 Q 块，提升低 batch 场景 GPU 占用率。 |
| **优化 Warp 内任务划分** | Warp 是 NVIDIA GPU 上最小的调度单元，每个 warp 中一般有 32 个 thread。消除 warp 之间大量共享内存同步开销，降低通信损耗。 |
| **精简 Online Softmax 冗余缩放运算** | 去掉多余浮点操作。 |
| **更好支持 GQA/MQA** | 适配 LLaMA 主流分组注意力。 |

**实测效果：**

- A100 上相比 FA1 **提速 ~2 倍**
- GPU 算力利用率由 **25%–40% 提升至 50%–73%**

---

### 2.3 FlashAttention-3

| 项目 | 内容 |
|------|------|
| **论文标题** | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| **arXiv链接** | https://arxiv.org/abs/2407.08608 |

#### 2.3.1 FA3 定位

专门针对 **NVIDIA Hopper 架构（H100）**，挖掘 TMA、异步 Tensor Core、FP8 硬件能力：

| 技术 | 说明 |
|------|------|
| **Warp-Specialization 流水线** | 数据搬运 warp 与计算 warp 分离，重叠通信与计算 |
| **异步交织 GEMM 与 Softmax 运算** | 掩盖非矩阵运算延迟 |
| **原生支持 FP8 低精度注意力** | 同时控制数值误差 |

**实测效果（H100）：**

- FP16 相比 FA2 **提速 1.5~2×**
- 峰值算力 **740 TFLOPS**


### 概念区分：
- FlashAttention：训练阶段算子优化，减少 HBM 访存，优化 Attention 计算；
- PagedAttention：推理阶段 KV Cache 内存管理机制，改变 KV 存储布局，复用KV，配套定制 CUDA 注意力内核。

---

## 三、DistServe：挤在一起相互拖累，把两个阶段拆到不同的 GPU 实例池，各自独立优化

| 项目 | 内容 |
|------|------|
| **论文标题** | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving |
| **arXiv链接** | https://arxiv.org/abs/2401.09670 |
| **作者** | Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang（北大 + UCSD） |
| **会议** | USENIX OSDI 2024（系统顶会） |
| **开源代码** | https://github.com/LLMServe/DistServe |

### 3.1 核心价值

证明了 **PD 分离各自优化**的价值，提升了整个推理系统的吞吐量，降低了 TTFT 和 TPOT 的时延。

### 3.2 架构设计

| 组件 | 职责 | 优化目标 |
|------|------|----------|
| **Prefill Pool** | 只负责处理输入 prompt，产出 KV Cache | 压低 **TTFT**（Time To First Token） |
| **Decode Pool** | 接收 Prefill 传来的 KV Cache，持续生成 token | 压低 **TPOT**（Time Per Output Token） |

### 3.3 请求流程

```
请求进入 Controller 路由器
    ↓
路由至 Prefill 实例执行前向计算，生成 KV Cache
    ↓
KV Cache 网络传输到 Decode 实例
    ↓
Decode 持续自回归生成 token
```

> **代价**：跨 GPU 传输 KV Cache；论文论证：现代高速互联（NVLink / 高速以太网）下通信开销可控，收益远大于传输损耗。

### 3.4 在线调度与实例放置策略

- **节点内优先同机部署**：利用 NVLink 降低 KV 传输开销
- **负载均衡路由**：动态分发请求到空闲 Prefill 节点
- **KV Cache 传输流水线**：隐藏通信延迟

### 3.5 主要实验结论

对比基线 vLLM（相比原始不支持 PD 分离的版本，现PD分离方案已证明收益，已成熟支持），在保证 SLO 达标（>90% 请求满足延迟）：

| 场景 | Goodput 提升 |
|------|-------------|
| 对话场景 | **2.0 ~ 3.41×** |
| 代码补全 | **3.2×** |
| 文本摘要 | **最高 4.48×** |
| 请求吞吐 | **最大提升 7.4 倍** |

也可以固定负载，显著收紧延迟指标。

### 3.6 重要贡献：
**DistServe 开创了 Prefill-Decode 分离式推理 流派**，后续大量工作基于该范式：
- Splitwise：异构加速器上的分离部署
- BiScale：面向节能的 Phase-aware 分离调度
- NVIDIA Dynamo 也借鉴 PD 分离思想

## 三篇论文串起来：推理系统的三层进化

把三篇论文放在一起，你会发现它们恰好构成了 LLM 推理系统从底层到上层的完整进化路径：

**第一层 · 算力优化**（FlashAttention）

解决"注意力计算太慢"——不是算力不够，是数据搬运太慢。用 IO 感知分块计算，把显存复杂度从 O(N²) 降到 O(N)。

**第二层 · 显存管理**（PagedAttention）

解决"KV Cache 浪费显存"——不是显存不够，是分配方式太粗。用 KV Block 按需分配，把利用率从 20% 拉到 96%。

**第三层 · 调度架构**（DistServe）

解决"Prefill 和 Decode 互相拖累"——不是 GPU 不够快，是两个阶段特性不合。用 PD 物理分离，让各自跑在最优配置上。

一层解决一层的问题，每一层都解决了重大严重瓶颈：没有 FlashAttention 的 IO 优化，注意力计算就是瓶颈；没有 PagedAttention 的显存管理，KV Cache 就浪费严重；没有 DistServe 的调度分离，两个阶段就互相拖累。


### 扩展阅读：
- 图解大模型计算加速系列之：vLLM核心技术PagedAttention原理：https://zhuanlan.zhihu.com/p/691038809
- 图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑：https://zhuanlan.zhihu.com/p/669926191
- CUDA编程入门极简教程：https://zhuanlan.zhihu.com/p/34587739

如果觉得有用，转发给身边正在学 LLM 推理的朋友。
