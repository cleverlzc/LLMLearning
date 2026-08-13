# 学习LLM推理系统不可不知的算子（operators）知识

在上一篇文章中，我们知道算子（operator）本质上是一个纯粹的数学运算，比如矩阵乘法、Softmax，底层可以进一步融合（kernel fusion），能够高效运行在GPU/NPU上，一种“特殊数学语义特定硬件的原子计算函数”。

LLM大模型的全部计算 = 几十种算子组合 × 重复成千上亿次。让它显得"智能"的，不是单个算子有多复杂，是那些训练出来的超大量的参数（千亿参数，GLM 5.2 7440亿即744B；万亿参数，KIMI K3 2.8万亿即2.8T），通过精密巧妙的算子计算和步骤组织拼成LLM。

因此我们可以认为，算子是拼成LLM的"乐高积木"——每一块做一件简单的事，但拼在一起就能涌现出翻译、写诗、写代码的"智能"。

本文将介绍一次标准的Transformer由哪些算子组成，拆解最常用的算子，以及算子的工程生态，性能实践，算子的未来演进。

---

## 一、算子是什么？

### 1.1 一个直觉类比

想象你在厨房做一道复杂的菜——比如佛跳墙。

这道菜最终很好吃，但它不是一步完成的。它由很多**基本操作**组成：切丝、焯水、爆炒、炖煮、勾芡……每一个基本操作，就是一个"算子"。

- 切丝不会让菜变好吃，但没它不行
- 炖煮也不会让菜变好吃，但没它不行
- **所有基本操作组合在一起**，才成就了最终的佛跳墙

LLM也是一样。一个千亿参数的模型看起来很"智能"，但拆开来看，它的计算就是由一大堆**基础数学运算**组成的。这些基础运算，就是**算子（Operator / Kernel），Kernel是CUDA编程的核心概念**。

### 1.2 严格定义

在深度学习框架和推理引擎的语境下：

**算子 = 一个计算单元，接收一组输入，执行特定的数学运算，输出一组结果。**

这里的输入和输出往往是张量、向量和矩阵，算子就是纯数学数字的各种计算。

用代码的视角理解：

```python
# 这就是一个算子
def matmul(A, B):
    """矩阵乘法算子：输入两个矩阵，输出它们的乘积"""
    return A @ B

# 这也是一个算子
def softmax(x, dim=-1):
    """Softmax算子：输入 logits，输出概率分布"""
    exp_x = torch.exp(x - x.max(dim, keepdim=True).values)
    return exp_x / exp_x.sum(dim, keepdim=True)

# 还是一个算子
def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm算子：对输入做归一化"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return weight * (x - mean) / torch.sqrt(var + eps) + bias
```

**等一下——这不就是个函数吗？为什么要专门起个名字叫"算子"？**

从代码看，算子确实就是一个函数。但"算子"这个名字承载了普通函数没有的几层含义：

| | **普通函数** | **算子（Operator）**                     |
|---|---|--------------------------------------|
| **归属** | 编程语言的概念 | 计算图的概念                               |
| **方向** | 只有前向：输入→输出 | 有前向+反向：正向算结果，反向算梯度                   |
| **可组合性** | 函数调用函数，就是嵌套 | 算子与算子拼成计算图，框架自动管理数据流                 |
| **可优化性** | 编译器最多做内联、循环展开 | 框架可以做**算子融合**（多个算子合成一个Kernel Fusion） |
| **可部署性** | 跑在CPU上 | 可以调度到GPU/NPU/TPU等不同硬件上执行             |

用一个比喻来说：

> **普通函数像"菜谱上的一道步骤"**——写着"小火炖20分钟"，仅此而已。
> **算子像"厨房里一个独立的工位"**——有专门的灶台、专门的厨师、能自动清洗（反向传播）、还能跟相邻工位合并（算子融合）。

所以在深度学习框架里，`torch.matmul` 不是一个简单的 Python 函数——它是一个**带元信息（数据类型、形状推断）、有自动求导规则、能跨硬件部署、可被融合优化的计算节点**。叫"算子"而不叫"函数"，正是为了强调这些特殊身份。

### 1.3 算子的层级

"算子"这个词在不同语境下有不同的粒度：

| 层级 | 例子 | 谁关心它 |
|------|------|----------|
| **框架层算子** | `torch.matmul`、`F.softmax`、`F.linear` | 算法工程师、模型开发者 |
| **底层算子（Kernel）** | cuBLAS的GEMM、cuDNN的卷积、FlashAttention | 推理引擎开发者、性能工程师 |
| **硬件算子（Instruction）** | Tensor Core指令、WMMA指令 | 芯片架构师、CUDA开发者 |

本文主要聚焦在**框架层和底层Kernel之间**——这是理解LLM系统性能的关键地带。

---

## 二、LLM中的核心算子全景图

一个标准的Transformer解码器（LLaMA/GPT的骨架）由以下核心算子组成：

```
输入 Token Embedding算子 --> 向量
       │
       ▼
┌──────────── 以下重复N层 ─────────────────┐
│           Transformer Block × N         │
│                                         │
│  ┌─ RMSNorm ──────────────────────────┐ │
│  │  算子: x / rms(x) × weight        │ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ RoPE（旋转位置编码）──────────────┐ │
│  │  算子: 对 q, k 做旋转矩阵乘法      │ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ QKV Linear（矩阵乘法）───────────┐ │
│  │  算子: Q = X × Wq, K = X × Wk,   │ │
│  │        V = X × Wv                  │ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ Attention（注意力计算）──────────┐ │
│  │  算子: S = Q × K^T / √d         │ │
│  │        P = softmax(S)            │ │
│  │        O = P × V                 │ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ Output Linear（矩阵乘法）────────┐ │
│  │  算子: O_out = Attention_out × Wo  │ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ 残差连接 ────────────────────────┐ │
│  │  算子: output = input + sublayer(x)│ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ RMSNorm ──────────────────────────┐ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ FFN（前馈网络）──────────────────┐ │
│  │  算子: gate = SiLU(x × W_gate)    │ │
│  │        up   = x × W_up           │ │
│  │        ffn  = (gate × up) × W_down│ │
│  └────────────────────────────────────┘ │
│                  │                      │
│  ┌─ 残差连接 ────────────────────────┐ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
       │
       ▼
   Logits → 采样 → 输出 Token
```

看起来很多？其实**归类之后只有5大类**：

| 类别 | 包含的算子 | 计算特征 |
|------|-----------|----------|
| **矩阵乘法（GEMM）** | Linear、QKV投影、Attention中的QK^T和PV、FFN | 计算密集型，占80%+时间 |
| **归一化** | LayerNorm、RMSNorm | 访存密集型，单Token很快 |
| **激活函数** | ReLU、GELU、SiLU（Swish） | 逐元素运算，几乎不耗时 |
| **Softmax** | Attention中的概率计算 | 需要全局归约，有 reduction 操作 |
| **位置编码** | RoPE（旋转变换）、ALiBi（线性偏置） | 给token注入位置信息，方式各异 |

---

## 三、逐个击破：LLM核心算子详解

### 3.1 矩阵乘法（GEMM）—— 算子之王

**GEMM = General Matrix Multiply（通用矩阵乘法，对应MatMul矩阵乘法算子）**

$$C = \alpha \cdot A \times B + \beta \cdot C$$

在LLM中，几乎 everywhere 都是矩阵乘法：

- QKV投影：$Q = X \times W_q$
- 注意力分数：$S = Q \times K^T$
- 注意力输出：$O = P \times V$
- FFN中的三个线性层

LLM大模型中最最最重要的一个算子：
```
输入: 向量 x (768维), 权重矩阵 W (768×768)
操作: x × W
输出: 新向量 (768维)
```

**为什么矩阵乘法这么重要？**

因为**线性层 = 矩阵乘法 + 偏置**，而Transformer里到处都是线性层。一个标准的LLaMA-70B模型，推理时90%以上的时间花在矩阵乘法上。

**矩阵乘法在GPU上的执行**：

GPU上的矩阵乘法 ≠ 简单的三层for循环。
- 朴素实现：O(n³) 次乘法，每个线程算一个元素
- 高效实现：利用 GPU 的 Tensor Core

Tensor Core 的工作方式：
  一次指令完成一个小矩阵乘法：D = A × B + C
  其中 A, B, C, D 都是 4×4 或 8×8 的小矩阵

所以一个 SM（流多处理器）上的 Tensor Core：

```
  ┌───────────────────────────────┐
  │  输入矩阵 A（分块）            │
  │  ┌────┬────┬────┬────┐       │
  │  │    │    │    │    │       │
  │  ├────┼────┼────┼────┤       │
  │  │    │    │    │    │       │
  │  ├────┼────┼────┼────┤  ×    │
  │  │    │    │    │    │       │
  │  ├────┼────┼────┼────┤       │
  │  │    │    │    │    │       │
  │  └────┴────┴────┴────┘       │
  │                                │
  │  每个小块 → Tensor Core 指令   │
  │  一次完成 4×4 × 4×4 的乘法     │
  └───────────────────────────────┘
```

**关键性能指标**：

| 精度 | A100 Tensor Core算力 | A100 稀疏加速 |
|------|---------------------|---------------|
| FP64 | 19.5 TFLOPS | 39 TFLOPS |
| FP32 | 19.5 TFLOPS | 39 TFLOPS |
| TF32 | 156 TFLOPS | 312 TFLOPS |
| FP16/BF16 | 312 TFLOPS | 624 TFLOPS |
| INT8 | 624 TOPS | 1248 TOPS |
| FP8 (H100) | 1979 TFLOPS | 3958 TFLOPS |

> 划重点：**LLM推理的核心优化，本质上是"让矩阵乘法更快"**。其他所有优化，要么是让矩阵乘法跑得更好，要么是减少矩阵乘法要做的工作量。

### 3.2 注意力算子（Attention）—— 最复杂的算子

注意力算子其实是**一组算子的组合**，但它太重要了，值得单独拿出来讲。

先讲一下Softmax，注意力公式里面有个非常关键的计算，那就是Softmax，自然而然有了Softmax算子，在Attention打分后把分数变成概率，在最终输出时把logits变成词的概率。

Softmax算子：
```
输入: 一组分数 [2.0, 1.0, 0.5, 0.1]
操作: 对每个数取 e 的指数，再除以总和
输出: 概率分布 [0.52, 0.19, 0.12, 0.08]（加起来 = 1）
```

得到 Attention Weights（注意力权重），这些权重之和为 1，代表当前词对序列中其他各个词的“关注程度”概率分布。

#### 标准注意力计算流程

```
输入: Q (seq_len × d_k), K (seq_len × d_k), V (seq_len × d_v)

Step 1: S = Q × K^T                    → (seq_len × seq_len) 矩阵乘法
Step 2: S = S / √d_k                    → 逐元素缩放
Step 3: S = mask(S)                     → 因果掩码（上三角置为 -∞）
Step 4: P = softmax(S, dim=-1)          → 逐行 softmax
Step 5: O = P × V                       → (seq_len × d_v) 矩阵乘法

输出: O (seq_len × d_v)
```

**注意力的计算复杂度**：

- 时间复杂度：$O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是头维度
- 空间复杂度：$O(n^2)$，需要存储 $n \times n$ 的注意力矩阵

这就是为什么长序列（比如128K上下文）推理这么贵——注意力矩阵按序列长度的**平方**增长。

#### FlashAttention：重新定义注意力算子

标准注意力的问题：Step 1 产生的 $n \times n$ 矩阵太大，要写回HBM（显存），Step 4 又要从HBM读回来。HBM带宽是瓶颈。

**FlashAttention的核心思想：融合（Fusion）**

```
传统实现：
  Q × K^T → [写入 HBM] → 读回 → softmax → [写入 HBM] → 读回 → × V → 输出
              ↑ 慢！                    ↑ 慢！

FlashAttention：
  把 Q×K^T、softmax、×V 融合成一个Kernel
  中间结果全在 SRAM（片上高速缓存）里，不经过HBM

  Q × K^T → softmax → × V → 输出
  └──── 全在 SRAM 中完成 ────┘
```

**效果**：
- 显存占用：从 $O(n^2)$ 降到 $O(n)$（不需要存完整的注意力矩阵）
- 速度提升：2~4倍（HBM读写减少是主因）

> FlashAttention是"算子融合"最成功的案例——它证明了**有时候最快的算法不是计算最少的算法，而是访存最少的算法**。

### 3.3 归一化算子（LayerNorm / RMSNorm）

#### LayerNorm

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta$$

其中 $\mu = \frac{1}{d}\sum x_i$，$\sigma^2 = \frac{1}{d}\sum(x_i - \mu)^2$

LayerNorm对**最后一个维度**做归一化——即对每个Token的隐藏层向量独立做。


LayerNorm算子：
```
输入: 向量 [2.0, 4.0, 6.0, 8.0]
操作: 减均值，除标准差，乘 γ 加 β
输出: 归一化向量 [-1.34, -0.45, 0.45, 1.34] (保留两位小数)
```

#### RMSNorm（LLaMA系列使用）

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \times \gamma$$

其中 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$

RMSNorm更简单：**不减均值，只除RMS**。少算一次均值，少一次减法，更快。

#### 为什么归一化很重要？

| 作用 | 说明 |
|------|------|
| **稳定训练** | 防止激活值爆炸或消失 |
| **加速收敛** | 让梯度更平滑，学习率可以设更大 |
| **推理也必需** | 即使推理时batch_size=1，RMSNorm仍然要做（权重γ要乘上去） |

**归一化算子的性能特征**：

- 计算量很小（$O(d)$，d 是隐藏层维度）
- 但它是**访存密集型**的——需要读一整行数据，算均值/方差，再写回去
- 通常与残差连接融合成一个Kernel（`RMSNorm + Residual Add`）

### 3.4 激活函数算子

激活函数都是**逐元素（element-wise）运算**，计算极简：

| 激活函数 | 公式 | 使用者 |
|----------|------|--------|
| ReLU | $\max(0, x)$ | 早期模型 |
| GELU | $x \cdot \Phi(x)$（近似版：$0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$） | BERT、GPT-2 |
| SiLU/Swish | $x \cdot \sigma(x)$（$\sigma$是sigmoid） | LLaMA、PaLM |

**激活函数本身几乎不耗时**——但它们在FFN中的位置很关键：

```
SwiGLU（LLaMA使用的FFN结构）：

     ┌──→ W_gate → SiLU ──┐
x ───┤                     ├──→ 逐元素乘法 ──→ W_down → 输出
     └──→ W_up ───────────┘

三个矩阵乘法 + 一个SiLU + 一个逐元素乘法
SiLU在这里把 gate 的值"门控化"——控制 up 的信息能通过多少
```

GELU激活算子（FNN中间的激活函数）：
```
输入: 一组数字 [1.5, -0.3, 2.0, -1.0]
操作: 对每个数字应用 GELU 函数（正数基本保留，负数被压低）
输出: [1.47, -0.11, 1.99, -0.16]
```

### 3.5 位置编码算子（RoPE）

**RoPE = Rotary Position Embedding（旋转位置编码）**

LLaMA、Qwen、Mistral等主流模型都用RoPE。它的核心操作是：**对 q 和 k 向量做二维旋转**。

```
对于每对相邻维度 (x_{2i}, x_{2i+1})，旋转角度 θ = m × base^{-2i/d}：

┌ x'_{2i}   ┐   ┌ cos θ   -sin θ ┐   ┌ x_{2i}   ┐
│           │ = │                  │ × │           │
└ x'_{2i+1} ┘   └ sin θ    cos θ ┘   └ x_{2i+1} ┘

其中 m 是位置索引（第几个token），base 通常是 10000
```

**RoPE算子的性能特征**：
- 计算量：$O(n \cdot d)$，每个token的每个维度对做一次旋转
- 实现关键：cos/sin值可以预计算并缓存，实际执行只是乘加
- 通常与QKV投影融合，减少kernel launch开销

---

## 四、算子的性能分类：计算密集 vs 访存密集

这是理解LLM推理优化的**最关键概念**。

### 4.1 两个关键指标

| 指标 | 定义 | 含义 |
|------|------|------|
| **算力利用率（MFU）** | 实际FLOPS / 理论峰值FLOPS | 你的GPU"忙不忙" |
| **带宽利用率** | 实际带宽 / 理论峰值带宽 | 你的数据"搬得快不快" |

### 4.2  Roofline模型

用一张图理解算子的性能瓶颈在哪：

```
算力 (TFLOPS)
  │
  │          ┌─────────────── 算力上限（峰值FLOPS）
  │         ╱│
  │        ╱ │
  │       ╱  │
  │      ╱   │
  │     ╱    │
  │    ╱     │
  │   ╱      │
  │  ╱ ← 带宽上限（峰值带宽 × 算术强度）
  │ ╱
  │╱
  └──────────────────────────────→ 算术强度 (FLOPS/Byte)
       ↑              ↑
    访存瓶颈区      计算瓶颈区
```

**算术强度 = 计算量(FLOPS) / 访存量(Bytes)**

- 算术强度高 → 受算力限制 → 计算密集型（如GEMM）
- 算术强度低 → 受带宽限制 → 访存密集型（如softmax、layernorm）

### 4.3 LLM推理各阶段的算子分布

```
┌────────────────────────────────────────────────────────────┐
│              LLM推理两个阶段的算子特征                       │
│                                                            │
│  Prefill（预填充）阶段：                                    │
│  ─────────────────────                                     │
│  • 输入序列长（比如4096 tokens）                              │
│  • 矩阵乘法：大矩阵 × 大矩阵 → 计算密集                      │
│  • GPU利用率高（可达60%+ MFU）                               │
│  • 瓶颈：算力                                              │
│                                                            │
│  Decode（解码）阶段：                                       │
│  ──────────────────────                                    │
│  • 每次只生成1个token                                       │
│  • 矩阵乘法：1个向量 × 大矩阵 → 访存密集                     │
│  • GPU利用率低（通常<10% MFU）                               │
│  • 瓶颈：显存带宽（要搬权重但算力闲着）                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**这就是为什么LLM推理优化的核心矛盾**：

- Prefill阶段：算力没喂饱 → 需要更大的batch、更高效的GEMM
- Decode阶段：带宽被占满但算力闲置 → 需要量化（减少搬运量）、KV Cache优化

---

## 五、算子优化：让GPU跑得更快

### 5.1 算子融合（Kernel Fusion）

**核心思想：把多个小算子合并成一个大算子，减少中间结果的读写。**

```
融合前（3次kernel launch，3次HBM读写）：
  Kernel 1: y = x × W + b      → 写入HBM
  Kernel 2: z = ReLU(y)         → 从HBM读y，写入HBM
  Kernel 3: o = LayerNorm(z)    → 从HBM读z，写入HBM

融合后（1次kernel launch，1次HBM读写）：
  FusedKernel: o = LayerNorm(ReLU(x × W + b))
  → 中间结果 y, z 全在寄存器/共享内存中，不经过HBM
```

**常见的融合模式**：

| 融合模式 | 融合内容 | 使用者 |
|----------|----------|--------|
| GEMM + Bias + Activation | 线性层 + 偏置 + 激活函数 | cuBLAS + cutlass |
| FusedAttention | QK^T + scale + mask + softmax + PV | FlashAttention |
| RMSNorm + Residual | 归一化 + 残差加法 | FasterTransformer |
| FusedMoE | 路由 + 多个专家GEMM + 合并 | vLLM, DeepSpeed |

### 5.2 量化算子

量化 = 用更少的比特表示权重和激活值 → 减少显存搬运量 → 突破带宽瓶颈。

| 量化方案 | 权重精度 | 激活精度 | 效果 |
|----------|----------|----------|------|
| FP16 | 16位浮点 | 16位浮点 | 基线，精度无损 |
| INT8 (PTQ) | 8位整数 | 8位整数 | 显存减半，轻微精度损失 |
| INT4 (GPTQ/AWQ) | 4位整数 | 16位浮点 | 显存4倍压缩，精度可接受 |
| FP8 (H100) | 8位浮点 | 8位浮点 | 算力翻倍，精度接近FP16 |
| NF4 (QLoRA) | 4位正态分布 | 16位浮点 | 显存4倍压缩+更低 |

**量化算子的难点**：

```
FP16 GEMM：
  A(FP16) × B(FP16) → C(FP32)     ← Tensor Core原生支持

INT8 GEMM：
  A(INT8) × B(INT8) → C(INT32)    ← Tensor Core支持
  但需要：反量化 → FP32 结果         ← 额外开销

INT4 GEMM：
  A(INT4) × B(FP16) → ?           ← Tensor Core不直接支持！
  需要：INT4 → INT8 反量化 → INT8 GEMM → 结果修正

  注意：INT4 vs INT8 谁更快，取决于瓶颈在哪：
  - Decode阶段（带宽瓶颈）：INT4搬的数据少一半，反而可能更快
  - Prefill阶段（算力瓶颈）：反量化开销拖慢速度，可能更慢
  所以不是简单的"谁比谁快"，要看场景
```

### 5.3 PagedAttention：改变KV Cache的算子

传统注意力中，KV Cache需要**连续显存**。问题是：
- 不同请求的序列长度不同
- 预分配连续显存 → 大量碎片 → 显存浪费

**PagedAttention**（vLLM的核心创新）：

```
传统方式：
  请求1: [████████████████████░░░░]  ← 预分配了4096，实际用了2500
  请求2: [██████████░░░░░░░░░░░░░░]  ← 预分配了4096，实际用了1000
  请求3: [████████████████████████████████░░░░]  ← 预分配了4096...
  
  碎片率：~40%

PagedAttention：
  把KV Cache切成固定大小的"页"（比如16 tokens一页）
  按需分配，不浪费
  
  物理块: [块0][块1][块2][块3][块4]...
  请求1:  块0 → 块1 → 块3（逻辑上连续，物理上不连续）
  请求2:  块2 → 块4
  请求3:  块5 → 块6 → 块7 → 块8
  
  碎片率：<4%（最后一页的尾部浪费）
```

PagedAttention改变了**KV Cache的内存管理算子**——从连续分配变成按需分页，显存利用率大幅提升，能同时服务的请求数翻倍。

---

## 六、算子的工程生态

### 6.1 谁在实现这些算子？

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM算子的实现层级                           │
│                                                              │
│  ┌─ 模型层（HuggingFace Transformers）─────────────────────┐ │
│  │  torch.nn.Linear, torch.nn.functional.scaled_dot_product_attention │
│  │  特点: 正确性优先，性能一般                               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─ 推理框架层（vLLM / TensorRT-LLM / FasterTransformer）──┐ │
│  │  调用底层算子库，做算子融合、KV Cache管理、调度优化         │ │
│  │  特点: 针对LLM场景深度优化                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─ 算子库层（cuBLAS / cuDNN / cutlass / FlashAttention）──┐ │
│  │  针对NVIDIA GPU的底层高性能Kernel实现                      │ │
│  │  特点: 极致性能，需要深入理解GPU架构                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─ 硬件层（NVIDIA GPU + CUDA + Tensor Core）──────────────┐ │
│  │  物理执行单元                                             │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 主流算子库一览

| 算子库 | 主要功能 | 特点 |
|--------|----------|------|
| **cuBLAS** | 矩阵乘法（GEMM） | NVIDIA官方，最通用 |
| **cuDNN** | 卷积、RNN、归一化 | NVIDIA官方，CV/NLP通用 |
| **cutlass** | 高性能GEMM模板库 | NVIDIA开源，可定制性强 |
| **FlashAttention** | 融合注意力算子 | 显存高效，已成为标配 |
| **Triton** | Python写GPU Kernel | OpenAI出品，降低写Kernel门槛 |
| **DeepSpeed-Kernels** | LLM专用算子 | 微软出品，针对推理优化 |
| **xformers** | Transformer算子集合 | Meta出品，包含MemoryEfficientAttention |

### 6.3 Triton：用Python写高性能算子

传统写GPU算子需要精通CUDA C++。Triton让你用**类Python语法**写Kernel，编译器自动生成优化后的GPU代码：

```python
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,          # 输入指针
    w_ptr,          # 权重指针
    out_ptr,        # 输出指针
    n_cols,         # 隐藏层维度
    eps,            # 小常数
    BLOCK_SIZE: tl.constexpr,  # 线程块大小（编译期常量）
):
    """RMSNorm的Triton实现"""
    # 每个program处理一个token
    row = tl.program_id(0)
    
    # 加载这一行的数据
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    
    # 计算RMS
    x_sq = x * x
    rms = tl.sqrt(tl.sum(x_sq, axis=0) / n_cols + eps)
    
    # 归一化并乘以权重
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (x / rms) * w
    
    # 写回
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)
```

**Triton的意义**：让算法工程师也能写高性能算子，不需要从零学CUDA C++。许多第三方attention实现（如xformers的部分Kernel、mila-iqia的flash-attn-triton）都用Triton编写，但FlashAttention-1/2/3本身全部用CUDA C++实现以获得极致性能。

---

## 七、算子性能分析实战

### 7.1 用PyTorch Profiler看算子耗时

```python
import torch
from torch.profiler import profile, ProfilerActivity

# 创建一个简单的attention
q = torch.randn(1, 32, 4096, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 32, 4096, 128, device='cuda', dtype=torch.float16)
v = torch.randn(1, 32, 4096, 128, device='cuda', dtype=torch.float16)

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.no_grad():
        for _ in range(10):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

输出类似：

```
-------------------------  ---------------  ---------------
Name                       CPU Time         CUDA Time        
-------------------------  ---------------  ---------------
aten::scaled_dot_product   12.3 ms          45.2 ms    ← 注意力算子
aten::bmm                  2.1 ms           18.7 ms    ← 矩阵乘法
aten::softmax              1.5 ms           8.3 ms     ← softmax算子
aten::scaled_dot_product…  0.8 ms           5.1 ms     ← 其他
-------------------------  ---------------  ---------------
```

说明：
- CPU Time：CPU把活儿派出去的时间，包括Python → C++ → CUDA Driver → 把Kernel提交到GPU队列。
- CUDA Time：Kernel 在 GPU上实际执行的时间，包括GPU从接收指令到算完返回结果。

### 7.2 Nsight Systems：看算子的时间线

```
Nsight Systems 时间线视图：

时间 →
├─ CUDA Kernel ──────────────────────────────────────────┤
│  [GEMM][GEMM][GEMM][Attn][Softmax][GEMM][RMSNorm]...   │
│  ← 2.3μs →← 15μs →← 1μs →                            │
│                                                        │
├─ Memory Copy ──────────────────────────────────────────┤
│         [H2D]                    [D2H]                  │
└────────────────────────────────────────────────────────┘
```

Nsight Systems能看到：
- 每个Kernel的执行时间
- Kernel之间的gap（是否有调度空闲）
- 内存拷贝与计算的overlap情况
- GPU的SM利用率和带宽利用率

---

## 八、前沿趋势：算子的未来

### 8.1 算子自动调优（Auto-tuning）

不同的GPU、不同的矩阵大小，最优的算子实现可能完全不同。

```
比如矩阵乘法 C = A(m×k) × B(k×n)：
  - m, n, k 都很大 → 用大tile，充分占用Tensor Core
  - m 很小（decode阶段）→ 用split-k策略，并行切分k维度
  - k 很小 → 用warp-level策略，减少同步

cutlass 提供了数百种实现模板（tile size, pipeline depth, ...）
auto-tuner 会跑一遍所有配置，选最快的
```

### 8.2 注意力优化算子：减少KV开销

注意力的O(n²)复杂度来自每个token关注所有token。优化思路分两类：

```
第一类：真稀疏 —— 减少每个token关注的token数量
  - 滑动窗口注意力（Mistral）：只看最近w个token → O(n·w)
  - 局部+全局混合（Longformer）：局部窗口 + 少量全局token

第二类：KV头共享 —— 不改变时间复杂度，但大幅减少KV Cache显存
  - 多查询注意力MQA（Falcon）：所有Q头共享1组KV → KV显存÷头数
  - 分组查询注意力GQA（LLaMA-2 70B）：每8个Q头共享1组KV → KV显存÷8
```

### 8.3 异构算子：CPU也算？

大部分算子在GPU上跑，但有些算子在CPU上反而更高效：

| 算子 | 为什么适合CPU |
|------|--------------|
| Tokenizer（分词） | 字符串操作，GPU不擅长 |
| Top-K采样 | 数据量小，CPU延迟更低 |
| KV Cache卸载到CPU内存 | CPU内存便宜且大，带宽够用时可以接受 |
| Embedding查表 | 大表小batch，CPU可能更快 |

---

## 总结

**算子 = GPU的基本计算单元，也即LLM的基本计算单元**

**五大核心算子：**
- ① 矩阵乘法（GEMM）—— 算子之王，占90%+时间
- ② 注意力（Attention）—— 最复杂，FlashAttention是标杆
- ③ 归一化（RMSNorm）—— 小而关键，稳定训练和推理
- ④ 激活函数（SiLU/GELU）—— 逐元素运算，几乎不耗时
- ⑤ 位置编码（RoPE）—— 给token注入位置信息

**性能优化的核心逻辑：**
- 计算密集型算子 → 提高算力利用率（更大batch、更好GEMM）
- 访存密集型算子 → 减少数据搬运（量化、融合、paged管理）

**算子优化的三板斧：**
- ① 融合 —— 把多个小算子合成一个大算子，减少HBM读写
- ② 量化 —— 用更少比特表示数据，搬得更快
- ③ 分块 —— 大矩阵切成小块，塞进GPU的高速缓存

**一句话记住：**
LLM推理优化的本质 = 让每个算子都跑在它该跑的模式上（计算密集的就喂饱算力，访存密集的就减少搬运）

理解了算子，你就理解了LLM推理系统的"原子操作"。不管是读vLLM的源码、调TensorRT-LLM的参数、还是自己写推理引擎，算子层面的认知都是最底层的基本功。

如果觉得有用，转发给身边正在学 LLM 推理的朋友。
