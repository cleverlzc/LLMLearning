# 【硬核拆解vLLM】深入理解 block_size 和 head_size——两个数字如何决定 KV Cache 的

> **系列**: vLLM 技术博客系列 | **类型**: 核心概念深潜篇
> block_size=16，head_size=128——这两个看似简单的数字，决定了 KV Cache 的物理大小、碎片率、前缀缓存粒度、调度对齐规则，甚至影响你能跑多长的上下文

### 引言

翻开 vLLM 的配置，你会看到两个重要的信息：`block_size=16`，`head_size=128`。它们看起来只是两个参数，但实际上——**KV Cache 的每一个字节都由它们共同参与决定**。一个管"几个 token 打一包"，一个管"每个 token 的向量有多宽"。但 Block 的物理大小不是简单两者相乘，完整公式是：

```
Block 物理大小 = 2 × block_size × num_kv_heads × head_size × dtype_size × 层数
```

`block_size` 和 `head_size` 是公式中的两个关键乘数，但还有 `num_kv_heads`、`dtype_size`、`层数` 等因素共同参与。

今天我们就把这两个数字拆透：它们从哪来、影响什么、怎么共同决定你的显存账单。

---

### 一、block_size=16：几个 Token 打一包

##### 它是什么

`block_size` 是 KV Cache 的**存储页大小**——一个 block 固定存放 16 个 token 的 Key/Value 向量。

```
block_size = 16

┌─────────────── 1 个 Block ──────────────┐
│ token 0 的 KV │ token 1 的 KV │ ... │ token 15 的 KV │
└─────────────────────────────────────────┘
     16 个 token 的 KV 向量，打包在一起
```

就像一本书固定 16 页装订成一册，不管你写满了 1 页还是 16 页，这一册的物理厚度都一样。

##### 为什么是 16

| block_size | 问题 |
|---|---|
| 太小（如 1） | 每个 token 一个 block，block table 巨大，调度开销爆炸 |
| 太大（如 256） | 内部碎片严重，2 个 token 也要占 256 的 block，浪费 254 个 |
| **16** | **工程权衡：GPU 访存对齐友好，碎片可控，管理开销可接受** |

> 💡 **性能提示**: block_size 必须是 16 的倍数（FlashAttention 后端强制要求），否则会报错。源码中的检查：`if block_size % 16 != 0: raise ValueError("Block size must be a multiple of 16.")`

##### block_size 影响什么

| 影响维度 | 说明 |
|---|---|
| Block 物理大小 | `block_size × 每 token KV 大小`，直接决定一个 block 占多少显存 |
| Block 数量 | `ceil(token 数量 / block_size)`，决定需要分配多少个 block |
| 碎片率 | 最多浪费 `block_size - 1` 个 token 的空间（一个 block 内未填满的部分） |
| 前缀缓存粒度 | 以 block_size 个 token 为最小单位计算哈希、匹配前缀 |
| Chunked Prefill 对齐 | chunk 的大小必须是 block_size 的整数倍，否则 KV Cache 半写 |
| CUDA Graph 对齐 | 批次大小变化时，block table 的重分配粒度由 block_size 决定 |

##### 源码定义

```python
# vllm/config/cache.py
class CacheConfig:
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16
    block_size: int = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens."""
```

---

### 二、head_size=128：每个注意力头的向量有多宽

##### 它是什么

`head_size`（也叫 `head_dim`，严谨一点说head_size 是 vLLM 对 HuggingFace 中 head_dim 的内部命名，正常情况下两者等价，量化场景下 head_dim 可能是 head_size 的缩放值）是每个 KV head 的向量维度——一个 head 的 Key 或 Value 向量由 128 个浮点数组成。

```
head_size = 128

一个 KV head 的 Key 向量:   [0.12, -0.34, 0.56, ..., 0.78]   ← 128 个浮点数
一个 KV head 的 Value 向量:  [0.33, 0.45, -0.12, ..., 0.89]  ← 128 个浮点数
```

就像每本书固定 128 页——不管书名长短，内容多少，页数不变。

##### head_size 从哪来

由**模型结构**决定，写在模型的 `config.json` 里，训练后就固定了，推理时不能改：

| 模型 | head_size | num_kv_heads | 注意力类型 |
|---|---|---|---|
| LLaMA-7B | 128 | 32 | 标准 MHA |
| LLaMA-70B | 128 | 8 | GQA |
| LLaMA-3-8B | 128 | 8 | GQA |
| Gemma-2-9B | 256 | 4 | 更宽的 head |
| Mixtral-8x7B | 128 | 8 | GQA |

> 笔者注：head_size 和 num_attention_heads 是两个不同的东西。LLaMA-70B 有 64 个 attention head，但只有 8 个 KV head（GQA），所以 head_size=128 不变，但 KV Cache 的总量大幅减少——这就是 GQA 省显存的秘密。

##### head_size 影响什么

| 影响维度 | 说明 |
|---|---|
| 每 token KV 大小 | `2 × 层数 × num_kv_heads × head_size × dtype_size`，head_size 是乘数之一 |
| Block 物理大小 | 通过"每 token KV 大小"间接决定 |
| 注意力计算量 | Q × K^T 的矩阵乘法，head_size 影响 K 的维度 |
| GPU 显存占用 | head_size 越大，KV Cache 越大，同等显存能跑的上下文越短 |

---

### 三、两个数字如何共同影响 Block 的物理大小

这是本文的核心——**block_size 和 head_size 通过一个公式共同影响了 Block 的物理大小**。

##### 源码中的计算公式

```python
# vllm/v1/kv_cache_interface.py — AttentionSpec.real_page_size_bytes
def real_page_size_bytes(self) -> int:
    return (
        2                    # K 和 V 两份
        * self.block_size    # 每个 block 多少个 token（16）
        * self.num_kv_heads  # 多少个 KV head
        * head_dim           # 每个 head 多少维（= head_size = 128）
        * get_dtype_size(self.dtype)  # 每个浮点数占多少字节
    )
```

##### 逐步拆解：一个 Block 到底多大

以 LLaMA-70B FP16 为例：

```
block_size   = 16      ← 每个 block 装 16 个 token
num_kv_heads = 8       ← 8 个 KV head（GQA）
head_size    = 128     ← 每个 head 128 维
dtype_size   = 2       ← FP16，每个数字 2 字节

单层 1 个 Block = 2 × 16 × 8 × 128 × 2 = 65,536 bytes = 64 KB

80 层总计 = 64 KB × 80 = 5,120 KB ≈ 5 MB
```

**拆开看这个 5 MB 的结构**：

```
1 个 Block（单层）= 64 KB

┌─── K 部分 ─────────────────────────────┐
│ 16 tokens × 8 heads × 128 dim × 2 bytes│
│ = 32 KB                                 │
├─── V 部分 ─────────────────────────────┤
│ 16 tokens × 8 heads × 128 dim × 2 bytes│
│ = 32 KB                                 │
└─────────────────────────────────────────┘

FlashAttention 后端的 Tensor shape:
(num_blocks, 2, block_size, num_kv_heads, head_size)
 = (N,     2,    16,         8,            128)
```

##### 不同配置下的 Block 大小对比

| 模型 | block_size | head_size | num_kv_heads | 层数 | dtype | **单 Block 大小** |
|---|---|---|---|---|---|---|
| LLaMA-7B | 16 | 128 | 32 | 32 | FP16 | **8 MB** |
| LLaMA-70B | 16 | 128 | 8 | 80 | FP16 | **5 MB** |
| LLaMA-70B | 16 | 128 | 8 | 80 | FP8 | **2.5 MB** |
| Gemma-2-9B | 16 | 256 | 4 | 42 | FP16 | **10.8 MB** |

> 💡 **性能提示**: LLaMA-7B 的 Block 比 70B 还大（8 MB vs 5 MB）！因为 7B 用标准 MHA（32 个 KV head），70B 用 GQA（8 个 KV head）。所以"模型越小，KV Cache 越小"这个直觉是错的——关键看 num_kv_heads。

---

### 四、从两个数字到你的显存账单

知道 Block 大小后，就能算出你的显存到底花在哪了。

##### 场景：LLaMA-70B FP16，4K 上下文，并发 100 个请求

```
每个请求的 token 数 = 4096
block_size = 16
每个请求的 block 数 = ceil(4096 / 16) = 256
每个 block = 5 MB

单个请求的 KV Cache = 256 × 5 MB = 1,280 MB ≈ 1.25 GB

100 个并发 = 100 × 1.25 GB = 125 GB

再加上模型权重 ≈ 140 GB

总显存需求 ≈ 265 GB → 至少需要 4 张 H100 (80GB × 4 = 320 GB)
```

##### 调参影响

| 调什么 | 效果 | 代价 |
|---|---|---|
| FP16 → FP8 | Block 从 5 MB → 2.5 MB，显存减半 | 精度略有损失 |
| 减少 max_model_len | 4K → 2K，每个请求 block 数减半 | 能跑的上下文变短 |
| 减少 max_num_seqs | 100 → 50，并发减半 | 吞吐量下降 |
| 换 GQA 模型 | num_kv_heads 32 → 8，Block 大小缩小 4 倍 | 模型表达能力略降 |

---

### 五、block_size 和 head_size 的本质区别

| | block_size | head_size |
|---|---|---|
| 管的是什么 | **存储打包粒度**——几个 token 打一包 | **向量宽度**——每个 head 的向量有多宽 |
| 谁决定的 | 工程配置（可以改，默认 16） | 模型结构（训练后固定，不能改） |
| 影响谁 | 碎片率、调度粒度、前缀缓存粒度 | 每 token KV 大小、注意力计算量 |
| 改大会怎样 | 碎片率上升，但管理开销下降 | 每 token KV 变大，显存占用上升 |
| 改小会怎样 | 管理开销上升，但碎片率下降 | 不由你决定，模型结构说了算 |
| 在公式中的角色 | 乘数（block 数量的分母） | 乘数（每 token KV 大小的因子） |

**一句话：block_size 是你可以调的"打包粒度"，head_size 是模型给你的"向量宽度"——两者相乘，直接影响了 Block 的物理大小。**

---

### 六、比喻：图书馆的存书格

把 Block 想象成图书馆的**存书格**：

**head_size=128** 是每本书的**页数**——每本书固定 128 页，不管书名长短，页数不变。这个由出版社（模型训练）决定，读者（推理）改不了。

**block_size=16** 是每个存书格放**几本书**——每个格子固定放 16 本书。这个由图书馆管理员（vLLM 配置）决定，可以根据情况调整。

**num_kv_heads=8** 是同一层楼有**几个书架**——每个书架独立，但格式相同。

```
一个存书格（1 个 Block，单层）:
┌─────────────────────────────────────┐
│ 书架1: 16本书 × 128页               │  ← 1 个 KV head
│ 书架2: 16本书 × 128页               │  ← 1 个 KV head
│ ...                                  │
│ 书架8: 16本书 × 128页               │  ← 1 个 KV head
│ ──── 以上是 K，以下是 V ──────────  │
│ 书架1~8: 同样结构                    │  ← V 的部分
└─────────────────────────────────────┘

每个格子大小 = 2(K+V) × 16本书 × 8个书架 × 128页 × 每页大小 = 固定值
```

- **页数**（head_size）由出版社决定，改不了
- **每格几本**（block_size）由管理员决定，可以调
- **格子大小**由两者共同决定
- **需要多少个格子**由你的书量（token 数量）决定

**变的是格子数量，不变的是每个格子的大小。**

---

### 总结

| 问题 | 答案 |
|---|---|
| block_size 是什么？ | 每个 Block 装 16 个 token 的 KV，是存储打包粒度 |
| head_size 是什么？ | 每个 KV head 的向量维度 128，是向量宽度 |
| 谁能改？ | block_size 可以调（工程配置），head_size 不能改（模型结构） |
| Block 物理大小怎么算？ | `2 × block_size × num_kv_heads × head_size × dtype_size × 层数` |
| 为什么 7B 的 Block 比 70B 大？ | 7B 用 MHA（32 个 KV head），70B 用 GQA（8 个 KV head） |
| 调 block_size 影响什么？ | 碎片率、调度粒度、前缀缓存粒度 |
| 调 head_size 影响什么？ | 你调不了，但选模型时 head_size 大的显存开销更大 |

### 延伸阅读

- vLLM 源码：[vllm/config/cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) — `block_size` 定义与默认值
- vLLM 源码：[vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py) — `AttentionSpec` 与 `real_page_size_bytes` 计算
- vLLM 源码：[vllm/v1/attention/backends/flash_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) — `get_kv_cache_shape()` 与 block_size 对齐检查

---

*本文属于 [vLLM 技术博客系列]，欢迎持续关注。*
