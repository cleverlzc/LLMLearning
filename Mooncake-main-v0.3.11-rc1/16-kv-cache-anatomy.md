# KV Cache 在 Mooncake 系统中到底长什么样？ 从 Attention 公式到 SSD 磁盘上的字节，一条数据的六次变身

> **系列**: Mooncake 技术博客系列 | **类型**: 核心概念深潜篇
>
> 前面十几篇文章讲了 Mooncake 的架构、传输、存储、调优，但始终没有回答一个最基本的问题：KV Cache 到底长什么样？它在 GPU 显存里是什么结构？搬到 DRAM 呢？通过 RDMA 传到远程呢？写到 SSD 呢？

本文从 Attention 公式出发，追踪一条 KV Cache 数据从诞生到落盘的完整旅程，看它如何经历六次"变身"。

---

### 引言：一条数据的六次变身

KV Cache 不是一块模糊的"数据"——它有精确的数学定义、明确的内存布局、具体的字节大小。但在不同的存储介质和处理阶段，它的"外貌"会发生变化：

```
变身 1: Attention 公式中的 K、V 矩阵     →  数学定义
变身 2: GPU 显存中的分页块 (Paged Block)   →  内存布局
变身 3: RDMA 线路上的原始字节流             →  网络传输
变身 4: DRAM 中的 Slice 切片               →  分布式存储
变身 5: EP 分发中的 FP8 压缩包             →  专家并行
变身 6: SSD 磁盘上的桶文件                 →  持久化存储
```

每一次变身，数据的"外貌"变了，但本质不变——都是同一个 KV Cache。理解这六次变身，就理解了 Mooncake 的整个数据流。

---

### 变身 1：Attention 公式中的 K、V 矩阵

KV Cache 的数学定义来自 Attention 公式：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

其中：
- **Q（Query）**：当前 token 的查询向量，不需要缓存——每个 token 现算现用
- **K（Key）**：所有历史 token 的键向量，**必须缓存**——后续 token 要跟它做点积
- **V（Value）**：所有历史 token 的值向量，**必须缓存**——点积结果要跟它做加权求和

##### K 和 V 分别存了什么？

公式只告诉了计算规则，没说 K 和 V 的内容是什么。回到第一性原理——Attention 的本质是**信息检索**：当前 token 带着一个问题去历史 token 中找答案。这个检索分两步：

```
第一步: "谁跟我相关？"  →  Q × K^T  →  注意力权重
第二步: "把相关的信息拿过来"  →  权重 × V  →  输出
```

用图书馆做比喻：

```
K = 书脊上的标签（"我是讲什么的"）
    → 让读者判断"这本书跟我有关吗"
    → 对应第一步：匹配

V = 书里的内容（"我具体讲了什么"）
    → 读者决定要读之后，真正拿走的信息
    → 对应第二步：提取
```

但 K 和 V 不是人工设计的标签和内容——它们来自同一个源头，却经过不同的变换：

```
一个 token 的隐藏状态 x（如 8192 维向量）
        │
        ├──× W_K ──→ K   "这个 token 想被怎样检索到"
        │               128 维 × 8 头 = 1024 个浮点数
        │               是 x 经过 W_K 投影后的"检索特征"
        │
        └──× W_V ──→ V   "这个 token 要传递什么信息"
                        128 维 × 8 头 = 1024 个浮点数
                        是 x 经过 W_V 投影后的"信息载荷"
```

**K 和 V 的区别**：

| | K（Key） | V（Value） |
|--|---------|----------|
| 来源 | x × W_K | x × W_V |
| 存的是 | "我的检索特征" | "我的信息载荷" |
| 被谁用 | 别人的 Q 来查 | 别人查到后取走 |
| 类比 | 书脊标签——"我是讲量子力学的" | 书的内容——量子力学的具体知识 |
| 类比 | 简历关键词——"Python, 分布式系统" | 工作经历——具体做了什么项目 |
| 在公式中的角色 | 与 Q 做点积，决定"关注多少" | 被注意力权重加权求和，决定"拿到什么" |

**为什么 K 和 V 必须分开？** 因为"怎么被找到"和"提供什么信息"是两件不同的事。一本量子力学的书（K = "量子力学"），可能被一个研究黑洞的人检索到（Q 匹配了 K），但拿走的内容（V）是量子力学知识而不是黑洞知识。如果 K 和 V 不分开，检索特征和信息载荷就混在一起，模型就无法独立调控"谁关注谁"和"传递什么"。

**为什么 K 和 V 必须缓存？** 因为生成第 N 个 token 时，需要跟第 1 到 N-1 个 token 的 K、V 都做计算。如果不缓存，就要重新跑一遍前面所有 token 的前向传播——这就是 Prefill 做的事情。**KV Cache 的本质，就是用空间换时间——把计算结果存下来，避免重复计算。**

##### 一个 token 的 KV Cache 有多大？

对于单个 token、单个 Attention 层：

```
K 的形状 = [num_kv_heads, head_dim]    ← 2D 矩阵
V 的形状 = [num_kv_heads, head_dim]    ← 2D 矩阵
```

**K 和 V 都是二维矩阵**，不是一维数组，也不是更高维的张量。具体来说：

- 矩阵有 `num_kv_heads` 行——每行属于一个"注意力头"
- 每行有 `head_dim` 个浮点数——这就是这个头的向量

```
以 LLaMA-70B (GQA-8) 为例，一个 token 的 K:

     head_dim = 128
   ┌──────────────────────────┐
   │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 0 的 Key 向量 (128 个 float)
k  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 1 的 Key 向量
v  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 2 的 Key 向量
_  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 3 的 Key 向量
h  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 4 的 Key 向量
e  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 5 的 Key 向量
a  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 6 的 Key 向量
d  │ h₀  h₁  h₂ ... h₁₂₇    │  ← kv_head 7 的 Key 向量
s  └──────────────────────────┘
   = 8 × 128 = 1024 个 bf16 浮点数 = 2 KB

V 的结构完全相同，也是 [8, 128] = 2 KB

一个 token 的 KV = K(2KB) + V(2KB) = 4 KB / 层
```

**为什么是矩阵而不是向量？** 因为 Attention 是多头机制——每个头独立地计算"查询-键匹配"，需要自己的 K 和 V。8 个头的 K 拼在一起，就是 8 行 × 128 列的矩阵。

**为什么不是三维或更高维？** 因为"头"维度和"token 序列"维度是分开的。一个 token 只贡献一行，序列中所有 token 的 K 堆叠起来才形成三维张量 `[seq_len, num_kv_heads, head_dim]`。KV Cache 缓存的就是这个三维张量沿第 0 维的切片——每多生成一个 token，就在第 0 维追加一行。

```
K 的大小 = num_kv_heads × head_dim × sizeof(dtype)
V 的大小 = num_kv_heads × head_dim × sizeof(dtype)

KV Cache per token per layer = 2 × num_kv_heads × head_dim × sizeof(dtype)
```

其中 `2` 是因为 K 和 V 各一份。

##### GQA：为什么 70B 模型的 KV Cache 比 7B 还小？

关键在于 `num_kv_heads`——K/V 的头数不等于 Q 的头数：

| 注意力类型 | num_kv_heads vs num_attention_heads | 代表模型 |
|-----------|-----------------------------------|---------|
| MHA | num_kv_heads = num_attention_heads | Mistral-7B-v0.1 (32=32) |
| GQA | num_kv_heads < num_attention_heads | LLaMA-70B (8 < 64) |
| MLA | K/V 压缩为低秩潜向量 | DeepSeek-V3 (128→512维) |

```
MHA (Mistral-7B):  32 个 Q 头，32 个 KV 头 → 每个 KV 头独立存储
GQA (LLaMA-70B):  64 个 Q 头，8 个 KV 头  → 每 8 个 Q 头共享 1 个 KV 头
MLA (DeepSeek-V3): 128 个 Q 头，压缩到 512 维 → KV 整体压缩存储
```

##### 五个主流模型的 KV Cache 尺寸

以 bf16（2 字节/元素）为基准：

| 模型 | 注意力类型 | num_layers | num_kv_heads | head_dim | KV/token/层 | KV/token/全模型 |
|------|-----------|-----------|-------------|----------|------------|---------------|
| Mistral-7B | MHA | 32 | 32 | 128 | 16 KB | **512 KB** |
| LLaMA-70B | GQA-8 | 80 | 8 | 128 | 4 KB | **320 KB** |
| Qwen2-72B | GQA-8 | 80 | 8 | 128 | 4 KB | **320 KB** |
| DeepSeek-V3 | MLA | 61 | 压缩到512维 | — | 1.1 KB | **68.6 KB** |

> 笔者注：LLaMA-70B 的 KV Cache 比 Mistral-7B 还小（320 KB vs 512 KB），这就是 GQA 的威力——8 个 Q 头共享 1 个 KV 头，KV Cache 缩小 8 倍。如果 LLaMA-70B 用 MHA，每 token 的 KV Cache 将是 2.5 MB 而不是 320 KB。DeepSeek-V3 的 MLA 更激进，压缩到 68.6 KB——只有 LLaMA-70B 的 1/5。

---

### 变身 2：GPU 显存中的分页块（Paged Block）

Attention 公式定义了 KV Cache 的数学形态，但 GPU 显存里不是按"一个 token 一份"来存的——那样碎片化太严重。vLLM 使用 **PagedAttention**（2026.7月初，vLLM将注意力计算委托给成熟的第三方高性能注意力后端，FlashInfer，但分块思想未变），把 KV Cache 按**块（Block）**管理。

##### 块的内存布局

vLLM 的 `block_size = 16`（默认值），即每个块存储 16 个 token 的 KV 数据。

对于标准 MHA/GQA 模型，每个层每个方向（K 或 V）的块张量形状：

```
[num_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

其中 `num_kv_heads_per_rank = num_kv_heads / tp_size`（张量并行时每个 Rank 只存自己负责的 KV 头）。

一个块的**字节大小**：

```
block_len = block_size × num_kv_heads_per_rank × head_dim × sizeof(dtype)
```

##### K 和 V 在 Mooncake 代码中的"能见度"

一个关键事实：**Mooncake 本身不创建 K/V 张量，它只搬运 vLLM 分配好的张量**。从注册到传输到存储，K 和 V 的"身份"逐层模糊：

```
能见度递减:

vLLM 层     K 和 V 是两个独立的 torch.Tensor，形状明确
            k_cache: [num_blocks, 16, 1, 128]   ← 这是 K
            v_cache: [num_blocks, 16, 1, 128]   ← 这是 V
                 │
                 ▼ register_kv_caches()
Mooncake    K 和 V 变成 GPU 地址列表，不再区分名字
连接器层        kv_caches_base_addr = [0x7f3a..., 0x7f3b..., ...]
                160 个地址，前 80 个是 K，后 80 个是 V
                但代码不记录"哪个是 K"——只知道地址和 block_len
                 │
                 ▼ batch_transfer_sync_write()
传输引擎层   K 和 V 变成 (src_ptr, dst_ptr, length) 三元组
                TransferRequest {opcode, source, target_id, target_offset, length}
                完全不知道这是 K 还是 V，只知道"从哪搬到哪，搬多少字节"
                 │
                 ▼ Put()
存储层      K 和 V 变成 (ptr, size) 切片
                Slice {ptr, size}
                连"层"的概念都没有了，只是一段不透明的字节
```

**代码实锤**——注册时 K 和 V 的身份就已经被"抹平"了：

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
# register_kv_caches() 方法 (第 694-743 行)

def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    # vLLM 传入的 kv_caches: 每层一个 (k_cache, v_cache) 元组
    # 但 Mooncake 只提取 data_ptr() 和 nbytes

    self.split_k_and_v = not (self.use_mla or self._use_pallas_v1
                              or self._use_flashinfer)

    for layer_name, cache_or_caches in kv_caches.items():
        # split_k_and_v=True 时: cache_or_caches = (k_tensor, v_tensor)
        # split_k_and_v=False 时: cache_or_caches = 单个融合张量
        cache_list = cache_or_caches if self.split_k_and_v else [cache_or_caches]

        for cache in cache_list:       # K 和 V 被同等对待
            base_addr = cache.data_ptr()   # 只取 GPU 地址
            ...
            kv_data_ptrs.append(base_addr)  # 加入地址列表，不区分 K/V
            kv_data_lens.append(cache.nbytes)

    self.kv_caches_base_addr = seen_base_addresses
    # 结果: 80 层 × 2 = 160 个地址
    # 第 0-79 个是各层 K 的地址，第 80-159 个是各层 V 的地址
    # 但代码不记录"第 0 个是 K"——只依赖顺序对应
```

**传输时 K 和 V 完全透明**：

```python
# _build_transfer_params() 方法 (第 617-673 行)

for local_layer_addr, remote_layer_addr in zip(
    local_base_addr, remote_base_addr    # 遍历 160 个地址
):
    for group_local_block_id, group_remote_block_id in zip(
        group_local_block_ids, group_remote_block_ids
    ):
        src_ptrs.append(
            local_layer_addr + group_local_block_id[0] * block_len
        )
        dst_ptrs.append(
            remote_layer_addr + group_remote_block_id[0] * block_len
        )
        lengths.append(block_len * len(group_local_block_id))
# TransferRequest 只有 source/target/length，没有"K/V"字段
```

**存储层更彻底**——连"层"都不认了：

```cpp
// mooncake-store/include/types.h
struct Slice {
    void* ptr{nullptr};    // 内存地址
    size_t size{0};        // 字节数
    // 没有"层号"、"K/V方向"字段
};
```

##### split_k_and_v：K 和 V 分不分，取决于注意力后端

| 后端/模型 | split_k_and_v | vLLM 传入每层 | Mooncake 注册 |
|-----------|:------------:|-------------|-------------|
| 标准 MHA/GQA (CUDA) | True | 元组 `(k_cache, v_cache)` | 2 个地址/层 |
| MLA (DeepSeek-V3) | False | 单个融合张量 | 1 个地址/层 |
| Pallas (TPU) | False | 单个融合张量 | 1 个地址/层 |
| FlashInfer (vLLM V1) | False | 单个融合张量 | 1 个地址/层 |

```python
# split_k_and_v 的判定逻辑 (第 703-704 行)
self.split_k_and_v = not (self.use_mla or self._use_pallas_v1
                          or self._use_flashinfer)
```

对于 MLA 模型，K 和 V 在 vLLM 层面就已经合并了——不是 Mooncake 做的合并，而是 MLA 的数学结构决定的。MLA 的 K/V 被压缩为一个低秩潜向量，每个 token 每层的存储变成了：

```
MLA 融合张量: [kv_lora_rank + qk_rope_head_dim + index_head_dim] × sizeof(dtype)

以 GLM-5 为例: (512 + 64 + 128) × 2 = 1,408 字节/token/层
对比 GQA:     2 × 8 × 128 × 2 = 4,096 字节/token/层 (LLaMA-70B, tp=8)
```

> 笔者注：K 和 V 在 Mooncake 中的"能见度递减"不是设计缺陷，而是**正确的分层抽象**——传输引擎不需要知道数据的语义，只需要知道地址和长度。就像快递公司不需要知道包裹里装的是书还是衣服，只需要知道收件地址和重量。但调试时要注意：当你看到传输引擎的日志里出现 160 个地址时，前 80 个是 K、后 80 个是 V——这个顺序对应关系在代码中没有显式记录，全靠注册时的遍历顺序保证。

##### 具体数字

| 模型 (per tp_rank) | num_kv_heads/rank | block_len (K或V) | block_len (K+V/层) | 全模型/块 |
|-------------------|-------------------|-----------------|-------------------|----------|
| Mistral-7B (tp=1) | 32 | 128 KB | 256 KB | 32×256KB = **8 MB** |
| LLaMA-70B (tp=8) | 1 | 4 KB | 8 KB | 80×8KB = **640 KB** |
| LLaMA-70B (tp=4) | 2 | 8 KB | 16 KB | 80×16KB = **1.28 MB** |
| DeepSeek-V3 (tp=8, MLA) | — | 16 KB | 16 KB¹ | 61×16KB = **976 KB** |

¹ MLA 模型 K 和 V 不分开存储，压缩为一个张量，所以只有一份 block_len。

##### 一个请求的 KV Cache 有多大？

以 4096 token 的请求为例，`block_size = 16`，需要 256 个块：

| 模型 (per tp_rank) | 全模型/块 | 256 块总大小 |
|-------------------|----------|-------------|
| Mistral-7B | 8 MB | **2 GB** |
| LLaMA-70B (tp=8) | 640 KB | **160 MB** |
| DeepSeek-V3 (tp=8) | 976 KB | **244 MB** |

> 笔者注：Mistral-7B 一个 4096 token 请求的 KV Cache 就要 2 GB！这就是为什么 KV Cache 是 GPU 显存的最大消耗者——80 GB 的 HBM 最多同时服务 30-40 个请求。而 LLaMA-70B 用了 GQA，同样 4096 token 只需 160 MB——差距 12 倍。

##### 块在 GPU 显存中的物理位置

```
GPU 显存 (HBM)
┌──────────────────────────────────────────────────┐
│  Layer 0, K: [block_0][block_1]...[block_N]       │  ← 一个连续张量
│  Layer 0, V: [block_0][block_1]...[block_N]       │  ← 另一个连续张量
│  Layer 1, K: [block_0][block_1]...[block_N]       │
│  Layer 1, V: [block_0][block_1]...[block_N]       │
│  ...                                              │
│  Layer 79, K: [block_0][block_1]...[block_N]      │
│  Layer 79, V: [block_0][block_1]...[block_N]      │
└──────────────────────────────────────────────────┘

  LLaMA-70B (tp=8): 80 层 × 2 (K+V) = 160 个张量
  DeepSeek-V3 (tp=8): 61 层 × 1 (MLA) = 61 个张量
```

每个张量在 GPU 显存中是**连续分配**的，块 ID 就是这个张量内的偏移量：

```
block 的 GPU 地址 = 张量基地址 + block_id × block_len
```

这个简单的地址计算公式，是 MooncakeConnector 实现 RDMA 直传的基础——知道了基地址和 block_id，就能直接算出远程 GPU 上对应块的地址。

---

### 变身 3：RDMA 线路上的原始字节流

当 Prefill 节点要把 KV Cache 传给 Decode 节点时，数据通过 RDMA WRITE 直接写入远程 GPU 显存。**线路上没有序列化、没有协议帧、没有头部——就是 GPU 内存的原始字节拷贝。**

##### MooncakeConnector 的传输地址计算

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
# _build_transfer_params()

for local_layer_addr, remote_layer_addr in zip(
    local_base_addr, remote_base_addr
):
    for group_local_block_id, group_remote_block_id in zip(
        group_local_block_ids, group_remote_block_ids
    ):
        src_ptrs.append(
            local_layer_addr + group_local_block_id[0] * block_len
        )
        dst_ptrs.append(
            remote_layer_addr + group_remote_block_id[0] * block_len
        )
        lengths.append(block_len * len(group_local_block_id))
```

关键设计：
1. **逐层传输**：每个层的 K/V 张量独立传输——LLaMA-70B (tp=8) 有 160 个张量，每个请求触发 160 路并行 RDMA WRITE
2. **连续块合并**：如果 block_id 是连续的 [3,4,5]，合并为一次传输，长度 = 3 × block_len
3. **零拷贝**：源地址直接指向 GPU 显存，目标地址直接指向远程 GPU 显存——RDMA WRITE 绕过两端 CPU

##### 一个 LLaMA-70B 请求的传输全景

```
Prefill 节点 (tp=8, rank 0)          Decode 节点 (tp=8, rank 0)

Layer 0, K:
  src: gpu_addr_0 + 3×4KB = 12KB   ──RDMA WRITE──→  dst: gpu_addr_0' + 7×4KB = 28KB
  src: gpu_addr_0 + 6×4KB = 24KB   ──RDMA WRITE──→  dst: gpu_addr_0' + 9×4KB = 36KB
  ... (每个连续块组合并)

Layer 0, V:
  src: gpu_addr_1 + 3×4KB           ──RDMA WRITE──→  dst: gpu_addr_1' + 7×4KB
  ...

... (80 层 × 2 方向 = 160 个张量)

总计: 256 块 × 160 张量 = 40,960 次 RDMA WRITE (未合并)
     合并后约 5,000-10,000 次 (取决于块连续性)
     总数据量: 160 MB
```

##### 元数据：D 节点告诉 P 节点"往哪写"

传输前，Decode 节点通过 ZMQ 把自己的地址信息发给 Prefill 节点：

```python
class MooncakeAgentMetadata(msgspec.Struct):
    remote_hostname: str           # D 节点的 IP
    remote_port: int               # D 节点的 Transfer Engine RPC 端口
    request_ids: list[ReqId]       # 需要传输的请求 ID
    kv_caches_base_addr: list[int] # D 节点每个层张量的 GPU 基地址 (160个)
    block_ids: list[list[int]]     # D 节点上每个请求分配的 block ID
```

`kv_caches_base_addr` 就是那 160 个张量的 `data_ptr()`——GPU 显存中的物理地址。P 节点拿到这些地址后，直接 RDMA WRITE 到这些地址，数据就出现在 D 节点的 GPU 显存中——**不需要 D 节点做任何拷贝**。

---

### 变身 4：DRAM 中的 Slice 切片

当 KV Cache 进入 Mooncake Store 的分布式存储时，它的形态从"GPU 张量"变成了"对象 + 切片"。

##### 从 GPU 张量到 Slice

Mooncake Store 不理解"层"和"块"的概念——它只认"对象"和"切片"。一个 KV Cache 对象被拆分为多个 Slice：

```cpp
// mooncake-store/include/types.h
struct Slice {
    void* ptr{nullptr};    // 内存地址
    size_t size{0};        // 字节数
};
```

拆分规则：

```
kMaxSliceSize = 16,777,200 字节 (~16 MB)

一个 160 MB 的 KV Cache 对象:
  Slice 0: [0, 16 MB)
  Slice 1: [16 MB, 32 MB)
  ...
  Slice 9: [144 MB, 160 MB)

共 10 个 Slice，每个 ~16 MB
```

##### Slice 在 DRAM 中的分布

每个 Slice 可能被分配到不同的 DRAM 段（不同节点），实现负载均衡：

```
Master 决策: 10 个 Slice 分配到 3 个节点

节点 A (DRAM 段):  Slice 0, Slice 3, Slice 6, Slice 9
节点 B (DRAM 段):  Slice 1, Slice 4, Slice 7
节点 C (DRAM 段):  Slice 2, Slice 5, Slice 8
```

远程节点通过 RDMA READ 拉取 Slice 数据——不需要传输整个对象，只需要拉取需要的 Slice。

##### 对象键的格式

```cpp
// mooncake-store/include/types.h
enum class ObjectDataType : uint8_t {
    UNKNOWN = 0,
    KVCACHE = 1,    // ← KV Cache 对象
    TENSOR = 2,
    WEIGHT = 3,
    ...
};
```

Store 的键是字符串，多租户场景用 NUL 字节分隔租户 ID 和对象键：

```
"tenant_id\0request_12345_layer_0_block_3"
```

> 笔者注：Slice 是 Mooncake Store 的"最小调度单位"。一个 160 MB 的 KV Cache 被拆成 10 个 ~16 MB 的 Slice，分散到不同节点——这就像把一个大包裹拆成 10 个小包裹，分别送到不同的仓库。拉取时也只需要拉取需要的 Slice，不用拉整个对象。

---

### 变身 5：EP 分发中的 FP8 压缩包

在 MoE 模型的专家并行中，KV Cache 的形态又不一样——它不是按"层+块"存储的，而是按"token+专家"分发的。

##### Dispatch 数据包格式

每个 token 发送给专家时，数据包的格式：

```
┌─────────────────────────────────────────────────────────────┐
│  BF16 模式:                                                  │
│  [src_token_idx: 16B (int4)] [hidden_data: kHidden × 2B]    │
│                                                              │
│  FP8 模式:                                                   │
│  [src_token_idx: 16B (int4)] [hidden_data: kHidden × 1B]    │
│  [scales: (kHidden/128) × 4B]                               │
└─────────────────────────────────────────────────────────────┘
```

以 DeepSeek-V3 为例（`kHidden = 7168`）：

| 模式 | src_token_idx | hidden_data | scales | 总计 |
|------|-------------|-------------|--------|------|
| BF16 | 16 B | 14,336 B | — | **14,352 B** |
| FP8 | 16 B | 7,168 B | 224 B | **7,408 B** |

FP8 把数据量压缩到 BF16 的 **51.6%**——几乎减半。

##### FP8 量化细节

```
量化: 每 128 个隐藏元素共享一个 scale
  scale = 448 / max(|x[0:128]|)    ← E4M3 格式，amax=448
  x_fp8 = round(x / scale)          ← 量化到 int8 范围

反量化: 接收端在专家计算前恢复
  x_bf16 = x_fp8 × scale            ← 乘回原始尺度
```

注意：**Combine 阶段不用 FP8**——专家计算后的输出直接以 BF16 精度发回。只有 Dispatch（去程）用 FP8 压缩，Combine（回程）保持 BF16 精度。原因是专家输出的精度更敏感，FP8 量化可能影响模型质量。

##### 一个 DeepSeek-V3 请求的 EP 数据流

```
4096 token 请求, top_k=8, 256 专家, 8 Rank:

Dispatch (token → 专家):
  每个 token 选 8 个专家 → 4096 × 8 = 32,768 条路由
  平均每个 Rank 处理: 32,768 / 8 = 4,096 条路由
  FP8 数据量: 4,096 × 7,408 B ≈ 30 MB (per rank pair)
  
  All-to-All: 8×8 = 64 对 rank 之间的 RDMA 传输

Combine (专家 → token):
  专家计算完成 → 输出以 BF16 发回
  BF16 数据量: 4,096 × 14,352 B ≈ 59 MB (per rank pair)
  
  All-to-All: 同样 64 对 rank 之间的 RDMA 传输
```

> 笔者注：EP 的数据形态和 PD 传输完全不同。PD 传的是"层+块"维度的 KV Cache（按 GPU 内存布局原样搬运），EP 传的是"token+专家"维度的隐藏状态（按路由逻辑重新打包）。PD 是"搬仓库"，EP 是"送快递"——搬仓库不用拆包，送快递必须按收件人重新分拣。

---

### 变身 6：SSD 磁盘上的桶文件

当 DRAM 装不下，KV Cache 被卸载到 SSD。在磁盘上，它变成了桶文件中的二进制记录。

##### 从 Slice 到桶文件

卸载时，多个 KV Cache 对象的 Slice 被合并写入同一个桶文件：

```
一个桶文件最多包含:
  - 256 MB 数据
  - 500 个 key

桶文件布局:
┌─────────────────────────────────────────────────────┐
│  data 区域                                           │
│  [key1_data][key2_data][key3_data]...[keyN_data]     │
│                                                      │
│  每个 key_data 就是 KV Cache Slice 的原始字节          │
│  (从 DRAM 直接写入，无序列化、无压缩)                  │
├─────────────────────────────────────────────────────┤
│  metadata 区域                                       │
│  meta_size | data_size | keys[] | metadatas[]        │
│                                                      │
│  BucketObjectMetadata:                               │
│    {offset: 在 data 区域的偏移,                       │
│     key_size: key 长度,                              │
│     data_size: 数据长度}                              │
└─────────────────────────────────────────────────────┘
```

##### 一个 LLaMA-70B 请求在 SSD 上的样子

```
LLaMA-70B (tp=8), 4096 token 请求, 160 MB KV Cache:

在 GPU 显存中:
  160 个张量 × 256 块 = 40,960 个 block
  每个 block 4 KB

在 Mooncake Store 中:
  1 个对象, 10 个 Slice (每个 ~16 MB)
  对象键: "request_12345"

在 SSD 桶文件中:
  10 条记录 (每个 Slice 一条)
  每条记录: [key: "request_12345_slice_0"][data: 16MB 原始字节]
  合并到同一个桶文件中 (10 × 16MB = 160MB < 256MB 桶上限)
  
  读取时: BatchLoad() → O_DIRECT 直读到对齐缓冲区 → 零拷贝返回
```

##### 六种形态的尺寸对比

以 LLaMA-70B (tp=8)、4096 token 请求为例：

| 形态 | 存储位置 | 数据量 | 格式 |
|------|---------|--------|------|
| Attention K/V 矩阵 | 数学定义 | 320 KB/token | 2 × 8 × 128 × bf16 |
| GPU 分页块 | GPU 显存 | 160 MB | 160 个张量 × 256 块 × 4KB |
| RDMA 字节流 | 网络 | 160 MB | 原始字节，无帧头 |
| Store Slice | DRAM | 160 MB | 10 个 Slice × 16MB |
| EP FP8 包 | 网络 | ~30 MB/方向 | token_idx + hidden_fp8 + scales |
| SSD 桶记录 | NVMe SSD | 160 MB | 桶文件中的 key-value 记录 |

> 笔者注：除了 EP FP8 包，其他五种形态的数据量都是 160 MB——数据没有变，只是"包装"不同。RDMA 传输是零拷贝的原始字节，Store 切片是逻辑拆分，SSD 桶文件是物理合并。**数据本身从未被压缩或变换，只是组织方式在变。** EP 是唯一的例外——它对隐藏状态做了 FP8 量化，数据量减半，但这是为了降低 All-to-All 通信开销的必要代价。

---

### 数据变身全流程：一张图看清

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Prefill 节点 (GPU)                            │
│                                                                      │
│  变身1: Attention K/V 矩阵                                          │
│    Q × K^T → Attention Weights → × V                                │
│    K: [num_kv_heads, head_dim] per token per layer                  │
│    V: [num_kv_heads, head_dim] per token per layer                  │
│         │                                                            │
│         ▼                                                            │
│  变身2: GPU 分页块                                                   │
│    [num_blocks, block_size, num_kv_heads, head_dim]                 │
│    block_len = 16 × 1 × 128 × 2 = 4 KB (LLaMA-70B, tp=8)          │
│         │                                                            │
│         ├───── RDMA WRITE ──────────────────────────────┐           │
│         │        变身3: 原始字节流                        │           │
│         │        src = base_addr + block_id × 4KB       │           │
│         │        dst = remote_base + block_id × 4KB     │           │
│         │        零拷贝，无帧头，无序列化                 │           │
│         │                                                │           │
│         ▼                                                ▼           │
│  Mooncake Store (DRAM)                          Decode 节点 (GPU)   │
│                                                                  │   │
│  变身4: Slice 切片                              变身2: 分页块      │   │
│    10 × 16MB Slice                              (直接可用)         │   │
│    分散到不同 DRAM 段                                              │   │
│         │                                                          │   │
│         ▼                                                          │   │
│  SSD (NVMe)                                                        │   │
│                                                                  │   │
│  变身6: 桶文件记录                                                │   │
│    [key][16MB data] × 10                                         │   │
│    合并到 256MB 桶文件                                            │   │
│                                                                  │   │
└─────────────────────────────────────────────────────────────────────┘

  MoE 专家并行 (EP) 的另一条路径:

  变身1: Token 隐藏状态
       │
       ▼ FP8 量化
  变身5: EP FP8 包
    [token_idx: 16B][hidden_fp8: 7KB][scales: 224B]
    All-to-All → 远程专家计算 → BF16 Combine 回传
```

---

### 三种注意力机制的 KV Cache 对比

最后，用一张表总结三种注意力机制下 KV Cache 的本质差异：

| 维度 | MHA | GQA | MLA |
|------|-----|-----|-----|
| 代表模型 | Mistral-7B | LLaMA-70B | DeepSeek-V3 |
| KV 头数 | = Q 头数 | < Q 头数 | 压缩为潜向量 |
| 存储 | K 和 V 分开 | K 和 V 分开 | K/V 合并为一个张量 |
| split_k_and_v | True | True | **False** |
| 张量数/层 | 2 (K+V) | 2 (K+V) | **1 (压缩KV)** |
| 压缩比 | 1x | num_attn/num_kv 倍 | ~28x vs MHA |
| GPU 块布局 | [blocks, 16, kv_heads, 128] | [blocks, 16, kv_heads/rank, 128] | [blocks, 16, kv_lora_rank+rope_dim] |
| PD 传输 | 2L 个张量 | 2L 个张量 | **L 个张量** |
| EP 传输 | 标准 hidden | 标准 hidden | 标准 hidden |

> 笔者注：MLA 模型（如 DeepSeek-V3）的 KV Cache 在 Mooncake 中有一个重要差异——`split_k_and_v = False`，意味着 K 和 V 不分开存储，每个层只有一个压缩张量。这直接影响了 PD 传输的张量数量（61 vs 160）和 EP 的 buffer 布局。如果你的系统同时服务 MHA/GQA 和 MLA 模型，务必注意这个差异。

---

### 总结与行动指南

| 变身 | 存储介质 | 数据格式 | 关键数字 (LLaMA-70B, tp=8) |
|------|---------|---------|--------------------------|
| Attention K/V | 数学定义 | 矩阵 | 320 KB/token |
| GPU 分页块 | HBM | [blocks, 16, kv_heads, 128] | 4 KB/block, 160 个张量 |
| RDMA 字节流 | 网络 | 原始字节 | 160 MB, 零拷贝 |
| Store Slice | DRAM | ptr+size | 10 × 16MB |
| EP FP8 包 | 网络 | idx+hidden+scales | 7.4 KB/token, 压缩 48% |
| SSD 桶记录 | NVMe | key+value 二进制 | 160 MB, 合并到桶文件 |

**一行建议**: 理解 KV Cache 的六次变身后，调优和排障就有了锚点——PD 传输慢，看 GPU 块布局和 RDMA 地址计算；Store 吞吐低，看 Slice 大小和 DRAM 段分布；EP 延迟高，看 FP8 量化参数和 All-to-All buffer 配置。

**延伸阅读**：
- vLLM PagedAttention 论文：https://arxiv.org/abs/2309.06180
- DeepSeek-V3 MLA 技术报告：https://arxiv.org/abs/2412.19437
- FP8 格式规范：https://arxiv.org/abs/2209.05433

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
