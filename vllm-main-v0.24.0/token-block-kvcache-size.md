# 【硬核拆解vLLM】Token 长短不一，Block 大小凭什么固定？——详解 Token、Token ID、Block Size 与 KV Cache 物理空间的关系

> **系列**: vLLM 技术博客系列 | **类型**: 核心概念深潜篇
> 一句话说透：Block 装的是"向量"，不是"文字"——不管文字多长多短，向量大小都一样

### 引言

你去超市买东西，结账时收银员扫条形码。一瓶矿泉水的条形码是 13 位数字，一袋大米的条形码也是 13 位数字。矿泉水很小，大米很大，但**条形码的长度是一样的**——收银系统根本不关心商品多大多重，它只认条形码。

大模型处理文字也是这样。当你看到 `block_size=16` 时，可能会困惑：每个 token 的文本长度不同——"a" 是 1 个字符，"supercalifragilisticexpialidocious" 是 34 个字符，那每个 block 的物理空间大小是不是也不同？今天我们就来彻底拆解这条从文字到 Block 物理存储的完整链路。

> 一个Token到KV Cache存储的起点与终点：Token → Token ID（固定整型长度，一般4字节） → Token KV（计算向量，比如128维） → Block（比如16个Token KV打包成一个Block，进行存储）
---

### 一、文字先被切成小块（Token）

大模型不能直接读文字，它先把文字切成一个个小块，叫 **Token**。 专业解释在LLM系统中做这个事情的叫分词器，token 是模型处理文本时的最小单位，可以是一个词、一个字，甚至是一个字符片段（比如“un-”或“ing”），具体取决于分词切分算法。

```
原文:  "我 喜欢 猫"

切成 token:  ["我", "喜欢", "猫"]
                ↑       ↑       ↑
              1个字   2个字   1个字    ← 每个 token 的文字长短不一
```

就像你把一句话切成几个词。有的词短（"我"），有的词长（"喜欢"），这很正常。

问：在vLLM系统中Token分词的英文术语叫什么？ 将每个Token转换成Token ID的英文术语叫什么？

答： 文本 → Token即分词的转换，英文术语叫Tokenization，由 Tokenizer（分词器）执行，vLLM 使用 HuggingFace 的 tokenizer。

Token → Token ID的转换，英文术语叫Encoding（注意这里的encode只是纯粹的Token到Token ID的转换，不是Transformer架构中的encoder，下一篇会详细讲解），同样由 Tokenizer 完成，tokenizer.encode(text) 返回 prompt_token_ids。

实际上在 vLLM 中这两步是合一的——Tokenizer 的 encode() 方法一步完成 tokenization + encoding，直接输出 prompt_token_ids: list[int]。源码中字段名就叫 prompt_token_ids，不区分中间的 token 文本。


**但这一步只是把文字切成了小块，模型还完全不认识它们——要等下一步换成数字才行。**

---

### 二、每个 Token 换成一个编号（Token ID）

切好之后，模型给每个 token 分配一个**编号**，就像超市给每个商品一个条形码。

```
Token:    "我"    "喜欢"    "猫"
           ↓       ↓        ↓
Token ID:  234     5678     9012     ← 都是整数编号，大小一样
```

这个编号来自模型的"词汇表"——一本字典，里面每个词对应一个数字：

```
词汇表（就像商品目录）:
  编号    词
  ────    ────
  0       (空白)
  1       (开头标记)
  ...
  234     我
  ...
  5678    喜欢
  ...
  9012    猫
  ...
  32000   (结尾标记)        ← 词汇表大小 vocab_size = 32000
```

##### 关键发现

| 问题 | 回答 |
|---|---|
| "我"是 1 个字，"喜欢"是 2 个字，编号会不一样大吗？ | **不会**。编号就是字典里的行号，跟文字长短没关系 |
| 编号占多少空间？ | 固定的，就是一个整型数字（4 个字节） |

**从这一步开始，文字的长短已经不重要了**——所有 token 都变成了同样格式的数字编号。虽然有的是 234，有的是 9012，但从编程的角度，都是一个固定长度为 4 个字节的整型数字。全部 Token ID 都是！

> 💡 **核心洞察**: Token ID 是"文本世界"到"数值世界"的分水岭——跨过这道门，后面所有东西的大小都和文本长度无关了。

---

### 三、编号变成向量（这才是 KV Cache 存的东西）

有了编号之后，模型用这个编号去"查表"，得到一个**向量**（一串数字）。

你可以把向量想象成一张**特征卡片**——模型为每个 token 算出一张卡片，上面记录了这个 token 的各种"特征值"。

```
Token ID:  234         5678          9012
           ↓            ↓             ↓
Key 向量:  [0.12, -0.34, ...]  [0.56, 0.78, ...]  [0.91, -0.02, ...]
Value 向量: [0.33, 0.45, ...]  [-0.12, 0.89, ...]  [0.67, -0.55, ...]
           ↑                                              ↑
     都是 128 维的数字特征列表                           也是 128 维的数字特征列表
```

**不管原文是"我"（1 个字）还是"supercalifragilisticexpialidocious"（34 个字母），算出来的向量都是 128 维**——就像不管商品大小，条形码都是 13 位。

这就是 KV Cache 里真正存的东西：**每个 token 的 Key 向量和 Value 向量**。

##### 向量在系统中的数据类型

在 vLLM 源码中，KV Cache 的向量存储为 **`torch.Tensor`**（PyTorch 张量），其数据类型由 `AttentionSpec.dtype` 字段决定：

```python
# vllm/v1/kv_cache_interface.py
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype          # ← 这里决定向量的精度
    kv_quant_mode: KVQuantMode  # 量化模式
```

| 配置 | dtype 值 | 每个数字占 |
|---|---|---|
| FP16 | `torch.float16` | 2 字节 |
| BF16 | `torch.bfloat16` | 2 字节 |
| FP8 | `torch.float8_e4m3fn` | 1 字节 |

底层分配时，先用 `torch.int8` 分配原始内存（最细粒度），再 `.view(dtype)` 转换成目标精度：

```python
# vllm/v1/worker/gpu/attn_utils.py
kv_raw_tensor = torch.zeros(size, dtype=torch.int8, device=device)
kv_cache = kv_raw_tensor.view(kv_cache_spec.dtype).view(kv_cache_shape)
```

以 FlashAttention 后端为例，Tensor 的 shape 为：

```python
# vllm/v1/attention/backends/flash_attn.py
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
    return (num_blocks, 2, block_size, num_kv_heads, head_size)
    #        ↑         ↑    ↑           ↑              ↑
    #      block数   K+V  每block     KV head数      head维度
    #                合在一起  token数
```

---

### 四、每个 Token 的 KV 向量到底多大

向量的大小由**模型的设计图纸**决定，跟文字内容完全无关：

```
每个 token 的 KV 大小 = 2 × 层数 × KV头数 × 头维度 × 数据类型字节数
                        ↑    ↑      ↑       ↑          ↑
                      K+V  模型    模型     模型       你选的
                           有几层  几个头   多宽      精度
```

你不用记这个公式，只需要知道：**公式里没有任何一项跟"文字长短"有关**。

用 LLaMA-70B 举个例子：

```
层数     = 80
KV头数   = 8    （GQA，8 个 KV head）
头维度   = 128
数据类型 = FP16（每个数字占 2 字节）

每个 token 的 KV 大小 = 2 × 80 × 8 × 128 × 2 = 327,680 字节 ≈ 320 KB
```

来对比一下：

| Token 文本 | 文字多长 | KV 向量多大 |
|---|---|---|
| "a" | 1 个字母 | **320 KB** |
| "你好" | 2 个汉字 | **320 KB** |
| "supercalifragilisticexpialidocious" | 34 个字母 | **320 KB** |

**一模一样。320 KB，不多不少。**

> 笔者注：有趣的是，LLaMA-7B 每个 token 的 KV 反而比 70B 大（512 KB vs 320 KB）——因为 7B 没用 GQA，KV head 数更多（32 vs 8）。所以"模型越小，KV 越小"这个直觉是错的，关键看模型结构。

---

### 五、从 Token KV 向量到 Block

Block 就是把若干个 token 的 KV 向量打包在一起。

```
block_size = 16  （一个 block 装 16 个 token 的 KV）

┌─────────────── Block 0 ───────────────┐
│ token 0 的 KV │ token 1 的 KV │ ... │ token 15 的 KV │
│   320 KB      │   320 KB      │     │   320 KB       │
└───────────────────────────────────────┘
              16 × 320 KB = 5,120 KB ≈ 5 MB
```

**每个 block 都是 5 MB，固定不变。**

就像一排储物柜，每个柜子大小一样，不管你往里面放的是一双袜子还是一件羽绒服——柜子的大小是固定的，变的是你需要多少个柜子。

##### 不同模型的 Block 大小对比

| | LLaMA-7B FP16 | LLaMA-70B FP16 | LLaMA-70B FP8 |
|---|---|---|---|
| 每 token KV | 512 KB | 320 KB | 160 KB |
| block_size | 16 | 16 | 16 |
| **每 block 物理大小** | **8 MB** | **5 MB** | **2.5 MB** |

**同一个模型配置下，每个 block 的物理大小完全相同。**

---

### 六、文字长短到底影响了什么

文字长短不影响"每个柜子多大"，但影响"你需要几个柜子"：

```
短文本: "你好" → 2 个 token → ceil(2/16) = 1 个 block → 总空间 = 1 × 5 MB = 5 MB

长文本: 10000 字 → 约 5000 个 token → ceil(5000/16) = 313 个 block → 总空间 = 313 × 5 MB ≈ 1.5 GB
```

| | 短文本 "你好" | 长文本 10000 字 |
|---|---|---|
| token 数量 | 2 | ~5000 |
| 每个 token 的 KV 大小 | 320 KB | 320 KB |
| block 数量 | 1 | 313 |
| 每个 block 物理大小 | 5 MB | 5 MB |
| **总 KV Cache 空间** | **5 MB** | **~1.5 GB** |

**变的是数量，不变的是单个的大小。**

---

### 七、完整数据流：一张图看全貌

```
你的文字                    模型内部
─────────────────────────────────────────────────────────

"我 喜欢 猫"               
    │                      
    ▼ 切词                 
["我","喜欢","猫"]         ← token 长短不一（1字、2字、1字）
    │                      
    ▼ 查编号               
[234, 5678, 9012]          ← token ID 都是整数，大小一样（4字节）
    │                      
    ▼ 算向量               
[K向量, K向量, K向量]       ← 每个都是 128 维，大小一样
[V向量, V向量, V向量]       ← 每个都是 128 维，大小一样
    │                      
    ▼ 打包成 block         
┌──────── Block 0 ────────┐
│ 234的KV │ 5678的KV │ 9012的KV │ ... │
│ 320 KB  │ 320 KB   │ 320 KB   │     │
└─────────────────────────┘
         每个 block 固定 5 MB
```

##### 变量全景表

| 变量 | 是否固定 | 由谁决定 |
|---|---|---|
| Token ID 大小 | 固定 | 数据类型（通常 int32 = 4 bytes） |
| 每个 token 的 KV 向量大小 | 固定 | 模型结构（层数、head 数、维度）+ 数据类型 |
| block_size | 固定 | 配置参数（默认 16） |
| 每个 block 的物理大小 | 固定 | block_size × 每 token KV 大小 |
| token 数量 | 变长 | 文本内容 + 分词器 |
| block 数量 | 变长 | token 数量 / block_size |
| 总 KV Cache 物理空间 | 变长 | block 数量 × block 物理大小 |

---

### 八、两个比喻，一个结论

##### 比喻一：超市购物

| 概念 | 比喻 | 大小 |
|---|---|---|
| Token 文本 | 商品的名字 | 长短不一 |
| Token ID | 商品的条形码 | 固定 13 位 |
| KV 向量 | 商品在仓库里的储位 | 固定大小 |
| Block | 仓库里的一排货架 | 固定容量（16 个储位） |

- 矿泉水和大米，条形码都是 13 位——**名字长短不影响条形码长度**
- 仓库里每个储位大小一样——**条形码不影响储位大小**
- 一排货架放 16 个储位——**每排货架的物理空间固定**
- 买得多就需要更多排货架——**变的是排数，不变的是每排的大小**

##### 比喻二：飞机乘客

| 概念 | 比喻 | 大小 |
|---|---|---|
| Token 文本 | 乘客的姓名 | 长短不一 |
| Token ID | 登机牌编号 | 固定（一个数字） |
| KV 向量 | 飞机上的座位 | 固定大小 |
| Block | 一排座椅 | 固定容量（16 个座位） |

- 乘客叫"李明"还是"亚历山大·伊万诺维奇"，登机牌编号都只是一个数字——**姓名长短不影响编号大小**
- 拿着编号找到座位，每个座位大小完全一样——**编号大小不影响座位大小**
- 座椅的大小由飞机制造商决定（模型结构），不由乘客决定
- 一排 16 个座位（block_size=16），每排的物理空间完全一样
- 航班上有多少乘客（token 数量），决定了需要多少排座位（block 数量）
- **变的是排数，不变的是每排的大小**

---

### 总结

| 问题 | 答案 |
|---|---|
| token 文本长短影响 KV 向量大小吗？ | **不影响**。向量由模型结构决定，和文字无关 |
| token 文本长短影响 block 物理大小吗？ | **不影响**。block = block_size × 每 token KV，两项都固定 |
| token 文本长短影响什么？ | 影响 token 数量 → block 数量 → 总 KV Cache 空间 |
| 从哪一步开始大小就固定了？ | **Token ID**。编号就是整数，后面全是固定维度的向量 |

**一句话：文字长短只影响你需要多少个 block，不影响每个 block 有多大。**

### 延伸阅读

- vLLM 源码：[vllm/v1/kv_cache_interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py) — `AttentionSpec` 定义
- vLLM 源码：[vllm/v1/worker/gpu/attn_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/attn_utils.py) — KV Cache 分配与 view 转换
- vLLM 源码：[vllm/v1/attention/backends/flash_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) — `get_kv_cache_shape()`

---

*本文属于 [vLLM 技术博客系列]，欢迎持续关注。*
