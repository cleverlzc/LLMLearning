# vLLM Offload Memory 详解：GPU 装不下？借 CPU 的"仓库"用用

> **系列**: vLLM 技术博客系列 | **类型**: 核心概念深潜篇
> GPU 显存不够用时，别急着加卡——先把用不上的 KV Cache 搬到 CPU 内存，需要时再搬回来

进入正文之前，先简单科普一下offload memory，有个全局了解。"Offload memory"描述的是‌动作/机制‌（卸载到内存），并非某个特定开源 KV Cache 项目的专有名称，而是指‌将 KV Cache 从 GPU 显存卸载（Offload）至 CPU 内存或磁盘的通用技术策略‌；主流开源实现包括 vLLM 内置 CPU Offload‌、‌SGLang HiCache 及独立组件 LMCache‌。

当前业界offload memory，核心对应项目与特征：
- vLLM (Native CPU Offload)‌：推理引擎原生支持，通过 --kv-offloading-size 参数将 KV Block 移至 CPU 内存，不足时触发重算，适合单机显存受限场景 。
- SGLang HiCache‌：内置分层缓存系统（HiRadixCache），支持 GPU→CPU 内存→分布式存储（如 Mooncake/3FS）三级卸载，具备异步预取和零拷贝传输优化 。
- LMCache‌：独立守护进程式缓存引擎，可插拔后端（CPU RAM/本地磁盘/远程存储），支持非前缀缓存复用（CacheBlend），常作为 vLLM/SGLang 的外部插件使用 。
- Transformers (Hugging Face)‌：库级实现 cache_implementation="offloaded"，在层间迭代时动态将历史 KV 移至 CPU，适用于单卡小批量推理 。‌

选型建议‌：若需深度集成且追求低延迟，选 vLLM 原生（即自带的offload memory）‌或‌SGLang HiCache‌；若需跨引擎共享、持久化或非前缀复用，选 LMCache‌。‌最终的选择，可以根据实测 + 实际需求来看，也可以动态调整切换。

个人认为，vLLM自身发展比较快，上周就更新了一个大版本，演进到了V2版本，已经比较成熟，选型时可以优先考虑。如果不比三方组件性能差，维护等成本会降低，也不失为一种好的选择。


### 引言

想象你经营一家生意火爆的便利店。货架（GPU 显存）空间有限，但商品（KV Cache）越来越多。你有两个选择：一是租更大的店面（加 GPU），成本高昂；二是在隔壁租一个仓库（CPU 内存），把暂时不卖的商品先存过去，需要时再搬回来。

这就是 vLLM 的 Offload Memory 机制——**当 GPU 显存装不下所有 KV Cache 时，把"冷数据"搬到 CPU 内存，需要时再搬回 GPU**。听起来简单，但背后涉及一套精密的数据搬运、缓存淘汰和多级存储系统。

今天我们就把这套机制拆透：搬什么、怎么搬、搬多快、什么时候搬回来。

---

### 一、为什么需要 Offload：GPU 显存永远不够用

##### 1.1 核心矛盾

在 vLLM 的推理过程中，KV Cache 是显存消耗的大头。以 LLaMA-70B FP16 为例：

```
每个请求 4K 上下文的 KV Cache ≈ 1.25 GB
100 个并发 = 125 GB

模型权重 ≈ 140 GB
总需求 ≈ 265 GB → 至少 4 张 H100
```

但现实是：**不是所有 KV Cache 都在同时被使用**。当一个请求处于 decode 阶段时，它只需要最近几步的 KV Cache 做注意力计算，之前的大量 KV Cache 虽然"还在用"，但访问频率很低。

这就像便利店里，有些商品是"热销品"（正在 decode 的 KV Cache），需要放在货架上随手可取；有些是"冷门品"（已完成 prefill 但还在等待的 KV Cache），可以暂存仓库，需要时再拿。

##### 1.2 Offload 解决什么问题

| 问题 | Offload 如何解决 |
|:---|:---|
| GPU 显存不足，无法装下所有 KV Cache | 把冷 KV Cache 搬到 CPU 内存，释放 GPU 空间 |
| 长上下文请求占用大量 KV Cache | prefill 完成后立即 offload，decode 时再 load 回来 |
| 多租户场景下 KV Cache 竞争 | CPU 内存作为"缓冲池"，缓解 GPU 上的空间压力 |
| RLHF 场景需要频繁切换模型 | Sleep Mode 把模型权重 offload 到 CPU，腾出空间给另一个模型 |

---

### 二、Offload 的三大场景

vLLM 的 Offload 机制覆盖三个不同的场景，它们的目标和实现各不相同：

```
┌──────────────────────────────────────────────────────────────┐
│                  vLLM Offload 三大场景                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  KV Cache    │  │  KV Cache    │  │  模型权重     │       │
│  │  CPU Offload │  │  多级分层     │  │  Sleep Mode  │       │
│  │  (单级)      │  │  (GPU→CPU→FS)│  │  (GPU→CPU)   │       │
│  │              │  │              │  │              │       │
│  │  GPU ↔ CPU  │  │  GPU↔CPU↔磁盘│  │  GPU ↔ CPU   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│       场景1              场景2             场景3              │
└──────────────────────────────────────────────────────────────┘
```

| 场景 | 搬什么 | 搬到哪 | 什么时候搬回来 | 典型用途 |
|:---|:---|:---|:---|:---|
| KV Cache CPU Offload | KV Cache 块 | CPU 内存 | 请求 decode 需要时 | 扩大有效 KV Cache 容量 |
| KV Cache 多级分层 | KV Cache 块 | CPU → 文件系统/对象存储 | 逐级提升回 GPU | 超大规模 KV Cache 持久化 |
| 模型权重 Sleep Mode | 模型权重 + KV Cache | CPU 内存 | 切换回该模型时 | RLHF 多模型切换 |

---

### 三、KV Cache CPU Offload：单级搬运

##### 3.1 核心架构

这是最基础的 Offload 场景——KV Cache 在 GPU 和 CPU 内存之间来回搬运。

```
┌─────────────────────┐          ┌─────────────────────┐
│     GPU 显存         │          │     CPU 内存         │
│                     │          │                     │
│  ┌───┐ ┌───┐ ┌───┐ │  Store   │  ┌───┐ ┌───┐ ┌───┐ │
│  │ B │ │ B │ │ B │ │ ───────→ │  │ B │ │ B │ │ B │ │
│  └───┘ └───┘ └───┘ │ (GPU→CPU)│  └───┘ └───┘ └───┘ │
│                     │          │                     │
│  ┌───┐ ┌───┐       │  Load    │  ┌───┐ ┌───┐       │
│  │ B │ │ B │       │ ←─────── │  │ B │ │ B │       │
│  └───┘ └───┘       │ (CPU→GPU)│  └───┘ └───┘       │
└─────────────────────┘          └─────────────────────┘
       热数据                           冷数据
    (正在 decode)                  (等待中/已完成 prefill)
```

##### 3.2 OffloadingManager：搬运的"调度中心"

所有 Offload 操作都通过 `OffloadingManager` 抽象类统一管理：

```python
# vllm/v1/kv_offload/base.py
class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """查找块是否在 CPU 侧，返回 HIT/MISS/HIT_PENDING/RETRY"""

    @abstractmethod
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput | None:
        """准备将块从 GPU 搬到 CPU（可能触发淘汰）"""

    @abstractmethod
    def prepare_load(self, keys, req_context) -> LoadStoreSpec:
        """准备将块从 CPU 搬回 GPU（防止被淘汰）"""

    def complete_store(self, keys, req_context, success=True):
        """标记 Store 完成，块变为可 Load"""
```

四种查找结果：

| LookupResult | 含义 | 比喻 |
|:---|:---|:---|
| `HIT` | 块在 CPU 侧，随时可读 | 仓库里有货，随时可取 |
| `MISS` | 块不在 CPU 侧 | 仓库里没这个货 |
| `HIT_PENDING` | 块正在传输中，还没到 | 货在路上，还没到仓库 |
| `RETRY` | 传输已启动，稍后再试 | 货刚发出，等一下再来 |

##### 3.3 CPUOffloadingManager：单级实现

```python
# vllm/v1/kv_offload/cpu/manager.py
class CPUOffloadingManager(OffloadingManager):
    def __init__(
        self,
        num_blocks: int,                # CPU 侧可容纳的 block 数量
        cache_policy: Literal["lru", "arc"] = "lru",  # 淘汰策略
        enable_events: bool = False,     # 是否启用事件通知
        store_threshold: int = 1,        # 至少被查找几次才允许 offload
        max_tracker_size: int = 64_000,  # 查找追踪器最大条目数
    ):
```

**`store_threshold` 是一个重要参数**：一个 KV Cache 块必须被查找过至少 `store_threshold` 次，才会被 offload 到 CPU。这避免了"刚存进去又要搬回来"的抖动。

##### 3.4 数据搬运的执行：GPU Worker

实际的 GPU↔CPU 数据搬运由 `CPUOffloadingWorker` 完成：

```python
# vllm/v1/kv_offload/cpu/gpu_worker.py
class CPUOffloadingWorker(OffloadingWorker):
    """组合两个单向搬运处理器"""
    def __init__(self, kv_caches, block_size_factor, num_cpu_blocks, mmap_region=None):
        # 两个方向的搬运处理器
        self._store_handler = SingleDirectionOffloadingHandler(...)  # GPU → CPU
        self._load_handler = SingleDirectionOffloadingHandler(...)   # CPU → GPU
```

搬运的关键细节：

| 特性 | 说明 |
|:---|:---|
| 异步传输 | 使用 `cudaMemcpyAsync`，不阻塞 GPU 计算 |
| 独立 CUDA Stream | 每个 handler 有专属 stream，与计算 stream 并行 |
| 流序保证 | 同方向的传输按提交顺序串行执行（`wait_event`） |
| GPU→CPU 等待计算 | Store 方向会 `wait_stream(compute_stream)`，确保计算完成后再搬 |

##### 3.5 共享内存区域：多 Worker 的"公共仓库"

在多卡场景下，所有 Worker 的 CPU Offload 区域通过 `mmap` 共享：

```python
# vllm/v1/kv_offload/cpu/shared_offload_region.py
class SharedOffloadRegion:
    """mmap 支持的内存区域，所有 Worker 共享"""
    def __init__(self, instance_id, num_blocks, rank, kv_bytes_per_block, cpu_page_size):
        self.mmap_path = f"/dev/shm/vllm_offload_{instance_id}.mmap"
```

内存布局是**交错排列**的：

```
worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
...
```

> 笔者注：使用 `/dev/shm`（共享内存文件系统）意味着 CPU Offload 实际使用的是系统 RAM，而非磁盘。读写速度远快于文件系统，但容量受限于物理内存大小。

---

### 四、缓存淘汰策略：仓库满了怎么办

CPU 内存也不是无限的。当 CPU 侧的 Offload 区域满了，就需要淘汰一些"最不可能再用"的块。vLLM 提供了两种淘汰策略：

##### 4.1 LRU（最近最少使用）

```python
# vllm/v1/kv_offload/cpu/policies/lru.py
class LRUCachePolicy(CachePolicy):
    def __init__(self, cache_capacity: int):
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()  # LRU 链表
        self.blocks: dict[OffloadKey, BlockStatus] = {}
```

工作方式很简单——**最久没被访问的块最先被淘汰**：

```
访问顺序: A → B → C → D → B

淘汰前 LRU 链表: [A, B, C, D]  (A 最久没用)
淘汰 1 个: 移除 A

B 被访问后: [C, D, B]  (B 移到末尾)
淘汰 1 个: 移除 C
```

##### 4.2 ARC（自适应替换缓存）

```python
# vllm/v1/kv_offload/cpu/policies/arc.py
class ARCCachePolicy(CachePolicy):
    """四象限自适应策略"""
```

ARC 比 LRU 更智能——它同时跟踪"最近访问"和"频繁访问"两类模式：

```
┌──────────────────────────────────────────────────┐
│                   ARC 四象限                      │
│                                                  │
│  T1: 最近访问的块          T2: 频繁访问的块       │
│  (只被访问过 1 次)         (被访问过多次)          │
│                                                  │
│  B1: T1 淘汰的"幽灵"       B2: T2 淘汰的"幽灵"   │
│  (记录最近被淘汰的块)       (记录频繁被淘汰的块)   │
└──────────────────────────────────────────────────┘
```

ARC 的自适应逻辑：

| 事件 | 调整 | 含义 |
|:---|:---|:---|
| B1 命中（幽灵列表） | 增大 T1 的目标容量 | 最近访问的块更有用，偏向保留"新面孔" |
| B2 命中（幽灵列表） | 减小 T1 的目标容量 | 频繁访问的块更有用，偏向保留"老主顾" |

> 笔者注：ARC 适合访问模式变化较大的场景（如请求的上下文长度波动大），LRU 适合访问模式稳定的场景。默认使用 LRU。

---

### 五、KV Cache 多级分层：GPU → CPU → 文件系统/对象存储

##### 5.1 为什么需要多级

单级 CPU Offload 已经能缓解 GPU 显存压力，但在某些场景下 CPU 内存也不够——比如需要持久化大量 KV Cache 供后续复用，或者多个 vLLM 实例共享 KV Cache。

这时就需要引入更多存储层级：

```
┌──────────┐     ┌──────────┐     ┌──────────────┐
│   GPU    │ ←→  │   CPU    │ ←→  │  二级存储     │
│  (热数据) │     │ (温数据)  │     │ (冷数据)      │
│          │     │  内存     │     │ 文件系统/对象  │
│  最快     │     │  较快     │     │  存储         │
│  最贵     │     │  较贵     │     │  最慢/最便宜  │
└──────────┘     └──────────┘     └──────────────┘
   Level 0         Level 1          Level 2+
```

##### 5.2 TieringOffloadingManager：多级实现

```python
# vllm/v1/kv_offload/tiering/manager.py
class TieringOffloadingManager(OffloadingManager):
    def __init__(self, primary_tier, secondary_tiers):
        self.primary_tier = primary_tier      # CPU 主层级
        self.secondary_tiers = secondary_tiers  # 二级存储列表
```

**级联 Store（GPU → CPU → 二级存储）**：

```
GPU 块 ──Store──→ CPU 主层级 ──级联 Store──→ 二级存储1
                                        ──级联 Store──→ 二级存储2
                                        ──级联 Store──→ ...
```

当块被 Store 到 CPU 后，会**自动级联到所有二级存储**——确保每一级都有副本。

**逐级提升 Load（二级存储 → CPU → GPU）**：

```
1. 先查 CPU 主层级 → HIT？直接 Load 到 GPU
2. CPU 未命中 → 查二级存储 → HIT？先提升到 CPU，再 Load 到 GPU
3. 二级存储也没命中 → MISS，需要重新计算
```

##### 5.3 二级存储类型

vLLM 注册了三种二级存储：

| 类型 | 类名 | 存储位置 | 适用场景 |
|:---|:---|:---|:---|
| `fs` | `FileSystemTierManager` | 本地文件系统 | 单机大容量 KV Cache 持久化 |
| `obj` | `ObjectStoreSecondaryTierManager` | 对象存储（S3 等） | 跨实例 KV Cache 共享 |
| `example` | `ExampleSecondaryTierManager` | 示例实现 | 开发测试 |

二级存储的抽象接口：

```python
# vllm/v1/kv_offload/tiering/base.py
class SecondaryTierManager(ABC):
    @abstractmethod
    def lookup(self, key, req_context) -> LookupResult: ...

    @abstractmethod
    def submit_store(self, job_metadata): ...     # 主层级 → 二级存储

    @abstractmethod
    def submit_load(self, job_metadata): ...      # 二级存储 → 主层级

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]: ...  # 获取完成的任务
```

---

### 六、Sleep Mode：模型权重的 Offload

##### 6.1 为什么需要 Sleep Mode

在 RLHF（人类反馈强化学习）场景中，训练循环需要频繁在**奖励模型**和**策略模型**之间切换。两个模型都很庞大，GPU 显存装不下两个。

Sleep Mode 的思路：**当一个模型不用时，把它的权重 offload 到 CPU，腾出 GPU 空间给另一个模型**。

```
┌─────────────────────────────────────────────────────┐
│              RLHF 训练循环                            │
│                                                     │
│  Step 1: 策略模型推理                                │
│    策略模型权重在 GPU，奖励模型权重在 CPU              │
│                                                     │
│  Step 2: 切换！                                      │
│    策略模型 sleep → 权重 offload 到 CPU               │
│    奖励模型 wake up → 权重 load 回 GPU                │
│                                                     │
│  Step 3: 奖励模型推理                                │
│    奖励模型权重在 GPU，策略模型权重在 CPU              │
│                                                     │
│  Step 4: 切换！                                      │
│    奖励模型 sleep → 权重 offload 到 CPU               │
│    策略模型 wake up → 权重 load 回 GPU                │
│                                                     │
│  ... 循环往复 ...                                    │
└─────────────────────────────────────────────────────┘
```

##### 6.2 两个 Sleep 级别

```python
# vllm/v1/worker/gpu_worker.py
def sleep(self, level: int = 1) -> None:
    if level == 1:
        # 级别1：Offload 模型权重到 CPU，丢弃 KV Cache
        # 恢复时从 CPU 重新加载权重，KV Cache 需重新计算
        allocator.sleep(offload_tags=("weights",))

    elif level == 2:
        # 级别2：丢弃一切（权重 + KV Cache）
        # 先保存 buffer 到 CPU（用于 RLHF 权重更新场景）
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in model.named_buffers()
        }
        allocator.sleep(offload_tags=())  # 空 tuple = 全部丢弃
```

| 级别 | 权重处理 | KV Cache 处理 | Buffer 处理 | 恢复方式 |
|:---|:---|:---|:---|:---|
| Level 1 | Offload 到 CPU | 丢弃 | 不处理 | 从 CPU 加载权重，KV Cache 重新计算 |
| Level 2 | 全部丢弃 | 全部丢弃 | 保存到 CPU | 从 CPU 恢复 buffer，权重和 KV Cache 重新加载 |

```python
def wake_up(self, tags=None):
    allocator.wake_up(tags)
    # Level 2 恢复：从 CPU 拷回保存的 buffer
    if len(self._sleep_saved_buffers):
        for name, buffer in model.named_buffers():
            buffer.data.copy_(self._sleep_saved_buffers[name])
    # KV Cache 恢复
    if tags is None or "kv_cache" in tags:
        self.model_runner.post_kv_cache_wake_up()
```

##### 6.3 平台支持

```python
# vllm/platforms/interface.py
def is_sleep_mode_available(self) -> bool:
    """仅 CUDA、ROCm、XPU 支持睡眠模式"""
    return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.XPU)
```

> 笔者注：Sleep Mode 还需要启用自定义内存分配器（`enable_cumem_allocator=True`），通过 `--enable-sleep-mode` CLI 参数开启。

---

### 七、搬运的底层实现：Triton 内核与异步传输

##### 7.1 Swap Blocks 的 Triton 实现

GPU 和 CPU 之间的数据搬运使用自定义 Triton 内核：

```python
# vllm/v1/kv_offload/cpu/swap_blocks_triton.py
# 针对 H100 (PCIe Gen5) 调优的常量
NUM_SMS = 12           # 使用的 SM 数量
THRESHOLD_BYTES = 28 * 1024  # 28 KB 以下用 Triton，以上用 C++ fallback
MIN_N = 16             # 最小传输块数

def swap_blocks_batch(src_addrs, dst_addrs, sizes, is_src_access_order_any=False, *, bytes_per_chunk):
    """Triton 实现的小批量 CPU↔GPU 传输
    大批量传输回退到 C++ ops.swap_blocks_batch"""
```

##### 7.2 传输流水线

```
时间轴 →

GPU 计算:  ──[计算 Step N]──[计算 Step N+1]──[计算 Step N+2]──
                                    │
Store Stream:              ──[等计算完成]──[搬 Block A→CPU]──[搬 Block B→CPU]──
                                    │                      │
Load Stream:                                    ──[等 Store]──[搬 Block C→GPU]──
```

关键特性：
- **异步执行**：Store/Load 在独立 CUDA Stream 上运行，与计算并行
- **有序保证**：同方向的传输按提交顺序串行执行
- **依赖同步**：Store 方向等待当前 Step 计算完成（`wait_stream(compute_stream)`）

---

### 八、配置与使用

##### 8.1 KV Cache Offload 配置

通过 `--kv-transfer-config` 参数配置，传入 JSON 字符串：

```bash
# 基础 CPU Offload
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "CPUOffloadingSpec",
      "cpu_bytes_to_use": 10737418240,
      "block_size": 16,
      "eviction_policy": "lru"
    }
  }'
```

```bash
# 多级分层 Offload（CPU + 文件系统）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 10737418240,
      "block_size": 16,
      "eviction_policy": "lru",
      "secondary_tiers": [
        {
          "type": "fs",
          "root_dir": "/mnt/kv_cache",
          "n_read_threads": 32,
          "n_write_threads": 16
        }
      ]
    }
  }'
```

##### 8.2 配置参数详解

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `spec_name` | `CPUOffloadingSpec` | 单级用 `CPUOffloadingSpec`，多级用 `TieringOffloadingSpec` |
| `cpu_bytes_to_use` | 必填 | CPU 侧用于 KV Cache 的总字节数 |
| `block_size` | GPU block_size | Offload 块大小（必须是 GPU block_size 的倍数） |
| `eviction_policy` | `lru` | 淘汰策略：`lru` 或 `arc` |
| `store_threshold` | `0` | 块至少被查找几次才允许 offload（防止抖动） |
| `max_tracker_size` | `64000` | 查找追踪器最大条目数 |
| `secondary_tiers` | `[]` | 二级存储配置列表（仅 TieringOffloadingSpec） |
| `offload_prompt_only` | `true` | 仅 offload prefill 块，跳过 decode 块 |

##### 8.3 Sleep Mode 配置

```bash
# 启用 Sleep Mode
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-sleep-mode
```

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `enable_sleep_mode` | `False` | 是否启用睡眠模式 |
| `enable_cumem_allocator` | `False` | 是否启用自定义 CUDA 内存分配器（Sleep Mode 必需） |

##### 8.4 模型权重 Offload 配置

```python
# vllm/config/offload.py
class OffloadConfig:
    offload_backend: OffloadBackend = "auto"  # auto / uva / prefetch

    class UVAOffloadConfig:
        cpu_offload_gb: float = 0  # 每张 GPU offload 到 CPU 的 GB 数

    class PrefetchOffloadConfig:
        offload_group_size: int = 0       # 每 N 层为一组
        offload_num_in_group: int = 1     # 每组中 offload 的层数
        offload_prefetch_step: int = 1    # 预取提前步数
```

| 后端 | 原理 | 适用场景 |
|:---|:---|:---|
| `uva` | 统一虚拟寻址，CPU 内存映射到 GPU 地址空间 | 小规模 offload，延迟敏感 |
| `prefetch` | 分组 offload + 预取，提前加载下一组权重 | 大规模 offload，吞吐优先 |

---

### 九、比喻：便利店的仓库系统

把 Offload Memory 想象成便利店的仓库系统：

**GPU 显存 = 便利店货架**——空间有限，但取货最快。只有"热销品"（正在 decode 的 KV Cache）放在这里。

**CPU 内存 = 隔壁仓库**——空间大得多，取货稍慢。暂时不卖的"冷门品"（已完成 prefill 的 KV Cache）存在这里，需要时搬回货架。

**文件系统/对象存储 = 远郊冷库**——空间几乎无限，但取货最慢。长期不用的"滞销品"存在这里，只有仓库也没货时才去取。

**Store（搬出）** = 把货架上的商品搬到仓库。搬之前要确认这个商品"确实不急着卖"（`store_threshold`），避免刚搬走又得搬回来。

**Load（搬回）** = 从仓库把商品搬回货架。搬之前要确认货架有空间，没有的话先淘汰最久没卖的商品（LRU/ARC）。

**Sleep Mode = 闭店装修**——整个店面的商品全部搬走（模型权重 offload），腾出空间给另一家店（另一个模型）营业。装修完了再搬回来重新开张。

```
┌─────────────────────────────────────────────────────┐
│                  便利店仓库系统                       │
│                                                     │
│  ┌──────────┐   Store   ┌──────────┐   级联   ┌──────────┐
│  │  货架    │ ────────→ │  仓库    │ ────────→ │  冷库    │
│  │  (GPU)   │ ←──────── │  (CPU)   │ ←──────── │  (FS/S3) │
│  └──────────┘   Load    └──────────┘   提升    └──────────┘
│    最快最贵             较快较贵              最慢最便宜     │
│                                                     │
│  淘汰策略: LRU (最久没卖 → 先下架)                    │
│           ARC (自适应: 新面孔 vs 老主顾)               │
└─────────────────────────────────────────────────────┘
```

---

### 十、性能考量与最佳实践

##### 10.1 Offload 的代价

| 代价 | 说明 |
|:---|:---|
| 传输延迟 | PCIe 带宽约 64 GB/s（Gen5 x16），搬 1 个 Block（5 MB）约 80 us |
| 额外 CPU 内存 | 需要预留足够的 CPU 内存存放 offload 的 KV Cache |
| 淘汰开销 | CPU 侧满了需要淘汰，LRU/ARC 淘汰有计算开销 |
| 缓存未命中 | 如果 Load 时块已被淘汰，需要重新计算 KV Cache |

##### 10.2 何时适合 Offload

| 适合 | 不适合 |
|:---|:---|
| 长上下文请求多，KV Cache 大 | 短上下文请求为主，GPU 显存够用 |
| 请求有明显的冷热区分 | 所有请求都在同时 decode |
| 需要支持更多并发，但不想加卡 | 对延迟极度敏感，不能接受任何 Load 延迟 |
| RLHF 场景需要多模型切换 | 单模型推理，不需要 Sleep Mode |

##### 10.3 调参建议

| 参数 | 建议 | 原因 |
|:---|:---|:---|
| `cpu_bytes_to_use` | 设为 GPU 显存的 2-4 倍 | CPU 内存便宜，多分一些减少淘汰 |
| `eviction_policy` | 访问模式稳定用 `lru`，波动大用 `arc` | ARC 自适应能力强但开销略大 |
| `store_threshold` | 设为 1-2 | 避免刚 offload 就需要 load 回来 |
| `offload_prompt_only` | 默认 `true` 即可 | decode 块正在使用，offload 意义不大 |

---

### 总结

| 问题 | 答案 |
|:---|:---|
| Offload 是什么？ | 把 GPU 上暂时不用的 KV Cache（或模型权重）搬到 CPU 内存 |
| 为什么需要？ | GPU 显存不够用，CPU 内存便宜且容量大 |
| 搬什么？ | KV Cache 块（场景1/2）或模型权重（场景3） |
| 搬到哪？ | CPU 内存（单级）、CPU+文件系统/对象存储（多级） |
| 怎么搬？ | 异步 cudaMemcpy，独立 CUDA Stream，与计算并行 |
| 仓库满了怎么办？ | LRU 或 ARC 淘汰策略，淘汰最久没用/最不可能再用的块 |
| Sleep Mode 是什么？ | 把整个模型权重 offload 到 CPU，腾出 GPU 给另一个模型 |
| 代价是什么？ | 传输延迟、额外 CPU 内存、可能的缓存未命中 |

### 延伸阅读

- vLLM 源码：[vllm/v1/kv_offload/base.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_offload/base.py) — OffloadingManager 抽象接口
- vLLM 源码：[vllm/v1/kv_offload/cpu/manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_offload/cpu/manager.py) — CPUOffloadingManager 单级实现
- vLLM 源码：[vllm/v1/kv_offload/tiering/manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_offload/tiering/manager.py) — TieringOffloadingManager 多级实现
- vLLM 源码：[vllm/v1/worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_worker.py) — Sleep Mode 实现
- vLLM 文档：[docs/features/kv_offloading_usage.md](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md) — KV Offload 使用指南
- vLLM 文档：[docs/features/sleep_mode.md](https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md) — Sleep Mode 使用指南

---

*本文属于 [vLLM 技术博客系列]，欢迎持续关注。*
