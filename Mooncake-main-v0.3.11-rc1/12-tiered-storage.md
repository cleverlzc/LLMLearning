# Mooncake SSD分层存储：KV Cache 的"热货架与冷库"经济学

> **系列**: Mooncake 技术博客系列 | **类型**: 核心技术详解篇
>
> KV Cache 是 LLM 推理中最珍贵的中间产物——重新计算一次要消耗整个 Prefill 的算力。DRAM 放不下全部，直接丢弃又太浪费。Mooncake 的解法：给 KV Cache 建一套"热货架 + 冷库"，DRAM 装不下的自动搬进 SSD，需要时再搬回来，用延迟换容量，用容量换吞吐。

在前面的 Mooncake Store 文章中，提到了 SSD 卸载，本文将详细介绍一下 Mooncake 系统中，如何实现 SSD 分层存储的。

分层分级存储，在计算机体系结构里，已经是非常经典的一个思想了，其应用也是非常广泛和成熟。总而言之，言而总之，靠近CPU/GPU端，存储器价格太贵容量太小，想要低成本存储更大容量的数据，就自然而然扩展到分层分级存储。

学过操作系统原理的同学都知道，CPU会分L1、L2、L3分级缓存，CPU的分级缓存就像是一个为了追赶处理器极快速度而设计的“接力赛”系统。因为内存（RAM）的速度远跟不上CPU的处理速度，所以CPU内部集成了不同大小、不同速度的缓存，用来存放最可能被用到的数据。这种分层设计遵循了“局部性原理”，即程序倾向于重复访问最近使用过的数据或附近的内存地址。通过这种设计，CPU在大多数情况下都能以接近L1的速度获取数据，从而保持高效运行。这里做简单回顾。

CPU缓存体系通常分为三个层级，处理速度从快到慢、容量从小到大排列如下：

- 第一级缓存（L1 Cache）
这是离CPU核心最近、速度最快的缓存，通常只有几KB到几十KB。它被严格分为指令缓存和数据缓存两部分。指令缓存负责存放CPU即将执行的代码，数据缓存负责存放运算所需的数据。因为物理距离极短，访问延迟极低，通常在1个时钟周期左右就能完成读写。

- 第二级缓存（L2 Cache）
L2缓存比L1大，通常在几百KB到几MB之间，速度稍慢于L1，但依然很快，延迟可能在3到10个时钟周期左右。在现代架构中，L2通常是每个CPU核心私有的，专门服务于该核心。它的作用是作为L1的后备，当L1找不到数据时，就来这里找。

- 第三级缓存（L3 Cache）
L3缓存容量最大，从几MB到几十MB甚至上百MB不等，速度比L2慢，延迟较高。它通常是所有CPU核心共享的。如果数据在L1和L2中都找不到，CPU就会去L3中查找。L3的存在大大减少了CPU不得不去访问慢速主内存的情况，从而提升整体性能。

总结一下这个层级关系：
L1最快、最小、最私有，负责最频繁的数据交换。
L2中等，是每个核心的第二道防线。
L3最大、最慢、但共享，负责协调多个核心之间的数据需求，并减少对主内存的访问。

对于AI训练、推理领域，其思想和原理是一模一样的，从 HBM（GPU显存） 到 DRAM（物理机内存） 到 NAND 闪存（SSD盘），甚至到 SATA 大容量机械盘。 其中，HBM（L0）、DRAM（L1）、本地SSD（L2）、远程SSD（L3，当前业界一般是基于SSD盘的分布式文件系统），为当前最为经典的“三级缓存”方案（当然如果觉得SSD还是太贵，并且公司业务能够接受慢一点速度，整个系统架构设计上可以进一步扩展低成本存储，著名的KVCache开源系统LMCache已支持到成本相对最为低廉的对象存储，代价就是IO读写会慢一些，训练和推理速度不要预期那么高；通俗点讲就是，有钱就上高端的，各方面体验均是一流，没钱就用时间凑，忍受慢以及故障了排错周期长）。

L编号有的地方从0开始，有的地方从1开始，有的地方从DRAM开始编号，完全是自己系统定义，方便团队工程师高效协作、沟通。

---

### 引言

想象一家进口超市。热销的鲜奶面包放在门口的货架（DRAM）上，顾客一拿就走；季节限定的进口巧克力搬进后院的冷库（SSD），有顾客点名时再搬出来。你绝不会因为货架满了就扔掉巧克力——它们是高价商品，扔了就是亏钱。正确的做法是把暂时卖得慢的移到冷库，腾出货架放当季爆款；等情人节快到了，再把巧克力搬回货架。

KV Cache 就是这些"高价商品"。在 LLM 推理中，一条 KV Cache 动辄数十 MB，对应一次完整的 Prefill 计算——丢弃它意味着下次请求要重新跑一遍 Prefill，GPU 算力白白浪费。而 GPU 集群的 DRAM 总是稀缺的，不可能把所有 KV Cache 都留在内存里。

Mooncake Store 的分层存储让数据在 DRAM 和 SSD 之间自动流转：DRAM 装不下的卸载到 SSD，需要时再提升回来，用纳秒级延迟换微秒级延迟，用有限的 DRAM 撑起数十倍的逻辑容量。

---

### 三层存储架构：MEMORY → LOCAL_DISK → 分布式存储

Mooncake Store 实现了一个三层存储体系，由 Master 元数据平面协调数据在各层之间流动：

```
┌─────────────────────────────────────────────────────────────┐
│                      Master Service                         │
│  (元数据中枢: 对象/副本映射、驱逐策略、卸载/提升决策)           │
└────────┬──────────────────────────┬─────────────────────────┘
         │ 元数据 RPC               │ 元数据 RPC
         ▼                          ▼
┌────────────────┐          ┌────────────────┐
│   Client A     │          │   Client B     │
│ ┌────────────┐ │          │ ┌────────────┐ │
│ │ DRAM 段    │ │ RDMA 直传 │ │ DRAM 段    │ │
│ │ (MEMORY)   │ │◄────────►│ │ (MEMORY)   │ │
│ └────────────┘ │          │ └────────────┘ │
│ ┌────────────┐ │          │ ┌────────────┐ │
│ │ 本地 SSD   │ │          │ │ 本地 SSD   │ │
│ │(LOCAL_DISK)│ │          │ │(LOCAL_DISK)│ │
│ └────────────┘ │          │ └────────────┘ │
│ ┌────────────┐ │          │                │
│ │FileStorage │ │          │                │
│ │ (SSD管家)  │ │          │                │
│ └────────────┘ │          │                │
└────────────────┘          └────────────────┘
         │                          │
         └──── 分布式文件系统 ───────┘
              (3FS 等 DISK 层)
```

##### 四种副本类型

数据在每个对象上可以同时拥有多个不同类型的副本，形成从热到冷的分层：

| 副本类型 | 存储介质 | 延迟 | 容量 | 典型场景 |
|---------|---------|------|------|---------|
| `MEMORY` | DRAM | ~100ns | 有限 | 热数据，频繁访问的 KV Cache |
| `LOCAL_DISK` | 本地 SSD | ~10μs | 大 | 温数据，DRAM 驱逐后的降级目标 |
| `NOF_SSD` | NVMe-oF 远程 SSD | ~20μs | 大 | 共享 SSD 池，跨节点访问 |
| `DISK` | 分布式文件系统 | ~ms 级 | 海量 | 冷数据，3FS 等远程存储 |

```cpp
// mooncake-store/include/allocator.h
enum class ReplicaType {
    MEMORY,      // DRAM 内存副本
    DISK,        // 远程磁盘副本 (Master 侧文件系统)
    LOCAL_DISK,  // 本地磁盘副本 (Client 侧 SSD)
    NOF_SSD,     // Nvme-oF SSD 副本 (远程 NVMe over Fabrics)
    ALL,         // 所有内存和 NoF 副本
};
```

每个对象的副本数据使用 `std::variant` 实现多态存储：

```cpp
// mooncake-store/include/replica.h
std::variant<MemoryReplicaData, NoFReplicaData, DiskReplicaData, LocalDiskReplicaData> data_;
```

> 笔者注：四种副本类型不是互斥的——同一个对象可以同时拥有 MEMORY 副本和 LOCAL_DISK 副本。这就像一件商品同时在货架和冷库各放一份，货架上的卖完了直接从冷库补，不用等物流。

---

### 数据如何在层间流动？

分层存储的核心是三个操作：**卸载（Offload）、提升（Promotion）、驱逐（Eviction）**。它们构成了数据的"降级→升级→淘汰"生命周期。

```
                 ┌──────────┐
                 │  DRAM    │ ←── Promotion (提升: 冷→热)
                 │ (MEMORY) │ ──→ Offload (卸载: 热→冷)
                 └────┬─────┘
                      │
                      ▼
              ┌──────────────┐
              │   本地 SSD   │ ←── BatchLoad (按需读取)
              │ (LOCAL_DISK) │ ──→ SSD Eviction (磁盘驱逐)
              └──────┬───────┘
                     │
                     ▼
           ┌──────────────────┐
           │ 分布式文件系统     │
           │ (DISK / 3FS)     │
           └──────────────────┘
```

##### Offload：DRAM → SSD（降级）

**谁决定降级？** Master。Master 根据内存水位线，选择需要降级的对象，放入 `LocalDiskSegment::offloading_objects` 队列。

**谁执行降级？** Client 侧的 `FileStorage`。它通过心跳从 Master 拉取降级任务，然后将数据从 DRAM 写入本地 SSD。

完整流程：

```
1. Master 决策: DRAM 使用率超过高水位 → 选择降级对象
2. Client 心跳: FileStorage::Heartbeat() → OffloadObjectHeartbeat() 拉取任务
3. 数据搬迁: FileStorage::OffloadObjects() 执行
   ├── BatchQuerySegmentSlices() 从 DRAM 段获取数据切片
   ├── D2H 暂存: GPU 数据通过 PinnedBufferPool 做 D2H 拷贝
   ├── 批量写入: storage_backend_->BatchOffload() 写入 SSD
   └── 通知 Master: client_->NotifyOffloadSuccess() 注册 LOCAL_DISK 副本
```

关键代码——D2H 暂存（`file_storage.cpp`）：

```cpp
// D2H staging: replace device slices with host memory slices
// so that storage_backend always receives host pointers.
for (auto& [obj_key, slices] : batch_object) {
    std::vector<Slice> host_slices;
    for (const auto& slice : slices) {
        auto* device = runtime_accelerator.FindDeviceForPointer(slice.ptr, &info);
        if (device) {
            // GPU 数据: 通过 PinnedBufferPool 做 D2H 拷贝
            auto buf = pinned_buffer_pool_->Acquire(slice.size);
            device->Copy(buf.data, slice.ptr, slice.size,
                         device::CopyDirection::kDeviceToHost);
            host_slices.emplace_back(Slice{buf.data, slice.size});
            staging_bufs.push_back(std::move(buf));
        } else {
            // DRAM 数据: 直接使用
            host_slices.push_back(slice);
        }
    }
    host_batch_object[obj_key] = std::move(host_slices);
}
```

> 笔者注：PinnedBufferPool 是 GPU D2H 数据搬迁的关键。Pinned host memory（页锁定内存）比普通 pageable memory 提供 10x-100x 的 D2H 带宽提升。池化复用（最大 32 个缓冲区）避免了反复分配/释放的开销。如果 pinned 分配失败，系统会回退到 `new char[]`，性能下降但不会崩溃。

##### Promotion：SSD → DRAM（升级）

**谁触发升级？** 两种方式：
1. **命中提升**（`promotion_on_hit`）：Get 请求命中 SSD-only 的 key 时触发
2. **频率门控**：Master 用 Count-Min Sketch 追踪访问频率，只有"足够热"的数据才值得升级

Master 侧的频率门控逻辑：

```
TryPushPromotionQueue(object_id):
    1. 频率门控: Count-Min Sketch 计数 >= promotion_admission_threshold?
       ├── 否 → 跳过（冷数据不值得提升）
       └── 是 → 继续
    2. 水位门控: DRAM 使用率 < eviction_high_watermark_ratio?
       ├── 否 → 跳过（DRAM 没空间，提升也没地方放）
       └── 是 → 继续
    3. 加入提升队列: local_disk_segment->promotion_objects[key] = task
```

Client 侧执行提升（`file_storage.cpp`）：

```
ProcessPromotionTasks():
    1. 心跳拉取: client_->PromotionObjectHeartbeat() 获取提升任务
    2. 分配 DRAM: client_->PromotionAllocStart() 在 DRAM 段分配 PROCESSING 副本
    3. 读取 SSD: AllocateBatch() + BatchLoad() 从本地 SSD 读数据
    4. 写入 DRAM: client_->PromotionWrite() 通过 Transfer Engine 写入 DRAM 副本
    5. 提交: client_->NotifyPromotionSuccess() 标记副本为 COMPLETE
```

> 笔者注：提升是"尽力而为"的——任何一步失败都不会影响系统运行。DRAM 没空间？跳过。SSD 读取失败？跳过。Transfer Engine 写入失败？跳过。Master 的 reaper 机制会在 TTL 到期后清理残留状态。这种"宁可不做，不可做错"的设计，确保了提升永远不会阻塞正常的卸载和读取流程。

##### Eviction：空间回收

| 层 | 驱逐策略 | 实现 |
|----|---------|------|
| DRAM | LRU / FIFO | `EvictionStrategy` 抽象类，Master 侧执行 |
| SSD | FIFO / LRU / None | `BucketEvictionPolicy`，Client 侧执行 |

DRAM 驱逐策略的核心接口：

```cpp
// mooncake-store/include/eviction_strategy.h
class EvictionStrategy {
    virtual ErrorCode AddKey(const std::string& key) = 0;
    virtual ErrorCode UpdateKey(const std::string& key) = 0;
    virtual std::string EvictKey(void) = 0;  // 返回被选中的驱逐 key
};

class LRUEvictionStrategy : public EvictionStrategy {
    // AddKey/UpdateKey → 移到链表头部
    // EvictKey → 返回链表尾部（最久未访问）
};

class FIFOEvictionStrategy : public EvictionStrategy {
    // AddKey → 插入链表头部
    // UpdateKey → 不移动（FIFO 不关心访问顺序）
    // EvictKey → 返回链表尾部（最早进入）
};
```

---

### SSD 存储后端：四种实现，一个接口

SSD 存储通过 `StorageBackendInterface` 抽象，提供四种后端实现：

| 后端类型 | 实现类 | 存储方式 | 驱逐策略 | 适用场景 |
|----------|--------|----------|---------|---------|
| `kFilePerKey` | StorageBackendAdaptor | 每个 key 一个文件 | FIFO | 简单调试 |
| `kBucket` | BucketStorageBackend | 多 key 合并为桶文件 | FIFO/LRU/None | **生产推荐** |
| `kOffsetAllocator` | OffsetAllocator-based | 偏移量分配 | N/A | 高性能场景 |
| `kDistributed` | DistributedStorageBackend | 远程分布式文件系统 | 由 DFS 管理 | 3FS 等 |

##### BucketStorageBackend：生产推荐方案

为什么不用"一个 key 一个文件"？因为当 key 数量达到百万级，文件系统的 inode 和目录项会变成瓶颈。Bucket 后端把多个 key 的数据合并写入同一个桶文件，大幅减少文件数量。

```
桶文件布局:
┌─────────────────────────────────────────────────────┐
│  data 区域                                           │
│  [key1_data][key2_data][key3_data]...[keyN_data]     │
├─────────────────────────────────────────────────────┤
│  metadata 区域                                       │
│  meta_size | data_size | keys[] | metadatas[]        │
│  metadatas: [{offset, key_size, data_size}, ...]     │
└─────────────────────────────────────────────────────┘
```

桶的配置约束：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `bucket_size_limit` | 256 MB | 单桶最大大小 |
| `bucket_keys_limit` | 500 | 单桶最大 key 数 |
| `eviction_policy` | NONE | FIFO / LRU / None |
| `max_total_size` | 0 (无限) | 超过则触发驱逐 |

**安全删除**是桶设计中最精巧的部分：当驱逐一个桶时，可能还有正在进行的读取操作。`BucketReadGuard` 用原子计数器跟踪 in-flight reads，确保读取完成后再删桶文件：

```cpp
// mooncake-store/include/storage_backend.h
class BucketReadGuard {
    explicit BucketReadGuard(std::shared_ptr<BucketMetadata> bucket)
        : bucket_(std::move(bucket)) {
        if (bucket_) {
            bucket_->inflight_reads_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ~BucketReadGuard() {
        if (bucket_) {
            bucket_->inflight_reads_.fetch_sub(1, std::memory_order_release);
        }
    }
};
```

> 笔者注：`inflight_reads_` 用 `memory_order_relaxed` 递增（近似计数即可），用 `memory_order_release` 递减（确保之前的读操作对驱逐线程可见）。这种非对称内存序设计，在正确性和性能之间取得了精妙平衡。

这里单独提一下Guard，这一类实现其实可以总结为“Guard模式”，在一些生产系统中，不得不考虑各种极端情况。笔者曾经在存储系统中应用过Guard模式，当时还不知道其他生产系统中也会广泛使用，只是为了应对故障场景的方案，就自然想到了加一道Guard流程保护。

当时的场景是，涉及两个微服务的交互，为了确保数据安全（数据一致性）中间某个操作必须将业务进程停掉，停掉业务进程会短暂停服（客户端自带重试，控制在短时间只会卡不会报错），超过一定时间微服务没有下发拉起进程，比如网络故障，连不通了，则通过自身系统的Guard流程将业务进程拉起来，继续提供服务。

##### 文件 I/O：POSIX vs io_uring

`StorageFile` 抽象提供两种实现：

| 实现 | 特点 | 适用场景 |
|------|------|---------|
| **PosixFile** | `pread/pwrite` + scatter/gather I/O | 通用，兼容性好 |
| **UringFile** | Linux io_uring 异步 I/O + O_DIRECT | 高吞吐，低延迟 |

UringFile 的关键设计：
- **进程级共享 ring**：`SharedUringRing` 单例，避免每个文件的 mmap/munmap 开销
- **固定缓冲区注册**：`register_global_buffer()` 预注册 client buffer，避免每次 I/O 的内核态拷贝
- **批量读取**：`batch_read()` 一次提交最多 32 个独立读请求

> 笔者注：io_uring 是 Linux 5.1+ 引入的高性能异步 I/O 框架。传统 `pread/pwrite` 每次系统调用都要陷入内核态，而 io_uring 通过共享环形缓冲区，允许用户态批量提交 I/O 请求，内核批量完成，大幅减少系统调用次数。在 NVMe SSD 上，io_uring + O_DIRECT 可以逼近硬件带宽极限。

---

### 内存分配器：两种哲学

Mooncake Store 提供两种内存分配器，代表两种不同的设计哲学：

| 分配器 | 基础 | 特点 | 精确空闲统计 |
|--------|------|------|-------------|
| **CachelibBufferAllocator** | Meta CacheLib Slab | 4MB Slab，固定大小分配类 | 否（返回 unknown） |
| **OffsetBufferAllocator** | 二分伙伴系统 | 256 个大小类，线程安全 | 是（`getLargestFreeRegion()`） |

##### CacheLib Slab 分配器

来自 Meta 的 CacheLib，层次结构：

```
MemoryAllocator
  ├── MemoryPoolManager
  │     └── MemoryPool [最多 256 个池]
  │           └── AllocationClass [最多 128 个分配类]
  │                 └── Slab [4MB 固定大小]
  └── SlabAllocator
        └── 将连续内存切分为 4MB Slab
```

工作方式：将内存切分为 4MB 的 Slab，每个 Slab 内按固定大小分配。比如一个 Slab 专门分配 64 字节的块，另一个专门分配 128 字节的块。优点是分配/释放极快（无碎片化搜索），缺点是内部碎片（分配 65 字节要用 128 字节的块）。

##### OffsetAllocator 二分伙伴系统

源自 Sebastian Aaltonen 的 OffsetAllocator，核心设计：

```
32 个顶层 bin × 8 个叶 bin = 256 个大小类
Node 双向链表管理空闲和已用节点
乘数机制: uint32 偏移量 → 真实地址空间映射
```

关键特性：
- **线程安全**：所有操作通过 Mutex 保护
- **序列化支持**：完整的 serialize/deserialize，用于 fork 后恢复
- **精确统计**：能报告最大空闲区域大小

> 笔者注：两种分配器的选择取决于场景。CacheLib Slab 适合分配大小相对固定的场景（如 KV Cache block），分配速度极快但无法精确报告空闲空间。OffsetAllocator 适合分配大小多变的场景，能精确统计但分配速度稍慢。Mooncake 默认使用 OffsetAllocator（`memory_allocator: "offset"`）。

##### MmapArena：竞技场分配器

用于 SGLang HiCache 等追加写入场景：

```
mmap 分配大块内存 → CAS 无锁游标推进 → O(1) 分配 → 不支持释放
```

关键设计：
- **大页优先**：`MAP_HUGETLB` 尝试 2MB 大页，失败回退到普通页
- **无锁分配**：`alloc_cursor_` 用 CAS 原子推进，无锁无等待
- **Fork 安全**：`MADV_DONTFORK` 防止子进程继承大页映射
- **只进不出**：Arena 分配器不支持释放，适用于缓存场景的追加写入

> 笔者注：MmapArena 的"只进不出"设计看似浪费，但在缓存场景下非常合理。缓存数据要么被覆盖，要么整个区域重置。就像一本只往前写的笔记本，写满了翻页，不需要擦除。

---

### 心跳机制：分层存储的"脉搏"

`FileStorage` 的心跳线程以可配置间隔（默认 10 秒）驱动所有分层操作：

```
Heartbeat() 每个周期:
    ├── 1. 上报访问频率 → 向 Master 报告本地 SSD 对象的访问统计
    ├── 2. 拉取卸载任务 → OffloadObjectHeartbeat() 获取 DRAM→SSD 降级列表
    ├── 3. 执行卸载 → OffloadObjects() 异步写入 SSD
    ├── 4. 拉取提升任务 → PromotionObjectHeartbeat() 获取 SSD→DRAM 升级列表
    └── 5. 执行提升 → ProcessPromotionTasks() 从 SSD 读回并写入 DRAM
```

心跳还处理 Master 重启恢复：如果心跳返回 `SEGMENT_NOT_FOUND`，说明 Master 重启后丢失了 LOCAL_DISK 段信息。FileStorage 会自动重新挂载段，并触发异步 `ScanMeta` 重新注册所有 SSD 对象元数据。

> 笔者注：心跳间隔是分层存储性能的关键调优参数。间隔太短，Master 和 Client 之间的元数据同步更及时，但 CPU 和网络开销更大；间隔太长，降级和提升的延迟更高，可能导致 DRAM 溢出或 SSD 命中率下降。默认 10 秒是一个平衡点，生产环境可根据负载特征调整。

---

### 分配策略：数据该放在哪个段？

当 Master 需要为一个新对象分配空间时，选择哪个 DRAM 段？`AllocationStrategy` 定义了五种策略：

| 策略 | 算法 | 适用场景 |
|------|------|---------|
| RandomAllocationStrategy | 随机选择段 | 默认、简单场景 |
| FreeRatioFirstAllocationStrategy | 采样 6N 候选，按空闲比排序选 Top-N | DRAM 负载均衡 |
| SsdFreeRatioFirstAllocationStrategy | 按 SSD 空闲比排序 | SSD 段分配 |
| CxlAllocationStrategy | 指定 CXL 段分配 | CXL 共享内存 |

FreeRatioFirst 的核心算法：

```
1. 随机采样 min(6 × replica_num, total) 个候选段
2. 计算每段空闲比: free_bytes / capacity
3. 按空闲比降序排列，尝试从 Top-N 分配
4. 失败则回退到随机分配
```

> 笔者注：为什么是 6N 而不是全量扫描？因为集群可能有数百个段，全量排序的开销不可接受。6N 采样是"足够好的近似"——来自 Power of Two Choices 的理论保证：随机选 2 个就比随机选 1 个好很多，选 6N 几乎总是能找到足够空闲的段。

---

### 配置速查表

##### Master 侧分层存储配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_offload` | - | 是否启用 SSD 卸载 |
| `offload_on_evict` | false | 驱逐时是否降级到 SSD |
| `offload_force_evict` | false | 强制驱逐源副本 |
| `offloading_queue_limit` | 50000 | 降级队列上限 |
| `offload_cap_ratio` | 0.5 | 降级容量比例 |
| `promotion_on_hit` | false | 是否启用命中提升 |
| `promotion_admission_threshold` | 2 | 提升频率阈值（访问几次才提升） |
| `promotion_queue_limit` | 50000 | 提升队列上限 |
| `promotion_max_per_heartbeat` | 1 | 每次心跳最大提升任务数 |
| `eviction_high_watermark_ratio` | 0.95 | DRAM 高水位（触发驱逐） |
| `eviction_ratio` | 0.05 | 每次驱逐比例 |
| `enable_disk_eviction` | true | 是否启用磁盘驱逐 |
| `quota_bytes` | 0 | SSD 存储配额 |

##### Client 侧 SSD 配置（环境变量）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR` | 存储后端类型 | bucket |
| `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` | SSD 存储目录 | /data/file_storage |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | 本地缓冲区大小 | 1.28 GB |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | SSD 总容量限制 | 2 TB |
| `MOONCAKE_OFFLOAD_USE_URING` | 是否使用 io_uring | false |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | 桶驱逐策略 | none |

---

### 设计哲学：分层存储的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **数据不丢失** | 驱逐前先降级，降级后才驱逐 | `offload_on_evict` 配置项 |
| **提升需门槛** | 冷数据不会因为一次偶然访问就被提升 | Count-Min Sketch 频率门控 |
| **尽力而为** | 提升失败不影响正常流程 | Promotion 失败自动跳过 |

这三条原则回答了分层存储最核心的三个问题：

1. **数据会不会丢？** —— 不会。`offload_on_evict=true` 时，DRAM 驱逐前先把数据搬到 SSD，保证数据至少存在一个副本。
2. **冷数据会不会被反复搬运？** —— 不会。频率门控确保只有"足够热"的数据才值得提升，避免冷数据在 DRAM 和 SSD 之间反复横跳。
3. **搬运失败会不会拖垮系统？** —— 不会。提升是尽力而为，任何一步失败都自动跳过，不会阻塞正常的数据路径。

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| 三层架构 | MEMORY(DRAM) → LOCAL_DISK(SSD) → DISK(分布式)，热到冷自动降级 |
| Offload | DRAM → SSD 降级，Master 决策、Client 执行、心跳驱动 |
| Promotion | SSD → DRAM 升级，频率门控 + 水位门控，尽力而为 |
| Bucket 后端 | 多 key 合并桶文件，减少 inode 压力，生产推荐 |
| 双分配器 | CacheLib Slab（极速分配）vs OffsetAllocator（精确统计） |
| 心跳机制 | 10 秒周期驱动卸载/提升/元数据同步，Master 重启自动恢复 |

**建议**: 生产部署时务必设置 `offload_on_evict=true` 和 `promotion_on_hit=true`，让数据在 DRAM 和 SSD 之间自动流转——否则分层存储就只是"只降不升"的半成品。

**延伸阅读**：
- Meta CacheLib: https://cachelib.org/
- Linux io_uring: https://kernel.dk/io_uring.pdf
- Count-Min Sketch 论文: https://dl.acm.org/doi/10.1145/1073711.1073718

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
