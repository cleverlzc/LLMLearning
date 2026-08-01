# Mooncake 元数据与数据流：Master 不碰数据，Transfer Engine 不碰元数据

> **系列**: Mooncake 技术博客系列 | **类型**: 核心概念深潜篇

Mooncake 最核心的架构决策是：**元数据和数据走两条完全不同的路**。Master 只管"数据在哪里"，Transfer Engine 只管"把数据搬过去"，Client 在中间协调。数据永远不会流经 Master。本文用模拟对话的方式，追踪一次完整的 KV Cache 存取过程，看三个组件如何分工协作。

注意这个模式：Master服务只告诉你数据在哪里，即管理元数据。Transfer Engine实际移动数据。Client协调两者——首先向Master节点请求地址，然后指示Transfer Engine在该地址写入或读取数据。数据永远不会流经Master节点。

Master服务一般是Master Service API：Mooncake系统中用于追踪元数据（metadata）的中央协调器——记录哪些机器（which machines）拥有哪些数据（what data）以及数据的位置（where to find it）。

---

### 引言：为什么 Master 不能碰数据？

很多分布式存储系统的 Master 既管元数据又代理数据——客户端把数据发给 Master，Master 转发给存储节点。这种方式简单，但有一个致命问题：**Master 成为带宽瓶颈**。

Mooncake 面对的场景是 KV Cache 传输——数据量随序列长度暴涨：

| 模型 | 序列长度 | KV Cache 大小 | 场景 |
|------|---------|-------------|------|
| LLaMA-70B (GQA-8, tp=8) | 4K token | **160 MB** | 短对话 |
| LLaMA-70B (GQA-8, tp=8) | 32K token | **1.25 GB** | 长文档摘要 |
| LLaMA-70B (GQA-8, tp=8) | 128K token | **5 GB** | 超长上下文 |
| GLM-5 (MLA, tp=8) | 4K token | **437 MB** | 中文短对话 |
| GLM-5 (MLA, tp=8) | 32K token | **3.4 GB** | 中文长文档 |
| GLM-5 (MLA, tp=8) | 128K token | **13.7 GB** | 中文超长上下文，一张网卡传不完 |
| Kimi-K2.6 (MLA, tp=8) | 4K token | **274 MB** | 中文短对话 |
| Kimi-K2.6 (MLA, tp=8) | 32K token | **2.2 GB** | 中文长文档 |
| Kimi-K2.6 (MLA, tp=8) | 128K token | **8.7 GB** | 中文超长上下文 |
| Mistral-7B (MHA, tp=1) | 8K token | **4 GB** | MHA 模型，KV Cache 更大 |
| Mistral-7B (MHA, tp=1) | 32K token | **16 GB** | 长序列 MHA，一张网卡传不完 |

如果每次传输都经过 Master，Master 的网卡带宽就是整个集群的吞吐上限——一个 16 GB 的请求就能把 Master 的 100 Gbps 网卡堵 1.3 秒。

```
传统架构 (Master 代理数据):
  Client-A → Master → Client-B
                ↑
          带宽瓶颈！Master 的网卡带宽 = 集群吞吐上限

Mooncake 架构 (Master 只管元数据):
  Client-A → Master: "数据放哪？" → Master: "放 Client-B, 偏移 0x1A00"
  Client-A → Transfer Engine: "写到 Client-B:0x1A00" → RDMA 直传
                ↑
          Master 只传几 KB 的元数据，数据走 RDMA 直连
```

---

### 三个角色的分工

| 角色 | 职责 | 不做什么 |
|------|------|---------|
| **Master** | 记录"谁有什么数据、数据在哪里" | 不碰数据、不搬数据、不做传输 |
| **Transfer Engine** | 在节点之间搬数据 | 不知道搬的是什么、不知道为什么搬 |
| **Client** | 协调 Master 和 Transfer Engine | 是调用者，不是服务端 |

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Prefill Client (P)              Decode Client (D)                   │
│  ┌──────────────┐                ┌──────────────┐                   │
│  │ 1. 问 Master  │                │ 1. 问 Master  │                   │
│  │    数据放哪？ │                │    数据在哪？ │                   │
│  │ 3. 告诉 Master│                │ 3. 读取数据   │                   │
│  │    数据存好了 │                │              │                   │
│  └──────┬───────┘                └──────┬───────┘                   │
│         │ 2. 写数据                      │ 2. 读数据                 │
│         ▼                               ▼                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │              Transfer Engine (T)                   │               │
│  │   RDMA 直传: P 的 GPU → D 的 GPU                   │               │
│  │   不知道搬的是什么，只知道地址和长度                  │               │
│  └──────────────────────────────────────────────────┘               │
│         ▲                               ▲                           │
│         │ 元数据操作                     │ 元数据操作                 │
│  ┌──────┴───────────────────────────────┴───────┐                   │
│  │              Master (M)                        │                   │
│  │   追踪: 哪些机器有什么数据、数据在哪里          │                   │
│  │   不碰数据、不搬数据                            │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 场景 1：写入 KV Cache——Prefill 存数据

模拟 Prefill 节点完成 Prefill 计算后，将 KV Cache 存入 Mooncake 系统，对话过程如下：

```
P:  Hey Master, 我要为 key='prompt-42' 存储 2MB 的 KV Cache。
    我是 client-id=abc123，租户是 tenant-A。

M:  收到！我来分配空间。
    → 检查租户 tenant-A 的配额
    → 在 Client-B 的段上分配 2MB 空间
    → 预留配额
    → 返回地址: {段: Client-B, 偏移: 0x1A00, 长度: 2MB, 协议: rdma}
    注意: 此时数据状态是 PROCESSING（处理中），对其他 Client 不可见

P:  Transfer Engine, 把 2MB 数据写到 Client-B:0x1A00

T:  收到！
    → 打开 Client-B 的段 (openSegment)
    → 构建 TransferRequest {opcode=WRITE, source=本地GPU地址, target_id=段句柄,
                             target_offset=0x1A00, length=2MB}
    → RDMA WRITE 直传到 Client-B 的 GPU 内存
    → 完成！

P:  Hey Master, key='prompt-42' 的数据已经写好了 (PutEnd)

M:  收到！
    → 将数据状态从 PROCESSING 改为 COMPLETE
    → 将预留配额转为已提交配额
    → 授予租约 (lease)
    → 现在其他 Client 可以看到这条数据了
```

对话样例中的 prompt-42 为具体的提示词，比如：Explain quantum computing（解释量子计算）。


通过理解以上对话，想必对P（Prefill Client），M（Master），T（Transfer Engine），D（Decode Client）具体作用有了比较清晰的了解，

在进行具体的源码说明之前，我们先来介绍一下二阶段提交，以及PutStart到PutEnd过程的几个关键信息，先建立一个感性认识，再结合源码理解具体代码实现：

Master Service API揭示了**二阶段提交模式（two-phase commit pattern）**：PutStart预留空间并返回一个地址（address，比如Client-B segment, offset 0x1A00），PutEnd确认数据已存储。在这两次调用之间，传输引擎执行实际的工作。

- **PutStart**： “Master API，我需要为键 'prompt-42' 存储 2MB 的 KV 缓存。” Master节点检查可用空间，在一个或多个副本上预留该空间，并返回一个地址列表（段、偏移量、长度），数据应写入这些地址。

- **client_id**：谁在提问？每个客户端都会获得一个唯一的ID，以便Master可以追踪谁拥有什么。

- **key + tenant_id**：你存储的是什么，它属于哪个租户（一个为特定用户组提供的独立工作空间workspace或命名空间namespace——就像一个独立的办公楼层，其中一家公司的文件与另一家公司的文件分开存放）？key 是查找名称；tenant 用于隔离不同的用户（An isolated workspace or namespace for a group of users）。

- **slice_length**：数据有多大？Master程序需要这个信息来找到一块合适的内存。

- **ReplicateConfig**：需要多少份副本？客户端可以请求将数据复制到多台机器上以确保可靠性。

- **PutEnd**：“Master API，数据已写入，你可以将其标记为完成。” 这将关闭预留状态，并使数据对可能查找它的其他客户端可见。

请注意，`PutStart` 和 `PutEnd` 都不会移动任何数据。它们纯粹是元数据操作（purely metadata operations）。实际的数据移动发生在 PutStart 和 PutEnd 之间，此时客户端直接调用传输引擎（Transfer Engine）。Master节点是一个协调器，而不是数据代理。


##### PutStart 的源码说明

```cpp
// mooncake-store/src/master_service.cpp (第 3055-3222 行)
auto MasterService::PutStart(
    const UUID& client_id,       // 谁在存？
    const std::string& key,      // 存的是什么？
    const std::string& tenant_id,// 属于哪个租户？
    const uint64_t slice_length, // 数据有多大？
    const ReplicateConfig& config)// 需要几份副本？
    -> tl::expected<std::vector<Replica::Descriptor>, ErrorCode>
{
    // 1. 参数校验
    if (slice_length == 0) return tl::make_unexpected(INVALID_ARGUMENT);

    // 2. 获取对象操作锁 (防止并发写同一个 key)
    acquire_object_lock(key);

    // 3. 尝试分配空间
    attempt_once: {
        // 3a. 检查 key 是否已存在
        if (key_exists && status == COMPLETE)
            return OBJECT_ALREADY_EXISTS;

        // 3b. 预留租户配额
        auto quota_result = ReserveTenantQuota(tenant_id, reserved_quota);

        // 3c. 选择段并分配偏移
        auto allocation_result = allocation_strategy_->Allocate(
            allocator_manager,    // 可用的段和分配器
            slice_length,         // 需要的大小
            config.replica_num,   // 副本数
            preferred_segments,   // 优先的段 (如本机段)
            excluded_segments,    // 排除的段
            ReplicaType::MEMORY,  // 副本类型
            ssd_provider          // SSD 后端 (可选)
        );

        // 3d. 构建返回值: 每个副本的地址描述
        std::vector<Replica::Descriptor> replica_list;
        for (const auto& replica : replicas) {
            replica_list.emplace_back(replica.get_descriptor());
            // Descriptor 包含: buffer_address_, size_, protocol_, transport_endpoint_
        }

        // 3e. 插入元数据 (状态=PROCESSING)
        tenant_state.metadata.emplace(key, ObjectMetadata{
            client_id, now, slice_length, std::move(replicas), ...
        });
        tenant_state.processing_keys.insert(key);

        return replica_list;
    }
}
```

**PutStart 返回的 `Replica::Descriptor` 包含什么？**

```cpp
// mooncake-store/include/replica.h
struct Descriptor {
    uint64_t buffer_address_;          // 数据在目标段的内存地址
    uint64_t size_;                    // 分配的大小
    std::string protocol_;             // "rdma" / "tcp" / "cxl"
    std::string transport_endpoint_;   // 传输端点 (用于 openSegment)
};
```

这就是 Transfer Engine 需要的全部信息——**地址、大小、协议、端点**。Transfer Engine 不需要知道这是 KV Cache、不需要知道属于哪个 key、不需要知道是第几层。


##### 二阶段提交的本质

```
PutStart                    Transfer Engine               PutEnd
   │                            │                          │
   │ 1. 预留空间                 │                          │
   │ 2. 返回地址                 │                          │
   │ 3. 状态=PROCESSING          │                          │
   │ 4. 对其他 Client 不可见      │                          │
   │                            │                          │
   ▼                            ▼                          │
   ────────── 数据传输 (RDMA WRITE) ──────────              │
                                │                          │
                                │ 传输完成                   │
                                │                          ▼
                                                          │ 5. 状态=COMPLETE
                                                          │ 6. 配额从预留→已提交
                                                          │ 7. 授予租约
                                                          │ 8. 对其他 Client 可见
```

**为什么需要二阶段？** 因为数据传输可能失败。如果 PutStart 直接把数据标记为 COMPLETE，但传输实际失败了，其他 Client 就会读到不完整的数据。二阶段提交保证了：**只有确认传输成功的数据，才会对其他 Client 可见**。

---

### 场景 2：读取 KV Cache——Decode 取数据

模拟 Decode 节点需要读取 KV Cache 来开始 Decode 计算，对话过程如下：

```
D:  Hey Master, 我需要 key='prompt-42' 的 KV Cache，租户是 tenant-A。

M:  查到了！
    → 在元数据中找到 key='prompt-42'
    → 返回: {段: Client-B, 偏移: 0x1A00, 长度: 2MB, 协议: rdma}
    → 续约租约 (保证数据不会被驱逐)
    → 如果数据在 SSD 上，触发晋升 (promotion) 到 DRAM

D:  Transfer Engine, 从 Client-B:0x1A00 读取 2MB

T:  收到！
    → 打开 Client-B 的段
    → 构建 TransferRequest {opcode=READ, source=远程地址, target_id=本地段,
                             target_offset=本地偏移, length=2MB}
    → RDMA READ 从 Client-B 的 GPU 内存拉取
    → 已传递！这是你的KV Cache

D:  KV Cache 到手，开始 Decode 计算
```

##### GetReplicaList 的源码说明

```cpp
// mooncake-store/src/master_service.cpp (第 2527-2599 行)
auto MasterService::GetReplicaList(
    const std::string& key,
    const std::string& tenant_id)
    -> tl::expected<GetReplicaListResponse, ErrorCode>
{
    // 1. 查找元数据
    MetadataAccessorRO accessor(this, object_id);

    // 2. 收集所有 COMPLETE 状态的副本
    std::vector<Replica::Descriptor> replica_list;
    metadata.VisitReplicas(
        &Replica::fn_is_completed,   // 只看 COMPLETE 的
        [&replica_list](const Replica& replica) {
            replica_list.emplace_back(replica.get_descriptor());
        }
    );

    // 3. 续约租约 (防止数据被驱逐)
    metadata.GrantLease(default_kv_lease_ttl_, default_kv_soft_pin_ttl_);

    // 4. 检查是否需要从 SSD 晋升到 DRAM
    if (no_memory_replica && has_local_disk_replica) {
        trigger_promotion(key, tenant_id);
    }

    return GetReplicaListResponse{replica_list, lease_ttl_ms};
}
```

**注意**：读取没有"GetEnd"——因为读取不修改元数据状态，不需要确认。Client 拿到地址后直接读，读完了就完了。

---

### 场景 3：多副本写入——数据可靠性

当需要多份副本时，PutStart 返回多个地址，Client 依次写入，对话过程如下：

```
P:  Master, 我要为 key='prompt-42' 存储 2MB，需要 2 份副本。

M:  分配完成！返回 2 个地址:
    副本 1: {段: Client-B, 偏移: 0x1A00, 协议: rdma}
    副本 2: {段: Client-C, 偏移: 0x2B00, 协议: rdma}

P:  Transfer Engine, 把 2MB 写到 Client-B:0x1A00  ← 副本 1
T:  完成！

P:  Transfer Engine, 把 2MB 写到 Client-C:0x2B00  ← 副本 2
T:  完成！

P:  Master, 数据写好了 (PutEnd, 副本类型=ALL)

M:  两个副本都标记为 COMPLETE
```

##### Client 侧的完整 Put 流程

```cpp
// mooncake-store/src/client_service.cpp (第 1531-1639 行)
tl::expected<void, ErrorCode> Client::Put(
    const ObjectKey& key,
    std::vector<Slice>& slices,
    const ReplicateConfig& config)
{
    // Step 1: PutStart — 获取地址
    auto start_result = master_client_.PutStart(
        key, slice_lengths, client_cfg);
    // start_result = std::vector<Replica::Descriptor>

    // Step 2: 逐个副本传输数据
    TransferSummary transfer_summary;
    for (const auto& replica : start_result.value()) {
        if (replica.is_memory_replica()) {
            ErrorCode err = TransferWrite(replica, slices);
            transfer_summary.record(replica.type(), err);
        }
    }

    // Step 3: 决定如何收尾
    auto decision = DetermineFinalizeDecision(config, transfer_summary);
    // decision 包含: 哪些副本调 PutEnd，哪些调 PutRevoke

    // Step 4: PutEnd — 确认成功的副本
    if (decision.end_type.has_value()) {
        master_client_.PutEnd(key, *decision.end_type);
    }

    // Step 5: PutRevoke — 撤销失败的副本
    if (decision.revoke_type.has_value()) {
        master_client_.PutRevoke(key, *decision.revoke_type);
    }
}
```

**部分成功怎么办？** 如果 2 个副本中只有 1 个写入成功，`DetermineFinalizeDecision` 会：
- 对成功的副本调 PutEnd（标记为 COMPLETE）
- 对失败的副本调 PutRevoke（释放空间）
- 整体 Put 返回成功（至少有 1 份可用）

---

### 场景 4：Transfer Engine 的视角——只认地址不认人

Transfer Engine 看到的世界非常简单——只有地址和长度：

```cpp
// mooncake-transfer-engine/include/transport/transport.h (第 60-71 行)
struct TransferRequest {
    enum OpCode { READ, WRITE };

    OpCode opcode;           // 读还是写
    void *source;            // 源地址 (本地 GPU 内存指针)
    SegmentID target_id;     // 目标段 ID (openSegment 返回)
    uint64_t target_offset;  // 目标偏移
    size_t length;           // 传输字节数
};
```

**Transfer Engine 不知道的**：这是 KV Cache 还是模型权重、属于哪个 key、是第几层、是 K 还是 V。它只知道"从 source 搬 length 字节到 target_id:target_offset"。

##### 本地 vs 远程——Transfer Engine 的两种策略

```cpp
// mooncake-store/src/transfer_task.cpp (第 981-1036 行)
TransferSubmitter::submit(...) {
    if (is_local_memcpy) {
        // 本地: 源和目标在同一个进程，直接 memcpy
        strategy = LOCAL_MEMCPY;
        // 通过线程池并行拷贝
    } else {
        // 远程: 走 Transfer Engine (RDMA/TCP)
        strategy = TRANSFER_ENGINE;
        // 构建 TransferRequest, submitTransfer
    }
}
```

```
本地拷贝 (同一台机器):
  Client-A 的 GPU → memcpy → Client-B 的 GPU
  不经过网络，不走 RDMA，纯内存拷贝

远程传输 (不同机器):
  Client-A 的 GPU → RDMA WRITE → Client-B 的 GPU
  绕过 CPU，网卡直接 DMA 读写
```

---

### Master 内部的元数据结构

Master 为每个对象维护的元数据：

```cpp
// mooncake-store/include/master_service.h (第 856-970 行)
struct ObjectMetadata {
    UUID client_id;                                         // 谁存的
    std::chrono::system_clock::time_point put_start_time;   // 什么时候存的
    const size_t size;                                      // 多大
    const ObjectDataType data_type;                         // 数据类型
    const std::string group_id;                             // 路由分组
    const std::string tenant_id;                            // 租户
    const std::string user_key;                             // 用户可见的 key

    mutable std::chrono::system_clock::time_point lease_timeout;  // 硬租约超时
    mutable std::optional<...> soft_pin_timeout;                    // 软钉扎超时

    uint64_t reserved_quota_charge_bytes;   // PutStart 时预留的配额
    uint64_t committed_quota_charge_bytes;  // PutEnd 时提交的配额

    // 副本列表
    std::vector<Replica> replicas;          // 每个副本有自己的状态和地址
};
```

元数据按租户分片存储：

```
metadata_shards_[hash(tenant_id + key) % num_shards]
    .tenants[tenant_id]
    .metadata[key] = ObjectMetadata{...}
```

##### 副本的状态机

```
         PutStart              PutEnd
UNDEFINED ──────→ PROCESSING ──────→ COMPLETE
                    │                    │
                    │ PutRevoke          │ 租约过期 / 驱逐
                    ▼                    ▼
                  已释放               已释放
```

- **PROCESSING**：PutStart 后、PutEnd 前。对其他 Client 不可见。
- **COMPLETE**：PutEnd 后。对其他 Client 可见，可被 GetReplicaList 返回。
- 租约过期后，Master 可以驱逐数据，释放空间。

---

### 元数据持久化：内存为主 + etcd 兜底

上面的 `ObjectMetadata` 存在 Master 的**内存**中——1024 个分片，每个分片一把读写锁。这意味着：**Master 一重启，内存中的元数据就全丢了。**

那元数据如何持久化？Mooncake 用的是**快照 + 操作日志**的经典方案，和数据库的 WAL（Write-Ahead Log）思路一致：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Master 元数据持久化架构                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  内存 (Primary Master)                                       │    │
│  │  metadata_shards_[1024]  ← 日常读写都在这里，微秒级           │    │
│  │                                                              │    │
│  │  OpLogManager                                               │    │
│  │    └─ buffer_[deque<OpLogEntry>]  ← 每次写操作记一条日志      │    │
│  └──────────────────────┬──────────────────────────────────────┘    │
│                         │ 写入                                       │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  etcd (持久化存储)                                            │    │
│  │                                                              │    │
│  │  /oplog/{cluster}/{seq}     ← 操作日志 (每条一个 key)         │    │
│  │  /oplog/{cluster}/latest    ← 最新序列号                      │    │
│  │  mooncake_master_snapshot/  ← 定期快照 (msgpack + Zstd 压缩)  │    │
│  │  mooncake-store/{cluster}/  ← Leader 选举 (lease-based CAS)  │    │
│  │    master_view                                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  恢复路径:                                                           │
│  1. 加载最新快照 → 得到元数据基线                                     │
│  2. 重放快照之后的 OpLog → 追上最新状态                                │
│  3. Watch 前缀 → 实时同步后续变更                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

##### 操作日志 (OpLog)：每次写操作都留痕

Master 的每次元数据变更都会写入 OpLog：

```cpp
// mooncake-store/include/ha/oplog/oplog_manager.h (第 36-45 行)
struct OpLogEntry {
    uint64_t sequence_id{0};      // 单调递增的序列号
    uint64_t timestamp_ms{0};     // 时间戳
    OpType op_type;               // 操作类型
    std::string object_key;       // 对象 key
    std::string payload;          // 序列化后的元数据 (msgpack)
    uint32_t checksum{0};         // xxHash 校验和
    uint32_t prefix_hash{0};      // key 的哈希 (用于分片路由)
};

// 操作类型
enum class OpType : uint8_t {
    PUT_END    = 1,   // 对象创建/更新
    PUT_REVOKE = 2,   // 对象撤销 (PutStart 后未 PutEnd)
    REMOVE     = 3,   // 对象删除
};
```

**两种写入策略**——不同操作类型有不同的持久化要求：

| 操作类型 | 写入策略 | 为什么 | 延迟 |
|---------|---------|--------|------|
| PUT_END | 异步批量写入 | 数据已安全落盘，OpLog 只是让 Standby 追上 | 微秒级 |
| PUT_REVOKE | 同步等待 etcd 确认 | 防止 Standby 提升后"复活"已撤销的数据 | 毫秒级 |
| REMOVE | 同步等待 etcd 确认 | 防止 Standby 提升后"复活"已删除的数据 | 毫秒级 |

OpLog 在 etcd 中的 key 格式：

```
/oplog/my-cluster/00000000000000000001   ← 第 1 条日志
/oplog/my-cluster/00000000000000000002   ← 第 2 条日志
/oplog/my-cluster/00000000000000012345   ← 第 12345 条日志
/oplog/my-cluster/latest                 ← 值: "12345" (最新序列号)
```

序列号用 20 位零填充，保证 etcd 的字典序 = 时间序。

##### 快照 (Snapshot)：定期全量备份

OpLog 会不断增长，不能无限重放。快照就是 OpLog 的"截断点"——定期把内存中的完整元数据序列化存入 etcd：

```
快照在 etcd 中的 key 布局:

mooncake_master_snapshot/my-cluster/20260723_123456789/
    manifest.txt       → "messagepack|1.0.0" (协议|版本)
    descriptor.txt     → "12345|42|1700000000000" (最后序列号|视图版本|创建时间)
    metadata           → msgpack 二进制 (Zstd 压缩的 1024 分片元数据)
    segments           → 序列化的 SegmentManager 状态
```

快照的元数据用 msgpack 序列化 + Zstd 压缩——一个 100 万对象的集群，压缩后可能只有几十 MB。

##### 故障恢复：快照 + OpLog 回放

当 Master 重启（或 Standby 提升为 Primary）时，恢复流程：

```
1. 加载最新快照
   → 从 etcd 读取 mooncake_master_snapshot/.../latest
   → 反序列化: msgpack 解码 → Zstd 解压 → 逐分片填充 metadata_shards_
   → 得到快照时刻的完整元数据基线

2. 重放 OpLog
   → 从快照的 sequence_id 开始，读取 /oplog/{cluster}/ 后续所有条目
   → 逐条应用: PUT_END → 插入元数据, REMOVE → 删除元数据
   → 追到 latest 序列号为止

3. 实时同步
   → Watch /oplog/{cluster}/ 前缀
   → Primary 写入新 OpLog 时，Standby 实时收到并应用
   → 保持与 Primary 的元数据同步

4. 提升为 Primary
   → 参与选举 (etcd CAS / K8s Lease)
   → 当选后等待旧 Leader 的租约过期 (防脑裂)
   → 开始接受 Client 请求
```

**OpLog 缺口处理**：如果 Standby 在重放过程中发现序列号不连续（比如 etcd 丢了几条），处理策略：

| 缺口中的操作 | 处理方式 | 原因 |
|------------|---------|------|
| PUT_END | 丢弃 | 数据可能已不完整，宁可丢不可错 |
| REMOVE | 应用 | 删除是幂等的，多删一次无害 |
| PUT_REVOKE | 应用 | 撤销也是幂等的 |

##### Master 元数据 vs Transfer Engine 元数据

两者**完全隔离**，使用不同的 etcd 客户端：

| | Master Service | Transfer Engine |
|--|--------------|----------------|
| etcd 客户端 | `storeClient` | `globalClient` |
| key 前缀 | `/oplog/`, `mooncake-store/`, `mooncake_master_snapshot/` | 传输引擎自己的 key |
| 用途 | KV 对象元数据、OpLog、选举、快照 | 连接/路由元数据 |
| 持久化 | 完整 HA：OpLog + 快照 + Watch | 简单 key-value |
| 消息大小 | 默认 (~1.5MB) | 32 MB |

**为什么隔离？** 因为 Master 重启时需要重置 etcd 连接（重新建 Watch、重新获取租约），如果和 Transfer Engine 共用客户端，重置会影响正在进行的传输。

---

### 空间分配：Master 如何选择"数据放哪"

当 PutStart 请求分配空间时，Master 的 `AllocationStrategy` 决定数据放在哪个段上：

```cpp
// mooncake-store/include/allocation_strategy.h
// 默认策略: RandomAllocationStrategy

1. 获取所有可用段 (已 Mount 的 Client 贡献的内存)
2. 优先选择 preferred_segments (如请求来源 Client 的本地段)
3. 随机选择段，确保不同副本放在不同段上
4. 在选定的段上调用 allocator->allocate(slice_length) 分配偏移
5. 返回每个副本的地址描述
```

```
段选择示例 (2 副本):

可用段: [Client-A: 4GB, Client-B: 6GB, Client-C: 2GB]
请求来自: Client-A

优先: Client-A (本地优先)
随机: Client-B 或 Client-C (不同段)

结果:
  副本 1 → Client-A:0x1000 (本地，低延迟)
  副本 2 → Client-B:0x2000 (远程，高可靠)
```

---

### 全景时序：一次完整的 PD 解耦 KV Cache 传输

```
时间 ──────────────────────────────────────────────────────────────→

P (Prefill)                     M (Master)                D (Decode)

1. PutStart('prompt-42', 2MB) ──→
                                 2. 分配空间
                                    返回: Client-B:0x1A00
←── 3. 地址列表 ──────────────────

4. TransferWrite(Client-B:0x1A00)
   └→ RDMA WRITE 2MB ─────────────────────────────→ Client-B GPU

5. PutEnd('prompt-42') ─────────→
                                 6. 状态 PROCESSING→COMPLETE
                                    授予租约
                                    发布 KV 存储事件

                                 7. D 节点收到事件通知
                                    (或 D 主动查询)

                                 8. GetReplicaList('prompt-42')
←── 9. 地址: Client-B:0x1A00 ────

10. TransferRead(Client-B:0x1A00)
    └→ RDMA READ 2MB ←───────────────────────────── Client-B GPU

11. KV Cache 到手，开始 Decode
```

**关键观察**：

1. **数据路径**：P 的 GPU → RDMA → D 的 GPU。Master 不在数据路径上。
2. **元数据路径**：P → M（地址查询）、P → M（确认完成）、D → M（地址查询）。都是几 KB 的小消息。
3. **时间占比**：元数据操作 < 1ms，数据传输 ~2ms（RDMA 2MB）。元数据开销可忽略。
4. **Master 的角色**：协调者，不是代理者。它告诉 Client "数据在哪里"，但从不碰数据本身。

---

### 总结：三个不变量

| 不变量 | 含义 | 代码体现 |
|--------|------|---------|
| **Master 不碰数据** | Master 只做元数据操作，数据走 Transfer Engine 直传 | PutStart/PutEnd 只操作 ObjectMetadata，不搬运 Slice |
| **Transfer Engine 不碰元数据** | Transfer Engine 只认地址和长度，不知道 key/tenant/layer | TransferRequest 只有 source/target_id/offset/length |
| **Client 是唯一协调者** | Client 先问 Master 地址，再指挥 Transfer Engine 搬数据 | Client::Put() 串联 PutStart → TransferWrite → PutEnd |

这三个不变量保证了：**Master 的带宽永远不会成为瓶颈，数据传输的吞吐只受 RDMA 网络带宽限制**。

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
