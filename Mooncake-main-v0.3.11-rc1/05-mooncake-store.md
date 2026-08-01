# Mooncake Store 详解：分布式 KV Cache 的"中央仓储"系统

> **系列**: Mooncake 技术博客系列 | **类型**: 核心模块深潜篇
>
> 如果 Transfer Engine 是"高速公路"，那 Mooncake Store 就是高速公路两端的"中央仓储"——货物（KV Cache）在这里入库、出库、调拨、淘汰，而仓储经理（Master）从不下仓库搬货，只管账本。

---

### 引言

想象一个大型连锁仓储系统：中央办公室（Master）负责管理所有仓库的库存账本、空间分配和货物调度；每个门店（Client）既是货物的消费者，也是仓库空间的贡献者。门店之间可以直接调货（零拷贝 RDMA），无需经过中央办公室。中央办公室只管账本，从不碰货物——这就是 Mooncake Store 的 Master-Client 架构。

Mooncake Store 是建立在 Transfer Engine 之上的分布式 KV Cache 存储引擎。它为 LLM 推理场景提供了 `Put/Get/Remove` 等对象级操作，支持多副本、分层存储、弹性扩缩容和故障容忍。

今天这篇文章，我们深入 Mooncake Store 的内部设计。

---

### 核心架构：Master 与 Client

##### 两大组件

| 组件 | 职责 | 部署方式 |
|------|------|---------|
| **Master Service** | 集群协调、空间分配、元数据管理、驱逐策略 | 独立进程 |
| **Client** | 提供 Put/Get 接口；同时贡献内存给集群 | 嵌入推理引擎或独立运行 |

关键设计原则：**Master 永远不在数据路径上**。

```
┌────────────┐     元数据 RPC      ┌────────────┐
│  Client A  │◄──────────────────►│   Master   │
│ (Prefill)  │                    │  Service   │
└─────┬──────┘                    └────────────┘
      │                                  ▲
      │ 零拷贝 RDMA 直传                  │ 元数据 RPC
      │ (数据不经 Master)                 │
      ▼                                  │
┌────────────┐                    ┌─────┴──────┐
│  Client B  │◄──────────────────►│  Client C  │
│ (Decode)   │     元数据 RPC      │ (Store)    │
└────────────┘                    └────────────┘
```

##### Master 的核心 RPC 接口

| RPC | 说明 |
|-----|------|
| `MountSegment` | Client 注册内存段到集群 |
| `UnmountSegment` | Client 注销内存段 |
| `PutStart` | 开始写入：分配空间，返回目标位置 |
| `PutEnd` | 完成写入：标记数据可见 |
| `GetReplicaList` | 查询对象的副本位置 |
| `Remove` | 删除对象 |
| `Upsert/BatchUpsert` | 更新或插入对象 |

##### Client 的核心 API

```cpp
// mooncake-store/include/pyclient.h
class PyClient {
    // 对象操作
    int put(const std::string& key, std::span<const char> value,
            const ReplicateConfig& config = {});
    int64_t get_into(const std::string& key, void* buffer, size_t size);
    int remove(const std::string& key, bool force = false);
    
    // 批量操作
    int put_batch(const std::vector<std::string>& keys, ...);
    std::vector<int64_t> batch_get_into(...);
    std::vector<int> batch_remove(...);
    
    // 复制/移动任务
    tl::expected<UUID, ErrorCode> create_copy_task(key, targets);
    tl::expected<UUID, ErrorCode> create_move_task(key, source, target);
    tl::expected<QueryTaskResponse, ErrorCode> query_task(task_id);
    
    // 健康检查
    int health_check();
};
```

---

### 两阶段 Put：防止脏读的"入库双签"

Mooncake Store 的 `Put` 操作分为两个阶段——`PutStart` 和 `PutEnd`。这是保证数据一致性的关键设计。

##### 为什么需要两阶段？

在分布式系统中，数据写入不是瞬间完成的——尤其是通过 RDMA 跨节点传输几十 GB 的 KV Cache。如果写入过程中有其他 Client 读取，就会看到"半成品"数据——这就是脏读。

##### 两阶段流程

```
Client A                     Master                    Client B (存储节点)
   │                           │                           │
   │── PutStart(key, size) ──►│                           │
   │                           │── 分配空间 (选择目标) ───►│
   │◄── 目标位置信息 ─────────│                           │
   │                                                       │
   │── 直接写入数据到 Client B (零拷贝 RDMA) ────────────►│
   │   (此时 Get 看不到这个 key)                            │
   │                                                       │
   │── PutEnd(key) ──────────►│                           │
   │                           │── 标记数据可见 ──────────►│
   │◄── 确认 ─────────────────│                           │
   │                                                       │
   │                    此时 Get 可以读到完整数据            │
```

| 阶段 | 操作 | 可见性 |
|------|------|-------|
| PutStart | Master 分配空间，返回目标位置 | 不可见 |
| 数据写入 | Client 直传数据到存储节点 | 不可见 |
| PutEnd | Master 标记数据可见 | 可见 |

> 笔者注：两阶段 Put 的设计保证了 `Get` 永远读到**完整一致**的数据。但注意，这不保证"最新"——PutEnd 之前的写入对 Get 不可见。这是"最终一致性"的一种变体，适合 LLM 推理场景——推理系统更关心数据完整性而非实时性。

---

### 三种部署模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Embedded** | Client 嵌入推理引擎进程 | 单实例部署，最简单 |
| **Dummy-Real** | Dummy Client 转发到 Real Client | 多 GPU 推理（每 GPU 一个 Dummy） |
| **Standalone** | Client 作为独立存储服务 | 存储与计算分离 |

##### Embedded 模式

```
┌─────────────────────────────┐
│        vLLM 进程             │
│  ┌─────────┐  ┌──────────┐ │
│  │ vLLM    │  │ Mooncake │ │
│  │ Engine  │──│ Client   │ │
│  └─────────┘  │ (Real)   │ │
│               └────┬─────┘ │
│                    │ 内存   │
│               ┌────▼─────┐ │
│               │ 本地内存池 │ │
│               └──────────┘ │
└─────────────────────────────┘
```

##### Dummy-Real 模式

```
┌────────────┐  ┌────────────┐  ┌────────────┐
│ vLLM Rank0 │  │ vLLM Rank1 │  │ vLLM Rank2 │
│ Dummy      │  │ Dummy      │  │ Dummy      │
│ Client     │  │ Client     │  │ Client     │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │ IPC           │ IPC           │ IPC
      └───────────────┼───────────────┘
                      │
              ┌───────▼───────┐
              │  Real Client  │
              │  (拥有内存)    │
              └───────────────┘
```

在多 GPU 推理中，每个 GPU Rank 运行一个 Dummy Client，通过 IPC 转发请求到一个拥有实际内存的 Real Client。这避免了每个 Rank 都注册大量内存导致的资源浪费。

---

### 内存分配：OffsetBufferAllocator

Mooncake Store 的内存分配器专为 LLM 推理场景优化。

##### 两种分配器

| 分配器 | 算法 | 状态 | 适用场景 |
|--------|------|------|---------|
| OffsetBufferAllocator | O(1) 实时分配 | 默认 | LLM 推理（对象大小相对均匀） |
| CachelibBufferAllocator | Slab 分配 | 已弃用 | 通用缓存 |

##### 分配策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `random` | 随机选择目标 Segment | 默认，最快 |
| `free_ratio_first` | 优先选择空闲比例高的 Segment | 最佳内存均衡 |
| `ssd_free_ratio_first` | SSD 感知的空闲比例优先 | SSD 分层存储 |
| `cxl` | CXL 内存优先 | CXL 内存架构 |
| `local_first` | 优先选择本地 Segment | 同机部署 |

> 笔者注：`free_ratio_first` 看起来更合理，但 `random` 在大多数场景下表现更好——因为 LLM 推理的对象大小相对均匀，随机分配天然就能实现较好的均衡，而 `free_ratio_first` 的额外计算开销反而不划算。

---

### 多副本与驱逐策略

##### ReplicateConfig

```cpp
struct ReplicateConfig {
    int replica_count;              // 副本数
    std::vector<std::string> preferred_segments;  // 优先放置的 Segment
    bool soft_pin;                  // 软锁定：驱逐时尽量保留
    bool hard_pin;                  // 硬锁定：不可驱逐
};
```

##### 驱逐策略

当集群内存不足时，Master 会触发驱逐：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `eviction_ratio` | 0.5 | 驱逐时释放的比例 |
| `eviction_high_watermark_ratio` | 0.8 | 触发驱逐的内存水位 |
| `client_ttl` | - | Client 超时时间，超时后其内存可被回收 |

驱逐优先级：hard_pin > soft_pin > 无保护。被驱逐的对象如果仍有副本，Get 仍可成功。

---

### 分层存储：从 DRAM 到 SSD

分层存储，本质是扩展存储容量，HBM太贵了扩展DRAM，DRAM仍然很贵，就需要继续扩展下一层存储，就轮到SSD了。

Mooncake Store 支持多层存储——当 DRAM 不足时，冷数据可以自动卸载到 SSD。后面有一篇文章专门讲 SSD 分层存储的具体实现。

```
┌─────────────────────────────────┐
│           热数据层 (DRAM)         │  ← 命中率高，延迟低
│  KV Cache Block 0, 1, 2, ...   │
├─────────────────────────────────┤
│           冷数据层 (SSD/NVMe)    │  ← 容量大，延迟高
│  KV Cache Block 100, 101, ...   │
└─────────────────────────────────┘
```

| 操作 | 说明 |
|------|------|
| Offload | DRAM 不足时，冷数据写入 SSD |
| Promotion | SSD 数据被命中时，提升回 DRAM |

配置参数：

```bash
# Master 启动参数
--enable_offload=true
--offload_on_evict=true        # 驱逐时卸载到 SSD
--promotion_on_hit=true        # 命中时提升回 DRAM
--ssd_offload_path=/data/ssd   # SSD 存储路径
```

---

### 高可用：Master 集群

单点 Master 是故障单点。Mooncake Store 支持通过 etcd 或 Redis 实现 Master 高可用。

##### HA 架构

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Master 1 │     │ Master 2 │     │ Master 3 │
│ (Leader) │     │(Follower)│     │(Follower)│
└────┬─────┘     └──────────┘     └──────────┘
     │
     │ etcd Leader Election
     │
┌────▼─────┐
│   etcd   │
│ Cluster  │
└──────────┘
```

| HA 后端 | 配置 | 适用场景 |
|---------|------|---------|
| etcd | `--enable_ha --ha_backend_type=etcd --etcd_endpoints=...` | 生产环境 |
| Redis | `--enable_ha --ha_backend_type=redis` | 轻量级 HA |

HA 模式下还支持**快照与恢复**——Master 定期将元数据快照持久化，故障恢复后从快照重建状态。

---

### 多租户配额

Mooncake Store 支持多租户内存配额，防止单个租户占满集群内存：

```bash
# Master 启动参数
--enable_multi_tenants=true
--tenant_quota_connector_type=file  # 或 etcd
```

配额策略：

| 类型 | 说明 |
|------|------|
| 严格配额 | 超出配额的 Put 请求被拒绝 |
| 宽松配额 | 超出配额时发出告警但不拒绝 |

---

### 设计哲学：Mooncake Store 的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **控制面与数据面分离** | Master 只管元数据，数据直传 | Client-to-Client 零拷贝 RDMA |
| **一致性优先** | Get 永远读到完整数据 | 两阶段 Put，PutEnd 前不可见 |
| **弹性与容错** | 动态扩缩容，部分故障不影响正确性 | Master HA、Client 动态加入/退出 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| Master-Client | Master 管元数据，Client 持数据，数据直传不经 Master |
| 两阶段 Put | PutStart 分配空间 → 写入 → PutEnd 标记可见，防止脏读 |
| Embedded/Dummy-Real/Standalone | 三种部署模式，从简单到灵活 |
| OffsetBufferAllocator | O(1) 实时内存分配，专为 LLM 场景优化 |
| 分层存储 | DRAM → SSD 自动卸载，命中时提升 |
| Master HA | etcd/Redis Leader Election + 快照恢复 |

**建议**: 部署 Mooncake Store 时，先用 Embedded 模式跑通功能，再根据规模切换到 Dummy-Real 或 Standalone——Embedded 模式最简单，适合快速验证。

**延伸阅读**：
- Mooncake Store 设计文档：https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html
- 部署指南：https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
