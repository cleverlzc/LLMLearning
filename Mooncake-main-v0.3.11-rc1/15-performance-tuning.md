# Mooncake 性能调优实践：给数据流水线"疏通河道"的实战指南

> **系列**: Mooncake 技术博客系列 | **类型**: 实战调优篇
>
> 性能调优就像疏通一条河道——水（数据）从上游（GPU）流向下游（SSD/网络），任何一个河段的狭窄都会成为瓶颈。疏通了这一段，下一段又变成最窄的地方。Mooncake 有 80+ 个可调参数，但真正影响性能的只有十几个关键参数，而且它们之间有明确的优先级。

根据笔者的经验来看，任何的存储系统，都必须要根据实际的业务场景进行调优，没有一套存储系统能够包打天下，包打所有场景，LLM推理KV Cache系统也不例外，Mooncake也不例外。

---

### 引言：瓶颈会"转移"

想象一条河流经过五道闸门。第一道闸门太窄，水都堵在这里——你加宽了第一道闸门，水流畅了，但马上发现第二道闸门又太窄。**疏通了一处瓶颈，下一处瓶颈就会暴露出来**——这就是性能调优的本质。

Mooncake 的数据流经五道"闸门"：

```
GPU 显存 ──闸门1: D2H 带宽──→ DRAM ──闸门2: RDMA 传输──→ 远程 DRAM
                                     │
                                     └──闸门3: SSD I/O──→ 本地 SSD ──闸门4: NVMe-oF──→ 远程 SSD
```

每一道闸门都有对应的参数控制"开多大"。

本文按瓶颈优先级从高到低，逐道闸门讲解调优方法。

> 笔者注：80+ 个参数看着吓人，但遵循一个原则——**先量后调，一次只调一个**。用 benchmark 工具定位瓶颈，再对症下药。盲目调参只会让系统更不稳定。这一点在部署踩坑实录文章中提到过，生产级稳定系统必然要经历大量的工程工作，需要有耐心，性能测试也要不厌其烦的测试，有时候需要成千上百组测试，都是正常的。

---

### 第一步：定位瓶颈——用 Benchmark 量化出来

Mooncake 提供了完整的基准测试工具：

| 工具 | 位置 | 测什么 |
|------|------|--------|
| TE 带宽基准 | `mooncake-transfer-engine/benchmark/` | RDMA/TCP 传输带宽和延迟 |
| 存储后端基准 | `mooncake-store/benchmarks/storage_backend_bench.cpp` | SSD 读写吞吐（buffered vs direct） |
| Master 性能基准 | `mooncake-store/benchmarks/master_bench.cpp` | Master RPC 吞吐 |
| PG 基准 | `mooncake-pg/benchmark/pgbench.py` | 集合通信性能 |
| vLLM 集成基准 | `benchmarks/xypd_benchmarks/vllm-benchmarks/` | 端到端推理吞吐 |

**调优的第一步永远是跑 benchmark，不是改参数。** 知道瓶颈在哪里，才能对症下药。

---

### 闸门 1：RDMA 传输——最常被卡住的河段

Transfer Engine 是 Mooncake 的数据大动脉。RDMA 参数调得好，带宽可以从默认的 60% 利用率提升到 90%+。

##### 切片大小：吞吐 vs 延迟的平衡点

Transfer Engine 把大块数据切为 **Slice** 传输。切片大小是最关键的参数：

```
MC_SLICE_SIZE=65536  (默认 64KB)

切片太小 → 切片数量多 → 元数据开销大 → 吞吐上不去
切片太大 → 单个切片传输时间长 → 延迟下不来
```

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| 大块 KV Cache 传输（PD 解耦） | `262144` (256KB) | 减少切片开销，最大化吞吐 |
| 小请求低延迟推理 | `32768` (32KB) | 更快完成单个切片，降低尾延迟 |
| 模型权重分发（P2P Store） | `65536` (64KB，默认) | 权重分片已按 4GB 拆分，切片大小影响不大 |

##### QP 深度与 CQ 容量：加宽河道

QP（Queue Pair）是 RDMA 的发送/接收队列，CQ（Completion Queue）是完成事件队列。它们决定了"河道有多宽"：

| 参数 | 默认值 | 高吞吐推荐 | 作用 |
|------|--------|-----------|------|
| `MC_NUM_QP_PER_EP` | 2 | 4 | 每个 endpoint 的 QP 数——多 QP 提供链路级并行 |
| `MC_MAX_WR` | 256 | 512-1024 | QP 的 WR 深度——更深 = 更多未完成请求在途 |
| `MC_MAX_CQE_PER_CTX` | 4096 | 8192 | CQ 容量——太小会导致 CQ overflow 错误 |
| `MC_WORKERS_PER_CTX` | 2 | 4-8 | CQ 轮询线程数——更多线程更快收割完成事件 |

```
默认配置: 2 QP × 256 WR = 512 个在途请求
调优配置: 4 QP × 512 WR = 2048 个在途请求 → 吞吐提升 2-4x
```

> 笔者注：增大 QP 深度和 CQ 容量会消耗更多 GPU 卡上的 RDMA 资源。在多进程共享同一张网卡时，要注意资源总量限制——不是越大越好，而是够用就好。

##### MTU：两层都要配——RDMA 层和以太网层

MTU 有两个层级，容易混淆：

| 层级 | 参数 | 可选值 | 最大值 | 控制什么 |
|------|------|--------|--------|---------|
| **RDMA 层** | `MC_MTU` | 512 / 1024 / 2048 / 4096 | **4096**（IB 规范上限） | RDMA QP 的单包有效载荷 |
| **以太网层** | `ip link set mtu` | 1500 / 9000 | **9000+**（Jumbo Frame） | 以太网帧的最大 IP 包大小 |

```
一个 RDMA 包的完整路径:

  应用数据 (最大 4096 B)     ← MC_MTU 控制
       │
       ▼
  RDMA 传输头 + 有效载荷       ← RDMA 层
       │
       ▼
  IP 包 (可能分片)             ← Ethernet MTU 控制
       │
       ▼
  以太网帧                     ← 物理层
```

**RDMA 层**：`MC_MTU` 始终设为 **4096**——这是 IB 规范的最大值，没有更大的选项。4096 让每个 RDMA 包携带更多有效数据，减少包数量和协议开销。

**以太网层**：RoCE 环境下务必开启 **Jumbo Frame（MTU=9000）**。如果以太网 MTU 只有 1500，一个 4096 字节的 RDMA 包会被 IP 层拆成 3 个以太网帧——增加分片开销和 CPU 中断。设为 9000 后，一个 RDMA 包只需 1 个以太网帧即可承载。

```bash
# 检查当前以太网 MTU
ip link show mlx5_0 | grep mtu

# 设置 Jumbo Frame (需要 root 权限，所有节点和交换机都要配)
ip link set dev mlx5_0 mtu 9000

# 持久化配置 (Ubuntu/Debian)
/etc/network/interfaces:
  iface mlx5_0 inet static
    mtu 9000

# Mooncake RDMA 层 MTU
export MC_MTU=4096
```

| 场景 | RDMA MTU | 以太网 MTU | 原因 |
|------|---------|-----------|------|
| 纯 IB 网络 (InfiniBand) | 4096 | 不适用 | IB 没有以太网层 |
| RoCE 网络 | 4096 | **9000** | Jumbo Frame 避免 IP 分片 |
| RoCE 网络（交换机不支持 9000） | 4096 | 1500 | 退而求其次，但会有分片开销 |

> 笔者注：RoCE 集群中，以太网 MTU=9000 是**必须配置**的基础设施要求，不是可选优化。如果只配了 `MC_MTU=4096` 而以太网 MTU 还是 1500，RDMA 包会被 IP 层分片，吞吐下降、尾延迟升高。确认方法：`ib_write_bw` 测试带宽如果远低于标称值，先检查以太网 MTU。

##### QoS：让关键流量优先过闸

在共享网络中，KV Cache 传输和训练流量可能争抢带宽。通过 IB Traffic Class 和 Service Level 实现 QoS 隔离：

| 参数 | 默认值 | 作用 | 场景 |
|------|--------|------|------|
| `MC_IB_TC` | -1 (未设) | RoCEv2 DSCP 标记（0-255） | 交换机按 DSCP 做差分服务 |
| `MC_IB_SL` | -1 (未设) | IB Service Level（0-15） | 映射到交换机 Virtual Lane |

生产建议：为推理流量设置独立的 TC/SL，与训练流量隔离。例如 `MC_IB_TC=32`（DSCP=8，高优先级），交换机侧配置对应的优先级队列。

##### RDMA 传输核心参数速查表

| 参数 | 环境变量 | 默认值 | 高吞吐推荐 | 低延迟推荐 |
|------|---------|--------|-----------|-----------|
| 切片大小 | `MC_SLICE_SIZE` | 65536 | 262144 | 32768 |
| QP 数 | `MC_NUM_QP_PER_EP` | 2 | 4 | 2 |
| WR 深度 | `MC_MAX_WR` | 256 | 512 | 256 |
| CQ 容量 | `MC_MAX_CQE_PER_CTX` | 4096 | 8192 | 4096 |
| 轮询线程 | `MC_WORKERS_PER_CTX` | 2 | 4-8 | 2 |
| RDMA MTU | `MC_MTU` | 4096 | 4096 | 4096 |
| 以太网 MTU | `ip link set mtu` | 1500 | 9000 | 9000 |
| Inline | `MC_MAX_INLINE` | 64 | 64 | 128 |
| 重试次数 | `MC_RETRY_CNT` | 9 | 9 | 12 |
| QoS TC | `MC_IB_TC` | -1 | 32 | 32 |
| PCI Relaxed Ordering | `MC_IB_PCI_RELAXED_ORDERING` | 0 | 1 | 1 |

> 笔者注：`MC_IB_PCI_RELAXED_ORDERING=1` 可以降低 PCIe 读延迟，但需要 CPU 和 PCIe 根复合体支持。设为 `2` 表示自动检测。如果硬件不支持，设为 1 可能导致数据错乱——先确认硬件支持再开启。

---

### 闸门 2：拓扑与路由——走最短路径

数据从 GPU 到远程 DRAM，走哪条 RDMA 网卡？选错网卡，数据可能绕远路——就像导航选了一条堵车的高速，而旁边那条空荡荡的路你没选。

##### NIC 亲和性：让数据走最近的网卡

AI 服务器通常有多张 RDMA 网卡（如 4-8 张 ConnectX-6/7），每张网卡与不同 NUMA 节点的距离不同。**数据应该从距离最近的网卡发出**——否则要跨 NUMA 节点搬运，延迟增加 30-50%。

| 参数 | 默认值 | 推荐值 | 作用 |
|------|--------|--------|------|
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | false | true | 让 slice 路由到目标端同 NUMA 的 NIC |
| `MC_ENABLE_HCA_PEER_AFFINITY` | false | true | 基于 NIC 拓扑选择最优本地 HCA |

两个参数互斥——只能开一个。`DEST_DEVICE_AFFINITY` 基于目标端位置选路，`HCA_PEER_AFFINITY` 基于对端 NIC 拓扑选路。推荐先试 `HCA_PEER_AFFINITY`。

##### 手动指定 NIC 亲和规则

自动发现不满足需求时，可以手动指定：

```bash
# 格式: local_hca1=peer_hca1,peer_hca2;local_hca2=peer_hca3
# 含义: 本地 mlx5_0 优先与对端 mlx5_0/mlx5_1 通信
export MC_NIC_PEER_AFFINITY="mlx5_0=mlx5_0,mlx5_1;mlx5_1=mlx5_2,mlx5_3"
```

##### 设备过滤：只用指定的网卡

```bash
# EP 场景：只用 mlx5_0 和 mlx5_1
export MOONCAKE_EP_DEVICE_FILTER="mlx5_0,mlx5_1"

# PG 场景：代码中设置
MooncakeBackend::setDeviceFilter({"mlx5_0", "mlx5_1"})
```

生产建议：在多网卡服务器上，**务必开启 NIC 亲和性**。不开启的话，数据可能随机选网卡，跨 NUMA 访问延迟高 30-50%，相当于把 RDMA 的延迟优势白白浪费。

---

### 闸门 3：内存管理——减少搬运次数

数据在内存中每多搬一次，就多一次延迟。内存调优的核心是**减少不必要的拷贝和地址翻译**。

##### 大页：加速地址翻译

普通 4KB 页，1GB 内存需要 262144 个页表项；2MB 大页只需要 512 个——地址翻译快 500 倍。对 RDMA 注册内存尤其重要：

| 参数 | 环境变量 | 默认值 | 推荐值 | 影响 |
|------|---------|--------|--------|------|
| 启用大页 | `MC_STORE_USE_HUGEPAGE` | 未设 | `true` | 加速 MR 注册、降低 TLB miss |
| 大页大小 | `MC_STORE_HUGEPAGE_SIZE` | 未设 | `2097152` (2MB) | 2MB 是最通用的大页大小 |

```bash
# 系统层预留大页（需要 root 权限）
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages  # 预留 1024 个 2MB 大页 = 2GB

# Mooncake 启用大页
export MC_STORE_USE_HUGEPAGE=true
export MC_STORE_HUGEPAGE_SIZE=2097152
```

##### Pinned Buffer Pool：D2H 的关键

GPU 数据搬到 DRAM，必须经过 Pinned Memory（页锁定内存）。池化复用避免反复分配：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kDefaultMaxPoolSize` | 32 | 池中最大 pinned buffer 数量 |

32 个缓冲区对于单卡场景足够。多卡场景下，如果日志出现频繁的 pinned buffer 分配/释放，可以适当增大。

##### 段大小：全局内存池的容量

| 参数 | 环境变量 | 默认值 | 推荐值 | 说明 |
|------|---------|--------|--------|------|
| 全局段大小 | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 3.125 GiB | 物理内存的 50-70% | 贡献给全局内存池的 DRAM 容量 |
| 本地缓冲大小 | `MOONCAKE_LOCAL_BUFFER_SIZE` | 1.0 GiB | 1-2 GiB | Transfer Engine 本地缓冲 |

```bash
# 512GB 物理内存的服务器，贡献 320GB 给全局池
export MOONCAKE_GLOBAL_SEGMENT_SIZE=320gb
export MOONCAKE_LOCAL_BUFFER_SIZE=2gb
```

> 笔者注：`MOONCAKE_GLOBAL_SEGMENT_SIZE` 不是越大越好。要给操作系统、推理引擎、CUDA 留足够内存。一般设为物理内存的 50-70%。设太大可能导致 OOM，设太小则 Store 容纳的 KV Cache 不够，频繁驱逐。

##### 并行 MR 注册：加速启动

注册大块内存到 RDMA 网卡很耗时（1GB 可能需要数秒）。并行注册可以显著加速：

| 参数 | 环境变量 | 默认值 | 推荐值 |
|------|---------|--------|--------|
| 并行 MR 注册 | `MC_ENABLE_PARALLEL_REG_MR` | -1 (auto) | 1 (开启) |

默认 auto 会根据内存大小自动决定。如果启动时间过长，显式设为 1 强制开启并行注册。

---

### 闸门 4：SSD I/O——DRAM 装不下时的最后一道闸

当 DRAM 使用率超过高水位，KV Cache 必须卸载到 SSD。SSD 的读写速度决定了卸载/提升的延迟。

##### io_uring：从"人工搬运"升级到"传送带"

| 参数 | 环境变量 | 默认值 | 推荐值 |
|------|---------|--------|--------|
| 启用 io_uring | `MOONCAKE_OFFLOAD_USE_URING` | false | **true** |

这是 SSD 调优**最简单也最有效**的一步。开启后，SSD 读写从 POSIX `pread/pwrite`（2 次拷贝）切换到 io_uring + O_DIRECT（1 次直传），吞吐提升 30-50%。

##### SSD 容量与驱逐

| 参数 | 环境变量 | 默认值 | 推荐值 | 说明 |
|------|---------|--------|--------|------|
| SSD 总容量限制 | `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | 2 TiB | SSD 容量的 80-90% | 留余量避免磁盘满 |
| 桶驱逐策略 | `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | fifo | lru | LRU 保留热数据 |
| 桶最大大小 | `MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES` | 256 MiB | 256 MiB | 默认值通常无需调整 |
| 桶最大 key 数 | `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT` | 500 | 500 | 默认值通常无需调整 |

##### 心跳间隔：卸载/提升的响应速度

| 参数 | 环境变量 | 默认值 | 推荐值 | 说明 |
|------|---------|--------|--------|------|
| 心跳间隔 | `MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS` | 10 | 5-10 | 更短 = 更快响应，但 CPU/网络开销更大 |

心跳间隔决定了 Master 的卸载/提升决策多久生效一次。如果 DRAM 经常溢出，缩短到 5 秒可以更快触发卸载。

##### SPDK NVMe-oF：远程 SSD 专线

| 参数 | 环境变量 | 默认值 | 推荐值 | 说明 |
|------|---------|--------|--------|------|
| SPDK 工作线程 | `MC_NOF_WORKERS` | 4 | 匹配 CPU 核数 | 增大提高并发 I/O |
| 提交块大小 | `MC_NOF_SUBMIT_CHUNK_BYTES` | 128 KiB | 256 KiB | 增大提高顺序吞吐 |
| 在途字节上限 | `MC_NOF_INFLIGHT_BYTES_LIMIT` | 32 MiB | 64 MiB | 控制并发 IO 深度 |

---

### 闸门 5：Store 水位与驱逐——内存的"泄洪"策略

DRAM 内存有限，必须通过驱逐释放空间。驱逐策略决定了"什么时候泄洪、泄多少"。

##### 水位线：什么时候开始泄洪

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `eviction_high_watermark_ratio` | 0.95 | 0.85-0.90 | 高水位——超过此值触发强制驱逐 |
| `eviction_ratio` | 0.05 | 0.05-0.10 | 每次驱逐比例——驱逐到高水位以下多少 |

```
默认: 高水位 95%，驱逐 5% → 内存使用在 90%-95% 之间波动
调优: 高水位 85%，驱逐 10% → 内存使用在 75%-85% 之间波动，更安全

风险: 高水位设太低（如 0.70），内存利用率低，浪费 DRAM 容量
风险: 高水位设太高（如 0.98），驱逐来不及，可能 OOM
```

##### 卸载与提升联动

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `offload_on_evict` | false | **true** | 驱逐前先卸载到 SSD——数据不丢 |
| `promotion_on_hit` | false | **true** | SSD 命中时异步提升回 DRAM |
| `promotion_admission_threshold` | 2 | 2-3 | 访问几次才值得提升——防冷数据反复搬运 |
| `promotion_max_per_heartbeat` | 1 | 3-5 | 每次心跳最大提升任务数——默认太保守 |

> 笔者注：`offload_on_evict=true` 和 `promotion_on_hit=true` 是分层存储的"最低配置"。不开这两个参数，分层存储就只是"只降不升"的半成品——数据只能从 DRAM 搬到 SSD，却搬不回来。

---

### 场景化调优配方

不同场景的瓶颈不同，调优重点也不同。以下是三个典型场景的推荐配置。

##### 场景 1：PD 解耦推理——追求最低 KV 传输延迟

瓶颈在 P→D 的 RDMA 传输延迟。

```bash
# Transfer Engine: 低延迟优先
export MC_SLICE_SIZE=32768           # 更小切片，更快完成
export MC_MAX_INLINE=128             # 更多消息 inline
export MC_IB_PCI_RELAXED_ORDERING=1  # 降低 PCIe 读延迟

# 拓扑: 必须开 NIC 亲和
export MC_ENABLE_HCA_PEER_AFFINITY=true

# vLLM Connector: 增加发送并发
export VLLM_MOONCAKE_SENDER_WORKERS=16   # 默认 10，增大并发
export VLLM_MOONCAKE_PROTOCOL=rdma       # 必须用 RDMA，不要用 TCP

# Store: 内存充足，减少驱逐干扰
eviction_high_watermark_ratio=0.85
offload_on_evict=true
```

##### 场景 2：大规模 MoE 推理——追求最大 All-to-All 吞吐

瓶颈在 EP 的 All-to-All 集合通信带宽。

```bash
# Transfer Engine: 高吞吐优先
export MC_SLICE_SIZE=262144          # 更大切片，减少开销
export MC_NUM_QP_PER_EP=4           # 多 QP 并行
export MC_MAX_WR=512                 # 加深 WR 队列
export MC_MAX_CQE_PER_CTX=8192      # 扩大 CQ
export MC_WORKERS_PER_CTX=4         # 更多轮询线程

# EP: 设备过滤 + QoS
export MOONCAKE_EP_DEVICE_FILTER="mlx5_0,mlx5_1"  # 只用指定网卡
export MC_IB_TC=32                  # 推理流量优先级
export MOONCAKE_EP_ACTIVE_QPS_PER_RANK=8          # 每个 Rank 的活跃 QP

# PG: 预留足够空间给替补
maxWorldSize = world_size * 1.5
```

##### 场景 3：分层存储重度使用——DRAM 不足，SSD 补

瓶颈在 SSD 读写速度和 DRAM 驱逐频率。

```bash
# SSD: 必须开 io_uring
export MOONCAKE_OFFLOAD_USE_URING=true
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru  # LRU 保留热数据
export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=5  # 更快响应

# Store: 联动卸载与提升
offload_on_evict=true              # 驱逐前先卸载
promotion_on_hit=true              # 热数据自动提升
promotion_admission_threshold=3    # 访问 3 次才提升
promotion_max_per_heartbeat=5      # 每次心跳多提升几个

# 内存: 大页加速
export MC_STORE_USE_HUGEPAGE=true
export MC_STORE_HUGEPAGE_SIZE=2097152

# 水位: 更早驱逐，避免 OOM
eviction_high_watermark_ratio=0.85
eviction_ratio=0.10
```

---

### 调优检查清单：从高到低的优先级

按影响程度排序，**从上往下逐项检查**——上面的没调好，调下面的意义不大：

| 优先级 | 检查项 | 怎么确认 | 怎么调 |
|--------|--------|---------|--------|
| **P0** | 协议是否用了 RDMA | 检查 `MOONCAKE_PROTOCOL` 或 `VLLM_MOONCAKE_PROTOCOL` | 必须设为 `rdma`，TCP 带宽差 5-10x |
| **P0** | NIC 亲和是否开启 | 检查 `MC_ENABLE_HCA_PEER_AFFINITY` | 设为 `true`，避免跨 NUMA |
| **P0** | 以太网 MTU 是否 9000 | `ip link show` 检查 mtu 值 | RoCE 必须开 Jumbo Frame，避免 IP 分片 |
| **P0** | io_uring 是否开启 | 检查 `MOONCAKE_OFFLOAD_USE_URING` | 设为 `true`，SSD 吞吐提升 30-50% |
| **P1** | 大页是否启用 | 检查 `MC_STORE_USE_HUGEPAGE` | 设为 `true`，加速 MR 注册 |
| **P1** | 分层存储联动是否开启 | 检查 `offload_on_evict` 和 `promotion_on_hit` | 都设为 `true` |
| **P1** | 切片大小是否匹配场景 | 检查 `MC_SLICE_SIZE` | 高吞吐 256KB，低延迟 32KB |
| **P2** | QP 深度是否足够 | 跑 benchmark 看 CQ overflow 错误 | 增大 `MC_MAX_WR` 和 `MC_MAX_CQE_PER_CTX` |
| **P2** | QoS 是否配置 | 检查 `MC_IB_TC` | 与训练流量隔离 |
| **P2** | 水位线是否合理 | 看 Master 日志的驱逐频率 | 高水位 0.85-0.90 |
| **P3** | SPDK 参数是否调优 | NVMe-oF 场景看 IOPS | 增大 `MC_NOF_WORKERS` 和提交块大小 |
| **P3** | PCI Relaxed Ordering | 检查硬件是否支持 | `MC_IB_PCI_RELAXED_ORDERING=2` (auto) |

---

### 常见问题与排查

##### 问题 1：RDMA 连接建立失败

```
症状: 日志出现 "Failed to connect to remote endpoint" 或 "QP state transition failed"

排查:
1. 检查 GID 索引: export MC_GID_INDEX=3  (RoCEv2 常用值)
2. 检查 IB 端口: export MC_IB_PORT=1  (双口卡确认用哪个口)
3. 检查握手端口是否可达: telnet <remote_ip> 12001
4. 检查防火墙: RDMA 流量是否被阻断
```

##### 问题 2：传输带宽远低于预期

```
症状: RDMA 带宽只有 10-20 Gbps，而网卡是 200 Gbps

排查:
1. 确认协议: echo $MOONCAKE_PROTOCOL  → 必须是 rdma 不是 tcp
2. 确认 NIC 亲和: echo $MC_ENABLE_HCA_PEER_AFFINITY  → 必须是 true
3. 确认 MTU: echo $MC_MTU  → 必须是 4096
4. 跑 ib_write_bw 基准: 确认裸 RDMA 带宽是否正常
5. 检查网卡协商速率: ethtool mlx5_0 | grep Speed
```

##### 问题 3：内存频繁驱逐，SSD 读写压力大

```
症状: Master 日志频繁出现 eviction，SSD 读写延迟高

排查:
1. 检查全局段大小: echo $MOONCAKE_GLOBAL_SEGMENT_SIZE  → 是否太小
2. 检查水位线: eviction_high_watermark_ratio  → 是否太高 (0.95)
3. 检查 offload_on_evict: 是否为 true  → 否则驱逐直接丢数据
4. 检查 io_uring: echo $MOONCAKE_OFFLOAD_USE_URING  → 必须为 true
5. 检查大页: echo $MC_STORE_USE_HUGEPAGE  → 开启加速内存操作
```

##### 问题 4：EP All-to-All 通信延迟高

```
症状: MoE 推理的 dispatch/combine 阶段延迟高

排查:
1. 检查设备过滤: echo $MOONCAKE_EP_DEVICE_FILTER  → 是否选了正确的网卡
2. 检查 QP 数: echo $MOONCAKE_EP_ACTIVE_QPS_PER_RANK  → 增大到 8
3. 检查 NUMA 亲和: nvidia-smi topo -m  → 确认 GPU-NIC 拓扑
4. 检查 QoS: echo $MC_IB_TC  → 是否与训练流量隔离
```

---

### 设计哲学：调优的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **先量后调** | 先用 benchmark 定位瓶颈，再对症下药 | 不盲目调参，先跑 TE benchmark 看带宽 |
| **一次一个** | 每次只调一个参数，确认效果后再调下一个 | 同时改 5 个参数，出了问题不知道哪个导致的 |
| **够用就好** | 不追求极致参数，稳定比极限更重要 | QP 深度 512 够用就不上 1024，资源留给其他进程 |

---

### 总结与行动指南

| 调优层级 | 关键参数 | 一句话建议 |
|---------|---------|----------|
| P0 必须调 | 协议=RDMA, NIC 亲和, io_uring | 不调这三个，其他全调了也白搭 |
| P1 强烈建议 | 大页, 分层联动, 切片大小 | 性能提升 20-50% 的低垂果实 |
| P2 按需调 | QP 深度, CQ 容量, QoS, 水位线 | 高并发/大规模场景必须调 |
| P3 锦上添花 | SPDK 参数, PCI Relaxed Ordering | 极致优化，边际收益递减 |

**建议**: 新部署的 Mooncake 集群，先设 `MOONCAKE_PROTOCOL=rdma` + `MC_ENABLE_HCA_PEER_AFFINITY=true` + `MOONCAKE_OFFLOAD_USE_URING=true`，这三步做完，已经解决了 80% 的性能问题。

**延伸阅读**：
- Mooncake 配置文档：https://kvcache-ai.github.io/Mooncake/design/transfer-engine.html
- RDMA 性能调优指南：https://docs.nvidia.com/networking/display/ofedv230/Performance+Tuning
- Linux io_uring：https://kernel.dk/io_uring.pdf

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
