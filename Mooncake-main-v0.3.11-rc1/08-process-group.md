# Mooncake Process Group 后端：给 PyTorch 的"全员会议"装上缺席自动跳过和替补机制

> **系列**: Mooncake 技术博客系列 | **类型**: 核心模块深潜篇
>
> PyTorch 的 ProcessGroup 就像一个必须全员到齐才能开会的议事厅——AllReduce、Barrier 等集合操作要求所有成员同时参与，任何一个人缺席，整场会议就卡住了。Mooncake PG 给这个议事厅装上了两套机制：缺席者自动跳过（会议不卡），替补随时入座（不重启）。

---

### 引言

想象一个议事厅里有 8 位委员（GPU Rank），每次表决（AllReduce）必须全员到齐才能进行。传统规则下，任何一个委员缺席，整场表决就卡住——其他 7 位只能干等。

在大规模 MoE 推理中，这个问题尤为严重：256 个专家分布在数十张 GPU 上，任何一张 GPU 故障都会导致整个推理服务中断——就像一个委员堵车，全楼的人都没法开会。

Mooncake PG（Process Group）的解决方案：**缺席者自动跳过，表决照常进行；替补委员随时入座，无需重新组建议事厅**。

今天这篇文章，我们深入 Mooncake PG 的弹性恢复机制。

---

### PyTorch ProcessGroup 基础

##### 为什么需要 ProcessGroup？

单卡训练不需要通信——数据和计算都在一张 GPU 上完成。但一旦多卡协作，问题立刻出现：**梯度怎么同步？参数怎么广播？激活怎么交换？**

最直觉的做法是直接调用底层通信库：

```
不使用 ProcessGroup 时:

  GPU 0 → ncclAllReduce(梯度)    ← 直接调 NCCL
  GPU 1 → ncclAllReduce(梯度)

  换成 CPU 集群？→ 全部重写成 Gloo API
  换成 HPC 集群？→ 全部重写成 MPI API
  换成新硬件？  → 再重写一遍...
```

这就像每次换一个会议场地（NCCL 会议室 / Gloo 电话会议 / MPI 视频会议），就要重新制定一套会议规则——表决方式不同、签到方式不同、文件传递方式不同。**场地换了，规则就得重写**。

ProcessGroup 的解决方案：**定义一套统一的会议规则，不管你用什么场地**。

```
使用 ProcessGroup 后:

  dist.all_reduce(梯度)    ← 写一次，到处运行
       │
       ├── 后端=nccl   → 调用 NCCL
       ├── 后端=gloo   → 调用 Gloo
       ├── 后端=mpi    → 调用 MPI
       └── 后端=mooncake → 调用 Transfer Engine
```

用议事厅的比喻来说：**ProcessGroup 规定了"怎么开会"（AllReduce、Broadcast、Barrier……），后端决定了"在哪个房间开会"（NCCL、Gloo、Mooncake……）**。上层代码只关心议事规则，不关心会议室长什么样。

##### ProcessGroup 的核心抽象

ProcessGroup 将分布式通信抽象为三类操作：

| 操作类型 | 具体操作 | 议事厅比喻 |
|---------|---------|-----------|
| 集合通信 | AllReduce, Broadcast, AllGather, ReduceScatter, AllToAll | 全员表决——所有人同时参与，缺一不可 |
| 点对点通信 | Send, Recv | 两人私下递纸条——只涉及发送方和接收方 |
| 同步 | Barrier | 全员签到——所有人到齐才能进入下一项议程 |

其中**集合通信是 ProcessGroup 的核心**——也是它最严格的约束：所有 Rank 必须同时参与，任何一个缺席，操作就会卡住（死锁）。这个"全员到齐才能行动"的特性，正是 Mooncake PG 要解决的核心问题。

##### 常见后端

| 后端 | 通信库 | 适用场景 | 能否跳过缺席者？ |
|------|-------|---------|---------------|
| nccl | NCCL | NVIDIA GPU | 不能——全员必须参与 |
| gloo | Gloo | CPU | 不能——全员必须参与 |
| mpi | MPI | HPC 集群 | 不能——全员必须参与 |
| **mooncake** | Transfer Engine | 弹性推理 | **能——缺席者自动跳过** |

前三者（NCCL、Gloo、MPI）都遵循同一个假设：**所有 Rank 永远健康**。在训练场景下这个假设基本成立——训练任务可以重启，数据可以重放。但在推理服务中，这个假设不成立了——服务不能停，请求不能丢，GPU 故障必须就地恢复。

Mooncake PG 的存在意义正在于此：**给"全员必须到齐"的议事规则，装上"缺席自动跳过"的弹性机制**。

> 笔者注：NCCL 是 NVIDIA 的通信库，Gloo 是 Facebook 的通信库，Mooncake TE 是 Mooncake 自己的 Transfer Engine 通信库——它提供 RDMA 传输能力，是 Mooncake PG 底层的数据搬运工。

##### Rank：议事厅里的座次

在 ProcessGroup 中，每个参与通信的进程被称为一个 **Rank**，拥有唯一的编号（0, 1, 2, ...）。Rank 的总数称为 **world_size**。

```
world_size = 4 的 ProcessGroup:

  Rank 0 (GPU 0) ──┐
  Rank 1 (GPU 1) ──┤── 全员参与的议事厅
  Rank 2 (GPU 2) ──┤
  Rank 3 (GPU 3) ──┘

  AllReduce: Rank 0~3 全部参与，缺一不可
  Barrier:   Rank 0~3 全部到达，才能继续
```

在 MoE 推理中，一个 Rank 通常对应一张 GPU，承载若干个专家。256 个专家分布在 32 张 GPU 上，就是 world_size=32 的 ProcessGroup。

---

### MooncakeBackend：核心类

`MooncakeBackend` 继承自 `c10d::ProcessGroup`，注册为 `mooncake`（CUDA/MUSA）和 `mooncake-cpu`（CPU）两个后端。

##### 构造选项

```cpp
// mooncake-pg/include/mooncake_backend.h
struct MooncakeBackendOptions {
    at::Tensor activeRanks_;     // Rank 健康状态张量
    bool isExtension_ = false;   // 是否为加入的替补 Rank
    int maxWorldSize_ = -1;      // 预留的最大 world_size
};

// 构造示例
auto options = c10::make_intrusive<MooncakeBackendOptions>(
    activeRanks,    // [1,1,1,0,1]  ← Rank 3 故障
    false,          // 不是替补
    8               // 预留 8 个槽位
);
auto backend = c10d::ProcessGroupMooncake(params, options);
```

##### 支持的集合操作

| 操作 | 实现 |
|------|------|
| AllReduce | 支持 |
| Broadcast | 支持 |
| AllGather | 支持 |
| ReduceScatter | 支持 |
| AllToAll | 支持 |
| Barrier | 支持 |
| Reduce | 支持 |
| Gather | 支持 |
| Scatter | 支持 |
| Send/Recv | 通过 MooncakeP2PShim 支持 |

##### 静态配置

```cpp
// 设置主机 IP
MooncakeBackend::setHostIp("192.168.1.10");

// 设置设备过滤
MooncakeBackend::setDeviceFilter({"mlx5_0", "mlx5_1"});

// 注入外部 Transfer Engine
MooncakeBackend::setExternalEngine(&engine);
```

---

### 弹性恢复：核心创新

Mooncake PG 的核心创新是**弹性 Rank 恢复**——在部分 Rank 故障时继续服务，并允许替补 Rank 加入。

##### 核心概念

| 概念 | 说明 |
|------|------|
| `size` | 预留的 world_size（包含故障 Rank 的槽位） |
| `activeSize` | 当前活跃的 Rank 数 |
| `activeRanks_` | 张量标记每个 Rank 的健康状态 |
| `maxWorldSize_` | 预留的最大 Rank 数（为替补留空间） |
| `isExtension_` | 标记是否为替补 Rank |

##### 故障检测

```
初始状态:
  activeRanks_ = [1, 1, 1, 1, 1]  ← 5 个 Rank 全部健康
  size = 5, activeSize = 5

Rank 3 故障:
  activeRanks_ = [1, 1, 1, 0, 1]  ← Rank 3 标记为 0
  size = 5, activeSize = 4

集合操作自动跳过 Rank 3:
  AllReduce → 只在 Rank 0,1,2,4 上执行
```

##### 弹性恢复协议

```
阶段 1: 替补 Rank 初始化

  替补 Rank (Rank 3'):
    init_process_group(world_size=8, is_extension=True)
    │
    ├── 发布本地元数据（RDMA 信息、IPC 句柄等）
    │
    └── 等待恢复

阶段 2: 健康 Rank 恢复

  健康 Ranks:
    get_peer_state() → 发现 Rank 3' 已发布元数据
    │
    recover_ranks([3']) → 激活 Rank 3'
    │
    └── 集合通信现在包含 Rank 3'

阶段 3: 替补 Rank 加入

  Rank 3':
    join_group() 返回
    │
    └── 现在可以参与集合通信
```

##### ExtensionState 序列化

恢复过程中的状态需要序列化传输：

```cpp
struct ExtensionState {
    std::vector<bool> activeRanks;     // 活跃 Rank 列表
    std::vector<uint32_t> p2pEpochs;   // P2P 通信纪元
    int taskCount = -1;                 // 任务计数
};

// 序列化和反序列化
std::vector<uint8_t> serialize(const ExtensionState& state);
ExtensionState deserialize(const std::vector<uint8_t>& buffer);
```

---

### P2P 通信：MooncakeP2PShim

Mooncake PG 通过 `MooncakeP2PShim` 提供点对点通信：

```cpp
class MooncakeP2PShim final : public ::c10d::Backend {
    explicit MooncakeP2PShim(MooncakeBackend* owner);
    
    c10::intrusive_ptr<c10d::Work> send(tensors, dstRank, tag);
    c10::intrusive_ptr<c10d::Work> recv(tensors, srcRank, tag);
    c10::intrusive_ptr<c10d::Work> recvAnysource(tensors, tag);
    c10::intrusive_ptr<c10d::Work> barrier(opts);
};
```

`MooncakeP2PShim` 是 `MooncakeBackend` 的轻量代理——所有 P2P 操作委托给 `MooncakeBackend` 的底层实现。

---

### 与 EP 的协作

Mooncake PG 和 EP 紧密协作实现弹性 MoE 推理：

```
1. PG 检测到 Rank 故障 → 更新 activeRanks_
2. EP 从 PG 获取 activeRanks → dispatch/combine 跳过故障 Rank
3. PG 恢复替补 Rank → EP 调用 update_ep_member() 更新 peer 元数据
4. 恢复完成 → EP 和 PG 都包含新 Rank
```

```python
# SGLang 中的协作流程
backend = init_process_group("mooncake", options=...)

# 检测故障
peer_state = backend.get_peer_state([3])
# → {3: "failed"}

# 恢复替补 Rank
backend.recover_ranks([3])

# EP 更新
ep_buffer.update_ep_member()
```

---

### Transfer Engine 集成

Mooncake PG 内部使用 Transfer Engine 进行数据传输：

```cpp
// 内部创建 Transfer Engine
if (!engineInitialized_) {
    engine_ = new TransferEngine();
    engine_->init(metadata_conn_string, local_server_name, ...);
}

// 或注入外部 Transfer Engine
MooncakeBackend::setExternalEngine(&existing_engine);
```

外部注入的好处是**复用已有的 Transfer Engine 实例**——避免重复初始化和资源浪费，尤其在 EP 和 PG 共存的场景下。

---

### 设计哲学：Process Group 的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **故障不扩散** | 单个 Rank 故障不影响其他 Rank | activeRanks 张量、集合操作跳过非活跃 Rank |
| **弹性恢复** | 替补 Rank 可以动态加入 | isExtension 标记、两阶段恢复协议 |
| **PyTorch 原生** | 使用标准 PyTorch API | 注册为 c10d 后端，无需修改上层代码 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| MooncakeBackend | PyTorch ProcessGroup 后端，支持集合和 P2P 通信 |
| activeRanks | Rank 健康状态张量，集合操作自动跳过非活跃 Rank |
| 弹性恢复 | 替补 Rank 通过两阶段协议加入，不中断服务 |
| maxWorldSize | 预留最大 Rank 数，为替补留空间 |
| MooncakeP2PShim | P2P 通信的轻量代理 |
| EP 协作 | PG 检测故障 → EP 跳过故障 Rank → PG 恢复 → EP 更新 |

**建议**: 初始化 MooncakeBackend 时，`maxWorldSize` 至少设为 `world_size * 1.5`——为未来的替补 Rank 预留足够空间，避免恢复时需要重启整个服务。

**延伸阅读**：
- Mooncake PG 设计文档：https://kvcache-ai.github.io/Mooncake/design/mooncake-backend-pg.html
- PyTorch ProcessGroup 文档：https://pytorch.org/docs/stable/distributed.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
