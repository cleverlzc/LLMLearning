# Mooncake Expert Parallel 专家并行详解：当 MoE 模型的专家们需要"跨车间协作"

> **系列**: Mooncake 技术博客系列 | **类型**: 核心模块深潜篇
>
> 想象一座大型工厂有 256 台专用机床（Expert），每台机床固定在地面上搬不走。每个产品（Token）需要 8 台机床加工，问题是——这 8 台机床可能分布在不同车间、不同厂房。产品必须自己跑到机床那里加工（Dispatch），加工完还要跑回原产线装配（Combine），一来一回跑两趟。更扎心的是，路上跑的时间可能比机床加工的时间还长。Mooncake EP 就是那套"物流调度系统"，负责把产品精确地送到对应的机床，再安全地收回来。

---

### 引言

MoE（Mixture of Experts）模型是当前大模型的主流架构——DeepSeek-V3 有 256 个专家，每个 Token 只激活其中 8 个。专家并行（Expert Parallelism）下，每个专家的权重固定在某个 GPU 的显存里，不可能"出差"到别的 GPU。Token 的数据副本必须被发送到专家所在的 GPU 上加工（Dispatch），加工完结果再发回原始 GPU 装配（Combine）——**一来一回，两次 All-to-All 通信**。

这个"送过去 → 加工 → 收回来"的过程就是 Mooncake EP 的核心职责。它遵循 DeepEP 的低延迟编程模型，并在此基础上增加了**故障容忍**能力——当某个 GPU 故障时，系统可以自动绕过故障节点继续服务。

今天这篇文章，我们深入 Mooncake EP 的 dispatch/combine 机制和弹性专家并行设计。

---

### MoE 推理的三阶段流程

```
本地 Token x + topk_idx
        │
        ▼
   ┌─ Dispatch ──────────────► 打包的本地专家输入
   │    (Token 发送到专家所在 Rank)     │
   │                                    │
   │                              ┌─────▼──────┐
   │                              │ 本地专家计算  │
   │                              │ (MoE Forward)│
   │                              └─────┬──────┘
   │                                    │
   └─ Combine ◄────────────── 打包的专家输出
        (专家输出合并回 Token 所有者)
        │
        ▼
   合并后的本地 Token 输出
```

| 阶段 | 操作 | 通信模式 |
|------|------|---------|
| Dispatch | Token 按路由表发送到专家所在 Rank | All-to-All (Token → Expert) |
| Expert Compute | 本地专家对打包输入执行前向传播 | 本地计算 |
| Combine | 专家输出按路由权重合并回 Token 所有者 | All-to-All (Expert → Token) |

> 笔者注：Rank 是分布式系统中每个进程的全局唯一编号，EP 通信用 Rank 定位"专家住在谁家"。它来自 MPI（Message Passing Interface）和 PyTorch Distributed 的术语体系。torch.distributed.init_process_group() 初始化后，每个进程通过 torch.distributed.get_rank() 获取自己的编号。实际系统中，Rank会对应到GPU，但不绑定物理GPU，比如两台机器都有GPU 0，Rank全局唯一的逻辑编号，第二台GPU 0的对应Rank 8（每台机器8卡）。

---

### 软件概念与物理 GPU 的对应关系

前面一直说"Rank"、"Dispatch"、"Combine"，但这些软件概念到底对应物理世界中的什么？我们用一个具体的部署例子把映射关系讲清楚。

##### 部署场景：2 台服务器，每台 8 张 GPU，256 个专家

```
┌─────────────────── 服务器 A (Node 0) ───────────────────┐
│                                                          │
│  GPU 0          GPU 1          GPU 2          GPU 3      │
│  local_rank=0   local_rank=1   local_rank=2   local_rank=3│
│  Rank 0         Rank 1         Rank 2         Rank 3     │
│  Expert 0-31    Expert 32-63   Expert 64-95   Expert 96-127│
│                                                          │
│  GPU 4          GPU 5          GPU 6          GPU 7      │
│  local_rank=4   local_rank=5   local_rank=6   local_rank=7│
│  Rank 4         Rank 5         Rank 6         Rank 7     │
│  Expert 128-159 Expert 160-191 Expert 192-223 Expert 224-255│
│                                                          │
│  GPU 0-7 之间全部 NVLink 互通 (全连接)                    │
└──────────────────────────────────────────────────────────┘
                        ↕
                InfiniBand RDMA
                (跨节点通信)
                        ↕
┌─────────────────── 服务器 B (Node 1) ───────────────────┐
│                                                          │
│  GPU 0          GPU 1          GPU 2          GPU 3      │
│  local_rank=0   local_rank=1   local_rank=2   local_rank=3│
│  Rank 8         Rank 9         Rank 10        Rank 11    │
│  Expert 256-287 Expert 288-319 Expert 320-351 Expert 352-383│
│                                                          │
│  GPU 4          GPU 5          GPU 6          GPU 7      │
│  local_rank=4   local_rank=5   local_rank=6   local_rank=7│
│  Rank 12        Rank 13        Rank 14        Rank 15    │
│  Expert 384-415 Expert 416-447 Expert 448-479 Expert 480-511│
│                                                          │
│  GPU 0-7 之间全部 NVLink 互通 (全连接)                    │
└──────────────────────────────────────────────────────────┘
```

> 笔者注：这里用 512 个专家（2×8=16 Rank，每 Rank 32 专家）举例。如果只有 256 个专家，则每 Rank 分到 16 个专家——映射逻辑相同，只是专家数不同。另外注意：这里的"服务器"是**物理服务器**（8 卡），不是 NUMA 节点（4 卡）。虽然 8 卡服务器内部有 2 个 NUMA 节点，但 NVLink 是 8 卡全连接（HGX H100 的 NVLink 拓扑是 8×8 mesh），GPU 0 可以直接 NVLink 访问 GPU 7，不受 NUMA 边界限制。Mooncake EP 的 `discover_topology()` 用 `cudaGetDeviceCount()` 检测整台物理服务器的 GPU 数量，`P2pTransport` 基于 CUDA IPC（同一操作系统内的进程间通信），这些都以物理服务器为边界。NUMA 亲和性影响的是 CPU 侧内存访问延迟，不影响 GPU 间 NVLink 互通。

##### 三层编号体系

| 编号 | 含义 | 范围 | 谁分配 |
|------|------|------|--------|
| **GPU ID** | 物理设备编号 | 机器内唯一 (0-7) | CUDA 驱动 |
| **local_rank** | 节点内进程编号 | 机器内唯一 (0-7) | torchrun |
| **Rank** | 全局进程编号 | 集群内唯一 (0-15) | PyTorch Distributed |

三者的关系：

```
Rank = scaleout_rank_idx × num_scaleup_ranks + scaleup_rank_idx
     = 节点编号 × 每节点GPU数 + 节点内GPU编号
```

以 Rank 13 为例：
- `scaleout_rank_idx = 1`（服务器 B）
- `scaleup_rank_idx = 5`（服务器 B 的 GPU 5）
- `Rank = 1 × 8 + 5 = 13`

##### Mooncake EP 的拓扑发现

Mooncake EP 在初始化时自动发现物理拓扑（`discover_topology()`），把 Rank 映射到 NVLink/RDMA 两个通信域：

```cpp
// mooncake-ep/src/mooncake_ep_elastic_buffer.cpp
ElasticTopology MooncakeElasticBuffer::discover_topology(
    int rank, int num_ranks, bool allow_hybrid_mode) {
    // 读取环境变量或自动检测每节点 GPU 数
    int num_local_ranks = getenv_int("MOONCAKE_EP_NUM_LOCAL_RANKS", ...);

    ElasticTopology topology;
    topology.num_rdma_ranks = ceil_div(num_ranks, num_local_ranks);  // 跨节点 Rank 数
    topology.num_nvlink_ranks = num_local_ranks;                      // 节点内 Rank 数

    if (allow_hybrid_mode && num_rdma_ranks > 1) {
        // 混合模式：节点内 NVLink + 跨节点 RDMA
        topology.num_scaleout_ranks = num_rdma_ranks;   // ceil(总Rank数/每节点Rank数)
        topology.num_scaleup_ranks = num_nvlink_ranks;   // 8 (每服务器GPU数)
        topology.hybrid_enabled = true;
    } else {
        // 单节点或纯 RDMA 模式
        topology.num_scaleout_ranks = 1;
        topology.num_scaleup_ranks = num_ranks;
        topology.hybrid_enabled = false;
    }
    topology.scaleout_rank_idx = rank / num_scaleup_ranks;  // 我在第几个节点
    topology.scaleup_rank_idx = rank % num_scaleup_ranks;   // 我在节点内第几个GPU
}
```

以上面的 2×8 部署为例，Rank 13 看到的拓扑：

```
ElasticTopology for Rank 13:
  rank_idx = 13
  num_ranks = 16
  num_scaleout_ranks = 2     ← 2 台服务器
  num_scaleup_ranks = 8      ← 每台 8 张 GPU
  scaleout_rank_idx = 1      ← 我在服务器 B
  scaleup_rank_idx = 5       ← 我是服务器 B 的 GPU 5
  hybrid_enabled = true      ← 混合模式
```

##### 通信路径选择：NVLink 还是 RDMA？

Mooncake EP 根据目标 Rank 的拓扑位置，自动选择通信路径：

```
Token 从 Rank 13 发出:
  目标 Rank 14 (同节点, scaleout=1, scaleup=6)
    → NVLink/IPC 快速路径 (~160 GB/s)

  目标 Rank 1 (不同节点, scaleout=0, scaleup=1)
    → IBGDA/RDMA 快速路径 (~50 GB/s)
```

判断逻辑很简单：

```python
# 目标 Rank 和我在同一个 scaleout 组？
if target_scaleout == my_scaleout:
    → NVLink (节点内)
else:
    → RDMA (跨节点)
```

##### 一次 Dispatch 的完整物理路径

以 DeepSeek-V3 的一个 Token 为例（topk_idx=[2, 37, 130, 290, ...]），在 Rank 13 上发起 Dispatch：

```
Token 在 Rank 13 (服务器B/GPU5):
  Expert 2   → Rank 0  (服务器A/GPU0) → RDMA 跨节点传输
  Expert 37  → Rank 1  (服务器A/GPU1) → RDMA 跨节点传输
  Expert 130 → Rank 4  (服务器A/GPU4) → RDMA 跨节点传输
  Expert 290 → Rank 9  (服务器B/GPU1) → NVLink 节点内传输
  ...

物理路径:
  服务器B/GPU5 ─── RDMA ───→ 服务器A/GPU0  (Expert 2)
  服务器B/GPU5 ─── RDMA ───→ 服务器A/GPU1  (Expert 37)
  服务器B/GPU5 ─── RDMA ───→ 服务器A/GPU4  (Expert 130)
  服务器B/GPU5 ── NVLink ──→ 服务器B/GPU1  (Expert 290)
```

> 笔者注：Mooncake EP 的混合模式（hybrid_enabled）就是同时利用 NVLink 和 RDMA 两条路径。节点内的 Token 走 NVLink（快），跨节点的 Token 走 RDMA（慢但必要）。`connect()` 阶段交换的 `PeerMetadata` 同时包含 NVLink 的 IPC 句柄和 RDMA 的 QP 信息，运行时根据目标 Rank 自动选路——这一切对上层 dispatch/combine API 完全透明。

##### Buffer 在物理 GPU 上的内存布局

每个 Rank 的 Buffer 在 GPU 显存中分配，包含发送和接收两个方向的双缓冲：

```
Rank 13 的 GPU 显存:
┌──────────────────────────────────────────────────────────────┐
│ 模型权重 (Expert 416-447)                                     │
├──────────────────────────────────────────────────────────────┤
│ KV Cache / 工作空间                                           │
├──────────────────────────────────────────────────────────────┤
│ EP Buffer (由 ElasticBuffer 管理)                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ RDMA 信号区: send_signals[16] + recv_signals[16]         │ │
│ │ (每个远端 Rank 一个信号槽)                                 │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ RDMA 数据区: send_data + recv_data                       │ │
│ │ (Token 数据，最大 num_max_tokens_per_rank × hidden)       │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ NVLink/IPC 数据区: 同结构                                 │ │
│ ├──────────────────────────────────────────────────────────┤ │
│ │ CUDA 计数器 + 工作空间                                    │ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

Buffer 的双缓冲设计允许流水线：当 Buffer 0 在做 Combine 时，Buffer 1 已经可以开始下一轮 Dispatch——两套缓冲区交替使用，GPU 不用等通信完成。

##### 全景图：从 Token 到物理链路

```
软件层                    逻辑层                      物理层
─────────                ────────                    ────────

dispatch(x, topk_idx)    Token → Expert 映射          GPU 显存读写
     │                        │                          │
     ├── topk_idx=[2,37,130,290]  │                          │
     │   → Rank 0,1,4,9       │                          │
     │                        │                          │
     ├── Rank 0,1,4           │  scaleout≠me             │  RDMA WRITE
     │   → 跨节点             │  (不同服务器)             │  (InfiniBand)
     │                        │                          │
     └── Rank 9               │  scaleout==me            │  NVLink/IPC
         → 节点内             │  (同一服务器)             │  (GPU直连)
```

> 笔者注：理解 EP 的关键是建立"软件 Rank → 逻辑拓扑 → 物理链路"的三层映射。Rank 是软件概念，拓扑是 Mooncake 自动发现的逻辑结构，NVLink/RDMA 是物理通道。dispatch/combine API 只关心 Rank，不关心底层走 NVLink 还是 RDMA——`discover_topology()` + `connect()` 已经把映射关系建好了。

---

### Buffer 类：EP 的用户接口

`Buffer` 是 Mooncake EP 的用户面对象，封装了 dispatch/combine 的全部状态和操作。

```cpp
// mooncake-ep/include/elastic/mooncake_ep_elastic_buffer.h
class Buffer {
    // 从 torch.distributed 进程组构造
    Buffer(group, num_ep_buffer_bytes);
    
    // 连接：交换 peer 元数据（RDMA 信息、IPC 句柄、active_rank 掩码）
    connect();
    
    // Dispatch：将 Token 发送到专家所在 Rank
    dispatch(x, topk_idx, active_ranks, ...);
    
    // Combine：将专家输出合并回 Token 所有者
    combine(handle, ...);
    
    // 零拷贝 Combine 支持
    get_next_combine_buffer(handle);
};
```

##### Buffer 布局

Buffer 使用**双缓冲**设计——发送和接收缓冲区成对出现：

```
┌─────────────────────────────────────────────────┐
│                   Buffer 布局                    │
├─────────────────────────────────────────────────┤
│  RDMA 信号缓冲区 (send + recv)                   │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Send Signals │  │ Recv Signals │            │
│  └──────────────┘  └──────────────┘            │
├─────────────────────────────────────────────────┤
│  RDMA 数据缓冲区 (send + recv)                   │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Send Data    │  │ Recv Data    │            │
│  └──────────────┘  └──────────────┘            │
├─────────────────────────────────────────────────┤
│  CUDA 计数器缓冲区                                │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Counter      │  │ Data         │            │
│  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
```

---

### 三种执行模式

| 模式 | 传输方式 | 适用场景 |
|------|---------|---------|
| **IBGDA/RDMA 快速路径** | GPU 内存直接通过 RDMA 移动 | 跨节点，有 RDMA 硬件 |
| **P2P/IPC 快速路径** | NVLink/IPC 直接访问 | 节点内，GPU 间可互相访问 |
| **Python 回退** | 通过 Python 层转发 | 无 RDMA/NVLink 环境 |

##### 模式选择逻辑

```
connect() 交换元数据
    │
    ├── 检测 NVLink 可用性
    │   ├── 可用 → P2P/IPC 快速路径
    │   └── 不可用 → 继续
    │
    ├── 检测 RDMA 可用性
    │   ├── 可用 → IBGDA/RDMA 快速路径
    │   └── 不可用 → Python 回退
    │
    └── 混合模式：节点内走 IPC，跨节点走 RDMA
```

##### connect() 元数据交换

```cpp
// connect() 期间交换的信息
struct PeerMetadata {
    // RDMA 信息
    void* raddrs;           // 远程内存地址
    void* rkeys;            // 远程内存密钥
    void* qp_devctxs;       // QP 设备上下文
    
    // IPC 信息
    void* ipc_peer_ptrs;    // CUDA IPC 句柄
    
    // 状态信息
    int32_t* nvlink_available;  // NVLink 可用性
    int32_t* active_ranks;      // 活跃 Rank 掩码
};
```

---

### Dispatch 详解：Token 的"发货"

Dispatch 将本地 Token 按路由表（topk_idx）发送到对应的专家 Rank。

##### 核心函数签名

```cpp
// mooncake-ep/include/mooncake_ep_api.cuh
void dispatch(
    void* packed_recv_x,           // 接收缓冲区（打包的专家输入）
    float* packed_recv_x_scales,   // FP8 缩放因子
    int* packed_recv_src_info,     // 源信息（哪个 Rank 发来的）
    int64_t* packed_recv_layout_range, // 布局范围
    int* packed_recv_count,        // 每个 Rank 的接收计数
    int32_t* active_ranks,         // 活跃 Rank 掩码
    void* mxa_buffer,              // MXA 缓冲区
    // ... RDMA/IPC 缓冲区 ...
    const void* x,                 // 输入 Token
    const int64_t* topk_idx,       // 路由索引
    int* next_clean_buffer,        // 下一个可用缓冲区
    int num_tokens,                // Token 数量
    int hidden,                    // 隐藏维度
    int num_max_dispatch_tokens_per_rank,  // 每个 Rank 最大 dispatch Token 数
    int num_topk,                  // Top-K 值
    int num_experts,               // 专家数量
    int rank,                      // 当前 Rank
    int num_ranks,                 // 总 Rank 数
    bool use_fp8,                  // 是否使用 FP8
    void* workspace,               // 工作空间
    cudaStream_t stream,           // CUDA 流
    int64_t timeout_ticks,         // 超时 tick 数
    int phases,                    // 阶段数
    int active_qps_per_rank        // 每 Rank 的活跃 QP 数
);
```

##### Dispatch 流程

```
1. 根据 topk_idx 计算每个 Token 应发往哪些 Rank
2. 打包：按目标 Rank 分组，连续排列
3. 传输：
   - 跨节点：RDMA WRITE 到目标 Rank 的 recv 缓冲区
   - 节点内：NVLink/IPC 直接写入
4. 信号通知：发送完成信号
5. 接收端等待信号，然后处理打包数据
```

---

### Combine 详解：专家输出的"收货"

Combine 将专家的输出按路由权重合并回原始 Token 所有者。

##### 核心函数签名

```cpp
void combine(
    void* combined_x,              // 合并后的输出
    int32_t* active_ranks,         // 活跃 Rank 掩码
    // ... 缓冲区参数同 dispatch ...
    const void* x,                 // 专家输出
    const int64_t* topk_idx,       // 路由索引
    const float* topk_weights,     // 路由权重
    const int* src_info,           // 源信息（dispatch 时记录的）
    const int64_t* layout_range,   // 布局范围
    // ... 其他参数 ...
    bool zero_copy,                // 是否零拷贝
    // ...
);
```

##### 零拷贝 Combine

传统 Combine 需要两次拷贝：专家输出先写入中间缓冲区，再拷贝到最终位置。零拷贝 Combine 通过 `get_next_combine_buffer()` 直接获取目标地址，专家输出直接写入最终位置：

```python
# 传统方式
output = expert(input)           # 专家输出到临时缓冲区
combine(output, ...)             # GPU 显存拷贝到最终位置

# 零拷贝方式
buffer = get_next_combine_buffer()  # 直接获取最终位置
output = expert(input, out=buffer)  # 专家直接输出到最终位置
combine(output, ..., zero_copy=True)  # 无需 GPU 显存拷贝
```

> 笔者注：零拷贝省的是**GPU 显存的一次本地拷贝**，不是省通信。专家输出仍然需要通过 All-to-All 从专家 GPU 传回 Token 原始 GPU——这个 RDMA/NVLink 传输是省不掉的。零拷贝只是让传回来的数据直接写入最终输出位置，跳过中间缓冲区。

---

### 故障容忍：active_ranks 机制

这是 Mooncake EP 区别于 DeepEP 的关键创新。

##### active_ranks 张量

```cpp
int32_t* active_ranks;  // 形状: [num_ranks]
// active_ranks[i] = 1 表示 Rank i 活跃
// active_ranks[i] = 0 表示 Rank i 故障
```

##### 故障检测与处理

```
正常流程:
  Token → topk_idx=[2,5,7] → dispatch 到 Rank 2,5,7 → 专家计算 → combine

Rank 5 故障:
  active_ranks = [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1]  ← Rank 5 标记为 0
  Token → topk_idx=[2,5,7] → dispatch 跳过 Rank 5 → 专家计算 → combine 跳过 Rank 5
```

##### 超时检测

Dispatch 和 Combine 操作支持 `timeout_ticks` 参数。如果某个 Rank 在超时时间内未响应，内核会自动将其标记为非活跃：

```
dispatch(..., timeout_ticks=1000000)
    │
    ├── Rank 2: 信号在 500K ticks 到达 → 正常
    ├── Rank 5: 信号在 1000K ticks 仍未到达 → 标记为非活跃
    └── Rank 7: 信号在 600K ticks 到达 → 正常
```

> 笔者注：超时检测是在 CUDA Kernel 内部实现的，不需要 CPU 介入。这意味着故障检测不会阻塞其他 Rank 的正常执行——已就绪的 Rank 可以立即开始计算，不必等待超时的 Rank。

---

### FP8 支持

当 `use_fp8=True` 时，dispatch 返回 FP8 打包数据和 FP32 缩放因子：

```
发送端 (Token 所有者):
  BF16/FP16 Token → FP8 量化 → 发送

接收端 (专家所在 GPU):
  接收 FP8 数据 + FP32 scales → FP8 反量化 → BF16/FP16 → 专家计算
```

FP8 量化发生在 Dispatch 之前（发送端），反量化发生在 Dispatch 之后、专家计算之前（接收端）。通信量减半的效果仅作用于 Dispatch 阶段；Combine 阶段是否使用 FP8 取决于具体实现——有些场景下 Combine 数据量较小且对精度更敏感，会保持 BF16 传输。

---

### Phase ACK 机制

在多阶段 Dispatch/Combine 中，Phase ACK 机制确保各 Rank 的同步：

```cpp
// 标记当前阶段完成
void mark_phase_ack(void* mxa_buffer, ...);

// 等待其他 Rank 的阶段完成
void wait_phase_ack(int* ack_buffer, ..., int64_t timeout_ticks);

// 一步完成标记和等待
void mark_and_wait_phase_ack(...);
```

```
Rank 0: dispatch phase 1 → mark_phase_ack → wait_phase_ack → dispatch phase 2
Rank 1: dispatch phase 1 → mark_phase_ack → wait_phase_ack → dispatch phase 2
Rank 2: dispatch phase 1 → mark_phase_ack → wait_phase_ack → dispatch phase 2
                         ↑ 所有 Rank 都到达后才能进入 phase 2
```

---

### 设计哲学：Expert Parallel 的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **DeepEP 兼容** | API 与 DeepEP 低延迟模式一致 | Buffer 类的 dispatch/combine 接口 |
| **故障容忍** | 部分故障不阻塞整体 | active_ranks、timeout 检测 |
| **零拷贝优先** | 减少不必要的数据拷贝 | 零拷贝 Combine、FP8 传输 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| Dispatch | Token 按路由表发送到专家所在 Rank |
| Combine | 专家输出按权重合并回 Token 所有者 |
| Rank → 物理映射 | Rank = 节点编号 × 每节点GPU数 + 节点内编号，自动发现拓扑选路 |
| scaleup/scaleout | scaleup=节点内(NVLink)，scaleout=跨节点(RDMA)，混合模式自动选路 |
| Buffer | EP 的用户接口，封装双缓冲和 peer 元数据 |
| active_ranks | 故障容忍的关键——标记哪些 Rank 可用 |
| 零拷贝 Combine | 专家输出直接写入最终位置，无需中间拷贝 |
| FP8 | 传输时量化为 FP8，减少 50% 通信量 |

**建议**: 部署 MoE 模型时，先用单节点（8 Rank）验证 dispatch/combine 功能正确，再扩展到多节点——EP 的调试在大规模下非常困难，单节点验证是必要的的第一步。

**延伸阅读**：
- DeepEP 论文：https://arxiv.org/abs/2505.20181
- Mooncake EP 设计文档：https://kvcache-ai.github.io/Mooncake/design/mooncake-ep.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
