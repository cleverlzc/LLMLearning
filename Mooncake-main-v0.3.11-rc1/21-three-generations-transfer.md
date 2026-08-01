# 从 Mooncake 系统实现拆解数据传输三代演进：从 CPU 搬砖到 GPU 直连

> **系列**: Mooncake 技术博客系列 | **类型**: 核心技术详解篇

数据传输是 Mooncake 的核心关键——KV Cache 必须从 Prefill 节点搬到 Decode 节点，搬得越快，推理延迟越低。从 TCP 到 RDMA 到 GPUDirect，三代传输技术的核心差异只有一个：**CPU 参与得越少，传输越快**。

本文从探寻本质出发，追踪一个字节从源 GPU 到目标 GPU 的三条路径，并结合 Mooncake 源码，看 submitTransfer() 如何把请求变成 RDMA 网卡上的电信号。

三代数据传输技术
第一代： TCP
1. 应用程序将数据复制到内核缓冲区（os kernel buffer）：源应用程序将数据传递给操作系统内核。第一次CPU拷贝。
2. 内核（kernel）通过 TCP 协议栈发送：内核处理每个数据包：分段、校验和、确认。第二次 CPU 处理。
3. 远程内核接收并复制到应用程序：在接收端，内核解包数据并将其复制到目标应用程序的缓冲区中（app's buffer）。第三次CPU复制。

每个字节都经过CPU。应用程序将数据从源缓冲区复制 → CPU → 网络栈 → CPU → 目标缓冲区。速度较慢，但适用于所有场景。

第二代：RDMA（远程直接内存访问）
1. CPU寄存器完成一次传输，应用程序通知网卡NIC：“从地址X读取数据，发送至远程地址Y。”一次性设置。
2. 网卡直接从DRAM读取数据：通过DMA，网卡直接从内存中提取数据。无需CPU，也无需内核。
3. 远程网卡直接写入目标DRAM：远程网卡将数据直接放入目标内存。仅在传输完成时，目标CPU才会被唤醒。

网卡直接读写内存。CPU只需设置一次传输，之后由网卡处理所有操作。数据永远不会进入CPU流水线。

第三代： GPU直连RDMA技术（GPUDirect RDMA，简称GPUDirect）
1. NIC直接从GPU显存读取：无需复制到主机DRAM。NIC通过PCIe直接访问GPU内存。
2. 数据在网络中一跳传输：同为RDMA网络，但有效载荷直接来自GPU内存。
3. 远程网卡直接写入远程GPU显存：数据进入解码GPU的内存，准备就绪。DRAM复制步骤完全消失。

NIC直接从GPU内存读取数据，并直接写入远程GPU内存，甚至DRAM复制步骤也消失了。这是最快的路径。

---


### 引言：CPU 是最慢的搬运工

一个字节从 A 机器搬到 B 机器，本质上就是内存 → 网线 → 内存。问题在于：**谁来做这次搬运？**

```
第一代 TCP:    CPU 搬 → CPU 发 → CPU 收     每个字节过 3 次 CPU
第二代 RDMA:   CPU 说一声 → 网卡搬 → 网卡放  CPU 只参与 1 次设置
第三代 GPUDirect: CPU 说一声 → 网卡从 GPU 搬 → 网卡放到 GPU  CPU 完全不碰数据
```

三代技术的演进方向非常清晰：**把 CPU 从数据路径上赶出去**。

---

### 第一代：TCP——每个字节都过 CPU

```
源应用程序                源操作系统                  目标操作系统              目标应用程序
    │                        │                           │                       │
    │ ① write()              │                           │                       │
    │───→ 内核缓冲区          │                           │                       │
    │     (CPU 拷贝 1)        │                           │                       │
    │                        │ ② TCP 协议栈处理            │                       │
    │                        │   分段/校验和/确认           │                       │
    │                        │   (CPU 处理 2)             │                       │
    │                        │──── 网络发送 ────────────→ │                       │
    │                        │                           │ ③ 内核解包             │
    │                        │                           │   复制到应用缓冲区      │
    │                        │                           │   (CPU 拷贝 3)         │
    │                        │                           │────→ read()           │
```

**每个字节经过 CPU 三次**：应用→内核（拷贝 1）、TCP 协议栈处理（CPU 2）、内核→应用（拷贝 3）。

对于 KV Cache 传输，这意味着什么？

```
LLaMA-70B, 4K token, 160 MB KV Cache:
  TCP 传输 (100 Gbps 网卡, 假设 40% 协议开销):
    理论带宽: 100 Gbps × 60% = 12.5 GB/s
    传输时间: 160 MB / 12.5 GB/s ≈ 13 ms
    但 CPU 还要处理 3 次拷贝: 160 MB × 3 = 480 MB 的内存带宽占用

  如果是 128K token (5 GB):
    传输时间: 5 GB / 12.5 GB/s ≈ 400 ms
    CPU 拷贝: 15 GB 的内存带宽 → 严重挤占推理计算带宽
```

Mooncake 支持 TCP 传输（`TcpTransport`），但仅作为兼容方案——在没有 RDMA 网卡的环境中使用。

---

### 第二代：RDMA——网卡自己搬，CPU 只说一声

```
源应用程序                源网卡 NIC                 目标网卡 NIC              目标应用程序
    │                        │                           │                       │
    │ ① 提交传输请求          │                           │                       │
    │   (CPU 设置 1 次)       │                           │                       │
    │                        │ ② DMA 从 DRAM 读数据       │                       │
    │                        │   (CPU 不参与)             │                       │
    │                        │──── RDMA WRITE ──────────→ │                       │
    │                        │                           │ ③ DMA 写入目标 DRAM    │
    │                        │                           │   (目标 CPU 不参与)     │
    │                        │                           │                       │
    │   ← 完成通知            │                           │                       │
    │   (CPU 被唤醒)          │                           │                       │
```

**CPU 只参与一次**：提交传输请求时设置参数（源地址、目标地址、长度），之后网卡用 DMA 直接读写内存，CPU 完全不碰数据。

RDMA 传输需要四个参数——这就是打开远程内存的"万能钥匙"：

```
RDMA WRITE 四要素:
  source_addr  = 0x7f3a00000000   ← 本地内存地址 (数据从哪读)
  source_lkey  = 0x1a2b           ← 本地内存密钥 (允许网卡读这片内存)
  dest_addr    = 0x7fc100000000   ← 远程内存地址 (数据写到哪)
  dest_rkey    = 0x3c4d           ← 远程内存密钥 (允许网卡写那片内存)
  length       = 2097152          ← 传输字节数 (2 MB)
```

```
LLaMA-70B, 4K token, 160 MB KV Cache:
  RDMA 传输 (100 Gbps RoCEv2, 假设 90% 效率):
    有效带宽: 100 Gbps × 90% = 11.25 GB/s
    传输时间: 160 MB / 11.25 GB/s ≈ 14 ms
    CPU 开销: 几乎为零 (只提交请求)
    内存带宽: 0 (网卡 DMA 直接操作)
```

**但 RDMA 有一个问题**：数据还在 DRAM 里。如果数据在 GPU 显存中，必须先从 GPU 搬到 DRAM（PCIe D2H），再通过 RDMA 发出——多了一次"卸货再装车"。

---

### 第三代：GPUDirect RDMA——网卡直接从 GPU 搬

```
源应用程序                源网卡 NIC                 目标网卡 NIC              目标应用程序
    │                        │                           │                       │
    │ ① 提交传输请求          │                           │                       │
    │   (CPU 设置 1 次)       │                           │                       │
    │                        │ ② DMA 直接从 GPU 显存读    │                       │
    │                        │   (绕过 DRAM！)            │                       │
    │                        │──── RDMA WRITE ──────────→ │                       │
    │                        │                           │ ③ DMA 直接写入目标 GPU  │
    │                        │                           │   (绕过 DRAM！)        │
    │                        │                           │                       │
    │   ← 完成通知            │                           │                       │
```

**CPU 和 DRAM 都不碰数据**：网卡通过 PCIe 直接读写 GPU 显存，DRAM 完全被绕过。这就是 GPUDirect RDMA——数据从源 GPU 直达目标 GPU，中间没有任何"卸货再装车"。

```
对比三代技术的数据路径:

TCP:           GPU → DRAM → CPU → TCP栈 → 网线 → TCP栈 → CPU → DRAM → GPU
               │              │              │              │              │
               └── D2H ──────┘── 拷贝 ──────┘── 拷贝 ──────┘── H2D ──────┘

RDMA:          GPU → DRAM → ────── RDMA ────── → DRAM → GPU
               │          │                    │          │
               └── D2H ───┘                    └── H2D ───┘

GPUDirect:     GPU → ────── RDMA ────── → GPU
               │                          │
               └── 网卡直读/直写 ──────────┘
```

##### 三代技术的量化对比

| 指标 | TCP | RDMA | GPUDirect RDMA |
|------|-----|------|----------------|
| CPU 参与次数 | 3 次/字节 | 1 次设置 | 1 次设置 |
| 内存拷贝次数 | 3 次 | 0 次 | 0 次 |
| D2H/H2D 拷贝 | 无 (数据在 DRAM) | 2 次 (D2H + H2D) | 0 次 |
| 延迟 (2MB) | ~13 ms | ~14 ms + ~2ms D2H/H2D | ~14 ms |
| 延迟 (5GB) | ~400 ms | ~445 ms + ~400ms D2H/H2D | ~445 ms |
| CPU 开销 | 高 (协议栈处理) | 极低 | 极低 |
| 适用场景 | 兼容/无 RDMA | 有 RDMA 网卡 | RDMA + nvidia-peermem |

> 笔者注：RDMA 和 GPUDirect 的网络传输时间相同（都是 RDMA），差异在于 GPUDirect 省掉了 D2H 和 H2D 的 PCIe 拷贝。对于 2MB 的小请求差异不大（~2ms），但对于 5GB 的长上下文请求，省掉的 ~400ms 非常可观。

---

### Mooncake 如何实现 GPUDirect：GPU 内存注册

GPUDirect 的关键是让 RDMA 网卡能直接访问 GPU 显存。这需要**注册**——告诉网卡"这片 GPU 内存的地址和访问密钥"。

Mooncake 的 GPU 内存注册逻辑（`rdma_context.cpp` 第 346-525 行）：

```
注册 GPU 内存的两种路径:

路径 A: 安装了 nvidia-peermem 内核模块
  ibv_reg_mr(pd, gpu_addr, length, access)
  → nvidia-peermem 拦截注册请求，处理 GPU 页面故障
  → 网卡可以直接用 DMA 读写 GPU 显存
  → 返回 lkey + rkey

路径 B: 没有安装 peermem (使用 dmabuf)
  cudaPointerGetAttributes() → 获取 dmabuf fd 和偏移
  ibv_reg_dmabuf_mr(pd, gpu_addr, length, access, fd, offset)
  → 通过 DMA-BUF 机制注册
  → 返回 lkey + rkey
```

注册成功后，`lkey`（本地密钥）和 `rkey`（远程密钥）就是网卡访问这片 GPU 内存的凭证。`rkey` 通过元数据服务共享给远程节点，远程节点就能用 RDMA WRITE 直接把数据写到这片 GPU 内存。

---

### submitTransfer：从 API 到电信号

当 Mooncake 要把 KV Cache 从 Prefill 节点搬到 Decode 节点时，调用 `submitTransfer()`。这个函数把高层请求变成 RDMA 网卡上的电信号，经过五层传递：

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: TransferEngine (用户 API)                                  │
│  submitTransfer(batch_id, entries)                                   │
│  "我要搬数据"                                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Layer 2: MultiTransport (传输路由)                                   │
│  ① 检查批次容量 — "这个批次还有空间吗？"                                │
│  ② selectTransport() — 按段协议选传输方式 (RDMA/TCP/NVLink)           │
│  ③ 按传输方式分组 — 同一种传输的任务打包在一起                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Layer 3: RdmaTransport (RDMA 传输层)                                │
│  ① 把大请求切成 kBlockSize 的小片 (Slice)                             │
│  ② selectDevice() — 查拓扑矩阵选最优网卡                              │
│  ③ 为每个 Slice 设置 RDMA 参数: source_lkey, dest_addr, dest_rkey    │
│  ④ 按 RdmaContext 分组 — 同一个网卡的任务一起提交                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Layer 4: RdmaEndPoint (RDMA 端点)                                   │
│  ① 构建 SGE (Scatter/Gather Entry): addr, length, lkey              │
│  ② 构建 WR (Work Request): opcode=RDMA_WRITE, remote_addr, rkey     │
│  ③ 跨多个 QP 分发 — 多个 QP 并行传输                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Layer 5: ibv_post_send (内核/硬件)                                   │
│  网卡从源内存 DMA 读取 → 网线传输 → 网卡 DMA 写入目标内存              │
│  完成后 CQE (Completion Queue Entry) 通知 Worker 线程                 │
└─────────────────────────────────────────────────────────────────────┘
```

##### Layer 2：批次管理——容量检查与任务分组

```cpp
// mooncake-transfer-engine/src/multi_transport.cpp (第 115-154 行)
int MultiTransport::submitTransfer(BatchID batch_id,
                                    const std::vector<TransferRequest> &entries) {
    // 1. 安全检查: 批次还有空间吗？
    auto *batch_desc = toBatchDesc(batch_id);
    if (batch_desc->task_list.size() + entries.size() > batch_desc->batch_size)
        return -ENOMEM;  // 不会静默溢出

    // 2. 为每个请求选传输方式
    for (auto &entry : entries) {
        Transport *transport = selectTransport(entry);
        // 按段协议路由: "rdma" → RdmaTransport, "tcp" → TcpTransport
        submit_tasks[transport].push_back(entry);
    }

    // 3. 按传输方式分组提交
    for (auto &[transport, task_list] : submit_tasks) {
        transport->submitTransferTask(task_list);
    }
}
```

**BatchID 的巧妙设计**：`BatchID` 就是 `BatchDesc*` 的指针强转——没有哈希表查找，零开销：

```cpp
// mooncake-transfer-engine/include/transport/transport.h (第 102-104 行)
inline BatchDesc* toBatchDesc(BatchID id) {
    return reinterpret_cast<BatchDesc*>(id);  // 直接指针，无间接寻址
}
```

##### Layer 3：请求切片与拓扑选路

RDMA 传输不是一次搬完所有数据——大请求被切成固定大小的 Slice，每个 Slice 独立提交：

```
一个 2MB 的传输请求:
  kBlockSize = 256 KB (默认)
  → 切成 8 个 Slice
  → 每个 Slice 独立选择网卡和 QP
  → 并行传输，8 条 RDMA WRITE 同时飞
```

##### Layer 4：构建 RDMA Work Request

这是 CPU 最后一次参与——构建 `ibv_send_wr` 结构体，告诉网卡"从哪读、写到哪、多长"：

```cpp
// mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp (第 766-856 行)

// 1. 构建 SGE (Scatter/Gather Entry) — 描述本地内存
sge.addr = (uint64_t)slice->source_addr;    // 源地址 (可能在 GPU 显存中)
sge.length = slice->length;                  // 传输长度
sge.lkey = slice->rdma.source_lkey;          // 本地内存密钥

// 2. 构建 WR (Work Request) — 描述传输操作
wr.wr_id = (uint64_t)slice;                  // Slice 指针作回调 ID
wr.opcode = IBV_WR_RDMA_WRITE;               // RDMA WRITE 操作
wr.sg_list = &sge;
wr.send_flags = IBV_SEND_SIGNALED;           // 完成后通知
wr.wr.rdma.remote_addr = slice->rdma.dest_addr;  // 目标地址
wr.wr.rdma.rkey = slice->rdma.dest_rkey;         // 目标密钥

// 3. 提交到网卡
ibv_post_send(qp_list_[qp_index], wr_list.data(), &bad_wr);
// ↑ 从这一刻起，CPU 的工作结束了——网卡接管一切
```

**关键设计**：`wr.wr_id = (uint64_t)slice`——把 Slice 指针塞进 Work Request 的 ID 字段。当网卡完成传输，Worker 线程从完成队列 (CQ) 取出 CQE 时，用 `wr_id` 反查 Slice，标记为成功或失败。

##### 小结
当Mooncake想要将KV缓存从prefill阶段转移到decode阶段时，它会调用submitTransfer()函数。该函数将多个传输请求批量处理，并将其交给RDMA引擎。
- submitTransfer(batch_id, entries) - 入口点。有人想要移动数据。他们提供一个批次ID（该批次所属的批次）和一组转移请求（哪些数据要转移到哪里）。
- 安全检查 - “这个批次还有空间吗？”如果该批次已经有任务，而你添加的任务超过了它所能容纳的范围，它会拒绝并返回错误。不会静默溢出。
- 扩展任务列表 - 调整批次内部任务列表的大小以适应新的条目。可以将其想象为向气动管道队列中添加更多胶囊。
- 为每个请求创建TransferTask - 每个请求都会变成一个任务，该任务会记住：它属于哪个批次，以及需要转移什么数据。这些任务会被收集到一个列表中。
- submitTransferTask(task_list) - 真正的工作在这里进行。这个函数将任务列表交给RDMA硬件。CPU的工作到此结束——网卡将从这里接管。

---

### 拓扑感知选路：哪个网卡搬最快

一台 AI 服务器可能有 8 张 GPU、4 张 RDMA 网卡。GPU 和网卡之间的 PCIe 距离不同——**近的快，远的慢**。拓扑矩阵就是告诉传输引擎"对于每块内存，哪个网卡最快"。

```
一台 8-GPU 服务器的 PCIe 拓扑 (简化):

NUMA 0                          NUMA 1
┌──────────────────┐           ┌──────────────────┐
│  GPU 0  GPU 1    │           │  GPU 4  GPU 5    │
│    │      │      │           │    │      │      │
│  PCIe Switch     │           │  PCIe Switch     │
│    │      │      │           │    │      │      │
│  NIC 0  NIC 1    │           │  NIC 2  NIC 3    │
└──────────────────┘           └──────────────────┘

GPU 0 → NIC 0: 1 跳 PCIe (最快) → preferred_hca
GPU 0 → NIC 1: 2 跳 PCIe (较快) → preferred_hca
GPU 0 → NIC 2: 跨 NUMA (较慢)  → avail_hca (备用)
GPU 0 → NIC 3: 跨 NUMA (较慢)  → avail_hca (备用)
```

##### TopologyEntry 数据结构

```cpp
// mooncake-transfer-engine/include/topology.h (第 38-57 行)
struct TopologyEntry {
    std::string name;                    // 内存位置名 (如 "gpu0_vram")
    std::vector<std::string> preferred_hca;  // 首选网卡 (PCIe 跳数最少)
    std::vector<std::string> avail_hca;      // 备用网卡 (可用但较慢)
};

class TopologyMatrix {
    std::unordered_map<std::string, TopologyEntry> entries;
    // 查找表: "gpu0_vram" → {preferred: [nic0, nic1], avail: [nic2, nic3]}
};
```

##### 拓扑发现：自动探测 GPU 与网卡的 PCIe 距离

```cpp
// mooncake-transfer-engine/src/topology.cpp (第 533-593 行)
// GPU 拓扑发现流程:

1. cudaDeviceGetPCIBusId() → 获取每个 GPU 的 PCIe 插槽位置
2. getPciDistance(gpu_slot, hca_slot) → 计算 GPU 到每张网卡的 PCIe 距离
3. 按距离排序 → 距离最近的填入 preferred_hca，其余填入 avail_hca
4. 同 NUMA 优先 → 跨 NUMA 的网卡降级为备用
```

##### 选路逻辑：优先用最快的，坏了再用备用的

```
selectDevice() 选路流程:

1. 从 preferred_hca 中轮询选择
   → GPU 0 的数据，优先用 NIC 0 (1 跳 PCIe)
   → 轮询保证多张首选网卡负载均衡

2. 如果所有首选网卡都失败
   → 降级到 avail_hca
   → GPU 0 的数据，用 NIC 2 (跨 NUMA，较慢但可用)

3. retry_count 控制重试次数
   → 超过次数仍未成功 → 标记 Slice 为失败
```

##### 传输方式选择：没有自动降级

一个常见的误解是：RDMA 失败后自动降级到 TCP。**Mooncake 不做这种自动降级**——传输方式在初始化时确定，运行时不会切换：

```
初始化时:
  检测到 RDMA 网卡 → 安装 RdmaTransport
  TCP 启用 → 安装 TcpTransport (作为独立传输方式)
  CUDA 启用 → 安装 NvlinkTransport

运行时:
  selectTransport(entry) → 按段的 protocol 字段路由
  "rdma" → RdmaTransport
  "tcp"  → TcpTransport
  不会 "先试 RDMA，失败再试 TCP"
```

**但在 TENT（新版传输引擎）中**，有故障转移机制——如果 RDMA 传输失败，可以自动切换到 TCP。这是通过 `FaultProxyTransport` 装饰器实现的，详见本系列测试实践篇。


##### 小结
一台服务器可能配备多个HCA（网络卡）。拓扑矩阵会告诉传输引擎，对于每个内存位置，哪个网卡速度最快——以及哪些网卡可以作为备用。

- TopologyEntry：一个用于记录某个内存位置的条目。它包含一个名称（例如“gpu0_vram”）、一个首选网卡列表，以及一个可用但速度较慢的网卡列表。
- preferred_hca：该内存位置的最快网卡。如果GPU 0在物理上靠近NIC 0（PCIe跳数较少），则NIC 0会出现在此处。传输引擎会优先尝试这些网卡。
- avail_hca：备用网卡，可以访问该内存但路径较长。速度较慢，但在首选网卡繁忙或故障时仍能正常工作。
- TopologyMatrix：一个查找表：“对于每个内存位置，我应该使用哪些网卡？” 传输引擎在每次传输前都会参考此表，以选择最优路径。
- preferred_hca：首选网卡。到内存的最短PCIe路径。在所有设备均正常运行时使用。
- avail_hca：备用网卡。路径较长，速度稍慢，但仍可正常工作。当首选网卡出现拥塞或故障时，引擎会自动切换到这些备用网卡。

HCA：主机通道适配器（Host Channel Adapter）——服务器中支持RDMA功能的网络卡；不同的HCA可能连接到不同的网络路径，或具有不同的性能特征。

---

### 三代技术的 Mooncake 代码对应

| 技术代际 | Mooncake 传输方式 | 核心代码 | 数据路径 |
|---------|-----------------|---------|---------|
| 第一代 TCP | `TcpTransport` | `tcp_transport.cpp` | GPU → DRAM → CPU → TCP → CPU → DRAM → GPU |
| 第二代 RDMA | `RdmaTransport` | `rdma_transport.cpp` | GPU → DRAM → RDMA → DRAM → GPU |
| 第三代 GPUDirect | `RdmaTransport` + peermem | `rdma_context.cpp` GPU 内存注册 | GPU → RDMA → GPU |
| NVLink 直连 | `NvlinkTransport` | `nvlink_transport.cpp` | GPU → NVLink → GPU (同节点) |

##### NVLink Transport：同节点 GPU 直连

除了 GPUDirect RDMA（跨节点），Mooncake 还支持 NVLink 直连（同节点或跨节点）：

```
NVLink 传输机制:
  1. cudaIpcGetMemHandle() → 获取 GPU 内存的 IPC 句柄
  2. 通过元数据服务交换 IPC 句柄
  3. cudaIpcOpenMemHandle() → 远程进程打开 IPC 句柄
  4. cudaMemcpyAsync() → 通过 NVLink/PCIe 直传

或者 (CUDA 12.8+):
  1. cuMemExportToShareableHandle(FABRIC) → 导出 fabric 句柄
  2. cudaMemcpyBatchAsync() → 批量传输
```

NVLink 和 GPUDirect RDMA 的区别：

| | GPUDirect RDMA | NVLink Transport |
|--|---------------|-----------------|
| 适用场景 | 跨节点 (不同机器) | 同节点或 NVLink 直连 |
| 传输协议 | InfiniBand/RoCE | CUDA IPC / Fabric |
| 网卡参与 | 是 (RDMA WRITE) | 否 (CUDA 驱动处理) |
| QP/CQ | 需要 | 不需要 (用 CUDA Stream) |
| 带宽 | 取决于网卡 (100-400 Gbps) | 取决于 NVLink (300-900 GB/s) |

---

### 总结：三代传输的演进逻辑

```
TCP:    CPU 搬数据    → 通用但慢     → 兼容方案
RDMA:   网卡搬数据    → 快但数据在DRAM → 主力方案
GPUDirect: 网卡直接搬GPU数据 → 最快路径    → 极致方案

演进方向: CPU 参与越来越少 → 延迟越来越低 → 带宽越来越高
```

| 不变量 | 含义 |
|--------|------|
| **CPU 不碰数据** | 从 TCP 到 GPUDirect，CPU 从"搬数据"变成"说一声"，数据路径上没有 CPU |
| **拓扑决定性能** | GPU 和网卡的 PCIe 距离决定了传输速度，拓扑矩阵让引擎选最快的路 |
| **分片并行** | 大请求切成 Slice，跨多个 QP 并行传输，一条路堵了不影响其他路 |

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
