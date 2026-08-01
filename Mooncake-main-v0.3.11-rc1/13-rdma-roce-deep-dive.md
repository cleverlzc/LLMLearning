# RDMA/RoCE 深度拆解：从网卡硬件到驱动协议，LLM 推理的"高速公路"是怎么修的

> 前面通过 LMCache、vLLM、Mooncake 三个系列的技术博客，反复提到 RDMA——KV Cache 的"瞬移"、PD 解耦的"高速通道"、EP 的"GPU 直传"，底层都离不开 RDMA。但 RDMA 到底是什么？RoCE 和 InfiniBand 有什么区别？QP 状态机怎么转？GPUDirect RDMA 为什么能让 GPU 显存直接被远端网卡访问？这些底层问题不搞清楚，部署调优就是盲人摸象。

本文从硬件到驱动、从协议到编程模型、从原理到 Mooncake/vLLM 的工程实践，一次性把 RDMA/RoCE 拆透。

---

### 一、RDMA 是什么：为什么 LLM 推理离不开它

##### 1.1 传统 TCP 传输的 CPU 瓶颈

```
传统 TCP 传输路径（40GB KV Cache 传输）：

发送方 DRAM ──► CPU 拷贝到内核缓冲区 ──► TCP/IP 协议栈 ──► NIC ──► 网络
                  ↑ 上下文切换              ↑ 中断处理
                  ↑ 数据拷贝               ↑ 协议封装

接收方 DRAM ◄── CPU 拷贝到用户缓冲区 ◄── TCP/IP 协议栈 ◄── NIC ◄── 网络
                  ↑ 数据拷贝               ↑ 中断处理
                  ↑ 上下文切换              ↑ 协议解封装
```

40GB 传输，CPU 需要处理数百万次中断和数据拷贝。在 LLM 推理场景下，CPU 本来就在做调度、Tokenization、采样等关键工作——RDMA 传输的 CPU 开销直接挤占了推理计算的资源。

##### 1.2 RDMA 传输路径

```
RDMA 传输路径：

发送方 DRAM ──► NIC ──► 网络 ──► NIC ──► 接收方 DRAM
                  ↑ CPU 只在建立连接时参与，传输过程零参与
```

RDMA 的三大核心优势：

| 优势 | 说明 | 对 LLM 推理的意义 |
|------|------|-----------------|
| **零拷贝** | 数据直接从应用内存到网卡，不经过内核缓冲区 | 减少 CPU 内存带宽占用 |
| **内核旁路** | 传输不需要系统调用，不需要上下文切换 | 消除中断和调度延迟 |
| **CPU 卸载** | 传输过程中 CPU 完全不参与 | CPU 可以专注推理调度 |

##### 1.3 实际性能对比

| 场景 | TCP 带宽 | RDMA 带宽 (4×200Gbps RoCE) | 提升 |
|------|---------|---------------------------|------|
| 40GB KV Cache 传输 | ~36 GB/s | 87 GB/s | 2.4x |
| 40GB KV Cache 传输 (8×400Gbps) | ~41 GB/s | 190 GB/s | 4.6x |

> 笔者注：RDMA 的性能优势不仅来自"更快地传数据"，更来自"不占用 CPU"——在 LLM 推理中，CPU 资源极其宝贵（调度、采样、Tokenization 都在 CPU 上），RDMA 让 CPU 从数据搬运中解放出来。

---

### 二、RDMA 协议族：InfiniBand、RoCE v1/v2、iWARP

RDMA 不是一种协议，而是一族协议。它们共享同一套传输语义（Verbs API），但底层封装不同。

##### 2.1 协议栈对比

```
InfiniBand 协议栈：
┌─────────────────────────┐
│    应用层 (Verbs API)    │
├─────────────────────────┤
│    IB 传输层 (BTH)       │  ← 所有 RDMA 协议共享
├─────────────────────────┤
│    IB 网络层 (GRH)       │
├─────────────────────────┤
│    IB 链路层 (LID)       │  ← InfiniBand 专用
├─────────────────────────┤
│    IB 物理层              │
└─────────────────────────┘

RoCE v1 协议栈：
┌─────────────────────────┐
│    应用层 (Verbs API)    │
├─────────────────────────┤
│    IB 传输层 (BTH)       │  ← 与 InfiniBand 相同
├─────────────────────────┤
│    IB 网络层 (GRH)       │
├─────────────────────────┤
│    Ethernet (Ethertype   │  ← 替换 IB 链路层
│     0x8915)              │
└─────────────────────────┘
  ⚠ 仅 L2，不可跨路由器

RoCE v2 协议栈：
┌─────────────────────────┐
│    应用层 (Verbs API)    │
├─────────────────────────┤
│    IB 传输层 (BTH)       │  ← 与 InfiniBand 相同
├─────────────────────────┤
│    UDP (端口 4791)       │  ← 替换 IB 网络层
├─────────────────────────┤
│    IP (IPv4/IPv6)        │  ← 可跨路由器！
├─────────────────────────┤
│    Ethernet              │
└─────────────────────────┘

iWARP 协议栈：
┌─────────────────────────┐
│    应用层 (Verbs API)    │
├─────────────────────────┤
│    RDMAP / DDP           │  ← 不同于 IB 传输层
├─────────────────────────┤
│    MPA (Marker PDU)      │
├─────────────────────────┤
│    TCP                   │  ← 基于 TCP，无需无损网络
├─────────────────────────┤
│    IP                    │
├─────────────────────────┤
│    Ethernet              │
└─────────────────────────┘
```

##### 2.2 四种协议详细对比

| 维度 | InfiniBand | RoCE v1 | RoCE v2 | iWARP |
|------|-----------|---------|---------|-------|
| 封装 | 原生 IB | IB over Ethernet | IB over UDP/IP | RDMA over TCP/IP |
| 路由 | L2 only (子网内) | L2 only (同广播域) | **L3 可路由** | L3 可路由 |
| 交换机 | 专用 IB 交换机 | 普通以太网交换机 | **标准以太网交换机** | 标准以太网交换机 |
| 无损网络 | 天然无损 | 需 PFC | 需 PFC + ECN | 不需要（TCP 保证） |
| 带宽 | HDR 200G / NDR 400G | 取决于以太网 | 取决于以太网 | 取决于以太网 |
| 延迟 | ~0.5μs | ~1.5μs | ~1.5μs | ~5μs |
| 成本 | 高（专用设备） | 中 | **中** | 低 |
| 生态 | HPC、AI 训练 | 已淘汰 | **AI 推理主流** | 少数场景 |

> 笔者注：RoCE v2 是当前 LLM 推理部署的主流选择——它既有 RDMA 的性能优势，又能跑在标准以太网基础设施上，还支持 IP 路由。InfiniBand 性能更好但成本高，iWARP 延迟高不适合推理场景。

##### 2.3 RoCE v2 报文格式

```
┌─────────────────────────────────────────────────────────┐
│ Ethernet Header (14B)                                    │
│   dst_mac (6B) │ src_mac (6B) │ ethertype=0x0800 (2B)  │
├─────────────────────────────────────────────────────────┤
│ IP Header (20B)                                          │
│   src_ip │ dst_ip │ DSCP (QoS) │ protocol=17 (UDP)     │
├─────────────────────────────────────────────────────────┤
│ UDP Header (8B)                                          │
│   src_port (随机) │ dst_port=4791 │ length │ checksum   │
├─────────────────────────────────────────────────────────┤
│ IB Transport Header (BTH, 12B+)                          │
│   opcode │ pkey │ dst_qp │ psn │ ...                    │
├─────────────────────────────────────────────────────────┤
│ Payload (RDMA 数据)                                      │
├─────────────────────────────────────────────────────────┤
│ ICRC (4B) + FCS (4B)                                    │
└─────────────────────────────────────────────────────────┘
```

关键细节：
- **UDP 端口 4791**：IANA 保留给 RoCE v2，交换机可据此识别 RoCE 流量
- **DSCP 字段**：用于交换机 QoS 分类，将 RoCE 流量映射到无损优先级队列
- **BTH (Base Transport Header)**：与 InfiniBand 完全相同——这就是 RoCE v2 "复用 IB 传输层"的体现

---

### 三、RoCE v2 的无损网络：PFC + ECN + DCQCN

RoCE v2 跑在以太网上，而以太网默认是"尽力传输"——丢包是常态。但 RDMA 协议不处理丢包重传（这是 IB 传输层的前提假设），所以必须通过**无损网络**技术保证不丢包。

##### 3.1 PFC（Priority Flow Control）：防止交换机丢包

```
发送端 NIC → 交换机 → 接收端 NIC
                │
                │ 缓冲区快满了！
                ▼
         发送 PAUSE 帧给上游
                │
                ▼
         上游暂停发送（仅暂停该优先级）
                │
                │ 缓冲区恢复
                ▼
         发送 RESUME 帧，恢复传输
```

| 概念 | 说明 |
|------|------|
| 优先级 | 以太网帧的 802.1p 优先级（0-7），RoCE 通常用优先级 3 |
| PFC PAUSE | 按优先级暂停上游发送，不影响其他优先级 |
| 风险 | PFC 死锁（暂停帧循环传播）、PFC 风暴 |

配置示例：
```bash
# 设置 DSCP 信任模式，将 DSCP 映射到优先级
mlnx_qos -i eth0 --trust dscp
# 将 DSCP 26 映射到优先级 3（RoCE 无损队列）
mlnx_qos -i eth0 --dscp2prio set 26 3
```

##### 3.2 ECN（Explicit Congestion Notification）：拥塞信号

PFC 防止丢包，但不能解决拥塞——如果只靠 PFC，暂停帧会像多米诺骨牌一样传播到整个网络。ECN 是更优雅的方案：交换机在队列延迟超过阈值时，在 IP 头的 CE（Congestion Experienced）字段打标记，而不是丢包或暂停。

```
发送端 → 交换机（队列延迟 > 阈值，标记 CE）→ 接收端
                                                   │
                                                   │ 收到 CE 标记的包
                                                   ▼
                                            发送 CNP（拥塞通知包）给发送端
                                                   │
                                                   ▼
                                            发送端降低发送速率
```

##### 3.3 DCQCN：RoCE v2 的拥塞控制算法

DCQCN（Data Center Quantized Congestion Notification）是 Mellanox 网卡的默认拥塞控制算法，结合了交换机 ECN 和端主机速率控制：

```
            速率
             │
    ┌────────┤
    │ 最大速率 │
    │        │
    │  ╱╲    │          ╱‾‾‾‾‾‾‾‾╲    ← 速率恢复阶段
    │ ╱  ╲   │         ╱          ╲
    │╱    ╲  │        ╱            ╲
    │      ╲ │   ... ╱              ╲
    │       ╲│      ╱                ╲
    │        ╲     ╱                  → 时间
    │         ╲   ╱
    │          ╲ ╱  ← 收到 ECN/CNP，指数降低
    │
    └──────────┴──────────────────────────► 时间
    
    收到 ECN → 指数降速    无 ECN → 线性恢复
```

| 阶段 | 行为 | 参数 |
|------|------|------|
| 速率降低 | 收到 ECN/CNP，指数降低发送速率 | `rp_min_dec_fac`（最小衰减因子） |
| 速率恢复 | 无 ECN 信号，线性恢复速率 | `rp_g_rate`（恢复增长率） |
| 周期探测 | 定期发送探测包，检测拥塞是否缓解 | `rp_t_rate`（探测间隔） |

> 笔者注：DCQCN 参数通常不需要手动调优——Mellanox 网卡固件有合理的默认值。但在大规模集群中，如果观察到 RoCE 带宽不稳定，可以尝试调整 `rp_min_dec_fac`（降低衰减幅度）和 `rp_g_rate`（加快恢复速度）。

---

### 四、硬件：网卡、交换机、线缆

##### 4.1 Mellanox ConnectX 网卡家族

| 网卡 | 带宽 | PCIe | 端口配置 | 关键特性 |
|------|------|------|---------|---------|
| ConnectX-5 | 100 Gb/s | Gen3/4 x16 | 1×100G / 2×50G | 首代 100G RDMA，VPI（IB+以太网双模） |
| ConnectX-6 Dx | 200 Gb/s | Gen4 x16 | 1×200G / 2×100G | 硬件 TLS/IPsec，增强 RoCE |
| ConnectX-6 Lx | 25/50 Gb/s | Gen4 x8 | 2×25G | 成本优化，云规模部署 |
| **ConnectX-7** | **400 Gb/s** | **Gen5 x16** | **1×400G / 2×200G** | **AI 推理主力，~215M msg/s** |
| ConnectX-8 | 800 Gb/s | Gen5 x16 | 2×400G | 下一代 AI 工作负载 |

> 笔者注：ConnectX-7 是当前 AI 推理部署的甜点选择——400G 带宽足以支撑 LLaMA3-70B 的 KV Cache 传输，且生态成熟。ConnectX-8 是未来方向，但当前驱动和固件支持尚在完善中。

##### 4.2 NVIDIA BlueField DPU

BlueField 是带 Arm 核心的智能网卡（DPU），可以卸载 CPU 的网络/存储/安全处理：

| 型号 | 带宽 | Arm 核心 | 网络引擎 | 用途 |
|------|------|---------|---------|------|
| BlueField-2 | 100 Gb/s | 8 A72 | ConnectX-6 Dx | 存储/安全卸载 |
| BlueField-3 | 400 Gb/s | 16-24 A78 | ConnectX-7 | GPU-Direct RDMA + NVMe-oF |
| BlueField-4 | 800 Gb/s | - | ConnectX-8 | 下一代 |

##### 4.3 交换机要求

RoCE v2 对交换机有三个硬性要求：

| 要求 | 说明 | 配置方法 |
|------|------|---------|
| PFC | 在 RoCE 优先级上启用无损 | `mlnx_qos --pfc` |
| ECN | 在 RoCE 队列上启用 CE 标记 | 交换机 CLI 配置 |
| DSCP 信任 | 信任网卡设置的 DSCP 值 | `mlnx_qos --trust dscp` |

推荐交换机：NVIDIA Spectrum-1/2/3/4（以太网）、Quantum-2（InfiniBand）。

##### 4.4 线缆类型

| 类型 | 最大距离 | 速率 | 适用场景 |
|------|---------|------|---------|
| DAC（直连铜缆） | 1-5m | 100G-800G | 同机架，延迟最低 |
| AOC（有源光缆） | 1-100m | 100G-800G | 机架间 |
| QSFP28 SR4 + 多模光纤 | 100m (OM4) | 100G | 数据中心楼内 |
| QSFP28 LR4 + 单模光纤 | 10km | 100G | 跨楼宇 |
| QSFP56 SR4 | 100m | 200G | 200GbE 楼内 |
| OSFP | 不等 | 400G | NDR/HDR400 |
| OSFP-DD | 不等 | 800G | 下一代 |

> 笔者注：同机架用 DAC（最便宜、延迟最低），跨机架用 AOC 或光纤。线缆选择对延迟影响很小（纳秒级差异），但对成本和部署灵活性影响很大。

---

### 五、驱动与软件栈：从内核到用户态

##### 5.1 驱动栈全景

```
┌────────────────────────────────────────────────────────────┐
│                    应用层                                    │
│  Mooncake Transfer Engine / vLLM / NCCL / 自定义 RDMA 应用   │
├────────────────────────────────────────────────────────────┤
│                    用户态库                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ libibverbs   │  │ librdmacm    │  │ mlx5dv (MOFED)   │ │
│  │ (核心 Verbs) │  │ (连接管理)    │  │ (硬件加速特性)    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
├─────────┼─────────────────┼───────────────────┼───────────┤
│         │    /dev/infiniband/uverbs0           │           │
│         │    /dev/infiniband/rdma_cm           │           │
│  ┌──────▼─────────────────▼───────────────────▼─────────┐ │
│  │                  ib_uverbs                            │ │
│  │            (用户态 Verbs 接口内核模块)                   │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                  rdma_cm                              │ │
│  │            (RDMA 连接管理内核模块)                      │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                  mlx5_ib                              │ │
│  │            (Mellanox IB/RoCE Verbs 提供者)             │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                  mlx5_core                            │ │
│  │            (Mellanox ConnectX PCI 设备驱动)            │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │                  ib_core                              │ │
│  │            (RDMA 核心中间层)                            │ │
│  └──────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────┤
│                    硬件                                      │
│              Mellanox ConnectX NIC                          │
└────────────────────────────────────────────────────────────┘
```

##### 5.2 MOFED vs rdma-core

| 维度 | MOFED (Mellanox OFED) | rdma-core (linux-rdma) |
|------|----------------------|----------------------|
| 来源 | NVIDIA 私有驱动栈 | Linux 上游开源 |
| 安装 | `MLNX_OFED_LINUX-*.run` | `apt install rdma-core` |
| 包含 | `mlx5_core` + `mlx5_ib` + `mlx5dv` + perftest | `libibverbs` + `librdmacm` + 基本工具 |
| 额外特性 | DevX、LAG 端口均衡、QP UDP 源端口多样化 | 无 |
| Mooncake 依赖 | 推荐（`USE_MLX5DV` 编译选项） | 可用，但缺少部分优化 |

> 笔者注：Mooncake 的 `USE_MLX5DV` 编译选项启用了 MOFED 专属的硬件加速特性——LAG 端口均衡和 QP UDP 源端口多样化。前者让多端口的网卡实现负载均衡，后者让交换机的 ECMP 能区分不同 QP 的流量。生产环境建议使用 MOFED。

##### 5.3 关键内核模块

```bash
# 查看已加载的 RDMA 内核模块
lsmod | grep -E "rdma|ib_|mlx5"

# 典型输出：
rdma_ucm     27022  0        # RDMA 用户态连接管理
rdma_cm      65212  1 rdma_ucm  # RDMA 连接管理
ib_cm        53085  2 rdma_cm  # IB 通信管理
mlx5_ib     384793  0        # Mellanox IB/RoCE Verbs 提供者
mlx5_core  1360822  1 mlx5_ib  # Mellanox PCI 设备驱动
ib_uverbs   132833  2 mlx5_ib  # 用户态 Verbs 接口
ib_core     357959  8 ...      # RDMA 核心中间层
```

##### 5.4 设备文件

| 文件 | 说明 |
|------|------|
| `/dev/infiniband/uverbs0` | 第一个 RDMA 设备的用户态 Verbs 访问入口 |
| `/dev/infiniband/rdma_cm` | RDMA 连接管理器套接字接口 |
| `/sys/class/infiniband/mlx5_0/` | sysfs 属性（GID 表、端口信息、NUMA 节点） |
| `/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/0` | GID 索引 0 关联的网络设备 |

##### 5.5 诊断工具速查

| 工具 | 用途 | 示例 |
|------|------|------|
| `ibv_devinfo` | 列出所有 RDMA 设备详细信息 | `ibv_devinfo -d mlx5_0` |
| `ibstat` | 查看 IB 设备状态 | `ibstat` |
| `ibv_devices` | 仅列出设备名 | `ibv_devices` |
| `ib_write_bw` | 测试 RDMA 写带宽 | `ib_write_bw -d mlx5_0` |
| `ib_read_bw` | 测试 RDMA 读带宽 | `ib_read_bw -d mlx5_0` |
| `ib_write_lat` | 测试 RDMA 写延迟 | `ib_write_lat -d mlx5_0` |
| `show_gids` | 显示 GID 表 | `show_gids` |
| `mlxconfig` | 网卡固件配置 | `mlxconfig -d /dev/mst/mt4123 query` |
| `mlnx_qos` | QoS/PFC/ECN 配置 | `mlnx_qos -i eth0 --trust dscp` |

---

### 六、RDMA 编程模型：Verbs API 深度拆解

RDMA 的编程接口叫 **Verbs API**，定义在 `libibverbs` 中。理解 Verbs，就理解了 RDMA 的运行机制。

##### 6.1 核心对象关系

```
┌──────────────────────────────────────────────────────────┐
│                      Protection Domain (PD)               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Memory Region (MR)    Memory Region (MR)          │  │
│  │  lkey: 本地访问密钥    lkey: 本地访问密钥            │  │
│  │  rkey: 远程访问密钥    rkey: 远程访问密钥            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │  Queue Pair (QP)  │  │  Queue Pair (QP)  │             │
│  │  Send Queue (SQ) │  │  Send Queue (SQ) │             │
│  │  Recv Queue (RQ) │  │  Recv Queue (RQ) │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
│           │                     │                        │
│  ┌────────▼─────────────────────▼────────┐              │
│  │      Completion Queue (CQ)            │              │
│  │  (通知 WR 完成事件)                     │              │
│  └───────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

| 对象 | 说明 | 类比 |
|------|------|------|
| PD (Protection Domain) | 资源隔离域，同一 PD 内的 QP/MR 可以互相访问 | 公司 |
| MR (Memory Region) | 注册的内存区域，拥有 lkey（本地密钥）和 rkey（远程密钥） | 登记过的仓库 |
| QP (Queue Pair) | 发送/接收队列对，RDMA 通信的端点 | 收发窗口 |
| CQ (Completion Queue) | 完成队列，通知 WR 的执行结果 | 签收单 |
| WR (Work Request) | 工作请求，提交到 QP 的操作 | 快递单 |

##### 6.2 QP 类型

| 类型 | 连接模式 | 可靠性 | Mooncake 是否使用 |
|------|---------|-------|-----------------|
| **RC (Reliable Connected)** | 1:1 | 保证可靠、有序 | **是（唯一使用）** |
| UC (Unreliable Connected) | 1:1 | 不重传、有序 | 否 |
| UD (Unreliable Datagram) | 1:多 | 不保证 | 否 |
| DCT (Dynamic Connected) | 按需 1:1 | 可靠 | 否 |
| XRC (Extended Reliable) | 共享 RC | 可靠 | 否 |

> 笔者注：Mooncake 只使用 RC QP——这是最简单也最可靠的 QP 类型。RC 的 1:1 连接意味着每个 Endpoint 需要为每个远端 NIC 维护独立的 QP，在大规模集群中 QP 数量可能成为瓶颈（见 Endpoint 管理部分）。

##### 6.3 QP 状态机（源码验证）

QP 的生命周期经历严格的状态转换，Mooncake 的实现位于 `rdma_endpoint.cpp`：

```
                    ┌─────────┐
                    │  RESET  │ ← 创建 QP 时的初始状态
                    └────┬────┘
                         │ ibv_modify_qp(INIT)
                         ▼
                    ┌─────────┐
                    │  INIT   │ 设置端口、PKey、访问权限
                    └────┬────┘
                         │ ibv_modify_qp(RTR)
                         │ 交换对端信息：GID, LID, QP号
                         ▼
               ┌──────────────┐
               │  RTR         │ Ready to Receive
               │ (可接收数据)  │
               └──────┬───────┘
                      │ ibv_modify_qp(RTS)
                      ▼
               ┌──────────────┐
               │  RTS         │ Ready to Send
               │ (可发送+接收) │ ← 正常工作状态
               └──────┬───────┘
                      │ ibv_modify_qp(ERR)
                      ▼
               ┌──────────────┐
               │  ERR         │ 错误状态，刷新所有在途 WR
               └──────┬───────┘
                      │ ibv_modify_qp(RESET)
                      ▼
                    ┌─────────┐
                    │  RESET  │ 可重新初始化
                    └─────────┘
```

Mooncake 源码中的关键参数：

```cpp
// RESET → INIT
attr.port_num = context_.portNum();
attr.pkey_index = globalConfig().pkey_index;
attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

// INIT → RTR
attr.path_mtu = context_.activeMTU();
attr.dest_qp_num = peer_qp_num;     // 对端 QP 号
attr.ah_attr.grh.dgid = peer_gid;   // 对端 GID
attr.ah_attr.grh.sgid_index = local_gid_index;  // 本地 GID 索引
attr.max_dest_rd_atomic = 16;        // 最大并发 RDMA Read

// RTR → RTS
attr.timeout = 14;                   // 重传超时 (~280μs)
attr.retry_cnt = 7;                  // 重试次数
attr.rnr_retry = 7;                  // RNR 重试次数
attr.max_rd_atomic = 16;             // 最大并发 RDMA Read
```

##### 6.4 内存注册（源码验证）

RDMA 要求所有被访问的内存必须先"注册"——这会锁定物理页面，并为网卡生成访问密钥。

Mooncake 注册 MR 的访问权限（`rdma_transport.cpp`）：

```cpp
const int kBaseAccessRights = IBV_ACCESS_LOCAL_WRITE |    // 本地可写
                              IBV_ACCESS_REMOTE_WRITE |   // 远程可写
                              IBV_ACCESS_REMOTE_READ;     // 远程可读
// 可选：IBV_ACCESS_RELAXED_ORDERING（PCIe 乱序优化）
```

**GPU 内存的注册**有三种路径（`rdma_context.cpp`）：

| 路径 | 条件 | 方法 |
|------|------|------|
| CPU 内存 | `CU_MEMORYTYPE_HOST` | `ibv_reg_mr(pd, addr, length, access)` |
| GPU 内存 + nvidia-peermem | `CU_MEMORYTYPE_DEVICE` + 模块已加载 | `ibv_reg_mr(pd, addr, length, access)` 直接注册 |
| GPU 内存 + dmabuf | `CU_MEMORYTYPE_DEVICE` + 无 peermem | `ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access)` |

```
GPU 内存注册路径选择：

cuPointerGetAttribute(&memType, addr)
    │
    ├── CU_MEMORYTYPE_HOST → 标准 ibv_reg_mr()
    │
    ├── CU_MEMORYTYPE_DEVICE
    │   │
    │   ├── WITH_NVIDIA_PEERMEM 环境变量？
    │   │   ├── 是 → ibv_reg_mr() 直接注册
    │   │   │        (nvidia-peermem 模块负责 pin GPU 页面)
    │   │   │
    │   │   └── 否 → cuMemGetHandleForAddressRange() 导出 dmabuf fd
    │   │            → ibv_reg_dmabuf_mr() 注册
    │   │            (内核 dmabuf 框架负责 pin GPU 页面)
    │   │
    │   └── cudaMallocManaged() → ⚠ 不适合 RDMA！
    │        (统一内存页面会在 CPU/GPU 间迁移，导致 dmabuf 失效)
```

> 笔者注：`nvidia-peermem` 是推荐路径——它通过 `ib_register_peer_memory_client()` 注册为 IB 核心的"伙伴内存客户端"，当 `ibv_reg_mr()` 被调用时自动 pin GPU 页面。`ibv_reg_dmabuf_mr` 是备选路径，需要较新的内核（5.15+）和 `CONFIG_PCI_P2PDMA=y`。

##### 6.5 工作请求提交（源码验证）

Mooncake 只使用**单边操作**——RDMA_READ 和 RDMA_WRITE，数据路径上不使用 SEND/RECV：

```cpp
// rdma_endpoint.cpp
wr.opcode = slice->opcode == TransferRequest::READ
                ? IBV_WR_RDMA_READ
                : IBV_WR_RDMA_WRITE;
wr.num_sge = 1;
wr.sg_list = &sge;
wr.send_flags = IBV_SEND_SIGNALED;  // 所有 WR 都信号完成
wr.wr.rdma.remote_addr = slice->rdma.dest_addr;  // 远端地址
wr.wr.rdma.rkey = slice->rdma.dest_rkey;          // 远端密钥
```

为什么只用单边操作？

| 操作类型 | 说明 | CPU 参与（接收端） | Mooncake 使用 |
|---------|------|-----------------|-------------|
| RDMA_WRITE | 写入远端内存 | **零参与** | **是** |
| RDMA_READ | 读取远端内存 | **零参与** | **是** |
| SEND/RECV | 发送消息到远端 | 需要预先 post Receive Buffer | 否 |

单边操作的核心优势：**接收端 CPU 完全不参与**。Prefill Worker 执行 RDMA_WRITE 把 KV Cache 写到 Decode Worker 的内存，Decode Worker 的 CPU 甚至不知道数据已经到了——直到它主动检查或收到完成通知。

##### 6.6 完成队列轮询（源码验证）

```cpp
// worker_pool.cpp
const static size_t kPollCount = 64;
ibv_wc wc[kPollCount];
int nr_poll = context_.poll(kPollCount, wc, cq_index);

for (int i = 0; i < nr_poll; ++i) {
    Transport::Slice *slice = (Transport::Slice *)wc[i].wr_id;
    if (wc[i].status != IBV_WC_SUCCESS) {
        if (wc[i].status == IBV_WC_WR_FLUSH_ERR) {
            // QP 正常销毁时的刷新，不算错误
            slice->markFailed();
        } else {
            // 真正的传输错误，触发路径故障处理
            handlePathFailure(slice->peer_nic_path, slice->rdma.endpoint);
            if (shouldRetrySlice(slice)) { /* 重新调度 */ }
        }
    } else {
        slice->markSuccess();
    }
}
```

关键设计：
- **批量轮询**：一次 `ibv_poll_cq` 最多取 64 个完成事件，减少系统调用开销
- **WR_FLUSH_ERR 专门处理**：QP 销毁时在途 WR 会以 FLUSH 错误完成，这是正常行为
- **故障自动重试**：真正的传输错误触发路径故障处理，Slice 会被重新调度到其他路径

##### 6.7 Doorbell 机制

当应用调用 `ibv_post_send()` 时，驱动做了什么？

```
1. 构造 WQE (Work Queue Element) → 写入 Send Queue（主机内存，NIC 可 DMA 读取）
2. 敲门铃 (Doorbell) → 写入 NIC 的 MMIO 寄存器（BlueFlame 寄存器）
3. NIC 收到 Doorbell → DMA 读取 WQE → 执行 RDMA 操作
```

Mooncake 的 IBGDA（GPU 发起 RDMA）路径中，**GPU 线程直接构造 WQE 和敲 Doorbell**：

```cpp
// ibgda_device.cuh — GPU 端代码
// 1. GPU 线程构造 WQE
mc_ibgda_put(qp, dest_addr, rkey, source_addr, length);

// 2. 写入 Doorbell Record (DBR)
mc_st_release_u32(&qp->dbr->send_counter, ...);

// 3. 可选：写入 BlueFlame (BF) 寄存器（MMIO）
if (qp->bf != nullptr) {
    // GPU 直接写 MMIO，通知 NIC
}
```

这是 Mooncake EP 实现"GPU 直传"的底层机制——GPU 线程不需要 CPU 介入就能发起 RDMA 操作。

---

### 七、GPUDirect RDMA：GPU 显存直达远端

##### 7.1 传统路径 vs GPUDirect 路径

```
传统路径 (GPU → CPU → NIC → NIC → CPU → GPU):
┌──────┐  cudaMemcpy  ┌──────┐  RDMA WRITE  ┌──────┐  cudaMemcpy  ┌──────┐
│GPU A │─────────────►│CPU A │─────────────►│CPU B │─────────────►│GPU B │
│VRAM  │  (PCIe 往返1)│DRAM  │              │DRAM  │ (PCIe 往返2) │VRAM  │
└──────┘              └──────┘              └──────┘              └──────┘

GPUDirect RDMA 路径 (GPU → NIC → NIC → GPU):
┌──────┐              RDMA WRITE              ┌──────┐
│GPU A │─────────────────────────────────────►│GPU B │
│VRAM  │  (PCIe 直达 NIC，零拷贝)              │VRAM  │
└──────┘                                       └──────┘
```

| 维度 | 传统路径 | GPUDirect 路径 |
|------|---------|---------------|
| CPU 参与 | 2 次 GPU↔CPU 拷贝 | 零参与 |
| 延迟 | 高（2 次 PCIe 往返） | 低（1 次 PCIe 往返） |
| 带宽 | 受 CPU 内存带宽限制 | 受 PCIe/RDMA 带宽限制 |
| 内存需求 | 需要额外 CPU 缓冲区 | 无额外内存 |
| 前提条件 | 无 | NVIDIA GPU + ConnectX NIC + nvidia-peermem |

##### 7.2 nvidia-peermem 内核模块

```
nvidia-peermem 工作原理：

1. 加载模块：modprobe nvidia-peermem
2. 模块调用 ib_register_peer_memory_client() 注册到 IB 核心
3. 当 ibv_reg_mr() 被调用在 GPU 地址上时：
   a. IB 核心检测到这是 GPU 内存
   b. 调用 nvidia-peermem 的 pin 回调函数
   c. nvidia-peermem pin 住 GPU 页面，建立 DMA 映射
   d. NIC 获得 GPU 内存的物理地址，可以直接 DMA 访问
```

```bash
# 验证 nvidia-peermem 已加载
lsmod | grep nvidia_peermem

# 验证 GPUDirect RDMA 可用
cat /proc/driver/nvidia/gpus/*/peermem
```

##### 7.3 cudaMalloc vs cudaHostAlloc

| 分配方式 | 内存位置 | RDMA 注册方式 | 延迟 | 内存消耗 |
|---------|---------|-------------|------|---------|
| `cudaMalloc()` | GPU VRAM | nvidia-peermem 或 dmabuf | 最低 | GPU 显存 |
| `cudaHostAlloc()` | CPU DRAM (pinned) | `ibv_reg_mr()` 直接 | 较低 | CPU 内存 |
| `cudaMallocManaged()` | 统一内存 | **不适合 RDMA** | N/A | N/A |

> 笔者注：`cudaMallocManaged()` 的页面会在 CPU 和 GPU 之间迁移，导致 dmabuf 导出的文件描述符失效。Mooncake 明确警告不要对 HIP managed memory 使用 RDMA 注册。

---

### 八、InfiniBand 寻址：LID、GID、PKey

| 概念 | 大小 | 说明 | RoCE v2 中的对应 |
|------|------|------|----------------|
| LID (Local ID) | 16-bit | IB 子网内唯一，由子网管理器分配 | 不使用（以太网用 MAC） |
| GID (Global ID) | 128-bit | 全局唯一标识，RoCE v2 中由 IP 地址派生 | `::ffff:a.b.c.d`（IPv4 映射） |
| PKey (Partition Key) | 16-bit | 分区隔离，类似 VLAN | 索引 0 = 默认分区 |
| QKey (Queue Key) | 32-bit | UD QP 认证密钥 | RC QP 不使用 |

Mooncake 的 GID 自动选择逻辑（`rdma_gid_probe.h`）：

```
1. 枚举所有 GID 候选
2. 按网络可达性排序：IPv4 映射 > link-local > overlay
3. 尝试建立连接
4. 连接失败 → 自动切换到下一个 GID 候选
```

环境变量控制：

```bash
MC_GID_INDEX=0      # 手动指定 GID 索引（默认自动选择）
MC_IB_PORT=1        # IB 端口号
MC_PKEY_INDEX=0     # PKey 索引
```

---

### 九、Mooncake RDMA 传输架构：工程实践

##### 9.1 分层架构

```
RdmaTransport (顶层：管理 NIC、内存、握手)
  └→ RdmaContext (每 NIC：PD、CQ、MR 映射、WorkerPool、EndpointStore)
       └→ RdmaEndPoint (每对端 NIC：QP 列表、WR 深度追踪、连接状态机)
            └→ WorkerPool (每 Context 线程：submitPostSend + performPollCq)
       └→ EndpointStore (SIEVE 或 FIFO 淘汰策略的 Endpoint 缓存)
```

##### 9.2 数据流

```
1. submitTransferTask() → 将请求切分为 64KB Slice
2. 每个 Slice 分配 source_lkey + dest_rkey（拓扑感知设备选择）
3. Slice 分发到 WorkerPool::submitPostSend()（按 peer_nic_path 分片）
4. Worker 线程构建 ibv_send_wr 链表，调用 ibv_post_send()
5. CQ 轮询线程批量 ibv_poll_cq(64)，处理完成事件
6. 失败 Slice 自动重试到备用路径
```

##### 9.3 关键配置参数（`config.h`）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_cqe` | 4096 | 每 CQ 的完成条目数 |
| `max_wr` | 256 | 每 QP 的 WR 深度 |
| `num_qp_per_ep` | 2 | 每 Endpoint 的 QP 数 |
| `max_sge` | 4 | 最大 Scatter/Gather 条目数 |
| `max_inline` | 64 | 最大内联数据字节数 |
| `workers_per_ctx` | 2 | 每 NIC Context 的 Worker 线程数 |
| `slice_size` | 65536 | 传输 Slice 大小（64KB） |
| `retry_cnt` | 9 | 每 Slice 最大重试次数 |

##### 9.4 NCCL vs Mooncake 自定义 RDMA

| 维度 | NCCL | Mooncake 自定义 RDMA |
|------|------|---------------------|
| 设计目标 | 集合通信（AllReduce、AllGather） | 点对点 KV Cache 传输 |
| 操作模式 | 双边（SEND/RECV） | **单边（RDMA_WRITE/READ）** |
| 接收端 CPU | 需要预先 post Receive Buffer | **零参与** |
| Slice 流水线 | 无 | 64KB Slice 级流水线 |
| 拓扑感知 | 有 | 有（NUMA 亲和性） |
| Endpoint 复用 | 无 | SIEVE 淘汰缓存 |
| GPUDirect | 支持 | 支持（nvidia-peermem/dmabuf） |

---

### 十、vLLM 中的 RDMA 使用

vLLM 通过 MooncakeConnector 和 MooncakeStoreConnector 使用 RDMA：

```python
# vLLM 配置 — PD 解耦直传
--kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer",
    "mooncake_protocol": "rdma",
    "mooncake_metadata_server": "etcd://127.0.0.1:2379"
}'

# vLLM 配置 — 分布式 KV Cache 池
--kv-transfer-config '{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "mooncake_protocol": "rdma",
    "mooncake_master_addr": "127.0.0.1:50051"
}'
```

vLLM 的 MooncakeConnector 工作流：

```
1. MooncakeConnectorWorker 初始化 TransferEngine
2. 注册 KV Cache 张量 → engine.batch_register_memory()
3. Prefill 完成后 → engine.batch_transfer_sync_write()
   （RDMA WRITE 将 KV Cache 推送到 Decode Worker）
4. Decode Worker 通过 ZMQ 侧通道协调哪些 block 需要传输
```

> 笔者注：vLLM 的 `MultiConnector` 支持同时使用 MooncakeConnector（PD 传输）+ MooncakeStoreConnector（前缀缓存），实现"强1 + 强2 = 整个系统更强"。但要注意两种 connector 的 block_size 等参数必须对齐，否则会带来精度问题。

---

### 十一、性能调优实战

##### 11.1 NUMA 绑定

```bash
# 查看 NIC 的 NUMA 节点
cat /sys/class/infiniband/mlx5_0/device/numa_node
# 输出：0

# 绑定进程到同一 NUMA 节点
numactl --cpunodebind=0 --membind=0 ./your_application
```

Mooncake 自动读取 NUMA 信息并绑定 Worker 线程：

```cpp
// 从 sysfs 读取 NUMA 节点
int numa_node = readNumaNode("/sys/class/infiniband/" + device_name + "/device/numa_node");
// WorkerPool 绑定到对应 NUMA 节点
WorkerPool(context, numa_socket_id);
```

##### 11.2 CQ 调优

```bash
# 多 CQ 减少锁竞争
export MC_NUM_CQ_PER_CTX=4

# 多完成 Channel 提高事件通知吞吐
export MC_NUM_COMP_CHANNELS_PER_CTX=2
```

##### 11.3 PCIe 乱序优化

```bash
# 启用 PCIe 读取乱序完成（提高吞吐，代价是严格的内存排序）
export MC_IB_PCI_RELAXED_ORDERING=1
```

##### 11.4 LAG 端口均衡

```bash
# 在 bond/LAG 模式下，QP 均衡分配到物理端口
export MC_MLX5_QP_LAG_PORT_BALANCE=1
```

##### 11.5 UDP 源端口多样化

```bash
# 为每个 QP 设置不同的 UDP 源端口，让交换机 ECMP 能区分流量
export MC_MLX5_QP_UDP_SPORTS=auto
```

##### 11.6 流量类别与服务级别

```bash
# 设置 IB Traffic Class（对应 IP DSCP）
export MC_IB_TRAFFIC_CLASS=106    # DSCP 26

# 设置 IB Service Level
export MC_IB_SERVICE_LEVEL=0
```

---

### 十二、部署检查清单

| 检查项 | 命令 | 预期结果 |
|--------|------|---------|
| RDMA 设备 | `ibv_devinfo` | 显示 mlx5_* 设备信息 |
| 设备权限 | `ls -la /dev/infiniband/` | 有读写权限 |
| RDMA 带宽 | `ib_write_bw -d mlx5_0` | 接近网卡标称带宽 |
| MOFED 版本 | `ofed_info -s` | MLNX_OFED 5.x+ |
| nvidia-peermem | `lsmod \| grep peermem` | 已加载 |
| GPUDirect | `cat /proc/driver/nvidia/gpus/*/peermem` | 支持 |
| PFC 配置 | `mlnx_qos -i eth0` | RoCE 优先级启用 PFC |
| ECN 配置 | 交换机 CLI | RoCE 队列启用 ECN 标记 |
| GID 表 | `show_gids` | 包含 IPv4 映射 GID |
| NUMA 信息 | `cat /sys/class/infiniband/mlx5_0/device/numa_node` | 0 或 1 |

---

### 总结

| 层次 | 核心概念 | 一句话总结 |
|------|---------|----------|
| 协议 | RoCE v2 | IB 传输层 + UDP/IP 封装，可路由，LLM 推理主流 |
| 无损网络 | PFC + ECN + DCQCN | PFC 防丢包，ECN 发信号，DCQCN 控速率 |
| 硬件 | ConnectX-7 | 400G 带宽，AI 推理甜点选择 |
| 驱动 | MOFED + libibverbs | 用户态 Verbs API，内核旁路 |
| 编程模型 | QP/MR/CQ | 队列对通信，内存注册，完成通知 |
| GPUDirect | nvidia-peermem | GPU 显存直接被 NIC DMA 访问 |
| Mooncake | 单边 RDMA + Slice 流水线 | RDMA_WRITE/READ，64KB Slice，拓扑感知 |
| vLLM | MooncakeConnector | KV Cache 通过 RDMA "瞬移"到 Decode Worker |

**一行建议**: 部署 RDMA 环境，先跑通 `ib_write_bw`，再跑通 `nvidia-peermem + GPUDirect`，最后才接入 Mooncake——基础设施不稳，上层再精妙也白搭。

**延伸阅读**：
- RDMA 编程入门：https://www.rdmamojo.com/
- GPUDirect RDMA 文档：https://docs.nvidia.com/cuda/gpudirect-rdma/
- Mooncake Transfer Engine 设计文档：https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html
- DCQCN 论文：https://www.microsoft.com/en-us/research/publication/congestion-control-for-large-scale-rdma-deployments/
- RoCE v2 配置指南：https://docs.nvidia.com/networking/display/ConnectX7VPIOCP3/Introduction

---

*本文结合 Mooncake 和 vLLM 系统的工程实践，从硬件到驱动、从协议到编程模型，完整拆解了 RDMA/RoCE 技术栈。*
