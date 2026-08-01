# Mooncake 拓扑感知路由技术拆解：当集群有十张网卡，谁走哪条路？

> **系列**: Mooncake 技术博客系列 | **类型**: 核心技术详解篇
>
> 一座城市有十条高速路，每条路离不同的仓库远近不同。如果所有货车都挤最近的那条路，其他路空着——这就是没有拓扑感知的传输。Mooncake 的拓扑感知路由就像一个"交通指挥官"，根据出发地和目的地的位置，为每辆货车分配最优车道。

---

### 引言

想象你管理着一个大型物流中心。中心有 10 个仓库（NUMA 节点），10 条高速路（RDMA 网卡），每个仓库离不同的高速路入口远近不同。如果仓库 A 的货物总是走最近的路 1，路 1 很快就堵了，而路 2 到路 10 还空着。

问题出在哪？**你没有考虑"谁离哪条路近"**。

在 GPU 集群中，这个问题更加尖锐：一台 8-GPU 服务器通常有 4-8 张 RDMA 网卡，每张网卡与不同 NUMA 节点的距离不同。如果传输请求不考虑 NUMA 亲和性，数据可能要跨 NUMA 节点传输，额外付出 30-50% 的延迟代价（上一篇文中写到的 NUMA 以及 NUMA 亲和性原理）。

Mooncake 的拓扑感知路由就是为了解决这个问题。它通过**拓扑发现、矩阵构建、NIC 选择**三步，确保每个传输请求都走最优路径。

> 笔者注：一台物理物理服务器8张卡，是当前业界标配，不管是英伟达、昇腾、寒武纪等，还有自己设计制造AI服务器的字节、DeepSeek等，多台物理服务器计算面板组成一个机柜，比如16台，一个机柜就8*16=128张卡，字节有36个计算面板组成一个单机柜的，也就是8*36=228张卡。

> 同时，也不一定是单机柜装载计算面板越多越好，本月17号在2026世界人工智能大会（WAIC）亮相的华为昇腾950超节点（Atlas 950 SuperPoD），采用的单个计算柜16台计算面板，不过整体集群起步是16*64=1024张卡，最大8192张卡，巨无霸（据百度百科介绍，8192卡SuperPod超节点集群，128个计算柜和32个互联柜，共计160个机柜，占地面积约1000平方米），这涉及到网络互连、存储、计算以及像vLLM、Mooncake推理和KV Cache系统整套系统全栈的配合、协同和优化。

---

### 核心问题：为什么需要拓扑感知？

##### NUMA 亲和性对 RDMA 性能的影响

一台 8-GPU + 4-NIC 服务器分成两个 NUMA 节点：

`NUMA 0: GPU 0-3 + NIC 0,1` 和 `NUMA 1: GPU 4-7 + NIC 2,3`

- 有亲和性：GPU 0 → NIC 0（同在 NUMA 0节点，直连，延迟基准）
- 无亲和性：GPU 0 → NIC 2（跨 NUMA节点，延迟 +30-50%）

```
服务器拓扑 (8 GPU + 4 NIC):

NUMA Node 0                    NUMA Node 1
┌──────────────┐               ┌──────────────┐
│ GPU 0  GPU 1 │               │ GPU 4  GPU 5 │
│ GPU 2  GPU 3 │               │ GPU 6  GPU 7 │
│   NIC 0      │               │   NIC 2      │
│   NIC 1      │               │   NIC 3      │
└──────────────┘               └──────────────┘
       │                              │
       └──── QPI/UPI 互联 ───────────┘
              (跨 NUMA 延迟高 30-50%)
```

| 场景 | GPU → NIC 路径 | 额外延迟 |
|------|--------------|---------|
| GPU 0 → NIC 0 | 同 NUMA，直连 | 0（基准） |
| GPU 0 → NIC 2 | 跨 NUMA | +30-50% |
| GPU 4 → NIC 0 | 跨 NUMA | +30-50% |
| GPU 4 → NIC 2 | 同 NUMA，直连 | 0（基准） |

> 笔者注：在 8×400 Gbps 网络下，跨 NUMA 访问可能导致 10-20% 的带宽损失。对于 40GB 的 KV Cache 传输，这意味着额外数百微秒的延迟——在 SLO 严格的推理场景中，这是不可接受的。

##### Mooncake 怎么用 NUMA 亲和性？

拓扑矩阵为每种内存标记两类网卡：
- preferred_hca：同 NUMA 的网卡（首选，延迟最低）
- avail_hca：其他 NUMA 的网卡（备选，preferred 不可用时降级使用）

传输时优先选 preferred，失败了才轮询 avail，实现"首选最优、渐进降级"。

---

### 三步走：拓扑感知路由的完整流程

##### 第一步：拓扑发现（Topology Discovery）

`Topology` 类在初始化时自动发现服务器的硬件拓扑：

```cpp
// mooncake-transfer-engine/include/topology.h
class Topology {
    // 自动发现拓扑
    int discover();
    int discover(const std::vector<std::string>& filter);  // 过滤指定设备
    
    // 从 JSON 解析拓扑（用于自定义配置）
    int parse(const std::string& topology_json);
};
```

发现过程：

1. **扫描 IB 设备**：通过 sysfs 发现所有 RDMA 网卡（如 `mlx5_0` ~ `mlx5_7`）
2. **NUMA 亲和性检测**：查询每个 IB 设备的 NUMA 节点归属
3. **GPU-NIC 映射**：建立 GPU 到 NIC 的亲和性关系
4. **构建拓扑矩阵**：为每种内存类型生成 preferred/available NIC 列表

##### 第二步：拓扑矩阵构建

拓扑矩阵是拓扑感知路由的核心数据结构：

```cpp
struct TopologyEntry {
    std::string name;                      // 存储类型 (如 "dram", "gpu:0")
    std::vector<std::string> preferred_hca; // 首选网卡列表
    std::vector<std::string> avail_hca;     // 可用网卡列表
};

using TopologyMatrix = std::unordered_map<std::string, TopologyEntry>;
```

一个典型的拓扑矩阵：

```
TopologyMatrix:
┌──────────┬──────────────────────┬──────────────────────┐
│ 类型     │ preferred_hca        │ avail_hca            │
├──────────┼──────────────────────┼──────────────────────┤
│ dram     │ [mlx5_0, mlx5_1]     │ [mlx5_2, mlx5_3]    │
│ gpu:0    │ [mlx5_0]             │ [mlx5_1, mlx5_2, mlx5_3] │
│ gpu:1    │ [mlx5_0]             │ [mlx5_1, mlx5_2, mlx5_3] │
│ gpu:4    │ [mlx5_2]             │ [mlx5_0, mlx5_1, mlx5_3] │
│ gpu:5    │ [mlx5_2]             │ [mlx5_0, mlx5_1, mlx5_3] │
└──────────┴──────────────────────┴──────────────────────┘
```

解读：
- `dram` 的 preferred_hca 是 `mlx5_0, mlx5_1`——因为 DRAM 在 NUMA 0，这两张网卡也在 NUMA 0
- `gpu:0` 的 preferred_hca 是 `mlx5_0`——GPU 0 和 mlx5_0 在同一 NUMA 节点
- `gpu:4` 的 preferred_hca 是 `mlx5_2`——GPU 4 和 mlx5_2 在同一 NUMA 节点

##### 第三步：NIC 选择

当传输请求到来时，`selectDevice` 根据源内存位置选择最优 NIC：

```cpp
// mooncake-transfer-engine/include/topology.h
class Topology {
    // 为指定存储类型选择设备
    int selectDevice(const std::string storage_type, int retry_count = 0);
    
    // 根据本地 HCA 名称选择设备
    int selectDeviceByLocalHca(const std::string storage_type,
                               std::string_view local_hca, int retry_count = 0);
};
```

选择逻辑：

```
selectDevice("gpu:0")
    │
    ├── 查找 TopologyMatrix["gpu:0"]
    ├── preferred_hca = [mlx5_0]
    ├── avail_hca = [mlx5_1, mlx5_2, mlx5_3]
    │
    ├── retry_count == 0 → 选择 mlx5_0 (preferred)
    ├── retry_count == 1 → 选择 mlx5_1 (avail)
    ├── retry_count == 2 → 选择 mlx5_2 (avail)
    └── retry_count == 3 → 选择 mlx5_3 (avail)
```

> 笔者注：`retry_count` 参数在路径故障时递增。首次选择 preferred NIC，失败后依次尝试 avail NIC，实现了自动故障转移。

---

### 多网卡带宽聚合：Slice 级调度

拓扑感知路由不仅选择最优路径，还实现了**多网卡带宽聚合**——一个大传输被切分为多个 64KB Slice，每个 Slice 可以走不同的 NIC。

##### Slice 调度策略

```
40GB 传输请求 → 切分为 ~625,000 个 64KB Slice

Slice 0: selectDevice("gpu:0") → mlx5_0 (preferred)
Slice 1: selectDevice("gpu:0") → mlx5_1 (avail, 轮询)
Slice 2: selectDevice("gpu:0") → mlx5_2 (avail, 轮询)
Slice 3: selectDevice("gpu:0") → mlx5_3 (avail, 轮询)
Slice 4: selectDevice("gpu:0") → mlx5_0 (preferred, 循环)
...
```

调度策略的核心原则：

| 原则 | 说明 |
|------|------|
| **Preferred 优先** | 同 NUMA 的 NIC 优先分配 Slice |
| **轮询分配** | 同优先级的 NIC 轮询分配，避免单条链路过载 |
| **故障自动降级** | 某条链路失败后，Slice 自动分配到其他链路 |
| **负载均衡** | 根据实测完成时间动态调整分配比例（TENT 模式） |

##### 带宽聚合效果

```
单网卡:  ──────────────────────────── 25 GB/s
4 网卡:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87 GB/s (3.5x)
8 网卡:  ════════════════════════════════════════════════════════════ 190 GB/s (7.6x)
```

> 笔者注：带宽聚合不是完美的线性缩放。4 网卡 87 GB/s 相比单网卡 25 GB/s 是 3.5x 而非 4x，8 网卡 190 GB/s 相比单网卡是 7.6x 而非 8x。这主要因为跨 NUMA 的 Slice 引入了额外延迟，以及协议开销随 Slice 数量增加。

---

### 设备过滤：白名单控制

在多网卡环境下，你可能只想使用特定的网卡。`MC_TE_FILTERS` 环境变量提供了白名单控制：

```bash
# 只使用 mlx5_0 和 mlx5_1
export MC_TE_FILTERS="mlx5_0,mlx5_1"
```

```cpp
// 代码中也可以通过 filter 参数控制
Topology topo;
topo.discover({"mlx5_0", "mlx5_1"});  // 只发现这两张网卡

// 或禁用特定设备
topo.disableDevice("mlx5_3");
```

典型使用场景：

| 场景 | 配置 | 原因 |
|------|------|------|
| 只用同 NUMA 的网卡 | `MC_TE_FILTERS="mlx5_0,mlx5_1"` | 避免跨 NUMA 延迟 |
| 排除故障网卡 | `topo.disableDevice("mlx5_3")` | 网卡硬件故障 |
| 分离存储和计算网络 | `MC_TE_FILTERS="mlx5_4,mlx5_5"` | 存储 VLAN 和计算 VLAN 隔离 |

---

### 拓扑可视化：showLinks 调试工具

Transfer Engine 提供了 `showLinks` 方法来可视化当前拓扑：

```cpp
// 以 JSON 格式输出拓扑信息
std::string links = engine.showLinks(true /* json */);
```

输出示例：

```json
{
  "local_server_name": "node-A",
  "segments": {
    "node-B": {
      "segment_id": 1,
      "buffers": [
        {"addr": "0x7f0000000000", "size": 10737418240, "location": "dram"}
      ]
    }
  },
  "topology": {
    "dram": {
      "preferred_hca": ["mlx5_0", "mlx5_1"],
      "avail_hca": ["mlx5_2", "mlx5_3"]
    }
  }
}
```

> 笔者注：部署时务必先调用 `showLinks` 确认拓扑发现正确。如果 preferred_hca 列表为空，说明拓扑发现失败，传输会回退到随机选择 NIC，性能大打折扣。

---

### 高级调优：目标设备亲和性

`MC_ENABLE_DEST_DEVICE_AFFINITY` 环境变量启用一个更高级的优化——**考虑目标端设备的 NUMA 亲和性**。

默认情况下，拓扑感知只考虑源端的 NUMA 亲和性。启用目标设备亲和性后，传输请求会同时考虑源端和目标端的 NUMA 位置，选择两端都最优的路径。

```
源端: GPU 0 (NUMA 0) → preferred NIC: mlx5_0
目标端: DRAM (NUMA 1) → preferred NIC: mlx5_2

不启用目标亲和性: 选择 mlx5_0 (只看源端)
启用目标亲和性: 选择 mlx5_2 (综合考虑两端，或选择中间路径)
```

| 场景 | 建议 |
|------|------|
| 同 NUMA 通信 | 不需要，默认即可 |
| 跨 NUMA 通信 | 启用，可减少跨 NUMA 开销 |
| 多节点集群 | 启用，有助于全局最优路径选择 |

---

### 设计哲学：拓扑感知路由的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **自动发现** | 无需手动配置，自动探测硬件拓扑 | `Topology.discover()` 扫描 sysfs |
| **渐进降级** | preferred 不可用时自动降级到 avail | `retry_count` 递增选择 |
| **透明聚合** | 上层无需关心多网卡，带宽自动聚合 | Slice 级多路径调度 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| 拓扑矩阵 | 每种内存类型的 preferred/available NIC 列表 |
| NUMA 亲和性 | 同 NUMA 的 NIC 延迟低 30-50%，是首选路径 |
| Slice 调度 | 64KB Slice 级多路径分配，实现带宽聚合 |
| 设备过滤 | 白名单控制使用哪些网卡 |
| 目标设备亲和性 | 同时考虑源端和目标端的 NUMA 位置 |

**建议**: 部署时用 `showLinks` 验证拓扑发现结果，确认 preferred_hca 列表非空——如果为空，说明 NUMA 信息不可用，需要检查内核版本和 sysfs 挂载。

**延伸阅读**：
- Linux NUMA 架构：https://www.kernel.org/doc/html/latest/admin-guide/mm/numaperf.html
- Mellanox OFED 文档：https://docs.nvidia.com/networking/display/MLNXOFEDv5100

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。下一篇我们将深入 Mooncake Store 的分布式 KV Cache 存储引擎。*
