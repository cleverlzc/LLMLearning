# Mooncake 当故障发生时：水密舱室、心跳探针与弹性恢复

> **系列**: Mooncake 技术博客系列 | **类型**: 故障容错篇

本文中的比喻，作为一名软件从业者，如果同时又是一名军事迷，是幸福的，理解这些架构设计的背后逻辑，会更有感触。如果不是也没关系，闲暇时，强烈推荐去海军博物馆参观一些潜艇，听听潜艇自豪和悲壮的故事，对了，强烈推荐一部电视剧《深海利剑》。写到这里，突然有点泪目，珍惜来之不易的和平下的幸福生活，如果你现在在焦虑，生活在中国，还没有那么糟。

言归正传，在数据中心中，故障随时可能发生。GPU 过热、网线脱落、整台机器崩溃。Mooncake 不试图预防故障——它的设计初衷是能够在故障发生时继续提供服务。

就像一艘带有水密舱室的潜艇：一个舱室进水，舱门自动关闭，潜艇仍能保持漂浮，维修人员可以在不浮出水面的情况下更换受损部分。

---

### 引言

想象一艘潜艇。它由多个水密舱室组成，舱室之间有自动密封门。如果某个舱室被鱼雷击中进水，密封门会立刻关闭——潜艇不会沉没，只是少了一个舱室的浮力。维修人员可以在水下更换受损部件，然后重新打开密封门，恢复全部功能。

如果潜艇不是水密舱设计，一个舱室进水就会导致整艘潜艇沉没。

分布式系统也是一样。在数据中心中，GPU 过热、网线脱落、内存故障、整台机器崩溃——这些都是"鱼雷"。一个没有故障容错设计的系统，任何一个组件故障都可能导致整个系统不可用。而一个设计良好的系统，应该能在故障发生时继续提供服务，在不停机的情况下恢复完整容量。

Mooncake 的故障容错设计可以提炼为四个通用模式：**检测→隔离→降级→恢复**。这四个步骤不是 Mooncake 的发明，而是所有生产级分布式系统的共同选择。

---

### 模式卡片

| 项目 | 内容 |
|------|------|
| **模式名称** | 故障容错四步法：检测→隔离→降级→恢复 |
| **别名** | 水密舱模式、弹性恢复、优雅降级 |
| **问题** | 分布式系统中组件故障不可避免，如何在故障发生时继续提供服务 |
| **方案** | 心跳检测故障 → 掩码隔离故障节点 → 降级持续服务 → 弹性恢复完整容量 |
| **核心权衡** | 可用性优于一致性（CAP 定理的 AP 选择） |
| **关键力** | 检测速度 vs 误报率、降级深度 vs 服务质量、恢复速度 vs 数据一致性 |

---

### 第一步：检测故障——心跳探针

#### 模式

故障检测是容错的第一步——你必须先知道某个组件坏了，才能做出反应。最通用的检测机制是**心跳（Heartbeat）**：组件定期发送"我还活着"的信号，如果接收方在超时时间内没有收到心跳，就判定该组件故障。

#### Mooncake 的三层心跳

Mooncake 在不同层级使用了不同频率的心跳：

| 层级 | 谁心跳 | 频率 | 超时 | 检测什么 |
|------|--------|------|------|---------|
| Store 层 | Client → Master | 1 秒 | `client_live_ttl_sec_` | Client 进程是否存活 |
| PG 层 | ConnectionPoller 轮询 | 8ms~1024ms 指数退避 | 连接状态翻转 | GPU rank 间连接是否正常 |
| 传输层 | RDMA QP 重试 | 硬件级 | `timeout=14, retry=7` | 单次传输是否成功 |

**Store 层心跳**：Client 每秒向 Master 发送一次 Ping。Master 维护一个 TTL 表，每次收到 Ping 就刷新 TTL。如果某个 Client 的 TTL 过期，Master 判定该 Client 已死亡，触发清理流程。

```
Client A: Ping → Master (TTL 刷新为 now + ttl_sec)
Client A: Ping → Master (TTL 刷新)
Client A: 💥 崩溃
         ... ttl_sec 后 ...
Master: Client A 的 TTL 过期 → 标记为 expired → 清理资源
```

Client 侧也有反向检测：如果连续 3 次 Ping 失败（`max_ping_fail_count = 3`），Client 尝试切换到新的 Master Leader（主备切换）。

**PG 层连接轮询**：Process Group 的 `ConnectionPoller` 以指数退避（8ms→1024ms）轮询每个 peer 的连接状态。当检测到 peer 连接断开（`peerConnected` 翻转为 false），立即将该 rank 的 `activeRanks` 设为 false。

#### 通用模式

心跳检测在所有分布式系统中都是标配：

| 系统 | 心跳机制 | 超时行为 |
|------|---------|---------|
| Mooncake | Client→Master Ping + PG 连接轮询 | 清理资源 / 隔离 rank |
| K8s | kubelet→API Server 心跳 | 标记 Node 为 NotReady |
| ZooKeeper | Session 心跳 | 临时节点删除 |
| 数据库主从 | replication heartbeat | 触发故障切换 |
| 微服务 | Health Check 探针 | 摘除流量 |

**本质相同**：用周期性心跳判断存活，用超时容忍网络抖动。心跳间隔越短，检测越快，但误报率越高（网络抖动被误判为故障）；超时越长，误报越少，但检测越慢。这是检测速度与误报率之间的永恒张力。

---

### 第二步：隔离分区——关闭水密门

#### 模式

检测到故障后，第一步不是修复，而是**隔离**——把故障组件从系统中摘除，防止故障扩散。就像潜艇关闭水密门：你不需要立刻修好进水的舱室，但必须先把它隔离开，防止海水涌入其他舱室。

#### Mooncake 的 active_ranks 掩码

Mooncake Process Group 用一个布尔数组 `activeRanks` 跟踪哪些 GPU rank 是存活的：

```
activeRanks = [true, true, true, true]   ← 初始状态：4 个 rank 全部存活

Rank 1 故障！

activeRanks = [true, false, true, true]  ← Rank 1 被隔离

调度逻辑在下一次迭代中自动跳过 Rank 1：
  for (int j = 0; j < size; ++j) {
      if (!activeRanks[j]) continue;   ← 跳过故障 rank
      // 只向存活的 rank 分发令牌
  }
```

`activeRanks` 被分配为页锁定内存（`cudaHostAlloc`），CPU 和 GPU 都能直接访问。当 `ConnectionPoller` 检测到连接断开时，立即将 `activeRanks[failed_rank] = false`——这个修改对 GPU 内核立即可见，下一次 All-to-All 或 AllReduce 操作就会自动跳过故障 rank。

#### 传输层的设备隔离

在传输层，当一张网卡故障时，`Topology::disableDevice()` 将该网卡从所有拓扑条目的 `preferred_hca` 和 `avail_hca` 列表中移除：

```
故障前:
  "cpu:0" → preferred_hca: [mlx5_0, mlx5_1]  avail_hca: [mlx5_2]

mlx5_1 故障！

故障后:
  "cpu:0" → preferred_hca: [mlx5_0]  avail_hca: [mlx5_2]
  ← mlx5_1 被从所有列表中移除，后续传输不再使用它
```

#### 通用模式

| 系统 | 隔离机制 | 效果 |
|------|---------|------|
| Mooncake PG | `activeRanks` 掩码 | 跳过故障 rank |
| Mooncake 传输 | `disableDevice()` | 跳过故障网卡 |
| K8s | `cordon` / `drain` | 不再调度 Pod 到故障节点 |
| 微服务 | 熔断器（Circuit Breaker） | 停止向故障实例发请求 |
| 数据库 | 摘除只副本 | 读流量不再路由到故障副本 |

**本质相同**：用一个标记（掩码/位/标签）将故障组件从调度逻辑中排除。隔离是 O(1) 操作——只需翻转一个标志位，不需要移动数据或重启服务。

---

### 第三步：降级持续服务——少一个舱室也能浮

#### 模式

隔离之后，系统以降低的容量继续提供服务。这是故障容错最关键的决策：**宁可降级服务，也不中断服务**。

CAP 定理告诉我们，在网络分区时必须在一致性和可用性之间做选择。Mooncake 选择了可用性（AP）——即使部分组件故障，系统仍继续提供服务，接受某些数据可能暂时过时。这与 DNS、缓存系统和最终一致性数据库的权衡策略相同。

#### Mooncake 的降级场景

| 故障场景 | 降级方式 | 服务影响 |
|---------|---------|---------|
| GPU rank 故障 | 跳过该 rank 的专家，令牌路由到其他 rank | 专家数量减少，推理质量略降，但不丢请求 |
| 网卡故障 | 跳过该网卡，用剩余网卡传输 | 带宽减少，延迟略增，但传输不中断 |
| Client 故障 | 该 Client 的副本不可用，用其他副本 | 可用副本减少，但数据仍可访问 |
| Master 故障 | Standby 提升为新主 | 短暂不可用（秒级），之后完全恢复 |

以 GPU rank 故障为例，模拟对话展示降级过程：

```
EP RANK 0:  分发令牌到 rank 0, 1, 2, 3 的专家
EP RANK 1:  处理我的专家令牌...
EP RANK 1:  ... ... ...  💥 崩溃

PG COORDINATOR:  Rank 1 无响应！更新 active_ranks 掩码...
PG COORDINATOR:  active_ranks[1] = 0 — Rank 1 已标记为故障

EP RANK 0:  重新分发 — 跳过 Rank 1，令牌路由到 Rank 2
EP RANK 2:  收到额外令牌 — 系统继续服务！

PG COORDINATOR:  替换 rank 加入... active_ranks[1] = 1 — 恢复全部容量
```

关键观察：**系统没有卡住，没有丢弃请求，没有重启**。它只是少了一个专家，令牌被路由到其他专家。服务质量略有下降（专家数量减少），但服务没有中断。

#### 通用模式

| 系统 | 降级方式 | CAP 选择 |
|------|---------|---------|
| Mooncake | 跳过故障 rank/网卡/副本 | AP——可用性优先 |
| Cassandra | 降级读写一致性级别 | AP——可用性优先 |
| DNS | 返回缓存的旧记录 | AP——可用性优先 |
| 微服务 | 返回降级响应（默认值/缓存） | AP——可用性优先 |
| Spanner | 暂停写入直到分区恢复 | CP——一致性优先 |

**本质相同**：在故障发生时，选择"部分正确"而非"完全不可用"。这是生产系统几乎总是做出的选择——用户宁可看到稍微过时的数据，也不愿看到 502 错误。

---

### 第四步：弹性恢复——水下换零件

#### 模式

故障组件被隔离、系统降级运行后，最终需要恢复完整容量。**弹性恢复（Elastic Recovery）**是指在不中断服务的情况下动态添加替换节点，恢复到故障前的能力。

#### Mooncake 的弹性恢复

**PG 层——替换 rank 加入**：

`activeRanks` 数组预留了扩展槽位（`kMaxNumRanks`，远大于初始 `size`）。当替换 rank 加入时：

```
初始: activeRanks = [true, false, true, true, false, false, ...]
                                          ↑ 扩展槽位（false）

替换 rank 加入:
  1. 新 rank 建立 RDMA 连接到所有存活 rank（ConnectionPoller 自动处理）
  2. activeRanks[新rank] = true
  3. 下一次迭代，调度逻辑自动将令牌分发到新 rank
  4. 系统恢复全部容量——无重启，无停机
```

**Store 层——Master 主备切换**：

当 Master 故障时，Standby 节点通过以下步骤接管：

```
1. 检测: etcd lease 过期（Master 心跳停止 → lease 自动释放）
2. 选举: Standby 通过 etcd CreateWithLease 原子获取 leadership
3. 追赶: Standby 回放 OpLog，追赶到最新序列号
   ┌─ 快照基线: 加载最近的元数据快照
   ├─ OpLog 回放: 从快照点开始重放操作日志
   └─ 缺口修复: 从 etcd 拉取缺失的 OpLog 条目
4. 提升: PromoteStandby() → 状态机转为 PROMOTED
5. 服务: 启动 RPC 服务器，设置 Pod Label (mooncake.io/store-role=leader)
6. 客户端自动重连: Client 检测到 leader 变化 → SwitchLeader() → 连接新 Master
```

Standby 状态机的完整生命周期：

```
STOPPED → CONNECTING → SYNCING → WATCHING → PROMOTING → PROMOTED
                         │           │
                         ├──→ RECONNECTING → SYNCING
                         │           │
                         └──→ RECOVERING → SYNCING
```

**传输层——网卡恢复**：

故障网卡修复后，`Topology::enableDevice()` 将其重新加入 `preferred_hca` 和 `avail_hca` 列表。后续传输自动恢复使用该网卡。

#### 通用模式

| 系统 | 恢复机制 | 停机时间 |
|------|---------|---------|
| Mooncake PG | 替换 rank 加入，掩码翻转 | 零停机 |
| Mooncake Master | Standby 提升 + OpLog 追赶 | 秒级 |
| K8s | 新 Pod 调度 + 就绪探针 | 秒级 |
| 数据库 | 新副本同步 + 提升为主 | 分钟级 |
| 微服务 | 新实例启动 + 健康检查通过 | 秒级 |

**本质相同**：用"加入新节点 + 翻转标记位"代替"重启整个系统"。弹性恢复的关键是状态同步——新节点必须追赶到当前状态才能提供服务。追赶速度取决于日志（OpLog/WAL/binlog）的回放速度。

---

### 安全清理：不踩到在途请求

#### 模式

故障组件被隔离后，可能还有在途的操作（in-flight requests）。直接销毁资源会导致这些操作失败甚至崩溃。**两阶段销毁**确保在途操作完成后再释放资源。

#### Mooncake 的两阶段销毁

**RDMA 端点销毁**：

```
阶段 1: beginDestroy()
  ├─ active_ = false（不再接受新请求）
  ├─ QP 状态改为 ERR（硬件自动刷新在途 WR 到完成队列）
  └─ 端点进入 waiting_list_ 等待清理

  ... 在途 WR 逐渐完成 ...

阶段 2: finishDestroy()
  ├─ 检查 wr_depth_list_ 是否全部归零（所有在途 WR 已完成）
  ├─ 如果 30 秒内未排空 → 重试（最多 3 次）
  └─ 排空后 → ibv_destroy_qp() 正式释放
```

**段卸载（Segment Unmount）**：

```
阶段 1: GracefulUnmountSegment
  ├─ 通知 Master 开始计时
  ├─ 段从 mounted_segments_ 移到 gracefully_unmounting_segments_
  └─ TE MR 保持注册 → 远端 peer 仍可读取数据

  ... 宽限期内，远端 peer 完成最后的读取 ...

阶段 2: 宽限期到期
  ├─ Master 确认段已移除
  ├─ 注销 TE MR
  └─ 执行清理回调
```

#### 通用模式

| 系统 | 两阶段操作 | 安全保证 |
|------|-----------|---------|
| Mooncake RDMA | beginDestroy → finishDestroy | 在途 WR 不丢失 |
| Mooncake Segment | GracefulUnmount → CommitUnmount | 远端读取不中断 |
| K8s | PreStop hook → Termination | 优雅关闭容器 |
| 数据库 | 两阶段提交 | 事务原子性 |
| 微服务 | 优雅下线（drain） | 在途请求完成后再退出 |

**本质相同**：先标记"不再接受新请求"，等在途请求排空后再释放资源。这是从"硬杀"到"软杀"的转变——硬杀会导致级联失败，软杀保证平滑过渡。

---

### 四步法的协同

故障容错四步法在 Mooncake 的每个层级都有体现：

```
故障发生（GPU 崩溃 / 网卡脱落 / Master 宕机）
  │
  │  第一步：检测
  │  ├─ Store: Client 心跳超时
  │  ├─ PG: ConnectionPoller 连接状态翻转
  │  └─ 传输: RDMA QP 重试耗尽
  │
  │  第二步：隔离
  │  ├─ PG: activeRanks[failed] = false
  │  ├─ 传输: disableDevice(failed_nic)
  │  └─ Store: 从 ok_client_ 中移除
  │
  │  第三步：降级
  │  ├─ PG: 令牌路由到存活 rank
  │  ├─ 传输: 用剩余网卡传输
  │  └─ Store: 用其他副本服务请求
  │
  │  第四步：恢复
  │  ├─ PG: 替换 rank 加入，activeRanks[new] = true
  │  ├─ 传输: enableDevice(repaired_nic)
  │  └─ Store: Standby 提升为新 Master
  │
  ▼
系统恢复完整容量——全程无停机
```

| 层级 | 检测 | 隔离 | 降级 | 恢复 |
|------|------|------|------|------|
| PG（GPU rank） | 连接轮询 | activeRanks 掩码 | 跳过故障专家 | 替换 rank 加入 |
| 传输（网卡） | QP 重试耗尽 | disableDevice | 用剩余网卡 | enableDevice |
| Store（Client） | 心跳超时 | 移出 ok_client_ | 用其他副本 | 新 Client 注册 |
| Store（Master） | etcd lease 过期 | — | — | Standby 提升 |

---

### 可用性优于一致性：CAP 定理的实践选择

CAP 定理告诉我们，在网络分区时，必须在一致性（C）和可用性（A）之间做出选择。Mooncake 选择了可用性：

> **即使部分组件出现故障，系统仍继续提供服务，接受某些数据可能暂时过时。**

这与 DNS、缓存系统和最终一致性数据库的策略相同。为什么生产系统几乎总是选择 AP？

| 选择 | 用户体验 | 适用场景 |
|------|---------|---------|
| **AP（可用性优先）** | 看到稍微过时的数据，但服务不中断 | 面向用户的服务、缓存、推理服务 |
| **CP（一致性优先）** | 服务暂停直到一致，但数据始终正确 | 金融交易、配置管理、分布式锁 |

对于 LLM 推理服务，用户宁可看到稍微不完美的推理结果（少了一个专家），也不愿看到"服务不可用"。Mooncake 的 AP 选择意味着：

- GPU rank 故障 → 降级推理，不中断服务
- 副本不一致 → 用可用的副本，不等待所有副本同步
- Master 故障 → Standby 接管，接受秒级元数据延迟

---

### 这些模式不是 Mooncake 独有的

故障容错四步法在所有生产级分布式系统中反复出现：

| Mooncake 机制 | 通用模式 | 其他实例 |
|--------------|---------|---------|
| activeRanks 掩码 | 故障隔离标记 | K8s Node Condition、熔断器状态 |
| 心跳 + TTL | 存活检测 | ZooKeeper Session、K8s Node Heartbeat |
| 降级持续服务 | 优雅降级 | Cassandra 降级一致性、微服务降级响应 |
| Standby + OpLog | 热备 + 日志回放 | MySQL 主从、PostgreSQL 流复制 |
| 两阶段销毁 | 安全清理 | K8s PreStop Hook、数据库两阶段提交 |
| 弹性恢复 | 动态扩缩容 | K8s HPA、自动扩缩容组 |

**学系统，学的是模式，不是代码。** 故障容错四步法——检测→隔离→降级→恢复——是任何生产级分布式系统的通用框架。掌握了这个框架，你就能在任何系统中识别它、应用它。

---

### 小结

> Mooncake 不试图预防故障——它的设计初衷是能够在故障发生时继续提供服务。

| 步骤 | 模式 | Mooncake 实现 | 比喻 |
|------|------|-------------|------|
| 检测 | 心跳 + 超时 | Client Ping + PG 连接轮询 + QP 重试 | 声呐探测异常 |
| 隔离 | 掩码标记 | activeRanks = false / disableDevice | 关闭水密门 |
| 降级 | 优雅降级 | 跳过故障 rank/网卡/副本 | 少一个舱室继续航行 |
| 恢复 | 弹性恢复 | 替换 rank 加入 / Standby 提升 | 水下更换受损部件 |

核心洞察只有一句话：

> **可用性优于一致性。在故障发生时，宁可降级服务，也不中断服务。**

这艘"潜艇"不会因为一个舱室进水而沉没——它关闭水密门，用剩余舱室继续航行，在水下完成维修，然后恢复全部战斗力。这就是故障容错的工程哲学。

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
