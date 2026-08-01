# Mooncake P2P Store 详解：一火传千灯的模型权重分发

> **系列**: Mooncake 技术博客系列 | **类型**: 核心概念深潜篇
>
> 传统模型权重分发就像一个人拿着唯一的火种，逐个去点一千根蜡烛——点完一根再点下一根，手忙脚乱。P2P Store 的做法是传火——每根被点亮的蜡烛都能去点别的蜡烛，火种不减少，传火的人越来越多，整间屋子越来越快地亮起来。

---

前面介绍了 Mooncake Store，是个分布式内存池，Prefill节点的KV Cache数据save/put到共享内存池，Decode的时候再load/get上来，这样性能相对来说不好，甚至介绍SSD分层存储，性能相对来说只会更差，只是在增大存储容量避免频繁重算场景下的一种综合trade-off。

PD直接传输和P2P，直接从X节点的GPU到Y节点的GPU，比中间再经过一个存储池做中转，速度更快，性能更好。

### 引言

想象一间暗室里有一千根蜡烛需要点亮。你手里有唯一的火种，逐根去点——点完一根再点下一根，一千根蜡烛要点很久。更糟的是，你的火柴只有一根，一次只能点一支蜡烛。

但如果你换一种方式：**每根被点亮的蜡烛都能去点别的蜡烛**。第一根蜡烛帮你点第二根，两根蜡烛一起点第三和第四根，四根蜡烛一起点更多……火种不会因为点亮别人而变暗，但传火的能力在倍增。

这就是 P2P Store 的核心思想：**第一个节点持火（Register），其他节点借火（GetReplica），借到火后也能传火（自动注册为数据源）**。参与传火的节点越多，总带宽越大，分发越快。用过迅雷，还有一些高校有自己的BT站，个人电脑接入BT软件后既可以下载别人分享的蓝光高清资源，也可以当做节点为其他人的下载提供带宽。

今天这篇文章，我们深入 P2P Store 的设计与实现。

---

### 核心架构：传火式 P2P

与 Mooncake Store 的 Master-Client 架构不同，P2P Store 采用**传火式 P2P 架构**——没有 Master，元数据通过 etcd 管理。etcd 就像一本"火种簿"，记录谁有火、火在哪里，但不参与传火本身。

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Trainer  │     │ Infer 0  │     │ Infer 1  │
│ (持火)    │     │(借火+传火)│     │(借火+传火)│
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     │   Register     │  GetReplica    │  GetReplica
     │  (登记火种)    │ (从Trainer借火) │ (从Infer0借火)
     │                │                │
     └────────────────┼────────────────┘
                      │
              ┌───────▼───────┐
              │     etcd      │
              │  (火种簿)      │
              └───────────────┘
```

| 组件 | 职责 | 传火比喻 |
|------|------|---------|
| etcd | 存储文件元数据 | 火种簿——记录谁有火、火在哪里 |
| P2PStore 实例 | 既借火又传火 | 每根蜡烛——既能被点亮，也能点亮别人 |

---

### P2PStore 类：核心 API

P2P Store 用 Go 语言实现，定义在 `mooncake-p2p-store/src/p2pstore/core.go` 中。

##### 创建实例

```go
func NewP2PStore(metadataConnString string, localServerName string,
                 nicPriorityMatrix string) (*P2PStore, error)
```

创建实例时会：
1. 连接 etcd 元数据服务
2. 初始化本地 Transfer Engine
3. 注册本地 Segment

##### Register：持火

```go
func (store *P2PStore) Register(ctx context.Context,
    name string,           // 文件名
    addrList []uintptr,    // 内存地址列表
    sizeList []uint64,     // 每段大小
    maxShardSize uint64,   // 最大分片大小
    location string,       // 内存位置
    forceCreate bool,      // 是否强制创建
) error
```

Register 只把"我有火"这件事登记到火种簿（etcd），**不传火**。其他节点通过 GetReplica 借火时，才触发实际的数据传输。

##### GetReplica：借火

```go
func (store *P2PStore) GetReplica(ctx context.Context,
    name string,           // 文件名
    addrList []uintptr,    // 本地内存地址
    sizeList []uint64,     // 每段大小
) error
```

GetReplica 的流程——一次完整的借火：

1. 查火种簿：从 etcd 查询哪些节点有火（文件的元数据）
2. 选一个有火的节点作为火源
3. 通过 Transfer Engine 借火（拉取数据到本地）
4. 借到火后，自己也登记到火种簿——成为新的传火者

##### 其他操作

```go
// 熄灯——不再提供数据给他人
func (store *P2PStore) Unregister(ctx context.Context, name string) error

// 列出已注册的文件
func (store *P2PStore) List(ctx context.Context, namePrefix string) ([]PayloadInfo, error)

// 灭火——删除本地副本，不再传火
func (store *P2PStore) DeleteReplica(ctx context.Context, name string) error
```

---

### 分片机制：大火把拆成小火把

一根大火把（1T 模型权重）太大，一次传不完。P2P Store 把大火把拆成 **Shard**（小火把），每根小火把独立传火。

##### 分片策略

```
1T 模型权重文件
    │
    ▼ 切分为 maxShardSize 的 Shard
┌──────────┬──────────┬──────────┬─────┐
│ Shard 0  │ Shard 1  │ Shard 2  │ ... │
│ (4 GB)   │ (4 GB)   │ (4 GB)   │     │
└────┬─────┴────┬─────┴────┬─────┴─────┘
     │          │          │
     ▼          ▼          ▼
  Transfer    Transfer   Transfer
  Engine      Engine     Engine
```

```go
const MAX_CHUNK_SIZE uint64 = 4096 * 1024 * 1024  // 4GB
```

##### 分片传输

```go
func (store *P2PStore) performTransfer(ctx context.Context,
    source uintptr, shard Shard) error {
    // 重试循环，最多 max(3, shard.Count()) 次
    for retry := 0; retry < maxRetry; retry++ {
        // 1. 分配批量传输 ID
        // 2. 打开源 Segment
        // 3. 提交传输请求
        // 4. 轮询状态直到完成
        // 5. 失败则重试
    }
}
```

---

### 传火式分发模型：火种不灭，越传越快

P2P Store 的分发模型遵循传火的三个本质特征：

| 本质特征 | 传火世界 | P2P Store | 说明 |
|---------|---------|-----------|------|
| 火种不消耗 | 点亮别人的蜡烛，自己的火焰不变暗 | 复制数据给别人，自己的数据不减少 | 数据是"非竞争性"资源 |
| 借到火即传火 | 被点亮的蜡烛也能去点别人 | GetReplica 后自动注册为数据源 | 每个接收者都变成发送者 |
| 传火者越多越快 | 10 根蜡烛比 1 根更快点亮剩余的 | 10 个数据源比 1 个更快分发 | 总带宽随参与者倍增 |

##### 分发加速效果

```
1 根蜡烛逐个传火:
  蜡烛 0 → 蜡烛 1: 100s
  蜡烛 0 → 蜡烛 2: 100s  (串行)
  蜡烛 0 → 蜡烛 3: 100s
  总时间: ~300s

传火式（每根被点亮的蜡烛都参与传火）:
  蜡烛 0 → 蜡烛 1: 100s
  蜡烛 0+1 → 蜡烛 2: 50s   (两个火源)
  蜡烛 0+1+2 → 蜡烛 3: 33s (三个火源)
  总时间: ~183s
```

> 笔者注：在实际生产中，Kimi-K2 模型（1T 参数）的权重更新在数千张 GPU 上只需约 20 秒。这个速度主要得益于 RDMA 的高带宽和传火式的并行分发——每张 GPU 拿到权重后立刻成为新的火源，数千个火源同时传火，速度远超单点分发。

---

### Catalog：文件目录管理

```go
// mooncake-p2p-store/src/p2pstore/catalog.go
type Catalog struct {
    // 文件元数据管理
}
```

Catalog 管理 etcd 中的文件元数据，包括：

| 元数据 | 说明 | 传火比喻 |
|--------|------|---------|
| 文件名 | 唯一标识 | 火种名称 |
| 分片列表 | 每个分片的大小和位置 | 每根火把的位置 |
| 持火节点 | 哪些节点持有这个文件 | 谁有火 |
| 内存位置 | 每个分片的内存地址 | 火把的具体位置 |

元数据键格式：`mooncake/checkpoint/{name}`

---

### 部署流程

##### 典型部署步骤

```bash
# 1. 启动 etcd（火种簿）
etcd --listen-client-urls http://0.0.0.0:2379

# 2. 启动训练节点（持火）
# 在训练代码中：
store, _ := p2pstore.NewP2PStore("etcd://127.0.0.1:2379", "trainer-node")
store.Register(ctx, "kimi-k2-weights", addrList, sizeList, 4*1024*1024*1024, "dram")

# 3. 启动推理节点（借火）
# 在推理代码中：
store, _ := p2pstore.NewP2PStore("etcd://127.0.0.1:2379", "infer-node-0")
store.GetReplica(ctx, "kimi-k2-weights", localAddrList, localSizeList)
```

##### 实际生产场景

```
训练完成 → 新权重写入 GPU 内存
    │
    ▼ Register (登记火种，不传数据)
    │
    ├── 推理节点 0: GetReplica → 从训练节点借火 → 成为新的传火者
    ├── 推理节点 1: GetReplica → 从训练节点 + 推理节点 0 借火 → 成为新的传火者
    ├── 推理节点 2: GetReplica → 从多个火源并行借火 → 成为新的传火者
    └── ...
    │
    ▼ 所有推理节点都有新权重（整间屋子亮了）
    │
    ▼ 切换推理流量到新权重
```

---

### P2P Store vs PD 直传：两种"快递"模式的代码级对比

P2P Store 和 PD 直传（MooncakeConnector）都用 Transfer Engine 做 RDMA 数据搬运，但"点火方式"完全不同。用传火来比喻：

- **P2P Store** = 通过火种簿（etcd）传火——持火者把火种信息登记到火种簿，借火者查火种簿找到火源，自己去借火，借到后也登记到火种簿
- **PD 直传** = 直接递火把——持火者直接把火把递到对方手上，不查火种簿，不登记

##### 架构差异：火种簿传火 vs 直接递火

```
P2P Store 模式:

  Prefill 节点                etcd                Decode 节点
  ┌──────────┐           ┌─────────┐          ┌──────────┐
  │ Register │──元数据──→│  etcd   │          │          │
  │ (持火)    │           │ (火种簿) │←──查询──│ GetReplica│
  │          │           │         │          │ (借火)    │
  │          │←──RDMA READ───────────────────│          │
  │          │           │         │──CAS更新─│→ 登记为传火者│
  └──────────┘           └─────────┘          └──────────┘

  元数据路径: P → etcd ← D        (两跳)
  数据路径:   D → P (RDMA READ)   (D 借火)
  传火登记:   D → etcd            (额外一跳)


PD 直传模式:

  Prefill 节点              ZMQ              Decode 节点
  ┌──────────┐          ┌─────┐          ┌──────────┐
  │          │←─元数据──│ ZMQ │←─元数据──│ receive_kv│
  │          │          └─────┘          │ (报地址)  │
  │ send_kv  │─────RDMA WRITE──────────→│          │
  │ (推送)    │───ZMQ ACK──→             │          │
  └──────────┘                          └──────────┘

  元数据路径: D → ZMQ → P              (一跳，内存级延迟)
  数据路径:   P → D (RDMA WRITE)       (P 推送)
  无额外元数据操作
```

##### 元数据交换：etcd 事务 vs ZMQ 纸条

P2P Store 的元数据全部经过 etcd，使用乐观并发控制（CAS）：

```go
// mooncake-p2p-store/src/p2pstore/metadata.go
// 更新 Payload 元数据（添加副本位置）
func (m *Metadata) Update(ctx context.Context, key string,
    value *Payload, revision int64) error {
    txn := m.client.Txn(ctx).
        If(clientv3.Compare(clientv3.ModRevision(key), "=", revision)).
        Then(clientv3.OpPut(key, string(data)))
    // CAS: 如果 revision 不匹配（被其他人改过），则更新失败，需要重试
}
```

PD 直传的元数据通过 ZMQ 直接传递，无需任何并发控制：

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
# D 节点通过 ZMQ 把自己的地址信息发给 P 节点
class MooncakeAgentMetadata(msgspec.Struct):
    remote_hostname: str          # D 节点的 IP
    remote_port: int              # D 节点的 Transfer Engine RPC 端口
    request_ids: list[ReqId]      # 需要传输的请求 ID
    kv_caches_base_addr: list[int]  # D 节点 KV Cache 张量的基地址
    block_ids: list[list[int]]    # D 节点上的 block ID
```

| 维度 | P2P Store (etcd) | PD 直传 (ZMQ) |
|------|-----------------|---------------|
| 传输方式 | etcd 分布式 KV 存储 | ZMQ REQ/ROUTER 内存消息 |
| 延迟 | ~1-5 ms（网络 RTT） | ~0.1 ms（本地消息） |
| 并发控制 | etcd ModRevision CAS | 无需（点对点，无竞争） |
| 持久性 | 持久化（重启不丢） | 临时性（请求完成即消） |
| 编码 | JSON | msgspec/msgpack（二进制） |
| 额外依赖 | etcd 集群 | 无 |

##### 段发现：etcd 查询 vs P2PHANDSHAKE 直连

Transfer Engine 需要知道远程节点的 RDMA 信息（QP 号、MR 密钥、缓冲区地址）才能发起传输。两种模式获取这些信息的方式完全不同：

P2P Store 模式——通过 etcd 查询：

```cpp
// mooncake-transfer-engine/src/transfer_metadata.cpp
// 正常模式: 从 storage_plugin_ (etcd) 查询段描述符
Status TransferMetadata::getSegmentDesc(
    const std::string& segment_name,
    SegmentDesc& segment_desc) {
    // 从 etcd 读取远程节点的 RDMA 信息
    return storage_plugin_->getSegmentDesc(segment_name, segment_desc);
}
```

PD 直传模式——P2PHANDSHAKE 直连：

```cpp
// mooncake-transfer-engine/src/transfer_metadata.cpp
// P2PHANDSHAKE 模式: 跳过 storage_plugin_，直接 TCP 握手
if (conn_string == P2PHANDSHAKE) {
    p2p_handshake_mode_ = true;
    return;  // 不创建 storage_plugin_！
}

// getSegmentDesc 变成直接握手
Status TransferMetadata::getSegmentDesc(
    const std::string& segment_name,
    SegmentDesc& segment_desc) {
    if (p2p_handshake_mode_) {
        // segment_name 格式为 "ip:port"
        // 直接 TCP 连接对方，交换 RDMA 元数据
        return handshake_plugin_->exchangeMetadata(
            ip, rpc_port, local_attrs, peer_attrs);
    }
}
```

```
P2P Store 段发现:
  D 节点 → etcd 查询 → 获取 P 节点的 RDMA 信息 → 建立 RDMA 连接

PD 直传段发现:
  P 节点 → TCP 直连 D 节点(ip:port) → 交换 RDMA 信息 → 建立 RDMA 连接
  （首次 ~1-10 ms 握手，后续缓存）
```

##### 数据传输方向：拉取 vs 推送

这是最本质的差异之一——P2P Store 是**消费者拉取**（READ），PD 直传是**生产者推送**（WRITE）。

P2P Store——GetReplica 拉取：

```go
// mooncake-p2p-store/src/p2pstore/core.go
func (store *P2PStore) performTransfer(ctx context.Context,
    source uintptr, shard Shard) error {
    // OPCODE_READ: D 节点主动从 P 节点拉取数据
    transferRequest := TransferRequest{
        Opcode:     OPCODE_READ,
        Source:     source,                    // 本地缓冲区地址
        TargetID:   segment_id,                // 远程段 ID
        TargetOffset: remote_offset,            // 远程偏移
        Length:     shard.Length,
    }
    store.transfer.submitTransfer(batchID, []TransferRequest{transferRequest})
}
```

PD 直传——batch_transfer_sync_write 推送：

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
def _send_blocks(self, remote_session, src_ptrs, dst_ptrs, lengths):
    # OPCODE_WRITE: P 节点主动推送到 D 节点
    ret_value = self.engine.batch_transfer_sync_write(
        remote_session, src_ptrs, dst_ptrs, lengths)
```

```cpp
// mooncake-transfer-engine/src/transfer_engine_py.cpp
int TransferEnginePy::batchTransferSyncWrite(...) {
    // 构造 WRITE 请求
    TransferRequest request;
    request.opcode = OPCODE_WRITE;  // 推送模式
    request.source = local_kv_addr;
    request.target_id = remote_segment_handle;
    request.target_offset = remote_kv_addr;
    // ...
}
```

为什么 PD 直传选择推送而不是拉取？因为**只有 P 节点知道数据准备好了**。Prefill 完成后，P 节点立即推送，D 节点无需轮询等待。而 P2P Store 的场景是"数据已就绪，谁需要谁来取"——拉取更自然。

##### 内存管理：自分配 vs 借用

P2P Store 自己分配和管理内存，PD 直传借用推理框架（vLLM）的 KV Cache 张量。

P2P Store——自分配 + 分片注册：

```go
// mooncake-p2p-store/src/p2pstore/registered_memory.go
func (rm *RegisteredMemory) Add(addr uintptr, size uint64) error {
    // 按 MAX_CHUNK_SIZE (4GB) 分片
    for offset := uint64(0); offset < size; offset += MAX_CHUNK_SIZE {
        chunkSize := min(MAX_CHUNK_SIZE, size-offset)
        // 每个分片独立注册到 Transfer Engine
        rm.engine.registerLocalMemory(addr+offset, chunkSize, location)
    }
}
```

PD 直传——批量注册 vLLM 张量：

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
def register_kv_caches(self, kv_caches):
    kv_data_ptrs = [base_addr for tensor in kv_caches]
    kv_data_lens = [tensor_size_bytes for tensor in kv_caches]
    # 批量注册 vLLM 已有的 KV Cache 张量
    self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

```cpp
// mooncake-transfer-engine/src/transfer_engine_py.cpp
int TransferEnginePy::batchRegisterMemory(
    std::vector<uintptr_t> buffer_addresses,
    std::vector<size_t> capacities,
    const std::string& location) {
    // 一次性注册所有缓冲区
    return engine_->registerLocalMemoryBatch(buffers, location);
}
```

| 维度 | P2P Store | PD 直传 |
|------|----------|---------|
| 内存所有者 | P2P Store 自己分配 | vLLM 拥有，Connector 借用 |
| 注册方式 | 逐分片注册（4GB 一片） | 批量注册（一次注册全部张量） |
| 生命周期 | Register/Unregister 显式管理 | 进程生命周期，无需手动释放 |
| 引用计数 | 有（BufferHandle.refCount） | 无（张量随进程存活） |

##### 连续块合并：PD 直传的独有优化

PD 直传有一个 P2P Store 没有的优化——**连续块合并**。vLLM 的 KV Cache 由固定大小的 block 组成，相邻 block 在内存中是连续的。PD 直传会把连续的 block 合并为一次 RDMA WRITE，减少传输次数：

```python
# mooncake-wheel/mooncake/mooncake_connector_v1.py
async def _build_transfer_params(self, send_reqs, agent_meta):
    # 合并连续 block 为一次传输
    for layer_idx in range(num_layers):
        src_ptr = local_base_addr + local_block_id * block_len
        dst_ptr = remote_base_addr + remote_block_id * block_len
        # 如果下一个 block 也是连续的，合并长度
        length = block_len * num_consecutive_blocks
```

P2P Store 没有这个优化——它按 Shard（最大 4GB）传输，不关心 Shard 内部的 block 连续性。因为 P2P Store 的场景是传输大块模型权重，不像 KV Cache 那样有细粒度的 block 结构。

##### 完整传输流程对比

P2P Store 传输流程：

```
1. P 节点 Register() → 元数据写入 etcd
2. D 节点 GetReplica() → 从 etcd 读取元数据
3. D 节点 openSegment() → 从 etcd 获取 P 的 RDMA 信息
4. D 节点 submitTransfer(READ) → RDMA READ 从 P 拉取数据
5. D 节点 updatePayloadMetadata() → CAS 更新 etcd（D 登记为传火者）

元数据操作: 3 次 etcd 交互 (写 + 读 + CAS)
数据操作:   1 次 RDMA READ
```

PD 直传传输流程：

```
1. P 节点 Prefill 完成 → request_finished() 返回 kv_transfer_params
2. 代理服务器转发请求到 D 节点（携带 P 的 ZMQ 地址）
3. D 节点 receive_kv() → 通过 ZMQ 发送 MooncakeAgentMetadata 给 P
4. P 节点收到元数据 → openSegment(P2PHANDSHAKE) → TCP 握手获取 D 的 RDMA 信息
5. P 节点 batch_transfer_sync_write() → RDMA WRITE 推送到 D
6. P 节点通过 ZMQ 发送 TRANS_DONE 确认

元数据操作: 1 次 ZMQ 消息（~0.1 ms）
数据操作:   1 次批量 RDMA WRITE
```

##### 全景对比表

| 维度 | P2P Store | PD 直传 (MooncakeConnector) |
|------|----------|---------------------------|
| **场景** | 模型权重分发、KV Cache 快照 | 实时 PD 解耦推理 |
| **架构** | etcd 中心化元数据 | 无中心，ZMQ 直连 |
| **元数据服务** | etcd 集群 | 无（ZMQ 内存消息） |
| **段发现** | etcd 查询 | P2PHANDSHAKE TCP 握手 |
| **传输方向** | D 拉取（OPCODE_READ） | P 推送（OPCODE_WRITE） |
| **传输粒度** | Shard（最大 4GB） | Block（连续块合并） |
| **内存管理** | 自分配 + 分片注册 | 借用 vLLM 张量 + 批量注册 |
| **传火机制** | 借火后自动登记为传火者 | 无（一次性传输） |
| **并发控制** | etcd CAS | 无需 |
| **额外依赖** | etcd 集群 | 代理服务器 |
| **元数据延迟** | ~1-5 ms | ~0.1 ms |
| **语言** | Go（CGo 调用 C 引擎） | Python（pybind11 调用 C++ 引擎） |
| **适用拓扑** | 任意节点间 | 已知 P-D 配对 |

##### 选择指南

| 你的场景 | 选择 | 原因 |
|---------|------|------|
| 训练后权重分发到推理集群 | P2P Store | 传火模式让分发越传越快 |
| PD 解耦的实时推理 | PD 直传 | 零额外依赖，最低延迟 |
| KV Cache 快照持久化 | P2P Store | etcd 持久化，重启可恢复 |
| 多节点读取同一份数据 | P2P Store | 传火模式天然支持多读者 |
| 单次点对点 KV 传输 | PD 直传 | 无需部署 etcd，架构简单 |

> 笔者注：两种模式不是互斥的——同一个集群可以同时运行 P2P Store（用于权重分发）和 PD 直传（用于实时推理）。P2P Store 管"冷数据传火"，PD 直传管"热数据直递"，各司其职。

---

### 与 Mooncake Store 的对比

P2P Store 和 Mooncake Store 是 Mooncake 体系中两个互补的存储系统。用传火比喻来说：**Mooncake Store 是中央仓储，货物进出都要记账；P2P Store 是传火网络，火种自由传递，只记谁有火**。

##### 架构对比

| 维度 | Mooncake Store | P2P Store |
|------|---------------|-----------|
| 架构 | Master-Client | 传火式 P2P |
| 元数据服务 | Master 进程（单点决策） | etcd（分布式共识） |
| 数据模型 | KV Cache 对象（按 key 读写） | 文件/权重（整体分发） |
| 一致性 | 强一致（两阶段 Put） | 最终一致（CAS 更新） |
| 副本管理 | Master 协调分配 | 自动（借火即传火） |
| 语言 | C++/Go/Rust | Go |

##### 性能影响对比

两种架构的设计差异，直接导致了性能特征的根本不同：

| 性能维度 | Mooncake Store | P2P Store | 原因 |
|---------|---------------|-----------|------|
| **首次写入延迟** | ~5-15 ms（两阶段 Put: 申请→写入→确认） | ~1-3 ms（Register 只写 etcd 元数据） | Store 要等 Master 分配空间并确认；P2P Store 只登记"我有数据" |
| **首次读取延迟** | ~2-5 ms（查询 Master → 定位 → RDMA 读取） | ~5-15 ms（查询 etcd → 定位 → RDMA 读取 → CAS 更新做种） | P2P Store 的 GetReplica 需要额外写回 etcd 注册为传火者 |
| **大规模分发吞吐** | 受限于 Master 单点（所有分配经 Master） | 随参与者倍增（传火者越多越快） | Store 是 1 对 N；P2P Store 是 N 对 N |
| **并发写入** | 强一致，但 Master 是瓶颈 | CAS 乐观并发，冲突时重试 | Store 的 Master 串行化决策；P2P Store 的 etcd 支持并发但可能冲突 |
| **元数据开销** | 每次操作 1 次 Master RPC | 每次操作 2-3 次 etcd 交互 | P2P Store 需要额外 CAS 更新做种信息 |
| **Master/etcd 故障** | Master 宕机 → 集群不可用 | etcd 集群容忍少数节点故障 | etcd 基于 Raft 共识，3 节点容忍 1 节点故障 |

用一个具体数字来感受差异——分发 Kimi-K2 模型（1T 参数，约 2TB 权重数据）到 1000 张 GPU：

```
Mooncake Store 方式:
  每次写入: Master 分配 → Client 写入 → Master 确认
  1000 个 Client 同时写入 → Master 成为瓶颈
  估算: ~5-10 分钟（Master 串行化分配决策）

P2P Store 方式:
  Trainer Register → 1 个持火者
  推理节点 0 借火 → 2 个持火者
  推理节点 0+1 借火 → 4 个持火者
  推理节点 0+1+2+3 借火 → 8 个持火者
  ...指数级加速
  实际: ~20 秒（传火者倍增，带宽倍增）
```

##### 应用场景对比

| 场景 | Mooncake Store | P2P Store | 选择理由 |
|------|---------------|-----------|---------|
| **PD 解耦推理中 KV Cache 传递** | 适合 | 不适合 | KV Cache 生命周期短、读写频繁，需要强一致的 Put/Get |
| **训练后权重分发到推理集群** | 不适合 | 适合 | 一次性分发、只读、多读者，传火模式带宽倍增 |
| **KV Cache 跨节点共享（前缀缓存）** | 适合 | 不适合 | 需要按 key 精确读写、分层存储、驱逐策略 |
| **模型权重热更新（不停服换版本）** | 不适合 | 适合 | 权重整体分发、分发完切换流量、无需逐 key 操作 |
| **多租户 KV Cache 隔离** | 适合 | 不适合 | Master 支持租户配额、空间隔离 |
| **推理集群冷启动（加载权重）** | 不适合 | 适合 | 新节点加入时从最近的传火者借火，不经过中心 |
| **KV Cache 快照持久化到 SSD** | 适合 | 不适合 | Store 有分层存储（DRAM→SSD→分布式），P2P Store 只管内存 |

##### 典型生产部署

在 Mooncake 的实际生产部署中，两个系统各司其职、协同工作：

```
┌─────────────────────────────────────────────────────────────┐
│                     推理集群                                   │
│                                                               │
│  ┌─────────────────────────────────────────────┐             │
│  │  Mooncake Store (持续运行)                     │             │
│  │  • PD 解耦时 KV Cache 从 P 传到 D              │             │
│  │  • 前缀缓存跨节点共享                           │             │
│  │  • KV Cache 分层存储 (DRAM ↔ SSD)              │             │
│  └─────────────────────────────────────────────┘             │
│                                                               │
│  ┌─────────────────────────────────────────────┐             │
│  │  P2P Store (按需触发)                         │             │
│  │  • 训练完成后新权重传火到所有推理节点              │             │
│  │  • 新节点冷启动时从最近节点借火                   │             │
│  │  • 模型版本热更新                               │             │
│  └─────────────────────────────────────────────┘             │
│                                                               │
│  时间线:                                                       │
│  ──┬──────┬──────┬──────┬──────┬──────┬──────→ 时间           │
│    │      │      │      │      │      │                       │
│    训练完成  推理中  推理中  推理中  训练完成  推理中              │
│    ↓      ↓      ↓      ↓      ↓      ↓                     │
│    P2P    Store   Store   Store  P2P    Store                 │
│    传火    传KV    传KV    传KV   传火    传KV                  │
│    (20s)  (持续)  (持续)  (持续) (20s)  (持续)                 │
└─────────────────────────────────────────────────────────────┘
```

> 笔者注：P2P Store 和 Mooncake Store 解决的是不同场景的问题。P2P Store 适合"一次性大块分发"（如训练后更新权重），而 Mooncake Store 适合"持续精细读写"（如推理时 KV Cache 共享）。一个管"火种传播"，一个管"仓储管理"——火种传播解决"怎么让所有人最快拿到"，仓储管理解决"怎么高效地存取和调度"。

---

### 设计哲学：传火的三大原则

| 原则 | 含义 | 传火比喻 | 典型体现 |
|------|------|---------|---------|
| **火种不灭** | 数据复制不消耗源 | 点亮别人不灭自己 | Register 只登记元数据，不移动数据 |
| **借火即传火** | 拿到数据后自动成为数据源 | 被点亮的蜡烛也能点别人 | GetReplica 后自动注册 |
| **火把拆分** | 大文件切分传输，充分利用带宽 | 大火把拆成小火把，多路并行 | 4GB Shard，独立传输 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| Register | 持火——登记火种到火种簿，不传数据 |
| GetReplica | 借火——从持火者拉取，借到后自动成为传火者 |
| Shard | 火把——大文件分片，4GB 一根，独立传火 |
| 传火模型 | 火种不灭，借火即传火，传火者越多越快 |
| etcd | 火种簿——记录谁有火，替代 Master |

**建议**: 使用 P2P Store 分发权重时，`maxShardSize` 设为 4GB（默认值）——太小的 Shard 会增加元数据开销，太大的 Shard 会降低并行度。

**延伸阅读**：
- P2P Store 设计文档：https://kvcache-ai.github.io/Mooncake/design/p2p-store.html
- Mooncake P2P Store 开源公告：https://github.com/MoonshotAI/checkpoint-engine/

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
