# Mooncake PD 分离详解（PD两阶段解耦）：让 Prefill 和 Decode 各自为战，各自提升

> **系列**: Mooncake 技术博客系列 | **类型**: 核心概念深潜篇
>
> 传统推理就像一家餐厅，厨师（Prefill）和服务员（Decode）共用一个厨房——厨师炒菜时油烟呛到服务员，服务员摆盘时又挡住厨师的路。PD 解耦就是厨师留在厨房，服务员留在备餐间，中间用传菜窗口（Transfer Engine）连接。

在当下的LLM推理业界，PD分离已是共识，并且是非常成熟的方案了，Prefill是计算密集，Decode是缓存密集，各自负载特征不同，生产系统流量洪流冲击，为互不影响，自然将其解耦，各自提升。

前面解释推理引擎 vLLM 时已介绍过，架构理念是通用的，不管是对推理引擎，还是对KV Cache存储系统。PD分离后，自然需要推理引擎和KV Cache系统共同支持，再拼成一个完整的Transformer机制。

Prefill和Decode的分离部署，需要Prefill和Decode的KV Cache数据，跨节点传输，不仅涉及到计算调度，还涉及到数据存储。

前面写 vLLM 系列文章的时候写过PD分离，那是推理引擎的视角，本文将从Mooncake的实现，完成KV Cache高速数据传输以及共享前缀数据存储的闭环。

---

### 引言

想象一家繁忙的餐厅：厨师负责备菜和大火爆炒（Prefill，计算密集），服务员负责精细摆盘和送餐（Decode，内存带宽密集）。如果他们共用一个房间，厨师颠勺时要小心防止碰到服务员，服务员摆盘又在厨师间绕来绕去——各自都耍不开，不自如，效率大打折扣。

PD 解耦的核心思想就是**厨师留在厨房，服务员留在备餐间，中间用传菜窗口（Transfer Engine）连接**。Prefill 集群专注于计算——快速处理输入 Token 生成 KV Cache；Decode 集群专注于生成——利用 KV Cache 逐 Token 输出。中间通过 Mooncake Transfer Engine 的传菜窗口传递 KV Cache，让两个集群各按各的节奏运转。

今天这篇文章，我们深入 PD 解耦服务的设计与实现。

---

### 为什么需要 PD 解耦？

##### Prefill 与 Decode 的资源特征的巨大差异

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特征 | 计算密集（矩阵乘法） | 内存带宽密集（KV Cache 读取） |
| 批次大小 | 较小（请求级） | 较大（Token 级连续批处理） |
| 延迟关注 | TTFT（首 Token 延迟） | TBT（Token 间延迟） |
| GPU 利用率 | 高（大量并行计算） | 低（受 KV Cache 内存限制） |
| 最优 GPU 配置 | 高算力 GPU | 高显存 GPU |

##### 耦合架构的三大问题

| 问题 | 说明 |
|------|------|
| 资源争抢 | Prefill 占满计算资源时 Decode 排队，反之亦然 |
| 无法独立扩展 | 需要更多 Prefill 能力时必须加整卡，浪费 Decode 资源 |
| 调度冲突 | Prefill 和 Decode 对调度策略的需求矛盾 |

##### 解耦后的收益

| 收益 | 说明 |
|------|------|
| 独立扩展 | Prefill 和 Decode 按需独立扩缩容 |
| 专用优化 | Prefill 用高算力卡，Decode 用高显存卡 |
| 调度解耦 | Prefill 批量处理，Decode 连续批处理，互不干扰 |
| KV Cache 复用 | 跨实例共享 KV Cache，相同前缀只计算一次 |

> 笔者注：Mooncake 论文和系统实践显示，在真实负载下 PD 解耦架构让 Kimi 的请求吞吐提升了 75%。这个数字不是来自"更快地计算"，而是来自"更聪明地分配资源"——让 Prefill 和 Decode 各自在最擅长的场景下运行。

---

### PD 解耦架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                     负载均衡器                                │
│              (请求路由: Prefill → Decode)                     │
└───────────┬─────────────────────────────┬───────────────────┘
            │                             │
            ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│   Prefill 集群        │     │   Decode 集群          │
│  ┌─────┐ ┌─────┐     │     │  ┌─────┐ ┌─────┐     │
│  │ P0  │ │ P1  │     │     │  │ D0  │ │ D1  │     │
│  │GPU  │ │GPU  │     │     │  │GPU  │ │GPU  │     │
│  └──┬──┘ └──┬──┘     │     │  └──┬──┘ └──┬──┘     │
│     │       │        │     │     │       │        │
│  ┌──▼───────▼──┐     │     │  ┌──▼───────▼──┐     │
│  │Transfer Eng.│     │     │  │Transfer Eng.│     │
│  └──────┬──────┘     │     │  └──────┬──────┘     │
└────────┼─────────────┘     └────────┼─────────────┘
         │                            │
         └──────── RDMA 网络 ─────────┘
                    │
         ┌──────────▼──────────┐
         │   Mooncake Store    │
         │  (分布式 KV Cache 池) │
         └─────────────────────┘
```

##### 请求处理流程

1. **请求到达** → 路由到 Prefill Worker
2. **Prefill 计算** → 生成 KV Cache
3. **KV Cache 传输** → 通过 Transfer Engine 写入 Store 或直传 Decode Worker
4. **Decode 生成** → 使用 KV Cache 逐 Token 生成
5. **Token 返回** → 流式返回给客户端

---

### 两种 KV Cache 传输模式

注意，两种模式不是二选一，一个是pd间直传，一个是分布式共享，作用场景不同。这一点在介绍 vLLM MultiConnector的时候提到过。

##### 模式一：直传模式（Transfer Engine Connector）

KV Cache 直接从 Prefill Worker 传到 Decode Worker，不经过 Store。

```
Prefill Worker                    Decode Worker
     │                                  │
     │── submitTransfer(WRITE) ────────►│
     │  (KV Cache via RDMA)             │
     │                                  │
```

适用场景：低延迟要求、无需跨实例复用。

##### 模式二：存储模式（Store Connector）

KV Cache 先写入 Mooncake Store，Decode Worker 从 Store 读取。

```
Prefill Worker       Mooncake Store       Decode Worker
     │                     │                     │
     │── Put(key, kv) ───►│                     │
     │                     │── Get(key) ────────►│
     │                     │                     │
```

适用场景：需要跨实例 KV Cache 复用、多轮对话、Agent 场景。

| 模式 | 延迟 | KV Cache 复用 | 适用场景 |
|------|------|-------------|---------|
| 直传 | 更低 | 不支持 | 单次推理 |
| 存储 | 略高 | 支持 | 多轮对话、Agent |

> 笔者注：在 Agent 场景中，同一个用户的多次请求通常有相同的 system prompt 前缀。存储模式下，system prompt 的 KV Cache 只需计算一次，后续请求直接命中缓存——这可以节省 30-50% 的 Prefill 计算量。

---

### 代码实现：MooncakeConnector 的完整传输流程

前面讲了架构和模式，现在我们钻进代码，看看一次完整的 PD KV Cache 传输到底是怎么发生的。核心实现全部在 `mooncake-wheel/mooncake/mooncake_connector_v1.py`（966 行）。

##### 三角色分工

MooncakeConnector 在构造时就按角色分流：

```python
# mooncake_connector_v1.py
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role):
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorScheduler(...)  # 调度侧
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(...)        # 工作侧
```

三个 KV 角色决定了线程模型的差异：

| 角色 | 谁来当 | 启动什么线程 |
|------|--------|-------------|
| `kv_producer` | Prefill 实例 | Sender 线程池 + ZMQ ROUTER 监听 |
| `kv_consumer` | Decode 实例 | Receiver 异步线程 |
| `kv_both` | 实验性双角色 | 两者都启动 |

```python
# kv_producer (Prefill) 启动发送线程
if self.kv_role != "kv_consumer":
    self._sender_executor = ThreadPoolExecutor(max_workers=10)
    self.sender_loop = asyncio.new_event_loop()

# kv_consumer (Decode) 启动接收线程
if self.kv_role != "kv_producer":
    self.receiver_loop = asyncio.new_event_loop()
```

> 笔者注：Sender 线程池默认 10 个 worker，但实际并发任务数是 20（`num_sender_tasks = num_sender_workers * 2`）。2x 是一个经典的超配策略——发送任务中有异步等待（等 `ready` 事件），超配可以防止线程池空闲。

##### 一次完整传输的七步流程

```
┌──────────┐     ①请求      ┌──────────┐     ③kv_transfer_params     ┌──────────┐
│  Proxy   │───────────────►│ Prefill  │────────────────────────────►│  Decode  │
│ (路由)   │                │ (厨房)    │                              │ (备餐间)  │
└──────────┘                └─────┬─────┘                              └────┬─────┘
                                  │                                         │
                            ② Prefill计算                              ⑥ RDMA写入
                            生成KV Cache                               KV Cache直达
                                  │                                         │
                            ④ ZMQ通知:                                   │
                            "我准备好了"                                  │
                                  │                                         │
                                  └────────── ⑤ Transfer Engine ──────────┘
                                               (传菜窗口/RDMA直传)
```

**① Proxy 路由请求到 Prefill**

Proxy（`vllm_v1_proxy_server.py`）收到请求后，先发给 Prefill，并注入 `kv_transfer_params`：

```python
# vllm_v1_proxy_server.py
req_data["kv_transfer_params"] = {
    "do_remote_decode": True,    # 告诉 Prefill: 完成后把 KV Cache 传走
    "do_remote_prefill": False,
}
req_data["max_tokens"] = 1       # Prefill 只做预填充，不生成
```

> 笔者注：`max_tokens = 1` 是关键——Prefill 只负责计算 KV Cache，不做任何生成。这就像厨师只管炒菜不管端盘，炒完放传菜窗口就完事。

**② Prefill 完成计算，触发发送**

Prefill 的 Scheduler 在 `request_finished()` 中判断：如果 `do_remote_decode=True` 且请求正常结束，就延迟释放 KV Cache 块，并返回自己的地址信息：

```python
# MooncakeConnectorScheduler.request_finished()
if params.get("do_remote_decode") and request.status == FINISHED_LENGTH_CAPPED:
    delay_free_blocks = len(block_ids) > 0
    if delay_free_blocks:
        self._reqs_need_send[request.request_id] = block_ids  # 暂不释放！
    return delay_free_blocks, dict(
        do_remote_prefill=True,      # 告诉 Decode: 从远程拉取
        do_remote_decode=False,
        remote_host=self.side_channel_host,   # Prefill 的 IP
        remote_port=self.side_channel_port,   # Prefill 的端口
    )
```

**③ Proxy 转发 kv_transfer_params 给 Decode**

Proxy 从 Prefill 的响应中提取 `kv_transfer_params`，转发给 Decode 实例：

```python
# vllm_v1_proxy_server.py
response_json = response.json()
kv_transfer_params = response_json.get("kv_transfer_params", {})
if kv_transfer_params:
    req_data["kv_transfer_params"] = kv_transfer_params  # 带上 Prefill 的地址
```

**④ Decode 通过 ZMQ 向 Prefill 发送元数据**

Decode 的 Worker 在 `start_load_kv()` 中，构造 `MooncakeAgentMetadata` 并通过 ZMQ REQ 发给 Prefill：

```python
# MooncakeConnectorWorker.receive_kv()
metadata = MooncakeAgentMetadata(
    remote_hostname=self.hostname,         # Decode 的 IP
    remote_port=self.rpc_port,             # Decode 的 Transfer Engine 端口
    request_ids=remote_req_ids,            # 请求 ID
    kv_caches_base_addr=self.kv_caches_base_addr,  # Decode 端 KV Cache 基地址
    block_ids=block_ids,                   # Decode 端分配的 block ID
)
encoded_data = self._encoder.encode(metadata)  # msgpack 编码
await sock.send(encoded_data)                   # ZMQ REQ 发送
ret_msg = await sock.recv()                     # 等待 TRANS_DONE 确认
```

**⑤ Prefill 收到请求，构建传输参数**

Prefill 的 ZMQ ROUTER 收到 Decode 的元数据后，`send_kv_to_decode()` 构建 RDMA 写入参数：

```python
# MooncakeConnectorWorker._build_transfer_params()
for local_layer_addr, remote_layer_addr in zip(local_base_addr, remote_base_addr):
    for group_local_block_id, group_remote_block_id in zip(...):
        src_ptrs.append(local_layer_addr + group_local_block_id[0] * block_len)
        dst_ptrs.append(remote_layer_addr + group_remote_block_id[0] * block_len)
        lengths.append(block_len * len(group_local_block_id))
```

> 笔者注：`group_consecutive_contiguous()` 是一个关键优化——用 NumPy 向量化操作把连续的 block ID 合并成大段传输，减少 RDMA 写操作次数。比如 block [3,4,5] → block [10,11,12] 合并为一次写入，而非三次。

**⑥ Transfer Engine 执行 RDMA 直写**

```python
# MooncakeConnectorWorker._send_blocks()
ret_value = self.engine.batch_transfer_sync_write(
    remote_session,   # "decode-host:port"
    src_ptrs,         # Prefill GPU 内存地址列表
    dst_ptrs,         # Decode GPU 内存地址列表
    lengths,          # 每段长度列表
)
```

这是整个传输的核心——C++ Transfer Engine 通过 RDMA WRITE 直接从 Prefill GPU 写入 Decode GPU，数据不经过 CPU、不经过内核态。

**⑦ 传输完成，双方清理**

Prefill 发送 `TRANS_DONE` 确认，Decode 收到后标记请求完成，Prefill 延迟释放的 KV Cache 块终于可以回收。

##### 通信协议：ZMQ Side Channel

Prefill 和 Decode 之间的元数据交换使用 ZMQ，数据传输使用 Transfer Engine（RDMA/TCP）：

```
元数据通道 (ZMQ):
  Prefill: ROUTER socket 监听 side_channel_port + tp_rank
  Decode:  REQ socket  连接 prefill_host:side_channel_port + tp_rank
  协议: msgpack 编码的 MooncakeAgentMetadata

数据通道 (Transfer Engine):
  模式: P2PHANDSHAKE（点对点握手，无需 etcd）
  传输: batch_transfer_sync_write（RDMA 直写）
```

```python
# Transfer Engine 初始化
self.engine = TransferEngine()
ret_value = self.engine.initialize(
    self.hostname,
    "P2PHANDSHAKE",        # 点对点模式，不需要中心化元数据服务器
    VLLM_MOONCAKE_PROTOCOL, # "rdma" 或 "tcp"
    "",
)
```

> 笔者注：`P2PHANDSHAKE` 模式是专门为 PD 解耦设计的。传统模式需要 etcd/redis 作为元数据服务器来发现对端，但在 PD 场景中，只有 Prefill 和 Decode 两方需要互相发现，Proxy 已经在 `kv_transfer_params` 中传递了对方的地址，所以用点对点握手就够了——少一个外部依赖，部署更简单。

##### KV Cache 内存注册

在传输之前，必须把 GPU 上的 KV Cache 内存注册到 Transfer Engine：

```python
# MooncakeConnectorWorker.register_kv_caches()
kv_data_ptrs = []
kv_data_lens = []

for layer_name, cache in kv_caches.items():
    base_addr = cache.data_ptr()       # GPU 张量的内存地址
    kv_data_ptrs.append(base_addr)
    kv_data_lens.append(cache.nbytes)

self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

注册后，Transfer Engine 才能对这些内存区域发起 RDMA 操作。注意 MLA（Multi-head Latent Attention）模型和标准模型的区别——MLA 不拆分 K/V，标准模型需要拆分后分别注册：

```python
self.split_k_and_v = not (self.use_mla or self._use_pallas_v1 or self._use_flashinfer)
```

##### 超时与容错

Prefill 侧的 `reqs_need_send` 字典跟踪待发送的请求，每个请求带有过期时间：

```python
# 请求注册时设置超时
send_meta.expire_time = time.perf_counter() + VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT  # 默认 120 秒

# 定期检查过期请求
expired_reqs = [
    req_id for req_id, send_meta in self.reqs_need_send.items()
    if send_meta.expire_time < now
]
for req_id in expired_reqs:
    del self.reqs_need_send[req_id]  # 超时则强制释放 Prefill 侧的 KV Cache 块
```

> 笔者注：超时机制是防止"Decode 拉取请求丢失导致 Prefill 内存泄漏"的安全网。如果 Decode 崩溃或网络分区，Prefill 不会永远持有已完成的 KV Cache 块——120 秒后自动释放。这个时间窗口需要权衡：太短会导致正常传输被误杀，太长会浪费 Prefill 的显存。

---

### vLLM 集成：MooncakeConnector

Mooncake 通过 vLLM 的 KV Connector 接口集成 PD 解耦服务。

##### MooncakeTransferEngineConnector

用于直传模式，直接通过 Transfer Engine 传输 KV Cache：

```python
# vLLM 配置
--kv-transfer-config \
    '{"kv_connector":"MooncakeTransferEngineConnector",
      "kv_role":"kv_producer",  # Prefill Worker
      "kv_role":"kv_consumer",  # Decode Worker
      "mooncake_metadata_server":"etcd://127.0.0.1:2379",
      "mooncake_protocol":"rdma"}'
```

##### MooncakeStoreConnector

用于存储模式，通过 Mooncake Store 共享 KV Cache：

```python
# vLLM 配置
--kv-transfer-config \
    '{"kv_connector":"MooncakeStoreConnector",
      "kv_role":"kv_both",
      "mooncake_metadata_server":"etcd://127.0.0.1:2379",
      "mooncake_protocol":"rdma",
      "mooncake_master_addr":"127.0.0.1:50051"}'
```

##### 两种 Connector 对比

| 维度 | TransferEngineConnector | StoreConnector |
|------|------------------------|----------------|
| 传输路径 | Prefill → Decode 直传 | Prefill → Store → Decode |
| KV Cache 复用 | 不支持 | 支持（基于前缀哈希） |
| 额外组件 | 无 | Mooncake Store Master |
| 延迟 | 更低 | 略高（多一跳） |
| 适用场景 | 单次推理 | 多轮对话、Agent |

---

### SGLang 集成：HiCache 与 EPD

SGLang 的 PD 解耦集成更加深入，提供了三种模式：

##### 1. PD Disaggregated Serving

通过 Mooncake Transfer Engine 传输 KV Cache：

```
Prefill Worker ──── Transfer Engine ──── Decode Worker
```

##### 2. HiCache（分层 KV 缓存）

SGLang 的 RadixAttention 扩展为多级缓存，Mooncake Store 作为远程存储层：

```
┌───────────────────────────────────┐
│ Device Cache (GPU VRAM)           │  ← 最快，容量小
├───────────────────────────────────┤
│ Host Cache (CPU DRAM)             │  ← 较快，容量中
├───────────────────────────────────┤
│ Remote Cache (Mooncake Store)     │  ← 较慢，容量大
└───────────────────────────────────┘
```

##### 3. EPD（Encode-Prefill-Decode）

将多模态编码器也解耦出来：

```
Encoder Worker ── Transfer Engine ── Prefill Worker ── Transfer Engine ── Decode Worker
(ViT 编码)                         (LLM Prefill)                          (LLM Decode)
```

EPD 适用于多模态推理——计算密集的 Vision Transformer 可以部署在专用 GPU 上，通过 Mooncake 的 RDMA 引擎零拷贝传输大型嵌入向量。

---

### 性能考量：KV Cache 传输的开销

KV Cache 传输引入了额外的延迟。让我们量化这个开销：

| 模型 | KV Cache 大小 (128K Token) | RDMA 传输时间 (4×200Gbps) | 占 Prefill 时间比例 |
|------|--------------------------|-------------------------|-------------------|
| LLaMA3-8B | ~2 GB | ~23 μs | < 1% |
| LLaMA3-70B | ~40 GB | ~460 μs | ~5% |
| DeepSeek-V3 | ~80 GB | ~920 μs | ~8% |

> 笔者注：KV Cache 传输时间占 Prefill 时间比例很小（通常 < 10%），但 Decode 端的等待时间取决于传输何时完成。在直传模式下，Decode Worker 在 KV Cache 传输完成后立即开始生成，延迟增加约等于传输时间。

---

### 设计哲学：PD 解耦的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **计算与存储分离** | Prefill 和 Decode 使用不同资源 | Prefill 用高算力卡，Decode 用高显存卡 |
| **KV Cache 一等公民** | KV Cache 是连接两个集群的核心数据 | Store、Transfer Engine 都围绕 KV Cache 设计 |
| **传输透明** | 上层推理引擎无需关心 KV Cache 如何传输 | KV Connector 统一接口 |

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| PD 解耦 | Prefill 和 Decode 分离部署，各自独立扩展 |
| 直传模式 | KV Cache 直接从 Prefill 传到 Decode，延迟最低 |
| 存储模式 | KV Cache 写入 Store，支持跨实例复用 |
| MooncakeConnector | vLLM KV Connector 实现，Scheduler+Worker 双角色分流 |
| ZMQ Side Channel | 元数据交换通道，msgpack 编码，ROUTER/REQ 模式 |
| P2PHANDSHAKE | 点对点握手模式，PD 场景无需 etcd |
| 连续块合并 | NumPy 向量化合并连续 block，减少 RDMA 写次数 |
| HiCache | SGLang 的多级 KV 缓存，Store 作为远程层 |
| EPD | 多模态场景下编码器也解耦 |

**建议**: 如果你的场景有大量相同前缀（如 system prompt），使用存储模式 + 前缀缓存——这是 PD 解耦最大的收益来源。

即最佳实践是 vLLM 配置MultiConnector，同时配置MooncakeConnector和MooncakeStoreConnector，最大化推理系统性能和收益。

**延伸阅读**：
- vLLM Mooncake Connector 文档：https://docs.vllm.ai/en/latest/features/mooncake_connector_usage.html
- SGLang HiCache 博客：https://lmsys.org/blog/2025-09-10-sglang-hicache/

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*


