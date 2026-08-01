# Mooncake 生态集成：如何融入 vLLM、SGLang 和 TensorRT-LLM

> **系列**: Mooncake 技术博客系列 | **类型**: 生态实战篇
>
> Mooncake 不是一座孤岛——它是一条运河，连接着 LLM 推理的"内河航运"网络。vLLM、SGLang、TensorRT-LLM 是河上的"货轮"，Mooncake 让它们可以共享"货物"（KV Cache），协同"航行"（PD 解耦），甚至"编队"（弹性专家并行）。

---

### 引言

Mooncake 的价值不仅在于它自身有多快，而且在于它能让已有的推理引擎变得更快、更弹性、更可靠。目前 Mooncake 已经深度集成了 vLLM、SGLang、TensorRT-LLM 等主流推理引擎，以及 LMCache、LMDeploy、NIXL、TorchSpec、FlexKV 等生态项目。

今天这篇文章，我们梳理 Mooncake 的生态集成全景，看看它是如何融入 LLM 推理生态的。

---

### 集成架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                     上层推理引擎                              │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐     │
│  │  vLLM   │  │  SGLang │  │TRT-LLM   │  │ LMDeploy│     │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬────┘     │
└───────┼────────────┼────────────┼──────────────┼──────────┘
        │            │            │              │
┌───────▼────────────▼────────────▼──────────────▼──────────┐
│              mooncake-integration/                          │
│  ┌────────────────────┐  ┌──────────────────────────┐     │
│  │ Store Python 绑定   │  │ Transfer Engine Python 绑定│     │
│  │ (store_py.cpp)     │  │ (transfer_engine_py.cpp)  │     │
│  └────────────────────┘  └──────────────────────────┘     │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                Mooncake 核心模块                             │
│  Transfer Engine │ Store │ EP │ PG │ P2P Store             │
└───────────────────────────────────────────────────────────┘
```

---

### vLLM 集成

vLLM 是 Mooncake 最深入的集成之一，支持两种 Connector。

##### MooncakeTransferEngineConnector

用于 PD 解耦的直传模式——KV Cache 直接从 Prefill Worker 传到 Decode Worker。

```python
# Prefill Worker 配置
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --kv-transfer-config '{
        "kv_connector": "MooncakeTransferEngineConnector",
        "kv_role": "kv_producer",
        "mooncake_metadata_server": "etcd://127.0.0.1:2379",
        "mooncake_protocol": "rdma",
        "mooncake_device_name": "mlx5_0"
    }'

# Decode Worker 配置
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --kv-transfer-config '{
        "kv_connector": "MooncakeTransferEngineConnector",
        "kv_role": "kv_consumer",
        "mooncake_metadata_server": "etcd://127.0.0.1:2379",
        "mooncake_protocol": "rdma",
        "mooncake_device_name": "mlx5_0"
    }'
```

##### MooncakeStoreConnector

用于分布式 KV Cache 池——多个 vLLM 实例共享 KV Cache。

```python
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --kv-transfer-config '{
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "mooncake_metadata_server": "etcd://127.0.0.1:2379",
        "mooncake_protocol": "rdma",
        "mooncake_master_addr": "127.0.0.1:50051",
        "mooncake_device_name": "mlx5_0"
    }'
```

##### vLLM 集成能力矩阵

| 功能 | TransferEngineConnector | StoreConnector |
|------|------------------------|----------------|
| PD 解耦 | 支持 | 支持 |
| KV Cache 复用 | 不支持 | 支持（前缀哈希） |
| 跨实例共享 | 不支持 | 支持 |
| 多模态 | vLLM-Omni 支持 | vLLM-Omni 支持 |
| Ascend NPU | 支持 | 支持 |

---

### SGLang 集成

SGLang 与 Mooncake 的集成最为深入，覆盖了 PD 解耦、分层缓存、弹性专家并行、多模态解耦和 RL 训练。

##### 1. PD 解耦服务

通过 Mooncake Transfer Engine 传输 KV Cache，实现 Prefill 和 Decode 分离。

##### 2. HiCache（分层 KV 缓存）

SGLang 的 RadixAttention 扩展为三级缓存：

```
┌───────────────────────┐
│ Device Cache (VRAM)   │  ← GPU 显存，最快
├───────────────────────┤
│ Host Cache (DRAM)     │  ← CPU 内存，较快
├───────────────────────┤
│ Remote Cache (Store)  │  ← Mooncake Store，容量大
└───────────────────────┘
```

HiCache 的工作流程：

1. 请求到达 → 先查 Device Cache
2. 未命中 → 查 Host Cache
3. 未命中 → 查 Remote Cache (Mooncake Store)
4. 命中 → 将 KV Cache 提升到更高级别缓存

##### 3. Elastic Expert Parallel (EEP)

SGLang 使用 Mooncake EP 和 PG 实现弹性专家并行：

- EP 提供 dispatch/combine 通信
- PG 提供集合通信和弹性恢复
- 两者协作实现故障容忍的 MoE 推理

##### 4. EPD（Encode-Prefill-Decode）

多模态场景下的三段式解耦：

```
Encoder (ViT) ──TE──► Prefill (LLM) ──TE──► Decode (LLM)
   专用 GPU              专用 GPU              专用 GPU
```

##### 5. RL 训练的权重同步

SGLang 使用 Mooncake Transfer Engine 实现 RDMA P2P 权重传输：

```
训练节点 ──RDMA P2P──► 推理节点
  (权重更新)              (权重加载)
```

在 Kimi-K2（1T 参数）的生产训练中，P2P 权重更新从 53 秒缩短到 7.2 秒——**7 倍加速**。

---

### TensorRT-LLM 集成

TensorRT-LLM 通过 `mooncake_utils` 模块集成 Mooncake Transfer Engine，用于 PD 解耦场景下的 KV Cache 传输。

```cpp
// TensorRT-LLM 中的 Mooncake 集成
// cpp/tensorrt_llm/executor/cache_transmission/mooncake_utils/
```

集成方式与 vLLM 类似，但使用 C++ API 而非 Python API。

---

### 其他生态集成

| 项目 | 集成方式 | 用途 |
|------|---------|------|
| **LMCache** | MooncakeStoreConnector | 远程 KV Cache 连接器 |
| **LMDeploy** | Transfer Engine | PD 解耦后端 |
| **NIXL** | 插件后端 | NVIDIA 传输层 |
| **TorchSpec** | 隐状态管理 | 推测解码训练 |
| **FlexKV** | Transfer Engine | 分布式 KV 复用 |
| **xLLM** | Mooncake Store | 混合 KV Cache 管理 |
| **LightX2V** | Transfer Engine | 视频生成解耦部署 |
| **vLLM-Ascend** | Transfer Engine + Store | Ascend NPU 上的 PD 解耦 |
| **vLLM-Omni** | Store + TE Connector | 多模态管线通信 |
| **RBG** | HiCache + Store | 云原生 SGLang 部署 |

---

### Python 绑定：integration 层的实现

Mooncake 通过 `mooncake-integration/` 目录提供 Python 绑定，让上层推理引擎可以方便地调用 C++ 核心功能。

##### Store Python 绑定

```cpp
// mooncake-integration/store/store_py.cpp
// PyTensorInfo: PyTorch Tensor 信息提取
struct PyTensorInfo {
    uintptr_t data_ptr;
    size_t tensor_size;
    TensorMetadata metadata;
    py::object owner;
};
```

关键功能：
- 从 PyTorch Tensor 提取数据指针和元数据
- 支持张量分片（sharding）——按维度切分
- 支持并行读写（parallel_read/parallel_write）

##### Transfer Engine Python 绑定

```cpp
// mooncake-integration/transfer_engine/transfer_engine_py.cpp
class TransferEnginePy {
    // 协议特定的内存分配器
    void initMemoryAllocator(const char* protocol);
    
    // 传输超时配置
    int64_t transfer_timeout_nsec_;  // 默认 30s，可通过 MC_TRANSFER_TIMEOUT 环境变量配置
};
```

---

### 集成模式总结

| 集成模式 | 说明 | 典型项目 |
|---------|------|---------|
| **KV Connector** | 通过推理引擎的 KV 传输接口集成 | vLLM, SGLang |
| **存储后端** | 作为分层缓存的远程存储层 | SGLang HiCache, LMCache |
| **通信后端** | 作为分布式通信的传输层 | TensorRT-LLM, LMDeploy |
| **ProcessGroup** | 作为 PyTorch 分布式后端 | SGLang (EP + PG) |
| **P2P 分发** | 作为权重/检查点分发工具 | SGLang RL, checkpoint-engine |

---

### 设计哲学：生态集成的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **接口优先** | 通过标准接口集成，不修改推理引擎核心 | vLLM KV Connector、PyTorch ProcessGroup |
| **多语言支持** | C++/Python/Go/Rust 多语言绑定 | store_py.cpp, Go client, Rust client |
| **可插拔** | 各模块可独立使用，不强依赖 | Store 和 TE 可分别使用 |

---

### 总结与行动指南

| 集成项目 | 一句话总结 |
|---------|----------|
| vLLM | KV Connector 接口，支持直传和存储两种模式 |
| SGLang | 最深入集成：PD 解耦 + HiCache + EP + EPD + RL |
| TensorRT-LLM | C++ API 集成，PD 解耦 KV Cache 传输 |
| LMCache | Store 作为远程 KV Cache 连接器 |
| LMDeploy | Transfer Engine 作为 PD 解耦后端 |

**建议**: 如果你在选型，vLLM 适合快速验证（KV Connector 接口简单），SGLang 适合生产部署（集成最深入，功能最全）。

**延伸阅读**：
- vLLM 集成指南：https://kvcache-ai.github.io/Mooncake/getting_started/examples/vllm-integration/index.html
- SGLang 集成指南：https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/index.html
- vLLM Mooncake Store 博客：https://vllm.ai/blog/mooncake-store

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
