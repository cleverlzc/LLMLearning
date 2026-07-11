# vLLM 设计模式解析：从源码中学习可复用的架构智慧

> **系列**: vLLM 技术博客系列 | **类型**: 设计模式篇
>
> 好的架构不是"设计"出来的，而是"生长"出来的——vLLM 的七大设计模式，就是这棵大树最坚实的根系。

## 引言

想象一座现代化的机场：值机柜台根据航班类型自动切换自助与人工通道（策略模式），航站楼的候机区根据客流大小动态增减（对象池模式），行李从托运到转盘经历了一条看不见的流水线（生产者-消费者模式）。优秀的系统架构与优秀的机场一样，核心不在于"多复杂"，而在于"多灵活"——面对变化，能否像换登机口一样从容？

vLLM 作为当前最活跃的 LLM 推理引擎之一，在短短两年内就从单一 GPU 支持扩展到了 CUDA、ROCm、TPU、CPU、XPU 五大平台，Attention 后端从 1 个增长到 15+ 个，Executor 从单进程演化出多进程、Ray、外部启动器四种模式。支撑这种快速演化的，正是深植于代码中的经典设计模式。

今天，我们就翻开 vLLM 的源码，逐一解析其中最核心的七大设计模式。每一个模式，我们都将从"它解决什么问题"出发，找到源码中的实现位置，提取简化代码片段，最终提炼出它的架构价值。

---

## 一、策略模式（Strategy Pattern）—— Attention 后端的"可插拔引擎"

### 1.1 问题：Attention 算子千差万别

LLM 推理中，Attention 是计算密度最高的环节。但不同硬件、不同模型对 Attention 算子的需求截然不同：

| 场景 | 首选后端 |
|------|---------|
| NVIDIA A100/H100 | FlashAttention 或 FlashInfer |
| AMD MI300 | ROCm Aiter FA |
| CPU 推理 | Triton / CPU Attention |
| DeepSeek MLA | 专用 MLA 后端 |
| Mamba 模型 | Mamba SSM 后端 |

如果没有统一的抽象层，每新增一种 Attention 实现，都需要在模型代码里塞满 `if-else`，修改一处牵动全身。

### 1.2 源码实现

vLLM 定义了 `AttentionBackend` 抽象基类，将每种 Attention 算法封装为一个可替换的"策略"。

```
# vllm/v1/attention/backend.py

class AttentionBackend(ABC):
    """Attention 后端的抽象基类——策略模式的接口"""

    @staticmethod
    @abstractmethod
    def get_name() -> str: ...          # 返回后端名称

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type: ...     # 返回实际计算实现类

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type: ...  # 返回元数据构建器类

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(...) -> tuple: ...  # 返回 KV Cache 张量形状
```

具体的后端实现只需继承并实现这些抽象方法：

```
# vllm/v1/attention/backends/flash_attn.py
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: return "FLASH_ATTN"

# vllm/v1/attention/backends/flashinfer.py
class FlashInferBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: return "FLASHINFER"

# vllm/v1/attention/backends/triton_attn.py
class TritonAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: return "TRITON_ATTN"
```

选择哪个策略由 `selector.py` 根据运行时环境自动决定：

```
# vllm/v1/attention/selector.py

def get_attn_backend(head_size, dtype, kv_cache_dtype, ...) -> type[AttentionBackend]:
    """根据运行时配置自动选择 Attention 后端"""
    # 委托给当前平台的策略选择逻辑
    attention_cls = current_platform.get_attn_backend_cls(
        backend, attn_selector_config=attn_selector_config, ...)
    return resolve_obj_by_qualname(attention_cls)
```

### 1.3 架构图

```
┌──────────────────────────────────────────────────┐
│              AttentionBackend (ABC)               │
│  get_name() | get_impl_cls() | get_builder_cls() │
└──────────┬──────────┬──────────┬──────────┬──────┘
           │          │          │          │
     ┌─────▼──┐ ┌─────▼──┐ ┌────▼───┐ ┌───▼──────┐
     │FlashAttn│ │FlashInf│ │Triton  │ │ROCm Aiter│
     │Backend  │ │erBack. │ │AttnBack│ │Backend   │
     └─────────┘ └────────┘ └────────┘ └──────────┘

         ┌──────────────────┐
         │  AttentionSelector│ ←── 运行时选择
         │  (selector.py)    │
         └──────────────────┘
```

### 1.4 收益

| 维度 | 无策略模式 | 有策略模式 |
|------|-----------|-----------|
| 新增后端 | 修改模型层代码，散布 if-else | 新增一个 Backend 子类即可 |
| 切换后端 | 手动改代码重新编译 | 配置 `--attention-backend` 即切换 |
| 测试 | 需要测试所有分支组合 | 每个后端独立测试 |
| 扩展性 | 线性退化 | 线性增长，无退化 |

> 💡 **性能提示**: vLLM 的 `_cached_get_attn_backend` 使用 `@cache` 装饰器，确保同一配置只做一次选择，避免重复决策开销。

---

## 二、工厂模式（Factory Pattern）—— Executor 的"智能装配线"

### 2.1 问题：一种配置，多种执行器

vLLM 支持多种分布式执行方式：单进程、多进程、Ray 分布式、外部启动器。用户只需要指定 `--distributed-executor-backend`，系统就应该自动"装配"出正确的 Executor。如果让调用方直接 `import` 具体类，耦合就太重了。

### 2.2 源码实现

`Executor.get_class()` 是一个静态工厂方法，根据 `VllmConfig.parallel_config` 动态决定创建哪种执行器：

```python
# vllm/v1/executor/abstract.py

class Executor(ABC):
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """工厂方法：根据并行配置返回合适的 Executor 子类"""
        parallel_config = vllm_config.parallel_config
        backend = parallel_config.distributed_executor_backend

        if isinstance(backend, type):
            # 用户传入自定义 Executor 类
            executor_class = backend
        elif backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2
                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor
                executor_class = RayDistributedExecutor
        elif backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor
            executor_class = UniProcExecutor
        elif backend == "external_launcher":
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError(f"Unknown backend: {backend}")
        return executor_class
```

工厂方法的调用方——`EngineCore`——完全不需要知道具体类名：

```python
# vllm/v1/engine/core.py

class EngineCore:
    def __init__(self, vllm_config, executor_class, ...):
        # executor_class 由外部通过工厂方法获得
        self.model_executor = executor_class(vllm_config)
```

### 2.3 架构图

```
┌─────────────────────────────────────────────────┐
│              Executor.get_class()                │
│            (静态工厂方法)                         │
└───┬──────────┬──────────┬──────────┬────────────┘
    │          │          │          │
┌───▼──┐  ┌───▼──────┐ ┌─▼──────┐ ┌▼──────────────────┐
│UniProc│  │Multiproc │ │RayDist │ │ExtLauncher        │
│Exec.  │  │Executor  │ │Executor│ │(继承 UniProcExec.) │
└───────┘  └──────────┘ └────────┘ └───────────────────┘
```

### 2.4 收益

| 维度 | 无工厂模式 | 有工厂模式 |
|------|-----------|-----------|
| 新增执行器 | 修改所有调用方 | 只修改工厂方法 |
| 延迟导入 | 无法控制，启动即加载全量 | 工厂内按需 `import`，减少启动开销 |
| 配置驱动 | 代码驱动 | 一行 `--distributed-executor-backend` 即切换 |

> 笔者注：工厂方法内部的 `from ... import` 延迟导入非常巧妙——只有在真正需要时才加载 Ray、多进程等重型依赖，对单 GPU 用户来说启动速度更快。

---

## 三、观察者/事件模式（Observer/Event Pattern）—— 引擎输出的"广播站"

### 3.1 问题：一次推理，多方关注

一次模型推理的输出，至少有三方需要关注：

1. **反分词器（Detokenizer）** —— 将 token ID 转回文本
2. **指标收集器（Metrics）** —— 记录延迟、吞吐等统计
3. **请求输出流（RequestOutputCollector）** —— 将结果推送给等待的 asyncio 任务

如果让模型执行器直接调用这三方，耦合就太紧了。vLLM 采用了一种"事件驱动"的思路：引擎输出作为事件，由不同的"观察者"独立消费。

### 3.2 源码实现

核心流转在 `EngineCore.step()` 中：

```python
# vllm/v1/engine/core.py

class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        # 1. 调度
        scheduler_output = self.scheduler.schedule(...)
        # 2. 执行模型
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        model_output = future.result()
        # 3. 将输出"广播"给调度器更新状态
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)
        return engine_core_outputs, ...
```

在 API 层，`OutputProcessor` 和 `RequestOutputCollector` 扮演观察者角色：

```python
# vllm/v1/engine/output_processor.py

class RequestOutputCollector:
    """请求级别的输出收集器——每个请求一个观察者"""

    def __init__(self, output_kind, request_id):
        self.ready = asyncio.Event()  # 通知机制

    def put(self, output):
        """生产者调用：非阻塞写入"""
        self.output = output
        self.ready.set()  # 通知等待者

    async def get(self):
        """消费者调用：阻塞等待"""
        while self.output is None:
            await self.ready.wait()
        result = self.output
        self.output = None
        self.ready.clear()
        return result
```

### 3.3 架构图

```
                     EngineCore.step()
                           │
                 ┌─────────▼──────────┐
                 │   ModelExecutor     │
                 │   execute_model()   │
                 └─────────┬──────────┘
                           │ ModelRunnerOutput
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼──┐  ┌─────▼─────┐  ┌──▼───────────┐
     │Scheduler  │  │OutputProc │  │MetricsLogger │
     │update_from│  │(Detokenize│  │(统计收集)     │
     │_output()  │  │ + push)   │  │              │
     └───────────┘  └─────┬─────┘  └──────────────┘
                          │
                 ┌────────▼────────┐
                 │RequestOutput    │
                 │Collector (async)│
                 │  per-request    │
                 └─────────────────┘
```

### 3.4 收益

| 维度 | 直接调用 | 事件模式 |
|------|---------|---------|
| 新增消费者 | 修改生产者代码 | 新增观察者即可 |
| 消费者失败 | 可能阻塞生产者 | 观察者独立处理，互不影响 |
| 异步流式输出 | 难以实现 | `asyncio.Event` 天然支持 |

---

## 四、模板方法模式（Template Method Pattern）—— GPUModelRunner 的"标准作业流程"

### 4.1 问题：模型推理步骤多且固定

GPU 上的模型推理有一套固定的"标准作业流程"：更新状态 → 准备输入 → 构建 Attention 元数据 → 前向传播 → 采样 → 后处理。不同模型的差异主要在"前向传播"和"采样"的具体逻辑，但整体骨架不变。

### 4.2 源码实现

`GPUModelRunner.execute_model()` 就是那个"模板方法"——它定义了推理的完整骨架，每个步骤通过调用内部方法（"钩子"）来实现：

```python
# vllm/v1/worker/gpu_model_runner.py

class GPUModelRunner(
    LoRAModelRunnerMixin,          # LoRA 相关钩子
    KVConnectorModelRunnerMixin,   # KV Connector 钩子
    ECConnectorModelRunnerMixin,   # EC Connector 钩子
):
    def execute_model(self, scheduler_output, ...):
        # 步骤1: 更新持久化批状态
        deferred_state_corrections_fn = self._update_states(scheduler_output)

        # 步骤2: 准备输入（token IDs, positions, 等）
        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output, num_scheduled_tokens_np)

        # 步骤3: 确定 batch 执行模式和 padding
        cudagraph_mode, batch_desc, ... = self._determine_batch_execution_and_padding(...)

        # 步骤4: 构建 Attention 元数据
        attn_metadata, ... = self._build_attention_metadata(...)

        # 步骤5: 获取 slot mappings
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)

        # 步骤6: 前向传播（由模型对象执行，模型可覆盖）
        hidden_states = self.model(...)

        # 步骤7: 采样
        sampler_output = self._sample(logits, spec_decode_metadata, ...)

        # 步骤8: 后处理与输出
        return model_output
```

注意类声明中的 Mixin 继承：`LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin`。这些 Mixin 可以在模板方法的不同阶段注入自己的逻辑，相当于"可插拔的模板钩子"。

### 4.3 架构图

```
┌──────────────────────────────────────────────────────────┐
│             GPUModelRunner.execute_model()                │
│                  (模板方法 —— 定义骨架)                     │
├──────────────────────────────────────────────────────────┤
│ 1. _update_states()       ←── 状态更新                    │
│ 2. _prepare_inputs()      ←── 输入准备                    │
│ 3. _determine_batch_...() ←── 批次决策                    │
│ 4. _build_attention_...() ←── 元数据构建                   │
│ 5. _get_slot_mappings()   ←── Slot 映射                   │
│ 6. self.model(...)        ←── 前向传播 (模型可覆盖)        │
│ 7. _sample()              ←── 采样                        │
│ 8. 后处理 & 返回           ←── 输出构造                    │
└──────────────────────────────────────────────────────────┘
          ▲              ▲               ▲
          │              │               │
  LoRAModelRunner   KVConnector      ECConnector
     Mixin           Mixin             Mixin
  (注入LoRA逻辑)   (注入KV迁移逻辑)  (注入EC逻辑)
```

### 4.4 收益

| 维度 | 无模板方法 | 有模板方法 |
|------|-----------|-----------|
| 流程保证 | 靠开发者自觉 | 骨架方法强制执行顺序 |
| 差异化 | 散布在各处 | 通过 Mixin 集中管理 |
| 可读性 | 需要读完整代码才能理解流程 | 一眼看清"先做什么后做什么" |

> 💡 **性能提示**: 模板方法中的 `_determine_batch_execution_and_padding()` 是性能关键路径——它决定是否启用 CUDA Graph、如何 padding，直接影响推理延迟。

---

## 五、适配器模式（Adapter Pattern）—— 平台抽象的"万能转接头"

### 5.1 问题：五种硬件，一套代码

vLLM 需要同时支持 CUDA、ROCm、TPU、XPU、CPU 五种硬件平台。每种平台的 Attention 后端选择逻辑、分布式通信后端、内存管理方式都不同。如果到处写 `if platform == "cuda": ...`，代码会变成一团乱麻。

### 5.2 源码实现

vLLM 的解决方案是经典的适配器模式——`Platform` 基类定义统一接口，各平台子类做适配：

```python
# vllm/platforms/interface.py

class Platform:
    """平台抽象基类——适配器模式的接口"""
    _enum: PlatformEnum
    device_name: str
    device_type: str
    dispatch_key: str = "CPU"
    dist_backend: str = ""

    def is_cuda(self) -> bool:  return self._enum == PlatformEnum.CUDA
    def is_rocm(self) -> bool:  return self._enum == PlatformEnum.ROCM
    def is_tpu(self) -> bool:   return self._enum == PlatformEnum.TPU

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, ...) -> str:
        """返回平台适用的 Attention 后端类路径"""
        return ""  # 默认空，子类覆盖

    @classmethod
    def inference_mode(cls):
        """平台特定的推理模式（TPU 等平台不支持 inference_mode 时会回退到 torch.no_grad）"""
        return torch.inference_mode(mode=True)

    @classmethod
    def is_sleep_mode_available(cls) -> bool:
        """仅 CUDA、ROCm、XPU 支持睡眠模式，TPU 和 CPU 不支持"""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.XPU)
```

CUDA 平台的适配器实现在 `vllm/platforms/cuda.py`，它覆盖了 Attention 后端选择、FP8 支持、自定义 AllReduce 等平台特定逻辑。

平台的自动检测在 `vllm/platforms/__init__.py` 中完成：

```python
# vllm/platforms/__init__.py

def resolve_current_platform_cls_qualname() -> str:
    """自动检测当前硬件平台，返回对应的 Platform 类路径"""
    builtin_platform_plugins = {
        "tpu":  tpu_platform_plugin,    # → "vllm.platforms.tpu.TpuPlatform"
        "cuda": cuda_platform_plugin,   # → "vllm.platforms.cuda.CudaPlatform"
        "rocm": rocm_platform_plugin,   # → "vllm.platforms.rocm.RocmPlatform"
        "xpu":  xpu_platform_plugin,    # → "vllm.platforms.xpu.XPUPlatform"
        "cpu":  cpu_platform_plugin,     # → "vllm.platforms.cpu.CpuPlatform"
    }
    # ... 逐一探测，返回第一个匹配的
```

调用方只需 `from vllm.platforms import current_platform`，即可获得当前平台的适配器实例，无需关心底层差异。

### 5.3 架构图

```
                    from vllm.platforms import current_platform
                                    │
                        ┌───────────▼───────────┐
                        │      Platform (ABC)    │
                        │  get_attn_backend_cls()│
                        │  inference_mode()      │
                        │  is_sleep_mode_available│
                        └──┬────┬────┬────┬────┬┘
                           │    │    │    │    │
                     ┌─────▼┐┌──▼──┐┌▼───┐┌▼──┐┌▼────┐
                     │Cuda  ││Rocm ││TPU ││XPU││CPU  │
                     │Plat. ││Plat.││Plat││Pl.││Plat.│
                     └──────┘└─────┘└────┘└───┘└─────┘
```

### 5.4 收益

| 维度 | 无适配器 | 有适配器 |
|------|---------|---------|
| 新增平台 | 散布全局 if-else | 新增一个 Platform 子类 |
| 代码复用 | 复制粘贴 | 基类提供默认实现，子类只覆盖差异 |
| 运行时切换 | 不可能 | 自动检测 + 懒加载，零配置 |
| OOT 扩展 | 不支持 | 插件机制支持 Out-of-Tree 平台 |

> 💡 **性能提示**: `current_platform` 采用懒加载 + `__getattr__` 代理模式，首次访问时才做平台检测，不影响启动速度。对于 `torch.cuda` 等原生 API 的缺失属性，它会自动代理到对应的 PyTorch 模块，实现"无感适配"。

---

## 六、生产者-消费者模式（Producer-Consumer Pattern）—— API → Engine Core → GPU 的"三级流水线"

### 6.1 问题：请求处理需要解耦

vLLM 的请求处理涉及三个阶段：API Server 接收请求、Engine Core 调度与执行、GPU Worker 实际计算。如果这三个阶段同步串行执行，任何一个阶段慢了都会阻塞整个系统。特别是 GPU 推理与 CPU 调度之间，天然就是异步的——GPU 在计算第 N 步时，CPU 可以同时调度第 N+1 步。

### 6.2 源码实现

vLLM 的 V1 架构将整个推理管线设计为三级生产者-消费者模型：

**第一级：AsyncLLM → EngineCore**

```python
# vllm/v1/engine/async_llm.py

class AsyncLLM(EngineClient):
    """API 层 —— 生产者：将请求推入 Engine Core"""
    async def generate(self, prompt, sampling_params, ...):
        # 将请求序列化后发送给 EngineCore
        self.engine_core.add_request_async(request)
```

**第二级：EngineCore → Executor（异步执行）**

```python
# vllm/v1/engine/core.py

class EngineCore:
    def step(self):
        # 调度（CPU 侧）
        scheduler_output = self.scheduler.schedule(...)
        # 异步执行模型（GPU 侧），non_block=True 立即返回 Future
        future = self.model_executor.execute_model(
            scheduler_output, non_block=True)
        # ... 可以在此做其他 CPU 工作
        model_output = future.result()  # 等待 GPU 完成
```

**第三级：Executor → Worker**

```python
# vllm/v1/executor/uniproc_executor.py

class UniProcExecutor(Executor):
    def collective_rpc(self, method, ..., non_block=False):
        if not non_block:
            # 同步模式：直接执行
            result = run_method(self.driver_worker, method, args, kwargs)
            return [result]
        else:
            # 异步模式：返回 Future，GPU 在后台执行
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
```

### 6.3 架构图

```
┌──────────────┐    ZMQ/IPC    ┌──────────────┐   non_block    ┌──────────────┐
│  AsyncLLM    │──────────────▶│ EngineCore   │───────────────▶│   Executor   │
│ (API Server) │  request()    │  (Scheduler) │  execute_model │  (Process    │
│  生产者       │               │  中间消费者    │   + 生产者     │   Manager)   │
└──────────────┘               └──────────────┘                └──────┬───────┘
                                                                      │
                                                              ┌───────▼───────┐
                                                              │  GPU Worker   │
                                                              │  (消费者)      │
                                                              │  execute_model│
                                                              └───────────────┘

数据流向:  HTTP Request → ZMQ Queue → Schedule → Future → GPU Compute → Output
```

### 6.4 收益

| 维度 | 同步串行 | 生产者-消费者 |
|------|---------|-------------|
| GPU 利用率 | 调度时空转 | 异步调度 + CUDA Stream 实现真正的双工 |
| 流式输出 | 必须等整批完成 | 中间结果即可推送 |
| 批队列 | 无法实现 | `batch_queue` 支持多批次流水，消除 PP 气泡 |

> 笔者注：`EngineCore.step_with_batch_queue()` 是生产者-消费者模式的高级形态——当 `max_concurrent_batches > 1` 时，它可以同时调度多个 batch，让 GPU 与 CPU 真正重叠工作，对 Pipeline Parallelism 尤为关键。

---

## 七、对象池模式（Object Pool Pattern）—— KV Cache Block Pool 的"共享车库"

### 7.1 问题：KV Cache 块的反复分配与释放

在 PagedAttention 机制中，KV Cache 被划分为固定大小的 Block。请求到来时分配 Block，请求完成后释放 Block。如果每次都向操作系统申请/释放内存，开销巨大且容易产生碎片。更关键的是，Prefix Caching 需要一种"借用-归还"机制——当一个请求释放的 Block 恰好是另一个请求的前缀，它不应该被立即销毁，而应该留在池中等待复用。

### 7.2 源码实现

`BlockPool` 就是 vLLM 的 KV Cache 对象池：

```python
# vllm/v1/core/block_pool.py

class BlockPool:
    """KV Cache Block 对象池——管理分配、释放、缓存驱逐"""

    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # 预分配所有 Block 对象
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # 空闲 Block 队列（双向链表，支持 LRU 驱逐）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 前缀缓存：hash → Block 的映射
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """从池中获取空闲 Block"""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError("Cannot get enough free blocks")
        ret = self.free_block_queue.popleft_n(num_blocks)
        for block in ret:
            self._maybe_evict_cached_block(block)  # 需要时驱逐缓存
            block.ref_cnt += 1                       # 引用计数 +1
        return ret

    def free_blocks(self, ordered_blocks) -> None:
        """归还 Block 到池中"""
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                # 有 hash 的 Block 放入 LRU 尾部（可被驱逐或复用）
                # 无 hash 的 Block 放入 LRU 头部（优先驱逐）
                ...

    def touch(self, blocks) -> None:
        """增加引用计数（Prefix Cache 命中时调用）"""
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)  # 从空闲队列移除
            block.ref_cnt += 1
```

`KVCacheManager` 在 `BlockPool` 之上构建了请求级别的管理逻辑：

```python
# vllm/v1/core/kv_cache_manager.py

class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, ...):
        self.coordinator = get_kv_cache_coordinator(...)
        # BlockPool 在 coordinator 内部
```

### 7.3 架构图

```
┌────────────────────────────────────────────────────────┐
│                     BlockPool                          │
│                                                        │
│  ┌──────────────────────────────────────────┐          │
│  │      blocks: list[KVCacheBlock]          │          │
│  │  [0] [1] [2] [3] ... [num_gpu_blocks-1] │          │
│  └──────────────────────────────────────────┘          │
│                                                        │
│  ┌──────────────────────────────────────────┐          │
│  │    FreeKVCacheBlockQueue (双向链表)        │          │
│  │  HEAD ←→ [Block5] ←→ [Block2] ←→ TAIL   │          │
│  │  (优先驱逐)              (可缓存复用)       │          │
│  └──────────────────────────────────────────┘          │
│                                                        │
│  ┌──────────────────────────────────────────┐          │
│  │  cached_block_hash_to_block (前缀缓存)    │          │
│  │  {hash_A: Block3, hash_B: Block7, ...}   │          │
│  └──────────────────────────────────────────┘          │
│                                                        │
│  get_new_blocks()  →  分配  →  ref_cnt++               │
│  free_blocks()     →  归还  →  ref_cnt--               │
│  touch()           →  复用  →  ref_cnt++ (从空闲队列移出) │
│  evict_blocks()    →  驱逐  →  清除 hash, 放回空闲队列    │
└────────────────────────────────────────────────────────┘
```

### 7.4 收益

| 维度 | 动态分配/释放 | 对象池 |
|------|-------------|-------|
| 内存碎片 | 严重 | 预分配，零碎片 |
| 分配延迟 | 不确定（受 GC 影响） | O(1) 链表操作 |
| Prefix Caching | 无法实现 | hash 映射 + 引用计数天然支持 |
| 内存利用率 | 低（无法跨请求共享） | 高（空闲 Block 可被驱逐或复用） |

> 💡 **性能提示**: `BlockPool.free_blocks()` 将有 hash 的 Block 放入 LRU 尾部、无 hash 的放入头部——这样前缀缓存命中过的 Block 会被"保护"更久，而从未被复用的 Block 会优先被驱逐，这是非常精巧的缓存友好策略。

---

## 八、七大模式一图总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         vLLM V1 架构全景                             │
│                                                                     │
│  ┌────────────┐  生产者-消费者  ┌────────────┐  生产者-消费者         │
│  │ AsyncLLM   │───────────────▶│EngineCore  │───────────────────▶  │
│  │ (API层)    │    ZMQ/IPC     │ (调度层)    │   non_block Future   │
│  └────────────┘                └──────┬─────┘                      │
│                                       │                             │
│          观察者/事件                    │  工厂模式                   │
│          (OutputProcessor)             │  Executor.get_class()      │
│                  ▲                     ▼                             │
│                  │          ┌─────────────────────┐                 │
│                  │          │    Executor          │                 │
│                  │          │  ┌───────────────┐  │                 │
│                  │          │  │GPUModelRunner │  │  模板方法        │
│                  │          │  │ execute_model │  │  (7步骨架)       │
│                  │          │  └───────┬───────┘  │                 │
│                  │          │          │          │                 │
│                  │          │    ┌─────▼─────┐    │                 │
│                  │          │    │Attention  │    │  策略模式         │
│                  │          │    │Backend    │    │  (可插拔后端)     │
│                  │          │    │(FlashAttn/│    │                 │
│                  │          │    │ FlashInfer)│   │                 │
│                  │          │    └───────────┘    │                 │
│                  │          └─────────────────────┘                 │
│                  │                                                   │
│                  │          适配器模式                                 │
│                  │          current_platform                          │
│                  │          (CUDA/ROCm/TPU/CPU/XPU)                  │
│                  │                                                   │
│  ┌───────────────▼────────────────────────────────────────┐         │
│  │                BlockPool (对象池)                        │         │
│  │  预分配 KV Cache Block  │  引用计数  │  前缀缓存         │         │
│  └────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 九、模式速查表

| 设计模式 | 核心文件 | 核心类/方法 | 解决的问题 | 一句话收益 |
|---------|---------|------------|-----------|-----------|
| 策略模式 | `v1/attention/backend.py` | `AttentionBackend` (ABC) | Attention 算法多样化 | 新增后端只加一个子类 |
| 工厂模式 | `v1/executor/abstract.py` | `Executor.get_class()` | 执行器类型动态选择 | 配置驱动，延迟导入 |
| 观察者/事件 | `v1/engine/output_processor.py` | `RequestOutputCollector` | 输出多方消费 | 消费者独立，异步流式 |
| 模板方法 | `v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model()` | 推理流程固定但细节不同 | 骨架统一，差异用 Mixin |
| 适配器模式 | `vllm/platforms/interface.py` | `Platform` (ABC) | 多硬件平台适配 | 一套代码五种硬件 |
| 生产者-消费者 | `v1/engine/core.py` | `EngineCore.step()` | 调度与计算解耦 | CPU/GPU 重叠执行 |
| 对象池 | `v1/core/block_pool.py` | `BlockPool` | KV Cache Block 频繁分配释放 | O(1) 分配 + 前缀缓存 |

---

## 十、给你的行动建议

如果你正在阅读 vLLM 源码或设计自己的推理引擎，这里有一条最高效的切入路径：

**先读懂策略模式和工厂模式**——它们是 vLLM 扩展性的两大支柱。掌握了 `AttentionBackend` 和 `Executor.get_class()`，你就理解了 vLLM 为什么能在两年内支持 15+ 种 Attention 后端和 4 种执行器，而核心代码几乎不用改。

然后，尝试自己写一个自定义的 `AttentionBackend`——你会发现只需要实现 `get_name()`、`get_impl_cls()`、`get_builder_cls()`、`get_kv_cache_shape()` 四个抽象方法，系统就会自动接纳你的后端。这就是设计模式的力量：**好的架构不是限制你的自由，而是让你在正确的位置发挥创造力。**

---

## 延伸阅读

- Design Patterns: Elements of Reusable Object-Oriented Software — GoF 经典，策略、工厂、观察者、模板方法的定义出处：https://en.wikipedia.org/wiki/Design_Patterns)
- PagedAttention 论文 — 理解对象池模式需要先理解 PagedAttention 的 Block 机制：https://arxiv.org/abs/2309.06180


---

*本文属于 [vLLM 技术博客系列]，欢迎持续关注。*
