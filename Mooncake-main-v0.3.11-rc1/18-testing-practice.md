# Mooncake 测试实践：1,900+ 测试用例背后的分层防线

> **系列**: Mooncake 技术博客系列 | **类型**: 测试实践篇
>
> 一个分布式系统，组件跨网络、跨介质、跨硬件，任何一层都可能出错。Mooncake 用 1,900+ 个测试用例构建了从单元到混沌的五层防线——但防线并非无懈可击，有些门还敞着（TestCase根据代码仓源码统计，未包括代码仓之外的CI用例）。

本文从测试体系出发，逐层拆解 Mooncake 的测试架构：哪些做得到位，哪些留了缺口，以及如何在生产部署前补全。

标题直接用了当前系统的用例数，多少有点标题党，不过主要目的是，量化的数字可以让人最直观的建立认知，对立与统一，各有侧重，理解万岁。

---

### 引言：分布式系统的测试困境

单机程序的测试相对简单——函数输入输出、边界条件、异常路径，基本够用。但分布式系统不一样：

```
单机测试:    给定输入 → 断言输出    → 确定性
分布式测试:  给定输入 → 网络/磁盘/CPU 可能出错 → 不确定性

分布式系统的三类故障:
1. 组件故障: Master 挂了、Agent 断连、etcd 不可达
2. 介质故障: GPU OOM、SSD 写满、RDMA 链路断开
3. 时序故障: 请求乱序、时钟漂移、竞态条件
```

**测试的第一性原理**：测试的目标不是"证明程序正确"，而是**在最短的时间内，用最低的成本，发现最多的问题**。这意味着：

- **分层测试**：单元测试快而窄，集成测试慢而宽，混沌测试贵而深——每层覆盖不同的故障域
- **故障注入**：不靠等故障发生，而是主动制造故障——"与其祈祷不出错，不如主动搞出错"
- **可观测性**：测试不仅要断言"对不对"，还要验证"快不快"——性能回归也是 bug

Mooncake 的测试体系遵循这个分层逻辑，下面逐层拆解。

---

### 测试全景：五层防线

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  第 5 层: 混沌测试 (Chaos)                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  随机杀 Master、随机杀 Client、验证数据完整性                  │    │
│  │  chaos_test / chaos_rand_test / chaosctl                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第 4 层: 端到端测试 (E2E)                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  完整 PD 解耦流程、真实推理框架 (SGLang/vLLM) 集成             │    │
│  │  T-one 外部测试平台、Ascend NPU 真机测试                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第 3 层: 集成测试 (Integration)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  InProcMaster 真实 RPC、故障注入 (FaultProxyTransport)        │    │
│  │  RDMA 链路模拟 (--wrap)、HA 主从切换、SSD 真实 I/O            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第 2 层: 单元测试 (Unit)                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1,900+ 用例、参数化组合、Mock 对象、自跳过机制                │    │
│  │  Google Test + pytest + Go test + cargo test                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  第 1 层: 静态防护 (Static)                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  ASAN/LSAN、clang-format、spell-check、代码覆盖率              │    │
│  │  -Werror=thread-safety、pre-commit hooks                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 第 1 层：静态防护——编译器就是最好的测试员

静态防护不需要写测试用例——编译器和工具链自动帮你查问题。Mooncake 的静态防线：

| 工具 | 检查内容 | 配置位置 |
|------|---------|---------|
| ASAN + LSAN | 地址越界、Use-After-Free、内存泄露 | `common.cmake` 第 32-39 行，CI 默认启用 |
| clang-format | C++ 代码格式 | `pre-commit-config.yaml`，CI 强制检查 |
| ruff | Python 代码格式 + lint | `pre-commit-config.yaml` |
| codespell | 拼写错误 | `pre-commit-config.yaml`，忽略 `te/mooncake/KVCache/cann` |
| cmake-format | CMake 格式 | `pre-commit-config.yaml` |
| `-Werror=thread-safety` | Clang 线程安全静态分析 | `common.cmake` 第 28-30 行 |
| spell-check | typos 检查 | CI `spell-check` job |
| 代码覆盖率 | lcov + gcovr + Codecov | CI `build` job，过滤测试/三方/基准代码 |

```cmake
# common.cmake — ASAN 配置
option(ENABLE_ASAN "enable address sanitizer" OFF)

if(ENABLE_ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")      # 内存泄露
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")   # 地址越界
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()
```

**覆盖率收集流程**（CI 中的实际步骤）：

```bash
# 1. 编译时注入覆盖率插桩
CXXFLAGS=--coverage CFLAGS=--coverage LDFLAGS=--coverage cmake -DENABLE_ASAN=ON ..

# 2. 运行测试
ctest -j --output-on-failure

# 3. 收集覆盖率数据
lcov --capture --directory . --output-file coverage.info

# 4. 过滤无关代码
lcov --remove coverage.info '/usr/*' '*/test/*' '*/third_party/*' '*/benchmarks/*' \
     --output-file coverage.filtered.info

# 5. 上传到 Codecov
codecov --files coverage.filtered.info --flags unittests
```

##### 静态防线的缺口

| 缺口 | 现状 | 风险 | 建议 |
|------|------|------|------|
| UBSAN | 未配置 | 整数溢出、空指针解引用等未定义行为无检测 | CI 应启用 |
| TSAN | 未配置 | 数据竞争无检测，仅靠 `-Werror=thread-safety` 静态分析 | CI 中可选启用 |
| MSAN | 未配置 | 未初始化内存读取无检测 | CI 中可选启用 |
| 模糊测试 | 无 | 输入校验路径无自动化探索 | 建议对 RPC/元数据解析添加 libFuzzer |

---

### 第 2 层：单元测试——1,900+ 用例的细粒度防线

##### 测试规模

| 模块 | 测试文件 | 测试用例 | 框架 |
|------|---------|------|------|
| mooncake-store | 80 .cpp + 1 .py | 1298 | Google Test |
| mooncake-transfer-engine | 36 .cpp + 6 .py | 279  | Google Test |
| mooncake-transfer-engine/tent | 26 .cpp | 291  | Google Test |
| mooncake-wheel | 26 .py | —    | unittest |
| mooncake-pg | 6 .py | —    | unittest |
| mooncake-ep | 2 .py | —    | torchrun |
| mooncake-common | 2 .cpp + 2 .go | 40   | Google Test + Go test |
| mooncake-store/go | 1 .go | 5    | Go test |
| mooncake-p2p-store | 0 | 0    | — |

**总计：1,900+ 测试用例（C++ gtest 统计 1,908 个），覆盖 C++ / Python / Go / Rust 四种语言。**

##### 六个值得学习的测试模式

**模式 1：参数化组合测试**

不是为每种策略写一个测试，而是用 `TEST_P` + `Combine` 自动生成所有组合：

```cpp
// mooncake-store/tests/allocation_strategy_test.cpp
class AllocationStrategyParameterizedTest
    : public ::testing::TestWithParam<
          std::tuple<AllocationStrategyType, BufferAllocatorType>> {};

// 2 种策略 × 2 种分配器 = 4 个测试用例，自动生成
INSTANTIATE_TEST_SUITE_P(AllCombinations,
    AllocationStrategyParameterizedTest,
    ::testing::Combine(kStrategyTypes, kAllocatorTypes));
```

**模式 2：进程内 Master 服务器**

测试需要真实 RPC，但又不想启动独立进程。`InProcMaster` 在测试进程内启动 coro_rpc 服务器，自动分配空闲端口，避免冲突：

```cpp
// mooncake-store/tests/test_server_helpers.h
class InProcMaster {
    // 自动寻找空闲 TCP 端口
    std::vector<int> getFreeTcpPorts(int count);
    // 进程内启动 Master RPC 服务器
    void start();
    // 支持: 嵌入式 HTTP 元数据服务器、CXL、配额等
};
```

**模式 3：RAII 环境变量隔离**

测试中修改环境变量，但不想影响其他测试。`ScopedEnvVar` 在构造时保存旧值，析构时恢复：

```cpp
// mooncake-store/tests/ha/snapshot/snapshot_test_utils.h
class ScopedEnvVar {
    std::string name_, old_value_;
    bool had_value_;
public:
    ScopedEnvVar(const std::string& name, const std::string& value)
        : name_(name) {
        const char* env = getenv(name.c_str());
        had_value_ = (env != nullptr);
        if (had_value_) old_value_ = env;
        setenv(name.c_str(), value.c_str(), 1);
    }
    ~ScopedEnvVar() {
        had_value_ ? setenv(name_.c_str(), old_value_.c_str(), 1)
                   : unsetenv(name_.c_str());
    }
};
```

**模式 4：自跳过机制**

有些测试依赖特定硬件（RDMA 网卡、etcd 服务），没有就跳过而不是失败：

```cpp
// mooncake-store/tests/ha/leadership/high_availability_test.cpp
// 优雅的自跳过：探测 etcd 可用性，不可用则 GTEST_SKIP
std::optional<std::string> GetEtcdSkipReason() {
    static std::optional<std::string> reason;
    std::call_once(flag, [&] {
        if (!etcd_available) reason = "etcd not available";
    });
    return reason;
}

TEST_F(HighAvailabilityTest, LeaderElection) {
    if (auto skip = GetEtcdSkipReason()) GTEST_SKIP() << *skip;
    // ... 实际测试
}
```

**模式 5：毒字节防御性 Mock**

Mock TPU 设备指针时，如果直接解引用就返回毒字节 `0xDD`——这样如果代码错误地直接访问了 mock 指针，会立即暴露而不是静默通过：

```cpp
// mooncake-transfer-engine/tent/tests/tpu/mock_tpu_pjrt_adapter.cpp
// 设备指针是不透明 TOKEN，直接解引用返回 0xDD
// 捕获"内部指针"bug——普通内存 mock 会隐藏这类错误
```

为什么叫"毒字节"（poison byte）？ 本质上是一种测试陷阱设计。因为碰到它就意味着你犯了错——就像游戏里的毒药，吃到就掉血。毒字节不是正常数据的一部分，它只会在错误的代码路径上出现，所以一旦在输出中看到 0xDD，就能立刻定位到"有人直接解引用了设备指针"这个具体 bug。

毒字节 0xDD 是一个故意的数据陷阱：用错误方式访问设备指针会"中毒"（读到 0xDD），从而让 bug 暴露为测试失败，而不是伪装成正确结果悄悄溜过，毒字节就是用来打破这种"假绿"用例的。

核心思路来自源码中的这段注释：
> "Nothing should ever read it; if a test sees 0xDD it means something dereferenced a device token directly." — mock_tpu_pjrt_adapter.cpp:50-52

**模式 6：历史格式兼容性测试**

快照序列化格式经过 5 次演进，测试确保所有历史格式都能正确反序列化：

```cpp
// mooncake-store/tests/ha/snapshot/snapshot_test_utils.h
enum SnapshotMetadataFormat {
    kLegacy,              // 最初格式
    kDataTypeOnly,        // + 数据类型
    kHardPinnedOnly,      // + 硬钉扎标记
    kDataTypeAndHardPinned,
    kWithGroupId,         // + 组 ID
};
// 测试所有 5 种格式的反序列化正确性
```

> 生产系统迭代演进不可避免，但UT一定要覆盖所有场景，否则任何一个场景，都可能会产生一个生产级灾难。大型分布式复杂系统，不可能靠某个人来记住并手动验证，必须要靠系统UT或者CI自动防护，跑不过就失败，提前发现问题。

##### 单元测试的缺口

| 缺口 | 现状 | 影响 |
|------|------|------|
| mooncake-p2p-store 零测试 | 0 个测试文件 | P2P Store 传火逻辑无自动化验证 |
| mooncake-rl 零测试 | 0 个测试文件 | 强化学习模块无自动化验证 |
| 部分 RDMA 测试未注册 ctest | `rdma_transport_test` 等需手动运行 | CI 不跑这些测试，回归风险 |

---

### 第 3 层：集成测试——组件之间的"接缝检查"

单元测试验证单个函数，集成测试验证组件之间的交互。Mooncake 的集成测试覆盖三个关键"接缝"：

##### 接缝 1：故障注入——FaultProxyTransport

这是 Mooncake 测试体系中最精巧的设计。`FaultProxyTransport` 是一个**传输装饰器**，包裹真实传输层，注入可配置的故障：

```cpp
// mooncake-transfer-engine/tent/include/tent/transport/fault_proxy/
struct FaultPolicy {
    double submit_fail_rate = 0.0;        // 提交失败概率
    double status_corrupt_rate = 0.0;      // 状态翻转概率 (COMPLETED→FAILED)
    uint64_t submit_delay_us = 0;          // 人工延迟
    int fail_after_n_submits = -1;         // 前 N 次成功，之后失败
    bool fail_install = false;             // install() 直接失败
};
```

用它可以模拟真实世界中不可能轻易复现的故障场景：

```
场景: RDMA 提交成功，但状态报告 FAILED → 引擎必须故障转移到 TCP

FaultPolicy policy;
policy.status_corrupt_rate = 1.0;  // 100% 状态翻转
proxy->setRDMAPolicy(policy);      // RDMA 全部"假失败"
proxy->setTCPPolicy({});           // TCP 正常

结果: 引擎检测到 RDMA "失败" → 自动切换到 TCP → 传输成功
验证: 故障转移逻辑在真实故障下也能工作
```

##### 接缝 2：RDMA 链路模拟——链接器符号替换

真实 RDMA 故障难以在 CI 中复现（需要物理网卡）。Mooncake 用 GCC 的 `--wrap` 链接选项拦截 libibverbs 调用，注入故障：

```cmake
# mooncake-transfer-engine/tests/CMakeLists.txt
target_link_options(rdma_endpoint_reestablish_test PRIVATE
    "-Wl,--wrap=ibv_modify_qp"       # 拦截 QP 状态修改
    "-Wl,--wrap=ibv_query_gid"       # 拦截 GID 查询
    "-Wl,--wrap=_ibv_query_gid_ex"   # 拦截 GID 扩展查询
)
```

```cpp
// 测试中注入 RTR 阶段 EINVAL 错误
real_ibv_modify_qp = dlsym(RTLD_DEFAULT, "__real_ibv_modify_qp");
__wrap_ibv_modify_qp(qp, attr, attr_mask) {
    if (inject_failure) {
        errno = EINVAL;
        return EINVAL;  // 模拟 RTR 失败
    }
    return real_ibv_modify_qp(qp, attr, attr_mask);
}
```

这让没有 RDMA 网卡的 CI 也能测试 RDMA 故障恢复逻辑。

##### 接缝 3：HA 主从切换——进程级测试

HA 测试启动真实的 Master 进程，用 `process_handler` 管理子进程生命周期：

```cpp
// mooncake-store/tests/e2e/process_handler.h
class ProcessHandler {
    pid_t pid_;
public:
    void start(const std::string& cmd);
    void kill();     // SIGKILL
    void restart();  // kill + start
    bool isAlive();
};
```

`chaos_test.cpp` 用它模拟 Master 故障：

```
3 个 Master 组成 HA 集群:
  Master-0 (Leader) ← kill → 验证 Master-1 接管
  Master-1 (Leader) ← kill → 验证 Master-2 接管
  Master-2 (Leader) ← kill → 验证 Client 正确报错
  全部重启 → 验证集群恢复
```

##### 故障转移端到端测试

TENT 的 `tent_engine_failover_e2e_test.cpp` 驱动**真实的 `TransferEngineImpl`**，通过 FaultProxyTransport 注入故障，测试完整的故障转移状态机：

| 测试场景 | 注入的故障 | 期望行为 |
|---------|----------|---------|
| RDMA 状态翻转 | status_corrupt_rate=1.0 | 自动切换到 TCP |
| 禁用轮询故障转移 | enable_auto_failover_on_poll=false | 任务保持 FAILED，需 progressBatch 驱动 |
| 预算耗尽 | max_failover_attempts=0 | 第一次失败即永久失败 |
| 混合故障 | 30% RDMA 翻转率，10 个任务 | 所有任务最终通过 TCP 完成 |
| 每任务独立计数 | task0 耗尽预算，task1 仍可用 RDMA | 故障转移计数器相互独立 |

---

### 第 4 层：端到端测试——真实负载下的验证

集成测试验证组件交互，端到端测试验证**真实推理框架下的完整流程**。

##### T-one 外部测试平台

Mooncake 通过 T-one（OpenAnolis 测试平台）运行真实推理框架的集成测试：

| 测试 | 推理框架 | 场景 |
|------|---------|------|
| 1P1D ERDMA | SGLang | 1 个 Prefill + 1 个 Decode，跨机 RDMA |
| MoE Mooncake | SGLang | MoE 模型专家并行 |
| HiCache Storage | SGLang | HiCache 使用 Mooncake 作为后端 |
| 不同 TP 的解耦 | SGLang | Prefill 和 Decode 使用不同的张量并行度 |
| vLLM 1P1D ERDMA | vLLM | vLLM 框架下的 PD 解耦 |
| EPD SGLang | SGLang | 专家并行分发 |

CI 流程：PR 打上 `run-e2e-ci` 标签 → 触发 T-one 提交作业 → 轮询结果（总超时 24 小时）→ 自动清理标签。

##### Ascend NPU 真机测试

在自托管 Runner 上运行，8 张 Ascend NPU 设备：

```yaml
# .github/workflows/ci_ascend.yml
两种场景 × 两个测试用例 = 4 次运行
场景 1: HCCL_INTRA_ROCE_ENABLE=1
场景 2: ASCEND_BUFFER_POOL=4:8
测试: batch_put_get_sample.py, batch_put_get_multi_buffers_sample.py
2 个 Rank (device_id 0 和 2), --distributed --world_size=2
```

##### Python Wheel 安装测试

验证从安装到运行的完整生命周期：

```bash
# scripts/test_installation.sh
1. 创建干净 Python venv
2. 验证 import mooncake.engine 失败（未安装）
3. 安装 wheel
4. 安装系统依赖 (libibverbs-dev, libnuma-dev, ...)
5. 验证 import mooncake.engine 成功
6. 运行 test_import_structure.py + test_mooncake_config.py
7. 验证 CLI 入口点: mooncake_master, mooncake_client, transfer_engine_bench
```

##### E2E 随机测试

长时间随机操作，验证数据完整性：

```cpp
// mooncake-store/tests/e2e/e2e_rand_test.cpp
DEFINE_int32(run_sec, 3600, "Run duration in seconds");  // 默认 1 小时
// 1 个 Master + 2 个 Client
// 随机执行: Put → Get → Delete → ExistKey
// 验证: Get 成功则数据必须与 Put 一致
```

---

### 第 5 层：混沌测试——主动制造混乱

混沌测试回答一个关键问题：**当系统真的出问题时，数据会不会丢？当某个节点故障时，系统会不会不可用？**

##### 三种混沌测试

| 测试 | 方式 | 验证内容 |
|------|------|---------|
| `chaos_test` | 确定性：按预设顺序杀 Master | 主从切换、Client 容错、优雅关闭 |
| `chaos_rand_test` | 随机：随机杀 Master | 数据完整性——Put 成功后 Get 必须返回正确数据 |
| `chaosctl` | 持续混沌锤：随机杀/重启 N 个 Master 和 Client | 统计 PUT/GET/MOUNT/UNMOUNT 成功率 |

`chaos_rand_test` 的数据完整性验证尤其关键——它区分了两种场景：

```
小 value（不触发驱逐）:
  Put 成功 → Get 必须成功 → 数据必须一致

大 value（触发 SSD 驱逐）:
  Put 成功 → Get 可能因驱逐失败（可接受）
  但如果 Get 成功 → 数据必须逐字节一致
```

##### 混沌测试的局限

| 局限 | 说明 |
|------|------|
| 只测 Master 故障 | 不测 Agent 故障、Transfer Engine 故障、etcd 故障 |
| 不测网络分区 | 不模拟脑裂、不对称网络延迟 |
| 不测 SSD 故障 | 不模拟 SSD 写满、I/O 错误、数据损坏 |
| 不测 RDMA 链路故障 | 不模拟 RKey 失效、QP 错误、PKey 变更 |

---

### CI 流水线：24 个工作流的守门体系

Mooncake 的 CI 有 **24 个 GitHub Actions 工作流**，主 CI (`ci.yml`) 有 1114 行，编排 13 个 job：

```
ci.yml (主 CI)
├── spell-check          → 拼写检查
├── clang-format         → C++ 格式检查
├── check-paths          → 路径过滤（无代码变更则跳过）
├── build                → ASAN + 覆盖率构建 + ctest + Rust + Go
├── build-musa           → 摩尔线程 GPU 构建
├── build-arm64          → ARM64 + CUDA 13.0
├── test-wheel-ubuntu    → Ubuntu 22.04/24.04 × Python 3.10/3.12
├── build-flags          → 多配置构建 + cargo test
├── build-docker         → Docker 镜像构建
├── docs-check           → Sphinx 文档严格模式 (-W)
├── build-wheel-cu13     → CUDA 13.0 wheel
├── build-wheel-efa      → AWS EFA wheel
├── tent-ci              → TENT 编译 + ctest (无 GPU 的 Runner)
├── ascend-test          → Ascend NPU 真机测试
├── integration-test     → T-one 外部测试
└── ci-gate              → 汇总所有 job 结果
```

##### 多平台覆盖

| 平台 | 架构 | GPU | CI Job |
|------|------|-----|--------|
| Ubuntu 22.04/24.04 | x86_64 | CUDA 12/13 | build, test-wheel-ubuntu |
| ARM64 | aarch64 | CUDA 13.0 | build-arm64 |
| MThreads (摩尔线程) | x86_64 | MUSA | build-musa |
| Ascend NPU | x86_64 | 昇腾 | ascend-test |
| AWS EFA | x86_64 | EFA/libfabric | build-wheel-efa |
| ROCm/HIP | x86_64 | AMD GPU | build (条件编译) |

> 笔者注：Ascend NPU 本身为达芬奇架构 AI 加速器（PCIe 卡），不绑定 CPU 架构，在华为生态中最常见搭配鲲鹏 ARM（aarch64）处理器，Ascend NPU PCIe 卡也可插入标准 x86_64 服务器。

##### EFA Wheel 的"不含验证"

一个值得注意的 CI 检查——EFA wheel 必须确保 `libfabric`/`libefa` 不被打包进 wheel（它们在运行时从系统路径 `/opt/amazon/efa/lib` 加载）：

```yaml
# ci_efa.yml 第 139-148 行
# 验证 wheel 中不包含 libfabric 和 libefa
- name: Verify wheel contents
  run: |
    if python -m zipfile -l dist/*.whl | grep -q "libfabric\|libefa"; then
      echo "ERROR: libfabric or libefa found in wheel!"
      exit 1
    fi
```

##### 条件测试注册

不是所有测试都能在所有环境运行。Mooncake 用 CMake 条件 + ctest 标签管理：

```cmake
# 有硬件才注册的测试
if(USE_TCP)
  add_store_test(tcp_transport_test ...)
endif()
if(USE_CXL)
  add_store_test(cxl_client_integration_test ...)
endif()

# 需要真实 RDMA 的测试标记为 "rdma" 标签
set_tests_properties(endpoint_store_integration_test
    PROPERTIES LABELS "rdma")
# ctest -LE rdma  → 跳过所有 RDMA 测试
# ctest -L rdma   → 只跑 RDMA 测试

# 构建但不注册（手动运行）
# ub_transport_test: "may still have race conditions, keep out of CI"
```

---

### 基准测试：性能回归也是 Bug

在生产实践中，功能正确只是底线，性能回归同样是 bug。一般都是建立性能基线，新版本的性能测试不能比性能基线差，一般也不能比上一个版本性能差。

Mooncake 的基准测试覆盖两个维度：

##### 传输引擎基准 (`tebench`)

```
操作类型: read / write / mix
后端: Classic TE / TENT
硬件: CUDA / ROCm / Sunrise
参数: block_size, batch_size, num_threads, duration
验证: check_consistency 标志 → 传输后校验数据一致性
```

##### 存储后端基准 (`storage_backend_bench`)

| 场景 | 测什么 | 为什么重要 |
|------|--------|----------|
| init | 初始化耗时 | 冷启动延迟 |
| offload | 写入 SSD 吞吐 | 卸载带宽上限 |
| load | 从 SSD 读取延迟 | TTFT（首 token 延迟） |
| concurrent_load | 并发读取吞吐 | 多请求并发恢复 |
| exist | 元数据查找延迟 | 前缀缓存命中判断 |
| mixed_rw | 读写混合吞吐 | 真实负载下的竞争 |
| churn | 接近容量的稳态 | 驱逐压力下的表现 |
| restart | 重启后恢复耗时 | 故障恢复时间 |

##### CI 中的 Rust 基准

```yaml
# ci.yml 第 162-167 行
MC_RUST_BENCH_ITERATIONS=4
MC_RUST_BENCH_VALUE_SIZE=4096
MC_RUST_BENCH_WARMUP=1
cargo run --release --example store_benchmark
```

---

### 测试缺口与行动指南

##### 缺口全景

| 缺口 | 严重度 | 现状 | 建议 |
|------|:------:|------|------|
| P2P Store 零测试 | **高** | 0 个测试文件 | 至少补核心传火逻辑的单元测试 |
| 无模糊测试 | **高** | 无 libFuzzer/AFL | 对 RPC/元数据解析路径添加模糊测试 |
| UBSAN 未启用 | 中 | CI 未配置 | CI 中启用 `-fsanitize=undefined` |
| RDMA 测试未入 CI | 中 | 需手动运行 | 在有 RDMA 网卡的 Runner 上跑 |
| 混沌测试不覆盖网络分区 | 中 | 只杀进程 | 添加 tc netem 网络故障注入 |
| 混沌测试不覆盖 SSD 故障 | 中 | 只杀 Master | 添加 I/O 错误注入 |
| TSAN 未启用 | 低 | 仅静态分析 | CI 中可选启用（~10x 开销） |
| 无属性测试 | 低 | 无 Hypothesis/proptest | 对 Slice 拆分/合并逻辑添加属性测试 |

##### 测试金字塔健康度

```
理想金字塔:           Mooncake 实际:

      /\                    /\
     /  \   混沌           /  \   混沌 (只杀 Master)
    /    \                 /    \  E2E (依赖外部平台)
   /------\               /------\
  / 集成   \             / 集成   \  (故障注入做得好)
 /----------\           /----------\
/   单元     \         /   单元     \  (1,900+ 用例)
/--------------\       /--------------\
/    静态       \     /    静态       \  (ASAN 有，UBSAN 缺)
------------------     ------------------
```

Mooncake 的单元测试和集成测试做得很扎实，特别是故障注入机制（FaultProxyTransport、`--wrap`、InProcMaster）是业界最佳实践水平。但**混沌测试的覆盖面偏窄**（只覆盖 Master 故障），**模糊测试完全缺失**，这两个缺口在高安全要求场景下需要补全。

---

### 总结与行动指南

| 测试层 | 做得好的 | 需要补的 |
|--------|---------|---------|
| 静态防护 | ASAN + 覆盖率 + clang-format + thread-safety | UBSAN、模糊测试 |
| 单元测试 | 1,900+ 用例、6 种测试模式、4 种语言 | P2P Store 零测试、RDMA 测试未入 CI |
| 集成测试 | FaultProxyTransport、`--wrap` 链路模拟、InProcMaster | — |
| E2E 测试 | T-one 真实推理框架、Ascend 真机、多平台 | 更多推理框架组合 |
| 混沌测试 | 确定性 + 随机 + 持续锤、数据完整性验证 | 网络分区、SSD 故障、Agent 故障 |

**建议**: Mooncake 的测试体系在"组件正确性"和"故障注入"上做得扎实，但在"混沌覆盖面"和"模糊测试"上留了缺口。生产部署前，至少补全 P2P Store 测试和 RPC 模糊测试——这两处缺口在真实负载下最容易被触发。

**延伸阅读**：
- Google Test 文档：https://google.github.io/googletest/
- Chaos Engineering 原则：https://principlesofchaos.org/
- libFuzzer 入门：https://llvm.org/docs/LibFuzzer.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
