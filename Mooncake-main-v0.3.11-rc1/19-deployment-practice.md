# Mooncake 部署实践：从裸机到云原生的全栈交付

> **系列**: Mooncake 技术博客系列 | **类型**: 部署实践篇

Mooncake 是一个"库优先、进程级"的分布式系统——它不替你编排容器、不替你写 K8s YAML、不替你管理 GPU 资源。它提供的是一组精心设计的"积木"：容器镜像、配置体系、HA 后端、健康检查、监控面板。如何把这些积木搭成生产级服务，是部署工程师的功课。

本文从部署的角度出发，逐层拆解 Mooncake 的部署架构：裸机直装、容器化、K8s 编排，以及每个形态下必须面对的问题。

在企业里面，能够在环境上快速、自动化的一键部署和运行整套系统，也是非常关键的一环。

---

### 引言：部署的视角

部署是什么？**把正确的二进制，放到正确的机器上，用正确的配置跑起来，并保证它一直正确地跑下去。**

这句话拆开来看，可以分为：

```
把正确的二进制   → 编译构建与打包成容器镜像
放到正确的机器上 → 资源调度与亲和性
用正确的配置     → 配置管理与分层
跑起来           → 启动序列与服务发现
一直正确地跑下去 → 健康检查、HA、监控、优雅关闭
```

Mooncake 的部署挑战在于：它涉及 GPU、RDMA、SSD、HugePage 四种特殊硬件资源，跨节点 RDMA 通信要求网络拓扑感知，PD 解耦要求 Prefill 和 Decode 分开调度。这不是一个"写个 Deployment YAML 就完事"的系统。

---

### 部署形态全景：三种交付方式

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  形态 1: 裸机直装 (Bare Metal)                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  pip install → 配置环境变量 → 直接启动                        │    │
│  │  适合: 快速验证、小规模部署                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  形态 2: 容器化 (Docker)                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  官方镜像 + docker run --device + 环境变量                    │    │
│  │  适合: 单节点部署、CI/CD 集成                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  形态 3: K8s 编排 (Kubernetes)                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  自定义 YAML + Device Plugin + K8s Lease 选举                 │    │
│  │  适合: 生产级多节点部署、弹性伸缩                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 形态 1：裸机直装——最简单也最容易踩坑

##### 安装

```bash
# 1. 安装系统依赖
sudo apt-get install -y build-essential cmake libibverbs-dev libnuma-dev \
    libssl-dev libcurl4-openssl-dev liburing-dev

# 2. 安装 Python wheel
pip install mooncake-transfer-engine

# 3. 验证
python3 -c "import mooncake.engine; print('OK')"
```

##### 四项硬件前置检查

裸机部署最大的坑不是软件，而是硬件环境。在启动任何 Mooncake 组件之前，先跑完这四项检查：

| 检查项 | 命令 | 期望结果 |
|--------|------|---------|
| RDMA 网卡可用 | `ibv_devinfo` | 列出 `mlx5_*` 设备 |
| RDMA 带宽达标 | `ib_write_bw -d mlx5_0` | 接近网卡标称带宽 |
| HugePage 已分配 | `cat /proc/meminfo \| grep Huge` | `HugePages_Total` >= 所需数量 |
| memlock 无限制 | `ulimit -l` | `unlimited` 或足够大 |

```bash
# 分配 HugePage (2MB 大页, 512GB = 262144 页)
sudo sysctl -w vm.nr_hugepages=262144

# 持久化
echo "vm.nr_hugepages=262144" | sudo tee /etc/sysctl.d/90-mooncake-hugepages.conf

# 解除 memlock 限制 (io_uring 固定缓冲区需要)
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

> 笔者注：这里假设AI服务器上有 512GB 可以分给KV Cache系统，高端服务器DRAM内存更大，可以分配更多，比如1TB，甚至2TB。内存越大，服务器价格越高，推理系统吞吐越强。

##### 启动 Master

```bash
# 最简启动
mooncake_master --rpc_port=50051

# 生产级启动
mooncake_master \
    --rpc_port=50051 \
    --rpc_thread_num=4 \
    --enable_metric_reporting=true \
    --metrics_port=9003 \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=8080 \
    --log_dir=/var/log/mooncake
```

##### 启动 Client

```bash
# 环境变量配置 (推荐方式)
export MOONCAKE_MASTER=192.168.1.10:50051
export MOONCAKE_TE_META_DATA_SERVER=http://192.168.1.10:8080/metadata
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=mlx5_1,mlx5_2
export MOONCAKE_GLOBAL_SEGMENT_SIZE=8GB
export MOONCAKE_LOCAL_BUFFER_SIZE=2GB
export MOONCAKE_LOCAL_HOSTNAME=192.168.1.20   # 必须是外部可达 IP，不能是 127.0.0.1

# 启动推理服务 (以 vLLM 为例)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --tensor-parallel-size 8 \
    --disaggregated-config prefill_config.yaml
```

##### 裸机部署的典型问题

| 问题 | 原因 | 解法 |
|------|------|------|
| `ibv_open_device` 失败 | RDMA 驱动未装或权限不足 | `ibv_devinfo` 排查，检查 `/dev/infiniband/` 权限 |
| 性能远低于预期 | 实际走了 TCP 回退 | 检查日志中是否有 "fall back to TCP"，确认 `MOONCAKE_PROTOCOL=rdma` |
| HugePage 分配失败 | 内存碎片化 | 重启后再分配，或用 1GB 大页 |
| `MOONCAKE_LOCAL_HOSTNAME=127.0.0.1` | 其他节点无法连接 | 必须设为外部可达 IP |

---

### 形态 2：容器化——官方镜像与设备透传

##### 官方 Master 镜像

Mooncake 提供官方 Docker 镜像 `docker.io/kvcacheai/mooncake`，这是**唯一推荐的 Master 容器镜像**：

```
镜像构建 (两阶段):
  构建阶段: nvidia/cuda:12.8.1-devel-ubuntu22.04
    → 安装编译依赖、CMake 构建、打包 wheel

  运行阶段: nvidia/cuda:12.8.1-runtime-ubuntu22.04
    → 只安装运行时库: ibverbs, rdmacm, liburing, yaml, curl
    → pip install wheel
    → 入口: tini -g (信号转发)
```

关键设计：**Master 镜像不包含真实 GPU 运行时**——它用 stub libcuda/libcudart 满足链接器，运行时不需要 GPU。GPU 只在推理节点（Client 侧）需要。

##### 容器启动 Master

```bash
docker run -d \
    --name mooncake-master \
    --network host \
    -v /var/log/mooncake:/var/log/mooncake \
    -e MOONCAKE_MASTER=0.0.0.0:50051 \
    -e MOONCAKE_TE_META_DATA_SERVER=http://0.0.0.0:8080/metadata \
    kvcacheai/mooncake:latest \
    mooncake_master \
        --rpc_port=50051 \
        --enable_http_metadata_server=true \
        --enable_metric_reporting=true \
        --metrics_port=9003
```

##### 容器启动 Client（RDMA 节点）

RDMA 容器是最复杂的部署形态——需要透传 RDMA 设备、HugePage、共享内存：

```bash
docker run -d \
    --name mooncake-client \
    --network host \
    --device /dev/infiniband/uverbs0 \     # RDMA 设备
    --device /dev/infiniband/rdma_cm \      # RDMA 连接管理
    --volume /dev/infiniband:/dev/infiniband \  # 或整体挂载
    --shm-size=64g \                        # 共享内存 (KV Cache 交换)
    --ulimit memlock=-1:-1 \                # 解除 memlock 限制
    -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \  # HugePage
    -e MOONCAKE_MASTER=192.168.1.10:50051 \
    -e MOONCAKE_PROTOCOL=rdma \
    -e MOONCAKE_DEVICE=mlx5_1,mlx5_2 \
    -e MOONCAKE_GLOBAL_SEGMENT_SIZE=8GB \
    -e MOONCAKE_LOCAL_HOSTNAME=192.168.1.20 \
    --gpus all \                            # GPU 透传
    kvcacheai/mooncake:latest \
    python -m vllm.entrypoints.openai.api_server ...
```

##### Ascend NPU 容器

```bash
docker run -d \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    ... \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -e LD_PRELOAD=/usr/lib64/libjemalloc.so.2 \
    ...
```

##### 镜像发布流程

官方镜像的发布有严格的质量门禁：

```
1. 版本号校验: 必须匹配 ^[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$
2. PyPI 预检: 确认 cp312 的 x86_64 和 aarch64 wheel 都已发布
3. 多架构构建: linux/amd64 + linux/arm64
4. 冒烟测试 (amd64): import mooncake.engine, mooncake.store + mooncake_master --version
5. 冒烟测试 (arm64/QEMU): 同上，在 QEMU 模拟器中验证
6. :latest 标签: 只在冒烟通过后才用 imagetools create 按摘要复制
   → :latest 永远不会指向冒烟失败的构建
```

---

### 形态 3：K8s 编排——最复杂也最生产级

Mooncake本身 **不提供 K8s YAML 清单**——它提供的是 K8s 集成所需的库代码和接口，部署清单由运维团队编写。这就像提供了发动机和方向盘，但车身需要自己组装。

不过 Mooncake 官方在部署文档环节还是给了 K8s 部署 YAML，体现了生态的完整性。

一般情况下，是和推理引擎，比如vLLM或者SGLang一起组装到K8s里面部署的。Mooncake 自身的K8s部署文档，官方也给了详细介绍，链接在文末。

##### K8s 集成的四个核心组件

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Kubernetes 集群                                    │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Master Pod A │    │ Master Pod B │    │ Master Pod C │             │
│  │  (Leader)    │    │  (Standby)   │    │  (Standby)   │             │
│  │              │    │              │    │              │             │
│  │ ① K8s Lease  │    │ ① K8s Lease  │    │ ① K8s Lease  │             │
│  │   选举成功    │    │   等待选举    │    │   等待选举    │             │
│  │              │    │              │    │              │             │
│  │ ② Pod Label  │    │              │    │              │             │
│  │   role=leader│    │              │    │              │             │
│  └──────┬───────┘    └─────────────┘    └─────────────┘             │
│         │                                                            │
│         │ ③ K8s Service (label selector: role=leader)               │
│         │    → 自动路由到当前 Leader Pod                              │
│         │                                                            │
│    ┌────▼─────┐                                                      │
│    │ Client   │  ④ 通过 Service DNS 发现 Leader                      │
│    │ Pods     │     mooncake-master.default.svc.cluster.local:50051  │
│    └──────────┘                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

##### 组件 1：K8s Lease 选举

Mooncake 的 K8s 选举通过 `libk8s_lease_wrapper.so`（Go c-shared 库）实现，使用 Kubernetes 原生的 `coordinationv1.Lease` 对象：

```go
// mooncake-common/k8s-lease/k8s_lease_wrapper.go
// 使用 client-go 的 leaderelection 包

lock := &resourcelock.LeaseLock{
    LeaseMeta: metav1.ObjectMeta{
        Name:      "mooncake-master",    // Lease 对象名
        Namespace: "mooncake-system",    // 命名空间
    },
    Client: globalClient.CoordinationV1(),
    LockConfig: resourcelock.ResourceLockConfig{
        Identity: "node1:50051",         // Master 的 RPC 地址
    },
}

// 选举参数 (C++ 侧默认值)
LeaseDuration:  5 秒    // Lease 有效期
RenewDeadline:  3 秒    // 续约截止时间
RetryPeriod:    1 秒    // 重试间隔
ReleaseOnCancel: true   // 退出时主动释放 Lease (比 etcd 后端更干净)
```

**关键细节——过期 Lease 处理**：当 Leader Pod 非正常退出（如节点宕机），它无法执行 `ReleaseOnCancel`，Lease 会一直保留直到过期。Mooncake 的代码检测到过期 Lease 后，将其视为"无持有者"，允许 Standby 接管：

```go
// k8s_lease_wrapper.go:211-219
if holder != "" && lease.Spec.RenewTime != nil && lease.Spec.LeaseDurationSeconds != nil {
    expiry := lease.Spec.RenewTime.Time.Add(
        time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second)
    if time.Now().After(expiry) {
        holder = ""  // 过期 Lease 视为无持有者
    }
}
```

这意味着**故障转移时间 = LeaseDuration (5s) + 检测延迟**，比 etcd 的租约过期更快。

##### 组件 2：Pod Label 路由

选举成功后，Leader Pod 会被打上标签 `mooncake.io/store-role=leader`：

```cpp
// mooncake-store/src/master_service_supervisor.cpp:39-50
// 选举成功后设置 Pod Label
K8sLeaseHelper::SetPodLabel(pod_namespace, pod_name,
    "mooncake.io/store-role", "leader");
```

这个标签是**服务发现的关键**——K8s Service 用 label selector 跟随 Leader：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mooncake-master
spec:
  selector:
    mooncake.io/store-role: leader    # 自动路由到当前 Leader
  ports:
    - name: rpc
      port: 50051
      targetPort: 50051
    - name: metrics
      port: 9003
      targetPort: 9003
```

当 Leader 切换时，标签从旧 Pod 移除、打上新 Pod，Service 自动更新端点——**Client 无需知道 Leader 是谁**。

##### 组件 3：配置体系——四层配置优先级

Mooncake 的配置从高到低有四层：

```
优先级 (高 → 低):

1. 命令行参数 (gflags)
   --rpc_port=50051 --enable_ha=true

2. 配置文件 (YAML/JSON, --config_path)
   rpc_port: 50051
   enable_ha: true

3. MOONCAKE_* 环境变量 (高层语义)
   MOONCAKE_MASTER=mooncake-master:50051
   MOONCAKE_PROTOCOL=rdma

4. MC_* 环境变量 (底层引擎调优)
   MC_STORE_USE_HUGEPAGE=1
   MC_MS_AUTO_DISC=1
```

##### 组件 4：资源调度——GPU、RDMA、HugePage 三重亲和性

K8s 部署 Mooncake 的最大挑战是**三种特殊资源的协同调度**：

```
GPU:   nvidia.com/gpu: 8          ← NVIDIA Device Plugin
RDMA:  rdma/0000:3b:00.0: 1       ← RDMA Device Plugin / Network Operator
HugePage: hugepages-2Mi: 512Gi    ← K8s 内置 HugePage 资源
```

推理节点的 Pod 必须同时获得这三种资源，且它们必须在**同一个 NUMA 节点**上：

```yaml
# 推理节点 Pod 示例
resources:
  limits:
    nvidia.com/gpu: 8
    rdma/0000:3b:00.0: 1           # RDMA 网卡 0
    rdma/0000:3b:00.1: 1           # RDMA 网卡 1
    hugepages-2Mi: 512Gi
    memory: 512Gi
  requests:
    nvidia.com/gpu: 8
    rdma/0000:3b:00.0: 1
    rdma/0000:3b:00.1: 1
    hugepages-2Mi: 512Gi
    memory: 512Gi
```

**拓扑约束**：GPU 和 RDMA 网卡必须在同一个 NUMA 节点，否则 PCIe 跨 NUMA 访问会严重降低 RDMA 带宽。K8s 的 `TopologyManager` 配合 Device Plugin 的 NUMA 亲和信息可以实现这一点，但需要：

1. NVIDIA Device Plugin 开启 `--topology-manager-policy=single-numa-node`
2. RDMA Device Plugin 报告网卡的 NUMA 亲和性
3. Pod 的资源请求在同一个 NUMA 节点上满足

##### K8s 部署中 Mooncake 不提供的（需要自己做的）

| 需要自己做的 | 说明 | 参考实现 |
|------------|------|---------|
| Deployment/StatefulSet YAML | Mooncake 不提供 K8s 清单 | 根据本文模板编写 |
| RDMA Device Plugin | 透传 RDMA 网卡到 Pod | NVIDIA Network Operator / k8s-rdma-device-plugin |
| HugePage 预分配 | 节点级 sysctl 配置 | `vm.nr_hugepages=262144` + DaemonSet 初始化 |
| NetworkPolicy | 网络隔离策略 | 按安全需求编写 |
| Log Aggregation | 日志收集 (Fluentd/Loki) | 标准 K8s 日志方案 |
| HPA/VPA | 弹性伸缩 | 需自定义 Metrics (Prometheus Adapter) |

---

### 配置体系详解：从 gflags 到环境变量

##### Master 的关键配置参数

| 参数 | 默认值 | 说明 | 生产建议 |
|------|--------|------|---------|
| `--rpc_port` | 50051 | Master RPC 端口 | — |
| `--rpc_thread_num` | 1 | RPC 线程数 | 设为 4-8 |
| `--rpc_interface` | — | 从网卡名解析 IP | 容器中用 `eth0` |
| `--enable_ha` | false | 启用 HA | 生产必须 |
| `--ha_backend_type` | etcd | HA 后端: etcd/redis/k8s | K8s 环境用 k8s |
| `--enable_http_metadata_server` | false | HTTP 元数据服务 | 推荐 true |
| `--enable_metric_reporting` | true | Prometheus 指标 | 保持 true |
| `--metrics_port` | 9003 | 指标端口 | — |
| `--pod_name` | — | K8s Pod 名 (HA 用) | `metadata.name` |
| `--pod_namespace` | — | K8s 命名空间 | `metadata.namespace` |
| `--config_path` | — | 配置文件路径 | 推荐 ConfigMap 挂载 |

##### Client 的环境变量配置

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `MOONCAKE_MASTER` | Master 地址 | `mooncake-master:50051` |
| `MOONCAKE_TE_META_DATA_SERVER` | 元数据服务 | `http://master:8080/metadata` |
| `MOONCAKE_PROTOCOL` | 传输协议 | `rdma` / `tcp` / `efa` / `cxl` |
| `MOONCAKE_DEVICE` | RDMA 设备列表 | `mlx5_1,mlx5_2` |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 贡献的 DRAM 大小 | `8GB` |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | 本地缓冲区大小 | `2GB` |
| `MOONCAKE_LOCAL_HOSTNAME` | 外部可达 IP | **不能是 127.0.0.1** |
| `MOONCAKE_TENANT_ID` | 租户 ID | 多租户场景必填 |
| `MC_STORE_USE_HUGEPAGE` | 启用 HugePage | `1` |
| `MC_STORE_HUGEPAGE_SIZE` | 大页大小 | `2MB` / `1GB` |
| `MC_MS_AUTO_DISC` | RDMA 拓扑自动发现 | `1` |

---

### 服务发现：四种元数据服务

Mooncake 支持四种元数据服务，从去中心化到完全托管：

| 方式 | 架构 | 依赖 | 适用场景 |
|------|------|------|---------|
| P2PHANDSHAKE | 去中心化，节点间直连 | 无 | 快速验证、小集群 |
| HTTP 元数据服务器 | Master 内嵌 | Master 进程 | 单 Master 部署 |
| etcd | 外部集群 | etcd 集群 | 生产级 HA |
| Redis | 外部集群 | Redis 实例 | 已有 Redis 基础设施 |

```
P2PHANDSHAKE (推荐起点):
  节点 A ←→ 节点 B    直连交换元数据
  无中心服务，无单点故障
  限制: 不支持跨网段路由

HTTP 元数据服务器:
  Client → Master:8080/metadata → 查询/注册段信息
  内嵌在 Master 中，无需额外部署
  限制: 单 Master 无 HA

etcd (生产推荐):
  Client → etcd 集群 → 查询段信息 + Leader 选举
  强一致性，支持 HA
  限制: 需要运维 etcd 集群
```

---

### 健康检查与监控：让系统"可观测"

##### 健康检查端点

Master 暴露一组 HTTP 端点，适合用作 K8s 探针：

| 端点 | 用途 | K8s 探针类型 |
|------|------|------------|
| `GET /health` | 健康状态 | livenessProbe |
| `GET /role` | 当前角色 (leader/standby) | readinessProbe |
| `GET /ha_status` | 完整 HA 状态 | — |
| `GET /leader` | 当前 Leader 信息 | — |
| `GET /metrics` | Prometheus 指标 | — |

K8s 探针配置示例：

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 9003
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /role              # 只有 Leader 才 ready
    port: 9003
  initialDelaySeconds: 5
  periodSeconds: 5
```

##### 监控栈

Mooncake 提供开箱即用的 Prometheus + Grafana 监控：

```
mooncake-master:9003/metrics    ← Prometheus 抓取
        │
        ▼
Prometheus (v2.54.0)           ← 存储 + 查询
        │
        ▼
Grafana (11.0.0)               ← 可视化面板
        │
        └→ mooncake.json 仪表盘: RPC 请求速率
```

启动方式：

```bash
# 方式 1: docker-compose
cd monitoring && docker-compose up -d

# 方式 2: 手动脚本
cd monitoring && bash run.sh
```

关键指标：

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `mooncake_master_rpc_requests_total` | RPC 请求总数 | — |
| `mooncake_tenant_quota_used_bytes` | 租户已用配额 | > 90% 配额 |
| `mooncake_tenant_evict_bytes_total` | 驱逐字节数 | 突增 = 容量不足 |

---

### 启动与关闭：优雅的生命周期管理

##### 启动序列

```
Master 启动流程:

1. 解析 gflags + 配置文件
2. 解析 RPC 地址 (如果设了 --rpc_interface，从网卡名解析 IP)
3. 启动 HTTP 元数据服务器 (如果启用)
4. 分支:
   ├── HA 模式:
   │   ├── 启动 MasterServiceSupervisor
   │   ├── 参与选举 (etcd/Redis/K8s Lease)
   │   ├── 等待当选 Leader
   │   └── 当选后启动 RPC 服务 + 设置 Pod Label
   └── 非 HA 模式:
       └── 直接启动 coro_rpc_server
5. 启动指标上报线程 (每周期输出 HA 状态 + Master 摘要)
```

##### 优雅关闭

```
关闭流程:

1. 收到 SIGTERM (容器: tini -g 转发给所有子进程)
2. MasterAdminServer::Stop()
   ├── 停止指标上报线程
   ├── 停止 HTTP 服务器
   └── 释放停止信号量
3. K8s Lease: ReleaseOnCancel=true → 主动释放 Lease
   → Standby 立即可参与选举 (无需等 Lease 过期)
4. Snapshot 子进程: SIGTERM → 等待数秒 → SIGKILL (兜底)
5. Client 侧: gracefully_unmounting_segments_ 标记段为"只读"
   → 排空进行中的传输 → 删除段
```

**K8s Lease vs etcd 的关闭差异**：

| | K8s Lease | etcd |
|--|----------|------|
| 正常关闭 | 主动释放 Lease → Standby 立即选举 | 依赖租约过期 (~5s) |
| 异常关闭 (节点宕机) | 等 Lease 过期 (5s) → Standby 接管 | 等租约过期 → Standby 接管 |
| 故障转移时间 | 正常: <1s; 异常: ~5s | 正常: ~5s; 异常: ~5s |

---

### PD 解耦部署：Prefill 和 Decode 分开调度

PD 解耦是 Mooncake 的核心场景，也是部署最复杂的场景——Prefill 和 Decode 是不同的推理进程，运行在不同的节点集合上，通过 RDMA 传输 KV Cache。

##### 部署拓扑

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Prefill 集群 (GPU 密集)              Decode 集群 (GPU + 内存密集)   │
│                                                                      │
│  ┌──────────┐  ┌──────────┐         ┌──────────┐  ┌──────────┐    │
│  │ P-Node 0 │  │ P-Node 1 │         │ D-Node 0 │  │ D-Node 1 │    │
│  │ 8×A100   │  │ 8×A100   │         │ 8×A100   │  │ 8×A100   │    │
│  │ 2×RDMA   │  │ 2×RDMA   │         │ 2×RDMA   │  │ 2×RDMA   │    │
│  │ vLLM P   │  │ vLLM P   │         │ vLLM D   │  │ vLLM D   │    │
│  └────┬─────┘  └────┬─────┘         └────┬─────┘  └────┬─────┘    │
│       │              │                    │              │          │
│       └──────┬───────┘                    └──────┬───────┘          │
│              │                                   │                  │
│              │         ┌──────────────┐          │                  │
│              └────────→│   Master     │←─────────┘                  │
│                        │  (HA 集群)    │                             │
│                        └──────────────┘                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

##### 关键配置差异

| 配置 | Prefill 节点 | Decode 节点 |
|------|-------------|------------|
| vLLM 角色 | `prefill` | `decode` |
| GPU 用途 | 计算密集 (Prefill 算 Q/K/V) | 内存密集 (存 KV Cache) |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 较小 (不需要存 KV Cache) | 较大 (存所有请求的 KV Cache) |
| RDMA 方向 | 主要发 (RDMA WRITE) | 主要收 |
| 扩缩容策略 | 按 Prefill 延迟扩容 | 按 KV Cache 内存压力扩容 |

##### 调度亲和性

在 K8s 中，Prefill 和 Decode 应该部署在不同的节点池，使用不同的标签：

```yaml
# Prefill 节点池
nodeSelector:
  node-role: prefill
  nvidia.com/gpu.count: "8"

# Decode 节点池
nodeSelector:
  node-role: decode
  nvidia.com/gpu.count: "8"
  hugepages-2Mi: available    # Decode 节点需要大页
```

---

### SSD 分层存储部署：三种后端的选择

当 DRAM 装不下所有 KV Cache，需要卸载到 SSD。Mooncake 提供三种 SSD 存储后端：

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| `bucket_storage_backend` | 批量文件、FIFO/LRU 驱逐、支持重启恢复 | **推荐**，生产首选 |
| `file_per_key_storage_backend` | 每个对象一个文件、简单 | 小规模验证 |
| `offset_allocator_storage_backend` | 预分配单文件、高性能 | **不支持重启恢复**，临时场景 |

```bash
# SSD 卸载配置
export MOONCAKE_OFFLOAD_ENABLED=true
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/nvme/mooncake_offload
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=2147483648000  # 2TB
export MOONCAKE_OFFLOAD_USE_URING=true  # 启用 io_uring (推荐)
```

NVMe-oF 部署需要额外的 SPDK 配置和 HugePage 预分配：

```bash
# SPDK 需要的 HugePage (至少 512 页)
echo 512 | sudo tee /proc/sys/vm/nr_hugepages

# 启动 SPDK target
python mooncake-wheel/mooncake/spdk_tgt_create.py
```

---

### 总结与行动指南

| 部署形态 | 适合场景 | 核心挑战 | Mooncake 提供的 |
|---------|---------|---------|----------------|
| 裸机直装 | 快速验证、小规模 | 硬件环境配置 | pip wheel、环境变量 |
| 容器化 | 单节点、CI/CD | RDMA/HugePage 透传 | 官方镜像、tini 信号转发 |
| K8s 编排 | 生产级多节点 | 资源亲和性、HA、服务发现 | K8s Lease 库、Pod Label 路由、健康端点 |

**建议**: Mooncake 的部署哲学是"提供积木，不提供蓝图"——K8s Lease 选举、Pod Label 路由、健康检查端点这些积木都准备好了，但 Deployment YAML、Device Plugin 配置、NetworkPolicy 需要你根据集群环境自己组装。建议从裸机验证 → 容器化封装 → K8s 编排逐步推进，每一层都跑通后再进入下一层。

**延伸阅读**：
- Mooncake 部署指南：https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide
- NVIDIA K8s Device Plugin：https://github.com/NVIDIA/k8s-device-plugin
- K8s HugePage 管理：https://kubernetes.io/docs/tasks/manage-hugepages/scheduling-hugepages/

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
