# Mooncake 部署踩坑实录：那些文档没告诉你的事儿

> **系列**: Mooncake 技术博客系列 | **类型**: 踩坑避坑篇
>
> 部署 Mooncake 就像驾驶一艘新式快艇穿越暗礁密布的海峡——海图上标注的航道很清晰，但真正让你搁浅的，往往是那些藏在水面下的礁石：RDMA 驱动版本不兼容、etcd 连接超时、GPU 内存注册失败……

理想很美好，一个系统要真正商用落地，会经历大量的工程工作，而工程工作会遇到非常繁杂的系统问题，往往要经历九九八十一难，疑难问题定位分析、性能问题诊断排查，有时加班熬夜甚至通宵达旦，经历大量的dirty work，才能打磨出生产级稳定运行的系统。

---

### 引言

想象你是一名远洋船长，手握一张标注了主航道的海图，满怀信心地驶向目的地。海图上清清楚楚地画着从 A 到 B 的最优路径，就像 Mooncake 官方文档告诉你的：`pip install mooncake-transfer-engine`，配置 etcd，启动 Master，一行命令跑起来。

然而，真正让你心惊肉跳的从来不是主航道上的风浪，而是那些海图上没有标注的暗礁——RDMA 网卡权限不足、QP 创建失败、Master 内存分配策略选错导致 OOM……

我们团队在部署 Mooncake 的过程中，踩过形形色色的坑。

今天这篇文章，就是我们在 Mooncake 部署航行中绘制的"暗礁分布图"。每一个坑位，我们都按照"症状 → 原因 → 解法 → 预防"的格式完整记录。

---

### 一、RDMA 环境配置：最常见的暗礁

RDMA 环境配置是部署 Mooncake 的第一大坑。它涉及驱动、网卡、权限、内核版本等多个层面，任何一环出问题都会导致 RDMA 不可用。

#### 1.1 症状：RDMA 初始化失败

```
ERROR: Failed to create RDMA context for device mlx5_0
ERROR: ibv_open_device failed: Cannot allocate memory
```

或更隐蔽的症状：Mooncake 可以启动，但实际走的是 TCP 回退路径，性能远低于预期。

#### 1.2 原因：RDMA 驱动和权限问题

| 原因 | 检查方法 |
|------|---------|
| RDMA 驱动未安装 | `ibv_devinfo` 无输出 |
| 网卡权限不足 | `ibv_devinfo` 报 Permission denied |
| 内核版本过低 | `uname -r` 检查是否 >= 5.x |
| MOFED 版本不兼容 | `ofed_info -s` 检查版本 |
| RDMA 设备被其他进程占用 | `ibstat` 检查设备状态 |

#### 1.3 解法：逐步排查

```bash
# 1. 检查 RDMA 设备
ibv_devinfo
# 应该看到 mlx5_0, mlx5_1 等设备信息

# 2. 检查设备权限
ls -la /dev/infiniband/rdma_cm
# 应该有读写权限

# 3. 测试 RDMA 带宽
ib_write_bw -d mlx5_0
# 应该能达到网卡标称带宽

# 4. 检查 MOFED 版本
ofed_info -s
# 建议使用 MLNX_OFED 5.x+

# 5. 添加用户到 rdma 组
sudo usermod -aG rdma $USER
```

#### 1.4 预防

- 部署前先用 `ibv_devinfo` 和 `ib_write_bw` 验证 RDMA 环境
- 使用容器部署时，确保容器有 `--device /dev/infiniband` 和 `--cap-add IPC_LOCK`
- 记录 MOFED 版本，确保与 Mooncake 编译时使用的版本一致

> 笔者注：如果 `ibv_devinfo` 无输出，不要急着安装 MOFED——先确认你的网卡确实是 Mellanox/ConnectX 系列。其他厂商的 RDMA 网卡（如华为、Broadcom）需要不同的驱动。

---

### 二、etcd 连接问题：元数据服务的"咽喉"

#### 2.1 症状：Transfer Engine 初始化超时

```
ERROR: Failed to connect to metadata server: etcd://127.0.0.1:2379
ERROR: Connection timeout after 30s
```

#### 2.2 原因

| 原因 | 检查方法 |
|------|---------|
| etcd 未启动 | `systemctl status etcd` |
| 防火墙阻止 2379 端口 | `telnet 127.0.0.1 2379` |
| etcd 集群不健康 | `etcdctl endpoint health` |
| 连接串格式错误 | 检查是否为 `etcd://host:port` |

#### 2.3 解法

```bash
# 1. 检查 etcd 状态
etcdctl endpoint health --endpoints=http://127.0.0.1:2379

# 2. 检查端口可达性
telnet 127.0.0.1 2379

# 3. 使用 P2PHANDSHAKE 模式（无需 etcd）
# 连接串改为 "P2PHANDSHAKE://"
```

#### 2.4 预防

- 生产环境使用 etcd 集群（3 节点），避免单点故障
- 开发测试可用 P2PHANDSHAKE 模式，无需部署 etcd
- 注意 etcd 的连接串格式：`etcd://` 前缀是必需的

---

### 三、GPU 内存注册失败：GPUDirect RDMA 的"门槛"

#### 3.1 症状：registerLocalMemory 失败

```
ERROR: Failed to register GPU memory at 0x7f0000000000
ERROR: ibv_reg_mr failed: Invalid argument
```

#### 3.2 原因

| 原因 | 说明 |
|------|------|
| GPUDirect RDMA 未启用 | 需要加载 nvidia-peermem 模块 |
| GPU 内存未 pin 住 | RDMA 需要物理连续内存 |
| CUDA 版本不兼容 | GPUDirect 需要 CUDA 11.x+ |
| 网卡不支持 GPUDirect | 非 Mellanox ConnectX 网卡可能不支持 |

#### 3.3 解法

```bash
# 1. 加载 nvidia-peermem 模块
sudo modprobe nvidia-peermem

# 2. 验证 GPUDirect RDMA 可用
cat /proc/driver/nvidia/gpus/0000\:3b\:00.0/peermem
# 应该显示支持

# 3. 如果不支持 GPUDirect，使用 CPU 中转模式
# 注册内存时标记位置为 "dram" 而非 "gpu:0"
engine.registerLocalMemory(cpu_buffer, size, "dram");
```

#### 3.4 预防

- 部署前确认 GPU 和网卡的 GPUDirect 兼容性
- 使用 `nvidia-peermem` 模块确保内核支持
- 如果 GPUDirect 不可用，退而求其次使用 CPU 中转——功能不受影响，性能比较差

---

### 四、Master 内存分配策略选错

#### 4.1 症状：Store 写入失败或 OOM

```
ERROR: PutStart failed: no available segments
ERROR: Out of memory: allocation failed
```

#### 4.2 原因

| 策略 | 问题场景 |
|------|---------|
| `random` | 某些 Segment 被打满，其他 Segment 空闲 |
| `free_ratio_first` | 大量小对象导致内存碎片 |
| `ssd_free_ratio_first` | SSD 配置不正确 |

#### 4.3 解法

```bash
# 对于 LLM 推理场景（对象大小均匀），使用默认的 random
--allocation_strategy=random

# 对于对象大小差异大的场景，使用 free_ratio_first
--allocation_strategy=free_ratio_first

# 对于有 SSD 分层的场景
--allocation_strategy=ssd_free_ratio_first
--enable_offload=true
--ssd_offload_path=/data/ssd
```

#### 4.4 预防

- 监控 Master 的内存水位指标（Prometheus `--metrics_port=9003`）
- 设置合理的驱逐水位：`--eviction_high_watermark_ratio=0.8`
- 预留 20% 的内存缓冲区

---

### 五、多网卡环境下的拓扑发现问题

#### 5.1 症状：带宽远低于预期

在 4×200 Gbps 网络下，实际带宽只有 25 GB/s（单网卡水平），而非预期的 87 GB/s。

#### 5.2 原因

| 原因 | 说明 |
|------|------|
| 拓扑发现失败 | preferred_hca 列表为空 |
| 设备过滤不当 | MC_TE_FILTERS 排除了部分网卡 |
| NUMA 信息不可用 | 容器内无法读取 sysfs |

#### 5.3 解法

```bash
# 1. 检查拓扑发现结果
# 在代码中调用 engine.showLinks(true) 查看 JSON 输出

# 2. 检查 NUMA 信息
ls /sys/class/infiniband/mlx5_0/device/numa_node
# 应该返回 0 或 1

# 3. 容器内需要挂载 sysfs
docker run -v /sys:/sys ...

# 4. 手动指定设备过滤
export MC_TE_FILTERS="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
```

#### 5.4 预防

- 部署时用 `showLinks` 验证拓扑发现结果
- 容器部署时确保 sysfs 可读
- 在 NUMA 架构的服务器上，确保进程绑定到正确的 NUMA 节点

---

### 六、QP 资源耗尽

#### 6.1 症状：大规模集群下连接失败

```
ERROR: Failed to create QP: cannot allocate memory
ERROR: ibv_create_qp failed: Resource temporarily unavailable
```

#### 6.2 原因

每个 RDMA 连接需要一对 QP，QP 占用网卡内存。在 1000+ 节点集群中，每个节点可能需要数千个 QP。

#### 6.3 解法

```bash
# 1. 增大最大 Endpoint 数
export MC_MAX_EP_PER_CTX=131072

# 2. 检查网卡 QP 容量
cat /sys/class/infiniband/mlx5_0/ports/1/gid_table/0
ibv_devinfo -d mlx5_0 | grep max_qp
# 典型值：131072

# 3. 使用 SIEVE 淘汰策略（默认启用）
# 不活跃的 Endpoint 会被自动回收
```

#### 6.4 预防

- 大规模集群部署前，计算 QP 需求：每个节点需要 `num_peers * num_qps_per_peer` 个 QP
- 监控 QP 使用率
- 考虑使用连接复用或共享 QP 策略

---

### 七、容器部署的特殊坑

#### 7.1 症状：容器内 RDMA 不可用

```
ERROR: No IB devices found
```

#### 7.2 原因与解法

```bash
# Docker 运行时需要额外参数
docker run \
    --device /dev/infiniband \
    --device /dev/infiniband/rdma_cm \
    --cap-add IPC_LOCK \
    --cap-add NET_ADMIN \
    -v /sys:/sys \
    --network host \
    ...
```

| 参数 | 说明 |
|------|------|
| `--device /dev/infiniband` | 暴露 RDMA 设备 |
| `--cap-add IPC_LOCK` | 允许内存锁定（RDMA 需要） |
| `--cap-add NET_ADMIN` | 允许网络管理操作 |
| `-v /sys:/sys` | 挂载 sysfs（拓扑发现需要） |
| `--network host` | 使用主机网络（简化 RDMA 配置） |

---

### 八、环境变量速查表

| 环境变量 | 默认值 | 说明 | 常见问题 |
|---------|-------|------|---------|
| `MC_TRANSFER_TIMEOUT` | 30s | 传输超时 | 太短导致大传输失败 |
| `MC_TE_FILTERS` | - | 网卡白名单 | 设置错误导致网卡不可用 |
| `MC_MAX_EP_PER_CTX` | 65536 | 最大 Endpoint 数 | 大集群需要调大 |
| `MC_IB_PORT` | 1 | IB 端口 | 多端口网卡需正确设置 |
| `MC_GID_INDEX` | 0 | GID 索引 | RoCEv2 需要设为非零值 |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | false | 目标设备亲和性 | 跨 NUMA 场景建议开启 |
| `MC_DISABLE_METACACHE` | false | 禁用元数据缓存 | 调试时开启 |

---

### 总结：部署检查清单

| 检查项 | 命令 | 预期结果 |
|--------|------|---------|
| RDMA 设备 | `ibv_devinfo` | 显示设备信息 |
| RDMA 带宽 | `ib_write_bw` | 接近网卡标称带宽 |
| etcd 连接 | `etcdctl endpoint health` | healthy |
| GPUDirect | `cat /proc/driver/nvidia/gpus/*/peermem` | 支持 |
| 拓扑发现 | `engine.showLinks()` | preferred_hca 非空 |
| QP 容量 | `ibv_devinfo \| grep max_qp` | 满足集群规模 |
| 内存水位 | Prometheus metrics | 低于 high_watermark |

**建议**: 部署 Mooncake 前，先按此清单逐项检查——90% 的部署问题都出在环境配置上，而非 Mooncake 本身。

**延伸阅读**：
- Mooncake 故障排除文档：https://kvcache-ai.github.io/Mooncake/troubleshooting/troubleshooting.html
- Mooncake 错误码参考：https://kvcache-ai.github.io/Mooncake/troubleshooting/error-code.html
- Mooncake 部署指南：https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。感谢阅读！*
