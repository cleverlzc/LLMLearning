# Mooncake 安全实践：六道信任边界的攻防审视

> **系列**: Mooncake 技术博客系列 | **类型**: 安全实践篇
>
> 分布式系统的安全，根本原则只有一个：**每个组件边界都是信任边界/非信任边界**。把系统拆成多少个组件，就有多少道需要守护的门。Mooncake 有六道门——网络、内存、数据、进程、凭证、编译——默认配置全部敞开。

本文从攻击者视角出发，逐道审视这些信任边界：哪里有锁，哪里没锁，以及如何在生产环境中做安全加固。

---

### 引言：分布式系统的信任模型

一个单机程序的安全很简单——进程隔离 + 操作系统兜底。但分布式系统不一样：**组件之间通过网络通信，网络是不可信的；数据跨节点存储，存储是不可信的；进程跨机器部署，身份是不可信的。**

每多一个组件边界，就多一道信任检查点。Mooncake 的架构拆成 Master、Agent、Transfer Engine、Store、etcd 五大组件，跨 GPU/DRAM/SSD 三级存储，走 RDMA/TCP/NVMe-oF 三条传输路径——**组件越多、介质越多、路径越多，信任边界越多**。

```
信任边界的本质:

单机程序:     [进程] ← 操作系统隔离 → [进程]        1 道门
分布式系统:   [节点A] ← 网络 ← ??? → [节点B]       N 道门
              ↑ 谁都能发数据过来，你信任谁？

Mooncake:     6 道信任边界 × 默认全部敞开 = 攻击面全景
```

Mooncake 的默认安全模型是**信任内网**——假设所有能访问集群网络的节点都是善意的。这在**私有集群**中可以接受，性能最大化，但在多租户云环境或跨组织协作中不够。

**RDMA 是其中最危险的例证**——网卡绕过 CPU 直接读写内存，拿到 RKey 就等于拿到远程内存的万能钥匙，远程 CPU 完全无感。但 RDMA 只是六道信任边界中的一道（内存边界）的具体表现，不是全貌。

今天这篇文章，我们从攻击者视角出发，逐道审视 Mooncake 的六道信任边界。

---

### 攻击面全景：六个安全域

Mooncake 的安全边界可以划分为五个域：

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  域 1: 网络边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Master RPC (0.0.0.0:50051)    ← 无认证·无 TLS              │    │
│  │  P2P Handshake (0.0.0.0:12001) ← 无认证·接受任意连接         │    │
│  │  etcd (http://localhost:2379)   ← 无认证·明文 HTTP           │    │
│  │  RDMA Fabric                    ← 无加密·依赖网络隔离         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  域 2: 内存边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  RDMA RKey/LKey → 明文存储在 etcd/HTTP 元数据中              │    │
│  │  共享内存 (shm_open 0666) → 任何本地用户可读写               │    │
│  │  memfd_create → fd 可通过 Unix Socket 传递                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  域 3: 数据边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  SSD 存储 → 明文写入，无加密                                 │    │
│  │  RDMA/TCP 传输 → 明文传输，无加密                            │    │
│  │  S3 存储 → 可选 HTTPS + 校验和                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  域 4: 进程边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Docker → 默认 root 运行                                     │    │
│  │  SPDK → 需要 root 或 VFIO 权限                               │    │
│  │  无 seccomp · 无 capability dropping                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  域 5: 凭证边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  AWS 密钥 → 环境变量明文                                     │    │
│  │  Redis 密码 → 环境变量明文                                   │    │
│  │  etcd → 无认证                                               │    │
│  │  Master RPC → 无认证                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  域 6: 编译边界                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  全局 → 仅 -fPIC，无栈保护/无 RELRO/无 FORTIFY              │    │
│  │  Ascend 模块 → 唯一启用安全编译选项的模块                    │    │
│  │  UBSAN/TSAN/MSAN → 构建系统未配置                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

下面逐域分析。

---

### 域 1：网络边界——最危险的攻击面

##### Master RPC：无认证 + 无加密 = 任何人都能读写你的 KV Cache

```cpp
// mooncake-store/src/master.cpp
DEFINE_string(rpc_address, "0.0.0.0", "RPC listen address");
// 默认监听所有接口，无 TLS，无认证
```

```cpp
// mooncake-store/src/rpc_service.cpp
// ExistKey 方法：直接查询，无任何认证检查
void RpcServiceImpl::ExistKey(const ExistKeyRequest* request,
                               ExistKeyResponse* response) {
    auto status = master_service_->ExistKey(request->key());
    // 任何能连上 50051 端口的客户端都能查询/删除任意 key
}
```

**攻击路径**：任何能访问 Master RPC 端口（默认 50051）的客户端，无需任何凭证，就能执行 Put/Get/Delete/ExistKey 等全部操作。在多租户环境中，一个恶意客户端可以读取其他租户的 KV Cache，甚至删除所有数据。

**加固方案**：

| 方案 | 实现方式 | 防护等级 |
|------|---------|---------|
| 网络隔离 | Master 只监听内网 IP，防火墙限制访问 | 基础 |
| mTLS | coro_rpc 支持 SSL，配置服务端/客户端证书 | 强 |
| Token 认证 | 在 RPC 请求头中携带 token，Master 校验 | 中 |
| 专用 VPC | Master 部署在独立 VPC，只有推理节点可达 | 强 |

##### P2P Handshake：接受任意连接

```cpp
// mooncake-transfer-engine/src/transfer_metadata_plugin.cpp
// startDaemon(): 监听握手请求
bind_address.sin_addr.s_addr = INADDR_ANY;  // 监听所有接口
// 接受任何连接，无 IP 白名单，无认证
```

```cpp
// receivePeerProbe(): 无条件返回成功
// 任何节点发来的探测都回复 "status: success"
// 没有验证对方身份
```

**攻击路径**：恶意节点可以通过握手获取 RDMA 段描述符（包含 RKey），然后直接 RDMA READ 远程内存。

**加固方案**：

| 方案 | 实现方式 | 防护等级 |
|------|---------|---------|
| IP 白名单 | Handshake 只接受已知节点 IP | 基础 |
| 绑定内网 IP | 不用 INADDR_ANY，绑定 RDMA 专用网卡 IP | 基础 |
| Pre-shared Key | 握手时交换共享密钥 | 中 |

##### etcd：无认证的元数据服务

```json
// mooncake-store/conf/master.json
"etcd_endpoints": "http://localhost:2379"  // 明文 HTTP，无认证
```

**攻击路径**：如果 etcd 暴露在网络中（不只是 localhost），任何客户端都能直接读写元数据——包括 RDMA 段描述符中的 RKey。

**加固方案**：

```bash
# 1. etcd 启用 TLS + 认证
etcd --listen-client-urls https://0.0.0.0:2379 \
     --cert-file /etc/etcd/server.crt \
     --key-file /etc/etcd/server.key \
     --auth-token-ttl 300

# 2. etcd 启用 RBAC
etcdctl user add root
etcdctl auth enable
```

---

### 域 2：内存边界——RKey 就是万能钥匙

这是 Mooncake 最核心的安全风险。理解它之前，先搞清楚 RDMA 内存访问的机制：

```
RDMA 远程读写的四个必要参数:

  1. 目标地址 (addr)     → 数据在远程内存中的起始地址
  2. 数据长度 (length)   → 要读写的字节数
  3. 远程密钥 (rkey)     → RDMA 网卡的"门禁卡"
  4. 目标 QP 号          → 远程网卡的队列对编号

  四个参数齐全 → 网卡直接 DMA 读写远程内存
  远程 CPU 完全无感 → 操作系统被绕过
```

##### RKey/LKey 明文存储在元数据中

```cpp
// mooncake-transfer-engine/src/transfer_metadata.cpp
// 段描述符中包含完整的 RDMA 访问信息
json buffer_meta;
buffer_meta["addr"] = buffer_desc.addr;      // 内存地址
buffer_meta["length"] = buffer_desc.length;   // 长度
buffer_meta["rkey"] = buffer_desc.rkey;       // 远程访问密钥
buffer_meta["lkey"] = buffer_desc.lkey;       // 本地访问密钥
// 这些信息被写入 etcd 或通过 HTTP 发布
```

**攻击路径**：读取 etcd → 获取 RKey + 地址 → 直接 RDMA READ 远程内存。这是一个**完整的攻击链**——从元数据窃取到任意内存读取，无需远程 CPU 参与。

```
攻击者视角:

1. 扫描 etcd (http://target:2379)
   → 获取所有段描述符 JSON

2. 解析 JSON，提取:
   addr = 0x7f3a00000000
   rkey = 0x1a2b3c4d
   length = 3355443200

3. 构造 RDMA READ 请求:
   ibv_post_send(qp, {
     opcode: IBV_WR_RDMA_READ,
     remote_addr: 0x7f3a00000000,
     rkey: 0x1a2b3c4d,
     length: 1048576
   })

4. 网卡直接 DMA 读取远程内存
   → 远程 CPU 完全不知道有人正在读它的内存
   → 获取 KV Cache 数据、模型权重、甚至其他进程的内存
```

**加固方案**：

| 方案 | 防护原理 | 防护等级 |
|------|---------|---------|
| **RDMA Fabric 隔离** | 物理隔离 RDMA 网络，不与业务网互通 | 强 |
| **PKey 分区** | IB Partition Key 限制哪些节点可以通信 | 强 |
| **etcd 认证 + TLS** | 防止 RKey 被未授权读取 | 中 |
| **RoCEv2 VNI/VLAN** | 网络层隔离，限制 RDMA 流量可达范围 | 中 |
| **零信任 RDMA** | 每次传输重新协商 RKey，用后即焚 | 最强（需开发） |

> 笔者注：RDMA 的 RKey 泄露等同于内存泄露。这不是 Mooncake 独有的问题——所有 RDMA 系统都面临同样的风险。核心原则是：**RDMA 网络必须是物理隔离的专用网络，绝不能暴露在公网或不信任的网络中。**

##### 共享内存权限过宽

```cpp
// mooncake-transfer-engine/tent/src/transport/rdma/shared_quota.cpp
shm_open(name_.c_str(), O_RDWR | O_CREAT, 0666);  // 任何用户可读写

// mooncake-transfer-engine/tent/src/transport/shm/shm_transport.cpp
shm_open(path.c_str(), O_CREAT | O_RDWR, 0644);    // 任何用户可读
```

**加固方案**：

```bash
# 共享内存权限应限制为 0600（仅所有者可读写）
# 修改代码中的 shm_open 调用:
shm_open(name, O_RDWR | O_CREAT, 0600);  # 仅所有者
```

---

### 域 3：数据边界——明文存储与明文传输

##### SSD 存储：无加密

```cpp
// mooncake-store/src/file_storage.cpp
// KV Cache 数据直接写入文件系统，无加密层
// 任何能访问 SSD 文件系统的人都能读取 KV Cache 数据
```

**加固方案**：

| 方案 | 实现方式 | 性能影响 |
|------|---------|---------|
| 磁盘级加密 | LUKS / dm-crypt 全盘加密 | ~3-5% 吞吐下降 |
| 文件系统加密 | ext4+fscrypt / Btrfs 子卷加密 | ~2-3% |
| 应用层加密 | 写入前加密，读取后解密 | ~5-10%，需开发 |

##### RDMA/TCP 传输：无加密

RDMA 本身不支持加密——这是硬件限制。TCP 传输也没有 TLS。

**加固方案**：

| 传输方式 | 加固方案 | 说明 |
|---------|---------|------|
| RDMA | Fabric 物理隔离 | RDMA 不支持加密，只能靠网络隔离 |
| TCP | 升级为 TLS | asio 支持 SSL stream，需开发 |
| NVMe-oF | 启用 Header/Data Digest | 防篡改（非加密） |

##### 已有的数据完整性保护

Mooncake 在部分路径上有完整性校验：

```bash
# NVMe-oF 数据完整性校验
export MC_NVME_HEADER_DIGEST=1    # 头部校验和
export MC_NVME_DATA_DIGEST=1      # 数据校验和
```

```cpp
// mooncake-store/src/ha/oplog/oplog_applier.cpp
// HA OpLog 校验和验证
OpLogManager::VerifyChecksum(entry);
// 日志: "Possible data corruption or tampering"
```

---

### 域 4：进程边界——最小权限原则

##### Docker 默认 root 运行

```dockerfile
# docker/mooncake.Dockerfile
# 没有 USER 指令 → 默认 root 运行
# 容器逃逸后获得宿主机 root 权限
```

**加固方案**：

```dockerfile
# 添加非 root 用户
RUN useradd -m -s /bin/bash mooncake
USER mooncake
WORKDIR /home/mooncake

# 或在 docker-compose 中指定
services:
  mooncake-master:
    image: mooncake:latest
    user: "1000:1000"
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - IPC_LOCK    # RDMA 内存锁定
    security_opt:
      - no-new-privileges:true
```

##### SPDK 需要 root 权限

```cpp
// mooncake-store/src/spdk/spdk_wrapper.cpp
// SPDK 初始化通常需要 root 或 VFIO 权限
spdk_env_init(...);  // 需要 UIO/VFIO 设备访问权限
```

**加固方案**：

```bash
# 使用 VFIO 替代 UIO，允许非 root 访问
# 1. 绑定设备到 VFIO
echo 0000:3b:00.0 > /sys/bus/pci/drivers/nvme/unbind
echo 0000:3b:00.0 > /sys/bus/pci/drivers/vfio-pci/bind

# 2. 设置 VFIO 组权限
chmod 660 /dev/vfio/12
chown mooncake:mooncake /dev/vfio/12

# 3. 以非 root 用户运行 SPDK
```

---

### 域 5：凭证边界——密钥管理

##### 凭证存储现状

| 凭证 | 存储方式 | 风险 |
|------|---------|------|
| AWS Access Key | `MOONCAKE_AWS_ACCESS_KEY_ID` 环境变量 | `/proc/<pid>/environ` 可读 |
| AWS Secret Key | `MOONCAKE_AWS_SECRET_ACCESS_KEY` 环境变量 | 同上 |
| Redis 密码 | `MC_REDIS_PASSWORD` 环境变量 | 同上 |
| etcd | 无认证 | 任何人可读写 |

**加固方案**：

```bash
# 1. 使用密钥管理服务（推荐）
#    从 Vault/KMS 动态获取凭证，不存环境变量
export MOONCAKE_AWS_ACCESS_KEY_ID=$(vault read -field=access_key secret/mooncake/aws)

# 2. 限制 /proc 权限
sudo sysctl -w kernel.yama.ptrace_scope=2  # 限制 ptrace 访问
mount -o remount,rw /proc
echo 1 > /proc/sys/kernel/kptr_restrict    # 限制内核地址泄露

# 3. etcd 启用认证
etcdctl user add root
etcdctl role add mooncake
etcdctl role grant-permission mooncake readwrite --prefix=true /mooncake/
etcdctl user grant-role root root
etcdctl auth enable
```

> 笔者注：环境变量存储密钥是业界常见做法，不是 Mooncake 独有的问题。但在高安全要求场景下，应使用 Vault/KMS 等专业密钥管理服务。Mooncake系统中是没有硬编码凭证的。

---

### 域 6：编译边界——二进制加固的"安全带"

前五个域关注的是运行时的攻击面，但还有一个容易被忽视的防线：**编译时安全选项**。就像汽车的安全带和气囊——平时感觉不到，撞车时能救命。安全编译选项在缓冲区溢出、代码注入等攻击发生时，提供最后一层兜底。

##### 全局编译选项现状

```cmake
# mooncake-common/common.cmake (第 7-8 行)
# 全局 CXX 编译选项:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -fPIC")
#                                              ↑ 警告    ↑ 警告      ↑ 忽略未用参数   ↑ 位置无关代码
```

全局只设了 `-fPIC`（支持 ASLR 地址随机化），这是共享库的必要选项，不是安全加固。**真正意义上的安全编译选项，全局一个都没有。**

##### Ascend 传输模块：唯一的"系安全带"者

```cmake
# mooncake-transfer-engine/src/transport/ascend_transport/.../CMakeLists.txt (第 12-13 行)
target_compile_options(ascend_transport_mem BEFORE PRIVATE
    "-fstack-protector-strong"    # 栈金丝雀：缓冲区溢出时检测并中止
    "-Wl,-z,relro"               # 只读重定位：GOT 表只读，防止 GOT 覆写攻击
    "-Wl,-z,now"                 # 立即绑定：启动时解析所有符号，配合 relro 实现 Full RELRO
    "-Wl,-z,noexecstack"         # 不可执行栈：栈上的数据不能被当作代码执行（DEP/NX）
)
target_link_options(ascend_transport_mem BEFORE PRIVATE
    "-fstack-protector-strong"
    "-Wl,-z,relro"
    "-Wl,-z,now"
    "-Wl,-z,noexecstack"
)
```

这是**整个项目中唯一**显式启用安全编译选项的目标。其余所有模块——核心传输引擎、Mooncake Store、EP/PG 扩展——都依赖编译器默认行为。

##### 安全编译选项全景对比

| 安全选项 | 防护原理 | 防御的攻击 | Mooncake 全局 | Ascend 模块 | 推荐值 |
|---------|---------|-----------|:------------:|:----------:|:-----:|
| `-fstack-protector-strong` | 函数入口插入金丝雀值，返回前校验 | 栈缓冲区溢出 → ROP 攻击 | ✗ | ✓ | **必须** |
| `-D_FORTIFY_SOURCE=2` | 编译时 + 运行时检查 `memcpy`/`strcpy` 等参数 | 缓冲区溢出 | ✗ | ✗ | **必须** |
| `-fPIE` + `-pie` | 位置无关可执行文件，支持 ASLR | 基于固定地址的代码注入 | ✗ | ✗ | **必须** |
| `-fPIC` | 位置无关代码（共享库） | 支持 ASLR | ✓ | ✓ | 已有 |
| `-z relro` + `-z now` | Full RELRO，GOT 表只读 | GOT 覆写 → 劫持控制流 | ✗ | ✓ | **必须** |
| `-z noexecstack` | 标记栈为不可执行 | 栈上注入 shellcode | ✗ | ✓ | **必须** |
| `-z separate-code` | 代码段与数据段分开，页对齐 | 代码复用攻击 | ✗ | ✗ | 推荐 |

> 注：`-fPIC` 和 `-fPIE` 的区别——前者用于共享库（.so），后者用于可执行文件。Mooncake 全局设了 `-fPIC`，但可执行文件（如 `mooncake_master`）没有 `-fPIE`，意味着主程序本身的 ASLR 支持依赖编译器默认行为。

##### ASAN：CI 有，生产没有

```cmake
# mooncake-common/common.cmake (第 32-39 行)
option(ENABLE_ASAN "enable address sanitizer" OFF)  # 默认关闭

if(ENABLE_ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")     # 内存泄露检测
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")  # 地址越界检测
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()
```

ASAN 在 CI 中启用（`.github/workflows/ci.yml` 第 107 行 `-DENABLE_ASAN=ON`），但生产构建不启用——这是正确做法。ASAN 有 ~2x 性能开销，只适合测试。但 UBSAN（未定义行为检测）和 TSAN（线程竞争检测）在构建系统中完全没有配置，建议至少在 CI 中加入。

##### 缺失的 Sanitizer

| Sanitizer | 检测内容 | CI 可用性 | 性能开销 | 建议 |
|-----------|---------|:--------:|:-------:|------|
| ASAN | 地址越界、Use-After-Free | ✓ 已启用 | ~2x | 保持 CI 启用 |
| LSAN | 内存泄露 | ✓ 已启用 | ~1.1x | 随 ASAN 一起 |
| UBSAN | 未定义行为（整数溢出、空指针解引用等） | ✗ 未配置 | ~1.1x | **CI 应启用** |
| TSAN | 数据竞争 | ✗ 未配置 | ~10x | CI 中可选启用 |
| MSAN | 未初始化内存读取 | ✗ 未配置 | ~3x | CI 中可选启用 |

##### 加固方案

在 `common.cmake` 中添加全局安全编译选项：

```cmake
# mooncake-common/common.cmake — 在 Release 构建中添加安全加固

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  # 栈保护
  add_compile_options(-fstack-protector-strong)
  
  # 源码级缓冲区检查（需要 -O1 及以上优化级别）
  add_compile_options(-D_FORTIFY_SOURCE=2)
  
  # 可执行文件位置无关（ASLR）
  add_compile_options(-fPIE)
  add_link_options(-pie)
  
  # Full RELRO：GOT 表只读
  add_link_options(-Wl,-z,relro -Wl,-z,now)
  
  # 不可执行栈
  add_link_options(-Wl,-z,noexecstack)
  
  # 代码段与数据段分离
  add_link_options(-Wl,-z,separate-code)
endif()

# CI 中额外启用 UBSAN
option(ENABLE_UBSAN "enable undefined behavior sanitizer" OFF)
if(ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
endif()
```

> 笔者注：安全编译选项的性能开销通常 < 1%，但能将缓冲区溢出类攻击从"任意代码执行"降级为"程序崩溃"——从安全角度看，这是巨大的差距。**Ascend 模块已经做了，其他模块没有理由不做。**

还有更多的安全编译选项，比如`-ftrapv`、`-fstack-check`等，这些对性能影响较大，并且在像Ascend加了那些之后，安全影响已不大，可以不加，上面Ascend的案例中安全编译选项都加上就可以了，这部分强烈建议加上。

---

### 已有的安全防护：Mooncake 做对了什么

在指出问题之余，也要肯定 Mooncake 已有的安全防护：

| 防护措施 | 位置 | 防护内容 |
|---------|------|---------|
| 路径遍历防护 | `file_storage.cpp:108-117` | 拒绝 `..` 路径组件，防止目录逃逸 |
| 握手消息长度限制 | `common.h:411-468` | 限制 1MB（可配置），防止缓冲区溢出 |
| TCP 地址校验 | `tcp_transport.cpp:156-163` | 验证远程地址在已注册缓冲区范围内 |
| 租户配额 | `tenant_quota_policy_store.h` | 限制单租户存储消耗，防止资源耗尽 |
| 卸载队列上限 | `master_service.cpp:322-327` | 限制队列大小，防止 DoS |
| OpLog 校验和 | `oplog_applier.cpp:40-47` | 检测数据篡改或损坏 |
| memfd_create CLOEXEC | `shm_helper.cpp:101` | 防止 fd 在 exec 后意外泄露 |
| S3 HTTPS + 校验和 | `s3_helper.cpp` | 可选启用传输加密和数据校验 |
| 无硬编码凭证 | 全局 | 所有密钥通过环境变量/配置文件外部化 |
| Ascend 安全编译 | `ascend_transport/.../CMakeLists.txt` | 唯一启用 `-fstack-protector-strong` + Full RELRO + noexecstack 的模块 |

---

### 生产部署安全检查清单

按优先级从高到低，**逐项检查**：

| 优先级 | 检查项 | 怎么做 | 不做的后果 |
|--------|--------|--------|----------|
| **P0** | RDMA 网络物理隔离 | RDMA 网卡只连接专用交换机，不与业务网互通 | 任何人可通过 RDMA 读取远程内存 |
| **P0** | Master RPC 不暴露公网 | `rpc_address` 绑定内网 IP，防火墙限制 | 任何人可读写/删除全部 KV Cache |
| **P0** | etcd 不暴露公网 | etcd 只监听 localhost 或内网 IP | RKey 泄露 → 远程内存可被任意读取 |
| **P1** | etcd 启用认证 + TLS | 配置 etcd RBAC + HTTPS | 元数据可被篡改/窃取 |
| **P1** | SSD 全盘加密 | LUKS/dm-crypt | 硬盘被物理移除后数据泄露 |
| **P1** | Docker 非 root 运行 | Dockerfile 添加 USER 指令 | 容器逃逸 = 宿主机 root |
| **P2** | 共享内存权限收紧 | shm_open 权限改为 0600 | 本地用户可读取 RDMA 配额/传输数据 |
| **P2** | 凭证使用密钥管理服务 | Vault/KMS 替代环境变量 | `/proc/<pid>/environ` 泄露密钥 |
| **P2** | NVMe-oF 启用 Digest | `MC_NVME_HEADER_DIGEST=1` | 数据在传输中被篡改无法检测 |
| **P2** | 安全编译选项 | 添加 `-fstack-protector-strong` `-D_FORTIFY_SOURCE=2` `-fPIE` RELRO | 缓冲区溢出 → 任意代码执行 |
| **P3** | RPC 连接超时 | `rpc_conn_timeout_seconds > 0` | 连接耗尽型 DoS |
| **P3** | 日志脱敏 | 错误日志不输出 JSON 原文和凭证状态 | 日志文件泄露敏感信息 |

---

### 安全架构设计原则

| 原则 | 含义 | Mooncake 中的体现 |
|------|------|------------------|
| **纵深防御** | 不依赖单一防线，多层保护 | 网络隔离 + etcd 认证 + RKey 保护 + 磁盘加密 + 安全编译 |
| **最小权限** | 每个组件只拥有必要的权限 | Docker 非 root、共享内存 0600、SPDK 用 VFIO、栈不可执行 |
| **零信任网络** | 不因"在内网"就信任 | RDMA Fabric 隔离、P2P 握手需认证、etcd TLS |
| **默认安全** | 默认配置应该是安全的 | 当前不符合——默认配置全部不安全，需加固 |

> 笔者注：Mooncake 的安全现状是"信任内网"模型——假设所有能访问 RDMA 网络和 Master 端口的节点都是可信的。这在私有集群中可以接受，但在多租户云环境或跨组织协作中不够。**如果你要把 Mooncake 部署在非完全信任的环境中，P0 项必须全部完成。**

---

### 总结与行动指南

| 安全域 | 核心风险 | 一句话建议 |
|--------|---------|----------|
| 网络边界 | Master/Handshake 无认证 | 网络隔离是第一道也是最重要的防线 |
| 内存边界 | RKey 明文存储 = 万能钥匙 | 保护 etcd = 保护 RKey = 保护内存 |
| 数据边界 | SSD 和传输均无加密 | RDMA 靠隔离，SSD 靠 LUKS |
| 进程边界 | 默认 root 运行 | Docker 非 root + 最小 capability |
| 凭证边界 | 密钥在环境变量中 | 高安全场景用 Vault/KMS |
| 编译边界 | 全局无安全编译选项 | Ascend 已做，其他模块应跟进 |

**建议**: Mooncake 部署的第一天，先做三件事——**RDMA 网络物理隔离、Master RPC 绑定内网 IP、etcd 启用认证**。这三步做完，堵住了 90% 的攻击面。

**延伸阅读**：
- RDMA 安全白皮书：https://www.rdmaconsortium.org/
- etcd 安全配置：https://etcd.io/docs/latest/op-guide/security/
- 容器安全最佳实践：https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
