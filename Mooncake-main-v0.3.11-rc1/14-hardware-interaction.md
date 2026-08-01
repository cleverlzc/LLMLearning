# Mooncake 硬件交互：KV Cache 如何在 GPU、DRAM 与 SSD 之间"搬家"

> **系列**: Mooncake 技术博客系列 | **类型**: 核心技术详解篇
>
> 前文我们讲了 Mooncake 分层存储的"热货架与冷库"——DRAM 装不下的 KV Cache 搬进 SSD，需要时再搬回来。但"搬家"这件事，说起来容易做起来难：GPU 上的数据怎么搬到 CPU 内存？CPU 内存怎么写到 SSD？写 SSD 走什么 I/O 路径？能不能绕过操作系统内核直接跟硬件对话？

前面讲了那么多的软件系统层面的内容，是不是对于AI服务器还是没有实感？本文我们从物理硬件的视角，追踪一条 KV Cache 从 GPU 显存到 DRAM 内存到 SSD 硬盘的完整旅程。

---

### 引言：存储器的"不可能三角"

KV Cache 在 AI 服务器里要存，但存哪里？这背后是一个根本性的物理约束——**存储器不可能三角**：快、大、便宜，三者不可兼得。

```
                    快 (低延迟 · 高带宽)
                       ╱ ╲
                      ╱   ╲
                     ╱     ╲
                    ╱ 不可能 ╲
                   ╱  三角！  ╲
                  ╱           ╲
                 ╱             ╲
           大 (高容量) ────── 便宜 (低成本)
```

现实中的存储器，都落在这个三角的某条边上——**快的一定小，大的一定慢，便宜的一定慢**：

```
  延迟        带宽          容量/机         成本/GB       技术
  ──────     ──────       ────────       ────────     ──────
  ~100ns     ~2 TB/s      80-192 GB      ~$20         HBM3e (GPU)
  ~100ns     ~50 GB/s     256 GB-2 TB    ~$5          DDR5 (DRAM)
  ~10μs      ~7 GB/s      3.84-15 TB     ~$0.3        NVMe (SSD)
  ~20μs      ~6 GB/s      集群级          ~$0.1        NVMe-oF (远程SSD)
  
  ──→ 每下一层: 延迟 ×100   带宽 ÷10   容量 ×10   成本 ÷10 ──→
```

用物流中心来比喻——**越贵的仓库越小但越快，越便宜的仓库越大但越慢**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─────────────────────┐    容量: 80-192 GB/卡                       │
│  │                     │    延迟: ~100ns  带宽: ~2 TB/s              │
│  │   加工车间           │    成本: ~$20/GB                            │
│  │   (GPU 显存 / HBM)   │                                              │
│  │   最快·最小·最贵      │    KV Cache 在此产生                       │
│  │                     │    放不下 → 必须搬家                         │
│  └──────────┬──────────┘                                              │
│             │                                                         │
│             │  叉车 (cudaMemcpy D2H)                                  │
│             │  ~25 GB/s · PCIe 4.0 专用通道                           │
│             │  走 pinned memory 快车道                                │
│             │                                                         │
│             ▼                                                         │
│  ┌──────────────────────────────────────────┐   容量: 256 GB-2 TB/机  │
│  │                                          │   延迟: ~100ns           │
│  │            主仓库                          │   带宽: ~50 GB/s        │
│  │            (DRAM)                         │   成本: ~$5/GB          │
│  │                                          │                         │
│  │   比 HBM 大 10x · 比 HBM 便宜 4x          │   大页加速              │
│  │   比 SSD 快 100x · 比 SSD 贵 15x          │   RDMA 注册可远程访问   │
│  │                                          │                         │
│  │   ┌────────────────────────────────┐     │                         │
│  │   │ Pinned 卸货台 (cudaMallocHost) │     │   叉车直达·DMA 无中转   │
│  │   └────────────────────────────────┘     │                         │
│  │                                          │                         │
│  └──────┬───────────────────────┬───────────┘                         │
│         │                       │                                     │
│         │ 传送带                 │ RDMA 直通                           │
│         │ io_uring + O_DIRECT   │ Transfer Engine                     │
│         │ ~7 GB/s               │ ~50 GB/s                            │
│         │ 绕过页缓存直写SSD      │ 远程 DRAM 直达                      │
│         │                       │                                     │
│         ▼                       ▼                                     │
│  ┌──────────────────────────────────────────────────┐                │
│  │                                                    │                │
│  │              堆场 (本地 NVMe SSD)                    │                │
│  │                                                    │                │
│  │   容量: 3.84-15 TB     延迟: ~10μs                 │                │
│  │   带宽: ~7 GB/s        成本: ~$0.3/GB              │                │
│  │                                                    │                │
│  │   比 DRAM 大 10x · 比 DRAM 便宜 15x                │                │
│  │   比 DRAM 慢 100x · 断电不丢数据                    │                │
│  │                                                    │                │
│  │   桶文件存储 · O_DIRECT 零拷贝 · LRU 驱逐           │                │
│  │                                                    │                │
│  └────────────────────────┬───────────────────────────┘                │
│                           │                                           │
│                           │ 直升机 (SPDK NVMe-oF)                     │
│                           │ ~6 GB/s · RDMA 直连                       │
│                           │ 绕过远程主机 CPU + 操作系统                 │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────┐                │
│  │                                                    │                │
│  │              远程仓库 (NVMe-oF SSD)                 │                │
│  │                                                    │                │
│  │   容量: 集群级          延迟: ~20μs                 │                │
│  │   带宽: ~6 GB/s         成本: ~$0.1/GB              │                │
│  │                                                    │                │
│  │   容量近乎无限 · 但延迟最高                          │                │
│  │   SPDK 用户态驱动 · 无系统调用 · 无锁               │                │
│  │                                                    │                │
│  └────────────────────────────────────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

每一层的核心矛盾：

| 层 | 核心矛盾 | Mooncake 的解法 |
|---|---------|---------------|
| HBM | 极快但极小——80 GB 只够 30 个请求 | 放不下的 KV Cache 搬到 DRAM |
| DRAM | 够快但不够大——2 TB 也装不下全部 | 装不下的卸载到 SSD，需要时提升回来 |
| SSD | 够大但不够快——比 DRAM 慢 100x | io_uring + O_DIRECT 压榨最大吞吐 |
| 远程 SSD | 容量无限但延迟最高 | SPDK 绕过远程 CPU，减少中间环节 |

**分层存储的本质，就是在这个不可能三角上做"用延迟换容量、用容量换吞吐"的经济学**——把热数据放在快但贵的地方，冷数据放在大但便宜的地方，中间用高效的搬家工具连接。

搬家的效率，取决于你用什么交通工具：

| 交通工具 | 对应技术 | 速度 | 路线 |
|---------|---------|------|------|
| 叉车 | cudaMemcpy (D2H/H2D) | ~25 GB/s (PCIe 4.0) | 车间 ↔ 主仓库卸货台 |
| 传送带 | io_uring + O_DIRECT | ~7 GB/s | 主仓库 ↔ 堆场 |
| RDMA 直通 | Transfer Engine | ~50 GB/s | 主仓库 ↔ 远程主仓库 |
| 直升机 | SPDK NVMe-oF | ~6 GB/s | 本地 → 远程堆场直达 |

选错交通工具，搬家时间可能比加工时间还长。在 MoE 推理中，一条 KV Cache 动辄数十 MB，如果 D2H 拷贝走普通内存通道（pageable memory），带宽只有 pinned memory 的 1/10 到 1/100——，比如如果Pageable D2H 大约是 ~1 GB/s，而 Pinned 大约是 ~25 GB/s，这大约是 25 倍的提升，**叉车直接变成了手推车**。

今天这篇文章，我们深入 Mooncake 与物理硬件的交互层，看看每一步"搬家"到底是怎么完成的。

---

### 内存层级全景：从 GPU 到 SSD 的物理距离

在深入每个环节之前，先看一眼全貌——注意延迟的量级跳跃：

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 显存 (VRAM/HBM)                       │
│  容量: 80-192 GB/卡   延迟: ~100ns   带宽: ~2 TB/s          │
│  技术: HBM3e        访问方式: CUDA 核心                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ PCIe 4.0 x16 (~25 GB/s) / PCIe 5.0 x16 (~50 GB/s)
                            │ cudaMemcpy D2H
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Pinned Host Memory (页锁定 DRAM)               │
│  容量: 按需分配       延迟: ~100ns   带宽: ~25 GB/s         │
│  技术: cudaMallocHost  访问方式: DMA 直传                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ 内存拷贝
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DRAM (普通内存)                            │
│  容量: 256 GB-2 TB/机  延迟: ~100ns   带宽: ~50 GB/s        │
│  技术: DDR5 ECC       访问方式: mmap/memfd_create/hugepage  │
│  注册: Transfer Engine registerLocalMemory()                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ POSIX I/O 或 io_uring
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    本地 NVMe SSD                             │
│  容量: 3.84-15.36 TB  延迟: ~10μs    带宽: ~7 GB/s          │
│  技术: NAND 闪存      访问方式: pread/pwrite 或 io_uring    │
└───────────────────────────┬─────────────────────────────────┘
                            │ RDMA (NVMe-oF)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    远程 NVMe SSD (NVMe-oF)                   │
│  容量: 集群级        延迟: ~20μs    带宽: ~6 GB/s           │
│  技术: SPDK          访问方式: NVMe over Fabrics RDMA       │
└─────────────────────────────────────────────────────────────┘
```

从 DRAM 的 100ns 到 SSD 的 10μs，差了 100 倍；从 SSD 的 10μs 到分布式存储的 ms 级，又差了 100 倍。

这就是分层存储必须"精打细算"的原因：每往下一层，访问成本就跳一个数量级。

> 笔者注：Pinned Memory 的 D2H 带宽取决于 PCIe 代际：PCIe 3.0 x16 实测 ~12 GB/s（V100），PCIe 4.0 x16 实测 ~25 GB/s（A100），PCIe 5.0 x16 实测 ~50 GB/s（H100）。本文以 A100（PCIe 4.0）为基准。SXM 版本 GPU 通过 NVLink 连接，GPU 间带宽可达 ~300 GB/s，但 D2H 仍走 PCIe。

---

### DRAM 管理：主仓库怎么建

Mooncake 的 DRAM 管理涉及三个问题：**怎么分配、怎么共享、怎么让远程访问**。

##### 内存分配：mmap + memfd_create

Mooncake 使用 `ShmHelper` 在 DRAM 中分配大块共享内存：

```cpp
// mooncake-store/src/shm_helper.cpp
void* ShmHelper::allocate(size_t size) {
    // 1. 创建匿名共享内存文件
    int fd = memfd_create(MOONCAKE_SHM_NAME, flags);

    // 2. 设置大小
    ftruncate(fd, size);

    // 3. 映射到进程地址空间
    void* base_addr = mmap(nullptr, size,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, fd, 0);
    return base_addr;
}
```

为什么用 `memfd_create` + `mmap`，而不是简单的 `malloc`？

| 方式 | 跨进程共享 | 大页支持 | 固定地址 | RDMA 注册 |
|------|----------|---------|---------|----------|
| `malloc` | 不支持 | 不支持 | 不支持 | 困难 |
| `mmap` + `memfd_create` | 支持（fd 传递） | 支持 | 支持 | 支持 |

`memfd_create` 创建的是一个**匿名文件**——没有磁盘上的对应文件，但拥有文件描述符（fd）。这个 fd 可以在进程之间传递（通过 Unix Socket），接收方用 `mmap` 映射同一块物理内存，实现零拷贝共享。

##### 大页：减少地址翻译开销

普通内存页大小 4KB，1GB 内存需要 262144 个页表项。大页（HugePage）使用 2MB 或 1GB 的页大小，1GB 内存只需要 512 个页表项——地址翻译快了 500 倍。

```cpp
// ShmHelper 中的大页支持
if (use_hugepage) {
    flags |= MAP_HUGETLB;  // 尝试 2MB 大页
    // 失败则回退到普通页
}
```

大页在 RDMA 场景下尤其重要。RDMA 注册内存时，需要把虚拟地址映射到物理地址——页表越少，注册越快，MR（Memory Region）的效率越高。如果用普通 4KB 页，注册 1GB 内存可能需要数秒；用 2MB 大页，只需要几十毫秒。

当前业界实践，KV Cache动辄十几M起步，编码等Agent智能体长程任务甚至GB，均已采用2MB大页，**已成为当前业界KV Cache系统标配**。个别地方，可能会考虑采用1GB，或者将内存切分为2MB大页和1GB大页分层混用。

总之，还是根据实际场景、实际系统性能目标，定向调整，达到最优。

##### 共享内存：跨进程零拷贝

Mooncake Store 的 Client 和 Transfer Engine 可能是不同进程。它们如何共享同一块 DRAM？

```
进程 A (Client):
  fd = memfd_create("mooncake_shm", ...)
  addr_A = mmap(fd, size)  ← 映射到进程 A 的地址空间

进程 B (Transfer Engine):
  recv_fd = recvmsg(unix_socket, fd)  ← 通过 Unix Socket 接收 fd
  addr_B = mmap(recv_fd, size)  ← 映射到进程 B 的地址空间

  addr_A 和 addr_B 指向同一块物理内存！
  进程 A 写入的数据，进程 B 直接可见，无需拷贝。
```

##### Transfer Engine 注册：让 DRAM 可以被远程 RDMA 访问

分配了 DRAM 还不够——要让它能被远程节点通过 RDMA 访问，必须注册到 Transfer Engine：

```cpp
// mooncake-transfer-engine/example/memory_pool.cpp
void* addr = allocateMemoryPool(dram_buffer_size, i);
engine->registerLocalMemory(addr, dram_buffer_size, "cpu:" + std::to_string(i));
engine->installTransport("rdma", args);
```

`registerLocalMemory` 做了两件事：

```
1. 调用 ibv_reg_mr() 将内存注册到 RDMA 网卡
   → 网卡获得这块内存的虚拟地址→物理地址映射
   → 网卡可以直接 DMA 读写这块内存，不经过 CPU

2. 将内存信息发布到元数据服务
   → 远程节点查询元数据，获取这块内存的地址和访问密钥
   → 远程节点可以直接 RDMA Read/Write 这块内存
```

未注册的内存，RDMA 网卡无法访问——就像没有门牌号的房间，快递员找不到。注册之后，网卡拿到"门牌号"（LKey/RKey），后续所有 RDMA 操作都用这个密钥直接访问。

---

### GPU ↔ DRAM：叉车卸货——车间到主仓库的专用通道

KV Cache 最初在 GPU 显存中产生（Prefill 阶段）。要把它存到 DRAM 或 SSD，第一步是 D2H（Device to Host）拷贝。

##### 为什么需要 Pinned Memory？

普通内存（pageable memory）和 GPU 之间的数据传输，操作系统需要先把数据从可换页内存拷贝到页锁定内存，再通过 DMA 传到 GPU——多了一次中间拷贝。而页锁定内存（pinned memory）可以直接被 GPU 的 DMA 引擎访问，省掉中间环节。

```
Pageable Memory 的 D2H 路径:
  GPU ──DMA──→ Pinned Staging Buffer ──拷贝──→ Pageable Buffer
                 (操作系统自动分配)              (用户缓冲区)
                 ↑ 多了一次拷贝！

Pinned Memory 的 D2H 路径:
  GPU ──DMA──→ Pinned Buffer
                 (用户直接分配)
                 ↑ 一步到位！
```

| 特性 | Pageable Memory | Pinned Memory |
|------|----------------|---------------|
| 分配方式 | `malloc` / `new` | `cudaMallocHost` |
| D2H 带宽 | ~1 GB/s | ~25 GB/s |
| DMA 直传 | 不支持（需中间拷贝） | 支持 |
| 内存占用 | 可被 OS 换出到 SSD | 锁定在物理内存，不可换出 |
| 分配速度 | 快 | 慢（需要锁定物理页） |

> 笔者注：Pinned Memory 在注册大块内存的时候，比如1TB或者2TB，可能需要比较长的时间，可能是3-5分钟级，Mooncake采用的做法是，多线程分批去注册，比如8线程均分，提升注册速度。
> 为什么注册时间也要优化？因为系统在最初的初始化时，这个时候没有生产业务，注册速度可能影响还可控，但在实际生产业务系统中，正在提供服务，节点重启，再正式重新接管任务，每一分每一秒都非常宝贵。

##### PinnedBufferPool：叉车车队

Pinned memory 分配慢（需要锁定物理页），但用起来快（DMA 直传）。Mooncake 的解法：**池化复用**——分配一次，反复使用。

```cpp
// mooncake-store/include/pinned_buffer_pool.h
class PinnedBufferPool {
    // 池化复用，最大 32 个缓冲区
    Buffer Acquire(size_t size) {
        // 优先从池中获取
        // 池空则新分配: cudaMallocHost 或等价物
    }

    void Release(Buffer&& buf) {
        // 归还到池中，下次复用
    }
};
```

分配的具体实现，根据硬件平台自动选择：

```cpp
// mooncake-store/src/device/cuda_like_accelerator_device.cpp
PinnedHostBuffer AllocatePinnedHost(size_t size) const override {
    void* addr = nullptr;
    if (cudaMallocHost(&addr, size) != cudaSuccess) {
        // 分配失败，回退到普通内存
        return PinnedHostBuffer();
    }
    return PinnedHostBuffer(addr, size, FreeCudaLikePinnedHostBuffer);
}
```

PinnedBufferPool 的三层保护：
1. **池化复用**：Acquire 优先从池中取，避免重复分配
2. **平台适配**：自动检测 NVIDIA/AMD/昇腾，调用对应的 pinned 分配 API
3. **优雅降级**：pinned 分配失败 → 回退到 `new char[]`，带宽下降但不会崩溃

##### D2H 拷贝：Offload 中的关键一步

在 Offload（DRAM → SSD）流程中，如果数据还在 GPU 上，需要先做 D2H 暂存：

```cpp
// mooncake-store/src/file_storage.cpp
// D2H staging: replace device slices with host memory slices
for (auto& [obj_key, slices] : batch_object) {
    std::vector<Slice> host_slices;
    for (const auto& slice : slices) {
        auto* device = runtime_accelerator.FindDeviceForPointer(slice.ptr, &info);
        if (device) {
            // GPU 数据: 通过 PinnedBufferPool 做 D2H 拷贝
            auto buf = pinned_buffer_pool_->Acquire(slice.size);
            device->Copy(buf.data, slice.ptr, slice.size,
                         device::CopyDirection::kDeviceToHost);
            host_slices.emplace_back(Slice{buf.data, slice.size});
            staging_bufs.push_back(std::move(buf));
        } else {
            // DRAM 数据: 直接使用，无需拷贝
            host_slices.push_back(slice);
        }
    }
    host_batch_object[obj_key] = std::move(host_slices);
}
```

这段代码的关键设计：**自动检测数据位置**。`FindDeviceForPointer` 判断指针属于 GPU 还是 DRAM——如果是 GPU 指针，走 D2H 拷贝；如果是 DRAM 指针，直接使用。上层逻辑不需要关心数据在哪里。

---

### DRAM ↔ SSD：传送带搬运——主仓库到堆场的批量运输

数据到了 DRAM，下一步是写入 SSD。Mooncake 提供两种 I/O 路径，就像两种搬运方式——人工搬运 vs 传送带。

##### POSIX I/O：人工搬运

```
应用程序:  pwrite(fd, buffer, size, offset)
    │
    ├── 用户态缓冲区 → 内核页缓存 (一次拷贝)
    │
    └── 内核页缓存 → SSD 硬盘 (DMA)

    特点: 两次拷贝，每次系统调用都陷入内核态
```

POSIX I/O 使用 `preadv`/`pwritev`（scatter/gather I/O），一次系统调用可以读写多个不连续的缓冲区：

```cpp
// mooncake-store/src/posix_file.cpp
tl::expected<size_t, ErrorCode> PosixFile::vector_write(
    const iovec* iov, int iovcnt, off_t offset) {
    return pwritev(fd_, iov, iovcnt, offset);
}
```

POSIX I/O 的好处是通用、兼容性好，坏处是每次 I/O 都要陷入内核态，数据要经过页缓存中转。对于大块顺序写入（KV Cache 的典型场景），页缓存反而成了累赘——数据马上就要写到 SSD，根本不需要缓存。

##### io_uring + O_DIRECT：传送带系统

```
应用程序:  提交 I/O 请求到共享环形缓冲区
    │
    ├── 用户态直接写入 SSD (O_DIRECT 跳过页缓存)
    │
    └── 内核批量完成，用户态批量收割结果

    特点: 零次额外拷贝，批量提交/完成，减少系统调用
```

```cpp
// mooncake-store/include/file_interface.h
class UringFile : public StorageFile {
    // 进程级共享 ring
    static bool register_global_buffer(void* buffer, size_t length);

    // O_DIRECT 对齐读写
    tl::expected<size_t, ErrorCode> read_aligned(
        void* buffer, size_t length, off_t offset = 0);
    tl::expected<size_t, ErrorCode> write_aligned(
        const void* buffer, size_t length, off_t offset = 0);

    // 批量读取（一次提交最多 32 个独立读请求）
    tl::expected<size_t, ErrorCode> batch_read(
        const ReadDesc* descs, int cnt);

    // 写持久化
    tl::expected<void, ErrorCode> datasync();
};
```

两种路径的关键对比：

| 特性 | POSIX (preadv/pwritev) | io_uring + O_DIRECT |
|------|----------------------|---------------------|
| 系统调用 | 每次 I/O 一次 | 批量提交，一次系统调用处理多个 I/O |
| 数据拷贝 | 2 次（用户态→页缓存→SSD） | 1 次（用户态→SSD 直传） |
| 缓冲区对齐 | 无要求 | 4096 字节对齐 |
| 批量操作 | scatter/gather | 批量提交 + 批量收割 |
| 适用场景 | 通用、兼容性好 | 高吞吐、低延迟 |

##### O_DIRECT 的对齐要求

O_DIRECT 要求缓冲区 4096 字节对齐。Mooncake 使用 `AlignedClientBufferAllocator` 确保客户端缓冲区满足对齐要求：

```cpp
// mooncake-store/include/aligned_client_buffer.h
class AlignedClientBufferAllocator : public ClientBufferAllocator {
    // 在 OffsetBufferAllocator 基础上
    // 每次分配都 4096 对齐
};
```

如果缓冲区不对齐就开 O_DIRECT，I/O 会直接报错。这是使用 io_uring 的前提条件。

##### BucketStorageBackend 的写入路径

生产推荐的 Bucket 后端，写入时把多个 key 的数据合并为一个桶文件：

```
BatchOffload() 写入流程:
  1. AllocateOffloadingBuckets() → 将多个 key 分组到桶中
     ├── 每桶最大 256 MB
     └── 每桶最多 500 个 key

  2. WriteBucket() → 写入桶文件
     ├── io_uring 模式: write_aligned() + datasync()
     └── POSIX 模式: vector_write() (pwritev)

  3. 桶文件布局:
     ┌─────────────────────────────┐
     │  data 区域                    │
     │  [key1_data][key2_data]...   │
     ├─────────────────────────────┤
     │  metadata 区域               │
     │  meta_size | data_size |     │
     │  keys[] | metadatas[]        │
     └─────────────────────────────┘
```

读取时，`BatchLoad` 同样分两种路径，io_uring 模式下有一个精巧的零拷贝设计：

```
BatchLoad() 读取流程:
  ├── io_uring 模式: read_aligned() 直读到客户端缓冲区
  │   → 零拷贝: 缓冲区指针调整 offset_in_buffer 跳过对齐填充
  │   → 避免额外 memcpy
  │
  └── POSIX 模式: vector_read() (preadv)
      → 标准 scatter/gather 读取
```

O_DIRECT 要求缓冲区 4096 对齐，但客户端实际需要的数据可能从非对齐偏移开始。Mooncake 的做法是：分配对齐的缓冲区，读取后调整指针跳过对齐填充，直接返回有效数据的起始地址——无需额外拷贝。

---

### NVMe-oF：直升机空投——直达远程堆场

当本地 SSD 不够用时，Mooncake 支持通过 NVMe over Fabrics (NVMe-oF) 访问远程 SSD——绕过远程主机的操作系统，直接通过 RDMA 访问远程 SSD 硬件。

##### 传统路径 vs SPDK 路径

传统 NVMe 驱动路径：

```
应用程序 → 系统调用 → 内核 NVMe 驱动 → SSD
                              │
                    中断处理、上下文切换、锁竞争
```

SPDK (Storage Performance Development Kit) 的路径：

```
应用程序 → SPDK 用户态驱动 → SSD (轮询模式)
           │
     无系统调用、无中断、无锁
```

SPDK 把 NVMe 驱动从内核搬到了用户态，用轮询代替中断，用无锁队列代替锁——代价是占用一个 CPU 核心专门轮询，但换来的是极低的 I/O 延迟。

```cpp
// mooncake-store/include/spdk/spdk_wrapper.h
class SpdkWrapper {
    bool InitializeEnv();
    void* Alloc(size_t size, size_t align, int socket_id = -1);
    nof_seg_handle* OpenNofSegment(const std::string& tr_str);
    int SubmitRequest(const nof_seg_handle* seg_handle,
                      void* ptr, uint64_t lba,
                      uint32_t lba_count, int op,
                      spdk_nvme_cmd_cb cb_fn, void* cb_ctx);
};
```

SPDK 的关键数据结构直接映射到 NVMe 硬件：

```cpp
struct nof_seg_handle {
    struct spdk_nvme_qpair *qpair;  // NVMe 提交队列/完成队列对
    struct spdk_nvme_ns *ns;        // NVMe 命名空间（逻辑卷）
};
```

##### NVMe-oF 通信：RDMA 传输

NVMe-oF 默认使用 RDMA 作为传输层：

```cpp
// mooncake-store/src/ssd_register_client.cpp
const char* trtype_env = std::getenv("MC_NOF_TRTYPE");
std::string trtype = trtype_env ? trtype_env : "RDMA";  // 默认 RDMA
```

```
本地节点                           远程节点
┌──────────┐    RDMA 网络    ┌──────────────┐
│ SPDK     │ ──────────────→ │ NVMe SSD     │
│ 用户态   │  NVMe-oF 协议   │ (通过 RDMA   │
│ 驱动     │ ←────────────── │  直接访问)   │
└──────────┘    响应数据     └──────────────┘

绕过了远程节点的 CPU 和操作系统！
```

传统方式访问远程 SSD：本地 → 网络 → 远程内核 → 远程 SSD → 远程内核 → 网络 → 本地。NVMe-oF：本地 SPDK → RDMA → 远程 SSD。中间环节从 6 步缩减到 2 步。

##### SPDK 内存分配：DMA 兼容

SPDK 需要 DMA 兼容的内存——物理连续、地址对齐。Mooncake 通过 `SpdkWrapper::Alloc` 分配：

```cpp
// mooncake-store/src/memory_alloc.cpp
void* hugepage_memory_alloc(size_t size) {
    return mooncake::SpdkWrapper::GetInstance().Alloc(size, 0x1000, -1);
    // 0x1000 = 4096 字节对齐
}
```

SPDK 的内存分配本质上也是大页——DMA 需要物理连续的内存，大页天然满足这个要求。普通 4KB 页可能分散在物理内存的任何位置，而 2MB 大页保证连续。这就是为什么 SPDK 和 RDMA 都"偏爱"大页。

---

### 完整数据路径：一条 KV Cache 的硬件旅程

让我们追踪一条 KV Cache 从 GPU 产生到存入 SSD 的完整路径：

```
阶段 1: GPU Prefill 产生 KV Cache
  ┌──────────────────┐
  │  GPU 显存 (VRAM)  │  KV Cache 在这里产生
  │  [KV Cache 数据]  │  大小: ~50 MB
  └────────┬─────────┘
           │
           │ cudaMemcpy D2H (通过 PinnedBufferPool)
           │ 带宽: ~25 GB/s, 耗时: ~2 ms
           ▼
阶段 2: D2H 暂存到 Pinned Memory
  ┌──────────────────┐
  │ Pinned Host Memory│  页锁定内存，DMA 直传
  │ [暂存缓冲区]      │  来自 PinnedBufferPool 池化复用
  └────────┬─────────┘
           │
           │ 内存拷贝 (或直接使用，如果已在 DRAM)
           │
           ▼
阶段 3: 存入 DRAM 段 (MEMORY 副本)
  ┌──────────────────┐
  │     DRAM 段       │  mmap/memfd_create 分配
  │ [KV Cache 副本]   │  已注册到 Transfer Engine (RDMA 可访问)
  └────────┬─────────┘
           │
           │ Offload: storage_backend_->BatchOffload()
           │ io_uring O_DIRECT 或 POSIX pwritev
           │ 带宽: ~7 GB/s, 耗时: ~7 ms
           ▼
阶段 4: 写入本地 SSD (LOCAL_DISK 副本)
  ┌──────────────────┐
  │   本地 NVMe SSD   │  Bucket 文件 (多 key 合并)
  │ [bucket_xxx.data] │  data + metadata 双文件
  └──────────────────┘
```

反过来，当需要从 SSD 读回数据时（Promotion）：

```
阶段 4 → 阶段 3: BatchLoad() 从 SSD 读取到对齐缓冲区
  ├── io_uring: read_aligned() 零拷贝直读
  └── POSIX: preadv() 标准读取

阶段 3 → DRAM: PromotionWrite() 通过 Transfer Engine 写入 DRAM 副本
  └── RDMA Write: 从对齐缓冲区直接写入远程 DRAM
```

整条路径中，最耗时的环节是 D2H 拷贝（~2 ms）和 SSD 写入（~7 ms）。对于 50 MB 的 KV Cache，总耗时约 9 ms。相比之下，DRAM 内部的内存拷贝只需 ~1 μs，几乎可以忽略。这就是为什么 Mooncake 要尽量让数据留在 DRAM——每往下一层，搬家成本就高一个数量级。

---

### 多硬件平台适配：一套代码，多种硬件

Mooncake 的硬件交互层通过 `AcceleratorDevice` 抽象适配多种硬件平台：

```cpp
// mooncake-store/include/device/
class AcceleratorDevice {
    virtual PinnedHostBuffer AllocatePinnedHost(size_t size) = 0;
    virtual bool Copy(void* dst, const void* src, size_t size,
                      CopyDirection direction) = 0;
};
```

`GetAcceleratorRegistry().RuntimeAccelerators()` 自动检测当前系统中可用的加速器——不需要在代码中硬编码硬件类型。同一份 Mooncake Store 代码，在 NVIDIA 集群上用 cudaMallocHost，在昇腾集群上用 Ascend VMM，无需修改。

已适配的硬件平台：

| 平台 | GPU/NPU | Pinned Memory | 共享内存 |
|------|---------|--------------|---------|
| NVIDIA | H100/A100 | cudaMallocHost | memfd_create + mmap |
| AMD | MI300 | cudaMallocHost (ROCm) | memfd_create + mmap |
| 华为昇腾 | Ascend 910 | Ascend VMM | ascend_allocate_vmm_memory_direct |
| 摩尔线程 | MUSA GPU | musaMallocHost | memfd_create + mmap |

昇腾 NPU 的特殊之处在于共享内存——它不用 `memfd_create` + `mmap`，而是使用 `ascend_allocate_vmm_memory_direct()` 直接分配 fabric memory，这是昇腾硬件特有的跨进程共享机制。

---

### 设计哲学：硬件交互的三大原则

| 原则 | 含义 | 典型体现 |
|------|------|---------|
| **能直传就不拷贝** | 减少数据在内存中的搬运次数 | Pinned Memory DMA 直传、io_uring O_DIRECT 零拷贝、RDMA 远程直读 |
| **能复用就不新建** | 避免重复分配/释放的开销 | PinnedBufferPool 池化复用、SharedUringRing 进程级共享 |
| **能降级就不崩溃** | 高性能方案失败时自动回退 | Pinned 分配失败→普通内存、io_uring 不可用→POSIX、大页不足→普通页 |

这三条原则回答了硬件交互最核心的三个问题：

1. **数据搬几次？** —— 能直传就不拷贝。Pinned Memory 让 GPU↔DRAM 少一次拷贝，O_DIRECT 让 DRAM↔SSD 少一次拷贝，RDMA 让远程访问零拷贝。
2. **分配快不快？** —— 能复用就不新建。PinnedBufferPool 池化复用避免反复 cudaMallocHost，SharedUringRing 进程级共享避免每个文件单独建 ring。
3. **挂了怎么办？** —— 能降级就不崩溃。Pinned 分配失败回退普通内存，io_uring 不可用回退 POSIX，大页不足回退普通页。每一层都有安全网。

---

### 总结与行动指南

| 核心概念 | 一句话总结 |
|---------|----------|
| memfd_create + mmap | DRAM 分配方式，支持跨进程零拷贝共享和大页 |
| PinnedBufferPool | GPU D2H 的叉车车队，页锁定内存池化复用，10x-100x 带宽提升 |
| io_uring + O_DIRECT | SSD 读写的高速传送带，零拷贝直传，批量提交/完成 |
| SPDK NVMe-oF | 远程 SSD 的直达专线，绕过内核，RDMA 直连 |
| registerLocalMemory | 让 DRAM 可被 RDMA 访问，注册到网卡获取"门牌号" |
| AcceleratorDevice | 多硬件适配层，一套代码支持 NVIDIA/AMD/昇腾/摩尔线程 |

**建议**: 生产部署时，务必开启 `MOONCAKE_OFFLOAD_USE_URING=true` 和大页支持——这两项配置可以让 SSD 读写带宽提升 30%-50%，对 Offload/Promotion 的延迟有直接影响。

**延伸阅读**：
- Linux io_uring: https://man7.org/linux/man-pages/man7/io_uring.7.html
- SPDK 文档: https://spdk.io/doc/
- NVIDIA cudaMallocHost: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- NVMe over Fabrics 规范: https://nvmexpress.org/specifications/

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
