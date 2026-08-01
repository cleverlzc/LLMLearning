# Mooncake Count-Min Sketch：16KB 内存如何管住百万 KV Cache 的"热度账本"

> **系列**: Mooncake 技术博客系列 | **类型**: 核心技术详解篇

当存储空间不足时，系统必须决定将哪些KV Cache淘汰。但精确追踪每个KV缓存块的访问次数本身会占用过多内存。

因此，Mooncake使用了Count-Min Sketch：一种巧妙的数据结构，利用固定且微小的内存来估算访问频率。虽然并非完全准确，但对于淘汰决策来说已足够好。

---

### 引言

Mooncake 面对的问题：集群中可能有百万级 KV Cache 对象，Master 需要决定哪些值得从 SSD 提升到 DRAM、Client 需要决定哪些值得缓存到本地热缓存。

精确计数？仅计数的数据存储开销就会大到爆炸。做过存储的同学（尤其是文件系统）应该都知道，当数据块切分的越小，元数据就会越多，到一定程度时，有限的存储容量，可能30%左右都给元数据用了，拿一块1TB硬盘来说，格式化完之后，再储存文件，可能也就只能存储650GB左右的数据，从用户的角度，利用率65%。

对于本身就很昂贵的KV Cache来说，这不可接受。于是，Mooncake采用了 Count-Min Sketch 巧妙的算法， 只需用 16KB 就够了。

##### Count-Min Sketch的核心思想：

创建一个网格：4行 x 4096列的小计数器。每个计数器仅占1字节（0-255）。总共只有16KB——远比单独追踪每个对象要节省得多。

当访问一个键时，对其进行4次哈希（每行一次）。在每次哈希位置处递增计数器。就像为同一本书盖4张不同的借阅卡一样。

将所有4个计数器的最小值作为估计的频率。最小值最接近真实情况——其他计数器可能因不同键的哈希冲突而被高估。

当所有计数器累计递增次数足够多时，对所有计数器进行衰减（将所有计数器减半）。这样逐渐遗忘久远的历史，使最近的访问模式占据主导——就像图书管理员更关注本周的借阅情况，而不是去年的。

##### 三大优势：
- **恒定空间**：无论你追踪多少百万个对象，这个草图（sketch）始终只占用宽度 x 深度字节。以Mooncake的默认设置为例：4 x 4096 = 16KB。这不过是沧海一粟。
- **足够好以用于淘汰**：该估计值可能高估（由于哈希冲突），但绝不会低估。对于淘汰而言，这意味着我们可能会稍微多保留一些稍微热门的项目——这是一个安全的错误，而非危险的错误。
- **自动衰减**：经过足够多的增量后，衰减会将所有计数器减半。旧的访问模式逐渐淡出，而最近热门的项目则始终保持在前列。图书管理员总能知道当前最受欢迎的书籍是什么。

---

### Count-Min Sketch：一张 4×4096 的"热度网格"

#### 数据结构

Count-Min Sketch 的核心就是一个二维计数器网格。Mooncake 的默认配置：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `width_` | 4096 | 每行计数器数量 |
| `depth_` | 4 | 行数（哈希函数个数） |
| 计数器类型 | `uint8_t` | 每个计数器 1 字节，范围 0-255 |
| **总内存** | **4 × 4096 = 16,384 字节 ≈ 16KB** | |

源码实现（`mooncake-store/include/count_min_sketch.h`）：

```cpp
class CountMinSketch {
   public:
    explicit CountMinSketch(size_t width = 4096, size_t depth = 4)
        : width_(width > 0 ? width : kDefaultWidth),
          depth_(depth > 0 ? depth : kDefaultDepth),
          table_(depth_, std::vector<uint8_t>(width_, 0)),
          total_increments_(0) {}
   private:
    static constexpr size_t kDefaultWidth = 4096;
    static constexpr size_t kDefaultDepth = 4;
    const size_t width_;
    const size_t depth_;
    std::vector<std::vector<uint8_t>> table_;  // depth × width 的计数网格
    size_t total_increments_;                  // 全局递增计数器，触发衰减
    mutable std::mutex mu_;                    // 线程安全
};
```

16KB 是什么概念？一个 GLM-5 的单条 KV Cache 约 160MB，16KB 连它的万分之一都不到。用万分之一的开销，管住百万条 KV Cache 的热度追踪——这就是概率数据结构的威力。

#### 写入：给同一本书盖 4 张借阅卡

当一个 KV Cache 对象被访问时，`increment()` 被调用：

```
对象 key → 4 次独立哈希 → 在 4 行各递增 1 个计数器
```

```cpp
uint8_t increment(const std::string &key) {
    std::lock_guard<std::mutex> lock(mu_);
    uint8_t min_val = UINT8_MAX;
    for (size_t i = 0; i < depth_; ++i) {
        size_t idx = hash(key, i) % width_;
        if (table_[i][idx] < UINT8_MAX) {
            ++table_[i][idx];          // 饱和递增，上限 255
        }
        min_val = std::min(min_val, table_[i][idx]);
    }
    if (++total_increments_ >= width_ * depth_) {
        decayLocked();                 // 自动衰减
    }
    return min_val;                    // 返回最小值作为频率估计
}
```

就像图书馆给同一本书在 4 个借阅站各盖一张借阅卡。每借一次，4 张卡上各盖一个戳。不同书籍的哈希可能撞到同一张卡——但 4 张卡同时被撞的概率极低。

#### 查询：4 张卡里取最小值

查询某个 key 的估计频率时，取 4 个计数器的**最小值**：

```cpp
uint8_t count(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint8_t min_val = UINT8_MAX;
    for (size_t i = 0; i < depth_; ++i) {
        size_t idx = hash(key, i) % width_;
        min_val = std::min(min_val, table_[i][idx]);
    }
    return min_val;
}
```

为什么取最小值？因为哈希冲突只会让计数器**偏高**——别的 key 可能恰好哈希到同一个位置，多盖了戳。最小值最接近真实情况，其他 3 个可能因冲突被高估。

这意味着 Count-Min Sketch 有一个极其重要的数学性质：

> **估计值 ≥ 真实值，永远不会低估。**

对于淘汰决策来说，这是"安全的错误"——我们可能会稍微多保留一些热门项，但绝不会误杀真正热门的对象。

#### 哈希：4 行各用不同的种子

每行使用不同种子的哈希函数，确保 4 个位置尽量分散：

```cpp
size_t hash(const std::string &key, size_t seed) const {
    size_t h = std::hash<std::string>{}(key);
    h ^= seed * 0x9e3779b97f4a7c15ULL + 0x517cc1b727220a95ULL;
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    return h;
}
```

这是 Knuth 风格的乘法散列混合——三轮 XOR 位移 + 乘法，用黄金分割常数 `0x9e3779b9` 和 FNV 常数 `0xff51afd7ed558ccd` 打散分布。`seed` 参数让每行产生独立的哈希函数。

---

### 自动衰减：让"热度账本"只记最近的事

#### 问题：旧热度会"沉淀"

如果不做任何处理，一个半年前被频繁访问、但现在已经无人问津的对象，它的计数器仍然很高——旧热度"沉淀"在网格里，干扰当前的淘汰决策。

就像图书管理员如果只看累计借阅次数，会发现《新华字典》永远是"最热门"的——但它现在的借阅量可能已经很低了。

#### 解法：累计递增达到阈值后，全体减半

Mooncake 的衰减策略极其简洁：

```cpp
void decayLocked() {
    for (size_t i = 0; i < depth_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            table_[i][j] >>= 1;   // 右移一位 = 除以 2 取整
        }
    }
    total_increments_ = 0;        // 重置全局计数器
}
```

触发条件：当 `total_increments_` 累计达到 `width_ × depth_ = 16,384` 次时，自动触发一次衰减。

```
时间线:
  ── increment ── increment ── ... ── 第 16384 次 ── 衰减! ── 重新计数 ──
     计数器: 7    计数器: 8              计数器: 42      计数器: 21
```

衰减的效果：

| 场景 | 衰减前 | 衰减后 | 效果 |
|------|--------|--------|------|
| 最近频繁访问的对象 | 42 | 21 | 仍然高，继续被识别为热门 |
| 久远的历史热门 | 8 | 4 | 迅速衰减，很快被遗忘 |
| 从未被访问 | 0 | 0 | 不受影响 |

衰减是指数级的——每衰减一次，旧数据的影响力就减半。经过若干次衰减后，远古历史几乎归零，而最近热门的对象始终保持在前列。图书管理员总能知道**本周**最受欢迎的书是什么。

#### 为什么是 16,384 次触发？

`width_ × depth_ = 4096 × 4 = 16,384`。这个阈值的设计有讲究：

- 太小（如 100）：衰减太频繁，热门对象的计数器还没积累够就被砍半了
- 太大（如 1,000,000）：衰减太慢，旧热度迟迟消不下去
- `width × depth`：恰好等于网格的总格子数，意味着平均每个格子被递增约 1 次就触发衰减——既不会让计数器溢出，又不会衰减过频

---

### Mooncake 中的两个应用场景

Count-Min Sketch 在 Mooncake 中有**两个完全不同的使用场景**：一个在 Client 端做本地热缓存准入，一个在 Master 端做 SSD→DRAM 提升决策。同一个数据结构，两种角色。

#### 场景一：Client 端——本地热缓存的"二刷准入"

**问题**：Client 有一个本地热缓存（Local Hot Cache），容量有限。每次 Get 到远程数据后，要不要把数据也缓存到本地？如果把所有数据都缓存，冷数据会挤占热数据的空间。

**解法**：用 Count-Min Sketch 做"二刷准入"——只有被访问 ≥ 2 次的对象才允许进入热缓存。

```
Get 请求到来
  │
  ├─ 本地热缓存命中? ── 是 ──→ 直接返回（不递增 Sketch）
  │
  └─ 未命中 ──→ 远程 RDMA 读取
                  │
                  └─ 读取成功后: ShouldAdmitToHotCache(key, cache_used=false)
                       │
                       ├─ Sketch 递增 → 估计值 < 2? ──→ 不缓存（"一刷"冷数据）
                       │
                       └─ Sketch 递增 → 估计值 ≥ 2? ──→ 缓存到本地热缓存
```

源码实现（`client_service.h:640-648`）：

```cpp
bool ShouldAdmitToHotCache(const std::string& key, bool cache_used) {
    if (!(hot_cache_ && !cache_used)) {
        return false;   // 没有热缓存 或 数据已从缓存返回 → 不准入
    }
    if (admission_sketch_ == nullptr) {
        return true;    // 没有 Sketch → 无条件准入
    }
    return admission_sketch_->increment(key) >= admission_threshold_;
}
```

关键细节：

| 设计点 | 实现 | 原因 |
|--------|------|------|
| 缓存命中时不递增 | `cache_used=true` 时直接返回 false | 避免已缓存的对象反复递增计数器，防止"富者愈富" |
| 默认阈值 = 2 | `admission_threshold_ = 2` | 第一次访问不缓存，第二次才缓存——过滤一次性冷数据 |
| 环境变量可调 | `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD` | 不同业务负载的最优阈值不同 |

**为什么是"二刷"而不是"一刷"？** 因为 LLM 推理中有大量一次性访问的 KV Cache——Prefill 产生后可能再也不用。如果一刷就缓存，这些"僵尸数据"会迅速填满热缓存，把真正反复使用的热数据挤出去。二刷准入就像超市只给"回头客"办会员卡——来一次的过客不值得建档。

#### 场景二：Master 端——SSD→DRAM 的"四道闸门"提升

**问题**：Master 管理着分层存储，有些 KV Cache 只存在 SSD 上（LOCAL_DISK 副本），访问延迟高。如果某个 SSD 上的对象被频繁读取，应该把它提升到 DRAM（MEMORY 副本）。但提升本身有成本（SSD 读取 + RDMA 写入），不能对所有对象都做。

**解法**：`TryPushPromotionQueue()` 用四道闸门层层过滤，Count-Min Sketch 是第一道——频率闸门。

```
Get 请求命中 SSD-only 对象
  │
  └─ TryPushPromotionQueue()
       │
       ├─ 闸门 1: 频率闸门（Count-Min Sketch）
       │    Sketch 递增 → 估计值 < 2? ──→ 拒绝（"还不够热"）
       │
       ├─ 闸门 2: 水位闸门
       │    DRAM 使用率 ≥ 高水位? ──→ 拒绝（"DRAM 压力太大，提升进去也会被驱逐"）
       │
       ├─ 闸门 3: 去重闸门
       │    已有提升任务在途? 或已有 MEMORY 副本? ──→ 拒绝（"已经在做了"）
       │
       ├─ 闸门 4: 容量闸门
       │    全局在途提升数 ≥ 50000? ──→ 拒绝（"队列满了"）
       │
       └─ 全部通过 ──→ 入队提升任务
                         Client 下次心跳时领取任务
                         SSD 读取 → RDMA 写入 DRAM
```

源码实现（`master_service.cpp:5248-5308`）：

```cpp
void MasterService::TryPushPromotionQueue(const ObjectIdentity& object_id) {
    if (!promotion_on_hit_ || !promotion_sketch_) {
        return;
    }
    const auto admission_key =
        MakeTenantScopedStorageKey(object_id.tenant_id, key);

    // 闸门 1: 频率闸门
    const uint8_t freq = promotion_sketch_->increment(admission_key);
    if (freq < promotion_admission_threshold_) {
        MasterMetricManager::instance().inc_promotion_rejected_frequency();
        return;
    }

    // 闸门 2: 水位闸门
    const double used_ratio =
        MasterMetricManager::instance().get_global_mem_used_ratio();
    if (used_ratio >= eviction_high_watermark_ratio_) {
        MasterMetricManager::instance().inc_promotion_rejected_watermark();
        return;
    }

    // 闸门 3: 去重闸门
    if (tenant_state.promotion_tasks.count(key) > 0) { return; }
    if (metadata.HasReplica(&Replica::fn_is_memory_replica)) { return; }

    // 闸门 4: 容量闸门
    if (promotion_in_flight_.load(std::memory_order_relaxed) >=
        promotion_queue_limit_) {
        MasterMetricManager::instance().inc_promotion_rejected_cap();
        return;
    }
    // ... 入队提升任务 ...
}
```

四道闸门各有各的考量：

| 闸门 | 检查什么 | 为什么需要 |
|------|----------|------------|
| 频率闸门 | CMS 估计值 ≥ 阈值 | 只提升真正热门的对象，避免为一次性冷数据浪费 IO |
| 水位闸门 | DRAM 使用率 < 高水位 | DRAM 已经在驱逐了，提升进去也会被立即驱逐，白费 IO |
| 去重闸门 | 无重复在途任务 | 避免同一个对象被重复提升 |
| 容量闸门 | 在途任务数 < 50,000 | 限制全局提升并发，防止 IO 风暴压垮集群 |

---

### 两张 Sketch 的对比

虽然 Client 和 Master 各自持有一个 CountMinSketch 实例，但它们的角色和配置略有不同：

| 维度 | Client 端（本地热缓存准入） | Master 端（SSD→DRAM 提升） |
|------|---------------------------|---------------------------|
| **Sketch 实例** | `admission_sketch_` | `promotion_sketch_` |
| **追踪对象** | 本地 Get 请求的 key | 全局 SSD-only 对象的 key |
| **决策** | 是否缓存到本地 DRAM | 是否提升到远端 DRAM |
| **默认阈值** | 2（环境变量 `MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD`） | 2（启动参数 `--promotion_admission_threshold`） |
| **触发时机** | 每次 Get 完成后 | 每次 GetReplicaList 命中 SSD-only 对象时 |
| **数据流向** | 远端 DRAM → 本地 DRAM | SSD → 远端 DRAM |
| **额外闸门** | 无（只看频率） | 水位 + 去重 + 容量 |
| **Sketch 维度** | 4096 × 4 = 16KB | 4096 × 4 = 16KB |
| **衰减阈值** | 16,384 次递增 | 16,384 次递增 |

一个有趣的细节：Client 端在缓存命中时**不递增** Sketch 计数器。这是刻意的设计——如果缓存命中也递增，那已经缓存的热对象计数器会越来越高，永远不会被衰减淘汰，形成"富者愈富"的正反馈。不递增则让 Sketch 只记录"从远程获取"的频率，更准确地反映缓存未命中的热度。

---

### 为什么不用精确计数？

一个自然的疑问：为什么不用 `std::unordered_map<std::string, uint64_t>` 精确计数？

| 维度 | 精确计数（HashMap） | Count-Min Sketch |
|------|---------------------|-------------------|
| **内存开销** | 每个key约 100-200 字节（string + hash 开销） | 固定 16KB，与对象数无关 |
| **100 万对象** | ~100-200 MB | **16 KB** |
| **精度** | 精确 | 可能高估，绝不低估 |
| **衰减** | 需要遍历所有条目 | 整体右移，O(width × depth) |
| **线程安全** | 细粒度锁或无锁 | 单 mutex（16KB 操作极快） |

100 万对象的精确计数需要 100-200 MB 内存——这在 Master 进程中不是不可接受，但在 Client 端（每个推理进程一个 Client 实例）中就是沉重的负担。更关键的是，精确计数需要**逐条衰减**，遍历百万条目的开销远大于对 16KB 网格做一次右移。

Count-Min Sketch 的精髓在于：**淘汰决策不需要精确数字，只需要"够不够热"的相对判断**。估计值是 5 还是 7 不重要，重要的是它是否 ≥ 2（阈值）。高估只会让我们多保留一些对象——这是安全的错误。

---

### 恒定空间：无论百万还是十亿

Count-Min Sketch 最优雅的数学性质是**空间与对象数无关**：

```
追踪 1 万个对象 → 16 KB
追踪 100 万个对象 → 16 KB
追踪 10 亿个对象 → 还是 16 KB
```

网格大小只取决于两个参数：`width`（列数）和 `depth`（行数）。对象再多，哈希函数也会把它们映射到固定的网格位置。冲突会增加，但取 4 行最小值的机制保证了即使冲突，估计值也只是偏高而非偏低。

用信息论的视角看：Count-Min Sketch 用 16KB 的信息量，编码了百万对象的"热度排序"。它当然不可能精确还原每个对象的频率——那需要远超 16KB 的信息量。但它编码了**足够做淘汰决策的近似排序**，而这正是系统需要的。

---

### 从数学到工程：三个关键设计决策

#### 决策 1：为什么计数器是 uint8_t 而不是 uint32_t？

`uint8_t` 的范围只有 0-255，看似太小。但配合自动衰减，计数器永远不会溢出——达到 255 时饱和不递增，而衰减会在计数器到达高位之前就触发减半。

如果用 `uint32_t`，每个计数器 4 字节，网格膨胀到 64KB——仍然不大，但衰减时需要遍历的数据量翻了 4 倍。`uint8_t` 的选择是"够用就好"的工程哲学：淘汰决策只需要区分"0 次、1 次、≥ 2 次"三档，8 位精度绰绰有余。

#### 决策 2：为什么 depth = 4？

depth 决定了哈希函数的个数，直接影响估计精度。理论上，depth 越大，4 行同时冲突的概率越低，估计越准确。但每增加一行，increment 和 count 的开销就多一次哈希计算和一次内存访问。

depth = 4 是经典选择——在论文和实践中，4-5 行的 Count-Min Sketch 已能将估计误差控制在可接受范围内。Mooncake 用默认值 4，不做参数化暴露，说明实测中 4 行已经足够。

#### 决策 3：为什么衰减用右移而不是除以 2？

```cpp
table_[i][j] >>= 1;   // 右移
// 而不是
table_[i][j] /= 2;    // 除法
```

对于无符号整数，右移和除以 2 的数学结果完全相同（都是向下取整）。但右移是 CPU 的单周期指令，除法可能需要十几个周期。对 16,384 个计数器做批量操作，这个微优化是有意义的。

> 笔者注：纵观深度学习的发展规律，好的主流的模型之所以会大浪淘沙，成为主流，其首先在数学上一定自然且必然的，在工程结构实现上一定是简洁与统一的，并且，有着良好的可解释性和易理解性，以及通用朴素与泛化能力。不符合这些特征的，仅能够在短期大力出奇迹，长期会淹没在历史长河中。

---

### 小结

Count-Min Sketch 在 Mooncake 中扮演的角色，可以用一句话概括：

> **用 16KB 的"热度网格"，为百万 KV Cache 对象维护一个"只记最近、宁可多留、绝不误杀"的近似频率账本。**

三个核心机制环环相扣：

1. **4 行哈希 + 取最小值**：用 4 个独立哈希函数映射到网格，取最小值作为估计。哈希冲突只会高估，不会低估——对淘汰决策来说是安全的错误。

2. **自动衰减**：累计递增达到 16,384 次后，全体计数器减半。旧热度指数级消退，新热度持续积累——系统始终关注"最近谁最热"。

3. **二刷准入**：默认阈值 = 2，第一次访问不触发任何动作，第二次才准入缓存或提升。过滤掉一次性冷数据，把宝贵的缓存空间留给真正的热对象。

在 Client 端，它是本地热缓存的"门卫"——只放行回头客；在 Master 端，它是 SSD→DRAM 提升的"频率闸门"——只提升真正热门的对象。同一个 16KB 的数据结构，两种角色，一个目标：**用最小的内存开销，做出足够好的淘汰决策**。

---

*本文属于 [Mooncake 技术博客系列]，欢迎持续关注。*
