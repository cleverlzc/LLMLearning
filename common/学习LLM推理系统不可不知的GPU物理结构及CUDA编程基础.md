# 学习LLM推理系统不可不知的GPU物理结构及CUDA编程基础

> 副标题：GPU不是更快的CPU：一文入门CUDA并行编程

要学习LLM大模型推理，离不开GPU计算，离不开GPU和CPU的数据存储及传输，那就必须要理解GPU的物理结构，以及GPU的编程逻辑。GPU是硬件，直接对硬件编码对编程不友好，因此有了CUDA（国内有了昇腾NPU及CANN，核心原理和概念是一样的）。

本文整体定位是一个入门科普，目的是为了更好的掌握LLM推理系统及系统优化，不得不学习和理解的GPU物理结构和并发编程基础，行文逻辑从"为什么需要CUDA/GPU"出发，用第一性原理推导出CUDA的存在逻辑，拆解GPU的物理组成，再手把手带你写出第一个CUDA程序。

---

## 一、概念篇：为什么需要CUDA？

### 1.1 GPU是什么

**GPU = Graphics Processing Unit，图形处理器**。

最初，GPU是为了打游戏而生的——3D游戏需要把数百万个像素点渲染到屏幕上，这是海量的并行计算。后来人们发现：除了画图，GPU特别适合干任何"大量重复计算"的活，比如深度学习、科学计算、视频编码。

### 1.2 CPU与GPU：搭档而非对手

#### 两条不同的设计路线

| | **CPU（中央处理器）** | **GPU（图形处理器）** |
|---|---|---|
| **核心数** | 少（4~64核） | 多（数千个） |
| **单个核心能力** | 强，擅长复杂逻辑 | 弱，只做简单算术 |
| **擅长的事** | 串行任务、复杂分支判断 | 并行任务、大量相同计算 |
| **比喻** | 一个数学教授 | 一万个小学生的算术班 |
| **典型场景** | 操作系统、业务逻辑、数据库 | 图像渲染、矩阵乘法、AI训练 |

**一句话总结**：CPU像一把瑞士军刀，啥都能干但一次干一件；GPU像一万台计算器，只能算数但能同时算。

#### 协同工作：项目经理与工人

CPU和GPU不是竞争关系，而是搭档。在一台典型的服务器里，它们各司其职，形成一条完整的计算流水线：

```
┌──────────────────────────────────────────────────────────┐
│                    一次AI训练的完整流程                      │
│                                                          │
│  CPU 负责                          GPU 负责               │
│  ────────                          ────────               │
│  ① 读取训练数据（磁盘I/O）                                   │
│  ② 数据预处理（解码、增强、shuffle）                           │
│  ③ ──────────────────────────→  ④ 前向传播（矩阵乘法）      │
│                                ⑤ 计算损失                  │
│                                ⑥ 反向传播（梯度计算）        │
│                                ⑦ 更新模型参数（optimizer）    │
│  ⑧ ←──────────────────────────                           │
│  ⑨ 记录日志、检查点                                          │
│  ⑩ 判断是否收敛、是否结束                                    │
│  ……回到①继续下一轮……                                       │
└──────────────────────────────────────────────────────────┘
```

**分工原则很简单**：

| 谁干 | 干什么 | 为什么 |
|------|--------|--------|
| **CPU** | 数据搬运、逻辑判断、系统调度、错误处理 | 这些事情有复杂的分支、依赖外部设备，GPU不擅长 |
| **GPU** | 矩阵运算、大规模并行的数值计算 | 数据量大但计算模式统一，CPU太慢 |
| **CPU→GPU** | 把数据从内存搬到显存 | CPU是"调度员"，决定什么时候让GPU干什么 |
| **GPU→CPU** | 把计算结果搬回来 | GPU是"干活的大军"，干完了把成果交回 |

用一个现实的比喻：**CPU是工地项目经理，GPU是几百个工人**。项目经理自己不搬砖，但他负责看图纸、安排工序、协调材料、检查质量；工人不负责决策，但人多力量大，砌墙浇筑全上。没有项目经理，工人不知道干什么；没有工人，项目经理一个人也盖不了楼。

### 1.3 为什么需要CUDA：从物理极限推导

理解了CPU和GPU的区别，接下来回答一个更深层的问题：**为什么CUDA会出现？**

#### 推导一：单核性能撞墙了

很长一段时间里，计算机性能的提升靠的是"把CPU主频做高"——从1GHz到3GHz再到5GHz。但这条路走到了尽头：

- **功耗墙**：芯片功耗与频率的三次方成正比（P ∝ f³）。频率翻倍，功耗涨八倍。散热扛不住。
- **内存墙**：CPU算得再快，也得等内存喂数据。内存速度每年只涨几个百分点，远远跟不上CPU。
- **指令级并行极限**：一个程序里能并行执行的指令就那么多，乱序执行、分支预测的收益越来越小。

**结论**：单核性能已经接近物理极限，不能再指望"把一颗核心做得更快"。

#### 推导二：既然不能更快，那就更多

单核撞墙了，唯一的出路就是**并行**——多个核心同时干活。这就是为什么CPU从单核走向了多核（4核、8核、64核）。

但CPU的多核有两个天花板：

1. **核心太贵**：CPU的每个核心要支持乱序执行、分支预测、大容量缓存、复杂的中断和调度——一颗核心的设计成本和晶体管预算都很高。堆到几十核已经是经济极限。
2. **核心太"聪明"**：CPU核心擅长处理复杂的分支逻辑，但很多计算任务（图像、矩阵、AI）根本不需要复杂逻辑，只需要"对大量数据重复做同一种简单运算"。让CPU干这种活，等于让博学的教授去抄写一万份试卷——大材小用，还慢。

#### 推导三：GPU的设计哲学——吞吐量优先

既然有些任务不需要"聪明"，只需要"多"，那就换一种芯片设计思路：

> **CPU的设计哲学**：低延迟。用少量强大的核心，把单个任务做完、做快。

> **GPU的设计哲学**：高吞吐。用大量简单的核心，让尽可能多的任务同时跑。

GPU把CPU花在单核复杂度上的晶体管预算，改投到"堆更多核心"上。一颗A100 GPU有6912个CUDA核心，每个核心比CPU核心简单得多，但胜在数量碾压。

**关键洞察**：GPU不是"更快的CPU"，而是"另一种权衡"——用单任务延迟换总吞吐量。

#### 推导四：硬件有了，但程序员用不上

GPU硬件的设计哲学很美好，但这里出现了一个**巨大的鸿沟**：

- GPU原本是为图形渲染设计的，它的指令集、编程接口都是图形相关的（顶点、像素、纹理）。
- 通用程序员想用GPU算矩阵、跑神经网络，却没法直接编程——得用OpenGL/DirectX把计算任务"伪装"成画图操作，代码难写、难调、难维护。

#### 推导五：CUDA = 在GPU硬件和程序员之间架桥

2007年NVIDIA正式推出CUDA SDK（2006年底发布预览版），本质上是做了三件事：

1. **提供编程模型**：让你用类似C语言的语法写GPU代码（`__global__`标记核函数，`<<<blocks, threads>>>`配置并行度），不用再伪装成图形操作。
2. **抽象硬件细节**：你不需要手动管"哪个核心干哪条指令"，只需要声明"我要启动多少线程"，CUDA和GPU硬件帮你调度。
3. **提供工具链**：编译器（nvcc）、运行时库（cudaMalloc/cudaMemcpy）、调试器（cuda-gdb）、性能分析工具（Nsight），形成完整的开发体验。

#### 小结：从物理极限到CUDA的完整逻辑链

```
单核性能撞墙（功耗墙、内存墙）
        ↓
出路只能是并行
        ↓
CPU多核路线：核心强但贵，堆到几十核就到顶
        ↓
GPU换思路：核心简单但多，用吞吐量换延迟
        ↓
但GPU原本只能画图，程序员用不上
        ↓
CUDA出现：让通用程序员能用C语言指挥GPU
        ↓
深度学习爆发：算力需求井喷，GPU+CUDA成为AI基础设施
```

> **CUDA = Compute Unified Device Architecture，统一计算设备架构**。GPU是硬件（几千个计算核心），CUDA是软件（指挥这些核心的编程语言+工具链）。你用CUDA写程序，程序跑在GPU上。

---

## 二、模型篇：CUDA编程模型

理解了"为什么需要CUDA"，接下来搞懂CUDA的编程模型。

### 2.1 Host与Device

GPU并不是一个独立运行的计算平台，而需要与CPU协同工作，可以看成是**CPU的协处理器**，因此当我们在说GPU并行计算时，其实是指的**基于CPU+GPU的异构计算架构**。

在异构计算架构中，GPU与CPU通过PCIe总线连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device）。

因此，CUDA编程模型是一个**异构模型**，需要CPU和GPU协同工作：

- **Host**：指CPU及其内存，负责调度和控制
- **Device**：指GPU及其显存，负责并行计算

> 这两个概念在LLM推理系统、RDMA/RoCE编程中非常常见。

CUDA程序中既包含Host代码，又包含Device代码，它们分别在CPU和GPU上运行，同时可以通过数据拷贝进行通信。

### 2.2 GPU的"工厂架构"

如果把GPU想象成一个**大型工厂**，那么会有如下物理划分：

| 概念 | 比喻 | 说明 |
|------|------|------|
| **SM（流多处理器）** | 车间 | GPU里有几十个车间，每个车间能同时干活 |
| **线程（Thread）** | 工人 | 每个工人执行相同的任务，但处理不同的数据 |
| **线程块（Block）** | 工作组 | 一组工人在一起协作，能共享工具和材料 |
| **网格（Grid）** | 全厂 | 所有工作组的总和 |

**关键数字**（以NVIDIA A100为例）：
- 108个SM（车间）
- 每个SM最多2048个线程（工人）
- 总共能同时运行超过10万个线程

每个线程都有唯一编号，用来决定自己处理哪部分数据：

```c
// 一维情况
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 解释：
// blockIdx.x = 你在第几个工作组（从0开始）
// blockDim.x = 每个工作组有多少工人
// threadIdx.x = 你在工作组内的编号（从0开始）
```

**比喻**：
你要找"第3工作组第5号工人"：
- `blockIdx.x = 3`（第3组）
- `blockDim.x = 256`（每组256人）
- `threadIdx.x = 5`（组内第5号）
- 全局编号 = 3 × 256 + 5 = **773号工人**

### 2.3 CUDA程序的典型执行流程

一个典型的CUDA程序遵循以下5步：

1. 分配Host内存，并进行数据初始化；
2. 分配Device内存，并从Host将数据拷贝到Device上（**H2D**，Host to Device）；
3. 调用CUDA的核函数在Device上完成指定的运算；
4. 将Device上的运算结果拷贝到Host上（**D2H**，Device to Host）；
5. 释放Device和Host上分配的内存。

后面的实战章节，你会反复看到这5步模式。

---

## 三、实战篇：动手写CUDA程序

### 3.1 第一个程序：Hello GPU

一个CUDA程序有两部分：
1. **Host代码**：在CPU上运行，负责调度、数据传输
2. **Device代码**：在GPU上运行，负责并行计算

```c
#include <stdio.h>

// ===== Device代码：在GPU上执行 =====
__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

// ===== Host代码：在CPU上执行 =====
int main() {
    // 启动GPU函数，256个线程并行执行
    helloKernel<<<1, 256>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    printf("Done!\n");
    return 0;
}
```

**关键标记**：
- `__global__`：告诉编译器"这是GPU函数，从CPU调用，在GPU上运行（数学并行计算）"
- `<<<1, 256>>>`：执行配置（1个线程块，每块256个线程）

**编译和运行**：

```bash
# 编译（使用nvcc编译器）
nvcc hello.cu -o hello

# 运行
./hello
```

**输出**（顺序可能不同）：
```
Hello from GPU thread 0!
Hello from GPU thread 128!
Hello from GPU thread 1!
...
Done!
```

**注意**：线程执行顺序是随机的！这是并行计算的本质。

### 3.2 让GPU处理数组

真实场景中，GPU要处理大量数据。看这个例子：把数组每个元素加1。

**CPU版本（串行）**：

```c
void addOneCPU(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] + 1.0f;
    }
}
```

**问题**：100万个元素，循环100万次，慢！

**GPU版本（并行）**：

```c
// GPU核函数：每个线程处理一个元素
__global__ void addOneGPU(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 防止越界
    if (i < n) {
        data[i] = data[i] + 1.0f;
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // 1. 分配CPU内存
    float* h_data = (float*)malloc(size);
    
    // 2. 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }
    
    // 3. 分配GPU内存
    float* d_data;
    cudaMalloc(&d_data, size);
    
    // 4. 把数据从CPU复制到GPU
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 5. 启动GPU计算
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;  // 向上取整
    addOneGPU<<<numBlocks, blockSize>>>(d_data, n);
    
    // 6. 把结果从GPU复制回CPU
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // 7. 清理
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

**关键步骤解析**：

| 步骤 | 函数 | 作用        |
|------|------|-----------|
| 分配GPU内存 | `cudaMalloc` | 在GPU上开辟空间 |
| CPU→GPU | `cudaMemcpy` (HostToDevice) | 把数据搬到GPU  |
| 启动计算 | `<<<numBlocks, blockSize>>>` | 启动并行执行    |
| GPU→CPU | `cudaMemcpy` (DeviceToHost) | 把结果搬回CPU  |
| 释放GPU内存 | `cudaFree` | 清理GPU资源   |

**为什么这么麻烦？**
因为CPU和GPU有各自独立的内存，数据必须显式传输。这是GPU编程的核心挑战之一。

> 笔者注：这个步骤是LLM运行的基础，算子（operator）开发的基础。LLM的“智能”正式组合了几十种算子（常用的只有十几种）经过成千上亿次的重复计算出来的。

#### 线程块数量怎么算？

100万个元素，每个线程块256个线程，需要多少个线程块？

```c
int n = 1000000;
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;  // = 3907
```

**为什么用这个公式？**
- 1000000 ÷ 256 = 3906.25
- 向上取整 = 3907
- `(n + blockSize - 1) / blockSize` 是整数除法的向上取整技巧

**验证**：
- 3906个块 × 256线程 = 999936个线程
- 还差64个元素 → 需要第3907个块
- 第3907个块有256个线程，但只有64个干活，其余192个闲置

**优化建议**：
`blockSize` 通常选 128、256、512（2的幂次），这样GPU调度更高效。

### 3.3 GPU的内存层次

GPU内存分好几层，速度差异巨大：

| 内存类型 | 速度 | 大小 | 谁能访问 | 生命周期 |
|----------|------|------|----------|----------|
| **寄存器** | 最快 | 极少 | 单个线程 | 线程结束就消失 |
| **共享内存** | 很快 | 48KB~164KB/块（视架构而定） | 同块内线程共享 | 核函数结束就消失 |
| **全局内存** | 较慢 | 几十GB | 所有线程都能访问 | 你手动分配/释放 |
| **常量内存** | 较快 | 64KB | 所有线程只读 | 程序运行期间 |

#### 共享内存实战：矩阵转置

**问题**：把矩阵行列互换
**朴素方法**：每个线程读一次全局内存，写一次全局内存
**优化方法**：用共享内存做中转，减少全局内存访问

```c
#define TILE_DIM 32

__global__ void transposeGPU(float* out, const float* in, int width, int height) {
    // 分配共享内存
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    // 计算当前线程要处理的块
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 从全局内存读到共享内存（合并访问）
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    
    // 等待所有线程完成
    __syncthreads();
    
    // 计算转置后的位置
    int newX = blockIdx.y * TILE_DIM + threadIdx.x;
    int newY = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 从共享内存写到全局内存（合并访问）
    if (newX < height && newY < width) {
        out[newY * height + newX] = tile[threadIdx.x][threadIdx.y];
    }
}
```

**关键技巧**：
- `__shared__`：声明共享内存
- `__syncthreads()`：同步点，确保所有线程都完成读操作后再写
- **合并访问**：相邻线程访问相邻内存，GPU能一次性读取，效率更高

### 3.4 综合实战：向量点积

**问题**：计算两个向量的点积 `sum = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]`

```c
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// GPU核函数：每个线程计算一部分点积
__global__ void dotProduct(const float* a, const float* b, float* partialSums, int n) {
    __shared__ float sdata[256];  // 共享内存
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算自己的部分
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();
    
    // 归约：在共享内存中求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 每个线程块的第一个线程把结果写到全局内存
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // 分配CPU内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float h_result = 0.0f;
    
    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 计算线程配置
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // 分配GPU内存
    float *d_a, *d_b, *d_partialSums;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));
    
    // 传输数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // 启动核函数
    dotProduct<<<numBlocks, blockSize>>>(d_a, d_b, d_partialSums, n);
    
    // 把部分和传回CPU
    float* h_partialSums = (float*)malloc(numBlocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    // CPU上求和
    for (int i = 0; i < numBlocks; i++) {
        h_result += h_partialSums[i];
    }
    
    printf("点积结果: %f\n", h_result);  // 应该是 2000000.0
    
    // 清理
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partialSums);
    free(h_a); free(h_b); free(h_partialSums);
    
    return 0;
}
```

**核心思想**：
1. 每个线程计算一对元素的乘积
2. 用共享内存在块内归约求和
3. 每个块输出一个部分和到全局内存
4. CPU最后把所有部分和加起来

---

## 四、避坑与优化篇

### 4.1 常见错误与调试

#### 错误1：越界访问

```c
__global__ void badKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = 1.0f;  // 危险！可能越界
}

// 正确做法：加边界检查
__global__ void goodKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {  // 防止越界
        data[i] = 1.0f;
    }
}
```

#### 错误2：忘记同步

```c
__global__ void syncBug() {
    __shared__ float shared[256];
    
    shared[threadIdx.x] = threadIdx.x;
    // 这里没有 __syncthreads()
    
    float val = shared[255 - threadIdx.x];  // 可能读到未初始化的数据！
}
```

#### 错误3：内存泄漏

```c
float* d_data;
cudaMalloc(&d_data, size);
// 忘记 cudaFree(d_data);  // 内存泄漏！
```

#### 调试工具

```bash
# 检查CUDA错误（CUDA 11.2+推荐用compute-sanitizer，旧版用cuda-memcheck）
compute-sanitizer ./your_program

# 使用cuda-gdb调试
cuda-gdb ./your_program
```

**错误检查宏**（推荐加到代码里）：

```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 使用
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### 4.2 性能优化四板斧

#### 优化1：减少CPU-GPU数据传输

**错误示范**：
```c
for (int i = 0; i < 1000; i++) {
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // 每次都传！
    kernel<<<...>>>(d_data);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // 每次都传！
}
```

**正确做法**：一次性传大量数据，GPU批量处理，再一次性传回。

#### 优化2：合并内存访问

**错误示范**（跨步访问）：
```c
__global__ void badAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * 100] = 1.0f;  // 相邻线程访问间隔100的位置，慢！
}
```

**正确做法**（连续访问）：
```c
__global__ void goodAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = 1.0f;  // 相邻线程访问相邻位置，快！
}
```

#### 优化3：选择合适的线程块大小

**经验法则**：
- 线程块大小：128、256、512（2的幂次）
- 每个SM至少启动4个线程块，才能充分利用硬件
- 用 `cudaOccupancyMaxPotentialBlockSize` 让CUDA帮你选最优值

```c
int blockSize;
int minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
```

#### 优化4：使用CUDA事件计时

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<...>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel执行时间: %f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 五、进阶与资源

### 5.1 学习路线

#### 进阶方向
1. CUDA流与异步执行（重叠计算与传输）
2. Warp级编程与协作组（Cooperative Groups）
3. 矩阵运算库（cuBLAS）与深度学习库（cuDNN）
4. 多GPU编程（P2P、NCCL）

#### 持续精进
1. 性能分析工具（Nsight Systems、Nsight Compute）
2. 读优秀开源代码（TensorRT、PyTorch底层、vLLM）
3. 研究特定领域的优化技术（FlashAttention、推测解码等）

### 5.2 常用资源

| 资源 | 链接 | 说明 |
|------|------|------|
| CUDA官方文档 | https://docs.nvidia.com/cuda/ | 最权威的参考 |
| CUDA编程指南 | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | 必读！详细解释所有概念 |
| NVIDIA开发者论坛 | https://forums.developer.nvidia.com/ | 遇到问题先搜索这里 |
| GitHub CUDA示例 | https://github.com/NVIDIA/cuda-samples | 官方代码示例 |

---

## 总结

**CUDA编程的核心要点**：

1. **分工明确**：CPU负责调度和数据传输，GPU负责并行计算
2. **线程模型**：网格→线程块→线程，每个线程有唯一编号
3. **内存管理**：CPU和GPU内存独立，数据需要显式传输
4. **性能关键**：减少数据传输、合并内存访问、合理使用共享内存

**最后一句话**：
GPU编程不难，难的是转变思维方式——从"一步步执行"变成"让成千上万个线程同时干活"。多写、多调试、多优化，你也能成为CUDA高手！

### 附录，什么是算子（operator）？

ps. 算子这个词，乍听起来有点高深，笔者的一位朋友在前年转到算子开发的时候，大家还在饭桌上热烈讨论过算子是什么，确实有点神秘。其实算子本质上是一个纯粹的数学运算，比如矩阵乘法、Softmax，底层可以进一步融合（kernel fusion），能够高效运行在GPU/NPU上，一种“特殊数学语义特定硬件的原子计算函数”。

kernel fusion或者operator fusion，operator在CUDA编程的概念里叫kernel，fusion本质上是将多个operator/kernel融合在一起优化整体任务的计算效率。

LLM大模型中最最最重要的一个算子：
```
输入: 向量 x (768维), 权重矩阵 W (768×768)
操作: x × W
输出: 新向量 (768维)
```
那就是MatMul（矩阵乘法，Matrix Multiplication）算子。Q/K/V 投影、FFN 的 W₁·x 和 W₂·x、LM Head 的输出映射，全都是这个算子。大模型的计算量大部分花在矩阵乘法上。

大模型的全部计算 = 几十种算子组合 × 重复成千上亿次，让它显得"智能"的，不是单个算子有多复杂，是那些训练出来的超大量的参数（千亿参数，GLM 5.2 7440亿即744B；万亿参数，KIMI K3 2.8万亿即2.8T），如何组织步骤和算子计算。


### 扩展阅读
CUDA编程入门极简教程：https://zhuanlan.zhihu.com/p/34587739

> 最后总结一句话，CPU 是一个什么都会的全能选手，GPU 是一万个只会做简单数学运算（比如矩阵乘法、加法、取指数等）的工人。大模型的大，大量参数（一堆向量、矩阵数字）快速运算，需要的恰好是后者——不需要有多聪明，只需要人多、动作快。大模型的“智能”离不开GPU，不是因为GPU有什么“魔法”，只是因为大模型的核心运算--矩阵乘法--恰好是GPU最最最擅长的事情。

如果觉得有用，转发给身边正在学 LLM 大模型推理的朋友。
