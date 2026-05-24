---
title: 目录
nav_order: 1
---

# vLLM 源码深度分析

## 背景

大语言模型的能力越来越强，企业本地部署开源模型的需求也在快速增长。但在 vLLM 出现之前，部署 LLM 面临三个棘手问题：

- **显存浪费严重**：传统推理引擎为每个请求预分配一整块连续显存来存放 KV cache，按最大长度预留——实际只用了一小部分，大量显存白白空置。
- **并发上不去**：显存被碎片化占用后，能同时服务的请求数大打折扣，GPU 算力大量闲置。
- **成本居高不下**：单次推理价格降不下来，规模化服务难以为继。

[vLLM](https://github.com/vllm-project/vllm) 于 2023 年 6 月由 UC Berkeley [Sky Computing Lab](https://sky.cs.berkeley.edu) 开源，最初是 **PagedAttention** 技术的展示项目。PagedAttention 借鉴操作系统虚拟内存的思想，将 KV cache 划分为固定大小的 block（页），一举消除了预分配导致的碎片问题——官方数据显示内存浪费不到 4%，吞吐量相比 HuggingFace Transformers 最高提升 **24 倍**。

这一突破迅速引发社区关注。在开源之前，vLLM 已在 [Chatbot Arena](https://arena.ai) 悄然支撑了数百万用户的 Vicuna 对话服务——用 vLLM 替代原始 HF Transformers 后端后，内部基准测试显示吞吐提升高达 **30 倍**。此后 vLLM 被大量公司集成到生产系统，社区贡献者超过 2000 人，成为生产系统上事实上的 LLM 推理标准。

然而 vLLM 源码规模庞大（Python 约 20 万行 + C++/CUDA 数万行 + Rust 前端），架构复杂（多进程通信、三套算子注册机制、六种 attention backend、数十种量化方法），官方文档侧重于使用和配置，对内部实现鲜有涉及。

本系列文档从源码出发，逐模块拆解 vLLM 的核心机制——由浅入深，从架构全景逐步深入到 CUDA kernel 的地址翻译细节。

## 文档导航

| # | 文档 | |
|---|------|------|
| 1 | [架构概述](01-架构概述.md) | 全局鸟瞰：六层架构、完整生命周期、关键子系统 |
| 2 | [代码结构分析](02-代码结构分析.md) | 顶层目录、CMake/setuptools 构建、C++/CUDA/Rust 代码组织 |
| 3 | [初始化流程分析](03-初始化流程分析.md) | Engine 启动序列：配置解析 → Worker/Executor → KV cache 分配 → Scheduler |
| 4 | [模型加载流程分析](04-模型加载流程分析.md) | 9 种 ModelLoader、600+ 模型注册表、权重迭代与 TP 分片 |
| 5 | [推理流程分析](05-推理流程分析.md) | Step 循环：调度 → Attention → 采样 → 输出 |
| 6 | [算子注册与分发](06-算子注册与分发.md) | Python ↔ C++/CUDA/Triton 的桥梁 |
| 7 | [PagedAttention 实现分析](07-PagedAttention-实现分析.md) | Block Table、KV cache 管理、CUDA/Triton kernel 深入 |


## 如何使用

```bash
git clone --recurse-submodules https://github.com/chen3feng/vllm-analysis.git
```

在 VS Code 中打开，`Cmd/Ctrl + 点击` 代码引用即可跳转到 vLLM 源码对应行。


## 生成方法

这些文档由 AI（Claude Code）通过系统性阅读 vLLM 源码生成。流程：

1. **探索**：围绕具体问题，从入口点跟踪调用链，精确搜索行号
2. **生成**：结构化文档 + ASCII 架构图 + 可点击代码引用
3. **版本锁定**：vLLM 代码作为 git submodule 固定 commit，保证行号永久有效

详见 [README](https://github.com/chen3feng/vllm-analysis/blob/master/README.md)。
