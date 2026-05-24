---
title: 目录
nav_order: 1
---

# vLLM 源码深度分析

vLLM 是目前最流行的 LLM 推理引擎，以 PagedAttention 技术闻名。这份文档集逐模块拆解其核心机制——从架构全景到 C++/CUDA kernel 细节。

所有代码引用均可点击直达源码对应行号。


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
