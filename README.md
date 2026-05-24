# vllm-analysis

vLLM 源码深度分析——逐模块拆解 PagedAttention、调度器、KV cache 管理、模型加载、构建系统等核心机制。所有代码引用均为可点击链接，精确到源码行号。

## 快速开始

```bash
git clone --recurse-submodules https://github.com/chen3feng/vllm-analysis.git
cd vllm-analysis
```

源码通过 git submodule 引入，固定在文档生成时的 commit。在 VS Code 中打开，`Cmd/Ctrl + 点击` 代码引用即可跳转到对应行。

## 文档索引

| # | 文档 | 内容 |
|---|------|------|
| 1 | [架构概述](docs/01-架构概述.md) | vLLM 是什么、分层架构、一次请求的完整生命周期、关键子系统、所有文档入口 |
| 2 | [代码结构分析](docs/02-代码结构分析.md) | 顶层目录、CMake/setuptools 构建系统、C++/CUDA/Rust 代码组织、CI 与测试 |
| 3 | [初始化流程分析](docs/03-初始化流程分析.md) | Engine 启动序列：配置解析 → Executor/Worker → 模型加载 → 显存 Profiling → KV cache 分配 → BlockPool → Scheduler |
| 4 | [模型加载流程分析](docs/04-模型加载流程分析.md) | 9 种 ModelLoader、架构解析（600+ 模型注册表）、权重迭代器与 TP 分片、量化后处理 |
| 5 | [推理流程分析](docs/05-推理流程分析.md) | Step 循环：请求提交 → 调度 → 模型执行 → Attention 读写 KV cache → 采样 → 输出处理 → 请求完成 |
| 6 | [算子注册与分发](docs/06-算子注册与分发.md) | Python ↔ C++/CUDA/Triton 的桥梁：三种注册机制、`torch.ops._C.*` vs `torch.ops.vllm.*` vs `@triton.jit`、attention 算子的完整分发路线 |
| 7 | [PagedAttention 实现分析](docs/07-PagedAttention-实现分析.md) | Block Table 机制（纯软件抽象 vs GPU 硬件）、KV cache 内存管理、CUDA/Triton kernel 实现、prefix caching、为何传统引擎没有此机制 |

## 文档特点

- **精确行号**：每个代码引用形如 `[filename.py:123](vllm/path/to/file.py#L123)`，可点击直接跳转
- **可验证**：所有行号基于 submodule 中的固定 commit，不会溯源失效
- **架构图**：ASCII 流程图和调用链，无需外部工具即可阅读
- **表格总结**：每个模块末尾有关键设计决策和行号对照表

## 这些文档是如何生成的

### 探索

让 AI 按照人类分析代码的路径，系统性地阅读代码：

1. **确定分析角度**——每篇文档围绕一个核心问题（"PagedAttention 如何实现？""一次推理经过哪些步骤？"）
2. **逐层探索**——从入口点开始，跟踪调用链；找到关键类和函数后，读取具体实现
3. **回到源头**——文档中的行号都是工具通过 grep/read 精确查找得到的，不是推测

### 生成

探索完成后，AI 整理出结构化的文档：

1. **代码引用可点击**——每个类名、函数名、关键常量后附带 `[file:line](path#L行号)` 格式的链接
2. **架构图**——生成 ASCII 流程图、调用链、层次关系图
3. **交叉引用**——文档间互相链接，形成体系

### 版本锁定

`vllm/` 作为 git submodule，固定在文档生成时的 commit `5bb8d2767`。这保证了：

- 所有行号永久有效
- 查看者可以在 GitHub 上看到**文档引用的那个版本的**源码
- 后续更新时，只需更新 submodule 指针并检查差异

## 更新到新版本

```bash
cd vllm-analysis
cd vllm && git fetch origin && git checkout <new-commit> && cd ..
git add vllm
# 对比新旧版本差异，更新文档
git commit -m "Update vllm submodule to <new-commit>"
```

## 贡献

欢迎提交 PR 修正错误或补充新内容。请确保引用的行号与当前 submodule 版本一致。

## License

MIT
