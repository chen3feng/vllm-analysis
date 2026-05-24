---
title: FlashAttention 实现分析
nav_order: 8
---

# FlashAttention 在 vLLM 中的实现分析

## 1. 概述

### 1.1 背景：Attention 计算的 IO 瓶颈

Transformer 的标准 attention 计算需要 O(N²) 的内存来存储 attention 矩阵（N 为序列长度）。对于长序列，这个矩阵动辄几十 GB，远超 GPU SRAM 容量，迫使数据在 HBM（高带宽显存）和 SRAM 之间频繁搬运——而 HBM 带宽（~2 TB/s）远低于计算吞吐（~312 TFLOPS），IO 成为瓶颈。

FlashAttention 通过 **tiling + online softmax** 解决了这个问题。核心思想是将 Q、K、V 切分成小块（tile），每次只加载一个小块到 SRAM 中计算局部 softmax，然后通过 online softmax 校正合并——全程避免将完整的 N×N attention 矩阵写入 HBM。

### 1.2 FlashAttention 在 vLLM 中的角色

vLLM 的默认 attention backend 就是 FlashAttention。在 CUDA 平台上，它是优先级最高的后端（SM80+ 架构），负责将 PagedAttention 的 block_table 机制与高效 attention 计算连接起来。此外 vLLM 对上游 FlashAttention 做了多项扩展：

| 扩展 | 说明 |
|------|------|
| **Block table 集成** | `block_table` 参数传入 kernel，实现分页 KV cache 寻址 |
| **Cascade Attention** | 共享前缀只算一次，后缀分别计算后 LSE 加权合并 |
| **AOT 调度器元数据** | FA3 静态调度，支持 CUDA graph 确定性执行 |
| **Attention Sinks** | `s_aux` 参数，可学习的 sink token |
| **DCP** | 跨 GPU 的 KV cache 分片 + 合并 |

### 1.3 FlashAttention 版本演进

| 版本 | GPU 架构 | 关键特性 |
|------|---------|---------|
| **FA2** | SM80+ (Ampere) | tiling + online softmax，基础 paged KV |
| **FA3** | SM90+ (Hopper) | AOT 调度器元数据，`num_splits` 并行，sink |
| **FA4** | SM100+ (Blackwell) | CuteDSL 实现，TMEM 加速，更大 head size |

---

## 2. Backend 架构

### 2.1 类层次

[flash_attn.py:68](../vllm/vllm/v1/attention/backends/flash_attn.py#L68) — `FlashAttentionBackend` 继承 `AttentionBackend`：

```
AttentionBackend                      [backend.py:55]
  └─ FlashAttentionBackend            [flash_attn.py:68]
       ├─ get_impl_cls()   → FlashAttentionImpl
       ├─ get_builder_cls() → FlashAttentionMetadataBuilder
       ├─ get_kv_cache_shape() → (2, num_blocks, block_size, num_kv_heads, head_size)
       └─ supported_dtypes: [float16, bfloat16]
```

| 类 | 行号 | 职责 |
|----|------|------|
| `FlashAttentionBackend` | [L68](../vllm/vllm/v1/attention/backends/flash_attn.py#L68) | 工厂：dtype 检测、KV cache shape、创建 impl/builder |
| `FlashAttentionMetadata` | [L223](../vllm/vllm/v1/attention/backends/flash_attn.py#L223) | 元数据：block_table、slot_mapping、cascade/DCP 参数 |
| `FlashAttentionMetadataBuilder` | [L276](../vllm/vllm/v1/attention/backends/flash_attn.py#L276) | 每 step 构建元数据、CUDA graph 支持、AOT 调度 |
| `FlashAttentionImpl` | [L592](../vllm/vllm/v1/attention/backends/flash_attn.py#L592) | per-layer 实现：forward + KV cache 更新 |

### 2.2 Backend 选择

[vllm/platforms/cuda.py:79](../vllm/vllm/platforms/cuda.py#L79) — CUDA 平台优先级：

```
SM100 (Blackwell): FlashInfer > FlashAttention > Triton > FlexAttention > TurboQuant
SM80+ (其他):      FlashAttention > FlashInfer > Triton > FlexAttention > TurboQuant
```

每个 backend 通过 `validate_configuration()` 验证能力，选第一个验证通过的。

---

## 3. FlashAttention 版本选择

[fa_utils.py:56](../vllm/vllm/v1/attention/backends/fa_utils.py#L56) — `get_flash_attn_version()` 决定每个层使用哪个版本：

```
1. 根据 SM 能力默认:
   SM 9.x (Hopper)      → FA3
   SM 10.x (Blackwell)  → FA4
   其他                  → FA2

2. 用户可通过 attention_config.flash_attn_version 覆盖

3. 降级链条:
   ├─ FA3 on Blackwell → 降为 FA4 或 FA2
   ├─ ALiBi with FA3/4 → 降为 FA2
   ├─ Head size > 256 on SM90 → 升级为 FA4
   ├─ Batch invariance with FA4 → 降为 FA2
   └─ FA4 TMEM 容量限制 → 降为 FA2
```

版本检测在 [flash_attn_interface.py](../vllm/vllm/vllm_flash_attn/flash_attn_interface.py) 中通过 try/except ImportError 实现——三个版本独立编译为 `.so`，按 GPU 能力选择性加载。

---

## 4. forward() 执行流程

[flash_attn.py:667](../vllm/vllm/v1/attention/backends/flash_attn.py#L667) — `FlashAttentionImpl.forward()` 按 attention type 分派：

```
forward(query, key, value, kv_cache, attn_metadata, output)
  │
  ├─ ENCODER / ENCODER_ONLY [L721]
  │   └─ flash_attn_varlen_func(q, k, v)  ← 无 KV cache
  │
  ├─ DCP (Decode Context Parallel) [L776]
  │   └─ _forward_with_dcp() [L885]
  │       ├─ All-Gather Q across ranks
  │       ├─ flash_attn_varlen_func (causal=False) → context attention
  │       ├─ LSE 合并 (AllGather + ReduceScatter)
  │       ├─ flash_attn_varlen_func (causal=True) → query attention
  │       └─ merge_attn_states → 最终输出
  │
  ├─ Cascade Attention [L822]
  │   └─ cascade_attention() [L1132]
  │       ├─ flash_attn_varlen_func (causal=False) → 共享 prefix
  │       ├─ flash_attn_varlen_func (causal=True) → per-query suffix
  │       └─ merge_attn_states → LSE 加权合并
  │
  └─ 标准 decode/prefill [L758]
      └─ key_cache, value_cache = kv_cache.unbind(dim=0)
          flash_attn_varlen_func(
              q, key_cache, value_cache,
              cu_seqlens_q=..., seqused_k=...,
              block_table=...,          ← 分页寻址
              scheduler_metadata=...,  ← FA3 AOT 调度
          )
```

### 4.1 `flash_attn_varlen_func` 分发

[flash_attn_interface.py:176](../vllm/vllm/vllm_flash_attn/flash_attn_interface.py#L176) — 根据 `fa_version` 调用不同的 CUDA kernel：

| 版本 | Kernel 调用 |
|------|-----------|
| FA2 | `torch.ops._vllm_fa2_C.varlen_fwd()` |
| FA3 | `torch.ops._vllm_fa3_C.fwd()` |
| FA4 | CuteDSL `_flash_attn_fwd()` |

### 4.2 KV Cache 更新

[flash_attn.py:850](../vllm/vllm/v1/attention/backends/flash_attn.py#L850) — `do_kv_cache_update()` 调用 `reshape_and_cache_flash()`，将新计算的 key/value 按 `slot_mapping` 散射写入分页 KV cache。底层 CUDA kernel 在 [cache_kernels.cu:304](../csrc/cache_kernels.cu#L304)。

---

## 5. 与 PagedAttention 的集成

FlashAttention 通过 `block_table` 参数原生支持分页 KV cache 寻址。从 kernel 的角度看：

```python
# flash_attn_varlen_func 的参数中
flash_attn_varlen_func(
    ...
    k=k_cache,            # [num_blocks, block_size, num_kv_heads, head_size]
    v=v_cache,            # 同上
    block_table=bt,       # [num_seqs, max_blocks]  int32 页表
    seqused_k=seq_lens,   # 每个请求的实际 KV 长度
)
```

kernel 内部遍历逻辑块时，通过 `block_table[seq_idx][logical_block]` 查表得到物理块号，再算偏移访问 KV cache。这与 PagedAttention 自研 kernel（[attention_kernels.cuh](../csrc/attention/attention_kernels.cuh#L85)）中的 `block_table[block_idx]` 查表机制完全相同。

**PagedAttention kernel vs FlashAttention kernel 的关系**：两者是互补的，不是替代关系。PagedAttention kernel（`paged_attention_v1/v2`）是 vLLM 自研的 CUDA kernel，专为分页 KV cache 设计；FlashAttention 是外部库，vLLM 通过自定义 fork 使其支持 `block_table` 参数。在 CUDA 平台上，FlashAttention 是默认首选——它性能更优；PagedAttention kernel 则作为不具备 FlashAttention 的环境中的后备方案。

---

## 6. Cascade Attention

### 6.1 场景

当所有请求共享同一个 system prompt（如 Chatbot Arena 的对话前缀），可以将 KV cache 分为两块：共享前缀块 + 每个请求独有的后缀块。前缀只算一次，后缀各算各的。

### 6.2 决策逻辑

[flash_attn.py:1054](../vllm/vllm/v1/attention/backends/flash_attn.py#L1054) — `use_cascade_attention()` 的启发式条件：

```
1. common_prefix_len >= 256          ← 前缀够长才值得
2. 无 ALiBi / sliding window / local attention
3. batch 中 >= 8 个请求
4. DCP 未启用
5. 若 FlashDecoding 未启用 → 直接用 cascade
6. 若 FlashDecoding 启用 → 性能模型估算:
     cascade_ctas vs flash_decoding_ctas → 选时间短的
```

### 6.3 执行

[flash_attn.py:1132](../vllm/vllm/v1/attention/backends/flash_attn.py#L1132) — `cascade_attention()` 的三步：

```
1. Prefix pass: flash_attn_varlen_func(causal=False)
   → 所有 query 对 block_table[:1] 计算 attention
   → 输出 LSE (log-sum-exp)

2. Suffix pass: flash_attn_varlen_func(causal=True)
   → 每个 query 对 block_table[:, num_common_kv_blocks:] 计算

3. Merge: merge_attn_states(prefix_out, suffix_out, prefix_lse, suffix_lse)
   → LSE 加权合并 → 最终输出
```

`merge_attn_states` 有两个实现：
- CUDA kernel：[merge_attn_states.cu](../csrc/attention/merge_attn_states.cu)
- Triton fallback：[triton_merge_attn_states.py](../vllm/vllm/v1/attention/ops/triton_merge_attn_states.py)

---

## 7. AOT 调度器元数据（FA3）

FA3 引入了 Ahead-of-Time 调度。传统 FlashAttention 在 kernel 启动时动态决定 tile 分配；FA3 预先计算调度方案，使 kernel 行为确定化，从而支持 CUDA graph 捕获。

[flash_attn.py:414-428](../vllm/vllm/v1/attention/backends/flash_attn.py#L414) — MetadataBuilder 中 AOT 调度的关键：

```python
# 调用 FA3 的调度器元数据生成
scheduler_metadata = get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k,
    num_heads_q, num_heads_kv, head_size,
    cache_seqlens=seq_lens,
    page_size=block_size,
    ...
)
```

**CUDA graph 兼容性**：[flash_attn.py:295](../vllm/vllm/v1/attention/backends/flash_attn.py#L295) — FA3 支持 `ALWAYS` 级别的 CUDA graph，FA2 仅支持 `UNIFORM_BATCH`（batch 大小固定）。

---

## 8. 自定义 CUDA Kernel

### 8.1 外部 FlashAttention 库

vLLM 不自己维护 FlashAttention CUDA kernel。kernel 代码从 `https://github.com/vllm-project/flash-attention.git` 通过 CMake `FetchContent` 拉取（[vllm_flash_attn.cmake](../cmake/external_projects/vllm_flash_attn.cmake)）。vLLM 维护自己的 fork，添加了 block_table、s_aux、AOT 调度等参数。

编译为三个独立的 `.so`：
- `vllm_flash_attn/_vllm_fa2_C.abi3.so`
- `vllm_flash_attn/_vllm_fa3_C.abi3.so`
- `vllm_flash_attn/_vllm_fa4_C.abi3.so`（CuteDSL）

### 8.2 vLLM 自研的辅助 kernel

这些是 vLLM 自己写的，不在上游 FlashAttention 中：

| Kernel | 文件 | 用途 |
|--------|------|------|
| `reshape_and_cache_flash` | [cache_kernels.cu:304](../csrc/cache_kernels.cu#L304) | 散射写入 KV cache |
| `merge_attn_states` | [merge_attn_states.cu](../csrc/attention/merge_attn_states.cu) | LSE 加权合并（cascade/DCP） |

---

## 9. DCP（Decode Context Parallel）

DCP 将 KV cache 分片到多张 GPU 上，每张 GPU 持有一部分 KV block。decode 时：

```
1. All-Gather: 收集所有 GPU 的 query
2. Context attention: 每张 GPU 对本地的 KV shard 计算 attention (causal=False)
3. LSE merge: AllGather + ReduceScatter 合并各 GPU 的部分结果
4. Query attention: 对当前 step 的 KV 计算 attention (causal=True)
5. merge_attn_states: 合并 context 和 query 结果
```

实现在 [flash_attn.py:885](../vllm/vllm/v1/attention/backends/flash_attn.py#L885) `_forward_with_dcp()`。

---

## 10. 与其他 Backend 的对比

| | FlashAttention | Triton | PagedAttention Kernel |
|---|---|---|---|
| **实现** | 外部 CUDA (vllm fork) | 纯 Triton | vLLM 自研 CUDA |
| **FA 版本** | FA2/FA3/FA4 | N/A | N/A |
| **Cascade** | ✓ | ✗ | ✗ |
| **DCP** | ✓ | ✗ | ✗ |
| **CUDA Graph** | FA3=ALWAYS / FA2=UNIFORM_BATCH | ALWAYS | 取决于版本 |
| **Sinks** | FA3/FA4 | ✓ | ✗ |
| **FP8 KV cache** | ✓ | ✓ | ✓ |
| **Head size** | ≤256 (FA2/3), ≤512 (FA4) | ≥32 | 32-256 |
| **平台** | CUDA | CUDA/ROCm/XPU | CUDA/ROCm/CPU |
| **依赖** | 需要编译 flash-attn | 无外部依赖 | 无外部依赖 |

---

## 11. 关键文件索引

| 组件 | 文件 | 核心行号 |
|------|------|---------|
| FlashAttentionBackend | [flash_attn.py](../vllm/vllm/v1/attention/backends/flash_attn.py) | [L68](../vllm/vllm/v1/attention/backends/flash_attn.py#L68) (Backend), [L223](../vllm/vllm/v1/attention/backends/flash_attn.py#L223) (Metadata), [L276](../vllm/vllm/v1/attention/backends/flash_attn.py#L276) (Builder), [L592](../vllm/vllm/v1/attention/backends/flash_attn.py#L592) (Impl) |
| forward() | 同上 | [L667](../vllm/vllm/v1/attention/backends/flash_attn.py#L667) |
| do_kv_cache_update() | 同上 | [L850](../vllm/vllm/v1/attention/backends/flash_attn.py#L850) |
| cascade_attention() | 同上 | [L1132](../vllm/vllm/v1/attention/backends/flash_attn.py#L1132) |
| use_cascade_attention() | 同上 | [L1054](../vllm/vllm/v1/attention/backends/flash_attn.py#L1054) |
| _forward_with_dcp() | 同上 | [L885](../vllm/vllm/v1/attention/backends/flash_attn.py#L885) |
| 版本选择 | [fa_utils.py](../vllm/vllm/v1/attention/backends/fa_utils.py) | [L56](../vllm/vllm/v1/attention/backends/fa_utils.py#L56) |
| Kernel 分发 | [flash_attn_interface.py](../vllm/vllm/vllm_flash_attn/flash_attn_interface.py) | [L176](../vllm/vllm/vllm_flash_attn/flash_attn_interface.py#L176) |
| reshape_and_cache_flash | [cache_kernels.cu](../csrc/cache_kernels.cu) | [L304](../csrc/cache_kernels.cu#L304) |
| merge_attn_states | [merge_attn_states.cu](../csrc/attention/merge_attn_states.cu) | — |
| CMake 外部项目 | [vllm_flash_attn.cmake](../cmake/external_projects/vllm_flash_attn.cmake) | — |
| Backend 选择 (CUDA) | [cuda.py](../vllm/vllm/platforms/cuda.py) | [L79](../vllm/vllm/platforms/cuda.py#L79) |
