---
title: PagedAttention 实现分析
nav_order: 8
---

# PagedAttention 在 vLLM 中的实现分析

## 1. 概述

PagedAttention 是 vLLM 的核心创新（Kwon et al., SOSP 2023），其关键思想是将 KV cache 划分为**固定大小的 block（页）**，类似于操作系统的虚拟内存分页。这带来了三个核心优势：

- **零内存浪费**：KV cache 可以非连续分配，消除内部碎片
- **Prefix 共享**：多个请求共享相同的物理 block（通过引用计数），避免重复计算
- **按需分配**：仅为实际计算的 token 分配 block，支持动态调度

---

## 2. 整体架构

```
请求到达
  │
  ├─ KVCacheManager ────────── 分配物理 block，管理 prefix cache
  │     ├─ KVCacheCoordinator ─ 协调多种 attention type
  │     └─ SingleTypeKVCacheManager ─ 单一 attention type 的 block 管理
  │
  ├─ AttentionMetadataBuilder ─ 构建 block_table、slot_mapping 等元数据
  │
  └─ AttentionImpl.forward() ── 执行注意力计算
        └─ CUDA/Triton kernel ─ 通过 block_table 间接寻址 KV cache
```

核心代码路径：

| 模块 | 路径 |
|------|------|
| Backend 注册与选择 | [registry.py](vllm/v1/attention/backends/registry.py#L34), [selector.py](vllm/v1/attention/selector.py#L52) |
| 后端抽象类 | [backend.py](vllm/v1/attention/backend.py#L55) |
| FlashAttention 后端（主路径） | [flash_attn.py](vllm/v1/attention/backends/flash_attn.py#L68) |
| Triton 后端 | [triton_attn.py](vllm/v1/attention/backends/triton_attn.py#L271) |
| Triton decode kernel | [triton_decode_attention.py](vllm/v1/attention/ops/triton_decode_attention.py#L729) |
| Triton prefill kernel | [triton_prefill_attention.py](vllm/v1/attention/ops/triton_prefill_attention.py#L191) |
| Triton unified kernel | [triton_unified_attention.py](vllm/v1/attention/ops/triton_unified_attention.py#L763) |
| CUDA kernel V1 | [paged_attention_v1.cu](csrc/attention/paged_attention_v1.cu#L160) |
| CUDA kernel V2 | [paged_attention_v2.cu](csrc/attention/paged_attention_v2.cu#L27) |
| CUDA kernel 核心算法 | [attention_kernels.cuh](csrc/attention/attention_kernels.cuh#L85) |
| Cache CUDA kernel | [cache_kernels.cu](csrc/cache_kernels.cu#L244) |
| PagedAttention ops | [paged_attn.py](vllm/v1/attention/ops/paged_attn.py#L15) |
| Block 池 | [block_pool.py](vllm/v1/core/block_pool.py#L130) |
| KV Cache 工具 | [kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py#L116) |
| KV Cache 管理器 | [kv_cache_manager.py](vllm/v1/core/kv_cache_manager.py#L110) |
| 单一类型管理器 | [single_type_kv_cache_manager.py](vllm/v1/core/single_type_kv_cache_manager.py#L31) |
| 协调器 | [kv_cache_coordinator.py](vllm/v1/core/kv_cache_coordinator.py#L276) |

---

## 3. Block Table 机制 —— 分页的核心

### 3.1 概念

每个请求维护一个 **block table**（页表），形状为 `[num_seqs, max_num_blocks_per_seq]`，将**逻辑 block 序号**映射到**物理 block 序号**。

KV cache 的寻址路径：

```
逻辑 token 位置
  │
  ├─ 计算: logical_block = token_idx // block_size
  │         offset = token_idx % block_size
  │
  ├─ 查表: physical_block = block_table[seq_idx][logical_block]
  │
  └─ 访问: kv_cache[physical_block * block_size_stride + offset]
```

### 3.2 Slot Mapping

每个 token 被映射为一个全局 slot：

```python
slot = block_number * block_size + block_offset
```

- 正数 slot：有效的 token 位置
- 负数 slot：padding token（被忽略）

### 3.3 KV Cache 布局

v1 采用紧凑布局：

```
shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        ↑
    key / value 分离

或者 HND 布局（head-major）：
shape: [2, num_blocks, num_kv_heads, block_size, head_size]
```

对比 v0 的 legacy 布局（用于 `paged_attention_v1/v2` kernel）：

```
key_cache:   [num_blocks, num_kv_heads, head_size//x, block_size, x]
value_cache: [num_blocks, num_kv_heads, head_size, block_size]
              其中 x = 16 / sizeof(element)，用于 16 字节对齐读取
```

### 3.4 Block Table 的本质：纯软件抽象，不是 GPU 硬件支持

#### 3.4.1 Kernel 中的地址翻译是纯手工完成的

`block_table` 在 CUDA kernel 层面就是一个普通的 **`const int*`** 数组，存放在 GPU 全局内存中。地址翻译的全过程只有三步普通的 GPU 指令：

```c++
// attention_kernels.cuh:202 — 第一步：拿到当前序列的 block_table 指针
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// attention_kernels.cuh:252-253 — 第二步：软件数组查表，拿到物理 block 号
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);

// attention_kernels.cuh:268-270 — 第三步：手工乘加，算出物理地址
const cache_t* k_ptr =
    k_cache + physical_block_number * kv_block_stride +
    kv_head_idx * kv_head_stride + physical_block_offset * x;
```

不存在以下任何 GPU 硬件机制：

- **没有 GPU MMU 参与** — GPU 的页表走查（page table walk）仅用于 CUDA 虚拟地址空间管理，不会参与 `k_cache + offset` 这种指针算术
- **没有 TLB** — 地址翻译不是通过 TLB 查询完成的，而是 ALU 的整数乘法
- **没有纹理单元** — 整个 `csrc/` 目录中不存在 `tex1Dfetch`、`cudaTextureObject_t` 等纹理硬件的使用
- **没有 surface object** — 同样不存在任何 `surf2Dwrite` 之类的用法

这在所有计算平台上是完全一致的 —— CUDA kernel（[attention_kernels.cuh](csrc/attention/attention_kernels.cuh#L253)）、ROCm kernel（[rocm/attention.cu](csrc/rocm/attention.cu#L438)）、CPU kernel（[cpu/cpu_attn_impl.hpp](csrc/cpu/cpu_attn_impl.hpp#L963)）全都是同样的 `block_table[block_idx]` 软件查表模式。

#### 3.4.2 类比：这和 OS 虚拟内存的类比仅停留在概念层面

PagedAttention 借用了 OS 虚拟内存的**元语和命名**，但两者的实现机制完全不同：

| | OS 虚拟内存 | PagedAttention |
|---|---|---|
| **页表** | 硬件页表（多级树，MMU 走查） | GPU 全局内存中的 `int32` 数组 |
| **地址翻译** | 硬件 TLB + 页表走查，每个 load/store 自动触发 | kernel 软件显式查表 + 整数乘加 |
| **页大小** | 4KB / 2MB / 1GB（硬件定义） | 16/32 tokens（应用层可配） |
| **缺页** | 硬件 page fault → OS 处理 | 不存在缺页概念；block 未分配时 token 尚未生成 |
| **换页** | OS 换出到磁盘 | 不支持 swap；KV cache 全量驻留 GPU 显存 |
| **隔离** | 硬件保护（user/kernel mode） | 纯软件隔离（不同请求访问不同的 block_table 行） |
| **TLB shootdown** | 硬件/OS 处理 | 不适用；无 TLB，无同步问题 |

关键的差异在于：**OS 用虚拟内存是为了"欺骗"进程（让每个进程以为自己有独立的大地址空间）；PagedAttention 用 block table 是为了消除 KV cache 的内部碎片和实现 prefix 共享。** 两者的动机不同，实现层次也不同。

#### 3.4.3 那 cuMemCreate/cuMemMap 是干什么的？

代码库中确实有 CUDA Virtual Memory API 的使用，位于 [cumem_allocator.cpp](csrc/cumem_allocator.cpp)，但它的角色和 block table 截然不同：

```
CUDA Virtual Memory API (cuMemCreate / cuMemMap / cuMemSetAccess)
└── cumem_allocator.cpp
    └── 作为 torch CUDAPluggableAllocator
        └── 管理整个 KV cache tensor 的底层物理内存分配
            （替代 cudaMalloc，提供更精细的显存控制）

PagedAttention block_table
└── attention_kernels.cuh
    └── 在已分配好的 KV cache tensor 内部
        └── 用 int32 数组做逻辑 block → 物理 block 的映射
            （纯软件，kernel 内手工地址计算）
```

这两个是完全解耦的：底层分配器用不用虚拟内存 API，block_table 机制都不受影响。**PagedAttention 的"分页地址翻译"发生在 kernel 内部的指针算术中，而不是 GPU 的 MMU 层级上。**

---

## 4. KV Cache 内存管理

### 4.1 数据结构

#### KVCacheBlock — [kv_cache_utils.py:116](vllm/v1/core/kv_cache_utils.py#L116)

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int           # 物理 block 编号 (0..num_gpu_blocks-1)
    ref_cnt: int = 0        # 引用计数（用于 prefix 共享）
    _block_hash: ...        # 链式哈希（用于 prefix caching 查找）
    prev_free_block: ...    # 双向链表（空闲队列）
    next_free_block: ...
    is_null: bool = False   # 空占位
```

#### FreeKVCacheBlockQueue — [kv_cache_utils.py:164](vllm/v1/core/kv_cache_utils.py#L164)

所有空闲 block 组成**双向链表**，关键操作均为 O(1)：
- `popleft()` / `popleft_n(n)` — FIFO 分配（缓存使能时实现 LRU 淘汰）
- `append(block)` / `append_n(blocks)` — 归还到队尾
- `remove(block)` — 从中间移除（用于 prefix cache 命中时增加引用计数）

#### BlockHashToBlockMap

哈希表：`(block_hash, group_id) → KVCacheBlock`，用于 prefix caching 的快速查找。成员变量在 [block_pool.py:171](vllm/v1/core/block_pool.py#L171) 初始化。

### 4.2 分配流程 — [kv_cache_manager.py:110](vllm/v1/core/kv_cache_manager.py#L110)

```
新请求到达
  │
  ├─ 1. 计算 token hashes: hash((parent_hash, token_ids, extra_keys))
  │      └─ hash_block_tokens() 位于 [kv_cache_utils.py:541](vllm/v1/core/kv_cache_utils.py#L541)
  │
  ├─ 2. find_longest_cache_hit() — [single_type_kv_cache_manager.py:373](vllm/v1/core/single_type_kv_cache_manager.py#L373)
  │     ├─ 从左到右扫描 logical block
  │     ├─ 查找 BlockHashToBlockMap
  │     └─ 找到最长连续匹配 → 命中 N 个 block
  │
  ├─ 3. touch(命中 block) — [block_pool.py:402](vllm/v1/core/block_pool.py#L402)
  │     └─ ref_cnt += 1（共享引用）
  │
  ├─ 4. allocate_new_blocks(num_new_blocks) — [single_type_kv_cache_manager.py:243](vllm/v1/core/single_type_kv_cache_manager.py#L243)
  │     ├─ 从 FreeKVCacheBlockQueue popleft()
  │     └─ 记录新 block 的 slot_mapping
  │
  ├─ 5. cache_blocks(新 block) — [single_type_kv_cache_manager.py:278](vllm/v1/core/single_type_kv_cache_manager.py#L278)
  │     └─ 将 block hash 写入 BlockHashToBlockMap
  │
  └─ 6. 返回 KVCacheBlocks（包含 block_table + 新分配信息）
```

请求结束时：

```
free(request) — [kv_cache_manager.py:429](vllm/v1/core/kv_cache_manager.py#L429)
  ├─ ref_cnt -= 1（递减引用计数）
  ├─ 若 ref_cnt == 0 → 归还到 FreeKVCacheBlockQueue
  └─ 若 block 在缓存哈希表中 → 从哈希表移除
```

### 4.3 Prefix Caching 的链式哈希 — [kv_cache_utils.py:541](vllm/v1/core/kv_cache_utils.py#L541)

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    return hash((parent_hash, token_ids, extra_keys))
```

`extra_keys` 包含：LoRA 名称、多模态特征哈希、cache salt、prompt embed 哈希，保证不同配置下的缓存隔离。

### 4.4 Attention Spec 类型 — [kv_cache_interface.py](vllm/v1/kv_cache_interface.py)

| Spec | 行号 | 用途 | 特殊行为 |
|------|------|------|---------|
| `FullAttentionSpec` | [L188](vllm/v1/kv_cache_interface.py#L188) | 标准全注意力 | 请求结束前持有所有 block |
| `SlidingWindowSpec` | [L435](vllm/v1/kv_cache_interface.py#L435) | 滑动窗口注意力 | 窗口外的旧 block 自动释放 |
| `ChunkedLocalAttentionSpec` | [L407](vllm/v1/kv_cache_interface.py#L407) | 分块局部注意力 | 仅保留当前 chunk 的 block |
| `SinkFullAttentionSpec` | [L615](vllm/v1/kv_cache_interface.py#L615) | 带 sink token 的注意力 | sink block 固定不释放 |
| `MLAAttentionSpec` | [L337](vllm/v1/kv_cache_interface.py#L337) | DeepSeek MLA | 类似全注意力 |
| `CrossAttentionSpec` | [L602](vllm/v1/kv_cache_interface.py#L602) | 编码器-解码器交叉注意力 | 不支持 prefix caching |
| `MambaSpec` | [L563](vllm/v1/kv_cache_interface.py#L563) | Mamba SSM 状态缓存 | 特殊的 per-request state block |
| `SlidingWindowMLASpec` | [L498](vllm/v1/kv_cache_interface.py#L498) | MLA + 滑动窗口 | DeepSeek V4 混合模式 |
| `HiddenStateCacheSpec` | [L400](vllm/v1/kv_cache_interface.py#L400) | Hidden state 缓存 | 用于 extract_hidden_states |

对应的管理器（[single_type_kv_cache_manager.py](vllm/v1/core/single_type_kv_cache_manager.py)）：

| Manager | 行号 | 对应的 Spec |
|---------|------|------------|
| `SingleTypeKVCacheManager` (基类) | [L31](vllm/v1/core/single_type_kv_cache_manager.py#L31) | — |
| `FullAttentionManager` | [L481](vllm/v1/core/single_type_kv_cache_manager.py#L481) | FullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec |
| `SlidingWindowManager` | [L542](vllm/v1/core/single_type_kv_cache_manager.py#L542) | SlidingWindowSpec, SlidingWindowMLASpec |
| `ChunkedLocalAttentionManager` | [L692](vllm/v1/core/single_type_kv_cache_manager.py#L692) | ChunkedLocalAttentionSpec |
| `MambaManager` | [L842](vllm/v1/core/single_type_kv_cache_manager.py#L842) | MambaSpec |
| `CrossAttentionManager` | [L1122](vllm/v1/core/single_type_kv_cache_manager.py#L1122) | CrossAttentionSpec |
| `SinkFullAttentionManager` | [L1176](vllm/v1/core/single_type_kv_cache_manager.py#L1176) | SinkFullAttentionSpec |

### 4.5 多种 Attention Type 的协调 — [kv_cache_coordinator.py](vllm/v1/core/kv_cache_coordinator.py)

| 协调器 | 行号 | 说明 |
|--------|------|------|
| `KVCacheCoordinatorNoPrefixCache` | [L276](vllm/v1/core/kv_cache_coordinator.py#L276) | 禁用 prefix caching，直接返回空命中 |
| `UnitaryKVCacheCoordinator` | [L324](vllm/v1/core/kv_cache_coordinator.py#L324) | 单一 attention type，直接委托 |
| `HybridKVCacheCoordinator` | [L392](vllm/v1/core/kv_cache_coordinator.py#L392) | 多种 attention type，固定点迭代协调 |

`HybridKVCacheCoordinator` 处理一个模型存在多种 attention type（如 sliding window + full attention 混合）的情况：

1. 各 type 的 manager 独立计算 cache hit
2. 若任何 type 缩短了 hit 长度，触发**固定点迭代**重新计算
3. 最终 hit 长度必须是所有 type 的 block size 的 LCM 的倍数
4. 使用 [BlockHashListWithBlockSize](vllm/v1/core/kv_cache_utils.py#L2079) 在不同 block size 粒度间转换哈希

### 4.6 Block 大小与内存计算

```python
page_size_bytes = block_size * num_kv_heads * head_size * dtype_size * 2
# ×2 是因为 key + value 各一份
num_blocks = available_memory // page_size // num_layers
```

常见 block size：
- **16**：默认值，平衡灵活性和粒度
- **32**：更大的 block，减少 block table 开销
- FlashAttention 要求 16 的倍数
- Triton 某些配置支持 `[16, 32, 64]`

---

## 5. Backend 体系

### 5.1 Backend 注册 — [registry.py:34](vllm/v1/attention/backends/registry.py#L34)

通过 `AttentionBackendEnum` 枚举注册：

```python
class AttentionBackendEnum(enum.Enum):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    # ... 还有 MLA 变体
```

### 5.2 Backend 选择 — [selector.py:52](vllm/v1/attention/selector.py#L52)

`get_attn_backend()` 根据以下因素选择后端：

- `head_size`, `dtype`, `kv_cache_dtype`, `block_size`
- `use_mla`（DeepSeek MLA 模型）
- `has_sink`, `use_sparse`
- `attn_type`, `use_non_causal`, `use_batch_invariant`

| Backend | 行号 | 特点 |
|---------|------|------|
| **FlashAttention** | [flash_attn.py:68](vllm/v1/attention/backends/flash_attn.py#L68) | 主路径，性能最优，原生支持 block table 和 cascade attention |
| **Triton** | [triton_attn.py:271](vllm/v1/attention/backends/triton_attn.py#L271) | 纯 Triton 实现，支持 ALiBi、sliding window、FP8 量化、block sparsity |
| **FlashInfer** | — | 第三方 FlashInfer 库 |
| **FlexAttention** | — | PyTorch 原生 flex attention |
| **ROCm** | — | AMD GPU 专用 |
| **CPU** | — | CPU 后端 |

### 5.3 抽象类设计 — [backend.py](vllm/v1/attention/backend.py)

| 抽象类 | 行号 | 职责 |
|--------|------|------|
| `AttentionBackend` | [L55](vllm/v1/attention/backend.py#L55) | 静态方法：能力检测、KV cache shape、支持的 block size / head size / dtype |
| `AttentionMetadataBuilder` | [L516](vllm/v1/attention/backend.py#L516) | 构建 per-step 元数据（block_table, slot_mapping 等） |
| `AttentionImpl` | [L763](vllm/v1/attention/backend.py#L763) | per-layer 实现：`forward()` 和 `do_kv_cache_update()` |
| `CommonAttentionMetadata` | [L353](vllm/v1/attention/backend.py#L353) | 所有 backend 共享的元数据 |

```python
class AttentionBackend:
    """静态方法，能力检测"""
    get_name() → str
    get_impl_cls() → AttentionImpl
    get_builder_cls() → AttentionMetadataBuilder
    get_kv_cache_shape(num_blocks, block_size, ...) → tuple
    get_supported_kernel_block_sizes() → list
    validate_configuration() → None

class AttentionMetadataBuilder:
    """构建 per-step 元数据（block_table, slot_mapping 等）"""
    build(common_prefix_len, common_attn_metadata) → metadata

class AttentionImpl:
    """per-layer 实现"""
    forward(layer, query, key, value, kv_cache, attn_metadata, output)
    do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
```

### 5.4 CommonAttentionMetadata — [backend.py:353](vllm/v1/attention/backend.py#L353)

所有 backend 共享的元数据：

```python
class CommonAttentionMetadata:
    query_start_loc      # 每个请求的 query 起始位置（varlen）
    seq_lens             # 每个请求的序列长度
    num_reqs             # 当前 batch 中的请求数
    num_actual_tokens    # 实际 token 数（不含 padding）
    block_table_tensor   # [num_seqs, max_num_blocks] 页表
    slot_mapping         # 每个 token 对应的缓存 slot
    max_query_len        # 最大 query 长度（decode 时为 1）
    max_seq_len          # 最大序列长度
```

---

## 6. FlashAttention 后端（主路径）

路径：[flash_attn.py](vllm/v1/attention/backends/flash_attn.py)

### 6.1 类总览

| 类 | 行号 | 职责 |
|----|------|------|
| `FlashAttentionBackend` | [L68](vllm/v1/attention/backends/flash_attn.py#L68) | Backend 入口，能力检测 |
| `FlashAttentionMetadata` | [L223](vllm/v1/attention/backends/flash_attn.py#L223) | 包含 block_table、slot_mapping、seq_lens 等 |
| `FlashAttentionMetadataBuilder` | [L276](vllm/v1/attention/backends/flash_attn.py#L276) | 构建元数据，支持 cascade attention、AOT 调度 |
| `FlashAttentionImpl` | [L592](vllm/v1/attention/backends/flash_attn.py#L592) | per-layer 实现 |

### 6.2 核心 forward 流程 — [flash_attn.py:667](vllm/v1/attention/backends/flash_attn.py#L667)

```python
class FlashAttentionImpl.forward():
    key_cache, value_cache = kv_cache.unbind(dim=0)
    # key_cache: [num_blocks, block_size, num_kv_heads, head_size]
    # value_cache: [num_blocks, block_size, num_kv_heads, head_size]

    # 使用 flash_attn_varlen_func（原生支持 block table）
    output = flash_attn_varlen_func(
        query, key_cache, value_cache,
        cu_seqlens_q=query_start_loc,   # varlen 索引
        max_seqlen_q=max_query_len,
        seqlen_k=seq_lens,
        block_table=block_table,         # ← 分页的关键参数
        softmax_scale=scale,
        causal=True,
        ...
    )
```

FlashAttention 库原生支持 `block_table` 参数，在 kernel 内部完成逻辑→物理 block 的间接寻址。

### 6.3 KV Cache 更新 — [flash_attn.py:850](vllm/v1/attention/backends/flash_attn.py#L850)

```python
class FlashAttentionImpl.do_kv_cache_update():
    # 将新计算的 key/value 散射写入 paged cache
    reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping,    # 每个 token 的目标 slot
        kv_cache_dtype,  # FP8 量化时会做格式转换
        k_scale, v_scale
    )
```

### 6.4 Cascade Attention

当一个 batch 中所有请求共享相同前缀时：

```
共享 prefix                    各请求独有的 suffix
┌─────────────────────┐  ┌──────┐ ┌──────┐ ┌──────┐
│ non-causal attention │  │ req0 │ │ req1 │ │ req2 │
│ (只算一次)            │  │causal│ │causal│ │causal│
└─────────────────────┘  └──────┘ └──────┘ └──────┘
         │                    │        │        │
         └────── merge ───────┴────────┴────────┘
              (online softmax 校正)
```

关键逻辑：
1. `get_num_common_prefix_blocks()` — [kv_cache_manager.py:476](vllm/v1/core/kv_cache_manager.py#L476) 计算所有请求的最长公共前缀
2. Prefix 部分用 `causal=False` 计算一次
3. 各请求的 suffix 用 `causal=True` 分别计算
4. `merge_attn_states()` 用 online softmax 公式合并两部分

---

## 7. CUDA Kernel 实现

### 7.1 Grid 组织

```cuda
kernel<<<grid(num_heads, num_seqs), block(NUM_THREADS=128)>>>(...)
```

每个 thread block 处理 **一个 head × 一个 sequence**。128 个线程，按 warp (32 threads) 组织。

### 7.2 核心算法 — [attention_kernels.cuh:85](csrc/attention/attention_kernels.cuh#L85)

`paged_attention_kernel()` 是 V1 使用的 device 函数，五个阶段：

```
┌─────────────────────────────────────────────────┐
│ 1. 加载 Query 到共享内存                         │
│    q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD]│
├─────────────────────────────────────────────────┤
│ 2. QK 点积                                      │
│    for each logical block:                      │
│      physical_block = block_table[block_idx]  ← 分页的核心！│
│      key = k_cache[physical_block * stride + offset]        │
│      scores = Q @ K^T                            │
│      (可选) + ALiBi bias                        │
├─────────────────────────────────────────────────┤
│ 3. Online Softmax                               │
│    - warp shuffle 归约 → qk_max                 │
│    - 跨 warp 归约 → 全局 qk_max                 │
│    - exp_sum 跨所有 token 求和                   │
│    - softmax = exp(score - max) / sum           │
├─────────────────────────────────────────────────┤
│ 4. Value 累加                                   │
│    for each token:                              │
│      v = v_cache[physical_block * v_stride + ...]│
│      acc += softmax(token) * v                  │
├─────────────────────────────────────────────────┤
│ 5. 输出                                         │
│    - warp 内归约                                │
│    - 跨 warp 树形归约（shared memory）           │
│    - 写入 output[num_seqs, num_heads, head_size] │
└─────────────────────────────────────────────────┘
```

核心的 QK 点积使用 [Qk_dot](csrc/attention/attention_utils.cuh#L50) 模板结构体，在 [attention_kernels.cuh:289](csrc/attention/attention_kernels.cuh#L289) 处调用。

### 7.3 V1 vs V2 Kernel

| | V1 | V2 |
|---|---|---|
| **入口** | [paged_attention_v1.cu:160](csrc/attention/paged_attention_v1.cu#L160) | [paged_attention_v2.cu](csrc/attention/paged_attention_v2.cu) |
| **Kernel 定义** | [attention_kernels.cuh:85](csrc/attention/attention_kernels.cuh#L85) (device 函数)<br>[attention_kernels.cuh:497](csrc/attention/attention_kernels.cuh#L497) (global kernel) | [attention_kernels.cuh:529](csrc/attention/attention_kernels.cuh#L529) (forward)<br>[attention_kernels.cuh:562](csrc/attention/attention_kernels.cuh#L562) (reduce) |
| **适用场景** | 中短序列 | 长序列 |
| **策略** | 单 kernel，串行遍历所有 block | partition 并行 + reduce 合并 |
| **Partition 大小** | N/A | 512 tokens |
| **Grid** | `(num_heads, num_seqs)` | `(num_heads, num_seqs, max_num_partitions)` |
| **合并方式** | N/A | online softmax 校正 |

V2 的优势：对于超长上下文，将 KV cache 分成多个 partition 并行计算，每个 partition 独立做 attention，最后通过 rescale + online softmax 校正合并。

### 7.4 KV Cache 写入 Kernel — [cache_kernels.cu](csrc/cache_kernels.cu)

| Kernel | 行号 | 说明 |
|--------|------|------|
| `reshape_and_cache_kernel` | [L244](csrc/cache_kernels.cu#L244) | 基础版本：`[num_blocks, num_heads, head_size//x, block_size, x]` 布局 |
| `reshape_and_cache_flash_kernel` | [L304](csrc/cache_kernels.cu#L304) | 灵活版本：支持 NHD/HND 布局 + FP8 量化 |

```cuda
// reshape_and_cache_kernel
// key_cache: [num_blocks, num_heads, head_size//x, block_size, x]
// value_cache: [num_blocks, num_heads, head_size, block_size]
// 其中 x = 16 / sizeof(element), 16 bytes per memory transaction

for each token:
    slot = slot_mapping[token_idx]
    if slot >= 0:  // 跳过 padding
        write key[token] → key_cache[slot]
        write value[token] → value_cache[slot]
```

---

## 8. Triton 实现

路径：[vllm/v1/attention/ops/](vllm/v1/attention/ops/)

| Kernel | 行号 | 用途 |
|--------|------|------|
| `decode_attention_fwd` | [triton_decode_attention.py:729](vllm/v1/attention/ops/triton_decode_attention.py#L729) | 自回归 decode 阶段（1 query token） |
| `context_attention_fwd` | [triton_prefill_attention.py:191](vllm/v1/attention/ops/triton_prefill_attention.py#L191) | 预填充阶段（多 query tokens） |
| `unified_attention` | [triton_unified_attention.py:763](vllm/v1/attention/ops/triton_unified_attention.py#L763) | 统一的 prefill + decode kernel |
| `triton_reshape_and_cache_flash` | [triton_reshape_and_cache_flash.py:319](vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L319) | KV cache 写入 |

额外功能：
- ALiBi 位置编码
- Sliding window 掩码
- Attention sink tokens
- FP8 KV cache 量化
- Block sparsity 支持

---

## 9. PagedAttention Ops — [paged_attn.py:15](vllm/v1/attention/ops/paged_attn.py#L15)

`PagedAttention` 类提供两个关键静态方法：

| 方法 | 行号 | 功能 |
|------|------|------|
| `split_kv_cache` | [L17](vllm/v1/attention/ops/paged_attn.py#L17) | 将 flat KV cache tensor 拆分为 key/value cache 视图（v0 布局） |
| `write_to_paged_cache` | [L32](vllm/v1/attention/ops/paged_attn.py#L32) | 调用 `ops.reshape_and_cache()` 散射写入 KV 数据 |

---

## 10. 补充特性

### 10.1 Decode Context Parallelism (DCP)

将 KV cache 分布到多张 GPU 上，每张 GPU 持有 KV cache 的一个不重叠分区。配合 `paged_attention_v2` 的 partition 机制，支持超长 context 的 decode。

### 10.2 Batch Invariance 模式

强制 `num_splits=1`，保证相同输入在多次运行中产生完全相同的输出（用于 CUDA graph 捕获）。

### 10.3 AOT Scheduling (FlashAttention 3)

预计算 FlashAttention 3 的调度器元数据，避免运行时开销。

---

## 11. 传统的推理引擎为什么没有这种机制

### 11.1 传统做法的设计假设

HuggingFace Transformers、FasterTransformer、TensorRT-LLM（早期版本）等引擎，在处理 KV cache 时共享同一个设计前提：

```
每个请求 = 一段连续预分配的内存
KV cache shape: [batch_size, max_seq_len, num_heads, head_size]
                 ↑
        按最大长度预分配，连续存放
```

这个假设直接继承自 **Transformer 训练** —— 训练时 batch 内所有样本 pad 到相同长度，KV cache 天然是连续方阵。推理引擎对此没有质疑。

```python
# 传统引擎的典型做法（伪代码）
kv_cache = torch.empty(batch_size, max_seq_len, num_heads, head_size)
# 每个请求占一行连续内存：kv_cache[req_id, :seq_len, :, :]
```

### 11.2 为什么这个假设在推理时是有害的

| | 训练 | 推理（传统引擎） | 推理（理想情况） |
|---|---|---|---|
| **序列长度** | 固定（pad 到 max） | 动态变化 | 动态变化 |
| **分配时机** | 一次性 | 预留最大长度 | 按需逐步分配 |
| **共享** | 无 | 无 | Prefix 可以共享 |
| **内存浪费来源** | padding token | 预留但从未使用的空间 | 几乎为零 |

传统引擎为每个请求预留 `max_model_len` 的空间。但实际场景中：
- 一个 `max_model_len=8192` 的部署，大多数请求可能只生成 200-500 token
- 每多预留一个位置，就浪费了 `2 × num_layers × num_kv_heads × head_size × dtype_size` 字节
- 当并发请求数达到 100+ 时，预留空间的浪费可以让显存利用率降到 30% 以下

### 11.3 为什么之前没有人这样做

**a) Kernel 复杂度的惯性思维**

连续内存下的 attention kernel 是经典的 tiled softmax matmul —— 学术界和工业界已经优化了十年。加上 block table 间接寻址后：
- 每个 key/value token 的地址需要额外一次 global memory load（查 block_table）+ 一次整数乘加（算偏移）
- 指针不再是规则的 `i += stride` 递增模式，编译器自动向量化的机会减少
- 需要在正确性和性能之间重新权衡 tile 大小、warp 分配等参数

对于大部分团队来说，用额外 5-10% 的 kernel 开销换取内存利用率提升，在单请求或少请求场景下是不划算的 —— 连续分配的浪费在 batch size 小时不明显。

**b) 训练代码的文化惯性**

大多数推理引擎是训练代码库（Fairseq、HuggingFace Transformers）的"推理模式"衍生版本。"KV cache 是连续的" —— 这个假设在训练代码中无处不在，模型的 `forward()` 方法、attention 实现、甚至 checkpoint 格式都隐含依赖于此。在训练框架上打补丁做推理优化时，很少有团队会质疑这个底层前提。

**c) 收益场景的滞后出现**

LLM 推理服务的规模化部署（数百并发请求、序列长度差异极大、大量共享 system prompt）是 2023 年下半年才成为普遍需求的。在此之前：
- 单用户对话场景 → 浪费不严重
- 批量离线推理 → 可以按 max_seq_len 做 padding
- API 服务的并发规模较小 → 简单的预分配够用

vLLM 正是在这个转折点上出现的 —— 当 OpenAI、Anthropic 等 API 服务商开始服务百万级用户时，KV cache 显存浪费从"可以忽略"变成了"首要瓶颈"。

**d) 这不是单点优化，而是系统工程**

实现 PagedAttention 需要的不只是一个带 block_table 的 kernel：

```
Kernel 层：attention 计算 + KV cache 读写全部支持间接寻址
    ↓
分配层：block pool、空闲链表、引用计数
    ↓
缓存层：prefix caching 哈希表、链式哈希、跨 block size 协调
    ↓
调度层：prefix-aware 调度、preemption / recomputation
    ↓
混合模型层：多种 attention type 的 joint prefix caching 协调
```

这整套系统需要从 kernel 到 scheduler 的垂直重构。投入成本高，且成功的可能性在 2023 年初并不明显。vLLM 团队的核心贡献正是**识别出这个问题值得投入系统性工程**，并将 OS 虚拟内存的经典思想映射到了 KV cache 管理这个新领域。

### 11.4 从"类比"到"工程"的关键一跳

PagedAttention 和 OS 虚拟内存的关系本质上是**概念类比**，而非实现复用：

| | OS 虚拟内存 | PagedAttention |
|---|---|---|
| **解决的问题** | 物理内存不足、进程隔离 | KV cache 内部碎片、prefix 无法共享 |
| **"虚拟地址"** | CPU 发出的地址，经 MMU 翻译 | token 的逻辑位置 `(seq_idx, block_idx)` |
| **"物理地址"** | DRAM 中的实际位置 | KV cache tensor 中的物理 block 偏移 |
| **"页表"** | 多级硬件页表，MMU/TLB 加速 | `int32` 数组，kernel 手工查表 |
| **"页"** | 4KB (x86)，硬件定义 | 16/32 tokens，应用层超参数 |
| **"缺页中断"** | 硬件 page fault → OS 分配物理页 | 不存在；block 未分配时对应的 token 尚未生成 |
| **"换页"** | 换出到磁盘 | 不支持；全量驻留 GPU 显存 |

PagedAttention 的创新不在于"发现了 GPU 硬件支持分页"，而在于**用软件实现了分页所需的间接寻址，并将整套系统工程的复杂性控制在可接受范围内**。这是 2023 年 LLM 服务化的关键基础设施突破。

---

## 12. 总结：关键设计决策

| 决策 | 代码位置 | 设计 | 理由 |
|------|----------|------|------|
| **Block 粒度** | — | 16/32 tokens/block | 平衡内存碎片和管理开销 |
| **间接寻址** | [attention_kernels.cuh:85](csrc/attention/attention_kernels.cuh#L85) | block_table 在 kernel 内查表（纯软件） | 实现物理/逻辑解耦的根本机制 |
| **Online softmax** | [attention_kernels.cuh:305-341](csrc/attention/attention_kernels.cuh#L305) | 分块计算 + 在线归一化 | 不需要一次性加载全部 KV cache |
| **引用计数** | [kv_cache_utils.py:116](vllm/v1/core/kv_cache_utils.py#L116) | `ref_cnt` 管理共享 | 多个请求安全共享 prefix block |
| **空闲链表** | [kv_cache_utils.py:164](vllm/v1/core/kv_cache_utils.py#L164) | 双向链表 O(1) 分配/释放 | 高频分配场景的性能关键 |
| **链式哈希** | [kv_cache_utils.py:541](vllm/v1/core/kv_cache_utils.py#L541) | 考虑父 block + 额外 key | 前缀隔离 + 精准缓存匹配 |
| **Cascade attention** | [flash_attn.py:276](vllm/v1/attention/backends/flash_attn.py#L276) | 共享前缀一次计算 | 批量场景减少重复 KV 计算 |
| **V2 长序列** | [attention_kernels.cuh:529](csrc/attention/attention_kernels.cuh#L529) | Partition 并行 + reduce | 超长 context 的并行加速 |
| **多 type 协调** | [kv_cache_coordinator.py:392](vllm/v1/core/kv_cache_coordinator.py#L392) | 固定点迭代 | 混合 attention type 模型的正确性保证 |
