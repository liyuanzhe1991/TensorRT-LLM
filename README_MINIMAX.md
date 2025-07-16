# MiniMaxText01 Model for TensorRT-LLM

这个文档描述了如何在TensorRT-LLM中使用MiniMaxText01模型。该模型已从vLLM成功迁移到TensorRT-LLM的torchflow实现。

## 模型特点

MiniMaxText01是一个创新的语言模型，具有以下特点：

1. **混合注意力机制**：结合了Linear Attention（基于Lightning Attention）和标准的Multi-Head Attention
2. **MoE层**：支持专家混合（Mixture of Experts）架构
3. **共享MLP**：在MoE层中支持共享的MLP组件
4. **自定义RMSNorm**：支持张量并行的RMSNorm实现
5. **灵活的层配置**：可以自由配置每一层使用哪种注意力机制

## 主要组件

### 1. MiniMaxText01LinearAttention
- 实现了Linear Attention机制
- 使用slope rate进行衰减
- 支持张量并行
- **支持特殊的KV缓存管理**

### 2. MiniMaxText01Attention  
- 标准的Multi-Head Attention
- 支持RoPE位置编码
- 兼容GQA（Grouped Query Attention）

### 3. MiniMaxText01MoE
- 专家混合层实现
- 支持Top-K路由
- 可配置专家数量

### 4. MiniMaxText01RMSNormTP
- 支持张量并行的RMSNorm
- 自动处理权重分片

### 5. MiniMaxLinearCacheManager
- 专门管理Linear Attention的KV缓存
- 缓存形状：`[num_linear_layers, max_batch_size, num_heads, head_dim, head_dim]`
- 支持动态slot分配和释放
- 与标准attention的KV缓存独立管理

## KV缓存管理

MiniMaxText01模型实现了两种不同的KV缓存机制：

### 1. 标准Attention的KV缓存
- 使用TensorRT-LLM内置的KV缓存管理机制
- 支持paged attention等优化

### 2. Linear Attention的增强缓存管理
- 使用增强版`MiniMaxLinearCacheManager`类管理
- 缓存形状：`[num_layers, batch_size, num_heads, head_dim, head_dim]`
- 递归更新机制：`kv_state = decay * kv_state + k * v^T`

### 增强的缓存管理功能

#### 1. **请求级别的缓存管理**
```python
# 分配缓存时关联请求ID
state_indices = cache_manager.get_state_indices(seq_ids, request_id="req_001")

# 释放整个请求的所有缓存
model.free_cache_request("req_001")
```

#### 2. **缓存统计和监控**
```python
# 获取缓存使用统计
stats = model.get_cache_stats()
print(f"缓存利用率: {stats['utilization']:.2%}")
print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"已用/总槽位: {stats['used_slots']}/{stats['total_slots']}")
print(f"缓存大小: {stats['cache_size_mb']:.2f} MB")
```

#### 3. **CUDA图优化支持**
```python
# 在CUDA图捕获前准备输入
input_buffers = model.copy_inputs_before_cuda_graphs(
    input_buffers, 
    seq_ids=seq_ids
)

# 获取序列长度无关的捕获输入
capture_inputs = model.get_seqlen_agnostic_capture_inputs(batch_size)
```

#### 4. **缓存持久化**
```python
# 保存缓存状态到磁盘
model.save_cache_state("cache_checkpoint.pt")

# 从磁盘加载缓存状态
model.load_cache_state("cache_checkpoint.pt")
```

#### 5. **预填充缓存清理**
```python
# 自动清理新预填充序列的缓存
hidden_states = model(
    input_ids=input_ids,
    seq_ids=seq_ids,
    context_lens=[0, 512, 0],  # 0表示新的预填充
)
```

### 缓存管理最佳实践

1. **批处理优化**
   ```python
   # 批量分配缓存槽位
   seq_ids = list(range(batch_size))
   state_indices = cache_manager.get_state_indices(
       seq_ids, 
       request_id="batch_001"
   )
   ```

2. **内存管理**
   ```python
   # 定期检查缓存使用情况
   if cache_stats['utilization'] > 0.9:
       # 触发缓存清理或扩容逻辑
       logger.warning("Cache utilization high, consider cleanup")
   ```

3. **错误恢复**
   ```python
   try:
       # 推理过程
       output = model(...)
   except RuntimeError as e:
       if "No free cache slots" in str(e):
           # 清理未使用的请求
           model.free_cache_request("old_request")
           # 重试
   ```

### 高级缓存配置

```python
# 创建带高级配置的缓存管理器
cache_manager = MiniMaxLinearCacheManager(
    num_linear_layers=20,
    max_batch_size=64,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
    device='cuda',
    enable_cuda_graph=True,  # 启用CUDA图优化
    max_context_len=32768,   # 最大上下文长度
)
```

### 性能调优建议

1. **缓存预分配**: 在高负载场景下，预先分配常用的缓存槽位
2. **批量操作**: 尽量批量处理请求以减少缓存管理开销
3. **定期清理**: 实现定期清理未使用缓存的机制
4. **监控告警**: 设置缓存利用率和命中率的监控告警

## 使用方法

### 1. 加载模型

```python
from tensorrt_llm._torch.models import MiniMaxText01ForCausalLM, MiniMaxText01Config
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

# 创建配置
config = MiniMaxText01Config(
    vocab_size=200064,
    hidden_size=7168,
    intermediate_size=18944,
    num_hidden_layers=60,
    num_attention_heads=56,
    num_key_value_heads=8,
    decoder_attention_types=[0]*20 + [1]*40,  # 前20层使用linear attention
    # ... 其他配置
)

# 创建映射
mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

# 创建模型配置
model_config = ModelConfig(
    pretrained_config=config,
    mapping=mapping,
    dtype='float16',
    max_batch_size=32,  # 需要设置以初始化缓存管理器
)

# 创建模型
model = MiniMaxText01ForCausalLM(model_config)
```

### 2. 运行推理

```python
import torch
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata

# 准备输入
input_ids = torch.randint(0, config.vocab_size, (batch_size * seq_len,), device='cuda')
position_ids = torch.arange(seq_len, device='cuda').repeat(batch_size)

# 创建attention metadata
attn_metadata = VanillaAttentionMetadata(
    seq_lens=seq_lens,
    seq_lens_cuda=torch.tensor(seq_lens, device='cuda'),
    # ... 其他metadata
)

# 设置序列ID（用于缓存管理）
seq_ids = list(range(batch_size))

# 前向传播
with torch.no_grad():
    hidden_states = model.model(
        attn_metadata=attn_metadata,
        input_ids=input_ids,
        position_ids=position_ids,
        seq_ids=seq_ids,  # 传递序列ID
    )
    logits = model.compute_logits(hidden_states, None)

# 清理缓存（序列结束后）
for seq_id in seq_ids:
    model.model.free_cache_slot(seq_id)
```

## 配置参数说明

### 基础参数
- `vocab_size`: 词汇表大小
- `hidden_size`: 隐藏层维度
- `intermediate_size`: FFN中间层维度
- `num_hidden_layers`: 总层数
- `num_attention_heads`: 注意力头数
- `num_key_value_heads`: KV头数（用于GQA）

### MiniMax特有参数
- `decoder_attention_types`: 每层使用的注意力类型列表（0=linear, 1=standard）
- `block_size`: Linear attention的块大小
- `num_local_experts`: 每层的专家数量
- `num_experts_per_tok`: 每个token激活的专家数
- `shared_intermediate_size`: 共享MLP的中间层维度
- `shared_moe_mode`: 共享MLP的组合模式（'softmax' 或 'sigmoid'）
- `layernorm_*_alpha/beta`: 层归一化的缩放和偏移参数

## 注意事项

1. **Lightning Attention**: 当前实现使用了简化的linear attention占位符。在生产环境中，应该集成实际的Lightning Attention kernel以获得最佳性能。

2. **内存管理**: Linear attention需要特殊的KV缓存管理。确保为linear attention层分配足够的缓存空间。

3. **张量并行**: 模型支持张量并行，但需要正确配置mapping参数。

4. **权重加载**: 从Hugging Face或vLLM加载权重时，`load_weights`方法会自动处理权重名称的映射。

## 测试

运行测试脚本以验证模型功能：

```bash
cd trtllm_06_17
python test_minimax.py
```

## 与vLLM的差异

1. **缓存管理**: TensorRT-LLM使用不同的缓存管理机制，linear attention的缓存需要特殊处理
2. **并行策略**: 使用TensorRT-LLM的分布式框架而非vLLM的并行机制
3. **优化**: 可以利用TensorRT-LLM的各种优化特性，如INT8量化、Flash Attention等

## 后续优化建议

1. 实现真正的Lightning Attention kernel
2. 添加对FP8/INT8量化的支持
3. 优化MoE层的路由机制
4. 实现更高效的linear attention缓存管理
5. 支持CUDA Graph优化

## 贡献

欢迎提交Issue和Pull Request来改进这个实现！ 