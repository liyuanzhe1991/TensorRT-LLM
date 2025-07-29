"""
MiniMaxText01 model implementation for TensorRT-LLM torchflow
"""
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, MoEAllReduce,allgather)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import HfLoraLoader

from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (create_moe, MoEWeightLoadingMode,
                                 BaseMoeRoutingMethod)
from ..modules.fused_moe.routing import RenormalizeMoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode,WeightMode, WeightsLoadingConfig
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, register_auto_model)
from collections.abc import Iterable


class MiniMaxLinearCacheManager:
    """Cache manager for MiniMax linear attention layers, inspired by vLLM's ConstantSizeCache"""
    
    def __init__(
        self,
        num_linear_layers: int,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
        enable_cuda_graph: bool = True,
        max_context_len: int = 2048,
        tp_rank: int = 0,
    ):
        self.num_linear_layers = num_linear_layers
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.enable_cuda_graph = enable_cuda_graph
        self.max_context_len = max_context_len
        self.tp_rank = tp_rank
        # Initialize the cache tensor
        # Shape: [num_linear_layers, max_batch_size, num_heads, head_dim, head_dim]
        self.seq_id_to_slot_idx: Dict[int, int] = {}
        # List of available cache slots
        self.free_slots = list(range(max_batch_size))
        self.cache = torch.zeros(
            (num_linear_layers, max_batch_size, num_heads, head_dim, head_dim),
            dtype=dtype,
            device=device,
        )
        #print(f"trt init linear cache manager finish shape of cache is {self.cache.shape}, size of ....")
        # Mapping system inspired by vLLM
        # Maps seq_id -> cache slot index
        
        #exit(-1)
        # CUDA graph support
        if enable_cuda_graph:
            self._init_cuda_graph_buffers()
        
    def _init_cuda_graph_buffers(self):
        """Initialize buffers for CUDA graph capture"""
        # Pre-allocate tensors for CUDA graph
        self.cuda_graph_cache_indices = torch.zeros(
            self.max_batch_size, dtype=torch.long, device=self.device
        )
    
    def get_layer_cache(self, layer_idx: int) -> torch.Tensor:
        """Get cache for a specific layer"""
        return self.cache[layer_idx]
    
    def allocate_slot_for_seq(self, seq_id: int) -> int:
        """Allocate a cache slot for a sequence ID"""
        # If already allocated, return existing slot
        if seq_id in self.seq_id_to_slot_idx:
            return self.seq_id_to_slot_idx[seq_id]
        
        # Allocate new slot
        if not self.free_slots:
            raise RuntimeError(f"No free cache slots available (max_batch_size={self.max_batch_size})")
        
        slot_idx = self.free_slots.pop(0)
        self.seq_id_to_slot_idx[seq_id] = slot_idx
        
        # Clear the cache slot
        self.cache[:, slot_idx].zero_()
        
        return slot_idx
    
    def get_state_indices_from_seq_ids(self, seq_ids: List[int]) -> torch.Tensor:
        """Convert sequence IDs to cache slot indices"""
        slot_indices = [] # 【req_idx,slot_idx]
        
        for seq_id in seq_ids:
            if seq_id in self.seq_id_to_slot_idx:
                slot_indices.append(self.seq_id_to_slot_idx[seq_id])
            else:
                # Allocate new slot for this sequence
                slot_idx = self.allocate_slot_for_seq(seq_id)
                slot_indices.append(slot_idx)
        
        return torch.tensor(slot_indices, device=self.device, dtype=torch.long)
    
    def free_seq(self, seq_id: int):
        """Free the cache slot used by a sequence"""
        #print("trt free_seq",seq_id)
        if seq_id in self.seq_id_to_slot_idx.keys():
            slot_idx = self.seq_id_to_slot_idx[seq_id]
            del self.seq_id_to_slot_idx[seq_id]
            self.free_slots.append(slot_idx)
            # Clear the cache slot
            self.cache[:, slot_idx].zero_()
    
    def clear_all(self):
        """Clear all cache slots and mappings"""
        self.seq_id_to_slot_idx.clear()
        self.free_slots = list(range(self.max_batch_size))
        self.cache.zero_()


class MiniMaxLinearCacheResourceManager:
    """Resource manager wrapper for MiniMaxLinearCacheManager to integrate with PyExecutor"""
    
    def __init__(self, num_linear_layers: int,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
        enable_cuda_graph: bool = True,
        max_context_len: int = 2048,
        tp_rank: int = 0):
        self.linear_cache_manager = MiniMaxLinearCacheManager(
            num_linear_layers=num_linear_layers,
            max_batch_size=max_batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            tp_rank=tp_rank,
        )
        self.name = "minimax_linear_cache_manager"
        
    def prepare_resources(self, scheduled_requests):
        """Called before forward pass - allocate slots for new sequences"""
        # Extract sequence IDs from scheduled requests
        seq_ids = []
        
        for req in scheduled_requests.context_requests:
            if hasattr(req, 'request_id'):
                seq_ids.append(req.request_id)
            elif hasattr(req, 'py_request_id'):
                seq_ids.append(req.py_request_id)
                
        for req in scheduled_requests.generation_requests:
            if hasattr(req, 'request_id'):
                seq_ids.append(req.request_id)
            elif hasattr(req, 'py_request_id'):
                seq_ids.append(req.py_request_id)
        
        # Pre-allocate slots for all sequences
        for seq_id in seq_ids:
            self.linear_cache_manager.allocate_slot_for_seq(seq_id)
            
    def update_resources(self, scheduled_requests):
        """Called after forward pass - currently no-op for linear cache"""
        pass
        
    def free_resources(self, seq_id):
        """Called when request is terminated - free the cache slot"""
        if seq_id is not None:
            self.linear_cache_manager.free_seq(seq_id)
            
    def get_kv_cache_stats(self):
        """Get cache statistics"""
        return {
            'allocated_slots': len(self.linear_cache_manager.seq_id_to_slot_idx),
            'free_slots': len(self.linear_cache_manager.free_slots),
            'max_slots': self.linear_cache_manager.max_batch_size,
            'seq_id_to_slot_mapping': dict(self.linear_cache_manager.seq_id_to_slot_idx),
        }
        
    def shutdown(self):
        """Clean up resources on shutdown"""
        self.linear_cache_manager.clear_all()


class MiniMaxText01Config(PretrainedConfig):
    """Configuration class for MiniMaxText01."""
    model_type = "minimax_text_01"
    
    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6144,
        intermediate_size=9216,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=1000000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        sliding_window=None,
        attention_bias=False,
        attention_dropout=0.0,
        # MiniMax specific
        decoder_attention_types=None,  # List of 0s and 1s (0=linear, 1=standard)
        attn_type_list=None,  # Alias for decoder_attention_types
        block_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        shared_intermediate_size=0,
        shared_moe_mode='softmax',
        layernorm_linear_attention_alpha=1.0,
        layernorm_linear_attention_beta=1.0,
        layernorm_full_attention_alpha=1.0,
        layernorm_full_attention_beta=1.0,
        layernorm_mlp_alpha=1.0,
        layernorm_mlp_beta=1.0,
        postnorm=False,
        rotary_dim=None,
        
        **kwargs,
    ):
     
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        # MiniMax specific
        self.decoder_attention_types = decoder_attention_types or attn_type_list
        if self.decoder_attention_types is None:
            self.decoder_attention_types = [1] * num_hidden_layers
        self.attn_type_list = self.decoder_attention_types
        self.block = block_size
        self.block_size = block_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_moe_mode = shared_moe_mode
        self.layernorm_linear_attention_alpha = layernorm_linear_attention_alpha
        self.layernorm_linear_attention_beta = layernorm_linear_attention_beta
        self.layernorm_full_attention_alpha = layernorm_full_attention_alpha
        self.layernorm_full_attention_beta = layernorm_full_attention_beta
        self.layernorm_mlp_alpha = layernorm_mlp_alpha
        self.layernorm_mlp_beta = layernorm_mlp_beta
        self.postnorm = postnorm
        self.rotary_dim = rotary_dim
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class MiniMaxText01RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniMaxText01RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,dtype=torch.bfloat16))
        self.variance_epsilon = eps
        self.variance_res=None
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        #print("trt rmsnorm forward hidden_states",hidden_states.shape,hidden_states.dtype,"weight",self.weight.shape,self.weight.dtype)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        self.variance_res=variance

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * (hidden_states).to(input_dtype)
    
class MiniMaxText01RMSNormTP(nn.Module):
    """RMSNorm with tensor parallelism support for MiniMax Linear Attention."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, mapping=None) -> None:
        super().__init__()
        # Get TP info from mapping if provided
        assert mapping is not None
        #if mapping is not None:
        self.tp_world = mapping.tp_size
        self.tp_rank = mapping.tp_rank
       
        
        # hidden_size is the full size, we need to divide by TP size
        self.full_hidden_size = hidden_size
        self.local_hidden_size = hidden_size // self.tp_world
        
        # Create weight parameter for the local portion
        self.weight = nn.Parameter(torch.ones(self.local_hidden_size,dtype=torch.bfloat16))
        self.variance_epsilon = eps
        # Use float32 for all_reduce to ensure compatibility
        self.all_reduce = AllReduce(mapping=mapping, dtype=torch.float32,strategy=AllReduceStrategy.NCCL)
        
        # self.variance_res=None
        # self.input_layernorm_input=None
        # self.input_layernorm_output=None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, local_hidden_size] where local_hidden_size = tp_heads * head_dim
    
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        #self.input_layernorm_input=x.clone()
   
        # Compute variance locally first
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        #print("variance",variance.shape,variance.dtype)
        if self.tp_world > 1:
            variance = self.all_reduce(input=variance) / self.tp_world
        #print("tp variance",variance)
        #self.variance_res=variance
      
        
        # Apply RMS normalization
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        
        # Handle weight size mismatch
        weight = self.weight
        #print("x",x.shape,"weight",self.weight.shape)
        if x.size(-1) != self.weight.size(0):
            if self.weight.size(0) < x.size(-1):
                repeat_count = (x.size(-1) + self.weight.size(0) - 1) // self.weight.size(0)
                full_weight = self.weight.repeat(repeat_count)
                weight = full_weight[:x.size(-1)]
            else:
                weight = self.weight[:x.size(-1)]
        
        # Apply weight
        x = x.to(orig_dtype) * weight
        #self.input_layernorm_output=x.clone()
        return x
    
    
    def load_weights(
        self,
        weights: torch.Tensor,
    ) -> None:
        
        #print(f"RMSNormTP load_weights ................",weights[0]['weight'].shape,self.tp_world,self.tp_rank,self.weight.shape)
        assert len(weights) == 1
        shard_size = weights[0]['weight'].shape[0] // self.tp_world
        #print("tp rank",self.tp_rank,"shard_size",shard_size)
        shard = slice(self.tp_rank * shard_size, (self.tp_rank + 1) * shard_size)
        
        # Avoid in-place operation on parameter that requires grad
        with torch.no_grad():
            self.weight.copy_(weights[0]['weight'][shard])
        return

class MiniMaxText01LinearAttention(nn.Module):
    """Linear attention implementation for MiniMaxText01.
    
    Lightning Attention is a linear attention mechanism that achieves O(n) complexity
    instead of O(n²) for standard attention. It uses a cumulative KV state approach
    that inherently implements causal masking:
    
    1. Causal Masking: The algorithm naturally enforces causality by accumulating
       KV states sequentially. At position i, the output only depends on positions
       0 to i (inclusive), making explicit causal masks unnecessary.
       
    2. State Accumulation: Instead of computing attention over all pairs, it maintains
       a running state matrix that accumulates key-value outer products with decay:
       state_t = decay * state_{t-1} + k_t ⊗ v_t
       
    3. Efficient Computation: The output at each position is computed as:
       output_t = q_t · state_t
       This gives O(n·d²) complexity where d is the head dimension.
       
    4. Memory Efficiency: Only requires storing a state matrix of size [heads, dim, dim]
       per sequence, regardless of sequence length.
    """
    
    def __init__(
        self,
        model_config: ModelConfig[MiniMaxText01Config],
        layer_idx: int,
        linear_layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        
        self.layer_idx = layer_idx
        self.linear_layer_idx = linear_layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_inner_size = self.head_dim * self.num_heads
        
        # Handle block_size attribute - HF config might use 'block' instead of 'block_size'
        self.block_size = getattr(config, 'block_size', getattr(config, 'block', 256))
        
        # Get tensor parallel info from model config
        self.tp_size = model_config.mapping.tp_size
        self.tp_rank = model_config.mapping.tp_rank
        self.tp_heads = self.num_heads // self.tp_size
        self.mapping=model_config.mapping
   
        
        # Projections
        self.linear_qkv_proj = Linear(
            self.hidden_size,
            self.hidden_inner_size*3,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.VANILLA)
            
        )# first make sure the qkv_proj is correct
        
        self.output_gate = Linear(
            self.hidden_size,
            self.hidden_inner_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        
        self.out_proj = Linear(
            self.hidden_inner_size,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )

        self.output_proj_res=None
        #self
        # Use custom RMSNorm that handles tensor parallelism correctly
        self.norm = MiniMaxText01RMSNormTP(self.hidden_inner_size, eps=config.rms_norm_eps, mapping=model_config.mapping)
        
        # Build slope tensor on the correct device
        slope_rate = self._build_slope_tensor(self.num_heads)
        #print("trt slope_rate",slope_rate.shape)
        
        self.slope_rate = slope_rate * (1 - layer_idx / (80 - 1) + 1e-5)
        # self.tp_slope = self.slope_rate[self.tp_rank *
        #                                 self.tp_heads:(self.tp_rank + 1) *
        #                                 self.tp_heads].contiguous()
        # Move to GPU and make it a buffer (not a parameter)
        self.register_buffer('tp_slope', 
                           self.slope_rate[self.tp_rank * self.tp_heads:(self.tp_rank + 1) * self.tp_heads].contiguous())
        #print("trt tp_slope",self.tp_slope.shape)
        #reset tp_slope to 1
        #self.tp_slope.fill_(1)
        
        self.q=None
        self.k=None
        self.v=None
        self.qkv=None
        self.lightning_output=None
        self.attn_norm_output=None
        self.input_hidden_states=None
        #self.all_gather = allgather(mapping=model_config.mapping, dtype=torch.bfloat16)
        
        self.decode_kv=None
        self.decode_kv_state=None
        self.decode_qkv=None
        self.decode_q=None
        
    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + 
                        get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32).reshape(n_attention_heads, 1, 1)
        return slopes
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.IntTensor],
        attn_metadata: AttentionMetadata,
        kv_cache: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        
        #self.input_hidden_states=hidden_states.clone()
        #hidden_states=hidden_states.to(torch.float32)
        qkv = self.linear_qkv_proj(hidden_states)
        #qkv=qkv.to(torch.float32)
        qkv = F.silu(qkv)
        #print("trt qkv shape",qkv.shape)#[s,3*1024]
        #_qkv=qkv.split([1024]*3,dim=-1)
       
        #self.qkv=qkv
        #if it's possiable that all q?
        new_shape = qkv.size()[:-1] + (self.tp_heads, -1)
        
        #print("hf new_shape",new_shape)
        qkv = qkv.view(*new_shape)
        #print("trt qkv",qkv.shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=-1)
        #self.q=allgather(q.clone(),mapping=self.mapping,dim=1)
        #self.k=allgather(k.clone(),mapping=self.mapping,dim=1)
        #self.v=allgather(v.clone(),mapping=self.mapping,dim=1)
        #print("trt q",self.q.shape,self.q.dtype,"k",self.k.shape,self.k.dtype,"v",self.v.shape,self.v.dtype)
    
        #new_shape = qkv.size()[:-1] + (self.tp_heads, -1)
     
        #qkv = qkv.view(*new_shape)
        #print("trt qkv",qkv.shape)
        #q = qkv[..., 0:1024]
        #k = qkv[..., 1024:2048]
        #v = qkv[..., 2048:]
        #print("sum q {} k {} v {}".format(q.sum(),k.sum(),v.sum()))
        # self.q=allgather(q.clone(),mapping=self.mapping,dim=1)
        # self.k=allgather(k.clone(),mapping=self.mapping,dim=1)
        # self.v=allgather(v.clone(),mapping=self.mapping,dim=1)
        # self.q=self.q.reshape(self.q.shape[0],self.num_heads,-1)
        # self.k=self.k.reshape(self.k.shape[0],self.num_heads,-1)
        # self.v=self.v.reshape(self.v.shape[0],self.num_heads,-1)
        # q=q.reshape(q.shape[0],self.tp_heads,-1)
        # k=k.reshape(k.shape[0],self.tp_heads,-1)
        # v=v.reshape(v.shape[0],self.tp_heads,-1)
        #print(f"diff debug {(self.q[:,self.tp_rank*self.tp_heads:(self.tp_rank+1)*self.tp_heads,:]-local_q).sum()}")
        #
        # print("trt q",self.q.shape,self.q.dtype,"k",self.k.shape,self.k.dtype,"v",self.v.shape,self.v.dtype)
        #test end
        #qkv32 = qkv.to(torch.float32)
        
        #qkvact=qkv.clone()
        
        
        # Reshape and split
        # num_tokens = hidden_states.shape[0]
        # qkvact = qkvact.view(num_tokens, self.tp_heads, -1)#[s,tpheads,3(q,k,v)*head_dim]
        # print("trt qkvact",qkvact.shape)
        # q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)#split correct?
        
        #let's allgather here to see if q,k,v are correct
        # self.q=allgather(q.clone(),mapping=self.mapping,dim=1)
        # self.k=allgather(k.clone(),mapping=self.mapping,dim=1)
        # self.v=allgather(v.clone(),mapping=self.mapping,dim=1)
        #print("trt q",q.shape,q.dtype,"k",k.shape,k.dtype,"v",v.shape,v.dtype)

        #print("trt q",self.q.shape,self.q.dtype,"k",self.k.shape,self.k.dtype,"v",self.v.shape,self.v.dtype)
        if kv_cache is None:
            # For inference, this should be provided by the KV cache manager
            # Here we create a placeholder for testing
            max_batch_size = getattr(attn_metadata, 'max_batch_size', 1)
            kv_cache = torch.zeros(
                (max_batch_size, self.tp_heads, self.head_dim, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
        
        # Handle different phases: prefill vs decode
        # Lightning Attention inherently implements causal masking through its cumulative design
        # The causal mask is implicit in the algorithm: at position i, we only have access to
        # the accumulated KV state from positions 0 to i-1, plus the current position i
        if hasattr(attn_metadata, 'num_contexts') and attn_metadata.num_contexts > 0:
            output = self._prefill_forward(q, k, v, kv_cache, state_indices, attn_metadata)
        else:
            output = self._decode_forward(q, k, v, kv_cache, state_indices, attn_metadata)
        
        output_with_heads = output.reshape(output.shape[0], -1)
    
    # 可选：重塑为 [seq_len, tp_heads, head_dim] 以便调试
        output_with_heads = output.reshape(output_with_heads.shape[0], self.tp_heads, self.head_dim)
        #self.lightning_output=allgather(output_with_heads.clone(),mapping=self.mapping,dim=1)
        #print("trt lightning_output",self.lightning_output.shape)#【4,8,128]
        # Apply normalization
        output = self.norm(output)
        
        # Apply output gate
        gate = self.output_gate(hidden_states)
        output = F.sigmoid(gate) * output
        output = output.to(torch.bfloat16)
        
        # Apply output projection
        output = self.out_proj(output)
        self.output_proj_res=output
        return output
    
    def _prefill_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        state_indices: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Handle prefill phase with multiple tokens and proper masking"""
        outputs = []
        
        # Process each sequence in the batch
        start = 0
        assert getattr(attn_metadata, 'request_ids', None) is not None, "request_ids is not found in attn_metadata"
        
        for seq_idx in range(len(getattr(attn_metadata, 'request_ids'))):
            if hasattr(attn_metadata, 'seq_lens'):
                seq_len = attn_metadata.seq_lens[seq_idx]
                end = start + seq_len
                
            else:
                raise ValueError("seq_lens is not found in attn_metadata")
            
            # Get cache slot for this sequence
            cache_idx = state_indices[seq_idx] if state_indices is not None else seq_idx
            
            # Extract sequence data
            seq_q = q[start:end]  # [seq_len, tp_heads, head_dim]
            seq_k = k[start:end]  # [seq_len, tp_heads, head_dim]
            seq_v = v[start:end]  # [seq_len, tp_heads, head_dim]
            
            # Process with Lightning Attention (causal masking is implicit)
            #print("trt tp_slope",self.tp_slope)
            #print("trt prefill seq_idx",seq_idx,"cache_idx",cache_idx)
            seq_output = self._lightning_attention_forward(
                seq_q,
                seq_k,
                seq_v,
                kv_cache[cache_idx],
                self.tp_slope
            )
            outputs.append(seq_output)
            start = end
            
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.tp_heads * self.head_dim), device=q.device, dtype=q.dtype)
    
    def _decode_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        state_indices: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Handle decode phase with single token per sequence"""
        outputs = []
        
        # Process each sequence
        for seq_idx in range(len(attn_metadata.request_ids)):
            # Get cache slot for this sequence
            
            cache_idx = state_indices[seq_idx].item() if state_indices is not None else seq_idx
            
            # Ensure cache_idx is within bounds
            if cache_idx >= kv_cache.shape[0]:
                raise RuntimeError(f"Cache index {cache_idx} is out of bounds for kv_cache with shape {kv_cache.shape}")
            
            # Process single token for this sequence
            # In decode phase, each sequence has exactly one new token
            #print("trt decode seq_idx",seq_idx,"cache_idx",cache_idx)
            seq_output = self._lightning_attention_decode(
                q[seq_idx:seq_idx+1],  # [1, tp_heads, head_dim]
                k[seq_idx:seq_idx+1],  # [1, tp_heads, head_dim]
                v[seq_idx:seq_idx+1],  # [1, tp_heads, head_dim]
                kv_cache[cache_idx],
                self.tp_slope
            )
            outputs.append(seq_output)
        
        return torch.cat(outputs, dim=0)
    
    def _lightning_attention_forward(
        self,
        q: torch.Tensor,  # [seq_len, tp_heads, head_dim]
        k: torch.Tensor,  # [seq_len, tp_heads, head_dim]
        v: torch.Tensor,  # [seq_len, tp_heads, head_dim]
        kv_state: torch.Tensor,  # [tp_heads, head_dim, head_dim]
        slope_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Lightning attention for prefill with block-wise processing and causal masking"""
        #print(q.shape)
        slope_rate = slope_rate.to(torch.float32)
        seq_len = q.shape[0]  # n
        num_heads = q.shape[1]  # h
        head_dim = q.shape[2]  # d
        
        # Convert to float32 for numerical stability
        orig_dtype = q.dtype
       
        
        # Reshape to match HF format: [1, h, n, d]
        q = q.transpose(0, 1).unsqueeze(0)  # [1, h, n, d]
        k = k.transpose(0, 1).unsqueeze(0)  # [1, h, n, d]
        v = v.transpose(0, 1).unsqueeze(0)  # [1, h, n, d]
        
        
        b, h, n, d = q.shape
        e = v.shape[-1]
        
        # Block size
        BLOCK = self.block_size
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
        
        # Compute decay factors (same as HF)
        array = torch.arange(BLOCK).to(q.device) + 1
        q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
        k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
        #print("trt qdecay ... layer",self.layer_idx,q_decay.shape,q_decay)
        #rint("trt kdecay ... layer",self.layer_idx,k_decay.shape,k_decay)
        # Diagonal decay matrix
        index = array[:, None] - array[None, :]
        s_index = slope_rate * index[None, None,]
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)
        
        #self.diag_decay=diag_decay
        # Initialize KV state
        kv = kv_state.float()  # [h, d, e]
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        
        # Process blocks
        #print("trt NUM_BLOCK",NUM_BLOCK)
        
        #print("trt kv debug",kv.min(),kv.max(),kv.mean())
        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = min(si + BLOCK, n)
            #print("trt si",si,"ei",ei)
            m = ei - si
            
            # Extract block
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
                
            # Non-diagonal part: interaction with previous KV state
            # print("trt qi",qi.shape,qi.dtype,qi.device)
            # print("trt q_decay",q_decay.shape,q_decay.dtype,q_decay.device)
            # print("trt m",m)
            #print("trt kv",kv.shape,kv.dtype,kv.device)
            qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv.unsqueeze(0)).to(torch.float32)#inter blcok
         
            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]#
            qkv_diag = torch.matmul(qk, vi.to(torch.float32))

            # Combine outputs
            output[:, :, si:ei] = qkv_none_diag + qkv_diag
            # Update KV state for next block
            block_decay = torch.exp(-slope_rate * m)
            kv = block_decay * kv + torch.matmul(
                (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), 
                vi
            ).squeeze(0)  # Remove batch dimension for kv update
        
        # Update the KV state
        #print("trt kv",kv.shape,kv.dtype,kv.device)
        
        with torch.no_grad():
            kv_state.copy_(kv)
        # if self.layer_idx==0 and self.tp_rank==0:
        #     print("layer {} rank {} trt kv_state prefill".format(self.layer_idx,self.tp_rank),kv_state.shape,kv_state.dtype,kv_state.device,kv_state.min(),kv_state.max(),kv_state.mean())
        # Convert back to original format and dtype
        output = output.squeeze(0).transpose(0, 1).contiguous()  # [n, h, d]
        #output = output.to(orig_dtype)
        
        # Reshape output
        return output.reshape(seq_len, -1)  # [seq_len, tp_heads * head_dim]
    
    def _lightning_attention_decode(
        self,
        q: torch.Tensor,  # [1, tp_heads, head_dim]
        k: torch.Tensor,  # [1, tp_heads, head_dim]
        v: torch.Tensor,  # [1, tp_heads, head_dim]
        kv_state: torch.Tensor,  # [tp_heads, head_dim, head_dim]
        slope_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Lightning attention for single token decode with numerical stability"""
        # Convert to float32 for numerical stability
        orig_dtype = q.dtype
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        #print("trt decode q k v shape",q.shape,k.shape,v.shape)
        kv_state_float = kv_state.to(torch.float32)
        #if self.layer_idx==0 and self.tp_rank==0:
        #    print("layer {} rank {} trt kv_state decode".format(self.layer_idx,self.tp_rank),kv_state.shape,kv_state.dtype,kv_state.device,kv_state.min(),kv_state.max(),kv_state.mean())
        #print("trt kv_state_float",kv_state_float.device,kv_state_float.dtype,kv_state_float.shape)
        # Get ratio (same as HF)
        ratio = torch.exp(-slope_rate)  # [tp_heads, 1, 1]
        
        # Update KV state (following HF implementation)
        # kv = ratio * kv + torch.einsum("... n d, ... n e -> ... d e", k, v)
        # Since we have single token, n=1
        kv_update = torch.einsum('nhd,nhe->hde', k, v)  # [tp_heads, head_dim, head_dim]
        # print("trt kv_update",kv_update.device,kv_update.dtype,kv_update.shape)
        # print("trt ratio",ratio.device,ratio.dtype,ratio.shape)
        #self.decode_kv_state=kv_state_float.clone()
        new_kv_state = ratio * kv_state_float + kv_update
        #self.decode_kv=new_kv_state.clone()
        # Update the original kv_state in-place
        with torch.no_grad():
            kv_state.copy_(new_kv_state.to(orig_dtype))
        
        # Compute output
        # qkv = torch.einsum("... n e, ... e d -> ... n d", q, kv)
        self.decode_q=q.clone()
        # 1 8 128， kvsata is 8 128 128
        
        # 使用 HF 风格的 einsum - 添加必要的维度
        q_expanded = q.unsqueeze(2)  # [1, tp_heads, 1, head_dim]
        kv_expanded = new_kv_state.unsqueeze(0)  # [1, tp_heads, head_dim, head_dim]
        output = torch.einsum('...ne,...ed->...nd', q_expanded, kv_expanded.to(q.dtype))
        output = output.squeeze(2)  # [1, tp_heads, head_dim]
        #self.decode_qkv=output.clone()
        # Convert back to original dtype
        output = output.to(orig_dtype)
        
        # Reshape to expected format
        return output.reshape(1, -1)  # [1, tp_heads * head_dim]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    # print("trt q",q.shape,q.dtype)
    # print("trt k",k.shape,k.dtype)
    # print("trt cos",cos.shape,cos.dtype)
    # print("trt sin",sin.shape,sin.dtype)
    
    q_, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    # print("trt q_pass",q_pass.shape,q_pass.dtype)
    # print("trt k_pass",k_pass.shape,k_pass.dtype)
    #print("position_ids 2 ",position_ids.shape,position_ids.dtype)
    #print("cos",cos.shape,cos.dtype)
    #print("sin",sin.shape,sin.dtype)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    
    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)
    # print("q_embed",q_embed.shape,q_embed.dtype)
    # print("k_embed",k_embed.shape,k_embed.dtype)
    return torch.cat((q_embed, q_pass), dim=-1).to(dtype), torch.cat((k_embed, k_pass), dim=-1).to(dtype)


class MiniMaxText01RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None,head_dim=96,num_heads=32,num_kv_heads=8):
        super().__init__()
        
        self.head_dim=head_dim
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
   
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # self.emb=None
        # self.cos=None
        # self.sin=None
        # self.q=None
        # self.k=None
        # self.v=None
        #self.q_post_rope=None
        #self.k_post_rope=None
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.bfloat16
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        #self.emb=emb
        #print("hf .... emb ....",emb.shape)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        #self.emb=emb
        
    def _forward_impl(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        #print("trt rotary emb _forward_impl seq_len",seq_len)
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=torch.float32)

        return (
            self.cos_cached[:seq_len].to(dtype=torch.float32),
            self.sin_cached[:seq_len].to(dtype=torch.float32),
        )
    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor],v:torch.Tensor,position_ids:torch.Tensor,seq_len:int,past_kv_len:int=0):
        bsz = 1
        q_len, _ = q.size()
        q = q.reshape(bsz, q_len,self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len,self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len,self.num_kv_heads, self.head_dim).transpose(1, 2)  
        #self.q=q
        #self.k=k
        #self.v=v
        position_ids = position_ids.view(bsz, seq_len)
        #print("trt rotary emb forward position_ids",position_ids)
        kv_seq_len = k.shape[-2]
        if past_kv_len > 0:
            kv_seq_len += past_kv_len
        #print("trt rotary emb forward past_kv_len",past_kv_len," kv_seq_len ",kv_seq_len," position_ids ",position_ids[:,-1].max().item())
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        # print("trt rotary emb forward q",q.shape,q.dtype)
        # print("trt rotary emb forward k",k.shape,k.dtype)
        # print("trt rotary emb forward v",v.shape,v.dtype)
        # print("trt rotary emb forward position_ids",position_ids.shape,position_ids.dtype)
        # print("trt rotary emb forward seq_len",seq_len)
        cos, sin = self._forward_impl(v, rotary_seq_len)
        # self.cos=cos
        # self.sin=sin
        #print("position_ids",position_ids.shape,position_ids.dtype)
        q, k = apply_rotary_pos_emb(q, k,  cos, sin, position_ids)
        #self.q_post_rope=q
        #self.k_post_rope=k  
        q = q.transpose(1, 2).reshape(q_len, -1)
        k = k.transpose(1, 2).reshape(q_len, -1)
        return q, k
    
class MiniMaxText01Attention(Attention):
    """Standard attention for MiniMaxText01."""
    
    def __init__(
        self,
        model_config: ModelConfig[MiniMaxText01Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads or config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=getattr(config, 'attention_bias', False),
            
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )
        
        self.rotary_emb = MiniMaxText01RotaryEmbedding(
            dim=config.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
        )


class MiniMaxText01MoE(nn.Module):
    """MoE implementation for MiniMaxText01."""
    
    def __init__(
        self,
        model_config: ModelConfig[MiniMaxText01Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # self.topk_indices = None
        # self.topk_values = None
        # self.router_logits = None
        # Router
        self.gate = Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
        )
        
        # Create MoE experts
        #print("trt create_moe, layer_idx",layer_idx,"num_experts",self.num_experts)
        self.experts = create_moe(
            routing_method=self._create_routing_method(),
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=True,
            model_config=model_config,
            layer_idx=layer_idx

        )
        #print("trt creating moe, layer_idx",layer_idx,"num_experts",len(self.experts))
        #print("trt creating moe, self.experts",type(self.ex)
        #print("trt self.experts",type(self.experts),self.experts)
        #print("trt self.experts[0]",type(self.experts[0]),self.experts[0])
        #print(f"self.experts: {self.experts}")
        #self.gate_input=None
        #self.gate_output=None
    def _create_routing_method(self):
 
        moe_instance = self  
        
        class MiniMaxRoutingMethod(BaseMoeRoutingMethod):
            def __init__(self, top_k):
                super().__init__()
                self.top_k = top_k
                
            def apply(self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
                # Simple top-k routing
                #print("trt router_logits",router_logits.shape,router_logits.dtype)
                moe_instance.router_logits=router_logits
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32 )
                routing_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                # we cast back to the input dtype
                #routing_weights = routing_weights.to(router_logits.dtype)
                # moe_instance.topk_indices=topk_indices
                # moe_instance.topk_values=routing_weights.to(router_logits.dtype)    
                
                #topk_values = topk_values.to(router_logits.dtype)
                return topk_indices.to(torch.int32),routing_weights.to(router_logits.dtype)
                
        return MiniMaxRoutingMethod(self.top_k)
        
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        #self.gate_input=hidden_states.clone()
        router_logits = self.gate(hidden_states)
        #self.gate_output=router_logits.clone()
        output = self.experts(hidden_states, router_logits)
        #print("trt output",output.shape,output.dtype,output)
        return output


class MiniMaxText01DecoderLayer(DecoderLayer):
    """Decoder layer for MiniMaxText01."""
    
    def __init__(
        self,
        model_config: ModelConfig[MiniMaxText01Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        
        # Ensure decoder_attention_types exists
        if not hasattr(config, 'decoder_attention_types') or config.decoder_attention_types is None:
            config.decoder_attention_types = [1] * config.num_hidden_layers
        
        self.attention_type = config.decoder_attention_types[layer_idx]
        #self.input_layernorm_output=None
        # Layer norms
        self.input_layernorm = MiniMaxText01RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            #dtype=config.torch_dtype,
        )
        
        self.post_attention_layernorm = MiniMaxText01RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            #dtype=config.torch_dtype,
        )
        
        # Attention
        if self.attention_type == 0:  # Linear attention
            linear_layer_idx = sum(1 for i in range(layer_idx) 
                                 if config.decoder_attention_types[i] == 0)
            self.self_attn = MiniMaxText01LinearAttention(
                model_config,
                layer_idx=layer_idx,
                linear_layer_idx=linear_layer_idx,
            )
        else:  # Standard attention
            self.self_attn = MiniMaxText01Attention(
                model_config,
                layer_idx=layer_idx,
            )
            
        # MLP or MoE
        if hasattr(config, 'num_local_experts') and isinstance(config.num_local_experts, list):
            expert_num = config.num_local_experts[layer_idx]
        elif hasattr(config, 'num_local_experts'):
            expert_num = config.num_local_experts
        else:
            expert_num = 1
        #print(f"expert_num: {expert_num}")
        if expert_num > 1:
            self.mlp = MiniMaxText01MoE(model_config, layer_idx)
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )
            
        # Shared MLP for MoE
        if expert_num > 1 and config.shared_intermediate_size > 0:
            self.shared_mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
            )
            self.coefficient = Linear(
                config.hidden_size,
                1,
                bias=False,
                dtype=torch.float32,
                mapping=model_config.mapping,
            )
            self.shared_moe_mode = config.shared_moe_mode
        else:
            self.shared_mlp = None
            
        # Layer norm parameters
        if self.attention_type == 0:
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta
        
        self.postnorm = config.postnorm
        
        #self.mlp_input=None
        #self.mlp_output=None
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        kv_cache: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layernorm_input = hidden_states
        #print("trt decoder layer forward layernorm_input",layernorm_input.shape,layernorm_input.dtype)
        #self.input_layernorm_input=layernorm_input.clone()
        layernorm_output = self.input_layernorm(layernorm_input)
        #print("trt decoder layer forward layernorm_output",layernorm_output.shape,layernorm_output.dtype)
        #self.input_layernorm_output=layernorm_output.clone()
        residual = layernorm_output if self.postnorm else layernorm_input
       # print("trt decoder layer forward position_ids",position_ids)
        # Self attention
        # if self.model_config.mapping.tp_rank == 0:
        #     print("trt deocde layer forward hidden_states",hidden_states.shape,hidden_states.dtype)
         
        if self.attention_type == 0:  # Linear attention
            attn_output = self.self_attn(
                position_ids=position_ids,
                hidden_states=layernorm_output,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
                state_indices=state_indices,
                **kwargs,
            )
        else:  # Standard attention
            # Standard attention uses KV cache through attn_metadata
            # The Attention base class will handle KV cache automatically
            attn_output = self.self_attn(
                position_ids=position_ids,
                hidden_states=layernorm_output,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        
        # Apply residual and layer norm
        #residual = residual.to(torch.float32) * self.layernorm_attention_alpha
        #attn_output = (attn_output.to(torch.float32) *self.layernorm_attention_beta)

        
        layernorm_input = residual * self.layernorm_attention_alpha \
                        + attn_output * self.layernorm_attention_beta
        #layernorm_input = layernorm_input.to(torch.bfloat16)
        residual=layernorm_input
        hidden_states = self.post_attention_layernorm(layernorm_input)
        residual = hidden_states if self.postnorm else layernorm_input
        # MLP
        
        #self.mlp_input=hidden_states.clone()
        if self.shared_mlp is not None:
            mlp_output = self.mlp(hidden_states)
            shared_output = self.shared_mlp(hidden_states)
            coef = self.coefficient(hidden_states.to(torch.float32))
            
            if self.shared_moe_mode == 'softmax':
                coef = F.softmax(coef, dim=-1)
            elif self.shared_moe_mode == 'sigmoid':
                coef = torch.sigmoid(coef)
                
            mlp_output = mlp_output * (1 - coef) + shared_output * coef
            mlp_output = mlp_output.to(hidden_states.dtype)
        else:
            mlp_output = self.mlp(hidden_states)
            
        # Apply residual
        #self.mlp_output=mlp_output.clone()
        hidden_states = residual * self.layernorm_mlp_alpha \
                        + mlp_output * self.layernorm_mlp_beta
        # if self.model_config.mapping.tp_rank == 0:
        #     print("trt deocde layer forward hidden_states",hidden_states.shape,hidden_states.dtype)
        return hidden_states, hidden_states


class MiniMaxText01Model(DecoderModel):
    """MiniMaxText01 model."""
    
    def __init__(self, model_config: ModelConfig[MiniMaxText01Config],device:torch.device):
        config = model_config.pretrained_config
        super().__init__(model_config)
        
        # Ensure decoder_attention_types exists
        if not hasattr(config, 'decoder_attention_types') or config.decoder_attention_types is None:
            # Default to all standard attention if not specified
            config.decoder_attention_types = [1] * config.num_hidden_layers
        
        # Count attention layers by type
        self.linear_layer_nums = sum(1 for i in range(config.num_hidden_layers)
                                   if config.decoder_attention_types[i] == 0)
        self.standard_layer_nums = sum(1 for i in range(config.num_hidden_layers)
                                     if config.decoder_attention_types[i] == 1)
        
        # Store decoder attention types for reference
        self.decoder_attention_types = config.decoder_attention_types
        
        self.linear_cache_resource_manager: MiniMaxLinearCacheResourceManager = None 
        # Initialize linear cache manager if we have linear attention layers
        #print("mapping debug tp_rank",self.model_config.mapping.tp_rank,self.model_config.mapping.tp_size,"ep_rank",self.model_config.mapping.moe_ep_rank,"ep_size",self.model_config.mapping.moe_ep_size)
        #exit()
        self.tp_rank=self.model_config.mapping.tp_rank
        self.tp_size=self.model_config.mapping.tp_size
        self.ep_rank=self.model_config.mapping.moe_ep_rank
        self.ep_size=self.model_config.mapping.moe_ep_size
        
        
        self.input_output_dict={}
            # for layer_idx in range(config.num_hidden_layers):
            #     self.input_output_dict[layer_idx]={"input":None,"output":None}
                
        if self.linear_layer_nums > 0:
            tp_size = model_config.mapping.tp_size
            tp_heads = config.num_attention_heads // tp_size
            max_batch_size = model_config.extra_attrs.get('minimax_max_batch_size', 32)
            
            self.linear_cache_resource_manager = MiniMaxLinearCacheResourceManager(
                num_linear_layers=self.linear_layer_nums,
                max_batch_size=max_batch_size,
                num_heads=tp_heads,
                head_dim=config.head_dim,
                dtype=torch.float32,
                device=device,
                enable_cuda_graph=model_config.use_cuda_graph,
                max_context_len=32768,
                tp_rank=self.model_config.mapping.tp_rank,
            )
    
            
        # Note: Standard attention KV cache is managed by TensorRT-LLM framework
        # through AttentionMetadata. Each standard attention layer will automatically
        # handle its own KV cache during forward pass.
        
        # Create model components
        from ..modules.embedding import Embedding
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        
        # Create decoder layers
        
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer = MiniMaxText01DecoderLayer(model_config, layer_idx)
            self.layers.append(layer)
        #print("trt creating layers, self.layers",len(self.layers))
        # Create the final RMSNorm layer
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        # Get embeddings
        # if self.tp_rank == 0:
        #     print("trt forward input_ids",input_ids.shape,input_ids.dtype)
        if self.model_config.mapping.is_first_pp_rank():
            if inputs_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            # Pipeline parallel: get from previous rank
            assert inputs_embeds is not None
            hidden_states = inputs_embeds
            residual = None
        # Handle linear cache if needed
        state_indices = None
        #print("trt forward hidden_states",hidden_states.shape,hidden_states.dtype,hidden_states.mean(),hidden_states.std())
        # every time forward, it will try to get the state_indices from the linear_cache_resource_manager
        if self.linear_cache_resource_manager is not None:
            if hasattr(attn_metadata, 'request_ids') and attn_metadata.request_ids:
                # Use request IDs as sequence IDs for cache management
                seq_ids = attn_metadata.request_ids
                
                # Use the cache manager to get slot indices for seq_ids
                state_indices = self.linear_cache_resource_manager.linear_cache_manager.get_state_indices_from_seq_ids(seq_ids) # build up the slot_indice to req_id
            else:
                raise ValueError("No request_ids found for linear attention cache management")
        else:
             raise ValueError("No linear_cache_resource_manager found")  # Alternative: try to get request info from kwargs this may never use
                # if 'request_ids_to_seq_ids' in kwargs:
                #     request_ids_to_seq_ids = kwargs['request_ids_to_seq_ids']
                #     # Flatten to get all seq_ids
                #     seq_ids = []
                #     for req_id, seq_list in request_ids_to_seq_ids.items():
                #         seq_ids.extend(seq_list)
                    
                #     if seq_ids:
                #         state_indices = linear_cache_manager.get_state_indices_from_seq_ids(seq_ids)
                # else:
                #     state_indices = None
        
        # Process through layers
        linear_layer_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            # Get KV cache for linear attention if needed
            if hasattr(layer, 'attention_type') and layer.attention_type == 0:
                if self.linear_cache_resource_manager is not None:
                    kv_cache = self.linear_cache_resource_manager.linear_cache_manager.get_layer_cache(linear_layer_idx)
                    
                    linear_layer_idx += 1
                else:
                    kv_cache = None
            else:
                kv_cache = None
            
            # Forward through layer
            #record the input output of each layer
           
            # if self.tp_rank == 0 and self.ep_rank == 0:
               
            #     self.input_output_dict[f"layer_{layer_idx}_input"]=hidden_states.detach().float().cpu().numpy()
               
            hidden_states, residual = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                kv_cache=kv_cache,
                state_indices=state_indices,
                **kwargs,
            )
            # if self.tp_rank == 0 and self.ep_rank == 0:
            #     self.input_output_dict[f"layer_{layer_idx}_output"]=hidden_states.detach().float().cpu().numpy() 
            #     if  (layer_idx+1)%8!=0:
            #         self.input_output_dict[f"layer_{layer_idx}_attn_output"]=layer.self_attn.output_proj_res.detach().float().cpu().numpy()
            #         #self.input_output_dict[f"layer_{layer_idx}_layernorm_output"]=layer.input_layernorm_output.detach().float().cpu().numpy()  
            #         #self.input_output_dict[f"layer_{layer_idx}_layernorm_input"]=layer.input_layernorm_input.detach().float().cpu().numpy()  
            #         self.input_output_dict[f"layer_{layer_idx}_lightning_output"]=layer.self_attn.lightning_output.detach().float().cpu().numpy()
            #         self.input_output_dict[f"layer_{layer_idx}_mlp_input"]=layer.mlp_input.detach().float().cpu().numpy()
            #         self.input_output_dict[f"layer_{layer_idx}_mlp_output"]=layer.mlp_output.detach().float().cpu().numpy()
            #         if getattr(layer.self_attn,"q",None) is not None:
            #             self.input_output_dict[f"layer_{layer_idx}_variance"]=layer.self_attn.norm.variance_res.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_q"]=layer.self_attn.q.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_k"]=layer.self_attn.k.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_v"]=layer.self_attn.v.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_attn_input"]=layer.self_attn.input_hidden_states.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_qkv_res"]=layer.self_attn.qkv.detach().float().cpu().numpy()
            #         if getattr(layer.self_attn,"decode_kv",None) is not None:
            #             self.input_output_dict[f"layer_{layer_idx}_decode_kv"]=layer.self_attn.decode_kv.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_decode_kv_state"]=layer.self_attn.decode_kv_state.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_decode_qkv"]=layer.self_attn.decode_qkv.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_decode_q"]=layer.self_attn.decode_q.detach().float().cpu().numpy()
            
            # if layer_idx==0:
            #     print("trt layer_0 ep rank0 exeperts list len",len(layer.mlp.experts))
            #     self.input_output_dict[f"layer_{layer_idx}_mlp_weights_gate_input"]=layer.mlp.gate_input.detach().float().cpu().numpy()
            #     self.input_output_dict[f"layer_{layer_idx}_mlp_weights_gate_output"]=layer.mlp.gate_output.detach().float().cpu().numpy()
            #     self.input_output_dict[f"layer_{layer_idx}_mlp_token_selected_experts"]=layer.mlp.experts.token_selected_experts.detach().float().cpu().numpy()
            #     self.input_output_dict[f"layer_{layer_idx}_mlp_token_final_scales"]=layer.mlp.experts.token_final_scales.detach().float().cpu().numpy()
            #     for i in range(layer.mlp.num_experts):
            #         # Check if the expert is not an Identity module (placeholder for unused experts in EP)
            #         if not isinstance(layer.mlp.experts[i], nn.Identity) and layer.mlp.experts[i].gate_up_proj_res_input is not None:
            #             self.input_output_dict[f"layer_{layer_idx}_mlp_expert_{i}_gate_up_proj_input"]=layer.mlp.experts[i].gate_up_proj_res_input.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_mlp_expert_{i}_gate_up_proj_res"]=layer.mlp.experts[i].gate_up_proj_res.detach().float().cpu().numpy()
            #             self.input_output_dict[f"layer_{layer_idx}_mlp_expert_{i}_down_proj_res"]=layer.mlp.experts[i].down_proj_res.detach().float().cpu().numpy()
                    
            #         else:
            #             print("trt layer_idx",layer_idx,"rank",self.ep_rank,"expert",i,"is an Identity module")
            #     # Get MoE gate (router) weights
            #     if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'weight'):
            #         self.input_output_dict[f"layer_{layer_idx}_moe_gate_weight"]=layer.mlp.gate.weight.detach().float().cpu().numpy()
                
            #     # Get router logits if available (set during forward pass)
            #     if hasattr(layer.mlp, 'router_logits') and layer.mlp.router_logits is not None:
            #         self.input_output_dict[f"layer_{layer_idx}_moe_router_logits"]=layer.mlp.router_logits.detach().float().cpu().numpy()
                    
                #exit()
        # Final layer norm
        if self.model_config.mapping.is_last_pp_rank():
            if residual is not None:
                # Use fused residual connection for better performance
                #print("trt residual",residual.shape,residual.dtype)
                hidden_states, residual = self.norm(hidden_states, residual)
            else:
                #print("trt hidden_states",hidden_states.shape,hidden_states.dtype)
                hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def free_cache_slot(self, seq_id: int):
        """Free a cache slot for a sequence"""
        if self.linear_cache_manager is not None:
            self.linear_cache_manager.free_seq(seq_id)
            
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics"""
        stats = {}
        
        # Linear attention cache info
        if self.linear_cache_manager is not None:
            cache_size_bytes = self.linear_cache_manager.cache.element_size() * \
                             self.linear_cache_manager.cache.numel()
            stats['linear_attention'] = {
                'num_layers': self.linear_layer_nums,
                'max_batch_size': self.linear_cache_manager.max_batch_size,
                'cache_shape': list(self.linear_cache_manager.cache.shape),
                'cache_size_mb': cache_size_bytes / (1024 * 1024),
                'allocated_slots': len(self.linear_cache_manager.seq_id_to_slot_idx),
                'free_slots': len(self.linear_cache_manager.free_slots),
                'seq_id_to_slot_mapping': dict(self.linear_cache_manager.seq_id_to_slot_idx),
            }
            
        # Standard attention info
        if self.standard_layer_nums > 0:
            stats['standard_attention'] = {
                'num_layers': self.standard_layer_nums,
                'type': 'framework_managed',
                'note': 'Standard attention KV cache is managed by TensorRT-LLM framework'
            }
            
        # Overall attention distribution
        stats['attention_distribution'] = {
            'total_layers': len(self.decoder_attention_types),
            'linear_layers': self.linear_layer_nums,
            'standard_layers': self.standard_layer_nums,
            'layer_types': self.decoder_attention_types,
        }
        
        return stats if stats else None


@register_auto_model("MiniMaxText01ForCausalLM")
class MiniMaxText01ForCausalLM(DecoderModelForCausalLM[MiniMaxText01Model, MiniMaxText01Config]):
    """MiniMaxText01 model for causal language modeling."""
    
    # Interface declarations
    has_inner_state: bool = True  # Has linear attention cache
    is_hybrid: bool = True  # Has both linear and standard attention
    supports_v0_only: bool = True  # Currently only supports v0 style
    
    """
    MiniMaxText01 uses a hybrid attention mechanism:
    
    1. Linear Attention (type=0):
       - Uses custom KV cache managed by MiniMaxLinearCacheManager
       - Cache shape: [tp_heads, head_dim, head_dim]
       - Efficient for very long context (up to 1M tokens)
       
    2. Standard Attention (type=1):
       - Uses standard KV cache managed by TensorRT-LLM framework
       - Cache managed through AttentionMetadata
       - Better for local attention patterns
       
    The attention type for each layer is specified in decoder_attention_types.
    """
    
    def __init__(
        self,
        model_config: ModelConfig[MiniMaxText01Config],
        **kwargs
    ):
        
        
        
        # Initialize the model first  
        if isinstance(model_config.pretrained_config.torch_dtype,str):
            model_config.pretrained_config.torch_dtype = getattr(torch, model_config.pretrained_config.torch_dtype)
        # Option 1: Temporarily unfreeze to modify skip_create_weights_in_init
        # This is hack
        was_frozen = getattr(model_config, '_frozen', False)
        if was_frozen:
            model_config._frozen = False
            model_config.skip_create_weights_in_init = False
            model_config._frozen = True
        else:
            model_config.skip_create_weights_in_init = False
            
        self.num_heads=model_config.pretrained_config.num_attention_heads
        self.head_dim=model_config.pretrained_config.head_dim
        self.hidden_size=model_config.pretrained_config.hidden_size
        model = MiniMaxText01Model(model_config,kwargs.get("device",torch.device("cuda")))
        #print("trt creating model finish ....")
        # Call parent class constructor with the model
        super().__init__(
            model,
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        #print("trt creating model super finish ....")
        # Additional attributes for compatibility
        # Note: config, vocab_size, and hidden_size are already set by parent class
        
        # Model initialization complete
        
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,

        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with cache management support"""
        # Forward through model
        # if self.model.tp_rank == 0 and self.model.ep_rank == 0 and attn_metadata.kv_cache_params.num_cached_tokens_per_seq[0]==4:
        #     print("trt input ids is",input_ids)
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        
            
        #first  only fwrd for one token
        # if self.model_config.mapping.tp_rank == 0:
        #     print("trt hidden_states",hidden_states.shape,hidden_states.dtype,hidden_states.device)
        #     exit()
        # Compute logits
        logits = self.compute_logits(hidden_states, attn_metadata, return_context_logits)
        #print("trt save debug attn_metadata",attn_metadata)
        # if self.model.tp_rank == 0 and self.model.ep_rank == 0 and attn_metadata.kv_cache_params.num_cached_tokens_per_seq[0]==4:
        #     #save the per layer input output to a npz file
            
        #     print("trt save layer input output to npz file")
        #     for key,value in self.model.input_output_dict.items():
        #         print("trt key",key,value.shape)
        #     self.model.input_output_dict["logits"]=logits.detach().float().cpu().numpy()
        #     import numpy as np
        #     np.savez(f"layer_input_output.npz",**self.model.input_output_dict)
        return logits
        
    def copy_inputs_before_cuda_graphs(
        self, 
        input_buffers: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for CUDA graph capture"""
        # For now, just return the input buffers as-is
        # CUDA graph support can be added later if needed
        return input_buffers
    
    def get_seqlen_agnostic_capture_inputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get inputs for sequence-length agnostic CUDA graph capture"""
        inputs = {}
        
        # Return empty dict for now
        # CUDA graph support can be added later if needed
        return inputs
    
    def make_empty_intermediate_tensors(
        self, 
        batch_size: int, 
        dtype: torch.dtype,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Create empty intermediate tensors for pipeline parallelism"""
        return {
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "residual": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
        }
    
    def attach_linear_cache_manager(self):
        """Return the linear cache resource manager for registration"""
        return self.model.linear_cache_resource_manager
        
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]):
        """Load weights with MiniMax specific weight names."""
        # Handle MiniMax specific weight naming
        renamed_weights = {}
        
        # Check for tied embeddings
        skip_lm_head = self.config.tie_word_embeddings
        #needs to make the sure all weight are loaded in correct order
        
        
        
        # layer0_mlp_experts0_gate_proj=weights["model.layers.0.block_sparse_moe.experts.0.w1.weight"]
        # layer0_mlp_experts0_up_proj=weights["model.layers.0.block_sparse_moe.experts.0.w3.weight"]
        # layer0_mlp_experts0_gate_up_proj=torch.cat((layer0_mlp_experts0_up_proj,layer0_mlp_experts0_gate_proj),dim=0)
        # print("layer0_mlp_experts0_gate_up_proj",layer0_mlp_experts0_gate_up_proj)
        #exit(-1)
        for name, weight in weights.items():
            # Skip lm_head if embeddings are tied
            if skip_lm_head and "lm_head" in name:
                continue
            # must split q,k,v into three weights andt then combine them into one weight
            # Handle linear attention specific weights
        
            
            if "self_attn" in name:
                #split q,k,v into three weights andt then store them with name as q_proj,k_proj,v_proj instead of qkv_proj 
                # if "qkv_proj" in name:
                #     weight=weight.reshape(self.num_heads,-1,self.hidden_size)
                #     q, k, v = torch.split(weight, [self.head_dim] * 3, dim=1)
                #     #renamed_weights[name] = weight
                # #     print("hf qkv_proj",name,weight.shape)
                # #     #num_heads=64
                # #     #head_dim=128
                    
                #     q_proj_name = name.replace("qkv_proj", "q_proj")
                #     k_proj_name = name.replace("qkv_proj", "k_proj")
                #     v_proj_name = name.replace("qkv_proj", "v_proj")
                #     #weight.shape=[num_heads*3*head_dim,head_dim]
                #     renamed_weights[q_proj_name] = q.reshape(-1,self.hidden_size) #Q ->[num_heads*head_dim,hidden_size]
                #     renamed_weights[k_proj_name] = k.reshape(-1,self.hidden_size) #K ->[num_heads*head_dim,hidden_size]
                #     renamed_weights[v_proj_name] = v.reshape(-1,self.hidden_size) #V ->[num_heads*head_dim,hidden_size]
                #if layer_idx       
                #if is linear attention then rename qkv_proj to linear_qkv_proj
                #print("hf name",name)
                layer_idx=int(name.split(".")[2])
                #print("trt layer_idx",layer_idx)
                #print("trt decoder_attention_types",self.model_config.pretrained_config.decoder_attention_types)
                if layer_idx<len(self.model_config.pretrained_config.decoder_attention_types) and self.model_config.pretrained_config.decoder_attention_types[layer_idx]==0:
           
                    new_name = name.replace("qkv_proj", "linear_qkv_proj")
                    renamed_weights[new_name] = weight
                else:
                    renamed_weights[name] = weight
                #pass
            # Handle MoE specific weights
            elif "block_sparse_moe" in name:
                #print("hf block_sparse_moe",name)
                if self.model_config.moe_backend == "cutlass":
                    new_name = name.replace("block_sparse_moe", "mlp")
                
                    renamed_weights[new_name] = weight
                elif self.model_config.moe_backend == "Vanilla":
                    new_name = name.replace("block_sparse_moe", "mlp")
                    new_name = new_name.replace("w2", "down_proj")
                    new_name = new_name.replace("w1", "gate_proj")
                    new_name = new_name.replace("w3", "up_proj")
                    #print("hf new_name",new_name)
                    renamed_weights[new_name] = weight
                     
                else:
                    raise ValueError(f"Unsupported moe backend: {self.model_config.moe_backend}")
            # Handle shared MoE coefficient
            elif "coefficient.weight" in name and "shared" not in name:
                # This is the shared MoE coefficient weight
                layer_idx = int(name.split('.')[2])  # Extract layer index
                new_name = f"model.layers.{layer_idx}.coefficient.weight"
                
                renamed_weights[new_name] = weight
            # Handle RMSNorm TP weights
            elif "MiniMaxText01RMSNormTP" in name or "norm" in name:
                # Ensure norm weights are handled correctly for TP
                
                renamed_weights[name] = weight
            else:
                renamed_weights[name] = weight
                
        #Let parent class handle the actual loading
        # for name, weight in renamed_weights.items():
        #     # Skip lm_head if embeddings are tied
        #     if "layers.10" in name:
        #     else:
        #         pass
            
        # exit(-1)
                
        return super().load_weights(renamed_weights)
    
    # Cache management API
    def free_cache_slot(self, seq_id: int):
        """Free a cache slot for a sequence"""
        self.model.free_cache_slot(seq_id)
        
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics"""
        return self.model.get_cache_stats()
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the embedding layer"""
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module):
        """Set the embedding layer"""
        self.model.embed_tokens = value
    
    def compute_logits(
        self, 
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        compute_context_logits: bool = False,
    ) -> torch.Tensor:
        """Compute logits from hidden states"""
        return self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,  # attn_metadata
            compute_context_logits,  # return_context_logits
        )
    
    def infer_max_seq_len(self) -> int:
        """Infer maximum sequence length with special handling for linear attention"""
        # Get base max seq len from parent class
        base_max_seq_len = super().infer_max_seq_len()
        
        # MiniMax uses very long context with linear attention
        # Ensure we support the full context length
        if hasattr(self.config, 'max_position_embeddings'):
            return max(base_max_seq_len, self.config.max_position_embeddings)
        
        return base_max_seq_len
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.IntTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step"""
        # This is a helper method for generation pipelines
        # Most logic is handled by the framework, but we provide this for compatibility
        
        # If past_key_values are provided, only use the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        position_ids = kwargs.get('position_ids', None)
        if position_ids is None and input_ids is not None:
            seq_length = input_ids.shape[1]
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]  # Assuming standard KV cache format
            position_ids = torch.arange(
                past_length, seq_length + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
        
@register_auto_model("MiniMaxM1ForCausalLM")
class MiniMaxM1ForCausalLM(MiniMaxText01ForCausalLM):
    def __init__(self, model_config: ModelConfig[MiniMaxText01Config], **kwargs):
        super().__init__(model_config, **kwargs)
