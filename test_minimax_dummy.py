#!/usr/bin/env python3
"""
Test MiniMaxText01 model with dummy weights (no download required)
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the TensorRT-LLM path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_llm._torch.models.modeling_minimax import (
    MiniMaxText01Config,
    MiniMaxText01ForCausalLM,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm.mapping import Mapping


def create_dummy_config() -> MiniMaxText01Config:
    """Create a small dummy configuration for testing"""
    config = MiniMaxText01Config(
        # Small model for testing
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        
        # MiniMax specific - mix of linear and standard attention
        decoder_attention_types=[0, 1, 0, 1],  # Alternating linear and standard
        block_size=256,
        num_local_experts=2,  # Small MoE for testing
        num_experts_per_tok=1,
        shared_intermediate_size=256,
        shared_moe_mode='softmax',
        
        # Layer norm parameters
        layernorm_linear_attention_alpha=1.0,
        layernorm_linear_attention_beta=0.0,
        layernorm_full_attention_alpha=1.0,
        layernorm_full_attention_beta=0.0,
        layernorm_mlp_alpha=1.0,
        layernorm_mlp_beta=0.0,
        postnorm=False,
        
        # Set torch dtype
        torch_dtype=torch.float16,
    )
    return config


def create_dummy_weights(model: torch.nn.Module, config: MiniMaxText01Config) -> Dict[str, torch.Tensor]:
    """Create dummy weights for the model"""
    weights = {}
    
    # Helper function to create random tensor
    def rand_weight(*shape, dtype=torch.float16):
        return torch.randn(*shape, dtype=dtype) * 0.02
    
    # Embedding weights
    weights['model.embed_tokens.weight'] = rand_weight(config.vocab_size, config.hidden_size)
    
    # Layer weights
    for i in range(config.num_hidden_layers):
        prefix = f'model.layers.{i}'
        
        # Layer norms
        weights[f'{prefix}.input_layernorm.weight'] = torch.ones(config.hidden_size, dtype=torch.float16)
        weights[f'{prefix}.post_attention_layernorm.weight'] = torch.ones(config.hidden_size, dtype=torch.float16)
        
        # Attention weights
        attn_type = config.decoder_attention_types[i]
        if attn_type == 0:  # Linear attention
            # QKV projection
            weights[f'{prefix}.self_attn.qkv_proj.weight'] = rand_weight(
                3 * config.hidden_size, config.hidden_size
            )
            # Output gate
            weights[f'{prefix}.self_attn.output_gate.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
            # Output projection
            weights[f'{prefix}.self_attn.out_proj.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
            # Norm
            weights[f'{prefix}.self_attn.norm.weight'] = torch.ones(
                config.hidden_size, dtype=torch.float16
            )
        else:  # Standard attention
            # Q, K, V projections
            weights[f'{prefix}.self_attn.q_proj.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
            weights[f'{prefix}.self_attn.k_proj.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
            weights[f'{prefix}.self_attn.v_proj.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
            # Output projection
            weights[f'{prefix}.self_attn.o_proj.weight'] = rand_weight(
                config.hidden_size, config.hidden_size
            )
        
        # MLP/MoE weights
        if config.num_local_experts > 1:
            # MoE gate
            weights[f'{prefix}.mlp.gate.weight'] = rand_weight(
                config.num_local_experts, config.hidden_size
            )
            # Expert weights
            for j in range(config.num_local_experts):
                weights[f'{prefix}.mlp.experts.{j}.gate_proj.weight'] = rand_weight(
                    config.intermediate_size, config.hidden_size
                )
                weights[f'{prefix}.mlp.experts.{j}.up_proj.weight'] = rand_weight(
                    config.intermediate_size, config.hidden_size
                )
                weights[f'{prefix}.mlp.experts.{j}.down_proj.weight'] = rand_weight(
                    config.hidden_size, config.intermediate_size
                )
        else:
            # Standard MLP
            weights[f'{prefix}.mlp.gate_proj.weight'] = rand_weight(
                config.intermediate_size, config.hidden_size
            )
            weights[f'{prefix}.mlp.up_proj.weight'] = rand_weight(
                config.intermediate_size, config.hidden_size
            )
            weights[f'{prefix}.mlp.down_proj.weight'] = rand_weight(
                config.hidden_size, config.intermediate_size
            )
        
        # Shared MLP if enabled
        if config.shared_intermediate_size > 0:
            weights[f'{prefix}.shared_mlp.gate_proj.weight'] = rand_weight(
                config.shared_intermediate_size, config.hidden_size
            )
            weights[f'{prefix}.shared_mlp.up_proj.weight'] = rand_weight(
                config.shared_intermediate_size, config.hidden_size
            )
            weights[f'{prefix}.shared_mlp.down_proj.weight'] = rand_weight(
                config.hidden_size, config.shared_intermediate_size
            )
            weights[f'{prefix}.coefficient.weight'] = rand_weight(1, config.hidden_size, dtype=torch.float32)
    
    # Final layer norm
    weights['model.norm.weight'] = torch.ones(config.hidden_size, dtype=torch.float16)
    
    # LM head
    weights['lm_head.weight'] = rand_weight(config.vocab_size, config.hidden_size)
    
    return weights


def test_forward_pass():
    """Test a simple forward pass with dummy data"""
    print("Creating dummy configuration...")
    config = create_dummy_config()
    
    print(f"Configuration summary:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Attention types: {config.decoder_attention_types}")
    print(f"  - Vocab size: {config.vocab_size}")
    
    # Create model config with mapping
    mapping = Mapping(
        world_size=1,
        tp_size=1,
        pp_size=1,
        rank=0,
    )
    
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        max_batch_size=2,
        max_seq_len=1024,
    )
    
    print("\nCreating model...")
    model = MiniMaxText01ForCausalLM(model_config)
    model = model.cuda()
    model.eval()
    
    print("Creating dummy weights...")
    weights = create_dummy_weights(model, config)
    
    print("Loading weights...")
    model.load_weights(weights)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    
    print(f"\nTesting forward pass with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create attention metadata
    attn_metadata = AttentionMetadata(
        num_prefills=batch_size,
        num_decode_tokens=0,
        slot_mapping=torch.arange(batch_size * seq_len, device='cuda'),
        seq_lens=torch.tensor([seq_len] * batch_size, device='cuda'),
        block_tables=None,
        use_cuda_graph=False,
    )
    
    # Create inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size * seq_len,), device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1).flatten()
    
    # Test forward pass
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            print("Running forward pass...")
            logits = model(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
            )
            
            print(f"Output shape: {logits.shape}")
            print(f"Expected shape: ({batch_size * seq_len}, {config.vocab_size})")
            
            # Basic checks
            assert logits.shape == (batch_size * seq_len, config.vocab_size), \
                f"Output shape mismatch: {logits.shape} vs ({batch_size * seq_len}, {config.vocab_size})"
            assert not torch.isnan(logits).any(), "Output contains NaN values"
            assert not torch.isinf(logits).any(), "Output contains Inf values"
            
            print("✅ Forward pass successful!")
    
    # Test cache management
    print("\nTesting cache management...")
    if model.model.linear_cache_manager is not None:
        # Allocate some cache slots
        seq_ids = [0, 1]
        for seq_id in seq_ids:
            model.model.linear_cache_manager.allocate_slot(seq_id, request_id="test_request")
        
        # Get cache stats
        stats = model.get_cache_stats()
        print("Cache statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Free cache
        model.free_cache_request("test_request")
        print("✅ Cache management working!")
    
    print("\n✅ All tests passed!")


def test_generation():
    """Test generation capability"""
    print("\n" + "="*50)
    print("Testing generation capability...")
    
    config = create_dummy_config()
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        max_batch_size=1,
        max_seq_len=100,
    )
    
    model = MiniMaxText01ForCausalLM(model_config)
    model = model.cuda()
    model.eval()
    
    weights = create_dummy_weights(model, config)
    model.load_weights(weights)
    
    # Test with a single token
    batch_size = 1
    prompt_len = 5
    
    # Initial prompt
    input_ids = torch.randint(0, config.vocab_size, (prompt_len,), device='cuda')
    
    print(f"Generating from prompt of length {prompt_len}...")
    
    generated_tokens = []
    current_len = prompt_len
    max_new_tokens = 10
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i in range(max_new_tokens):
                # Create attention metadata for current sequence
                attn_metadata = AttentionMetadata(
                    num_prefills=0 if i > 0 else 1,
                    num_decode_tokens=1 if i > 0 else 0,
                    slot_mapping=torch.arange(current_len, device='cuda'),
                    seq_lens=torch.tensor([current_len], device='cuda'),
                    block_tables=None,
                    use_cuda_graph=False,
                )
                
                # Use only the last token for generation steps
                if i > 0:
                    input_token = generated_tokens[-1].unsqueeze(0)
                    position_id = torch.tensor([current_len - 1], device='cuda')
                else:
                    input_token = input_ids
                    position_id = torch.arange(prompt_len, device='cuda')
                
                # Forward pass
                logits = model(
                    attn_metadata=attn_metadata,
                    input_ids=input_token,
                    position_ids=position_id,
                )
                
                # Get next token (greedy)
                if i > 0:
                    next_token = logits.argmax(dim=-1)
                else:
                    next_token = logits[-1:].argmax(dim=-1)
                
                generated_tokens.append(next_token)
                current_len += 1
                
                print(f"  Generated token {i+1}: {next_token.item()}")
    
    print(f"✅ Successfully generated {len(generated_tokens)} tokens!")


if __name__ == "__main__":
    print("Testing MiniMaxText01 with dummy weights")
    print("="*50)
    
    # Set up environment
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    
    try:
        # Test basic forward pass
        test_forward_pass()
        
        # Test generation
        test_generation()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 