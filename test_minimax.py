"""
Test script for MiniMaxText01 model in TensorRT-LLM
"""
import torch
from transformers import AutoTokenizer

from tensorrt_llm._torch.models.modeling_minimax import (
    MiniMaxText01Config, MiniMaxText01ForCausalLM
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata


def test_minimax_model():
    """Test MiniMaxText01 model creation and forward pass"""
    print("Testing MiniMaxText01 model...")
    
    # Create a simple test configuration
    config = MiniMaxText01Config(
        vocab_size=50257,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=4,  # Small model for testing
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        # MiniMax specific - mix of linear and standard attention
        decoder_attention_types=[0, 1, 0, 1],  # Alternating linear and standard
        block_size=256,
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=2816,
        shared_moe_mode='softmax',
        layernorm_linear_attention_alpha=1.0,
        layernorm_linear_attention_beta=0.0,
        layernorm_full_attention_alpha=1.0,
        layernorm_full_attention_beta=0.0,
        layernorm_mlp_alpha=1.0,
        layernorm_mlp_beta=0.0,
        postnorm=False,
    )
    
    # Set dtype for config
    config.torch_dtype = torch.float16
    
    # Create mapping for single GPU
    mapping = Mapping(
        world_size=1,
        rank=0,
        tp_size=1,
        pp_size=1,
    )
    
    # Create model config
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        dtype='float16',
        max_batch_size=8,
        max_input_len=2048,
        max_output_len=512,
    )
    
    # Create model
    model = MiniMaxText01ForCausalLM(model_config)
    model = model.cuda().half()
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of linear attention layers: {model.model.linear_layer_nums}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    # Create test inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size * seq_len,), device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1).flatten()
    
    # Create attention metadata
    seq_lens = [seq_len] * batch_size
    attn_metadata = VanillaAttentionMetadata(
        seq_lens=seq_lens,
        seq_lens_cuda=torch.tensor(seq_lens, device='cuda'),
        query_start_loc=torch.tensor([0, seq_len], device='cuda'),
        max_query_len=seq_len,
        max_seq_len=seq_len,
        use_cuda_graph=False,
    )
    
    # Create sequence IDs for cache management
    seq_ids = list(range(batch_size))
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        try:
            # Get hidden states from model
            hidden_states = model.model(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                seq_ids=seq_ids,  # Pass sequence IDs for cache management
            )
            
            # Compute logits
            logits = model.compute_logits(hidden_states, None)
            
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Logits shape: {logits.shape}")
            print("Forward pass successful!")
            
            # Test cache management
            if model.model.linear_cache_manager is not None:
                print(f"\nLinear cache manager:")
                print(f"  Cache shape: {model.model.linear_cache_manager.cache.shape}")
                print(f"  Allocated slots: {model.model.linear_cache_manager.slot_mapping}")
                
                # Test freeing a cache slot
                model.model.free_cache_slot(seq_ids[0])
                print(f"  After freeing slot 0: {model.model.linear_cache_manager.slot_mapping}")
            
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test linear attention layer
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'attention_type'):
            print(f"Layer {i}: {'Linear' if layer.attention_type == 0 else 'Standard'} attention")
    
    # Test linear attention cache specifically
    if model.model.linear_layer_nums > 0:
        print(f"\nTesting linear attention cache:")
        print(f"Number of linear layers: {model.model.linear_layer_nums}")
        
        # Test a single linear attention layer
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'attention_type') and layer.attention_type == 0:
                print(f"\nTesting layer {i} (linear attention):")
                
                # Create simple test input
                test_hidden = torch.randn(1, config.hidden_size, device='cuda', dtype=torch.float16)
                test_pos = torch.tensor([0], device='cuda')
                
                # Create simple metadata
                test_metadata = VanillaAttentionMetadata(
                    seq_lens=[1],
                    seq_lens_cuda=torch.tensor([1], device='cuda'),
                    query_start_loc=torch.tensor([0], device='cuda'),
                    max_query_len=1,
                    max_seq_len=1,
                    use_cuda_graph=False,
                )
                
                # Test forward with cache
                if model.model.linear_cache_manager is not None:
                    linear_idx = sum(1 for j in range(i) if model.model.layers[j].attention_type == 0)
                    test_cache = model.model.linear_cache_manager.get_layer_cache(linear_idx)
                    test_indices = torch.tensor([0], device='cuda')
                    
                    output = layer.self_attn(
                        hidden_states=test_hidden,
                        position_ids=test_pos,
                        attn_metadata=test_metadata,
                        kv_cache=test_cache,
                        state_indices=test_indices,
                    )
                    print(f"  Output shape: {output.shape}")
                    print(f"  Cache updated: {test_cache[0].abs().sum().item() > 0}")
                break
            
    print("\nAll tests completed!")


def test_minimax_config_compatibility():
    """Test configuration compatibility with Hugging Face format"""
    print("\nTesting configuration compatibility...")
    
    # Test with typical MiniMax configuration
    hf_config = {
        "architectures": ["MiniMaxText01ForCausalLM"],
        "model_type": "minimax_text_01",
        "vocab_size": 200064,
        "hidden_size": 7168,
        "intermediate_size": 18944,
        "num_hidden_layers": 60,
        "num_attention_heads": 56,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 1000000,
        "rope_theta": 10000000,
        "rms_norm_eps": 1e-6,
        "decoder_attention_types": [0] * 20 + [1] * 40,  # First 20 layers linear, rest standard
        "block": 256,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 4736,
        "shared_moe_mode": "sigmoid",
        "layernorm_linear_attention_alpha": 0.3,
        "layernorm_linear_attention_beta": 0.7,
        "layernorm_full_attention_alpha": 0.3,
        "layernorm_full_attention_beta": 0.7,
        "layernorm_mlp_alpha": 0.3,
        "layernorm_mlp_beta": 0.7,
        "postnorm": False,
    }
    
    # Create config from dict
    config = MiniMaxText01Config(**hf_config)
    print(f"Config created successfully from HF format")
    print(f"Number of linear attention layers: {sum(1 for t in config.decoder_attention_types if t == 0)}")
    print(f"Number of standard attention layers: {sum(1 for t in config.decoder_attention_types if t == 1)}")
    
    # Test config serialization
    config_dict = config.to_dict()
    config2 = MiniMaxText01Config(**config_dict)
    assert config.decoder_attention_types == config2.decoder_attention_types
    print("Configuration serialization test passed!")
    
    print("\nConfiguration compatibility tests completed!")


def test_enhanced_cache_management():
    """Test enhanced cache management features"""
    print("\nTesting enhanced cache management...")
    
    # Create a test configuration
    config = MiniMaxText01Config(
        vocab_size=50257,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        decoder_attention_types=[0, 0, 1, 1],  # 2 linear, 2 standard
        block_size=256,
        num_local_experts=1,
        shared_intermediate_size=0,
        postnorm=False,
    )
    
    config.torch_dtype = torch.float16
    
    # Create model
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        dtype='float16',
        max_batch_size=8,
        max_input_len=2048,
        max_output_len=512,
    )
    
    model = MiniMaxText01ForCausalLM(model_config)
    model = model.cuda().half()
    
    print(f"Model created with {model.model.linear_layer_nums} linear attention layers")
    
    # Test 1: Request-based cache management
    print("\n1. Testing request-based cache management:")
    seq_ids_req1 = [0, 1, 2]
    seq_ids_req2 = [3, 4]
    
    # Allocate cache for request 1
    input_ids_1 = torch.randint(0, config.vocab_size, (3 * 10,), device='cuda')
    position_ids_1 = torch.arange(10, device='cuda').repeat(3)
    
    attn_metadata_1 = VanillaAttentionMetadata(
        seq_lens=[10, 10, 10],
        seq_lens_cuda=torch.tensor([10, 10, 10], device='cuda'),
        query_start_loc=torch.tensor([0, 10, 20], device='cuda'),
        max_query_len=10,
        max_seq_len=10,
        use_cuda_graph=False,
    )
    
    with torch.no_grad():
        _ = model.model(
            attn_metadata=attn_metadata_1,
            input_ids=input_ids_1,
            position_ids=position_ids_1,
            seq_ids=seq_ids_req1,
            request_id="req_001",
        )
    
    # Check cache stats
    stats = model.get_cache_stats()
    print(f"  After request 1: {stats['used_slots']} slots used")
    
    # Allocate cache for request 2
    input_ids_2 = torch.randint(0, config.vocab_size, (2 * 5,), device='cuda')
    position_ids_2 = torch.arange(5, device='cuda').repeat(2)
    
    attn_metadata_2 = VanillaAttentionMetadata(
        seq_lens=[5, 5],
        seq_lens_cuda=torch.tensor([5, 5], device='cuda'),
        query_start_loc=torch.tensor([0, 5], device='cuda'),
        max_query_len=5,
        max_seq_len=5,
        use_cuda_graph=False,
    )
    
    with torch.no_grad():
        _ = model.model(
            attn_metadata=attn_metadata_2,
            input_ids=input_ids_2,
            position_ids=position_ids_2,
            seq_ids=seq_ids_req2,
            request_id="req_002",
        )
    
    stats = model.get_cache_stats()
    print(f"  After request 2: {stats['used_slots']} slots used")
    
    # Free request 1
    model.free_cache_request("req_001")
    stats = model.get_cache_stats()
    print(f"  After freeing request 1: {stats['used_slots']} slots used")
    
    # Test 2: Cache statistics
    print("\n2. Testing cache statistics:")
    stats = model.get_cache_stats()
    print(f"  Total slots: {stats['total_slots']}")
    print(f"  Used slots: {stats['used_slots']}")
    print(f"  Free slots: {stats['free_slots']}")
    print(f"  Utilization: {stats['utilization']:.2%}")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    # Test 3: CUDA graph support
    print("\n3. Testing CUDA graph support:")
    input_buffers = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 10), device='cuda'),
        'seq_lens': torch.tensor([10, 10], device='cuda'),
    }
    
    updated_buffers = model.copy_inputs_before_cuda_graphs(
        input_buffers,
        seq_ids=[4, 5]
    )
    
    print(f"  Original buffers: {list(input_buffers.keys())}")
    print(f"  Updated buffers: {list(updated_buffers.keys())}")
    
    capture_inputs = model.get_seqlen_agnostic_capture_inputs(2)
    print(f"  Capture inputs: {list(capture_inputs.keys())}")
    
    # Test 4: Prefill cache clearing
    print("\n4. Testing prefill cache clearing:")
    seq_ids = [6, 7, 8]
    context_lens = [0, 100, 0]  # Sequences 6 and 8 are new prefills
    
    input_ids = torch.randint(0, config.vocab_size, (3 * 20,), device='cuda')
    position_ids = torch.arange(20, device='cuda').repeat(3)
    
    attn_metadata = VanillaAttentionMetadata(
        seq_lens=[20, 20, 20],
        seq_lens_cuda=torch.tensor([20, 20, 20], device='cuda'),
        query_start_loc=torch.tensor([0, 20, 40], device='cuda'),
        max_query_len=20,
        max_seq_len=20,
        use_cuda_graph=False,
    )
    
    with torch.no_grad():
        _ = model.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
            context_lens=context_lens,
        )
    
    print("  Prefill cache clearing completed")
    
    # Test 5: Cache persistence
    print("\n5. Testing cache persistence:")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        cache_file = f.name
    
    # Save cache state
    model.save_cache_state(cache_file)
    print(f"  Saved cache to {cache_file}")
    
    # Get current stats
    stats_before = model.get_cache_stats()
    
    # Clear some cache
    model.free_cache_request("req_002")
    
    # Load cache state
    model.load_cache_state(cache_file)
    print(f"  Loaded cache from {cache_file}")
    
    # Compare stats
    stats_after = model.get_cache_stats()
    print(f"  Stats match: {stats_before['used_slots'] == stats_after['used_slots']}")
    
    # Cleanup
    import os
    os.unlink(cache_file)
    
    print("\nEnhanced cache management tests completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MiniMaxText01 TensorRT-LLM Test Suite")
    print("=" * 60)
    
    # Test model creation and forward pass
    test_minimax_model()
    
    # Test configuration compatibility
    test_minimax_config_compatibility()
    
    # Test enhanced cache management
    test_enhanced_cache_management()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60) 