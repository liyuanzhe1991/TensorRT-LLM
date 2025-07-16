#!/usr/bin/env python3
"""
Run MiniMaxText01 with TensorRT-LLM PyTorch backend (torchflow)
This demonstrates the correct way to use torchflow - no weight conversion needed!
Supports dummy initialization for testing without actual model weights.
"""
import argparse
import os
import tempfile
from pathlib import Path
from typing import List

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import KvCacheConfig

# Try to import transformers for config preparation
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Example prompts
example_prompts = [
    "你好，请介绍一下人工智能的发展历史。",
    "什么是深度学习？",
    "解释一下大语言模型的工作原理。",
]


def prepare_dummy_model_dir(model_name: str, output_dir: str = None) -> str:
    """
    Prepare a directory with only config and tokenizer files for dummy initialization.
    
    Args:
        model_name: HuggingFace model ID to get config from
        output_dir: Optional output directory, uses temp dir if not specified
    
    Returns:
        Path to the prepared directory
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for preparing dummy model directory")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="minimax_dummy_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preparing dummy model directory at: {output_dir}")
    
    try:
        # Download and save config
        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.save_pretrained(output_dir)
        
        # Download and save tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Successfully prepared dummy model directory with config and tokenizer")
        return output_dir
    except Exception as e:
        print(f"Error preparing dummy model directory: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run MiniMaxText01 with TensorRT-LLM PyTorch backend"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/Minimax_weight/",
        help="Model name or path (can be HuggingFace model ID)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy weights initialization (no actual model weights needed)",
    )
    parser.add_argument(
        "--prepare-dummy-dir",
        type=str,
        default=None,
        help="Directory to save config/tokenizer files for dummy initialization",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default="hi how are you",
        help="Input prompts for generation",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling parameter",
    )
    
    # Parallelism configuration
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallelism size",
    )
    parser.add_argument(
        "--moe-ep-size",
        type=int,
        default=8,
        help="MoE expert parallelism size",
    )
    
    # Runtime configuration
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=100,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens in a batch",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8", "fp16", "fp32"],
        help="KV cache data type",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="TRTLLM",
        choices=["VANILLA", "TRTLLM", "FLASHINFER"],
        help="Attention backend to use",
    )
    parser.add_argument(
        "--moe-backend",
        type=str,
        default="Vanilla",
        choices=["CUTLASS", "TRTLLM", "Vanilla"],
        help="MoE backend to use",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Use CUDA graphs for optimization",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    
    args = parser.parse_args()
    
    # Use provided prompts or default examples
    prompts = args.prompts if args.prompts else example_prompts
    
    print("=" * 80)
    print("MiniMaxText01 - TensorRT-LLM PyTorch Backend (torchflow)")
    if args.dummy:
        print("Mode: DUMMY WEIGHTS INITIALIZATION")
    print("=" * 80)
    print(f"Model: {args.model_dir}")
    print(f"Tensor Parallel: {args.tp_size}")
    print(f"MoE Expert Parallel: {args.moe_ep_size}")
    print(f"Backend: PyTorch (torchflow)")
    print("=" * 80)
    
    # Handle dummy initialization
    model_dir = args.model_dir
    if args.dummy:
        # Check if model_dir is a local directory with config files
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
            print(f"\nUsing existing config directory: {model_dir}")
        else:
            # Prepare dummy directory from HuggingFace model
            print(f"\nPreparing dummy model directory from: {model_dir}")
            model_dir = prepare_dummy_model_dir(model_dir, args.prepare_dummy_dir)
    
    # Initialize LLM with PyTorch backend
    # This is the key difference - we use backend='pytorch' and the framework
    # handles everything automatically!
    print("\nInitializing LLM...")
    
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.1,  # Use 90% of free GPU memory for KV cache
    )
    
    # Build LLM kwargs
    llm_kwargs = {
        "model": model_dir,
        "backend": 'pytorch',  # This is the key - use PyTorch backend!
        
        # Parallelism configuration
        "tensor_parallel_size": args.tp_size,
        "pipeline_parallel_size": args.pp_size,
        "moe_expert_parallel_size": args.moe_ep_size,
        
        # Model configuration
        "max_seq_len": args.max_seq_len,
        "max_batch_size": args.max_batch_size,
        "max_num_tokens": args.max_num_tokens,
        
        # KV cache configuration
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_cache_config": kv_cache_config,
        
        # Backend configuration
        "attn_backend": args.attention_backend,
        "moe_backend": args.moe_backend,
        
        # Runtime optimization
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "use_cuda_graph": args.use_cuda_graph,
        
        # Other options
        "trust_remote_code": args.trust_remote_code,
    }
    print("llm_kwargs",llm_kwargs)
    # Add load_format for dummy initialization
    if args.dummy:
        llm_kwargs["load_format"] = "dummy"
        print("Using dummy weights (random initialization)")
    
    #print(llm_kwargs)
    llm = LLM(**llm_kwargs)
    
    print("LLM initialized successfully!")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Generate responses
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    if args.dummy:
        print("NOTE: Output will be random/meaningless due to dummy weights!")
    print("-" * 80)
    
    outputs = llm.generate(prompts, sampling_params)
    print("trt output ",outputs)
    # Print results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\n[Prompt {i+1}]")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
        print("-" * 80)
    
    print("\nGeneration completed!")
    
    # Clean up temporary directory if created
    if args.dummy and args.prepare_dummy_dir is None and model_dir.startswith("/tmp/"):
        print(f"\nCleaning up temporary directory: {model_dir}")
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main() 