"""
Example script for MiniMaxText01 inference with TensorRT-LLM
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tensorrt_llm._torch.models.modeling_minimax import (
    MiniMaxText01Config, MiniMaxText01ForCausalLM
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata
from tensorrt_llm.logger import logger


def load_pretrained_config(model_path: str) -> MiniMaxText01Config:
    """Load configuration from pretrained model directory"""
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create MiniMaxText01Config from the loaded dict
    config = MiniMaxText01Config(**config_dict)
    
    # Set torch dtype
    config.torch_dtype = torch.float16
    
    return config


def load_pretrained_weights(model: MiniMaxText01ForCausalLM, model_path: str):
    """Load pretrained weights from model directory"""
    import safetensors.torch
    
    model_path = Path(model_path)
    
    # Try to load from safetensors first
    safetensors_files = list(model_path.glob("*.safetensors"))
    
    if safetensors_files:
        logger.info(f"Loading weights from safetensors files: {safetensors_files}")
        weights = {}
        for file in safetensors_files:
            weights.update(safetensors.torch.load_file(file))
    else:
        # Fallback to PyTorch files
        pt_files = list(model_path.glob("*.bin"))
        if not pt_files:
            raise FileNotFoundError(f"No weight files found in {model_path}")
        
        logger.info(f"Loading weights from PyTorch files: {pt_files}")
        weights = {}
        for file in pt_files:
            weights.update(torch.load(file, map_location='cpu'))
    
    # Load weights into model
    model.load_weights(weights)
    logger.info("Weights loaded successfully!")


def generate_text(
    model: MiniMaxText01ForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text from a prompt"""
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    batch_size, seq_len = input_ids.shape
    
    # Flatten for TensorRT-LLM format
    input_ids_flat = input_ids.flatten()
    
    # Initialize generation
    generated_ids = input_ids_flat.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Create position ids
            cur_len = generated_ids.shape[0]
            position_ids = torch.arange(cur_len, device='cuda')
            
            # Create attention metadata
            seq_lens = [cur_len]
            attn_metadata = VanillaAttentionMetadata(
                seq_lens=seq_lens,
                seq_lens_cuda=torch.tensor(seq_lens, device='cuda'),
                query_start_loc=torch.tensor([0], device='cuda'),
                max_query_len=cur_len,
                max_seq_len=cur_len,
                use_cuda_graph=False,
            )
            
            # Forward pass
            hidden_states = model.model(
                attn_metadata=attn_metadata,
                input_ids=generated_ids,
                position_ids=position_ids,
            )
            
            # Get logits for the last token
            last_hidden = hidden_states[-1:, :]
            logits = model.compute_logits(last_hidden, None)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)])
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated_ids.cpu().numpy(), skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='MiniMaxText01 inference example')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to pretrained model directory')
    parser.add_argument('--prompt', type=str, default="Hello, I am",
                        help='Input prompt for generation')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p value for nucleus sampling')
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--pp-size', type=int, default=1,
                        help='Pipeline parallel size')
    
    args = parser.parse_args()
    
    logger.info("Initializing MiniMaxText01 model...")
    
    # Load configuration
    config = load_pretrained_config(args.model_path)
    
    # Create mapping
    mapping = Mapping(
        world_size=args.tp_size * args.pp_size,
        rank=0,  # Assuming single process for this example
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )
    
    # Create model config
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        dtype='float16',
        max_batch_size=1,
        max_input_len=2048,
        max_output_len=512,
    )
    
    # Create model
    model = MiniMaxText01ForCausalLM(model_config)
    model = model.cuda().half()
    
    # Load weights
    logger.info("Loading pretrained weights...")
    load_pretrained_weights(model, args.model_path)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Generate text
    logger.info(f"Generating text from prompt: '{args.prompt}'")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    print(generated_text)
    print("="*50 + "\n")


if __name__ == "__main__":
    main() 