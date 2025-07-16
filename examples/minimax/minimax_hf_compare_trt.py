#!/usr/bin/env python3
"""
MiniMax 模型单元测试
参考 test_modeling_qwen_moe.py 的结构
"""

import json
import os
import sys
import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from parameterized import parameterized
from transformers import AutoConfig

minimax_model_path = "/Minimax_weight"
# if os.path.exists(minimax_model_path):
#     sys.path.insert(0, minimax_model_path)
#     try:
#         from modeling_minimax_text_01 import MiniMaxText01ForCausalLM as HFMiniMaxForCausalLM
#         print(f"Successfully loaded MiniMax model from {minimax_model_path}")
#     except ImportError as e:
#         print(f"Failed to import MiniMax model from {minimax_model_path}: {e}")
#         # 回退到 transformers 的 AutoModelForCausalLM
#         from transformers import AutoModelForCausalLM as HFMiniMaxForCausalLM
# else:
#     print(f"Path {minimax_model_path} does not exist, using transformers AutoModelForCausalLM")
from transformers import AutoModelForCausalLM as HFMiniMaxForCausalLM

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimax import MiniMaxText01ForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import DecodingCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm._torch.models import AutoModelForCausalLM
import numpy as np

def load_minimax_config(config_path="trtllm_06_17/examples/minimax/minimax_dummy_config/config.json"):
    """从 JSON 文件加载 MiniMax 配置"""
    if not os.path.exists(config_path):
        # 尝试相对路径
        alt_paths = [
            "minimax_dummy_config/config.json",
            "./minimax_dummy_config/config.json",
            "../minimax_dummy_config/config.json",
            "../../minimax_dummy_config/config.json",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Cannot find config file at {config_path} or alternative paths")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 添加一些测试可能需要的默认值
    defaults = {
        "torch_dtype": torch.bfloat16,
        "attention_bias": False,
        "rope_scaling": None,
        "block_size": 64,
        "decoder_attention_types": config_dict.get("attn_type_list", [1] * config_dict.get("num_hidden_layers", 1)),
    }
    
    # 合并默认值
    for key, value in defaults.items():
        if key not in config_dict:
            config_dict[key] = value
    
    # 确保 decoder_attention_types 的长度匹配 num_hidden_layers
    if len(config_dict["decoder_attention_types"]) != config_dict["num_hidden_layers"]:
        # 如果长度不匹配，重复或截断
        num_layers = config_dict["num_hidden_layers"]
        if len(config_dict["decoder_attention_types"]) < num_layers:
            # 重复最后一个值
            last_type = config_dict["decoder_attention_types"][-1] if config_dict["decoder_attention_types"] else 1
            config_dict["decoder_attention_types"].extend([last_type] * (num_layers - len(config_dict["decoder_attention_types"])))
        else:
            # 截断
            config_dict["decoder_attention_types"] = config_dict["decoder_attention_types"][:num_layers]
    
    return config_dict


def create_test_config(base_config_path="minimax_dummy_config/config.json", 
                      overrides=None):
    """
    创建测试配置，允许覆盖特定参数
    
    Args:
        base_config_path: 基础配置文件路径
        overrides: 要覆盖的参数字典
    
    Returns:
        配置字典
    """
    config = load_minimax_config(base_config_path)
    
    if overrides:
        config.update(overrides)
    
    return config


# 从配置文件加载默认配置
try:
    MINIMAX_TEST_CONFIG = load_minimax_config()
    print(f"Successfully loaded MiniMax config from file")
    print(f"Config summary: hidden_size={MINIMAX_TEST_CONFIG['hidden_size']}, "
          f"num_layers={MINIMAX_TEST_CONFIG['num_hidden_layers']}, "
          f"num_heads={MINIMAX_TEST_CONFIG['num_attention_heads']}")
except Exception as e:
    print(f"Failed to load config from file: {e}")
    # 回退到硬编码的配置
    MINIMAX_TEST_CONFIG = {
        "architectures": ["MiniMaxText01ForCausalLM"],
        "vocab_size": 200064,
        "hidden_size": 768,
        "intermediate_size": 1152,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 96,
        "hidden_act": "silu",
        "max_position_embeddings": 1000000,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "sliding_window": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "decoder_attention_types": [0],
        "block_size": 64,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 0,
        "shared_moe_mode": 'softmax',
        "layernorm_linear_attention_alpha": 1.0,
        "layernorm_linear_attention_beta": 1.0,
        "layernorm_full_attention_alpha": 1.0,
        "layernorm_full_attention_beta": 1.0,
        "layernorm_mlp_alpha": 1.0,
        "layernorm_mlp_beta": 1.0,
        "postnorm": False,
        "rotary_dim": None,
        "torch_dtype": torch.bfloat16,
        "model_type": "minimax_text_01",
    }


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False
    use_linear_cache: bool = True  # MiniMax 特有的线性注意力缓存

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-cuda_graph:{self.use_cuda_graph}-linear_cache:{self.use_linear_cache}"


class TestMiniMax(unittest.TestCase):

    def setUp(self):
        """设置测试环境"""
        # 检查是否有可用的 GPU
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        
        # 设置随机种子
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

   

    @parameterized.expand([
        Scenario(backend="TRTLLM"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_minimax_allclose_to_hf(self, scenario: Scenario):
        """
        对比 TensorRT-LLM 和 HuggingFace 实现的输出
        """
        print(f"test_minimax_allclose_to_hf ........ ")
        backend = scenario.backend


        torch.random.manual_seed(0)


        #dtype = torch.bfloat16
        device = torch.device("cuda")
        
        world_size = 8
        layers_per_device = 8 // world_size
        # set device map
        device_map = {
            'model.embed_tokens': 'cuda:0',
            'model.norm': f'cuda:{world_size - 1}',
            'lm_head': f'cuda:{world_size - 1}'
        }
        for i in range(world_size):
            for j in range(layers_per_device):
                device_map[f'model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'
        
        
        # 使用device_map实现逐层加载和offload
        hf_minimax = HFMiniMaxForCausalLM.from_pretrained(
            "/Minimax_weight", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配设备，实现逐层加载
            low_cpu_mem_usage=True,  # 降低CPU内存使用
            offload_folder="./offload_cache",  # 设置offload缓存目录
            offload_state_dict=True  # 启用状态字典offload
        ).eval()
        input_ids = torch.tensor(
            [8685, 1124, 457, 390],
            dtype=torch.int32,
            device=device
        )
  
      

        # 上下文阶段
        position_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0).cuda()
        
        # 调试：逐层对比输出
        print("\n" + "="*80)
        print("Layer-by-layer output comparison:")
        print("="*80)
        
        # 获取中间层输出的钩子
        hf_outputs = {}

        
       
        
        
        # 添加获取输入的钩子函数
      
        def get_hf_hook(name,tp_size=8,num_heads=64,head_dim=128):
            def hook(module, input, output):
                #raise Exception(f"hf_hook: {name}")
                #exit(-1)
                #print(f"hf_hook: {name}")
                # if name.endswith("qkv"):
                #     org_qkv=module.qkv_proj.weight.detach().float().cpu()
                #     print("hf qkv_proj weight",org_qkv.shape)
                #     # org_q=org_qkv[:1024,:]
                #     # org_k=org_qkv[8192:8192+1024,:]
                #     # org_v=org_qkv[8192*2:8192*2+1024,:]                    
                #     hf_outputs[name]=org_qkv[:num_heads*head_dim//tp_size*3,:]
                #     return
                if name.endswith("mlp"):
                    hf_outputs[name+f"_gate_input"]=module.gate_input.detach().float().cpu()
                    hf_outputs[name+f"_gate_output"]=module.router_logits.detach().float().cpu()
                    all_experts=module.experts
                    hf_outputs[name+f"_token_selected_experts"]=module.topk_indices.detach().float().cpu()
                    hf_outputs[name+f"_token_final_scales"]=module.topk_values.detach().float().cpu()
                    hf_outputs[name+f"_output"]=module.mlp_output.detach().float().cpu()
                    # for j in range(len(all_experts)):       
                      
                    #     hf_outputs[name+f"_expert_{j}_gate_up_proj_res_input"]=all_experts[j].gate_up_proj_res_input.clone().detach().float().cpu()
                    #     hf_outputs[name+f"_expert_{j}_gate_up_proj_res"]=all_experts[j].gate_up_proj.clone().detach().float().cpu()
                      
                    #     hf_outputs[name+f"_expert_{j}_down_proj_res"]=all_experts[j].down_proj_res.clone().detach().float().cpu()
              
                    return
                if name.endswith("decode_kv") and getattr(module,"decode_kv",None) is not None:
                    hf_outputs[name]=module.decode_kv.detach().float().cpu()[0,:8,...]
                    return
                if name.endswith("decode_kv_state") and getattr(module,"decode_kv_state",None) is not None:
                    hf_outputs[name]=module.decode_kv_state.detach().float().cpu()[0,:8,...]
                    return
                if name.endswith("decode_qkv") and getattr(module,"decode_qkv",None) is not None:
                    #print("hf decode_qkv shape",module.decode_qkv.shape)
                    hf_outputs[name]=module.decode_qkv.detach().float().cpu()[:,:8,...].squeeze(2)
                    return
                if name.endswith("decode_q") and getattr(module,"decode_q",None) is not None:
                    hf_outputs[name]=module.decode_q.detach().float().cpu()[:,:8,...].squeeze(2)
                    return
                if name.endswith("variance"):
                    hf_outputs[name]=module.variance_res.detach().float().cpu()
                    return
                if name.endswith("lightning_output"):
                    #print("hf lightning_output shape",module.lightning_output.shape)
                    hf_outputs[name]=module.lightning_output.detach().float().cpu()
                    return
                if name.endswith("attn_norm_output"):
                    hf_outputs[name]=module.attn_norm_output.detach().float().cpu().reshape(4,-1)
                    return
                if name.endswith("q"):
                    hf_outputs[name]=module.q.detach().float().cpu()
                    return
                if name.endswith("k"):
                    hf_outputs[name]=module.k.detach().float().cpu()
                if name.endswith("v"):
                    hf_outputs[name]=module.v.detach().float().cpu()
                    return
                if name.endswith("qkv_res"):
                    hf_outputs[name]=module.qkv.detach().float().cpu()[...,:num_heads*head_dim//tp_size*3]
                    return
                if name.endswith("attn_input"):
                    hf_outputs[name]=module.input_hidden_states.detach().float().cpu()
                    return
                if name.endswith("mlp_input"):
                    hf_outputs[name]=module.mlp_input.detach().float().cpu()
                    return
       
                    return
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                try:
                    hf_outputs[name+"_input"] = input_tensor.detach().float().cpu()
                    print(f"hf_hook: {name} input",input_tensor.shape,input_tensor.dtype,input_tensor.mean(),input_tensor.std())
                except:
                    pass
                if name.endswith("layernorm_output"):
                    hf_outputs[name]=module.input_layernorm_output.detach().float().cpu()
                    return
                if name.endswith("layernorm_input"):
                    hf_outputs[name]=module.input_layernorm_input.detach().float().cpu()
                    return
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                hf_outputs[name+"_output"] = output_tensor.detach().float().cpu()
                try:
                    print(f"hf_hook: {name} output",output_tensor.shape,output_tensor.dtype,output_tensor.mean(),output_tensor.std())
                except:
                    pass
            return hook
        
        
        
        # 注册钩子
        # HF 模型钩子
        # HF 模型钩子 - 按操作注册
        hf_minimax.model.embed_tokens.register_forward_hook(get_hf_hook("embed_tokens"))
        for i, layer in enumerate(hf_minimax.model.layers):
            # 注册层输入的钩子
            layer.register_forward_hook(get_hf_hook(f"layer_{i}"))
            layer.block_sparse_moe.register_forward_hook(get_hf_hook(f"layer_{i}_mlp"))
            layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_attn"))
            layer.register_forward_hook(get_hf_hook(f"layer_{i}_mlp_input"))
            if (i+1)%8!=0:
                layer.self_attn.norm.register_forward_hook(get_hf_hook(f"layer_{i}_variance"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_lightning_output"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_attn_input"))
                #layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_attn_norm_output"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_q"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_k"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_v"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_qkv"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_qkv_res"))
                
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_decode_kv"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_decode_kv_state"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_decode_qkv"))
                layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_decode_q"))
                
                layer.register_forward_hook(get_hf_hook(f"layer_{i}_layernorm_output"))
                layer.register_forward_hook(get_hf_hook(f"layer_{i}_layernorm_input"))
        # TRT 模型钩子 - 按操作注册
        
            
            # HF 前向传播
        hf_output = hf_minimax.forward(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids,
            use_cache=True
        )
        print("hf_output",hf_output)
        hf_outputs["logits"]=hf_output.logits[:,-1,:].detach().float().cpu()
        
        from transformers import GenerationConfig,AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/Minimax_weight")
        generation_config = GenerationConfig(
            max_new_tokens=20,
            eos_token_id=200020,
            use_cache=True,
        )
        
        model_inputs = tokenizer("hi how are you", return_tensors="pt").to("cuda")
        print("model_inputs",model_inputs)
        generated_ids = hf_minimax.generate(**model_inputs, generation_config=generation_config)
        print(f"hf generated_ids: {generated_ids}")
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("hf response",response)
            # TRT 前向传播

        trt_outputs=np.load("layer_input_output.npz")
        for k,v in trt_outputs.items():
            print("trt_outputs key",k)
        for k,v in hf_outputs.items():
            print("hf_outputs key",k)
        for name in hf_outputs.keys():
            if name in trt_outputs:
                hf_out = hf_outputs[name]
                trt_out = trt_outputs[name]
                
   
                    #exit(-1)
                # 调整形状以匹配
                if hf_out.shape[0] == 1 and len(hf_out.shape)-len(trt_out.shape)==1:
                    hf_out = hf_out.squeeze(0)
                
                if hf_out.shape == trt_out.shape:
                    hf_out=hf_out.to(torch.float32)
                    trt_out=torch.from_numpy(trt_out).to(torch.float32)
                    diff = (hf_out - trt_out).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    # 计算相对误差
                    hf_abs = hf_out.abs()
                    rel_diff = diff / (hf_abs + 1e-8)  # 添加小值避免除零
                    max_rel_diff = rel_diff.max().item()
                    mean_rel_diff = rel_diff.mean().item()
                    
                    print(f"\n{name}:")
                    print(f"  Shape: {hf_out.shape}")
                    print(f"  Max diff: {max_diff:.10f}")
                    print(f"  Mean diff: {mean_diff:.10f}")
                    print(f"  Max relative diff: {max_rel_diff:.10f}")
                    print(f"  Mean relative diff: {mean_rel_diff:.10f}")
                    print(f"  hf mean{hf_out.mean()}, hf std{hf_out.std()}, hf min{hf_out.min()}, hf max{hf_out.max()}, trt mean{trt_out.mean()}, trt std{trt_out.std()}, trt min{trt_out.min()}, trt max{trt_out.max()}")
                    if max_diff > 1 or max_rel_diff > 1:
                        print(f"  WARNING: Large difference detected!")
                        # 找出绝对差异最大的位置
                        max_diff_idx = diff.argmax()
                        max_diff_idx = np.unravel_index(max_diff_idx.item(), diff.shape)
                        print(f"  Max diff at index {max_diff_idx}: F={hf_out[max_diff_idx].item():.6f}, TRT={trt_out[max_diff_idx].item():.6f}")
                        
                        # 找出相对差异最大的位置
                        max_rel_idx = rel_diff.argmax()
                        max_rel_idx = np.unravel_index(max_rel_idx.item(), rel_diff.shape)
                        print(f"  Max rel diff at index {max_rel_idx}: HF={hf_out[max_rel_idx].item():.6f}, TRT={trt_out[max_rel_idx].item():.6f}")
                else:
                    print(f"\n{name}: Shape mismatch! HF={hf_out.shape}, TRT={trt_out.shape}")
        
        # 比较中间层输入
        print("\n" + "="*80)
        print("Layer-by-layer input comparison:")
        print("="*80)
        
        
        # 最终输出比较
        print("\n" + "="*80)
        print("Final output comparison:")
        print("="*80)
        
        # hf_logits = hf_output.logits[:, -1].float()
        # trt_logits = trt_output.float()
        
        # print(f"HF logits shape: {hf_logits.shape}")
        # print(f"TRT logits shape: {trt_logits.shape}")
        # print(f"HF logits range: [{hf_logits.min().item():.6f}, {hf_logits.max().item():.6f}]")
        # print(f"TRT logits range: [{trt_logits.min().item():.6f}, {trt_logits.max().item():.6f}]")
        
        # # 找出差异最大的几个位置
        # diff = (hf_logits - trt_logits).abs()
        # topk_diff, topk_idx = torch.topk(diff.flatten(), k=10)
        
        # print("\nTop 10 differences:")
        # for i, (d, idx) in enumerate(zip(topk_diff, topk_idx)):
        #     idx = idx.item()
        #     print(f"  {i+1}. Index {idx}: diff={d.item():.6f}, HF={hf_logits.flatten()[idx].item():.6f}, TRT={trt_logits.flatten()[idx].item():.6f}")
        
        # 绘制分布对比和直方图
    #     try:
    #         import matplotlib.pyplot as plt
    #         #import numpy as np
            
    #         # 转换为numpy数组
    #         hf_logits_np = hf_logits.cpu().numpy().flatten()
    #         trt_logits_np = trt_logits.cpu().numpy().flatten()
    #         diff_np = diff.cpu().numpy().flatten()
            
    #         # 创建子图
    #         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    #         fig.suptitle('HF vs TRT Logits 分布对比', fontsize=16)
            
    #         # 1. HF和TRT logits分布对比
    #         axes[0, 0].hist(hf_logits_np, bins=50, alpha=0.7, label='HF Logits', color='blue', density=True)
    #         axes[0, 0].hist(trt_logits_np, bins=50, alpha=0.7, label='TRT Logits', color='red', density=True)
    #         axes[0, 0].set_xlabel('Logits')
    #         axes[0, 0].set_ylabel('Density')
    #         axes[0, 0].set_title('Logits Distribution Comparison')
    #         axes[0, 0].legend()
    #         axes[0, 0].grid(True, alpha=0.3)
            
    #         # 2. 差异分布直方图
    #         axes[0, 1].hist(diff_np, bins=50, alpha=0.7, color='green', density=True)
    #         axes[0, 1].set_xlabel('Absolute Difference')
    #         axes[0, 1].set_ylabel('Density')
    #         axes[0, 1].set_title('Absolute Difference Distribution')
    #         axes[0, 1].grid(True, alpha=0.3)
            
    #         # 3. 散点图对比
    #         sample_indices = np.random.choice(len(hf_logits_np), min(1000, len(hf_logits_np)), replace=False)
    #         axes[1, 0].scatter(hf_logits_np[sample_indices], trt_logits_np[sample_indices], alpha=0.6, s=1)
    #         axes[1, 0].plot([hf_logits_np.min(), hf_logits_np.max()], 
    #                        [hf_logits_np.min(), hf_logits_np.max()], 'r--', label='y=x')
    #         axes[1, 0].set_xlabel('HF Logits')
    #         axes[1, 0].set_ylabel('TRT Logits')
    #         axes[1, 0].set_title('HF vs TRT Scatter Plot')
    #         axes[1, 0].legend()
    #         axes[1, 0].grid(True, alpha=0.3)
            
    #         # 4. 相对误差分布
    #         rel_diff_np = (diff_np / (np.abs(hf_logits_np) + 1e-8))
    #         axes[1, 1].hist(rel_diff_np, bins=50, alpha=0.7, color='orange', density=True)
    #         axes[1, 1].set_xlabel('Relative Error')
    #         axes[1, 1].set_ylabel('Density')
    #         axes[1, 1].set_title('Relative Error Distribution')
    #         axes[1, 1].grid(True, alpha=0.3)
            
    #         plt.tight_layout()
    #         plt.savefig('logits_comparison.png', dpi=300, bbox_inches='tight')
    #         plt.show()
            
    #         # 打印统计信息
    #         print(f"\n分布统计信息:")
    #         print(f"HF Logits - 均值: {hf_logits_np.mean():.6f}, 标准差: {hf_logits_np.std():.6f}")
    #         print(f"TRT Logits - 均值: {trt_logits_np.mean():.6f}, 标准差: {trt_logits_np.std():.6f}")
    #         print(f"绝对差异 - 均值: {diff_np.mean():.6f}, 标准差: {diff_np.std():.6f}")
    #         print(f"相对误差 - 均值: {rel_diff_np.mean():.6f}, 标准差: {rel_diff_np.std():.6f}")
            
    #     except ImportError:
    #         print("matplotlib未安装，跳过绘图")
    #     except Exception as e:
    #         print(f"绘图时出现错误: {e}")
    #     # 对比输出
    #     torch.testing.assert_close(
    #         trt_logits,
    #         hf_logits,
    #         atol=0.1,
    #         rtol=0.1
    #     )

    # def test_linear_attention_cache(self):
    #     """测试线性注意力缓存功能"""
    #     config_dict = deepcopy(MINIMAX_TEST_CONFIG)
    #     config_dict["num_hidden_layers"] = 2
    #     config_dict["decoder_attention_types"] = [0, 0]  # 全部使用线性注意力
        
    #     minimax_config = type('MiniMaxConfig', (), config_dict)()
    #     model_config = ModelConfig(pretrained_config=minimax_config)
        
    #     device = torch.device("cuda")
    #     minimax = MiniMaxText01ForCausalLM(model_config).to(device)
        
    #     # 确保模型附加了线性缓存管理器
    #     minimax.attach_linear_cache_manager()
        
    #     # 创建输入
    #     batch_size = 2
    #     seq_len = 8
    #     input_ids = torch.randint(0, 1000, (batch_size * seq_len,), device=device)
        
    #     # 运行前向传播
    #     with torch.inference_mode():
    #         # 这里需要适当的 metadata 设置
    #         pass
        
    #     print("Linear attention cache test completed")

    # def _sync_weights(self, trtllm_model, hf_model):
    #     """手动同步权重（当自动加载失败时）"""
    #     # 这里实现权重同步逻辑
    #     # 例如：
    #     # - 嵌入层权重
    #     # - 注意力权重
    #     # - MLP 权重
    #     # - 层归一化权重
    #     pass

    # def _sync_weights_detailed(self, trtllm_model, hf_model, weight_mapping):
    #     """详细的手动权重同步"""
    #     print("Attempting detailed weight synchronization...")
        
    #     hf_state = hf_model.state_dict()
    #     trt_state = trtllm_model.state_dict()
    #     for hf_key, hf_value in hf_state.items():
    #         print(f"hf_key: {hf_key}, hf_value: {hf_value.shape}")
    #     for trt_key, trt_value in trt_state.items():
    #         print(f"trt_key: {trt_key}, trt_value: {trt_value.shape}")
    #     # 特别处理一些关键层
    #     key_mappings = {
    #         # Embeddings
    #         "model.embed_tokens.weight": "model.embed_tokens.weight",
    #         "embed_tokens.weight": "model.embed_tokens.weight",
            
    #         # LM head
    #         "lm_head.weight": "lm_head.weight",
            
    #         # Final norm
    #         "model.norm.weight": "model.norm.weight",
    #         "norm.weight": "model.norm.weight",
    #     }
        
    #     # 处理层内的权重
    #     for i in range(trtllm_model.config.num_hidden_layers):
    #         layer_mappings = {
    #             # Layer norms
    #             f"model.layers.{i}.input_layernorm.weight": f"model.layers.{i}.input_layernorm.weight",
    #             f"model.layers.{i}.post_attention_layernorm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
                
    #             # Attention weights
    #             f"model.layers.{i}.self_attn.q_proj.weight": f"model.layers.{i}.self_attn.q_proj.weight",
    #             f"model.layers.{i}.self_attn.k_proj.weight": f"model.layers.{i}.self_attn.k_proj.weight",
    #             f"model.layers.{i}.self_attn.v_proj.weight": f"model.layers.{i}.self_attn.v_proj.weight",
    #             f"model.layers.{i}.self_attn.o_proj.weight": f"model.layers.{i}.self_attn.o_proj.weight",
                
    #             # MLP weights
    #             f"model.layers.{i}.mlp.gate_proj.weight": f"model.layers.{i}.mlp.gate_proj.weight",
    #             f"model.layers.{i}.mlp.up_proj.weight": f"model.layers.{i}.mlp.up_proj.weight",
    #             f"model.layers.{i}.mlp.down_proj.weight": f"model.layers.{i}.mlp.down_proj.weight",
    #         }
    #         key_mappings.update(layer_mappings)
        
    #     # 同步权重
    #     for hf_key, trt_key in key_mappings.items():
    #         if hf_key in hf_state and trt_key in trt_state:
    #             try:
    #                 trt_param = trt_state[trt_key]
    #                 hf_param = hf_state[hf_key]
                    
    #                 if trt_param.shape == hf_param.shape:
    #                     trt_param.copy_(hf_param)
    #                     print(f"✓ Synced {hf_key} -> {trt_key}")
    #                 else:
    #                     print(f"✗ Shape mismatch for {hf_key}: HF={hf_param.shape}, TRT={trt_param.shape}")
    #             except Exception as e:
    #                 print(f"✗ Error syncing {hf_key}: {e}")

    # def test_specific_scenario(self):
    #     # 创建一个特定的测试配置
    #     test_config = create_test_config(
    #         overrides={
    #             "num_hidden_layers": 2,
    #             "decoder_attention_types": [0, 1],  # 第一层线性，第二层标准
    #             "hidden_size": 512,  # 更小的模型用于快速测试
    #         }
    #     )
        
    #     # 使用这个配置创建模型


if __name__ == "__main__":
    unittest.main() 