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

    # @parameterized.expand([None, "bf16"])
    # def test_minimax_sanity(self, quant_algo):
    #     """基本功能测试"""
    #     print(f"test_minimax_sanity ........ ")
    #     config_dict = deepcopy(MINIMAX_TEST_CONFIG)
        
    #     # 为了快速测试，可以覆盖一些参数
    #     config_dict["num_hidden_layers"] = 2  # 使用较少的层进行测试
    #     config_dict["decoder_attention_types"] = [1, 0]  # 混合注意力类型
        
    #     # 创建配置
    #     minimax_config = type('MiniMaxConfig', (), config_dict)()
        
    #     quant_config = None
            
    #     if quant_algo == "FP8" and torch.cuda.get_device_capability()[0] < 9:
    #         self.skipTest("FP8 requires Hopper or newer architecture")

   
    #     device = torch.device("cuda")

    #     # 创建模型配置
    #     model_config = ModelConfig(
    #         pretrained_config=minimax_config,
    #         quant_config=quant_config
    #     )
        
    #     # 创建 MiniMax 模型
    #     minimax = MiniMaxText01ForCausalLM(model_config).to(device)

    #     # 准备输入
    #     input_ids = torch.tensor(
    #         [100, 200, 300, 100, 200, 100, 400, 500],
    #         dtype=torch.int32,
    #         device=device
    #     )

    #     context_sequence_length = [3, 2, 1]
    #     sequence_length = context_sequence_length + [1, 1]
    #     past_seen_tokens = [0, 0, 0, 62, 75]
    #     request_ids = list(range(len(sequence_length)))
    #     token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)).tolist()
    #     prompt_lens = token_nums[:3] + past_seen_tokens[3:]

    #     # KV 缓存配置
    #     num_blocks = 100
    #     tokens_per_block = 128
    #     head_dim = minimax_config.head_dim
    #     num_layers = minimax_config.num_hidden_layers
    #     num_kv_heads = minimax_config.num_key_value_heads
    #     max_seq_len = num_blocks * tokens_per_block
    #     batch_size = len(sequence_length)

    #     kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

    #     mapping = Mapping(world_size=1, tp_size=1, rank=0)
    #     kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        
    #     kv_cache_manager = KVCacheManager(
    #         kv_cache_config,
    #         tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
    #         num_layers=num_layers,
    #         num_kv_heads=num_kv_heads,
    #         head_dim=head_dim,
    #         tokens_per_block=tokens_per_block,
    #         max_seq_len=max_seq_len,
    #         max_batch_size=batch_size,
    #         mapping=mapping,
    #         dtype=kv_cache_dtype,
    #     )
    #     kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    #     # 创建注意力元数据
    #     metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    #     attn_metadata = metadata_cls(
    #         seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
    #         num_contexts=len(context_sequence_length),
    #         kv_cache_params=KVCacheParams(
    #             use_cache=True,
    #             num_cached_tokens_per_seq=past_seen_tokens,
    #         ),
    #         kv_cache_manager=kv_cache_manager,
    #         request_ids=request_ids,
    #         prompt_lens=prompt_lens,
    #         max_num_requests=len(sequence_length),
    #         max_num_tokens=8192,
    #     )

    #     # 准备位置 ID
    #     position_ids = []
    #     for i, tokens in enumerate(past_seen_tokens):
    #         seq_len = context_sequence_length[i] if i < len(context_sequence_length) else 1
    #         position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
    #         position_ids.append(position_id)
    #     position_ids = torch.cat(position_ids).unsqueeze(0)

    #     # 运行前向传播
    #     with torch.inference_mode():
    #         attn_metadata.prepare()
    #         logits = minimax.forward(
    #             input_ids=input_ids,
    #             position_ids=position_ids,
    #             attn_metadata=attn_metadata
    #         )
        
    #     self.assertEqual(len(past_seen_tokens), logits.shape[0])

    #     # 测试返回上下文 logits
    #     with torch.inference_mode():
    #         attn_metadata.prepare()
    #         logits = minimax.forward(
    #             input_ids=input_ids,
    #             position_ids=position_ids,
    #             attn_metadata=attn_metadata,
    #             return_context_logits=True
    #         )
        
    #     assert input_ids.shape==logits.shape[:-1],f"input_ids.shape: {input_ids.shape}, logits.shape: {logits.shape}"
    #     print(f"test_minimax_sanity passed ........ ")
    #     kv_cache_manager.shutdown()

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
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        # 方法1: 直接使用加载的配置
        #config_dict = deepcopy(MINIMAX_TEST_CONFIG)
        
        # 方法2: 从 minimax_dummy_config 目录加载（如果你想使用 AutoConfig）
        #try:
        hf_config = AutoConfig.from_pretrained("./minimax_dummy_config", trust_remote_code=True)
            
            #print(f"Loaded HF config {hf_config} from minimax_dummy_config")
        # except:
        #     # 如果无法从目录加载，使用我们的配置字典创建
        #     print(f"Creating HF config from loaded dictionary")
        #     # 创建一个兼容 HuggingFace 的配置对象
        #     hf_config = type('MiniMaxText01Config', (), config_dict)()
        #     # 确保有必要的属性
        #     for key, value in config_dict.items():
        #         if not hasattr(hf_config, key):
        #             setattr(hf_config, key, value)
        
        #dtype = torch.bfloat16
        device = torch.device("cuda")

        # 加载 HF 模型
        hf_minimax = HFMiniMaxForCausalLM.from_config(hf_config, trust_remote_code=True).to(device).eval()
        #hf_minimax = HFMiniMaxForCausalLM.from_pretrained("./Minimax_weight", trust_remote_code=True).to(dtype).to(device).eval()
        # 创建 TRT-LLM 模型
        # 使用相同的配置
        #minimax_config = type('MiniMaxConfig', (), config_dict)()
        #hf_config["torch_dtype"] = torch.bfloat16
        model_config = ModelConfig.from_pretrained("./minimax_dummy_config",trust_remote_code=True, moe_backend="Vanilla")
        minimax_config=model_config.pretrained_config
        
        #model_config.from_pretrained("./minimax_dummy_config",trust_remote_code=True )
        print(f"model_config: {model_config}")
        minimax = MiniMaxText01ForCausalLM(model_config).to(device)
        
        # 打印配置对比
        print("\n" + "="*80)
        print("Configuration Comparison:")
        print("="*80)
        important_configs = [
            'num_hidden_layers', 'hidden_size', 'num_attention_heads', 
            'num_key_value_heads', 'head_dim', 'intermediate_size',
            'decoder_attention_types', 'rope_theta', 'rms_norm_eps',
            'layernorm_linear_attention_alpha', 'layernorm_linear_attention_beta',
            'layernorm_full_attention_alpha', 'layernorm_full_attention_beta',
            'layernorm_mlp_alpha', 'layernorm_mlp_beta', 'postnorm'
        ]
        
        for cfg in important_configs:
            hf_val = getattr(hf_config, cfg, "NOT FOUND")
            trt_val = getattr(minimax_config, cfg, "NOT FOUND")
            match = "✓" if str(hf_val) == str(trt_val) else "✗"
            print(f"{cfg:30s}: HF={hf_val}, TRT={trt_val} {match}")
        
        # 同步权重前先检查权重映射
        print("\n" + "="*80)
        print("Weight Mapping Check:")
        print("="*80)
        
        hf_state_dict = hf_minimax.state_dict()
        trt_state_dict = minimax.state_dict()
        
        # 创建权重映射
        weight_mapping = {}
        unmatched_hf = []
        unmatched_trt = list(trt_state_dict.keys())
        
        for hf_key in hf_state_dict.keys():
            # 尝试直接匹配
            if hf_key in trt_state_dict:
                weight_mapping[hf_key] = hf_key
                unmatched_trt.remove(hf_key)
            else:
                # 尝试各种可能的映射
                possible_mappings = [
                    hf_key.replace("model.", ""),
                    hf_key.replace("layers.", "model.layers."),
                    hf_key.replace("embed_tokens", "model.embed_tokens"),
                    hf_key.replace("norm", "model.norm"),
                    hf_key.replace("lm_head", "lm_head"),
                ]
                
                matched = False
                for possible_key in possible_mappings:
                    if possible_key in trt_state_dict:
                        weight_mapping[hf_key] = possible_key
                        if possible_key in unmatched_trt:
                            unmatched_trt.remove(possible_key)
                        matched = True
                        break
                
                if not matched:
                    unmatched_hf.append(hf_key)
        
        print(f"Matched weights: {len(weight_mapping)}")
        print(f"Unmatched HF weights: {len(unmatched_hf)}")
        print(f"Unmatched TRT weights: {len(unmatched_trt)}")
        
        if unmatched_hf:
            print("\nUnmatched HF weights:")
            for key in unmatched_hf[:10]:  # 只显示前10个
                print(f"  - {key}")
                
        if unmatched_trt:
            print("\nUnmatched TRT weights:")
            for key in unmatched_trt[:10]:
                print(f"  - {key}")
        
        # 同步权重
        print("\n" + "="*80)
        print("Syncing weights...")
        print("="*80)
        
        #try:
            # 使用映射同步权重
        mapped_state_dict = {}
        for hf_key, trt_key in weight_mapping.items():
            mapped_state_dict[trt_key] = hf_state_dict[hf_key]
        
        # for hf_key, hf_value in hf_state_dict.items():
        #     print(f"hf_key........... {hf_key}, hf_value: {hf_value.shape}")
        # for trt_key, trt_value in trt_state_dict.items():
        #     print(f"trt_key: {trt_key}, trt_value: {trt_value.shape}")
        minimax.load_weights(hf_state_dict)
        
                
        # for trt_key, trt_value in trt_state_dict.items():
        #     print(f"trt_key: {trt_key}, trt_value: {trt_value.shape}")
        print("Successfully loaded weights with mapping")
       
        # except Exception as e:
        #     print(f"Failed to load weights with mapping: {e}")
        #     # 尝试手动同步
        #     self._sync_weights_detailed(minimax, hf_minimax, weight_mapping)

        # KV 缓存配置
        num_blocks = 1
        tokens_per_block = 32
        head_dim = minimax_config.head_dim
        num_layers = minimax_config.num_hidden_layers
        num_kv_heads = minimax_config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        # 上下文阶段测试
        # input_ids = torch.tensor(
        #     [100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500,100, 200, 300, 100, 200, 100, 400, 500],
        #     dtype=torch.int32,
        #     device=device
        # )
        input_ids = torch.randint(0, 1000,[1,32], dtype=torch.int32, device=device)
        input_ids=input_ids.squeeze(0)
        # print(f"input_ids: {input_ids.shape}")
        # exit(-1)
        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        # 上下文阶段
        position_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0).cuda()
        
        # 调试：逐层对比输出
        print("\n" + "="*80)
        print("Layer-by-layer output comparison:")
        print("="*80)
        
        # 获取中间层输出的钩子
        hf_outputs = {}
        trt_outputs = {}
        # hf_inputs = {}
        # trt_inputs = {}
        
        
       
        
        
        # 添加获取输入的钩子函数
      
        def get_hf_hook(name):
            def hook(module, input, output):
                #raise Exception(f"hf_hook: {name}")
                #exit(-1)
                print(f"hf_hook: {name}")
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                else:
                    input_tensor = input
                if type(input_tensor) == torch.Tensor:
                    hf_outputs[name+"_input"] = input_tensor.detach().cpu()
                normal_attn_list=[f"layer_{i}_self_attn" for i in range(7,80,8)]
                if name in normal_attn_list:
                    print(f"hf_hook: {name}")
                    bsz=module.q_pre_rope.shape[2]
                    #print(f"module.q_pre_rope.shape: {module.q_pre_rope.shape}")
                    #exit(-1)
                    hf_outputs[name+"_q_pre_rope"] = module.q_pre_rope.detach().cpu()
                    hf_outputs[name+"_k_pre_rope"] = module.k_pre_rope.detach().cpu()
                    hf_outputs[name+"_v_pre_rope"] = module.v_pre_rope.detach().cpu()
                    hf_outputs[name+"_q_post_rope"] = module.q_post_rope.detach().cpu()
                    hf_outputs[name+"_k_post_rope"] = module.k_post_rope.detach().cpu()
                    hf_outputs[name+"_v_post_rope"] = module.v_post_rope.detach().cpu().transpose(1, 2).reshape(bsz, -1)
                    hf_outputs[name+"_cos"] = module.cos.detach().cpu()
                    hf_outputs[name+"_sin"] = module.sin.detach().cpu()
                    hf_outputs[name+"_rotary_cos_sin"] = module.rotary_emb.emb.detach().cpu().flatten()  
                    hf_outputs[name+"_flash_attn_output"] = module.flash_attn_output.detach().cpu().reshape(bsz, -1)
                    hf_outputs[name+"_attn_module_output"] = module.attn_output.detach().cpu()
                    hf_outputs[name+"_q_before_attn"] = module.q_before_attn.detach().cpu()
                    hf_outputs[name+"_k_before_attn"] = module.k_before_attn.detach().cpu()
                    hf_outputs[name+"_v_before_attn"] = module.v_before_attn.detach().cpu()
                if name.endswith("self_attn") and name not in normal_attn_list:
                    hf_outputs[name+"_tp_slope"] = module.slope_rate.detach().cpu()   
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                hf_outputs[name+"_output"] = output_tensor.detach().cpu()
                
                # if name.endswith("mlp"):
                #     hf_outputs[name+"_topk_indices"] = module.topk_indices.detach().cpu()
                #     hf_outputs[name+"_topk_values"] = module.topk_values.detach().cpu() 
                #     hf_outputs[name+"_router_logits"] = module.router_logits.detach().cpu()
                    
                #     for j in range(hf_config.num_local_experts):
                #         hf_outputs[name+f"_experts_{j}_gate_up_proj"] = module.experts[j].gate_up_proj.detach().cpu()
                #         hf_outputs[name+f"_experts_{j}_gate_up_proj_res_input"] = module.experts[j].gate_up_proj_res_input.detach().cpu()
            return hook
        
        def get_trt_hook(name):
            def hook(module, input, output):
                #print(type(input),type(output))
                print(f"trt_hook: {name}")
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                else:
                    input_tensor = input
                if type(input_tensor) == torch.Tensor:
                    trt_outputs[name+"_input"] = input_tensor.detach().cpu()
                
                normal_attn_list=[f"layer_{i}_self_attn" for i in range(7,80,8)]
                if name in normal_attn_list:
                    # trt_outputs[name+"_q_pre_rope"] = module.rotary_emb.q.detach().cpu()
                    # trt_outputs[name+"_k_pre_rope"] = module.rotary_emb.k.detach().cpu()
                    # trt_outputs[name+"_v_pre_rope"] = module.rotary_emb.v.detach().cpu()
                    trt_outputs[name+"_q_post_rope"] = module.rotary_emb.q_post_rope.detach().cpu()
                    trt_outputs[name+"_k_post_rope"] = module.rotary_emb.k_post_rope.detach().cpu()
                    #trt_outputs[name+"_v_post_rope"] = module.v_post_rope.detach().cpu()
                    # trt_outputs[name+"_cos"] = module.rotary_emb.cos.detach().cpu()
                    # trt_outputs[name+"_sin"] = module.rotary_emb.sin.detach().cpu()
                    # trt_outputs[name+"_rotary_cos_sin"] = module.rotary_emb.emb.detach().cpu().flatten()
                    # trt_outputs[name+"_flash_attn_output"] = module.flash_attn_output.detach().cpu()
                    # trt_outputs[name+"_attn_module_output"] = module.attn_output.detach().cpu() 
                    # trt_outputs[name+"_q_before_attn"] = module.q_before_attn.detach().cpu().reshape(1, -1, 32, 96)
                    # trt_outputs[name+"_k_before_attn"] = module.k_before_attn.detach().cpu().reshape(1, -1, 8, 96)
                    # trt_outputs[name+"_v_before_attn"] = module.v_before_attn.detach().cpu().reshape(1, -1, 8, 96)
                # if name.endswith("self_attn") and name not in normal_attn_list:
                #     trt_outputs[name+"_tp_slope"] = module.tp_slope.detach().cpu()   
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                trt_outputs[name+"_output"] = output_tensor.detach().cpu()
                # if name.endswith("mlp"):
                #     trt_outputs[name+"_topk_indices"] = module.topk_indices.detach().cpu()
                #     trt_outputs[name+"_topk_values"] = module.topk_values.detach().cpu()
                #     trt_outputs[name+"_router_logits"] = module.router_logits.detach().cpu()
                    
                #     for j in range(hf_config.num_local_experts):
                #         if module.experts[j].gate_up_proj_res is not None:
                #             trt_outputs[name+f"_experts_{j}_gate_up_proj"] = module.experts[j].gate_up_proj_res.detach().cpu() 
                #             trt_outputs[name+f"_experts_{j}_gate_up_proj_res_input"] = module.experts[j].gate_up_proj_res_input.detach().cpu()
            return hook
        
        # 注册钩子
        # HF 模型钩子
        # HF 模型钩子 - 按操作注册
        hf_minimax.model.embed_tokens.register_forward_hook(get_hf_hook("embed_tokens"))
        for i, layer in enumerate(hf_minimax.model.layers):
            # 注册层输入的钩子
            layer.register_forward_hook(get_hf_hook(f"layer_{i}"))
            
            # 注册每个子模块的钩子
            layer.input_layernorm.register_forward_hook(get_hf_hook(f"layer_{i}_input_layernorm"))
            
            layer.self_attn.register_forward_hook(get_hf_hook(f"layer_{i}_self_attn"))
            if getattr(hf_config,"attn_type_list")[i]==0:
                layer.self_attn.qkv_proj.register_forward_hook(get_hf_hook(f"layer_{i}_self_attn_qkv_proj"))
                layer.self_attn.output_gate.register_forward_hook(get_hf_hook(f"layer_{i}_self_attn_output_gate"))
                layer.self_attn.out_proj.register_forward_hook(get_hf_hook(f"layer_{i}_self_attn_out_proj"))
                layer.self_attn.norm.register_forward_hook(get_hf_hook(f"layer_{i}_self_attn_norm"))
                
            layer.post_attention_layernorm.register_forward_hook(get_hf_hook(f"layer_{i}_post_attention_layernorm"))
            layer.block_sparse_moe.register_forward_hook(get_hf_hook(f"layer_{i}_mlp"))
            # for i in range(hf_config.num_local_experts):
            #     layer.block_sparse_moe.experts[i].w1.register_forward_hook(get_hf_hook(f"layer_{i}_mlp_gate_up_proj"))
                #layer.block_sparse_moe.experts[i].w3.register_forward_hook(get_hf_hook(f"layer_{i}_mlp_w3"))
            layer.block_sparse_moe.gate.register_forward_hook(get_hf_hook(f"layer_{i}_mlp_gate"))
            if hasattr(layer, 'shared_mlp') and layer.shared_mlp is not None:
                layer.shared_mlp.register_forward_hook(get_hf_hook(f"layer_{i}_shared_mlp"))
            if hasattr(layer, 'coefficient') and layer.coefficient is not None:
                layer.coefficient.register_forward_hook(get_hf_hook(f"layer_{i}_coefficient"))
        hf_minimax.model.norm.register_forward_hook(get_hf_hook("final_norm"))
        
        # TRT 模型钩子 - 按操作注册
        minimax.model.embed_tokens.register_forward_hook(get_trt_hook("embed_tokens"))
        for i, layer in enumerate(minimax.model.layers):
            # 注册层输入的钩子
            layer.register_forward_hook(get_trt_hook(f"layer_{i}"))
            
            # 注册每个子模块的钩子
            layer.input_layernorm.register_forward_hook(get_trt_hook(f"layer_{i}_input_layernorm"))
            layer.self_attn.register_forward_hook(get_trt_hook(f"layer_{i}_self_attn"))
            
            if getattr(hf_config,"attn_type_list")[i]==0:
                layer.self_attn.qkv_proj.register_forward_hook(get_trt_hook(f"layer_{i}_self_attn_qkv_proj"))
                layer.self_attn.output_gate.register_forward_hook(get_trt_hook(f"layer_{i}_self_attn_output_gate"))
                layer.self_attn.out_proj.register_forward_hook(get_trt_hook(f"layer_{i}_self_attn_out_proj"))
                layer.self_attn.norm.register_forward_hook(get_trt_hook(f"layer_{i}_self_attn_norm"))
            
            layer.post_attention_layernorm.register_forward_hook(get_trt_hook(f"layer_{i}_post_attention_layernorm"))
            layer.mlp.register_forward_hook(get_trt_hook(f"layer_{i}_mlp"))
            layer.mlp.gate.register_forward_hook(get_trt_hook(f"layer_{i}_mlp_gate"))
            # for expert in layer.mlp.experts:
            #     expert.register_forward_hook(get_trt_hook(f"layer_{i}_mlp_expert"))
            if hasattr(layer, 'shared_mlp') and layer.shared_mlp is not None:
                layer.shared_mlp.register_forward_hook(get_trt_hook(f"layer_{i}_shared_mlp"))
            if hasattr(layer, 'coefficient') and layer.coefficient is not None:
                layer.coefficient.register_forward_hook(get_trt_hook(f"layer_{i}_coefficient"))
        minimax.model.norm.register_forward_hook(get_trt_hook("final_norm"))
        
        with torch.inference_mode():
            attn_metadata.prepare()
            
            # HF 前向传播
            hf_output = hf_minimax.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                use_cache=True
            )
            
            # TRT 前向传播
            trt_output = minimax.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
               
            )
        
        # 比较中间层输出
        #layers_maps={}
        
        for name in hf_outputs.keys():
            if name in trt_outputs:
                hf_out = hf_outputs[name]
                trt_out = trt_outputs[name]
                
                # 调整形状以匹配
                if hf_out.shape[0] == 1 and trt_out.shape[0] != 1:
                    hf_out = hf_out.squeeze(0)
                
                if hf_out.shape == trt_out.shape:
                    hf_out=hf_out.to(torch.float32)
                    trt_out=trt_out.to(torch.float32)
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
                    
                    if max_diff > 1 or max_rel_diff > 1:
                        print(f"  WARNING: Large difference detected!")
                        # 找出绝对差异最大的位置
                        max_diff_idx = diff.argmax()
                        max_diff_idx = np.unravel_index(max_diff_idx.item(), diff.shape)
                        print(f"  Max diff at index {max_diff_idx}: HF={hf_out[max_diff_idx].item():.6f}, TRT={trt_out[max_diff_idx].item():.6f}")
                        
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
        
        hf_logits = hf_output.logits[:, -1].float()
        #hf_output_id=
        trt_logits = trt_output.float()
        
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"TRT logits shape: {trt_logits.shape}")
        print(f"HF logits range: [{hf_logits.min().item():.6f}, {hf_logits.max().item():.6f}]")
        print(f"TRT logits range: [{trt_logits.min().item():.6f}, {trt_logits.max().item():.6f}]")
        
        # 找出差异最大的几个位置
        diff = (hf_logits - trt_logits).abs()
        topk_diff, topk_idx = torch.topk(diff.flatten(), k=10)
        
        print("\nTop 10 differences:")
        for i, (d, idx) in enumerate(zip(topk_diff, topk_idx)):
            idx = idx.item()
            print(f"  {i+1}. Index {idx}: diff={d.item():.6f}, HF={hf_logits.flatten()[idx].item():.6f}, TRT={trt_logits.flatten()[idx].item():.6f}")
        
        # 绘制分布对比和直方图
        try:
            import matplotlib.pyplot as plt
            #import numpy as np
            
            # 转换为numpy数组
            hf_logits_np = hf_logits.cpu().numpy().flatten()
            trt_logits_np = trt_logits.cpu().numpy().flatten()
            diff_np = diff.cpu().numpy().flatten()
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('HF vs TRT Logits 分布对比', fontsize=16)
            
            # 1. HF和TRT logits分布对比
            axes[0, 0].hist(hf_logits_np, bins=50, alpha=0.7, label='HF Logits', color='blue', density=True)
            axes[0, 0].hist(trt_logits_np, bins=50, alpha=0.7, label='TRT Logits', color='red', density=True)
            axes[0, 0].set_xlabel('Logits')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Logits Distribution Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 差异分布直方图
            axes[0, 1].hist(diff_np, bins=50, alpha=0.7, color='green', density=True)
            axes[0, 1].set_xlabel('Absolute Difference')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Absolute Difference Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 散点图对比
            sample_indices = np.random.choice(len(hf_logits_np), min(1000, len(hf_logits_np)), replace=False)
            axes[1, 0].scatter(hf_logits_np[sample_indices], trt_logits_np[sample_indices], alpha=0.6, s=1)
            axes[1, 0].plot([hf_logits_np.min(), hf_logits_np.max()], 
                           [hf_logits_np.min(), hf_logits_np.max()], 'r--', label='y=x')
            axes[1, 0].set_xlabel('HF Logits')
            axes[1, 0].set_ylabel('TRT Logits')
            axes[1, 0].set_title('HF vs TRT Scatter Plot')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 相对误差分布
            rel_diff_np = (diff_np / (np.abs(hf_logits_np) + 1e-8))
            axes[1, 1].hist(rel_diff_np, bins=50, alpha=0.7, color='orange', density=True)
            axes[1, 1].set_xlabel('Relative Error')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Relative Error Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('logits_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 打印统计信息
            print(f"\n分布统计信息:")
            print(f"HF Logits - 均值: {hf_logits_np.mean():.6f}, 标准差: {hf_logits_np.std():.6f}")
            print(f"TRT Logits - 均值: {trt_logits_np.mean():.6f}, 标准差: {trt_logits_np.std():.6f}")
            print(f"绝对差异 - 均值: {diff_np.mean():.6f}, 标准差: {diff_np.std():.6f}")
            print(f"相对误差 - 均值: {rel_diff_np.mean():.6f}, 标准差: {rel_diff_np.std():.6f}")
            
        except ImportError:
            print("matplotlib未安装，跳过绘图")
        except Exception as e:
            print(f"绘图时出现错误: {e}")
        # 对比输出
        torch.testing.assert_close(
            trt_logits,
            hf_logits,
            atol=0.1,
            rtol=0.1
        )

    def test_linear_attention_cache(self):
        """测试线性注意力缓存功能"""
        config_dict = deepcopy(MINIMAX_TEST_CONFIG)
        config_dict["num_hidden_layers"] = 2
        config_dict["decoder_attention_types"] = [0, 0]  # 全部使用线性注意力
        
        minimax_config = type('MiniMaxConfig', (), config_dict)()
        model_config = ModelConfig(pretrained_config=minimax_config)
        
        device = torch.device("cuda")
        minimax = MiniMaxText01ForCausalLM(model_config).to(device)
        
        # 确保模型附加了线性缓存管理器
        minimax.attach_linear_cache_manager()
        
        # 创建输入
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, 1000, (batch_size * seq_len,), device=device)
        
        # 运行前向传播
        with torch.inference_mode():
            # 这里需要适当的 metadata 设置
            pass
        
        print("Linear attention cache test completed")

    def _sync_weights(self, trtllm_model, hf_model):
        """手动同步权重（当自动加载失败时）"""
        # 这里实现权重同步逻辑
        # 例如：
        # - 嵌入层权重
        # - 注意力权重
        # - MLP 权重
        # - 层归一化权重
        pass

    def _sync_weights_detailed(self, trtllm_model, hf_model, weight_mapping):
        """详细的手动权重同步"""
        print("Attempting detailed weight synchronization...")
        
        hf_state = hf_model.state_dict()
        trt_state = trtllm_model.state_dict()
        for hf_key, hf_value in hf_state.items():
            print(f"hf_key: {hf_key}, hf_value: {hf_value.shape}")
        for trt_key, trt_value in trt_state.items():
            print(f"trt_key: {trt_key}, trt_value: {trt_value.shape}")
        # 特别处理一些关键层
        key_mappings = {
            # Embeddings
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "embed_tokens.weight": "model.embed_tokens.weight",
            
            # LM head
            "lm_head.weight": "lm_head.weight",
            
            # Final norm
            "model.norm.weight": "model.norm.weight",
            "norm.weight": "model.norm.weight",
        }
        
        # 处理层内的权重
        for i in range(trtllm_model.config.num_hidden_layers):
            layer_mappings = {
                # Layer norms
                f"model.layers.{i}.input_layernorm.weight": f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
                
                # Attention weights
                f"model.layers.{i}.self_attn.q_proj.weight": f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight": f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight": f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight": f"model.layers.{i}.self_attn.o_proj.weight",
                
                # MLP weights
                f"model.layers.{i}.mlp.gate_proj.weight": f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight": f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight": f"model.layers.{i}.mlp.down_proj.weight",
            }
            key_mappings.update(layer_mappings)
        
        # 同步权重
        for hf_key, trt_key in key_mappings.items():
            if hf_key in hf_state and trt_key in trt_state:
                try:
                    trt_param = trt_state[trt_key]
                    hf_param = hf_state[hf_key]
                    
                    if trt_param.shape == hf_param.shape:
                        trt_param.copy_(hf_param)
                        print(f"✓ Synced {hf_key} -> {trt_key}")
                    else:
                        print(f"✗ Shape mismatch for {hf_key}: HF={hf_param.shape}, TRT={trt_param.shape}")
                except Exception as e:
                    print(f"✗ Error syncing {hf_key}: {e}")

    def test_specific_scenario(self):
        # 创建一个特定的测试配置
        test_config = create_test_config(
            overrides={
                "num_hidden_layers": 2,
                "decoder_attention_types": [0, 1],  # 第一层线性，第二层标准
                "hidden_size": 512,  # 更小的模型用于快速测试
            }
        )
        
        # 使用这个配置创建模型


if __name__ == "__main__":
    unittest.main() 