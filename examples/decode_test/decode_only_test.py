import unittest
from dataclasses import dataclass

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine,LoadFormat
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             ResourceManager)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.executor import KvCacheConfig, ExecutorConfig
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.mapping import Mapping
# 添加 KvCacheCreator 的导入
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

# 导入并应用补丁
import executor_config_patch

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams




def _create_request(num_tokens, req_id: int):
    sampling_params = SamplingParams()
    kwargs = {
        "request_id":
        req_id,
        "max_new_tokens":
        1,
        "input_tokens": [0] * num_tokens,
        "sampling_config":
        tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        "is_streaming":
        False,
    }
    result = LlmRequest(**kwargs)
    result.state = LlmRequestState.GENERATION_IN_PROGRESS
    result.context_current_position = num_tokens
    #result.state = LlmRequestState.GENERATION_IN_PROGRESS
    result.py_prompt_len = num_tokens
    result.context_current_position = result.py_prompt_len
    
    result.add_new_token(0,0)
    result.decoding_iter = 1
    result.py_decoding_iter = 1
    #result.py_decoding_iter = 0
    result.paged_kv_block_ids = []
    return result




def create_model_engine_and_kvcache(config: PyTorchConfig = None,max_tokens:int=1024,batch_size:int=32,tokens_per_block:int=128):
    


    config = config if config else PyTorchConfig(load_format=LoadFormat.AUTO)
   
    mapping = Mapping(
        world_size=8,
        tp_size=8,
        moe_ep_size=8,
        rank=tensorrt_llm.mpi_rank(),
        gpus_per_node=torch.cuda.device_count()
    )
    # device_id = tensorrt_llm.mpi_rank() % torch.cuda.device_count()
    # torch.cuda.set_device(device_id)
    
    model_engine = PyTorchModelEngine(model_path="/ds_weight",
                                      pytorch_backend_config=config,
                                      batch_size=batch_size,
                                      max_num_tokens=max_tokens,
                                      max_seq_len=max_tokens,
                                      mapping=mapping)
    
    # 创建 KvCacheConfig
    kv_cache_config = KvCacheConfig(
        max_tokens=max_tokens*batch_size
    )
    
    # 创建 ExecutorConfig - 使用构造函数的可选参数
    executor_config = ExecutorConfig(
        kv_cache_config=kv_cache_config,
        max_batch_size=batch_size,
        max_num_tokens=max_tokens,
        #speculative_config=None
    )
    
   
    executor_config.max_seq_len = max_tokens
    executor_config.tokens_per_block = tokens_per_block
    
    # 使用 KvCacheCreator 创建 KV cache
    kv_cache_creator = KvCacheCreator(
        executor_config=executor_config,
        model_engine=model_engine,
        draft_model_engine=None,  # 没有 draft model
        mapping=mapping,
        net_max_seq_len=max_tokens
    )
    
    # 创建资源字典
    resources = {}
    
    # 构建 KV cache managers
    kv_cache_creator.build_managers(resources)
    
    # 从资源中获取 kv_cache_manager
    from tensorrt_llm._torch.pyexecutor.model_engine import KV_CACHE_MANAGER_KEY
    kv_cache_manager = resources[KV_CACHE_MANAGER_KEY]
    
    print(f"model_engine.model.config {model_engine.model.config}")
    print(f"KV cache manager created successfully: {kv_cache_manager}")
    
    return model_engine, kv_cache_manager, resources

def decode_only_test(isl,batch_size:int=32,tokens_per_block:int=128):

    max_tokens=isl+1
    device_id = tensorrt_llm.mpi_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    model_engine, kv_cache_manager, resources = create_model_engine_and_kvcache(max_tokens=max_tokens,batch_size=batch_size,tokens_per_block=tokens_per_block)
    
    
    esource_manager = ResourceManager(
            {"kv_cache_manager": kv_cache_manager})

    request_ids = list(range(batch_size))
    requests = [_create_request(1024, request_id) for request_id in request_ids]

    batch = ScheduledRequests()
    batch.context_requests = []
    batch.generation_requests = requests
 
    
    token_nums = [isl]*batch_size
    prompt_lens = [isl]*batch_size
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    print("forward model")
    output=model_engine.forward(batch,esource_manager)
    print(f"output: {output}, output.logits.shape: {output['logits'].shape}")

if __name__ == "__main__":
    decode_only_test(isl=1024)
