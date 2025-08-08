#!/usr/bin/env python3
"""
Run MiniMaxText01 with TensorRT-LLM PyTorch backend (torchflow)
This demonstrates the correct way to use torchflow - no weight conversion needed!
Supports dummy initialization for testing without actual model weights.
"""
import argparse
import asyncio
import os
import tempfile
import time
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

# Example prompts - 24 prompts for testing
example_prompts = [
    """请详细介绍人工智能的发展历史，从图灵测试的提出开始，到现代深度学习和大语言模型的兴起。在你的回答中，请涵盖以下几个重要的历史节点：1950年代图灵测试和早期AI概念的提出；1960年代专家系统的发展；1980年代神经网络的第一次复兴；1990年代机器学习算法的突破；2000年代支持向量机和集成学习的流行；2010年代深度学习革命的开始，包括卷积神经网络在图像识别上的突破；2017年Transformer架构的提出及其对自然语言处理的革命性影响；2018年BERT模型的发布；2019年GPT-2的震撼发布；2020年GPT-3的问世及其展现的惊人能力；2022年ChatGPT的发布及其对整个AI行业的颠覆性影响；以及2023年以来大语言模型的快速发展和多模态AI的兴起。请在每个历史节点中详细描述当时的技术突破、代表性人物、重要论文、以及对后续发展的影响。同时，请分析每个阶段AI发展面临的主要挑战和限制，以及是如何被后续的技术突破所解决的。最后，请展望未来AI发展的可能方向，包括AGI的实现路径、AI安全问题、以及AI对社会各个领域可能带来的深远影响。""",
    
    """深度学习作为机器学习的一个重要分支，已经成为现代人工智能发展的核心驱动力。请从以下几个维度全面深入地解释深度学习的概念、原理、发展历程和应用：首先，请详细解释什么是深度学习，它与传统机器学习方法的本质区别在哪里，为什么被称为"深度"学习；其次，请深入阐述深度学习的数学基础，包括神经网络的基本结构、前向传播和反向传播算法的数学原理、梯度下降优化算法的工作机制、以及各种激活函数的作用和选择原则；然后，请详细介绍深度学习的主要网络架构，包括多层感知机、卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）、门控循环单元（GRU）、注意力机制、Transformer架构等，并解释每种架构的设计思想、适用场景和优缺点；接下来，请分析深度学习在计算机视觉、自然语言处理、语音识别、推荐系统、游戏AI等领域的具体应用案例，并解释为什么深度学习在这些领域能够取得突破性进展；最后，请讨论深度学习目前面临的主要挑战，如数据需求量大、计算资源消耗高、模型可解释性差、容易过拟合等问题，以及研究者们正在探索的解决方案。""",
    
    """大语言模型（Large Language Models, LLMs）作为当前人工智能领域最引人注目的技术突破，正在深刻改变我们与AI交互的方式。请全面详细地解释大语言模型的工作原理，包括以下几个核心方面：首先，请解释什么是语言模型，从统计语言模型到神经语言模型的演进历程，以及大语言模型的定义和特点；其次，请深入分析Transformer架构作为大语言模型基础的重要性，详细解释自注意力机制的数学原理、多头注意力的设计思想、位置编码的作用、以及编码器-解码器结构与仅解码器结构的区别；然后，请详细描述大语言模型的训练过程，包括预训练阶段的无监督学习、大规模文本数据的处理、训练目标函数的设计、以及分布式训练的技术挑战；接下来，请解释指令微调（Instruction Tuning）和人类反馈强化学习（RLHF）等对齐技术的原理和重要性，以及它们如何让模型更好地理解和执行人类指令；然后，请分析大语言模型的涌现能力（Emergent Abilities），如上下文学习、思维链推理、代码生成等，并解释这些能力是如何随着模型规模的增大而出现的；最后，请讨论大语言模型的局限性和挑战，包括幻觉问题、知识截止、计算成本、安全风险等，以及当前研究界正在探索的改进方向。""",
    
    """机器学习和深度学习虽然都属于人工智能的范畴，但它们之间存在着重要的区别和联系。请从多个角度深入分析这两个概念的异同：首先，请从历史发展的角度解释机器学习和深度学习的起源和演进过程，机器学习作为一个更广泛的概念是如何发展起来的，而深度学习又是在什么背景下从机器学习中分化出来并获得独立发展的；其次，请从技术原理的角度详细比较两者的差异，包括特征工程的重要性差异（传统机器学习需要人工设计特征，而深度学习能够自动学习特征表示）、模型复杂度的差异（深度学习模型通常具有更多的参数和更复杂的结构）、以及学习能力的差异（深度学习在处理高维数据和复杂模式识别方面具有优势）；然后，请分析两者在数据需求、计算资源、训练时间等方面的不同要求，解释为什么深度学习通常需要更大的数据集和更强的计算能力；接下来，请比较两者在不同应用场景下的适用性，什么情况下应该选择传统机器学习方法，什么情况下深度学习更有优势；然后，请详细介绍传统机器学习的主要算法类别，如监督学习（线性回归、逻辑回归、决策树、随机森林、支持向量机等）、无监督学习（聚类、降维等）、强化学习等，以及深度学习的主要网络架构；最后，请展望机器学习和深度学习未来的发展趋势，包括两者的融合发展、新兴技术的出现等。""",
    
    """自然语言处理（Natural Language Processing, NLP）作为人工智能的一个重要分支，致力于让计算机理解、处理和生成人类语言。请全面深入地介绍自然语言处理的各个方面：首先，请解释什么是自然语言处理，它的研究目标和意义，以及为什么让计算机理解人类语言是一个极具挑战性的问题，包括自然语言的歧义性、上下文依赖性、语言的多样性和演化性等特点；其次，请详细介绍NLP的发展历程，从早期的基于规则的方法，到统计方法的兴起，再到深度学习革命对NLP领域的颠覆性影响，特别是Transformer架构和预训练语言模型的出现如何改变了整个领域的研究范式；然后，请系统地介绍NLP的主要任务和技术，包括词法分析（分词、词性标注、命名实体识别）、句法分析（依存句法分析、成分句法分析）、语义分析（词义消歧、语义角色标注、语义解析）、语用分析等基础任务，以及机器翻译、文本摘要、问答系统、对话系统、情感分析、文本分类等应用任务；接下来，请深入分析现代NLP中的关键技术，包括词向量表示（Word2Vec、GloVe等）、预训练语言模型（BERT、GPT系列、T5等）的原理和应用、注意力机制在NLP中的作用、以及大语言模型如何通过规模化实现了通用的语言理解和生成能力；最后，请讨论NLP当前面临的挑战和未来发展方向，包括多语言处理、低资源语言、常识推理、可解释性、偏见和公平性等问题。""",
    
    """计算机视觉（Computer Vision）作为人工智能的重要应用领域，旨在让计算机能够像人类一样"看懂"和理解视觉世界。请全面详细地介绍计算机视觉的应用和发展：首先，请解释计算机视觉的基本概念和研究目标，以及为什么视觉信息处理对人工智能如此重要，包括视觉信息在人类认知中的核心地位、视觉数据的丰富性和复杂性等；其次，请详细介绍计算机视觉的发展历程，从早期的边缘检测和特征提取方法，到传统机器学习时代的SIFT、HOG等手工特征，再到深度学习时代卷积神经网络的革命性突破，特别是AlexNet、VGG、ResNet、Inception等经典网络架构的贡献；然后，请系统地介绍计算机视觉的主要任务和应用领域，包括图像分类（识别图像中的主要对象）、目标检测（定位和识别图像中的多个对象）、语义分割（像素级别的分类）、实例分割（区分同类对象的不同实例）、人脸识别、光学字符识别（OCR）、图像生成、风格迁移等基础任务，以及在自动驾驶、医疗影像分析、安防监控、工业质检、农业监测、文物保护等领域的具体应用案例；接下来，请深入分析现代计算机视觉中的关键技术，包括卷积神经网络的设计原理、注意力机制在视觉任务中的应用、Vision Transformer等新兴架构、多模态学习（结合视觉和语言信息）、生成对抗网络在图像生成中的应用、以及自监督学习在视觉表示学习中的重要作用；最后，请讨论计算机视觉当前面临的挑战和未来发展趋势，包括小样本学习、域适应、3D视觉理解、视频分析、实时处理、隐私保护等问题，以及计算机视觉与其他AI技术融合的发展方向。""",
    
    """强化学习（Reinforcement Learning, RL）作为机器学习的一个重要分支，通过智能体与环境的交互来学习最优策略，在游戏AI、机器人控制、推荐系统等领域取得了显著成功。请全面深入地解释强化学习的概念、原理和应用：首先，请详细解释什么是强化学习，它与监督学习和无监督学习的根本区别，强化学习的基本要素包括智能体（Agent）、环境（Environment）、状态（State）、动作（Action）、奖励（Reward）、策略（Policy）等概念的定义和相互关系；其次，请深入分析强化学习的数学基础，包括马尔可夫决策过程（MDP）的数学框架、贝尔曼方程的推导和意义、价值函数和Q函数的概念、以及最优策略的数学定义；然后，请详细介绍强化学习的主要算法类别，包括基于价值的方法（如Q-learning、SARSA、DQN等）、基于策略的方法（如REINFORCE、Actor-Critic等）、以及模型基础的方法，解释每种方法的核心思想、算法流程、优缺点和适用场景；接下来，请分析深度强化学习的发展，包括深度Q网络（DQN）如何解决了传统强化学习在高维状态空间中的困难、策略梯度方法的改进（PPO、A3C等）、以及AlphaGo、AlphaStar等里程碑式应用的技术原理；然后，请介绍强化学习在各个领域的具体应用案例，包括游戏AI（围棋、电子竞技游戏）、机器人控制（机械臂操作、移动机器人导航）、自动驾驶、金融交易、推荐系统、资源调度等，分析强化学习在这些领域成功的原因和面临的挑战；最后，请讨论强化学习当前的研究热点和未来发展方向，包括多智能体强化学习、层次化强化学习、元学习、安全强化学习、离线强化学习等前沿技术。""",
    
    """神经网络作为深度学习和现代人工智能的基础，其基本概念和原理对理解整个AI领域至关重要。请全面详细地解释神经网络的基本概念：首先，请从生物学角度解释人工神经网络的灵感来源，包括生物神经元的基本结构和工作原理、神经元之间的连接方式、以及大脑神经网络的基本特征，然后说明人工神经网络是如何抽象和简化这些生物特征的；其次，请详细介绍人工神经元（感知机）的数学模型，包括输入、权重、偏置、激活函数等组成部分的作用，以及线性组合和非线性变换的数学表达，解释为什么需要激活函数以及常用激活函数（Sigmoid、Tanh、ReLU、Leaky ReLU、Swish等）的特点和适用场景；然后，请深入分析多层神经网络的结构和原理，包括输入层、隐藏层、输出层的作用，网络深度和宽度对模型能力的影响，以及万能逼近定理如何从理论上保证了神经网络的强大表达能力；接下来，请详细解释神经网络的训练过程，包括前向传播的计算流程、损失函数的设计原则、反向传播算法的数学推导和计算过程、梯度下降及其变种（SGD、Adam、RMSprop等）优化算法的工作机制；然后，请分析神经网络训练中的常见问题和解决方案，包括梯度消失和梯度爆炸问题、过拟合和欠拟合、局部最优解、以及正则化技术（L1/L2正则化、Dropout、Batch Normalization等）的作用原理；最后，请介绍神经网络的发展历程和未来趋势，从最早的感知机到现代的深度神经网络，以及神经架构搜索、可解释AI、神经符号结合等前沿研究方向。"""
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
        default="强化学习（Reinforcement Learning, RL）作为机器学习的一个重要分支，通过智能体与环境的交互来学习最优策略，在游戏AI、机器人控制、推荐系统等领域取得了显著成功。请全面深入地解释强化学习的概念、原理和应用：首先，请详细解释什么是强化学习，它与监督学习和无监督学习的根本区别，强化学习的基本要素包括智能体（Agent）、环境（Environment）、状态（State）、动作（Action）、奖励（Reward）、策略（Policy）等概念的定义和相互关系；其次，请深入分析强化学习的数学基础，包括马尔可夫决策过程（MDP）的数学框架、贝尔曼方程的推导和意义、价值函数和Q函数的概念、以及最优策略的数学定义；然后，请详细介绍强化学习的主要算法类别，包括基于价值的方法（如Q-learning、SARSA、DQN等）、基于策略的方法（如REINFORCE、Actor-Critic等）、以及模型基础的方法，解释每种方法的核心思想、算法流程、优缺点和适用场景；接下来，请分析深度强化学习的发展，包括深度Q网络（DQN）如何解决了传统强化学习在高维状态空间中的困难、策略梯度方法的改进（PPO、A3C等）、以及AlphaGo、AlphaStar等里程碑式应用的技术原理；然后，请介绍强化学习在各个领域的具体应用案例，包括游戏AI（围棋、电子竞技游戏）、机器人控制（机械臂操作、移动机器人导航）、自动驾驶、金融交易、推荐系统、资源调度等，分析强化学习在这些领域成功的原因和面临的挑战；最后，请讨论强化学习当前的研究热点和未来发展方向，包括多智能体强化学习、层次化强化学习、元学习、安全强化学习、离线强化学习等前沿技术。",
        help="Input prompts for generation",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
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
        default=1,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=500,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=600,
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
        default="CUTLASS",
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
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for async generation",
    )
    parser.add_argument(
        "--print-iter-log",
        action="store_true",
        help="Print iteration logs during generation",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token and continue generating",
    )
    
    args = parser.parse_args()
    
    # Use provided prompts or default examples
    prompts = [args.prompts] if args.prompts else example_prompts
    
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
        free_gpu_memory_fraction=0.5,  # Use 90% of free GPU memory for KV cache
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
        "print_iter_log": args.print_iter_log,
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
        temperature=args.temperature if args.temperature > 0 else 0.7,  # Use 0.7 if temperature is 0
        top_k=args.top_k,
        top_p=args.top_p,
        # Add more parameters to control generation
        repetition_penalty=1.1,  # Avoid repetition
        min_tokens=50,  # Generate at least 100 tokens
        ignore_eos=True,  # Control whether to ignore EOS token
        stop_token_ids=None,  # Let the model use its default stop tokens

        # prompt_logprobs在PyTorch后端不支持，已移除
        return_context_logits=True,  # 返回context的logits
        return_generation_logits=True,  # 返回生成的logits
    )
    
    print(f"\nSampling parameters:")
    print(f"  max_tokens: {sampling_params.max_tokens}")
    print(f"  temperature: {sampling_params.temperature}")
    print(f"  top_k: {sampling_params.top_k}")
    print(f"  top_p: {sampling_params.top_p}")
    print(f"  repetition_penalty: {sampling_params.repetition_penalty}")
    print(f"  min_tokens: {sampling_params.min_tokens}")
    print(f"  ignore_eos: {sampling_params.ignore_eos}")
    
    # Generate responses using async for IFB-like inference
    print(f"\nGenerating responses for {len(prompts)} prompts using async (IFB-like)...")
    if args.streaming:
        print("Mode: STREAMING")
    else:
        print("Mode: NON-STREAMING")
    if args.dummy:
        print("NOTE: Output will be random/meaningless due to dummy weights!")
    print("-" * 80)
    
    # Track timing information
    request_times = []
    
    # Async generation function for IFB-like behavior
    async def generate_async_batch():
        if args.streaming:
            # Streaming mode - collect outputs as they stream
            async def stream_prompt(i, prompt):
                start_time = time.time()
                print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Submitting streaming request {i+1}: {prompt[:50]}...")
                final_output = None
                async for output in llm.generate_async(prompt, sampling_params, streaming=True):
                    final_output = output
                    # Optionally print partial outputs
                    # print(f"[Request {i+1}] Partial: {output.outputs[0].text}")
                end_time = time.time()
                request_times.append({
                    'request_id': i+1,
                    'prompt': prompt[:50] + '...',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
                return final_output
            
            tasks = [stream_prompt(i, prompt) for i, prompt in enumerate(prompts)]
            outputs = await asyncio.gather(*tasks)
        else:
            # Non-streaming mode - track individual request completion
            async def track_request(i, prompt):
                start_time = time.time()
                print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Submitting request {i+1}: {prompt[:50]}...")
                output = await llm.generate_async(prompt, sampling_params, streaming=False)
                end_time = time.time()
                
                request_times.append({
                    'request_id': i+1,
                    'prompt': prompt[:50] + '...',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
                print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Request {i+1} completed in {end_time - start_time:.3f}s")
                return output
            
            # Create tasks for all prompts
            #prompts=prompts if isinstance(prompts, list) else [prompts]
            tasks = [track_request(i, prompt) for i, prompt in enumerate(prompts)]
            
            # Wait for all tasks to complete
            print(f"\n[{time.strftime('%H:%M:%S.%f')[:-3]}] Waiting for {len(tasks)} async requests to complete...")
            outputs = await asyncio.gather(*tasks)
        
        return outputs
    
    # Run the async generation
    generation_start_time = time.time()
    outputs = asyncio.run(generate_async_batch())
    generation_end_time = time.time()
    total_generation_time = generation_end_time - generation_start_time
    
    # Print results
    print("\n" + "=" * 80)
    print("GENERATION RESULTS")
    print("=" * 80)
    
    # 保存所有logits
    all_logits = []
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        print("output",output)
        generated_text = output.outputs[0].text
        completion_output = output.outputs[0]
        
        print(f"\n[Prompt {i+1}]")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
        # print(f"Output length: {len(generated_text.split())} words, {len(completion_output.token_ids)} tokens")
        # print(f"Finish reason: {completion_output.finish_reason}")
        if hasattr(completion_output, 'stop_reason'):
            print(f"Stop reason: {completion_output.stop_reason}")
        
        # 获取并显示logits信息
        if hasattr(completion_output, 'logprobs') and completion_output.logprobs:
            print(f"Log probabilities available: {len(completion_output.logprobs)} tokens")
            # 显示前几个token的log prob
            for j, logprob_info in enumerate(completion_output.logprobs[:5]):
                if logprob_info:
                    print(f"  Token {j}: logprob={logprob_info.logprob:.4f}")
        
        # 获取logits
        if hasattr(output, 'context_logits'):
            print(f"Context logits shape: {output.context_logits.shape if output.context_logits is not None else 'None'}")
            all_logits.append({
                'prompt_id': i,
                'context_logits': output.context_logits,
            })
        
        if hasattr(completion_output, 'logits'):
            print(f"Generation logits available: {completion_output.logits is not None}")
            if 'prompt_id' in all_logits[-1]:
                all_logits[-1]['generation_logits'] = completion_output.logits
        
        print("-" * 80)
    
    # # 显示logits内容
    # for logit in all_logits:
    #     print(f"Prompt ID: {logit['prompt_id']}")
    #     print(f"Context Logits: {logit['context_logits']}")
    #     print(f"Generation Logits: {logit['generation_logits']}")
    #     print("-" * 80)
    
    # Print timing statistics
    print("\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"Total generation time: {total_generation_time:.3f} seconds")
    print(f"Number of requests: {len(prompts)}")
    print(f"Average time per request: {total_generation_time/len(prompts):.3f} seconds")
    
    # Print detailed timing for each request
    if request_times:
        print("\nDetailed request timing:")
        print(f"{'Request':<10} {'Start Time':<15} {'End Time':<15} {'Duration (s)':<12} {'Prompt':<50}")
        print("-" * 102)
        
        for req in sorted(request_times, key=lambda x: x['request_id']):
            start_str = time.strftime('%H:%M:%S.%f', time.localtime(req['start_time']))[:-3]
            end_str = time.strftime('%H:%M:%S.%f', time.localtime(req['end_time']))[:-3]
            print(f"{req['request_id']:<10} {start_str:<15} {end_str:<15} {req['duration']:<12.3f} {req['prompt']:<50}")
        
        # Calculate statistics
        durations = [req['duration'] for req in request_times]
        min_duration = min(durations)
        max_duration = max(durations)
        avg_duration = sum(durations) / len(durations)
        
        print("\nRequest duration statistics:")
        print(f"  Min duration: {min_duration:.3f} seconds")
        print(f"  Max duration: {max_duration:.3f} seconds")
        print(f"  Avg duration: {avg_duration:.3f} seconds")
        
        # Calculate throughput
        if args.streaming:
            # For streaming, requests complete at different times
            throughput = len(prompts) / max_duration
        else:
            # For non-streaming, all complete together
            throughput = len(prompts) / total_generation_time
        
        print(f"\nThroughput: {throughput:.2f} requests/second")
    
    print("\nAsync generation completed!")
    
    # Clean up temporary directory if created
    if args.dummy and args.prepare_dummy_dir is None and model_dir.startswith("/tmp/"):
        print(f"\nCleaning up temporary directory: {model_dir}")
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main() 