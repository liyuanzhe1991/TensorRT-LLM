/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/linearAttention/linearAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>

namespace torch_ext
{

void linear_attention_prefill(
    torch::Tensor output,
    torch::Tensor state,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seq_lens,
    int64_t num_seqs,        // Changed from int32_t to int64_t
    int64_t num_qo_heads,    // Changed from int32_t to int64_t
    int64_t num_kv_heads,    // Changed from int32_t to int64_t
    int64_t head_size,       // Changed from int32_t to int64_t
    double scale,            // Changed from float to double
    double decay)            // Changed from float to double
{
    // Check tensor types
    TORCH_CHECK(q.dtype() == torch::kFloat16, "Q tensor must be float16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "K tensor must be float16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "V tensor must be float16");
    TORCH_CHECK(output.dtype() == torch::kFloat16, "Output tensor must be float16");
    TORCH_CHECK(state.dtype() == torch::kFloat32, "State tensor must be float32");
    TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt64, "cu_seq_lens must be int64");
    
    // Check contiguity
    TORCH_CHECK(q.is_contiguous(), "Q tensor must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K tensor must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V tensor must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");
    TORCH_CHECK(state.is_contiguous(), "State tensor must be contiguous");
    TORCH_CHECK(cu_seq_lens.is_contiguous(), "cu_seq_lens must be contiguous");
    
    // Check CUDA
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA");
    TORCH_CHECK(state.is_cuda(), "State tensor must be on CUDA");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be on CUDA");
    
    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Get data pointers - using half for float16
    auto output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    auto state_ptr = state.data_ptr<float>();
    auto q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    auto k_ptr = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    auto v_ptr = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    auto cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();

    // Call the CUDA kernel with proper type conversions
    tensorrt_llm::kernels::launchLinearAttentionPrefillKernel<half, half, float>(
        stream,
        output_ptr,
        state_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        cu_seq_lens_ptr,
        static_cast<int32_t>(num_seqs),       // Convert to int32_t for kernel
        static_cast<int32_t>(num_qo_heads),    // Convert to int32_t for kernel
        static_cast<int32_t>(num_kv_heads),    // Convert to int32_t for kernel
        static_cast<int32_t>(head_size),       // Convert to int32_t for kernel
        static_cast<float>(scale),             // Convert to float for kernel
        static_cast<float>(decay)              // Convert to float for kernel
    );
}

int64_t linear_attention_prefill_smem_size(  // Changed return type from size_t to int64_t
    int64_t num_qo_heads,    // Changed from int32_t to int64_t
    int64_t num_kv_heads,    // Changed from int32_t to int64_t
    int64_t head_size)       // Changed from int32_t to int64_t
{
    size_t smem_size = tensorrt_llm::kernels::getLinearAttentionPrefillKernelSmemSize<half, half, float>(
        static_cast<int32_t>(num_qo_heads),    // Convert to int32_t for kernel
        static_cast<int32_t>(num_kv_heads),    // Convert to int32_t for kernel
        static_cast<int32_t>(head_size)        // Convert to int32_t for kernel
    );
    return static_cast<int64_t>(smem_size);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("linear_attention_prefill("
          "Tensor output"
          ", Tensor state"
          ", Tensor q"
          ", Tensor k"
          ", Tensor v"
          ", Tensor cu_seq_lens"
          ", int num_seqs"
          ", int num_qo_heads"
          ", int num_kv_heads"
          ", int head_size"
          ", float scale"        // Schema uses 'float' even though function uses 'double'
          ", float decay"        // Schema uses 'float' even though function uses 'double'
          ") -> ()");

    m.def("linear_attention_prefill_smem_size("
          "int num_qo_heads"
          ", int num_kv_heads"
          ", int head_size"
          ") -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("linear_attention_prefill", &torch_ext::linear_attention_prefill);
    m.impl("linear_attention_prefill_smem_size", &torch_ext::linear_attention_prefill_smem_size);
} 