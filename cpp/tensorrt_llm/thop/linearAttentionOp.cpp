/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/linearAttention/linearAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <torch/extension.h>
#include "cute/numeric/numeric_types.hpp"
using OptionalTensor = std::optional<torch::Tensor>;

namespace torch_ext
{
enum class DecayHead {
        AllShared,
        PerQOHead,
        // PerKVHead,  // NOTE: not implemented
    };


std::tuple<torch::Tensor /*output*/, torch::Tensor /*state*/>
linear_attention_prefill(
    OptionalTensor       output_,
    OptionalTensor       state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& cu_seqlens,
    double                scale,
    double                decay,
    OptionalTensor       per_head_decay,
    int64_t              decay_exponent_offset
) {
  int64_t num_seqs    = cu_seqlens.size(0) - 1;
  int64_t packed_seq  = q.size(0);
  int64_t num_qo_head = q.size(1);
  int64_t head_size   = q.size(2);
  int64_t num_kv_head = k.size(1);

  DecayHead decay_head = DecayHead::AllShared;
  if (per_head_decay.has_value()) {
    decay_head = DecayHead::PerQOHead;
  }
  // NOTE: if per_head_decay, aka, each qo head has a dedicated decay factor, we will output
  int32_t state_heads = decay_head == DecayHead::PerQOHead ? num_qo_head : num_kv_head;

  torch::Tensor output;
  if (output_.has_value()) {
    output = *output_;
  } else {
    output = torch::empty_like(q);
  }

  torch::Tensor state;
  if (state_.has_value()) {
    state = *state_;
  } else {
    auto options = q.options().dtype(torch::kFloat32);
    state = torch::zeros({num_seqs, state_heads, head_size, head_size}, options);
  }

  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(state.is_contiguous());
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(k.is_contiguous());
  TORCH_CHECK(v.is_contiguous());
  TORCH_CHECK(cu_seqlens.is_contiguous());

  auto device = q.device();
  TORCH_CHECK(device.is_cuda());
  TORCH_CHECK_EQ(device, output.device());
  TORCH_CHECK_EQ(device, state.device());
  TORCH_CHECK_EQ(device, q.device());
  TORCH_CHECK_EQ(device, k.device());
  TORCH_CHECK_EQ(device, v.device());
  TORCH_CHECK_EQ(device, cu_seqlens.device());

  TORCH_CHECK(output.dtype() == torch::kFloat16 || output.dtype() == torch::kBFloat16);
  TORCH_CHECK_EQ(torch::kFloat32, state.dtype());
  TORCH_CHECK_EQ(output.dtype(), q.dtype());
  TORCH_CHECK_EQ(output.dtype(), k.dtype());
  TORCH_CHECK_EQ(output.dtype(), v.dtype());
  TORCH_CHECK_EQ(torch::kInt64, cu_seqlens.dtype());

  TORCH_CHECK_EQ(packed_seq, k.size(0));
  TORCH_CHECK_EQ(packed_seq, v.size(0));
  TORCH_CHECK_EQ(packed_seq, output.size(0));

  TORCH_CHECK_EQ(num_seqs, state.size(0));
  TORCH_CHECK_EQ(state_heads, state.size(1));

  TORCH_CHECK_EQ(head_size, output.size(2));
  TORCH_CHECK_EQ(head_size, k.size(2));
  TORCH_CHECK_EQ(head_size, v.size(2));
  TORCH_CHECK_EQ(head_size, state.size(2));
  TORCH_CHECK_EQ(head_size, state.size(3));

  TORCH_CHECK(decay_exponent_offset == 0 || decay_exponent_offset == 1);

  if (scale == 0.0f) {
    scale = 1 / sqrt(head_size);
  }

  float const* per_head_decay_ptr = nullptr;
  if (per_head_decay.has_value()) {
    TORCH_CHECK_EQ(decay, 1.0f) << "decay must be 1.0f if per_head_decay is used";
    TORCH_CHECK(per_head_decay->dtype() == torch::kFloat32, "per_head_decay must be float32 tensor");
    TORCH_CHECK_EQ(num_qo_head, per_head_decay->numel());
    TORCH_CHECK_EQ(device, per_head_decay->device());
    per_head_decay_ptr = per_head_decay->data_ptr<float>();
  }

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  auto invoke_kernel_launcher = [&](auto dtype) {
    using DType = decltype(dtype);
    tensorrt_llm::kernels::launchLinearAttentionPrefillKernel<DType, DType, float>(
        cuda_stream,
        reinterpret_cast<DType*>(output.data_ptr()),
        state.data_ptr<float>(),
        reinterpret_cast<DType const*>(q.data_ptr()),
        reinterpret_cast<DType const*>(k.data_ptr()),
        reinterpret_cast<DType const*>(v.data_ptr()),
        cu_seqlens.data_ptr<int64_t>(),
        num_seqs,
        num_qo_head,
        num_kv_head,
        head_size,
        static_cast<float>(scale),
        static_cast<float>(decay),
        per_head_decay_ptr,
        decay_exponent_offset
    );
  };

  if (output.dtype() == torch::kFloat16) {
    invoke_kernel_launcher(half{});
  } else if (output.dtype() == torch::kBFloat16) {
    invoke_kernel_launcher(cute::bfloat16_t{});
  } else {
    TORCH_CHECK(false, "unsupported dtype ", torch::kBFloat16);
  }

  return {output, state};
}



} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("linear_attention_prefill("
          "Tensor? output"
          ", Tensor? state"
          ", Tensor q"
          ", Tensor k"
          ", Tensor v"
          ", Tensor cu_seq_lens"
          ", float scale"
          ", float decay"
          ", Tensor? per_head_decay"
          ", int decay_exponent_offset"
          ") -> (Tensor, Tensor)");

}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("linear_attention_prefill", &torch_ext::linear_attention_prefill);
} 