#pragma once

#include <cstdint>
#include "cuda_runtime_api.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename TO, typename TQKV, typename TState>
void launchLinearAttentionPrefillKernel(
    cudaStream_t stream,
    TO* output,               // ["packed_seq", Hqo, dq]
    TState* state,            // [num_seqs, Hkv, dv, dk], aka, KV
    TQKV const* q,            // ["packed_seq", Hqo, dq]
    TQKV const* k,            // ["packed_seq", Hkv, dk]
    TQKV const* v,            // ["packed_seq", Hkv, dv]
    int64_t const* seq_lens,  // [num_seqs + 1], prefix scan of packed length of sequences in the batch
    int32_t num_seqs,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_size,
    float scale,
    float decay,
    float const* per_head_deacy,
    int32_t decay_exponent_offset);


} // namespace kernels
} // namespace tensorrt_llm 


