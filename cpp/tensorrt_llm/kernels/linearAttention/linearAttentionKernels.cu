#include "linearAttentionKernels.h"
#include "linearAttentionPipeline.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
{

template <typename TO, typename TQKV, typename TState>
void launchLinearAttentionPrefillKernel(
    cudaStream_t stream,
    TO* output,
    TState* state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    int64_t const* seq_lens,
    int32_t num_seqs,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_size,
    float scale,
    float decay)
{
    constexpr int NumThreads = 256;
    constexpr int BlockSize = 32;

#define LAUNCH(head_size_)                                                                                             \
    {                                                                                                                  \
        auto bytes = linear_attention_prefill_kernel_smem_size<NumThreads, (head_size_), BlockSize, TO, TQKV, TState>(); \
        void* ptr = (void*) linear_attention_prefill_kernel<NumThreads, (head_size_), BlockSize, TO, TQKV, TState>;     \
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));               \
                                                                                                                       \
        int32_t num_ctas = num_qo_heads * num_seqs;                                                                    \
        linear_attention_prefill_kernel<NumThreads, (head_size_), BlockSize><<<num_ctas, NumThreads, bytes, stream>>>( \
            output, state, q, k, v, seq_lens, num_seqs, num_qo_heads, num_kv_heads, scale, decay                      \
        );                                                                                                             \
    }

    if (head_size == 128)
    {
        LAUNCH(128);
    }
    // TODO: Add support for head_size == 64
    // else if (head_size == 64)
    // {
    //     LAUNCH(64);
    // }
    else
    {
        TLLM_THROW("Unsupported head size %d for linear attention", head_size);
    }
#undef LAUNCH
}

template <typename TO, typename TQKV, typename TState>
size_t getLinearAttentionPrefillKernelSmemSize(
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_size)
{
    constexpr int NumThreads = 256;
    constexpr int BlockSize = 32;

    if (head_size == 128)
    {
        return linear_attention_prefill_kernel_smem_size<NumThreads, 128, BlockSize, TO, TQKV, TState>();
    }
    // TODO: Add support for head_size == 64
    // else if (head_size == 64)
    // {
    //     return linear_attention_prefill_kernel_smem_size<NumThreads, 64, BlockSize, TO, TQKV, TState>();
    // }
    else
    {
        TLLM_THROW("Unsupported head size %d for linear attention", head_size);
    }
}

// Explicit instantiations - only for half type
template void launchLinearAttentionPrefillKernel<half, half, float>(
    cudaStream_t stream,
    half* output,
    float* state,
    half const* q,
    half const* k,
    half const* v,
    int64_t const* seq_lens,
    int32_t num_seqs,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_size,
    float scale,
    float decay);

// Remove float instantiation as CUTLASS copy operations don't support it
// template void launchLinearAttentionPrefillKernel<float, float, float>(
//     cudaStream_t stream,
//     float* output,
//     float* state,
//     float const* q,
//     float const* k,
//     float const* v,
//     int64_t const* seq_lens,
//     int32_t num_seqs,
//     int32_t num_qo_heads,
//     int32_t num_kv_heads,
//     int32_t head_size,
//     float scale,
//     float decay);

template size_t getLinearAttentionPrefillKernelSmemSize<half, half, float>(
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_size);

// Remove float instantiation
// template size_t getLinearAttentionPrefillKernelSmemSize<float, float, float>(
//     int32_t num_qo_heads,
//     int32_t num_kv_heads,
//     int32_t head_size);

} // namespace kernels
} // namespace tensorrt_llm 