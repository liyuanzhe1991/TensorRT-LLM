#include "linearAttentionKernels.h"
#include "linearAttentionPipeline.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#define CUDA_CHECK(expr)                                        \
  do {                                                          \
    cudaError_t err = (expr);                                   \
    if (err != cudaSuccess) {                                   \
      std::string buffer(1024, '\0');                           \
      sprintf(                                                  \
          buffer.data(), "CUDA Error: %s, Code: %d at %s:%d\n", \
          cudaGetErrorName(err), err, __FILE__, __LINE__        \
      );                                                        \
      throw std::runtime_error(buffer.c_str());                 \
    }                                                           \
  } while (0)

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
    float decay,
    float const* per_head_decay,
    int32_t decay_exponent_offset
)
{
    constexpr int NumThreads = 256;
    constexpr int BlockSize  = 32;
  
  #define LAUNCH(head_size)                                                                                            \
    {                                                                                                                  \
      auto  bytes = linear_attention_prefill_kernel_smem_size<NumThreads, (head_size), BlockSize, TO, TQKV, TState>(); \
      void* ptr   = (void*)linear_attention_prefill_kernel<NumThreads, (head_size), BlockSize, TO, TQKV, TState>;      \
      CUDA_CHECK(cudaFuncSetAttribute(ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));                       \
                                                                                                                       \
      int32_t num_ctas = num_qo_heads * num_seqs;                                                                      \
      linear_attention_prefill_kernel<NumThreads, (head_size), BlockSize><<<num_ctas, NumThreads, bytes, stream>>>(    \
          output, state, q, k, v, seq_lens, num_seqs, num_qo_heads, num_kv_heads, scale,                               \
          decay, per_head_decay, decay_exponent_offset                                                                 \
      );                                                                                                               \
    }
  
    if (head_size == 128) {
      LAUNCH(128);
      // } else if (head_size == 64) {
      //   LAUNCH(64);
    } else {
      throw std::runtime_error("unsupported head size " + std::to_string(head_size));
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
    float decay,
    float const* per_head_decay,
    int32_t decay_exponent_offset);

using bf16 = cute::bfloat16_t;

template void launchLinearAttentionPrefillKernel<bf16, bf16, float>(
        cudaStream_t   stream,
        bf16*          output,
        float*         state,
        bf16 const*    q,
        bf16 const*    k,
        bf16 const*    v,
        int64_t const* cu_seqlens,
        int32_t        num_seqs,
        int32_t        num_qo_heads,
        int32_t        num_kv_heads,
        int32_t        head_size,
        float          scale,
        float          decay,
        float const*   per_head_deacy,
        int32_t        decay_exponent_offset
    );
} // namespace kernels
} // namespace tensorrt_llm 