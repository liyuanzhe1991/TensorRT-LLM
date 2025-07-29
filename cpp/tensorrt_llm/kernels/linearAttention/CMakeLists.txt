# Linear Attention Kernels

set(LINEAR_ATTENTION_KERNELS_SRCS
    linearAttentionKernels.cu
)

add_library(linear_attention_kernels STATIC ${LINEAR_ATTENTION_KERNELS_SRCS})

set_property(TARGET linear_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET linear_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(linear_attention_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CUTLASS_DIR}/include
    PUBLIC ${CUTLASS_DIR}/tools/util/include
)

target_compile_options(linear_attention_kernels PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
) 