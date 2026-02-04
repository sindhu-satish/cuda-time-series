#include "cuda_ts/operators/ema.h"
#include "cuda_ts/core/error_handler.h"
#include "cuda_ts/core/memory_manager.h"
#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ts {

namespace {

// CUDA kernel for EMA computation
// Since EMA is recursive, we need to process sequentially
// We'll use a single thread block to maintain order
__global__ void ema_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    float alpha
) {
    // Use a single thread to maintain sequential order
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n == 0) return;
        
        // First value is just the input
        output[0] = input[0];
        
        // Compute EMA for remaining values
        for (size_t i = 1; i < n; ++i) {
            output[i] = alpha * input[i] + (1.0f - alpha) * output[i - 1];
        }
    }
}

} 

void ema_kernel(
    const float* input,
    float* output,
    size_t n,
    int span,
    cudaStream_t stream
) {
    if (n == 0 || span <= 0) return;
    
    // Compute alpha from span: alpha = 2 / (span + 1)
    float alpha = 2.0f / static_cast<float>(span + 1);
    
    // EMA requires sequential processing, so use single thread block
    ema_kernel_impl<<<1, 1, 0, stream>>>(
        input, output, n, alpha
    );
    
    KERNEL_CHECK("ema_kernel");
}

} 

