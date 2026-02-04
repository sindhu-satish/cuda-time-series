#include "cuda_ts/operators/differencing.h"
#include "cuda_ts/core/error_handler.h"
#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ts {

namespace {

// CUDA kernel for differencing
__global__ void differencing_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int lag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // check if we have enough data for this lag
    if (idx < lag) {
        output[idx] = NAN;
        return;
    }
    
    output[idx] = input[idx] - input[idx - lag];
}

} 

void differencing_kernel(
    const float* input,
    float* output,
    size_t n,
    int lag,
    cudaStream_t stream
) {
    if (n == 0 || lag <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    differencing_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, lag
    );
    
    KERNEL_CHECK("differencing_kernel");
}

} 

