#include "cuda_ts/operators/rolling_mean.h"
#include "cuda_ts/core/error_handler.h"
#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ts {

namespace {

// CUDA kernel for rolling mean computation
__global__ void rolling_mean_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // check if we have enough data for this window
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    
    // sum over the window
    float sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        sum += input[idx - i];
    }
    
    output[idx] = sum / static_cast<float>(window);
}

// CUDA kernel for multiple window sizes
__global__ void rolling_mean_multi_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    const int* windows,
    int num_windows
) {
    int window_idx = blockIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (window_idx >= num_windows || idx >= n) return;
    
    int window = windows[window_idx];
    int output_offset = window_idx * n;
    
    if (idx < window - 1) {
        output[output_offset + idx] = NAN;
        return;
    }
    
    // sum over the window
    float sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        sum += input[idx - i];
    }
    
    output[output_offset + idx] = sum / static_cast<float>(window);
}

} 

void rolling_mean_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_mean_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, window
    );
    
    KERNEL_CHECK("rolling_mean_kernel");
}

void rolling_mean_multi_kernel(
    const float* input,
    float* output,
    size_t n,
    const int* windows,
    int num_windows,
    cudaStream_t stream
) {
    if (n == 0 || num_windows == 0) return;
    
    const int threads_per_block = 256;
    dim3 blocks(num_windows, (n + threads_per_block - 1) / threads_per_block);
    dim3 threads(1, threads_per_block);
    
    rolling_mean_multi_kernel_impl<<<blocks, threads, 0, stream>>>(
        input, output, n, windows, num_windows
    );
    
    KERNEL_CHECK("rolling_mean_multi_kernel");
}

} 

