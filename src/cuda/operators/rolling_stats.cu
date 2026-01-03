#include "cuda_ts/operators/rolling_stats.h"
#include "cuda_ts/core/error_handler.h"
#include "cuda_ts/core/memory_manager.h"
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


// CUDA kernel for rolling variance/std
__global__ void rolling_var_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    
    // compute mean first
    float sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        sum += input[idx - i];
    }
    float mean = sum / static_cast<float>(window);
    
    // compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        float diff = input[idx - i] - mean;
        var_sum += diff * diff;
    }
    
    output[idx] = var_sum / static_cast<float>(window);
}

// CUDA kernel for rolling standard deviation
__global__ void rolling_std_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    
    // compute mean first
    float sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        sum += input[idx - i];
    }
    float mean = sum / static_cast<float>(window);
    
    // compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < window; ++i) {
        float diff = input[idx - i] - mean;
        var_sum += diff * diff;
    }
    
    float variance = var_sum / static_cast<float>(window);
    output[idx] = sqrtf(variance);
}

// CUDA kernel for rolling minimum
__global__ void rolling_min_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    
    float min_val = input[idx];
    for (int i = 1; i < window; ++i) {
        min_val = fminf(min_val, input[idx - i]);
    }
    
    output[idx] = min_val;
}

// CUDA kernel for rolling maximum
__global__ void rolling_max_kernel_impl(
    const float* input,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    
    float max_val = input[idx];
    for (int i = 1; i < window; ++i) {
        max_val = fmaxf(max_val, input[idx - i]);
    }
    
    output[idx] = max_val;
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

void rolling_var_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_var_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, window
    );
    
    KERNEL_CHECK("rolling_var_kernel");
}

void rolling_std_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_std_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, window
    );
    
    KERNEL_CHECK("rolling_std_kernel");
}

void rolling_min_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_min_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, window
    );
    
    KERNEL_CHECK("rolling_min_kernel");
}

void rolling_max_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_max_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, output, n, window
    );
    
    KERNEL_CHECK("rolling_max_kernel");
}

namespace {

// CUDA kernel for rolling z-score
__global__ void rolling_zscore_kernel_impl(
    const float* input,
    const float* mean,
    const float* std,
    float* output,
    size_t n,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (idx < window - 1) {
        output[idx] = NAN;
        return;
    }
    float std_val = std[idx];
    if (std_val < 1e-10f) {
        output[idx] = 0.0f;
    } else {
        output[idx] = (input[idx] - mean[idx]) / std_val;
    }
}

} 

void rolling_zscore_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream
) {
    if (n == 0 || window <= 0) return;
    
    // Allocate temporary memory for mean and std
    DeviceMemory<float> mean_output(n);
    DeviceMemory<float> std_output(n);
    
    // Compute rolling mean
    rolling_mean_kernel(input, mean_output.get(), n, window, stream);
    
    // Compute rolling std
    rolling_std_kernel(input, std_output.get(), n, window, stream);
    
    // Synchronize before computing z-score
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Compute z-score: (x - mean) / std
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    rolling_zscore_kernel_impl<<<blocks, threads_per_block, 0, stream>>>(
        input, mean_output.get(), std_output.get(), output, n, window
    );
    
    KERNEL_CHECK("rolling_zscore_kernel");
}

} 

