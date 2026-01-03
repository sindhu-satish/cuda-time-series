#include "cuda_ts/operators/acf.h"
#include "cuda_ts/core/error_handler.h"
#include "cuda_ts/core/memory_manager.h"
#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ts {

namespace {

// kernel to compute mean of input array
__global__ void compute_mean_kernel(
    const float* input,
    float* mean_out,
    size_t n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load data into shared memory
    float val = (idx < n) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean_out[blockIdx.x] = sdata[0];
    }
}

// kernel to compute variance
__global__ void compute_variance_kernel(
    const float* input,
    const float mean,
    float* variance_out,
    size_t n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? (input[idx] - mean) : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        variance_out[blockIdx.x] = sdata[0];
    }
}

// kernel to compute ACF for a single lag
__global__ void acf_lag_kernel(
    const float* input,
    const float mean,
    const float variance,
    float* acf_out,
    size_t n,
    int lag
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // compute autocovariance for this lag
    float cov = 0.0f;
    if (idx < n - lag) {
        cov = (input[idx] - mean) * (input[idx + lag] - mean);
    }
    sdata[tid] = cov;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        acf_out[blockIdx.x] = sdata[0];
    }
}

} 

void acf_kernel(
    const float* input,
    float* output,
    size_t n,
    const int* lags,
    int num_lags,
    cudaStream_t stream
) {
    if (n == 0 || num_lags == 0) return;
    
    // allocate temporary memory (GPU memory)
    DeviceMemory<float> temp_mean(1);
    DeviceMemory<float> temp_variance(1);
    DeviceMemory<float> block_sums((n + 255) / 256);
    
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem = threads_per_block * sizeof(float);
    
    compute_mean_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
        input, block_sums.get(), n
    );
    KERNEL_CHECK("compute_mean_kernel");
    
    // TODO: Replace simple host-side reduction with proper device-side reduction for mean computation
    float* host_sums = new float[blocks];
    CUDA_CHECK(cudaMemcpyAsync(host_sums, block_sums.get(), 
                                blocks * sizeof(float), 
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float mean = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        mean += host_sums[i];
    }
    mean /= n;
    
    // compute variance
    compute_variance_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
        input, mean, block_sums.get(), n
    );
    KERNEL_CHECK("compute_variance_kernel");
    
    CUDA_CHECK(cudaMemcpyAsync(host_sums, block_sums.get(), 
                                blocks * sizeof(float), 
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float variance = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        variance += host_sums[i];
    }
    variance /= n;
    delete[] host_sums;  // Delete only once after both uses
    
    if (variance < 1e-10f) {
        // if zero variance, set all ACF to 0
        CUDA_CHECK(cudaMemsetAsync(output, 0, num_lags * sizeof(float), stream));
        return;
    }
    
    int* host_lags = new int[num_lags];
    CUDA_CHECK(cudaMemcpyAsync(host_lags, lags, num_lags * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // compute ACF for each lag
    DeviceMemory<float> acf_block_sums(blocks);
    DeviceMemory<float> acf_output(num_lags);
    
    for (int lag_idx = 0; lag_idx < num_lags; ++lag_idx) {
        int lag = host_lags[lag_idx];
        if (lag < 0 || lag >= static_cast<int>(n)) {
            float nan_val = NAN;
            CUDA_CHECK(cudaMemcpyAsync(acf_output.get() + lag_idx, &nan_val, sizeof(float),
                                        cudaMemcpyHostToDevice, stream));
            continue;
        }
        
        acf_lag_kernel<<<blocks, threads_per_block, shared_mem, stream>>>(
            input, mean, variance, acf_block_sums.get(), n, lag
        );
        KERNEL_CHECK("acf_lag_kernel");
        
        float* host_acf_sums = new float[blocks];
        CUDA_CHECK(cudaMemcpyAsync(host_acf_sums, acf_block_sums.get(),
                                    blocks * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        float autocov = 0.0f;
        for (int i = 0; i < blocks; ++i) {
            autocov += host_acf_sums[i];
        }
        autocov /= (n - lag);
        
        float acf_value = autocov / variance;
        CUDA_CHECK(cudaMemcpyAsync(acf_output.get() + lag_idx, &acf_value, sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        delete[] host_acf_sums;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(output, acf_output.get(), num_lags * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    
    delete[] host_lags;
}

} 

