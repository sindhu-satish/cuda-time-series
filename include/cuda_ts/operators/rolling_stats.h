#ifndef CUDA_TS_OPERATORS_ROLLING_STATS_H
#define CUDA_TS_OPERATORS_ROLLING_STATS_H

#include <cuda_runtime.h>
#include <cstddef>

namespace cuda_ts {

// Rolling mean
void rolling_mean_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

// Rolling mean for multiple window sizes
void rolling_mean_multi_kernel(
    const float* input,
    float* output,
    size_t n,
    const int* windows,
    int num_windows,
    cudaStream_t stream = nullptr
);

// Rolling standard deviation
void rolling_std_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

// Rolling variance
void rolling_var_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

// Rolling minimum
void rolling_min_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

// Rolling maximum
void rolling_max_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

// Rolling z-score: (x[i] - rolling_mean[i]) / rolling_std[i]
void rolling_zscore_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

} 

#endif 

