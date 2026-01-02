#ifndef CUDA_TS_OPERATORS_ROLLING_MEAN_H
#define CUDA_TS_OPERATORS_ROLLING_MEAN_H

#include <cuda_runtime.h>
#include <cstddef>

namespace cuda_ts {

    
void rolling_mean_kernel(
    const float* input,
    float* output,
    size_t n,
    int window,
    cudaStream_t stream = nullptr
);

void rolling_mean_multi_kernel(
    const float* input,
    float* output,
    size_t n,
    const int* windows,
    int num_windows,
    cudaStream_t stream = nullptr
);

} 

#endif 

