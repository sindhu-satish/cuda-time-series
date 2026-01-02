#ifndef CUDA_TS_OPERATORS_ACF_H
#define CUDA_TS_OPERATORS_ACF_H

#include <cuda_runtime.h>
#include <cstddef>

namespace cuda_ts {

void acf_kernel(
    const float* input,
    float* output,
    size_t n,
    const int* lags,
    int num_lags,
    cudaStream_t stream = nullptr
);

} 

#endif 

