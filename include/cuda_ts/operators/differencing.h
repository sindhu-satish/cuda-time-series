#ifndef CUDA_TS_OPERATORS_DIFFERENCING_H
#define CUDA_TS_OPERATORS_DIFFERENCING_H

#include <cuda_runtime.h>
#include <cstddef>

namespace cuda_ts {

/**
 * Compute first-order differencing: output[i] = input[i] - input[i-lag]
 * @param input Input time series
 * @param output Output time series
 * @param n Number of elements
 * @param lag Lag value (default 1 for first-order differencing)
 * @param stream CUDA stream
 */
void differencing_kernel(
    const float* input,
    float* output,
    size_t n,
    int lag = 1,
    cudaStream_t stream = nullptr
);

} 

#endif 

