#ifndef CUDA_TS_OPERATORS_EMA_H
#define CUDA_TS_OPERATORS_EMA_H

#include <cuda_runtime.h>
#include <cstddef>

namespace cuda_ts {

/**
 * Compute Exponential Moving Average (EMA)
 * EMA[i] = alpha * input[i] + (1 - alpha) * EMA[i-1]
 * where alpha = 2 / (span + 1)
 * 
 * @param input Input time series
 * @param output Output time series
 * @param n Number of elements
 * @param span Span parameter (larger = smoother, default 10)
 * @param stream CUDA stream
 */
void ema_kernel(
    const float* input,
    float* output,
    size_t n,
    int span = 10,
    cudaStream_t stream = nullptr
);

} 

#endif 

