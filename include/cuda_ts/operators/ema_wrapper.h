#ifndef CUDA_TS_OPERATORS_EMA_WRAPPER_H
#define CUDA_TS_OPERATORS_EMA_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>

namespace cuda_ts {

/**
 * Compute Exponential Moving Average (EMA)
 * 
 * @param input Input time series
 * @param span Span parameter (larger = smoother, default 10)
 * @param stream Optional CUDA stream
 * @return TimeSeries with EMA values
 */
TimeSeries ema(const TimeSeries& input, int span = 10, cudaStream_t stream = nullptr);

} // namespace cuda_ts

#endif // CUDA_TS_OPERATORS_EMA_WRAPPER_H

