#ifndef CUDA_TS_OPERATORS_DIFFERENCING_WRAPPER_H
#define CUDA_TS_OPERATORS_DIFFERENCING_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>

namespace cuda_ts {

/**
 * Compute first-order differencing: output[i] = input[i] - input[i-lag]
 * 
 * @param input Input time series
 * @param lag Lag value (default 1 for first-order differencing)
 * @param stream Optional CUDA stream
 * @return TimeSeries with differenced values (NaN for invalid indices)
 */
TimeSeries differencing(const TimeSeries& input, int lag = 1, cudaStream_t stream = nullptr);

} // namespace cuda_ts

#endif // CUDA_TS_OPERATORS_DIFFERENCING_WRAPPER_H

