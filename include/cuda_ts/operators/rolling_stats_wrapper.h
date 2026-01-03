#ifndef CUDA_TS_OPERATORS_ROLLING_STATS_WRAPPER_H
#define CUDA_TS_OPERATORS_ROLLING_STATS_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>
#include <vector>

namespace cuda_ts {

/**
 * Compute rolling mean for a single window size
 */
TimeSeries rolling_mean(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

/**
 * Compute rolling mean for multiple window sizes
 */
std::vector<TimeSeries> rolling_mean_multi(
    const TimeSeries& input,
    const std::vector<int>& windows,
    cudaStream_t stream = nullptr
);

/**
 * Compute rolling standard deviation
 */
TimeSeries rolling_std(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

/**
 * Compute rolling variance
 */
TimeSeries rolling_var(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

/**
 * Compute rolling minimum
 */
TimeSeries rolling_min(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

/**
 * Compute rolling maximum
 */
TimeSeries rolling_max(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

/**
 * Compute rolling z-score: (x[i] - rolling_mean[i]) / rolling_std[i]
 */
TimeSeries rolling_zscore(const TimeSeries& input, int window, cudaStream_t stream = nullptr);

} // namespace cuda_ts

#endif // CUDA_TS_OPERATORS_ROLLING_STATS_WRAPPER_H

