#ifndef CUDA_TS_OPERATORS_ROLLING_MEAN_WRAPPER_H
#define CUDA_TS_OPERATORS_ROLLING_MEAN_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>
#include <vector>

namespace cuda_ts {

    
TimeSeries rolling_mean(const TimeSeries& input, int window, cudaStream_t stream = nullptr);


std::vector<TimeSeries> rolling_mean_multi(
    const TimeSeries& input,
    const std::vector<int>& windows,
    cudaStream_t stream = nullptr
);

} 

#endif 

