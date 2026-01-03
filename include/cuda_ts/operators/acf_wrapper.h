#ifndef CUDA_TS_OPERATORS_ACF_WRAPPER_H
#define CUDA_TS_OPERATORS_ACF_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>
#include <vector>

namespace cuda_ts {

std::vector<float> acf(const TimeSeries& input, const std::vector<int>& lags, cudaStream_t stream = nullptr);

float acf_single(const TimeSeries& input, int lag, cudaStream_t stream = nullptr);

} 

#endif 

