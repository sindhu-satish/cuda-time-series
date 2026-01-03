#include "cuda_ts/operators/acf.h"
#include "cuda_ts/operators/acf_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/memory_manager.h"
#include "cuda_ts/core/stream_manager.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>
#include <memory>
#include <cmath>

namespace cuda_ts {

// compute autocorrelation function (ACF) for specified lags
std::vector<float> acf(const TimeSeries& input, const std::vector<int>& lags, cudaStream_t stream) {
    if (input.empty() || lags.empty()) {
        return {};
    }
    
    DeviceMemory<float> output(lags.size());
    
    DeviceMemory<int> lags_dev(lags.size());
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(lags_dev.get(), lags.data(),
                                   lags.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(lags_dev.get(), lags.data(),
                              lags.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    
    // launch GPU kernel
    acf_kernel(input.data(), output.get(), input.size(),
               lags_dev.get(), static_cast<int>(lags.size()), stream);
    
    // Synchronize before copying results back
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    std::vector<float> results(lags.size());
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(results.data(), output.get(),
                                  lags.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(results.data(), output.get(),
                             lags.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    
    return results;
}

// compute ACF for a single lag
float acf_single(const TimeSeries& input, int lag, cudaStream_t stream) {
    if (input.empty()) {
        return NAN;
    }
    
    std::vector<int> lags = {lag};
    auto results = acf(input, lags, stream);
    return results.empty() ? NAN : results[0];
}

} 
