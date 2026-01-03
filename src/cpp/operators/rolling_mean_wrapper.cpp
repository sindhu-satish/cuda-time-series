#include "cuda_ts/operators/rolling_mean.h"
#include "cuda_ts/operators/rolling_mean_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/memory_manager.h"
#include "cuda_ts/core/stream_manager.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>
#include <memory>

namespace cuda_ts {

// compute rolling mean for a single window size
TimeSeries rolling_mean(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_mean_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

// compute rolling mean for multiple window sizes
std::vector<TimeSeries> rolling_mean_multi(
    const TimeSeries& input,
    const std::vector<int>& windows,
    cudaStream_t stream
) {
    if (input.empty() || windows.empty()) {
        return {};
    }
    
    size_t total_output_size = input.size() * windows.size();
    auto output_memory = std::make_unique<DeviceMemory<float>>(total_output_size);
    
    
    DeviceMemory<int> windows_dev(windows.size());
    CUDA_CHECK(cudaMemcpyAsync(windows_dev.get(), windows.data(),
                               windows.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream ? stream : cudaStreamDefault));
    
                               
    rolling_mean_multi_kernel(input.data(), output_memory->get(), input.size(),
                            windows_dev.get(), static_cast<int>(windows.size()), stream);
    
                            
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    
    std::vector<TimeSeries> results;
    results.reserve(windows.size());
    
    for (size_t i = 0; i < windows.size(); ++i) {
        size_t offset = i * input.size();
        auto window_memory = std::make_unique<DeviceMemory<float>>(input.size());
        CUDA_CHECK(cudaMemcpy(window_memory->get(), 
                             output_memory->get() + offset,
                             input.size() * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        results.emplace_back(std::move(window_memory), input.size());
    }
    
    return results;
}

} 
