#include "cuda_ts/operators/rolling_stats.h"
#include "cuda_ts/operators/rolling_stats_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/memory_manager.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>
#include <memory>

namespace cuda_ts {

TimeSeries rolling_mean(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_mean_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

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
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(windows_dev.get(), windows.data(),
                                   windows.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(windows_dev.get(), windows.data(),
                              windows.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    
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

TimeSeries rolling_std(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_std_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

TimeSeries rolling_var(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_var_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

TimeSeries rolling_min(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_min_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

TimeSeries rolling_max(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_max_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

TimeSeries rolling_zscore(const TimeSeries& input, int window, cudaStream_t stream) {
    if (input.empty() || window <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    rolling_zscore_kernel(input.data(), output.data(), input.size(), window, stream);
    
    return output;
}

} // namespace cuda_ts

