#ifndef CUDA_TS_CORE_TIMESERIES_H
#define CUDA_TS_CORE_TIMESERIES_H

#include "cuda_ts/core/memory_manager.h"
#include "cuda_ts/core/error_handler.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>

namespace cuda_ts {

// TimeSeries class - high-level wrapper for time series data on GPU
// manages GPU memory for time series data and provides convenient methods for copying data to/from host
class TimeSeries {
public:
    // create an empty TimeSeries
    TimeSeries() : size_(0) {}
    
    // create a TimeSeries from host data
    explicit TimeSeries(const std::vector<float>& data, cudaStream_t stream = nullptr)
        : data_(std::make_unique<DeviceMemory<float>>(data.size())), size_(data.size()) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(data_->get(), data.data(), 
                                      size_ * sizeof(float),
                                      cudaMemcpyHostToDevice, 
                                      stream ? stream : cudaStreamDefault));
            if (!stream) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }
    
    // create a TimeSeries with pre-allocated GPU memory
    explicit TimeSeries(size_t size)
        : data_(std::make_unique<DeviceMemory<float>>(size)), size_(size) {}
    
    // create a TimeSeries from existing device memory (takes ownership)
    TimeSeries(std::unique_ptr<DeviceMemory<float>> device_memory, size_t size)
        : data_(std::move(device_memory)), size_(size) {}
    
        
    TimeSeries(TimeSeries&& other) noexcept
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }
    
    
    TimeSeries& operator=(TimeSeries&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
    
    TimeSeries(const TimeSeries&) = delete;
    TimeSeries& operator=(const TimeSeries&) = delete;
    
    float* data() { return data_ ? data_->get() : nullptr; }
    const float* data() const { return data_ ? data_->get() : nullptr; }
    
    size_t size() const { return size_; }
    
    
    bool empty() const { return size_ == 0 || !data_; }
    
    
    void copy_from_host(const std::vector<float>& host_data, cudaStream_t stream = nullptr) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Size mismatch in copy_from_host");
        }
        if (size_ > 0 && data_) {
            CUDA_CHECK(cudaMemcpyAsync(data_->get(), host_data.data(),
                                      size_ * sizeof(float),
                                      cudaMemcpyHostToDevice,
                                      stream ? stream : cudaStreamDefault));
            if (!stream) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }
    
    
    std::vector<float> copy_to_host(cudaStream_t stream = nullptr) const {
        std::vector<float> host_data(size_);
        if (size_ > 0 && data_) {
            CUDA_CHECK(cudaMemcpyAsync(host_data.data(), data_->get(),
                                      size_ * sizeof(float),
                                      cudaMemcpyDeviceToHost,
                                      stream ? stream : cudaStreamDefault));
            if (!stream) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        return host_data;
    }
    
    
    TimeSeries slice(size_t offset, size_t length) const {
        if (offset + length > size_) {
            throw std::runtime_error("Slice out of bounds");
        }
        
        auto sliced_memory = std::make_unique<DeviceMemory<float>>(length);
        CUDA_CHECK(cudaMemcpy(sliced_memory->get(), data_->get() + offset,
                             length * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        return TimeSeries(std::move(sliced_memory), length);
    }
    
    
    DeviceMemory<float>* get_device_memory() { return data_.get(); }
    const DeviceMemory<float>* get_device_memory() const { return data_.get(); }

private:
    std::unique_ptr<DeviceMemory<float>> data_;
    size_t size_;
};

} 

#endif 

