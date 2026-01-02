#ifndef CUDA_TS_CORE_STREAM_MANAGER_H
#define CUDA_TS_CORE_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace cuda_ts {


class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    explicit CudaStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
private:
    cudaStream_t stream_;
};


class StreamPool {
public:
    explicit StreamPool(size_t num_streams = 4) {
        streams_.reserve(num_streams);
        for (size_t i = 0; i < num_streams; ++i) {
            streams_.emplace_back(std::make_unique<CudaStream>());
        }
    }
    
    CudaStream& get_stream(size_t index) {
        return *streams_[index % streams_.size()];
    }
    
    size_t size() const { return streams_.size(); }
    
    void synchronize_all() {
        for (auto& stream : streams_) {
            stream->synchronize();
        }
    }
    
private:
    std::vector<std::unique_ptr<CudaStream>> streams_;
};

} 

#endif 

