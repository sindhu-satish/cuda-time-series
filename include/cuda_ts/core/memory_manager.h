#ifndef CUDA_TS_CORE_MEMORY_MANAGER_H
#define CUDA_TS_CORE_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <memory>
#include <cstddef>

namespace cuda_ts {

template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceMemory(size_t size) : size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
        }
    }
    
    ~DeviceMemory() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    
    
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory&
    
    DeviceMemory(DeviceMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    
private:
    T* ptr_;
    size_t size_;
};

template<typename T>
class PinnedMemory {
public:
    PinnedMemory() : ptr_(nullptr), size_(0) {}
    
    explicit PinnedMemory(size_t size) : size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, size * sizeof(T)));
        }
    }
    
    ~PinnedMemory() {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
    }
    
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;
    
    PinnedMemory(PinnedMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    
private:
    T* ptr_;
    size_t size_;
};


class MemoryPool {
public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }
    
    void* allocate(size_t bytes) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
    
private:
    MemoryPool() = default;
    ~MemoryPool() = default;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
};

} 

#endif 

