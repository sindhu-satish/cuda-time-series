#ifndef CUDA_TS_CORE_ERROR_HANDLER_H
#define CUDA_TS_CORE_ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda_ts {

// check CUDA error and throw exception if error occurred
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(error)
        );
    }
}

#define CUDA_CHECK(call) check_cuda_error(call, __FILE__, __LINE__)

// check for kernel launch errors
inline void check_kernel_launch(const char* kernel_name, const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Kernel launch error for ") + kernel_name +
            " at " + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(error)
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define KERNEL_CHECK(kernel_name) check_kernel_launch(kernel_name, __FILE__, __LINE__)

} 

#endif 

