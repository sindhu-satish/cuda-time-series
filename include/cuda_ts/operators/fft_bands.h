#ifndef CUDA_TS_OPERATORS_FFT_BANDS_H
#define CUDA_TS_OPERATORS_FFT_BANDS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstddef>

namespace cuda_ts {

void fft_bands_kernel(
    const float* input,
    float* output,
    size_t n,
    const float* bands,  // [low0, high0, low1, high1, ...]
    int num_bands,
    const char* window_type,
    int nfft,
    cudaStream_t stream = nullptr
);

} 

#endif 

