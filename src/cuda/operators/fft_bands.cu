#include "cuda_ts/operators/fft_bands.h"
#include "cuda_ts/core/error_handler.h"
#include "cuda_ts/core/memory_manager.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstring>

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            throw std::runtime_error("cuFFT error: " + std::to_string(err)); \
        } \
    } while(0)

namespace cuda_ts {

namespace {

// kernel to apply window function to input
__global__ void apply_window_kernel(
    const float* input,
    float* windowed,
    size_t n,
    int window_type  // 0=none, 1=hann, 2=hamming, 3=blackman
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float window_val = 1.0f;
    
    if (n <= 1) {
        window_val = 1.0f;
    } else if (window_type == 1) {  // hann window
        window_val = 0.5f * (1.0f - cosf(2.0f * M_PI * idx / (n - 1)));
    } else if (window_type == 2) {  // hamming window
        window_val = 0.54f - 0.46f * cosf(2.0f * M_PI * idx / (n - 1));
    } else if (window_type == 3) {  // blackman window
        float a0 = 0.42f;
        float a1 = 0.5f;
        float a2 = 0.08f;
        float x = 2.0f * M_PI * idx / (n - 1);
        window_val = a0 - a1 * cosf(x) + a2 * cosf(2.0f * x);
    }
    
    windowed[idx] = input[idx] * window_val;
}

// kernel to compute power spectral density from FFT output
__global__ void compute_power_kernel(
    const cufftComplex* fft_output,
    float* power,
    int nfft
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfft) return;
    
    float real = fft_output[idx].x;
    float imag = fft_output[idx].y;
    power[idx] = real * real + imag * imag;
}

// kernel to integrate power over frequency bands
__global__ void integrate_bands_kernel(
    const float* power,
    float* band_power,
    int nfft,
    const float* bands,  // [low0, high0, low1, high1, ...]
    int num_bands,
    float sample_rate  // normalized to 1.0, so nyquist is 0.5
) {
    int band_idx = blockIdx.x;
    if (band_idx >= num_bands) return;
    
    float low = bands[band_idx * 2];
    float high = bands[band_idx * 2 + 1];
    
    // convert normalized frequencies to bin indices
    int low_bin = static_cast<int>(low * nfft);
    int high_bin = static_cast<int>(high * nfft);
    
    // clamp to valid range
    low_bin = max(0, min(low_bin, nfft / 2));
    high_bin = max(low_bin, min(high_bin, nfft / 2));
    
    // integrate power over band (only positive frequencies)
    float sum = 0.0f;
    for (int bin = low_bin; bin <= high_bin; ++bin) {
        sum += power[bin];
    }
    
    // also include negative frequencies (mirror)
    for (int bin = nfft - high_bin; bin <= nfft - low_bin && bin < nfft; ++bin) {
        if (bin >= 0) {
            sum += power[bin];
        }
    }
    
    band_power[band_idx] = sum;
}

} 

void fft_bands_kernel(
    const float* input,
    float* output,
    size_t n,
    const float* bands,
    int num_bands,
    const char* window_type,
    int nfft,
    cudaStream_t stream
) {
    if (n == 0 || num_bands == 0 || nfft < static_cast<int>(n)) return;
    
    int window_type_int = 0;  
    if (window_type != nullptr) {
        if (strcmp(window_type, "hann") == 0) {
            window_type_int = 1;
        } else if (strcmp(window_type, "hamming") == 0) {
            window_type_int = 2;
        } else if (strcmp(window_type, "blackman") == 0) {
            window_type_int = 3;
        }
    }
    
    // allocate device memory (GPU memory)
    DeviceMemory<float> windowed(nfft);
    DeviceMemory<cufftComplex> fft_output(nfft);
    DeviceMemory<float> power(nfft);
    DeviceMemory<float> bands_dev(num_bands * 2);
    
    // copy bands to device (GPU memory)
    CUDA_CHECK(cudaMemcpyAsync(bands_dev.get(), bands, num_bands * 2 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    
    // zero-padding for input
    CUDA_CHECK(cudaMemsetAsync(windowed.get(), 0, nfft * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(windowed.get(), input, n * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    
    // apply window (only to first n elements, rest are zero-padded)
    if (window_type_int != 0) {
        const int threads_per_block = 256;
        const int blocks = (n + threads_per_block - 1) / threads_per_block;
        apply_window_kernel<<<blocks, threads_per_block, 0, stream>>>(
            windowed.get(), windowed.get(), n, window_type_int
        );
        KERNEL_CHECK("apply_window_kernel");
    }
    
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftSetStream(plan, stream));
    CUFFT_CHECK(cufftPlan1d(&plan, nfft, CUFFT_R2C, 1));
    
    // execute FFT
    CUFFT_CHECK(cufftExecR2C(plan, 
                              reinterpret_cast<cufftReal*>(windowed.get()),
                              fft_output.get()));
    
    // compute power spectral density
    const int threads_per_block = 256;
    const int blocks = (nfft + threads_per_block - 1) / threads_per_block;
    compute_power_kernel<<<blocks, threads_per_block, 0, stream>>>(
        fft_output.get(), power.get(), nfft
    );
    KERNEL_CHECK("compute_power_kernel");
    
    integrate_bands_kernel<<<num_bands, 1, 0, stream>>>(
        power.get(), output, nfft, bands_dev.get(), num_bands, 1.0f
    );
    KERNEL_CHECK("integrate_bands_kernel");
    
    CUFFT_CHECK(cufftDestroy(plan));
}

} 

