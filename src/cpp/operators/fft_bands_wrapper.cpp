#include "cuda_ts/operators/fft_bands.h"
#include "cuda_ts/operators/fft_bands_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/memory_manager.h"
#include "cuda_ts/core/stream_manager.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace cuda_ts {

/**
 * Compute FFT band power for specified frequency bands
 * 
 * @param input Input time series
 * @param bands Vector of frequency band pairs [low0, high0, low1, high1, ...]
 *              Frequencies should be normalized (0.0 to 0.5, where 0.5 is Nyquist)
 * @param window_type Window function: "none", "hann", "hamming", "blackman"
 * @param nfft FFT size (should be >= input.size(), typically power of 2)
 * @param stream Optional CUDA stream
 * @return Vector of band power values, one for each band
 */
std::vector<float> fft_bands(
    const TimeSeries& input,
    const std::vector<float>& bands,
    const std::string& window_type,
    int nfft,
    cudaStream_t stream
) {
    if (input.empty() || bands.empty() || bands.size() % 2 != 0) {
        throw std::invalid_argument("Invalid input: empty time series or invalid bands");
    }
    
    if (nfft < static_cast<int>(input.size())) {
        throw std::invalid_argument("nfft must be >= input size");
    }
    
    int num_bands = static_cast<int>(bands.size() / 2);
    
    // Allocate output memory
    DeviceMemory<float> output(num_bands);
    
    // Copy bands to device
    DeviceMemory<float> bands_dev(bands.size());
    CUDA_CHECK(cudaMemcpyAsync(bands_dev.get(), bands.data(),
                               bands.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream ? stream : cudaStreamDefault));
    
    // Launch kernel
    fft_bands_kernel(input.data(), output.get(), input.size(),
                    bands_dev.get(), num_bands, window_type.c_str(), nfft, stream);
    
    // Copy results back to host
    std::vector<float> results(num_bands);
    CUDA_CHECK(cudaMemcpyAsync(results.data(), output.get(),
                              num_bands * sizeof(float),
                              cudaMemcpyDeviceToHost, stream ? stream : cudaStreamDefault));
    
    // Synchronize to ensure data is ready
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return results;
}

/**
 * Compute FFT band power with default window (hann) and auto nfft
 * 
 * @param input Input time series
 * @param bands Vector of frequency band pairs
 * @param stream Optional CUDA stream
 * @return Vector of band power values
 */
std::vector<float> fft_bands(
    const TimeSeries& input,
    const std::vector<float>& bands,
    cudaStream_t stream
) {
    // Auto-determine nfft as next power of 2 >= input.size()
    int nfft = 1;
    while (nfft < static_cast<int>(input.size())) {
        nfft <<= 1;
    }
    
    return fft_bands(input, bands, "hann", nfft, stream);
}

