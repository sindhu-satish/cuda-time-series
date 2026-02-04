#ifndef CUDA_TS_OPERATORS_FFT_BANDS_WRAPPER_H
#define CUDA_TS_OPERATORS_FFT_BANDS_WRAPPER_H

#include "cuda_ts/core/timeseries.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

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
    cudaStream_t stream = nullptr
);

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
    cudaStream_t stream = nullptr
);

} // namespace cuda_ts

#endif // CUDA_TS_OPERATORS_FFT_BANDS_WRAPPER_H

