# CUDA Time Series Library

A high-performance, CUDA-accelerated time series feature extraction library with strict contracts, versioned operators, and deterministic streaming support.

## Current Status

The core CUDA infrastructure and three initial operators are implemented:

- Core infrastructure (memory management, error handling, stream management)
- `rolling_mean` - Rolling mean with configurable window sizes
- `acf` - Autocorrelation function for specified lags
- `fft_bands` - FFT-based band power computation with windowing

