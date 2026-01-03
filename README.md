# CUDA Time Series Library

A high-performance, CUDA-accelerated time series feature extraction library with strict contracts, versioned operators, and deterministic streaming support.

## Current Status

### Core Infrastructure
- GPU memory management (`DeviceMemory`, `PinnedMemory`, `MemoryPool`)
- CUDA stream management (`CudaStream`, `StreamPool`)
- Error handling utilities
- High-level `TimeSeries` class for GPU memory abstraction

### C++ API Layer
- **TimeSeries class**: High-level wrapper for managing time series data on GPU
  - Automatic memory management (RAII)
  - Easy data transfer between CPU and GPU
  - Move semantics for efficient transfers
  
- **Operator wrappers**: High-level C++ API functions
  - `rolling_mean()` - Compute rolling mean with single or multiple window sizes
  - `acf()` - Compute autocorrelation function for specified lags
  - Operator registry system for operator discovery and metadata

### Implemented Operators
- `rolling_mean` - Rolling mean with configurable window sizes (CUDA kernel + C++ wrapper)
- `acf` - Autocorrelation function for specified lags (CUDA kernel + C++ wrapper)

### Testing
- `test_airpassengers.cpp` - Test program using real-world AirPassengers dataset
  - Tests rolling mean with multiple window sizes
  - Tests ACF with various lag values
  - Demonstrates usage of the C++ API

## Quick Start

### Building

```bash
mkdir build && cd build
cmake ..
make
```

### Running Tests

```bash
./build/bin/test_airpassengers
```

### Usage Example

```cpp
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/operators/rolling_mean_wrapper.h"
#include "cuda_ts/operators/acf_wrapper.h"

// Create TimeSeries from host data
std::vector<float> data = {1.0, 2.0, 3.0, ...};
cuda_ts::TimeSeries ts(data);

// Compute rolling mean
auto rolling_result = cuda_ts::rolling_mean(ts, 12);
auto rolling_values = rolling_result.copy_to_host();

// Compute ACF
std::vector<int> lags = {1, 5, 10, 20};
auto acf_values = cuda_ts::acf(ts, lags);
```

