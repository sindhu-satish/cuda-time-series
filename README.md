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
- `rolling_mean`, `rolling_std`, `rolling_var`, `rolling_min`, `rolling_max`, `rolling_zscore` - Rolling statistics
- `acf` - Autocorrelation function for specified lags

### Python API (Phase 5)
- **Python bindings** using pybind11
- NumPy array integration
- High-level Python API matching C++ functionality
- See `examples/python_example.py` for usage

### Benchmarking Framework (Phase 6)
- Performance comparison against pandas/numpy
- Speedup measurements
- Accuracy validation
- Multiple data size testing (1K, 10K, 100K points)
- Statistical aggregation (median times, summary stats)

### Testing
- `test_airpassengers.cpp` - Test program using real-world AirPassengers dataset
  - Tests rolling mean with multiple window sizes
  - Tests ACF with various lag values
  - Demonstrates usage of the C++ API
- `examples/python_example.py` - Python API usage examples

## Quick Start

### Building

```bash
apt-get update && apt-get install -y cmake build-essential // if cmake is not installed
mkdir build && cd build
cmake ..
make
```

### Running Tests

**C++ Tests:**
```bash
./build/bin/test_airpassengers
```

**Python Module:**
```bash
# Build Python module
mkdir build && cd build
cmake ..
make

# Run Python example
cd ..
python examples/python_example.py
```

**Benchmarks:**
```bash
# Install Python dependencies
pip install numpy pandas

# Run benchmarks
cd benchmarks
python feature_benchmarks.py
```

The benchmark script compares CUDA implementations against pandas/numpy and outputs:
- **CUDA time**: Time taken by CUDA implementation
- **CPU time**: Time taken by pandas/numpy implementation
- **Speedup**: CPU time / CUDA time ratio
- **Accuracy**: Normalized accuracy metric (1.0 = perfect match)

Benchmarks are run for:
- Rolling statistics (mean, std) vs pandas
- ACF vs numpy correlation
- Multiple data sizes (1K, 10K, 100K points)
- Multiple iterations with median time reporting

### Customizing Benchmarks

You can modify `benchmarks/feature_benchmarks.py` to:
- Change data sizes: `run_all_benchmarks(data_sizes=[1000, 10000, 100000])`
- Change number of iterations: `iterations=10`
- Add custom test data
- Add new benchmark functions

### Performance Metrics

The framework measures:
- **Throughput**: Points processed per second
- **Latency**: Time per operation (median times)
- **Speedup**: Ratio of CPU time to GPU time
- **Accuracy**: Numerical precision compared to reference (RMSE-based)

### Usage Examples

**C++ API:**
```cpp
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/operators/rolling_stats_wrapper.h"
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

**Python API:**
```python
import numpy as np
import cuda_ts_py

# Create TimeSeries from numpy array
data = np.random.randn(1000).cumsum()
ts = cuda_ts_py.TimeSeries(data.tolist())

# Compute rolling mean
rolling_mean = cuda_ts_py.rolling_mean(ts, 20)

# Compute ACF
acf_values = cuda_ts_py.acf(ts, [1, 5, 10, 20])

# Compute rolling statistics
rolling_std = cuda_ts_py.rolling_std(ts, 20)
rolling_var = cuda_ts_py.rolling_var(ts, 20)
rolling_min = cuda_ts_py.rolling_min(ts, 20)
rolling_max = cuda_ts_py.rolling_max(ts, 20)
rolling_zscore = cuda_ts_py.rolling_zscore(ts, 20)

# Multiple rolling windows
multi_rolling = cuda_ts_py.rolling_mean_multi(ts, [10, 20, 50])
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
delete_file

