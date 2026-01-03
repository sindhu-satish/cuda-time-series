# Benchmarking Framework

This directory contains benchmarking scripts to compare CUDA Time Series performance against reference implementations (pandas, numpy, statsmodels).

## Running Benchmarks

### Prerequisites

1. Build the Python module:
```bash
mkdir build && cd build
cmake ..
make
```

2. Install Python dependencies:
```bash
pip install numpy pandas
```

### Running Feature Benchmarks

```bash
cd benchmarks
python feature_benchmarks.py
```

This will:
- Compare rolling statistics (mean, std, var, min, max, z-score)
- Compare ACF implementation
- Measure speedup ratios
- Validate numerical accuracy

### Benchmark Output

The script outputs:
- **CUDA time**: Time taken by CUDA implementation
- **CPU time**: Time taken by pandas/numpy implementation
- **Speedup**: CPU time / CUDA time
- **Accuracy**: Normalized accuracy metric (1.0 = perfect match)

### Customizing Benchmarks

You can modify `feature_benchmarks.py` to:
- Change data sizes: `run_all_benchmarks(data_sizes=[1000, 10000, 100000])`
- Change number of iterations: `iterations=10`
- Add custom test data
- Add new benchmark functions

## Performance Metrics

The framework measures:
- **Throughput**: Points processed per second
- **Latency**: Time per operation (p50, p95, p99)
- **Speedup**: Ratio of CPU time to GPU time
- **Accuracy**: Numerical precision compared to reference
