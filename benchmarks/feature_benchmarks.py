import time
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Any
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import cuda_ts_py
    CUDA_TS_AVAILABLE = True
except ImportError:
    print("Warning: cuda_ts_py not available. Install the Python module first.")
    CUDA_TS_AVAILABLE = False


class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self, name: str, cuda_time: float = None, cpu_time: float = None,
                 speedup: float = None, accuracy: float = None):
        self.name = name
        self.cuda_time = cuda_time
        self.cpu_time = cpu_time
        self.speedup = speedup
        self.accuracy = accuracy
    
    def __repr__(self):
        return (f"BenchmarkResult({self.name}, "
                f"cuda={self.cuda_time:.4f}s, cpu={self.cpu_time:.4f}s, "
                f"speedup={self.speedup:.2f}x, accuracy={self.accuracy:.6f})")


def generate_test_data(size: int = 10000) -> np.ndarray:
    """Generate synthetic time series data"""
    np.random.seed(42)
    trend = np.linspace(0, 100, size)
    noise = np.random.normal(0, 5, size)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(size) / 365)
    return trend + seasonal + noise


def benchmark_rolling_mean(data: np.ndarray, window: int = 20, 
                           iterations: int = 10) -> BenchmarkResult:
    """Benchmark rolling mean operation"""
    name = f"rolling_mean(window={window}, size={len(data)})"
    
    cuda_times = []
    if CUDA_TS_AVAILABLE:
        for _ in range(iterations):
            ts = cuda_ts_py.TimeSeries(data.tolist())
            start = time.perf_counter()
            result_cuda = cuda_ts_py.rolling_mean(ts, window)
            end = time.perf_counter()
            cuda_times.append(end - start)
        cuda_time = statistics.median(cuda_times)
    else:
        cuda_time = None
    
    cpu_times = []
    df = pd.Series(data)
    for _ in range(iterations):
        start = time.perf_counter()
        result_pandas = df.rolling(window=window, min_periods=1).mean()
        end = time.perf_counter()
        cpu_times.append(end - start)
    cpu_time = statistics.median(cpu_times)
    
    speedup = cpu_time / cuda_time if cuda_time else None
    
    if CUDA_TS_AVAILABLE:
        result_cuda_array = np.array(result_cuda)
        result_pandas_array = result_pandas.values
        mask = ~(np.isnan(result_cuda_array) | np.isnan(result_pandas_array))
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((result_cuda_array[mask] - result_pandas_array[mask])**2))
            accuracy = 1.0 / (1.0 + rmse)  # normalized accuracy
        else:
            accuracy = 0.0
    else:
        accuracy = None
    
    return BenchmarkResult(name, cuda_time, cpu_time, speedup, accuracy)


def benchmark_rolling_std(data: np.ndarray, window: int = 20,
                           iterations: int = 10) -> BenchmarkResult:
    """Benchmark rolling standard deviation"""
    name = f"rolling_std(window={window}, size={len(data)})"
    
    # CUDA 
    cuda_times = []
    if CUDA_TS_AVAILABLE:
        for _ in range(iterations):
            ts = cuda_ts_py.TimeSeries(data.tolist())
            start = time.perf_counter()
            result_cuda = cuda_ts_py.rolling_std(ts, window)
            end = time.perf_counter()
            cuda_times.append(end - start)
        cuda_time = statistics.median(cuda_times)
    else:
        cuda_time = None
    
    # pandas 
    cpu_times = []
    df = pd.Series(data)
    for _ in range(iterations):
        start = time.perf_counter()
        result_pandas = df.rolling(window=window, min_periods=1).std()
        end = time.perf_counter()
        cpu_times.append(end - start)
    cpu_time = statistics.median(cpu_times)
    
    speedup = cpu_time / cuda_time if cuda_time else None
    
    
    if CUDA_TS_AVAILABLE:
        result_cuda_array = np.array(result_cuda)
        result_pandas_array = result_pandas.values
        mask = ~(np.isnan(result_cuda_array) | np.isnan(result_pandas_array))
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((result_cuda_array[mask] - result_pandas_array[mask])**2))
            accuracy = 1.0 / (1.0 + rmse)
        else:
            accuracy = 0.0
    else:
        accuracy = None
    
    return BenchmarkResult(name, cuda_time, cpu_time, speedup, accuracy)


def benchmark_acf(data: np.ndarray, lags: List[int] = [1, 5, 10, 20],
                 iterations: int = 10) -> BenchmarkResult:
    """Benchmark autocorrelation function"""
    name = f"acf(lags={lags}, size={len(data)})"
    
    # CUDA 
    cuda_times = []
    if CUDA_TS_AVAILABLE:
        for _ in range(iterations):
            ts = cuda_ts_py.TimeSeries(data.tolist())
            start = time.perf_counter()
            result_cuda = cuda_ts_py.acf(ts, lags)
            end = time.perf_counter()
            cuda_times.append(end - start)
        cuda_time = statistics.median(cuda_times)
    else:
        cuda_time = None
    
    # NumPy (using correlation)
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result_numpy = []
        for lag in lags:
            if lag < len(data):
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                result_numpy.append(corr if not np.isnan(corr) else 0.0)
            else:
                result_numpy.append(0.0)
        end = time.perf_counter()
        cpu_times.append(end - start)
    cpu_time = statistics.median(cpu_times)
    
    speedup = cpu_time / cuda_time if cuda_time else None
    
    
    if CUDA_TS_AVAILABLE:
        result_cuda_array = np.array(result_cuda)
        result_numpy_array = np.array(result_numpy)
        mask = ~(np.isnan(result_cuda_array) | np.isnan(result_numpy_array))
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((result_cuda_array[mask] - result_numpy_array[mask])**2))
            accuracy = 1.0 / (1.0 + rmse)
        else:
            accuracy = 0.0
    else:
        accuracy = None
    
    return BenchmarkResult(name, cuda_time, cpu_time, speedup, accuracy)



def run_all_benchmarks(data_sizes: List[int] = [1000, 10000, 100000],
                       iterations: int = 10) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes"""
    results = []
    
    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Running benchmarks for data size: {size}")
        print(f"{'='*60}")
        
        data = generate_test_data(size)
        
        
        print(f"Benchmarking rolling_mean...")
        results.append(benchmark_rolling_mean(data, window=20, iterations=iterations))
        
        
        print(f"Benchmarking rolling_std...")
        results.append(benchmark_rolling_std(data, window=20, iterations=iterations))
        
        
        print(f"Benchmarking acf...")
        results.append(benchmark_acf(data, lags=[1, 5, 10, 20], iterations=iterations))
        
    
    return results


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    print("\n" + "="*10)
    print("BENCHMARK RESULTS")
    print("="*10)
    print(f"{'Operation':<40} {'CUDA (s)':<12} {'CPU (s)':<12} {'Speedup':<12} {'Accuracy':<12}")
    print("-"*10)
    
    for result in results:
        cuda_str = f"{result.cuda_time:.6f}" if result.cuda_time else "N/A"
        cpu_str = f"{result.cpu_time:.6f}" if result.cpu_time else "N/A"
        speedup_str = f"{result.speedup:.2f}x" if result.speedup else "N/A"
        accuracy_str = f"{result.accuracy:.6f}" if result.accuracy else "N/A"
        
        print(f"{result.name:<40} {cuda_str:<12} {cpu_str:<12} {speedup_str:<12} {accuracy_str:<12}")
    
    print("="*10)


if __name__ == "__main__":
    print("CUDA Timeseries benchmarking")
    
    if not CUDA_TS_AVAILABLE:
        print("WARNING: cuda_ts_py module not found. Only CPU benchmarks will run.")
        print("Build the Python module first: mkdir build && cd build && cmake .. && make")
        print()
    results = run_all_benchmarks(data_sizes=[1000, 10000, 100000], iterations=5)
    print_results(results)
    
    if CUDA_TS_AVAILABLE:
        speedups = [r.speedup for r in results if r.speedup]
        if speedups:
            print(f"\nAverage speedup: {statistics.mean(speedups):.2f}x")
            print(f"Median speedup: {statistics.median(speedups):.2f}x")
            print(f"Max speedup: {max(speedups):.2f}x")
            print(f"Min speedup: {min(speedups):.2f}x")

