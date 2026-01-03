import time
import numpy as np
import pandas as pd
import sys
import os
import argparse
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


def load_airpassengers(filename: str = None) -> np.ndarray:
    """Load AirPassengers dataset from CSV file"""
    df = pd.read_csv(filename)
    
    if len(df.columns) < 2:
        raise ValueError("CSV file must have at least 2 columns")
    
    # get the passenger column 
    passenger_col = df.iloc[:, 1]  
    data = passenger_col.values.astype(np.float32)
    
    return data


def benchmark_rolling_mean(data: np.ndarray, window: int = 20, 
                           iterations: int = 10) -> BenchmarkResult:
    """benchmark rolling mean operation"""
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
    """benchmark rolling standard deviation"""
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
    """benchmark autocorrelation function"""
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



def run_all_benchmarks(iterations: int = 10,
                       csv_filepath: str = None) -> List[BenchmarkResult]:
    """Run all benchmarks using CSV dataset"""
    results = []
    
    print(f"\n{'='*10}")
    if csv_filepath:
        print(f"loading dataset from: {csv_filepath}")
    else:
        print("loading AirPassengers.csv dataset...")
    print(f"{'='*10}")
    
    data = load_airpassengers(csv_filepath)
    filename = csv_filepath if csv_filepath else "AirPassengers.csv"
    print(f"loaded {len(data)} data points from {filename}")
    print(f"data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"data mean: {data.mean():.2f}")
    print(f"data std: {data.std():.2f}")
    
    print(f"\n{'='*10}")
    print(f"running benchmarks for dataset (size={len(data)})")
    print(f"{'='*10}")
    
    print(f"benchmarking rolling_mean...")
    results.append(benchmark_rolling_mean(data, window=12, iterations=iterations))
    
    print(f"benchmarking rolling_std...")
    results.append(benchmark_rolling_std(data, window=12, iterations=iterations))
    
    print(f"benchmarking acf...")
    results.append(benchmark_acf(data, lags=[1, 5, 10, 12, 24], iterations=iterations))
    
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
    parser = argparse.ArgumentParser(
        description="CUDA Timeseries benchmarking framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_benchmarks.py
  python feature_benchmarks.py --file AirPassengers.csv
  python feature_benchmarks.py --file /path/to/data.csv --iterations 10
        """
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to CSV file to use for benchmarking (default: auto-detect AirPassengers.csv)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=5,
        help='Number of iterations per benchmark (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("CUDA Timeseries benchmarking framework")
    
    if not CUDA_TS_AVAILABLE:
        print("WARNING: cuda_ts_py module not found. Only CPU benchmarks will run.")
        print("Build the Python module first: mkdir build && cd build && cmake .. && make")
        print()
    
    
    try:
        results = run_all_benchmarks(
            iterations=args.iterations,
            csv_filepath=args.file
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please provide a valid CSV file using --file option.")
        sys.exit(1)
    
    
    print_results(results)
    
    
    if CUDA_TS_AVAILABLE:
        speedups = [r.speedup for r in results if r.speedup]
        if speedups:
            print(f"\n{'='*10}")
            print("SUMMARY STATISTICS")
            print(f"{'='*10}")
            print(f"average speedup: {statistics.mean(speedups):.2f}x")
            print(f"median speedup: {statistics.median(speedups):.2f}x")
            print(f"max speedup: {max(speedups):.2f}x")
            print(f"min speedup: {min(speedups):.2f}x")
            print(f"{'='*10}")

