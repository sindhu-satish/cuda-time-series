import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import cuda_ts_py
except ImportError:
    print("Error: cuda_ts_py module not found.")
    print("Please build the Python module first:")
    print("  mkdir build && cd build")
    print("  cmake ..")
    print("  make")
    sys.exit(1)


def main():
    print("CUDA Timeseries Python API example")
    print("=" * 10)
    
    
    print("\n1. creating timeseries from numpy array...")
    data = np.random.randn(1000).cumsum() + 100
    ts = cuda_ts_py.TimeSeries(data.tolist())
    print(f"   created timeseries with {ts.size()} points")
    
    print("\n2. computing rolling mean (window=20)...")
    rolling_mean_result = cuda_ts_py.rolling_mean(ts, 20)
    print(f"   result shape: {len(rolling_mean_result)}")
    print(f"   first 10 values: {rolling_mean_result[:10]}")
    
    print("\n3. computing rolling standard deviation (window=20)...")
    rolling_std_result = cuda_ts_py.rolling_std(ts, 20)
    print(f"   result shape: {len(rolling_std_result)}")
    print(f"   first 10 values: {rolling_std_result[:10]}")
    
    print("\n4. computing ACF for lags [1, 5, 10, 20]...")
    lags = [1, 5, 10, 20]
    acf_result = cuda_ts_py.acf(ts, lags)
    print(f"   ACF values: {dict(zip(lags, acf_result))}")
    
    print("\n7. computing rolling mean for multiple windows [10, 20, 50]...")
    windows = [10, 20, 50]
    multi_rolling = cuda_ts_py.rolling_mean_multi(ts, windows)
    print(f"   number of results: {len(multi_rolling)}")
    for i, window in enumerate(windows):
        print(f"   window {window}: shape={len(multi_rolling[i])}")
    
    print("\n" + "=" * 10)


if __name__ == "__main__":
    main()

