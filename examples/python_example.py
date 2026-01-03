import sys
import os
import argparse
import pandas as pd
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


def load_csv_data(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")
    
    df = pd.read_csv(filename)
    if len(df.columns) < 2:
        raise ValueError("CSV file must have at least 2 columns")
    
    data = df.iloc[:, 1].values.astype(np.float32)
    return data


def main():
    parser = argparse.ArgumentParser(description="CUDA timeseries Python API example")
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='Path to CSV file (default: auto-detect AirPassengers.csv)')
    args = parser.parse_args()
    print("CUDA timeseries Python API example")
    print("=" * 10)
    
    print(f"\nloading data from: {args.file}")
    data = load_csv_data(args.file)
    ts = cuda_ts_py.TimeSeries(data.tolist())
    print(f"loaded {ts.size()} data points")
    
    print("\ncomputing rolling mean (window=12)...")
    rolling_mean_result = cuda_ts_py.rolling_mean(ts, 12)
    print(f"   result shape: {len(rolling_mean_result)}")
    print(f"   first 10 values: {rolling_mean_result[:10]}")
    
    print("\ncomputing rolling standard deviation (window=12)...")
    rolling_std_result = cuda_ts_py.rolling_std(ts, 12)
    print(f"   result shape: {len(rolling_std_result)}")
    print(f"   first 10 values: {rolling_std_result[:10]}")
    
    print("\ncomputing ACF for lags [1, 5, 10, 12, 24]...")
    lags = [1, 5, 10, 12, 24]
    acf_result = cuda_ts_py.acf(ts, lags)
    print(f"   ACF values: {dict(zip(lags, acf_result))}")
    
    print("\ncomputing rolling mean for multiple windows [6, 12, 24]...")
    windows = [6, 12, 24]
    multi_rolling = cuda_ts_py.rolling_mean_multi(ts, windows)
    print(f"   number of results: {len(multi_rolling)}")
    for i, window in enumerate(windows):
        print(f"   window {window}: shape={len(multi_rolling[i])}")
    
    print("\n" + "=" * 10)


if __name__ == "__main__":
    main()

