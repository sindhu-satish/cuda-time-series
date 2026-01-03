

#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/operators/rolling_stats_wrapper.h"
#include "cuda_ts/operators/acf_wrapper.h"
#include "cuda_ts/operators/differencing_wrapper.h"
#include "cuda_ts/operators/ema_wrapper.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>

// Simple CSV parser - reads passenger numbers from AirPassengers.csv
std::vector<float> load_airpassengers(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        
        if (first_line or line.empty()) {
            first_line = false;
            continue;
        }        
        std::istringstream iss(line);
        std::string date, value_str;
        
        if (std::getline(iss, date, ',') && std::getline(iss, value_str)) {
            try {
                float value = std::stof(value_str);
                data.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse line: " << line << std::endl;
            }
        }
    }
    
    file.close();
    return data;
}

int main() {
    std::cout << "CUDA Time Series Library - AirPassengers Dataset\n";
    
    try {
        std::cout << "loading AirPassengers.csv...\n";
        std::vector<float> data = load_airpassengers("AirPassengers.csv");
        
        if (data.empty()) {
            std::cerr << "error: no data loaded from CSV file\n";
            return 1;
        }
        
        std::cout << "loaded " << data.size() << " data points\n";
        std::cout << "first 5 values: ";
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), data.size()); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
        std::cout << "last 5 values: ";
        for (size_t i = std::max(static_cast<size_t>(0), data.size() - 5); i < data.size(); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n\n";
        
        // create "TimeSeries" from loaded data
        std::cout << "creating TimeSeries from data...\n";
        cuda_ts::TimeSeries ts(data);
        std::cout << "TimeSeries created with size=" << ts.size() << "\n\n";
        
        // test 1: Rolling Mean
        int window = 12;  // 12 months 
        std::cout << "test 1: rolling mean with window=" << window << " (1 year)\n";
        std::cout << "computing rolling mean with window=" << window << " (1 year)...\n";
        
        auto rolling_result = cuda_ts::rolling_mean(ts, window);
        auto rolling_host = rolling_result.copy_to_host();
        
        std::cout << "\nrolling mean results (first 10 valid values, starting at index " 
                  << (window - 1) << "):\n";
        std::cout << std::fixed << std::setprecision(2);
        for (int i = window - 1; i < window + 9 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i 
                      << " (month " << (i + 1) << "): original=" << std::setw(6) << data[i]
                      << ", rolling mean=" << std::setw(6) << rolling_host[i] << "\n";
        }
        
        std::cout << "\nlast 5 rolling mean values:\n";
        for (int i = std::max(window - 1, static_cast<int>(data.size()) - 5); 
             i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i 
                      << " (month " << (i + 1) << "): original=" << std::setw(6) << data[i]
                      << ", rolling mean=" << std::setw(6) << rolling_host[i] << "\n";
        }
        std::cout << "\n";
        
        // test 2: Multiple Rolling Windows
        std::cout << "test 2: multiple rolling windows\n";
        std::vector<int> windows = {3, 6, 12};  // 3 months, 6 months, 1 year
        std::cout << "computing rolling mean for windows: ";
        for (size_t i = 0; i < windows.size(); ++i) {
            std::cout << windows[i];
            if (i < windows.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        
        auto multi_rolling = cuda_ts::rolling_mean_multi(ts, windows);
        std::cout << "\ncomparison at index " << (windows.back() - 1) << ":\n";
        for (size_t i = 0; i < windows.size(); ++i) {
            auto result = multi_rolling[i].copy_to_host();
            std::cout << "  window " << std::setw(2) << windows[i] 
                      << ": rolling mean = " << std::setw(6) 
                      << result[windows[i] - 1] << "\n";
        }
        std::cout << "\n";
        
        // test 3: Autocorrelation Function (ACF)
        std::cout << "test 3: autocorrelation function (ACF)\n";
        std::vector<int> lags = {1, 3, 6, 12, 24};  // 1 month, 3 months, 6 months, 1 year, 2 years
        std::cout << "computing ACF for lags: ";
        for (size_t i = 0; i < lags.size(); ++i) {
            std::cout << lags[i];
            if (i < lags.size() - 1) std::cout << ", ";
        }
        std::cout << "\n\n";
        
        auto acf_results = cuda_ts::acf(ts, lags);
        std::cout << "ACF values:\n";
        std::cout << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < lags.size(); ++i) {
            std::cout << "  lag " << std::setw(3) << lags[i] 
                      << " (" << std::setw(2) << lags[i] << " months): " 
                      << std::setw(10) << acf_results[i] << "\n";
        }
        std::cout << "\n";
        
        std::cout << "test 4: single lag ACF\n";
        int lag = 12;  // 1 year 
        std::cout << "computing ACF for lag=" << lag << " (1 year)...\n";
        float acf_val = cuda_ts::acf_single(ts, lag);
        std::cout << "ACF(lag=" << lag << ") = " << std::fixed << std::setprecision(6) 
                  << acf_val << "\n\n";
        
        // test 5: Rolling Standard Deviation
        std::cout << "test 5: rolling standard deviation\n";
        std::cout << "computing rolling std with window=" << window << "...\n";
        auto rolling_std_result = cuda_ts::rolling_std(ts, window);
        auto rolling_std_host = rolling_std_result.copy_to_host();
        std::cout << "First 5 valid rolling std values:\n";
        for (int i = window - 1; i < window + 4 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": std = " 
                      << std::fixed << std::setprecision(2) << rolling_std_host[i] << "\n";
        }
        std::cout << "\n";
        
        // test 6: Rolling Variance
        std::cout << "test 6: rolling variance\n";
        auto rolling_var_result = cuda_ts::rolling_var(ts, window);
        auto rolling_var_host = rolling_var_result.copy_to_host();
        std::cout << "First 5 valid rolling variance values:\n";
        for (int i = window - 1; i < window + 4 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": var = " 
                      << std::fixed << std::setprecision(2) << rolling_var_host[i] << "\n";
        }
        std::cout << "\n";
        
        // test 7: Rolling Min/Max
        std::cout << "test 7: rolling min and max\n";
        auto rolling_min_result = cuda_ts::rolling_min(ts, window);
        auto rolling_max_result = cuda_ts::rolling_max(ts, window);
        auto rolling_min_host = rolling_min_result.copy_to_host();
        auto rolling_max_host = rolling_max_result.copy_to_host();
        std::cout << "Comparison at index " << (window + 5) << ":\n";
        int idx = window + 5;
        if (idx < static_cast<int>(data.size())) {
            std::cout << "  original = " << std::setw(6) << data[idx] << "\n";
            std::cout << "  min      = " << std::setw(6) << rolling_min_host[idx] << "\n";
            std::cout << "  max      = " << std::setw(6) << rolling_max_host[idx] << "\n";
            std::cout << "  range    = " << std::setw(6) 
                      << (rolling_max_host[idx] - rolling_min_host[idx]) << "\n";
        }
        std::cout << "\n";
        
        // test 8: Rolling Z-Score
        std::cout << "test 8: rolling z-score\n";
        auto rolling_zscore_result = cuda_ts::rolling_zscore(ts, window);
        auto rolling_zscore_host = rolling_zscore_result.copy_to_host();
        std::cout << "First 5 valid rolling z-score values:\n";
        for (int i = window - 1; i < window + 4 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": z-score = " 
                      << std::fixed << std::setprecision(4) << rolling_zscore_host[i] << "\n";
        }
        std::cout << "\n";
        
        // test 9: Differencing
        std::cout << "test 9: differencing (first-order, lag=1)\n";
        auto diff_result = cuda_ts::differencing(ts, 1);
        auto diff_host = diff_result.copy_to_host();
        std::cout << "First 10 differenced values:\n";
        for (int i = 1; i < 11 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": " 
                      << std::setw(6) << data[i] << " - " 
                      << std::setw(6) << data[i-1] << " = " 
                      << std::setw(6) << diff_host[i] << "\n";
        }
        std::cout << "\n";
        
        // test 10: Exponential Moving Average
        std::cout << "test 10: exponential moving average (EMA)\n";
        int span = 12;  // 12-month span
        std::cout << "computing EMA with span=" << span << "...\n";
        auto ema_result = cuda_ts::ema(ts, span);
        auto ema_host = ema_result.copy_to_host();
        std::cout << "First 10 EMA values:\n";
        for (int i = 0; i < 10 && i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": original = " 
                      << std::setw(6) << data[i] << ", EMA = " 
                      << std::setw(6) << ema_host[i] << "\n";
        }
        std::cout << "Last 5 EMA values:\n";
        for (int i = std::max(0, static_cast<int>(data.size()) - 5); 
             i < static_cast<int>(data.size()); ++i) {
            std::cout << "  index " << std::setw(3) << i << ": original = " 
                      << std::setw(6) << data[i] << ", EMA = " 
                      << std::setw(6) << ema_host[i] << "\n";
        }
        std::cout << "\n";
        
        std::cout << "all tests completed successfully\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

