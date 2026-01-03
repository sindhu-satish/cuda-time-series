#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/operators/rolling_stats_wrapper.h"
#include "cuda_ts/operators/acf_wrapper.h"
#include <vector>

namespace py = pybind11;
using namespace cuda_ts;

// helper function to convert numpy array to TimeSeries
TimeSeries numpy_to_timeseries(py::array_t<float> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("TimeSeries requires 1D array");
    }
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;
    std::vector<float> data(ptr, ptr + size);
    return TimeSeries(data);
}

// helper function to convert TimeSeries to numpy array
py::array_t<float> timeseries_to_numpy(const TimeSeries& ts) {
    auto data = ts.copy_to_host();
    return py::cast(data);
}

PYBIND11_MODULE(cuda_ts_py, m) {
    m.doc() = "CUDA Timeseries Library - Python bindings";

    // timeseries class
    py::class_<TimeSeries>(m, "TimeSeries")
        .def(py::init<>())
        .def(py::init<const std::vector<float>&>(), 
             "Create TimeSeries from list/array",
             py::arg("data"))
        .def(py::init([](py::array_t<float> arr) {
            return numpy_to_timeseries(arr);
        }), "Create TimeSeries from numpy array", py::arg("data"))
        .def("size", &TimeSeries::size, "Get the size of the time series")
        .def("empty", &TimeSeries::empty, "Check if the time series is empty")
        .def("copy_to_host", &TimeSeries::copy_to_host, 
             "Copy data from GPU to CPU",
             py::return_value_policy::move)
        .def("to_numpy", [](const TimeSeries& ts) {
            return timeseries_to_numpy(ts);
        }, "Convert TimeSeries to numpy array")
        .def("__len__", &TimeSeries::size)
        .def("__repr__", [](const TimeSeries& ts) {
            return "<TimeSeries size=" + std::to_string(ts.size()) + ">";
        });

    // rolling statistics
    m.def("rolling_mean", 
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_mean(input, window));
          },
          "compute rolling mean",
          py::arg("input"), py::arg("window"));

    m.def("rolling_mean_multi",
          [](const TimeSeries& input, const std::vector<int>& windows) {
              auto results = rolling_mean_multi(input, windows);
              py::list py_results;
              for (const auto& ts : results) {
                  py_results.append(timeseries_to_numpy(ts));
              }
              return py_results;
          },
          "compute rolling mean for multiple window sizes",
          py::arg("input"), py::arg("windows"));

    m.def("rolling_std",
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_std(input, window));
          },
          "compute rolling standard deviation",
          py::arg("input"), py::arg("window"));

    m.def("rolling_var",
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_var(input, window));
          },
          "compute rolling variance",
          py::arg("input"), py::arg("window"));

    m.def("rolling_min",
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_min(input, window));
          },
          "compute rolling minimum",
          py::arg("input"), py::arg("window"));

    m.def("rolling_max",
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_max(input, window));
          },
          "compute rolling maximum",
          py::arg("input"), py::arg("window"));

    m.def("rolling_zscore",
          [](const TimeSeries& input, int window) {
              return timeseries_to_numpy(rolling_zscore(input, window));
          },
          "compute rolling z-score",
          py::arg("input"), py::arg("window"));

    // ACF
    m.def("acf",
          [](const TimeSeries& input, const std::vector<int>& lags) {
              return acf(input, lags);
          },
          "compute autocorrelation function",
          py::arg("input"), py::arg("lags"));

    m.def("acf_single",
          &acf_single,
          "compute ACF for a single lag",
          py::arg("input"), py::arg("lag"));

    
    // convenience function to create timeseries from numpy array
    m.def("from_numpy",
          [](py::array_t<float> arr) {
              return TimeSeries(numpy_to_timeseries(arr));
          },
          "create timeseries from numpy array",
          py::arg("data"),
          py::return_value_policy::move);
}

