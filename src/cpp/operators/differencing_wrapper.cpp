#include "cuda_ts/operators/differencing.h"
#include "cuda_ts/operators/differencing_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>

namespace cuda_ts {

TimeSeries differencing(const TimeSeries& input, int lag, cudaStream_t stream) {
    if (input.empty() || lag <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    differencing_kernel(input.data(), output.data(), input.size(), lag, stream);
    
    return output;
}

} // namespace cuda_ts

