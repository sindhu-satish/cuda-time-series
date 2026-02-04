#include "cuda_ts/operators/ema.h"
#include "cuda_ts/operators/ema_wrapper.h"
#include "cuda_ts/core/timeseries.h"
#include "cuda_ts/core/error_handler.h"
#include <vector>

namespace cuda_ts {

TimeSeries ema(const TimeSeries& input, int span, cudaStream_t stream) {
    if (input.empty() || span <= 0) {
        return TimeSeries(0);
    }
    
    TimeSeries output(input.size());
    ema_kernel(input.data(), output.data(), input.size(), span, stream);
    
    return output;
}

} // namespace cuda_ts

