cmake_minimum_required(VERSION 3.18)
project(cuda_time_series LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING "CUDA architectures")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/core/error_handler.cu
    src/cuda/operators/rolling_mean.cu
    src/cuda/operators/acf.cu   
)

# C++ sources
set(CPP_SOURCES
    src/cpp/operators/operator_registry.cpp
    src/cpp/operators/rolling_mean_wrapper.cpp
    src/cpp/operators/acf_wrapper.cpp
)

# Create library
add_library(cuda_ts_lib STATIC
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_link_libraries(cuda_ts_lib
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
)

target_compile_options(cuda_ts_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Wall -Wextra
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3
    >
)

# Set output directory
set_target_properties(cuda_ts_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Test executables
add_executable(test_airpassengers examples/test_airpassengers.cpp)
target_link_libraries(test_airpassengers cuda_ts_lib)
set_target_properties(test_airpassengers PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install targets
install(TARGETS cuda_ts_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

