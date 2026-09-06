cmake_minimum_required(VERSION 3.24)
project(cuda4as_m1_cmake_device_link LANGUAGES CXX CUDA)

add_executable(cuda4as_m1_cmake_device_link
    main.cu
    kernel.cu
    device-function.cu
)
set_target_properties(cuda4as_m1_cmake_device_link PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
