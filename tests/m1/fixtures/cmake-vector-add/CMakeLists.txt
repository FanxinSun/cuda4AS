cmake_minimum_required(VERSION 3.24)
project(cuda4as_m1_cmake_vector_add LANGUAGES CXX CUDA)

add_executable(cuda4as_m1_cmake_vector_add main.cu)
set_target_properties(cuda4as_m1_cmake_vector_add PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
