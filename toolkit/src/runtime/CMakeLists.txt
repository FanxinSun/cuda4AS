# Shared runtime library: libcudart.dylib
add_library(cudart SHARED
    cudart.mm
)

target_include_directories(cudart PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CUDA4AS_INSTALL_INCLUDEDIR}>
)

target_compile_options(cudart PRIVATE
    "-fobjc-arc"
)

target_link_libraries(cudart PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

set_target_properties(cudart PROPERTIES
    VERSION "${CUDA_COMPAT_VERSION}"
    SOVERSION "${CUDA_COMPAT_MAJOR}"
    OUTPUT_NAME "cudart"
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${CUDA4AS_INSTALL_LIBDIR}"
)

# Static runtime library: libcudart_static.a
add_library(cudart_static STATIC
    cudart.mm
)

target_include_directories(cudart_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CUDA4AS_INSTALL_INCLUDEDIR}>
)

target_compile_options(cudart_static PRIVATE
    "-fobjc-arc"
)

target_link_libraries(cudart_static PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

set_target_properties(cudart_static PROPERTIES
    OUTPUT_NAME "cudart_static"
)

# Device runtime library stub: libcudadevrt.a
add_library(cudadevrt STATIC
    cudadevrt.cpp
)

set_target_properties(cudadevrt PROPERTIES
    OUTPUT_NAME "cudadevrt"
)

# Install libraries
install(TARGETS cudart cudart_static cudadevrt
    ARCHIVE DESTINATION "${CUDA4AS_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CUDA4AS_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CUDA4AS_INSTALL_BINDIR}"
)

# Symlinks for libcuda compatibility (required by CCCL dlopen("libcuda.so.1"))
install(CODE "
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libcudart.dylib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CUDA4AS_INSTALL_LIBDIR}/libcuda.dylib\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libcudart.dylib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CUDA4AS_INSTALL_LIBDIR}/libcuda.so.1\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libcudart.dylib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CUDA4AS_INSTALL_LIBDIR}/libcuda.so\")
")
