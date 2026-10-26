cmake_minimum_required(VERSION 3.20)
project(cuda4AS-toolkit
    VERSION 0.1.0
    DESCRIPTION "Native CUDA Toolchain for Apple Silicon"
    LANGUAGES C CXX OBJCXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Version definitions
set(CUDA_COMPAT_VERSION "12.0.0")
set(CUDA_COMPAT_MAJOR "12")
set(CUDA_COMPAT_MINOR "0")

# Install layout mirroring CUDA toolkit
set(CUDA4AS_INSTALL_BINDIR "bin")
set(CUDA4AS_INSTALL_INCLUDEDIR "include")
set(CUDA4AS_INSTALL_LIBDIR "lib")

# Configure version files for CMake FindCUDAToolkit discovery
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.txt.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.json"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/version.json"
    DESTINATION "."
)

# Install headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CUDA4AS_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# Install nvvm/libdevice structure required by CMake CUDA discovery
install(DIRECTORY DESTINATION "nvvm/libdevice")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libdevice.10.bc" "")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libdevice.10.bc"
    DESTINATION "nvvm/libdevice"
)

# Install translator tool
install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/src/translator/cuda4as_translate.py"
    DESTINATION "${CUDA4AS_INSTALL_BINDIR}"
    RENAME "cuda4as-translate"
)

# Subdirectories
add_subdirectory(src/driver)
add_subdirectory(src/runtime)
