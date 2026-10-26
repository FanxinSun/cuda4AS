add_executable(nvcc nvcc_main.cpp)

target_compile_definitions(nvcc PRIVATE
    CUDA4AS_VERSION="${PROJECT_VERSION}"
    CUDA_COMPAT_VERSION="${CUDA_COMPAT_VERSION}"
)

add_executable(nvlink nvlink_main.cpp)
add_executable(fatbinary fatbinary_main.cpp)

install(TARGETS nvcc nvlink fatbinary
    RUNTIME DESTINATION "${CUDA4AS_INSTALL_BINDIR}"
)
