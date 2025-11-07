cmake_minimum_required(VERSION 4.0.2)

set(SKBUILD_PROJECT_VERSION "0.1.0" CACHE STRING "Project version")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(CIANNA VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)

# Select backends to enables
option(CIANNA_USE_BLAS "Enable OpenBLAS support" ON)
option(CIANNA_USE_CUDA "Enable CUDA support" ON)
option(CIANNA_CUDA_AUTO_GEN "Enable `nv_bfloat16`  / `__half` if supported" ON)

# Select features to enables
option(CIANNA_USE_OPENMP "Enable OpenMP library" ON)
option(CIANNA_USE_LPTHREAD "Link with pthread library" ON)

option(CIANNA_PYTHON_INTERFACE "Enable python interface" OFF)

# COMPILER DEFINES STRUCTURES
add_compile_definitions(
        MAX_LAYERS_NB=200
        MAX_NETWORKS_NB=10
        CUDA_THREADS_PER_BLOCKS=256
)

include(CheckPIESupported)
check_pie_supported(LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-unused-result -Wno-uninitialized -fmax-errors=2 -Wno-unknown-pragmas")
set(INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Backends
## BLAS
if (CIANNA_USE_BLAS)
    find_package(BLAS COMPONENTS C REQUIRED)
    add_compile_definitions(BLAS)
endif ()

## CUDA
if (CIANNA_USE_CUDA)
    enable_language(CUDA)
    check_pie_supported(LANGUAGES CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(CUDA)
    include(cmake/cuda.cmake)
    # TODO: add -arch=sm_89 -D GEN_AMPERE
endif ()


# Math library
find_library(MATH_LIBRARY m)
# OpenMP
if (CIANNA_USE_OPENMP)
    find_package(OpenMP COMPONENTS C REQUIRED)
    add_compile_definitions(OPEN_MP)
    set(OPENMP_LIBS OpenMP::OpenMP_C)
endif ()
# lpthread
if (CIANNA_USE_LPTHREAD)
    set(PTHREAD_LIBS pthread)
endif ()

add_subdirectory(src)

# install targets
if (SKBUILD OR CIANNA_PYTHON_INTERFACE)
    install(TARGETS CIANNA DESTINATION .)
endif ()
