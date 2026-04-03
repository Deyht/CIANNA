cmake_minimum_required(VERSION 3.16)

project(CIANNA LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==========================================================
# Options
# ==========================================================

option(BUILD_MODULE "Build CIANNA as a Python module instead of an executable" ON)
option(USE_OPENMP "Enable OpenMP parallelism" ON)
option(USE_BLAS "Enable OpenBLAS acceleration" ON)
option(USE_CUDA "Enable CUDA acceleration" ON)
set(ARCH "" CACHE STRING "Manual CUDA architecture override (e.g. 70, 75, 86, 89, 90). If empty, auto-detect from GPU.")
set(BLAS_PATH "" CACHE PATH "Manual BLAS path override. Must be the root folder containing lib/ and include/.")
set(CUDA_PATH "" CACHE PATH "Manual CUDA path override. Should point to the CUDA toolkit root (e.g. /usr/local/cuda).")

# ==========================================================
# Global compile options
# ==========================================================

add_definitions(
    -DMAX_LAYERS_NB=400
    -DMAX_NETWORKS_NB=10
)

# ==========================================================
# Sources
# ==========================================================

set(CIANNA_CORE
    src/conv_layer.c
    src/dense_layer.c
    src/pool_layer.c
    src/norm_layer.c
    src/lrn_layer.c
    src/activ_functions.c
    src/weights_initializers.c
    src/vars.c
    src/auxil.c
    src/dataset.c
    src/network.c
)

set(CIANNA_NAIV
    src/naiv/naiv_dense_layer.c
    src/naiv/naiv_conv_layer.c
    src/naiv/naiv_pool_layer.c
    src/naiv/naiv_norm_layer.c
)

set(CIANNA_BLAS "")
set(CIANNA_CUDA "")

# ==========================================================
# BLAS
# ==========================================================
if(USE_BLAS)

    set(BLAS_FOUND FALSE)
    set(HAVE_OPENBLAS 0)
    unset(BLAS_LIBRARIES)
    unset(BLAS_INCLUDE_DIR)

    if(BLAS_PATH)
        message(STATUS "BLAS_PATH specified: ${BLAS_PATH}")

        if(NOT IS_DIRECTORY "${BLAS_PATH}")
            message(WARNING "BLAS_PATH must be a root directory containing include/ and lib/ or lib64/: ${BLAS_PATH}")
        else()

            find_library(BLAS_MANUAL_LIB
                NAMES openblas blas
                PATHS
                    "${BLAS_PATH}/lib"
                    "${BLAS_PATH}/lib64"
                NO_DEFAULT_PATH
            )
            if(BLAS_MANUAL_LIB)
                set(BLAS_LIBRARIES "${BLAS_MANUAL_LIB}")
            else()

            endif()

            find_path(BLAS_MANUAL_INCLUDE
                NAMES cblas.h blas.h
                PATHS
                    "${BLAS_PATH}/include"
                NO_DEFAULT_PATH
            )
            if(BLAS_MANUAL_INCLUDE)
                set(BLAS_INCLUDE_DIR "${BLAS_MANUAL_INCLUDE}")
            else()
                message(WARNING "No BLAS include directory found under ${BLAS_PATH}/include")
            endif()


            if(BLAS_LIBRARIES AND BLAS_INCLUDE_DIR)
                set(BLAS_FOUND TRUE)
                message(STATUS "Using BLAS library from BLAS_PATH: ${BLAS_LIBRARIES}")
                message(STATUS "Using BLAS include directory from BLAS_PATH: ${BLAS_INCLUDE_DIR}")
            endif()
        endif()

    else()
        find_package(BLAS)

        if(BLAS_FOUND)
            find_path(BLAS_INCLUDE_DIR NAMES cblas.h blas.h )
            if (NOT BLAS_INCLUDE_DIR)
                message(WARNING "BLAS library found but no include directory with cblas.h or blas.h found.")
                set(BLAS_FOUND FALSE)
            endif()
        endif()

    endif()

    if(BLAS_FOUND)

        list(APPEND CIANNA_BLAS
            src/blas/blas_dense_layer.c
            src/blas/blas_conv_layer.c
        )

        if(BLAS_INCLUDE_DIR AND EXISTS "${BLAS_INCLUDE_DIR}/openblas_config.h")
            set(HAVE_OPENBLAS 1)
        else()
            message(STATUS "Blas library has been found, but it does not seems that it is the openblas implementation, some optimization features dedicated to openblas will be deactivated")
        endif()

        add_definitions(-DBLAS=1 -DHAVE_OPENBLAS=${HAVE_OPENBLAS})

        message(STATUS "BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
        message(STATUS "BLAS_INCLUDE_DIR = ${BLAS_INCLUDE_DIR}")

    else()
        message(STATUS "BLAS library not found, disabling USE_BLAS.")
        set(USE_BLAS OFF)
    endif()

endif()

# ==========================================================
# OpenMP
# ==========================================================

if(USE_OPENMP)

    find_package(OpenMP)

    if(OpenMP_C_FOUND)
        add_definitions(-DOPEN_MP=1)
    else()
        message("OpenMP not found, disabling USE_OPENMP.")
        set(USE_OPENMP OFF)
    endif()

endif()

# ==========================================================
# CUDA
# ==========================================================

set(CUDA_FOUND FALSE)
set(CUDA_INCLUDE_DIRS "")
set(CUDA_LINK_LIBS "")
set(CUDA_Old_Flag "")
set(CUDA_GEN_FLAG "")
set(CUDA_VERSION_STRING "")

if(USE_CUDA)

    if(CUDA_PATH)
        message(STATUS "CUDA_PATH specified: ${CUDA_PATH}")
        message(STATUS "Using manual CUDA discovery only (find_package(CUDAToolkit) will NOT be used).")

        if(NOT IS_DIRECTORY "${CUDA_PATH}")
            message(WARNING "CUDA_PATH is not a valid directory: ${CUDA_PATH}")
            set(USE_CUDA OFF)
        else()
            set(CUDA_NVCC "${CUDA_PATH}/bin/nvcc")
            set(CUDA_INCLUDE_DIR "${CUDA_PATH}/include")

            if(NOT EXISTS "${CUDA_NVCC}")
                message(WARNING "nvcc not found at: ${CUDA_NVCC}")
                set(USE_CUDA OFF)
            elseif(NOT IS_DIRECTORY "${CUDA_INCLUDE_DIR}")
                message(WARNING "CUDA include directory not found at: ${CUDA_INCLUDE_DIR}")
                set(USE_CUDA OFF)
            else()
                find_library(CUDA_CUDART_LIBRARY
                    NAMES cudart
                    PATHS
                        "${CUDA_PATH}/lib64"
                        "${CUDA_PATH}/lib"
                    NO_DEFAULT_PATH
                )

                find_library(CUDA_CUBLAS_LIBRARY
                    NAMES cublas
                    PATHS
                        "${CUDA_PATH}/lib64"
                        "${CUDA_PATH}/lib"
                    NO_DEFAULT_PATH
                )

                find_library(CUDA_CURAND_LIBRARY
                    NAMES curand
                    PATHS
                        "${CUDA_PATH}/lib64"
                        "${CUDA_PATH}/lib"
                    NO_DEFAULT_PATH
                )

                if(NOT CUDA_CUDART_LIBRARY)
                    message(WARNING "Could not find cudart under ${CUDA_PATH}/lib64 or ${CUDA_PATH}/lib")
                    set(USE_CUDA OFF)
                elseif(NOT CUDA_CUBLAS_LIBRARY)
                    message(WARNING "Could not find cublas under ${CUDA_PATH}/lib64 or ${CUDA_PATH}/lib")
                    set(USE_CUDA OFF)
                elseif(NOT CUDA_CURAND_LIBRARY)
                    message(WARNING "Could not find curand under ${CUDA_PATH}/lib64 or ${CUDA_PATH}/lib")
                    set(USE_CUDA OFF)
                else()
                    set(CMAKE_CUDA_COMPILER "${CUDA_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
                    enable_language(CUDA)

                    set(CUDA_FOUND TRUE)
                    set(CUDA_INCLUDE_DIRS "${CUDA_INCLUDE_DIR}")
                    set(CUDA_LINK_LIBS
                        "${CUDA_CUDART_LIBRARY}"
                        "${CUDA_CUBLAS_LIBRARY}"
                        "${CUDA_CURAND_LIBRARY}"
                    )

                    message(STATUS "Using manually specified nvcc: ${CMAKE_CUDA_COMPILER}")
                    message(STATUS "CUDA include directory: ${CUDA_INCLUDE_DIRS}")
                    message(STATUS "CUDA libraries: ${CUDA_LINK_LIBS}")

                    # Try to get CUDA version from nvcc
                    execute_process(
                        COMMAND "${CUDA_NVCC}" --version
                        OUTPUT_VARIABLE CUDA_NVCC_VERSION_OUT
                        ERROR_VARIABLE CUDA_NVCC_VERSION_ERR
                        RESULT_VARIABLE CUDA_NVCC_VERSION_RESULT
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                    )

                    if(CUDA_NVCC_VERSION_RESULT EQUAL 0)
                        string(REGEX MATCH "V([0-9]+\\.[0-9]+)" _cuda_ver_match "${CUDA_NVCC_VERSION_OUT}")
                        if(CMAKE_MATCH_1)
                            set(CUDA_VERSION_STRING "${CMAKE_MATCH_1}")
                            message(STATUS "Detected CUDA version from nvcc: ${CUDA_VERSION_STRING}")
                        endif()
                    endif()
                endif()
            endif()
        endif()

    else()
        find_package(CUDAToolkit)

        if(CUDAToolkit_FOUND)
            enable_language(CUDA)

            set(CUDA_FOUND TRUE)
            set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
            set(CUDA_LINK_LIBS
                CUDA::cudart
                CUDA::cublas
                CUDA::curand
            )
            set(CUDA_VERSION_STRING "${CUDAToolkit_VERSION}")

            message(STATUS "Found CUDAToolkit: ${CUDAToolkit_FOUND}")
            message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")
        else()
            message("CUDA toolkit not found, disabling USE_CUDA.")
            set(USE_CUDA OFF)
        endif()
    endif()

    if(USE_CUDA AND CUDA_FOUND)

        set(CIANNA_CUDA
            src/cuda/cuda_auxil.cu
            src/cuda/cuda_conv_layer.cu
            src/cuda/cuda_dense_layer.cu
            src/cuda/cuda_pool_layer.cu
            src/cuda/cuda_norm_layer.cu
            src/cuda/cuda_lrn_layer.cu
            src/cuda/cuda_activ_functions.cu
        )

        if(CUDA_VERSION_STRING AND CUDA_VERSION_STRING VERSION_LESS "11.1")
            set(CUDA_Old_Flag CUDA_OLD=1)
            message(STATUS "CUDA version is less than 11.1, adding CUDA_OLD variable.")
        endif()

        # ----------------------------------------------------------
        # CUDA architecture selection
        # - If ARCH is provided manually, use it
        # - Otherwise detect GPU compute capability from device 0
        # ----------------------------------------------------------

        set(CUDA_CC_RAW "")
        set(CUDA_ARCH "")

        if(ARCH)
            set(CUDA_ARCH "${ARCH}")
            message(STATUS "Using manually specified CUDA architecture: ${CUDA_ARCH}")

            # Try to reconstruct a dotted compute capability string for version comparisons
            string(LENGTH "${CUDA_ARCH}" ARCH_LEN)
            if(ARCH_LEN EQUAL 2)
                string(SUBSTRING "${CUDA_ARCH}" 0 1 ARCH_MAJOR)
                string(SUBSTRING "${CUDA_ARCH}" 1 1 ARCH_MINOR)
                set(CUDA_CC_RAW "${ARCH_MAJOR}.${ARCH_MINOR}")
            elseif(ARCH_LEN EQUAL 3)
                string(SUBSTRING "${CUDA_ARCH}" 0 2 ARCH_MAJOR)
                string(SUBSTRING "${CUDA_ARCH}" 2 1 ARCH_MINOR)
                set(CUDA_CC_RAW "${ARCH_MAJOR}.${ARCH_MINOR}")
            else()
                message(WARNING "ARCH='${ARCH}' has an unexpected format. Version-based CUDA generation flags may be inaccurate.")
            endif()

            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")

        else()
            # ----------------------------------------------------------
            # Detect GPU compute capability from device 0
            # ----------------------------------------------------------
            set(CUDA_CC_SRC "${CMAKE_BINARY_DIR}/detect_cuda_cc.cu")
            set(CUDA_CC_BIN "${CMAKE_BINARY_DIR}/detect_cuda_cc${CMAKE_EXECUTABLE_SUFFIX}")

            file(WRITE ${CUDA_CC_SRC}
            "#include <stdio.h>
            #include <cuda_runtime.h>

            int main() {
                cudaDeviceProp p;
                cudaError_t err = cudaGetDeviceProperties(&p, 0);
                if (err != cudaSuccess) {
                    return 1;
                }
                printf(\"%d.%d\\n\", p.major, p.minor);
                return 0;
            }
            ")

            execute_process(
                COMMAND ${CMAKE_CUDA_COMPILER} ${CUDA_CC_SRC} -o ${CUDA_CC_BIN}
                RESULT_VARIABLE CUDA_CC_COMPILE_RESULT
                OUTPUT_VARIABLE CUDA_CC_COMPILE_OUT
                ERROR_VARIABLE CUDA_CC_COMPILE_ERR
            )

            if(CUDA_CC_COMPILE_RESULT EQUAL 0)
                execute_process(
                    COMMAND ${CUDA_CC_BIN}
                    RESULT_VARIABLE CUDA_CC_RUN_RESULT
                    OUTPUT_VARIABLE CUDA_CC_RAW
                    ERROR_VARIABLE CUDA_CC_RUN_ERR
                )

                if(CUDA_CC_RUN_RESULT EQUAL 0)
                    string(STRIP "${CUDA_CC_RAW}" CUDA_CC_RAW)
                    string(REPLACE "." "" CUDA_ARCH "${CUDA_CC_RAW}")
                    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
                    message(STATUS "Detected GPU compute capability: ${CUDA_CC_RAW}")
                    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")

                else()
                    message(FATAL_ERROR "Failed to run CUDA compute capability detector: ${CUDA_CC_RUN_ERR}")
                endif()

            else()
                message(FATAL_ERROR "Failed to compile CUDA compute capability detector:\n${CUDA_CC_COMPILE_ERR}")
            endif()
        endif()

        # Optional architecture-specific define
        if(CUDA_CC_RAW VERSION_GREATER_EQUAL "7.0" AND CUDA_CC_RAW VERSION_LESS "8.0")
            set(CUDA_GEN_FLAG GEN_VOLTA=1)
            message(STATUS "Added Flag GEN_VOLTA")
        elseif(CUDA_CC_RAW VERSION_GREATER_EQUAL "8.0")
            set(CUDA_GEN_FLAG GEN_AMPERE=1)
            message(STATUS "Added Flag GEN_AMPERE")
        else()
            set(CUDA_GEN_FLAG "")
        endif()

    elseif(USE_CUDA)
        message("CUDA setup failed, disabling USE_CUDA.")
        set(USE_CUDA OFF)
    endif()

endif()

message("USE_CUDA : " ${USE_CUDA})

# ==========================================================
# Main executable
# ==========================================================

if(NOT BUILD_MODULE)
    add_executable(cianna
        src/main.c
        ${CIANNA_CORE}
        ${CIANNA_NAIV}
        ${CIANNA_BLAS}
        ${CIANNA_CUDA}
    )

    if(USE_CUDA)
        target_compile_definitions(cianna PRIVATE
            $<$<COMPILE_LANGUAGE:C>:CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA_THREADS_PER_BLOCKS=256>
            $<$<COMPILE_LANGUAGE:CUDA>:comp_CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_Old_Flag}>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_GEN_FLAG}>
        )
    endif()

    # C compilation flags handling
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    target_compile_options(cianna PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Werror -Wno-unknown-pragmas>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${CUDA_ARCH},code=[compute_${CUDA_ARCH},sm_${CUDA_ARCH}]>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(cianna PRIVATE $<$<COMPILE_LANGUAGE:C>:-fmax-errors=2 -Wno-unused-result -Wno-uninitialized>)
    endif()

    target_include_directories(cianna PRIVATE src)

    target_link_libraries(cianna m)

    if(USE_BLAS)
        target_include_directories(cianna PRIVATE ${BLAS_INCLUDE_DIR})
        target_link_libraries(cianna ${BLAS_LIBRARIES})
    endif()

    if(USE_OPENMP)
        target_link_libraries(cianna OpenMP::OpenMP_C)
    endif()

    if(USE_CUDA)
        target_include_directories(cianna PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(cianna ${CUDA_LINK_LIBS})
    endif()
endif()

# ==========================================================
# Python module
# ==========================================================
if(BUILD_MODULE)

    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

    add_library(cianna_module MODULE
        src/python_module.c
        ${CIANNA_CORE}
        ${CIANNA_NAIV}
        ${CIANNA_BLAS}
        ${CIANNA_CUDA}
    )

    target_include_directories(cianna_module PRIVATE
        src
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )

    target_link_libraries(cianna_module PRIVATE
        Python3::Python
        m
    )

    # C compilation flags handling
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    target_compile_options(cianna_module PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Werror -Wno-unknown-pragmas>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${CUDA_ARCH},code=[compute_${CUDA_ARCH},sm_${CUDA_ARCH}]>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(cianna_module PRIVATE $<$<COMPILE_LANGUAGE:C>:-fmax-errors=2 -Wno-unused-result -Wno-uninitialized>)
    endif()

    if(USE_BLAS)
        target_include_directories(cianna_module PRIVATE ${BLAS_INCLUDE_DIR})
        target_link_libraries(cianna_module PRIVATE ${BLAS_LIBRARIES})
    endif()

    if(USE_OPENMP)
        target_link_libraries(cianna_module PRIVATE OpenMP::OpenMP_C)
    endif()

    if(USE_CUDA)
        target_compile_definitions(cianna_module PRIVATE
            $<$<COMPILE_LANGUAGE:C>:CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA_THREADS_PER_BLOCKS=256>
            $<$<COMPILE_LANGUAGE:CUDA>:comp_CUDA=1>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_Old_Flag}>
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_GEN_FLAG}>
        )

        target_include_directories(cianna_module PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(cianna_module PRIVATE ${CUDA_LINK_LIBS})
    endif()

    set_target_properties(cianna_module PROPERTIES
        PREFIX ""
        OUTPUT_NAME "CIANNA"
    )

endif()
