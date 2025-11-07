# Backends
add_subdirectory(naiv)

if (CIANNA_USE_BLAS)
    add_subdirectory(blas)
endif ()

if (CIANNA_USE_CUDA)
    add_subdirectory(cuda)
endif ()

# common
include(common.cmake)

# Python interface
if (SKBUILD OR CIANNA_PYTHON_INTERFACE)
    include(python.cmake)
endif ()

# main
include(main.cmake)
