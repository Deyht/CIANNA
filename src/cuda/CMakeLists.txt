file(GLOB BACKEND_CUDA_SRC *.cu)
add_library(backend_cuda ${BACKEND_CUDA_SRC})

target_link_libraries(backend_cuda PRIVATE CUDA::curand CUDA::cudart CUDA::cublas)
set_target_properties(backend_cuda PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(cianna::backend::cuda ALIAS backend_cuda)
