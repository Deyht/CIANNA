file(GLOB BACKEND_BLAS_SRC *.c)
add_library(backend_blas ${BACKEND_BLAS_SRC})

target_link_libraries(backend_blas PRIVATE BLAS::BLAS ${PTHREAD_LIBS} ${OPENMP_LIBS} ${MATH_LIBRARY})
set_target_properties(backend_blas PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(cianna::backend::blas ALIAS backend_blas)
