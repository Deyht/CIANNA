add_library(common
        activ_functions.c
        auxil.c
        conv_layer.c
        dense_layer.c
        initializers.c
        lrn_layer.c
        norm_layer.c
        pool_layer.c
        vars.c)

target_link_libraries(common PUBLIC ${PTHREAD_LIBS} ${OPENMP_LIBS} ${MATH_LIBRARY})
set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(cianna::common ALIAS common)

target_link_libraries(common PUBLIC cianna::backend::naive)

if (CIANNA_USE_BLAS)
    target_link_libraries(common PUBLIC cianna::backend::blas)
endif ()

if (CIANNA_USE_CUDA)
    target_link_libraries(common PUBLIC cianna::backend::cuda)
endif ()