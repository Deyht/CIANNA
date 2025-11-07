file(GLOB BACKEND_NAIVE_SRC *.c)
add_library(backend_naive ${BACKEND_NAIVE_SRC})

target_link_libraries(backend_naive PRIVATE ${PTHREAD_LIBS} ${OPENMP_LIBS} ${MATH_LIBRARY})
set_target_properties(backend_naive PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(cianna::backend::naive ALIAS backend_naive)
