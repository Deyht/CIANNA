#set(Python_FIND_VIRTUALENV FIRST)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# NumPy headers
execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

Python_add_library(CIANNA MODULE WITH_SOABI python_module.c)

target_link_libraries(CIANNA PRIVATE cianna::common Python::NumPy)
target_link_options(CIANNA PRIVATE --disable-gil)
