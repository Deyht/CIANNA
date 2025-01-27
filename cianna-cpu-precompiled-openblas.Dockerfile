FROM alpine:3.21
WORKDIR /home
RUN apk add make nano gcc musl-dev gfortran openblas-dev git python3-dev py3-numpy-dev py3-setuptools py3-matplotlib
RUN git clone https://github.com/Deyht/CIANNA  && cd CIANNA && git reset --hard 8b34ac4f0fa5fd91adbaed0c8708e6cfa39335e9
WORKDIR /home/CIANNA
RUN sed  -i 's/openblas_include_dir="\/opt\/OpenBLAS\/include\/"/openblas_include_dir="\/usr\/include\/"/g' compile.cp
RUN sed  -i 's/openblas_lib_dir="\/opt\/OpenBLAS\/lib"/openblas_lib_dir="\/usr\/lib"/g' compile.cp
RUN sed  -i "s/blas_include = \['\/opt\/OpenBLAS\/include'\]/blas_include = \['\/opt\/lib'\]/g" src/python_module_setup.py
RUN sed  -i "s/blas_extra = \['-lopenblas', '-L\/opt\/OpenBLAS\/lib'\]/blas_extra = \['-lopenblas','-L\/usr\/lib'\]/g" src/python_module_setup.py
RUN ./compile.cp BLAS OPEN_MP PY_INTERF
# Below is to run the mnist example
ENV OMP_NUM_THREADS=4
RUN sed -i 's/comp_meth="C_CUDA"/comp_meth="C_BLAS"/g' examples/MNIST/mnist_train.py
# To have bash as an entry point
RUN apk add bash
SHELL ["/bin/bash","-c"]
RUN echo 'echo "This container is set to use ${OMP_NUM_THREADS} cores by default. You can adjust this by setting the OMP_NUM_THREADS variable to another value"' > /root/.bashrc
ENTRYPOINT ["/bin/bash"]
