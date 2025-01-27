FROM alpine:3.21
WORKDIR /home
RUN apk add make nano gcc musl-dev gfortran git python3-dev py3-numpy-dev py3-setuptools py3-matplotlib
# Manual OpenBLAS install
RUN git clone https://github.com/xianyi/OpenBLAS
WORKDIR /home/OpenBLAS
RUN make USE_OPENMP=1
RUN make install
RUN echo 'export PATH=/opt/OpenBLAS/include${PATH:+:${PATH}}' > /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/opt/OpenBLAS/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /root/.bashrc
RUN source /root/.bashrc
# Main CIANNA part
WORKDIR /home
RUN git clone https://github.com/Deyht/CIANNA  && cd CIANNA && git reset --hard 8b34ac4f0fa5fd91adbaed0c8708e6cfa39335e9
WORKDIR /home/CIANNA
RUN ./compile.cp BLAS OPEN_MP PY_INTERF
# Below is to run the mnist example
ENV OMP_NUM_THREADS=4
RUN sed -i 's/comp_meth="C_CUDA"/comp_meth="C_BLAS"/g' examples/MNIST/mnist_train.py
# To have bash as an entry point
RUN apk add bash
SHELL ["/bin/bash","-c"]
RUN echo 'echo "This container is set to use ${OMP_NUM_THREADS} cores by default. You can adjust this by setting the OMP_NUM_THREADS variable to another value"' >> /root/.bashrc
ENTRYPOINT ["/bin/bash"]
