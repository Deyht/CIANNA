#ifndef DEFS_H
#define DEFS_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <tgmath.h>
#include <string.h>
#include <sys/time.h>

#ifdef comp_CUDA
#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#endif
#endif

#ifdef BLAS
#include <cblas.h>
#endif

#ifdef OPEN_MP
#include <omp.h>
#endif

static const double two_pi = 2.0*3.14159265358979323846;

#define FLOAT
#define real float
#define cublasnrm2 cublasSnrm2
#define cublasgemm cublasSgemm


#endif //DEFS_H
