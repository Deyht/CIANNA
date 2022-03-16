
/*
	Copyright (C) 2020 David Cornu
	for the Convolutional Interactive Artificial 
	Neural Networks by/for Astrophysicists (CIANNA) Code
	(https://github.com/Deyht/CIANNA)

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/





#ifndef DEFS_H
#define DEFS_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <tgmath.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef comp_CUDA
#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
#include <cuda_fp16.h>
#endif

#if defined(GEN_AMPERE) 
#include <cuda_bf16.h>
#endif

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

#endif //DEFS_H







