include(CheckSourceRuns)
# The following tests should ensure that the features are enabled only when supported.

if (CIANNA_CUDA_AUTO_GEN)
    check_source_runs(CUDA
            "
#include <cuda_fp16.h>
__global__ void f(__half x) {}
int main() { return 0; }
"
            GEN_VOLTA)

    check_source_runs(CUDA
            "
#include <cuda_bf16.h>
__global__ void f(nv_bfloat16 x) {}
int main() { return 0; }
"
            GEN_AMPERE)
endif ()
