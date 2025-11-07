# CMake

Cmake is an out-of-source build system.
You can create a build directory as such:

```shell
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build
```

```shell
cmake --build build
```

## Options

This is a list of options that can be enabled or disabled to compile only specific features of CIANNA.
_They should be passed to CMake using the `-D` option._

| Variable             | Description                                   | Values     |
|----------------------|-----------------------------------------------|------------|
| CIANNA_USE_CUDA      | Enable CUDA support                           | `ON`/`OFF` |
| CIANNA_USE_BLAS      | Enable OpenBLAS support                       | `ON`/`OFF` |
| CIANNA_USE_OPENMP    | Enable OpenMP library                         | `ON`/`OFF` |
| CIANNA_USE_LPTHREAD  | Link with pthread library                     | `ON`/`OFF` |
| CIANNA_CUDA_AUTO_GEN | Enable `nv_bfloat16`  / `__half` if supported | `ON`/`OFF` |
| GEN_AMPERE           | Assume cuda support `__half`                  |            |
| GEN_VOLTA            | Assume cuda support `nv_bfloat16`             |            |

> Note: `GEN_AMPERE` and `GEN_VOLTA` should be passed without values, just `GEN_AMPERE` or `GEN_VOLTA`.

## Variables

This is a list of variables that can be used to modify the behavior of the build.
_They should be passed to CMake using the `-D` option._

| Variable                                                                                                         | Recommended | Alternative                   | Description                                        |
|------------------------------------------------------------------------------------------------------------------|-------------|-------------------------------|----------------------------------------------------|
| [CMAKE_BUILD_TYPE](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html#variable:CMAKE_BUILD_TYPE) | `Release`   | `Debug`, `RelWithDebInfo`     |                                                    |
| [CUDAToolkit_ROOT](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#search-behavior)              |             | -DCUDAToolkit_ROOT=/some/path |                                                    |
| [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)                 | `native`    | `89`                          | Allow to select the cuda architecture to use       |
| BLAS_DIR                                                                                                         |             |                               | Allow to specify a path to the blas implementation |
| BLA_VENDOR                                                                                                       |             | `OPENBLAS`                    | Allow to set the vendor to use                     |

> Example
> ```shell
> cmake -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OPENBLAS -DCMAKE_CUDA_ARCHITECTURES='native' -B build
> cmake --build build
> ```

## Check compile commands

Cmake is set to output a [compile_commands.json](build/compile_commands.json) file with the command used to compile the
code.

## Cuda stuff

By setting `CMAKE_CUDA_ARCHITECTURES` to `native` the build system will automatically detect the architecture of the
GPU.

Using `CheckSourceRuns`, `__half` and `nv_bfloat16` support can be checked by CMake easily.
`cmake/cuda.cmake` implement the necessary code to define, if supported, `GEN_AMPERE` and `GEN_VOLTA`.

> Note: May not work if targeting older gpu than the one used to compile the code.

```cmake
include(CheckSourceRuns)

check_source_runs(CUDA
        "
#include <cuda_fp16.h>
__global__ void f(__half x) {}
int main() { return 0; }
"
        GEN_VOLTA)
```

