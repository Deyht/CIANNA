import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_temp / 'bin'}",
            "-DPY_INTERF=ON",
            "-DBUILD_CLI=OFF",
        ]

        # For multi-config generators
        cmake_args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={build_temp / 'bin'}",
        ]

        # Pass Python executable explicitly
        cmake_args.append(f"-DPython3_EXECUTABLE={sys.executable}")

        # Optional toggles from environment
        for var in ["USE_CUDA", "USE_BLAS", "USE_OPENMP", "PY_INTERF", "BUILD_CLI"]:
            value = os.environ.get(var)
            if value is not None:
                cmake_args.append(f"-D{var}={value}")

        # Generic extra CMake args
        extra_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if extra_cmake_args:
            cmake_args += shlex.split(extra_cmake_args)

        build_args = ["--config", cfg]

        # Parallel build if available
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += ["--parallel", str(self.parallel)]

        print("Running CMake configure with:", cmake_args)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
        )

        print("Running CMake build with:", build_args)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )


setup(
    name="cianna",
    version="1.0.1.3",
    description="CIANNA Python bindings",
    long_description="CIANNA Python bindings built with setuptools + CMake",
    ext_modules=[CMakeExtension("CIANNA", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
