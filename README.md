
<img src="cianna_logo_v1.1.png" alt="drawing" width="160" height="160"/>

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists - is a general purpose deep learning framework, but is mainly developed and used for astrophysical applications. Functionalities and optimizations are added based on relevance for (our subseset of) astrophysical problem solving.

Main developer : David Cornu, FR - LERMA / Observatoire de Paris, PSL
david.cornu@observatoiredeparis.psl.eu

See Copyight (C) and [License](#License) terms.



#############################################################
##          /!\ /!\ /!\ WARNING /!\ /!\ /!\
#############################################################

CIANNA is not in a "released" / "stable" state. The framework itself and the associated interface are subject to significant changes between versions (no guaranteed forward or backward compatibility for now). One must pay attention to what have been changed before performing updates.



#############################################################
##                         Installation
#############################################################

#### Dependencies

CIANNA is codded in C99 and requires at least a C compiler to be used. Additionally, it supports several compute methods:
- **C_NAIV**: No dependency, very simple CPU implementation (mainly for pedagogical purpose). Support basic multi-CPU with OpenMP.
- **C_BLAS**: Require OpenBLAS, much more optimized multi-CPU implementation. Non-matrix operations can also be multi-threaded with OpenMP. (An OpenMP installation for OpenBLAS is advised)
- **C_CUDA**: (Recommended) Most efficient implementation relying on Nvidia GPUs. It should work on GPUs from Maxwell to Ampere architecture, and can be compiled using CUDA 9.0 to CUDA 11.4, most recent being recommended.

More details on the [System Requirements](https://github.com/Deyht/CIANNA/wiki/Sytem-Requirements) page

#### How to install and compile

1. Clone the git repository

2. Edit the shell script *compile.cp* to update the few paths regarding your own system
   - Check the various paths (GCC, OpenBLAS, CUDA, ...)

   CUDA Only:
   - Check all the references to cublas and nvcc

     Edit cuda_arg="...":
     - Update the -arch parameter to fit your GPU architecture
     - Add -D CUDA_OLD if using CUDA < 11.1
     - Add -D GEN_VOLTA (Volta, Pascal, Turing) or -D GEN_AMPERE (Ampere only) for various mixed precision type support)

3. Execute *compile.cp* to compile the source code.
Each of the following optional argument adds support for a given compute methode
   - CUDA 	   : compile additional cuda files
   - OPEN_MP   : add multi-thread for some operations (for C_NAIV and C_BLAS)
   - BLAS 	   : add OpenBLAS gemm (mutli-threaded) operations
   - PY_INTERF : build the Python interface at the end

   Multiple parameters can be used at the same time:
   ```
   ./compile.cp CUDA OPEN_MP BLAS PY_INTERF
   ```
   NB: These parameters ***allow*** the use of specific features, they do not ***enable*** it. For example: one can compile with all the parameters and choose to use CUDA or BLAS at execution time.

4. The previous compilation script creates a *main* executable which is a simple example performing MNIST classification by default.
The C interface works by editing *src/main.c* and recompile using *compile.cp*. (A more convenient C interface is at work.)

#### Optional step

5. Build the Python interface.

    First check if any path or compile option need to be adapted for your need in the file *src/python_module_setup.py* (GCC, CUDA, OpenBLAS, ...).
Then, the interface can be build automatically by adding the PY_INTERF argument to the *compile.cp* command, or manually by going into the *src* directory and execute:
   ```
   python3 python_module_setup.py build
   ```
   To used the locally built interface the explicit build path must be given to the Python script (see example).
   
   To provide access to the framework system wide (including using the PY_INTERF option), execute into the *src* directory:
   ```
   sudo python3 python_module_setup.py install
   ```

   The created Python interface module has no dependency with *main.c*. Any Python code invoking CIANNA can be written with no need for new compilation.



#############################################################
##                   Various Recommandations
#############################################################

Please read the [How to use](https://github.com/Deyht/CIANNA/wiki/How-to-use-(Python-interface)) Wiki page. The Wiki page containing all the interface functions details is under construction.
Also "Troubleshooting" and "F.A.Q" pages will be added soon.

This framework take into account various modern neural network optimizations. 
However, since there is no fancy automated gradient optimization, it might happen that the network gradient "explode" or "vanish" on specific conditions. 
Users must be aware of those issues and able to identify them in order to make proper use of the framework.


\ 


########################################################################################

#### License

These files are Copyright &copy; 2020 [David Cornu](https://github.com/Deyht/CIANNA), but released under the [Apache2 License](https://github.com/Deyht/CIANNA/blob/master/LICENSE.md).

\ 

#### Contributor License Agreement
*While you are free to duplicate and modify this repository under the Apache2 License above, by being allowed to submit a contribution to this repository you agree to the following terms:*

- *You grant to the present CIANNA framework (and its Author) your copyright license to reproduce and distribute your contributions and such derivative works.*

- *To the fullest extent permitted, you agree not to assert, all of your “moral rights” in or relating to your contributions for the benefit of the present CIANNA framework.*

- *Your contribution was created in whole or in part by you and you have the right to submit it under the open source license indicated in the LICENCE file; or the contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right to submit that work with modifications.*



