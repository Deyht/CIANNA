
<img src="cianna_logo_v1.1.png" alt="drawing" width="160" height="160"/>

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists - is a deep learning framework mainly used to address astrophysical problems. Functionalities and optimizations are mainly added as they are identified as relevant for astrophysical problems solving.

Main developer : David Cornu, FR - UTINAM / Univ. Franche Comté
david.cornu@utinam.cnrs.fr

See Copyight (C) and [License](#License) terms.



#############################################################
##          /!\ /!\ /!\ WARNING /!\ /!\ /!\
#############################################################

CIANNA is not in a "released" / "stable" state. It means that the framework and the associated interfaces are subject to significant changes between versions. One must pay attention to what have been changed before updating to a more recent version.


I can provide some help, information, advises, ... please contact
me at: david.cornu@observatoiredeparis.psl.eu

Be aware that i am not a support team all by myself.



#############################################################
##                         Installation
#############################################################

#### Dependencies

CIANNA is codded in C99 and requires at least a C compiler to be used. Additionally, it supports several compute methods:
- **C_NAIV**: No dependency, very simple CPU implementation (mainly for pedagogical purpose). Support basic multi-CPU with OpenMP.
- **C_BLAS**: Require OpenBLAS, much more optimized multi-CPU implementation. Non-matrix operations can also be multi-threaded with OpenMP. (We recommend an OpenMP installation for OpenBLAS)
- **C_CUDA**: (Recommended) Most efficient implementation relying on Nvidia GPU. Require a recent version of CUDA (At least CUDA 10.1, and latest CUDA 11.1 recommended)

#### How to install and compile

1. Clone the git repository

2. Edit the shell script *compile.cp* to update the few paths regarding your own system
(Check the various paths (GCC, CUDA, OpenBLAS, ...), for CUDA check all the references to cublas and nvcc, also think to update the -arch parameter to fit your GPU architecture)

3. Execute *compile.cp* to compile the source code.
The following arguments can be associated can be used to specify compute methods to compile
   - CUDA 	  : compile additional cuda files
   - OPEN_MP   : add multi-thread for some operations (for C_NAIV and C_BLAS)
   - BLAS 	  : add OpenBLAS gemm (mutli-threaded) operations
   - PY_INTERF : build the Python interface at the end

   Multiple parameters can be used at the same time:
   ```
   ./compile.cp CUDA OPEN_MP BLAS PY_INTERF
   ```
   NB: These parameters ***allow*** the use of specific features, they do not ***enable*** it. For example you can compile with all the parameters and choose to use CUDA or BLAS at execution time.

3. The previous script create a *main* executable which is by default a simple example performing MNIST classification.
If you choose to work using the C interface you must edit *src/main.c* and recompile using *compile.cp*. (A more convenient C interface is at work.)

#### Optional step

4. You can build a Python interface to use the framework.
First check if any path or compile option need to be adapted for your need in the file *src/python_module_setup.py* (GCC, CUDA, OpenBLAS, ...).
Then, the interface can be build automatically by adding the PY_INTERF argument to the *compile.cp* command, or manually by going into the *src* directory and execute:
   ```
   python3 python_module_setup.py build
   ```
   If you want to provide access to the framework system wide, you can use:
   ```
   sudo python3 python_module_setup.py install
   ```
   If you want to call the locally built interface you must add the path in your Python script (see example).

   The created Python interface module has no dependency with *main.c* and therefore
any code can be written with the interface with no need for new compilation.




#############################################################
##                   Various Recommandations
#############################################################

There is currently no tutorial on how to use either the main C code or the Python interface.
Users could have be given specific instructions by the author, or should have a good C (and CUDA) knowledge to figure out how to use the framework by themselves.

This framework take into account for various modern neural network optimizations. 
However, since there is no automated gradient optimization, it might happen that the network gradient "explode" or "vanish" on specific conditions. 
Users must be aware of those issues and be able to identify them in order to make proper use of the framework.


\ 


########################################################################################

#### License

These files are Copyright &copy; 2020 [David Cornu](https://github.com/Deyht/CIANNA), but released under the [Apache2 License](https://github.com/Deyht/CIANNA/blob/master/LICENSE.md).

\ 

#### Contributor License Agreement
*While you are free to duplicate and modify this repository under the Apache2 License above, by being allowed to submit a contribution to this repository you agree to the following terms:*

- *You grant to the present CIANNA framework your copyright license to reproduce and distribute your contributions and such derivative works.*

- *To the fullest extent permitted, you agree not to assert, all of your “moral rights” in or relating to your contributions for the benefit of the present CIANNA framework.*

- *Your contribution was created in whole or in part by you and you have the right to submit it under the open source license indicated in the LICENCE file; or the contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right to submit that work with modifications.*



