########################################################################################
# CIANNA - Convolutional Interactive Artificial Neural Network by/for Astrophysicists
########################################################################################
CIANNA - Convolutional Interactive Artificial Neural Network by/for Astrophysicists - is a deep learning framework developed to address astrophysical problems. Functionalities and optimizations are added as they are identified as relevant for astrophysical problems solving.

Main developer : David Cornu, FR - UTINAM / Univ. Franche Comté
david.cornu@utinam.cnrs.fr

This work is done in the context of a PhD thesis funded by the CNES,
usage must have received an explicit authorization.

All rights reserved to the Author.





############################################
# /!\ /!\ /!\ /!\ WARNING /!\ /!\ /!\ /!\
############################################

THIS FRAMEWORK IS NEITHER IN A STABLE NOR FINISHED STATE.
IF YOU WHERE GRANTED ACCESS TO THIS REPOSITORY / CODE BY ANY OTHER WAY
THAN THROUGH A CONTACT WITH THE AUTHOR, PLEASE TAKE CONTACT BEFORE USING IT.

If in need for any help, information, advises, ... please contact
me at : david.cornu@utinam.cnrs.fr

############################################
# /!\ /!\ /!\ /!\ WARNING /!\ /!\ /!\ /!\
############################################








############################################
#               INSTALLATION
############################################
WARNING : Currently, the framework only work using CUDA (version 9.2 minimum, 10.1 up 2 recommended).
It will soon support basic CPU implementation and an OpenBLAS version, both with an OpenMP multi-thread support.


1. Edit the shell script "train.cp" to update the few paths regarding your own system

2. Execute "train.cp" to compile the source code
It can (and currently must) be associated with parameters to specify specific parts to compile
CUDA 	: compile additional cuda files
OPEN_MP : add some multi-thread for some operations
BLAS 	: add OpenBLAS gemm (mutli-threaded) operations

Multiple parameters can be used at the same time ex:
$ ./train.cp CUDA OPEN_MP BLAS

NB: These parameters "allow" the use of specific features, they do not "enable" it. For example you can compile
with all the parameters and choose to use CUDA or BLAS at execution time.

3. It creates a "main" executable which is by default a simple example performing MNIST classification.
If you choose to work using the C language you must edit "src/main.c" and recompile using "train.cp".

4. OPTIONAL: You can build a Python interface to use the framework.
To do so you must compile with the desire options using "train.cp".
Then you can go in the "src" directory and execute:
$ Python3 python_module_setup.py build

If you want to provide access to the framework system wide, you can use:
$ sudo Python3 python_module_setup.py install

If you want to call the locally built interface you must add the path in you python script (see example) 

The created python interface module has no dependency with "main.c" and therefore
any code can be written with the interface with no need for new compilation.




############################################
#          VARIOUS RECOMMENDATIONS
############################################

There is currently no tutorial on how to use either the main C code or the Python interface.
Users may have be given specific instructions by the author, or should have a good C (and CUDA) knowledge
to figure out how to use the framework by themselves.

This framework take into account for various modern neural network optimizations. 
However, since there is no automated gradient optimization, it will happen that the network gradient "explode" or"vanish". 
Users must be aware of those issues and be able to identify them in order to make proper use of the framework.












