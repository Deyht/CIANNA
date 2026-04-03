#!/bin/sh

#Root directory
rm main
rm -rf build/
rm -rf cianna.egg-info

#Compilation files
cd ./src/
rm *.o
rm *.gch
rm -rf dist/
rm -rf CIANNA.egg-info/

rm blas/*.o
rm cuda/*.o
rm naiv/*.o

rm -rf build/

#All examples data and by-products
cd ../examples/

cd MNIST

rm error.txt
rm *.tar.gz
rm -rf mnist_dat
rm -rf arch
rm -rf fwd_res
rm -rf net_save
rm -rf optim_save
rm -rf __pycache__


