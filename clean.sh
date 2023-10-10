#!/bin/sh

#Root directory
rm main

#Compilation files
cd ./src/
rm *.o
rm *.gch

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


cd ../ImageNET

rm error.txt
rm -rf ImageNET_aux_data
rm *.tar.gz
rm *.jpg
rm -rf arch
rm -rf fwd_res
rm -rf net_save


cd ../PASCAL


cd ../COCO


cd ../SDC1


