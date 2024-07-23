#!/bin/sh

#Root directory
rm main

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
rm -rf __pycache__


cd ../ImageNET

rm error.txt
rm -rf ImageNET_aux_data
rm -rf bin_blocks
rm *.tar.gz
rm *.jpg
rm -rf arch
rm -rf fwd_res
rm -rf net_save
rm -rf __pycache__


cd ../PASCAL

rm error.txt
rm *.dat
rm *.tar.gz
rm *.tar
rm *.jpg
rm -rf VOCdevkit
rm -rf arch
rm -rf fwd_res
rm -rf net_save
rm -rf __pycache__


cd ../COCO

rm error.txt
rm *.dat
rm *.tar.gz
rm *.jpg
rm *.zip
rm -rf annotations
rm -rf train2017
rm -rf val2017
rm -rf arch
rm -rf fwd_res
rm -rf net_save
rm -rf __pycache__


cd ../SKAO_SDC1
rm error.txt
rm train_cat_norm_limits.txt
rm iter_score_list.txt
rm -rf figures
rm -rf SDC1_data
rm -rf arch
rm -rf fwd_res
rm -rf net_save
rm -rf __pycache__



