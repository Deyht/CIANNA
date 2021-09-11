import time

import numpy as np
import sys
sys.path.insert(0,'/home/dcornu/Work/MINERVA/YOLO_SDC2_extra/CIANNA_last/src/build/lib.linux-x86_64-3.8')
import CIANNA as cnn

############################################################################
##              Data reading (your mileage may vary)
############################################################################

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

print ("Reading inputs ... ", end = "", flush=True)

#Loading clear formated files
#data = np.loadtxt("mnist_dat/mnist_input.txt", dtype="float32")
#target = np.loadtxt("mnist_dat/mnist_target.txt", dtype="float32")

#Loading binary files
data = np.fromfile("mnist_dat/mnist_input.dat", dtype="float32")
data = np.reshape(data, (80000,28*28))
target = np.fromfile("mnist_dat/mnist_target.dat", dtype="float32")
target = np.reshape(target, (80000,10))


data_train = data[:60000,:]
data_valid = data[60000:70000,:]
data_test  = data[70000:80000,:]

target_train = target[:60000,:]
target_valid = target[60000:70000,:]
target_test  = target[70000:80000,:]

print ("Done !", flush=True)

############################################################################
##               CIANNA network construction and use
############################################################################

#Details about the functions and parameters are given in the GitHub Wiki

cnn.init_network(dims=i_ar([28,28,1,1]), out_dim=10, \
		bias=0.1, b_size=32, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="TF32C_FP32A") #Change to C_BLAS or C_NAIV


cnn.create_dataset("TRAIN", size=60000, input=data_train, target=target_train)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid)
cnn.create_dataset("TEST", size=10000, input=data_test, target=target_test)

del (data_train, target_train, data_valid, target_valid, data_test, target_test)

#cnn.load_network("net_save/net0_s0010.dat", 10)

cnn.conv_create(f_size=i_ar([5,5,1]), nb_filters=8, stride=i_ar([1,1,1]), padding=i_ar([2,2,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.conv_create(f_size=i_ar([5,5,1]), nb_filters=16, stride=i_ar([1,1,1]), padding=i_ar([2,2,0]), activation="RELU")
cnn.pool_create(p_size=i_ar([2,2,1]), p_type="MAX")
cnn.dense_create(nb_neurons=512, activation="RELU", drop_rate=0.5)
#cnn.conv_create(f_size=i_ar([1,1,1]), nb_filters=128, stride=i_ar([1,1,1]), padding=i_ar([0,0,0]), activation="RELU", drop_rate = 0.5)
cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.2)
#cnn.conv_create(f_size=i_ar([1,1,1]), nb_filters=64, stride=i_ar([1,1,1]), padding=i_ar([0,0,0]), activation="RELU", drop_rate = 0.2)
cnn.dense_create(nb_neurons=10, activation="SOFTMAX")


cnn.train_network(nb_epoch=10, learning_rate=0.0004, momentum=0.9, confmat=1, save_each=10)
#Change save_each in previous function to save network weights
cnn.perf_eval()


#Uncomment to save network prediction
#cnn.forward_network(repeat=1)

exit()


