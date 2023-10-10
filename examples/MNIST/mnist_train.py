
import numpy as np
import matplotlib.pyplot as plt
import os

#Comment to access system wide install
import sys, glob
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn

############################################################################
##              Data reading (your mileage may vary)
############################################################################

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

if(not os.path.isdir("mnist_dat")):
	os.system("wget https://share.obspm.fr/s/EkYR5B2Wc2gNis3/download/mnist.tar.gz")
	os.system("tar -xvzf mnist.tar.gz")
	

print ("Reading inputs ... ", end = "", flush=True)

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

cnn.init(in_dim=i_ar([28,28]), in_nb_ch=1, out_dim=10,
		bias=0.1, b_size=16, comp_meth="C_CUDA", #Change to C_BLAS or C_NAIV
		dynamic_load=1, mixed_precision="FP32C_FP32A") 

cnn.create_dataset("TRAIN", size=60000, input=data_train, target=target_train)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid)
cnn.create_dataset("TEST", size=10000, input=data_test, target=target_test)

#Python side datasets are not required anymore, they can be released to save RAM
#del (data_train, target_train, data_valid, target_valid, data_test, target_test)

#Used to load a saved network at a given iteration
load_step = 0
if(load_step > 0):
	cnn.load("net_save/net0_s%04d.dat"%(load_step), load_step, bin=1)
else:
	cnn.conv(f_size=i_ar([5,5]), nb_filters=8 , padding=i_ar([2,2]), activation="RELU")
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([5,5]), nb_filters=16, padding=i_ar([2,2]), activation="RELU")
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.dense(nb_neurons=256, activation="RELU", drop_rate=0.5)
	cnn.dense(nb_neurons=128, activation="RELU", drop_rate=0.2)
	cnn.dense(nb_neurons=10, strict_size=1, activation="SMAX")


#To create a latex table and associated pdf with the current architecture	
#cnn.print_arch_tex("./arch/", "arch", activation=1)

cnn.train(nb_iter=20, learning_rate=0.004, momentum=0.8, confmat=1, save_every=0)
cnn.perf_eval()

#Uncomment to save network prediction
#cnn.forward(repeat=1, drop_mode="AVG_MODEL")














