
import numpy as np
import os

#Comment to access system wide install
import sys, glob
#if glob.glob('../../src/build/lib.*/'):
#	sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1]) #if compiled with compile.cp
if glob.glob('../../build/'):
	sys.path.insert(0,glob.glob('../../build/')[-1]) #if compiled with cmake
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

cnn.init(in_dim=i_ar([28,28]), in_nb_ch=1, out_dim=10, b_size=16,
		optimizer=cnn.adam(beta1=0.95), wema=1,
		comp_meth="C_CUDA", dynamic_load=1, mixed_precision="BF16C_FP32A")

cnn.create_dataset("TRAIN", size=60000, input=data_train, target=target_train)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid)
cnn.create_dataset("TEST" , size=10000, input=data_test, target=target_test)


a_relu = cnn.relu(leaking=0.1, saturation=640000.0)

def conv_res_block(nb_filters):
	cnn.norm(group_size=4, activation=a_relu)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=nb_filters, nb_groups=4, stride=i_ar([1,1]), padding=i_ar([1,1]), activation="LIN")
	cnn.norm(group_size=4, activation=a_relu)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=nb_filters, stride=i_ar([1,1]), padding=i_ar([1,1]), activation="LIN")
	l_layer = cnn.merge(-1, -5, "ADD")
	return l_layer


#Used to load a saved network at a given iteration
load_step = 0
if(load_step > 0):
	cnn.load("net_save/net0_s%04d.dat"%(load_step), load_step, bin=1)
else:
	cnn.conv(f_size=i_ar([5,5]), nb_filters=16 , padding=i_ar([2,2]), activation="LIN")
	conv_res_block(16)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.norm(group_size=4, activation=a_relu)
	
	cnn.conv(f_size=i_ar([5,5]), nb_filters=32, padding=i_ar([2,2]), activation="LIN")
	conv_res_block(32)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.norm(group_size=4, activation=a_relu)
	
	cnn.dense(nb_neurons=256, activation=a_relu, drop_rate=0.5)
	cnn.dense(nb_neurons=128, activation=a_relu, drop_rate=0.2)
	cnn.dense(nb_neurons=10 , strict_size=1, activation="SMAX")

#To create a latex table and associated pdf with the current architecture	
#cnn.print_arch_tex("./arch/", "arch", activation=1)


cnn.train(nb_iter=10, learning_rate=0.0002, weight_decay=0.0001, decoupled_wdecay=1, wema_rate=0.999, confmat=1, save_every=10, save_optim_every=0, save_bin=1, shuffle_every=0)
#cnn.perf_eval()

pred = cnn.forward(drop_mode="AVG_MODEL", no_error=0, saving=1, return_output=1)
print (pred)












