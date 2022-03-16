
import numpy as np
from threading import Thread
#Uncomment to access a locally compiled version
#import sys
#sys.path.insert(0,'./src/build/lib.linux-x86_64-3.8')
import CIANNA as cnn

############################################################################
##              Data reading (your mileage may vary)
############################################################################

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

def create_augm_batch(data_raw, targ_raw, augm_size):
	
	data_augm = np.zeros((augm_size,np.shape(data_raw)[1]), dtype="float32")
	targ_augm  = np.zeros((augm_size,np.shape(targ_raw)[1]), dtype="float32")
	
	for i in range(0,augm_size):
		i_d = int(np.random.random()*np.shape(data_raw)[0])
		
		patch = np.copy(np.reshape(data_raw[i_d],(28,28)))
		
		patch = roll_zeropad(patch, np.random.randint(0,4), axis=0)
		patch = roll_zeropad(patch, np.random.randint(0,4), axis=1)
		
		data_augm[i] = np.copy(patch.flatten())
		targ_augm[i] = targ_raw[i_d]
	
	return data_augm, targ_augm

def data_augm_fct(data_raw, targ_raw, augm_size):
	data_augm, targ_augm = create_augm_batch(data_raw, targ_raw, augm_size)
	cnn.delete_dataset("TRAIN_buf")
	cnn.create_dataset("TRAIN_buf", 20000, data_augm, targ_augm)
	return



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

cnn.init_network(in_dim=i_ar([28,28]), in_nb_ch=1, out_dim=10, \
		bias=0.1, b_size=24, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A") #Change to C_BLAS or C_NAIV

data_augm, target_augm = create_augm_batch(data_train, target_train, 20000)
cnn.create_dataset("TRAIN", size=20000, input=data_augm, target=target_augm)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid)
cnn.create_dataset("TEST", size=10000, input=data_test, target=target_test)

#Used to load a saved network at a given epoch
load_step = 0
if(load_step > 0):
	cnn.load_network("net_save/net0_s%04d.dat"%(load_step), load_step)
else:
	cnn.conv_create(f_size=i_ar([5,5]), nb_filters=8, padding=i_ar([2,2]), activation="RELU")
	cnn.pool_create(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv_create(f_size=i_ar([5,5]), nb_filters=16, padding=i_ar([2,2]), activation="RELU")
	cnn.pool_create(p_size=i_ar([2,2]), p_type="MAX")
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.5)
	cnn.dense_create(nb_neurons=128, activation="RELU", drop_rate=0.2)
	cnn.dense_create(nb_neurons=10, activation="SOFTMAX")


for k in range(0,20):

	t = Thread(target=data_augm_fct(data_train, target_train, 20000))
	t.start()
	
	cnn.train_network(nb_epoch=2, learning_rate=0.0004, momentum=0.9, confmat=1, save_every=0, silent=1)
	
	t.join()
		
	cnn.swap_data_buffers("TRAIN")


cnn.perf_eval()

#Uncomment to save network prediction
cnn.forward_network(repeat=1)

exit()












