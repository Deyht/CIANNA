
import numpy as np
import os

#Comment to access system wide install
import sys, glob
#if glob.glob('../../src/build/lib.*/'):
#	sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1]) #if compiled with compile.cp
if glob.glob('../../build/'):
	sys.path.insert(0,glob.glob('../../build/')[-1]) #if compiled with cmake

import CIANNA as cnn


if(not os.path.isdir("mnist_dat")):
	#share.obspm not available currently, so mnist.tar.gz is store in git for now 
	#os.system("wget https://share.obspm.fr/s/EkYR5B2Wc2gNis3/download/mnist.tar.gz")
	os.system("tar -xvzf mnist.tar.gz")

mnist_data = np.fromfile("mnist_dat/mnist_input.dat", dtype="float32")
mnist_data = np.reshape(mnist_data, (80000,28*28))
mnist_targets = np.fromfile("mnist_dat/mnist_target.dat", dtype="float32")
mnist_targets = np.reshape(mnist_targets, (80000,10))

data_train = mnist_data[:60000,:]
data_valid = mnist_data[60000:70000,:]
data_test  = mnist_data[70000:80000,:]

#Label smoothing
target_train = mnist_targets[:60000,:]*0.9 + 0.1/10
target_valid = mnist_targets[60000:70000,:]*0.9 + 0.1/10
target_test  = mnist_targets[70000:80000,:]*0.9 + 0.1/10



# Optimizer selection and configuration is available in init
# Input shape is no longer an option of init
cnn.init(b_size=16, optimizer=cnn.adam(beta1=0.90, beta2=0.999), wema=1,
		comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP16C_FP32A")

cnn.create_dataset("TRAIN", size=60000, input=data_train, target=target_train)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid)
# Target is no longer mandatory. Avoid creating an empty target array when using the model in inference only.
cnn.create_dataset("TEST" , size=10000, input=data_test)

a_relu = cnn.relu(leaking=0.05, saturation=640000.0)

load_step = 0
if(load_step > 0):
	cnn.load("optim_save/net_optim0_s%04d.dat"%(load_step), load_step, bin=1)
else:
    #New input layer, must provide spatial dimension and number of features
	im_input = cnn.input(spatial_shape=(28,28), nb_features=1)

    #All configuration keywords that expected numpy arrays now also support lists and tuple formats with variable length
	cnn.conv(f_size=(5,5), nb_filters=8 , padding=(2,2), activation=a_relu)
	cnn.pool(p_size=(2,2), p_type="MAX")
	cnn.norm(group_size=4)

	cnn.conv(f_size=(5,5), nb_filters=16, padding=(2,2), activation=a_relu)
	cnn.pool(p_size=(2,2), p_type="MAX")
	cnn.norm(group_size=4)

	cnn.dense(nb_neurons=256, activation=a_relu, drop_rate=0.8)
	cnn.dense(nb_neurons=128, activation=a_relu, drop_rate=0.5)
	cnn.dense(nb_neurons=10 , strict_size=1, activation="SMAX")


# Example of advanced learning rate scheduling
# Here cyclical cosine decay (currently configured with only one cycle)
warmup = 2
cycle_length = 40
cycle_decay = 0.8

l_r_max = 5e-4
l_r_min = 2e-6

for i in range(0,cycle_length): #adjust the number of cycle here

    if(i < warmup):
        l_r = l_r_min + (l_r_max - l_r_min) * min(i/warmup,1.0)
    else:
        cycle_id = i//cycle_length
        in_cycle_progress = (i%cycle_length)/cycle_length
        l_r = l_r_min + 0.5*(l_r_max*cycle_decay**cycle_id - l_r_min)*(1 + np.cos(np.pi*in_cycle_progress))

    # Some new optimization-related keywords. Also allow saving an optimizer complete save state for warm restart
    cnn.train(nb_iter=1, learning_rate=l_r,
        weight_decay=0.05, decoupled_wdecay=1, wema_rate=0.998, wema_replace_every=cycle_length,
    	confmat=3, control_interv=1, save_every=10, save_optim_every=10, save_bin=1)
    if(i == 0):
        cnn.perf_eval()

# Forward can now return a numpy array for subsequent processing without having to load a fwd_res/---.txt file
pred = cnn.forward(drop_mode="AVG_MODEL", no_error=1, saving=0, return_output=1)

pred_class = np.argmax(pred, axis=1)
targ_class = np.argmax(target_test, axis=1)

accuracy = len(np.where(pred_class == targ_class)[0])/len(targ_class)
print ("Test set accuracy: %4.2f %%"%(accuracy*100))










