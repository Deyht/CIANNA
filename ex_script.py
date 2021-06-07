

import numpy as np
import sys
sys.path.insert(0,'/home/dcornu/Work/MINERVA/YOLO_SDC2/CIANNA_exp/src/build/lib.linux-x86_64-3.8')
import CIANNA as cnn

############################################################################
##              Data reading (your mileage may vary)
############################################################################

print ("Reading inputs ... ", end = "", flush=True)
#max_rows argument requires python 3.7
#mnist.dat can be split to ease the reading
data_train = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=1,max_rows=60000*28,dtype="float32")
data_valid = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
data_test = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
print ("Done !",flush=True)

print ("Reading targets ... ", end = "", flush=True)
target_train = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
target_valid = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
target_test = np.loadtxt("mnist_dat/mnist.dat", \
	skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
print ("Done !", flush=True)

############################################################################
##               CIANNA network construction and use
############################################################################

#Details about the functions and parameters are given in the GitHub Wiki

cnn.init_network(dims=np.array([28,28,1]), out_dim=10, \
		bias=0.1, b_size=32,comp_meth='C_CUDA', dynamic_load = 1, mixed_precision = 1) #Change to C_BLAS or C_NAIV

cnn.create_dataset("TRAIN", size=60000, input=data_train, target=target_train, flat=0)
cnn.create_dataset("VALID", size=10000, input=data_valid, target=target_valid, flat=0)
cnn.create_dataset("TEST", size=10000, input=data_test, target=target_test, flat=0)

del (data_train, target_train, data_valid, target_valid, data_test, target_test)


cnn.conv_create(f_size=5, nb_filters=8, stride=1, padding=2, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=2, activation="RELU")
cnn.pool_create(pool_size=2)
#cnn.dense_create(nb_neurons=1024, activation="RELU", drop_rate=0.5)
cnn.conv_create(f_size=1, nb_filters=128, stride=1, padding=0, activation="RELU", drop_rate = 0.5)
#cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.2)
cnn.conv_create(f_size=1, nb_filters=64, stride=1, padding=0, activation="RELU", drop_rate = 0.2)
cnn.dense_create(nb_neurons=10, activation="SOFTMAX")


cnn.train_network(nb_epoch=30, learning_rate=0.0002, momentum=0.9, confmat=1, save_each=20)
#Change save_each in previous function to save network weights

#Uncomment to save network prediction
#cnn.forward_network(repeat=1)

exit()


