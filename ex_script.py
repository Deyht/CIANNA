import numpy as np
#import sys
#sys.path.insert(0,'/path/to/CIANNA/src/build/lib.linux-x86_64-X.X')
import CIANNA as cnn

import time


#Regular loading scheme

if(0):

	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument requires python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)

	start = time.perf_counter()
	
	cnn.init_network(np.array([28,28,1]),10,0.1,64,'C_CUDA', dynamic_load=1, mixed_precision=1)

	cnn.create_dataset("TRAIN", 60000, data_train, target_train, flat=0)
	cnn.create_dataset("VALID", 10000, data_valid, target_valid, flat=0)
	cnn.create_dataset("TEST", 10000, data_test, target_test, flat=0)
	
	del (data_train, target_train, data_valid, target_valid, data_test, target_test)
	
	cnn.conv_create(f_size=5, nb_filters=8, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	#cnn.conv_create(f_size=3, nb_filters=48, stride=1, padding=1, activation="RELU")
	cnn.dense_create(nb_neurons=1024, activation="RELU", drop_rate=0.5)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.2)
	cnn.dense_create(nb_neurons=10, activation="SOFTMAX")
	
	cnn.train_network(nb_epoch=10, learning_rate=0.0002, end_learning_rate=0.0001, control_interv=1, momentum=0.9, decay=0.009, confmat=1, shuffle_gpu=0, save_each=20)
	
	end = time.perf_counter()
	print(end-start)

	exit()



# Prevent memory double usage


if(0):

	cnn.init_network(np.array([28,28,1]),10,0.1,64,'C_CUDA', dynamic_load=0, mixed_precision=0)
	
	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument require python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)
	
	cnn.write_formated_dataset("train.dat", 60000, data_train, "FP32", target_train, "FP32", flat=0)
	cnn.write_formated_dataset("valid.dat", 10000, data_valid, "FP32", target_valid, "FP32", flat=0)
	cnn.write_formated_dataset("test.dat", 10000, data_test, "FP32", target_test, "FP32", flat=0)
	
	cnn.load_formated_dataset("TRAIN", "train.dat", "FP32", "FP32")
	cnn.load_formated_dataset("VALID", "valid.dat", "FP32", "FP32")
	cnn.load_formated_dataset("TEST", "test.dat", "FP32", "FP32")


	cnn.conv_create(f_size=5, nb_filters=8, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	#cnn.conv_create(f_size=3, nb_filters=48, stride=1, padding=1, activation="RELU")
	cnn.dense_create(nb_neurons=1024, activation="RELU", drop_rate=0.5)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.2)
	cnn.dense_create(nb_neurons=10, activation="SOFTMAX")
	
	cnn.train_network(nb_epoch=5, learning_rate=0.0002, end_learning_rate=0.0001, control_interv=1, momentum=0.9, decay=0.009, confmat=1, shuffle_gpu=0, save_each=20)

	exit()


# Prevent memory double usage and reduce intermediate disk usage


if(1):

	cnn.init_network(np.array([28,28,1]),10,0.1,16,'C_CUDA', dynamic_load=1, mixed_precision=1)

	
	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument require python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist_dat/mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)
	
	print (np.shape(target_train))
	#Data was already normalized, this is an exemple to illustrate how to reduce inermediate disk usage
	data_train *= 255.0
	data_valid *= 255.0
	data_test *= 255.0
	
	cnn.write_formated_dataset("train.dat", 60000, data_train, "UINT8", target_train, "UINT8", flat=0)
	cnn.write_formated_dataset("valid.dat", 10000, data_valid, "UINT8", target_valid, "UINT8", flat=0)
	cnn.write_formated_dataset("test.dat", 10000, data_test, "UINT8", target_test, "UINT8", flat=0)
	
	del (data_train, data_valid, data_test, target_train, target_valid, target_test)
	
	
	
	cnn.set_normalize_factors(np.array([0.0]),np.array([255.0]),28*28, np.array([0.0]),np.array([1.0]),10)
	
	cnn.load_formated_dataset("TRAIN", "train.dat", "UINT8", "UINT8")
	cnn.load_formated_dataset("VALID", "valid.dat", "UINT8", "UINT8")
	cnn.load_formated_dataset("TEST", "test.dat", "UINT8", "UINT8")

	

	cnn.conv_create(f_size=5, nb_filters=8, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	#cnn.conv_create(f_size=3, nb_filters=48, stride=1, padding=1, activation="RELU")
	cnn.dense_create(nb_neurons=1024, activation="RELU", drop_rate=0.5)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.2)
	cnn.dense_create(nb_neurons=10, activation="SOFTMAX")
	
	cnn.train_network(nb_epoch=5, learning_rate=0.0002, end_learning_rate=0.0001, control_interv=1, momentum=0.9, decay=0.009, confmat=1, shuffle_gpu=0, save_each=20)

	exit()













