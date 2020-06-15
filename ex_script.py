import numpy as np
#sys.path.append('/path/to/CIANNA/src/build/lib.linux-x86_64-X.X')
import CIANNA as cnn




#Regular loading scheme

if(0):

	cnn.init_network(np.array([28,28,1]),10,0.1,32,'C_CUDA', dynamic_load=1)


	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument require python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)


	cnn.create_dataset("TRAIN", 60000, data_train, target_train, flat=0)
	cnn.create_dataset("VALID", 10000, data_valid, target_valid, flat=0)
	cnn.create_dataset("TEST", 10000, data_test, target_test, flat=0)


	del (data_train, data_valid, data_test, target_train, target_valid, target_test)

	cnn.conv_create(f_size=5, nb_filters=6, stride=1, padding=0, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=4, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(10, activation="SOFTMAX")


	cnn.train_network(nb_epoch=10, learning_rate=0.0003, end_learning_rate=0.0001, control_interv=1, momentum=0.7, decay=0.009, save_each=0, shuffle_gpu=0, shuffle_every=1, confmat=1)

	exit()



# Prevent memory double usage


if(0):

	cnn.init_network(np.array([28,28,1]),10,0.1,32,'C_CUDA', dynamic_load=1)


	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument require python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)
	
	cnn.write_formated_dataset("train.dat", 60000, data_train, "FP32", target_train, "FP32", flat=0)
	cnn.write_formated_dataset("valid.dat", 10000, data_valid, "FP32", target_valid, "FP32", flat=0)
	cnn.write_formated_dataset("test.dat", 10000, data_test, "FP32", target_test, "FP32", flat=0)
	
	
	del (data_train, data_valid, data_test, target_train, target_valid, target_test)
	
	cnn.load_formated_dataset("TRAIN", "train.dat", "FP32", "FP32")
	cnn.load_formated_dataset("VALID", "valid.dat", "FP32", "FP32")
	cnn.load_formated_dataset("TEST", "test.dat", "FP32", "FP32")


	cnn.conv_create(f_size=5, nb_filters=6, stride=1, padding=0, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=4, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(10, activation="SOFTMAX")


	cnn.train_network(nb_epoch=1, learning_rate=0.0003, end_learning_rate=0.0001, control_interv=1, momentum=0.7, decay=0.009, save_each=0, shuffle_gpu=0, shuffle_every=1, confmat=1)

	exit()


# Prevent memory double usage and reduce intermediate disk usage


if(1):

	cnn.init_network(np.array([28,28,1]),10,0.1,128,'C_CUDA', dynamic_load=1)


	print ("Reading inputs ... ", end = "", flush=True)
	#max_rows argument require python 3.7
	#mnist.dat can be split to ease the reading
	data_train = np.loadtxt("mnist.dat", skiprows=1,max_rows=60000*28,dtype="float32")
	data_valid = np.loadtxt("mnist.dat", skiprows=(60000*28)+2,max_rows=10000*28,dtype="float32")
	data_test = np.loadtxt("mnist.dat", skiprows=(70000*28)+3,max_rows=10000*28,dtype="float32")
	print ("Done !",flush=True)

	print (np.shape(data_train))

	print ("Reading targets ... ", end = "", flush=True)
	target_train = np.loadtxt("mnist.dat", skiprows=(80000*28)+4,max_rows=60000,dtype="float32")
	target_valid = np.loadtxt("mnist.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000,dtype="float32")
	target_test = np.loadtxt("mnist.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000,dtype="float32")
	print ("Done !", flush=True)
	
	#Data was already normalized, this is an exemple to illustrate how to reduce inermediate disk usage
	data_train *= 255.0
	data_valid *= 255.0
	data_test *= 255.0
	
	
	cnn.write_formated_dataset("train.dat", 60000, data_train, "UINT8", target_train, "UINT8", flat=0)
	cnn.write_formated_dataset("valid.dat", 10000, data_valid, "UINT8", target_valid, "UINT8", flat=0)
	cnn.write_formated_dataset("test.dat", 10000, data_test, "UINT8", target_test, "UINT8", flat=0)
	
	
	del (data_train, data_valid, data_test, target_train, target_valid, target_test)
	
	cnn.load_formated_dataset("TRAIN", "train.dat", "UINT8", "UINT8")
	cnn.load_formated_dataset("VALID", "valid.dat", "UINT8", "UINT8")
	cnn.load_formated_dataset("TEST", "test.dat", "UINT8", "UINT8")
	
	#Normalize directly all dataset loaded in the C framework (for a specific network, default 0)
	cnn.normalize_datasets(np.array([0.0]),np.array([255.0]),28*28, np.array([0.0]),np.array([1.0]),10)


	cnn.conv_create(f_size=5, nb_filters=6, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=5, nb_filters=12, stride=1, padding=2, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=3, nb_filters=18, stride=1, padding=1, activation="RELU")
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(nb_neurons=256, activation="RELU", drop_rate=0.0)
	cnn.dense_create(10, activation="SOFTMAX")


	cnn.train_network(nb_epoch=20, learning_rate=0.0002, end_learning_rate=0.0001, control_interv=1, momentum=0.7, decay=0.009, save_each=0, shuffle_gpu=0, shuffle_every=1, confmat=1)

	exit()













