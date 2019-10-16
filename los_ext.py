import numpy as np
import sys
#sys.path.append('/Home/Users/dcornu/WORK/GALMAP/CIANNA/src/build/lib.linux-x86_64-3.6')
import CIANNA as cnn
import matplotlib.pyplot as plt
import random

cnn.init_network(np.array([32,32,1]),100,8,'C_CUDA')


data = np.loadtxt("../raw_data/fancy/test3/CMDs.txt",skiprows=2,max_rows=100000)

target = np.loadtxt("../raw_data/fancy/test3/Profiles.txt", skiprows=1, max_rows=100000)

data = np.reshape(data, (100000,64,64))

data2 = np.zeros((100000, 32, 32))

import skimage.measure
for i in range(0,100000):
	data2[i] = skimage.measure.block_reduce(data[i], (2,2), np.max)

data = np.reshape(data2, (100000,32*32))

"""
target = target[:]
index = np.where(np.argmax(target,axis=1) > 100)

data[index,64*64:] = data[index,:64*64]

target = target[:,0:100]
"""

"""
index = np.array(range(0,100000))

random.shuffle(index)
print (np.shape(index))

data = data[index[:],:]
target = target[index[:],:]
"""

val_max = np.max(data)
print ("max: ", val_max)
data_train = data[:90000]
data_valid = data[90000:95000]
data_test = data[95000:]

print (np.max(data_train), np.max(data_valid), np.max(data_test))


data_train /= val_max
data_valid /= val_max
data_test /= val_max

print (np.shape(data_train))
print (np.shape(data_valid))
print (np.shape(data_test))


target_train = target[:90000]
target_valid = target[90000:95000]
target_test = target[95000:]


out_norm = 6.

target_train /= out_norm
target_valid /= out_norm
target_test /= out_norm


print (np.shape(target_train))
print (np.shape(target_valid))
print (np.shape(target_test))

np.savetxt("../raw_data/test_set_CMDS.txt", data_test)
np.savetxt("../raw_data/test_set_Profiles.txt", target_test)

#plt.hist(np.argmax(target_train[:,:], axis = 1),bins=150)

cnn.create_dataset("TRAIN", 90000, data_train, target_train, 0.1, 1)
cnn.create_dataset("VALID", 5000, data_valid, target_valid, 0.1, 1)
cnn.create_dataset("TEST", 5000, data_test, target_test, 0.1, 1)


load_net = 0


if (load_net <= 0):
	"""
	cnn.conv_create(f_size=5, nb_filters=8, stride=1, padding=2, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=12, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=12, stride=1, padding=0, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=3, nb_filters=24, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=24, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=24, stride=1, padding=0, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.dense_create(nb_neurons=2048, activation="RELU",drop_rate=0.0)
	cnn.dense_create(nb_neurons=2048, activation="RELU",drop_rate=0.0)
	cnn.dense_create(nb_neurons=100, activation="LINEAR")
	"""
	
	cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=0, activation="RELU")
	cnn.pool_create(pool_size=2)
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
	cnn.dense_create(nb_neurons=2048, activation="RELU",drop_rate=0.0)
	cnn.dense_create(nb_neurons=2048, activation="RELU",drop_rate=0.0)
	cnn.dense_create(nb_neurons=100, activation="LINEAR")
	
else:
	cnn.load_network("net_save/net_s%04d.dat"%(load_net), load_net)

cnn.train_network(nb_epoch=100, learning_rate=0.01, end_learning_rate=0.005, control_interv=2, momentum=0.0, decay=0.004, save_each=10)
cnn.forward_network(repeat=1)

exit()



