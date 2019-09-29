import numpy as np
import sys
#sys.path.append('/Home/Users/dcornu/WORK/GALMAP/CIANNA/src/build/lib.linux-x86_64-3.6')
import CIANNA as cnn
import matplotlib.pyplot as plt
import random

cnn.init_network(np.array([64,64,2]),100,8,'C_CUDA')

cmd_width = 64

data = np.loadtxt("../raw_data/CMDs_cut.txt",skiprows=2)

target = np.loadtxt("../raw_data/Profiles_cut.txt", skiprows=1)
target = target[:]
index = np.where(np.argmax(target,axis=1) > 100)

data[index,:64*64] = data[index,64*64:]
target = target[:,0:100]

index = np.array(range(0,20000))

random.shuffle(index)
print (np.shape(index))

data = data[index[:],:]
target = target[index[:],:]

data_train = data[:16000]
data_valid = data[16000:18000]
data_test = data[18000:]

print (np.max(data_train), np.max(data_valid), np.max(data_test))

data_train /= 633.0
data_valid /= 633.0
data_test /= 633.0

print (np.shape(data_train))
print (np.shape(data_valid))
print (np.shape(data_test))


target_train = target[:16000]
target_valid = target[16000:18000]
target_test = target[18000:]

target_train /= 3.0
target_valid /= 3.0
target_test /= 3.0

print (np.shape(target_train))
print (np.shape(target_valid))
print (np.shape(target_test))

np.savetxt("../raw_data/test_set_CMDS.txt", data_test)
np.savetxt("../raw_data/test_set_Profiles.txt", target_test)

#plt.hist(np.argmax(target_train[:,:], axis = 1),bins=150)

cnn.create_dataset("TRAIN", 16000, data_train, target_train, 0.1, 1)
cnn.create_dataset("VALID", 2000, data_valid, target_valid, 0.1, 1)
cnn.create_dataset("TEST", 2000, data_test, target_test, 0.1, 1)

cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=3, activation="RELU")
cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=0, activation="RELU")
cnn.conv_create(f_size=3, nb_filters=16, stride=1, padding=0, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
cnn.conv_create(f_size=3, nb_filters=32, stride=1, padding=0, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.dense_create(nb_neurons=512, activation="RELU",drop_rate=0.1)
cnn.dense_create(nb_neurons=512, activation="RELU",drop_rate=0.1)
cnn.dense_create(nb_neurons=100, activation="LINEAR")

cnn.train_network(nb_epoch=1, learning_rate=0.05, control_interv=1, momentum=0.0, decay=0.007)
#cnn.forward_network(repeat=100)

exit()



