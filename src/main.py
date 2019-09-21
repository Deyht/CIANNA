import numpy as np
import CIANNA as cnn

cnn.init_network(np.array([24,24,1]),10)

data_train = np.loadtxt("input.dat", skiprows=1,max_rows=60000)
data_valid = np.loadtxt("input.dat", skiprows=60000+1,max_rows=10000)
data_test = np.loadtxt("input.dat", skiprows=70000+1,max_rows=10000)

target_train = np.loadtxt("input.dat", skiprows=80000+1,max_rows=60000)
target_valid = np.loadtxt("input.dat", skiprows=140000+1,max_rows=10000)
target_test = np.loadtxt("input.dat", skiprows=150000+1,max_rows=10000)

train = cnn.create_dataset(60000, data_train, target_train)
test = cnn.create_dataset(10000, data_valid, target_valid)
valid = cnn.create_dataset(10000, data_test, target_test)


cnn.conv_create(f_size= 5, nb_filters = 6, stride = 1, padding = 4, activation="RELU")
cnn.pool_create()
cnn.conv_create(f_size= 5, nb_filters = 16, stride = 1, padding = 0, activation="RELU")
cnn.pool_create()
cnn.dense_create(nb_neurons = 256, activation="RELU")
cnn.dense_create(nb_neurons = 128, activation="RELU")
cnn.dense_create(20, activation="SOFTMAX")


