import numpy as np
import CIANNA as cnn

cnn.init_network(np.array([28,28,1]),10,256,'C_CUDA')

data_train = np.loadtxt("input.dat", skiprows=1,max_rows=60000*28)
data_valid = np.loadtxt("input.dat", skiprows=(60000*28)+2,max_rows=10000*28)
data_test = np.loadtxt("input.dat", skiprows=(70000*28)+3,max_rows=10000*28)

print (np.shape(data_train))

target_train = np.loadtxt("input.dat", skiprows=(80000*28)+4,max_rows=60000)
target_valid = np.loadtxt("input.dat", skiprows=(80000*28 + 60000)+5,max_rows=10000)
target_test = np.loadtxt("input.dat", skiprows=(80000*28 + 70000)+6,max_rows=10000)

cnn.create_dataset("TRAIN", 60000, data_train, target_train)
cnn.create_dataset("VALID", 10000, data_valid, target_valid)
cnn.create_dataset("TEST", 10000, data_test, target_test)

cnn.conv_create(f_size=5, nb_filters=6, stride=1, padding=4, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.conv_create(f_size=5, nb_filters=16, stride=1, padding=0, activation="RELU")
cnn.pool_create(pool_size=2)
cnn.dense_create(nb_neurons=256, activation="RELU")
cnn.dense_create(nb_neurons=128, activation="RELU")
cnn.dense_create(10, activation="SOFTMAX")

cnn.train_network(20, 0.0002, 1, 0.0, 0.001, 1)

exit()

