
import numpy as np
from threading import Thread
from aux_fct import *
import gc

#Comment to access system wide install
import sys, glob
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

#Change image test mode in aux_fct to change network resolution in all functions

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class, bias=0.1,
	 b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", adv_size=35)

init_data_gen(test_mode=1)

#Compute on only half the validation set to reduce memory footprint
input_test, targets_test = create_val_batch()

cnn.create_dataset("TEST", nb_keep_val, input_test[:,:], targets_test[:,:])

del (input_test, targets_test)
gc.collect()

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch, load_epoch, bin=1)
else:
	#Not trained as a resolution agnostic network
	if(image_size == 224):
		if(not os.path.isfile("ImageNET_aux_data/net_pretrain_medium_224_acc70.dat")):
			os.system("wget -P ImageNET_aux_data/ https://share.obspm.fr/s/dj69Fm5Gyaenzjw/download/net_pretrain_medium_224_acc70.dat")
		cnn.load("ImageNET_aux_data/net_pretrain_medium_224_acc70.dat", 0, bin=1)
	elif(image_size == 448):
		if(not os.path.isfile("ImageNET_aux_data/net_pretrain_medium_448_acc74.dat")):
			os.system("wget -P ImageNET_aux_data/ https://share.obspm.fr/s/PJaJ6an7amZiQBC/download/net_pretrain_medium_448_acc74.dat")
		cnn.load("ImageNET_aux_data/net_pretrain_medium_448_acc74.dat", 0, bin=1)
	else:
		print("No trained network for the define image resolution")
		exit()

cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")



val_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_1000classes_val.txt", dtype="str")

for top_error in [1,5]:
		
	count = 0
	count_max = 0

	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%(load_epoch), dtype="float32")
	predict = np.reshape(pred_raw, (nb_keep_val,nb_class))

	for i in range(0, nb_keep_val):
		ind = np.argpartition(predict[i], -top_error)[-top_error:]
		count_max += np.max(predict[i])
		
		if(np.isin(int(val_list[i,1]), ind[:])):
			count += 1

	print ("Top-%d error"%(top_error))
	print (1.0 - count/(nb_keep_val))
	print ("Avg max class value")
	print (count_max/(nb_keep_val))

	

	

















