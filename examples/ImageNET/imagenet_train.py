	
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

def data_augm():
	input_data, targets = create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images_per_iter, input_data[:,:], targets[:,:], silent=1)
	return

nb_iter_per_augm = 1
total_iter = 50000

load_iter = 0
if (len(sys.argv) > 1):
	load_iter = int(sys.argv[1])

start_iter = int(load_iter / nb_iter_per_augm)

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class, bias=0.1,
	 b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", adv_size=30)

init_data_gen()

input_val, targets_val = create_val_batch()
cnn.create_dataset("VALID", nb_keep_val, input_val[:,:], targets_val[:,:])
del (input_val, targets_val)
gc.collect()

input_data, targets = create_train_batch()
cnn.create_dataset("TRAIN", nb_images_per_iter, input_data[:,:], targets[:,:])

if(load_iter > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_iter, load_iter, bin=1)
else:
	cnn.conv(f_size=i_ar([3,3]), nb_filters=32  , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=4)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=64  , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=8)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=128 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=8)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=64  , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=8)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=128 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=8)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=256 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=128 , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=256 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=512 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=256 , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=512 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=256 , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=512 , padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=32)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=512 , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=32)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=512 , padding=i_ar([0,0]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=32)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=nb_class , padding=i_ar([0,0]), activation="LIN")
	cnn.pool(p_size=i_ar([1,1]), p_type="AVG", p_global=1, activation="SMAX")

cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=0)

for run_iter in range(start_iter,int(total_iter/nb_iter_per_augm)):
	
	t = Thread(target=data_augm)
	t.start()
	
	cnn.train(nb_iter=nb_iter_per_augm, learning_rate=0.003, end_learning_rate=0.0000002, shuffle_every=0,\
			 control_interv=20, confmat=3, momentum=0.9, lr_decay=0.0006, weight_decay=0.0002, save_every=200,\
			 silent=0, save_bin=1, TC_scale_factor=256.0)
	
	if(run_iter == start_iter):
		cnn.perf_eval()

	t.join()
	cnn.swap_data_buffers("TRAIN")

	

















