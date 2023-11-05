
import numpy as np
from threading import Thread
from aux_fct import *
import gc

#Comment to access system wide install
import sys, glob
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


def data_augm():
	input_data, targets = create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images_per_iter, input_data[:,:], targets[:,:], silent=1)
	return

init_data_gen()

#Start / loading epoch can be specified on the command line or here
load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=1+max_nb_obj_per_image*(7+0+1),
	bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", adv_size=30)

input_data, targets = create_train_batch()
input_val, targets_val = create_val_batch()

cnn.create_dataset("TRAIN", nb_images_per_iter, input_data, targets)
cnn.create_dataset("VALID", nb_keep_val, input_val, targets_val)
cnn.create_dataset("TEST" , nb_keep_val, input_val, targets_val)


##### YOLO parameters tuning #####

nb_epoch_per_augm = 1

#Size priors for all possible boxes per grid. element 
prior_w = f_ar([32.0, 92.0,150.0,208.0,333.0])
prior_h = f_ar([32.0,150.0, 92.0,333.0,208.0])
prior_size = np.vstack((prior_w, prior_h))

prior_noobj_prob = f_ar([0.05,0.1,0.1,0.1,0.1])

#Relative scaling of each error "type" :
error_scales = cnn.set_error_scales(position = 12.0, size = 6.0, probability = 0.5, objectness = 6.0, classes = 0.4)

fit_parts = cnn.set_fit_parts(position = 1, size = 1, probability = 1, objectness = 1, classes = 1)

#Various IoU limit conditions
IoU_limits = cnn.set_IoU_limits(good_IoU_lim = 0.5, low_IoU_best_box_assoc = -0.1, min_prob_IoU_lim = -1.0, min_obj_IoU_lim = -1.0,
		min_class_IoU_lim = -0.1, min_param_IoU_lim = -1.0, diff_IoU_lim = 0.5, diff_obj_lim = 0.3)

slopes_and_maxes = cnn.set_slopes_and_maxes(
						position    = cnn.set_sm_single(slope=1.0, fmax=6.0, fmin=-6.0),
						size        = cnn.set_sm_single(slope=0.5, fmax=1.6, fmin=-1.6),
						probability = cnn.set_sm_single(slope=1.0, fmax=6.0, fmin=-6.0),
						objectness  = cnn.set_sm_single(slope=1.0, fmax=6.0, fmin=-6.0),
						classes     = cnn.set_sm_single(slope=1.0, fmax=6.0, fmin=-6.0))

strict_box_size = 3

# REGULAR TRAINING 
learning_rate = 0.0005
weight_decay = 0.0005
total_epochs = 740

nb_yolo_filters = cnn.set_yolo_params(nb_box = nb_box, nb_class = nb_class, nb_param = 0, max_nb_obj_per_image = max_nb_obj_per_image,
				prior_size = prior_size, prior_noobj_prob = prior_noobj_prob, IoU_type = "DIoU", prior_dist_type = "OFFSET", 
				error_scales = error_scales, slopes_and_maxes = slopes_and_maxes, IoU_limits = IoU_limits,
				fit_parts = fit_parts, strict_box_size = strict_box_size, min_prior_forced_scaling = 1.2, diff_flag = 1,
				rand_startup = nb_images_per_iter*20, rand_prob_best_box_assoc = 0.05, rand_prob = 0.0, class_softmax = 1, error_type = "natural", no_override = 1)

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
else:
	if(not os.path.isfile("net_pretrain_medium_448_acc74.dat")):
		os.system("wget https://share.obspm.fr/s/PJaJ6an7amZiQBC/download/net_pretrain_medium_448_acc74.dat")
	#Load pretrain ImageNET network but drop last layers to be replaced by 3 conv layers + yolo output layer
	cnn.load("net_pretrain_medium_448_acc74.dat",0, nb_layers=40, bin=1)
	
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=16)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=nb_yolo_filters, padding=i_ar([0,0]), activation="YOLO")


cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=0)

start_block = int(load_epoch / nb_epoch_per_augm)

for block in range(start_block,int(total_epochs/nb_epoch_per_augm)):
	
	t = Thread(target=data_augm)
	t.start()
	
	#Epochs limit can be adjusted, especially if changes are made to some hyper-parameters
	if((block+1)*nb_epoch_per_augm <= 20):
		loc_lr = learning_rate*0.01
	elif((block+1)*nb_epoch_per_augm <= 300): 
		loc_lr = learning_rate
	elif((block+1)*nb_epoch_per_augm <= 420):
		loc_lr = learning_rate*0.2
	else:
		loc_lr = learning_rate*0.05
	
	cnn.train(nb_iter=nb_epoch_per_augm, learning_rate=loc_lr, shuffle_every=0, control_interv=1, \
			momentum=0.9, lr_decay=0.0, weight_decay=weight_decay, save_every=20, silent=0, save_bin=1, TC_scale_factor=64.0)
	if(block == 0):
		cnn.perf_eval()
	t.join()
	
	cnn.swap_data_buffers("TRAIN")

	if((block+1)*nb_epoch_per_augm >= 50 and np.mod((block+1)*nb_epoch_per_augm,1) == 20):
		cnn.forward(repeat=1,no_error=1, saving=2, drop_mode="AVG_MODEL")
		pred_postprocess((block+1)*nb_epoch_per_augm, obj_threshold=0.03, class_soft_limit=0.3, nms_threshold_same=0.4, nms_threshold_diff=0.95)
		pred_compute_map()
		
	#Best epoch is not necessarly the last one due to possible overtraining










