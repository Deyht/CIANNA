
from threading import Thread
from data_gen import *

#Comment to access system wide install
sys.path.insert(0,glob.glob("../../src/build/lib.*/")[-1])
import CIANNA as cnn



def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

load_epoch = 0
if(len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

def data_augm():
	input_data, targets = create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images_iter, input_data[:,:], targets[:,:], silent=1)
	return
	
	
######################################################################
#####                   CIANNA INIT AND DATA                     #####
######################################################################

#Construct training set using custom selection function
dataset_perscut(data_path+"TrainingSet_B1_v2.txt",data_path+"TrainingSet_perscut.txt", 18) #18 is the number of header line removed

#lower b_size if low GPU memory
cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param),
	bias=0.1, b_size=16, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=30)

init_data_gen()

input_data, targets = create_train_batch()
input_valid, targets_valid = create_valid_batch()

cnn.create_dataset("TRAIN", nb_images_iter, input_data[:,:], targets[:,:])
cnn.create_dataset("VALID", nb_valid, input_valid[:,:], targets_valid[:,:])



######################################################################
#####                   YOLO PARAMETER TUNING                    #####
######################################################################

#Size priors for all possible boxes per grid. element
prior_w = f_ar([6.0,6.0,6.0,6.0,6.0,6.0,12.0, 9.0,24.0])
prior_h = f_ar([6.0,6.0,6.0,6.0,6.0,6.0, 9.0,12.0,24.0])
prior_size = np.vstack((prior_w, prior_h))

#No obj probability prior to rebalance the size distribution
prior_noobj_prob = f_ar([0.15,0.15,0.15,0.15,0.15,0.15,0.01,0.01,0.01])

#Relative scaling of each error "type" : 
error_scales = cnn.set_error_scales(position = 36.0, size = 0.2, probability = 0.5, objectness = 2.0, parameters = 5.0)

#Relative scaling of each extra paramater
param_ind_scales = f_ar([2.0,2.0,1.0,0.5,0.5])

#Various IoU limit conditions
IoU_limits = cnn.set_IoU_limits(
	good_IoU_lim 			=  0.5, 
	low_IoU_best_box_assoc 	= -0.1, 
	min_prob_IoU_lim 		= -0.3,
	min_obj_IoU_lim 		= -0.3, 
	min_param_IoU_lim 		= -0.1)

#Activate / deactivate some parts of the loss
fit_parts = cnn.set_fit_parts(position = 1, size = 1, probability = 1, objectness = 1, parameters = 1)

#Supplementary parameters for activation function of each part
slopes_and_maxes = cnn.set_slopes_and_maxes(
	position    = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
	size        = cnn.set_sm_single(slope = 0.5, fmax = 1.2, fmin = -1.2),
	probability = cnn.set_sm_single(slope = 0.2, fmax = 6.0, fmin = -6.0),
	objectness  = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
	parameters  = cnn.set_sm_single(slope = 0.5, fmax = 1.5, fmin = -0.2))
					
#Other parameters
IoU_type 					= "DIoU"
prior_dist_type 			= "OFFSET"
strict_box_size 			= 0
min_prior_forced_scaling 	= 0.0
rand_startup 				= nb_images_iter*10
rand_prob_best_box_assoc 	= 0.90
rand_prob 					= 0.02
error_type 					= "natural"
no_override 				= 1
raw_output 					= 0

nb_yolo_filters = cnn.set_yolo_params(
	nb_box = nb_box, nb_class = nb_class, nb_param = nb_param, max_nb_obj_per_image = max_nb_obj_per_image,
	prior_size = prior_size, prior_noobj_prob = prior_noobj_prob, IoU_type = IoU_type, prior_dist_type = prior_dist_type,
	error_scales = error_scales, param_ind_scales = param_ind_scales, slopes_and_maxes = slopes_and_maxes, IoU_limits = IoU_limits,
	fit_parts = fit_parts, strict_box_size = strict_box_size, min_prior_forced_scaling = min_prior_forced_scaling,
	rand_startup = rand_startup, rand_prob_best_box_assoc = rand_prob_best_box_assoc, rand_prob = rand_prob, 
	error_type = error_type, no_override = no_override, raw_output = raw_output)



######################################################################
#####                   NETWORK BACKBONE SETUP                   #####
######################################################################

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
else:

	cnn.conv(f_size=i_ar([5,5]), nb_filters=32  , stride=i_ar([1,1]), padding=i_ar([2,2]), activation="RELU")

	cnn.conv(f_size=i_ar([2,2]), nb_filters=16  , stride=i_ar([2,2]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=24  , stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=32  , stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")

	cnn.conv(f_size=i_ar([2,2]), nb_filters=64  , stride=i_ar([2,2]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([1,1]), nb_filters=128 , stride=i_ar([1,1]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=192 , stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")

	cnn.conv(f_size=i_ar([2,2]), nb_filters=128 , stride=i_ar([2,2]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([1,1]), nb_filters=192 , stride=i_ar([1,1]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=384 , stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")
	cnn.conv(f_size=i_ar([1,1]), nb_filters=256 , stride=i_ar([1,1]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=384 , stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")

	cnn.conv(f_size=i_ar([2,2]), nb_filters=512 , stride=i_ar([2,2]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([1,1]), nb_filters=768 , stride=i_ar([1,1]), padding=i_ar([0,0]), activation="RELU")
	cnn.conv(f_size=i_ar([3,3]), nb_filters=1024, stride=i_ar([1,1]), padding=i_ar([1,1]), activation="RELU")
	cnn.norm(group_size=4)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=2048, stride=i_ar([1,1]), padding=i_ar([0,0]), activation="RELU", drop_rate=0.25)
	cnn.conv(f_size=i_ar([1,1]), nb_filters=nb_yolo_filters, stride=i_ar([1,1]), padding=i_ar([0,0]), activation="YOLO")
	
cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=1)


######################################################################
#####                 TRAINING NETWORK PROCEDURE                 #####
######################################################################

learning_rate = 0.00015
warmup_delay = 40
end_lr_prop = 0.02
momentum = 0.8
lr_decay = 0.0005
weight_decay = 0.0
save_every = 100

for block in range(load_epoch,max_epoch):
	
	#Dynamic load data generation
	t = Thread(target=data_augm)
	t.start()
	
	if((block+1) <= warmup_delay):
		loc_lr = 0.98*learning_rate*((block+1)/warmup_delay)+0.02*learning_rate
	else:
		loc_lr = learning_rate
		
	cnn.train(nb_iter = 1, learning_rate = loc_lr, end_learning_rate = loc_lr*end_lr_prop, 
		shuffle_every = 0, momentum = momentum, lr_decay = lr_decay, weight_decay = weight_decay, 
		save_every = save_every, silent = 0, save_bin = 1)
	
	if(block == 0):
		cnn.perf_eval()

	t.join()
	cnn.swap_data_buffers("TRAIN")
		
exit()









