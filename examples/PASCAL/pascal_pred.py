
import numpy as np
from aux_fct import *

#Comment to access system wide install
import sys, glob
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


init_data_gen()
input_val, targets_val = create_val_batch()

if(1):#switch off after a first prediction to explore post-process parameters
	load_epoch = 0
	if (len(sys.argv) > 1):
		load_epoch = int(sys.argv[1])

	cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=1, b_size=32,
		comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", inference_only=1)

	cnn.create_dataset("TEST", nb_keep_val, input_val, targets_val)

	cnn.set_yolo_params()

	if(load_epoch > 0):
		cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
	else:
		if(not os.path.isfile("net_train_pascal_416_fp16_75.3map.dat")):
			os.system("wget https://share.obspm.fr/s/XxY3gXnpXgsxA24/download/net_train_pascal_416_fp16_75.3map.dat")
		cnn.load("net_train_pascal_416_fp16_75.3map.dat", 0, bin=1)

	cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=0)

	cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

pred_postprocess(load_epoch=0, obj_threshold=0.03, class_soft_limit=0.3, nms_threshold_same=0.4, nms_threshold_diff=0.95)
pred_compute_map()


"""
AP_all (0.50): 76.095814%
**** Per class AP ****
AP aeroplane      : 77.77%   Total:  285 - T:  239 - F: 2474
AP bicycle        : 86.65%   Total:  337 - T:  314 - F: 1347
AP bird           : 71.22%   Total:  459 - T:  373 - F: 3318
AP boat           : 68.39%   Total:  263 - T:  222 - F: 3899
AP bottle         : 49.42%   Total:  469 - T:  335 - F: 7119
AP bus            : 83.14%   Total:  213 - T:  190 - F: 1547
AP car            : 81.77%   Total: 1201 - T: 1051 - F: 9559
AP cat            : 86.86%   Total:  358 - T:  324 - F: 1443
AP chair          : 57.53%   Total:  756 - T:  633 - F: 12270
AP cow            : 74.96%   Total:  244 - T:  211 - F: 1077
AP diningtable    : 74.50%   Total:  206 - T:  190 - F: 1813
AP dog            : 82.15%   Total:  489 - T:  437 - F: 1856
AP horse          : 86.26%   Total:  348 - T:  313 - F: 1011
AP motorbike      : 83.77%   Total:  325 - T:  288 - F: 1149
AP person         : 78.82%   Total: 4528 - T: 3945 - F: 24082
AP pottedplant    : 48.52%   Total:  480 - T:  346 - F: 10257
AP sheep          : 74.96%   Total:  242 - T:  206 - F: 1279
AP sofa           : 78.36%   Total:  239 - T:  224 - F: 1487
AP train          : 87.93%   Total:  282 - T:  271 - F: 2110
AP tvmonitor      : 72.71%   Total:  308 - T:  258 - F: 3483

**** mAP (0.50): 75.285394% ****
"""
#Note: mAP@50 for an input resolution of 416. 
#The network is resiliant to slight augment in image resolution, which increase the mAP
#We recommand changing image_size by step of 64 (2 grid elements)







