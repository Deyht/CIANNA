
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
AP_all (0.50): 75.989870%
**** Per class AP ****
AP aeroplane      : 77.67%   Total:  285 - T:  245 - F: 2755
AP bicycle        : 86.26%   Total:  337 - T:  310 - F: 1461
AP bird           : 71.34%   Total:  459 - T:  376 - F: 3981
AP boat           : 67.09%   Total:  263 - T:  216 - F: 4808
AP bottle         : 48.45%   Total:  469 - T:  328 - F: 8662
AP bus            : 82.77%   Total:  213 - T:  189 - F: 1584
AP car            : 81.60%   Total: 1201 - T: 1045 - F: 11001
AP cat            : 86.88%   Total:  358 - T:  327 - F: 1594
AP chair          : 57.04%   Total:  756 - T:  632 - F: 15643
AP cow            : 74.49%   Total:  244 - T:  208 - F: 1020
AP diningtable    : 73.61%   Total:  206 - T:  187 - F: 2283
AP dog            : 82.88%   Total:  489 - T:  441 - F: 1949
AP horse          : 86.27%   Total:  348 - T:  315 - F: 1040
AP motorbike      : 84.03%   Total:  325 - T:  288 - F: 1234
AP person         : 78.81%   Total: 4528 - T: 3949 - F: 26780
AP pottedplant    : 47.55%   Total:  480 - T:  351 - F: 13960
AP sheep          : 74.69%   Total:  242 - T:  204 - F: 1401
AP sofa           : 77.55%   Total:  239 - T:  220 - F: 1583
AP train          : 87.47%   Total:  282 - T:  268 - F: 2302
AP tvmonitor      : 73.26%   Total:  308 - T:  263 - F: 3863

**** mAP (0.50): 74.984838% ****
"""
#Note: mAP@50 for an input resolution of 416. 
#The network is resiliant to slight augment in image resolution, which increase the mAP
#We recommand changing image_size by step of 64 (2 grid elements)







