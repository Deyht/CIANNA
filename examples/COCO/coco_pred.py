
import numpy as np
from aux_fct import *

#Comment to access system wide install
import sys, glob, os
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


init_data_gen()
input_val, targets_val = create_val_batch()

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

if(1):#switch off after a first prediction to explore post-process parameters
	cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=1, b_size=32,
		comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", inference_only=1)

	cnn.create_dataset("TEST", nb_val, input_val, targets_val)

	cnn.set_yolo_params()

	if(load_epoch > 0):
		cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
	else:
		if(not os.path.isfile("net_train_coco_map50_39.9_fp16.dat")):
			os.system("wget https://share.obspm.fr/s/pqG4jFrkEWi3SWt/download/net_train_coco_map50_39.9_fp16.dat")
		cnn.load("net_train_coco_map50_39.9_fp16.dat", 0, bin=1)

	cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=0)

	cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

pred_postprocess(load_epoch=0, obj_threshold=0.03, class_soft_limit=0.25, nms_threshold_same=0.4, nms_threshold_diff=0.95)
os.system("python3 coco_eval.py")


"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.548
"""
#Note: mAP for an input resolution of 416x416
#The network is resiliant to slight augment in image resolution, which increase the mAP
#We recommand changing image_size by step of 64 (2 grid elements)







