
from aux_fct import *
import time

#Change the if values of select a subset of visualizations

init_data_gen()

if(1):
	print("Generating random augmented training examples")
	start = time.time()
	create_train_batch()
	print (time.time()-start)

	print("Create visualization of the augmented examples")
	visual_aug(visual_w=8, visual_h=4)

if(1):
	print("\nOrdered validation examples")
	create_val_batch()
	
	print("Create visualization of the validation dataset")
	visual_val(visual_w=8, visual_h=4)

if(0):
	#Require trained network to be forwarded before visualization (imagenet_pred.py)
	print("\nVisualize predictions")
	display_target = 1
	if(display_target):
		create_val_batch()
	pred_postprocess(load_epoch=0, obj_threshold=0.3, class_soft_limit=0.3, nms_threshold_same=0.4, nms_threshold_diff=0.95)
	visual_pred(visual_w=8, visual_h=4, display_target=display_target)

free_data_gen()
