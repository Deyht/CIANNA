
from aux_fct import *
import time

#Change the if values of select a subset of visualizations

init_data_gen(0)

if(1):
	print("Generating random augmented training examples")
	start = time.time()
	create_train_batch()
	print (time.time()-start)

	print("Create visualization of the augmented examples")
	visual_aug(8,4)

if(1):
	print("\nOrdered validation examples")
	create_val_batch(8,4)

if(0):
	#Require trained network to be forwarded before visualization (imagenet_pred.py)
	print("\nVisualize predictions")
	visual_pred(load_epoch=0, visual_w=8, visual_h=6)

free_data_gen()
