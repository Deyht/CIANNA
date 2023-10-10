
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os, glob
from threading import Thread
import time
import albumentations as A
import cv2

processed_data_path = "./"

class_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_class_list.txt", dtype="str")[:,1]

np.random.seed(int(time.time()))
	
nb_images_per_iter = 64*50
image_size_raw = 500
nb_class = 1000

nb_keep_val_raw = nb_class*50
nb_keep_val = nb_class*25 
#only half the validation set for computing the val/test loss during training to limit the memory footprint

nb_workers = 8
config_type = 1 #change in both dataset_gen and imagenet_train
#For detection pretraining: first trained at low resolution, then slight training in high res before using for detection

if(config_type == 0):
	image_size = 224
	image_size_val = 256
	max_scale = 480.0
	min_scale = 256.0
else:
	image_size = 448
	image_size_val = 480
	max_scale = 480.0
	min_scale = 256.0

block_size = int(nb_images_per_iter / nb_workers)

flat_image_slice = image_size*image_size

transform_val = A.Compose([
	A.SmallestMaxSize(max_size=image_size_val, interpolation=1, p=1.0),
	A.CenterCrop(width=image_size, height=image_size, p=1.0),
])


## Data augmentation
def init_data_gen(test_mode = 0):
	
	global nb_images_per_iter, nb_keep_val_raw, nb_keep_val, image_size_raw, image_size, image_size_val, max_scale, min_scale
	global flat_image_slice, class_count, nb_workers, block_size, transform_val
	global input_data, targets, input_val, targets_val, targets_zero, nb_process, nb_class
	
	class_count = np.loadtxt(processed_data_path+"bin_blocks/bin_images_%d/class_count.txt"%(image_size_raw))
	
	targets_zero = np.zeros((nb_class), dtype="float32")
	
	if(test_mode == 0): #used to control the behavior regarging training, validation and test 
		input_data = np.zeros((nb_images_per_iter,image_size*image_size*3), dtype="float32")
		targets = np.zeros((nb_images_per_iter,nb_class), dtype="float32")
		
	input_val = np.zeros((nb_keep_val,image_size*image_size*3), dtype="float32")
	targets_val = np.zeros((nb_keep_val,nb_class), dtype="float32")



def create_train_aug(i, rf_c, rf_id, rf_scale):

	for l in range(0,block_size):

		r_class = int(rf_c[l]*nb_class)
		r_id = int(rf_id[l]*class_count[r_class])
		
		patch = np.load(processed_data_path+"bin_blocks/bin_images_%d/C%04d/train_img_%04d.npy"%(image_size_raw,r_class+1,r_id), allow_pickle=False)
		
		l_scale = int(rf_scale[l]*(max_scale-min_scale)+min_scale)
		l_translate = int(np.maximum(0,(image_size+(image_size/14.0)-l_scale)*0.5))
		
		transform = A.Compose([
			#Affine here act more as an aspect ratio transform than scaling
			A.Affine(scale=(0.9,1.1), rotate=(-15,15), fit_output=True, interpolation=1, p=1.0),
			
			A.SmallestMaxSize(max_size=l_scale, interpolation=1, p=1.0),
			A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
			A.Affine(translate_px=(-l_translate,l_translate),p=1.0),
			
			A.RandomCrop(width=image_size, height=image_size, p=1.0),
			A.HorizontalFlip(p=0.5),
			
			A.ColorJitter(brightness=(0.66,1.5), contrast=(0.66,1.5), saturation=(0.66,1.5), hue=0.15, p=1.0),
			A.ToGray(p=0.02),
			
			A.OneOf([
	        	A.ISONoise(p=0.1),
	        	A.MultiplicativeNoise(per_channel=False, elementwise=True, p=2.0),
	        	A.GaussNoise(var_limit=(0.0,0.03*255), per_channel=False, p=0.5),
	        	A.PixelDropout(dropout_prob=0.03, per_channel=False, p=2.0),
	        	A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0),
				A.GaussianBlur(p=1.0),
			], p=0.0),
		#Various types of noise / image alterations. Tend to reduce the validation accuracy, but build a more resilient network for deployement or other applications
		])
		
		transformed = transform(image=patch)

		patch_aug = transformed['image']
		
		for depth in range(0,3):
			input_data[i[l],depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0
		
		targets[i[l],:] = np.copy(targets_zero[:])
		targets[i[l],r_class] = 1.0
	
	
def visual_aug(visual_w, visual_h):
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		p_c = np.argmax(targets[i,:])
		
		c_text = ax[c_x,c_y].text(15, 25, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=6, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
	
	plt.savefig("augm_mosaic.jpg", dpi=200)


def create_train_batch():
	
	nb_blocks = nb_images_per_iter / block_size

	i_d = np.arange(nb_images_per_iter)
	rf_c = np.random.random(nb_images_per_iter)
	rf_id = np.random.random(nb_images_per_iter)
	rf_scale = np.random.random(nb_images_per_iter)
	
	t_list = []
	b_count = 0
	
	for k in range(0,nb_workers):
		t = Thread(target=create_train_aug, args=[i_d[b_count*block_size:(b_count+1)*block_size], \
												rf_c[b_count*block_size:(b_count+1)*block_size], \
												rf_id[b_count*block_size:(b_count+1)*block_size], \
												rf_scale[b_count*block_size:(b_count+1)*block_size]])
		t.start()
		t_list = np.append(t_list, t)
		b_count += 1
	
	for k in range(0,nb_workers):
		t_list[k].join()	
	
	return input_data, targets

def create_val_batch(visual_w=0, visual_h=0):
	print("Loading validation data ...")
	
	visual_iter = 0

	val_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_1000classes_val.txt", dtype="str")

	for i in range(0, nb_keep_val):
		
		patch = np.load(processed_data_path+"bin_blocks/bin_images_%d/val/valid_img_%04d.npy"%(image_size_raw,i), allow_pickle=False)

		transformed = transform_val(image=patch)
		patch_aug = transformed['image']	
		
		for depth in range(0,3):
			input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0
		
		targets_val[i,:] = np.copy(targets_zero[:])
		targets_val[i,int(val_list[i,1])] = 1.0
		
		if(visual_w*visual_h > 0):
			if(visual_iter == 0):
				fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
			
			c_x = visual_iter // visual_w
			c_y = visual_iter % visual_w
			
			ax[c_x,c_y].imshow(patch_aug)
			ax[c_x,c_y].axis('off')
			
			p_c = int(val_list[i,1])
			
			c_text = ax[c_x,c_y].text(15, 25, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=6, clip_on=True)
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
			
			visual_iter += 1
			if(visual_iter >= visual_w*visual_h):
				plt.savefig("val_target_mosaic.jpg", dpi=200)
				return
	
	return input_val, targets_val
	
	
def visual_pred(load_epoch=0, visual_w=8, visual_h=6):
	
	top_error = 5
	visual_iter = 0

	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%(load_epoch), dtype="float32")
	predict = np.reshape(pred_raw, (nb_keep_val,nb_class))

	val_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_1000classes_val.txt", dtype="str")

	for i in range(0, nb_keep_val):
		
		patch = np.load(processed_data_path+"bin_blocks/bin_images_%d/val/valid_img_%04d.npy"%(image_size_raw,i), allow_pickle=False)

		transformed = transform_val(image=patch)
		patch_aug = transformed['image']
		
		
		ind_best = np.argpartition(predict[i], -top_error)[-top_error:]
		ind_sort = ind_best[np.argsort(predict[i, ind_best])][::-1]
		pred_values = predict[i, ind_sort]
		
		if(visual_w*visual_h > 0):
			if(visual_iter == 0):
				fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
			
			c_x = visual_iter // visual_w
			c_y = visual_iter % visual_w
			
			ax[c_x,c_y].imshow(patch_aug)
			ax[c_x,c_y].axis('off')
			
			p_c = int(val_list[i,1])
			
			c_text = ax[c_x,c_y].text(15, image_size - 20, "Targ: %s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=6, clip_on=True)
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
			
			for k in range(0, top_error):
				c_text = ax[c_x,c_y].text(10, 20+k*25, "%0.2f - %s"%(pred_values[k], class_list[ind_sort[k]]), 
					c=plt.cm.tab20(ind_sort[k]%20), fontsize=6, clip_on=True)
				c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
			
			
			visual_iter += 1
			if(visual_iter >= visual_w*visual_h):
				plt.savefig("pred_mosaic.jpg", dpi=200)
				return

def free_data_gen():
  global input_data, targets, input_val, targets_val
  del (input_data, targets, input_val, targets_val)
  return



