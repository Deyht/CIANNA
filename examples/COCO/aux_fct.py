
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import json
from threading import Thread
from tqdm import tqdm
import os, re, glob

from numba import jit

import albumentations as A
import cv2


data_path = "./"

#Data are download in data_prep.py

with open("classnames.txt") as f:
	class_list = [line.rstrip('\n') for line in f]

class_id_conv = np.arange(1,92)
class_id_conv = np.delete(class_id_conv, [11,25,28,29,44,65,67,68,70,82,90])

class_list_short = class_list
color_offset = 0

train_list_2017 = {}
with open(data_path+"annotations/instances_train2017.json", "r") as f:
	train2017_instances = json.load(f)
for item in train2017_instances["images"]:
	train_list_2017[item["id"]] = item["file_name"]
	
train_im_path_2017 = list(train_list_2017.values())
train_im_id_2017 = list(train_list_2017.keys())

val_list_2017 = {}
with open(data_path+"annotations/instances_val2017.json", "r") as f:
	val2017_instances = json.load(f)
for item in val2017_instances["images"]:
	val_list_2017[item["id"]] = item["file_name"]
	
val_im_path_2017 = list(val_list_2017.values())
val_im_id_2017 = list(val_list_2017.keys())


image_size = 416
flat_image_slice = image_size*image_size
nb_class = 80
max_nb_obj_per_image = 70
nb_box = 5
yolo_reg_size = 32
yolo_nb_reg = int(image_size/yolo_reg_size)


nb_train = 117947
nb_val = 5000
nb_images_per_iter = 4000

val_size = np.zeros((nb_val,2), dtype="int")


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")


def init_data_gen():
	global nb_train, nb_val, nb_workers, block_size, flat_image_slice, val_size, transform_train, transform_val
	global nb_class, max_nb_obj_per_image, image_size_raw, image_size, flat_image_slice, nb_box, yolo_reg_size, yolo_nb_reg
	global input_data, targets, input_val, targets_val

	nb_workers = 6
	block_size = int(nb_images_per_iter / nb_workers)
	
	transform_train = A.Compose([
		#Affine here act more as an aspect ratio transform than scaling
		A.Affine(scale=(0.85,1.15), rotate=(-7,7), fit_output=True, interpolation=1, p=1.0),
		A.LongestMaxSize(max_size=480, interpolation=1, p=1.0),
		A.PadIfNeeded(min_width=480, min_height=480, border_mode=cv2.BORDER_CONSTANT, p=1.0),
		A.OneOf([
			A.RandomSizedCrop(min_max_height=[224,448], height=image_size, width=image_size, interpolation=1, p=6.0),
			A.Affine(scale=(0.7,1.0), translate_percent=(-0.3,0.3), keep_ratio=True, interpolation=1, p=3.0),
			A.Affine(scale=(0.4,0.7), translate_percent=(-0.6,0.6), keep_ratio=True, interpolation=1, p=3.0),
		], p=1.0),
		A.Resize(width=image_size, height=image_size, interpolation=1, p=1.0),

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
		], p=0.05),
		#Various types of noise / image alterations. Tend to reduce the mAP, but build a more resilient network for deployement or other applications
	], bbox_params=A.BboxParams(format='coco', min_visibility=0.3, min_area=(12*12)))
	
	transform_val = A.Compose([
		A.LongestMaxSize(max_size=image_size, interpolation=1, p=1.0),
		A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
	], bbox_params=A.BboxParams(format='coco'))
	
	#Processed images are just a reshap and padding so the largest side is equal to image_size_raw
	input_data = np.zeros((nb_images_per_iter,image_size*image_size*3), dtype="float32")
	targets = np.zeros((nb_images_per_iter,1+max_nb_obj_per_image*(7+1)), dtype="float32")

	input_val = np.zeros((nb_val,image_size*image_size*3), dtype="float32")
	targets_val = np.zeros((nb_val,1+max_nb_obj_per_image*(7+1)), dtype="float32")


def create_train_aug(i, r_id):

	for l in range(0,block_size):
		im_id = int(r_id[l] * nb_train)
		
		no_box = 0
		patch = np.load(data_path+"train2017/%s.npy"%(train_im_path_2017[im_id][:-4]), allow_pickle=False)
		if(os.path.exists(data_path+"train2017/bbox_%s.txt"%(train_im_id_2017[im_id]))):
			bbox_list = np.loadtxt(data_path+"train2017/bbox_%s.txt"%(train_im_id_2017[im_id]))
		else:
			no_box = 1
		
		if(no_box == 0):
			if(bbox_list.ndim == 1):
				bbox_list = np.reshape(bbox_list, (1,5))
			
			transformed = transform_train(image=patch, bboxes=bbox_list)

			patch_aug = transformed['image']
			bbs_aug = np.asarray(transformed['bboxes'])
		else:
			transformed = transform_train(image=patch, bboxes=[])

			patch_aug = transformed['image']
			bbs_aug = np.array([])
		
		
		for depth in range(0,3):
			input_data[i[l],depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0
		
		targets[i[l],:] = 0.0
		targets[i[l],0] = np.shape(bbs_aug)[0]
		if(targets[i[l],0] > max_nb_obj_per_image):
			print ("Max_obj_per_image limit reached: ", int(targets[i[l],0]))
			targets[i[l],0] = max_nb_obj_per_image
		for k in range(0, int(targets[i[l],0])):
			
			xmin = bbs_aug[k,0]
			ymin = bbs_aug[k,1]
			xmax = bbs_aug[k,0] + bbs_aug[k,2]
			ymax = bbs_aug[k,1] + bbs_aug[k,3]
			
			x_mean = (xmin+xmax)/2.0
			y_mean = (ymin+ymax)/2.0
			
			c_size = 32.0
			
			diff = 0
			if((x_mean < c_size or x_mean > image_size - c_size or y_mean < c_size or y_mean > image_size - c_size)
				or (abs(xmax-xmin)*abs(ymax-ymin)) < 1024.0):
				diff = 1
			
			width = max(16.0, xmax - xmin)
			height = max(16.0, ymax - ymin)
				
			xmin = max(0.0, x_mean - width/2.0)
			ymin = max(0.0, y_mean - height/2.0)
			xmax = min(image_size, x_mean + width/2.0)
			ymax = min(image_size, y_mean + height/2.0)
				
			targets[i[l],1+k*8:1+(k+1)*8] = np.array([np.where(class_id_conv[:] == bbs_aug[k,4])[0][0] + 1, xmin, ymin,0.0, xmax, ymax,1.0, diff])


def create_train_batch():

	nb_blocks = nb_images_per_iter / block_size
	
	i_d = np.arange(nb_images_per_iter)
	r_id = np.random.random(nb_images_per_iter)
	
	t_list = []
	b_count = 0
	
	for k in range(0,nb_workers):
		t = Thread(target=create_train_aug, args=[i_d[b_count*block_size:(b_count+1)*block_size], \
												r_id[b_count*block_size:(b_count+1)*block_size]])
		t.start()
		t_list = np.append(t_list, t)
		b_count += 1
	
	for k in range(0,nb_workers):
		t_list[k].join()	
	
	return input_data, targets


def visual_aug(visual_w, visual_h):
	
	l_style = ["-","--"]
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=140, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		targ_boxes = targets[i]
		if(targ_boxes[0] == -1):
			targ_boxes[0] = 1
		for k in range(0, int(targ_boxes[0])):
			xmin = targ_boxes[1+k*8+1]
			ymin = targ_boxes[1+k*8+2]
			xmax = targ_boxes[1+k*8+4]
			ymax = targ_boxes[1+k*8+5]
			p_c = int(targ_boxes[1+k*8+0]) - 1
			diff = int(targ_boxes[1+k*8+7])
		
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.6, ls=l_style[diff], fill=False, 
				color=plt.cm.tab20((p_c+color_offset)%20), zorder=3)
			c_patch = ax[c_x,c_y].add_patch(el)
			c_text = ax[c_x,c_y].text(xmin+4, ymin+10, "%s %d"%(class_list_short[p_c], (xmax-xmin)*(ymax-ymin)),
				c=plt.cm.tab20((p_c+color_offset)%20), fontsize=2, clip_on=True)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='black'), path_effects.Normal()])
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='black'), path_effects.Normal()])
	
	plt.savefig("augm_mosaic.jpg", dpi=400)


def create_val_batch():

	for i in range(0, nb_val):

		no_box = 0
		patch = np.load(data_path+"val2017/%s.npy"%(val_im_path_2017[i][:-4]), allow_pickle=False)
		val_size[i,:] = (np.shape(patch)[:2])
		
		if(os.path.exists(data_path+"val2017/bbox_%s.txt"%(val_im_id_2017[i]))):
			bbox_list = np.loadtxt(data_path+"val2017/bbox_%s.txt"%(val_im_id_2017[i]))
		else:
			no_box = 1
		
		if(no_box == 0):
			if(bbox_list.ndim == 1):
				bbox_list = np.reshape(bbox_list, (1,5))
			
			transformed = transform_val(image=patch, bboxes=bbox_list)

			patch_aug = transformed['image']
			bbs_aug = np.asarray(transformed['bboxes'])
		else:
			transformed = transform_val(image=patch, bboxes=[])

			patch_aug = transformed['image']
			bbs_aug = np.array([])
		
		for depth in range(0,3):
			input_val[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = (patch_aug[:,:,depth].flatten("C")-100.0)/155.0
		
		targets_val[i,:] = 0.0
		targets_val[i,0] = np.shape(bbs_aug)[0]
		if(targets_val[i,0] > max_nb_obj_per_image):
			print ("Max_obj_per_image limit reached: ", int(targets_val[i,0]))
			targets_val[i,0] = max_nb_obj_per_image
		for k in range(0, int(targets_val[i,0])):
			
			xmin = bbs_aug[k,0]
			ymin = bbs_aug[k,1]
			xmax = bbs_aug[k,0] + bbs_aug[k,2]
			ymax = bbs_aug[k,1] + bbs_aug[k,3]
			
			targets_val[i,1+k*8:1+(k+1)*8] = np.array([np.where(class_id_conv[:] == bbs_aug[k,4])[0][0] + 1,xmin,ymin,0.0,xmax,ymax,1.0,0])
	
	return input_val, targets_val


def visual_val(visual_w, visual_h):
	
	l_style = ["-","--"]
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=140, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		targ_boxes = targets_val[i]
		if(targ_boxes[0] == -1):
			targ_boxes[0] = 1
		for k in range(0, int(targ_boxes[0])):
			xmin = targ_boxes[1+k*8+1]
			ymin = targ_boxes[1+k*8+2]
			xmax = targ_boxes[1+k*8+4]
			ymax = targ_boxes[1+k*8+5]
			p_c = int(targ_boxes[1+k*8+0]) - 1
			diff = int(targ_boxes[1+k*8+7])
		
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.6, ls=l_style[diff], fill=False, 
				color=plt.cm.tab20((p_c+color_offset)%20), zorder=3)
			c_patch = ax[c_x,c_y].add_patch(el)
			c_text = ax[c_x,c_y].text(xmin+4, ymin+10, "%s %d"%(class_list_short[p_c], (xmax-xmin)*(ymax-ymin)),
				c=plt.cm.tab20((p_c+color_offset)%20), fontsize=2, clip_on=True)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='black'), path_effects.Normal()])
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='black'), path_effects.Normal()])
	
	plt.savefig("val_mosaic.jpg", dpi=400)


def free_data_gen():
  global input_data, targets, input_val, targets_val
  del (input_data, targets, input_val, targets_val)
  return


@jit(nopython=True, cache=True, fastmath=False)
def fct_IoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1)
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1)
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0] + 1)*abs(box1[3] - box1[1] + 1) + \
		abs(box2[2]-box2[0] + 1)*abs(box2[3] - box2[1] + 1) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

  # DIoU
	return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)
  # GIoU
	#return float(inter_2d)/float(uni_2d) - float(enclose_2d - uni_2d)/float(enclose_2d)


@jit(nopython=True, cache=True, fastmath=False)
def box_extraction(c_pred, c_box, c_tile, obj_threshold, class_soft_limit):
	c_nb_box = 0
	for i in range(0,yolo_nb_reg):
		for j in range(0,yolo_nb_reg):
			for k in range(0,nb_box):
				offset = int(k*(8+nb_class)) #no +1 for box prior in prediction
				c_box[4] = c_pred[offset+6,i,j]
				c_box[5] = c_pred[offset+7,i,j]
				p_c = np.max(c_pred[offset+8:offset+8+nb_class,i,j])
				cl = np.argmax(c_pred[offset+8:offset+8+nb_class,i,j])

				if(c_box[5] >= obj_threshold and c_box[5]*p_c**1 >= 0.01 and p_c > class_soft_limit):
					c_box[0] = c_pred[offset,i,j]
					c_box[1] = c_pred[offset+1,i,j]
					c_box[2] = c_pred[offset+3,i,j]
					c_box[3] = c_pred[offset+4,i,j]

					c_box[6] = k
					c_box[7:] = c_pred[offset+8:offset+8+nb_class,i,j]
					c_tile[c_nb_box,:] = c_box[:]
					c_nb_box +=1

	return c_nb_box


@jit(nopython=True, cache=True, fastmath=False)
def apply_NMS(c_tile, c_tile_kept, c_box, c_nb_box, amax_array, nms_threshold_same, nms_threshold_diff):
	c_nb_box_final = 0
	c_box_size_prev = c_nb_box

	while(c_nb_box > 0):
		max_objct = np.argmax(c_tile[:c_box_size_prev,5]*amax_array[:c_box_size_prev])
		c_box = np.copy(c_tile[max_objct])
		c_tile[max_objct,5] = 0.0
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1
		c_nb_box -= 1
		i = 0

		for i in range(0,c_box_size_prev):
			if(c_tile[i,5] < 0.00000001):
				continue
			IoU = fct_IoU(c_box[:4], c_tile[i,:4])
			
			if((IoU > nms_threshold_same and np.argmax(c_box[7:]) == np.argmax(c_tile[i,7:]))
				or (IoU > nms_threshold_diff and np.argmax(c_box[7:]) != np.argmax(c_tile[i,7:]))):
				c_tile[i] = 0.0
				c_nb_box -= 1
	 
	return c_nb_box_final


def pred_postprocess(load_epoch, obj_threshold, class_soft_limit, nms_threshold_same, nms_threshold_diff):
	
	global final_boxes
	
	final_boxes = []
	box_list = []
	
	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
	c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
	c_box = np.zeros((6+1+nb_class),dtype="float32")
	
	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%load_epoch, dtype="float32")
	predict = np.reshape(pred_raw, (nb_val,nb_box*(8+nb_class),yolo_nb_reg,yolo_nb_reg))
	
	for l in tqdm(range(0, nb_val)):
		
		im_id = val_im_id_2017[l]
		
		dim_long = np.argmax(val_size[l,:])
		ratio = image_size/val_size[l,dim_long]
		
		other_dim = int(np.mod(dim_long+1,2))
		offset = np.zeros((2))
		offset[dim_long] = 0.0
		offset[other_dim] = max(0.0,image_size - val_size[l,other_dim]*ratio)/2.0
	
		c_tile[:,:] = 0.0
		c_tile_kept[:,:] = 0.0

		c_pred = predict[l,:,:,:]
		c_nb_box = box_extraction(c_pred, c_box, c_tile, obj_threshold, class_soft_limit)			

		c_nb_box_final = c_nb_box
		amax_array = np.amax(c_tile[:,7:], axis=1)
		c_nb_box_final = apply_NMS(c_tile, c_tile_kept, c_box, c_nb_box, amax_array, nms_threshold_same, nms_threshold_diff)
		
		final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))
		
		for k in range(0, c_nb_box_final):
		
			x_min  = float(round((c_tile_kept[k,0]-offset[1])/ratio,2))
			y_min  = float(round((c_tile_kept[k,1]-offset[0])/ratio,2))
			width  = float(round((c_tile_kept[k,2]-offset[1])/ratio - (c_tile_kept[k,0]-offset[1])/ratio,2))
			height = float(round((c_tile_kept[k,3]-offset[0])/ratio - (c_tile_kept[k,1]-offset[0])/ratio,2))
			cat_id = int(class_id_conv[np.argmax(c_tile_kept[k,7:])])
			score  = float(round(c_tile_kept[k,5],4))
		
			box_list.append({"image_id": int(im_id), "category_id": cat_id,"bbox": [x_min,y_min, width,height],"score": score})
	
	with open("fwd_res/pred_%04d.json"%(load_epoch), "w") as f:
		json.dump(list(box_list), f)
	
	
def visual_pred(visual_w, visual_h, display_target=0, id_start=0):
	#Visualize at network rinput esolution
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2.0*visual_w,2.0*visual_h), dpi=210, constrained_layout=True)
	
	l_patch = np.zeros((image_size, image_size, 3))
	
	for l in tqdm(range(id_start, id_start + visual_w*visual_h)):
		
		c_x = (l-id_start) // visual_w
		c_y = (l-id_start) % visual_w
		im_id = val_im_id_2017[l]
	
		patch = np.load(data_path+"val2017/%s.npy"%(val_im_path_2017[l][:-4]), allow_pickle=False)
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_val[l,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis("off")
	
		im_boxes = final_boxes[l]
	
		if(display_target):
			targ_boxes = targets_val[l]
			if(targ_boxes[0] == -1):
				targ_boxes[0] = 1
			for k in range(0, int(targ_boxes[0])):
				xmin = targ_boxes[1+k*8+1]
				ymin = targ_boxes[1+k*8+2]
				xmax = targ_boxes[1+k*8+4]
				ymax = targ_boxes[1+k*8+5]
				p_c = int(targ_boxes[1+k*8+0]) - 1
				diff = int(targ_boxes[1+k*8+7])
			
				el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.4, ls="--", fill=False, 
					color=plt.cm.tab20((p_c+color_offset)%20), zorder=3)
				c_patch = ax[c_x,c_y].add_patch(el)
				c_text = ax[c_x,c_y].text(xmin+4, ymin+10, "%s %d"%(class_list_short[p_c], (xmax-xmin)*(ymax-ymin)), 
					c=plt.cm.tab20((p_c+color_offset)%20), fontsize=2, clip_on=True)
				c_patch.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground="black"), path_effects.Normal()])
				c_text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground="black"), path_effects.Normal()])	
		
		for k in range(0, np.shape(im_boxes)[0]):
		
			xmin  = float(round((im_boxes[k,0]),2))
			ymin  = float(round((im_boxes[k,1]),2))
			width  = float(round((im_boxes[k,2]) - (im_boxes[k,0]),2))
			height = float(round((im_boxes[k,3]) - (im_boxes[k,1]),2))
			p_c = np.argmax(im_boxes[k,7:])
			score  = float(round(im_boxes[k,5],4))
		
			el = patches.Rectangle((xmin,ymin), width, height, linewidth=0.4, fill=False, color=plt.cm.tab20((p_c+color_offset)%20), zorder=3)
			c_patch = ax[c_x,c_y].add_patch(el)
			c_text = ax[c_x,c_y].text(xmin+5, ymin+height-4, "%s:%d-%0.2f-%0.2f"%(class_list[p_c],im_boxes[k,6],im_boxes[k,5],np.max(im_boxes[k,7:])),
				c=plt.cm.tab20((p_c+color_offset)%20), fontsize=2,clip_on=True)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground="black"), path_effects.Normal()])
			c_text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground="black"), path_effects.Normal()])
		
	plt.savefig("pred_mosaic.jpg",dpi=500, bbox_inches='tight')
	
	
	
	
	
	
	

