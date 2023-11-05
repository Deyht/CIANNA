
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os, re, glob

from numba import jit

import albumentations as A
import cv2


data_path = "./"

image_size_raw = 480
image_size = 416
flat_image_slice = image_size*image_size
nb_class = 20
max_nb_obj_per_image = 56
nb_box = 5
yolo_reg_size = 32
yolo_nb_reg = int(image_size/yolo_reg_size)


nb_train_2012 = 11540
nb_train_2007 = 5011
nb_test_2007 = 4952
orig_nb_images = nb_train_2012 + nb_train_2007 + nb_test_2007

nb_images_per_iter = 4000
nb_keep_val = 4952

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def init_data_gen():
	global nb_train_2012, nb_train_2007, nb_test_2007, orig_nb_images, nb_images_per_iter, nb_keep_val, transform_train, transform_val
	global nb_class, max_nb_obj_per_image, image_size_raw, image_size, flat_image_slice, nb_box, yolo_reg_size, yolo_nb_reg
	global input_data, targets, input_val, targets_val, all_im, all_im_prop, class_count_val
	global class_list, train_list_2012, train_list_2007, test_list_2007
	
	class_list = np.array(["aeroplane","bicycle","bird","boat","bottle","bus","car",\
		"cat","chair","cow","diningtable","dog","horse","motorbike",\
		"person","pottedplant","sheep","sofa","train","tvmonitor"], dtype="str")
	
	train_list_2012 = np.loadtxt(data_path+"VOCdevkit/VOC2012/ImageSets/Main/trainval.txt", dtype="str")
	train_list_2007 = np.loadtxt(data_path+"VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", dtype="str")
	test_list_2007  = np.loadtxt(data_path+"VOCdevkit/VOC2007/ImageSets/Main/test.txt"    , dtype="str")
	
	transform_train = A.Compose([
		#Affine here act more as an aspect ratio transform than scaling
		A.Affine(scale=(0.85,1.15), rotate=(-7,7), interpolation=1, p=1.0),
		A.OneOf([
			A.RandomSizedCrop(min_max_height=[224,448], height=image_size, width=image_size, interpolation=1, p=6.0),
			A.Affine(scale=(0.7,1.0), translate_px=(-72,72), keep_ratio=True, interpolation=1, p=3.0),
			A.Affine(scale=(0.4,0.7), translate_px=(-144,144), keep_ratio=True, interpolation=1, p=3.0),
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
		], p=0.0),
		#Various types of noise / image alterations. Tend to reduce the mAP, but build a more resilient network for deployement or other applications
	], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, min_area=(12*12)))
	
	transform_val = A.Compose([
		A.Resize(width=image_size,height=image_size,interpolation=1)
	], bbox_params=A.BboxParams(format='pascal_voc'))
	
	#Processed images are just a reshap and padding so the largest side is equal to image_size_raw
	all_im = np.fromfile(data_path+"all_im.dat", dtype="uint8")
	all_im_prop = np.fromfile(data_path+"all_im_prop.dat", dtype="float32")
	all_im = np.reshape(all_im, ((orig_nb_images, image_size_raw, image_size_raw, 3)))
	all_im_prop = np.reshape(all_im_prop,(orig_nb_images, 4))

	class_count_val = np.zeros((nb_class))

	input_data = np.zeros((nb_images_per_iter,flat_image_slice*3), dtype="float32")
	targets = np.zeros((nb_images_per_iter,1+max_nb_obj_per_image*(7+1)), dtype="float32")

	input_val = np.zeros((nb_keep_val,flat_image_slice*3), dtype="float32")
	targets_val = np.zeros((nb_keep_val,1+max_nb_obj_per_image*(7+1)), dtype="float32")


def create_train_batch():

	for i in range(0, nb_images_per_iter):
		
		if(np.random.random() > 0.0):
		
			i_d = np.random.randint(0,orig_nb_images - nb_keep_val)
			
			if(i_d < nb_train_2012):
				tree = ET.parse(data_path+"VOCdevkit/VOC2012/Annotations/"+train_list_2012[i_d]+".xml")
			elif(i_d < nb_train_2012+nb_train_2007):
				tree = ET.parse(data_path+"VOCdevkit/VOC2007/Annotations/"+train_list_2007[i_d - nb_train_2012]+".xml")
			else:
				tree = ET.parse(data_path+"VOCdevkit/VOC2007/Annotations/"+test_list_2007[i_d - nb_train_2012 - nb_train_2007]+".xml")
			
			root = tree.getroot()
			x_offset, y_offset, width, height = all_im_prop[i_d]
			patch = np.copy(all_im[i_d])

			obj_list = root.findall("object", namespaces=None)
			bbox_list = np.zeros((len(obj_list),7))
			
			k = 0
			for obj in obj_list:
				diff = obj.find("difficult", namespaces=None)
				oclass = obj.find("name", namespaces=None)
				bndbox = obj.find("bndbox", namespaces=None)
				
				max_dim = max(width, height)
				int_class = int(np.where(class_list[:] == oclass.text)[0])
				
				xmin = (float(bndbox.find("xmin").text)+x_offset)*image_size_raw/width
				ymin = (float(bndbox.find("ymin").text)+y_offset)*image_size_raw/height
				xmax = (float(bndbox.find("xmax").text)+x_offset)*image_size_raw/width
				ymax = (float(bndbox.find("ymax").text)+y_offset)*image_size_raw/height
				
				x_mean = (xmin+xmax)/2.0
				y_mean = (ymin+ymax)/2.0
				
				c_size = int(image_size_raw/13.0)
				
				bbox_list[k,:] = np.array([xmin,ymin,xmax,ymax,int_class,0,k])
				if(diff.text == "1" or (x_mean < c_size or x_mean > image_size_raw - c_size or y_mean < c_size or y_mean > image_size_raw - c_size)
					or (abs(xmax-xmin)*abs(ymax-ymin)) < c_size*c_size):
					bbox_list[k,5] = 1
				k += 1
				
			bbs = np.copy(bbox_list[:,:])
			transformed = transform_train(image=patch,bboxes=bbs)
			patch_aug = transformed['image']
			bbs_aug = np.asarray(transformed['bboxes'])
			
			for depth in range(0,3):
				input_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C")-100.0)/155.0
			
			targets[i,:] = 0.0
			targets[i,0] = np.shape(bbs_aug)[0]
			for k in range(0, np.shape(bbs_aug)[0]):
				
				xmin = bbs_aug[k,0]
				ymin = bbs_aug[k,1]
				xmax = bbs_aug[k,2]
				ymax = bbs_aug[k,3]
				
				x_mean = (xmin+xmax)/2.0
				y_mean = (ymin+ymax)/2.0
				
				c_size = int(image_size/13.0)
				
				if((x_mean < c_size or x_mean > image_size - c_size or y_mean < c_size or y_mean > image_size - c_size)
					or (abs(xmax-xmin)*abs(ymax-ymin)) < c_size*c_size):
					bbs_aug[k,5] = 1
				
				width = max(16.0, xmax - xmin)
				height = max(16.0, ymax - ymin)
				
				xmin = max(0.0, x_mean - width/2.0)
				ymin = max(0.0, y_mean - height/2.0)
				xmax = min(image_size, x_mean + width/2.0)
				ymax = min(image_size, y_mean + height/2.0)
				
				targets[i,1+k*8:1+(k+1)*8] = np.array([bbs_aug[k,4]+1, xmin, ymin,0.0, xmax, ymax,1.0, bbs_aug[k,5]])
				
			if(targets[i,0] > max_nb_obj_per_image):
				targets[i,0] = max_nb_obj_per_image

	return input_data, targets


def visual_aug(visual_w, visual_h):
	
	l_style = ["-","--"]
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
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
		
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.6, ls=l_style[diff], fill=False, color=plt.cm.tab20(p_c), zorder=3)
			c_patch = ax[c_x,c_y].add_patch(el)
			c_text = ax[c_x,c_y].text(xmin+4, ymin+15, "%s %d"%(class_list[p_c], (xmax-xmin)*(ymax-ymin)), c=plt.cm.tab20(p_c), fontsize=3, clip_on=True)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground='black'),
											path_effects.Normal()])
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='black'),
											path_effects.Normal()])
	
	plt.savefig("augm_mosaic.jpg", dpi=400)


def create_val_batch():

	for i in range(0, nb_keep_val):
		
		i_d = nb_train_2012+nb_train_2007+nb_test_2007-nb_keep_val+i

		tree = ET.parse(data_path+"VOCdevkit/VOC2007/Annotations/"+test_list_2007[nb_test_2007-nb_keep_val+i]+".xml")
		root = tree.getroot()
		
		patch = np.copy(all_im[i_d])

		x_offset, y_offset, width, height = all_im_prop[i_d]

		obj_list = root.findall("object", namespaces=None)
		
		max_dim = max(width, height)
		bbox_list = np.zeros((len(obj_list),7))
		
		k = 0
		for obj in obj_list:
			diff = obj.find("difficult", namespaces=None)
			oclass = obj.find("name", namespaces=None)
			bndbox = obj.find("bndbox", namespaces=None)
			
			int_class = int(np.where(class_list[:] == oclass.text)[0])
			xmin = (float(bndbox.find("xmin").text)+x_offset)*image_size_raw/width
			ymin = (float(bndbox.find("ymin").text)+y_offset)*image_size_raw/height
			xmax = (float(bndbox.find("xmax").text)+x_offset)*image_size_raw/width
			ymax = (float(bndbox.find("ymax").text)+y_offset)*image_size_raw/height
			
			bbox_list[k,:] = np.array([xmin,ymin,xmax,ymax,int_class,0,k])
			if(diff.text != "1"):
				class_count_val[int_class] += 1
			else:
				bbox_list[k,5] = 1
			k += 1
			
		bbs = bbox_list[:,:]
		transformed = transform_val(image=patch,bboxes=bbs)
		patch_aug = transformed['image']
		bbs_aug = np.asarray(transformed['bboxes'])
		
		for depth in range(0,3):
			input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C")-100.0)/155.0
		
		targets_val[i,:] = 0.0
		targets_val[i,0] = np.shape(bbs_aug)[0]
		for k in range(0, np.shape(bbs_aug)[0]):
			
			xmin = bbs_aug[k,0]
			ymin = bbs_aug[k,1]
			xmax = bbs_aug[k,2]
			ymax = bbs_aug[k,3]
			
			orig_box = bbox_list[int(bbs_aug[k,6])]
			
			diff = bbs_aug[k,5]
				
			targets_val[i,1+k*8:1+(k+1)*8] = np.array([bbs_aug[k,4]+1,xmin,ymin,0.0,xmax,ymax,1.0,diff])
			
		if(targets_val[i,0] > max_nb_obj_per_image):
			targets_val[i,0] = max_nb_obj_per_image
	
	return input_val, targets_val


def visual_val(visual_w, visual_h):
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		targ_boxes = targets_val[i]
		for k in range(0, int(targ_boxes[0])):
			xmin = targ_boxes[1+k*8+1]
			ymin = targ_boxes[1+k*8+2]
			xmax = targ_boxes[1+k*8+4]
			ymax = targ_boxes[1+k*8+5]
			p_c = int(targ_boxes[1+k*8+0]) - 1
		
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.8, ls="--", fill=False, color=plt.cm.tab20(p_c), zorder=3)
			c_patch = ax[c_x,c_y].add_patch(el)
			c_text = ax[c_x,c_y].text(xmin+4, ymin+15, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c), fontsize=6, clip_on=True)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground='black'),
											path_effects.Normal()])
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'),
											path_effects.Normal()])
	
	plt.savefig("val_mosaic.jpg", dpi=400)


def free_data_gen():
  global all_im, all_im_prop, input_data, targets, input_val, targets_val
  del (all_im, all_im_prop, input_data, targets, input_val, targets_val)
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
def fct_classical_IoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1)
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1)
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0] + 1)*abs(box1[3] - box1[1] + 1) + \
		abs(box2[2]-box2[0] + 1)*abs(box2[3] - box2[1] + 1) - inter_2d

	return float(inter_2d)/float(uni_2d)


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
	
	c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
	c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
	c_box = np.zeros((6+1+nb_class),dtype="float32")
	
	final_boxes = []
	
	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%load_epoch, dtype="float32")
	predict = np.reshape(pred_raw, (nb_keep_val,nb_box*(8+nb_class),yolo_nb_reg,yolo_nb_reg))
	
	for l in tqdm(range(0, nb_keep_val)):
	
		c_tile[:,:] = 0.0
		c_tile_kept[:,:] = 0.0

		c_pred = predict[l,:,:,:]
		c_nb_box = box_extraction(c_pred, c_box, c_tile, obj_threshold, class_soft_limit)

		c_nb_box_final = c_nb_box
		amax_array = np.amax(c_tile[:,7:], axis=1)
		c_nb_box_final = apply_NMS(c_tile, c_tile_kept, c_box, c_nb_box, amax_array, nms_threshold_same, nms_threshold_diff)
		
		final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))


def pred_compute_map(AP_IoU_val=0.5, save_fig=0):

	global final_boxes
	
	for l in range(0,nb_keep_val):
		p_c = np.amax(final_boxes[l][:,7:], axis=1)
		final_boxes[l] = (final_boxes[l][(final_boxes[l][:,5]*p_c[:]**1).argsort()])[::-1]

	recall_precision = np.empty((nb_keep_val), dtype="object")

	print("Find associations ...", flush=True)

	for i_d in range(0, nb_keep_val):
	 
		recall_precision[i_d] = np.zeros((np.shape(final_boxes[i_d])[0], 6))

		if(np.shape(final_boxes[i_d])[0] == 0):
			continue

		recall_precision[i_d][:,0] = np.amax(final_boxes[i_d][:,7:], axis=1)
		recall_precision[i_d][:,1] = final_boxes[i_d][:,5]

		recall_precision[i_d][:,5] = np.argmax(final_boxes[i_d][:,7:], axis=1)

		kept_boxes = targets_val[i_d]
		kept_mask = np.zeros(int(kept_boxes[0]), dtype="int")

		for i in range(0,np.shape(final_boxes[i_d])[0]):
			best_IoU = -2.0
			best_targ = -1
			for j in range(0,int(kept_boxes[0])):
				xmin = (kept_boxes[1+j*8+1])
				ymin = (kept_boxes[1+j*8+2])
				xmax = (kept_boxes[1+j*8+4])
				ymax = (kept_boxes[1+j*8+5])
				c_kept_box = np.array([xmin, ymin, xmax, ymax])
				c_IoU = fct_classical_IoU(c_kept_box, final_boxes[i_d][i,:4])
				if(c_IoU > best_IoU and np.argmax(final_boxes[i_d][i,7:]) == int(kept_boxes[1+j*8+0]-1) and kept_mask[j] == 0):
					best_IoU = c_IoU
					best_targ = j
				
			if (best_IoU >= AP_IoU_val):
				if(kept_boxes[1+best_targ*8+7] > 0.99):
					recall_precision[i_d][i,2] = -1
				else:
					recall_precision[i_d][i,2] = 1
					recall_precision[i_d][i,3] = best_targ
					recall_precision[i_d][i,4] = c_IoU
					kept_mask[best_targ] = 1


	print("Process and flatten the mAP result")
	flatten = np.vstack(recall_precision.flatten())

	recall_precision_f = np.zeros((np.shape(flatten)[0], 10))
	recall_precision_f[:,:6] = flatten[:,:]

	recall_precision_fs = (recall_precision_f[(recall_precision_f[:,1]*recall_precision_f[:,0]**1).argsort()])[::-1]

	ignore_index = np.where(recall_precision_fs[:,2] == -1)[0]

	recall_precision_fs = np.delete(recall_precision_fs,ignore_index, axis=0)

	recall_precision_fs[:,6] = np.cumsum(recall_precision_fs[:,2])
	recall_precision_fs[:,7] = np.cumsum(1.0 - recall_precision_fs[:,2])
	recall_precision_fs[:,8] = recall_precision_fs[:,6] / (recall_precision_fs[:,6]+recall_precision_fs[:,7])
	recall_precision_fs[:,9] = recall_precision_fs[:,6] / np.sum(class_count_val)

	interp_curve = np.maximum.accumulate(recall_precision_fs[::-1,8])[::-1]

	AP_all = np.trapz(interp_curve, recall_precision_fs[:,9])
	print ("AP_all (%.2f): %f%%"%(AP_IoU_val, AP_all*100.0))
	
	if(save_fig):
		plt.figure(figsize=(4*1.0,3*1.0), dpi=200, constrained_layout=True)
		plt.plot(recall_precision_fs[:,9], recall_precision_fs[:,8])
		plt.plot(recall_precision_fs[:,9], interp_curve, label="New")
		plt.xlabel(r"Recall")
		plt.ylabel(r"Precision")
		plt.title("All classes as one AP curve", fontsize=8)

	sumAP = 0
	print ("**** Per class AP ****")
	if(save_fig):
		fig, ax = plt.subplots(figsize=(4*1.3,3*1.3), dpi=200, constrained_layout=True)
		plt.xlabel(r"Recall")
		plt.ylabel(r"Precision")
	
	for k in range(0, nb_class):
		index = np.where(recall_precision_fs[:,5] == k)
		l_recall_precision_fs = recall_precision_fs[index[0]]

		l_recall_precision_fs[:,6] = np.cumsum(l_recall_precision_fs[:,2])
		l_recall_precision_fs[:,7] = np.cumsum(1.0 - l_recall_precision_fs[:,2])
		l_recall_precision_fs[:,8] = l_recall_precision_fs[:,6] / (l_recall_precision_fs[:,6]+l_recall_precision_fs[:,7])
		l_recall_precision_fs[:,9] = l_recall_precision_fs[:,6] / class_count_val[k]
		 
		interp_curve = np.maximum.accumulate(l_recall_precision_fs[::-1,8])[::-1]
		
		AP = np.trapz(interp_curve, l_recall_precision_fs[:,9])
		sumAP += AP
		
		if(save_fig):
			plt.plot(l_recall_precision_fs[:,9], interp_curve, label=class_list[k],c=plt.cm.tab20(k))
		
		print("AP %-15s: %5.2f%%   Total: %4d - T: %4d - F: %4d"%(class_list[k], AP*100.0, class_count_val[k], l_recall_precision_fs[-1,6], l_recall_precision_fs[-1,7]))

	print ("\n**** mAP (%.2f): %f%% ****"%(AP_IoU_val, sumAP/nb_class*100.0))
	
	if(save_fig):
		plt.legend(bbox_to_anchor=(1.02,0.98), fontsize=8)
		plt.title("Per class AP curve", fontsize=8)
		plt.savefig("AP_curve_@%.2f_per_class.jpg"%(AP_IoU_val))
	
	
def visual_pred(visual_w, visual_h, display_target=0, id_start=0):
	#Visualize at image_size_raw resolution
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(1.5*visual_w,1.5*visual_h), dpi=210, constrained_layout=True)
	
	for i in range(0, visual_h):
		for j in range(0, visual_w):
			i_d = i*visual_w + j + id_start
			
			c_data = all_im[nb_train_2007 + nb_test_2007 + nb_train_2012 - nb_keep_val + i_d]/255.0
			ax[i,j].imshow(c_data)
			ax[i,j].axis('off')
			
			im_boxes = final_boxes[i_d]
			
			if(display_target):
				targ_boxes = targets_val[i_d]
				for k in range(0, int(targ_boxes[0])):
					xmin = targ_boxes[1+k*8+1] *(image_size_raw/image_size)
					ymin = targ_boxes[1+k*8+2] *(image_size_raw/image_size)
					xmax = targ_boxes[1+k*8+4] *(image_size_raw/image_size)
					ymax = targ_boxes[1+k*8+5] *(image_size_raw/image_size)
					p_c = int(targ_boxes[1+k*8+0]) - 1
				
					el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.4, ls="--", fill=False, color=plt.cm.tab20(p_c), zorder=3)
					c_patch = ax[i,j].add_patch(el)
					c_text  = ax[i,j].text(xmin+4, ymin+10, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c), fontsize=2, clip_on=True)
					c_patch.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
													path_effects.Normal()])
					c_text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
													path_effects.Normal()])
				

			for k in range(0, np.shape(im_boxes)[0]):
				xmin = max(-0.5,(im_boxes[k,0])*(image_size_raw/image_size) - 0.5)
				ymin = max(-0.5,(im_boxes[k,1])*(image_size_raw/image_size) - 0.5)
				xmax = min(image_size_raw-0.5,(im_boxes[k,2])*(image_size_raw/image_size) - 0.5)
				ymax = min(image_size_raw-0.5,(im_boxes[k,3])*(image_size_raw/image_size) - 0.5)
				
				p_c = np.argmax(im_boxes[k,7:])
				
				el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.4, fill=False, color=plt.cm.tab20(p_c), zorder=3)
				c_patch = ax[i,j].add_patch(el)
				c_text = ax[i,j].text(xmin+5, ymax-4, "%s:%d-%0.2f-%0.2f"%(class_list[p_c],im_boxes[k,6],im_boxes[k,5],np.max(im_boxes[k,7:])), c=plt.cm.tab20(p_c), fontsize=2,clip_on=True)
				c_patch.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
													path_effects.Normal()])
				c_text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
													path_effects.Normal()])

	plt.savefig("pred_mosaic.jpg",dpi=500, bbox_inches='tight')
	
	
	
	
	
	
	

