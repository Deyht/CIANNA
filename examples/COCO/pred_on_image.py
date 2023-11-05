
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import patches
from PIL import Image
import albumentations as A
import cv2
from numba import jit

#Comment to access system wide install
import sys, glob, os
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


#Minimum deployement setup for prediction on a single image

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

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

#The network is resiliant to slight augment in image resolution, which increase the mAP
#We recommand changing image_size by step of 64 (2 grid elements)
#Here training resolution was 416

image_size = 416 + 64*3
flat_image_slice = image_size*image_size
nb_box = 5
nb_class = 80

color_offset = 0

max_nb_obj_per_image = 70

yolo_nb_reg = int(image_size/32)
c_size = 32

with open("classnames.txt") as f:
    class_list = [line.rstrip('\n') for line in f]

class_id_conv = np.arange(1,92)
class_id_conv = np.delete(class_id_conv, [11,25,28,29,44,65,67,68,70,82,90])


if(not os.path.isfile("office_1.jpg")):
	os.system("wget https://share.obspm.fr/s/GynmcyDtkrsbyLe/download/office_1.jpg")

im = Image.open("office_1.jpg", mode='r')

if(im.format != "RGB"):
	im = im.convert('RGB')

patch = np.asarray(im)

dim_long = np.argmax(im.size)
ratio = image_size/im.size[dim_long]

other_dim = int(np.mod(dim_long+1,2))
offset = np.zeros((2))
offset[dim_long] = 0.0
offset[other_dim] = max(0.0,image_size - im.size[other_dim]*ratio)/2.0

transform = A.Compose([
	A.LongestMaxSize(max_size=image_size, interpolation=1, p=1.0),
	A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
])

transformed = transform(image=patch)
patch_aug = transformed['image']

input_data = f_ar(np.zeros((1,3*image_size*image_size)))
empty_target = f_ar(np.zeros((1,1)))

for depth in range(0,3):
	input_data[0,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0



cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=1, b_size=1,
	comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", inference_only=1)

cnn.create_dataset("TEST", 1, input_data, empty_target)

cnn.set_yolo_params()

load_epoch = 0
if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
else:
	if(not os.path.isfile("net_train_coco_map50_39.9_fp16.dat")):
		os.system("wget https://share.obspm.fr/s/pqG4jFrkEWi3SWt/download/net_train_coco_map50_39.9_fp16.dat")
	cnn.load("net_train_coco_map50_39.9_fp16.dat", 0, bin=1)

cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=0)

cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")



pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%load_epoch, dtype="float32")
predict = np.reshape(pred_raw, (1, nb_box*(8+nb_class),yolo_nb_reg,yolo_nb_reg))

c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_class)),dtype="float32")
c_box = np.zeros((6+1+nb_class),dtype="float32")

final_boxes = []

#Choice of filters that produce visually appealing results (!= best mAP )
obj_threshold = 0.45
class_soft_limit = 0.3
nms_threshold_same = 0.4
nms_threshold_diff = 0.9


c_tile[:,:] = 0.0
c_tile_kept[:,:] = 0.0

c_pred = predict[0,:,:,:]
c_nb_box = box_extraction(c_pred, c_box, c_tile, obj_threshold, class_soft_limit)

c_nb_box_final = c_nb_box
amax_array = np.amax(c_tile[:,7:], axis=1)
c_nb_box_final = apply_NMS(c_tile, c_tile_kept, c_box, c_nb_box, amax_array, nms_threshold_same, nms_threshold_diff)
final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))


#Image is displayed at full resolution. Changing imshow and removing ratio allows to visualize the prediction at the resolution seen by the network.
fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=200, constrained_layout=True)

ax.imshow(patch)
ax.axis('off')

im_boxes = final_boxes[0]

for k in range(0, np.shape(im_boxes)[0]):
	xmin = max(0.0,(im_boxes[k,0]-offset[0])/ratio)
	ymin = max(0.0,(im_boxes[k,1]-offset[1])/ratio)
	xmax = min(im.size[0],(im_boxes[k,2]-offset[0])/ratio)
	ymax = min(im.size[1],(im_boxes[k,3]-offset[1])/ratio)
	
	p_c = np.argmax(im_boxes[k,7:])
	
	el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.4, fill=False, color=plt.cm.tab20((p_c+color_offset)%20), zorder=3)
	c_patch = ax.add_patch(el)
	c_text = ax.text(xmin+8, ymax-15, "%s:%d-%0.2f-%0.2f"%(class_list[p_c],im_boxes[k,6],im_boxes[k,5],np.max(im_boxes[k,7:])), c=plt.cm.tab20((p_c+color_offset)%20), fontsize=2,clip_on=True)
	c_patch.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
										path_effects.Normal()])
	c_text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
										path_effects.Normal()])

plt.savefig("pred_on_image.jpg",dpi=400, bbox_inches='tight')






