
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import albumentations as A
import os, cv2
from PIL import Image

#Comment to access system wide install
import sys, glob
sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

if(not os.path.isdir("ImageNET_aux_data")):
	os.system("wget https://share.obspm.fr/s/Pqbn4cC2azo84Z2/download/ImageNET_aux_data.tar.gz")
	os.system("tar -xvzf ImageNET_aux_data.tar.gz")

#Minimum deployement setup for prediction on a single image

image_size_val = 480
image_size = 448
flat_image_slice = image_size*image_size
nb_class = 1000

class_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_class_list.txt", dtype="str")[:,1]

if(not os.path.isfile("ImageNET_aux_data/office_1.jpg")):
	os.system("wget -P ImageNET_aux_data/ https://share.obspm.fr/s/GynmcyDtkrsbyLe/download/office_1.jpg")

im = Image.open("ImageNET_aux_data/office_1.jpg", mode='r')

if(im.format != "RGB"):
	im = im.convert('RGB')

patch = np.asarray(im)

transform = A.Compose([
	A.SmallestMaxSize(max_size=image_size_val, interpolation=1, p=1.0),
	A.PadIfNeeded(min_width=image_size_val, min_height=image_size_val, border_mode=cv2.BORDER_CONSTANT, p=1.0),
	A.CenterCrop(width=image_size, height=image_size, p=1.0),
])

transformed = transform(image=patch)
patch_aug = transformed['image']

input_data = f_ar(np.zeros((1,3*image_size*image_size)))
empty_target = f_ar(np.zeros((1,1000)))

for depth in range(0,3):
	input_data[0,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class, bias=0.1,
	 b_size=1, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=35)

cnn.create_dataset("TEST", 1, input_data, empty_target)

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch, load_epoch, bin=1)
else:
	#Not trained as a resolution agnostic network
	if(image_size == 224):
		if(not os.path.isfile("ImageNET_aux_data/CIANNA_net_model_imagenet_v1.0_darknet19custom_res224_acc70.dat")):
			os.system("wget -P ImageNET_aux_data/ https://zenodo.org/records/12801421/files/CIANNA_net_model_imagenet_v1.0_darknet19custom_res224_acc70.dat")
		cnn.load("ImageNET_aux_data/CIANNA_net_model_imagenet_v1.0_darknet19custom_res224_acc70.dat", 0, bin=1)
	elif(image_size == 448):
		if(not os.path.isfile("ImageNET_aux_data/CIANNA_net_model_imagenet_v1.0_darknet19custom_res448_acc74.dat")):
			os.system("wget -P ImageNET_aux_data/ https://zenodo.org/records/12801421/files/CIANNA_net_model_imagenet_v1.0_darknet19custom_res448_acc74.dat")
		cnn.load("ImageNET_aux_data/CIANNA_net_model_imagenet_v1.0_darknet19custom_res448_acc74.dat", 0, bin=1)
	else:
		print("No trained network for the define image resolution")
		exit()

cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")



top_error = 5

pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%(load_epoch), dtype="float32")
predict = np.reshape(pred_raw, (1,nb_class))
	
ind_best = np.argpartition(predict[0], -top_error)[-top_error:]
ind_sort = ind_best[np.argsort(predict[0, ind_best])][::-1]
pred_values = predict[0, ind_sort]


fig, ax = plt.subplots(1, 1, dpi=200, constrained_layout=True)

ax.imshow(patch)
ax.axis('off')

for k in range(0, top_error):
	c_text = ax.text(0.02, 1.0-0.04-k*0.04, "%0.2f - %s"%(pred_values[k], class_list[ind_sort[k]]), 
		c=plt.cm.tab20(ind_sort[k]%20), fontsize=6, clip_on=True, transform=ax.transAxes)
	c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])

plt.savefig("pred_on_image.jpg", dpi=200)
	

















