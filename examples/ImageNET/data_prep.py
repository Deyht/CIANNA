
import numpy as np
from PIL import Image
import albumentations as A
import cv2
from multiprocessing import Pool
import os, glob


#/!\ Warning doawnloading ImageNET 2012 requires ~300 GB of free disk space. The .tar.gz files can be deleted after extraction if needed
#Deleting raw data after processing is possible but not advised in case you want to change the raw image resolution
#Downloading data folowing instructions from https://github.com/DoranLyong/ImageNet2012-download

data_path = "/Data-Linux/Work/MINERVA/ImageNet-2012/download_and_prepare_imagenet_dataset/"
processed_data_path = "/Data-Linux/Work/MINERVA/ImageNet-2012/"

#Get preprocessed path list and 1000 classes association
#Lower class count subdivision can be obtained at https://github.com/minyoungg/wmigftl

if(not os.path.isdir("ImageNET_aux_data")):
	os.system("wget https://share.obspm.fr/s/Pqbn4cC2azo84Z2/download/ImageNET_aux_data.tar.gz")
	os.system("tar -xvzf ImageNET_aux_data.tar.gz")

train_list = np.loadtxt("ImageNET_aux_data/imagenet_2012_1000classes_train.txt", dtype="str")
val_list  = np.loadtxt("ImageNET_aux_data/imagenet_2012_1000classes_val.txt", dtype="str")


nb_workers = 12

nb_train = 1281167
nb_val = 10000
image_size = 500
nb_class = 1000
	
transform = A.Compose([
	A.LongestMaxSize(max_size=image_size, interpolation=1),
	A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT),
])

unique, counts = np.unique(train_list[:,1], return_counts=True)

class_count = np.zeros((nb_class), dtype="int")

for i in range(0,nb_class):
	class_count[int(unique[i])] = int(counts[i])
	
if(not os.path.isdir(processed_data_path+"bin_blocks")):
	os.mkdir("%sbin_blocks"%(processed_data_path))

if(not os.path.isdir(processed_data_path+"bin_blocks/bin_images_%d"%(image_size))):
	os.mkdir("%sbin_blocks/bin_images_%d"%(processed_data_path,image_size))
	
os.system("rm -rf %sbin_blocks/bin_images_%d/*"%(processed_data_path,image_size))
for i in range(1,nb_class+1):
	os.mkdir("%sbin_blocks/bin_images_%d/C%04d"%(processed_data_path,image_size, i))
os.mkdir("%sbin_blocks/bin_images_%d/val"%(processed_data_path,image_size))

np.savetxt(processed_data_path+"bin_blocks/bin_images_%d/class_count.txt"%(image_size), class_count, fmt="%d")

def train_gen(i_d):
	
	pathes = glob.glob(data_path+"train/*/"+train_list[i_d,0][10:])

	if(np.shape(pathes)[0] == 0):
		pathes = glob.glob(data_path+"train/*/"+train_list[i_d,0][10:]+"G")

	im = Image.open(pathes[0], mode='r')
	
	if(im.format != "RGB"):
		im = im.convert('RGB')
	
	patch = np.asarray(im)
	
	transformed = transform(image=patch)
	patch_aug = transformed['image']
	
	np.save(processed_data_path+"bin_blocks/bin_images_%d/C%04d/train_img_raw%04d.npy"%(image_size, int(train_list[i_d,1])+1, i_d), patch_aug, allow_pickle=False)


def val_gen(i_d):
	
	pathes = glob.glob(data_path+"val/*/"+val_list[i_d,0])

	im = Image.open(pathes[0], mode='r')
	
	if(im.format != "RGB"):
		im = im.convert('RGB')

	patch = np.asarray(im)
		
	transformed = transform(image=patch)
	patch_aug = transformed['image']
	
	np.save(processed_data_path+"bin_blocks/bin_images_%d/val/valid_img_%04d.npy"%(image_size, i_d), patch_aug, allow_pickle=False)
"""
# Around 45 min with 12 workers on a 5900X CPU and a SATA SSD for storage
print("Starting training set images processing on %d threads ... This might take a while !"%(nb_workers))

with Pool(nb_workers) as p:
	p.map(train_gen, np.arange(nb_train))
"""
print("Starting validation set images processing on %d threads ..."%(nb_workers))

with Pool(nb_workers) as p:
	p.map(val_gen, np.arange(nb_val))
"""
print("Renaming training images ...")

#Remap proccesed names for easy random selection in batch augmentation
for k in range(0, nb_class):
	pathes = glob.glob(processed_data_path+"bin_blocks/bin_images_%d/C%04d/*"%(image_size,k+1))
	
	i_d = 0
	for path in pathes:
		os.rename(path,processed_data_path+"bin_blocks/bin_images_%d/C%04d/train_img_%04d.npy"%(image_size,k+1,i_d))
		i_d += 1
"""	


		
		
		
		
		
