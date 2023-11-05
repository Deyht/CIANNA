import numpy as np
from tqdm import tqdm
from PIL import Image
import os, glob

data_path = "./"

def make_square(im, min_size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

if(not os.path.isdir(data_path+"VOCdevkit")):
	os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
	os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar")
	os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar")

	os.system("tar -xf VOCtrainval_11-May-2012.tar")
	os.system("tar -xf VOCtrainval_06-Nov-2007.tar")
	os.system("tar -xf VOCtest_06-Nov-2007.tar")

#Training in train2012 + train2007 and testing on test 2007

train_list_2012 = np.loadtxt(data_path+"VOCdevkit/VOC2012/ImageSets/Main/trainval.txt", dtype="str")
train_list_2007 = np.loadtxt(data_path+"VOCdevkit/VOC2007/ImageSets/Main/trainval.txt", dtype="str")
test_list_2007  = np.loadtxt(data_path+"VOCdevkit/VOC2007/ImageSets/Main/test.txt"    , dtype="str")

nb_train_2012 = 11540
nb_train_2007 = 5011
nb_test_2007 = 4952
orig_nb_images = nb_train_2012 + nb_train_2007 + nb_test_2007
nb_keep_val = 4952
image_size_raw = 480
nb_class = 20


all_im = np.zeros((orig_nb_images, image_size_raw, image_size_raw, 3), dtype="uint8")
all_im_prop = np.zeros((orig_nb_images, 4), dtype="float32")

for i in tqdm(range(0, orig_nb_images)):
	
	if(i < nb_train_2012):
		im = Image.open(data_path+"VOCdevkit/VOC2012/JPEGImages/"+train_list_2012[i]+".jpg")
	elif(i < nb_train_2012+nb_train_2007):
		im = Image.open(data_path+"VOCdevkit/VOC2007/JPEGImages/"+train_list_2007[i - nb_train_2012]+".jpg")
	else:
		im = Image.open(data_path+"VOCdevkit/VOC2007/JPEGImages/"+test_list_2007[i - nb_train_2012 - nb_train_2007]+".jpg")
	
	width, height = im.size

	im = make_square(im, image_size_raw)
	width2, height2 = im.size

	x_offset = int((width2 - width)*0.5)
	y_offset = int((height2 - height)*0.5)

	all_im_prop[i] = [x_offset, y_offset, width2, height2]

	im = im.resize((image_size_raw,image_size_raw), resample=Image.BILINEAR)
	im_array = np.asarray(im)
	for depth in range(0,3):
		all_im[i,:,:,depth] = im_array[:,:,depth]
	
all_im.tofile("all_im.dat")
all_im_prop.tofile("all_im_prop.dat")









