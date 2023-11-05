
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os


data_path = "./"

#Downloading Training and Validation datasets if not already present
#datapath is define in aux_fct.py

if(not os.path.isdir(data_path+"annotations")):
	os.system("wget -P %s http://images.cocodataset.org/annotations/annotations_trainval2017.zip"%(data_path))
	os.system("unzip %sannotations_trainval2017.zip"%(data_path))
	os.system("rm %sannotations_trainval2017.zip"%(data_path))

if(not os.path.isdir(data_path+"train2017")):
	os.system("wget -P %s http://images.cocodataset.org/zips/train2017.zip"%(data_path))
	os.system("unzip %strain2017.zip"%(data_path))
	os.system("rm %strain2017.zip"%(data_path))

if(not os.path.isdir(data_path+"val2017")):
	os.system("wget -P %s http://images.cocodataset.org/zips/val2017.zip"%(data_path))
	os.system("unzip %sval2017.zip"%(data_path))
	os.system("rm %sval2017.zip"%(data_path))



data_prefix_list = ["val2017", "train2017"]

for data_prefix in data_prefix_list:

	with open(data_path+"annotations/instances_%s.json"%(data_prefix), "r") as f:
		data = json.load(f)
	print (data.keys())

	data_list = {}

	for item in data["images"]:
		data_list[item["id"]] = item["file_name"]

	key_id = list(data_list.keys())
	f_file = list(data_list.values())


	for i in tqdm(range(0, len(key_id))):
		
		im = Image.open(data_path+data_prefix+"/"+f_file[i], mode='r')
		if(im.format != "RGB"):
			im = im.convert('RGB')

		patch = np.asarray(im)
		
		np.save(data_path+data_prefix+"/"+f_file[i][:-4], patch, allow_pickle=False)

	data_list = {}
	data_list2 = {}
	data_list3 = {}

	for item in data["annotations"]:
		data_list[item["id"]] = item["bbox"]
		data_list2[item["id"]] = item["image_id"]
		data_list3[item["id"]] = item["category_id"]

	key_id = list(data_list.keys())
	bbox_list = list(data_list.values())
	im_id_list = list(data_list2.values())
	class_list = list(data_list3.values())

	for i in tqdm(range(0, len(key_id))):
		f = open(data_path+data_prefix+"/bbox_%d.txt"%(im_id_list[i]), "a")  # append mode
		f.write("%f %f %f %f %d\n"%(bbox_list[i][0], bbox_list[i][1], max(1.0,bbox_list[i][2]), max(1.0, bbox_list[i][3]), class_list[i]))
		f.close()



