
from threading import Thread
from data_gen import *

#Comment to access system wide install
sys.path.insert(0,glob.glob("../../src/build/lib.*/")[-1])
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

#Construct training set using custom selection function
dataset_perscut(data_path+"TrainingSet_B1_v2.txt",data_path+"TrainingSet_perscut.txt", 18) #18 is the number of header line removed

#Prediction can be done in mixed_precision="FP16C_FP32A" with almost no loss in detection score
cnn.init(in_dim=i_ar([fwd_image_size,fwd_image_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param),
	bias=0.1, b_size=8, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=30, inference_only=1)

init_data_gen()
		
input_data = create_full_pred()
nb_images_all = np.shape(input_data)[0]
targets = np.zeros((nb_images_all,1+max_nb_obj_per_image*(7+nb_param)), dtype="float32")

cnn.create_dataset("TEST", nb_images_all, input_data[:,:], targets[:,:])

nb_yolo_filters = cnn.set_yolo_params(raw_output=0)

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch, load_epoch, bin=1)
else:
	if(not os.path.isfile(data_path+"YOLO_CIANNA_ref_SDC1_s480k_MINERVA_Cornu2024.dat")):
			os.system("wget -P %s https://zenodo.org/records/12801421/files/YOLO_CIANNA_ref_SDC1_s480k_MINERVA_Cornu2024.dat"%(data_path))
	cnn.load(data_path+"YOLO_CIANNA_ref_SDC1_s480k_MINERVA_Cornu2024.datt", 0, bin=1)
	
cnn.forward(no_error=1, saving=2, repeat=1, drop_mode="AVG_MODEL")











