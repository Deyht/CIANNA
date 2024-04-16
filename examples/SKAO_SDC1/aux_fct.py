
import numpy as np
from scipy.interpolate import interpn
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs import utils
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from tqdm import tqdm
import os,re,sys,glob
from numba import jit

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
class ScalarFormatterForceFormat(ScalarFormatter):
	def _set_format(self):  # Override function that finds format to use.
		self.format = "%1.1f"  # Give format here

from ska_sdc import Sdc1Scorer


plt.rcParams.update({"font.size": 10})

global map_pixel_size, beam_size, pixel_size
global full_img, wcs_img, data_beam, wcs_beam
global image_size, min_pix, max_pix
global nb_images_iter, add_noise_prop, nb_valid, max_nb_obj_per_image, max_epoch
global nb_param, nb_box, nb_class, c_size, yolo_nb_reg
global flip_hor, flip_vert, rotate_flag
global box_clip, flux_lim_frac, bmaj_clip, bmin_clip, pa_res_lim
global patch_shift, orig_offset, nb_area_w, nb_area_h, nb_images_all, overlap
global val_med_lims, val_med_obj, first_nms_thresholds 
global first_nms_obj_thresholds, second_nms_threshold 
global data_path 

data_path = "./SDC1_data/"

if(not os.path.isdir(data_path)):
	os.system("mkdir %s"%(data_path))


######	  GLOBAL VARIABLES AND DATA	  #####
map_pixel_size = 32768 # Full SDC1 image size
beam_size = 0.625 #in arcsec
pixel_size = 0.000167847 # In degree


#Load the full SDC1 image
#Get the data from SKAO SDC1 website, the data can be downloaded elswhere if needed
if(not os.path.isfile(data_path+"SKAMid_B1_1000h_v3.fits")):
	os.system("wget -P %s https://owncloud.ia2.inaf.it/index.php/s/hbasFhd4YILNkCr/download -O %sSKAMid_B1_1000h_v3.fits"%(data_path, data_path))
print (data_path+"SKAMid_B1_1000h_v3.fits")
hdul     = fits.open(data_path+"SKAMid_B1_1000h_v3.fits")
full_img = hdul[0].data[0,0]
wcs_img  = WCS(hdul[0].header)

#Load primary beam for flux correction
if(not os.path.isfile(data_path+"PrimaryBeam_B1.fits")):
	os.system("wget -P %s https://owncloud.ia2.inaf.it/index.php/s/ZbaSDe7zGBYgxL1/download -O %sPrimaryBeam_B1.fits"%(data_path, data_path))
hdul_beam = fits.open(data_path+"PrimaryBeam_B1.fits")
data_beam = hdul_beam[0].data[0,0]
wcs_beam  = WCS(hdul_beam[0].header)


#Load the source catalogs
if(not os.path.isfile(data_path+"TrainingSet_B1_v2.txt")):
	os.system("wget -P %s https://owncloud.ia2.inaf.it/index.php/s/iTOVkIL6EfXkcdR/download -O %sTrainingSet_B1_v2.txt"%(data_path, data_path))
	
if(not os.path.isfile(data_path+"True_560_v2.txt")):
	os.system("wget -P %s https://owncloud.ia2.inaf.it/index.php/s/CZENkk6JdyVqIHw/download -O %sTrue_560_v2.txt"%(data_path, data_path))


#Input clipping before normalization
min_pix = 0.4e-6
max_pix = 0.4e-4

#####    NETWORK RELATED GLOBAL VARIABLES     #####
image_size 	= 256
nb_param  	= 5
nb_box 		= 9
nb_class	= 0
max_epoch 	= 5000
max_nb_obj_per_image = int(340*((image_size*image_size)/(256*256)))


#####    TRAINING RELATED GLOBAL VARIABLES    #####
nb_images_iter 	= 1600
add_noise_prop 	= 0.05 #Proportion of "noise" field examples in nb_images
nb_valid 		= int(100*((256*256)/(image_size*image_size)))

flip_hor 	= 0.5  #total proportion
flip_vert 	= 0.5
rotate_flag = 1

box_clip 	= [5.0,64.0]
flux_clip 	= [1.9e-06, 0.002]
bmaj_clip 	= [0.9, 60.0]
bmin_clip 	= [0.3, 30.0]
pa_res_lim 	= 1.8


#####   INFERENCE RELATED GLOBAL VARIABLES    #####
fwd_image_size = 512
c_size = 16 #Grid element size / reduction factor
yolo_nb_reg = int(fwd_image_size/c_size) #Number of grid element per dimension

overlap 	= c_size*2
patch_shift = fwd_image_size - overlap #240
orig_offset = int(int(map_pixel_size/2) - int(fwd_image_size/2) + 1)//patch_shift

nb_area_w = int((map_pixel_size-orig_offset)/patch_shift)
nb_area_h = int((map_pixel_size-orig_offset)/patch_shift)

nb_images_all = nb_area_w*nb_area_h

val_med_lims = np.array([0.6,0.3,0.1])
val_med_obj  = np.array([0.9,0.7,0.5])

first_nms_thresholds 	 = np.array([0.05,-0.1,-0.3,-0.5]) #lower is stricter
first_nms_obj_thresholds = np.array([1.0,0.70,0.50,0.30])
second_nms_threshold 	 = -0.15


@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0])*abs(box1[3] - box1[1]) + \
			 abs(box2[2]-box2[0])*abs(box2[3] - box2[1]) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

	return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)


@jit(nopython=True, cache=True, fastmath=False)
def tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, patch, val_med_lim, val_med_obj, hist_count):
	c_nb_box = 0
	for i in range(0,yolo_nb_reg):
		for j in range(0,yolo_nb_reg):
			kept_count = 0
			for k in range(0,nb_box):
				offset = int(k*(8+nb_param))
				c_box[4] = c_pred[offset+6,i,j] #probability
				c_box[5] = c_pred[offset+7,i,j] #objectness
				#Manual objectness penality on the edges of the images (help for both obj selection and NMS)
				if((j == 0 or j == yolo_nb_reg-1 or i == 0 or i == yolo_nb_reg-1)):
					c_box[4] = max(0.03,c_box[4]-0.05)
					c_box[5] = max(0.03,c_box[5]-0.05)
				
				if(c_box[5] >= prob_obj_cases[k]):
					bx = (c_pred[offset+0,i,j] + c_pred[offset+3,i,j])*0.5
					by = (c_pred[offset+1,i,j] + c_pred[offset+4,i,j])*0.5
					bw = max(5.0, c_pred[offset+3,i,j] - c_pred[offset+0,i,j])
					bh = max(5.0, c_pred[offset+4,i,j] - c_pred[offset+1,i,j])
					
					c_box[0] = bx - bw*0.5; c_box[1] = by - bh*0.5
					c_box[2] = bx + bw*0.5; c_box[3] = by + bh*0.5
					
					xmin = max(0,int(c_box[0]-5)); xmax = min(fwd_image_size,int(c_box[2]+5))
					ymin = max(0,int(c_box[1]-5)); ymax = min(fwd_image_size,int(c_box[3]+5))					
					
					#Remove false detections over very large and very bright sources
					med_val_box = np.median(patch[ymin:ymax,xmin:xmax])
					if((med_val_box > val_med_lim[0] and c_box[5] < val_med_obj[0]) or\
					   (med_val_box > val_med_lim[1] and c_box[5] < val_med_obj[1]) or\
					   (med_val_box > val_med_lim[2] and c_box[5] < val_med_obj[2])):
						continue
					
					c_box[6] = k
					c_box[7:7+nb_param] = c_pred[offset+8:offset+8+nb_param,i,j]
					c_box[-1] = i*yolo_nb_reg+j
					c_tile[c_nb_box,:] = c_box[:]
					c_nb_box += 1
					kept_count += 1
					
			hist_count[kept_count] += 1
			
	return c_nb_box


@jit(nopython=True, cache=True, fastmath=False)
def first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, nms_thresholds, obj_thresholds):
	c_nb_box_final = 0
	is_match = 1
	c_box_size_prev = c_nb_box
	
	while(c_nb_box > 0):
		max_objct = np.argmax(c_tile[:c_box_size_prev,5])
		c_box = np.copy(c_tile[max_objct])
		c_tile[max_objct,5] = 0.0
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1; c_nb_box -= 1; i = 0
		
		for i in range(0,c_box_size_prev):
			if(c_tile[i,5] < 0.0000000001):
				continue
			IoU = fct_DIoU(c_box[:4], c_tile[i,:4])
			if(((IoU > nms_thresholds[0] and c_tile[i,5] < obj_thresholds[0]) or
			    (IoU > nms_thresholds[1] and c_tile[i,5] < obj_thresholds[1]) or
			    (IoU > nms_thresholds[2] and c_tile[i,5] < obj_thresholds[2]) or
			    (IoU > nms_thresholds[3] and c_tile[i,5] < obj_thresholds[3]))):
				c_tile[i,5] = 0.0
				c_nb_box -= 1
				
	return c_nb_box_final


@jit(nopython=True, cache=True, fastmath=False)
def second_NMS_local(boxes, comp_boxes, c_tile, direction, nms_threshold):
	c_tile[:,:] = 0.0
	nb_box_kept = 0
	
	mask_keep = np.where((boxes[:,0] > overlap) & (boxes[:,2] < patch_shift) &\
					(boxes[:,1] > overlap) & (boxes[:,3] < patch_shift))[0]
	mask_remain = np.where((boxes[:,0] <= overlap) | (boxes[:,2] >= patch_shift) |\
					(boxes[:,1] <= overlap) | (boxes[:,3] >= patch_shift))[0]
	
	nb_box_kept = np.shape(mask_keep)[0]
	c_tile[0:nb_box_kept,:] = boxes[mask_keep,:]
	shift_array = np.array([direction[0],direction[1],direction[0],direction[1]])
	comp_boxes[:,0:4] += shift_array[:]*patch_shift
	
	comp_mask_keep = np.where((comp_boxes[:,0] < fwd_image_size) & (comp_boxes[:,2] > 0) &\
					(comp_boxes[:,1] < fwd_image_size) & (comp_boxes[:,3] > 0))[0]
	
	for b_ref in mask_remain:
		found = 0
		for b_comp in comp_mask_keep:
			IoU = fct_DIoU(boxes[b_ref,:4], comp_boxes[b_comp,:4])
			if(IoU > nms_threshold and boxes[b_ref,5] < comp_boxes[b_comp,5]):
				found = 1
				break
		if(found == 0):
			c_tile[nb_box_kept,:] = boxes[b_ref,:]
			nb_box_kept += 1
		   
	return nb_box_kept





