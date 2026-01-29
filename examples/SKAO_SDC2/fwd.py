
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from config import *
from aux_fwd import *
from aux_post_proc import *


### LDEV data download and pre-process ###

load_pre_norm = 0

if(not os.path.isfile("cont_ldev.fits")):
	os.system("wget --content-disposition https://www.dropbox.com/scl/fo/e847a6pnjtqk7xmxz6flt/AFDf-W-zw5X7vTVjlGa2ASY/cont_ldev.fits?rlkey=84kkeaw021ajh7t9lqur2n7p8")

if(load_pre_norm):
	if(not os.path.isfile("LDEV_norm_cube.bin")):
		os.system("wget --content-disposition https://share.obspm.fr/s/MxkMxeE3Scrza7C/download/LDEV_norm_cube.bin")
else:
	if(not os.path.isfile("sky_ldev_v2.fits")):
		os.system("wget --content-disposition https://www.dropbox.com/scl/fo/e847a6pnjtqk7xmxz6flt/AHOoJIR6IHfCa6bWMu0YIrg/sky_ldev_v2.fits?rlkey=84kkeaw021ajh7t9lqur2n7p8")
	
	print("Normalizing LDEV cube, this will take a while ...")
	cube_norm("sky_ldev_v2.fits","cont_ldev.fits")



### Actual network inference ###

# sys.path.insert(0,glob.glob('path_to_CIANNA/src/build/lib.*/')[-1])
import CIANNA as cnn

inputs_test = create_test_batch()
targets_test = np.empty((0,0))
nb_test = np.shape(inputs_test)[0]

# b_size and mixed_precision can be adapted regarding the hardware used for inference
# Note that changing these values can affect the final result by a small margin (change intermediate computation numerical approximations)
cnn.init(in_dim=np.array([sky_size,sky_size,freq_size], dtype="int"), in_nb_ch=1, out_dim=0,
    bias=0.1, b_size=32, comp_meth='C_CUDA', dynamic_load=1,
    mixed_precision="FP16C_FP32A", inference_only=1, adv_size=30)

cnn.create_dataset("TEST", nb_test, inputs_test[:,:], targets_test[:,:])

nb_yolo_filters = cnn.set_yolo_params(no_override = 0, raw_output = 0)

cnn.load("../../models/YOLO_CIANNA_net_model_SDC2_MC-BT2_MINERVA_Cornu2026.dat", 0, bin=1)

cnn.forward(saving=2, no_error=1)
cnn.delete_dataset("TEST")


### Post-processing (filtering / NMS) ###

pred_data = np.fromfile("fwd_res/net0_0000.dat", dtype="float32")
pred_data = np.reshape(pred_data, (nb_area_freq, nb_area_sky_ldev, nb_area_sky_ldev, nb_box*(8+nb_param), yolo_nb_freq_reg, yolo_nb_sky_reg, yolo_nb_sky_reg))

pred_boxes = []

c_tile = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")
c_tile_kept = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")
c_box = np.zeros((8+1+nb_param+1),dtype="float32")

total_n_box = 0
for p_freq in range(0,nb_area_freq):
	for p_dec in range(0,nb_area_sky_ldev):
		for p_ra in range(0,nb_area_sky_ldev):
			
			c_tile[:,:] = 0.0
			c_tile_kept[:,:] = 0.0
			c_pred = pred_data[p_freq,p_dec,p_ra,:,:,:]
			
			c_nb_box_final = tile_filter(c_pred, c_box, c_tile, nb_box, nb_param, yolo_nb_sky_reg, yolo_nb_freq_reg)
			c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box_final, -0.1)
			
			total_n_box += c_nb_box_final
			pred_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

if(total_n_box <= 1):
	print ("No prediction found, closing fwd. Final score: 0")
	exit()
else:
	pred_boxes = np.reshape(np.array(pred_boxes, dtype="object"), (nb_area_freq, nb_area_sky_ldev, nb_area_sky_ldev))

c_tile = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")

A = np.array([-1, 0, 1]); B = np.array([-1, 0, 1]); C = np.array([-1, 0, 1])
x, y, z = np.meshgrid(A, B, C)
dir_array = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

l_overlap = np.array((overlap_sky, overlap_sky, overlap_freq))
l_patch_shift = np.array((patch_shift_sky, patch_shift_sky, patch_shift_freq))
l_patch_size = np.array((sky_size, sky_size, freq_size))

# Second NMS over all the overlapping patches
for p_freq in range(0,nb_area_freq):
	for p_dec in range(0,nb_area_sky_ldev):
		for p_ra in range(0,nb_area_sky_ldev):
			boxes = np.copy(pred_boxes[p_freq,p_dec,p_ra])
			for l in range(0,np.shape(dir_array)[0]):
				if(p_freq+dir_array[l,2] >= 0 and p_freq+dir_array[l,2] <= nb_area_freq-1 and\
				   p_dec +dir_array[l,1] >= 0 and p_dec +dir_array[l,1] <= nb_area_sky_ldev-1  and\
				   p_ra  +dir_array[l,0] >= 0 and p_ra  +dir_array[l,0] <= nb_area_sky_ldev-1 ):
					comp_boxes = np.copy(pred_boxes[p_freq+dir_array[l,2],p_dec+dir_array[l,1],p_ra+dir_array[l,0]])
					c_nb_box = inter_patch_NMS(boxes, comp_boxes, c_tile, dir_array[l], l_overlap, l_patch_shift, l_patch_size, -0.3)
					boxes = np.copy(c_tile[0:c_nb_box,:])
			
			pred_boxes[p_freq,p_dec,p_ra] = np.copy(boxes)

# Convert boxes from per-input coordinates to original cube pixel coordinates
for p_freq in range(0,nb_area_freq):
	box_freq_offset = p_freq*patch_shift_freq
	for p_dec in range(0,nb_area_sky_ldev):
		box_dec_offset = p_dec*patch_shift_sky
		for p_ra in range(0,nb_area_sky_ldev):
			box_ra_offset = p_ra*patch_shift_sky
			
			pred_boxes[p_freq,p_dec,p_ra][:,0] = box_ra_offset   + pred_boxes[p_freq,p_dec,p_ra][:,0] - orig_offset_sky_ldev - 0.5
			pred_boxes[p_freq,p_dec,p_ra][:,1] = box_dec_offset  + pred_boxes[p_freq,p_dec,p_ra][:,1] - orig_offset_sky_ldev - 0.5
			pred_boxes[p_freq,p_dec,p_ra][:,2] = box_freq_offset + pred_boxes[p_freq,p_dec,p_ra][:,2] - orig_offset_freq     - 0.5
			pred_boxes[p_freq,p_dec,p_ra][:,3] = box_ra_offset   + pred_boxes[p_freq,p_dec,p_ra][:,3] - orig_offset_sky_ldev - 0.5
			pred_boxes[p_freq,p_dec,p_ra][:,4] = box_dec_offset  + pred_boxes[p_freq,p_dec,p_ra][:,4] - orig_offset_sky_ldev - 0.5
			pred_boxes[p_freq,p_dec,p_ra][:,5] = box_freq_offset + pred_boxes[p_freq,p_dec,p_ra][:,5] - orig_offset_freq     - 0.5

# Merge predicted box list from all input regions 
box_cat = np.vstack(pred_boxes.flatten())
box_cat = box_cat[box_cat[:,7].argsort(),:][::-1]

# Save filtered box catalog in detector format ordered by objectness
np.savetxt("pred_ldev_cat_filtered_repos_ordered.dat", box_cat)

# Load cube WCS and convert box pixel (x,y) coordinates to (RA,DEC)
hdul_ldev = fits.open("sky_ldev_v2.fits", memmap=True)
wcs_ldev = WCS(hdul_ldev[0].header)

cls = utils.pixel_to_skycoord((box_cat[:,3]+box_cat[:,0])*0.5, (box_cat[:,4]+box_cat[:,1])*0.5, wcs_ldev)
ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

# Convert all predicted quantities to the SDC2 source catalog format
cat_header = "id ra dec hi_size line_flux_integral central_freq pa i w20"
cat_size = int(np.shape(box_cat)[0])
final_box_cat = np.zeros((cat_size,9), dtype="float32")

lims = np.loadtxt("../../metadata/MC-BT2_train_cat_lims.txt")

final_box_cat[:,0] = np.arange(0,cat_size)
final_box_cat[:,[1,2]] = ra_dec_coords.T
final_box_cat[:,3] = box_cat[:,10]*lims[1,0] + lims[1,1]
final_box_cat[:,4] = np.exp(box_cat[:,9]*lims[0,0] + lims[0,1])
final_box_cat[:,5] = (box_cat[:,5]+box_cat[:,2])*0.5*pixel_size_freq + 9.5e8
final_box_cat[:,6] = np.mod(np.arctan2(np.clip(box_cat[:,12],0.0,1.0)*2.0-1.0, np.clip(box_cat[:,13],0.0,1.0)*2.0-1.0)*180.0/np.pi,360.0)
final_box_cat[:,7] = np.arccos(np.clip(box_cat[:,14],0.0,1.0))*180.0/np.pi
final_box_cat[:,8] = (np.exp(box_cat[:,11]*lims[2,0] + lims[2,1])*pixel_size_freq)/(final_box_cat[:,5]**2/1.4204e9)*299792.458

np.savetxt("pred_ldev_final_catalog.txt", final_box_cat, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f")


### Scoring the predicted catalog ###

# Require the ska-sdc python package
from ska_sdc import Sdc2Scorer

if(not os.path.isfile("sky_ldev_truthcat_v2.txt")):
	os.system("wget --content-disposition https://www.dropbox.com/scl/fo/e847a6pnjtqk7xmxz6flt/AIo4631BbKfwhW2NGhvtbkw/sky_ldev_truthcat_v2.txt?rlkey=84kkeaw021ajh7t9lqur2n7p8")

min_obj = 0.65 # Apply a permissive objectness filtering by default to reduce the optimization range
max_size = np.shape(final_box_cat)[0] - np.searchsorted(box_cat[:,7][::-1], min_obj)
pred_cat = final_box_cat[:max_size]
raw_cat = box_cat[:max_size]

# First scoring to extract per source score for threshold optimization
np.savetxt("pre_opt_cat.txt", pred_cat, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f")

sub_cat_path = "pre_opt_cat.txt"
truth_cat_path = "sky_ldev_truthcat_v2.txt"

scorer = Sdc2Scorer.from_txt(sub_cat_path, truth_cat_path, sub_skiprows=0, truth_skiprows=0)
scorer.run(detail=True)
score_details = scorer.score.scores_df

per_source_score = np.zeros((max_size)) - 1.0
per_source_score[score_details["id"]] = score_details.to_numpy()[:,1:].sum(axis=1)/7.0

nb_freq_bins = 20
nb_obj_bins = 35
optimized_catalog = []

for k in range(0, nb_freq_bins):
	index_c_freq = np.where((pred_cat[:,5] > 9.5e8 + k*2e8/nb_freq_bins) & (pred_cat[:,5] < 9.5e8 + (k+1)*2e8/nb_freq_bins))[0]
	l_pred = pred_cat[index_c_freq]
	l_obj = raw_cat[index_c_freq,7]
	l_score = per_source_score[index_c_freq]
	
	dig_bins = np.logspace(np.log10(min_obj),0,num=nb_obj_bins+1)
	dig_index = np.digitize(l_obj, bins=dig_bins, right=True)
	
	opt_array = np.zeros((nb_obj_bins,4))
	for l in range(0,nb_obj_bins):
		bin_object_id = np.where((dig_index[:] == l))[0]
		nb_tot_bin = int(np.shape(bin_object_id)[0])
		match_id = np.where(l_score[bin_object_id] > 0.0)[0]
		nb_match = np.shape(match_id)[0]
		
		avg_score = 0; l_purity = 0
		if(nb_match > 0):
			avg_score = np.mean(l_score[bin_object_id])
		if(nb_tot_bin > 0):
			l_purity = nb_match/nb_tot_bin
		add_score = np.sum(l_score[bin_object_id[match_id]]) - (nb_tot_bin-nb_match)
		
		opt_array[l,:] = [nb_tot_bin, l_purity, avg_score, add_score]
		
	for l in range(0,nb_obj_bins-1):
		if(np.all(np.cumsum(opt_array[l:,3]) > 0)):
			id_opt = l
			break
	opt_bin_obj_select = dig_bins[l-1]
	max_opt_size = np.shape(l_obj)[0] - np.searchsorted(l_obj[::-1], opt_bin_obj_select)
	optimized_catalog.append(l_pred[:max_opt_size])

optimized_catalog = np.vstack(optimized_catalog)
np.savetxt("pred_ldev_final_catalog_optimized.txt", optimized_catalog, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f")


# Final scoring from the optimized predicted source catalog
sub_cat_path = "pred_ldev_final_catalog_optimized.txt"
truth_cat_path = "sky_dev_truthcat_v2.txt"

scorer = Sdc2Scorer.from_txt(sub_cat_path, truth_cat_path, sub_skiprows=0, truth_skiprows=0)
scorer.run(detail=True)

print("Final score: {}".format(scorer.score.value))
print ("Ndet:", scorer.score.n_det, "\nNmatch:", scorer.score.n_match, "\nNfalse:", scorer.score.n_bad + scorer.score.n_false)
print ("Avg characterization score:",scorer.score.acc_pc, "\nPurity:", (scorer.score.n_match/(scorer.score.n_match+scorer.score.n_false)))

score_details = scorer.score.scores_df

print ("Avg subscores:")
print ("Position:",np.mean(score_details["position"]))
print ("Central frequency:",np.mean(score_details["central_freq"]))
print ("Line flux", np.mean(score_details["flux"]))
print ("HI size",np.mean(score_details["hi_size"]))
print ("Line width",np.mean(score_details["w20"]))
print ("PA",np.mean(score_details["pa"]))
print ("Inclination",np.mean(score_details["i"]))





