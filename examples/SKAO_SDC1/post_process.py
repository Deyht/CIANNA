

from aux_fct import *


#Run with load_epoch = 0 to apply post process to the doawnloaded ref model
load_epoch = 0
if (len(sys.argv) > 1):
	epoch_start = int(sys.argv[1])
	epoch_end = epoch_start
else:
	epoch_start = 200
	epoch_end   = 5000

epoch_interv 	= 200
training_only 	= False		#select training or test area
run_pred 		= 0 		#Only required if the prediction has not been made
opt_display 	= 0 		#Used to display all the objectness interval content for each detection unit
score_type 		= "score" 	#"score" or "purity"


#full_data_norm is used for the high flux in box rejection criteria
full_data_norm = np.clip(full_img, min_pix,max_pix)
full_data_norm = (full_data_norm - min_pix) / (max_pix-min_pix)
full_data_norm = np.tanh(3.0*full_data_norm)

opt_score_array = np.zeros((int((epoch_end-epoch_start)/epoch_interv+1),2))

for current_epoch in np.arange(epoch_start, epoch_end+1, epoch_interv):

	load_epoch = current_epoch

	if(run_pred == 1):
		os.system("python3 pred_network.py %d"%(load_epoch))
	pred_data = np.fromfile("fwd_res/net0_%04d.dat"%load_epoch, dtype="float32")

	#Repeat corresponds to the number of MC_MODEL predictions
	repeat = 1
	fwd_batch_size = 16
	#Only keep the mean, but any other statistic can be computed here
	if(repeat > 1):
		predict = np.reshape(pred_data, (int(nb_area_h * nb_area_w / fwd_batch_size), repeat, fwd_batch_size, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))
		predict = np.mean(predict, axis=1)
		predict = np.reshape(predict, (nb_area_h, nb_area_w, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))
	else:
		predict = np.reshape(pred_data, (nb_area_h, nb_area_w, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))
	
	
	for opt_search in range(0,4):

		if(opt_search == 0):
			#Low threshold to keep almost all non-zero predictions
			prob_obj_cases = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05,0.05])
			prob_obj_edges = prob_obj_cases + 0.0
		else:
			#Optimized thresholds from scorer
			prob_obj_cases = opt_thresholds
			prob_obj_edges = prob_obj_cases + 0.0

		final_boxes = []

		c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
		c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
		c_box = np.zeros((6+1+nb_param+1),dtype="float32")
		patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")

		box_count_per_reg_hist = np.zeros((nb_box+1), dtype="int")

		for ph in tqdm(range(0,nb_area_h)):
			for pw in range(0, nb_area_w):
				
				c_tile[:,:] = 0.0
				c_tile_kept[:,:] = 0.0
				
				xmin = pw*patch_shift - orig_offset
				xmax = pw*patch_shift + fwd_image_size - orig_offset
				ymin = ph*patch_shift - orig_offset
				ymax = ph*patch_shift + fwd_image_size - orig_offset

				if(ph == 0 or ph == nb_area_h-1 or pw == 0 or pw == nb_area_w-1):
					patch[:,:] = 0.0
				else:
					patch[:,:] = full_data_norm[ymin:ymax,xmin:xmax]
				
				c_pred = predict[ph,pw,:,:,:]
				c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, patch, val_med_lims, val_med_obj, box_count_per_reg_hist)

				c_nb_box_final = c_nb_box
				c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, first_nms_thresholds, first_nms_obj_thresholds)
				
				out_range = 1
				if(ph < out_range or ph >= nb_area_h-out_range or pw < out_range or pw >= nb_area_w-out_range):
				    c_nb_box_final = 0
						
				final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

		final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (nb_area_h, nb_area_w))

		c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")

		dir_array = np.array([[-1,0],[+1,0],[0,-1],[0,+1],[-1,+1],[+1,+1],[-1,-1],[+1,-1]])
		
		#Second NMS over all the overlapping patches
		for ph in tqdm(range(0, nb_area_h)):
			for pw in range(0, nb_area_w):
				boxes = np.copy(final_boxes[ph,pw])
				for l in range(0,8):
					if(ph+dir_array[l,1] >= 0 and ph+dir_array[l,1] <= nb_area_h-1 and\
						pw+dir_array[l,0] >= 0 and pw+dir_array[l,0] <= nb_area_w-1):
						comp_boxes = np.copy(final_boxes[ph+dir_array[l,1],pw+dir_array[l,0]])
						c_nb_box = second_NMS_local(boxes, comp_boxes, c_tile, dir_array[l], second_nms_threshold)
						boxes = np.copy(c_tile[0:c_nb_box,:])
				
				final_boxes[ph,pw] = np.copy(boxes)
		
		c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
		#Manually set all edge predictions to 0 (after second NMS !)
		#The network was not train to interpret these cases which results in artifacts
		#Moreover, there is almost no chance that a source is detectable at these locations
		boxes = np.copy(c_tile[0:0,:])
		for pw in range(0, nb_area_h):
			final_boxes[0,pw] = np.copy(boxes)
			final_boxes[nb_area_h-1,pw] = np.copy(boxes)
		for ph in range(0, nb_area_w):
			final_boxes[ph,0] = np.copy(boxes)
			final_boxes[ph,nb_area_w-1] = np.copy(boxes)
		
		#Convert back to full image pixel coordinates
		final_boxes_scaled = np.copy(final_boxes)
		for p_h in range(0, nb_area_h):
			box_h_offset = p_h*patch_shift - orig_offset
			for p_w in range(0, nb_area_w):
				box_w_offset = p_w*patch_shift - orig_offset
				final_boxes_scaled[p_h,p_w][:,0] = box_w_offset + final_boxes_scaled[p_h,p_w][:,0]
				final_boxes_scaled[p_h,p_w][:,2] = box_w_offset + final_boxes_scaled[p_h,p_w][:,2]
				final_boxes_scaled[p_h,p_w][:,1] = box_h_offset + final_boxes_scaled[p_h,p_w][:,1]
				final_boxes_scaled[p_h,p_w][:,3] = box_h_offset + final_boxes_scaled[p_h,p_w][:,3]

		#Order predictions by objectness score and convert to SDC scorer format
		flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
		flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,5].argsort(),:][::-1]

		#Pixel coordinates are shift by 0.5 due to the difference of pixel coordinate system between CIANNA and classical FITS format
		x_y_flat_kept = np.copy(flat_kept_scaled[:,0:2])
		x_y_flat_kept[:,0] = (flat_kept_scaled[:,0]+flat_kept_scaled[:,2])*0.5 - 0.5
		x_y_flat_kept[:,1] = (flat_kept_scaled[:,1]+flat_kept_scaled[:,3])*0.5 - 0.5

		if(training_only): #hardcoded for the default SDC1 training area
			training_area_id = np.where((x_y_flat_kept[:,0] > 16383) & (x_y_flat_kept[:,0] < 19853) &\
										(x_y_flat_kept[:,1] > 16730) & (x_y_flat_kept[:,1] < 19921))[0]
			flat_kept_scaled = flat_kept_scaled[training_area_id]
			x_y_flat_kept = x_y_flat_kept[training_area_id]

		#Convert all the predicted parameters to the scorer format and fill non-predicted values using default settings
		cls = utils.pixel_to_skycoord(x_y_flat_kept[:,0], x_y_flat_kept[:,1], wcs_img)
		ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

		w, h = flat_kept_scaled[:,2]-flat_kept_scaled[:,0], flat_kept_scaled[:,3]-flat_kept_scaled[:,1]

		catalog_size = np.shape(flat_kept_scaled)[0]

		cat_header = "RA(deg), DEC(deg), X(pix), Y(pix), W(pix), H(pix), Objectness(real), Flux(Jy/Beam), BMAJ(arcsec), BMIN(arcsec), PA(degree)"
		box_catalog = np.zeros((catalog_size,10), dtype="float32")

		lims = np.loadtxt("train_cat_norm_lims.txt")

		box_catalog[:,[0,1]] = ra_dec_coords.T
		box_catalog[:,[2,3]] = np.array([w[:], h[:]]).T
		box_catalog[:,4] = flat_kept_scaled[:,4]
		box_catalog[:,5] = flat_kept_scaled[:,5]
		box_catalog[:,6] = np.exp(flat_kept_scaled[:,7]*(lims[0,0] - lims[0,1]) + lims[0,1])
		box_catalog[:,7] = np.exp(flat_kept_scaled[:,8]*(lims[1,0] - lims[1,1]) + lims[1,1])
		box_catalog[:,8] = np.exp(flat_kept_scaled[:,9]*(lims[2,0] - lims[2,1]) + lims[2,1])
		box_catalog[:,9] =  np.clip(np.arctan2(np.clip(flat_kept_scaled[:,11],0.0,1.0)*2.0 - 1.0, np.clip(flat_kept_scaled[:,10],0.0,1.0))*180.0/np.pi,-90,90)

		coords = SkyCoord(box_catalog[:,0]*u.deg, box_catalog[:,1]*u.deg)
		index = np.where(coords.ra.deg > 90.0)
		
		ra = coords.ra.deg
		dec = coords.dec.deg

		ra[index[0]] -= 360.0

		#Hard coded for the default SDC1 training area
		index_train = np.where((ra[:] < -0.0) & (ra[:] > -0.6723) & (dec[:] < -29.4061) & (dec[:] > -29.9400))[0]

		xbeam, ybeam = utils.skycoord_to_pixel(coords, wcs_beam)
		new_data_beam = np.nan_to_num(data_beam)
		beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
			new_data_beam, (ybeam, xbeam), method="splinef2d")

		FLUX_JY = (box_catalog[:,6]/beamval)

		empty = np.zeros((np.shape(coords.ra.deg)[0]))
		scoring_table = np.vstack((np.arange(0,np.shape(box_catalog)[0]), ra, dec, ra, dec,\
									FLUX_JY, empty+0.0375, box_catalog[:,7], box_catalog[:,8], \
									box_catalog[:,9], empty+2.0, empty+3.0))

		np.savetxt(data_path+"YOLO_CIANNA_sdc1_catalog.txt", scoring_table.T, fmt="%d %1.8f %2.8f %1.8f %2.8f %g %0.8f %f %f %f %d %d")

		#Set the scorer parameters and compute the score for the current prediction
		sub_cat_path = data_path+"YOLO_CIANNA_sdc1_catalog.txt"
		truth_cat_path = data_path+"True_560_v2.txt"
		skip_truth = 0

		scorer = Sdc1Scorer.from_txt(sub_cat_path, truth_cat_path, freq=560, sub_skiprows=0, truth_skiprows=skip_truth)

		scorer.run(
			mode	= 0, 			 # 0, 1 for core, centroid position modes respectively
			train	= training_only, # True to score based on training area only, else exclude
			detail	= True, 		 # True to return per-source scores and match catalogue
			)

		detailed_score_display = 0 
		if(detailed_score_display):
			print("Final score: {}".format(scorer.score.value))

			print ("Ndet:", scorer.score.n_det)
			print ("Nmatch:", scorer.score.n_match)
			print ("Nbad:", scorer.score.n_bad)
			print ("Nfalse:", scorer.score.n_false)

			print ("Score det:",scorer.score.score_det)
			print ("Accuracy:",scorer.score.acc_pc)
			print ("Purity:", (scorer.score.n_match/(scorer.score.n_match+scorer.score.n_false)))

		opt_score_array[int((current_epoch-epoch_start)/epoch_interv),:] = np.array([load_epoch,scorer.score.value])

		matched = scorer.score.match_df
		id_match = matched.id[:]
		match_array = np.zeros((np.shape(box_catalog)[0]))
		match_array[id_match] = 1

		scores_df = scorer.score.scores_df
		score_array = np.zeros((np.shape(box_catalog)[0]))
		score_array[id_match] = scores_df.to_numpy()[:,1:].sum(axis=1)/7.0

		box_id = flat_kept_scaled[:,6]
		
		test_catalog = np.delete(box_catalog, index_train, axis=0)
		test_match_array = np.delete(match_array, index_train, axis=0)
		test_score_array = np.delete(score_array, index_train, axis=0)
		test_box_id = np.delete(box_id, index_train, axis=0)

		opt_sampling = 60
		
		if(score_type == "score"):
			bins = np.logspace(-1.5,0,num=opt_sampling) #Log scale for score to better sample low objectness
		else:
			bins = np.linspace(0,1,num=opt_sampling) #Linear scale for putiry to better sample high objectness

		dig_index = np.digitize(test_catalog[:,5], bins=bins, right=True)

		opt_array = np.zeros((nb_box, opt_sampling, 4))
		opt_thresholds = np.zeros((nb_box))

		if(opt_display):
			print("    Bin     |  N.det |  Purity  | AvgScore | AddScore")

		for k in range(0, nb_box):
			if(opt_display):
				print("BOX PRIOR %d"%(k))
			for i in range(0,opt_sampling-1):
				bin_object_id = np.where((dig_index[:] == i) & (test_box_id[:] == k))[0]
				nb_tot_bin = int(np.shape(bin_object_id)[0])
				nb_match = np.sum(test_match_array[bin_object_id])
				avg_score = 0
				l_purity = 0
				if(nb_match > 0):
					avg_score = np.sum(test_score_array[bin_object_id]*test_match_array[bin_object_id])/nb_match
				if(nb_tot_bin > 0):
					l_purity = nb_match/nb_tot_bin
				add_score = np.sum(test_score_array[bin_object_id]*test_match_array[bin_object_id]) - (nb_tot_bin-nb_match)
				
				opt_array[k,i,:] = np.array([nb_tot_bin, l_purity, avg_score, add_score])
				if(opt_display):
					print("[%0.3f-%0.3f] | %6d | %6f | %6f | %7.1f"%(bins[i], bins[i+1], nb_tot_bin, l_purity, avg_score, add_score))
		
		if(score_type == "score"):
		
			for k in range(0, nb_box):
				id_opt = opt_sampling-1
				for i in range(0,opt_sampling-1):
					if(opt_array[k,i,1] <= 0.630):
						opt_array[k,i,3] = 0.0
				for i in range(0,opt_sampling-1):
					if(np.all(np.cumsum(opt_array[k,i:,3]) > 0) and opt_array[k,i,0] >= 10):
						id_opt = i
						break
				if(opt_search < 1):
					opt_thresholds[k] = bins[id_opt-2]
				else:
					opt_thresholds[k] = bins[id_opt-1]
					
		if(score_type == "purity"):
		
			for k in range(0, nb_box):
				id_opt = opt_sampling-1
				for i in range(0,opt_sampling-1):
					if(opt_array[k,i,1] <= 0.6):
						opt_array[k,i,1] = 0.0
				for i in range(0,opt_sampling-1):
					if(opt_array[k,i,0] < 1):
						continue
					if(np.sum(opt_array[k,i:,0]*opt_array[k,i:,1]) / np.sum(opt_array[k,i:,0]) > 0.99):
						id_opt = i
						break
				if(opt_search < 1):
					opt_thresholds[k] = bins[id_opt-2]
				else:
					opt_thresholds[k] = bins[id_opt-1]

		print (opt_score_array)
		print (' '.join(map(str, np.around(opt_thresholds,4))))

		np.savetxt("iter_score_list.txt",opt_score_array)


























