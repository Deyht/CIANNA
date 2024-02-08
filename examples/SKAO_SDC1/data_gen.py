
from aux_fct import *

#Source list files format
#COLUMN1:    ID    [none]    Source ID
#COLUMN2:    RA (core)    [degs]    Right ascension of the source core
#COLUMN3:    DEC (core)    [degs]    DEcination of the source core
#COLUMN4:    RA (centroid)    [degs]    Right ascension of the source centroid
#COLUMN5:    DEC (centroid)    [degs]    Declination of the source centroid
#COLUMN6:    FLUX    [Jy]    integrated flux density
#COLUMN7:    Core frac    [none]    integrated flux density of core/total
#COLUMN8:    BMAJ    [arcsec]    major axis dimension
#COLUMN9:    BMIN    [arcsec]    minor axis dimension
#COLUMN10:    PA    [degs]
#PA (measured clockwise from the longitude-wise direction)
#COLUMN11:    SIZE    [none]    1,2,3 for LAS, Gaussian, Exponential
#COLUMN12:    CLASS    [none]    1,2,3 for SS-AGNs, FS-AGNs,SFGs
#COLUMN13:    SELECTION    [none]    0,1 to record that the source has not/has been injected in the simulated map due to noise level
#COLUMN14:    x    [none]    pixel x coordinate of the centroid, starting from 0
#COLUMN15:    y    [none]    pixel y coordinate of the centroid,starting from 0


#Creating custom sample selection
def dataset_perscut(dataset_path, out_file, skiprows):
	cat = np.loadtxt(dataset_path, skiprows=skiprows)
	if(np.shape(cat)[1] >= 13):
		index = np.where(cat[:,12] == 1)[0]
		cat = cat[index]
	print ("Orig. Dataset size: ", np.shape(cat))

	c = SkyCoord(ra=cat[:,1]*u.degree, dec=cat[:,2]*u.degree, frame="icrs")
	x, y = utils.skycoord_to_pixel(c, wcs_img)
	xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
	new_data_beam = np.nan_to_num(data_beam)
	beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")

	#Get the "apparent flux" to correspond to the visible flux in the beam convolved input image
	flux_beam = cat[:,5]*beamval
	bmaj_pix = np.clip(cat[:,7], 1.2, None)/(3600.0*pixel_size)*2
	bmin_pix = np.clip(cat[:,8], 0.6, None)/(3600.0*pixel_size)*2
	surface = bmaj_pix*bmin_pix
	
	index = np.where((((flux_beam/(surface) > 1.0e-7) & (flux_beam >= 7.0e-6)) |
		             ((flux_beam/(surface) > 2.5e-7) & (flux_beam > 1.65e-6) & (flux_beam < 7.0e-6))))

	print ("TrainingSet size after selection function: ", np.shape(index)[1])

	new_cat = cat[index]
	
	np.savetxt(out_file, new_cat[:,:12],
		fmt="%d %.8f %.8f %.8f %.8f %.6g %.8f %.3f %.3f %.3f %d %d")


def init_data_gen():

	global min_ra_train_pix, max_ra_train_pix, min_dec_train_pix
	global cut_data, coords, flux_list, bmaj_list, bmin_list, pa_list
	global area_width, area_height, noise_size
	global norm_data, norm_data_noise_1, norm_data_noise_2, norm_flux_data
	global input_data, targets, input_valid, targets_valid
	global full_cat_loaded, lims
	
	######################################################################
	#####                  TRAINING AREA DEFINITION                  #####
	######################################################################

	# From the scoring pipeline
	#560: {"ra_min": -0.6723, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.4061},

	#Loading the result of the training selection function
	train_list = np.loadtxt(data_path+"TrainingSet_perscut.txt")

	#Define the training zone in RA, DEC from the training source catalog
	min_ra_train = np.min(train_list[:,1]) ; max_ra_train = np.max(train_list[:,1])
	min_dec_train = np.min(train_list[:,2]); max_dec_train = np.max(train_list[:,2])

	#Define the training part of the image as the area containing the training examples
	#min-max ordering is set regarding the 0,0 pixel coordinate origine
	c_min = SkyCoord(ra = max_ra_train*u.degree, dec = min_dec_train*u.degree, frame="icrs")
	c_max = SkyCoord(ra = min_ra_train*u.degree, dec = max_dec_train*u.degree, frame="icrs")

	#Convert the training zone coordinates in image pixel coordinate
	min_ra_train_pix, min_dec_train_pix = utils.skycoord_to_pixel(c_min, wcs_img)
	max_ra_train_pix, max_dec_train_pix = utils.skycoord_to_pixel(c_max, wcs_img)
	
	min_ra_train_pix = int(min_ra_train_pix); min_dec_train_pix = int(min_dec_train_pix) 
	max_ra_train_pix = int(max_ra_train_pix); max_dec_train_pix = int(max_dec_train_pix)
	
	#Prevent some non-labeled edge sources to appear in training examples
	min_ra_train_pix += 10; max_ra_train_pix -= 10
	min_dec_train_pix += 10; max_dec_train_pix -= 10
	
	print ("\nTraining area edges:")
	print (min_ra_train, max_ra_train, min_dec_train, max_dec_train)
	print (min_ra_train_pix, max_ra_train_pix, min_dec_train_pix, max_dec_train_pix)
	
	area_width = (max_ra_train_pix - min_ra_train_pix)
	area_height = (max_dec_train_pix - min_dec_train_pix)
	print ("Training area size (pixels)")
	print (area_width, area_height)
	
	
	######################################################################
	#####                 AUX EMPTY AREA DEFINITION                  #####
	######################################################################
	
	#Define two zones of the whole unlabeled-image where there should be no detectable source anymore
	#The propotion used at training time is defined by add_noise_prop in aux_fct
	noise_size = [2000,5600]
	noise_offset = 250
	
	min_x_noise = [noise_offset,map_pixel_size-noise_offset-noise_size[0]]
	max_x_noise = [noise_offset+noise_size[0],map_pixel_size-noise_offset]
	min_y_noise = [int((map_pixel_size-noise_size[1])/2.0),int((map_pixel_size-noise_size[1])/2.0)]
	max_y_noise = [int((map_pixel_size+noise_size[1])/2.0),int((map_pixel_size+noise_size[1])/2.0)]

	#Get 3 cutouts from the full image: The training cutout, and two "no-sources / noise-only" cutouts
	cut_data = full_img[min_dec_train_pix:max_dec_train_pix, min_ra_train_pix:max_ra_train_pix]
	cut_data_noise_1 = full_img[min_y_noise[0]:max_y_noise[0], min_x_noise[0]:max_x_noise[0]]
	cut_data_noise_2 = full_img[min_y_noise[1]:max_y_noise[1], min_x_noise[1]:max_x_noise[1]]


	######################################################################
	#####             TRAINING SOURCE CATALOG DEFINITION             #####
	######################################################################	
	
	#Get the sky coordinate of all sources in the selected training catalog
	c = SkyCoord(ra=train_list[:,1]*u.degree, dec=train_list[:,2]*u.degree, frame="icrs")
	x, y = utils.skycoord_to_pixel(c, wcs_img)

	n_w       = np.zeros((np.shape(train_list)[0]))
	n_h       = np.zeros((np.shape(train_list)[0]))
	coords    = np.zeros((np.shape(train_list)[0],4))
	flux_list = np.zeros((np.shape(train_list)[0]))
	bmaj_list = np.zeros((np.shape(train_list)[0]))
	bmin_list = np.zeros((np.shape(train_list)[0]))
	pa_list   = np.zeros((np.shape(train_list)[0]))

	flux_list[:] = train_list[:,5]
	bmaj_list[:] = train_list[:,7]
	bmin_list[:] = train_list[:,8]
	pa_list[:]   = train_list[:,9]
	
	#Remap all the PA values so they are all in the range [-90,90]
	index = np.where((pa_list[:] > 90.0) & (pa_list[:] <= 270.0))
	pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 90.0)
	index = np.where((pa_list[:] > 270.0) & (pa_list[:] < 360.0))
	pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 270.0)

	w = train_list[:,7]/(3600.0*pixel_size)*2
	h = train_list[:,8]/(3600.0*pixel_size)*2
	
	for i in range(0,np.shape(train_list)[0]):
		W = w[i]
		H = h[i]
		vertices = np.array([[-W*0.5,-H*0.5],[-W*0.5,H*0.5],[W*0.5, -H*0.5],[W*0.5,H*0.5]])

		vertices_new = np.zeros((4,2))
		vertices_new[:,0] = np.cos(train_list[i,9]*np.pi/180.0)*vertices[:,0]  \
			+ np.sin(train_list[i,9]*np.pi/180.0)*vertices[:,1]
		vertices_new[:,1] = - np.sin(train_list[i,9]*np.pi/180.0)*vertices[:,0]\
			+ np.cos(train_list[i,9]*np.pi/180.0)*vertices[:,1]

		n_w[i] = max(vertices_new[:,0]) - min(vertices_new[:,0])
		n_h[i] = max(vertices_new[:,1]) - min(vertices_new[:,1])
	
	#Clip the too small boxes (in pixel size)
	n_w = np.clip(n_w, box_clip[0], box_clip[1])
	n_h = np.clip(n_h, box_clip[0], box_clip[1])

	#Convert the positions and sizes into coordinates inside the full image
	coords[:,0] = x - n_w[:]*0.5
	coords[:,1] = x + n_w[:]*0.5
	coords[:,2] = y - n_h[:]*0.5
	coords[:,3] = y + n_h[:]*0.5
	
	#Get the "apparent flux" to correspond to the visible flux in the beam convolved map
	#Require to get the value of the beam for each source position (approximation)
	xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
	new_data_beam = np.nan_to_num(data_beam)
	beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")

	flux_list[:] = flux_list[:]*beamval[:]
	
	#Cap the minimum and maximum value for the Flux, Bmaj and Bmin for the regression 
	#(not linked to the bounding boxes)
	flux_list = np.clip(flux_list, flux_clip[0], flux_clip[1])
	bmaj_list = np.clip(bmaj_list, bmaj_clip[0], bmaj_clip[1]) # In arcsec
	bmin_list = np.clip(bmin_list, bmin_clip[0], bmin_clip[1]) # In arcsec
	
	#Flag very small objects for which PA estimation is too difficult and set their target PA to 0
	small_id = np.where(bmaj_list[:] <= pa_res_lim) #In arcsec = 3.0 pixels
	pa_list[small_id] = 0.0

	#Switch to logscale for Flux, Bmaj, Bmin to obtain flatter distributions across scales
	flux_list = np.log(flux_list); bmaj_list = np.log(bmaj_list); bmin_list = np.log(bmin_list)

	#Get the normalization limits and save them to convert network predictions back to physical quantities
	flux_min = np.min(flux_list); flux_max = np.max(flux_list)
	bmaj_min = np.min(bmaj_list); bmaj_max = np.max(bmaj_list)
	bmin_min = np.min(bmin_list); bmin_max = np.max(bmin_list)

	lims = np.zeros((3,2))
	lims[0] = [flux_max, flux_min]; lims[1] = [bmaj_max, bmaj_min]; lims[2] = [bmin_max, bmin_min]

	print ("\nMin-Max values used for normalization (Flux, Bmaj, Bmin):")
	print (lims[0], lims[1], lims[2])
	np.savetxt("train_cat_norm_lims.txt", lims)
	
	#Normalize the extra parameters
	flux_list[:] = (flux_list[:] - flux_min)/(flux_max - flux_min)
	bmaj_list[:] = (bmaj_list[:] - bmaj_min)/(bmaj_max - bmaj_min)
	bmin_list[:] = (bmin_list[:] - bmin_min)/(bmin_max - bmin_min)
	
	######################################################################
	#####             Network input data normalization               #####
	######################################################################	
	
	#Normalize all possible input fields using a tanh scaling
	cut_data = np.clip(cut_data,min_pix,max_pix)
	norm_data = (cut_data - min_pix) / (max_pix-min_pix)
	norm_data = np.tanh(3.0*norm_data)
	
	cut_data_noise_1 = np.clip(cut_data_noise_1,min_pix,max_pix)
	norm_data_noise_1 = (cut_data_noise_1 - min_pix) / (max_pix-min_pix)
	norm_data_noise_1 = np.tanh(3.0*norm_data_noise_1)
	
	cut_data_noise_2 = np.clip(cut_data_noise_2,min_pix,max_pix)
	norm_data_noise_2 = (cut_data_noise_2 - min_pix) / (max_pix-min_pix)
	norm_data_noise_2 = np.tanh(3.0*norm_data_noise_2)
	
	input_data = np.zeros((nb_images_iter,image_size*image_size), dtype="float32")
	targets = np.zeros((nb_images_iter,1+max_nb_obj_per_image*(7+nb_param)), dtype="float32")
	
	input_valid = np.zeros((nb_valid,image_size*image_size), dtype="float32")
	targets_valid = np.zeros((nb_valid,1+max_nb_obj_per_image*(7+nb_param)), dtype="float32")


## Data augmentation
def create_train_batch():
	
	#Construct a randomly augmented batch from the training area and source catalog	
	for i in range(0, nb_images_iter):
		#####      RANDOM POSITION IN TRAINING REGION      #####
		if(np.random.rand() > add_noise_prop):
		
			#Select a random position inside the traning area
			p_x = np.random.randint(0,area_width-image_size)
			p_y = np.random.randint(0,area_height-image_size)
			
			patch = np.copy(norm_data[p_y:p_y+image_size,p_x:p_x+image_size])
			
			#Randomly set the image to be flipped (hor/vert) or rotated (-90,+90)
			flip_w = 0; flip_h = 0
			
			rot_90 = 0
			rot_rand = np.random.random()
			if(rotate_flag and rot_rand < 0.33):
				rot_90 = -1; patch = np.rot90(patch, k=-1, axes=(0,1))
			elif(rotate_flag and rot_rand < 0.66):
				rot_90 = 1; patch = np.rot90(patch, k=1, axes=(0,1))
			
			if(np.random.random() < flip_hor):
				flip_w = 1; patch = np.flip(patch, axis=1)
			if(np.random.random() < flip_vert):
				flip_h = 1; patch = np.flip(patch, axis=0)
			
			#The input is flatten to be in the proper format for CIANNA
			input_data[i,:] = patch.flatten("C")
			
			#Find all boxes that are fully contained in the selected cutout
			patch_boxes_id = np.where((coords[:,0] > min_ra_train_pix + p_x) &\
					(coords[:,1] < min_ra_train_pix + p_x + image_size) &\
					(coords[:,2] > min_dec_train_pix + p_y) &\
					(coords[:,3] < min_dec_train_pix + p_y + image_size))[0]
			
			keep_box_coords = coords[patch_boxes_id]
			pa_kept = np.copy(pa_list[patch_boxes_id[:]])
			
			#Convert to local image coordinate
			keep_box_coords[:,0:2] -= (min_ra_train_pix + p_x) - 0.5
			keep_box_coords[:,2:4] -= (min_dec_train_pix + p_y) - 0.5
			
			if(rot_90 == -1):
				mod_keep_box_coords_x = (image_size) - np.copy(keep_box_coords[:,2:4])
				mod_keep_box_coords_y = np.copy(keep_box_coords[:,0:2])
				keep_box_coords[:,0:2] = mod_keep_box_coords_x
				keep_box_coords[:,2:4] = mod_keep_box_coords_y
			elif(rot_90 == 1):
				mod_keep_box_coords_x = np.copy(keep_box_coords[:,2:4])
				mod_keep_box_coords_y = (image_size) - np.copy(keep_box_coords[:,0:2])
				keep_box_coords[:,0:2] = mod_keep_box_coords_x
				keep_box_coords[:,2:4] = mod_keep_box_coords_y
			
			keep_box_coords[:,0:2] = flip_w*(image_size) + np.sign(0.5-flip_w)*keep_box_coords[:,0:2]
			keep_box_coords[:,2:4] = flip_h*(image_size) + np.sign(0.5-flip_h)*keep_box_coords[:,2:4]
			
			if(rot_90 == -1):
				pa_kept[:] = -np.sign(pa_kept)*(90.0-np.abs(pa_kept[:]))
			elif(rot_90 == 1):
				pa_kept[:] = -np.sign(pa_kept)*(90.0-np.abs(pa_kept[:]))
			
			if(flip_h):
				pa_kept[:] = -pa_kept[:]
			if(flip_w):
				pa_kept[:] = -pa_kept[:]
			
			targets[i,:] = 0.0
			targets[i,0] = min(max_nb_obj_per_image, np.shape(patch_boxes_id)[0])
			if(targets[i,0] == max_nb_obj_per_image):
				print("Max obj per image limit reached")
			for k in range(0,int(targets[i,0])):
				xmin = min(keep_box_coords[k,0:2])
				xmax = max(keep_box_coords[k,0:2])
				ymin = min(keep_box_coords[k,2:4])
				ymax = max(keep_box_coords[k,2:4])
				
				targets[i,1+k*(7+nb_param):1+(k+1)*(7+nb_param)] = \
						np.array([1.0,xmin,ymin,0.0,xmax,ymax,1.0,\
						flux_list[int(patch_boxes_id[k])],\
						bmaj_list[int(patch_boxes_id[k])],\
						bmin_list[int(patch_boxes_id[k])],\
						np.cos(pa_kept[k]*np.pi/180.0),\
						(np.sin(pa_kept[k]*np.pi/180.0)+1.0)*0.5])
				
		else:
			#####      RANDOM POSITION IN NOISE REGIONS      #####
			p_x = np.random.randint(0,noise_size[0]-image_size)
			p_y = np.random.randint(0,noise_size[1]-image_size)

			#No target sources in this region (which is a simplification)
			keep_box_coords = np.empty(0)
			#Select one of the two noise region
			if(np.random.rand() > 0.5):
				patch = np.flip(np.copy(norm_data_noise_1[p_y:p_y+image_size,\
									 p_x:p_x+image_size]),axis=0)
			else:
				patch = np.flip(np.copy(norm_data_noise_2[p_y:p_y+image_size,\
									 p_x:p_x+image_size]),axis=0)
			
			#Randomly set the image to be flipped (hor/vert) or rotated (-90,+90)
			flip_w = 0; flip_h = 0
			
			rot_90 = 0
			rot_rand = np.random.random()
			if(rotate_flag and rot_rand < 0.33):
				rot_90 = -1; patch = np.rot90(patch, k=-1, axes=(0,1))
			elif(rotate_flag and rot_rand < 0.66):
				rot_90 = 1; patch = np.rot90(patch, k=1, axes=(0,1))
			
			if(np.random.random() < flip_hor):
				flip_w = 1; patch = np.flip(patch, axis=1)
			if(np.random.random() < flip_vert):
				flip_h = 1; patch = np.flip(patch, axis=0)
			
			input_data[i,:] = patch.flatten("C")
			targets[i,:] = 0.0
		
	return input_data, targets


def create_valid_batch():
	# Define a "static" regularly sampled "valid/test" dataset
	# Note: This dataset is not "distinct" from the training dataset in the sense that it is defined on the same training area.
	# This is not sufficient to properly monitor overtraining, but it is acceptable in the present context due to the presence
	# of the "scorer" on the full image (minus the training area) that is used afterward as a real "valid/test" dataset.

	patch_shift = image_size

	for i in range(0, nb_valid):
		
		p_x = int(i/int(area_height/image_size))*patch_shift
		p_y = int(i%int(area_height/image_size))*patch_shift
		
		patch = np.copy(norm_data[p_y:p_y+image_size,p_x:p_x+image_size])
		#The input is flattened to be in the proper format for CIANNA
		input_valid[i,:] = patch.flatten("C")
		
		#Find all boxes that are fully contained in the selected cutout
		patch_boxes_id = np.where((coords[:,0] > min_ra_train_pix + p_x) &\
				(coords[:,1] < min_ra_train_pix + p_x + image_size) &\
				(coords[:,2] > min_dec_train_pix + p_y) &\
				(coords[:,3] < min_dec_train_pix + p_y + image_size))[0]
		
		keep_box_coords = coords[patch_boxes_id]
		pa_kept = np.copy(pa_list[patch_boxes_id[:]])
		
		#Convert to local image coordinate
		keep_box_coords[:,0:2] -= (min_ra_train_pix + p_x) - 0.5
		keep_box_coords[:,2:4] -= (min_dec_train_pix + p_y) - 0.5
		
		targets_valid[i,:] = 0.0
		targets_valid[i,0] = min(max_nb_obj_per_image, np.shape(patch_boxes_id)[0])
		for k in range(0,int(targets_valid[i,0])):
				
			xmin = min(keep_box_coords[k,0:2])
			xmax = max(keep_box_coords[k,0:2])
			ymin = min(keep_box_coords[k,2:4])
			ymax = max(keep_box_coords[k,2:4])
			
			targets_valid[i,1+k*(7+nb_param):1+(k+1)*(7+nb_param)] = \
					np.array([1.0,xmin,ymin,0.0,xmax,ymax,1.0,
					flux_list[patch_boxes_id[k]], bmaj_list[patch_boxes_id[k]],\
					bmin_list[patch_boxes_id[k]], np.cos(pa_kept[k]*np.pi/180.0),\
					(np.sin(pa_kept[k]*np.pi/180.0)+1.0)*0.5])
	
	return input_valid, targets_valid


def create_full_pred():
	#Decompose the full SDC1 image into patches of the appropriate input size with partial overlap between them
	pred_all = np.zeros((nb_images_all,fwd_image_size*fwd_image_size), dtype="float32")
	patch = np.zeros((fwd_image_size,fwd_image_size), dtype="float32")
	#Re-build norm, since this function can be called without data_gen_init
	full_data_norm = np.clip(full_img,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) / (max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)
	
	for i_d in range(0,nb_images_all):
	
		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)

		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + fwd_image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + fwd_image_size - orig_offset

		px_min = 0; px_max = fwd_image_size
		py_min = 0; py_max = fwd_image_size

		set_zero = 0

		if(xmin < 0):
			px_min = -xmin; xmin = 0; set_zero = 1
		if(ymin < 0):
			py_min = -ymin; ymin = 0; set_zero = 1
		if(xmax > map_pixel_size):
			px_max = fwd_image_size - (xmax-map_pixel_size); xmax = map_pixel_size; set_zero = 1
		if(ymax > map_pixel_size):
			py_max = fwd_image_size - (ymax-map_pixel_size); ymax = map_pixel_size; set_zero = 1

		if(set_zero):
			patch[:,:] = 0.0
		
		patch[py_min:py_max,px_min:px_max] = full_data_norm[ymin:ymax,xmin:xmax]
		pred_all[i_d,:] = patch.flatten("C")
	
	return pred_all

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	


