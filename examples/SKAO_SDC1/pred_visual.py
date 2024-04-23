
from data_gen import *

plt.rcParams.update({'font.size': 12})

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap



# This script reproduces the Figures from Cornu et al. 2024
# We recommend looking at the captions in the paper for a full figure description
 

######################################################################
#####          Data preparation and prediction setup             #####
######################################################################

#Data normalization to display the dynamic of the network input
full_data_norm = np.clip(full_img, min_pix,max_pix)
full_data_norm = (full_data_norm - min_pix) / (max_pix-min_pix)
full_data_norm = np.tanh(3.0*full_data_norm)

#Used for some visualisation, take from train_network.py
prior_w = np.array([6.0,6.0,6.0,6.0,6.0,6.0,12.0, 9.0,24.0]) 
prior_h = np.array([6.0,6.0,6.0,6.0,6.0,6.0, 9.0,12.0,24.0])
prior_size = np.vstack((prior_w, prior_h))

#Hard coded for the training area of the SDC1
ra_min = -0.6723; ra_max = 0.0; dec_min = -29.9400; dec_max = -29.4061

c_min = SkyCoord(ra=ra_max*u.degree, dec=dec_min*u.degree, frame='icrs')
c_max = SkyCoord(ra=ra_min*u.degree, dec=dec_max*u.degree, frame='icrs')

min_ra_train_pix, min_dec_train_pix = utils.skycoord_to_pixel(c_min, wcs_img)
max_ra_train_pix, max_dec_train_pix = utils.skycoord_to_pixel(c_max, wcs_img)

load_iter = 0 #Change for your best iteration, using the pretrained model it should be 0

#Obj thresholds should be optimize for specif training or iter (using post_process.py)

#For best score (YOLO-CIANNA-ref)
prob_obj_cases = np.array([0.2314, 0.1449, 0.2602, 0.1289, 0.2454, 0.2183, 0.0602, 0.0677, 0.0536])

#For good precision (YOLO-CIANNA-alt)
#prob_obj_cases = np.array([0.6102, 0.5424, 0.6441, 0.4407, 0.6271, 0.6102, 0.7119, 0.7288, 0.88])

#To sample the full mAP curve (require low value for all)
#prob_obj_cases = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]) 

#To evaluate only on the training area
training_only = False

#If savefig is 1 then figures are stored, if 0 they are displays in order
savefig_all = 1

if(savefig_all and not os.path.isdir("figures")):
	os.system("mkdir figures")

######################################################################
#####            Display a cutout in the SDC1 image	             #####
######################################################################

cutout_size = 512

patch = Cutout2D(full_img, SkyCoord(ra=-0.426*u.deg, dec=-29.684*u.deg, frame="icrs"), (cutout_size,cutout_size), wcs_img.celestial)

fig = plt.figure(figsize=(8,6), dpi=150, constrained_layout=True)
ax = plt.subplot(projection=patch.wcs.celestial)

gcf = ax.imshow(patch.data*1e6, cmap="hot", vmax = 1.0*max_pix*1e6, vmin = min_pix*1e6)

ax.coords[0].set_major_formatter(r"d.dd")
ax.coords[1].set_major_formatter(r"d.dd")

ax.set_xlabel(r"RA (ICRS) [deg]", fontsize=14)
ax.set_ylabel(r"Dec (ICRS) [deg]", fontsize=14)

plt.tick_params(axis='y', which='both', left=True, labelleft=True, right=False, labelright=False)
plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
ax2 = ax.twiny(); ax2 = ax2.twinx()
ax2.set_xlabel("X [pix]", fontsize=14); ax2.set_ylabel("Y [pix]", fontsize=14)
ax2.set_yticks(np.linspace(-0.5,cutout_size-0.5,9))
ax2.set_xticks(np.linspace(-0.5,cutout_size-0.5,9))
ax2.set_yticklabels(np.linspace(0,cutout_size,9, dtype="int"))
ax2.set_xticklabels(np.linspace(0,cutout_size,9, dtype="int"))
ax2.invert_yaxis()

cbar = plt.colorbar(gcf, ax=ax, extend='max', pad=0.10)
cbar.set_label(label=r"Flux [$\mathrm{\mu Jy\,/\,beam}$]", fontsize=14)
if(savefig_all):
	plt.savefig("figures/example_field.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()



######################################################################
#####           Selection function flux distribution             #####
######################################################################

print("Visualize training sample selection function")

bins = 10**(np.linspace(-9,2,60))

full_train = np.loadtxt(data_path+"TrainingSet_B1_v2.txt", skiprows=18)
index = np.where(full_train[:,12] == 1)[0]
full_train = full_train[index]

c = SkyCoord(ra=full_train[:,1]*u.degree, dec=full_train[:,2]*u.degree, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")
flux_beam = full_train[:,5]*beamval
bmaj_pix = full_train[:,7]/(3600.0*pixel_size)*2
bmin_pix = full_train[:,8]/(3600.0*pixel_size)*2
surface = bmaj_pix*bmin_pix

selected_train = np.loadtxt(data_path+"TrainingSet_perscut.txt")
c = SkyCoord(ra=selected_train[:,1]*u.degree, dec=selected_train[:,2]*u.degree, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")

flux_beam_selected = selected_train[:,5]*beamval
bmaj_pix = selected_train[:,7]/(3600.0*pixel_size)*2
bmin_pix = selected_train[:,8]/(3600.0*pixel_size)*2
surface_selected = bmaj_pix*bmin_pix

fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=150, constrained_layout=True)

rwidth = 0.88
n, bins, patches1 = ax.hist(flux_beam, bins=bins, histtype="bar", color = "grey", rwidth = rwidth, 
		log=True, label="Full train", edgecolor='white', linewidth=0.0, rasterized=True)
n2, bins2, patches2 = ax.hist(flux_beam_selected, bins=bins, histtype="bar", 
		color="limegreen", alpha=0.8, rwidth = rwidth, log=True, label="Selected", edgecolor='white', linewidth=0.0, rasterized=True)

ax.set_xscale('log')
ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
ax.set_xlim(1e-9,1e1)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel("N. of objects", fontsize=18)
ax.set_xlabel("Apparent flux [Jy]", fontsize=17)

ax.legend(prop={'size': 16})

if(savefig_all):
	plt.savefig("figures/selection_function_dist_flux.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()



######################################################################
#####         Selection function in surface vs brightness        #####
######################################################################

cmap = truncate_colormap("terrain_r",0.08,1.0)

f_min = np.min(flux_beam); f_max = np.max(flux_beam)
s_min = np.min(surface); s_max = np.max(surface)

fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=150, constrained_layout=True)

bins_x = 10**np.linspace(-8.2, -1.4,80)
bins_y = 10**np.linspace(-2.6,6.15,80)

h = plt.hist2d(flux_beam,surface, bins=[bins_x,bins_y], norm=colors.LogNorm(), cmap=cmap, rasterized=True)
plt.xscale("log")
plt.yscale("log")
ax.tick_params(axis='both', which='major', labelsize=16)
plt.ylabel("Surface [pix²]", fontsize=18)
plt.xlabel("Apparent flux [Jy]", fontsize=17)
plt.xlim(f_min,1e-1)

plt.plot([7e-6,7e-6],[7e-6/2.5e-7,7e-6/1e-7], ls="--", color="red")
plt.plot([7e-6,f_max],[7e-6/1e-7,f_max/1e-7], ls="--", color="red")

plt.plot([1.65e-6,1.65e-6],[s_min,1.65e-6/2.5e-7], ls="--", color="red")
plt.plot([1.65e-6,7e-6],[1.65e-6/2.5e-7,7e-6/2.5e-7], ls="--", color="red")

plt.plot([f_min, f_max], [2.88,2.88], ls=":", color="grey")

plt.text(0.80,0.17,"Selected\nsources", color="red", fontweight="bold", fontsize=17,
		transform=ax.transAxes, verticalalignment="center", horizontalalignment="center")

cb = fig.colorbar(h[3], ax=ax, pad=-0.002)
cb.set_label("N. of sources", fontsize=16)
cb.ax.tick_params(labelsize=14)

if(savefig_all):
	plt.savefig("figures/selection_function_surface_luminosity.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()




######################################################################
#####                   Validation loss curve                    #####
######################################################################

#Activate if you retrained the network and generated the error file and the iter_score_list using post_process.py

if(0):
	iter_score_list = np.loadtxt("iter_score_list.txt")
	loss_iter = np.loadtxt("error.txt")

	skip = 20

	part_labels = ["Position", "Size", "Probability", "Objectness", "Parameters"]
	part_ids = [2,3,4,5,7]
	part_scaling = [1.0,20.0,1.0,0.5,1.0]
	iter_bins = [20,500,4100]

	fig, axs = plt.subplots(1,2, figsize=(10,2.5), dpi=150, constrained_layout=True, gridspec_kw={"width_ratios":[1.0,2.6]})

	axs[0].set_ylabel("Loss", fontsize=11)

	for i in range(0,2):
		axs[i].set_xlabel("Iteration", fontsize=11)
		for j in range(0,5):
			axs[i].plot(loss_iter[iter_bins[i]:iter_bins[i+1],0], 
						loss_iter[iter_bins[i]:iter_bins[i+1],part_ids[j]]*part_scaling[j], 
						label=part_labels[j]+" x %.1f"%(part_scaling[j]), lw=0.8)
		y_lims = axs[i].get_ylim(); x_lims = axs[i].get_xlim()
		y_range = y_lims[1] - y_lims[0]; x_range = x_lims[1] - x_lims[0]
		if(i != 0):
			axs[i].set_ylim(y_lims[0], y_lims[1]+y_range*0.33)
		text_pos_y = np.max(loss_iter[int(iter_bins[i]/200+1)*200,part_ids[:]]*part_scaling[:])
		for j in range(int(iter_bins[i]/200),int(iter_bins[i+1]/200)):
			pos_x = (j+1)*200; pos_y = np.max(loss_iter[(j+1)*200,part_ids[:]]*part_scaling[:])
			axs[i].annotate("%d"%(iter_score_list[j,1]), (pos_x, pos_y+0.05*y_range), (pos_x, text_pos_y+((np.clip(i,0,1)*((j+1)%2))*0.1+0.18)*y_range), 
							color = "grey", fontsize = 8, horizontalalignment="center", verticalalignment="bottom",
							arrowprops=dict(color="grey", shrink=0.05, width=0.1, headwidth=5, headlength=5))

	axs[0].legend(prop={'size': 8})
	if(savefig_all):
		plt.savefig("figures/loss_curve_with_score.pdf", dpi=300, bbox_inches='tight')
	else:
		plt.show()
	plt.close()


######################################################################
#####          Re apply post process of the prediction           #####
######################################################################


print ("Reading Network prediction ... ")
pred_data = np.fromfile("fwd_res/net0_%04d.dat"%load_iter, dtype="float32")

#Repeat corresponds to the number of MC_MODEL realization
repeat = 1
predict = np.reshape(pred_data, (repeat, nb_area_h, nb_area_w, nb_box*(8+nb_param),yolo_nb_reg,yolo_nb_reg))
#Only keep the mean, but any realization statistic can be computed here
predict = np.mean(predict, axis=0)

print (np.shape(predict))
print ("1st order predictions filtering ... ")

final_boxes = []
c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
c_tile_kept = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")
c_box = np.zeros((6+1+nb_param+1),dtype="float32")
patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")

box_count_per_reg_hist = np.zeros((nb_box+1), dtype="int")

cumul_filter_box = 0; cumul_post_nms = 0

for ph in tqdm(range(0,nb_area_h)):
	for pw in range(0, nb_area_w):
		
		c_tile[:,:] = 0.0; c_tile_kept[:,:] = 0.0
				
		p_x = pw; p_y = ph
		
		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + fwd_image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + fwd_image_size - orig_offset

		px_min = 0; px_max = fwd_image_size
		py_min = 0; py_max = fwd_image_size

		if(ph == 0 or ph == nb_area_h-1 or pw == 0 or pw == nb_area_w-1):
			patch[:,:] = 0.0
		else:
			patch[:,:] = full_data_norm[ymin:ymax,xmin:xmax]

		c_pred = predict[ph,pw,:,:,:]
		c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, patch, val_med_lims, val_med_obj, box_count_per_reg_hist)			
		
		cumul_filter_box += c_nb_box
		c_nb_box_final = c_nb_box
		c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, first_nms_thresholds, first_nms_obj_thresholds)
		cumul_post_nms += c_nb_box_final
		
		#Manually set all edge predictions to 0
		#The network was not train to interpret these cases which results in artifacts
		out_range = 2
		if(ph < out_range or ph >= nb_area_h-out_range or pw < out_range or pw >= nb_area_w-out_range):
			c_nb_box_final = 0
		
		final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (nb_area_h, nb_area_w))

flat_kept = np.vstack(final_boxes.flatten())

print ("NMS removed average frac:", (cumul_filter_box-cumul_post_nms)/cumul_filter_box)


######################################################################
#####           Per patch source density distribution            #####
######################################################################


density_map = np.zeros((nb_area_h,nb_area_w))

for i in range(0,nb_area_h):
	for j in range(0,nb_area_w):
		density_map[i,j] = np.shape(final_boxes[i,j])[0]

print ("Total Nb. prediction kept: ", np.sum(density_map[:,:]))

plt.figure(figsize=(8,6), dpi=150, constrained_layout=True)
ax = plt.gca()
plt.imshow(density_map.T, origin="lower", vmax=280*(fwd_image_size*fwd_image_size)/(256*256), cmap="gist_earth")
ax.set_xlabel("X pred. grid", fontsize=15)
ax.set_ylabel("Y pred. grid", fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
cbar = plt.colorbar(extend='max')
cbar.set_label("N. of detections", fontsize=15)
ax.set_aspect(1)
if(savefig_all):
	plt.savefig("figures/pred_grid_density_map.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()


######################################################################
#####    Histogram of the number of predictions per grid cell    #####
######################################################################


nb_pred = np.zeros((nb_box))

for i in range(0,nb_area_h):
	for j in range(0,nb_area_w):
		if(np.shape(final_boxes[i,j])[0] != 0):
			u_b, u_n = np.unique(final_boxes[i,j][:,-1], return_counts=True)
			for k in u_n:
				nb_pred[k-1] += 1

plt.figure(figsize=(8,5), dpi=150, constrained_layout=True)
plt.bar(np.linspace(1,nb_box,nb_box), nb_pred, width=0.5, color="C0")
plt.yscale("log")
plt.ylim(1,np.max(nb_pred)*4.0)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(np.linspace(1,nb_box,nb_box))
plt.xlabel("N. of predictions per cell", fontsize=14)
plt.ylabel("N. of cells", fontsize=14)

for i in range(0,nb_box):
	if(nb_pred[i] != 0):
		plt.text(i+1, nb_pred[i], "%d"%(nb_pred[i]), fontsize=10, fontweight="bold", horizontalalignment="center",verticalalignment="bottom")

if(savefig_all):
	plt.savefig("figures/pred_per_reg_dist.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()

print("Avg:", np.sum(nb_pred*np.linspace(1,nb_box,nb_box))/np.sum(nb_pred))
print (nb_pred[:])
print (np.cumsum(nb_pred))
print (np.sum(nb_pred))
print("Sum:", np.sum(nb_pred))

l_cum_sum = np.zeros((nb_box))
l_cum_sum[1:] = np.cumsum(nb_pred)[:-1]

prop_sup_content = (np.sum(nb_pred)-l_cum_sum)/np.sum(nb_pred)*100
print(prop_sup_content)


######################################################################
#####     Distribution of the detection per detection unit       #####
######################################################################


plt.figure(figsize=(8,5), dpi=150, constrained_layout=True)
n, bins, pat = plt.hist(flat_kept[:,6], bins=np.linspace(-0.5,nb_box-0.5,nb_box+1), width = 0.5, color="C0")

str_labels = np.char.add("Id. ", np.array(range(0,nb_box)).astype("str"))
str_labels = np.char.add(str_labels," - [")
str_labels = np.char.add(str_labels,(prior_size[0,:].astype(int)).astype("str"))
str_labels = np.char.add(str_labels,",")
str_labels = np.char.add(str_labels,(prior_size[1,:].astype(int)).astype("str"))
str_labels = np.char.add(str_labels,"]")
plt.xticks(np.linspace(0,nb_box-1,nb_box)-0.25, str_labels, fontsize=14, rotation = 60)
plt.ylabel("N. of predictions", fontsize=14)
plt.xlim(-0.75, nb_box - 0.75)
plt.ylim(0,np.max(n)*1.15)
plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useOffset=False)
for i in range(0,nb_box):
	plt.text(i - 0.25, n[i] + np.max(n)*0.01, "%d"%(n[i]), fontsize=10, fontweight="bold", verticalalignment="bottom", horizontalalignment="center")

if(savefig_all):
	plt.savefig("figures/pred_prior_size_dist.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()
	


######################################################################
###  Distribution of the pred. size and aspect ratio per det-unit  ###
######################################################################


fig, axs = plt.subplots(nb_box, 2, figsize=(12,1.75*nb_box), dpi=70, constrained_layout=True, gridspec_kw={"width_ratios":[3,1]})
fig.set_constrained_layout_pads(hspace=0)

for i in range(0, nb_box):
	id_box = np.where(flat_kept[:,6] == i)[0] 
	select_w = flat_kept[id_box,2] - flat_kept[id_box,0]
	select_h = flat_kept[id_box,3] - flat_kept[id_box,1]

	vmin = box_clip[0]
	vmax = box_clip[1]
	bins = 10**(np.linspace(np.log10(vmin),np.log10(vmax),50))

	# SIZE DIST HISTOGRAM
	n, n_bins, patch1 = axs[i,0].hist(select_w, bins=bins, 
		histtype="step", ls="-", color="C%d"%(i), lw = 1.5, label="Width")
	n2, n_bins2, patch2 = axs[i,0].hist(select_h, bins=bins, 
		histtype="step", ls="--", color="C%d"%(i), lw=1.5, label="Height")
	axs[i,0].plot([prior_size[1,i],prior_size[1,i]],[0,1e4], color="red", ls=":", lw=1.5)
	if(prior_size[0,i] != prior_size[1,i]):
		axs[i,0].plot([prior_size[0,i],prior_size[0,i]],[0,1e4], color="red", ls="-", lw=1.5)

	axs[i,0].set_xscale('log'); axs[i,0].set_yscale('log')
	axs[i,0].set_ylim(None,10**5); axs[i,0].set_xlim(4.9,80)

	axs[i,0].set_yticks([1e1,1e3,1e5])
	axs[i,0].set_yticks([1e2,1e4], minor=True)

	axs[i,0].tick_params(axis='x', which='major')
	axs[i,0].tick_params(axis='y', which='major', labelsize=11)
	axs[i,0].tick_params(axis='y', which='minor', labelleft=False)

	axs[i,0].grid(axis="y", which="both", ls="--")
	axs[i,0].text(0.95, 0.8,"Id. %d"%(i), color="C%d"%(i), verticalalignment="center", fontsize=15,
		horizontalalignment="center", fontweight="bold", bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[i,0].transAxes)
	axs[i,0].text(0.05, 0.95,"Total: %d"%(np.shape(select_w)[0]), color="grey", verticalalignment="top", fontsize=12,
		horizontalalignment="left", transform=axs[i,0].transAxes)

	# ASPECT RATIO DIST HISTOGRAM
	select_aspect_ratio = (select_w-select_h)/np.maximum(select_w,select_h)
	axs[i,1].hist(select_aspect_ratio, bins=30, histtype="step", color="C%d"%(i), lw=1.5)

	axs[i,1].set_yscale('log')
	axs[i,1].set_xlim(-1.0,1.0); axs[i,1].set_ylim(None,10**5)
	axs[i,1].set_yticks([1e1,1e3,1e5])
	axs[i,1].set_yticks([1e2,1e4], minor=True)

	axs[i,1].grid(axis="y", which="both", ls="--")
	axs[i,1].tick_params(axis='both', which='both', labelleft=False)
	if(i != nb_box-1):
		axs[i,0].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
		axs[i,1].tick_params(axis='x', which='both', bottom=True, labelbottom=False)

	id_pos = np.where(select_aspect_ratio > 0)[0]
	id_neg = np.where(select_aspect_ratio < 0)[0]
	axs[i,1].text(0.05,0.95,"%3.1f%% -"%(np.shape(id_neg)[0]/np.shape(select_w)[0]*100), color="grey", fontweight="bold",
		transform=axs[i,1].transAxes, verticalalignment="top", horizontalalignment="left")
	axs[i,1].text(0.95,0.95,"%3.1f%% +"%(np.shape(id_pos)[0]/np.shape(select_w)[0]*100), color="grey", fontweight="bold",
		transform=axs[i,1].transAxes, verticalalignment="top", horizontalalignment="right")

# LEGEND
axs[-1,0].tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=11)
axs[-1,1].tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=11)
axs[-1,0].set_xlabel("Size [pix]", fontsize=14)
axs[-1,1].set_xlabel("Aspect ratio", fontsize=14)

plt.gcf().text(-0.03,0.5, "N. of predictions", va="center", rotation="vertical", fontsize=14)
handles = [patch1[0],patch2[0]]

leg = axs[0,0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,1.6), 
		fancybox=False, shadow=True, ncol=2, fontsize=18, markerscale=2.0)
leg.get_frame().set_linewidth(1.8)
leg.get_frame().set_edgecolor("black")
leg.legendHandles[0].set_color("black")
leg.legendHandles[1].set_color("black")

if(savefig_all):
	plt.savefig("figures/pred_dist_size_per_prior.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()


######################################################################
#####                 Load the full True catalog                 #####
######################################################################

print ("Loading GroundTruth from TrainingSet_perscut ... ")

if(not os.path.isfile(data_path+"True_560_perscut.txt")):
	dataset_perscut(data_path+"True_560_v2.txt",data_path+"True_560_perscut.txt", 0)
target_list = np.loadtxt(data_path+"True_560_perscut.txt", skiprows=0)

#Get the sky coordinate of all sources in the selected training catalog
c = SkyCoord(ra=target_list[:,1] * u.deg, dec=target_list[:,2] * u.deg, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)

#Compute the bmaj and bmin in pixel size (approximate)
#Only used to define the bouding boxes
w = target_list[:,7]/(3600.0*pixel_size)*2
h = target_list[:,8]/(3600.0*pixel_size)*2

#Get the value of the beam for each source position as well
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")

n_w = np.zeros((np.shape(target_list)[0]))
n_h = np.zeros((np.shape(target_list)[0]))
coords = np.zeros((np.shape(target_list)[0],4))
flux_list = np.zeros((np.shape(target_list)[0]))
bmaj_list = np.zeros((np.shape(target_list)[0]))
bmin_list = np.zeros((np.shape(target_list)[0]))
pa_list   = np.zeros((np.shape(target_list)[0]))

#Convert all bmaj, bmin size onto regular "boxes" as defined for classical detection task
#The new box must include the source as defined by Bmaj and Bmin
for i in range(0,np.shape(target_list)[0]):
	W = w[i]
	H = h[i]
	vertices = np.array([[-W*0.5,-H*0.5],[-W*0.5,H*0.5],[W*0.5, -H*0.5],[W*0.5,H*0.5]])

	vertices_new = np.zeros((4,2))
	vertices_new[:,0] = np.cos(target_list[i,9]*np.pi/180.0)*vertices[:,0] + np.sin(target_list[i,9]*np.pi/180.0)*vertices[:,1]
	vertices_new[:,1] = - np.sin(target_list[i,9]*np.pi/180.0)*vertices[:,0] + np.cos(target_list[i,9]*np.pi/180.0)*vertices[:,1]

	n_w[i] = max(vertices_new[:,0]) - min(vertices_new[:,0])
	n_h[i] = max(vertices_new[:,1]) - min(vertices_new[:,1])

#Clip the too small boxes (in pixel size)
n_w = np.clip(n_w, box_clip[0], box_clip[1])
n_h = np.clip(n_h, box_clip[0], box_clip[1])

#Convert the "local" vertice coordinates in edges coordinates using the full image size
coords[:,0] = x - n_w[:]*0.5
coords[:,1] = x + n_w[:]*0.5
coords[:,2] = y - n_h[:]*0.5
coords[:,3] = y + n_h[:]*0.5

flux_list[:] = target_list[:,5]
bmaj_list[:] = target_list[:,7]
bmin_list[:] = target_list[:,8]
pa_list[:]   = target_list[:,9]

#Remap all the PA values so they are all in the range [-90,90]
index = np.where((pa_list[:] > 90.0) & (pa_list[:] <= 270.0))
pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 90.0)
index = np.where((pa_list[:] > 270.0) & (pa_list[:] < 360.0))
pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 270.0)

#Get the "apparent flux" to correspond to the visible flux in the beam convolved map
flux_list[:] = flux_list[:]*beamval[:]

target_select_boxes = np.array([target_list[:,1],target_list[:,2],n_w,n_h,flux_list,bmaj_list,bmin_list,target_list[:,0]]).T

print(" Done!")



######################################################################
#####   Distriubtion of the targets over the closest size-prior  #####
######################################################################

prior_types = np.unique(prior_size, axis=1)
nb_prior_types = np.shape(prior_types)[1]

dist = np.sqrt(np.subtract(np.tile(n_w[:],(nb_prior_types,1)).T,prior_types[0,:])**2
			   +np.subtract(np.tile(n_h[:],(nb_prior_types,1)).T,prior_types[1,:])**2)
best_prior = np.argmin(dist, axis=1)

plt.figure(figsize=(8,2.5), dpi=150)
n, bins, pat = plt.hist(best_prior, color="green", alpha=0.8, bins=np.linspace(-0.5,nb_prior_types-0.5,nb_prior_types+1), height=0.5, orientation='horizontal')

str_labels = np.char.add("[",(prior_types[0,:].astype(int)).astype("str"))
str_labels = np.char.add(str_labels,",")
str_labels = np.char.add(str_labels,(prior_types[1,:].astype(int)).astype("str"))
str_labels = np.char.add(str_labels,"]")
plt.yticks(np.linspace(0,nb_prior_types-1,nb_prior_types)-0.25, str_labels, rotation = 45)
plt.xlabel("N. of targets", fontsize=14)
plt.ylim(-0.75, nb_prior_types - 0.75); plt.xlim(1,np.max(n)*1.15)
plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 4), useOffset=False)

for i in range(0,nb_prior_types):
	plt.text(n[i]+ np.max(n)*0.01, i - 0.25 , "%d"%(n[i]), fontsize=10, fontweight="bold", horizontalalignment="left",verticalalignment="center")
if(savefig_all):
	plt.savefig("figures/targ_prior_size_dist.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()



######################################################################
#####                Inter-patch NMS filtering                   #####
######################################################################

print ("2nd order predictions filtering ... ")

overlap = fwd_image_size - patch_shift

c_tile = np.zeros((yolo_nb_reg*yolo_nb_reg*nb_box,(6+1+nb_param+1)),dtype="float32")

dir_array = np.array([[-1,0],[+1,0],[0,-1],[0,+1],[-1,+1],[+1,+1],[-1,-1],[+1,-1]])

#Second NMS over all the overlapping patches
for ph in range(0, nb_area_h):
	for pw in range(0, nb_area_w):
		boxes = np.copy(final_boxes[ph,pw])
		for l in range(0,8):
			if(ph+dir_array[l,1] >= 0 and ph+dir_array[l,1] <= nb_area_h-1 and				pw+dir_array[l,0] >= 0 and pw+dir_array[l,0] <= nb_area_w-1):
				comp_boxes = np.copy(final_boxes[ph+dir_array[l,1],pw+dir_array[l,0]])
				c_nb_box = second_NMS_local(boxes, comp_boxes, c_tile, dir_array[l], second_nms_threshold)
				boxes = np.copy(c_tile[0:c_nb_box,:])

		final_boxes[ph,pw] = np.copy(boxes)

flat_kept = np.vstack(final_boxes.flatten())

print ("Total N. of predictions kept: ", np.shape(flat_kept)[0])




######################################################################
#####   Convert predictions to source catalog in scorer format   #####
######################################################################

#Order predictions by objectness score and convert to SDC scorer format

final_boxes_scaled = np.copy(final_boxes)
for p_h in range(0, nb_area_h):
	box_h_offset = p_h*patch_shift - orig_offset
	for p_w in range(0, nb_area_w):
		box_w_offset = p_w*patch_shift - orig_offset
		final_boxes_scaled[p_h,p_w][:,0] = box_w_offset + final_boxes_scaled[p_h,p_w][:,0]
		final_boxes_scaled[p_h,p_w][:,2] = box_w_offset + final_boxes_scaled[p_h,p_w][:,2]
		final_boxes_scaled[p_h,p_w][:,1] = box_h_offset + final_boxes_scaled[p_h,p_w][:,1]
		final_boxes_scaled[p_h,p_w][:,3] = box_h_offset + final_boxes_scaled[p_h,p_w][:,3]

flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,5].argsort(),:][::-1]

x_y_flat_kept = np.copy(flat_kept_scaled[:,0:2])
x_y_flat_kept[:,0] = np.clip((flat_kept_scaled[:,0]+flat_kept_scaled[:,2])*0.5 - 0.5, 0, map_pixel_size)
x_y_flat_kept[:,1] = np.clip((flat_kept_scaled[:,1]+flat_kept_scaled[:,3])*0.5 - 0.5, 0, map_pixel_size)

if(training_only):
	training_area_id = np.where((x_y_flat_kept[:,0] > min_ra_train_pix) & (x_y_flat_kept[:,0] < max_ra_train_pix) &	
								(x_y_flat_kept[:,1] > min_dec_train_pix) & (x_y_flat_kept[:,1] < max_dec_train_pix))[0]

	flat_kept_scaled = flat_kept_scaled[training_area_id]
	x_y_flat_kept = x_y_flat_kept[training_area_id]

cls = utils.pixel_to_skycoord(x_y_flat_kept[:,0], x_y_flat_kept[:,1], wcs_img)
ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])
w, h = flat_kept_scaled[:,2]-flat_kept_scaled[:,0], flat_kept_scaled[:,3]-flat_kept_scaled[:,1]

catalog_size = np.shape(flat_kept_scaled)[0]
cat_header = "RA(deg), DEC(deg), X(pix), Y(pix), W(pix), H(pix), Objectness(real), Apparent Flux(Jy), BMAJ(arcsec), BMIN(arcsec), PA(degree)"
box_catalog = np.zeros((catalog_size,11), dtype="float32")

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

xbeam, ybeam = utils.skycoord_to_pixel(coords, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")

FLUX_JY = box_catalog[:,6]/beamval

empty = np.zeros((np.shape(coords.ra.deg)[0]))
print (np.arange(0,np.shape(box_catalog)[0]))
scoring_table = np.vstack((np.arange(0,np.shape(box_catalog)[0]), ra, dec, ra, dec,	
						FLUX_JY, empty+0.0375, box_catalog[:,7], box_catalog[:,8],	
						box_catalog[:,9], empty+2.0, empty+3.0))

np.savetxt(data_path+"YOLO_CIANNA_sdc1_catalog.txt", scoring_table.T, fmt="%d %1.8f %2.8f %1.8f %2.8f %g %0.8f %f %f %f %d %d")



######################################################################
#####                 Score a source catalog                     #####
######################################################################

#Start the script here with other sub_cat_path to produce analysis figures for other reference detection calogs
#The participating team catalogs can be found at : https://owncloud.ia2.inaf.it/index.php/s/xdeXBymC32og1ol/download
#The catalog of the JLRAT2 team from Yu et al. 2022 can be found at : https://github.com/MoerAttempts/JLRAT-SKA-SDC1

sub_cat_path = data_path+"YOLO_CIANNA_sdc1_catalog.txt"; c_mode=0; skip_cat = 0 #480k
#sub_cat_path = data_path+"SDC1_submissions/EngageSKA_560MHz_1000h_v2.txt"; c_mode=0; skip_cat = 1 #200k
#sub_cat_path = data_path+"SDC1_submissions/Shanghai_560MHz_1000h_v5.txt"; c_mode=1; skip_cat = 1 #159k
#sub_cat_path = data_path+"SDC1_submissions/ICRAR_560MHz_1000h_v16.txt"; c_mode=1; skip_cat = 1 #142k

#sub_cat_path = data_path+"JLRAT_560_1000h_submit_vf2.txt" #298k #core skip

truth_cat_path = data_path+"True_560_v2.txt"
skip_cat = 0
skip_truth = 0

scorer = Sdc1Scorer.from_txt(sub_cat_path, truth_cat_path, freq=560, sub_skiprows=skip_cat, truth_skiprows=skip_truth)

scorer.run(mode=0, # 0, 1 for core, centroid position modes respectively
	train=training_only, # True to score based on training area only, else exclude
	detail=True, # True to return per-source scores and match catalogue
	)

print("Final score: {}".format(scorer.score.value))

print ("Ndet:", scorer.score.n_det)
print ("Nmatch:", scorer.score.n_match)
print ("Nbad:", scorer.score.n_bad)
print ("Nfalse:", scorer.score.n_false)
print ("Score det:",scorer.score.score_det)
print ("Average match score:",scorer.score.acc_pc)

"""
id, ra_core, dec_core, ra_cent, dec_cent, flux, core_frac, b_maj, b_min, pa, size_id, class, a_flux, conv_size, id_t, ra_core_t, dec_core_t
ra_cent_t, dec_cent_t, flux_t, core_frac_t, b_maj_t, b_min_t, pa_t, size_id_t, class_t, a_flux_t, conv_size_t, multi_d_err
"""

matched = scorer.score.match_df
scores_df = scorer.score.scores_df

cat_dat = np.loadtxt(sub_cat_path, skiprows=skip_cat)
truth_dat = np.loadtxt(truth_cat_path, skiprows=0)

print (np.shape(truth_dat))
index = np.where((truth_dat[:,1] < ra_max) & (truth_dat[:,1] > ra_min) &
				 (truth_dat[:,2] < dec_max) & (truth_dat[:,2] > dec_min))

truth_dat = np.delete(truth_dat,index,axis=0)
full_truth_cat_size = np.shape(truth_dat)[0]
print (np.shape(truth_dat))

print("Avg. pos score:",np.mean(scores_df["position"]))
print("Avg. bmaj score:",np.mean(scores_df["b_maj"]))
print("Avg. bmin score:",np.mean(scores_df["b_min"]))
print("Avg. flux score:",np.mean(scores_df["flux"]))
print("Avg. PA score:",np.mean(scores_df["pa"]))
print("Avg. cf score:",np.mean(scores_df["core_frac"]))
print("Avg. class score:",np.mean(scores_df["class"]))

print("Precision / Reliability:", scorer.score.n_match/scorer.score.n_det)
print("(With bad) Precision / Reliability:", (scorer.score.n_match+scorer.score.n_bad)/scorer.score.n_det)


"""
Final score: 479758.3633303649
Ndet: 718760
Nmatch: 677025
Nbad: 15787
Nfalse: 41735
Score det: 521493.3633303649
Average match score: 77.02719446554632
(5251292, 12)
(5060741, 12)
Avg. pos score: 0.9641746456033577
Avg. bmaj score: 0.6267978830542554
Avg. bmin score: 0.5939226150247435
Avg. flux score: 0.6460584446067026
Avg. PA score: 0.6014046299807417
Avg. cf score: 0.9864277103590732
Avg. class score: 0.9731176839850818
Precision / Reliability: 0.9419347209082308
(With bad) Precision / Reliability: 0.963898937058267

"""



######################################################################
#####         Flux distribution of the detected sources          #####
######################################################################


print("Loading catalogs for flux distribution display")

match = scorer.score.match_df
bins = 10**(np.linspace(-9,2,60))
false = np.delete(cat_dat, match["id"], axis=0)

fig, axs = plt.subplots(2,1, figsize=(8,6), dpi=120, gridspec_kw={'height_ratios': [1, 0.5]}, constrained_layout=True, sharex="all")

rwidth = 0.88

c = SkyCoord(ra=truth_dat[:,1]*u.degree, dec=truth_dat[:,2]*u.degree, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")
flux_beam_truth = truth_dat[:,5]*beamval

n, bins, patches1 = axs[0].hist(flux_beam_truth, bins=bins, histtype="bar", color = "grey", rwidth = rwidth, 
		log=True, label="True cat.", edgecolor='white', linewidth=0.0, rasterized=True)

c = SkyCoord(ra=match["ra_core_t"].values*u.degree, dec=match["dec_core_t"].values*u.degree, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")
flux_beam_match = match["flux_t"].values*beamval

c = SkyCoord(ra=false[:,1]*u.degree, dec=false[:,2]*u.degree, frame='icrs')
x, y = utils.skycoord_to_pixel(c, wcs_img)
xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
new_data_beam = np.nan_to_num(data_beam)
beamval = interpn((np.arange(0,np.shape(data_beam)[0]), np.arange(0,np.shape(data_beam)[1])), 
		new_data_beam, (ybeam, xbeam), method="splinef2d")
flux_beam_false = false[:,5]*beamval

n2, bins2, patches2 = axs[0].hist(np.append(flux_beam_false, flux_beam_match), bins=bins, histtype="bar", 
		color = "gold", alpha=0.85, rwidth = rwidth, log=True, label="True+False det.", edgecolor='white', linewidth=0.0, rasterized=True)

n3, bins3, patches3 = axs[0].hist(flux_beam_match, bins=bins, histtype="bar", color = "lightgrey", rwidth = rwidth, 
		log=True, label="True det.", edgecolor='white', linewidth=0.0, rasterized=True)

axs[0].set_xscale('log')
axs[0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

axs[0].set_xlim(1e-9,1e1)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_ylabel("N. of objects", fontsize=15)
axs[0].legend(prop={'size': 13}, loc="upper right")

axs[1].step((bins[:-1]+bins[1:])*0.5, n3/n2, where="mid", label="Purity")
axs[1].step((bins[:-1]+bins[1:])*0.5, n3/n, where="mid", label="Completness")
axs[1].set_xscale('log')
axs[1].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[1].set_xlim(1e-9,1e1)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend(prop={'size': 13}, loc="upper left")


axs[1].set_xlabel("Apparent flux [Jy]", fontsize=15)

if(savefig_all):
	plt.savefig("figures/apparent_flux_distribution_full_false_true.pdf", dpi=400, bbox_inches='tight')
else:
	plt.show()
plt.close()



print("Loading catalogs for flux distribution display")

match = scorer.score.match_df
bins = 10**(np.linspace(-9,2,60))
false = np.delete(cat_dat, match["id"], axis=0)

fig, axs = plt.subplots(2,1, figsize=(8,6), dpi=120, gridspec_kw={'height_ratios': [1, 0.5]}, constrained_layout=True, sharex="all")

rwidth = 0.88
n, bins, patches1 = axs[0].hist(truth_dat[:,5], bins=bins, histtype="bar", color = "grey", rwidth = rwidth, 
		log=True, label="True cat.", edgecolor='white', linewidth=0.0, rasterized=True)
n2, bins2, patches2 = axs[0].hist(np.append(false[:,5], match["flux_t"]), bins=bins, histtype="bar", 
		color = "gold", alpha=0.85, rwidth = rwidth, log=True, label="True+False det.", edgecolor='white', linewidth=0.0, rasterized=True)
n3, bins3, patches3 = axs[0].hist(match["flux_t"], bins=bins, histtype="bar", color = "lightgrey", rwidth = rwidth, 
		log=True, label="True det.", edgecolor='white', linewidth=0.0, rasterized=True)

axs[0].set_xscale('log')
axs[0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

axs[0].set_xlim(1e-9,1e1)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_ylabel("N. of objects", fontsize=15)
axs[0].legend(prop={'size': 13}, loc="upper right")

axs[1].step((bins[:-1]+bins[1:])*0.5, n3/n2, where="mid", label="Purity")
axs[1].step((bins[:-1]+bins[1:])*0.5, n3/n, where="mid", label="Completness")
axs[1].set_xscale('log')
axs[1].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[1].set_xlim(1e-9,1e1)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend(prop={'size': 13}, loc="upper left")



axs[1].set_xlabel("Flux [Jy]", fontsize=15)

if(savefig_all):
	plt.savefig("figures/flux_distribution_full_false_true.pdf", dpi=400, bbox_inches='tight')
else:
	plt.show()
plt.close()




######################################################################
#####           Detected sources over examples fields            #####
######################################################################

nb_fields = 3
fields = np.array([[0.1,-30.2],
				   [1.22,-29.60],
				   [-1.74,-30.14]])

def sim_360(val):
	tab = np.copy(val)
	for i in range(0,len(tab)):
		if (tab[i] > 180):
			tab[i] = tab[i]-360.0
	return tab

def sim_360_rev(val):
	tab = np.copy(val)
	for i in range(0,len(tab)):
		if (tab[i] < 0):
			tab[i] = 360.0 + tab[i]
	return tab

cutout_size = 256
deg_size = cutout_size*1.2*pixel_size

id_loc_mask = np.isin(target_select_boxes[:,-1], matched.id_t.to_numpy())
id_loc_found = np.where(id_loc_mask)

target_missed = np.delete(target_select_boxes, id_loc_found, axis=0)
id_loc_mask = np.isin(matched.id_t.to_numpy(), target_select_boxes[:,-1])

id_loc_found = np.where(id_loc_mask == True)[0]
id_loc_not_found = np.where(id_loc_mask == False)[0]

flagged_boxes = np.zeros((np.shape(box_catalog)[0],np.shape(box_catalog)[1]+1))
flagged_boxes[:,:-1] = box_catalog[:,:]

flagged_boxes[matched.id.to_numpy()[id_loc_found],-1] = 1
flagged_boxes[matched.id.to_numpy()[id_loc_not_found],-1] = 2

fig, axs = plt.subplots(nb_fields, 3, figsize=(15.2,5*nb_fields), dpi=60, 
	gridspec_kw={'width_ratios': [1, 1, 1]}, constrained_layout=True)

for run_f in range(0,nb_fields):

	print("Coords: ra: %f (deg) dec: %f (deg)", fields[run_f,0], fields[run_f,1])
	patch = Cutout2D(full_data_norm, SkyCoord(ra=fields[run_f,0]*u.deg, dec=fields[run_f,1]*u.deg, frame="icrs"), 
					 (cutout_size,cutout_size), wcs_img.celestial)

	for i in range(0,3):
		gcf = axs[run_f,i].imshow(patch.data, cmap="hot", vmax = 0.5*1.0, vmin = 0)
		
		axs[run_f,i].set_xlim(-0.5,cutout_size-0.5)
		axs[run_f,i].set_ylim(-0.5,cutout_size-0.5)
		
		axs[run_f,i].tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
		axs[run_f,i].tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
		
	missed_kept_id = np.where((np.abs(target_missed[:,0] - fields[run_f,0]) < deg_size*0.5) &
			   (np.abs(target_missed[:,1] - fields[run_f,1]) < deg_size*0.5))[0]
	target_missed_kept = target_missed[missed_kept_id,:]

	full_truth_kept_id =  np.where((np.abs(truth_dat[:,1] - fields[run_f,0]) < deg_size*0.5) &
			   (np.abs(truth_dat[:,2] - fields[run_f,1]) < deg_size*0.5))[0]
	
	print (np.min(flagged_boxes[:,0]), np.max(flagged_boxes[:,0]))
	kept_boxes_id = np.where((np.abs(sim_360(flagged_boxes[:,0]) - fields[run_f,0]) < deg_size*0.5) &
				(np.abs(flagged_boxes[:,1] - fields[run_f,1] < deg_size*0.5)))[0]
	kept_boxes_pred = flagged_boxes[kept_boxes_id,:]
	print (np.shape(kept_boxes_pred))
	
	px, py = utils.skycoord_to_pixel(SkyCoord(sim_360_rev(truth_dat[full_truth_kept_id,1])*u.deg,
				truth_dat[full_truth_kept_id,2]*u.deg, frame="icrs"), wcs=patch.wcs.celestial)
	
	if(np.shape(px)[0] > 0):
		axs[run_f,0].scatter(px, py, c="green", s=4.0,lw=0.6, 
					   marker="x", alpha=1.0, label="All True")
		axs[run_f,0].set_title("Field %d"%(run_f+1), fontweight="bold", fontsize=16,
								rotation="vertical",x=-0.15,y=0.5, verticalalignment="center", horizontalalignment="center")
		axs[run_f,0].text(-16,cutout_size*0.5, "RA=%0.2f, Dec=%0.2f [deg]"%(fields[run_f,0], fields[run_f,1]), fontsize=14,
						  rotation="vertical", verticalalignment="center", horizontalalignment="center")
	
	px, py = utils.skycoord_to_pixel(SkyCoord(sim_360_rev(kept_boxes_pred[:,0])*u.deg,kept_boxes_pred[:,1]*u.deg, frame="icrs"), wcs=patch.wcs.celestial)

	for i in range(0, np.shape(kept_boxes_pred)[0]):

		xmin = px[i] - kept_boxes_pred[i,2]*0.5
		ymin = py[i] - kept_boxes_pred[i,3]*0.5
		xmax = px[i] + kept_boxes_pred[i,2]*0.5
		ymax = py[i] + kept_boxes_pred[i,3]*0.5
		
		if(kept_boxes_pred[i,-1] == 1):
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.7, ls="-", fill=False, color="dodgerblue", zorder=3)
			c_patch = axs[run_f,1].add_patch(el)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.8, foreground='black'),
					   path_effects.Normal()])
		elif(kept_boxes_pred[i,-1] == 2):
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.7, ls="-", fill=False, color="aqua", zorder=3)
			c_patch = axs[run_f,1].add_patch(el)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.8, foreground='black'),
					   path_effects.Normal()])
		else:
			el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.7, ls="-", fill=False, color="violet", zorder=3)
			c_patch = axs[run_f,2].add_patch(el)
			c_patch.set_path_effects([path_effects.Stroke(linewidth=1.8, foreground='black'),
					   path_effects.Normal()])
	
	px, py = utils.skycoord_to_pixel(SkyCoord(target_missed_kept[:,0]*u.deg,target_missed_kept[:,1]*u.deg, frame="icrs"), wcs=patch.wcs.celestial)

	for i in range(0, np.shape(target_missed_kept)[0]):

		xmin = px[i] - target_missed_kept[i,2]*0.5
		ymin = py[i] - target_missed_kept[i,3]*0.5
		xmax = px[i] + target_missed_kept[i,2]*0.5
		ymax = py[i] + target_missed_kept[i,3]*0.5

		el = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=0.8, ls="-", fill=False, color="lightgreen", zorder=2)
		c_patch = axs[run_f,2].add_patch(el)
		c_patch.set_path_effects([path_effects.Stroke(linewidth=1.6, foreground='black'),
					   path_effects.Normal()])
	

	if(run_f == 0):
		sc_handle, sc_label = axs[run_f,0].get_legend_handles_labels()

		match_in_label = patches.Patch(color="dodgerblue", label="Match In", linewidth=2.0, fill=False)
		match_in_label.set_path_effects([path_effects.Stroke(linewidth=4.0, foreground='black'),
					   path_effects.Normal()])

		match_out_label = patches.Patch(color="aqua", label="Match Out", linewidth=2.0, fill=False)
		match_out_label.set_path_effects([path_effects.Stroke(linewidth=4.0, foreground='black'),
					   path_effects.Normal()])

		match_false_label = patches.Patch(color="violet", label="Bad / False", linewidth=2.0, fill=False)
		match_false_label.set_path_effects([path_effects.Stroke(linewidth=4.0, foreground='black'),
					   path_effects.Normal()])

		match_missed_label = patches.Patch(color="lightgreen", label="Missed In", linewidth=2.0, fill=False)
		match_missed_label.set_path_effects([path_effects.Stroke(linewidth=4.0, foreground='black'),
					   path_effects.Normal()])

		leg1 = axs[run_f,0].legend(handles=sc_handle, loc="upper center", bbox_to_anchor=(0.5,1.18), 
			fancybox=False, shadow=True, ncol=1, fontsize=17, markerscale=2.0)
		leg1.legendHandles[0]._sizes = [100]
		leg1.legendHandles[0].set_linewidth(2.0)

		leg2 = axs[run_f,1].legend(handles=[match_in_label,match_out_label], loc="upper center", bbox_to_anchor=(0.5,1.18), 
			fancybox=False, shadow=True, ncol=2, fontsize=17, markerscale=2.0)
		leg3 = axs[run_f,2].legend(handles=[match_false_label,match_missed_label], loc="upper center", bbox_to_anchor=(0.5,1.18), 
			fancybox=False, shadow=True, ncol=2, fontsize=17, markerscale=2.0)

		leg1.get_frame().set_linewidth(1.8)
		leg1.get_frame().set_edgecolor("black")
		leg2.get_frame().set_linewidth(1.8)
		leg2.get_frame().set_edgecolor("black")
		leg3.get_frame().set_linewidth(1.8)
		leg3.get_frame().set_edgecolor("black")

if(savefig_all):
	plt.savefig("figures/expl_det_field.pdf", dpi=350, bbox_inches='tight')
else:
	plt.show()
plt.close()



######################################################################
#####          2D histogram of characterization errors           #####
######################################################################

lims = np.loadtxt("train_cat_norm_lims.txt")

l_lw = 1.5
l_alpha = 0.2
rel_hist_nb_bins=60

cmap = truncate_colormap("terrain_r",0.08,1.0)
fig, axs = plt.subplots(5,4, figsize=(16,20), dpi=50, gridspec_kw={'width_ratios': [3, 0.05, 1,0.5]}, constrained_layout=True, sharey="row")
fig.set_constrained_layout_pads(hspace=0.05)

for i in range(0,5):
	axs[i,1].set_visible(False)

######### POSITION #########

rel_err_lims = [0.0,2.5]
rel_err_offset = 0.25
param_lims = [0.0,2.2]
err_threshold = 0.3

# Calculate coordinate separation array
coord_sub = SkyCoord(ra=matched.ra_cent, dec=matched.dec_cent, frame="fk5", unit="deg")
coord_truth = SkyCoord(ra=matched.ra_cent_t, dec=matched.dec_cent_t, frame="fk5", unit="deg")
sep_arr = coord_truth.separation(coord_sub)

source_size = (matched.b_maj_t + matched.b_min_t)*0.5
mod_source_size = np.sqrt(4*beam_size**2 + source_size**2)
res_lim = np.sqrt(4*beam_size**2)
pos_acc = sep_arr.arcsecond / mod_source_size

rel_err_x = np.linspace(rel_err_lims[0],rel_err_lims[1],400)
score_fct = np.minimum(1.0,np.abs(err_threshold/rel_err_x[:]))
avg_position_score = np.mean(scores_df["position"])

bins_x = 10**(np.linspace(param_lims[0], param_lims[1],120))
bins_y = np.linspace(rel_err_lims[0],rel_err_lims[1],60)
h = axs[0,0].hist2d(mod_source_size,pos_acc, bins=[bins_x,bins_y], cmap=cmap, norm=colors.LogNorm(), rasterized=True)

axs[0,0].plot([10**(param_lims[0]),10**(param_lims[1])],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--") #Max score interval
axs[0,0].plot([10**(param_lims[0]),10**(param_lims[1])],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--") #Max score interval

axs[0,0].set_xscale("log")
axs[0,0].set_xlim(10**param_lims[0], 10**param_lims[1])
axs[0,0].grid(axis="y", which="major", ls="--")
axs[0,0].set_ylim(rel_err_lims[0]-rel_err_offset, rel_err_lims[1])
axs[0,0].set_ylabel(r"Position error", fontsize=14)
axs[0,0].set_xlabel(r"Reference size ($\hat{S}'')$ [arcsec]", fontsize=14)
axs[0,0].yaxis.set_label_coords(-0.05, 0.5)
axs[0,0].set_title("Position", fontweight="bold", fontsize=18, rotation="vertical", x=-0.12,y=0.47, verticalalignment="center", horizontalalignment="center")

cb = fig.colorbar(h[3], ax=axs[0,0], pad=-0.002)
cb.set_label("N. of predictions", fontsize=12)


n_h1, bins_h1, patches_h1 = axs[0,2].hist(pos_acc, bins=rel_hist_nb_bins, range=rel_err_lims, 
	histtype="bar", rwidth = 0.85, orientation="horizontal", color="grey", rasterized=True)
axs[0,2].set_xlabel("N. of predictions", fontsize=12)
x_lims = axs[0,2].get_xlim()
axs[0,2].plot(x_lims,[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[0,2].plot(x_lims,[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[0,2].ticklabel_format(style="sci", scilimits=(0,1))
axs[0,2].grid(axis="y", which="major", ls="--")
axs[0,2].text(0.5, 0.92,"Pos. mean score: %0.3f"%(avg_position_score), color="black", verticalalignment="center",
	fontsize=12, horizontalalignment="center", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[0,2].transAxes)

id_threshold = np.where(np.abs(bins_h1[1:]) > 0.5)[0]
for i_d in id_threshold[:]:
	patches_h1[i_d].set_facecolor("red")

real_n_id = np.where(pos_acc > 0.5)[0]
real_frac = len(real_n_id)/len(pos_acc)
axs[0,2].text(n_h1[np.min(id_threshold)]*2, 0.5,r"$%0.2f\; \%% \; w/\; Dist > 0.5 \times T_{Size}$"%(real_frac*100), color="red", 
	verticalalignment="bottom", fontsize=10, horizontalalignment="left", fontweight="bold")


axs[0,3].plot(score_fct, rel_err_x, c="black", ls="--", lw=l_lw)
axs[0,3].plot([0,1],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[0,3].plot([0,1],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[0,3].set_xlabel("Position score fct.", fontsize=12)
axs[0,3].grid(axis="y", which="major", ls="--")
axs[0,3].text(0.9, 0.92,"Thr.: %0.1f"%(err_threshold), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="right", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[0,3].transAxes)


#########   FLUX   #########

rel_err_lims = [-1.2,2.0]
rel_err_offset = 0.3
param_lims = [-6.3,-2.0]
p_train_lims = [-5.705,-2.699]
err_threshold = 0.1

rel_err_x = np.linspace(rel_err_lims[0],rel_err_lims[1],400)
score_fct = np.minimum(1.0,np.abs(err_threshold/rel_err_x[:]))
param_low_lim_x = np.linspace(param_lims[0],param_lims[1],400)
param_low_lim_fct = (10**p_train_lims[0] - 10**param_low_lim_x)/10**param_low_lim_x
avg_flux_score = np.mean(scores_df["flux"]) 

bins_x = 10**(np.linspace(param_lims[0], param_lims[1],120))
bins_y = np.linspace(rel_err_lims[0]-rel_err_offset,rel_err_lims[1],60)
h = axs[1,0].hist2d(matched.a_flux_t,(matched.a_flux - matched.a_flux_t)/matched.a_flux_t, bins=[bins_x,bins_y], cmap=cmap, norm=colors.LogNorm(), rasterized=True)
axs[1,0].set_xscale("log")
axs[1,0].grid(axis="y", which="major", ls="--")
axs[1,0].plot([10**(param_lims[0]),10**(param_lims[1])],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--") #Max score interval
axs[1,0].plot([10**(param_lims[0]),10**(param_lims[1])],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--") #Max score interval

axs[1,0].annotate(text="",xy=(10**p_train_lims[0],0.05),
	xytext=(10**p_train_lims[1],0.05),
	xycoords=axs[1,0].get_xaxis_transform(),
	arrowprops=dict(arrowstyle="|-|", color="C1", linewidth=l_lw*1.5, 
	mutation_scale=5.0, shrinkA = 0.0, shrinkB=0.0))
axs[1,0].text(10**((p_train_lims[0]+p_train_lims[1])*0.5), 0.08, "Training interval", fontweight="bold", fontstyle="italic", fontsize=9,
	transform=axs[1,0].get_xaxis_transform(), color="C1", horizontalalignment="center", verticalalignment="center")
axs[1,0].plot(10**param_low_lim_x, param_low_lim_fct, c="black", lw=l_lw*1.5, ls=":")

axs[1,0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[1,0].set_ylim(rel_err_lims[0]-rel_err_offset,rel_err_lims[1])
axs[1,0].set_ylabel(r"Flux error", fontsize=14)
axs[1,0].set_xlabel(r"Target flux [Jy]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[1,0], pad=-0.002)
cb.set_label("N. of predictions", fontsize=12)
axs[1,0].yaxis.set_label_coords(-0.05, 0.5)
axs[1,0].set_title("Flux", fontweight="bold", fontsize=18, rotation="vertical",x=-0.12,y=0.47, 
	verticalalignment="center", horizontalalignment="center")

n_h2, bins_h2, patches_h2 = axs[1,2].hist((matched.a_flux - matched.a_flux_t)/matched.a_flux_t, bins=rel_hist_nb_bins, range=rel_err_lims, 
	histtype="bar", rwidth = 0.85, orientation="horizontal", color="grey", rasterized=True)
axs[1,2].set_xlabel("N. of predictions", fontsize=12)
x_lims = axs[1,2].get_xlim()
axs[1,2].plot(x_lims,[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[1,2].plot(x_lims,[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[1,2].ticklabel_format(style="sci", scilimits=(0,1))
axs[1,2].grid(axis="y", which="major", ls="--")
axs[1,2].text(0.5, 0.92,"Flux mean score: %0.3f"%(avg_flux_score), color="black", verticalalignment="center",
	fontsize=11, horizontalalignment="center", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[1,2].transAxes)

id_threshold = np.where(bins_h2[1:] > 1.0)[0]
for i_d in id_threshold[:]:
	patches_h2[i_d].set_facecolor("red")

real_n_id = np.where(matched.a_flux > 2.0*matched.a_flux_t)[0]
real_frac = len(real_n_id)/len(matched.a_flux)
axs[1,2].text(n_h2[np.min(id_threshold)]*2, 1.0,r"$%0.2f\; \%% \; w/\; P_{Flux} > 2 \times T_{Flux}$"%(real_frac*100), color="red", 
	verticalalignment="bottom", fontsize=10, horizontalalignment="left", fontweight="bold")

axs[1,3].plot(score_fct, rel_err_x, c="black", ls="--", lw=l_lw)
axs[1,3].plot([0,1],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[1,3].plot([0,1],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[1,3].set_xlabel("Flux score fct.", fontsize=12)
axs[1,3].grid(axis="y", which="major", ls="--")
axs[1,3].text(0.9, 0.92,"Thr.: %0.1f"%(err_threshold), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="right", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[1,3].transAxes)

#########   BMAJ   #########

rel_err_lims = [-1.2,4.0]
rel_err_offset = 0.5
param_lims = [-1.3,2.0]
p_train_lims = [-0.04576,1.778]
err_threshold = 0.3

rel_err_x = np.linspace(rel_err_lims[0],rel_err_lims[1],400)
score_fct = np.minimum(1.0,np.abs(err_threshold/rel_err_x[:]))
param_low_lim_x = np.linspace(param_lims[0],param_lims[1],400)
param_low_lim_fct = (10**p_train_lims[0] - 10**param_low_lim_x)/10**param_low_lim_x
avg_bmaj_score = np.mean(scores_df["b_maj"]) 

bins_x = 10**(np.linspace(param_lims[0], param_lims[1],120))
bins_y = np.linspace(rel_err_lims[0],rel_err_lims[1],60)
h = axs[2,0].hist2d(matched.b_maj_t,(matched.b_maj - matched.b_maj_t)/matched.b_maj_t, bins=[bins_x,bins_y], cmap=cmap, norm=colors.LogNorm(), rasterized=True)
axs[2,0].set_xscale("log")
axs[2,0].grid(axis="y", which="major", ls="--")
axs[2,0].fill_between(np.array([param_lims[0],beam_size]), np.array([1.0,1.0]), np.array([0.0,0.0]),
	facecolor="gray", alpha=l_alpha, transform=axs[2,0].get_xaxis_transform())
axs[2,0].plot([10**(param_lims[0]),10**(param_lims[1])],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--") #Max score interval
axs[2,0].plot([10**(param_lims[0]),10**(param_lims[1])],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--") #Max score interval

axs[2,0].annotate(text="",xy=(10**p_train_lims[0],0.05),
	xytext=(10**p_train_lims[1],0.05),
	xycoords=axs[2,0].get_xaxis_transform(),
	arrowprops=dict(arrowstyle="|-|", color="C1", linewidth=l_lw*1.5, 
	mutation_scale=5.0, shrinkA = 0.0, shrinkB=0.0))
axs[2,0].text(0.01, 0.05, "< Beam size", fontweight="bold", fontstyle="italic", fontsize=9,
	transform=axs[2,0].transAxes, color="gray", horizontalalignment="left", verticalalignment="center")

axs[2,0].plot(10**param_low_lim_x, param_low_lim_fct, c="black", lw=l_lw*1.5, ls=":")
axs[2,0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[2,0].set_ylim(rel_err_lims[0]-rel_err_offset,rel_err_lims[1])
axs[2,0].set_ylabel(r"$Bmaj$ error", fontsize=14)
axs[2,0].set_xlabel(r"Target $Bmaj$ [arcsec]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[2,0], pad=-0.002)
cb.set_label("N. of predictions", fontsize=12)
axs[2,0].yaxis.set_label_coords(-0.05, 0.5)
axs[2,0].set_title("Bmaj", fontweight="bold", fontsize=18, rotation="vertical",x=-0.12,y=0.47, verticalalignment="center", horizontalalignment="center")

axs[2,2].hist((matched.b_maj - matched.b_maj_t)/matched.b_maj_t, bins=rel_hist_nb_bins, range=rel_err_lims, 
	histtype="bar", rwidth = 0.85, orientation="horizontal", color="grey", rasterized=True)
axs[2,2].set_xlabel("N. of predictions", fontsize=11)
x_lims = axs[2,2].get_xlim()
axs[2,2].plot(x_lims,[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[2,2].plot(x_lims,[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[2,2].ticklabel_format(style="sci", scilimits=(0,1))
axs[2,2].grid(axis="y", which="major", ls="--")
axs[2,2].text(0.5, 0.92,"Bmaj mean score: %0.3f"%(avg_bmaj_score), color="black", verticalalignment="center",
	fontsize=12, horizontalalignment="center", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[2,2].transAxes)

axs[2,3].plot(score_fct, rel_err_x, c="black", ls="--", lw=l_lw)
axs[2,3].plot([0,1],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[2,3].plot([0,1],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[2,3].set_xlabel("Bmaj score fct.", fontsize=12)
axs[2,3].grid(axis="y", which="major", ls="--")
axs[2,3].text(0.9, 0.92,"Thr.: %0.1f"%(err_threshold), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="right", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[2,3].transAxes)


#########   BMIN   #########

rel_err_lims = [-1.2,4.0]
rel_err_offset = 0.5
param_lims = [-1.3,2.0]
p_train_lims = [-0.5229,1.4771]
err_threshold = 0.3

rel_err_x = np.linspace(rel_err_lims[0],rel_err_lims[1],400)
score_fct = np.minimum(1.0,np.abs(err_threshold/rel_err_x[:]))
param_low_lim_x = np.linspace(param_lims[0],param_lims[1],400)
param_low_lim_fct = (10**p_train_lims[0] - 10**param_low_lim_x)/10**param_low_lim_x
avg_bmin_score = np.mean(scores_df["b_min"]) 

bins_x = 10**(np.linspace(param_lims[0], param_lims[1],120))
bins_y = np.linspace(rel_err_lims[0],rel_err_lims[1],60)
h = axs[3,0].hist2d(matched.b_min_t,(matched.b_min - matched.b_min_t)/matched.b_min_t, bins=[bins_x,bins_y], cmap=cmap, norm=colors.LogNorm(), rasterized=True)
axs[3,0].set_xscale("log")
axs[3,0].grid(axis="y", which="major", ls="--")
axs[3,0].fill_between(np.array([param_lims[0],beam_size]), np.array([1.0,1.0]), np.array([0.0,0.0]),
	facecolor="gray", alpha=l_alpha, transform=axs[3,0].get_xaxis_transform())
axs[3,0].plot([10**(param_lims[0]),10**(param_lims[1])],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--") #Max score interval
axs[3,0].plot([10**(param_lims[0]),10**(param_lims[1])],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--") #Max score interval

axs[3,0].annotate(text="",xy=(10**p_train_lims[0],0.05),
	xytext=(10**p_train_lims[1],0.05),
	xycoords=axs[3,0].get_xaxis_transform(),
	arrowprops=dict(arrowstyle="|-|", color="C1", linewidth=l_lw*1.5, 
	mutation_scale=5.0, shrinkA = 0.0, shrinkB=0.0))
axs[3,0].text(0.01, 0.05, "< Beam size", fontweight="bold", fontstyle="italic", fontsize=9,
	transform=axs[3,0].transAxes, color="gray", horizontalalignment="left", verticalalignment="center")
axs[3,0].plot(10**param_low_lim_x, param_low_lim_fct, c="black", lw=l_lw*1.5, ls=":")

axs[3,0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[3,0].set_ylim(rel_err_lims[0]-rel_err_offset,rel_err_lims[1])
axs[3,0].set_ylabel(r"$Bmin$ error", fontsize=14)
axs[3,0].set_xlabel(r"Target $Bmin$ [arcsec]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[3,0], pad=-0.002)
cb.set_label("N. of predictions", fontsize=12)
axs[3,0].yaxis.set_label_coords(-0.05, 0.5)
axs[3,0].set_title("Bmin", fontweight="bold", fontsize=18, rotation="vertical",x=-0.12,y=0.47, verticalalignment="center", horizontalalignment="center")

axs[3,2].hist((matched.b_min - matched.b_min_t)/matched.b_min_t, bins=rel_hist_nb_bins, range=rel_err_lims, 
	histtype="bar", rwidth = 0.85, orientation="horizontal", color="grey", rasterized=True)
axs[3,2].set_xlabel("N. of predictions", fontsize=12)
x_lims = axs[3,2].get_xlim()
axs[3,2].plot(x_lims,[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[3,2].plot(x_lims,[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")

axs[3,2].ticklabel_format(style="sci", scilimits=(0,1))
axs[3,2].grid(axis="y", which="major", ls="--")
axs[3,2].text(0.5, 0.92,"Bmin mean score: %0.3f"%(avg_bmin_score), color="black", verticalalignment="center",
	fontsize=12, horizontalalignment="center", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[3,2].transAxes)

axs[3,3].plot(score_fct, rel_err_x, c="black", ls="--", lw=l_lw)
axs[3,3].plot([0,1],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[3,3].plot([0,1],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[3,3].set_xlabel("Bmin score fct.", fontsize=12)
axs[3,3].grid(axis="y", which="major", ls="--")
axs[3,3].text(0.9, 0.92,"Thr.: %0.1f"%(err_threshold), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="right", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[3,3].transAxes)


#########   PA   #########

rel_err_lims = [-90.0,90.0] # Here = absolute error
rel_err_offset = 20
param_lims = [0,1.8]
p_train_lims = [0.45269,1.0]
err_threshold = 10.0

rel_err_x = np.linspace(rel_err_lims[0],rel_err_lims[1],400)
score_fct = np.minimum(1.0,np.abs(err_threshold/rel_err_x[:]))
avg_pa_score = np.mean(scores_df["pa"]) 

pa_t = np.copy(matched.pa_t)
pa = np.copy(matched.pa)

pa_t[pa_t > 180] -= 180
pa_t[pa_t > 90] -= 90
pa_t[pa_t > 45] -= 45
pa_t[pa_t < -45] += 45

pa[pa > 180] -= 180
pa[pa > 90] -= 90
pa[pa > 45] -= 45
pa[pa < -45] += 45

diff_pa = (pa-pa_t)

bins_x = 10**(np.linspace(param_lims[0], param_lims[1],120))
bins_y = np.linspace(rel_err_lims[0],rel_err_lims[1],60)
h = axs[4,0].hist2d(mod_source_size,diff_pa, bins=[bins_x,bins_y], cmap=cmap, norm=colors.LogNorm(), rasterized=True)
axs[4,0].set_xscale("log")
axs[4,0].grid(axis="y", which="major", ls="--")
axs[4,0].fill_between(np.array([param_lims[0],beam_size]), np.array([1.0,1.0]), np.array([0.0,0.0]),
	facecolor="gray", alpha=l_alpha, transform=axs[4,0].get_xaxis_transform())
axs[4,0].plot([10**(param_lims[0]),10**(param_lims[1])],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--") #Max score interval
axs[4,0].plot([10**(param_lims[0]),10**(param_lims[1])],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--") #Max score interval

axs[4,0].annotate(text="",xy=(10**p_train_lims[0],0.05),
	xytext=(10**p_train_lims[1],0.05),
	xycoords=axs[4,0].get_xaxis_transform(),
	arrowprops=dict(arrowstyle="<-", color="C1", linewidth=l_lw*1.5, 
	mutation_scale=20.0, shrinkA = 0.0))
axs[4,0].xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axs[4,0].set_ylim(rel_err_lims[0]-rel_err_offset,rel_err_lims[1])
axs[4,0].set_ylabel(r"PA error", fontsize=14)
axs[4,0].set_xlabel(r"Reference size ($\hat{S}''$) [arcsec]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[4,0], pad=-0.002)
cb.set_label("N. of predictions", fontsize=12)
axs[4,0].yaxis.set_label_coords(-0.05, 0.5)
axs[4,0].set_title("PA", fontweight="bold", fontsize=18, rotation="vertical",x=-0.12,y=0.47, verticalalignment="center", horizontalalignment="center")

axs[4,2].hist(diff_pa, bins=rel_hist_nb_bins, range=rel_err_lims, 
	histtype="bar", rwidth = 0.85, orientation="horizontal", color="grey", rasterized=True)
axs[4,2].set_xlabel("N. of predictions", fontsize=12)
x_lims = axs[4,2].get_xlim()
axs[4,2].plot(x_lims,[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[4,2].plot(x_lims,[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")

axs[4,2].set_yticks(np.arange(-100,100,20))
axs[4,2].ticklabel_format(style="sci", scilimits=(0,1))
axs[4,2].grid(axis="y", which="major", ls="--")
axs[4,2].text(0.5, 0.92,"PA avg. score: %0.3f"%(avg_pa_score), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="center", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[4,2].transAxes)

axs[4,3].plot(score_fct, rel_err_x, c="black", ls="--", lw=l_lw)
axs[4,3].plot([0,1],[err_threshold,err_threshold], c="red", lw=l_lw, ls="--")
axs[4,3].plot([0,1],[-err_threshold,-err_threshold], c="red", lw=l_lw, ls="--")
axs[4,3].set_xlabel("PA score fct.", fontsize=12)
axs[4,3].grid(axis="y", which="major", ls="--")
axs[4,3].text(0.9, 0.92,"Thr.: %0.1f"%(err_threshold), color="black", verticalalignment="center", 
	fontsize=12, horizontalalignment="right", fontweight="bold", 
	bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"), transform=axs[4,3].transAxes)

if(savefig_all):
	plt.savefig("figures/all_param_pred.pdf", dpi=250, bbox_inches='tight')
else:
	plt.show()
plt.close()




######################################################################
#####    Distribution of the sources over the full image field   #####
######################################################################

c_map = "gist_earth"
c_map_lim = 110

n_x_bin = 200 #nb_area_w
n_y_bin = 200 #nb_area_h


id_list = np.arange(0,np.shape(box_catalog)[0])
id_list = np.delete(id_list, matched.id[:])

no_match = box_catalog[id_list,:]

index = np.where(no_match[:,0] > 90.0)
no_match[index[0],0] -= 360.0

index = np.where((no_match[:,0] < ra_max) & (no_match[:,0] > ra_min) &
				 (no_match[:,1] < dec_max) & (no_match[:,1] > dec_min))
no_match = np.delete(no_match,index, axis=0)

index = np.where((target_missed[:,0] < ra_max) & (target_missed[:,0] > ra_min) &
				 (target_missed[:,1] < dec_max) & (target_missed[:,1] > dec_min))
target_missed = np.delete(target_missed,index, axis=0)


fig, axs = plt.subplots(1,5, figsize=(19,4.5), dpi=300, gridspec_kw={'width_ratios': [1, 1, 0.02,1, 1]}, constrained_layout=True)
axs[2].set_visible(False)

c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
x_all, y_all = utils.skycoord_to_pixel(c, wcs_img)

h = axs[0].hist2d(x_all,y_all, bins=(n_x_bin,n_y_bin), range=[[0,32768],[0,32768]], vmax=c_map_lim, cmap=c_map, rasterized=True)
axs[0].tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
axs[0].set_ylabel("Y [pix]", fontsize=14)
axs[0].set_xlabel("X [pix]", fontsize=14)
axs[0].set_title("All predictions", fontweight="bold", fontsize=16)

c = SkyCoord(ra=matched.ra_cent.values*u.degree, dec=matched.dec_cent.values*u.degree, frame='icrs')
x_match, y_match = utils.skycoord_to_pixel(c, wcs_img)

h = axs[1].hist2d(x_match,y_match, bins=(n_x_bin,n_y_bin), range=[[0,32768],[0,32768]], vmax=c_map_lim, cmap=c_map, rasterized=True)
axs[1].tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
axs[1].set_xlabel("X [pix]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[1], pad=0.02, extend='max')
cb.set_label("N. of objects", fontsize=15, labelpad=4)
cb.set_ticks(np.arange(0,c_map_lim,25))
axs[1].set_title("Match only", fontweight="bold", fontsize=16)

c = SkyCoord(ra=no_match[:,0]*u.degree, dec=no_match[:,1]*u.degree, frame='icrs')
x_nomatch, y_nomatch = utils.skycoord_to_pixel(c, wcs_img)

h = axs[3].hist2d(x_nomatch,y_nomatch, bins=(n_x_bin,n_y_bin), range=[[0,32768],[0,32768]], vmax=0.12*c_map_lim, cmap="hot", rasterized=True)
axs[3].tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
axs[3].tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
axs[3].set_ylabel("Y [pix]", fontsize=14)
axs[3].set_xlabel("X [pix]", fontsize=14)
axs[3].set_title("False / Bad", fontweight="bold", fontsize=16)

c = SkyCoord(ra=target_missed[:,0]*u.degree, dec=target_missed[:,1]*u.degree, frame='icrs')
x_missed, y_missed = utils.skycoord_to_pixel(c, wcs_img)

h = axs[4].hist2d(x_missed,y_missed, bins=(n_x_bin,n_y_bin), range=[[0,32768],[0,32768]], vmax=0.12*c_map_lim, cmap="hot", rasterized=True)
axs[4].tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
axs[4].tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
axs[4].set_xlabel("X [pix]", fontsize=14)
cb = fig.colorbar(h[3], ax=axs[4], pad=0.02, extend='max')
cb.set_label("N. of objects", fontsize=15, labelpad=4)
cb.set_ticks(np.arange(0,0.12*c_map_lim,5))

axs[4].set_title("Missed", fontweight="bold", fontsize=16)

if(savefig_all):
	plt.savefig("figures/pred_match_false_field_map.pdf", dpi=300, bbox_inches='tight')
else:
	plt.show()
plt.close()




######################################################################
#####             Recall-precision curve and AP                  #####
######################################################################

# WARNING : TO OBTAIN FULL PRECISION-RECALL CURVE,
# THE SCRIPT MUST BE RE-RUN WITH A VERY LOW OBJECTNESS SELECTION threshold <= 0.03

print("Compute the recall-precision curve and AP value")

if(prob_obj_cases[0] < 0.04):

	recall_precision = np.zeros((np.shape(flat_kept_scaled)[0], 7))

	recall_precision[:,0] = flat_kept_scaled[:,6]
	recall_precision[:,1] = flat_kept_scaled[:,5]
	recall_precision[matched.id,2] = 1

	recall_precision[:,3] = np.cumsum(recall_precision[:,2])
	recall_precision[:,4] = np.cumsum(1.0 - recall_precision[:,2])
	recall_precision[:,5] = recall_precision[:,3] / (recall_precision[:,3]+recall_precision[:,4])
	recall_precision[:,6] = recall_precision[:,3] / full_truth_cat_size

	#Go in reverse to set the value for the all point interpolation
	interp_curve = np.maximum.accumulate(recall_precision[::-1,5])[::-1]

	AP_all = np.trapz(interp_curve, recall_precision[:,6])
	print ("AP = %f"%(AP_all))


	fig, axs = plt.subplots(2,3, figsize=(16,8), dpi=300, gridspec_kw={'width_ratios': [1, 1,1]}, constrained_layout=True, sharey="all")

	axs[0,0].plot(recall_precision[:,6], recall_precision[:,5], label="All points", c="black", lw=2.0)
	axs[0,0].set_ylabel("Precision", fontsize=11)
	axs[0,0].set_xticks([0.0,0.05,0.1,0.15])

	axs[1,0].plot(recall_precision[:,6], interp_curve, label="Smoothed", c="black", lw=2.0)
	axs[1,0].set_ylabel("Precision", fontsize=11)
	axs[1,0].set_xlabel("Recall", fontsize=11)
	axs[1,0].set_xticks([0.0,0.05,0.1,0.15])

	axs[0,1].plot(recall_precision[:,1], recall_precision[:,5], ls=(0,(4,4)), label="All", c="black", zorder=3)
	axs[1,1].plot(recall_precision[:,1], interp_curve, ls=(0,(4,4)), label="All", c="black", zorder=3)

	for i_b in range(0,nb_box):
		c_box_id = np.where(recall_precision[:,0] == i_b)[0]
		per_box_precision = np.zeros((np.shape(c_box_id)[0], 5))
		per_box_precision[:,0] = recall_precision[c_box_id,1]
		per_box_precision[:,1] = recall_precision[c_box_id,2]
		per_box_precision[:,2] = np.cumsum(per_box_precision[:,1])
		per_box_precision[:,3] = np.cumsum(1.0 - per_box_precision[:,1])
		per_box_precision[:,4] = per_box_precision[:,2] / (per_box_precision[:,2]+per_box_precision[:,3])
		axs[0,1].plot(per_box_precision[:,0], per_box_precision[:,4], label="Id. %d"%(i_b))
		
		interp_curve = np.maximum.accumulate(per_box_precision[::-1,4])[::-1]
		axs[1,1].plot(per_box_precision[:,0], interp_curve, label="Id. %d"%(i_b))
		
	axs[0,1].invert_xaxis(); axs[1,1].invert_xaxis()
	axs[1,1].set_xlabel("Objectness", fontsize=11)

	bin_size = np.array([0.02,0.02,0.02,0.02,0.02,0.02,0.05,0.05,0.05])

	for i_b in range(0,nb_box):

		nb_bins = int(0.95/bin_size[i_b])
		purity_per_bin = np.zeros((nb_bins,2))

		c_box_id = np.where(recall_precision[:,0] == i_b)[0]
		l_recall_precision = recall_precision[c_box_id,:]
		
		for i in range(nb_bins):
			start_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-bin_size[i_b]*i))
			end_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-bin_size[i_b]*(i+1)))
			nb_true = np.shape(np.where(l_recall_precision[start_id:end_id,2] == 1)[0])
			if(end_id-start_id > 0):
				purity_per_bin[i,0] = nb_true/(end_id-start_id)
			purity_per_bin[i,1] = 0.95 - bin_size[i_b]*(i+1) + bin_size[i_b]/2.
		
		axs[0,2].step(purity_per_bin[:,1], purity_per_bin[:,0], label="Id. %d"%(i_b), where="mid")
		interp_curve = np.maximum.accumulate(purity_per_bin[::-1,0])[::-1]
		axs[1,2].step(purity_per_bin[:,1], interp_curve, label="Id. %d"%(i_b), where="mid")

	#All boxes binned purity
	nb_bins = int(0.95/0.01)
	purity_per_bin = np.zeros((nb_bins,2))

	l_recall_precision = recall_precision[:,:]

	for i in range(nb_bins):
		start_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-0.01*i))
		end_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-0.01*(i+1)))
		nb_true = np.shape(np.where(l_recall_precision[start_id:end_id,2] == 1)[0])
		if(end_id-start_id > 0):
			purity_per_bin[i,0] = nb_true/(end_id-start_id)
		purity_per_bin[i,1] = 0.95 - 0.01*(i+1) + 0.01/2.

	axs[0,2].step(purity_per_bin[:,1], purity_per_bin[:,0], ls=(0,(4,4)), label="All", c="black", where="mid")
	interp_curve = np.maximum.accumulate(purity_per_bin[::-1,0])[::-1]
	axs[1,2].step(purity_per_bin[:,1], interp_curve, ls=(0,(4,4)), label="All", c="black", where="mid")

	axs[0,2].legend()
	axs[0,2].invert_xaxis(); axs[1,2].invert_xaxis()
	axs[1,2].set_xlabel("Objectness", fontsize=11)

	axs[0,0].set_title("P-R curve", fontweight="bold",fontsize=14, x=0.5,y=1.04)
	axs[0,1].set_title("Integrated", fontweight="bold",fontsize=14, x=0.5,y=1.04)
	axs[0,2].set_title("Binned", fontweight="bold",fontsize=14, x=0.5,y=1.04)

	axs[0,0].text(x=-0.17,y=0.5,s="Running", fontweight="bold", fontsize=14, rotation="vertical", 
		verticalalignment="center", horizontalalignment="center", transform=axs[0,0].transAxes)
	axs[1,0].text(x=-0.17,y=0.5,s="Smoothed", fontweight="bold", fontsize=14, rotation="vertical", 
		verticalalignment="center", horizontalalignment="center", transform=axs[1,0].transAxes)

	axs[1,0].text(x=0.025,y=0.1,s="AP = %0.3f"%(AP_all), fontweight="bold", fontsize=14, color="red",
		verticalalignment="center", horizontalalignment="center",
		bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"))

	for i in range(0,2):
		for j in range(0,3):
			axs[i,j].grid(axis="both", which="major", ls="--")


	if(savefig_all):
		plt.savefig("figures/recall_precision_objectness_curves.pdf", dpi=300, bbox_inches='tight')
	else:
		plt.show()
	plt.close()



	#Alternative version of previous plots

	print("Compute the recall-precision curve and AP value")

	recall_precision = np.zeros((np.shape(flat_kept_scaled)[0], 7))

	recall_precision[:,0] = flat_kept_scaled[:,6]
	recall_precision[:,1] = flat_kept_scaled[:,5]
	recall_precision[matched.id,2] = 1

	recall_precision[:,3] = np.cumsum(recall_precision[:,2])
	recall_precision[:,4] = np.cumsum(1.0 - recall_precision[:,2])
	recall_precision[:,5] = recall_precision[:,3] / (recall_precision[:,3]+recall_precision[:,4])
	recall_precision[:,6] = recall_precision[:,3] / full_truth_cat_size

	#Go in reverse to set the value for the all point interpolation
	interp_curve = np.maximum.accumulate(recall_precision[::-1,5])[::-1]

	AP_all = np.trapz(interp_curve, recall_precision[:,6])
	print ("AP = %f"%(AP_all))

	fig, axs = plt.subplots(1,1, figsize=(5,3), dpi=200, constrained_layout=True)

	axs.plot(recall_precision[:,6], recall_precision[:,5], label="All points", lw=2.0)
	axs.plot(recall_precision[:,6], interp_curve, label="Smoothed", ls="--", lw=2.0)
	axs.set_xticks([0.0,0.05,0.1,0.15])
	axs.set_ylabel("Purity", fontsize=12)
	axs.set_xlabel("Completeness", fontsize=12)
	axs.grid(axis="both", which="major", ls="--")
	axs.legend()

	axs.text(x=0.5,y=0.5,s="AP = %0.4f"%(AP_all*100), fontweight="bold", fontsize=13, color="red",
		verticalalignment="center", horizontalalignment="center", transform=axs.transAxes,
		bbox=dict(boxstyle='square', facecolor="white", edgecolor="black"))

	if(prob_obj_cases[0] < 0.04):
		plt.savefig("figures/precision_recall_curve.pdf", dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()


	fig, axs = plt.subplots(1,1, figsize=(5,3), dpi=200, constrained_layout=True)
	bin_size = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.05,0.05,0.1])

	for i_b in range(0,nb_box):

		nb_bins = int(0.95/bin_size[i_b])
		purity_per_bin = np.zeros((nb_bins,2))

		c_box_id = np.where(recall_precision[:,0] == i_b)[0]
		l_recall_precision = recall_precision[c_box_id,:]
		
		for i in range(nb_bins):
			start_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-bin_size[i_b]*i))
			end_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-bin_size[i_b]*(i+1)))
			nb_true = np.shape(np.where(l_recall_precision[start_id:end_id,2] == 1)[0])
			if(end_id-start_id > 0):
				purity_per_bin[i,0] = nb_true/(end_id-start_id)
			purity_per_bin[i,1] = 0.95 - bin_size[i_b]*(i+1) + bin_size[i_b]/2.
		
		axs.step(purity_per_bin[:,1], purity_per_bin[:,0], label="Id. %d"%(i_b), where="mid")
		
	nb_bins = int(0.95/0.01)
	purity_per_bin = np.zeros((nb_bins,2))

	l_recall_precision = recall_precision[:,:]

	for i in range(nb_bins):
		start_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-0.01*i))
		end_id = np.searchsorted(-l_recall_precision[:,1], -(0.95-0.01*(i+1)))
		nb_true = np.shape(np.where(l_recall_precision[start_id:end_id,2] == 1)[0])
		if(end_id-start_id > 0):
			purity_per_bin[i,0] = nb_true/(end_id-start_id)
		purity_per_bin[i,1] = 0.95 - 0.01*(i+1) + 0.01/2.

	axs.step(purity_per_bin[:,1], purity_per_bin[:,0], ls=(0,(4,4)), label="All", c="black", where="mid")

	axs.legend()
	axs.invert_xaxis() #shared so do-not repeat for axs[0,1] or it would go back to normal
	axs.set_ylabel("Binned purity", fontsize=12)
	axs.set_xlabel("Objectness", fontsize=12)

	if(savefig_all):
		plt.savefig("figures/binned_precision_objectness.pdf", dpi=300, bbox_inches='tight')
	else:
		plt.show()
	plt.close()






