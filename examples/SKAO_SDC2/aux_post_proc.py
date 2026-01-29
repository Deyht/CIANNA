
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from config import *

### Post processing helper functions ###

# Distance IoU in 3D
@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU(box1, box2):
	inter_w = max(0.0, min(box1[3], box2[3]) - max(box1[0], box2[0]))
	inter_h = max(0.0, min(box1[4], box2[4]) - max(box1[1], box2[1]))
	inter_d = max(0.0, min(box1[5], box2[5]) - max(box1[2], box2[2]))

	inter_3d = inter_w * inter_h * inter_d
	uni_3d =  abs(box1[3]-box1[0])*abs(box1[4]-box1[1])*abs(box1[5]-box1[2]) \
		    + abs(box2[3]-box2[0])*abs(box2[4]-box2[1])*abs(box2[5]-box2[2]) \
		    - inter_3d
	enclose_w = (max(box1[3], box2[3]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[4], box2[4]) - min(box1[1], box2[1]))
	enclose_d = (max(box1[5], box2[5]) - min(box1[2], box2[2]))

	cx_a = (box1[3] + box1[0])*0.5; cx_b = (box2[3] + box2[0])*0.5
	cy_a = (box1[4] + box1[1])*0.5; cy_b = (box2[4] + box2[1])*0.5
	cz_a = (box1[5] + box1[2])*0.5; cz_b = (box2[5] + box2[2])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d)

	return float(inter_3d)/float(uni_3d) - float(dist_cent)/float(diag_enclose)
    

@jit(nopython=True, cache=True, fastmath=False)
def tile_filter(c_pred, c_box, c_tile, nb_box, nb_param, yolo_nb_sky_reg, yolo_nb_freq_reg):
	c_nb_box = 0
	for p_f in range(0, yolo_nb_freq_reg):
		for p_dec in range(0,yolo_nb_sky_reg):
			for p_ra in range(0,yolo_nb_sky_reg):
				
				for k in range(0,nb_box):
					offset = int(k*(8+nb_param))
					c_box[6] = c_pred[offset+6,p_f,p_dec,p_ra] #probability
					c_box[7] = c_pred[offset+7,p_f,p_dec,p_ra] #objectness
					#Manual objectness penality on the edges of the images (help for both obj selection and NMS)
					if(p_ra == 0 or p_ra == yolo_nb_sky_reg-1 or \
					   p_dec == 0 or p_dec == yolo_nb_sky_reg-1 or\
					   p_f == 0 or p_f == yolo_nb_freq_reg-1):
						c_box[6] = max(0.03,c_box[6]-0.05)
						c_box[7] = max(0.03,c_box[7]-0.05)
					
					if(c_box[7] >= 0.1):
						bx = (c_pred[offset+0,p_f,p_dec,p_ra] + c_pred[offset+3,p_f,p_dec,p_ra])*0.5
						by = (c_pred[offset+1,p_f,p_dec,p_ra] + c_pred[offset+4,p_f,p_dec,p_ra])*0.5
						bz = (c_pred[offset+2,p_f,p_dec,p_ra] + c_pred[offset+5,p_f,p_dec,p_ra])*0.5
						bw = max(5.0, c_pred[offset+3,p_f,p_dec,p_ra] - c_pred[offset+0,p_f,p_dec,p_ra])
						bh = max(5.0, c_pred[offset+4,p_f,p_dec,p_ra] - c_pred[offset+1,p_f,p_dec,p_ra])
						bd = max(5.0, c_pred[offset+5,p_f,p_dec,p_ra] - c_pred[offset+2,p_f,p_dec,p_ra])
						
						c_box[0] = bx - bw*0.5; c_box[3] = bx + bw*0.5
						c_box[1] = by - bh*0.5; c_box[4] = by + bh*0.5
						c_box[2] = bz - bd*0.5; c_box[5] = bz + bd*0.5
						
						c_box[8] = k
						c_box[9:9+nb_param] = c_pred[offset+8:offset+8+nb_param,p_f,p_dec,p_ra]
						c_box[-1] = p_f*yolo_nb_freq_reg*yolo_nb_sky_reg + p_dec*yolo_nb_sky_reg + p_ra
						c_tile[c_nb_box,:] = c_box[:]
						c_nb_box += 1
			
	return c_nb_box


@jit(nopython=True, cache=True, fastmath=False)
def first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, nms_threshold):
	c_nb_box_final = 0
	is_match = 1
	c_box_size_prev = c_nb_box
	
	while(c_nb_box > 0):
		max_objct = np.argmax(c_tile[:c_box_size_prev,7])
		c_box = np.copy(c_tile[max_objct])
		c_tile[max_objct,7] = 0.0
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1; c_nb_box -= 1; i = 0
		
		for i in range(0,c_box_size_prev):
			if(c_tile[i,7] < 0.0000000001):
				continue
			IoU = fct_DIoU(c_box[:6], c_tile[i,:6])
			if(IoU >  nms_threshold):
				c_tile[i,7] = 0.0
				c_nb_box -= 1
				
	return c_nb_box_final


@jit(nopython=True, cache=True, fastmath=False)
def inter_patch_NMS(boxes, comp_boxes, c_tile, direction, overlap, patch_shift, patch_size, nms_threshold):
	c_tile[:,:] = 0.0
	nb_box_kept = 0
	
	mask_keep = np.where((boxes[:,0] > overlap[0]) & (boxes[:,3] < patch_shift[0]) &\
						 (boxes[:,1] > overlap[1]) & (boxes[:,4] < patch_shift[1]) &\
						 (boxes[:,2] > overlap[2]) & (boxes[:,5] < patch_shift[2]))[0]
	mask_remain = np.where((boxes[:,0] <= overlap[0]) | (boxes[:,3] >= patch_shift[0]) |\
						   (boxes[:,1] <= overlap[1]) | (boxes[:,4] >= patch_shift[1]) |\
						   (boxes[:,2] <= overlap[2]) | (boxes[:,5] >= patch_shift[2]))[0]
	
	nb_box_kept = np.shape(mask_keep)[0]
	c_tile[0:nb_box_kept,:] = boxes[mask_keep,:]
	comp_boxes[:,0:3] += direction[:]*patch_shift[:]
	comp_boxes[:,3:6] += direction[:]*patch_shift[:]
	
	comp_mask_keep = np.where((comp_boxes[:,0] < patch_size[0]) & (comp_boxes[:,3] > 0) &\
							  (comp_boxes[:,1] < patch_size[1]) & (comp_boxes[:,4] > 0) &\
							  (comp_boxes[:,2] < patch_size[2]) & (comp_boxes[:,5] > 0))[0]
	
	for b_ref in mask_remain:
		found = 0
		for b_comp in comp_mask_keep:
			IoU = fct_DIoU(boxes[b_ref,:6], comp_boxes[b_comp,:6])
			if(IoU > nms_threshold and boxes[b_ref,7] < comp_boxes[b_comp,7]):
				found = 1
				break
		if(found == 0):
			c_tile[nb_box_kept,:] = boxes[b_ref,:]
			nb_box_kept += 1
		   
	return nb_box_kept


