
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from config import *

### Training helper functions ###

def create_test_batch():
	l_map_pixel_size = map_pixel_size_ldev + 2*orig_offset_sky_ldev
	l_map_pixel_freq_size = map_pixel_freq_size + 2*orig_offset_freq

	norm_data = np.fromfile("LDEV_norm_cube.bin", dtype="uint16")
	norm_data = np.reshape(norm_data, (l_map_pixel_freq_size, l_map_pixel_size, l_map_pixel_size))

	nb_inputs = nb_area_sky_ldev*nb_area_sky_ldev*nb_area_freq
	inputs = np.zeros((nb_inputs,sky_size*sky_size*freq_size), dtype="float32")

	for patch_freq in range(0,nb_area_freq):
		for patch_dec in range(0,nb_area_sky_ldev):
			for patch_ra in range(0,nb_area_sky_ldev):

				i = patch_freq*nb_area_sky_ldev*nb_area_sky_ldev + patch_dec*nb_area_sky_ldev + patch_ra
				p_ra   = patch_ra*patch_shift_sky
				p_dec  = patch_dec*patch_shift_sky
				p_freq = patch_freq*patch_shift_freq

				patch = np.copy(norm_data[p_freq:p_freq+freq_size, p_dec:p_dec+sky_size, p_ra:p_ra+sky_size])
				inputs[i,:] = (patch.flatten("C")/65535.0)*2.0 - 1.0

	del (norm_data)
	return inputs
  

# Simplified cube normalization function
def cube_norm(cube_path, cont_path):
	global cube_data, continuum_data, c_norm

	hdul = fits.open(cube_path, memmap=True)
	hdul2 = fits.open(cont_path, memmap=True)
	wcs_cube = WCS(hdul[0].header)

	cube_data = hdul[0].data
	continuum_data = hdul2[0].data
	# Search bright pixels from the continuum that causes residual substraction errors in the cube
	index = np.where(np.mean(np.asarray(continuum_data,dtype="float32"),axis=0) > 0.5*6e-3)

	c_norm = np.zeros(np.shape(cube_data)[0])
	for i in range(0, np.shape(cube_data)[0]):
		cube_slice = np.asarray(cube_data[i], dtype="float32")
		cube_slice[index] = 0.0
		c_norm[i] = np.std(cube_slice, axis=(0,1))
		np.savetxt("LDEV_c_norm.dat", c_norm)
		# STD per channel values evaluated on the LDEV cube (after bright pixels removal but before normalization)
		# are used as normalization factor for both LDEV and MAIN cubes for all models

	cube_data[:,index[0][:],index[1][:]] = 0.0
	for i in range(0, np.shape(cube_data)[0]):
		cube_data[i,:,:] = (np.tanh(prenorm_scaling*cube_data[i,:,:] / c_norm[i]) + 1.0)*0.5

	l_map_pixel_size = map_pixel_size_ldev + 2*orig_offset_sky_ldev
	l_map_pixel_freq_size = map_pixel_freq_size + 2*orig_offset_freq

	norm_data = np.zeros((l_map_pixel_freq_size, l_map_pixel_size,l_map_pixel_size), dtype="uint16")
	norm_data[:,:,:] = 0.5*65535.0
	norm_data[orig_offset_freq:-orig_offset_freq,orig_offset_sky_ldev:-orig_offset_sky_ldev,orig_offset_sky_ldev:-orig_offset_sky_ldev] \
		= np.asarray(cube_data[:,:,:] * 65535.0, dtype="uint16")
	norm_data.tofile("LDEV_norm_cube.bin")

	del (norm_data, cube_data, continuum_data)


