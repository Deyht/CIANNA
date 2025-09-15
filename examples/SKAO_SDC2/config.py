
#	Athor and copyright (C) 2025 - David Cornu
#   Code associated with the acrticle Cornu et al. 2025 (A&A)
#   Released as part of the archived deposit zenodo/xxxxx


#####       IMPORTS        #####

import numpy as np
import os, gc, sys, glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs import utils
from astropy.coordinates import SkyCoord
from numba import jit

#####   GLOBAL VARIABLES   #####

# Cube setup variables
# Data are expected to be square in the RA and DEC axes
map_pixel_size_ldev = 1286 # For the square deg LDEV
map_pixel_freq_size = 6668
pixel_size = 7.77777777778E-04 #In degree
pixel_size_freq = 3.00000000000E+04 #In Hz
beam_size = 1.94444449153E-03 #In degree

# Normalization variables
do_norm = 0
cont_removal_threshold = 0.5*6e-3 #in Jy
prenorm_scaling = 0.3

# Network setup variables
sky_size = 64
freq_size = 256
nb_param = 6
nb_box = 1

# Inference setup variables
c_size_sky = 8
c_size_freq = 16
yolo_nb_sky_reg = int(sky_size/c_size_sky)
yolo_nb_freq_reg = int(freq_size/c_size_freq)

overlap_sky = c_size_sky
overlap_freq = 2*c_size_freq
patch_shift_sky = sky_size - overlap_sky
patch_shift_freq = freq_size - overlap_freq

orig_offset_sky_ldev = patch_shift_sky - ((int(map_pixel_size_ldev/2) - int(sky_size/2) + patch_shift_sky)%patch_shift_sky)
orig_offset_freq = patch_shift_freq - ((int(map_pixel_freq_size/2) - int(freq_size/2) + patch_shift_freq)%patch_shift_freq)

nb_area_sky_ldev = int((map_pixel_size_ldev+2*orig_offset_sky_ldev)/patch_shift_sky)
nb_area_freq = int((map_pixel_freq_size+2*orig_offset_freq)/patch_shift_freq)
