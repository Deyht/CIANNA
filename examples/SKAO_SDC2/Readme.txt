
#	Copyright (C) 2025 David Cornu
#	for the Convolutional Interactive Artificial 
#	Neural Network by/for Astrophysicists (CIANNA) Code
#	(https://github.com/Deyht/CIANNA)
#
#	Licensed under the Apache License, Version 2.0 (the "License");
#	you may not use this file except in compliance with the License.
#	You may obtain a copy of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
#	Unless required by applicable law or agreed to in writing, software
#	distributed under the License is distributed on an "AS IS" BASIS,
#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#	See the License for the specific language governing permissions and
#	limitations under the License.



####################################################
                   Reference work
####################################################

These scripts are a simplified version of the scripts used in Cornu et al. 2025
presenting the application of the YOLO-CIANNA-3D method over the SKAO SDC2 dataset.
These scripts use model trained over the MAIN cube and apply them to the smaller LDEV cube.
This allow to reproduce the results from Appendix E of the paper.
All models, catalog and code associated with the paper (including training scripts)
will be archived on Zenodo, and are for now accessible at https://share.obspm.fr/s/swyCT7BgEGjtZK3.

This directory also include a Google Colab notebook that demonstrate the application of the
MC-BT2 model over the smaller DEV (10GB) subpart of the LDEV cube. The last few cells of the
notebook illustrate how to use the SDC2 scorer code to evaluate the final score of one of our
catalog and can easiliy be adapted to score any catalog from Cornu et al. 2025. 


####################################################
          Detail on the provided scripts
####################################################

- config.py imports the required packages, and set the structural properties of the target cube. 
- aux_fwd.py contains the function that decomposes the cube into overlapping sub-inputs that can be processed by our detector.
  It also includes a simplified version of the cube normalisation process.
- aux_post_proc.py contains a set of metrics and filtering functions used in the post-processing of the raw catalog.
- fwd.py contains the actual download of the raw data, the inference through CIANNA and the detailed post-process and scoring of the catalog.
  This file depends on all others, and is the one to run to obtain the results.
- direct_catalog_scoring.py allows to directly select a provided catalog and score it against the corresponding data cube to reproduce tables results from the paper.
















