
Author and copyright (C) 2026 - David Cornu  
Code and data associated with the article Cornu et al. 2026 (A&A)  
Released as part of the archived deposit 10.5281/zenodo.18403011  



####################################################
                   Reference work
####################################################

These scripts are a simplified version of the scripts used in Cornu et al. 2026
presenting the application of the YOLO-CIANNA-3D method over the SKAO SDC2 dataset.
These scripts use a model trained over the MAIN cube and apply it to the smaller LDEV cube, 
reproducing the results from Appendix E of the paper.
All models, catalogs, and codes associated with the paper (including training scripts)
are archived on Zenodo at 10.5281/zenodo.18403011.

This directory also includes a Google Colab notebook demonstrating the application of the
MC-BT2 model over the smaller DEV (10GB) subpart of the LDEV cube. The last few cells of the
notebook illustrate how to use the SDC2 scorer code to evaluate the final score of one of our
catalogs and can easily be adapted to score any catalog from Cornu et al. 2026. 


####################################################
          Detail on the provided scripts
####################################################

- config.py imports the required packages and sets the structural properties of the target cube. 
- aux_fwd.py contains the function that decomposes the cube into overlapping sub-inputs that can be processed by our detector.
  It also includes a simplified version of the cube normalisation process.
- aux_post_proc.py contains a set of metrics and filtering functions used in the post-processing of the raw catalog.
- fwd.py contains the actual download of the raw data, the inference through CIANNA, and the detailed post-process and scoring of the catalog.
  This file depends on all others and is the one to run to obtain the results.
- direct_catalog_scoring.py allows for the direct selection of a provided catalog and scoring it against the corresponding data cube to reproduce the scores presented in the paper.
















