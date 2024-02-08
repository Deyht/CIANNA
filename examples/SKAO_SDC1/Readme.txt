
#	Copyright (C) 2020 David Cornu
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

These scripts are a simplified version of the scripts used in Cornu et al. 2024
presenting the application of the YOLO-CIANNA method over the SKAO SDC1 dataset.
These scripts can be used to reproduce the reference result of the paper by 
downloading the corresponding saved network model.

For the sake of simplicity, the scripts producing other results presented in 
the paper are not provided but can be simply reproduced by ajusting the network 
architecture, a few hyperparameters, or the SDC1 scorer code itself 
(modifying one line in dc_defns.py and recompiling the scorer should be enough).



####################################################
   A few guideline for using the provided scripts
####################################################

A full prediction and visualisation using the pretrained model is done in sdc1_pred_notebook.ipynb
which can be open in Google colab for a quick visualisation of the method capabilities


####################################################
  To do a prediction using the pretrain SDC1 model 
####################################################

1. Install CIANNA if not already done, install the ska_sdc scorer package, and the few python dependencies

2. Run pred_network.py whith no parameters after updating the cnn.init() parameters for your config
	- It will download the trained model
	- It will perform a prediction on the full image and save it in raw prediction formal

3. Run the pred_visual.py to generate all the figures and compute the complete score


####################################################
       To train an SDC1 detector from scratch
####################################################

1. Install CIANNA if not already done, install the ska_sdc scorer package, and the few python dependencies

2. Simply run train_network.py after updating the cnn.init() parameters for your config
	- It will download the data when laoding aux_fct.py
	- It will generate the training sample in data_gen.py
	- By default it will train the reference custom architecture for 5000 epochs

3. Run post_process.py after selecting a range of iterations to evaluate
	- It will help identify the top scoring iteration
	- It will provide the optimized objectness thresholds to achieve the best score

4. Run the pred_visual.py to generate all the figures and compute the complete score














