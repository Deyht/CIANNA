
<p align="center">
<img src="https://github.com/Deyht/CIANNA/assets/21009408/90708962-e7ed-4dcb-88e7-f832a04753ff" alt="cianna_logo" width="80%"/>
</p>
*Logo made by &copy; Sarah E. Anderson*  

&nbsp;

<p align="left">
	<a href="https://github.com/Deyht/CIANNA/releases" alt="Release-version">
		<img src="https://img.shields.io/badge/Latest%20release-1.0-green" /></a>
	<a href="https://github.com/Deyht/CIANNA/" alt="Current-version">
		<img src="https://img.shields.io/badge/Current%20version-1.0-green" /></a>
	<a href="https://github.com/Deyht/CIANNA/wiki" alt="Wiki-read">
		<img src="https://img.shields.io/badge/Wiki-Read-blue" /></a>
</p>
<p align="left">
	<a href="https://doi.org/10.5281/zenodo.12806324" alt="DOI-ref">
		<img src="https://img.shields.io/badge/DOI-10.5281/zenodo.12806324-blue" /></a>
	<a href="https://ascl.net/2501.005" alt="ascl-id">
		<img src="https://img.shields.io/badge/ascl-2501.005-blue.svg?colorB=262255" alt="ascl:2501.005" /></a>
</p>
<p align="left">
	<a href="https://github.com/Deyht/CIANNA/wiki/2)-Installation-instructions#dockerfile-installer" alt="Docker">
		<img src="https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white" /></a>
  	<a href="https://launchpad.net/~dcornu/+archive/ubuntu/cianna" alt="ppa_ref">
		<img src="https://img.shields.io/badge/ppa:-dcornu/cianna-orange" /></a>

</p>

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA is a general-purpose deep learning framework primarily developed and used for astronomical data analysis. Functionalities and optimizations are added based on relevance for astrophysical problem-solving. CIANNA can be used to build and train large neural network models for various tasks and is provided with a high-level Python interface (similar to keras, pytorch, etc.). One of the specificities of CIANNA is its custom implementation of a YOLO-inspired object detector used in the context of galaxy detection in 2D or 3D radio-astronomical data products. The framework is fully GPU-accelerated through low-level CUDA programming.

**[Development team](https://cianna.obspm.fr/content/Team/Team.html)**  
[David Cornu](https://vm-weblerma.obspm.fr/dcornu/) - creator and lead dev, post-doc researcher, AI Fellow PR[AI]RIE, FR - LUX / Observatoire de Paris, PSL  
Gregory Sainton - dev, AI Research engineer, FR - LUX / Observatoire de Paris  
Aristide Doussot - dev, HPC Research engineer, FR - LUX / Observatoire de Paris

Preferred contact point: contact.cianna@sympa.obspm.fr

See Copyright &copy; and [License](#License) terms at the end.

&nbsp;

**Quick access:**
* [CIANNA examples](https://cianna.obspm.fr/content/Examples/Examples.html)
* [Installation instructions](https://cianna.obspm.fr/content/Installation/Installation.html)
* [How to use](https://cianna.obspm.fr/content/How_to_use/How_to_use.html)
* [API documentation](https://cianna.obspm.fr/content/API/API.html)
* [Dev Blog](https://cianna.obspm.fr/content/Dev_Blog/Dev_Blog.html)
* [Publications](https://cianna.obspm.fr/content/Publications/Publications.html)


&nbsp;

**CIANNA status update (03/2026)**  
* CIANNA now has a dedicated website: [cianna.obspm.fr](https://cianna.obspm.fr) (thanks to the hard work of Aristide Doussot). This website aims to improve the framework's visibility and provide a more flexible interface to host content from the GitHub Wiki page. This includes installation instructions, application examples, tutorials, a dev blog, known publications, contact information, ...
* A new "experimental" branch has been added. This branch will be used to accumulate new functionalities that have reached a reasonable level of maturity in preparation for the next CIANNA release. This branch can be used for testing and prototyping using new features and reworked interfaces. Highlight of the recently added functionalities (see the experimental patch note for more details):
  * Optimizer choice and a high-level interface to interact with them (SGD, ADAM, RMSprop). Also add decoupled weight decay for all optimizers.
  * New merge layer (Add and concatenate) and reworked gradient flow. Enable the creation of residual blocks and U-net style architectures.
  * Weight Exponential Moving Average (WEMA) support.
  * Rework of model saving to keep optimizer and ema states. Enable clean model training to restart from a saved state.
  * The conv layer is now a grouped convolution. Classical convolution corresponds to a single group. Also provide access to depth-wise convolution.
  * The forward function now returns a numpy array with the network prediction directly (writing of a prediction result file is still possible, but optional).


**CIANNA status update (01/2026)**  
Minor update V-1.0.1 patches some identified bugs and adds a few functionalities in the stable version (see the patch note for more details). This update also prepares the distribution of the experimental branch of CIANNA (including merge/skip layers, multiple optimizers, etc) and other planned changes and new functionalities to the framework.


*Older status updates have been moved to [cianna's dev blog](https://cianna.obspm.fr/content/Dev_Blog/StatusUpdate.html)*

&nbsp;

## CIANNA application examples

Python scripts and Google-Colab-compatible notebooks are available under the [examples](https://github.com/Deyht/CIANNA/tree/CIANNA/examples) directory for most of the following examples.

| &#160;&#160;&#160;&#160;&#160;&#160; Description&#160;-&#160;Dataset &#160;&#160;&#160;&#160;&#160;&#160;  |  Visualization | Animation&#160;or&#160;real&#160;time |
| :---:  | :---:   | :---: |
| *** | <br> ***Classical computer vision examples*** <br> &#160;| *** |
| **Image&#160;classification <br> MNIST** <br> Top-1 accuracy ~99.3% <br> *Net. ~LeNet-5* <br> *630000 ips \@28p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/MNIST/mnist_train_notebook.ipynb)       | <img src="https://github.com/Deyht/CIANNA/assets/21009408/802f5772-da5f-415b-8e49-cea75fba510b" alt="mnist_expl"/> |
| **Image&#160;classification <br> Imagenet - 1000 classes** <br> Top-1 acc ~74.7% <br> Top-5 acc ~91.7%  <br> *Net. ~Darknet19* <br> *740 ips \@448p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/ImageNET/imagenet_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/b7adde2f-e435-4bc1-907d-fc8052e58409" alt="imagenet_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/81b60e1e-79c9-4861-b212-791dca33c8dc" alt="imagenet_vid" width="100%"/> |
| **Object&#160;detection <br> COCO - 1000 classes** <br> mAP\@50 ~40.1% <br> COCO-mAP ~21.9% <br> *Net. ~Darknet19* <br> *690 ips \@416p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/COCO/coco_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/98ab135d-bba8-4f33-9d5d-46b0e095904e" alt="coco_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/b1948394-597d-44aa-aa9c-602783ce55f6" alt="coco_vid" width="100%"/> <br> *Real-time on a laptop GPU* |
| *** | <br> ***Astronomical dataset examples*** <br> &#160;| *** |
| **Source&#160;detection <br> SKA SDC1 <br> 2D continuum** <br> 560MHz - 1000h <br> score 479372 pts <br> *Net. 17 conv. layers* <br> *500 ips \@512p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/SKAO_SDC1/sdc1_pred_notebook.ipynb) <br> [![DOI](https://zenodo.org/badge/doi/10.1051/0004-6361/202449548.svg)](https://ui.adsabs.harvard.edu/abs/2024A%26A...690A.211C/abstract) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/a96112ba-0399-45b6-9804-533c921eb3a2" alt="apparent_flux_distribution" width="90%"/> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/10a31010-263b-4d97-887f-733b726f284e" alt="sdc1_det_anim" width="75%"/> <br> *Not real-time here, only animated* |
| **Source&#160;detection <br> SKA SDC2 <br> 3D HI emission** <br> 950-1150MHz - 2000h <br> score 24664 pts <br> *Net. 23 conv. layers* <br> *300 ips \@64x64x256** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/SKAO_SDC2/sdc2_pred_notebook.ipynb) <br> [![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2201.05571.svg)](https://ui.adsabs.harvard.edu/abs/2025arXiv250912082C/abstract) | <img src="https://github.com/user-attachments/assets/8af5fa73-e205-425d-96c4-b405cc2d6d9b" alt="line_flux_distribution" width="85%"/> | <img src="https://github.com/user-attachments/assets/15a3c2d2-5e27-4083-9ee8-fa5c80561ab0" alt="sdc2_det_expl" width="80%"/> |
| **Profile&#160;regression <br> 3D Galactic extinction mapping** <br> *Net. [C5x5.12-P2-{D3072}x2-D2048-D128]* <br> *120000 ips \@64p**<br> [![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2201.05571.svg)](https://ui.adsabs.harvard.edu/abs/2022arXiv220105571C/abstract) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/e3987887-8553-4cea-85e3-239112e6a74a" alt="galmap_polar_map_disc" width="70%"/> <br> *Face-on view of the galactic plane in a 45° "cone" toward the Carina arm (derived from the 3D map)* | *Per LOS prediction examples* <br> <img src="https://github.com/Deyht/CIANNA/assets/21009408/67a4be8e-8de0-4aa9-9659-f77c3fe9f5bb" alt="galmap_vid" width="100%"/> <br> <br> *Integrated extinction skyview* <br> <img src="https://github.com/Deyht/CIANNA/assets/21009408/797a895c-fd41-4fbc-8f57-6e9a231d59fa" alt="integrated_ext_map" width="100%"/> | 
| **Fake&#160;galaxy&#160;generation <br> Based on galaxy zoo 2 <br> Cascaded DDPM** <br> *Ensemble of U-Nets* <br> *~40M param.* <br> *A few ips @192p** <br> *Made with the experimental branch of CIANNA* | <img src="https://github.com/user-attachments/assets/a0d98fa7-74a2-439f-b0ee-2821a88d069c" alt="cascaded_scheme_illust" width="100%"/> <br> *Cascading pipeline with 3 DDPM models* | *Generated examples* <br> <img src="https://github.com/user-attachments/assets/e1617c98-6460-46c7-aa24-2a8772c66871" alt="gen_real_galaxy_comp" width="100%"/> |

**Images (or Inputs) per second (ips) are provided for an RTX 4090 GPU in inference using FP16C_FP32A mixed precision at the specified resolution and with the maximum batch size to saturate performance*.


&nbsp;

###

## Installation

#### 

Please take a look at the [system requirements](https://github.com/Deyht/CIANNA/wiki/1\)-System-Requirements) and the [installation instructions](https://github.com/Deyht/CIANNA/wiki/2\)-Installation-instructions) wiki pages.  
=> A complete **step-by-step installation guide** for CIANNA and its dependencies from a fresh Ubuntu 20.04 is available [here](https://github.com/Deyht/CIANNA/wiki/Step-by-step-installation-guide-\(Ubuntu-20.04\)).

&nbsp;

## How to use

Please read the [How to use](https://github.com/Deyht/CIANNA/wiki/3\)-How-to-use-(Python-interface)) Wiki page for a minimalistic tour of CIANNA capabilities on a simple example script and dataset.  
A full description of all the Python interface functions is available as an [API documentation](https://github.com/Deyht/CIANNA/wiki/4\)-Interface-API-documentation) page on the Wiki.  
Please also consider consulting the [Step-by-step installation guide](https://github.com/Deyht/CIANNA/wiki/Step-by-step-installation-guide-\(Ubuntu-20.04\)) to verify everything was installed correctly.  
Several Python scripts and notebooks are provided as [examples](https://github.com/Deyht/CIANNA/tree/CIANNA/examples) for different datasets and applications.


&nbsp;


## Publications

List of known [publications](https://github.com/Deyht/CIANNA/wiki/Related-publications) that make use of or directly refer to the CIANNA framework.

####


## Preferred citation method

When referring to a specific functionality or application, feel free to cite the relevant publication.
In all cases, if your work makes use of any version of CIANNA, please cite the non-version-specific DOI from Zenodo [10.5281/zenodo.12806324](https://doi.org/10.5281/zenodo.12806324).

####

&nbsp;


###########################################################################

## License

These files are Copyright &copy; 2026-- [David Cornu](https://vm-weblerma.obspm.fr/dcornu/), but released under the [Apache2 License](https://github.com/Deyht/CIANNA/blob/master/LICENSE.md).

&nbsp;

#### Contributor License Agreement
*While you are free to duplicate and modify this repository under the Apache2 License above, by being allowed to submit a contribution to this repository, you agree to the following terms:*

- *You grant to the present CIANNA framework (and its Author) your copyright license to reproduce and distribute your contributions and such derivative works.*

- *To the fullest extent permitted, you agree not to assert all of your "moral rights" in or relating to your contributions to the benefit of the present CIANNA framework.*

- *Your contribution was created in whole or in part by you and you have the right to submit it under the open source license indicated in the LICENSE file; or the contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right to submit that work with modifications.*









