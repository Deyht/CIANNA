
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


### The first CIANNA release (V-1.0) is here! Check the [release page](https://github.com/Deyht/CIANNA/releases)!

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA is a general-purpose deep learning framework primarily developed and used for astronomical data analysis. Functionalities and optimizations are added based on relevance for astrophysical problem-solving. CIANNA can be used to build and train large neural network models for various tasks and is provided with a high-level Python interface (similar to keras, pytorch, etc.). One of the specificities of CIANNA is its custom implementation of a YOLO-inspired object detector used in the context of galaxy detection in 2D or 3D radio-astronomical data products. The framework is fully GPU-accelerated through low-level CUDA programming.

**Development team**  
[David Cornu](https://vm-weblerma.obspm.fr/dcornu/) - creator and lead dev, post-doc researcher, AI Fellow PR[AI]RIE, FR - LERMA / Observatoire de Paris, PSL  
Gregory Sainton - dev, AI Research engineer, FR - LERMA / Observatoire de Paris

Preferred contact point: david.cornu@observatoiredeparis.psl.eu

See Copyright &copy; and [License](#License) terms at the end.

&nbsp;

## CIANNA application examples

Python scripts and Google-Colab-compatible notebooks are available under the [examples](https://github.com/Deyht/CIANNA/tree/CIANNA/examples) directory for most of the following examples.

| &#160;&#160;&#160;&#160;&#160;&#160; Description&#160;-&#160;Dataset &#160;&#160;&#160;&#160;&#160;&#160;  |  Visualization | Animation&#160;or&#160;real&#160;time |
| :---:  | :---:   | :---: |
| *** | <br> ***Classical computer vision examples*** <br> &#160;| *** |
| **Image&#160;classification <br> MNIST** <br> Top-1 accuracy ~99.3% <br> *Net. ~LeNet-5* <br> *630000 ips \@28p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/MNIST/mnist_pred_notebook.ipynb)       | <img src="https://github.com/Deyht/CIANNA/assets/21009408/802f5772-da5f-415b-8e49-cea75fba510b" alt="mnist_expl"/> |
| **Image&#160;classification <br> Imagenet - 1000 classes** <br> Top-1 acc ~74.7% <br> Top-5 acc ~91.7%  <br> *Net. ~Darknet19* <br> *740 ips \@448p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/ImageNET/imagenet_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/b7adde2f-e435-4bc1-907d-fc8052e58409" alt="imagenet_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/81b60e1e-79c9-4861-b212-791dca33c8dc" alt="imagenet_vid" width="100%"/> |
| **Object&#160;detection <br> COCO - 1000 classes** <br> mAP\@50 ~40.1% <br> COCO-mAP ~21.9% <br> *Net. ~Darknet19* <br> *690 ips \@416p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/COCO/coco_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/98ab135d-bba8-4f33-9d5d-46b0e095904e" alt="coco_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/b1948394-597d-44aa-aa9c-602783ce55f6" alt="coco_vid" width="100%"/> <br> *Real-time on a laptop GPU* |
| *** | <br> ***Astronomical dataset examples*** <br> &#160;| *** |
| **Source&#160;detection <br> SKA SDC1 <br> 2D continuum** <br> 560MHz - 1000h <br> score 479372 pts <br> *Net. 17 conv. layers* <br> *500 ips \@512p** <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/SKAO_SDC1/sdc1_pred_notebook.ipynb) <br> [![DOI](https://zenodo.org/badge/doi/10.1051/0004-6361/202449548.svg)](https://ui.adsabs.harvard.edu/abs/2024A%26A...690A.211C/abstract) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/a96112ba-0399-45b6-9804-533c921eb3a2" alt="apparent_flux_distribution" width="90%"/> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/10a31010-263b-4d97-887f-733b726f284e" alt="galmap_vid" width="75%"/> <br> *Not real-time here, only animated* |
| **Profile&#160;regression <br> 3D Galactic extinction mapping** <br> *Net. [C5x5.12-P2-{D3072}x2-D2048-D128]* <br> *120000 ips \@64p**<br> [![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2201.05571.svg)](https://ui.adsabs.harvard.edu/abs/2022arXiv220105571C/abstract) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/e3987887-8553-4cea-85e3-239112e6a74a" alt="galmap_polar_map_disc" width="70%"/> <br> *Face-on view of the galactic plane in a 45° "cone" toward the Carina arm (derived from the 3D map)* | *Per LOS prediction examples* <br> <img src="https://github.com/Deyht/CIANNA/assets/21009408/67a4be8e-8de0-4aa9-9659-f77c3fe9f5bb" alt="galmap_vid" width="100%"/> <br> <br> *Integrated extinction skyview* <br> <img src="https://github.com/Deyht/CIANNA/assets/21009408/797a895c-fd41-4fbc-8f57-6e9a231d59fa" alt="integrated_ext_map" width="100%"/> | 

**Images (or Inputs) per second (ips) are given for an RTX 4090 GPU in inference using FP16C_FP32A mixed-precision at the specified resolution and with maximum batch size to saturate performances*.

&nbsp;

###

## Installation

#### 

Please take a look at the [system requirements](https://github.com/Deyht/CIANNA/wiki/1\)-System-Requirements) and the [installation instructions](https://github.com/Deyht/CIANNA/wiki/2\)-Installation-instructions) wiki pages.  
=> A complete **step-by-step installation guide** of CIANNA and its dependencies from a fresh Ubuntu 20.04 is accessible [here](https://github.com/Deyht/CIANNA/wiki/Step-by-step-installation-guide-\(Ubuntu-20.04\)).

&nbsp;

## How to use

Please read the [How to use](https://github.com/Deyht/CIANNA/wiki/3\)-How-to-use-(Python-interface)) Wiki page for a minimalistic tour of CIANNA capabilities on a simple example script and dataset.  
A full description of all the Python interface functions is available as an [API documentation](https://github.com/Deyht/CIANNA/wiki/4\)-Interface-API-documentation) page on the Wiki.  
Please also consider consulting the [Step-by-step installation guide](https://github.com/Deyht/CIANNA/wiki/Step-by-step-installation-guide-\(Ubuntu-20.04\)) to verify everything was installed correctly.  
Several Python scripts and notebooks are provided as [examples](https://github.com/Deyht/CIANNA/tree/CIANNA/examples) for different datasets and applications.


&nbsp;


## Publications

List of known [publications](https://github.com/Deyht/CIANNA/wiki/Related-publications) that make use or directly refer to the CIANNA framework.

####

&nbsp;


###########################################################################

## License

These files are Copyright &copy; 2024 [David Cornu](https://vm-weblerma.obspm.fr/dcornu/), but released under the [Apache2 License](https://github.com/Deyht/CIANNA/blob/master/LICENSE.md).

&nbsp;

#### Contributor License Agreement
*While you are free to duplicate and modify this repository under the Apache2 License above, by being allowed to submit a contribution to this repository, you agree to the following terms:*

- *You grant to the present CIANNA framework (and its Author) your copyright license to reproduce and distribute your contributions and such derivative works.*

- *To the fullest extent permitted, you agree not to assert all of your "moral rights" in or relating to your contributions to the benefit of the present CIANNA framework.*

- *Your contribution was created in whole or in part by you and you have the right to submit it under the open source license indicated in the LICENSE file; or the contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right to submit that work with modifications.*









