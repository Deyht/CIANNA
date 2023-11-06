
<p align="center">
<img src="https://github.com/Deyht/CIANNA/assets/21009408/90708962-e7ed-4dcb-88e7-f832a04753ff" alt="cianna_logo" width="80%"/>
</p>

*Logo made by &copy; Sarah E. Anderson*

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA is a general-purpose deep learning framework mainly developed and used for astrophysical applications. Functionalities and optimizations are added based on relevance for astrophysical problem-solving. CIANNA can be used to build and train large neural network models for a wide variety of tasks and is provided with a high-level Python interface (similar to keras, pytorch, etc.). One of the specificities of CIANNA is its custom implementation of a YOLO-inspired object detector used in the context of galaxy detection in 3D radio-astronomical data cubes. The framework is fully GPU-accelerated through low-level CUDA programming.

Developer : [David Cornu](https://vm-weblerma.obspm.fr/dcornu/), FR - LERMA / Observatoire de Paris, PSL

david.cornu@observatoiredeparis.psl.eu

See Copyright &copy; and [License](#License) terms at the end.

&nbsp;

**Note:** The current version is transitioning to the 1.0 release. The source code and the API documentation are up-to-date, but several modifications will happen in the upcoming days for all other pages/examples/etc.

&nbsp;

## CIANNA application examples

Python scripts and Google Colab compatible notebooks are available under the [examples](https://github.com/Deyht/CIANNA/tree/CIANNA/examples) directory for most cases listed here.

| &#160;&#160;&#160;&#160;&#160;&#160; Description&#160;-&#160;Dataset &#160;&#160;&#160;&#160;&#160;&#160;  | Visualization | &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; Animation&#160;or&#160;real&#160;time&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; |
| :---  | :---:   | :---: |
| **Image&#160;classification <br> MNIST** <br> Top-1 accuracy ~99.3% <br> *Network ~LeNet-5* <br> **FPS\@28p: 630000* <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/MNIST/mnist_pred_notebook.ipynb)       | <img src="https://github.com/Deyht/CIANNA/assets/21009408/802f5772-da5f-415b-8e49-cea75fba510b" alt="mnist_expl"/> |
| **Image&#160;classification <br> Imagenet - 1000 classes** <br> Top-1 accuracy ~74.0% <br> Top-5 accuracy ~92.0%  <br> *Network ~Darknet19* <br> **FPS\@448p: 740* <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/ImageNET/imagenet_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/b173c036-95dd-460b-bb9e-c7b5bdc71bbd" alt="imagenet_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/81b60e1e-79c9-4861-b212-791dca33c8dc" alt="imagenet_vid"/> |
| **Object&#160;detection <br> COCO - 1000 classes** <br> mAP\@50 ~40.0% <br> COCO-mAP ~22.2% <br> *Network ~Darknet19* <br> **FPS\@416p: 690* <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deyht/CIANNA/blob/CIANNA/examples/COCO/coco_pred_notebook.ipynb) | <img src="https://github.com/Deyht/CIANNA/assets/21009408/98ab135d-bba8-4f33-9d5d-46b0e095904e" alt="coco_expl"> | <img src="https://github.com/Deyht/CIANNA/assets/21009408/25f39534-1b3e-4209-9b76-988c22d41010" alt="coco_vid"/> |

**FPS are given for an RTX 4090 GPU in inference using FP16C_FP32A mixed-precision at the specified resolution and with maximum batch size to saturate performances*.

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

These files are Copyright &copy; 2023 [David Cornu](https://vm-weblerma.obspm.fr/dcornu/), but released under the [Apache2 License](https://github.com/Deyht/CIANNA/blob/master/LICENSE.md).

&nbsp;

#### Contributor License Agreement
*While you are free to duplicate and modify this repository under the Apache2 License above, by being allowed to submit a contribution to this repository, you agree to the following terms:*

- *You grant to the present CIANNA framework (and its Author) your copyright license to reproduce and distribute your contributions and such derivative works.*

- *To the fullest extent permitted, you agree not to assert all of your "moral rights" in or relating to your contributions to the benefit of the present CIANNA framework.*

- *Your contribution was created in whole or in part by you and you have the right to submit it under the open source license indicated in the LICENSE file; or the contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open source license and you have the right to submit that work with modifications.*









