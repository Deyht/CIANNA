
<p align="center">
<img src="https://github.com/Deyht/CIANNA/assets/21009408/90708962-e7ed-4dcb-88e7-f832a04753ff" alt="cianna_logo" width="80%"/>
</p>
*Logo made by &copy; Sarah E. Anderson*  

&nbsp;


<p align="left">
	<a href="https://doi.org/10.5281/zenodo.12806324" alt="DOI-ref">
		<img src="https://img.shields.io/badge/DOI-10.5281/zenodo.12806324-blue" /></a>
	<a href="https://ascl.net/2501.005" alt="ascl-id">
		<img src="https://img.shields.io/badge/ascl-2501.005-blue.svg?colorB=262255" alt="ascl:2501.005" /></a>
</p>


## WARNING - YOU ARE CURRENTLY ON THE EXPERIMENTAL BRANCH OF CIANNA
**This branch provides early access to new functionalities currently being tested and verified.**  
**This branch should be used for testing and prototyping, not to produce sensitive results or model deployment.**  
**Please note that:**

* The published documentation might not reflect the functions of this branch.
* Some functions might be deprecated and function interfaces might have changed in an undocumented way.
* Some specific configuration combination might crush the code.
* Some functionalities might not crash the code but still provide wrong result or behavior.
* API interface, model save format, and functionalities are all subject to changes over short time periods with no warnings.

**Use this branch at you own risks!**

&nbsp;

**Development Update 19/03/2026:**

This branch aimed at accumulating new functionalities that have a reasonable level of maturity in preparion for the next CIANNA release.  
Active development is not conducted here but on a private branch, so functionalities that end up here are those that we already tested on simples cases.  
The MNIST example script has been modified to illustrate most of the new functionalities.  

Highlight of new features (check the patch_note.txt file a detailed view):
* Add optmizers diversity and high level interface for them (SGD, ADAM, RMSprop). Also add decoupled weight decay for all optimizers.
* Add merge layers (Add and concatenate) and rework gradient flow. Enable the creation of residual blocks and U-net style architectures.
* Add Weight Exponential Moving Average (WEMA) support.
* Rework model saving to keep optimizer and ema states. Enable clean model training restart from a save state.
* The couv layer is now a grouped convolution. Classical convolution correspond to a single group. Also provide access to depth-wise convolution.
* The forward function now returns a numpy array with the network prediction directly (writing of a prediction result file is still possible but optional).
* Many QOL changes



&nbsp;

## CIANNA - Convolutional Interactive Artificial Neural Networks by/for Astrophysicists

CIANNA is a general-purpose deep learning framework primarily developed and used for astronomical data analysis. Functionalities and optimizations are added based on relevance for astrophysical problem-solving. CIANNA can be used to build and train large neural network models for various tasks and is provided with a high-level Python interface (similar to keras, pytorch, etc.). One of the specificities of CIANNA is its custom implementation of a YOLO-inspired object detector used in the context of galaxy detection in 2D or 3D radio-astronomical data products. The framework is fully GPU-accelerated through low-level CUDA programming.

**Development team**  
[David Cornu](https://vm-weblerma.obspm.fr/dcornu/) - creator and lead dev, post-doc researcher, AI Fellow PR[AI]RIE, FR - LUX / Observatoire de Paris, PSL  
Gregory Sainton - dev, AI Research engineer, FR - LUX / Observatoire de Paris  
Aristide Doussot - dev, HPC Research engineer, FR - LUX / Observatoire de Paris

Preferred contact point: david.cornu@observatoiredeparis.psl.eu

See Copyright &copy; and [License](#License) terms at the end.

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









