
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
**This branch should be used for testing and prototyping, not to produce sensitive results or model deployment. Please note that:**

* The published documentation might not reflect the functions of this branch.
* Some functions might be deprecated, and function interfaces might have changed in an undocumented way (see the python_module.c file for up-to-date API functions).
* Some specific configuration combination might crush the code.
* Some functionalities might not crash the code, but still provide wrong results or behaviors.
* API interface, model save format, and functionalities are all subject to changes over short time periods with no warnings.

**Use this branch at your own risk!**

&nbsp;

**Development Update 28/10/2026:**
Experimental branch updated with the latest dev branch content (1.1.0.4_dev).  
Many chanches from 1.1.0.1, see patch note for an exhausive list of changes.  
**New save format is not compatible with previous versions (also can't load older models).**
* Rework the Python interface. All dimension arguments can now be provided as list and tuble in addition to the previous int np.arrays format. All "prev_layer" or "cond_layer" keyworks now support both absolute and relative network position. Move many default configuration setting from the python librarie to the C code.
* Rework all CIANNA input design. Add a new INPUT layer. Move input size declaration from cnn.init to cnn.input. Support multiple input layers. Create dataset now support providing a 
list of numpy arrays corresponding to the data from each input branch (must match the number of declared input layer in the architecture). Provide support for much more complex architectures.
* Add support for Adaptive Group Normalization (AdaGN). Added to the existing GN. Takes a extra conditionning layer and declare an hidden dense projection layer to learn gamma and beta. Y = gamma(X, C)*GN(X) + beta(X,C). Gradient propagates (accumulated) to the conditioning layer if it is not an input layer.
* Rework CIANNA's signature pivot-bias propagation in FLAT style layers to allow more flexible architecture (multi-branch).
* Improve bin saving format robustness across systems. Also improve partial architecture loading scheme.
* Rework YOLO-CIANNA assignement process: simplified and remove undesired correlations between refinement steps. Also add a new "regret-based" assignement, as an alternative to best fIoU.
* Rework merge to allow > 2 previous layer and extend previous layer type compatibility.
* Rework Conv layer to allow cross-group parameter sharing (similar to a 4D convolution) as opt-in, enable some new architectures like "MLP-MIXER".

**Development Update 29/04/2026:**
Experimental branch updated with the latest dev branch content (1.1.0.1_dev).
* Fix a few issues in the regarding dense to conv branching
* NORM layer has been rework to have one (gamma,beta) pair for each output channel, instead of one per group.
* Add a new GRN layer allowing to build residual blocks similar to ConvNext v2
* NORM and GRN layer CUDA versions are now fully on GPU, improving performances (especially latency).

**Development Update 19/03/2026:**

This branch aims to accumulate new functionalities with a reasonable level of maturity in preparation for the next CIANNA release.  
Active development is not conducted here but on a private branch. Functionalities that end up here are those we have already tested in  simple cases.  
The MNIST example script has been modified to illustrate most of the new functionalities.  

Highlight of new features (check the patch_note.txt file for a detailed view):
* Add optimizer choice and a high-level interface to interact with them (SGD, ADAM, RMSprop). Also, add decoupled weight decay for all optimizers.
* Add merge layers (Add and concatenate) and rework gradient flow. Enable the creation of residual blocks and U-net style architectures.
* Add Weight Exponential Moving Average (WEMA) support.
* Rework model saving to keep optimizer and ema states. Enable clean model training restart from a save state.
* The conv layer is now a grouped convolution. Classical convolution corresponds to a single group. Also enable depth-wise convolution.
* The forward function now returns a numpy array with the network prediction directly (writing of a prediction result file is still possible, but optional).
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









