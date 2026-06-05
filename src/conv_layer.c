
/*
	Copyright (C) 2026-... David Cornu
	for the Convolutional Interactive Artificial 
	Neural Networks by/for Astrophysicists (CIANNA) Code
	(https://github.com/Deyht/CIANNA)

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/


#include "prototypes.h"

// Local variables
static conv_param *c_param;

// Public are in prototypes.h

// Private prototypes


//compute the number of area to convolve regarding the filters parameters
int nb_area_comp(int size, int f_size, int padding, int int_padding, int stride)
{
	int l_nb_area = 0;
	
	if((size + padding*2 - f_size)%stride != 0)
	{
		printf("\n WARNING: unable to divide current input volume into \
an integer number of conv/pool regions\n\
 This might produce unstable results !\n");
	}		
	l_nb_area = (size + (size-1)*int_padding + padding*2 - f_size) / stride + 1;

	if(l_nb_area < 1)
	{
		printf("\n ERROR: conv layer configuration resulted in a number of region < 1 in at least one dimension!\n\n");
		exit(EXIT_FAILURE);
	}

	return l_nb_area;
}


int conv_create(network *net, layer *previous, int *f_size, size_t nb_filters, size_t nb_groups, int *stride, int *padding, 
	int *int_padding, int *in_shape, const char *activation, float *bias, float drop_rate, 
	const char *init_fct, float init_scaling, FILE *f_load, int load_optim_state, int f_bin)
{
	size_t i, k;
	size_t subdim_a, subdim_b;
	size_t spatial_f_size, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	size_t mem_approx = 0, batch_size;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	batch_size = net->batch_size;
	
	printf("L:%d - CREATING CONVOLUTIONAL LAYER ...\n", net->nb_layers);
	
	current->type                = CONV;
	current->output_type         = SPATIAL;
	current->frozen              = 0;
	current->wema_replace_signal = 0;
	current->dropout_rate        = drop_rate;
	current->previous            = previous;
	
	current->output_dim = (int*) calloc(4, sizeof(int));
	
	c_param = (conv_param*) malloc(sizeof(conv_param));
	current->param = c_param;
	
	c_param->f_size      = (int*) calloc(3, sizeof(int));
	c_param->stride      = (int*) calloc(3, sizeof(int));
	c_param->padding     = (int*) calloc(3, sizeof(int));
	c_param->int_padding = (int*) calloc(3, sizeof(int));
	
	for(k = 0; k < 3; k++)
	{
		c_param->f_size[k]      = f_size[k];
		c_param->stride[k]      = stride[k];
		c_param->padding[k]     = padding[k];
		c_param->int_padding[k] = int_padding[k];
	}
	
	spatial_f_size = f_size[0] * f_size[1] * f_size[2];
	
	//Compute the number of areas to be convolved in the input image
	if(previous == NULL)
	{
		current->prev_dim = net->in_dims;
		current->input = net->input;
	}
	else
	{
		if(previous->output_type == SPATIAL)
		{
			current->prev_dim = previous->output_dim;
		}
		else //FLAT
		{
			if(in_shape == NULL)
			{
				printf("\n ERROR: dense to conv conversion requires input_shape argument to be defined.\n\n");
				exit(EXIT_FAILURE);
			}
			if(previous->output_dim[3] != in_shape[0]*in_shape[1]*in_shape[2]*in_shape[3])
			{
				printf("\n ERROR: dense to conv input_shape mismatch.\n\n");
				exit(EXIT_FAILURE);
			}
			//in shape vector might be destroyed especially if called from the python inteface. Must be copied in a stable memory space.
			current->prev_dim = (int*) calloc(4,sizeof(int));
			for(k = 0; k < 3; k++)
				current->prev_dim[k] = in_shape[k];
		}
		current->input = previous->output;
	}
	
	for(k = 0; k < 3; k++)
		current->output_dim[k] = nb_area_comp(current->prev_dim[k], c_param->f_size[k], 
			c_param->padding[k], c_param->int_padding[k], c_param->stride[k]);
	current->output_dim[3] = nb_filters;
	
	nb_regions_out   = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
	nb_regions_in    =   current->prev_dim[0] *   current->prev_dim[1] *   current->prev_dim[2];
	prev_nb_channels =   current->prev_dim[3];
	
	if(nb_groups < 1)
		nb_groups = 1;
	c_param->nb_groups = nb_groups;
	if(prev_nb_channels % nb_groups != 0 || nb_filters % nb_groups != 0)
	{
		printf("\n ERROR: invalid nb_groups setting. Both number of input and output channels must be a multiple of nb_groups.\n");
		exit(EXIT_FAILURE);
	}
	chan_per_group_in  = prev_nb_channels / nb_groups;
	chan_per_group_out = nb_filters       / nb_groups;
	
	c_param->TC_padding = 0;
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC != FP32C_FP32A)
		c_param->TC_padding = 8 - (spatial_f_size * chan_per_group_in + 1) % 8;
	#endif
	
	//Activation related pre-computed sizes and shapes
	current->a_size       = nb_regions_out * nb_filters * batch_size; //here nb_filters = chan_per_group_out * nb_groups
	current->a_dim        = nb_regions_out;
	current->a_biased_dim = nb_regions_out;
	current->a_offset     = batch_size;
	
	define_activation_param(current, activation);
	
	if(bias != NULL)
		current->bias_value = *bias;
	
	
	//#############################################
	
	//For clarity we will express the subgroup conv problem dimensions for each data array
	//subdim_a is the leading dimension (number of rows in col-Major), subdim_b is the other dimension (number of columns in col-Major)
	//For all data array the dimensions are replicated for each group with a stride of subdim_a x subdim_b
	
	//######## Input related data arrays  ########
	
	subdim_a = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	subdim_b = batch_size * nb_regions_out;
	
	c_param->im2col_input = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
	mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
	//set bias value for the current layer, this value will not move during training
	for(i = 0; i < nb_groups * subdim_b; i++)
		((float*)c_param->im2col_input)[i*subdim_a + subdim_a - 1 - c_param->TC_padding] = current->bias_value;
	//done before cuda conv define so GPU version of im2col is already set
	
	//#############################################
	
	//######## Weights related data arrays ########
	
	subdim_a = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	subdim_b = chan_per_group_out;
	
	current->weights = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
	current->FP32_weights = current->weights;
	mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
	current->nb_params = nb_groups * chan_per_group_out * (spatial_f_size * chan_per_group_in + 1); //no padding
	
	if(!net->inference_only)
	{
		if(net->use_wema)
		{
			current->ema_weights = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
			mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
		}
	
		//allocate the weight gradient space
		current->gradient = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
		mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
		//Create optimize data arrays based on the weights dimensions
		mem_approx += define_optimizer_var(current, nb_groups * subdim_a * subdim_b);
	}
	
	if(f_load == NULL)
	{
		if(init_scaling < 0)
			init_scaling = 1.0f;
		
		initialize_weights(init_fct, current->weights, subdim_a - c_param->TC_padding, nb_groups * subdim_b, 
			0, 0.0, c_param->TC_padding, init_scaling);
	}
	else
		load_layer_weights(f_load, current, nb_groups * subdim_a * subdim_b, subdim_a - c_param->TC_padding, 
			c_param->TC_padding, load_optim_state, f_bin);
	
	if(net->use_wema && !net->inference_only && (f_load == NULL || !load_optim_state))
	{
		//printf("Copying regular weights into EMA weights\n");
		for(i = 0; i < nb_groups * subdim_a * subdim_b; i++)
			current->ema_weights[i] = ((float*)current->weights)[i];
	}
	
	//#############################################
	
	//######## Output related data arrays #########
	
	subdim_a = batch_size * nb_regions_out;
	subdim_b = chan_per_group_out;
	
	//Activation maps are not continuous for each image : 
	//	A1_im1, A1_im2, A1_im3, ... , A2_im1, A2_im2, A2_im3, ... 
	current->output = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
	mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
	if(drop_rate > 0.01f)
	{
		current->dropout_mask = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
		mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	}
	
	if(!net->inference_only)
	{
		//allocate output error comming from next layers
		current->delta_o = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
		mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	}
	
	//#############################################
	
	//######## Backprop related data arrays #######
	
	if(!net->inference_only)
	{	
		subdim_a = spatial_f_size * chan_per_group_out;
		subdim_b = batch_size * nb_regions_in;
		
		c_param->im2col_delta_o = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
		mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
		//Depth-wise rotated fitlers used for backpropagation of the error to the previous layer
		//are trimmed of bias and TC_padding. The corresponding dimensions are re-exposed of clarity.
		subdim_a = spatial_f_size * chan_per_group_out;
		subdim_b = chan_per_group_in;
	
		c_param->rotated_filters = (float*) calloc(nb_groups * subdim_a * subdim_b, sizeof(float));
		mem_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
	
		//temporary output error used for dense to conv link backprop / no groups
		if(previous != NULL && previous->output_type == FLAT)
		{
			subdim_a = prev_nb_channels * nb_regions_in;
			subdim_b = batch_size;
			
			c_param->temp_delta_o = (float*) calloc(subdim_a * subdim_b, sizeof(float));
			mem_approx += subdim_a * subdim_b* sizeof(float);
		}
	}
	
	//#############################################
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_conv_define(current);
			mem_approx = cuda_convert_conv_layer(current);
			cuda_define_activation_fct(current);
			#endif
			break;
		case C_BLAS:
			#ifdef BLAS
			blas_conv_define(current);
			define_activation_fct(current);
			#endif
			break;
		case C_NAIV:
			naiv_conv_define(current);
			define_activation_fct(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);
	printf("      Input: %dx%dx%dx%ld, Output: %dx%dx%dx%ld \n\
      Nb_groups: %ld, Filters: %ldf %dx%dx%dx%ld,\n\
      Stride: %d:%d:%d, padding: %d:%d:%d, int_padding: %d:%d:%d,  \n\
      Activation: %s, Bias: %0.2f, dropout rate: %0.2f\n\
      Nb. weights: %ld, Approx layer RAM/VRAM requirement: %d MB\n",
		current->prev_dim[0], current->prev_dim[1], current->prev_dim[2], prev_nb_channels,
		current->output_dim[0], current->output_dim[1], current->output_dim[2], nb_filters,
		nb_groups, nb_filters, c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], chan_per_group_in, 
		c_param->stride[0], c_param->stride[1], c_param->stride[2], 
		c_param->padding[0], c_param->padding[1], c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		activ, current->bias_value, current->dropout_rate,
		current->nb_params, (int)(mem_approx/1000000));
	net->total_nb_param += current->nb_params;
	net->memory_footprint += mem_approx;
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC != FP32C_FP32A)
	{
		if((batch_size * nb_regions_out) % 8 != 0 || chan_per_group_out % 8 != 0)
			printf(" WARNING: Forward gemm TC data misalignement due to layer size mismatch\n");
		if(current->previous != NULL && !net->inference_only && 
			( (batch_size * nb_regions_in) % 8 != 0 
			|| (spatial_f_size * chan_per_group_out) % 8 != 0
			|| chan_per_group_in % 8 != 0))
			printf(" WARNING: Backprop gemm TC data misalignment due to layer size mismatch\n");
		if(!net->inference_only && 
			( (batch_size * nb_regions_out) % 8 != 0 
			|| chan_per_group_out % 8))
			printf(" WARNING: Weights update gemm TC data misalignment due to layer size mismatch\n");
	}
	#endif

	return net->nb_layers - 1;
}


void conv_save(FILE *f, layer *current, int save_optim_state, int f_bin)
{
	size_t i, j;
	size_t subdim_a, subdim_b;
	size_t spatial_f_size, chan_per_group_in, chan_per_group_out;
	char layer_type = 'C';
	int nb_filters, prev_nb_channels;
	network *net;

	c_param = (conv_param*)current->param;
	net = current->c_network;
	yolo_param *y_param = NULL;
	nb_filters = current->output_dim[3];
	prev_nb_channels = current->prev_dim[3];
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&nb_filters, sizeof(int), 1, f);
		fwrite(&c_param->nb_groups, sizeof(int), 1, f);
		fwrite(c_param->f_size, sizeof(int), 3, f);
		fwrite(c_param->stride, sizeof(int), 3, f);
		fwrite(c_param->padding, sizeof(int), 3, f);
		fwrite(c_param->int_padding, sizeof(int), 3, f);
		fwrite(current->prev_dim, sizeof(int), 4, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		fwrite(&current->bias_value, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
		if(current->activation_type == YOLO)
		{
			y_param = net->y_param;
			fwrite(&y_param->nb_box, sizeof(int), 1, f);
			fwrite(&y_param->nb_class, sizeof(int), 1, f);
			fwrite(&y_param->nb_param, sizeof(int), 1, f);
			fwrite(&y_param->fit_dim, sizeof(int), 1, f);
			fwrite(&y_param->class_softmax, sizeof(int), 1, f);
			/* For save format compatibility */
			for(i = 0; i < 3; i++)
				for(j = 0; j < y_param->nb_box; j++)
					fwrite(y_param->prior_size + j*3 + i, sizeof(float), 1, f);
			for(i = 0; i < 6; i++)
				fwrite(&(y_param->slopes_and_maxes_tab[i][0]), sizeof(float), 3, f);
		}
	}
	else
	{
		fprintf(f,"C");
		fprintf(f, "%df%dg%dx%dx%d.%dx%dx%ds%dx%dx%dp%dx%dx%dip%dx%dx%dx%didim%fd%fb", 
			nb_filters, c_param->nb_groups,
			c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], 
			c_param->stride[0], c_param->stride[1], c_param->stride[2], 
			c_param->padding[0], c_param->padding[1], c_param->padding[2], 
			c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2], 
			current->prev_dim[0], current->prev_dim[1], current->prev_dim[2], 
			prev_nb_channels, current->dropout_rate, current->bias_value);
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n");
		if(current->activation_type == YOLO)
		{
			y_param = (yolo_param*)net->y_param;
			fprintf(f, "%d %d %d %d %d\n", y_param->nb_box, y_param->nb_class, 
				y_param->nb_param, y_param->fit_dim, y_param->class_softmax);
			
			for(i = 0; i < 3; i++)
			{
				for(j = 0; j < y_param->nb_box; j++)
					fprintf(f, "%g ", y_param->prior_size[j*3+i]);
				fprintf(f, "\n");
			}
			for(i = 0; i < 6; i++)
			{
				for(j = 0; j < 3; j++)
					fprintf(f, "%g ", y_param->slopes_and_maxes_tab[i][j]);
				fprintf(f, "\n");
			}
		}
	}
	
	spatial_f_size = c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2];
	chan_per_group_in = prev_nb_channels / c_param->nb_groups;
	chan_per_group_out = nb_filters / c_param->nb_groups;
	
	subdim_a = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	subdim_b = chan_per_group_out;
	
	save_layer_weights(f, current, c_param->nb_groups * subdim_a * subdim_b,
		subdim_a - c_param->TC_padding, c_param->TC_padding, 0, save_optim_state, f_bin);
}

void conv_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer)
{
	size_t i, j;
	int nb_filters, nb_groups;
	size_t nb_parameters;
	int f_size[3], stride[3], padding[3], int_padding[3], input_shape[4];
	float dropout_rate, bias, temp_read;
	
	int nb_box, nb_class, nb_yolo_param, fit_dim, class_softmax;
	float *prior_size = NULL;
	float slopes_and_maxes[6][3];
	
	char activ_type[40];
	char display_class_type[60];
	layer *previous;
	yolo_param* y_param = NULL;
	const char *IoU_type_char = "empty", *prior_dist_type_char = "empty", *error_type = "empty";
	
	if(!skip_layer)
		printf("Loading conv layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&nb_filters, sizeof(int), 1, f);
		fread(&nb_groups, sizeof(int), 1, f);
		fread(f_size, sizeof(int), 3, f);
		fread(stride, sizeof(int), 3, f);
		fread(padding, sizeof(int), 3, f);
		fread(int_padding, sizeof(int), 3, f);
		fread(input_shape, sizeof(int), 4, f);
		fread(&dropout_rate, sizeof(float), 1, f);
		fread(&bias, sizeof(float), 1, f);
		fread(activ_type, sizeof(char), 40, f);
		if(strncmp(activ_type, "YOLO", 4) == 0)
		{
			fread(&nb_box, sizeof(int), 1, f);
			fread(&nb_class, sizeof(int), 1, f);
			fread(&nb_yolo_param, sizeof(int), 1, f);
			fread(&fit_dim, sizeof(int), 1, f);
			fread(&class_softmax, sizeof(int), 1, f);
			prior_size = (float*) calloc(3*nb_box, sizeof(float));
			/* For save format compatibility */
			for(i = 0; i < 3; i++)
				for(j = 0; j < nb_box; j++)
					fread(prior_size + j*3 + i, sizeof(float), 1, f);
			for(i = 0; i < 6; i++)
				fread(slopes_and_maxes[i], sizeof(float), 3, f);
		}
	}
	else
	{
		fscanf(f, "%df%dg%dx%dx%d.%dx%dx%ds%dx%dx%dp%dx%dx%dip%dx%dx%dx%didim%fd%fb%s\n", 
			&nb_filters, &nb_groups,
			&f_size[0], &f_size[1], &f_size[2],
		 	&stride[0], &stride[1], &stride[2], 
		 	&padding[0], &padding[1], &padding[2], 
		 	&int_padding[0], &int_padding[1], &int_padding[2],
			&input_shape[0], &input_shape[1], &input_shape[2], &input_shape[3], 
			&dropout_rate, &bias, activ_type);
		if(strcmp(activ_type, "YOLO") == 0)
		{
			fscanf(f, "%d %d %d %d %d\n", &nb_box, &nb_class, &nb_yolo_param, &fit_dim, &class_softmax);
			prior_size = (float*) calloc(3*nb_box, sizeof(float));
			for(i = 0; i < 3; i++)
				for(j = 0; j < nb_box; j++)
					fscanf(f, "%f", &(prior_size[j*3+i]));
			for(i = 0; i < 6; i++)
				for(j = 0; j < 3; j++)
					fscanf(f, "%f", &(slopes_and_maxes[i][j]));
		}
	}
	
	if(strncmp(activ_type, "YOLO", 4) == 0)
	{
		if(net->y_param == NULL)
		{
			printf(" WARNING: Loading a YOLO layer with no prior call to the set_yolo_config function.\n");
			printf(" Loading will proceed with available parameters from the saved model (not suited for further training).\n");
			
			/*To compare with python_module.c*/
			set_yolo_config(net, 0/*nb_box*/, 0/*nb_class*/, 0/*nb_yolo_param*/, /*max_nb_obj_per_image*/0, 
				IoU_type_char, prior_dist_type_char, NULL/*C_prior_size*/, NULL/*C_prior_noobj_prob*/, 0/*fit_dim*/, 0/*strict_box_size_association*/, 
				0/*rand_startup*/, 0.0f/*rand_prob_best_box_assoc*/, 0.0f/*rand_prob*/, 0.0f/*min_prior_forced_scaling*/, NULL/*error_scales*/, NULL/*slopes_and_maxes*/, 
				NULL/*param_ind_scales*/, NULL/*IoU_limits*/, NULL/*fit_parts*/, 0/*class_softmax*/, 0/*diff_flag*/, error_type, 0/*no_override*/, 0/*raw_output*/);
			
		}
		
		if(net->y_param->no_override != 1)
		{
			y_param = (yolo_param*)net->y_param;
			y_param->nb_box = nb_box;
			y_param->nb_class = nb_class;
			y_param->nb_param = nb_yolo_param;
			y_param->fit_dim = fit_dim;
			y_param->class_softmax = class_softmax;
			
			free(y_param->prior_size);
			y_param->prior_size = prior_size;
			
			for(i = 0; i < 6; i++)
				for(j = 0; j < 3; j++)
					y_param->slopes_and_maxes_tab[i][j] = slopes_and_maxes[i][j];
			
			if(net->y_param->class_softmax == 0)
				sprintf(display_class_type,"sigmoid-MSE");
			else
				sprintf(display_class_type,"softmax-CrossEntropy");
			
			printf(" WARNING: Overriding the following YOLO parameters from save file:\n");
			printf(" Nboxes = %d, Nclasses = %d, Nparams = %d\n",
			y_param->nb_box, y_param->nb_class, y_param->nb_param);
			printf(" Classification type: %s\n", display_class_type);
			printf(" Nb dim fitted : %d\n\n", y_param->fit_dim);
			printf(" W priors = [");
			for(i = 0; i < net->y_param->nb_box; i++)
				printf("%4.4f ", net->y_param->prior_size[i*3+0]);
			printf("]\n H priors = [");
			for(i = 0; i < net->y_param->nb_box; i++)
				printf("%4.4f ", net->y_param->prior_size[i*3+1]);
			printf("]\n D priors = [");
			for(i = 0; i < net->y_param->nb_box; i++)
				printf("%4.4f ", net->y_param->prior_size[i*3+2]);
			printf("]\n");
			printf("\n Activation slopes and limits: \n   = ");
			for(i = 0; i < 6; i++)
				printf("[%6.2f %6.2f %6.2f]\n     ", 
				net->y_param->slopes_and_maxes_tab[i][0],
				net->y_param->slopes_and_maxes_tab[i][1],
				net->y_param->slopes_and_maxes_tab[i][2]);
			printf("\n");
		}
	}
	else
	{
		if(prior_size != NULL)
			free(prior_size);
	}
	
	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
			previous = NULL;
		else
			previous = net->net_layers[net->nb_layers-1];
		
		conv_create(net, previous, f_size, nb_filters, nb_groups, stride, padding, int_padding, 
			input_shape, activ_type, &bias, dropout_rate, NULL, -1.0f, f, load_optim_state, f_bin);
	}
	else
	{
		nb_parameters = (size_t)(f_size[0] * f_size[1] * f_size[2]) * (input_shape[3]/nb_groups) + 1;
		nb_parameters *= (size_t)nb_filters;
		
		if(f_bin)
			fseek(f, nb_parameters, SEEK_CUR);
		else
			for(i = 0; i < nb_parameters; i++)
				fscanf(f, "%f", &temp_read);
		
		for(int i = 0; i < 3; i++)
			net->skip_in_dims[i] = nb_area_comp(net->skip_in_dims[i], f_size[i], padding[i], int_padding[i], stride[i]);
		
		net->skip_in_dims[3] = nb_filters;
	}
}


void free_conv(layer *current)
{
	c_param = (conv_param*) current->param;
	yolo_param *y_param = NULL;
	
	if(current->previous != NULL && current->previous->output_type == FLAT)
		free(current->prev_dim);
	free(current->output_dim);
	free(c_param->f_size);
	free(c_param->stride);
	free(c_param->padding);
	free(c_param->int_padding);

	#ifdef CUDA	
	if(current->c_network->compute_method == C_CUDA)
	{
		cuda_free_conv(current);
	}
	else
	#endif
	{
		free(current->weights);
		free(current->output);
		free(c_param->im2col_input);
		if(current->dropout_rate > 0.01f)
			free(current->dropout_mask);
		
		if(!current->c_network->inference_only)
		{
			if(current->c_network->use_wema)
				free(current->ema_weights);
			free(current->gradient);
			free(c_param->rotated_filters);
			free(current->delta_o);
			free(c_param->im2col_delta_o);
			if(current->previous != NULL && current->previous->output_type == FLAT)
				free(c_param->temp_delta_o);
			
			free_optimizer_var(current);
		}
	}
	
	if(current->activation_type == YOLO)
	{
		y_param = (yolo_param*) current->activ_param;
		
		#ifdef CUDA
		if(current->c_network->compute_method == C_CUDA)
		{
			cuda_free_yolo_activ_param(current);
		}
		else
		#endif
		{
			free(y_param->slopes_and_maxes_tab[0]);
			free(y_param->slopes_and_maxes_tab);
			free(y_param->prior_size);
			free(y_param->cell_size);
			free(y_param->IoU_monitor);
			free(y_param->target_cell_mask);
			free(y_param->IoU_table);
			free(y_param->dist_prior);
			free(y_param->box_locked);
			free(y_param->box_in_pix);
		}
		//Global YOLO parameters attached to net (from set_yolo_config) are freed by a specific destructor
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}














