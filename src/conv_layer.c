
/*
	Copyright (C) 2023 David Cornu
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

//##############################
//       Local variables
//##############################
static conv_param *c_param;

//public are in prototypes.h

//compute the number of area to convolve regarding the filters parameters
int nb_area_comp(int size, int f_size, int padding, int int_padding, int stride)
{
	if((size + padding*2 - f_size)%stride != 0)
	{
		printf(" WARNING: unable to divide current input volume into \
an integer number of conv/pool regions\n\
 This might produce unstable results !\n\n");
	}		
	return (size + (size-1)*int_padding + padding*2 - f_size) / stride + 1;
}


void conv_define_activation_param(layer *current, const char *activ)
{
	int size, dim, biased_dim, offset;
	c_param = (conv_param*) current->param;
	
	size = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * c_param->nb_filters * current->c_network->batch_size;
	dim  =  c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
	biased_dim = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
	offset = current->c_network->batch_size;
	
	switch(current->activation_type)
	{
		case RELU:
			set_relu_activ(current, size, biased_dim, dim, offset, activ);
			break;
			
		case LOGISTIC:
			set_logistic_activ(current, size, dim, biased_dim, offset, activ);
			break;
			
		case SOFTMAX:
			set_softmax_activ(current, size, dim, biased_dim, offset);
			break;
			
		case YOLO:
			set_yolo_activ(current);
			break;
			
		case LINEAR:
		default:
			set_linear_activ(current, size, dim, biased_dim, offset);
			break;
	}
}


//Used to allocate a convolutionnal layer
int conv_create(network *net, layer *previous, int *f_size, int nb_filters, int *stride, int *padding, 
	int *int_padding, int *in_shape, const char *activation, float *bias, float drop_rate, 
	const char *init_fct, float init_scaling, FILE *f_load, int f_bin)
{
	size_t i;
	int j, k;
	long long int mem_approx = 0;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING CONVOLUTIONAL LAYER ...\n", net->nb_layers);
	
	//allocate the space holder for conv layer parameters
	c_param = (conv_param*) malloc(sizeof(conv_param));
	
	//define the parameters values
	current->type = CONV;
	load_activ_param(current, activation);
	
	current->frozen = 0;
	c_param->nb_area = (int*) calloc(3, sizeof(int));
	c_param->prev_size = (int*) calloc(3, sizeof(int));
	c_param->f_size = (int*) calloc(3, sizeof(int));
	c_param->stride = (int*) calloc(3, sizeof(int));
	c_param->padding = (int*) calloc(3, sizeof(int));
	c_param->int_padding = (int*) calloc(3, sizeof(int));
	
	for(k = 0; k < 3; k++)
	{
		if(stride[k] > f_size[k])
		{
			printf("\nERROR: filter size cannot be smaller than stride size in a given dimension !\n");
			exit(EXIT_FAILURE);
		}
		c_param->f_size[k] = f_size[k];
		c_param->stride[k] = stride[k];
		c_param->padding[k] = padding[k];
		c_param->int_padding[k] = int_padding[k];
	}
	c_param->nb_filters = nb_filters;
	current->dropout_rate = drop_rate;
	
	current->previous = previous;
	
	//compute the number of areas to be convolved in the input image
	if(previous == NULL)
	{
		//Case of the first layer
		c_param->prev_size[0] = net->in_dims[0];
		c_param->prev_size[1] = net->in_dims[1];
		c_param->prev_size[2] = net->in_dims[2];
		c_param->prev_depth = net->in_dims[3];
		c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] * net->in_dims[3] + 1);
		//input pointer must be set at the begining of forward
		current->input = net->input;
	}
	else
	{
		//regular case	
		switch(previous->type)
		{
			case DENSE:
				if(previous->dropout_rate)
				{
					printf("ERROR: dropout on a dense layer output used as input for a conv layer is not authorized.\n\n");
					exit(EXIT_FAILURE);
				}
				if(in_shape == NULL)
				{
					printf("ERROR: dense to conv conversion requires input_shape to be defined in conv_layer.\n\n");
					exit(EXIT_FAILURE); //add verification of shape match between dense size and conv sizes
				}
				for(k = 0; k < 3; k++)
					c_param->prev_size[k] = in_shape[k];
				c_param->prev_depth = in_shape[3];
				c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] * in_shape[3] + 1);
				break;
		
			case POOL:
				for(k = 0; k < 3; k++)
					c_param->prev_size[k] = ((pool_param*)previous->param)->nb_area[k];
				c_param->prev_depth =  ((pool_param*)previous->param)->nb_maps;
				c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] * ((pool_param*)previous->param)->nb_maps + 1);
				break;
		
			case CONV:
			default:
				for(k = 0; k < 3; k++)
					c_param->prev_size[k] = ((conv_param*)previous->param)->nb_area[k];
				c_param->prev_depth =  ((conv_param*)previous->param)->nb_filters;
				c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] * ((conv_param*)previous->param)->nb_filters + 1);
				break;
			
			case NORM:
			case LRN:
				switch(previous->previous->type)
				{
					case DENSE:
						if(in_shape == NULL)
						{
							printf("ERROR: dense to conv conversion requires input_shape to be defined in conv_layer.\n\n");
							exit(EXIT_FAILURE); //add verification of shape match between dense size and conv sizes
						}
						for(k = 0; k < 3; k++)
							c_param->prev_size[k] = in_shape[k];
						c_param->prev_depth = in_shape[3];
						c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] * in_shape[3] + 1);
						break;
				
					case POOL:
						for(k = 0; k < 3; k++)
							c_param->prev_size[k] = ((pool_param*)previous->previous->param)->nb_area[k];
						c_param->prev_depth =  ((pool_param*)previous->previous->param)->nb_maps;
						c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] 
							* ((pool_param*)previous->previous->param)->nb_maps + 1);
						break;
				
					case CONV:
					default:
						for(k = 0; k < 3; k++)
							c_param->prev_size[k] = ((conv_param*)previous->previous->param)->nb_area[k];
						c_param->prev_depth = ((conv_param*)previous->previous->param)->nb_filters;
						c_param->flat_f_size = (f_size[0] * f_size[1] * f_size[2] 
							* ((conv_param*)previous->previous->param)->nb_filters + 1);
						break;
				}
				break;
		}
		current->input = previous->output;
	}
	
	c_param->TC_padding = 0;
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC != FP32C_FP32A)
		c_param->TC_padding = 8 - c_param->flat_f_size % 8;
	#endif
	
	for(k = 0; k < 3; k++)
		c_param->nb_area[k] = nb_area_comp(c_param->prev_size[k], c_param->f_size[k], 
			c_param->padding[k], c_param->int_padding[k], c_param->stride[k]);
	
	//printf("Layer output: %d %d %d\n", c_param->nb_area[0],c_param->nb_area[1],c_param->nb_area[2]);
	
	//allocate all the filters in a flatten table. One filter is continuous. (include bias weight)
	c_param->filters = (float*) calloc(nb_filters * (c_param->flat_f_size + c_param->TC_padding), sizeof(float));
	mem_approx += nb_filters * (c_param->flat_f_size + c_param->TC_padding) * sizeof(float);
	
	if(drop_rate > 0.01f)
	{
		c_param->dropout_mask = (float*) calloc(c_param->nb_filters 
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
			* net->batch_size, sizeof(float));
		mem_approx += c_param->nb_filters 
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
			* net->batch_size * sizeof(float);
	}
	
	//allocate the resulting flatten activation map regarding the batch size
	//Activation maps are not continuous for each image : 
	//		A1_im1, A1_im2, A1_im3, ... , A2_im1, A2_im2, A2_im3, ... 
	
	//printf("%d %d %d %d\n", c_param->nb_filters, c_param->nb_area_w, c_param->nb_area_h, net->batch_size);
	current->output = (float*) calloc( c_param->nb_filters 
		* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
		* net->batch_size, sizeof(float));
	mem_approx += c_param->nb_filters * (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
		* net->batch_size * sizeof(float);
		
	//allocate the im2col input flatten table regarding the batch size
	c_param->im2col_input = (float*) calloc((c_param->flat_f_size + c_param->TC_padding) 
		* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2])
		* net->batch_size, sizeof(float));
	mem_approx += (c_param->flat_f_size + c_param->TC_padding) 
		* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2])
		* net->batch_size * sizeof(float);
	
	if(!net->inference_only)
	{
		//allocate the update for the filters
		c_param->update = (float*) calloc(nb_filters * (c_param->flat_f_size + c_param->TC_padding), sizeof(float));
		mem_approx += nb_filters * (c_param->flat_f_size + c_param->TC_padding) * sizeof(float);
	
		c_param->rotated_filters = (float*) calloc(nb_filters * (c_param->flat_f_size-1), sizeof(float));
		mem_approx += nb_filters * (c_param->flat_f_size-1) * sizeof(float);
	
		//allocate output error comming from next layer
		current->delta_o = (float*) calloc( c_param->nb_filters 
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
			* net->batch_size, sizeof(float));
		mem_approx += c_param->nb_filters 
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
			* net->batch_size * sizeof(float);
	
		//temporary output error used for dense to conv link backprop
		if(previous != NULL && previous->type == DENSE)
		{
			c_param->temp_delta_o = (float*) calloc( c_param->prev_depth * (size_t)(c_param->prev_size[0] 
				* c_param->prev_size[1] * c_param->prev_size[2]) * current->c_network->batch_size, sizeof(float));
			mem_approx += (c_param->prev_depth * (size_t)(c_param->prev_size[0] 
				* c_param->prev_size[1] * c_param->prev_size[2]) * current->c_network->batch_size * sizeof(float));
		}
	
		c_param->im2col_delta_o = (float*) calloc( (long long int) net->batch_size 
			* (size_t)(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) 
			* /* flat_filter*/(f_size[0]*f_size[1]*f_size[2]*c_param->nb_filters), sizeof(float));
		mem_approx += (long long int) (net->batch_size 
			* (size_t)(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) 
			* /* flat_filter*/(f_size[0]*f_size[1]*f_size[2]*c_param->nb_filters)) * sizeof(float);
	}
	
	current->nb_params = nb_filters * c_param->flat_f_size;

	current->param = c_param;
	
	conv_define_activation_param(current, activation);
	
	c_param = (conv_param*)current->param;
	
	if(bias != NULL)
		current->bias_value = *bias;
	
	if(current->previous == NULL)
		current->bias_value = net->input_bias;
	
	/*if(current->previous->drop_rate > 0.01f)
		current->bias_value = 0.0f;
	*/
	//set bias value for the current layer, this value will not move during training
	for(i = 0; i < (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size; i++)
		((float*)c_param->im2col_input)[i*(c_param->flat_f_size+c_param->TC_padding) 
			+ c_param->flat_f_size - 1] = current->bias_value;
	
	if(f_load == NULL)
	{	
		if(init_scaling < 0)
			init_scaling = 1.0f;
		
		//Should add a defaut weight init depending on the activation function
		switch(get_init_type(init_fct))
		{
			default:
			case N_XAVIER:
				xavier_normal(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
			case U_XAVIER:
				xavier_uniform(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
			case N_LECUN:
				lecun_normal(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
			case U_LECUN:
				lecun_uniform(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
			case N_RAND:
				rand_normal(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
			case U_RAND:
				rand_uniform(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0, c_param->TC_padding, init_scaling);
				break;
		}
	}
	else
	{
		if(f_bin)
		{
			for(j = 0; j < nb_filters; j++)
				fread(&(((float*)c_param->filters)[j*(c_param->flat_f_size+c_param->TC_padding)]), 
					sizeof(float), c_param->flat_f_size, f_load);
		}
		else
		{
			for(j = 0; j < nb_filters; j++)
			{
				for(k = 0; k < c_param->flat_f_size; k++)
					fscanf(f_load, "%f", &(((float*)c_param->filters)[j*(c_param->flat_f_size+c_param->TC_padding) + k]));
			}
		}	
	}
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_conv_define(current);
			mem_approx = cuda_convert_conv_layer(current);
			cuda_define_activation(current);
			#endif
			break;
		case C_BLAS:
			#ifdef BLAS
			blas_conv_define(current);
			define_activation(current);
			#endif
			break;
		case C_NAIV:
			naiv_conv_define(current);
			define_activation(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	print_string_activ_param(current, activ);
	printf("      Input: %dx%dx%dx%d, Filters: %df %dx%dx%dx%d, Output: %dx%dx%dx%d \n\
      Stride: %d:%d:%d, padding: %d:%d:%d, int_padding: %d:%d:%d,  \n\
      Activation: %s, Bias: %0.2f, dropout rate: %0.2f\n\
      Nb. weights: %d, Approx layer RAM/VRAM requirement: %d MB\n",
		c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
		c_param->prev_depth, c_param->nb_filters, 
		c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], c_param->prev_depth, 
		c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], c_param->nb_filters,
		c_param->stride[0], c_param->stride[1], c_param->stride[2], 
		c_param->padding[0], c_param->padding[1],  c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		activ, current->bias_value, current->dropout_rate,
		nb_filters * c_param->flat_f_size, (int)(mem_approx/1000000));
	net->total_nb_param += nb_filters * c_param->flat_f_size;
	net->memory_footprint += mem_approx;
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC != FP32C_FP32A)
	{
	
		if((c_param->flat_f_size + c_param->TC_padding) % 8 != 0 
				|| current->c_network->batch_size * (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]) % 8 != 0 
				|| c_param->nb_filters % 8 != 0)
			printf(" WARNING: Forward gemm TC data misalignement due to layer size mismatch\n");
		if(current->previous != NULL &&
				( c_param->prev_depth % 8 != 0 
				|| c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2] * current->c_network->batch_size % 8 != 0 
				|| c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2] * c_param->nb_filters % 8 != 0))
			printf(" WARNING: Backprop gemm TC data misalignment due to layer size mismatch\n");
		if( (c_param->flat_f_size + c_param->TC_padding) % 8 != 0 
				|| c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size % 8 != 0 
				|| c_param->nb_filters % 8 != 0)
			printf(" WARNING: Weights update gemm TC data misalignment due to layer size mismatch\n");
	}
	#endif

	return net->nb_layers - 1;
}


void conv_save(FILE *f, layer *current, int f_bin)
{
	int i, j;
	void *host_filters = NULL;
	char layer_type = 'C';

	c_param = (conv_param*)current->param;	
	yolo_param *y_param = NULL;
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&c_param->nb_filters, sizeof(int), 1, f);
		fwrite(c_param->f_size, sizeof(int), 3, f);
		fwrite(c_param->stride, sizeof(int), 3, f);
		fwrite(c_param->padding, sizeof(int), 3, f);
		fwrite(c_param->int_padding, sizeof(int), 3, f);
		fwrite(c_param->prev_size, sizeof(int), 3, f);
		fwrite(&c_param->prev_depth, sizeof(int), 1, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		fwrite(&current->bias_value, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
		if(current->activation_type == YOLO)
		{
			y_param = current->c_network->y_param;
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
		fprintf(f, "%df%dx%dx%d.%dx%dx%ds%dx%dx%dp%dx%dx%dip%dx%dx%dx%didim%fd%fb", c_param->nb_filters, 
			c_param->f_size[0], c_param->f_size[1], c_param->f_size[2],
			c_param->stride[0], c_param->stride[1], c_param->stride[2], 
			c_param->padding[0], c_param->padding[1], c_param->padding[2],
			c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2], 
			c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
			c_param->prev_depth, current->dropout_rate, current->bias_value);
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n");
		if(current->activation_type == YOLO)
		{
			y_param = (yolo_param*)current->c_network->y_param;
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
	
	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_filters = (float*) calloc(c_param->nb_filters*(c_param->flat_f_size + c_param->TC_padding), sizeof(float));
		switch(current->c_network->cu_inst.use_cuda_TC)
		{
			default:
			case FP32C_FP32A:
			case TF32C_FP32A:
				cuda_get_table_FP32((float*)c_param->filters, (float*)host_filters, 
					c_param->nb_filters*(c_param->flat_f_size + c_param->TC_padding));
				break;
			
			case FP16C_FP32A:
			case FP16C_FP16A:
				cuda_get_table_FP32((float*)c_param->FP32_filters, (float*)host_filters, 
					c_param->nb_filters*(c_param->flat_f_size + c_param->TC_padding));
				break;
			
			case BF16C_FP32A:
				cuda_get_table_FP32((float*)c_param->FP32_filters, (float*)host_filters, 
					c_param->nb_filters*(c_param->flat_f_size + c_param->TC_padding));
				break;
		}
		#endif
	}
	else
	{
		host_filters = c_param->filters;
	}
	
	if(f_bin)
	{
		for(i = 0; i < c_param->nb_filters; i++)
			fwrite(&((float*)host_filters)[i*(c_param->flat_f_size+c_param->TC_padding)], sizeof(float), c_param->flat_f_size, f);
	}
	else
	{
		for(i = 0; i < c_param->nb_filters; i++)
		{
			for(j = 0; j < c_param->flat_f_size; j++)
				fprintf(f, "%g ", ((float*)host_filters)[i*(c_param->flat_f_size+c_param->TC_padding) + j]);
			fprintf(f,"\n");	
		}
		fprintf(f, "\n");
	}
	
	if(current->c_network->compute_method == C_CUDA)
		free(host_filters);
}

void conv_load(network *net, FILE *f, int f_bin)
{
	int i, j;
	int nb_filters;
	int f_size[3], stride[3], padding[3], int_padding[3], input_shape[4];
	float dropout_rate, bias;
	
	int nb_box, nb_class, nb_param, fit_dim, class_softmax;
	float *prior_size = NULL;
	float slopes_and_maxes[6][3];
	
	char activ_type[40];
	char display_class_type[60];
	layer *previous;
	yolo_param* y_param = NULL;
	
	printf("Loading conv layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&nb_filters, sizeof(int), 1, f);
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
			fread(&nb_param, sizeof(int), 1, f);
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
		fscanf(f, "%df%dx%dx%d.%dx%dx%ds%dx%dx%dp%dx%dx%dip%dx%dx%dx%didim%fd%fb%s\n", &nb_filters, 
			&f_size[0], &f_size[1], &f_size[2],
		 	&stride[0], &stride[1], &stride[2], 
		 	&padding[0], &padding[1], &padding[2], 
		 	&int_padding[0], &int_padding[1], &int_padding[2],
			&input_shape[0], &input_shape[1], &input_shape[2], &input_shape[3], 
			&dropout_rate, &bias, activ_type);
		if(strcmp(activ_type, "YOLO") == 0)
		{
			fscanf(f, "%d %d %d %d %d\n", &nb_box, &nb_class, &nb_param, &fit_dim, &class_softmax);
			prior_size = (float*) calloc(3*nb_box, sizeof(float));
			for(i = 0; i < 3; i++)
				for(j = 0; j < nb_box; j++)
					fscanf(f, "%f", &(prior_size[j*3+i]));
		}
	}
	
	if(strncmp(activ_type, "YOLO", 4) == 0 && net->y_param->no_override != 1)
	{
		y_param = (yolo_param*)net->y_param;
		y_param->nb_box = nb_box;
		y_param->nb_class = nb_class;
		y_param->nb_param = nb_param;
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
	else
	{
		if(prior_size != NULL)
			free(prior_size);
	}	
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	conv_create(net, previous, f_size, nb_filters, stride, padding, int_padding, 
		input_shape, activ_type, &bias, dropout_rate, NULL, -1.0f, f, f_bin);
}














