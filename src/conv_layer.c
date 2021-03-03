
/*
	Copyright (C) 2020 David Cornu
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

//private
int nb_area_comp(int size);
void conv_define_activation_param(layer *current);


//compute the number of area to convolve regarding the filters parameters
int nb_area_comp(int size)
{
	if((size + c_param->padding*2 - c_param->f_size)%c_param->stride != 0)
	{
		printf("Warning: unable to divide current input volume into \
an integer number of convolution regions\n \
This might produce unstable results !\n\n");
	}		
	return (size + c_param->padding*2 - c_param->f_size) / c_param->stride + 1;
}


void conv_define_activation_param(layer *current)
{
	c_param = (conv_param*) current->param;
	
	switch(current->activation_type)
	{
		case RELU:
			current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
			((ReLU_param*)current->activ_param)->size = c_param->nb_area_w * 
				c_param->nb_area_h * c_param->nb_filters * current->c_network->batch_size;
			((ReLU_param*)current->activ_param)->dim = ((ReLU_param*)current->activ_param)->size;
			((ReLU_param*)current->activ_param)->biased_dim = ((ReLU_param*)current->activ_param)->dim;
			((ReLU_param*)current->activ_param)->leaking_factor = 0.01;
			c_param->bias_value = 0.1;
			break;
			
		case LOGISTIC:
			current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
			((logistic_param*)current->activ_param)->size = c_param->nb_area_w 
				* c_param->nb_area_h *  c_param->nb_filters * current->c_network->batch_size;
			((logistic_param*)current->activ_param)->dim = ((logistic_param*)current->activ_param)->size;
			((logistic_param*)current->activ_param)->biased_dim = ((logistic_param*)current->activ_param)->dim;
			((logistic_param*)current->activ_param)->beta = 1.0;
			((logistic_param*)current->activ_param)->saturation = 10.0;
			c_param->bias_value = -1.0;
			break;
			
		case SOFTMAX:
			//could add a dedicated cuda kernel or conversion to compute it
			printf("Softmax activation function must not be used for conv layer\n");
			/*
			current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
			((softmax_param*)current->activ_param)->block_size = c_param->nb_area_w * c_param->nb_area_h;
			((softmax_param*)current->activ_param)->nb_blocks = c_param->nb_filters;
			*/
			exit(EXIT_FAILURE);
			break;
			
		case YOLO:
			current->activ_param = (yolo_param*) malloc(sizeof(yolo_param));
			if(current->c_network->yolo_nb_box*(5+current->c_network->yolo_nb_class
					+ current->c_network->yolo_nb_param) != c_param->nb_filters)
			{
				printf("ERROR: Nb filters size mismatch in YOLO dimensions!\n");
				exit(EXIT_FAILURE);
			}
			((yolo_param*)current->activ_param)->nb_box = current->c_network->yolo_nb_box;
			((yolo_param*)current->activ_param)->nb_class = current->c_network->yolo_nb_class;
			((yolo_param*)current->activ_param)->nb_param = current->c_network->yolo_nb_param;
			//Priors table must be sent to GPU memory if C_CUDA
			((yolo_param*)current->activ_param)->prior_w = current->c_network->yolo_prior_w;
			((yolo_param*)current->activ_param)->prior_h = current->c_network->yolo_prior_h;
			((yolo_param*)current->activ_param)->size = c_param->nb_area_w 
				* c_param->nb_area_h *  c_param->nb_filters * current->c_network->batch_size;
			((yolo_param*)current->activ_param)->dim = ((yolo_param*)current->activ_param)->size;
			((yolo_param*)current->activ_param)->biased_dim = ((yolo_param*)current->activ_param)->dim;
			((yolo_param*)current->activ_param)->cell_w = current->c_network->input_width / c_param->nb_area_w;
			((yolo_param*)current->activ_param)->cell_h = current->c_network->input_height / c_param->nb_area_h;
			((yolo_param*)current->activ_param)->beta = 1.0;
			((yolo_param*)current->activ_param)->saturation = 10.0;
			c_param->bias_value = 0.1;
			break;
			
		case LINEAR:
		default:
			current->activ_param = (linear_param*) malloc(sizeof(linear_param));
			((linear_param*)current->activ_param)->size = c_param->nb_area_w * 
				c_param->nb_area_h * c_param->nb_filters * current->c_network->batch_size;
			((linear_param*)current->activ_param)->dim = ((linear_param*)current->activ_param)->size;
			((linear_param*)current->activ_param)->biased_dim = ((linear_param*)current->activ_param)->dim;
			c_param->bias_value = 0.5;
			break;
	
	}
}


//Used to allocate a convolutionnal layer
void conv_create(network *net, layer *previous, int f_size, int nb_filters, int stride, int padding, int activation, FILE *f_load)
{
	int i, j;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	//allocate the space holder for conv layer parameters
	c_param = (conv_param*) malloc(sizeof(conv_param));
	
	//define the parameters values
	current->type = CONV;
	current->activation_type = activation;
	c_param->f_size = f_size;
	c_param->stride = stride;
	c_param->padding = padding;
	c_param->nb_filters = nb_filters;
	
	current->previous = previous;
	
	//compute the number of areas to be convolved in the input image
	if(previous == NULL)
	{
		//Case of the first layer
		c_param->prev_size_w = net->input_width;
		c_param->prev_size_h = net->input_height;
		c_param->prev_depth = net->input_depth;
		c_param->flat_f_size = (f_size * f_size * net->input_depth + 1);
		//input pointer must be set at the begining of forward
		current->input = net->input;
	}
	else
	{
		//regular case	
		switch(previous->type)
		{
			case POOL:
				c_param->prev_size_w = ((pool_param*)previous->param)->nb_area_w;
				c_param->prev_size_h = ((pool_param*)previous->param)->nb_area_h;
				c_param->prev_depth =  ((pool_param*)previous->param)->nb_maps;
				c_param->flat_f_size = (f_size * f_size * ((pool_param*)previous->param)->nb_maps + 1);
				((pool_param*)previous->param)->next_layer_type = current->type;
				break;
		
			case CONV:
			default:
				c_param->prev_size_w = ((conv_param*)previous->param)->nb_area_w;
				c_param->prev_size_h = ((conv_param*)previous->param)->nb_area_h;
				c_param->prev_depth =  ((conv_param*)previous->param)->nb_filters;
				c_param->flat_f_size = (f_size * f_size * ((conv_param*)previous->param)->nb_filters + 1);
				break;
		}
		current->input = (float*) calloc(c_param->prev_depth * (c_param->prev_size_w * c_param->prev_size_h) *
		net->batch_size, sizeof(float));
		
	}
	
	c_param->nb_area_w = nb_area_comp(c_param->prev_size_w);
	c_param->nb_area_h = nb_area_comp(c_param->prev_size_h);
	
	printf("Layer output: %d %d\n", c_param->nb_area_w,c_param->nb_area_h);
	
	//allocate all the filters in a flatten table. One filter is continuous. (include bias weight)
	c_param->filters = (float*) malloc(nb_filters * c_param->flat_f_size * sizeof(float));
	//allocate the update for the filters
	c_param->update = (float*) calloc(nb_filters * c_param->flat_f_size, sizeof(float));
	
	c_param->rotated_filters = (float*) malloc(nb_filters * (c_param->flat_f_size-1) * sizeof(float));
	
	
	//allocate the resulting flatten activation map regarding the batch size
	//Activation maps are not continuous for each image : 
	//		A1_im1, A1_im2, A1_im3, ... , A2_im1, A2_im2, A2_im3, ... 
	current->output = (float*) calloc( c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) *
		net->batch_size, sizeof(float));
	//allocate output error comming from next layer
	current->delta_o = (float*) calloc( c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) * 
		net->batch_size, sizeof(float));
	
	//temporary output error used for format conversion
	c_param->temp_delta_o = (float*) calloc( c_param->prev_depth * (c_param->prev_size_w 
		* c_param->prev_size_h) * current->c_network->batch_size, sizeof(float));
		
	//allocate the im2col input flatten table regarding the batch size
	c_param->im2col_input = (float*) calloc( (c_param->flat_f_size * c_param->nb_area_w * c_param->nb_area_h)
		* net->batch_size, sizeof(float));
	
	c_param->im2col_delta_o = (float*) calloc( (c_param->prev_size_w*c_param->prev_size_h) * 
		/* flat_filter*/(f_size*f_size*c_param->nb_filters) * net->batch_size,  sizeof(float));

	current->param = c_param;

	conv_define_activation_param(current);
	
	if(current->previous == NULL)
		((conv_param*)current->param)->bias_value = net->input_bias;
	
	//set bias value for the current layer, this value will not move during training
	for(i = 1; i <= c_param->nb_area_w * c_param->nb_area_h * net->batch_size; i++)
		((float*)c_param->im2col_input)[i*(c_param->flat_f_size) - 1] = c_param->bias_value;
	
	if(f_load == NULL)
	{
		printf("Xavier init\n");
		xavier_normal(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0, 0.0);
	}
	else
	{
		for(i = 0; i < nb_filters; i++)
			for(j = 0; j < c_param->flat_f_size; j++)
				 fscanf(f_load, "%f", &(((float*)c_param->filters)[i*c_param->flat_f_size + j]));
	}
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_conv_define(current);
			cuda_convert_conv_layer(current);
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
	
	char activ[10];
	get_string_activ_param(activ, current->activation_type);
	printf("L:%d - Convolutional layer created:\n \
Input: %dx%dx%d, Filters: %dx%dx%d, Output: %dx%dx%d \n \
Activation: %s, Stride: %d, padding: %d\n",
		net->nb_layers, c_param->prev_size_w, c_param->prev_size_h, 
		c_param->prev_depth, c_param->f_size, c_param->f_size, c_param->nb_filters,
		c_param->nb_area_w, c_param->nb_area_h, c_param->nb_filters,
		activ, c_param->stride, c_param->padding);
	
	if(net->compute_method == C_CUDA && net->use_cuda_TC)
	{
	
		if(c_param->flat_f_size % 8 != 0 
				|| current->c_network->batch_size * (c_param->nb_area_w*c_param->nb_area_h) % 8 != 0 
				|| c_param->nb_filters % 8 != 0)
			printf("Warning : Forward gemm fallback to non TC version due to layer size mismatch\n");
			
		if(current->previous != NULL &&
				( c_param->prev_depth % 8 != 0 
				|| c_param->prev_size_w * c_param->prev_size_h * current->c_network->batch_size % 8 != 0 
				|| c_param->f_size * c_param->f_size * c_param->nb_filters % 8 != 0))
			printf("Warning : Backprop gemm fallback to non TC version due to layer size mismatch\n");

		if( c_param->flat_f_size % 8 != 0 
				|| c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size % 8 != 0 
				|| c_param->nb_filters % 8 != 0)
			printf("Warning : Weights update gemm fallback to non TC version due to layer size mismatch\n");
	}
	
}


void conv_save(FILE *f, layer *current)
{
	int i, j;
	void *host_filters = NULL;

	c_param = (conv_param*)current->param;	
	
	fprintf(f,"C");
	fprintf(f, "%df%d.%ds%dp", c_param->nb_filters, c_param->f_size, c_param->stride, c_param->padding);
	print_activ_param(f, current->activation_type);
	fprintf(f, "\n");
	
	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_filters = (float*) malloc(c_param->nb_filters*c_param->flat_f_size*sizeof(float));
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_get_table_FP32(current->c_network, (float*)c_param->filters, (float*)host_filters, 
					c_param->nb_filters*c_param->flat_f_size);
				break;
			case 1:
				cuda_get_table_FP32(current->c_network, (float*)c_param->FP32_filters, (float*)host_filters, 
					c_param->nb_filters*c_param->flat_f_size);
				break;
		
		}
		#endif
	}
	else
	{
		host_filters = c_param->filters;
	}
	
	for(i = 0; i < c_param->nb_filters; i++)
	{
		for(j = 0; j < c_param->flat_f_size; j++)
			fprintf(f, "%g ", ((float*)host_filters)[i*c_param->flat_f_size + j]);
		fprintf(f,"\n");	
	}
	fprintf(f, "\n");
	
	if(current->c_network->compute_method == C_CUDA)
		free(host_filters);
}

void conv_load(network *net, FILE *f)
{
	int nb_filters, f_size, stride, padding;
	char activ_type[20];
	layer *previous;
	
	printf("Loading conv layer, L:%d\n", net->nb_layers);
	
	fscanf(f, "%df%d.%ds%dp%s\n", &nb_filters, &f_size, &stride, &padding, activ_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	conv_create(net, previous, f_size, nb_filters, stride, padding, load_activ_param(activ_type), f);
}























