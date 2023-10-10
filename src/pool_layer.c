
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
static pool_param *p_param;


void print_pool_type(FILE *f, int type, int f_bin)
{
	char temp_string[40];
	
	switch(type)
	{
		default:
		case MAX_pool:
			sprintf(temp_string,"MAX");
			break;
			
		case AVG_pool:
			sprintf(temp_string,"AVG");
			break;
	}
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s", temp_string);
}

void get_string_pool_type(char* str, int type)
{
	switch(type)
	{
		default:
		case MAX_pool:
			sprintf(str,"MAX");
			break;
		
		case AVG_pool:
			sprintf(str,"AVG");
			break;
	}
}

int load_pool_type(const char *type)
{
	if(strcmp(type, "MAX") == 0)
		return MAX_pool;
	else if(strcmp(type, "AVG") == 0)
		return AVG_pool;
	else
		return MAX_pool;
}

void pool_define_activation_param(layer *current, const char *activ)
{
	int size, dim, biased_dim, offset;
	p_param = (pool_param*) current->param;
	
	size = p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * current->c_network->batch_size;
	dim =  p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2];
	biased_dim =  p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2];
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

int pool_create(network *net, layer *previous, int *pool_size, int* stride, int *padding, const char *char_pool_type, const char *activation, int global, float drop_rate)
{
	int k;
	long long int mem_approx = 0;
	layer* current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;

	p_param = (pool_param*) malloc(sizeof(pool_param));
	
	printf("L:%d - CREATING POOL LAYER ...\n", net->nb_layers);

	current->type = POOL;
	//activation type not used for now but could be add for optimization
	load_activ_param(current, activation);
	current->previous = previous;
	current->dropout_rate = drop_rate;
	
	if(current->previous != NULL && current->previous->dropout_rate > 0.01f)
	{
		printf("\nERROR: A pooling layer cannot be set if dropout is used in the previous layer due to problem with weight/output rescaling.\n");
		printf("Consider adding the dropout on the present pooling layer or inserting a non-droping layer between the two.\n");
		exit(EXIT_FAILURE);
	}
	/*
	if(current->previous != NULL && current->previous->type == NORM && current->dropout_rate > 0.01f)
	{
		printf("\nERROR: A normalization layer cannot be followed by a pooling layer with dropout due to problem with weight/output rescaling.\n");
		printf("Consider inserting a non-droping conv layer between the two.\n");
		exit(EXIT_FAILURE);
	}*/
	
	p_param->nb_area = (int*) calloc(3, sizeof(int));
	p_param->prev_size = (int*) calloc(3, sizeof(int));
	p_param->p_size = (int*) calloc(3, sizeof(int));
	p_param->stride = (int*) calloc(3, sizeof(int));
	p_param->padding = (int*) calloc(3, sizeof(int));
	for(k = 0; k < 3; k++)
	{
		if(stride[k] > pool_size[k])
		{
			printf("\nERROR: pool size cannot be smaller than stride size in a given dimension !\n");
			exit(EXIT_FAILURE);
		}
		if(padding[k] > pool_size[k])
		{
			printf("\nERROR: pool size cannot be equal or smaller than padding in a given dimension !\n");
			exit(EXIT_FAILURE);
		}
		p_param->p_size[k] = pool_size[k];
		p_param->stride[k] = stride[k];
		p_param->padding[k] = padding[k];
	}
	p_param->pool_type = load_pool_type(char_pool_type);
	p_param->global = global;
	
	if(previous == NULL)
	{
		//Case of the first layer
		p_param->prev_size[0] = net->in_dims[0];
		p_param->prev_size[1] = net->in_dims[1];
		p_param->prev_size[2] = net->in_dims[2];
		p_param->prev_depth = net->in_dims[3];
		//input pointer must be set at the begining of forward
		current->input = net->input;
		
		//printf("ERROR : Starting with a pooling layer is currently not allowed.\n");
		//exit(EXIT_FAILURE);
	}
	else
	{
		//regular case	
		switch(previous->type)
		{
			default:
			case CONV:
				for(k = 0; k < 3; k++)
					p_param->prev_size[k] = ((conv_param*)previous->param)->nb_area[k];
				p_param->prev_depth =  ((conv_param*)previous->param)->nb_filters;
				break;
			case POOL:
				printf("ERROR: Bad network design, no use of two successive pooling layer.\n");
				exit(EXIT_FAILURE);
				break;
			case NORM:
			case LRN:
				if(previous->previous->type != CONV)
				{
					printf("ERROR: Unsuported layer types stacking.");
					exit(EXIT_FAILURE);
				}
				for(k = 0; k < 3; k++)
					p_param->prev_size[k] = ((conv_param*)previous->previous->param)->nb_area[k];
				p_param->prev_depth = ((conv_param*)previous->previous->param)->nb_filters;
				break;
		}
		
		current->input = previous->output;
	}
	
	if(global)
		for(k = 0; k < 3; k++)
		{
			p_param->p_size[k] = p_param->prev_size[k];
			p_param->stride[k] = p_param->prev_size[k];
			p_param->padding[k] = 0;
		}
	
	for(k = 0; k < 3; k++)
		p_param->nb_area[k] = nb_area_comp(p_param->prev_size[k], p_param->p_size[k], p_param->padding[k], 0, p_param->stride[k]);
	
	p_param->nb_maps = p_param->prev_depth;
	
	current->output = (float*) malloc((size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
		* net->batch_size * sizeof(float));
	mem_approx += (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
		* net->batch_size * sizeof(float);
		
	if(drop_rate > 0.01f)
	{
		p_param->dropout_mask = (float*) calloc(p_param->nb_maps 
			* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size, sizeof(float));
		mem_approx += p_param->nb_maps 
			* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size * sizeof(float);
	}
	
	if(!net->inference_only)
	{
		p_param->pool_map = (int*) malloc((size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
			* net->batch_size * sizeof(int));
		mem_approx += (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
			* net->batch_size * sizeof(int);
	
		current->delta_o = (float*) malloc((size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
			* net->batch_size * sizeof(float));
		mem_approx += (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps 
			* net->batch_size * sizeof(float);
	}
	else
		p_param->pool_map = NULL;
	
	current->param = p_param;
	
	//Linear activation only = no activation
	pool_define_activation_param(current, activation);
	
	p_param = (pool_param*) current->param;
	
	//No weights initialization in a pool layer
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_pool_define(current);
			mem_approx = cuda_convert_pool_layer(current);
			cuda_define_activation(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			pool_define(current);
			define_activation(current);
			break;
		default:
			break;
	}
	
	char s_pool_type[10];
	get_string_pool_type(s_pool_type, p_param->pool_type);
	char activ[40];
	print_string_activ_param(current, activ);
	printf("      Input: %dx%dx%dx%d, Output: %dx%dx%dx%d\n\
      P. size: %dx%dx%d, Stride: %dx%dx%d, padding: %dx%dx%d \n\
      Pool type: %s, Global: %d, Activation: %s, dropout rate: %0.2f\n\
      Approx layer RAM/VRAM requirement: %d MB\n",
		p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
		p_param->prev_depth, p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
		p_param->nb_maps, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2], 
		p_param->stride[0], p_param->stride[1], p_param->stride[2],
		p_param->padding[0], p_param->padding[1], p_param->padding[2],
		s_pool_type, p_param->global, activ, current->dropout_rate,
		(int)(mem_approx/1000000));
	
	return net->nb_layers - 1;
}

void pool_save(FILE *f, layer *current, int f_bin)
{
	p_param = (pool_param*) current->param;
	char layer_type = 'P';
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(p_param->p_size, sizeof(int), 3, f);
		fwrite(p_param->stride, sizeof(int), 3, f);
		fwrite(p_param->padding, sizeof(int), 3, f);
		fwrite(&p_param->global, sizeof(int), 1, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		print_pool_type(f, p_param->pool_type, f_bin);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f, "P%dx%dx%d.%dx%dx%ds%dx%dx%dp%dg_%fd", 
			p_param->p_size[0], p_param->p_size[1], p_param->p_size[2], 
			p_param->stride[0], p_param->stride[1], p_param->stride[2],
			p_param->padding[0], p_param->padding[1], p_param->padding[2],
			p_param->global, current->dropout_rate);
		print_pool_type(f, p_param->pool_type, f_bin);
		fprintf(f, " ");
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n\n");
	}
}

void pool_load(network *net, FILE *f, int f_bin)
{
	int p_size[3], stride[3], padding[3], global;
	float dropout_rate;
	char pool_type[40];
	char activ_type[40];
	layer* previous;

	printf("Loading pool layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(p_size, sizeof(int), 3, f);
		fread(stride, sizeof(int), 3, f);
		fread(padding, sizeof(int), 3, f);
		fread(&global, sizeof(int), 1, f);
		fread(&dropout_rate, sizeof(float), 1, f);
		fread(pool_type, sizeof(char), 40, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, "%dx%dx%d.%dx%dx%ds%dx%dx%dp%dg_%fd%s %s\n", 
			&p_size[0], &p_size[1], &p_size[2], 
			&stride[0], &stride[1], &stride[2],
			&padding[0], &padding[1], &padding[2],
			&global, &dropout_rate, pool_type, activ_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	pool_create(net, previous, p_size, stride, padding, pool_type, activ_type, global, dropout_rate);
	
}







