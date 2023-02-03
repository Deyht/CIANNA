
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

void pool_define_activation_param(layer *current)
{
	p_param = (pool_param*) current->param;

	//Linear
	current->activ_param = (linear_param*) malloc(sizeof(linear_param));
	((linear_param*)current->activ_param)->size = p_param->nb_area[0] * 
		p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * current->c_network->batch_size;
	((linear_param*)current->activ_param)->dim = ((linear_param*)current->activ_param)->size;
	((linear_param*)current->activ_param)->biased_dim = ((linear_param*)current->activ_param)->dim;
}

void pool_create(network *net, layer* previous, int *pool_size, const char *char_pool_type, int global, float drop_rate)
{
	int k;
	long long int mem_approx = 0;
	layer* current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;

	p_param = (pool_param*) malloc(sizeof(pool_param));

	current->type = POOL;
	//activation type not used for now but could be add for optimization
	load_activ_param(current, NULL);
	current->previous = previous;
	current->dropout_rate = drop_rate;
	
	p_param->nb_area = (int*) calloc(3, sizeof(int));
	p_param->prev_size = (int*) calloc(3, sizeof(int));
	p_param->p_size = (int*) calloc(3, sizeof(int));
	for(k = 0; k < 3; k++)
		p_param->p_size[k] = pool_size[k];
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
				printf("ERROR : Bad network design, no use of two succesiv pooling layer.\n");
				exit(EXIT_FAILURE);
				break;
		}
		
		current->input = previous->output;
	}
	
	if(global)
		for(k = 0; k < 3; k++)
			p_param->p_size[k] = p_param->prev_size[k];
	
	if(p_param->global == 0 && (p_param->prev_size[0] % p_param->p_size[0] != 0 || p_param->prev_size[1] % p_param->p_size[1] || p_param->prev_size[2] % p_param->p_size[2]))
	{
		printf("ERROR : Non global Pool layer can not handle unheaven activation map size for now, please change network architecture.\n");
		exit(EXIT_FAILURE);
	}
	
	for(k = 0; k < 3; k++)
		p_param->nb_area[k] = p_param->prev_size[k] / p_param->p_size[k];
	
	p_param->nb_maps = p_param->prev_depth;
	
	p_param->pool_map = (int*) malloc(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(int));
	mem_approx += p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(int);
	current->output = (float*) malloc(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(float));
	mem_approx += p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(float);
		
	if(drop_rate > 0.01f)
	{
		p_param->dropout_mask = (int*) calloc(p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]), sizeof(int));
		mem_approx += p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * sizeof(int);
	}
	
	current->delta_o = (float*) malloc(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(float));
	mem_approx += p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps 
		* net->batch_size * sizeof(float);
	
	current->param = p_param;
	
	//Linear activation only = no activation
	pool_define_activation_param(current);
	
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
	
	printf("L:%d - Pooling layer layer created, type %s:\n\
\t Input: %dx%dx%dx%d, Output: %dx%dx%dx%d, P. size: %dx%dx%d, Global: %d, dropout rate: %f\n\
\t Approx layer RAM/VRAM requirement: %d MB\n",
		net->nb_layers, s_pool_type, p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
		p_param->prev_depth, p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
		p_param->nb_maps, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2], p_param->global, current->dropout_rate,
		(int)(mem_approx/1000000));
	
}

void pool_save(FILE *f, layer *current, int f_bin)
{
	p_param = (pool_param*) current->param;
	char layer_type = 'P';
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(p_param->p_size, sizeof(int), 3, f);
		fwrite(&p_param->global, sizeof(int), 1, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		print_pool_type(f, p_param->pool_type, f_bin);
	}
	else
	{
		fprintf(f, "P%dx%dx%d:%d_%fd", p_param->p_size[0], p_param->p_size[1], p_param->p_size[2], p_param->global, current->dropout_rate);
		print_pool_type(f, p_param->pool_type, f_bin);
		fprintf(f, "\n\n");
	}
}

void pool_load(network *net, FILE *f, int f_bin)
{
	int p_size[3], global;
	float dropout_rate;
	char s_pool_type[40];
	layer* previous;

	printf("Loading pool layer, L:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(p_size, sizeof(int), 3, f);
		fread(&global, sizeof(int), 1, f);
		fread(&dropout_rate, sizeof(float), 1, f);
		fread(s_pool_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, "%dx%dx%d:%d_%fd%s\n", &p_size[0], &p_size[1], &p_size[2], &global, &dropout_rate, s_pool_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	pool_create(net, previous, p_size, s_pool_type, global, dropout_rate);
	
}







