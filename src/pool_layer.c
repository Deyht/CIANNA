
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



void pool_create(network *net, layer* previous, int pool_size)
{
	layer* current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;

	p_param = (pool_param*) malloc(sizeof(pool_param));

	current->type = POOL;
	//activation type not used for now but could be add for optimization
	current->activation_type = LINEAR;
	current->previous = previous;
	
	p_param->p_size = pool_size;
	
	
	if(previous == NULL)
	{
		//Case of the first layer
		p_param->prev_size_w = net->input_width;
		p_param->prev_size_h = net->input_height;
		p_param->prev_depth = net->input_depth;
		//input pointer must be set at the begining of forward
		current->input = net->input;
		
		printf("ERROR : Starting with a pooling layer is currently not allowed.\n");
		exit(EXIT_FAILURE);
	}
	else
	{
		//regular case	
		switch(previous->type)
		{
			case CONV:
			default:
				p_param->prev_size_w = ((conv_param*)previous->param)->nb_area_w;
				p_param->prev_size_h = ((conv_param*)previous->param)->nb_area_h;
				p_param->prev_depth =  ((conv_param*)previous->param)->nb_filters;
				break;
			case POOL:
				printf("ERROR : Bad network design, no use of two succesiv pooling layer.\n");
				exit(EXIT_FAILURE);
				break;
			
		}
		current->input = previous->output;
	}
	
	if(p_param->prev_size_w % pool_size != 0 || p_param->prev_size_h % pool_size)
	{
		printf("ERROR : Pool layer can not handle unheaven activation map size for now, please change network architecture.\n");
		exit(EXIT_FAILURE);
	}
	
	p_param->nb_area_w = p_param->prev_size_w / pool_size;
	p_param->nb_area_h = p_param->prev_size_h / pool_size;
	p_param->nb_maps = p_param->prev_depth;
	
	if(p_param->nb_area_w != p_param->nb_area_h)
	{
		printf("ERROR : Pool layer can not hander non square map size for now, please change network architecture.\n");
		exit(EXIT_FAILURE);
	}
	
	p_param->pool_map = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps 
		* net->batch_size * sizeof(real));
	current->output = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps 
		* net->batch_size * sizeof(real));
	
	current->delta_o = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps 
		* net->batch_size * sizeof(real));
	p_param->temp_delta_o = (real*) malloc(p_param->prev_size_w * p_param->prev_size_h 
		* p_param->prev_depth * net->batch_size * sizeof(real));
	
	//No activation for this layer for now
	current->activ_param = NULL;
	
	current->param = p_param;
	
	//No weights initialization in a pool layer
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_pool_define(current);
			cuda_define_activation(current);
			cuda_convert_pool_layer(current);
			#endif
			break;
			
		default:
			break;
	}
	
	#ifndef CUDA
	printf("ERROR : Non CUDA compute not enable right know !\n");
	exit(EXIT_FAILURE);
	#endif
	
	
}

void pool_save(FILE *f, layer *current)
{
	p_param = (pool_param*) current->param;
	
	fprintf(f, "P%d\n", p_param->p_size);
	fprintf(f, "\n");
}

void pool_load(network *net, FILE *f)
{
	int p_size;
	layer* previous;

	printf("Loading pool layer, L:%d\n", net->nb_layers);
	
	fscanf(f, "%d", &p_size);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	pool_create(net, previous, p_size);
	
}










