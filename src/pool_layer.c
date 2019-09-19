#include "prototypes.h"


//##############################
//       Local variables
//##############################
static pool_param *p_param;



void pool_create(layer* previous, int pool_size)
{
	layer* current;
	
	current = (layer*) malloc(sizeof(layer));
	net_layers[nb_layers] = current;
	nb_layers++;

	p_param = (pool_param*) malloc(sizeof(pool_param));

	current->type = POOL;
	//activation type not used for now but could be add for optimization
	current->activation_type = LINEAR;
	current->previous = previous;
	
	p_param->p_size = pool_size;
	
	
	if(previous == NULL)
	{
		//Case of the first layer
		p_param->prev_size_w = input_width;
		p_param->prev_size_h = input_height;
		p_param->prev_depth = input_depth;
		//input pointer must be set at the begining of forward
		current->input = input;
		
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
	
	p_param->pool_map = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps * batch_size 
		* sizeof(real));
	current->output = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps * batch_size 
		* sizeof(real));
	
	current->delta_o = (real*) malloc(p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps * batch_size 
		* sizeof(real));
	p_param->temp_delta_o = (real*) malloc(p_param->prev_size_w * p_param->prev_size_h 
		* p_param->prev_depth * batch_size * sizeof(real));
	
	//No activation for this layer for now
	current->activ_param = NULL;
	
	current->param = p_param;
	
	//No weights initialization in a pool layer
	
	//associate the conv specific functions to the layer
	switch(compute_method)
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











