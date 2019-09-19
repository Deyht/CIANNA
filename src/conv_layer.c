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
				c_param->nb_area_h * c_param->nb_filters * batch_size;
			((ReLU_param*)current->activ_param)->dim = ((ReLU_param*)current->activ_param)->size;
			((ReLU_param*)current->activ_param)->leaking_factor = 0.01;
			break;
			
		case LOGISTIC:
			current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
			((logistic_param*)current->activ_param)->size = c_param->nb_area_w 
				* c_param->nb_area_h *  c_param->nb_filters * batch_size;
			((logistic_param*)current->activ_param)->dim = ((logistic_param*)current->activ_param)->size;
			((logistic_param*)current->activ_param)->beta = 1.0;
			((logistic_param*)current->activ_param)->saturation = 14.0;
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
			
		case LINEAR:
		default:
			current->activ_param = (linear_param*) malloc(sizeof(linear_param));
			((linear_param*)current->activ_param)->size = c_param->nb_area_w * 
				c_param->nb_area_h * c_param->nb_filters * batch_size;
			((linear_param*)current->activ_param)->dim = ((linear_param*)current->activ_param)->size;
			break;
	
	}
}


//Used to allocate a convolutionnal layer
void conv_create(layer *previous, int f_size, int nb_filters, int stride, int padding, int activation)
{
	int i;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net_layers[nb_layers] = current;
	nb_layers++;
	
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
	
	c_param->bias_value = 0.1;
	
	//compute the number of areas to be convolved in the input image
	if(previous == NULL)
	{
		//Case of the first layer
		c_param->prev_size_w = input_width;
		c_param->prev_size_h = input_height;
		c_param->prev_depth = input_depth;
		c_param->flat_f_size = (f_size * f_size * input_depth + 1);
		//input pointer must be set at the begining of forward
		current->input = input;
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
		current->input = (real*) calloc( c_param->prev_depth * (c_param->prev_size_w * c_param->prev_size_h) *
		batch_size, sizeof(real));
		
	}
	
	c_param->nb_area_w = nb_area_comp(c_param->prev_size_w);
	c_param->nb_area_h = nb_area_comp(c_param->prev_size_h);
	
	//allocate all the filters in a flatten table. One filter is continuous. (include bias weight)
	c_param->filters = (real*) malloc(nb_filters * c_param->flat_f_size * sizeof(real));
	//allocate the update for the filters
	c_param->update = (real*) calloc(nb_filters * c_param->flat_f_size, sizeof(real));
	
	c_param->rotated_filters = (real*) malloc(nb_filters * (c_param->flat_f_size-1) * sizeof(real));
	
	
	//allocate the resulting flatten activation map regarding the batch size
	//Activation maps are not continuous for each image : 
	//		A1_im1, A1_im2, A1_im3, ... , A2_im1, A2_im2, A2_im3, ... 
	current->output = (real*) calloc( c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) *
		batch_size, sizeof(real));
	//allocate output error comming from next layer
	current->delta_o = (real*) calloc( c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) * 
		batch_size, sizeof(real));
	
	//temporary output error used for format conversion
	c_param->temp_delta_o = (real*) calloc( c_param->prev_depth * (c_param->prev_size_w 
		* c_param->prev_size_h) * batch_size, sizeof(real));
		
	//printf("\n%d %d %d\n", c_param->flat_f_size, c_param->nb_area_w, c_param->nb_area_h);
	//allocate the im2col input flatten table regarding the batch size
	c_param->im2col_input = (real*) calloc( (c_param->flat_f_size * c_param->nb_area_w * c_param->nb_area_h)
												* batch_size, sizeof(real));
	c_param->im2col_delta_o = (real*) calloc( (c_param->prev_size_w*c_param->prev_size_h) * 
		/* flat_filter*/(f_size*f_size*c_param->nb_filters) * batch_size,  sizeof(real));

	current->param = c_param;

	conv_define_activation_param(current);
	
	//set bias value for the current layer, this value will not move during training
	for(i = 1; i <= c_param->nb_area_w * c_param->nb_area_h* batch_size; i++)
		c_param->im2col_input[i*(c_param->flat_f_size) - 1] = c_param->bias_value;
	
	xavier_normal(c_param->filters, c_param->flat_f_size, c_param->nb_filters, 0);
	
	//associate the conv specific functions to the layer
	switch(compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_conv_define(current);
			cuda_define_activation(current);
			cuda_convert_conv_layer(current);
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



























