#include "prototypes.h"


//##############################
//       Local variables
//##############################
static dense_param *d_param;

//public are in "prototypes.h"


//private

void dense_define_activation_param(layer *current);


void dense_define_activation_param(layer *current)
{
	d_param = (dense_param*) current->param;
	printf("IN\n");
	switch(current->activation_type)
	{
		case RELU:
			current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
			((ReLU_param*)current->activ_param)->size = (d_param->nb_neurons + 1)*batch_size;
			((ReLU_param*)current->activ_param)->dim = d_param->nb_neurons;
			((ReLU_param*)current->activ_param)->leaking_factor = 0.01;
			d_param->bias_value = 0.1;
			break;
			
		case LOGISTIC:
			printf("Logistic activ_param def\n");
			current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
			((logistic_param*)current->activ_param)->size = (d_param->nb_neurons+1)*batch_size;
			((logistic_param*)current->activ_param)->dim = d_param->nb_neurons;
			((logistic_param*)current->activ_param)->beta = 1.0;
			((logistic_param*)current->activ_param)->saturation = 16.0;
			d_param->bias_value = -1.0;
			break;
			
		case SOFTMAX:
			current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
			((softmax_param*)current->activ_param)->dim = d_param->nb_neurons;
			d_param->bias_value = -1.0;
			break;
			
		case LINEAR:
		default:
			current->activ_param = (linear_param*) malloc(sizeof(linear_param));
			((linear_param*)current->activ_param)->size = (d_param->nb_neurons + 1)*batch_size;
			((linear_param*)current->activ_param)->dim = d_param->nb_neurons;
			d_param->bias_value = -1.0;
			break;
	
	}
	printf("OUT\n");
}


void dense_create(layer *current, layer* previous, int nb_neurons, int activation)
{
	
	d_param = (dense_param*) malloc(sizeof(dense_param));
	
	current->type = DENSE;
	current->activation_type = activation;
	d_param->nb_neurons = nb_neurons;
	
	current->previous = previous;
	
	//WARNING : MUST ADAPT VALUE TO ACTIVATION FUNCTION !! IN REGARDE OF WEIGHTS
	d_param->bias_value = 0.1;
	
	
	if(previous == NULL)
	{
		d_param->in_size = input_width*input_height*input_depth+1;
		current->input = input;
	}
	else
	{
		switch(previous->type)
		{
			case CONV:
				d_param->in_size = ((conv_param*)previous->param)->nb_area_w*
					((conv_param*)previous->param)->nb_area_h *
					((conv_param*)previous->param)->nb_filters + 1;
				d_param->flat_delta_o = (real*) calloc(d_param-> in_size * batch_size, sizeof(real));
				break;
			
			case POOL:
				d_param->in_size = ((pool_param*)previous->param)->nb_area_w 
					* ((pool_param*)previous->param)->nb_area_h * ((pool_param*)previous->param)->nb_maps + 1;
				d_param->flat_delta_o = (real*) calloc(d_param->in_size * batch_size, sizeof(real));
				((pool_param*)previous->param)->next_layer_type = current->type;
				break;
			
			case DENSE:
			default:
				d_param->in_size = ((dense_param*)previous->param)->nb_neurons+1;
				d_param->flat_delta_o = previous->delta_o;
				break;
		
		}
		current->input = previous->output;
		d_param->flat_input = (real*) malloc(d_param->in_size*batch_size*sizeof(real));
	}

	d_param->weights = (real*) malloc(d_param->in_size*(nb_neurons+1)*sizeof(real));
	d_param->update = (real*) calloc(d_param->in_size*(nb_neurons+1), sizeof(real));
	
	current->output = (real*) calloc((nb_neurons+1)*batch_size, sizeof(real));
	current->delta_o = (real*) calloc((nb_neurons+1)*batch_size, sizeof(real));
	
	//must be before the association functions
	current->param = d_param;
	
	dense_define_activation_param(current);
	xavier_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1);
	//d_param->weights[0] = -2.0;
	
	switch(compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_dense_define(current);
			cuda_define_activation(current);
			cuda_convert_dense_layer(current);
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






