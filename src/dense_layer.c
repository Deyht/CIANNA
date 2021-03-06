
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
static dense_param *d_param;

//public are in "prototypes.h"


//private

void dense_define_activation_param(layer *current);


void dense_define_activation_param(layer *current)
{
	d_param = (dense_param*) current->param;
	switch(current->activation_type)
	{
		case RELU:
			current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
			((ReLU_param*)current->activ_param)->size = (d_param->nb_neurons + 1) 
				* current->c_network->batch_size;
			((ReLU_param*)current->activ_param)->dim = d_param->nb_neurons;
			((ReLU_param*)current->activ_param)->biased_dim = d_param->nb_neurons+1;
			((ReLU_param*)current->activ_param)->leaking_factor = 0.01;
			d_param->bias_value = 0.1;
			break;
			
		case LOGISTIC:
			current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
			((logistic_param*)current->activ_param)->size = (d_param->nb_neurons+1) 
				* current->c_network->batch_size;
			((logistic_param*)current->activ_param)->dim = d_param->nb_neurons;
			((logistic_param*)current->activ_param)->biased_dim = d_param->nb_neurons+1;
			((logistic_param*)current->activ_param)->beta = 1.0;
			((logistic_param*)current->activ_param)->saturation = 10.0;
			d_param->bias_value = -1.0;
			break;
			
		case SOFTMAX:
			current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
			((softmax_param*)current->activ_param)->dim = d_param->nb_neurons;
			((softmax_param*)current->activ_param)->biased_dim = d_param->nb_neurons+1;
			d_param->bias_value = -1.0;
			break;
			
		case LINEAR:
		default:
			current->activ_param = (linear_param*) malloc(sizeof(linear_param));
			((linear_param*)current->activ_param)->size = (d_param->nb_neurons + 1) 
				* current->c_network->batch_size;
			((linear_param*)current->activ_param)->dim = d_param->nb_neurons;
			((linear_param*)current->activ_param)->biased_dim = d_param->nb_neurons+1;
			//Change to expect output between 0 and 1
			d_param->bias_value = 0.5;
			break;
	}
}


void dense_create(network *net, layer* previous, int nb_neurons, int activation, float drop_rate, FILE *f_load)
{
	int i, j;
	float bias_padding_value;
	layer* current;
	
	if(net->compute_method == C_CUDA && net->use_cuda_TC && nb_neurons % 8 == 0)
		nb_neurons -= 1;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	d_param = (dense_param*) malloc(sizeof(dense_param));
	
	current->type = DENSE;
	current->activation_type = activation;
	d_param->nb_neurons = nb_neurons;
	d_param->dropout_rate = drop_rate;
	
	current->previous = previous;
	
	
	if(previous == NULL)
	{
		d_param->in_size = net->input_width*net->input_height*net->input_depth+1;
		current->input = net->input;
	}
	else
	{
		switch(previous->type)
		{
			case CONV:
				d_param->in_size = ((conv_param*)previous->param)->nb_area_w
					* ((conv_param*)previous->param)->nb_area_h 
					* ((conv_param*)previous->param)->nb_filters + 1;
				d_param->flat_delta_o = (float*) calloc(d_param-> in_size * net->batch_size, sizeof(float));
				break;
			
			case POOL:
				d_param->in_size = ((pool_param*)previous->param)->nb_area_w 
					* ((pool_param*)previous->param)->nb_area_h 
					* ((pool_param*)previous->param)->nb_maps + 1;
				d_param->flat_delta_o = (float*) calloc(d_param->in_size * net->batch_size, sizeof(float));
				((pool_param*)previous->param)->next_layer_type = current->type;
				break;
			
			case DENSE:
			default:
				d_param->in_size = ((dense_param*)previous->param)->nb_neurons+1;
				d_param->flat_delta_o = previous->delta_o;
				break;
		
		}
		current->input = previous->output;
		d_param->flat_input = (float*) calloc(d_param->in_size*net->batch_size,sizeof(float));
	}

	d_param->weights = (float*) malloc(d_param->in_size*(nb_neurons+1)*sizeof(float));
	
	d_param->update = (float*) calloc(d_param->in_size*(nb_neurons+1), sizeof(float));
	d_param->dropout_mask = (int*) calloc(d_param->nb_neurons, sizeof(int));
	
	current->output = (float*) calloc((nb_neurons+1)*net->batch_size, sizeof(float));
	current->delta_o = (float*) calloc((nb_neurons+1)*net->batch_size, sizeof(float));
	
	
	//must be before the association functions
	current->param = d_param;
	
	dense_define_activation_param(current);
	
	if(current->previous == NULL)
		((dense_param*)current->param)->bias_value = net->input_bias;
	
	if(f_load == NULL)
	{
		if(current->previous == NULL || current->previous->type != DENSE)
		{
			//need a modification to allow bias change in first layer
			bias_padding_value = 1.0;
		}
		else
		{
			bias_padding_value = (float) d_param->bias_value/((dense_param*)current->previous->param)->bias_value;
		}
		xavier_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, bias_padding_value);
	}
	else
	{
		for(i = 0; i < d_param->in_size; i++)
			for(j = 0; j < (d_param->nb_neurons+1); j++)
				fscanf(f_load, "%f", &(((float*)d_param->weights)[i*(d_param->nb_neurons+1) + j]));
	}
	
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_dense_define(current);
			cuda_define_activation(current);
			cuda_convert_dense_layer(current);
			#endif
			break;
			
		case C_BLAS:
			#ifdef BLAS
			blas_dense_define(current);
			define_activation(current);
			#endif
			break;
			
		case C_NAIV:
			naiv_dense_define(current);
			define_activation(current);
			break;
			
		default:
			break;
	}
	
		char activ[10];
	get_string_activ_param(activ, current->activation_type);
	printf("L:%d - Dense layer created:\n \
Input: %d, Nb. Neurons: %d \n \
Activation: %s, Dropout: %f\n",
		net->nb_layers, d_param->in_size,  d_param->nb_neurons+1, 
		activ, d_param->dropout_rate);
		
	if(net->compute_method == C_CUDA && net->use_cuda_TC)
	{
		if(d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0)
			printf("Warning : Forward gemm fallback to non TC version due to layer size mismatch\n");
		if(current->previous != NULL && (d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0))
			{
			printf("%d %d %d\n", d_param->in_size, current->c_network->batch_size, (d_param->nb_neurons+1));
			printf("Warning : Backprop gemm fallback to non TC version due to layer size mismatch\n");
			}
		if(d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0)
			printf("Warning : Weights update gemm fallback to non TC version due to layer size mismatch\n");
	}
	
}


void dense_save(FILE *f, layer *current)
{
	int i, j;
	float* host_weights = NULL;

	d_param = (dense_param*)current->param;	
	
	fprintf(f,"D");
	fprintf(f, "%dn%fd", d_param->nb_neurons, d_param->dropout_rate);
	print_activ_param(f, current->activation_type);
	fprintf(f,"\n");
	
	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_weights = (float*) malloc(d_param->in_size * (d_param->nb_neurons+1) * sizeof(float));
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_get_table_FP32(current->c_network, (float*)d_param->weights, (float*)host_weights,
					d_param->in_size * (d_param->nb_neurons+1));
				break;
			case 1:
				cuda_get_table_FP32(current->c_network, (float*)d_param->FP32_weights, (float*)host_weights,
					d_param->in_size * (d_param->nb_neurons+1));
				break;
		}
		#endif
	}
	else
	{
		host_weights = d_param->weights;
	}
	
	for(i = 0; i < d_param->in_size; i++)
	{
		for(j = 0; j < (d_param->nb_neurons+1); j++)
			fprintf(f, "%g ", host_weights[i*(d_param->nb_neurons+1) + j]);
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
	
	if(current->c_network->compute_method == C_CUDA)
		free(host_weights);
}

void dense_load(network *net, FILE* f)
{
	int nb_neurons;
	float dropout_rate;
	char activ_type[20];
	layer *previous;
	
	printf("Loading dense layer, L:%d\n", net->nb_layers);
	
	fscanf(f, "%dn%fd%s\n", &nb_neurons, &dropout_rate, activ_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];

	dense_create(net, previous, nb_neurons, load_activ_param(activ_type), dropout_rate, f);
}








