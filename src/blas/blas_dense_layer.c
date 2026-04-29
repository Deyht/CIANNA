
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


#include "../prototypes.h"

// Local variables
static dense_param *d_param;

// Public are in "prototypes.h"

// Private prototypes
void blas_forward_dense_layer(layer *current);
void blas_backward_dense_layer(layer* current);


void blas_forward_dense_layer(layer *current)
{
	int i;
	int nb_neurons;
	size_t flat_in_size = 1;
	float *ref_input, *l_weights;
	
	network* net = current->c_network;
	d_param = (dense_param*) current->param;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	if(current->previous == NULL)
	{
		ref_input = net->input;
		for(i = 0; i < net->batch_size; i++)
			ref_input[i*(net->input_dim+1) + net->input_dim] = current->bias_value;
		
		current->input = net->input;
	}
	else
		current->input = current->previous->output;
	
	ref_input = current->input;
	
	if(net->is_inference == 1 && (net->use_wema && !net->inference_only))
		l_weights = (void*) current->ema_weights;
	else
		l_weights = (void*) current->weights;
	
	if(current->previous != NULL && current->previous->output_type != FLAT)
	{
		flat_dense(current->input, d_param->flat_input, current->bias_value, 
			current->prev_dim[0]*current->prev_dim[1]*current->prev_dim[2], flat_in_size, 
			current->prev_dim[3], net->batch_size, flat_in_size * net->batch_size);
		
		ref_input = d_param->flat_input;
	}
	
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb_neurons+1, 
		net->batch_size, flat_in_size, 1.0f, l_weights, 
		nb_neurons+1, ref_input, flat_in_size, 0.0f, 
		current->output, nb_neurons+1);
	
	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			dropout_select_dense(current->dropout_mask, (nb_neurons+1), (nb_neurons+1)*net->batch_size, current->dropout_rate);
			dropout_apply_dense(current->output, current->dropout_mask, (nb_neurons+1)*net->batch_size);
		}
		else
			dropout_scale_dense(current->output, (nb_neurons+1), (nb_neurons+1)*net->batch_size, current->dropout_rate);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
	{
		memset(current->delta_o, 0, (nb_neurons+1) * net->batch_size * sizeof(float));
		if(current->previous != NULL && current->previous->output_type != FLAT)
			memset(d_param->flat_delta_o, 0, flat_in_size * net->batch_size * sizeof(float));
	}
}


void blas_backward_dense_layer(layer* current)
{
	size_t i;
	int nb_neurons;
	size_t flat_in_size = 1;
	float* ref_input;
	
	network* net = current->c_network;
	d_param = (dense_param*) current->param;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
		dropout_apply_dense(current->delta_o, current->dropout_mask, (nb_neurons+1)*net->batch_size);
	
	//######################## ERROR PROPAGATION ########################
	ref_input = current->input;
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, flat_in_size, net->batch_size,
			nb_neurons+1, 1.0f, current->weights, nb_neurons+1, current->delta_o,
			nb_neurons+1, 1.0f, d_param->flat_delta_o, flat_in_size);
		
		//if previous layer is dense then flat_delta_o = previous->delta_o
		if(current->previous->output_type == SPATIAL)
		{
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			reroll_batch(d_param->flat_delta_o, current->previous->delta_o,
				current->prev_dim[0]*current->prev_dim[1]*current->prev_dim[2], flat_in_size, 
				current->output_dim[3], net->batch_size, (flat_in_size-1) * net->batch_size);
		}
	}
		
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		if(current->previous != NULL && current->previous->output_type != FLAT)
			ref_input = d_param->flat_input;
	
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, nb_neurons+1, flat_in_size,
			current->c_network->batch_size, 1.0f, 
			current->delta_o, nb_neurons+1, ref_input, flat_in_size, 
			0.0f, current->gradient, nb_neurons+1);
		
		net->optim_update_fct(current, (flat_in_size-1)*(nb_neurons+1), 
			flat_in_size*(nb_neurons+1), flat_in_size*(nb_neurons+1));
			
		if(current->wema_replace_signal > 0)
		{
			for(i = 0; i < flat_in_size*(nb_neurons+1); i++)
				current->FP32_weights[i] = current->ema_weights[i];
			current->wema_replace_signal = 0;
		}
	}
}


void blas_dense_define(layer *current)
{
	current->forward = blas_forward_dense_layer;
	current->backprop = blas_backward_dense_layer;
}












