


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




#include "../prototypes.h"


static dense_param *d_param;

//public are in prototypes.h

//private
void blas_forward_dense_layer(layer *current);
void blas_backward_dense_layer(layer* current);


void blas_dense_define(layer *current)
{
	current->forward = blas_forward_dense_layer;
	current->backprop = blas_backward_dense_layer;
}


void blas_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, depth;
	void* ref_input;
	
	if(current->c_network->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
		current->input = current->c_network->input;
		
	ref_input = current->input;
	
	if(current->previous != NULL && current->previous->type != DENSE)
	{
		//Use a converted (flatten) input if needed
		switch(current->previous->type)
		{
			case CONV:
				nb_area_w = ((conv_param*)current->previous->param)->nb_area_w;
				nb_area_h = ((conv_param*)current->previous->param)->nb_area_h;
				depth = ((conv_param*)current->previous->param)->nb_filters;
				break;
			
			case POOL:
			default:
				nb_area_w = ((pool_param*)current->previous->param)->nb_area_w;
				nb_area_h = ((pool_param*)current->previous->param)->nb_area_h;
				depth = ((pool_param*)current->previous->param)->nb_maps;
				break;
		}
		
		flat_dense(current->input, d_param->flat_input, d_param->bias_value, 
			nb_area_w * nb_area_h , nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size, 
			(nb_area_w * nb_area_h * depth + 1) * current->c_network->batch_size);
		
		ref_input = d_param->flat_input;
	}
	
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_param->nb_neurons+1, 
		current->c_network->batch_size, d_param->in_size, 1.0f, d_param->weights, 
		d_param->nb_neurons+1, ref_input, d_param->in_size, 0.0f, 
		current->output, d_param->nb_neurons+1);
	
	current->activation(current);
	
	if(d_param->dropout_rate > 0.01)
	{
		dropout_select(d_param->dropout_mask, d_param->nb_neurons+1, d_param->dropout_rate);
		dropout_apply(current->output, current->c_network->batch_size, 
			d_param->nb_neurons, d_param->dropout_mask);
	}
}




void blas_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, depth;
	void* ref_input;

	d_param = (dense_param*) current->param;
	
	if(d_param->dropout_rate > 0.01)
		dropout_apply(current->delta_o, current->c_network->batch_size, d_param->nb_neurons,
					d_param->dropout_mask);
	
	//######################## ERROR PROPAGATION ########################
	ref_input = current->input;

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, d_param->in_size, current->c_network->batch_size,
			d_param->nb_neurons+1, 1.0f, d_param->weights, d_param->nb_neurons+1, current->delta_o,
			d_param->nb_neurons+1, 0.0f, d_param->flat_delta_o, d_param->in_size);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		
		if(current->previous->type == POOL || current->previous->type == CONV)
		{
			switch(current->previous->type)
			{
				case POOL:
					nb_area_w = ((pool_param*)current->previous->param)->nb_area_w;
					nb_area_h = ((pool_param*)current->previous->param)->nb_area_h;
					depth = ((pool_param*)current->previous->param)->nb_maps;
					break;
			
				case CONV:
				default:
					nb_area_w = ((conv_param*)current->previous->param)->nb_area_w;
					nb_area_h = ((conv_param*)current->previous->param)->nb_area_h;
					depth = ((conv_param*)current->previous->param)->nb_filters;
					break;	
			}
			
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			reroll_batch(d_param->flat_delta_o, current->previous->delta_o,
				nb_area_w * nb_area_h, nb_area_w * nb_area_h * depth + 1, depth, 
				current->c_network->batch_size, nb_area_w * nb_area_h * depth 
				* current->c_network->batch_size);
		}
		
		current->previous->deriv_activation(current->previous);

	}
	
		
	//########################  WEIGHTS UPDATE   ########################
	
	if(current->previous != NULL && current->previous->type != DENSE)
		ref_input = d_param->flat_input;
		
	//based on the recovered delta_o provided by the next layer propagation

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_param->nb_neurons+1, d_param->in_size,
		current->c_network->batch_size, current->c_network->learning_rate,	current->delta_o, 
		d_param->nb_neurons+1, ref_input, d_param->in_size, current->c_network->momentum,
		d_param->update, d_param->nb_neurons+1);

	update_weights(d_param->weights, d_param->update, d_param->in_size*(d_param->nb_neurons+1));
}
















