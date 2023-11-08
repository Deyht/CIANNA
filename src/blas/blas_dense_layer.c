

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


#include "../prototypes.h"

static dense_param *d_param;

//public are in prototypes.h


void blas_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, nb_area_d, depth;
	void* ref_input;
	
	network* net = current->c_network;
	
	if(net->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
		current->input = net->input;
	
	ref_input = current->input;
	
	if(current->previous != NULL && current->previous->type != DENSE)
	{
		//Use a converted (flatten) input if needed
		switch(current->previous->type)
		{
			case CONV:
				nb_area_w = ((conv_param*)current->previous->param)->nb_area[0];
				nb_area_h = ((conv_param*)current->previous->param)->nb_area[1];
				nb_area_d = ((conv_param*)current->previous->param)->nb_area[2];
				depth = ((conv_param*)current->previous->param)->nb_filters;
				break;
				
			case NORM:
			case LRN:
				switch(current->previous->previous->type)
				{
					default:
					case CONV:
						nb_area_w = ((conv_param*)current->previous->previous->param)->nb_area[0];
						nb_area_h = ((conv_param*)current->previous->previous->param)->nb_area[1];
						nb_area_d = ((conv_param*)current->previous->previous->param)->nb_area[2];
						depth = ((conv_param*)current->previous->previous->param)->nb_filters;
						break;
					case POOL:
						nb_area_w = ((pool_param*)current->previous->previous->param)->nb_area[0];
						nb_area_h = ((pool_param*)current->previous->previous->param)->nb_area[1];
						nb_area_d = ((pool_param*)current->previous->previous->param)->nb_area[2];
						depth = ((pool_param*)current->previous->previous->param)->nb_maps;
						break;
				}
				break;
			
			case POOL:
			default:
				nb_area_w = ((pool_param*)current->previous->param)->nb_area[0];
				nb_area_h = ((pool_param*)current->previous->param)->nb_area[1];
				nb_area_d = ((pool_param*)current->previous->param)->nb_area[2];
				depth = ((pool_param*)current->previous->param)->nb_maps;
				break;
		}
		
		flat_dense(current->input, d_param->flat_input, current->bias_value, 
			nb_area_w * nb_area_h * nb_area_d, nb_area_w * nb_area_h * nb_area_d * depth + 1, 
			depth, net->batch_size, (nb_area_w * nb_area_h * nb_area_d * depth + 1) * net->batch_size);
		
		ref_input = d_param->flat_input;
	}
	
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_param->nb_neurons+1, 
		net->batch_size, d_param->in_size, 1.0f, d_param->weights, 
		d_param->nb_neurons+1, ref_input, d_param->in_size, 0.0f, 
		current->output, d_param->nb_neurons+1);
	
	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			dropout_select_dense(d_param->dropout_mask, (d_param->nb_neurons+1), (d_param->nb_neurons+1)*net->batch_size, current->dropout_rate);
			dropout_apply_dense(current->output, d_param->dropout_mask, (d_param->nb_neurons+1)*net->batch_size);
		}
		else
			dropout_scale_dense(current->output, (d_param->nb_neurons+1), (d_param->nb_neurons+1)*net->batch_size, current->dropout_rate);
	}
	
	current->activation(current);
}


void blas_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, nb_area_d, depth;
	void* ref_input;
	
	network* net = current->c_network;
	
	d_param = (dense_param*) current->param;
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
		dropout_apply_dense(current->delta_o, d_param->dropout_mask, (d_param->nb_neurons+1)*net->batch_size);
	
	//######################## ERROR PROPAGATION ########################
	ref_input = current->input;

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, d_param->in_size, net->batch_size,
			d_param->nb_neurons+1, 1.0f, d_param->weights, d_param->nb_neurons+1, current->delta_o,
			d_param->nb_neurons+1, 0.0f, d_param->flat_delta_o, d_param->in_size);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		
		if(current->previous->type == POOL || current->previous->type == CONV 
			|| current->previous->type == NORM || current->previous->type == LRN)
		{
			switch(current->previous->type)
			{
				case POOL:
					nb_area_w = ((pool_param*)current->previous->param)->nb_area[0];
					nb_area_h = ((pool_param*)current->previous->param)->nb_area[1];
					nb_area_d = ((pool_param*)current->previous->param)->nb_area[2];
					depth = ((pool_param*)current->previous->param)->nb_maps;
					break;
					
				case NORM:
				case LRN:
					switch(current->previous->previous->type)
					{
						default:
						case CONV:
							nb_area_w = ((conv_param*)current->previous->previous->param)->nb_area[0];
							nb_area_h = ((conv_param*)current->previous->previous->param)->nb_area[1];
							nb_area_d = ((conv_param*)current->previous->previous->param)->nb_area[2];
							depth = ((conv_param*)current->previous->previous->param)->nb_filters;
							break;
						case POOL:
							nb_area_w = ((pool_param*)current->previous->previous->param)->nb_area[0];
							nb_area_h = ((pool_param*)current->previous->previous->param)->nb_area[1];
							nb_area_d = ((pool_param*)current->previous->previous->param)->nb_area[2];
							depth = ((pool_param*)current->previous->previous->param)->nb_maps;
							break;
					}
					break;
			
				case CONV:
				default:
					nb_area_w = ((conv_param*)current->previous->param)->nb_area[0];
					nb_area_h = ((conv_param*)current->previous->param)->nb_area[1];
					nb_area_d = ((conv_param*)current->previous->param)->nb_area[2];
					depth = ((conv_param*)current->previous->param)->nb_filters;
					break;
			}
			
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			reroll_batch(d_param->flat_delta_o, current->previous->delta_o, nb_area_w * nb_area_h * nb_area_d, 
				nb_area_w * nb_area_h * nb_area_d * depth + 1, depth, net->batch_size,
				nb_area_w * nb_area_h * nb_area_d * depth * net->batch_size);
		}
		current->previous->deriv_activation(current->previous);
	}
		
	//########################  WEIGHTS UPDATE   ########################
	if(current->previous != NULL && current->previous->type != DENSE)
		ref_input = d_param->flat_input;

	if(!current->frozen)
	{
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_param->nb_neurons+1, d_param->in_size,
			current->c_network->batch_size, current->c_network->learning_rate/current->c_network->batch_size, 
			current->delta_o, d_param->nb_neurons+1, ref_input, d_param->in_size, 
			current->c_network->momentum, d_param->update, d_param->nb_neurons+1);
		
		update_weights(d_param->weights, d_param->update, net->learning_rate*net->weight_decay, 
			(d_param->nb_neurons+1), d_param->in_size*(d_param->nb_neurons+1));
	}
}


void blas_dense_define(layer *current)
{
	current->forward = blas_forward_dense_layer;
	current->backprop = blas_backward_dense_layer;
}












