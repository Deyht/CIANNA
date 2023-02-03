

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
void naiv_forward_dense_layer(layer *current);
void naiv_backward_dense_layer(layer* current);


//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
void flat_dense(void* in, void* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i;
	int map_id, image_id, pos;
	
	float *f_in = (float*) in;
	float *f_out = (float*) out;
	
	#pragma omp parallel for private(image_id, map_id, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;
		
		if(map_id >= nb_map)
			f_out[i] = bias;
		else
			f_out[i] = f_in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


void reroll_batch(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i;
	int map_id, image_id, pos;
	
	float *f_in = (float*) in;
	float *f_out = (float*) out;
	
	#pragma omp parallel for private(image_id, map_id, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		f_out[i] = f_in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}


void dropout_select_dense(int* mask, int size, float drop_rate)
{
	int i;
	float rand;
	
	//#pragma omp parallel for private(rand) schedule(guided,4)
	//OMP overhead is too high for "small" dense layers
	//Performance is limited by CPU cache size and speed regardless of core count
	for(i = 0; i < size; i++)
	{
		rand = random_uniform();
		if(rand < drop_rate)
			mask[i] = 0;
		else
			mask[i] = 1;
	}
}


void dropout_apply_dense(void* table, int batch_size, int dim, int* mask)
{
	int i;
	int j;
	
	float* f_table = (float*) table;
	
	for(i = 0; i < batch_size; i++)
	{
		for(j = 0; j < dim; j++)
		{
			f_table[i*(dim+1) + j] *= mask[j];
		}
	}
}



void naiv_dense_define(layer *current)
{
	current->forward = naiv_forward_dense_layer;
	current->backprop = naiv_backward_dense_layer;
}


void naiv_forward_dense_layer(layer *current)
{
	int i, j, b;
	double h, w_alpha;
	int nb_area_w, nb_area_h, nb_area_d, depth;
	void* ref_input;
	
	float prev_drop_rate = 0.0f;
	
	network* net = current->c_network;
	
	if(net->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
		current->input = net->input;
	
	ref_input = current->input;

	float *f_weights = (float*) d_param->weights;
	float *f_output = (float*) current->output;
	
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
	
	//bias weight is included in drop, should change this behavior ?
	if(net->is_inference && net->inference_drop_mode == AVG_MODEL && current->previous != NULL)
	{
		prev_drop_rate = current->previous->dropout_rate;
		w_alpha = ((d_param->in_size-1)*(1.0f-prev_drop_rate)+1)/d_param->in_size;
	}
	else
	{
		w_alpha = 1.0f;
	}
	
	float *f_input = (float*) ref_input;
	//Strongly affected by performance drop of cache miss
	//Could be optimized by transposing the matrix first => better use OpenBLAS directly
	#pragma omp parallel for private(i, j, h) shared(f_weights) collapse(2) schedule(guided, 2)
	for(b = 0; b < net->batch_size; b++)
	{
		for(i = 0; i < d_param->nb_neurons+1; i++)
		{
			h = 0.0;
			for(j = 0; j < d_param->in_size; j++)
			{
				h += f_weights[j*(d_param->nb_neurons+1) + i]
						* f_input[b*d_param->in_size + j];
			}
			
			f_output[b*(d_param->nb_neurons+1)+i] = w_alpha * h;
		}
	}
	
	current->activation(current);
	
	if(current->dropout_rate > 0.01f && (!net->is_inference || net->inference_drop_mode == MC_MODEL))
	{
		dropout_select_dense(d_param->dropout_mask, d_param->nb_neurons, current->dropout_rate);
		dropout_apply_dense(current->output, net->batch_size, d_param->nb_neurons, d_param->dropout_mask);
	}
}


void naiv_backward_dense_layer(layer* current)
{
	int i, j, b;
	double h;
	int nb_area_w, nb_area_h, nb_area_d, depth;
	void* ref_input;
	
	network* net = current->c_network;
	
	d_param = (dense_param*) current->param;
	
	float *f_weights = (float*) d_param->weights;
	float *f_delta_o = (float*) current->delta_o;
	float *f_flat_delta_o = (float*) d_param->flat_delta_o;
	float *f_update = (float*) d_param->update;
	
	if(current->dropout_rate > 0.01f)
		dropout_apply_dense(current->delta_o, net->batch_size, d_param->nb_neurons,
					d_param->dropout_mask);
	
	//######################## ERROR PROPAGATION ########################
	ref_input = current->input;

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < net->batch_size; b++)
		{
			for(i = 0; i <  d_param->in_size; i++)
			{
				h = 0.0;
				for(j = 0; j < d_param->nb_neurons+1; j++)
				{
					h += f_weights[i*(d_param->nb_neurons+1) + j]
							* f_delta_o[b*(d_param->nb_neurons+1) + j];
				}
				f_flat_delta_o[b*(d_param->in_size)+i] = h;
			}
		}
		
		//if previous layer is dense then flat_delta_o = previous->delta_o
		if(current->previous->type == POOL || current->previous->type == CONV)
		{
			switch(current->previous->type)
			{
				case POOL:
					nb_area_w = ((pool_param*)current->previous->param)->nb_area[0];
					nb_area_h = ((pool_param*)current->previous->param)->nb_area[1];
					nb_area_d = ((pool_param*)current->previous->param)->nb_area[2];
					depth = ((pool_param*)current->previous->param)->nb_maps;
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
			reroll_batch(d_param->flat_delta_o, current->previous->delta_o,
				nb_area_w * nb_area_h * nb_area_d, nb_area_w * nb_area_h * nb_area_d * depth + 1, depth, 
				net->batch_size, nb_area_w * nb_area_h * nb_area_d * depth * net->batch_size);
		}
		current->previous->deriv_activation(current->previous);
	}
		
	//########################  WEIGHTS UPDATE   ########################
	if(current->previous != NULL && current->previous->type != DENSE)
		ref_input = d_param->flat_input;
	
	float *f_input = (float*) ref_input;
	//based on the recovered delta_o provided by the next layer propagation

	if(!current->frozen)
	{
		#pragma omp parallel for private(j, b, h) collapse(2) schedule(guided, 4)
		for(i = 0; i <  d_param->in_size; i++)
		{
			for(j = 0; j < d_param->nb_neurons+1; j++)
			{
				h = 0.0;
				for(b = 0; b < net->batch_size; b++)
				{
					h += f_delta_o[b*(d_param->nb_neurons+1) + j]
							* f_input[b*d_param->in_size + i];
				}
				f_update[i*(d_param->nb_neurons+1)+j] = net->learning_rate*h 
						+ net->momentum * f_update[i*(d_param->nb_neurons+1)+j];
			}
		}
		
		update_weights(d_param->weights, d_param->update, net->learning_rate*net->weight_decay, d_param->in_size*(d_param->nb_neurons+1));
	}
}










