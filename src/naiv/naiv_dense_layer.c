
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
void naiv_forward_dense_layer(layer *current);
void naiv_backward_dense_layer(layer* current);


//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
void flat_dense(void *in, void *out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
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


void flat_dense_back(void *in, void *out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
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
		
		if(map_id < nb_map)
			f_out[i] += f_in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


void reroll_batch(void *in, void *out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
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
		
		f_out[i] += f_in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}


void dropout_select_dense(float *mask, int biased_dim, size_t size, float drop_rate)
{
	int i;
	float rand;
	
	//#pragma omp parallel for private(rand) schedule(guided,4)
	//OMP overhead is too high for "small" dense layers
	//Performance is limited by CPU cache size and speed regardless of core count
	for(i = 0; i < size; i++)
	{
		rand = random_uniform();
		if(rand >= drop_rate || (i+1) % biased_dim == 0)
			mask[i] = 1.0f;
		else
			mask[i] = 0.0f;
	}
}


void dropout_apply_dense(void *table, float *mask, size_t size)
{
	int i;
	
	float *f_table = (float*) table;
	
	for(i = 0; i < size; i++)
		f_table[i] = f_table[i]*mask[i];
}


void dropout_scale_dense(void *table, int biased_dim, size_t size, float drop_rate)
{
	int i;
	
	float *f_table = (float*) table;
	
	for(i = 0; i < size; i++)
		if((i+1) % biased_dim != 0)
			f_table[i] = f_table[i]*(1.0f-drop_rate);
}


void naiv_forward_dense_layer(layer *current)
{
	size_t i, j, b;
	double h;
	int nb_neurons;
	size_t flat_in_size = 1;
	float *ref_input;
	
	network* net = current->c_network;
	d_param = (dense_param*) current->param;
	
	float *f_weights;
	float *f_output = (float*) current->output;
	
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
	
	ref_input = (float*)current->input;
	
	if(net->is_inference == 1 && (net->use_wema && !net->inference_only))
		f_weights = (void*) current->ema_weights;
	else
		f_weights = (void*) current->weights;
	
	if(current->previous != NULL && current->previous->output_type != FLAT)
	{
		flat_dense(current->input, d_param->flat_input, current->bias_value, 
			current->prev_dim[0]*current->prev_dim[1]*current->prev_dim[2], flat_in_size, 
			current->prev_dim[3], net->batch_size, flat_in_size * net->batch_size);
		
		ref_input = (float*)d_param->flat_input;
	}
	
	//Strongly affected by performance drop of cache miss
	//Could be optimized by transposing the matrix first => better use OpenBLAS directly
	#pragma omp parallel for private(i, j, h) shared(f_weights) collapse(2) schedule(guided, 4)
	for(b = 0; b < net->batch_size; b++)
	{
		for(i = 0; i < nb_neurons+1; i++)
		{
			h = 0.0;
			for(j = 0; j < flat_in_size; j++)
			{
				h += f_weights[j*(nb_neurons+1) + i]
					* ref_input[b*flat_in_size + j];
			}
			
			f_output[b*(nb_neurons+1)+i] = h;
		}
	}
	
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


void naiv_backward_dense_layer(layer* current)
{
	size_t i, j, b;
	double h;
	int nb_neurons;
	size_t flat_in_size = 1;
	float *ref_input;
	
	network* net = current->c_network;
	d_param = (dense_param*) current->param;
	
	float *f_weights = (float*) current->weights;
	float *f_delta_o = (float*) current->delta_o;
	float *f_flat_delta_o = (float*) d_param->flat_delta_o;
	float *f_update = (float*) current->gradient;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
		dropout_apply_dense(current->delta_o, current->dropout_mask, (nb_neurons+1)*net->batch_size);
	
	//######################## ERROR PROPAGATION ########################

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < net->batch_size; b++)
		{
			for(i = 0; i <  flat_in_size; i++)
			{
				h = 0.0;
				for(j = 0; j < nb_neurons+1; j++)
				{
					h += (double)f_weights[i*(nb_neurons+1) + j]
							* (double)f_delta_o[b*(nb_neurons+1) + j];
				}
				f_flat_delta_o[b*(flat_in_size)+i] += h;
			}
		}
		
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
		ref_input = (float*) current->input;
	
		if(current->previous != NULL && current->previous->output_type != FLAT)
			ref_input = (float*) d_param->flat_input;
	
		#pragma omp parallel for private(j, b, h) collapse(2) schedule(guided, 4)
		for(i = 0; i <  flat_in_size; i++)
		{
			for(j = 0; j < nb_neurons+1; j++)
			{
				h = 0.0;
				for(b = 0; b < net->batch_size; b++)
				{
					h += (double)f_delta_o[b*(nb_neurons+1) + j]
							* (double)ref_input[b*flat_in_size + i];
				}
				f_update[i*(nb_neurons+1)+j] = h;
			}
		}
		
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


void naiv_dense_define(layer *current)
{
	current->forward = naiv_forward_dense_layer;
	current->backprop = naiv_backward_dense_layer;
}







