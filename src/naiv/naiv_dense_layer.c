


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



void naiv_dense_define(layer *current)
{
	current->forward = naiv_forward_dense_layer;
	current->backprop = naiv_backward_dense_layer;
}


void naiv_forward_dense_layer(layer *current)
{
	int i, j, b;
	double h;
	int nb_area_w, nb_area_h, depth;
	
	if(current->c_network->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
		current->input = current->c_network->input;

	
	if(current->previous->type == DENSE)
	{
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 2)
		for(b = 0; b < current->c_network->batch_size; b++)
		{
			for(i = 0; i < d_param->nb_neurons+1; i++)
			{
				h = 0.0;
				for(j = 0; j < d_param->in_size; j++)
				{
					h += d_param->weights[j*(d_param->nb_neurons+1) + i]
							* current->input[b*d_param->in_size + j];
				}
				current->output[b*(d_param->nb_neurons+1)+i] = h;
			}
		}
	}
	else if(current->previous->type != DENSE)
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
		
		//Strongly affected by performance drop of cache miss
		//Could be optimized by transposing the matrix first => better use OpenBLAS directly
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < current->c_network->batch_size; b++)
		{
			for(i = 0; i < d_param->nb_neurons+1; i++)
			{
				h = 0.0;
				for(j = 0; j < d_param->in_size; j++)
				{
					h += d_param->weights[j*(d_param->nb_neurons+1) + i]
							* d_param->flat_input[b*d_param->in_size + j];
				}
				current->output[b*(d_param->nb_neurons+1)+i] = h;
			}
		}
	}
	
	dropout_select(d_param->dropout_mask, d_param->nb_neurons+1, d_param->dropout_rate);
	
	current->activation(current);
	
	if(d_param->dropout_rate > 0.01)
		dropout_apply(current->output, current->c_network->batch_size, 
			d_param->nb_neurons, d_param->dropout_mask);
	
}




void naiv_backward_dense_layer(layer* current)
{
	int i, j, b;
	double h;
	int nb_area_w, nb_area_h, depth;
	
	d_param = (dense_param*) current->param;
	
	if(d_param->dropout_rate > 0.01)
		dropout_apply(current->delta_o, current->c_network->batch_size, d_param->nb_neurons,
					d_param->dropout_mask);
	
	//######################## ERROR PROPAGATION ########################

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < current->c_network->batch_size; b++)
		{
			for(i = 0; i <  d_param->in_size; i++)
			{
				h = 0.0;
				for(j = 0; j < d_param->nb_neurons+1; j++)
				{
					h += d_param->weights[i*(d_param->nb_neurons+1) + j]
							* current->delta_o[b*(d_param->nb_neurons+1) + j];
				}
				d_param->flat_delta_o[b*(d_param->in_size)+i] = h;
			}
		}
		
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
	
	//based on the recovered delta_o provided by the next layer propagation
	if(current->previous != NULL && current->previous->type != DENSE)
	{
		#pragma omp parallel for private(j, b, h) collapse(2) schedule(guided, 4)
		for(i = 0; i <  d_param->in_size; i++)
		{
			for(j = 0; j < d_param->nb_neurons+1; j++)
			{
				h = 0.0;
				for(b = 0; b < current->c_network->batch_size; b++)
				{
					h += current->delta_o[b*(d_param->nb_neurons+1) + j]
							* d_param->flat_input[b*d_param->in_size + i];
				}
				d_param->update[i*(d_param->nb_neurons+1)+j] = current->c_network->learning_rate*h 
						+ current->c_network->momentum * d_param->update[i*(d_param->nb_neurons+1)+j];
			}
		}
	}
	else
	{
		#pragma omp parallel for private(j, b, h) collapse(2) schedule(guided, 4)
		for(i = 0; i <  d_param->in_size; i++)
		{
			for(j = 0; j < d_param->nb_neurons+1; j++)
			{
				h = 0.0;
				for(b = 0; b < current->c_network->batch_size; b++)
				{
					h += current->delta_o[b*(d_param->nb_neurons+1) + j]
							* current->input[b*d_param->in_size + i];
				}
				d_param->update[i*(d_param->nb_neurons+1)+j] = current->c_network->learning_rate*h 
						+ current->c_network->momentum * d_param->update[i*(d_param->nb_neurons+1)+j];
			}
		}
	}

	update_weights(d_param->weights, d_param->update, d_param->in_size*(d_param->nb_neurons+1));
}



//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
void flat_dense(real* in, real* out, real bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i;
	int map_id, image_id, pos;
	
	#pragma omp parallel for private(image_id, map_id, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;
		
		if(map_id >= nb_map)
			out[i] = bias;
		else
			out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


void reroll_batch(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i;
	int map_id, image_id, pos;
	
	#pragma omp parallel for private(image_id, map_id, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}


void dropout_select(real* mask, int size, real drop_rate)
{
	int i;
	real rand;
	
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


void dropout_apply(real* table, real batch_size, int dim, real* mask)
{
	int i;
	int j;
	
	for(i = 0; i < batch_size; i++)
	{
		for(j = 0; j < dim; j++)
		{
			table[i*(dim+1) + j] *= mask[j];
		}
	}
}









