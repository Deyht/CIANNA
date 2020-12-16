
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

static pool_param *p_param;

//public are in prototypes.h

//private
void forward_pool_layer(layer* current);
void backward_pool_layer(layer* current);
void deltah_pool(void* delta_o, void* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length);
void deltah_pool_cont(void* delta_o, void* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int column_length);


void pool_define(layer *current)
{
	current->forward = forward_pool_layer;
	current->backprop = backward_pool_layer;
}

void forward_pool_layer(layer* current)
{
	int i, j, k;
	int x, y, x_max, y_max, pos, pos_out;

	if(current->c_network->length == 0)
		return;

	p_param = (pool_param*) current->param;
	
	#pragma omp parallel for private(i, j, x, y, x_max, y_max, pos, pos_out) collapse(3) schedule(guided,2)
	for(k = 0; k < p_param->nb_maps * current->c_network->batch_size; k++)
		for(i = 0; i < p_param->nb_area_w; i++)
			for(j = 0; j < p_param->nb_area_w; j++)
			{
				pos_out = k*(p_param->nb_area_w*p_param->nb_area_w) + i + j*p_param->nb_area_w;	
				pos = k*p_param->prev_size_w*p_param->prev_size_w + i*p_param->p_size 
					+ j*p_param->p_size*p_param->prev_size_w;
				
				x_max = 0;
				y_max = 0;
				for(x = 0; x < p_param->p_size; x++)
					for(y = 0; y < p_param->p_size; y++)
						if(((float*)current->input)[pos + x_max*p_param->prev_size_w + y_max] 
							< ((float*)current->input)[pos + x*p_param->prev_size_w + y])
						{
							x_max = x;
							y_max = y;
						}
				p_param->pool_map[pos_out] = x_max*p_param->p_size + y_max;
				((float*)current->output)[pos_out] = ((float*)current->input)[pos + x_max*p_param->prev_size_w + y_max];
			}
}


void backward_pool_layer(layer* current)
{	
	int i, pos;
	int image_size, column_length;
	
	p_param = (pool_param*) current->param;
	
	image_size = p_param->nb_maps * p_param->nb_area_w * p_param->nb_area_h;
	column_length = p_param->nb_area_w;

	if(current->previous != NULL)
	{
		if(current->previous->type == CONV)
		{
			//array must be set to 0 as deltah_pool do not erase previous values
			memset(current->previous->delta_o, 0.0, p_param->prev_depth * p_param->prev_size_w 
				* p_param->prev_size_h * current->c_network->batch_size*sizeof(float));
			
			#pragma omp parallel for private(pos) schedule(guided,4)
			for(i = 0; i < current->c_network->length*image_size; i++)
			{
				//add mask of locations
				pos = (i/column_length) * column_length * p_param->p_size * p_param->p_size 
					+ (i%column_length) * p_param->p_size + (((int)p_param->pool_map[i])/p_param->p_size) 
					* column_length * p_param->p_size + (((int)p_param->pool_map[i])%p_param->p_size);
				
				((float*)current->previous->delta_o)[pos] = ((float*)current->delta_o)[i];
			}	
		}
		
		current->previous->deriv_activation(current->previous);
	}
}











