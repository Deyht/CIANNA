
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

static int cu_blocks;
static pool_param *p_param;

//public are in prototypes.h

//private
void cuda_forward_pool_layer(layer* current);
void cuda_backward_pool_layer(layer* current);

__global__ void  pooling_kernel(real *input, real *output, real* pool_map, int pool_size, int w_size, int w_size_out, int  length);
__global__ void deltah_pool(real* delta_o, real* delta_o_unpool, real* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length);
__global__ void deltah_pool_cont(real* delta_o, real* delta_o_unpool, real* pool_map, int pool_size, int len, int batch_size, int image_size, int column_length);


void cuda_pool_define(layer *current)
{
	current->forward = cuda_forward_pool_layer;
	current->backprop = cuda_backward_pool_layer;
}

void cuda_convert_pool_layer(layer *current)
{
	p_param = (pool_param*)current->param;

	cuda_convert_table(&(p_param->pool_map), p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps 
		* current->c_network->batch_size);
	cuda_convert_table(&(current->output), p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps 
		* current->c_network->batch_size);
	
	cuda_convert_table(&(current->delta_o), p_param->nb_area_w * p_param->nb_area_h * p_param->nb_maps
		* current->c_network->batch_size);
	
	cuda_convert_table(&(p_param->temp_delta_o), p_param->prev_size_w * p_param->prev_size_h 
		* p_param->prev_depth * current->c_network->batch_size);
}


void cuda_forward_pool_layer(layer* current)
{
	if(current->c_network->length == 0)
		return;
		
	p_param = (pool_param*) current->param;
	
	//late declaration of CUDA kernel sizes
	dim3 threadsPerBlock(8, 8, 8);
	//create numBlocks regarding the layer dimensions
    dim3 numBlocks((p_param->nb_area_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
    	(p_param->nb_area_h +  threadsPerBlock.y - 1) / threadsPerBlock.y,
    	(current->c_network->batch_size * p_param->nb_maps + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
	pooling_kernel<<< numBlocks , threadsPerBlock >>>(current->input, current->output, 
		p_param->pool_map, p_param->p_size, p_param->prev_size_w, p_param->nb_area_w, 
		p_param->nb_maps * current->c_network->batch_size);
}


void cuda_backward_pool_layer(layer* current)
{	
	p_param = (pool_param*) current->param;

	if(current->previous != NULL)
	{
	
		if(current->previous->type == CONV)
		{
			//array must be set to 0 as deltah_pool do not erase previous values
			cudaMemset(current->previous->delta_o, 0.0, p_param->prev_depth * p_param->prev_size_w 
				* p_param->prev_size_h * current->c_network->batch_size*sizeof(real));
		
			cu_blocks = (current->c_network->batch_size*(p_param->nb_maps * p_param->nb_area_w 
				* p_param->nb_area_h) + cu_threads - 1) / cu_threads;

			deltah_pool_cont<<< cu_blocks, cu_threads >>>(current->delta_o, current->previous->delta_o,
					p_param->pool_map, p_param->p_size, current->c_network->length, 
					current->c_network->batch_size, p_param->nb_maps * p_param->nb_area_w 
					* p_param->nb_area_h, p_param->nb_area_w);
			
		}
		
		current->previous->deriv_activation(current->previous);
	}
}


__global__ void  pooling_kernel(real *input, real *output, real* pool_map, int pool_size, int w_size, int w_size_out, int  length)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y; 
	int k = blockIdx.z*blockDim.z + threadIdx.z;
	int x, y, x_max, y_max, pos, pos_out;
	
	pos_out = k*(w_size_out*w_size_out) + i + j*w_size_out;
	
	pos = k*w_size*w_size + i*pool_size + j*pool_size*w_size;
	
	if(i < w_size_out && j < w_size_out && k < length)
	{
		x_max = 0;
		y_max = 0;
		for(x = 0; x < pool_size; x++)
			for(y = 0; y < pool_size; y++)
				if(input[pos + x_max*w_size + y_max] < input[pos + x*w_size + y])
				{
					x_max = x;
					y_max = y;
				}
		pool_map[pos_out] = (real)(x_max*pool_size + y_max);
		output[pos_out] = input[pos + x_max*w_size + y_max];
	}
}


// Do the same thing as the funciton below but slightly slower
__global__ void deltah_pool(real* delta_o, real* delta_o_unpool, real* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, map_col, map_pos;

	if(i > batch_size*image_size)
		return;

	map_id = i / (map_size*batch_size);
	image_id = i % (map_size*batch_size) / map_size;
	map_col =  i % (map_size*batch_size) % map_size / column_length;
	map_pos = i % (map_size*batch_size) % map_size % column_length;
	
	delta_o_unpool += (map_id * (map_size*batch_size) + image_id * map_size) * pool_size * pool_size;
	delta_o_unpool += map_col * column_length * pool_size * pool_size + map_pos * pool_size;
	delta_o_unpool += + (int(pool_map[i])/pool_size) * column_length * pool_size 
		+ (int(pool_map[i])%pool_size);
	
	if(i < len*image_size)
		*delta_o_unpool = delta_o[i];
	else
		*delta_o_unpool = 0.0;
}

__global__ void deltah_pool_cont(real* delta_o, real* delta_o_unpool, real* pool_map, int pool_size, int len, int batch_size, int image_size, int column_length)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;

	pos = i;
	
	if(i < len*image_size)
	{
		//add mask of locations
		delta_o_unpool += (i/column_length) * column_length * pool_size * pool_size 
			+ (i%column_length) * (pool_size) + (int(pool_map[i])/pool_size) * column_length 
			* pool_size + (int(pool_map[i])%pool_size);
		
		*delta_o_unpool = delta_o[pos];
	}
	
}











