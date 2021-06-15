
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

__global__ void  pooling_kernel_FP32(float* input, float* output, int* pool_map, int pool_size, int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int length);
__global__ void  pooling_kernel_FP16(half* input, half* output, int* pool_map, int pool_size, int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int length);
/*__global__ void deltah_pool_FP32(float* delta_o, float* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length);
__global__ void deltah_pool_FP16(half* delta_o, half* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length);*/
__global__ void deltah_pool_cont_FP32(float* delta_o, float* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int w_size, int h_size);
__global__ void deltah_pool_cont_FP16(half* delta_o, half* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int w_size, int h_size);


void cuda_pool_define(layer *current)
{
	current->forward = cuda_forward_pool_layer;
	current->backprop = cuda_backward_pool_layer;
}

void cuda_convert_pool_layer(layer *current)
{
	p_param = (pool_param*)current->param;

	cuda_convert_table_int(current->c_network, &(p_param->pool_map), p_param->nb_area_w 
		* p_param->nb_area_h * p_param->nb_area_d * p_param->nb_maps * current->c_network->batch_size);
	cuda_convert_table(current->c_network, &(current->output), p_param->nb_area_w 
		* p_param->nb_area_h * p_param->nb_area_d * p_param->nb_maps * current->c_network->batch_size);
	
	cuda_convert_table(current->c_network, &(current->delta_o), p_param->nb_area_w 
		* p_param->nb_area_h * p_param->nb_area_d * p_param->nb_maps * current->c_network->batch_size);
	
	cuda_convert_table(current->c_network, &(p_param->temp_delta_o), p_param->prev_size_w 
		* p_param->prev_size_h * p_param->prev_size_d * p_param->prev_depth * current->c_network->batch_size);
}


void cuda_forward_pool_layer(layer* current)
{
	if(current->c_network->length == 0)
		return;
		
	p_param = (pool_param*) current->param;
	
	//late declaration of CUDA kernel sizes
	dim3 threadsPerBlock(8, 8);
	//create numBlocks regarding the layer dimensions
    dim3 numBlocks((p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d + threadsPerBlock.x - 1) / threadsPerBlock.x,
    	(current->c_network->batch_size * p_param->nb_maps + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			pooling_kernel_FP32<<< numBlocks , threadsPerBlock >>>((float*)current->input, 
				(float*)current->output, p_param->pool_map, p_param->p_size, p_param->prev_size_w,
				p_param->prev_size_h, p_param->prev_size_d, p_param->nb_area_w, p_param->nb_area_h,
				p_param->nb_area_d, p_param->nb_maps * current->c_network->batch_size);
			break;
		case 1:
			pooling_kernel_FP16<<< numBlocks , threadsPerBlock >>>((half*)current->input, 
				(half*)current->output, p_param->pool_map, p_param->p_size, p_param->prev_size_w,
				p_param->prev_size_h, p_param->prev_size_d, p_param->nb_area_w, p_param->nb_area_h,
				p_param->nb_area_d, p_param->nb_maps * current->c_network->batch_size);
			break;
	}
	/*
	cudaDeviceSynchronize();
	
	printf("Nb_maps %d\n", p_param->nb_maps);
	printf("Input Image\n");
	cuda_print_table_4d(current->c_network, current->input, p_param->prev_size_w, p_param->prev_size_h, p_param->prev_size_d, p_param->prev_depth * current->c_network->batch_size, 0);
	
	printf("\n");
	//cuda_print_table_4d(current->c_network, c_param->im2col_input, c_param->nb_area_w*c_param->nb_area_h*c_param->nb_area_d*current->c_network->batch_size, c_param->flat_f_size, 1, 1, 0);
	cuda_print_table(current->c_network, current->output, p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d * current->c_network->batch_size * p_param->nb_maps, p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d);
	cuda_print_table_int(current->c_network, p_param->pool_map, p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d * current->c_network->batch_size * p_param->nb_maps, p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d);
	
	exit(1);*/
}


void cuda_backward_pool_layer(layer* current)
{	
	p_param = (pool_param*) current->param;

	if(current->previous != NULL)
	{
	
		if(current->previous->type == CONV)
		{
			switch(current->c_network->use_cuda_TC)
			{
				default:
				case 0:
					//array must be set to 0 as deltah_pool do not erase previous values
					cudaMemset(current->previous->delta_o, 0.0f, p_param->prev_depth 
						* p_param->prev_size_w * p_param->prev_size_h * p_param->prev_size_d
						* current->c_network->batch_size*sizeof(float));
				
					cu_blocks = (current->c_network->batch_size*(p_param->nb_maps * p_param->nb_area_w 
						* p_param->nb_area_h * p_param->nb_area_d) + cu_threads - 1) / cu_threads;

					deltah_pool_cont_FP32<<< cu_blocks, cu_threads >>>((float*)current->delta_o, 
						(float*)current->previous->delta_o, p_param->pool_map, p_param->p_size, 
						current->c_network->length, current->c_network->batch_size, p_param->nb_maps 
						* p_param->nb_area_w * p_param->nb_area_h, p_param->nb_area_w, p_param->nb_area_h);
					break;
				case 1:
					//array must be set to 0 as deltah_pool do not erase previous values
					cudaMemset(current->previous->delta_o, 0.0f, p_param->prev_depth 
						* p_param->prev_size_w * p_param->prev_size_h *p_param->prev_size_d
						* current->c_network->batch_size*sizeof(half));
				
					cu_blocks = (current->c_network->batch_size*(p_param->nb_maps * p_param->nb_area_w 
						* p_param->nb_area_h * p_param->nb_area_d) + cu_threads - 1) / cu_threads;

					deltah_pool_cont_FP16<<< cu_blocks, cu_threads >>>((half*)current->delta_o, 
						(half*)current->previous->delta_o, p_param->pool_map, p_param->p_size, 
						current->c_network->length, current->c_network->batch_size, p_param->nb_maps 
						* p_param->nb_area_w * p_param->nb_area_h * p_param->nb_area_d, p_param->nb_area_w, p_param->nb_area_h);
					break;
			}
		}
		
		current->previous->deriv_activation(current->previous);
	}
}


__global__ void pooling_kernel_FP32(float* input, float* output, int* pool_map, int pool_size, int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int length)
{
	/*
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = blockIdx.y*blockDim.y + threadIdx.y;
	int x, y, z, x_max, y_max, z_max, pos, pos_x, pos_y, pos_z, pos_out;
	
	pos_z = i / (w_size_out*h_size_out); 
	pos_y = (i % (w_size_out*h_size_out)) / w_size_out;
	pos_x = (i % (w_size_out*h_size_out)) % w_size_out;
	
	pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);
	
	pos = k*w_size*h_size*d_size + pos_x*pool_size + pos_y*pool_size*w_size + pos_z*pool_size*w_size*h_size;
	
	if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)
	{
		x_max = 0;
		y_max = 0;
		z_max = 0;
		for(x = 0; x < pool_size; x++)
			for(y = 0; y < pool_size; y++)
				for(z = 0; z < pool_size; z++)
					if(input[pos + x_max*w_size*h_size + y_max*w_size + z_max] < input[pos + x*w_size*h_size + y*w_size + z])
					{
						x_max = x;
						y_max = y;
						z_max = z;
					}
		pool_map[pos_out] = (x_max*pool_size*pool_size + y_max*pool_size + z_max);
		output[pos_out] = input[pos + x_max*w_size*h_size + y_max*w_size + z_max];
	}*/
}

__global__ void pooling_kernel_FP16(half* input, half* output, int* pool_map, int pool_size, int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int length)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int k = blockIdx.y*blockDim.y + threadIdx.y;
	int x, y, z, x_max, y_max, z_max, pos, pos_x, pos_y, pos_z, pos_out;
	
	pos_z = i / (w_size_out*h_size_out); 
	pos_y = (i % (w_size_out*h_size_out)) / w_size_out;
	pos_x = (i % (w_size_out*h_size_out)) % w_size_out;
	
	pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);
	
	pos = k*w_size*h_size*d_size + pos_x*pool_size + pos_y*pool_size*w_size + pos_z*pool_size*w_size*h_size;
	
	if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)
	{
		x_max = 0;
		y_max = 0;
		z_max = 0;
		for(x = 0; x < pool_size; x++)
			for(y = 0; y < pool_size; y++)
				for(z = 0; z < pool_size; z++)
					if(input[pos + x_max*w_size*h_size + y_max*w_size + z_max] < input[pos + x*w_size*h_size + y*w_size + z])
					{
						x_max = x;
						y_max = y;
						z_max = z;
					}
		pool_map[pos_out] = (x_max*pool_size*pool_size + y_max*pool_size + z_max);
		output[pos_out] = input[pos + x_max*w_size*h_size + y_max*w_size + z_max];
	}
}

/*
// Do the same thing as the funciton below but slightly slower
__global__ void deltah_pool_FP32(float* delta_o, float* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length)
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

__global__ void deltah_pool_FP16(half* delta_o, half* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int map_size, int column_length)
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
	delta_o_unpool += (int(pool_map[i])/pool_size) * column_length * pool_size 
		+ (int(pool_map[i])%pool_size);
	
	if(i < len*image_size)
		*delta_o_unpool = delta_o[i];
	else
		*delta_o_unpool = (half)0.0f;
}
*/

__global__ void deltah_pool_cont_FP32(float* delta_o, float* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int w_size, int h_size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;

	pos = i;
	
	if(i < len*image_size)
	{
		//add mask of locations
		delta_o_unpool += (i/(w_size*h_size)) * (w_size*h_size) * pool_size * pool_size * pool_size
			+ ((i%(w_size*h_size))/w_size) * w_size * pool_size * pool_size
			+ ((i%(w_size*h_size))%w_size) * pool_size +
			+ (int(pool_map[i])/(pool_size*pool_size)) * w_size*h_size * pool_size*pool_size 
			+ ((int(pool_map[i])%(pool_size*pool_size))/pool_size) * w_size * pool_size
			+ ((int(pool_map[i])%(pool_size*pool_size))%pool_size);
		
		*delta_o_unpool = delta_o[pos];
	}
	
}


__global__ void deltah_pool_cont_FP16(half* delta_o, half* delta_o_unpool, int* pool_map, int pool_size, int len, int batch_size, int image_size, int w_size, int h_size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;

	pos = i;
	
	if(i < len*image_size)
	{
		//add mask of locations
		delta_o_unpool += (i/(w_size*h_size)) * (w_size*h_size) * pool_size * pool_size * pool_size
			+ ((i%(w_size*h_size))/w_size) * w_size * pool_size * pool_size
			+ ((i%(w_size*h_size))%w_size) * pool_size +
			+ (int(pool_map[i])/(pool_size*pool_size)) * w_size*h_size * pool_size*pool_size 
			+ ((int(pool_map[i])%(pool_size*pool_size))/pool_size) * w_size * pool_size
			+ ((int(pool_map[i])%(pool_size*pool_size))%pool_size);
		
		*delta_o_unpool = delta_o[pos];
	}
	
}











