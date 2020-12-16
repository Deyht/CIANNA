
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
static dense_param *d_param;

//public are in prototypes.h

//private
void cuda_forward_dense_layer(layer *current);
void cuda_backward_dense_layer(layer* current);

__global__ void cuda_flat_dense_FP32(float* in, float* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_flat_dense_FP16(half* in, half* out, half bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_reroll_batch_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_reroll_batch_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void init_block_state(unsigned int seed, curandState_t* states);
__global__ void cuda_dropout_select(int* mask, int size, float drop_rate, curandState_t* states);
__global__ void cuda_dropout_apply_FP32(float* table, int batch_size, int dim, int* mask);
__global__ void cuda_dropout_apply_FP16(half* table, int batch_size, int dim, int* mask);

void cuda_dense_define(layer *current)
{
	current->forward = cuda_forward_dense_layer;
	current->backprop = cuda_backward_dense_layer;
}


void cuda_convert_dense_layer(layer *current)
{
	d_param = (dense_param*)current->param;
	float* temp_tab;
	
	if(current->previous != NULL)
	{
		switch(current->previous->type)
		{	
			case CONV:
				cuda_convert_table(current->c_network, &(d_param->flat_input), d_param->in_size*current->c_network->batch_size);
				cuda_convert_table(current->c_network, &(d_param->flat_delta_o),
					(((conv_param*)current->previous->param)->nb_area_w 
						* ((conv_param*)current->previous->param)->nb_area_h 
						* ((conv_param*)current->previous->param)->nb_filters + 1) 
						* current->c_network->batch_size);
				break;
				
			case POOL:
				cuda_convert_table(current->c_network, &(d_param->flat_input), d_param->in_size * current->c_network->batch_size);
				cuda_convert_table(current->c_network, &(d_param->flat_delta_o),
					(((pool_param*)current->previous->param)->nb_area_w 
						* ((pool_param*)current->previous->param)->nb_area_h 
						* ((pool_param*)current->previous->param)->nb_maps + 1) 
						* current->c_network->batch_size);
				break;
				
			case DENSE:
			default:
				d_param->flat_delta_o = current->previous->delta_o;
				break;
		}
	}
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			cuda_convert_table(current->c_network, &(d_param->weights), d_param->in_size*(d_param->nb_neurons+1));
			break;
		case 1:
			temp_tab = (float*)d_param->weights;
			cudaMalloc(&(d_param->FP32_weights), d_param->in_size*(d_param->nb_neurons+1)*sizeof(float));
			cudaMemcpy(d_param->FP32_weights, temp_tab, d_param->in_size 
				* (d_param->nb_neurons+1) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(d_param->weights),d_param->in_size*(d_param->nb_neurons+1)*sizeof(half));
			break;
	}
	
	cuda_convert_table(current->c_network, &(d_param->update), d_param->in_size*(d_param->nb_neurons+1));
	cuda_convert_table_int(current->c_network, &(d_param->dropout_mask), d_param->nb_neurons);
	cudaMalloc((void**) &d_param->block_state, (d_param->nb_neurons) * sizeof(curandState_t));
	cu_blocks = (d_param->nb_neurons);
	init_block_state<<< cu_blocks, 1>>>(time(NULL),(curandState_t*)d_param->block_state);
	
	cuda_convert_table(current->c_network, &(current->output), (d_param->nb_neurons+1) 
		* current->c_network->batch_size);
	cuda_convert_table(current->c_network, &(current->delta_o), (d_param->nb_neurons+1) 
		* current->c_network->batch_size);
}


void cuda_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, depth;
	
	void* ref_input;
	
	if(current->c_network->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->c_network->use_cuda_TC == 1)
		cuda_master_weight_FP32_to_FP16((float*)d_param->FP32_weights, (half*)d_param->weights, 
			d_param->in_size*(d_param->nb_neurons+1));
	
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
		
		cu_blocks = ((nb_area_w * nb_area_h * depth + 1) 
			* current->c_network->batch_size + cu_threads - 1) / cu_threads;
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_flat_dense_FP32<<< cu_blocks, cu_threads >>>((float*)current->input, 
					(float*)d_param->flat_input, d_param->bias_value, nb_area_w * nb_area_h,
					nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size, 
					(nb_area_w * nb_area_h * depth + 1) * current->c_network->batch_size);
				break;
			case 1:
				cuda_flat_dense_FP16<<< cu_blocks, cu_threads >>>((half*)current->input, 
					(half*)d_param->flat_input, d_param->bias_value, nb_area_w * nb_area_h, 
					nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size, 
					(nb_area_w * nb_area_h * depth + 1) * current->c_network->batch_size);
				break;
		}
		
		ref_input = d_param->flat_input;
	}
	
	cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, 
		current->c_network->batch_size, d_param->in_size, &cu_alpha, d_param->weights, cuda_data_type, 
		d_param->nb_neurons+1, ref_input, cuda_data_type, d_param->in_size, &cu_beta, 
		current->output, cuda_data_type, d_param->nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
	current->activation(current);

	if(d_param->dropout_rate > 0.01)
	{
		// Must check performance impact -> the present approach is due to the curand behavior
		cu_blocks = (d_param->nb_neurons);
		cuda_dropout_select<<<cu_blocks, 1>>>(d_param->dropout_mask, d_param->nb_neurons+1, 
			d_param->dropout_rate, (curandState_t*) d_param->block_state);	

		dim3 threadsPerBlock(8, 32);
		dim3 numBlocks((current->c_network->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_dropout_apply_FP32<<<numBlocks, threadsPerBlock>>>((float*)current->output, 
					current->c_network->batch_size, d_param->nb_neurons, d_param->dropout_mask);
				break;
			case 1:
				cuda_dropout_apply_FP16<<<numBlocks, threadsPerBlock>>>((half*)current->output, 
					current->c_network->batch_size, d_param->nb_neurons, d_param->dropout_mask);
				break;
		}
	}
}


void cuda_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, depth;
	void* ref_input;

	d_param = (dense_param*) current->param;	
	
	dim3 threadsPerBlock(8, 32);
	dim3 numBlocks((current->c_network->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
	if(d_param->dropout_rate > 0.01)
	{
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_dropout_apply_FP32<<<numBlocks, threadsPerBlock>>>((float*)current->delta_o, 
					current->c_network->batch_size, d_param->nb_neurons, d_param->dropout_mask);
				break;
			case 1:
				cuda_dropout_apply_FP16<<<numBlocks, threadsPerBlock>>>((half*)current->delta_o, 
					current->c_network->batch_size, d_param->nb_neurons, d_param->dropout_mask);
				break;
		}
	}
	
	//######################## ERROR PROPAGATION ########################

	ref_input = current->input;

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, d_param->in_size, 
			current->c_network->batch_size, d_param->nb_neurons+1, &cu_alpha, d_param->weights, cuda_data_type, 
			d_param->nb_neurons+1, current->delta_o, cuda_data_type, d_param->nb_neurons+1, &cu_beta, 
			d_param->flat_delta_o, cuda_data_type, d_param->in_size, cuda_compute_type,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
			cu_blocks = (nb_area_w * nb_area_h * depth 
				* current->c_network->batch_size + cu_threads - 1) / cu_threads;
				
			switch(current->c_network->use_cuda_TC)
			{
				default:
				case 0:
					cuda_reroll_batch_FP32<<< cu_blocks, cu_threads >>>((float*)d_param->flat_delta_o, 
						(float*)current->previous->delta_o, nb_area_w * nb_area_h, 
						nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size,
						nb_area_w * nb_area_h * depth * current->c_network->batch_size);
					break;
				case 1:
					cuda_reroll_batch_FP16<<< cu_blocks, cu_threads >>>((half*)d_param->flat_delta_o,
						(half*)current->previous->delta_o, nb_area_w * nb_area_h,
						nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size,
						nb_area_w * nb_area_h * depth * current->c_network->batch_size);
					break;
			}
		}
		
		current->previous->deriv_activation(current->previous);
	}
	
		
	//########################  WEIGHTS UPDATE   ########################
	
	if(current->previous != NULL && current->previous->type != DENSE)
		ref_input = d_param->flat_input;
		
	cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
		current->c_network->batch_size, &current->c_network->learning_rate,	current->delta_o, cuda_data_type, 
		d_param->nb_neurons+1, ref_input, cuda_data_type, d_param->in_size, &current->c_network->momentum, 
		d_param->update, cuda_data_type, d_param->nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
	switch(current->c_network->use_cuda_TC)
	{
		case 0:
			cuda_update_weights(current->c_network, d_param->weights, d_param->update, d_param->in_size 
				* (d_param->nb_neurons+1));
			break;
		case 1:
			cuda_update_weights(current->c_network, d_param->FP32_weights, d_param->update, d_param->in_size 
				* (d_param->nb_neurons+1));
			break;
	}
}


//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
__global__ void cuda_flat_dense_FP32(float* in, float* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	//SHOULD TEST OPTIMIZATION
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;

	if(i < size)
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

__global__ void cuda_flat_dense_FP16(half* in, half* out, half bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	//SHOULD TEST OPTIMIZATION
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;

	if(i < size)
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

__global__ void cuda_reroll_batch_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}

__global__ void cuda_reroll_batch_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}

__global__ void init_block_state(unsigned int seed,  curandState_t* states)
{
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}


__global__ void cuda_dropout_select(int* mask, int size, float drop_rate, curandState_t* states)
{
	int i = blockIdx.x;
	
	float rand;
	if(i < size)
	{
		rand = curand_uniform(&states[i]);
		if(rand < drop_rate)
			mask[i] = 0;
		else
			mask[i] = 1;
	}
}

__global__ void cuda_dropout_apply_FP32(float* table, int batch_size, int dim, int* mask)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(i < batch_size && j < dim)
	{
		table[i*(dim+1) + j] *= mask[j];
	}
}

__global__ void cuda_dropout_apply_FP16(half* table, int batch_size, int dim, int* mask)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(i < batch_size && j < dim)
	{
		table[i*(dim+1) + j] *= mask[j];
	}
}











