
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
static int cu_blocks;
static norm_param *n_param;

// Public are in "prototypes.h"

// Private prototypes
__device__ int cuda_id_to_conv_fmt(int id, int block_id, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size);
__device__ void warpReduce(volatile float *sdata, size_t blockSize, unsigned int tid);
void cuda_forward_norm_layer(layer *current);
void cuda_backward_norm_layer(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead


__device__ int cuda_id_to_conv_fmt(int id, int block_id, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size)
{
	size_t group_id = block_id % nb_group;
	size_t batch_id = block_id / nb_group;
	
	size_t in_group_id = id / flat_a_size;
	size_t map_pos_id = id % flat_a_size;
	
	return batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;
}


__device__ void warpReduce(volatile float *sdata, size_t blockSize, unsigned int tid) 
{
	if (blockSize >= 64)
		sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32)
		sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16)
		sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)
		sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)
		sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)
		sdata[tid] += sdata[tid + 1];
}


#define reduce_group_mean_conv_kernel(name, type) 																								\
__global__ void reduce_group_mean_conv_kernel_##name(void *idata, float *group_mean, 															\
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, int sum_div, size_t sum_size) 									\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	int tid = threadIdx.x;																														\
	int block_id = blockIdx.x;																													\
	size_t blockSize = blockDim.x;																												\
	int i = tid;																																\
	sdata[tid] = 0;																																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		sdata[tid] += (float)input[cuda_id_to_conv_fmt(i, block_id, group_size, nb_group, flat_a_size, batch_size)];							\
		i += blockSize;																															\
	}																																			\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128) 																															\
			sdata[tid] += sdata[tid + 128];																										\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
			sdata[tid] += sdata[tid + 64];																										\
		__syncthreads();																														\
	}																																			\
	if (tid < 32) 																																\
		warpReduce(sdata, blockSize, tid);																										\
	if (tid == 0) 																																\
		group_mean[block_id] = sdata[0]/(sum_div);																								\
}


#define reduce_group_var_conv_kernel(name, type) 																								\
__global__ void reduce_group_var_conv_kernel_##name(void *idata, float *group_var, float *group_mean, 											\
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div, size_t sum_size) 								\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	float l_val;																																\
	sdata[tid] = 0;																																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		l_val = (float)input[cuda_id_to_conv_fmt(i, block_id, group_size, nb_group, flat_a_size, batch_size)];									\
		sdata[tid] += (l_val - group_mean[block_id])*(l_val - group_mean[block_id]);															\
																																				\
		i += blockSize;																															\
	}																																			\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128) 																															\
			sdata[tid] += sdata[tid + 128];																										\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
			sdata[tid] += sdata[tid + 64];																										\
		__syncthreads();																														\
	}																																			\
	if (tid < 32) 																																\
		warpReduce(sdata, blockSize, tid);																										\
	if (tid == 0) 																																\
		group_var[block_id] = sdata[0]/(sum_div);																								\
}


#define reduce_group_dgamma_conv_kernel(name, type) 																							\
__global__ void reduce_group_dgamma_conv_kernel_##name(void *idata, void *d_output, float *d_gamma,												\
	float *group_var, float *group_mean, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_size) 			\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	type* delta_output = (type*) d_output;																										\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	float eps = 0.000001f;																														\
	sdata[tid] = 0;																																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		sdata[tid] += ((float)delta_output[cuda_id_to_conv_fmt(i, block_id, group_size, nb_group, flat_a_size, batch_size)] 					\
			* ((float)input[cuda_id_to_conv_fmt(i, block_id, group_size, nb_group, flat_a_size, batch_size)] - group_mean[block_id]));			\
		i += blockSize;																															\
	}																																			\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128) 																															\
			sdata[tid] += sdata[tid + 128];																										\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
			sdata[tid] += sdata[tid + 64];																										\
		__syncthreads();																														\
	}																																			\
	if (tid < 32) 																																\
		warpReduce(sdata, blockSize, tid);																										\
	if (tid == 0) 																																\
		d_gamma[block_id] = sdata[0]*(1.0f/sqrt(group_var[block_id]+eps));																		\
}


#define group_normalization_conv_kernel(name, type) 																							\
__global__ void group_normalization_conv_kernel_##name(void *i_output, void *i_input, float *gamma, float *beta, float *group_mean,				\
	float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, int nb_filters, size_t flat_a_size, size_t set_off)	\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	float l_val, eps = 0.000001f;																												\
	float mean = 0.0f, var = 0.0f;																												\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t group_id, batch_id;																													\
	size_t in_group_id, map_pos_id, conv_id;																									\
																																				\
	if(i < flat_a_size*group_size && j < nb_group*b_size)																						\
	{																																			\
		group_id = j % nb_group;																												\
		batch_id = j / nb_group;																												\
																																				\
		in_group_id = i / flat_a_size; 																											\
		map_pos_id = i % flat_a_size;																											\
																																				\
		conv_id = batch_id*flat_a_size + (group_id*group_size + in_group_id)*filter_offset + map_pos_id;										\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			mean = group_mean[batch_id*nb_group + group_id];																					\
			var  = group_var[batch_id*nb_group + group_id];																						\
																																				\
			l_val = (float)input[conv_id];																										\
			if(group_id < nb_group - set_off)																									\
				output[conv_id] = (type)(gamma[group_id]*((l_val - mean)/sqrt(var + eps)) + beta[group_id]);									\
			else																																\
				output[conv_id] = input[conv_id];																								\
		}																																		\
		else																																	\
			output[conv_id] = (type) 0.0f;																										\
	}																																			\
}


#define group_normalization_conv_back_kernel(name, type) 																						\
__global__ void group_normalization_conv_back_kernel_##name(																					\
	void *i_input, void *i_delta_output, void *i_delta_input, float *gamma, float *beta, float *d_gamma, float * d_beta, float *group_mean,		\
	float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, int nb_filters, size_t flat_a_size, size_t set_off)	\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* delta_input = (type*) i_delta_input;																									\
	type* delta_output = (type*) i_delta_output;																								\
	float eps = 0.000001f;																														\
	float mean = 0.0f, var = 0.0f;																												\
	float l_d_gamma, l_d_beta;																													\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t group_id, batch_id;																													\
	size_t in_group_id, map_pos_id, conv_id;																									\
																																				\
	if(i < flat_a_size*group_size && j < nb_group*b_size)																						\
	{																																			\
		group_id = j % nb_group;																												\
		batch_id = j / nb_group;																												\
																																				\
		in_group_id = i / flat_a_size; 																											\
		map_pos_id = i % flat_a_size;																											\
																																				\
		conv_id = batch_id*flat_a_size + (group_id*group_size + in_group_id)*filter_offset + map_pos_id;										\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			mean = group_mean[batch_id*nb_group + group_id];																					\
			var  = group_var[batch_id*nb_group + group_id];																						\
			l_d_gamma = d_gamma[batch_id*nb_group + group_id];																					\
			l_d_beta  = d_beta[batch_id*nb_group + group_id];																					\
																																				\
			if(group_id < nb_group - set_off)																									\
				delta_input[conv_id] += (type)((1.0f/(group_size*flat_a_size)) * gamma[group_id] * (1.0f/sqrt(var + eps))						\
					* (group_size*flat_a_size*(float)delta_output[conv_id] - l_d_beta															\
					- ((float)input[conv_id] - mean) * (1.0f/sqrt(var + eps))*l_d_gamma));														\
			else																																\
				delta_input[conv_id] += delta_output[conv_id];																					\
		}																																		\
		else																																	\
			delta_input[conv_id] += (type) 0.0f;																								\
	}																																			\
}


reduce_group_mean_conv_kernel(FP32, float);
reduce_group_var_conv_kernel(FP32, float);
reduce_group_dgamma_conv_kernel(FP32, float);
group_normalization_conv_kernel(FP32, float);
group_normalization_conv_back_kernel(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
reduce_group_mean_conv_kernel(FP16, half);
reduce_group_var_conv_kernel(FP16, half);
reduce_group_dgamma_conv_kernel(FP16, half);
group_normalization_conv_kernel(FP16, half);
group_normalization_conv_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
reduce_group_mean_conv_kernel(BF16, nv_bfloat16);
reduce_group_var_conv_kernel(BF16, nv_bfloat16);
reduce_group_dgamma_conv_kernel(BF16, nv_bfloat16);
group_normalization_conv_kernel(BF16, nv_bfloat16);
group_normalization_conv_back_kernel(BF16, nv_bfloat16);
#endif


void cuda_norm_init(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel = reduce_group_mean_conv_kernel_FP32; 
			net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel = reduce_group_var_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel = group_normalization_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel = group_normalization_conv_back_kernel_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel = reduce_group_mean_conv_kernel_FP16; 
			net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel = reduce_group_var_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel = group_normalization_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel = group_normalization_conv_back_kernel_FP16;
			#else
			printf("\n ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel = reduce_group_mean_conv_kernel_BF16; 
			net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel = reduce_group_var_conv_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel = group_normalization_conv_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel = group_normalization_conv_back_kernel_BF16;
			#else
			printf("\n ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_norm_layer(layer *current)
{
	size_t vram_approx = 0;
	n_param = (norm_param*)current->param;
	network* net = current->c_network;

	vram_approx += cuda_convert_table(net, &(current->output), current->a_size, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->gamma_gpu), n_param->nb_group, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->beta_gpu) , n_param->nb_group, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->mean), n_param->nb_group*net->batch_size, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->var) , n_param->nb_group*net->batch_size, 0);
	
	if(!net->inference_only)
	{		
		vram_approx += cuda_convert_table(net, &(current->delta_o), current->a_size, 0);
		
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->d_gamma_gpu), n_param->nb_group*net->batch_size, 0);
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->d_beta_gpu), n_param->nb_group*net->batch_size, 0);
	}
	
	return vram_approx;
}


void cuda_free_norm(layer *current)
{
	n_param = (norm_param*)current->param;
	
	cudaFree(current->output);

	cudaFree(n_param->mean);
	cudaFree(n_param->var);
	
	cudaFree(n_param->gamma_gpu);
	cudaFree(n_param->beta_gpu);
	
	if(!current->c_network->inference_only)
	{
		cudaFree(current->delta_o);
		cudaFree(n_param->d_gamma_gpu);
		cudaFree(n_param->d_beta_gpu);
	
		//no cuda_free_optimizer_var as optim is done on CPU for norm layers
	}
}


void cuda_forward_norm_layer(layer *current)
{
	size_t i;
	size_t dim_offset = 1, flat_output_dim = 1;
	float *l_gamma, *l_beta;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;
	//Previous verification should ensure that it is not the first layer 
	current->input = current->previous->output;
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		flat_output_dim = dim_offset * current->output_dim[3];
		
		if(net->is_inference == 1 && net->use_wema)
		{
			l_gamma = current->ema_weights;
			l_beta = ((float*)current->ema_weights) + n_param->nb_group;
		}
		else
		{
			l_gamma = n_param->gamma;
			l_beta = n_param->beta;
		}
		
		cuda_put_table_FP32(n_param->gamma_gpu, l_gamma, n_param->nb_group);
		cuda_put_table_FP32(n_param->beta_gpu, l_beta, n_param->nb_group);
		
		cu_blocks = n_param->nb_group*net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel<<<cu_blocks, 256>>>(current->input, 
			n_param->mean, n_param->group_size, n_param->nb_group, dim_offset, net->batch_size, 
			dim_offset*n_param->group_size, dim_offset*n_param->group_size);
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel<<<cu_blocks, 256>>>(current->input, n_param->var, 
			n_param->mean, n_param->group_size, n_param->nb_group, dim_offset, net->batch_size,
			dim_offset*n_param->group_size, dim_offset*n_param->group_size);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((dim_offset*n_param->group_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(n_param->nb_group*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel<<<numBlocks,threadsPerBlock>>>(
			current->output, current->input, n_param->gamma_gpu, n_param->beta_gpu, n_param->mean, n_param->var, net->length, 
			net->batch_size, n_param->group_size, n_param->nb_group, current->output_dim[3], dim_offset, n_param->set_off);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, flat_output_dim * net->batch_size);
}


void cuda_backward_norm_layer(layer *current)
{
	int i, j;
	size_t dim_offset = 1;
	float sum_dgamma = 0.0f, sum_dbeta = 0.0f;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;	
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		
		cuda_put_table_FP32(n_param->gamma_gpu, n_param->gamma, n_param->nb_group);
		cuda_put_table_FP32(n_param->beta_gpu, n_param->beta, n_param->nb_group);
		
		cu_blocks = n_param->nb_group*net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel<<<cu_blocks, 256>>>(current->delta_o, 
			n_param->d_beta_gpu, n_param->group_size, n_param->nb_group, dim_offset, net->batch_size, 1, 
			dim_offset*n_param->group_size);
			
		net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel<<<cu_blocks, 256>>>(current->input, current->delta_o,
			n_param->d_gamma_gpu, n_param->var, n_param->mean, n_param->group_size, n_param->nb_group, dim_offset, net->batch_size,
			dim_offset*n_param->group_size);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((dim_offset*n_param->group_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(n_param->nb_group*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
			
		net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel<<<numBlocks, threadsPerBlock>>>(
			current->input, current->delta_o, current->previous->delta_o, n_param->gamma_gpu, n_param->beta_gpu, 
			n_param->d_gamma_gpu, n_param->d_beta_gpu, n_param->mean, n_param->var, net->length, net->batch_size, 
			n_param->group_size, n_param->nb_group, current->output_dim[3], dim_offset, n_param->set_off);
	}
	
	if(!current->frozen)
	{
		cuda_get_table_FP32(n_param->d_gamma_gpu, n_param->d_gamma, n_param->nb_group*net->batch_size);
		cuda_get_table_FP32(n_param->d_beta_gpu, n_param->d_beta, n_param->nb_group*net->batch_size);
	
		for(j = 0; j < n_param->nb_group - n_param->set_off; j++)
		{
			sum_dgamma = 0.0f;
			sum_dbeta = 0.0f;
			for(i = 0; i < net->batch_size; i++)
			{
				sum_dgamma += n_param->d_gamma[i*n_param->nb_group + j];
				sum_dbeta  += n_param->d_beta[i*n_param->nb_group + j];
			}
			n_param->gamma_update[j] = sum_dgamma;
			n_param->beta_update[j] = sum_dbeta;
		}
		
		net->optim_update_fct(current, 0, 2*n_param->nb_group, 2*n_param->nb_group);
		//No decay for gamma and beta
	
		if(current->wema_replace_signal > 0)
		{
			for(i = 0; i < 2*n_param->nb_group; i++)
				current->FP32_weights[i] = current->ema_weights[i];
			current->wema_replace_signal = 0;
		}
	}
}


void cuda_norm_define(layer *current)
{
	current->forward = cuda_forward_norm_layer;
	current->backprop = cuda_backward_norm_layer;
}


