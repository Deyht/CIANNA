

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
static norm_param *n_param;

//public are in prototypes.h

//#####################################################
//       Layer normalization related templates
//#####################################################




__device__ int cuda_id_to_conv_fmt(int id, int block_id, int group_size, int nb_group, int flat_a_size, int batch_size)
{
	int group_id = block_id % nb_group;
	int batch_id = block_id / nb_group;
	
	int in_group_id = id / flat_a_size;
	int map_pos_id = id % flat_a_size;
	
	return batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;
}

__device__ void warpReduce(volatile float *sdata, int blockSize, unsigned int tid) 
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
	int group_size, int nb_group, int flat_a_size, int batch_size, int sum_div, int sum_size) 													\
{																																				\
	extern __shared__ float sdata[];																											\
	type* input = (type*) idata;																												\
	int tid = threadIdx.x;																														\
	int block_id = blockIdx.x;																													\
	int blockSize = blockDim.x;																													\
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
	int group_size, int nb_group, int flat_a_size, int batch_size, int sum_div, int sum_size) 													\
{																																				\
	extern __shared__ float sdata[];																											\
	type* input = (type*) idata;																												\
	int tid = threadIdx.x;																														\
	int block_id = blockIdx.x;																													\
	int blockSize = blockDim.x;																													\
	int i = tid;																																\
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
	float *group_var, float *group_mean, int group_size, int nb_group, int flat_a_size, int batch_size, int sum_size) 							\
{																																				\
	extern __shared__ float sdata[];																											\
	type* input = (type*) idata;																												\
	type* delta_output = (type*) d_output;																										\
	int tid = threadIdx.x;																														\
	int block_id = blockIdx.x;																													\
	int blockSize = blockDim.x;																													\
	int i = tid;																																\
	float eps = 0.00001f;																														\
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
		d_gamma[block_id] = (1.0f/sqrt(group_var[block_id]+eps))*sdata[0];																		\
}

#define group_normalization_conv_kernel(name, type) 																							\
__global__ void group_normalization_conv_kernel_##name(void *i_output, void *i_input, float *gamma, float *beta, float *group_mean,				\
	float *group_var, int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off)						\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	float l_val, eps = 0.00001f;																												\
	float mean = 0.0f, var = 0.0f;																												\
	int filter_offset = flat_a_size*b_size;																										\
	int group_id, batch_id;																														\
	int in_group_id, map_pos_id, conv_id;																										\
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
	float *group_var, int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off)						\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* delta_input = (type*) i_delta_input;																									\
	type* delta_output = (type*) i_delta_output;																								\
	float eps = 0.00001f;																														\
	float mean = 0.0f, var = 0.0f;																												\
	float l_d_gamma, l_d_beta;																													\
	int filter_offset = flat_a_size*b_size;																										\
	int group_id, batch_id;																														\
	int in_group_id, map_pos_id, conv_id;																										\
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
				delta_input[conv_id] = (type)((1.0f/(group_size*flat_a_size)) * gamma[group_id] * (1.0f/sqrt(var + eps))						\
					* (group_size*flat_a_size*(float)delta_output[conv_id] - l_d_beta															\
					- ((float)input[conv_id] - mean) * (1.0f/sqrt(var + eps))*l_d_gamma));														\
			else																																\
				delta_input[conv_id] = delta_output[conv_id];																					\
		}																																		\
		else																																	\
			delta_input[conv_id] = (type) 0.0f;																									\
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
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
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
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_norm_layer(layer *current)
{
	n_param = (norm_param*)current->param;
	size_t vram_approx = 0;

	network* net = current->c_network;

	vram_approx += cuda_convert_table(net, &(current->output), n_param->output_dim, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->gamma_gpu), n_param->nb_group, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->beta_gpu), n_param->nb_group, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->mean), n_param->nb_group*net->batch_size, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->var), n_param->nb_group*net->batch_size, 0);
	
	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table(net, &(current->delta_o), n_param->output_dim, 0);
		
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->d_gamma_gpu), n_param->nb_group*net->batch_size, 0);
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->d_beta_gpu), n_param->nb_group*net->batch_size, 0);
	}
	
	return vram_approx;
}


void cuda_forward_norm_layer(layer *current)
{
	n_param = (norm_param*)current->param;		
	network* net = current->c_network;
	
	current->input = current->previous->output;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{
		cuda_put_table_FP32(n_param->gamma_gpu, n_param->gamma, n_param->nb_group);
		cuda_put_table_FP32(n_param->beta_gpu, n_param->beta, n_param->nb_group);
		
		cu_blocks = n_param->nb_group*net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel<<<cu_blocks, 256>>>(current->input, 
			n_param->mean, n_param->group_size, n_param->nb_group, n_param->dim_offset, net->batch_size, 
			n_param->dim_offset*n_param->group_size, n_param->dim_offset*n_param->group_size);
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel<<<cu_blocks, 256>>>(current->input, n_param->var, 
			n_param->mean, n_param->group_size, n_param->nb_group, n_param->dim_offset, net->batch_size,
			n_param->dim_offset*n_param->group_size, n_param->dim_offset*n_param->group_size);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((n_param->dim_offset*n_param->group_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(n_param->nb_group*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel<<<numBlocks,threadsPerBlock>>>(
			current->output, current->input, n_param->gamma_gpu, n_param->beta_gpu, n_param->mean, n_param->var, net->length, 
			net->batch_size, n_param->group_size, n_param->nb_group, n_param->n_dim, n_param->dim_offset, n_param->set_off);
	}
}

void cuda_backward_norm_layer(layer *current)
{
	int i, j;
	float sum_dgamma = 0.0f, sum_dbeta = 0.0f;
	n_param = (norm_param*)current->param;	
	network* net = current->c_network;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{	
		cuda_put_table_FP32(n_param->gamma_gpu, n_param->gamma, n_param->nb_group);
		cuda_put_table_FP32(n_param->beta_gpu, n_param->beta, n_param->nb_group);
		
		cu_blocks = n_param->nb_group*net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel<<<cu_blocks, 256>>>(current->delta_o, 
			n_param->d_beta_gpu, n_param->group_size, n_param->nb_group, n_param->dim_offset, net->batch_size, 1, 
			n_param->dim_offset*n_param->group_size);
			
		net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel<<<cu_blocks, 256>>>(current->input, current->delta_o,
			n_param->d_gamma_gpu, n_param->var, n_param->mean, n_param->group_size, n_param->nb_group, n_param->dim_offset, net->batch_size,
			n_param->dim_offset*n_param->group_size);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((n_param->dim_offset*n_param->group_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(n_param->nb_group*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
			
		net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel<<<numBlocks, threadsPerBlock>>>(
			current->input, current->delta_o, current->previous->delta_o, n_param->gamma_gpu, n_param->beta_gpu, 
			n_param->d_gamma_gpu, n_param->d_beta_gpu, n_param->mean, n_param->var, net->length, net->batch_size, 
			n_param->group_size, n_param->nb_group, n_param->n_dim, n_param->dim_offset, n_param->set_off);
		
		cuda_get_table_FP32(n_param->d_gamma_gpu, n_param->d_gamma, n_param->nb_group*net->batch_size);
		cuda_get_table_FP32(n_param->d_beta_gpu, n_param->d_beta, n_param->nb_group*net->batch_size);
		
		if(!current->frozen)
		{
			for(j = 0; j < n_param->nb_group - n_param->set_off; j++)
			{
				sum_dgamma = 0.0f;
				sum_dbeta = 0.0f;
				for(i = 0; i < net->batch_size; i++)
				{
					sum_dgamma += n_param->d_gamma[i*n_param->nb_group + j];
					sum_dbeta  += n_param->d_beta[i*n_param->nb_group + j];
				}

				n_param->gamma_update[j] = net->momentum*n_param->gamma_update[j] + net->learning_rate*sum_dgamma;
				n_param->beta_update[j]  = net->momentum*n_param->beta_update[j]  + net->learning_rate*sum_dbeta;
				
				n_param->gamma[j] -= n_param->gamma_update[j] / net->TC_scale_factor;
				n_param->beta[j] -= n_param->beta_update[j] / net->TC_scale_factor; 
			}
		}
	}
	
	current->previous->deriv_activation(current->previous);
}

void cuda_norm_define(layer *current)
{
	current->forward = cuda_forward_norm_layer;
	current->backprop = cuda_backward_norm_layer;
}


