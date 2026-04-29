
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
void cuda_forward_norm_layer(layer *current);
void cuda_backward_norm_layer(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead


inline __device__ void warpReduce(volatile float *sdata, size_t blockSize, unsigned int tid) 
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
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div, size_t sum_size) 								\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t conv_id, group_id, batch_id, in_group_id, map_pos_id;																				\
	double sum = 0.0f;																															\
																																				\
	group_id = block_id % nb_group;																												\
	batch_id = block_id / nb_group;																												\
																																				\
	while (i < sum_size)																														\
	{																																			\
		in_group_id = i / flat_a_size;																											\
		map_pos_id  = i % flat_a_size;																											\
		conv_id     = batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;							\
																																				\
		sum += (float)input[conv_id];																											\
		i += blockSize;																															\
	}																																			\
	sdata[tid] = sum;																															\
																																				\
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
		group_mean[block_id] = sdata[0]/sum_div;																								\
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
	size_t conv_id, group_id, batch_id, in_group_id, map_pos_id;																				\
	float l_val;																																\
	double sum = 0.0f;																															\
																																				\
	group_id = block_id % nb_group;																												\
	batch_id = block_id / nb_group;																												\
																																				\
	while (i < sum_size)																														\
	{																																			\
		in_group_id = i / flat_a_size;																											\
		map_pos_id  = i % flat_a_size;																											\
		conv_id     = batch_id * flat_a_size + (group_id * group_size + in_group_id) * flat_a_size * batch_size + map_pos_id;					\
																																				\
		l_val = (float)input[conv_id];																											\
		sum += (l_val - group_mean[block_id])*(l_val - group_mean[block_id]);																	\
		i += blockSize;																															\
	}																																			\
	sdata[tid] = sum;																															\
																																				\
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
		group_var[block_id] = sdata[0]/sum_div;																									\
}

#define reduce_norm_dbeta_conv_kernel(name, type) 																								\
__global__ void reduce_norm_dbeta_conv_kernel_##name(void *i_d_output, void *i_d_beta,															\
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size)																	\
{																																				\
	__shared__ float sdata[256];																												\
	type* d_output = (type*) i_d_output;																										\
	type* d_beta = (type*) i_d_beta;																											\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t feature_id, batch_id, conv_id;																										\
	float sum = 0.0f;																															\
																																				\
	feature_id = block_id % nb_features;																										\
	batch_id   = block_id / nb_features;																										\
	conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;																\
																																				\
	while(i < sum_size)																															\
	{																																			\
		sum += (float)d_output[conv_id + i];																									\
		i += blockSize;																															\
	}																																			\
	sdata[tid] = sum;																															\
																																				\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128)																															\
			sdata[tid] += sdata[tid + 128];																										\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
			sdata[tid] += sdata[tid + 64];																										\
		__syncthreads();																														\
	}																																			\
	if (tid < 32)																																\
		warpReduce(sdata, blockSize, tid);																										\
	if (tid == 0)																																\
		d_beta[block_id] = (type) sdata[0];																										\
}


#define reduce_group_dgamma_conv_kernel(name, type) 																							\
__global__ void reduce_group_dgamma_conv_kernel_##name(void *idata, void *i_d_output, void *i_d_gamma,											\
	float *group_var, float *group_mean, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_size) 			\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	type* d_output = (type*) i_d_output;																										\
	type* d_gamma = (type*) i_d_gamma;																											\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	float eps = 0.000001f;																														\
	size_t nb_features = group_size * nb_group;																									\
	size_t conv_id, feature_id, batch_id, group_id, group_pos;																					\
	double sum = 0.0f;																															\
																																				\
	feature_id = block_id % nb_features;																										\
	batch_id   = block_id / nb_features;																										\
	group_id   = feature_id / group_size;																										\
	group_pos  = batch_id * nb_group + group_id;																								\
	conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		sum += (float)d_output[conv_id + i] * ((float)input[conv_id + i] - group_mean[group_pos]);												\
		i += blockSize;																															\
	}																																			\
	sdata[tid] = sum;																															\
																																				\
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
		d_gamma[block_id] = (type) (sdata[0]*(1.0f/sqrt(group_var[group_pos]+eps)));															\
}


#define reduce_norm_AB_kernel(name, type) 																										\
__global__ void reduce_norm_AB_kernel_##name(void *i_d_gamma, void *i_d_beta, 																	\
	float *gamma, float *A, float *B, size_t group_size, size_t nb_group)																		\
{																																				\
	__shared__ float sdata_A[256];																												\
	__shared__ float sdata_B[256];																												\
	type* d_gamma = (type*) i_d_gamma;																											\
	type* d_beta  = (type*) i_d_beta;																											\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t batch_id, group_id, feature_id, nb_features, grad_id;																				\
	double sum_A = 0.0f;																														\
	double sum_B = 0.0f;																														\
																																				\
	nb_features = group_size * nb_group;																										\
	group_id = block_id % nb_group;																												\
	batch_id = block_id / nb_group;																												\
																																				\
	while(i < group_size)																														\
	{																																			\
		feature_id = group_id * group_size + i;																									\
		grad_id = batch_id * nb_features + feature_id;																							\
																																				\
		sum_A += gamma[feature_id] * (float)d_beta[grad_id];																					\
		sum_B += gamma[feature_id] * (float)d_gamma[grad_id];																					\
		i += blockSize;																															\
	}																																			\
	sdata_A[tid] = sum_A;																														\
	sdata_B[tid] = sum_B;																														\
																																				\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128)																															\
		{																																		\
			sdata_A[tid] += sdata_A[tid + 128];																									\
			sdata_B[tid] += sdata_B[tid + 128];																									\
		}																																		\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
		{																																		\
			sdata_A[tid] += sdata_A[tid + 64];																									\
			sdata_B[tid] += sdata_B[tid + 64];																									\
		}																																		\
		__syncthreads();																														\
	}																																			\
	if (tid < 32)																																\
	{																																			\
		warpReduce(sdata_A, blockSize, tid);																									\
		warpReduce(sdata_B, blockSize, tid);																									\
	}																																			\
																																				\
	if (tid == 0)																																\
	{																																			\
		A[block_id] = sdata_A[0];																												\
		B[block_id] = sdata_B[0];																												\
	}																																			\
}


#define reduce_norm_param_grads_kernel(name, type) 																								\
__global__ void reduce_norm_param_grads_kernel_##name(void *i_d_gamma, void *i_d_beta, void *i_gamma_grad, void *i_beta_grad,					\
	size_t nb_features, size_t batch_size)																										\
{																																				\
	__shared__ float sdata_gamma[256];																											\
	__shared__ float sdata_beta[256];																											\
	type* d_gamma = (type*) i_d_gamma;																											\
	type* d_beta  = (type*) i_d_beta;																											\
	type* gamma_grad = (type*) i_gamma_grad;																									\
	type* beta_grad = (type*) i_beta_grad;																										\
	size_t tid = threadIdx.x;																													\
	size_t feature_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t l_id;																																\
	double sum_gamma = 0.0f;																													\
	double sum_beta  = 0.0f;																													\
																																				\
	while(i < batch_size)																														\
	{																																			\
		l_id = i*nb_features + feature_id;																										\
		sum_gamma += (float)d_gamma[l_id];																										\
		sum_beta  += (float)d_beta[l_id];																										\
		i += blockSize;																															\
	}																																			\
	sdata_gamma[tid] = sum_gamma;																												\
	sdata_beta[tid]  = sum_beta;																												\
																																				\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if (tid < 128)																															\
		{																																		\
			sdata_gamma[tid] += sdata_gamma[tid + 128];																							\
			sdata_beta[tid]  += sdata_beta[tid + 128];																							\
		}																																		\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if (tid < 64)																															\
		{																																		\
			sdata_gamma[tid] += sdata_gamma[tid + 64];																							\
			sdata_beta[tid]  += sdata_beta[tid + 64];																							\
		}																																		\
		__syncthreads();																														\
	}																																			\
	if (tid < 32)																																\
	{																																			\
		warpReduce(sdata_gamma, blockSize, tid);																								\
		warpReduce(sdata_beta,  blockSize, tid);																								\
	}																																			\
																																				\
	if (tid == 0)																																\
	{																																			\
		gamma_grad[feature_id] = sdata_gamma[0];																								\
		beta_grad[feature_id]  = sdata_beta[0];																									\
	}																																			\
}


#define group_normalization_conv_kernel(name, type) 																							\
__global__ void group_normalization_conv_kernel_##name(void *i_output, void *i_input, float *gamma, float *beta, float *group_mean,				\
	float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t nb_filters, size_t flat_a_size)				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	float l_val, eps = 0.000001f;																												\
	float mean = 0.0f, var = 0.0f;																												\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t group_id, batch_id, feature_id;																										\
	size_t in_group_id, map_pos_id, conv_id;																									\
																																				\
	if(i < flat_a_size*group_size && j < nb_group*b_size)																						\
	{																																			\
		group_id    = j % nb_group;																												\
		batch_id    = j / nb_group;																												\
		in_group_id = i / flat_a_size; 																											\
		map_pos_id  = i % flat_a_size;																											\
		feature_id  = group_id * group_size + in_group_id;																						\
		conv_id     = batch_id * flat_a_size + (group_id * group_size + in_group_id) * filter_offset + map_pos_id;								\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			mean = group_mean[batch_id*nb_group + group_id];																					\
			var  = group_var[batch_id*nb_group + group_id];																						\
																																				\
			l_val = (float)input[conv_id];																										\
			output[conv_id] = (type)(gamma[feature_id]*((l_val - mean)/sqrt(var + eps)) + beta[feature_id]);									\
		}																																		\
		else																																	\
			output[conv_id] = (type) 0.0f;																										\
	}																																			\
}


#define group_normalization_conv_back_kernel(name, type) 																						\
__global__ void group_normalization_conv_back_kernel_##name(																					\
	void *i_input, void *i_d_output, void *i_d_input, float *gamma, float *A, float *B, float *group_mean,										\
	float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t nb_filters, size_t flat_a_size)				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* d_input = (type*) i_d_input;																											\
	type* d_output = (type*) i_d_output;																										\
	float eps = 0.000001f;																														\
	float mean = 0.0f, var = 0.0f;																												\
	float l_A, l_B;																																\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t group_id, batch_id, feature_id;																										\
	size_t in_group_id, map_pos_id, conv_id;																									\
																																				\
	if(i < flat_a_size*group_size && j < nb_group*b_size)																						\
	{																																			\
		group_id    = j % nb_group;																												\
		batch_id    = j / nb_group;																												\
		in_group_id = i / flat_a_size; 																											\
		map_pos_id  = i % flat_a_size;																											\
		feature_id  = group_id * group_size + in_group_id;																						\
		conv_id     = batch_id * flat_a_size + (group_id * group_size + in_group_id) * filter_offset + map_pos_id;								\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			mean = group_mean[batch_id*nb_group + group_id];																					\
			var  = group_var[batch_id*nb_group + group_id];																						\
			l_A = A[batch_id*nb_group + group_id];																								\
			l_B = B[batch_id*nb_group + group_id];																								\
																																				\
			d_input[conv_id] += (type)((1.0f/(group_size*flat_a_size)) * (1.0f/sqrt(var + eps))													\
				* (gamma[feature_id]*group_size*flat_a_size*(float)d_output[conv_id] - l_A														\
				- ((float)input[conv_id] - mean) * (1.0f/sqrt(var + eps)) * l_B));																\
		}																																		\
	}																																			\
}


reduce_group_mean_conv_kernel(FP32, float);
reduce_group_var_conv_kernel(FP32, float);
reduce_norm_dbeta_conv_kernel(FP32, float);
reduce_group_dgamma_conv_kernel(FP32, float);
reduce_norm_AB_kernel(FP32, float);
reduce_norm_param_grads_kernel(FP32, float);
group_normalization_conv_kernel(FP32, float);
group_normalization_conv_back_kernel(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
reduce_group_mean_conv_kernel(FP16, half);
reduce_group_var_conv_kernel(FP16, half);
reduce_norm_dbeta_conv_kernel(FP16, half);
reduce_group_dgamma_conv_kernel(FP16, half);
reduce_norm_AB_kernel(FP16, half);
reduce_norm_param_grads_kernel(FP16, half);
group_normalization_conv_kernel(FP16, half);
group_normalization_conv_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
reduce_group_mean_conv_kernel(BF16, nv_bfloat16);
reduce_group_var_conv_kernel(BF16, nv_bfloat16);
reduce_norm_dbeta_conv_kernel(BF16, nv_bfloat16);
reduce_group_dgamma_conv_kernel(BF16, nv_bfloat16);
reduce_norm_AB_kernel(BF16, nv_bfloat16);
reduce_norm_param_grads_kernel(BF16, nv_bfloat16);
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
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_dbeta_conv_kernel = reduce_norm_dbeta_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_AB_kernel = reduce_norm_AB_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_param_grads_kernel = reduce_norm_param_grads_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_kernel = group_normalization_conv_kernel_FP32;
			net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel = group_normalization_conv_back_kernel_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_norm_fcts.cu_reduce_group_mean_conv_kernel = reduce_group_mean_conv_kernel_FP16; 
			net->cu_inst.cu_norm_fcts.cu_reduce_group_var_conv_kernel = reduce_group_var_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_dbeta_conv_kernel = reduce_norm_dbeta_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_AB_kernel = reduce_norm_AB_kernel_FP16;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_param_grads_kernel = reduce_norm_param_grads_kernel_FP16;
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
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_dbeta_conv_kernel = reduce_norm_dbeta_conv_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel = reduce_group_dgamma_conv_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_AB_kernel = reduce_norm_AB_kernel_BF16;
			net->cu_inst.cu_norm_fcts.cu_reduce_norm_param_grads_kernel = reduce_norm_param_grads_kernel_BF16;
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
	size_t vram_approx = 0, temp_allocated_size = 0;
	size_t nb_features;
	n_param = (norm_param*)current->param;
	network* net = current->c_network;

	nb_features = current->output_dim[3];

	vram_approx += cuda_convert_table(net, &(current->output), current->a_size, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(current->weights), 2*nb_features, 0);
	current->FP32_weights = (float*)current->weights;
	n_param->gamma = (float*) current->weights;
	n_param->beta = ((float*) current->weights) + nb_features;
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->mean), n_param->nb_group*net->batch_size, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->var) , n_param->nb_group*net->batch_size, 0);
	
	if(!net->inference_only)
	{
		if(net->use_wema)
			vram_approx += cuda_convert_table_FP32((void**)&(current->ema_weights), 2*nb_features, 0);
		vram_approx += cuda_convert_table(net, &(current->delta_o), current->a_size, 0);
		
		temp_allocated_size = cuda_convert_table(net, (void**)&(current->gradient), 2*nb_features, 0);
		n_param->gamma_grad = current->gradient;
		n_param->beta_grad = (void*)((unsigned char *)current->gradient + temp_allocated_size/2);	
		vram_approx += temp_allocated_size;
		
		vram_approx += cuda_convert_table(net, (void**)&(n_param->d_gamma), nb_features*net->batch_size, 0);
		vram_approx += cuda_convert_table(net, (void**)&(n_param->d_beta) , nb_features*net->batch_size, 0);
		
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->temp_A), n_param->nb_group * net->batch_size, 0);
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->temp_B) , n_param->nb_group * net->batch_size, 0);
	
		vram_approx += cuda_convert_optimizer_var(current, 2*nb_features);
	}
	
	return vram_approx;
}


void cuda_free_norm(layer *current)
{
	n_param = (norm_param*)current->param;
	
	cudaFree(current->weights);
	cudaFree(current->output);

	cudaFree(n_param->mean);
	cudaFree(n_param->var);
	
	if(!current->c_network->inference_only)
	{
		if(current->c_network->use_wema)
			cudaFree(current->ema_weights);
		
		cudaFree(current->delta_o);
		cudaFree(current->gradient);
		
		cudaFree(n_param->d_gamma);
		cudaFree(n_param->d_beta);
		
		cudaFree(n_param->temp_A);
		cudaFree(n_param->temp_B);
		
		cuda_free_optimizer_var(current);
	}
}


void cuda_forward_norm_layer(layer *current)
{
	int i;
	size_t dim_offset = 1, nb_features;
	float *l_gamma, *l_beta;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;
	//Previous verification should ensure that it is not the first layer 
	current->input = current->previous->output;
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		if(net->is_inference == 1 && (net->use_wema && !net->inference_only))
		{
			l_gamma = (float*)current->ema_weights;
			l_beta = ((float*)current->ema_weights) + nb_features;
		}
		else
		{
			l_gamma = n_param->gamma;
			l_beta = n_param->beta;
		}
		
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
			current->output, current->input, l_gamma, l_beta, n_param->mean, n_param->var, net->length, 
			net->batch_size, n_param->group_size, n_param->nb_group, nb_features, dim_offset);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, current->a_size);
}


void cuda_backward_norm_layer(layer *current)
{
	int i;
	size_t dim_offset = 1, nb_features;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;	
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		cu_blocks = nb_features * net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_norm_dbeta_conv_kernel<<<cu_blocks, 256>>>(
			current->delta_o, n_param->d_beta, nb_features, dim_offset, net->batch_size, dim_offset);
		
		net->cu_inst.cu_norm_fcts.cu_reduce_group_dgamma_conv_kernel<<<cu_blocks, 256>>>(
			current->input, current->delta_o, n_param->d_gamma, n_param->var, n_param->mean, 
			n_param->group_size, n_param->nb_group, dim_offset, net->batch_size, dim_offset);
		
		cu_blocks = n_param->nb_group * net->batch_size;
		
		net->cu_inst.cu_norm_fcts.cu_reduce_norm_AB_kernel<<<cu_blocks, 256>>>(n_param->d_gamma, n_param->d_beta,
			n_param->gamma, n_param->temp_A, n_param->temp_B, n_param->group_size, n_param->nb_group);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((dim_offset*n_param->group_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(n_param->nb_group*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
			
		net->cu_inst.cu_norm_fcts.cu_group_normalization_conv_back_kernel<<<numBlocks, threadsPerBlock>>>(
			current->input, current->delta_o, current->previous->delta_o, n_param->gamma, 
			n_param->temp_A, n_param->temp_B, n_param->mean, n_param->var, net->length, net->batch_size, 
			n_param->group_size, n_param->nb_group, current->output_dim[3], dim_offset);
	}
	
	if(!current->frozen)
	{
		cu_blocks = nb_features;
		net->cu_inst.cu_norm_fcts.cu_reduce_norm_param_grads_kernel<<<cu_blocks, 256>>>(
			n_param->d_gamma, n_param->d_beta, n_param->gamma_grad, n_param->beta_grad, nb_features, net->batch_size);
		
		net->optim_update_fct_gpu(current, 0, 2*nb_features, 2*nb_features);
		//No decay for gamma and beta
	
		if(current->wema_replace_signal > 0)
		{
			cudaMemcpy(current->FP32_weights, current->ema_weights, nb_features*2*sizeof(float), cudaMemcpyDeviceToDevice);
			current->wema_replace_signal = 0;
		}
	}
}


void cuda_norm_define(layer *current)
{
	current->forward = cuda_forward_norm_layer;
	current->backprop = cuda_backward_norm_layer;
}







