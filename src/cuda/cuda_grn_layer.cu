
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
static grn_param *n_param;

// Public are in "prototypes.h"

// Private prototypes
void cuda_forward_grn_layer(layer *current);
void cuda_backward_grn_layer(layer *current);

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


// Used in GRN layer but set here for consistency
#define reduce_l2norm_conv_kernel(name, type) 																									\
__global__ void reduce_l2norm_conv_kernel_##name(void *idata, float *group_l2norm, 																\
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size) 																\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t feature_id, batch_id, conv_id;																										\
	float l_val;																																\
	double sum = 0.0f;																															\
																																				\
	feature_id = block_id % nb_features;																										\
	batch_id   = block_id / nb_features;																										\
	conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		l_val = (float)input[conv_id + i];																										\
		sum += l_val*l_val;																														\
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
		group_l2norm[block_id] = sqrt(sdata[0] + 0.000001f);																					\
}


#define reduce_grn_dgamma_conv_kernel(name, type) 																								\
__global__ void reduce_grn_dgamma_conv_kernel_##name(void *idata, void *i_d_output, void *i_d_gamma,											\
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size) 																\
{																																				\
	__shared__ float sdata[256];																												\
	type* input = (type*) idata;																												\
	type* d_output = (type*) i_d_output;																										\
	type* d_gamma = (type*) i_d_gamma;																											\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t feature_id, batch_id, conv_id;																										\
	double sum = 0.0f;																															\
																																				\
	feature_id = block_id % nb_features;																										\
	batch_id   = block_id/nb_features;																											\
	conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;																\
																																				\
	while (i < sum_size)																														\
	{																																			\
		sum += (float)d_output[conv_id + i] * (float)input[conv_id + i];																		\
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
		d_gamma[block_id] = (type) sdata[0];																									\
}


#define reduce_grn_dbeta_conv_kernel(name, type) 																								\
__global__ void reduce_grn_dbeta_conv_kernel_##name(void *i_d_output, void *i_d_beta,															\
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size) 																\
{																																				\
	__shared__ float sdata[256];																												\
	type* d_output = (type*) i_d_output;																										\
	type* d_beta = (type*) i_d_beta;																											\
	size_t tid = threadIdx.x;																													\
	size_t block_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t feature_id, batch_id, conv_id;																										\
	double sum = 0.0f;																															\
																																				\
	feature_id = block_id % nb_features;																										\
	batch_id   = block_id / nb_features;																										\
	conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;																\
																																				\
	while (i < sum_size)																														\
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
		d_beta[block_id] = (type)sdata[0];																										\
}


__global__ void reduce_grn_mean_kernel(float *feature_norm, float *mean, size_t nb_features, size_t batch_size)
{
	__shared__ float sdata[256];
	size_t tid = threadIdx.x;
	size_t block_id = blockIdx.x;
	size_t blockSize = blockDim.x;
	size_t i = tid;
	double sum = 0.0;
	
	if(block_id < batch_size)
	{
		while(i < nb_features)
		{
			sum += feature_norm[block_id*nb_features + i];
			i += blockSize;
		}
	}
	sdata[tid] = sum;
	
	__syncthreads();
	if (blockSize >= 256)
	{
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if (tid < 32)
		warpReduce(sdata, blockSize, tid);
	
	if (tid == 0)
		mean[block_id] = sdata[0] / nb_features;
}


__global__ void compute_relative_importance_kernel(float *feature_norm, float *mean, 
	float *relative_importance, size_t nb_features, size_t batch_size)
{
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	size_t batch_id;
	float eps = 0.000001f;
	
	if(i < nb_features*batch_size)
	{
		batch_id   = i / nb_features;
		relative_importance[i] = feature_norm[i] / (mean[batch_id] + eps);
	}
}

#define reduce_grn_param_grads_kernel(name, type)																								\
__global__ void reduce_grn_param_grads_kernel_##name(void *i_d_gamma, void *i_d_beta, float *relative_importance,								\
	void *i_gamma_grad, void *i_beta_grad, size_t nb_features, size_t batch_size)																\
{																																				\
	__shared__ float sdata_gamma[256];																											\
	__shared__ float sdata_beta[256];																											\
																																				\
	type* d_gamma = (type*) i_d_gamma;																											\
	type* d_beta = (type*) i_d_beta;																											\
	type* gamma_grad = (type*) i_gamma_grad;																									\
	type* beta_grad = (type*) i_beta_grad;																										\
	size_t tid = threadIdx.x;																													\
	size_t feature_id = blockIdx.x;																												\
	size_t blockSize = blockDim.x;																												\
	size_t i = tid;																																\
	size_t l_id;																																\
																																				\
	double sum_gamma = 0.0;																														\
	double sum_beta  = 0.0;																														\
																																				\
	while(i < batch_size)																														\
	{																																			\
		l_id = i * nb_features + feature_id;																									\
		sum_gamma += (float)d_gamma[l_id] * relative_importance[l_id];																			\
		sum_beta += (float)d_beta[l_id];																										\
		i += blockSize;																															\
	}																																			\
																																				\
	sdata_gamma[tid] = sum_gamma;																												\
	sdata_beta[tid]  = sum_beta;																												\
																																				\
	__syncthreads();																															\
	if (blockSize >= 256)																														\
	{																																			\
		if(tid < 128)																															\
		{																																		\
			sdata_gamma[tid] += sdata_gamma[tid + 128];																							\
			sdata_beta[tid]  += sdata_beta[tid + 128];																							\
		}																																		\
		__syncthreads();																														\
	}																																			\
	if (blockSize >= 128)																														\
	{																																			\
		if(tid < 64)																															\
		{																																		\
			sdata_gamma[tid] += sdata_gamma[tid + 64];																							\
			sdata_beta[tid]  += sdata_beta[tid + 64];																							\
		}																																		\
		__syncthreads();																														\
	}																																			\
																																				\
	if (tid < 32)																																\
	{																																			\
		warpReduce(sdata_gamma, blockSize, tid);																								\
		warpReduce(sdata_beta,  blockSize, tid);																								\
	}																																			\
																																				\
	if (tid == 0)																																\
	{																																			\
		gamma_grad[feature_id] = (type) sdata_gamma[0];																							\
		beta_grad[feature_id]  = (type) sdata_beta[0];																							\
	}																																			\
}



#define grn_conv_kernel(name, type) 																											\
__global__ void grn_conv_kernel_##name(void *i_output, void *i_input, float *gamma, float *beta,												\
	float *relative_importance, int residual, size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	float l_val, l_rel_imp;																														\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t feature_id, batch_id, conv_id;																										\
																																				\
	if(i < flat_a_size && j < nb_features*b_size)																								\
	{																																			\
		feature_id = j % nb_features;																											\
		batch_id = j / nb_features;																												\
		conv_id = batch_id*flat_a_size + feature_id*filter_offset + i;																			\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			l_rel_imp = relative_importance[batch_id*nb_features + feature_id];																	\
			l_val = (float)input[conv_id];																										\
																																				\
			output[conv_id] = (type)(gamma[feature_id] * l_val * l_rel_imp + beta[feature_id]);													\
			if(residual)																														\
				output[conv_id] += (type) l_val;																								\
		}																																		\
		else																																	\
			output[conv_id] = (type) 0.0f;																										\
	}																																			\
}


#define grn_conv_back_kernel(name, type) 																										\
__global__ void grn_conv_back_kernel_##name(void *i_input, void *i_d_output, void *i_d_input, 													\
	float *gamma, void *i_d_gamma, float *feature_norm, float *relative_importance, int residual, 												\
	size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size)																		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j = blockIdx.y*blockDim.y + threadIdx.y;																								\
	type* input = (type*) i_input;																												\
	type* d_input = (type*) i_d_input;																											\
	type* d_output = (type*) i_d_output;																										\
	type* d_gamma = (type*) i_d_gamma;																											\
	float l_d_gamma, l_rel_imp, l_feat_norm, l_mean_eps, l_val, l_grad;																			\
	size_t filter_offset = flat_a_size*b_size;																									\
	size_t feature_id, batch_id, conv_id;																										\
																																				\
	if(i < flat_a_size && j < nb_features*b_size)																								\
	{																																			\
		feature_id = j % nb_features;																											\
		batch_id = j / nb_features;																												\
		conv_id = batch_id*flat_a_size + feature_id*filter_offset + i;																			\
																																				\
		if(batch_id < b_length)																													\
		{																																		\
			l_d_gamma = d_gamma[batch_id*nb_features + feature_id];																				\
			l_feat_norm = feature_norm[batch_id*nb_features + feature_id];																		\
			l_rel_imp = relative_importance[batch_id*nb_features + feature_id];																	\
			l_mean_eps = l_feat_norm / l_rel_imp;																								\
			l_val = (float)input[conv_id];																										\
			l_grad = (float)d_output[conv_id];																									\
																																				\
			d_input[conv_id] += (type)(gamma[feature_id] * l_rel_imp * l_grad																	\
				+ gamma[feature_id] * l_d_gamma * l_val																							\
				* (1.0f/(l_mean_eps*l_feat_norm) - 1.0f/(nb_features*l_mean_eps*l_mean_eps)));													\
				/*refactored to remove un-necessary divisions*/																					\
			if(residual)																														\
				d_input[conv_id] += (type) l_grad;																								\
		}																																		\
	}																																			\
}


reduce_l2norm_conv_kernel(FP32, float);
reduce_grn_dgamma_conv_kernel(FP32, float);
reduce_grn_dbeta_conv_kernel(FP32, float);
reduce_grn_param_grads_kernel(FP32, float);
grn_conv_kernel(FP32, float);
grn_conv_back_kernel(FP32, float);


#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
reduce_l2norm_conv_kernel(FP16, half);
reduce_grn_dgamma_conv_kernel(FP16, half);
reduce_grn_dbeta_conv_kernel(FP16, half);
reduce_grn_param_grads_kernel(FP16, half);
grn_conv_kernel(FP16, half);
grn_conv_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
reduce_l2norm_conv_kernel(BF16, nv_bfloat16);
reduce_grn_dgamma_conv_kernel(BF16, nv_bfloat16);
reduce_grn_dbeta_conv_kernel(BF16, nv_bfloat16);
reduce_grn_param_grads_kernel(BF16, nv_bfloat16);
grn_conv_kernel(BF16, nv_bfloat16);
grn_conv_back_kernel(BF16, nv_bfloat16);
#endif


void cuda_grn_init(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_grn_fcts.cu_reduce_l2norm_conv_kernel = reduce_l2norm_conv_kernel_FP32;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dgamma_conv_kernel = reduce_grn_dgamma_conv_kernel_FP32;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dbeta_conv_kernel = reduce_grn_dbeta_conv_kernel_FP32;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_param_grads_kernel = reduce_grn_param_grads_kernel_FP32;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_kernel = grn_conv_kernel_FP32;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_back_kernel = grn_conv_back_kernel_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_grn_fcts.cu_reduce_l2norm_conv_kernel = reduce_l2norm_conv_kernel_FP16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dgamma_conv_kernel = reduce_grn_dgamma_conv_kernel_FP16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dbeta_conv_kernel = reduce_grn_dbeta_conv_kernel_FP16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_param_grads_kernel = reduce_grn_param_grads_kernel_FP16;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_kernel = grn_conv_kernel_FP16;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_back_kernel = grn_conv_back_kernel_FP16;
			#else
			printf("\n ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_grn_fcts.cu_reduce_l2norm_conv_kernel = reduce_l2norm_conv_kernel_BF16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dgamma_conv_kernel = reduce_grn_dgamma_conv_kernel_BF16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_dbeta_conv_kernel = reduce_grn_dbeta_conv_kernel_BF16;
			net->cu_inst.cu_grn_fcts.cu_reduce_grn_param_grads_kernel = reduce_grn_param_grads_kernel_BF16;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_kernel = grn_conv_kernel_BF16;
			net->cu_inst.cu_grn_fcts.cu_grn_conv_back_kernel = grn_conv_back_kernel_BF16;
			#else
			printf("\n ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_grn_layer(layer *current)
{
	size_t vram_approx = 0, temp_allocated_size = 0;
	size_t nb_features;
	n_param = (grn_param*)current->param;
	network* net = current->c_network;
	
	nb_features = current->output_dim[3];

	vram_approx += cuda_convert_table(net, &(current->output), current->a_size, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->feature_norm), nb_features*net->batch_size, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->relative_importance), nb_features*net->batch_size, 0);
	vram_approx += cuda_convert_table_FP32((void**)&(n_param->mean), net->batch_size, 0);
	
	vram_approx += cuda_convert_table_FP32((void**)&(current->weights), 2*nb_features, 0);
	current->FP32_weights = (float*)current->weights;
	n_param->gamma = (float*) current->weights;
	n_param->beta = ((float*) current->weights) + nb_features;
	
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
	
		vram_approx += cuda_convert_optimizer_var(current, 2*nb_features);
	}
	
	return vram_approx;
}


void cuda_free_grn(layer *current)
{
	n_param = (grn_param*)current->param;
	
	cudaFree(current->weights);
	cudaFree(current->output);
	
	cudaFree(n_param->feature_norm);
	cudaFree(n_param->relative_importance);
	cudaFree(n_param->mean);
	
	if(!current->c_network->inference_only)
	{
		if(current->c_network->use_wema)
			cudaFree(current->ema_weights);
		
		cudaFree(current->delta_o);
		cudaFree(current->gradient);
		
		cudaFree(n_param->d_gamma);
		cudaFree(n_param->d_beta);
		
		cuda_free_optimizer_var(current);
	}
}


void cuda_forward_grn_layer(layer *current)
{
	size_t i;
	size_t dim_offset = 1, nb_features;
	float *l_gamma, *l_beta;
	
	network* net = current->c_network;
	n_param = (grn_param*)current->param;
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
		
		cu_blocks = nb_features*net->batch_size;
		
		net->cu_inst.cu_grn_fcts.cu_reduce_l2norm_conv_kernel<<<cu_blocks, 256>>>(current->input, 
			n_param->feature_norm, nb_features, dim_offset, net->batch_size, dim_offset);
		
		cu_blocks = net->batch_size;
		
		reduce_grn_mean_kernel<<<cu_blocks, 256>>>(n_param->feature_norm, 
			n_param->mean, nb_features, net->batch_size);
			
		cu_blocks = (nb_features*net->batch_size + cu_threads - 1) / cu_threads ;

		compute_relative_importance_kernel<<<cu_blocks, cu_threads>>>(n_param->feature_norm,
			n_param->mean, n_param->relative_importance, nb_features, net->batch_size);
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((dim_offset + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(nb_features*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_grn_fcts.cu_grn_conv_kernel<<<numBlocks,threadsPerBlock>>>(current->output, current->input,
			l_gamma, l_beta, n_param->relative_importance, n_param->residual, 
			net->length, net->batch_size, nb_features, dim_offset);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, current->a_size);
}


void cuda_backward_grn_layer(layer *current)
{
	size_t i;
	size_t dim_offset = 1, nb_features = 1;
	
	network* net = current->c_network;
	n_param = (grn_param*)current->param;	
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		cu_blocks = nb_features*net->batch_size;
		
		net->cu_inst.cu_grn_fcts.cu_reduce_grn_dbeta_conv_kernel<<<cu_blocks, 256>>>(current->delta_o, n_param->d_beta,
			nb_features, dim_offset, net->batch_size, dim_offset);
			
		net->cu_inst.cu_grn_fcts.cu_reduce_grn_dgamma_conv_kernel<<<cu_blocks, 256>>>(current->input, current->delta_o, 
			n_param->d_gamma, nb_features, dim_offset, net->batch_size, dim_offset);
		//Here d_gamma is missing a multiplication with relative_importance[block_id]
		//Still, the current reduction is usefull for propagating the gradient. 
		//-> Multiplication with relative_importance[block_id] is postponed
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((dim_offset + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(nb_features*net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
			
		net->cu_inst.cu_grn_fcts.cu_grn_conv_back_kernel<<<numBlocks, threadsPerBlock>>>(current->input, current->delta_o, 
			current->previous->delta_o, n_param->gamma, n_param->d_gamma, 
			n_param->feature_norm, n_param->relative_importance, n_param->residual,
			net->length, net->batch_size, nb_features, dim_offset);
	}
	
	if(!current->frozen)
	{	
		cu_blocks = nb_features;
		net->cu_inst.cu_grn_fcts.cu_reduce_grn_param_grads_kernel<<<cu_blocks, 256>>>(
			n_param->d_gamma, n_param->d_beta,
			n_param->relative_importance, n_param->gamma_grad,
			n_param->beta_grad, nb_features, net->batch_size);
		
		net->optim_update_fct_gpu(current, 0, 2*nb_features, 2*nb_features);
		//No decay for gamma and beta
		
		if(current->wema_replace_signal > 0)
		{
			cudaMemcpy(current->FP32_weights, current->ema_weights, nb_features*2*sizeof(float), cudaMemcpyDeviceToDevice);
			current->wema_replace_signal = 0;
		}
	}
}


void cuda_grn_define(layer *current)
{
	current->forward = cuda_forward_grn_layer;
	current->backprop = cuda_backward_grn_layer;
}


