
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
static merge_param *m_param;

// Public are in "prototypes.h"

// Private prototypes


#define cuda_merge_add_kernel(name, type)																										\
__global__ void cuda_merge_add_kernel_##name(int nb_previous, int *prev_type_list, void **prev_output_list,										\
	size_t chan_offset, int batch_size, void *output_new, size_t size)																			\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j;																																	\
	size_t l_channel, l_batch, l_pixel;																											\
	size_t chan_batch_offset = chan_offset*batch_size;																							\
	size_t nb_channels = size/chan_batch_offset;																								\
	float acc;																																	\
	type *out_new = (type*) output_new;																											\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	l_channel =  i / chan_batch_offset;																											\
	l_batch   = (i % chan_batch_offset) / chan_offset;																							\
	l_pixel   =  i % chan_offset;																												\
																																				\
	acc = 0.0f;																																	\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		l_prev = (type*) prev_output_list[j];																									\
		if(prev_type_list[j] != INPUT)																											\
			acc += (float) l_prev[i];																											\
		else																																	\
			acc += (float) l_prev[l_batch*(nb_channels*chan_offset + 1) + l_channel*chan_offset + l_pixel];										\
	}																																			\
	out_new[i] = acc;																															\
}


#define cuda_merge_add_back_kernel(name, type)																									\
__global__ void cuda_merge_add_back_kernel_##name(int nb_previous, int *prev_type_list, void **prev_delta_o_list,								\
	void *delta_o, size_t size)																													\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j;																																	\
	type *dlt = (type*) delta_o;																												\
	type dlt_i;																																	\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	dlt_i = dlt[i];																																\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(prev_type_list[j] == INPUT)																											\
			continue;																															\
		l_prev = (type*) prev_delta_o_list[j];																									\
		l_prev[i] += dlt_i;																														\
	}																																			\
}


#define cuda_merge_concatenate_kernel(name, type)																								\
__global__ void cuda_merge_concatenate_kernel_##name(int nb_previous, int *prev_type_list, void **prev_output_list,								\
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *output_new, size_t size)													\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, l_channel, channel_acc, l_batch, l_pixel;																							\
	size_t chan_batch_offset = chan_offset*batch_size;																							\
	type *out_new = (type*) output_new;																											\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	l_channel =  i / chan_batch_offset;																											\
	l_batch   = (i % chan_batch_offset) / chan_offset;																							\
	l_pixel   =  i % chan_offset;																												\
																																				\
	channel_acc = 0;																															\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(l_channel < channel_acc + prev_nb_channels[j])																						\
			break;																																\
		channel_acc += prev_nb_channels[j];																										\
	}																																			\
																																				\
	l_prev = (type*) prev_output_list[j];																										\
	l_channel = l_channel - channel_acc;																										\
	if(prev_type_list[j] != INPUT)																												\
		out_new[i] = l_prev[l_channel*chan_batch_offset + l_batch*chan_offset + l_pixel];														\
	else																																		\
		out_new[i] = l_prev[l_batch*(prev_nb_channels[j]*chan_offset + 1) + l_channel*chan_offset + l_pixel];									\
}


#define cuda_merge_concatenate_back_kernel(name, type)																							\
__global__ void cuda_merge_concatenate_back_kernel_##name(int nb_previous, int *prev_type_list, void **prev_delta_o_list,						\
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *delta_o, size_t size)														\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, l_channel, channel_acc, l_batch, l_pixel;																							\
	size_t chan_batch_offset = chan_offset*batch_size;																							\
	type *dlt = (type*) delta_o;																												\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	l_channel =  i / chan_batch_offset;																											\
	l_batch   = (i % chan_batch_offset) / chan_offset;																							\
	l_pixel   =  i % chan_offset;																												\
																																				\
	channel_acc = 0;																															\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(l_channel < channel_acc + prev_nb_channels[j])																						\
			break;																																\
		channel_acc += prev_nb_channels[j];																										\
	}																																			\
																																				\
	l_channel = l_channel - channel_acc;																										\
	if(prev_type_list[j] == INPUT)																												\
		return;																																	\
																																				\
	l_prev = (type*) prev_delta_o_list[j];																										\
	l_prev[l_channel*chan_batch_offset + l_batch*chan_offset + l_pixel] += dlt[i];																\
}


#define cuda_merge_add_flat_kernel(name, type)																									\
__global__ void cuda_merge_add_flat_kernel_##name(int nb_previous, void **prev_output_list,														\
	size_t output_dim, void *output_new, size_t size)																							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, feature_id;																														\
	float acc;																																	\
	type *out_new = (type*) output_new;																											\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	feature_id = i % output_dim;																												\
	if(feature_id == output_dim - 1)																											\
		return;																																	\
																																				\
	acc = 0.0f;																																	\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		l_prev = (type*) prev_output_list[j];																									\
		acc += (float) l_prev[i];																												\
	}																																			\
	out_new[i] = acc;																															\
}


#define cuda_merge_add_flat_back_kernel(name, type)																								\
__global__ void cuda_merge_add_flat_back_kernel_##name(int nb_previous, int *prev_type_list, void **prev_delta_o_list,							\
	size_t output_dim, void *delta_o, size_t size)																								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, feature_id;																														\
	type *dlt = (type*) delta_o;																												\
	type dlt_i;																																	\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	feature_id = i % output_dim;																												\
	if(feature_id == output_dim - 1)																											\
		return;																																	\
																																				\
	dlt_i = dlt[i];																																\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(prev_type_list[j] == INPUT)																											\
			continue;																															\
		l_prev = (type*) prev_delta_o_list[j];																									\
		l_prev[i] += dlt_i;																														\
	}																																			\
}


#define cuda_merge_concatenate_flat_kernel(name, type)																							\
__global__ void cuda_merge_concatenate_flat_kernel_##name(int nb_previous, void **prev_output_list,												\
	int *prev_nb_neurons, size_t output_dim, void *output_new, size_t size)																		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, batch_id, feature_id, feature_acc;																								\
	type *out_new = (type*) output_new;																											\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	batch_id = i / output_dim;																													\
	feature_id = i % output_dim;																												\
	if(feature_id == output_dim - 1)																											\
		return;																																	\
																																				\
	feature_acc = 0;																															\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(feature_id < feature_acc + prev_nb_neurons[j])																						\
			break;																																\
		feature_acc += prev_nb_neurons[j];																										\
	}																																			\
																																				\
	l_prev = (type*) prev_output_list[j];																										\
	out_new[i] = l_prev[batch_id*(prev_nb_neurons[j]+1) + (feature_id - feature_acc)];															\
}


#define cuda_merge_concatenate_flat_back_kernel(name, type)																						\
__global__ void cuda_merge_concatenate_flat_back_kernel_##name(int nb_previous, int *prev_type_list,											\
	void **prev_delta_o_list, int *prev_nb_neurons, size_t output_dim, void *delta_o, size_t size)												\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, batch_id, feature_id, feature_acc;																								\
	type *dlt = (type*) delta_o;																												\
	type *l_prev;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
	batch_id = i / output_dim;																													\
	feature_id = i % output_dim;																												\
	if(feature_id == output_dim - 1)																											\
		return;																																	\
																																				\
	feature_acc = 0;																															\
	for(j = 0; j < nb_previous; j++)																											\
	{																																			\
		if(feature_id < feature_acc + prev_nb_neurons[j])																						\
			break;																																\
		feature_acc += prev_nb_neurons[j];																										\
	}																																			\
																																				\
	if(prev_type_list[j] == INPUT)																												\
		return;																																	\
	l_prev = (type*) prev_delta_o_list[j];																										\
	l_prev[batch_id*(prev_nb_neurons[j]+1) + (feature_id - feature_acc)] += dlt[i];																\
}


cuda_merge_add_kernel(FP32, float);
cuda_merge_add_back_kernel(FP32, float);
cuda_merge_concatenate_kernel(FP32, float);
cuda_merge_concatenate_back_kernel(FP32, float);
cuda_merge_add_flat_kernel(FP32, float);
cuda_merge_add_flat_back_kernel(FP32, float);
cuda_merge_concatenate_flat_kernel(FP32, float);
cuda_merge_concatenate_flat_back_kernel(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
cuda_merge_add_kernel(FP16, half);
cuda_merge_add_back_kernel(FP16, half);
cuda_merge_concatenate_kernel(FP16, half);
cuda_merge_concatenate_back_kernel(FP16, half);
cuda_merge_add_flat_kernel(FP16, half);
cuda_merge_add_flat_back_kernel(FP16, half);
cuda_merge_concatenate_flat_kernel(FP16, half);
cuda_merge_concatenate_flat_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
cuda_merge_add_kernel(BF16, nv_bfloat16);
cuda_merge_add_back_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_back_kernel(BF16, nv_bfloat16);
cuda_merge_add_flat_kernel(BF16, nv_bfloat16);
cuda_merge_add_flat_back_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_flat_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_flat_back_kernel(BF16, nv_bfloat16);
#endif


void cuda_merge_init(network *net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_merge_fcts.merge_add = cuda_merge_add_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_add_back = cuda_merge_add_back_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_concatenate = cuda_merge_concatenate_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_concatenate_back = cuda_merge_concatenate_back_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_add_flat = cuda_merge_add_flat_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_add_flat_back = cuda_merge_add_flat_back_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat = cuda_merge_concatenate_flat_kernel_FP32;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat_back = cuda_merge_concatenate_flat_back_kernel_FP32;
			break;
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_merge_fcts.merge_add = cuda_merge_add_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_add_back = cuda_merge_add_back_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate = cuda_merge_concatenate_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_back = cuda_merge_concatenate_back_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_add_flat = cuda_merge_add_flat_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_add_flat_back = cuda_merge_add_flat_back_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat = cuda_merge_concatenate_flat_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat_back = cuda_merge_concatenate_flat_back_kernel_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_merge_fcts.merge_add = cuda_merge_add_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_add_back = cuda_merge_add_back_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_concatenate = cuda_merge_concatenate_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_back = cuda_merge_concatenate_back_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_add_flat = cuda_merge_add_flat_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_add_flat_back = cuda_merge_add_flat_back_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat = cuda_merge_concatenate_flat_kernel_BF16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat_back = cuda_merge_concatenate_flat_back_kernel_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_merge_layer(layer *current)
{
	size_t vram_approx = 0;
	network *net = current->c_network;
	m_param = (merge_param*) current->param;
	void **temp_tab;
	
	vram_approx += cuda_convert_table(net, &(current->output), current->a_size, 0);
	vram_approx += cuda_convert_table_int(&(m_param->prev_type_list), m_param->nb_previous, 0);
	
	if(m_param->merge_type == CONCAT_merge)
		vram_approx += cuda_convert_table_int(&(m_param->prev_nb_channels), m_param->nb_previous, 0);
	
	cudaMalloc((void**)&(m_param->prev_output_list_gpu), m_param->nb_previous * sizeof(void*));
	vram_approx += m_param->nb_previous * sizeof(void*);
	cudaMemcpy(m_param->prev_output_list_gpu, m_param->prev_output_list, m_param->nb_previous * sizeof(void*), cudaMemcpyHostToDevice);
	
	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table(net, &(current->delta_o), current->a_size, 0);
		temp_tab = m_param->prev_delta_o_list;
		cudaMalloc((void**)&(m_param->prev_delta_o_list), m_param->nb_previous * sizeof(void*));
		vram_approx += m_param->nb_previous * sizeof(void*);
		cudaMemcpy(m_param->prev_delta_o_list, temp_tab, m_param->nb_previous * sizeof(void*), cudaMemcpyHostToDevice);
		free(temp_tab);
	}
	return vram_approx;
}


void cuda_free_merge(layer *current)
{
	m_param = (merge_param*) current->param;
	
	cudaFree(current->output);
	cudaFree(m_param->prev_type_list);
	
	if(m_param->prev_nb_channels != NULL)
		cudaFree(m_param->prev_nb_channels);
	cudaFree(m_param->prev_output_list_gpu);
	
	if(!current->c_network->inference_only)
	{
		cudaFree(current->delta_o);
		cudaFree(m_param->prev_delta_o_list);
	}
}


void cuda_forward_merge_layer(layer *current)
{
	int i;
	network *net = current->c_network;
	m_param = (merge_param*) current->param;
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	for(i = 0; i < m_param->nb_previous; i++) //required for INPUT type layers
		m_param->prev_output_list[i] = m_param->prev_list[i]->output;
	cudaMemcpy(m_param->prev_output_list_gpu, m_param->prev_output_list, m_param->nb_previous * sizeof(void*), cudaMemcpyHostToDevice);
	
	if(m_param->merge_type == ADD_merge)
	{
		if(current->output_type == SPATIAL)
			net->cu_inst.cu_merge_fcts.merge_add<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list, 
				m_param->prev_output_list_gpu, current->a_dim, 
				net->batch_size, current->output, current->a_size);
		else
			net->cu_inst.cu_merge_fcts.merge_add_flat<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_output_list_gpu, 
				current->a_biased_dim, current->output, current->a_size);
	}
	else
	{
		if(current->output_type == SPATIAL)
			net->cu_inst.cu_merge_fcts.merge_concatenate<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list, 
				m_param->prev_output_list_gpu, m_param->prev_nb_channels, 
				current->a_dim, net->batch_size, current->output, current->a_size);
		else
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_output_list_gpu, 
				m_param->prev_nb_channels, current->a_biased_dim, 
				current->output, current->a_size);
	}
	
	current->activation(current);
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, current->a_size);
}


void cuda_backward_merge_layer(layer *current)
{
	network *net = current->c_network;
	m_param = (merge_param*) current->param;
	current->deriv_activation(current);
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	if(m_param->merge_type == ADD_merge)
	{
		if(current->output_type == SPATIAL)
			net->cu_inst.cu_merge_fcts.merge_add_back<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list, 
				m_param->prev_delta_o_list, current->delta_o, current->a_size);
		else
			net->cu_inst.cu_merge_fcts.merge_add_flat_back<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list, 
				m_param->prev_delta_o_list, current->a_biased_dim, current->delta_o, current->a_size);
	}
	else
	{
		if(current->output_type == SPATIAL)
			net->cu_inst.cu_merge_fcts.merge_concatenate_back<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list, 
				m_param->prev_delta_o_list, m_param->prev_nb_channels, current->a_dim, 
				net->batch_size, current->delta_o, current->a_size);
		else
			net->cu_inst.cu_merge_fcts.merge_concatenate_flat_back<<<cu_blocks, cu_threads>>>(
				m_param->nb_previous, m_param->prev_type_list,
				m_param->prev_delta_o_list, m_param->prev_nb_channels, 
				current->a_biased_dim, current->delta_o, current->a_size);
	}
}


void cuda_merge_define(layer *current)
{
	current->forward = cuda_forward_merge_layer;
	current->backprop = cuda_backward_merge_layer;
}




