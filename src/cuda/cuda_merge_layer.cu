
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
__global__ void cuda_merge_add_kernel_##name(void *output_a, void *output_b, void *output_new, size_t size)										\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type *out_a = (type*) output_a;																												\
	type *out_b = (type*) output_b;																												\
	type *out_new = (type*) output_new;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	out_new[i] = out_a[i] + out_b[i];																											\
}


#define cuda_merge_add_back_kernel(name, type)																									\
__global__  void cuda_merge_add_back_kernel_##name(void *delta_o_a, void *delta_o_b, void *delta_o, size_t size)								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type *dlt_a = (type*) delta_o_a;																											\
	type *dlt_b = (type*) delta_o_b;																											\
	type *dlt = (type*) delta_o;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	type dlt_i = dlt[i];																														\
	dlt_a[i] += dlt_i;																															\
	dlt_b[i] += dlt_i;																															\
}


#define cuda_merge_concatenate_kernel(name, type)																								\
__global__ void cuda_merge_concatenate_kernel_##name(void *output_a, void *output_b, 															\
	size_t size_a, size_t size_b, void *output_new, size_t size)																				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type *out_a = (type*) output_a;																												\
	type *out_b = (type*) output_b;																												\
	type *out_new = (type*) output_new;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(i < size_a)																																\
		out_new[i] = out_a[i];																													\
	else																																		\
		out_new[i] = out_b[i-size_a];																											\
}


#define cuda_merge_concatenate_back_kernel(name, type)																							\
__global__  void cuda_merge_concatenate_back_kernel_##name(void *delta_o_a, void *delta_o_b, 													\
	size_t size_a, size_t size_b, void *delta_o, size_t size)																					\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type *dlt_a = (type*) delta_o_a;																											\
	type *dlt_b = (type*) delta_o_b;																											\
	type *dlt = (type*) delta_o;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(i < size_a)																																\
		dlt_a[i] += dlt[i];																														\
	else																																		\
		dlt_b[i-size_a] += dlt[i];																												\
}


cuda_merge_add_kernel(FP32, float);
cuda_merge_add_back_kernel(FP32, float);
cuda_merge_concatenate_kernel(FP32, float);
cuda_merge_concatenate_back_kernel(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
cuda_merge_add_kernel(FP16, half);
cuda_merge_add_back_kernel(FP16, half);
cuda_merge_concatenate_kernel(FP16, half);
cuda_merge_concatenate_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
cuda_merge_add_kernel(BF16, nv_bfloat16);
cuda_merge_add_back_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_kernel(BF16, nv_bfloat16);
cuda_merge_concatenate_back_kernel(BF16, nv_bfloat16);
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
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_merge_fcts.merge_add = cuda_merge_add_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_add_back = cuda_merge_add_back_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate = cuda_merge_concatenate_kernel_FP16;
			net->cu_inst.cu_merge_fcts.merge_concatenate_back = cuda_merge_concatenate_back_kernel_FP16;
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
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_merge_layer(layer *current)
{
	m_param = (merge_param*)current->param;
	size_t vram_approx = 0;

	network *net = current->c_network;
	
	vram_approx += cuda_convert_table(net, &(current->output), current->a_size, 0);
	
	if(!net->inference_only)
		vram_approx += cuda_convert_table(net, &(current->delta_o), current->a_size, 0);
	
	return vram_approx;
}

void cuda_free_merge(layer *current)
{
	cudaFree(current->output);
	
	if(!current->c_network->inference_only)
		cudaFree(current->delta_o);
}


void cuda_forward_merge_layer(layer *current)
{
	layer *prev_a, *prev_b;
	network *net = current->c_network;
		
	m_param = (merge_param*) current->param;
	
	prev_a = m_param->previous_a;
	prev_b = m_param->previous_b;
	
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	if(m_param->merge_type == ADD_merge)
		net->cu_inst.cu_merge_fcts.merge_add<<<cu_blocks, cu_threads>>>(
			prev_a->output, prev_b->output, current->output, current->a_size);
	else
		net->cu_inst.cu_merge_fcts.merge_concatenate<<<cu_blocks, cu_threads>>>(
			prev_a->output, prev_b->output, prev_a->a_size, prev_b->a_size,
			current->output, current->a_size);
	
	current->activation(current);
	
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, current->a_size);
}


void cuda_backward_merge_layer(layer *current)
{
	layer *prev_a, *prev_b;
	network *net = current->c_network;
	
	m_param = (merge_param*) current->param;
	
	prev_a = m_param->previous_a;
	prev_b = m_param->previous_b;
	
	current->deriv_activation(current);
	
	// The previous == NULL case is supposed excluded from layer creation condition
	
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	if(m_param->merge_type == ADD_merge)
		net->cu_inst.cu_merge_fcts.merge_add_back<<<cu_blocks, cu_threads>>>(
			prev_a->delta_o, prev_b->delta_o, current->delta_o, current->a_size);
	else
		net->cu_inst.cu_merge_fcts.merge_concatenate_back<<<cu_blocks, cu_threads>>>(
			prev_a->delta_o, prev_b->delta_o, prev_a->a_size, prev_b->a_size, current->delta_o, current->a_size);
}


void cuda_merge_define(layer *current)
{
	current->forward = cuda_forward_merge_layer;
	current->backprop = cuda_backward_merge_layer;
}




