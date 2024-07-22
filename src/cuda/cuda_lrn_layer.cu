

/*
	Copyright (C) 2024 David Cornu
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
static lrn_param *n_param;

//public are in prototypes.h

//#####################################################
// Local Response Normalization layer related templates
//#####################################################

#define lrn_conv_kernel(name, type) 																											\
__global__ void lrn_conv_kernel_##name(void *i_output, void *i_input,																			\
	float *local_scale, int range, float k, float alpha, float beta,																			\
	int b_size, int nb_channel, int flat_a_size)																								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	int channel_offset = flat_a_size*b_size;																									\
	int channel_id, min_ch, max_ch, j;																											\
	float l_val, local_sum = 0.0f, l_local_scale;																								\
																																				\
	if(i >= flat_a_size*nb_channel*b_size)																										\
		return;																																	\
																																				\
	channel_id = i/(channel_offset);																											\
																																				\
	min_ch = max(0, channel_id-range/2);																										\
	max_ch = min(nb_channel-1, channel_id+range/2);																								\
																																				\
	for(j = min_ch; j <= max_ch; j++)																											\
	{																																			\
		l_val = (float) input[i+(j-channel_id)*channel_offset];																					\
		local_sum += l_val*l_val;																												\
	}																																			\
																																				\
	l_local_scale = k + alpha*local_sum/range;																									\
																																				\
	output[i] = (type)((float)input[i]/powf(l_local_scale, beta));																				\
																																				\
	if(local_scale != NULL)																														\
		local_scale[i] = l_local_scale;																											\
}


#define lrn_conv_back_kernel(name, type) 																										\
__global__ void lrn_conv_back_kernel_##name(																									\
	void *i_output, void *i_input, void *i_delta_output, void *i_delta_input,																	\
	float *local_scale, int range, float k, float alpha, float beta,																			\
	int b_size, int nb_channel, int flat_a_size)																								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type* input = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
	type* delta_input = (type*) i_delta_input;																									\
	type* delta_output = (type*) i_delta_output;																								\
	int channel_offset = flat_a_size*b_size;																									\
	int channel_id, min_ch, max_ch, l_id, j;																									\
	float local_sum = 0.0f;																														\
																																				\
	if(i >= flat_a_size*nb_channel*b_size)																										\
		return;																																	\
																																				\
	channel_id = i/(channel_offset);																											\
																																				\
	min_ch = max(0, channel_id-range/2);																										\
	max_ch = min(nb_channel-1, channel_id+range/2);																								\
																																				\
	for(j = min_ch; j <= max_ch; j++)																											\
	{																																			\
		l_id = i+(j-channel_id)*channel_offset;																									\
		local_sum += (float)delta_output[l_id]*(float)output[l_id]/(float)local_scale[l_id];													\
	}																																			\
																																				\
	delta_input[i] = (type)((float)delta_output[i]/powf(local_scale[i],beta)																	\
							- 2.0f*alpha*beta*(float)input[i]*local_sum/range);																	\
}


lrn_conv_kernel(FP32, float);
lrn_conv_back_kernel(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
lrn_conv_kernel(FP16, half);
lrn_conv_back_kernel(FP16, half);
#endif

#if defined (GEN_AMPERE)
lrn_conv_kernel(BF16, nv_bfloat16);
lrn_conv_back_kernel(BF16, nv_bfloat16);
#endif


void cuda_lrn_init(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_kernel = lrn_conv_kernel_FP32; 
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_back_kernel = lrn_conv_back_kernel_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_kernel = lrn_conv_kernel_FP16; 
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_back_kernel = lrn_conv_back_kernel_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_kernel = lrn_conv_kernel_BF16; 
			net->cu_inst.cu_lrn_fcts.cu_lrn_conv_back_kernel = lrn_conv_back_kernel_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_lrn_layer(layer *current)
{
	n_param = (lrn_param*)current->param;
	size_t vram_approx = 0;

	network* net = current->c_network;
	
	vram_approx += cuda_convert_table(net, &(current->output), n_param->output_dim, 0);
	
	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table(net, &(current->delta_o), n_param->output_dim, 0);
		vram_approx += cuda_convert_table_FP32((void**)&(n_param->local_scale), n_param->output_dim, 0);
	}
	
	return vram_approx;
}


void cuda_forward_lrn_layer(layer *current)
{
	n_param = (lrn_param*)current->param;		
	network* net = current->c_network;
	
	current->input = current->previous->output;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{
		cu_blocks = (n_param->output_dim + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_lrn_fcts.cu_lrn_conv_kernel<<< cu_blocks, cu_threads >>>(current->output, current->input, 
			n_param->local_scale, n_param->range, n_param->k, n_param->alpha, n_param->beta, 
			net->batch_size, n_param->n_dim, n_param->dim_offset);
	}

	current->activation(current);
}

void cuda_backward_lrn_layer(layer *current)
{
	n_param = (lrn_param*)current->param;	
	network* net = current->c_network;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{
		cu_blocks = (n_param->output_dim + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_lrn_fcts.cu_lrn_conv_back_kernel<<< cu_blocks, cu_threads >>>(
			current->output, current->input, current->delta_o, current->previous->delta_o,
			n_param->local_scale, n_param->range, n_param->k, n_param->alpha, n_param->beta, 
			net->batch_size, n_param->n_dim, n_param->dim_offset);
	}
	
	current->previous->deriv_activation(current->previous);
}

void cuda_lrn_define(layer *current)
{
	current->forward = cuda_forward_lrn_layer;
	current->backprop = cuda_backward_lrn_layer;
}


