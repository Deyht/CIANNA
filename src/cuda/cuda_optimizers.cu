
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

// Public are in "prototypes.h"

// Private prototypes


//#####################################################
//		          SGD related functions
//#####################################################

size_t cuda_convert_sgd_var(layer *current, size_t param_size)
{
	sgd_param *o_param = (sgd_param*)current->c_network->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	
	if(o_param->momentum < 0.01f)
		return 0;
	
	cuda_convert_table_FP32((void**)&(o_var->velocity), param_size, 0);
	
	return param_size;
}


#define cuda_update_weights_sgd_kernel(name, type)																								\
__global__ void cuda_update_weights_sgd_kernel_##name(float *weights, float *ema_weights, void *gradient, 										\
	float weight_decay, int decoupled_wdecay, float wema_rate, float learning_rate, int batch_size, 											\
	float momentum, float *velocity, size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor)						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type *c_gradient = ((type*)gradient);																										\
	int decay_mask = 1;																															\
	float l_grad;																																\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(((i+1) % flat_dim_offset) >= bias_weight_offset)																							\
		decay_mask = 0; /*prevent decay for the bias weights*/																					\
																																				\
	if(decoupled_wdecay)																														\
	{																																			\
		weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];																			\
		l_grad = (float)c_gradient[i]/batch_size;																								\
	}																																			\
	else																																		\
		l_grad = (float)c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];															\
																																				\
	if(momentum >= 0.01)																														\
	{																																			\
		velocity[i] = learning_rate*l_grad + momentum*velocity[i];																				\
		weights[i] -= velocity[i];																												\
	}																																			\
	else																																		\
		weights[i] -= learning_rate*l_grad;																										\
																																				\
	if(wema_rate > 0.01f)																														\
		ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];																\
}


void cuda_update_weights_sgd(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	sgd_param *o_param = (sgd_param*) net->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	 
	cu_blocks = (nb_weights + cu_threads - 1) / cu_threads;
	
	net->cu_inst.cu_optimizer_fcts.sgd_fct<<< cu_blocks, cu_threads >>>
		((float*)current->FP32_weights, current->ema_weights, current->gradient, 
		net->weight_decay, net->decoupled_wdecay, net->wema_rate,
		net->learning_rate, net->length, o_param->momentum, o_var->velocity, 
		bias_weight_offset, flat_dim_offset, nb_weights, net->TC_scale_factor);
}


void cuda_free_sgd_var(layer *current)
{
	sgd_param *o_param = (sgd_param*) current->c_network->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;

	if(o_param->momentum < 0.01f)
		return;
		
	cudaFree(o_var->velocity);	
}


//#####################################################
//		          ADAM related functions
//#####################################################


size_t cuda_convert_adam_var(layer *current, size_t param_size)
{
	adam_param *o_param = (adam_param*)current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	cuda_convert_table_FP32((void**)&(o_var->first_mom), param_size, 0);
	cuda_convert_table_FP32((void**)&(o_var->second_mom), param_size, 0);
	
	if(o_param->ams_grad <= 0)
		return 0;
	
	cuda_convert_table_FP32((void**)&(o_var->max_second_mom), param_size, 0);
	
	return param_size;
}


#define cuda_update_weights_adam_kernel(name, type)																								\
__global__ void cuda_update_weights_adam_kernel_##name(float *weights, float *ema_weights, void *gradient, 										\
	float weight_decay, int decoupled_wdecay, float wema_rate, float learning_rate, int batch_size, 											\
	float beta_1, float beta_2, float eps, int ams_grad, int opt_step,																			\
	float *first_mom, float *second_mom, float *max_second_mom,																					\
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor)														\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type *c_gradient = ((type*)gradient);																										\
	int decay_mask = 1;																															\
	float l_grad, moving_first_mom, moving_second_mom;																							\
																																				\
	if(i >= size)																																\
		return; 																																\
																																				\
	decay_mask = 1;																																\
	if(((i+1) % flat_dim_offset) >= bias_weight_offset)																							\
		decay_mask = 0; /*prevent decay for the bias weights*/																					\
																																				\
	if(decoupled_wdecay)																														\
	{																																			\
		weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];																			\
		l_grad = (float)c_gradient[i]/batch_size;																								\
	}																																			\
	else																																		\
		l_grad = (float)c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];															\
																																				\
	first_mom[i]  = beta_1*first_mom[i]  + (1.0-beta_1)*l_grad;																					\
	second_mom[i] = beta_2*second_mom[i] + (1.0-beta_2)*l_grad*l_grad;																			\
																																				\
	moving_first_mom = first_mom[i] / (1.0-powf(beta_1,opt_step));																				\
																																				\
	if(ams_grad > 0)																															\
	{																																			\
		max_second_mom[i] = max(max_second_mom[i], second_mom[i]);																				\
		moving_second_mom = max_second_mom[i]/(1.0 - powf(beta_2,opt_step));																	\
	}																																			\
	else																																		\
		moving_second_mom = second_mom[i]/(1.0 - powf(beta_2,opt_step));																		\
																																				\
	weights[i] -= learning_rate*moving_first_mom/(sqrtf(moving_second_mom) + eps);																\
																																				\
	if(wema_rate > 0.01f)																														\
		ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];																\
}


void cuda_update_weights_adam(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	adam_param *o_param = (adam_param*) net->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	if(o_var->opt_step < SIZE_MAX)
		o_var->opt_step += 1;
	
	cu_blocks = (nb_weights + cu_threads - 1) / cu_threads;
	
	net->cu_inst.cu_optimizer_fcts.adam_fct<<< cu_blocks, cu_threads >>>
		((float*)current->FP32_weights, current->ema_weights, current->gradient, 
		net->weight_decay, net->decoupled_wdecay, net->wema_rate, net->learning_rate, net->length,
		o_param->beta_1, o_param->beta_2, o_param->eps, o_param->ams_grad, o_var->opt_step,
		o_var->first_mom, o_var->second_mom, o_var->max_second_mom,
		bias_weight_offset, flat_dim_offset, nb_weights, net->TC_scale_factor);
}


void cuda_free_adam_var(layer *current)
{
	adam_param *o_param = (adam_param*) current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	cudaFree(o_var->first_mom);
	cudaFree(o_var->second_mom);
	if(o_param->ams_grad <= 0)
		return;
	
	cudaFree(o_var->max_second_mom);
}


//#####################################################
//		         RMSprop related functions
//#####################################################


size_t cuda_convert_rmsprop_var(layer *current, size_t param_size)
{
	size_t alloc_size = param_size;
	
	rmsprop_param *o_param = (rmsprop_param*)current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	cuda_convert_table_FP32((void**)&(o_var->sqrt_avg), param_size, 0);

	if(o_param->momentum >= 0.01)
	{
		cuda_convert_table_FP32((void**)&(o_var->velocity), param_size, 0);
		alloc_size += param_size;
	}
	
	if(o_param->centered > 0)
	{
		cuda_convert_table_FP32((void**)&(o_var->grad_avg), param_size, 0);
		alloc_size += param_size;
	}
	
	return alloc_size;
}


#define cuda_update_weights_rmsprop_kernel(name, type)																							\
__global__ void cuda_update_weights_rmsprop_kernel_##name(float *weights, float *ema_weights, void *gradient,									\
	float weight_decay, int decoupled_wdecay, float wema_rate, float learning_rate, int batch_size, 											\
	float alpha, float momentum, float eps, int centered,																						\
	float *sqrt_avg, float *velocity, float *grad_avg,																							\
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor)														\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type *c_gradient = ((type*)gradient);																										\
	int decay_mask = 1;																															\
	float l_grad, l_sqrt_avg_centered;																											\
																																				\
	if(i >= size)																																\
		return; 																																\
																																				\
	decay_mask = 1;																																\
	if(((i+1) % flat_dim_offset) >= bias_weight_offset)																							\
		decay_mask = 0; /*prevent decay for the bias weights*/																					\
																																				\
	if(decoupled_wdecay)																														\
	{																																			\
		weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];																			\
		l_grad = (float)c_gradient[i]/batch_size;																								\
	}																																			\
	else																																		\
		l_grad = (float)c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];															\
																																				\
	sqrt_avg[i] = alpha*sqrt_avg[i] + (1.0 - alpha)*l_grad*l_grad;																				\
	l_sqrt_avg_centered = sqrt_avg[i];																											\
																																				\
	if(centered > 0)																															\
	{																																			\
		grad_avg[i] = alpha*grad_avg[i] + (1.0 - alpha)*l_grad;																					\
		l_sqrt_avg_centered -= grad_avg[i]*grad_avg[i];																							\
	}																																			\
																																				\
	if(momentum >= 0.01)																														\
	{																																			\
		velocity[i] = momentum*velocity[i] + l_grad/(sqrt(l_sqrt_avg_centered) + eps);															\
		weights[i] -= learning_rate*velocity[i];																								\
	}																																			\
	else																																		\
		weights[i] -= learning_rate*l_grad/(sqrt(l_sqrt_avg_centered) + eps);																	\
																																				\
	if(wema_rate > 0.01f)																														\
		ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];																\
}


void cuda_update_weights_rmsprop(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	rmsprop_param *o_param = (rmsprop_param*) net->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	cu_blocks = (nb_weights + cu_threads - 1) / cu_threads;
	
	net->cu_inst.cu_optimizer_fcts.rmsprop_fct<<< cu_blocks, cu_threads >>>
		((float*)current->FP32_weights, current->ema_weights, current->gradient, 
		net->weight_decay, net->decoupled_wdecay, net->wema_rate, net->learning_rate, net->length,
		o_param->alpha, o_param->momentum, o_param->eps, o_param->centered,
		o_var->sqrt_avg, o_var->velocity, o_var->grad_avg,
		bias_weight_offset, flat_dim_offset, nb_weights, net->TC_scale_factor);
}


void cuda_free_rmsprop_var(layer *current)
{
	rmsprop_param *o_param = (rmsprop_param*) current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	cudaFree(o_var->sqrt_avg);
	
	if(o_param->momentum >= 0.01)
		cudaFree(o_var->velocity);
	
	if(o_param->centered > 0)
		cudaFree(o_var->grad_avg);
}


//#####################################################
//		         Generic functions
//#####################################################


#define typed_cuda_optimizer_fct_association(name)																								\
void typed_cuda_optimizer_fct_association_##name(network *net)																					\
{																																				\
	net->cu_inst.cu_optimizer_fcts.sgd_fct = cuda_update_weights_sgd_kernel_##name;																\
	net->cu_inst.cu_optimizer_fcts.adam_fct = cuda_update_weights_adam_kernel_##name;															\
	net->cu_inst.cu_optimizer_fcts.rmsprop_fct = cuda_update_weights_rmsprop_kernel_##name;														\
}


cuda_update_weights_sgd_kernel(FP32, float);
cuda_update_weights_adam_kernel(FP32, float);
cuda_update_weights_rmsprop_kernel(FP32, float);
typed_cuda_optimizer_fct_association(FP32);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
cuda_update_weights_sgd_kernel(FP16, half);
cuda_update_weights_adam_kernel(FP16, half);
cuda_update_weights_rmsprop_kernel(FP16, half);
typed_cuda_optimizer_fct_association(FP16);
#endif

#if defined(GEN_AMPERE)
cuda_update_weights_sgd_kernel(BF16, nv_bfloat16);
cuda_update_weights_adam_kernel(BF16, nv_bfloat16);
cuda_update_weights_rmsprop_kernel(BF16, nv_bfloat16);
typed_cuda_optimizer_fct_association(BF16);
#endif


void init_typed_cuda_optimizer(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			typed_cuda_optimizer_fct_association_FP32(net);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			typed_cuda_optimizer_fct_association_FP16(net);
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
		
		case BF16C_FP32A:
			#if defined(GEN_AMPERE)
			typed_cuda_optimizer_fct_association_BF16(net);
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_optimizer_var(layer *current, size_t param_size)
{
	network *net = current->c_network;
	size_t allocate_size = 0;
	
	switch(net->optimizer)
	{
		default:
		case(SGD):
			allocate_size = cuda_convert_sgd_var(current, param_size);
			break;
		case(ADAM):
			allocate_size = cuda_convert_adam_var(current, param_size);
			break;
		case(RMS_PROP):
			allocate_size = cuda_convert_rmsprop_var(current, param_size);
			break;
	}
	return allocate_size;
}


void cuda_set_optimizer_update_function(network *net)
{
	switch(net->optimizer)
	{
		default:
		case(SGD):
			net->optim_update_fct_gpu = cuda_update_weights_sgd;
			break;
		case(ADAM):
			net->optim_update_fct_gpu = cuda_update_weights_adam;
			break;
		case(RMS_PROP):
			net->optim_update_fct_gpu = cuda_update_weights_rmsprop;
			break;
	}
}


void cuda_free_optimizer_var(layer *current)
{
	switch(current->c_network->optimizer)
	{
		default:
		case(SGD):
			cuda_free_sgd_var(current);
			break;
		case(ADAM):
			cuda_free_adam_var(current);
			break;
		case(RMS_PROP):
			cuda_free_rmsprop_var(current);
			break;
	}	
}




