
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
static dense_param *d_param;

// Public are in "prototypes.h"

// Private prototypes
void cuda_forward_dense_layer(layer *current);
void cuda_backward_dense_layer(layer* current);

// Functions that result from templates are not listed here but at the end of the file instead


//used to reshape output of Conv layer that has the result of filter 1 continuous for the batch
//convert into all filters continuous for image 1, then image 2, ...
#define cuda_flat_dense(name, type) 																											\
__global__ void cuda_flat_dense_##name																											\
	(void* i_in, void* i_out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, size_t size)								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int map_id, image_id, pos;																													\
																																				\
	type* in = (type*) i_in;																													\
	type* out = (type*) i_out;																													\
																																				\
	if(i < size)																																\
	{																																			\
		image_id = i / flatten_size;																											\
		map_id = (i % flatten_size)/map_size;																									\
		pos = (i % flatten_size)%map_size;																										\
																																				\
		if(map_id >= nb_map)																													\
			out[i] = (type) bias;																												\
		else																																	\
			out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];																\
	}																																			\
}


#define cuda_reroll_batch(name, type) 																											\
__global__ void cuda_reroll_batch_##name																										\
	(void* i_in, void* i_out, int map_size, int flatten_size, int nb_map, int batch_size, size_t size)											\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int map_id, image_id, pos;																													\
																																				\
	type* in = (type*) i_in;																													\
	type* out = (type*) i_out;																													\
																																				\
	if(i < size)																																\
	{																																			\
		map_id = i / (map_size*batch_size);																										\
		image_id = (i % (map_size*batch_size))/map_size;																						\
		pos = (i % (map_size*batch_size))%map_size;																								\
																																				\
		out[i] += in[image_id*(flatten_size) + map_id*map_size + pos];																			\
	}																																			\
}


#define cuda_dropout_apply_dense(name, type) 																									\
__global__ void cuda_dropout_apply_dense_##name(void* i_table, float* mask, size_t size, int biased_dim, float drop_rate)						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(mask[i] >= drop_rate || (i+1) % biased_dim == 0)																							\
		mask[i] = 1.0f;																															\
	else																																		\
		mask[i] = 0.0f;																															\
	 																																			\
	table[i] = (type)((float)table[i]*mask[i]); 																								\
}


#define cuda_dropout_scale_dense(name, type) 																									\
__global__ void cuda_dropout_scale_dense_##name(void* i_table, float* mask, size_t size, int biased_dim, float drop_rate)						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if((i+1) % biased_dim != 0)																													\
		table[i] = (type)((float)table[i]*(1.0f-drop_rate)); 																					\
}

#define cuda_set_input_bias_dense(name, type) 																									\
__global__ void cuda_set_input_bias_dense_##name(void* i_table, size_t unbiased_dim, float bias_value, size_t size)								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	table[i*(unbiased_dim+1) + unbiased_dim] = bias_value;																						\
}

cuda_flat_dense(FP32, float);
cuda_reroll_batch(FP32, float);
cuda_dropout_apply_dense(FP32, float);
cuda_dropout_scale_dense(FP32, float);
cuda_set_input_bias_dense(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
cuda_flat_dense(FP16, half);
cuda_reroll_batch(FP16, half);
cuda_dropout_apply_dense(FP16, half);
cuda_dropout_scale_dense(FP16, half);
cuda_set_input_bias_dense(FP16, half);
#endif

#if defined (GEN_AMPERE)
cuda_flat_dense(BF16, nv_bfloat16);
cuda_reroll_batch(BF16, nv_bfloat16);
cuda_dropout_apply_dense(BF16, nv_bfloat16);
cuda_dropout_scale_dense(BF16, nv_bfloat16);
cuda_set_input_bias_dense(BF16, nv_bfloat16);
#endif



void cuda_dense_init(network *net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_dense_fcts.flat_dense_fct = cuda_flat_dense_FP32;
			net->cu_inst.cu_dense_fcts.reroll_fct = cuda_reroll_batch_FP32;
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_dense_FP32;
			net->cu_inst.cu_dense_fcts.drop_scale_fct = cuda_dropout_scale_dense_FP32;
			net->cu_inst.cu_dense_fcts.set_input_bias = cuda_set_input_bias_dense_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_dense_fcts.flat_dense_fct = cuda_flat_dense_FP16;
			net->cu_inst.cu_dense_fcts.reroll_fct = cuda_reroll_batch_FP16;
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_dense_FP16;
			net->cu_inst.cu_dense_fcts.drop_scale_fct = cuda_dropout_scale_dense_FP16;
			net->cu_inst.cu_dense_fcts.set_input_bias = cuda_set_input_bias_dense_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_dense_fcts.flat_dense_fct = cuda_flat_dense_BF16;
			net->cu_inst.cu_dense_fcts.reroll_fct = cuda_reroll_batch_BF16;
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_dense_BF16;
			net->cu_inst.cu_dense_fcts.drop_scale_fct = cuda_dropout_scale_dense_BF16;
			net->cu_inst.cu_dense_fcts.set_input_bias = cuda_set_input_bias_dense_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}

size_t cuda_convert_dense_layer(layer *current)
{
	int i, nb_neurons;
	size_t flat_in_size = 1, vram_approx = 0;
	#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
	float* temp_tab;
	#endif
	
	network* net = current->c_network;
	d_param = (dense_param*)current->param;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	if(current->previous != NULL)
	{
		if(current->previous->output_type == SPATIAL)
		{
			vram_approx += cuda_convert_table(net, &(d_param->flat_input), flat_in_size * net->batch_size, 0);
			if(!net->inference_only)
				vram_approx += cuda_convert_table(net, &(d_param->flat_delta_o), flat_in_size * net->batch_size, 0);
		}
		else
			d_param->flat_delta_o = current->previous->delta_o;
	}
	
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			vram_approx += cuda_convert_table(net, &(current->weights), flat_in_size*(nb_neurons+1), 0);
			current->FP32_weights = (float*) current->weights;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			temp_tab = (float*)current->weights;
			cudaMalloc(&(current->FP32_weights), flat_in_size*(nb_neurons+1)*sizeof(float));
			vram_approx += flat_in_size*(nb_neurons+1)*sizeof(float);
			cudaMemcpy(current->FP32_weights, temp_tab, flat_in_size 
				* (nb_neurons+1) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(current->weights), flat_in_size*(nb_neurons+1)*sizeof(half));
			vram_approx += flat_in_size*(nb_neurons+1)*sizeof(half);
			#endif
			break;
			
		case BF16C_FP32A:
			#if defined(GEN_AMPERE) 
			temp_tab = (float*)current->weights;
			cudaMalloc(&(current->FP32_weights), flat_in_size*(nb_neurons+1)*sizeof(float));
			vram_approx += flat_in_size*(nb_neurons+1)*sizeof(float);
			cudaMemcpy(current->FP32_weights, temp_tab, flat_in_size 
				* (nb_neurons+1) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(current->weights),flat_in_size*(nb_neurons+1)*sizeof(nv_bfloat16));
			vram_approx += flat_in_size*(nb_neurons+1)*sizeof(nv_bfloat16);
			#endif
			break;
	}
	
	vram_approx += cuda_convert_table(net, &(current->output), (nb_neurons+1) * net->batch_size, 0);
		
	if(current->dropout_rate > 0.01f)
		vram_approx += cuda_convert_table_FP32((void**)&(current->dropout_mask), (nb_neurons+1) * net->batch_size, 0);
	
	if(!net->inference_only)
	{
		if(net->use_wema)
			vram_approx += cuda_convert_table_FP32((void**)&(current->ema_weights), flat_in_size*(nb_neurons+1), 0);
		
		vram_approx += cuda_convert_table(net, &(current->gradient), flat_in_size*(nb_neurons+1), 0);
		vram_approx += cuda_convert_table(net, &(current->delta_o), (nb_neurons+1) * net->batch_size, 0);
		
		vram_approx += cuda_convert_optimizer_var(current, flat_in_size * (nb_neurons+1));
	}

	return vram_approx;
}

void cuda_free_dense(layer *current)
{
	d_param = (dense_param*) current->param;
	
	cudaFree(current->weights);
	if(current->c_network->cu_inst.use_cuda_TC != FP32C_FP32A && current->c_network->cu_inst.use_cuda_TC != TF32C_FP32A)
		cudaFree(current->FP32_weights);
	cudaFree(current->output);
	
	if(current->dropout_rate > 0.01f)
		cudaFree(current->dropout_mask);
	
	if(current->previous != NULL && current->previous->output_type != FLAT)
		cudaFree(d_param->flat_input);
	
	if(!current->c_network->inference_only)
	{
		if(current->c_network->use_wema)
			cudaFree(current->ema_weights);
		cudaFree(current->gradient);
		cudaFree(current->delta_o);
		if(current->previous != NULL && current->previous->output_type != FLAT)
			cudaFree(d_param->flat_delta_o);
		
		cuda_free_optimizer_var(current);
	}
}


void cuda_forward_dense_layer(layer *current)
{
	int i, nb_neurons;
	size_t flat_in_size = 1;
	void *ref_input, *l_weights;
	
	network* net = current->c_network;
	d_param = (dense_param*) current->param;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	if(current->previous == NULL)
	{
		ref_input = net->input;
		
		cu_blocks = (net->batch_size + cu_threads - 1) / cu_threads;
		net->cu_inst.cu_dense_fcts.set_input_bias<<< cu_blocks, cu_threads >>>(
			ref_input, flat_in_size-1, current->bias_value, net->batch_size);
		
		current->input = net->input;
	}
	else
		current->input = current->previous->output;
	
	ref_input = current->input;
	
	if(net->is_inference == 1 && net->use_wema)
	{
		if(current->FP32_weights == current->weights) //Equivalent to test if mixed precision is off or FP32C_FP32A
			l_weights = (void*) current->ema_weights;
		else
		{
			cuda_master_weight_copy(net, (float*)current->ema_weights, current->weights, flat_in_size*(nb_neurons+1));
			l_weights = current->weights;
		}	
	}
	else
	{
		if(current->FP32_weights == current->weights)
			l_weights = (void*) current->FP32_weights;
		else
		{
			cuda_master_weight_copy(net, (float*)current->FP32_weights, current->weights, flat_in_size*(nb_neurons+1));
			l_weights = current->weights;
		}
	}
	
	if(current->previous != NULL && current->previous->output_type != FLAT)
	{
		cu_blocks = (flat_in_size * net->batch_size + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_dense_fcts.flat_dense_fct<<< cu_blocks, cu_threads >>>(
			current->input, d_param->flat_input, current->bias_value, 
			current->prev_dim[0]*current->prev_dim[1]*current->prev_dim[2],
			flat_in_size, current->prev_dim[3], net->batch_size, flat_in_size * net->batch_size);
		
		ref_input = d_param->flat_input;
	}
	
	cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, nb_neurons+1, 
		net->batch_size, flat_in_size, cu_alpha,  l_weights, cuda_data_type, 
		nb_neurons+1, ref_input, cuda_data_type, flat_in_size, cu_beta, 
		current->output, cuda_data_type, nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	if(current->dropout_rate > 0.01f)
	{
	
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			cu_blocks = ((nb_neurons+1) * net->batch_size + cu_threads - 1) / cu_threads;
			cuda_random_vector(current->dropout_mask, (nb_neurons+1) * net->batch_size);
			net->cu_inst.cu_dense_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->output, 
				current->dropout_mask, (nb_neurons+1) * net->batch_size, (nb_neurons+1), current->dropout_rate);
		}
		else
			net->cu_inst.cu_dense_fcts.drop_scale_fct<<<cu_blocks, cu_threads>>>(current->output, 
				current->dropout_mask, (nb_neurons+1) * net->batch_size, (nb_neurons+1), current->dropout_rate);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
	{
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, (nb_neurons+1) * net->batch_size);
		if(current->previous != NULL && current->previous->output_type != FLAT)
			net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(d_param->flat_delta_o, 0, flat_in_size * net->batch_size);
	}
}


void cuda_backward_dense_layer(layer* current)
{
	size_t i, nb_neurons;
	size_t flat_in_size = 1;
	void* ref_input;

	network* net = current->c_network;
	d_param = (dense_param*) current->param;	

	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
	{
		cu_blocks = ((nb_neurons+1) * net->batch_size + cu_threads - 1) / cu_threads;
		net->cu_inst.cu_dense_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->delta_o, 
			current->dropout_mask, (nb_neurons+1) * net->batch_size, (nb_neurons+1), current->dropout_rate);
	}
	
	//######################## ERROR PROPAGATION ########################
	ref_input = current->input;
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, flat_in_size, 
			net->batch_size, nb_neurons+1, cu_alpha, current->weights, cuda_data_type, 
			nb_neurons+1, current->delta_o, cuda_data_type, nb_neurons+1, cu_alpha, 
			d_param->flat_delta_o, cuda_data_type, flat_in_size, cuda_compute_type,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		
		if(current->previous->output_type == SPATIAL)
		{
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			cu_blocks = ((flat_in_size-1) * net->batch_size + cu_threads - 1) / cu_threads;
			
			net->cu_inst.cu_dense_fcts.reroll_fct<<< cu_blocks, cu_threads >>>(d_param->flat_delta_o, 
				current->previous->delta_o, current->prev_dim[0]*current->prev_dim[1]*current->prev_dim[2], 
				flat_in_size, current->output_dim[3], net->batch_size,
				(flat_in_size-1) * net->batch_size);
		}
	}
		
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		if(current->previous != NULL && current->previous->output_type != FLAT)
			ref_input = d_param->flat_input;
		
		cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, nb_neurons+1, flat_in_size,
			net->batch_size, cu_alpha, current->delta_o, cuda_data_type, 
			nb_neurons+1, ref_input, cuda_data_type, flat_in_size, cu_beta,
			current->gradient, cuda_data_type, nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		
		net->optim_update_fct_gpu(current, (flat_in_size-1)*(nb_neurons+1), 
			flat_in_size*(nb_neurons+1), flat_in_size*(nb_neurons+1));
		
		if(current->wema_replace_signal > 0)
		{
			cudaMemcpy(current->FP32_weights, current->ema_weights, 
				flat_in_size * (nb_neurons+1) * sizeof(float), cudaMemcpyDeviceToDevice);
			current->wema_replace_signal = 0;
		}
	}
}


void cuda_dense_define(layer *current)
{
	current->forward = cuda_forward_dense_layer;
	current->backprop = cuda_backward_dense_layer;
}






