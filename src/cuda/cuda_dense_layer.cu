

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


//used to reshape output of Conv layer that has the result of filter 1 continuous for the batch
//convert into all filters continuous for image 1, then image 2, ...
#define cuda_flat_dense(name, type) 																			\
__global__ void cuda_flat_dense_##name																			\
	(void* i_in, void* i_out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)	\
{																												\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																\
	int map_id, image_id, pos;																					\
																												\
	type* in = (type*) i_in;																					\
	type* out = (type*) i_out;																					\
																												\
	if(i < size)																								\
	{																											\
		image_id = i / flatten_size;																			\
		map_id = (i % flatten_size)/map_size;																	\
		pos = (i % flatten_size)%map_size;																		\
																												\
		if(map_id >= nb_map)																					\
			out[i] = (type) bias;																				\
		else																									\
			out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];								\
	}																											\
}


#define cuda_reroll_batch(name, type) 																			\
__global__ void cuda_reroll_batch_##name																		\
	(void* i_in, void* i_out, int map_size, int flatten_size, int nb_map, int batch_size, int size)				\
{																												\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																\
	int map_id, image_id, pos;																					\
																												\
	type* in = (type*) i_in;																					\
	type* out = (type*) i_out;																					\
																												\
	if(i < size)																								\
	{																											\
		map_id = i / (map_size*batch_size);																		\
		image_id = (i % (map_size*batch_size))/map_size;														\
		pos = (i % (map_size*batch_size))%map_size;																\
																												\
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];											\
	}																											\
}


__global__ void init_block_state(unsigned int seed,  curandState_t* states)
{
	curand_init((seed << 20) + blockIdx.x, /* the seed can be the same for each core, here we pass the time in from the CPU */
              0, /* the sequence number should be different for each core (unless you want all
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

#define cuda_dropout_apply(name, type) 																			\
__global__ void cuda_dropout_apply_##name(void* i_table, int batch_size, int dim, int* mask)					\
{																												\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																\
	int j = blockIdx.y*blockDim.y + threadIdx.y;																\
																												\
	type* table = (type*) i_table;																				\
																												\
	if(i < batch_size && j < dim)																				\
		table[i*(dim+1) + j] *= mask[j];																		\
}



cuda_flat_dense(FP32, float);
cuda_reroll_batch(FP32, float);
cuda_dropout_apply(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
cuda_flat_dense(FP16, half);
cuda_reroll_batch(FP16, half);
cuda_dropout_apply(FP16, half);
#endif

#if defined (GEN_AMPERE)
cuda_flat_dense(BF16, nv_bfloat16);
cuda_reroll_batch(BF16, nv_bfloat16);
cuda_dropout_apply(BF16, nv_bfloat16);
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
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_dense_fcts.flat_dense_fct = cuda_flat_dense_FP16;
			net->cu_inst.cu_dense_fcts.reroll_fct = cuda_reroll_batch_FP16;
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_dense_fcts.flat_dense_fct = cuda_flat_dense_BF16;
			net->cu_inst.cu_dense_fcts.reroll_fct = cuda_reroll_batch_BF16;
			net->cu_inst.cu_dense_fcts.drop_apply_fct = cuda_dropout_apply_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}

void cuda_dense_define(layer *current)
{
	current->forward = cuda_forward_dense_layer;
	current->backprop = cuda_backward_dense_layer;
}


size_t cuda_convert_dense_layer(layer *current)
{
	d_param = (dense_param*)current->param;
	size_t vram_approx = 0;
	#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
	float* temp_tab;
	#endif
	
	network* net = current->c_network;
	
	if(current->previous != NULL)
	{
		switch(current->previous->type)
		{	
			case CONV:
				vram_approx += cuda_convert_table(net, &(d_param->flat_input), d_param->in_size*net->batch_size);
				vram_approx += cuda_convert_table(net, &(d_param->flat_delta_o),
					(((conv_param*)current->previous->param)->nb_area[0] 
						* ((conv_param*)current->previous->param)->nb_area[1] 
						* ((conv_param*)current->previous->param)->nb_area[2] 
						* ((conv_param*)current->previous->param)->nb_filters + 1) 
						* net->batch_size);
				break;
				
			case POOL:
				vram_approx += cuda_convert_table(net, &(d_param->flat_input), d_param->in_size * net->batch_size);
				vram_approx += cuda_convert_table(net, &(d_param->flat_delta_o),
					(((pool_param*)current->previous->param)->nb_area[0]
						* ((pool_param*)current->previous->param)->nb_area[1] 
						* ((pool_param*)current->previous->param)->nb_area[2] 
						* ((pool_param*)current->previous->param)->nb_maps + 1) 
						* net->batch_size);
				break;
				
			case DENSE:
			default:
				d_param->flat_delta_o = current->previous->delta_o;
				break;
		}
	}
	
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			vram_approx += cuda_convert_table(net, &(d_param->weights), d_param->in_size*(d_param->nb_neurons+1));
			d_param->FP32_weights = d_param->weights;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			temp_tab = (float*)d_param->weights;
			cudaMalloc(&(d_param->FP32_weights), d_param->in_size*(d_param->nb_neurons+1)*sizeof(float));
			vram_approx += d_param->in_size*(d_param->nb_neurons+1)*sizeof(float);
			cudaMemcpy(d_param->FP32_weights, temp_tab, d_param->in_size 
				* (d_param->nb_neurons+1) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(d_param->weights), d_param->in_size*(d_param->nb_neurons+1)*sizeof(half));
			vram_approx += d_param->in_size*(d_param->nb_neurons+1)*sizeof(half);
			#endif
			break;
			
		case BF16C_FP32A:
			#if defined(GEN_AMPERE) 
			temp_tab = (float*)d_param->weights;
			cudaMalloc(&(d_param->FP32_weights), d_param->in_size*(d_param->nb_neurons+1)*sizeof(float));
			vram_approx += d_param->in_size*(d_param->nb_neurons+1)*sizeof(float);
			cudaMemcpy(d_param->FP32_weights, temp_tab, d_param->in_size 
				* (d_param->nb_neurons+1) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(d_param->weights),d_param->in_size*(d_param->nb_neurons+1)*sizeof(nv_bfloat16));
			vram_approx += d_param->in_size*(d_param->nb_neurons+1)*sizeof(nv_bfloat16);
			#endif
			break;
	}
	
	vram_approx += cuda_convert_table(net, &(d_param->update), d_param->in_size*(d_param->nb_neurons+1));
	vram_approx += cuda_convert_table(net, &(current->output), (d_param->nb_neurons+1) 
		* net->batch_size);
	vram_approx += cuda_convert_table(net, &(current->delta_o), (d_param->nb_neurons+1) 
		* net->batch_size);
		
	if(current->dropout_rate > 0.01f)
	{
		vram_approx += cuda_convert_table_int(&(d_param->dropout_mask), d_param->nb_neurons);
		cudaMalloc((void**) &d_param->block_state, (d_param->nb_neurons) * sizeof(curandState_t));
		vram_approx += (d_param->nb_neurons) * sizeof(curandState_t);
		cu_blocks = (d_param->nb_neurons);
		init_block_state<<< cu_blocks, 1>>>(time(NULL),(curandState_t*)d_param->block_state);
	}

	return vram_approx;
}


void cuda_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, nb_area_d, depth;
	
	void *ref_input;
	void *w_alpha;
	
	float w_f_alpha;
	
	float prev_drop_rate = 0.0f;
	
	network* net = current->c_network;
	
	#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
	half w_h_alpha;
	if(net->cu_inst.use_cuda_TC == FP16C_FP16A)
		w_alpha = &w_h_alpha;	
	else
	#endif
		w_alpha = &w_f_alpha;
	
	if(net->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	cuda_master_weight_copy(net, (float*)d_param->FP32_weights, d_param->weights, 
		d_param->in_size*(d_param->nb_neurons+1));
	
	if(current->previous == NULL)
		current->input = net->input;
	
	ref_input = current->input;
		
	if(current->previous != NULL && current->previous->type != DENSE)
	{
		//Use a converted (flatten) input if needed
		switch(current->previous->type)
		{
			case CONV:
				nb_area_w = ((conv_param*)current->previous->param)->nb_area[0];
				nb_area_h = ((conv_param*)current->previous->param)->nb_area[1];
				nb_area_d = ((conv_param*)current->previous->param)->nb_area[2];
				depth = ((conv_param*)current->previous->param)->nb_filters;
				break;
			
			case POOL:
			default:
				nb_area_w = ((pool_param*)current->previous->param)->nb_area[0];
				nb_area_h = ((pool_param*)current->previous->param)->nb_area[1];
				nb_area_d = ((pool_param*)current->previous->param)->nb_area[2];
				depth = ((pool_param*)current->previous->param)->nb_maps;
				break;
		}
		
		cu_blocks = ((nb_area_w * nb_area_h * nb_area_d * depth + 1) 
			* net->batch_size + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_dense_fcts.flat_dense_fct<<< cu_blocks, cu_threads >>>(current->input, 
			d_param->flat_input, current->bias_value, nb_area_w * nb_area_h * nb_area_d ,
			nb_area_w * nb_area_h * nb_area_d * depth + 1, depth, net->batch_size, 
			(nb_area_w * nb_area_h * nb_area_d * depth + 1) * net->batch_size);
		
		ref_input = d_param->flat_input;
	}
	prev_drop_rate = current->previous->dropout_rate;
	
	
	if(net->is_inference && net->inference_drop_mode == AVG_MODEL && current->previous != NULL)
	{
		if(net->cu_inst.use_cuda_TC == FP16C_FP16A)
			*((half*)w_alpha) = (1.0f/(1.0f + prev_drop_rate));	
		else
			*((float*)w_alpha)  = (1.0f/(1.0f + prev_drop_rate));
		 //bias weight is included in drop, should change this behavior ?
	}
	else
	{
		if(net->cu_inst.use_cuda_TC == FP16C_FP16A)
			*((half*)w_alpha) = 1.0f;	
		else
			*((float*)w_alpha) = 1.0f;
	}
	
	cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, 
		net->batch_size, d_param->in_size, w_alpha, d_param->weights, cuda_data_type, 
		d_param->nb_neurons+1, ref_input, cuda_data_type, d_param->in_size, cu_beta, 
		current->output, cuda_data_type, d_param->nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
	current->activation(current);

	if(current->dropout_rate > 0.01f && (!net->is_inference || net->inference_drop_mode == MC_MODEL))
	{
		// Must check performance impact -> the present approach is due to the curand behavior
		cu_blocks = (d_param->nb_neurons);
		cuda_dropout_select<<<cu_blocks, 1>>>(d_param->dropout_mask, d_param->nb_neurons, 
			current->dropout_rate, (curandState_t*) d_param->block_state);	

		dim3 threadsPerBlock(8, 32);
		dim3 numBlocks((net->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_dense_fcts.drop_apply_fct<<<numBlocks, threadsPerBlock>>>(current->output, 
			net->batch_size, d_param->nb_neurons, d_param->dropout_mask);
	}
}


void cuda_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, nb_area_d, depth;
	void* ref_input;

	network* net = current->c_network;

	d_param = (dense_param*) current->param;	
	
	if(current->dropout_rate > 0.01f)
	{
		dim3 threadsPerBlock(8, 32);
		dim3 numBlocks((net->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_dense_fcts.drop_apply_fct<<<numBlocks, threadsPerBlock>>>(current->delta_o, 
			net->batch_size, d_param->nb_neurons, d_param->dropout_mask);
	}
	
	//######################## ERROR PROPAGATION ########################

	ref_input = current->input;
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, d_param->in_size, 
			net->batch_size, d_param->nb_neurons+1, cu_alpha, d_param->weights, cuda_data_type, 
			d_param->nb_neurons+1, current->delta_o, cuda_data_type, d_param->nb_neurons+1, cu_beta, 
			d_param->flat_delta_o, cuda_data_type, d_param->in_size, cuda_compute_type,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		
		if(current->previous->type == POOL || current->previous->type == CONV)
		{
			switch(current->previous->type)
			{
				case POOL:
					nb_area_w = ((pool_param*)current->previous->param)->nb_area[0];
					nb_area_h = ((pool_param*)current->previous->param)->nb_area[1];
					nb_area_d = ((pool_param*)current->previous->param)->nb_area[2];
					depth = ((pool_param*)current->previous->param)->nb_maps;
					break;
			
				case CONV:
				default:
					nb_area_w = ((conv_param*)current->previous->param)->nb_area[0];
					nb_area_h = ((conv_param*)current->previous->param)->nb_area[1];
					nb_area_d = ((conv_param*)current->previous->param)->nb_area[2];
					depth = ((conv_param*)current->previous->param)->nb_filters;
					break;
			}
			
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			cu_blocks = (nb_area_w * nb_area_h * nb_area_d * depth 
				* net->batch_size + cu_threads - 1) / cu_threads;
			
			net->cu_inst.cu_dense_fcts.reroll_fct<<< cu_blocks, cu_threads >>>(d_param->flat_delta_o, 
				current->previous->delta_o, nb_area_w * nb_area_h * nb_area_d, 
				nb_area_w * nb_area_h * nb_area_d * depth + 1, depth, net->batch_size,
				nb_area_w * nb_area_h * nb_area_d * depth * net->batch_size);
		}
		
		current->previous->deriv_activation(current->previous);
	}
		
	//########################  WEIGHTS UPDATE   ########################
	if(current->previous != NULL && current->previous->type != DENSE)
		ref_input = d_param->flat_input;
	
	if(!current->frozen)
	{
		set_cu_learning_rate_and_momentum(net);
		
		cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
			net->batch_size, cu_learning_rate, current->delta_o, cuda_data_type, 
			d_param->nb_neurons+1, ref_input, cuda_data_type, d_param->in_size, cu_momentum,
			d_param->update, cuda_data_type, d_param->nb_neurons+1, cuda_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		
		cuda_update_weights(net, d_param->FP32_weights, d_param->update, d_param->in_size 
			* (d_param->nb_neurons+1));
	}
}







