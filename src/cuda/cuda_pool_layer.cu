
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
static pool_param *p_param;

//public are in prototypes.h

//private
void cuda_forward_pool_layer(layer* current);
void cuda_backward_pool_layer(layer* current);


#define max_pooling_kernel(name, type)																				\
__global__ void max_pooling_kernel_##name																			\
	(void* i_input, void* i_output, int* pool_map,																	\
	int pool_size_w, int pool_size_h, int pool_size_d, 																\
	int w_size, int h_size, int d_size, 																			\
	int w_size_out, int h_size_out, int d_size_out, int length)														\
{																													\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																	\
	int k = blockIdx.y*blockDim.y + threadIdx.y;																	\
	int x, y, z, x_max, y_max, z_max, pos, pos_x, pos_y, pos_z, pos_out;											\
																													\
	type* input  = (type*) i_input;																					\
	type* output = (type*) i_output;																				\
																													\
	pos_z = i / (w_size_out*h_size_out); 																			\
	pos_y = (i % (w_size_out*h_size_out)) / w_size_out;																\
	pos_x = (i % (w_size_out*h_size_out)) % w_size_out;																\
																													\
	pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);		\
																													\
	pos = k*w_size*h_size*d_size + pos_x*pool_size_w + pos_y*pool_size_h*w_size + pos_z*pool_size_d*w_size*h_size;	\
																													\
	if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)								\
	{																												\
		x_max = 0; y_max = 0; z_max = 0;																			\
		for(x = 0; x < pool_size_d; x++)																			\
			for(y = 0; y < pool_size_h; y++)																		\
				for(z = 0; z < pool_size_w; z++)																	\
					if(input[pos + x_max*w_size*h_size + y_max*w_size + z_max] 										\
						< input[pos + x*w_size*h_size + y*w_size + z])												\
					{																								\
						x_max = x; y_max = y; z_max = z;															\
					}																								\
		pool_map[pos_out] = (x_max*pool_size_w*pool_size_h + y_max*pool_size_w + z_max);							\
		output[pos_out] = input[pos + x_max*w_size*h_size + y_max*w_size + z_max];									\
	}																												\
}


#define avg_pooling_kernel(name, type)																				\
__global__ void avg_pooling_kernel_##name																			\
	(void* i_input, void* i_output, int* pool_map, 																	\
	int pool_size_w, int pool_size_h, int pool_size_d,																\
	int w_size, int h_size, int d_size, 																			\
	int w_size_out, int h_size_out, int d_size_out, int length)														\
{																													\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																	\
	int k = blockIdx.y*blockDim.y + threadIdx.y;																	\
	int x, y, z, pos, pos_x, pos_y, pos_z, pos_out;																	\
	float r_avg = 0.0f;																								\
																													\
	type* input  = (type*) i_input;																					\
	type* output = (type*) i_output;																				\
																													\
	pos_z = i / (w_size_out*h_size_out); 																			\
	pos_y = (i % (w_size_out*h_size_out)) / w_size_out;																\
	pos_x = (i % (w_size_out*h_size_out)) % w_size_out;																\
																													\
	pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);		\
																													\
	pos = k*w_size*h_size*d_size + pos_x*pool_size_w + pos_y*pool_size_h*w_size + pos_z*pool_size_d*w_size*h_size;	\
																													\
	if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)								\
	{																												\
		for(x = 0; x < pool_size_d; x++)																			\
			for(y = 0; y < pool_size_h; y++)																		\
				for(z = 0; z < pool_size_w; z++)																	\
					r_avg += (float) input[pos + x*w_size*h_size + y*w_size + z];									\
																													\
		output[pos_out] = (type) (r_avg/(pool_size_w*pool_size_h*pool_size_d));										\
	}																												\
}

#define deltah_max_pool_cont(name, type)																			\
__global__ void deltah_max_pool_cont_##name																			\
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 														\
	int pool_size_w, int pool_size_h, int pool_size_d, 																\
	int len, int batch_size, int image_size, int w_size, int h_size)												\
{																													\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																	\
																													\
	type* delta_o = (type*) i_delta_o;																				\
	type* delta_o_unpool = (type*) i_delta_o_unpool;																\
																													\
	if(i < len*image_size)																							\
	{																												\
		/*add mask of locations*/																					\
		delta_o_unpool += (i/(w_size*h_size)) * (w_size*h_size) * pool_size_w * pool_size_h * pool_size_d			\
			+ ((i%(w_size*h_size))/w_size) * w_size * pool_size_w * pool_size_h										\
			+ ((i%(w_size*h_size))%w_size) * pool_size_w +															\
			+ (int(pool_map[i])/(pool_size_w*pool_size_h)) * w_size*h_size * pool_size_w*pool_size_h 				\
			+ ((int(pool_map[i])%(pool_size_w*pool_size_h))/pool_size_h) * w_size * pool_size_w						\
			+ ((int(pool_map[i])%(pool_size_w*pool_size_h))%pool_size_h);											\
																													\
		*delta_o_unpool = delta_o[i];																				\
	}																												\
}


#define deltah_avg_pool_cont(name, type)																			\
__global__ void deltah_avg_pool_cont_##name																			\
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 														\
	int pool_size_w, int pool_size_h, int pool_size_d,																\
	int len, int batch_size, int image_size, int w_size, int h_size)												\
{																													\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																	\
	int x, y, z;																									\
																													\
	type* delta_o = (type*) i_delta_o;																				\
	type* delta_o_unpool = (type*) i_delta_o_unpool;																\
																													\
	if(i < len*image_size)																							\
	{																												\
		/*add mask of locations*/																					\
		delta_o_unpool += (i/(w_size*h_size)) * (w_size*h_size) * pool_size_w * pool_size_h * pool_size_d			\
						+ ((i%(w_size*h_size))/h_size) * h_size * pool_size_w * pool_size_h							\
						+ ((i%(w_size*h_size))%h_size) * pool_size_w;												\
																													\
		for(x = 0; x < pool_size_d; x++)																			\
			for(y = 0; y < pool_size_h; y++)																		\
				for(z = 0; z < pool_size_w; z++)																	\
					 delta_o_unpool[(x) * w_size * h_size * pool_size_w * pool_size_h 								\
						+ (y) * w_size * pool_size_w + (z)] 														\
						= (type)((float)delta_o[i]/(pool_size_w*pool_size_h*pool_size_d));							\
	}																												\
}

__global__ void init_block_state_pool(unsigned int seed,  curandState_t* states)
{
	curand_init((seed << 20) + blockIdx.x, /* the seed can be the same for each core, here we pass the time in from the CPU */
              0, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! 
			     Currently use an alternative definition with Id adjunct to seed*/
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}


__global__ void cuda_dropout_select_pool(int* mask, int size, float drop_rate, curandState_t* states)
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

#define cuda_dropout_apply_pool(name, type) 																\
__global__ void cuda_dropout_apply_pool_##name(void* i_table, int batch_size, int dim, int* mask, int size)	\
{																											\
	int j = blockIdx.x*blockDim.x + threadIdx.x;															\
	int i = blockIdx.y*blockDim.y + threadIdx.y;															\
																											\
	int c_depth = j / dim;																					\
	int current_id = j % dim;																				\
	int offset = dim*batch_size;																			\
																											\
	type* table = (type*) i_table;																			\
																											\
	if(i < batch_size && j < size)																			\
		table[i*dim + c_depth*offset + current_id] *= mask[j];												\
}

#define cuda_typed_memset(name, type)																		\
void cuda_typed_memset_##name(void* i_table, int value, int size)											\
{																											\
	type* table = (type*) i_table;																			\
																											\
	cudaMemset(table,  value, size * sizeof(type));															\
}



max_pooling_kernel(FP32, float);
avg_pooling_kernel(FP32, float);
deltah_max_pool_cont(FP32, float);
deltah_avg_pool_cont(FP32, float);
cuda_dropout_apply_pool(FP32, float);
cuda_typed_memset(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
max_pooling_kernel(FP16, half);
avg_pooling_kernel(FP16, half);
deltah_max_pool_cont(FP16, half);
deltah_avg_pool_cont(FP16, half);
cuda_dropout_apply_pool(FP16, half);
cuda_typed_memset(FP16, half);
#endif

#if defined (GEN_AMPERE)
max_pooling_kernel(BF16, nv_bfloat16);
avg_pooling_kernel(BF16, nv_bfloat16);
deltah_max_pool_cont(BF16, nv_bfloat16);
deltah_avg_pool_cont(BF16, nv_bfloat16);
cuda_dropout_apply_pool(BF16, nv_bfloat16);
cuda_typed_memset(BF16, nv_bfloat16);
#endif


void cuda_pool_init(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_pool_fcts.max_pool_fct = max_pooling_kernel_FP32;
			net->cu_inst.cu_pool_fcts.max_deltah_pool_fct = deltah_max_pool_cont_FP32;
			net->cu_inst.cu_pool_fcts.avg_pool_fct = avg_pooling_kernel_FP32;
			net->cu_inst.cu_pool_fcts.avg_deltah_pool_fct = deltah_avg_pool_cont_FP32;
			net->cu_inst.cu_pool_fcts.drop_apply_fct = cuda_dropout_apply_pool_FP32;
			net->cu_inst.cu_pool_fcts.typed_memset_fct = cuda_typed_memset_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_pool_fcts.max_pool_fct = max_pooling_kernel_FP16;
			net->cu_inst.cu_pool_fcts.max_deltah_pool_fct = deltah_max_pool_cont_FP16;
			net->cu_inst.cu_pool_fcts.avg_pool_fct = avg_pooling_kernel_FP16;
			net->cu_inst.cu_pool_fcts.avg_deltah_pool_fct = deltah_avg_pool_cont_FP16;
			net->cu_inst.cu_pool_fcts.drop_apply_fct = cuda_dropout_apply_pool_FP16;
			net->cu_inst.cu_pool_fcts.typed_memset_fct = cuda_typed_memset_FP16;
			break;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_pool_fcts.max_pool_fct = max_pooling_kernel_BF16;
			net->cu_inst.cu_pool_fcts.max_deltah_pool_fct = deltah_max_pool_cont_BF16;
			net->cu_inst.cu_pool_fcts.avg_pool_fct = avg_pooling_kernel_BF16;
			net->cu_inst.cu_pool_fcts.avg_deltah_pool_fct = deltah_avg_pool_cont_BF16;
			net->cu_inst.cu_pool_fcts.drop_apply_fct = cuda_dropout_apply_pool_BF16;
			net->cu_inst.cu_pool_fcts.typed_memset_fct = cuda_typed_memset_BF16;
			break;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}

void cuda_pool_define(layer *current)
{
	current->forward = cuda_forward_pool_layer;
	current->backprop = cuda_backward_pool_layer;
}

size_t cuda_convert_pool_layer(layer *current)
{
	p_param = (pool_param*)current->param;
	size_t vram_approx = 0;
	
	network* net = current->c_network;

	vram_approx += cuda_convert_table_int(&(p_param->pool_map), p_param->nb_area[0] 
		* p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * net->batch_size);
	vram_approx += cuda_convert_table(net, &(current->output), p_param->nb_area[0] 
		* p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * net->batch_size);
	
	vram_approx += cuda_convert_table(net, &(current->delta_o), p_param->nb_area[0] 
		* p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * net->batch_size);
		
	if(p_param->dropout_rate > 0.01)
	{
		vram_approx += cuda_convert_table_int(&(p_param->dropout_mask), p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));
		cudaMalloc((void**) &p_param->block_state, (p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2])) * sizeof(curandState_t));
		cu_blocks = (p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));
		init_block_state_pool<<< cu_blocks, 1>>>(time(NULL),(curandState_t*)p_param->block_state);
	}
	
	return vram_approx;
}


void cuda_forward_pool_layer(layer* current)
{
	network* net = current->c_network;
	
	if(net->length == 0)
		return;
		
	p_param = (pool_param*) current->param;
	
	//late declaration of CUDA kernel sizes
	dim3 threadsPerBlock(8, 8);
	//create numBlocks regarding the layer dimensions
    dim3 numBlocks((p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] + threadsPerBlock.x - 1) / threadsPerBlock.x,
    	(net->batch_size * p_param->nb_maps + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	switch(p_param->pool_type)
	{
		default:
		case MAX_pool:
			net->cu_inst.cu_pool_fcts.max_pool_fct<<< numBlocks , threadsPerBlock >>>(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				p_param->nb_maps * net->batch_size);
			break;
		case AVG_pool:
			net->cu_inst.cu_pool_fcts.avg_pool_fct<<< numBlocks , threadsPerBlock >>>(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				p_param->nb_maps * net->batch_size);
			break;
	}

	if(p_param->dropout_rate > 0.01 && (!net->is_inference || net->inference_drop_mode == MC_MODEL))
	{
		cu_blocks = (p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));
		cuda_dropout_select_pool<<<cu_blocks, 1>>>(p_param->dropout_mask, p_param->nb_maps 
			* (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]), p_param->dropout_rate, (curandState_t*) p_param->block_state);	
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		
		net->cu_inst.cu_pool_fcts.drop_apply_fct<<<numBlocks, threadsPerBlock>>>(current->output, 
			net->batch_size, (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]),
			p_param->dropout_mask, p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));

	}
}


void cuda_backward_pool_layer(layer* current)
{	
	p_param = (pool_param*) current->param;
	
	network* net = current->c_network;
	
	if(p_param->dropout_rate > 0.01)
	{
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(net->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		net->cu_inst.cu_pool_fcts.drop_apply_fct<<<numBlocks, threadsPerBlock>>>(current->delta_o, 
			net->batch_size, (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]), 
			p_param->dropout_mask, p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));
	}

	if(current->previous != NULL)
	{
		if(current->previous->type == CONV)
		{		
			net->cu_inst.cu_pool_fcts.typed_memset_fct(current->previous->delta_o, 0, p_param->nb_maps 
				* p_param->prev_size[0] * p_param->prev_size[1] *p_param->prev_size[2]
				* net->batch_size);
				
			cu_blocks = (net->batch_size*(p_param->nb_maps * p_param->nb_area[0] 
				* p_param->nb_area[1] * p_param->nb_area[2]) + cu_threads - 1) / cu_threads;
			switch(p_param->pool_type)
			{
				default:
				case MAX_pool:
					net->cu_inst.cu_pool_fcts.max_deltah_pool_fct<<< cu_blocks, cu_threads >>>(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						net->length, net->batch_size, p_param->nb_maps 
						* p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2], p_param->nb_area[0], p_param->nb_area[1]);
					break;
				
				case AVG_pool:
					net->cu_inst.cu_pool_fcts.avg_deltah_pool_fct<<< cu_blocks, cu_threads >>>(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						net->length, net->batch_size, p_param->nb_maps 
						* p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2], p_param->nb_area[0], p_param->nb_area[1]);
					break;
			}
		}
	}
}








