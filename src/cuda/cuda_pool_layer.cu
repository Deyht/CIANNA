

/*
	Copyright (C) 2023 David Cornu
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

#define max_pooling_kernel(name, type)																											\
__global__ void max_pooling_kernel_##name																										\
	(void* i_input, void* i_output, int* pool_map,																								\
	int pool_size_w, int pool_size_h, int pool_size_d, 																							\
	int stride_w, int stride_h ,int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int w_size, int h_size, int d_size, 																										\
	int w_size_out, int h_size_out, int d_size_out, int bias_in, int length)																	\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int k = blockIdx.y*blockDim.y + threadIdx.y;																								\
	int x, y, z, x_max, y_max, z_max, o_pos[3], i_pos[3], max_found = 0;																		\
	size_t l_pos;																																\
																																				\
	type val_max;																																\
																																				\
	type* input  = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
																																				\
	if(i >= w_size_out*h_size_out*d_size_out || k >= length)																					\
		return;																																	\
																																				\
	o_pos[2] = i / (w_size_out*h_size_out); 																									\
	o_pos[1] = (i % (w_size_out*h_size_out)) / w_size_out;																						\
	o_pos[0] = (i % (w_size_out*h_size_out)) % w_size_out;																						\
																																				\
	output += k*(size_t)(w_size_out*h_size_out*d_size_out);																						\
	if(pool_map != NULL)																														\
		pool_map += k*(size_t)(w_size_out*h_size_out*d_size_out);																				\
																																				\
	input += k*(size_t)(w_size*h_size*d_size+bias_in);																							\
																																				\
	i_pos[0] = o_pos[0]*stride_w;																												\
	i_pos[1] = o_pos[1]*stride_h;																												\
	i_pos[2] = o_pos[2]*stride_d;																												\
																																				\
	for(z = 0; z < pool_size_d; z++)																											\
	{																																			\
		if((i_pos[2] + z) < padding_d || (i_pos[2] + z) >= (d_size + padding_d))																\
			continue;																															\
		for(y = 0; y < pool_size_h; y++)																										\
		{																																		\
			if((i_pos[1] + y) < padding_h || (i_pos[1] + y) >= (h_size + padding_h))															\
				continue;																														\
			for(x = 0; x < pool_size_w; x++)																									\
			{																																	\
				if((i_pos[0] + x) < padding_w || (i_pos[0] + x) >= (w_size + padding_w))														\
					continue;																													\
				if(max_found == 0)																												\
				{																																\
					max_found = 1;																												\
					x_max = x; y_max = y; z_max = z;																							\
					l_pos = (i_pos[0] + x - padding_w) 																							\
						  + (i_pos[1] + y - padding_h) * w_size 																				\
						  + (i_pos[2] + z - padding_d) * (size_t)(w_size * h_size);																\
					val_max = input[l_pos];																										\
				}																																\
				else																															\
				{																																\
					l_pos = (i_pos[0] + x - padding_w) 																							\
						  + (i_pos[1] + y - padding_h) * w_size 																				\
						  + (i_pos[2] + z - padding_d) * (size_t)(w_size * h_size);																\
					if(input[l_pos] > val_max)																									\
					{																															\
						x_max = x; y_max = y; z_max = z;																						\
						val_max = input[l_pos];																									\
					}																															\
				}																																\
			}																																	\
		}																																		\
	}																																			\
	if(max_found == 0)																															\
	{																																			\
		if(pool_map != NULL)																													\
			pool_map[i] = -1;																													\
		output[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if(pool_map != NULL)																													\
			pool_map[i] = (z_max*(size_t)(pool_size_w*pool_size_h) + y_max*pool_size_w + x_max);												\
		output[i] = (type) val_max;																												\
	}																																			\
}


#define avg_pooling_kernel(name, type)																											\
__global__ void avg_pooling_kernel_##name																										\
	(void* i_input, void* i_output, int* pool_map, 																								\
	int pool_size_w, int pool_size_h, int pool_size_d,																							\
	int stride_w, int stride_h ,int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int w_size, int h_size, int d_size, 																										\
	int w_size_out, int h_size_out, int d_size_out, int bias_in, int length)																	\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int k = blockIdx.y*blockDim.y + threadIdx.y;																								\
	int x, y, z, o_pos[3], i_pos[3]; 																											\
	double r_avg = 0.0f;																														\
	int sum_elem = 0;																															\
																																				\
	type* input  = (type*) i_input;																												\
	type* output = (type*) i_output;																											\
																																				\
	if(i >= w_size_out*h_size_out*d_size_out || k >= length)																					\
		return;																																	\
																																				\
	o_pos[2] = i / (w_size_out*h_size_out); 																									\
	o_pos[1] = (i % (w_size_out*h_size_out)) / w_size_out;																						\
	o_pos[0] = (i % (w_size_out*h_size_out)) % w_size_out;																						\
																																				\
	output += k*(size_t)(w_size_out*h_size_out*d_size_out);																						\
	input += k*(size_t)(w_size*h_size*d_size+bias_in);																							\
																																				\
	i_pos[0] = o_pos[0]*stride_w;																												\
	i_pos[1] = o_pos[1]*stride_h;																												\
	i_pos[2] = o_pos[2]*stride_d;																												\
																																				\
	for(z = 0; z < pool_size_d; z++)																											\
	{																																			\
		if((i_pos[2] + z) < padding_d || (i_pos[2] + z) >= (d_size + padding_d))																\
			continue;																															\
		for(y = 0; y < pool_size_h; y++)																										\
		{																																		\
			if((i_pos[1] + y) < padding_h || (i_pos[1] + y) >= (h_size + padding_h))															\
				continue;																														\
			for(x = 0; x < pool_size_w; x++)																									\
			{																																	\
				if((i_pos[0] + x) < padding_w || (i_pos[0] + x) >= (w_size + padding_w))														\
					continue;																													\
				r_avg += (float) input[(i_pos[0] + x - padding_w) 																				\
						  + (i_pos[1] + y - padding_h) * w_size 																				\
						  + (i_pos[2] + z - padding_d) * (size_t)(w_size * h_size)];															\
				sum_elem += 1;																													\
			}																																	\
		}																																		\
	}																																			\
	output[i] = (type) (r_avg/(sum_elem));																										\
}

#define deltah_max_pool_cont(name, type)																										\
__global__ void deltah_max_pool_cont_##name																										\
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 																					\
	int pool_size_w, int pool_size_h, int pool_size_d, 																							\
	int stride_w, int stride_h ,int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int w_size, int h_size, int d_size, 																										\
	int w_size_out, int h_size_out, int d_size_out, size_t length)																				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	int map_batch, in_im_pos, loc;																												\
	int x, y, z, f_pos[3], pos[3];																												\
	float l_delta_h = 0.0f;																														\
																																				\
	type* delta_o = (type*) i_delta_o;																											\
	type* delta_o_unpool = (type*) i_delta_o_unpool;																							\
																																				\
	if(i >= length)																																\
		return;																																	\
																																				\
	map_batch = i / (size_t)(w_size*h_size*d_size);																								\
	in_im_pos =  i % (size_t)(w_size*h_size*d_size);																							\
																																				\
	delta_o +=  map_batch * (size_t)(w_size_out * h_size_out * d_size_out);																		\
	pool_map += map_batch * (size_t)(w_size_out * h_size_out * d_size_out);																		\
	delta_o_unpool += map_batch * (size_t)(w_size * h_size * d_size);																			\
	/*Note, only span non-padded array, so should get padded to compensate, or pad in the loops ?*/												\
	pos[2] = in_im_pos / (w_size*h_size) + padding_d;																							\
	pos[1] = (in_im_pos % (w_size*h_size)) / w_size + padding_h;																				\
	pos[0] = (in_im_pos % (w_size*h_size)) % w_size + padding_w;																				\
																																				\
	for(z = pos[2]/stride_d; (pos[2]-z*stride_d < pool_size_d); z -=1)																			\
	{																																			\
		f_pos[2] = pos[2]-z*stride_d;																											\
		if((z < 0) || (z >= d_size_out))																										\
			continue;																															\
		for(y = pos[1]/stride_h; (pos[1]-y*stride_h < pool_size_h); y -=1)																		\
		{																																		\
			f_pos[1] = pos[1]-y*stride_h;																										\
			if((y < 0) || (y >= h_size_out))																									\
				continue;																														\
			for(x = pos[0]/stride_w; (pos[0]-x*stride_w < pool_size_w); x -=1)																	\
			{																																	\
				f_pos[0] = pos[0]-x*stride_w;																									\
				if((x < 0) || (x >= w_size_out))																								\
					continue;																													\
				loc = z*w_size_out*h_size_out + y*w_size_out + x;																				\
				if(pool_map[loc] == (f_pos[2]*pool_size_w*pool_size_h + f_pos[1]*pool_size_w + f_pos[0]))										\
					l_delta_h += (float) delta_o[loc];																							\
			}																																	\
		}																																		\
	}																																			\
	delta_o_unpool[(pos[2]-padding_d)*(size_t)(w_size*h_size) + (pos[1]-padding_h)*w_size + (pos[0]-padding_w)] = (type) l_delta_h;				\
}


#define deltah_avg_pool_cont(name, type)																										\
__global__ void deltah_avg_pool_cont_##name																										\
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 																					\
	int pool_size_w, int pool_size_h, int pool_size_d,																							\
	int stride_w, int stride_h ,int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int w_size, int h_size, int d_size, 																										\
	int w_size_out, int h_size_out, int d_size_out, size_t length)																				\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	int map_batch, in_im_pos;																													\
	int x, y, z, pos[3];																														\
	float l_delta_h = 0.0f;																														\
																																				\
	type* delta_o = (type*) i_delta_o;																											\
	type* delta_o_unpool = (type*) i_delta_o_unpool;																							\
																																				\
	if(i >= length)																																\
		return;																																	\
																																				\
	map_batch = i / (size_t)(w_size*h_size*d_size);																								\
	in_im_pos =  i % (size_t)(w_size*h_size*d_size);																							\
																																				\
	delta_o +=  map_batch * (size_t)(w_size_out * h_size_out * d_size_out);																		\
	delta_o_unpool += map_batch * (size_t)(w_size * h_size * d_size);																			\
																																				\
	pos[2] = in_im_pos / (w_size*h_size) + padding_d;																							\
	pos[1] = (in_im_pos % (w_size*h_size)) / w_size + padding_h;																				\
	pos[0] = (in_im_pos % (w_size*h_size)) % w_size + padding_w;																				\
																																				\
	for(z = pos[2]/stride_d; (pos[2]-z*stride_d < pool_size_d); z -=1)																			\
	{																																			\
		if((z < 0) || (z >= d_size_out))																										\
			continue;																															\
		for(y = pos[1]/stride_h; (pos[1]-y*stride_h < pool_size_h); y -=1)																		\
		{																																		\
			if((y < 0) || (y >= h_size_out))																									\
				continue;																														\
			for(x = pos[0]/stride_w; (pos[0]-x*stride_w < pool_size_w); x -=1)																	\
			{																																	\
				if((x < 0) || (x >= w_size_out))																								\
					continue;																													\
				l_delta_h += (float)delta_o[z*(size_t)(w_size_out*h_size_out) + y*w_size_out + x]												\
					/(float)(pool_size_w*pool_size_h*pool_size_d);																				\
			}																																	\
		}																																		\
	}																																			\
	delta_o_unpool[(pos[2]-padding_d)*(size_t)(w_size*h_size) + (pos[1]-padding_h)*w_size + (pos[0]-padding_w)] = (type) l_delta_h;				\
}


#define cuda_dropout_apply_pool(name, type) 																									\
__global__ void cuda_dropout_apply_pool_##name(void* i_table, float* mask, size_t size, float drop_rate)										\
{ 																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x; 																							\
																																				\
	type *table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(mask[i] >= drop_rate)																													\
		mask[i] = 1.0f;																															\
	else																																		\
		mask[i] = 0.0f;																															\
	 																																			\
	table[i] = (type)((float)table[i]*mask[i]); 																										\
}


#define cuda_dropout_scale_pool(name, type) 																									\
__global__ void cuda_dropout_scale_pool_##name(void* i_table, float* mask, size_t size, float drop_rate)										\
{ 																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x; 																							\
																																				\
	type *table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	table[i] = (type)((float)table[i]*(1.0f-drop_rate)); 																						\
}


#define cuda_typed_memset(name, type)																											\
void cuda_typed_memset_##name(void* i_table, int value, size_t size)																			\
{																																				\
	type* table = (type*) i_table;																												\
																																				\
	cudaMemset(table,  value, size * sizeof(type));																								\
}



max_pooling_kernel(FP32, float);
avg_pooling_kernel(FP32, float);
deltah_max_pool_cont(FP32, float);
deltah_avg_pool_cont(FP32, float);
cuda_dropout_apply_pool(FP32, float);
cuda_dropout_scale_pool(FP32, float);
cuda_typed_memset(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
max_pooling_kernel(FP16, half);
avg_pooling_kernel(FP16, half);
deltah_max_pool_cont(FP16, half);
deltah_avg_pool_cont(FP16, half);
cuda_dropout_apply_pool(FP16, half);
cuda_dropout_scale_pool(FP16, half);
cuda_typed_memset(FP16, half);
#endif

#if defined (GEN_AMPERE)
max_pooling_kernel(BF16, nv_bfloat16);
avg_pooling_kernel(BF16, nv_bfloat16);
deltah_max_pool_cont(BF16, nv_bfloat16);
deltah_avg_pool_cont(BF16, nv_bfloat16);
cuda_dropout_apply_pool(BF16, nv_bfloat16);
cuda_dropout_scale_pool(BF16, nv_bfloat16);
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
			net->cu_inst.cu_pool_fcts.drop_scale_fct = cuda_dropout_scale_pool_FP32;
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
			net->cu_inst.cu_pool_fcts.drop_scale_fct = cuda_dropout_scale_pool_FP16;
			net->cu_inst.cu_pool_fcts.typed_memset_fct = cuda_typed_memset_FP16;
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
			net->cu_inst.cu_pool_fcts.drop_scale_fct = cuda_dropout_scale_pool_BF16;
			net->cu_inst.cu_pool_fcts.typed_memset_fct = cuda_typed_memset_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}

size_t cuda_convert_pool_layer(layer *current)
{
	p_param = (pool_param*)current->param;
	size_t vram_approx = 0;
	
	network* net = current->c_network;

	vram_approx += cuda_convert_table(net, &(current->output), p_param->nb_area[0]
		* p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * net->batch_size, 0);
	
	if(current->dropout_rate > 0.01f)
	{
		vram_approx += cuda_convert_table_FP32((void**)&(p_param->dropout_mask), p_param->nb_maps
			* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size, 0);
	}
	
	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table_int(&(p_param->pool_map), (size_t)(p_param->nb_area[0]
			* p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps * net->batch_size, 0);
		vram_approx += cuda_convert_table(net, &(current->delta_o), (size_t)(p_param->nb_area[0]
			* p_param->nb_area[1] * p_param->nb_area[2]) * p_param->nb_maps * net->batch_size, 0);
	}
	
	return vram_approx;
}


void cuda_forward_pool_layer(layer* current)
{
	int bias_in = 0;
	network* net = current->c_network;
	
	if(net->length == 0)
		return;
	
	if(current->previous == NULL)
	{
		current->input = net->input;
		bias_in = 1;
	}
		
	p_param = (pool_param*) current->param;
	
	dim3 threadsPerBlock(32, 8);
    dim3 numBlocks(((size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) + threadsPerBlock.x - 1) / threadsPerBlock.x,
    	(net->batch_size * p_param->nb_maps + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	switch(p_param->pool_type)
	{
		default:
		case MAX_pool:
			net->cu_inst.cu_pool_fcts.max_pool_fct<<< numBlocks , threadsPerBlock >>>(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->stride[0], p_param->stride[1], p_param->stride[2],
				p_param->padding[0], p_param->padding[1], p_param->padding[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				bias_in, p_param->nb_maps * net->batch_size);
			break;
		case AVG_pool:
			net->cu_inst.cu_pool_fcts.avg_pool_fct<<< numBlocks , threadsPerBlock >>>(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->stride[0], p_param->stride[1], p_param->stride[2],
				p_param->padding[0], p_param->padding[1], p_param->padding[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				bias_in, p_param->nb_maps * net->batch_size);
			break;
	}
	
	//Linear == No activation
	current->activation(current);

	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			cu_blocks = ((size_t)(p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) 
				* net->batch_size) + cu_threads - 1) / cu_threads;
			
			cuda_random_vector(p_param->dropout_mask, p_param->nb_maps * net->batch_size
				* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]));
			
			net->cu_inst.cu_pool_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->output, p_param->dropout_mask, p_param->nb_maps 
				* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2])* net->batch_size, current->dropout_rate);
		}
		else
			net->cu_inst.cu_pool_fcts.drop_scale_fct<<<cu_blocks, cu_threads>>>(current->output, p_param->dropout_mask, p_param->nb_maps 
				* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2])* net->batch_size, current->dropout_rate);
	}
}


void cuda_backward_pool_layer(layer* current)
{
	network* net = current->c_network;
	
	p_param = (pool_param*) current->param;
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
	{
		cu_blocks = ((size_t)(p_param->nb_maps * (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) 
			* net->batch_size) + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_pool_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->delta_o, p_param->dropout_mask, p_param->nb_maps 
			* (size_t)(p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size, current->dropout_rate);
	}

	if(current->previous != NULL)
	{
		if(current->previous->type == CONV ||
			((current->previous->type == NORM || current->previous->type == LRN) && current->previous->previous->type == CONV))
		{		
			net->cu_inst.cu_pool_fcts.typed_memset_fct(current->previous->delta_o, 0, p_param->nb_maps 
				* (size_t)(p_param->prev_size[0] * p_param->prev_size[1] * p_param->prev_size[2])
				* net->batch_size);
				
			cu_blocks = (net->batch_size * p_param->nb_maps *(size_t)(p_param->prev_size[0] 
				* p_param->prev_size[1] * p_param->prev_size[2]) + cu_threads - 1) / cu_threads;
			switch(p_param->pool_type)
			{
				default:
				case MAX_pool:
					net->cu_inst.cu_pool_fcts.max_deltah_pool_fct<<< cu_blocks, cu_threads >>>(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						p_param->stride[0], p_param->stride[1], p_param->stride[2],
						p_param->padding[0], p_param->padding[1], p_param->padding[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * (size_t)(p_param->prev_size[0] * p_param->prev_size[1] *p_param->prev_size[2]));
					break;
				
				case AVG_pool:
					net->cu_inst.cu_pool_fcts.avg_deltah_pool_fct<<< cu_blocks, cu_threads >>>(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						p_param->stride[0], p_param->stride[1], p_param->stride[2],
						p_param->padding[0], p_param->padding[1], p_param->padding[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * (size_t)(p_param->prev_size[0] * p_param->prev_size[1] *p_param->prev_size[2]));
					break;
			}
		}
		current->previous->deriv_activation(current->previous);
	}
}


void cuda_pool_define(layer *current)
{
	current->forward = cuda_forward_pool_layer;
	current->backprop = cuda_backward_pool_layer;
}





