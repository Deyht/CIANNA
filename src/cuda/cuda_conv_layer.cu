

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
static conv_param *c_param;

//public are in prototypes.h


//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only one load per element of the original image.
//VERSION 5.3
#define im2col_kernel_v5(name, type) 																											\
__global__ void im2col_kernel_v5_##name																											\
	(void* i_output, void* i_input, 																											\
	int image_size, size_t flat_image_size, 																									\
	int stride_w, int stride_h ,int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int internal_padding_w, int internal_padding_h, int internal_padding_d, 																	\
	int channel, int channel_padding, int image_padding, int TC_padding, 																		\
	int batch_size, int f_size_w, int f_size_h, int f_size_d, int flat_f_size, 																	\
	int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int nb_area_d, int bias_in, int bias_out) 								\
{																																				\
	int p = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int c = blockIdx.y*blockDim.y + threadIdx.y;																								\
	int i = blockIdx.z*blockDim.z + threadIdx.z;																								\
																																				\
	type local_pix;																																\
	type *output = (type*) i_output;																											\
	type *input  = (type*) i_input;																												\
																																				\
	int w, h, d, x, y, z;																														\
	int pos_w_filter, pos_h_filter, pos_d_filter;																								\
	long long int loc;																															\
																																				\
	if(i < batch_size)																															\
	{																																			\
		input += i*(image_padding + bias_in);																									\
		output += i*(flat_image_size);																											\
																																				\
		if(c < channel)																															\
		{																																		\
			input += c * channel_padding;																										\
			output += c * f_size_w*f_size_h*f_size_d;																							\
			if(p < image_size)																													\
			{																																	\
				local_pix = input[p];																											\
																																				\
				d = (p / (w_size*h_size)) * (1+internal_padding_d) + padding_d;																	\
				h = (p % (w_size*h_size) / w_size) * (1+internal_padding_h) + padding_h;														\
				w = (p % (w_size*h_size) % w_size) * (1+internal_padding_w) + padding_w;														\
																																				\
				for(z = d/stride_d; (d-z*stride_d < f_size_d); z -=1)																			\
				{																																\
					pos_d_filter = d-z*stride_d;																								\
					if((z < 0) || (pos_d_filter > d_size + (d_size-1)*internal_padding_d + 2*padding_d - f_size_d))								\
						continue;																												\
					for(y = h/stride_h; (h-y*stride_h < f_size_h); y -= 1)																		\
					{																															\
						pos_h_filter = h-y*stride_h;																							\
						if((y < 0) || (pos_h_filter > h_size + (h_size-1)*internal_padding_h + 2*padding_h - f_size_h))							\
							continue;																											\
						for(x = w/stride_w; (w-x*stride_w < f_size_w); x -= 1)																	\
						{																														\
							pos_w_filter = w-x*stride_w;																						\
							if((x < 0) || (pos_w_filter > w_size + (w_size-1)*internal_padding_w + 2*padding_w - f_size_w))						\
								continue;																										\
							loc = (z*(long long int)nb_area_w*nb_area_h + y*nb_area_w + x)*(flat_f_size+TC_padding)								\
								 + pos_w_filter + pos_h_filter*f_size_w + pos_d_filter*f_size_w*f_size_h;										\
							if((bias_out && (loc)%(flat_f_size+TC_padding) >= flat_f_size - 1))													\
								continue;																										\
							if(loc >= 0 && loc < flat_image_size)																				\
								output[loc] = local_pix;																						\
						}																														\
					}																															\
				}																																\
			}																																	\
		}																																		\
	}																																			\
}


#define cuda_rotate_filter_matrix(name, type) 																									\
__global__ void cuda_rotate_filter_matrix_##name																								\
	(void* i_in, void* i_out, int nb_rows, int TC_padding, int depth_size, int nb_filters_in, int len)											\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int x, y, depth_id;																															\
																																				\
	type* in  = (type*) i_in;																													\
	type* out = (type*) i_out;																													\
																																				\
	if(i < len)																																	\
	{																																			\
		/*Rotate and move the filters*/																											\
		x = i / nb_rows;																														\
		y = i % nb_rows;																														\
		/*remove the weights of the bias nodes*/																								\
		if(y < nb_rows-1-TC_padding) 																											\
		{																																		\
			depth_id = y / depth_size;																											\
			out[depth_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];						\
		}																																		\
	}																																			\
}


#define cuda_dropout_apply_conv(name, type) 																									\
__global__ void cuda_dropout_apply_conv_##name(void* i_table, float* mask, size_t size, float drop_rate)										\
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
	table[i] = (type)((float)table[i]*mask[i]);																									\
}


#define cuda_dropout_scale_conv(name, type) 																									\
__global__ void cuda_dropout_scale_conv_##name(void* i_table, float* mask, size_t size, float drop_rate)										\
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



im2col_kernel_v5(FP32, float);
cuda_rotate_filter_matrix(FP32, float); 
cuda_dropout_apply_conv(FP32, float);
cuda_dropout_scale_conv(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
im2col_kernel_v5(FP16, half);
cuda_rotate_filter_matrix(FP16, half);
cuda_dropout_apply_conv(FP16, half);
cuda_dropout_scale_conv(FP16, half);
#endif

#if defined (GEN_AMPERE)
im2col_kernel_v5(BF16, nv_bfloat16);
cuda_rotate_filter_matrix(BF16, nv_bfloat16); 
cuda_dropout_apply_conv(BF16, nv_bfloat16);
cuda_dropout_scale_conv(BF16, nv_bfloat16);
#endif


void cuda_conv_init(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_v5_FP32;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_FP32;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_FP32;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_v5_FP16;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_FP16;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_FP16;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_v5_BF16;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_BF16;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_BF16;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_conv_layer(layer *current)
{
	c_param = (conv_param*)current->param;
	size_t vram_approx = 0;
	#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
	float* temp_tab;
	#endif

	network* net = current->c_network;

	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			vram_approx += cuda_convert_table(net, &(c_param->filters), c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding),0);
			c_param->FP32_filters = c_param->filters;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			temp_tab = (float*)c_param->filters;
			cudaMalloc(&(c_param->FP32_filters), c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(float));
			vram_approx += c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(float);
			cudaMemcpy(c_param->FP32_filters, temp_tab, c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(c_param->filters), c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(half));
			vram_approx += c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(half);
			#endif
			break;
		
		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			temp_tab = (float*)c_param->filters;
			cudaMalloc(&(c_param->FP32_filters), c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(float));
			vram_approx += c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(float);
			cudaMemcpy(c_param->FP32_filters, temp_tab, c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding) * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(c_param->filters), c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(nv_bfloat16));
			vram_approx += c_param->nb_filters 
				* (c_param->flat_f_size + c_param->TC_padding)*sizeof(nv_bfloat16);
			#endif
			break;
	}
	
	vram_approx += cuda_convert_table(net, &(current->output), c_param->nb_filters 
		* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size,0);
	
	vram_approx += cuda_convert_table(net, &(c_param->im2col_input), 
		(c_param->flat_f_size + c_param->TC_padding) * (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
		* net->batch_size,0);
	
	if(current->dropout_rate > 0.01f)
	{
		vram_approx += cuda_convert_table_FP32((void**)&(c_param->dropout_mask), c_param->nb_filters 
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size,0);
	}
	
	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table(net, &(c_param->update), c_param->nb_filters 
			* (c_param->flat_f_size + c_param->TC_padding),0);
	
		vram_approx += cuda_convert_table(net, &(c_param->rotated_filters), c_param->nb_filters 
			* (c_param->flat_f_size-1),0);
	
		vram_approx += cuda_convert_table(net, &(current->delta_o), c_param->nb_filters * net->batch_size
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), 0);
	
		if(current->previous != NULL && current->previous->type == DENSE)
			vram_approx += cuda_convert_table(net, &(c_param->temp_delta_o), c_param->prev_depth * current->c_network->batch_size
				* (size_t)(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]), 0);
	
		vram_approx += cuda_convert_table(net, &(c_param->im2col_delta_o), 
			net->batch_size * (size_t)(c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]) 
			* (c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2] * c_param->nb_filters), 0);
	}
	
	return vram_approx;
}


void cuda_forward_conv_layer(layer *current)
{
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;
	int dim_a, dim_b, dim_c;
	
	network* net = current->c_network;
	if(net->length == 0)
		return;
	c_param = (conv_param*) current->param;
	
	if(current->previous == NULL || current->previous->type == DENSE)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it as continuous RGB images
		//size in line format
		depth_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2];
		image_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2] * c_param->prev_depth;
		if(current->previous == NULL)
			current->input = net->input;
		else
			current->input = current->previous->output;
		im2col_prev_bias = 1;
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all images in R, then alls images in B, ...
		//it also not contain a bias directly in the image
		depth_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2] * net->batch_size;
		image_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2];
		current->input = current->previous->output;
		im2col_prev_bias = 0;
	}
		
	cuda_master_weight_copy(net, (float*)c_param->FP32_filters, c_param->filters, 
		c_param->nb_filters * (c_param->flat_f_size + c_param->TC_padding));
	
	if(net->batch_size <= 2)
		dim_c = 1;
	else
		dim_c = 2;

	if(c_param->nb_filters > 16)
			dim_b = 16;
		else if(c_param->nb_filters > 8)
			dim_b = 8;
		else
			dim_b = 4;
		
	if(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] <= 8)
		dim_a = 4;
	else
		dim_a = 8;
	
	dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
	dim3 numBlocks2(((c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
    	(c_param->prev_depth + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
    	(net->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
	
	net->cu_inst.cu_conv_fcts.im2col_fct<<< numBlocks2, threadsPerBlock2 >>>(c_param->im2col_input,
		current->input, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2], 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * 
		(c_param->flat_f_size + c_param->TC_padding), c_param->stride[0], c_param->stride[1], c_param->stride[2],
		c_param->padding[0], c_param->padding[1], c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		c_param->prev_depth, depth_padding, image_padding, c_param->TC_padding, net->batch_size, 
		c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], c_param->flat_f_size, 
		c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
		c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], im2col_prev_bias, 1);

	//Input X filters matrix multiplication for the all batch
	cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, net->batch_size 
		* (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]), c_param->nb_filters,
		(c_param->flat_f_size + c_param->TC_padding), cu_alpha, c_param->im2col_input, cuda_data_type,
		(c_param->flat_f_size + c_param->TC_padding), c_param->filters, cuda_data_type,  
		(c_param->flat_f_size + c_param->TC_padding), cu_beta, current->output, cuda_data_type,
		net->batch_size * (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]),
		cuda_compute_type, CUBLAS_GEMM_DEFAULT);
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			cu_blocks = (c_param->nb_filters * net->batch_size
				* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2])  + cu_threads - 1) / cu_threads;
			
			cuda_random_vector(c_param->dropout_mask, c_param->nb_filters * net->batch_size
				* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]));
			
			net->cu_inst.cu_conv_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->output, c_param->dropout_mask,
				c_param->nb_filters * net->batch_size * (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), current->dropout_rate);
		}
		else
		{
			cu_blocks = (c_param->nb_filters * net->batch_size
				* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2])  + cu_threads - 1) / cu_threads;
				
			net->cu_inst.cu_conv_fcts.drop_scale_fct<<<cu_blocks, cu_threads>>>(current->output, c_param->dropout_mask,
				c_param->nb_filters * net->batch_size * (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), current->dropout_rate);
		}
	}
	
}


void cuda_backward_conv_layer(layer *current)
{
	int k;
	int depth_padding;
	int back_padding[3];
	int image_padding;
	int flat_f_size;
	int dim_a, dim_b, dim_c;
	void *c_prev_delta_o;
	
	network* net = current->c_network;

	c_param = (conv_param*) current->param;
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
	{
		cu_blocks = (c_param->nb_filters * net->batch_size
			* (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_conv_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(current->delta_o, c_param->dropout_mask, 
			c_param->nb_filters * net->batch_size * (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), current->dropout_rate);
	}
	
	//######################## ERROR PROPAGATION ########################
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//rotate the filters
		//so the new matrix can be considered as flat_filter_size * net->batch_size rows against input_depth
		cu_blocks = (c_param->nb_filters * (c_param->flat_f_size+c_param->TC_padding) + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_conv_fcts.rotate_filter_fct<<< cu_blocks, cu_threads >>>(c_param->filters, 
			c_param->rotated_filters, (c_param->flat_f_size+c_param->TC_padding), 
			c_param->TC_padding, c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2],
			c_param->nb_filters, c_param->nb_filters*(c_param->flat_f_size+c_param->TC_padding));
		
		//In the backward formalism we assume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this means that the meaning of nb_area and prev_size are reversed in the following operations
		depth_padding = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * net->batch_size;
		image_padding = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
		flat_f_size = c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2] * c_param->nb_filters;
		//this flat size remove the bias != c_param->flat_f_size
		
		for(k = 0; k < 3; k++)
		{
			back_padding[k] = c_param->f_size[k] - c_param->padding[k] - 1;
			if(back_padding[k] < 0)
				back_padding[k] = 0;
		}
		
		//Note : having higher dimensions on the left dim3 dim(a,b,c) grants better results
		if(net->batch_size <= 2)
			dim_c = 1;
		else
			dim_c = 2;
		
		if(c_param->nb_filters > 16)
			dim_b = 16;
		else if(c_param->nb_filters > 8)
			dim_b = 8;
		else
			dim_b = 4;
			
		if(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] <= 8)
			dim_a = 4;
		else
			dim_a = 8;
			
		//dim_c = 1; dim_b = 1; dim_a = 32;
		
		dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
		dim3 numBlocks2((c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
			(c_param->nb_filters + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
			(net->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
		
		net->cu_inst.cu_conv_fcts.im2col_fct<<< numBlocks2, threadsPerBlock2 >>>(c_param->im2col_delta_o,
			current->delta_o, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], 
			(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) * flat_f_size, 
			c_param->int_padding[0] + 1,  c_param->int_padding[1] + 1, c_param->int_padding[2] + 1,
			back_padding[0], back_padding[1], back_padding[2],
			c_param->stride[0] - 1 , c_param->stride[1] - 1 , c_param->stride[2] - 1,
			c_param->nb_filters, depth_padding, image_padding, 0, net->batch_size,
			c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], flat_f_size, 
			c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
			c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 0, 0);
					
		if(current->previous->type == DENSE)
			c_prev_delta_o = c_param->temp_delta_o;
		else
			c_prev_delta_o = current->previous->delta_o;

		cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] 
			*net->batch_size, c_param->prev_depth, c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2] 
			*c_param->nb_filters, cu_alpha, c_param->im2col_delta_o, cuda_data_type, c_param->f_size[0] 
			*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, c_param->rotated_filters, cuda_data_type, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, cu_beta, 
			c_prev_delta_o, cuda_data_type, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]
			*net->batch_size, cuda_compute_type, CUBLAS_GEMM_DEFAULT);
		
		if(current->previous->type == DENSE)
		{
			cu_blocks = ((c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] * c_param->prev_depth + 1) 
				* net->batch_size + cu_threads - 1) / cu_threads;
			
			net->cu_inst.cu_dense_fcts.flat_dense_fct<<< cu_blocks, cu_threads >>>(c_param->temp_delta_o, 
				current->previous->delta_o, 0, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2],
				c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] * c_param->prev_depth + 1, c_param->prev_depth, 
				net->batch_size, (c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] * c_param->prev_depth + 1) 
				* net->batch_size);
		}

		current->previous->deriv_activation(current->previous);
		
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		set_cu_learning_rate_and_momentum(net);
		//based on the recovered delta_o provided by the next layer propagation
		//CUBLAS_OP_N ,in this case, is a transpose of regular input (see forward function)
		cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, (c_param->flat_f_size+c_param->TC_padding), c_param->nb_filters, 
			c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * net->batch_size, 
			cu_learning_rate, c_param->im2col_input, cuda_data_type, 
			(c_param->flat_f_size + c_param->TC_padding), current->delta_o, cuda_data_type, 
			c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * net->batch_size,
			cu_momentum, c_param->update, cuda_data_type, 
			(c_param->flat_f_size + c_param->TC_padding), cuda_compute_type, CUBLAS_GEMM_DEFAULT);
			
		cuda_update_weights(net, c_param->FP32_filters, c_param->update, net->learning_rate*net->weight_decay,
			c_param->flat_f_size, (c_param->flat_f_size + c_param->TC_padding) * c_param->nb_filters);
	}
}


void cuda_conv_define(layer *current)
{
	current->forward = cuda_forward_conv_layer;
	current->backprop = cuda_backward_conv_layer;
}




