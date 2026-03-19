
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
static conv_param *c_param;

// Public are in "prototypes.h"

// Private prototypes
void cuda_forward_conv_layer(layer *current);
void cuda_backward_conv_layer(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead


//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only one load per element of the original image.
//Version 6+ add depth-wise convolution support. We note that using im2col + matmul il likely less efficient than a dedicated kernel
//when the number of filter per input channel is one. Such dedicaed kernel might be implemented at some point.
//VERSION 6.0
#define im2col_kernel(name, type) 																												\
__global__ void im2col_kernel_##name																											\
	(void* i_output, void* i_input, 																											\
	int stride_w, int stride_h, int stride_d, 																									\
	int padding_w, int padding_h, int padding_d, 																								\
	int internal_padding_w, int internal_padding_h, int internal_padding_d, 																	\
	int f_size_w, int f_size_h, int f_size_d,																									\
	size_t w_size, size_t h_size, size_t d_size,																								\
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d,																						\
	size_t in_nb_channels, size_t in_group_size, size_t in_image_size, size_t in_image_offset, size_t in_channel_offset,  						\
	size_t out_image_offset, size_t out_group_offset, int TC_padding,																			\
	int batch_size, int bias_out) 																												\
{																																				\
	size_t p = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t c = blockIdx.y*blockDim.y + threadIdx.y;																								\
	size_t i = blockIdx.z*blockDim.z + threadIdx.z;																								\
																																				\
	type *output = (type*) i_output;																											\
	type *input  = (type*) i_input;																												\
	type local_pix;																																\
																																				\
	long long int w, h, d, x, y, z;																												\
	long long int pos_w_filter, pos_h_filter, pos_d_filter;																						\
	size_t loc, spatial_f_size, flat_f_size;																									\
																																				\
	spatial_f_size = f_size_w * f_size_h * f_size_d;																							\
	flat_f_size = spatial_f_size * in_group_size + bias_out + TC_padding;																		\
																																				\
	if(i < batch_size)																															\
	{																																			\
		input += i * in_image_offset;																											\
		output += i * out_image_offset;																											\
																																				\
		if(c < in_nb_channels)																													\
		{																																		\
			input += c * in_channel_offset;																										\
			output += (c/in_group_size) * out_group_offset;																						\
			output += (c%in_group_size) * spatial_f_size;																						\
																																				\
			if(p < in_image_size)																												\
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
					if((z < 0) || (z > (d_size + (d_size-1)*internal_padding_d + 2*padding_d - f_size_d)/stride_d))								\
						continue;																												\
					for(y = h/stride_h; (h-y*stride_h < f_size_h); y -= 1)																		\
					{																															\
						pos_h_filter = h-y*stride_h;																							\
						if((y < 0) || (y > (h_size + (h_size-1)*internal_padding_h + 2*padding_h - f_size_h)/stride_h))							\
							continue;																											\
						for(x = w/stride_w; (w-x*stride_w < f_size_w); x -= 1)																	\
						{																														\
							pos_w_filter = w-x*stride_w;																						\
							if((x < 0) || (x > (w_size + (w_size-1)*internal_padding_w + 2*padding_w - f_size_w)/stride_w))						\
								continue;																										\
							loc = (z*(size_t)nb_area_w*nb_area_h + y*nb_area_w + x)*flat_f_size													\
								 + pos_w_filter + pos_h_filter*f_size_w + pos_d_filter*f_size_w*f_size_h;										\
							if((bias_out && loc%flat_f_size >= flat_f_size-bias_out-TC_padding))												\
								continue;																										\
							if(loc < out_image_offset)	/* loc is > 0 by construction */														\
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
	(void* i_in, void* i_out, size_t nb_rows, size_t spatial_f_size, size_t out_group_size, int TC_padding, size_t len)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t x, y;																																\
																																				\
	type* in  = (type*) i_in;																													\
	type* out = (type*) i_out;																													\
																																				\
	if(i >= len)																																\
		return;																																	\
																																				\
	/*Rotate and move the filters*/																												\
	x = i / nb_rows;																															\
	y = i % nb_rows;																															\
																																				\
	/*remove the weights of the bias nodes*/																									\
	if(y >= nb_rows-1-TC_padding) 																												\
		return;																																	\
																																				\
	out += (x/out_group_size) * (nb_rows-1-TC_padding) * out_group_size;   																		\
	out += (x%out_group_size) * spatial_f_size;																									\
	out += (y/spatial_f_size) * spatial_f_size * out_group_size;																				\
	out[spatial_f_size - 1 - y%spatial_f_size] = in[x*nb_rows+y];																				\
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



im2col_kernel(FP32, float);
cuda_rotate_filter_matrix(FP32, float); 
cuda_dropout_apply_conv(FP32, float);
cuda_dropout_scale_conv(FP32, float);

#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
im2col_kernel(FP16, half);
cuda_rotate_filter_matrix(FP16, half);
cuda_dropout_apply_conv(FP16, half);
cuda_dropout_scale_conv(FP16, half);
#endif

#if defined (GEN_AMPERE)
im2col_kernel(BF16, nv_bfloat16);
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
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_FP32;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_FP32;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_FP32;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_FP32;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_FP16;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_FP16;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_FP16;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_FP16;
			#else
			printf("\n ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_conv_fcts.im2col_fct = im2col_kernel_BF16;
			net->cu_inst.cu_conv_fcts.rotate_filter_fct = cuda_rotate_filter_matrix_BF16;
			net->cu_inst.cu_conv_fcts.drop_apply_fct = cuda_dropout_apply_conv_BF16;
			net->cu_inst.cu_conv_fcts.drop_scale_fct = cuda_dropout_scale_conv_BF16;
			#else
			printf("\n ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


size_t cuda_convert_conv_layer(layer *current)
{
	c_param = (conv_param*)current->param;
	size_t vram_approx = 0, nb_groups, batch_size;
	size_t subdim_a, subdim_b;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	
	#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
	float* temp_tab;
	#endif

	network* net = current->c_network;
	nb_groups = c_param->nb_groups;
	batch_size = net->batch_size;
	
	spatial_f_size   =     c_param->f_size[0] *     c_param->f_size[1] *     c_param->f_size[2];
	nb_regions_in    =   current->prev_dim[0] *   current->prev_dim[1] *   current->prev_dim[2];
	nb_regions_out   = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
	nb_filters       = current->output_dim[3];
	prev_nb_channels =   current->prev_dim[3];
	
	chan_per_group_in  = prev_nb_channels / nb_groups;
	chan_per_group_out = nb_filters       / nb_groups;

	//######## Input related data arrays  ########
	
	subdim_a = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	subdim_b = batch_size * nb_regions_out;
	
	vram_approx += cuda_convert_table(net, &(c_param->im2col_input), nb_groups * subdim_a * subdim_b, 0);
	
	//#############################################
	
	//######## Weights related data arrays ########
	
	subdim_a = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	subdim_b = chan_per_group_out;
	
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			vram_approx += cuda_convert_table(net, &(current->weights), nb_groups * subdim_a * subdim_b, 0);
			current->FP32_weights = (float*)current->weights;
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
			temp_tab = (float*)current->weights;
			cudaMalloc(&(current->FP32_weights), nb_groups * subdim_a * subdim_b * sizeof(float));
			vram_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
			cudaMemcpy(current->FP32_weights, temp_tab, nb_groups * subdim_a * subdim_b * sizeof(float), cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(current->weights), nb_groups * subdim_a * subdim_b * sizeof(half));
			vram_approx += nb_groups * subdim_a * subdim_b * sizeof(half);
			#endif
			break;
		
		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			temp_tab = (float*)current->weights;
			cudaMalloc(&(current->FP32_weights), nb_groups * subdim_a * subdim_b * sizeof(float));
			vram_approx += nb_groups * subdim_a * subdim_b * sizeof(float);
			cudaMemcpy(current->FP32_weights, temp_tab, nb_groups * subdim_a * subdim_b * sizeof(float), cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(current->weights), nb_groups * subdim_a * subdim_b * sizeof(nv_bfloat16));
			vram_approx += nb_groups * subdim_a * subdim_b * sizeof(nv_bfloat16);
			#endif
			break;
	}
	
	if(!net->inference_only)
	{
		if(net->use_wema)
			vram_approx += cuda_convert_table_FP32((void**)&(current->ema_weights), nb_groups * subdim_a * subdim_b, 0);
		
		vram_approx += cuda_convert_table(net, &(current->gradient), nb_groups * subdim_a * subdim_b, 0);
		
		vram_approx += cuda_convert_optimizer_var(current, nb_groups * subdim_a * subdim_b);
	}
	
	//#############################################
	
	//######## Output related data arrays #########
	
	subdim_a = batch_size * nb_regions_out;
	subdim_b = chan_per_group_out;
	
	vram_approx += cuda_convert_table(net, &(current->output), nb_groups * subdim_a * subdim_b, 0);
	
	if(current->dropout_rate > 0.01f)
		vram_approx += cuda_convert_table_FP32((void**)&(current->dropout_mask), nb_groups * subdim_a * subdim_b, 0);
	
	if(!net->inference_only)
		vram_approx += cuda_convert_table(net, &(current->delta_o), nb_groups * subdim_a * subdim_b, 0);
	
	//#############################################
	
	//######## Backprop related data arrays #######
	
	if(!net->inference_only)
	{
		subdim_a = spatial_f_size * chan_per_group_out;
		subdim_b = batch_size * nb_regions_in;
		
		vram_approx += cuda_convert_table(net, &(c_param->im2col_delta_o), nb_groups * subdim_a * subdim_b, 0);
		
		subdim_a = spatial_f_size * chan_per_group_out;
		subdim_b= chan_per_group_in;
		
		vram_approx += cuda_convert_table(net, &(c_param->rotated_filters), nb_groups * subdim_a * subdim_b, 0);
	
		if(current->previous != NULL && current->previous->output_type == FLAT)
		{
			subdim_a = prev_nb_channels * nb_regions_in;
			subdim_b = batch_size;
			
			vram_approx += cuda_convert_table(net, &(c_param->temp_delta_o), subdim_a * subdim_b, 0);
		}
	}
	
	//#############################################
	
	return vram_approx;
}


void cuda_free_conv(layer *current)
{
	c_param = (conv_param*)current->param;
	
	cudaFree(current->weights);
	if(current->c_network->cu_inst.use_cuda_TC != FP32C_FP32A && current->c_network->cu_inst.use_cuda_TC != TF32C_FP32A)
		cudaFree(current->FP32_weights);
	
	cudaFree(current->output);
	cudaFree(c_param->im2col_input);
	if(current->dropout_rate > 0.01f)
		cudaFree(current->dropout_mask);
	if(!current->c_network->inference_only)
	{
		if(current->c_network->use_wema)
			cudaFree(current->ema_weights);
		cudaFree(current->gradient);
		cudaFree(c_param->rotated_filters);
		cudaFree(current->delta_o);
		if(current->previous != NULL && current->previous->output_type == FLAT)
			cudaFree(c_param->temp_delta_o);
		cudaFree(c_param->im2col_delta_o);
		
		cuda_free_optimizer_var(current);
	}
}

void cuda_forward_conv_layer(layer *current)
{
	int dim_a, dim_b, dim_c;
	size_t subdim_M, subdim_N, subdim_K, nb_groups, batch_size;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	void *l_weights;
	
	network* net = current->c_network;
	c_param = (conv_param*) current->param;
	batch_size = net->batch_size;
	nb_groups = c_param->nb_groups;
	
	spatial_f_size   =     c_param->f_size[0] *     c_param->f_size[1] *     c_param->f_size[2];
	nb_regions_in    =   current->prev_dim[0] *   current->prev_dim[1] *   current->prev_dim[2];
	nb_regions_out   = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
	nb_filters       = current->output_dim[3];
	prev_nb_channels =   current->prev_dim[3];
	
	chan_per_group_in  = prev_nb_channels / nb_groups;
	chan_per_group_out = nb_filters       / nb_groups;
	
	if(current->previous == NULL || (current->previous != NULL && current->previous->output_type == FLAT))
	{
		//If previous is input, each images is stored as continuous flat arrays with all R pixels, all G pixel, all B pixels + input bias
		//Different images from the batch are append on after the other
		in_image_offset   = nb_regions_in * prev_nb_channels + 1;
		in_channel_offset = nb_regions_in;
		if(current->previous == NULL)
			current->input = net->input;
		else
			current->input = current->previous->output;
	}
	else
	{
		//If previous layer is SPATIAL then the format is all images R flat, all images G flat, all images B flat. 
		in_image_offset   = nb_regions_in;
		in_channel_offset = nb_regions_in * batch_size;
		current->input    = current->previous->output;
	}
	
	
	//########## Subproblem dimensions ##########
	
	subdim_M = batch_size * nb_regions_out;
	subdim_N = chan_per_group_out;
	subdim_K = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	
	//########## Preparing Im2col(K,M) ##########
	
	if(batch_size <= 2)  dim_c = 1;  else dim_c = 2;
	if(prev_nb_channels > 16) dim_b = 16; else if(prev_nb_channels > 8) dim_b = 8; else dim_b = 4;
	if(nb_regions_in <= 8)    dim_a = 4;  else dim_a = 8;
	
	dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
	dim3 numBlocks2((nb_regions_in + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
    	(prev_nb_channels + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
    	(batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
	
	net->cu_inst.cu_conv_fcts.im2col_fct<<< numBlocks2, threadsPerBlock2 >>>(
		c_param->im2col_input, current->input, 
		     c_param->stride[0],      c_param->stride[1],      c_param->stride[2],
		    c_param->padding[0],     c_param->padding[1],     c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		     c_param->f_size[0],      c_param->f_size[1],      c_param->f_size[2],
		   current->prev_dim[0],    current->prev_dim[1],    current->prev_dim[2],
		 current->output_dim[0],  current->output_dim[1],  current->output_dim[2],
		prev_nb_channels, chan_per_group_in, nb_regions_in, in_image_offset, in_channel_offset,
		subdim_K*nb_regions_out, subdim_K*subdim_M, c_param->TC_padding, batch_size, 1);
	

	//######### Preparing Weights(K,N) ##########
	
	if(net->is_inference == 1 && net->use_wema)
	{
		if(current->FP32_weights == current->weights) //Equivalent to test if mixed precision is off or FP32C_FP32A
			l_weights = (void*) current->ema_weights;
		else
		{
			cuda_master_weight_copy(net, (float*)current->ema_weights, current->weights, nb_groups * subdim_K * subdim_N);
			l_weights = current->weights;
		}	
	}
	else
	{
		if(current->FP32_weights == current->weights)
			l_weights = (void*) current->FP32_weights;
		else
		{
			cuda_master_weight_copy(net, (float*)current->FP32_weights, current->weights, nb_groups * subdim_K * subdim_N);
			l_weights = current->weights;
		}
	}
	
	
	//####### Im2col_T(M,K) x Weights(K,N) #######
	
	cublasGemmStridedBatchedEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, subdim_M, subdim_N, subdim_K, cu_alpha, 
		/*A*/c_param->im2col_input, cuda_data_type, /*ldA*/subdim_K, /*strideA*/subdim_K*subdim_M,
		/*B*/l_weights            , cuda_data_type, /*ldB*/subdim_K, /*strideB*/subdim_K*subdim_N, cu_beta,
		/*C*/current->output      , cuda_data_type, /*ldC*/subdim_M, /*strideC*/subdim_M*subdim_N,
		nb_groups, cuda_compute_type, CUBLAS_GEMM_DEFAULT);
	
	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
			//here current->a_size = nb_groups * subdim_M * sumbdim_N
			cuda_random_vector(current->dropout_mask, current->a_size);
			
			net->cu_inst.cu_conv_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(
				current->output, current->dropout_mask, current->a_size, current->dropout_rate);
		}
		else
		{
			cu_blocks = (current->a_size  + cu_threads - 1) / cu_threads;

			net->cu_inst.cu_conv_fcts.drop_scale_fct<<<cu_blocks, cu_threads>>>(
				current->output, current->dropout_mask, current->a_size, current->dropout_rate);
		}
	}
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	if(!net->inference_only)
		net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct(current->delta_o, 0, current->a_size);
}


void cuda_backward_conv_layer(layer *current)
{
	size_t k;
	int dim_a, dim_b, dim_c;
	size_t subdim_M, subdim_N, subdim_K, nb_groups, batch_size;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	int back_padding[3];
	void *c_prev_delta_o;

	network* net = current->c_network;
	c_param = (conv_param*) current->param;
	nb_groups = c_param->nb_groups;
	batch_size = net->batch_size;
	
	spatial_f_size   =     c_param->f_size[0] *     c_param->f_size[1] *     c_param->f_size[2];
	nb_regions_in    =   current->prev_dim[0] *   current->prev_dim[1] *   current->prev_dim[2];
	nb_regions_out   = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
	nb_filters       = current->output_dim[3];
	prev_nb_channels =   current->prev_dim[3];
	
	chan_per_group_in  = prev_nb_channels / nb_groups;
	chan_per_group_out = nb_filters       / nb_groups;
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->dropout_rate > 0.01f && 
		(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
	{
		cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_conv_fcts.drop_apply_fct<<<cu_blocks, cu_threads>>>(
			current->delta_o, current->dropout_mask, current->a_size, current->dropout_rate);
	}
	
	//######################## ERROR PROPAGATION ########################
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//Set prev_delta_o pointer depending on previous layer type
		if(current->previous->output_type == FLAT)
			c_prev_delta_o = c_param->temp_delta_o;
		else
			c_prev_delta_o = current->previous->delta_o;
	
		//########## Preparing dw_rot_weights(K,N) ##########
		//dimensions from regular weight matrix
		subdim_N = chan_per_group_out;
		subdim_K = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
	
		cu_blocks = (nb_groups*subdim_N*subdim_K + cu_threads - 1) / cu_threads;
		
		net->cu_inst.cu_conv_fcts.rotate_filter_fct<<< cu_blocks, cu_threads >>>(
			current->weights, c_param->rotated_filters, subdim_K, 
			spatial_f_size, chan_per_group_out, c_param->TC_padding, 
			nb_groups*subdim_N*subdim_K);
		
		//########## Subproblem dimensions ##########
		
		subdim_M = batch_size * nb_regions_in;
		subdim_N = chan_per_group_in;
		subdim_K = spatial_f_size * chan_per_group_out;
		
		//########## Preparing Im2col_delta_o(K,M) ##########
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//therefore "in" and "out" variables are inverted regarding im2col function arguments
		in_image_offset   = nb_regions_out;
		in_channel_offset = nb_regions_out * batch_size;
		
		for(k = 0; k < 3; k++)
		{
			back_padding[k] = c_param->f_size[k] - c_param->padding[k] - 1;
			if(back_padding[k] < 0)
				back_padding[k] = 0;
		}
		
		if(batch_size <= 2) dim_c = 1; else dim_c = 2;
		if(nb_filters > 16) dim_b = 16; else if(nb_filters > 8) dim_b = 8; else dim_b = 4;
		if(nb_regions_out <= 8) dim_a = 4; else dim_a = 8;
		
		dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
		dim3 numBlocks2((nb_regions_out + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
			(nb_filters + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
			(batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
		
		net->cu_inst.cu_conv_fcts.im2col_fct<<< numBlocks2, threadsPerBlock2 >>>(
			c_param->im2col_delta_o, current->delta_o, 
			/*stride*/ c_param->int_padding[0] + 1, c_param->int_padding[1] + 1, c_param->int_padding[2] + 1,
			/*padding*/        back_padding[0]    ,         back_padding[1]    ,         back_padding[2]    ,
			/*int_padding*/ c_param->stride[0] - 1,      c_param->stride[1] - 1,      c_param->stride[2] - 1,
			                c_param->f_size[0]    ,      c_param->f_size[1]    ,      c_param->f_size[2]    ,
			/*in_size*/ current->output_dim[0]    ,  current->output_dim[1]    ,  current->output_dim[2]    ,
			/*out_size*/  current->prev_dim[0]    ,    current->prev_dim[1]    ,    current->prev_dim[2]    ,
			nb_filters, chan_per_group_out, nb_regions_out, in_image_offset, in_channel_offset,
			subdim_K*nb_regions_in, subdim_K*subdim_M, 0, batch_size, 0);
		
		
		//####### Im2col_delta_o_T(M,K) x dw_rot_weights(K,N) #######
		
		cublasGemmStridedBatchedEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, subdim_M, subdim_N, subdim_K, cu_alpha, 
			/*A*/c_param->im2col_delta_o , cuda_data_type, /*ldA*/subdim_K, /*strideA*/subdim_K*subdim_M,
			/*B*/c_param->rotated_filters, cuda_data_type, /*ldB*/subdim_K, /*strideB*/subdim_K*subdim_N, cu_alpha,
			/*C*/c_prev_delta_o          , cuda_data_type, /*ldC*/subdim_M, /*strideC*/subdim_M*subdim_N,
			nb_groups, cuda_compute_type , CUBLAS_GEMM_DEFAULT);
		
		if(current->previous->output_type == FLAT)
		{
			cu_blocks = ((nb_regions_in * prev_nb_channels + 1) * batch_size + cu_threads - 1) / cu_threads;
			
			net->cu_inst.cu_dense_fcts.flat_dense_fct<<< cu_blocks, cu_threads >>>(
				c_param->temp_delta_o, current->previous->delta_o, 0, nb_regions_in,
				(nb_regions_in * prev_nb_channels + 1), prev_nb_channels, batch_size, 
				(nb_regions_in * prev_nb_channels + 1) * batch_size);
		}
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		//########## Subproblem dimensions ##########
		
		subdim_M = spatial_f_size * chan_per_group_in + 1 + c_param->TC_padding;
		subdim_N = chan_per_group_out;
		subdim_K = batch_size * nb_regions_out;
		
		//####### Im2col(M,K) x delta_o(K,N) #######
		
		cublasGemmStridedBatchedEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, subdim_M, subdim_N, subdim_K, cu_alpha, 
			/*A*/c_param->im2col_input, cuda_data_type, /*ldA*/subdim_M, /*strideA*/subdim_K*subdim_M,
			/*B*/current->delta_o     , cuda_data_type, /*ldB*/subdim_K, /*strideB*/subdim_K*subdim_N, cu_beta,
			/*C*/current->gradient    , cuda_data_type, /*ldC*/subdim_M, /*strideC*/subdim_M*subdim_N,
			nb_groups, cuda_compute_type , CUBLAS_GEMM_DEFAULT);
		
		net->optim_update_fct_gpu(current, subdim_M - c_param->TC_padding, subdim_M, nb_groups*subdim_M*subdim_N);
	
		if(current->wema_replace_signal > 0)
		{
			cudaMemcpy(current->FP32_weights, current->ema_weights, nb_groups*subdim_M*subdim_N*sizeof(float), cudaMemcpyDeviceToDevice);
			current->wema_replace_signal = 0;
		}
	}
}


void cuda_conv_define(layer *current)
{
	current->forward = cuda_forward_conv_layer;
	current->backprop = cuda_backward_conv_layer;
}




