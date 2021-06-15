

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
static conv_param *c_param;

//public are in prototypes.h

//private
void cuda_forward_conv_layer(layer *current);
void cuda_backward_conv_layer(layer *current);

__global__ void cuda_add_bias_im2col_FP32(float* output, float bias_value, int flat_f_size, int size);
__global__ void cuda_add_bias_im2col_FP16(half* output, float bias_value, int flat_f_size, int size);
__global__ void cuda_rotate_filter_matrix_FP32(float* in, float* out, int nb_rows, int depth_size, int nb_filters_in, int len);
__global__ void cuda_rotate_filter_matrix_FP16(half* in, half* out, int nb_rows, int depth_size, int nb_filters_in, int len);
/*
__global__ void cuda_unroll_conv_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_unroll_conv_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_reroll_delta_o_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void cuda_reroll_delta_o_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
*/
__global__ void im2col_kernel_v4_FP32(float* output, float* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias);
__global__ void im2col_kernel_v4_FP16(half* output, half* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias);
/*__global__ void im2col_kernel_v5_FP16(half* output, half* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias);*/

__global__ void init_block_state_conv(unsigned int seed, curandState_t* states);
__global__ void cuda_dropout_select_conv(int* mask, int size, float drop_rate, curandState_t* states);
__global__ void cuda_dropout_apply_conv_FP32(float* table, int batch_size, int dim, int* mask, int size);
__global__ void cuda_dropout_apply_conv_FP16(half* table, int batch_size, int dim, int* mask, int size);


void cuda_conv_define(layer *current)
{
	current->forward = cuda_forward_conv_layer;
	current->backprop = cuda_backward_conv_layer;
}

void cuda_convert_conv_layer(layer *current)
{
	c_param = (conv_param*)current->param;
	float* temp_tab;

	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			cuda_convert_table(current->c_network, &(c_param->filters), c_param->nb_filters 
				* c_param->flat_f_size);
			break;
		case 1:
			temp_tab = (float*)c_param->filters;
			cudaMalloc(&(c_param->FP32_filters), c_param->nb_filters 
				* c_param->flat_f_size*sizeof(float));
			cudaMemcpy(c_param->FP32_filters, temp_tab, c_param->nb_filters 
				* c_param->flat_f_size * sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			cudaMalloc(&(c_param->filters), c_param->nb_filters 
				* c_param->flat_f_size*sizeof(half));
			break;
	}

	cuda_convert_table(current->c_network, &(c_param->update), c_param->nb_filters 
		* c_param->flat_f_size);
	
	cuda_convert_table(current->c_network, &(c_param->rotated_filters), c_param->nb_filters 
		* (c_param->flat_f_size-1));
	
	cuda_convert_table(current->c_network, &(current->output), c_param->nb_filters 
		* (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d) * current->c_network->batch_size);
	cuda_convert_table(current->c_network, &(current->delta_o), c_param->nb_filters 
		* (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d) * current->c_network->batch_size);
	/*cuda_convert_table(current->c_network, &(c_param->temp_delta_o), c_param->prev_depth 
		* (c_param->prev_size_w * c_param->prev_size_h) * current->c_network->batch_size);*/
	
	cuda_convert_table(current->c_network, &(c_param->im2col_input), 
		(c_param->flat_f_size * c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d) 
		* current->c_network->batch_size);
	cuda_convert_table(current->c_network, &(c_param->im2col_delta_o), 
		(long long int) current->c_network->batch_size * (c_param->prev_size_w*c_param->prev_size_h*c_param->prev_size_d) 
		* (c_param->f_size * c_param->f_size * c_param->f_size * c_param->nb_filters));
	
	if(c_param->dropout_rate > 0.01)
	{
		cuda_convert_table_int(current->c_network, &(c_param->dropout_mask), c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
		cudaMalloc((void**) &c_param->block_state, (c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d)) * sizeof(curandState_t));
		cu_blocks = (c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
		init_block_state_conv<<< cu_blocks, 1>>>(time(NULL),(curandState_t*)c_param->block_state);
	}

	if(current->previous != NULL)
	{
		cuda_convert_table(current->c_network, &(current->input), c_param->prev_depth 
			* (c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d) * current->c_network->batch_size);
	}
}


void cuda_forward_conv_layer(layer *current)
{
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;
	int dim_a, dim_b, dim_c;
	float w_alpha;

	if(current->c_network->length == 0)
		return;
	c_param = (conv_param*) current->param;
	
	
	if(current->previous == NULL)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it as continuous RGB images
		//size in line format
		depth_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d;
		image_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d * c_param->prev_depth;
		current->input = current->c_network->input;
		im2col_prev_bias = 1;
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all image in R, then all image in B, ...
		//it also not contain a bias directly in the image
		depth_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d * current->c_network->batch_size;
		image_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d;
		im2col_prev_bias = 0;
		current->input = current->previous->output;
	}
	
	if(c_param->stride == 0)
	{
		printf("Conv input\n");
		cuda_print_table_4d(current->c_network, current->input, c_param->prev_size_w, c_param->prev_size_h, c_param->prev_depth * current->c_network->batch_size, c_param->prev_size_d, im2col_prev_bias);
	}
	
	if(current->c_network->use_cuda_TC == 1)
		cuda_master_weight_FP32_to_FP16((float*)c_param->FP32_filters, (half*)c_param->filters, 
			c_param->nb_filters * c_param->flat_f_size);
	
	if(current->c_network->batch_size <= 2)
		dim_c = 1;
	//else if(current->c_network->batch_size <= 6)
	//	dim_c = 2;
	else
		dim_c = 2;

	if(c_param->nb_filters <= 8)
		dim_b = 4;
	else
		dim_b = 8;
		
	if(c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d <= 8)
		dim_a = 4;
	else
		dim_a = 8;
	
	//dim_c = 1; dim_b = 1; dim_a = 32;
	
	dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
	//create numBlocks regarding the layer dimensions
	dim3 numBlocks2(((c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d) + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
    	(c_param->prev_depth + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
    	(current->c_network->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			im2col_kernel_v4_FP32<<< numBlocks2, threadsPerBlock2 >>>((float*)c_param->im2col_input,
				(float*)current->input, c_param->prev_size_w*c_param->prev_size_h*c_param->prev_size_d, 
				c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d * c_param->flat_f_size, 
				c_param->stride, c_param->padding, 0, c_param->prev_depth, depth_padding, image_padding,
				current->c_network->batch_size, c_param->f_size, c_param->flat_f_size, 
				c_param->prev_size_w, c_param->prev_size_h, c_param->prev_size_d, c_param->nb_area_w, c_param->nb_area_w, im2col_prev_bias);
			break;
		case 1:
			if(c_param->stride == 0)
				printf("%d %d %d %d %d %d %d %d %d %d %d %d %d \n", c_param->prev_size_w, c_param->prev_size_h, c_param->prev_size_d, 
					c_param->nb_area_w, c_param->nb_area_h, c_param->nb_area_d,
					c_param->flat_f_size, c_param->stride, c_param->prev_depth, 
					c_param->padding, c_param->f_size,
					depth_padding, image_padding);
			im2col_kernel_v4_FP16<<< numBlocks2, threadsPerBlock2 >>>((half*)c_param->im2col_input,
				(half*)current->input, c_param->prev_size_w*c_param->prev_size_h*c_param->prev_size_d, 
				c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d * c_param->flat_f_size, 
				c_param->stride, c_param->padding, 0, c_param->prev_depth, depth_padding, image_padding,
				current->c_network->batch_size, c_param->f_size, c_param->flat_f_size, 
				c_param->prev_size_w, c_param->prev_size_h, c_param->prev_size_d, c_param->nb_area_w, c_param->nb_area_h, im2col_prev_bias);
			break;
	}
	
	if(c_param->stride == 0)
	{
		cudaDeviceSynchronize();
		
		printf("im2col converted\n");
		//cuda_print_table_4d(current->c_network, c_param->im2col_input, c_param->nb_area_w*c_param->nb_area_h*c_param->nb_area_d*current->c_network->batch_size, c_param->flat_f_size, 1, 1, 0);
		cuda_print_table(current->c_network, c_param->im2col_input, c_param->flat_f_size * c_param->nb_area_w*c_param->nb_area_h*c_param->nb_area_d * current->c_network->batch_size, c_param->flat_f_size);
		
		exit(1);
	}

	if(current->c_network->is_inference && current->c_network->inference_drop_mode == AVG_MODEL)
	{
		if(current->previous != NULL)
			w_alpha = (1.0f - ((conv_param*)current->previous->param)->dropout_rate);
		else
			w_alpha = 1.0f;
	}
	else
		w_alpha = 1.0f;

	//Input X filters matrix multiplication for the all batch
	cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h*c_param->nb_area_d), c_param->nb_filters, c_param->flat_f_size, &w_alpha, 
		c_param->im2col_input, cuda_data_type, c_param->flat_f_size, c_param->filters, cuda_data_type,  
		c_param->flat_f_size, &cu_beta, current->output, cuda_data_type, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h*c_param->nb_area_d), cuda_compute_type, CUBLAS_GEMM_DEFAULT);
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	if(c_param->dropout_rate > 0.01 && (!current->c_network->is_inference || current->c_network->inference_drop_mode == MC_MODEL))
	{
		cu_blocks = (c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
		cuda_dropout_select_conv<<<cu_blocks, 1>>>(c_param->dropout_mask, c_param->nb_filters 
			* (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d), c_param->dropout_rate, (curandState_t*) c_param->block_state);	
		
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(current->c_network->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_dropout_apply_conv_FP32<<<numBlocks, threadsPerBlock>>>((float*)current->output, 
					current->c_network->batch_size, (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d),
					c_param->dropout_mask, c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
				break;
			case 1:
				cuda_dropout_apply_conv_FP16<<<numBlocks, threadsPerBlock>>>((half*)current->output, 
					current->c_network->batch_size, (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d), 
					c_param->dropout_mask, c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
				break;
		}
	}
}




void cuda_backward_conv_layer(layer *current)
{
	int depth_padding;
	int back_padding;
	int image_padding;
	int flat_f_size;
	int dim_a, dim_b, dim_c;

	c_param = (conv_param*) current->param;
	
	if(c_param->dropout_rate > 0.01)
	{
		dim3 threadsPerBlock(32, 8);
		dim3 numBlocks((c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d) + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(current->c_network->batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
		
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_dropout_apply_conv_FP32<<<numBlocks, threadsPerBlock>>>((float*)current->delta_o, 
					current->c_network->batch_size, (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d), 
					c_param->dropout_mask, c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
				break;
			case 1:
				cuda_dropout_apply_conv_FP16<<<numBlocks, threadsPerBlock>>>((half*)current->delta_o, 
					current->c_network->batch_size, (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d), 
					c_param->dropout_mask, c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d));
				break;
		}
	}
	
	//######################## ERROR PROPAGATION ########################
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//rotate the filters
		//so the new matrix can be considered as flat_filter_size * current->c_network->batch_size rows against input_depth
		cu_blocks = (c_param->nb_filters * c_param->flat_f_size + cu_threads - 1) / cu_threads;
		
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				cuda_rotate_filter_matrix_FP32<<< cu_blocks, cu_threads >>>((float*)c_param->filters, 
					(float*)c_param->rotated_filters, c_param->flat_f_size, c_param->f_size * 
					c_param->f_size * c_param->f_size, c_param->nb_filters, c_param->nb_filters*c_param->flat_f_size);
				break;
			case 1:
				cuda_rotate_filter_matrix_FP16<<< cu_blocks, cu_threads >>>((half*)c_param->filters,
					(half*)c_param->rotated_filters, c_param->flat_f_size, c_param->f_size * 
					c_param->f_size * c_param->f_size, c_param->nb_filters, c_param->nb_filters*c_param->flat_f_size);
				break;
		}
		
		//In the backward formalism we asume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this means that the meaning of nb_area_w/h and prev_size_w/h are reversed in the following operation
		
		depth_padding = c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d * current->c_network->batch_size;
		image_padding = c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d;
		flat_f_size = c_param->f_size * c_param->f_size * c_param->f_size * c_param->nb_filters;
		
		back_padding =  c_param->f_size -  c_param->padding - 1;
		if(back_padding < 0)
			back_padding = 0;
		
		//Note : having higher dimensions on the left dim3 dim(a,b,c) grants better results 
		// (profiling shows reduction of compute time near to ~ 17% (on Modified LeNet 5 - MNIST))
		//limit is L2 cache usage, having dim3 a < (16,1,1) allows to maximse it on P2000
		if(current->c_network->batch_size <= 2)
			dim_c = 1;
		else
			dim_c = 2;
		
		if(c_param->nb_filters <= 8)
			dim_b = 4;
		else
			dim_b = 16;
			
		if(c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d <= 8)
			dim_a = 4;
		else
			dim_a = 16;
			
		//dim_c = 1; dim_b = 1; dim_a = 32;
		
		dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
		//create numBlocks regarding the layer dimensions
		dim3 numBlocks2((c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
			(c_param->nb_filters + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
			(current->c_network->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
		
		switch(current->c_network->use_cuda_TC)
		{
			default:
			case 0:
				im2col_kernel_v4_FP32<<< numBlocks2, threadsPerBlock2 >>>((float*)c_param->im2col_delta_o, 
					(float*)current->delta_o, c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d, 
					(c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d) * flat_f_size, 1, back_padding, 
					c_param->stride - 1 , c_param->nb_filters, depth_padding, image_padding, 
					current->c_network->batch_size, c_param->f_size, flat_f_size, c_param->nb_area_w,
					c_param->nb_area_h, c_param->nb_area_d, c_param->prev_size_w, c_param->prev_size_h, 0);
				break;
			case 1:
				im2col_kernel_v4_FP16<<< numBlocks2, threadsPerBlock2>>>((half*)c_param->im2col_delta_o,
					(half*)current->delta_o, c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d, 
					(c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d) * flat_f_size, 1, back_padding,
					c_param->stride - 1 , c_param->nb_filters, depth_padding, image_padding,
					current->c_network->batch_size, c_param->f_size, flat_f_size, c_param->nb_area_w, 
					c_param->nb_area_h, c_param->nb_area_d, c_param->prev_size_w, c_param->prev_size_h, 0);
				break;
		}

		cublasGemmEx(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d 
			* current->c_network->batch_size, c_param->prev_depth, c_param->f_size * c_param->f_size * c_param->f_size 
			* c_param->nb_filters, &cu_alpha, c_param->im2col_delta_o, cuda_data_type, c_param->f_size 
			* c_param->f_size * c_param->f_size * c_param->nb_filters, c_param->rotated_filters, cuda_data_type, 
			c_param->f_size * c_param->f_size * c_param->f_size  * c_param->nb_filters, &cu_beta, current->previous->delta_o,
			cuda_data_type, c_param->prev_size_w * c_param->prev_size_h * c_param->prev_size_d * current->c_network->batch_size,
			cuda_compute_type, CUBLAS_GEMM_DEFAULT);

		//update gradiant regarding the previous layer activation function
		//WARNING : ONLY WORK IF PREVIOUS LAYER IS A CONV AS OUTPUT AND DELTA_O SHARE THE SAME DATA ORDER
		current->previous->deriv_activation(current->previous);

	}
	
	//########################  WEIGHTS UPDATE   ########################
	
	if(!current->frozen)
	{
		//based on the recovered delta_o provided by the next layer propagation
		//CUBLAS_OP_N ,in this case, is a transpose of regular input (see forward function)
		cublasGemmEx(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, c_param->flat_f_size, c_param->nb_filters, 
			c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d * current->c_network->batch_size, 
			&current->c_network->learning_rate, c_param->im2col_input, cuda_data_type, c_param->flat_f_size, 
			current->delta_o, cuda_data_type, c_param->nb_area_w * c_param->nb_area_h * c_param->nb_area_d
			* current->c_network->batch_size, &current->c_network->momentum, c_param->update, cuda_data_type, 
			c_param->flat_f_size, cuda_compute_type, CUBLAS_GEMM_DEFAULT);
		
		switch(current->c_network->use_cuda_TC)
		{
			case 0:
				cuda_update_weights(current->c_network, c_param->filters, c_param->update, 
					c_param->flat_f_size * c_param->nb_filters);
				break;
			case 1:
				cuda_update_weights(current->c_network, c_param->FP32_filters, c_param->update, 
					c_param->flat_f_size * c_param->nb_filters);
				break;
		}
	}

}



//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only 1 load per element of the original image.
//VERSION 4.1

__global__ void im2col_kernel_v4_FP32(float* output, float* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int channel, int channel_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias)
{
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	int c = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.z*blockDim.z + threadIdx.z;
	
	half local_pix;
	
	int w, h, d, x, y, z;
	int pos_w_filter, pos_h_filter, pos_d_filter;
	int loc;
	
	if( i < batch_size)
	{
		input += i*(image_padding + bias);
		output += i*(flat_image_size);
		
		if(c < channel)
		{
			input += c * channel_padding;
			output += c * f_size*f_size*f_size;
			if(p < image_size)
			{
				local_pix = input[p];
			
				d = (p / (w_size*h_size))*(1 + internal_padding) + padding;
				h = (p % (w_size*h_size) % w_size)*(1 + internal_padding) + padding;
				w = (p % (w_size*h_size) / w_size)*(1 + internal_padding) + padding;
				
				for(z = d/stride; (d-z*stride < f_size); z -=1)
				{
					if((z < 0) || (z*stride > d_size*(1 + internal_padding) + 2*padding - f_size))
						continue;
					pos_d_filter = d-z*stride;
					for(x = w/stride; (w-x*stride < f_size) /*&& (x >= 0)*/; x -= 1)
					{
						if((x < 0) || (x*stride > w_size*(1 + internal_padding) + 2*padding - f_size))
							continue;
						pos_w_filter = w-x*stride;
						for(y = h/stride; (h-y*stride < f_size) /*&& (y >= 0)*/; y -= 1)
						{
							if((y < 0) || (y*stride > h_size*(1 + internal_padding) + 2*padding - f_size))
								continue;
							pos_h_filter = h-y*stride;
							loc = z*nb_area_w*nb_area_h*flat_f_size + y*nb_area_w*flat_f_size + x*flat_f_size + pos_w_filter + pos_h_filter*f_size + pos_d_filter*f_size*f_size;
							if(loc >= 0 && loc < flat_image_size)
								output[loc] = local_pix;
						}
					}
				}
			}
		}
	}
}

__global__ void im2col_kernel_v4_FP16(half* output, half* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int channel, int channel_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias)
{
	int p = blockIdx.x*blockDim.x + threadIdx.x;
	int c = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.z*blockDim.z + threadIdx.z;
	
	half local_pix;
	
	int w, h, d, x, y, z;
	int pos_w_filter, pos_h_filter, pos_d_filter;
	int loc;
	
	if( i < batch_size)
	{
		input += i*(image_padding + bias);
		output += i*(flat_image_size);
		
		if(c < channel)
		{
			input += c * channel_padding;
			output += c * f_size*f_size*f_size;
			if(p < image_size)
			{
				local_pix = input[p];
			
				d = (p / (w_size*h_size))*(1 + internal_padding) + padding;
				h = (p % (w_size*h_size) % w_size)*(1 + internal_padding) + padding;
				w = (p % (w_size*h_size) / w_size)*(1 + internal_padding) + padding;
				
				for(z = d/stride; (d-z*stride < f_size); z -=1)
				{
					if((z < 0) || (z*stride > d_size*(1 + internal_padding) + 2*padding - f_size))
						continue;
					pos_d_filter = d-z*stride;
					for(x = w/stride; (w-x*stride < f_size) /*&& (x >= 0)*/; x -= 1)
					{
						if((x < 0) || (x*stride > w_size*(1 + internal_padding) + 2*padding - f_size))
							continue;
						pos_w_filter = w-x*stride;
						for(y = h/stride; (h-y*stride < f_size) /*&& (y >= 0)*/; y -= 1)
						{
							if((y < 0) || (y*stride > h_size*(1 + internal_padding) + 2*padding - f_size))
								continue;
							pos_h_filter = h-y*stride;
							loc = z*nb_area_w*nb_area_h*flat_f_size + y*nb_area_w*flat_f_size + x*flat_f_size + pos_w_filter + pos_h_filter*f_size + pos_d_filter*f_size*f_size;
							if(loc >= 0 && loc < flat_image_size)
								output[loc] = local_pix;
						}
					}
				}
			}
		}
	}
}

//Attempt to reduce compute charge of kernel -> no effect on P2000, must check on less memory bound GPU like V100
/*
__global__ void im2col_kernel_v5_FP16(half* output, half* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias)
{
	int z = blockIdx.x*blockDim.x + threadIdx.x;
	int d = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.z*blockDim.z + threadIdx.z;
	
	half local_pix;
	
	int w, h, x, y;
	int pos_w_filter, pos_h_filter;
	int loc_a, loc_b;
	
	if( i < batch_size)
	{	
		if(d < depth)
		{
			input += d * depth_padding + i*(image_padding + bias);
			output += d * f_size*f_size + i*(flat_image_size);
			if(z < image_size)
			{
				local_pix = input[z];
			
				w = (z % w_size)*(1 + internal_padding) + padding;
				h = (z / w_size)*(1 + internal_padding) + padding;
				 
				for(y = h/stride; (h-y*stride < f_size) && (y >= 0); y -= 1)
				{
					//pos_h_filter = (h-y*stride)*f_size;
					loc_a = (h-y*stride)*f_size + y*nb_area_w*flat_f_size;
					for(x = w/stride; (w-x*stride < f_size) && (x >= 0); x -= 1)
					{
						//pos_w_filter = w-x*stride;
						loc_b = loc_a + x*flat_f_size + w-x*stride;
						if(loc_b >= 0 && loc_b < flat_image_size)
							output[loc_b] = local_pix;
					}
				}
			}
		}
	}
}
*/


__global__ void cuda_rotate_filter_matrix_FP32(float* in, float* out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int x, y, depth_id;
	
	if(i < len)
	{
		//#####################################
		//Rotate and move the filters
		x = i / nb_rows;
		y = i % nb_rows;
		
		if(y < nb_rows-1) //remove the weights of the bias nodes
		{
			depth_id = y / depth_size;
			
			out[depth_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];
		}
		
	}
	
}


__global__ void cuda_rotate_filter_matrix_FP16(half* in, half* out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int x, y, depth_id;
	
	if(i < len)
	{
		//#####################################
		//Rotate and move the filters
		x = i / nb_rows;
		y = i % nb_rows;
		
		if(y < nb_rows-1) //remove the weights of the bias nodes
		{
			depth_id = y / depth_size;
			
			out[depth_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];
		}
		
	}
	
}

/*
__global__ void cuda_unroll_conv_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;

		out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


__global__ void cuda_unroll_conv_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;

		out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


__global__ void cuda_reroll_delta_o_FP32(float* in, float* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}


__global__ void cuda_reroll_delta_o_FP16(half* in, half* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}
*/

__global__ void init_block_state_conv(unsigned int seed,  curandState_t* states)
{
	curand_init((seed << 20) + blockIdx.x, /* the seed can be the same for each core, here we pass the time in from the CPU */
              0, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! 
			     Currently use an alternative definition with Id adjunct to seed*/
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}


__global__ void cuda_dropout_select_conv(int* mask, int size, float drop_rate, curandState_t* states)
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

__global__ void cuda_dropout_apply_conv_FP32(float* table, int batch_size, int dim, int* mask, int size)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	
	int c_depth = j / dim;
	int current_id = j % dim;
	int offset = dim*batch_size;

	if(i < batch_size && j < size)
	{
		table[i*dim + c_depth*offset + current_id] *= mask[j];
	}
}

__global__ void cuda_dropout_apply_conv_FP16(half* table, int batch_size, int dim, int* mask, int size)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	int c_depth = j / dim;
        int current_id = j % dim;
        int offset = dim*batch_size;

        if(i < batch_size && j < size)
        {
                table[i*dim + c_depth*offset + current_id] *= mask[j];
        }
}






