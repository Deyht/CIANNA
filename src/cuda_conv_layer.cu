

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






#include "prototypes.h"

static int cu_blocks;
static conv_param *c_param;

//public are in prototypes.h

//private
void cuda_forward_conv_layer(layer *current);
void cuda_backward_conv_layer(layer *current);

__global__ void add_bias_im2col(real* output, real bias_value, int flat_f_size, int size);
__global__ void rotate_filter_matrix(real* in, real* out, int nb_rows, int depth_size, int nb_filters_in, int len);
__global__ void unroll_conv(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void reroll_delta_o(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void im2col_kernel_v3(real* output, real* input, int image_size, int flat_image_size, int stride, int padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias);



void cuda_conv_define(layer *current)
{
	current->forward = cuda_forward_conv_layer;
	current->backprop = cuda_backward_conv_layer;
}

void cuda_convert_conv_layer(layer *current)
{
	c_param = (conv_param*)current->param;

	cuda_convert_table(&(c_param->filters), c_param->nb_filters * c_param->flat_f_size);
	cuda_convert_table(&(c_param->update), c_param->nb_filters * c_param->flat_f_size);
	
	cuda_convert_table(&(c_param->rotated_filters), c_param->nb_filters * (c_param->flat_f_size-1));
	
	cuda_convert_table(&(current->output), c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h)
		* current->c_network->batch_size);
	cuda_convert_table(&(current->delta_o), c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) 
		* current->c_network->batch_size);
	cuda_convert_table(&(c_param->temp_delta_o), c_param->prev_depth * (c_param->prev_size_w 
		* c_param->prev_size_h) * current->c_network->batch_size);
	
	cuda_convert_table(&(c_param->im2col_input), (c_param->flat_f_size * c_param->nb_area_w 
		* c_param->nb_area_h)* current->c_network->batch_size);
	cuda_convert_table(&(c_param->im2col_delta_o), (c_param->prev_size_w*c_param->prev_size_h) 
		* (c_param->f_size*c_param->f_size*c_param->nb_filters) * current->c_network->batch_size);
	
	if(current->previous != NULL)
	{
		cuda_convert_table(&(current->input), c_param->prev_depth * (c_param->prev_size_w 
			* c_param->prev_size_h) * current->c_network->batch_size);
	}
}


void cuda_forward_conv_layer(layer *current)
{
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;
	int dim_a, dim_b, dim_c;

	if(current->c_network->length == 0)
		return;
	c_param = (conv_param*) current->param;
	
	if(current->previous == NULL)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it that continuous RGB images
		//size in line format
		depth_padding = c_param->prev_size_w * c_param->prev_size_h;
		image_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth;
		current->input = current->c_network->input;
		im2col_prev_bias = 1;
		
		//printf("im_size : %d, depth_padding :%d \n", image_size, depth_padding);
		
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all image in R, then all image in B, ...
		//it also not contain a bias directly in the image
		depth_padding = c_param->prev_size_w * c_param->prev_size_h * current->c_network->batch_size;
		image_padding = c_param->prev_size_w * c_param->prev_size_h;
		im2col_prev_bias = 0;
		current->input = current->previous->output;
		
		/*
		//need to convert the previous output in a format similar as a regular input
		//before giving it to im2col kernel. Need input with continuity of depth for each image.
		cu_blocks = ((c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth) 
			* current->c_network->batch_size + cu_threads - 1) / cu_threads;
		unroll_conv<<< cu_blocks, cu_threads >>>(current->previous->output, current->input, 
			c_param->prev_size_w * c_param->prev_size_h, (c_param->prev_size_w * c_param->prev_size_h 
			* c_param->prev_depth),  c_param->prev_depth, current->c_network->batch_size, 
			(c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth) 
			* current->c_network->batch_size);
		*/
	}
	

	depth_padding = depth_padding;
	
	dim_c = 1;
		
	if(c_param->prev_depth <= 1)
		dim_b =  c_param->prev_depth;
	else
		dim_b = 1;
		
	if(c_param->nb_area_w * c_param->nb_area_h <= 16)
		dim_a = c_param->prev_size_w * c_param->prev_size_h;
	else
		dim_a = 16;
	
	dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
	//create numBlocks regarding the layer dimensions
    dim3 numBlocks2(((c_param->prev_size_w * c_param->prev_size_h) + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
    	(c_param->prev_depth + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
    	(current->c_network->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
	//cuda im2col conversion kernel -> one of the most complex function, go see details above
	im2col_kernel_v3<<< numBlocks2, threadsPerBlock2 >>>(c_param->im2col_input, current->input, 
		c_param->prev_size_w*c_param->prev_size_h, c_param->nb_area_w * c_param->nb_area_h 
		* c_param->flat_f_size, c_param->stride, c_param->padding, c_param->prev_depth, 
		depth_padding, image_padding, current->c_network->batch_size, c_param->f_size, 
		c_param->flat_f_size, c_param->prev_size_w, c_param->nb_area_w, im2col_prev_bias);

	//cuda_print_table_transpose(real* tab, int line_size, int column_size)
	
	//Input X filters matrix multiplication for the all batch
	cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h), c_param->nb_filters, c_param->flat_f_size, &cu_alpha, 
		/*A*/ c_param->im2col_input, c_param->flat_f_size, /*B*/ c_param->filters, c_param->flat_f_size,
		&cu_beta, /*C*/ current->output, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h));
		

	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
}


void cuda_backward_conv_layer(layer *current)
{
	int depth_padding;
	int back_padding;
	int image_padding;
	int flat_f_size;
	int dim_a, dim_b, dim_c;
	
	c_param = (conv_param*) current->param;
	
	
	//######################## ERROR PROPAGATION ########################
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//rotate the filters
		//so the new matrix can be considered as flat_filter_size * current->c_network->batch_size rows against input_depth
		
		
		
		cu_blocks = (c_param->nb_filters * c_param->flat_f_size + cu_threads - 1) / cu_threads;
		rotate_filter_matrix<<< cu_blocks, cu_threads >>>(c_param->filters, c_param->rotated_filters, 
			c_param->flat_f_size, c_param->f_size*c_param->f_size, c_param->nb_filters,
			c_param->nb_filters*c_param->flat_f_size);
		

		//In the backward formalism we asume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this mean that the meaning of nb_area_w/h and prev_size_w/h are reversed in the following operation
		
		
		depth_padding = c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size;
		image_padding = c_param->nb_area_w * c_param->nb_area_h;
		flat_f_size = c_param->f_size * c_param->f_size * c_param->nb_filters;
		
		back_padding = c_param->prev_size_w - c_param->nb_area_w;
		if(back_padding < 0)
			back_padding = 0;
		
		
		//Note : having higher dimensions on the left dim3 dim(a,b,c) grants better results 
		// (profiling shows reduction of compute time near to ~ 17% (on Modified LeNet 5 - MNIST))
		//limit is L2 cache usage, having dim3 a < (16,1,1) allows to maximse it on P2000
		dim_c = 1;
			
		if(c_param->nb_filters <= 1)
			dim_b =  c_param->nb_filters;
		else
			dim_b = 1;
			
		if(c_param->nb_area_w * c_param->nb_area_h <= 16)
			dim_a = c_param->nb_area_w * c_param->nb_area_h;
		else
			dim_a = 16;
		
		dim3 threadsPerBlock2(dim_a, dim_b, dim_c);
		//create numBlocks regarding the layer dimensions
		dim3 numBlocks2((c_param->nb_area_w * c_param->nb_area_h + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
			(c_param->nb_filters + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
			(current->c_network->batch_size + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
		
		im2col_kernel_v3<<< numBlocks2, threadsPerBlock2 >>>(c_param->im2col_delta_o, current->delta_o,
			c_param->nb_area_w * c_param->nb_area_h, (c_param->prev_size_w * c_param->prev_size_h) 
			* flat_f_size, c_param->stride, back_padding, c_param->nb_filters, depth_padding,
			image_padding, current->c_network->batch_size, c_param->f_size, flat_f_size, 
			c_param->nb_area_w, c_param->prev_size_w, 0);

		
		cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, c_param->prev_size_w * c_param->prev_size_h 
			* current->c_network->batch_size, c_param->prev_depth, c_param->f_size * c_param->f_size 
			* c_param->nb_filters, &cu_alpha, /*A*/c_param->im2col_delta_o, c_param->f_size 
			* c_param->f_size * c_param->nb_filters, /*B*/c_param->rotated_filters, c_param->f_size 
			* c_param->f_size*c_param->nb_filters, &cu_beta, /*C*/current->previous->delta_o, 
			c_param->prev_size_w*c_param->prev_size_h*current->c_network->batch_size);

		//update gradiant regarding the previous layer activation function
		//WARNING : ONLY WORK IF PREVIOUS LAYER IS A CONV AS OUTPUT AND DELTA_O SHARE THE SAME DATA ORDER
		current->previous->deriv_activation(current->previous);

	}
	
	//########################  WEIGHTS UPDATE   ########################
	

	//based on the recovered delta_o provided by the next layer propagation
	//CUBLAS_OP_N ,in this case, is a transpose of regular input (see forward function)
	cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, c_param->flat_f_size, c_param->nb_filters, 
		c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size, 
		&current->c_network->learning_rate, c_param->im2col_input, c_param->flat_f_size, 
		current->delta_o, c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size,
		&current->c_network->momentum, c_param->update, c_param->flat_f_size);
	
	cu_blocks = (c_param->flat_f_size * c_param->nb_filters + cu_threads - 1) / cu_threads;
	cuda_update_weights<<< cu_blocks , cu_threads >>>(c_param->filters, c_param->update, c_param->flat_f_size 
		* c_param->nb_filters);

}



//One of the most important function, aim to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but speed up significantly the calculation
//VERSION 3
__global__ void im2col_kernel_v3(real* output, real* input, int image_size, int flat_image_size, int stride, int padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias)
{
	int z = blockIdx.x*blockDim.x + threadIdx.x;
	int d = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.z*blockDim.z + threadIdx.z;
	
	int w, h, x, y;
	
	if( i < batch_size)
	{
		input += i*(image_padding + bias);
		output += i*(flat_image_size);
		
		if(d < depth)
		{
			input += d * depth_padding;
			output += d * f_size*f_size;
			if(z < image_size)
			{
				w = z % w_size + padding;
				h = z / w_size + padding;
				for(x = 0; x < f_size; x += stride)
					for(y = 0; y < f_size; y+= stride)
						if((w-x) >= 0 && (h-y) >= 0 && (w-x) < nb_area_w && (h-y) < nb_area_w)
							output[(w-x) * flat_f_size + (h-y) * nb_area_w * flat_f_size + x + y*f_size] = input[z];
			}
		}
	}
}



__global__ void rotate_filter_matrix(real* in, real* out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int x, y, depth_id;
	
	/*
	if(i < len)
	{
		//#####################################
		//Rotate the filters
		x = i / nb_rows;
		y = i % nb_rows;
		
		if(y < nb_rows-1) //ignore the weights of the bias nodes
		{
			depth_id = y / depth_size;
			
			out[x * (nb_rows-1) + depth_id * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];
		}
		
	}*/
	
	
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


__global__ void unroll_conv(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
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


__global__ void reroll_delta_o(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
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




