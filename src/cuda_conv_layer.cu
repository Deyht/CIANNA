#include "prototypes.h"

static int cu_blocks;
static conv_param *c_param;


//public are in prototypes.h

//private
void cuda_forward_conv_layer(layer *current);
void cuda_backward_conv_layer(layer *current);

__global__ void im2col_kernel(real* output, real* input, int im_size_col, int im_size, int nb_area_w, 
	int nb_area_h, int im_width, int im_height, int depth, int depth_padding, int filter_size, int stride, 
	int padding, int bias, real bias_value, int in_size);
__global__ void update_weights(real *w, real *up, int size);
__global__ void rotate_filter_matrix(real* in, real* out, int nb_rows, int depth_size, int nb_filters_in, int len);
__global__ void unroll_conv(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void reroll_delta_o(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);



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
		* batch_size);
	cuda_convert_table(&(current->delta_o), c_param->nb_filters * (c_param->nb_area_w * c_param->nb_area_h) 
		* batch_size);
	cuda_convert_table(&(c_param->temp_delta_o), c_param->prev_depth * (c_param->prev_size_w 
		* c_param->prev_size_h) * batch_size);
	
	cuda_convert_table(&(c_param->im2col_input), (c_param->flat_f_size * c_param->nb_area_w 
		* c_param->nb_area_h)* batch_size);
	cuda_convert_table(&(c_param->im2col_delta_o), (c_param->prev_size_w*c_param->prev_size_h) * 
		/* flat_filter*/(c_param->f_size*c_param->f_size*c_param->nb_filters) * batch_size);
	
	if(current->previous != NULL)
	{
		cuda_convert_table(&(current->input), c_param->prev_depth * (c_param->prev_size_w 
			* c_param->prev_size_h) * batch_size);

	}
}


void cuda_forward_conv_layer(layer *current)
{
	int image_size;
	int depth_padding;

	if(length == 0)
		return;
		
	c_param = (conv_param*) current->param;
	
	
	if(current->previous == NULL)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it that continuous RGB images
		image_size = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth + 1; //size in line format
		depth_padding = c_param->prev_size_w * c_param->prev_size_h;
		current->input = input;
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all image in R, then all image in B, ...
		//it also not contain a bias directly in the image
		image_size = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth;
		depth_padding = c_param->prev_size_w * c_param->prev_size_h;
		
		//need to convert the previous output in a format similar as a regular input
		//before giving it to im2col kernel. Need input with continuity of depth for each image.
		cu_blocks = ((c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth) * batch_size + cu_threads - 1) / cu_threads;
		unroll_conv<<< cu_blocks, cu_threads >>>(current->previous->output, current->input, 
			c_param->prev_size_w * c_param->prev_size_h, (c_param->prev_size_w * c_param->prev_size_h 
			* c_param->prev_depth),  c_param->prev_depth, batch_size, (c_param->prev_size_w 
			* c_param->prev_size_h * c_param->prev_depth) * batch_size);
	
		//printf("input 2nd layer layer\n");
		//cuda_print_table_transpose(current->input, c_param->prev_depth, (c_param->prev_size_w 
		//	* c_param->prev_size_h ) * batch_size);
	}
	
	
	//cuda_print_table(input, batch_size*image_size, 28);
	//cuda_print_table(current->input, batch_size*image_size, 28);
	
	//printf("bias value : %f\n", c_param->bias_value);
	
	//printf("\n %d %d %d %d %d %d %d %d %d %d %d %d\n",(c_param->nb_area_w * c_param->nb_area_h) * c_param->flat_f_size, image_size, c_param->nb_area_w, c_param->nb_area_h,  c_param->prev_size_w, c_param->prev_size_h, c_param->prev_depth, depth_padding, filter_size, stride, padding, nb_area_w*nb_area_h*batch_size);

	//cuda im2col conversion kernel -> one of the most complex function, go see details above
	cu_blocks = (c_param->nb_area_w * c_param->nb_area_h * batch_size + cu_threads - 1) / cu_threads;
	im2col_kernel<<< cu_blocks, cu_threads >>>(c_param->im2col_input, current->input, 
		/*im_size_col*/ (c_param->nb_area_w * c_param->nb_area_h) * c_param->flat_f_size, 
		/* im_size*/ image_size, c_param->nb_area_w, c_param->nb_area_h, c_param->prev_size_w,
		c_param->prev_size_h, c_param->prev_depth, depth_padding, c_param->f_size, c_param->stride,
		c_param->padding, /*add bias*/ 1, c_param->bias_value, (c_param->nb_area_w * c_param->nb_area_h)
		* batch_size);
	
	//printf("Conv filters\n");
	//cuda_print_table_transpose(c_param->filters, c_param->nb_filters, c_param->flat_f_size);
	
	//printf("Input conv layer\n");
	//cuda_print_table(c_param->im2col_input, (c_param->flat_f_size * c_param->nb_area_w * c_param->nb_area_h)*batch_size, c_param->flat_f_size);
	
	
	//Input X filters matrix multiplication for the all batch
	cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, batch_size*(c_param->nb_area_w*c_param->nb_area_h), 
		c_param->nb_filters, c_param->flat_f_size, &cu_alpha, /*A*/ c_param->im2col_input, 
		c_param->flat_f_size, /*B*/ c_param->filters, c_param->flat_f_size, &cu_beta, 
		/*C*/ current->output, batch_size*(c_param->nb_area_w*c_param->nb_area_h));
	
	//cuda_print_table_transpose(current->output, c_param->nb_filters, batch_size*(c_param->nb_area_w*c_param->nb_area_h));

	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	//printf("layer output\n");
	//cuda_print_table_transpose(current->output, c_param->nb_filters, batch_size*(c_param->nb_area_w*c_param->nb_area_h));
	
	//exit(EXIT_SUCCESS);

}


void cuda_backward_conv_layer(layer *current)
{
	int image_size;
	int depth_padding;
	int flat_f_size;
	
	c_param = (conv_param*) current->param;
	
	//printf("prev delta_o conv\n");
	//cuda_print_table_transpose(current->delta_o, c_param->nb_filters, c_param->nb_area_w * c_param->nb_area_h * batch_size);
	//printf("end delta_o\n");
	
	
	//######################## ERROR PROPAGATION ########################

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//printf("program gone in prop for only conv layer\n");
		//exit(EXIT_FAILURE);
		//rotate the filters, also reorganise to get filters regarding depth
		//organized as F1_D1, F2_D1, F3_D1, ..., F1_D2, F2_D2, F3_D3, ...
		//so the new matrix can be considered as flat_filter_size * batch_size rows against input_depth
		
		//printf("Conv filters before rotation\n");
		//cuda_print_table_transpose(c_param->filters, c_param->nb_filters, c_param->flat_f_size);
		cu_blocks = (c_param->nb_filters * c_param->flat_f_size + cu_threads - 1) / cu_threads;
		rotate_filter_matrix<<< cu_blocks, cu_threads >>>(c_param->filters, c_param->rotated_filters, 
			c_param->flat_f_size, c_param->f_size*c_param->f_size, c_param->nb_filters,
			c_param->nb_filters*(c_param->flat_f_size));
		//printf("Conv filters after rotation\n");
		//cuda_print_table_transpose(c_param->rotated_filters, c_param->prev_depth, c_param->f_size*c_param->f_size*c_param->nb_filters);
		
		//In the backward formalism we asume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this mean that the meaning of nb_area_w/h and prev_size_w/h are reversed in the following operation
		image_size = c_param->nb_area_w * c_param->nb_area_h * c_param->nb_filters; //size in line format
		depth_padding = c_param->nb_area_w * c_param->nb_area_h;
		flat_f_size = c_param->f_size * c_param->f_size * c_param->nb_filters; //no bias here in backprop
		
		
		cu_blocks = (c_param->prev_size_w * c_param->prev_size_h * batch_size + cu_threads - 1) / cu_threads;
		im2col_kernel<<< cu_blocks, cu_threads >>>(c_param->im2col_delta_o, current->delta_o,
			/*im_size_col*/ (c_param->prev_size_w * c_param->prev_size_h)*flat_f_size , /* im_size*/ image_size,
			c_param->prev_size_w, c_param->prev_size_h, c_param->nb_area_w , c_param->nb_area_h, 
			c_param->nb_filters, depth_padding, c_param->f_size, c_param->stride, /*padding*/c_param->f_size-1,
			/*no bias*/ 0, c_param->bias_value, (c_param->prev_size_w * c_param->prev_size_h)*batch_size);
			
		//printf("delta_o_im2col \n");
		//cuda_print_table(c_param->im2col_delta_o, flat_f_size*(c_param->prev_size_w * c_param->prev_size_h)*batch_size, flat_f_size);
		
		//cuda_print_table_transpose(c_param->rotated_filters, c_param->prev_depth, c_param->f_size*c_param->f_size*c_param->nb_filters);
		
		//printf("%d %d %d %d\n", c_param->prev_size_w, c_param->prev_size_h, c_param->prev_depth, c_param->nb_filters);
		cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, c_param->prev_size_w*c_param->prev_size_h*batch_size,
			c_param->prev_depth, c_param->f_size*c_param->f_size*c_param->nb_filters, &cu_alpha,
			/*A*/c_param->im2col_delta_o, c_param->f_size*c_param->f_size*c_param->nb_filters, 
			/*B*/c_param->rotated_filters, c_param->f_size*c_param->f_size*c_param->nb_filters, &cu_beta, 
			/*C*/current->previous->delta_o, c_param->prev_size_w*c_param->prev_size_h*batch_size);
			
		//printf("temp_delta_o\n");
		//cuda_print_table_transpose(c_param->temp_delta_o, c_param->prev_depth, c_param->prev_size_w*c_param->prev_size_h*batch_size);
		
		//Convert previous->delta_o in proper format
		//A1B1 - A1B2 - A1B2 - A2B1 - A2B2 - A2B3 ...   =>  A1B1 - A2B1 - A3B1 - A1B2 - A2B2 - A2B2 - ...
		/*
		cu_blocks = (c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth * batch_size 
			+ cu_threads - 1) / cu_threads;
		reroll_delta_o<<< cu_blocks, cu_threads >>>(c_param->temp_delta_o, current->previous->delta_o,
			c_param->prev_size_w * c_param->prev_size_h , c_param->prev_size_w * c_param->prev_size_h 
			* c_param->prev_depth,  c_param->prev_depth, batch_size, c_param->prev_size_w 
			* c_param->prev_size_h * c_param->prev_depth * batch_size);
		*/
		//current->previous->delta_o = c_param->temp_delta_o;
		
		//cuda_print_table_transpose(current->previous->delta_o, c_param->prev_depth, c_param->prev_size_w*c_param->prev_size_h*batch_size);
		//update gradiant regarding the previous layer activation function
		current->previous->deriv_activation(current->previous);
		//printf("Activated\n");
		//cuda_print_table_transpose(current->previous->delta_o, c_param->prev_depth, c_param->prev_size_w*c_param->prev_size_h*batch_size);
	}
	
	//########################  WEIGHTS UPDATE   ########################
	
	//based on the recovered delta_o provided by the next layer propagation
	//printf("\n\nDelta_o activated\n\n");
	//cuda_print_table_transpose(current->delta_o, c_param->nb_filters, (c_param->nb_area_w * c_param->nb_area_h)*batch_size);

	//printf("%d %d %d %d %d %f\n", c_param->flat_f_size, c_param->nb_filters, c_param->nb_area_w, c_param->nb_area_h, batch_size, learning_rate);
	//CUBLAS_OP_N ,in this case, is a transpose of regular input (see forward function)
	
	//printf("momentum : %f\n", momentum);
	cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, c_param->flat_f_size, c_param->nb_filters, 
		c_param->nb_area_w * c_param->nb_area_h * batch_size, &learning_rate, c_param->im2col_input, 
		c_param->flat_f_size, current->delta_o, c_param->nb_area_w * c_param->nb_area_h * batch_size,
		&momentum, c_param->update, c_param->flat_f_size);
	
	//printf("\n Update conv \n");
	//cuda_print_table_transpose(c_param->update, c_param->nb_filters, c_param->flat_f_size);
	
	cu_blocks = (c_param->flat_f_size * c_param->nb_filters + cu_threads - 1) / cu_threads;
	update_weights<<< cu_blocks , cu_threads >>>(c_param->filters, c_param->update, c_param->flat_f_size 
		* c_param->nb_filters);
	//cuda_print_table_transpose(c_param->filters, c_param->nb_filters, c_param->flat_f_size);

}



//One of the most important function, aim to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but speed up significantly the calculation
__global__ void im2col_kernel(real* output, real* input, int im_size_col, int im_size, int nb_area_w, 
	int nb_area_h, int im_width, int im_height, int depth, int depth_padding, int filter_size, int stride, 
	int padding, int bias, real bias_value, int in_size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j, k, d;
	int depth_temp, depth_temp_im;
	int row_temp;
	int row_temp_im;
	int pos_out, x, y;
	
	if(i < in_size)
	{
		output += (i/(nb_area_w*nb_area_h))*im_size_col;
		input += (i/(nb_area_w*nb_area_h))*im_size;
		i = i % (nb_area_w*nb_area_h);
	
		x = (i/nb_area_w) * stride - padding;
		y = (i%nb_area_h) * stride - padding;
		pos_out = i*(filter_size*filter_size*depth + bias);
		
		for(d = 0; d < depth; d++)
		{	
			depth_temp = d*filter_size*filter_size;
			depth_temp_im = d*depth_padding;
			for(j = 0; j < filter_size; j++)
			{
				row_temp = j*filter_size;
				row_temp_im = (x+j)*(im_width);
				for(k = 0; k < filter_size; k++)
				{
					if(x + j >= 0 && x + j < im_width && y + k >= 0 && y + k < im_height)
						output[pos_out + depth_temp + row_temp + k] = input[depth_temp_im + row_temp_im + (y+k)];
					else
						output[pos_out + depth_temp + row_temp + k] = 0;
				}
			}
		}
		if(bias != 0)
			output[pos_out + filter_size*filter_size*depth] = bias_value;
	}
}


__global__ void update_weights(real *w, real *up, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < size)
		w[i] -= up[i];
}



__global__ void rotate_filter_matrix(real* in, real* out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int x, y, filter_id;
	
	if(i < len)
	{
		//#####################################
		//Rotate the filters
		x = i / nb_rows;
		y = i % nb_rows;
		
		if(y < nb_rows-1) //remove the weights of the bias nodes
		{
			filter_id = y / depth_size;
			
			out[filter_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];
		}
		
	}
}


__global__ void unroll_conv(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	/*
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j, pos_line, all_filter_batch_size;
	
	if( i < size)
	{
		pos_line = i%(all_filter_size) + (i /all_filter_size)*(all_filter_size*nb_filters);
		//printf("%d\n",i/(all_filter_size));
		//printf("pos_line : %d\n", pos_line);
		all_filter_batch_size = all_filter_size*batch_size;
		for(j = 0; j < nb_filters; j++)
		{
			out[pos_line] = in[i];
			in += all_filter_batch_size;
			out += all_filter_size;
		}
	}
	*/
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




