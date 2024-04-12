

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

static conv_param *c_param;

//public are in prototypes.h


//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only one load per element of the original image.
//VERSION 5.4
void im2col_fct
	(void *i_output, void *i_input, int image_size, int flat_image_size, 
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int internal_padding_w, int internal_padding_h, int internal_padding_d, 
	int channel, int channel_padding, int image_padding, int batch_size, 
	int f_size_w, int f_size_h, int f_size_d, int flat_f_size, 
	int w_size, int h_size, int d_size, 
	int nb_area_w, int nb_area_h, int nb_area_d, int bias_in, int bias_out)
{
	float local_pix;
	float *t_output = (float*) i_output;
	float *t_input  = (float*) i_input;

	float *output, *input;

	int i, c, p, w, h, d, x, y, z;
	int pos_w_filter, pos_h_filter, pos_d_filter;
	int loc;

	#pragma omp parallel for private(c, p, local_pix, w, h, d, x, y, z, pos_w_filter, pos_h_filter, pos_d_filter, input, output) collapse(2) schedule(guided,1)
	for(i = 0; i < batch_size; i++)
	{
		for(c = 0; c < channel; c++)
		{
			input = t_input + i*(image_padding + bias_in) + c * channel_padding ;
			output = t_output + i*(flat_image_size) + c * f_size_w*f_size_h*f_size_d;
			for(p = 0; p < image_size; p++)
			{
				local_pix = input[p];

				d = (p / (w_size*h_size))*(1 + internal_padding_d) + padding_d;
				h = (p % (w_size*h_size) / w_size)*(1 + internal_padding_h) + padding_h;
				w = (p % (w_size*h_size) % w_size)*(1 + internal_padding_w) + padding_w;

				for(z = d/stride_d; (d-z*stride_d < f_size_d); z -=1)
				{
					pos_d_filter = d-z*stride_d;
					if((z < 0) || (z > (d_size + (d_size-1)*internal_padding_d + 2*padding_d - f_size_d)/stride_d))
						continue;
					for(y = h/stride_h; (h-y*stride_h < f_size_h); y -= 1)
					{
						pos_h_filter = h-y*stride_h;
						if((y < 0) || (y > (h_size + (h_size-1)*internal_padding_h + 2*padding_h - f_size_h)/stride_h))
							continue;
						for(x = w/stride_w; (w-x*stride_w < f_size_w); x -= 1)
						{
							pos_w_filter = w-x*stride_w;
							if((x < 0) || (x > (w_size + (w_size-1)*internal_padding_w + 2*padding_w - f_size_w)/stride_w))
								continue;
							loc = z*(size_t)nb_area_w*nb_area_h*flat_f_size + y*nb_area_w*flat_f_size 
								+ x*flat_f_size + pos_w_filter + pos_h_filter*f_size_w + pos_d_filter*f_size_w*f_size_h;
							if((bias_out && (loc)%flat_f_size >= flat_f_size - 1))
								continue;
							if(loc >= 0 && loc < flat_image_size)
								output[loc] = local_pix;
						}
					}
				}
			}
		}
	}
}

void rotate_filter_matrix_fct(void *i_in, void *i_out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i, x, y, depth_id;

	float* in  = (float*) i_in;
	float* out = (float*) i_out;

	for(i = 0; i < len; i ++)
	{
		/*Rotate and move the filters*/
		x = i / nb_rows;
		y = i % nb_rows;
		/*remove the weights of the bias nodes*/
		if(y < nb_rows-1) 
		{
			depth_id = y / depth_size;
			out[depth_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] = in[x*nb_rows+y];
		}
	}
}

void dropout_select_conv(float *mask, size_t size, float drop_rate)
{
	size_t i;
	float rand;
	
	//#pragma omp parallel for private(rand) schedule(guided,4)
	//OMP overhead is too high for "small" dense layers
	//Performance is limited by CPU cache size and speed regardless of core count
	//AND can result in seed problem with random_uniform (see YOLO activ function for a proper example)
	for(i = 0; i < size; i++)
	{
		rand = random_uniform();
		if(rand >= drop_rate)
			mask[i] = 1.0f;
		else
			mask[i] = 0.0f;
	}
}


void dropout_apply_conv(void *i_table, float* mask, size_t size)
{
	size_t i;
	float* table = (float*) i_table;
	
	for(i = 0; i < size; i++)
		table[i] = table[i]*mask[i];
}


void dropout_scale_conv(void *i_table, size_t size, float drop_rate)
{
	size_t i;
	float* table = (float*) i_table;
	
	for(i = 0; i < size; i++)
		table[i] = table[i]*(1.0f-drop_rate);
}


void forward_conv_layer(layer *current)
{
	int i, j, b;
	double h;
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;
	
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
	
	//im2col conversion fct -> one of the most complex function, go see details above
	im2col_fct(c_param->im2col_input, current->input, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2], 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * c_param->flat_f_size, 
		c_param->stride[0], c_param->stride[1], c_param->stride[2],
		c_param->padding[0], c_param->padding[1], c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		c_param->prev_depth, depth_padding, image_padding, net->batch_size, 
		c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], c_param->flat_f_size,
		c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
		c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], im2col_prev_bias, 1);

	//Input X filters matrix multiplication for the all batch
	
	float *f_im2col_input = (float*) c_param->im2col_input;
	float *f_filters = (float*) c_param->filters;
	float *f_output = (float*) current->output;
	
	#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 2)
	for(b = 0; b < net->batch_size * (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]); b++)
	{
		for(i = 0; i <  c_param->nb_filters; i++)
		{
			h = 0.0;
			for(j = 0; j < c_param->flat_f_size; j++)
			{
				h += f_im2col_input[b*(c_param->flat_f_size) + j]
						* f_filters[i*(c_param->flat_f_size) + j];
			}
			f_output[i*(net->batch_size * (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]))+b] = h;
		}
	}
	
	if(current->dropout_rate > 0.01f)
	{
	
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			dropout_select_conv(c_param->dropout_mask, c_param->nb_filters 
				* (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size, current->dropout_rate);
			dropout_apply_conv(current->output, c_param->dropout_mask, c_param->nb_filters 
				* (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size);
		}
		else
			dropout_scale_conv(current->output, c_param->nb_filters 
				* (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size, current->dropout_rate);
	}
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
}

void backward_conv_layer(layer *current)
{
	int i, j, k, b;
	double h;
	int depth_padding;
	int back_padding[3];
	int image_padding;
	int flat_f_size;
	float *c_prev_delta_o;
	
	network* net = current->c_network;
	
	c_param = (conv_param*) current->param;
	
	if(current->dropout_rate > 0.01f && (net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL)))
	{
		dropout_apply_conv(current->delta_o, c_param->dropout_mask, c_param->nb_filters 
			* (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * net->batch_size);
	}
	
	//######################## ERROR PROPAGATION ########################
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//rotate the filters
		//so the new matrix can be considered as flat_filter_size * net->batch_size rows against input_depth
		rotate_filter_matrix_fct(c_param->filters, c_param->rotated_filters, 
			c_param->flat_f_size, c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2], 
			c_param->nb_filters, c_param->nb_filters * c_param->flat_f_size);

		//In the backward formalism we asume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this mean that the meaning of nb_area and prev_size are reversed in the following operation
		depth_padding = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * net->batch_size;
		image_padding = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
		flat_f_size = c_param->f_size[0] * c_param->f_size[1] * c_param->f_size[2] * c_param->nb_filters;
		//this flat size remove the bias != c_param->flat_f_size
		
		for(k = 0; k < 3; k++)
		{
			back_padding[k] =  c_param->f_size[k] -  c_param->padding[k] - 1;
			if(back_padding[k] < 0)
				back_padding[k] = 0;
		}
		
		im2col_fct(c_param->im2col_delta_o,
			current->delta_o, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], 
			(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) * flat_f_size, 
			c_param->int_padding[0] + 1, c_param->int_padding[1] + 1, c_param->int_padding[2] + 1,
			back_padding[0], back_padding[1], back_padding[2],
			c_param->stride[0] - 1, c_param->stride[1] - 1, c_param->stride[2] - 1,
			c_param->nb_filters, depth_padding, image_padding, net->batch_size,
			c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], flat_f_size, 
			c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
			c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 0, 0);
		
		if(current->previous->type == DENSE)
			c_prev_delta_o = c_param->temp_delta_o;
		else
			c_prev_delta_o = current->previous->delta_o;
		
		float *f_im2col_delta_o = (float*) c_param->im2col_delta_o;
		float *f_rotated_filters = (float*) c_param->rotated_filters;
		
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size; b++)
		{
			for(i = 0; i < c_param->prev_depth; i++)
			{
				h = 0.0;
				for(j = 0; j < flat_f_size; j++)
				{
					h += f_im2col_delta_o[b*(flat_f_size) + j]
							* f_rotated_filters[i*(flat_f_size) + j];
				}
				c_prev_delta_o[i*(c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size)+b] = h;
			}
		}
		
		if(current->previous->type == DENSE)
		{	
			flat_dense(c_param->temp_delta_o, current->previous->delta_o, 0, 
				c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2],
				c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] 
				* c_param->prev_depth + 1, c_param->prev_depth, net->batch_size, 
				(c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2] 
				* c_param->prev_depth + 1) * net->batch_size);
		}
		
		current->previous->deriv_activation(current->previous);
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		int dim_batch = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*net->batch_size;
		
		float *f_im2col_input = (float*) c_param->im2col_input;
		float *f_delta_o = (float*) current->delta_o;
		float *f_update = (float*) c_param->update;
		
		#pragma omp parallel for private(j, b, h) collapse(2) schedule(dynamic, 1)
		for(i = 0; i <  c_param->nb_filters; i++)
		{
			for(j = 0; j < c_param->flat_f_size; j++)
			{
				h = 0.0;
				for(b = 0; b < dim_batch; b++)
				{
					h += f_im2col_input[b*(c_param->flat_f_size) + j]
							* f_delta_o[i*(dim_batch) + b];
				}
				f_update[i*(c_param->flat_f_size) + j] = net->learning_rate/net->batch_size*h 
						+ net->momentum * f_update[i*(c_param->flat_f_size) + j];
			}
		}
		
		update_weights(c_param->filters, c_param->update, net->learning_rate*net->weight_decay, 
			0, c_param->flat_f_size*c_param->nb_filters);
	}
}


void naiv_conv_define(layer *current)
{
	current->forward = forward_conv_layer;
	current->backprop = backward_conv_layer;
}






