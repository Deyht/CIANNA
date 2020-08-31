


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

static conv_param *c_param;

//public are in prototypes.h

//private
static void forward_conv_layer(layer *current);
static void backward_conv_layer(layer *current);


void naiv_conv_define(layer *current)
{
	current->forward = forward_conv_layer;
	current->backprop = backward_conv_layer;
}


void forward_conv_layer(layer *current)
{
	int i, j, b;
	double h;
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;

	if(current->c_network->length == 0)
		return;
	
	c_param = (conv_param*) current->param;
	
	if(current->previous == NULL)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it as continuous RGB images
		//size in line format
		depth_padding = c_param->prev_size_w * c_param->prev_size_h;
		image_padding = c_param->prev_size_w * c_param->prev_size_h * c_param->prev_depth;
		current->input = current->c_network->input;
		im2col_prev_bias = 1;
		
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all image in R, then all image in B, ...
		//it also not contain a bias directly in the image
		depth_padding = c_param->prev_size_w * c_param->prev_size_h * current->c_network->batch_size;
		image_padding = c_param->prev_size_w * c_param->prev_size_h;
		im2col_prev_bias = 0;
		current->input = current->previous->output;
	}
	
	//print_table(current->input + 785, (c_param->prev_size_w*c_param->prev_size_h+1), 1);
	
	//im2col conversion fct -> one of the most complex function, go see details above
	im2col_fct_v4(c_param->im2col_input, current->input, 
		c_param->prev_size_w*c_param->prev_size_h, c_param->nb_area_w * c_param->nb_area_h 
		* c_param->flat_f_size, c_param->stride, c_param->padding, 0, c_param->prev_depth, 
		depth_padding, image_padding, current->c_network->batch_size, c_param->f_size, 
		c_param->flat_f_size, c_param->prev_size_w, c_param->nb_area_w, im2col_prev_bias);

	//Input X filters matrix multiplication for the all batch
	
	float *f_im2col_input = (float*) c_param->im2col_input;
	float *f_filters = (float*) c_param->filters;
	float *f_output = (float*) current->output;
	
	#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 2)
	for(b = 0; b < current->c_network->batch_size * (c_param->nb_area_w*c_param->nb_area_h); b++)
	{
		for(i = 0; i <  c_param->nb_filters; i++)
		{
			h = 0.0;
			for(j = 0; j < c_param->flat_f_size; j++)
			{
				h += f_im2col_input[b*(c_param->flat_f_size) + j]
						* f_filters[i*(c_param->flat_f_size) + j];
			}
			f_output[i*(current->c_network->batch_size * (c_param->nb_area_w*c_param->nb_area_h))+b] = h;
		}
	}
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
}




void backward_conv_layer(layer *current)
{
	int i, j, b;
	double h;
	int depth_padding;
	int back_padding;
	int image_padding;
	int flat_f_size;
	//struct timeval ep_timer;
	
	c_param = (conv_param*) current->param;
	
	
	//######################## ERROR PROPAGATION ########################
	
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
		//rotate the filters
		//so the new matrix can be considered as flat_filter_size * current->c_network->batch_size rows against input_depth
		rotate_filter_matrix(c_param->filters, c_param->rotated_filters, 
			c_param->flat_f_size, c_param->f_size*c_param->f_size, c_param->nb_filters,
			c_param->nb_filters*c_param->flat_f_size);
		

		//In the backward formalism we asume continuous images (the activation maps)
		//the backprop process generate bias nodes so they must be taken into account
		
		//Warning : the convolution processed is reversed using full convolution with padding
		//this mean that the meaning of nb_area_w/h and prev_size_w/h are reversed in the 
		//following operation
		depth_padding = c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size;
		image_padding = c_param->nb_area_w * c_param->nb_area_h;
		flat_f_size = c_param->f_size * c_param->f_size * c_param->nb_filters;
		
		back_padding =  c_param->f_size -  c_param->padding - 1;
		if(back_padding < 0)
			back_padding = 0;
		
		im2col_fct_v4(c_param->im2col_delta_o, current->delta_o,
			c_param->nb_area_w * c_param->nb_area_h, (c_param->prev_size_w * c_param->prev_size_h) 
			* flat_f_size, 1, back_padding, c_param->stride - 1 , c_param->nb_filters, 
			depth_padding, image_padding, current->c_network->batch_size, c_param->f_size, flat_f_size, 
			c_param->nb_area_w, c_param->prev_size_w, 0);
			
		float *f_im2col_delta_o = (float*) c_param->im2col_delta_o;
		float *f_rotated_filters = (float*) c_param->rotated_filters;
		float *f_previous_delta_o = (float*) current->previous->delta_o;
		
		#pragma omp parallel for private(i, j, h) collapse(2) schedule(guided, 4)
		for(b = 0; b < c_param->prev_size_w * c_param->prev_size_h * current->c_network->batch_size; b++)
		{
			for(i = 0; i < c_param->prev_depth; i++)
			{
				h = 0.0;
				for(j = 0; j < c_param->f_size * c_param->f_size * c_param->nb_filters; j++)
				{
					h += f_im2col_delta_o[b*(c_param->f_size * c_param->f_size * c_param->nb_filters) + j]
							* f_rotated_filters[i*(c_param->f_size * c_param->f_size*c_param->nb_filters) + j];
				}
				f_previous_delta_o[i*(c_param->prev_size_w*c_param->prev_size_h*current->c_network->batch_size)+b] = h;
			}
		}
		
		//update gradiant regarding the previous layer activation function
		//WARNING : ONLY WORK IF PREVIOUS LAYER IS A CONV AS OUTPUT AND DELTA_O SHARE THE SAME DATA ORDER
		current->previous->deriv_activation(current->previous);
		
	}
	
	//########################  WEIGHTS UPDATE   ########################
	
	int dim_batch = c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size;
	
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
			f_update[i*(c_param->flat_f_size) + j] = current->c_network->learning_rate*h 
					+ current->c_network->momentum * f_update[i*(c_param->flat_f_size) + j];
		}
	}
	
	update_weights(c_param->filters, c_param->update, c_param->flat_f_size 
		* c_param->nb_filters);

}






//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only 1 load per element of the original image.
//VERSION 4.1

void im2col_fct_v4(void* output, void* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias)
{
	int z, d, i;
	float local_pix;
	int w, h, x, y;
	int pos_w_filter, pos_h_filter;
	float *t_in, *t_out;
	int loc;
	
	float* f_input = (float*) input;
	float* f_output = (float*) output;
	
	// Must test various OpenMP writing and compare performance
	#pragma omp parallel for private(d, z, local_pix, w, h, x, y, pos_w_filter, pos_h_filter, t_in, t_out) collapse(2) schedule(guided,2)
	for(i = 0; i < batch_size; i++)
	{
		for(d = 0; d < depth; d++)
		{
			t_in = f_input + i*(image_padding + bias) + d * depth_padding;
			t_out =  f_output + i*(flat_image_size) + d * f_size*f_size;
			for(z = 0; z < image_size; z++)
			{
				local_pix = t_in[z];
				
				w = (z % w_size)*(1 + internal_padding) + padding;
				h = (z / w_size)*(1 + internal_padding) + padding;
				
				for(x = w/stride; (w-x*stride < f_size) && (x >= 0); x -= 1)
				{
					pos_w_filter = w-x*stride;
					for(y = h/stride; (h-y*stride < f_size) && (y >= 0); y -= 1)
					{
						pos_h_filter = h-y*stride;
						loc = x*flat_f_size + y*nb_area_w*flat_f_size + pos_w_filter + pos_h_filter*f_size;
						if(loc >= 0 && loc < flat_image_size)
							t_out[loc] = local_pix;
					}
				}
				
			}
		}
	}
}

void rotate_filter_matrix(void* in, void* out, int nb_rows, int depth_size, int nb_filters_in, int len)
{
	int i, x, y, depth_id;

	float *f_in = (float*) in;
	float *f_out = (float*) out;

	#pragma omp parallel for private(x, y, depth_id) schedule(dynamic,1)
	for(i = 0; i < len; i++)
	{
		//#####################################
		//Rotate and move the filters
		x = i / nb_rows;
		y = i % nb_rows;
		
		if(y < nb_rows-1) //remove the weights of the bias nodes
		{
			depth_id = y / depth_size;
			
			f_out[depth_id * depth_size*nb_filters_in + x * depth_size + (depth_size - 1 - y%depth_size)] 
				= f_in[x*nb_rows+y];
		}
		
	}
	
}


void unroll_conv(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	//presently unused

	int i;
	int map_id, image_id, pos;
	
	float *f_in = (float*) in;
	float *f_out = (float*) out;
	
	for(i = 0; i < size; i++)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;

		f_out[i] = f_in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}


void reroll_delta_o(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	//presently unused
	int i;
	int map_id, image_id, pos;
	
	float *f_in = (float*) in;
	float *f_out = (float*) out;
	
	for(i = 0; i < size; i++)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		f_out[i] = f_in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}



