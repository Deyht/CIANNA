
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
static conv_param *c_param;

// Public are in "prototypes.h"

// Private prototypes
void naiv_forward_conv_layer(layer *current);
void naiv_backward_conv_layer(layer *current);


//One of the most important function, aims to convert an image into a table that contains all the
//areas that will be used for convolution. Highly redundant but still allows a significant speed up
//due to subsequent matrix operations. Currently memory bound despite only one load per element of the original image.
//VERSION 5.4
void im2col_fct(void* i_output, void* i_input,
	int stride_w, int stride_h ,int stride_d,
	int padding_w, int padding_h, int padding_d,
	int internal_padding_w, int internal_padding_h, int internal_padding_d,
	int f_size_w, int f_size_h, int f_size_d,
	size_t w_size, size_t h_size, size_t d_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d,
	size_t in_nb_channels, size_t in_group_size, size_t in_image_size, size_t in_image_offset, size_t in_channel_offset,
	size_t out_image_offset, size_t out_group_offset,
	int batch_size, int bias_out)
{
	float *t_output = (float*) i_output;
	float *t_input  = (float*) i_input;

	float *output, *input;
	float local_pix;
	
	int i, c, p, w, h, d, x, y, z;
	int pos_w_filter, pos_h_filter, pos_d_filter;
	size_t loc, spatial_f_size, flat_f_size;
	
	spatial_f_size = f_size_w * f_size_h * f_size_d;
	flat_f_size = spatial_f_size * in_group_size + bias_out;

	#pragma omp parallel for private(input, output, local_pix, c, p, w, h, d, x, y, z, loc, pos_w_filter, pos_h_filter, pos_d_filter) collapse(2) schedule(guided,1)
	for(i = 0; i < batch_size; i++)
	{
		for(c = 0; c < in_nb_channels; c++)
		{
			input = t_input + i * in_image_offset;
			output = t_output + i * out_image_offset;
			
			input += c * in_channel_offset;
			output += (c/in_group_size) * out_group_offset;
			output += (c%in_group_size) * spatial_f_size;
			
			for(p = 0; p < in_image_size; p++)
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
							loc = (z*(size_t)nb_area_w*nb_area_h + y*nb_area_w + x)*flat_f_size 
								+ pos_w_filter + pos_h_filter*f_size_w + pos_d_filter*f_size_w*f_size_h;
							if((bias_out && loc%flat_f_size >= flat_f_size - bias_out))
								continue;
							if(loc < out_image_offset) /* loc is > 0 by construction */
								output[loc] = local_pix;
						}
					}
				}
			}
		}
	}
}


void rotate_filter_matrix_fct(void* i_in, void* i_out, size_t nb_rows, size_t spatial_f_size, size_t out_group_size, size_t len)
{
	int i, x, y;

	float *in  = (float*) i_in;
	float *out = (float*) i_out;
	
	float *l_out;

	#pragma omp parallel for private(x, y, l_out) schedule(guided,4) if(len >= 128)
	for(i = 0; i < len; i ++)
	{
		/*Rotate and move the filters*/
		x = i / nb_rows;
		y = i % nb_rows;
		
		/*remove the weights of the bias nodes*/
		if(y < nb_rows-1)
		{
			l_out = out + (x/out_group_size) * (nb_rows-1) * out_group_size;
			l_out += (x%out_group_size) * spatial_f_size;
			l_out += (y/spatial_f_size) * spatial_f_size * out_group_size;
			l_out[spatial_f_size - 1 - y%spatial_f_size] = in[x*nb_rows+y];
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


void naiv_forward_conv_layer(layer *current)
{
	size_t g, m, n, k;
	double acc;
	size_t subdim_M, subdim_N, subdim_K, nb_groups, batch_size;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	
	float *f_im2col_input, *f_filters, *f_output;
	
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
		in_channel_offset = nb_regions_in * net->batch_size;
		current->input    = current->previous->output;
	}
	
	//########## Subproblem dimensions ##########
	
	subdim_M = batch_size * nb_regions_out;
	subdim_N = chan_per_group_out;
	subdim_K = spatial_f_size * chan_per_group_in + 1;
	
	//########## Preparing Im2col(K,M) ##########
	
	im2col_fct(c_param->im2col_input, current->input, 
		     c_param->stride[0],      c_param->stride[1],      c_param->stride[2],
		    c_param->padding[0],     c_param->padding[1],     c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		     c_param->f_size[0],      c_param->f_size[1],      c_param->f_size[2],
		   current->prev_dim[0],    current->prev_dim[1],    current->prev_dim[2],
		 current->output_dim[0],  current->output_dim[1],  current->output_dim[2],
		prev_nb_channels, chan_per_group_in, nb_regions_in, in_image_offset, in_channel_offset,
		subdim_K*nb_regions_out, subdim_K*subdim_M, batch_size, 1);
		
	//####### Im2col_T(M,K) x Weights(K,N) #######
	
	for(g = 0; g < nb_groups; g++)
	{
		f_im2col_input = (float*)c_param->im2col_input + g * subdim_K*subdim_M;
		f_output       = (float*)current->output       + g * subdim_M*subdim_N;
				
		if(net->is_inference == 1 && net->use_wema)
			f_filters = current->ema_weights  + g * subdim_K*subdim_N;
		else
			f_filters = current->FP32_weights + g * subdim_K*subdim_N;
	
		#pragma omp parallel for private(n, k, acc) collapse(2) schedule(guided, 4)
		for(m = 0; m < subdim_M; m++)
		{
			for(n = 0; n <  subdim_N; n++)
			{
				acc = 0.0;
				for(k = 0; k < subdim_K; k++)
				{
					acc += (double)f_im2col_input[m*subdim_K + k]
							* (double)f_filters[n*subdim_K + k];
				}
				f_output[n*subdim_M + m] = acc;
			}
		}
	}
	
	if(current->dropout_rate > 0.01f)
	{
		if(net->is_inference == 0 || (net->is_inference == 1 && net->inference_drop_mode == MC_MODEL))
		{
			dropout_select_conv(current->dropout_mask, current->a_size, current->dropout_rate);
			dropout_apply_conv(current->output, current->dropout_mask, current->a_size);
		}
		else
			dropout_scale_conv(current->output, current->a_size, current->dropout_rate);
	}
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, current->a_size*sizeof(float));
}


void naiv_backward_conv_layer(layer *current)
{
	size_t g, m, n, k;
	double acc;
	size_t subdim_M, subdim_N, subdim_K, nb_groups, batch_size;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	int back_padding[3];
	
	float *f_im2col_input, *f_delta_o, *f_prev_delta_o, *f_gradient, *f_im2col_delta_o, *f_rotated_filters;
	
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
		dropout_apply_conv(current->delta_o, current->dropout_mask, current->a_size);
	}
	
	//######################## ERROR PROPAGATION ########################
	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{
	
		//########## Preparing dw_rot_weights(K,N) ##########
		//dimensions from regular weight matrix
		subdim_N = chan_per_group_out;
		subdim_K = spatial_f_size * chan_per_group_in + 1;
		
		rotate_filter_matrix_fct(current->weights, c_param->rotated_filters, subdim_K, 
			spatial_f_size, chan_per_group_out, nb_groups*subdim_N*subdim_K);

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
		
		im2col_fct(c_param->im2col_delta_o, current->delta_o, 
			/*stride*/ c_param->int_padding[0] + 1, c_param->int_padding[1] + 1, c_param->int_padding[2] + 1,
			/*padding*/        back_padding[0]    ,         back_padding[1]    ,         back_padding[2]    ,
			/*int_padding*/ c_param->stride[0] - 1,      c_param->stride[1] - 1,      c_param->stride[2] - 1,
			                c_param->f_size[0]    ,      c_param->f_size[1]    ,      c_param->f_size[2]    ,
			/*in_size*/ current->output_dim[0]    ,  current->output_dim[1]    ,  current->output_dim[2]    ,
			/*out_size*/  current->prev_dim[0]    ,    current->prev_dim[1]    ,    current->prev_dim[2]    ,
			nb_filters, chan_per_group_out, nb_regions_out, in_image_offset, in_channel_offset,
			subdim_K*nb_regions_in, subdim_K*subdim_M, batch_size, 0);
		
		//####### Im2col_delta_o_T(M,K) x dw_rot_weights(K,N) #######
		
		for(g = 0; g < nb_groups; g++)
		{
			f_im2col_delta_o  = (float*) c_param->im2col_delta_o  + g * subdim_K*subdim_M;
			f_rotated_filters = (float*) c_param->rotated_filters + g * subdim_K*subdim_N;
			
			//Set prev_delta_o pointer depending on previous layer type
			if(current->previous->output_type == FLAT)
				f_prev_delta_o = (float*) c_param->temp_delta_o      + g * subdim_M*subdim_N;
			else
				f_prev_delta_o = (float*) current->previous->delta_o + g * subdim_M*subdim_N;
			
			#pragma omp parallel for private(n, k, acc) collapse(2) schedule(guided, 4)
			for(m = 0; m < subdim_M; m++)
			{
				for(n = 0; n < subdim_N; n++)
				{
					acc = 0.0;
					for(k = 0; k < subdim_K; k++)
					{
						acc += (double)f_im2col_delta_o[m*subdim_K + k]
						     * (double)f_rotated_filters[n*subdim_K + k];
					}
					f_prev_delta_o[n*subdim_M+m] += acc;
				}
			}
		}
		
		if(current->previous->output_type == FLAT)
		{	
			flat_dense(c_param->temp_delta_o, current->previous->delta_o, 0, nb_regions_in,
				(nb_regions_in * prev_nb_channels + 1), prev_nb_channels, batch_size,
				(nb_regions_in * prev_nb_channels + 1) * batch_size);
		}
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		//########## Subproblem dimensions ##########
		
		subdim_M = spatial_f_size * chan_per_group_in + 1;
		subdim_N = chan_per_group_out;
		subdim_K = batch_size * nb_regions_out;
		
		for(g = 0; g < nb_groups; g++)
		{
			f_im2col_input = (float*) c_param->im2col_delta_o + g * subdim_K*subdim_M;
			f_delta_o      = (float*) current->delta_o        + g * subdim_K*subdim_N;
			f_gradient     = (float*) current->gradient       + g * subdim_M*subdim_N;
		
			#pragma omp parallel for private(m, k, acc) collapse(2) schedule(dynamic, 4)
			for(n = 0; n < subdim_N; n++)
			{
				for(m = 0; m < subdim_M; m++)
				{
					acc = 0.0;
					for(k = 0; k < subdim_K; k++)
					{
						acc += (double)f_im2col_input[k*subdim_M + m]
						   * (double)f_delta_o[n*subdim_K + k];
					}
					f_gradient[n*subdim_M + m] = acc;
				}
			}
		}
		
		net->optim_update_fct(current, subdim_M, subdim_M, nb_groups*subdim_M*subdim_N);
	
		if(current->wema_replace_signal > 0)
		{
			for(k = 0; k < nb_groups*subdim_M*subdim_N; k++)
				current->FP32_weights[k] = current->ema_weights[k];
			current->wema_replace_signal = 0;
		}
	}
}


void naiv_conv_define(layer *current)
{
	current->forward = naiv_forward_conv_layer;
	current->backprop = naiv_backward_conv_layer;
}






