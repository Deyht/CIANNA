
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
void blas_forward_conv_layer(layer *current);
void blas_backward_conv_layer(layer *current);


void blas_forward_conv_layer(layer *current)
{
	size_t g;
	size_t subdim_M, subdim_N, subdim_K, nb_groups;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	
	float *f_im2col_input, *f_filters, *f_output;
	
	network* net = current->c_network;
	c_param = (conv_param*) current->param;
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
	
	subdim_M = net->batch_size * nb_regions_out;
	subdim_N = chan_per_group_out;
	subdim_K = spatial_f_size * chan_per_group_in + 1;
	
	//########## Preparing Im2col(K,M) ##########
	
	//im2col conversion fct -> one of the most complex function, go see details above
	im2col_fct(c_param->im2col_input, current->input, 
		     c_param->stride[0],      c_param->stride[1],      c_param->stride[2],
		    c_param->padding[0],     c_param->padding[1],     c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		     c_param->f_size[0],      c_param->f_size[1],      c_param->f_size[2],
		   current->prev_dim[0],    current->prev_dim[1],    current->prev_dim[2],
		 current->output_dim[0],  current->output_dim[1],  current->output_dim[2],
		prev_nb_channels, chan_per_group_in, nb_regions_in, in_image_offset, in_channel_offset,
		subdim_K*nb_regions_out, subdim_K*subdim_M, net->batch_size, 1);

	//Input X filters matrix multiplication for the all batch
	
	for(g = 0; g < nb_groups; g++)
	{
		f_im2col_input = (float*)c_param->im2col_input + g * subdim_K*subdim_M;
		f_output       = (float*)current->output       + g * subdim_M*subdim_N;
				
		if(net->is_inference == 1 && (net->use_wema && !net->inference_only))
			f_filters = current->ema_weights  + g * subdim_K*subdim_N;
		else
			f_filters = current->FP32_weights + g * subdim_K*subdim_N;
	
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
			subdim_M, subdim_N, subdim_K, 1.0f,
			/*A*/f_im2col_input, /*ldA*/subdim_K,
			/*B*/f_filters     , /*ldB*/subdim_K, 0.0f,
			/*C*/f_output      , /*ldC*/subdim_M);
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


void blas_backward_conv_layer(layer *current)
{
	size_t g, k;
	size_t subdim_M, subdim_N, subdim_K, nb_groups;
	size_t in_image_offset, in_channel_offset;
	size_t spatial_f_size, nb_filters, prev_nb_channels;
	size_t chan_per_group_in, chan_per_group_out;
	size_t nb_regions_in, nb_regions_out;
	int back_padding[3];
	
	float *f_im2col_input, *f_delta_o, *f_prev_delta_o, *f_gradient, *f_im2col_delta_o, *f_rotated_filters;
	
	network* net = current->c_network;
	c_param = (conv_param*) current->param;
	nb_groups = c_param->nb_groups;
	
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
			spatial_f_size, chan_per_group_out, nb_groups*subdim_N*subdim_K);;


		//########## Subproblem dimensions ##########
		
		subdim_M = net->batch_size * nb_regions_in;
		subdim_N = chan_per_group_in;
		subdim_K = spatial_f_size * chan_per_group_out;
		
		//########## Preparing Im2col_delta_o(K,M) ##########

		//Warning : the convolution processed is reversed using full convolution with padding
		//therefore "in" and "out" variables are inverted regarding im2col function arguments
		in_image_offset   = nb_regions_out;
		in_channel_offset = nb_regions_out * net->batch_size;
		
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
			subdim_K*nb_regions_in, subdim_K*subdim_M, net->batch_size, 0);
		
		//####### Im2col_delta_o_T(M,K) x dw_rot_weights(K,N) #######
		
		if(current->previous->output_type == FLAT)
			memset(c_param->temp_delta_o, 0, nb_groups*subdim_M*subdim_N*sizeof(float));
		
		for(g = 0; g < nb_groups; g++)
		{
			f_im2col_delta_o  = (float*) c_param->im2col_delta_o  + g * subdim_K*subdim_M;
			f_rotated_filters = (float*) c_param->rotated_filters + g * subdim_K*subdim_N;
			
			//Set prev_delta_o pointer depending on previous layer type
			if(current->previous->output_type == FLAT)
				f_prev_delta_o = (float*) c_param->temp_delta_o      + g * subdim_M*subdim_N;
			else
				f_prev_delta_o = (float*) current->previous->delta_o + g * subdim_M*subdim_N;
		
			cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				subdim_M, subdim_N, subdim_K, 1.0,
				/*A*/f_im2col_delta_o , /*ldA*/subdim_K,
				/*B*/f_rotated_filters, /*ldB*/subdim_K, 1.0,
				/*C*/f_prev_delta_o   , /*ldC*/subdim_M);
		}
		
		if(current->previous->output_type == FLAT)
		{	
			flat_dense_back(c_param->temp_delta_o, current->previous->delta_o, nb_regions_in,
				(nb_regions_in * prev_nb_channels + 1), prev_nb_channels, net->batch_size,
				(nb_regions_in * prev_nb_channels + 1) * net->batch_size);
		}
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		//########## Subproblem dimensions ##########
		
		subdim_M = spatial_f_size * chan_per_group_in + 1;
		subdim_N = chan_per_group_out;
		subdim_K = net->batch_size * nb_regions_out;
		
		for(g = 0; g < nb_groups; g++)
		{
			f_im2col_input = (float*) c_param->im2col_input + g * subdim_K*subdim_M;
			f_delta_o      = (float*) current->delta_o      + g * subdim_K*subdim_N;
			f_gradient     = (float*) current->gradient     + g * subdim_M*subdim_N;
			
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
				subdim_M, subdim_N, subdim_K, 1.0f, 
				/*A*/f_im2col_input, /*ldA*/subdim_M,
				/*B*/f_delta_o     , /*ldB*/subdim_K, 0.0f,
				/*C*/f_gradient    , /*ldC*/subdim_M);
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


void blas_conv_define(layer *current)
{
	current->forward = blas_forward_conv_layer;
	current->backprop = blas_backward_conv_layer;
}




