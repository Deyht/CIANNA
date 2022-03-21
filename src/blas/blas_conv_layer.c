


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


void blas_conv_define(layer *current)
{
	current->forward = forward_conv_layer;
	current->backprop = backward_conv_layer;
}


void forward_conv_layer(layer *current)
{
	double c_dr, w_alpha;
	int depth_padding;
	int image_padding;
	int im2col_prev_bias;
	
	network* net = current->c_network;

	if(net->length == 0)
		return;
	c_param = (conv_param*) current->param;
	
	if(current->previous == NULL)
	{
		//if previous layer is input layer then remove the added bias on the image
		//and interpret it as continuous RGB images
		//size in line format
		depth_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2];
		image_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2] * c_param->prev_depth;
		current->input = net->input;
		im2col_prev_bias = 1;
	}
	else
	{
		//if previous layer is a CONV (or pool) then the format is all image in R, then all image in B, ...
		//it also not contain a bias directly in the image
		depth_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2] * net->batch_size;
		image_padding = c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2];
		im2col_prev_bias = 0;
		current->input = current->previous->output;
	}
	
	//im2col conversion fct -> one of the most complex function, go see details above
	im2col_fct_v5(c_param->im2col_input,
		current->input, c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2], 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * 
		c_param->flat_f_size, c_param->stride[0], c_param->stride[1], c_param->stride[2],
		c_param->padding[0], c_param->padding[1], c_param->padding[2], 0, 0 ,0, 
		c_param->prev_depth, depth_padding, image_padding, net->batch_size, 
		c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], c_param->flat_f_size, 
		c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
		c_param->nb_area[0], c_param->nb_area[1], im2col_prev_bias, 1);
	
	if(net->is_inference && net->inference_drop_mode == AVG_MODEL && current->previous != NULL)
	{
		if(current->previous->type == CONV || current->previous->type == POOL)
			c_dr = current->previous->dropout_rate;
		else
			c_dr = 0.0;
		c_dr = 1.0 - (((c_param->flat_f_size-1)*(1.0-c_dr) + 1)/c_param->flat_f_size);
		//w_alpha = (1.0f - c_dr); //account for the bias node that is never dropped
		w_alpha = (1.0/(1.0 + c_dr));
	}
	else
		w_alpha = 1.0;

	//Input X filters matrix multiplication for the all batch
	
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, net->batch_size 
		* (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]), c_param->nb_filters, c_param->flat_f_size, w_alpha, 
		/*A*/ c_param->im2col_input, c_param->flat_f_size, /*B*/ c_param->filters, c_param->flat_f_size,
		0.0f, /*C*/ current->output, net->batch_size 
		* (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]));
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
	
	if(current->dropout_rate > 0.01f && (!net->is_inference || net->inference_drop_mode == MC_MODEL))
	{
		dropout_select_conv(c_param->dropout_mask, c_param->nb_filters 
			* (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), current->dropout_rate);	
		
		dropout_apply_conv(current->output, net->batch_size, 
			(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), c_param->dropout_mask, 
			c_param->nb_filters * (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]));
	}
}


void backward_conv_layer(layer *current)
{
	int k;
	int depth_padding;
	int back_padding[3];
	int image_padding;
	int flat_f_size;
	
	network* net = current->c_network;
	
	c_param = (conv_param*) current->param;
	
	if(current->dropout_rate > 0.01f)
	{
		dropout_apply_conv(current->delta_o, net->batch_size, 
			(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), c_param->dropout_mask, 
			c_param->nb_filters * (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]));
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
		
		im2col_fct_v5(c_param->im2col_delta_o,
			current->delta_o, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], 
			(c_param->prev_size[0] * c_param->prev_size[1] * c_param->prev_size[2]) * flat_f_size, 
			1, 1, 1, back_padding[0], back_padding[1], back_padding[2], 
			c_param->stride[0] - 1 , c_param->stride[1] - 1 , c_param->stride[2] - 1,
			c_param->nb_filters, depth_padding, image_padding, net->batch_size,
			c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], flat_f_size, 
			c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
			c_param->prev_size[0], c_param->prev_size[1], 0, 0);
			
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
			c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size, c_param->prev_depth, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, 1.0f, /*A*/c_param->im2col_delta_o, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, /*B*/c_param->rotated_filters, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, 0.0f, /*C*/current->previous->delta_o, 
			c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size);
		
		//update gradiant regarding the previous layer activation function
		//WARNING : ONLY WORK IF PREVIOUS LAYER IS A CONV AS OUTPUT AND DELTA_O SHARE THE SAME DATA ORDER
		current->previous->deriv_activation(current->previous);
	}
	
	//########################  WEIGHTS UPDATE   ########################
	if(!current->frozen)
	{
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c_param->flat_f_size, c_param->nb_filters, 
		c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size, 
		current->c_network->learning_rate, c_param->im2col_input, c_param->flat_f_size, 
		current->delta_o, c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size,
		current->c_network->momentum, c_param->update, c_param->flat_f_size);
		
		update_weights(c_param->filters, c_param->update, c_param->flat_f_size*c_param->nb_filters);
	}
}







