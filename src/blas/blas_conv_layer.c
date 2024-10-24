

/*
	Copyright (C) 2024 David Cornu
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


void blas_forward_conv_layer(layer *current)
{
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
	im2col_fct(c_param->im2col_input, current->input, 
		c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2], 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * c_param->flat_f_size, 
		c_param->stride[0], c_param->stride[1], c_param->stride[2],
		c_param->padding[0], c_param->padding[1], c_param->padding[2],
		c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2],
		c_param->prev_depth, depth_padding, image_padding, net->batch_size, 
		c_param->f_size[0], c_param->f_size[1], c_param->f_size[2], c_param->flat_f_size, 
		c_param->prev_size[0], c_param->prev_size[1], c_param->prev_size[2], 
		c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], im2col_prev_bias, 1);

	//Input X filters matrix multiplication for the all batch
	
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, net->batch_size 
		* (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]), c_param->nb_filters, c_param->flat_f_size, 1.0f, 
		/*A*/ c_param->im2col_input, c_param->flat_f_size, /*B*/ c_param->filters, c_param->flat_f_size, 0.0f, 
		/*C*/ current->output, net->batch_size * (c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]));
	
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


void blas_backward_conv_layer(layer *current)
{
	int k;
	int depth_padding;
	int back_padding[3];
	int image_padding;
	int flat_f_size;
	void *c_prev_delta_o;
	
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
			back_padding[k] =  c_param->f_size[k] - c_param->padding[k] - 1;
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
		
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
			c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size, c_param->prev_depth, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, 1.0f, /*A*/c_param->im2col_delta_o, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, /*B*/c_param->rotated_filters, 
			c_param->f_size[0]*c_param->f_size[1]*c_param->f_size[2]*c_param->nb_filters, 0.0f, /*C*/c_prev_delta_o, 
			c_param->prev_size[0]*c_param->prev_size[1]*c_param->prev_size[2]*net->batch_size);
		
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
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c_param->flat_f_size, c_param->nb_filters, 
		c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size, 
		current->c_network->learning_rate/current->c_network->batch_size, c_param->im2col_input, c_param->flat_f_size, 
		current->delta_o, c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size,
		current->c_network->momentum, c_param->update, c_param->flat_f_size);
		
		update_weights(c_param->filters, c_param->update, net->learning_rate*net->weight_decay, 
			0, c_param->flat_f_size*c_param->nb_filters);
	}
}


void blas_conv_define(layer *current)
{
	current->forward = blas_forward_conv_layer;
	current->backprop = blas_backward_conv_layer;
}




