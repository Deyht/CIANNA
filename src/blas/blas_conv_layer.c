


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
    
	//im2col conversion fct -> one of the most complex function, go see details above
	im2col_fct_v4(c_param->im2col_input, current->input, 
		c_param->prev_size_w*c_param->prev_size_h, c_param->nb_area_w * c_param->nb_area_h 
		* c_param->flat_f_size, c_param->stride, c_param->padding, 0, c_param->prev_depth, 
		depth_padding, image_padding, current->c_network->batch_size, c_param->f_size, 
		c_param->flat_f_size, c_param->prev_size_w, c_param->nb_area_w, im2col_prev_bias);

	//Input X filters matrix multiplication for the all batch
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h), c_param->nb_filters, c_param->flat_f_size, 1.0f, 
		/*A*/ c_param->im2col_input, c_param->flat_f_size, /*B*/ c_param->filters, c_param->flat_f_size,
		0.0f, /*C*/ current->output, current->c_network->batch_size 
		* (c_param->nb_area_w*c_param->nb_area_h));
	
	//Proceed to activation of the given maps regarding the activation parameter
	current->activation(current);
}




void backward_conv_layer(layer *current)
{
	int depth_padding;
	int back_padding;
	int image_padding;
	int flat_f_size;
	
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

		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, c_param->prev_size_w * c_param->prev_size_h 
			* current->c_network->batch_size, c_param->prev_depth, c_param->f_size * c_param->f_size 
			* c_param->nb_filters, 1.0f, /*A*/c_param->im2col_delta_o, c_param->f_size 
			* c_param->f_size * c_param->nb_filters, /*B*/c_param->rotated_filters, c_param->f_size 
			* c_param->f_size*c_param->nb_filters, 0.0f, /*C*/current->previous->delta_o, 
			c_param->prev_size_w*c_param->prev_size_h*current->c_network->batch_size);

		//update gradiant regarding the previous layer activation function
		//WARNING : ONLY WORK IF PREVIOUS LAYER IS A CONV AS OUTPUT AND DELTA_O SHARE THE SAME DATA ORDER
		current->previous->deriv_activation(current->previous);

	}
	
	//########################  WEIGHTS UPDATE   ########################
	

	//based on the recovered delta_o provided by the next layer propagation
	//CUBLAS_OP_N ,in this case, is a transpose of regular input (see forward function)
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c_param->flat_f_size, c_param->nb_filters, 
		c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size, 
		current->c_network->learning_rate, c_param->im2col_input, c_param->flat_f_size, 
		current->delta_o, c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size,
		current->c_network->momentum, c_param->update, c_param->flat_f_size);
	
	update_weights(c_param->filters, c_param->update, c_param->flat_f_size 
		* c_param->nb_filters);

}





