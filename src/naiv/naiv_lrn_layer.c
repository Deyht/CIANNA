
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
static lrn_param *n_param;

// Public are in "prototypes.h"

// Private prototypes
void lrn_conv(void *i_output, void *i_input, float *local_scale, int range, 
	float k, float alpha, float beta, int b_size, int nb_channel, size_t flat_a_size);
void lrn_conv_back(void *i_output, void *i_input, void *i_delta_output, 
	void *i_delta_input, float *local_scale, int range, 
	float k, float alpha, float beta, int b_size, int nb_channel, size_t flat_a_size);
void naiv_forward_lrn_layer(layer *current);
void naiv_backward_lrn_layer(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead


void lrn_conv(void *i_output, void *i_input,
	float *local_scale, int range, float k, float alpha, float beta,
	int b_size, int nb_channel, size_t flat_a_size)
{
	size_t i;
	
	float* input = (float*) i_input;
	float* output = (float*) i_output;
	int channel_offset = flat_a_size*b_size;
	
	#pragma omp parallel for schedule(guided, 2)
	for(i = 0; i < flat_a_size*nb_channel*b_size; i++)
	{
		int channel_id, min_ch, max_ch, j;
		float l_val, local_sum = 0.0f, l_local_scale;
		
		channel_id = i/(channel_offset);
		
		min_ch = fmax(0, channel_id-range/2);
		max_ch = fmin(nb_channel-1, channel_id+range/2);
		
		for(j = min_ch; j <= max_ch; j++)
		{
			l_val = (float) input[i+(j-channel_id)*channel_offset];
			local_sum += l_val*l_val;
		}
		
		l_local_scale = k + alpha*local_sum/range;
		
		output[i] = ((float)input[i]/powf(l_local_scale, beta));
		
		if(local_scale != NULL)
			local_scale[i] = l_local_scale;
	}
}


void lrn_conv_back(
	void *i_output, void *i_input, void *i_delta_output, void *i_delta_input,
	float *local_scale, int range, float k, float alpha, float beta,
	int b_size, int nb_channel, size_t flat_a_size)
{
	size_t i;
	float* input = (float*) i_input;
	float* output = (float*) i_output;
	float* delta_input = (float*) i_delta_input;
	float* delta_output = (float*) i_delta_output;
	int channel_offset = flat_a_size*b_size;
	
	#pragma omp parallel for schedule(guided, 2)
	for(i = 0; i < flat_a_size*nb_channel*b_size; i++)
	{
		int channel_id, min_ch, max_ch, l_id, j;
		float local_sum = 0.0f;
	
		channel_id = i/(channel_offset);
		
		min_ch = fmax(0, channel_id-range/2);
		max_ch = fmin(nb_channel-1, channel_id+range/2);
		
		for(j = min_ch; j <= max_ch; j++)
		{
			l_id = i+(j-channel_id)*channel_offset;
			local_sum += (float)delta_output[l_id]*(float)output[l_id]/(float)local_scale[l_id];
		}
		
		delta_input[i] += ((float)delta_output[i]/powf(local_scale[i],beta)
			- 2.0f*alpha*beta*(float)input[i]*local_sum/range);
	}
}


void naiv_forward_lrn_layer(layer *current)
{
	int i;
	size_t flat_output_dim = 1;

	network* net = current->c_network;
	n_param = (lrn_param*)current->param;		
	//Previous verification should ensure that it is not the first layer 
	current->input = current->previous->output;
	
	if(current->output_type == FLAT)
		flat_output_dim = current->output_dim[3] + 1;
	else
	{
		for(i = 0; i < 4; i++)
			flat_output_dim *= current->output_dim[i];
		lrn_conv(current->output, current->input, n_param->local_scale, n_param->range, n_param->k, 
			n_param->alpha, n_param->beta, net->batch_size, current->output_dim[3], current->a_dim);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, flat_output_dim * net->batch_size * sizeof(float));
}

void naiv_backward_lrn_layer(layer *current)
{
	network* net = current->c_network;
	n_param = (lrn_param*)current->param;	
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == FLAT)
	{

	}
	else
	{
		lrn_conv_back(current->output, current->input, current->delta_o, current->previous->delta_o,
			n_param->local_scale, n_param->range, n_param->k, n_param->alpha, n_param->beta, 
			net->batch_size, current->output_dim[3], current->a_dim);
	}
}

void lrn_define(layer *current)
{
	current->forward = naiv_forward_lrn_layer;
	current->backprop = naiv_backward_lrn_layer;
}


