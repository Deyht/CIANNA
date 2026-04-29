
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
static norm_param *n_param;

// Public are in "prototypes.h"

// Private prototypes
void reduce_group_var_conv_fct(float *input, float *group_var, float *group_mean,
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div);
void reduce_group_dgamma_conv_fct(float *input, float *delta_output, float *d_gamma,
	float *group_var, float *group_mean, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size);
void group_normalization_conv_fct(float *output, float *input, float *gamma, float *beta, float *group_mean, float *group_var,
	size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t flat_a_size);
void group_normalization_conv_back_fct(float *input, float *delta_output, float *delta_input, float *gamma, 
	float *A, float *B, float *group_mean, float *group_var, size_t b_length, size_t b_size, size_t group_size, 
	size_t nb_group, size_t flat_a_size);


void reduce_group_mean_conv_fct(float *input, float *group_mean,
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div)
{
	size_t i, j;
	size_t group_id, batch_id, in_group_id, map_pos_id, conv_id;
	double sum;
	
	#pragma omp parallel for private(j, group_id, batch_id, in_group_id, map_pos_id, conv_id, sum) schedule(guided,2)
	for(i = 0; i < (size_t)(nb_group*batch_size); i++)
	{
		group_id = i % nb_group;
		batch_id = i / nb_group;

		sum = 0.0;
		for(j = 0; j < group_size*flat_a_size; j++)
		{
			in_group_id = j / flat_a_size;
			map_pos_id  = j % flat_a_size;
			conv_id     =  batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;
		
			sum += input[conv_id];
		}
		group_mean[i] = sum/sum_div;
	}
}


void reduce_group_var_conv_fct(float *input, float *group_var, float *group_mean,
	size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div)
{
	size_t i, j;
	size_t group_id, batch_id, in_group_id, map_pos_id, conv_id;
	float l_val;
	double sum;
	
	#pragma omp parallel for private(j, group_id, batch_id, in_group_id, map_pos_id, conv_id, l_val, sum) schedule(guided,2)
	for(i = 0; i < nb_group*batch_size; i++)
	{
		group_id = i % nb_group;
		batch_id = i / nb_group;
		
		sum = 0.0;
		for(j = 0; j < group_size*flat_a_size; j++)
		{
			in_group_id = j / flat_a_size;
			map_pos_id  = j % flat_a_size;
			conv_id     =  batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;
			
			l_val = input[conv_id];
			sum += (l_val - group_mean[i])*(l_val - group_mean[i]);
		}
		group_var[i] = sum/sum_div;
	}
}


void reduce_group_dgamma_conv_fct(float *input, float *delta_output, float *d_gamma,
	float *group_var, float *group_mean, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size)
{	
	size_t i, j;
	float eps = 0.000001f;
	size_t conv_id, feature_id, batch_id, group_id, group_pos;
	size_t nb_features = group_size * nb_group;
	double sum;
	
	#pragma omp parallel for private(j, conv_id, feature_id, batch_id, group_id, group_pos, sum) schedule(guided,2)
	for(i = 0; i < nb_features*batch_size; i++)
	{
		feature_id = i % nb_features;
		batch_id   = i / nb_features;
		group_id   = feature_id/group_size;
		group_pos  = batch_id * nb_group + group_id;
		conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;
	
		sum = 0.0;
		for(j = 0; j < flat_a_size; j++)
			sum += delta_output[conv_id + j] * (input[conv_id + j] - group_mean[group_pos]);
		d_gamma[i] = sum*(1.0f/sqrt(group_var[group_pos]+eps));
	}
}


void group_normalization_conv_fct(float *output, float *input, float *gamma, float *beta, float *group_mean, float *group_var,
	size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t flat_a_size)
{
	/* Could be optimized with advanced multi-thread reduction */
	size_t i, j;
	float l_val, eps = 0.000001f;
	float mean = 0.0f, var = 0.0f;
	size_t filter_offset = flat_a_size*b_size;
	size_t group_id, batch_id, in_group_id, map_pos_id, feature_id, conv_id;
	
	#pragma omp parallel for private(i, group_id, batch_id, in_group_id, map_pos_id,\
		 feature_id, conv_id, mean, var, l_val) schedule(guided,2)
	for(j = 0; j < nb_group*b_size; j++)
	{
		group_id = j % nb_group;
		batch_id = j / nb_group;
		
		for(i = 0; i < flat_a_size*group_size; i++)
		{
			
			in_group_id = i / flat_a_size;
			map_pos_id  = i % flat_a_size;
			feature_id  = group_id * group_size + in_group_id;
			conv_id     = batch_id * flat_a_size + (group_id * group_size + in_group_id) * filter_offset + map_pos_id;
			
			if(batch_id < b_length)
			{
				mean = group_mean[batch_id*nb_group + group_id];
				var  = group_var[batch_id*nb_group + group_id];
				
				l_val = input[conv_id];
				output[conv_id] = (gamma[feature_id]*((l_val - mean)/sqrt(var + eps)) + beta[feature_id]);
			}
			else
				output[conv_id] = 0.0f;
		}
	}
}


void group_normalization_conv_back_fct(
	float *input, float *delta_output, float *delta_input, float *gamma, float *A, float *B, float *group_mean,
	float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t flat_a_size)
{
	size_t i, j;

	float eps = 0.000001f;
	float mean = 0.0f, var = 0.0f;
	float l_A, l_B;
	size_t filter_offset = flat_a_size*b_size;
	size_t group_id, batch_id, in_group_id, map_pos_id, feature_id, conv_id;
	
	#pragma omp parallel for private(i, group_id, batch_id, in_group_id, map_pos_id, \
		feature_id, conv_id, mean, var, l_A, l_B) schedule(guided,2)
	for(j = 0; j < nb_group*b_size; j++)
	{
		group_id = j % nb_group;
		batch_id = j / nb_group;
		
		for(i = 0; i < flat_a_size*group_size; i++)
		{
			in_group_id = i / flat_a_size;
			map_pos_id  = i % flat_a_size;
			feature_id  = group_id * group_size + in_group_id;
			conv_id     = batch_id * flat_a_size + (group_id * group_size + in_group_id) * filter_offset + map_pos_id;
			
			if(batch_id < b_length)
			{
				mean = group_mean[batch_id*nb_group + group_id];
				var  = group_var[batch_id*nb_group + group_id];
				l_A = A[batch_id*nb_group + group_id];
				l_B = B[batch_id*nb_group + group_id];
				
				delta_input[conv_id] += ((1.0f/(group_size*flat_a_size)) * (1.0f/sqrt(var + eps))
					* (gamma[feature_id]*group_size*flat_a_size*delta_output[conv_id] - l_A
					- (input[conv_id] - mean) * (1.0f/sqrt(var + eps)) * l_B));
			}
		}
	}
}


void forward_norm_layer(layer *current)
{
	int i;
	size_t dim_offset = 1, nb_features;
	float *l_gamma, *l_beta;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;
	//Previous verification should ensure that it is not the first layer
	current->input = current->previous->output;

	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		if(net->is_inference == 1 && (net->use_wema && !net->inference_only))
		{
			l_gamma = current->ema_weights;
			l_beta = ((float*)current->ema_weights) + nb_features;
		}
		else
		{
			l_gamma = n_param->gamma;
			l_beta = n_param->beta;
		}
		
		reduce_group_mean_conv_fct(current->input, n_param->mean, n_param->group_size, n_param->nb_group, 
			dim_offset, net->batch_size, dim_offset*n_param->group_size);
		
		reduce_group_var_conv_fct(current->input, n_param->var, n_param->mean, n_param->group_size, 
			n_param->nb_group, dim_offset, net->batch_size, dim_offset*n_param->group_size);

		group_normalization_conv_fct(current->output, current->input, l_gamma, l_beta, n_param->mean, n_param->var, 
			net->length, net->batch_size, n_param->group_size, n_param->nb_group, dim_offset);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, current->a_size * sizeof(float));
}


void backward_norm_layer(layer *current)
{
	int i, j, k;
	size_t dim_offset = 1, nb_features;
	double sum_dgamma = 0.0, sum_dbeta = 0.0;
	
	network* net = current->c_network;
	n_param = (norm_param*)current->param;

	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		reduce_group_mean_conv_fct(current->delta_o, n_param->d_beta, 1, nb_features, dim_offset, net->batch_size, 1);
	
		reduce_group_dgamma_conv_fct(current->input, current->delta_o, n_param->d_gamma, n_param->var, 
			n_param->mean, n_param->group_size, n_param->nb_group, dim_offset, net->batch_size);
		
		for(i = 0; i < net->batch_size; i++)
		{
			for(j = 0; j < n_param->nb_group; j++)
			{
				n_param->temp_A[i*n_param->nb_group + j] = 0.0f;
				n_param->temp_B[i*n_param->nb_group + j] = 0.0f;
				for(k = 0; k < n_param->group_size; k++)
				{	
					 n_param->temp_A[i*n_param->nb_group + j] += n_param->gamma[j*n_param->group_size + k]
					 	*((float*)n_param->d_beta)[i*nb_features + j*n_param->group_size + k];
					 n_param->temp_B[i*n_param->nb_group + j] += n_param->gamma[j*n_param->group_size + k]
					 	*((float*)n_param->d_gamma)[i*nb_features + j*n_param->group_size + k]; 
				}
			}
		}
		
		group_normalization_conv_back_fct(current->input, current->delta_o, current->previous->delta_o, n_param->gamma, 
			n_param->temp_A, n_param->temp_B, n_param->mean, n_param->var, net->length, net->batch_size, n_param->group_size, 
			n_param->nb_group, dim_offset);
	}
	
	if(!current->frozen)
	{
		for(j = 0; j < (int)nb_features; j++)
		{
			sum_dgamma = 0.0f;
			sum_dbeta = 0.0f;
			for(i = 0; i < net->batch_size; i++)
			{
				sum_dgamma += ((float*)n_param->d_gamma)[i*nb_features + j];
				sum_dbeta  += ((float*)n_param->d_beta)[i*nb_features + j];
			}
			((float*)n_param->gamma_grad)[j] = sum_dgamma;
			((float*)n_param->beta_grad)[j] = sum_dbeta;
		}
		
		net->optim_update_fct(current, 0, 2*nb_features, 2*nb_features);
		//No decay for gamma and beta
		
		if(current->wema_replace_signal > 0)
		{
			for(i = 0; i < 2*nb_features; i++)
				current->FP32_weights[i] = current->ema_weights[i];
			current->wema_replace_signal = 0;
		}
	}
}


void norm_define(layer *current)
{
	current->forward = forward_norm_layer;
	current->backprop = backward_norm_layer;
}


