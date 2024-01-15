

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

static norm_param *n_param;

//public are in prototypes.h

//#####################################################
//       Layer normalization related templates
//#####################################################


int id_to_conv_fmt(int id, int block_id, int group_size, int nb_group, int flat_a_size, int batch_size)
{
	int group_id = block_id % nb_group;
	int batch_id = block_id / nb_group;
	
	int in_group_id = id / flat_a_size;
	int map_pos_id = id % flat_a_size;
	
	return batch_id*flat_a_size + (group_id*group_size + in_group_id)*flat_a_size*batch_size + map_pos_id;
}


void reduce_group_mean_conv_fct(float *input, float *group_mean,
	int group_size, int nb_group, int flat_a_size, int batch_size, int sum_div)
{
	int i, j;
	double sum;
	
	#pragma omp parallel for private(j, sum) schedule(guided,2)
	for(i = 0; i < nb_group*batch_size; i++)
	{
		sum = 0.0;
		for(j = 0; j < group_size*flat_a_size; j++)
		{
			sum += input[id_to_conv_fmt(j, i, group_size, nb_group, flat_a_size, batch_size)];
		}
		group_mean[i] = sum/(sum_div);
	}
}

void reduce_group_var_conv_fct(float *input, float *group_var, float *group_mean,
	int group_size, int nb_group, int flat_a_size, int batch_size, int sum_div)
{
	int i, j;
	float l_val;
	double sum;
	
	#pragma omp parallel for private(j, l_val, sum) schedule(guided,2)
	for(i = 0; i < nb_group*batch_size; i++)
	{
		sum = 0.0;
		for(j = 0; j < group_size*flat_a_size; j++)
		{
			l_val = input[id_to_conv_fmt(j, i, group_size, nb_group, flat_a_size, batch_size)];
			sum += (l_val - group_mean[i])*(l_val - group_mean[i]);
		}
		group_var[i] = sum/(sum_div);
	}
}


void reduce_group_dgamma_conv_fct(float *input, float *delta_output, float *d_gamma,
	float *group_var, float *group_mean, int group_size, int nb_group, int flat_a_size, int batch_size)
{	
	int i, j;
	float eps = 0.001f;
	double sum;
	
	#pragma omp parallel for private(j, sum) schedule(guided,2)
	for(i = 0; i < nb_group*batch_size; i++)
	{
		sum = 0.0;
		for(j = 0; j < group_size*flat_a_size; j++)
		{
			sum += delta_output[id_to_conv_fmt(j, i, group_size, nb_group, flat_a_size, batch_size)]
				* (input[id_to_conv_fmt(j, i, group_size, nb_group, flat_a_size, batch_size)] - group_mean[i]);
		}
		d_gamma[i] = sum*(1.0f/sqrt(group_var[i]+eps));
	}
}


void group_normalization_conv_fct(float *output, float *input, float *gamma, float *beta, float *group_mean, float *group_var,
	int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off)
{
	/* Could be optimized with advanced multi-thread reduction */
	int i, j;
	float l_val, eps = 0.001f;
	float mean = 0.0f, var = 0.0f;
	int filter_offset = flat_a_size*b_size;
	int group_id, batch_id;
	int in_group_id, map_pos_id, conv_id;
	
	#pragma omp parallel for private(i, j, group_id, batch_id, in_group_id, map_pos_id, conv_id, \
		mean, var, l_val) schedule(guided,2)
	for(j = 0; j < nb_group*b_size; j++)
	{
		for(i = 0; i < flat_a_size*group_size; i++)
		{
			group_id = j % nb_group;
			batch_id = j / nb_group;
			
			in_group_id = i / flat_a_size;
			map_pos_id = i % flat_a_size;
			
			conv_id = batch_id*flat_a_size + (group_id*group_size + in_group_id)*filter_offset + map_pos_id;
			
			if(batch_id < b_length)
			{
				mean = group_mean[batch_id*nb_group + group_id];
				var  = group_var[batch_id*nb_group + group_id];
				
				l_val = input[conv_id];
				if(group_id < nb_group - set_off)
					output[conv_id] = (gamma[group_id]*((l_val - mean)/sqrt(var + eps)) + beta[group_id]);
				else
					output[conv_id] = l_val;
			}
			else
				output[conv_id] = 0.0f;
		}
	}
}


void group_normalization_conv_back_fct(
	float *input, float *delta_output, float *delta_input, float *gamma, float *beta, float *d_gamma, float * d_beta, float *group_mean,
	float *group_var, int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off)
{
	int i, j;

	float eps = 0.001f;
	float mean = 0.0f, var = 0.0f;
	float l_d_gamma, l_d_beta;
	int filter_offset = flat_a_size*b_size;
	int group_id, batch_id;
	int in_group_id, map_pos_id, conv_id;
	
	#pragma omp parallel for private(i, j, group_id, batch_id, in_group_id, map_pos_id, conv_id, \
		mean, var, l_d_gamma, l_d_beta) schedule(guided,2)
	for(j = 0; j < nb_group*b_size; j++)
	{
		for(i = 0; i < flat_a_size*group_size; i++)
		{	
			group_id = j % nb_group;
			batch_id = j / nb_group;
			
			in_group_id = i / flat_a_size;
			map_pos_id = i % flat_a_size;
			
			conv_id = batch_id*flat_a_size + (group_id*group_size + in_group_id)*filter_offset + map_pos_id;
			
			if(batch_id < b_length)
			{
				mean = group_mean[batch_id*nb_group + group_id];
				var  = group_var[batch_id*nb_group + group_id];
				l_d_gamma = d_gamma[batch_id*nb_group + group_id];
				l_d_beta  = d_beta[batch_id*nb_group + group_id];
				
				if(group_id < nb_group - set_off)
					delta_input[conv_id] = ((1.0f/(group_size*flat_a_size)) * gamma[group_id] * (1.0f/sqrt(var + eps))
						* (group_size*flat_a_size*delta_output[conv_id] - l_d_beta
						- (input[conv_id] - mean) * (1.0f/sqrt(var + eps))*l_d_gamma));
				else
					delta_input[conv_id] = delta_output[conv_id];
			}
			else
				delta_input[conv_id] = 0.0f;
		}
	}
}

void naiv_forward_norm_layer(layer *current)
{
	n_param = (norm_param*)current->param;
	network* net = current->c_network;
	
	current->input = current->previous->output;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{
		reduce_group_mean_conv_fct(current->input, n_param->mean, n_param->group_size, n_param->nb_group, 
			n_param->dim_offset, net->batch_size, n_param->dim_offset*n_param->group_size);

		reduce_group_var_conv_fct(current->input, n_param->var, n_param->mean, n_param->group_size, 
			n_param->nb_group, n_param->dim_offset, net->batch_size, n_param->dim_offset*n_param->group_size);

		group_normalization_conv_fct(current->output, current->input, n_param->gamma, n_param->beta, n_param->mean, n_param->var, 
			net->length, net->batch_size, n_param->group_size, n_param->nb_group, n_param->n_dim, n_param->dim_offset, n_param->set_off);
	}
}

void naiv_backward_norm_layer(layer *current)
{
	int i, j;
	double sum_dgamma = 0.0, sum_dbeta = 0.0;
	n_param = (norm_param*)current->param;
	network* net = current->c_network;
	
	if(current->previous->type == DENSE)
	{
	
	}
	else
	{
		reduce_group_mean_conv_fct(current->delta_o, n_param->d_beta, n_param->group_size, 
			n_param->nb_group, n_param->dim_offset, net->batch_size, 1);
	
		reduce_group_dgamma_conv_fct(current->input, current->delta_o, n_param->d_gamma, n_param->var, 
			n_param->mean, n_param->group_size, n_param->nb_group, n_param->dim_offset, net->batch_size);
	
		group_normalization_conv_back_fct(current->input, current->delta_o, current->previous->delta_o, n_param->gamma, n_param->beta, 
			n_param->d_gamma, n_param->d_beta, n_param->mean, n_param->var, net->length, net->batch_size, n_param->group_size, 
			n_param->nb_group, n_param->n_dim, n_param->dim_offset, n_param->set_off);
				
		if(!current->frozen)
		{
			for(j = 0; j < n_param->nb_group - n_param->set_off; j++)
			{
				sum_dgamma = 0.0f;
				sum_dbeta = 0.0f;
				for(i = 0; i < net->batch_size; i++)
				{
					sum_dgamma += n_param->d_gamma[i*n_param->nb_group + j];
					sum_dbeta  += n_param->d_beta[i*n_param->nb_group + j];
				}
				//could add momentum
				n_param->gamma_update[j] = net->momentum*n_param->gamma_update[j] 
					+ net->learning_rate*(sum_dgamma/net->batch_size);
					/*+ 0.0f*net->weight_decay*(1.0f-n_param->gamma[j]));*/
				n_param->beta_update[j] = net->momentum*n_param->beta_update[j]  
					+ net->learning_rate*(sum_dbeta/net->batch_size);
					/*+ 0.0f*net->weight_decay*n_param->beta[j]);*/
				
				n_param->gamma[j] -= n_param->gamma_update[j];
				n_param->beta[j] -= n_param->beta_update[j]; 
			}
		}
	}
	
	current->previous->deriv_activation(current->previous);
}

void naiv_norm_define(layer *current)
{
	current->forward = naiv_forward_norm_layer;
	current->backprop = naiv_backward_norm_layer;
}


