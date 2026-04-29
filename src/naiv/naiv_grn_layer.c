
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
static grn_param *n_param;

// Public are in "prototypes.h"

// Private prototypes
void reduce_l2norm_conv_fct(float *input, float *group_l2norm,
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size);
void reduce_grn_dgamma_conv_fct(float *input, float *d_output, float *d_gamma,
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size);
void grn_conv_fct(float *output, float *input, float *gamma, float *beta,
	float *relative_importance, int residual, size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size);
void grn_conv_back_fct(float *input, float *delta_output, float *delta_input, 
	float *gamma, float *d_gamma, float *feature_norm, float *relative_importance, int residual,
	size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size);


void reduce_l2norm_conv_fct(float *input, float *group_l2norm,
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size)
{
	size_t i, j;
	size_t conv_id, feature_id, batch_id;
	double l_val, sum;
	
	#pragma omp parallel for private(j, conv_id, feature_id, batch_id, sum) schedule(guided,2)
	for(i = 0; i < nb_features*batch_size; i++)
	{
		feature_id = i % nb_features;
		batch_id   = i / nb_features;
		conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;

		sum = 0.0;
		for(j = 0; j < sum_size; j++)
		{
			l_val = input[conv_id + j];
			sum += l_val*l_val;
		}
		group_l2norm[i] = sqrt(sum + 0.000001f);
	}
}


void reduce_grn_dgamma_conv_fct(float *input, float *d_output, float *d_gamma,
	size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size)
{
	size_t i, j;
	size_t conv_id, feature_id, batch_id;
	double sum;

	#pragma omp parallel for private(j, conv_id, feature_id, batch_id, sum) schedule(guided,2)
	for(i = 0; i < nb_features*batch_size; i++)
	{
		feature_id = i % nb_features;
		batch_id   = i / nb_features;
		conv_id    = batch_id * flat_a_size + feature_id * flat_a_size * batch_size;

		sum = 0.0;
		for(j = 0; j < sum_size; j++)
			sum += d_output[conv_id + j] * input[conv_id + j];
		d_gamma[i] = sum;
	}
}


void grn_conv_fct(float *output, float *input, float *gamma, float *beta,
	float *relative_importance, int residual, size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size)
{
	size_t i, j;
	size_t conv_id, feature_id, batch_id;
	size_t filter_offset = flat_a_size*b_size;
	double l_val, l_rel_imp;

	#pragma omp parallel for private(j, conv_id, feature_id, batch_id, l_val, l_rel_imp) schedule(guided,2)
	for(i = 0; i < nb_features*b_size; i++)
	{
		feature_id = i % nb_features;
		batch_id   = i / nb_features;

		l_rel_imp = relative_importance[batch_id*nb_features + feature_id];

		for(j = 0; j < flat_a_size; j++)
		{
			conv_id = batch_id * flat_a_size + feature_id*filter_offset + j;
			l_val = input[conv_id];
			
			if(batch_id < b_length)
			{
				output[conv_id] = gamma[feature_id] * l_val * l_rel_imp + beta[feature_id];
				if(residual)
					output[conv_id] += l_val;
			}
			else
				output[conv_id] = 0.0f;
		}
	}
}


void grn_conv_back_fct(float *input, float *delta_output, float *delta_input, 
	float *gamma, float *d_gamma, float *feature_norm, float *relative_importance, int residual,
	size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size)
{
	size_t i, j;
	size_t conv_id, feature_id, batch_id;
	size_t filter_offset = flat_a_size*b_size;
	double l_d_gamma, l_rel_imp, l_feat_norm, l_mean_eps, l_val, l_grad;
	
	#pragma omp parallel for private(j, conv_id, feature_id, batch_id, l_d_gamma, l_rel_imp, l_feat_norm, l_mean_eps, l_val, l_grad) schedule(guided,2)
	for(i = 0; i < nb_features*b_size; i++)
	{
		feature_id = i % nb_features;
		batch_id   = i / nb_features;

		l_d_gamma = d_gamma[batch_id*nb_features + feature_id];
		l_feat_norm = feature_norm[batch_id*nb_features + feature_id];
		l_rel_imp = relative_importance[batch_id*nb_features + feature_id];
		l_mean_eps = l_feat_norm / l_rel_imp;

		for(j = 0; j < flat_a_size; j++)
		{
			conv_id = batch_id * flat_a_size + feature_id*filter_offset + j;
			l_val = input[conv_id];
			l_grad = delta_output[conv_id];
			
			if(batch_id < b_length)
			{
				delta_input[conv_id] += gamma[feature_id] * l_rel_imp * l_grad
					+ gamma[feature_id] * l_d_gamma * l_val
					* (1.0f/(l_mean_eps*l_feat_norm) - 1.0f/(nb_features*l_mean_eps*l_mean_eps));
				if(residual)
					delta_input[conv_id] += l_grad;
			}
		}
	}
}


void forward_grn_layer(layer *current)
{
	size_t i, j;
	size_t dim_offset = 1, nb_features;
	double l_mean;
	float *l_gamma, *l_beta;
	float eps = 0.000001f;
	
	network* net = current->c_network;
	n_param = (grn_param*)current->param;
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
		
		reduce_l2norm_conv_fct(current->input, n_param->feature_norm, nb_features, dim_offset, net->batch_size, dim_offset);
		
		for(i = 0; i < (size_t)net->batch_size; i++)
		{
			l_mean = 0.0f;
			for(j = 0; j < nb_features; j++)
				l_mean += n_param->feature_norm[i*nb_features + j];
			l_mean /= nb_features;
			
			for(j = 0; j < nb_features; j++)
				n_param->relative_importance[i*nb_features + j] = 
					n_param->feature_norm[i*nb_features + j] / (l_mean + eps);
		}
		
		grn_conv_fct(current->output, current->input, l_gamma, l_beta, 
			n_param->relative_importance, n_param->residual, 
			net->length, net->batch_size, nb_features, dim_offset);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, current->a_size * sizeof(float));
}


void backward_grn_layer(layer *current)
{
	size_t i, j;
	size_t dim_offset = 1, nb_features = 1;
	double sum_dgamma = 0.0f, sum_dbeta = 0.0f;
	
	network* net = current->c_network;
	n_param = (grn_param*)current->param;	
	
	//Must be done here so all layers can add their contribution to current layer delta_o (merging / branching)
	current->deriv_activation(current);
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			dim_offset *= current->output_dim[i];
		nb_features = current->output_dim[3];
		
		reduce_group_mean_conv_fct(current->delta_o, n_param->d_beta, 1, nb_features, dim_offset, net->batch_size, 1);
		
		reduce_grn_dgamma_conv_fct(current->input, current->delta_o,
			n_param->d_gamma, nb_features, dim_offset, net->batch_size, dim_offset);
		//Here d_gamma is missing a multiplication with relative_importance[block_id]
		//Still, the current reduction is usefull for propagating the gradient. 
		//-> Multiplication with relative_importance[block_id] is postponed
			
		grn_conv_back_fct(current->input, current->delta_o, 
			current->previous->delta_o, n_param->gamma, n_param->d_gamma, 
			n_param->feature_norm, n_param->relative_importance, n_param->residual,
			net->length, net->batch_size, nb_features, dim_offset);
	}
	
	if(!current->frozen)
	{
		for(j = 0; j < nb_features; j++)
		{
			sum_dgamma = 0.0f;
			sum_dbeta = 0.0f;
			for(i = 0; i < (size_t)net->batch_size; i++)
			{
				sum_dgamma += ((float*)n_param->d_gamma)[i*nb_features + j]
					*n_param->relative_importance[i*nb_features + j]; //delayed relative importance multiplication
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


void grn_define(layer *current)
{
	current->forward = forward_grn_layer;
	current->backprop = backward_grn_layer;
}




