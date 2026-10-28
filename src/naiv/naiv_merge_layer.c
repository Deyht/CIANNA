
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
static merge_param *m_param;

// Public are in "prototypes.h"

// Private prototypes
void merge_add_fct(int nb_previous, int *prev_type_list, void **prev_output_list, 
	size_t chan_offset, int batch_size, void *output_new, size_t size);
void merge_add_back_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list, void *delta_o, size_t size);
void merge_concatenate_fct(int nb_previous, int *prev_type_list, void **prev_output_list,
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *output_new, size_t size);
void merge_concatenate_back_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list, 
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *delta_o, size_t size);
void merge_add_flat_fct(int nb_previous, void **prev_output_list, size_t output_dim, void *output_new, size_t size);
void merge_add_back_flat_fct(int nb_previous, int *prev_type_list,
	void **prev_delta_o_list, size_t output_dim, void *delta_o, size_t size);
void merge_concatenate_flat_fct(int nb_previous, void **prev_output_list,
	int *prev_nb_neurons, size_t output_dim, void *output_new, size_t size);
void merge_concatenate_back_flat_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list,
	int *prev_nb_neurons, size_t output_dim, void *delta_o, size_t size);

void merge_add_fct(int nb_previous, int *prev_type_list, void **prev_output_list,
	size_t chan_offset, int batch_size, void *output_new, size_t size)
{
	size_t i, j;
	size_t l_channel, l_batch, l_pixel;
	size_t chan_batch_offset = chan_offset*batch_size;
	size_t nb_channels = size/chan_batch_offset;
	float *out_new = (float*) output_new;
	float *l_prev, acc;
	
	#pragma omp parallel for private(j, l_channel, l_batch, l_pixel, l_prev, acc) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		l_channel =  i / chan_batch_offset;
		l_batch   = (i % chan_batch_offset) / chan_offset;
		l_pixel   =  i % chan_offset;
		
		acc = 0.0f;
		for(j = 0; j < nb_previous; j++)
		{
			l_prev = (float*) prev_output_list[j];
			if(prev_type_list[j] != INPUT)
				acc += l_prev[i];
			else
				acc += l_prev[l_batch*(nb_channels*chan_offset + 1) + l_channel*chan_offset + l_pixel];
		}
		out_new[i] = acc;
	}
}


void merge_add_back_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list, void *delta_o, size_t size)
{
	size_t i, j;
	float *dlt = (float*) delta_o;
	float dlt_i;
	float *l_prev;
	
	#pragma omp parallel for private(dlt_i, j, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		dlt_i = dlt[i];
		for(j = 0; j < nb_previous; j++)
		{
			if(prev_type_list[j] == INPUT)
				continue;
		
			l_prev = (float*) prev_delta_o_list[j];
			l_prev[i] += dlt_i;
		}
	}
}


void merge_concatenate_fct(int nb_previous, int *prev_type_list, void **prev_output_list,
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *output_new, size_t size)
{
	size_t i, j;
	size_t l_channel, channel_acc;
	size_t l_batch, l_pixel;
	size_t chan_batch_offset = chan_offset*batch_size;
	float *out_new = (float*) output_new;
	float *l_prev;
	
	#pragma omp parallel for private(j, l_channel, channel_acc, l_batch, l_pixel, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		l_channel =  i / chan_batch_offset;
		l_batch   = (i % chan_batch_offset) / chan_offset;
		l_pixel   =  i % chan_offset;
		
		channel_acc = 0;
		for(j = 0; j < nb_previous; j++)
		{
			if(l_channel < channel_acc + prev_nb_channels[j])
				break;
			channel_acc += prev_nb_channels[j];
		}
		
		l_prev = (float*) prev_output_list[j];
		l_channel = l_channel - channel_acc;
		
		if(prev_type_list[j] != INPUT)
			out_new[i] = l_prev[l_channel*chan_batch_offset + l_batch*chan_offset + l_pixel];
		else
			out_new[i] = l_prev[l_batch*(prev_nb_channels[j]*chan_offset + 1) + l_channel*chan_offset + l_pixel];
	}
}


void merge_concatenate_back_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list, 
	int *prev_nb_channels, size_t chan_offset, int batch_size, void *delta_o, size_t size)
{
	size_t i, j;
	size_t l_channel, channel_acc;
	size_t l_batch, l_pixel;
	size_t chan_batch_offset = chan_offset*batch_size;
	float *dlt = (float*) delta_o;
	float *l_prev;
	
	#pragma omp parallel for private(j, l_channel, channel_acc, l_batch, l_pixel, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		l_channel =  i / chan_batch_offset;
		l_batch   = (i % chan_batch_offset) / chan_offset;
		l_pixel   =  i % chan_offset;
		
		channel_acc = 0;
		
		for(j = 0; j < nb_previous; j++)
		{
			if(l_channel < channel_acc + prev_nb_channels[j])
				break;
			channel_acc += prev_nb_channels[j];
		}
		
		l_channel = l_channel - channel_acc;
		
		if(prev_type_list[j] == INPUT) //no propagation to input
			continue;
		
		l_prev = (float*) prev_delta_o_list[j];
		l_prev[l_channel*chan_batch_offset + l_batch*chan_offset + l_pixel] += dlt[i];
	}
}


void merge_add_flat_fct(int nb_previous, void **prev_output_list, size_t output_dim, void *output_new, size_t size)
{ //No specific handling for INPUT layers as they share format with FLAT layers
	size_t i, j;
	size_t feature_id;
	float *out_new = (float*) output_new;
	float *l_prev, acc;
	
	#pragma omp parallel for private(j, feature_id, l_prev, acc) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		feature_id = i % output_dim;
		if(feature_id == output_dim - 1)
			continue;
			
		acc = 0.0f;
		for(j = 0; j < nb_previous; j++)
		{
			l_prev = (float*) prev_output_list[j];
			acc += l_prev[i];
		}
		out_new[i] = acc;
	}
}


void merge_add_back_flat_fct(int nb_previous, int *prev_type_list,
	void **prev_delta_o_list, size_t output_dim, void *delta_o, size_t size)
{
	size_t i, j;
	size_t feature_id;
	float *dlt = (float*) delta_o;
	float dlt_i;
	float *l_prev;
	
	#pragma omp parallel for private(j, feature_id, dlt_i, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		feature_id = i % output_dim;
		if(feature_id == output_dim - 1)
			continue;
		
		dlt_i = dlt[i];
		for(j = 0; j < nb_previous; j++)
		{
			if(prev_type_list[j] == INPUT)
				continue;
			l_prev = (float*) prev_delta_o_list[j];
			l_prev[i] += dlt_i;
		}
	}
}


void merge_concatenate_flat_fct(int nb_previous, void **prev_output_list,
	int *prev_nb_neurons, size_t output_dim, void *output_new, size_t size)
{
	size_t i, j;
	size_t batch_id, feature_id, feature_acc;
	float *out_new = (float*) output_new;
	float *l_prev;
	
	#pragma omp parallel for private(j, batch_id, feature_id, feature_acc, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		batch_id = i / output_dim;
		feature_id = i % output_dim;
		
		if(feature_id == output_dim - 1)
			continue;
		
		feature_acc = 0;
		for(j = 0; j < nb_previous; j++)
		{
			if(feature_id < feature_acc + prev_nb_neurons[j])
				break;
			feature_acc += prev_nb_neurons[j];
		}
		
		l_prev = (float*) prev_output_list[j];
		out_new[i] = l_prev[batch_id*(prev_nb_neurons[j]+1) + (feature_id - feature_acc)];
	}
}


void merge_concatenate_back_flat_fct(int nb_previous, int *prev_type_list, void **prev_delta_o_list,
	int *prev_nb_neurons, size_t output_dim, void *delta_o, size_t size)
{
	size_t i, j;
	size_t batch_id, feature_id, feature_acc;
	float *dlt = (float*) delta_o;
	float *l_prev;
	
	#pragma omp parallel for private(j, batch_id, feature_id, feature_acc, l_prev) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		batch_id = i / output_dim;
		feature_id = i % output_dim;
		
		if(feature_id == output_dim - 1)
			continue;
		
		feature_acc = 0;
		for(j = 0; j < nb_previous; j++)
		{
			if(feature_id < feature_acc + prev_nb_neurons[j])
				break;
			feature_acc += prev_nb_neurons[j];
		}
		
		if(prev_type_list[j] == INPUT)
			continue;
		
		l_prev = (float*)prev_delta_o_list[j];
		l_prev[batch_id*(prev_nb_neurons[j]+1) + (feature_id - feature_acc)] += dlt[i];
	}
}


void forward_merge_layer(layer *current)
{
	int i;
	network *net = current->c_network;
	m_param = (merge_param*) current->param;
	
	for(i = 0; i < m_param->nb_previous; i++) //required for INPUT type layers
		m_param->prev_output_list[i] = m_param->prev_list[i]->output;
	
	if(m_param->merge_type == ADD_merge)
	{
		if(current->output_type == SPATIAL)
			merge_add_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_output_list, 
				current->a_dim, net->batch_size, current->output, current->a_size);
		else
			merge_add_flat_fct(m_param->nb_previous, m_param->prev_output_list,
				current->a_biased_dim, current->output, current->a_size);
	}
	else
	{
		if(current->output_type == SPATIAL)
			merge_concatenate_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_output_list, 
				m_param->prev_nb_channels, current->a_dim, net->batch_size, current->output, current->a_size);
		else
			merge_concatenate_flat_fct(m_param->nb_previous, m_param->prev_output_list, m_param->prev_nb_channels,
				current->a_biased_dim, current->output, current->a_size);
	}
	
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, current->a_size * sizeof(float));
}


void backward_merge_layer(layer *current)
{
	network *net = current->c_network;
	m_param = (merge_param*) current->param;
	
	//rebind of delta_o list for INPUT layer is not required as there is no backprop in this case
	
	current->deriv_activation(current);
	
	if(m_param->merge_type == ADD_merge)
	{
		if(current->output_type == SPATIAL)
			merge_add_back_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_delta_o_list, 
				current->delta_o, current->a_size);
		else
			merge_add_back_flat_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_delta_o_list, 
				current->a_biased_dim, current->delta_o, current->a_size);
	}
	else
	{
		if(current->output_type == SPATIAL)
			merge_concatenate_back_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_delta_o_list, 
				m_param->prev_nb_channels, current->a_dim, net->batch_size, current->delta_o, current->a_size);
		else
			merge_concatenate_back_flat_fct(m_param->nb_previous, m_param->prev_type_list, m_param->prev_delta_o_list, 
				m_param->prev_nb_channels, current->a_biased_dim, current->delta_o, current->a_size);
	}
}


void merge_define(layer *current)
{
	current->forward = forward_merge_layer;
	current->backprop = backward_merge_layer;
}




