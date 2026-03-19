
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
void merge_add_fct(void *output_a, void *output_b, void *output_new, size_t size);
void merge_add_back_fct(void *delta_o_a, void *delta_o_b, void *delta_o, size_t size);
void merge_concatenate_fct(void *output_a, void *output_b, size_t size_a, size_t size_b, void *output_new, size_t size);
void merge_concatenate_back_fct(void *delta_o_a, void *delta_o_b, size_t size_a, size_t size_b, void *delta_o, size_t size);

void merge_add_fct(void *output_a, void *output_b, void *output_new, size_t size)
{
	size_t i;
	
	float *out_a = (float*) output_a;
	float *out_b = (float*) output_b;
	float *out_new = (float*) output_new;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
		out_new[i] = out_a[i] + out_b[i];
}


void merge_add_back_fct(void *delta_o_a, void *delta_o_b, void *delta_o, size_t size)
{
	size_t i;
	
	float *dlt_a = (float*) delta_o_a;
	float *dlt_b = (float*) delta_o_b;
	float *dlt = (float*) delta_o;
	float dlt_i;
	
	#pragma omp parallel for private(dlt_i) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		dlt_i = dlt[i];
		dlt_a[i] += dlt_i;
		dlt_b[i] += dlt_i;
	}
}


void merge_concatenate_fct(void *output_a, void *output_b,
	size_t size_a, size_t size_b, void *output_new, size_t size)
{
	size_t i;
	
	float *out_a = (float*) output_a;
	float *out_b = (float*) output_b;
	float *out_new = (float*) output_new;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size_a; i++)
		out_new[i] = out_a[i];
	#pragma omp parallel for schedule(guided,4)
	for(i = size_a; i < size; i++)
		out_new[i] = out_b[i-size_a];
}


void merge_concatenate_back_fct(void *delta_o_a, void *delta_o_b,
	size_t size_a, size_t size_b, void *delta_o, size_t size)
{
	size_t i;
	
	float *dlt_a = (float*) delta_o_a;
	float *dlt_b = (float*) delta_o_b;
	float *dlt = (float*) delta_o;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size_a; i++)
		dlt_a[i] += dlt[i];
	#pragma omp parallel for schedule(guided,4)
	for(i = size_a; i < size; i++)
		dlt_b[i-size_a] += dlt[i];
}


void forward_merge_layer(layer *current)
{
	layer *prev_a, *prev_b;
	network *net = current->c_network;
		
	m_param = (merge_param*) current->param;
	
	prev_a = m_param->previous_a;
	prev_b = m_param->previous_b;
	
	if(m_param->merge_type == ADD_merge)
		merge_add_fct(prev_a->output, prev_b->output, current->output, current->a_size);
	else
		merge_concatenate_fct(prev_a->output, prev_b->output, prev_a->a_size, prev_b->a_size,
			current->output, current->a_size);
	
	current->activation(current);
	
	if(!net->inference_only)
		memset(current->delta_o, 0, current->a_size * sizeof(float));
}


void backward_merge_layer(layer *current)
{
	layer *prev_a, *prev_b;
	network *net = current->c_network;
	
	m_param = (merge_param*) current->param;
	
	prev_a = m_param->previous_a;
	prev_b = m_param->previous_b;
	
	current->deriv_activation(current);
	
	// The previous == NULL case is supposed excluded from layer creation condition
	
	if(m_param->merge_type == ADD_merge)
		merge_add_back_fct(prev_a->delta_o, prev_b->delta_o, current->delta_o, current->a_size);
	else
		merge_concatenate_back_fct(prev_a->delta_o, prev_b->delta_o, prev_a->a_size, prev_b->a_size,
			current->delta_o, current->a_size);
}


void merge_define(layer *current)
{
	current->forward = forward_merge_layer;
	current->backprop = backward_merge_layer;
}




