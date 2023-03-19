

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

static pool_param *p_param;

//public are in prototypes.h

void max_pooling_fct
	(void* i_input, void* i_output, int* pool_map,
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int length)
{	
	int i, k, x, y, z, x_max, y_max, z_max, pos, pos_x, pos_y, pos_z, pos_out;
	
	float* input  = (float*) i_input;
	float* output = (float*) i_output;
	
	#pragma omp parallel for private(k, x, y, z, x_max, y_max, z_max, pos, pos_x, pos_y, pos_z, pos_out) collapse(2) schedule(guided, 2)
	for(i = 0; i < w_size_out * h_size_out * d_size_out; i++)
	{
		for(k = 0; k < length; k++)
		{
			pos_z = i / (w_size_out*h_size_out); 
			pos_y = (i % (w_size_out*h_size_out)) / w_size_out;
			pos_x = (i % (w_size_out*h_size_out)) % w_size_out;
			
			pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);
			
			pos = k*w_size*h_size*d_size + pos_x*pool_size_w + pos_y*pool_size_h*w_size + pos_z*pool_size_d*w_size*h_size;
			
			if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)
			{
				x_max = 0; y_max = 0; z_max = 0;
				for(x = 0; x < pool_size_d; x++)
					for(y = 0; y < pool_size_h; y++)
						for(z = 0; z < pool_size_w; z++)
							if(input[pos + x_max*w_size*h_size + y_max*w_size + z_max] 
								< input[pos + x*w_size*h_size + y*w_size + z])
							{
								x_max = x; y_max = y; z_max = z;
							}
				pool_map[pos_out] = (x_max*pool_size_w*pool_size_h + y_max*pool_size_w + z_max);
				output[pos_out] = input[pos + x_max*w_size*h_size + y_max*w_size + z_max];
			}
		}
	}
	
	
}


void avg_pooling_fct
	(void* i_input, void* i_output, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int length)
{
	int i, k, x, y, z, pos, pos_x, pos_y, pos_z, pos_out;
	double r_avg = 0.0;
	
	float* input  = (float*) i_input;
	float* output = (float*) i_output;
	
	#pragma omp parallel for private(k, x, y, z, pos, pos_x, pos_y, pos_z, pos_out, r_avg) collapse(2) schedule(guided, 2)
	for(i = 0; i < w_size_out * h_size_out * d_size_out; i++)
	{
		for(k = 0; k < length; k++)
		{
			pos_z = i / (w_size_out*h_size_out); 
			pos_y = (i % (w_size_out*h_size_out)) / w_size_out;
			pos_x = (i % (w_size_out*h_size_out)) % w_size_out;
			
			pos_out = k*(w_size_out*h_size_out*d_size_out) + pos_x + pos_y*w_size_out + pos_z*(w_size_out*h_size_out);
			
			pos = k*w_size*h_size*d_size + pos_x*pool_size_w + pos_y*pool_size_h*w_size + pos_z*pool_size_d*w_size*h_size;
			
			r_avg = 0.0;
			if(pos_x < w_size_out && pos_y < h_size_out && pos_z < d_size_out && k < length)
			{
				for(x = 0; x < pool_size_d; x++)
					for(y = 0; y < pool_size_h; y++)
						for(z = 0; z < pool_size_w; z++)
							r_avg += input[pos + x*w_size*h_size + y*w_size + z];
			
				output[pos_out] = (r_avg/(pool_size_w*pool_size_h*pool_size_d));
			}
		}
	}
}


void deltah_max_pool_cont_fct
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int w_size, int h_size, int d_size,
	int w_size_out, int h_size_out, int d_size_out, int length)
{
	int i;
	float* delta_o = (float*) i_delta_o;
	float* t_delta_o_unpool = (float*) i_delta_o_unpool;
	float* delta_o_unpool = NULL;
	
	#pragma omp parallel for private(delta_o_unpool) schedule(guided, 2)
	for(i = 0; i < length; i++)
	{
		int map_batch, in_im_pos;
		int pos_x, pos_y, pos_z;
		int pool_x, pool_y, pool_z;
		
		map_batch = i / (w_size_out*h_size_out*d_size_out);
		in_im_pos =  i % (w_size_out*h_size_out*d_size_out);
		pos_z = in_im_pos / (w_size_out*h_size_out);
		pos_y = (in_im_pos % (w_size_out*h_size_out)) / w_size_out;
		pos_x = (in_im_pos % (w_size_out*h_size_out)) % w_size_out;
		
		pool_z = pool_map[i]/(pool_size_w*pool_size_h);
		pool_y = (pool_map[i] % (pool_size_w*pool_size_h)) / pool_size_w;
		pool_x = (pool_map[i] % (pool_size_w*pool_size_h)) % pool_size_w;
	
		/*add mask of locations*/
		delta_o_unpool = t_delta_o_unpool + map_batch * w_size * h_size * d_size
			+ pos_z * pool_size_d * w_size*h_size + pos_y * pool_size_h * w_size + pos_x * pool_size_w
			+ pool_z * w_size * h_size + pool_y * w_size + pool_x;
		
		*delta_o_unpool = delta_o[i];
	}
}


void deltah_avg_pool_cont_fct
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int w_size, int h_size, int d_size,
	int w_size_out, int h_size_out, int d_size_out, int length)
{
	int i;
	
	float* delta_o = (float*) i_delta_o;
	float* t_delta_o_unpool = (float*) i_delta_o_unpool;
	float* delta_o_unpool = NULL;
	
	#pragma omp parallel for private(delta_o_unpool) schedule(guided, 2)
	for(i = 0; i < length; i++)
	{
		int map_batch, in_im_pos;
		int pos_x, pos_y, pos_z;
		int x, y, z;
	
		map_batch = i / (w_size_out*h_size_out*d_size_out);
		in_im_pos =  i % (w_size_out*h_size_out*d_size_out);
		pos_z = in_im_pos / (w_size_out*h_size_out);
		pos_y = (in_im_pos % (w_size_out*h_size_out)) / w_size_out;
		pos_x = (in_im_pos % (w_size_out*h_size_out)) % w_size_out;
	
		/*add mask of locations*/
		delta_o_unpool = t_delta_o_unpool + map_batch * w_size * h_size * d_size
			+ pos_z * pool_size_d * w_size*h_size + pos_y * pool_size_h * w_size + pos_x * pool_size_w;
	
		for(z = 0; z < pool_size_d; z++)
			for(y = 0; y < pool_size_h; y++)
				for(x = 0; x < pool_size_w; x++)
					 delta_o_unpool[z * w_size * h_size	+ y * w_size + x]
						= (float)delta_o[i]/(pool_size_w*pool_size_h*pool_size_d);
	}
}

void dropout_select_pool(int* mask, int size, float drop_rate)
{
	int i;
	float rand;
	
	//#pragma omp parallel for private(rand) schedule(guided,4)
	//OMP overhead is too high for "small" dense layers
	//Performance is limited by CPU cache size and speed regardless of core count
	for(i = 0; i < size; i++)
	{
		rand = random_uniform();
		if(rand < drop_rate)
			mask[i] = 0;
		else
			mask[i] = 1;
	}
}

void dropout_apply_pool(void* i_table, int* mask, int size)
{
	int i;
	float* table = (float*) i_table;
	
	for(i = 0; i < size; i++)
		table[i] *= mask[i];
}


void forward_pool_layer(layer* current)
{
	network* net = current->c_network;

	if(net->length == 0)
		return;

	p_param = (pool_param*) current->param;
	
	switch(p_param->pool_type)
	{
		default:
		case MAX_pool:
			max_pooling_fct(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				p_param->nb_maps * net->batch_size);
			break;
		case AVG_pool:
			avg_pooling_fct(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				p_param->nb_maps * net->batch_size);
			break;
	}

	current->activation(current);

	if(current->dropout_rate > 0.01f && (!net->is_inference || net->inference_drop_mode == MC_MODEL))
	{
		dropout_select_pool(p_param->dropout_mask, p_param->nb_maps 
			* (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size, current->dropout_rate);	
		
		dropout_apply_pool(current->output, p_param->dropout_mask, p_param->nb_maps 
			* (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size);
	}
}


void backward_pool_layer(layer* current)
{	
	int i;

	network* net = current->c_network;

	p_param = (pool_param*) current->param;

	if(current->dropout_rate > 0.01f)
	{
		dropout_apply_pool(current->delta_o, p_param->dropout_mask, p_param->nb_maps 
			* (p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]) * net->batch_size);
	}

	if(current->previous != NULL)
	{
		if(current->previous->type == CONV ||
			(current->previous->type == NORM && current->previous->previous->type == CONV))
		{		
			int size = p_param->nb_maps*p_param->prev_size[0]*p_param->prev_size[1]*p_param->prev_size[2]*net->batch_size;
			float* f_tab = (float*) current->previous->delta_o;
			for(i = 0; i < size; i++)
				f_tab[i] = 0.0f;
			
			switch(p_param->pool_type)
			{
				default:
				case MAX_pool:
					deltah_max_pool_cont_fct(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]);
					break;
				
				case AVG_pool:
					deltah_avg_pool_cont_fct(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2]);
					break;
			}
		}
		current->previous->deriv_activation(current->previous);
	}
}


void pool_define(layer *current)
{
	current->forward = forward_pool_layer;
	current->backprop = backward_pool_layer;
}









