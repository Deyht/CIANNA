

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

static pool_param *p_param;

//public are in prototypes.h

void max_pooling_fct
	(void* i_input, void* i_output, int* pool_map,
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int bias_in, int length)
{	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	#endif
	
	float* input;
	float* output;
	int* l_pool_map;
	
	int x, y, z, x_max, y_max, z_max, o_pos[3], i_pos[3], max_found = 0;
	size_t l_pos;
	float val_max;
	
	#pragma omp for collapse(2) schedule(guided, 2)
	for(int i = 0; i < w_size_out*h_size_out*d_size_out; i++)
	{	
		for(int k = 0; k < length; k++)
		{
			max_found = 0;
		
			o_pos[2] = i / (w_size_out*h_size_out); 
			o_pos[1] = (i % (w_size_out*h_size_out)) / w_size_out;
			o_pos[0] = (i % (w_size_out*h_size_out)) % w_size_out;
			
			output = (float*)i_output+ k*(size_t)(w_size_out*h_size_out*d_size_out);
			if(pool_map != NULL)
				l_pool_map = pool_map + k*(size_t)(w_size_out*h_size_out*d_size_out);
			
			input = (float*)i_input + k*(size_t)(w_size*h_size*d_size+bias_in);
			
			i_pos[0] = o_pos[0]*stride_w;
			i_pos[1] = o_pos[1]*stride_h;
			i_pos[2] = o_pos[2]*stride_d;
			
			for(z = 0; z < pool_size_d; z++)
			{
				if((i_pos[2] + z) < padding_d || (i_pos[2] + z) >= (d_size + padding_d))
					continue;
				for(y = 0; y < pool_size_h; y++)
				{
					if((i_pos[1] + y) < padding_h || (i_pos[1] + y) >= (h_size + padding_h))
						continue;
					for(x = 0; x < pool_size_w; x++)
					{
						if((i_pos[0] + x) < padding_w || (i_pos[0] + x) >= (w_size + padding_w))
							continue;
						if(max_found == 0)
						{
							max_found = 1;
							x_max = x; y_max = y; z_max = z;
							l_pos = (i_pos[0] + x - padding_w) 
								  + (i_pos[1] + y - padding_h) * w_size 
								  + (i_pos[2] + z - padding_d) * (size_t)(w_size * h_size);
							val_max = input[l_pos];
						}
						else
						{
							l_pos = (i_pos[0] + x - padding_w) 
								  + (i_pos[1] + y - padding_h) * w_size 
								  + (i_pos[2] + z - padding_d) * (size_t)(w_size * h_size);
							if(input[l_pos] > val_max)
							{
								x_max = x; y_max = y; z_max = z;
								val_max = input[l_pos];
							}
						}
					}
				}
			}
			if(max_found == 0)
			{
				if(pool_map != NULL)
					l_pool_map[i] = -1;
				output[i] = 0.0f;
			}
			else
			{
				if(pool_map != NULL)
					l_pool_map[i] = (z_max*(size_t)(pool_size_w*pool_size_h) + y_max*pool_size_w + x_max);
				output[i] = val_max;
			}
		}
	}
	#ifdef OPEN_MP
	}
	#endif
}


void avg_pooling_fct
	(void* i_input, void* i_output, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int bias_in, int length)
{	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	#endif
	
	float* input;
	float* output;

	int x, y, z, o_pos[3], i_pos[3]; 
	double r_avg = 0.0f;
	int sum_elem = 0;
	
	#pragma omp for collapse(2) schedule(guided, 2)
	for(int i = 0; i < w_size_out*h_size_out*d_size_out; i++)
	{	
		for(int k = 0; k < length; k++)
		{
			r_avg = 0.0f;
			sum_elem = 0;
		
			o_pos[2] = i / (w_size_out*h_size_out); 
			o_pos[1] = (i % (w_size_out*h_size_out)) / w_size_out;
			o_pos[0] = (i % (w_size_out*h_size_out)) % w_size_out;
			
			output = (float*)i_output + k*(size_t)(w_size_out*h_size_out*d_size_out);
			
			input = (float*)i_input + k*(size_t)(w_size*h_size*d_size+bias_in);
			
			i_pos[0] = o_pos[0]*stride_w;
			i_pos[1] = o_pos[1]*stride_h;
			i_pos[2] = o_pos[2]*stride_d;
			
			for(z = 0; z < pool_size_d; z++)
			{
				if((i_pos[2] + z) < padding_d || (i_pos[2] + z) >= (d_size + padding_d))
					continue;
				for(y = 0; y < pool_size_h; y++)
				{
					if((i_pos[1] + y) < padding_h || (i_pos[1] + y) >= (h_size + padding_h))
						continue;
					for(x = 0; x < pool_size_w; x++)
					{
						if((i_pos[0] + x) < padding_w || (i_pos[0] + x) >= (w_size + padding_w))
							continue;
						r_avg += (float) input[(i_pos[0] + x - padding_w) 
								  + (i_pos[1] + y - padding_h) * w_size 
								  + (i_pos[2] + z - padding_d) * w_size * h_size];
						sum_elem += 1;
					}
				}
			}
			output[i] = (r_avg/sum_elem);
		}
	}
	#ifdef OPEN_MP
	}
	#endif
}


void deltah_max_pool_cont_fct
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, size_t length)
{	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	#endif
	
	float* delta_o;
	float* delta_o_unpool;
	int* l_pool_map;

	int map_batch, in_im_pos, loc;
	int x, y, z, f_pos[3], pos[3];
	float l_delta_h = 0.0f;
	
	#pragma omp for schedule(guided, 2)
	for(int i = 0; i < length; i++)
	{	
		l_delta_h = 0.0f;
		
		map_batch = i / (size_t)(w_size*h_size*d_size);
		in_im_pos =  i % (size_t)(w_size*h_size*d_size);
		
		delta_o = (float*)i_delta_o + map_batch * (size_t)(w_size_out * h_size_out * d_size_out);
		l_pool_map = pool_map + map_batch * (size_t)(w_size_out * h_size_out * d_size_out);
		delta_o_unpool = (float*)i_delta_o_unpool + map_batch * (size_t)(w_size * h_size * d_size);
		/*Note, only span non-padded array, so should get padded to compensate, or pad in the loops ?*/
		pos[2] = in_im_pos / (w_size*h_size) + padding_d;
		pos[1] = (in_im_pos % (w_size*h_size)) / w_size + padding_h;
		pos[0] = (in_im_pos % (w_size*h_size)) % w_size + padding_w;
		
		for(z = pos[2]/stride_d; (pos[2]-z*stride_d < pool_size_d); z -=1)
		{
			f_pos[2] = pos[2]-z*stride_d;
			if((z < 0) || (z >= d_size_out))
				continue;
			for(y = pos[1]/stride_h; (pos[1]-y*stride_h < pool_size_h); y -=1)
			{
				f_pos[1] = pos[1]-y*stride_h;
				if((y < 0) || (y >= h_size_out))
					continue;
				for(x = pos[0]/stride_w; (pos[0]-x*stride_w < pool_size_w); x -=1)
				{
					f_pos[0] = pos[0]-x*stride_w;
					if((x < 0) || (x >= w_size_out))
						continue;
					loc = z*w_size_out*h_size_out + y*w_size_out + x;
					if(l_pool_map[loc] == (f_pos[2]*pool_size_w*pool_size_h + f_pos[1]*pool_size_w + f_pos[0]))
						l_delta_h += (float) delta_o[loc];
				}
			}
		}
		delta_o_unpool[(pos[2]-padding_d)*(size_t)(w_size*h_size) + (pos[1]-padding_h)*w_size + (pos[0]-padding_w)] = l_delta_h;
	}
	#ifdef OPEN_MP
	}
	#endif
}


void deltah_avg_pool_cont_fct
	(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, size_t length)
{	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	#endif
	
	float* delta_o;
	float* delta_o_unpool;

	int map_batch, in_im_pos;
	int x, y, z, pos[3];
	float l_delta_h = 0.0f;
	
	#pragma omp for schedule(guided, 2)
	for(int i = 0; i < length; i++)
	{
		l_delta_h = 0.0f;
		
		map_batch = i / (size_t)(w_size*h_size*d_size);
		in_im_pos =  i % (size_t)(w_size*h_size*d_size);
		
		delta_o = (float*)i_delta_o + map_batch * (size_t)(w_size_out * h_size_out * d_size_out);
		delta_o_unpool = (float*)i_delta_o_unpool + map_batch * (size_t)(w_size * h_size * d_size);
		
		pos[2] = in_im_pos / (w_size*h_size) + padding_d;
		pos[1] = (in_im_pos % (w_size*h_size)) / w_size + padding_h;
		pos[0] = (in_im_pos % (w_size*h_size)) % w_size + padding_w;
		
		for(z = pos[2]/stride_d; (pos[2]-z*stride_d < pool_size_d); z -=1)
		{
			if((z < 0) || (z >= d_size_out))
				continue;
			for(y = pos[1]/stride_h; (pos[1]-y*stride_h < pool_size_h); y -=1)
			{
				if((y < 0) || (y >= h_size_out))
					continue;
				for(x = pos[0]/stride_w; (pos[0]-x*stride_w < pool_size_w); x -=1)
				{
					if((x < 0) || (x >= w_size_out))
						continue;
					l_delta_h += (float)delta_o[z*(size_t)(w_size_out*h_size_out) + y*w_size_out + x]
						/(float)(pool_size_w*pool_size_h*pool_size_d);
				}
			}
		}
		delta_o_unpool[(pos[2]-padding_d)*(size_t)(w_size*h_size) + (pos[1]-padding_h)*w_size + (pos[0]-padding_w)] = l_delta_h;
	}
	#ifdef OPEN_MP
	}
	#endif
}

void dropout_select_pool(float* mask, size_t size, float drop_rate)
{
	size_t i;
	float rand;
	
	//#pragma omp parallel for private(rand) schedule(guided,4)
	//OMP overhead is too high for "small" dense layers
	//Performance is limited by CPU cache size and speed regardless of core count
	for(i = 0; i < size; i++)
	{
		rand = random_uniform();
		if(rand < drop_rate)
			mask[i] = 0.0f;
		else
			mask[i] = 1.0f;
	}
}

void dropout_apply_pool(void* i_table, float* mask, size_t size)
{
	size_t i;
	float* table = (float*) i_table;
	
	for(i = 0; i < size; i++)
		table[i] *= mask[i];
}


void forward_pool_layer(layer* current)
{
	int bias_in = 0;
	network* net = current->c_network;

	if(net->length == 0)
		return;
		
	if(current->previous == NULL)
	{
		current->input = net->input;
		bias_in = 1;
	}

	p_param = (pool_param*) current->param;
	
	switch(p_param->pool_type)
	{
		default:
		case MAX_pool:
			max_pooling_fct(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->stride[0], p_param->stride[1], p_param->stride[2],
				p_param->padding[0], p_param->padding[1], p_param->padding[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				bias_in, p_param->nb_maps * net->batch_size);
			break;
		case AVG_pool:
			avg_pooling_fct(current->input, current->output, 
				p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
				p_param->stride[0], p_param->stride[1], p_param->stride[2],
				p_param->padding[0], p_param->padding[1], p_param->padding[2],
				p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2], 
				p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2], 
				bias_in, p_param->nb_maps * net->batch_size);
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
			((current->previous->type == NORM || current->previous->type == LRN) && current->previous->previous->type == CONV))
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
						p_param->stride[0], p_param->stride[1], p_param->stride[2],
						p_param->padding[0], p_param->padding[1], p_param->padding[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * (size_t)(p_param->prev_size[0] * p_param->prev_size[1] *p_param->prev_size[2]));
					break;
				
				case AVG_pool:
					deltah_avg_pool_cont_fct(current->delta_o, current->previous->delta_o, 
						p_param->pool_map, p_param->p_size[0], p_param->p_size[1], p_param->p_size[2],
						p_param->stride[0], p_param->stride[1], p_param->stride[2],
						p_param->padding[0], p_param->padding[1], p_param->padding[2],
						p_param->prev_size[0], p_param->prev_size[1], p_param->prev_size[2],
						p_param->nb_area[0], p_param->nb_area[1], p_param->nb_area[2],
						net->batch_size * p_param->nb_maps * (size_t)(p_param->prev_size[0] * p_param->prev_size[1] *p_param->prev_size[2]));
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









