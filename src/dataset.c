
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


#include "prototypes.h"

// Public are in "prototypes.h"

// Private prototypes
Dataset create_dataset_host(network *net, int nb_elem);
void copy_to_host(float* in_tab, void* out_tab, int out_offset, size_t size);


Dataset create_dataset(network *net, int nb_elem)
{
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		return cuda_create_dataset(net, nb_elem);
	}
	else
	#endif
	{
		return create_dataset_host(net, nb_elem);
	}
}


Dataset create_dataset_host(network *net, int nb_elem)
{
	int i,j;
	Dataset data;
	
	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (void**) malloc(data.nb_batch*sizeof(float*));
	data.target = (void**) malloc(data.nb_batch*sizeof(float*));
	data.localization = HOST;
	data.cont_copy = copy_to_host;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		((float**)data.input)[i] = (float*) calloc(net->batch_size * (net->input_dim + 1), sizeof(float));
		((float**)data.target)[i] = (float*) calloc(net->batch_size * net->output_dim, sizeof(float));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			((float**)data.input)[i][j*(net->input_dim+1) + net->input_dim] = net->input_bias;
		}
	}
	
	return data;
}


void copy_to_host(float* in_tab, void* out_tab, int out_offset, size_t size)
{
	float* f_out_tab = (float*) out_tab;
	for(size_t i = 0; i < size; i++)
		*(f_out_tab + out_offset + i) = (*(in_tab + i));
}


void host_only_shuffle(network *net, Dataset data)
{
	int i, j, k;
	float temp;
	int pos, pos2, batch, batch2;

	for(i = 0; i < data.size - 1; i++)
	{
		j = i + random_uniform() * (double)(data.size-i);
		pos = i%net->batch_size;
		batch = i/net->batch_size;
		pos2 = j%net->batch_size;
		batch2 = j/net->batch_size;
		
		for(k = 0; k < net->input_dim+1; k++)
		{
			temp = ((float**)data.input)[batch][pos*(net->input_dim + 1) + k];
			((float**)data.input)[batch][pos*(net->input_dim + 1) + k] = ((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k];
			((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k] = temp;
		}
		
		for(k = 0; k < net->output_dim; k++)
		{
			temp = ((float**)data.target)[batch][pos*net->output_dim + k];
			
			((float**)data.target)[batch][pos*net->output_dim + k] = ((float**)data.target)[batch2][pos2*net->output_dim + k];
			((float**)data.target)[batch2][pos2*net->output_dim + k] = temp;
		}
	}
}


void free_dataset(Dataset *data)
{
	int i;
		
	if(data->localization == HOST)
	{
		if(data->input != NULL)
		{
			for(i = 0; i < data->nb_batch; i++)
			{
				free(data->input[i]);
				free(data->target[i]);
			}
		}
		if(&data->input[0] != NULL)
		{
			free(&data->input[0]);
			free(&data->target[0]);
		}
	}
	#ifdef CUDA
	else if(data->localization == DEVICE)
	{
		cuda_free_dataset(data);
		
		if(&data->input[0] != NULL)
		{
			free(&data->input[0]);
			free(&data->target[0]);
		}
	}
	#endif
	
	data->localization = NO_LOC;
}
