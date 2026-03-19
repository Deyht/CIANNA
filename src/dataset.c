
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
Dataset create_dataset_host(network *net, int with_target, size_t nb_elem);
void copy_to_host(float* in_tab, void* out_tab, int out_offset, size_t size);


Dataset create_dataset(network *net, int with_target, size_t nb_elem)
{
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		return cuda_create_dataset(net, with_target, nb_elem);
	}
	else
	#endif
	{
		return create_dataset_host(net, with_target, nb_elem);
	}
}


Dataset create_dataset_host(network *net, int with_target, size_t nb_elem)
{
	size_t i;
	Dataset data;
	
	data.size = nb_elem;
	data.localization = HOST;
	data.cont_copy = copy_to_host;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	
	data.input = (void**) malloc(data.nb_batch*sizeof(float*));
	
	for(i = 0; i < data.nb_batch; i++)
		((float**)data.input)[i] = (float*) calloc(net->batch_size * (net->input_dim + 1), sizeof(float));
	
	if(with_target)
	{
		data.target = (void**) malloc(data.nb_batch*sizeof(float*));
		
		for(i = 0; i < data.nb_batch; i++)
			((float**)data.target)[i] = (float*) calloc(net->batch_size * net->output_dim, sizeof(float));
	}
	else
		data.target = NULL;
	
	return data;
}


void copy_to_host(float *in_tab, void *out_tab, int out_offset, size_t size)
{
	float* f_out_tab = (float*) out_tab;
	for(size_t i = 0; i < size; i++)
		*(f_out_tab + out_offset + i) = (*(in_tab + i));
}


void host_only_shuffle(network *net, Dataset data)
{
	size_t i, j, k;
	float temp;
	size_t pos, pos2, batch, batch2;

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
		
		if(data.target != NULL)
		{
			for(k = 0; k < net->output_dim; k++)
			{
				temp = ((float**)data.target)[batch][pos*net->output_dim + k];
				
				((float**)data.target)[batch][pos*net->output_dim + k] = ((float**)data.target)[batch2][pos2*net->output_dim + k];
				((float**)data.target)[batch2][pos2*net->output_dim + k] = temp;
			}
		}
	}
}


void free_dataset(Dataset *data)
{
	size_t i;
	
	if(data->localization == HOST)
	{
		if(data->input != NULL)
		{
			for(i = 0; i < data->nb_batch; i++)
				free(data->input[i]);
			free(&data->input[0]);
		}
		
		if(data->target != NULL)
		{
			for(i = 0; i < data->nb_batch; i++)
				free(data->target[i]);
			free(&data->target[0]);
		}
	}
	#ifdef CUDA
	else if(data->localization == DEVICE)
	{
		cuda_free_dataset(data);
		
		if(data->input != NULL)
			free(&data->input[0]);
		if(data->target != NULL)
			free(&data->target[0]);
	}
	#endif
	
	data->localization = NO_LOC;
}


Dataset* get_dataset_from_type(network *net, const char *dataset_type, int silent)
{
	Dataset *data = NULL;
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		if(silent == 0)
			printf("Setting train set\n");
		data = &net->train;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		if(silent == 0)
			printf("Setting valid set\n");
		data = &net->valid;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		if(silent == 0)
			printf("Setting test set\n");
		data = &net->test;
	}
	else if(strcmp(dataset_type,"TRAIN_buf") == 0)
	{
		if(silent == 0)
			printf("Setting train buffer set\n");
		data = &net->train_buf;
	}
	else if(strcmp(dataset_type,"VALID_buf") == 0)
	{
		if(silent == 0)
			printf("Setting valid buffer set\n");
		data = &net->valid_buf;
	}
	else if(strcmp(dataset_type,"TEST_buf") == 0)
	{
		if(silent == 0)
			printf("Setting test buffer set\n");
		data = &net->test_buf;
	}
	else
	{
		printf("\n ERROR: Unrecognized dataset type!\n");
		exit(EXIT_FAILURE);
	}
	
	return data;
}











