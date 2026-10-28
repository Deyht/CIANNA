
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
Dataset create_dataset_host(network *net, int with_target, size_t nb_elem, 
	int nb_branch_in, size_t *branch_in_dim, int nb_branch_targ, size_t *branch_targ_dim);
void copy_to_host(float* in_tab, void* out_tab, int out_offset, size_t size);


Dataset create_dataset(network *net, int with_target, size_t nb_elem, 
	int nb_branch_in, size_t *branch_in_dim, int nb_branch_targ, size_t *branch_targ_dim)
{
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		return cuda_create_dataset(net, with_target, nb_elem, nb_branch_in, branch_in_dim, nb_branch_targ, branch_targ_dim);
	}
	else
	#endif
	{
		return create_dataset_host(net, with_target, nb_elem, nb_branch_in, branch_in_dim, nb_branch_targ, branch_targ_dim);
	}
}


Dataset create_dataset_host(network *net, int with_target, size_t nb_elem, 
	int nb_branch_in, size_t *branch_in_dim, int nb_branch_targ, size_t *branch_targ_dim)
{
	size_t i, b;
	Dataset data;
	
	data.nb_branch_in   = nb_branch_in;
	data.nb_branch_targ = nb_branch_targ;
	data.size           = nb_elem;
	data.localization   = HOST;
	data.cont_copy      = copy_to_host;
	data.nb_batch       = (data.size - 1) / net->batch_size + 1;
	
	data.input = (void***) malloc(nb_branch_in*sizeof(void**));
	data.branch_in_dim = (size_t*) malloc(nb_branch_in*sizeof(size_t));
	
	for(b = 0; b < nb_branch_in; b++)
	{
		data.input[b] = (void**) malloc(data.nb_batch*sizeof(float*));
		data.branch_in_dim[b] = branch_in_dim[b];
	
		for(i = 0; i < data.nb_batch; i++)
			((float**)data.input[b])[i] = (float*) calloc(net->batch_size * (branch_in_dim[b] + 1), sizeof(float));
	}
	
	if(with_target)
	{
		data.target = (void***) malloc(nb_branch_targ*sizeof(void**));
		data.branch_targ_dim = (size_t*) malloc(nb_branch_targ*sizeof(size_t));
		
		for(b = 0; b < nb_branch_targ; b++)
		{
			data.target[b] = (void**) malloc(data.nb_batch*sizeof(float*));
			data.branch_targ_dim[b] = branch_targ_dim[b];
			
			for(i = 0; i < data.nb_batch; i++)
				((float**)data.target[b])[i] = (float*) calloc(net->batch_size * branch_targ_dim[b], sizeof(float));
		}
	}
	else
	{
		data.target = NULL;
		data.branch_targ_dim = NULL;
	}
	
	return data;
}


void copy_to_host(float *in_tab, void *out_tab, int out_offset, size_t size)
{
	float* f_out_tab = (float*) out_tab;
	for(size_t i = 0; i < size; i++)
		*(f_out_tab + out_offset + i) = (*(in_tab + i));
}


void host_only_shuffle(network *net, Dataset *data)
{
	size_t i, j, k, b;
	float temp;
	size_t pos, pos2, batch, batch2;
	float **c_input, **c_target;

	for(i = 0; i < data->size - 1; i++)
	{
		j      = i + random_int_below(data->size - i);
		pos    = i%net->batch_size;
		batch  = i/net->batch_size;
		pos2   = j%net->batch_size;
		batch2 = j/net->batch_size;
		
		for(b = 0; b < data->nb_branch_in; b++)
		{
			c_input = (float**)data->input[b];
			
			for(k = 0; k < data->branch_in_dim[b]+1; k++)
			{
				temp = ((float*)c_input[batch])[pos*(data->branch_in_dim[b] + 1) + k];
				((float*)c_input[batch])[pos*(data->branch_in_dim[b] + 1) + k] = 
					((float*)c_input[batch2])[pos2*(data->branch_in_dim[b] + 1) + k];
				((float*)c_input[batch2])[pos2*(data->branch_in_dim[b] + 1) + k] = temp;
			}
		}
		
		if(data->target != NULL)
		{
			for(b = 0; b < data->nb_branch_targ; b++)
			{
				c_target = (float**)data->target[b];
				
				for(k = 0; k < data->branch_targ_dim[b]; k++)
				{
					temp = ((float*)c_target[batch])[pos*data->branch_targ_dim[b] + k];
					((float*)c_target[batch])[pos*data->branch_targ_dim[b] + k] = 
						((float*)c_target[batch2])[pos2*data->branch_targ_dim[b] + k];
					((float*)c_target[batch2])[pos2*data->branch_targ_dim[b] + k] = temp;
				}
			}
		}
	}
}


void free_dataset(Dataset *data)
{
	size_t i, b;
	
	if(data->localization == HOST)
	{
		if(data->input != NULL)
		{
			for(b = 0; b < data->nb_branch_in; b++)
			{
				if(data->input[b] == NULL)
					continue;
				for(i = 0; i < data->nb_batch; i++)
					free(data->input[b][i]);
				free(&data->input[b][0]);
			}
			free(&data->input[0]);
		}
		
		if(data->target != NULL)
		{
			for(b = 0; b < data->nb_branch_targ; b++)
			{
				if(data->target[b] == NULL)
					continue;
				for(i = 0; i < data->nb_batch; i++)
					free(data->target[b][i]);
				free(&data->target[b][0]);
			}
			free(&data->target[0]);
		}
	}
	#ifdef CUDA
	else if(data->localization == DEVICE)
	{
		cuda_free_dataset(data);
		
		if(data->input != NULL)
			free(&data->input[0]);
		if(data->input_device != NULL)
			free(&data->input_device[0]);
		if(data->target != NULL)
			free(&data->target[0]);
		if(data->target_device != NULL)
			free(&data->target_device[0]);
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











