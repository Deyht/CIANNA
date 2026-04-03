
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

// Local variables
static merge_param *m_param;

// Public are in prototypes.h

// Private prototypes
void print_merge_type(FILE *f, int type, int f_bin);
void get_string_merge_type(char* str, int type);
int load_merge_type(const char *type);


void print_merge_type(FILE *f, int type, int f_bin)
{
	char temp_string[40];
	
	switch(type)
	{
		default:
		case ADD_merge:
			sprintf(temp_string,"ADD");
			break;
			
		case CONCAT_merge:
			sprintf(temp_string,"CONCAT");
			break;
	}
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s", temp_string);
}

void get_string_merge_type(char* str, int type)
{
	switch(type)
	{
		default:
		case ADD_merge:
			sprintf(str,"ADD");
			break;
		
		case CONCAT_merge:
			sprintf(str,"CONCAT");
			break;
	}
}

int load_merge_type(const char *type)
{
	if(strcmp(type, "ADD") == 0)
		return ADD_merge;
	else if(strcmp(type, "CONCAT") == 0)
		return CONCAT_merge;
	else
		return ADD_merge;
}


int merge_create(network *net, int previous_id_a, int previous_id_b, int merge_type, const char *activation)
{
	size_t i, nb_features;
	size_t flat_output_dim = 1;
	size_t mem_approx = 0;

	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING MERGE LAYER ...\n", net->nb_layers);
	
	current->type = MERGE;
	current->frozen = 0;
	current->dropout_rate = 0.0f;
	current->previous = NULL;
	current->input = NULL;
	
	m_param = (merge_param*) malloc(sizeof(merge_param));
	current->param = m_param;
	m_param->merge_type = merge_type;
	
	if(previous_id_a > net->nb_layers-2 || previous_id_a < -net->nb_layers)
	{
		printf("\n ERROR: Invalid layer id value for previous A in merge layer!\n");
		exit(EXIT_FAILURE);
	}
	if(previous_id_b > net->nb_layers-2 || previous_id_b < -net->nb_layers)
	{
		printf("\n ERROR: Invalid layer id value for previous B in merge layer!\n");
		exit(EXIT_FAILURE);
	}
	
	m_param->previous_id_a = previous_id_a;
	m_param->previous_id_b = previous_id_b;
	
	if(previous_id_a < 0)
		m_param->previous_a = net->net_layers[net->nb_layers - 1 + previous_id_a];
	else
		m_param->previous_a = net->net_layers[previous_id_a];
	if(previous_id_b < 0)
		m_param->previous_b = net->net_layers[net->nb_layers - 1 + previous_id_b];
	else
		m_param->previous_b = net->net_layers[previous_id_b];
	
	if(m_param->previous_a->output_type != m_param->previous_b->output_type)
	{
		printf("\n ERROR: trying to merge layers of different kind!\n");
		exit(EXIT_FAILURE);
	}

	current->output_type = m_param->previous_a->output_type;
	current->output_dim = (int*) calloc(4, sizeof(int));
	
	if(m_param->merge_type == ADD_merge)
	{
		for(i = 0; i < 4; i++)
			if(m_param->previous_a->output_dim[i] != m_param->previous_b->output_dim[i])
			{
				printf("\n ERROR: ADD merge layer require the two input layers to have the exact same shape!\n");
				exit(EXIT_FAILURE);
			}	
		
		for(i = 0; i < 4; i++)
		{
			current->output_dim[i] = m_param->previous_a->output_dim[i];
			//prev_dim
		}
	}
	else
	{
		for(i = 0; i < 3; i++)
			if(m_param->previous_a->output_dim[i] != m_param->previous_b->output_dim[i]) //should also work with dense
			{
				printf("\n ERROR: CONCATENATE merge layer require the two input layers to the same spatial shape!\n");
				exit(EXIT_FAILURE);
			}
		
		for(i = 0; i < 3; i++)
		{
			current->output_dim[i] = m_param->previous_a->output_dim[i];
			//prev_dim
		}
		current->output_dim[3] = m_param->previous_a->output_dim[3] + m_param->previous_b->output_dim[3];
	}
	
	nb_features = current->output_dim[3];
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 3; i++)
			flat_output_dim *= current->output_dim[i];
	
		current->a_size       = flat_output_dim * nb_features * net->batch_size;
		current->a_dim        = flat_output_dim;
		current->a_biased_dim = flat_output_dim;
		current->a_offset     = net->batch_size;
	}
	else
	{	
		flat_output_dim = nb_features + 1;
	
		current->a_size       = flat_output_dim * net->batch_size;
		current->a_dim        = nb_features;
		current->a_biased_dim = flat_output_dim;
		current->a_offset     = 1;
	}

	current->output = (float*) calloc(current->a_size, sizeof(float));
	mem_approx += current->a_size * sizeof(float);
	
	if(!net->inference_only)
	{
		//allocate output error comming from next layer
		current->delta_o = (float*) calloc(current->a_size, sizeof(float));
		mem_approx += current->a_size * sizeof(float);
	}
	
	current->nb_params = 0;
	
	define_activation_param(current, activation);
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_merge_define(current);
			mem_approx = cuda_convert_merge_layer(current);
			cuda_define_activation_fct(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			merge_define(current);
			define_activation_fct(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	char m_type[40];
	
	get_string_merge_type(m_type, merge_type);
	fill_string_activ_param(current, activ, 0);
	
	printf("      Prev_A: %d, Prev_B: %d, Type: %s\n\
      Output: %dx%dx%dx%ld, Activation: %s\n\
      Approx layer RAM/VRAM requirement: %d MB\n",
		previous_id_a, previous_id_b, m_type,
		current->output_dim[0], current->output_dim[1], current->output_dim[2], nb_features,
		activ, (int)(mem_approx/1000000));
	net->total_nb_param += 0;
	net->memory_footprint += mem_approx;

	return net->nb_layers - 1;
}



void merge_save(FILE *f, layer *current, int f_bin)
{
	char layer_type = 'M';
	
	m_param = (merge_param*)current->param;
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&m_param->previous_id_a, sizeof(int), 1, f);
		fwrite(&m_param->previous_id_b, sizeof(int), 1, f);
		fwrite(current->output_dim, sizeof(int), 4, f);
		print_merge_type(f, m_param->merge_type, f_bin);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"M"); //output dim required for partial save loading through skip_layers
		fprintf(f, "%dpa%dpb odim%dx%dx%dx%d ", m_param->previous_id_a, m_param->previous_id_b,
			current->output_dim[0], current->output_dim[1], current->output_dim[2], current->output_dim[3]);
		print_merge_type(f, m_param->merge_type, f_bin);
		fprintf(f, " ");
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n\n");
	}
}

void merge_load(network *net, FILE *f, int f_bin, int skip_layer)
{
	int i;
	char merge_type[40];
	char activ_type[40];
	int output_dim[4];
	int previous_id_a, previous_id_b;
	
	if(!skip_layer)
		printf("Loading merge layer, M:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&previous_id_a, sizeof(int), 1, f);
		fread(&previous_id_b, sizeof(int), 1, f);
		fread(output_dim, sizeof(int), 4, f);
		fread(merge_type, sizeof(char), 40, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, "%dpa%dpb odim%dx%dx%dx%d %s %s\n", &previous_id_a, &previous_id_b,
			&output_dim[0], &output_dim[1], &output_dim[2], &output_dim[3], merge_type, activ_type);
	
	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
		{
			printf("\n ERROR: Cannot start a network with a merge layer!\n");
			exit(EXIT_FAILURE);
		}
		
		merge_create(net, previous_id_a, previous_id_b, load_merge_type(merge_type), activ_type);
	}
	else
	{
		for(i = 0; i < 4; i++)
			net->skip_in_dims[i] = output_dim[i];
	}
}

void free_merge(layer *current)
{
	//free(current->output_dim);
		
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{	
		cuda_free_merge(current);
	}
	else
	#endif
	{
		free(current->output);
		if(!current->c_network->inference_only)
			free(current->delta_o);
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}













