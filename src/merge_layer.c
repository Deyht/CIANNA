
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


int merge_create(network *net, int nb_previous, int *prev_list_id, int merge_type, const char *activation, int silent)
{
	size_t i, j, nb_features;
	size_t flat_output_dim = 1;
	size_t mem_approx = 0;

	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	if(!silent)
		printf("L:%d - CREATING MERGE LAYER ...\n", net->nb_layers - 1);
	
	current->type         = MERGE;
	current->layer_id     = net->nb_layers-1;
	current->frozen       = 0;
	current->dropout_rate = 0.0f;
	current->previous     = NULL;
	current->input        = NULL;
	
	if(current->layer_id == 0)
	{
		printf("\n ERROR: Cannot start a network with a merge layer!\n");
		exit(EXIT_FAILURE);
	}
	
	if(nb_previous < 2 || prev_list_id == NULL)
	{
		printf("\n ERROR: Merge layer requires a list of at least two valid previous layer ids!\n");
		exit(EXIT_FAILURE);
	}
	
	m_param = (merge_param*) malloc(sizeof(merge_param));
	current->param = m_param;
	m_param->merge_type = merge_type;
	m_param->nb_previous = nb_previous;
	m_param->prev_list = (layer**) malloc(nb_previous*sizeof(layer*));
	m_param->prev_type_list = (int*) malloc(nb_previous*sizeof(int));
	m_param->prev_nb_channels = NULL;
	
	for(i = 0; i < nb_previous; i++)
	{
		if(prev_list_id[i] < 0 && prev_list_id[i] + net->nb_layers-1 >= 0)
			m_param->prev_list[i] = net->net_layers[current->layer_id + prev_list_id[i]];
		else if(prev_list_id[i] >= 0 && prev_list_id[i] < net->nb_layers-1)
			m_param->prev_list[i] = net->net_layers[prev_list_id[i]];
		else
		{
			printf("\n ERROR: Invalid previous layer id provided!\n");
			exit(EXIT_FAILURE);
		}
	}
	
	for(i = 0; i < nb_previous; i++)
		for(j = i + 1; j < nb_previous; j++)
			if(m_param->prev_list[i] == m_param->prev_list[j])
			{
				printf("\n ERROR: All previous layer IDs must be different!\n");
				exit(EXIT_FAILURE);
			}
	
	
	for(i = 0; i < nb_previous; i++)
		m_param->prev_type_list[i] = m_param->prev_list[i]->type;
	
	current->output_type = -1;
	for(i = 0; i < nb_previous; i++)
	{	//Output type is determined by the first non-input layer
		if(m_param->prev_type_list[i] != INPUT)
		{
			current->output_type = m_param->prev_list[i]->output_type;
			break;
		}
	}
	if(current->output_type == -1) //All merged layers are input
	{
		flat_output_dim = 1;
		for(i = 0; i < 3; i++)
			flat_output_dim *= m_param->prev_list[0]->output_dim[i];
		if(flat_output_dim == 1)
			current->output_type = FLAT;
		else
			current->output_type = SPATIAL;
	}
	
	for(i = 0; i < nb_previous; i++)
		if((m_param->prev_list[i]->output_type != current->output_type))
		{
			printf("\n ERROR: Can't merge layers of different kind!\n");
			exit(EXIT_FAILURE);
		}
	
	current->output_dim = (int*) calloc(4, sizeof(int));
	current->prev_dim = m_param->prev_list[0]->output_dim; //used only as a placeholder here
	
	if(m_param->merge_type == ADD_merge)
	{
		for(i = 1; i < nb_previous; i++)
			for(j = 0; j < 4; j++)
				if(m_param->prev_list[i]->output_dim[j] != current->prev_dim[j])
				{
					printf("\n ERROR: ADD merge layer require all input layers to have the exact same shape!\n");
					exit(EXIT_FAILURE);
				}
		
		for(j = 0; j < 4; j++)
			current->output_dim[j] = current->prev_dim[j];
	}
	else // CONCAT_MERGE
	{
		for(i = 1; i < nb_previous; i++)
			for(j = 0; j < 3; j++)
				if(m_param->prev_list[i]->output_dim[j] != current->prev_dim[j]) //should also work with dense
				{
					printf("\n ERROR: CONCATENATE merge layer require all input layers to have the same spatial shape!\n");
					exit(EXIT_FAILURE);
				}
		
		for(j = 0; j < 3; j++)
			current->output_dim[j] = current->prev_dim[j];
	
		m_param->prev_nb_channels = (int*) malloc(nb_previous*sizeof(int));
		
		current->output_dim[3] = 0;
		for(i = 0; i < nb_previous; i++)
		{
			m_param->prev_nb_channels[i] = m_param->prev_list[i]->output_dim[3];
			current->output_dim[3] += m_param->prev_list[i]->output_dim[3];
		}
	}
	
	nb_features = current->output_dim[3];
	
	if(current->output_type == SPATIAL)
	{
		flat_output_dim = 1;
		for(i = 0; i < 3; i++)
			flat_output_dim *= current->output_dim[i];
	
		current->a_size       = flat_output_dim * nb_features * net->batch_size;
		current->a_dim        = flat_output_dim;
		current->a_biased_dim = flat_output_dim;
		current->a_offset     = net->batch_size;
	}
	else
	{
		nb_features += 1;
	
		current->a_size       = nb_features * net->batch_size;
		current->a_dim        = nb_features - 1;
		current->a_biased_dim = nb_features;
		current->a_offset     = 1;
	}

	current->output = (float*) calloc(current->a_size, sizeof(float));
	mem_approx += current->a_size * sizeof(float);
	
	m_param->prev_output_list = (void**) malloc(nb_previous * sizeof(void*));
	mem_approx += nb_previous * sizeof(void*);
	
	for(i = 0; i < nb_previous; i++)
		m_param->prev_output_list[i] = m_param->prev_list[i]->output;
	// prev_list[i]->output is already a GPU adress if CUDA_comp enabled as previous layers are converted on the fly
	// Just need to move this array of GPU adresses itself on the GPU memory to be access by kernels when using CUDA_comp
	
	if(!net->inference_only)
	{
		//allocate output error comming from next layer
		current->delta_o = (float*) calloc(current->a_size, sizeof(float));
		mem_approx += current->a_size * sizeof(float);
		
		m_param->prev_delta_o_list = (void**) malloc(nb_previous * sizeof(void*));
		mem_approx += nb_previous * sizeof(void*);
		
		for(i = 0; i < nb_previous; i++)
		{
			if(m_param->prev_type_list[i] != INPUT)
				m_param->prev_delta_o_list[i] = m_param->prev_list[i]->delta_o;
			else
				m_param->prev_delta_o_list[i] = NULL;
		}
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
	
	if(!silent)
	{
		printf("      Nb_previous: %d, prev_list_id:", nb_previous);
		for(i = 0; i < nb_previous; i++)
			printf(" %d", prev_list_id[i]);
		printf("\n      Type: %s, Output: %dx%dx%dx%ld, Activation: %s\n\
      Approx layer RAM/VRAM requirement: %d MB\n",
			m_type, current->output_dim[0], current->output_dim[1], current->output_dim[2], 
			nb_features, activ, (int)(mem_approx/1000000));
	}
	net->total_nb_param += 0;
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}



void merge_save(FILE *f, layer *current, int f_bin)
{
	int i;
	char layer_type = 'M';
	
	m_param = (merge_param*)current->param;
	
	int nb_previous = m_param->nb_previous;
	int prev_layer_id[nb_previous];
	
	for(i = 0; i < nb_previous; i++)
		prev_layer_id[i] = m_param->prev_list[i]->layer_id - current->layer_id;
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&nb_previous, sizeof(int), 1, f);
		fwrite(prev_layer_id, sizeof(int), nb_previous, f);
		fwrite(current->output_dim, sizeof(int), 4, f);
		print_merge_type(f, m_param->merge_type, f_bin);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"M %d", nb_previous); //output dim required for partial save loading through skip_layers
		for(i = 0; i < nb_previous; i++)
			fprintf(f, " %d", prev_layer_id[i]);
		fprintf(f, " odim%dx%dx%dx%d ", current->output_dim[0], current->output_dim[1], 
			current->output_dim[2], current->output_dim[3]);
		print_merge_type(f, m_param->merge_type, f_bin);
		fprintf(f, " ");
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n\n");
	}
}

void merge_load(network *net, FILE *f, int f_bin, int skip_layer, int silent)
{
	int i, nb_previous;
	char merge_type[40];
	char activ_type[40];
	int output_dim[4];
	int *prev_layer_id;
	
	if(!skip_layer && !silent)
		printf("Loading merge layer, M:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(&nb_previous, sizeof(int), 1, f);
		prev_layer_id = (int*) malloc(nb_previous*sizeof(int));
		// Static array scope should be enough since merge_create is called in this function
		// and the array is only used for startup (replaced with a m_param-stored array of layer pointers)
		fread(prev_layer_id, sizeof(int), nb_previous, f);
		fread(output_dim, sizeof(int), 4, f);
		fread(merge_type, sizeof(char), 40, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
	{
		fscanf(f, "%d", &nb_previous);
		prev_layer_id = (int*) malloc(nb_previous*sizeof(int));
		for(i = 0; i < nb_previous; i++)
			fscanf(f, " %d", &(prev_layer_id[i]));
		fscanf(f, " odim%dx%dx%dx%d %s %s\n", &output_dim[0], &output_dim[1], 
			&output_dim[2], &output_dim[3], merge_type, activ_type);
	}
	
	if(!skip_layer)
		merge_create(net, nb_previous, prev_layer_id, load_merge_type(merge_type), activ_type, silent);
	
	free(prev_layer_id);
}

void free_merge(layer *current)
{
	m_param = (merge_param*) current->param;
	
	free(m_param->prev_list);
	free(current->output_dim);
	free(m_param->prev_output_list);
	
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{	
		cuda_free_merge(current);
	}
	else
	#endif
	{
		free(current->output);
		free(m_param->prev_type_list);
		if(m_param->prev_nb_channels != NULL)
			free(m_param->prev_nb_channels);
		if(!current->c_network->inference_only)
		{
			free(current->delta_o);
			free(m_param->prev_delta_o_list);
		}
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}













