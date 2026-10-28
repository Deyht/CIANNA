
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
static norm_param *n_param;

// Public are in prototypes.h

// Private prototypes



int norm_create(network *net, int prev_layer, const char *activation, int group_size, 
	float gamma_start, int cond_layer_id, FILE *f_load, int load_optim_state, int f_bin, int silent)
{
	size_t i;
	size_t flat_output_dim = 1, nb_features;
	long long int mem_approx = 0;
	float eps = 0.000001f;
	int is_cond = 0;
	layer *current = NULL;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	if(!silent)
		printf("L:%d - CREATING NORMALIZATION LAYER ...\n", net->nb_layers - 1);
	
	current->type                = NORM;
	current->layer_id            = net->nb_layers-1;
	current->frozen              = 0;
	current->wema_replace_signal = 0;
	current->dropout_rate        = 0.0f;
	
	if(prev_layer < 0 && prev_layer + net->nb_layers-1 >= 0)
		current->previous = net->net_layers[current->layer_id + prev_layer];
	else if(prev_layer >= 0 && prev_layer < net->nb_layers-1)
		current->previous = net->net_layers[prev_layer];
	else
	{
		printf("\n ERROR: Invalid previous layer id provided!\n");
		exit(EXIT_FAILURE);
	}
	
	if(current->previous->type == INPUT)
	{
		printf("\n ERROR: NORM layer is not autorized as first layer.\n");
		exit(EXIT_FAILURE);
	}
	
	n_param = (norm_param*) malloc(sizeof(norm_param));
	current->param = n_param;
	
	current->input = current->previous->output;
	current->output_type = current->previous->output_type;
	current->output_dim = (int*) calloc(4, sizeof(int));
	for(i = 0; i < 4; i++)
		current->output_dim[i] = current->previous->output_dim[i];
	nb_features = current->output_dim[3];
	current->prev_dim = current->previous->output_dim;
	
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
		printf("\n ERROR: Norm from flat/dense layers is not supported yet. Convert to 1d conv for merging.\n");
		exit(EXIT_FAILURE);
		
		flat_output_dim = nb_features + 1;
	
		current->a_size       = flat_output_dim * net->batch_size;
		current->a_dim        = nb_features;
		current->a_biased_dim = flat_output_dim;
		current->a_offset     = 1;
	}
	
	n_param->group_size = group_size;
	
	if(n_param->group_size <= 0)
	{
		printf(" WARNING: Group Normalization cannot be set with group size <= 0, setting it to 1.\n");
		n_param->group_size = 1;
	}
	if(n_param->group_size > nb_features)
	{
		printf(" WARNING: Group size is larger than the number of input dimensions, falling back to layer normalization.\n");
		n_param->group_size = nb_features;
	}
	
	if(nb_features%n_param->group_size == 0)
		n_param->nb_group = nb_features/n_param->group_size;
	else
	{
		printf("\n ERROR: In Group Norm layer, the number of features must be a multiple of number of groups!\n");
		exit(EXIT_FAILURE);
	}
	
	n_param->gamma_start = gamma_start;
	
	n_param->mean = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
	n_param->var  = (float*) malloc(n_param->nb_group * net->batch_size * sizeof(float));
	for(i = 0; i < n_param->nb_group * net->batch_size; i++)
		n_param->var[i] = (1.0f-eps);
	
	mem_approx += 2 * n_param->nb_group * net->batch_size * sizeof(float);
	
	current->output = (float*) calloc(current->a_size, sizeof(float));
	mem_approx += current->a_size * sizeof(float);
	
	if(!net->inference_only)
	{
		n_param->temp_A = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
		n_param->temp_B = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
		mem_approx += 2 * n_param->nb_group * net->batch_size * sizeof(float);
		
		current->delta_o = (float*) calloc(current->a_size, sizeof(float));
		mem_approx += current->a_size * sizeof(float);
	}
	
	current->weights       = NULL;
	current->FP32_weights  = NULL;
	current->ema_weights   = NULL;
	current->gradient      = NULL;
	current->optimizer_var = NULL;
	
	if(cond_layer_id != auxil_input_id)
	{
		is_cond = 1;
		//account for the interleaved norm layer
		if(cond_layer_id < 0)
			cond_layer_id -= 1;
	}
	
	//Create condition projection layer	
	dense_create(net, cond_layer_id, 2*nb_features, "LIN", NULL,
		0.0f, 1, "xavier", -1.0f, f_load, load_optim_state, f_bin, 1);
	//load layer weight is already included when f_load != NULL (without header reading)
	
	n_param->auxil_layer = net->net_layers[net->nb_layers-1];
	net->net_layers[net->nb_layers-1] = NULL;
	net->nb_layers--;
	
	current->nb_params = n_param->auxil_layer->nb_params;
	
	define_activation_param(current, activation);
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_norm_define(current);
			mem_approx = cuda_convert_norm_layer(current);
			cuda_define_activation_fct(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			norm_define(current);
			define_activation_fct(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);

	//for display purposes, restore the relative cond_id to the norm layer	
	if(cond_layer_id < 0)
		cond_layer_id += 1;
	
	if(!silent)
		printf("      Group size: %d, Nb. groups: %d, Activation: %s\n\
      gamma start: %f, is cond.: %d, cond layer id: %d, \n\
      Nb. params: %ld, Approx layer RAM/VRAM requirement: %d MB\n",
			n_param->group_size, n_param->nb_group, activ,
			gamma_start, is_cond, cond_layer_id,
			current->nb_params,(int)(mem_approx/1000000));
	
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}



void norm_save(FILE *f, layer *current, int save_optim_state, int f_bin)
{
	char layer_type = 'N';
	size_t nb_features;
	int prev_id = -1, is_cond = 0, cond_id = auxil_input_id;

	n_param = (norm_param*)current->param;
	
	prev_id = current->previous->layer_id - current->layer_id;
	nb_features = current->output_dim[3];
	
	if(n_param->auxil_layer->previous->layer_id != auxil_input_id)
	{
		is_cond = 1;
		cond_id = n_param->auxil_layer->previous->layer_id - current->layer_id;
	}
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&prev_id, sizeof(int), 1, f);
		fwrite(&n_param->group_size, sizeof(int), 1, f);
		fwrite(&is_cond, sizeof(int), 1, f);
		fwrite(&cond_id, sizeof(int), 1, f);
		fwrite(&n_param->gamma_start, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"N prev%d group_size%d is_cond%d cond_id%d gamma_start%f ", 
			prev_id, n_param->group_size, is_cond, cond_id, n_param->gamma_start);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
	
	save_layer_weights(f, n_param->auxil_layer, n_param->auxil_layer->nb_params, 2*nb_features+1, 0, 0, save_optim_state, f_bin);
}

void norm_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer, int silent)
{
	int group_size, prev_id, is_cond, cond_id;
	float gamma_start;
	char activ_type[40];
	
	if(!skip_layer && !silent)
		printf("Loading norm layer, L:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(&prev_id, sizeof(int), 1, f);
		fread(&group_size, sizeof(int), 1, f);
		fread(&is_cond, sizeof(int), 1, f);
		fread(&cond_id, sizeof(int), 1, f);
		fread(&gamma_start, sizeof(float), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, " prev%d group_size%d is_cond%d cond_id%d gamma_start%f %s\n", 
			&prev_id, &group_size, &is_cond, &cond_id, &gamma_start, activ_type);
	
	if(!is_cond)
		cond_id = auxil_input_id;

	if(!skip_layer)
		norm_create(net, prev_id, activ_type, group_size, gamma_start, cond_id, f, load_optim_state, f_bin, silent);
	else
		skip_layer_weights(f, f_bin);
}


void free_norm(layer *current)
{
	n_param = (norm_param*)current->param;
	
	free_dense(n_param->auxil_layer);
	n_param->auxil_layer = NULL;
	
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{	
		cuda_free_norm(current);
	}
	else
	#endif
	{
		free(current->output);
		
		free(n_param->mean);
		free(n_param->var);
		free(n_param->temp_A);
		free(n_param->temp_B);
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}






