
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
void norm_define_activation_param(layer *current, const char *activ);
void print_norm_type(FILE *f, layer *current, int f_bin);


void print_norm_type(FILE *f, layer *current, int f_bin)
{
	char temp_string[40];
	n_param = (norm_param*)current->param;
		
	sprintf(temp_string, "GN");
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s ", temp_string);
}

//public are in prototypes.h

int norm_create(network *net, layer *previous, const char *norm_type, const char *activation, int group_size, 
	int set_off, FILE *f_load, int load_optim_state, int f_bin)
{
	size_t i, nb_features;
	size_t flat_output_dim = 1;
	long long int mem_approx = 0;
	float eps = 0.000001f;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING NORMALIZATION LAYER ...\n", net->nb_layers);
	
	current->type = NORM;
	current->frozen = 0;
	current->wema_replace_signal = 0;
	current->dropout_rate = 0.0f;
	current->previous = previous;
	
	if(previous == NULL)
	{
		printf("\n ERROR: Normalization layer is not autorized as first layer.\n");
		exit(EXIT_FAILURE);
	}
	
	n_param = (norm_param*) malloc(sizeof(norm_param));
	current->param = n_param;
	
	current->input = previous->output;
	current->output_type = previous->output_type;
	current->output_dim = (int*) calloc(4, sizeof(int));
	for(i = 0; i < 4; i++)
		current->output_dim[i] = previous->output_dim[i];
	nb_features = current->output_dim[3];
	current->prev_dim = previous->output_dim;
	
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
	
	n_param->group_size = group_size;
	n_param->set_off = set_off;
	
	if(n_param->group_size <= 0)
	{
		printf(" WARNING: Group Normalization cannot be set with group size <= 0, setting it to 1.\n");
		n_param->group_size = 1;
		n_param->set_off = 0;
	}
	if(n_param->group_size > nb_features)
	{
		printf(" WARNING: Group size is larger than the number of input dimensions, falling back to layer normalization.\n");
		n_param->group_size = nb_features;
		n_param->set_off = 0;
	}
	
	if(nb_features%n_param->group_size == 0)
		n_param->nb_group = nb_features/n_param->group_size;
	else
		n_param->nb_group = nb_features/n_param->group_size + 1;
	
	current->weights = (float*) calloc(2*n_param->nb_group, sizeof(float));
	current->FP32_weights = current->weights;
	n_param->gamma = current->weights;
	n_param->beta  = ((float*)current->weights) + n_param->nb_group;
	mem_approx += 2*n_param->nb_group*sizeof(float);
	
	for(i = 0; i < n_param->nb_group; i++)
		n_param->gamma[i] = 1.0f;
	
	if(net->compute_method == C_CUDA)
	{
		n_param->gamma_gpu = (float*) calloc(n_param->nb_group, sizeof(float));
		n_param->beta_gpu = (float*) calloc(n_param->nb_group, sizeof(float));
		mem_approx += 2 * n_param->nb_group * sizeof(float);
	}
	
	n_param->mean = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
	n_param->var  = (float*) malloc(n_param->nb_group * net->batch_size * sizeof(float));
	for(i = 0; i < n_param->nb_group * net->batch_size; i++)
		n_param->var[i] = (1.0f-eps);
	
	mem_approx += 2 * n_param->nb_group * net->batch_size * sizeof(float);
	
	current->output = (float*) calloc(current->a_size, sizeof(float));
	mem_approx += current->a_size * sizeof(float);
	
	if(!net->inference_only)
	{
		if(net->use_wema)
		{
			current->ema_weights = (float*) calloc(2*n_param->nb_group, sizeof(float));
			for(i = 0; i < 2*n_param->nb_group; i++)
				current->ema_weights[i] = ((float*)current->weights)[i];
			mem_approx += 2*n_param->nb_group*sizeof(float);
		}
		
		current->gradient = (float*) calloc(2*n_param->nb_group, sizeof(float));
		n_param->gamma_update = current->gradient;
		n_param->beta_update  = ((float*)current->gradient) + n_param->nb_group;
		mem_approx += 2 * n_param->nb_group * sizeof(float);
		
		n_param->d_gamma = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
		n_param->d_beta  = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
		mem_approx += 2 * n_param->nb_group * net->batch_size * sizeof(float);
		
		if(net->compute_method == C_CUDA)
		{
			n_param->d_gamma_gpu = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
			n_param->d_beta_gpu  = (float*) calloc(n_param->nb_group * net->batch_size, sizeof(float));
			mem_approx += 2 * n_param->nb_group * net->batch_size * sizeof(float);
		}
		
		current->delta_o = (float*) calloc(current->a_size, sizeof(float));
		mem_approx += current->a_size * sizeof(float);
		
		mem_approx += define_optimizer_var(current, 2*n_param->nb_group);
	}
	
	current->nb_params = 2 * n_param->nb_group - n_param->set_off;
	
	if(f_load != NULL)
		load_layer_weights(f_load, current, 2*n_param->nb_group, n_param->nb_group, 0, load_optim_state, f_bin);
	
	define_activation_param(current, activation);
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_norm_define(current);
			mem_approx = cuda_convert_norm_layer(current);
			//optim is done on CPU for GNorm for the moment
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
	
	printf("      Group size: %d, Nb. groups: %d, Set-off: %d\n\
      Activation: %s\n\
      Nb. params: %d, Approx layer RAM/VRAM requirement: %d MB\n",
		n_param->group_size, n_param->nb_group, n_param->set_off,
		activ, 2*n_param->nb_group,(int)(mem_approx/1000000));
	net->total_nb_param += (2*n_param->nb_group - n_param->set_off);
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}



void norm_save(FILE *f, layer *current, int save_optim_state, int f_bin)
{
	int i;
	char layer_type = 'N';

	n_param = (norm_param*)current->param;	
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		print_norm_type(f, current, f_bin);
		fwrite(&n_param->group_size, sizeof(int), 1, f);
		fwrite(&n_param->set_off, sizeof(int), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"N ");
		print_norm_type(f, current, f_bin);
		fprintf(f, "S%d_O%d", n_param->group_size, n_param->set_off);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
	
	// Note: gamma and beta are kept host side regardless of compute type.
	save_layer_weights(f, current, 2*n_param->nb_group, n_param->nb_group, 0, 1, save_optim_state, f_bin);
}

void norm_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer)
{
	int group_size, set_off, nb_group;
	float temp_read;
	char norm[40];
	char activ_type[40];
	layer *previous;
	
	if(!skip_layer)
		printf("Loading norm layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(norm, sizeof(char), 40, f);
		fread(&group_size, sizeof(int), 1, f);
		fread(&set_off, sizeof(int), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
	{
		fscanf(f, " %s S%d_O%d%s\n", norm, &group_size, &set_off, activ_type);
	}

	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
			previous = NULL;
		else
			previous = net->net_layers[net->nb_layers-1];
		
		norm_create(net, previous, norm, activ_type, group_size, set_off, f, load_optim_state, f_bin);
	}
	else
	{
		if(net->skip_in_dims[3]%group_size == 0)
			nb_group = net->skip_in_dims[3]/group_size;
		else
			nb_group = net->skip_in_dims[3]/group_size + 1;
	
		if(f_bin)
			fseek(f, nb_group*2, SEEK_CUR);
		else
			for(int i = 0; i < nb_group*2; i++)
				fscanf(f, "%f", &temp_read);
	}
}


void free_norm(layer *current)
{
	n_param = (norm_param*)current->param;
	
	free(current->weights);
	
	if(!current->c_network->inference_only)
	{
		if(current->c_network->use_wema)
			free(current->ema_weights);
		free(n_param->d_gamma);
		free(n_param->d_beta);
		
		free(current->gradient);
		
		free_optimizer_var(current);
	}
	
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
		
		if(!current->c_network->inference_only)
			free(current->delta_o);
	}
	
	free(current->activ_param);
	free(current->param);
	free(current);
}






