
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
static grn_param *n_param;

// Public are in prototypes.h

// Private prototypes


int grn_create(network *net, layer *previous, const char *activation, int residual, float gamma_init, FILE *f_load, int load_optim_state, int f_bin)
{
	size_t i;
	size_t flat_output_dim = 1, nb_features;
	long long int mem_approx = 0;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING GLOBAL RESPONSE NORMALIZATION LAYER ...\n", net->nb_layers);
	
	current->type = GRN;
	current->frozen = 0;
	current->wema_replace_signal = 0;
	current->dropout_rate = 0.0f; //no dropout authorized here
	current->previous = previous;
	
	if(previous == NULL)
	{
		printf("\n ERROR: GRN layer is not autorized as first layer.\n");
		exit(EXIT_FAILURE);
	}
	
	n_param = (grn_param*) malloc(sizeof(grn_param));
	current->param = n_param;
	
	n_param->residual = residual;
	
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
	
	n_param->feature_norm = (float*) calloc(nb_features*net->batch_size, sizeof(float));
	n_param->relative_importance = (float*) calloc(nb_features*net->batch_size, sizeof(float));
	mem_approx += 2 * nb_features * net->batch_size * sizeof(float);
	
	//Only useful for th GPU implementation but allocated anyway
	n_param->mean = (float*) calloc(net->batch_size, sizeof(float));
	mem_approx += net->batch_size * sizeof(float);
	
	current->weights = (float*) calloc(2 * nb_features, sizeof(float));
	current->FP32_weights = current->weights;
	n_param->gamma = current->weights;
	n_param->beta  = ((float*)current->weights) + nb_features;
	mem_approx += 2 * nb_features * sizeof(float);
	
	for(i = 0; i < nb_features; i++)
		n_param->gamma[i] = gamma_init;
	
	current->output = (float*) calloc(current->a_size, sizeof(float));
	mem_approx += current->a_size * sizeof(float);
	
	if(!net->inference_only)
	{
		if(net->use_wema)
		{
			current->ema_weights = (float*) calloc(2*nb_features, sizeof(float));
			for(i = 0; i < 2*nb_features; i++)
				current->ema_weights[i] = ((float*)current->weights)[i];
			mem_approx += 2*nb_features*sizeof(float);
		}
		
		current->gradient = (float*) calloc(2*nb_features, sizeof(float));
		n_param->gamma_grad = current->gradient;
		n_param->beta_grad  = ((float*)current->gradient) + nb_features;
		mem_approx += 2 * nb_features * sizeof(float);
		
		n_param->d_gamma = (float*) calloc(nb_features * net->batch_size, sizeof(float));
		n_param->d_beta  = (float*) calloc(nb_features * net->batch_size, sizeof(float));
		mem_approx += 2 * nb_features * net->batch_size * sizeof(float);
		
		current->delta_o = (float*) calloc(current->a_size, sizeof(float));
		mem_approx += current->a_size * sizeof(float);
		
		mem_approx += define_optimizer_var(current, 2*nb_features);
	}
	
	current->nb_params = 2 * nb_features;
	
	if(f_load != NULL)
		load_layer_weights(f_load, current, 2*nb_features, nb_features, 0, load_optim_state, f_bin);
	
	define_activation_param(current, activation);
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_grn_define(current);
			mem_approx = cuda_convert_grn_layer(current);
			//optim is done on CPU for GRN for the moment
			cuda_define_activation_fct(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			grn_define(current);
			define_activation_fct(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);
	
	printf("      N. features %ld, Activation: %s\n\
      Nb. params: %ld, Approx layer RAM/VRAM requirement: %d MB\n",
		nb_features, activ, current->nb_params,(int)(mem_approx/1000000));
	net->total_nb_param += current->nb_params;
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}



void grn_save(FILE *f, layer *current, int save_optim_state, int f_bin)
{
	char layer_type = 'G';
	size_t nb_features;

	n_param = (grn_param*)current->param;	
	nb_features = current->output_dim[3];
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&n_param->residual, sizeof(int), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"G res_%d ", n_param->residual);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
	
	save_layer_weights(f, current, 2*nb_features, nb_features, 0, 0, save_optim_state, f_bin);
}

void grn_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer)
{
	int residual, nb_features;
	float temp_read;
	char activ_type[40];
	layer *previous;
	
	if(!skip_layer)
		printf("Loading GRN layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&residual, sizeof(int), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
	{
		fscanf(f, " res_%d %s\n", &residual, activ_type);
	}

	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
			previous = NULL;
		else
			previous = net->net_layers[net->nb_layers-1];
		
		grn_create(net, previous, activ_type, residual, 1.0f, f, load_optim_state, f_bin);
	}
	else
	{
		nb_features = net->skip_in_dims[3];
		
		if(f_bin)
			fseek(f, nb_features*2, SEEK_CUR);
		else
			for(int i = 0; i < nb_features*2; i++)
				fscanf(f, "%f", &temp_read);
	}
}


void free_grn(layer *current)
{
	n_param = (grn_param*)current->param;
	
	
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{	
		cuda_free_grn(current);
	}
	else
	#endif
	{
		free(current->weights); //gamma and beta
		free(current->output);
		
		free(n_param->feature_norm);
		free(n_param->relative_importance);
		free(n_param->mean);
		
		if(!current->c_network->inference_only)
		{
			if(current->c_network->use_wema)
				free(current->ema_weights);
			free(current->gradient);
			free(current->delta_o);
			
			free(n_param->d_gamma);
			free(n_param->d_beta);
			
			free_optimizer_var(current);
		}
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}






