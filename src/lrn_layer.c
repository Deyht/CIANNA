
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
static lrn_param *n_param;

// Public are in prototypes.h

// Private prototypes


int lrn_create(network *net, int prev_layer, const char *activation, int range, float k, float alpha, float beta, int silent)
{
	int i;
	size_t flat_output_dim = 1;
	long long int mem_approx = 0;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	if(!silent)
		printf("L:%d - CREATING LOCAL RESPONSE NORMALIZATION LAYER ...\n", net->nb_layers - 1);
	
	current->type         = LRN;
	current->layer_id     = net->nb_layers-1;
	current->frozen       = 0;
	current->dropout_rate = 0.0f;
	
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
		printf("\n ERROR: normalization layer is not autorized as first layer.\n");
		exit(EXIT_FAILURE);
	}
	
	n_param = (lrn_param*) malloc(sizeof(lrn_param));
	current->param = n_param;
	
	current->input = current->previous->output;
	current->output_type = current->previous->output_type;
	current->output_dim = (int*) calloc(4, sizeof(int));
	for(i = 0; i < 4; i++)
		current->output_dim[i] = current->previous->output_dim[i];
	
	n_param->range = range;
	n_param->k     = k;
	n_param->alpha = alpha;
	n_param->beta  = beta;
	n_param->local_scale = NULL;
	
	if(current->output_type == SPATIAL)
	{
		for(i = 0; i < 4; i++)
			flat_output_dim *= current->output_dim[i];
	
		current->a_size       = flat_output_dim * net->batch_size;
		current->a_dim        = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
		current->a_biased_dim = current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
		current->a_offset     = net->batch_size;
	}
	else
	{
		printf("\n ERROR: Norm from flat/dense layers is not supported yet. Convert to 1d conv for merging.\n");
		exit(EXIT_FAILURE);
		
		flat_output_dim = current->output_dim[3] + 1;
	
		current->a_size       = flat_output_dim * net->batch_size;
		current->a_dim        = current->output_dim[3];
		current->a_biased_dim = flat_output_dim;
		current->a_offset     = 1;
	}
	
	current->output = (float*) calloc(flat_output_dim * net->batch_size, sizeof(float));
	mem_approx += flat_output_dim * net->batch_size * sizeof(float);
	
	if(!net->inference_only)
	{
		n_param->local_scale = (float*) calloc(flat_output_dim * net->batch_size, sizeof(float));
		mem_approx += flat_output_dim * net->batch_size * sizeof(float);
	
		current->delta_o = (float*) calloc(flat_output_dim * net->batch_size, sizeof(float));
		mem_approx += flat_output_dim * net->batch_size * sizeof(float);
	}
	
	current->nb_params = 0;
	
	define_activation_param(current, activation);
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_lrn_define(current);
			mem_approx = cuda_convert_lrn_layer(current);
			cuda_define_activation_fct(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			lrn_define(current);
			define_activation_fct(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);
	
	if(!silent)
		printf("      Range: %d, k: %f, Alpha: %f, Beta: %f, Activation: %s\n\
      Approx layer RAM/VRAM requirement: %d MB\n",
			n_param->range, n_param->k, n_param->alpha, n_param->beta,
		activ, (int)(mem_approx/1000000));
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}


void lrn_save(FILE *f, layer *current, int f_bin)
{
	char layer_type = 'L';
	int prev_id = -1;
	
	n_param = (lrn_param*)current->param;
	
	prev_id = current->previous->layer_id - current->layer_id;
	
	if(f_bin)
	{
		fwrite(&layer_type    , sizeof(char)  , 1, f);
		fwrite(&prev_id       , sizeof(int)   , 1, f);
		fwrite(&n_param->range, sizeof(int)   , 1, f);
		fwrite(&n_param->k    , sizeof(float) , 1, f);
		fwrite(&n_param->alpha, sizeof(float) , 1, f);
		fwrite(&n_param->beta , sizeof(float) , 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"L prev%d ", prev_id);
		fprintf(f, "%d %f %f %f ", n_param->range, n_param->k, n_param->alpha, n_param->beta);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
}


void lrn_load(network *net, FILE *f, int f_bin, int skip_layer, int silent)
{
	int range, prev_id;
	float k, alpha, beta;
	char activ_type[40];
	
	if(!skip_layer)
		printf("Loading Local Response Normalization layer, L:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(&prev_id, sizeof(int), 1, f);
		fread(&range, sizeof(int)   , 1, f);
		fread(&k    , sizeof(float) , 1, f);
		fread(&alpha, sizeof(float) , 1, f);
		fread(&beta , sizeof(float) , 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, " prev%d %d %f %f %f %s", &prev_id, &range, &k, &alpha, &beta, activ_type);

	if(!skip_layer)
		lrn_create(net, prev_id, activ_type, range, k, alpha, beta, silent);
}


void free_lrn(layer *current)
{
	n_param = (lrn_param*) current->param;
	
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{
		cuda_free_lrn(current);
	}
	else
	#endif
	{
		free(current->output);
		
		if(!current->c_network->inference_only)
		{
			free(n_param->local_scale);
			free(current->delta_o);
		}
	}
	
	if(current->activ_param != NULL)
		free(current->activ_param);
	free(current->param);
	free(current);
}






