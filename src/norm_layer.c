
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




#include "prototypes.h"


//##############################
//       Local variables
//##############################
static norm_param *n_param;

void norm_define_activation_param(layer *current, const char *activ)
{
	int size, dim, biased_dim, offset;
	n_param = (norm_param*) current->param;
	conv_param *c_param;
	pool_param *p_param;
	
	switch(current->previous->type)
	{
		default:
		case CONV:
			c_param = (conv_param*)n_param->prev_param;
			size = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * c_param->nb_filters * current->c_network->batch_size;
			dim  = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
			biased_dim =  c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
			offset = current->c_network->batch_size;
			break;
		case POOL:
			p_param = (pool_param*)n_param->prev_param;
			size = p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2] * p_param->nb_maps * current->c_network->batch_size;
			dim  = p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2];
			biased_dim =  p_param->nb_area[0] * p_param->nb_area[1] * p_param->nb_area[2];
			offset = current->c_network->batch_size;
			break;
		case DENSE:
			printf("\nERROR: normalization layer is not authorized after dense layers atm.\n");
			exit(EXIT_FAILURE);
			break;
		case NORM:
		case LRN:
			printf("\nERROR: stacking two normalization layers is not allowed.\n");
			exit(EXIT_FAILURE);
			break;
	}
	
	switch(current->activation_type)
	{
		case RELU:
			set_relu_activ(current, size, biased_dim, dim, offset, activ);
			break;
			
		case LOGISTIC:
			set_logistic_activ(current, size, dim, biased_dim, offset, activ);
			break;
			
		case SOFTMAX:
			printf("\nERROR: softmax activation for normalization layer is not authorized\n");
			exit(EXIT_FAILURE);
			break;
			
		case YOLO:
			printf("\nERROR: YOLO activation for normalization layer is not authorized\n");
			exit(EXIT_FAILURE);
			break;
			
		case LINEAR:
		default:
			set_linear_activ(current, size, dim, biased_dim, offset);
			break;
	}
}


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

//Used to allocate a normalization layer
int norm_create(network *net, layer *previous, const char *norm_type, const char *activation, int group_size, int set_off, FILE *f_load, int f_bin)
{
	int i;
	long long int mem_approx = 0;
	layer *current;
	float eps = 0.00001f;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING NORMALIZATION LAYER ...\n", net->nb_layers);
	
	//allocate the space holder for conv layer parameters
	n_param = (norm_param*) malloc(sizeof(norm_param));

	//define the parameters values
	current->type = NORM;
	current->dropout_rate = 0.0f;
	
	current->frozen = 0;
	
	// define batch_size and set_off
	if(strncmp(norm_type, "GN", 2) != 0)
	{
		printf("Warning: Unrecognized normalization type, use default : none\n");
		n_param->group_size = 0;
		n_param->set_off= 0;
		return net->nb_layers - 1;
	}
	else
	{
		if(group_size <= 0)
		{
			printf("\nERROR: Group Normalization cannot be set with group size <= 0.\n");
			exit(EXIT_FAILURE);
		}
		n_param->group_size = group_size;
		n_param->set_off = set_off;
	}
	
	current->previous = previous;
	n_param->prev_param = current->previous->param;
	current->input = previous->output;
	
	if(current->previous == NULL)
	{
		printf("\nERROR: normalization layer is not autorized as first layer.\n");
		exit(EXIT_FAILURE);
	}
	switch(current->previous->type)
	{
		default:
		case CONV:
			n_param->data_format = CONV;
			n_param->n_dim = ((conv_param*)n_param->prev_param)->nb_filters;
			n_param->dim_offset = ((conv_param*)n_param->prev_param)->nb_area[0] 
				* ((conv_param*)n_param->prev_param)->nb_area[1] 
				* ((conv_param*)n_param->prev_param)->nb_area[2];
			n_param->output_dim = ((conv_param*)n_param->prev_param)->nb_filters * net->batch_size * n_param->dim_offset;
			break;
		case POOL:
			//printf("\nERROR: normalization layer is not authorized after pooling atm.\n");
			//exit(EXIT_FAILURE);
			n_param->data_format = CONV;
			n_param->n_dim = ((pool_param*)n_param->prev_param)->nb_maps;
			n_param->dim_offset = ((pool_param*)n_param->prev_param)->nb_area[0] 
				* ((pool_param*)n_param->prev_param)->nb_area[1] 
				* ((pool_param*)n_param->prev_param)->nb_area[2];
			n_param->output_dim = ((pool_param*)n_param->prev_param)->nb_maps * net->batch_size * n_param->dim_offset;
			break;
		case DENSE:
			printf("\nERROR: normalization layer is not authorized after dense layers atm.\n");
			n_param->data_format = DENSE;
			n_param->n_dim = ((dense_param*)n_param->prev_param)->nb_neurons;
			n_param->dim_offset = 1;
			n_param->output_dim = (n_param->n_dim+1) * net->batch_size;
			break;
		case NORM:
		case LRN:
			printf("\nERROR: stacking two normalization layers is not allowed.\n");
			exit(EXIT_FAILURE);
			break;
	}
	
	load_activ_param(current, activation);
	
	if(n_param->n_dim%n_param->group_size == 0)
		n_param->nb_group = n_param->n_dim/n_param->group_size;
	else
		n_param->nb_group = n_param->n_dim/n_param->group_size + 1;
	n_param->gamma = (float*) malloc(n_param->nb_group * sizeof(float));
	for(i = 0; i < n_param->nb_group; i++)
		n_param->gamma[i] = 1.0f;
	n_param->beta  = (float*) calloc(n_param->nb_group, sizeof(float));
	mem_approx += 2*n_param->nb_group*sizeof(float);
	
	if(net->compute_method == C_CUDA)
	{
		n_param->gamma_gpu = (float*) calloc(n_param->nb_group, sizeof(float));
		n_param->beta_gpu = (float*) calloc(n_param->nb_group, sizeof(float));
		mem_approx += 2*n_param->nb_group*sizeof(float);
	}
	
	n_param->mean = (float*) calloc(n_param->nb_group*net->batch_size, sizeof(float));
	n_param->var  = (float*) malloc(n_param->nb_group*net->batch_size * sizeof(float));
	for(i = 0; i < n_param->nb_group*net->batch_size; i++)
		n_param->var[i] = (1.0f-eps);
	
	mem_approx += 2*n_param->nb_group*net->batch_size*sizeof(float);
	
	current->output = (float*) calloc(n_param->output_dim, sizeof(float));
	mem_approx += n_param->output_dim*sizeof(float);
	
	if(!net->inference_only)
	{
		n_param->gamma_update = (float*) calloc(n_param->nb_group, sizeof(float));
		n_param->beta_update  = (float*) calloc(n_param->nb_group, sizeof(float));
		mem_approx += 2*n_param->nb_group*sizeof(float);
		
		n_param->d_gamma = (float*) calloc(n_param->nb_group*net->batch_size, sizeof(float));
		n_param->d_beta  = (float*) calloc(n_param->nb_group*net->batch_size, sizeof(float));
		mem_approx += 2*n_param->nb_group*net->batch_size*sizeof(float);
		
		if(net->compute_method == C_CUDA)
		{
			n_param->d_gamma_gpu = (float*) calloc(n_param->nb_group*net->batch_size, sizeof(float));
			n_param->d_beta_gpu  = (float*) calloc(n_param->nb_group*net->batch_size, sizeof(float));
			mem_approx += 2*n_param->nb_group*net->batch_size*sizeof(float);
		}
		
		current->delta_o = (float*) calloc(n_param->output_dim, sizeof(float));
		mem_approx += n_param->output_dim*sizeof(float);
	}
	
	current->nb_params = 2*n_param->nb_group - n_param->set_off;
	
	current->param = n_param;

	norm_define_activation_param(current, activation);

	n_param = (norm_param*)current->param;

	if(f_load != NULL)
	{
		if(f_bin)
		{
			fread(((float*)n_param->gamma), sizeof(float), n_param->nb_group, f_load);
			fread(((float*)n_param->beta), sizeof(float), n_param->nb_group, f_load);
		}
		else
		{
			for(i = 0; i < n_param->nb_group; i++)
				fscanf(f_load, "%f", &(((float*)n_param->gamma)[i]));
			for(i = 0; i < n_param->nb_group; i++)
				fscanf(f_load, "%f", &(((float*)n_param->beta)[i]));
		}	
	}
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_norm_define(current);
			mem_approx = cuda_convert_norm_layer(current);
			cuda_define_activation(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			naiv_norm_define(current);
			define_activation(current);
			break;
		default:
			break;
	}
	
	char activ[40];
	print_string_activ_param(current, activ);
	
	printf("      Group size: %d, Nb. groups: %d, Set-off: %d\n\
      Activation: %s\n\
      Nb. params: %d, Approx layer RAM/VRAM requirement: %d MB\n",
		n_param->group_size, n_param->nb_group, n_param->set_off,
		activ, 2*n_param->nb_group,(int)(mem_approx/1000000));
	net->total_nb_param += (2*n_param->nb_group - n_param->set_off);
	net->memory_footprint += mem_approx;
	
	return net->nb_layers - 1;
}



void norm_save(FILE *f, layer *current, int f_bin)
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
	
	// Note: gamma and beta are kept host size regardless of compute type.
	if(f_bin)
	{
		fwrite(n_param->gamma, sizeof(float), n_param->nb_group, f);
		fwrite(n_param->beta, sizeof(float), n_param->nb_group, f);
	}
	else
	{	
		for(i = 0; i < n_param->nb_group; i++)
			fprintf(f, "%g ", n_param->gamma[i]);
		fprintf(f,"\n");
		for(i = 0; i < n_param->nb_group; i++)
			fprintf(f, "%g ", n_param->beta[i]);
		fprintf(f,"\n\n");
	}
	
}

void norm_load(network *net, FILE *f, int f_bin)
{
	int group_size, set_off;
	char norm[40];
	char activ_type[40];
	layer *previous;
	
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

	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];
	
	norm_create(net, previous, norm, activ_type, group_size, set_off, f, f_bin);
}






