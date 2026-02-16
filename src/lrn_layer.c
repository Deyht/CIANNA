
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
void lrn_define_activation_param(layer *current, const char *activ);


void lrn_define_activation_param(layer *current, const char *activ)
{
	int size, dim, biased_dim, offset;
	n_param = (lrn_param*) current->param;
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
			printf("\n ERROR: normalization layer is not authorized after dense layers atm.\n");
			exit(EXIT_FAILURE);
			break;
		case NORM:
		case LRN:
			printf("\n ERROR: stacking two normalization layers is not allowed.\n");
			exit(EXIT_FAILURE);
			break;
	}
	
	
	switch(current->activation_type)
	{
		case RELU:
			set_relu_param(current, size, dim, biased_dim, offset, activ);
			break;
			
		case LOGISTIC:
			set_logistic_param(current, size, dim, biased_dim, offset, activ);
			break;
			
		case SOFTMAX:
			printf("\n ERROR: softmax activation for normalization layer is not authorized\n");
			exit(EXIT_FAILURE);
			break;
			
		case YOLO:
			printf("\n ERROR: YOLO activation for normalization layer is not authorized\n");
			exit(EXIT_FAILURE);
			break;
			
		case LINEAR:
		default:
			set_linear_param(current, size, dim, biased_dim, offset);
			break;
	}
}

//public are in prototypes.h

int lrn_create(network *net, layer *previous, const char *activation, int range, float k, float alpha, float beta)
{
	long long int mem_approx = 0;
	layer *current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING LOCAL RESPONSE NORMALIZATION LAYER ...\n", net->nb_layers);
	
	//allocate the space holder for conv layer parameters
	n_param = (lrn_param*) malloc(sizeof(lrn_param));

	//define the parameters values
	current->type = LRN;
	current->dropout_rate = 0.0f;
	
	current->frozen = 0;
	
	current->previous = previous;
	n_param->prev_param = current->previous->param;
	current->input = previous->output;
	
	if(current->previous == NULL)
	{
		printf("\n ERROR: normalization layer is not autorized as first layer.\n");
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
			n_param->data_format = CONV;
			n_param->n_dim = ((pool_param*)n_param->prev_param)->nb_maps;
			n_param->dim_offset = ((pool_param*)n_param->prev_param)->nb_area[0] 
				* ((pool_param*)n_param->prev_param)->nb_area[1] 
				* ((pool_param*)n_param->prev_param)->nb_area[2];
			n_param->output_dim = ((pool_param*)n_param->prev_param)->nb_maps * net->batch_size * n_param->dim_offset;
			break;
		case DENSE:
			printf("\n ERROR: normalization layer is not authorized after dense layers atm.\n");
			n_param->data_format = DENSE;
			n_param->n_dim = ((dense_param*)n_param->prev_param)->nb_neurons;
			n_param->dim_offset = 1;
			n_param->output_dim = (n_param->n_dim+1) * net->batch_size;
			break;
		case NORM:
		case LRN:
			printf("\n ERROR: stacking two normalization layers is not allowed.\n");
			exit(EXIT_FAILURE);
			break;
	}
	
	load_activation_type(current, activation);
	
	n_param->range = range;
	n_param->k = k;
	n_param->alpha = alpha;
	n_param->beta = beta;
	n_param->local_scale = NULL;
	
	current->output = (float*) calloc(n_param->output_dim, sizeof(float));
	mem_approx += n_param->output_dim*sizeof(float);
	
	if(!net->inference_only)
	{
		n_param->local_scale = (float*) calloc(n_param->output_dim, sizeof(float));
		mem_approx += n_param->output_dim*sizeof(float);
	
		current->delta_o = (float*) calloc(n_param->output_dim, sizeof(float));
		mem_approx += n_param->output_dim*sizeof(float);
	}
	
	current->nb_params = 0;
	
	current->param = n_param;

	lrn_define_activation_param(current, activation);

	n_param = (lrn_param*)current->param;
	
	//associate the conv specific functions to the layer
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_lrn_define(current);
			mem_approx = cuda_convert_lrn_layer(current);
			cuda_define_activation(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
			printf("\n ERROR: LRN layer is only available with CUDA compute method ATM.\n");
			exit(EXIT_FAILURE);
			break;
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);
	
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

	n_param = (lrn_param*)current->param;	
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&n_param->range, sizeof(int), 1, f);
		fwrite(&n_param->k, sizeof(float), 1, f);
		fwrite(&n_param->alpha, sizeof(float), 1, f);
		fwrite(&n_param->beta, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f,"L ");
		fprintf(f, "%d %f %f %f ", n_param->range, n_param->k, n_param->alpha, n_param->beta);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
}

void lrn_load(network *net, FILE *f, int f_bin, int skip_layer)
{
	int range;
	float k, alpha, beta;
	char activ_type[40];
	layer *previous;
	
	if(!skip_layer)
		printf("Loading Local Response Normalization layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&range, sizeof(int), 1, f);
		fread(&k, sizeof(float), 1, f);
		fread(&alpha, sizeof(float), 1, f);
		fread(&beta, sizeof(float), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
	{
		fscanf(f, " %d %f %f %f %s", &range, &k, &alpha, &beta, activ_type);
	}

	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
			previous = NULL;
		else
			previous = net->net_layers[net->nb_layers-1];
	
		lrn_create(net, previous, activ_type, range, k, alpha, beta);
	}
}

void get_lrn_output_dim(layer *current, int *dim)
{
	int i;
	pool_param *p_param;
	conv_param *c_param;
	
	if(current->previous == NULL)
	{
		printf("\n ERROR: incompatible previous type in norm layer!\n");
		exit(EXIT_FAILURE);
	}
	
	switch(current->previous->type)
	{
		case POOL:
			p_param = (pool_param*) current->previous->param;
			for (i = 0; i < 3; i++)
				dim[i] = p_param->nb_area[i];
			dim[3] = p_param->nb_maps;
			break;
		
		case CONV:
			c_param = (conv_param*) current->previous->param;
			for (i = 0; i < 3; i++)
				dim[i] = c_param->nb_area[i];
			dim[3] = c_param->nb_filters;
			break;
		
		default:
			printf("\n ERROR: incompatible previous type in norm layer!\n");
			exit(EXIT_FAILURE);
			break;
	}
	//lrn layer has no impact on skip_input_dim
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
	
	free(current->activ_param);
	free(current->param);
	free(current);
}






