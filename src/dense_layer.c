
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
static dense_param *d_param;

// Public are in "prototypes.h"

// Private prototypes


int dense_create(network *net, layer* previous, int nb_neurons, const char *activation, float *bias,
	float drop_rate, int strict_size, const char *init_fct, float init_scaling, FILE *f_load, int load_optim_state, int f_bin)
{
	int i, j;
	size_t flat_in_size = 1, prev_weight_matrix_size = 1;
	size_t mem_approx = 0;
	layer* current;
	
	#ifdef CUDA
	if(f_load == NULL && !strict_size && net->compute_method == C_CUDA 
		&& net->cu_inst.use_cuda_TC != FP32C_FP32A && nb_neurons % 8 == 0)
		nb_neurons -= 1;
	#endif
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	printf("L:%d - CREATING DENSE LAYER ...\n", net->nb_layers);
	
	current->type = DENSE;
	current->output_type = FLAT;
	current->frozen = 0;
	current->wema_replace_signal = 0;
	current->dropout_rate = drop_rate;
	current->previous = previous;
	
	current->output_dim = (int*) calloc(4, sizeof(int));
	for(i = 0; i < 3; i++)
		current->output_dim[i] = 1;
	current->output_dim[3] = nb_neurons;
	
	d_param = (dense_param*) malloc(sizeof(dense_param));
	current->param = d_param;
	
	if(previous == NULL)
	{
		current->prev_dim = net->in_dims;
		current->input = net->input;
	}
	else
	{
		current->prev_dim = previous->output_dim;
		current->input = previous->output;
	}
	
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	if(previous != NULL && previous->output_type == SPATIAL)
	{
		if(!net->inference_only)
		{
			d_param->flat_delta_o = (float*) calloc(flat_in_size * net->batch_size, sizeof(float));
			mem_approx += flat_in_size * net->batch_size * sizeof(float);
		}
		d_param->flat_input = (float*) calloc(flat_in_size * net->batch_size,sizeof(float));
		mem_approx += flat_in_size*net->batch_size * sizeof(float);
	}
	
	if(previous != NULL && previous->output_type == FLAT)
		d_param->flat_delta_o = previous->delta_o;

	current->weights = (float*) malloc(flat_in_size * (nb_neurons+1)*sizeof(float));
	current->FP32_weights = current->weights;
	mem_approx += flat_in_size * (nb_neurons+1) * sizeof(float);
	
	if(drop_rate > 0.01f)
	{
		current->dropout_mask = (float*) calloc((nb_neurons+1) * net->batch_size, sizeof(float));
		mem_approx += (nb_neurons+1) * net->batch_size * sizeof(float);
	}
	
	current->output = (float*) calloc((nb_neurons+1) * net->batch_size, sizeof(float));
	mem_approx += (nb_neurons+1) * net->batch_size * sizeof(float);
	
	if(!net->inference_only)
	{
		if(net->use_wema)
		{
			current->ema_weights = (float*) malloc(flat_in_size * (nb_neurons+1)*sizeof(float));
			mem_approx += flat_in_size * (nb_neurons+1) * sizeof(float);
		}
	
		current->gradient = (float*) calloc(flat_in_size * (nb_neurons+1), sizeof(float));
		mem_approx += flat_in_size * (nb_neurons+1) * sizeof(float);
		
		current->delta_o = (float*) calloc((nb_neurons+1) * net->batch_size, sizeof(float));
		mem_approx += (nb_neurons+1) * net->batch_size * sizeof(float);
		
		mem_approx += define_optimizer_var(current, flat_in_size * (nb_neurons+1));
	}
	
	current->nb_params = flat_in_size * (nb_neurons+1);
	
	if(f_load == NULL)
	{
		if(init_scaling < 0)
			init_scaling = 1.0f;
		
		initialize_weights(init_fct, current->weights, nb_neurons, flat_in_size, 1, 0.0f, 0, init_scaling);
	}
	else
		load_layer_weights(f_load, current, flat_in_size * (nb_neurons+1), nb_neurons+1, 0, load_optim_state, f_bin);
	
	if(net->use_wema && !net->inference_only && (f_load == NULL || !load_optim_state))
	{
		for(i = 0; i < flat_in_size * (nb_neurons+1); i++)
			current->ema_weights[i] = ((float*)current->weights)[i];
	}
	
	
	current->a_size       = (nb_neurons + 1) * net->batch_size;
	current->a_dim        =  nb_neurons;
	current->a_biased_dim =  nb_neurons + 1;
	current->a_offset     =  1;
	
	define_activation_param(current, activation);
	
	if(bias != NULL)
		current->bias_value = *bias;
	
	//Set pivot value in previous layer weight matrix to generate current layer input bias automatically
	if(current->previous != NULL && previous->output_type == FLAT)
	{
		for(i = 0; i < 4; i++)
			prev_weight_matrix_size *= previous->prev_dim[i];
		prev_weight_matrix_size += 1;
		prev_weight_matrix_size *= (previous->output_dim[3]+1);
		
		if(net->compute_method == C_CUDA)
		{
			//Previous layer weights have already been moved to GPU, need a cuda kernel to update the pivot.
			#ifdef CUDA
			cuda_float_memset(((float*)previous->FP32_weights + prev_weight_matrix_size - 1),
				(float)current->bias_value/current->previous->bias_value, 1);
			if(!net->inference_only && net->use_wema)
				cuda_float_memset(((float*)previous->ema_weights + prev_weight_matrix_size - 1),
					(float)current->bias_value/current->previous->bias_value, 1);
			#endif
		}
		else
		{
			*((float*)previous->weights + prev_weight_matrix_size - 1) = 
				(float) current->bias_value/current->previous->bias_value;
			if(!net->inference_only && net->use_wema)
				*((float*)previous->ema_weights + prev_weight_matrix_size - 1) = 
					(float) current->bias_value/current->previous->bias_value;
		}
	} //For spatial layers, the bias is added at flattening time, no need for the pivot trick
	
	
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_dense_define(current);
			mem_approx = cuda_convert_dense_layer(current);
			cuda_define_activation_fct(current);
			#endif
			break;
			
		case C_BLAS:
			#ifdef BLAS
			blas_dense_define(current);
			define_activation_fct(current);
			#endif
			break;
			
		case C_NAIV:
			naiv_dense_define(current);
			define_activation_fct(current);
			break;
			
		default:
			break;
	}
	
	char activ[40];
	fill_string_activ_param(current, activ,0);
	printf("      Input: %ld, Nb. Neurons: %d\n\
      Activation: %s, Bias: %0.2f, dropout rate: %0.2f\n\
      Nb. weights: %ld, Approx layer RAM/VRAM requirement: %d MB\n",
		flat_in_size,  nb_neurons, activ, current->bias_value, current->dropout_rate,
		(nb_neurons+1)*flat_in_size, (int)(mem_approx/1000000));
	net->total_nb_param += (nb_neurons+1)*flat_in_size;
	net->memory_footprint += mem_approx;
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC)
	{
		if(flat_in_size % 8 != 0 || net->batch_size % 8 != 0 
				|| (nb_neurons+1) % 8 != 0)
			printf(" WARNING: Forward gemm TC data misalignment due to layer size mismatch\n");
		if(current->previous != NULL && (flat_in_size % 8 != 0 || net->batch_size % 8 != 0 
				|| (nb_neurons+1) % 8 != 0))
			printf(" WARNING: Backprop gemm TC data misalignment due to layer size mismatch\n");
		if(flat_in_size % 8 != 0 || net->batch_size % 8 != 0 
				|| (nb_neurons+1) % 8 != 0)
			printf(" WARNING: Weights update gemm TC data misalignment due to layer size mismatch\n");
	}
	#endif
	
	return net->nb_layers - 1;
}


void dense_save(FILE *f, layer *current, int save_optim_state, int f_bin)
{
	int i, j;
	float* host_weights = NULL;
	char layer_type = 'D';
	int nb_neurons;
	size_t flat_in_size = 1;
	
	nb_neurons = current->output_dim[3];
	for(i = 0; i < 4; i++)
		flat_in_size *= current->prev_dim[i];
	flat_in_size += 1;
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&nb_neurons, sizeof(int), 1, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		fwrite(&current->bias_value, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{	
		fprintf(f,"D");
		fprintf(f, "%dn%fd%fb", nb_neurons, current->dropout_rate, current->bias_value);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
	
	save_layer_weights(f, current, flat_in_size * (nb_neurons+1), nb_neurons+1, 0, 0, save_optim_state, f_bin);
}

void dense_load(network *net, FILE* f, int load_optim_state, int f_bin, int skip_layer)
{
	int nb_neurons;
	size_t in_size;
	float dropout_rate, bias, temp_read;
	char activ_type[40];
	layer *previous;
	
	if(!skip_layer)
		printf("Loading dense layer, L:%d\n", net->nb_layers+1);
	
	if(f_bin)
	{
		fread(&nb_neurons, sizeof(int), 1, f);
		fread(&dropout_rate, sizeof(float), 1, f);
		fread(&bias, sizeof(float), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, "%dn%fd%fb%s\n", &nb_neurons, &dropout_rate, &bias, activ_type);

	if(!skip_layer)
	{
		if(net->nb_layers <= 0)
			previous = NULL;
		else
			previous = net->net_layers[net->nb_layers-1];
		
		dense_create(net, previous, nb_neurons, activ_type, &bias, dropout_rate, 1, NULL, -1.0, f, load_optim_state, f_bin);
	}
	else
	{
		in_size = net->skip_in_dims[0]*net->skip_in_dims[1]*net->skip_in_dims[2]*net->skip_in_dims[3]+1;
		
		if(f_bin)
			fseek(f, in_size*(nb_neurons+1), SEEK_CUR);
		else
			for(int i = 0; i < in_size*(nb_neurons+1); i++)
				fscanf(f, "%f", &temp_read);
	
		net->skip_in_dims[0] = nb_neurons;
		
		for(int i = 1; i < 4; i++)
			net->skip_in_dims[i] = 1;
	}
}


void free_dense(layer *current)
{
	d_param = current->param;
	
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
	{
		cuda_free_dense(current);
	}
	else
	#endif
	{
		free(current->weights);
		free(current->output);
		if(current->dropout_rate > 0.01f)
			free(current->dropout_mask);
		
		if(current->previous != NULL && current->previous->output_type != FLAT)
			free(d_param->flat_input);
		
		if(!current->c_network->inference_only)
		{
			if(current->c_network->use_wema)
				free(current->ema_weights);
			free(current->gradient);
			free(current->delta_o);
			if(current->previous != NULL && current->previous->output_type != FLAT)
				free(d_param->flat_delta_o);
			
			free_optimizer_var(current);
		}
	}
	
	free(current->activ_param);
	free(current->param);
	free(current);
}






