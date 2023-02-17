
/*
	Copyright (C) 2020 David Cornu
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
static dense_param *d_param;

//public are in "prototypes.h"


void dense_define_activation_param(layer *current, const char* activ)
{
	int size, dim, biased_dim;
	d_param = (dense_param*) current->param;
	switch(current->activation_type)
	{
		case RELU:
			size = (d_param->nb_neurons + 1) * current->c_network->batch_size;
			dim = d_param->nb_neurons;
			biased_dim = d_param->nb_neurons+1;
			set_relu_activ(current, size, dim, biased_dim, activ);
			break;
			
		case LOGISTIC:
			size = (d_param->nb_neurons+1) * current->c_network->batch_size;
			dim = d_param->nb_neurons;
			biased_dim = d_param->nb_neurons+1;
			set_logistic_activ(current, size, dim, biased_dim, activ);
			break;
			
		case SOFTMAX:
			dim = d_param->nb_neurons;
			biased_dim = d_param->nb_neurons+1;
			set_softmax_activ(current, dim, biased_dim);
			break;
			
		case YOLO:
			printf("Error: YOLO activation is not compatible with a dense layer!");
			exit(EXIT_FAILURE);
			break;
			
		case LINEAR:
		default:
			size = (d_param->nb_neurons + 1) * current->c_network->batch_size;
			dim = d_param->nb_neurons;
			biased_dim = d_param->nb_neurons+1;
			set_linear_activ(current, size, dim, biased_dim);
			break;
	}
}


void dense_create(network *net, layer* previous, int nb_neurons, const char *activation, float *bias, float drop_rate, 
	int strict_size, const char *init_fct, float init_scaling, FILE *f_load, int f_bin)
{
	int i, j;
	//float bias_padding_value; //depreciated
	long long int mem_approx = 0;
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
	
	printf("L:%d - Creating dense layer ...\n", net->nb_layers);
	
	d_param = (dense_param*) malloc(sizeof(dense_param));
	
	current->type = DENSE;
	load_activ_param(current, activation);
	current->frozen = 0;
	d_param->nb_neurons = nb_neurons;
	current->dropout_rate = drop_rate;
	
	current->previous = previous;
	
	
	if(previous == NULL)
	{
		d_param->in_size = net->in_dims[0]*net->in_dims[1]*net->in_dims[2]*net->in_dims[3]+1;
		current->input = net->input;
	}
	else
	{
		switch(previous->type)
		{
			case CONV:
				d_param->in_size = ((conv_param*)previous->param)->nb_area[0]
					* ((conv_param*)previous->param)->nb_area[1] 
					* ((conv_param*)previous->param)->nb_area[2]
					* ((conv_param*)previous->param)->nb_filters + 1;
				d_param->flat_delta_o = (float*) calloc(d_param-> in_size * net->batch_size, sizeof(float));
				mem_approx += d_param-> in_size * net->batch_size * sizeof(float);
				d_param->flat_input = (float*) calloc(d_param->in_size*net->batch_size,sizeof(float));
				mem_approx += d_param->in_size*net->batch_size * sizeof(float);
				break;
			
			case POOL:
				d_param->in_size = ((pool_param*)previous->param)->nb_area[0] 
					* ((pool_param*)previous->param)->nb_area[1]
					* ((pool_param*)previous->param)->nb_area[2]
					* ((pool_param*)previous->param)->nb_maps + 1;
				d_param->flat_delta_o = (float*) calloc(d_param->in_size * net->batch_size, sizeof(float));
				mem_approx += d_param-> in_size * net->batch_size * sizeof(float);
				d_param->flat_input = (float*) calloc(d_param->in_size*net->batch_size,sizeof(float));
				mem_approx += d_param->in_size*net->batch_size * sizeof(float);
				break;
			
			case DENSE:
			default:
				d_param->in_size = ((dense_param*)previous->param)->nb_neurons+1;
				d_param->flat_delta_o = previous->delta_o;
				break;
		
		}
		current->input = previous->output;
	}

	d_param->weights = (float*) malloc(d_param->in_size*(nb_neurons+1)*sizeof(float));
	mem_approx += d_param->in_size*(nb_neurons+1)*sizeof(float);
	
	d_param->update = (float*) calloc(d_param->in_size*(nb_neurons+1), sizeof(float));
	mem_approx += d_param->in_size*(nb_neurons+1)*sizeof(float);
	if(drop_rate > 0.01f)
	{
		d_param->dropout_mask = (int*) calloc(d_param->nb_neurons, sizeof(int));
		mem_approx += d_param->nb_neurons * sizeof(int);
	}
	
	current->output = (float*) calloc((nb_neurons+1)*net->batch_size, sizeof(float));
	mem_approx += (nb_neurons+1)*net->batch_size * sizeof(float);
	current->delta_o = (float*) calloc((nb_neurons+1)*net->batch_size, sizeof(float));
	mem_approx += (nb_neurons+1)*net->batch_size * sizeof(float);
	
	//must be before the activation association function
	current->param = d_param;
	
	dense_define_activation_param(current, activation);
	
	d_param = (dense_param*)current->param;	
	
	if(bias != NULL)
		current->bias_value = *bias;
	
	if(current->previous == NULL)
		current->bias_value = net->input_bias;
	
	if(f_load == NULL)
	{
		if(current->previous != NULL)
		{
			if(previous->type == DENSE)
			{
				if(net->compute_method == C_CUDA)
				{
					#ifdef CUDA
					cuda_set_mem_value((void*)((float*)((dense_param*)previous->param)->weights + ((((dense_param*)previous->param)->nb_neurons+1)
						*((dense_param*)previous->param)->in_size - 1)),
						(float) current->bias_value/current->previous->bias_value, sizeof(float));
					#endif
				}
				else
				{
					*((float*)((dense_param*)previous->param)->weights + ((((dense_param*)previous->param)->nb_neurons+1)
						*((dense_param*)previous->param)->in_size - 1)) = 
						(float) current->bias_value/current->previous->bias_value;
				}
			}
			//For other previous layer types, the bias is added by the flatten (or similar) function
				
		}
		//Should add a defaut weight init depending on the activation function
		//Should add user control over the init
		if(init_scaling < 0)
			init_scaling = 1.0f;
		
		//Should add a defaut weight init depending on the activation function
		switch(get_init_type(init_fct))
		{
			default:
			case N_XAVIER:
				xavier_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
			case U_XAVIER:
				xavier_uniform(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
			case N_LECUN:
				lecun_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
			case U_LECUN:
				lecun_uniform(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
			case N_RAND:
				rand_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
			case U_RAND:
				rand_uniform(d_param->weights, d_param->nb_neurons, d_param->in_size, 1, 0.0f, 0, init_scaling);
				break;
		}
	}
	else
	{
		if(f_bin)
		{
			for(i = 0; i < d_param->in_size; i++)
				fread(&(((float*)d_param->weights)[i*(d_param->nb_neurons+1)]), sizeof(float), (d_param->nb_neurons+1), f_load);
		}
		else
		{
			for(i = 0; i < d_param->in_size; i++)
			{
				for(j = 0; j < (d_param->nb_neurons+1); j++)
					fscanf(f_load, "%f", &(((float*)d_param->weights)[i*(d_param->nb_neurons+1) + j]));
			}
		}	
	}
	
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_dense_define(current);
			cuda_define_activation(current);
			mem_approx = cuda_convert_dense_layer(current);
			#endif
			break;
			
		case C_BLAS:
			#ifdef BLAS
			blas_dense_define(current);
			define_activation(current);
			#endif
			break;
			
		case C_NAIV:
			naiv_dense_define(current);
			define_activation(current);
			break;
			
		default:
			break;
	}
	
	char activ[40];
	print_string_activ_param(current, activ);
	printf("\t Input: %d, Nb. Neurons: %d, Activation: %s, Bias: %0.2f, Dropout: %0.2f\n\
\t Nb. weights: %d, Approx layer RAM/VRAM requirement: %d MB\n",
		d_param->in_size,  d_param->nb_neurons, 
		activ, current->bias_value, current->dropout_rate,
		(d_param->nb_neurons+1)*d_param->in_size, (int)(mem_approx/1000000));
	net->total_nb_param += (d_param->nb_neurons+1)*d_param->in_size;
	net->memory_footprint += mem_approx;
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.use_cuda_TC)
	{
		if(d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0)
			printf("Warning : Forward gemm TC data misalignement due to layer size mismatch\n");
		if(current->previous != NULL && (d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0))
			printf("Warning : Backprop gemm TC data misalignment due to layer size mismatch\n");
		if(d_param->in_size % 8 != 0 || current->c_network->batch_size % 8 != 0 
				|| (d_param->nb_neurons+1) % 8 != 0)
			printf("Warning : Weights update gemm TC data misalignment due to layer size mismatch\n");
	}
	#endif
}


void dense_save(FILE *f, layer *current, int f_bin)
{
	int i, j;
	float* host_weights = NULL;
	char layer_type = 'D';

	d_param = (dense_param*)current->param;	
	
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&d_param->nb_neurons, sizeof(int), 1, f);
		fwrite(&current->dropout_rate, sizeof(float), 1, f);
		fwrite(&current->bias_value, sizeof(float), 1, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{	
		fprintf(f,"D");
		fprintf(f, "%dn%fd%fb", d_param->nb_neurons, current->dropout_rate, current->bias_value);
		print_activ_param(f, current, f_bin);
		fprintf(f,"\n");
	}
	
	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_weights = (float*) malloc(d_param->in_size * (d_param->nb_neurons+1) * sizeof(float));
		switch(current->c_network->cu_inst.use_cuda_TC)
		{
			default:
			case FP32C_FP32A:
			case TF32C_FP32A:
				cuda_get_table_FP32((float*)d_param->weights, (float*)host_weights,
					d_param->in_size * (d_param->nb_neurons+1));
				break;
			
			case FP16C_FP32A:
			case FP16C_FP16A:
				cuda_get_table_FP32((float*)d_param->FP32_weights, (float*)host_weights,
					d_param->in_size * (d_param->nb_neurons+1));
				break;
				
			case BF16C_FP32A:
				cuda_get_table_FP32((float*)d_param->FP32_weights, (float*)host_weights,
					d_param->in_size * (d_param->nb_neurons+1));
				break;
		}
		#endif
	}
	else
	{
		host_weights = d_param->weights;
	}
	
	if(f_bin)
	{
		for(i = 0; i < d_param->in_size; i++)
			fwrite(&host_weights[i*(d_param->nb_neurons+1)], sizeof(float), (d_param->nb_neurons+1), f);
	}	
	else
	{
		for(i = 0; i < d_param->in_size; i++)
		{
			for(j = 0; j < (d_param->nb_neurons+1); j++)
				fprintf(f, "%g ", host_weights[i*(d_param->nb_neurons+1) + j]);
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	
	if(current->c_network->compute_method == C_CUDA)
		free(host_weights);
}

void dense_load(network *net, FILE* f, int f_bin)
{
	int nb_neurons;
	float dropout_rate;
	float bias;
	char activ_type[40];
	layer *previous;
	
	printf("Loading dense layer, L:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(&nb_neurons, sizeof(int), 1, f);
		fread(&dropout_rate, sizeof(float), 1, f);
		fread(&bias, sizeof(float), 1, f);
		fread(activ_type, sizeof(char), 40, f);
	}
	else
		fscanf(f, "%dn%fd%fb%s\n", &nb_neurons, &dropout_rate, &bias, activ_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];

	dense_create(net, previous, nb_neurons, activ_type, &bias, dropout_rate, 1, NULL, -1.0, f, f_bin);
}








