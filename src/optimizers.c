
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

// Public are in "prototypes.h"

// Private prototypes
size_t define_sgd_var(layer *current, size_t param_size);
size_t define_adam_var(layer *current, size_t param_size);
size_t define_rmsprop_var(layer *current, size_t param_size);

void update_weights_sgd_fct(float *weights, float *ema_weights, void* gradient,
	float weight_decay, int decoupled_wdecay, float wema_rate,
	float learning_rate, int batch_size, float momentum, float *velocity,
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size);
void update_weights_adam_fct(float *weights, float *ema_weights, void* gradient,
	float weight_decay, int decoupled_wdecay, float wema_rate,
	float learning_rate, int batch_size,
	float beta_1, float beta_2, float eps, int ams_grad, int iter,
	float *first_mom, float *second_mom, float *max_second_mom,
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size);
void update_weights_rmsprop_fct(float *weights, float *ema_weights, void* gradient,
	float weight_decay, int decoupled_wdecay, float wema_rate, float learning_rate, int batch_size, 
	float alpha, float momentum, float eps, int centered,
	float *sqrt_avg, float *velocity, float *grad_avg,
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size);

void update_weights_sgd(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights);
void update_weights_adam(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights);
void update_weights_rmsprop(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights);

void save_sgd_state(FILE *f, layer *current, size_t param_size,
	size_t return_dim, size_t padding, int stay_on_host, int f_bin);
void save_adam_state(FILE *f, layer *current, size_t param_size,
	size_t return_dim, size_t padding, int stay_on_host, int f_bin);
void save_rmsprop_state(FILE *f, layer *current, size_t param_size,
	size_t return_dim, size_t padding, int stay_on_host, int f_bin);

void free_sgd_var(layer *current);
void free_adam_var(layer *current);
void free_rmsprop_var(layer *current);


//#####################################################
//		          SGD related functions
//#####################################################


void set_sgd_param_from_string(network *net, const char* string)
{
	char *temp = NULL;
	float momentum = 0.0f;
	
	if(strncmp(string, "SGD", 3) != 0)
	{
		printf("\n ERROR: Cannot change optimizer type after init!\n");
		exit(EXIT_FAILURE);
	}
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (sgd_param*) malloc(sizeof(sgd_param));
	
	sgd_param *temp_param = (sgd_param*) net->optimizer_param;
	temp_param->momentum = 0.0f;
	
	temp = strstr(string, "_mom");
	if(temp != NULL)
		sscanf(temp, "_mom%f", &momentum);
	
	if(net->nb_layers > 0 && (momentum > 0.01f && temp_param->momentum <= 0.01f))
		printf(" Warning: SGD momentum cannot be activated after some layers have been initialized!\n");
	else
		temp_param->momentum = momentum;
	
	printf("SGD optimizer set with momentum: %0.4f\n", temp_param->momentum);
}


size_t define_sgd_var(layer *current, size_t param_size)
{
	sgd_param *o_param = (sgd_param*)current->c_network->optimizer_param;
	sgd_var *o_var = (sgd_var*) malloc(sizeof(sgd_var));
	current->optimizer_var = o_var;
	
	if(o_param->momentum < 0.01f)
		return 0;
	
	o_var->velocity = (float*) calloc(param_size, sizeof(float*));
	
	return param_size;
}


void update_weights_sgd_fct(float *weights, float *ema_weights, void *gradient,
	float weight_decay, int decoupled_wdecay, float wema_rate, 
	float learning_rate, int batch_size, float momentum, float *velocity,
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size)
{
	size_t i;
	float *c_gradient = gradient;
	int decay_mask = 1;
	float l_grad;
	
	#pragma omp parallel for private(decay_mask,l_grad) schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++) 
	{
		decay_mask = 1;
		if(((i+1) % flat_dim_offset) >= bias_weight_offset)
			decay_mask = 0; /*prevent decay for the bias weights*/
		
		if(decoupled_wdecay)
		{
			weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];
			l_grad = c_gradient[i]/batch_size;
		}
		else
			l_grad = c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];
		
		if(momentum >= 0.01f) //else, velocity is NULL (unallocated)
		{
			velocity[i] = learning_rate*l_grad + momentum*velocity[i];
			weights[i] -= velocity[i];
		}
		else
			weights[i] -= learning_rate*l_grad;
		
		if(wema_rate >= 0.01f)
			ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];
	}
}


void update_weights_sgd(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	sgd_param *o_param = (sgd_param*) net->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	
	update_weights_sgd_fct(current->weights, current->ema_weights, current->gradient, 
		net->weight_decay, net->decoupled_wdecay, net->wema_rate, net->learning_rate, net->length, 
		o_param->momentum, o_var->velocity, bias_weight_offset, flat_dim_offset, nb_weights);
}


void fprint_sgd_header(FILE *f, network *net, int f_bin)
{
	char *opt_name = "SGD";
	sgd_param *o_param = (sgd_param*) net->optimizer_param;
	
	if(f_bin)
	{
		fwrite(opt_name, sizeof(char), strlen(opt_name), f);
		fwrite(&o_param->momentum, sizeof(float), 1, f);
	}
	else
		fprintf(f, "%s_mom%g", opt_name, o_param->momentum);
}


void save_sgd_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int stay_on_host, int f_bin)
{
	network *net = current->c_network;
	sgd_param *o_param = (sgd_param*)net->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	
	if(o_param->momentum < 0.01f)
		return;
		
	fprint_layer_params(net, f, o_var->velocity, param_size, return_dim, padding, stay_on_host, f_bin);
}


int load_sgd_bin_config(FILE *f, network *net, int override)
{
	char *opt_name = "SGD";
	char temp_name[10];
	float momentum;

	fread(temp_name, sizeof(char), strlen(opt_name), f);
		if(strncmp(opt_name, temp_name, strlen(opt_name)) != 0)
			return 0;
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (sgd_param*) malloc(sizeof(sgd_param));
	sgd_param *o_param = (sgd_param*)net->optimizer_param;
	
	fread(&momentum, sizeof(float), 1, f);
	
	if(override)
	{
		o_param->momentum = momentum;
		return 1;
	}
	else
	{
		if((o_param->momentum > 0.01f && momentum <= 0.01f)
			|| (o_param->momentum <= 0.01f && momentum > 0.01f))
			return 0;
		else
		{
			o_param->momentum = momentum;
			return 1;
		}
	}
}


void load_sgd_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int f_bin)
{
	network *net = current->c_network;
	sgd_param *o_param = (sgd_param*)net->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	
	if(o_param->momentum < 0.01f)
		return;
		
	fread_layer_params(net, f, o_var->velocity, param_size, return_dim, padding, f_bin);
}


void free_sgd_var(layer *current)
{
	sgd_param *o_param = (sgd_param*) current->c_network->optimizer_param;
	sgd_var *o_var = (sgd_var*) current->optimizer_var;
	
	if(o_param->momentum < 0.01f)
		return;
	
	free(o_var->velocity);
}


//#####################################################
//		          ADAM related functions
//#####################################################


void set_adam_param_from_string(network *net, const char* string)
{
	char *temp = NULL;
	int ams_grad = 0;
	
	if(strncmp(string, "ADAM", 4) != 0)
	{
		printf("\n ERROR: Cannot change optimizer type after init!\n");
		exit(EXIT_FAILURE);
	}
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (adam_param*) malloc(sizeof(adam_param));
	adam_param *temp_param = (adam_param*) net->optimizer_param;
	
	temp_param->beta_1 = 0.9f;
	temp_param->beta_2 = 0.999f;
	temp_param->eps = 0.00000001f;
	temp_param->ams_grad = 0;
	
	
	temp = strstr(string, "_b1");
	if(temp != NULL)
		sscanf(temp, "_b1%f", &temp_param->beta_1);
	temp = strstr(string, "_b2");
	if(temp != NULL)
		sscanf(temp, "_b2%f", &temp_param->beta_2);
	temp = strstr(string, "_eps");
	if(temp != NULL)
		sscanf(temp, "_eps%f", &temp_param->eps);
	temp = strstr(string, "_ams");
	if(temp != NULL)
		sscanf(temp, "_ams%d", &ams_grad);
	
	if(net->nb_layers > 0 && ams_grad != temp_param->ams_grad)
		printf(" Warning: ams_grad cannot be activated after some layers have been initialized!\n");
	else
		temp_param->ams_grad = ams_grad;
	
	printf("ADAM optimizer set with beta_1: %0.4f, beta_2: %0.4f, eps: %g, ams_grad: %d\n",
		temp_param->beta_1, temp_param->beta_2, temp_param->eps, temp_param->ams_grad);
	
}


size_t define_adam_var(layer *current, size_t param_size)
{
	adam_param *o_param = (adam_param*)current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) malloc(sizeof(adam_var));
	current->optimizer_var = o_var;
	
	o_var->opt_step = 0;
	o_var->first_mom = (float*) calloc(param_size, sizeof(float*));
	o_var->second_mom = (float*) calloc(param_size, sizeof(float*));
	if(o_param->ams_grad <= 0)
		return 2*param_size;
	
	o_var->max_second_mom = (float*) calloc(param_size, sizeof(float*));
	
	return 3*param_size;
}


void update_weights_adam_fct(float *weights, float *ema_weights, void *gradient,
	float weight_decay, int decoupled_wdecay, float wema_rate, 
	float learning_rate, int batch_size, 
	float beta_1, float beta_2, float eps, int ams_grad, int opt_step,
	float *first_mom, float *second_mom, float *max_second_mom,
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size)
{
	size_t i;
	float *c_gradient = gradient;
	int decay_mask = 1;
	float l_grad, moving_first_mom, moving_second_mom;
	
	#pragma omp parallel for private(decay_mask, l_grad,  moving_first_mom, moving_second_mom) schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++) 
	{
		decay_mask = 1;
		if(((i+1) % flat_dim_offset) >= bias_weight_offset)
			decay_mask = 0; /*prevent decay for the bias weights*/
		
		if(decoupled_wdecay)
		{
			weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];
			l_grad = c_gradient[i]/batch_size;
		}
		else
			l_grad = c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];
		
		first_mom[i]  = beta_1*first_mom[i]  + (1.0-beta_1)*l_grad;
		second_mom[i] = beta_2*second_mom[i] + (1.0-beta_2)*l_grad*l_grad;
		
		moving_first_mom = first_mom[i] / (1.0-pow(beta_1,opt_step));
		
		if(ams_grad > 0)
		{
			max_second_mom[i] = fmax(max_second_mom[i], second_mom[i]);
			moving_second_mom = max_second_mom[i]/(1.0 - pow(beta_2,opt_step));
		}
		else
			moving_second_mom = second_mom[i]/(1.0 - pow(beta_2,opt_step));
		
		weights[i] -= learning_rate*moving_first_mom/(sqrt(moving_second_mom) + eps);
		
		if(wema_rate >= 0.01f)
			ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];
	}
}


void update_weights_adam(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	adam_param *o_param = (adam_param*) net->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	if(o_var->opt_step < SIZE_MAX)
		o_var->opt_step += 1;
	
	update_weights_adam_fct(current->weights, current->ema_weights, current->gradient, 
		net->weight_decay, net->decoupled_wdecay, net->wema_rate, net->learning_rate, net->length,
		o_param->beta_1, o_param->beta_2, o_param->eps, o_param->ams_grad, o_var->opt_step,
		o_var->first_mom, o_var->second_mom, o_var->max_second_mom,
		bias_weight_offset, flat_dim_offset, nb_weights);
}


void fprint_adam_header(FILE *f, network *net, int f_bin)
{
	char *opt_name = "ADAM";
	adam_param *o_param = (adam_param*) net->optimizer_param;
	
	if(f_bin)
	{
		fwrite(opt_name, sizeof(char), strlen(opt_name), f);
		fwrite(&o_param->beta_1, sizeof(float), 1, f);
		fwrite(&o_param->beta_2, sizeof(float), 1, f);
		fwrite(&o_param->eps, sizeof(float), 1, f);
		fwrite(&o_param->ams_grad, sizeof(int), 1, f);
	}
	else
		fprintf(f, "%s_b1%g_b2%g_eps%g_ams%d\n", 
			opt_name, o_param->beta_1, o_param->beta_2, o_param->eps, o_param->ams_grad);
}


void save_adam_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int stay_on_host, int f_bin)
{
	network *net = current->c_network;
	adam_param *o_param = (adam_param*)current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	unsigned long long int opt_step = o_var->opt_step;
	
	if(f_bin)
		fwrite(&opt_step, sizeof(unsigned long long int), 1, f);
	else
		fprintf(f, "OptStep %ld\n", o_var->opt_step);
	
	fprint_layer_params(net, f, o_var->first_mom, param_size, return_dim, padding, stay_on_host, f_bin);
	fprint_layer_params(net, f, o_var->second_mom, param_size, return_dim, padding, stay_on_host, f_bin);
	
	if(o_param->ams_grad > 0)
		fprint_layer_params(net, f, o_var->max_second_mom, param_size, return_dim, padding, stay_on_host, f_bin);
}


int load_adam_bin_config(FILE *f, network *net, int override)
{
	char *opt_name = "ADAM";
	char temp_name[10];
	float beta_1, beta_2, eps;
	int ams_grad;

	fread(temp_name, sizeof(char), strlen(opt_name), f);
		if(strncmp(opt_name, temp_name, strlen(opt_name)) != 0)
			return 0;
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (adam_param*) malloc(sizeof(adam_param));
	adam_param *o_param = (adam_param*)net->optimizer_param;
	
	fread(&beta_1, sizeof(float), 1, f);
	fread(&beta_2, sizeof(float), 1, f);
	fread(&eps, sizeof(float), 1, f);
	fread(&ams_grad, sizeof(int), 1, f);
	
	if(override)
	{
		o_param->beta_1   = beta_1;
		o_param->beta_2   = beta_2;
		o_param->eps      = eps;
		o_param->ams_grad = ams_grad;
		return 1;
	}
	else
	{
		if(o_param->ams_grad != ams_grad)
			return 0;
		else
		{
			o_param->beta_1   = beta_1;
			o_param->beta_2   = beta_2;
			o_param->eps      = eps;
			o_param->ams_grad = ams_grad;
			return 1;
		}
	}
}


void load_adam_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int f_bin)
{
	network *net = current->c_network;
	adam_param *o_param = (adam_param*)current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	unsigned long long int opt_step = o_var->opt_step;
	
	if(f_bin)
		fread(&opt_step, sizeof(unsigned long long int), 1, f);
	else
		fscanf(f, "\nOptStep %ld\n", &(o_var->opt_step));
	
	fread_layer_params(net, f, o_var->first_mom, param_size, return_dim, padding, f_bin);
	fread_layer_params(net, f, o_var->second_mom, param_size, return_dim, padding, f_bin);
	
	if(o_param->ams_grad > 0)
		fread_layer_params(net, f, o_var->max_second_mom, param_size, return_dim, padding, f_bin);
}


void free_adam_var(layer *current)
{
	adam_param *o_param = (adam_param*) current->c_network->optimizer_param;
	adam_var *o_var = (adam_var*) current->optimizer_var;
	
	free(o_var->first_mom);
	free(o_var->second_mom);
	if(o_param->ams_grad <= 0)
		return;
	
	free(o_var->max_second_mom);
}


//#####################################################
//		         RMSprop related functions
//#####################################################


void set_rmsprop_param_from_string(network *net, const char* string)
{
	char *temp = NULL;
	int centered = 0;
	float momentum = 0.0f;
	
	if(strncmp(string, "RMSprop", 7) != 0)
	{
		printf("\n ERROR: Cannot change optimizer type after init!\n");
		exit(EXIT_FAILURE);
	}
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (rmsprop_param*) malloc(sizeof(rmsprop_param));
	rmsprop_param *temp_param = (rmsprop_param*) net->optimizer_param;
	temp_param->alpha = 0.99;
	temp_param->momentum = 0.0;
	temp_param->eps = 0.00000001f;
	temp_param->centered = 0;
	
	
	temp = strstr(string, "_a");
	if(temp != NULL)
		sscanf(temp, "_a%f", &temp_param->alpha);
	temp = strstr(string, "_mom");
	if(temp != NULL)
		sscanf(temp, "_mom%f", &momentum);
	temp = strstr(string, "_eps");
	if(temp != NULL)
		sscanf(temp, "_eps%f", &temp_param->eps);
	temp = strstr(string, "_cent");
	if(temp != NULL)
		sscanf(temp, "_cent%d", &centered);
	
	if(net->nb_layers > 0 && centered != temp_param->centered)
		printf(" Warning: ams_grad cannot be activated after some layers have been initialized!\n");
	else
		temp_param->centered = centered;
	
	if(net->nb_layers > 0 && momentum != temp_param->centered)
		printf(" Warning: momentum cannot be activated after some layers have been initialized!\n");
	else
		temp_param->momentum = momentum;
	
	printf("RMSprop optimizer set with alpha: %0.4f, momentum: %0.4f, eps: %g, centered: %d\n",
		temp_param->alpha, temp_param->momentum, temp_param->eps, temp_param->centered);
}


size_t define_rmsprop_var(layer *current, size_t param_size)
{
	size_t alloc_size = param_size;
	
	rmsprop_param *o_param = (rmsprop_param*)current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) malloc(sizeof(rmsprop_var));
	current->optimizer_var = o_var;
	
	o_var->sqrt_avg = (float*) calloc(param_size, sizeof(float*));

	if(o_param->momentum >= 0.01f)
	{
		o_var->velocity = (float*) calloc(param_size, sizeof(float*));
		alloc_size += param_size;
	}	
	
	if(o_param->centered > 0)
	{
		o_var->grad_avg = (float*) calloc(param_size, sizeof(float*));
		alloc_size += param_size;
	}
	
	return alloc_size;
}


void update_weights_rmsprop_fct(float *weights, float *ema_weights, void *gradient, 
	float weight_decay, int decoupled_wdecay, float wema_rate, 
	float learning_rate, int batch_size, 
	float alpha, float momentum, float eps, int centered, 
	float *sqrt_avg, float *velocity, float *grad_avg, 
	size_t bias_weight_offset, size_t flat_dim_offset, size_t size)
{
	size_t i;
	float *c_gradient = gradient;
	int decay_mask = 1;
	float l_grad, l_sqrt_avg_centered;
	
	
	#pragma omp parallel for private(decay_mask, l_grad, l_sqrt_avg_centered) schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++) 
	{
		decay_mask = 1;
		if(((i+1) % flat_dim_offset) >= bias_weight_offset)
			decay_mask = 0; /*prevent decay for the bias weights*/
		
		if(decoupled_wdecay)
		{
			weights[i] -= learning_rate*decay_mask*weight_decay*weights[i];
			l_grad = c_gradient[i]/batch_size;
		}
		else
			l_grad = c_gradient[i]/batch_size + decay_mask*weight_decay*weights[i];
		
		sqrt_avg[i] = alpha*sqrt_avg[i] + (1.0 - alpha)*l_grad*l_grad;
		l_sqrt_avg_centered = sqrt_avg[i];
		
		if(centered > 0)
		{
			grad_avg[i] = alpha*grad_avg[i] + (1.0 - alpha)*l_grad;
			l_sqrt_avg_centered -= grad_avg[i]*grad_avg[i];
		}
		
		if(momentum >= 0.01f)
		{
			velocity[i] = momentum*velocity[i] + l_grad/(sqrt(l_sqrt_avg_centered) + eps);
			weights[i] -= learning_rate*velocity[i];
		}
		else
			weights[i] -= learning_rate*l_grad/(sqrt(l_sqrt_avg_centered) + eps);
	
		if(wema_rate >= 0.01f)
			ema_weights[i] = wema_rate*ema_weights[i] + (1.0f - wema_rate)*weights[i];
	}
}


void update_weights_rmsprop(layer *current, size_t bias_weight_offset, size_t flat_dim_offset, size_t nb_weights)
{
	network *net = current->c_network;
	rmsprop_param *o_param = (rmsprop_param*) net->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	update_weights_rmsprop_fct(current->weights, current->ema_weights, current->gradient,
		net->weight_decay, net->decoupled_wdecay, net->wema_rate, net->learning_rate, net->length,
		o_param->alpha, o_param->momentum, o_param->eps, o_param->centered,
		o_var->sqrt_avg, o_var->velocity, o_var->grad_avg,
		bias_weight_offset, flat_dim_offset, nb_weights);
}

void fprint_rmsprop_header(FILE *f, network *net, int f_bin)
{
	char *opt_name = "RMSprop";
	rmsprop_param *o_param = (rmsprop_param*) net->optimizer_param;
	
	if(f_bin)
	{
		fwrite(opt_name, sizeof(char), strlen(opt_name), f);
		fwrite(&o_param->alpha, sizeof(float), 1, f);
		fwrite(&o_param->momentum, sizeof(float), 1, f);
		fwrite(&o_param->eps, sizeof(float), 1, f);
		fwrite(&o_param->centered, sizeof(int), 1, f);
	}
	else
		fprintf(f, "%s_a%g_mom%g_eps%g_cent%d\n", 
			opt_name, o_param->alpha, o_param->momentum, o_param->eps, o_param->centered);
}


void save_rmsprop_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int stay_on_host, int f_bin)
{
	network *net = current->c_network;
	rmsprop_param *o_param = (rmsprop_param*)current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	fprint_layer_params(net, f, o_var->sqrt_avg, param_size, return_dim, padding, stay_on_host, f_bin);
		
	if(o_param->momentum >= 0.01f)
		fprint_layer_params(net, f, o_var->velocity, param_size, return_dim, padding, stay_on_host, f_bin);
		
	if(o_param->centered > 0)
		fprint_layer_params(net, f, o_var->grad_avg, param_size, return_dim, padding, stay_on_host, f_bin);
}


int load_rmsprop_bin_config(FILE *f, network *net, int override)
{
	char *opt_name = "RMSprop";
	char temp_name[10];
	float alpha, momentum, eps;
	int centered;

	fread(temp_name, sizeof(char), strlen(opt_name), f);
		if(strncmp(opt_name, temp_name, strlen(opt_name)) != 0)
			return 0;
	
	if(net->optimizer_param == NULL)
		net->optimizer_param = (rmsprop_param*) malloc(sizeof(rmsprop_param));
	rmsprop_param *o_param = (rmsprop_param*)net->optimizer_param;
	
	fread(&alpha, sizeof(float), 1, f);
	fread(&momentum, sizeof(float), 1, f);
	fread(&eps, sizeof(float), 1, f);
	fread(&centered, sizeof(int), 1, f);
	
	if(override)
	{
		o_param->alpha    = alpha;
		o_param->momentum = momentum;
		o_param->eps      = eps;
		o_param->centered = centered;
		return 1;
	}
	else
	{
		if((o_param->momentum > 0.01f && momentum <= 0.01f)
			|| (o_param->momentum <= 0.01f && momentum > 0.01f)
			|| (o_param->centered != centered))
			return 0;
		else
		{
			o_param->alpha    = alpha;
			o_param->momentum = momentum;
			o_param->eps      = eps;
			o_param->centered = centered;
			return 1;
		}
	}
}


void load_rmsprop_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int f_bin)
{
	network *net = current->c_network;
	rmsprop_param *o_param = (rmsprop_param*)current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	fread_layer_params(net, f, o_var->sqrt_avg, param_size, return_dim, padding, f_bin);
	
	if(o_param->momentum >= 0.01f)
		fread_layer_params(net, f, o_var->velocity, param_size, return_dim, padding, f_bin);
		
	if(o_param->centered > 0)
		fread_layer_params(net, f, o_var->grad_avg, param_size, return_dim, padding, f_bin);
}



void free_rmsprop_var(layer *current)
{
	rmsprop_param *o_param = (rmsprop_param*) current->c_network->optimizer_param;
	rmsprop_var *o_var = (rmsprop_var*) current->optimizer_var;
	
	free(o_var->sqrt_avg);
	
	if(o_param->momentum >= 0.01f)
		free(o_var->velocity);
	
	if(o_param->centered > 0)
		free(o_var->grad_avg);
}


//#####################################################
//		         Generic functions
//#####################################################


void set_optimizer_from_string(network* net, const char* string)
{
	if(strncmp(string, "SGD", 3) == 0)
		net->optimizer = SGD;
	else if(strncmp(string, "ADAM", 4) == 0)
		net->optimizer = ADAM;
	else if(strncmp(string, "RMSprop", 7) == 0)
		net->optimizer = RMS_PROP;
	else
	{
		printf("\n ERROR: invalid optimizer!\n");
		exit(EXIT_FAILURE);
	}
}


void set_optimizer_param_from_string(network* net, const char* string)
{
	switch(net->optimizer)
	{
		default:
		case SGD:
			set_sgd_param_from_string(net, string);
			break;
			
		case ADAM:
			set_adam_param_from_string(net, string);
			break;
			
		case RMS_PROP:
			set_rmsprop_param_from_string(net, string);
			break;
	}
}


int load_optimizer_bin_config(FILE *f, network *net, int override)
{
	switch(net->optimizer)
	{
		default:
		case SGD:
			return load_sgd_bin_config(f, net, override);
			break;
			
		case ADAM:
			return load_adam_bin_config(f, net, override);
			break;
			
		case RMS_PROP:
			return load_rmsprop_bin_config(f, net, override);
			break;
	}
}


size_t define_optimizer_var(layer *current, size_t param_size)
{
	size_t allocated_size;
	network *net = current->c_network;

	switch(net->optimizer)
	{
		default:
		case(SGD):
			allocated_size = define_sgd_var(current, param_size);
			break;
		case(ADAM):
			allocated_size = define_adam_var(current, param_size);
			break;
		case(RMS_PROP):
			allocated_size = define_rmsprop_var(current, param_size);
			break;
	}
	
	return allocated_size;
}

void set_optimizer_update_function(network *net)
{
	switch(net->optimizer)
	{
		default:
		case(SGD):
			net->optim_update_fct = update_weights_sgd;
			break;
		case(ADAM):
			net->optim_update_fct = update_weights_adam;
			break;
		case(RMS_PROP):
			net->optim_update_fct = update_weights_rmsprop;
			break;
	}
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
		cuda_set_optimizer_update_function(net);
	#endif
}


void save_optimizer_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int stay_on_host, int f_bin)
{
	switch(current->c_network->optimizer)
	{
		default:
		case(SGD):
			save_sgd_state(f, current, param_size, return_dim, padding, stay_on_host, f_bin);
			break;
		case(ADAM):
			save_adam_state(f, current, param_size, return_dim, padding, stay_on_host, f_bin);
			break;
		case(RMS_PROP):
			save_rmsprop_state(f, current, param_size, return_dim, padding, stay_on_host, f_bin);
			break;
	}
}


void load_optimizer_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int f_bin)
{
	switch(current->c_network->optimizer)
	{
		default:
		case(SGD):
			load_sgd_state(f, current, param_size, return_dim, padding, f_bin);
			break;
		case(ADAM):
			load_adam_state(f, current, param_size, return_dim, padding, f_bin);
			break;
		case(RMS_PROP):
			load_rmsprop_state(f, current, param_size, return_dim, padding, f_bin);
			break;
	}
}


void fprint_optimizer_header(FILE *f, network *net, int f_bin)
{
	char *optim_key = "OptimSave";
	
	if(f_bin)
	{
		fwrite(optim_key, sizeof(char), strlen(optim_key), f);
		fwrite(&net->use_wema, sizeof(int), 1, f);
		fwrite(&net->wema_rate, sizeof(float), 1, f);
	}
	else
	{
		fprintf(f, "%s\n", optim_key);
		fprintf(f, "use_wema %d wema_rate %g\n", net->use_wema, net->wema_rate);
	}
	
	switch(net->optimizer)
	{
		default:
		case(SGD):
			fprint_sgd_header(f, net, f_bin);
			break;
		case(ADAM):
			fprint_adam_header(f, net, f_bin);
			break;
		case(RMS_PROP):
			fprint_rmsprop_header(f, net, f_bin);
			break;
	}	
}


int fread_optim_save_state(FILE *f, network *net, int f_bin, int silent)
{
	char *optim_key = "OptimSave";
	char temp_key[20];
	char optim_config_string[100];
	int optim_save_format = 0, override = 0;
	int use_wema;
	float wema_rate;
	
	if(f_bin)
	{
		fread(temp_key, sizeof(char), strlen(optim_key), f);
		if(strncmp(optim_key, temp_key, strlen(optim_key)) == 0)
			optim_save_format = 1;
		else
		{
			optim_save_format = 0;
			fseek(f, 0, SEEK_SET);
		}
	}
	else
	{
		fscanf(f, "%s\n", temp_key);
		if(strncmp(optim_key, temp_key, strlen(optim_key)) == 0)
			optim_save_format = 1;
		else
		{
			optim_save_format = 0;
			fseek(f, 0, SEEK_SET);
		}
	}
	
	if(optim_save_format)
	{
		if(net->nb_layers == 0)
			override = 1;
		
		if(!silent)
		{
			printf("Complete optimizer save state file provided:\n");
			if(override)
			{
				printf("- No network structure exist yet -> all WEMA and optimizer\n");
	 			printf("  settings will be override using the provided file configuration.\n");
			}
			else
			{
				printf("- A network sturcture already exist -> WEMA and optimizer choices must match\n");
				printf("  those from the provided file configuration. Only the detailed configuration will be override.\n");
			}
			printf("- After loading, the detailed configuration can still be changed using the set_optimizer function and the wema_rate argument.\n");
		}
		
		if(f_bin)
		{
			fread(&use_wema, sizeof(int), 1, f);
			fread(&wema_rate, sizeof(float), 1, f);
		}
		else
			fscanf(f, "use_wema %d wema_rate %g\n", 
				&use_wema, &wema_rate);
		
		if(!override && use_wema != net->use_wema)
		{
			printf("\n ERROR: trying to load an optim save state with an different use_wema setting!\n");
			exit(EXIT_FAILURE);
		}
		net->use_wema = use_wema;
		net->wema_rate = wema_rate;
		printf("Set use_wema=%d and wema_rate=%f\n", use_wema, wema_rate);
		
		if(f_bin)
		{
			fread(temp_key, sizeof(char), 10, f);
			if(override)
			{
				set_optimizer_from_string(net, temp_key);
				free(net->optimizer_param);
				net->optimizer_param = NULL;
			}
			fseek(f, -10*sizeof(char), SEEK_CUR);
			if(!load_optimizer_bin_config(f, net, override))
			{
				printf("\n ERROR: loaded optim save file configuration is incompatible with current optimizer setup!\n");
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			fscanf(f, "%s\n", optim_config_string);
			if(override)
			{
				set_optimizer_from_string(net, optim_config_string);
				free(net->optimizer_param);
				net->optimizer_param = NULL;
			}
			set_optimizer_param_from_string(net, optim_config_string);
		}
	}
	
	return optim_save_format;
}


void free_optimizer_var(layer *current)
{
	switch(current->c_network->optimizer)
	{
		default:
		case(SGD):
			free_sgd_var(current);
			break;
		case(ADAM):
			free_adam_var(current);
			break;
		case(RMS_PROP):
			free_rmsprop_var(current);
			break;
	}
}







