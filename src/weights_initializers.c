
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

// Public are in prototypes.h

// Private prototypes
void xavier_normal( void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void xavier_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void lecun_normal(  void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void lecun_uniform( void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void rand_normal(   void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void rand_uniform(  void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);


int get_init_type(const char *s_init)
{
	if(strcmp(s_init, "xavier") == 0)
		return N_XAVIER;
	else if(strcmp(s_init, "xavier_U") == 0)
		return U_XAVIER;
	else if(strcmp(s_init, "lecun") == 0)
		return N_LECUN;
	else if(strcmp(s_init, "lecun_U") == 0)
		return U_LECUN;
	else if(strcmp(s_init, "normal") == 0)
		return N_RAND;
	else if(strcmp(s_init, "uniform") == 0)
		return U_RAND;
	else
		return N_XAVIER;
}

void initialize_weights(const char *init_fct, void *weights, int dim_in, int dim_out, 
	int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	switch(get_init_type(init_fct))
	{
		default:
		case N_XAVIER:
			xavier_normal( weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
		case U_XAVIER:
			xavier_uniform(weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
		case N_LECUN:
			lecun_normal(  weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
		case U_LECUN:
			lecun_uniform( weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
		case N_RAND:
			rand_normal(   weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
		case U_RAND:
			rand_uniform(  weights, dim_in, dim_out, bias_padding, bias_padding_value, zero_padding, manual_scaling);
			break;
	}
}


//Indices computation is identical, could be merged into a single function with only a function pointer to the appropriate numerical init

void xavier_normal(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;

	printf("      Xavier Normal weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{	
			f_tab[i] = random_normal()*sqrt(2.0f/(dim_in+dim_out))*manual_scaling; 
		}
	}
	
	// Depreciated, the pivot value is now set back by the following layer in dense
	// in other layers the bias is set during input transformation by the running layer 
	if(bias_padding) 
		f_tab[size-1] = bias_padding_value;

}

void xavier_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;

	printf("      Xavier Uniform weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{	
			f_tab[i] = (random_uniform()*2.0f-1.0f)*sqrt(12.0/(dim_in+dim_out))*manual_scaling; //real Xavier normal
		}
	}
	
	if(bias_padding)
		f_tab[size-1] = bias_padding_value;
}


void lecun_normal(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;
	
	printf("      LeCun Normal weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{	
			f_tab[i] = random_normal()*sqrt(1.0f/(dim_in))*manual_scaling;
		}
	}
	
	// depreciated the pivot value is now set back by the following layer in dense
	// in other layer the bias is set during input transformation by the running layer 
	if(bias_padding) 
		f_tab[size-1] = bias_padding_value;
}

void lecun_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;
	
	printf("      LeCun Uniform weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{
			f_tab[i] = (random_uniform()*2.0f-1.0f)*sqrt(3.0f/(dim_in))*manual_scaling;
		}
	}
	
	// depreciated the pivot value is now set back by the following layer in dense
	// in other layer the bias is set during input transformation by the running layer 
	if(bias_padding) 
		f_tab[size-1] = bias_padding_value;
}


void rand_normal(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;
	
	printf("      Random Normal (user scale) weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{	
			f_tab[i] = random_normal()*manual_scaling; 
		}
	}
	
	// depreciated the pivot value is now set back by the following layer in dense
	// in other layer the bias is set during input transformation by the running layer 
	if(bias_padding) 
		f_tab[size-1] = bias_padding_value;

}

void rand_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling)
{
	int i;
	int size;
	int limit;
	
	float* f_tab = (float*) tab;
	
	printf("      Random Uniform (user scale) weight initialization\n");
	
	size = (dim_in+zero_padding)*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(((i) % (dim_in+zero_padding+bias_padding) >= (dim_in + bias_padding)) 
			|| (bias_padding && (i+1) % (dim_in + bias_padding) == 0))
			f_tab[i] = 0.0;
		else
		{	
			f_tab[i] = (random_uniform()*2.0f-1.0f)*manual_scaling; 
		}
	}
	
	// depreciated the pivot value is now set back by the following layer in dense
	// in other layer the bias is set during input transformation by the running layer 
	if(bias_padding) 
		f_tab[size-1] = bias_padding_value;

}


