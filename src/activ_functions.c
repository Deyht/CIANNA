
/*
	Copyright (C) 2020 David Cornu
	for the Convolutional Interactive Artificial 
	Neural Network by/for Astrophysicists (CIANNA) Code
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


//#####################################################


void output_deriv_error(layer* current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_deriv_output_error(current);
			#endif
			break;
		
		case C_BLAS:
			#ifdef BLAS
			printf("BLAS computation do not exist yet\n");
			#endif
			break;
		default:
			printf("default computaiton do not exist yet\n");
			exit(EXIT_FAILURE);
			break;
	}	
}

void output_error_fct(layer* current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_output_error_fct(current);
			#endif
			break;
		
		case C_BLAS:
			#ifdef BLAS
			printf("BLAS computation do not exist yet\n");
			#endif
			break;
		default:
			printf("default computaiton do not exist yet\n");
			exit(EXIT_FAILURE);
			break;
	}
}


void print_activ_param(FILE *f, int type)
{
	switch(type)
	{
		case LOGISTIC:
			fprintf(f,"(LOGI)");
			break;
		
		case SOFTMAX:
			fprintf(f,"(SMAX)");
			break;
	
		case LINEAR:
			fprintf(f,"(LIN)");
			break;
	
		case RELU:
		default:
			fprintf(f,"(RELU)");
			break;
	}	
}

int load_activ_param(char *type)
{
	if(strcmp(type, "(SMAX)") == 0)
		return SOFTMAX;
	else if(strcmp(type, "(LIN)") == 0)
		return LINEAR;
	else if(strcmp(type, "(LOGI)") == 0)
		return LOGISTIC;
	else if(strcmp(type, "(RELU)") == 0)
		return RELU;
	else
		return RELU;
}


//#####################################################
//          Linear activation related functions
//#####################################################

void linear_activation(layer *current)
{
	//Do nothing in a linear activation
}

void linear_deriv(layer *current)
{
	//the linear derivativ is always 1, no action is required
}

//#####################################################


