
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
void deriv_output_error(layer *current);
void output_error_fct(layer *current);

void set_linear_param(layer *current);
void linear_activation(layer *current);
void linear_deriv(layer *previous);
void linear_deriv_output_error(layer *current);
void linear_output_error(layer *current);

void set_relu_param(layer *current, const char *activ);
void fill_string_relu_activ_param(layer *current, char *activ);
void ReLU_activation(layer *current);
void ReLU_deriv(layer *previous);
void ReLU_deriv_output_error(layer *current);
void ReLU_output_error(layer *current);

void set_logistic_param(layer *current, const char *activ);
void fill_string_logistic_activ_param(layer *current, char *activ);
void logistic_activation(layer *current);
void logistic_deriv(layer *previous);
void logistic_deriv_output_error(layer *current);
void logistic_output_error(layer *current);

void set_softmax_param(layer *current);
void softmax_activation(layer *current);
void softmax_deriv(layer *previous);
void softmax_deriv_output_error(layer *current);
void softmax_output_error(layer *current);

float IoU_fct(float *output, float *target);
float GIoU_fct(float *output, float *target);
float DIoU_fct(float *output, float *target);
float DIoU2_fct(float *output, float *target);

void set_yolo_param(layer *current);
void YOLO_activation(layer *current);
void YOLO_deriv(layer *previous);
void YOLO_deriv_output_error(layer *current);
void YOLO_output_error(layer *current);

void linear_batch_edge_fct(void *tab, size_t dim, size_t biased_dim, size_t length, size_t batch_size, size_t nb_features);
void linear_bias_deriv_fct(void *deriv, size_t biased_dim, size_t dim, size_t batch_size);
void ReLU_activation_fct(void *tab, size_t dim, size_t biased_dim, size_t offset, float saturation, float leaking_factor, size_t length, size_t size);
void ReLU_deriv_fct(void *deriv, void *value, size_t dim, size_t biased_dim, size_t offset, float saturation, float leaking_factor, size_t length, size_t size);
void quadratic_deriv_output_error(void *delta_o, void *output, void *target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void quadratic_output_error(void *output_error, void *output, void *target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void logistic_activation_fct(void *tab, float beta, float saturation, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void logistic_deriv_fct(void *deriv, void *value, float beta, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void softmax_activation_fct(void *tab, size_t dim, size_t biased_dim, size_t offset, size_t length, int batch_size, size_t size);
void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void cross_entropy_output_error(void *output_error, void *output, void *target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
void YOLO_activation_fct(void *i_tab, size_t flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax);
void YOLO_deriv_error_fct(void *i_delta_o, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d, yolo_param y_param, size_t size, size_t nb_im_iter);
void YOLO_error_fct(float *i_output_error, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d, yolo_param y_param, size_t size);


void define_activation_param(layer *current, const char *activ)
{
	//Check if previous should be last layer as this is not done by each layer anymore
	if(current->previous != NULL && current->previous->type != INPUT && current->previous->activation_type == SOFTMAX)
	{
		printf("\n ERROR: Adding new layers after a softmax / CE activated layer is not supported!\n");
		exit(EXIT_FAILURE);
	}
	//here test on null mostly verify that current->type != merge
	if(current->previous != NULL && current->previous->type != INPUT && current->previous->activation_type == YOLO)
	{
		printf("\n ERROR: Adding new layers after a YOLO activated layer is not supported!\n");
		exit(EXIT_FAILURE);
	}

	load_activation_type(current, activ);

	switch(current->activation_type)
	{
		case RELU:
			set_relu_param(current, activ);
			break;
		
		case LOGISTIC:
			set_logistic_param(current, activ);
			break;
			
		case YOLO:
			set_yolo_param(current);
			break;
		
		default:
		case LINEAR:	
			set_linear_param(current);
			break;
	}

}


//Separate from define_activation_param as CUDA need its own function as well
void define_activation_fct(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			current->activation = ReLU_activation;
			current->deriv_activation = ReLU_deriv;
			break;
		
		case LOGISTIC:
			current->activation = logistic_activation;
			current->deriv_activation = logistic_deriv;
			break;
			
		case SOFTMAX:
			current->activation = softmax_activation;
			current->deriv_activation = softmax_deriv;
			break;
			
		case YOLO:
			current->activation = YOLO_activation;
			current->deriv_activation = YOLO_deriv;
			break;
		
		default:
		case LINEAR:	
			current->activation = linear_activation;
			current->deriv_activation = linear_deriv;
			break;
	}
}


void deriv_output_error(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_deriv_output_error(current);
			break;
			
		case YOLO:
			YOLO_deriv_output_error(current);
			break;
			
		case LINEAR:
		default:
			linear_deriv_output_error(current);
			break;
	
	}
}


void output_error_fct(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_output_error(current);
			break;
		
		case YOLO:
			YOLO_output_error(current);
			break;
		
		case LINEAR:
		default:
			linear_output_error(current);
			break;
	}
}


void output_error(layer *current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_output_error_fct(current);
			#endif
			break;
		
		case C_NAIV:
		case C_BLAS:
		default:
			output_error_fct(current);
			break;
	}	
}


void output_deriv_error(layer *current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_deriv_output_error(current);
			#endif
			break;
		
		case C_NAIV:
		case C_BLAS:
			deriv_output_error(current);
			break;
			
		default:
			deriv_output_error(current);
			break;
	}	
}


void fill_string_activ_param(layer *current, char *activ, int no_param)
{
	switch(current->activation_type)
	{
		case LOGISTIC:
			if(no_param)
				sprintf(activ,"LOGI");
			else
				fill_string_logistic_activ_param(current, activ);
			break;
		case SOFTMAX:
			sprintf(activ,"SMAX");
			break;
		case YOLO:
			sprintf(activ,"YOLO");
			break;
		case RELU:
			if(no_param)
				sprintf(activ,"RELU");
			else
				fill_string_relu_activ_param(current, activ);
			break;
		case LINEAR:
		default:
			sprintf(activ,"LIN");
			break;
	}
}


void print_activ_param(FILE *f, layer *current, int f_bin)
{
	char temp_string[40];

	fill_string_activ_param(current, temp_string, 0);
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s ", temp_string);
}


void load_activation_type(layer *current, const char *activ)
{
	if(activ == NULL)
	{
		current->activation_type = LINEAR;
		return;
	}

	if(strncmp(activ, "SMAX", 4) == 0)
		current->activation_type = SOFTMAX;
	else if(strncmp(activ, "LIN", 3) == 0)
		current->activation_type = LINEAR;
	else if(strncmp(activ, "LOGI", 4) == 0)
		current->activation_type = LOGISTIC;
	else if(strncmp(activ, "YOLO", 4) == 0)
		current->activation_type = YOLO;
	else if(strncmp(activ, "RELU", 4) == 0)
		current->activation_type = RELU;
	else
		current->activation_type = LINEAR;
}


//#####################################################
//		 Linear activation related functions
//#####################################################

void set_linear_param(layer *current)
{
	current->activ_param = NULL;
	current->bias_value = 0.500000f;
}


void linear_batch_edge_fct(void *tab, size_t dim, size_t biased_dim, size_t length, size_t batch_size, size_t nb_features)
{
	size_t i, j;
	float *f_tab = (float*) tab;
	
	if(biased_dim > dim)
	{
		#pragma omp parallel for schedule(guided,4) if((batch_size-length)*biased_dim>=128)
		for(i = 0; i < (batch_size-length)*biased_dim; i++)
			f_tab[length*biased_dim + i] = 0.0f;
	}
	else
	{
		#pragma omp parallel for private(j) schedule(guided,4) if((batch_size-length)*dim*nb_features>=128)
		for(i = 0; i < nb_features; i++)
		{
			for(j = 0; j < (batch_size-length)*dim; j++)
				f_tab[i*dim*batch_size + length*dim + j] = 0.0f;
		}
	}
}


void linear_bias_deriv_fct(void *deriv, size_t biased_dim, size_t dim, size_t batch_size)
{
	size_t i;
	float *f_deriv = (float*) deriv;
	
	#pragma omp parallel for schedule(guided,4) if(batch_size>=128)
	for(i = 0; i < batch_size; i++)
		f_deriv[i*biased_dim + dim] = 0.0f;
}


void linear_activation(layer *current)
{
	size_t nb_features = 1;
	network *net = current->c_network;
	
	if(net->length >= net->batch_size)
		return;
	
	if(current->a_biased_dim <= current->a_dim)
		nb_features = current->a_size / (current->a_dim * current->a_offset);
	
	linear_batch_edge_fct(current->output, current->a_dim, current->a_biased_dim,
		net->length, net->batch_size, nb_features);
}


void linear_deriv(layer *current)
{
	size_t nb_features = 1;
	network *net = current->c_network;
	
	//FLAT layers must remove the extra bias gradient
	if(current->a_biased_dim > current->a_dim)
		linear_bias_deriv_fct(current->delta_o, current->a_biased_dim,
			current->a_dim, net->batch_size);
	
	if(net->length >= net->batch_size)
		return;
	
	if(current->a_biased_dim <= current->a_dim)
		nb_features = current->a_size / (current->a_dim * current->a_offset);
	
	linear_batch_edge_fct(current->delta_o, current->a_dim, current->a_biased_dim,
		net->length, net->batch_size, nb_features);
}


void linear_deriv_output_error(layer *current)
{
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void linear_output_error(layer *current)
{	
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->dyn_target[0], 
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


//#####################################################



//#####################################################
//		 ReLU activation related functions
//#####################################################

void set_relu_param(layer *current, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	
	param->saturation = 800.0f;
	param->leaking_factor = 0.05f;
	current->bias_value = 0.100000f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_L");
	if(temp != NULL)
		sscanf(temp, "_L%f", &param->leaking_factor);
	
}


void fill_string_relu_activ_param(layer *current, char *activ)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	sprintf(activ,"RELU_S%0.2f_L%0.2f", param->saturation, param->leaking_factor);
}


//Is in fact a leaky ReLU, to obtain true ReLU define leaking_factor to 0
void ReLU_activation_fct(void *tab, size_t dim, size_t biased_dim, size_t offset, 
	float saturation, float leaking_factor, size_t length, size_t size)
{
	size_t i;
	float *f_tab = (float*) tab;
	
	#pragma omp parallel for schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < length*biased_dim)
			{
				if((i+1)%(dim+1) != 0)
				{
					if(f_tab[i] <= 0.0f)
						f_tab[i] *= leaking_factor;
					else if(f_tab[i] > saturation)
						f_tab[i] = saturation + (f_tab[i] - saturation)*leaking_factor;
				}
			}
			else
				f_tab[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				if(f_tab[i] <= 0.0f)
					f_tab[i] *= leaking_factor;
				else if(f_tab[i] > saturation)
					f_tab[i] = saturation + (f_tab[i] - saturation)*(leaking_factor);
			}
			else
				f_tab[i] = 0.0f;
		}
	}
}


void ReLU_deriv_fct(void *deriv, void *value, size_t dim, size_t biased_dim, size_t offset,	
	 float saturation, float leaking_factor, size_t length, size_t size)
{
	size_t i;
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				if(f_value[i] <= 0.0f)
					f_deriv[i] *= leaking_factor;
				else if(f_deriv[i] > saturation)
					f_deriv[i] *= leaking_factor;
			}
			else
				f_deriv[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				if(f_value[i] <= 0.0f)
					f_deriv[i] *= leaking_factor;
				else if(f_deriv[i] > saturation)
					f_deriv[i] *= leaking_factor;
			}
			else
				f_deriv[i] = 0.0f;
		}
	}
}


void ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	ReLU_activation_fct(current->output, current->a_dim, current->a_biased_dim, current->a_offset, param->saturation, 
		param->leaking_factor, current->c_network->length, current->a_size);
}


void ReLU_deriv(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	ReLU_deriv_fct(current->delta_o, current->output, current->a_dim, current->a_biased_dim, current->a_offset, 
		param->saturation, param->leaking_factor, current->c_network->length, current->a_size);
}


void ReLU_deriv_output_error(layer *current)
{	
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->dyn_target[0], current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void ReLU_output_error(layer *current)
{	
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->dyn_target[0], 
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void quadratic_deriv_output_error(void *delta_o, void *output, void *target, size_t dim, 
	size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i;
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_delta_o = (float*) delta_o;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	size_t dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(pos) schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{	
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
	}
}


void quadratic_output_error(void *output_error, void *output, void *target, size_t dim, 
	size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i;
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	size_t dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(pos) schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_output_error[i] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
			}
			else
				f_output_error[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_output_error[i] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
			}
			else
				f_output_error[i] = 0.0f;
		}
	}
}

//#####################################################


//#####################################################
//		 Logistic activation related functions
//#####################################################


void set_logistic_param(layer *current, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
	logistic_param *param = (logistic_param*)current->activ_param;	
	
	param->saturation = 6.0f;
	param->beta = 1.0f;
	current->bias_value = -1.000000f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_B");
	if(temp != NULL)
		sscanf(temp, "_B%f", &param->beta);
	
}


void fill_string_logistic_activ_param(layer *current, char *activ)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	sprintf(activ,"LOGI_S%0.2f_B%0.2f", param->saturation, param->beta);
}


void logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	logistic_activation_fct(current->output, param->beta, param->saturation, current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void logistic_activation_fct(void *tab, float beta, float saturation, size_t dim,
	size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i = 0;
	float *f_tab = (float*) tab;

	#pragma omp parallel for schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < length*biased_dim)
			{
				if((i+1)%(dim+1) != 0)
				{
					f_tab[i] = -beta*f_tab[i];
					if(f_tab[i] > saturation)
						f_tab[i] = saturation;
					f_tab[i] = 1.0f/(1.0f + expf(f_tab[i]));
				}
			}
			else
				f_tab[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				f_tab[i] = -beta*f_tab[i];
				if(f_tab[i] > saturation)
					f_tab[i] = saturation;
				f_tab[i] = 1.0f/(1.0f + expf(f_tab[i]));
			}
			else
				f_tab[i] = 0.0f;
		}
	}
}


void logistic_deriv(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	logistic_deriv_fct(current->delta_o, current->output, param->beta, current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void logistic_deriv_fct(void *deriv, void *value, float beta, size_t dim,
	size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i;
	
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
				f_deriv[i] *= beta*f_value[i]*(1.0-f_value[i]);
			else
				f_deriv[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
				f_deriv[i] *= beta*f_value[i]*(1.0-f_value[i]);
			else
				f_deriv[i] = 0.0f;
		}
	}
}


void logistic_deriv_output_error(layer *current)
{
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->dyn_target[0], current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void logistic_output_error(layer *current)
{
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}

//#####################################################



//#####################################################
//		 Soft-Max activation related functions
//#####################################################

void set_softmax_param(layer *current)
{
	current->activ_param = NULL;
	current->bias_value = 0.100000f;
}


void softmax_activation(layer *current)
{
	softmax_activation_fct(current->output, current->a_dim, current->a_biased_dim, current->a_offset, 
		current->c_network->length, current->c_network->batch_size, current->a_size);
}


void softmax_activation_fct(void *tab, size_t dim, size_t biased_dim,
	size_t offset, size_t length, int batch_size, size_t size)
{
	size_t i;
	size_t j, k, l;
	float *pos, *off_pos;
	float vmax;
	float normal = 0.0f;
	size_t batched_dim = dim * batch_size;
	size_t nb_filters = size / batched_dim;
	
	#pragma omp parallel for private(j, k, l, pos, off_pos, vmax, normal) schedule(guided,2) if(size>=128)
	for(i = 0; i < batch_size; i++)
	{
		pos = (float*)tab + i*biased_dim;
		normal = 0.0f;
		
		if(biased_dim > dim)
		{
			if(i < length)
			{
				vmax = *pos;
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					if(*off_pos > vmax)
						vmax = *off_pos;
				}
				
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos = expf(*off_pos-vmax);
					normal += *off_pos;
				}
				pos[dim*offset] = 0.0f;
				
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos /= normal;
				}
				pos[dim*offset] = 0.0f;
			}
			else
			{
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos = 0.0f;
				}
				pos[dim*offset] = 0.0f;
			}
		}
		else
		{
			if(i < length)
			{
				vmax = *pos;
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						if(*off_pos > vmax)
							vmax = *off_pos;
					}
				}
				
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos = expf((*off_pos-vmax));
						normal += *off_pos;
					}
				}
				
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos /= normal;
					}
				}
			}
			else
			{
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos = 0.0f;
					}
				}
			}
		}
	}
}


void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, 
	size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i;
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_delta_o = (float*) delta_o; 
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	size_t length_biased_dim = length * biased_dim;
	size_t dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(c_batch, c_filter, in_filter_pos, pos) schedule(guided,2) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length_biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
	}
}


void cross_entropy_output_error(void *output_error, void *output, void *target, 
	size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size)
{
	size_t i;
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	size_t length_biased_dim = length * biased_dim;
	size_t dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(c_batch, c_filter, in_filter_pos, pos) schedule(guided,2) if(size>=128)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length_biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				if(f_output[i] > 0.000001f)
					f_output_error[i] = -f_target[pos]*logf(f_output[i]);
				else
					f_output_error[i] = -f_target[pos]*logf(0.000001f);
			}
			else
				f_output_error[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				if(f_output[i] > 0.000001f)
					f_output_error[i] = -f_target[pos]*logf(f_output[i]);
				else
					f_output_error[i] = -f_target[pos]*logf(0.000001f);
			}
			else
				f_output_error[i] = 0.0f;
		}
	}
}


void softmax_deriv(layer *current)
{
	//Empty on purpose
	//Only the last layer can be softmax activated with a cross entropy error
	//In this case, activation derivation is handled by deriv output_error directly (next function).
}


void softmax_deriv_output_error(layer *current)
{
	cross_entropy_deriv_output_error(current->delta_o, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void softmax_output_error(layer *current)
{
	cross_entropy_output_error(current->c_network->output_error, current->output, 
		current->c_network->dyn_target[0], current->a_dim, current->a_biased_dim, current->a_offset, 
		current->c_network->length, current->a_size);
}


//#####################################################


//#####################################################
//		 YOLO activation related functions
//#####################################################

void set_yolo_param(layer *current)
{
	size_t i, j;
	float *temp = NULL;
	
	current->activ_param = (yolo_param*) malloc(sizeof(yolo_param));
	yolo_param *param = (yolo_param*)current->activ_param;
	
	//From global YOLO settings
	yolo_param *global_param = (yolo_param*)current->c_network->y_param;
	
	//Content copy (and not only pointer) to keep network properties accessible
	//all necessary pointers are redifined in the following lines
	*param = *(current->c_network->y_param);
	
	if(current->output_type != SPATIAL)
	{
		printf("\n ERROR: YOLO activation is only authorized with a SPATIAL type layer!");
		exit(EXIT_FAILURE);
	}
	
	size_t nb_box = param->nb_box;
	size_t nb_class = param->nb_class;
	size_t nb_param = global_param->nb_param;
	size_t max_obj_per_image = param->max_nb_obj_per_image;
	size_t output_offset = 8+nb_class+nb_param;
	
	size_t nb_filters = current->output_dim[3];
	size_t total_nb_area = (size_t)current->output_dim[0] * current->output_dim[1] * current->output_dim[2];
	size_t batched_total_nb_area = total_nb_area * current->c_network->batch_size;
	
	if(nb_box*output_offset != nb_filters)
	{
		printf("\n ERROR: Nb filters size mismatch in YOLO dimensions!\n");
		printf("%ld %ld\n", nb_box*output_offset, nb_filters);
		exit(EXIT_FAILURE);
	}
	
	temp = (float*) calloc(6*3, sizeof(float));
	param->slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
	param->prior_size = (float*) calloc(nb_box*3, sizeof(float));
	
	/* Having prior relative to image size is a good idea in principle, but it hides the fact that the network has a given receptive field related to its architecture.
	Therefore, increasing the input resolution will end up to cases where object sizes are too large.
	For now we prefer to have priors as fixed pixel sizes, allowing to processing cutout of different sizes from a given input image (without rescale).*/
	for(i = 0; i < nb_box; i++)
		for(j = 0; j < 3; j++)
			param->prior_size[i*3+j] = fmax(1.0f, global_param->prior_size[i*3+j]);
	
	for(i = 0; i < 6; i++)
	{
		param->slopes_and_maxes_tab[i] = &temp[i*3];
		for(j = 0; j < 3; j++)
			  param->slopes_and_maxes_tab[i][j] = global_param->slopes_and_maxes_tab[i][j];
	}
	
	param->cell_size = (int*) calloc(3, sizeof(int));
	for (i = 0; i < 3; i++)
		param->cell_size[i] = current->c_network->net_layers[param->input_branch_id]->output_dim[i] / current->output_dim[i];
	
	param->IoU_monitor = (float*) calloc(2 * nb_box * batched_total_nb_area, sizeof(float));
	param->IoU_table = (float*) calloc(batched_total_nb_area * max_obj_per_image * nb_box, sizeof(float));
	param->dist_prior = (float*) calloc(batched_total_nb_area * max_obj_per_image * nb_box, sizeof(float));
	param->in_cell_target_id = (int*) calloc(batched_total_nb_area * max_obj_per_image, sizeof(int));
	param->prior_assigned_target = (int*) calloc(batched_total_nb_area * nb_box, sizeof(int));
	param->box_ids = (int*) calloc(batched_total_nb_area * nb_box, sizeof(int));
	param->box_locked = (int*) calloc(batched_total_nb_area * nb_box, sizeof(int));
	param->box_in_pix = (float*) calloc(batched_total_nb_area * 6 * nb_box, sizeof(float));
	
	current->bias_value = 0.500000f;
}


float IoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


float GIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}


//order: xmin, ymin, zmin, xmax, ymax, zmax
float DIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}


//order: xmin, ymin, zmin, xmax, ymax, zmax
float DIoU2_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2]) - inter_3d;
			
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = ((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = (enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}


int set_yolo_config(network *net, size_t nb_box, int nb_class, int nb_param, int max_nb_obj_per_image, const char *IoU_type_char, 
	const char *prior_dist_type_char, float *prior_size, float *yolo_noobj_prob_prior, int fit_dim, 
	int strict_box_size, int regret_based, int rand_startup, float rand_prob_best_box_assoc, float rand_prob, 
	float min_prior_forced_scaling, float *scale_tab, float **slopes_and_maxes_tab, float *param_ind_scale, 
	float *IoU_limits, int *fit_parts, int class_softmax, float label_smoothing,
	int diff_flag, const char *error_type, int no_override, int raw_output, int input_branch_id)
{
	size_t i;
	float *temp;
	float **sm;
	float *l_IoU_limits, *l_scale_tab;
	float **l_slopes_and_maxes_tab;
	int *l_fit_parts;
	char display_IoU_type_char[40];
	char display_prior_dist_type[40];
	char display_error_type[40];
	char display_class_type[60];
	char display_difficult[40];
	
	if(net->y_param != NULL && net->y_param->fit_dim > 0)
	{
		printf("\n ERROR: Trying to update existing YOLO layer setup is not supported yet\n");
		exit(EXIT_FAILURE);
	}
	
	// Default setting
	net->y_param = (yolo_param*) malloc(sizeof(yolo_param));
	net->y_param->nb_box = 0;
	net->y_param->IoU_type = IOU;
	net->y_param->strict_box_size_association = 0;
	net->y_param->regret_based = 0;
	net->y_param->nb_class = 0;
	net->y_param->nb_param = 0;
	net->y_param->max_nb_obj_per_image = 0;
	net->y_param->fit_dim = 0;
	
	net->y_param->rand_startup = 0;
	net->y_param->rand_prob_best_box_assoc = 0.0f;
	net->y_param->min_prior_forced_scaling = -1.0f;
	
	net->y_param->class_softmax = 0;
	net->y_param->label_smoothing = 0.0f;
	net->y_param->diff_flag = 0;
	net->y_param->no_override = 0;
	net->y_param->raw_output = 0;
	
	net->y_param->no_override = no_override;
	net->y_param->raw_output = raw_output;
	net->y_param->flat_target_size = (1+max_nb_obj_per_image*(7+nb_param+diff_flag));
	
	if(input_branch_id < 0)
		net->y_param->input_branch_id = 0;
	else
		net->y_param->input_branch_id = input_branch_id;
	
	/* Check removed for now, should think of a way to reinstore it with multi input/output design (likely at forward time)
	if(max_nb_obj_per_image > 0 && (1+max_nb_obj_per_image*(7+nb_param+diff_flag)) != net->train.branch_targ_dim[0])
	{
		printf("\n ERROR: Network output dim (target) specified in init_network and YOLO's \"max_nb_obj_per_image\" values do not match.\n");
		printf(" Output_dim should be equal to 1+max_nb_obj_per_image*(7+nb_param).\n");
		printf(" Got output_dim = %ld, and max_nb_obj_per_image = %d \n\n", net->train.branch_targ_dim[0], max_nb_obj_per_image);
		exit(EXIT_FAILURE);
	}*/
	
	if(strcmp(IoU_type_char, "IoU") == 0)
	{
		net->y_param->IoU_type = IOU;
		sprintf(display_IoU_type_char, "Classical IoU");
		net->y_param->c_IoU_fct = IoU_fct;
	}
	else if(strcmp(IoU_type_char, "GIoU") == 0)
	{
		net->y_param->IoU_type = GIOU;
		sprintf(display_IoU_type_char, "Generalized GIoU");
		net->y_param->c_IoU_fct = GIoU_fct;
	}
	else if(strcmp(IoU_type_char, "DIoU") == 0)
	{
		net->y_param->IoU_type = DIOU;
		sprintf(display_IoU_type_char, "Distance DIoU");
		net->y_param->c_IoU_fct = DIoU_fct;
	}
	else if(strcmp(IoU_type_char, "DIoU2") == 0)
	{
		net->y_param->IoU_type = DIOU2;
		sprintf(display_IoU_type_char, "Distance DIoU2");
		net->y_param->c_IoU_fct = DIoU2_fct;
	}
	else
	{
		printf(" WARNING: Unrecognized IoU type: %s, fallback to default GIoU\n", IoU_type_char);
		net->y_param->IoU_type = GIOU;
		sprintf(display_IoU_type_char, "Generalized GIoU");
		net->y_param->c_IoU_fct = GIoU_fct;
	}
	
	if(strcmp(prior_dist_type_char, "IoU") == 0 ||
		strcmp(prior_dist_type_char, "IOU") == 0)
	{
		net->y_param->prior_dist_type = DIST_IOU;
		sprintf(display_prior_dist_type, "Prior dist. IoU");
	}
	else if(strcmp(prior_dist_type_char, "SIZE") == 0)
	{
		net->y_param->prior_dist_type = DIST_SIZE;
		sprintf(display_prior_dist_type, "Prior dist. SIZE");
	}
	else if(strcmp(prior_dist_type_char, "OFFSET") == 0)
	{
		net->y_param->prior_dist_type = DIST_OFFSET;
		sprintf(display_prior_dist_type, "Prior dist. OFFSET");
	}
	else
	{
		printf(" WARNING: Unrecognized prior dist. type: %s, fallback to default dist. Size\n", prior_dist_type_char);
		net->y_param->prior_dist_type = DIST_SIZE;
		sprintf(display_prior_dist_type, "Prior dist. SIZE");
	}
	
	net->y_param->strict_box_size_association = strict_box_size;
	net->y_param->regret_based = regret_based;
	
	if(rand_startup < 0)
		net->y_param->rand_startup = 64000;
	else
		net->y_param->rand_startup = rand_startup;
	
	if(rand_prob_best_box_assoc < 0.0f)
		net->y_param->rand_prob_best_box_assoc = 0.0f;
	else
		net->y_param->rand_prob_best_box_assoc = rand_prob_best_box_assoc;
		
	if(rand_prob < 0.0f)
		net->y_param->rand_prob = 0.0f;
	else
		net->y_param->rand_prob = rand_prob;
	
	if(min_prior_forced_scaling <= 0.0f)
		net->y_param->min_prior_forced_scaling = 0.0f;
	else
		net->y_param->min_prior_forced_scaling = min_prior_forced_scaling;
	
	
	net->y_param->fit_dim = fit_dim;
	if(prior_size == NULL)
	{
		prior_size = (float*) calloc(3*nb_box, sizeof(float));
		for(i = 0; i < 3*nb_box; i++)
			prior_size[i] = 0.0f;
	}
	else
	{
		for(i = 0; i < 3*nb_box; i++)
			prior_size[i] = prior_size[i];
	}
	
	if(yolo_noobj_prob_prior == NULL)
	{
		yolo_noobj_prob_prior = (float*) calloc(nb_box, sizeof(float));
		for(i = 0; i < nb_box; i++)
			yolo_noobj_prob_prior[i] = 0.2f;
	}
	
	l_scale_tab = (float*) calloc(6,sizeof(float));

	l_scale_tab[0] = 2.0f; /*Pos */  l_scale_tab[1] = 2.0f; /*Size */
	l_scale_tab[2] = 1.0f; /*Proba*/ l_scale_tab[3] = 2.0f; /*Objct*/
	l_scale_tab[4] = 1.0f; /*Class*/ l_scale_tab[5] = 1.0f; /*Param*/

	if(scale_tab != NULL)
		for(i = 0; i < 6; i++)
			if(scale_tab[i] > 0.0f)
				l_scale_tab[i] = scale_tab[i];
	
	
	temp = (float*) calloc(6*3, sizeof(float));
	l_slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
	for(i = 0; i < 6; i++)
		l_slopes_and_maxes_tab[i] = &temp[i*3];
	
	sm = l_slopes_and_maxes_tab;
	sm[0][0] = 1.0f; sm[0][1] = 6.0f; sm[0][2] = -6.0f;
	sm[1][0] = 1.0f; sm[1][1] = 1.6f; sm[1][2] = -1.6f;
	sm[2][0] = 1.0f; sm[2][1] = 6.0f; sm[2][2] = -6.0f;
	sm[3][0] = 1.0f; sm[3][1] = 6.0f; sm[3][2] = -6.0f;
	sm[4][0] = 1.0f; sm[4][1] = 6.0f; sm[4][2] = -6.0f;
	sm[5][0] = 1.0f; sm[5][1] = 1.2f; sm[5][2] = -0.2f;
	
	if(slopes_and_maxes_tab != NULL)
	{
		for(i = 0; i < 6; i++)
		{
			if(slopes_and_maxes_tab[i][0] > 0.0f)
				sm[i][0] = slopes_and_maxes_tab[i][0];
			if(slopes_and_maxes_tab[i][1] < 100000.0f)
				sm[i][1] = slopes_and_maxes_tab[i][1];
			if(slopes_and_maxes_tab[i][2] > -100000.0f)
				sm[i][2] = slopes_and_maxes_tab[i][2];
		}
	}
	
	if(param_ind_scale == NULL)
	{
		param_ind_scale = (float*) calloc(nb_param, sizeof(float));
		for(i = 0; i < nb_param; i++)
			param_ind_scale[i] = 1.0f;
	}

	l_IoU_limits = (float*) calloc(8,sizeof(float));
	switch(net->y_param->IoU_type)
	{
		case IOU:
			l_IoU_limits[0] =  0.5f;  l_IoU_limits[1] =  0.1f;
			l_IoU_limits[2] =  0.0f;  l_IoU_limits[3] =  0.0f;
			l_IoU_limits[4] =  0.2f;  l_IoU_limits[5] =  0.2f;
			l_IoU_limits[6] =  0.5f;  l_IoU_limits[7] =  0.3f;
			break;
		
		default:
		case GIOU:
			l_IoU_limits[0] =  0.4f; l_IoU_limits[1] = -0.5f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.3f; l_IoU_limits[5] = -0.3f;
			l_IoU_limits[6] =  0.4f; l_IoU_limits[7] =  0.2f;
			break;
		
		case DIOU:
			l_IoU_limits[0] =  0.3f; l_IoU_limits[1] = -0.6f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.5f; l_IoU_limits[5] = -0.5f;
			l_IoU_limits[6] =  0.3f; l_IoU_limits[7] =  0.1f;
			break;
			
		case DIOU2:
			l_IoU_limits[0] =  0.3f; l_IoU_limits[1] = -0.5f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.4f; l_IoU_limits[5] = -0.4f;
			l_IoU_limits[6] =  0.3f; l_IoU_limits[7] =  0.1f;
			break;
		
	}
	
	if(IoU_limits != NULL)
		for(i = 0; i < 8; i++)
			if(IoU_limits[i] > -1.99f)
				l_IoU_limits[i] = IoU_limits[i];
	
	
	l_fit_parts = (int*) calloc(6,sizeof(int));
	l_fit_parts[0] = 1; /*Position */ 
	l_fit_parts[1] = 1; /*Size */ 
	l_fit_parts[2] = 1; /*Prob */
	l_fit_parts[3] = 1; /*Object*/
	l_fit_parts[4] = 1; /*Class*/
	l_fit_parts[5] = 1; /*Param*/
	
	if(nb_class <= 0)
		l_fit_parts[4] = -1;
	if(nb_param <= 0) /*Param*/
		l_fit_parts[5] = -1;
	
	if(fit_parts != NULL)
		for(i = 0; i < 6; i++)
			if(fit_parts[i] > -2)
				l_fit_parts[i] = fit_parts[i];
	
	net->y_param->nb_box = nb_box;
	net->y_param->prior_size = prior_size;
	net->y_param->noobj_prob_prior = yolo_noobj_prob_prior;
	net->y_param->nb_class = nb_class;
	net->y_param->nb_param = nb_param;
	net->y_param->max_nb_obj_per_image = max_nb_obj_per_image;
	net->y_param->class_softmax = class_softmax;
	net->y_param->label_smoothing = label_smoothing;
	net->y_param->diff_flag = diff_flag;
	
	if(net->y_param->class_softmax == 0)
		sprintf(display_class_type,"sigmoid-MSE");
	else
		sprintf(display_class_type,"softmax-CrossEntropy");
	
	if(net->y_param->diff_flag == 0)
		sprintf(display_difficult,"False");
	else
		sprintf(display_difficult,"True");
	
	//Priors table must be sent to GPU memory if C_CUDA
	net->y_param->scale_tab = l_scale_tab;
	net->y_param->slopes_and_maxes_tab = l_slopes_and_maxes_tab;
	net->y_param->param_ind_scale = param_ind_scale;
	net->y_param->IoU_limits = l_IoU_limits;
	net->y_param->fit_parts = l_fit_parts;
	
	if(strcmp(error_type, "complete") == 0)
	{
		net->y_param->error_type = ERR_COMPLETE;
		sprintf(display_error_type, "COMPLETE");
	}
	else if(strcmp(error_type, "natural") == 0)
	{
		net->y_param->error_type = ERR_NATURAL;
		sprintf(display_error_type, "NATURAL");
	}
	else
	{
		printf(" WARNING: Unrecognized YOLO display error type %s, fallback to default \"natural\"\n", error_type);
		net->y_param->error_type = ERR_NATURAL;
		sprintf(display_error_type, "NATURAL");
	}
	
	printf("\n YOLO layer setup \n");
	printf(" -------------------------------------------------------------------\n");
	printf(" Nboxes = %d\n Nclasses = %d\n Nparams = %d\n IoU type = %s\n",
			net->y_param->nb_box, net->y_param->nb_class, net->y_param->nb_param, display_IoU_type_char);
	printf(" Classification type: %s\n", display_class_type);
	printf(" Class label smooting: %f\n", net->y_param->label_smoothing);
	printf(" Nb dim fitted : %d\n\n", net->y_param->fit_dim);
	printf(" W priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+0]);
	printf("]\n H priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+1]);
	printf("]\n D priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+2]);
	printf("]\n");
	printf(" No obj. prob. priors\n          = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->noobj_prob_prior[i]);
	printf("]\n");
	printf(" Fit parts: (Pos., Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf(" %d ",net->y_param->fit_parts[i]);
	printf("]\n");
	printf(" Error scales: (Pos., Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf(" %5.3f ",net->y_param->scale_tab[i]);
	printf("]\n");
	printf(" IoU lim.: (GdNotBest, LowBest, Prob., Obj., Class., Param., diffIoUlim, diffObjlim)\n   = [");
	for(i = 0; i < 8; i++)
		printf("%7.3f ", net->y_param->IoU_limits[i]);
	printf("]\n");
	
	if(net->y_param->nb_param > 0)
	{
		printf(" Individual param. error scaling: \n   = [");
		for(i = 0; i < net->y_param->nb_param; i++)
			printf("%7.3f ", net->y_param->param_ind_scale[i]);
		printf("]\n");
	}
	
	printf("\n Activation slopes and limits: \n   = ");
	for(i = 0; i < 6; i++)
		printf("[%6.2f %6.2f %6.2f]\n     ", 
			net->y_param->slopes_and_maxes_tab[i][0],
			net->y_param->slopes_and_maxes_tab[i][1],
			net->y_param->slopes_and_maxes_tab[i][2]);
	
	printf("\n *** Other training hyper-parameters *** \n");
	if(net->y_param->strict_box_size_association > 0)
	{
		printf("  Strict box size association is ENABLED\n");
		printf("  Strict association Nb. good priors = %d\n", 
			net->y_param->strict_box_size_association);
	}
	else
	{	
		printf("  Strict box size association is DISABLED\n");
	}
	printf("  Regret-based association : %d\n", net->y_param->regret_based);
	printf("  Startup random association Nb. item : %d\n", net->y_param->rand_startup);
	printf("  Proportion of forced best prior assoc.: %5.3f\n", net->y_param->rand_prob_best_box_assoc);
	printf("  Proportion of forced random prior assoc.: %5.3f\n", net->y_param->rand_prob);
	printf("  Forced smallest prior association scaling : %6.3f\n", net->y_param->min_prior_forced_scaling);
	printf("  Closest prior association type : %s\n", display_prior_dist_type);
	printf("  Difficult flag in use: %s\n", display_difficult);
	printf("  Display error type : %s\n", display_error_type);
	printf("\n -------------------------------------------------------------------\n\n");
	
	return (net->y_param->nb_box * (8 + net->y_param->nb_class + net->y_param->nb_param));
}

void free_yolo_params(network *net)
{
	yolo_param *y_param = (yolo_param*) net->y_param;
	
	free(y_param->prior_size);
	free(y_param->noobj_prob_prior);
	free(y_param->scale_tab);
	free(y_param->slopes_and_maxes_tab[0]);
	free(y_param->slopes_and_maxes_tab);
	free(y_param->param_ind_scale);
	free(y_param->IoU_limits);
	free(y_param->fit_parts);
	
	free(y_param);
	net->y_param = NULL;
}


void YOLO_activation_fct(void *i_tab, size_t flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax)
{	
	float *tab = (float*) i_tab;
	
	size_t nb_class = y_param.nb_class, nb_param = y_param.nb_param;
	/*Default values are in activ_function.c (set_yolo_config)*/
	float **sm_tab = y_param.slopes_and_maxes_tab;
	int fit_dim = y_param.fit_dim;	
	size_t i, col, in_col;
	size_t output_offset = 8+nb_class+nb_param;
	
	#pragma omp parallel for private(col, in_col) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		float normal = 0.0f, vmax;
		size_t j;
		col = i / flat_offset;
		in_col = col%output_offset;
		
		/*Position*/
		if(in_col >= 0 && in_col < 3)	
		{
			if(fit_dim > in_col)
			{	
				tab[i] = -sm_tab[0][0]*tab[i];
				if(tab[i] > sm_tab[0][1])
					tab[i] = sm_tab[0][1];	
				else if(tab[i] < sm_tab[0][2])
					tab[i] = sm_tab[0][2];	
				tab[i] = 1.0f/(1.0f + expf(tab[i]));
			}	
			else
				tab[i] = 0.5f; /*Center of the cell*/
			continue;
		}
	
		/*Box size*/
		if(in_col >= 3 && in_col < 6)	
		{
			if(fit_dim > in_col-3)
			{	
				tab[i] = sm_tab[1][0]*tab[i];	
				if(tab[i] > sm_tab[1][1])
					tab[i] = sm_tab[1][1];	
				else if(tab[i] < (sm_tab[1][2]))
					tab[i] = (sm_tab[1][2]);
			}	
			else
				tab[i] = 0.0f; /*Output = prior*/	
			continue;
		}
	
		/*Object probability*/
		if(in_col == 6)
		{
			tab[i] = -sm_tab[2][0]*tab[i];	
			if(tab[i] > sm_tab[2][1])
				tab[i] = sm_tab[2][1];
			else if(tab[i] < sm_tab[2][2])	
				tab[i] = sm_tab[2][2];
			tab[i] = 1.0f/(1.0f + expf(tab[i]));	
			continue;
		}
	
		/*Objectness (Obj. quality => based on IoU)*/
		if(in_col == 7)
		{
			tab[i] = -sm_tab[3][0]*tab[i];	
			if(tab[i] > sm_tab[3][1])
				tab[i] = sm_tab[3][1];
			else if(tab[i] < sm_tab[3][2])	
				tab[i] = sm_tab[3][2];
			tab[i] = 1.0f/(1.0f + expf(tab[i]));	
			continue;
		}
	
		/*Classes*/
		if(in_col >= 8 && in_col < 8+nb_class)
		{
			if(class_softmax)
			{
				if(in_col != 8)
					continue;
				vmax = tab[i];
				for(j = 1; j < nb_class; j++)
					if(tab[i+j*flat_offset] > vmax)
						vmax = tab[i+j*flat_offset];
				
				for(j = 0; j < nb_class; j++)
				{
					tab[i+j*flat_offset] = expf((tab[i+j*flat_offset]-vmax));
					normal += (float)tab[i+j*flat_offset];
				}
				
				for(j = 0; j < nb_class; j++)
					tab[i+j*flat_offset] = ((float)tab[i+j*flat_offset]/normal);
			}
			else
			{
				tab[i] = -sm_tab[4][0]*tab[i];	
				if(tab[i] > sm_tab[4][1])
					tab[i] = sm_tab[4][1];
				else if(tab[i] < sm_tab[4][2])	
					tab[i] = sm_tab[4][2];
				tab[i] = 1.0f/(1.0f + expf(tab[i]));
			}
			continue;
		}
	
		/*Additional parameters (regression)*/
		if(in_col >= 8+nb_class)
		{
			tab[i] = sm_tab[5][0]*tab[i];
			if(tab[i] > sm_tab[5][1])
				tab[i] = sm_tab[5][1];
			else if(tab[i] < (sm_tab[5][2]))	
				tab[i] = (sm_tab[5][2]);
			continue;
		}	
	}
}


void YOLO_deriv_error_fct
	(void *i_delta_o, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d, yolo_param y_param, size_t size, size_t nb_im_iter)
{
	float *t_delta_o = (float*)i_delta_o;
	float *t_output  = (float*)i_output;
	float *t_target  = (float*)i_target;

	/* Define many "shorts" for y_param content to enhance code redeability*/
	int nb_box                      = y_param.nb_box;
	int nb_class                    = y_param.nb_class;
	int nb_param                    = y_param.nb_param;
	int regret_based                = y_param.regret_based;
	int fit_dim                     = y_param.fit_dim;
	size_t rand_startup             = y_param.rand_startup;
	float rand_prob_best_box_assoc  = y_param.rand_prob_best_box_assoc;
	float rand_prob                 = y_param.rand_prob;
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;
	int class_softmax               = y_param.class_softmax;
	int diff_flag                   = y_param.diff_flag;
	int prior_dist_type             = y_param.prior_dist_type;

	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];

	float *prior_size                 = y_param.prior_size;
	int   *cell_size                  = y_param.cell_size;
	float *param_ind_scale            = y_param.param_ind_scale;
	float *lambda_noobj_prior         = y_param.noobj_prob_prior;
	float **sm_tab                    = y_param.slopes_and_maxes_tab;
	int   *t_in_cell_target_id        = y_param.in_cell_target_id;
	float *t_IoU_table                = y_param.IoU_table;
	float *t_dist_prior               = y_param.dist_prior;
	int   *t_prior_assigned_target    = y_param.prior_assigned_target;
	int   *t_box_ids                  = y_param.box_ids;
	int   *t_box_locked               = y_param.box_locked;
	float *t_box_in_pix               = y_param.box_in_pix;

	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2];
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];
	float positive_class_target, negative_class_target;

	size_t c_nb_area[3];
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;
	size_t c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];

	if(nb_class > 0)
	{
		positive_class_target = y_param.label_smoothing/nb_class + 1.0f - y_param.label_smoothing;
		negative_class_target = y_param.label_smoothing/nb_class;
	}

	#pragma omp parallel
	{
	#ifdef _OPENMP
	srand((int)time(NULL) ^ omp_get_thread_num());
	#endif
	#pragma omp for schedule(guided,4)
	for(size_t c_pix = 0; c_pix < size; c_pix++)
	{
		float *delta_o = t_delta_o, *output = t_output, *target = t_target;
		int *in_cell_target_id = t_in_cell_target_id, *prior_assigned_target = t_prior_assigned_target;
		int *box_ids = t_box_ids, *box_locked = t_box_locked;
		float *IoU_table = t_IoU_table, *dist_prior = t_dist_prior, *box_in_pix = t_box_in_pix;
		int strict_box_size_association = y_param.strict_box_size_association;
		int j, k, l, l_o, l_t;
		size_t i = c_pix, f_offset, total_area_and_cell_offset;
		int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0, rand_pos, temp_id;
		int nb_in_cell, id_in_cell_offset, targ_diff_flag = 0;
		int resp_box, resp_targ, candidate_box, assigned_count;
		float best_dist, c_dist, max_IoU, current_IoU, l_scale, l_targ;
		float best_target_IoU, second_target_IoU, effective_second_IoU, current_regret, best_regret;
		size_t cell_pos[3], obj_c[3];
		float *c_box_in_pix, *c_prior_size;
		float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];
		float class_only_IoU = -2.0f;
		
		c_batch = i / flat_output_size;
		target += flat_target_size * c_batch;
		f_offset = size;
		output_offset = 8+nb_class+nb_param;
		target_offset = 7+nb_param+diff_flag;
	
		i = i % flat_output_size;
		cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);
		cell_pos[1] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];
		cell_pos[0] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];
	
		total_area_and_cell_offset = c_total_nb_area * c_batch
			+ cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];
	
		delta_o += total_area_and_cell_offset;
		output  += total_area_and_cell_offset;
	
		in_cell_target_id		+= total_area_and_cell_offset * y_param.max_nb_obj_per_image;
		IoU_table  				+= total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		dist_prior 				+= total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		prior_assigned_target 	+= total_area_and_cell_offset * nb_box;
		box_ids 				+= total_area_and_cell_offset * nb_box;
		box_locked 				+= total_area_and_cell_offset * nb_box;
		box_in_pix 				+= total_area_and_cell_offset * 6 * nb_box;
	
		nb_obj_target = target[0];
		target++;
		
		if(nb_obj_target == -1)	/*Used for class only detection pre-training (likely obsolete)*/
		{
			nb_obj_target = 1;
			class_only_IoU = good_IoU_lim;
		}
	
		best_dist = 1000000;	/*Search for smallest prior id*/
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			box_ids[k] = k;
			c_box_in_pix = box_in_pix + k*6;
			c_prior_size = prior_size + k*3;
			l_o = k*output_offset;
			for(l = 0; l < 3; l++)
				c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];
			for(l = 0; l < 3; l++)
				c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);
	
			c_dist = c_prior_size[0]*c_prior_size[0]
				   + c_prior_size[1]*c_prior_size[1]
				   + c_prior_size[2]*c_prior_size[2];
			if(c_dist < best_dist)
			{
				best_dist = c_dist;
				s_p_i = k;
			}
		}
	
		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*target_offset;
			/* Search for targets that should be predicted by the current cell element */
			for(l = 0; l < 3; l++)
				obj_c[l] = (size_t)( ((float)target[l_t+l+4] + (float)target[l_t+l+1])*0.5f / cell_size[l]);
	
			if(obj_c[0] == cell_pos[0] && obj_c[1] == cell_pos[1] && obj_c[2] == cell_pos[2])
			{
				in_cell_target_id[nb_in_cell] = j;
				nb_in_cell++;
			}
		}
	
		/* Flag all the "Good but not best boxes" for all detection units regardless of the target grid element */
		for(k = 0; k < nb_box; k++)
		{
			c_box_in_pix = box_in_pix+k*6;
			for(l = 0; l < 6; l++)
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
	
			max_IoU = -2.0f; resp_targ = -1;
			for(j = 0; j < nb_obj_target; j++)
			{
				l_t = j*target_offset;
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
	
				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_targ = j;
				}
			}
			if(resp_targ >= 0 && max_IoU > good_IoU_lim)
			{
				box_locked[k] = 1;
				prior_assigned_target[k] = resp_targ;	/* Can be used to update class for GBNB associations if no other association */
			}
		}
	
		/*Random association. Used during startup or at a given rand_prob rate*/
		if(nb_im_iter <= rand_startup || random_uniform() < rand_prob)
		{
			/* Random association is done by scrambling in_cell_target_id and box_ids, then associating in order*/
			for(k = 0; k+1 < nb_in_cell; k++)
			{
				rand_pos = k + (int)(random_uniform()*(nb_in_cell-k));
				temp_id = in_cell_target_id[k];
				in_cell_target_id[k] = in_cell_target_id[rand_pos];
				in_cell_target_id[rand_pos] = temp_id;
			}
	
			for(k = 0; k+1 < nb_box; k++)
			{
				rand_pos = k + (int)(random_uniform()*(nb_box-k));
				temp_id = box_ids[k];
				box_ids[k] = box_ids[rand_pos];
				box_ids[rand_pos] = temp_id;
			}
	
			assigned_count = 0;
			for(k = 0; k < nb_in_cell && assigned_count < nb_box; k++)
			{	/* Difficult target are included during random startup but ignored for occasional random regularization */
				if(diff_flag && nb_im_iter > rand_startup && (int)((float)target[in_cell_target_id[k]*target_offset+7+nb_param]) > 0)
					continue;
				prior_assigned_target[box_ids[assigned_count]] = in_cell_target_id[k];
				box_locked[box_ids[assigned_count]] = 2;
				assigned_count++;
			}
		}
		else
		{
			/* For all targets in cell compute the IoU against all predictions and distances to priors */
			for(j = 0; j < nb_in_cell; j++)
			{
				id_in_cell_offset = j*nb_box;
				l_t = in_cell_target_id[j]*target_offset;
	
				for(k = 0; k < nb_box; k++)
				{
					c_box_in_pix = box_in_pix+k*6;
					for(l = 0; l < 6; l++)
						out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
					for(l = 0; l < 6; l++)
						targ_int[l] = target[l_t+1+l];
					for(l = 0; l < 3; l++)
						targ_size[l] = targ_int[l+3] - targ_int[l];
	
					current_IoU = y_param.c_IoU_fct(out_int, targ_int);
					IoU_table[id_in_cell_offset + k] = current_IoU;
	
					c_prior_size = prior_size + k*3;
					switch(prior_dist_type)
					{
						case DIST_IOU:	/* Compute a zero centered IoU */
							for(l = 0; l < 6; l++)
								out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];
							for(l = 0; l < 6; l++)
								targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];
							dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);
							break;
	
						default:
						case DIST_SIZE:
							dist_prior[id_in_cell_offset + k] =
								 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
								+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
								+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]);
							break;
	
						case DIST_OFFSET:
							for(l = 0; l < 3; l++)
							{
								obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
								if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);
								else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);
								else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
							}
	
							dist_prior[id_in_cell_offset + k] =	obj_in_offset[3]*obj_in_offset[3]
								                              + obj_in_offset[4]*obj_in_offset[4]
								                              + obj_in_offset[5]*obj_in_offset[5];
							break;
					}
				}
			}
	
			/* Restrict association to the best theoretical prior at a given rand_prob_best_box_assoc rate*/
			if(random_uniform() < rand_prob_best_box_assoc)
				strict_box_size_association = 1;
	
			for(j = 0; j < nb_in_cell; j++)
			{
				id_in_cell_offset = j*nb_box;
				l_t = in_cell_target_id[j]*target_offset;
	
				/* Restrict association to the smallest prior (or identical) if the target is too small */
				/* Smallest might not be part of the N closest prior, so it need to be handled beforehand */
				/* Smallest association is also considered to replace "single best prior association" for small targets*/
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
	
				if(    targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]
					&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]
					&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])
				{
					for(k = 0; k < nb_box; k++)
						if( fabsf(prior_size[s_p_i*3+0] - prior_size[k*3+0]) > 0.0001f ||
							fabsf(prior_size[s_p_i*3+1] - prior_size[k*3+1]) > 0.0001f ||
							fabsf(prior_size[s_p_i*3+2] - prior_size[k*3+2]) > 0.0001f)
							dist_prior[id_in_cell_offset + k] = -2.0f;
				}
				else if(strict_box_size_association > 0)
				{
					/* Restrict association to the N best different theoretical priors (identical priors are considered as a single one)*/
					c_dist = -1.0f;
					for(l = 0; l < strict_box_size_association; l++)
					{
						best_dist = 1000000.0f;	resp_box = -1;
						for(k = 0; k < nb_box; k++)
						{
							if(dist_prior[id_in_cell_offset + k] > -1.9f && dist_prior[id_in_cell_offset + k] < best_dist
									&& dist_prior[id_in_cell_offset + k] > c_dist + 0.0001f)
							{
								best_dist = dist_prior[id_in_cell_offset + k];
								resp_box = k;
							}
						}
						if(resp_box == -1)
							break;
						c_dist = best_dist;
					}
	
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset + k] > c_dist + 0.0001f)
							dist_prior[id_in_cell_offset + k] = -2.0f;
				}
			}
	
			/* Prediction aware association */
			while(1)
			{
				max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;
				if(!regret_based) /* Legacy best match first association */
				{
					for(j = 0; j < nb_in_cell; j++)
					{
						id_in_cell_offset = j*nb_box;
						for(k = 0; k < nb_box; k++)
							if(dist_prior[id_in_cell_offset + k] > -1.9f && IoU_table[id_in_cell_offset + k] > max_IoU)
							{
								max_IoU = IoU_table[id_in_cell_offset + k];
								resp_targ = j; resp_box = k;
							}
					}
				}
				else /* Regret based approach -> try to find a more global good association looking at the effect on second best match */
				{	 /* Search the pair for which missing the association would be the most problematic (second best association would be bad) */
					best_regret = -1.0f;
					for(j = 0; j < nb_in_cell; j++)
					{
						id_in_cell_offset = j*nb_box;
						best_target_IoU = -2.0f; second_target_IoU = -2.0f; candidate_box = -1;
						for(k = 0; k < nb_box; k++)
						{
							if(dist_prior[id_in_cell_offset + k] <= -1.9f)
								continue;
							current_IoU = IoU_table[id_in_cell_offset + k];
	
							if(current_IoU > best_target_IoU)
							{
								second_target_IoU = best_target_IoU;
								best_target_IoU   = current_IoU;
								candidate_box     = k;
							}
							else if(current_IoU > second_target_IoU)
								second_target_IoU = current_IoU;
						}
	
						/* If all available predictions are bad, ignore the target. It will be handled by best prior instead. */
						if(candidate_box == -1 || best_target_IoU < low_IoU_best_box_assoc)
							continue;
	
						effective_second_IoU = second_target_IoU;
						if(effective_second_IoU < low_IoU_best_box_assoc)
							effective_second_IoU = low_IoU_best_box_assoc;
	
						current_regret = best_target_IoU - effective_second_IoU;
						/* If identical regret (might append due to ceiling effects) prefer the best match */
						if(current_regret > best_regret + 0.0001f
							|| (fabsf(current_regret - best_regret) <= 0.0001f && best_target_IoU > max_IoU))
						{
							best_regret = current_regret;
							max_IoU     = best_target_IoU;
							resp_targ   = j;
							resp_box    = candidate_box;
						}
					}
				}
	
				if(resp_box == -1 || max_IoU < low_IoU_best_box_assoc)
				{
					best_dist = 1000000.0f;
					resp_box  = -1;	resp_targ = -1;
					for(j = 0; j < nb_in_cell; j++)
					{
						id_in_cell_offset = j*nb_box;
						for(k = 0; k < nb_box; k++)
							if(dist_prior[id_in_cell_offset + k] > -1.9f && dist_prior[id_in_cell_offset + k] < best_dist)
							{
								best_dist = dist_prior[id_in_cell_offset + k];
								resp_box = k; resp_targ = j;
							}
					}
					if(resp_box == -1)
						break;
	
					/* If multiple identical best priors are avalaible, */
					/* they are seperated based on the IoU between the corresponding prediction and the target */
					max_IoU = IoU_table[resp_targ*nb_box + resp_box];
					id_in_cell_offset = resp_targ*nb_box;
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset + k] > -1.9f && fabsf(dist_prior[id_in_cell_offset + k] - best_dist) < 0.0001f
							&& IoU_table[id_in_cell_offset + k] > max_IoU)
							{
								max_IoU = IoU_table[id_in_cell_offset + k];
								resp_box = k;
							}
				}
				if(resp_box == -1)
					break;
				l_t = in_cell_target_id[resp_targ]*target_offset;
	
				/* Difficult target handling -> a difficult object must already be confidently detected to be set as an actual target */
				/* If not, it is removed from the array but not associated. Note that difficult objects still count during GBNB check, */
				/* so a good but uncertain detection most often lead to no gradient. */
				if(diff_flag && (int)((float)target[l_t+7+nb_param]) > 0
					&& (max_IoU < diff_IoU_lim || (float)output[(resp_box*output_offset+7)*f_offset] < diff_obj_lim))
				{
					dist_prior[resp_targ*nb_box + resp_box] = -2.0f;
					continue;
				}
				/* The found association is removed from the dist_prior table */
				for(j = 0; j < nb_in_cell; j++)
					dist_prior[j*nb_box + resp_box] = -2.0f;
				for(k = 0; k < nb_box; k++)
					dist_prior[resp_targ*nb_box + k] = -2.0f;
	
				prior_assigned_target[resp_box] = in_cell_target_id[resp_targ];
				box_locked[resp_box] = 2;
			}
		}
	
		/*Use previous assigment to update all detection units*/
		for(j = 0; j < nb_box; j++)
		{
			l_o = j*output_offset;
			targ_diff_flag = 0;
	
			/*Set the all box vector to 0, and only modify if necessary. Vastly simplify the logic*/
			for(k = 0; k < 8+nb_class+nb_param; k++)
				delta_o[(l_o+k)*f_offset] = (float) 0.0f;
	
			if(box_locked[j] == 2)
			{
				l_t = prior_assigned_target[j]*target_offset;
				if(diff_flag)
					targ_diff_flag = (int)((float)target[l_t+7+nb_param]);
	
				c_box_in_pix = box_in_pix + j*6;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
	
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
	
				max_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(max_IoU > 0.98f)
					max_IoU = 0.98f;
	
				if(class_only_IoU > -1.9f)
					max_IoU = class_only_IoU;
	
				for(l = 0; l < 3; l++)
					obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];
	
				c_prior_size = prior_size + 3*j;
				for(l = 0; l < 3; l++)
				{
					obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
					if(obj_in_offset[l+3] < size_min_sat)		obj_in_offset[l+3] = logf(size_min_sat);
					else if(obj_in_offset[l+3] > size_max_sat)	obj_in_offset[l+3] = logf(size_max_sat);
					else 										obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
				}
			}
	
			/***** POSITION LOSS *****/
			if(fit_pos != -1 && box_locked[j] == 2 && class_only_IoU < -1.9 && (diff_flag == 0 || targ_diff_flag < 3))
			{
				for(k = 0; k < fit_dim; k++) /*delta_o = 0 above fit_dim */
				{
					l_scale = sm_tab[0][0]*coord_scale
						* (float)output[(l_o+k)*f_offset]*(1.0f-(float)output[(l_o+k)*f_offset]);
					if(fit_pos == 1) l_targ = obj_in_offset[k];
					else 			 l_targ = 0.5f;
					delta_o[(l_o+k)*f_offset] = (float)(l_scale*((float)output[(l_o+k)*f_offset]-l_targ));
				}
			}
	
			/*****  SIZE LOSS  *****/
			if(fit_size != -1 && box_locked[j] == 2 && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))
			{
				l_scale = sm_tab[1][0]*size_scale;
				for(k = 0; k < fit_dim; k++) /*delta_o = 0 above fit_dim */
				{
					if(fit_size == 1) l_targ = obj_in_offset[k+3];
					else 			  l_targ = 0.0f;
					delta_o[(l_o+k+3)*f_offset] = (float) (l_scale*((float)output[(l_o+k+3)*f_offset]-l_targ));
				}
			}
	
			/*****  PROB LOSS  *****/
			if(fit_prob != -1 && ((box_locked[j] == 2 && max_IoU > min_prob_IoU_lim) || box_locked[j] == 0))
			{
				l_scale = sm_tab[2][0]*prob_scale
					* (float)output[(l_o+6)*f_offset]*(1.0f-(float)output[(l_o+6)*f_offset]);
				if(box_locked[j] == 0)
					l_scale *= lambda_noobj_prior[j];
				if(fit_prob == 1)
				{
					if(box_locked[j] == 2) 		l_targ = 0.98f;
					else if(box_locked[j] == 0) l_targ = 0.02f;
				}
				else							l_targ = 0.5f;
				delta_o[(l_o+6)*f_offset] = (float) (l_scale*((float)output[(l_o+6)*f_offset]-l_targ));
			}
	
			/*****  OBJ LOSS   *****/
			if(fit_obj != -1 && ((box_locked[j] == 2 && max_IoU > min_obj_IoU_lim) || box_locked[j] == 0))
			{
				l_scale = sm_tab[3][0]*obj_scale
					* (float)output[(l_o+7)*f_offset]*(1.0f-(float)output[(l_o+7)*f_offset]);
				if(box_locked[j] == 0)
					l_scale *= lambda_noobj_prior[j];
				if(fit_obj == 1)
				{
					if(box_locked[j] == 2) 		l_targ = (1.0+max_IoU)*0.5;
					else if(box_locked[j] == 0) l_targ = 0.02f;
				}
				else							l_targ = 0.5f;
				delta_o[(l_o+7)*f_offset] = (float) (l_scale*((float)output[(l_o+7)*f_offset]-l_targ));
			}
	
			/*****  CLASS LOSS *****/
			if(fit_class != -1 && box_locked[j] == 2 && max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
			{
				for(k = 0; k < nb_class; k++)
				{
					l_scale = class_scale;
					if(!class_softmax)
						l_scale *= sm_tab[4][0]*(float)output[(l_o+8+k)*f_offset]*(1.0f-(float)output[(l_o+8+k)*f_offset]);
					if(fit_class == 1)
					{
						if(k == (int) target[l_t]-1)	l_targ = positive_class_target;
						else 							l_targ = negative_class_target;
					}
					else								l_targ = 0.5f;
					delta_o[(l_o+8+k)*f_offset] = (float) (l_scale*((float)output[(l_o+8+k)*f_offset]-l_targ));
				}
			}
	
			/*****  PARAM LOSS  *****/
			if(fit_param != -1 && box_locked[j] == 2 && max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
			{
				for(k = 0; k < nb_param; k++)
				{
					l_scale = sm_tab[5][0]*param_scale*param_ind_scale[k];
					if(fit_param ==1)	l_targ = (float)target[l_t+7+k];
					else				l_targ = 0.5f;
					delta_o[(l_o+8+nb_class+k)*f_offset] = (float) (l_scale*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ));
				}
			}
		}
	}
	#ifdef _OPENMP
	}
	#endif
}


void YOLO_error_fct
	(float *i_output_error, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d, yolo_param y_param, size_t size)
{
	float *t_output = (float*)i_output;
	float *t_target = (float*)i_target;

	int nb_box                      = y_param.nb_box;
	int nb_class                    = y_param.nb_class;
	int nb_param                    = y_param.nb_param;
	int regret_based                = y_param.regret_based;
	int strict_box_size_association = y_param.strict_box_size_association;
	int fit_dim                     = y_param.fit_dim;
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;
	int class_softmax               = y_param.class_softmax;
	int diff_flag                   = y_param.diff_flag;
	int prior_dist_type             = y_param.prior_dist_type;
	int error_type                  = y_param.error_type;

	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];

	float *prior_size                 = y_param.prior_size;
	int   *cell_size                  = y_param.cell_size;
	float *param_ind_scale            = y_param.param_ind_scale;
	float *lambda_noobj_prior         = y_param.noobj_prob_prior;
	float **sm_tab                    = y_param.slopes_and_maxes_tab;
	float *t_IoU_monitor              = y_param.IoU_monitor;
	int   *t_in_cell_target_id        = y_param.in_cell_target_id;
	float *t_IoU_table                = y_param.IoU_table;
	float *t_dist_prior               = y_param.dist_prior;
	int   *t_prior_assigned_target    = y_param.prior_assigned_target;
	int   *t_box_locked               = y_param.box_locked;
	float *t_box_in_pix               = y_param.box_in_pix;

	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2];
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];
	float positive_class_target, negative_class_target;

	size_t c_nb_area[3];
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;
	size_t c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];

	if(nb_class > 0)
	{
		positive_class_target = y_param.label_smoothing/nb_class + 1.0f - y_param.label_smoothing;
		negative_class_target = y_param.label_smoothing/nb_class;
	}

	#pragma omp parallel
	#ifdef _OPENMP
	{
	srand((int)time(NULL) ^ omp_get_thread_num());
	#endif
	#pragma omp for schedule(guided,4)
	for(size_t c_pix = 0; c_pix < size; c_pix++)
	{
		float *output, *target, *output_error;
		float *IoU_table, *dist_prior, *box_in_pix, *IoU_monitor;
		int *in_cell_target_id, *prior_assigned_target, *box_locked;
		int j, k, l, l_o, l_t;
		size_t i, f_offset, total_area_and_cell_offset;
		int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;
		int nb_in_cell, id_in_cell_offset, targ_diff_flag = 0;
		int resp_box, resp_targ, candidate_box;
		float best_dist, c_dist, max_IoU, current_IoU, l_scale, l_targ;
		float best_target_IoU, second_target_IoU, effective_second_IoU, current_regret, best_regret;
		size_t cell_pos[3], obj_c[3];
		float *c_box_in_pix, *c_prior_size;
		float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];
		float class_only_IoU = -2.0f;

		c_batch = c_pix / flat_output_size;
		target = t_target + flat_target_size*c_batch;
		f_offset = size;
		output_offset = 8 + nb_class + nb_param;
		target_offset = 7 + nb_param + diff_flag;

		i = c_pix % flat_output_size;
		cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);
		cell_pos[1] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];
		cell_pos[0] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];

		total_area_and_cell_offset = c_total_nb_area*c_batch
		    + cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];

		output_error = i_output_error + total_area_and_cell_offset;
		output = t_output + total_area_and_cell_offset;

		IoU_monitor           = t_IoU_monitor + 2*nb_box*total_area_and_cell_offset;
		in_cell_target_id     = t_in_cell_target_id + total_area_and_cell_offset*y_param.max_nb_obj_per_image;
		IoU_table             = t_IoU_table + total_area_and_cell_offset*y_param.max_nb_obj_per_image*nb_box;
		dist_prior            = t_dist_prior + total_area_and_cell_offset*y_param.max_nb_obj_per_image*nb_box;
		prior_assigned_target = t_prior_assigned_target + total_area_and_cell_offset*nb_box;
		box_locked            = t_box_locked + total_area_and_cell_offset*nb_box;
		box_in_pix            = t_box_in_pix + total_area_and_cell_offset*6*nb_box;

		nb_obj_target = target[0];
		target++;
		
		if(nb_obj_target == -1)
		{
			nb_obj_target = 1;
			class_only_IoU = good_IoU_lim;
		}
	
		best_dist = 1000000.0f;
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			IoU_monitor[2*k] = -1.0f; IoU_monitor[2*k+1] = -1.0f;
			c_box_in_pix = box_in_pix + 6*k;
			c_prior_size = prior_size + 3*k;
			l_o = k*output_offset;
			for(l = 0; l < 3; l++)
				c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l])*cell_size[l];
			for(l = 0; l < 3; l++)
				c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);
	
			c_dist = c_prior_size[0]*c_prior_size[0]
			       + c_prior_size[1]*c_prior_size[1]
			       + c_prior_size[2]*c_prior_size[2];
			if(c_dist < best_dist)
			{
				best_dist = c_dist;
				s_p_i = k;
			}
		}
	
		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*target_offset;
			for(l = 0; l < 3; l++)
				obj_c[l] = (size_t)(((float)target[l_t+l+4] + (float)target[l_t+l+1])*0.5f/cell_size[l]);
			if(obj_c[0] == cell_pos[0] && obj_c[1] == cell_pos[1] && obj_c[2] == cell_pos[2])
			{
				in_cell_target_id[nb_in_cell] = j;
				nb_in_cell++;
			}
		}
	
		for(k = 0; k < nb_box; k++)
		{
			c_box_in_pix = box_in_pix + 6*k;
			for(l = 0; l < 6; l++)
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];
	
			max_IoU = -2.0f; resp_targ = -1;
			for(j = 0; j < nb_obj_target; j++)
			{
				l_t = j*target_offset;
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
	
				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_targ = j;
				}
			}
			if(resp_targ >= 0 && max_IoU > good_IoU_lim)
			{
				box_locked[k] = 1;
				prior_assigned_target[k] = resp_targ;
			}
		}
	
		/* No random association in error computation (only used for deriv) */
	
		for(j = 0; j < nb_in_cell; j++)
		{
			id_in_cell_offset = j*nb_box;
			l_t = in_cell_target_id[j]*target_offset;
	
			for(k = 0; k < nb_box; k++)
			{
				c_box_in_pix = box_in_pix + 6*k;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
	
				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				IoU_table[id_in_cell_offset+k] = current_IoU;
	
				if(error_type == ERR_NATURAL)
				{
					dist_prior[id_in_cell_offset+k] = 0.0f;
					continue;
				}
				c_prior_size = prior_size + 3*k;
				switch(prior_dist_type)
				{
					case DIST_IOU:
						for(l = 0; l < 6; l++)
							out_int[l] = copysignf(0.5f, l-2.5f)*c_prior_size[l%3];
						for(l = 0; l < 6; l++)
							targ_int[l] = copysignf(0.5f, l-2.5f)*targ_size[l%3];
						dist_prior[id_in_cell_offset+k] =  1.0f - y_param.c_IoU_fct(out_int, targ_int);
						break;
	
					default:
					case DIST_SIZE:
						dist_prior[id_in_cell_offset+k] =
					          (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
					        + (targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
					        + (targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]);
						break;
	
					case DIST_OFFSET:
						for(l = 0; l < 3; l++)
						{
							obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
							if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);
							else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);
							else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
						}
						dist_prior[id_in_cell_offset+k] = obj_in_offset[3]*obj_in_offset[3]
					                                    + obj_in_offset[4]*obj_in_offset[4]
						                                + obj_in_offset[5]*obj_in_offset[5];
						break;
				}
			}
		}
	
		if(error_type == ERR_COMPLETE)
		{
			for(j = 0; j < nb_in_cell; j++)
			{
				id_in_cell_offset = j*nb_box;
				l_t = in_cell_target_id[j]*target_offset;
	
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
	
				if(    targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]
				    && targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]
				    && targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])
				{
					for(k = 0; k < nb_box; k++)
						if( fabsf(prior_size[s_p_i*3+0] - prior_size[k*3+0]) > 0.0001f ||
						    fabsf(prior_size[s_p_i*3+1] - prior_size[k*3+1]) > 0.0001f ||
						    fabsf(prior_size[s_p_i*3+2] - prior_size[k*3+2]) > 0.0001f)
							dist_prior[id_in_cell_offset+k] = -2.0f;
				}
				else if(strict_box_size_association > 0)
				{
					c_dist = -1.0f;
					for(l = 0; l < strict_box_size_association; l++)
					{
						best_dist = 1000000.0f;
						resp_box = -1;
						for(k = 0; k < nb_box; k++)
						{
							if(dist_prior[id_in_cell_offset+k] > -1.9f
							    && dist_prior[id_in_cell_offset+k] < best_dist
							    && dist_prior[id_in_cell_offset+k] > c_dist + 0.0001f)
							{
								best_dist = dist_prior[id_in_cell_offset+k];
								resp_box = k;
							}
						}
						if(resp_box == -1)
							break;
						c_dist = best_dist;
					}
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset+k] > c_dist + 0.0001f)
							dist_prior[id_in_cell_offset+k] = -2.0f;
				}
			}
		}
		
		while(1)
		{
			max_IoU = -2.0f; resp_box = -1; resp_targ = -1;
			if(error_type == ERR_NATURAL || !regret_based)
			{
				for(j = 0; j < nb_in_cell; j++)
				{
					id_in_cell_offset = j*nb_box;
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset+k] > -1.9f
						    && IoU_table[id_in_cell_offset+k] > max_IoU)
						{
							max_IoU = IoU_table[id_in_cell_offset+k];
							resp_targ = j;
							resp_box = k;
					}
				}
			}
			else
			{
				best_regret = -1.0f;
				for(j = 0; j < nb_in_cell; j++)
				{
					id_in_cell_offset = j*nb_box;
					best_target_IoU = -2.0f; second_target_IoU = -2.0f; candidate_box = -1;
					for(k = 0; k < nb_box; k++)
					{
						if(dist_prior[id_in_cell_offset+k] <= -1.9f)
							continue;
						current_IoU = IoU_table[id_in_cell_offset+k];
	
						if(current_IoU > best_target_IoU)
						{
							second_target_IoU = best_target_IoU;
							best_target_IoU = current_IoU;
							candidate_box = k;
						}
						else if(current_IoU > second_target_IoU)
							second_target_IoU = current_IoU;
					}
	
					if(candidate_box == -1 || best_target_IoU < low_IoU_best_box_assoc)
						continue;
	
					effective_second_IoU = second_target_IoU;
					if(effective_second_IoU < low_IoU_best_box_assoc)
						effective_second_IoU = low_IoU_best_box_assoc;
	
					current_regret = best_target_IoU - effective_second_IoU;
					if(current_regret > best_regret + 0.0001f
					    || (fabsf(current_regret-best_regret) <= 0.0001f
					        && best_target_IoU > max_IoU))
					{
						best_regret = current_regret;
						max_IoU     = best_target_IoU;
						resp_targ   = j;
						resp_box    = candidate_box;
					}
				}
			}
			if(error_type == ERR_COMPLETE  && (resp_box == -1 || max_IoU < low_IoU_best_box_assoc))
			{
				best_dist = 1000000.0f;
				resp_box = -1; resp_targ = -1;
				for(j = 0; j < nb_in_cell; j++)
				{
					id_in_cell_offset = j*nb_box;
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset+k] > -1.9f
						    && dist_prior[id_in_cell_offset+k] < best_dist)
						{
							best_dist = dist_prior[id_in_cell_offset+k];
							resp_box = k;
							resp_targ = j;
						}
				}
				if(resp_box == -1)
					break;
	
				id_in_cell_offset = resp_targ*nb_box;
				max_IoU = IoU_table[id_in_cell_offset+resp_box];
				for(k = 0; k < nb_box; k++)
					if(dist_prior[id_in_cell_offset+k] > -1.9f && fabsf(dist_prior[id_in_cell_offset+k]-best_dist) < 0.0001f
					    && IoU_table[id_in_cell_offset+k] > max_IoU)
					{
						max_IoU = IoU_table[id_in_cell_offset+k];
						resp_box = k;
					}
			}
			if(resp_box == -1)
				break;
			l_t = in_cell_target_id[resp_targ]*target_offset;
	
			if(error_type == ERR_COMPLETE && diff_flag && (int)((float)target[l_t+7+nb_param]) > 0
			    && (max_IoU < diff_IoU_lim || (float)output[(resp_box*output_offset+7)*f_offset] < diff_obj_lim))
			{
				dist_prior[resp_targ*nb_box+resp_box] = -2.0f;
				continue;
			}
	
			for(j = 0; j < nb_in_cell; j++)
				dist_prior[j*nb_box+resp_box] = -2.0f;
			for(k = 0; k < nb_box; k++)
				dist_prior[resp_targ*nb_box+k] = -2.0f;
	
			prior_assigned_target[resp_box] = in_cell_target_id[resp_targ];
			box_locked[resp_box] = 2;
		}
	
		for(j = 0; j < nb_box; j++)
		{
			l_o = j*output_offset;
			targ_diff_flag = 0;
	
			for(k = 0; k < output_offset; k++)
				output_error[(l_o+k)*f_offset] = 0.0f;
	
			if(box_locked[j] == 2)
			{
				l_t = prior_assigned_target[j]*target_offset;
				if(diff_flag)
					targ_diff_flag = (int)((float)target[l_t+7+nb_param]);
	
				c_box_in_pix = box_in_pix + 6*j;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];
	
				for(l = 0; l < 6; l++)
					targ_int[l] = target[l_t+1+l];
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
	
				max_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(max_IoU > 0.98f)
					max_IoU = 0.98f;
	
				if(class_only_IoU > -1.9f)
					max_IoU = class_only_IoU;
	
				IoU_monitor[2*j] = (float)output[(l_o+7)*f_offset];
				IoU_monitor[2*j+1] = max_IoU;
	
				for(l = 0; l < 3; l++)
					obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];
	
				c_prior_size = prior_size + 3*j;
				for(l = 0; l < 3; l++)
				{
					obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
					if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);
					else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);
					else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
				}
			}
	
			if(fit_pos != -1 && box_locked[j] == 2 && class_only_IoU < -1.9f &&
			    (error_type == ERR_NATURAL || diff_flag == 0 || targ_diff_flag < 3))
			{
				for(k = 0; k < fit_dim; k++)
				{
					if(fit_pos == 1) l_targ = obj_in_offset[k];
					else             l_targ = 0.5f;
					output_error[(l_o+k)*f_offset] = 0.5f*coord_scale
						*((float)output[(l_o+k)*f_offset] - l_targ)
						*((float)output[(l_o+k)*f_offset] - l_targ);
				}
			}
	
			if(fit_size != -1 && box_locked[j] == 2	&& class_only_IoU < -1.9f &&
			    (error_type == ERR_NATURAL || diff_flag == 0 || targ_diff_flag < 3))
			{
				for(k = 0; k < fit_dim; k++)
				{
					if(fit_size == 1) l_targ = obj_in_offset[k+3];
					else              l_targ = 0.0f;
					output_error[(l_o+k+3)*f_offset] =  0.5f*size_scale
						*((float)output[(l_o+k+3)*f_offset] - l_targ)
						*((float)output[(l_o+k+3)*f_offset] - l_targ);
				}
			}
	
			if(fit_prob != -1 && ((box_locked[j] == 2 && (error_type == ERR_NATURAL || max_IoU > min_prob_IoU_lim)) || box_locked[j] == 0))
			{
				if(fit_prob == 1)
				{
					if(box_locked[j] == 2) l_targ = 0.98f;
					else                   l_targ = 0.02f;
				}
				else
					l_targ = 0.5f;
				l_scale = prob_scale;
				if(box_locked[j] == 0)
					l_scale *= lambda_noobj_prior[j];
				output_error[(l_o+6)*f_offset] =  0.5f*l_scale
					*((float)output[(l_o+6)*f_offset] - l_targ)
					*((float)output[(l_o+6)*f_offset] - l_targ);
			}
	
			if(fit_obj != -1 && ((box_locked[j] == 2 && (error_type == ERR_NATURAL || max_IoU > min_obj_IoU_lim)) || box_locked[j] == 0))
			{
				if(fit_obj == 1)
				{
					if(box_locked[j] == 2) l_targ = (1.0f+max_IoU)*0.5f;
					else                   l_targ = 0.02f;
				}
				else
					l_targ = 0.5f;
				l_scale = obj_scale;
				if(box_locked[j] == 0)
					l_scale *= lambda_noobj_prior[j];
				output_error[(l_o+7)*f_offset] =  0.5f*l_scale
					*((float)output[(l_o+7)*f_offset] - l_targ)
					*((float)output[(l_o+7)*f_offset] - l_targ);
			}
			if(fit_class != -1 && box_locked[j] == 2 &&
				(error_type == ERR_NATURAL|| (max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))))
			{
				if(!(fit_class == 0 && class_softmax)) /* else output_error must remain at 0*/
				{
					for(k = 0; k < nb_class; k++)
					{
						if(fit_class == 1)
						{
							if(k == (int)target[l_t]-1) l_targ = positive_class_target;
							else                        l_targ = negative_class_target;
						}
						else
							l_targ = 0.5f;
						if(class_softmax)
						{
							if((float)output[(l_o+8+k)*f_offset] < 0.0000001f)
								output_error[(l_o+8+k)*f_offset] = -class_scale*l_targ*logf(0.0000001f);
							else
								output_error[(l_o+8+k)*f_offset] = -class_scale*l_targ*logf((float)output[(l_o+8+k)*f_offset]);
						}
						else
						{
							output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
								*((float)output[(l_o+8+k)*f_offset] - l_targ)
								*((float)output[(l_o+8+k)*f_offset] - l_targ);
						}
					}
				}
			}
	
			if(fit_param != -1 && box_locked[j] == 2 &&
			    (error_type == ERR_NATURAL || (max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))))
			{
				for(k = 0; k < nb_param; k++)
				{
					if(fit_param == 1) l_targ = (float)target[l_t+7+k];
					else               l_targ = 0.5f;
					output_error[(l_o+8+nb_class+k)*f_offset] =	0.5f*param_ind_scale[k]*param_scale
						*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ)
						*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ);
				}
			}
		}
	}
	#ifdef _OPENMP
	}
	#endif
}


void YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	
	YOLO_activation_fct(current->output, 
		(size_t)(current->output_dim[0] * current->output_dim[1] * current->output_dim[2]) 
		* current->c_network->batch_size, current->a_biased_dim * current->c_network->length, 
		*a_param, current->a_size, a_param->class_softmax);
}


void YOLO_deriv(layer *current)
{
	//Empty on purpose
	//Only the last layer can be YOLO activated with a dedicated error
	//In this case, activation derivation is handled by deriv output_error directly (next function).
}


void YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	
	YOLO_deriv_error_fct(current->delta_o, current->output, current->c_network->dyn_target[0], a_param->flat_target_size, 
		(size_t)(current->output_dim[0] * current->output_dim[1] * current->output_dim[2]), 
		current->output_dim[0], current->output_dim[1], current->output_dim[2], *a_param, 
		(size_t)(current->output_dim[0] * current->output_dim[1] * current->output_dim[2]) * current->c_network->batch_size,
		current->c_network->iter * current->c_network->train.size);
}


void YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	
	YOLO_error_fct((float*)current->c_network->output_error, current->output, 
		current->c_network->dyn_target[0], a_param->flat_target_size, 
		(size_t)(current->output_dim[0] * current->output_dim[1] * current->output_dim[2]), 
		current->output_dim[0], current->output_dim[1], current->output_dim[2], *a_param, 
		(size_t)(current->output_dim[0] * current->output_dim[1] * current->output_dim[2]) * current->c_network->batch_size);
}


//#####################################################





