
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

//public are in "prototypes.h"

//private prototypes
void linear_activation(layer *current);
void linear_deriv(layer *previous);
void linear_deriv_output_error(layer* current);
void linear_output_error(layer* current);

void print_relu_activ_param(layer *current, char *activ);
void ReLU_activation(layer *current);
void ReLU_deriv(layer *previous);
void ReLU_deriv_output_error(layer* current);
void ReLU_output_error(layer* current);

void print_logistic_activ_param(layer *current, char *activ);
void logistic_activation(layer *current);
void logistic_deriv(layer *previous);
void logistic_deriv_output_error(layer* current);
void logistic_output_error(layer* current);

void softmax_activation(layer *current);
void softmax_deriv(layer *previous);
void softmax_deriv_output_error(layer *current);
void softmax_output_error(layer *current);

void YOLO_activation(layer *current);
void YOLO_deriv(layer *previous);
void YOLO_deriv_output_error(layer *current);
void YOLO_output_error(layer *current);

void ReLU_activation_fct(void *tab, int len, int dim, float saturation, float leaking_factor);
void ReLU_deriv_fct(void *deriv, void *value, int len, int dim, float saturation, float leaking_factor, int size);
void quadratic_deriv_output_error(void *delta_o, void *output, void *target, 
	int dim, int len, int size);
void quadratic_output_error(void *output_error, void *output, void *target, 
	int dim, int len, int size);
void logistic_activation_fct(void *tab, float beta, float saturation, int dim, int len, int size);
void logistic_deriv_fct(void *deriv, void* value, float beta, int len, int dim, int size);
void softmax_activation_fct(void *tab, int len, int dim, int size);
void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, int len, int dim, int size);
void cross_entropy_output_error(void *output_error, void *output, void *target, int len, int dim, int size);

//#####################################################


void define_activation(layer *current)
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
			current->deriv_activation = YOLO_deriv; //should not be needed
			//YOLO_activ_init(current); //needed ?
			break;
			
		case LINEAR:
			default:
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


void output_error_fct(layer* current)
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


void output_deriv_error(layer* current)
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

void output_error(layer* current)
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
			output_error_fct(current);
			break;
			
		default:
			output_error_fct(current);
			break;
	}	
}

void print_string_activ_param(layer *current, char* activ)
{
	switch(current->activation_type)
	{
		case LOGISTIC:
			print_logistic_activ_param(current, activ);
			break;
		case SOFTMAX:
			sprintf(activ,"SMAX");
			break;
		case YOLO:
			sprintf(activ,"YOLO");
			break;
		case RELU:
			print_relu_activ_param(current, activ);
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

	print_string_activ_param(current, temp_string);
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s", temp_string);
}


void load_activ_param(layer *current, const char *activ)
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

void set_linear_activ(layer *current, int size, int dim, int biased_dim)
{
	current->activ_param = (linear_param*) malloc(sizeof(linear_param));
	linear_param *param = (linear_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	current->bias_value = 0.5f;
}

void linear_activation(layer *current)
{
	//empty on purpose
}


void linear_deriv(layer *previous)
{
	//empty on purpose
}


void linear_deriv_output_error(layer *current)
{
	linear_param *param = (linear_param*)current->activ_param;
	quadratic_deriv_output_error(current->delta_o, current->output,
		current->c_network->target, (param->biased_dim)*current->c_network->length, param->dim, param->size);
}

void linear_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	quadratic_output_error(current->c_network->output_error, 
		current->output, current->c_network->target, (param->biased_dim)*current->c_network->length, param->dim, param->size);
}


//#####################################################




//#####################################################
//		 ReLU activation related functions
//#####################################################

void set_relu_activ(layer *current, int size, int dim, int biased_dim, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->saturation = 400.0f;
	param->leaking_factor = 0.1f;
	current->bias_value = 0.1f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_L");
	if(temp != NULL)
		sscanf(temp, "_L%f", &param->leaking_factor);
	
}

void print_relu_activ_param(layer *current, char *activ)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	sprintf(activ,"RELU_S%0.2f_L%0.2f", param->saturation, param->leaking_factor);
}

void ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	ReLU_activation_fct(current->output, param->size, param->dim, 
		param->saturation, param->leaking_factor);
}

//Is in fact a leaky ReLU, to obtain true ReLU define leaking_factor to 0
void ReLU_activation_fct(void *tab, int len, int dim, float saturation, float leaking_factor)
{
	int i;
	int pos;
	float *f_tab = (float*) tab;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < len; i++)
	{
		pos = i + i/dim;
		if(f_tab[pos] <= 0.0f)
			f_tab[pos] *= leaking_factor;
		else if(f_tab[pos] > saturation)
			f_tab[pos] = saturation + (f_tab[pos] - saturation)*leaking_factor;
	}
}


void ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	ReLU_deriv_fct(previous->delta_o, previous->output, param->size, param->dim,
		param->saturation, param->leaking_factor, param->size);
}


//should be adapted for both conv and dense layer if dim is properly defined
void ReLU_deriv_fct(void *deriv, void *value, int len, int dim, float saturation, float leaking_factor, int size)
{
	int i;
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
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

// Should re write a output function to take into account ReLU for Conv output format
void ReLU_deriv_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->target,
		(param->biased_dim) * current->c_network->length, param->dim, param->size);
	ReLU_deriv_fct(current->delta_o, current->output, 
		param->size, param->dim, param->saturation, param->leaking_factor, param->size);
}


void ReLU_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	quadratic_output_error(current->c_network->output_error, 
		current->output, current->c_network->target, (param->biased_dim)*current->c_network->length, 
		param->dim, param->size);
}


void quadratic_deriv_output_error(void *delta_o, void *output, void *target, int len, int dim, int size)
{
	int i;
	int pos;
	
	float *f_delta_o = (float*) delta_o;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{	
		if(i < len && (i+1)%(dim+1) != 0)
		{
			pos = i - i/(dim+1);
			f_delta_o[i] = (f_output[i] - f_target[pos]);
		}
		else
		{
			f_delta_o[i] = 0.0;
		}
	}
}



void quadratic_output_error(void *output_error, void *output, void *target, int len, int dim, int size)
{
	int i;
	int pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
		{
			pos = i - i/(dim+1);
			f_output_error[i] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
		}
		else
			f_output_error[i] = 0.0f;
	}
}


//#####################################################


//#####################################################
//		 Logistic activation related functions
//#####################################################


void set_logistic_activ(layer *current, int size, int dim, int biased_dim, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
	logistic_param *param = (logistic_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->saturation = 8.0f;
	param->beta = 1.0f;
	current->bias_value = -1.0f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_B");
	if(temp != NULL)
		sscanf(temp, "_B%f", &param->beta);
	
}


void print_logistic_activ_param(layer *current, char *activ)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	sprintf(activ,"LOGI_S%0.2f_B%0.2f", param->saturation, param->beta);
}

void logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	logistic_activation_fct(current->output, param->beta, param->saturation, (param->biased_dim)*current->c_network->length, param->dim, param->size);
}

void logistic_activation_fct(void *tab, float beta, float saturation, int len, int dim, int size)
{
	int i = 0;
	int pos;
	
	float *f_tab = (float*) tab;

	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len)
		{
			pos = i + i / dim;
			f_tab[pos] = -beta*f_tab[pos];
			if(f_tab[pos] > saturation)
				f_tab[pos] = saturation;
			f_tab[pos] = 1.0/(1.0 + expf(f_tab[pos]));
		}
		else
		{
			f_tab[i] = 0.0;
		}
	}
}


void logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	logistic_deriv_fct(previous->delta_o, previous->output, param->beta,
		(param->biased_dim)*previous->c_network->length, param->dim, param->size);
}

void logistic_deriv_fct(void *deriv, void* value, float beta, int len, int dim, int size)
{
	int i;
	
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
		{
			f_deriv[i] *= beta*f_value[i]*(1.0-f_value[i]);
		}
		else
			f_deriv[i] = 0.0;
	}
}


void logistic_deriv_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	quadratic_deriv_output_error(current->delta_o, current->output,
		current->c_network->target, (param->biased_dim)*current->c_network->length, param->dim, param->size);
	logistic_deriv_fct(current->delta_o, current->output, param->beta,
		(param->biased_dim)*current->c_network->length, param->dim, param->size);
	
}

void logistic_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	quadratic_output_error(current->c_network->output_error, 
		current->output, current->c_network->target, (param->biased_dim)*current->c_network->length, 
		param->dim, param->size);	
}

//#####################################################



//#####################################################
//		 Soft-Max activation related functions
//#####################################################


void set_softmax_activ(layer *current, int dim, int biased_dim)
{
	current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
	softmax_param *param = (softmax_param*)current->activ_param;	
	
	param->dim = dim;
	param->biased_dim = dim;
	current->bias_value = -1.0f;
}


void softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	softmax_activation_fct(current->output, current->c_network->length, param->dim, current->c_network->batch_size);
}

void softmax_activation_fct(void *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
	//provides a probabilistic output
	int i;
	int j;
	float *pos;
	float vmax;
	float normal = 0.0000001f;
	
	#pragma omp parallel for private(j, pos, vmax, normal) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		normal = 0.0000001f;
		if(i < len)
		{
			pos = (float*)tab + i*(dim+1);
			
			vmax = pos[0];
			for(j = 1; j < dim; j++)
				if(pos[j] > vmax)
					vmax = pos[j];
			
			for(j = 0; j < dim; j++)
			{	
				pos[j] = expf(pos[j]-vmax);
				normal += pos[j];
			}		
			pos[dim] = 0.0f;
			
			for(j = 0; j < dim; j++)
				pos[j] /= normal;
			pos[dim] = 0.0f;
		}
		else
		{
			pos = (float*)tab + i*(dim+1);		
			for(j = 0; j < dim; j++)
				pos[j] = 0.0f;
			pos[dim] = 0.0f;
		}
	}
}


void softmax_deriv(layer *previous)
{
	printf("Error : Softmax can not be used in the middle of the network !n");
	exit(EXIT_FAILURE);
}

void softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cross_entropy_deriv_output_error(current->delta_o, current->output,
		current->c_network->target, (param->biased_dim)*current->c_network->length, param->dim, 
		(param->biased_dim)*current->c_network->batch_size);
		
}

void softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cross_entropy_output_error(current->c_network->output_error,
		current->output, current->c_network->target, (param->biased_dim)*current->c_network->length,
		param->dim, (param->biased_dim)*current->c_network->batch_size);
		
}


void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, int len, int dim, int size)
{
	int i;
	int pos;
	
	float *f_delta_o = (float*) delta_o; 
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
		{
			pos = i - i/(dim+1);
			f_delta_o[i] = (f_output[i] - f_target[pos]);
		}
		else
			f_delta_o[i] = 0.0f;
	}
}

void cross_entropy_output_error(void *output_error, void *output, void *target, int len, int dim, int size)
{
	int i;
	int pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
		{
			pos = i - i/(dim+1);
			if(f_output[i] > 0.00001f)
				f_output_error[i] = -f_target[pos]*logf(f_output[i]);
			else
				f_output_error[i] = -f_target[pos]*logf(0.00001f);
		}
		else
			f_output_error[i] = 0.0f;
	}
}


//#####################################################


//#####################################################
//		 YOLO activation related functions
//#####################################################

//conv only activation, so most of elements can be extracted from c_param directly
void set_yolo_activ(layer *current)
{
	current->activ_param = (yolo_param*) malloc(sizeof(yolo_param));
	yolo_param *param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	if(current->c_network->y_param->nb_box*(8+current->c_network->y_param->nb_class
			+ current->c_network->y_param->nb_param) != c_param->nb_filters)
	{
		printf("%d %d\n", current->c_network->y_param->nb_box*(8+current->c_network->y_param->nb_class
			+ current->c_network->y_param->nb_param), c_param->nb_filters);
		printf("ERROR: Nb filters size mismatch in YOLO dimensions!n");
		exit(EXIT_FAILURE);
	}
	
	printf("Nb_elem IoU monitor %d\n", 2 * current->c_network->y_param->nb_box
		* c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
	//real copy to keep network properties accessible
	*param = *(current->c_network->y_param);	
	param->size = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] 
		* c_param->nb_filters * current->c_network->batch_size;
	printf(" %d %d %d %d\n", c_param->nb_filters, c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2]);
	param->dim = ((yolo_param*)current->activ_param)->size;
	param->biased_dim = ((yolo_param*)current->activ_param)->dim;
	param->cell_w = current->c_network->in_dims[0] / c_param->nb_area[0];
	param->cell_h = current->c_network->in_dims[1] / c_param->nb_area[1];
	param->cell_d = current->c_network->in_dims[2] / c_param->nb_area[2];
	param->IoU_monitor = (float*) calloc(2 * current->c_network->y_param->nb_box * c_param->nb_area[0] 
		* c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, sizeof(float));
	current->bias_value = 0.5;
}

float IoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


float GIoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
			- inter_3d;
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}

//order: xmin, ymin, zmin, xmax, ymax, zmax
float DIoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
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


int set_yolo_params(network *net, int nb_box, int IoU_type, float *prior_w, float *prior_h, float *prior_d, 
	float *yolo_noobj_prob_prior, int nb_class, int nb_param, int fit_dim, int strict_box_size, int rand_startup, 
	float rand_prob_best_box_assoc, float min_prior_forced_scaling, float *scale_tab, float **slopes_and_maxes_tab, 
	float *param_ind_scale, float *IoU_limits, int *fit_parts)
{
	int i;
	float *temp;
	float **sm;
	char IoU_type_char[40];
	
	if(net->y_param->fit_dim > 0)
	{
		printf("\n ERROR: Trying to update existing YOLO layer setup \n Not supported yet \n");
	}
	
	net->y_param->IoU_type = IoU_type;
	net->y_param->strict_box_size_association = strict_box_size;
	
	if(rand_startup < 0)
		net->y_param->rand_startup = 64000;
	else
		net->y_param->rand_startup = rand_startup;
	
	if(rand_prob_best_box_assoc < 0.0f)
		net->y_param->rand_prob_best_box_assoc = 0.05f;
	else
		net->y_param->rand_prob_best_box_assoc = rand_prob_best_box_assoc;
	
	if(min_prior_forced_scaling < 0.0f)
		net->y_param->min_prior_forced_scaling = 1.5f;
	else
		net->y_param->min_prior_forced_scaling = min_prior_forced_scaling;
		
	if(fit_dim <= 0)
	{
		fit_dim = 0;
		if(prior_w != NULL)
			fit_dim++;
		if(prior_h != NULL)
			fit_dim++;
		if(prior_d != NULL)
			fit_dim++;
	}
	net->y_param->fit_dim = fit_dim;
	
	if(prior_w == NULL)
	{
		prior_w = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			prior_w[i] = 1.0f;
	}
	
	if(prior_h == NULL)
	{
		prior_h = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			prior_h[i] = 1.0f;
	}
	
	if(prior_d == NULL)
	{
		prior_d = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			prior_d[i] = 1.0f;
	}
	
	if(yolo_noobj_prob_prior == NULL)
	{
		yolo_noobj_prob_prior = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			yolo_noobj_prob_prior[i] = 0.2f;
	}
	
	if(scale_tab == NULL)
	{
		scale_tab = (float*) calloc(6, sizeof(float));
		scale_tab[0] = 2.0f; /*Pos */ scale_tab[1] = 2.0f; /*Size */
		scale_tab[2] = 1.0f; /*Proba*/ scale_tab[3] = 2.0f; /*Objct*/
		scale_tab[4] = 1.0f; /*Class*/ scale_tab[5] = 1.0f; /*Param*/
	}
	
	if(slopes_and_maxes_tab == NULL)
	{
		temp = (float*) calloc(6*3, sizeof(float));
		slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
		for(i = 0; i < 6; i++)
			slopes_and_maxes_tab[i] = &temp[i*3];
	
		sm = slopes_and_maxes_tab;
		sm[0][0] = 1.0f; sm[0][1] = 4.5f; sm[0][2] = -4.5f;
		sm[1][0] = 1.0f; sm[1][1] = 1.2f; sm[1][2] = -1.4f;
		sm[2][0] = 1.0f; sm[2][1] = 4.5f; sm[2][2] = -4.5f;
		sm[3][0] = 1.0f; sm[3][1] = 4.5f; sm[3][2] = -4.5f;
		sm[4][0] = 1.0f; sm[4][1] = 4.5f; sm[4][2] = -4.5f;
		sm[5][0] = 1.0f; sm[5][1] = 2.0f; sm[5][2] = -0.2f;
	}
	
	if(param_ind_scale == NULL)
	{
		param_ind_scale = (float*) calloc(nb_param, sizeof(float));
		for(i = 0; i < nb_param; i++)
			param_ind_scale[i] = 1.0f;
	}

	if(IoU_limits == NULL)
	{
		IoU_limits = (float*) calloc(6,sizeof(float));
		switch(IoU_type)
		{
			case IOU:
				IoU_limits[0] = 0.5f; IoU_limits[1] = 0.1f;
				IoU_limits[2] = 0.0f; IoU_limits[3] = 0.0f;
				IoU_limits[4] = 0.2f; IoU_limits[5] = 0.2f;
				break;
			case GIOU:
				IoU_limits[0] = 0.4f; IoU_limits[1] = -0.5f;
				IoU_limits[2] = -1.0f; IoU_limits[3] = -1.0f;
				IoU_limits[4] = -0.3f; IoU_limits[5] = -0.3f;
				break;
			default:
			case DIOU:
				IoU_limits[0] = 0.3f; IoU_limits[1] = -0.5f;
				IoU_limits[2] = -1.0f; IoU_limits[3] = -1.0f;
				IoU_limits[4] = -0.4f; IoU_limits[5] = -0.4f;
				break;
		}
	}
	
	if(fit_parts == NULL)
	{
		fit_parts = (int*) calloc(5,sizeof(int));
		fit_parts[0] = 1; /*Size */ fit_parts[1] = 1; /*Prob */
		fit_parts[2] = 1; /*Object*/
		if(nb_class > 0) /*Class*/
			fit_parts[3] = 1;
		else
			fit_parts[3] = 0; 
		if(nb_param > 0) /*Param*/
			fit_parts[4] = 1;
		else
			fit_parts[4] = 0;
	}
	
	net->y_param->nb_box = nb_box;
	net->y_param->prior_w = prior_w;
	net->y_param->prior_h = prior_h;
	net->y_param->prior_d = prior_d;
	net->y_param->noobj_prob_prior = yolo_noobj_prob_prior;
	net->y_param->nb_class = nb_class;
	net->y_param->nb_param = nb_param;
	
	//Priors table must be sent to GPU memory if C_CUDA
	net->y_param->scale_tab = scale_tab;
	net->y_param->slopes_and_maxes_tab = slopes_and_maxes_tab;
	net->y_param->param_ind_scale = param_ind_scale;
	net->y_param->IoU_limits = IoU_limits;
	net->y_param->fit_parts = fit_parts;
	
	switch(net->y_param->IoU_type)
	{
		default:
		case IOU:
			sprintf(IoU_type_char, "Classical IoU");
			net->y_param->c_IoU_fct = IoU_fct;
			break;
		case GIOU:
			sprintf(IoU_type_char, "Generalized GIoU");
			net->y_param->c_IoU_fct = GIoU_fct;
			break;
		case DIOU:
			sprintf(IoU_type_char, "Distance DIoU");
			net->y_param->c_IoU_fct = DIoU_fct;
			break;
	}
	
	printf("\n YOLO layer setup \n");
	printf(" -------------------------------------------------------------------\n");
	printf(" Nboxes = %d\n Nclasses = %d\n Nparams = %d\n IoU type = %s\n",
			net->y_param->nb_box, net->y_param->nb_class, net->y_param->nb_param, IoU_type_char);
	printf(" W priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.2f ", net->y_param->prior_w[i]);
	printf("]\n H priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.2f ", net->y_param->prior_h[i]);
	printf("]\n D priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.2f ", net->y_param->prior_d[i]);
	printf("]\n");
	printf(" Nb dim fitted : %d\n",net->y_param->fit_dim);
	printf(" No obj. prob. priors\n          = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.3f ", net->y_param->noobj_prob_prior[i]);
	printf("]\n");
	printf(" Fit parts: (Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 5; i++)
		printf(" %d ",net->y_param->fit_parts[i]);
	printf("]\n");
	printf(" Error scales: (Pos., Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf(" %5.3f ",net->y_param->scale_tab[i]);
	printf("]\n");
	printf(" IoU lim.: (GdNotBest, LowBest, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf("%7.3f ", net->y_param->IoU_limits[i]);
	printf("]\n");
	// Lots of new (and some old) arguments not displayed !
	
	if(net->y_param->nb_param > 0)
	{
		printf(" Individual param. error scaling: \n   = [");
		for(i = 0; i < net->y_param->nb_param; i++)
			printf("%7.3f ", net->y_param->param_ind_scale[i]);
		printf("]\n");
	}
	printf("\n *** Association process parameters ***\n");
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
	printf("  Startup random association Nb. item : %d\n", net->y_param->rand_startup);
	printf("  Low best IoU threshold for best prior assoc.: %5.3f\n", net->y_param->IoU_limits[1]);
	printf("  Random proportion of best prior assoc.: %5.3f\n", net->y_param->rand_prob_best_box_assoc);
	printf("  Forced smallest prior association scaling : %6.3f\n", net->y_param->min_prior_forced_scaling);
	printf("\n -------------------------------------------------------------------\n\n");
	
	return (nb_box*(8+nb_class+nb_param));
}


void YOLO_activation_fct(void *i_tab, int flat_offset, int len, yolo_param y_param, int size)
{	
	float* tab = (float*) i_tab;
	
	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;
	/*Default values are in activ_function.c (set_yolo_params)*/
	float **sm_tab = y_param.slopes_and_maxes_tab;
	int fit_dim = y_param.fit_dim;	
	int i, col, in_col;
	
	#pragma omp parallel for private(col, in_col) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		col = i / flat_offset;
		in_col = col%(8+nb_class+nb_param);
		
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
			tab[i] = -sm_tab[4][0]*tab[i];	
			if(tab[i] > sm_tab[4][1])
				tab[i] = sm_tab[4][1];
			else if(tab[i] < sm_tab[4][2])	
				tab[i] = sm_tab[4][2];
			tab[i] = 1.0f/(1.0f + expf(tab[i]));	
	
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

// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
void YOLO_deriv_error_fct
	(void *i_delta_o, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, int nb_im_epoch)
{
	
	float* t_delta_o = (float*) i_delta_o;
	float* t_output = (float*) i_output;
	float* t_target = (float*) i_target;

	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	int fit_dim = y_param.fit_dim, rand_startup = y_param.rand_startup;
	float rand_prob_best_box_assoc = y_param.rand_prob_best_box_assoc;
	float min_prior_forced_scaling = y_param. min_prior_forced_scaling;

	float coord_scale = y_param.scale_tab[0], size_scale = y_param.scale_tab[1];
	float prob_scale = y_param.scale_tab[2], obj_scale  = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];

	float *param_ind_scale = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;

	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];
	float min_prob_IoU_lim = y_param.IoU_limits[2], min_obj_IoU_lim = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim = y_param.IoU_limits[5];
	int fit_size = y_param.fit_parts[0], fit_prob = y_param.fit_parts[1], fit_obj = y_param.fit_parts[2];
	int fit_class = y_param.fit_parts[3], fit_param = y_param.fit_parts[4];

	int c_pix;
	
	#pragma omp parallel for schedule(guided,4)
	for(c_pix = 0; c_pix < size; c_pix++)
	{
		//All private variables inside the loop for convenience
		//Should be marginal since one iteration cost is already high
		
		float *delta_o, *output, *target;
		int l_o, l_t;
		int i, j, k, l;
		int c_batch, f_offset;
		int nb_obj_target;
		int is_in_cell, nb_in_cell, id_in_cell, resp_box = -1;
		float best_dist;
		int dist_id;
		float max_IoU, current_IoU;
		int cell_x, cell_y, cell_z;
		int obj_cx, obj_cy, obj_cz;
		float *box_in_pix, *c_box_in_pix;
		float obj_in_offset[6];
		float *IoU_table, *dist_prior;
		int *box_locked;
		float out_int[6], targ_int[6];
		float targ_w, targ_h, targ_d;
		
		c_batch = c_pix / flat_output_size;
		target = t_target + flat_target_size * c_batch;
		f_offset = size;

		i = c_pix % flat_output_size;
		cell_z = i / (nb_area_w*nb_area_h);
		cell_y = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
		cell_x = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;

		delta_o = t_delta_o + (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
		output = t_output + (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

		nb_obj_target = target[0];
		target++;

		if(nb_obj_target == 0)
			continue;

		IoU_table = (float*) malloc(nb_box*nb_obj_target*sizeof(float));
		dist_prior = (float*) malloc(nb_box*nb_obj_target*sizeof(float));
		box_locked = (int*) malloc(nb_box*sizeof(int));
		box_in_pix = (float*) malloc(nb_box*6*sizeof(float));

		
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			c_box_in_pix = box_in_pix+k*6;
			l_o = k*(8+nb_class+nb_param);
			c_box_in_pix[0] = ((float)output[(l_o+0)*f_offset] + cell_x) * cell_w;
			c_box_in_pix[1] = ((float)output[(l_o+1)*f_offset] + cell_y) * cell_h;
			c_box_in_pix[2] = ((float)output[(l_o+2)*f_offset] + cell_z) * cell_d;
			c_box_in_pix[3] = prior_w[k]*expf((float)output[(l_o+3)*f_offset]);
			c_box_in_pix[4] = prior_h[k]*expf((float)output[(l_o+4)*f_offset]);
			c_box_in_pix[5] = prior_d[k]*expf((float)output[(l_o+5)*f_offset]);
		}

		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*(7+nb_param);
			for(k = 0; k < 6; k++)
				targ_int[k] = target[l_t+1+k];

			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];

			is_in_cell = 0;

			obj_cx = (int)( ((float)target[l_t+4] + (float)target[l_t+1])*0.5f / cell_w);
			obj_cy = (int)( ((float)target[l_t+5] + (float)target[l_t+2])*0.5f / cell_h);
			obj_cz = (int)( ((float)target[l_t+6] + (float)target[l_t+3])*0.5f / cell_d);

			if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
			{
				is_in_cell = 1;
				nb_in_cell++;
			}

			for(k = 0; k < nb_box; k++)
			{
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(box_locked[k] == 0 && current_IoU > good_IoU_lim)
					box_locked[k] = 1;

				if(is_in_cell)
				{
					IoU_table[j*nb_box + k] = current_IoU;
					dist_prior[j*nb_box + k] = sqrt(
						 (targ_w-prior_w[k])*(targ_w-prior_w[k])
						+(targ_h-prior_h[k])*(targ_h-prior_h[k])
						+(targ_d-prior_d[k])*(targ_d-prior_d[k]));
				}
				else
				{
					IoU_table[j*nb_box + k] = -2.0f;
					dist_prior[j*nb_box + k] = 1.0f;
				}
			}

			if(is_in_cell && strict_box_size_association > 0)
			{
				for(l = 0; l < strict_box_size_association; l++)
				{
					best_dist = 10000000000;
					for(k = 0; k < nb_box; k++)	/* Find the closest theoritical prior */
						if(dist_prior[j*nb_box+k] > 0 && dist_prior[j*nb_box+k] < best_dist)
							best_dist = dist_prior[j*nb_box+k];
					if(best_dist < 10000000000)
						for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any)*/
							if(abs(dist_prior[j*nb_box+k]-best_dist) < 0.001f)
								dist_prior[j*nb_box+k] = -1.0f;
				}
			}
			else
				for(k = 0; k < nb_box; k++)
					dist_prior[j*nb_box+k] = -1.0f;
		}

		for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)
		{
			max_IoU = -2.0f;
			resp_box = -1;
			for(k = 0; k < nb_obj_target*nb_box; k++)
				if(IoU_table[k] > max_IoU && dist_prior[k] < 0.0f)
				{
					max_IoU = IoU_table[k];
					resp_box = k;
				}

			if(resp_box == -1)	/* Might happen if all good priors are already taken. In that case relax the constrain*/
				for(k = 0; k < nb_obj_target*nb_box; k++)
					if(IoU_table[k] > max_IoU)
					{
						max_IoU = IoU_table[k];
						resp_box = k;
					}

			if(resp_box == -1) /* Only happen if all the box are taken (more targets in the cell than boxes) */
				break;

			j = resp_box / nb_box;
			resp_box = resp_box % nb_box;
			l_t = j*(7+nb_param);
			for(k = 0; k < 6; k++)
				targ_int[k] = target[l_t+1+k];

			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];

			for(k = 0; k < nb_box; k++)
				dist_prior[j*nb_box+k] = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])
									+(targ_h-prior_h[k])*(targ_h-prior_h[k])
									+(targ_d-prior_d[k])*(targ_d-prior_d[k]));

			if(max_IoU < low_IoU_best_box_assoc || 
				random_uniform() < rand_prob_best_box_assoc)
			{
				best_dist = 10000000000;
				dist_id = -1;
				for(k = 0; k < nb_box; k++)
					if(dist_prior[j*nb_box+k] < best_dist && box_locked[k] != 2)
					{
						best_dist = dist_prior[j*nb_box+k];
						dist_id = k;
					}
				resp_box = dist_id;
			}

			for(k = 0; k < nb_box; k++)
				IoU_table[j*nb_box + k] = -2.0f;
			
			if(targ_w*targ_h*targ_d < min_prior_forced_scaling*prior_w[0]*prior_h[0]*prior_d[0] && box_locked[0] != 2)
				resp_box = 0;

			if(nb_im_epoch < rand_startup)
				for(k = 0; k < 10; k++)
				{
					resp_box = random_uniform()*nb_box;
					if(box_locked[resp_box] != 2)
						break;
				}

			if(resp_box == -1) /* Only happen if all the box are taken (more targets in the cell than boxes) */
				break;

			l_o = resp_box*(8+nb_class+nb_param);
			for(k = 0; k < nb_obj_target; k++)
				IoU_table[k*nb_box + resp_box] = -2.0f;

			box_locked[resp_box] = 2;

			c_box_in_pix = box_in_pix+resp_box*6;
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

			max_IoU = y_param.c_IoU_fct(out_int, targ_int);
			if(max_IoU > 0.98f)
				max_IoU = 0.98f;

			obj_in_offset[0] = ((targ_int[3] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;
			obj_in_offset[1] = ((targ_int[4] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;
			obj_in_offset[2] = ((targ_int[5] + targ_int[2])*0.5f - cell_z*cell_d)/(float)cell_d;
			obj_in_offset[3] = (targ_w)/(float)prior_w[resp_box];
			if(obj_in_offset[3] < size_min_sat)
				obj_in_offset[3] = logf(size_min_sat);
			else if(obj_in_offset[3] > size_max_sat)
				obj_in_offset[3] = logf(size_max_sat);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);
			obj_in_offset[4] = (targ_h)/(float)prior_h[resp_box];
			if(obj_in_offset[4] < size_min_sat)
				obj_in_offset[4] = logf(size_min_sat);
			else if(obj_in_offset[4] > size_max_sat)
				obj_in_offset[4] = logf(size_max_sat);
			else
				obj_in_offset[4] = logf(obj_in_offset[4]);
			obj_in_offset[5] = (targ_d)/(float)prior_d[resp_box];
			if(obj_in_offset[5] < size_min_sat)
				obj_in_offset[5] = logf(size_min_sat);
			else if(obj_in_offset[5] > size_max_sat)
				obj_in_offset[5] = logf(size_max_sat);
			else
				obj_in_offset[5] = logf(obj_in_offset[5]);

			for(k = 0; k < 3; k++)
			{
				if(fit_dim > k)
					delta_o[(l_o+k)*f_offset] = (
						sm_tab[0][0]*coord_scale*(float)output[(l_o+k)*f_offset]
						*(1.0f-(float)output[(l_o+k)*f_offset])*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]));
				else
					delta_o[(resp_box*(8+nb_class+nb_param)+k)*f_offset] = (0.0f);
			}

			switch(fit_size)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							delta_o[(l_o+k+3)*f_offset] = (sm_tab[1][0]*size_scale
								*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]));
						else
							delta_o[(l_o+k+3)*f_offset] = (0.0f);
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							delta_o[(l_o+k+3)*f_offset] = (sm_tab[1][0]*size_scale
								*((float)output[(l_o+k+3)*f_offset]-0.0f));
						else
							delta_o[(l_o+k+3)*f_offset] = (0.0f);
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						delta_o[(l_o+k+3)*f_offset] = (0.0f);
					break;
			}

			switch(fit_prob)
			{
				case 1:
					if(max_IoU > min_prob_IoU_lim)
						delta_o[(l_o+6)*f_offset] = (
							sm_tab[2][0]*prob_scale*(float)output[(l_o+6)*f_offset]
							*(1.0f-(float)output[(l_o+6)*f_offset])*((float)output[(l_o+6)*f_offset]-0.98f));
					else
						delta_o[(l_o+6)*f_offset] = (0.0f);
					break;
				case 0:
					delta_o[(l_o+6)*f_offset] = (
						sm_tab[2][0]*prob_scale*(float)output[(l_o+6)*f_offset]
						*(1.0f-(float)output[(l_o+6)*f_offset])*((float)output[(l_o+6)*f_offset]-0.5f));
					break;
				case -1:
					delta_o[(l_o+6)*f_offset] = (0.0f);
					break;
			}

			switch(fit_obj)
			{
				case 1:
					if(max_IoU > min_obj_IoU_lim)
						delta_o[(l_o+7)*f_offset] = (
							sm_tab[3][0]*obj_scale*(float)output[(l_o+7)*f_offset]
							*(1.0f-(float)output[(l_o+7)*f_offset])*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5));
					else
						delta_o[(l_o+7)*f_offset] = (0.0f);
					break;
				case 0:
					delta_o[(l_o+7)*f_offset] = (
						sm_tab[3][0]*obj_scale*(float)output[(l_o+7)*f_offset]
						*(1.0f-(float)output[(l_o+7)*f_offset])*((float)output[(l_o+7)*f_offset]-0.5f));
					break;
				case -1:
					delta_o[(l_o+7)*f_offset] = (0.0f);
					break;
			}

			/*Note : mean square error on classes => could be changed to soft max but difficult to balance*/
			switch(fit_class)
			{
				case 1:
					if(max_IoU > min_class_IoU_lim)
						for(k = 0; k < nb_class; k++)
						{
							if(k == (int) target[l_t]-1)
								delta_o[(l_o+8+k)*f_offset] = 
									 (sm_tab[4][0]*class_scale*(float)output[(l_o+8+k)*f_offset]
									*(1.0f-(float)output[(l_o+8+k)*f_offset])*((float)output[(l_o+8+k)*f_offset]-0.98f));
							else
								delta_o[(l_o+8+k)*f_offset] = 
									 (sm_tab[4][0]*class_scale*(float)output[(l_o+8+k)*f_offset]
									*(1.0f-(float)output[(l_o+8+k)*f_offset])*((float)output[(l_o+8+k)*f_offset]-0.02f));
						}
					else
						for(k = 0; k < nb_class; k++)
							delta_o[(l_o+8+k)*f_offset] = (0.0f);
					break;
				case 0:
					for(k = 0; k < nb_class; k++)
						delta_o[(l_o+8+k)*f_offset] = 
							 (sm_tab[4][0]*class_scale*(float)output[(l_o+8+k)*f_offset]
							*(1.0f-(float)output[(l_o+8+k)*f_offset])*((float)output[(l_o+8+k)*f_offset]-0.5f));
					break;
				case -1:
					for(k = 0; k < nb_class; k++)
						delta_o[(l_o+8+k)*f_offset] = (0.0f);
					break;
			}

			/*Linear activation of additional parameters*/
			switch(fit_param)
			{
				case 1:
					if(max_IoU > min_param_IoU_lim)
						for(k = 0; k < nb_param; k++)
							delta_o[(l_o+8+nb_class+k)*f_offset] = 
								 (param_ind_scale[k]*sm_tab[5][0]*param_scale
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));
					else
						for(k = 0; k < nb_param; k++)
							delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
					break;
				case 0:
					for(k = 0; k < nb_param; k++)
						delta_o[(l_o+8+nb_class+k)*f_offset] = 
							 (param_ind_scale[k]*sm_tab[5][0]*param_scale
							*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));
					break;
				case -1:
					for(k = 0; k < nb_param; k++)
						delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
					break;
			}
		}

		for(j = 0; j < nb_box; j++)
		{
			/*If no match (means no IoU > 0.5) only update Objectness toward 0 */
			/*(here it means error compute)! (no coordinate nor class update)*/
			l_o = j*(8+nb_class+nb_param);
			if(box_locked[j] != 2)
			{
				for(k = 0; k < 6; k++)
					delta_o[(l_o+k)*f_offset] = 0.0f;

				if(box_locked[j] == 1)
				{
					delta_o[(l_o+6)*f_offset] = 0.0f;
					delta_o[(l_o+7)*f_offset] = 0.0f;
				}
				else
				{
					switch(fit_prob)
					{
						case 1:
							delta_o[(l_o+6)*f_offset] = (
								sm_tab[2][0]*(lambda_noobj_prior[j])*prob_scale*(float)output[(l_o+6)*f_offset]
								*(1.0f-(float)output[(l_o+6)*f_offset])*((float)output[(l_o+6)*f_offset]-0.02f));
							break;
						case 0:
							delta_o[(l_o+6)*f_offset] = (
								sm_tab[2][0]*(lambda_noobj_prior[j])*prob_scale*(float)output[(l_o+6)*f_offset]
								*(1.0f-(float)output[(l_o+6)*f_offset])*((float)output[(l_o+6)*f_offset]-0.5f));
							break;
						case -1:
							delta_o[(l_o+6)*f_offset] = (0.0f);
							break;
					}
					switch(fit_obj)
					{
						case 1:
							delta_o[(l_o+7)*f_offset] = (
								sm_tab[3][0]*(lambda_noobj_prior[j])*obj_scale*(float)output[(l_o+7)*f_offset]
								*(1.0f-(float)output[(l_o+7)*f_offset])*((float)output[(l_o+7)*f_offset]-0.02f));
							break;
						case 0:
							delta_o[(l_o+7)*f_offset] = (
								sm_tab[3][0]*(lambda_noobj_prior[j])*obj_scale*(float)output[(l_o+7)*f_offset]
								*(1.0f-(float)output[(l_o+7)*f_offset])*((float)output[(l_o+7)*f_offset]-0.5f));
							break;
						case -1:
							delta_o[(l_o+7)*f_offset] = (0.0f);
							break;
					}
				}

				for(k = 0; k < nb_class; k++)
					delta_o[(l_o+8+k)*f_offset] = (0.0f);

				for(k = 0; k < nb_param; k++)
					delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
			}
		}
		free(IoU_table);
		free(dist_prior);
		free(box_in_pix);
		free(box_locked);
	}
}

// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
void YOLO_error_fct
	(float *i_output_error, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)
{		
	float* t_output = (float*) i_output;
	float* t_target = (float*) i_target;
	
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;	  
	int fit_dim =  y_param.fit_dim;

	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;

	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[2], min_obj_IoU_lim = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim = y_param.IoU_limits[5];
	int fit_size = y_param.fit_parts[0], fit_prob = y_param.fit_parts[1], fit_obj = y_param.fit_parts[2];
	int fit_class = y_param.fit_parts[3], fit_param = y_param.fit_parts[4];

	float *param_ind_scale = y_param.param_ind_scale;
	float *t_IoU_monitor = y_param.IoU_monitor;
	
	int c_pix;
	
	#pragma omp parallel for schedule(guided,4)
	for(c_pix = 0; c_pix < size; c_pix++)
	{	
		float *output, *target, *output_error, *IoU_monitor;
		int l_o, l_t;
		int i, j, k;
		int c_batch, f_offset;
		int nb_obj_target;
		int is_in_cell, nb_in_cell, id_in_cell, resp_box = -1;
		float max_IoU, current_IoU;
		int cell_x, cell_y, cell_z;
		int obj_cx, obj_cy, obj_cz;
		float *box_in_pix, *c_box_in_pix;
		float obj_in_offset[6];
		float *IoU_table, *dist_prior;
		int *box_locked;
		float out_int[6], targ_int[6];
		float targ_w, targ_h, targ_d;
	
		c_batch = c_pix / flat_output_size;
		target = t_target + flat_target_size * c_batch;
		f_offset = size;
		
		i = c_pix % flat_output_size;
		cell_z = i / (nb_area_w*nb_area_h);
		cell_y = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
		cell_x = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
		
		output_error = i_output_error + (nb_area_w*nb_area_h*nb_area_d) * c_batch 
			+ cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
		output = t_output + (nb_area_w*nb_area_h*nb_area_d) * c_batch 
			+ cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
		
		IoU_monitor = t_IoU_monitor + 2 * nb_box * ((nb_area_w*nb_area_h*nb_area_d) * c_batch 
			+ cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x);
		
		nb_obj_target = target[0];
		target ++;
		
		IoU_table = (float*) malloc(nb_box*nb_obj_target*sizeof(float));
		dist_prior = (float*) malloc(nb_box*nb_obj_target*sizeof(float));
		box_locked = (int*) malloc(nb_box*sizeof(int));
		box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
		
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			c_box_in_pix = box_in_pix+k*6;
			l_o = k*(8+nb_class+nb_param);
			c_box_in_pix[0] = ((float)output[(l_o+0)*f_offset] + cell_x) * cell_w;
			c_box_in_pix[1] = ((float)output[(l_o+1)*f_offset] + cell_y) * cell_h;
			c_box_in_pix[2] = ((float)output[(l_o+2)*f_offset] + cell_z) * cell_d;
			c_box_in_pix[3] = prior_w[k]*expf((float)output[(l_o+3)*f_offset]);
			c_box_in_pix[4] = prior_h[k]*expf((float)output[(l_o+4)*f_offset]);
			c_box_in_pix[5] = prior_d[k]*expf((float)output[(l_o+5)*f_offset]);
	
			IoU_monitor[k*2] = 0.0f;
			IoU_monitor[k*2+1] = -1.0f;
		}
	
		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*(7+nb_param);
			for(k = 0; k < 6; k++)
				targ_int[k] = target[l_t+1+k];
	
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
	
			is_in_cell = 0;
	
			obj_cx = (int)( ((float)target[l_t+4] + (float)target[l_t+1])*0.5f / cell_w);
			obj_cy = (int)( ((float)target[l_t+5] + (float)target[l_t+2])*0.5f / cell_h);
			obj_cz = (int)( ((float)target[l_t+6] + (float)target[l_t+3])*0.5f / cell_d);
	
			if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
			{
				is_in_cell = 1;
				nb_in_cell++;
			}
	
			for(k = 0; k < nb_box; k++)
			{
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];
	
				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				if(box_locked[k] == 0 && current_IoU > good_IoU_lim)
					box_locked[k] = 1;
	
				if(is_in_cell)
				{
					IoU_table[j*nb_box + k] = current_IoU;
					dist_prior[j*nb_box + k] = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])
									+(targ_h-prior_h[k])*(targ_h-prior_h[k])
									+(targ_d-prior_d[k])*(targ_d-prior_d[k]));
				}
				else
				{
					IoU_table[j*nb_box + k] = -2.0f;
					dist_prior[j*nb_box + k] = 1.0f;
				}
			}
	
			for(k = 0; k < nb_box; k++)
				dist_prior[j*nb_box+k] = -1.0f;
		}
	
		for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)
		{
			max_IoU = -2.0f;
			resp_box = -1;
			for(k = 0; k < nb_obj_target*nb_box; k++)
				if(IoU_table[k] > max_IoU && dist_prior[k] < 0.0f)
				{
					max_IoU = IoU_table[k];
					resp_box = k;
				}
	
			if(resp_box == -1) /* Only happen if all the box are taken (more targets in the cell than boxes) */
				break;
	
			j = resp_box / nb_box;
			resp_box = resp_box % nb_box;
			l_t = j*(7+nb_param);
	
			for(k = 0; k < 6; k++)
				targ_int[k] = target[l_t+1+k];
	
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
	
			for(k = 0; k < nb_box; k++)
				IoU_table[j*nb_box + k] = -2.0f;
	
			if(resp_box == -1)	/* Only happen if all the box are taken (more targets in the cell than boxes) */
				break;
	
			l_o = resp_box*(8+nb_class+nb_param);
			for(k = 0; k < nb_obj_target; k++)
				IoU_table[k*nb_box + resp_box] = -2.0f;
	
			box_locked[resp_box] = 2;
	
			if(max_IoU > 0.98f)
				max_IoU = 0.98f;
	
			c_box_in_pix = box_in_pix+resp_box*6;
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];
	
			max_IoU = y_param.c_IoU_fct(out_int, targ_int);
	
			IoU_monitor[resp_box*2] = 1.0f;
			IoU_monitor[resp_box*2+1] = max_IoU*(float)output[(l_o+6)*f_offset];
	
			obj_in_offset[0] = fmaxf(0.01f,fminf(0.99,((targ_int[3] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w));
			obj_in_offset[1] = fmaxf(0.01f,fminf(0.99,((targ_int[4] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h));
			obj_in_offset[2] = fmaxf(0.01f,fminf(0.99,((targ_int[5] + targ_int[2])*0.5f - cell_z*cell_d)/(float)cell_d));
			obj_in_offset[3] = (targ_w)/(float)prior_w[resp_box];
			if(obj_in_offset[3] < size_min_sat)
				obj_in_offset[3] = logf(size_min_sat);
			else if(obj_in_offset[3] > size_max_sat)
				obj_in_offset[3] = logf(size_max_sat);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);
			obj_in_offset[4] = (targ_h)/(float)prior_h[resp_box];
			if(obj_in_offset[4] < size_min_sat)
				obj_in_offset[4] = logf(size_min_sat);
			else if(obj_in_offset[4] > size_max_sat)
				obj_in_offset[4] = logf(size_max_sat);
			else
				obj_in_offset[4] = logf(obj_in_offset[4]);
			obj_in_offset[5] = (targ_d)/(float)prior_d[resp_box];
			if(obj_in_offset[5] < size_min_sat)
				obj_in_offset[5] = logf(size_min_sat);
			else if(obj_in_offset[5] > size_max_sat)
				obj_in_offset[5] = logf(size_max_sat);
			else
				obj_in_offset[5] = logf(obj_in_offset[5]);
	
			for(k = 0; k < 3; k++)
			{
				if(fit_dim > k)
					output_error[(l_o+k)*f_offset] = 0.5f*coord_scale
						*((float)output[(l_o+k)*f_offset]-obj_in_offset[k])*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]);
				else
					output_error[(l_o+k)*f_offset] = 0.0f;
			}
	
			switch(fit_size)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale
							*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3])
							*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]);
						else
							output_error[(l_o+k+3)*f_offset] = 0.0f;
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale
							*((float)output[(l_o+k+3)*f_offset]-0.0f)*((float)output[(l_o+k+3)*f_offset]-0.0f);
						else
							output_error[(l_o+k+3)*f_offset] = 0.0f;
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						output_error[(l_o+k+3)*f_offset] = 0.0f;
					break;
			}
	
			switch(fit_prob)
			{
				case 1:
					if(max_IoU > min_prob_IoU_lim)
						output_error[(l_o+6)*f_offset] = 0.5f*prob_scale
							*((float)output[(l_o+6)*f_offset]-0.98f)*((float)output[(l_o+6)*f_offset]-0.98f);
					else
						output_error[(l_o+6)*f_offset] = 0.0f;
					break;
				case 0:
					output_error[(l_o+6)*f_offset] = 0.5f*prob_scale
						*((float)output[(l_o+6)*f_offset]-0.5f)*((float)output[(l_o+6)*f_offset]-0.5f);
					break;
				case -1:
					output_error[(l_o+6)*f_offset] = 0.0f;
					break;
			}
	
			switch(fit_obj)
			{
				case 1:
					if(max_IoU > min_obj_IoU_lim)
						output_error[(l_o+7)*f_offset] = 0.5f*obj_scale
							*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5)*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5);
					else
						output_error[(l_o+7)*f_offset] = 0.0f;
					break;
				case 0:
					output_error[(l_o+7)*f_offset] = 0.5f*obj_scale
						*((float)output[(l_o+7)*f_offset]-0.5)*((float)output[(l_o+7)*f_offset]-0.5);
					break;
				case -1:
					output_error[(l_o+7)*f_offset] = 0.0f;
					break;
			}
	
			/*Note : mean square error on classes => could be changed to soft max but difficult to balance*/
			switch(fit_class)
			{
				case 1:
					if(max_IoU > min_class_IoU_lim)
						for(k = 0; k < nb_class; k++)
						{
							if(k == (int)target[l_t]-1)
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
									*((float)output[(l_o+8+k)*f_offset]-0.98f)*((float)output[(l_o+8+k)*f_offset]-0.98f);
							else
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
									*((float)output[(l_o+8+k)*f_offset]-0.02f)*((float)output[(l_o+8+k)*f_offset]-0.02f);
						}
					else
						for(k = 0; k < nb_class; k++)
							output_error[(l_o+8+k)*f_offset] = 0.0f;
					break;
				case 0:
					for(k = 0; k < nb_class; k++)
						output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(l_o+8+k)*f_offset]-0.5f)*((float)output[(l_o+8+k)*f_offset]-0.5f);
					break;
				case -1:
					for(k = 0; k < nb_class; k++)
						output_error[(l_o+8+k)*f_offset] = 0.0f;
					break;
			}
	
			/*Linear error of additional parameters*/
			switch(fit_param)
			{
				case 1:
					if(max_IoU > min_param_IoU_lim)
						for(k = 0; k < nb_param; k++)
							output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k])
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));
					else
						for(k = 0; k < nb_param; k++)
							output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
					break;
				case 0:
					for(k = 0; k < nb_param; k++)
						output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale
							*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f)*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));
					break;
				case -1:
					for(k = 0; k < nb_param; k++)
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
					break;
			}
		}
	
		for(j = 0; j < nb_box; j++)
		{
			/*If no match (means no IoU > 0.5) only update Objectness toward 0 */
			/*(here it means error compute)! (no coordinate nor class update)*/
			l_o = j*(8+nb_class+nb_param);
			if(box_locked[j] != 2)
			{
				for(k = 0; k < 6; k++)
					output_error[(l_o+k)*f_offset] = 0.0f;
	
				if(box_locked[j] == 1)
				{
					output_error[(l_o+6)*f_offset] = 0.0f;
					output_error[(l_o+7)*f_offset] = 0.0f;
				}
				else
				{
					switch(fit_prob)
					{
						case 1:
							output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale
								*((float)output[(l_o+6)*f_offset]-0.02f)*((float)output[(l_o+6)*f_offset]-0.02f);
							break;
						case 0:
							output_error[(j*(8+nb_class+nb_param)+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale
								*((float)output[(l_o+6)*f_offset]-0.5f)*((float)output[(l_o+6)*f_offset]-0.5f);
							break;
						case -1:
							output_error[(l_o+6)*f_offset] = 0.0f;
							break;
					}
	
					switch(fit_obj)
					{
						case 1:
							output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale
								*((float)output[(l_o+7)*f_offset]-0.02f)*((float)output[(l_o+7)*f_offset]-0.02f);
							break;
						case 0:
							output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale
								*((float)output[(l_o+7)*f_offset]-0.5f)*((float)output[(l_o+7)*f_offset]-0.5f);
							break;
						case -1:
							output_error[(l_o+7)*f_offset] = 0.0f;
							break;
					}
				}
	
				for(k = 0; k < nb_class; k++)
					output_error[(l_o+8+k)*f_offset] = 0.0f;
	
				for(k = 0; k < nb_param; k++)
					output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
	
			}
		}
	
		free(IoU_table);
		free(dist_prior);
		free(box_in_pix);
		free(box_locked);
	}
}


void YOLO_activation(layer* current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_activation_fct(current->output, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] 
		* current->c_network->batch_size, a_param->biased_dim*current->c_network->length, *a_param, a_param->size);
}

void YOLO_deriv(layer *previous)
{
	printf("Error : YOLO activation can not be used in the middle of the network !n");
	exit(EXIT_FAILURE);
}

void YOLO_deriv_output_error(layer* current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_deriv_error_fct(current->delta_o, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size,
		current->c_network->epoch * current->c_network->train.size);
}

void YOLO_output_error(layer* current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_error_fct((float*)current->c_network->output_error, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
}




//#####################################################

