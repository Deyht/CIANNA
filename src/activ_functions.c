
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

void ReLU_activation(layer *current);
void ReLU_deriv(layer *previous);
void ReLU_deriv_output_error(layer* current);
void ReLU_output_error(layer* current);

void logistic_activation(layer *current);
void logistic_deriv(layer *previous);
void logistic_deriv_output_error(layer* current);
void logistic_output_error(layer* current);

void softmax_activation(layer *current);
void softmax_deriv(layer *previous);
void softmax_deriv_output_error(layer *current);
void softmax_output_error(layer *current);

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
		case RELU_6:
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
		case RELU_6:
			ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_deriv_output_error(current);
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
		case RELU_6:
			ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_output_error(current);
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
			
		case YOLO:
			fprintf(f,"(YOLO)");
			break;
			
		case RELU_6:
			fprintf(f,"(RELU_6)");
			break;
	
		case RELU:
		default:
			fprintf(f,"(RELU)");
			break;
	}	
}

void get_string_activ_param(char* activ, int type)
{
	switch(type)
	{
		case LOGISTIC:
			sprintf(activ,"(LOGI)");
			break;
		
		case SOFTMAX:
			sprintf(activ,"(SMAX)");
			break;
	
		case LINEAR:
			sprintf(activ,"(LIN)");
			break;
			
		case YOLO:
			sprintf(activ,"(YOLO)");
			break;
			
		case RELU_6:
			sprintf(activ,"(RELU_6)");
			break;
	
		case RELU:
		default:
			sprintf(activ,"(RELU)");
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
	else if(strcmp(type, "(YOLO)") == 0)
		return YOLO;
	else if(strcmp(type, "(RELU_6)") == 0)
		return RELU_6;
	else if(strcmp(type, "(RELU)") == 0)
		return RELU;
	else
		return RELU;
}


//#####################################################
//         Linear activation related functions
//#####################################################

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
//          ReLU activation related functions
//#####################################################


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
		if(f_tab[pos] <= 0.0)
			f_tab[pos] *= leaking_factor;
		else if(f_tab[i] > saturation)
			f_tab[i] = saturation + (f_tab[i] - saturation)*leaking_factor;
	}
}


void ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	ReLU_deriv_fct(previous->delta_o, previous->output, param->size, param->dim,
		param->saturation, param->leaking_factor, param->size);
}


//should be adapted for both conv and dense layer if dim is properly defined
void ReLU_deriv_fct(void *deriv, void *value, int len, int dim,  float saturation, float leaking_factor, int size)
{
	int i;
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(i < len && (i+1)%(dim+1) != 0)
		{
			if(f_value[i] <= 0.0)
				f_deriv[i] *= leaking_factor;
			else if(f_deriv[i] > saturation)
				f_deriv[i] *= leaking_factor;
		}
		else
			f_deriv[i] = 0.0;
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
			f_output_error[pos] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
		}
	}
}


//#####################################################





//#####################################################
//          Logistic activation related functions
//#####################################################


void logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	
	logistic_activation_fct(current->output, param->beta, param->saturation, (param->biased_dim)*current->c_network->length,  param->dim, param->size);
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
	for(i = 0; i < size;  i++)
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
//          Soft-Max activation related functions
//#####################################################


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
	float normal = 0.0000001;
	
	#pragma omp parallel for private(j, pos, vmax, normal) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		normal = 0.0000001;
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
			pos[j] = 0.0;
			
			for(j = 0; j < dim; j++)
					pos[j] /= normal;
					
			pos[j] = 0.0;
		}
		else
		{
			pos = (float*)tab + i*(dim+1);		
			for(j = 0; j < dim; j++)
				pos[j] = 0.0;
			pos[j] = 0.0;
		}
	}
}


void softmax_deriv(layer *previous)
{
	printf("Error : Softmax can not be used in the middle of the network !\n");
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
		{
			f_delta_o[i] = 0.0;
		}
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
			if(f_output[i] > 0.00001)
				f_output_error[pos] = -f_target[pos]*log(f_output[i]);
			else
				f_output_error[pos] = -f_target[pos]*log(0.00001);
		}
	}
}


//#####################################################


//#####################################################
//          YOLO activation related functions
//#####################################################


int set_yolo_params(network *net, int nb_box, int IoU_type, float *prior_w, float *prior_h, float *prior_d, float *yolo_noobj_prob_prior, int nb_class, int nb_param, int strict_box_size, float *scale_tab, float **slopes_and_maxes_tab, float *IoU_limits, int *fit_parts)
{
	int i;
	float *temp;
	float **sm;
	char IoU_type_char[40];
	
	net->y_param->IoU_type = IoU_type;
	net->y_param->strict_box_size_association = strict_box_size;
	
	switch(net->y_param->IoU_type)
	{
		//place holder for CPU IoU_fct assignement
		//currently assigned in gpu activation function conversion
		default:
			net->y_param->c_IoU_fct = NULL;
	}
	
	if(scale_tab == NULL)
	{
		scale_tab = (float*) calloc(6, sizeof(float));
		for(i = 0; i < 6; i++)
			scale_tab[i] = 1.0f;
	}
	
	if(slopes_and_maxes_tab == NULL)
	{
		temp = (float*) calloc(6*3, sizeof(float));
		slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
		for(i = 0; i < 6; i++)
			slopes_and_maxes_tab[i] = &temp[i*3];
	
		sm = slopes_and_maxes_tab;
		sm[0][0] = 1.0f; sm[0][1] = 8.0f; sm[0][2] = 0.0f;
		sm[1][0] = 1.0f; sm[1][1] = 1.8f; sm[1][2] = -1.4f;
		sm[2][0] = 1.0f; sm[2][1] = 8.0f; sm[2][2] = 0.0f;
		sm[3][0] = 1.0f; sm[3][1] = 8.0f; sm[3][2] = 0.0f;
		sm[4][0] = 1.0f; sm[4][1] = 8.0f; sm[4][2] = 0.0f;
		sm[5][0] = 1.0f; sm[5][1] = 2.0f; sm[5][2] = -0.2f;
	}

	if(IoU_limits == NULL)
	{
		IoU_limits = (float*) calloc(5,sizeof(float));
		IoU_limits[0] = 0.3f;
		IoU_limits[1] = 0.0f; IoU_limits[2] = 0.3f;
		IoU_limits[3] = 0.3f; IoU_limits[4] = 0.3f;
	}
	
	if(fit_parts == NULL)
	{
		fit_parts = (int*) calloc(5,sizeof(int));
		for(i = 0; i < 5; i++)
			fit_parts[i] = 1;
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
	net->y_param->IoU_limits = IoU_limits;
	net->y_param->fit_parts = fit_parts;
	
	switch(net->y_param->IoU_type)
	{
		case IOU:
			sprintf(IoU_type_char, "Classical IoU");
			break;
		case GIOU:
			sprintf(IoU_type_char, "Generalized GIoU");
			break;
		case DIOU:
			sprintf(IoU_type_char, "Distance DIoU");
			break;
		default:
			break;
	}
	
	printf("\nYOLO layer set with:\n Nboxes = %d\n Ndimensions = %d\n Nclasses = %d\n Nparams = %d\n IoU type = %s\n",
			net->y_param->nb_box, 3, net->y_param->nb_class, net->y_param->nb_param, IoU_type_char);
	printf(" W priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.3f ", net->y_param->prior_w[i]);
	printf("]\n H priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.3f ", net->y_param->prior_h[i]);
	printf("]\n D priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.3f ", net->y_param->prior_d[i]);
	printf("]\n");
	if(net->y_param->strict_box_size_association)
		printf(" Strict box size association is ENABLED\n");
	printf(" No obj. prob. priors\n          = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%7.6f ", net->y_param->noobj_prob_prior[i]);
	printf("]\n");
	printf(" Error scales: Posit.   Size   Proba.  Objct.  Class.  Param.\n          = [");
	for(i = 0; i < 6; i++)
		printf("  %5.3f ",net->y_param->scale_tab[i]);
	printf("]\n IoU lim. = [");
	for(i = 0; i < 5; i++)
		printf("%7.3f ", net->y_param->IoU_limits[i]);
	printf("]\n\n");
	
	return(nb_box*(8+nb_class+nb_param));
}


//#####################################################

