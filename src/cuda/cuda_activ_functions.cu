
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






#include "../prototypes.h"


static int cu_blocks;

//public are in "prototypes.h"

//private prototypes
void cuda_linear_activation(layer *current);
void cuda_linear_deriv(layer *previous);
void cuda_linear_deriv_output_error(layer* current);
void cuda_linear_output_error(layer* current);

void cuda_ReLU_activation(layer *current);
void cuda_ReLU_deriv(layer *previous);
void cuda_ReLU_deriv_output_error(layer* current);
void cuda_ReLU_output_error(layer* current);

void cuda_logistic_activation(layer *current);
void cuda_logistic_deriv(layer *previous);
void cuda_logistic_deriv_output_error(layer* current);
void cuda_logistic_output_error(layer* current);

void cuda_softmax_activation(layer *current);
void cuda_softmax_deriv(layer *previous);
void cuda_softmax_deriv_output_error(layer *current);
void cuda_softmax_output_error(layer *current);

__global__ void ReLU_activation_kernel_FP32(float *tab, int len, int dim, float leaking_factor);
__global__ void ReLU_activation_kernel_FP16(half *tab, int len, int dim, float leaking_factor);
__global__ void ReLU_deriv_kernel_FP32(float *deriv, float *value, int len, int dim, float leaking_factor, int size);
__global__ void ReLU_deriv_kernel_FP16(half *deriv, half *value, int len, int dim, float leaking_factor, int size);
__global__ void quadratic_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int dim, int len, int size);
__global__ void quadratic_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int dim, int len, int size, half TC_scale_factor);
__global__ void quadratic_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int dim, int len, int size);
__global__ void quadratic_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int dim, int len, int size);
__global__ void logistic_activation_kernel_FP32(float *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_activation_kernel_FP16(half *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_deriv_kernel_FP32(float *deriv, float *value, float beta, int len, int dim, int size);
__global__ void logistic_deriv_kernel_FP16(half *deriv, half *value, float beta, int len, int dim, int size, half scaling);
__global__ void softmax_activation_kernel_FP32(float *tab, int len, int dim, int size);
__global__ void softmax_activation_kernel_FP16(half *tab, int len, int dim, int size);
__global__ void cross_entropy_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int len, int dim, int size);
__global__ void cross_entropy_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int len, int dim, int size, half TC_scale_factor);
__global__ void cross_entropy_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int len, int dim, int size);
__global__ void cross_entropy_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int len, int dim, int size);


void cuda_define_activation(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			current->activation = cuda_ReLU_activation;
			current->deriv_activation = cuda_ReLU_deriv;
			break;
		
		case LOGISTIC:
			current->activation = cuda_logistic_activation;
			current->deriv_activation = cuda_logistic_deriv;
			break;
			
		case SOFTMAX:
			current->activation = cuda_softmax_activation;
			current->deriv_activation = cuda_softmax_deriv;
			break;
			
		case LINEAR:
			default:
			current->activation = cuda_linear_activation;
			current->deriv_activation = cuda_linear_deriv;
			break;
	}

}

void cuda_deriv_output_error(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_deriv_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_deriv_output_error(current);
			break;
	
	}
}

void cuda_output_error_fct(layer* current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_output_error(current);
			break;
	
	}
}

//#####################################################
//         Linear activation related functions
//#####################################################

void cuda_linear_activation(layer *current)
{
	//empty on purpose
}


void cuda_linear_deriv(layer *previous)
{
	//empty on purpose
}


void cuda_linear_deriv_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size);
			break;
		case 1:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			break;
	}
}

void cuda_linear_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output,
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		case 1:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output,
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length,
				param->dim, param->size);
			break;
	}
}


//#####################################################




//#####################################################
//          ReLU activation related functions
//#####################################################


void cuda_ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			ReLU_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output, 
				param->size, param->dim, param->leaking_factor);
			break;
		case 1:
			ReLU_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output, 
				param->size, param->dim, param->leaking_factor);
			break;
	}
}

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
__global__ void ReLU_activation_kernel_FP32(float *tab, int len, int dim, float leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= 0.0f)
			tab[i] *= leaking_factor;
	}
}

__global__ void ReLU_activation_kernel_FP16(half *tab, int len, int dim, float leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= (half)0.0f)
			tab[i] *= leaking_factor;
	}
}


void cuda_ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	switch(previous->c_network->use_cuda_TC)
	{
		default:
		case 0:
			ReLU_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)previous->delta_o, 
				(float*)previous->output, param->size, param->dim, param->leaking_factor, param->size);
			break;
		case 1:
			ReLU_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)previous->delta_o, 
				(half*)previous->output, param->size, param->dim, param->leaking_factor, param->size);
			break;
	}
}

__global__ void ReLU_deriv_kernel_FP32(float *deriv, float *value, int len, int dim, float leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		if(value[i] <= 0.0f)
			deriv[i] *= leaking_factor;
	}
	else
		deriv[i] = 0.0f;
}


__global__ void ReLU_deriv_kernel_FP16(half *deriv, half *value, int len, int dim, float leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		if(value[i] <= (half)0.0f)
			deriv[i] *= leaking_factor;
	}
	else
		deriv[i] = 0.0f;
}

// Should re write a output function to take into account ReLU for Conv output format
void cuda_ReLU_deriv_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim) * current->c_network->length, param->dim, param->size);
			ReLU_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->delta_o, 
				(float*)current->output, param->size, param->dim, param->leaking_factor, param->size);
			break;
		case 1:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim) * current->c_network->length, param->dim, param->size, TC_scale_factor);
			ReLU_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->delta_o,
				(half*)current->output,	param->size, param->dim, param->leaking_factor, param->size);
			break;
	}
}

void cuda_ReLU_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		case 1:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
	}
}

__global__ void quadratic_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos]);
	}
	else
	{
		delta_o[i] = 0.0f;
	}
}

__global__ void quadratic_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, int len, int dim, int size, half TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos])*(half)TC_scale_factor;
	}
	else
	{
		delta_o[i] = (half)0.0f;
	}
}



__global__ void quadratic_output_error_kernel_FP32(float *output_error, float *output, float *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		output_error[pos] = 0.5f*(output[i] - target[pos])*(output[i] - target[pos]);
	}
}


__global__ void quadratic_output_error_kernel_FP16(float *output_error, half *output, half *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		output_error[pos] = (0.5f)*(float)(output[i] - target[pos])*(float)(output[i] - target[pos]);
	}
}


//#####################################################





//#####################################################
//          Logistic activation related funcitons
//#####################################################


void cuda_logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;

	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:	
			logistic_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				param->beta, param->saturation, (param->biased_dim)*current->c_network->length, param->dim, param->size);
			break;
		case 1:
			logistic_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output, 
				param->beta, param->saturation, (param->biased_dim)*current->c_network->length, param->dim, param->size);
			break;
	}
}

__global__ void logistic_activation_kernel_FP32(float *tab, float beta, float saturation, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len)
	{
		i += i / dim;
		tab[i] = -beta*tab[i];
		if(tab[i] > saturation)
			tab[i] = saturation;
		tab[i] = 1.0f/(1.0f + expf(tab[i]));
	}
	else
	{
		tab[i] = 0.0f;
	}
}

__global__ void logistic_activation_kernel_FP16(half *tab, float beta, float saturation, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
		
	half half_one = (half) 1.0f;
	half half_beta = (half) beta;
	half half_saturation = (half) saturation;
	
	//Check if the function works better with compute in FP32 than with FP16
	//It might be the case due to the exponential
	if(i < len)
	{
		i += i / dim;
		tab[i] = -half_beta*tab[i];
		if(tab[i] > half_saturation)
			tab[i] = half_saturation;
		tab[i] = half_one/(half_one + hexp(tab[i]));
	}
	else
	{
		tab[i] = (half)0.0f;
	}
}



void cuda_logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	switch(previous->c_network->use_cuda_TC)
	{
		default:
		case 0:	
			logistic_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)previous->delta_o, 
				(float*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size);
			break;
		case 1:
			logistic_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)previous->delta_o,
				(half*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size, (half)1.0f);
			break;
	}
}



__global__ void logistic_deriv_kernel_FP32(float *deriv, float *value, float beta, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		deriv[i] *= beta*value[i]*(1.0f-value[i]);
	}
	else
		deriv[i] = 0.0f;
}

__global__ void logistic_deriv_kernel_FP16(half *deriv, half *value, float beta, int len, int dim, int size, half scaling)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
		
	half half_beta = (half) beta;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		deriv[i] *= half_beta*value[i]*((half)1.0f-value[i])*scaling;
	}
	else
		deriv[i] = 0.0f;
}


void cuda_logistic_deriv_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:	
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size);
			logistic_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->delta_o,
			 	(float*)current->output, param->beta, (param->biased_dim)*current->c_network->length,
			 	param->dim, param->size);
			break;
		case 1:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			logistic_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->delta_o, 
				(half*)current->output, param->beta, (param->biased_dim)*current->c_network->length,
				param->dim, param->size, (half)TC_scale_factor);
			break;
	}
	
}

void cuda_logistic_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:	
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output,
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		case 1:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
	}		
}

//#####################################################



//#####################################################
//          Soft-Max activation related funcitons
//#####################################################


void cuda_softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:	
			softmax_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				current->c_network->length, param->dim, current->c_network->batch_size);
			break;
		case 1:
			softmax_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output,
				current->c_network->length, param->dim, current->c_network->batch_size);
			break;
	}
}

__global__ void softmax_activation_kernel_FP32(float *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
	//provides a probabilistic output
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;
	float *pos;
	float vmax;
	float normal = 0.0000001f;
	
	if(i >= size)
		return;
		
	if(i < len)
	{
		pos = tab + i*(dim+1);
		
		vmax = pos[0];
		for(j = 1; j < dim; j++)
			if(pos[j] > vmax)
				vmax = pos[j];
		
		for(j = 0; j < dim; j++)
		{	
			pos[j] = expf(pos[j]-vmax);
			normal += pos[j];
		}		
		pos[j] = 0.0f;
		
		for(j = 0; j < dim; j++)
				pos[j] /= normal;
				
		pos[j] = 0.0f;
	}
	else
	{
		pos = tab + i*(dim+1);		
		for(j = 0; j < dim; j++)
			pos[j] = 0.0f;
		pos[j] = 0.0f;
	}
}

__global__ void softmax_activation_kernel_FP16(half *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
	//provides a probabilistic output
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;
	half *pos;
	half vmax;
	float normal = 0.0000001f;
	
	if(i >= size)
		return;
		
	if(i < len)
	{
		pos = tab + i*(dim+1);
		
		vmax = pos[0];
		for(j = 1; j < dim; j++)
			if(pos[j] > vmax)
				vmax = pos[j];
		
		for(j = 0; j < dim; j++)
		{	
			pos[j] = hexp(pos[j]-vmax);
			normal += (float)pos[j];
		}		
		pos[j] = 0.0f;
		
		for(j = 0; j < dim; j++)
			pos[j] /= (half)normal;
				
		pos[j] = 0.0f;
	}
	else
	{
		pos = tab + i*(dim+1);		
		for(j = 0; j < dim; j++)
			pos[j] = 0.0f;
		pos[j] = 0.0f;
	}
}


void cuda_softmax_deriv(layer *previous)
{
	printf("Error : Softmax can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}

void cuda_softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = ((param->biased_dim)*current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			cross_entropy_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, 
				(param->biased_dim) * current->c_network->batch_size);
			break;
		case 1:
			cross_entropy_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, 
				(param->biased_dim)*current->c_network->batch_size, TC_scale_factor);
			break;
	}
}

void cuda_softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = ((param->biased_dim)*current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			cross_entropy_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, (param->dim+1)*current->c_network->length,
				param->dim, (param->biased_dim)*current->c_network->batch_size);
			break;
		case 1:
			cross_entropy_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->dim+1)*current->c_network->length,
				param->dim, (param->biased_dim)*current->c_network->batch_size);
			break;
	}
}


__global__ void cross_entropy_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos]);
	}
	else
	{
		delta_o[i] = 0.0f;
	}
}

__global__ void cross_entropy_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, int len, int dim, int size, half TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos])*(half)TC_scale_factor;
	}
	else
	{
		delta_o[i] = 0.0f;
	}
}

__global__ void cross_entropy_output_error_kernel_FP32(float *output_error, float *output, float *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		if(output[i] > 0.00001)
			output_error[pos] = -target[pos]*logf(output[i]);
		else
			output_error[pos] = -target[pos]*logf(0.00001f);
	}
}

__global__ void cross_entropy_output_error_kernel_FP16(float *output_error, half *output, half *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		if((float)output[i] > 0.00001f)
			output_error[pos] = -(float)target[pos]*logf((float)output[i]);
		else
			output_error[pos] = -(float)target[pos]*logf(0.00001f);
	}
}







//#####################################################










