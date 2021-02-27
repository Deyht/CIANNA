
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

void cuda_YOLO_activation(layer *current);
void cuda_YOLO_deriv_output_error(layer *current);
void cuda_YOLO_output_error(layer *current);


__global__ void ReLU_activation_kernel_FP32(float *tab, int len, int dim, float leaking_factor);
__global__ void ReLU_activation_kernel_FP16(half *tab, int len, int dim, float leaking_factor);
__global__ void ReLU_deriv_kernel_FP32(float *deriv, float *value, int len, int dim, float leaking_factor, int size);
__global__ void ReLU_deriv_kernel_FP16(half *deriv, half *value, int len, int dim, float leaking_factor, int size);

__global__ void quadratic_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int dim, int len, int size);
__global__ void quadratic_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int dim, int len, int size, float TC_scale_factor);
__global__ void quadratic_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int dim, int len, int size);
__global__ void quadratic_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int dim, int len, int size);
	
__global__ void logistic_activation_kernel_FP32(float *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_activation_kernel_FP16(half *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_deriv_kernel_FP32(float *deriv, float *value, float beta, int len, int dim, int size);
__global__ void logistic_deriv_kernel_FP16(half *deriv, half *value, float beta, int len, int dim, int size);

__global__ void softmax_activation_kernel_FP32(float *tab, int len, int dim, int size);
__global__ void softmax_activation_kernel_FP16(half *tab, int len, int dim, int size);

__global__ void cross_entropy_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int len, int dim, int size);
__global__ void cross_entropy_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int len, int dim, int size, float TC_scale_factor);
__global__ void cross_entropy_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int len, int dim, int size);
__global__ void cross_entropy_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int len, int dim, int size);
	
__global__ void YOLO_activation_kernel_FP32(float *tab, float beta, float saturation, int flat_offset, int len, int nb_class, int nb_param, int size);
__global__ void YOLO_activation_kernel_FP16(half *tab, float beta, float saturation, int flat_offset, int len, int nb_class, int nb_param, int size);
__global__ void YOLO_deriv_error_kernel_FP32(float *delta_o, float *output, float *target, int beta, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size);
__global__ void YOLO_deriv_error_kernel_FP16(half *delta_o, half *output, half *target, int beta, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size, float TC_scale_factor);
__global__ void YOLO_error_kernel_FP32(float *output_error, float *output, float *target, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size);
__global__ void YOLO_error_kernel_FP16(float *output_error, half *output, half *target, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size);


void cuda_define_activation(layer *current)
{
	void *a_param;
	float *temp_tab;
	
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
			
		case YOLO:
			a_param = (yolo_param*)current->activ_param;
			current->activation = cuda_YOLO_activation;
			current->deriv_activation = cuda_softmax_deriv;
			float *device_prior_w, *device_prior_h;
			
			temp_tab = ((yolo_param*)a_param)->prior_w;
			cudaMalloc(&device_prior_w, 
					((yolo_param*)a_param)->nb_box*sizeof(float));
			cudaMemcpy(device_prior_w, temp_tab,
					((yolo_param*)a_param)->nb_box
					*sizeof(float),cudaMemcpyHostToDevice);
					
			temp_tab = ((yolo_param*)a_param)->prior_h;
			cudaMalloc(&device_prior_h, 
					((yolo_param*)a_param)->nb_box*sizeof(float));
			cudaMemcpy(device_prior_h, temp_tab,
					((yolo_param*)a_param)->nb_box
					*sizeof(float),cudaMemcpyHostToDevice);
					
			((yolo_param*)a_param)->prior_w = device_prior_w;
			((yolo_param*)a_param)->prior_h = device_prior_h;
			
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
			
		case YOLO:
			cuda_YOLO_deriv_output_error(current);
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
			
		case YOLO:
			cuda_YOLO_output_error(current);
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

__global__ void quadratic_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, int len, int dim, int size, float TC_scale_factor)
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
				(half*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size);
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

__global__ void logistic_deriv_kernel_FP16(half *deriv, half *value, float beta, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
		
	half half_beta = (half) beta;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		deriv[i] *= half_beta*value[i]*((half)1.0f-value[i]);
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
				param->dim, param->size);
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

__global__ void cross_entropy_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, int len, int dim, int size, float TC_scale_factor)
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

//#####################################################
//    YOLO final layer activation related functions
//#####################################################


void cuda_YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
        conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (current->c_network->out_size *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			YOLO_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				4.0f/*a_param->beta*/, a_param->saturation, 
				c_param->nb_area_w*c_param->nb_area_h*current->c_network->batch_size, 
				a_param->biased_dim*current->c_network->length,
				a_param->nb_class, a_param->nb_param, a_param->size);
			break;
		case 1:
			YOLO_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output,
				4.0f/*a_param->beta*/, a_param->saturation, 
				c_param->nb_area_w*c_param->nb_area_h*current->c_network->batch_size,
				a_param->biased_dim*current->c_network->length,
				a_param->nb_class, a_param->nb_param, a_param->size);
			break;
	}
}


__global__ void YOLO_activation_kernel_FP32(float *tab, float beta, float saturation, int flat_offset, int len, int nb_class, int nb_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;

	int col,  in_col;

	col = i / flat_offset;
	in_col = col%(5+nb_class+nb_param);

	if(in_col == 2 || in_col == 3 || in_col >= 5+nb_class)
	{
		if(tab[i] > 10.0f)
			tab[i] = 10.0f;
		else if(tab[i] < -6.0f)
			tab[i] = -6.0f;
		return;
	}
	
	tab[i] = -beta*tab[i];
	if(tab[i] > 10.0f)
		tab[i] = 10.0f;
	tab[i] = 1.0f/(1.0f + expf(tab[i]));
	
	
}


__global__ void YOLO_activation_kernel_FP16(half *tab, float beta, float saturation, int flat_offset, int len, int nb_class, int nb_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;

	int col,  in_col;

	col = i / flat_offset;
	in_col = col%(5+nb_class+nb_param);

	if(in_col == 2 || in_col == 3 || in_col >= 5+nb_class)
	{
		if(tab[i] > (half)10.0f)
			tab[i] = (half)10.0f;
		else if(tab[i] < (half)(-6.0f))
			tab[i] = (half)(-6.0f);
		return;
	}
	
	tab[i] = -(half)beta*tab[i];
	if(tab[i] > (half)10.0f)
		tab[i] = (half)10.0f;
	tab[i] = (half)1.0f/((half)1.0f + hexp(tab[i]));
	
	
}


void cuda_YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area_w * c_param->nb_area_h *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			YOLO_deriv_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, 
				(float*)current->c_network->target, 4.0f/*a_param->beta*/, current->c_network->output_dim, 
				c_param->nb_area_w*c_param->nb_area_h, a_param->cell_w, a_param->cell_h, 
				c_param->nb_area_w, c_param->nb_area_h, a_param->nb_box, a_param->nb_class,
				a_param->nb_param, a_param->prior_w, a_param->prior_h,
				c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size);
			
			break;
		case 1:
			YOLO_deriv_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, 
				(half*)current->c_network->target, 4.0f/*a_param->beta*/, current->c_network->output_dim, 
				c_param->nb_area_w*c_param->nb_area_h, a_param->cell_w, a_param->cell_h, 
				c_param->nb_area_w, c_param->nb_area_h, a_param->nb_box, a_param->nb_class,
				a_param->nb_param, a_param->prior_w, a_param->prior_h,
				c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size, TC_scale_factor);
			break;
		
	}
}


void cuda_YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area_w * c_param->nb_area_h *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case 0:
			YOLO_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area_w*c_param->nb_area_h, a_param->cell_w, a_param->cell_h, 
				c_param->nb_area_w, c_param->nb_area_h, a_param->nb_box, a_param->nb_class,
				a_param->nb_param, a_param->prior_w, a_param->prior_h,
				c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size);
			
			break;
		case 1:
			YOLO_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area_w*c_param->nb_area_h, a_param->cell_w, a_param->cell_h, 
				c_param->nb_area_w, c_param->nb_area_h, a_param->nb_box, a_param->nb_class,
				a_param->nb_param, a_param->prior_w, a_param->prior_h,
				c_param->nb_area_w * c_param->nb_area_h * current->c_network->batch_size);
			break;
		
	}
}


__device__ float gpu_IoU(int* output, int* target)
{
	int inter_w, inter_h, inter_2d, uni_2d;
	
	inter_w = max(0, min(output[2], target[2]) - max(output[0], target[0]));
	inter_h = max(0, min(output[3], target[3]) - max(output[1], target[1]));
	inter_2d = inter_w * inter_h;
	uni_2d =  abs(output[2]-output[0])*abs(output[3]-output[1])
			+ abs(target[2]-target[0])*abs(target[3]-target[1])
			- inter_2d;
	return ((float)inter_2d)/(float)uni_2d;
}


// Very Naiv kernel, should be check for preformance
__global__ void YOLO_deriv_error_kernel_FP32(float *delta_o, float *output, float *target, int beta, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= size)
                return;

	int j, k;
	int c_batch;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y;
	int obj_cx, obj_cy;
	int f_offset;
	int *box_in_pix;
	int *c_box_in_pix;
	float obj_in_offset[4];
	
	int obj_surf;
	int dist_surf;
	
	int *box_locked;
	
	float lambda_coord = 2.0, lambda_size = 2.0, lambda_noobj = 0.5, obj_scale = 2.0;
	int out_int[4], targ_int[4];
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (int*) malloc(nb_box*4*sizeof(int));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	
	f_offset = size;

	i = i % flat_output_size;
	cell_x = i % nb_area_w;
	cell_y = i / nb_area_w;
	
	delta_o += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	output  += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;

	
	nb_obj_target = target[0];
	target++;
	
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*4;
		c_box_in_pix[0] = (output[(k*(5+nb_class+nb_param)+0)*f_offset] + cell_x)*cell_w;
		c_box_in_pix[1] = (output[(k*(5+nb_class+nb_param)+1)*f_offset] + cell_y)*cell_h;
		if(output[(k*(5+nb_class+nb_param)+2)*f_offset] > 10.0f)
			c_box_in_pix[2] = prior_w[k]*expf(10.0f);
		else
			c_box_in_pix[2] = prior_w[k]*expf(output[(k*(5+nb_class+nb_param)+2)*f_offset]);
		if(output[(k*(5+nb_class+nb_param)+3)*f_offset] > 10.0f)
			c_box_in_pix[3] = prior_h[k]*expf(10.0f);
		else
			c_box_in_pix[3] = prior_h[k]*expf(output[(k*(5+nb_class+nb_param)+3)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(5+nb_param)] == 0)
			break;
		obj_cx = (int)( (target[j*(5+nb_param)+3] + target[j*(5+nb_param)+1])*0.5f / cell_w);
		obj_cy = (int)( (target[j*(5+nb_param)+4] + target[j*(5+nb_param)+2])*0.5f / cell_h);
		
		if(obj_cx == cell_x && obj_cy == cell_y)
		{
			targ_int[0] = (int)target[j*(5+nb_param)+1]; targ_int[1] = (int)target[j*(5+nb_param)+2];
			targ_int[2] = (int)target[j*(5+nb_param)+3]; targ_int[3] = (int)target[j*(5+nb_param)+4];
		
			resp_box = 0;
			max_IoU = 0.0f;			
			for(k = 0; k < nb_box; k++)
			{
				if(box_locked[k] == 2)
					continue;
			
				c_box_in_pix = box_in_pix+k*4;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[2];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[3];
				out_int[2] = c_box_in_pix[0] + 0.5f*c_box_in_pix[2];
				out_int[3] = c_box_in_pix[1] + 0.5f*c_box_in_pix[3];

				current_IoU = gpu_IoU(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > 0.5f) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
				 	
			if(max_IoU < 0.001)
			{
				obj_surf = abs(targ_int[2] - targ_int[0]) * abs(targ_int[3] - targ_int[1]);
				resp_box = 0;
				dist_surf = abs(obj_surf-(prior_w[0]*prior_h[0]));
				for(k = 1; k < nb_box; k++)
				{
					if(abs(obj_surf-(prior_w[k]*prior_h[k])) < dist_surf)
					{
						dist_surf = abs(obj_surf-(prior_w[k]*prior_h[k]));
						resp_box = k;
					}
				}
			}
			
			if(box_locked[resp_box] == 2)
				continue;
			box_locked[resp_box] = 2;
			
			//Already compute error for the responsible box
			
			obj_in_offset[0] = ((targ_int[2] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;
			obj_in_offset[1] = ((targ_int[3] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;
			obj_in_offset[2] = (targ_int[2] - targ_int[0])/prior_w[resp_box];
			if(obj_in_offset[2] < 0.0001f)
				obj_in_offset[2] = logf(0.0001f);
			else
				obj_in_offset[2] = logf(obj_in_offset[2]);
			obj_in_offset[3] = (targ_int[3] - targ_int[1])/prior_h[resp_box];
			if(obj_in_offset[3] < 0.0001f)
				obj_in_offset[3] = logf(0.0001f);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);

			
			//add an eventual lambda_coord ? unclear if still usefull in YOLO-V2 & V3
			for(k = 0; k < 2; k++)
				delta_o[(resp_box*(5+nb_class+nb_param)+k)*f_offset] = 
					beta*lambda_coord*output[(resp_box*(5+nb_class+nb_param)+k)*f_offset]
					*(1.0f-output[(resp_box*(5+nb_class+nb_param)+k)*f_offset])
					*(output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 2; k++)
				delta_o[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] = lambda_size*
					(output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2]);
			
			
			delta_o[(resp_box*(5+nb_class+nb_param)+4)*f_offset] = 
					beta*obj_scale*output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]
					*(1.0f-output[(resp_box*(5+nb_class+nb_param)+4)*f_offset])
					*(output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-1.0);

			//cross entropy error on classes
			for(k = 0; k < nb_class; k++)
			{
				if(k == (int) target[j*(5+nb_param)]-1)
					delta_o[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = 
						(output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]-1.0f);
				else
					delta_o[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = 
						output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset];
			}
			
			if(max_IoU > 0.3)
			{
				//linear activation of additional parameters
				for(k = 0; k < nb_param; k++)
				{
					delta_o[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 
						(output[(resp_box*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - target[j*(5+nb_param)+5+k]);
				}
			}
			else
			{
				for(k = 0; k < nb_param; k++)
				{
					delta_o[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 0.0f;
				}
			}
			
		}
	}
	
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 4; k++)
				delta_o[(j*(5+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
				delta_o[(j*(5+nb_class+nb_param)+4)*f_offset] = 0.0f;
			else
				delta_o[(j*(5+nb_class+nb_param)+4)*f_offset] = 
					beta*lambda_noobj*output[(j*(5+nb_class+nb_param)+4)*f_offset]
					*(1.0f-output[(j*(5+nb_class+nb_param)+4)*f_offset])
					*(output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f);
						
			for(k = 0; k < nb_class; k++)
				delta_o[(j*(5+nb_class+nb_param)+5+k)*f_offset] = 0.0f;
				
			for(k = 0; k < nb_param; k++)
				delta_o[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}



// Very Naiv kernel, should be check for preformance
__global__ void YOLO_deriv_error_kernel_FP16(half *delta_o, half *output, half *target, int beta, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= size)
                return;

	int j, k;
	int c_batch;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y;
	int obj_cx, obj_cy;
	int f_offset;
	int *box_in_pix;
	int *c_box_in_pix;
	float obj_in_offset[4];
	
	int obj_surf;
	int dist_surf;
	
	int *box_locked;
	
	float lambda_coord = 2.0, lambda_size = 2.0, lambda_noobj = 0.5, obj_scale = 2.0;
	int out_int[4], targ_int[4];
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (int*) malloc(nb_box*4*sizeof(int));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	
	f_offset = size;

	i = i % flat_output_size;
	cell_x = i % nb_area_w;
	cell_y = i / nb_area_w;
	
	delta_o += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	output  += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;

	
	nb_obj_target = target[0];
	target++;
	
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*4;
		c_box_in_pix[0] = ((float)output[(k*(5+nb_class+nb_param)+0)*f_offset] + cell_x)*cell_w;
		c_box_in_pix[1] = ((float)output[(k*(5+nb_class+nb_param)+1)*f_offset] + cell_y)*cell_h;
		if((float)output[(k*(5+nb_class+nb_param)+2)*f_offset] > 10.0f)
			c_box_in_pix[2] = prior_w[k]*expf(10.0f);
		else
			c_box_in_pix[2] = prior_w[k]*expf((float)output[(k*(5+nb_class+nb_param)+2)*f_offset]);
		if((float)output[(k*(5+nb_class+nb_param)+3)*f_offset] > 10.0f)
			c_box_in_pix[3] = prior_h[k]*expf(10.0f);
		else
			c_box_in_pix[3] = prior_h[k]*expf((float)output[(k*(5+nb_class+nb_param)+3)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(5+nb_param)] == 0)
			break;
		obj_cx = (int)( ((float)target[j*(5+nb_param)+3] + (float)target[j*(5+nb_param)+1])*0.5f / cell_w);
		obj_cy = (int)( ((float)target[j*(5+nb_param)+4] + (float)target[j*(5+nb_param)+2])*0.5f / cell_h);
		
		if(obj_cx == cell_x && obj_cy == cell_y)
		{
			targ_int[0] = (int)target[j*(5+nb_param)+1]; targ_int[1] = (int)target[j*(5+nb_param)+2];
			targ_int[2] = (int)target[j*(5+nb_param)+3]; targ_int[3] = (int)target[j*(5+nb_param)+4];
		
			resp_box = 0;
			max_IoU = 0.0f;			
			for(k = 0; k < nb_box; k++)
			{
				if(box_locked[k] == 2)
					continue;
			
				c_box_in_pix = box_in_pix+k*4;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[2];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[3];
				out_int[2] = c_box_in_pix[0] + 0.5f*c_box_in_pix[2];
				out_int[3] = c_box_in_pix[1] + 0.5f*c_box_in_pix[3];

				current_IoU = gpu_IoU(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > 0.5f) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
				 	
			if(max_IoU < 0.001)
			{
				obj_surf = abs(targ_int[2] - targ_int[0]) * abs(targ_int[3] - targ_int[1]);
				resp_box = 0;
				dist_surf = abs(obj_surf-(prior_w[0]*prior_h[0]));
				for(k = 1; k < nb_box; k++)
				{
					if(abs(obj_surf-(prior_w[k]*prior_h[k])) < dist_surf)
					{
						dist_surf = abs(obj_surf-(prior_w[k]*prior_h[k]));
						resp_box = k;
					}
				}
			}
			
			if(box_locked[resp_box] == 2)
				continue;
			box_locked[resp_box] = 2;
			
			//Already compute error for the responsible box
			
			obj_in_offset[0] = ((targ_int[2] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;
			obj_in_offset[1] = ((targ_int[3] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;
			obj_in_offset[2] = (targ_int[2] - targ_int[0])/(float)prior_w[resp_box];
			if(obj_in_offset[2] < 0.0001f)
				obj_in_offset[2] = logf(0.0001f);
			else
				obj_in_offset[2] = logf(obj_in_offset[2]);
			obj_in_offset[3] = (targ_int[3] - targ_int[1])/(float)prior_h[resp_box];
			if(obj_in_offset[3] < 0.0001f)
				obj_in_offset[3] = logf(0.0001f);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);

			
			//add an eventual lambda_coord ? unclear if still usefull in YOLO-V2 & V3
			for(k = 0; k < 2; k++)
				delta_o[(resp_box*(5+nb_class+nb_param)+k)*f_offset] = (half)(
					TC_scale_factor*beta*lambda_coord*(float)output[(resp_box*(5+nb_class+nb_param)+k)*f_offset]
					*(1.0f-(float)output[(resp_box*(5+nb_class+nb_param)+k)*f_offset])
					*((float)output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]));
			for(k = 0; k < 2; k++)
				delta_o[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] = (half) (TC_scale_factor*lambda_size*
					((float)output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2]));
			
			if(max_IoU > 0.3f)
			{
				delta_o[(resp_box*(5+nb_class+nb_param)+4)*f_offset] = (half)(
					TC_scale_factor*beta*obj_scale*(float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]
					*(1.0f-(float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset])
					*((float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-max_IoU));
			}
			else
			{
				delta_o[(resp_box*(5+nb_class+nb_param)+4)*f_offset] = (half)(
                                        TC_scale_factor*beta*obj_scale*(float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]
                                        *(1.0f-(float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset])
                                        *((float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-0.3f));
			}
			//cross entropy error on classes
			for(k = 0; k < nb_class; k++)
			{
				if(k == (int) target[j*(5+nb_param)]-1)
					delta_o[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = 
						(half) ((float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]-1.0f);
				else
					delta_o[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = 
						(half) ((float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]);
			}
			
			//linear activation of additional parameters
			if(max_IoU > 0.3f)
			{
				for(k = 0; k < nb_param; k++)
				{
					delta_o[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 
						(half) (TC_scale_factor*2.0f*((float)output[(resp_box*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - (float)target[j*(5+nb_param)+5+k]));
				}
			}
			else
			{
				for(k = 0; k < nb_param; k++)
				{
					delta_o[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = (half) 0.0f;
				}
			}
		}
	}
	
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 4; k++)
				delta_o[(j*(5+nb_class+nb_param)+k)*f_offset] = (half) 0.0f;
				
			if(box_locked[j] == 1)
			{
				delta_o[(j*(5+nb_class+nb_param)+4)*f_offset] = (half) 0.0f;
				
				for(k = 0; k < nb_param; k++)
                                	delta_o[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = (half) 0.0f;
			}
			else
			{
				delta_o[(j*(5+nb_class+nb_param)+4)*f_offset] = (half)(
					TC_scale_factor*beta*lambda_noobj*(float)output[(j*(5+nb_class+nb_param)+4)*f_offset]
					*(1.0f-(float)output[(j*(5+nb_class+nb_param)+4)*f_offset])
					*((float)output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f));

				for(k = 0; k < nb_param; k++)
	                                delta_o[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = (half)
						(TC_scale_factor*lambda_noobj*((float)output[(j*(5+nb_class+nb_param)
                                                +5+nb_class+k)*f_offset] - 0.0f));
			}
			for(k = 0; k < nb_class; k++)
				delta_o[(j*(5+nb_class+nb_param)+5+k)*f_offset] = (half) 0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}



// Very Naiv kernel, should be check for preformance
__global__ void YOLO_error_kernel_FP32(float *output_error, float *output, float *target, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h,  int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	int j, k;
	int c_batch;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y;
	int obj_cx, obj_cy;
	int f_offset;
	int *box_in_pix;
	int *c_box_in_pix;
	float obj_in_offset[4];
	
	int obj_surf;
	int dist_surf;
	
	int *box_locked;
	
	float lambda_coord = 2.0, lambda_size = 2.0, lambda_noobj = 0.5, obj_scale = 2.0;
	int out_int[4], targ_int[4];
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (int*) malloc(nb_box*4*sizeof(int));
	
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	
	f_offset = size;

	i = i % flat_output_size;
	cell_x = i % nb_area_w;
	cell_y = i / nb_area_w;
	
	output_error += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	output += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	
	nb_obj_target = target[0];
	target++;
	
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*4;
		c_box_in_pix[0] = (output[(k*(5+nb_class+nb_param)+0)*f_offset] + cell_x)*cell_w;
		c_box_in_pix[1] = (output[(k*(5+nb_class+nb_param)+1)*f_offset] + cell_y)*cell_h;
		if(output[(k*(5+nb_class+nb_param)+2)*f_offset] > 10.0f)
			c_box_in_pix[2]  = prior_w[k]*expf(10.0f);
		else
			c_box_in_pix[2] = prior_w[k]*expf(output[(k*(5+nb_class+nb_param)+2)*f_offset]);
		if(output[(k*(5+nb_class+nb_param)+3)*f_offset] > 10.0f)
			c_box_in_pix[3] = prior_h[k]*expf(10.0f);
		else
			c_box_in_pix[3] = prior_h[k]*expf(output[(k*(5+nb_class+nb_param)+3)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(5+nb_param)] == 0)
			break;
		obj_cx = (int)( (target[j*(5+nb_param)+3] + target[j*(5+nb_param)+1])*0.5f / cell_w);
		obj_cy = (int)( (target[j*(5+nb_param)+4] + target[j*(5+nb_param)+2])*0.5f / cell_h);
		
		if(obj_cx == cell_x && obj_cy == cell_y)
		{
			resp_box = 0;
			max_IoU = 0.0f;			
			for(k = 0; k < nb_box; k++)
			{
				if(box_locked[k] == 2)
					continue;
				targ_int[0] = (int)target[j*(5+nb_param)+1]; targ_int[1] = (int)target[j*(5+nb_param)+2];
				targ_int[2] = (int)target[j*(5+nb_param)+3]; targ_int[3] = (int)target[j*(5+nb_param)+4];
			
				c_box_in_pix = box_in_pix+k*4;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[2];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[3];
				out_int[2] = c_box_in_pix[0] + 0.5f*c_box_in_pix[2];
				out_int[3] = c_box_in_pix[1] + 0.5f*c_box_in_pix[3];

				current_IoU = gpu_IoU(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > 0.5f) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;

			}
			
			if(max_IoU < 0.001)
			{
				obj_surf = abs(targ_int[2] - targ_int[0]) * abs(targ_int[3] - targ_int[1]);
				resp_box = 0;
				dist_surf = abs(obj_surf-(prior_w[0]*prior_h[0]));
				for(k = 1; k < nb_box; k++)
				{
					if(abs(obj_surf-(prior_w[k]*prior_h[k])) < dist_surf)
					{
						dist_surf = abs(obj_surf-(prior_w[k]*prior_h[k]));
						resp_box = k;
					}
				}
			}
			
			if(box_locked[resp_box] == 2)
                                continue;
			box_locked[resp_box] = 2;
		
			//Already compute error for the responsible box
			
			obj_in_offset[0] = ((targ_int[2] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;
			obj_in_offset[1] = ((targ_int[3] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;
			obj_in_offset[2] = (targ_int[2] - targ_int[0])/prior_w[resp_box];
			if(obj_in_offset[2] < 0.0001f)
				obj_in_offset[2] = logf(0.0001f);
			else
				obj_in_offset[2] = logf(obj_in_offset[2]);
			obj_in_offset[3] = (targ_int[3] - targ_int[1])/prior_h[resp_box];
			if(obj_in_offset[3] < 0.0001f)
				obj_in_offset[3] = logf(0.0001f);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);

			for(k = 0; k < 2; k++)
				output_error[(resp_box*(5+nb_class+nb_param)+k)*f_offset] = 
					0.5f*lambda_coord*(output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k])
					*(output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 2; k++)
				output_error[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] = 
					0.5f*lambda_size*(output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2])
					*(output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2]);
			
			output_error[(resp_box*(5+nb_class+nb_param)+4)*f_offset] = 
					0.5f*obj_scale*(output[(resp_box*(5+nb_class)+nb_param+4)*f_offset]-1.0)
					*(output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-1.0);
			
			//cross entropy error on classes
			for(k = 0; k < nb_class; k++)
			{
				if(k == (int)target[j*(5+nb_param)]-1)
				{
					if(output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] < 0.0001f)
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(0.0001f);
					else
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = 
							-logf(output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]);
				}
				else
				{
					if(output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] > 0.999f)
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(0.001f);
					else
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(1.0f - output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]);
				}
			}
			
			//linear error of additional parameters
			for(k = 0; k < nb_param; k++)
			{
				output_error[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 
					(0.5f*(output[(resp_box*(5+nb_class+nb_param)
					+5+nb_class+k)*f_offset] - target[j*(5+nb_param)+5+k])
					*(output[(resp_box*(5+nb_class+nb_param)
					+5+nb_class+k)*f_offset] - target[j*(5+nb_param)+5+k]));
			}
			
		}
	}
	
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 4; k++)
				output_error[(j*(5+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
				output_error[(j*(5+nb_class+nb_param)+4)*f_offset] = 0.0f;
			else
				output_error[(j*(5+nb_class+nb_param)+4)*f_offset] = 
					0.5f*lambda_noobj*(output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f)
					*(output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f);
						
			for(k = 0; k < nb_class; k++)
				output_error[(j*(5+nb_class+nb_param)+5+k)*f_offset] =  0.0f;
				
			for(k = 0; k < nb_param; k++)
				output_error[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] =  0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);

}



// Very Naiv kernel, should be check for preformance
__global__ void YOLO_error_kernel_FP16(float *output_error, half *output, half *target, int flat_target_size, int flat_output_size, int cell_w, int cell_h, int nb_area_w, int nb_area_h, const int nb_box, int nb_class, int nb_param, float *prior_w, float *prior_h, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	int j, k;
	int c_batch;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y;
	int obj_cx, obj_cy;
	int f_offset;
	int *box_in_pix;
	int *c_box_in_pix;
	float obj_in_offset[4];
	
	int obj_surf;
	int dist_surf;
	
	int *box_locked;
	
	float lambda_coord = 2.0, lambda_size = 2.0, lambda_noobj = 0.5, obj_scale = 2.0;
	int out_int[4], targ_int[4];
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (int*) malloc(nb_box*4*sizeof(int));
	
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	
	f_offset = size;

	i = i % flat_output_size;
	cell_x = i % nb_area_w;
	cell_y = i / nb_area_w;
	
	output_error += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	output += (nb_area_w*nb_area_h) * c_batch + cell_y*nb_area_w + cell_x;
	
	nb_obj_target = target[0];
	target++;
	
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*4;
		c_box_in_pix[0] = ((float)output[(k*(5+nb_class+nb_param)+0)*f_offset] + cell_x)*cell_w;
		c_box_in_pix[1] = ((float)output[(k*(5+nb_class+nb_param)+1)*f_offset] + cell_y)*cell_h;
		if((float)output[(k*(5+nb_class+nb_param)+2)*f_offset] > 10.0f)
			c_box_in_pix[2]  = prior_w[k]*expf(10.0f);
		else
			c_box_in_pix[2] = prior_w[k]*expf(output[(k*(5+nb_class+nb_param)+2)*f_offset]);
		if((float)output[(k*(5+nb_class+nb_param)+3)*f_offset] > 10.0f)
			c_box_in_pix[3] = prior_h[k]*expf(10.0f);
		else
			c_box_in_pix[3] = prior_h[k]*expf(output[(k*(5+nb_class+nb_param)+3)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(5+nb_param)] == 0)
			break;
		obj_cx = (int)( ((float)target[j*(5+nb_param)+3] + (float)target[j*(5+nb_param)+1])*0.5f / cell_w);
		obj_cy = (int)( ((float)target[j*(5+nb_param)+4] + (float)target[j*(5+nb_param)+2])*0.5f / cell_h);
		
		if(obj_cx == cell_x && obj_cy == cell_y)
		{
			resp_box = 0;
			max_IoU = 0.0f;			
			for(k = 0; k < nb_box; k++)
			{
				if(box_locked[k] == 2)
					continue;
				targ_int[0] = (int)target[j*(5+nb_param)+1]; targ_int[1] = (int)target[j*(5+nb_param)+2];
				targ_int[2] = (int)target[j*(5+nb_param)+3]; targ_int[3] = (int)target[j*(5+nb_param)+4];
			
				c_box_in_pix = box_in_pix+k*4;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[2];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[3];
				out_int[2] = c_box_in_pix[0] + 0.5f*c_box_in_pix[2];
				out_int[3] = c_box_in_pix[1] + 0.5f*c_box_in_pix[3];

				current_IoU = gpu_IoU(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > 0.5f) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;

			}
			
			if(max_IoU < 0.001)
			{
				obj_surf = abs(targ_int[2] - targ_int[0]) * abs(targ_int[3] - targ_int[1]);
				resp_box = 0;
				dist_surf = abs(obj_surf-(prior_w[0]*prior_h[0]));
				for(k = 1; k < nb_box; k++)
				{
					if(abs(obj_surf-(prior_w[k]*prior_h[k])) < dist_surf)
					{
						dist_surf = abs(obj_surf-(prior_w[k]*prior_h[k]));
						resp_box = k;
					}
				}
			}
			
			if(box_locked[resp_box] == 2)
                                continue;
			box_locked[resp_box] = 2;
		
			//Already compute error for the responsible box
			
			obj_in_offset[0] = ((targ_int[2] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;
			obj_in_offset[1] = ((targ_int[3] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;
			obj_in_offset[2] = (targ_int[2] - targ_int[0])/prior_w[resp_box];
			if(obj_in_offset[2] < 0.0001f)
				obj_in_offset[2] = logf(0.0001f);
			else
				obj_in_offset[2] = logf(obj_in_offset[2]);
			obj_in_offset[3] = (targ_int[3] - targ_int[1])/prior_h[resp_box];
			if(obj_in_offset[3] < 0.0001f)
				obj_in_offset[3] = logf(0.0001f);
			else
				obj_in_offset[3] = logf(obj_in_offset[3]);

			for(k = 0; k < 2; k++)
				output_error[(resp_box*(5+nb_class+nb_param)+k)*f_offset] =
					0.5f*lambda_coord*((float)output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k])
					*((float)output[(resp_box*(5+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 2; k++)
				output_error[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] =
					0.5f*lambda_size*((float)output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2])
					*((float)output[(resp_box*(5+nb_class+nb_param)+k+2)*f_offset] - obj_in_offset[k+2]);
			
			if(max_IoU > 0.3f)
			{
				output_error[(resp_box*(5+nb_class+nb_param)+4)*f_offset] =
					0.5f*obj_scale*((float)output[(resp_box*(5+nb_class)+nb_param+4)*f_offset]-max_IoU)
					*((float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-max_IoU);
			}
			else
			{
				output_error[(resp_box*(5+nb_class+nb_param)+4)*f_offset] =
					0.5f*obj_scale*((float)output[(resp_box*(5+nb_class)+nb_param+4)*f_offset]-0.3f)
					*((float)output[(resp_box*(5+nb_class+nb_param)+4)*f_offset]-0.3f);
			}
			//cross entropy error on classes
			for(k = 0; k < nb_class; k++)
			{
				if(k == (int)target[j*(5+nb_param)]-1)
				{
					if((float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] < 0.0001f)
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(0.0001f);
					else
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] =
							-logf((float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]);
				}
				else
				{
					if((float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] > 0.999f)
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(0.001f);
					else
						output_error[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset] = -logf(1.0f - (float)output[(resp_box*(5+nb_class+nb_param)+5+k)*f_offset]);
				}
			}
			
			//linear error of additional parameters
			if(max_IoU > 0.3f)
			{
				for(k = 0; k < nb_param; k++)
				{
					output_error[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 
						(0.5f*2.0f*((float)output[(resp_box*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - (float) target[j*(5+nb_param)+5+k])
						*((float)output[(resp_box*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - (float) target[j*(5+nb_param)+5+k]));
				}
			}
			else
			{
				for(k = 0; k < nb_param; k++)
				{
					output_error[(resp_box*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 0.0f;
				}
			}

		}
	}
	
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 4; k++)
				output_error[(j*(5+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
			{
				output_error[(j*(5+nb_class+nb_param)+4)*f_offset] = 0.0f;
			
				for(k = 0; k < nb_param; k++)
	                                output_error[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] = 0.0f;
			}
			else
			{
				output_error[(j*(5+nb_class+nb_param)+4)*f_offset] =
					0.5f*lambda_noobj*((float)output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f)
					*((float)output[(j*(5+nb_class+nb_param)+4)*f_offset]-0.0f);
			
				for(k = 0; k < nb_param; k++)
					output_error[(j*(5+nb_class+nb_param)+5+nb_class+k)*f_offset] =
						0.5f*lambda_noobj*((float)output[(j*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - 0.0f)
						*((float)output[(j*(5+nb_class+nb_param)
						+5+nb_class+k)*f_offset] - 0.0f);

			}
			for(k = 0; k < nb_class; k++)
				output_error[(j*(5+nb_class+nb_param)+5+k)*f_offset] = 0.0f;
				
		}
	}
	
	free(box_in_pix);
	free(box_locked);

}



//#####################################################








