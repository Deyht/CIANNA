	
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


__global__ void ReLU_activation_kernel_FP32(float *tab, int len, int dim, float saturation, float leaking_factor);
__global__ void ReLU_activation_kernel_FP16(half *tab, int len, int dim, float saturation, float leaking_factor);
__global__ void ReLU_activation_kernel_BF16(nv_bfloat16 *tab, int len, int dim, float saturation, float leaking_factor);
__global__ void ReLU_deriv_kernel_FP32(float *deriv, float *value, int len, int dim, float saturation, float leaking_factor, int size);
__global__ void ReLU_deriv_kernel_FP16(half *deriv, half *value, int len, int dim, float saturation, float leaking_factor, int size);
__global__ void ReLU_deriv_kernel_BF16(nv_bfloat16 *deriv, nv_bfloat16 *value, int len, int dim, float saturation, float leaking_factor, int size);

__global__ void quadratic_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int dim, int len, int size, float TC_scale_factor);
__global__ void quadratic_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int dim, int len, int size, float TC_scale_factor);
__global__ void quadratic_deriv_output_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, 
	int dim, int len, int size, float TC_scale_factor);
__global__ void quadratic_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int dim, int len, int size);
__global__ void quadratic_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int dim, int len, int size);
__global__ void quadratic_output_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, 
	int dim, int len, int size);
	
__global__ void logistic_activation_kernel_FP32(float *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_activation_kernel_FP16(half *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_activation_kernel_BF16(nv_bfloat16 *tab, float beta, float saturation, int dim, int len, int size);
__global__ void logistic_deriv_kernel_FP32(float *deriv, float *value, float beta, int len, int dim, int size);
__global__ void logistic_deriv_kernel_FP16(half *deriv, half *value, float beta, int len, int dim, int size);
__global__ void logistic_deriv_kernel_BF16(nv_bfloat16 *deriv, nv_bfloat16 *value, float beta, int len, int dim, int size);

__global__ void softmax_activation_kernel_FP32(float *tab, int len, int dim, int size);
__global__ void softmax_activation_kernel_FP16(half *tab, int len, int dim, int size);
__global__ void softmax_activation_kernel_BF16(nv_bfloat16 *tab, int len, int dim, int size);

__global__ void cross_entropy_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, 
	int len, int dim, int size, float TC_scale_factor);
__global__ void cross_entropy_deriv_output_error_kernel_FP16(half *delta_o, half *output, half *target, 
	int len, int dim, int size, float TC_scale_factor);
__global__ void cross_entropy_deriv_output_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, 
	int len, int dim, int size, float TC_scale_factor);
__global__ void cross_entropy_output_error_kernel_FP32(float *output_error, float *output, float *target, 
	int len, int dim, int size);
__global__ void cross_entropy_output_error_kernel_FP16(float *output_error, half *output, half *target, 
	int len, int dim, int size);
__global__ void cross_entropy_output_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, 
	int len, int dim, int size);

__device__ float gpu_IoU_fct(float* output, float* target);
__device__ float gpu_GIoU_fct(float* output, float* target);
__device__ float gpu_DIoU_fct(float* output, float* target);
typedef float(*pointFunction_gpu_IoU)(float*, float*); 
__device__ pointFunction_gpu_IoU device_gpu_IoU_fct  = gpu_IoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_GIoU_fct = gpu_GIoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_DIoU_fct = gpu_DIoU_fct;

__global__ void YOLO_activation_kernel_FP32(float *tab, int flat_offset, int len, yolo_param y_param, int size);
__global__ void YOLO_activation_kernel_FP16(half *tab, int flat_offset, int len, yolo_param y_param, int size);
__global__ void YOLO_activation_kernel_BF16(nv_bfloat16 *tab, int flat_offset, int len, yolo_param y_param, int size);
__global__ void YOLO_deriv_error_kernel_FP32(float *delta_o, float *output, float *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor);
__global__ void YOLO_deriv_error_kernel_FP16(half *delta_o, half *output, half *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor);
__global__ void YOLO_deriv_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor);
__global__ void YOLO_error_kernel_FP32(float *output_error, float *output, float *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size);
__global__ void YOLO_error_kernel_FP16(float *output_error, half *output, half *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size);
__global__ void YOLO_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size);



void cuda_define_activation(layer *current)
{
	void *a_param;
	float *temp_tab, *temp_tab2, **temp_tab3;
	int *temp_int;
	
	switch(current->activation_type)
	{
		case RELU:
		case RELU_6:
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
			float *device_prior_w, *device_prior_h, *device_prior_d, *device_noobj_prob_prior;
			float *device_scale_tab, **device_slopes_and_maxes_tab, *device_param_ind_scale, *device_IoU_limits, *device_IoU_monitor;
			int *device_fit_parts;
			
			switch(((yolo_param*)a_param)->IoU_type)
			{
				case IOU:
					cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_IoU_fct, sizeof(pointFunction_gpu_IoU));
					break;
					
				case GIOU:
					cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_GIoU_fct, sizeof(pointFunction_gpu_IoU));
					break;
					
				default:
				case DIOU:
					cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_DIoU_fct, sizeof(pointFunction_gpu_IoU));
					break;
			}
			
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
					
			temp_tab = ((yolo_param*)a_param)->prior_d;
			cudaMalloc(&device_prior_d, 
					((yolo_param*)a_param)->nb_box*sizeof(float));
			cudaMemcpy(device_prior_d, temp_tab,
					((yolo_param*)a_param)->nb_box
					*sizeof(float),cudaMemcpyHostToDevice);
			
			temp_tab = ((yolo_param*)a_param)->noobj_prob_prior;
			cudaMalloc(&device_noobj_prob_prior, 
					((yolo_param*)a_param)->nb_box*sizeof(float));
			cudaMemcpy(device_noobj_prob_prior, temp_tab,
					((yolo_param*)a_param)->nb_box
					*sizeof(float),cudaMemcpyHostToDevice);
					
			temp_tab = ((yolo_param*)a_param)->scale_tab;
			cudaMalloc(&device_scale_tab, 6*sizeof(float));
			cudaMemcpy(device_scale_tab, temp_tab,
					6*sizeof(float),cudaMemcpyHostToDevice);
					
			temp_tab = ((yolo_param*)a_param)->slopes_and_maxes_tab[0];
			cudaMalloc(&temp_tab2, 6*3*sizeof(float));
			cudaMemcpy(temp_tab2, temp_tab, 6*3*sizeof(float),cudaMemcpyHostToDevice);
			for(int i = 0; i < 6; i++)
				((yolo_param*)a_param)->slopes_and_maxes_tab[i] = &temp_tab2[i*3];
			temp_tab3 = ((yolo_param*)a_param)->slopes_and_maxes_tab;
			cudaMalloc(&device_slopes_and_maxes_tab, 6*sizeof(float*));
			cudaMemcpy(device_slopes_and_maxes_tab, temp_tab3,
					6*sizeof(float*),cudaMemcpyHostToDevice);
			
			temp_tab = ((yolo_param*)a_param)->param_ind_scale;
			cudaMalloc(&device_param_ind_scale, ((yolo_param*)a_param)->nb_param*sizeof(float));
			cudaMemcpy(device_param_ind_scale, temp_tab,
					((yolo_param*)a_param)->nb_param*sizeof(float),cudaMemcpyHostToDevice);
			
			temp_tab = ((yolo_param*)a_param)->IoU_limits;
			cudaMalloc(&device_IoU_limits, 5*sizeof(float));
			cudaMemcpy(device_IoU_limits, temp_tab,
					5*sizeof(float),cudaMemcpyHostToDevice);
					
			temp_int = ((yolo_param*)a_param)->fit_parts;
			cudaMalloc(&device_fit_parts, 5*sizeof(int));
			cudaMemcpy(device_fit_parts, temp_int,
					5*sizeof(int),cudaMemcpyHostToDevice);
			
			temp_tab = ((yolo_param*)a_param)->IoU_monitor;
			cudaMalloc(&device_IoU_monitor, 2 * ((yolo_param*)a_param)->nb_box
				* ((conv_param*)current->param)->nb_area[0]
				* ((conv_param*)current->param)->nb_area[1]
				* ((conv_param*)current->param)->nb_area[2]
				* current->c_network->batch_size * sizeof(float));
			cudaMemcpy(device_IoU_monitor, temp_tab,
				2 * ((yolo_param*)a_param)->nb_box
				* ((conv_param*)current->param)->nb_area[0]
				* ((conv_param*)current->param)->nb_area[1]
				* ((conv_param*)current->param)->nb_area[2]
				* current->c_network->batch_size * sizeof(float), cudaMemcpyHostToDevice);
			
			((yolo_param*)a_param)->prior_w = device_prior_w;
			((yolo_param*)a_param)->prior_h = device_prior_h;
			((yolo_param*)a_param)->prior_d = device_prior_d;
			((yolo_param*)a_param)->noobj_prob_prior = device_noobj_prob_prior;
			((yolo_param*)a_param)->scale_tab = device_scale_tab;
			((yolo_param*)a_param)->slopes_and_maxes_tab = device_slopes_and_maxes_tab;
			((yolo_param*)a_param)->param_ind_scale = device_param_ind_scale;
			((yolo_param*)a_param)->IoU_limits = device_IoU_limits;
			((yolo_param*)a_param)->fit_parts = device_fit_parts;
			((yolo_param*)a_param)->IoU_monitor = device_IoU_monitor;
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
		case RELU_6:
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
		case RELU_6:
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			break;
		
		case BF16C_FP32A:
			quadratic_deriv_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(nv_bfloat16*)current->delta_o, (nv_bfloat16*)current->output, (nv_bfloat16*)current->c_network->target,
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output,
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output,
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length,
				param->dim, param->size);
			break;
			
		case BF16C_FP32A:
			quadratic_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (nv_bfloat16*)current->output,
				(nv_bfloat16*)current->c_network->target, (param->biased_dim)*current->c_network->length,
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			ReLU_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output, 
				param->size, param->dim, param->saturation, param->leaking_factor);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			ReLU_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output, 
				param->size, param->dim, param->saturation, param->leaking_factor);
			break;
		
		case BF16C_FP32A:
			ReLU_activation_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->output, 
				param->size, param->dim, param->saturation, param->leaking_factor);
			break;
	}
}

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
__global__ void ReLU_activation_kernel_FP32(float *tab, int len, int dim, float saturation, float leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= 0.0f)
			tab[i] *= leaking_factor;
		else if(tab[i] > saturation)
			tab[i] = saturation + (tab[i] - saturation)*leaking_factor;
	}
}

__global__ void ReLU_activation_kernel_FP16(half *tab, int len, int dim, float saturation, float leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= (half) 0.0f)
			tab[i] *= leaking_factor;
		else if(tab[i] > (half) saturation)
			tab[i] = (half) saturation + (tab[i] - (half)saturation)*((half)leaking_factor);
	}
}

__global__ void ReLU_activation_kernel_BF16(nv_bfloat16 *tab, int len, int dim, float saturation, float leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= (nv_bfloat16) 0.0f)
			tab[i] *= leaking_factor;
		else if(tab[i] > (nv_bfloat16) saturation)
			tab[i] = (nv_bfloat16) saturation + (tab[i] - (nv_bfloat16)saturation)*((nv_bfloat16)leaking_factor);
	}
}


void cuda_ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	switch(previous->c_network->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			ReLU_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)previous->delta_o, 
				(float*)previous->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			ReLU_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)previous->delta_o, 
				(half*)previous->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
			break;
		
		case BF16C_FP32A:
			ReLU_deriv_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)previous->delta_o, 
				(nv_bfloat16*)previous->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
			break;
	}
}

__global__ void ReLU_deriv_kernel_FP32(float *deriv, float *value, int len, int dim, float saturation, float leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		if(value[i] <= 0.0f)
			deriv[i] *= leaking_factor;
		if(value[i] > saturation)
			deriv[i] *= leaking_factor;
	}
	else
		deriv[i] = 0.0f;
}


__global__ void ReLU_deriv_kernel_FP16(half *deriv, half *value, int len, int dim, float saturation, float leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		if(value[i] <= (half) 0.0f)
			deriv[i] *= leaking_factor;
		if(value[i] > (half) saturation)
			deriv[i] *= leaking_factor;
	}
	else
		deriv[i] = 0.0f;
}


__global__ void ReLU_deriv_kernel_BF16(nv_bfloat16 *deriv, nv_bfloat16 *value, int len, int dim, float saturation, float leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		if(value[i] <= (nv_bfloat16) 0.0f)
			deriv[i] *= leaking_factor;
		if(value[i] > (nv_bfloat16) saturation)
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim) * current->c_network->length, param->dim, param->size, TC_scale_factor);
			ReLU_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->delta_o, 
				(float*)current->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim) * current->c_network->length, param->dim, param->size, TC_scale_factor);
			ReLU_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->delta_o,
				(half*)current->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
			break;
		
		case BF16C_FP32A:
			quadratic_deriv_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(nv_bfloat16*)current->delta_o, (nv_bfloat16*)current->output, (nv_bfloat16*)current->c_network->target,
				(param->biased_dim) * current->c_network->length, param->dim, param->size, TC_scale_factor);
			ReLU_deriv_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->delta_o,
				(nv_bfloat16*)current->output, param->size, param->dim, param->saturation, param->leaking_factor, param->size);
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		
		case BF16C_FP32A:
			quadratic_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (nv_bfloat16*)current->output, 
				(nv_bfloat16*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
	}
}

__global__ void quadratic_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, int len, int dim, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos]) / TC_scale_factor;
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

__global__ void quadratic_deriv_output_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, int len, int dim, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos])*(nv_bfloat16)TC_scale_factor;
	}
	else
	{
		delta_o[i] = (nv_bfloat16)0.0f;
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


__global__ void quadratic_output_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, int len, int dim, int size)
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			logistic_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				param->beta, param->saturation, (param->biased_dim)*current->c_network->length, param->dim, param->size);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			logistic_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output, 
				param->beta, param->saturation, (param->biased_dim)*current->c_network->length, param->dim, param->size);
			break;
		
		case BF16C_FP32A:
			logistic_activation_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->output, 
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


__global__ void logistic_activation_kernel_BF16(nv_bfloat16 *tab, float beta, float saturation, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
		
	nv_bfloat16 bf_one = (nv_bfloat16) 1.0f;
	nv_bfloat16 bf_beta = (nv_bfloat16) beta;
	nv_bfloat16 bf_saturation = (nv_bfloat16) saturation;
	
	//Check if the function works better with compute in FP32 than with FP16
	//It might be the case due to the exponential
	if(i < len)
	{
		i += i / dim;
		tab[i] = -bf_beta*tab[i];
		if(tab[i] > bf_saturation)
			tab[i] = bf_saturation;
		tab[i] = bf_one/(bf_one + hexp(tab[i]));
	}
	else
	{
		tab[i] = (nv_bfloat16)0.0f;
	}
}


void cuda_logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	switch(previous->c_network->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			logistic_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)previous->delta_o, 
				(float*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			logistic_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)previous->delta_o,
				(half*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size);
			break;
		
		case BF16C_FP32A:
			logistic_deriv_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)previous->delta_o,
				(nv_bfloat16*)previous->output, param->beta, (param->biased_dim)*previous->c_network->length, param->dim, param->size);
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


__global__ void logistic_deriv_kernel_BF16(nv_bfloat16 *deriv, nv_bfloat16 *value, float beta, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
		
	nv_bfloat16 bf_beta = (nv_bfloat16) beta;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		deriv[i] *= bf_beta*value[i]*((nv_bfloat16)1.0f-value[i]);
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			logistic_deriv_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->delta_o,
			 	(float*)current->output, param->beta, (param->biased_dim)*current->c_network->length,
			 	param->dim, param->size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			logistic_deriv_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->delta_o, 
				(half*)current->output, param->beta, (param->biased_dim)*current->c_network->length,
				param->dim, param->size);
			break;
		
		case BF16C_FP32A:
			quadratic_deriv_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(nv_bfloat16*)current->delta_o, (nv_bfloat16*)current->output, (nv_bfloat16*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, param->size, TC_scale_factor);
			logistic_deriv_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->delta_o, 
				(nv_bfloat16*)current->output, param->beta, (param->biased_dim)*current->c_network->length,
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			quadratic_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output,
				(float*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			quadratic_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
				param->dim, param->size);
			break;
		
		case BF16C_FP32A:
			quadratic_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (nv_bfloat16*)current->output, 
				(nv_bfloat16*)current->c_network->target, (param->biased_dim)*current->c_network->length, 
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
		case FP32C_FP32A:
		case TF32C_FP32A:

			softmax_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				current->c_network->length, param->dim, current->c_network->batch_size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			softmax_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output,
				current->c_network->length, param->dim, current->c_network->batch_size);
			break;
		
		case BF16C_FP32A:
			softmax_activation_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->output,
				current->c_network->length, param->dim, current->c_network->batch_size);
			break;
	}
}

__global__ void softmax_activation_kernel_FP32(float *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
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


__global__ void softmax_activation_kernel_BF16(nv_bfloat16 *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;
	nv_bfloat16 *pos;
	nv_bfloat16 vmax;
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
			pos[j] /= (nv_bfloat16)normal;
				
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			cross_entropy_deriv_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, (float*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, 
				(param->biased_dim) * current->c_network->batch_size, TC_scale_factor);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			cross_entropy_deriv_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, (half*)current->c_network->target,
				(param->biased_dim)*current->c_network->length, param->dim, 
				(param->biased_dim)*current->c_network->batch_size, TC_scale_factor);
			break;
		
		case BF16C_FP32A:
			cross_entropy_deriv_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(nv_bfloat16*)current->delta_o, (nv_bfloat16*)current->output, (nv_bfloat16*)current->c_network->target,
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			cross_entropy_output_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, (param->dim+1)*current->c_network->length,
				param->dim, (param->biased_dim)*current->c_network->batch_size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			cross_entropy_output_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, (param->dim+1)*current->c_network->length,
				param->dim, (param->biased_dim)*current->c_network->batch_size);
			break;
		
		case BF16C_FP32A:
			cross_entropy_output_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (nv_bfloat16*)current->output, 
				(nv_bfloat16*)current->c_network->target, (param->dim+1)*current->c_network->length,
				param->dim, (param->biased_dim)*current->c_network->batch_size);
			break;
	}
}


__global__ void cross_entropy_deriv_output_error_kernel_FP32(float *delta_o, float *output, float *target, int len, int dim, int size, float TC_scale_factor)
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


__global__ void cross_entropy_deriv_output_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, int len, int dim, int size, float TC_scale_factor)
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


__global__ void cross_entropy_output_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, int len, int dim, int size)
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
		case FP32C_FP32A:
		case TF32C_FP32A:
			YOLO_activation_kernel_FP32<<< cu_blocks, cu_threads >>>((float*)current->output,
				c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size,
				a_param->biased_dim*current->c_network->length,
				*a_param, a_param->size);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			YOLO_activation_kernel_FP16<<< cu_blocks, cu_threads >>>((half*)current->output, 
				c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size,
				a_param->biased_dim*current->c_network->length,
				*a_param, a_param->size);
			break;
		
		case BF16C_FP32A:
			YOLO_activation_kernel_BF16<<< cu_blocks, cu_threads >>>((nv_bfloat16*)current->output, 
				c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2]*current->c_network->batch_size,
				a_param->biased_dim*current->c_network->length,
				*a_param, a_param->size);
			break;
	}
}


__global__ void YOLO_activation_kernel_FP32(float *tab, int flat_offset, int len, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if(i >= size)
		return;

	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;
	//Default values are in activ_function.c (set_yolo_params)
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	int col, in_col;

	col = i / flat_offset;
	in_col = col%(8+nb_class+nb_param);

	//Position
	if(in_col >= 0 && in_col < 3)
	{
		tab[i] = -(float)sm_tab[0][0]*tab[i];
		if(tab[i] > (float)sm_tab[0][1])
			tab[i] = (float)sm_tab[0][1];
		tab[i] = (float)1.0f/((float)1.0f + expf(tab[i]));
		return;
	}
	
	//Box size
	if(in_col >= 3 && in_col < 6)
	{
		tab[i] = (float)sm_tab[1][0]*tab[i];
		if(tab[i] > (float)sm_tab[1][1])
			tab[i] = (float)sm_tab[1][1];
		else if(tab[i] < (float)(sm_tab[1][2]))
			tab[i] = (float)(sm_tab[1][2]);
		return;
	}
	
	//Object probability
	if(in_col == 6)
	{
		tab[i] = -(float)sm_tab[2][0]*tab[i];
		if(tab[i] > (float)sm_tab[2][1])
			tab[i] = (float)sm_tab[2][1];
		tab[i] = (float)1.0f/((float)1.0f + expf(tab[i]));
		return;
	}
	
	//Objectness (Obj. quality => based on IoU)
	if(in_col == 7)
	{
		tab[i] = -(float)sm_tab[3][0]*tab[i];
		if(tab[i] > (float)sm_tab[3][1])
			tab[i] = (float)sm_tab[3][1];
		tab[i] = (float)1.0f/((float)1.0f + expf(tab[i]));
		return;
	}

	//Classes
	if(in_col >= 8 && in_col < 8+nb_class)
	{
		tab[i] = -(float)sm_tab[4][0]*tab[i];
		if(tab[i] > (float)sm_tab[4][1])
			tab[i] = (float)sm_tab[4][1];
		tab[i] = (float)1.0f/((float)1.0f + expf(tab[i]));
		return;
	}

	//Additional parameters (regression)
	if(in_col >= 8+nb_class)
	{
		tab[i] = (float)sm_tab[5][0]*tab[i];
		if(tab[i] > (float)sm_tab[5][1])
			tab[i] = (float)sm_tab[5][1];
		else if(tab[i] < (float)(sm_tab[5][2]))
			tab[i] = (float)(sm_tab[5][2]);
		return;
	}
}


__global__ void YOLO_activation_kernel_FP16(half *tab, int flat_offset, int len, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if(i >= size)
		return;

	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;
	//Default values are in activ_function.c (set_yolo_params)
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	int col, in_col;

	col = i / flat_offset;
	in_col = col%(8+nb_class+nb_param);

	//Position
	if(in_col >= 0 && in_col < 3)
	{
		tab[i] = -(half)sm_tab[0][0]*tab[i];
		if(tab[i] > (half)sm_tab[0][1])
			tab[i] = (half)sm_tab[0][1];
		tab[i] = (half)1.0f/((half)1.0f + hexp(tab[i]));
		return;
	}
	
	//Box size
	if(in_col >= 3 && in_col < 6)
	{
		tab[i] = (half)sm_tab[1][0]*tab[i];
		if(tab[i] > (half)sm_tab[1][1])
			tab[i] = (half)sm_tab[1][1];
		else if(tab[i] < (half)(sm_tab[1][2]))
			tab[i] = (half)(sm_tab[1][2]);
		return;
	}
	
	//Object probability
	if(in_col == 6)
	{
		tab[i] = -(half)sm_tab[2][0]*tab[i];
		if(tab[i] > (half)sm_tab[2][1])
			tab[i] = (half)sm_tab[2][1];
		tab[i] = (half)1.0f/((half)1.0f + hexp(tab[i]));
		return;
	}
	
	//Objectness (Obj. quality => based on IoU)
	if(in_col == 7)
	{
		tab[i] = -(half)sm_tab[3][0]*tab[i];
		if(tab[i] > (half)sm_tab[3][1])
			tab[i] = (half)sm_tab[3][1];
		tab[i] = (half)1.0f/((half)1.0f + hexp(tab[i]));
		return;
	}

	//Classes
	if(in_col >= 8 && in_col < 8+nb_class)
	{
		tab[i] = -(half)sm_tab[4][0]*tab[i];
		if(tab[i] > (half)sm_tab[4][1])
			tab[i] = (half)sm_tab[4][1];
		tab[i] = (half)1.0f/((half)1.0f + hexp(tab[i]));
		return;
	}

	//Additional parameters (regression)
	if(in_col >= 8+nb_class)
	{
		tab[i] = (half)sm_tab[5][0]*tab[i];
		if(tab[i] > (half)sm_tab[5][1])
			tab[i] = (half)sm_tab[5][1];
		else if(tab[i] < (half)(sm_tab[5][2]))
			tab[i] = (half)(sm_tab[5][2]);
		return;
	}
}


__global__ void YOLO_activation_kernel_BF16(nv_bfloat16 *tab, int flat_offset, int len, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if(i >= size)
		return;

	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;
	//Default values are in activ_function.c (set_yolo_params)
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	int col, in_col;

	col = i / flat_offset;
	in_col = col%(8+nb_class+nb_param);

	//Position
	if(in_col >= 0 && in_col < 3)
	{
		tab[i] = -(nv_bfloat16)sm_tab[0][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[0][1])
			tab[i] = (nv_bfloat16)sm_tab[0][1];
		tab[i] = (nv_bfloat16)1.0f/((nv_bfloat16)1.0f + hexp(tab[i]));
		return;
	}
	
	//Box size
	if(in_col >= 3 && in_col < 6)
	{
		tab[i] = (nv_bfloat16)sm_tab[1][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[1][1])
			tab[i] = (nv_bfloat16)sm_tab[1][1];
		else if(tab[i] < (nv_bfloat16)(sm_tab[1][2]))
			tab[i] = (nv_bfloat16)(sm_tab[1][2]);
		return;
	}
	
	//Object probability
	if(in_col == 6)
	{
		tab[i] = -(nv_bfloat16)sm_tab[2][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[2][1])
			tab[i] = (nv_bfloat16)sm_tab[2][1];
		tab[i] = (nv_bfloat16)1.0f/((nv_bfloat16)1.0f + hexp(tab[i]));
		return;
	}
	
	//Objectness (Obj. quality => based on IoU)
	if(in_col == 7)
	{
		tab[i] = -(nv_bfloat16)sm_tab[3][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[3][1])
			tab[i] = (nv_bfloat16)sm_tab[3][1];
		tab[i] = (nv_bfloat16)1.0f/((nv_bfloat16)1.0f + hexp(tab[i]));
		return;
	}

	//Classes
	if(in_col >= 8 && in_col < 8+nb_class)
	{
		tab[i] = -(nv_bfloat16)sm_tab[4][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[4][1])
			tab[i] = (nv_bfloat16)sm_tab[4][1];
		tab[i] = (nv_bfloat16)1.0f/((nv_bfloat16)1.0f + hexp(tab[i]));
		return;
	}

	//Additional parameters (regression)
	if(in_col >= 8+nb_class)
	{
		tab[i] = (nv_bfloat16)sm_tab[5][0]*tab[i];
		if(tab[i] > (nv_bfloat16)sm_tab[5][1])
			tab[i] = (nv_bfloat16)sm_tab[5][1];
		else if(tab[i] < (nv_bfloat16)(sm_tab[5][2]))
			tab[i] = (nv_bfloat16)(sm_tab[5][2]);
		return;
	}
}


void cuda_YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			YOLO_deriv_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->delta_o, (float*)current->output, 
				(float*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, TC_scale_factor);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			YOLO_deriv_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(half*)current->delta_o, (half*)current->output, 
				(half*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, TC_scale_factor);
			break;
		
		case BF16C_FP32A:
			YOLO_deriv_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(nv_bfloat16*)current->delta_o, (nv_bfloat16*)current->output, 
				(nv_bfloat16*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, TC_scale_factor);
			break;
	}
}


void cuda_YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	switch(current->c_network->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			YOLO_error_kernel_FP32<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (float*)current->output, 
				(float*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			YOLO_error_kernel_FP16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (half*)current->output, 
				(half*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
			break;
		
		case BF16C_FP32A:
			YOLO_error_kernel_BF16<<< cu_blocks, cu_threads >>>(
				(float*)current->c_network->output_error, (nv_bfloat16*)current->output, 
				(nv_bfloat16*)current->c_network->target, current->c_network->output_dim, 
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], *a_param,
				c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
			break;
	}
}

__device__ float gpu_IoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


__device__ float gpu_GIoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}

//order: xmin, ymin, zmin, xmax, ymax, zmax
__device__ float gpu_DIoU_fct(float* output, float* target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  abs(output[3]-output[0])*abs(output[4]-output[1])*abs(output[5]-output[2])
			+ abs(target[3]-target[0])*abs(target[4]-target[1])*abs(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}

// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
__global__ void YOLO_deriv_error_kernel_FP32(float *delta_o, float *output, float *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
		return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	int fit_size = y_param.fit_parts[0], fit_prob = y_param.fit_parts[1], fit_obj = y_param.fit_parts[2];
	int fit_class = y_param.fit_parts[3], fit_param = y_param.fit_parts[4];
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	
	delta_o += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output  += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
			
			box_locked[resp_box] = 2;
		
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
				delta_o[(resp_box*(8+nb_class+nb_param)+k)*f_offset] = (float)(
					TC_scale_factor*sm_tab[0][0]*coord_scale*(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset]
					*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]));
			}
			
			if(fit_size)
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (float) (TC_scale_factor*sm_tab[1][0]*size_scale*
						((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]));
			}
			else
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (float) (0.0f);
			}
			
			
			
			if(fit_prob && max_IoU > min_prob_IoU_lim)
				delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (float)(
		                    	TC_scale_factor*sm_tab[2][0]*prob_scale*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]
		                    	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset])
		                    	*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f));
	        else
	        	delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (float)(0.0f);
		        
	        if(fit_obj && max_IoU > min_obj_IoU_lim)
	        {
	        	if(max_IoU > 0.99f)
	        		max_IoU = 0.99f;
	        	delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (float)(
		            	TC_scale_factor*sm_tab[3][0]*obj_scale*(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]
		            	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset])
		            	*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5));
			}
			else
				delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (float)(0.0f);			
			
			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(fit_class && max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int) target[j*(7+nb_param)]-1)
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(float) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f));
					else
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(float) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f));
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear activation of additional parameters
			if(fit_param && max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(float) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float)target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (float) 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				delta_o[(j*(8+nb_class+nb_param)+k)*f_offset] = (float) 0.0f;
				
			if(box_locked[j] == 1)
			{
				delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (float) 0.0f;
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (float) 0.0f;
			}
			else
			{
				if(fit_prob)
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (float)(
						TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])*prob_scale
						*(float)output[(j*(8+nb_class+nb_param)+6)*f_offset]
						*(1.0f-(float)output[(j*(8+nb_class+nb_param)+6)*f_offset])
						*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f));
				else
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (float)(0.0f);
				
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (float) 0.0f;
			
			}			
			
			for(k = 0; k < nb_class; k++)
				delta_o[(j*(8+nb_class+nb_param)+8+k)*f_offset] = (float) 0.0f;
				
			for(k = 0; k < nb_param; k++)
                	delta_o[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (float) 0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);

}


// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
__global__ void YOLO_deriv_error_kernel_FP16(half *delta_o, half *output, half *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
		return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	int fit_size = y_param.fit_parts[0], fit_prob = y_param.fit_parts[1], fit_obj = y_param.fit_parts[2];
	int fit_class = y_param.fit_parts[3], fit_param = y_param.fit_parts[4];
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	
	delta_o += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output  += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
			
			box_locked[resp_box] = 2;
		
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
				delta_o[(resp_box*(8+nb_class+nb_param)+k)*f_offset] = (half)(
					TC_scale_factor*sm_tab[0][0]*coord_scale*(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset]
					*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]));
			}
			
			if(fit_size)
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (half) (TC_scale_factor*sm_tab[1][0]*size_scale*
						((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]));
			}
			else
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (half) (0.0f);
			}
			
			
			
			if(fit_prob && max_IoU > min_prob_IoU_lim)
				delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (half)(
		                    	TC_scale_factor*sm_tab[2][0]*prob_scale*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]
		                    	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset])
		                    	*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f));
	        else
	        	delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (half)(0.0f);
		        
	        if(fit_obj && max_IoU > min_obj_IoU_lim)
	        {
	        	if(max_IoU > 0.99f)
	        		max_IoU = 0.99f;
	        	delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (half)(
		            	TC_scale_factor*sm_tab[3][0]*obj_scale*(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]
		            	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset])
		            	*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5));
			}
			else
				delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (half)(0.0f);			
			
			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(fit_class && max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int) target[j*(7+nb_param)]-1)
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(half) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f));
					else
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(half) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f));
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear activation of additional parameters
			if(fit_param && max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(half) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float)target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (half) 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				delta_o[(j*(8+nb_class+nb_param)+k)*f_offset] = (half) 0.0f;
				
			if(box_locked[j] == 1)
			{
				delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (half) 0.0f;
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (half) 0.0f;
			}
			else
			{
				if(fit_prob)
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (half)(
						TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])*prob_scale
						*(float)output[(j*(8+nb_class+nb_param)+6)*f_offset]
						*(1.0f-(float)output[(j*(8+nb_class+nb_param)+6)*f_offset])
						*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f));
				else
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (half)(0.0f);
				
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (half) 0.0f;
			
			}			
			
			for(k = 0; k < nb_class; k++)
				delta_o[(j*(8+nb_class+nb_param)+8+k)*f_offset] = (half) 0.0f;
				
			for(k = 0; k < nb_param; k++)
                	delta_o[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (half) 0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}


__global__ void YOLO_deriv_error_kernel_BF16(nv_bfloat16 *delta_o, nv_bfloat16 *output, nv_bfloat16 *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
		return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	int fit_size = y_param.fit_parts[0], fit_prob = y_param.fit_parts[1], fit_obj = y_param.fit_parts[2];
	int fit_class = y_param.fit_parts[3], fit_param = y_param.fit_parts[4];
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	
	delta_o += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output  += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
			
			box_locked[resp_box] = 2;
		
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
				delta_o[(resp_box*(8+nb_class+nb_param)+k)*f_offset] = (nv_bfloat16)(
					TC_scale_factor*sm_tab[0][0]*coord_scale*(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset]
					*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]));
			}
			
			if(fit_size)
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (nv_bfloat16) (TC_scale_factor*sm_tab[1][0]*size_scale*
						((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]));
			}
			else
			{
				for(k = 0; k < 3; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] = (nv_bfloat16) (0.0f);
			}
			
			
			
			if(fit_prob && max_IoU > min_prob_IoU_lim)
				delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (nv_bfloat16)(
		                    	TC_scale_factor*sm_tab[2][0]*prob_scale*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]
		                    	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset])
		                    	*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f));
	        else
	        	delta_o[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = (nv_bfloat16)(0.0f);
		        
	        if(fit_obj && max_IoU > min_obj_IoU_lim)
	        {
	        	if(max_IoU > 0.99f)
	        		max_IoU = 0.99f;
	        	delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (nv_bfloat16)(
		            	TC_scale_factor*sm_tab[3][0]*obj_scale*(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]
		            	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset])
		            	*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5));
			}
			else
				delta_o[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = (nv_bfloat16)(0.0f);			
			
			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(fit_class && max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int) target[j*(7+nb_param)]-1)
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(nv_bfloat16) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f));
					else
						delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 
							(nv_bfloat16) (TC_scale_factor*sm_tab[4][0]*class_scale*(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]
			                	*(1.0f-(float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset])
			                	*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f));
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear activation of additional parameters
			if(fit_param && max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(nv_bfloat16) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float)target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					delta_o[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (nv_bfloat16) 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				delta_o[(j*(8+nb_class+nb_param)+k)*f_offset] = (nv_bfloat16) 0.0f;
				
			if(box_locked[j] == 1)
			{
				delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (nv_bfloat16) 0.0f;
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (nv_bfloat16) 0.0f;
			}
			else
			{
				if(fit_prob)
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (nv_bfloat16)(
						TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])*prob_scale
						*(float)output[(j*(8+nb_class+nb_param)+6)*f_offset]
						*(1.0f-(float)output[(j*(8+nb_class+nb_param)+6)*f_offset])
						*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f));
				else
					delta_o[(j*(8+nb_class+nb_param)+6)*f_offset] = (nv_bfloat16)(0.0f);
				
				delta_o[(j*(8+nb_class+nb_param)+7)*f_offset] = (nv_bfloat16) 0.0f;
			
			}			
			
			for(k = 0; k < nb_class; k++)
				delta_o[(j*(8+nb_class+nb_param)+8+k)*f_offset] = (nv_bfloat16) 0.0f;
				
			for(k = 0; k < nb_param; k++)
                	delta_o[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = (nv_bfloat16) 0.0f;
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}



// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
__global__ void YOLO_error_kernel_FP32(float *output_error, float *output, float *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
    	return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *IoU_monitor = y_param.IoU_monitor;
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	
	output_error += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	IoU_monitor += 2 * nb_box * ((nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x);

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
		
		IoU_monitor[k*2] = 0.0f;
		IoU_monitor[k*2+1] = -1.0f;
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
            
			box_locked[resp_box] = 2;
			IoU_monitor[resp_box*2] = 1.0f;
			IoU_monitor[resp_box*2+1] = max_IoU*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset];
		
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
			
			//Already compute error for the responsible box
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k)*f_offset] =
					0.5f*coord_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] =
					0.5f*size_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]);
			
			if(max_IoU > min_prob_IoU_lim)
				output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] =
		                        0.5f*prob_scale*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f)
		                        *((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f);
            else
            	output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
			
			if(max_IoU > min_obj_IoU_lim)
			{
				if(max_IoU > 0.99f)
				    	max_IoU = 0.99f;
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] =
						     0.5f*obj_scale*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5)
						     *((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5);
			}
			else
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;

			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int)target[j*(7+nb_param)]-1)
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f);
					else
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f);
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear error of additional parameters
			if(max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(param_ind_scale[k]*0.5f*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k])
						*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				output_error[(j*(8+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			else
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] =
					0.5f*(lambda_noobj_prior[j])*prob_scale*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f)
					*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f);
				
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			
			for(k = 0; k < nb_class; k++)
				output_error[(j*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
				
			for(k = 0; k < nb_param; k++)
					output_error[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}


// Only minimal optimisation has been performed for now => might be responsible for a significant portion of the total network time
__global__ void YOLO_error_kernel_FP16(float *output_error, half *output, half *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
    	return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *IoU_monitor = y_param.IoU_monitor;
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	
	output_error += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	IoU_monitor += 2 * nb_box * ((nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x);

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
		
		IoU_monitor[k*2] = 0.0f;
		IoU_monitor[k*2+1] = -1.0f;
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
            
			box_locked[resp_box] = 2;
			IoU_monitor[resp_box*2] = 1.0f;
			IoU_monitor[resp_box*2+1] = max_IoU*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset];
		
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
			
			//Already compute error for the responsible box
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k)*f_offset] =
					0.5f*coord_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] =
					0.5f*size_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]);
			
			if(max_IoU > min_prob_IoU_lim)
				output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] =
		                        0.5f*prob_scale*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f)
		                        *((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f);
            else
            	output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
			
			if(max_IoU > min_obj_IoU_lim)
			{
				if(max_IoU > 0.99f)
				    	max_IoU = 0.99f;
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] =
						     0.5f*obj_scale*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5)
						     *((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5);
			}
			else
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;

			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int)target[j*(7+nb_param)]-1)
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f);
					else
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f);
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear error of additional parameters
			if(max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(param_ind_scale[k]*0.5f*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k])
						*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				output_error[(j*(8+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			else
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] =
					0.5f*(lambda_noobj_prior[j])*prob_scale*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f)
					*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f);
				
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			
			for(k = 0; k < nb_class; k++)
				output_error[(j*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
				
			for(k = 0; k < nb_param; k++)
					output_error[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}


__global__ void YOLO_error_kernel_BF16(float *output_error, nv_bfloat16 *output, nv_bfloat16 *target, int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= size)
    	return;
                
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;
	int strict_box_size_association = y_param.strict_box_size_association;
	
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab = y_param.slopes_and_maxes_tab;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim = y_param.IoU_limits[0];
	float min_prob_IoU_lim = y_param.IoU_limits[1], min_obj_IoU_lim = y_param.IoU_limits[2];
	float min_class_IoU_lim = y_param.IoU_limits[3], min_param_IoU_lim = y_param.IoU_limits[4];
	
	float *param_ind_scale = y_param.param_ind_scale;
	float *IoU_monitor = y_param.IoU_monitor;
	
	int j, k, l;
	int c_batch, f_offset;
	int nb_obj_target;
	int resp_box = -1;
	float max_IoU, current_IoU;
	int cell_x, cell_y, cell_z;
	int obj_cx, obj_cy, obj_cz;
	float *box_in_pix, *c_box_in_pix;
	float obj_in_offset[6];
	int *box_locked;
	float out_int[6], targ_int[6];
	
	float targ_w, targ_h, targ_d;
	int larger_box, smaller_box;
	
	box_locked = (int*) malloc(nb_box*sizeof(int));
	box_in_pix = (float*) malloc(nb_box*6*sizeof(float));
	
	c_batch = i / flat_output_size;
	target += flat_target_size * c_batch;
	f_offset = size;

	i = i % flat_output_size;
	cell_z = i / (nb_area_w*nb_area_h);
	cell_y = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;
	cell_x = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;
	
	output_error += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;
	output += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;

	IoU_monitor += 2 * nb_box * ((nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x);

	nb_obj_target = target[0];
	target++;
	
	for(k = 0; k < nb_box; k++)
	{
		box_locked[k] = 0;
		c_box_in_pix = box_in_pix+k*6;
		c_box_in_pix[0] = ((float)output[(k*(8+nb_class+nb_param)+0)*f_offset] + cell_x) * cell_w;
		c_box_in_pix[1] = ((float)output[(k*(8+nb_class+nb_param)+1)*f_offset] + cell_y) * cell_h;
		c_box_in_pix[2] = ((float)output[(k*(8+nb_class+nb_param)+2)*f_offset] + cell_z) * cell_d;
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(k*(8+nb_class+nb_param)+3)*f_offset]);
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(k*(8+nb_class+nb_param)+4)*f_offset]);
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(k*(8+nb_class+nb_param)+5)*f_offset]);
		
		IoU_monitor[k*2] = 0.0f;
		IoU_monitor[k*2+1] = -1.0f;
	}
	
	for(j = 0; j < nb_obj_target; j++)
	{
		if((int) target[j*(7+nb_param)] == 0)
			break;
		obj_cx = int( ((float)target[j*(7+nb_param)+4] + (float)target[j*(7+nb_param)+1])*0.5f / cell_w);
		obj_cy = int( ((float)target[j*(7+nb_param)+5] + (float)target[j*(7+nb_param)+2])*0.5f / cell_h);
		obj_cz = int( ((float)target[j*(7+nb_param)+6] + (float)target[j*(7+nb_param)+3])*0.5f / cell_d);
		
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)
		{
			for(k = 0; k < 6; k++)
				targ_int[k] = target[j*(7+nb_param)+1+k];
			
			targ_w = targ_int[3] - targ_int[0];
			targ_h = targ_int[4] - targ_int[1];
			targ_d = targ_int[5] - targ_int[2];
		
			resp_box = -1;
			max_IoU = -1.0f;
			for(k = 0; k < nb_box; k++)
			{
				larger_box = 0;
				smaller_box = 0;
				if(strict_box_size_association)
				{
					for(l = k; l < nb_box - 1; l++)
					{
						if(prior_w[l+1]*prior_h[l+1]*prior_d[l+1] > prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d >= prior_w[l+1]*prior_h[l+1]*prior_d[l+1])
								larger_box = 1;
					}
					for(l = k; l > 0; l--)
					{
						if(prior_w[l-1]*prior_h[l-1]*prior_d[l+1] < prior_w[k]*prior_h[k]*prior_d[k])
							if(targ_w*targ_h*targ_d < prior_w[l-1]*prior_h[l-1]*prior_h[l-1])
								smaller_box = 1;
					}
				}
			
				if(box_locked[k] == 2 || larger_box || smaller_box)
					continue;
			
				c_box_in_pix = box_in_pix+k*6;
				out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];
				out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];
				out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];
				out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];
				out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];
				out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];

				current_IoU = y_param.c_IoU_fct(out_int, targ_int);
				
				if(current_IoU > max_IoU)
				{
					max_IoU = current_IoU;
					resp_box = k;
				}
				if(current_IoU > good_IoU_lim) //Avoid update of non best but still good match boxes
					box_locked[k] = 1;
			}
			
			if(resp_box == -1 || box_locked[resp_box] == 2)
            	continue;
            
			box_locked[resp_box] = 2;
			IoU_monitor[resp_box*2] = 1.0f;
			IoU_monitor[resp_box*2+1] = max_IoU*(float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset];
		
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
			
			//Already compute error for the responsible box
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k)*f_offset] =
					0.5f*coord_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k)*f_offset] - obj_in_offset[k]);
			for(k = 0; k < 3; k++)
				output_error[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] =
					0.5f*size_scale*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3])
					*((float)output[(resp_box*(8+nb_class+nb_param)+k+3)*f_offset] - obj_in_offset[k+3]);
			
			if(max_IoU > min_prob_IoU_lim)
				output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] =
		                        0.5f*prob_scale*((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f)
		                        *((float)output[(resp_box*(8+nb_class+nb_param)+6)*f_offset]-0.99f);
            else
            	output_error[(resp_box*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
			
			if(max_IoU > min_obj_IoU_lim)
			{
				if(max_IoU > 0.99f)
				    	max_IoU = 0.99f;
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] =
						     0.5f*obj_scale*((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5)
						     *((float)output[(resp_box*(8+nb_class+nb_param)+7)*f_offset]-(1.0+max_IoU)*0.5);
			}
			else
				output_error[(resp_box*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;

			//mean square error on classes => could be changed to soft max (change in activation needed as well)
			if(max_IoU > min_class_IoU_lim)
			{
				for(k = 0; k < nb_class; k++)
				{
					if(k == (int)target[j*(7+nb_param)]-1)
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.99f);
					else
						output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.5f*class_scale
							*((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f)
				            *((float)output[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset]-0.01f);
				}
			}
			else
			{
				for(k = 0; k < nb_class; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
			}
			
			//linear error of additional parameters
			if(max_IoU > min_param_IoU_lim)
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 
						(param_ind_scale[k]*0.5f*param_scale*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k])
						*((float)output[(resp_box*(8+nb_class+nb_param)
						+8+nb_class+k)*f_offset] - (float) target[j*(7+nb_param)+7+k]));
			}
			else
			{
				for(k = 0; k < nb_param; k++)
					output_error[(resp_box*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			}
		}
	}
	
	for(j = 0; j < nb_box; j++)
	{
		//If no match (means no IoU > 0.5) only update Objectness toward 0 (here it means error compute)! (no coordinate nor class update)
		if(box_locked[j] != 2)
		{
			for(k = 0; k < 6; k++)
				output_error[(j*(8+nb_class+nb_param)+k)*f_offset] = 0.0f;
				
			if(box_locked[j] == 1)
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] = 0.0f;
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			else
			{
				output_error[(j*(8+nb_class+nb_param)+6)*f_offset] =
					0.5f*(lambda_noobj_prior[j])*prob_scale*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f)
					*((float)output[(j*(8+nb_class+nb_param)+6)*f_offset]-0.01f);
				
				output_error[(j*(8+nb_class+nb_param)+7)*f_offset] = 0.0f;
			}
			
			for(k = 0; k < nb_class; k++)
				output_error[(j*(8+nb_class+nb_param)+8+k)*f_offset] = 0.0f;
				
			for(k = 0; k < nb_param; k++)
					output_error[(j*(8+nb_class+nb_param)+8+nb_class+k)*f_offset] = 0.0f;
			
		}
	}
	
	free(box_in_pix);
	free(box_locked);
}



//#####################################################








