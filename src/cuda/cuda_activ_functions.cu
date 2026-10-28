	
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


#include "../prototypes.h"

// Local variables
static int cu_blocks;

// Public are in "prototypes.h"

// Private prototypes
__device__ float gpu_IoU_fct(float *output, float *target);
__device__ float gpu_GIoU_fct(float *output, float *target);
__device__ float gpu_DIoU_fct(float *output, float *target);
__device__ float gpu_DIoU2_fct(float *output, float *target);
void cuda_linear_activation(layer *current);
void cuda_linear_deriv(layer *current);
void cuda_linear_deriv_output_error(layer *current);
void cuda_linear_output_error(layer *current);
void cuda_ReLU_activation(layer *current);
void cuda_ReLU_deriv(layer *current);
void cuda_ReLU_deriv_output_error(layer* current);
void cuda_ReLU_output_error(layer* current);
void cuda_logistic_activation(layer *current);
void cuda_logistic_deriv(layer *current);
void cuda_logistic_deriv_output_error(layer* current);
void cuda_logistic_output_error(layer* current);
void cuda_softmax_activation(layer *current);
void cuda_softmax_deriv(layer *current);
void cuda_softmax_deriv_output_error(layer *current);
void cuda_softmax_output_error(layer *current);
void cuda_YOLO_activation(layer *current);
void cuda_YOLO_deriv(layer *current);
void cuda_YOLO_deriv_output_error(layer *current);
void cuda_YOLO_output_error(layer *current);
void cuda_YOLO_activ_init(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead


#define linear_batch_edge_kernel(name, type)																									\
__global__ void linear_batch_edge_kernel_##name(void *i_tab, size_t dim, size_t biased_dim, size_t offset,										\
	size_t length, size_t batch_size)																											\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t feature_id = blockIdx.y;																												\
	type* tab = (type*) i_tab;																													\
																																				\
	if(biased_dim > dim)																														\
		if(i < (batch_size - length) * biased_dim)																								\
			tab[length*biased_dim + i] = (type) 0.0f;																							\
	else																																		\
		if(i < (batch_size - length) * dim)																										\
			tab[feature_id*dim*batch_size + length*dim + i] = (type) 0.0f;																		\
}

#define linear_bias_deriv_kernel(name, type)																									\
__global__ void linear_bias_deriv_kernel_##name(void *i_deriv, size_t biased_dim, size_t dim, size_t batch_size)								\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type* deriv = (type*) i_deriv;																												\
																																				\
	if(i < batch_size)																															\
		deriv[i*biased_dim + dim] = (type) 0.0f;																								\
}


//#####################################################
//		  ReLU activation related templates
//#####################################################

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
#define ReLU_activation_kernel(name, type)																										\
__global__ void ReLU_activation_kernel_##name(void *i_tab, size_t dim, size_t biased_dim, size_t offset,										\
	float saturation, float leaking_factor, size_t length, size_t size)																			\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < length*biased_dim)																												\
		{																																		\
			if((i+1)%(dim+1) != 0)																												\
			{																																	\
				if(tab[i] <= (type) 0.0f)																										\
					tab[i] *= (type) leaking_factor;																							\
				else if(tab[i] > (type) saturation)																								\
					tab[i] = (type) saturation + (tab[i] - (type) saturation)*((type)leaking_factor);											\
			}																																	\
		}																																		\
		else																																	\
			tab[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			if(tab[i] <= (type) 0.0f)																											\
				tab[i] *= (type) leaking_factor;																								\
			else if(tab[i] > (type) saturation)																									\
				tab[i] = (type) saturation + (tab[i] - (type) saturation)*((type)leaking_factor);												\
		}																																		\
		else																																	\
			tab[i] = (type) 0.0f;																												\
	}																																			\
}


#define ReLU_deriv_kernel(name, type)																											\
__global__ void ReLU_deriv_kernel_##name(void *i_deriv, void *i_value, size_t dim, size_t biased_dim, size_t offset,							\
	 float saturation, float leaking_factor, size_t length, size_t size)																		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			if(value[i] <= (type) 0.0f)																											\
				deriv[i] *= leaking_factor;																										\
			else if(value[i] > (type) saturation)																								\
				deriv[i] *= leaking_factor;																										\
		}																																		\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			if(value[i] <= (type) 0.0f)																											\
				deriv[i] *= leaking_factor;																										\
			else if(value[i] > (type) saturation)																								\
				deriv[i] *= leaking_factor;																										\
		}																																		\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}


#define quadratic_deriv_output_error_kernel(name, type)																							\
__global__ void quadratic_deriv_output_error_kernel_##name																						\
	(void *i_delta_o, void *i_output, void *i_target, size_t dim, size_t biased_dim, 															\
	size_t offset, size_t length, size_t size, float TC_scale_factor)																			\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;																					\
																																				\
	type* delta_o = (type*) i_delta_o;																											\
	type* output  = (type*) i_output;																											\
	type* target  = (type*) i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define quadratic_output_error_kernel(name, type)																								\
__global__ void quadratic_output_error_kernel_##name																							\
	(float *output_error, void *i_output, void *i_target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size)				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;																					\
																																				\
	type* output = (type*) i_output;																											\
	type* target = (type*) i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			output_error[i] = (0.5f*((float)output[i] - (float)target[pos])*((float)output[i] - (float)target[pos]));							\
		}																																		\
		else																																	\
			output_error[i]	= 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			output_error[i] = (0.5f*((float)output[i] - (float)target[pos])*((float)output[i] - (float)target[pos]));							\
		}																																		\
		else																																	\
			output_error[i]	= 0.0f;																												\
	}																																			\
}

//#####################################################


//#####################################################
//		  Logistic activation related templates
//#####################################################

#define logistic_activation_kernel(name, type, exp_fct)																							\
__global__ void logistic_activation_kernel_##name(void *i_tab, float beta, float saturation, size_t dim, 										\
	size_t biased_dim, size_t offset, size_t length, size_t size)																				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
	float t_one = (type) 1.0f;																													\
	type t_beta = (type) beta;																													\
	type t_saturation = (type) saturation;																										\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < length*biased_dim)																												\
		{																																		\
			if((i+1)%(dim+1) != 0)																												\
			{																																	\
				tab[i] = -t_beta*tab[i];																										\
				if(tab[i] > t_saturation)																										\
					tab[i] = t_saturation;																										\
				tab[i] = t_one/(t_one + exp_fct((float)tab[i]));																				\
			}																																	\
		}																																		\
		else																																	\
			tab[i] = (type)0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			tab[i] = -t_beta*tab[i];																											\
			if(tab[i] > t_saturation)																											\
				tab[i] = t_saturation;																											\
			tab[i] = t_one/(t_one + exp_fct((float)tab[i]));																					\
		}																																		\
		else																																	\
			tab[i] = (type)0.0f;																												\
	}																																			\
}


#define logistic_deriv_kernel(name, type)																										\
__global__ void logistic_deriv_kernel_##name(void *i_deriv, void *i_value, float beta, size_t dim, 												\
	size_t biased_dim, size_t offset, size_t length, size_t size)																				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}

//#####################################################


//#####################################################
//		  Soft-Max activation related templates
//#####################################################


#define softmax_activation_kernel(name, type, exp_fct)																							\
__global__ void softmax_activation_kernel_##name(void *i_tab, size_t dim, size_t biased_dim, 													\
	size_t offset, size_t length, size_t batch_size, size_t size)																				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t j, k, l;																																\
	size_t nb_filters;																															\
	type *pos, *off_pos;																														\
	type vmax;																																	\
	float normal = 0.0f;																														\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= batch_size)																															\
		return;																																	\
																																				\
	pos = tab + i*(biased_dim);																													\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < length)																															\
		{																																		\
			vmax = *pos;																														\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				if(*off_pos > vmax)																												\
					vmax = *off_pos;																											\
			}																																	\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = exp_fct((float)(*off_pos-vmax));																						\
				normal += (float)*off_pos;																										\
			}																																	\
			pos[dim] = 0.0f;																													\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = (type)((float)*off_pos/normal);																						\
			}																																	\
			pos[dim] = 0.0f;																													\
		}																																		\
		else																																	\
		{																																		\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = 0.0f;																												\
			}																																	\
			pos[dim] = 0.0f;																													\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		nb_filters = size / (dim*batch_size);																									\
		if(i < length)																															\
		{																																		\
			vmax = *pos;																														\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					if(*off_pos > vmax)																											\
						vmax = *off_pos;																										\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = exp_fct((float)(*off_pos-vmax));																					\
					normal += (float)*off_pos;																									\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = (type)((float)*off_pos/normal);																					\
				}																																\
			}																																	\
		}																																		\
		else																																	\
		{																																		\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = 0.0f;																											\
				}																																\
			}																																	\
		}																																		\
	}																																			\
}


#define cross_entropy_deriv_output_error_kernel(name, type)																						\
__global__ void cross_entropy_deriv_output_error_kernel_##name																					\
	(void *i_delta_o, void *i_output, void *i_target, size_t dim, size_t biased_dim, 															\
		size_t offset, size_t length, size_t size, float TC_scale_factor)																		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;																					\
																																				\
	type* delta_o = (type*)i_delta_o;																											\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define cross_entropy_output_error_kernel(name, type)																							\
__global__ void cross_entropy_output_error_kernel_##name																						\
	(float *output_error, void *i_output, void *i_target, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size)				\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	size_t nb_filters, c_batch, c_filter, in_filter_pos, pos;																					\
																																				\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			if((float)output[i] > 0.000001f)																									\
				output_error[i] = -(float)target[pos] * logf((float)output[i]);																	\
			else																																\
				output_error[i] = -(float)target[pos] * logf((float)0.000001f);																	\
		}																																		\
		else																																	\
			output_error[i] = 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			if((float)output[i] > 0.000001f)																									\
				output_error[i] = -(float)target[pos] * logf((float)output[i]);																	\
			else																																\
				output_error[i] = -(float)target[pos] * logf((float)0.000001f);																	\
		}																																		\
		else																																	\
			output_error[i] = 0.0f;																												\
	}																																			\
}


//#####################################################
//		  YOLO activation related templates
//#####################################################

#define YOLO_activation_kernel(name, type, exp_fct)																								\
__global__ void YOLO_activation_kernel_##name(void *i_tab, size_t flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax)	\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *tab = (type*) i_tab;																													\
																																				\
	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;																				\
	/*Default values are in activ_function.c (set_yolo_params)*/																				\
	float **sm_tab = y_param.slopes_and_maxes_tab;																								\
	float normal = 0.0f;																														\
	type vmax;																																	\
	int fit_dim = y_param.fit_dim;																												\
	size_t col, in_col, j;																														\
																																				\
	col = i / flat_offset;																														\
	in_col = col%(8+nb_class+nb_param);																											\
																																				\
	/*Position*/																																\
	if(in_col < 3)																																\
	{																																			\
		if(fit_dim > in_col)																													\
		{																																		\
			tab[i] = -(type)sm_tab[0][0]*tab[i];																								\
			if(tab[i] > (type)sm_tab[0][1])																										\
				tab[i] = (type)sm_tab[0][1];																									\
			else if(tab[i] < (type)sm_tab[0][2])																								\
				tab[i] = (type)sm_tab[0][2];																									\
			tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																								\
		}																																		\
		else																																	\
			tab[i] = 0.5f; /*Center of the cell*/																								\
		return;																																	\
	}																																			\
																																				\
	/*Box size*/																																\
	if(in_col >= 3 && in_col < 6)																												\
	{																																			\
		if(fit_dim > in_col-3)																													\
		{																																		\
			tab[i] = (type)sm_tab[1][0]*tab[i];																									\
			if(tab[i] > (type)sm_tab[1][1])																										\
				tab[i] = (type)sm_tab[1][1];																									\
			else if(tab[i] < (type)(sm_tab[1][2]))																								\
				tab[i] = (sm_tab[1][2]);																										\
		}																																		\
		else																																	\
			tab[i] = 0.0f; /*Output = prior*/																									\
		return;																																	\
	}																																			\
																																				\
	/*Object probability*/																														\
	if(in_col == 6)																																\
	{																																			\
		tab[i] = -(type)sm_tab[2][0]*tab[i];																									\
		if(tab[i] > (type)sm_tab[2][1])																											\
			tab[i] = (type)sm_tab[2][1];																										\
		else if(tab[i] < (type)sm_tab[2][2])																									\
			tab[i] = (type)sm_tab[2][2];																										\
		tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																									\
		return;																																	\
	}																																			\
																																				\
	/*Objectness (Obj. quality => based on IoU)*/																								\
	if(in_col == 7)																																\
	{																																			\
		tab[i] = -(type)sm_tab[3][0]*tab[i];																									\
		if(tab[i] > (type)sm_tab[3][1])																											\
			tab[i] = (type)sm_tab[3][1];																										\
		else if(tab[i] < (type)sm_tab[3][2])																									\
			tab[i] = (type)sm_tab[3][2];																										\
		tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																									\
		return;																																	\
	}																																			\
																																				\
	/*Classes*/																																	\
	if(in_col >= 8 && in_col < 8+nb_class)																										\
	{																																			\
		if(class_softmax)																														\
		{																																		\
			if(in_col != 8)																														\
				return;																															\
			vmax = tab[i];																														\
			for(j = 1; j < nb_class; j++)																										\
				if(tab[i+j*flat_offset] > vmax)																									\
					vmax = tab[i+j*flat_offset];																								\
																																				\
			for(j = 0; j < nb_class; j++)																										\
			{																																	\
				tab[i+j*flat_offset] = exp_fct((tab[i+j*flat_offset]-vmax));																	\
				normal += (float)tab[i+j*flat_offset];																							\
			}																																	\
																																				\
			for(j = 0; j < nb_class; j++)																										\
				tab[i+j*flat_offset] = (type)((float)tab[i+j*flat_offset]/normal);																\
		}																																		\
		else																																	\
		{																																		\
			tab[i] = -(type)sm_tab[4][0]*tab[i];																								\
			if(tab[i] > (type)sm_tab[4][1])																										\
				tab[i] = (type)sm_tab[4][1];																									\
			else if(tab[i] < (type)sm_tab[4][2])																								\
				tab[i] = (type)sm_tab[4][2];																									\
			tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																								\
		}																																		\
		return;																																	\
	}																																			\
																																				\
	/*Additional parameters (regression)*/																										\
	if(in_col >= 8+nb_class)																													\
	{																																			\
		tab[i] = (type)sm_tab[5][0]*tab[i];																										\
		if(tab[i] > (type)sm_tab[5][1])																											\
			tab[i] = (type)sm_tab[5][1];																										\
		else if(tab[i] < (type)(sm_tab[5][2]))																									\
			tab[i] = (sm_tab[5][2]);																											\
		return;																																	\
	}																																			\
}

__device__ float gpu_IoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


__device__ float gpu_GIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}

//order: xmin, ymin, zmin, xmax, ymax, zmax
// Take into acount the distance in IoU, useful for crowded images
// or to put the emhasis on positionning in objectness score
__device__ float gpu_DIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
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

// Distance penalty is less in this version for a given distance between boxes
// More suited for usual VOC images, or sparse astro images
__device__ float gpu_DIoU2_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = ((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = (enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}


typedef float(*pointFunction_gpu_IoU)(float*, float*); 
__device__ pointFunction_gpu_IoU device_gpu_IoU_fct  = gpu_IoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_GIoU_fct = gpu_GIoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_DIoU_fct = gpu_DIoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_DIoU2_fct = gpu_DIoU2_fct;


#define YOLO_deriv_error_kernel(name, type)																										\
__global__ void YOLO_deriv_error_kernel_##name																									\
	(void *i_delta_o, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size, 										\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, size_t size, float TC_scale_factor, size_t nb_im_iter)						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *delta_o = (type*) i_delta_o;																											\
	type *output  = (type*) i_output;																											\
	type *target  = (type*) i_target;																											\
																																				\
	/* Define many "shorts" for y_param content to enhance code redeability*/																	\
	int nb_box                      = y_param.nb_box;																							\
	int nb_class                    = y_param.nb_class;																							\
	int nb_param                    = y_param.nb_param; 																						\
	int regret_based                = y_param.regret_based;																						\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim                     = y_param.fit_dim;																							\
	size_t rand_startup             = y_param.rand_startup;																						\
	float rand_prob_best_box_assoc  = y_param.rand_prob_best_box_assoc;																			\
	float rand_prob                 = y_param.rand_prob;																						\
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;																			\
	int class_softmax               = y_param.class_softmax;																					\
	int diff_flag                   = y_param.diff_flag;																						\
	int prior_dist_type             = y_param.prior_dist_type;																					\
																																				\
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
																																				\
	float *prior_size            = y_param.prior_size;																							\
	int   *cell_size             = y_param.cell_size;																							\
	float *param_ind_scale       = y_param.param_ind_scale;																						\
	float *lambda_noobj_prior    = y_param.noobj_prob_prior;																					\
	float **sm_tab               = y_param.slopes_and_maxes_tab;																				\
	int   *in_cell_target_id     = y_param.in_cell_target_id;																					\
	float *IoU_table             = y_param.IoU_table;																							\
	float *dist_prior            = y_param.dist_prior;																							\
	int   *prior_assigned_target = y_param.prior_assigned_target;																				\
	int   *box_ids               = y_param.box_ids;																								\
	int   *box_locked            = y_param.box_locked;																							\
	float *box_in_pix            = y_param.box_in_pix;																							\
	void  *block_state           = y_param.block_state;																							\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];											\
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];											\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];											\
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];											\
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2]; 									\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
	float positive_class_target, negative_class_target;																							\
																																				\
	int j, k, l, l_o, l_t;																														\
	size_t rand_seed, f_offset, c_total_nb_area, total_area_and_cell_offset;																	\
	int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0, rand_pos, temp_id;														\
	int nb_in_cell, id_in_cell_offset, targ_diff_flag = 0;																						\
	int resp_box, resp_targ, candidate_box, assigned_count;																						\
	float best_dist, c_dist, max_IoU, current_IoU, l_scale, l_targ;																				\
	float best_target_IoU, second_target_IoU, effective_second_IoU, current_regret, best_regret;												\
	size_t cell_pos[3], c_nb_area[3], obj_c[3];																									\
	float *c_box_in_pix, *c_prior_size;																											\
	float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];																				\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;																\
	c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];																					\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size * c_batch;																										\
	f_offset = size;																															\
	output_offset = 8+nb_class+nb_param;																										\
	target_offset = 7+nb_param+diff_flag;																										\
																																				\
	rand_seed = i;																																\
	i = i % flat_output_size;																													\
	cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);																								\
	cell_pos[1] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];																		\
	cell_pos[0] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];																		\
																																				\
	total_area_and_cell_offset = c_total_nb_area * c_batch																						\
		+ cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];														\
																																				\
	delta_o += total_area_and_cell_offset;																										\
	output  += total_area_and_cell_offset;																										\
																																				\
	in_cell_target_id		+= total_area_and_cell_offset * y_param.max_nb_obj_per_image;														\
	IoU_table  				+= total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;												\
	dist_prior 				+= total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;												\
	prior_assigned_target 	+= total_area_and_cell_offset * nb_box;																				\
	box_ids 				+= total_area_and_cell_offset * nb_box;																				\
	box_locked 				+= total_area_and_cell_offset * nb_box;																				\
	box_in_pix 				+= total_area_and_cell_offset * 6 * nb_box;																			\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_class > 0)																															\
	{																																			\
		positive_class_target = y_param.label_smoothing/nb_class + 1.0f - y_param.label_smoothing;												\
		negative_class_target = y_param.label_smoothing/nb_class;																				\
	}																																			\
																																				\
	if(nb_obj_target == -1)	/*Used for class only detection pre-training (likely obsolete)*/													\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim; 																											\
	}																																			\
																																				\
	best_dist = 1000000;	/*Search for smallest prior id*/																					\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		box_ids[k] = k;																															\
		c_box_in_pix = box_in_pix + k*6;																										\
		c_prior_size = prior_size + k*3;																										\
		l_o = k*output_offset;																													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);														\
																																				\
		c_dist = c_prior_size[0]*c_prior_size[0] 																								\
			   + c_prior_size[1]*c_prior_size[1]																								\
			   + c_prior_size[2]*c_prior_size[2];																								\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*target_offset;																													\
		/* Search for targets that should be predicted by the current cell element */															\
		for(l = 0; l < 3; l++)																													\
			obj_c[l] = (size_t)( ((float)target[l_t+l+4] + (float)target[l_t+l+1])*0.5f / cell_size[l]);										\
																																				\
		if(obj_c[0] == cell_pos[0] && obj_c[1] == cell_pos[1] && obj_c[2] == cell_pos[2])														\
		{																																		\
			in_cell_target_id[nb_in_cell] = j;																									\
			nb_in_cell++;																														\
		}																																		\
	}																																			\
																																				\
	/* Flag all the "Good but not best boxes" for all detection units regardless of the target grid element */									\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		c_box_in_pix = box_in_pix+k*6;																											\
		for(l = 0; l < 6; l++)																													\
			out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];														\
																																				\
		max_IoU = -2.0f; resp_targ = -1;																										\
		for(j = 0; j < nb_obj_target; j++)																										\
		{																																		\
			l_t = j*target_offset;																												\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
			if(current_IoU > max_IoU)																											\
			{																																	\
				max_IoU = current_IoU;																											\
				resp_targ = j;																													\
			}																																	\
		}																																		\
		if(resp_targ >= 0 && max_IoU > good_IoU_lim)																							\
		{																																		\
			box_locked[k] = 1;																													\
			prior_assigned_target[k] = resp_targ;	/* Can be used to update class for GBNB associations if no other association */				\
		}																																		\
	}																																			\
																																				\
	/*Random association. Used during startup or at a given rand_prob rate*/																	\
	if(nb_im_iter <= rand_startup || curand_uniform(&(((curandState_t*)block_state)[rand_seed])) < rand_prob)									\
	{																																			\
		/* Random association is done by scrambling in_cell_target_id and box_ids, then associating in order*/									\
		for(k = 0; k+1 < nb_in_cell; k++)																										\
		{																																		\
			rand_pos = k + curand(&(((curandState_t*)block_state)[rand_seed])) % (nb_in_cell - k);												\
			temp_id = in_cell_target_id[k];																										\
			in_cell_target_id[k] = in_cell_target_id[rand_pos];																					\
			in_cell_target_id[rand_pos] = temp_id;																								\
		}																																		\
																																				\
		for(k = 0; k+1 < nb_box; k++)																											\
		{																																		\
			rand_pos = k + curand(&(((curandState_t*)block_state)[rand_seed])) % (nb_box - k);													\
			temp_id = box_ids[k];																												\
			box_ids[k] = box_ids[rand_pos];																										\
			box_ids[rand_pos] = temp_id;																										\
		}																																		\
																																				\
		assigned_count = 0;																														\
		for(k = 0; k < nb_in_cell && assigned_count < nb_box; k++)																				\
		{	/* Difficult target are included during random startup but ignored for occasional random regularization */							\
			if(diff_flag && nb_im_iter > rand_startup && (int)((float)target[in_cell_target_id[k]*target_offset+7+nb_param]) > 0)				\
				continue; 																														\
			prior_assigned_target[box_ids[assigned_count]] = in_cell_target_id[k];																\
			box_locked[box_ids[assigned_count]] = 2;																							\
			assigned_count++;																													\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		/* For all targets in cell compute the IoU against all predictions and distances to priors */											\
		for(j = 0; j < nb_in_cell; j++)																											\
		{																																		\
			id_in_cell_offset = j*nb_box;																										\
			l_t = in_cell_target_id[j]*target_offset;																							\
																																				\
			for(k = 0; k < nb_box; k++)																											\
			{																																	\
				c_box_in_pix = box_in_pix+k*6;																									\
				for(l = 0; l < 6; l++)																											\
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];												\
				for(l = 0; l < 6; l++)																											\
					targ_int[l] = target[l_t+1+l];																								\
				for(l = 0; l < 3; l++)																											\
					targ_size[l] = targ_int[l+3] - targ_int[l];																					\
																																				\
				current_IoU = y_param.c_IoU_fct(out_int, targ_int);																				\
				IoU_table[id_in_cell_offset + k] = current_IoU;																					\
																																				\
				c_prior_size = prior_size + k*3;																								\
				switch(prior_dist_type)																											\
				{																																\
					case DIST_IOU:	/* Compute a zero centered IoU */																			\
						for(l = 0; l < 6; l++)																									\
							out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];																\
						for(l = 0; l < 6; l++)																									\
							targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];																\
						dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);										\
						break;																													\
																																				\
					default:																													\
					case DIST_SIZE:																												\
						dist_prior[id_in_cell_offset + k] = 																					\
							 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])														\
							+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])														\
							+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]);														\
						break;																													\
																																				\
					case DIST_OFFSET:																											\
						for(l = 0; l < 3; l++)																									\
						{																														\
							obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																	\
							if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);									\
							else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);									\
							else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);							\
						}																														\
																																				\
						dist_prior[id_in_cell_offset + k] =	obj_in_offset[3]*obj_in_offset[3]													\
						                                  + obj_in_offset[4]*obj_in_offset[4]													\
						                                  + obj_in_offset[5]*obj_in_offset[5];													\
						break;																													\
				}																																\
			}																																	\
		}																																		\
																																				\
		/* Restrict association to the best theoretical prior at a given rand_prob_best_box_assoc rate*/										\
		if(curand_uniform(&(((curandState_t*)block_state)[rand_seed])) < rand_prob_best_box_assoc)												\
			strict_box_size_association = 1;																									\
																																				\
		for(j = 0; j < nb_in_cell; j++)																											\
		{																																		\
			id_in_cell_offset = j*nb_box;																										\
			l_t = in_cell_target_id[j]*target_offset;																							\
																																				\
			/* Restrict association to the smallest prior (or identical) if the target is too small */											\
			/* Smallest might not be part of the N closest prior, so it need to be handled beforehand */										\
			/* Smallest association is also considered to replace "single best prior association" for small targets*/							\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			if(    targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]																\
				&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]																\
				&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])																\
			{																																	\
				for(k = 0; k < nb_box; k++)																										\
					if( fabsf(prior_size[s_p_i*3+0] - prior_size[k*3+0]) > 0.0001f ||															\
						fabsf(prior_size[s_p_i*3+1] - prior_size[k*3+1]) > 0.0001f ||															\
						fabsf(prior_size[s_p_i*3+2] - prior_size[k*3+2]) > 0.0001f)																\
						dist_prior[id_in_cell_offset + k] = -2.0f;																				\
			}																																	\
			else if(strict_box_size_association > 0)																							\
			{																																	\
				/* Restrict association to the N best different theoretical priors (identical priors are considered as a single one)*/			\
				c_dist = -1.0f;																													\
				for(l = 0; l < strict_box_size_association; l++)																				\
				{																																\
					best_dist = 1000000.0f;	resp_box = -1;																						\
					for(k = 0; k < nb_box; k++)																									\
					{																															\
						if(dist_prior[id_in_cell_offset + k] > -1.9f && dist_prior[id_in_cell_offset + k] < best_dist 							\
								&& dist_prior[id_in_cell_offset + k] > c_dist + 0.0001f)														\
						{																														\
							best_dist = dist_prior[id_in_cell_offset + k];																		\
							resp_box = k;																										\
						}																														\
					}																															\
					if(resp_box == -1)																											\
						break;																													\
					c_dist = best_dist;																											\
				}																																\
																																				\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset + k] > c_dist + 0.0001f)																	\
						dist_prior[id_in_cell_offset + k] = -2.0f;																				\
			}																																	\
		}																																		\
																																				\
		/* Prediction aware association */																										\
		while(1)	 																															\
		{																																		\
			max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;																						\
			if(!regret_based) /* Legacy best match first association */																			\
			{																																	\
				for(j = 0; j < nb_in_cell; j++)																									\
				{																																\
					id_in_cell_offset = j*nb_box;																								\
					for(k = 0; k < nb_box; k++)																									\
						if(dist_prior[id_in_cell_offset + k] > -1.9f && IoU_table[id_in_cell_offset + k] > max_IoU)								\
						{																														\
							max_IoU = IoU_table[id_in_cell_offset + k];																			\
							resp_targ = j; resp_box = k;																						\
						}																														\
				}																																\
			}																																	\
			else /* Regret based approach -> try to find a more global good association looking at the effect on second best match */			\
			{	 /* Search the pair for which missing the association would be the most problematic (second best association would be bad) */	\
				best_regret = -1.0f;																											\
				for(j = 0; j < nb_in_cell; j++)																									\
				{																																\
					id_in_cell_offset = j*nb_box;																								\
					best_target_IoU = -2.0f; second_target_IoU = -2.0f; candidate_box = -1;														\
					for(k = 0; k < nb_box; k++)																									\
					{																															\
						if(dist_prior[id_in_cell_offset + k] <= -1.9f)																			\
							continue;																											\
						current_IoU = IoU_table[id_in_cell_offset + k];																			\
																																				\
						if(current_IoU > best_target_IoU)																						\
						{																														\
							second_target_IoU = best_target_IoU;																				\
							best_target_IoU   = current_IoU;																					\
							candidate_box     = k;																								\
						}																														\
						else if(current_IoU > second_target_IoU)																				\
							second_target_IoU = current_IoU;																					\
					}																															\
																																				\
					/* If all available predictions are bad, ignore the target. It will be handled by best prior instead. */					\
					if(candidate_box == -1 || best_target_IoU < low_IoU_best_box_assoc)															\
						continue;																												\
																																				\
					effective_second_IoU = second_target_IoU;																					\
					if(effective_second_IoU < low_IoU_best_box_assoc)																			\
						effective_second_IoU = low_IoU_best_box_assoc;																			\
																																				\
					current_regret = best_target_IoU - effective_second_IoU;																	\
					/* If identical regret (might append due to ceiling effects) prefer the best match */										\
					if(current_regret > best_regret + 0.0001f 																					\
						|| (fabsf(current_regret - best_regret) <= 0.0001f && best_target_IoU > max_IoU))										\
					{																															\
						best_regret = current_regret;																							\
						max_IoU     = best_target_IoU;																							\
						resp_targ   = j; 																										\
						resp_box    = candidate_box;																							\
					}																															\
				}																																\
			}																																	\
																																				\
			if(resp_box == -1 || max_IoU < low_IoU_best_box_assoc)																				\
			{																																	\
				best_dist = 1000000.0f;																											\
				resp_box  = -1;	resp_targ = -1;																									\
				for(j = 0; j < nb_in_cell; j++)																									\
				{																																\
					id_in_cell_offset = j*nb_box;																								\
					for(k = 0; k < nb_box; k++)																									\
						if(dist_prior[id_in_cell_offset + k] > -1.9f && dist_prior[id_in_cell_offset + k] < best_dist)							\
						{																														\
							best_dist = dist_prior[id_in_cell_offset + k];																		\
							resp_box = k; resp_targ = j;																						\
						}																														\
				}																																\
				if(resp_box == -1)																												\
					break;																														\
																																				\
				/* If multiple identical best priors are avalaible, */																			\
				/* they are seperated based on the IoU between the corresponding prediction and the target */									\
				max_IoU = IoU_table[resp_targ*nb_box + resp_box];																				\
				id_in_cell_offset = resp_targ*nb_box;																							\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset + k] > -1.9f && fabsf(dist_prior[id_in_cell_offset + k] - best_dist) < 0.0001f				\
						&& IoU_table[id_in_cell_offset + k] > max_IoU)																			\
						{																														\
							max_IoU = IoU_table[id_in_cell_offset + k];																			\
							resp_box = k;																										\
						}																														\
			}																																	\
			if(resp_box == -1)																													\
				break;																															\
			l_t = in_cell_target_id[resp_targ]*target_offset;																					\
																																				\
			/* Difficult target handling -> a difficult object must already be confidently detected to be set as an actual target */			\
			/* If not, it is removed from the array but not associated. Note that difficult objects still count during GBNB check, */			\
			/* so a good but uncertain detection most often lead to no gradient. */ 															\
			if(diff_flag && (int)((float)target[l_t+7+nb_param]) > 0 																			\
				&& (max_IoU < diff_IoU_lim || (float)output[(resp_box*output_offset+7)*f_offset] < diff_obj_lim))								\
			{																																	\
				dist_prior[resp_targ*nb_box + resp_box] = -2.0f;																				\
				continue;																														\
			}																																	\
			/* The found association is removed from the dist_prior table */																	\
			for(j = 0; j < nb_in_cell; j++)																										\
				dist_prior[j*nb_box + resp_box] = -2.0f;																						\
			for(k = 0; k < nb_box; k++)																											\
				dist_prior[resp_targ*nb_box + k] = -2.0f;																						\
																																				\
			prior_assigned_target[resp_box] = in_cell_target_id[resp_targ];																		\
			box_locked[resp_box] = 2;																											\
		}																																		\
	}																																			\
																																				\
	/*Use previous assigment to update all detection units*/																					\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		l_o = j*output_offset;																													\
		targ_diff_flag = 0;																														\
																																				\
		/*Set the all box vector to 0, and only modify if necessary. Vastly simplify the logic*/												\
		for(k = 0; k < 8+nb_class+nb_param; k++) 																								\
			delta_o[(l_o+k)*f_offset] = (type) 0.0f;																							\
																																				\
		if(box_locked[j] == 2)																													\
		{																																		\
			l_t = prior_assigned_target[j]*target_offset;																						\
			if(diff_flag)																														\
				targ_diff_flag = (int)((float)target[l_t+7+nb_param]);																			\
																																				\
			c_box_in_pix = box_in_pix + j*6;																									\
			for(l = 0; l < 6; l++)																												\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];													\
																																				\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			max_IoU = y_param.c_IoU_fct(out_int, targ_int);																						\
			if(max_IoU > 0.98f)																													\
				max_IoU = 0.98f;																												\
																																				\
			if(class_only_IoU > -1.9f)																											\
			    max_IoU = class_only_IoU;																										\
																																				\
			for(l = 0; l < 3; l++)																												\
				obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];							\
																																				\
			c_prior_size = prior_size + 3*j;																									\
			for(l = 0; l < 3; l++)																												\
			{																																	\
				obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																				\
				if(obj_in_offset[l+3] < size_min_sat)		obj_in_offset[l+3] = logf(size_min_sat);											\
				else if(obj_in_offset[l+3] > size_max_sat)	obj_in_offset[l+3] = logf(size_max_sat);											\
				else 										obj_in_offset[l+3] = logf(obj_in_offset[l+3]);										\
			}																																	\
		}																																		\
																																				\
		/***** POSITION LOSS *****/																												\
		if(fit_pos != -1 && box_locked[j] == 2 && class_only_IoU < -1.9 && (diff_flag == 0 || targ_diff_flag < 3))								\
		{																																		\
			for(k = 0; k < fit_dim; k++) /*delta_o = 0 above fit_dim */																			\
			{																																	\
				l_scale = TC_scale_factor*sm_tab[0][0]*coord_scale																				\
					* (float)output[(l_o+k)*f_offset]*(1.0f-(float)output[(l_o+k)*f_offset]);													\
				if(fit_pos == 1) l_targ = obj_in_offset[k];																						\
				else 			 l_targ = 0.5f;																									\
				delta_o[(l_o+k)*f_offset] = (type)(l_scale*((float)output[(l_o+k)*f_offset]-l_targ));											\
			}																																	\
		}																																		\
																																				\
		/*****  SIZE LOSS  *****/																												\
		if(fit_size != -1 && box_locked[j] == 2 && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))							\
		{																																		\
			l_scale = TC_scale_factor*sm_tab[1][0]*size_scale;																					\
			for(k = 0; k < fit_dim; k++) /*delta_o = 0 above fit_dim */																			\
			{																																	\
				if(fit_size == 1) l_targ = obj_in_offset[k+3];																					\
				else 			  l_targ = 0.0f;																								\
				delta_o[(l_o+k+3)*f_offset] = (type) (l_scale*((float)output[(l_o+k+3)*f_offset]-l_targ));										\
			}																																	\
		}																																		\
																																				\
		/*****  PROB LOSS  *****/																												\
		if(fit_prob != -1 && ((box_locked[j] == 2 && max_IoU > min_prob_IoU_lim) || box_locked[j] == 0))										\
		{																																		\
			l_scale = TC_scale_factor*sm_tab[2][0]*prob_scale																					\
				* (float)output[(l_o+6)*f_offset]*(1.0f-(float)output[(l_o+6)*f_offset]);														\
			if(box_locked[j] == 0)																												\
				l_scale *= lambda_noobj_prior[j];																								\
			if(fit_prob == 1)																													\
			{																																	\
				if(box_locked[j] == 2) 		l_targ = 0.98f;																						\
				else if(box_locked[j] == 0) l_targ = 0.02f;																						\
			}																																	\
			else							l_targ = 0.5f;																						\
			delta_o[(l_o+6)*f_offset] = (type) (l_scale*((float)output[(l_o+6)*f_offset]-l_targ));												\
		}																																		\
																																				\
		/*****  OBJ LOSS   *****/																												\
		if(fit_obj != -1 && ((box_locked[j] == 2 && max_IoU > min_obj_IoU_lim) || box_locked[j] == 0))											\
		{																																		\
			l_scale = TC_scale_factor*sm_tab[3][0]*obj_scale																					\
				* (float)output[(l_o+7)*f_offset]*(1.0f-(float)output[(l_o+7)*f_offset]);														\
			if(box_locked[j] == 0)																												\
				l_scale *= lambda_noobj_prior[j];																								\
			if(fit_obj == 1)																													\
			{																																	\
				if(box_locked[j] == 2) 		l_targ = (1.0+max_IoU)*0.5;																			\
				else if(box_locked[j] == 0) l_targ = 0.02f;																						\
			}																																	\
			else							l_targ = 0.5f;																						\
			delta_o[(l_o+7)*f_offset] = (type) (l_scale*((float)output[(l_o+7)*f_offset]-l_targ));												\
		}																																		\
																																				\
		/*****  CLASS LOSS *****/																												\
		if(fit_class != -1 && box_locked[j] == 2 && max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))						\
		{																																		\
			for(k = 0; k < nb_class; k++)																										\
			{																																	\
				l_scale = TC_scale_factor*class_scale;																							\
				if(!class_softmax)																												\
					l_scale *= sm_tab[4][0]*(float)output[(l_o+8+k)*f_offset]*(1.0f-(float)output[(l_o+8+k)*f_offset]);							\
				if(fit_class == 1)																												\
				{																																\
					if(k == (int) target[l_t]-1)	l_targ = positive_class_target;																\
					else 							l_targ = negative_class_target;																\
				}																																\
				else								l_targ = 0.5f;																				\
				delta_o[(l_o+8+k)*f_offset] = (type) (l_scale*((float)output[(l_o+8+k)*f_offset]-l_targ));										\
			}																																	\
		}																																		\
																																				\
		/*****  PARAM LOSS  *****/																												\
		if(fit_param != -1 && box_locked[j] == 2 && max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))						\
		{																																		\
			for(k = 0; k < nb_param; k++)																										\
			{																																	\
				l_scale = TC_scale_factor*sm_tab[5][0]*param_scale*param_ind_scale[k];															\
				if(fit_param ==1)	l_targ = (float)target[l_t+7+k];																			\
				else				l_targ = 0.5f;																								\
				delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (l_scale*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ));					\
			}																																	\
		}																																		\
	}																																			\
}


#define YOLO_error_kernel(name, type)																											\
__global__ void YOLO_error_kernel_##name																										\
	(float *output_error, void *i_output, void *i_target, size_t flat_target_size, size_t flat_output_size,										\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, size_t size)																\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *output = (type*)i_output;																												\
	type *target = (type*)i_target;																												\
																																				\
	int nb_box                      = y_param.nb_box;																							\
	int nb_class                    = y_param.nb_class;																							\
	int nb_param                    = y_param.nb_param;																							\
	int regret_based                = y_param.regret_based;																						\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim                     = y_param.fit_dim;																							\
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;																			\
	int class_softmax               = y_param.class_softmax;																					\
	int diff_flag                   = y_param.diff_flag;																						\
	int prior_dist_type             = y_param.prior_dist_type;																					\
	int error_type                  = y_param.error_type;																						\
																																				\
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
																																				\
	float *prior_size            = y_param.prior_size;																							\
	int   *cell_size             = y_param.cell_size;																							\
	float *param_ind_scale       = y_param.param_ind_scale;																						\
	float *lambda_noobj_prior    = y_param.noobj_prob_prior;																					\
	float **sm_tab               = y_param.slopes_and_maxes_tab;																				\
	float *IoU_monitor           = y_param.IoU_monitor;																							\
	int   *in_cell_target_id     = y_param.in_cell_target_id;																					\
	float *IoU_table             = y_param.IoU_table;																							\
	float *dist_prior            = y_param.dist_prior;																							\
	int   *prior_assigned_target = y_param.prior_assigned_target;																				\
	int   *box_locked            = y_param.box_locked;																							\
	float *box_in_pix            = y_param.box_in_pix;																							\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];											\
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];											\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];											\
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];											\
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2];										\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
	float positive_class_target, negative_class_target;																							\
																																				\
	int j, k, l, l_o, l_t;																														\
	size_t f_offset, c_total_nb_area, total_area_and_cell_offset;																				\
	int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;																		\
	int nb_in_cell, id_in_cell_offset, targ_diff_flag = 0;																						\
	int resp_box, resp_targ, candidate_box;																										\
	float best_dist, c_dist, max_IoU, current_IoU, l_scale, l_targ;																				\
	float best_target_IoU, second_target_IoU, effective_second_IoU, current_regret, best_regret;												\
	size_t cell_pos[3], c_nb_area[3], obj_c[3];																									\
	float *c_box_in_pix, *c_prior_size;																											\
	float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];																				\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;																\
	c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];																					\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size*c_batch;																											\
	f_offset = size;																															\
	output_offset = 8 + nb_class + nb_param;																									\
	target_offset = 7 + nb_param + diff_flag;																									\
																																				\
	i = i % flat_output_size;																													\
	cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);																								\
	cell_pos[1] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];																		\
	cell_pos[0] = (size_t)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];																		\
																																				\
	total_area_and_cell_offset = c_total_nb_area*c_batch																						\
	    + cell_pos[2]*c_nb_area[0]*c_nb_area[1]	+ cell_pos[1]*c_nb_area[0] + cell_pos[0];														\
																																				\
	output_error += total_area_and_cell_offset;																									\
	output += total_area_and_cell_offset;																										\
																																				\
	IoU_monitor           += 2*nb_box*total_area_and_cell_offset;																				\
	in_cell_target_id     += total_area_and_cell_offset*y_param.max_nb_obj_per_image;															\
	IoU_table             += total_area_and_cell_offset*y_param.max_nb_obj_per_image*nb_box;													\
	dist_prior            += total_area_and_cell_offset*y_param.max_nb_obj_per_image*nb_box;													\
	prior_assigned_target += total_area_and_cell_offset*nb_box;																					\
	box_locked            += total_area_and_cell_offset*nb_box;																					\
	box_in_pix            += total_area_and_cell_offset*6*nb_box;																				\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_class > 0)																															\
	{																																			\
		positive_class_target = y_param.label_smoothing/nb_class + 1.0f - y_param.label_smoothing;												\
		negative_class_target = y_param.label_smoothing/nb_class;																				\
	}																																			\
																																				\
	if(nb_obj_target == -1)																														\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim;																											\
	}																																			\
																																				\
	best_dist = 1000000.0f;																														\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		IoU_monitor[2*k] = -1.0f; IoU_monitor[2*k+1] = -1.0f;																					\
		c_box_in_pix = box_in_pix + 6*k;																										\
		c_prior_size = prior_size + 3*k;																										\
		l_o = k*output_offset;																													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l])*cell_size[l];														\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);														\
																																				\
		c_dist = c_prior_size[0]*c_prior_size[0]																								\
		       + c_prior_size[1]*c_prior_size[1]																								\
		       + c_prior_size[2]*c_prior_size[2];																								\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*target_offset;																													\
		for(l = 0; l < 3; l++)																													\
			obj_c[l] = (size_t)(((float)target[l_t+l+4] + (float)target[l_t+l+1])*0.5f/cell_size[l]);											\
		if(obj_c[0] == cell_pos[0] && obj_c[1] == cell_pos[1] && obj_c[2] == cell_pos[2])														\
		{																																		\
			in_cell_target_id[nb_in_cell] = j;																									\
			nb_in_cell++;																														\
		}																																		\
	}																																			\
																																				\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		c_box_in_pix = box_in_pix + 6*k;																										\
		for(l = 0; l < 6; l++)																													\
			out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];														\
																																				\
		max_IoU = -2.0f; resp_targ = -1;																										\
		for(j = 0; j < nb_obj_target; j++)																										\
		{																																		\
			l_t = j*target_offset;																												\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
			if(current_IoU > max_IoU)																											\
			{																																	\
				max_IoU = current_IoU;																											\
				resp_targ = j;																													\
			}																																	\
		}																																		\
		if(resp_targ >= 0 && max_IoU > good_IoU_lim)																							\
		{																																		\
			box_locked[k] = 1;																													\
			prior_assigned_target[k] = resp_targ;																								\
		}																																		\
	}																																			\
																																				\
	/* No random association in error computation (only used for deriv) */																		\
																																				\
	for(j = 0; j < nb_in_cell; j++)																												\
	{																																			\
		id_in_cell_offset = j*nb_box;																											\
		l_t = in_cell_target_id[j]*target_offset;																								\
																																				\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			c_box_in_pix = box_in_pix + 6*k;																									\
			for(l = 0; l < 6; l++)																												\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];													\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
			IoU_table[id_in_cell_offset+k] = current_IoU;																						\
																																				\
			if(error_type == ERR_NATURAL)																										\
			{																																	\
				dist_prior[id_in_cell_offset+k] = 0.0f;																							\
				continue;																														\
			}																																	\
			c_prior_size = prior_size + 3*k;																									\
			switch(prior_dist_type)																												\
			{																																	\
				case DIST_IOU:																													\
					for(l = 0; l < 6; l++)																										\
						out_int[l] = copysignf(0.5f, l-2.5f)*c_prior_size[l%3];																	\
					for(l = 0; l < 6; l++)																										\
						targ_int[l] = copysignf(0.5f, l-2.5f)*targ_size[l%3];																	\
					dist_prior[id_in_cell_offset+k] =  1.0f - y_param.c_IoU_fct(out_int, targ_int);												\
					break;																														\
																																				\
				default:																														\
				case DIST_SIZE:																													\
					dist_prior[id_in_cell_offset+k] =																							\
				          (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])															\
				        + (targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])															\
				        + (targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]);														\
					break;																														\
																																				\
				case DIST_OFFSET:																												\
					for(l = 0; l < 3; l++)																										\
					{																															\
						obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																		\
						if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);										\
						else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);										\
						else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);								\
					}																															\
					dist_prior[id_in_cell_offset+k] = obj_in_offset[3]*obj_in_offset[3]															\
				                                    + obj_in_offset[4]*obj_in_offset[4]															\
					                                + obj_in_offset[5]*obj_in_offset[5];														\
					break;																														\
			}																																	\
		}																																		\
	}																																			\
																																				\
	if(error_type == ERR_COMPLETE)																												\
	{																																			\
		for(j = 0; j < nb_in_cell; j++)																											\
		{																																		\
			id_in_cell_offset = j*nb_box;																										\
			l_t = in_cell_target_id[j]*target_offset;																							\
																																				\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			if(    targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]																\
			    && targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]																\
			    && targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])																\
			{																																	\
				for(k = 0; k < nb_box; k++)																										\
					if( fabsf(prior_size[s_p_i*3+0] - prior_size[k*3+0]) > 0.0001f ||															\
					    fabsf(prior_size[s_p_i*3+1] - prior_size[k*3+1]) > 0.0001f ||															\
					    fabsf(prior_size[s_p_i*3+2] - prior_size[k*3+2]) > 0.0001f)																\
						dist_prior[id_in_cell_offset+k] = -2.0f;																				\
			}																																	\
			else if(strict_box_size_association > 0)																							\
			{																																	\
				c_dist = -1.0f;																													\
				for(l = 0; l < strict_box_size_association; l++)																				\
				{																																\
					best_dist = 1000000.0f;																										\
					resp_box = -1;																												\
					for(k = 0; k < nb_box; k++)																									\
					{																															\
						if(dist_prior[id_in_cell_offset+k] > -1.9f																				\
						    && dist_prior[id_in_cell_offset+k] < best_dist																		\
						    && dist_prior[id_in_cell_offset+k] > c_dist + 0.0001f)																\
						{																														\
							best_dist = dist_prior[id_in_cell_offset+k];																		\
							resp_box = k;																										\
						}																														\
					}																															\
					if(resp_box == -1)																											\
						break;																													\
					c_dist = best_dist;																											\
				}																																\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset+k] > c_dist + 0.0001f)																		\
						dist_prior[id_in_cell_offset+k] = -2.0f;																				\
			}																																	\
		}																																		\
	}																																			\
																																				\
	while(1)																																	\
	{																																			\
		max_IoU = -2.0f; resp_box = -1; resp_targ = -1;																							\
		if(error_type == ERR_NATURAL || !regret_based)																							\
		{																																		\
			for(j = 0; j < nb_in_cell; j++)																										\
			{																																	\
				id_in_cell_offset = j*nb_box;																									\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset+k] > -1.9f																					\
					    && IoU_table[id_in_cell_offset+k] > max_IoU)																			\
					{																															\
						max_IoU = IoU_table[id_in_cell_offset+k];																				\
						resp_targ = j;																											\
						resp_box = k;																											\
					}																															\
			}																																	\
		}																																		\
		else																																	\
		{																																		\
			best_regret = -1.0f;																												\
			for(j = 0; j < nb_in_cell; j++)																										\
			{																																	\
				id_in_cell_offset = j*nb_box;																									\
				best_target_IoU = -2.0f; second_target_IoU = -2.0f; candidate_box = -1;															\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					if(dist_prior[id_in_cell_offset+k] <= -1.9f)																				\
						continue;																												\
					current_IoU = IoU_table[id_in_cell_offset+k];																				\
																																				\
					if(current_IoU > best_target_IoU)																							\
					{																															\
						second_target_IoU = best_target_IoU;																					\
						best_target_IoU = current_IoU;																							\
						candidate_box = k;																										\
					}																															\
					else if(current_IoU > second_target_IoU)																					\
						second_target_IoU = current_IoU;																						\
				}																																\
																																				\
				if(candidate_box == -1 || best_target_IoU < low_IoU_best_box_assoc)																\
					continue;																													\
																																				\
				effective_second_IoU = second_target_IoU;																						\
				if(effective_second_IoU < low_IoU_best_box_assoc)																				\
					effective_second_IoU = low_IoU_best_box_assoc;																				\
																																				\
				current_regret = best_target_IoU - effective_second_IoU;																		\
				if(current_regret > best_regret + 0.0001f																						\
				    || (fabsf(current_regret-best_regret) <= 0.0001f																			\
				        && best_target_IoU > max_IoU))																							\
				{																																\
					best_regret = current_regret;																								\
					max_IoU     = best_target_IoU;																								\
					resp_targ   = j; 																											\
					resp_box    = candidate_box;																								\
				}																																\
			}																																	\
		}																																		\
		if(error_type == ERR_COMPLETE  && (resp_box == -1 || max_IoU < low_IoU_best_box_assoc))													\
		{																																		\
			best_dist = 1000000.0f;																												\
			resp_box = -1; resp_targ = -1;																										\
			for(j = 0; j < nb_in_cell; j++)																										\
			{																																	\
				id_in_cell_offset = j*nb_box;																									\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset+k] > -1.9f																					\
					    && dist_prior[id_in_cell_offset+k] < best_dist)																			\
					{																															\
						best_dist = dist_prior[id_in_cell_offset+k];																			\
						resp_box = k;																											\
						resp_targ = j;																											\
					}																															\
			}																																	\
			if(resp_box == -1)																													\
				break;																															\
																																				\
			id_in_cell_offset = resp_targ*nb_box;																								\
			max_IoU = IoU_table[id_in_cell_offset+resp_box];																					\
			for(k = 0; k < nb_box; k++)																											\
				if(dist_prior[id_in_cell_offset+k] > -1.9f && fabsf(dist_prior[id_in_cell_offset+k]-best_dist) < 0.0001f						\
				    && IoU_table[id_in_cell_offset+k] > max_IoU)																				\
				{																																\
					max_IoU = IoU_table[id_in_cell_offset+k];																					\
					resp_box = k;																												\
				}																																\
		}																																		\
		if(resp_box == -1)																														\
			break;																																\
		l_t = in_cell_target_id[resp_targ]*target_offset;																						\
																																				\
		if(error_type == ERR_COMPLETE && diff_flag && (int)((float)target[l_t+7+nb_param]) > 0													\
		    && (max_IoU < diff_IoU_lim || (float)output[(resp_box*output_offset+7)*f_offset] < diff_obj_lim))									\
		{																																		\
			dist_prior[resp_targ*nb_box+resp_box] = -2.0f;																						\
			continue;																															\
		}																																		\
																																				\
		for(j = 0; j < nb_in_cell; j++)																											\
			dist_prior[j*nb_box+resp_box] = -2.0f;																								\
		for(k = 0; k < nb_box; k++)																												\
			dist_prior[resp_targ*nb_box+k] = -2.0f;																								\
																																				\
		prior_assigned_target[resp_box] = in_cell_target_id[resp_targ];																			\
		box_locked[resp_box] = 2;																												\
	}																																			\
																																				\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		l_o = j*output_offset;																													\
		targ_diff_flag = 0;																														\
																																				\
		for(k = 0; k < output_offset; k++)																										\
			output_error[(l_o+k)*f_offset] = 0.0f;																								\
																																				\
		if(box_locked[j] == 2)																													\
		{																																		\
			l_t = prior_assigned_target[j]*target_offset;																						\
			if(diff_flag)																														\
				targ_diff_flag = (int)((float)target[l_t+7+nb_param]);																			\
																																				\
			c_box_in_pix = box_in_pix + 6*j;																									\
			for(l = 0; l < 6; l++)																												\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f, l-2.5f)*c_box_in_pix[3+l%3];													\
																																				\
			for(l = 0; l < 6; l++)																												\
				targ_int[l] = target[l_t+1+l];																									\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			max_IoU = y_param.c_IoU_fct(out_int, targ_int);																						\
			if(max_IoU > 0.98f)																													\
				max_IoU = 0.98f;																												\
																																				\
			if(class_only_IoU > -1.9f)																											\
				max_IoU = class_only_IoU;																										\
																																				\
			IoU_monitor[2*j] = (float)output[(l_o+7)*f_offset];																					\
			IoU_monitor[2*j+1] = max_IoU;																										\
																																				\
			for(l = 0; l < 3; l++)																												\
				obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];							\
																																				\
			c_prior_size = prior_size + 3*j;																									\
			for(l = 0; l < 3; l++)																												\
			{																																	\
				obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																				\
				if(obj_in_offset[l+3] < size_min_sat)      obj_in_offset[l+3] = logf(size_min_sat);												\
				else if(obj_in_offset[l+3] > size_max_sat) obj_in_offset[l+3] = logf(size_max_sat);												\
				else                                       obj_in_offset[l+3] = logf(obj_in_offset[l+3]);										\
			}																																	\
		}																																		\
																																				\
		if(fit_pos != -1 && box_locked[j] == 2 && class_only_IoU < -1.9f && 																	\
		    (error_type == ERR_NATURAL || diff_flag == 0 || targ_diff_flag < 3))																\
		{																																		\
			for(k = 0; k < fit_dim; k++)																										\
			{																																	\
				if(fit_pos == 1) l_targ = obj_in_offset[k];																						\
				else             l_targ = 0.5f;																									\
				output_error[(l_o+k)*f_offset] = 0.5f*coord_scale																				\
					*((float)output[(l_o+k)*f_offset] - l_targ)																					\
					*((float)output[(l_o+k)*f_offset] - l_targ);																				\
			}																																	\
		}																																		\
																																				\
		if(fit_size != -1 && box_locked[j] == 2	&& class_only_IoU < -1.9f && 																	\
		    (error_type == ERR_NATURAL || diff_flag == 0 || targ_diff_flag < 3))																\
		{																																		\
			for(k = 0; k < fit_dim; k++)																										\
			{																																	\
				if(fit_size == 1) l_targ = obj_in_offset[k+3];																					\
				else              l_targ = 0.0f;																								\
				output_error[(l_o+k+3)*f_offset] =  0.5f*size_scale																				\
					*((float)output[(l_o+k+3)*f_offset] - l_targ)																				\
					*((float)output[(l_o+k+3)*f_offset] - l_targ);																				\
			}																																	\
		}																																		\
																																				\
		if(fit_prob != -1 && ((box_locked[j] == 2 && (error_type == ERR_NATURAL || max_IoU > min_prob_IoU_lim)) || box_locked[j] == 0))			\
		{																																		\
			if(fit_prob == 1)																													\
			{																																	\
				if(box_locked[j] == 2) l_targ = 0.98f;																							\
				else                   l_targ = 0.02f;																							\
			}																																	\
			else																																\
				l_targ = 0.5f;																													\
			l_scale = prob_scale;																												\
			if(box_locked[j] == 0)																												\
				l_scale *= lambda_noobj_prior[j];																								\
			output_error[(l_o+6)*f_offset] =  0.5f*l_scale																						\
				*((float)output[(l_o+6)*f_offset] - l_targ)																						\
				*((float)output[(l_o+6)*f_offset] - l_targ);																					\
		}																																		\
																																				\
		if(fit_obj != -1 && ((box_locked[j] == 2 && (error_type == ERR_NATURAL || max_IoU > min_obj_IoU_lim)) || box_locked[j] == 0))			\
		{																																		\
			if(fit_obj == 1)																													\
			{																																	\
				if(box_locked[j] == 2) l_targ = (1.0f+max_IoU)*0.5f;																			\
				else                   l_targ = 0.02f;																							\
			}																																	\
			else																																\
				l_targ = 0.5f;																													\
			l_scale = obj_scale;																												\
			if(box_locked[j] == 0)																												\
				l_scale *= lambda_noobj_prior[j];																								\
			output_error[(l_o+7)*f_offset] =  0.5f*l_scale																						\
				*((float)output[(l_o+7)*f_offset] - l_targ)																						\
				*((float)output[(l_o+7)*f_offset] - l_targ);																					\
		}																																		\
		if(fit_class != -1 && box_locked[j] == 2 && 																							\
			(error_type == ERR_NATURAL|| (max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))))								\
		{																																		\
			if(!(fit_class == 0 && class_softmax)) /* else output_error must remain at 0*/														\
			{																																	\
				for(k = 0; k < nb_class; k++)																									\
				{																																\
					if(fit_class == 1)																											\
					{																															\
						if(k == (int)target[l_t]-1) l_targ = positive_class_target;																\
						else                        l_targ = negative_class_target;																\
					}																															\
					else																														\
						l_targ = 0.5f;																											\
					if(class_softmax)																											\
					{																															\
						if((float)output[(l_o+8+k)*f_offset] < 0.0000001f)																		\
							output_error[(l_o+8+k)*f_offset] = -class_scale*l_targ*logf(0.0000001f);											\
						else																													\
							output_error[(l_o+8+k)*f_offset] = -class_scale*l_targ*logf((float)output[(l_o+8+k)*f_offset]);						\
					}																															\
					else																														\
					{																															\
						output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																		\
							*((float)output[(l_o+8+k)*f_offset] - l_targ)																		\
							*((float)output[(l_o+8+k)*f_offset] - l_targ);																		\
					}																															\
				}																																\
			}																																	\
		}																																		\
																																				\
		if(fit_param != -1 && box_locked[j] == 2 && 																							\
		    (error_type == ERR_NATURAL || (max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))))								\
		{																																		\
			for(k = 0; k < nb_param; k++)																										\
			{																																	\
				if(fit_param == 1) l_targ = (float)target[l_t+7+k];																				\
				else               l_targ = 0.5f;																								\
				output_error[(l_o+8+nb_class+k)*f_offset] =	0.5f*param_ind_scale[k]*param_scale													\
					*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ)																		\
					*((float)output[(l_o+8+nb_class+k)*f_offset] - l_targ);    																	\
			}																																	\
		}																																		\
	}																																			\
}


#define typed_cuda_activ_fct_association(name)																									\
void typed_cuda_activ_fct_association_##name(network *net)																						\
{																																				\
	net->cu_inst.cu_linear_activ_fcts.batch_edge_fct = linear_batch_edge_kernel_##name;															\
	net->cu_inst.cu_linear_activ_fcts.bias_deriv_fct = linear_bias_deriv_kernel_##name;															\
	net->cu_inst.cu_linear_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;										\
	net->cu_inst.cu_linear_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name;													\
																																				\
	net->cu_inst.cu_ReLU_activ_fcts.activ_fct = ReLU_activation_kernel_##name;																	\
	net->cu_inst.cu_ReLU_activ_fcts.deriv_fct = ReLU_deriv_kernel_##name;																		\
	net->cu_inst.cu_ReLU_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;										\
	net->cu_inst.cu_ReLU_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name; 													\
																																				\
	net->cu_inst.cu_logistic_activ_fcts.activ_fct = logistic_activation_kernel_##name;															\
	net->cu_inst.cu_logistic_activ_fcts.deriv_fct = logistic_deriv_kernel_##name;																\
	net->cu_inst.cu_logistic_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;									\
	net->cu_inst.cu_logistic_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name;												\
																																				\
	net->cu_inst.cu_softmax_activ_fcts.activ_fct = softmax_activation_kernel_##name;															\
	net->cu_inst.cu_softmax_activ_fcts.deriv_output_error_fct = cross_entropy_deriv_output_error_kernel_##name;									\
	net->cu_inst.cu_softmax_activ_fcts.output_error_fct = cross_entropy_output_error_kernel_##name;												\
																																				\
	net->cu_inst.cu_YOLO_activ_fcts.activ_fct = YOLO_activation_kernel_##name;																	\
	net->cu_inst.cu_YOLO_activ_fcts.deriv_output_error_fct = YOLO_deriv_error_kernel_##name;													\
	net->cu_inst.cu_YOLO_activ_fcts.output_error_fct = YOLO_error_kernel_##name;																\
																																				\
}


linear_batch_edge_kernel(FP32, float);
linear_bias_deriv_kernel(FP32, float);
ReLU_activation_kernel(FP32, float);
ReLU_deriv_kernel(FP32, float);
quadratic_deriv_output_error_kernel(FP32, float);
quadratic_output_error_kernel(FP32, float);
logistic_activation_kernel(FP32, float, expf);
logistic_deriv_kernel(FP32, float);
softmax_activation_kernel(FP32, float, expf);
cross_entropy_deriv_output_error_kernel(FP32, float);
cross_entropy_output_error_kernel(FP32, float);
YOLO_activation_kernel(FP32, float, expf);
YOLO_deriv_error_kernel(FP32, float);
YOLO_error_kernel(FP32, float);
typed_cuda_activ_fct_association(FP32);


#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
linear_batch_edge_kernel(FP16, half);
linear_bias_deriv_kernel(FP16, half);
ReLU_activation_kernel(FP16, half);
ReLU_deriv_kernel(FP16, half);
quadratic_deriv_output_error_kernel(FP16, half);
quadratic_output_error_kernel(FP16, half);
logistic_activation_kernel(FP16, half, expf);
logistic_deriv_kernel(FP16, half);
softmax_activation_kernel(FP16, half, expf);
cross_entropy_deriv_output_error_kernel(FP16, half);
cross_entropy_output_error_kernel(FP16, half);
YOLO_activation_kernel(FP16, half, expf);
YOLO_deriv_error_kernel(FP16, half);
YOLO_error_kernel(FP16, half);
typed_cuda_activ_fct_association(FP16);
#endif


#if defined(GEN_AMPERE) 
linear_batch_edge_kernel(BF16, nv_bfloat16);
linear_bias_deriv_kernel(BF16, nv_bfloat16);
ReLU_activation_kernel(BF16, nv_bfloat16);
ReLU_deriv_kernel(BF16, nv_bfloat16);
quadratic_deriv_output_error_kernel(BF16, nv_bfloat16);
quadratic_output_error_kernel(BF16, nv_bfloat16);
logistic_activation_kernel(BF16, nv_bfloat16, expf);
logistic_deriv_kernel(BF16, nv_bfloat16);
softmax_activation_kernel(BF16, nv_bfloat16, expf);
cross_entropy_deriv_output_error_kernel(BF16, nv_bfloat16);
cross_entropy_output_error_kernel(BF16, nv_bfloat16);
YOLO_activation_kernel(BF16, nv_bfloat16, expf);
YOLO_deriv_error_kernel(BF16, nv_bfloat16);
YOLO_error_kernel(BF16, nv_bfloat16);
typed_cuda_activ_fct_association(BF16);
#endif



//#####################################################
//		 Linear activation related functions
//#####################################################

void cuda_linear_activation(layer *current)
{
	size_t nb_features;
	network *net = current->c_network;
	
	if(net->length >= net->batch_size)
		return;
	
	if(current->a_biased_dim > current->a_dim)
	{
		cu_blocks = ((net->batch_size - net->length) * current->a_biased_dim + cu_threads - 1) / cu_threads;
		net->cu_inst.cu_linear_activ_fcts.batch_edge_fct<<<cu_blocks, cu_threads>>>(
			current->output, current->a_dim, current->a_biased_dim,
			current->a_offset, net->length, net->batch_size);
	}
	else
	{
		nb_features = current->a_size / (current->a_dim * current->a_offset);
		dim3 numBlocks(((net->batch_size - net->length) * current->a_dim + cu_threads - 1) / cu_threads, nb_features);
		net->cu_inst.cu_linear_activ_fcts.batch_edge_fct<<<numBlocks, cu_threads>>>(
			current->output, current->a_dim, current->a_biased_dim,
			current->a_offset, net->length, net->batch_size);
	}
}


void cuda_linear_deriv(layer *current)
{
	size_t nb_features;
	network *net = current->c_network;
	
	//FLAT layers must remove the extra bias gradient
	if(current->a_biased_dim > current->a_dim)
	{
		cu_blocks = (net->batch_size + cu_threads - 1) / cu_threads;
		net->cu_inst.cu_linear_activ_fcts.bias_deriv_fct<<<cu_blocks, cu_threads>>>(
			current->delta_o, current->a_biased_dim,
			current->a_dim, net->batch_size);
	}
	
	if(net->length >= net->batch_size)
		return;
	
	if(current->a_biased_dim > current->a_dim)
	{
		cu_blocks = ((net->batch_size - net->length) * current->a_biased_dim + cu_threads - 1) / cu_threads;
		net->cu_inst.cu_linear_activ_fcts.batch_edge_fct<<<cu_blocks, cu_threads>>>(
			current->delta_o, current->a_dim, current->a_biased_dim,
			current->a_offset, net->length, net->batch_size);
	}
	else
	{
		nb_features = current->a_size / (current->a_dim * current->a_offset);
		dim3 numBlocks(((net->batch_size - net->length) * current->a_dim + cu_threads - 1) / cu_threads, nb_features);
		net->cu_inst.cu_linear_activ_fcts.batch_edge_fct<<<numBlocks, cu_threads>>>(
			current->delta_o, current->a_dim, current->a_biased_dim,
			current->a_offset, net->length, net->batch_size);
	}
}


void cuda_linear_deriv_output_error(layer *current)
{	
	cu_blocks = ( current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, 
		current->a_size, current->c_network->TC_scale_factor);
}


void cuda_linear_output_error(layer *current)
{	
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->dyn_target[0], 
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


//#####################################################
//		 ReLU activation related functions
//#####################################################

void cuda_ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, current->a_dim, current->a_biased_dim, current->a_offset, param->saturation, 
		param->leaking_factor, current->c_network->length, current->a_size);
}


void cuda_ReLU_deriv(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->a_dim, current->a_biased_dim, current->a_offset, 
		param->saturation, param->leaking_factor, current->c_network->length, current->a_size);
}


void cuda_ReLU_deriv_output_error(layer* current)
{
	cu_blocks = ( current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->dyn_target[0], current->a_dim, current->a_biased_dim,
		current->a_offset, current->c_network->length, current->a_size, current->c_network->TC_scale_factor);
}


void cuda_ReLU_output_error(layer* current)
{
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->dyn_target[0], 
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


//#####################################################
//		 Logistic activation related functions
//#####################################################

void cuda_logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;

	current->c_network->cu_inst.cu_logistic_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->beta, param->saturation, current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void cuda_logistic_deriv(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, param->beta, current->a_dim, 
		current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}


void cuda_logistic_deriv_output_error(layer* current)
{
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->dyn_target[0], current->a_dim, current->a_biased_dim, 
		current->a_offset, current->c_network->length, current->a_size, current->c_network->TC_scale_factor);
}


void cuda_logistic_output_error(layer* current)
{
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length, current->a_size);
}

//#####################################################
//		 Softmax activation related functions
//#####################################################

void cuda_softmax_activation(layer *current)
{
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, current->a_dim, current->a_biased_dim, current->a_offset, 
		current->c_network->length, current->c_network->batch_size, current->a_size);
}


void cuda_softmax_deriv(layer *current)
{
	//Empty on purpose
	//Only the last layer can be softmax activated with a cross entropy error
	//In this case, activation derivation is handled by deriv output_error directly (next function).
}


void cuda_softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->dyn_target[0],
		current->a_dim, current->a_biased_dim, current->a_offset, current->c_network->length,
		current->a_size, current->c_network->TC_scale_factor);
}


void cuda_softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, 
		current->c_network->dyn_target[0], current->a_dim, current->a_biased_dim, current->a_offset, 
		current->c_network->length, current->a_size);
}

//#####################################################
//		 YOLO activation related functions
//#####################################################

void cuda_YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	cu_blocks = (current->c_network->out_size *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, current->a_dim * current->c_network->batch_size,
		current->a_biased_dim * current->c_network->batch_size, *a_param, current->a_size, a_param->class_softmax);
}


void cuda_YOLO_deriv(layer *current)
{
	//Empty on purpose
	//Only the last layer can be YOLO activated with a dedicated error
	//In this case, activation derivation is handled by deriv output_error directly (next function).
}


void cuda_YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	cu_blocks = (current->a_dim * current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->dyn_target[0], a_param->flat_target_size, 
		current->a_dim, current->output_dim[0], current->output_dim[1], current->output_dim[2], 
		*a_param, current->a_dim * current->c_network->batch_size, 
		current->c_network->TC_scale_factor, current->c_network->iter * current->c_network->train.size);
}


void cuda_YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	cu_blocks = (current->a_dim * current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->dyn_target[0], a_param->flat_target_size, 
		current->a_dim, current->output_dim[0], current->output_dim[1], current->output_dim[2], 
		*a_param, current->a_dim * current->c_network->batch_size);
}


void cuda_YOLO_activ_init(layer *current)
{
	float *temp_tab, *temp_tab2, **temp_tab3;
	
	size_t nb_area_flat;

	yolo_param* a_param = (yolo_param*)current->activ_param;
	
	nb_area_flat = current->a_dim;
	
	switch(a_param->IoU_type)
	{
		case IOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_IoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		default:
		case GIOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_GIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		case DIOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_DIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
		
		case DIOU2:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_DIoU2_fct, sizeof(pointFunction_gpu_IoU));
			break;
	}
	
	cuda_convert_table_FP32((void**)&(a_param->prior_size), a_param->nb_box * 3, 1);
	cuda_convert_table_FP32((void**)&(a_param->noobj_prob_prior), a_param->nb_box, 1);
	cuda_convert_table_FP32((void**)&(a_param->scale_tab), 6, 1);
	
	temp_tab = a_param->slopes_and_maxes_tab[0];
	cudaMalloc(&temp_tab2, 6 * 3 * sizeof(float));
	cudaMemcpy(temp_tab2, temp_tab, 6 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 6; i++)
		a_param->slopes_and_maxes_tab[i] = &temp_tab2[i*3];
	temp_tab3 = a_param->slopes_and_maxes_tab;
	cudaMalloc(&(a_param->slopes_and_maxes_tab), 6 * sizeof(float*));
	cudaMemcpy(a_param->slopes_and_maxes_tab, temp_tab3, 6 * sizeof(float*), cudaMemcpyHostToDevice);
	
	cuda_convert_table_FP32((void**)&(a_param->param_ind_scale), a_param->nb_param, 1);
	cuda_convert_table_FP32((void**)&(a_param->IoU_limits), 8, 1);
	cuda_convert_table_int(&(a_param->fit_parts), 6, 1);
	
	cudaMalloc((void**)(&(a_param->block_state)), current->a_size * sizeof(curandState_t));
	cu_blocks = (current->a_size + cu_threads - 1) / cu_threads;
	init_block_state<<< cu_blocks, cu_threads>>>(time(NULL),(curandState_t*)(a_param->block_state), current->a_size);
	
	cuda_convert_table_int(&(a_param->cell_size), 3, 0);
	cuda_convert_table_FP32((void**)&(a_param->IoU_monitor),
		2 *a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->IoU_table),
		a_param->max_nb_obj_per_image * a_param->nb_box 
		* current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->dist_prior),
		a_param->max_nb_obj_per_image * a_param->nb_box 
		* current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->in_cell_target_id),
		a_param->max_nb_obj_per_image * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_int(&(a_param->prior_assigned_target),
		a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_int(&(a_param->box_ids),
		a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_int(&(a_param->box_locked),
		a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->box_in_pix),
		6 * a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
}


void cuda_free_yolo_activ_param(layer *current)
{
	yolo_param* a_param = (yolo_param*)current->activ_param;
	float **temp_tab;
	
	cudaFree(a_param->prior_size);
	cudaFree(a_param->noobj_prob_prior);
	cudaFree(a_param->scale_tab);
	
	temp_tab = (float**) malloc(6*sizeof(float*));
	cudaMemcpy(temp_tab, a_param->slopes_and_maxes_tab, 6 * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(temp_tab[0]);
	free(temp_tab);
	cudaFree(a_param->slopes_and_maxes_tab);
	
	cudaFree(a_param->param_ind_scale);
	cudaFree(a_param->IoU_limits);
	cudaFree(a_param->fit_parts);
	cudaFree(a_param->block_state);
	
	cudaFree(a_param->cell_size);
	cudaFree(a_param->IoU_monitor);
	cudaFree(a_param->IoU_table);
	cudaFree(a_param->prior_assigned_target);
	cudaFree(a_param->dist_prior);
	cudaFree(a_param->box_ids);
	cudaFree(a_param->box_locked);
	cudaFree(a_param->box_in_pix);
}


//#####################################################
//		 GENERAL FUNCTION ASSOCIATIONS
//#####################################################


void init_typed_cuda_activ(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			typed_cuda_activ_fct_association_FP32(net);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			typed_cuda_activ_fct_association_FP16(net);
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
		
		case BF16C_FP32A:
			#if defined(GEN_AMPERE)
			typed_cuda_activ_fct_association_BF16(net);
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


void cuda_define_activation_fct(layer *current)
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
			
		case YOLO:
			current->activation = cuda_YOLO_activation;
			current->deriv_activation = cuda_YOLO_deriv;
			cuda_YOLO_activ_init(current);
			break;
		
		default:
		case LINEAR:
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
			
		default:
		case LINEAR:
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








