	
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

//#####################################################


//#####################################################
//		  ReLU activation related templates
//#####################################################

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
#define ReLU_activation_kernel(name, type)																										\
__global__ void ReLU_activation_kernel_##name(void *i_tab, int len, int dim, int biased_dim, 													\
	float saturation, float leaking_factor, int size)																							\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < len && (i+1)%(dim+1) != 0)																										\
		{																																		\
			if(tab[i] <= (type) 0.0f)																											\
				tab[i] *= (type) leaking_factor;																								\
			else if(tab[i] > (type) saturation)																									\
				tab[i] = (type) saturation + (tab[i] - (type) saturation)*((type)leaking_factor);												\
		}																																		\
		else																																	\
			tab[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
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
__global__ void ReLU_deriv_kernel_##name(void *i_deriv, void *i_value, int len, int dim, int biased_dim,										\
	 float saturation, float leaking_factor, int size)																							\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < len && (i+1)%(dim+1) != 0)																										\
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
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
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
	(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int biased_dim, int offset, int size, float TC_scale_factor)			\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int pos;																																	\
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
		if(i < len && (i+1)%(dim+1) != 0)																										\
		{																																		\
			pos = i - i/(dim+1);																												\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
		{																																		\
			pos = (i/offset) + (i % offset)*dim;																								\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define quadratic_output_error_kernel(name, type)																								\
__global__ void quadratic_output_error_kernel_##name																							\
	(float *output_error, void *i_output, void *i_target, int len, int dim, int biased_dim, int offset, int size)								\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int pos;																																	\
																																				\
	type* output = (type*) i_output;																											\
	type* target = (type*) i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < len && (i+1)%(dim+1) != 0)																										\
		{																																		\
			pos = i - i/(dim+1);																												\
			output_error[i] = (0.5f*((float)output[i] - (float)target[pos])*((float)output[i] - (float)target[pos]));							\
		}																																		\
		else																																	\
			output_error[i]	= 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
		{																																		\
			pos = (i/offset) + (i % offset)*dim;																								\
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
__global__ void logistic_activation_kernel_##name(void *i_tab, float beta, float saturation, int len, int dim, int biased_dim, int size)		\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
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
		if(i < len && (i+1)%(dim+1) != 0)																										\
		{																																		\
			tab[i] = -t_beta*tab[i];																											\
			if(tab[i] > t_saturation)																											\
				tab[i] = t_saturation;																											\
			tab[i] = t_one/(t_one + exp_fct((float)tab[i]));																					\
		}																																		\
		else																																	\
			tab[i] = (type)0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
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
__global__ void logistic_deriv_kernel_##name(void *i_deriv, void *i_value, float beta, int len, int dim, int biased_dim, int size)				\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < len && (i+1)%(dim+1) != 0)																										\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
		{																																		\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		}																																		\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}

//#####################################################


//#####################################################
//		  Soft-Max activation related templates
//#####################################################

#define softmax_activation_kernel(name, type, exp_fct)																							\
__global__ void softmax_activation_kernel_##name(void *i_tab, int len, int dim, int biased_dim, int offset, int size)							\
{																																				\
	/*difficult to further optimize but can be invastigated*/																					\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j;																																		\
	type *pos, *off_pos;																														\
	type vmax;																																	\
	float normal = 0.0f;																														\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(i < len)																																	\
	{																																			\
		if(biased_dim > dim)																													\
			pos = tab + i*(biased_dim);																											\
		else																																	\
			pos = tab + i;																														\
																																				\
		vmax = *pos;																															\
		for(j = 0; j < dim; j++)																												\
		{																																		\
			off_pos = pos + j*offset;																											\
			if(*off_pos > vmax)																													\
				vmax = *off_pos;																												\
		}																																		\
																																				\
		for(j = 0; j < dim; j++)																												\
		{																																		\
			off_pos = pos + j*offset;																											\
			*off_pos = exp_fct((float)(*off_pos-vmax));																							\
			normal += (float)*off_pos;																											\
		}																																		\
		if(biased_dim > dim)																													\
			pos[dim] = 0.0f;																													\
																																				\
		for(j = 0; j < dim; j++)																												\
		{																																		\
			off_pos = pos + j*offset;																											\
			*off_pos = (type)((float)*off_pos/normal);																							\
		}																																		\
		if(biased_dim > dim)																													\
			pos[dim] = 0.0f;																													\
	}																																			\
	else																																		\
	{																																			\
		if(biased_dim > dim)																													\
			pos = tab + i*(biased_dim);																											\
		else																																	\
			pos = tab + i;																														\
																																				\
		for(j = 0; j < dim; j++)																												\
		{																																		\
			off_pos = pos + j*offset;																											\
			*off_pos = 0.0f;																													\
		}																																		\
		if(biased_dim > dim)																													\
			pos[dim] = 0.0f;																													\
	}																																			\
}

#define cross_entropy_deriv_output_error_kernel(name, type)																						\
__global__ void cross_entropy_deriv_output_error_kernel_##name																					\
	(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int biased_dim, int offset, int size, float TC_scale_factor)			\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int pos;																																	\
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
		if(i < len && (i+1)%(biased_dim) != 0)																									\
		{																																		\
			pos = i - i/(biased_dim);																											\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
		{																																		\
			pos = (i/offset) + (i % offset)*dim;																								\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define cross_entropy_output_error_kernel(name, type)																							\
__global__ void cross_entropy_output_error_kernel_##name																						\
	(float *output_error, void *i_output, void *i_target, int len, int dim, int biased_dim, int offset, int size)								\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int pos;																																	\
																																				\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < len && (i+1)%(biased_dim) != 0)																									\
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
		int length_size = len/biased_dim;																										\
		int batch_size = size/biased_dim;																										\
		if(i % batch_size < length_size)																										\
		{																																		\
			pos = (i/offset) + (i % offset)*dim;																								\
			if((float)output[i] > 0.000001f)																									\
				output_error[i] = -(float)target[pos] * logf((float)output[i]);																	\
			else																																\
				output_error[i] = -(float)target[pos] * logf((float)0.000001f);																	\
		}																																		\
		else																																	\
			output_error[i] = 0.0f;																												\
	}																																			\
}


#define group_normalization_dense_kernel(name, type) 																							\
__global__ void group_normalization_dense_kernel_##name(void *i_tab, int b_length, int b_size, 													\
	int dim, int biased_dim, int group_size, int nb_group)																						\
{																																				\
	/* Could be optimized with advanced multi-thread reduction */																				\
	int j = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int i = blockIdx.y*blockDim.y + threadIdx.y;																								\
	int k;																																		\
	type* tab = (type*) i_tab;																													\
	float l_val, eps = 0.00001f;																												\
	double mean = 0.0, var = 0.0;																												\
																																				\
	if(i < b_length && j < nb_group)																											\
	{																																			\
											/* Here dim = dim - set_off */																		\
		for(k = j*group_size; k < (j+1)*group_size && k < dim; k++)																				\
			mean += (float)tab[i*biased_dim + k];																								\
		mean /= group_size;																														\
																																				\
		for(k = j*group_size; k < (j+1)*group_size && k < dim; k++)																				\
		{																																		\
			l_val = (float)tab[i*biased_dim + k];																								\
			var += (l_val - mean)*(l_val - mean);																								\
		}																																		\
		var /= group_size;																														\
																																				\
		for(k = j*group_size; k < (j+1)*group_size && k < dim; k++)																				\
		{																																		\
			l_val = (float)tab[i*biased_dim + k];																								\
			tab[i*biased_dim + k] = (type) ((l_val - mean)/sqrt(var + eps));																	\
		}																																		\
	}																																			\
	/* Don't do anything for objects after length*/																								\
}



//#####################################################
//Exp activation (SGAN discriminator) related templates
//#####################################################

#define exp_disc_activation_kernel(name, type, exp_fct)																							\
__global__ void exp_disc_activation_kernel_##name(void *i_tab, int len, int dim, int size, int halved, int revert)								\
{																																				\
/*difficult to further optimize but can be invastigated*/																						\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j;																																		\
	type* pos;																																	\
	type vmax;																																	\
	float normal = 0.000001f + 0.0f;																											\
	/*float add_node = 0.0f;*/																													\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	pos = tab + i*(dim+1);																														\
																																				\
	/*if(revert || (halved && i < size/2))*/																									\
	/*	add_node = 1.0f;				  */																									\
																																				\
	if(i < len)																																	\
	{																																			\
		if((0 && revert) || (0 && (halved && i < size/2)))																						\
		{																																		\
			vmax = 0.000001f; /*the "fake" label is set to 0 (exp(0) = 1 in normal offset)	*/													\
			/*vmax = pos[0];				*/																									\
			/*for(j = 0; j < dim; j++)*/																										\
			/*	if(pos[j] > vmax)	*/																											\
			/*		vmax = pos[j];	*/																											\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				normal += exp_fct((float)(pos[j] - vmax));																						\
			}																																	\
																																				\
			for(j = 0; j < dim; j++)																											\
				/*pos[j] = exp_fct((float)(pos[j] - vmax));*/																					\
				pos[j] = (type) (normal/(normal+1.0f));																							\
			pos[dim] = 0.0f;																													\
		}																																		\
		else																																	\
		{																																		\
			/*if(1 || (revert || (halved && i < size/2)))	*/																					\
				vmax = 0.000001f;  /*the "fake" label is set to 0 (exp(0) = 1 in normal offset)	*/												\
			/*else								*/																								\
			/*	vmax = pos[0];					*/																								\
			/*for(j = 0; j < dim; j++)			*/																								\
			/*	if(pos[j] > vmax)				*/																								\
			/*		vmax = pos[j];				*/																								\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				/*if(pos[j] > (type) 6.0f)		*/																								\
				/*	pos[j] = 6.0f;				*/																								\
				pos[j] = exp_fct((float)(pos[j]-vmax));																							\
				normal += (float)pos[j];																										\
			}																																	\
			pos[dim] = 0.0f;																													\
																																				\
			/*for(j = 0; j < dim; j++)*/																										\
			/*	pos[j] /= (type)(normal + 1.0f);*/																								\
			/*pos[dim] = 0.0f;*/																												\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		for(j = 0; j < dim; j++)																												\
			pos[j] = 0.0f;																														\
		pos[dim] = 0.0f;																														\
	}																																			\
}


#define exp_disc_deriv_output_kernel(name, type)																								\
__global__ void exp_disc_deriv_output_kernel_##name																								\
(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, int halved, int revert)											\
{																																				\
/*difficult to further optimize but can be invastigated*/																						\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j;																																		\
																																				\
	type* delta_o = (type*)i_delta_o;																											\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	float sum = 0.000001f + 0.0f;																												\
	float vmax = 0.000001f;																														\
	int arg_max = 0;																															\
																																				\
	delta_o += i*(dim+1);																														\
	output  += i*(dim+1);																														\
	target  += i*(dim);																															\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(i < len)																																	\
	{																																			\
																																				\
		vmax = (float)output[0];																												\
		/*sum += (float)output[0];*/																											\
		for(j = 0; j < dim; j++)																												\
		{																																		\
			if((float)output[j] > vmax)																											\
			{																																	\
				vmax = (float)output[j];																										\
				arg_max = j;																													\
			}																																	\
			sum += (float)output[j];																											\
		}																																		\
																																				\
		if(revert)																																\
		{																																		\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				if(j == arg_max)																												\
					delta_o[j] = (type) (((float)(((float)output[j])/(sum+1.0f)) - 0.9f)/**((float)output[j])/(sum+0.0f)*/);					\
				else																															\
					delta_o[j] = (type) (((float)(((float)output[j])/(sum+1.0f)) - 0.0f)/**((float)output[j])/(sum+0.0f)*/);					\
			}																																	\
			delta_o[dim] = (type) 0.0f;																											\
		}																																		\
		else																																	\
		{																																		\
			if(halved && i < size/2)																											\
			{																																	\
				for(j = 0; j < dim; j++)																										\
					delta_o[j] = (type) (((float)(((float)output[j])/(sum+1.0f)) - 0.0f));														\
				delta_o[dim] = (type) 0.0f;																										\
			}																																	\
			else																																\
			{																																	\
				for(j = 0; j < dim; j++)																										\
					delta_o[j] = (type) (((float)(((float)output[j])/(sum+1.0f)) - (float)target[j]));											\
				delta_o[dim] = (type) 0.0f;																										\
			}																																	\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		for(j = 0; j < dim; j++)																												\
			delta_o[j] = (type) 0.0f;																											\
		delta_o[dim] = (type) 0.0f;																												\
	}																																			\
}



//#####################################################
//		  YOLO activation related templates
//#####################################################

#define YOLO_activation_kernel(name, type, exp_fct)																								\
__global__ void YOLO_activation_kernel_##name(void *i_tab, int flat_offset, int len, yolo_param y_param, int size, int class_softmax)			\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type* tab = (type*) i_tab;																													\
																																				\
	int nb_class = y_param.nb_class, nb_param = y_param.nb_param;																				\
	/*Default values are in activ_function.c (set_yolo_params)*/																				\
	float **sm_tab = y_param.slopes_and_maxes_tab;																								\
	float normal = 0.0000001f;																													\
	type vmax;																																	\
	int fit_dim = y_param.fit_dim;																												\
	int col, in_col, j;																															\
																																				\
	col = i / flat_offset;																														\
	in_col = col%(8+nb_class+nb_param);																											\
																																				\
	/*Position*/																																\
	if(in_col >= 0 && in_col < 3)																												\
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

__device__ float gpu_DIoU2_fct(float* output, float* target)
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
	(void *i_delta_o, void *i_output, void *i_target, int flat_target_size, int flat_output_size, 												\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, float TC_scale_factor, int nb_im_epoch, void *block_state)		\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type* delta_o = (type*) i_delta_o;																											\
	type* output  = (type*) i_output;																											\
	type* target  = (type*) i_target;																											\
	int l_o, l_t;																																\
																																				\
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 														\
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;													\
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;																\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim =  y_param.fit_dim, rand_startup =  y_param.rand_startup;																		\
	float rand_prob_best_box_assoc = y_param.rand_prob_best_box_assoc;																			\
	float min_prior_forced_scaling = y_param.min_prior_forced_scaling;																			\
																																				\
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
																																				\
	float *param_ind_scale = y_param.param_ind_scale;																							\
	float *lambda_noobj_prior = y_param.noobj_prob_prior;																						\
	float **sm_tab = y_param.slopes_and_maxes_tab;																								\
	int *target_cell_mask = y_param.target_cell_mask;																							\
	float *IoU_table = y_param.IoU_table;																										\
	float *dist_prior = y_param.dist_prior;																										\
	int *box_locked = y_param.box_locked;																										\
	float *box_in_pix = y_param.box_in_pix;																										\
	int class_softmax = y_param.class_softmax;																									\
	int diff_flag = y_param.diff_flag;																											\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];													\
	float min_prob_IoU_lim = y_param.IoU_limits[2], min_obj_IoU_lim = y_param.IoU_limits[3];													\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim = y_param.IoU_limits[5];													\
	float diff_IoU_lim = y_param.IoU_limits[6], diff_obj_lim = y_param.IoU_limits[7];															\
	int fit_pos = y_param.fit_parts[0], fit_size = y_param.fit_parts[1], fit_prob = y_param.fit_parts[2]; 										\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
																																				\
	int j, k, l;																																\
	int c_batch, f_offset;																														\
	int nb_obj_target, s_p_i = 0;																												\
	int nb_in_cell, id_in_cell, resp_box = -1, resp_targ = -1;																					\
	float best_dist, c_dist;																													\
	float max_IoU, current_IoU;																													\
	int cell_x, cell_y, cell_z;																													\
	int obj_cx, obj_cy, obj_cz;																													\
	float *c_box_in_pix;																														\
	float obj_in_offset[6];																														\
	float out_int[6], targ_int[6];																												\
	float targ_w, targ_h, targ_d;																												\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size * c_batch;																										\
	f_offset = size;																															\
																																				\
	i = i % flat_output_size;																													\
	cell_z = i / (nb_area_w*nb_area_h);																											\
	cell_y = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;																						\
	cell_x = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;																						\
																																				\
	delta_o += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;								\
	output  += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;								\
																																				\
	target_cell_mask +=	((nb_area_w*nb_area_h*nb_area_d)*c_batch * y_param.max_nb_obj_per_image);												\
	target_cell_mask +=	(cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) *  y_param.max_nb_obj_per_image;								\
																																				\
	IoU_table += ((nb_area_w*nb_area_h*nb_area_d)*c_batch * y_param.max_nb_obj_per_image * nb_box);												\
	IoU_table += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) *  y_param.max_nb_obj_per_image * nb_box;								\
																																				\
	dist_prior += ((nb_area_w*nb_area_h*nb_area_d)*c_batch * y_param.max_nb_obj_per_image * nb_box);											\
	dist_prior += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) *  y_param.max_nb_obj_per_image * nb_box;							\
																																				\
	box_locked += ((nb_area_w*nb_area_h*nb_area_d) * c_batch * nb_box);																			\
	box_locked += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * nb_box;															\
																																				\
	box_in_pix += ((nb_area_w*nb_area_h*nb_area_d) * c_batch * 6 * nb_box);																		\
	box_in_pix += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * 6 * nb_box;														\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_obj_target == -1)																														\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim; 																											\
	}																																			\
																																				\
	best_dist = 100000000;																														\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		c_box_in_pix = box_in_pix+k*6;																											\
		l_o = k*(8+nb_class+nb_param);																											\
		c_box_in_pix[0] = ((float)output[(l_o+0)*f_offset] + cell_x) * cell_w;																	\
		c_box_in_pix[1] = ((float)output[(l_o+1)*f_offset] + cell_y) * cell_h;																	\
		c_box_in_pix[2] = ((float)output[(l_o+2)*f_offset] + cell_z) * cell_d;																	\
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(l_o+3)*f_offset]);																		\
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(l_o+4)*f_offset]);																		\
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(l_o+5)*f_offset]);																		\
																																				\
		/* Min prior best association could be improved by using the "fit_dim" parameter to avoid definition issues with unused dimensions */	\
		/* This would allow to verify if each used dimension is smaller, rather that using a surface criteria (later in this function) */		\
		c_dist = sqrt(prior_w[k]*prior_w[k] + prior_h[k]*prior_h[k]	+ prior_d[k]*prior_d[k]);													\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
																																				\
		for(l = 0; l < y_param.max_nb_obj_per_image * nb_box; l++)																				\
		{																																		\
			IoU_table[l] = -2.0f;																												\
			dist_prior[l] = 100000000;																											\
		}																																		\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*(7+nb_param+diff_flag);																											\
		for(k = 0; k < 6; k++)																													\
			targ_int[k] = target[l_t+1+k];																										\
																																				\
		obj_cx = (int)( ((float)target[l_t+4] + (float)target[l_t+1])*0.5f / cell_w);															\
		obj_cy = (int)( ((float)target[l_t+5] + (float)target[l_t+2])*0.5f / cell_h);															\
		obj_cz = (int)( ((float)target[l_t+6] + (float)target[l_t+3])*0.5f / cell_d);															\
																																				\
		target_cell_mask[j] = 0;																												\
		/* Flag the targets that should be predicted by the current cell element */																\
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)																			\
		{																																		\
			target_cell_mask[j] = 1;																											\
			nb_in_cell++;																														\
		}																																		\
																																				\
		/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */												\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			if(box_locked[k] != 0)																												\
				continue;																														\
			c_box_in_pix = box_in_pix+k*6;																										\
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																				\
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																				\
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																				\
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																				\
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																				\
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																				\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
			if(current_IoU > good_IoU_lim)																										\
				box_locked[k] = 1;																												\
		}																																		\
	}																																			\
																																				\
	/* For all target in cell compute the IoU with the prediciton and distance to the prior */													\
	id_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		if(target_cell_mask[j] == 0)																											\
			continue;																															\
																																				\
		l_t = j*(7+nb_param+diff_flag);																											\
		for(k = 0; k < 6; k++)																													\
			targ_int[k] = target[l_t+1+k];																										\
																																				\
		targ_w = targ_int[3] - targ_int[0];																										\
		targ_h = targ_int[4] - targ_int[1];																										\
		targ_d = targ_int[5] - targ_int[2];																										\
																																				\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			c_box_in_pix = box_in_pix+k*6;																										\
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																				\
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																				\
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																				\
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																				\
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																				\
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																				\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
																																				\
			IoU_table[id_in_cell*nb_box + k] = current_IoU;																						\
			dist_prior[id_in_cell*nb_box + k] = sqrt(																							\
				 (targ_w-prior_w[k])*(targ_w-prior_w[k])																						\
				+(targ_h-prior_h[k])*(targ_h-prior_h[k])																						\
				+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																						\
		}																																		\
																																				\
		/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */									\
		if(strict_box_size_association > 0)																										\
		{																																		\
			for(l = 0; l < strict_box_size_association; l++)																					\
			{																																	\
				best_dist = 100000000;																											\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell*nb_box+k] > 0.0f && dist_prior[id_in_cell*nb_box+k] < best_dist)									\
						best_dist = dist_prior[id_in_cell*nb_box+k];																			\
				if(best_dist < 100000000)																										\
					for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */							\
						if(abs(dist_prior[id_in_cell*nb_box+k]-best_dist) < 0.001f)																\
							dist_prior[id_in_cell*nb_box+k] = -1.0f;																			\
			}																																	\
		}																																		\
		else																																	\
		{																																		\
			for(k = 0; k < nb_box; k++)																											\
				dist_prior[id_in_cell*nb_box+k] = -1.0f;																						\
		}																																		\
																																				\
		id_in_cell++;																															\
	}																																			\
																																				\
	for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)																					\
	{																																			\
		/* Force a random box association with only criteria being that the box is not already used */											\
		/* Used as a startup phase to get all the priors closer to the objects to detect */														\
		if(nb_im_epoch < rand_startup)																											\
		{																																		\
			resp_targ = id_in_cell; /* First select a target */																					\
			resp_box = -1;																														\
			for(k = 0; k < 2*nb_box; k++)																										\
			{																																	\
				resp_box = (int)(curand_uniform(&(((curandState_t*)block_state)[blockIdx.x]))*nb_box);											\
				if(dist_prior[resp_targ*nb_box+resp_box] < 0.0f && box_locked[resp_box] != 2)													\
					break;																														\
			}																																	\
																																				\
			if(resp_box == -1)																													\
				continue;																														\
																																				\
			k = 0;																																\
			for(j = 0; j < nb_obj_target; j++)																									\
			{																																	\
				k += target_cell_mask[j];																										\
				if(k == resp_targ + 1)																											\
					break;																														\
			}																																	\
			l_t = j*(7+nb_param+diff_flag);																										\
			for(k = 0; k < 6; k++)																												\
				targ_int[k] = target[l_t+1+k];																									\
																																				\
			targ_w = targ_int[3] - targ_int[0];																									\
			targ_h = targ_int[4] - targ_int[1];																									\
			targ_d = targ_int[5] - targ_int[2];																									\
		}																																		\
		else																																	\
		{																																		\
			max_IoU = -2.0f;																													\
			resp_box = -1;																														\
			resp_targ = -1;																														\
			for(l = 0; l < nb_in_cell; l++)																										\
				for(k = 0; k < nb_box; k++)																										\
					if(IoU_table[l*nb_box+k] > max_IoU && dist_prior[l*nb_box+k] < 0.0f)														\
					{																															\
						max_IoU = IoU_table[l*nb_box+k];																						\
						resp_targ = l;																											\
						resp_box = k;																											\
					}																															\
																																				\
			/* Happen if stric_box_size > 0 and no more good prior is available, or if there is more targets than boxes */						\
			/* In that case all the remaining target are unable to be associated to */ 															\
			/* any other box and the id_in_cell loop must be stoped */																			\
			if(resp_box == -1)																													\
				continue;																														\
																																				\
			/* l is the "best" index in the "in cell" list */																					\
			/* Need to get back the original target index from the "in cell" index */															\
			k = 0;																																\
			for(j = 0; j < nb_obj_target; j++)																									\
			{																																	\
				k += target_cell_mask[j];																										\
				if(k == resp_targ + 1)																											\
					break;																														\
			}																																	\
			/* The appropriate j value is set after this early stop loop */																		\
			l_t = j*(7+nb_param+diff_flag);																										\
			for(k = 0; k < 6; k++)																												\
				targ_int[k] = target[l_t+1+k];																									\
																																				\
			targ_w = targ_int[3] - targ_int[0];																									\
			targ_h = targ_int[4] - targ_int[1];																									\
			targ_d = targ_int[5] - targ_int[2];																									\
																																				\
			/* If prediction is too bad, associate it with the best theoritical prior instead (might found the same box again) */				\
			/* Also force the best theoritical prior association at a small rate */																\
			if(max_IoU < low_IoU_best_box_assoc || 																								\
				curand_uniform(&(((curandState_t*)block_state)[blockIdx.x])) < rand_prob_best_box_assoc)										\
			{																																	\
				best_dist = 100000000;																											\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_dist = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])																		\
						+(targ_h-prior_h[k])*(targ_h-prior_h[k])																				\
						+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																				\
					if(c_dist < best_dist)																										\
						best_dist = c_dist;																										\
				}																																\
				max_IoU = -2.0f;																												\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_dist = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])																		\
						+(targ_h-prior_h[k])*(targ_h-prior_h[k])																				\
						+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																				\
					if(abs(c_dist-best_dist) < 0.001f && IoU_table[resp_targ*nb_box+k] > max_IoU)												\
					{																															\
						max_IoU = IoU_table[resp_targ*nb_box+k];																				\
						resp_box = k;																											\
					}																															\
				}																																\
				/* Should always get a resp_box != -1, regarding all previous conditions */														\
			}																																	\
																																				\
			/* Force the association to the smallest prior (or identical) if the target is too small */											\
			best_dist = prior_w[s_p_i]*prior_h[s_p_i]*prior_d[s_p_i];																			\
			if(targ_w*targ_h*targ_d < min_prior_forced_scaling*best_dist)																		\
			{																																	\
				max_IoU = -2.0f;																												\
				for(k = 0; k < nb_box; k++)																										\
					if((prior_w[k]*prior_h[k]*prior_d[k] - best_dist) < 0.001f 																	\
							&& IoU_table[resp_targ*nb_box+k] > max_IoU)																			\
					{																															\
						max_IoU = IoU_table[resp_targ*nb_box+k];																				\
						resp_box = k;																											\
					}																															\
				/* If the smallest prior (or identical) is unavailable the resp_box is unchanged*/												\
			}																																	\
		}																																		\
																																				\
		/* Mark the target as already associated by removing its contributions to the IoU table */												\
		for(k = 0; k < nb_box; k++)																												\
			IoU_table[resp_targ*nb_box + k] = -2.0f;																							\
																																				\
		c_box_in_pix = box_in_pix+resp_box*6;																									\
		out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																					\
		out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																					\
		out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																					\
		out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																					\
		out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																					\
		out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																					\
																																				\
		max_IoU = y_param.c_IoU_fct(out_int, targ_int);																							\
		if(max_IoU > 0.98f)																														\
			max_IoU = 0.98f;																													\
		if(class_only_IoU > -2.0f)																												\
			max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/										\
																																				\
		l_o = resp_box*(8+nb_class+nb_param);																									\
																																				\
		/* Positive reinforcement */ 																											\
		/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */				\
		/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/				\
		if(diff_flag && (float)target[l_t+7+nb_param] > 0.9f && (max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))		\
			continue;																															\
																																				\
		/* Mark the box as already associated by removing its contributions to the IoU table */													\
		for(k = 0; k < nb_in_cell; k++)																											\
			IoU_table[k*nb_box + resp_box] = -2.0f;																								\
																																				\
		box_locked[resp_box] = 2;																												\
																																				\
		obj_in_offset[0] = ((targ_int[3] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w;													\
		obj_in_offset[1] = ((targ_int[4] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h;													\
		obj_in_offset[2] = ((targ_int[5] + targ_int[2])*0.5f - cell_z*cell_d)/(float)cell_d;													\
		obj_in_offset[3] = (targ_w)/(float)prior_w[resp_box];																					\
		if(obj_in_offset[3] < size_min_sat)																										\
			obj_in_offset[3] = logf(size_min_sat);																								\
		else if(obj_in_offset[3] > size_max_sat)																								\
			obj_in_offset[3] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[3] = logf(obj_in_offset[3]);																							\
		obj_in_offset[4] = (targ_h)/(float)prior_h[resp_box];																					\
		if(obj_in_offset[4] < size_min_sat)																										\
			obj_in_offset[4] = logf(size_min_sat);																								\
		else if(obj_in_offset[4] > size_max_sat)																								\
			obj_in_offset[4] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[4] = logf(obj_in_offset[4]);																							\
		obj_in_offset[5] = (targ_d)/(float)prior_d[resp_box];																					\
		if(obj_in_offset[5] < size_min_sat)																										\
			obj_in_offset[5] = logf(size_min_sat);																								\
		else if(obj_in_offset[5] > size_max_sat)																								\
			obj_in_offset[5] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[5] = logf(obj_in_offset[5]);																							\
																																				\
		switch(fit_pos)																															\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f)																					\
						delta_o[(l_o+k)*f_offset] = (type)(TC_scale_factor*sm_tab[0][0]															\
							*coord_scale*(float)output[(l_o+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+k)*f_offset])																				\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]));																\
					else																														\
						delta_o[(l_o+k)*f_offset] = (type)(0.0f);																				\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						delta_o[(l_o+k)*f_offset] = (type)(TC_scale_factor*sm_tab[0][0]															\
							*coord_scale*(float)output[(l_o+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+k)*f_offset])																				\
							*((float)output[(l_o+k)*f_offset]-0.5f));																			\
					else																														\
						delta_o[(l_o+k)*f_offset] = (type)(0.0f);																				\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					delta_o[(l_o+k)*f_offset] = (type)(0.0f);																					\
				break;																															\
		}																																		\
																																				\
		switch(fit_size)																														\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f)																					\
						delta_o[(l_o+k+3)*f_offset] = (type) (TC_scale_factor*sm_tab[1][0]														\
							*size_scale*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]));												\
					else																														\
						delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																			\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						delta_o[(l_o+k+3)*f_offset] = (type) (TC_scale_factor*sm_tab[1][0]														\
							*size_scale*((float)output[(l_o+k+3)*f_offset]-0.0f));																\
					else																														\
						delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																				\
				break;																															\
		}																																		\
																																				\
		switch(fit_prob)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_prob_IoU_lim)																									\
					delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*sm_tab[2][0]																\
						*prob_scale*(float)output[(l_o+6)*f_offset]																				\
						*(1.0f-(float)output[(l_o+6)*f_offset])																					\
						*((float)output[(l_o+6)*f_offset]-0.98f));																				\
				else																															\
					delta_o[(l_o+6)*f_offset] = (type)(0.0f);																					\
				break;																															\
			case 0:																																\
				delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*sm_tab[2][0]																	\
					*prob_scale*(float)output[(l_o+6)*f_offset]																					\
					*(1.0f-(float)output[(l_o+6)*f_offset])																						\
					*((float)output[(l_o+6)*f_offset]-0.5f));																					\
				break;																															\
			case -1:																															\
				delta_o[(l_o+6)*f_offset] = (type)(0.0f);																						\
				break;																															\
		}																																		\
																																				\
		switch(fit_obj)																															\
		{																																		\
			case 1:																																\
				if(max_IoU > min_obj_IoU_lim)																									\
					delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*sm_tab[3][0]																\
						*obj_scale*(float)output[(l_o+7)*f_offset]																				\
						*(1.0f-(float)output[(l_o+7)*f_offset])																					\
						*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5));																	\
				else																															\
					delta_o[(l_o+7)*f_offset] = (type)(0.0f);																					\
				break;																															\
			case 0:																																\
				delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*sm_tab[3][0]																	\
					*obj_scale*(float)output[(l_o+7)*f_offset]																					\
					*(1.0f-(float)output[(l_o+7)*f_offset])																						\
					*((float)output[(l_o+7)*f_offset]-0.5f));																					\
				break;																															\
			case -1:																															\
				delta_o[(l_o+7)*f_offset] = (type)(0.0f);																						\
				break;																															\
		}																																		\
																																				\
		/* Note : mean square error on classes => could be changed to soft max but difficult to balance */										\
		switch(fit_class)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_class_IoU_lim)																									\
				{																																\
					if(class_softmax)																											\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int) target[l_t]-1)																						\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor															\
									*class_scale*((float)output[(l_o+8+k)*f_offset]-1.0f));														\
							else																												\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor															\
									*class_scale*((float)output[(l_o+8+k)*f_offset]-0.0f));														\
						}																														\
					}																															\
					else																														\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int) target[l_t]-1)																						\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]												\
									*class_scale*(float)output[(l_o+8+k)*f_offset]																\
									*(1.0f-(float)output[(l_o+8+k)*f_offset])																	\
									*((float)output[(l_o+8+k)*f_offset]-0.98f));																\
							else																												\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]												\
									*class_scale*(float)output[(l_o+8+k)*f_offset]																\
									*(1.0f-(float)output[(l_o+8+k)*f_offset])																	\
									*((float)output[(l_o+8+k)*f_offset]-0.02f));																\
						}																														\
					}																															\
				}																																\
				else																															\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																			\
				break;																															\
			case 0:																																\
				if(class_softmax)																												\
				{																																\
					/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */			\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																			\
				}																																\
				else																															\
				{																																\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]														\
							*class_scale*(float)output[(l_o+8+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+8+k)*f_offset])																			\
							*((float)output[(l_o+8+k)*f_offset]-0.5f));																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < nb_class; k++)																									\
					delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																				\
				break;																															\
		}																																		\
																																				\
		/* Linear activation of additional parameters */																						\
		switch(fit_param)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_param_IoU_lim)																									\
					for(k = 0; k < nb_param; k++)																								\
						delta_o[(l_o+8+nb_class+k)*f_offset] = 																					\
							(type) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale													\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));												\
				else																															\
					for(k = 0; k < nb_param; k++)																								\
						delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																	\
				break;																															\
			case 0:																																\
				for(k = 0; k < nb_param; k++)																									\
					delta_o[(l_o+8+nb_class+k)*f_offset] = 																						\
						(type) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale														\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));																	\
				break;																															\
			default:																															\
			case -1:																															\
				for(k = 0; k < nb_param; k++)																									\
					delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																		\
				break;																															\
		}																																		\
	}																																			\
																																				\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		/* If no match only update Objectness toward 0 */																						\
		/* (here it means error compute)! (no coordinate nor class update) */																	\
		l_o = j*(8+nb_class+nb_param);																											\
		if(box_locked[j] != 2)																													\
		{																																		\
			for(k = 0; k < 6; k++)																												\
				delta_o[(l_o+k)*f_offset] = (type) 0.0f;																						\
																																				\
			if(box_locked[j] == 1)																												\
			{																																	\
				delta_o[(l_o+6)*f_offset] = (type) 0.0f;																						\
				delta_o[(l_o+7)*f_offset] = (type) 0.0f;																						\
			}																																	\
			else																																\
			{																																	\
				switch(fit_prob)																												\
				{																																\
					case 1:																														\
						delta_o[(l_o+6)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[2][0]*(lambda_noobj_prior[j])																\
							*prob_scale*(float)output[(l_o+6)*f_offset]																			\
							*(1.0f-(float)output[(l_o+6)*f_offset])																				\
							*((float)output[(l_o+6)*f_offset]-0.02f));																			\
						break;																													\
					case 0:																														\
						delta_o[(l_o+6)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[2][0]*(lambda_noobj_prior[j])																\
							*prob_scale*(float)output[(l_o+6)*f_offset]																			\
							*(1.0f-(float)output[(l_o+6)*f_offset])																				\
							*((float)output[(l_o+6)*f_offset]-0.5f));																			\
						break;																													\
					case -1:																													\
						delta_o[(l_o+6)*f_offset] = (type)(0.0f);																				\
						break;																													\
				}																																\
				switch(fit_obj)																													\
				{																																\
					case 1:																														\
						delta_o[(l_o+7)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])																\
							*obj_scale*(float)output[(l_o+7)*f_offset]																			\
							*(1.0f-(float)output[(l_o+7)*f_offset])																				\
							*((float)output[(l_o+7)*f_offset]-0.02f));																			\
						break;																													\
					case 0:																														\
						delta_o[(l_o+7)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])																\
							*obj_scale*(float)output[(l_o+7)*f_offset]																			\
							*(1.0f-(float)output[(l_o+7)*f_offset])																				\
							*((float)output[(l_o+7)*f_offset]-0.5f));																			\
						break;																													\
					case -1:																													\
						delta_o[(l_o+7)*f_offset] = (type)(0.0f);																				\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_class; k++)																										\
				delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																					\
																																				\
			for(k = 0; k < nb_param; k++)																										\
				delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																			\
		}																																		\
	}																																			\
}


#define YOLO_error_kernel(name, type)																											\
__global__ void YOLO_error_kernel_##name																										\
	(float *output_error, void *i_output, void *i_target, int flat_target_size, int flat_output_size, 											\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)																	\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type* output = (type*) i_output;																											\
	type* target = (type*) i_target;																											\
	int l_o, l_t;																																\
																																				\
	int nb_box = y_param.nb_box, nb_class = y_param.nb_class, nb_param = y_param.nb_param; 														\
	float *prior_w = y_param.prior_w, *prior_h = y_param.prior_h, *prior_d = y_param.prior_d;													\
	int cell_w = y_param.cell_w, cell_h = y_param.cell_h, cell_d = y_param.cell_d;																\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim =  y_param.fit_dim;																												\
	float min_prior_forced_scaling = y_param.min_prior_forced_scaling;																			\
																																				\
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
	float *lambda_noobj_prior = y_param.noobj_prob_prior;																						\
	float **sm_tab = y_param.slopes_and_maxes_tab;																								\
	float *param_ind_scale = y_param.param_ind_scale;																							\
	float *IoU_monitor = y_param.IoU_monitor;																									\
	int *target_cell_mask = y_param.target_cell_mask;																							\
	float *IoU_table = y_param.IoU_table;																										\
	float *dist_prior = y_param.dist_prior;																										\
	int *box_locked = y_param.box_locked;																										\
	float *box_in_pix = y_param.box_in_pix;																										\
	int class_softmax = y_param.class_softmax;																									\
	int diff_flag = y_param.diff_flag;																											\
	int error_type = y_param.error_type;																										\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];													\
	float min_prob_IoU_lim = y_param.IoU_limits[2], min_obj_IoU_lim = y_param.IoU_limits[3];													\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim = y_param.IoU_limits[5];													\
	float diff_IoU_lim = y_param.IoU_limits[6], diff_obj_lim = y_param.IoU_limits[7];															\
	int fit_pos = y_param.fit_parts[0], fit_size = y_param.fit_parts[1], fit_prob = y_param.fit_parts[2]; 										\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
																																				\
	int j, k, l;																																\
	int c_batch, f_offset;																														\
	int nb_obj_target, s_p_i = 0;																												\
	int nb_in_cell, id_in_cell, resp_box = -1, resp_targ = -1;																					\
	float best_dist, c_dist;																													\
	float max_IoU, current_IoU;																													\
	int cell_x, cell_y, cell_z;																													\
	int obj_cx, obj_cy, obj_cz;																													\
	float *c_box_in_pix;																														\
	float obj_in_offset[6];																														\
	float out_int[6], targ_int[6];																												\
	float targ_w, targ_h, targ_d;																												\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size * c_batch;																										\
	f_offset = size;																															\
																																				\
	i = i % flat_output_size;																													\
	cell_z = i / (nb_area_w*nb_area_h);																											\
	cell_y = (int)(i % (nb_area_w*nb_area_h)) % nb_area_w;																						\
	cell_x = (int)(i % (nb_area_w*nb_area_h)) / nb_area_w;																						\
																																				\
	output_error += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;							\
	output += (nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x;								\
																																				\
	IoU_monitor += 2 * nb_box * ((nb_area_w*nb_area_h*nb_area_d) * c_batch + cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x);			\
																																				\
	target_cell_mask +=	(nb_area_w*nb_area_h*nb_area_d) * c_batch * y_param.max_nb_obj_per_image;												\
	target_cell_mask +=	(cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * y_param.max_nb_obj_per_image;								\
																																				\
	IoU_table += ((nb_area_w*nb_area_h*nb_area_d) * c_batch * y_param.max_nb_obj_per_image * nb_box);											\
	IoU_table += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * y_param.max_nb_obj_per_image * nb_box;								\
																																				\
	dist_prior += ((nb_area_w*nb_area_h*nb_area_d)*c_batch * y_param.max_nb_obj_per_image * nb_box);											\
	dist_prior += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) *  y_param.max_nb_obj_per_image * nb_box;							\
																																				\
	box_locked += ((nb_area_w*nb_area_h*nb_area_d) * c_batch * nb_box);																			\
	box_locked += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * nb_box;															\
																																				\
	box_in_pix += ((nb_area_w*nb_area_h*nb_area_d) * c_batch * 6 * nb_box);																		\
	box_in_pix += (cell_z*nb_area_w*nb_area_h + cell_y*nb_area_w + cell_x) * 6 * nb_box;														\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_obj_target == -1)																														\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim; 																											\
	}																																			\
																																				\
	best_dist = 100000000;																														\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		c_box_in_pix = box_in_pix+k*6;																											\
		l_o = k*(8+nb_class+nb_param);																											\
		c_box_in_pix[0] = ((float)output[(l_o+0)*f_offset] + cell_x) * cell_w;																	\
		c_box_in_pix[1] = ((float)output[(l_o+1)*f_offset] + cell_y) * cell_h;																	\
		c_box_in_pix[2] = ((float)output[(l_o+2)*f_offset] + cell_z) * cell_d;																	\
		c_box_in_pix[3] = prior_w[k]*expf((float)output[(l_o+3)*f_offset]);																		\
		c_box_in_pix[4] = prior_h[k]*expf((float)output[(l_o+4)*f_offset]);																		\
		c_box_in_pix[5] = prior_d[k]*expf((float)output[(l_o+5)*f_offset]);																		\
																																				\
		c_dist = sqrt(prior_w[k]*prior_w[k] + prior_h[k]*prior_h[k]	+ prior_d[k]*prior_d[k]);													\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
																																				\
		for(l = 0; l < y_param.max_nb_obj_per_image * nb_box; l++)																				\
		{																																		\
			IoU_table[l] = -2.0f;																												\
			dist_prior[l] = 100000000;																											\
		}																																		\
																																				\
		IoU_monitor[k*2] = -1.0f;																												\
		IoU_monitor[k*2+1] = -1.0f;																												\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*(7+nb_param+diff_flag);																											\
		for(k = 0; k < 6; k++)																													\
			targ_int[k] = target[l_t+1+k];																										\
																																				\
		obj_cx = (int)( ((float)target[l_t+4] + (float)target[l_t+1])*0.5f / cell_w);															\
		obj_cy = (int)( ((float)target[l_t+5] + (float)target[l_t+2])*0.5f / cell_h);															\
		obj_cz = (int)( ((float)target[l_t+6] + (float)target[l_t+3])*0.5f / cell_d);															\
																																				\
		target_cell_mask[j] = 0;																												\
		if(obj_cx == cell_x && obj_cy == cell_y && obj_cz == cell_z)																			\
		{																																		\
			target_cell_mask[j] = 1;																											\
			nb_in_cell++;																														\
		}																																		\
																																				\
		/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */												\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			c_box_in_pix = box_in_pix+k*6;																										\
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																				\
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																				\
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																				\
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																				\
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																				\
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																				\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
			if(box_locked[k] == 0 && current_IoU > good_IoU_lim)																				\
				box_locked[k] = 1;																												\
		}																																		\
	}																																			\
																																				\
	id_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		if(target_cell_mask[j] == 0)																											\
			continue;																															\
																																				\
		l_t = j*(7+nb_param+diff_flag);																											\
		for(k = 0; k < 6; k++)																													\
			targ_int[k] = target[l_t+1+k];																										\
																																				\
		targ_w = targ_int[3] - targ_int[0];																										\
		targ_h = targ_int[4] - targ_int[1];																										\
		targ_d = targ_int[5] - targ_int[2];																										\
																																				\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			c_box_in_pix = box_in_pix+k*6;																										\
			out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																				\
			out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																				\
			out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																				\
			out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																				\
			out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																				\
			out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																				\
																																				\
			current_IoU = y_param.c_IoU_fct(out_int, targ_int);																					\
																																				\
			IoU_table[id_in_cell*nb_box + k] = current_IoU;																						\
			dist_prior[id_in_cell*nb_box + k] = sqrt(																							\
				 (targ_w-prior_w[k])*(targ_w-prior_w[k])																						\
				+(targ_h-prior_h[k])*(targ_h-prior_h[k])																						\
				+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																						\
		}																																		\
																																				\
		/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */									\
		if(error_type == ERR_COMPLETE && strict_box_size_association > 0)																		\
		{																																		\
			for(l = 0; l < strict_box_size_association; l++)																					\
			{																																	\
				best_dist = 100000000;																											\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell*nb_box+k] > 0.0f && dist_prior[id_in_cell*nb_box+k] < best_dist)									\
						best_dist = dist_prior[id_in_cell*nb_box+k];																			\
				if(best_dist < 100000000)																										\
					for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */							\
						if(abs(dist_prior[id_in_cell*nb_box+k]-best_dist) < 0.001f)																\
							dist_prior[id_in_cell*nb_box+k] = -1.0f;																			\
			}																																	\
		}																																		\
		else																																	\
		{																																		\
			for(k = 0; k < nb_box; k++)																											\
				dist_prior[id_in_cell*nb_box+k] = -1.0f;																						\
		}																																		\
																																				\
		id_in_cell++;																															\
	}																																			\
																																				\
	for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)																					\
	{																																			\
		/* No random association in error display*/																								\
		max_IoU = -2.0f;																														\
		resp_box = -1;																															\
		resp_targ = -1;																															\
		for(l = 0; l < nb_in_cell; l++)																											\
			for(k = 0; k < nb_box; k++)																											\
				if(IoU_table[l*nb_box+k] > max_IoU && dist_prior[l*nb_box+k] < 0.0f)															\
				{																																\
					max_IoU = IoU_table[l*nb_box+k];																							\
					resp_targ = l;																												\
					resp_box = k;																												\
				}																																\
																																				\
		if(resp_box == -1) /* Only happen if all the boxes are taken (more targets in the cell than boxes) */									\
			continue;																															\
																																				\
		/*Need to get back the original target index from the "in cell" index*/																	\
		k = 0;																																	\
		for(j = 0; j < nb_obj_target; j++)																										\
		{																																		\
			k += target_cell_mask[j];																											\
			if(k == resp_targ + 1)																												\
				break;																															\
		}																																		\
		/* The appropriate j is defined after this early stop loop*/																			\
		l_t = j*(7+nb_param+diff_flag);																											\
		for(k = 0; k < 6; k++)																													\
			targ_int[k] = target[l_t+1+k];																										\
																																				\
		targ_w = targ_int[3] - targ_int[0];																										\
		targ_h = targ_int[4] - targ_int[1];																										\
		targ_d = targ_int[5] - targ_int[2];																										\
																																				\
		/* If prediction is too bad, associate it with the best theoritical prior instead (might found the same box again) */					\
		if(error_type == ERR_COMPLETE && max_IoU < low_IoU_best_box_assoc)																		\
		{																																		\
			best_dist = 100000000;																												\
			for(k = 0; k < nb_box; k++)																											\
			{																																	\
				c_dist = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])																			\
					+(targ_h-prior_h[k])*(targ_h-prior_h[k])																					\
					+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																					\
				if(c_dist < best_dist)																											\
					best_dist = c_dist;																											\
			}																																	\
			max_IoU = -2.0f;																													\
			for(k = 0; k < nb_box; k++)																											\
			{																																	\
				c_dist = sqrt((targ_w-prior_w[k])*(targ_w-prior_w[k])																			\
					+(targ_h-prior_h[k])*(targ_h-prior_h[k])																					\
					+(targ_d-prior_d[k])*(targ_d-prior_d[k]));																					\
				if(abs(c_dist-best_dist) < 0.001f && IoU_table[resp_targ*nb_box+k] > max_IoU)													\
				{																																\
					max_IoU = IoU_table[resp_targ*nb_box+k];																					\
					resp_box = k;																												\
				}																																\
			}																																	\
			/* Should always get a resp_box != -1, regarding all previous conditions */															\
		}																																		\
																																				\
		/* Force the association to the smallest prior (or identical) if the target is too small */												\
		best_dist = prior_w[s_p_i]*prior_h[s_p_i]*prior_d[s_p_i];																				\
		if(error_type == ERR_COMPLETE && targ_w*targ_h*targ_d < min_prior_forced_scaling*best_dist)												\
		{																																		\
			max_IoU = -2.0f;																													\
			for(k = 0; k < nb_box; k++)																											\
				if((prior_w[k]*prior_h[k]*prior_d[k] - best_dist) < 0.001f 																		\
						&& IoU_table[resp_targ*nb_box+k] > max_IoU)																				\
				{																																\
					max_IoU = IoU_table[resp_targ*nb_box+k];																					\
					resp_box = k;																												\
				}																																\
			/* If the smallest prior (or identical) is unavailable the resp_box is unchanged*/													\
		}																																		\
																																				\
		/* Mark the target as already associated by removing its contributions to the IoU table */												\
		for(k = 0; k < nb_box; k++)																												\
			IoU_table[resp_targ*nb_box + k] = -2.0f;																							\
																																				\
		c_box_in_pix = box_in_pix+resp_box*6;																									\
		out_int[0] = c_box_in_pix[0] - 0.5f*c_box_in_pix[3];																					\
		out_int[1] = c_box_in_pix[1] - 0.5f*c_box_in_pix[4];																					\
		out_int[2] = c_box_in_pix[2] - 0.5f*c_box_in_pix[5];																					\
		out_int[3] = c_box_in_pix[0] + 0.5f*c_box_in_pix[3];																					\
		out_int[4] = c_box_in_pix[1] + 0.5f*c_box_in_pix[4];																					\
		out_int[5] = c_box_in_pix[2] + 0.5f*c_box_in_pix[5];																					\
																																				\
		max_IoU = y_param.c_IoU_fct(out_int, targ_int);																							\
		if(max_IoU > 0.98f)																														\
			max_IoU = 0.98f;																													\
		if(class_only_IoU > -2.0f)																												\
			max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/										\
																																				\
		l_o = resp_box*(8+nb_class+nb_param);																									\
																																				\
		/* Positive reinforcement */ 																											\
		/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */				\
		/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/				\
		if(diff_flag && (float)target[l_t+7+nb_param] > 0.9f 																					\
			&& (error_type == ERR_NATURAL || max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))							\
			continue;																															\
																																				\
		/* Mark the box as already associated by removing its contributions to the IoU table */													\
		for(k = 0; k < nb_in_cell; k++)																											\
			IoU_table[k*nb_box + resp_box] = -2.0f;																								\
																																				\
		box_locked[resp_box] = 2;																												\
																																				\
		IoU_monitor[resp_box*2] = (float)output[(l_o+7)*f_offset];																				\
		IoU_monitor[resp_box*2+1] = max_IoU;																									\
																																				\
		obj_in_offset[0] = fmaxf(0.01f,fminf(0.99f,((targ_int[3] + targ_int[0])*0.5f - cell_x*cell_w)/(float)cell_w));							\
		obj_in_offset[1] = fmaxf(0.01f,fminf(0.99f,((targ_int[4] + targ_int[1])*0.5f - cell_y*cell_h)/(float)cell_h));							\
		obj_in_offset[2] = fmaxf(0.01f,fminf(0.99f,((targ_int[5] + targ_int[2])*0.5f - cell_z*cell_d)/(float)cell_d));							\
		obj_in_offset[3] = (targ_w)/(float)prior_w[resp_box];																					\
		if(obj_in_offset[3] < size_min_sat)																										\
			obj_in_offset[3] = logf(size_min_sat);																								\
		else if(obj_in_offset[3] > size_max_sat)																								\
			obj_in_offset[3] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[3] = logf(obj_in_offset[3]);																							\
		obj_in_offset[4] = (targ_h)/(float)prior_h[resp_box];																					\
		if(obj_in_offset[4] < size_min_sat)																										\
			obj_in_offset[4] = logf(size_min_sat);																								\
		else if(obj_in_offset[4] > size_max_sat)																								\
			obj_in_offset[4] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[4] = logf(obj_in_offset[4]);																							\
		obj_in_offset[5] = (targ_d)/(float)prior_d[resp_box];																					\
		if(obj_in_offset[5] < size_min_sat)																										\
			obj_in_offset[5] = logf(size_min_sat);																								\
		else if(obj_in_offset[5] > size_max_sat)																								\
			obj_in_offset[5] = logf(size_max_sat);																								\
		else																																	\
			obj_in_offset[5] = logf(obj_in_offset[5]);																							\
																																				\
		switch(fit_pos)																															\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f)																					\
						output_error[(l_o+k)*f_offset] = 0.5f*coord_scale																		\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k])																	\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]);																\
					else																														\
						output_error[(l_o+k)*f_offset] = 0.0f;																					\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						output_error[(l_o+k)*f_offset] = 0.5f*coord_scale																		\
							*((float)output[(l_o+k)*f_offset]-0.5f)																				\
							*((float)output[(l_o+k)*f_offset]-0.5f);																			\
					else																														\
						output_error[(l_o+k)*f_offset] = 0.0f;																					\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					output_error[(l_o+k)*f_offset] = 0.0f;																						\
				break;																															\
		}																																		\
																																				\
		switch(fit_size)																														\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f)																					\
						output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale																		\
						*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3])																	\
						*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]);																\
					else																														\
						output_error[(l_o+k+3)*f_offset] = 0.0f;																				\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale																		\
						*((float)output[(l_o+k+3)*f_offset]-0.0f)																				\
						*((float)output[(l_o+k+3)*f_offset]-0.0f);																				\
					else																														\
						output_error[(l_o+k+3)*f_offset] = 0.0f;																				\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					output_error[(l_o+k+3)*f_offset] = 0.0f;																					\
				break;																															\
		}																																		\
																																				\
		switch(fit_prob)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_prob_IoU_lim || error_type == ERR_NATURAL)																		\
					output_error[(l_o+6)*f_offset] = 0.5f*prob_scale																			\
						*((float)output[(l_o+6)*f_offset]-0.98f)																				\
						*((float)output[(l_o+6)*f_offset]-0.98f);																				\
				else																															\
					output_error[(l_o+6)*f_offset] = 0.0f;																						\
				break;																															\
			case 0:																																\
				output_error[(l_o+6)*f_offset] = 0.5f*prob_scale																				\
					*((float)output[(l_o+6)*f_offset]-0.5f)																						\
					*((float)output[(l_o+6)*f_offset]-0.5f);																					\
				break;																															\
			case -1:																															\
				output_error[(l_o+6)*f_offset] = 0.0f;																							\
				break;																															\
		}																																		\
																																				\
		switch(fit_obj)																															\
		{																																		\
			case 1:																																\
				if(max_IoU > min_obj_IoU_lim || error_type == ERR_NATURAL)																		\
					output_error[(l_o+7)*f_offset] = 0.5f*obj_scale																				\
						*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5)																	\
						*((float)output[(l_o+7)*f_offset]-(1.0+max_IoU)*0.5);																	\
				else																															\
					output_error[(l_o+7)*f_offset] = 0.0f;																						\
				break;																															\
			case 0:																																\
				output_error[(l_o+7)*f_offset] = 0.5f*obj_scale																					\
					*((float)output[(l_o+7)*f_offset]-0.5)																						\
					*((float)output[(l_o+7)*f_offset]-0.5);																						\
				break;																															\
			case -1:																															\
				output_error[(l_o+7)*f_offset] = 0.0f;																							\
				break;																															\
		}																																		\
																																				\
		/*Note : mean square error on classes => could be changed to soft max but difficult to balance*/										\
		switch(fit_class)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_class_IoU_lim || error_type == ERR_NATURAL)																	\
				{																																\
					if(class_softmax)																											\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int)target[l_t]-1)																							\
							{																													\
								if((float)output[(l_o+8+k)*f_offset] > 0.0000001f)																\
									output_error[(l_o+8+k)*f_offset] = class_scale																\
										*(-logf((float)output[(l_o+8+k)*f_offset]));															\
								else																											\
									output_error[(l_o+8+k)*f_offset] = class_scale*(-logf(0.0000001f));											\
							}																													\
							else																												\
								output_error[(l_o+8+k)*f_offset] = 0.0f;																		\
						}																														\
					}																															\
					else																														\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int)target[l_t]-1)																							\
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																\
									*((float)output[(l_o+8+k)*f_offset]-0.98f)																	\
									*((float)output[(l_o+8+k)*f_offset]-0.98f);																	\
							else																												\
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																\
									*((float)output[(l_o+8+k)*f_offset]-0.02f)																	\
									*((float)output[(l_o+8+k)*f_offset]-0.02f);																	\
						}																														\
					}																															\
				}																																\
				else																															\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.0f;																				\
				break;																															\
			case 0:																																\
				if(class_softmax)																												\
				{																																\
					/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */			\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.0f;																				\
				}																																\
				else																															\
				{																																\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																		\
							*((float)output[(l_o+8+k)*f_offset]-0.5f)																			\
							*((float)output[(l_o+8+k)*f_offset]-0.5f);																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < nb_class; k++)																									\
					output_error[(l_o+8+k)*f_offset] = 0.0f;																					\
				break;																															\
		}																																		\
																																				\
		/*Linear error of additional parameters*/																								\
		switch(fit_param)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_param_IoU_lim || error_type == ERR_NATURAL)																	\
					for(k = 0; k < nb_param; k++)																								\
						output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale										\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k])												\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));												\
				else																															\
					for(k = 0; k < nb_param; k++)																								\
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																		\
				break;																															\
			case 0:																																\
				for(k = 0; k < nb_param; k++)																									\
					output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale											\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f)																		\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));																	\
				break;																															\
			default:																															\
			case -1:																															\
				for(k = 0; k < nb_param; k++)																									\
					output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																			\
				break;																															\
		}																																		\
	}																																			\
																																				\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		/*If no match only update Objectness toward 0 */																						\
		/*(here it means error compute)! (no coordinate nor class update)*/																		\
		l_o = j*(8+nb_class+nb_param);																											\
		if(box_locked[j] != 2)																													\
		{																																		\
			for(k = 0; k < 6; k++)																												\
				output_error[(l_o+k)*f_offset] = 0.0f;																							\
																																				\
			if(box_locked[j] == 1)																												\
			{																																	\
				output_error[(l_o+6)*f_offset] = 0.0f;																							\
				output_error[(l_o+7)*f_offset] = 0.0f;																							\
			}																																	\
			else																																\
			{																																	\
				switch(fit_prob)																												\
				{																																\
					case 1:																														\
						output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale												\
							*((float)output[(l_o+6)*f_offset]-0.02f)																			\
							*((float)output[(l_o+6)*f_offset]-0.02f);																			\
						break;																													\
					case 0:																														\
						output_error[(j*(8+nb_class+nb_param)+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale							\
							*((float)output[(l_o+6)*f_offset]-0.5f)																				\
							*((float)output[(l_o+6)*f_offset]-0.5f);																			\
						break;																													\
					case -1:																													\
						output_error[(l_o+6)*f_offset] = 0.0f;																					\
						break;																													\
				}																																\
																																				\
				switch(fit_obj)																													\
				{																																\
					case 1:																														\
						output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale													\
							*((float)output[(l_o+7)*f_offset]-0.02f)																			\
							*((float)output[(l_o+7)*f_offset]-0.02f);																			\
						break;																													\
					case 0:																														\
						output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale													\
							*((float)output[(l_o+7)*f_offset]-0.5f)																				\
							*((float)output[(l_o+7)*f_offset]-0.5f);																			\
						break;																													\
					case -1:																													\
						output_error[(l_o+7)*f_offset] = 0.0f;																					\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_class; k++)																										\
				output_error[(l_o+8+k)*f_offset] = 0.0f;																						\
																																				\
			for(k = 0; k < nb_param; k++)																										\
				output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																				\
																																				\
		}																																		\
	}																																			\
}


#define typed_cuda_activ_fct_association(name)																									\
void typed_cuda_activ_fct_association_##name(network *net)																						\
{																																				\
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
	net->cu_inst.cu_auxil_fcts.cu_exp_disc_activation_kernel = exp_disc_activation_kernel_##name;												\
	net->cu_inst.cu_auxil_fcts.cu_exp_disc_deriv_output_kernel = exp_disc_deriv_output_kernel_##name;											\
}


ReLU_activation_kernel(FP32, float);
ReLU_deriv_kernel(FP32, float);
quadratic_deriv_output_error_kernel(FP32, float);
quadratic_output_error_kernel(FP32, float);
logistic_activation_kernel(FP32, float, expf);
logistic_deriv_kernel(FP32, float);
softmax_activation_kernel(FP32, float, expf);
cross_entropy_deriv_output_error_kernel(FP32, float);
cross_entropy_output_error_kernel(FP32, float);
exp_disc_activation_kernel(FP32, float, expf);
exp_disc_deriv_output_kernel(FP32, float);
YOLO_activation_kernel(FP32, float, expf);
YOLO_deriv_error_kernel(FP32, float);
YOLO_error_kernel(FP32, float);
typed_cuda_activ_fct_association(FP32);


#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
ReLU_activation_kernel(FP16, half);
ReLU_deriv_kernel(FP16, half);
quadratic_deriv_output_error_kernel(FP16, half);
quadratic_output_error_kernel(FP16, half);
logistic_activation_kernel(FP16, half, expf);
logistic_deriv_kernel(FP16, half);
softmax_activation_kernel(FP16, half, expf);
cross_entropy_deriv_output_error_kernel(FP16, half);
cross_entropy_output_error_kernel(FP16, half);
exp_disc_activation_kernel(FP16, half, expf);
exp_disc_deriv_output_kernel(FP16, half)
YOLO_activation_kernel(FP16, half, expf);
YOLO_deriv_error_kernel(FP16, half);
YOLO_error_kernel(FP16, half);
typed_cuda_activ_fct_association(FP16);
#endif


#if defined(GEN_AMPERE) 
ReLU_activation_kernel(BF16, nv_bfloat16);
ReLU_deriv_kernel(BF16, nv_bfloat16);
quadratic_deriv_output_error_kernel(BF16, nv_bfloat16);
quadratic_output_error_kernel(BF16, nv_bfloat16);
logistic_activation_kernel(BF16, nv_bfloat16, expf);
logistic_deriv_kernel(BF16, nv_bfloat16);
softmax_activation_kernel(BF16, nv_bfloat16, expf);
cross_entropy_deriv_output_error_kernel(BF16, nv_bfloat16);
cross_entropy_output_error_kernel(BF16, nv_bfloat16);
exp_disc_activation_kernel(BF16, nv_bfloat16, expf);
exp_disc_deriv_output_kernel(BF16, nv_bfloat16);
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
	
	current->c_network->cu_inst.cu_linear_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target,
		(param->biased_dim)*current->c_network->length, param->dim, param->biased_dim, param->offset, 
		param->size, current->c_network->TC_scale_factor);
}

void cuda_linear_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output,
		current->c_network->target, (param->biased_dim)*current->c_network->length, 
		param->dim, param->biased_dim, param->offset, param->size);
}


//#####################################################
//		 ReLU activation related functions
//#####################################################

void cuda_ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->size, param->dim, param->biased_dim, param->saturation, param->leaking_factor, param->size);
}


void cuda_ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	previous->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(previous->delta_o, previous->output, param->size, param->dim, param->biased_dim, param->saturation, param->leaking_factor, param->size);
}


// Should re write an output function to take into account ReLU for Conv output format
void cuda_ReLU_deriv_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target,
		(param->biased_dim) * current->c_network->length, param->dim, param->biased_dim, param->offset, 
		param->size, current->c_network->TC_scale_factor);
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, param->size, param->dim, param->biased_dim,
		param->saturation, param->leaking_factor, param->size);
}

void cuda_ReLU_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target, 
		(param->biased_dim)*current->c_network->length, param->dim, param->biased_dim, param->offset, param->size);
}


//#####################################################
//		 Logistic activation related functions
//#####################################################

void cuda_logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;

	current->c_network->cu_inst.cu_logistic_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->beta, param->saturation, (param->biased_dim)*current->c_network->length, 
		param->dim, param->biased_dim, param->size);
}


void cuda_logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	previous->c_network->cu_inst.cu_logistic_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(previous->delta_o, previous->output, param->beta, (param->biased_dim)*previous->c_network->length, 
		param->dim, param->biased_dim, param->size);
}


void cuda_logistic_deriv_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, (param->biased_dim)*current->c_network->length,
		 param->dim, param->biased_dim, param->offset, param->size, current->c_network->TC_scale_factor);
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, param->beta, (param->biased_dim)*current->c_network->length, 
		param->dim, param->biased_dim, param->size);
}

void cuda_logistic_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target,
		(param->biased_dim)*current->c_network->length, param->dim, param->biased_dim, param->offset, param->size);
}

//#####################################################
//		 Softmax activation related functions
//#####################################################

void cuda_softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, current->c_network->length, param->dim, param->biased_dim, param->offset, current->c_network->batch_size);
}


void cuda_softmax_deriv(layer *previous)
{
	printf("Error : Softmax activation can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}

void cuda_softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = ((param->biased_dim)*current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target,
		(param->biased_dim)*current->c_network->length, param->dim, param->biased_dim, param->offset,
		(param->biased_dim)*current->c_network->batch_size, current->c_network->TC_scale_factor);
}

void cuda_softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = ((param->biased_dim)*current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, 
		current->c_network->target, (param->biased_dim)*current->c_network->length,
		param->dim, param->biased_dim, param->offset, (param->biased_dim)*current->c_network->batch_size);
}

void cuda_semi_supervised_gan_deriv_output_error(layer *current, int halved, int reversed)
{
	//First half unsuperfvised fake	
	//Second half supervised true (for now)
	linear_param *param = (linear_param*)current->activ_param;
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	current->c_network->cu_inst.cu_auxil_fcts.cu_exp_disc_activation_kernel<<< cu_blocks, cu_threads >>>
		(current->output, current->c_network->batch_size, param->dim, current->c_network->length, halved, reversed);
	
	if(0 && current->c_network->epoch%10 == 0)
	{
		printf("in output\n");
		cuda_print_table(current->c_network, current->output, 11*current->c_network->batch_size, 11);
		//cuda_print_table(current->c_network, current->c_network->target, 10*current->c_network->batch_size, 10);
	}
	
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	current->c_network->cu_inst.cu_auxil_fcts.cu_exp_disc_deriv_output_kernel<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, current->c_network->length,
		 param->dim, current->c_network->batch_size, halved, reversed);
}

//#####################################################
//		 YOLO activation related functions
//#####################################################

void cuda_YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (current->c_network->out_size *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size,
		a_param->biased_dim*current->c_network->batch_size, *a_param, a_param->size, a_param->class_softmax);
}

void cuda_YOLO_deriv(layer *previous)
{
	printf("Error : YOLO activation can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}


void cuda_YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, 
		current->c_network->TC_scale_factor, current->c_network->epoch * current->c_network->train.size, c_param->block_state);
}


void cuda_YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = (c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
}

void cuda_YOLO_activ_init(layer *current)
{
	float *temp_tab, *temp_tab2, **temp_tab3;
	
	int nb_area_flat;

	yolo_param* a_param = (yolo_param*)current->activ_param;
	
	nb_area_flat = ((conv_param*)current->param)->nb_area[0]
		* ((conv_param*)current->param)->nb_area[1]
		* ((conv_param*)current->param)->nb_area[2];
	
	switch(((yolo_param*)a_param)->IoU_type)
	{
		case IOU:
			cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_IoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		default:
		case GIOU:
			cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_GIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		case DIOU:
			cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_DIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
		
		case DIOU2:
			cudaMemcpyFromSymbol(&((yolo_param*)a_param)->c_IoU_fct, device_gpu_DIoU2_fct, sizeof(pointFunction_gpu_IoU));
			break;
	}
	
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->prior_w,
		((yolo_param*)a_param)->nb_box,1);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->prior_h,
		((yolo_param*)a_param)->nb_box,1);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->prior_d,
		((yolo_param*)a_param)->nb_box,1);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->noobj_prob_prior,
		((yolo_param*)a_param)->nb_box,1);
		
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->scale_tab, 6,1);
	
	temp_tab = ((yolo_param*)a_param)->slopes_and_maxes_tab[0];
	cudaMalloc(&temp_tab2, 6*3*sizeof(float));
	cudaMemcpy(temp_tab2, temp_tab, 6*3*sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 6; i++)
		((yolo_param*)a_param)->slopes_and_maxes_tab[i] = &temp_tab2[i*3];
	temp_tab3 = ((yolo_param*)a_param)->slopes_and_maxes_tab;
	cudaMalloc(&((yolo_param*)a_param)->slopes_and_maxes_tab, 6*sizeof(float*));
	cudaMemcpy(((yolo_param*)a_param)->slopes_and_maxes_tab, temp_tab3,
			6*sizeof(float*), cudaMemcpyHostToDevice);
	
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->param_ind_scale,
		((yolo_param*)a_param)->nb_param,1);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->IoU_limits, 8,1);
	cuda_convert_table_int(&((yolo_param*)a_param)->fit_parts, 6,1);
	
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->IoU_monitor,
		2 *((yolo_param*)a_param)->nb_box * current->c_network->batch_size * nb_area_flat,0);
	cuda_convert_table_int(&((yolo_param*)a_param)->target_cell_mask,
		((yolo_param*)a_param)->max_nb_obj_per_image * current->c_network->batch_size * nb_area_flat,0);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->IoU_table,
		((yolo_param*)a_param)->max_nb_obj_per_image * ((yolo_param*)a_param)->nb_box 
		* current->c_network->batch_size * nb_area_flat,0);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->dist_prior,
		((yolo_param*)a_param)->max_nb_obj_per_image * ((yolo_param*)a_param)->nb_box 
		* current->c_network->batch_size * nb_area_flat,0);
	cuda_convert_table_int(&((yolo_param*)a_param)->box_locked,
		((yolo_param*)a_param)->nb_box * current->c_network->batch_size * nb_area_flat,0);
	cuda_convert_table_FP32((void**)&((yolo_param*)a_param)->box_in_pix,
		6 * ((yolo_param*)a_param)->nb_box * current->c_network->batch_size * nb_area_flat,0);
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
			
		case YOLO:
			current->activation = cuda_YOLO_activation;
			current->deriv_activation = cuda_YOLO_deriv;
			cuda_YOLO_activ_init(current);
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








