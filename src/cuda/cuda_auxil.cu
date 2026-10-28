
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
int cu_threads = CUDA_THREADS_PER_BLOCKS;
void *cu_alpha, *cu_beta;
float cu_f_alpha = 1.0f, cu_f_beta = 0.0f;
half cu_h_alpha = 1.0f, cu_h_beta = 0.0f;
cublasHandle_t cu_handle;
curandGenerator_t cu_gen;
cudaDataType cuda_data_type = CUDA_R_32F;
#if defined(CUDA_OLD)
cudaDataType cuda_compute_type = CUDA_R_32F;
#else
cublasComputeType_t cuda_compute_type = CUBLAS_COMPUTE_32F;
#endif


int set_TC_scale_factor_error_mem = 0;

// Public are in "prototypes.h"

// Private prototypes
void cuda_get_table_FP32_to_FP32(void *cuda_table, float *table, size_t size, void *buffer);
void cuda_convert_batched_table_FP32(void **tab, int batch_size, int nb_batch, size_t size);
void cuda_convert_batched_table(network* net, void **tab, int batch_size, int nb_batch, size_t size);
void cuda_get_batched_table(network* net, void **tab, int batch_size, int nb_batch, size_t size);
void cuda_convert_batched_host_table_FP32_to_FP32(void **tab, int batch_size, int nb_batch, size_t size);
__global__ void cuda_master_weight_FP32_to_FP32_kernel(float *master, void *copy, size_t size);

// Functions that result from templates are not listed here but at the end of the file instead


void cuda_set_TC_scale_factor(network* net, float val)
{
	if(net->cu_inst.use_cuda_TC == FP16C_FP32A || net->cu_inst.use_cuda_TC == FP16C_FP16A)
	{
		net->TC_scale_factor = val;
	}
	else if(!set_TC_scale_factor_error_mem)
	{
		if(val != 1.0f)
			printf("\nWARNING: Tried to set TC_scale_factor but the mixed precision mode is incompatible.\nScale kept to 1.\n");
		net->TC_scale_factor = 1.0f;
		set_TC_scale_factor_error_mem = 1; //prevent warning spam
	}
}


void cuda_sync(void)
{
	cudaDeviceSynchronize();
}


void cuda_free_table(void*tab)
{
	cudaFree(tab);
}


__global__ void init_block_state(unsigned int seed,  curandState_t* states, size_t size)
{
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		curand_init((seed << 20), /* the seed can be the same for each core, here we pass the time in from the CPU */
              i, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! 
			     Currently use an alternative definition with Id adjunct to seed*/
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[i]);
	}
}


void cuda_random_vector(float* tab, size_t size)
{
	curandGenerateUniform(cu_gen, tab, size);
}


#define cuda_host_copy_to(name, type, conversion_fct)																							\
void copy_to_##name(float* in_tab, void* out_tab, int out_offset, size_t size)																	\
{																																				\
	for(size_t i = 0; i < size; i++)																											\
		*((type*)out_tab + out_offset + i) = conversion_fct(*((float*)in_tab + i));																\
}


#define cuda_create_host_table_fct(name, type)																									\
void cuda_create_host_table_##name(void **tab, size_t size)																						\
{																																				\
	*tab = (type*) malloc(size*sizeof(type));																									\
}


void cuda_create_host_table(network* net, void **tab, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_create_host_table_fct(tab, size);
}


#define cuda_convert_table_fct(name, type)																										\
size_t cuda_convert_table_##name(void **tab, size_t size, int keep_host)																		\
{																																				\
	void* temp_tab;																																\
	size_t l_vram = 0;																															\
																																				\
	temp_tab = *tab;																															\
	type* temp_aux;																																\
																																				\
	if(strcmp(#name, "FP32") != 0)																												\
	{																																			\
		temp_aux = (type*) malloc(size*sizeof(type));																							\
		copy_to_##name((float*)temp_tab, temp_aux, 0, size);																					\
		free(temp_tab);																															\
	}																																			\
	else																																		\
		temp_aux = (type*) temp_tab;																											\
																																				\
	cudaMalloc(tab, size*sizeof(type));																											\
	l_vram += size*sizeof(type);																												\
	cudaMemcpy(*tab, temp_aux, size*sizeof(type),cudaMemcpyHostToDevice);																		\
	if(keep_host == 0)																															\
		free(temp_aux);																															\
																																				\
	return l_vram;																																\
}


size_t cuda_convert_table(network* net, void **tab, size_t size, int keep_host)
{	
	return net->cu_inst.cu_auxil_fcts.cu_convert_table_fct(tab, size, keep_host);
}


size_t cuda_convert_table_int(int **tab, size_t size, int keep_host)
{
	int* temp_tab;
	size_t l_vram = 0;
	
	temp_tab = *tab;

	cudaMalloc(tab, size*sizeof(int));
	cudaMemcpy(*tab, temp_tab, size*sizeof(int),cudaMemcpyHostToDevice);
	l_vram += size*sizeof(int);
	if(keep_host == 0)
		free(temp_tab);
	
	return l_vram;
}


#define cuda_float_typed_memset_kernel(name, type)																								\
__global__ void cuda_float_typed_memset_kernel_##name(void* i_table, size_t offset, float value, size_t size)									\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type* table = (type*) i_table;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	table[i + offset] = (type) value;																											\
}


void cuda_float_memset(network *net, void* table, size_t offset, float value, size_t size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	net->cu_inst.cu_auxil_fcts.cu_float_typed_memset_kernel<<< cu_blocks, cu_threads >>>(table, offset, value, size);
}


#define cuda_typed_memset(name, type)																											\
void cuda_typed_memset_##name(void* i_table, int value, size_t size)																			\
{																																				\
	type* table = (type*) i_table;																												\
																																				\
	cudaMemset(table,  value, size * sizeof(type));																								\
}


#define cuda_create_table_fct(name, type)																										\
void cuda_create_table_##name(void **tab, size_t size)																							\
{																																				\
	cudaMalloc(tab, size*sizeof(type));																											\
	cudaMemset(*tab, 0, size*sizeof(type));																										\
}


void cuda_create_table(network* net, void **tab, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_create_table_fct(tab, size);
}

void cuda_get_table_FP32_to_FP32(void *cuda_table, float *table, size_t size, void *buffer)
{
	cudaMemcpy((void*)table, (void*)cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
}


#define cuda_get_table_to_FP32_fct(name, type)																									\
void cuda_get_table_##name##_to_FP32(void *cuda_table, float *table, size_t size, void* buffer)													\
{																																				\
	type *temp;																																	\
	int mem = 0;																																\
																																				\
	if(buffer != NULL)																															\
		temp = (type*) buffer;																													\
	else																																		\
	{																																			\
		temp = (type*) malloc(size*sizeof(type));																								\
		mem = 1;																																\
	}																																			\
	cudaMemcpy(temp, cuda_table, size*sizeof(type), cudaMemcpyDeviceToHost);																	\
																																				\
	for(size_t i = 0; i < size; i++)																											\
		((float*)table)[i] = (float) temp[i];																									\
																																				\
	if(mem == 1)																																\
		free(temp);																																\
}


void cuda_get_table_to_FP32(network* net, void *cuda_table, float *table, size_t size, void* buffer)
{
	net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct(cuda_table, table, size, buffer);
}


#define cuda_get_table_host_to(name, type)																										\
void cuda_get_table_##name(void *cuda_table, void *table, size_t size)																			\
{																																				\
	cudaMemcpy(table, cuda_table, size*sizeof(type), cudaMemcpyDeviceToHost);																	\
}

void cuda_get_table(network* net, void *cuda_table, void *table, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_get_table_fct(cuda_table, table, size);
}


#define cuda_get_typed_host_table_fct(name, type)																								\
void cuda_get_typed_host_table_##name(void *typed_table, float *out_table, size_t size)															\
{																																				\
	for(size_t i = 0; i < size; i++)																											\
		out_table[i] = (float) ((type*)typed_table)[i];																							\
}

void cuda_get_typed_host_table(network* net, void *typed_table, float *out_table, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct(typed_table, out_table, size);
}


#define cuda_put_table_fct(name, type)																											\
void cuda_put_table_##name(void *cuda_table, void *table, size_t size)																			\
{																																				\
	cudaMemcpy((type*)cuda_table, (type*)table, size*sizeof(type), cudaMemcpyHostToDevice);														\
}


void cuda_put_table(network* net, void *cuda_table, void *table, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_put_table_fct(cuda_table, table, size);
}


#define cuda_host_FP32_to_device_typed_inplace_copy_kernel(name, type)																			\
__global__ void cuda_host_FP32_to_device_typed_inplace_copy_kernel_##name(void *cuda_table, float *buffer, size_t size)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type *l_table = (type*) cuda_table;																											\
																																				\
	if(i < size)																																\
		l_table[i] = (type) buffer[i];																											\
} 


void cuda_host_FP32_to_device_typed_inplace_copy(network *net, void *cuda_table, float *host_table, float *gpu_buffer, size_t size)
{
	cudaMemcpy(gpu_buffer, host_table, size*sizeof(float),cudaMemcpyHostToDevice);
	
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	net->cu_inst.cu_auxil_fcts.cu_host_FP32_to_device_typed_inplace_copy_kernel<<< cu_blocks, cu_threads >>>
		(cuda_table, gpu_buffer, size);
}


void cuda_convert_batched_table_FP32(void **tab, int batch_size, int nb_batch, size_t size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;

	for(i = 0; i < nb_batch; i++)
	{
		temp_tab = tab[i];
		cudaMalloc(&(tab[i]), batch_size*size*sizeof(float));
		cudaMemcpy(tab[i], temp_tab, batch_size*size*sizeof(float),cudaMemcpyHostToDevice);
		free(temp_tab);
	}
}


#define cuda_convert_batched_table_fct(name, type)																								\
void cuda_convert_batched_table_##name(void **tab, int batch_size, int nb_batch, size_t size)													\
{																																				\
	int i;																																		\
	void* temp_tab;																																\
																																				\
	temp_tab = *tab;																															\
	type *temp_aux;																																\
																																				\
	temp_aux = (type*) malloc(batch_size*size*sizeof(type));																					\
	for(i = 0; i < nb_batch; i++)																												\
	{																																			\
		temp_tab = tab[i];																														\
		copy_to_##name((float*)temp_tab, temp_aux, 0, batch_size*size);																			\
		free(temp_tab);																															\
		cudaMalloc(&(tab[i]), batch_size*size*sizeof(type));																					\
		cudaMemcpy(tab[i], temp_aux, batch_size*size*sizeof(type),cudaMemcpyHostToDevice);														\
	}																																			\
	free(temp_aux);																																\
}


void cuda_convert_batched_table(network* net, void **tab, int batch_size, int nb_batch, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct(tab, batch_size, nb_batch, size);
}

//Used only in the C interface to convert an existing array
void cuda_convert_dataset(network *net, Dataset *data)
{
	int b;
	//input device is just the pointer reference of all the batches 
	//in order to have access to it in the shuffle kernel
	
	data->input_device = (void***) malloc(data->nb_branch_in*sizeof(void**));
	
	for(b = 0; b < data->nb_branch_in; b++)
	{
		cuda_convert_batched_table(net, data->input[b], net->batch_size, data->nb_batch, data->branch_in_dim[b] + 1);
		cudaMalloc(&(data->input_device[b]), data->nb_batch*sizeof(void*));
		cudaMemcpy(data->input_device[b], data->input[b], data->nb_batch*sizeof(void*), cudaMemcpyHostToDevice);
	}
	
	if(data->target != NULL)
	{
		data->target_device = (void***) malloc(data->nb_branch_targ*sizeof(void**));
		
		for(b = 0; b < data->nb_branch_targ; b++)
		{
			cuda_convert_batched_table(net, data->target[b], net->batch_size, data->nb_batch, data->branch_targ_dim[b]);
			cudaMalloc(&(data->target_device[b]), data->nb_batch*sizeof(void*));
			cudaMemcpy(data->target_device[b], data->target[b], data->nb_batch*sizeof(void*), cudaMemcpyHostToDevice);
		}
	}
	data->localization = DEVICE;
}


#define cuda_get_batches_table_fct(name, type)																									\
void cuda_get_batched_table_##name(void **tab, int batch_size, int nb_batch, size_t size)														\
{																																				\
	int i;																																		\
	type* temp_tab = (type*) *tab;																												\
																																				\
	for(i = 0; i < nb_batch; i++)																												\
	{																																			\
		temp_tab = (type*) tab[i];																												\
		cudaMalloc(&(tab[i]), batch_size*size*sizeof(type));																					\
		cudaMemcpy(tab[i], temp_tab, batch_size*size*sizeof(type),cudaMemcpyHostToDevice);														\
		free(temp_tab);																															\
	}																																			\
}


void cuda_get_batched_table(network* net, void **tab, int batch_size, int nb_batch, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct(tab, batch_size, nb_batch, size);
}


void cuda_get_batched_dataset(network *net, Dataset *data)
{
	int b;
	//input device is just the pointer reference of all the batches 
	//in order to have access to it in the shuffle kernel

	data->input_device = (void***) malloc(data->nb_branch_in*sizeof(void**));
	
	for(b = 0; b < data->nb_branch_in; b++)
	{
		cuda_get_batched_table(net, data->input[b], net->batch_size, data->nb_batch, data->branch_in_dim[b] + 1);
		cudaMalloc(&(data->input_device[b]), data->nb_batch*sizeof(void*));
		cudaMemcpy(data->input_device[b], data->input[b], data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
	}
	
	if(data->target != NULL)
	{
		data->target_device = (void***) malloc(data->nb_branch_targ*sizeof(void**));
		
		for(b = 0; b < data->nb_branch_targ; b++)
		{
			cuda_get_batched_table(net, data->target[b], net->batch_size, data->nb_batch, data->branch_targ_dim[b]);
			cudaMalloc(&(data->target_device[b]), data->nb_batch*sizeof(void*));
			cudaMemcpy(data->target_device[b], data->target[b], data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
		}
	}
	data->localization = DEVICE;
}


void cuda_convert_batched_host_table_FP32_to_FP32(void **tab, int batch_size, int nb_batch, size_t size)
{
	//empty on purpose
}


#define cuda_convert_batched_host_table_FP32_to(name, type)																						\
void cuda_convert_batched_host_table_FP32_to_##name																								\
	(void **tab, int batch_size, int nb_batch, size_t size)																						\
{																																				\
	int i;																																		\
	void* temp_tab;																																\
																																				\
	temp_tab = *tab;																															\
																																				\
	for(i = 0; i < nb_batch; i++)																												\
	{																																			\
		temp_tab = tab[i];																														\
		tab[i] = (type*) malloc(batch_size*size*sizeof(type));																					\
		copy_to_##name((float*)temp_tab, ((type*)tab[i]), 0, batch_size*size);																	\
		free(temp_tab);																															\
	}																																			\
}

//Used only in the C interface to convert an existing array
void cuda_convert_host_dataset(network *net, Dataset *data)
{
	int b;
	
	for(b = 0; b < data->nb_branch_in; b++)
		net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct
			(data->input[b], net->batch_size, data->nb_batch, data->branch_in_dim[b] + 1);
	
	if(data->target != NULL)
	{
		for(b = 0; b < data->nb_branch_targ; b++)
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct
				(data->target[b], net->batch_size, data->nb_batch, data->branch_targ_dim[b]);
	}
	data->localization = HOST;
}


#define cuda_create_dataset_fct(name, type)																										\
Dataset cuda_create_dataset_##name(network *net, int with_target, size_t nb_elem, 																\
	int nb_branch_in, size_t *branch_in_dim, int nb_branch_targ, size_t *branch_targ_dim)														\
{																																				\
	int i, b;																																	\
	Dataset data;																																\
																																				\
	data.nb_branch_in   = nb_branch_in;																											\
	data.nb_branch_targ = nb_branch_targ;																										\
	data.size           = nb_elem;																												\
	data.localization   = HOST;																													\
	data.cont_copy      = copy_to_##name;																										\
	data.nb_batch       = (data.size - 1) / net->batch_size + 1;																				\
																																				\
	data.input = (void***) malloc(nb_branch_in*sizeof(void**));																					\
	data.branch_in_dim = (size_t*) malloc(nb_branch_in*sizeof(size_t));																			\
																																				\
	for(b = 0; b < nb_branch_in; b++)																											\
	{																																			\
		data.input[b] = (void**) malloc(data.nb_batch*sizeof(type*));																			\
		data.branch_in_dim[b] = branch_in_dim[b];																								\
																																				\
		for(i = 0; i < data.nb_batch; i++)																										\
			((type**)data.input[b])[i] = (type*) calloc(net->batch_size * (branch_in_dim[b] + 1), sizeof(type));								\
	}																																			\
																																				\
	if(with_target)																																\
	{																																			\
		data.target = (void***) malloc(nb_branch_targ*sizeof(void**));																			\
		data.branch_targ_dim = (size_t*) malloc(nb_branch_targ*sizeof(size_t));																	\
																																				\
		for(b = 0; b < nb_branch_targ; b++)																										\
		{																																		\
			data.target[b] = (void**) malloc(data.nb_batch*sizeof(type*));																		\
			data.branch_targ_dim[b] = branch_targ_dim[b];																						\
																																				\
			for(i = 0; i < data.nb_batch; i++)																									\
				((type**)data.target[b])[i] = (type*) calloc(net->batch_size * branch_targ_dim[b], sizeof(type));								\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		data.target = NULL;																														\
		data.branch_targ_dim = NULL;																											\
	}																																			\
	return data;																																\
}


Dataset cuda_create_dataset(network *net, int with_target, size_t nb_elem, 
	int nb_branch_in, size_t *branch_in_dim, int nb_branch_targ, size_t *branch_targ_dim)
{
	return net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct(net, with_target, 
		nb_elem, nb_branch_in, branch_in_dim, nb_branch_targ, branch_targ_dim);
}


void cuda_free_dataset(Dataset *data)
{
	int i, b;
	
	if(data->input != NULL)
	{
		for(b = 0; b < data->nb_branch_in; b++)
		{
			if(data->input[b] != NULL)
				for(i = 0; i < data->nb_batch; i++)
					cudaFree(data->input[b][i]);
			if(data->input_device[b] != NULL)
				cudaFree(data->input_device[b]);
		}
	}
	
	if(data->target != NULL)
	{
		for(b = 0; b < data->nb_branch_targ; b++)
		{
			if(data->target[b] != NULL)
				for(i = 0; i < data->nb_batch; i++)
					cudaFree(data->target[b][i]);
			if(data->target_device[b] != NULL)
				cudaFree(data->target_device[b]);
		}
	}
}


__global__ void cuda_master_weight_FP32_to_FP32_kernel(float *master, void *copy, size_t size)
{
	//nothing to do
}


#define cuda_master_weight_FP32_to_kernel(name, type)																							\
__global__ void cuda_master_weight_FP32_to_##name##_kernel(float *master, void *copy, size_t size)												\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	if(i < size)																																\
		((type*)copy)[i] = (type) master[i];																									\
}


void cuda_master_weight_copy(network* net, float *master, void *copy, size_t size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel<<< cu_blocks, cu_threads >>>(master, copy, size);
}



#define cuda_print_table_fct(name, type)																										\
void cuda_print_table_##name(void* tab, size_t size, int return_every)																			\
{																																				\
	size_t i;																																	\
	void *temp;																																	\
																																				\
	temp = (void*) malloc(size*sizeof(type));																									\
	cudaMemcpy(temp, tab, size*sizeof(type), cudaMemcpyDeviceToHost);																			\
	for(i = 0; i < size; i ++)																													\
	{																																			\
		if(i%return_every == 0)																													\
			printf("\n");																														\
		printf("%g \t ", (float)((type*)temp)[i]);																								\
	}																																			\
																																				\
	free(temp);																																	\
}


void cuda_print_table(network* net, void* tab, size_t size, int return_every)
{	
	net->cu_inst.cu_auxil_fcts.cu_print_table_fct(tab, size, return_every);
	printf("\n");
}


void cuda_print_table_int(int* tab, size_t size, int return_every)
{
	size_t i;
	int *temp = NULL;
	
	printf("\n");
	temp = (int*) malloc(size * sizeof(int));
	cudaMemcpy(temp, tab, size * sizeof(int), cudaMemcpyDeviceToHost);
	for(i = 0; i < size; i ++)
	{
		if(i%return_every == 0)
			printf("\n");
		printf("%d ", (int)(((int*)temp)[i]));
	}
	printf("\n");
	
	free(temp);
}


#define cuda_print_table_host_fct(name, type)																									\
void cuda_print_table_host_##name(void* tab, size_t size, int return_every)																		\
{																																				\
	size_t i;																																	\
	for(i = 0; i < size; i ++)																													\
	{																																			\
		if(i%return_every == 0)																													\
			printf("\n");																														\
		printf("%g \t ", (float)(((type*)tab)[i]));																								\
	}																																			\
}


void* cuda_timer_init(void)
{
	cudaEvent_t *cuda_timer;
	cuda_timer = (cudaEvent_t*) malloc(2*sizeof(cudaEvent_t)),
	cudaEventCreate(&cuda_timer[0]);
	cudaEventCreate(&cuda_timer[1]);
	
	return (void*)cuda_timer;
}


void cuda_timer_start(void* timer)
{
	cudaEventRecord(((cudaEvent_t*)timer)[0]);
}


double cuda_ellapsed_time(void *timer)
{
	float time = 0.0f;
	cudaEvent_t* t_timer = (cudaEvent_t*) timer;
	
	cudaEventRecord(t_timer[1]);
	cudaEventSynchronize(t_timer[1]);
	cudaEventElapsedTime(&time, t_timer[0], t_timer[1]);
	
	return (time*1000); //microseconds
}

void cuda_timer_free(void *timer)
{
	cudaFree(timer);
}


#define shfl_kern_fct(name, type)																												\
__global__ void shfl_kern_##name(void **i_tab, void **i_dupl, int *index, size_t nb_elem, int b_size, size_t i_dim)								\
{																																				\
	int j, batch, batch2, pos, pos2;																											\
																																				\
	type **tab  = (type**) i_tab;																												\
	type **dupl = (type**) i_dupl;																												\
																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i < nb_elem)																																\
	{																																			\
		pos = i%b_size;																															\
		batch = i/b_size;																														\
		pos2 = (int)(index[i])%b_size;																											\
		batch2 = (int)(index[i])/b_size;																										\
		for(j = 0; j < i_dim; j++)																												\
			dupl[batch2][pos2*i_dim + j] = tab[batch][pos*i_dim + j];																			\
	}																																			\
}


#define get_back_shuffle_fct(name, type)																										\
__global__ void get_back_shuffle_##name																											\
	(void **i_tab, void **i_dupl, int *index, size_t nb_elem, int b_size, size_t i_dim)															\
{																																				\
	size_t j;																																	\
	int batch, pos;																																\
																																				\
	type **tab  = (type**) i_tab;																												\
	type **dupl = (type**) i_dupl;																												\
																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i < nb_elem)																																\
	{																																			\
		pos = i%b_size;																															\
		batch = i/b_size;																														\
		for(j = 0; j < i_dim; j++)																												\
			tab[batch][pos*i_dim + j] = dupl[batch][pos*i_dim + j];																				\
	}																																			\
}


void cuda_shuffle(network *net, Dataset *data, Dataset *duplicate, int *index_shuffle, int *index_shuffle_device)
{
	int i, j, b;
	int temp;

	for(i = 0; i+1 < data->size; i++)
	{
		j = i + random_int_below(data->size - i);
		temp = index_shuffle[i];
		index_shuffle[i] = index_shuffle[j];
		index_shuffle[j] = temp;
	}
	
	cudaMemcpy(index_shuffle_device, index_shuffle, data->size*sizeof(int), cudaMemcpyHostToDevice);
	cu_blocks = (data->size + cu_threads - 1) / cu_threads;
	
	for(b = 0; b < data->nb_branch_in; b++)
	{
		net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct<<< cu_blocks, cu_threads>>>
			(data->input_device[b], duplicate->input_device[b], index_shuffle_device, 
			data->size, net->batch_size, data->branch_in_dim[b] + 1);
	}
	
	for(b = 0; b < data->nb_branch_targ; b++)
	{
		net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct<<< cu_blocks, cu_threads>>>
			(data->target_device[b], duplicate->target_device[b], index_shuffle_device, 
			data->size, net->batch_size, data->branch_targ_dim[b]);
	}
}


#define host_shuffle_typed(name, type)																											\
void cuda_host_shuffle_##name(network *net, Dataset *data, Dataset *duplicate)																	\
{																																				\
	int i, j, b;																																\
	size_t k;																																	\
	type temp;																																	\
	int pos, pos2, batch, batch2;																												\
																																				\
	type *d_in_A, *d_in_B, *d_targ_A, *d_targ_B;																								\
																																				\
	for(b = 0; b < data->nb_branch_in; b++)																										\
		for(i = 0; i < data->nb_batch; i++)																										\
			cudaMemcpy(duplicate->input[b][i], data->input[b][i], net->batch_size 																\
				* (data->branch_in_dim[b] + 1)*sizeof(type), cudaMemcpyDeviceToHost);															\
																																				\
	if(data->target != NULL)																													\
		for(b = 0; b < data->nb_branch_targ; b++)																								\
			for(i = 0; i < data->nb_batch; i++)																									\
				cudaMemcpy(duplicate->target[b][i], data->target[b][i], net->batch_size 														\
					* (data->branch_targ_dim[b])*sizeof(type), cudaMemcpyDeviceToHost);															\
																																				\
	for(i = 0; i+1 < data->size; i++)																											\
	{																																			\
		j = i + random_int_below(data->size - i);																								\
		pos = i%net->batch_size;																												\
		batch = i/net->batch_size;																												\
		pos2 = (int)(j)%net->batch_size;																										\
		batch2 = (int)(j)/net->batch_size;																										\
																																				\
		for(b = 0; b < data->nb_branch_in; b++)																									\
		{																																		\
			d_in_A = ((type*) duplicate->input[b][batch]);																						\
			d_in_B = ((type*) duplicate->input[b][batch2]);																						\
																																				\
			for(k = 0; k < data->branch_in_dim[b] + 1; k++)																						\
			{																																	\
				temp = d_in_A[pos*(data->branch_in_dim[b] + 1) + k];																			\
				d_in_A[pos*(data->branch_in_dim[b] + 1) + k] = d_in_B[pos2*(data->branch_in_dim[b] + 1) + k];									\
				d_in_B[pos2*(data->branch_in_dim[b] + 1) + k] = temp;																			\
			}																																	\
		}																																		\
																																				\
		for(b = 0; b < data->nb_branch_targ; b++)																								\
		{																																		\
			if(data->target != NULL)																											\
			{																																	\
				d_targ_A = ((type*) duplicate->target[b][batch]);																				\
				d_targ_B = ((type*) duplicate->target[b][batch2]);																				\
																																				\
				for(k = 0; k < data->branch_targ_dim[b]; k++)																					\
				{																																\
					temp = d_targ_A[pos*data->branch_targ_dim[b] + k];																			\
					d_targ_A[pos*data->branch_targ_dim[b] + k] = d_targ_B[pos2*data->branch_targ_dim[b] + k];									\
					d_targ_B[pos2*data->branch_targ_dim[b] + k] = temp;																			\
				}																																\
			}																																	\
		}																																		\
	}																																			\
																																				\
	for(b = 0; b < data->nb_branch_in; b++)																										\
		for(i = 0; i < data->nb_batch; i++)																										\
			cudaMemcpy(data->input[b][i], duplicate->input[b][i], net->batch_size 																\
				* (data->branch_in_dim[b] + 1)*sizeof(type), cudaMemcpyHostToDevice);															\
																																				\
	if(data->target != NULL)																													\
		for(b = 0; b < data->nb_branch_targ; b++)																								\
			for(i = 0; i < data->nb_batch; i++)																									\
				cudaMemcpy(data->target[b][i], duplicate->target[b][i], net->batch_size 														\
					* data->branch_targ_dim[b]*sizeof(type), cudaMemcpyHostToDevice);															\
}


void cuda_host_shuffle(network *net, Dataset *data, Dataset *duplicate)
{
	net->cu_inst.cu_auxil_fcts.cu_host_shuffle_fct(net, data, duplicate);
}


#define cuda_host_only_shuffle_type(name, type)																									\
void cuda_host_only_shuffle_##name(network *net, Dataset *data)																					\
{																																				\
	int i, j, b;																																\
	size_t k;																																	\
	type temp;																																	\
	int pos, pos2, batch, batch2;																												\
	type **c_input, **c_target;																													\
																																				\
	for(i = 0; i < data->size - 1; i++)																											\
	{																																			\
		j = i + random_int_below(data->size - i);																								\
		pos = i%net->batch_size;																												\
		batch = i/net->batch_size;																												\
		pos2 = j%net->batch_size;																												\
		batch2 = j/net->batch_size;																												\
																																				\
		for(b = 0; b < data->nb_branch_in; b++)																									\
		{																																		\
			c_input = (type**)data->input[b];																									\
																																				\
			for(k = 0; k < data->branch_in_dim[b]+1; k++)																						\
			{																																	\
				temp = ((type*)c_input[batch])[pos*(data->branch_in_dim[b] + 1) + k];															\
				((type*)c_input[batch])[pos*(data->branch_in_dim[b] + 1) + k] = 																\
					((type*)c_input[batch2])[pos2*(data->branch_in_dim[b] + 1) + k];															\
				((type*)c_input[batch2])[pos2*(data->branch_in_dim[b] + 1) + k] = temp;															\
			}																																	\
		}																																		\
																																				\
		if(data->target != NULL)																												\
		{																																		\
			for(b = 0; b < data->nb_branch_targ; b++)																							\
			{																																	\
				c_target = (type**)data->target[b];																								\
																																				\
				for(k = 0; k < data->branch_targ_dim[b]; k++)																					\
				{																																\
					temp = ((type*)c_target[batch])[pos*data->branch_targ_dim[b] + k];															\
					((type*)c_target[batch])[pos*data->branch_targ_dim[b] + k] = 																\
						((type*)c_target[batch2])[pos2*data->branch_targ_dim[b] + k];															\
					((type*)c_target[batch2])[pos2*data->branch_targ_dim[b] + k] = temp;														\
				}																																\
			}																																	\
		}																																		\
	}																																			\
}


void cuda_host_only_shuffle(network *net, Dataset *data)
{
	net->cu_inst.cu_auxil_fcts.cu_host_only_shuffle_fct(net, data);
}


cuda_host_copy_to(FP32, float, ); //last argument empty en purpose
cuda_create_host_table_fct(FP32, float);
cuda_convert_table_fct(FP32, float);
cuda_create_table_fct(FP32, float);
cuda_float_typed_memset_kernel(FP32, float);
cuda_typed_memset(FP32, float);
cuda_get_table_host_to(FP32, float);
cuda_get_typed_host_table_fct(FP32, float);
cuda_put_table_fct(FP32, float);
cuda_host_FP32_to_device_typed_inplace_copy_kernel(FP32, float);
cuda_get_batches_table_fct(FP32, float);
cuda_create_dataset_fct(FP32, float);
cuda_print_table_fct(FP32, float);
cuda_print_table_host_fct(FP32, float);
shfl_kern_fct(FP32, float);
get_back_shuffle_fct(FP32, float);
host_shuffle_typed(FP32, float);
cuda_host_only_shuffle_type(FP32, float);


#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
cuda_host_copy_to(FP16, half, __float2half_rz);
cuda_create_host_table_fct(FP16, half);
cuda_convert_table_fct(FP16, half);
cuda_create_table_fct(FP16, half);
cuda_float_typed_memset_kernel(FP16, half);
cuda_typed_memset(FP16, half);
cuda_get_table_to_FP32_fct(FP16, half);
cuda_get_table_host_to(FP16, half);
cuda_get_typed_host_table_fct(FP16, half);
cuda_put_table_fct(FP16, half);
cuda_host_FP32_to_device_typed_inplace_copy_kernel(FP16, half);
cuda_convert_batched_table_fct(FP16, half);
cuda_get_batches_table_fct(FP16, half);
cuda_convert_batched_host_table_FP32_to(FP16, half);
cuda_create_dataset_fct(FP16, half);
cuda_master_weight_FP32_to_kernel(FP16, half);
cuda_print_table_fct(FP16, half);
cuda_print_table_host_fct(FP16, half);
shfl_kern_fct(FP16, half);
get_back_shuffle_fct(FP16, half);
host_shuffle_typed(FP16, half);
cuda_host_only_shuffle_type(FP16, half);
#endif

#if defined(GEN_AMPERE)
cuda_host_copy_to(BF16, nv_bfloat16, __float2bfloat16_rz);
cuda_create_host_table_fct(BF16, nv_bfloat16);
cuda_convert_table_fct(BF16, nv_bfloat16);
cuda_create_table_fct(BF16, nv_bfloat16);
cuda_float_typed_memset_kernel(BF16, nv_bfloat16);
cuda_typed_memset(BF16, nv_bfloat16);
cuda_get_table_to_FP32_fct(BF16, nv_bfloat16);
cuda_get_table_host_to(BF16, nv_bfloat16);
cuda_get_typed_host_table_fct(BF16, nv_bfloat16);
cuda_put_table_fct(BF16, nv_bfloat16);
cuda_host_FP32_to_device_typed_inplace_copy_kernel(BF16, nv_bfloat16);
cuda_convert_batched_table_fct(BF16, nv_bfloat16);
cuda_get_batches_table_fct(BF16, nv_bfloat16);
cuda_convert_batched_host_table_FP32_to(BF16, nv_bfloat16);
cuda_create_dataset_fct(BF16, nv_bfloat16);
cuda_master_weight_FP32_to_kernel(BF16, nv_bfloat16);
cuda_print_table_fct(BF16, nv_bfloat16);
cuda_print_table_host_fct(BF16, nv_bfloat16);
shfl_kern_fct(BF16, nv_bfloat16);
get_back_shuffle_fct(BF16, nv_bfloat16);
host_shuffle_typed(BF16, nv_bfloat16);
cuda_host_only_shuffle_type(BF16, nv_bfloat16);
#endif


void init_auxil_cuda(network* net)
{	
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			net->cu_inst.cu_auxil_fcts.cu_create_host_table_fct = cuda_create_host_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_convert_table_fct = cuda_convert_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_create_table_fct = cuda_create_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_float_typed_memset_kernel = cuda_float_typed_memset_kernel_FP32;
			net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct = cuda_typed_memset_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_FP32_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_host_FP32_to_device_typed_inplace_copy_kernel = cuda_host_FP32_to_device_typed_inplace_copy_kernel_FP32;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_FP32_kernel;
			net->cu_inst.cu_auxil_fcts.cu_print_table_fct = cuda_print_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct = shfl_kern_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_back_shuffle_fct = get_back_shuffle_FP32;
			net->cu_inst.cu_auxil_fcts.cu_host_shuffle_fct = cuda_host_shuffle_FP32;
			net->cu_inst.cu_auxil_fcts.cu_host_only_shuffle_fct = cuda_host_only_shuffle_FP32;
			break;
		
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			net->cu_inst.cu_auxil_fcts.cu_create_host_table_fct =cuda_create_host_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_convert_table_fct = cuda_convert_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_create_table_fct = cuda_create_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_float_typed_memset_kernel = cuda_float_typed_memset_kernel_FP16;
			net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct = cuda_typed_memset_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_FP16_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_host_FP32_to_device_typed_inplace_copy_kernel = cuda_host_FP32_to_device_typed_inplace_copy_kernel_FP16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_FP16;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_FP16_kernel;
			net->cu_inst.cu_auxil_fcts.cu_print_table_fct = cuda_print_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct = shfl_kern_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_back_shuffle_fct = get_back_shuffle_FP16;
			net->cu_inst.cu_auxil_fcts.cu_host_shuffle_fct = cuda_host_shuffle_FP16;
			net->cu_inst.cu_auxil_fcts.cu_host_only_shuffle_fct = cuda_host_only_shuffle_FP16;
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;

		case BF16C_FP32A:
			#if defined (GEN_AMPERE)
			net->cu_inst.cu_auxil_fcts.cu_create_host_table_fct = cuda_create_host_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_convert_table_fct = cuda_convert_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_create_table_fct = cuda_create_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_float_typed_memset_kernel = cuda_float_typed_memset_kernel_BF16;
			net->cu_inst.cu_auxil_fcts.cu_typed_memset_fct = cuda_typed_memset_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_BF16_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_host_FP32_to_device_typed_inplace_copy_kernel = cuda_host_FP32_to_device_typed_inplace_copy_kernel_BF16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_BF16;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_BF16_kernel;
			net->cu_inst.cu_auxil_fcts.cu_print_table_fct = cuda_print_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct = shfl_kern_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_back_shuffle_fct = get_back_shuffle_BF16;
			net->cu_inst.cu_auxil_fcts.cu_host_shuffle_fct = cuda_host_shuffle_BF16;
			net->cu_inst.cu_auxil_fcts.cu_host_only_shuffle_fct = cuda_host_only_shuffle_BF16;
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
	
}


void cuda_init(network* net, char* string_comp, const char* cuda_TC_string, int dynamic_load)
{
	int c_mixed_precision = FP32C_FP32A;
		
	if(strcmp(cuda_TC_string,"off") == 0)
	{
		c_mixed_precision = FP32C_FP32A;
		sprintf(string_comp+5, "(FP32C_FP32A)");
	}
	else if(strcmp(cuda_TC_string,"on") == 0)
	{
		c_mixed_precision = FP16C_FP32A;
		sprintf(string_comp+5, "(FP16C_FP32A)");
	}
	else if(strcmp(cuda_TC_string,"FP32C_FP32A") == 0)
	{
		c_mixed_precision = FP32C_FP32A;
		sprintf(string_comp+5, "(FP32C_FP32A)");
	}
	else if(strcmp(cuda_TC_string,"TF32C_FP32A") == 0)
	{
		c_mixed_precision = TF32C_FP32A;
		sprintf(string_comp+5, "(TF32C_FP32A)");
	}
	else if(strcmp(cuda_TC_string,"FP16C_FP32A") == 0)
	{
		c_mixed_precision = FP16C_FP32A;
		sprintf(string_comp+5, "(FP16C_FP32A)");
	}
	else if(strcmp(cuda_TC_string,"FP16C_FP16A") == 0)
	{
		c_mixed_precision = FP16C_FP16A;
		sprintf(string_comp+5, "(FP16C_FP16A)");
	}
	else if(strcmp(cuda_TC_string,"BF16C_FP32A") == 0)
	{
		c_mixed_precision = BF16C_FP32A;
		sprintf(string_comp+5, "(BF16C_FP32A)");
	}
	
	net->cu_inst.dynamic_load = dynamic_load;
	net->cu_inst.use_cuda_TC = c_mixed_precision;

	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

	stat = cublasCreate(&cu_handle);

	if(stat != CUBLAS_STATUS_SUCCESS)
	{
		switch(stat)
		{
			case CUBLAS_STATUS_SUCCESS:
			    printf("CUBLAS_STATUS_SUCCESS");
			    break;
			case CUBLAS_STATUS_NOT_INITIALIZED:
			    printf("CUBLAS_STATUS_NOT_INITIALIZED");
				break;
			case CUBLAS_STATUS_ALLOC_FAILED:
			    printf("CUBLAS_STATUS_ALLOC_FAILED");
				break;
			case CUBLAS_STATUS_INVALID_VALUE:
			    printf("CUBLAS_STATUS_INVALID_VALUE");
				break;
			case CUBLAS_STATUS_ARCH_MISMATCH:
			    printf("CUBLAS_STATUS_ARCH_MISMATCH");
				break;
			case CUBLAS_STATUS_MAPPING_ERROR:
			    printf("CUBLAS_STATUS_MAPPING_ERROR");
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED:
			    printf("CUBLAS_STATUS_EXECUTION_FAILED");
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR:
			    printf("CUBLAS_STATUS_INTERNAL_ERROR");
				break;
			default:
				break;
		}

		printf("\n ERROR: GPU handle create fail\n");
		exit(EXIT_FAILURE);
	}
	
	//CUDA version <= 11.0
	#if defined(CUDA_OLD)
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_32F;
			cuda_compute_type = CUDA_R_32F;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;
		
		#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
		case FP16C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_TENSOR_OP_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUDA_R_32F;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;
			
		case FP16C_FP16A:
			cublasSetMathMode(cu_handle, CUBLAS_TENSOR_OP_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUDA_R_16F;
			cu_alpha = &cu_h_alpha; cu_beta = &cu_h_beta;
			break;
		#endif
	}

	//CUDA version >= 11.1
	#else
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_PEDANTIC_MATH);
			cuda_data_type = CUDA_R_32F;
			cuda_compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;

		#if defined(GEN_AMPERE) 
		case TF32C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_TF32_TENSOR_OP_MATH);
			cuda_data_type = CUDA_R_32F;
			cuda_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;
		#endif
		
		#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
		case FP16C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUBLAS_COMPUTE_32F;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;
			
		case FP16C_FP16A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUBLAS_COMPUTE_16F;
			cu_alpha = &cu_h_alpha; cu_beta = &cu_h_beta;
			break;
		#endif
		
		#if defined(GEN_AMPERE)
		case BF16C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16BF;
			cuda_compute_type = CUBLAS_COMPUTE_32F;
			cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
			break;
		#endif
	}
	#endif
	
	//set typed function according to USE_CUDA_TC
	curandCreateGenerator(&cu_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(cu_gen, time(NULL));

	init_auxil_cuda(net);
	
	init_typed_cuda_optimizer(net);
	init_typed_cuda_activ(net);
	cuda_dense_init(net);
	cuda_conv_init(net);
	cuda_pool_init(net);
	cuda_norm_init(net);
	cuda_lrn_init(net);
	cuda_grn_init(net);
	cuda_merge_init(net);
	
	cuda_set_optimizer_update_function(net);
	
	cuda_convert_table(net, &net->auxil_input->output, net->batch_size*2, 0);
}


void cuda_free_network(network *net)
{
	cudaFree(net->auxil_input->output);

	cublasDestroy(cu_handle);
	curandDestroyGenerator(cu_gen);
}


