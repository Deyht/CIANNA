
/*
	Copyright (C) 2024 David Cornu
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
int cu_threads = CUDA_THREADS_PER_BLOCKS;
void *cu_alpha, *cu_beta;
void *cu_learning_rate, *cu_momentum;
float cu_f_alpha = 1.0f, cu_f_beta = 0.0f;
half cu_h_alpha = 1.0f, cu_h_beta = 0.0f;
float cu_f_learning_rate = 1.0f, cu_f_momentum = 0.0f;
half cu_h_learning_rate = 1.0f, cu_h_momentum = 0.0f;
cublasHandle_t cu_handle;
curandGenerator_t cu_gen;
cudaDataType cuda_data_type = CUDA_R_32F;
#if defined(CUDA_OLD)
cudaDataType cuda_compute_type = CUDA_R_32F;
#else
cublasComputeType_t cuda_compute_type = CUBLAS_COMPUTE_32F;
#endif

cudaEvent_t cu_perf_start, cu_perf_stop;
cudaEvent_t cu_batch_start, cu_batch_stop;
cudaEvent_t cu_epoch_start, cu_epoch_stop;
int set_TC_scale_factor_error_mem = 0;

//local prototypes

void set_cu_learning_rate_and_momentum(network* net)
{
	if(net->cu_inst.use_cuda_TC == FP16C_FP16A)
	{
		cu_h_learning_rate = net->learning_rate/net->batch_size;
		cu_h_momentum = net->momentum;
	}
	else
	{
		cu_f_learning_rate = net->learning_rate/net->batch_size;
		cu_f_momentum = net->momentum;
	}
}

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

void cuda_free_table(void* tab)
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

#define cuda_create_table_fct(name, type)																										\
void cuda_create_table_##name(void **tab, size_t size)																							\
{																																				\
	cudaMalloc(tab, size*sizeof(type));																											\
	cudaMemset(*tab, 0, size*sizeof(type));																										\
}

void cuda_set_mem_value(void* device_mem_loc, float value, size_t size)
{
	void *temp;
	
	temp = (void*) &value;
	cudaMemcpy(device_mem_loc, temp, size, cudaMemcpyHostToDevice);
}



void cuda_create_table(network* net, void **tab, size_t size)
{
	net->cu_inst.cu_auxil_fcts.cu_create_table_fct(tab, size);
}

void cuda_get_table_FP32_to_FP32(void *cuda_table, float *table, size_t size, void* buffer)
{
	cudaMemcpy(table, cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
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

void cuda_convert_dataset(network *net, Dataset *data)
{
	cuda_convert_batched_table(net, data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
	cuda_convert_batched_table(net, data->target, net->batch_size, data->nb_batch, net->output_dim);
	//input device is just the pointer reference of all the batches 
	//in order to have access to it in the shuffle kernel
	cudaMalloc(&(data->input_device), data->nb_batch*sizeof(void*));
	cudaMemcpy(data->input_device, data->input, data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
	cudaMalloc(&(data->target_device), data->nb_batch*sizeof(void*));
	cudaMemcpy(data->target_device, data->target, data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
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
	cuda_get_batched_table(net, data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
	cuda_get_batched_table(net, data->target, net->batch_size, data->nb_batch, net->output_dim);
	
	cudaMalloc(&(data->input_device), data->nb_batch*sizeof(void*));
	cudaMemcpy(data->input_device, data->input, data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
	cudaMalloc(&(data->target_device), data->nb_batch*sizeof(void*));
	cudaMemcpy(data->target_device, data->target, data->nb_batch*sizeof(void*),cudaMemcpyHostToDevice);
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


void cuda_convert_host_dataset(network *net, Dataset *data)
{
	net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct
		(data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
	data->localization = HOST;
}


#define cuda_create_dataset_fct(name, type)																										\
Dataset cuda_create_dataset_##name(network *net, int nb_elem)																					\
{																																				\
	int i, j;																																	\
	Dataset data;																																\
																																				\
	data.size = nb_elem;																														\
	data.nb_batch = (data.size - 1) / net->batch_size + 1;																						\
	data.input = (void**) malloc(data.nb_batch*sizeof(type*));																					\
	data.target = (void**) malloc(data.nb_batch*sizeof(type*));																					\
	data.localization = HOST;																													\
	data.cont_copy = copy_to_##name;																											\
																																				\
	for(i = 0; i < data.nb_batch; i++)																											\
	{																																			\
		((type**)data.input)[i] = (type*) calloc(net->batch_size * (net->input_dim + 1), sizeof(type));											\
		((type**)data.target)[i] = (type*) calloc(net->batch_size * net->output_dim, sizeof(type));												\
	}																																			\
																																				\
	for(i = 0; i < data.nb_batch; i++)																											\
		for(j = 0; j < net->batch_size; j++)																									\
			((type**)data.input)[i][j*(net->input_dim+1) + net->input_dim] = net->input_bias;													\
																																				\
	return data;																																\
}

Dataset cuda_create_dataset(network *net, int nb_elem)
{
	return net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct(net, nb_elem);
}


void cuda_free_dataset(Dataset *data)
{
	int i;
	
	if(data->input != NULL)
	{
		for(i = 0; i < data->nb_batch; i++)
		{
			cudaFree(data->input[i]);
			cudaFree(data->target[i]);
		}
	}
	if(data->input_device != NULL)
	{
		cudaFree(data->input_device);
		cudaFree(data->target_device);
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


#define cuda_update_weights_kernel(name, type)																									\
__global__ void cuda_update_weights_##name(float *weights, void* update, 																		\
	float weight_decay, int is_pivot, size_t size, float TC_scale_factor)																		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	type *c_update = ((type*)update);																											\
																																				\
	if(i < size-is_pivot)																														\
	{	/*Here the weight_decay variable include the learning rate scaling*/																	\
		c_update[i] += weight_decay*weights[i]*TC_scale_factor;																					\
		weights[i] -= (float)(((float)c_update[i]) / TC_scale_factor);																			\
	}																																			\
}


void cuda_update_weights(network* net, void *weights, void* update, float weight_decay, int is_pivot, size_t size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	net->cu_inst.cu_auxil_fcts.cu_update_weights_kernel<<< cu_blocks, cu_threads >>>
		((float*)weights, update, weight_decay, is_pivot, size, net->TC_scale_factor);
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

void cuda_perf_eval_init(void)
{
	cudaEventCreate(&cu_perf_start);
	cudaEventCreate(&cu_perf_stop);
}
void cuda_batch_eval_init(void)
{
	cudaEventCreate(&cu_batch_start);
	cudaEventCreate(&cu_batch_stop);
}
void cuda_epoch_eval_init(void)
{
	cudaEventCreate(&cu_epoch_start);
	cudaEventCreate(&cu_epoch_stop);
}

void cuda_perf_eval_in(void)
{
	cudaEventRecord(cu_perf_start);
}
void cuda_batch_eval_in(void)
{
	cudaEventRecord(cu_batch_start);
}
void cuda_epoch_eval_in(void)
{
	cudaEventRecord(cu_epoch_start);
}

float cuda_perf_eval_out(void)
{
	float time = 0.0f; //milliseconds
	cudaEventRecord(cu_perf_stop);
	cudaEventSynchronize(cu_perf_stop);
	cudaEventElapsedTime(&time, cu_perf_start, cu_perf_stop);
	
	return time*1000; //microseconds
}
float cuda_batch_eval_out(void)
{
	float time = 0.0f; //milliseconds
	cudaEventRecord(cu_batch_stop);
	cudaEventSynchronize(cu_batch_stop);
	cudaEventElapsedTime(&time, cu_batch_start, cu_batch_stop);
	
	return time*1000; //microseconds
}
float cuda_epoch_eval_out(void)
{
	float time = 0.0f; //milliseconds
	cudaEventRecord(cu_epoch_stop);
	cudaEventSynchronize(cu_epoch_stop);
	cudaEventElapsedTime(&time, cu_epoch_start, cu_epoch_stop);
	
	return time*1000; //microseconds
}

#define shfl_kern_fct(name, type)																												\
__global__ void shfl_kern_##name																												\
	(void **i_in, void **i_targ, void **i_train_dupl, void **i_targ_dupl,																		\
	int *index, size_t in_size, int b_size, int d_in, int d_out)																				\
{																																				\
	int j, batch, batch2, pos, pos2;																											\
																																				\
	type **in   = (type**) i_in;																												\
	type **targ = (type**) i_targ;																												\
	type **train_dupl = (type**) i_train_dupl;																									\
	type **targ_dupl  = (type**) i_targ_dupl;																									\
																																				\
																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i < in_size)																																\
	{																																			\
		pos = i%b_size;																															\
		batch = i/b_size;																														\
		pos2 = (int)(index[i])%b_size;																											\
		batch2 = (int)(index[i])/b_size;																										\
		for(j = 0; j < d_in; j++)																												\
			train_dupl[batch2][pos2*d_in+j] = in[batch][pos*d_in + j];																			\
		for(j = 0; j < d_out; j++)																												\
			targ_dupl[batch2][pos2*d_out+j] = targ[batch][pos*d_out + j];																		\
	}																																			\
}

#define get_back_shuffle_fct(name, type)																										\
__global__ void get_back_shuffle_##name																											\
	(void **i_in, void **i_targ, void **i_train_dupl, void **i_targ_dupl,																		\
	size_t in_size, int b_size, int d_in, int d_out)																							\
{																																				\
	size_t j;																																	\
	int batch, pos;																																\
																																				\
	type **in   = (type**) i_in;																												\
	type **targ = (type**) i_targ;																												\
	type **train_dupl = (type**) i_train_dupl;																									\
	type **targ_dupl  = (type**) i_targ_dupl;																									\
																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i < in_size)																																\
	{																																			\
		pos = i%b_size;																															\
		batch = i/b_size;																														\
		for(j = 0; j < d_in; j++)																												\
			in[batch][pos*d_in + j] = train_dupl[batch][pos*d_in+j];																			\
		for(j = 0; j < d_out; j++)																												\
			targ[batch][pos*d_out + j] = targ_dupl[batch][pos*d_out+j];																			\
	}																																			\
}


void cuda_shuffle(network *net, Dataset data, Dataset duplicate, int *index_shuffle, int *index_shuffle_device)
{
	int i, j;
	int temp;

	for(i = 0; i < data.size - 1; i++)
	{
		j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
		temp = index_shuffle[i];
		index_shuffle[i] = index_shuffle[j];
		index_shuffle[j] = temp;
	}
	
	cudaMemcpy(index_shuffle_device, index_shuffle, data.size*sizeof(int), cudaMemcpyHostToDevice);
	
	cu_blocks = (data.size + cu_threads - 1) / cu_threads;

	net->cu_inst.cu_auxil_fcts.cu_shfl_kern_fct<<< cu_blocks, cu_threads>>>
		(data.input_device, data.target_device, duplicate.input_device, duplicate.target_device,
		index_shuffle_device, data.size, net->batch_size, net->input_dim+1, net->output_dim);
	net->cu_inst.cu_auxil_fcts.cu_get_back_shuffle_fct<<< cu_blocks, cu_threads>>>
		(data.input_device, data.target_device, duplicate.input_device, duplicate.target_device, 
		data.size, net->batch_size, net->input_dim+1, net->output_dim);
}

#define host_shuffle_typed(name, type)																											\
void cuda_host_shuffle_##name(network *net, Dataset data, Dataset duplicate)																	\
{																																				\
	int i, j;																																	\
	size_t k;																																	\
	type temp;																																	\
	int pos, pos2, batch, batch2;																												\
																																				\
	type *d_in_A, *d_in_B, *d_targ_A, *d_targ_B;																								\
																																				\
	for(i = 0; i < data.nb_batch; i++)																											\
	{																																			\
		cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 																			\
			* (net->input_dim + 1)*sizeof(type), cudaMemcpyDeviceToHost);																		\
		cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 																		\
			* (net->output_dim)*sizeof(type), cudaMemcpyDeviceToHost);																			\
	}																																			\
																																				\
	for(i = 0; i < data.size - 1; i++)																											\
	{																																			\
		j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));																	\
		pos = i%net->batch_size;																												\
		batch = i/net->batch_size;																												\
		pos2 = (int)(j)%net->batch_size;																										\
		batch2 = (int)(j)/net->batch_size;																										\
																																				\
		d_in_A = ((type*) duplicate.input[batch]);																								\
		d_targ_A = ((type*) duplicate.target[batch]);																							\
		d_in_B = ((type*) duplicate.input[batch2]);																								\
		d_targ_B = ((type*) duplicate.target[batch2]);																							\
																																				\
		for(k = 0; k < net->input_dim+1; k++)																									\
		{																																		\
			temp = d_in_A[pos*(net->input_dim + 1) + k];																						\
			d_in_A[pos*(net->input_dim + 1) + k] = d_in_B[pos2*(net->input_dim + 1) + k];														\
			d_in_B[pos2*(net->input_dim + 1) + k] = temp;																						\
		}																																		\
		for(k = 0; k < (size_t)net->output_dim; k++)																							\
		{																																		\
			temp = d_targ_A[pos*net->output_dim + k];																							\
			d_targ_A[pos*net->output_dim + k] = d_targ_B[pos2*net->output_dim + k];																\
			d_targ_B[pos2*net->output_dim + k] = temp;																							\
		}																																		\
	}																																			\
																																				\
	for(i = 0; i < data.nb_batch; i++)																											\
	{																																			\
		cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 																			\
			* (net->input_dim + 1)*sizeof(type), cudaMemcpyHostToDevice);																		\
		cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 																		\
			* (net->output_dim)*sizeof(type), cudaMemcpyHostToDevice);																			\
	}																																			\
}

void cuda_host_shuffle(network *net, Dataset data, Dataset duplicate)
{
	net->cu_inst.cu_auxil_fcts.cu_host_shuffle_fct(net, data, duplicate);
}

#define cuda_host_only_shuffle_type(name, type)																									\
void cuda_host_only_shuffle_##name(network *net, Dataset data)																					\
{																																				\
	int i, j;																																	\
	size_t k;																																	\
	type temp;																																	\
	int pos, pos2, batch, batch2;																												\
	type **c_input, **c_target;																													\
																																				\
	for(i = 0; i < data.size - 1; i++)																											\
	{																																			\
		j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));																	\
		pos = i%net->batch_size;																												\
		batch = i/net->batch_size;																												\
		pos2 = j%net->batch_size;																												\
		batch2 = j/net->batch_size;																												\
																																				\
		c_input = (type**)data.input;																											\
		c_target = (type**)data.target;																											\
																																				\
		for(k = 0; k < net->input_dim+1; k++)																									\
		{																																		\
			temp = ((type*)c_input[batch])[pos*(net->input_dim + 1) + k];																		\
			((type*)c_input[batch])[pos*(net->input_dim + 1) + k] = 																			\
				((type*)c_input[batch2])[pos2*(net->input_dim + 1) + k];																		\
			((type*)c_input[batch2])[pos2*(net->input_dim + 1) + k] = temp;																		\
		}																																		\
																																				\
		for(k = 0; k < (size_t)net->output_dim; k++)																							\
		{																																		\
			temp = ((type*)c_target[batch])[pos*net->output_dim + k];																			\
			((type*)c_target[batch])[pos*net->output_dim + k] = 																				\
				((type*)c_target[batch2])[pos2*net->output_dim + k];																			\
			((type*)c_target[batch2])[pos2*net->output_dim + k] = temp;																			\
		}																																		\
	}																																			\
}

void cuda_host_only_shuffle(network *net, Dataset data)
{
	net->cu_inst.cu_auxil_fcts.cu_host_only_shuffle_fct(net, data);
}


cuda_host_copy_to(FP32, float, ); //last argument empty en purpose
cuda_create_host_table_fct(FP32, float);
cuda_convert_table_fct(FP32, float);
cuda_create_table_fct(FP32, float);
cuda_get_table_host_to(FP32, float);
cuda_get_typed_host_table_fct(FP32, float);
cuda_put_table_fct(FP32, float);
cuda_get_batches_table_fct(FP32, float);
cuda_create_dataset_fct(FP32, float);
cuda_update_weights_kernel(FP32, float);
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
cuda_get_table_to_FP32_fct(FP16, half);
cuda_get_table_host_to(FP16, half);
cuda_get_typed_host_table_fct(FP16, half);
cuda_put_table_fct(FP16, half);
cuda_convert_batched_table_fct(FP16, half);
cuda_get_batches_table_fct(FP16, half);
cuda_convert_batched_host_table_FP32_to(FP16, half);
cuda_create_dataset_fct(FP16, half);
cuda_master_weight_FP32_to_kernel(FP16, half);
cuda_update_weights_kernel(FP16, half);
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
cuda_get_table_to_FP32_fct(BF16, nv_bfloat16);
cuda_get_table_host_to(BF16, nv_bfloat16);
cuda_get_typed_host_table_fct(BF16, nv_bfloat16);
cuda_put_table_fct(BF16, nv_bfloat16);
cuda_convert_batched_table_fct(BF16, nv_bfloat16);
cuda_get_batches_table_fct(BF16, nv_bfloat16);
cuda_convert_batched_host_table_FP32_to(BF16, nv_bfloat16);
cuda_create_dataset_fct(BF16, nv_bfloat16);
cuda_master_weight_FP32_to_kernel(BF16, nv_bfloat16);
cuda_update_weights_kernel(BF16, nv_bfloat16);
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
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_FP32_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_FP32;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_FP32;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_FP32_kernel;
			net->cu_inst.cu_auxil_fcts.cu_update_weights_kernel = cuda_update_weights_FP32;
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
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_FP16_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_FP16;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_FP16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_FP16;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_FP16_kernel;
			net->cu_inst.cu_auxil_fcts.cu_update_weights_kernel = cuda_update_weights_FP16;
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
			net->cu_inst.cu_auxil_fcts.cu_get_table_fct = cuda_get_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_typed_host_table_fct = cuda_get_typed_host_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_table_to_FP32_fct = cuda_get_table_BF16_to_FP32;
			net->cu_inst.cu_auxil_fcts.cu_put_table_fct = cuda_put_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_table_fct = cuda_convert_batched_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_create_dataset_fct = cuda_create_dataset_BF16;
			net->cu_inst.cu_auxil_fcts.cu_get_batched_table_fct = cuda_get_batched_table_BF16;
			net->cu_inst.cu_auxil_fcts.cu_convert_batched_host_table_FP32_to_fct = cuda_convert_batched_host_table_FP32_to_BF16;
			net->cu_inst.cu_auxil_fcts.cu_master_weight_copy_kernel = cuda_master_weight_FP32_to_BF16_kernel;
			net->cu_inst.cu_auxil_fcts.cu_update_weights_kernel = cuda_update_weights_BF16;
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


void init_cuda(network* net)
{
	cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
	if(!is_cuda_init)
	{
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

			printf("\nGPU handle create fail\n");
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
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;
			
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			case FP16C_FP32A:
				cublasSetMathMode(cu_handle, CUBLAS_TENSOR_OP_MATH);
				cuda_data_type = CUDA_R_16F;
				cuda_compute_type = CUDA_R_32F;
				cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;
				
			case FP16C_FP16A:
				cublasSetMathMode(cu_handle, CUBLAS_TENSOR_OP_MATH);
				cuda_data_type = CUDA_R_16F;
				cuda_compute_type = CUDA_R_16F;
				cu_alpha = &cu_h_alpha; cu_beta = &cu_h_beta;
				cu_learning_rate = &cu_h_learning_rate; cu_momentum = &cu_h_momentum;
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
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;

			#if defined(GEN_AMPERE) 
			case TF32C_FP32A:
				cublasSetMathMode(cu_handle, CUBLAS_TF32_TENSOR_OP_MATH);
				cuda_data_type = CUDA_R_32F;
				cuda_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
				cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;
			#endif
			
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			case FP16C_FP32A:
				cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
				cuda_data_type = CUDA_R_16F;
				cuda_compute_type = CUBLAS_COMPUTE_32F;
				cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;
				
			case FP16C_FP16A:
				cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
				cuda_data_type = CUDA_R_16F;
				cuda_compute_type = CUBLAS_COMPUTE_16F;
				cu_alpha = &cu_h_alpha; cu_beta = &cu_h_beta;
				cu_learning_rate = &cu_h_learning_rate; cu_momentum = &cu_h_momentum;
				break;
			#endif
			
			#if defined(GEN_AMPERE)
			case BF16C_FP32A:
				cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
				cuda_data_type = CUDA_R_16BF;
				cuda_compute_type = CUBLAS_COMPUTE_32F;
				cu_alpha = &cu_f_alpha; cu_beta = &cu_f_beta;
				cu_learning_rate = &cu_f_learning_rate; cu_momentum = &cu_f_momentum;
				break;
			#endif
		}
		#endif
		
		//set typed function according to USE_CUDA_TC
		curandCreateGenerator(&cu_gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(cu_gen, time(NULL));
	}
	
	is_cuda_init = 1;
	

	init_auxil_cuda(net);
	init_typed_cuda_activ(net);
	cuda_dense_init(net);
	cuda_conv_init(net);
	cuda_pool_init(net);
	cuda_norm_init(net);
	cuda_lrn_init(net);
	
	//place holder for device selection
}


