
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
int cu_threads = CUDA_THREADS_PER_BLOCKS;
float cu_alpha = 1.0f, cu_beta= 0.0f;
float TC_scale_factor = 1.0f;
cublasHandle_t cu_handle;
cudaDataType cuda_data_type = CUDA_R_32F;
cublasComputeType_t cuda_compute_type = CUBLAS_COMPUTE_32F;

cudaEvent_t cu_event_start, cu_event_stop;

//local prototypes
__global__ void add_confmat(network* net, void *out, void *targ, float *mat, int len, int o_dim);
//__device__ int cuda_argmax(void* tab, int pos, int len, int size, int format);


void init_cuda(network* net)
{
	cublasStatus_t stat = cublasCreate(&cu_handle);
	
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_PEDANTIC_MATH);
			cuda_data_type = CUDA_R_32F;
			cuda_compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
			TC_scale_factor = 1.0f;
			break;
	
		case TF32C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_TF32_TENSOR_OP_MATH);
			cuda_data_type = CUDA_R_32F;
			cuda_compute_type = CUBLAS_COMPUTE_32F;
			TC_scale_factor = 1.0f;
			break;
			
		case FP16C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUBLAS_COMPUTE_32F;
			TC_scale_factor = 8.0f;
			break;
			
		case FP16C_FP16A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16F;
			cuda_compute_type = CUBLAS_COMPUTE_16F;
			TC_scale_factor = 8.0f;
			break;
			
		case BF16C_FP32A:
			cublasSetMathMode(cu_handle, CUBLAS_DEFAULT_MATH);
			cuda_data_type = CUDA_R_16BF;
			cuda_compute_type = CUBLAS_COMPUTE_32F;
			TC_scale_factor = 1.0f;
			break;
	}
	
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
	
	//place holder for device selection
}

void cuda_set_TC_scale_factor(network* net, float val)
{
	if(net->use_cuda_TC == FP16C_FP32A || net->use_cuda_TC == FP16C_FP16A)
	{
		TC_scale_factor = val;
	}
	else
	{
		if(val != 1.0f)
			printf("\nWARNING: Tried to set TC_scale_factor but the compute mode is incompatible.\nScale kept to 1.\n");
		TC_scale_factor = 1.0f;
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

void copy_to_half(void* in_tab, void* out_tab, int out_offset, int size)
{
	for(int i = 0; i < size; i++)
		*((half*)out_tab + out_offset + i) = __float2half_rz(*((float*)in_tab + i));
}

void copy_to_bfloat16(void* in_tab, void* out_tab, int out_offset, int size)
{
	for(int i = 0; i < size; i++)
		*((nv_bfloat16*)out_tab + out_offset + i) = __float2bfloat16_rz(*((float*)in_tab + i));
}


void cuda_create_host_table_FP16(network* net, void **tab, int size)
{
	*tab = (half*) malloc(size*sizeof(half));
}

void cuda_create_host_table_BF16(network* net, void **tab, int size)
{
	*tab = (nv_bfloat16*) malloc(size*sizeof(nv_bfloat16));
}

long long int cuda_convert_table(network* net, void **tab, long long int size)
{
	void* temp_tab;
	long long int l_vram = 0;
	
	temp_tab = *tab;
	half *temp_half;
	nv_bfloat16 *temp_bf16;

	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			cudaMalloc(tab, size*sizeof(float));
			l_vram += size*sizeof(float);
			cudaMemcpy(*tab, temp_tab, size*sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			temp_half = (half*) malloc(size*sizeof(half));
			copy_to_half((float*)temp_tab, temp_half, 0, size);
			free(temp_tab);
			cudaMalloc(tab, size*sizeof(half));
			l_vram += size*sizeof(half);
			cudaMemcpy(*tab, temp_half, size*sizeof(half),cudaMemcpyHostToDevice);
			free(temp_half);
			break;
			
		case BF16C_FP32A:
			temp_bf16 = (nv_bfloat16*) malloc(size*sizeof(nv_bfloat16));
			copy_to_bfloat16((float*)temp_tab, temp_bf16, 0, size);
			free(temp_tab);
			cudaMalloc(tab, size*sizeof(nv_bfloat16));
			l_vram += size*sizeof(nv_bfloat16);
			cudaMemcpy(*tab, temp_bf16, size*sizeof(nv_bfloat16),cudaMemcpyHostToDevice);
			free(temp_bf16);
			break;
	}
	return l_vram;
}

long long int cuda_convert_table_int(network* net, int **tab, int size)
{
	int* temp_tab;
	long long int l_vram = 0;
	
	temp_tab = *tab;

	cudaMalloc(tab, size*sizeof(int));
	cudaMemcpy(*tab, temp_tab, size*sizeof(int),cudaMemcpyHostToDevice);
	l_vram += size*sizeof(int);
	free(temp_tab);
	
	return l_vram;
}

void cuda_create_table_FP32(network* net, float **tab, float size)
{
	cudaMalloc(tab, size*sizeof(float));
	cudaMemset(*tab, 0.0, size*sizeof(float));
}

void cuda_create_table(network* net, void **tab, int size)
{
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			cudaMalloc(tab, size*sizeof(float));
			cudaMemset(*tab, 0.0, size*sizeof(float));
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			cudaMalloc(tab, size*sizeof(half));
			cudaMemset(*tab, 0.0, size*sizeof(half));
			break;
			
		case BF16C_FP32A:
			cudaMalloc(tab, size*sizeof(nv_bfloat16));
			cudaMemset(*tab, 0.0, size*sizeof(nv_bfloat16));
			break;
	}
}


void cuda_get_table_FP16_to_FP32(void *cuda_table, void *table, int size, void* buffer)
{
	half *temp_half;
	int mem = 0;
	
	if(buffer != NULL)
		temp_half = (half*) buffer;
	else
	{
		temp_half = (half*) malloc(size*sizeof(half));
		mem = 1;
	}	
	cudaMemcpy(temp_half, cuda_table, size*sizeof(half), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < size; i++)
		((float*)table)[i] = (float) temp_half[i];
	
	if(mem == 1)
		free(temp_half);

}

void cuda_get_table_BF16_to_FP32(void *cuda_table, void *table, int size, void* buffer)
{
	nv_bfloat16 *temp_half;
	int mem = 0;
	
	if(buffer != NULL)
		temp_half = (nv_bfloat16*) buffer;
	else
	{
		temp_half = (nv_bfloat16*) malloc(size*sizeof(nv_bfloat16));
		mem = 1;
	}	
	cudaMemcpy(temp_half, cuda_table, size*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < size; i++)
		((float*)table)[i] = (float) temp_half[i];
	
	if(mem == 1)
		free(temp_half);

}

void cuda_get_table_FP32(network* net, float *cuda_table, float *table, int size)
{
	cudaMemcpy(table, cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_get_table(network* net, void *cuda_table, void *table, int size)
{
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			cudaMemcpy(table, cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
			break;
		case FP16C_FP32A:
		case FP16C_FP16A:
			cudaMemcpy(table, cuda_table, size*sizeof(half), cudaMemcpyDeviceToHost);
			break;
		case BF16C_FP32A:
			cudaMemcpy(table, cuda_table, size*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
			break;
	}
}

void cuda_put_table_FP32(network* net, float *cuda_table, float *table, int size)
{
	cudaMemcpy(cuda_table, table, size*sizeof(float), cudaMemcpyHostToDevice);
}

void cuda_put_table(network* net, void *cuda_table, void *table, int size)
{
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			cudaMemcpy(cuda_table, table, size*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case FP16C_FP32A:
		case FP16C_FP16A:
			cudaMemcpy(cuda_table, table, size*sizeof(half), cudaMemcpyHostToDevice);
			break;
		case BF16C_FP32A:
			cudaMemcpy(cuda_table, table, size*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
			break;
	}
}

void cuda_convert_batched_table(network* net, void **tab, int batch_size, int nb_batch, int size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;
	half *temp_half;
	nv_bfloat16 *temp_bf16;

	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			for(i = 0; i < nb_batch; i++)
			{
				temp_tab = tab[i];
				cudaMalloc(&(tab[i]), batch_size*size*sizeof(float));
				cudaMemcpy(tab[i], temp_tab, batch_size*size*sizeof(float),cudaMemcpyHostToDevice);
				free(temp_tab);
			}
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			temp_half = (half*) malloc(batch_size*size*sizeof(half));
			for(i = 0; i < nb_batch; i++)
			{
				temp_tab = tab[i];
				copy_to_half((float*)temp_tab, temp_half, 0, batch_size*size);
				free(temp_tab);
				cudaMalloc(&(tab[i]), batch_size*size*sizeof(half));
				cudaMemcpy(tab[i], temp_half, batch_size*size*sizeof(half),cudaMemcpyHostToDevice);
			}
			free(temp_half);
			break;
			
		case BF16C_FP32A:
			temp_bf16 = (nv_bfloat16*) malloc(batch_size*size*sizeof(nv_bfloat16));
			for(i = 0; i < nb_batch; i++)
			{
				temp_tab = tab[i];
				copy_to_bfloat16((float*)temp_tab, temp_bf16, 0, batch_size*size);
				free(temp_tab);
				cudaMalloc(&(tab[i]), batch_size*size*sizeof(nv_bfloat16));
				cudaMemcpy(tab[i], temp_bf16, batch_size*size*sizeof(nv_bfloat16),cudaMemcpyHostToDevice);
			}
			free(temp_bf16);
			break;
	}
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


void cuda_convert_batched_host_table_FP32_to_FP16(network* net, void **tab, int batch_size, int nb_batch, int size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;

	
	for(i = 0; i < nb_batch; i++)
	{
		temp_tab = tab[i];
		tab[i] = (half*) malloc(batch_size*size*sizeof(half));
		copy_to_half((float*)temp_tab, ((half*)tab[i]), 0, batch_size*size);
		free(temp_tab);
	}
}

void cuda_convert_batched_host_table_FP32_to_BF16(network* net, void **tab, int batch_size, int nb_batch, int size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;

	
	for(i = 0; i < nb_batch; i++)
	{
		temp_tab = tab[i];
		tab[i] = (nv_bfloat16*) malloc(batch_size*size*sizeof(nv_bfloat16));
		copy_to_half((float*)temp_tab, ((half*)tab[i]), 0, batch_size*size);
		free(temp_tab);
	}
}


void cuda_convert_host_dataset_FP32(network *net, Dataset *data)
{
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			//nothing to do
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			cuda_convert_batched_host_table_FP32_to_FP16(net, data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
			cuda_convert_batched_host_table_FP32_to_FP16(net, data->target, net->batch_size, data->nb_batch, net->output_dim);
			break;
		
		case BF16C_FP32A:
			cuda_convert_batched_host_table_FP32_to_BF16(net, data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
			cuda_convert_batched_host_table_FP32_to_BF16(net, data->target, net->batch_size, data->nb_batch, net->output_dim);
			break;
	}
	data->localization = HOST;
}


Dataset create_dataset_FP16(network *net, int nb_elem)
{
	int i,j;
	Dataset data;
	
	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (void**) malloc(data.nb_batch*sizeof(half*));
	data.target = (void**) malloc(data.nb_batch*sizeof(half*));
	data.localization = HOST;
	data.cont_copy = copy_to_half;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		((half**)data.input)[i] = (half*) calloc(net->batch_size * (net->input_dim + 1), sizeof(half));
		((half**)data.target)[i] = (half*) calloc(net->batch_size * net->output_dim, sizeof(half));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			((half**)data.input)[i][j*(net->input_dim+1) + net->input_dim] = net->input_bias;
		}
	}
	
	return data;
}


Dataset create_dataset_BF16(network *net, int nb_elem)
{
	int i,j;
	Dataset data;
	
	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (void**) malloc(data.nb_batch*sizeof(nv_bfloat16*));
	data.target = (void**) malloc(data.nb_batch*sizeof(nv_bfloat16*));
	data.localization = HOST;
	data.cont_copy = copy_to_bfloat16;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		((nv_bfloat16**)data.input)[i] = (nv_bfloat16*) calloc(net->batch_size * (net->input_dim + 1), sizeof(nv_bfloat16));
		((nv_bfloat16**)data.target)[i] = (nv_bfloat16*) calloc(net->batch_size * net->output_dim, sizeof(nv_bfloat16));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			((nv_bfloat16**)data.input)[i][j*(net->input_dim+1) + net->input_dim] = net->input_bias;
		}
	}
	
	return data;
}

void cuda_free_dataset(Dataset *data)
{
	int i;
	
	for(i = 0; i < data->nb_batch; i++)
	{
		cudaFree(data->input[i]);
		cudaFree(data->target[i]);
	}
	cudaFree(data->input_device);
	cudaFree(data->target_device);
}

__global__ void cuda_master_weight_FP32_to_FP16_kernel(float *master, half *copy, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		copy[i] = (half)master[i];
	}
}

__global__ void cuda_master_weight_FP32_to_BF16_kernel(float *master, nv_bfloat16 *copy, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		copy[i] = (nv_bfloat16)master[i];
	}
}

void cuda_master_weight_FP32_to_FP16(float *master, half *copy, int size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	cuda_master_weight_FP32_to_FP16_kernel<<< cu_blocks, cu_threads >>>(master, copy, size);
}

void cuda_master_weight_FP32_to_BF16(float *master, nv_bfloat16 *copy, int size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	cuda_master_weight_FP32_to_BF16_kernel<<< cu_blocks, cu_threads >>>(master, copy, size);
}


__global__ void cuda_update_weights_FP32(float *weights, float* update, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= update[i] / (TC_scale_factor);
	}
}

__global__ void cuda_update_weights_FP16_mixed(float *weights, half* update, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= (float)update[i] / (TC_scale_factor);
	}
}

__global__ void cuda_update_weights_BF16_mixed(float *weights, nv_bfloat16* update, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= (float)update[i] / (TC_scale_factor);
		//weights[i] = (weights[i] - (float)update[i] / TC_scale_factor)*0.9999f;
	}
}

void cuda_update_weights(network* net, void *weights, void* update, int size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			cuda_update_weights_FP32<<< cu_blocks, cu_threads >>>((float*)weights, (float*)update, size, TC_scale_factor);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			cuda_update_weights_FP16_mixed<<< cu_blocks, cu_threads >>>((float*)weights, (half*)update, size, TC_scale_factor);
			break;
			
		case BF16C_FP32A:
			cuda_update_weights_BF16_mixed<<< cu_blocks, cu_threads >>>((float*)weights, (nv_bfloat16*)update, size, TC_scale_factor);
			break;
	}
}


void cuda_print_table_FP32(network* net, float* tab, int size, int return_every)
{
	int i;
	void *temp;
	
	printf("\n");

	temp = (void*) malloc(size*sizeof(float));
	cudaMemcpy(temp, tab, size*sizeof(float), cudaMemcpyDeviceToHost);
	for(i = 0; i < size; i ++)
	{
		if(i%return_every == 0)
			printf("\n");
		printf("%g \t ", ((float*)temp)[i]);
	}
	
	free(temp);
}


void cuda_print_table_4d(network* net, void* tab, int w_size, int h_size, int d_size, int last_dim, int biased)
{
	int i, j, k, l;
	void *temp;
	
	int flat_3d_size = w_size*h_size*d_size + biased;
	
	printf("\n");
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			for(i = 0; i < last_dim; i++)
			{
				printf("Cube %d\n", i);
				temp = (void*) malloc(flat_3d_size*sizeof(float));
				cudaMemcpy(temp, (float*)tab+i*flat_3d_size, flat_3d_size*sizeof(float), cudaMemcpyDeviceToHost);
				for(j = 0; j < d_size; j ++)
				{
					printf("Depth %d\n", j);
					for(k = 0; k < h_size; k++)
					{
						for(l = 0; l < w_size; l++)
						{
							printf("%5.4f ", ((float*)temp)[j*w_size*h_size + k*w_size + l]);
						}
						printf("\n");
					}
				}
				free(temp);
			}
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			for(i = 0; i < last_dim; i++)
			{
				printf("Cube %d\n", i);
				temp = (void*) malloc(flat_3d_size*sizeof(half));
				cudaMemcpy(temp, (half*)tab+i*flat_3d_size, flat_3d_size*sizeof(half), cudaMemcpyDeviceToHost);
				for(j = 0; j < d_size; j ++)
				{
					printf("Depth %d\n", j);
					for(k = 0; k < h_size; k++)
					{
						for(l = 0; l < w_size; l++)
						{
							printf("%5.4f ", (float)(((half*)temp)[j*w_size*h_size + k*w_size + l]));
						}
						printf("\n");
					}
				}
				free(temp);
			}
			break;
		case BF16C_FP32A:
			printf("BF16 print4D unsuported ATM ...\n");
			break;
	}
	printf("\n");
}


void cuda_print_table(network* net, void* tab, int size, int return_every)
{
	int i;
	void *temp;
	
	printf("\n");
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			temp = (void*) malloc(size*sizeof(float));
			cudaMemcpy(temp, tab, size*sizeof(float), cudaMemcpyDeviceToHost);
			for(i = 0; i < size; i ++)
			{
				if(i%return_every == 0)
					printf("\n");
				printf("%5.4f ", ((float*)temp)[i]);
			}
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			temp = (void*) malloc(size*sizeof(half));
			cudaMemcpy(temp, tab, size*sizeof(half), cudaMemcpyDeviceToHost);
			for(i = 0; i < size; i ++)
			{
				if(i%return_every == 0)
					printf("\n");
				printf("%5.4f ", (float)(((half*)temp)[i]));
			}
			break;
			
		case BF16C_FP32A:
			temp = (void*) malloc(size*sizeof(nv_bfloat16));
			cudaMemcpy(temp, tab, size*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
			for(i = 0; i < size; i ++)
			{
				if(i%return_every == 0)
					printf("\n");
				printf("%5.4f ", (float)(((nv_bfloat16*)temp)[i]));
			}
			break;
	}
	printf("\n");
	
	free(temp);
}

void cuda_print_table_int(network* net, int* tab, int size, int return_every)
{
	int i;
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

void cuda_print_table_host_FP16(network* net, void* tab, int size, int return_every)
{
	int i;
	for(i = 0; i < size; i ++)
	{
		if(i%return_every == 0)
			printf("\n");
		printf("%g \t ", (float)(((half*)tab)[i]));
	}
}

void cuda_print_table_host_BF16(network* net, void* tab, int size, int return_every)
{
	int i;
	for(i = 0; i < size; i ++)
	{
		if(i%return_every == 0)
			printf("\n");
		printf("%g \t ", (float)(((nv_bfloat16*)tab)[i]));
	}
}

/*
void cuda_print_table_transpose(void* tab, int line_size, int column_size)
{
	int i, j;
	int size;
	void *temp;
	
	size = line_size*column_size;
	temp = (void*) malloc(size*sizeof(void));
	cudaMemcpy(temp, tab, size*sizeof(void), cudaMemcpyDeviceToHost);
	
	printf("\n");
	for(i = 0; i < column_size; i ++)
	{
		for(j = 0; j < line_size; j++)
		{
			printf("%g \t ", temp[j*column_size + i]);
		}
		printf("\n");
	}
	printf("\n");
	
	free(temp);
}
*/

__device__ int cuda_argmax_FP32(float* tab, int dim_out)
{
	int i;
	float vmax;
	int imax;

	vmax = tab[0];
	imax = 0;
	for(i = 1; i < dim_out; i++)
	{
		if(tab[i] > vmax)
		{
			vmax = tab[i];
			imax = i;
		}
	}
	
	return imax;
}

__device__ int cuda_argmax_FP16(half* tab, int dim_out)
{
	int i;
	half vmax;
	int imax;

	vmax = tab[0];
	imax = 0;
	for(i = 1; i < dim_out; i++)
	{
		if(tab[i] > vmax)
		{
			vmax = tab[i];
			imax = i;
		}
	}
	
	return imax;
}

__device__ int cuda_argmax_BF16(nv_bfloat16* tab, int dim_out)
{
	int i;
	nv_bfloat16 vmax;
	int imax;

	vmax = tab[0];
	imax = 0;
	for(i = 1; i < dim_out; i++)
	{
		if(tab[i] > vmax)
		{
			vmax = tab[i];
			imax = i;
		}
	}
	
	return imax;
}

__global__ void add_confmat_FP32(float *out, float *targ, float *mat, int len, int o_dim)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arg1, arg2;
	if( i < len)
	{
		targ += i*o_dim;
		out += i*(o_dim+1);
		
		arg1 = cuda_argmax_FP32(targ, o_dim);
		arg2 = cuda_argmax_FP32(out, o_dim);
		
		atomicAdd(&(mat[arg2+o_dim*arg1]), 1);
	}
}

__global__ void add_confmat_FP16(half *out, half *targ, float *mat, int len, int o_dim)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arg1, arg2;
	if( i < len)
	{
		targ += i*o_dim;
		out += i*(o_dim+1);
		
		arg1 = cuda_argmax_FP16(targ, o_dim);
		arg2 = cuda_argmax_FP16(out, o_dim);
		
		atomicAdd(&(mat[arg2+o_dim*arg1]), 1);
	}
}

__global__ void add_confmat_BF16(nv_bfloat16 *out, nv_bfloat16 *targ, float *mat, int len, int o_dim)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arg1, arg2;
	if( i < len)
	{
		targ += i*o_dim;
		out += i*(o_dim+1);
		
		arg1 = cuda_argmax_BF16(targ, o_dim);
		arg2 = cuda_argmax_BF16(out, o_dim);
		
		atomicAdd(&(mat[arg2+o_dim*arg1]), 1);
	}
}

void cuda_confmat(network *net, float* mat)
{
	cu_blocks = (net->length + cu_threads - 1) / cu_threads;
	
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			add_confmat_FP32<<< cu_blocks, cu_threads>>>((float*)net->net_layers[net->nb_layers-1]->output, (float*)net->target, mat, net->length, net->output_dim);
			break;
		case FP16C_FP32A:
		case FP16C_FP16A:
			add_confmat_FP16<<< cu_blocks, cu_threads>>>((half*)net->net_layers[net->nb_layers-1]->output, (half*)net->target, mat, net->length, net->output_dim);
			break;
		case BF16C_FP32A:
			add_confmat_BF16<<< cu_blocks, cu_threads>>>((nv_bfloat16*)net->net_layers[net->nb_layers-1]->output, (nv_bfloat16*)net->target, mat, net->length, net->output_dim);
			break;
	}
}

void cuda_perf_eval_init(void)
{
	cudaEventCreate(&cu_event_start);
	cudaEventCreate(&cu_event_stop);
}

void cuda_perf_eval_in(void)
{
	cudaEventRecord(cu_event_start);
}

float cuda_perf_eval_out()
{
	float time = 0.0f; //milliseconds
	cudaEventRecord(cu_event_stop);
	cudaEventSynchronize(cu_event_stop);
	cudaEventElapsedTime(&time, cu_event_start, cu_event_stop);
	
	return time;
}


__global__ void shfl_kern_FP32(float **in, float **targ, float** train_dupl, float** targ_dupl,
									int* index, int in_size, int b_size, int d_in, int d_out)
{
	int j;
	int batch, batch2, pos, pos2;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		pos2 = (int)(index[i])%b_size;
		batch2 = (int)(index[i])/b_size;
		for(j = 0; j < d_in; j++)
			train_dupl[batch2][pos2*d_in+j] = in[batch][pos*d_in + j];
		for(j = 0; j < d_out; j++)
			targ_dupl[batch2][pos2*d_out+j] = targ[batch][pos*d_out + j];
	}
}

__global__ void shfl_kern_FP16(half **in, half **targ, half** train_dupl, half** targ_dupl,
									int* index, int in_size, int b_size, int d_in, int d_out)
{
	int j;
	int batch, batch2, pos, pos2;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		pos2 = (int)(index[i])%b_size;
		batch2 = (int)(index[i])/b_size;
		for(j = 0; j < d_in; j++)
			train_dupl[batch2][pos2*d_in+j] = in[batch][pos*d_in + j];
		for(j = 0; j < d_out; j++)
			targ_dupl[batch2][pos2*d_out+j] = targ[batch][pos*d_out + j];
	}
}


__global__ void shfl_kern_BF16(nv_bfloat16 **in, nv_bfloat16 **targ, nv_bfloat16** train_dupl, nv_bfloat16** targ_dupl,
									int* index, int in_size, int b_size, int d_in, int d_out)
{
	int j;
	int batch, batch2, pos, pos2;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		pos2 = (int)(index[i])%b_size;
		batch2 = (int)(index[i])/b_size;
		for(j = 0; j < d_in; j++)
			train_dupl[batch2][pos2*d_in+j] = in[batch][pos*d_in + j];
		for(j = 0; j < d_out; j++)
			targ_dupl[batch2][pos2*d_out+j] = targ[batch][pos*d_out + j];
	}
}



__global__ void get_back_shuffle_FP32(float **in, float **targ, float** train_dupl, float** targ_dupl,
									int in_size, int b_size, int d_in, int d_out)
{	
	int j;
	int batch, pos;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		for(j = 0; j < d_in; j++)
			in[batch][pos*d_in + j] = train_dupl[batch][pos*d_in+j];
		for(j = 0; j < d_out; j++)
			targ[batch][pos*d_out + j] = targ_dupl[batch][pos*d_out+j];
	}
}

__global__ void get_back_shuffle_FP16(half **in, half **targ, half **train_dupl, half **targ_dupl,
									int in_size, int b_size, int d_in, int d_out)
{	
	int j;
	int batch, pos;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		for(j = 0; j < d_in; j++)
			in[batch][pos*d_in + j] = train_dupl[batch][pos*d_in+j];
		for(j = 0; j < d_out; j++)
			targ[batch][pos*d_out + j] = targ_dupl[batch][pos*d_out+j];
	}
}

__global__ void get_back_shuffle_BF16(nv_bfloat16 **in, nv_bfloat16 **targ, nv_bfloat16 **train_dupl, nv_bfloat16 **targ_dupl,
									int in_size, int b_size, int d_in, int d_out)
{	
	int j;
	int batch, pos;
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < in_size)
	{
		pos = i%b_size;
		batch = i/b_size;
		for(j = 0; j < d_in; j++)
			in[batch][pos*d_in + j] = train_dupl[batch][pos*d_in+j];
		for(j = 0; j < d_out; j++)
			targ[batch][pos*d_out + j] = targ_dupl[batch][pos*d_out+j];
	}
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

	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			shfl_kern_FP32<<< cu_blocks, cu_threads>>>((float**)data.input_device,
				(float**)data.target_device, (float**)duplicate.input_device,
				(float**)duplicate.target_device, index_shuffle_device, 
				data.size, net->batch_size, net->input_dim+1, net->output_dim);
			get_back_shuffle_FP32<<< cu_blocks, cu_threads>>>((float**)data.input_device,
				(float**)data.target_device, (float**)duplicate.input_device,
				(float**)duplicate.target_device, data.size, net->batch_size, 
				net->input_dim+1, net->output_dim);
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			shfl_kern_FP16<<< cu_blocks, cu_threads>>>((half**)data.input_device, 
				(half**)data.target_device, (half**)duplicate.input_device, 
				(half**)duplicate.target_device, index_shuffle_device, 
				data.size, net->batch_size, net->input_dim+1, net->output_dim);
			get_back_shuffle_FP16<<< cu_blocks, cu_threads>>>((half**)data.input_device,
				(half**)data.target_device, (half**)duplicate.input_device, 
				(half**)duplicate.target_device, data.size, net->batch_size, 
				net->input_dim+1, net->output_dim);
			break;
		
		case BF16C_FP32A:
			shfl_kern_BF16<<< cu_blocks, cu_threads>>>((nv_bfloat16**)data.input_device, 
				(nv_bfloat16**)data.target_device, (nv_bfloat16**)duplicate.input_device, 
				(nv_bfloat16**)duplicate.target_device, index_shuffle_device, 
				data.size, net->batch_size, net->input_dim+1, net->output_dim);
			get_back_shuffle_BF16<<< cu_blocks, cu_threads>>>((nv_bfloat16**)data.input_device,
				(nv_bfloat16**)data.target_device, (nv_bfloat16**)duplicate.input_device, 
				(nv_bfloat16**)duplicate.target_device, data.size, net->batch_size, 
				net->input_dim+1, net->output_dim);
			break;
	}
}

void host_shuffle(network *net, Dataset data, Dataset duplicate)
{
	int i, j, k;
	float temp;
	half temp_half;
	nv_bfloat16 temp_bf16;
	int pos, pos2, batch, batch2;

	float *f_d_in_A, *f_d_in_B, *f_d_targ_A, *f_d_targ_B;
	half *h_d_in_A, *h_d_in_B, *h_d_targ_A, *h_d_targ_B;
	nv_bfloat16 *bf_d_in_A, *bf_d_in_B, *bf_d_targ_A, *bf_d_targ_B;

	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 
					* (net->output_dim)*sizeof(float), cudaMemcpyDeviceToHost);
			}
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(half), cudaMemcpyDeviceToHost);
				cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 
					* (net->output_dim)*sizeof(half), cudaMemcpyDeviceToHost);
			}
			break;
		
		case BF16C_FP32A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
				cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 
					* (net->output_dim)*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
			}
			break;
	}
	

	for(i = 0; i < data.size - 1; i++)
	{
		j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
		pos = i%net->batch_size;
		batch = i/net->batch_size;
		pos2 = (int)(j)%net->batch_size;
		batch2 = (int)(j)/net->batch_size;
		
		switch(net->use_cuda_TC)
		{
			default:
			case FP32C_FP32A:
			case TF32C_FP32A:
				f_d_in_A = ((float*) duplicate.input[batch]);
				f_d_targ_A = ((float*) duplicate.target[batch]);
				f_d_in_B = ((float*) duplicate.input[batch2]);
				f_d_targ_B = ((float*) duplicate.target[batch2]);
				
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp = f_d_in_A[pos*(net->input_dim + 1) + k];
					f_d_in_A[pos*(net->input_dim + 1) + k] = f_d_in_B[pos2*(net->input_dim + 1) + k];
					f_d_in_B[pos2*(net->input_dim + 1) + k] = temp;
				}
				for(k = 0; k < net->output_dim; k++)
				{
					temp = f_d_targ_A[pos*net->output_dim + k];
					f_d_targ_A[pos*net->output_dim + k] = f_d_targ_B[pos2*net->output_dim + k];
					f_d_targ_B[pos2*net->output_dim + k] = temp;
				}
				break;
				
			case FP16C_FP32A:
			case FP16C_FP16A:
				h_d_in_A = ((half*) duplicate.input[batch]);
				h_d_targ_A = ((half*) duplicate.target[batch]);
				h_d_in_B = ((half*) duplicate.input[batch2]);
				h_d_targ_B = ((half*) duplicate.target[batch2]);
		
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp_half = h_d_in_A[pos*(net->input_dim + 1) + k];
					h_d_in_A[pos*(net->input_dim + 1) + k] = h_d_in_B[pos2*(net->input_dim + 1) + k];
					h_d_in_B[pos2*(net->input_dim + 1) + k] = temp_half;
				}
				
				for(k = 0; k < net->output_dim; k++)
				{
					temp_half = h_d_targ_A[pos*net->output_dim + k];
					h_d_targ_A[pos*net->output_dim + k] = h_d_targ_B[pos2*net->output_dim + k];
					h_d_targ_B[pos2*net->output_dim + k] = temp_half;
				}
				break;
			
			case BF16C_FP32A:
				bf_d_in_A = ((nv_bfloat16*) duplicate.input[batch]);
				bf_d_targ_A = ((nv_bfloat16*) duplicate.target[batch]);
				bf_d_in_B = ((nv_bfloat16*) duplicate.input[batch2]);
				bf_d_targ_B = ((nv_bfloat16*) duplicate.target[batch2]);
		
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp_bf16 = bf_d_in_A[pos*(net->input_dim + 1) + k];
					bf_d_in_A[pos*(net->input_dim + 1) + k] = bf_d_in_B[pos2*(net->input_dim + 1) + k];
					bf_d_in_B[pos2*(net->input_dim + 1) + k] = temp_bf16;
				}
				
				for(k = 0; k < net->output_dim; k++)
				{
					temp_bf16 = bf_d_targ_A[pos*net->output_dim + k];
					bf_d_targ_A[pos*net->output_dim + k] = bf_d_targ_B[pos2*net->output_dim + k];
					bf_d_targ_B[pos2*net->output_dim + k] = temp_bf16;
				}
				break;
		}
	}
	
	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 
					* (net->output_dim)*sizeof(float), cudaMemcpyHostToDevice);
			}
			break;
		
		case FP16C_FP32A:
		case FP16C_FP16A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(half), cudaMemcpyHostToDevice);
				cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 
					* (net->output_dim)*sizeof(half), cudaMemcpyHostToDevice);
			}
			break;
		
		case BF16C_FP32A:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
				cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 
					* (net->output_dim)*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
			}
			break;
	}
	
}


void cuda_host_only_shuffle(network *net, Dataset data)
{
	int i, j, k;
	float temp;
	int pos, pos2, batch, batch2;

	switch(net->use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			for(i = 0; i < data.size - 1; i++)
			{
				j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
				pos = i%net->batch_size;
				batch = i/net->batch_size;
				pos2 = j%net->batch_size;
				batch2 = j/net->batch_size;
				
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp = ((float**)data.input)[batch][pos*(net->input_dim + 1) + k];
					((float**)data.input)[batch][pos*(net->input_dim + 1) + k] = ((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k];
					((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k] = temp;
				}
				
				for(k = 0; k < net->output_dim; k++)
				{
					temp = ((float**)data.target)[batch][pos*net->output_dim + k];
					
					((float**)data.target)[batch][pos*net->output_dim + k] = ((float**)data.target)[batch2][pos2*net->output_dim + k];
					((float**)data.target)[batch2][pos2*net->output_dim + k] = temp;
				}
			}
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			for(i = 0; i < data.size - 1; i++)
			{
				j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
				pos = i%net->batch_size;
				batch = i/net->batch_size;
				pos2 = j%net->batch_size;
				batch2 = j/net->batch_size;
				
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp = ((half**)data.input)[batch][pos*(net->input_dim + 1) + k];
					((half**)data.input)[batch][pos*(net->input_dim + 1) + k] = ((half**)data.input)[batch2][pos2*(net->input_dim + 1) + k];
					((half**)data.input)[batch2][pos2*(net->input_dim + 1) + k] = temp;
				}
				
				for(k = 0; k < net->output_dim; k++)
				{
					temp = ((half**)data.target)[batch][pos*net->output_dim + k];
					
					((half**)data.target)[batch][pos*net->output_dim + k] = ((half**)data.target)[batch2][pos2*net->output_dim + k];
					((half**)data.target)[batch2][pos2*net->output_dim + k] = temp;
				}
			}
			break;
			
		case BF16C_FP32A:
			for(i = 0; i < data.size - 1; i++)
			{
				j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
				pos = i%net->batch_size;
				batch = i/net->batch_size;
				pos2 = j%net->batch_size;
				batch2 = j/net->batch_size;
				
				for(k = 0; k < net->input_dim+1; k++)
				{
					temp = ((nv_bfloat16**)data.input)[batch][pos*(net->input_dim + 1) + k];
					((nv_bfloat16**)data.input)[batch][pos*(net->input_dim + 1) + k] = ((nv_bfloat16**)data.input)[batch2][pos2*(net->input_dim + 1) + k];
					((nv_bfloat16**)data.input)[batch2][pos2*(net->input_dim + 1) + k] = temp;
				}
				
				for(k = 0; k < net->output_dim; k++)
				{
					temp = ((nv_bfloat16**)data.target)[batch][pos*net->output_dim + k];
					
					((nv_bfloat16**)data.target)[batch][pos*net->output_dim + k] = ((nv_bfloat16**)data.target)[batch2][pos2*net->output_dim + k];
					((nv_bfloat16**)data.target)[batch2][pos2*net->output_dim + k] = temp;
				}
			}
			break;
	}
}


