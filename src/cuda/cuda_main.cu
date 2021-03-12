
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
float cu_alpha = 1.0f, cu_beta = 0.0f;
float TC_scale_factor = 64.0f;
cublasHandle_t cu_handle;
cudaDataType cuda_data_type = CUDA_R_32F;
cudaDataType cuda_compute_type = CUDA_R_32F;

//local prototypes
__global__ void add_confmat(network* net, void *out, void *targ, float *mat, int len, int o_dim);
//__device__ int cuda_argmax(void* tab, int pos, int len, int size, int format);


void init_cuda(network* net)
{
	cublasStatus_t stat = cublasCreate(&cu_handle);
	
	if(net->use_cuda_TC)
	{
		cublasSetMathMode(cu_handle, CUBLAS_TENSOR_OP_MATH);
		cuda_data_type = CUDA_R_16F;
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

void cuda_set_TC_scale_factor(float val)
{
	TC_scale_factor = val;
}

void cuda_sync(void)
{
	cudaDeviceSynchronize();
}

void cuda_free_table(void* tab)
{
	cudaFree(tab);
}

void cuda_copy_to_half(float* in_tab, half* out_tab, int size)
{
	for(int i = 0; i < size; i++)
		out_tab[i] = (half)in_tab[i];
}


void cuda_create_host_table_FP16(network* net, void **tab, int size)
{
	*tab = (half*) malloc(size*sizeof(half));
}

void cuda_convert_table(network* net, void **tab, int size)
{
	void* temp_tab;
	
	temp_tab = *tab;
	half* temp_half;

	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			cudaMalloc(tab, size*sizeof(float));
			cudaMemcpy(*tab, temp_tab, size*sizeof(float),cudaMemcpyHostToDevice);
			free(temp_tab);
			break;
		case 1:
			temp_half = (half*) malloc(size*sizeof(half));
			cuda_copy_to_half((float*)temp_tab, temp_half, size);
			free(temp_tab);
			cudaMalloc(tab, size*sizeof(half));
			cudaMemcpy(*tab, temp_half, size*sizeof(half),cudaMemcpyHostToDevice);
			free(temp_half);
			break;
	}
}

void cuda_convert_table_int(network* net, int **tab, int size)
{
	int* temp_tab;
	
	temp_tab = *tab;

	cudaMalloc(tab, size*sizeof(int));
	cudaMemcpy(*tab, temp_tab, size*sizeof(int),cudaMemcpyHostToDevice);
	free(temp_tab);
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
		case 0:
			cudaMalloc(tab, size*sizeof(float));
			cudaMemset(*tab, 0.0, size*sizeof(float));
			break;
		case 1:
			cudaMalloc(tab, size*sizeof(half));
			cudaMemset(*tab, 0.0, size*sizeof(half));
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

void cuda_get_table_FP32(network* net, float *cuda_table, float *table, int size)
{
	cudaMemcpy(table, cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_get_table(network* net, void *cuda_table, void *table, int size)
{
	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			cudaMemcpy(table, cuda_table, size*sizeof(float), cudaMemcpyDeviceToHost);
			break;
		case 1:
			cudaMemcpy(table, cuda_table, size*sizeof(half), cudaMemcpyDeviceToHost);
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
		case 0:
			cudaMemcpy(cuda_table, table, size*sizeof(float), cudaMemcpyHostToDevice);
			break;
		case 1:
			cudaMemcpy(cuda_table, table, size*sizeof(half), cudaMemcpyHostToDevice);
			break;
	}
}

void cuda_convert_batched_table(network* net, void **tab, int batch_size, int nb_batch, int size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;
	half* temp_half;

	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			for(i = 0; i < nb_batch; i++)
			{
				temp_tab = tab[i];
				cudaMalloc(&(tab[i]), batch_size*size*sizeof(float));
				cudaMemcpy(tab[i], temp_tab, batch_size*size*sizeof(float),cudaMemcpyHostToDevice);
				free(temp_tab);
			}
			break;
		case 1:
			temp_half = (half*) malloc(batch_size*size*sizeof(half));
			for(i = 0; i < nb_batch; i++)
			{
				temp_tab = tab[i];
				cuda_copy_to_half((float*)temp_tab, temp_half, batch_size*size);
				free(temp_tab);
				cudaMalloc(&(tab[i]), batch_size*size*sizeof(half));
				cudaMemcpy(tab[i], temp_half, batch_size*size*sizeof(half),cudaMemcpyHostToDevice);
			}
			free(temp_half);
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


void cuda_convert_batched_host_table_FP32(network* net, void **tab, int batch_size, int nb_batch, int size)
{
	int i;
	void* temp_tab;
	
	temp_tab = *tab;

	
	for(i = 0; i < nb_batch; i++)
	{
		temp_tab = tab[i];
		tab[i] = (half*) malloc(batch_size*size*sizeof(half));
		cuda_copy_to_half((float*)temp_tab, ((half*)tab[i]), batch_size*size);
		free(temp_tab);
	}		
}


void cuda_convert_host_dataset_FP32(network *net, Dataset *data)
{
	cuda_convert_batched_host_table_FP32(net, data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
	cuda_convert_batched_host_table_FP32(net, data->target, net->batch_size, data->nb_batch, net->output_dim);
	data->localization = HOST;
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

void cuda_master_weight_FP32_to_FP16(float *master, half *copy, int size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	cuda_master_weight_FP32_to_FP16_kernel<<< cu_blocks, cu_threads >>>(master, copy, size);
}


__global__ void cuda_update_weights_FP32(float *weights, float* update, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= update[i];
	}
}

__global__ void cuda_update_weights_FP16_mixed(float *weights, half* update, int size, float TC_scale_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= (float)update[i] / TC_scale_factor;
	}
}

void cuda_update_weights(network* net, void *weights, void* update, int size)
{
	cu_blocks = (size + cu_threads - 1) / cu_threads;
	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			cuda_update_weights_FP32<<< cu_blocks, cu_threads >>>((float*)weights, (float*)update, size);
			break;
		case 1:
			cuda_update_weights_FP16_mixed<<< cu_blocks, cu_threads >>>((float*)weights, (half*)update, size, TC_scale_factor);
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


void cuda_print_table(network* net, void* tab, int size, int return_every)
{
	int i;
	void *temp;
	
	printf("\n");
	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			temp = (void*) malloc(size*sizeof(float));
			cudaMemcpy(temp, tab, size*sizeof(float), cudaMemcpyDeviceToHost);
			for(i = 0; i < size; i ++)
			{
				if(i%return_every == 0)
					printf("\n");
				printf("%g \t ", ((float*)temp)[i]);
			}
			break;
		case 1:
			temp = (void*) malloc(size*sizeof(half));
			cudaMemcpy(temp, tab, size*sizeof(half), cudaMemcpyDeviceToHost);
			for(i = 0; i < size; i ++)
			{
				if(i%return_every == 0)
					printf("\n");
				printf("%g \t ", (float)(((half*)temp)[i]));
			}
			break;
	}
	printf("\n");
	
	free(temp);
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

void cuda_confmat(network *net, float* mat)
{
	cu_blocks = (net->length + cu_threads - 1) / cu_threads;
	
	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			add_confmat_FP32<<< cu_blocks, cu_threads>>>((float*)net->net_layers[net->nb_layers-1]->output, (float*)net->target, mat, net->length, net->output_dim);
			break;
		case 1:
			add_confmat_FP16<<< cu_blocks, cu_threads>>>((half*)net->net_layers[net->nb_layers-1]->output, (half*)net->target, mat, net->length, net->output_dim);
			break;
	}
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

__global__ void get_back_shuffle_FP16(half **in, half **targ, half** train_dupl, half** targ_dupl,
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
		case 0:
			shfl_kern_FP32<<< cu_blocks, cu_threads>>>((float**)data.input_device,
				(float**)data.target_device, (float**)duplicate.input_device,
				(float**)duplicate.target_device, index_shuffle_device, 
				data.size, net->batch_size, net->input_dim+1, net->output_dim);
			get_back_shuffle_FP32<<< cu_blocks, cu_threads>>>((float**)data.input_device,
				(float**)data.target_device, (float**)duplicate.input_device,
				(float**)duplicate.target_device, data.size, net->batch_size, 
				net->input_dim+1, net->output_dim);
			break;
		case 1:
			shfl_kern_FP16<<< cu_blocks, cu_threads>>>((half**)data.input_device, 
				(half**)data.target_device, (half**)duplicate.input_device, 
				(half**)duplicate.target_device, index_shuffle_device, 
				data.size, net->batch_size, net->input_dim+1, net->output_dim);
			get_back_shuffle_FP16<<< cu_blocks, cu_threads>>>((half**)data.input_device,
				(half**)data.target_device, (half**)duplicate.input_device, 
				(half**)duplicate.target_device, data.size, net->batch_size, 
				net->input_dim+1, net->output_dim);
			break;
	}
}

void host_shuffle(network *net, Dataset data, Dataset duplicate)
{
	int i, j, k;
	float temp;
	half temp_half;
	int pos, pos2, batch, batch2;

	float *f_d_in_A, *f_d_in_B, *f_d_targ_A, *f_d_targ_B;
	half *h_d_in_A, *h_d_in_B, *h_d_targ_A, *h_d_targ_B;

	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 
					* (net->output_dim)*sizeof(float), cudaMemcpyDeviceToHost);
			}
			break;
		case 1:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(duplicate.input[i], data.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(half), cudaMemcpyDeviceToHost);
				cudaMemcpy(duplicate.target[i], data.target[i], net->batch_size 
					* (net->output_dim)*sizeof(half), cudaMemcpyDeviceToHost);
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
			case 0:
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
			case 1:
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
		}
	}
	
	switch(net->use_cuda_TC)
	{
		default:
		case 0:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 
					* (net->output_dim)*sizeof(float), cudaMemcpyHostToDevice);
			}
			break;
		case 1:
			for(i = 0; i < data.nb_batch; i++)
			{
				cudaMemcpy(data.input[i], duplicate.input[i], net->batch_size 
					* (net->input_dim + 1)*sizeof(half), cudaMemcpyHostToDevice);
				cudaMemcpy(data.target[i], duplicate.target[i], net->batch_size 
					* (net->output_dim)*sizeof(half), cudaMemcpyHostToDevice);
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
		case 0:
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
		case 1:
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
	}
}







