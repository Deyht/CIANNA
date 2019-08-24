#include "prototypes.h"

int cu_threads = 128;
static int cu_blocks;
real cu_alpha = 1.0, cu_beta = 0.0;
cublasHandle_t cu_handle;

//local prototypes
__global__ void add_confmat(real *out, real *targ, float *mat, int len, int batch_size, int o_dim);
__device__ int cuda_argmax(real* tab, int pos, int len, int size, int format);


void init_cuda(void)
{
	if(cublasCreate(&cu_handle) != CUBLAS_STATUS_SUCCESS) 
	{
		printf("GPU handle create fail\n");
		exit(EXIT_FAILURE);
	}
	//place holder for device selection
}

void cuda_convert_table(real **tab, int size)
{
	//to adapt to convert all the batches at once

	real* temp_tab;
	
	temp_tab = *tab;
	cudaMalloc(tab, size*sizeof(real));
	cudaMemcpy(*tab, temp_tab, size*sizeof(real),cudaMemcpyHostToDevice);
	free(temp_tab);
}

void cuda_convert_batched_table(real **tab, int nb_batch, int batch_size, int size)
{
	int i;
	real* temp_tab;
	
	for(i = 0; i < nb_batch; i++)
	{
		temp_tab = tab[i];
		cudaMalloc(&(tab[i]), batch_size*size*sizeof(real));
		cudaMemcpy(tab[i], temp_tab, batch_size*size*sizeof(real),cudaMemcpyHostToDevice);
		free(temp_tab);
	}
}



__global__ void cuda_update_weights(real *weights, real* update, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	weights[i] -= update[i];

}


void cuda_print_table(real* tab, int size, int return_every)
{
	int i;
	real *temp;
	
	temp = (real*) malloc(size*sizeof(real));
	cudaMemcpy(temp, tab, size*sizeof(real), cudaMemcpyDeviceToHost);
	
	printf("\n");
	for(i = 0; i < size; i ++)
	{
		if(i%return_every == 0)
			printf("\n");
		printf("%g ", (temp[i]));
	}
	printf("\n");
	
	free(temp);
}

void cuda_print_table_transpose(real* tab, int line_size, int column_size)
{
	int i, j;
	int size;
	real *temp;
	
	size = line_size*column_size;
	temp = (real*) malloc(size*sizeof(real));
	cudaMemcpy(temp, tab, size*sizeof(real), cudaMemcpyDeviceToHost);
	
	printf("\n");
	for(i = 0; i < column_size; i ++)
	{
		for(j = 0; j < line_size; j++)
		{
			printf("%g ", temp[j*column_size + i]);
		}
		printf("\n");
	}
	printf("\n");
	
	free(temp);
}


__device__ int cuda_argmax(real* tab, int pos, int len, int size, int format)
{
	int i;
	real vmax;
	int imax;
	
	
	if(format == 1)
	{
		vmax = tab[pos];
		imax = 0;
		for(i = 1; i < size; i++)
		{
			if(tab[pos + i*len] > vmax)
			{
				vmax = tab[pos + i*len];
				imax = i;
			}
		}
	}
	else
	{
		vmax = tab[pos*size];
		imax = 0;
		for(i = 1; i < size; i++)
		{
			if(tab[pos*size + i] > vmax)
			{
				vmax = tab[pos*size + i];
				imax = i;
			}
		}
	}
	return imax;
}


__global__ void add_confmat(real *out, real *targ, float *mat, int len, int batch_size, int o_dim)
{
	//For CNN the output 
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arg1, arg2;
	if( i < len)
	{
		arg1 = cuda_argmax(targ, i, batch_size, o_dim, 0);
		arg2 = cuda_argmax(out, i, batch_size, o_dim+1, 0);
		//Row major
		//if(out[i + arg2*batch_size] > 0.99)
		atomicAdd(&(mat[arg2+o_dim*arg1]), 1);
	}
}

void cuda_confmat(real** data, real** target_data, int nb_data, float** out_mat, layer *net_layer)
{
	int j, k;
	float *mat;
	
	int o;
	
	o = ((dense_param*)net_layer[nb_layers-1].param)->nb_neurons;
	
	cudaMalloc(&mat, o*o*sizeof(float));
	cudaMemset(mat, 0.0, o*o*sizeof(float));
	
	
	for(k = 0; k < nb_batch; k++)
	{
		if(k == nb_batch-1)
		{
			length = nb_data%batch_size;
		}
		else
			length = batch_size;
		
		//Loop on all batch for one epoch
		input = data[k];
		target = target_data[k];
		//forward
		for(j = 0; j < nb_layers; j++)
		{
			//printf("\t layer : %d\n", j);
			net_layer[j].forward(&net_layer[j]);
		}
		cu_blocks = (batch_size + cu_threads - 1) / cu_threads;
		add_confmat<<< cu_blocks, cu_threads >>>(net_layer[nb_layers-1].output, target, mat, length, batch_size, o);
	}
	
	cudaMemcpy(out_mat[0], mat, o*o*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(mat);
}



