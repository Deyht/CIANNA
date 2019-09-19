#include "prototypes.h"

static int cu_blocks;
int cu_threads = 128;
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

void cuda_free_table(real* tab)
{
	cudaFree(tab);
}

void cuda_convert_table(real **tab, int size)
{
	real* temp_tab;
	
	temp_tab = *tab;
	cudaMalloc(tab, size*sizeof(real));
	cudaMemcpy(*tab, temp_tab, size*sizeof(real),cudaMemcpyHostToDevice);
	free(temp_tab);
}

void cuda_create_table(real **tab, int size)
{
	cudaMalloc(tab, size*sizeof(real));
	cudaMemset(*tab, 0.0, size*sizeof(real));
}

void cuda_get_table(real **cuda_table, real **table, int size)
{
	cudaMemcpy(*table, *cuda_table, size*sizeof(real), cudaMemcpyDeviceToHost);
}

void cuda_convert_batched_table(real **tab, int nb_batch, int size)
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

void cuda_convert_dataset(Dataset data)
{
	cuda_convert_batched_table(data.input, data.nb_batch, input_dim + 1);
	cuda_convert_batched_table(data.target, data.nb_batch, output_dim);
}

__global__ void cuda_update_weights(real *weights, real* update, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < size)
	{
		weights[i] -= update[i];
	}
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


__device__ int cuda_argmax(real* tab, int dim_out)
{
	int i;
	real vmax;
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

__global__ void add_confmat(real *out, real *targ, real *mat, int len, int o_dim)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int arg1, arg2;
	if( i < len)
	{
		targ += i*o_dim;
		out += i*(o_dim+1);
		
		arg1 = cuda_argmax(targ, o_dim);
		arg2 = cuda_argmax(out, o_dim);
		
		atomicAdd(&(mat[arg2+o_dim*arg1]), 1);
	}
}

void cuda_confmat(real *out, real* mat)
{
	cu_blocks = (length + cu_threads - 1) / cu_threads;
	add_confmat<<< cu_blocks, cu_threads>>>(out, target, mat, length, output_dim);
}




