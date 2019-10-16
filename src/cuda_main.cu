#include "prototypes.h"

static int cu_blocks;
int cu_threads = CUDA_THREADS_PER_BLOCKS;
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

void cuda_put_table(real **cuda_table, real **table, int size)
{
	cudaMemcpy(*cuda_table, *table, size*sizeof(real), cudaMemcpyHostToDevice);
}

void cuda_convert_batched_table(real **tab, int batch_size, int nb_batch, int size)
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

void cuda_convert_dataset(network *net, Dataset *data)
{
	cuda_convert_batched_table(data->input, net->batch_size, data->nb_batch, net->input_dim + 1);
	cuda_convert_batched_table(data->target, net->batch_size, data->nb_batch, net->output_dim);
	cudaMalloc(&(data->input_device), data->nb_batch*sizeof(real*));
	cudaMemcpy(data->input_device, data->input, data->nb_batch*sizeof(real*),cudaMemcpyHostToDevice);
	cudaMalloc(&(data->target_device), data->nb_batch*sizeof(real*));
	cudaMemcpy(data->target_device, data->target, data->nb_batch*sizeof(real*),cudaMemcpyHostToDevice);
	data->localization = DEVICE;
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

void cuda_confmat(network *net, real* mat)
{
	cu_blocks = (net->length + cu_threads - 1) / cu_threads;
	add_confmat<<< cu_blocks, cu_threads>>>(net->net_layers[net->nb_layers-1]->output, net->target, mat, net->length, net->output_dim);
}

__global__ void shfl_kern(real **in, real **targ, real** train_dupl, real** targ_dupl,
									real* index, int in_size, int b_size, int d_in, int d_out)
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



__global__ void get_back_shuffle(real **in, real **targ, real** train_dupl, real** targ_dupl,
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


void cuda_shuffle(network *net, Dataset data, Dataset duplicate, real *index_shuffle, real *index_shuffle_device)
{
	int i, j;
	real temp;

	for(i = 0; i < data.size - 1; i++)
	{
		j = (i + rand() / ((real)RAND_MAX) *(data.size-i));
		temp = index_shuffle[i];
		index_shuffle[i] = index_shuffle[j];
		index_shuffle[j] = temp;
	}
	
	cudaMemcpy(index_shuffle_device, index_shuffle, data.size*sizeof(real), cudaMemcpyHostToDevice);
	
	cu_blocks = (data.size + cu_threads - 1) / cu_threads;
	shfl_kern<<< cu_blocks, cu_threads>>>(data.input_device, data.target_device, duplicate.input_device, 
		duplicate.target_device, index_shuffle_device, data.size, net->batch_size, 
		net->input_dim+1, net->output_dim);

	get_back_shuffle<<< cu_blocks, cu_threads>>>(data.input_device, data.target_device,
		duplicate.input_device, duplicate.target_device, data.size, net->batch_size, 
		net->input_dim+1, net->output_dim);

}






