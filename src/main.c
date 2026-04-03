
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


#include "prototypes.h"

// This file is here to illustrate how CIANNA could be used directly in C.
// There is no wiki for the C interface ATM, but all the usefull C functions are used in the python_module.c file.
// The MNIST data loading is not done here, run the python example first to automatically download the data files.

int main()
{
	FILE *f = NULL;
	int i, j;
	int train_size, test_size, valid_size;
	int dims[4];
	float *temp;
	
	int out_dim;
	network *net;

	train_size = 60000; test_size = 10000; valid_size = 10000;
	dims[0] = 28; dims[1] = 28; dims[2] = 1; dims[3] = 1; out_dim = 10;
	
	init_network(0, dims, out_dim, 16, "ADAM", 1, "C_CUDA", 1, "FP32C_FP32A", 0, 0, 0);
	
	
	net = networks[0];
	
	net->train = create_dataset(net, 1, train_size);
	net->valid = create_dataset(net, 1, valid_size);
	net->test  = create_dataset(net, 0, test_size );
	
	f = fopen("examples/MNIST/mnist_dat/mnist_input.dat", "rb+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}
	
	temp = (float*) malloc(net->input_dim*sizeof(float));
	
	for(i = 0; i < net->train.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->train.size)
				continue;
			fread(temp, sizeof(float), net->input_dim, f);
			net->train.cont_copy(temp, net->train.input[i], j*(net->input_dim + 1), net->input_dim);
		}
	}

	for(i = 0; i < net->valid.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->valid.size)
				continue;
			fread(temp, sizeof(float), net->input_dim, f);
			net->valid.cont_copy(temp, net->valid.input[i], j*(net->input_dim + 1), net->input_dim);
		}
	}
	
	for(i = 0; i < net->test.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->test.size)
				continue;
			fread(temp, sizeof(float), net->input_dim, f);
			net->test.cont_copy(temp, net->test.input[i], j*(net->input_dim + 1), net->input_dim);
		}
	}
	
	fclose(f);
	free(temp);
	
	f = fopen("examples/MNIST/mnist_dat/mnist_target.dat", "rb+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}
	temp = (float*) malloc(net->output_dim*sizeof(float));
	
	for(i = 0; i < net->train.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->train.size)
				continue;
			fread(temp, sizeof(float), net->output_dim, f);
			net->train.cont_copy(temp, net->train.target[i], j*net->output_dim, net->output_dim);
		}
	}
	
	for(i = 0; i < net->valid.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->valid.size)
				continue;
			fread(temp, sizeof(float), net->output_dim, f);
			net->valid.cont_copy(temp, net->valid.target[i], j*net->output_dim, net->output_dim);
		}
	}
	
	fclose(f);
	free(temp);
	
	//Must be converted if Dynamic load is off !
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.dynamic_load == 0)
	{
		cuda_get_batched_dataset(net, &net->train);
		cuda_get_batched_dataset(net, &net->valid);
		cuda_get_batched_dataset(net, &net->test);
	}
	#endif
	
	int f_size[3]  = {5,5,1};
	int stride[3]  = {1,1,1};
	int padding[3] = {2,2,0};
	int int_pad[3] = {0,0,0};
	
	int pooling[3] = {2,2,1};
	int pool_padding[3] = {0,0,0};
	int pool_stride[3] = {2,2,1};
	
	conv_create(net, NULL, f_size, 8, 1, stride, padding, int_pad, NULL, "RELU", NULL, 0.0, "xavier", -1.0, NULL, 0, 0);
	pool_create(net, net->net_layers[net->nb_layers-1], pooling, pool_stride, pool_padding, "MAX", NULL, 0, 0.0);
	conv_create(net, net->net_layers[net->nb_layers-1], f_size, 16, 1, stride, padding, int_pad, NULL, "RELU", NULL, 0.0, "xavier", -1.0, NULL, 0, 0);
	pool_create(net, net->net_layers[net->nb_layers-1], pooling, pool_stride, pool_padding, "MAX", NULL, 0, 0.0);
	dense_create(net, net->net_layers[net->nb_layers-1], 256, "RELU", NULL, 0.5, 0, "xavier", -1.0, NULL, 0, 0);
	dense_create(net, net->net_layers[net->nb_layers-1], 128, "RELU", NULL, 0.2, 0, "xavier", -1.0, NULL, 0, 0);
	dense_create(net, net->net_layers[net->nb_layers-1], net->output_dim, "SMAX", NULL, 0.0, 1, "xavier", -1.0, NULL, 0, 0);
	
	printf("Start learning phase ...\n");
	
	train_network(net, 10, 1, 0.0002, 0.0, 0.0, 0.0, 1, 0.999, 0, 1, 10, 0, 0, 0, 0, 1.0, 0);

	exit(EXIT_SUCCESS);
}





