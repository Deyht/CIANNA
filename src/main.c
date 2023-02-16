
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




#include "prototypes.h"

int main()
{
	FILE *f = NULL;
	int i, j, k;
	int train_size, test_size, valid_size;
	int dims[4];
	
	int out_dim;
	network *net;
	
	//Common randomized seed based on time at execution
	srand(time(NULL));
	

	train_size = 60000; test_size = 10000; valid_size = 10000;
	dims[0] = 28; dims[1] = 28; dims[2] = 1; dims[3] = 1; out_dim = 10;
	
	init_network(0, dims, out_dim, 0.1, 24, "C_CUDA", 1, "off", 0, 0);
	
	
	net = networks[0];
	
	net->train = create_dataset(net, train_size);
	net->test  = create_dataset(net, test_size );
	net->valid = create_dataset(net, valid_size);
	
	
	f = fopen("mnist_dat/mnist_input.dat", "rb+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}
	
	for(i = 0; i < net->train.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->train.size)
				continue;
			for(k = 0; k < net->input_dim; k ++)
				fread(&((float**)net->train.input)[i][j*(net->input_dim+1) + k], sizeof(float), 1, f);
			//bias value should be adapted somehow based on 1st layer
		}
	}
	
	for(i = 0; i < net->test.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->test.size)
				continue;
			for(k = 0; k < net->input_dim; k ++)
				fread(&((float**)net->test.input)[i][j*(net->input_dim+1) + k], sizeof(float), 1, f);
		}
	}
	
	for(i = 0; i < net->valid.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->valid.size)
				continue;
			for(k = 0; k < net->input_dim; k ++)
				fread(&((float**)net->valid.input)[i][j*(net->input_dim+1) + k], sizeof(float), 1, f);
		}
	}
	
	fclose(f);
	
	f = fopen("mnist_dat/mnist_target.dat", "rb+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}
	
	
	for(i = 0; i < net->train.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->train.size)
				continue;
			for(k = 0; k < net->output_dim; k ++)
				fread(&((float**)net->train.target)[i][j*(net->output_dim) + k], sizeof(float), 1, f);
		}
	}
	
	for(i = 0; i < net->test.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->test.size)
				continue;
			for(k = 0; k < net->output_dim; k ++)
				fread(&((float**)net->test.target)[i][j*(net->output_dim) + k], sizeof(float), 1, f);
		}
	}
	
	for(i = 0; i < net->valid.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->valid.size)
				continue;
			for(k = 0; k < net->output_dim; k ++)
				fread(&((float**)net->valid.target)[i][j*(net->output_dim) + k], sizeof(float), 1, f);
		}
	}
	
	fclose(f);
	
	
	
	//Must be converted if Dynamic load is off !
	#ifdef CUDA
	
	if(net->compute_method == C_CUDA && net->cu_inst.dynamic_load == 0)
	{
		cuda_convert_dataset(net, &net->train);
		cuda_convert_dataset(net, &net->test);
		cuda_convert_dataset(net, &net->valid);
	}
	else if(net->compute_method == C_CUDA && net->cu_inst.dynamic_load == 1 && net->cu_inst.use_cuda_TC)
	{
		cuda_convert_host_dataset(net, &net->train);
		cuda_convert_host_dataset(net, &net->test);
		cuda_convert_host_dataset(net, &net->valid);
	}
	#endif
	
	int f_size[3]  = {5,5,1};
	int stride[3]  = {1,1,1};
	int padding[3] = {2,2,0};
	int int_pad[3] = {0,0,0};
	
	int pooling[3] = {2,2,1};
	
	conv_create(net, NULL, f_size, 8, stride, padding, int_pad, NULL, "RELU", NULL, 0.0, NULL, -1.0, NULL, 0);
	pool_create(net, net->net_layers[net->nb_layers-1], pooling, "MAX", 0, 0.0);
	conv_create(net, net->net_layers[net->nb_layers-1], f_size, 16, stride, padding, int_pad, NULL, "RELU", NULL, 0.0, NULL, -1.0, NULL, 0);
	pool_create(net, net->net_layers[net->nb_layers-1], pooling, "MAX", 0, 0.0);
	dense_create(net, net->net_layers[net->nb_layers-1], 256, "RELU", NULL, 0.5, 0, NULL, -1.0, NULL, 0);
	dense_create(net, net->net_layers[net->nb_layers-1], 128, "RELU", NULL, 0.2, 0, NULL, -1.0, NULL, 0);
	dense_create(net, net->net_layers[net->nb_layers-1], net->output_dim, "SOFTMAX", NULL, 0.0, 0, NULL, -1.0, NULL, 0);
	
	printf("Start learning phase ...\n");
	
	train_network(net, 10, 1, 0.0002, 0.0, 0.9, 0.0, 0.0, 1, 5, 0, 1, 1, 1.0, 0);

	exit(EXIT_SUCCESS);
}





