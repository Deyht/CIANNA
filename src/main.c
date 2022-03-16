
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
	int dims[3];
	
	int out_dim;
	
	//Common randomized seed based on time at execution
	srand(time(NULL));
	

	f = fopen("mnist_dat/mnist.dat", "r+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}

	fscanf(f, "%d %d %d %dx%dx%d %d", &train_size, &test_size, &valid_size, &dims[0], &dims[1],
		&dims[2], &out_dim);
	
	init_network(0, dims, out_dim, 0.1, 64, C_CUDA, 0, 1, 0);
	
	//train_size = 128;
	//test_size = 128;
	//valid_size = 128;
	
	networks[0]->train = create_dataset(networks[0], train_size);
	networks[0]->test  = create_dataset(networks[0], test_size );
	networks[0]->valid = create_dataset(networks[0], valid_size);
	
	
	for(i = 0; i < networks[0]->train.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->train.size)
				continue;
			for(k = 0; k < networks[0]->input_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->train.input)[i][j*(networks[0]->input_dim+1) + k]);
			//bias value should be adapted somehow based on 1st layer
		}
	}
	
	for(i = 0; i < networks[0]->test.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->test.size)
				continue;
			for(k = 0; k < networks[0]->input_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->test.input)[i][j*(networks[0]->input_dim+1) + k]);
		}
	}
	
	for(i = 0; i < networks[0]->valid.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->valid.size)
				continue;
			for(k = 0; k < networks[0]->input_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->valid.input)[i][j*(networks[0]->input_dim+1) + k]);
		}
	}
	
	for(i = 0; i < networks[0]->train.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->train.size)
				continue;
			for(k = 0; k < networks[0]->output_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->train.target)[i][j*(networks[0]->output_dim) + k]);
		}
	}
	
	for(i = 0; i < networks[0]->test.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->test.size)
				continue;
			for(k = 0; k < networks[0]->output_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->test.target)[i][j*(networks[0]->output_dim) + k]);
		}
	}
	
	for(i = 0; i < networks[0]->valid.nb_batch; i++)
	{
		for(j = 0; j < networks[0]->batch_size; j++)
		{
			if(i*networks[0]->batch_size + j >= networks[0]->valid.size)
				continue;
			for(k = 0; k < networks[0]->output_dim; k ++)
				fscanf(f, "%f", &((float**)networks[0]->valid.target)[i][j*(networks[0]->output_dim) + k]);
		}
	}
	
	fclose(f);
	
	
	
	//Must be converted if Dynamic load is off !
	#ifdef CUDA
	
	if(networks[0]->compute_method == C_CUDA && networks[0]->cu_inst.dynamic_load == 0)
	{
		cuda_convert_dataset(networks[0], &networks[0]->train);
		cuda_convert_dataset(networks[0], &networks[0]->test);
		cuda_convert_dataset(networks[0], &networks[0]->valid);
	}
	else if(networks[0]->compute_method == C_CUDA && networks[0]->cu_inst.dynamic_load == 1 && networks[0]->cu_inst.use_cuda_TC)
	{
		cuda_convert_host_dataset(networks[0], &networks[0]->train);
		cuda_convert_host_dataset(networks[0], &networks[0]->test);
		cuda_convert_host_dataset(networks[0], &networks[0]->valid);
	}
	#endif
	
	/* MUST UPDATE FILTER, STRIDE AND PADDING ARGUMENT PASSING
	conv_create(networks[0], NULL, 5, 8, 1, 2, RELU, 0.0, NULL);
	pool_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 2, 0.0);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 5, 16, 1, 2, RELU, 0.0, NULL);
	pool_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 2, 0.0);
	//conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 48, 1, 1, RELU, 0.0, NULL);
	dense_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 1024, RELU, 0.5, NULL);
	dense_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 256, RELU, 0.2, NULL);
	dense_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 
		networks[0]->output_dim, SOFTMAX, 0.0, NULL);
	*/
	printf("Start learning phase ...\n");
	
	train_network(networks[0], 1, 1, 0.0002, 0.0001, 0.9, 0.009, 1, 5, 1, 1, 2.0);

	exit(EXIT_SUCCESS);
}





