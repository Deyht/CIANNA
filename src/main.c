
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
	int train_size, test_size, valid_size;
	int dims[4];
	char compute_method[10];
	int out_dim;
	network *net;

	train_size = 60000; test_size = 10000; valid_size = 10000;
	dims[0] = 28; dims[1] = 28; dims[2] = 1; dims[3] = 1; out_dim = 10;
	
	strncpy(compute_method, "C_NAIV", 10);
	#ifdef CUDA
	strncpy(compute_method, "C_CUDA", 10);
	printf(" /!\\ Highest backend detected is CUDA. CIANNA will use the C_CUDA compute method. /!\\\n");
	#elif BLAS == 1
	strncpy(compute_method, "C_BLAS", 10);
	printf(" Highest backend detected is BLAS. CIANNA will use the C_BLAS compute method.\n");
	#else
	printf(" Highest backend detected is NAIV. CIANNA will use the C_NAIV compute method.\n");
	#endif
	
	init_network(0,     /*network_number*/
		dims,           /*u_input_dim*/
		out_dim,        /*u_output_dim*/
		0.1,            /*in_bias*/
		16,             /*u_batch_size*/
		compute_method, /*compute_method_string*/
		1,              /*u_dynamic_load*/
		"off",          /*cuda_TC_string*/
		0,              /*inference_only*/
		0,              /*no_logo*/
		0               /*adv_size*/);
	
	net = networks[0];
	
	/*Create CIANNA dataset structures*/
	net->train = create_dataset(net, train_size);
	net->test  = create_dataset(net, test_size );
	net->valid = create_dataset(net, valid_size);
	
	//For testing on closed systems or environments without access to network
	#if !defined TEST_MODE
	int i, j, k;
	FILE *f = NULL;
	
	/*Download the dataset if not available*/
	if(access("mnist_dat", F_OK) != 0)
	{
		system("wget https://share.obspm.fr/s/EkYR5B2Wc2gNis3/download/mnist.tar.gz");
		system("tar -xvzf mnist.tar.gz");
	}
	
	f = fopen("mnist_dat/mnist_input.dat", "rb+");
	if(f == NULL)
	{
		printf("ERROR: Can not open input file ...\n");
		exit(1);
	}
	
	/*Fill CIANNA datasets*/
	for(i = 0; i < net->train.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= net->train.size)
				continue;
			for(k = 0; k < net->input_dim; k ++)
				fread(&((float**)net->train.input)[i][j*(net->input_dim+1) + k], sizeof(float), 1, f);
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
	
	#endif //TEST_MODE
	
	
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
	
	/*Generic layer configurations*/
	int f_size[3]  = {5,5,1};
	int stride[3]  = {1,1,1};
	int padding[3] = {2,2,0};
	int int_pad[3] = {0,0,0};
	
	int pooling[3] = {2,2,1};
	int pool_padding[3] = {0,0,0};
	int pool_stride[3] = {2,2,1};
	
	/*########## Sequential backbone creation ##########*/

	//CONV 1
	conv_create(net,                        /*network*/
		NULL,                               /*previous_layer*/ 
		f_size,                             /*f_size*/ 
		8,                                  /*nb_filters*/
		stride,                             /*stride*/
		padding,                            /*padding*/
		int_pad,                            /*int_padding*/
		NULL,                               /*in_shape*/
		"RELU",                             /*activation*/
		NULL,                               /*bias*/
		0.0,                                /*drop_rate*/
		"xavier",                           /*init_fct*/
		-1.0,                               /*init_scaling*/
		NULL,                               /*file_load*/
		0                                   /*f_bin*/);
	
	//POOL 1
	pool_create(net,                        /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/
		pooling,                            /*pool_size*/
		pool_stride,                        /*stride*/
		pool_padding,                       /*padding*/
		"MAX",                              /*char_pool_type*/
		NULL,                               /*activation*/
		0,                                  /*global*/
		0.0                                 /*drop_rate*/);
	
	//CONV 2
	conv_create(net,                        /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/ 
		f_size,                             /*f_size*/ 
		16,                                 /*nb_filters*/
		stride,                             /*stride*/
		padding,                            /*padding*/
		int_pad,                            /*int_padding*/
		NULL,                               /*in_shape*/
		"RELU",                             /*activation*/
		NULL,                               /*bias*/
		0.0,                                /*drop_rate*/
		"xavier",                           /*init_fct*/
		-1.0,                               /*init_scaling*/
		NULL,                               /*file_load*/
		0                                   /*f_bin*/);
	
	//POOL 2
	pool_create(net,                        /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/
		pooling,                            /*pool_size*/
		pool_stride,                        /*stride*/
		pool_padding,                       /*padding*/
		"MAX",                              /*char_pool_type*/
		NULL,                               /*activation*/
		0,                                  /*global*/
		0.0                                 /*drop_rate*/);
	
	//DENSE 1
	dense_create(net,                       /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/
		256,                                /*nb_neurons*/
		"RELU",                             /*activation*/
		NULL,                               /*bias*/
		0.5,                                /*drop_rate*/
		0,                                  /*strict_size*/
		"xavier",                           /*init_fct*/
		-1.0,                               /*init_scaling*/
		NULL,                               /*f_load*/
		0                                   /*f_bin*/);
	
	dense_create(net,                       /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/
		128,                                /*nb_neurons*/
		"RELU",                             /*activation*/
		NULL,                               /*bias*/
		0.2,                                /*drop_rate*/
		0,                                  /*strict_size*/
		"xavier",                           /*init_fct*/
		-1.0,                               /*init_scaling*/
		NULL,                               /*f_load*/
		0                                   /*f_bin*/);
	
	dense_create(net,                       /*network*/
		net->net_layers[net->nb_layers-1],  /*previous_layer*/
		net->output_dim,                    /*nb_neurons*/
		"SMAX",                             /*activation*/
		NULL,                               /*bias*/
		0.0,                                /*drop_rate*/
		1,                                  /*strict_size*/
		"xavier",                           /*init_fct*/
		-1.0,                               /*init_scaling*/
		NULL,                               /*f_load*/
		0                                   /*f_bin*/);
	
	/*##################################################*/
	
	printf("Start learning phase ...\n");
	
	/*Fit the model */
	train_network(net,  /*network*/
		2,              /*nb_iter*/
		1,              /*control_interv*/
		0.001,          /*u_begin_learning_rate*/
		0.0,            /*u_end_learning_rate*/
		0.9,            /*u_momentum*/
		0.0,            /*u_decay*/
		0.0,            /*u_weight_decay*/
		1,              /*show_confmat*/
		5,              /*save_every*/
		0,              /*save_bin*/
		1,              /*shuffle_gpu*/
		1,              /*shuffle_every*/
		1.0,            /*c_TC_scale_factor*/
		0               /*silent*/);

	perf_eval_display(net);

	#if defined TEST_MODE
	printf("\nCOMPILE TEST PASSED !\n");
	#endif

	exit(EXIT_SUCCESS);
}





