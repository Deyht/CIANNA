


#include "prototypes.h"


int main()
{
	FILE *f;
	int i, j, k;
	int train_size, test_size, valid_size;
	int dims[3];
	
	int out_dim;
	
	Dataset train, test, valid;
	
	//Common randomized seed based on time at execution
	//srand(time(NULL));
	
	
	#ifdef CUDA
	init_cuda();
	#endif
	
	f = fopen("input.dat", "r+");

	fscanf(f, "%d %d %d %dx%dx%d %d", &train_size, &test_size, &valid_size, &dims[0], &dims[1],
		&dims[2], &out_dim);
		
	
	init_network(dims, out_dim, 512, C_CUDA);
	
	train = create_dataset(train_size);
	test = create_dataset(test_size);
	valid = create_dataset(valid_size);
	
	for(i = 0; i < train.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= train.size)
				continue;
			for(k = 0; k < input_dim; k ++)
				fscanf(f, "%f", &train.input[i][j*(input_dim+1) + k]);
			train.input[i][j*(input_dim+1) + input_dim] = 0.1; 
			//bias value should be adapted somehow based on 1st layer
		}
	}
	
	for(i = 0; i < test.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= test.size)
				continue;
			for(k = 0; k < input_dim; k ++)
				fscanf(f, "%f", &test.input[i][j*(input_dim+1) + k]);
			test.input[i][j*(input_dim+1) + input_dim] = 0.1;
		}
	}
	
	for(i = 0; i < valid.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= valid.size)
				continue;
			for(k = 0; k < input_dim; k ++)
				fscanf(f, "%f", &valid.input[i][j*(input_dim+1) + k]);
			valid.input[i][j*(input_dim+1) + input_dim] = 0.1;
		}
	}
	
	
	for(i = 0; i < train.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= train.size)
				continue;
			for(k = 0; k < output_dim; k ++)
				fscanf(f, "%f", &train.target[i][j*(output_dim) + k]);
		}
	}
	
	for(i = 0; i < test.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= test.size)
				continue;
			for(k = 0; k < output_dim; k ++)
				fscanf(f, "%f", &test.target[i][j*(output_dim) + k]);
		}
	}
	
	for(i = 0; i < valid.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= valid.size)
				continue;
			for(k = 0; k < output_dim; k ++)
				fscanf(f, "%f", &valid.target[i][j*(output_dim) + k]);
		}
	}

	fclose(f);
	
	/*
	for(i = 0; i < 28; i++)
	{
		for( j = 0; j < 28; j++)
			printf("%d \t",(int)(255*train_data[nb_batch-1][i*28+j]));
		printf("\n"); 
	}
	*/
	
	
	
	#ifdef CUDA
	cuda_convert_dataset(train);
	cuda_convert_dataset(test);
	cuda_convert_dataset(valid);
	#endif
	
	/*
	//LeNet-1
	conv_create(&net_layer[0], NULL, 5, 4, 1, 0, RELU);
	pool_create(&net_layer[1], &net_layer[0], 2);
	conv_create(&net_layer[2], &net_layer[1], 5, 12, 1, 0, RELU);
	pool_create(&net_layer[3], &net_layer[2], 2);
	dense_create(&net_layer[4], &net_layer[3], output_dim, SOFTMAX);
	*/
	
	/*
	//LeNet-4
	conv_create(&net_layer[0], NULL, 5, 4, 1, 4, RELU);
	pool_create(&net_layer[1], &net_layer[0], 2);
	conv_create(&net_layer[2], &net_layer[1], 5, 16, 1, 0, RELU);
	pool_create(&net_layer[3], &net_layer[2], 2);
	dense_create(&net_layer[4], &net_layer[3], 120, RELU);
	dense_create(&net_layer[5], &net_layer[4], output_dim, SOFTMAX);
	*/
	
	//Modified LeNet-5
	
	conv_create(NULL, 5, 6, 1, 4, RELU);
	pool_create(net_layers[nb_layers-1], 2);
	conv_create(net_layers[nb_layers-1], 5, 16, 1, 0, RELU);
	pool_create(net_layers[nb_layers-1], 2);
	dense_create(net_layers[nb_layers-1], 256, RELU);
	dense_create(net_layers[nb_layers-1], 128, RELU);
	dense_create(net_layers[nb_layers-1], output_dim, SOFTMAX);
	
	
	/*
	dense_create(&net_layer[0], NULL, 200, LOGISTIC);
	dense_create(&net_layer[1], &net_layer[0], 200, LOGISTIC);
	dense_create(&net_layer[2], &net_layer[1], output_dim, SOFTMAX);
	*/
	
	printf("Start learning phase ...\n");
	
	enable_confmat();
	
	train_network(train, valid, 40, 1, 0.0004, 0.6, 0.003);

	exit(EXIT_SUCCESS);
}





