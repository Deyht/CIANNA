#include "prototypes.h"

real **train_data;
real **train_target;
int train_size;

real **test_data;
real **test_target;
int test_size;

real **valid_data;
real **valid_target;
int valid_size;

real* input;
int input_width = 4, input_height = 1, input_depth = 1;
int output_dim;
real* target;
int batch_size = 2;
int nb_batch;
int length = 2;
real learning_rate = 0.000001;
real momentum = 0.6;
int compute_method = C_CUDA;

int nb_layers;



void confmat(layer *net_layer)
{
	int i, j, k;
	float** mat;
	float* temp;
	real count;
	//int arg1, arg2;
	real *rapp_err, *rapp_err_rec;
	int proxy_count = 0;
	int o;
	
	printf("\nConfMat (test set)\n");
	
	o = ((dense_param*)net_layer[nb_layers-1].param)->nb_neurons;
	
	rapp_err = (real*) malloc(o*sizeof(real));
	rapp_err_rec = (real*) malloc(o*sizeof(real));
	mat = (float**) malloc(o*sizeof(float*));
	temp = (float*) malloc(o*o*sizeof(float));
	for(i = 0; i < o; i++)
		mat[i] = &(temp[i*o]);

	switch(compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_confmat(mat, net_layer);
			#endif
			break;
			
		default:
			break;
	}
	
	for(j = 0; j < o; j++)
	{
		rapp_err[j] = 0.0;
		rapp_err_rec[j] = 0.0;
		for(k = 0; k < o; k++)
		{
			rapp_err[j] += mat[j][k];
			rapp_err_rec[j] += mat[k][j];
		}
		rapp_err[j] = mat[j][j]/rapp_err[j]*100.0;
		rapp_err_rec[j] = mat[j][j]/rapp_err_rec[j]*100.0;
	}
	printf("%*s\n", (o+2)*11, "Recall");
	for(j = 0; j < o; j++)
	{
		printf("%*s", 10, " ");
		for(k = 0; k < o; k++)
			printf("%10d |", (int) mat[j][k]);
		printf("\t %.2f%%\n", rapp_err[j]);
	}
	printf("%10s", "Precision :");
	for(j = 0; j < o; j++)
		printf("%9.2f%%  ", rapp_err_rec[j]);
	
	count = 0.0;
	for(i = 0; i < o; i++)
		count += mat[i][i];
	
	printf("\nNumb extracted : %d\n", proxy_count);
	printf("Correct : %f%%\n", count/train_size*100);
	
	free(temp);
	free(rapp_err);
	free(rapp_err_rec);
}



int main()
{
	FILE *f;
	int i, j, k;
	
	int input_dim;
	float dump;
	
	layer net_layer[4];	
	nb_layers = 4;
	
	//Common randomized seed based on time at execution
	srand(time(NULL));
	
	#ifdef CUDA
	init_cuda();
	#endif
	
	f = fopen("input.dat", "r+");

	fscanf(f, "%d %d %d %dx%dx%d %d", &train_size, &test_size, &valid_size, &input_width, &input_height,
		&input_depth, &output_dim);
		
	input_dim = input_width*input_height*input_depth;
	
	nb_batch = (train_size) / batch_size + 1;
	train_data = (real**) malloc(nb_batch*sizeof(real*));
	train_target = (real**) malloc(nb_batch*sizeof(real*));
	
	for(i = 0; i < nb_batch; i++)
	{
		train_data[i] = (real*) malloc(batch_size * (input_dim + 1) * sizeof(real));
		train_target[i] = (real*) malloc(batch_size * output_dim * sizeof(real));
	}
	
	for(i = 0; i < nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= train_size)
				continue;
			for(k = 0; k < input_dim; k ++)
				fscanf(f, "%f", &train_data[i][j*(input_dim+1) + k]);
			train_data[i][j*(input_dim+1) + input_dim] = 0.1;
		}
	}
	for(i = 0; i < test_size; i++)
	{
		for(j = 0; j < input_dim; j++)
			fscanf(f, "%f", &dump);
	}
	for(i = 0; i < valid_size; i++)
	{
		for(j = 0; j < input_dim; j++)
			fscanf(f, "%f", &dump);
	}
	
	for(i = 0; i < nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			if(i*batch_size + j >= train_size)
				continue;
			for(k = 0; k < output_dim; k ++)
				fscanf(f, "%f", &train_target[i][j*(output_dim) + k]);
		}
	}

	fclose(f);
	
	
	for(i = 0; i < 28; i++)
	{
		for( j = 0; j < 28; j++)
			printf("%d \t",(int)(255*train_data[0][i*28+j]));
		printf("\n"); 
	
	}
	
	
	#ifdef CUDA
	for(i = 0; i < nb_batch; i++)
	{
		cuda_convert_table(&train_data[i], batch_size * (input_dim + 1));
		cuda_convert_table(&train_target[i], batch_size * output_dim);
	}
	#endif
	
	cuda_print_table(train_data[0], batch_size * (input_dim + 1), 28);
	
	printf("pre conv create\n");
	conv_create(&net_layer[0], NULL, 5, 4, 1, 0, CONV);
	printf("post conv create\n");
	dense_create(&net_layer[1], &net_layer[0], 100, RELU);
	dense_create(&net_layer[2], &net_layer[1], 100, RELU);
	dense_create(&net_layer[3], &net_layer[2], output_dim, SOFTMAX);
	
		
	printf("Start learning phase ...\n");
	
	for(i = 0; i < 1000; i++)
	{
		printf("loop : %d\n", i);
		for(k = 0; k < nb_batch; k++)
		{
			//Loop on all batch for one epoch
			input = train_data[k];
			target = train_target[k];
			//forward
			for(j = 0; j < nb_layers; j++)
			{
				printf("\t layer : %d\n", j);
				net_layer[j].forward(&net_layer[j]);
			}
			
			//backward
			printf("output_error\n");
			output_error(&net_layer[nb_layers-1]);
			for(j = 0; j < nb_layers; j++)
			{
				printf("\t layer : %d\n", nb_layers-1-j);
				net_layer[nb_layers-1-j].backprop(&net_layer[nb_layers-1-j]);
			}
		}
		//confmat(net_layer);
	}

	exit(EXIT_SUCCESS);
}





