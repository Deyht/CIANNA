#include "prototypes.h"


void init_network(int u_input_dim[3], int u_output_dim, int u_batch_size, int u_compute_method)
{

	srand(time(NULL));
	#ifdef CUDA
	init_cuda();
	#endif

	input_width = u_input_dim[0]; 
	input_height = u_input_dim[1];
	input_depth = u_input_dim[2];
	input_dim = input_width*input_height*input_depth;
	output_dim = u_output_dim;
	batch_size = u_batch_size;
	compute_method = u_compute_method;
	
	output_error = (real*) calloc(batch_size*output_dim,sizeof(real));
	#ifdef CUDA
	if(compute_method == C_CUDA)
		cuda_create_table(&output_error_cuda, batch_size*output_dim);
	#endif
}

void enable_confmat(void)
{
	confusion_matrix = 1;
}


Dataset create_dataset(int nb_elem, real bias)
{
	int i,j;
	Dataset data;

	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / batch_size + 1;
	data.input = (real**) malloc(data.nb_batch*sizeof(real*));
	data.target = (real**) malloc(data.nb_batch*sizeof(real*));
	data.localization = HOST;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		data.input[i] = (real*) calloc(batch_size * (input_dim + 1), sizeof(real));
		data.target[i] = (real*) calloc(batch_size * output_dim, sizeof(real));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < batch_size; j++)
		{
			data.input[i][j*(input_dim+1)+input_dim] = bias;
		}
	}
	
	return data;
}

int argmax(real *tab, int size)
{
	int i;
	real max;
	int imax;

	max = *tab;
	imax = 0;
	
	
	for(i = 1; i < size; i++)
	{
		if(tab[i] >= max)
		{
			max = tab[i];
			imax = i;
		}
	}
	
	return imax;
}


void compute_error(Dataset data, int saving, int repeat)
{
	int j, k, l, r;
	float** mat = NULL; 
	real* cuda_mat;
	float* temp = NULL;
	int arg1, arg2;
	real count;
	real *rapp_err = NULL, *rapp_err_rec = NULL;
	int o, pos;
	real total_error;
	real* temp_error = NULL;
	real* output_save = NULL;
	
	o = output_dim;
	
	if(confusion_matrix)
	{
		printf("Confusion matrix load\n");
		rapp_err = (real*) malloc(o*sizeof(real));
		rapp_err_rec = (real*) malloc(o*sizeof(real));
		mat = (float**) malloc(o*sizeof(float*));
		temp = (float*) malloc(o*o*sizeof(float));
		for(j = 0; j < o; j++)
			mat[j] = &(temp[j*o]);
		
		#ifdef CUDA
		if(compute_method == C_CUDA)
		{
			cuda_create_table(&cuda_mat, o*o);
		}
		#endif
	}
	
	if(saving == 1)
	{
		printf("before_switch\n");
		output_save = (real*) calloc(batch_size*out_size,sizeof(real));	
	}
	
	for(r = 0; r < repeat; r++)
	{
	total_error = 0.0;
	
	for(j = 0; j < data.nb_batch; j++)
	{
		if(compute_method == C_CUDA)
		{
			temp_error = output_error;
			output_error = output_error_cuda;
		}
		
		//##########################################################
		
		if(j == data.nb_batch - 1 && data.size%batch_size > 0)
		{
			length = data.size%batch_size;
		}
		else
			length = batch_size;
		
		//Loop on all batch for one epoch
		input = data.input[j];
		target = data.target[j];
		//forward
		for(k = 0; k < nb_layers; k++)
		{
			net_layers[k]->forward(net_layers[k]);
		}
		output_error_fct(net_layers[nb_layers-1]);
		
		//##########################################################
		
		#ifdef CUDA
		if(compute_method == C_CUDA)
		{
			cuda_get_table(&output_error, &temp_error, batch_size*output_dim);
			output_error = temp_error;	
			if(saving == 1)
			{
				cuda_get_table(&(net_layers[nb_layers-1]->output), &output_save, batch_size*out_size);
				for(k = 0; k < length; k++)
				{
					for(l = 0; l < out_size; l++)
						fprintf(f_save, "%g ", output_save[k*out_size + l]);
					fprintf(f_save, "\n");
					
				}
			}
			
		}
		#endif
		
		pos = 0;
		for(k = 0; k < length; k++)
		{
			for(l = 0; l < output_dim; l++)
			{
				pos++;
				total_error += output_error[pos];
			}
			
			if(confusion_matrix && compute_method != C_CUDA)
			{
				arg1 = argmax(&(target[k*output_dim]), output_dim);
				arg2 = argmax(&(net_layers[nb_layers-1]->output[k*(output_dim+1)]), output_dim);
				mat[arg1][arg2]++;

			}
		}
		
		if(confusion_matrix && compute_method == C_CUDA)
		{
			#ifdef CUDA
			cuda_confmat(net_layers[nb_layers-1]->output, cuda_mat);
			#endif
		}
	}
	printf("Cumulated error: \t %g\n", total_error/data.size);
	
	}
	if(confusion_matrix && compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_get_table(&cuda_mat, mat, o*o);
		cuda_free_table(cuda_mat);
		#endif
	}
	
	if(saving == 1)
	{
		if(output_save != NULL)
			free(output_save);
	}
	
	if(confusion_matrix)
	{
		printf("\nConfMat (valid set)\n");
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
		for(j = 0; j < o; j++)
			count += mat[j][j];
		
		printf("Correct : %f%%\n", count/data.size*100);
		
		free(temp);
		free(mat);
		free(rapp_err);
		free(rapp_err_rec);
	}
}


void train_network(Dataset train_set, Dataset valid_set, int nb_epochs, int control_interv, real u_learning_rate, real u_momentum, real u_decay)
{
	int i, j, k;
	Dataset shuffle_duplicate;
	real *index_shuffle, *index_shuffle_device;
	
	shuffle_duplicate = create_dataset(train_set.size, 0.1);
	index_shuffle = (real*)  calloc(train_set.size,sizeof(real));
	for(i = 0; i < train_set.size; i++)
		index_shuffle[i] = i;
	#ifdef CUDA
	if(compute_method == C_CUDA)
	{
		index_shuffle_device = (real*)  calloc(train_set.size,sizeof(real));
		cuda_convert_dataset(&shuffle_duplicate);
		cuda_convert_table(&index_shuffle_device, train_set.size);
	}
	#endif
	
	learning_rate = u_learning_rate;
	momentum = u_momentum;
	decay = u_decay;
	
	switch(net_layers[nb_layers-1]->type)
	{
		case CONV:
			out_size = ((conv_param*)net_layers[nb_layers-1]->param)->nb_filters * ((conv_param*)net_layers[nb_layers-1]->param)->nb_area_w * 
				((conv_param*)net_layers[nb_layers-1]->param)->nb_area_h;
			break;
			
		case POOL:
			out_size = ((pool_param*)net_layers[nb_layers-1]->param)->prev_depth * ((pool_param*)net_layers[nb_layers-1]->param)->nb_area_w * 
				((pool_param*)net_layers[nb_layers-1]->param)->nb_area_h;
			break;
	
		case DENSE:
		default:
			out_size = ((dense_param*)net_layers[nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	
	for(i = 0; i < nb_epochs; i++)
	{
		#ifdef CUDA
		cuda_shuffle(train_set, shuffle_duplicate, index_shuffle, index_shuffle_device);
		#endif
		//Loop on all batch for one epoch
		for(j = 0; j < train_set.nb_batch; j++)
		{
			if(j == train_set.nb_batch-1 && train_set.size%batch_size > 0)
				length = train_set.size%batch_size;
			else
				length = batch_size;
			
			input = train_set.input[j];
			target = train_set.target[j];
		
			//printf("input\n");
			//cuda_print_table_transpose(input, batch_size, output_dim+1);
		
			//Forward on all layers
			for(k = 0; k < nb_layers; k++)
				net_layers[k]->forward(net_layers[k]);
			//printf("layer 1\n");
			//cuda_print_table_transpose(((dense_param*)net_layers[0]->param)->weights, ((dense_param*)net_layers[0]->param)->in_size, ((dense_param*)net_layers[0]->param)->nb_neurons+1);
			//cuda_print_table_transpose(net_layers[0]->output, batch_size, ((dense_param*)net_layers[0]->param)->nb_neurons+1);
			//printf("layer 2\n");
			//cuda_print_table_transpose(((dense_param*)net_layers[1]->param)->weights, ((dense_param*)net_layers[1]->param)->in_size, ((dense_param*)net_layers[1]->param)->nb_neurons+1);
			//cuda_print_table_transpose(net_layers[nb_layers-1]->output, batch_size, output_dim+1);
			//printf("targer\n");
			//cuda_print_table_transpose(target, batch_size, output_dim);
			
			output_deriv_error(net_layers[nb_layers-1]);
			//printf("deriv\n");
			//cuda_print_table_transpose(net_layers[nb_layers-1]->delta_o, batch_size, output_dim+1);
			
			//if(j == 2)
			//	exit(1);
			
			//Propagate error through all layers
			for(k = 0; k < nb_layers; k++)
				net_layers[nb_layers-1-k]->backprop(net_layers[nb_layers-1-k]);
		}
		
		if((i+1) % control_interv == 0)
		{
			printf("Epoch: %d\n", i+1);
			compute_error(valid_set, 0, 1);
			printf("\n");
		}
		
		learning_rate *= (1.0-decay);
	}
	
	
	#ifdef CUDA
	if(compute_method == C_CUDA)
	{
		//add cuda free functions
	}
	#endif
	
}


void forward_network(Dataset test_set, int train_step, const char *pers_file_name, int repeat)
{
	char file_name[100];
	
	//update out_size in case of forward with no training
	switch(net_layers[nb_layers-1]->type)
	{
		case CONV:
			out_size = ((conv_param*)net_layers[nb_layers-1]->param)->nb_filters * ((conv_param*)net_layers[nb_layers-1]->param)->nb_area_w * 
				((conv_param*)net_layers[nb_layers-1]->param)->nb_area_h;
			break;
			
		case POOL:
			out_size = ((pool_param*)net_layers[nb_layers-1]->param)->prev_depth * ((pool_param*)net_layers[nb_layers-1]->param)->nb_area_w * 
				((pool_param*)net_layers[nb_layers-1]->param)->nb_area_h;
			break;
	
		case DENSE:
		default:
			out_size = ((dense_param*)net_layers[nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	
	if(train_step >= 0)
	{
	
	}
	else
	{
		if(pers_file_name != NULL)
			strcpy(file_name, pers_file_name);
		else
			sprintf(file_name,"fwd_res/fwd_step_end.txt");
	}
	
	printf("%s\n", file_name);
	f_save = fopen(file_name, "w+");
	
	printf("before compute forward\n");
	compute_error(test_set, 1, repeat);
	printf("after compute forward\n");
	fclose(f_save);
}

/*
void save_network(char *filename)
{
	int i;
	FILE* f;
	
	f = fopen(filename, "w+");
	fprintf("%dx%dx%d ", input_width, input_height, input_depth);
	for(i = 0; i < nb_layers; i++)
	{
		switch(net_layers[i].type)
		{
			case CONV:
				
				
				break;
			
			case POOL:
			
				break;
		
			case DENSE:
			default:
			
				break;
		}
	}
	
	fclose(f);
}
*/













