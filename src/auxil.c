#include "prototypes.h"


void init_network(int network_number, int u_input_dim[3], int u_output_dim, int u_batch_size, int u_compute_method, int u_dynamic_load)
{

	networks[network_number] = (network*) malloc(sizeof(network));
	networks[network_number]->id = network_number;
	networks[network_number]->dynamic_load = u_dynamic_load;
	nb_networks++;
	
	if(!is_init)
	{
		srand(time(NULL));
		#ifdef CUDA
		init_cuda();
		#endif
		is_init = 1;
	}

	networks[network_number]->input_width = u_input_dim[0]; 
	networks[network_number]->input_height = u_input_dim[1];
	networks[network_number]->input_depth = u_input_dim[2];
	networks[network_number]->input_dim = u_input_dim[0]*u_input_dim[1]*u_input_dim[2];
	networks[network_number]->output_dim = u_output_dim;
	networks[network_number]->batch_size = u_batch_size;
	networks[network_number]->compute_method = u_compute_method;
	networks[network_number]->nb_layers = 0;
	networks[network_number]->epoch = 0;
	
	networks[network_number]->output_error = (real*) calloc(networks[network_number]->batch_size*
		networks[network_number]->output_dim,sizeof(real));
	#ifdef CUDA
	if(u_compute_method == C_CUDA)
		cuda_create_table(&networks[network_number]->output_error_cuda, 
			networks[network_number]->batch_size*networks[network_number]->output_dim);
	#endif
}


Dataset create_dataset(network *net, int nb_elem, real bias)
{
	int i,j;
	Dataset data;

	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (real**) malloc(data.nb_batch*sizeof(real*));
	data.target = (real**) malloc(data.nb_batch*sizeof(real*));
	data.localization = HOST;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		data.input[i] = (real*) calloc(net->batch_size * (net->input_dim + 1), sizeof(real));
		data.target[i] = (real*) calloc(net->batch_size * net->output_dim, sizeof(real));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			data.input[i][j*(net->input_dim+1) + net->input_dim] = bias;
		}
	}
	
	return data;
}

void free_dataset(Dataset data)
{
	int i;
		
	if(data.localization == HOST)
	{
		for(i = 0; i < data.nb_batch; i++)
		{
			free(data.input[i]);
			free(data.target[i]);
		}
	}
	#ifdef CUDA
	else
	{
		cuda_free_dataset(&data);
	}
	#endif
	
	free(data.input);
	free(data.target);
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

void save_network(network *net, char *filename)
{
	int i;
	FILE* f;
	
	f = fopen(filename, "w+");
	fprintf(f, "%dx%dx%d\n", net->input_width, net->input_height, net->input_depth);
	for(i = 0; i < net->nb_layers; i++)
	{
		switch(net->net_layers[i]->type)
		{
			case CONV:
				conv_save(f, net->net_layers[i]);
				break;
			
			case POOL:
				pool_save(f, net->net_layers[i]);
				break;
		
			case DENSE:
			default:
				dense_save(f, net->net_layers[i]);
				break;
		}
	}
	
	fclose(f);
}

void load_network(network *net, char *filename, int epoch)
{
	FILE* f = NULL;
	int width, height, depth;
	char layer_type;
	
	net->epoch = epoch;
	
	f = fopen(filename, "r+");
	if(f == NULL)
	{
		printf("ERROR : cannot load/find %s file\n", filename);
		exit(EXIT_FAILURE);
	}
	fscanf(f, "%dx%dx%d\n", &width, &height, &depth);
	
	if(net->input_width != width || net->input_height != height || net->input_depth != depth)
	{
		printf("ERROR : Wrong image format to load the network\nExpect : W = %d, H = %d, D = %d\n",width, height, depth);
		exit(EXIT_FAILURE);
	}
	
	do
	{
		if(fscanf(f, "%c", &layer_type) == EOF)
			break;
		
		switch(layer_type)
		{
			case 'C':
				conv_load(net, f);
				break;
			
			case 'P':
				pool_load(net, f);
				break;
		
			case 'D':
				dense_load(net, f);
				break;
			default:
				break;
		}
	}while(1);
	
	fclose(f);
}

void host_only_shuffle(network *net, Dataset data)
{
	int i, j, k;
	real temp;
	int pos, pos2, batch, batch2;


	for(i = 0; i < data.size - 1; i++)
	{
		j = i + (int)((rand() / ((double)RAND_MAX) ) * (double)(data.size-i));
		pos = i%net->batch_size;
		batch = i/net->batch_size;
		pos2 = j%net->batch_size;
		batch2 = j/net->batch_size;
		
		for(k = 0; k < net->input_dim+1; k++)
		{
			temp = data.input[batch][pos*(net->input_dim + 1) + k];
			data.input[batch][pos*(net->input_dim + 1) + k] = data.input[batch2][pos2*(net->input_dim + 1) + k];
			data.input[batch2][pos2*(net->input_dim + 1) + k] = temp;
		}
		
		for(k = 0; k < net->output_dim; k++)
		{
			temp = data.target[batch][pos*net->output_dim + k];
			
			data.target[batch][pos*net->output_dim + k] = data.target[batch2][pos2*net->output_dim + k];
			data.target[batch2][pos2*net->output_dim + k] = temp;
		}
	}
}



void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat)
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
	FILE *f_save;
	char f_save_name[100];

	o = net->output_dim;
	
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
		if(net->compute_method == C_CUDA)
		{
			cuda_create_table(&cuda_mat, o*o);
		}
		#endif
	}
	
	if(saving == 1)
	{
		output_save = (real*) calloc(net->batch_size*net->out_size,sizeof(real));
		sprintf(f_save_name, "fwd_res/net%d_%04d.dat", net->id, net->epoch);
		f_save = fopen(f_save_name, "w+");
		if(f_save == NULL)
		{
			printf("ERROR : can not oppen %s !\n", f_save_name);
		}
	}
	
	for(r = 0; r < repeat; r++)
	{
		printf("Fwd repeat step: %d /repeat", i, repeat);
		total_error = 0.0;
		
		for(j = 0; j < data.nb_batch; j++)
		{
			if(net->compute_method == C_CUDA)
			{
				temp_error = net->output_error;
				net->output_error = net->output_error_cuda;
			}
			
			//##########################################################
			
			if(j == data.nb_batch - 1 && data.size%net->batch_size > 0)
			{
				net->length = data.size%net->batch_size;
			}
			else
				net->length = net->batch_size;
			
			if(net->dynamic_load)
			{
				cuda_put_table(&net->input, &(data.input[j]), net->batch_size*(net->input_dim+1));
				cuda_put_table(&net->target, &(data.target[j]), net->batch_size*(net->output_dim));
			}
			else
			{
				net->input = data.input[j];
				net->target = data.target[j];
			}
			
			//forward
			for(k = 0; k < net->nb_layers; k++)
			{
				net->net_layers[k]->forward(net->net_layers[k]);
			}
			output_error_fct(net->net_layers[net->nb_layers-1]);
			
			//##########################################################
			
			#ifdef CUDA
			if(net->compute_method == C_CUDA)
			{
				cuda_get_table(&net->output_error, &temp_error, net->batch_size*net->output_dim);
				net->output_error = temp_error;	
				if(saving == 1)
				{
					cuda_get_table(&(net->net_layers[net->nb_layers-1]->output), &output_save, 
						net->batch_size*net->out_size);
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < net->out_size; l++)
							fprintf(f_save, "%g ", output_save[k*net->out_size + l]);
						fprintf(f_save, "\n");
						
					}
				}
				
			}
			#endif
			
			pos = 0;
			for(k = 0; k < net->length; k++)
			{
				for(l = 0; l < net->output_dim; l++)
				{
					pos++;
					total_error += net->output_error[pos];
				}
				
				if(confusion_matrix && net->compute_method != C_CUDA)
				{
					arg1 = argmax(&(net->target[k*net->output_dim]), net->output_dim);
					arg2 = argmax(&(net->net_layers[net->nb_layers-1]->output[k*(net->output_dim+1)]),
						net->output_dim);
					mat[arg1][arg2]++;

				}
			}
			
			if(confusion_matrix && net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_confmat(net, cuda_mat);
				#endif
			}
		}
		printf("Cumulated error: \t %g\n", total_error/data.size);
		
	}
	
	if(confusion_matrix && net->compute_method == C_CUDA)
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
		fclose(f_save);
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


void train_network(network* net, int nb_epochs, int control_interv, real u_begin_learning_rate, real u_end_learning_rate, real u_momentum, real u_decay, int show_confmat, int save_net, int shuffle_gpu)
{
	int i, j, k;
	real begin_learn_rate;
	real end_learn_rate;
	real decay;
	char net_save_file_name[200];
	Dataset shuffle_duplicate;
	real *index_shuffle = NULL, *index_shuffle_device = NULL;
	
	if(net->dynamic_load)
	{
		cuda_create_table(&(net->input), net->batch_size*(net->input_dim+1));
		cuda_create_table(&(net->target), net->batch_size*(net->output_dim));
	}
	else
	{
		shuffle_duplicate = create_dataset(net, net->train.size, 0.1);
		if(shuffle_gpu)
		{
			#ifdef CUDA
			index_shuffle = (real*) calloc(net->train.size,sizeof(real));
			for(i = 0; i < net->train.size; i++)
				index_shuffle[i] = i;
			index_shuffle_device = (real*)  calloc(net->train.size,sizeof(real));
			cuda_convert_dataset(net, &shuffle_duplicate);
			cuda_convert_table(&index_shuffle_device, net->train.size);
			#endif
		}
	}
	
	begin_learn_rate = u_begin_learning_rate;
	end_learn_rate = u_end_learning_rate;
	net->momentum = u_momentum;
	decay = u_decay;
	
	switch(net->net_layers[net->nb_layers-1]->type)
	{
		case CONV:
			net->out_size = ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_filters *
				((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_w * 
				((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_h;
			break;
			
		case POOL:
			net->out_size = ((pool_param*)net->net_layers[net->nb_layers-1]->param)->prev_depth *
				((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_w * 
				((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_h;
			break;
	
		case DENSE:
		default:
			net->out_size = ((dense_param*)net->net_layers[net->nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	
	for(i = 0; i < nb_epochs; i++)
	{
		net->epoch++;
		printf("Epoch: %d\n", net->epoch);
		
		net->learning_rate = end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-decay*i);
		printf("Learning rate : %g\n", net->learning_rate);
		
		
		if(net->dynamic_load)
		{
			host_only_shuffle(net, net->train);
		}
		else
		{
			#ifdef CUDA
			if(shuffle_gpu)
				cuda_shuffle(net, net->train, shuffle_duplicate, index_shuffle, index_shuffle_device);
			else
				host_shuffle(net, net->train, shuffle_duplicate);
			#endif
		}
		
		
		//Loop on all batch for one epoch
		for(j = 0; j < net->train.nb_batch; j++)
		{
			if(j == net->train.nb_batch-1 && net->train.size%net->batch_size > 0)
				net->length = net->train.size%net->batch_size;
			else
				net->length = net->batch_size;
			
			if(net->dynamic_load)
			{
				cuda_put_table(&net->input, &(net->train.input[j]), net->batch_size*(net->input_dim+1));
				cuda_put_table(&net->target, &(net->train.target[j]), net->batch_size*(net->output_dim));
			}
			else
			{
				net->input = net->train.input[j];
				net->target = net->train.target[j];
			}
		
			for(k = 0; k < net->nb_layers; k++)
				net->net_layers[k]->forward(net->net_layers[k]);
			
			output_deriv_error(net->net_layers[net->nb_layers-1]);
			
			//Propagate error through all layers
			for(k = 0; k < net->nb_layers; k++)
				net->net_layers[net->nb_layers-1-k]->backprop(net->net_layers[net->nb_layers-1-k]);
		}
		
		if(((net->epoch) % control_interv == 0) || (i == nb_epochs - 1))
		{
			compute_error(net, net->valid, 0, show_confmat, 1);
			printf("\n");
		}
		if(save_net > 0 && ((net->epoch) % save_net) == 0)
		{
			sprintf(net_save_file_name, "net_save/net%d_s%04d.dat", net->id, net->epoch);
			printf("Saving network for epoch: %d\n", net->epoch);
			save_network(net, net_save_file_name);
		}
		
	}
	
	#ifdef CUDA
	if(net->dynamic_load)
	{
		cuda_free_table(net->input);
		cuda_free_table(net->target);
	}
	else if(shuffle_gpu)
	{
		cuda_free_dataset(&shuffle_duplicate);
		free(index_shuffle);
	}
	else
	{
		free_dataset(shuffle_duplicate);
	}
	if(net->compute_method == C_CUDA)
		cuda_free_table(index_shuffle_device);
	#endif
}



void forward_testset(network *net, int train_step, int repeat)
{
	//update out_size in case of forward with no training
	
	switch(net->net_layers[net->nb_layers-1]->type)
	{
		case CONV:
			net->out_size = ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_filters 
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_w 
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_h;
			break;
			
		case POOL:
			net->out_size = ((pool_param*)net->net_layers[net->nb_layers-1]->param)->prev_depth 
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_w 
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area_h;
			break;
	
		case DENSE:
		default:
			net->out_size = ((dense_param*)net->net_layers[net->nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	if(net->dynamic_load)
	{
		cuda_create_table(&(net->input), net->batch_size*(net->input_dim+1));
		cuda_create_table(&(net->target), net->batch_size*(net->output_dim));
	}
	
	if(train_step <= 0)
	{
		printf("Warning, can not forward %d step\n", train_step);
	}
	else
	{
		printf("before compute forward\n");
		compute_error(net, net->test, 1, 0, repeat);
		printf("after compute forward\n");
	}
	
	if(net->dynamic_load)
	{
		cuda_free_table(net->input);
		cuda_free_table(net->target);
	}
}
















