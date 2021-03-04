
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

void init_timing(struct timeval* tstart)
{
    gettimeofday(tstart, NULL);
}

float ellapsed_time(struct timeval tstart)
{
    struct timeval tmp;
    long long diff;
    gettimeofday(&tmp, NULL);
    diff = tmp.tv_usec - tstart.tv_usec;
    diff += (tmp.tv_sec - tstart.tv_sec) * 1000000;
    return ((float)diff*1.0e-6);
}

void init_network(int network_number, int u_input_dim[3], int u_output_dim, float in_bias, int u_batch_size, int u_compute_method, int u_dynamic_load, int u_use_cuda_TC)
{

	printf("############################################################\n\
CIANNA V-0.9.2.2 EXPERIMENTAL BUILD (03/2021), by D.Cornu\n\
############################################################\n\n");


	network *net;

	net = (network*) malloc(sizeof(network));
	networks[network_number] = net;
	
	net->id = network_number;
	net->dynamic_load = u_dynamic_load;
	net->use_cuda_TC = u_use_cuda_TC;
	
	//Additional security, but all statements should be safe on its own
	//note that FP32/FP32 Tensore Core operation exist (different than TF32)
	if(u_compute_method != C_CUDA)
		networks[network_number]->use_cuda_TC = 0;
	
	nb_networks++;
	
	if(!is_init)
	{
		srand(time(NULL));
		#ifdef CUDA
		if(u_compute_method == C_CUDA)
			init_cuda(networks[network_number]);
		#endif
		
		#ifndef CUDA
		if(u_compute_method == C_CUDA)
		{
			printf("ERROR: compute method set to CUDA while CIANNA was not compiled for it.\n");
			printf("Install Nvidia CUDA and recompile CIANNA with the appropriate option.\n\n");
			exit(EXIT_FAILURE);
		}
		#endif
		
		#ifndef BLAS
		if(u_compute_method == C_BLAS)
		{
			printf("ERROR: compute method set to BLAS while CIANNA was not compiled for it.\n");
			printf("Install OpenBLAS and recompile CIANNA with the appropriate option.\n\n");
			exit(EXIT_FAILURE);
		}
		#endif
		if(u_compute_method == C_NAIV)
		{
			printf("WARNING: compute method set to NAIV, which is not optimal.\n");
			printf("We recommand the use of OpenBLAS for a better usage of CPU ressources.\n");
			printf("If NAIV with single CPU thread is your only option, we recommand the use of the SGD learning scheme, enabled by setting the batch size to 1.\n\n");
		}
		is_init = 1;
	}

	net->input_width = u_input_dim[0]; 
	net->input_height = u_input_dim[1];
	net->input_depth = u_input_dim[2];
	net->input_dim = u_input_dim[0]*u_input_dim[1]*u_input_dim[2];
	net->output_dim = u_output_dim;
	
	net->input_bias = in_bias;
	if(u_batch_size > 1)
	{
		net->batch_size = u_batch_size;
		net->batch_param = OFF;
	}
	else if(u_batch_size == 1)
	{
		net->batch_size = 1;
		net->batch_param = SGD;
		printf("Automatically switch to SGD scheme (batch_size = 1)\n");
	}
	else if(u_batch_size <= 0)
	{
		net->batch_size = 256;
		net->batch_param = FULL;
		printf("Undefined batch size -> automatic value is 256\n");
	}
	net->compute_method = u_compute_method;
	net->nb_layers = 0;
	net->epoch = 0;
	net->norm_factor_defined = 0;
	
	//YOLO null setting
	net->yolo_nb_box = 0;
	net->yolo_prior_w = NULL;
	net->yolo_prior_h = NULL;
	net->yolo_nb_class = 0;
	net->yolo_nb_param = 0;

}


Dataset create_dataset(network *net, int nb_elem)
{
	int i,j;
	Dataset data;
	
	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (void**) malloc(data.nb_batch*sizeof(float*));
	data.target = (void**) malloc(data.nb_batch*sizeof(float*));
	data.localization = HOST;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		((float**)data.input)[i] = (float*) calloc(net->batch_size * (net->input_dim + 1), sizeof(float));
		((float**)data.target)[i] = (float*) calloc(net->batch_size * net->output_dim, sizeof(float));
	}
	
	for(i = 0; i < data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			((float**)data.input)[i][j*(net->input_dim+1) + net->input_dim] = net->input_bias;
		}
	}
	
	return data;
}

void print_table(float* tab, int column_size, int nb_column)
{
	int i, j;
	
	for(i = 0; i < nb_column; i++)
	{
		for(j = 0; j < column_size; j++)
		{
			printf("%f ", tab[i*column_size+j]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_epoch_advance(int c_batch, int nb_batch, float loss)
{
	int i;
	int size = 60;
	
	printf("\e[?25l");
	printf("\r[");
	for(i = 0; i < size*((float)c_batch/nb_batch); i++)
		printf("#");
	for(i = size*((float)c_batch/nb_batch); i < size; i++)
		printf("-");
	printf("] %d/%d    Loss: %f", c_batch, nb_batch, loss);
	printf("\e[?25h");
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


void write_formated_dataset(network *net, const char *filename, Dataset *data, int input_data_type, int output_data_type)
{
	// Create and load a dataset from a format specific file

	FILE *f = NULL;
	f = fopen(filename, "wb"); 
	int i, j, k;
	int datasize;
	
	fwrite(&data->size, sizeof(int), 1, f);
	fwrite(&net->input_width, sizeof(int), 1, f);
	fwrite(&net->input_height, sizeof(int), 1, f);
	fwrite(&net->input_depth, sizeof(int), 1, f);
	fwrite(&net->output_dim, sizeof(int), 1, f);
	
	// Should rework this function to avoid repetions, try to use a void pointer that is 
	// properly converted and try to get function pointer to type cast 
	
	switch(input_data_type)
	{
		case c_UINT8:
		{
			unsigned char *temp_input;
			datasize = sizeof(unsigned char);
			temp_input = (unsigned char *) calloc(net->input_dim, datasize);
			
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_input[k] = (unsigned char) ((float**)data->input)[i][j*(net->input_dim+1) + k];
					fwrite(temp_input, datasize, net->input_dim, f);
				}
			}
			free(temp_input);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_input;
			datasize = sizeof(unsigned short);
			temp_input = (unsigned short *) calloc(net->input_dim, datasize);
			
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_input[k] = (unsigned short) ((float**)data->input)[i][j*(net->input_dim+1) + k];
					fwrite(temp_input, datasize, net->input_dim, f);
				}
			}
			free(temp_input);
			break;
		}
		case c_FP32:
		default:
		{
			float *temp_input;
			datasize = sizeof(float);
			temp_input = (float *) calloc(net->input_dim, datasize);
			
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_input[k] = (float) ((float**)data->input)[i][j*(net->input_dim+1) + k];
					fwrite(temp_input, datasize, net->input_dim, f);
				}
			}
			free(temp_input);
			break;
		}
	}
	
	
	switch(output_data_type)
	{
		case c_UINT8:
		{
			unsigned char *temp_output;
			datasize = sizeof(unsigned char);
			temp_output = (unsigned char *) calloc(net->output_dim, datasize);
			
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_output[k] = (unsigned char) ((float**)data->target)[i][j*(net->output_dim) + k];
					fwrite(temp_output, datasize, net->output_dim, f);
				}
			}
			free(temp_output);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_output;
			datasize = sizeof(unsigned short);
			temp_output = (unsigned short *) calloc(net->output_dim, datasize);
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_output[k] = (unsigned short) ((float**)data->target)[i][j*(net->output_dim) + k];
					fwrite(temp_output, datasize, net->output_dim, f);
				}
			}
			free(temp_output);
			break;
		}
			
		case c_FP32:
		default:
		{
			float *temp_output;
			datasize = sizeof(float);
			temp_output = (float *) calloc(net->output_dim, datasize);
			break;
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= data->size)
						continue;
					for(k = 0; k < net->input_dim; k++)
						temp_output[k] = (float) ((float**)data->target)[i][j*(net->output_dim) + k];
					fwrite(temp_output, datasize, net->output_dim, f);
				}
			}
			free(temp_output);
		}
	}
	
	fclose(f);
	
}




Dataset load_formated_dataset(network *net, const char *filename, int input_data_type, int output_data_type)
{
	// Create and load a dataset from a format specific file
	Dataset data;

	FILE *f = NULL;
	f = fopen(filename, "rb"); 
	int size, width, height, depth, out_dim;
	int i, j, k;
	int datasize;
	
	if(f == NULL)
	{
		printf("ERROR : file %s does not exist !", filename);
		exit(EXIT_FAILURE);
	}
	
	fread(&size, sizeof(int), 1, f);
	fread(&width, sizeof(int), 1, f);
	fread(&height, sizeof(int), 1, f);
	fread(&depth, sizeof(int), 1, f);
	fread(&out_dim, sizeof(int), 1, f);
	
	
	if( width * height * depth != net->input_dim || out_dim != net->output_dim)
	{
		printf("\nERROR : input dimensions do not match in file %s !\n", filename);
		printf("File dimensions are, size: %d, input dimensions : %dx%dx%d, output dimension : %d\n"
					, size, width, height, depth, out_dim);
		exit(EXIT_FAILURE);
	}
	
	data = create_dataset(net, size);
	
	
	switch(input_data_type)
	{
		case c_UINT8:
		{
			unsigned char *temp_input;
			datasize = sizeof(unsigned char);
			temp_input = (unsigned char *) calloc(net->input_dim, datasize);
			
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_input, datasize, net->input_dim, f);
					for(k = 0; k < net->input_dim; k++)
						((float**)data.input)[i][j*(net->input_dim+1) + k] = (float) temp_input[k];
				}	
			}
			free(temp_input);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_input;
			datasize = sizeof(unsigned short);
			temp_input = (unsigned short *) calloc(net->input_dim, datasize);
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_input, datasize, net->input_dim, f);
					for(k = 0; k < net->input_dim; k++)
						((float**)data.input)[i][j*(net->input_dim+1) + k] = (float) temp_input[k];
				}	
			}
			free(temp_input);
			break;
		}
			
		case c_FP32:
		default:
		{
			float *temp_input;
			datasize = sizeof(float);
			temp_input = (float *) calloc(net->input_dim, datasize);
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_input, datasize, net->input_dim, f);
					for(k = 0; k < net->input_dim; k++)
						((float**)data.input)[i][j*(net->input_dim+1) + k] = (float) temp_input[k];
				}	
			}
			free(temp_input);
			break;
		}
	}
	
		
	switch(output_data_type)
	{
		case c_UINT8:
		{
			unsigned char *temp_output;
			datasize = sizeof(unsigned char);
			temp_output = (unsigned char *) calloc(net->output_dim, datasize);
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_output, datasize, net->output_dim, f);
					for(k = 0; k < net->output_dim; k++)
						((float**)data.target)[i][j*(net->output_dim) + k] = (float) temp_output[k];
				}
			}
			free(temp_output);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_output;
			datasize = sizeof(unsigned short);
			temp_output = (unsigned short *) calloc(net->output_dim, datasize);
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_output, datasize, net->output_dim, f);
					for(k = 0; k < net->output_dim; k++)
						((float**)data.target)[i][j*(net->output_dim) + k] = (float) temp_output[k];
				}
			}
			free(temp_output);
			break;
		}
			
		case c_FP32:
		default:
		{
			float *temp_output;
			datasize = sizeof(float);
			temp_output = (float *) calloc(net->output_dim, datasize);
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_output, datasize, net->output_dim, f);
					for(k = 0; k < net->output_dim; k++)
						((float**)data.target)[i][j*(net->output_dim) + k] = (float) temp_output[k];
				}
			}
			free(temp_output);
			break;
		}
	}
	
	fclose(f);
	
	return data;
}

void set_normalize_dataset_parameters(network *net, float *offset_input, float *norm_input, int dim_size_input, float *offset_output, float *norm_output, int dim_size_output)
{
	net->norm_factor_defined = 1;

 	net->offset_input = offset_input;
 	net->offset_output = offset_output;
	net->norm_input = norm_input;
	net->norm_output = norm_output;
	net->dim_size_input = dim_size_input;
	net->dim_size_output = dim_size_output;
}

void normalize_dataset(network *net, Dataset c_data)
{
	int i, j, k, l;
	
	int nb_dim_in = net->input_dim / net->dim_size_input;
	int nb_dim_out = net->output_dim / net->dim_size_output;
	
	if(net->norm_factor_defined != 1)
		return;
	
	for(i = 0; i < c_data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= c_data.size)
				continue;
			for(k = 0; k < nb_dim_in; k++)
			{
				for(l = 0; l < net->dim_size_input; l++)
				{
					((float**)c_data.input)[i][j*(net->input_dim+1) + k*net->dim_size_input + l] 
						+= net->offset_input[k];
					((float**)c_data.input)[i][j*(net->input_dim+1) + k*net->dim_size_input + l] 
						/= net->norm_input[k];
				}
			}
		}
	}
	
	for(i = 0; i < c_data.nb_batch; i++)
	{
		for(j = 0; j < net->batch_size; j++)
		{
			if(i*net->batch_size + j >= c_data.size)
				continue;
			for(k = 0; k < nb_dim_out; k++)
			{
				for(l = 0; l < net->dim_size_output; l++)
				{
					((float**)c_data.target)[i][j*(net->output_dim) + k*net->dim_size_output + l] 
						+= net->offset_output[k];
					((float**)c_data.target)[i][j*(net->output_dim) + k*net->dim_size_output + l] 
						/= net->norm_output[k];
				}
			}
		}
	}
	
	
}


int argmax(float *tab, int size)
{
	int i;
	float max;
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
	net->nb_layers = 0;
	
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
	float temp;
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
			temp = ((float**)data.input)[batch][pos*(net->input_dim + 1) + k];
			((float**)data.input)[batch][pos*(net->input_dim + 1) + k] = ((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k];
			((float**)data.input)[batch2][pos2*(net->input_dim + 1) + k] = temp;
		}
		
		for(k = 0; k < net->output_dim; k++)
		{
			temp = ((float**)data.target)[batch][pos*net->output_dim + k];
			
			((float**)data.target)[batch][pos*net->output_dim + k] = ((float**)data.target)[batch2][pos2*net->output_dim + k];
			((float**)data.target)[batch2][pos2*net->output_dim + k] = temp;
		}
	}
}



void update_weights(void *weights, void* update, int size)
{
	int i;
	
	float* f_weights = (float*) weights;
	float* f_update = (float*) update;
	
	for(i = 0; i < size; i++)
		f_weights[i] -= f_update[i];
}



void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat)
{
	int j, k, l, m, r;
	float** mat = NULL; 
	float* temp = NULL;
	int arg1, arg2;
	float count;
	float *rapp_err = NULL, *rapp_err_rec = NULL;
	int o, pos;
	float total_error;
	float batch_error = 0.0;
	void* output_save = NULL;
	void* output_buffer = NULL;
	struct timeval ep_timer;
	float items_per_s = 0.0;
	FILE *f_save = NULL;
	char f_save_name[100];
	conv_param *c_param;
	
	#ifdef CUDA
	float* cuda_mat;
	void* temp_error = NULL;
	#endif

	//FILE *f_err;
	
	o = net->output_dim;
	
	if(confusion_matrix)
	{
		printf("Confusion matrix load\n");
		rapp_err = (float*) malloc(o*sizeof(float));
		rapp_err_rec = (float*) malloc(o*sizeof(float));
		mat = (float**) malloc(o*sizeof(float*));
		temp = (float*) calloc(o*o,sizeof(float));
		for(j = 0; j < o; j++)
			mat[j] = &(temp[j*o]);
		
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
		{
			cuda_create_table_FP32(net, &cuda_mat, o*o);
		}
		#endif
	}	
	
	if(saving == 1)
	{
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
			output_save = (float*) calloc(net->batch_size*net->out_size, sizeof(float));
		if(net->compute_method == C_CUDA && net->use_cuda_TC)
			cuda_create_host_table_FP16(net, &output_buffer, net->batch_size*net->out_size);
		#endif
		sprintf(f_save_name, "fwd_res/net%d_%04d.dat", net->id, net->epoch);
		f_save = fopen(f_save_name, "w+");
		if(f_save == NULL)
		{
			printf("ERROR : can not oppen %s !\n", f_save_name);
		}
	}
	
	for(r = 0; r < repeat; r++)
	{
		init_timing(&ep_timer);
		if(repeat > 1)
			printf("Forward repeat step: %d /%d\n", r+1, repeat);
		total_error = 0.0;
		
		if(repeat <= 1)
			printf("\nForward: %d\n", net->epoch);
		for(j = 0; j < data.nb_batch; j++)
		{
		
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				temp_error = net->output_error;
				net->output_error = net->output_error_cuda;
				#endif
			}
			
			//##########################################################
			
			if(j == data.nb_batch - 1 && data.size%net->batch_size > 0)
			{
				net->length = data.size%net->batch_size;
			}
			else
				net->length = net->batch_size;
			
			if(net->compute_method == C_CUDA && net->dynamic_load)
			{
				#ifdef CUDA
				cuda_put_table(net, net->input, data.input[j], net->batch_size*(net->input_dim+1));
				cuda_put_table(net, net->target, data.target[j], net->batch_size*(net->output_dim));
				#endif
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
			
			output_error(net->net_layers[net->nb_layers-1]);
				
			//##########################################################
			
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(net, net->output_error, temp_error, net->batch_size*net->out_size);
				net->output_error = temp_error;	
				if(saving == 1)
				{
					switch(net->use_cuda_TC)
					{
						default:
						case 0:
							cuda_get_table(net, net->net_layers[net->nb_layers-1]->output,
								output_save, net->batch_size*net->out_size);
							break;
						case 1:
							cuda_get_table_FP16_to_FP32(net->net_layers[net->nb_layers-1]->output,
								output_save, net->batch_size*net->out_size, output_buffer);
							break;
					}
					
					switch(net->net_layers[net->nb_layers-1]->type)
					{
						default:
						case DENSE:
							for(k = 0; k < net->length; k++)
							{
								for(l = 0; l < net->out_size; l++)
									fprintf(f_save, "%g ", ((float*)output_save)[k*net->out_size + l]);
								fprintf(f_save, "\n");
							}
							break;
						case CONV:
							c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
							int batch_offset = c_param->nb_area_w*c_param->nb_area_h;
							int filter_offset = batch_offset*net->batch_size;
							for(k = 0; k < net->length; k++)
							{
								for(l = 0; l < c_param->nb_filters; l++)
								{
									for(m = 0; m < batch_offset; m++)
										fprintf(f_save,"%g ", ((float*)output_save)[k*batch_offset + l*filter_offset + m]);
								}
								fprintf(f_save, "\n");
							}
							break;
					}
				}
				#endif
			}
			else
			{
				if(saving == 1)
				{
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < net->out_size; l++)
							fprintf(f_save, "%g ", ((float*)net->net_layers[net->nb_layers-1]->output)[k*net->out_size + l]);
						fprintf(f_save, "\n");
					}
				}
			}
			
			pos = 0;
			batch_error = 0;
			switch(net->net_layers[net->nb_layers-1]->type)
			{
				default:
				case DENSE:
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < net->out_size; l++)
						{
							pos++;
							batch_error += ((float*)net->output_error)[pos];
							total_error += ((float*)net->output_error)[pos];
						}
						
						if(confusion_matrix && net->compute_method != C_CUDA)
						{
							arg1 = argmax(&(((float*)net->target)[k*net->output_dim]), net->output_dim);
							arg2 = argmax(&(((float*)net->net_layers[net->nb_layers-1]->output)[k*(net->output_dim+1)]),
								net->output_dim);
							mat[arg1][arg2]++;
						}
					}
					break;
				case CONV:
					c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
					int batch_offset = c_param->nb_area_w*c_param->nb_area_h;
					int filter_offset = batch_offset*net->batch_size;
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < c_param->nb_filters; l++)
						{
							for(m = 0; m < batch_offset; m++)
							{
								batch_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
								total_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
							}
						}
					}
					break;
			}
			batch_error /= net->length;
			print_epoch_advance(j+1, data.nb_batch, batch_error);

			if(confusion_matrix && net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_confmat(net, cuda_mat);
				#endif
			}
		}
		
		//f_err = fopen("error.txt", "a");
		
		items_per_s = data.size/ellapsed_time(ep_timer);
		printf("\nNet. forward perf.: %0.2f items/s\n", items_per_s);
		printf("Cumulated error: \t %g\n", total_error/data.size);
		
		//fprintf(f_err, "%g\n",  total_error/data.size);
		//fclose(f_err);
		
		
	}
	
	if(confusion_matrix && net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_get_table_FP32(net, (float*)cuda_mat, *mat, o*o);
		cuda_free_table(cuda_mat);
		#endif
	}
	
	if(saving == 1)
	{
		if(output_save != NULL)
			free(output_save);
		if(output_buffer != NULL)
			free(output_buffer);
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


void train_network(network* net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, float u_decay, int show_confmat, int save_net, int shuffle_gpu, int shuffle_every)
{
	int i, j, k, l, m;
	float begin_learn_rate;
	float end_learn_rate;
	float decay;
	float batch_error = 0.0;
	char net_save_file_name[200];
	struct timeval ep_timer;
	float items_per_s = 0.0;
	int batch_loc;
	int pos;
	conv_param *c_param;
	
	#ifdef CUDA
	Dataset shuffle_duplicate;
	void* temp_error = NULL;
	int *index_shuffle = NULL, *index_shuffle_device = NULL;
	
	if(net->compute_method == C_CUDA)
	{
		if(net->dynamic_load)
		{
			cuda_create_table(net, &(net->input), net->batch_size*(net->input_dim+1));
			cuda_create_table(net, &(net->target), net->batch_size*(net->output_dim));
		}
		else
		{
			shuffle_duplicate = create_dataset(net, net->train.size);
			if(shuffle_gpu)
			{
				
				index_shuffle = (void*) calloc(net->train.size,sizeof(int));
				for(i = 0; i < net->train.size; i++)
					index_shuffle[i] = i;
				index_shuffle_device = (void*)  calloc(net->train.size,sizeof(int));
				cuda_convert_dataset(net, &shuffle_duplicate);
				cuda_convert_table_int(net, &index_shuffle_device, net->train.size);
			}
		}
		
	}
	#endif
	
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
	
	net->output_error = (float*) calloc(net->batch_size * net->out_size, sizeof(float));
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_create_table_FP32(net, (float**)&net->output_error_cuda, net->batch_size * net->out_size);
		#endif
	}
	
	
	for(i = 0; i < nb_epochs; i++)
	{
		net->learning_rate = end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-decay*net->epoch);
		net->epoch++;
	
		if((net->epoch) % shuffle_every == 0 && net->batch_param != SGD)
		{
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(net->dynamic_load)
					cuda_host_only_shuffle(net, net->train);
				else
				{
					if(shuffle_gpu)
						cuda_shuffle(net, net->train, shuffle_duplicate, index_shuffle, index_shuffle_device);
					else
						host_shuffle(net, net->train, shuffle_duplicate);
				}
				#endif
			}
			else
				host_only_shuffle(net, net->train);
			
		}
		
		init_timing(&ep_timer);
		
		//Loop on all batch for one epoch
		printf("\nEpoch: %d\n", net->epoch);
		for(j = 0; j < net->train.nb_batch; j++)
		{
		
			if(j == net->train.nb_batch-1 && net->train.size%net->batch_size > 0)
				net->length = net->train.size%net->batch_size;
			else
				net->length = net->batch_size;

			if(net->batch_param != SGD)
				batch_loc = j;
			else
				batch_loc = random_uniform() * net->train.size;
				
			if(net->dynamic_load && net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_put_table(net, net->input, net->train.input[batch_loc], net->batch_size*(net->input_dim+1));
				cuda_put_table(net, net->target, net->train.target[batch_loc], net->batch_size*(net->output_dim));
				#endif
			}
			else
			{
				net->input = net->train.input[batch_loc];
				net->target = net->train.target[batch_loc];
			}
			
			for(k = 0; k < net->nb_layers; k++)
			{
				net->net_layers[k]->forward(net->net_layers[k]);
			}
			
			output_deriv_error(net->net_layers[net->nb_layers-1]);
			
			//Propagate error through all layers
			for(k = 0; k < net->nb_layers; k++)
			{
				net->net_layers[net->nb_layers-1-k]->backprop(net->net_layers[net->nb_layers-1-k]);
			}
			
			
			// Live loss monitoring
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				temp_error = net->output_error;
				net->output_error = net->output_error_cuda;
				#endif
			}
			
			output_error(net->net_layers[net->nb_layers-1]);
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(net, net->output_error, temp_error, net->batch_size*net->out_size);
				net->output_error = temp_error;	
				#endif
			}
			pos = 0;
			batch_error = 0;
			switch(net->net_layers[net->nb_layers-1]->type)
			{
				default:
				case DENSE:
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < net->out_size; l++)
						{
							pos++;
							batch_error += ((float*)net->output_error)[pos];
						}
					}
					break;
				case CONV:
					c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
					int batch_offset = c_param->nb_area_w*c_param->nb_area_h;
					int filter_offset = batch_offset*net->batch_size;
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < c_param->nb_filters; l++)
						{
							for(m = 0; m < batch_offset; m++)
								batch_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
						}
					}
					break;
			}
			batch_error /= net->length;
			print_epoch_advance(j+1, net->train.nb_batch, batch_error);

		}
		
		items_per_s = net->train.size/ellapsed_time(ep_timer); 
		
		if(((net->epoch) % control_interv == 0) || (i == nb_epochs - 1))
		{
			printf("\nControl step epoch: %d\n", net->epoch);
			printf("Net. training perf.: %0.2f items/s\n", items_per_s);
			printf("Learning rate : %g\n", net->learning_rate);
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
	
	free(net->output_error);
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		cuda_free_table(net->output_error_cuda);
		if(net->dynamic_load)
		{
			cuda_free_table(net->input);
			cuda_free_table(net->target);
		}
		else if(shuffle_gpu)
		{
			cuda_free_dataset(&shuffle_duplicate);
			cuda_free_table(index_shuffle_device);
			free(index_shuffle);
		}
		else
		{
			free_dataset(shuffle_duplicate);
		}	
	}
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
	
	net->output_error = (float*) calloc(net->batch_size * net->out_size, sizeof(float));
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_create_table_FP32(net, (float**)&net->output_error_cuda, net->batch_size * net->out_size);
		#endif
	}
	
	
	if(net->dynamic_load)
	{
		#ifdef CUDA
		cuda_create_table(net, &(net->input), net->batch_size*(net->input_dim+1));
		cuda_create_table(net, &(net->target), net->batch_size*(net->output_dim));
		#endif
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
	
	free(net->output_error);
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_free_table(net->output_error_cuda);
		#endif	
	}
	
	if(net->dynamic_load)
	{
		#ifdef CUDA
		cuda_free_table(net->input);
		cuda_free_table(net->target);
		#endif
	}
}
















