
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

struct timeval t_perf_eval;

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

void sig_handler(int signo)
{
	if (signo == SIGINT)
		printf("\nWarning: program interrupted\n");
	//Could handle exit more gracefully by freeing everything (postponed)
	exit(EXIT_SUCCESS);
}

void init_network(int network_number, int u_input_dim[4], int u_output_dim, float in_bias, int u_batch_size, int u_compute_method, int u_dynamic_load, int u_use_cuda_TC, int no_logo)
{

	if(!is_init)
	{
	signal(SIGINT, sig_handler);
	
	if(!no_logo)
		printf("\n                   ..:^~!?JY5PB~                                                                                             \n\
           J5PGB#&&&&&#BGP55YY#&#!                                                                                           \n\
           &GGB##&&&@@@@@&#PJ?B#&B                                                                                           \n\
          .&#&@@@@@@@@@@@@@@@B##&G                                                                                           \n\
        ^G@@@@BJ^.   .~?P&@@@@@&&G                                                                                           \n\
      !&@@@B^   :!???^^..:7&@@@@&G      .~~~         :~~:         ~~~^        ~~~    ~~~^        ~~~         ^~~.            \n\
    .B@@@&^  !#@@@@@@@@&?^~J&@@@@B ^    !@@@.       :@@@@^       .@@@@G      .@@@:  .@@@@P      .@@@:       ~@@@@.           \n\
   .&@@@#  .J@@@#GP55PB&@&JJ5@@@@B ^G   !@@@.       &@@@@@.      .@@@@@&:     @@@:  .@@@@@&:    .@@@.      :@@@@@&           \n\
   #@@@@.  ?@@&Y?JJJJJJJP@5JY@@@@B ^@:  !@@@       #@@PY@@&      .@@@P@@@7    @@@:  .@@@P@@@7   .@@@.      &@@JG@@B          \n\
  :@@@@#  .@@@YJJJJJJJJJJGY?#@@@@B 5@!  !@@@      P@@#  G@@B     .@@@.:&@@G   @@@:  .@@@.:&@@P  .@@@.     #@@P  &@@5         \n\
  ~@Y&@#  .@@@P?JJJJJJJJJ?J#@@@@@G.@@~  !@@@     ?@@@:  .&@@5    .@@@.  G@@&. @@@:  .@@@.  G@@&. @@@.    P@@&.  :@@@7        \n\
  .@:#@@.  B@@@P5JJJJJJJ5B@@@@@@&#&@&   !@@@    ~@@@@@@@@@@@@7   .@@@.   ?@@@J@@@:  .@@@.   ?@@@J@@@.   ?@@@@@@@@@@@@^       \n\
   5^7@@#   G@@@&&&###&@@@@@@&&B@@@@^   !@@@   .@@@Y??????J@@@^  .@@@.    :&@@@@@:  .@@@.    :&@@@@@.  ~@@@J??????5@@@.      \n\
    : B@@&~  ^B@@@@@@@@@@&GPYJ#@@@@^    !@@@. .@@@J        7@@@: .@@@:      G@@@@:  .@@@.      G@@@@: :@@@!        5@@&.     \n\
       G@@@&?. .~5B&GJ7^:::7B@@@@&      .YJ?  ^YJ7          !JY~  ?JJ        !YJJ    JJJ        !YJJ  !YY!          ?JY^     \n\
        7&@@@@&G?~^^~!Y5G&@@@@@&&G                                                                                           \n\
          ?@@@@@@@@@@@@@@@&#GY##&G                                                                                           \n\
           #PB&@@@@&&##BGP5J??B#&B                                                                                           \n\
           Y55PGB##&&&#BGPP55Y#&#!                                                                                           \n\
                  ...:^~!?JY5PB~                                                                                             \n\n");

	printf("############################################################\n\
CIANNA V-0.9.2.8 EXPERIMENTAL BUILD (09/2021), by D.Cornu\n\
############################################################\n\n");
	
	}
	network *net;

	net = (network*) malloc(sizeof(network));
	networks[network_number] = net;
	
	net->id = network_number;
	
	#ifdef CUDA
	net->cu_inst.dynamic_load = u_dynamic_load;
	net->cu_inst.use_cuda_TC= u_use_cuda_TC;
	
	//Additional security, but all call to use_cuda_TC should be safe on its own
	if(u_compute_method != C_CUDA)
		networks[network_number]->cu_inst.use_cuda_TC = FP32C_FP32A;
	#endif
	
	nb_networks++;
	
	
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

	net->in_dims[0] = u_input_dim[0]; 
	net->in_dims[1] = u_input_dim[1];
	net->in_dims[2] = u_input_dim[2];
	net->in_dims[3] = u_input_dim[3];
	net->input_dim = u_input_dim[0]*u_input_dim[1]*u_input_dim[2]*u_input_dim[3];
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
		net->batch_size = 16;
		net->batch_param = FULL;
		printf("Undefined batch size -> automatic value is 16\n");
	}
	net->compute_method = u_compute_method;
	net->nb_layers = 0;
	net->epoch = 0;
	net->norm_factor_defined = 0; //depreciated
	net->is_inference = 0;
	net->inference_drop_mode = AVG_MODEL;
	net->no_error = 0;
	net->perf_eval = 1;
	
	net->train_buf.localization = NO_LOC;
	net->test_buf.localization = NO_LOC;
	net->valid_buf.localization = NO_LOC;
	
	//YOLO null setting
	net->y_param = (yolo_param*) malloc(sizeof(yolo_param));
	net->y_param->nb_box = 0;
	net->y_param->prior_w = NULL;
	net->y_param->prior_h = NULL;
	net->y_param->prior_d = NULL;
	net->y_param->IoU_type = IOU;
	net->y_param->strict_box_size_association = 0;
	net->y_param->c_IoU_fct = NULL;
	net->y_param->noobj_prob_prior = NULL;
	net->y_param->scale_tab = NULL;
	net->y_param->slopes_and_maxes_tab = NULL;
	net->y_param->IoU_limits = NULL;
	net->y_param->fit_parts = NULL;
	net->y_param->nb_class = 0;
	net->y_param->nb_param = 0;

}

void copy_to_host(float* in_tab, void* out_tab, int out_offset, int size)
{
	float* f_out_tab = (float*) out_tab;
	for(int i = 0; i < size; i++)
		*(f_out_tab + out_offset + i) = (*(in_tab + i));
}

Dataset create_dataset_host(network *net, int nb_elem)
{
	int i,j;
	Dataset data;
	
	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / net->batch_size + 1;
	data.input = (void**) malloc(data.nb_batch*sizeof(float*));
	data.target = (void**) malloc(data.nb_batch*sizeof(float*));
	data.localization = HOST;
	data.cont_copy = copy_to_host;
	
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


Dataset create_dataset(network *net, int nb_elem)
{
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		return cuda_create_dataset(net, nb_elem);
	}
	else
	#endif
	{
		return create_dataset_host(net, nb_elem);
	}
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
	
	//Must check for case where the total number of tick change
	printf("\e[?25l");
	printf("\r[");
	for(i = 0; i < size*((float)c_batch/nb_batch); i++)
		printf("#");
	for(i = size*((float)c_batch/nb_batch); i < size; i++)
		printf("-");
	printf("] %d/%d    Loss: %f", c_batch, nb_batch, loss);
	printf("\e[?25h");
}

void free_dataset(Dataset *data)
{
	int i;
		
	if(data->localization == HOST)
	{
		if(data->input != NULL)
		{
			for(i = 0; i < data->nb_batch; i++)
			{
				free(data->input[i]);
				free(data->target[i]);
			}
		}
		if(&data->input[0] != NULL)
		{
			free(&data->input[0]);
			free(&data->target[0]);
		}
	}
	#ifdef CUDA
	else if(data->localization == DEVICE)
	{
		cuda_free_dataset(data);
		
		if(&data->input[0] != NULL)
		{
			free(&data->input[0]);
			free(&data->target[0]);
		}
	}
	#endif
	
	data->localization = NO_LOC;
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


void save_network(network *net, char *filename, int f_bin)
{
	int i;
	FILE* f = NULL;
	char full_filename[100];
	struct stat st = {0};
	
	//will allow prefix/patch customization in the futur
	sprintf(full_filename, "net_save/%s", filename);
	
	if(stat("net_save", &st) == -1)
		mkdir("net_save", 0700);
	
	if(f_bin)
		f = fopen(full_filename, "wb+");
	else
		f = fopen(full_filename, "w+");
	if(f == NULL)
	{
		printf("ERROR : cannot save %s file\n", full_filename);
		exit(EXIT_FAILURE);
	}

	if(f_bin)
		fwrite(&net->in_dims, sizeof(int), 4, f);
	else
		fprintf(f, "%dx%dx%dx%d\n", net->in_dims[0], net->in_dims[1], net->in_dims[2], net->in_dims[3]);
	for(i = 0; i < net->nb_layers; i++)
	{
		switch(net->net_layers[i]->type)
		{
			case CONV:
				conv_save(f, net->net_layers[i], f_bin);
				break;
			
			case POOL:
				pool_save(f, net->net_layers[i], f_bin);
				break;
		
			case DENSE:
			default:
				dense_save(f, net->net_layers[i], f_bin);
				break;
		}
	}
	
	fclose(f);
}


void load_network(network *net, char *filename, int epoch, int f_bin)
{
	FILE* f = NULL;
	int temp_dim[4];
	char layer_type = 'N';
	
	net->epoch = epoch;
	net->nb_layers = 0;
	
	if(f_bin)
		f = fopen(filename, "rb+");
	else
		f = fopen(filename, "r+");
	
	if(f == NULL)
	{
		printf("ERROR : cannot load/find %s file\n", filename);
		exit(EXIT_FAILURE);
	}
	
	if(f_bin)
		fread(temp_dim, sizeof(int), 4, f);
	else
		fscanf(f, "%dx%dx%dx%d\n", &temp_dim[0], &temp_dim[1], &temp_dim[2], &temp_dim[3]);
	
	
	if(net->in_dims[0] != temp_dim[0] || net->in_dims[1] != temp_dim[1] || net->in_dims[2] != temp_dim[2] || net->in_dims[3] != temp_dim[3])
	{
		printf("ERROR : Wrong image format to load the network\nExpect : W = %d, H = %d, D = %d, C = %d\n", 
			 temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3]);
		exit(EXIT_FAILURE);
	}
	
	do
	{
		if(f_bin)
		{
			if(fread(&layer_type, sizeof(char), 1, f) != 1)
				break;
		}
		else
		{
			if(fscanf(f, "%c", &layer_type) == EOF)
				break;
		}
		
		switch(layer_type)
		{
			case 'C':
				conv_load(net, f, f_bin);
				break;
			
			case 'P':
				pool_load(net, f, f_bin);
				break;
		
			case 'D':
				dense_load(net, f, f_bin);
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


void perf_eval_init(network *net)
{
	if(net->perf_eval == 1)
	{
		net->fwd_perf =  (float*) calloc(net->nb_layers,sizeof(float));
		net->back_perf = (float*) calloc(net->nb_layers,sizeof(float));
		net->fwd_perf_n =  (int*) calloc(net->nb_layers,sizeof(int));
		net->back_perf_n = (int*) calloc(net->nb_layers,sizeof(int));
	
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
			cuda_perf_eval_init();
		#endif
		net->perf_eval = 2;
	}
	else
		return;
}


void perf_eval_in(network *net)
{
	if(net->perf_eval == 0)
		return;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_perf_eval_in();
		#endif
	}
	else
	{
		init_timing(&t_perf_eval);
	}
}


void perf_eval_out(network *net, int layer_id, float *vect, int *n_vect)
{
	float time = 0.0f;
	if(net->perf_eval == 0)
		return;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		time = cuda_perf_eval_out();
		#endif
	}
	else
	{
		time = ellapsed_time(t_perf_eval)*1000;
	}
	
	if(n_vect[layer_id] < 999)
	{
		vect[layer_id] += time;
		n_vect[layer_id] += 1;
	}
}


void perf_eval_display(network *net)
{
	int i;
	float *fwd_time, *back_time, *cumul_time;
	float total_fwd = 0.0f, total_back = 0.0f, total_cumul = 0.0f;
	
	fwd_time   = (float*) calloc(net->nb_layers, sizeof(float));
	back_time  = (float*) calloc(net->nb_layers, sizeof(float));
	cumul_time = (float*) calloc(net->nb_layers, sizeof(float));
	
	if (net->perf_eval == 0)
		return;
	
	for(i = 0; i < net->nb_layers; i++)
	{
		fwd_time[i] = net->fwd_perf[i]/net->fwd_perf_n[i];
		total_fwd += fwd_time[i];
		back_time[i] = net->back_perf[i]/net->back_perf_n[i];
		total_back += back_time[i];
		cumul_time[i] = fwd_time[i] + back_time[i];
		total_cumul += cumul_time[i];
		if(net->fwd_perf_n[i] == 0)
			printf("Warning: some layers were not benchmarked\n");
	}
	
	printf("\n   Layer         Forward            Backprop           Cumulated\n");
	printf("       N       [ms] / [%%]          [ms] / [%%]         [ms] / [%%]\n");	
	printf("  -------------------------------------------------------------------\n");
	for(i = 0; i < net->nb_layers; i++)
		printf("   %5d      %5.3f / %4.1f        %5.3f / %4.1f       %5.3f / %4.1f\n", i+1,
			fwd_time[i], fwd_time[i]/total_fwd*100.0, back_time[i], back_time[i]/total_back*100.0,
			cumul_time[i], cumul_time[i]/total_cumul*100.0);
	
	printf("  -------------------------------------------------------------------\n\n");
	
	free(fwd_time);
	free(back_time);
	free(cumul_time);
}


void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat)
{
	int j, k, l, m, r;
	float** mat = NULL; 
	float* temp = NULL;
	int arg1, arg2;
	float count;
	float *rapp_err = NULL, *rapp_err_rec = NULL;
	int o, pos, in_col;
	double total_error, batch_error = 0.0f;
	double pos_error = 0.0f, size_error = 0.0f, prob_error = 0.0f;
	double objectness_error = 0.0f, class_error = 0.0f, param_error = 0.0f;
	void* output_save = NULL;
	void* output_buffer = NULL;
	struct timeval ep_timer;
	float items_per_s = 0.0f;
	conv_param *c_param;
	yolo_param *a_param;
	float nb_IoU = 0.0f, sum_IoU = 0.0f;
	
	#ifdef CUDA
	float* cuda_mat;
	void* temp_error = NULL;
	#endif

	FILE *f_save = NULL;
	FILE *f_err;
	char f_save_name[100];
	struct stat st = {0};
	
	o = net->output_dim;
	
	if(confusion_matrix)
	{
		rapp_err = (float*) malloc(o*sizeof(float));
		rapp_err_rec = (float*) malloc(o*sizeof(float));
		mat = (float**) malloc(o*sizeof(float*));
		temp = (float*) calloc(o*o,sizeof(float));
		for(j = 0; j < o; j++)
			mat[j] = &(temp[j*o]);
		
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
		{
			cuda_create_table_FP32((void**) &cuda_mat, o*o);
		}
		#endif
	}	
	
	if(saving > 0)
	{
		if(stat("fwd_res", &st) == -1)
    		mkdir("fwd_res", 0700);
	
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
		{
			output_save = (float*) calloc(net->batch_size*net->out_size, sizeof(float));
			cuda_create_host_table(net, &output_buffer, net->batch_size*net->out_size);		
		}
		#endif
		sprintf(f_save_name, "fwd_res/net%d_%04d.dat", net->id, net->epoch);
		if(saving == 1)
			f_save = fopen(f_save_name, "w+");
		else if(saving == 2)
			f_save = fopen(f_save_name, "wb+");
		if(f_save == NULL)
		{
			printf("ERROR: can not oppen %s !\n", f_save_name);
		}
	}
	
	for(r = 0; r < repeat; r++)
	{
		if(r == 0)
			printf("Forward: %d\n", net->epoch);
		if(repeat > 1)
			printf("Forward repeat step: %d /%d\n", r+1, repeat);
		
		total_error = 0.0;
		pos_error = 0.0, size_error = 0.0, prob_error = 0.0;
		objectness_error = 0.0, class_error = 0.0, param_error = 0.0;
		nb_IoU = 0.0f;
		sum_IoU = 0.0f;
		
		init_timing(&ep_timer);
		
		for(j = 0; j < data.nb_batch; j++)
		{
		
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				temp_error = net->output_error;
				net->output_error = net->cu_inst.output_error_cuda;
				#endif
			}
			
			//##########################################################
			
			if(j == data.nb_batch - 1 && data.size%net->batch_size > 0)
			{
				net->length = data.size%net->batch_size;
			}
			else
				net->length = net->batch_size;
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(net->cu_inst.dynamic_load)
				{
					cuda_put_table(net, net->input, data.input[j], net->batch_size*(net->input_dim+1));
					cuda_put_table(net, net->target, data.target[j], net->batch_size*(net->output_dim));
				}
				else
				{
					net->input = data.input[j];
					net->target = data.target[j];
				}
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
				perf_eval_in(net);
				net->net_layers[k]->forward(net->net_layers[k]);
				perf_eval_out(net, k,net->fwd_perf, net->fwd_perf_n);
			}
			
			output_error(net->net_layers[net->nb_layers-1]);
				
			//##########################################################
			
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(net->output_error, temp_error, net->batch_size*net->out_size);
				net->output_error = temp_error;	
				if(saving > 0)
				{

					cuda_get_table_to_FP32(net, net->net_layers[net->nb_layers-1]->output,
						output_save, net->batch_size*net->out_size, output_buffer);
					
					switch(net->net_layers[net->nb_layers-1]->type)
					{
						default:
						case DENSE:
							if(saving == 1)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < net->out_size; l++)
										fprintf(f_save, "%g ", ((float*)output_save)[k*net->out_size + l]);
									fprintf(f_save, "\n");
								}
							}
							else if(saving == 2)
							{
								for(k = 0; k < net->length; k++)
									fwrite(&((float*)output_save)[k*net->out_size], sizeof(float), net->out_size, f_save);
							}
							break;
						case CONV:
							c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
							int batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
							int filter_offset = batch_offset*net->batch_size;
							if(saving == 1)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < c_param->nb_filters; l++)
									{
										for(m = 0; m < batch_offset; m++)
											fprintf(f_save,"%g ", ((float*)output_save)[k*batch_offset + l*filter_offset + m]);
									}
									fprintf(f_save, "\n");
								}
							}
							else if(saving == 2)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < c_param->nb_filters; l++)
										fwrite(&((float*)output_save)[k*batch_offset + l*filter_offset], sizeof(float), batch_offset, f_save);
								}
							}
							break;
					}
				}
				#endif
			}
			else
			{
				output_save = net->net_layers[net->nb_layers-1]->output;
				if(saving > 0)
				{
					switch(net->net_layers[net->nb_layers-1]->type)
					{
						default:
						case DENSE:
							if(saving == 1)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < net->out_size; l++)
										fprintf(f_save, "%g ", ((float*)output_save)[k*net->out_size + l]);
									fprintf(f_save, "\n");
								}
							}
							else if(saving == 2)
							{
								for(k = 0; k < net->length; k++)
									fwrite(&((float*)output_save)[k*net->out_size], sizeof(float), net->out_size, f_save);
							}
							break;
						case CONV:
							c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
							int batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
							int filter_offset = batch_offset*net->batch_size;
							if(saving == 1)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < c_param->nb_filters; l++)
									{
										for(m = 0; m < batch_offset; m++)
											fprintf(f_save,"%g ", ((float*)output_save)[k*batch_offset + l*filter_offset + m]);
									}
									fprintf(f_save, "\n");
								}
							}
							else if(saving == 2)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < c_param->nb_filters; l++)
										fwrite(&((float*)output_save)[k*batch_offset + l*filter_offset], sizeof(float), batch_offset, f_save);
								}
							}
							break;
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
					int batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
					int filter_offset = batch_offset*net->batch_size;
					float *host_IoU_monitor = NULL;
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
					
					if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
					{
						a_param = (yolo_param*)net->net_layers[net->nb_layers-1]->activ_param;
						for(k = 0; k < net->length; k++)
						{
							for(l = 0; l < c_param->nb_filters; l++)
							{
								in_col = l%(8+a_param->nb_class+a_param->nb_param);
								for(m = 0; m < batch_offset; m++)
								{
									if(in_col < 3)
										pos_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									else if(in_col < 6)
										size_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									else if(in_col < 7)
										prob_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									else if(in_col < 8)
										objectness_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									else if(in_col < 9 + a_param->nb_class)
										class_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									else if(in_col < 9 + a_param->nb_class + a_param->nb_param)
										param_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
								}
							}
						}
						
						//move the alloc and free to avoid having them at each batch
						#ifdef CUDA
						if(net->compute_method == C_CUDA)
						{
							host_IoU_monitor = (float*) calloc(2*a_param->nb_box*batch_offset*net->batch_size, sizeof(float));
							cuda_get_table_FP32(a_param->IoU_monitor, host_IoU_monitor, 2*a_param->nb_box*batch_offset*net->batch_size);
						}
						else
						#endif
						{
							host_IoU_monitor = a_param->IoU_monitor;
						}
						for(k = 0; k < 2*a_param->nb_box*batch_offset*net->batch_size; k += 2)
						{
							nb_IoU += host_IoU_monitor[k];
							sum_IoU += host_IoU_monitor[k]*host_IoU_monitor[k+1];
						}
						#ifdef CUDA
						if(net->compute_method == C_CUDA)
						{
							if(host_IoU_monitor != NULL)
								free(host_IoU_monitor);
						}
						#endif
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
		
		items_per_s = data.size/ellapsed_time(ep_timer);
		printf("\nNet. forward perf.: %0.2f items/s\n", items_per_s);
		if(net->no_error != 1)
		{
			if(net->epoch == 1)
				f_err = fopen("error.txt", "w+");
			else
				f_err = fopen("error.txt", "a");
			
			printf("Cumulated error: \t %g\n", total_error/data.size);
			if(net->net_layers[net->nb_layers-1]->type == CONV)
			{
				if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
				{
					printf("Error dist - Pos: %f, Size: %f, Prob: %f, Objct: %f, Class: %f, Param: %f  - Mean Prob. Max IoU = %f\n",
					pos_error/data.size, size_error/data.size, prob_error/data.size, 
					objectness_error/data.size, class_error/data.size, param_error/data.size, sum_IoU/nb_IoU);
				}
			}
			
			if(isnan(total_error))
			{
				printf("ERROR: Network divergence detected (Nan)!\n\n");
				exit(EXIT_FAILURE);
			}
			
			if(net->no_error == 0)
			{
				fprintf(f_err, "%d %g",  net->epoch, total_error/data.size);
				if(net->net_layers[net->nb_layers-1]->type == CONV)
				{
					if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
					{
						fprintf(f_err, " %g %g %g %g %g %g",  
						pos_error/data.size, size_error/data.size, prob_error/data.size, 
						objectness_error/data.size, class_error/data.size, param_error/data.size);
					}
				}
			}

			fprintf(f_err, "\n");
			fclose(f_err);
		}
		
	}
	
	if(confusion_matrix && net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_get_table_FP32((float*)cuda_mat, *mat, o*o);
		cuda_free_table(cuda_mat);
		#endif
	}
	
	if(saving > 0)
	{
		if(output_save != NULL)
			free(output_save);
		if(output_buffer != NULL)
			free(output_buffer);
		fclose(f_save);
	}

	if(confusion_matrix && net->no_error == 0)
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
		printf("%*s\n", (o)*12+22, "Recall");
		for(j = 0; j < o; j++)
		{
			printf("%*s", 10, " ");
			for(k = 0; k < o; k++)
				printf("%10d |", (int) mat[j][k]);
			printf("%11.2f%%\n", rapp_err[j]);
		}
		printf("%10s", "Precision  ");
		for(j = 0; j < o; j++)
			printf("%9.2f%%  ", rapp_err_rec[j]);
		
		count = 0.0;
		for(j = 0; j < o; j++)
			count += mat[j][j];
		
		printf("Acc %6.2f%%\n", count/data.size*100);
		
		free(temp);
		free(mat);
		free(rapp_err);
		free(rapp_err_rec);
	}
}


void train_network(network* net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, float u_decay, int show_confmat, int save_every, int save_bin, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor)
{
	int i, j, k, l, m;
	float begin_learn_rate;
	float end_learn_rate;
	float decay;
	double batch_error = 0.0;
	char net_save_file_name[200];
	struct timeval ep_timer;
	float items_per_s = 0.0;
	int batch_loc;
	int pos;
	conv_param *c_param;
	
	perf_eval_init(net);
	
	#ifdef CUDA
	Dataset shuffle_duplicate;
	void* temp_error = NULL;
	int *index_shuffle = NULL, *index_shuffle_device = NULL;
	
	cuda_set_TC_scale_factor(net, c_TC_scale_factor);
	
	if(net->compute_method == C_CUDA)
	{
		if(net->cu_inst.dynamic_load)
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
				cuda_get_batched_dataset(net, &shuffle_duplicate);
				cuda_convert_table_int(&index_shuffle_device, net->train.size);
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
				((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[0] * 
				((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[1] *
				((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
			break;
			
		case POOL:
			net->out_size = ((pool_param*)net->net_layers[net->nb_layers-1]->param)->prev_depth *
				((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[0] * 
				((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[1] * 
				((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
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
		cuda_create_table_FP32(&net->cu_inst.output_error_cuda, net->batch_size * net->out_size);
		#endif
	}
	
	
	for(i = 0; i < nb_epochs; i++)
	{
		net->learning_rate = end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-decay*net->epoch);
		net->epoch++;
	
		if((net->epoch+1) % shuffle_every == 0 && net->batch_param != SGD)
		{
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(net->cu_inst.dynamic_load)
					cuda_host_only_shuffle(net, net->train);
				else
				{
					if(shuffle_gpu)
						cuda_shuffle(net, net->train, shuffle_duplicate, index_shuffle, index_shuffle_device);
					else
						cuda_host_shuffle(net, net->train, shuffle_duplicate);
				}
				#endif
			}
			else
				host_only_shuffle(net, net->train);
			
		}
		
		init_timing(&ep_timer);
		
		//Loop on all batch for one epoch
		printf("\nEpoch: %d\n", net->epoch);
		net->is_inference = 0;
        net->inference_drop_mode = AVG_MODEL;
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
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(net->cu_inst.dynamic_load)
				{
					cuda_put_table(net, net->input, net->train.input[batch_loc], net->batch_size*(net->input_dim+1));
					cuda_put_table(net, net->target, net->train.target[batch_loc], net->batch_size*(net->output_dim));	
				}
				else
				{
					net->input = net->train.input[batch_loc];
					net->target = net->train.target[batch_loc];
				}
				#endif
			}
			else
			{
				net->input = net->train.input[batch_loc];
				net->target = net->train.target[batch_loc];
			}
			
			for(k = 0; k < net->nb_layers; k++)
			{
				perf_eval_in(net);
				net->net_layers[k]->forward(net->net_layers[k]);
				perf_eval_out(net, k,net->fwd_perf, net->fwd_perf_n);
			}
			
			//print_table(net->net_layers[net->nb_layers-1]->output, 8*8*16*14, net->batch_size);
			
			perf_eval_in(net); //Include output deriv error in the last layer performance metric
			output_deriv_error(net->net_layers[net->nb_layers-1]);
			
			//cuda_print_table(net, net->net_layers[net->nb_layers-1]->output, 28*28, net->batch_size);
			//cuda_print_table(net, net->net_layers[net->nb_layers-1]->delta_o, 28*28, net->batch_size);
			//if(j == 0)
			//	exit(1);
			
			//printf("After\n");
			
			//Propagate error through all layers
			for(k = 0; k < net->nb_layers; k++)
			{
				if(k != 0)
					perf_eval_in(net);
				net->net_layers[net->nb_layers-1-k]->backprop(net->net_layers[net->nb_layers-1-k]);
				perf_eval_out(net, net->nb_layers-1-k, net->back_perf, net->back_perf_n);
			}
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				temp_error = net->output_error;
				net->output_error = net->cu_inst.output_error_cuda;
				#endif
			}
			
			// Live loss monitoring
			output_error(net->net_layers[net->nb_layers-1]);
			
			//cuda_print_table_FP32(net->output_error, 28*28, net->batch_size);
			//if(j == 0)
			//	exit(1);
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(net->output_error, temp_error, net->batch_size*net->out_size);
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
					int batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
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
		
		if(((net->epoch) % control_interv == 0))
		{
			printf("\nNet. training perf.: %0.2f items/s\n", items_per_s);
			printf("Learning rate : %g\n", net->learning_rate);
			net->is_inference = 1;
			net->no_error = 0;
			compute_error(net, net->valid, 0, show_confmat, 1);
		}
		if(save_every > 0)
		{
			if(((net->epoch) % save_every) == 0)
			{
				sprintf(net_save_file_name, "net%d_s%04d.dat", net->id, net->epoch);
				printf("Saving network for epoch: %d\n", net->epoch);
				save_network(net, net_save_file_name, save_bin);
			}
		}
	}
	
	free(net->output_error);
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		cuda_free_table(net->cu_inst.output_error_cuda);
		if(net->cu_inst.dynamic_load)
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
			free_dataset(&shuffle_duplicate);
		}	
	}
	#endif

}


void forward_testset(network *net, int train_step, int saving, int repeat, int drop_mode)
{
	printf("Forwarding testset\n");

	perf_eval_init(net);

	//update out_size in case of forward with no training
	switch(net->net_layers[net->nb_layers-1]->type)
	{
		case CONV:
			net->out_size = ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_filters 
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[0] 
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[1]
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
			break;
			
		case POOL:
			net->out_size = ((pool_param*)net->net_layers[net->nb_layers-1]->param)->prev_depth 
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[0] 
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[1]
				* ((conv_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
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
		cuda_create_table_FP32(&net->cu_inst.output_error_cuda, net->batch_size * net->out_size);
		#endif
	}
	
	#ifdef CUDA
	if(net->cu_inst.dynamic_load)
	{
		cuda_create_table(net, &(net->input), net->batch_size*(net->input_dim+1));
		cuda_create_table(net, &(net->target), net->batch_size*(net->output_dim));
	}
	#endif
	
	if(train_step <= 0)
	{
		printf("Warning, can not forward %d step\n", train_step);
	}
	else
	{
		net->is_inference = 1;
        net->inference_drop_mode = drop_mode;
		compute_error(net, net->test, saving, 0, repeat);
	}
	
	free(net->output_error);
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_free_table(net->cu_inst.output_error_cuda);
		#endif	
	}
	
	#ifdef CUDA
	if(net->cu_inst.dynamic_load)
	{
		cuda_free_table(net->input);
		cuda_free_table(net->target);
	}
	#endif
}




#ifdef CUDA
// Experimental function for now, only for development purpose
void train_gan(network* gen, network* disc, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, float u_decay, float gen_disc_learn_rate_ratio, int save_every, int save_bin, int shuffle_gpu, int shuffle_every, int disc_only, float c_TC_scale_factor)
{
	//1)Generate data or use data provided in the form of a dataset
	//Forward the generative model
	//straightforward
	
	//2) Train the discriminator model
	//Cut the generator model in half and train the discriminator on  0.5 true/ 0.5 false
	//two passes possible
	
	//3) Train the generative model through the frozen discriminative model
	//Connect the train and generative nets (use a fake first layer in discriminator)
	//Freeze the discriminator part
	//Train on new generated data with invert labels
	
	
	int i, j, k, l, i_half;
	//int i,j,k, i_half;
	float begin_learn_rate;
	float end_learn_rate;
	float decay;
	double batch_error = 0.0;
	char net_save_file_name[200];
	struct timeval ep_timer;
	float items_per_s = 0.0;
	int batch_loc = 0;
	int pos;
	
	float *gan_half_target;
	float *gan_reverse_target;
	void* disc_fake_layer_output;
	void* target_point_back;
	
	perf_eval_init(gen);
	
	#ifdef CUDA
	Dataset shuffle_duplicate;
	void* temp_error = NULL;
	int *index_shuffle = NULL, *index_shuffle_device = NULL;
	
	cuda_set_TC_scale_factor(gen, c_TC_scale_factor);
	
	if(gen->compute_method == C_CUDA)
	{
		if(gen->cu_inst.dynamic_load)
		{
			cuda_create_table(gen, &(gen->input), gen->batch_size*(gen->input_dim+1));
			cuda_create_table(gen, &(gen->target), gen->batch_size*(gen->output_dim));
		}
	}
	
	if(disc->compute_method == C_CUDA)
	{
		cuda_create_table(disc, &(disc->input), disc->batch_size*(disc->input_dim+1));
		cuda_create_table(disc, &(disc->target), disc->batch_size*(disc->output_dim));
		
		if(!disc->cu_inst.dynamic_load)
		{
			shuffle_duplicate = create_dataset(disc, disc->train.size);
			if(shuffle_gpu)
			{
				
				index_shuffle = (void*) calloc(disc->train.size,sizeof(int));
				for(i = 0; i < disc->train.size; i++)
					index_shuffle[i] = i;
				index_shuffle_device = (void*)  calloc(disc->train.size,sizeof(int));
				cuda_get_batched_dataset(disc, &shuffle_duplicate);
				cuda_convert_table_int(&index_shuffle_device, disc->train.size);
			}
		}
		
		cuda_create_table(disc, (void**)&gan_half_target, disc->output_dim*disc->batch_size);
		cuda_create_table(disc, (void**)&gan_reverse_target, disc->output_dim*disc->batch_size);
		
		//cuda_create_gan_target(disc, gan_half_target, disc->batch_size, 0.5);
		
		cuda_create_gan_target(disc, gan_reverse_target, disc->target, 0.0, 0);
	
	}
	
	#endif
	
	begin_learn_rate = u_begin_learning_rate;
	end_learn_rate = u_end_learning_rate;
	gen->momentum = u_momentum;
	disc->momentum = u_momentum;
	decay = u_decay;
	
	switch(disc->net_layers[disc->nb_layers-1]->type)
	{
		case CONV:
			disc->out_size = ((conv_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_filters *
				((conv_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[0] * 
				((conv_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[1] *
				((conv_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[2];
			break;
			
		case POOL:
			disc->out_size = ((pool_param*)disc->net_layers[disc->nb_layers-1]->param)->prev_depth *
				((pool_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[0] * 
				((pool_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[1] * 
				((pool_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_area[2];
			break;
	
		case DENSE:
		default:
			disc->out_size = ((dense_param*)disc->net_layers[disc->nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	disc->output_error = (float*) calloc(disc->batch_size * disc->out_size, sizeof(float));
	
	if(disc->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_create_table_FP32(&disc->cu_inst.output_error_cuda, disc->batch_size * disc->out_size);
		#endif
	}
	
	disc_fake_layer_output = disc->net_layers[0]->output;
	target_point_back = disc->target;
	
	for(i = 0; i < nb_epochs; i++)
	{
		gen->learning_rate = end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-decay*gen->epoch);
		//gen->learning_rate = 0.0f;
		gen->epoch++;
		disc->learning_rate = gen_disc_learn_rate_ratio * (end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-decay*gen->epoch));
		disc->epoch++;
		
		printf("%g %g\n", gen->learning_rate, disc->learning_rate);
	
		if((disc->epoch+1) % shuffle_every == 0 && disc->batch_param != SGD)
		{
			if(disc->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(disc->cu_inst.dynamic_load)
					cuda_host_only_shuffle(disc, disc->train);
				else
				{
					if(shuffle_gpu)
						cuda_shuffle(disc, disc->train, shuffle_duplicate, index_shuffle, index_shuffle_device);
					else
						cuda_host_shuffle(disc, disc->train, shuffle_duplicate);
				}
				#endif
			}
			else
				host_only_shuffle(disc, disc->train);
			
		}
		
		init_timing(&ep_timer);
		
		//Loop on all batch for one epoch
		printf("\nEpoch: %d\n", gen->epoch);
		for(j = 0; j < gen->train.nb_batch/2; j++)
		{
			
			if(j*2 == gen->train.nb_batch-1 && gen->train.size%gen->batch_size > 0)
				continue;
			else
				gen->length = gen->batch_size;
				
			//no SGD
			batch_loc = j*2;
			
			if(gen->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(gen->cu_inst.dynamic_load)
				{
					cuda_put_table(gen, gen->input, gen->train.input[batch_loc], gen->batch_size*(gen->input_dim+1));
				}
				else
				{
					gen->input = gen->train.input[batch_loc];
				}
				#endif
			}
			else
			{
				gen->input = gen->train.input[batch_loc];
			}
			
			gen->is_inference = 0;
	        gen->inference_drop_mode = MC_MODEL;
			
			disc->is_inference = 0;
	        disc->inference_drop_mode = MC_MODEL;
			
			for(k = 0; k < gen->nb_layers; k++)
			{
				gen->net_layers[k]->forward(gen->net_layers[k]);
			}
			
			/*if(j == 1)
			{
				cuda_print_table(gen, gen->net_layers[0]->input, 129*gen->batch_size, 129);
				cuda_print_table(gen, gen->net_layers[0]->output, 7*7*129*gen->batch_size, 7*7*129);
				exit(EXIT_SUCCESS);
			}*/
		
			if(j == disc->train.nb_batch-1 && disc->train.size%disc->batch_size > 0)
				continue;
			else
				disc->length = disc->batch_size;
				
			//no SGD
			batch_loc = j;
			
			for(i_half = 0; i_half < 2; i_half++)
			{
				disc->target = target_point_back;
			
				if(disc->compute_method == C_CUDA)
				{
					#ifdef CUDA
					if(disc->cu_inst.dynamic_load)
					{
						cuda_put_table(disc, disc->input, disc->train.input[batch_loc], disc->batch_size*(disc->input_dim+1));
						cuda_put_table(disc, disc->target, disc->train.target[batch_loc], disc->batch_size*(disc->output_dim));
					}
					else
					{
						disc->input = disc->train.input[batch_loc];
						disc->target = disc->train.target[batch_loc];
					}
					#endif
				}
				else
				{
					disc->input = disc->train.input[batch_loc];
					disc->target = disc->train.target[batch_loc];
				}
	
				
			
				//have a "fake" first layer in disc, and create a "mix_up" function to put on this second layer
				//this second layer can then be connected to the gen net output so it can do a fwd+backprop pass easily
				//cuda_print_table(gen, gen->net_layers[gen->nb_layers-1]->output, 28*28*gen->batch_size, gen->batch_size);
				//cuda_print_table_FP32(disc->input, (28*28+1)*gen->batch_size, gen->batch_size);
				//cuda_print_table(disc, disc->net_layers[1]->input, 28*28*disc->batch_size, disc->batch_size);
			
				//disc->net_layers[0]->output = disc_fake_layer_output;
				//Seems OK !
				
				disc->net_layers[1]->previous = disc->net_layers[0];
				
				cuda_gan_disc_mix_input(gen->net_layers[gen->nb_layers-1], disc->net_layers[0], disc->input, i_half);
				cuda_create_gan_target(disc, gan_half_target, disc->target, 0.5f, i_half);
				disc->target = gan_half_target;
				
				//cuda_print_table(disc, disc->net_layers[0]->output, 28*28*disc->batch_size, 28);
				//cuda_print_table(disc, disc->target, 10*disc->batch_size, 10);
				//if(i_half == 1)
				//	exit(1);
				
				/*if(j == 1500)
				{
					cuda_print_table(disc, disc->net_layers[0]->output, 28*28*disc->batch_size, 28);
					exit(EXIT_SUCCESS);
				}*/
				
				//skip first fake layer
				for(k = 1; k < disc->nb_layers; k++)
				{
					disc->net_layers[k]->forward(disc->net_layers[k]);
				}
				
				if(0 && gen->epoch%5 == 0 && j==0)
				{
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->output, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
				}
				
				output_deriv_error(disc->net_layers[disc->nb_layers-1]);
				//cuda_semi_supervised_gan_deriv_output_error(disc->net_layers[disc->nb_layers-1], 1, 0);
				
				if(0 && gen->epoch%5 == 0 && j==0)
				{
					cuda_print_table(disc, disc->target, (disc->output_dim)*disc->batch_size, (disc->output_dim));
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->output, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->delta_o, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
					//exit(1);
				}
				
				for(k = 0; k < disc->nb_layers-1; k++)
				{
					disc->net_layers[disc->nb_layers-1-k]->frozen = 0;
					disc->net_layers[disc->nb_layers-1-k]->backprop(disc->net_layers[disc->nb_layers-1-k]);
				}
			}
			
			if(!disc_only)
			{
				//no SGD
				batch_loc = j*2+1;
				
				if(gen->compute_method == C_CUDA)
				{
					#ifdef CUDA
					if(gen->cu_inst.dynamic_load)
					{
						cuda_put_table(gen, gen->input, gen->train.input[batch_loc], gen->batch_size*(gen->input_dim+1));
					}
					else
					{
						gen->input = gen->train.input[batch_loc];
					}
					#endif
				}
				else
				{
					gen->input = gen->train.input[batch_loc];
				}
				
				//gen->net_layers[gen->nb_layers-1]->output = disc->net_layers[0]->output;
				
				for(k = 0; k < gen->nb_layers; k++)
				{
					gen->net_layers[k]->forward(gen->net_layers[k]);
				}
				
				//cuda_print_table(gen, gen->net_layers[gen->nb_layers-1]->output, 28*28*16, 16);
				
				disc->target = gan_reverse_target;
				
				disc->net_layers[0]->output = gen->net_layers[gen->nb_layers-1]->output;
				
				//disc->net_layers[1]->input = gen->net_layers[gen->nb_layers-1]->output;
				//gen->net_layers[gen->nb_layers-1]->delta_o = disc->net_layers[0]->delta_o;
				
				//disc->net_layers[0]->output = disc_fake_layer_output;
				//disc->net_layers[1]->previous = disc->net_layers[0];
				//Seems OK !
				//cuda_gan_disc_mix_input(gen->net_layers[gen->nb_layers-1], disc->net_layers[0], disc->input, -1);
				
				/*if(gen->epoch == 1)
				{
					cuda_print_table(gen, gen->net_layers[gen->nb_layers-1]->output, 28*28*16, 28);
					//cuda_print_table(disc, disc->net_layers[1]->input, 28*28*16, 16);
					//exit(1);
				}*/
				
				for(k = 1; k < disc->nb_layers; k++)
				{
					disc->net_layers[k]->forward(disc->net_layers[k]);
				}
				
				/*if(1 && gen->epoch%10 == 0 && j==0)
				{
					printf("prev_delta_o\n");
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->delta_o, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
				}*/
				
				output_deriv_error(disc->net_layers[disc->nb_layers-1]);
				//cuda_semi_supervised_gan_deriv_output_error(disc->net_layers[disc->nb_layers-1], 0, 1);
				if(0 && gen->epoch%5 == 0 && j==0)
				{
					printf("Revert\n");
					//cuda_print_table_FP32(disc->target, 10*16, 10);
					cuda_print_table(disc, disc->target, (disc->output_dim)*disc->batch_size, (disc->output_dim));
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->output, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
					cuda_print_table(disc, disc->net_layers[disc->nb_layers-1]->delta_o, (disc->output_dim+1)*disc->batch_size, (disc->output_dim+1));
					//exit(1);
				}
				
				//gen->net_layers[gen->nb_layers-1]->output = disc->net_layers[0]->output;
				disc->net_layers[1]->previous = gen->net_layers[gen->nb_layers-1];
				
				/*if(gen->epoch == 1)
				{
					cuda_print_table(gen, disc->net_layers[0]->output, 28*28*16, 28);
				}*/
				
				for(k = 0; k < disc->nb_layers-1; k++)
				{
					disc->net_layers[disc->nb_layers-1-k]->frozen = 1;
					disc->net_layers[disc->nb_layers-1-k]->backprop(disc->net_layers[disc->nb_layers-1-k]);
				}
				
				//gen->net_layers[gen->nb_layers-1]->delta_o = disc->net_layers[0]->delta_o;
				
				for(k = 0; k < gen->nb_layers; k++)
				{
					gen->net_layers[gen->nb_layers-1-k]->backprop(gen->net_layers[gen->nb_layers-1-k]);
				}
				
				/*if(gen->epoch == 1)
				{
					cuda_print_table(gen, gen->net_layers[gen->nb_layers-2]->delta_o, 28*28*16*16, 28);
					//cuda_print_table(gen, disc->net_layers[0]->delta_o, 28*28*16, 28);
					if(j==1)
						exit(1);
				}*/
			}
			
			
			if(disc->compute_method == C_CUDA)
			{
				#ifdef CUDA
				temp_error = disc->output_error;
				disc->output_error = disc->cu_inst.output_error_cuda;
				#endif
			}
			// Live loss monitoring
			output_error(disc->net_layers[disc->nb_layers-1]);
			
			
			if(disc->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(disc->output_error, temp_error, disc->batch_size*disc->out_size);
				disc->output_error = temp_error;
				#endif
			}
			pos = 0;
			batch_error = 0;
			switch(disc->net_layers[disc->nb_layers-1]->type)
			{
				default:
				case DENSE:
					for(k = 0; k < disc->length; k++)
					{
						for(l = 0; l < disc->out_size; l++)
						{
							pos++;
							batch_error += ((float*)disc->output_error)[pos];
						}
					}
					break;
			}
			batch_error /= disc->length;
			print_epoch_advance(j+1, disc->train.nb_batch, batch_error);

		}
		
		disc->net_layers[0]->output = disc_fake_layer_output;
		
		items_per_s = gen->train.size/ellapsed_time(ep_timer);
		
		if(((gen->epoch) % control_interv == 0))
		{
			//printf("\nControl step epoch: %d\n", net->epoch);
			printf("\nNet. training perf.: %0.2f items/s\n", items_per_s);
			printf("Learning rate : %g\n", gen->learning_rate);
			gen->is_inference = 1;
			gen->no_error = 0;
			//Could be updated, but not really usefull
			//compute_error(net, net->valid, 0, show_confmat, 1);
			//printf("\n");
		}
		if(save_every > 0)
		{
			if(((gen->epoch) % save_every) == 0)
			{
				sprintf(net_save_file_name, "net%d_s%04d.dat", gen->id, gen->epoch);
				printf("Saving network for epoch: %d\n", gen->epoch);
				save_network(gen, net_save_file_name, save_bin);
				
				sprintf(net_save_file_name, "net%d_s%04d.dat", disc->id, disc->epoch);
				printf("Saving network for epoch: %d\n", disc->epoch);
				save_network(disc, net_save_file_name, save_bin);
			}
		}
	}
	
	free(disc->output_error);
	
	#ifdef CUDA
	if(gen->compute_method == C_CUDA)
	{
		if(gen->cu_inst.dynamic_load)
		{
			cuda_free_table(gen->input);
			cuda_free_table(gen->target);
		}
	}
	#endif
	
	#ifdef CUDA
	if(disc->compute_method == C_CUDA)
	{
		cuda_free_table(disc->cu_inst.output_error_cuda);
		if(disc->cu_inst.dynamic_load)
		{
			cuda_free_table(disc->input);
			cuda_free_table(disc->target);
		}
		else if(shuffle_gpu)
		{
			cuda_free_dataset(&shuffle_duplicate);
			cuda_free_table(index_shuffle_device);
			free(index_shuffle);
		}
		else
		{
			free_dataset(&shuffle_duplicate);
		}
		
		cuda_free_table(gan_half_target);
		cuda_free_table(gan_reverse_target);	
	}
	#endif
	printf("\n");
}
#endif





/*
void write_formated_dataset(network *net, const char *filename, Dataset *data, int input_data_type, int output_data_type)
{
	// Create and load a dataset from a format specific file
	
	printf("Write formated dataset function from the pure C interface using CIANNA internal dataset type is not supported anymore.\n Try using the Python interface if realy neaded or write directly a file in the appropriate format.\n");
	exit(EXIT_FAILURE);

	//must be significantly modified to handle mixed precision types
	//Or only allow use for FP32 datasets ? Tricky to use without losing precision anyway

	FILE *f = NULL;
	f = fopen(filename, "wb"); 
	int i, j, k;
	int datasize;
	
	fwrite(&data->size, sizeof(int), 1, f);
	fwrite(&net->input_width, sizeof(int), 1, f);
	fwrite(&net->input_height, sizeof(int), 1, f);
	fwrite(&net->input_depth, sizeof(int), 1, f);
	fwrite(&net->input_channels, sizeof(int), 1, f);
	fwrite(&net->output_dim, sizeof(int), 1, f);
	
	// Should rework this function to avoid repetions, try to use a void pointer that is 
	// properly converted and try to get function pointer to type cast // or templates
	
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
*/

/*
Dataset load_formated_dataset(network *net, const char *filename, int input_data_type, int output_data_type)
{
	// Create and load a dataset from a format specific file
	Dataset data;

	FILE *f = NULL;
	f = fopen(filename, "rb"); 
	int size, width, height, depth, channels, out_dim, c_array_offset;
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
	fread(&channels, sizeof(int), 1, f);
	fread(&out_dim, sizeof(int), 1, f);
	
	
	if( width * height * depth * channels != net->input_dim || out_dim != net->output_dim)
	{
		printf("\nERROR : input dimensions do not match in file %s !\n", filename);
		printf("File dimensions are, size: %d, input dimensions : %dx%dx%dx%d, output dimension : %d\n"
					, size, width, height, depth, channels, out_dim);
		exit(EXIT_FAILURE);
	}
	
	data = create_dataset(net, size);
	
	switch(input_data_type)
	{
		case c_UINT8:
		{
			unsigned char *temp_input;
			float *temp_input_float;
			datasize = sizeof(unsigned char);
			temp_input = (unsigned char *) calloc(net->input_dim, datasize);
			temp_input_float = (float *) calloc(net->input_dim, sizeof(float));
			
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					c_array_offset = j*(net->input_dim + 1);
					fread(temp_input, datasize, net->input_dim, f);
					for(k = 0; k < net->input_dim; k++)
						temp_input_float[k] = (float) temp_input[k];
					data.cont_copy(temp_input_float, data.input[i], c_array_offset, net->input_dim);
					
				}	
			}
			free(temp_input);
			free(temp_input_float);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_input;
			float *temp_input_float;
			datasize = sizeof(unsigned short);
			temp_input = (unsigned short *) calloc(net->input_dim, datasize);
			temp_input_float = (float *) calloc(net->input_dim, sizeof(float));
			
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					c_array_offset = j*(net->input_dim + 1);
					fread(temp_input, datasize, net->input_dim, f);
					for(k = 0; k < net->input_dim; k++)
						temp_input_float[k] = (float) temp_input[k];
					data.cont_copy(temp_input_float, data.input[i], c_array_offset, net->input_dim);
				}	
			}
			free(temp_input);
			free(temp_input_float);
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
					c_array_offset = j*(net->input_dim + 1);
					fread(temp_input, datasize, net->input_dim, f);
					data.cont_copy(temp_input, data.input[i], c_array_offset, net->input_dim);
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
			float *temp_output_float;
			datasize = sizeof(unsigned char);
			temp_output = (unsigned char *) calloc(net->output_dim, datasize);
			temp_output_float = (float *) calloc(net->output_dim, sizeof(float));
			
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_output, datasize, net->output_dim, f);
					for(k = 0; k < net->output_dim; k++)
						temp_output_float[k] = (float) temp_output[k];
					data.cont_copy(temp_output_float, data.target[i], j*net->output_dim, net->output_dim);
				}
			}
			free(temp_output);
			free(temp_output_float);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_output;
			float *temp_output_float;
			datasize = sizeof(unsigned short);
			temp_output = (unsigned short *) calloc(net->output_dim, datasize);
			temp_output_float = (float *) calloc(net->output_dim, sizeof(float));
			
			for(i = 0; i < data.nb_batch; i++)
			{
				for(j = 0; j < net->batch_size; j++)
				{
					if(i*net->batch_size + j >= size)
						continue;
					fread(temp_output, datasize, net->output_dim, f);
					for(k = 0; k < net->output_dim; k++)
						temp_output_float[k] = (float) temp_output[k];
					data.cont_copy(temp_output_float, data.target[i], j*net->output_dim, net->output_dim);
				}
			}
			free(temp_output);
			free(temp_output_float);
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
					data.cont_copy(temp_output, data.target[i], j*net->output_dim, net->output_dim);
				}
			}
			free(temp_output);
			break;
		}
	}
	
	fclose(f);
	
	return data;
}
*/

/*
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
*/

/*
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
*/









