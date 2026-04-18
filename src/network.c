
/*
	Copyright (C) 2026-... David Cornu
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

// Public are in "prototypes.h"

// Private prototypes
void free_layer(layer *current);
void get_layer_output_dim(layer *current, int *dim);


void init_network(int network_number, int u_input_dim[4], int u_output_dim, float in_bias, int u_batch_size, 
	const char* compute_method_string, int u_dynamic_load, const char* cuda_TC_string, int inference_only, int no_logo, int adv_size)
{
	
	if(!is_init)
	{
	//Set list of networks to NULL at the first init
	for(int i = 0; i < MAX_NETWORKS_NB; i++)
		networks[i] = NULL;
	
	signal(SIGINT, sig_handler);
	
	if(!no_logo)
		printf("\n\
                   ..:^~!?JY5PB~                                                                                             \n\
           J5PGB#&&&&&#BGP55YY#&#!                                                                                           \n\
           &GGB##&&&@@@@@&#PJ?B#&B                                                                                           \n\
          .&#&@@@@@@@@@@@@@@@B##&G                                                                                           \n\
        ^G@@@@BJ^.   .~?P&@@@@@&&G                                                                                           \n\
      &&@@@B^   :!???^^..:7&@@@@&G      .~~~         :~~:         ~~~^        ~~~    ~~~^        ~~~         ^~~.            \n\
    .B@@@&^  !#@@@@@@@@&?^~J&@@@@B ^    !@@@.       :@@@@^       .@@@@G      .@@@:  .@@@@P      .@@@:       ~@@@@.           \n\
   .&@@@#  .J@@@#GP55PB&@&JJ5@@@@B ^G   !@@@.       &@@@@@.      .@@@@@&:     @@@:  .@@@@@&:    .@@@.      :@@@@@&           \n\
   &@@@@.  ?@@&Y?JJJJJJJP@5JY@@@@B ^@:  !@@@       #@@PY@@&      .@@@P@@@7    @@@:  .@@@P@@@7   .@@@.      &@@JG@@B          \n\
  :@@@@#  .@@@YJJJJJJJJJJGY?#@@@@B 5@!  !@@@      P@@#  G@@B     .@@@.:&@@G   @@@:  .@@@.:&@@P  .@@@.     #@@P  &@@5         \n\
  ~@Y&@#  .@@@P?JJJJJJJJJ?J#@@@@@G.@@~  !@@@     ?@@@:  .&@@5    .@@@.  G@@&. @@@:  .@@@.  G@@&. @@@.    P@@&.  :@@@7        \n\
  .@:#@@.  B@@@P5JJJJJJJ5B@@@@@@&#&@&   !@@@    ~@@@@@@@@@@@@7   .@@@.   ?@@@J@@@:  .@@@.   ?@@@J@@@.   ?@@@@@@@@@@@@^       \n\
   5^7@@#   G@@@&&&###&@@@@@@&&B@@@@^   !@@@   .@@@Y??????J@@@^  .@@@.    :&@@@@@:  .@@@.    :&@@@@@.  ~@@@J??????5@@@.      \n\
    : B@@&~  ^B@@@@@@@@@@&GPYJ#@@@@^    !@@@. .@@@J        7@@@: .@@@:      G@@@@:  .@@@.      G@@@@: :@@@!        5@@&.     \n\
       G@@@&?. .~5B&GJ7^:::7B@@@@&      .YJ?  ^YJ7          !JY~  ?JJ        !YJJ    JJJ        !YJJ  !YY!          ?JY^     \n\
        7&@@@@&G?~^^~!Y5G&@@@@@&&G                                                                                           \n\
          ?@@@@@@@@@@@@@@@&#GY##&G                                                                                           \n\
           &PB&@@@@&&##BGP5J??B#&B                                                                                           \n\
           Y55PGB##&&&#BGPP55Y#&#!                                                                                           \n\
                  ...:^~!?JY5PB~                                                                                             \n\n");

	printf("############################################################\n\
CIANNA V-1.0.1.2 stable build (04/2026), by D.Cornu\n\
############################################################\n\n");
	
	}
	
	char string_comp[50]; 
	int comp_int = C_CUDA;
	#if defined _OPENMP || BLAS
	int nb_proc_max, nb_threads_current;
	#endif
	#ifdef CUDA
	int c_mixed_precision = FP32C_FP32A;
	#endif
	network *net;
	
	if(networks[network_number] != NULL)
		free_network(networks[network_number]);
	
	net = (network*) malloc(sizeof(network));
	networks[network_number] = net;
	
	net->id = network_number;
	
	if(strcmp(compute_method_string,"C_CUDA") == 0)
	{
		comp_int = C_CUDA;
		sprintf(string_comp, "CUDA ");
		#ifdef CUDA
		if(strcmp(cuda_TC_string,"off") == 0)
		{
			c_mixed_precision = FP32C_FP32A;
			sprintf(string_comp+5, "(FP32C_FP32A)");
		}
		else if(strcmp(cuda_TC_string,"on") == 0)
		{
			c_mixed_precision = FP16C_FP32A;
			sprintf(string_comp+5, "(FP16C_FP32A)");
		}
		else if(strcmp(cuda_TC_string,"FP32C_FP32A") == 0)
		{
			c_mixed_precision = FP32C_FP32A;
			sprintf(string_comp+5, "(FP32C_FP32A)");
		}
		else if(strcmp(cuda_TC_string,"TF32C_FP32A") == 0)
		{
			c_mixed_precision = TF32C_FP32A;
			sprintf(string_comp+5, "(TF32C_FP32A)");
		}
		else if(strcmp(cuda_TC_string,"FP16C_FP32A") == 0)
		{
			c_mixed_precision = FP16C_FP32A;
			sprintf(string_comp+5, "(FP16C_FP32A)");
		}
		else if(strcmp(cuda_TC_string,"FP16C_FP16A") == 0)
		{
			c_mixed_precision = FP16C_FP16A;
			sprintf(string_comp+5, "(FP16C_FP16A)");
		}
		else if(strcmp(cuda_TC_string,"BF16C_FP32A") == 0)
		{
			c_mixed_precision = BF16C_FP32A;
			sprintf(string_comp+5, "(BF16C_FP32A)");
		}
		#endif
	}
	else if(strcmp(compute_method_string,"C_BLAS") == 0)
	{
		comp_int = C_BLAS;
		sprintf(string_comp, "BLAS");
	}
	else if(strcmp(compute_method_string,"C_NAIV") == 0)
	{
		comp_int = C_NAIV;
		sprintf(string_comp, "NAIV");
	}
	
	#ifdef CUDA
	net->cu_inst.dynamic_load = u_dynamic_load;
	net->cu_inst.use_cuda_TC = c_mixed_precision;
	
	//Additional security, but all call to use_cuda_TC should be safe on their own
	if(comp_int != C_CUDA)
		networks[network_number]->cu_inst.use_cuda_TC = FP32C_FP32A;
	#endif
	
	srand(time(NULL));
	#ifdef CUDA
	if(comp_int == C_CUDA)
		init_cuda(networks[network_number]);
	#endif
	
	#ifndef BLAS
	if(comp_int == C_BLAS)
	{
		printf("\n ERROR: compute method set to BLAS while CIANNA was not compiled for it.\n");
		printf(" Install OpenBLAS and recompile CIANNA with the appropriate option.\n\n");
		exit(EXIT_FAILURE);
	}
	#endif
	if(comp_int == C_NAIV)
	{
		printf("\n WARNING: compute method set to NAIV, which is not optimal.\n");
		printf(" We recommand the use of OpenBLAS for a better usage of CPU ressources.\n");
		printf(" If NAIV with single CPU thread is your only option, we recommand the use of the SGD learning scheme, enabled by setting the batch size to 1.\n\n");
	}
	is_init = 1;
	
	#if defined _OPENMP
	nb_proc_max = omp_get_num_procs();
	nb_threads_current = omp_get_max_threads();
	
	if(nb_threads_current >= nb_proc_max)
	{
		nb_threads_current = fmax(1, nb_proc_max/2);
		omp_set_num_threads(nb_threads_current);
		printf(" WARNING: Number of OpenMP threads likely not set by user.\n");
		printf(" OMP_MAX_THREADS set to %d  (half detected threads)\n", omp_get_max_threads());
		printf(" We recommend investigating manual configuration through environment variables\n\n");
	}
	#endif
	
	#ifdef HAVE_OPENBLAS
	nb_proc_max = openblas_get_num_procs();
	nb_threads_current = openblas_get_num_threads();
	
	if(nb_threads_current >= nb_proc_max)
	{
		nb_threads_current = fmax(1, nb_proc_max/2);
		openblas_set_num_threads(nb_threads_current);
		printf(" WARNING: Number of OpenBLAS threads likely not set by user.\n");
		printf(" OPENBLAS_MAX_THREADS set to %d  (half detected threads)\n", openblas_get_num_threads());
		printf(" We recommend investigating manual configuration through environment variables\n\n");
	}
	#elif BLAS
	printf(" WARNING: the BLAS library in use is not OpenBLAS.\n");
	printf(" Using the default number of threads can result in low performances.\n");
	printf(" We recommend investigating manual configuration through environment variables\n\n");
	#endif

	net->in_dims[0] = u_input_dim[0]; 
	net->in_dims[1] = u_input_dim[1];
	net->in_dims[2] = u_input_dim[2];
	net->in_dims[3] = u_input_dim[3];
	net->input_dim = ((size_t)u_input_dim[0])*u_input_dim[1]*u_input_dim[2]*u_input_dim[3];
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
		printf(" Automatically switch to SGD scheme (batch_size = 1)\n");
	}
	else if(u_batch_size <= 0)
	{
		net->batch_size = 16;
		net->batch_param = FULL;
		printf(" Undefined batch size -> automatic value is 16\n");
	}
	
	net->learning_rate = 0.0f;
	net->momentum = 0.0f;
	net->decay = 0.0f;
	net->weight_decay = 0.0f;
	
	net->compute_method = comp_int;
	net->inference_only = inference_only;
	net->nb_layers = 0;
	net->iter = 0;
	net->is_inference = 0;
	net->inference_drop_mode = AVG_MODEL;
	net->no_error = 0;
	net->perf_eval = 1;
	net->total_nb_param = 0;
	net->memory_footprint = 0;
	net->adv_size = adv_size;
	if(adv_size <= 0)
		net->adv_size = 30;
	
	net->train.localization = NO_LOC;
	net->test.localization = NO_LOC;
	net->valid.localization = NO_LOC;
	
	net->train_buf.localization = NO_LOC;
	net->test_buf.localization = NO_LOC;
	net->valid_buf.localization = NO_LOC;
	
	printf("Network (id: %d) initialized with : \n\
Input dimensions: %dx%dx%dx%d \n\
Output dimension: %d \n\
Batch size: %d \n\
Using %s compute method \n\
Inference only: %d\n\n",
			net->id, net->in_dims[0], net->in_dims[1], net->in_dims[2], net->in_dims[3], 
			net->output_dim, net->batch_size, string_comp, inference_only);
	
	net->TC_scale_factor = 1.0f;
	#ifdef CUDA
	if(net->compute_method == C_CUDA && net->cu_inst.dynamic_load)
		printf("Dynamic load ENABLED\n\n");
	#endif
	
	net->y_param = NULL;
}


void train_network(network* net, int nb_iter, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, 
	float u_decay, float u_weight_decay, int show_confmat, int save_every, int save_bin, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor, int silent)
{
	int i, j, k, l, m;
	float begin_learn_rate;
	float end_learn_rate;
	double batch_error = 0.0, total_error = 0.0;
	char net_save_file_name[200];
	float items_per_s = 0.0;
	int batch_loc;
	conv_param *c_param;
	pool_param *p_param;
	int batch_offset, filter_offset, nb_filters;
	
	if(net->inference_only)
	{
		printf("\n Network was loaded in inference only mode. \n Re-init network with inference only set to false to re-eanble training capability.\n");
		return;
	}
	
	eval_init(net);
	
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
				cuda_convert_table_int(&index_shuffle_device, net->train.size,0);
			}
		}
	}
	#endif
	
	begin_learn_rate = u_begin_learning_rate;
	end_learn_rate = u_end_learning_rate;
	net->momentum = u_momentum;
	net->decay = u_decay;
	net->weight_decay = u_weight_decay;
	
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
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
			break;
	
		case DENSE:
		default:
			net->out_size = ((dense_param*)net->net_layers[net->nb_layers-1]->param)->nb_neurons+1;
			break;
	}
	
	if(net->out_size != net->output_dim+1 && net->net_layers[net->nb_layers-1]->type == DENSE)
	{
		printf("\n ERROR: last layer size does not match the expected output dimensions.\n");
		exit(EXIT_FAILURE);
	}
	
	net->output_error = (float*) calloc(net->batch_size * net->out_size, sizeof(float));
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_create_table_FP32(&net->cu_inst.output_error_cuda, net->batch_size * net->out_size);
		#endif
	}
	
	if(net->iter == 0)
		remove("error.txt");
	
	for(i = 0; i < nb_iter; i++)
	{
		if(silent < 1)
			printf("\n");
		net->learning_rate = end_learn_rate + (begin_learn_rate - end_learn_rate) * expf(-net->decay*net->iter);
		net->iter++;
	
		if(shuffle_every > 0 && (net->iter+1) % shuffle_every == 0 && net->batch_param != SGD)
		{
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				if(net->cu_inst.dynamic_load)
				{
					cuda_host_only_shuffle(net, net->train);
				}
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
		
		epoch_eval_in(net);
		
		//Loop on all batches for one iteration
		total_error = 0.0;
		net->is_inference = 0;
        net->inference_drop_mode = AVG_MODEL;
		for(j = 0; j < net->train.nb_batch; j++)
		{
			
			batch_eval_in(net);
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
				perf_eval_out(net, k, net->fwd_perf, net->fwd_perf_n);
			}
			
			perf_eval_in(net); //Include output deriv error in the last layer performance metric
			output_deriv_error(net->net_layers[net->nb_layers-1]);
			
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
				for(k = 0; k < net->batch_size * net->out_size; k++)
					((float*)net->output_error)[k] = 0.0f;
				cuda_put_table_FP32(net->cu_inst.output_error_cuda, net->output_error, net->batch_size*net->out_size);
			
				temp_error = net->output_error;
				net->output_error = net->cu_inst.output_error_cuda;
				#endif
			}
			
			// Live loss monitoring
			output_error(net->net_layers[net->nb_layers-1]);
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_FP32(net->output_error, temp_error, net->batch_size*net->out_size);
				net->output_error = temp_error;
				#endif
			}
			
			batch_error = 0.0;
			switch(net->net_layers[net->nb_layers-1]->type)
			{
				default:
				case DENSE:
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < net->out_size; l++)
						{
							batch_error += ((float*)net->output_error)[k*net->out_size + l];
							total_error += ((float*)net->output_error)[k*net->out_size + l];
						}
					}
					break;
				case CONV:
				case POOL:
					if(net->net_layers[net->nb_layers-1]->type == CONV)
					{
						c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
						batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
						filter_offset = batch_offset*net->batch_size;
						nb_filters = c_param->nb_filters;
					}
					else
					{
						p_param = (pool_param*)net->net_layers[net->nb_layers-1]->param;
						batch_offset = p_param->nb_area[0]*p_param->nb_area[1]*p_param->nb_area[2];
						filter_offset = batch_offset*net->batch_size;
						nb_filters = p_param->nb_maps;
					}
					for(k = 0; k < net->length; k++)
					{
						for(l = 0; l < nb_filters; l++)
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
			if(silent < 1)
				print_iter_advance(net, j+1, net->train.nb_batch, batch_error, net->batch_size/batch_eval_out(net), 1);
			
		}
		
		items_per_s = net->train.size/epoch_eval_out(net);
		
		if(control_interv > 0 && (net->iter) % control_interv == 0)
		{
			if(silent < 1)
			{
				printf("\n%*s", 14, " ");
				printf("Average Training perf: %0.2f it/s |", items_per_s);
				printf(" Mean Loss: %.5g |", total_error/net->train.size);
				printf(" Learning rate: %.5g | Momentum: %.5g | Weight decay: %.5g\n", net->learning_rate, net->momentum, net->weight_decay);
			}
			net->is_inference = 1;
			net->no_error = 0;
			compute_error(net, net->valid, 0, show_confmat, 1, silent);
		}
		if(save_every > 0)
		{
			if(((net->iter) % save_every) == 0)
			{
				sprintf(net_save_file_name, "net_save/net%d_s%04d.dat", net->id, net->iter);
				printf("Saving network for iteration: %d (mode: %d)\n", net->iter, save_bin);
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


void forward_testset(network *net, int saving, int repeat, int drop_mode, int silent)
{
	if(repeat > 1 && silent != 1)
	{
		printf("Forwarding with repeat = %d", repeat);
		printf("\n");
	}	
	
	eval_init(net);

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
				* ((pool_param*)net->net_layers[net->nb_layers-1]->param)->nb_area[2];
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
		if(net->cu_inst.dynamic_load)
		{
			cuda_create_table(net, &(net->input), net->batch_size*(net->input_dim+1));
			cuda_create_table(net, &(net->target), net->batch_size*(net->output_dim));
		}
		#endif
	}
	
	net->is_inference = 1;
    net->inference_drop_mode = drop_mode;
	compute_error(net, net->test, saving, 0, repeat, silent);
	
	free(net->output_error);
	
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_free_table(net->cu_inst.output_error_cuda);
		if(net->cu_inst.dynamic_load)
		{
			cuda_free_table(net->input);
			cuda_free_table(net->target);
		}
		#endif	
	}
}


void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat, int silent)
{
	int j, k, l, m, r;
	float** mat = NULL; 
	float* temp = NULL;
	int arg1, arg2;
	float count;
	float *rapp_err = NULL, *rapp_err_rec = NULL;
	int o, in_col, width_conf, repeat_start;
	double total_error = 0.0, batch_error = 0.0;
	double pos_error = 0.0, size_error = 0.0, prob_error = 0.0;
	double objectness_error = 0.0, class_error = 0.0, param_error = 0.0;
	void* output_save = NULL;
	void* output_buffer = NULL;
	float* host_target = NULL;
	float items_per_s = 0.0f;
	conv_param *c_param;
	pool_param *p_param;
	yolo_param *a_param;
	int batch_offset, filter_offset, nb_filters;
	float nb_IoU = 0.0f, nb_good_IoU = 0.0f, sum_IoU = 0.0f, sum_objectness = 0.0f;
	int l_box;
	float grid_elem_size[3], priors[3];
	int l_nb_area[3], grid_elem[3];
	float l_out;
	
	#ifdef CUDA
	void* temp_error = NULL;
	#endif

	FILE *f_save = NULL;
	FILE *f_err;
	char f_save_name[100];
	struct stat st = {0};
	
	o = net->output_dim;
	
	if(confusion_matrix > 0)
	{
		rapp_err = (float*) malloc(o*sizeof(float));
		rapp_err_rec = (float*) malloc(o*sizeof(float));
		mat = (float**) malloc(o*sizeof(float*));
		temp = (float*) calloc(o*o,sizeof(float));
		for(j = 0; j < o; j++)
			mat[j] = &(temp[j*o]);
	}	
	
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		output_save = (float*) calloc(net->batch_size*net->out_size, sizeof(float));
		cuda_create_host_table(net, &output_buffer, net->batch_size*net->out_size);		
		host_target = (float*) calloc(net->batch_size*net->out_size, sizeof(float));
	}
	#endif
	
	if(saving > 0)
	{
		if(stat("fwd_res", &st) == -1)
    		mkdir("fwd_res", 0700);
		sprintf(f_save_name, "fwd_res/net%d_%04d.dat", net->id, net->iter);
		if(saving == 1)
			f_save = fopen(f_save_name, "w+");
		else if(saving == 2)
			f_save = fopen(f_save_name, "wb+");
		if(f_save == NULL)
		{
			printf("\n ERROR: can not oppen %s !\n", f_save_name);
			exit(EXIT_FAILURE);
		}
	}
		
	total_error = 0.0;
	pos_error = 0.0, size_error = 0.0, prob_error = 0.0;
	objectness_error = 0.0, class_error = 0.0, param_error = 0.0;
	nb_IoU = 0.0f;
	sum_IoU = 0.0f;
	sum_objectness = 0.0f;
	nb_good_IoU = 0.0f;
	
	epoch_eval_in(net);
	
	for(j = 0; j < data.nb_batch; j++)
	{
		batch_eval_in(net);
		
		//##########################################################
		if(j == data.nb_batch - 1 && data.size%net->batch_size > 0)
			net->length = data.size%net->batch_size;
		else
			net->length = net->batch_size;
		
		if(net->compute_method == C_CUDA)
		{
			#ifdef CUDA
			if(net->cu_inst.dynamic_load)
			{
				cuda_put_table(net, net->input, data.input[j], net->batch_size*(net->input_dim+1));
				cuda_put_table(net, net->target, data.target[j], net->batch_size*(net->output_dim));
				cuda_get_typed_host_table(net, data.target[j], host_target, net->batch_size*(net->output_dim));
			}
			else
			{
				net->input = data.input[j];
				net->target = data.target[j]; 
				cuda_get_table_to_FP32(net, data.target[j], host_target, net->batch_size*(net->output_dim), output_buffer);
			}
			#endif
		}
		else
		{
			net->input = data.input[j];
			net->target = data.target[j];
			host_target = data.target[j];
		}
		
		repeat_start = 0;
		for(r = 0; r < repeat; r++)
		{
			for(k = repeat_start; k < net->nb_layers; k++)
			{
				if(repeat_start == 0 && net->net_layers[k]->dropout_rate > 0.01f)
					repeat_start = k;
				perf_eval_in(net);
				net->net_layers[k]->forward(net->net_layers[k]);
				perf_eval_out(net, k,net->fwd_perf, net->fwd_perf_n);
			}

			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				for(k = 0; k < net->batch_size * net->out_size; k++)
					((float*)net->output_error)[k] = 0.0f;
				cuda_put_table_FP32(net->cu_inst.output_error_cuda, net->output_error, net->batch_size*net->out_size);
				
				temp_error = net->output_error;
				net->output_error = net->cu_inst.output_error_cuda;
				#endif
			}

			if(net->no_error != 1)
				output_error(net->net_layers[net->nb_layers-1]);

			//##########################################################
			
			if(net->compute_method == C_CUDA)
			{
				#ifdef CUDA
				cuda_get_table_to_FP32(net, net->net_layers[net->nb_layers-1]->output,
						output_save, net->batch_size*net->out_size, output_buffer);
				
				cuda_get_table_FP32(net->output_error, temp_error, net->batch_size*net->out_size);
				net->output_error = temp_error;
				#endif
			}
			else
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
							for(k = 0; k < net->length; k++)
								fwrite(&((float*)output_save)[k*net->out_size], sizeof(float), net->out_size, f_save);
						break;
					case CONV:
					case POOL:
						if(net->net_layers[net->nb_layers-1]->type == CONV)
						{
							c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
							batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
							filter_offset = batch_offset*net->batch_size;
							nb_filters = c_param->nb_filters;
							for(k = 0; k < 3; k++)
							{
								grid_elem_size[k] = net->in_dims[k]/c_param->nb_area[k]; 
								l_nb_area[k] = c_param->nb_area[k];
							}
						}
						else
						{
							p_param = (pool_param*)net->net_layers[net->nb_layers-1]->param;
							batch_offset = p_param->nb_area[0]*p_param->nb_area[1]*p_param->nb_area[2];
							filter_offset = batch_offset*net->batch_size;
							nb_filters = p_param->nb_maps;
							for(k = 0; k < 3; k++)
							{
								grid_elem_size[k] = net->in_dims[k]/p_param->nb_area[k]; 
								l_nb_area[k] = p_param->nb_area[k];
							}
						}
						
						if(net->net_layers[net->nb_layers-1]->activation_type == YOLO && net->y_param->raw_output == 0)
						{
							a_param = (yolo_param*)net->y_param;
							for(k = 0; k < net->length; k++)
							{
								for(l = 0; l < nb_filters; l++)
								{
									l_box = l/(8+a_param->nb_class+a_param->nb_param);
									in_col = l%(8+a_param->nb_class+a_param->nb_param);
									
									for(m = 0; m < 3; m++)
										priors[m] = a_param->prior_size[l_box*3+m];
									
									for(m = 0; m < batch_offset; m++)
									{
										grid_elem[2] = m / (l_nb_area[1]*l_nb_area[0]);
										grid_elem[1] = (m % (l_nb_area[1]*l_nb_area[0]) / l_nb_area[0]);
										grid_elem[0] = (m % (l_nb_area[1]*l_nb_area[0]) % l_nb_area[0]);
										
										if(in_col < 3)
										{
											l_out = grid_elem[in_col]*grid_elem_size[in_col];
											l_out += ((float*)output_save)[k*batch_offset + l*filter_offset + m] * grid_elem_size[in_col];
											l_out -= 0.5f*priors[in_col]*expf(((float*)output_save)[k*batch_offset + (l+3)*filter_offset + m]);
										}
										else if(in_col < 6)
										{
											l_out = grid_elem[in_col-3]*grid_elem_size[in_col-3];
											l_out += ((float*)output_save)[k*batch_offset + (l-3)*filter_offset + m] * grid_elem_size[in_col-3];
											l_out += 0.5f*priors[in_col-3]*expf(((float*)output_save)[k*batch_offset + l*filter_offset + m]);
										}
										else if(in_col >= 6)
											l_out = ((float*)output_save)[k*batch_offset + l*filter_offset + m];
									
										if(saving == 1)
											fprintf(f_save, "%g ", l_out);
										else if(saving == 2)
											fwrite(&l_out, sizeof(float), 1, f_save);
									}
								}
							}
						}
						else
						{
							if(saving == 1)
							{
								for(k = 0; k < net->length; k++)
								{
									for(l = 0; l < nb_filters; l++)
										for(m = 0; m < batch_offset; m++)
											fprintf(f_save,"%g ", ((float*)output_save)[k*batch_offset + l*filter_offset + m]);
									fprintf(f_save, "\n");
								}
							}
							else if(saving == 2)
							{
								for(k = 0; k < net->length; k++)
									for(l = 0; l < nb_filters; l++)
										fwrite(&((float*)output_save)[k*batch_offset + l*filter_offset], sizeof(float), batch_offset, f_save);
							}
						}
						break;
				}
			}
			
			if(net->no_error != 1)
			{
				batch_error = 0.0;
				switch(net->net_layers[net->nb_layers-1]->type)
				{
					default:
					case DENSE:
						for(k = 0; k < net->length; k++)
						{
							for(l = 0; l < net->out_size; l++)
							{
								batch_error += ((float*)net->output_error)[k*net->out_size + l];
								total_error += ((float*)net->output_error)[k*net->out_size + l];
							}
							
							if(confusion_matrix > 0)
							{
								arg1 = argmax(&(((float*)host_target)[k*net->output_dim]), net->output_dim);
								arg2 = argmax(&(((float*)output_save)[k*(net->output_dim+1)]),
									net->output_dim);
								mat[arg1][arg2]++;
							}
						}
						break;
					case CONV:
					case POOL:
						if(net->net_layers[net->nb_layers-1]->type == CONV)
						{
							c_param = (conv_param*)net->net_layers[net->nb_layers-1]->param;
							batch_offset = c_param->nb_area[0]*c_param->nb_area[1]*c_param->nb_area[2];
							filter_offset = batch_offset*net->batch_size;
							nb_filters = c_param->nb_filters;
						}
						else
						{
							p_param = (pool_param*)net->net_layers[net->nb_layers-1]->param;
							batch_offset = p_param->nb_area[0]*p_param->nb_area[1]*p_param->nb_area[2];
							filter_offset = batch_offset*net->batch_size;
							nb_filters = p_param->nb_maps;
						}
						
						for(k = 0; k < net->length; k++)
						{
							for(l = 0; l < nb_filters; l++)
							{
								for(m = 0; m < batch_offset; m++)
								{
									batch_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									total_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
								}
							}
							
							if(batch_offset == 1 && confusion_matrix > 0)
							{
								arg1 = argmax(&(((float*)host_target)[k*net->output_dim]), net->output_dim);
								arg2 = conv_argmax(&(((float*)output_save)[k]), filter_offset, nb_filters);
								mat[arg1][arg2]++;
							}
						}
						
						float *host_IoU_monitor = NULL;
						if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
						{
							a_param = (yolo_param*)net->net_layers[net->nb_layers-1]->activ_param;
							for(k = 0; k < net->length; k++)
							{
								for(l = 0; l < nb_filters; l++)
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
										else if(a_param->nb_class > 0 && in_col < 8 + a_param->nb_class)
											class_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
										else if(a_param->nb_param > 0 && in_col < 8 + a_param->nb_class + a_param->nb_param)
											param_error += ((float*)net->output_error)[k*batch_offset + l*filter_offset + m];
									}
								}
							}
							
							//could move the alloc and free to avoid having them at each batch
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
								if(host_IoU_monitor[k] > -0.98f)
								{
									nb_IoU += 1;
									sum_objectness += host_IoU_monitor[k];
									sum_IoU += host_IoU_monitor[k+1];
									if(host_IoU_monitor[k+1] >= ((yolo_param*)net->y_param)->IoU_limits[0])
										nb_good_IoU += 1;
								}
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
			}
		}
		batch_error /= net->length;
		if(silent != 1)
			print_iter_advance(net, j+1, data.nb_batch, batch_error, (net->batch_size)/batch_eval_out(net), 0);
	}
	
	//scaling data.size by repeat make no sense because an "item" is now a combination of full and partial forward
	//better to display the item time as the time for all repeat / item
	items_per_s = (data.size)/epoch_eval_out(net);
	
	if(silent != 1)
	{
		printf("\n%*s", 14, " ");
		printf("Average forward perf : %0.2f it/s ", items_per_s);
		if(net->no_error != 1)
		{	
			printf("| Mean Loss: %.5g", total_error/(data.size*repeat));
			if(net->net_layers[net->nb_layers-1]->type == CONV)
			{
				if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
				{
					printf("\nLoss dist. ||Pos: %.5f |Size: %.5f |Prob: %.5f |Obj: %.5f |Class: %.5f |Param: %.5f ||M IoU = %.4f |M Obj = %0.4f |P Good = %0.4f",
					pos_error/(data.size*repeat), size_error/(data.size*repeat), prob_error/(data.size*repeat), 
					objectness_error/(data.size*repeat), class_error/(data.size*repeat), param_error/(data.size*repeat), 
					sum_IoU/nb_IoU, sum_objectness/nb_IoU, (float)nb_good_IoU/(float)nb_IoU);
				}
			}
			
			if(isnan(total_error))
			{
				printf("\n ERROR: Network divergence detected (Nan)!\n\n");
				exit(EXIT_FAILURE);
			}
			
			if(net->no_error == 0)
			{
				
				f_err = fopen("error.txt", "a");
				if(f_err == NULL)
					f_err = fopen("error.txt", "w+");
			
				fprintf(f_err, "%d %g",  net->iter, total_error/data.size);
				if(net->net_layers[net->nb_layers-1]->type == CONV)
				{
					if(net->net_layers[net->nb_layers-1]->activation_type == YOLO)
					{
						fprintf(f_err, " %g %g %g %g %g %g",  
						pos_error/data.size, size_error/data.size, prob_error/data.size, 
						objectness_error/data.size, class_error/data.size, param_error/data.size);
					}
				}
				fprintf(f_err, "\n");
				fclose(f_err);
			}
		}
		printf("\n");
		
	}
	
	if(net->compute_method == C_CUDA)
	{
		if(output_save != NULL)
			free(output_save);
		if(output_buffer != NULL)
			free(output_buffer);
		if(host_target != NULL)
			free(host_target);
	}
	
	if(saving > 0)
		fclose(f_save);

	if(confusion_matrix > 0 && net->no_error == 0 && repeat <= 1)
	{
		if(silent != 1)
		{
			if(confusion_matrix == 1)
			{
				printf("\n   ");
				width_conf = (o*10) / 2;
				for(j = 0; j < width_conf - 3; j++)
					printf("*");
				printf("  ConfMat  ");
				for(j = 0; j < width_conf - 3; j++)
					printf("*");
				printf("   Recall\n");
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
				for(j = 0; j < o; j++)
				{
					printf("%*s", 5, " ");
					for(k = 0; k < o; k++)
						printf("%8d |", (int) mat[j][k]);
					printf("%11.2f%%\n", rapp_err[j]);
				}
				printf("%6s", "Prec. ");
				for(j = 0; j < o; j++)
					printf("%7.2f%%  ", rapp_err_rec[j]);
				
				count = 0.0;
				for(j = 0; j < o; j++)
					count += mat[j][j];
				
				printf("Acc %6.2f%%\n", count/data.size*100);
			}
			else if(confusion_matrix == 2)
			{
				printf("\n   ");
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
				printf("\n Recall:   ");
				for(j = 0; j < o; j++)
					printf("%7.2f%%  ", rapp_err[j]);
				printf("\n Precision:");
				for(j = 0; j < o; j++)
					printf("%7.2f%%  ", rapp_err_rec[j]);
				
				count = 0.0;
				for(j = 0; j < o; j++)
					count += mat[j][j];
				printf("\n Accuracy: %6.2f%%\n", count/data.size*100);
			}
			else if(confusion_matrix == 3)
			{
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
				
				count = 0.0;
				for(j = 0; j < o; j++)
					count += mat[j][j];
				printf("\n Accuracy: %6.2f%%\n", count/data.size*100);
			}
		}
		
		free(temp);
		free(mat);
		free(rapp_err);
		free(rapp_err_rec);
	}
}


void update_weights(void *weights, void* update, float weight_decay, int is_pivot, int size)
{
	int i;
	
	float* f_weights = (float*) weights;
	float* f_update = (float*) update;
	
	//No pragma parallel. No perf improvement. Could be re-tested since addition of weight decay
	for(i = 0; i < size-is_pivot; i++)
	{   //Here the weight_decay variable include the learning rate scaling
		f_update[i] += weight_decay*f_weights[i];
		f_weights[i] -= f_update[i];
	}
}


void set_frozen_layers(network *net, int* tab, int dim)
{
	int i;
	
	for(i = 0; i < dim; i++)
		net->net_layers[i]->frozen = tab[i];
}


void save_network(network *net, const char *filename, int f_bin)
{
	int i;
	FILE* f = NULL;
	char full_filename[300];
	struct stat st = {0};
	
	sprintf(full_filename, "%s", filename);
	
	if(stat("net_save", &st) == -1)
		mkdir("net_save", 0700);
	
	if(f_bin)
		f = fopen(full_filename, "wb+");
	else
		f = fopen(full_filename, "w+");
	if(f == NULL)
	{
		printf(" ERROR : cannot save %s file\n", full_filename);
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
		
			case NORM:
				norm_save(f, net->net_layers[i], f_bin);
				break;
			
			case LRN:
				lrn_save(f, net->net_layers[i], f_bin);
				break;
			
			case DENSE:
			default:
				dense_save(f, net->net_layers[i], f_bin);
				break;
		}
	}
	
	fclose(f);
}


void load_network(network *net, const char *filename, int iter, int nb_layers, int nb_skip_layers, int f_bin)
{
	int i;
	FILE* f = NULL;
	int temp_dim[4];
	int dim_prod[2];
	char layer_type = 'A';
	int layer_count = 0;
	int skip_layer;
	
	net->iter = iter;
	
	if(f_bin)
		f = fopen(filename, "rb+");
	else
		f = fopen(filename, "r+");
	
	if(f == NULL)
	{
		printf("\n ERROR: cannot load/find %s file\n", filename);
		exit(EXIT_FAILURE);
	}
	
	if(f_bin)
		fread(temp_dim, sizeof(int), 4, f);
	else
		fscanf(f, "%dx%dx%dx%d\n", &temp_dim[0], &temp_dim[1], &temp_dim[2], &temp_dim[3]);
	
	if(nb_skip_layers < 0)
		nb_skip_layers = 0;
	
	for(i = 0; i < 4; i++)
		net->skip_in_dims[i] = temp_dim[i];
	skip_layer = 1;
	
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
	
		if(layer_count == nb_skip_layers)
		{
			skip_layer = 0;
			if(net->nb_layers == 0)
				for(i = 0; i < 4; i++)
					temp_dim[i] = net->in_dims[i];
			else
				get_layer_output_dim(net->net_layers[net->nb_layers-1], temp_dim);
			
			switch(layer_type)
			{
				case 'C':
				case 'P':
				case 'N':
				case 'L':
					if(net->skip_in_dims[3] != temp_dim[3])
					{
						printf("\n ERROR: Incompatible input dimension (depth) when loading conv formated layer!\n");
						exit(EXIT_FAILURE);
					}
					break;
				case 'D':
					dim_prod[0] = 1; dim_prod[1] = 1;
					for(i = 0; i < 4; i++)
					{
						dim_prod[0] *= net->skip_in_dims[i];
						dim_prod[1] *= temp_dim[i]; 
					}
					if(dim_prod[0] != dim_prod[1])
					{
						printf("\n ERROR: Incompatible input dimension when loading dense formated layer!\n");
						exit(EXIT_FAILURE);
					}
					break;
				case ' ':
				case '\n':
				default:
					break;
			}
		}
		
		
		switch(layer_type)
		{
			case 'C':
				conv_load(net, f, f_bin, skip_layer);
				break;
			
			case 'P':
				pool_load(net, f, f_bin, skip_layer);
				break;
		
			case 'N':
				norm_load(net, f, f_bin, skip_layer);
				break;
			
			case 'L':
				lrn_load(net, f, f_bin, skip_layer);
				break;
			
			case 'D':
				dense_load(net, f, f_bin, skip_layer);
				break;
			case ' ':
			case '\n':
				layer_count--;
				break;
			default:
				printf("\n ERROR: Layer type not recognized when loading the model, likely file format error!\n");
				exit(EXIT_FAILURE);
				break;
		}
		layer_count++;
	
	}while(nb_layers <= 0 || layer_count < nb_skip_layers + nb_layers);
	
	fclose(f);
}


void free_network(network *net)
{	
	if(net == NULL)
		return;
		
	free_dataset(&net->train);
	free_dataset(&net->test);
	free_dataset(&net->valid);
	free_dataset(&net->train_buf);
	free_dataset(&net->test_buf);
	free_dataset(&net->valid_buf);

	for(int k = 0; k < net->nb_layers; k++)
		free_layer(net->net_layers[net->nb_layers-1-k]);

	if(net->y_param != NULL)
		free_yolo_params(net);
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
		free_cuda_network();
	#endif
	
	free(net);
	net = NULL;
}


void free_layer(layer *current)
{
	/****** WARNING ******
	This function is not meant to remove a layer in the middle of a network structure. 
	No check is done to verify if the network would still work with	the layer removed.
	Truncating a network backbone is done through partial loading from a save state in the load function.
	*/
	switch(current->type)
	{
		case DENSE:
			free_dense(current);
			break;
		
		case CONV:
			free_conv(current);
			break;
		
		case POOL:
			free_pool(current);
			break;
		
		case NORM:
			free_norm(current);
			break;
		
		case LRN:
			free_lrn(current);
			break;
	
		default:
			printf("\n ERROR: Unknown layer type in free_layer.\n");
			exit(EXIT_FAILURE);
			break;
	}
}

void get_layer_output_dim(layer *current, int *dim)
{
	switch(current->type)
	{
		case DENSE:
			get_dense_output_dim(current, dim);
			break;
		case CONV:
			get_conv_output_dim(current, dim);
			break;
		case POOL:
			get_pool_output_dim(current, dim);
			break;
		case NORM:
			get_norm_output_dim(current, dim);
			break;
		case LRN:
			get_lrn_output_dim(current, dim);
			break;
		default:
			break;
	}	
}
