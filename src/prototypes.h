
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





#ifndef PROTOTYPES_H
#define PROTOTYPES_H

#include "structs.h"


//######################################
//         Public variables
//######################################

extern network *networks[MAX_NETWOKRS_NB];
extern int nb_networks;
extern int is_init;

//######################################


//######################################
//       auxil.c prototypes
//######################################
void init_timing(struct timeval* tstart);
float ellapsed_time(struct timeval tstart);
void init_network(int network_number, int u_input_dim[3], int u_output_dim, real in_bias, int u_batch_size, int u_compute_method, int u_dynamic_load);
Dataset create_dataset(network *net, int nb_elem);
void free_dataset(Dataset data);
void write_formated_dataset(network *net, const char *filename, Dataset *data, int input_data_type, int output_data_type);
Dataset load_formated_dataset(network *net, const char *filename, int input_data_type, int output_data_type);
void normalize_datasets(network *net, float *offset_input, float *norm_input, int dim_size_input, float *offset_output, float *norm_output, int dim_size_output);
void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat);
void save_network(network *net, char *filename);
void load_network(network *net, char *filename, int epoch);
void train_network(network* net, int nb_epochs, int control_interv, real u_begin_learning_rate, real u_end_learning_rate, real u_momentum, real u_decay, int show_confmat, int save_net, int shuffle_gpu, int shuffle_every);
void forward_testset(network *net, int train_step, int repeat);


//activations function
void print_activ_param(FILE *f, int type);
void get_string_activ_param(char* activ, int type);
int load_activ_param(char *type);
void define_activation(layer* current);
void linear_activation(layer *current);
void linear_deriv(layer *current);
void output_deriv_error(layer* current);
void output_error_fct(layer* current);

//dense_layer.c
void dense_create(network *net, layer* previous, int nb_neurons, int activation, real drop_rate, FILE *f_load);
void dense_save(FILE *f, layer *current);
void dense_load(network *net, FILE* f);

//conv_layer.c
void conv_create(network *net, layer *previous, int f_size, int nb_filters, int stride, int padding, int activation, FILE *f_load);
void conv_save(FILE *f, layer *current);
void conv_load(network *net, FILE *f);

//pool_layer.c
void pool_create(network *net, layer* previous, int pool_size);
void pool_save(FILE *f, layer *current);
void pool_load(network *net, FILE *f);

//initializers
void xavier_normal(real *tab, int dim_in, int dim_out, int bias_padding);


//######################################


#ifdef CUDA
//######################################
//       CUDA public prototypes
//######################################

#ifdef comp_CUDA
//when compiled by nvcc must be exported as regular C prototypes
//when compiled by gcc act as regular prototype C natively
extern "C"
{
//only make since sources compiles with nvcc
//cuda_main.cu
extern int cu_threads;
extern real cu_alpha, cu_beta;
extern cublasHandle_t cu_handle;
__global__ void cuda_update_weights(real *weights, real* updates, int size);
__global__ void cuda_update_weights_dropout(real *weights, real* update, int size, real *drop_mask, int dim);
__device__ int cuda_argmax(real* tab, int dim_out);
__global__ void im2col_kernel_v4(real* output, real* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias);
#endif
void init_cuda(void);
void cuda_free_table(real* tab);
void cuda_convert_table(real **tab, int size);
void cuda_convert_dataset(network *net, Dataset *data);
void cuda_free_dataset(Dataset *data);
void cuda_create_table(real **tab, int size);
void cuda_get_table(real **cuda_table, real **table, int size);
void cuda_put_table(real **cuda_table, real **table, int size);
void cuda_print_table(real* tab, int size, int return_every);
void cuda_print_table_transpose(real* tab, int line_size, int column_size);
void cuda_confmat(network *net, real* mat);
void cuda_shuffle(network *net, Dataset data, Dataset duplicate, real *index_shuffle, real *index_shuffle_device);
void host_shuffle(network *net, Dataset data, Dataset duplicate);

void cuda_convert_network(layer** network);
void cuda_deriv_output_error(layer *current);
void cuda_output_error_fct(layer* current);
void cuda_output_error(layer* current);
void cuda_compute_error(real** data, real** target_data, int nb_data, float** out_mat, layer *net_layer);


//cuda_activ_functions.cu
void cuda_define_activation(layer *current);
void cuda_deriv_output_error(layer *current);

//cuda_dense_layer.cu
void cuda_dense_define(layer *current);
void cuda_convert_dense_layer(layer *current);

//cuda_conv_layer.cu
void cuda_conv_define(layer *current);
void cuda_convert_conv_layer(layer *current);

//cuda_pool_layer.cu
void cuda_pool_define(layer *current);
void cuda_convert_pool_layer(layer *current);

	
//######################################

#ifdef comp_CUDA
}
#endif

#endif // CUDA


#endif //PROTOTYPES_H


