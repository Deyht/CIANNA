
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

extern network *networks[MAX_NETWORKS_NB];
extern int nb_networks;
extern int is_init;
extern int verbose;

//######################################


//######################################
//auxil.c
void init_timing(struct timeval* tstart);
float ellapsed_time(struct timeval tstart);
void init_network(int network_number, int u_input_dim[4], int u_output_dim, float in_bias, int u_batch_size, int u_compute_method, int u_dynamic_load, int u_use_cuda_TC);
Dataset create_dataset(network *net, int nb_elem);
void free_dataset(Dataset data);
void print_table(float* tab, int column_size, int nb_column);
void write_formated_dataset(network *net, const char *filename, Dataset *data, int input_data_type, int output_data_type);
Dataset load_formated_dataset(network *net, const char *filename, int input_data_type, int output_data_type);
void set_normalize_dataset_parameters(network *net, float *offset_input, float *norm_input, int dim_size_input, float *offset_output, float *norm_output, int dim_size_output);
void normalize_dataset(network *net, Dataset c_data);
void update_weights(void *weights, void* update, int size);
void perf_eval_display(network *net);
void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat);
void save_network(network *net, char *filename);
void load_network(network *net, char *filename, int epoch);
void train_network(network* net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, float u_decay, int show_confmat, int save_net, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor);
void forward_testset(network *net, int train_step, int saving, int repeat, int drop_mode);


//activations.c
void define_activation(layer *current);
void output_error(layer* current);
void output_deriv_error(layer* current);
void print_activ_param(FILE *f, int type);
void get_string_activ_param(char* activ, int type);
int load_activ_param(char *type);
int set_yolo_params(network *net, int nb_box, int IoU_type, float *prior_w, float *prior_h, float *prior_d, float *yolo_noobj_prob_prior, int nb_class, int nb_param, int strict_box_size, float *scale_tab, float **slopes_and_maxes_tab, float *IoU_limits, int *fit_parts);

//dense_layer.c
void dense_create(network *net, layer* previous, int nb_neurons, int activation, float drop_rate, FILE *f_load);
void dense_save(FILE *f, layer *current);
void dense_load(network *net, FILE* f);

//conv_layer.c
void conv_create(network *net, layer *previous, int *f_size, int nb_filters, int *stride, int *padding, int activation, float drop_rate, FILE *f_load);
void conv_save(FILE *f, layer *current);
void conv_load(network *net, FILE *f);

void pool_create(network *net, layer* previous, int *pool_size, float drop_rate);
void pool_save(FILE *f, layer *current);
void pool_load(network *net, FILE *f);

//initializers.c
float random_uniform(void);
float random_normal(void);
void xavier_normal(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value);
void xavier_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value);


//######################################

#ifdef BLAS
void blas_dense_define(layer *current);
void blas_conv_define(layer *current);

#endif 

void pool_define(layer *current);
void naiv_dense_define(layer *current);
void naiv_conv_define(layer *current);

void flat_dense(void* in, void* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void reroll_batch(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void dropout_select(int* mask, int size, float drop_rate);
void dropout_apply(void* table, int batch_size, int dim, int* mask);

void add_bias_im2col(void* output, float bias_value, int flat_f_size, int size);
void rotate_filter_matrix(void* in, void* out, int nb_rows, int depth_size, int nb_filters_in, int len);
void unroll_conv(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void reroll_delta_o(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void im2col_fct_v4(void* output, void* input, int image_size, int flat_image_size, int stride, int padding, int internal_padding, int depth, int depth_padding, int image_padding, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias);


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
extern float cu_alpha, cu_beta;
extern float TC_scale_factor;
extern cublasHandle_t cu_handle;
extern cudaDataType cuda_data_type;
extern cudaDataType cuda_compute_type;
void cuda_master_weight_FP32_to_FP16(float *master, half *copy, int size);
void cuda_update_weights(network* net, void *weights, void* update, int size);

//__global__ void cuda_update_weights_dropout(void *weights, void* update, int size, int *drop_mask, int dim);
__device__ int cuda_argmax(void* tab, int dim_out);
#endif
void init_cuda(network* net);
void cuda_set_TC_scale_factor(float val);
void cuda_sync(void);
void cuda_free_table(void* tab);
void cuda_create_host_table_FP16(network* net, void **tab, int size);
void cuda_convert_table(network* net, void **tab, long long int size);
void cuda_convert_table_int(network* net, int **tab, int size);
void cuda_convert_dataset(network *net, Dataset *data);
void cuda_convert_host_dataset_FP32(network *net, Dataset *data);
Dataset create_dataset_FP16(network *net, int nb_elem);
void cuda_free_dataset(Dataset *data);
void cuda_create_table_FP32(network* net, float **tab, float size);
void cuda_create_table(network* net, void **tab, int size);
void cuda_get_table_FP32(network* net, float *cuda_table, float *table, int size);
void cuda_get_table_FP16_to_FP32(void *cuda_table, void *table, int size, void* buffer);
void cuda_get_table(network* net, void *cuda_table, void *table, int size);
void cuda_put_table_FP32(network* net, float *cuda_table, float *table, int size);
void cuda_put_table(network* net, void *cuda_table, void *table, int size);
void cuda_print_table_FP32(network* net, float* tab, int size, int return_every);
void cuda_print_table_4d(network* net, void* tab, int w_size, int h_size, int d_size, int last_dim, int biased);
void cuda_print_table(network* net, void* tab, int size, int return_every);
void cuda_print_table_int(network* net, int* tab, int size, int return_every);
void cuda_print_table_host_fp16(network* net, void* tab, int size, int return_every);
void cuda_print_table_transpose(void* tab, int line_size, int column_size);
void cuda_confmat(network *net, float* mat);
void cuda_perf_eval_init(void);
void cuda_perf_eval_in(void);
float cuda_perf_eval_out(void);
void cuda_shuffle(network *net, Dataset data, Dataset duplicate, int *index_shuffle, int *index_shuffle_device);
void host_shuffle(network *net, Dataset data, Dataset duplicate);
void cuda_host_only_shuffle(network *net, Dataset data);

void cuda_convert_network(layer** network);
void cuda_deriv_output_error(layer *current);
void cuda_output_error_fct(layer* current);
void cuda_output_error(layer* current);
void cuda_compute_error(void** data, void** target_data, int nb_data, float** out_mat, layer *net_layer);


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





