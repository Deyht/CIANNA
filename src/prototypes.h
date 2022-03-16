
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
extern int is_cuda_init;
extern int verbose;

//######################################


//######################################
//auxil.c
void init_timing(struct timeval* tstart);
float ellapsed_time(struct timeval tstart);
void init_network(int network_number, int u_input_dim[4], int u_output_dim, float in_bias, int u_batch_size, int u_compute_method, int u_dynamic_load, int u_use_cuda_TC, int no_logo);
Dataset create_dataset(network *net, int nb_elem);
void free_dataset(Dataset *data);
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
void train_network(network* net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, 
	float u_decay, int show_confmat, int save_net, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor);
void forward_testset(network *net, int train_step, int saving, int repeat, int drop_mode);
void train_gan(network* gen, network* disc, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, 
	float u_decay, float gen_disc_learn_rate_ratio, int save_net, int shuffle_gpu, int shuffle_every, int disc_only, float c_TC_scale_factor);


//activations.c
void define_activation(layer *current);
void output_error(layer* current);
void output_deriv_error(layer* current);
void print_activ_param(FILE *f, int type);
void get_string_activ_param(char* activ, int type);
int load_activ_param(char *type);
int set_yolo_params(network *net, int nb_box, int IoU_type, float *prior_w, float *prior_h, float *prior_d, float *yolo_noobj_prob_prior,
	 int nb_class, int nb_param, int strict_box_size, float *scale_tab, float **slopes_and_maxes_tab, float *param_ind_scale, float *IoU_limits, int *fit_parts);

//dense_layer.c
void dense_create(network *net, layer* previous, int nb_neurons, int activation, float drop_rate, int strict_size, FILE *f_load);
void dense_save(FILE *f, layer *current);
void dense_load(network *net, FILE* f);

//conv_layer.c
void conv_create(network *net, layer *previous, int *f_size, int nb_filters, int *stride, int *padding, 
	int *int_padding, int *in_shape, int activation, float drop_rate, FILE *f_load);
void conv_save(FILE *f, layer *current);
void conv_load(network *net, FILE *f);

void pool_create(network *net, layer* previous, int *pool_size, int pool_type, float drop_rate);
void pool_save(FILE *f, layer *current);
void pool_load(network *net, FILE *f);

//initializers.c
float random_uniform(void);
float random_normal(void);
void xavier_normal(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding);
void xavier_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding);


//######################################

#ifdef BLAS
void blas_dense_define(layer *current);
void blas_conv_define(layer *current);

#endif 

void pool_define(layer *current);
void naiv_dense_define(layer *current);
void naiv_conv_define(layer *current);

void max_pooling_fct(void* i_input, void* i_output, int* pool_map,
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int length);
void avg_pooling_fct(void* i_input, void* i_output, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int w_size, int h_size, int d_size, 
	int w_size_out, int h_size_out, int d_size_out, int length);
void deltah_max_pool_cont_fct(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d, 
	int len, int batch_size, int image_size, int w_size, int h_size);
void deltah_avg_pool_cont_fct(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, 
	int pool_size_w, int pool_size_h, int pool_size_d,
	int len, int batch_size, int image_size, int w_size, int h_size);
void dropout_select_pool(int* mask, int size, float drop_rate);
void dropout_apply_pool(void* i_table, int batch_size, int dim, int* mask, int size);

void flat_dense(void* in, void* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void reroll_batch(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void dropout_select_dense(int* mask, int size, float drop_rate);
void dropout_apply_dense(void* table, int batch_size, int dim, int* mask);

void rotate_filter_matrix_fct(void* i_in, void* i_out, int nb_rows, int depth_size, int nb_filters_in, int len);
void dropout_select_conv(int* mask, int size, float drop_rate);
void dropout_apply_conv(void* i_table, int batch_size, int dim, int* mask, int size);
void im2col_fct_v5
	(void* i_output, void* i_input, 
	int image_size, int flat_image_size, 
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int internal_padding_w, int internal_padding_h, int internal_padding_d, 
	int channel, int channel_padding, int image_padding, 
	int batch_size, int f_size_w, int f_size_h, int f_size_d, int flat_f_size, 
	int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias_in, int bias_out) ;

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
extern void *cu_alpha, *cu_beta;
extern void *cu_learning_rate, *cu_momentum;
void set_cu_learning_rate_and_momentum(network* net);
extern float TC_scale_factor;
extern cublasHandle_t cu_handle;
extern cudaDataType cuda_data_type;
#if defined(CUDA_OLD)
extern cudaDataType cuda_compute_type;
#else
extern cublasComputeType_t cuda_compute_type;
#endif
void cuda_master_weight_FP32_to_FP32(float *master, void *copy, int size);
void cuda_master_weight_FP32_to_FP16(float *master, void *copy, int size);
void cuda_master_weight_FP32_to_BF16(float *master, void *copy, int size);
void cuda_update_weights(network* net, void *weights, void* update, int size);

//__global__ void cuda_update_weights_dropout(void *weights, void* update, int size, int *drop_mask, int dim);
__device__ int cuda_argmax(void* tab, int dim_out);
#endif
void init_cuda(network* net);
void cuda_set_TC_scale_factor(network* net, float val);
void cuda_sync(void);
void cuda_free_table(void* tab);
void cuda_create_host_table(network* net, void **tab, int size);
size_t cuda_convert_table(network* net, void **tab, size_t size);
size_t cuda_convert_table_int(int **tab, int size);
void cuda_convert_dataset(network *net, Dataset *data);
void cuda_get_batched_dataset(network *net, Dataset *data);
void cuda_convert_host_dataset(network *net, Dataset *data);
Dataset cuda_create_dataset(network *net, int nb_elem);
void cuda_free_dataset(Dataset *data);
void cuda_create_table_FP32(void **tab, int size);
void cuda_create_table(network* net, void **tab, int size);
void cuda_master_weight_copy(network* net, float *master, void *copy, int size);
void cuda_get_table_FP32(void *cuda_table, void *table, int size);
void cuda_get_table_to_FP32(network* net, void *cuda_table, float *table, int size, void* buffer);
void cuda_get_table(network* net, void *cuda_table, void *table, int size);
void cuda_put_table_FP32(network* net, float *cuda_table, float *table, int size);
void cuda_put_table(network* net, void *cuda_table, void *table, int size);
void cuda_print_table_FP32(void* tab, int size, int return_every);
//void cuda_print_table_4d(network* net, void* tab, int w_size, int h_size, int d_size, int last_dim, int biased);
void cuda_print_table(network* net, void* tab, int size, int return_every);
void cuda_print_table_int(network* net, int* tab, int size, int return_every);
void cuda_print_table_host_FP16(network* net, void* tab, int size, int return_every);
void cuda_print_table_transpose(void* tab, int line_size, int column_size);
void cuda_confmat(network *net, float* mat);
void cuda_perf_eval_init(void);
void cuda_perf_eval_in(void);
float cuda_perf_eval_out(void);
void cuda_shuffle(network *net, Dataset data, Dataset duplicate, int *index_shuffle, int *index_shuffle_device);
void cuda_host_shuffle(network *net, Dataset data, Dataset duplicate);
void cuda_host_only_shuffle(network *net, Dataset data);
void cuda_gan_disc_mix_input(layer *gen_output, layer *disc_input, void* true_input, int half_offset);
void cuda_gan_disc_mix_target(void* mixed_target, void* true_target, int half_offset);
void cuda_create_gan_target(network* net, void* targ, void* true_targ, float frac_ones, int i_half);
void cuda_semi_supervised_gan_deriv_output_error(layer *current, int halved, int reversed);

void cuda_convert_network(layer** network);
void cuda_deriv_output_error(layer *current);
void cuda_output_error_fct(layer* current);
void cuda_output_error(layer* current);
void cuda_compute_error(void** data, void** target_data, int nb_data, float** out_mat, layer *net_layer);


//cuda_activ_functions.cu
void init_typed_cuda_activ(network* net);
void cuda_define_activation(layer *current);
void cuda_deriv_output_error(layer *current);

//cuda_dense_layer.cu
void cuda_dense_init(network* net);
void cuda_dense_define(layer *current);
size_t cuda_convert_dense_layer(layer *current);

//cuda_conv_layer.cu
void cuda_conv_init(network* net);
void cuda_conv_define(layer *current);
size_t cuda_convert_conv_layer(layer *current);

//cuda_pool_layer.cu
void cuda_pool_init(network* net);
void cuda_pool_define(layer *current);
size_t cuda_convert_pool_layer(layer *current);

	
//######################################

#ifdef comp_CUDA
}
#endif

#endif // CUDA


#endif //PROTOTYPES_H





