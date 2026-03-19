
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


#ifndef PROTOTYPES_H
#define PROTOTYPES_H

#include "structs.h"


//######################################
//   Public variables and functions
//######################################

extern network *networks[MAX_NETWORKS_NB];
extern int is_init;
extern int is_cuda_init;
extern int verbose;


//auxil.c
void init_timing(struct timeval *tstart);
float ellapsed_time(struct timeval tstart);
void sig_handler(int signo);
void print_table(float *tab, int column_size, int nb_column);
void print_iter_advance(network *net, int c_batch, int nb_batch, float loss, float c_perf, int is_training);
int argmax(float *tab, int size);
int conv_argmax(float *tab, int offset, int size);
float clip(float n, float lower, float upper);
double random_uniform(void);
double random_normal(void);
void eval_init(network *net);
void perf_eval_in(network *net);
void batch_eval_in(network *net);
void epoch_eval_in(network *net);
void perf_eval_out(network *net, int layer_id, float *vect, int *n_vect);
float batch_eval_out(network *net);
float epoch_eval_out(network *net);
void perf_eval_display(network *net);
void print_architecture_tex(network *net, const char *path, const char *file_name,
	int l_size, int l_in_size, int l_f_size, int l_out_size, int l_stride, int l_padding, 
	int l_in_padding, int l_activation, int l_bias, int l_dropout, int l_param_count);


//dataset.c
Dataset create_dataset(network *net, int with_target, size_t nb_elem);
void host_only_shuffle(network *net, Dataset data);
void free_dataset(Dataset *data);
Dataset* get_dataset_from_type(network *net, const char *dataset_type, int silent);


//network.c
void init_network(int network_number, int u_input_dim[4], int u_output_dim, int u_batch_size, const char *optimizer, int u_wema,
	const char *compute_method_string, int u_dynamic_load, const char *cuda_TC_string, int inference_only, int no_logo, int adv_size);
void train_network(network *net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, 
	float u_decay, float u_weight_decay, int u_decoupled_wdecay, float u_wema_rate, int wema_replace_every, int show_confmat, 
	int save_net, int save_bin, int save_optim_state, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor, int silent);
float* forward_testset(network *net, int saving, int repeat, int drop_mode, int silent, int return_result);
void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat, int silent, float *result);
void set_frozen_layers(network *net, int *tab, int dim);
void fprint_layer_params(network *net, FILE *f, void *params, size_t param_size, size_t return_dim, size_t padding, int stay_on_host, int f_bin);
void save_layer_weights(FILE *f, layer *current, size_t param_size, size_t return_dim, size_t padding, int stay_on_host, int save_optim_state, int f_bin);
void fread_layer_params(network *net, FILE *f, void *params, size_t param_size, size_t return_dim, size_t padding, int f_bin);
void load_layer_weights(FILE *f, layer *current, size_t param_size, size_t return_dim, size_t padding, int save_optim_state, int f_bin);
void save_network(network *net, const char *filename, int save_optim_state, int f_bin);
void load_network(network *net, const char *filename, int iter, int nb_layers, int nb_skip_layers, int f_bin);
void free_network(network *net);


//optimizers.c
void set_optimizer_from_string(network* net, const char* string);
void set_optimizer_param_from_string(network* net, const char* string);
size_t define_optimizer_var(layer *current, size_t param_size);
void set_optimizer_update_function(network *net);
int fread_optim_save_state(FILE *f, network *net, int f_bin, int silent);
void fprint_optimizer_header(FILE *f, network *net, int f_bin);
void save_optimizer_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int stay_on_host, int f_bin);
void load_optimizer_state(FILE *f, layer *current, size_t param_size, size_t return_dim, 
	size_t padding, int f_bin);
void free_optimizer_var(layer *current);


//activ_functions.c
void define_activation_param(layer *current, const char *activ);
void define_activation_fct(layer *current);
void output_error(layer *current);
void output_deriv_error(layer *current);
void fill_string_activ_param(layer *current, char *activ, int no_param);
void print_activ_param(FILE *f, layer *current, int f_bin);
void load_activation_type(layer *current, const char *activ);

int set_yolo_config(network *net, size_t nb_box, int nb_class, int nb_param, int max_nb_obj_per_image, const char *IoU_type_char, 
	const char *prior_dist_type_char, float *prior_size, float *yolo_noobj_prob_prior, int fit_dim, 
	int strict_box_size, int rand_startup, float rand_prob_best_box_assoc, float rand_prob, float min_prior_forced_scaling, float *scale_tab, 
	float **slopes_and_maxes_tab, float *param_ind_scale, float *IoU_limits, int *fit_parts, int class_softmax, 
	int diff_flag, const char *error_type, int no_override, int raw_output);
void free_yolo_params(network *net);


//dense_layer.c
int dense_create(network *net, layer *previous, int nb_neurons, const char *activation, float *bias,
	float drop_rate, int strict_size, const char *init_fct, float init_scaling, FILE *f_load, int load_optim_state, int f_bin);
void dense_save(FILE *f, layer *current, int save_optim_state, int f_bin);
void dense_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer);
void free_dense(layer *current);


//conv_layer.c
int nb_area_comp(int size, int f_size, int padding, int int_padding, int stride);
int conv_create(network *net, layer *previous, int *f_size, size_t nb_filters, size_t nb_groups, int *stride, int *padding, 
	int *int_padding, int *in_shape, const char *activation, float *bias, float drop_rate, 
	const char *init_fct, float init_scaling, FILE *f_load, int load_optim_state, int f_bin);
void conv_save(FILE *f, layer *current, int save_optim_state, int f_bin);
void conv_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer);
void free_conv();


//pool_layer.c
int pool_create(network *net, layer *previous, int *pool_size, int *stride, int *padding, 
	const char *char_pool_type, const char *activation, int global, float drop_rate);
void pool_save(FILE *f, layer *current, int f_bin);
void pool_load(network *net, FILE *f, int f_bin, int skip_layer);
void free_pool(layer *current);


//norm_layer.c
int norm_create(network *net, layer *previous, const char *norm_type, const char *activation, int group_size, int set_off, 
	FILE *f_load, int load_optim_state, int f_bin);
void norm_save(FILE *f, layer *current, int save_optim_state, int f_bin);
void norm_load(network *net, FILE *f, int load_optim_state, int f_bin, int skip_layer);
void free_norm(layer *current);


//lrn_layer.c
int lrn_create(network *net, layer *previous, const char *activation, int range, float k, float alpha, float beta);
void lrn_save(FILE *f, layer *current, int f_bin);
void lrn_load(network *net, FILE *f, int f_bin, int skip_layer);
void free_lrn(layer *current);


//merge_layer.c
int merge_create(network *net, int previous_id_a, int previous_id_b, int merge_type, const char *activation);
void merge_save(FILE *f, layer *current, int f_bin);
void merge_load(network *net, FILE *f, int f_bin, int skip_layer);
void free_merge(layer *current);


//weights_initializers.c
int get_init_type(const char *s_init);
void initialize_weights(const char *init_fct, void *weights, int dim_in, int dim_out, 
	int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);


//naiv_conv_layer.c
void im2col_fct(void* i_output, void* i_input,
	int stride_w, int stride_h ,int stride_d,
	int padding_w, int padding_h, int padding_d,
	int internal_padding_w, int internal_padding_h, int internal_padding_d,
	int f_size_w, int f_size_h, int f_size_d,
	size_t w_size, size_t h_size, size_t d_size,
	size_t nb_area_w, size_t nb_area_h, size_t nb_area_d,
	size_t in_nb_channels, size_t in_group_size, size_t in_image_size, size_t in_image_offset, size_t in_channel_offset,
	size_t out_image_offset, size_t out_group_offset,
	int batch_size, int bias_out);
void rotate_filter_matrix_fct(void* i_in, void* i_out, size_t nb_rows, size_t depth_size, size_t out_group_size, size_t len);
void dropout_select_conv(float *mask, size_t size, float drop_rate);
void dropout_apply_conv(void *i_table, float *mask, size_t size);
void dropout_scale_conv(void *i_table, size_t size, float drop_rate);
void naiv_conv_define(layer *current);

//naiv_dense_layer.c
void flat_dense(void *in, void *out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void reroll_batch(void *in, void *out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void dropout_select_dense(float *mask, int biased_dim, size_t size, float drop_rate);
void dropout_apply_dense(void *table, float *mask, size_t size);
void dropout_scale_dense(void *table, int biased_dim, size_t size, float drop_rate);
void naiv_dense_define(layer *current);

//naiv_norm_layer.c
//most functions are private as not required outside the norm layer file
void norm_define(layer *current);

//naiv_lrn_layer.c
//most functions are private as not required outside the lrn layer file
void lrn_define(layer *current);

//naiv_merge_layer.c
//most functions are private as not required outside the merge layer file
void merge_define(layer *current);

//naiv_pool_layer.c
void max_pooling_fct(void *i_input, void *i_output, int *pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
void avg_pooling_fct(void *i_input, void *i_output, int *pool_map,int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
void deltah_max_pool_cont_fct(void *i_delta_o, void *i_delta_o_unpool, int *pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, size_t length);
void deltah_avg_pool_cont_fct(void *i_delta_o, void *i_delta_o_unpool, int *pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, size_t length);
void dropout_select_pool(float *mask, size_t size, float drop_rate);
void dropout_apply_pool(void *i_table, float *mask, size_t size);
void dropout_scale_pool(void *i_table, size_t size, float drop_rate);
void pool_define(layer *current);



#ifdef BLAS
void blas_dense_define(layer *current);
void blas_conv_define(layer *current);
#endif 


#ifdef CUDA
//######################################
//       CUDA public prototypes
//######################################

#ifdef comp_CUDA
//When compiled by nvcc, variables and global functions must be exported as regular C prototypes
//so the act as regular C prototypes when linked by gcc
extern "C"
{
//cuda_auxil.cu
extern int cu_threads;
extern void *cu_alpha, *cu_beta;
extern void *cu_learning_rate, *cu_momentum;
extern float TC_scale_factor;
extern cublasHandle_t cu_handle;
extern cudaDataType cuda_data_type;

#if defined(CUDA_OLD)
extern cudaDataType cuda_compute_type;
#else
extern cublasComputeType_t cuda_compute_type;
#endif

__global__ void cuda_master_weight_FP32_to_FP32(float *master, void *copy, size_t size);
__global__ void cuda_master_weight_FP32_to_FP16(float *master, void *copy, size_t size);
__global__ void cuda_master_weight_FP32_to_BF16(float *master, void *copy, size_t size);
__global__ void init_block_state(unsigned int seed,  curandState_t *states, size_t size);

#endif

void set_cuda_batched_scaled(network *net);
void cuda_set_TC_scale_factor(network *net, float val);
void cuda_sync(void);
void cuda_free_table(void *tab);
void cuda_random_vector(float *tab, size_t size);
void cuda_create_host_table(network *net, void **tab, size_t size);
size_t cuda_convert_table(network *net, void **tab, size_t size, int keep_host);
size_t cuda_convert_table_FP32(void **tab, size_t size, int keep_host);
size_t cuda_convert_table_int(int **tab, size_t size, int keep_host);
void cuda_float_memset(void* table, float value, size_t size);
void cuda_typed_memset_FP32(void* i_table, int value, size_t size);
void cuda_create_table_FP32(void **tab, size_t size);
void cuda_get_table_FP32_to_FP32(void *cuda_table, float *table, size_t size, void *buffer);
void cuda_create_table(network *net, void **tab, size_t size);
void cuda_get_table_to_FP32(network *net, void *cuda_table, float *table, size_t size, void *buffer);
void cuda_get_table_FP32(void *cuda_table, void *table, size_t size);
void cuda_get_table(network *net, void *cuda_table, void *table, size_t size);
void cuda_get_typed_host_table(network *net, void *typed_table, float *out_table, size_t size);
void cuda_put_table_FP32(void *cuda_table, void *table, size_t size);
void cuda_put_table(network *net, void *cuda_table, void *table, size_t size);
void cuda_host_FP32_to_device_typed_inplace_copy(network *net, void *cuda_table, float *host_table, float *gpu_buffer, size_t size);
void cuda_convert_dataset(network *net, Dataset *data);
void cuda_get_batched_dataset(network *net, Dataset *data);
void cuda_convert_host_dataset(network *net, Dataset *data);
Dataset cuda_create_dataset(network *net, int with_target, size_t nb_elem);
void cuda_free_dataset(Dataset *data);
void cuda_master_weight_copy(network *net, float *master, void *copy, size_t size);
void cuda_print_table_FP32(void *tab, size_t size, int return_every);
void cuda_print_table(network *net, void *tab, size_t size, int return_every);
void cuda_print_table_int(network *net, int *tab, size_t size, int return_every);
void cuda_print_table_host_FP16(network *net, void *tab, size_t size, int return_every);
void cuda_perf_eval_init(void);
void cuda_batch_eval_init(void);
void cuda_epoch_eval_init(void);
void cuda_perf_eval_in(void);
void cuda_batch_eval_in(void);
void cuda_epoch_eval_in(void);
float cuda_perf_eval_out(void);
float cuda_batch_eval_out(void);
float cuda_epoch_eval_out(void);
void cuda_shuffle(network *net, Dataset data, Dataset duplicate, int *index_shuffle, int *index_shuffle_device);
void cuda_host_shuffle(network *net, Dataset data, Dataset duplicate);
void cuda_host_only_shuffle(network *net, Dataset data);
void init_cuda(network *net);
void free_cuda_network(void);


//cuda_optimizes.cu
void init_typed_cuda_optimizer(network* net);
size_t cuda_convert_optimizer_var(layer *current, size_t param_size);
void cuda_set_optimizer_update_function(network *net);
void cuda_free_optimizer_var(layer *current);


//cuda_activ_functions.cu
void init_typed_cuda_activ(network *net);
void cuda_define_activation_fct(layer *current);
void cuda_deriv_output_error(layer *current);
void cuda_output_error_fct(layer *current);
void cuda_free_yolo_activ_param(layer *current);


//cuda_dense_layer.cu
void cuda_dense_init(network *net);
size_t cuda_convert_dense_layer(layer *current);
void cuda_free_dense(layer *current);
void cuda_dense_define(layer *current);


//cuda_conv_layer.cu
void cuda_conv_init(network *net);
size_t cuda_convert_conv_layer(layer *current);
void cuda_free_conv(layer *current);
void cuda_conv_define(layer *current);


//cuda_pool_layer.cu
void cuda_pool_init(network *net);
size_t cuda_convert_pool_layer(layer *current);
void cuda_free_pool(layer *current);
void cuda_pool_define(layer *current);


//cuda_norm_layer.cu
void cuda_norm_init(network *net);
size_t cuda_convert_norm_layer(layer *current);
void cuda_free_norm(layer *current);
void cuda_norm_define(layer *current);


//cuda_lrn_layer.cu
void cuda_lrn_init(network *net);
size_t cuda_convert_lrn_layer(layer *current);
void cuda_free_lrn(layer *current);
void cuda_lrn_define(layer *current);


//cuda_merge_layer.cu
void cuda_merge_init(network *net);
size_t cuda_convert_merge_layer(layer *current);
void cuda_free_merge(layer *current);
void cuda_merge_define(layer *current);


#ifdef comp_CUDA
}
#endif

#endif // CUDA


#endif //PROTOTYPES_H





