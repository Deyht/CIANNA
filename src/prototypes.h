

/*
	Copyright (C) 2024 David Cornu
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
void init_network(int network_number, int u_input_dim[4], int u_output_dim, float in_bias, int u_batch_size, const char* compute_method_string, int u_dynamic_load, 
	const char* cuda_TC_string, int inference_only, int no_logo, int adv_size);
Dataset create_dataset(network *net, int nb_elem);
void free_dataset(Dataset *data);
float clip(float n, float lower, float upper);
void print_table(float* tab, int column_size, int nb_column);
void write_formated_dataset(network *net, const char *filename, Dataset *data, int input_data_type, int output_data_type);
Dataset load_formated_dataset(network *net, const char *filename, int input_data_type, int output_data_type);
void set_normalize_dataset_parameters(network *net, float *offset_input, float *norm_input, int dim_size_input, float *offset_output, float *norm_output, int dim_size_output);
void normalize_dataset(network *net, Dataset c_data);
void update_weights(void *weights, void* update, float weight_decay, int is_pivot, int size);
void perf_eval_display(network *net);
void print_architecture_tex(network *net, const char *path, const char *file_name,
	int l_size, int l_in_size, int l_f_size, int l_out_size, int l_stride, int l_padding, 
	int l_in_padding, int l_activation, int l_bias, int l_dropout, int l_param_count);
void compute_error(network *net, Dataset data, int saving, int confusion_matrix, int repeat, int silent);
void save_network(network *net, const char *filename, int f_bin);
void load_network(network *net, const char *filename, int epoch, int nb_layers, int f_bin);
void set_frozen_layers(network *net, int* tab, int dim);
void train_network(network* net, int nb_epochs, int control_interv, float u_begin_learning_rate, float u_end_learning_rate, float u_momentum, 
	float u_decay, float u_weight_decay, int show_confmat, int save_net, int save_bin, int shuffle_gpu, int shuffle_every, float c_TC_scale_factor, int silent);
void forward_testset(network *net, int saving, int repeat, int drop_mode, int silent);


//activations.c
void define_activation(layer *current);
void output_error(layer* current);
void output_deriv_error(layer* current);
void print_activ_param(FILE *f, layer *current, int f_bin);
void print_norm_param(FILE *f, layer *current, int f_bin);
void print_string_activ_param(layer *current, char* activ);
void print_string_activ(layer *current, char* activ);
void print_string_norm_param(layer *current, char* norm);
void load_activ_param(layer *current, const char *activ);
void load_norm_param(layer *current, const char *norm);

void set_linear_activ(layer *current, int size, int dim, int biased_dim, int offset);
void set_relu_activ(layer *current, int size, int dim, int biased_dim, int offset, const char *activ);
void set_logistic_activ(layer *current, int size, int dim, int biased_dim, int offset, const char *activ);
void set_softmax_activ(layer *current, int size, int dim, int biased_dim, int offset);
void set_yolo_activ(layer *current);
int set_yolo_params(network *net, size_t nb_box, int nb_class, int nb_param, int max_nb_obj_per_image, const char *IoU_type_char, 
	const char *prior_dist_type_char, float *prior_size, float *yolo_noobj_prob_prior, int fit_dim, 
	int strict_box_size, int rand_startup, float rand_prob_best_box_assoc, float rand_prob, float min_prior_forced_scaling, float *scale_tab, 
	float **slopes_and_maxes_tab, float *param_ind_scale, float *IoU_limits, int *fit_parts, int class_softmax, 
	int diff_flag, const char* error_type, int no_override, int raw_output);

//dense_layer.c
int dense_create(network *net, layer* previous, int nb_neurons, const char *activation, float *bias,
	float drop_rate, int strict_size, const char *init_fct, float init_scaling, FILE *f_load, int f_bin);
void dense_save(FILE *f, layer *current, int f_bin);
void dense_load(network *net, FILE* f, int f_bin);

//conv_layer.c
int nb_area_comp(int size, int f_size, int padding, int int_padding, int stride);
int conv_create(network *net, layer *previous, int *f_size, int nb_filters, int *stride, 
	int *padding, int *int_padding, int *in_shape, const char* activation, float *bias, 
	float drop_rate, const char *init_fct, float init_scaling, FILE *f_load, int f_bin);
void conv_save(FILE *f, layer *current, int f_bin);
void conv_load(network *net, FILE *f, int f_bin);

//pool_layer.c
int pool_create(network *net, layer *previous, int *pool_size, int* stride, int *padding, 
	const char *char_pool_type, const char *activation, int global, float drop_rate);
void pool_save(FILE *f, layer *current, int f_bin);
void pool_load(network *net, FILE *f, int f_bin);

//norm_layer.c
int norm_create(network *net, layer *previous, const char *norm_type, const char *activation, int group_size, int set_off, FILE *f_load, int f_bin);
void norm_save(FILE *f, layer *current, int f_bin);
void norm_load(network *net, FILE *f, int f_bin);

//lrn_layer.c
int lrn_create(network *net, layer *previous, const char *activation, int range, float k, float alpha, float beta, FILE *f_load, int f_bin);
void lrn_save(FILE *f, layer *current, int f_bin);
void lrn_load(network *net, FILE *f, int f_bin);

//initializers.c
int get_init_type(const char *s_init);
double random_uniform(void);
double random_normal(void);
void xavier_normal( void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void xavier_uniform(void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void lecun_normal(  void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void lecun_uniform( void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void rand_normal(   void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);
void rand_uniform(  void *tab, int dim_in, int dim_out, int bias_padding, float bias_padding_value, int zero_padding, float manual_scaling);

//######################################

#ifdef BLAS
void blas_dense_define(layer *current);
void blas_conv_define(layer *current);

#endif 

void pool_define(layer *current);
void naiv_dense_define(layer *current);
void naiv_conv_define(layer *current);
void naiv_norm_define(layer *current);

void max_pooling_fct(void* i_input, void* i_output, int* pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
void avg_pooling_fct(void* i_input, void* i_output, int* pool_map,int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
void deltah_max_pool_cont_fct(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, size_t length);
void deltah_avg_pool_cont_fct(void* i_delta_o, void* i_delta_o_unpool, int* pool_map, int pool_size_w, int pool_size_h, int pool_size_d, 
	int stride_w, int stride_h ,int stride_d, int padding_w, int padding_h, int padding_d, 
	int w_size, int h_size, int d_size, int w_size_out, int h_size_out, int d_size_out, size_t length);
void dropout_select_pool(float* mask, size_t size, float drop_rate);
void dropout_apply_pool(void* i_table, float* mask, size_t size);
void dropout_scale_pool(void* i_table, size_t size, float drop_rate);

void flat_dense(void* in, void* out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void reroll_batch(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
void dropout_select_dense(float* mask, int biased_dim, size_t size, float drop_rate);
void dropout_apply_dense(void* table, float* mask, size_t size);
void dropout_scale_dense(void *table, int biased_dim, size_t size, float drop_rate);

void rotate_filter_matrix_fct(void* i_in, void* i_out, int nb_rows, int depth_size, int nb_filters_in, int len);
void dropout_select_conv(float* mask, size_t size, float drop_rate);
void dropout_apply_conv(void* i_table, float* mask, size_t size);
void dropout_scale_conv(void* i_table, size_t size, float drop_rate);
void im2col_fct
	(void* i_output, void* i_input, int image_size, int flat_image_size, 
	int stride_w, int stride_h ,int stride_d, 
	int padding_w, int padding_h, int padding_d, 
	int internal_padding_w, int internal_padding_h, int internal_padding_d, 
	int channel, int channel_padding, int image_padding, int batch_size, 
	int f_size_w, int f_size_h, int f_size_d, int flat_f_size, 
	int w_size, int h_size, int d_size, 
	int nb_area_w, int nb_area_h, int nb_area_d, int bias_in, int bias_out);

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
void cuda_master_weight_FP32_to_FP32(float *master, void *copy, size_t size);
void cuda_master_weight_FP32_to_FP16(float *master, void *copy, size_t size);
void cuda_master_weight_FP32_to_BF16(float *master, void *copy, size_t size);
void cuda_update_weights(network* net, void *weights, void* update, float weight_decay, int is_pivot, size_t size);
__global__ void init_block_state(unsigned int seed,  curandState_t* states, size_t size);

#endif

void init_cuda(network* net);
void cuda_set_TC_scale_factor(network* net, float val);
void cuda_sync(void);
void cuda_free_table(void* tab);
void cuda_create_host_table(network* net, void **tab, size_t size);
size_t cuda_convert_table(network* net, void **tab, size_t size, int keep_host);
size_t cuda_convert_table_FP32(void **tab, size_t size, int keep_host);
size_t cuda_convert_table_int(int **tab, size_t size, int keep_host);
void cuda_convert_dataset(network *net, Dataset *data);
void cuda_get_batched_dataset(network *net, Dataset *data);
void cuda_convert_host_dataset(network *net, Dataset *data);
Dataset cuda_create_dataset(network *net, int nb_elem);
void cuda_free_dataset(Dataset *data);
void cuda_create_table_FP32(void **tab, size_t size);
void cuda_create_table(network* net, void **tab, size_t size);
void cuda_set_mem_value(void* device_mem_loc, float value, size_t size);
void cuda_master_weight_copy(network* net, float *master, void *copy, size_t size);
void cuda_get_table_FP32(void *cuda_table, void *table, size_t size);
void cuda_get_table_to_FP32(network* net, void *cuda_table, float *table, size_t size, void* buffer);
void cuda_get_table(network* net, void *cuda_table, void *table, size_t size);
void cuda_put_table_FP32(void *cuda_table, void *table, size_t size);
void cuda_get_typed_host_table(network* net, void *typed_table, float *out_table, size_t size);
void cuda_put_table(network* net, void *cuda_table, void *table, size_t size);
void cuda_print_table_FP32(void* tab, size_t size, int return_every);
void cuda_print_table(network* net, void* tab, size_t size, int return_every);
void cuda_print_table_int(network* net, int* tab, size_t size, int return_every);
void cuda_print_table_host_FP16(network* net, void* tab, size_t size, int return_every);
void cuda_print_table_transpose(void* tab, int line_size, int column_size);
void cuda_confmat(network *net, float* mat);
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
void cuda_random_vector(float* tab, size_t size);

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

//cuda_norm_layer.cu
void cuda_norm_init(network* net);
void cuda_norm_define(layer *current);
size_t cuda_convert_norm_layer(layer *current);

//cuda_lrn_layer.cu
void cuda_lrn_init(network* net);
void cuda_lrn_define(layer *current);
size_t cuda_convert_lrn_layer(layer *current);
	
//######################################

#ifdef comp_CUDA
}
#endif

#endif // CUDA


#endif //PROTOTYPES_H





