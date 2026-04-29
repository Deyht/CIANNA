
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


#ifndef STRUCTS_H
#define STRUCTS_H

#include "defs.h"


//############################################
//            Various Enumerations
//############################################

enum layer_type_enum{CONV, POOL, DENSE, NORM, LRN, GRN, MERGE};
enum output_type_enum{SPATIAL, FLAT};
enum activation_functions_enum{RELU, LOGISTIC, SOFTMAX, YOLO, LINEAR};
enum optimizer_enum{SGD, ADAM, RMS_PROP};
enum initializers_enum{N_XAVIER, U_XAVIER, N_LECUN, U_LECUN, U_RAND, N_RAND};
enum inference_modes_enum{AVG_MODEL, MC_MODEL};
enum batch_param_enum{OFF, SINGLE, FULL};
enum compute_method_enum{C_NAIV, C_BLAS, C_CUDA};
enum memory_localization_enum{NO_LOC, HOST, DEVICE};
enum IoU_types_enum{IOU, GIOU, DIOU, DIOU2};
enum pool_types_enum{MAX_pool, AVG_pool};
enum merge_types_enum{ADD_merge, CONCAT_merge};
enum yolo_error_type_enum{ERR_COMPLETE, ERR_NATURAL};
enum prior_dist_type_enum{DIST_IOU, DIST_SIZE, DIST_OFFSET};

typedef struct Dataset Dataset;
typedef struct layer layer;
typedef struct network network;

typedef struct dense_param dense_param;
typedef struct conv_param conv_param;
typedef struct pool_param pool_param;
typedef struct norm_param norm_param;
typedef struct lrn_param lrn_param;
typedef struct grn_param grn_param;
typedef struct merge_param merge_param;

typedef struct linear_param linear_param;
typedef struct ReLU_param ReLU_param;
typedef struct logistic_param logistic_param;
typedef struct softmax_param softmax_param;
typedef struct yolo_param yolo_param;

typedef struct sgd_param sgd_param;
typedef struct sgd_var sgd_var;
typedef struct adam_param adam_param;
typedef struct adam_var adam_var;
typedef struct rmsprop_param rmsprop_param;
typedef struct rmsprop_var rmsprop_var;


//############################################
//             CUDA specific
//############################################

#ifdef CUDA

enum GPU_type{FP32, FP16, BF16};
enum TC_comp_mode{FP32C_FP32A, TF32C_FP32A, FP16C_FP32A, FP16C_FP16A, BF16C_FP32A};

typedef struct cuda_net_instance cuda_net_instance;
typedef struct cuda_auxil_fcts cuda_auxil_fcts;
typedef struct cuda_optimizer_fcts cuda_optimizer_fcts;
typedef struct cuda_dense_fcts cuda_dense_fcts;
typedef struct cuda_conv_fcts cuda_conv_fcts;
typedef struct cuda_pool_fcts cuda_pool_fcts;
typedef struct cuda_norm_fcts cuda_norm_fcts;
typedef struct cuda_lrn_fcts cuda_lrn_fcts;
typedef struct cuda_grn_fcts cuda_grn_fcts;
typedef struct cuda_merge_fcts cuda_merge_fcts;

typedef struct cuda_linear_activ_fcts cuda_linear_activ_fcts;
typedef struct cuda_ReLU_activ_fcts cuda_ReLU_activ_fcts;
typedef struct cuda_logistic_activ_fcts cuda_logistic_activ_fcts;
typedef struct cuda_softmax_activ_fcts cuda_softmax_activ_fcts;
typedef struct cuda_YOLO_activ_fcts cuda_YOLO_activ_fcts;


struct cuda_auxil_fcts
{
	void (*cu_create_host_table_fct)(void **tab, size_t size);
	size_t (*cu_convert_table_fct)(void **tab, size_t size, int keep_host);
	void (*cu_create_table_fct)(void **tab, size_t size);
	void (*cu_typed_memset_fct)(void *i_table, int value, size_t size);
	void (*cu_get_table_fct)(void *cuda_table, void *table, size_t size);
	void (*cu_get_typed_host_table_fct)(void *typed_table, float *out_table, size_t size);
	void (*cu_get_table_to_FP32_fct)(void *cuda_table, float *table, size_t size, void *buffer);
	void (*cu_put_table_fct)(void *cuda_table, void *table, size_t size);
	void (*cu_host_FP32_to_device_typed_inplace_copy_kernel)(void *cuda_table, float *buffer, size_t size);
	void (*cu_convert_batched_table_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	Dataset (*cu_create_dataset_fct)(network *net, int with_target, size_t nb_elem);	
	void (*cu_get_batched_table_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	void (*cu_convert_batched_host_table_FP32_to_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	void (*cu_master_weight_copy_kernel)(float *master, void *copy, size_t size);
	void (*cu_print_table_fct)(void *tab, size_t size, int return_every);
	void (*cu_shfl_kern_fct)(void **i_in, void **i_targ, void **i_train_dupl, void **i_targ_dupl,
		int *index, size_t in_size, int b_size, int d_in, int d_out);
	void (*cu_get_back_shuffle_fct)(void **i_in, void **i_targ, void **i_train_dupl, void **i_targ_dupl,
		size_t in_size, int b_size, int d_in, int d_out);
	void (*cu_host_shuffle_fct)(network *net, Dataset data, Dataset duplicate);
	void (*cu_host_only_shuffle_fct)(network *net, Dataset data);
};


struct cuda_optimizer_fcts
{
	void (*sgd_fct)(float *weights, float *ema_weights, void *gradient, float weight_decay, int decoupled_wdecay, 
		float wema_rate, float learning_rate, int batch_size, float momentum, float *velocity, 
		size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor);
	void (*adam_fct)(float *weights, float *ema_weights, void *gradient, float weight_decay, int decoupled_wdecay,
	 	float wema_rate, float learning_rate, int batch_size, float beta_1, float beta_2, float eps, int ams_grad, 
		int opt_step, float *first_mom, float *second_mom, float *max_second_mom,
		size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor);
	void (*rmsprop_fct)(float *weights, float *ema_weights, void *gradient, float weight_decay, int decoupled_wdecay, 
		float wema_rate, float learning_rate, int batch_size, float alpha, float momentum, float eps, 
		int centered, float *sqrt_avg, float *velocity, float *grad_avg,
		size_t bias_weight_offset, size_t flat_dim_offset, size_t size, float TC_scale_factor);
};

struct cuda_dense_fcts
{
	void (*flat_dense_fct)(void *i_in, void *i_out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, size_t size);
	void (*flat_dense_back_fct)(void* i_in, void* i_out, int map_size, int flatten_size, int nb_map, int batch_size, size_t size);
	void (*reroll_fct)(void *in, void *out, int map_size, int flatten_size, int nb_map, int batch_size, size_t size);
	void (*drop_apply_fct)(void *i_table, float *mask, size_t size, int biased_dim, float drop_rate);
	void (*drop_scale_fct)(void *i_table, float *mask, size_t size, int biased_dim, float drop_rate);
	void (*set_input_bias)(void *i_table, size_t unbiased_dim, float bias_value, size_t size);
};


struct cuda_conv_fcts
{
	void (*im2col_fct)(void* i_output, void* i_input,
		int stride_w, int stride_h, int stride_d,
		int padding_w, int padding_h, int padding_d,
		int internal_padding_w, int internal_padding_h, int internal_padding_d,
		int f_size_w, int f_size_h, int f_size_d,
		size_t w_size, size_t h_size, size_t d_size,
		size_t nb_area_w, size_t nb_area_h, size_t nb_area_d,
		size_t in_nb_channels, size_t in_group_size, size_t in_image_size, size_t in_image_offset, size_t in_channel_offset,
		size_t out_image_offset, size_t out_group_offset, int TC_padding,
		int batch_size, int bias_out);
	
	void (*drop_apply_fct)(void *i_table, float *mask, size_t size, float drop_rate);
	void (*drop_scale_fct)(void *i_table, float *mask, size_t size, float drop_rate);
	void (*rotate_filter_fct)(void* i_in, void* i_out, size_t nb_rows, size_t depth_size, size_t out_group_size, int TC_padding, size_t len);
};


struct cuda_merge_fcts
{
	void (*merge_add)(void *output_a, void *output_b, void *output_new, size_t size);
	void (*merge_add_back)(void *delta_o_a, void *delta_o_b, void *delta_o, size_t size);
	void (*merge_concatenate)(void *output_a, void *output_b, size_t size_a, size_t size_b, void *output_new, size_t size);
	void (*merge_concatenate_back)(void *delta_o_a, void *delta_o_b, size_t size_a, size_t size_b, void *delta_o, size_t size);
};


struct cuda_pool_fcts
{
	void (*max_pool_fct)(void *i_input, void *i_output, int *pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
	void (*avg_pool_fct)(void *i_input, void *i_output, int *pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
	void (*max_deltah_pool_fct)(void *i_delta_o, void *i_delta_o_unpool, int *pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, size_t length);
	void (*avg_deltah_pool_fct)(void *i_delta_o, void *i_delta_o_unpool, int *pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, size_t length);
	void (*drop_apply_fct)(void *i_table, float *mask, size_t size, float drop_rate);
	void (*drop_scale_fct)(void *i_table, float *mask, size_t size, float drop_rate);
};


struct cuda_norm_fcts
{
	void (*cu_reduce_group_mean_conv_kernel)(void *idata, float *group_mean,
		size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div, size_t sum_size);
	void (*cu_reduce_group_var_conv_kernel)(void *idata, float *group_var, float *group_mean,
		size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_div, size_t sum_size);
	void (*cu_reduce_norm_dbeta_conv_kernel)(void *i_d_output, void *i_d_beta,
		size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size);
	void (*cu_reduce_group_dgamma_conv_kernel)(void *idata, void *d_output, void *i_d_gamma,
		float *group_var, float *group_mean, size_t group_size, size_t nb_group, size_t flat_a_size, size_t batch_size, size_t sum_size);
	void (*cu_reduce_norm_AB_kernel)(void *i_d_gamma, void *i_d_beta, float *gamma, 
		float *A, float *B, size_t group_size, size_t nb_group);
	void (*cu_reduce_norm_param_grads_kernel)(void *i_d_gamma, void *i_d_beta, 
		void *i_gamma_grad, void *i_beta_grad, size_t nb_features, size_t batch_size);
	void (*cu_group_normalization_conv_kernel)(void *i_output, void *i_input, float *gamma, float *beta, float *group_mean,
		float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t nb_filters, size_t flat_a_size);
	void (*cu_group_normalization_conv_back_kernel)(void *i_input, void *i_d_output, 
		void *i_d_input, float *gamma, float *A, float *B, float *group_mean,
		float *group_var, size_t b_length, size_t b_size, size_t group_size, size_t nb_group, size_t nb_filters, size_t flat_a_size);
};


struct cuda_lrn_fcts
{
	void (*cu_lrn_conv_kernel)(void *i_output, void *i_input, float *local_scale, int range, 
		float k, float alpha, float beta, int b_size, int nb_channel, size_t flat_a_size);
	void (*cu_lrn_conv_back_kernel)(void *i_output, void *i_input, void *i_d_output, void *i_d_input,
		float *local_scale, int range, float k, float alpha, float beta, int b_size, int nb_channel, size_t flat_a_size);
};


struct cuda_grn_fcts
{
	void (*cu_reduce_l2norm_conv_kernel)(void *idata, float *group_l2norm,
		size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size);
	void (*cu_reduce_grn_dgamma_conv_kernel)(void *idata, void *i_d_output, void *d_gamma,
		size_t nb_features, size_t flat_a_size, size_t batch_size, size_t sum_size);
	void (*cu_reduce_grn_param_grads_kernel)(void *i_d_gamma, void *i_d_beta, float *relative_importance, 
		void *i_gamma_grad, void *i_beta_grad, size_t nb_features, size_t batch_size);
	void (*cu_reduce_grn_dbeta_conv_kernel)(void *i_d_output, void *i_d_beta, size_t nb_features, 
		size_t flat_a_size, size_t batch_size, size_t sum_size);
	void (*cu_grn_conv_kernel)(void *i_output, void *i_input, float *gamma, float *beta,
		float *relative_importance, int residual, size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size);
	void (*cu_grn_conv_back_kernel)(void *i_input, void *i_d_output, void *i_d_input,
		float *gamma, void *i_d_gamma, float *feature_norm, float *relative_importance, int residual,
		size_t b_length, size_t b_size, size_t nb_features, size_t flat_a_size);
};


struct cuda_linear_activ_fcts
{
	void (*activ_fct)(void *i_tab, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
	void (*deriv_fct)(void *i_deriv, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
};


struct cuda_ReLU_activ_fcts
{
	void (*activ_fct)(void *i_tab, size_t dim, size_t biased_dim, size_t offset, 
		float saturation, float leaking_factor, size_t length, size_t size);
	void (*deriv_fct)(void *i_deriv, void *i_value, size_t dim, size_t biased_dim, 
		size_t offset, float saturation, float leaking_factor, size_t length, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
};


struct cuda_logistic_activ_fcts
{
	void (*activ_fct)(void *i_tab, float beta, float saturation, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
	void (*deriv_fct)(void *i_deriv, void *i_value, float beta, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t offset, size_t length, size_t size);
};


struct cuda_softmax_activ_fcts
{
	void (*activ_fct)(void *i_tab, size_t dim, size_t biased_dim, size_t offset, size_t length, size_t batch_size, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		size_t dim, size_t offset, size_t biased_dim, size_t length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		size_t dim, size_t biased_dim, size_t length, size_t offset, size_t size);
};


struct cuda_YOLO_activ_fcts
{
	void (*activ_fct)(void *i_tab, size_t flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, 
		size_t flat_target_size, size_t flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, size_t size, float TC_scale_factor, size_t nb_im_iter);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, 
		size_t flat_target_size, size_t flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, size_t size);
};


struct cuda_net_instance
{
	int dynamic_load;
	int use_cuda_TC;
	void *output_error_cuda;

	cuda_auxil_fcts cu_auxil_fcts;
	cuda_optimizer_fcts cu_optimizer_fcts;

	cuda_dense_fcts cu_dense_fcts;
	cuda_conv_fcts cu_conv_fcts;
	cuda_pool_fcts cu_pool_fcts;
	cuda_norm_fcts cu_norm_fcts;
	cuda_lrn_fcts cu_lrn_fcts;
	cuda_grn_fcts cu_grn_fcts;
	cuda_merge_fcts cu_merge_fcts;
	
	cuda_linear_activ_fcts cu_linear_activ_fcts;
	cuda_ReLU_activ_fcts cu_ReLU_activ_fcts;
	cuda_logistic_activ_fcts cu_logistic_activ_fcts;
	cuda_softmax_activ_fcts cu_softmax_activ_fcts;
	cuda_YOLO_activ_fcts cu_YOLO_activ_fcts;
};

#endif



//############################################
//                  Global
//############################################

struct Dataset
{
	void **input;
	void **target;
	void **input_device;
	void **target_device;
	void (*cont_copy)(float *elem_in, void *elem_out, 
		int out_offset, size_t nb_elem);
	int size;
	int nb_batch;
	int localization;
};


struct layer
{
	network *c_network;
	
	int type;
	int activation_type;
	
	int output_type;
	int *output_dim;
	int *prev_dim;
	
	void *param;
	void *input; //usually contains address of previous->output
	void *output;
	void *delta_o;
	layer *previous;
	
	void *weights;
	float *FP32_weights;
	float *ema_weights;
	void *gradient;
	void *optimizer_var;
	int frozen;
	int wema_replace_signal;
	float bias_value;
	float dropout_rate;
	float *dropout_mask;
	
	//Activation related precomputed sizes
	size_t a_size;
	size_t a_dim;
	size_t a_biased_dim;
	size_t a_offset;
	
	void (*forward)(layer *current);
	void (*backprop)(layer *current);
	
	void (*activation)(layer *current);
	void (*deriv_activation)(layer *current);
	void *activ_param;
	
	//utility
	size_t nb_params;
	float time_fwd;
	float time_back;
};


struct network
{
	layer *net_layers[MAX_LAYERS_NB];

	int id;
	int compute_method;
	int inference_only;
	int nb_layers;
	
	float learning_rate;
	float lr_decay;
	float weight_decay;
	int decoupled_wdecay;
	int use_wema;
	float wema_rate;
	int wema_replace_every;
	int optimizer;
	void *optimizer_param;
	void (*optim_update_fct)(layer *current, size_t bias_weight_offset, 
		size_t flat_dim_offset, size_t nb_weights);
	void (*optim_update_fct_gpu)(layer *current, size_t bias_weight_offset, 
		size_t flat_dim_offset, size_t nb_weights);
	
	Dataset train, test, valid;
	Dataset train_buf, test_buf, valid_buf;
	
	int in_dims[4];
	int skip_in_dims[4];
	size_t input_dim; // flat size
	size_t output_dim; //Correspond to the "target size"
	size_t out_size; //Correspond to the actual ouput size with paddings if needed
	int batch_size;
	int batch_param;
	int iter;
	
	int is_inference;
	int inference_drop_mode;
	int no_error;
	int perf_eval;
	float *fwd_perf, *back_perf;
	int *fwd_perf_n, *back_perf_n;
	long long int total_nb_param;
	long long int memory_footprint;
	int adv_size;
	
	void *input;
	void *target;
	int length;
	void *output_error;
	
	//Possible yolo_param
	yolo_param *y_param;

	float TC_scale_factor;
	#ifdef CUDA
	cuda_net_instance cu_inst;
	#endif

};


//############################################
//               Various Layers
//############################################

struct dense_param
{
	void *flat_input;
	void *flat_delta_o;
};


struct conv_param
{
	int *f_size;
	int *stride;
	int *padding;
	int *int_padding;
	
	int nb_groups;
	int TC_padding;

	void *im2col_input;
	void *im2col_delta_o;
	
	void *rotated_filters;
	void *temp_delta_o;
};


struct pool_param
{
	int *p_size;
	int *stride;
	int *padding;
	int pool_type;
	int global;
	
	int *pool_map;
	void *temp_delta_o;
};


struct norm_param
{
	int group_size;
	int nb_group;
	
	float *mean;
	float *var;
	float *gamma;
	float *beta;
	void *gamma_grad;
	void *beta_grad;
	void *d_gamma;
	void *d_beta;
	float *temp_A;
	float *temp_B;
};


struct lrn_param
{
	int range;
	float k;
	float alpha;
	float beta;
	
	float *local_scale;
};


struct grn_param
{
	int residual;
	float *feature_norm;
	float *relative_importance;
	float *mean;
	
	float *gamma;
	float *beta;
	void *gamma_grad;
	void *beta_grad;
	void *d_gamma;
	void *d_beta;
	
};


struct merge_param
{
	int merge_type;
	int previous_id_a;
	int previous_id_b;
	layer *previous_a;
	layer *previous_b;
};


//############################################
//            Activation functions
//############################################

struct ReLU_param
{
	float saturation;
	float leaking_factor;
};


struct logistic_param
{
	float beta;
	float saturation;
};


struct yolo_param
{
	int *cell_size;
	int no_override;

	int nb_box;
	int nb_class;
	int nb_param;
	int max_nb_obj_per_image;
	int fit_dim;
	int IoU_type;
	int prior_dist_type;
	float (*c_IoU_fct)(float*, float*);
	float *prior_size;
	float *noobj_prob_prior;
	int class_softmax;
	int diff_flag;
	int error_type;
	int raw_output;

	//Association related parameters
	int strict_box_size_association;
	void *block_state; //specific to CUDA
	int rand_startup;
	float rand_prob_best_box_assoc;
	float rand_prob;
	float min_prior_forced_scaling;
	
	//Error scaling, 6 elements
	float *scale_tab;
	//activation slopes, 6 times 3 elements
	float **slopes_and_maxes_tab;
	float *param_ind_scale;
	//Various IoU thresholds (Good but not best, low IoU re-association, fit limits ...)
	float *IoU_limits;
	//use to disable the fit of given loss parts
	int *fit_parts;
	
	//Shared ancillary arrays
	float *IoU_monitor;
	int *target_cell_mask;
	float *IoU_table;
	float *dist_prior;
	int *box_locked;
	float *box_in_pix;
};


//############################################
//                 Optimizers
//############################################


struct sgd_param
{
	float momentum;
};


struct sgd_var
{
	float *velocity;
};


struct adam_param
{
	float beta_1;
	float beta_2;
	float eps;
	int ams_grad;
};


struct adam_var
{
	float *first_mom;
	float *second_mom;
	float *max_second_mom;
	size_t opt_step;
};


struct rmsprop_param
{
	float alpha;
	float momentum;
	float eps;
	int centered;
};


struct rmsprop_var
{
	float *sqrt_avg;
	float *velocity;
	float *grad_avg;
};

#endif // STRUCTS_H








