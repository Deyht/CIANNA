
/*
	Copyright (C) 2023 David Cornu
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

enum layer_type_enum{CONV, POOL, DENSE, NORM, LRN};
enum activation_functions_enum{RELU, LOGISTIC, SOFTMAX, YOLO, LINEAR};
enum initializers_enum{N_XAVIER, U_XAVIER, N_LECUN, U_LECUN, U_RAND, N_RAND};
enum inference_modes_enum{AVG_MODEL, MC_MODEL};
enum batch_param_enum{OFF, SGD, FULL};
enum data_types_enum{c_FP32, c_UINT16, c_UINT8};
enum compute_method_enum{C_NAIV, C_BLAS, C_CUDA};
enum memory_localization_enum{NO_LOC, HOST, DEVICE};
enum IoU_types_enum{IOU, GIOU, DIOU, DIOU2};
enum pool_types_enum{MAX_pool, AVG_pool};
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

typedef struct linear_param linear_param;
typedef struct ReLU_param ReLU_param;
typedef struct logistic_param logistic_param;
typedef struct softmax_param softmax_param;
typedef struct yolo_param yolo_param;



//############################################
//             CUDA specific
//############################################

#ifdef CUDA

enum GPU_type{FP32, FP16, BF16};
enum TC_comp_mode{FP32C_FP32A, TF32C_FP32A, FP16C_FP32A, FP16C_FP16A, BF16C_FP32A};

typedef struct CUDA_net_instance CUDA_net_instance;
typedef struct cuda_auxil_fcts cuda_auxil_fcts;

typedef struct cuda_dense_fcts cuda_dense_fcts;
typedef struct cuda_conv_fcts cuda_conv_fcts;
typedef struct cuda_pool_fcts cuda_pool_fcts;
typedef struct cuda_norm_fcts cuda_norm_fcts;
typedef struct cuda_lrn_fcts cuda_lrn_fcts;

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
	void (*cu_get_table_fct)(void *cuda_table, void *table, size_t size);
	void (*cu_get_typed_host_table_fct)(void *typed_table, float *out_table, size_t size);
	void (*cu_get_table_to_FP32_fct)(void *cuda_table, float *table, size_t size, void* buffer);
	void (*cu_put_table_fct)(void *cuda_table, void *table, size_t size);
	void (*cu_convert_batched_table_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	Dataset (*cu_create_dataset_fct)(network *net, int nb_elem);	
	void (*cu_get_batched_table_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	void (*cu_convert_batched_host_table_FP32_to_fct)(void **tab, int batch_size, int nb_batch, size_t size);
	void (*cu_master_weight_copy_kernel)(float *master, void *copy, size_t size);
	void (*cu_update_weights_kernel)(float *weights, void* update, float weight_decay, int is_pivot, size_t size, float TC_scale_factor);
	void (*cu_print_table_fct)(void* tab, size_t size, int return_every);
	void (*cu_shfl_kern_fct)(void** i_in, void** i_targ, void** i_train_dupl, void** i_targ_dupl,
		int* index, size_t in_size, int b_size, int d_in, int d_out);
	void (*cu_get_back_shuffle_fct)(void** i_in, void** i_targ, void** i_train_dupl, void** i_targ_dupl,
		size_t in_size, int b_size, int d_in, int d_out);
	void (*cu_host_shuffle_fct)(network *net, Dataset data, Dataset duplicate);
	void (*cu_host_only_shuffle_fct)(network *net, Dataset data);
	void (*cu_gan_disc_mix_input_kernel)(void *gen_output, void *disc_input, void* targ_input, int nb_fake, int batch_offset,
		int nb_filters, int flat_f_size, int batch_size, size_t size);
	void (*cu_create_gan_target_kernel)(void* i_targ, void* i_true_targ, int out_size, 
		int batch_size, float frac_ones, int i_half, size_t len);
	void (*cu_exp_disc_activation_kernel)(void *i_tab, int dim, int biased_dim,
		int offset, int length, int batch_size, size_t size, int halved, int revert);
	void (*cu_exp_disc_deriv_output_kernel)(void *i_delta_o, void *i_output, void *i_target, int dim, int biased_dim,
		int offset, int length, int batch_size, size_t size, float TC_scale_factor, int halved, int revert);
	void (*cu_gan_invert_generator_deltao_kernel)(void *gen_delta_o, size_t size);
};


struct cuda_dense_fcts
{
	void (*flat_dense_fct)(void* i_in, void* i_out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
	void (*reroll_fct)(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
	void (*drop_apply_fct)(void* i_table, float* mask, size_t size, int biased_dim, float drop_rate);
	void (*drop_scale_fct)(void* i_table, float* mask, size_t size, int biased_dim, float drop_rate);
};

struct cuda_conv_fcts
{
	void (*im2col_fct)(void* i_output, void* i_input,
		int image_size, size_t flat_image_size,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int internal_padding_w, int internal_padding_h, int internal_padding_d,
		int channel, int channel_padding, int image_padding, int TC_padding,
		int batch_size, int f_size_w, int f_size_h, int f_size_d, int flat_f_size,
		int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int nb_area_d, int bias_in, int bias_out);
	void (*drop_apply_fct)(void* i_table, float* mask, size_t size, float drop_rate);
	void (*drop_scale_fct)(void* i_table, float* mask, size_t size, float drop_rate);
	void (*rotate_filter_fct)(void* in, void* out, int nb_rows, int TC_padding, int depth_size, int nb_filters_in, int len);
};

struct cuda_pool_fcts
{
	void (*max_pool_fct)(void* i_input, void* i_output, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
	void (*avg_pool_fct)(void* i_input, void* i_output, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int bias_in, int length);
	void (*max_deltah_pool_fct)(void* i_delta_o, void* i_delta_o_unpool, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, size_t length);
	void (*avg_deltah_pool_fct)(void* i_delta_o, void* i_delta_o_unpool, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, size_t length);
	void (*drop_apply_fct)(void* i_table, float* mask, size_t size, float drop_rate);
	void (*drop_scale_fct)(void* i_table, float* mask, size_t size, float drop_rate);
	void (*typed_memset_fct)(void* i_table, int value, size_t size);
};

struct cuda_norm_fcts
{
	void (*cu_reduce_group_mean_conv_kernel)(void *idata, float *group_mean, int group_size, 
		int nb_group, int flat_a_size, int batch_size, int sum_div, int sum_size);
	void (*cu_reduce_group_var_conv_kernel)(void *idata, float *group_var, float *group_mean, 
		int group_size, int nb_group, int flat_a_size, int batch_size, int sum_div, int sum_size);
	void (*cu_reduce_group_dgamma_conv_kernel)(void *idata, void *d_output, float *d_gamma,
		float *group_var, float *group_mean, int group_size, int nb_group, int flat_a_size, int batch_size, int sum_size);
	void (*cu_group_normalization_conv_kernel)(void *i_output, void *i_input, float *gamma, float *beta,
		float *group_mean, float *group_var, int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off);
	void (*cu_group_normalization_conv_back_kernel)(
		void *i_input, void* i_delta_output, void *i_delta_input, float *gamma, float *beta, float *d_gamma, float * d_beta, 
		float *group_mean, float *group_var, int b_length, int b_size, int group_size, int nb_group, int nb_filters, int flat_a_size, int set_off);
	void (*cu_group_normalization_dense_kernel)(void *i_tab, int b_length, int b_size, int dim, int biased_dim, int group_size, int nb_group);
	void (*cu_group_normalization_dense_back_kernel)(void *i_tab, int b_length, int b_size, int dim, int biased_dim, int group_size, int nb_group);
};

struct cuda_lrn_fcts
{
	void (*cu_lrn_conv_kernel)(void *i_output, void *i_input, float *local_scale, int range, 
		float k, float alpha, float beta, int b_size, int nb_channel, int flat_a_size);
	void (*cu_lrn_conv_back_kernel)(void *i_output, void *i_input, void *i_delta_output, void *i_delta_input,
		float *local_scale, int range, float k, float alpha, float beta, int b_size, int nb_channel, int flat_a_size);
};


struct cuda_linear_activ_fcts
{
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, int dim, 
		int biased_dim, int offset, int length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, int dim, 
		int biased_dim, int offset, int length, size_t size);
};

struct cuda_ReLU_activ_fcts
{
	void (*activ_fct)(void *i_tab, int dim, int biased_dim, int offset, 
		float saturation, float leaking_factor, int length, size_t size);
	void (*deriv_fct)(void *i_deriv, void *i_value, int dim, int biased_dim, 
		int offset, float saturation, float leaking_factor, int length, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		int dim, int biased_dim, int offset, int length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		int dim, int biased_dim, int offset, int length, size_t size);
};

struct cuda_logistic_activ_fcts
{
	void (*activ_fct)(void *i_tab, float beta, float saturation, int dim, int biased_dim, int offset, int length, size_t size);
	void (*deriv_fct)(void *i_deriv, void *i_value, float beta, int dim, int biased_dim, int offset, int length, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		int dim, int biased_dim, int offset, int length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, 
		int dim, int biased_dim, int offset, int length, size_t size);
};

struct cuda_softmax_activ_fcts
{
	void (*activ_fct)(void *i_tab, int dim, int biased_dim, int offset, int length, int batch_size, size_t size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target,
		int dim, int offset, int biased_dim, int length, size_t size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target,
		int dim, int biased_dim, int length, int offset, size_t size);
};

struct cuda_YOLO_activ_fcts
{
	void (*activ_fct)(void *i_tab, int flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, 
		int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, size_t size, float TC_scale_factor, int nb_im_iter);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, 
		int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, size_t size);
};

struct CUDA_net_instance
{
	int dynamic_load;
	int use_cuda_TC;
	void *output_error_cuda;

	cuda_auxil_fcts cu_auxil_fcts;

	cuda_dense_fcts cu_dense_fcts;
	cuda_conv_fcts cu_conv_fcts;
	cuda_pool_fcts cu_pool_fcts;
	cuda_norm_fcts cu_norm_fcts;
	cuda_lrn_fcts cu_lrn_fcts;
	
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
	void (*cont_copy)(float *elem_in, void *elem_out, int out_offset, size_t nb_elem);
	int size;
	int nb_batch;
	int localization;
};

struct layer
{
	int type;
	int activation_type;
	int initializer;
	network *c_network;
	
	void *param;
	void *input; //usually contain adress of previous->output
	void *output;
	void *delta_o;
	
	int frozen;
	layer *previous;
	
	float bias_value;
	float dropout_rate;
	
	void (*forward)(layer *parent);
	void (*backprop)(layer *parent);
	
	void (*activation)(layer *parent);
	void (*deriv_activation)(layer *parent);
	void *activ_param;
	
	//utility
	int nb_params;
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
	float input_bias;
	
	float learning_rate;
	float momentum;
	float decay;
	float weight_decay;
	
	Dataset train, test, valid;
	Dataset train_buf, test_buf, valid_buf;
	
	int in_dims[4];
	size_t input_dim; // flat size
	int output_dim; //Correspond to the "target size"
	int out_size; //Correspond to the actual ouput size with paddings if needed
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

	// Parameters used for the YOLO formated dataset laoding/saving
	int norm_factor_defined; 
	float *offset_input, *offset_output;
	float *norm_input, *norm_output;
	int dim_size_input, dim_size_output;
	
	float TC_scale_factor;
	#ifdef CUDA
	CUDA_net_instance cu_inst;
	#endif

};



//############################################
//               Various Layers
//############################################

struct dense_param
{
	int in_size;
	int nb_neurons;
	
	void *flat_input;
	void *flat_delta_o;
	
	void *weights;
	void *FP32_weights;
	void *update;
	float *dropout_mask;
};


struct conv_param
{
	int *f_size;
	int flat_f_size;
	int TC_padding;
	int *stride;
	int *padding;
	int *int_padding;
	
	int nb_filters;
	int *nb_area;

	int *prev_size;
	int prev_depth;

	void *im2col_input;
	void *im2col_delta_o;
	void *filters;
	void *FP32_filters;
	void *rotated_filters;
	void *temp_delta_o;
	void *update;
	float *dropout_mask;
};


struct pool_param
{
	int *p_size;
	int *nb_area;
	int *stride;
	int *padding;
	int nb_maps;
	int *prev_size;
	int prev_depth;
	int pool_type;
	int global;
	
	float *dropout_mask;
	
	int next_layer_type;
	
	int *pool_map;
	void *temp_delta_o;
};


struct norm_param
{
	int data_format;
	void *prev_param;
	int group_size;
	int set_off;
	int nb_group;
	int n_dim;
	int dim_offset;
	int output_dim;
	float *mean;
	float *var;
	float *gamma;
	float *beta;
	float *gamma_update;
	float *beta_update;
	float *d_gamma;
	float *d_beta;
	
	float *gamma_gpu;
	float *beta_gpu;
	float *d_gamma_gpu;
	float *d_beta_gpu;
};


struct lrn_param
{
	int data_format;
	void *prev_param;
	
	int n_dim;
	int dim_offset;
	int output_dim;
	
	int range;
	float k;
	float alpha;
	float beta;
	
	float *local_scale;
};


//############################################
//            Activation functions
//############################################

struct linear_param
{
	int size;
	int dim;
	int biased_dim;
	int offset;
};


struct ReLU_param
{
	int size;
	int dim;
	int biased_dim;
	int offset;
	float saturation;
	float leaking_factor;
};


struct logistic_param
{
	int size;
	int dim;
	int biased_dim;
	int offset;
	float beta;
	float saturation;
};


struct softmax_param
{
	int size;
	int dim;
	int biased_dim;
	int offset;
};


struct yolo_param
{
	int size;
	int dim;
	int biased_dim;
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
	void *block_state;
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


#endif // STRUCTS_H








