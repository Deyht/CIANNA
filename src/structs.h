
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


#ifndef STRUCTS_H
#define STRUCTS_H

#include "defs.h"


//############################################
//            Various Enumerations
//############################################

enum layer_type{CONV, POOL, DENSE};
enum activation_functions{RELU, RELU_6, LOGISTIC, SOFTMAX, YOLO, LINEAR};
enum initializers{N_XAVIER, U_XAVIER, N_LECUN, U_LECUN, U_RAND, N_RAND};
enum inference_modes{AVG_MODEL, MC_MODEL};
enum batch_param{OFF, SGD, FULL};
enum data_types{c_FP32, c_UINT16, c_UINT8};
enum compute_method{C_NAIV, C_BLAS, C_CUDA};
enum memory_localization{NO_LOC, HOST, DEVICE};
enum IoU_types{IOU, GIOU, DIOU};
enum pool_types{MAX_pool, AVG_pool};

typedef struct Dataset Dataset;
typedef struct layer layer;
typedef struct network network;

typedef struct dense_param dense_param;
typedef struct conv_param conv_param;
typedef struct pool_param pool_param;

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

typedef struct cuda_linear_activ_fcts cuda_linear_activ_fcts;
typedef struct cuda_ReLU_activ_fcts cuda_ReLU_activ_fcts;
typedef struct cuda_logistic_activ_fcts cuda_logistic_activ_fcts;
typedef struct cuda_softmax_activ_fcts cuda_softmax_activ_fcts;
typedef struct cuda_YOLO_activ_fcts cuda_YOLO_activ_fcts;


struct cuda_auxil_fcts
{
	void (*cu_create_host_table_fct)(void **tab, int size);
	size_t (*cu_convert_table_fct)(void **tab, size_t size);
	void (*cu_create_table_fct)(void **tab, int size);
	void (*cu_get_table_fct)(void *cuda_table, void *table, int size);
	void (*cu_get_table_to_FP32_fct)(void *cuda_table, float *table, int size, void* buffer);
	void (*cu_put_table_fct)(void *cuda_table, void *table, int size);
	void (*cu_convert_batched_table_fct)(void **tab, int batch_size, int nb_batch, int size);
	Dataset (*cu_create_dataset_fct)(network *net, int nb_elem);	
	void (*cu_get_batched_table_fct)(void **tab, int batch_size, int nb_batch, int size);
	void (*cu_convert_batched_host_table_FP32_to_fct)(void **tab, int batch_size, int nb_batch, int size);
	void (*cu_master_weight_copy_kernel)(float *master, void *copy, int size);
	void (*cu_update_weights_kernel)(float *weights, void* update, int size, float TC_scale_factor);
	void (*cu_print_table_fct)(void* tab, int size, int return_every);
	void (*cu_add_confmat_fct)(void *i_out, void *i_targ, float *mat, int len, int o_dim);
	void (*cu_shfl_kern_fct)(void** i_in, void** i_targ, void** i_train_dupl, void** i_targ_dupl,
		int* index, int in_size, int b_size, int d_in, int d_out);
	void (*cu_get_back_shuffle_fct)(void** i_in, void** i_targ, void** i_train_dupl, void** i_targ_dupl,
		int in_size, int b_size, int d_in, int d_out);
	void (*cu_host_shuffle_fct)(network *net, Dataset data, Dataset duplicate);
	void (*cu_host_only_shuffle_fct)(network *net, Dataset data);
	void (*cu_gan_disc_mix_input_kernel)(void *gen_output, void *disc_input, void* true_input,
		int half_offset, int im_flat_size, int nb_channels, int batch_size, int len);
	void (*cu_create_gan_target_kernel)(void* i_targ, void* i_true_targ, int out_size, int batch_size, float frac_ones, int i_half, int len);
	void (*cu_exp_disc_activation_kernel)(void *i_tab, int len, int dim, int size, int halved, int revert);
	void (*cu_exp_disc_deriv_output_kernel)(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, int halved, int revert);
};


struct cuda_dense_fcts
{
	void (*flat_dense_fct)(void* i_in, void* i_out, float bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
	void (*reroll_fct)(void* in, void* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
	void (*drop_apply_fct)(void* i_table, int batch_size, int dim, int* mask);
};

struct cuda_conv_fcts
{
	void (*im2col_fct)(void* i_output, void* i_input,
		int image_size, int flat_image_size,
		int stride_w, int stride_h ,int stride_d,
		int padding_w, int padding_h, int padding_d,
		int internal_padding_w, int internal_padding_h, int internal_padding_d,
		int channel, int channel_padding, int image_padding, int TC_padding,
		int batch_size, int f_size_w, int f_size_h, int f_size_d, int flat_f_size,
		int w_size, int h_size, int d_size, int nb_area_w, int nb_area_h, int bias_in, int bias_out);
	void (*drop_apply_fct)(void* i_table, int batch_size, int dim, int* mask, int size);
	void (*rotate_filter_fct)(void* in, void* out, int nb_rows, int TC_padding, int depth_size, int nb_filters_in, int len);
};

struct cuda_pool_fcts
{
	void (*max_pool_fct)(void* i_input, void* i_output, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int length);
	void (*avg_pool_fct)(void* i_input, void* i_output, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int w_size, int h_size, int d_size,
		int w_size_out, int h_size_out, int d_size_out, int length);
	void (*max_deltah_pool_fct)(void* i_delta_o, void* i_delta_o_unpool, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int len, int batch_size, int image_size, int w_size, int h_size);
	void (*avg_deltah_pool_fct)(void* i_delta_o, void* i_delta_o_unpool, int* pool_map,
		int pool_size_w, int pool_size_h, int pool_size_d,
		int len, int batch_size, int image_size, int w_size, int h_size);
	void (*drop_apply_fct)(void* i_table, int batch_size, int dim, int* mask, int size);
	void (*typed_memset_fct)(void* i_table, int value, int size);
};

struct cuda_linear_activ_fcts
{
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, int len, int dim, int size);
};

struct cuda_ReLU_activ_fcts
{
	void (*activ_fct)(void *i_tab, int len, int dim, float saturation, float leaking_factor);
	void (*deriv_fct)(void *i_deriv, void *i_value, int len, int dim, float saturation, float leaking_factor, int size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, int len, int dim, int size);
};

struct cuda_logistic_activ_fcts
{
	void (*activ_fct)(void *i_tab, float beta, float saturation, int len, int dim, int size);
	void (*deriv_fct)(void *i_deriv, void *i_value, float beta, int len, int dim, int size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, int len, int dim, int size);
};

struct cuda_softmax_activ_fcts
{
	void (*activ_fct)(void *i_tab, int len, int dim, int size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, int len, int dim, int size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, int len, int dim, int size);
};

struct cuda_YOLO_activ_fcts
{
	void (*activ_fct)(void *i_tab, int flat_offset, int len, yolo_param y_param, int size);
	void (*deriv_output_error_fct)(void *i_delta_o, void *i_output, void *i_target, 
		int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, int size, float TC_scale_factor);
	void (*output_error_fct)(float *output_error, void *i_output, void *i_target, 
		int flat_target_size, int flat_output_size, int nb_area_w, int nb_area_h, int nb_area_d,
		yolo_param y_param, int size);
};

struct CUDA_net_instance
{
	int dynamic_load;
	int use_cuda_TC;
	void* output_error_cuda;

	cuda_auxil_fcts cu_auxil_fcts;

	cuda_dense_fcts cu_dense_fcts;
	cuda_conv_fcts cu_conv_fcts;
	cuda_pool_fcts cu_pool_fcts;
	
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
	void (*cont_copy)(float *elem_in, void *elem_out, int out_offset, int nb_elem);
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
	
	void (*forward)(layer *parent);
	void (*backprop)(layer *parent);
	
	void (*activation)(layer *parent);
	void (*deriv_activation)(layer *parent);
	void *activ_param;
	
	//utility
	float time_fwd;
	float time_back;
	
};

struct network
{
	layer *net_layers[MAX_LAYERS_NB];

	int id;
	int compute_method;
	int nb_layers;
	float input_bias;
	
	float learning_rate;
	float momentum;
	
	Dataset train, test, valid;
	Dataset train_buf, test_buf, valid_buf;
	
	int in_dims[4];
	int input_dim; // flat size
	//Correspond to the "target size"
	int output_dim;
	//Correspond to the actual ouput size with various paddings if needed
	int out_size;
	int batch_size;
	int batch_param;
	int epoch;
	
	int is_inference;
	int inference_drop_mode;
	int no_error;
	int perf_eval;
	float *fwd_perf, *back_perf;
	int *fwd_perf_n, *back_perf_n;
	
	void* input;
	void* target;
	int length;
	void* output_error;
	
	//Possible yolo_param
	yolo_param *y_param;

	//Normalization parameters used for the formated dataset laoding
	int norm_factor_defined; 
	float *offset_input, *offset_output;
	float *norm_input, *norm_output;
	int dim_size_input, dim_size_output;
	
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
	
	int activation;
	
	void* flat_input;
	void* flat_delta_o;
	
	void* weights;
	void *FP32_weights;
	void* update;
	int* dropout_mask;
	void* block_state;
	
	float bias_value;
	float dropout_rate;
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
	int  *dropout_mask;
	void *block_state;
	
	float bias_value;
	float dropout_rate;
};


struct pool_param
{
	int *p_size;
	int *nb_area;
	int nb_maps;
	int *prev_size;
	int prev_depth;
	int pool_type;
	
	int* dropout_mask;
	void* block_state;
	float dropout_rate;
	
	int next_layer_type;
	
	int *pool_map;
	void* temp_delta_o;
};


//############################################
//            Activation functions
//############################################

struct linear_param
{
	int size;
	int dim;
	int biased_dim;
};

struct ReLU_param
{
	int size;
	int dim;
	int biased_dim;
	float saturation;
	float leaking_factor;
};

struct logistic_param
{
	int size;
	int dim;
	int biased_dim;
	float beta;
	float saturation;
};

struct softmax_param
{
	int dim;
	int biased_dim;
};

struct yolo_param
{
	int size;
	int dim;
	int biased_dim;
	int cell_w, cell_h, cell_d;

	int nb_box;
	int nb_class;
	int nb_param;
	int IoU_type;
	float (*c_IoU_fct)(float*, float*);
	float *prior_w;
	float *prior_h;
	float *prior_d;
	float *noobj_prob_prior;

	int strict_box_size_association;
	//Error scaling, 6 elements
	float *scale_tab;
	//activation slopes, 6 times 3 elements
	float **slopes_and_maxes_tab;
	//minimum IoU required before elements fit
	float *param_ind_scale;
	float *IoU_limits;
	//use to disable the fit of given loss parts
	int *fit_parts;
	
	//monitoring
	float *IoU_monitor;
};


#endif // STRUCTS_H








