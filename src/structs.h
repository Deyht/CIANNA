
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

enum GPU_type{FP32, FP16, BF16};
enum layer_type{CONV, POOL, DENSE};
enum activation_functions{RELU, RELU_6, LOGISTIC, SOFTMAX, YOLO, LINEAR};
enum initializers{N_XAVIER, U_XAVIER, N_LECUN, U_LECUN, U_RAND, N_RAND};
enum inference_modes{AVG_MODEL, MC_MODEL};
enum batch_param{OFF, SGD, FULL};
enum data_types{c_FP32, c_UINT16, c_UINT8};
enum compute_method{C_NAIV, C_BLAS, C_CUDA};
enum memory_localization{HOST, DEVICE};
enum IoU_types{IOU, GIOU, DIOU};

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
//                  Global
//############################################

struct Dataset
{
	void **input;
	void **target;
	void **input_device;
	void **target_device;
	void (*cont_copy)(void *elem_in, void *elem_out, int out_offset, int nb_elem);
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
	int dynamic_load;
	int use_cuda_TC;
	int nb_layers;
	float input_bias;
	
	float learning_rate;
	float momentum;
	
	Dataset train, test, valid;
	Dataset train_buf, test_buf, valid_buf;
	
	int input_width, input_height, input_depth;
	int input_dim;
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
	void* output_error_cuda;
	
	//Possible yolo_param
	yolo_param *y_param;

	//Normalization parameters used for the formated dataset laoding
	int norm_factor_defined; 
	float *offset_input, *offset_output;
	float *norm_input, *norm_output;
	int dim_size_input, dim_size_output;

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
	int f_size;
	int flat_f_size;
	int stride;
	int padding;
	
	int nb_filters;
	int nb_area_w;
	int nb_area_h;

	int prev_size_w;
	int prev_size_h;
	int prev_depth;

	void *im2col_input;
	void *im2col_delta_o;
	void *filters;
	void *FP32_filters;
	void *rotated_filters;
	void *temp_delta_o;
	void *update;
	int* dropout_mask;
	void* block_state;
	
	float bias_value;
	float dropout_rate;

};


struct pool_param
{
	int p_size;
	int nb_area_w;
	int nb_area_h;
	int nb_maps;
	int prev_size_w;
	int prev_size_h;
	int prev_depth;
	
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
	float *IoU_limits;
	//use to disable the fit of given loss parts
	int *fit_parts;
	
	//monitoring
	float *IoU_monitor;
};


#endif // STRUCTS_H








