
#ifndef STRUCTS_H
#define STRUCTS_H

#include "defs.h"


//############################################
//            Various Enumerations
//############################################


enum layer_type{CONV, POOL, DENSE};
enum activation_functions{RELU, LOGISTIC, SOFTMAX, LINEAR};
enum initializers{N_XAVIER, U_XAVIER, N_LECUN, U_LECUN, U_RAND, N_RAND};
enum compute_method{C_NAIV, C_BLAS, C_CUDA};
enum memory_localization{HOST, DEVICE};

//############################################
//                  Global
//############################################

typedef struct Dataset
{
	real **input;
	real **target;
	real **input_device;
	real **target_device;
	int size;
	int nb_batch;
	int localization;
} Dataset;

typedef struct layer layer;
typedef struct network network;

struct layer
{
	int type;
	int activation_type;
	int initializer;
	network *c_network;
	
	void *param;
	real *input; //usually contain adress of previous->output
	real *output;
	real *delta_o;
	
	layer *previous;
	
	void (*forward)(layer *parent);
	void (*backprop)(layer *parent);
	
	void (*activation)(layer *parent);
	void (*deriv_activation)(layer *parent);
	void *activ_param;
	
	//utility
	real time_fwd;
	real time_back;
	
};

struct network
{
	layer *net_layers[MAX_LAYERS_NB];

	int id;
	int compute_method;
	int nb_layers;
	
	real learning_rate;
	real momentum;
	
	Dataset train, test, valid;
	
	int input_width, input_height, input_depth;
	int input_dim;
	int output_dim;
	int out_size;
	int batch_size;
	int epoch;
	
	real* input;
	real* target;
	int length;
	real* output_error;
	real* output_error_cuda;

};


//############################################
//               Various Layers
//############################################

typedef struct dense_param
{
	int in_size;
	int nb_neurons;
	
	int activation;
	
	real* flat_input;
	real* flat_delta_o;
	
	real* weights;
	real* update;
	real* dropout_mask;
	void* block_state;
	
	real bias_value;
	real dropout_rate;

} dense_param;


typedef struct conv_param
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

	real *im2col_input;
	real *im2col_delta_o;
	real *filters;
	real *rotated_filters;
	real *temp_delta_o;
	real *update;
	
	real bias_value;

} conv_param;


typedef struct pool_param
{
	int p_size;
	int nb_area_w;
	int nb_area_h;
	int nb_maps;
	
	int prev_size_w;
	int prev_size_h;
	int prev_depth;
	
	int next_layer_type;
	
	real *pool_map;
	real* temp_delta_o;

} pool_param;


//############################################
//            Activation functions
//############################################

typedef struct ReLU_param
{
	int size;
	int dim;
	real leaking_factor;

} ReLU_param;

typedef struct logistic_param
{
	int size;
	int dim;
	real beta;
	real saturation;
	
} logistic_param;

typedef struct softmax_param
{
	int dim;
		
} softmax_param;

typedef struct linear_param
{
	int size;
	int dim;
	
} linear_param;


#endif // STRUCTS_H








