
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


//############################################
//                  Global
//############################################

typedef struct layer layer;
struct layer
{
	int type;
	int activation_type;
	int initializer;
	
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
	
	real bias_value;

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








