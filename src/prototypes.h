#ifndef PROTOTYPES_H
#define PROTOTYPES_H

#include "structs.h"


//######################################
//         Public variables
//######################################

//to re organise base on file declaration

extern real* input;
extern int input_width, input_height, input_depth;
extern real* target;
extern int batch_size;
extern int nb_batch;
extern int length;
extern real learning_rate;
extern real momentum;
extern int compute_method;

extern int nb_layers;

//######################################



//######################################
//       CUDA public prototypes
//######################################


//activations function
void output_error(layer* current);
void define_activation(layer* current);
void linear_activation(layer *current);
void linear_deriv(layer *current);

//dense_layer.c
void dense_create(layer *current, layer* previous, int nb_neurons, int activation);

//conv_layer.c
void conv_create(layer *current, layer* previous, int f_size, int nb_filters, int stride, int padding, int activation);

//pool_layer.c
void pool_create(layer *current, layer* previous, int pool_size);

//initializers
void xavier_normal(real *tab, int dim_in, int dim_out, int bias_padding);


//######################################


#ifdef CUDA
//######################################
//       CUDA public prototypes
//######################################

#ifdef comp_CUDA
//when compiled by nvcc must be exported as regular C prototypes
//when compiled by gcc act as regular prototype C natively
extern "C"
{
//only make sence sources compiles with nvcc
//cuda_main.cu
extern int cu_threads;
extern real cu_alpha, cu_beta;
extern cublasHandle_t cu_handle;
__global__ void cuda_update_weights(real *weights, real* updates, int size);
#endif
void init_cuda(void);
void cuda_convert_table(real **tab, int size);
void cuda_convert_batched_table(real **tab, int nb_batch, int batch_size, int size);
void cuda_print_table(real* tab, int size, int return_every);
void cuda_print_table_transpose(real* tab, int line_size, int column_size);
void cuda_convert_network(layer** network);
void cuda_confmat(real** data, real** target_data, int nb_data, float** out_mat, layer *net_layer);


//cuda_activ_functions.cu
void cuda_define_activation(layer *current);
void cuda_output_error(layer *current);

//cuda_dense_layer.cu
void cuda_dense_define(layer *current);
void cuda_convert_dense_layer(layer *current);

//cuda_conv_layer.cu
void cuda_conv_define(layer *current);
void cuda_convert_conv_layer(layer *current);

//cuda_pool_layer.cu
void cuda_pool_define(layer *current);
void cuda_convert_pool_layer(layer *current);

//timer.c
void init_timer(struct timeval* tstart);
real ellapsed_time(struct timeval tstart);
	
//######################################

#ifdef comp_CUDA
}
#endif

#endif // CUDA


#endif //PROTOTYPES_H


