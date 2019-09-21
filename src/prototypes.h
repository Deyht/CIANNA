#ifndef PROTOTYPES_H
#define PROTOTYPES_H

#include "structs.h"


//######################################
//         Public variables
//######################################

//to re organise base on file declaration

extern int input_width, input_height, input_depth;
extern int input_dim;
extern int output_dim;
extern int batch_size;
extern real learning_rate;
extern real momentum;
extern real decay;
extern int compute_method;
extern int confusion_matrix;

extern real *input;
extern real *target;
extern int nb_batch;
extern int length;
extern real *output_error;
extern real *output_error_cuda;

extern int nb_layers;
extern layer *net_layers[100];

//######################################


//######################################
//       auxil.c prototypes
//######################################

void init_network(int u_input_dim[3], int u_output_dim, int u_batch_size, int u_compute_method);
Dataset create_dataset(int nb_elem);
void enable_confmat(void);
void compute_error(Dataset data);
void train_network(Dataset train_set, Dataset valid_set, int nb_epochs, int control_interv, real u_learning_rate, real u_momentum, real u_decay);


//######################################
//       CUDA public prototypes
//######################################


//activations function
void define_activation(layer* current);
void linear_activation(layer *current);
void linear_deriv(layer *current);
void output_deriv_error(layer* current);
void output_error_fct(layer* current);

//dense_layer.c
void dense_create(layer* previous, int nb_neurons, int activation);

//conv_layer.c
void conv_create(layer* previous, int f_size, int nb_filters, int stride, int padding, int activation);

//pool_layer.c
void pool_create(layer* previous, int pool_size);

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
__device__ int cuda_argmax(real* tab, int dim_out);
#endif
void init_cuda(void);
void cuda_free_table(real* tab);
void cuda_convert_table(real **tab, int size);
void cuda_convert_dataset(Dataset data);
void cuda_create_table(real **tab, int size);
void cuda_get_table(real **cuda_table, real **table, int size);
void cuda_print_table(real* tab, int size, int return_every);
void cuda_print_table_transpose(real* tab, int line_size, int column_size);
void cuda_confmat(real *out, real* mat);

void cuda_convert_network(layer** network);
void cuda_deriv_output_error(layer *current);
void cuda_output_error_fct(layer* current);
void cuda_output_error(layer* current);
void cuda_compute_error(real** data, real** target_data, int nb_data, float** out_mat, layer *net_layer);


//cuda_activ_functions.cu
void cuda_define_activation(layer *current);
void cuda_deriv_output_error(layer *current);

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


