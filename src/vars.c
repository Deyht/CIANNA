#include "structs.h"

//USER DEFINED VARIABLES
int input_width = 4, input_height = 1, input_depth = 1;
int input_dim = 4;
int output_dim;
int batch_size = 10;
real learning_rate = 0.3;
real momentum = 0.6;
real decay = 0.001;
int compute_method = C_CUDA;
int confusion_matrix = 0;

//SHARED INNER VARIABLEs
real* input;
real* target;
int length;
real* output_error;
real* output_error_cuda;

int nb_layers = 0;
layer *net_layers[100];


