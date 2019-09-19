#include "prototypes.h"


static int cu_blocks;

//public are in "prototypes.h"

//private prototypes
void cuda_linear_activation(layer *current);
void cuda_linear_deriv(layer *previous);
void cuda_linear_deriv_output_error(layer* current);
void cuda_linear_output_error(layer* current);

void cuda_ReLU_activation(layer *current);
void cuda_ReLU_deriv(layer *previous);
void cuda_ReLU_deriv_output_error(layer* current);
void cuda_ReLU_output_error(layer* current);

void cuda_logistic_activation(layer *current);
void cuda_logistic_deriv(layer *previous);
void cuda_logistic_deriv_output_error(layer* current);
void cuda_logistic_output_error(layer* current);

void cuda_softmax_activation(layer *current);
void cuda_softmax_deriv(layer *previous);
void cuda_softmax_deriv_output_error(layer *current);
void cuda_softmax_output_error(layer *current);

__global__ void ReLU_activation_kernel(real *tab, int len, int dim, real leaking_factor);
__global__ void ReLU_deriv_kernel(real *deriv, real *value, int len, int dim, real leaking_factor, int size);
__global__ void quadratic_deriv_output_error_kernel(real *delta_o, real *output, real *target, 
	int dim, int len, int size);
__global__ void quadratic_output_error_kernel(real *output_error, real *output, real *target, 
	int dim, int len, int size);
__global__ void logistic_activation_kernel(real *tab, real beta, real saturation, int dim, int len, int size);
__global__ void logistic_deriv_kernel(real *deriv, real* value, real beta, int len, int dim, int size);
__global__ void softmax_activation_kernel(real *tab, int len, int dim, int size);
__global__ void cross_entropy_deriv_output_error_kernel(real *delta_o, real *output, real *target, int len, int dim, int size);
__global__ void cross_entropy_output_error_kernel(real *output_error, real *output, real *target, int len, int dim, int size);


void cuda_define_activation(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			current->activation = cuda_ReLU_activation;
			current->deriv_activation = cuda_ReLU_deriv;
			break;
		
		case LOGISTIC:
			current->activation = cuda_logistic_activation;
			current->deriv_activation = cuda_logistic_deriv;
			break;
			
		case SOFTMAX:
			current->activation = cuda_softmax_activation;
			current->deriv_activation = cuda_softmax_deriv;
			break;
			
		case LINEAR:
			default:
			current->activation = cuda_linear_activation;
			current->deriv_activation = cuda_linear_deriv;
			break;
	}

}

void cuda_deriv_output_error(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_deriv_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_deriv_output_error(current);
			break;
	
	}
}

void cuda_output_error_fct(layer* current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_output_error(current);
			break;
	
	}
}

//#####################################################
//         Linear activation related functions
//#####################################################

void cuda_linear_activation(layer *current)
{
	//empty on purpose
}


void cuda_linear_deriv(layer *previous)
{
	//empty on purpose
}


void cuda_linear_deriv_output_error(layer *current)
{	
	cu_blocks = ( *((int*)current->activ_param) + cu_threads - 1) / cu_threads;
	quadratic_deriv_output_error_kernel<<< cu_blocks, cu_threads >>>(current->delta_o, current->output,
		target, *((int*)current->activ_param), *((int*)current->activ_param), *((int*)current->activ_param));
}

void cuda_linear_output_error(layer *current)
{	
	cu_blocks = (*((int*)current->activ_param) + cu_threads - 1) / cu_threads;
	quadratic_output_error_kernel<<< cu_blocks, cu_threads >>>(output_error, current->output,
		target, *((int*)current->activ_param), *((int*)current->activ_param), *((int*)current->activ_param));
}


//#####################################################




//#####################################################
//          ReLU activation related functions
//#####################################################


void cuda_ReLU_activation(layer *current)
{

	//printf("relu activation\n");
	ReLU_param *param = (ReLU_param*)current->activ_param;
	//to update for new formalism
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	ReLU_activation_kernel <<< cu_blocks, cu_threads >>>(current->output, param->size, param->dim, 
		param->leaking_factor);
}

//Is in fact a leaky ReLU, to obtain true ReLU define leaking_factor to 0
__global__ void ReLU_activation_kernel(real *tab, int len, int dim, real leaking_factor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < len)
	{
		i += i/dim;
		if(tab[i] <= 0.0)
			tab[i] *= leaking_factor;
	}
}


void cuda_ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	//to update for new formalism
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	ReLU_deriv_kernel<<< cu_blocks, cu_threads >>>(previous->delta_o, previous->output, param->size, param->dim,
		param->leaking_factor, param->size);
	
}


//should be adapted for both conv and dense layer if dim is properly defined
__global__ void ReLU_deriv_kernel(real *deriv, real *value, int len, int dim, real leaking_factor, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		i += i/dim;
		if(value[i] <= 0.0)
			deriv[i] *= leaking_factor;
	}
	else
		deriv[i] = 0.0;
}

// Should re write a output function to take into account ReLU for Conv output format
void cuda_ReLU_deriv_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	quadratic_deriv_output_error_kernel<<< cu_blocks, cu_threads >>>(current->delta_o, current->output,
		target, (param->dim+1)*length, param->dim, param->size);
	ReLU_deriv_kernel<<< cu_blocks, cu_threads >>>(current->delta_o, current->output, 
		param->size, param->dim, param->leaking_factor, param->size);
}

void cuda_ReLU_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	quadratic_output_error_kernel<<< cu_blocks, cu_threads >>>(output_error, current->output,
		target, (param->dim+1)*length, param->dim, param->size);
}

__global__ void quadratic_deriv_output_error_kernel(real *delta_o, real *output, real *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && i%dim != 0)
	{
		pos = i - i/dim;
		delta_o[i] = (output[i] - target[pos]);
	}
	else
	{
		delta_o[i] = 0.0;
	}
}



__global__ void quadratic_output_error_kernel(real *output_error, real *output, real *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && i%dim != 0)
	{
		pos = i - i/dim;
		output_error[pos] = 0.5*(output[i] - target[pos])*(output[i] - target[pos]);
	}
}


//#####################################################





//#####################################################
//          Logistic activation related funcitons
//#####################################################


void cuda_logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	logistic_activation_kernel<<< cu_blocks, cu_threads >>>(current->output, param->beta, param->saturation, param->size,  param->dim, param->size);
}

__global__ void logistic_activation_kernel(real *tab, real beta, real saturation, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i >= size)
		return;
	
	if(i < len)
	{
		i += i / dim;
		tab[i] = -beta*tab[i];
		if(tab[i] > saturation)
			tab[i] = saturation;
		tab[i] = 1.0/(1.0 + expf(tab[i]));
	}
	else
	{
		tab[i] = 0.0;
	}
}


void cuda_logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	//to update for new formalism
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	logistic_deriv_kernel <<< cu_blocks, cu_threads >>>(previous->delta_o, previous->output, param->beta,
		param->size, param->dim, param->size);
}


__global__ void logistic_deriv_kernel(real *deriv, real* value, real beta, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/dim;
		deriv[i] *= beta*value[i]*(1.0-value[pos]);
	}
	else
		deriv[i] = 0.0;
}


void cuda_logistic_deriv_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	//to update for new formalism
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	quadratic_deriv_output_error_kernel<<< cu_blocks, cu_threads >>>(current->delta_o, current->output,
		target, (param->dim+1)*length, param->dim, param->size);
	logistic_deriv_kernel <<< cu_blocks, cu_threads >>>(current->delta_o, current->output, param->beta, (param->dim+1)*length, param->dim, param->size);
	
}

void cuda_logistic_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	//to update for new formalism
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	quadratic_output_error_kernel<<< cu_blocks, cu_threads >>>(output_error, current->output,
		target, (param->dim+1)*length, param->dim, param->size);	
}

//#####################################################



//#####################################################
//          Soft-Max activation related funcitons
//#####################################################


void cuda_softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	//to update for new formalism
	cu_blocks = (batch_size + cu_threads - 1) / cu_threads;
	softmax_activation_kernel<<< cu_blocks, cu_threads >>>(current->output, length, param->dim, batch_size);
}

__global__ void softmax_activation_kernel(real *tab, int len, int dim, int size)
{
	//difficult to optimize but can be invastigated
	//is giving a probabilistic output
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;
	real *pos;
	real vmax;
	real normal = 0.0000;
	
	if(i >= size)
		return;
		
	if(i < len)
	{
		pos = tab + i*(dim+1);
		
		vmax = pos[0];
		for(j = 1; j < dim; j++)
			if(pos[j] > vmax)
				vmax = pos[j];
		
		for(j = 0; j < dim; j++)
		{	
			pos[j] = expf(pos[j]-vmax);
			normal += pos[j];
		}		
		pos[j] = 0.0;
			
		for(j = 0; j < dim; j++)
				pos[j] /= normal;
				
		pos[j] = 0.0;
	}
	else
	{
		pos = tab + i*(dim+1);		
		for(j = 0; j < dim; j++)
			pos[j] = 0.0;
		pos[j] = 0.0;
	}
}


void cuda_softmax_deriv(layer *previous)
{
	printf("Error : Softmax can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}

void cuda_softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	
	cu_blocks = ((param->dim+1)*batch_size + cu_threads - 1) / cu_threads;
	cross_entropy_deriv_output_error_kernel<<< cu_blocks, cu_threads >>>(current->delta_o, current->output, target, (param->dim+1)*length, param->dim, (param->dim+1)*batch_size);
		
}

void cuda_softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	
	cu_blocks = ((param->dim+1)*batch_size + cu_threads - 1) / cu_threads;
	cross_entropy_output_error_kernel<<< cu_blocks, cu_threads >>>(output_error, current->output, target, (param->dim+1)*length, param->dim, (param->dim+1)*batch_size);
		
}


__global__ void cross_entropy_deriv_output_error_kernel(real *delta_o, real *output, real *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		delta_o[i] = (output[i] - target[pos]);
	}
	else
	{
		delta_o[i] = 0.0;
	}
}

__global__ void cross_entropy_output_error_kernel(real *output_error, real *output, real *target, int len, int dim, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos;
	
	if(i >= size)
		return;
	
	if(i < len && (i+1)%(dim+1) != 0)
	{
		pos = i - i/(dim+1);
		output_error[pos] = -target[pos]*log(output[i]);
	}
}







//#####################################################










