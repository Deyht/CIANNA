#include "prototypes.h"


//#####################################################


void output_error(layer* current)
{
	switch(compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_output_error(current);
			#endif
			break;
		
		case C_BLAS:
			#ifdef BLAS
			printf("BLAS computation do not exist yet\n");
			#endif
			break;
		default:
			printf("default computaiton do not exist yet\n");
			exit(EXIT_FAILURE);
			break;
	}	
}



//#####################################################
//          Linear activation related functions
//#####################################################

void linear_activation(layer *current)
{
	//Do nothing in a linear activation
}

void linear_deriv(layer *current)
{
	//the linear derivativ is always 1, no action is required
}

//#####################################################


