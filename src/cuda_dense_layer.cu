#include "prototypes.h"


static int cu_blocks;
static dense_param *d_param;

//public are in prototypes.h

//private
void cuda_forward_dense_layer(layer *current);
void cuda_backward_dense_layer(layer* current);

__global__ void flat_dense(real* in, real* out, real bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void reroll_batch(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);


void cuda_dense_define(layer *current)
{
	current->forward = cuda_forward_dense_layer;
	current->backprop = cuda_backward_dense_layer;
}

void cuda_convert_dense_layer(layer *current)
{
	d_param = (dense_param*)current->param;
	
	if(current->previous != NULL)
	{
		switch(current->previous->type)
		{	
			case CONV:
				cuda_convert_table(&(d_param->flat_input), d_param->in_size*batch_size);
				cuda_convert_table(&(d_param->flat_delta_o), 
					(((conv_param*)current->previous->param)->nb_area_w * 
					((conv_param*)current->previous->param)->nb_area_h * 
					((conv_param*)current->previous->param)->nb_filters + 1) * batch_size);
				break;
				
			case POOL:
				cuda_convert_table(&(d_param->flat_input), d_param->in_size*batch_size);
				cuda_convert_table(&(d_param->flat_delta_o), 
					(((pool_param*)current->previous->param)->nb_area_w * 
					((pool_param*)current->previous->param)->nb_area_h * 
					((pool_param*)current->previous->param)->nb_maps + 1) * batch_size);
				break;
				
			case DENSE:
			default:
				d_param->flat_delta_o = current->previous->delta_o;
				break;
		}
	}
	
	cuda_convert_table(&(d_param->weights), d_param->in_size*(d_param->nb_neurons+1));
	cuda_convert_table(&(d_param->update), d_param->in_size*(d_param->nb_neurons+1));
	
	cuda_convert_table(&(current->output), (d_param->nb_neurons+1)*batch_size);
	cuda_convert_table(&(current->delta_o), (d_param->nb_neurons+1)*batch_size);
}


void cuda_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, depth;
	
	//printf("in forward dense\n");
	
	if(length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
	{
		current->input = input;
	}
	else if(current->previous->type != DENSE)
	{
		//printf("previous type conv identified\n");
		//Use a converted (flatten) input if needed
		switch(current->previous->type)
		{
			case CONV:
				nb_area_w = ((conv_param*)current->previous->param)->nb_area_w;
				nb_area_h = ((conv_param*)current->previous->param)->nb_area_h;
				depth = ((conv_param*)current->previous->param)->nb_filters;
				break;
			
			case POOL:
			default:
				nb_area_w = ((pool_param*)current->previous->param)->nb_area_w;
				nb_area_h = ((pool_param*)current->previous->param)->nb_area_h;
				depth = ((pool_param*)current->previous->param)->nb_maps;
				break;
		}
		//printf("\n\n########### Previous output ##########\n\n");
		//cuda_print_table_transpose(current->input, depth, nb_area_w*nb_area_h*batch_size);
		//printf("nb_a_w : %d, nb_a_h : %d, depth : %d\n", nb_area_w, nb_area_h, depth);
		
		cu_blocks = ((nb_area_w * nb_area_h * depth + 1) * batch_size + cu_threads - 1) / cu_threads;
		flat_dense<<< cu_blocks, cu_threads >>>(current->input, d_param->flat_input, d_param->bias_value, 
			nb_area_w * nb_area_h , nb_area_w * nb_area_h * depth + 1, depth, batch_size, 
			(nb_area_w * nb_area_h * depth + 1) * batch_size);
		
		//printf("print flat input\n");
		//cuda_print_table_transpose(d_param->flat_input, batch_size, (nb_area_w*nb_area_h*depth+1));
		//current->input = d_param->flat_input;
		//printf("flatten input\n");
		//cuda_print_table_transpose(d_param->flat_input, batch_size, (nb_area_w*nb_area_h*depth+1));
		
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, batch_size, d_param->in_size, 
			&cu_alpha, d_param->weights, d_param->nb_neurons+1, d_param->flat_input, d_param->in_size,
			&cu_beta, current->output, d_param->nb_neurons+1);
	}
	else if (current->previous->type == DENSE)
	{
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, batch_size, d_param->in_size, 
			&cu_alpha, d_param->weights, d_param->nb_neurons+1, current->input, d_param->in_size, &cu_beta, 
			current->output, d_param->nb_neurons+1);
	}
	
	//printf("%d\n", d_param->in_size);
	//printf("weights dense\n");
	//cuda_print_table_transpose(d_param->weights, d_param->in_size, (d_param->nb_neurons+1));
	
	//printf("Output:\n");
	//cuda_print_table_transpose(current->output, batch_size, d_param->nb_neurons+1);
	
	current->activation(current);
	
	//printf("Activated:\n");
	//cuda_print_table_transpose(current->output, batch_size, d_param->nb_neurons+1);
	
	//cuda_print_table(current->output, batch_size*(d_param->nb_neurons+1), d_param->nb_neurons+1);
	/*
	if(current->previous != NULL)
	{
		printf("Target:\n");
		cuda_print_table(target, batch_size*(d_param->nb_neurons), d_param->nb_neurons);
	}
	*/

}


void cuda_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, depth;

	d_param = (dense_param*) current->param;	

	//######################## ERROR PROPAGATION ########################

	//skip error prop if previous is the input layer

	//printf("delta_o dense\n");
	//cuda_print_table_transpose(current->delta_o, batch_size , d_param->nb_neurons+1);
	
	if(current->previous != NULL)
	{
		//cuda_print_table_transpose(d_param->weights, d_param->in_size , d_param->nb_neurons+1);
		
		cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, d_param->in_size, batch_size, d_param->nb_neurons+1,
			&cu_alpha, d_param->weights, d_param->nb_neurons+1, current->delta_o, d_param->nb_neurons+1,
			&cu_beta, d_param->flat_delta_o, d_param->in_size);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		//cuda_print_table_transpose(d_param->flat_delta_o, batch_size , d_param->in_size);
		
		
		if(current->previous->type == POOL || current->previous->type == CONV)
		{
			
			switch(current->previous->type)
			{
				case POOL:
					nb_area_w = ((pool_param*)current->previous->param)->nb_area_w;
					nb_area_h = ((pool_param*)current->previous->param)->nb_area_h;
					depth = ((pool_param*)current->previous->param)->nb_maps;
					break;
			
				case CONV:
				default:
					nb_area_w = ((conv_param*)current->previous->param)->nb_area_w;
					nb_area_h = ((conv_param*)current->previous->param)->nb_area_h;
					depth = ((conv_param*)current->previous->param)->nb_filters;
					break;
				
				
			}
			
			//printf("prev delta_o conv\n");
			//cuda_print_table_transpose(d_param->flat_delta_o, batch_size, d_param->in_size);
			
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			
			cu_blocks = (nb_area_w * nb_area_h * depth * batch_size + cu_threads - 1) / cu_threads;
			reroll_batch<<< cu_blocks, cu_threads >>>(d_param->flat_delta_o, current->previous->delta_o,
				nb_area_w * nb_area_h, nb_area_w * nb_area_h * depth + 1, depth, batch_size, nb_area_w 
				* nb_area_h * depth * batch_size);
			
			//current->previous->delta_o = d_param->flat_delta_o;
			
			//printf("prev delta_o conv\n");
			//cuda_print_table_transpose(current->previous->delta_o, depth, nb_area_w * nb_area_h * batch_size);
		
		}
		current->previous->deriv_activation(current->previous);
		//printf("prev delta_o activation\n");
		//cuda_print_table(d_param->flat_delta_o, batch_size*(d_param->in_size), d_param->in_size);
		
	}
	
		
	//########################  WEIGHTS UPDATE   ########################
	
	//based on the recovered delta_o provided by the next layer propagation
	//cuda_print_table(current->delta_o, batch_size*(d_param->nb_neurons+1), d_param->nb_neurons+1);
	
	if(current->previous->type != DENSE)
	{
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
			batch_size, &learning_rate,	current->delta_o, d_param->nb_neurons+1, d_param->flat_input,
			d_param->in_size, &momentum, d_param->update, d_param->nb_neurons+1);
	}
	else
	{		
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
			batch_size, &learning_rate,	current->delta_o, d_param->nb_neurons+1, current->input,
			d_param->in_size, &momentum, d_param->update, d_param->nb_neurons+1);
	}
		
	
	//printf("Update dense:\n");
	//cuda_print_table(d_param->update, d_param->in_size*(d_param->nb_neurons+1), d_param->nb_neurons+1);
	
	cu_blocks = (d_param->in_size*(d_param->nb_neurons+1) + cu_threads - 1) / cu_threads;
	cuda_update_weights<<< cu_blocks, cu_threads >>>(d_param->weights, d_param->update, 
		d_param->in_size*(d_param->nb_neurons+1));
	
	//printf("weights dense after update:\n");
	//cuda_print_table(d_param->weights, d_param->in_size*(d_param->nb_neurons+1), d_param->nb_neurons+1);
}



/*
//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
__global__ void flat_dense(real* in, real* out, int activation_size, int nb_filters, int bias, int batch_size, int size)
{
	//MUST LOOK FOR OPTIMIZATION
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j, pos_line, all_filter_batch_size;
	
	if( i < size)
	{
		pos_line = i%(activation_size) + (i / (activation_size))*(activation_size*nb_filters+1);	
		all_filter_batch_size = activation_size*batch_size;
		pos_line += i/(activation_size*nb_filters);
		for(j = 0; j < nb_filters; j++)
		{
			out[pos_line] = in[i];
			in += all_filter_batch_size;
			out += activation_size;
		}
	}
}
*/

//used to reshape output of Conv layer that as the result of filter 1 continuous for the all batch
//convert into all filters continuous for image 1, then image 2, ...
__global__ void flat_dense(real* in, real* out, real bias, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	//SHOULD TEST OPTIMIZATION
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;

	if(i < size)
	{
		image_id = i / flatten_size;
		map_id = (i % flatten_size)/map_size;
		pos = (i % flatten_size)%map_size;
		
		if(map_id >= nb_map)
			out[i] = bias;
		else
			out[i] = in[map_id*(map_size*batch_size) + image_id*map_size + pos];
	}
}

/*
__global__ void reroll_batch(real* in, real* out, int activation_size, int nb_filters, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pos_col, pos_in_col;
	
	if(i < size)
	{
		pos_col = i / activation_size;
		pos_in_col = i % activation_size;
		
		out[pos_in_col + (pos_col%nb_filters)*batch_size*activation_size + activation_size*(pos_col/nb_filters)] = in[i];
		
	}
}
*/

__global__ void reroll_batch(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int map_id, image_id, pos;
	
	if(i < size)
	{
		map_id = i / (map_size*batch_size);
		image_id = (i % (map_size*batch_size))/map_size;
		pos = (i % (map_size*batch_size))%map_size;
		
		out[i] = in[image_id*(flatten_size) + map_id*map_size + pos];
	}
}






