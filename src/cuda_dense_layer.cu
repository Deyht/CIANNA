#include "prototypes.h"


static int cu_blocks;
static dense_param *d_param;

//public are in prototypes.h

//private
void cuda_forward_dense_layer(layer *current);
void cuda_backward_dense_layer(layer* current);

__global__ void flat_dense(real* in, real* out, real bias, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void reroll_batch(real* in, real* out, int map_size, int flatten_size, int nb_map, int batch_size, int size);
__global__ void init_block_state(unsigned int seed,  curandState_t* states);
__global__ void dropout_select(real* mask, int size, real drop_rate, curandState_t* states);
__global__ void dropout_apply(real* table, real batch_size, int dim, real *mask);

void cuda_dense_define(layer *current)
{
	current->forward = cuda_forward_dense_layer;
	current->backprop = cuda_backward_dense_layer;
}


void cuda_convert_dense_layer(layer *current)
{
	d_param = (dense_param*)current->param;
	
	int pos = 0;
	float value;
	//if there is a transition in activation function between dense, the pivot value must be adapted
	//this operation is performed regarding the compute methode
	
	if(current->previous != NULL && current->previous->type == DENSE)
	{
		pos = ((dense_param*)current->previous->param)->in_size 
			* (((dense_param*)current->previous->param)->nb_neurons+1) - 1;
		value = (real) d_param->bias_value/((dense_param*)current->previous->param)->bias_value;
		cudaMemcpy(((dense_param*)current->previous->param)->weights + pos, &value, sizeof(real), cudaMemcpyHostToDevice);
	}
	
	if(current->previous != NULL)
	{
		switch(current->previous->type)
		{	
			case CONV:
				cuda_convert_table(&(d_param->flat_input), d_param->in_size*current->c_network->batch_size);
				cuda_convert_table(&(d_param->flat_delta_o),
					(((conv_param*)current->previous->param)->nb_area_w 
						* ((conv_param*)current->previous->param)->nb_area_h 
						* ((conv_param*)current->previous->param)->nb_filters + 1) 
						* current->c_network->batch_size);
				break;
				
			case POOL:
				cuda_convert_table(&(d_param->flat_input), d_param->in_size * current->c_network->batch_size);
				cuda_convert_table(&(d_param->flat_delta_o),
					(((pool_param*)current->previous->param)->nb_area_w 
						* ((pool_param*)current->previous->param)->nb_area_h 
						* ((pool_param*)current->previous->param)->nb_maps + 1) 
						* current->c_network->batch_size);
				break;
				
			case DENSE:
			default:
				d_param->flat_delta_o = current->previous->delta_o;
				break;
		}
	}
	
	cuda_convert_table(&(d_param->weights), d_param->in_size*(d_param->nb_neurons+1));
	cuda_convert_table(&(d_param->update), d_param->in_size*(d_param->nb_neurons+1));
	cuda_convert_table(&(d_param->dropout_mask), d_param->nb_neurons);
	cudaMalloc((void**) &d_param->block_state, (d_param->nb_neurons) * sizeof(curandState_t));
	cu_blocks = (d_param->nb_neurons);
	init_block_state<<< cu_blocks, 1>>>(time(NULL),(curandState_t*)d_param->block_state);
	
	cuda_convert_table(&(current->output), (d_param->nb_neurons+1) * current->c_network->batch_size);
	cuda_convert_table(&(current->delta_o), (d_param->nb_neurons+1) * current->c_network->batch_size);
}


void cuda_forward_dense_layer(layer *current)
{
	int nb_area_w, nb_area_h, depth;
	
	if(current->c_network->length == 0)
		return;
	
	d_param = (dense_param*) current->param;
	
	if(current->previous == NULL)
	{
		current->input = current->c_network->input;
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, 
			current->c_network->batch_size, d_param->in_size, &cu_alpha, d_param->weights, 
			d_param->nb_neurons+1, current->input, d_param->in_size, &cu_beta, 
			current->output, d_param->nb_neurons+1);
	}
	else if(current->previous->type == DENSE)
	{
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, 
			current->c_network->batch_size, d_param->in_size, &cu_alpha, d_param->weights, 
			d_param->nb_neurons+1, current->input, d_param->in_size, &cu_beta, 
			current->output, d_param->nb_neurons+1);
	}
	else if(current->previous->type != DENSE)
	{
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
		
		cu_blocks = ((nb_area_w * nb_area_h * depth + 1) 
			* current->c_network->batch_size + cu_threads - 1) / cu_threads;
		flat_dense<<< cu_blocks, cu_threads >>>(current->input, d_param->flat_input, d_param->bias_value, 
			nb_area_w * nb_area_h , nb_area_w * nb_area_h * depth + 1, depth, current->c_network->batch_size, 
			(nb_area_w * nb_area_h * depth + 1) * current->c_network->batch_size);
		
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_param->nb_neurons+1, 
			current->c_network->batch_size, d_param->in_size, &cu_alpha, d_param->weights, 
			d_param->nb_neurons+1, d_param->flat_input, d_param->in_size, &cu_beta, 
			current->output, d_param->nb_neurons+1);
	}
	
	cu_blocks = (d_param->nb_neurons);
	dropout_select<<<cu_blocks, 1>>>(d_param->dropout_mask, d_param->nb_neurons+1, d_param->dropout_rate,
		(curandState_t*) d_param->block_state);
	current->activation(current);
	

	dim3 threadsPerBlock(8, 32);
	dim3 numBlocks((current->c_network->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
			
	if(d_param->dropout_rate > 0.01)
		dropout_apply<<<numBlocks, threadsPerBlock>>>(current->output, current->c_network->batch_size, 
			d_param->nb_neurons, d_param->dropout_mask);
}


void cuda_backward_dense_layer(layer* current)
{
	int nb_area_w, nb_area_h, depth;

	d_param = (dense_param*) current->param;	
	
	dim3 threadsPerBlock(8, 32);
	dim3 numBlocks((current->c_network->batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(d_param->nb_neurons + threadsPerBlock.y - 1) / threadsPerBlock.y);
	if(d_param->dropout_rate > 0.01)
		dropout_apply<<<numBlocks, threadsPerBlock>>>(current->delta_o, current->c_network->batch_size, d_param->nb_neurons,
			d_param->dropout_mask);
	
	//######################## ERROR PROPAGATION ########################

	//skip error prop if previous is the input layer
	if(current->previous != NULL)
	{	
		cublasgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, d_param->in_size, current->c_network->batch_size,
			d_param->nb_neurons+1, &cu_alpha, d_param->weights, d_param->nb_neurons+1, current->delta_o,
			d_param->nb_neurons+1, &cu_beta, d_param->flat_delta_o, d_param->in_size);
		//if previous layer is dense then flat_delta_o = previous->delta_o
		
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
			
			//Need to unroll delta_o to already be in the proper format for deriv calculation
			cu_blocks = (nb_area_w * nb_area_h * depth 
				* current->c_network->batch_size + cu_threads - 1) / cu_threads;
			reroll_batch<<< cu_blocks, cu_threads >>>(d_param->flat_delta_o, current->previous->delta_o,
				nb_area_w * nb_area_h, nb_area_w * nb_area_h * depth + 1, depth, 
				current->c_network->batch_size, nb_area_w * nb_area_h * depth 
				* current->c_network->batch_size);
		}
		
		current->previous->deriv_activation(current->previous);
	}
	
		
	//########################  WEIGHTS UPDATE   ########################
	
	//based on the recovered delta_o provided by the next layer propagation
	if(current->previous != NULL && current->previous->type != DENSE)
	{
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
			current->c_network->batch_size, &current->c_network->learning_rate,	current->delta_o,
			d_param->nb_neurons+1, d_param->flat_input, d_param->in_size, &current->c_network->momentum, 
			d_param->update, d_param->nb_neurons+1);
	}
	else
	{		
		cublasgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, d_param->nb_neurons+1, d_param->in_size,
			current->c_network->batch_size, &current->c_network->learning_rate,	current->delta_o, 
			d_param->nb_neurons+1, current->input, d_param->in_size, &current->c_network->momentum,
			d_param->update, d_param->nb_neurons+1);
	}

	cu_blocks = (d_param->in_size*(d_param->nb_neurons+1) + cu_threads - 1) / cu_threads;
	cuda_update_weights<<< cu_blocks, cu_threads >>>(d_param->weights, d_param->update, 
		d_param->in_size*(d_param->nb_neurons+1));
}

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

__global__ void init_block_state(unsigned int seed,  curandState_t* states)
{
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}


__global__ void dropout_select(real* mask, int size, real drop_rate, curandState_t* states)
{
	int i = blockIdx.x;
	
	real rand;
	if(i < size)
	{
		rand = curand_uniform(&states[i]);
		if(rand < drop_rate)
			mask[i] = 0;
		else
			mask[i] = 1;
	}
}

__global__ void dropout_apply(real* table, real batch_size, int dim, real* mask)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(i < batch_size && j < dim)
	{
		table[i*(dim+1) + j] *= mask[j];
	}
}











