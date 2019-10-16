#include "prototypes.h"


//##############################
//       Local variables
//##############################
static dense_param *d_param;

//public are in "prototypes.h"


//private

void dense_define_activation_param(layer *current);


void dense_define_activation_param(layer *current)
{
	d_param = (dense_param*) current->param;
	switch(current->activation_type)
	{
		case RELU:
			current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
			((ReLU_param*)current->activ_param)->size = (d_param->nb_neurons + 1) 
				* current->c_network->batch_size;
			((ReLU_param*)current->activ_param)->dim = d_param->nb_neurons;
			((ReLU_param*)current->activ_param)->leaking_factor = 0.01;
			d_param->bias_value = 0.1;
			break;
			
		case LOGISTIC:
			current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
			((logistic_param*)current->activ_param)->size = (d_param->nb_neurons+1) 
				* current->c_network->batch_size;
			((logistic_param*)current->activ_param)->dim = d_param->nb_neurons;
			((logistic_param*)current->activ_param)->beta = 1.0;
			((logistic_param*)current->activ_param)->saturation = 14.0;
			d_param->bias_value = -1.0;
			break;
			
		case SOFTMAX:
			current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
			((softmax_param*)current->activ_param)->dim = d_param->nb_neurons;
			d_param->bias_value = -1.0;
			break;
			
		case LINEAR:
		default:
			current->activ_param = (linear_param*) malloc(sizeof(linear_param));
			((linear_param*)current->activ_param)->size = (d_param->nb_neurons + 1) 
				* current->c_network->batch_size;
			((linear_param*)current->activ_param)->dim = d_param->nb_neurons;
			//Change to expect output between 0 and 1
			d_param->bias_value = 0.1;
			//d_param->bias_value = -1.0;
			break;
	
	}
}


void dense_create(network *net, layer* previous, int nb_neurons, int activation, real drop_rate, FILE *f_load)
{
	int i, j;
	layer* current;
	
	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	d_param = (dense_param*) malloc(sizeof(dense_param));
	
	current->type = DENSE;
	current->activation_type = activation;
	d_param->nb_neurons = nb_neurons;
	d_param->dropout_rate = drop_rate;
	
	current->previous = previous;
	
	//WARNING : MUST ADAPT VALUE TO ACTIVATION FUNCTION !! IN REGARDE OF WEIGHTS
	d_param->bias_value = 0.1;
	
	
	if(previous == NULL)
	{
		d_param->in_size = net->input_width*net->input_height*net->input_depth+1;
		current->input = net->input;
	}
	else
	{
		switch(previous->type)
		{
			case CONV:
				d_param->in_size = ((conv_param*)previous->param)->nb_area_w
					* ((conv_param*)previous->param)->nb_area_h 
					* ((conv_param*)previous->param)->nb_filters + 1;
				d_param->flat_delta_o = (real*) calloc(d_param-> in_size * net->batch_size, sizeof(real));
				break;
			
			case POOL:
				d_param->in_size = ((pool_param*)previous->param)->nb_area_w 
					* ((pool_param*)previous->param)->nb_area_h * ((pool_param*)previous->param)->nb_maps + 1;
				d_param->flat_delta_o = (real*) calloc(d_param->in_size * net->batch_size, sizeof(real));
				((pool_param*)previous->param)->next_layer_type = current->type;
				break;
			
			case DENSE:
			default:
				d_param->in_size = ((dense_param*)previous->param)->nb_neurons+1;
				d_param->flat_delta_o = previous->delta_o;
				break;
		
		}
		current->input = previous->output;
		d_param->flat_input = (real*) malloc(d_param->in_size*net->batch_size*sizeof(real));
	}

	d_param->weights = (real*) malloc(d_param->in_size*(nb_neurons+1)*sizeof(real));
	
	d_param->update = (real*) calloc(d_param->in_size*(nb_neurons+1), sizeof(real));
	d_param->dropout_mask = (real*) calloc(d_param->nb_neurons, sizeof(real));
	
	current->output = (real*) calloc((nb_neurons+1)*net->batch_size, sizeof(real));
	current->delta_o = (real*) calloc((nb_neurons+1)*net->batch_size, sizeof(real));
	
	
	//must be before the association functions
	current->param = d_param;
	
	dense_define_activation_param(current);
	
	if(f_load == NULL)
	{
		printf("Xavier init\n");
		xavier_normal(d_param->weights, d_param->nb_neurons, d_param->in_size, 1);
	}
	else
	{
		for(i = 0; i < d_param->in_size; i++)
			for(j = 0; j < (d_param->nb_neurons+1); j++)
				fscanf(f_load, "%f", &(d_param->weights[i*(d_param->nb_neurons+1) + j]));
	}
	
	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_dense_define(current);
			cuda_define_activation(current);
			cuda_convert_dense_layer(current);
			#endif
			break;
			
		default:
			break;
	}
	
	#ifndef CUDA
	printf("ERROR : Non CUDA compute not implemented at the moment !\n");
	exit(EXIT_FAILURE);
	#endif
}


void dense_save(FILE *f, layer *current)
{
	int i, j;
	real* host_weights;

	d_param = (dense_param*)current->param;	
	
	fprintf(f,"D");
	fprintf(f, "%dn%fd", d_param->nb_neurons, d_param->dropout_rate);
	print_activ_param(f, current->activation_type);
	fprintf(f,"\n");
	
	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_weights = (real*) malloc(d_param->in_size * (d_param->nb_neurons+1) * sizeof(real));
		cuda_get_table(&(d_param->weights), &host_weights, d_param->in_size * (d_param->nb_neurons+1));
		#endif
	}
	
	for(i = 0; i < d_param->in_size; i++)
	{
		for(j = 0; j < (d_param->nb_neurons+1); j++)
			fprintf(f, "%g ", host_weights[i*(d_param->nb_neurons+1) + j]);
		fprintf(f, "\n");
	}
	fprintf(f, "\n");
	
	if(current->c_network->compute_method == C_CUDA)
		free(host_weights);
}

void dense_load(network *net, FILE* f)
{
	int nb_neurons;
	real dropout_rate;
	char activ_type[20];
	layer *previous;
	
	printf("Loading dense layer, L:%d\n", net->nb_layers);
	
	fscanf(f, "%dn%fd%s\n", &nb_neurons, &dropout_rate, activ_type);
	
	if(net->nb_layers <= 0)
		previous = NULL;
	else
		previous = net->net_layers[net->nb_layers-1];

	dense_create(net, previous, nb_neurons, load_activ_param(activ_type), dropout_rate, f);
}







