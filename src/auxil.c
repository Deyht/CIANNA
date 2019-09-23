#include "prototypes.h"


void init_network(int u_input_dim[3], int u_output_dim, int u_batch_size, int u_compute_method)
{
	#ifdef CUDA
	init_cuda();
	#endif

	input_width = u_input_dim[0]; 
	input_height = u_input_dim[1];
	input_depth = u_input_dim[2];
	input_dim = input_width*input_height*input_depth;
	output_dim = u_output_dim;
	batch_size = u_batch_size;
	compute_method = u_compute_method;
	
	output_error = (real*) calloc(batch_size*output_dim,sizeof(real));
	#ifdef CUDA
	if(compute_method == C_CUDA)
		cuda_create_table(&output_error_cuda, batch_size*output_dim);
	#endif
}

void enable_confmat(void)
{
	confusion_matrix = 1;
}


Dataset create_dataset(int nb_elem)
{
	int i;
	Dataset data;

	data.size = nb_elem;
	data.nb_batch = (data.size - 1) / batch_size + 1;
	data.input = (real**) malloc(data.nb_batch*sizeof(real*));
	data.target = (real**) malloc(data.nb_batch*sizeof(real*));
	data.localization = HOST;
	
	for(i = 0; i < data.nb_batch; i++)
	{
		data.input[i] = (real*) calloc(batch_size * (input_dim + 1), sizeof(real));
		data.target[i] = (real*) calloc(batch_size * output_dim, sizeof(real));
	}
	return data;
}

int argmax(real *tab, int size)
{
	int i;
	real max;
	int imax;

	max = *tab;
	imax = 0;
	
	
	for(i = 1; i < size; i++)
	{
		if(tab[i] >= max)
		{
			max = tab[i];
			imax = i;
		}
	}
	
	return imax;
}


void compute_error(Dataset data)
{
	int j, k, l;
	float** mat = NULL; 
	real* cuda_mat;
	float* temp = NULL;
	int arg1, arg2;
	real count;
	real *rapp_err = NULL, *rapp_err_rec = NULL;
	int o, pos;
	real total_error;
	real* temp_error;
	
	o = output_dim;
	
	if(confusion_matrix)
	{
		rapp_err = (real*) malloc(o*sizeof(real));
		rapp_err_rec = (real*) malloc(o*sizeof(real));
		mat = (float**) malloc(o*sizeof(float*));
		temp = (float*) malloc(o*o*sizeof(float));
		for(j = 0; j < o; j++)
			mat[j] = &(temp[j*o]);
			
		#ifdef CUDA
		if(compute_method == C_CUDA)
			cuda_create_table(&cuda_mat, o*o);
		#endif
	}
	
	total_error = 0.0;
	
	for(j = 0; j < data.nb_batch; j++)
	{
		if(compute_method == C_CUDA)
		{
			temp_error = output_error;
			output_error = output_error_cuda;
		}
		
		//##########################################################
		
		if(j == data.nb_batch - 1)
		{
			length = data.size%batch_size;
		}
		else
			length = batch_size;
		
		//Loop on all batch for one epoch
		input = data.input[j];
		target = data.target[j];
		//forward
		for(k = 0; k < nb_layers; k++)
		{
			net_layers[k]->forward(net_layers[k]);
		}
		output_error_fct(net_layers[nb_layers-1]);
		
		//##########################################################
		
		#ifdef CUDA
		if(compute_method == C_CUDA)
		{
			cuda_get_table(&output_error, &temp_error, batch_size*output_dim);
			output_error = temp_error;
		}
		#endif
		
		pos = 0;
		for(k = 0; k < length; k++)
		{
			for(l = 0; l < output_dim; l++)
			{
				pos++;
				total_error += output_error[pos];
			}
			
			if(confusion_matrix && compute_method != C_CUDA)
			{
				arg1 = argmax(&(target[k*output_dim]), output_dim);
				arg2 = argmax(&(net_layers[nb_layers-1]->output[k*(output_dim+1)]), output_dim);
				mat[arg1][arg2]++;

			}
		}
		
		if(compute_method == C_CUDA)
		{
			#ifdef CUDA
			cuda_confmat(net_layers[nb_layers-1]->output, cuda_mat);
			#endif
		}
	}
	printf("Cumulated error: \t %g\n", total_error/data.size);
	
	if(compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_get_table(&cuda_mat, mat, o*o);
		cuda_free_table(cuda_mat);
		#endif
	}
	
	if(confusion_matrix)
	{
		printf("\nConfMat (valid set)\n");
		for(j = 0; j < o; j++)
		{
			rapp_err[j] = 0.0;
			rapp_err_rec[j] = 0.0;
			for(k = 0; k < o; k++)
			{
				rapp_err[j] += mat[j][k];
				rapp_err_rec[j] += mat[k][j];
			}
			rapp_err[j] = mat[j][j]/rapp_err[j]*100.0;
			rapp_err_rec[j] = mat[j][j]/rapp_err_rec[j]*100.0;
		}
		printf("%*s\n", (o+2)*11, "Recall");
		for(j = 0; j < o; j++)
		{
			printf("%*s", 10, " ");
			for(k = 0; k < o; k++)
				printf("%10d |", (int) mat[j][k]);
			printf("\t %.2f%%\n", rapp_err[j]);
		}
		printf("%10s", "Precision :");
		for(j = 0; j < o; j++)
			printf("%9.2f%%  ", rapp_err_rec[j]);
		
		count = 0.0;
		for(j = 0; j < o; j++)
			count += mat[j][j];
		
		printf("Correct : %f%%\n", count/data.size*100);
		
		free(temp);
		free(mat);
		free(rapp_err);
		free(rapp_err_rec);
	}
}


void train_network(Dataset train_set, Dataset valid_set, int nb_epochs, int control_interv, real u_learning_rate, real u_momentum, real u_decay)
{
	int i, j, k;
	
	learning_rate = u_learning_rate;
	momentum = u_momentum;
	decay = u_decay;
	
	
	for(i = 0; i < nb_epochs; i++)
	{
		//Loop on all batch for one epoch
		for(j = 0; j < train_set.nb_batch; j++)
		{
			if(j == train_set.nb_batch-1)
				length = train_set.size%batch_size;
			else
				length = batch_size;
			
			input = train_set.input[j];
			target = train_set.target[j];
		
			//Forward on all layers
			for(k = 0; k < nb_layers; k++)
				net_layers[k]->forward(net_layers[k]);
			//cuda_print_table_transpose(net_layers[nb_layers-1]->output, batch_size, output_dim+1);
			//cuda_print_table_transpose(target, batch_size, output_dim);

			output_deriv_error(net_layers[nb_layers-1]);
			//Propagate error through all layers
			for(k = 0; k < nb_layers; k++)
				net_layers[nb_layers-1-k]->backprop(net_layers[nb_layers-1-k]);
		}
		
		if((i+1) % control_interv == 0)
		{
			printf("Epoch: %d\n", i+1);
			compute_error(valid_set);
			printf("\n");
		}
		
		learning_rate *= (1.0-decay);
	}

}


