
#include "prototypes.h"


int main()
{
	FILE *f_cmds, *f_profiles;
	int i, j, k, l, m;
	//int size, freq, dim_c;
	//float b_x, e_x, b_y, e_y;
	//float max_ext, max_dist, dist_res;
	int dims1[3], dims2[3];
	int b_size, pos;
	char net_save_file_name[200];
	
	real total_error;
	real* temp_error = NULL;
	int out_dim1, out_dim2;
	real *net_1_temp_out;
	real *net_1_temp_back;
	
	int nb_obj = 80000;
	int nb_test = 3000;
	
	real *CMDs, *profiles;
	real *CMDs_test, *profiles_test;
	
	//Common randomized seed based on time at execution
	//srand(time(NULL));
	
	CMDs = (real*) calloc(nb_obj*64*64, sizeof(real));
	profiles = (real*) calloc(nb_obj*100, sizeof(real));
	
	CMDs_test = (real*) calloc(nb_test*64*64, sizeof(real));
	profiles_test = (real*) calloc(nb_test*100, sizeof(real));
	
	
	f_cmds = fopen("../raw_data/fancy/test3/CMDs.txt", "r+");

	//fscanf(f_cmds, "%d %d %d\n%f %f %f %f\n", &size, &freq, &dim_c, &b_x, &e_x, &b_y, &e_y);
	
	
	for(i = 0; i < nb_obj*64*64; i++)
	{
		fscanf(f_cmds, "%f", &CMDs[i]);
		CMDs[i] /= 2000.;
	}
	
	for(i = 0; i < nb_test*64*64; i++)
	{
		fscanf(f_cmds, "%f", &CMDs_test[i]);
		CMDs_test[i] /= 2000.;
	}
	
	

	fclose(f_cmds);
	
	
	
	f_profiles = fopen("../raw_data/fancy/test3/Profiles.txt", "r+");
	
	//fscanf(f_profiles, "%f %f %f\n",  &max_ext, &max_dist, &dist_res);
	
	for(i = 0; i < nb_obj*100; i++)
	{
		fscanf(f_profiles, "%f", &profiles[i]);
		profiles[i] /= 9.0;
	}
	
	for(i = 0; i < nb_test*100; i++)
	{
		fscanf(f_profiles, "%f", &profiles_test[i]);
		profiles_test[i] /= 9.0;
	}
	
	fclose(f_profiles);
	


	dims1[0] = 64; dims1[1] = 64; dims1[2] = 1;
	out_dim1 = 1024;

	b_size = 4;
	init_network(0, dims1, out_dim1, b_size, C_CUDA);
	
	net_1_temp_out = (real*) calloc(b_size*(out_dim1+1), sizeof(real));
	net_1_temp_back = (real*) calloc(b_size*(out_dim1+1+1)*100, sizeof(real));
	networks[0]->learning_rate = 0.001;
	networks[0]->momentum = 0.0;
	
	dims2[0] = (out_dim1 + 1); dims2[1] = 1; dims2[2] = 1;
	out_dim2 = 1;
	
	init_network(1, dims2, out_dim2, b_size*100, C_CUDA);
	networks[1]->learning_rate = 0.001;
	networks[1]->momentum = 0.0;

	//Network 0, convolution of CMDs
	conv_create(networks[0], NULL, 3, 8, 1, 0, RELU, NULL);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 8, 1, 0, RELU, NULL);
	pool_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 2);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 12, 1, 0, RELU, NULL);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 12, 1, 0, RELU, NULL);
	pool_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 2);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 24, 1, 0, RELU, NULL);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 24, 1, 0, RELU, NULL);
	conv_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 3, 24, 1, 0, RELU, NULL);
	dense_create(networks[0], networks[0]->net_layers[networks[0]->nb_layers-1], 
		networks[0]->output_dim, RELU, 0.0, NULL);
		
	//Network 1, dense network that combine conv results with scan in distance
	dense_create(networks[1], NULL, networks[1]->input_dim, RELU, 0.0, NULL);
	dense_create(networks[1], networks[1]->net_layers[networks[1]->nb_layers-1], 1024, RELU, 0.0, NULL);
	dense_create(networks[1], networks[1]->net_layers[networks[1]->nb_layers-1], 
		networks[1]->output_dim, LINEAR, 0.0, NULL);
	
	//global loo over epochs
	for(i = 0; i < 100; i++)
	{
		printf("Epoch %d\n", i+1);
		
		total_error = 0.0;
		networks[0]->epoch++;
		networks[1]->epoch++;
		for(j = 0; j < nb_obj/b_size; j++)
		{
			//Network 0, forward
			networks[0]->train = create_dataset(networks[0], b_size, 0.1);
			for(k = 0; k < b_size*networks[0]->input_dim; k++)
			{
				networks[0]->train.input[0][k + k/(networks[0]->input_dim)] =
					CMDs[j * b_size * networks[0]->input_dim + k];
			}
			
			#ifdef CUDA
			cuda_convert_dataset(networks[0], &networks[0]->train);
			#endif
			
			networks[0]->input = networks[0]->train.input[0];
			networks[0]->length = networks[0]->batch_size;
			
			for(k = 0; k < networks[0]->nb_layers; k++)
			{
				networks[0]->net_layers[k]->forward(networks[0]->net_layers[k]);
			}
			free_dataset(networks[0]->train);
			
			cuda_get_table(&(networks[0]->net_layers[networks[0]->nb_layers-1]->output), &net_1_temp_out, b_size*(networks[0]->output_dim+1));
			
			
			//Network 1, forward
			
			networks[1]->train = create_dataset(networks[1], b_size*100, 0.1);
			for(m = 0; m < b_size; m++)
			{
				for(l = 0; l < 100; l++)
				{
					for(k = 0; k < (networks[1]->input_dim-1); k++)
					{
						networks[1]->train.input[0][m * (100*(networks[1]->input_dim+1)) + l*(networks[1]->input_dim+1) + k] =
								net_1_temp_out[k + m * (networks[1]->input_dim)];
					}
				}
			}
			
			for(k = 0; k < b_size*100; k++)
			{
				networks[1]->train.input[0][(networks[1]->input_dim-1) + k * (networks[1]->input_dim+1)] = 0.1*0.1*(k+1) - k/100;
				networks[1]->train.target[0][k] = profiles[j*b_size*100 + k];
			} 
			
			#ifdef CUDA
			cuda_convert_dataset(networks[1], &networks[1]->train);
			#endif
			
			
			networks[1]->input = networks[1]->train.input[0];
			networks[1]->length = networks[1]->batch_size;
			
			networks[1]->net_layers[0]->output = networks[1]->input;
			//skip the first "fake" layer
			for(k = 1; k < networks[1]->nb_layers; k++)
			{
				networks[1]->net_layers[k]->forward(networks[1]->net_layers[k]);
			}
			
			//Compute error out of the network 1
			networks[1]->target = networks[1]->train.target[0];
			output_deriv_error(networks[1]->net_layers[networks[1]->nb_layers-1]);
			
			//Backprop error
			for(k = 0; k < networks[1]->nb_layers-1; k++)
				networks[1]->net_layers[networks[1]->nb_layers-1-k]->backprop(networks[1]->net_layers[networks[1]->nb_layers-1-k]);
			
			cuda_get_table(&(networks[1]->net_layers[0]->delta_o), &net_1_temp_back,
				100*b_size*(networks[0]->output_dim+1+1));

			for(k = 0; k < b_size * (networks[0]->output_dim+1); k++)
				net_1_temp_out[k] = 0.0;
			
			free_dataset(networks[1]->train);
			
			for(m = 0; m < b_size; m++)
			{
				for(l = 0; l < 100; l++)
				{
					for(k = 0; k < (networks[1]->input_dim-1); k++)
					{
						net_1_temp_out[m * (networks[0]->output_dim+1) + k] += 
							net_1_temp_back[m * (100*(networks[1]->input_dim+1)) + l 
							* (networks[1]->input_dim+1) + k];
					}
				}
			}
			
			
			#ifdef CUDA
			cuda_put_table(&networks[0]->net_layers[networks[0]->nb_layers-1]->delta_o,
				&net_1_temp_out,b_size*(networks[0]->output_dim+1));
			#endif
			
			for(k = 0; k < networks[0]->nb_layers; k++)
				networks[0]->net_layers[networks[0]->nb_layers-1-k]->backprop(networks[0]->net_layers[networks[0]->nb_layers-1-k]);
			
			
		}
		
		//#################################################################
		//          Apply network on test set to compute error
		//#################################################################
		
		if((i+1) % 1 == 0)
		{
			for(j = 0; j < nb_test/b_size; j++)
			{
				//Network 0, forward
				networks[0]->train = create_dataset(networks[0], b_size, 0.1);
				for(k = 0; k < b_size*networks[0]->input_dim; k++)
				{
					networks[0]->train.input[0][k + k/(networks[0]->input_dim)] =
						CMDs_test[j * b_size * networks[0]->input_dim + k];
				}
				
				#ifdef CUDA
				cuda_convert_dataset(networks[0], &networks[0]->train);
				#endif
				
				networks[0]->input = networks[0]->train.input[0];
				networks[0]->length = networks[0]->batch_size;
				
				for(k = 0; k < networks[0]->nb_layers; k++)
				{
					networks[0]->net_layers[k]->forward(networks[0]->net_layers[k]);
				}
				free_dataset(networks[0]->train);
				
				cuda_get_table(&(networks[0]->net_layers[networks[0]->nb_layers-1]->output), &net_1_temp_out, b_size*(networks[0]->output_dim+1));
				
				//Network 1, forward
				
				networks[1]->train = create_dataset(networks[1], b_size*100, 0.1);
				for(m = 0; m < b_size; m++)
				{
					for(l = 0; l < 100; l++)
					{
						for(k = 0; k < (networks[1]->input_dim-1); k++)
						{
							networks[1]->train.input[0][m * (100*(networks[1]->input_dim+1)) + l*(networks[1]->input_dim+1) + k] =
									net_1_temp_out[k + m * (networks[1]->input_dim)];
						}
					}
				}
				
				for(k = 0; k < b_size*100; k++)
				{
					networks[1]->train.input[0][(networks[1]->input_dim-1) + k * (networks[1]->input_dim+1)] = 0.1*0.1*(k+1) - k/100;
					networks[1]->train.target[0][k] = profiles_test[j*b_size*100 + k];
				} 
				
				#ifdef CUDA
				cuda_convert_dataset(networks[1], &networks[1]->train);
				#endif
				
				networks[1]->input = networks[1]->train.input[0];
				networks[1]->length = networks[1]->batch_size;
				
				networks[1]->net_layers[0]->output = networks[1]->input;
				networks[1]->net_layers[1]->input = networks[1]->input;
				//skip the first "fake" layer
				for(k = 1; k < networks[1]->nb_layers; k++)
				{
					networks[1]->net_layers[k]->forward(networks[1]->net_layers[k]);
				}
				
				
				//Compute error out of the network 1
				
				temp_error = networks[1]->output_error;
				networks[1]->output_error = networks[1]->output_error_cuda;
				networks[1]->target = networks[1]->train.target[0];
				
				output_error_fct(networks[1]->net_layers[networks[1]->nb_layers-1]);
				
				cuda_get_table(&networks[1]->output_error, &temp_error, networks[1]->batch_size*networks[1]->output_dim);
				networks[1]->output_error = temp_error;	
				
				free_dataset(networks[1]->train);
				
				pos = 0;
				for(k = 0; k < networks[1]->length; k++)
				{
					for(l = 0; l < networks[1]->output_dim; l++)
					{
						pos++;
						total_error += networks[1]->output_error[pos];
					}
				}
			}
			
			printf("Cumulated error: \t %g\n", total_error/(nb_test));
		}
		
		if((i+1)% 5 == 0)
		{
			sprintf(net_save_file_name, "net_save/net%d_s%04d.dat", networks[0]->id, networks[0]->epoch);
			printf("Saving network %d for epoch: %d\n",  networks[0]->id, networks[0]->epoch);
			save_network(networks[0], net_save_file_name);
			
			sprintf(net_save_file_name, "net_save/net%d_s%04d.dat", networks[1]->id, networks[1]->epoch);
			printf("Saving network %d for epoch: %d\n",  networks[1]->id, networks[1]->epoch);
			save_network(networks[1], net_save_file_name);
		}
	 }
	
	
	
	
	

	exit(EXIT_SUCCESS);
}





