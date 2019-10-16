
#include "prototypes.h"


int main()
{
	FILE *f_cmds, *f_profiles;
	int i, j, k, l;
	int dims1[3];
	int b_size;
	
	int out_dim1;
	
	int nb_obj = 145000;
	int nb_test = 5000;
	real dump;
	
	FILE *f_c_1, *f_c_2;
	real *map1, *map2; 
	int c_1_size, c_2_size;
	
	real *CMDs, *profiles;
	
	//#################################################################################
	//               WARNING ! This file do not perform training :
	//				For an example on how to use the framework see "main_old.c"
	//#################################################################################

	//Common randomized seed based on time at execution
	srand(time(NULL));
	
	CMDs = (real*) calloc(nb_obj*64*64, sizeof(real));
	profiles = (real*) calloc(nb_obj*100, sizeof(real));
	
	f_cmds = fopen("../raw_data/fancy/test3/CMDs.txt", "r+");
	
	
	for(i = 0; i < nb_obj*64*64; i++)
	{
		fscanf(f_cmds, "%f", &dump);
		CMDs[i] /= 1798.;
	}
	
	for(i = 0; i < nb_test*64*64; i++)
	{
		fscanf(f_cmds, "%f", &CMDs[i]);
		CMDs[i] /= 1798.;
	}
	
	

	fclose(f_cmds);
	
	
	
	f_profiles = fopen("../raw_data/fancy/test3/Profiles.txt", "r+");
	
	
	for(i = 0; i < nb_obj*100; i++)
	{
		fscanf(f_profiles, "%f", &dump);
		profiles[i] /= 6.0;
	}
	
	for(i = 0; i < nb_test*100; i++)
	{
		fscanf(f_profiles, "%f", &profiles[i]);
		profiles[i] /= 6.0;
	}
	
	fclose(f_profiles);


	dims1[0] = 64; dims1[1] = 64; dims1[2] = 1;
	out_dim1 = 100;

	b_size = 16;
	init_network(0, dims1, out_dim1, b_size, C_CUDA);
	
	load_network(networks[0], "net_save_acitv_maps.dat", 400);
	
	networks[0]->test = create_dataset(networks[0], nb_test, 0.1);
	
	#ifdef CUDA
	cuda_convert_dataset(networks[0], &networks[0]->train);
	#endif
	
	f_c_1 = fopen("activ_map_1.txt", "w+");
	f_c_2 = fopen("activ_map_2.txt", "w+");
	
	c_1_size = ((conv_param*)networks[0]->net_layers[0]->param)->nb_filters * ((conv_param*)networks[0]->net_layers[0]->param)->nb_area_w * ((conv_param*)networks[0]->net_layers[0]->param)->nb_area_h;
	c_2_size = ((conv_param*)networks[0]->net_layers[2]->param)->nb_filters * ((conv_param*)networks[0]->net_layers[2]->param)->nb_area_w * ((conv_param*)networks[0]->net_layers[2]->param)->nb_area_h;
	
	
	map1 = (real*) malloc(networks[0]->batch_size * c_1_size * sizeof(real));
	map2 = (real*) malloc(networks[0]->batch_size * c_2_size * sizeof(real));
	
	for(j = 0; j < networks[0]->test.nb_batch; j++)
	{

		if(j == networks[0]->test.nb_batch - 1 && networks[0]->test.size%networks[0]->batch_size > 0)
		{
			networks[0]->length = networks[0]->test.size%networks[0]->batch_size;
		}
		else
			networks[0]->length = networks[0]->batch_size;
		
		networks[0]->input = networks[0]->test.input[j];
		networks[0]->target = networks[0]->test.target[j];
		//forward
		for(k = 0; k < 3; k++)
		{
			networks[0]->net_layers[k]->forward(networks[0]->net_layers[k]);
		}
		
		cuda_get_table(&networks[0]->net_layers[0]->output, &map1, networks[0]->batch_size * c_1_size);
		cuda_get_table(&networks[0]->net_layers[2]->output, &map2, networks[0]->batch_size * c_2_size);
		
		for(k = 0; k < networks[0]->batch_size; k++)
		{
			for(l = 0; l < c_1_size ; l++)
				fprintf(f_c_1, "%g ", map1[l]);
			fprintf(f_c_1, "\n");
			
			for(l = 0; l < c_2_size ; l++)
				fprintf(f_c_2, "%g ", map2[l]);
			fprintf(f_c_1, "\n");
		}
	}
	
	free(map1);
	free(map2);
	
	fclose(f_c_1);
	fclose(f_c_2);
	
	

	exit(EXIT_SUCCESS);
}





