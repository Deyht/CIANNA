
/*
	Copyright (C) 2026-... David Cornu
	for the Convolutional Interactive Artificial 
	Neural Networks by/for Astrophysicists (CIANNA) Code
	(https://github.com/Deyht/CIANNA)

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/

#include "prototypes.h"

// Local variables

// Public are in prototypes.h

// Private prototypes


int input_create(network *net, int *spatial_shape, int nb_features, int silent)
{
	int i;
	int flat_spatial_dim;
	layer *current;

	current = (layer*) malloc(sizeof(layer));
	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;
	
	if(!silent)
		printf("L:%d - CREATING INPUT LAYER ...\n", net->nb_layers - 1);
	
	current->type                = INPUT;
	current->layer_id            = net->nb_layers-1;
	current->frozen              = 1;
	current->wema_replace_signal = 0;
	current->dropout_rate        = 0.0f;
	current->nb_params           = 0;
	
	//Use of existing variable instead of creating an input_param structure with a single argument
	current->activation_type = net->nb_branch_in;
	net->nb_branch_in++;
	
	current->output_dim = (int*) calloc(4, sizeof(int));
	
	flat_spatial_dim = 1;
	for(i = 0; i < 3; i++)
	{
		current->output_dim[i] = spatial_shape[i];
		flat_spatial_dim *= spatial_shape[i];
	}
	current->output_dim[3] = nb_features;
	
	if(flat_spatial_dim <= 1)
		current->output_type = FLAT;
	else
		current->output_type = SPATIAL;

	naiv_input_define(current);
	
	if(!silent)
		printf("      Input branch: %d, Shape:%dx%dx%dx%d\n",
			current->activation_type, current->output_dim[0], current->output_dim[1],
			current->output_dim[2], current->output_dim[3]);
	
	return net->nb_layers - 1;
}


void input_save(FILE *f, layer *current, int f_bin)
{
	char layer_type = 'I';
	
	// While branch_id is stored at execution it is not saved.
	// Branch_id increases as input layer are created.
	// This allows partial network loading, which would be difficult with a fixed saved id
	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(current->output_dim, sizeof(int), 4, f);
	}
	else
	{	
		fprintf(f,"I %dx%dx%dx%d", current->output_dim[0], current->output_dim[1], 
			current->output_dim[2], current->output_dim[3]);
		fprintf(f,"\n");
	}
}


void input_load(network *net, FILE* f, int f_bin, int skip_layer, int silent)
{
	int spatial_dim[3];
	int nb_features;
	
	if(!skip_layer && !silent)
		printf("Loading input layer, L:%d\n", net->nb_layers);
	
	if(f_bin)
	{
		fread(spatial_dim, sizeof(int), 3, f);
		fread(&nb_features, sizeof(int), 1, f);
	}
	else
		fscanf(f, " %dx%dx%dx%d\n", &spatial_dim[0], &spatial_dim[1], &spatial_dim[2], &nb_features);

	if(!skip_layer)
		input_create(net, spatial_dim, nb_features, silent);
}


void free_input(layer *current)
{
	free(current->output_dim);
	free(current);
}





