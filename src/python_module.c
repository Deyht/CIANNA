
/*
	Copyright (C) 2020 David Cornu
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





#include <Python.h>
#include <numpy/arrayobject.h>

#include <string.h>
#include "prototypes.h"


// Structures or object related
//############################################################


// Network paramaremeter and data management functions
//############################################################

static PyObject* py_init_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	PyArrayObject *py_dims;
	int i;
	double bias = 0.1;
	int dims[3] = {1,1,1}, out_dim, b_size, comp_int = C_CUDA, network_id = nb_networks;
	int dynamic_load = 0, mixed_precision = 0;
	char string_comp[10];
	const char *comp_meth = "C_CUDA";
	static char *kwlist[] = {"dims", "out_dim", "bias", "b_size", "comp_meth", "network_id", "dynamic_load", "mixed_precision", NULL};
	
	b_size = 10;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oid|isiii", kwlist, &py_dims, &out_dim, &bias, &b_size, &comp_meth, &network_id, &dynamic_load, &mixed_precision))
	    return Py_None;
	
	for(i = 0; i < py_dims->dimensions[0]; i++)
	{
		dims[i] = *(int *)(py_dims->data + i*py_dims->strides[0]);
	}
	
	if(strcmp(comp_meth,"C_CUDA") == 0)
	{
		comp_int = C_CUDA;
		sprintf(string_comp, "CUDA");
	}
	else if(strcmp(comp_meth,"C_BLAS") == 0)
	{
		comp_int = C_BLAS;
		sprintf(string_comp, "BLAS");
	}
	else if(strcmp(comp_meth,"C_NAIV") == 0)
	{
		comp_int = C_NAIV;
		sprintf(string_comp, "NAIV");
	}
	
    init_network(network_id, dims, out_dim, bias, b_size, comp_int, dynamic_load, mixed_precision);
    
	printf("Network have been initialized with : \nInput dimensions: %dx%dx%d \nOutput dimension: %d \nBatch size: %d \nUsing %s compute methode\n\n", dims[0], dims[1], dims[2], out_dim, b_size, string_comp);
	if(dynamic_load)
		printf("Dynamic load ENABLED\n\n");
	
    return Py_None;
}


static PyObject* py_create_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i, j, k, l, m;
	Dataset *data = NULL;
	const char *dataset_type;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size, flat = 0, silent = 0;
	int network_id = nb_networks-1;
	static char *kwlist[] = {"dataset", "size", "input", "target", "flat", "network_id", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "siOO|iii", kwlist, &dataset_type, &size, &py_data, &py_target, &flat, &network_id, &silent))
	    return Py_None;
	
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		if(silent == 0)
			printf("Setting training set\n");
		data = &networks[network_id]->train;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		if(silent == 0)
			printf("Setting valid set\n");
		data = &networks[network_id]->valid;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		if(silent == 0)
			printf("Setting testing test\n");
		data = &networks[network_id]->test;
	}
	
	
	*data = create_dataset(networks[network_id], size);
	
	if(silent == 0)
	{
		printf("input dim :%d,", networks[network_id]->input_dim);
		printf("Creating dataset with size %d (nb_batch = %d) ... ", data->size, data->nb_batch);
	}
	
	if(py_data != NULL && py_target != NULL)
	{
		for(i = 0; i < data->nb_batch; i++)
		{
			for(j = 0; j < networks[network_id]->batch_size; j++)
			{
				if(i*networks[network_id]->batch_size + j >= data->size)
					continue;
				for(k = 0; k < networks[network_id]->input_depth; k++)
				{
					for(l = 0; l < networks[network_id]->input_height; l++)
						for(m = 0; m < networks[network_id]->input_width; m++)
						{
							if(!flat)
	((float*)data->input[i])[j*(networks[network_id]->input_dim+1) + k * networks[network_id]->input_height 
		* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
		= *((float*)(py_data->data + (i*networks[network_id]->batch_size 
		* networks[network_id]->input_depth*networks[network_id]->input_height 
		+ j*networks[network_id]->input_depth*networks[network_id]->input_height 
		+ k*networks[network_id]->input_height + l)*py_data->strides[0] + m*py_data->strides[1]));
							else
	((float*)data->input[i])[j*(networks[network_id]->input_dim+1) + k * networks[network_id]->input_height
		* networks[network_id]->input_width + l * networks[network_id]->input_width + m] 
		= *((float*)(py_data->data + (i * networks[network_id]->batch_size + j) 
		* py_data->strides[0] + (k*networks[network_id]->input_height 
		* networks[network_id]->input_width + l * networks[network_id]->input_width + m) 
		* py_data->strides[1]));	
						}
				}
			}
		}
		for(i = 0; i < data->nb_batch; i++)
			for(j = 0; j < networks[network_id]->batch_size; j++)
			{
				if(i*networks[network_id]->batch_size + j >= data->size)
					continue;
				for(k = 0; k < networks[network_id]->output_dim; k++)
					((float*)data->target[i])[j*networks[network_id]->output_dim + k] 
					= *(float*)(py_target->data + i * (networks[network_id]->batch_size 
					* py_target->strides[0]) + j * py_target->strides[0] + k 
					* py_target->strides[1]);
			}
	}
	printf("Done !\n");
	
	#ifdef CUDA
	if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->dynamic_load == 0)
	{
		if(silent == 0)
			printf("Converting dataset to GPU device (CUDA)\n");
		cuda_convert_dataset(networks[network_id], data);
	}
	else if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->dynamic_load == 1 && networks[network_id]->use_cuda_TC)
	{
		if(silent == 0)
			printf("Converting dataset into host stored FP16\n");
		cuda_convert_host_dataset_FP32(networks[network_id], data);
	}
	
	#endif
	if(silent == 0)
		printf("\n");
	
	return Py_None;
}


static PyObject* py_delete_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	const char *dataset_type;
	int network_id = nb_networks-1, silent = 0;
	Dataset *data = NULL;
	static char *kwlist[] = {"dataset", "network_id", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ii", kwlist, &dataset_type, &network_id, &silent))
		return Py_None;
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		if(silent == 0)
			printf("Deleting training set\n");
		data = &networks[network_id]->train;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		if(silent == 0)
			printf("Deleting valid set\n");
		data = &networks[network_id]->valid;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		if(silent == 0)
			printf("Deleting testing test\n");
		data = &networks[network_id]->test;
	}
	else
	{
		printf("Warning: No matching dataset to delete!\n");
	}

	free_dataset(*data);

	return Py_None;
}


static PyObject* py_write_formated_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i, j, k, l, m;
	const char *filename;
	const char *input_data_type, *output_data_type;
	int input_data_type_C = FP32, output_data_type_C = FP32;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size, flat = 0;
	int network_id = nb_networks-1;
	int datasize;
	int silent_mode = 0;
	static char *kwlist[] = {"filename", "size", "input", "input_dtype", "target", "output_dtype", "flat", "network_id", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "siOsOs|iii", kwlist, &filename, &size, &py_data, &input_data_type, &py_target, &output_data_type, &flat, &network_id, &silent_mode))
	    return Py_None;
	
	if(py_data == NULL || py_target == NULL)
	{
		printf("ERROR : non recognized numpy array in write formated dataset \n");
		exit(EXIT_FAILURE);
	}
	
	if(strcmp(input_data_type,"UINT8") == 0)
		input_data_type_C = c_UINT8;
	else if(strcmp(input_data_type,"UINT16") == 0)
		input_data_type_C = c_UINT16;
	else if(strcmp(input_data_type,"FP32") == 0)
		input_data_type_C = c_FP32;
	else
	{
		printf("ERROR : Unsuported datatype %s\n", input_data_type);
		exit(EXIT_FAILURE);
	}
	
	if(silent_mode == 0)
	{
		printf("Saving formated file: %s\n", filename);
	}
	
	FILE *f = NULL;
	f = fopen(filename, "wb");
	
	fwrite(&size, sizeof(int), 1, f);
	fwrite(&networks[network_id]->input_width, sizeof(int), 1, f);
	fwrite(&networks[network_id]->input_height, sizeof(int), 1, f);
	fwrite(&networks[network_id]->input_depth, sizeof(int), 1, f);
	fwrite(&networks[network_id]->output_dim, sizeof(int), 1, f);
	
	// As for the C function, this one could probably be strongly simplified to avoid repetition
	
	switch(input_data_type_C)
	{
		case c_UINT8:
		{
			unsigned char *temp_input;
			datasize = sizeof(unsigned char);
			temp_input = (unsigned char *) calloc(networks[network_id]->input_dim, datasize);
			
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
			{
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->input_depth; k++)
					{
						for(l = 0; l < networks[network_id]->input_height; l++)
							for(m = 0; m < networks[network_id]->input_width; m++)
							{
								if(!flat)
		temp_input[k * networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= (unsigned char) *(float*)(py_data->data + (i*networks[network_id]->batch_size 
			* networks[network_id]->input_depth*networks[network_id]->input_height 
			+ j*networks[network_id]->input_depth*networks[network_id]->input_height 
			+ k*networks[network_id]->input_height + l)*py_data->strides[0] + m*py_data->strides[1]);
								else
		temp_input[k * networks[network_id]->input_height
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= (unsigned char) *(float*)(py_data->data + (i * networks[network_id]->batch_size + j) 
			* py_data->strides[0] + (k*networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m) 
			* py_data->strides[1]);	
							}
					}
					fwrite(temp_input, datasize, networks[network_id]->input_dim, f);
					
				}
			}
			free(temp_input);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_input;
			datasize = sizeof(unsigned short);
			temp_input = (unsigned short *) calloc(networks[network_id]->input_dim, datasize);
			
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
			{
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->input_depth; k++)
					{
						for(l = 0; l < networks[network_id]->input_height; l++)
							for(m = 0; m < networks[network_id]->input_width; m++)
							{
								if(!flat)
		temp_input[k * networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= (unsigned short) *(float*)(py_data->data + (i*networks[network_id]->batch_size 
			* networks[network_id]->input_depth*networks[network_id]->input_height 
			+ j*networks[network_id]->input_depth*networks[network_id]->input_height 
			+ k*networks[network_id]->input_height + l)*py_data->strides[0] + m*py_data->strides[1]);
								else
		temp_input[k * networks[network_id]->input_height
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= (unsigned short) *(float*)(py_data->data + (i * networks[network_id]->batch_size + j) 
			* py_data->strides[0] + (k*networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m) 
			* py_data->strides[1]);	
							}
					}
					fwrite(temp_input, datasize, networks[network_id]->input_dim, f);
					
				}
			}
			free(temp_input);
			break;
		}
		case c_FP32:
		default:
		{
			float *temp_input;
			datasize = sizeof(float);
			temp_input = (float *) calloc(networks[network_id]->input_dim, datasize);
			
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
			{
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->input_depth; k++)
					{
						for(l = 0; l < networks[network_id]->input_height; l++)
							for(m = 0; m < networks[network_id]->input_width; m++)
							{
								if(!flat)
		temp_input[k * networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= *(float*)(py_data->data + (i*networks[network_id]->batch_size 
			* networks[network_id]->input_depth*networks[network_id]->input_height 
			+ j*networks[network_id]->input_depth*networks[network_id]->input_height 
			+ k*networks[network_id]->input_height + l)*py_data->strides[0] + m*py_data->strides[1]);
								else
		temp_input[k * networks[network_id]->input_height
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m]
			= *(float*)(py_data->data + (i * networks[network_id]->batch_size + j) 
			* py_data->strides[0] + (k*networks[network_id]->input_height 
			* networks[network_id]->input_width + l * networks[network_id]->input_width + m) 
			* py_data->strides[1]);	
							}
					}
					fwrite(temp_input, datasize, networks[network_id]->input_dim, f);
					
				}
			}
			free(temp_input);
			break;
		}
	}
	
	
	if(strcmp(output_data_type,"UINT8") == 0)
		output_data_type_C = c_UINT8;
	else if(strcmp(output_data_type,"UINT16") == 0)
		output_data_type_C = c_UINT16;
	else if(strcmp(output_data_type,"FP32") == 0)
		output_data_type_C = c_FP32;
	else
	{
		printf("ERROR : Unsuported datatype %s\n", output_data_type);
		exit(EXIT_FAILURE);
	}
	
	
	switch(output_data_type_C)
	{
		case c_UINT8:
		{
			unsigned char *temp_output;
			datasize = sizeof(unsigned char);
			temp_output = (unsigned char *) calloc(networks[network_id]->output_dim, datasize);
			
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->output_dim; k++)
						temp_output[k]
						= (unsigned char) *(float*)(py_target->data + i * (networks[network_id]->batch_size 
						* py_target->strides[0]) + j * py_target->strides[0] + k 
						* py_target->strides[1]);
					fwrite(temp_output, datasize, networks[network_id]->output_dim, f);
				}
			free(temp_output);
			break;
		}
			
		case c_UINT16:
		{
			unsigned short *temp_output;
			datasize = sizeof(unsigned short);
			temp_output = (unsigned short *) calloc(networks[network_id]->output_dim, datasize);
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->output_dim; k++)
						temp_output[k]
						= (unsigned short) *(float*)(py_target->data + i * (networks[network_id]->batch_size 
						* py_target->strides[0]) + j * py_target->strides[0] + k 
						* py_target->strides[1]);
					fwrite(temp_output, datasize, networks[network_id]->output_dim, f);
				}
			free(temp_output);
			break;
		}
			
		case c_FP32:
		default:
		{
			float *temp_output;
			datasize = sizeof(float);
			temp_output = (float *) calloc(networks[network_id]->output_dim, datasize);
			for(i = 0; i < (size - 1) / networks[network_id]->batch_size + 1; i++)
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					if(i*networks[network_id]->batch_size + j >= size)
						continue;
					for(k = 0; k < networks[network_id]->output_dim; k++)
						temp_output[k]
						= *(float*)(py_target->data + i * (networks[network_id]->batch_size 
						* py_target->strides[0]) + j * py_target->strides[0] + k 
						* py_target->strides[1]);
					fwrite(temp_output, datasize, networks[network_id]->output_dim, f);
				}
			free(temp_output);
			break;
		}
	}
	
	fclose(f);
	
	return Py_None;
}



static PyObject* py_load_formated_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	Dataset data;
	const char *dataset_type;
	const char *filename;
	const char *input_data_type, *output_data_type;
	int input_data_type_C = FP32, output_data_type_C = FP32;
	int network_id = nb_networks-1, silent = 0;
	static char *kwlist[] = {"dataset", "filename", "input_dtype", "output_dtype", "network_id", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ssss|ii", kwlist, &dataset_type, &filename, &input_data_type, &output_data_type, &network_id, &silent))
	    return Py_None;
	
	if(strcmp(input_data_type,"UINT8") == 0)
		input_data_type_C = c_UINT8;
	else if(strcmp(input_data_type,"UINT16") == 0)
		input_data_type_C = c_UINT16;
	else if(strcmp(input_data_type,"FP32") == 0)
		input_data_type_C = c_FP32;
	else
	{
		printf("ERROR : Unsuported datatype %s\n", input_data_type);
		exit(EXIT_FAILURE);
	}
	
	
	if(strcmp(output_data_type,"UINT8") == 0)
		output_data_type_C = c_UINT8;
	else if(strcmp(output_data_type,"UINT16") == 0)
		output_data_type_C = c_UINT16;
	else if(strcmp(output_data_type,"FP32") == 0)
		output_data_type_C = c_FP32;
	else
	{
		printf("ERROR : Unsuported datatype %s\n", output_data_type);
		exit(EXIT_FAILURE);
	}
	
	if(silent == 0)
		printf("Loading dataset from file %s ...",filename);
	
	data = load_formated_dataset(networks[network_id], filename, input_data_type_C, output_data_type_C);
	normalize_dataset(networks[network_id], data);
	
	if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->dynamic_load == 0)
		cuda_convert_dataset(networks[network_id], &data);
	else if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->dynamic_load == 1 
					&& networks[network_id]->use_cuda_TC)
		cuda_convert_host_dataset_FP32(networks[network_id], &data);
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		//printf("Training set loaded\n");
		networks[network_id]->train = data;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		//printf("Valid set loaded\n");
		networks[network_id]->valid = data;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		//printf("Testing test loaded\n");
		networks[network_id]->test = data;
	}
	
	if(silent == 0)
		printf(" Done!\n");
		
	return Py_None;
}




static PyObject* py_set_normalize_factors(PyObject* self, PyObject *args, PyObject *kwargs)
{
	PyArrayObject *offset_input, *norm_input, *offset_output, *norm_output;
	float *c_offset_input, *c_norm_input, *c_offset_output, *c_norm_output;
	int dim_size_input, dim_size_output;
	int network_id = nb_networks-1;
	static char *kwlist[] = {"offset_in", "norm_in", "dim_in", "offset_out", "norm_out", "dim_out", "network_id", NULL};
	int i;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiOOi|i", kwlist, &offset_input, &norm_input, &dim_size_input, &offset_output, &norm_output, &dim_size_output, &network_id))
		return Py_None;
		
	if(networks[network_id]->input_dim != dim_size_input*offset_input->dimensions[0])
	{
		printf("ERROR : Input dimensions do not match dataset to normalize ...\n");
		exit(EXIT_FAILURE);
	}
	
	if(networks[network_id]->output_dim != dim_size_output*offset_output->dimensions[0])
	{
		printf("ERROR : Input dimensions do not match dataset to normalize ...\n");
		exit(EXIT_FAILURE);
	}
	
	c_offset_input = (float*) calloc(offset_input->dimensions[0], sizeof(float));
	c_norm_input = (float*) calloc(norm_input->dimensions[0], sizeof(float));
	c_offset_output = (float*) calloc(offset_output->dimensions[0], sizeof(float));
	c_norm_output = (float*) calloc(norm_output->dimensions[0], sizeof(float));
	
	for(i = 0; i < offset_input->dimensions[0]; i++)
		c_offset_input[i] = *(float *)(offset_input->data + i*offset_input->strides[0]);
	for(i = 0; i < norm_input->dimensions[0]; i++)
		c_norm_input[i] = *(float *)(norm_input->data + i*norm_input->strides[0]);
	for(i = 0; i < offset_output->dimensions[0]; i++)
		c_offset_output[i] = *(float *)(offset_output->data + i*offset_output->strides[0]);
	for(i = 0; i < norm_output->dimensions[0]; i++)
		c_norm_output[i] = *(float *)(norm_output->data + i*norm_output->strides[0]);
	
	printf("%f %f %f %f\n", c_offset_input[0], c_norm_input[0], c_offset_output[0], c_norm_output[0]);
	
	set_normalize_dataset_parameters(networks[network_id], c_offset_input, c_norm_input, dim_size_input,
		c_offset_output, c_norm_output, dim_size_output);
	
	return Py_None;
}



// Layers functions
//############################################################
static PyObject* py_dense_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int nb_neurons, prev_layer = -1, i_activ = RELU, network_id = nb_networks-1;
	const char *activation = "RELU";
	double drop_rate = 0.0;
	static char *kwlist[] = {"nb_neurons", "activation", "prev_layer", "drop_rate", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|sidi", kwlist, &nb_neurons, &activation, &prev_layer, &drop_rate, &network_id))
	    return Py_None;

	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
	
	if(strcmp(activation, "RELU") == 0)
		i_activ = RELU;
	else if(strcmp(activation, "LINEAR") == 0)
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC") == 0)
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX") == 0)
		i_activ = SOFTMAX;
	else if(strcmp(activation, "YOLO") == 0)
		i_activ = YOLO;
	
	printf("prev : %d\n", prev_layer);
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	dense_create(networks[network_id], prev, nb_neurons, i_activ, drop_rate, NULL);
	
	return Py_None;
}

static PyObject* py_conv_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int f_size, nb_filters, stride, padding, prev_layer = -1, i_activ = RELU, network_id = nb_networks-1;
	const char *activation = "RELU";
	double drop_rate = 0.0;
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "activation", "prev_layer", "drop_rate", "network", NULL};
	layer* prev;
	
	stride = 1; padding = 0;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iisidi", kwlist, &f_size, &nb_filters, &stride, &padding, 
		&activation, &prev_layer, &drop_rate, &network_id))
	    return Py_None;
	    
	if(prev_layer == -1)
	    prev_layer = networks[network_id]->nb_layers - 1;

	if(strcmp(activation, "RELU") == 0)
		i_activ = RELU;
	if(strcmp(activation, "RELU_6") == 0)
		i_activ = RELU_6;
	else if(strcmp(activation, "LINEAR") == 0)
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC") == 0)
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX") == 0)
		i_activ = SOFTMAX;
	else if(strcmp(activation, "YOLO") == 0)
		i_activ = YOLO;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	conv_create(networks[network_id], prev, f_size, nb_filters, stride, padding, i_activ, drop_rate, NULL);
	
	return Py_None;
}


static PyObject* py_pool_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int pool_size = 2, prev_layer = -1, network_id = nb_networks-1;
	static char *kwlist[] = {"pool_size", "prev_layer", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii", kwlist, &pool_size, &prev_layer, &network_id))
	    return Py_None;
	    
	if(prev_layer == -1)
	    prev_layer = networks[network_id]->nb_layers - 1;

	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	pool_create(networks[network_id], prev, pool_size);
	
	return Py_None;
}

static PyObject* py_set_yolo_params(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i,j;
	int nb_box, nb_class, nb_param, IoU_type;
	int network_id = 0;
	PyArrayObject *py_prior_w = NULL, *py_prior_h = NULL, *py_prior_noobj_prob;
	float *C_prior_w, *C_prior_h, *C_prior_noobj_prob;
	PyArrayObject *py_error_scales = NULL, *py_slopes_and_maxes = NULL, *py_IoU_limits = NULL, *py_fit_parts = NULL;
	const char* IoU_type_char = "empty";
	float *error_scales = NULL, **slopes_and_maxes = NULL, *IoU_limits = NULL;
	int *fit_parts = NULL;
	float* temp;
	static char *kwlist[] = {"nb_box", "prior_w", "prior_h", "prior_noobj_prob", "nb_class", "nb_param", "error_scales", "slopes_and_maxes", "IoU_limits", "fit_parts", "IoU_type", "network", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "iOOOii|OOOOsi", kwlist, 
			&nb_box, &py_prior_w, &py_prior_h, &py_prior_noobj_prob, &nb_class, &nb_param, &py_error_scales, &py_slopes_and_maxes, &py_IoU_limits, &py_fit_parts, &IoU_type_char, &network_id))
	    return PyLong_FromLong(0);

	C_prior_w = (float*) calloc(nb_box,sizeof(float));
	C_prior_h = (float*) calloc(nb_box,sizeof(float));
	C_prior_noobj_prob = (float*) calloc(nb_box,sizeof(float));
	
	if(strcmp(IoU_type_char, "IoU") == 0)
		IoU_type = IOU;
	else if(strcmp(IoU_type_char, "GIoU") == 0)
		IoU_type = GIOU;
	else if(strcmp(IoU_type_char, "DIoU") == 0)
		IoU_type = DIOU;
	else
	{
		printf("Warning: Unrecognized IoU type: %s, fallback to default DIoU\n", IoU_type_char);
		IoU_type = DIOU;
	}
	
	for(i = 0; i < nb_box; i++)
	{ 
		C_prior_w[i] = *((float*)(py_prior_w->data + i * py_prior_w->strides[0]));
		C_prior_h[i] = *((float*)(py_prior_h->data + i * py_prior_h->strides[0]));
		C_prior_noobj_prob[i] = *((float*)(py_prior_noobj_prob->data 
									 + i * py_prior_noobj_prob->strides[0]));
	}
	
	if(py_error_scales != NULL)
	{
		error_scales = (float*) calloc(6, sizeof(float));
		for(i = 0; i < 6; i++)
			error_scales[i] = *(float *)(py_error_scales->data + i*py_error_scales->strides[0]);
	}
	
	if(py_slopes_and_maxes != NULL)
	{
		temp = (float*) calloc(6*3, sizeof(float));
		slopes_and_maxes = (float**) malloc(6*sizeof(float*));
		for(i = 0; i < 6; i++)
			slopes_and_maxes[i] = &temp[i*3];
		for(i = 0; i < 6; i++)
			for(j = 0; j < 3; j++)
				slopes_and_maxes[i][j] = *(float *)(py_slopes_and_maxes->data + i*py_slopes_and_maxes->strides[0] + j*py_slopes_and_maxes->strides[1]);
	}
	
	if(py_IoU_limits != NULL)
	{
		IoU_limits = (float*) calloc(5, sizeof(float));
		for(i = 0; i < 5; i++)
			IoU_limits[i] = *(float *)(py_IoU_limits->data + i*py_IoU_limits->strides[0]);
	}
	
	if(py_fit_parts != NULL)
	{
		fit_parts = (int*) calloc(5, sizeof(int));
		for(i = 0; i < 5; i++)
			fit_parts[i] = *(int *)(py_fit_parts->data + i*py_fit_parts->strides[0]);
	}
	
	return PyLong_FromLong(set_yolo_params(networks[network_id], nb_box, IoU_type, C_prior_w, C_prior_h, C_prior_noobj_prob, nb_class, nb_param, error_scales, slopes_and_maxes, IoU_limits, fit_parts));
}

static PyObject* perf_eval(PyObject* self, PyObject* args)
{
	int network_id = nb_networks - 1;
	if(!PyArg_ParseTuple(args, "|i", &network_id))
	    return Py_None;

	perf_eval_display(networks[network_id]);
	
	return Py_None;
}


static PyObject* py_load_network(PyObject* self, PyObject* args)
{
	char* file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int epoch, network_id = nb_networks-1;

	if(!PyArg_ParseTuple(args, "si|i", &file, &epoch, &network_id))
	    return Py_None;
	    
	load_network(networks[network_id], file, epoch);
	
	return Py_None;
}


// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int py_nb_epoch, py_control_interv = 1, py_confmat = 0, save_net = 0, network_id = nb_networks-1;
	int shuffle_gpu = 1, shuffle_every = 1, silent = 0;
	double py_learning_rate=0.02, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0, py_TC_scale_factor = 4.0;
	static char *kwlist[] = {"nb_epoch", "learning_rate", "end_learning_rate", "control_interv", "momentum", "decay", "confmat", "save_each", "network", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "silent", NULL};
	
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|diddiiiiidi", kwlist, &py_nb_epoch, &py_learning_rate, &py_end_learning_rate, &py_control_interv, &py_momentum, &py_decay, &py_confmat, &save_net, &network_id, &shuffle_gpu, &shuffle_every, &py_TC_scale_factor, &silent))
	    return Py_None;
	
	if(silent == 0)
		printf("py_nb_epoch %d, py_control_interv %d, py_learning_rate %g, py_end_learning_rate %g , py_momentum %0.2f, py_decay %g, py_confmat %d, save_net %d, shuffle_gpu %d , shuffle_every %d, TC_scale_factor %g\n", 
			py_nb_epoch, py_control_interv, py_learning_rate, py_end_learning_rate, py_momentum, 
			py_decay, py_confmat, save_net, shuffle_gpu, shuffle_every, py_TC_scale_factor);
		
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS
		
	train_network(networks[network_id], py_nb_epoch, py_control_interv, 
		py_learning_rate, py_end_learning_rate, py_momentum, py_decay, py_confmat, save_net, shuffle_gpu, shuffle_every, py_TC_scale_factor);
		
	Py_END_ALLOW_THREADS

	return Py_None;
}


static PyObject* py_forward_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int step = -1, repeat = 1, network_id = nb_networks-1, C_drop_mode = AVG_MODEL, no_error = 0, saving = 1;
	const char *drop_mode = "AVG_MODEL";
	static char *kwlist[] = {"step", "repeat", "network_id", "saving", "drop_mode", "no_error", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiisi", kwlist, &step, &repeat, &network_id, &saving, &drop_mode, &no_error))
	    return Py_None;
	
	if(step == -1)
		step = networks[network_id]->epoch;
	
	if(strcmp(drop_mode, "AVG_MODEL") == 0)
                C_drop_mode = AVG_MODEL;
        else if(strcmp(drop_mode, "MC_MODEL") == 0)
                C_drop_mode = MC_MODEL;

	networks[network_id]->no_error = no_error;
	
	forward_testset(networks[network_id], step, saving, repeat, C_drop_mode);
	
	return Py_None;
}

// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
    { "init_network", (PyCFunction)py_init_network, METH_VARARGS | METH_KEYWORDS, "Initialize network basic sahpes and properties" },
    { "create_dataset", (PyCFunction)py_create_dataset, METH_VARARGS | METH_KEYWORDS, "Allocate dataset structure" },
    { "delete_dataset", (PyCFunction)py_delete_dataset, METH_VARARGS | METH_KEYWORDS, "Free dataset structure" },
    { "write_formated_dataset", (PyCFunction)py_write_formated_dataset, METH_VARARGS | METH_KEYWORDS, "Write a proper numpy table onto a formated binary dataset file"},
    { "load_formated_dataset", (PyCFunction)py_load_formated_dataset, METH_VARARGS | METH_KEYWORDS, "Read a formated binary dataset file and directly store it into a network dataset"},
    { "set_normalize_factors", (PyCFunction)py_set_normalize_factors, METH_VARARGS | METH_KEYWORDS, "Set normalization factor for transformation on all the datasets subsequently loaded (with formated loading) in the C framework"},
    { "dense_create", (PyCFunction)py_dense_create, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
    { "conv_create",(PyCFunction)py_conv_create, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
    { "set_yolo_params",(PyCFunction)py_set_yolo_params, METH_VARARGS | METH_KEYWORDS, "Set parameters for YOLO output layout" },
    { "pool_create",(PyCFunction)py_pool_create, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
    { "perf_eval", perf_eval, METH_VARARGS, "Display each layer time in ms and in percent of the total networj time" },
    { "load_network", py_load_network, METH_VARARGS, "Load a previous network structure pre-trained from a file" },
    { "train_network", (PyCFunction)py_train_network, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
    { "forward_network", (PyCFunction)py_forward_network, METH_VARARGS | METH_KEYWORDS, "Apply the trained network to the test set and save results" },
    { NULL, NULL, 0, NULL }
};


static struct PyModuleDef CIANNA = {
    PyModuleDef_HEAD_INIT,
    "CIANNA",
    "Convolutional Interactive Artificial Neural Network by/for Astrophysicists",
    -1,
    CIANNAMethods
};


PyMODINIT_FUNC PyInit_CIANNA(void)
{
	setbuf(stdout, NULL);
	import_array();
	
	printf("###################################################################\n\
Importing CIANNA Python module V-p.0.5.2 , by D.Cornu\n\
###################################################################\n\n");

	PyObject *m;
    m = PyModule_Create(&CIANNA);
    if (m == NULL)
        return NULL;
	
    return m;
}








