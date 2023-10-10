
/*
	Copyright (C) 2023 David Cornu
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


#define NPY_NO_DEPRECATED_API 0

#define PY_SSIZE_T_CLEAN
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
	int dims[4] = {1,1,1,1}, nb_channels = 1, out_dim, b_size=8, network_id = nb_networks;
	int dynamic_load = 1, no_logo = 0, adv_size = 0, inference_only = 0;
	const char *py_mixed_precision = "off";
	const char *comp_meth = "C_CUDA";
	static char *kwlist[] = {"in_dim", "in_nb_ch", "out_dim", "bias", "b_size", "comp_meth", "network", 
		"dynamic_load", "mixed_precision", "inference_only", "no_logo", "adv_size", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii|disiisiii", kwlist, &py_dims, &nb_channels, &out_dim, &bias, 
		&b_size, &comp_meth, &network_id, &dynamic_load, &py_mixed_precision, &inference_only, &no_logo, &adv_size))
		return Py_None;
	
	for(i = 0; i < py_dims->dimensions[0] && i < 3; i++)
	{
		dims[i] = *(int *)(py_dims->data + i*py_dims->strides[0]);
	}
	dims[3] = nb_channels;
	
	init_network(network_id, dims, out_dim, bias, b_size, comp_meth, dynamic_load, py_mixed_precision, inference_only, no_logo, adv_size);
	
	return Py_None;
}


static PyObject* py_create_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i, j, k, l;
	Dataset *data = NULL;
	const char *dataset_type;
	float *py_cont_array;
	int c_array_offset = 0;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size, silent = 0;
	int flat_image_size = 0;
	int network_id = nb_networks-1;
	static char *kwlist[] = {"dataset", "size", "input", "target", "network", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "siOO|ii", kwlist, &dataset_type, &size, &py_data, &py_target, &network_id, &silent))
		return Py_None;
	
	Py_BEGIN_ALLOW_THREADS
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		if(silent == 0)
			printf("Setting train set\n");
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
			printf("Setting test set\n");
		data = &networks[network_id]->test;
	}
	else if(strcmp(dataset_type,"TRAIN_buf") == 0)
	{
		if(silent == 0)
			printf("Setting train buffer set\n");
		data = &networks[network_id]->train_buf;
	}
	else if(strcmp(dataset_type,"VALID_buf") == 0)
	{
		if(silent == 0)
			printf("Setting valid buffer set\n");
		data = &networks[network_id]->valid_buf;
	}
	else if(strcmp(dataset_type,"TEST_buf") == 0)
	{
		if(silent == 0)
			printf("Setting test buffer set\n");
		data = &networks[network_id]->test_buf;
	}
	
	struct timeval time;
	init_timing(&time);
	
	*data = create_dataset(networks[network_id], size);
	//printf("Time raw create %f\n",ellapsed_time(time));
	
	if(silent == 0)
	{
		printf("input dim :%ld,", networks[network_id]->input_dim);
		printf("Creating dataset with size %d (nb_batch = %d) ... ", data->size, data->nb_batch);
	}
	
	init_timing(&time);
	
	flat_image_size = networks[network_id]->in_dims[2]*networks[network_id]->in_dims[1]*networks[network_id]->in_dims[0];
	
	if(py_data != NULL && py_target != NULL)
	{
		py_cont_array = (float*) calloc(flat_image_size*networks[network_id]->in_dims[3], sizeof(float));
		for(i = 0; i < data->nb_batch; i++)
		{
			for(j = 0; j < networks[network_id]->batch_size; j++) //data_size/batch_size == 0 not allowed for now here
			{
				if(i*networks[network_id]->batch_size + j >= data->size)
					continue;
				c_array_offset = j*(networks[network_id]->input_dim + 1);
				for(l = 0; l < flat_image_size*networks[network_id]->in_dims[3]; l++)
				{
					py_cont_array[l] = *((float*)(py_data->data + (i * networks[network_id]->batch_size + j)
						* py_data->strides[0] + l* py_data->strides[1]));
				}
				data->cont_copy(py_cont_array, data->input[i], c_array_offset, flat_image_size*networks[network_id]->in_dims[3]);
			}
		}
		free(py_cont_array);
		py_cont_array = (float*) calloc(networks[network_id]->output_dim, sizeof(float));
		for(i = 0; i < data->nb_batch; i++)
		{
			for(j = 0; j < networks[network_id]->batch_size; j++)
			{
				if(i*networks[network_id]->batch_size + j >= data->size)
					continue;
				for(k = 0; k < networks[network_id]->output_dim; k++)
					py_cont_array[k] = *((float*)(py_target->data + i * (networks[network_id]->batch_size 
					* py_target->strides[0]) + j * py_target->strides[0] + k 
					* py_target->strides[1]));

				data->cont_copy(py_cont_array, data->target[i], j*networks[network_id]->output_dim, networks[network_id]->output_dim);
			}
		}
		free(py_cont_array);
	}
	
	if(silent == 0)
		printf("Done !\n");
	
	//printf("Time copy %f\n",ellapsed_time(time));
	
	init_timing(&time);
	#ifdef CUDA
	if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->cu_inst.dynamic_load == 0)
	{
		if(silent == 0)
			printf("Converting dataset to GPU device (CUDA)\n");
		//cuda_convert_dataset(networks[network_id], data);
		cuda_get_batched_dataset(networks[network_id], data);
	}
	//printf("Time convert data for CUDA %f\n", ellapsed_time(time));
	#endif
	if(silent == 0)
		printf("\n");
	
	Py_END_ALLOW_THREADS
	return Py_None;
}


static PyObject* py_delete_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	const char *dataset_type;
	int network_id = nb_networks-1, silent = 0;
	Dataset *data = NULL;
	static char *kwlist[] = {"dataset", "network", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ii", kwlist, &dataset_type, &network_id, &silent))
		return Py_None;
	
	Py_BEGIN_ALLOW_THREADS
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		if(silent == 0)
			printf("Deleting train set\n");
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
			printf("Deleting test set\n");
		data = &networks[network_id]->test;
	}
	else if(strcmp(dataset_type,"TRAIN_buf") == 0)
	{
		if(silent == 0)
			printf("Deleting train buffer set\n");
		data = &networks[network_id]->train_buf;
	}
	else if(strcmp(dataset_type,"VALID_buf") == 0)
	{
		if(silent == 0)
			printf("Deleting valid buffer set\n");
		data = &networks[network_id]->valid_buf;
	}
	else if(strcmp(dataset_type,"TEST_buf") == 0)
	{
		if(silent == 0)
			printf("Deleting test buffer set\n");
		data = &networks[network_id]->test_buf;
	}
	else
	{
		printf("Warning: No matching dataset to delete!\n");
	}

	if(data != NULL)
		free_dataset(data);
	
	Py_END_ALLOW_THREADS

	return Py_None;
}


static PyObject* py_swap_data_buffers(PyObject* self, PyObject* args)
{
	int network_id = nb_networks - 1;
	const char *dataset_type;
	Dataset temp;
	if(!PyArg_ParseTuple(args, "s|i", &dataset_type, &network_id))
		return Py_None;
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		temp = networks[network_id]->train;
		networks[network_id]->train = networks[network_id]->train_buf;
		networks[network_id]->train_buf = temp;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		temp = networks[network_id]->valid;
		networks[network_id]->valid = networks[network_id]->valid_buf;
		networks[network_id]->valid_buf = temp;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		temp = networks[network_id]->test;
		networks[network_id]->test = networks[network_id]->test_buf;
		networks[network_id]->test_buf = temp;
	}
	else
	{
		printf("Warning: No matching dataset to delete!\n");
	}
	
	return Py_None;
}

static PyObject* py_linear(PyObject* self, PyObject *args, PyObject *kwargs)
{
	char *string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	string += sprintf(string, "LIN");
	
	return Py_BuildValue("s", string);
}

static PyObject* py_relu(PyObject* self, PyObject *args, PyObject *kwargs)
{
	double saturation = 0.0/0.0, leaking = 0.0/0.0;
	static char *kwlist[] = {"saturation", "leaking", NULL};

	char *string = NULL, *c_string = NULL;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &saturation, &leaking))
		return Py_None;
	
	string = (char*) malloc(40*sizeof(char));
	c_string = string;
	c_string += sprintf(c_string, "RELU");
	
	if(!isnan(saturation))
		c_string += sprintf(c_string, "_S%0.2f", (float) saturation);
	if(!isnan(leaking))
		c_string += sprintf(c_string, "_L%0.2f", (float) leaking);
	
	return Py_BuildValue("s", string);
}


static PyObject* py_logistic(PyObject* self, PyObject *args, PyObject *kwargs)
{
	double saturation = 0.0/0.0, beta = 0.0/0.0;
	static char *kwlist[] = {"saturation", "beta", NULL};

	char *string = NULL, *c_string = NULL;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &saturation, &beta))
		return Py_None;
	
	string = (char*) malloc(40*sizeof(char));
	c_string = string;
	c_string += sprintf(c_string, "LOGI");
	
	if(!isnan(saturation))
		c_string += sprintf(c_string, "_S%0.2f", (float) saturation);
	if(!isnan(beta))
		c_string += sprintf(c_string, "_B%0.2f", (float) beta);
	
	return Py_BuildValue("s", string);
}

static PyObject* py_softmax(PyObject* self, PyObject *args, PyObject *kwargs)
{
	char *string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	string += sprintf(string, "SMAX");
	
	return Py_BuildValue("s", string);
}


static PyObject* py_yolo(PyObject* self, PyObject *args, PyObject *kwargs)
{
	char *string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	string += sprintf(string, "YOLO");
	
	return Py_BuildValue("s", string);
}

// Layers functions
//############################################################
static PyObject* py_dense(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int nb_neurons, prev_layer = -1, network_id = nb_networks-1, strict_size = 0, current_layer_id = -1;
	const char *activation = "RELU", *init_fct = "xavier";
	double drop_rate = 0.0, py_bias = 0.0/0.0, init_scaling=-1.0;
	float *c_bias = NULL;
	static char *kwlist[] = {"nb_neurons", "activation", "bias", 
		"prev_layer", "drop_rate", "strict_size", "init_fct", "init_scaling", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|sdidisdi", kwlist, &nb_neurons, &activation, &py_bias, 
		&prev_layer, &drop_rate, &strict_size, &init_fct, &init_scaling, &network_id))
		return PyLong_FromLong(-1);

	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
	
	//if(py_bias == py_bias) //portable but not compatible with fast math
	if(!isnan(py_bias)) //work with math.h only for C99 and later
	{
		c_bias = (float*) calloc(1,sizeof(float));
		*c_bias = (float) py_bias;
	}
	
	if((strcmp(init_fct,"normal") == 0 || strcmp(init_fct,"uniform") == 0) && init_scaling < 0.0)
	{
		printf("ERROR: init_scaling keywork is mandatory when using custom normal or uniform weight initialisation.");
		exit(EXIT_FAILURE);
	}
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	current_layer_id = dense_create(networks[network_id], prev, nb_neurons, activation, c_bias, drop_rate, strict_size, init_fct, init_scaling, NULL, 0);
	
	return PyLong_FromLong(current_layer_id);
}

static PyObject* py_conv(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int i;
	int nb_filters, prev_layer = -1, network_id = nb_networks-1, current_layer_id = -1;
	PyArrayObject *py_f_size = NULL, *py_stride = NULL, *py_padding = NULL, *py_int_padding = NULL, *py_input_shape = NULL;
	int C_f_size[3] = {1,1,1}, C_stride[3] = {1,1,1}, C_padding[3] = {0,0,0}, C_int_padding[3] = {0,0,0}, C_input_shape[4];
	const char *activation = "RELU", *init_fct = "xavier";
	double drop_rate = 0.0, py_bias = 0.0/0.0, init_scaling=-1.0;
	float *c_bias = NULL;
	
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "int_padding", "activation", "bias", 
		"prev_layer", "input_shape", "drop_rate", "init_fct", "init_scaling", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|OOOsdiOdsdi", kwlist, &py_f_size, &nb_filters, &py_stride, &py_padding, 
		&py_int_padding, &activation, &py_bias, &prev_layer, &py_input_shape, &drop_rate, &init_fct, &init_scaling, &network_id))
		return PyLong_FromLong(-1);
	
	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
		
	if(py_f_size != NULL) //this one should never be null anyway
		for(i = 0; i < py_f_size->dimensions[0] && i < 3; i++)
			C_f_size[i]  = *(int *)(py_f_size->data  + i*py_f_size->strides[0]);
	if(py_stride != NULL)
		for(i = 0; i < py_stride->dimensions[0] && i < 3; i++)
			C_stride[i]  = *(int *)(py_stride->data  + i*py_stride->strides[0]);
	if(py_padding != NULL)
		for(i = 0; i < py_padding->dimensions[0] && i < 3; i++)	
			C_padding[i] = *(int *)(py_padding->data + i*py_padding->strides[0]);
	if(py_int_padding != NULL)
		for(i = 0; i < py_int_padding->dimensions[0] && i < 3; i++)	
			C_int_padding[i] = *(int *)(py_int_padding->data + i*py_int_padding->strides[0]);
		
	if(py_input_shape != NULL)
		for(i = 0; i < 4; i++)
			C_input_shape[i] = *(int *)(py_input_shape->data + i*py_input_shape->strides[0]);
	
	//if(py_bias == py_bias) //portable but not compatible with fast math
	if(!isnan(py_bias)) //work with math.h only for C99 and later
	{
		c_bias = (float*) calloc(1,sizeof(float));
		*c_bias = (float) py_bias;
	}
	
	if((strcmp(init_fct,"normal") == 0 || strcmp(init_fct,"uniform") == 0) && init_scaling < 0.0)
	{
		printf("ERROR: init_scaling keywork is mandatory when using custom normal or uniform weight initialisation.");
		exit(EXIT_FAILURE);
	}
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
	current_layer_id = conv_create(networks[network_id], prev, C_f_size, nb_filters, C_stride, C_padding, 
		C_int_padding, C_input_shape, activation, c_bias, drop_rate, init_fct, init_scaling, NULL, 0);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_pool(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int i;
	int prev_layer = -1, network_id = nb_networks-1, global = 0, current_layer_id = -1;
	PyArrayObject *py_pool_size = NULL, *py_pool_stride = NULL, *py_pool_padding = NULL;
	const char *s_pool_type = "MAX";
	const char *activation = "LIN";
	int C_pool_size[3] = {1,1,1};
	int C_pool_stride[3] = {1,1,1};
	int C_pool_padding[3] = {0,0,0};
	double drop_rate = 0.0;
	static char *kwlist[] = {"p_size", "stride", "padding", "prev_layer", "drop_rate", "p_type", "activation", "p_global", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOidssii", kwlist, &py_pool_size, &py_pool_stride, &py_pool_padding, 
		&prev_layer, &drop_rate, &s_pool_type, &activation, &global, &network_id))
		return PyLong_FromLong(-1);
	
	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;

	// Set default to 2 for all dim > 1
	for(i = 0; i < 3; i++)
		if(networks[network_id]->in_dims[i] > 1)
			C_pool_size[i] = 2;
	
	if(py_pool_size != NULL)
		for(i = 0; i < py_pool_size->dimensions[0] && i < 3; i++)
			C_pool_size[i]  = *(int *)(py_pool_size->data  + i*py_pool_size->strides[0]);
	
	if(py_pool_stride != NULL)
		for(i = 0; i < py_pool_stride->dimensions[0] && i < 3; i++)
			C_pool_stride[i]  = *(int *)(py_pool_stride->data  + i*py_pool_stride->strides[0]);
	else
		for(i = 0; i < 3; i++)
			C_pool_stride[i] = C_pool_size[i];
	
	if(py_pool_padding != NULL)
		for(i = 0; i < py_pool_padding->dimensions[0] && i < 3; i++)
			C_pool_padding[i]  = *(int *)(py_pool_padding->data  + i*py_pool_padding->strides[0]);
			
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	current_layer_id = pool_create(networks[network_id], prev, C_pool_size, C_pool_stride, C_pool_padding, s_pool_type, activation, global, drop_rate);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_norm(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int prev_layer = -1, network_id = nb_networks-1, current_layer_id = -1;
	const char *norm_type = "GN";
	const char *activation = "LIN";
	int group_size = 8, set_off = 0;
	static char *kwlist[] = {"normalization", "activation", "prev_layer", "group_size", "set_off", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|ssiiii", kwlist, &norm_type, &activation, &prev_layer, &group_size, &set_off, &network_id))
		return PyLong_FromLong(-1);
	
	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	current_layer_id = norm_create(networks[network_id], prev, norm_type, activation, group_size, set_off, NULL, 0);
	
	return PyLong_FromLong(current_layer_id);
}

/*EXPERIMENTAL, NOT FULLY TESTED*/
static PyObject* py_lrn(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int prev_layer = -1, network_id = nb_networks-1, current_layer_id = -1;
	const char *activation = "LIN";
	int range = 5;
	double k = 1.0, alpha = 1.0, beta = 0.5;
	static char *kwlist[] = {"activation", "prev_layer", "range", "k", "alpha", "beta", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|siidddi", kwlist, &activation, &prev_layer, &range, &k, &alpha, &beta, &network_id))
		return PyLong_FromLong(-1);
	
	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	current_layer_id = lrn_create(networks[network_id], prev, activation, range, k, alpha, beta, NULL, 0);
	
	return PyLong_FromLong(current_layer_id);
}

/*EXPERIMENTAL, NOT FULLY TESTED*/
static PyObject* py_set_frozen_layers(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int i;
	int network_id = nb_networks-1;
	PyArrayObject *py_froz_array = NULL;
	int *c_froz_array = NULL;
	static char *kwlist[] = {"froz_array", "network",NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &py_froz_array, &network_id))
		return Py_None;
	
	if(py_froz_array != NULL && py_froz_array->dimensions[0] > 0)
	{
		c_froz_array = (int*) calloc(py_froz_array->dimensions[0],sizeof(int));
		for(i = 0; i < py_froz_array->dimensions[0]; i++)
			c_froz_array[i] = *(int *)(py_froz_array->data  + i*py_froz_array->strides[0]);
	
		set_frozen_layers(networks[network_id], c_froz_array, py_froz_array->dimensions[0]);
	}
	
	return Py_None;
}


static PyObject* py_set_IoU_limits(PyObject* self, PyObject *args, PyObject *kwargs)
{
	float *IoU_limits_array = NULL;
	PyObject *py_IoU_limits_array = NULL;
	float good_IoU_lim = -2.0f, low_IoU_best_box_assoc = -2.0f, min_prob_IoU_lim = -2.0f, min_obj_IoU_lim = -2.0f;
	float min_class_IoU_lim = -2.0f, min_param_IoU_lim = -2.0f, diff_IoU_lim = -2.0f, diff_obj_lim = -2.0f;
	npy_intp dims[1];
	static char *kwlist[] = {"good_IoU_lim", "low_IoU_best_box_assoc", "min_prob_IoU_lim", "min_obj_IoU_lim",
		"min_class_IoU_lim", "min_param_IoU_lim", "diff_IoU_lim", "diff_obj_lim", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|ffffffff", kwlist, 
		&good_IoU_lim, &low_IoU_best_box_assoc, &min_prob_IoU_lim, &min_obj_IoU_lim,
		&min_class_IoU_lim, &min_param_IoU_lim, &diff_IoU_lim, &diff_obj_lim))
		return Py_None;
	
	IoU_limits_array = (float*) calloc(8, sizeof(float));
	
	IoU_limits_array[0] = good_IoU_lim;
	IoU_limits_array[1] = low_IoU_best_box_assoc;
	IoU_limits_array[2] = min_prob_IoU_lim;
	IoU_limits_array[3] = min_obj_IoU_lim;
	IoU_limits_array[4] = min_class_IoU_lim;
	IoU_limits_array[5] = min_param_IoU_lim;
	IoU_limits_array[6] = diff_IoU_lim;
	IoU_limits_array[7] = diff_obj_lim;
	
	dims[0] = 8;
	py_IoU_limits_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)IoU_limits_array);
	
	return py_IoU_limits_array;
}


static PyObject* py_set_fit_parts(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int *fit_parts_array = NULL;
	PyObject *py_fit_parts_array = NULL;
	int position = -2, size = -2, probability = -2, objectness = -2, classes = -2, parameters = -2;
	npy_intp dims[1];
	static char *kwlist[] = {"position", "size", "probability", "objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiii", kwlist, 
		&position, &size, &probability, &objectness, &classes, &parameters))
		return Py_None;
	
	fit_parts_array = (int*) calloc(6, sizeof(int));
	
	fit_parts_array[0] = position;
	fit_parts_array[1] = size;
	fit_parts_array[2] = probability;
	fit_parts_array[3] = objectness;
	fit_parts_array[4] = classes;
	fit_parts_array[5] = parameters;

	
	dims[0] = 6;
	py_fit_parts_array = PyArray_SimpleNewFromData(1, dims, NPY_INT, (void*)fit_parts_array);
	
	return py_fit_parts_array;
}


static PyObject* py_set_error_scales(PyObject* self, PyObject *args, PyObject *kwargs)
{
	float *error_scales_array = NULL;
	PyObject *py_error_scales_array = NULL;
	float position = -1.0f, size = -1.0f, probability = -1.0f, objectness = -1.0f, classes = -1.0f, parameters = -1.0f;
	npy_intp dims[1];
	static char *kwlist[] = {"position", "size", "probability", "objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|ffffff", kwlist, 
		&position, &size, &probability, &objectness, &classes, &parameters))
		return Py_None;
	
	error_scales_array = (float*) calloc(6, sizeof(float));
	
	error_scales_array[0] = position;
	error_scales_array[1] = size;
	error_scales_array[2] = probability;
	error_scales_array[3] = objectness;
	error_scales_array[4] = classes;
	error_scales_array[5] = parameters;

	
	dims[0] = 6;
	py_error_scales_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)error_scales_array);
	
	return py_error_scales_array;
}


static PyObject* py_set_sm_single(PyObject* self, PyObject *args, PyObject *kwargs)
{
	float *sm_single_array = NULL;
	PyObject *py_sm_single_array = NULL;
	float slope = -1.0f, max = 100000.0f, min = -100000.0f;
	npy_intp dims[1];
	static char *kwlist[] = {"slope", "fmin", "fmax", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|fff", kwlist, &slope, &min, &max))
		return Py_None;
	
	sm_single_array = (float*) calloc(3, sizeof(float));
	
	sm_single_array[0] = slope;
	sm_single_array[1] = max;
	sm_single_array[2] = min;
	
	dims[0] = 3;
	py_sm_single_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)sm_single_array);
	
	return py_sm_single_array;
}


static PyObject* py_set_slopes_and_maxes(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i, j;
	float *slopes_and_maxes = NULL;
	PyObject *py_slopes_and_maxes = NULL;
	PyArrayObject *py_position = NULL, *py_size = NULL, *py_probability = NULL;
	PyArrayObject *py_objectness = NULL, *py_classes = NULL, *py_parameters = NULL;
	PyArrayObject *local[6];
	npy_intp dims[2];
	static char *kwlist[] = {"position", "size", "probability","objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOOO", kwlist, 
		&py_position, &py_size, &py_probability, &py_objectness, &py_classes, &py_parameters))
		return Py_None;
	
	slopes_and_maxes = (float*) calloc(6*3, sizeof(float));
	
	local[0] = py_position; local[1] = py_size; local[2] = py_probability;
	local[3] = py_objectness; local[4] = py_classes; local[5] = py_parameters;
	
	for(i = 0; i < 6; i++)
	{
		if(local[i] != NULL)
		{
			for(j = 0; j < 3; j++)
				slopes_and_maxes[i*3+j] = *((float*)(local[i]->data + j * local[i]->strides[0]));
		}
		else
		{
			slopes_and_maxes[i*3] = -1.0f; 
			slopes_and_maxes[i*3+1] = 100000.0f; 
			slopes_and_maxes[i*3+2] = -100000.0f;
		}
	}
		
	dims[0] = 6; dims[1] = 3;
	py_slopes_and_maxes = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, (void*)slopes_and_maxes);
	
	return py_slopes_and_maxes;
}


static PyObject* py_set_yolo_params(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i,j;
	int nb_box = 0, nb_class = 0, nb_param = 0, max_nb_obj_per_image = 0;
	int strict_box_size_association = 0, fit_dim = 0, rand_startup = -1, network_id = 0;
	int class_softmax = 0, diff_flag = 0, no_override = 0, raw_output = 0;
	double rand_prob_best_box_assoc = 0.0f, rand_prob = 0.0f, min_prior_forced_scaling = 0.0f;
	PyArrayObject *py_prior_size = NULL, *py_prior_noobj_prob = NULL;
	float *C_prior_size = NULL, *C_prior_noobj_prob = NULL;
	PyArrayObject *py_error_scales = NULL, *py_slopes_and_maxes = NULL, *py_param_ind_scales = NULL, *py_IoU_limits = NULL, *py_fit_parts = NULL;
	const char *IoU_type_char = "empty", *prior_dist_type_char = "empty", *error_type = "empty";
	float *error_scales = NULL, **slopes_and_maxes = NULL, *param_ind_scales = NULL, *IoU_limits = NULL;
	int *fit_parts = NULL;
	float* temp;
	static char *kwlist[] = {"nb_box", "nb_class", "nb_param", "max_nb_obj_per_image", "prior_size", "prior_noobj_prob", "error_scales", "slopes_and_maxes", "param_ind_scales", "IoU_limits",
		"fit_parts", "IoU_type", "prior_dist_type", "strict_box_size", "fit_dim", "rand_startup", "rand_prob_best_box_assoc", "rand_prob", 
		"min_prior_forced_scaling", "class_softmax", "diff_flag", "network", "error_type", "no_override", "raw_output", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiOOOOOOOssiiidddiiisii", kwlist, 
			&nb_box, &nb_class, &nb_param, &max_nb_obj_per_image, &py_prior_size, &py_prior_noobj_prob, 
			&py_error_scales, &py_slopes_and_maxes, &py_param_ind_scales, &py_IoU_limits, &py_fit_parts, 
			&IoU_type_char, &prior_dist_type_char, &strict_box_size_association, &fit_dim, &rand_startup, &rand_prob_best_box_assoc, &rand_prob,
			&min_prior_forced_scaling, &class_softmax, &diff_flag, &network_id, &error_type, &no_override, &raw_output))
		return PyLong_FromLong(0);
	
	if(py_prior_size != NULL && fit_dim <= 0)
		fit_dim = py_prior_size->dimensions[0];
	else
	{
		if(py_prior_size != NULL && fit_dim > py_prior_size->dimensions[0])
		{
			printf("ERROR: fit_dim parameter cannot be larger than the number of dimensions of prior_size!\n");
			exit(EXIT_FAILURE);
		}
	}
	
	if(py_prior_size != NULL && py_prior_size->dimensions[1] != nb_box)
	{
		printf("ERROR: The prior_size array must have nb_box elements!");
		exit(EXIT_FAILURE);
	}
	
	if(py_prior_size != NULL)
	{
		C_prior_size = (float*) calloc(3*nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
		{
			for(j = 0; j < 3; j++)
			{
				if(j < fit_dim)
					C_prior_size[i*3+j] = *((float*)(py_prior_size->data 
											   + j * py_prior_size->strides[0] 
											   + i * py_prior_size->strides[1]));
				else
					C_prior_size[i*3+j] = 0.0f;
			}
		}
	}
	
	if(py_prior_noobj_prob != NULL)
	{
		C_prior_noobj_prob = (float*) calloc(nb_box,sizeof(float));
		if(py_prior_noobj_prob->dimensions[0] != nb_box)
		{
			printf("ERROR: The prior_noobj_prob array must have nb_box elements!");
			exit(EXIT_FAILURE);
		}
		for(i = 0; i < nb_box; i++)
			C_prior_noobj_prob[i] = *((float *)(py_prior_noobj_prob->data 
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
				slopes_and_maxes[i][j] = *(float *)(py_slopes_and_maxes->data 
					+ i*py_slopes_and_maxes->strides[0] + j*py_slopes_and_maxes->strides[1]);
	}
	
	if(py_param_ind_scales != NULL)
	{
		param_ind_scales = (float*) calloc(nb_param, sizeof(float));
		if(py_param_ind_scales->dimensions[0] != nb_param)
		{
			printf("ERROR: The param_ind_scales array must have nb_param elements!");
			exit(EXIT_FAILURE);
		}
		for(i = 0; i < nb_param; i++)
			param_ind_scales[i] = *(float *)(py_param_ind_scales->data + i*py_param_ind_scales->strides[0]);
	}
	
	if(py_IoU_limits != NULL)
	{
		IoU_limits = (float*) calloc(8, sizeof(float));
		for(i = 0; i < 8; i++)
			IoU_limits[i] = *(float *)(py_IoU_limits->data + i*py_IoU_limits->strides[0]);
	}
	
	if(py_fit_parts != NULL)
	{
		fit_parts = (int*) calloc(6, sizeof(int));
		for(i = 0; i < 6; i++)
			fit_parts[i] = *(int *)(py_fit_parts->data + i*py_fit_parts->strides[0]);
	}
	
	return PyLong_FromLong(set_yolo_params(networks[network_id], nb_box, nb_class, nb_param, max_nb_obj_per_image, 
		IoU_type_char, prior_dist_type_char, C_prior_size, C_prior_noobj_prob, fit_dim, strict_box_size_association, 
		rand_startup, rand_prob_best_box_assoc, rand_prob, min_prior_forced_scaling, error_scales, slopes_and_maxes, 
		param_ind_scales, IoU_limits, fit_parts, class_softmax, diff_flag, error_type, no_override, raw_output));
}

static PyObject* perf_eval(PyObject* self, PyObject* args)
{
	int network_id = nb_networks - 1;
	if(!PyArg_ParseTuple(args, "|i", &network_id))
		return Py_None;

	perf_eval_display(networks[network_id]);
	
	return Py_None;
}


static PyObject* py_load_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	const char *file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int iter, network_id = nb_networks-1, nb_layers = 0, f_bin = 0;
	static char *kwlist[] = {"file", "iteration", "network", "nb_layers", "bin",NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "si|iii", kwlist, &file, &iter, &network_id, &nb_layers, &f_bin))
		return Py_None;
		
	load_network(networks[network_id], file, iter, nb_layers, f_bin);
	
	return Py_None;
}

static PyObject* py_save_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	const char *file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int network_id = nb_networks-1, f_bin = 0;
	static char *kwlist[] = {"file", "network", "bin", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ii", kwlist, &file, &network_id, &f_bin))
		return Py_None;
		
	save_network(networks[network_id], file, f_bin);
	
	return Py_None;
}


// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	
	int py_nb_iter, py_control_interv = 1, py_confmat = 0, save_every = 0, network_id = nb_networks-1;
	int shuffle_gpu = 1, shuffle_every = 1, silent = 0, save_bin = 0;
	double py_learning_rate=0.0, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0, py_TC_scale_factor = 1.0, py_weight_decay = 0.0;
	static char *kwlist[] = {"nb_iter", "learning_rate", "end_learning_rate", "control_interv", "momentum", "lr_decay", 
		"weight_decay", "confmat", "save_every", "save_bin", "network", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|didddiiiiiidi", kwlist, &py_nb_iter, &py_learning_rate, &py_end_learning_rate, 
		&py_control_interv, &py_momentum, &py_decay, &py_weight_decay, &py_confmat, &save_every, &save_bin, &network_id, 
		&shuffle_gpu, &shuffle_every, &py_TC_scale_factor, &silent))
		return Py_None;
		
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS
	
	train_network(networks[network_id], py_nb_iter, py_control_interv, py_learning_rate, py_end_learning_rate, 
		py_momentum, py_decay, py_weight_decay, py_confmat, save_every, save_bin, shuffle_gpu, shuffle_every, py_TC_scale_factor, silent);
		
	Py_END_ALLOW_THREADS

	return Py_None;
}


static PyObject* py_forward_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int repeat = 1, network_id = nb_networks-1, C_drop_mode = AVG_MODEL, no_error = 0, saving = 1, silent = 0;
	const char *drop_mode = "AVG_MODEL";
	static char *kwlist[] = {"saving", "drop_mode", "no_error", "repeat", "network", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|isiiii", kwlist, &saving, &drop_mode, &no_error, &repeat, &network_id, &silent))
		return Py_None;
	
	if(strcmp(drop_mode, "AVG_MODEL") == 0)
	{
		C_drop_mode = AVG_MODEL;
	}
	else if(strcmp(drop_mode, "MC_MODEL") == 0)
	{
		C_drop_mode = MC_MODEL;
	}

	networks[network_id]->no_error = no_error;
	
	Py_BEGIN_ALLOW_THREADS
	
	forward_testset(networks[network_id], saving, repeat, C_drop_mode, silent);
	
	Py_END_ALLOW_THREADS
	
	return Py_None;
}

#ifdef CUDA
// Experimental function for now, only for development purposes
static PyObject* py_gan_train(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int py_nb_iter, py_control_interv = 1, py_confmat = 0, save_every = 0, gen_id = nb_networks-2, disc_id = nb_networks-1;
	int shuffle_gpu = 1, shuffle_every = 1, silent = 0, disc_only = 0;
	double py_learning_rate=0.02, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0;
	double py_TC_scale_factor = 1.0, py_gen_disc_lr_ratio = 2.0;
	static char *kwlist[] = {"gen_id", "disc_id", "nb_iter", "learning_rate", "end_learning_rate", "gen_disc_lr_ratio", "control_interv",
		 "momentum", "decay", "confmat", "save_every", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "disc_only", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "iiid|ddiddiiiidiu", kwlist, &gen_id, &disc_id, &py_nb_iter, &py_learning_rate, 
		&py_end_learning_rate, &py_gen_disc_lr_ratio, &py_control_interv, &py_momentum, &py_decay, &py_confmat, &save_every, 
		&shuffle_gpu, &shuffle_every, &py_TC_scale_factor, &disc_only, &silent))
		return Py_None;
	
	if(silent == 0)
		printf("py_nb_iter %d, py_control_interv %d, py_learning_rate %g, py_end_learning_rate %g , py_momentum %0.2f, \
			py_decay %g, py_confmat %d, save_every %d, shuffle_gpu %d , shuffle_every %d, TC_scale_factor %g\n", 
			py_nb_iter, py_control_interv, py_learning_rate, py_end_learning_rate, py_momentum, 
			py_decay, py_confmat, save_every, shuffle_gpu, shuffle_every, py_TC_scale_factor);
	
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS
	
	train_gan(networks[gen_id], networks[disc_id], py_nb_iter, py_control_interv, py_learning_rate, py_end_learning_rate, 
		py_momentum, py_decay, py_gen_disc_lr_ratio, save_every, 0, shuffle_gpu, shuffle_every, disc_only, py_TC_scale_factor, silent);
	
	Py_END_ALLOW_THREADS

	return Py_None;
}
#endif

static PyObject* py_print_architecture_tex(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int network_id = nb_networks-1, l_size = 1, l_in_size = 1, l_f_size = 1, l_out_size = 1, 
		l_stride = 1, l_padding = 1, l_in_padding = 0, l_activation = 0, l_bias = 0, l_dropout = 0, l_param_count = 0;
	const char *path = "relative_path_to_the_save_file_location_which_must_be_long_enough", *file_name = "A_long_filename_template_for_user_setup";
	static char *kwlist[] = {"path", "file_name", "size", "in_size", "f_size", "out_size", 
		"stride", "padding", "in_padding", "activation", "bias", "dropout", "param_count", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|iiiiiiiiiiii", kwlist, &path, &file_name, &l_size, &l_in_size, 
		&l_f_size, &l_out_size, &l_stride, &l_padding, &l_in_padding, &l_activation, &l_bias, &l_dropout, &l_param_count, &network_id))
		return Py_None;
	
	print_architecture_tex(networks[network_id], path, file_name,
		l_size, l_in_size, l_f_size, l_out_size, l_stride, l_padding, 
		l_in_padding, l_activation, l_bias, l_dropout, l_param_count);
	
	return Py_None;
}



// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
	{ "init", (PyCFunction)py_init_network, METH_VARARGS | METH_KEYWORDS, "Initialize network basic shapes and properties" },
	{ "create_dataset", (PyCFunction)py_create_dataset, METH_VARARGS | METH_KEYWORDS, "Allocate dataset structure" },
	{ "delete_dataset", (PyCFunction)py_delete_dataset, METH_VARARGS | METH_KEYWORDS, "Free dataset structure" },
	{ "swap_data_buffers", py_swap_data_buffers, METH_VARARGS, "Put the selected buffered dataset as current dataset for training"},
	{ "linear", (PyCFunction)py_linear, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Linear"},
	{ "relu", (PyCFunction)py_relu, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to ReLU"},
	{ "logistic", (PyCFunction)py_logistic, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Logistic"},
	{ "softmax", (PyCFunction)py_softmax, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Softmax"},
	{ "yolo", (PyCFunction)py_yolo, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to YOLO"},
	{ "dense", (PyCFunction)py_dense, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
	{ "conv",(PyCFunction)py_conv, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
	{ "pool",(PyCFunction)py_pool, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
	{ "norm", (PyCFunction)py_norm, METH_VARARGS | METH_KEYWORDS, "Add a normalization layer to the network"},
	{ "lrn", (PyCFunction)py_lrn, METH_VARARGS | METH_KEYWORDS, "Add a Local Response Normalization layer to the network"},
	{ "set_frozen_layers",(PyCFunction)py_set_frozen_layers, METH_VARARGS | METH_KEYWORDS, "Freeze the selected layers' weights for training" },
	{ "set_IoU_limits",(PyCFunction)py_set_IoU_limits, METH_VARARGS | METH_KEYWORDS, "Create an array from a list of IoU limits" },
	{ "set_fit_parts",(PyCFunction)py_set_fit_parts, METH_VARARGS | METH_KEYWORDS, "Create an array from a list of parts of the YOLO loss function" },
	{ "set_error_scales",(PyCFunction)py_set_error_scales, METH_VARARGS | METH_KEYWORDS, "Create an array from a list of error scaling" },
	{ "set_sm_single",(PyCFunction)py_set_sm_single, METH_VARARGS | METH_KEYWORDS, "Create an array from slope, min and max parameters for various activations" },
	{ "set_slopes_and_maxes",(PyCFunction)py_set_slopes_and_maxes, METH_VARARGS | METH_KEYWORDS, "Adjunct or create arrays representing all slopes and maxes" },
	{ "set_yolo_params",(PyCFunction)py_set_yolo_params, METH_VARARGS | METH_KEYWORDS, "Set parameters for YOLO output layout" },
	{ "perf_eval", perf_eval, METH_VARARGS, "Display each layer time in ms and in percent of the total networj time" },
	{ "load", (PyCFunction)py_load_network, METH_VARARGS | METH_KEYWORDS, "Load a previous network structure pre-trained from a file" },
	{ "save", (PyCFunction)py_save_network, METH_VARARGS | METH_KEYWORDS, "Save a currently loaded network structure to a file" },
	{ "train", (PyCFunction)py_train_network, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
	{ "forward", (PyCFunction)py_forward_network, METH_VARARGS | METH_KEYWORDS, "Apply the trained network to the test set and save results" },
	{ "print_arch_tex", (PyCFunction)py_print_architecture_tex, METH_VARARGS | METH_KEYWORDS, "Print the architecture in a table using .tex and export to .pdf" },
	#ifdef CUDA
	{ "gan_train", (PyCFunction)py_gan_train, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
	#endif
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef CIANNA = {
	PyModuleDef_HEAD_INIT,
	"CIANNA",
	"Convolutional Interactive Artificial Neural Networks by/for Astrophysicists",
	-1,
	CIANNAMethods
};


PyMODINIT_FUNC PyInit_CIANNA(void)
{
	setbuf(stdout, NULL);
	import_array();
	
	printf("###################################################################\n\
Importing CIANNA Python module V-p.0.6.6 , by D.Cornu\n\
###################################################################\n\n");

	PyObject *m;
	m = PyModule_Create(&CIANNA);
	if (m == NULL)
		return NULL;
	
	return m;
}




















