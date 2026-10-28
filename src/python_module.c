
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

#define NPY_NO_DEPRECATED_API 0

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "prototypes.h"


// Helper functions
//############################################################

static void parse_int_sequence(PyObject *obj, int *output, int min_size, int max_size, const char *arg_name)
{
	PyArrayObject *array;
	Py_ssize_t i, size;
	int *data;

	if(!PyList_Check(obj) && !PyTuple_Check(obj) && !PyArray_Check(obj))
	{
		printf("\n ERROR: %s argument must be a list, tuple, or Numpy array.\n", arg_name);
		exit(EXIT_FAILURE);
	}
	
	array = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
	
	if(array == NULL)
	{
		printf("\n ERROR: %s argument could not be converted to integers.\n", arg_name);
		exit(EXIT_FAILURE);
	}
	
	if(PyArray_NDIM(array) != 1)
	{
		printf("\n ERROR: %s argument must be one-dimensional.\n", arg_name);
		exit(EXIT_FAILURE);
	}
	
	size = PyArray_SIZE(array);
	if(size < min_size || size > max_size)
	{
		printf("\n ERROR: %s argument size should be between %d and %d.\n", arg_name, min_size, max_size);
		exit(EXIT_FAILURE);
	}

	data = (int *)PyArray_DATA(array);
	for(i = 0; i < size; i++)
		output[i] = data[i];
	
	Py_DECREF(array);
}


static PyArrayObject* get_array_from_float_sequence(PyObject *obj, const char *arg_name)
{
	PyArrayObject *array;

	if(obj == NULL)
		return NULL;

	if(!PyList_Check(obj) && !PyTuple_Check(obj) && !PyArray_Check(obj))
	{
		printf("\n ERROR: %s argument must be a list, tuple, or Numpy array.\n", arg_name);
		exit(EXIT_FAILURE);
	}
	
	array = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
	
	if(array == NULL)
	{
		printf("\n ERROR: %s argument could not be converted to floats.\n", arg_name);
		exit(EXIT_FAILURE);
	}
	
	return array;
}



// Network paramaremeter and data management functions
//############################################################

static PyObject* py_init_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int b_size = 8, network_id = 0;
	int dynamic_load = 1, wema = 0, no_logo = 0, silent = 0, adv_size = 0, inference_only = 0;
	const char *optimizer = "SGD";
	const char *mixed_precision = "off";
	const char *comp_meth = "C_CUDA";
	static char *kwlist[] = {"b_size", "optimizer", "wema", "comp_meth", "network", 
		"dynamic_load", "mixed_precision", "inference_only", "no_logo", "silent", "adv_size", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|isisiisiiii", kwlist, &b_size, 
		&optimizer, &wema, &comp_meth, &network_id, &dynamic_load, &mixed_precision, &inference_only, &no_logo, &silent, &adv_size))
		return NULL;
	
	init_network(network_id, b_size, optimizer, wema, comp_meth, dynamic_load, mixed_precision, inference_only, no_logo, silent, adv_size);
	
	Py_RETURN_NONE;
}

static PyObject* py_free_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int network_id = -1;
	static char *kwlist[] = {"network", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	free_network(networks[network_id]);
	
	Py_RETURN_NONE;
}


static PyObject* py_create_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int i, j;
	size_t b, k;
	Dataset *data = NULL;
	const char *dataset_type;
	float *py_cont_array;
	PyObject *py_input = NULL, *py_target = NULL;
	PyObject *seq_input = NULL, *seq_target = NULL;
	PyArrayObject **branch_input = NULL, **branch_target = NULL;
	float *branch_data = NULL;
	int size, silent = 0, with_target = 0;
	size_t nb_branch_in, nb_branch_targ, *branch_in_dim = NULL, *branch_targ_dim = NULL;
	int network_id = -1;
	static char *kwlist[] = {"dataset", "size", "input", "target", "network", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "siO|Oii", kwlist, &dataset_type, &size, &py_input, &py_target, &network_id, &silent))
		return NULL;

	if(network_id == -1)
		network_id = global_current_network_id;

	if(!PyList_Check(py_input) && !PyTuple_Check(py_input) && !PyArray_Check(py_input))
	{
		printf("\n ERROR: input argument must be a list, tuple, or Numpy array of data arrays.\n");
		exit(EXIT_FAILURE);
	}

	if(PyArray_Check(py_input) && PyArray_TYPE((PyArrayObject*)py_input) != NPY_OBJECT)
		seq_input = PyTuple_Pack(1, py_input);
	else
		seq_input = PySequence_Fast(py_input,"\n ERROR: input argument must be a sequence of Numpy arrays. \n");
	
	nb_branch_in = (size_t)PySequence_Fast_GET_SIZE(seq_input);
	branch_input = (PyArrayObject**) malloc(nb_branch_in * sizeof(PyArrayObject*));
	branch_in_dim = (size_t*) malloc(nb_branch_in*sizeof(size_t));

	for(b = 0; b < nb_branch_in; b++)
	{
		PyObject *branch_obj = PySequence_Fast_GET_ITEM(seq_input, (Py_ssize_t)b);
		
		branch_input[b] = (PyArrayObject *)PyArray_FROM_OTF(branch_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
		if(PyArray_NDIM(branch_input[b]) != 2)
		{
			printf("\n ERROR: input branch %ld must be two-dimensional.\n", b);
			exit(EXIT_FAILURE);
		}
		branch_in_dim[b] = (size_t)PyArray_DIM(branch_input[b], 1);
	}
	
	if(py_target != NULL && py_target != Py_None)
	{
		if(!PyList_Check(py_target) && !PyTuple_Check(py_target) && !PyArray_Check(py_target))
		{
			printf("\n ERROR: target argument must be a list, tuple, or Numpy array of data arrays.\n");
			exit(EXIT_FAILURE);
		}
		
		if(PyArray_Check(py_target) && PyArray_TYPE((PyArrayObject*)py_target) != NPY_OBJECT)
			seq_target = PyTuple_Pack(1, py_target);
		else
			seq_target = PySequence_Fast(py_target,"\n ERROR: target argument must be a sequence of Numpy arrays. \n");
		
		nb_branch_targ = (size_t)PySequence_Fast_GET_SIZE(seq_target);
		branch_target = (PyArrayObject**) malloc(nb_branch_targ * sizeof(PyArrayObject*));
		branch_targ_dim = (size_t*) malloc(nb_branch_targ*sizeof(size_t));
		
		for(b = 0; b < nb_branch_targ; b++)
		{
			PyObject *branch_obj = PySequence_Fast_GET_ITEM(seq_target, (Py_ssize_t)b);
			
			branch_target[b] = (PyArrayObject *)PyArray_FROM_OTF(branch_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
			if(PyArray_NDIM(branch_target[b]) != 2)
			{
				printf("\n ERROR: target branch %ld must be two-dimensional.\n", b);
				exit(EXIT_FAILURE);
			}
			branch_targ_dim[b] = (size_t)PyArray_DIM(branch_target[b], 1);
		}
		with_target = 1;
	}
	
	Py_BEGIN_ALLOW_THREADS;
	/* This function is designed to allow parallel thread execution with the train function of CIANNA
	   Still, this function is not considered thread safe. Arrays provided as argument are expected to remain untouched
	   for the duration of the function! */
	
	data = get_dataset_from_type(networks[network_id], dataset_type, silent);
	*data = create_dataset(networks[network_id], with_target, size, nb_branch_in, branch_in_dim, nb_branch_targ, branch_targ_dim);
	
	for(b = 0; b < nb_branch_in; b++)
	{
		branch_data = (float*)PyArray_DATA(branch_input[b]);
		py_cont_array = (float*) calloc(branch_in_dim[b], sizeof(float));
		
		for(i = 0; i < data->nb_batch; i++)
		{
			for(j = 0; j < networks[network_id]->batch_size; j++)
			{
				k = i*networks[network_id]->batch_size + j;
				if(k >= data->size)
					continue;
				
				memcpy(py_cont_array, branch_data + k * branch_in_dim[b], branch_in_dim[b] * sizeof(float));
				data->cont_copy(py_cont_array, data->input[b][i], j*(branch_in_dim[b] + 1), branch_in_dim[b]);
			}
		}
		free(py_cont_array);
	}
	
	if(with_target)
	{
		for(b = 0; b < nb_branch_targ; b++)
		{
			branch_data = (float*)PyArray_DATA(branch_target[b]);
			py_cont_array = (float*) calloc(branch_targ_dim[b], sizeof(float));
			
			for(i = 0; i < data->nb_batch; i++)
			{
				for(j = 0; j < networks[network_id]->batch_size; j++)
				{
					k = i*networks[network_id]->batch_size + j;
					if(k >= data->size)
						continue;
					
					memcpy(py_cont_array, branch_data + k * branch_targ_dim[b], branch_targ_dim[b] * sizeof(float));
					data->cont_copy(py_cont_array, data->target[b][i], j*branch_targ_dim[b], branch_targ_dim[b]);
				}
			}
			free(py_cont_array);
		}
	}
	
	#ifdef CUDA
	if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->cu_inst.dynamic_load == 0)
		cuda_get_batched_dataset(networks[network_id], data);
	#endif
	
	Py_END_ALLOW_THREADS;
	if(silent == 0)
		printf("\n");
	
	free(branch_in_dim);
	for(b = 0; b < nb_branch_in; b++)
		Py_DECREF(branch_input[b]);

	free(branch_input);
	Py_DECREF(seq_input);

	if(with_target)
	{
		free(branch_targ_dim);
		for(b = 0; b < nb_branch_targ; b++)
			Py_DECREF(branch_target[b]);

		free(branch_target);
		Py_DECREF(seq_target);
	}

	Py_RETURN_NONE;
}


static PyObject* py_delete_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	const char *dataset_type;
	int network_id = -1, silent = 0;
	Dataset *data = NULL;
	static char *kwlist[] = {"dataset", "network", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ii", kwlist, &dataset_type, &network_id, &silent))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	Py_BEGIN_ALLOW_THREADS;
	
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
	
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


static PyObject* py_swap_data_buffers(PyObject* self, PyObject* args)
{
	setlocale(LC_ALL, "C");
	int network_id = -1;
	const char *dataset_type;
	Dataset temp;
	if(!PyArg_ParseTuple(args, "s|i", &dataset_type, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
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
	
	Py_RETURN_NONE;
}

// Optimizer helper functions
//############################################################

static PyObject* py_set_optimizer_param(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	const char *string;
	int network_id = -1;
	
	static char *kwlist[] = {"param", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", kwlist, &string, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	set_optimizer_param_from_string(networks[network_id], string);
	
	Py_RETURN_NONE;
}


static PyObject* py_sgd(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	char *string = NULL, *c_string = NULL;
	double momentum = 0.0f;
	
	static char *kwlist[] = {"momentum", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|d", kwlist, &momentum))
		return Py_None;
	
	string = (char*) malloc(100*sizeof(char));
	c_string = string;
	c_string += sprintf(c_string, "SGD");
	
	c_string += sprintf(c_string, "_mom%f", momentum);
	
	return Py_BuildValue("s", string);
}


static PyObject* py_adam(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	char *string = NULL, *c_string = NULL;
	double beta_1 = 0.9f, beta_2 = 0.999f, eps = 0.00000001f;
	int ams_grad = 0;
	
	static char *kwlist[] = {"beta1", "beta2", "epsilon", "ams_grad", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dddi", kwlist, &beta_1, &beta_2, &eps, &ams_grad))
		return NULL;
	
	string = (char*) malloc(100*sizeof(char));
	c_string = string;
	c_string += sprintf(c_string, "ADAM");
	
	c_string += sprintf(c_string, "_b1%f", beta_1);
	c_string += sprintf(c_string, "_b2%f", beta_2);
	c_string += sprintf(c_string, "_eps%g", eps);
	c_string += sprintf(c_string, "_ams%d", ams_grad);
	
	return Py_BuildValue("s", string);
}


static PyObject* py_rms_prop(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	char *string = NULL, *c_string = NULL;
	double alpha = 0.99f, momentum = 0.0f, eps = 0.00000001f;
	int centered = 0;
	
	static char *kwlist[] = {"alpha", "momentum", "epsilon", "centered", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dddi", kwlist, &alpha, &momentum, &eps, &centered))
		return NULL;
	
	string = (char*) malloc(100*sizeof(char));
	c_string = string;
	c_string += sprintf(c_string, "RMSprop");
	
	c_string += sprintf(c_string, "_a%f", alpha);
	c_string += sprintf(c_string, "_mom%f", momentum);
	c_string += sprintf(c_string, "_eps%g", eps);
	c_string += sprintf(c_string, "_cent%d", centered);
	
	return Py_BuildValue("s", string);
}


// Activation helper functions
//############################################################

static PyObject* py_linear(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	char *string = NULL, *c_string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	c_string = string;
	c_string += sprintf(string, "LIN");
	
	return Py_BuildValue("s", string);
}

static PyObject* py_relu(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	double saturation = 0.0/0.0, leaking = 0.0/0.0;
	static char *kwlist[] = {"saturation", "leaking", NULL};

	char *string = NULL, *c_string = NULL;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &saturation, &leaking))
		return NULL;
	
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
	setlocale(LC_ALL, "C");
	double saturation = 0.0/0.0, beta = 0.0/0.0;
	static char *kwlist[] = {"saturation", "beta", NULL};

	char *string = NULL, *c_string = NULL;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist, &saturation, &beta))
		return NULL;
	
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
	setlocale(LC_ALL, "C");
	char *string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	string += sprintf(string, "SMAX");
	
	return Py_BuildValue("s", string);
}


static PyObject* py_yolo(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	char *string = NULL;
	
	string = (char*) malloc(40*sizeof(char));
	string += sprintf(string, "YOLO");
	
	return Py_BuildValue("s", string);
}

// Layers functions
//############################################################
static PyObject* py_input(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int nb_features = 0, network_id = -1, current_layer_id = -1, silent = 0;
	int spatial_shape[3] = {1,1,1};
	PyObject *py_dims = NULL;
	static char *kwlist[] = {"nb_features", "spatial_shape", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|Oii", kwlist, &nb_features, &py_dims, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_dims != NULL && py_dims != Py_None)
		parse_int_sequence(py_dims, spatial_shape, 1, 3, "spatial_shape");
	
	current_layer_id = input_create(networks[network_id], spatial_shape, nb_features, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_dense(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int nb_neurons, prev_layer = -1, network_id = -1, strict_size = 0, current_layer_id = -1, silent = 0;
	const char *activation = "LIN", *init_fct = "xavier";
	double drop_rate = 0.0, py_bias = 0.0/0.0, init_scaling=-1.0;
	float *C_bias = NULL;
	static char *kwlist[] = {"nb_neurons", "activation", "bias", 
		"prev_layer", "drop_rate", "strict_size", "init_fct", "init_scaling", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|sdidisdii", kwlist, &nb_neurons, &activation, &py_bias, 
		&prev_layer, &drop_rate, &strict_size, &init_fct, &init_scaling, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(!isnan(py_bias)) //work with math.h only for C99 and later
	{
		C_bias = (float*) calloc(1,sizeof(float));
		*C_bias = (float) py_bias;
	}
	
	if((strcmp(init_fct,"normal") == 0 || strcmp(init_fct,"uniform") == 0) && init_scaling < 0.0)
	{
		printf("\n ERROR: init_scaling keywork is mandatory when using custom normal or uniform weight initialisation.\n");
		exit(EXIT_FAILURE);
	}
		
	current_layer_id = dense_create(networks[network_id], prev_layer, nb_neurons, activation, C_bias, drop_rate, 
		strict_size, init_fct, init_scaling, NULL, 0, 0, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_conv(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");

	int nb_filters, nb_groups = 1, group_param_sharing = 0, prev_layer = -1, network_id = -1, current_layer_id = -1, silent = 0;
	PyObject *py_f_size = NULL, *py_stride = NULL, *py_padding = NULL, *py_int_padding = NULL, *py_input_shape = NULL;
	int C_f_size[3] = {1,1,1}, C_stride[3] = {1,1,1}, C_padding[3] = {0,0,0}, C_int_padding[3] = {0,0,0}, C_input_shape[4] = {0,0,0,0};
	const char *activation = "LIN", *init_fct = "xavier";
	double drop_rate = 0.0, py_bias = 0.0 / 0.0, init_scaling = -1.0;
	float *C_bias = NULL;

	static char *kwlist[] = { "f_size", "nb_filters", "nb_groups", "group_param_sharing", "stride", "padding", "int_padding",
		"activation", "bias", "prev_layer", "input_shape", "drop_rate", "init_fct", "init_scaling", "silent", "network", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|iiOOOsdiOdsdii", kwlist,
		&py_f_size, &nb_filters, &nb_groups, &group_param_sharing, &py_stride, &py_padding, &py_int_padding, &activation,
		&py_bias, &prev_layer, &py_input_shape, &drop_rate, &init_fct, &init_scaling, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_f_size == Py_None)
	{
		printf("\n ERROR: f_size is mandatory, cannot be None.\n");
		exit(EXIT_FAILURE);
	}

	parse_int_sequence(py_f_size, C_f_size, 1, 3, "f_size");

	if(py_stride != NULL && py_stride != Py_None)
		parse_int_sequence(py_stride, C_stride, 1, 3, "stride");

	if(py_padding != NULL && py_padding != Py_None)
		parse_int_sequence(py_padding, C_padding, 1, 3, "padding");

	if(py_int_padding != NULL && py_int_padding != Py_None)
		parse_int_sequence(py_int_padding, C_int_padding, 1, 3, "int_padding");

	if(py_input_shape != NULL && py_input_shape != Py_None)
		parse_int_sequence(py_input_shape, C_input_shape, 4, 4, "input_shape");

	if(!isnan(py_bias))
	{
		C_bias = (float*)calloc(1, sizeof(float));
		*C_bias = (float)py_bias;
	}

	if((strcmp(init_fct, "normal") == 0 || strcmp(init_fct, "uniform") == 0) && init_scaling < 0.0)
	{
		printf("\n ERROR: init_scaling keywork is mandatory when using custom normal or uniform weight initialisation.\n");
		exit(EXIT_FAILURE);
	}

	current_layer_id = conv_create(networks[network_id], prev_layer, C_f_size, nb_filters, nb_groups, group_param_sharing, 
		C_stride, C_padding, C_int_padding, C_input_shape, activation, C_bias, drop_rate, init_fct, init_scaling, NULL, 0, 0, silent);

	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_pool(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int i;
	int prev_layer = -1, network_id = -1, global = 0, current_layer_id = -1, silent = 0;
	PyObject *py_pool_size = NULL, *py_pool_stride = NULL, *py_pool_padding = NULL;
	const char *s_pool_type = "MAX";
	const char *activation = "LIN";
	int C_pool_size[3] = {1,1,1};
	int C_pool_stride[3] = {1,1,1};
	int C_pool_padding[3] = {0,0,0};
	double drop_rate = 0.0;
	static char *kwlist[] = {"p_size", "stride", "padding", "prev_layer", "drop_rate", "p_type", "activation", "p_global", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOidssiii", kwlist, &py_pool_size, &py_pool_stride, &py_pool_padding, 
		&prev_layer, &drop_rate, &s_pool_type, &activation, &global, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_pool_size == Py_None)
	{
		printf("\n ERROR: p_size is mandatory, cannot be None.\n");
		exit(EXIT_FAILURE);
	}
	
	parse_int_sequence(py_pool_size, C_pool_size, 1, 3, "p_size");
	
	if(py_pool_stride != NULL && py_pool_stride != Py_None)
		parse_int_sequence(py_pool_stride, C_pool_stride, 1, 3, "stride");
	else
		for(i = 0; i < 3; i++)
			C_pool_stride[i] = C_pool_size[i];
	
	if(py_pool_padding != NULL && py_pool_padding != Py_None)
		parse_int_sequence(py_pool_padding, C_pool_padding, 1, 3, "padding");
		
	current_layer_id = pool_create(networks[network_id], prev_layer, C_pool_size, C_pool_stride, C_pool_padding,
		s_pool_type, activation, global, drop_rate, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_norm(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int prev_layer = -1, network_id = -1, current_layer_id = -1, cond_layer = auxil_input_id, silent = 0;
	const char *activation = "LIN";
	double gamma_start = 1.0f;
	int group_size = 8;
	static char *kwlist[] = {"activation", "prev_layer", "group_size", "gamma_start", "cond_layer", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|siidiii", kwlist, &activation, &prev_layer, &group_size, 
		&gamma_start, &cond_layer, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	current_layer_id = norm_create(networks[network_id], prev_layer, activation, group_size, gamma_start, cond_layer, NULL, 0, 0, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_lrn(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int prev_layer = -1, network_id = -1, current_layer_id = -1, silent = 0;
	const char *activation = "LIN";
	int range = 5;
	double k = 1.0, alpha = 1.0, beta = 0.5;
	static char *kwlist[] = {"activation", "prev_layer", "range", "k", "alpha", "beta", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|siidddii", kwlist, &activation, &prev_layer, &range, 
		&k, &alpha, &beta, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	current_layer_id = lrn_create(networks[network_id], prev_layer, activation, range, k, alpha, beta, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_grn(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int prev_layer = -1, network_id = -1, current_layer_id = -1, silent = 0;
	const char *activation = "LIN";
	double gamma_start = 0.001f;
	int residual = 1;
	static char *kwlist[] = {"residual", "activation", "prev_layer", "gamma_start", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|isidii", kwlist, &residual, &activation, &prev_layer, &gamma_start, &silent, &network_id))
		return NULL;
		
	if(network_id == -1)
		network_id = global_current_network_id;
	
	current_layer_id = grn_create(networks[network_id], prev_layer, activation, residual, gamma_start, NULL, 0, 0, silent);
	
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_merge(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int nb_previous, c_merge_type = 0, network_id = -1, current_layer_id = -1, silent = 0;
	PyObject *py_prev_list = NULL;
	PyArrayObject *array = NULL;
	int *C_prev_list = NULL;
	const char *activation = "LIN";
	const char *merge_type = "ADD";
	static char *kwlist[] = {"prev_list", "merge_type", "activation", "silent", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ssii", kwlist, &py_prev_list, &merge_type, &activation, &silent, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_prev_list == Py_None)
	{
		printf("\n ERROR: prev_list is mandatory, cannot be None.\n");
		exit(EXIT_FAILURE);
	}
	
	array = (PyArrayObject *)PyArray_FROM_OTF(py_prev_list, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
	if(array == NULL)
	{
		printf("\n ERROR: prev_list argument could not be converted to integers.\n");
		exit(EXIT_FAILURE);
	}
	if(PyArray_NDIM(array) != 1)
	{
		printf("\n ERROR: prev_list argument must be one-dimensional.\n");
		exit(EXIT_FAILURE);
	}
	
	nb_previous = PyArray_SIZE(array); //Check on nb_previous >= 2 is included in merge_create
	C_prev_list = (int *)PyArray_DATA(array);
	
	if(strcmp(merge_type,"ADD") == 0)
		c_merge_type = ADD_merge;
	else if(strcmp(merge_type,"CONCAT") == 0)
		c_merge_type = CONCAT_merge;
	
	current_layer_id = merge_create(networks[network_id], nb_previous, C_prev_list, c_merge_type, activation, silent);
	
	Py_DECREF(array);
	return PyLong_FromLong(current_layer_id);
}


static PyObject* py_set_frozen_layers(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int size;
	int network_id = -1;
	PyObject *py_froz_array = NULL;
	PyArrayObject *array = NULL;
	int *c_froz_array = NULL;
	static char *kwlist[] = {"froz_array", "network",NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &py_froz_array, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_froz_array == Py_None)
	{
		printf("\n ERROR: froz_array is mandatory, cannot be None.\n");
		exit(EXIT_FAILURE);
	}
	
	array = (PyArrayObject *)PyArray_FROM_OTF(py_froz_array, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
	if(array == NULL)
	{
		printf("\n ERROR: prev_list argument could not be converted to integers.\n");
		exit(EXIT_FAILURE);
	}
	if(PyArray_NDIM(array) != 1)
	{
		printf("\n ERROR: prev_list argument must be one-dimensional.\n");
		exit(EXIT_FAILURE);
	}
	
	size = PyArray_SIZE(array);
	c_froz_array = (int *)PyArray_DATA(array);

	set_frozen_layers(networks[network_id], c_froz_array, size);
	
	Py_DECREF(array);
	Py_RETURN_NONE;
}


static PyObject* py_set_IoU_limits(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
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
		return NULL;
	
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
	py_IoU_limits_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)IoU_limits_array);
	
	return py_IoU_limits_array;
}


static PyObject* py_set_fit_parts(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	setlocale(LC_ALL, "C");
	int *fit_parts_array = NULL;
	PyObject *py_fit_parts_array = NULL;
	int position = -2, size = -2, probability = -2, objectness = -2, classes = -2, parameters = -2;
	npy_intp dims[1];
	static char *kwlist[] = {"position", "size", "probability", "objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiii", kwlist, 
		&position, &size, &probability, &objectness, &classes, &parameters))
		return NULL;
	
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
	setlocale(LC_ALL, "C");
	float *error_scales_array = NULL;
	PyObject *py_error_scales_array = NULL;
	float position = -1.0f, size = -1.0f, probability = -1.0f, objectness = -1.0f, classes = -1.0f, parameters = -1.0f;
	npy_intp dims[1];
	static char *kwlist[] = {"position", "size", "probability", "objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|ffffff", kwlist, 
		&position, &size, &probability, &objectness, &classes, &parameters))
		return NULL;
	
	error_scales_array = (float*) calloc(6, sizeof(float));
	
	error_scales_array[0] = position;
	error_scales_array[1] = size;
	error_scales_array[2] = probability;
	error_scales_array[3] = objectness;
	error_scales_array[4] = classes;
	error_scales_array[5] = parameters;
	
	dims[0] = 6;
	py_error_scales_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)error_scales_array);
	
	return py_error_scales_array;
}


static PyObject* py_set_sm_single(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	float *sm_single_array = NULL;
	PyObject *py_sm_single_array = NULL;
	float slope = -1.0f, max = 100000.0f, min = -100000.0f;
	npy_intp dims[1];
	static char *kwlist[] = {"slope", "fmin", "fmax", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|fff", kwlist, &slope, &min, &max))
		return NULL;
	
	sm_single_array = (float*) calloc(3, sizeof(float));
	
	sm_single_array[0] = slope;
	sm_single_array[1] = max;
	sm_single_array[2] = min;
	
	dims[0] = 3;
	py_sm_single_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)sm_single_array);
	
	return py_sm_single_array;
}


static PyObject* py_set_slopes_and_maxes(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int i, j;
	float *slopes_and_maxes = NULL;
	PyObject *py_slopes_and_maxes = NULL;
	PyObject *py_position = NULL, *py_size = NULL, *py_probability = NULL;
	PyObject *py_objectness = NULL, *py_classes = NULL, *py_parameters = NULL;
	PyArrayObject *local[6];
	npy_intp dims[2];
	static char *kwlist[] = {"position", "size", "probability","objectness", "classes", "parameters", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOOO", kwlist, 
		&py_position, &py_size, &py_probability, &py_objectness, &py_classes, &py_parameters))
		return NULL;
	
	slopes_and_maxes = (float*) calloc(6*3, sizeof(float));
	
	local[0] = get_array_from_float_sequence(py_position,"position");
	local[1] = get_array_from_float_sequence(py_size,"size"); 
	local[2] = get_array_from_float_sequence(py_probability,"probability");
	local[3] = get_array_from_float_sequence(py_objectness,"objectness"); 
	local[4] = get_array_from_float_sequence(py_classes,"classes"); 
	local[5] = get_array_from_float_sequence(py_parameters,"parameters");
	
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
	py_slopes_and_maxes = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, (void*)slopes_and_maxes);
	
	return py_slopes_and_maxes;
}


static PyObject* py_set_yolo_params(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int i,j;
	int nb_box = 0, nb_class = 0, nb_param = 0, max_nb_obj_per_image = 0;
	int strict_box_size_association = 0, regret_based = 0, fit_dim = 0, rand_startup = -1, network_id = -1;
	int class_softmax = 0, diff_flag = 0, no_override = 0, raw_output = 0, input_branch_id = -1;
	double rand_prob_best_box_assoc = 0.0f, rand_prob = 0.0f, min_prior_forced_scaling = 0.0f, label_smoothing = 0.0f;
	PyObject *py_prior_size = NULL, *py_prior_noobj_prob = NULL;
	PyObject *py_error_scales = NULL, *py_slopes_and_maxes = NULL, *py_param_ind_scales = NULL, *py_IoU_limits = NULL, *py_fit_parts = NULL;
	PyArrayObject *array_prior_size = NULL, *array_prior_noobj_prob = NULL;
	PyArrayObject *array_error_scales = NULL, *array_slopes_and_maxes = NULL, *array_param_ind_scales = NULL;
	PyArrayObject *array_IoU_limits = NULL, *array_fit_parts = NULL;
	float *C_prior_size = NULL, *C_prior_noobj_prob = NULL;
	const char *IoU_type_char = "empty", *prior_dist_type_char = "empty", *error_type = "empty";
	float *error_scales = NULL, **slopes_and_maxes = NULL, *param_ind_scales = NULL, *IoU_limits = NULL;
	int *fit_parts = NULL;
	float* temp;
	static char *kwlist[] = {"nb_box", "nb_class", "nb_param", "max_nb_obj_per_image", "prior_size", "prior_noobj_prob", "error_scales", "slopes_and_maxes", 
		"param_ind_scales", "IoU_limits", "fit_parts", "IoU_type", "prior_dist_type", "strict_box_size", "regret_based", "fit_dim", "rand_startup", 
		"rand_prob_best_box_assoc", "rand_prob", "min_prior_forced_scaling", "class_softmax", "label_smoothing", "diff_flag", "network", "error_type", 
		"no_override", "raw_output", "input_branch_id", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiOOOOOOOssiiiidddidiisii", kwlist, 
			&nb_box, &nb_class, &nb_param, &max_nb_obj_per_image, &py_prior_size, &py_prior_noobj_prob, 
			&py_error_scales, &py_slopes_and_maxes, &py_param_ind_scales, &py_IoU_limits, &py_fit_parts, 
			&IoU_type_char, &prior_dist_type_char, &strict_box_size_association, &regret_based, &fit_dim, &rand_startup, &rand_prob_best_box_assoc, &rand_prob,
			&min_prior_forced_scaling, &class_softmax, &label_smoothing, &diff_flag, &network_id, &error_type, &no_override, &raw_output, &input_branch_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(py_prior_size != NULL && py_prior_size != Py_None)
	{
		array_prior_size = get_array_from_float_sequence(py_prior_size, "prior_size");
		
		if(fit_dim <= 0)
			fit_dim = array_prior_size->dimensions[0];
		else if(fit_dim > array_prior_size->dimensions[0])
		{
			printf("\n ERROR: fit_dim parameter cannot be larger than the number of dimensions of prior_size!\n");
			exit(EXIT_FAILURE);
		}
		
		if(array_prior_size->dimensions[1] != nb_box)
		{
			printf("\n ERROR: The prior_size array must have nb_box elements!");
			exit(EXIT_FAILURE);
		}
	
		C_prior_size = (float*) calloc(3*nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
		{
			for(j = 0; j < 3; j++)
			{
				if(j < fit_dim)
					C_prior_size[i*3+j] = *((float*)(array_prior_size->data 
											   + j * array_prior_size->strides[0] 
											   + i * array_prior_size->strides[1]));
				else
					C_prior_size[i*3+j] = 0.0f;
			}
		}
		Py_DECREF(array_prior_size);
	}
	
	if(py_prior_noobj_prob != NULL)
	{
		array_prior_noobj_prob = get_array_from_float_sequence(py_prior_noobj_prob, "prior_noobj_prob");
		
		if(array_prior_noobj_prob->dimensions[0] != nb_box)
		{
			printf("ERROR: The prior_noobj_prob array must have nb_box elements!");
			exit(EXIT_FAILURE);
		}
		C_prior_noobj_prob = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			C_prior_noobj_prob[i] = *((float *)(array_prior_noobj_prob->data 
									+ i * array_prior_noobj_prob->strides[0]));
		Py_DECREF(array_prior_noobj_prob);
	}
	
	if(py_error_scales != NULL)
	{
		array_error_scales = get_array_from_float_sequence(py_error_scales, "error_scales");
		
		error_scales = (float*) calloc(6, sizeof(float));
		for(i = 0; i < 6; i++)
			error_scales[i] = *(float *)(array_error_scales->data + i*array_error_scales->strides[0]);
		Py_DECREF(array_error_scales);
	}
	
	if(py_slopes_and_maxes != NULL)
	{
		array_slopes_and_maxes = get_array_from_float_sequence(py_slopes_and_maxes, "slopes_and_maxes");
	
		temp = (float*) calloc(6*3, sizeof(float));
		slopes_and_maxes = (float**) malloc(6*sizeof(float*));
		for(i = 0; i < 6; i++)
			slopes_and_maxes[i] = &temp[i*3];
		for(i = 0; i < 6; i++)
			for(j = 0; j < 3; j++)
				slopes_and_maxes[i][j] = *(float *)(array_slopes_and_maxes->data 
					+ i*array_slopes_and_maxes->strides[0] + j*array_slopes_and_maxes->strides[1]);
		Py_DECREF(array_slopes_and_maxes);
	}
	
	if(py_param_ind_scales != NULL)
	{
		array_param_ind_scales = get_array_from_float_sequence(py_param_ind_scales, "param_ind_scales");
		
		if(array_param_ind_scales->dimensions[0] != nb_param)
		{
			printf("ERROR: The param_ind_scales array must have nb_param elements!");
			exit(EXIT_FAILURE);
		}
		param_ind_scales = (float*) calloc(nb_param, sizeof(float));
		for(i = 0; i < nb_param; i++)
			param_ind_scales[i] = *(float *)(array_param_ind_scales->data + i*array_param_ind_scales->strides[0]);
		Py_DECREF(array_param_ind_scales);
	}
	
	if(py_IoU_limits != NULL)
	{
		array_IoU_limits = get_array_from_float_sequence(py_IoU_limits, "IoU_limits");
		
		IoU_limits = (float*) calloc(8, sizeof(float));
		for(i = 0; i < 8; i++)
			IoU_limits[i] = *(float *)(array_IoU_limits->data + i*array_IoU_limits->strides[0]);
		Py_DECREF(array_IoU_limits);
	}
	
	if(py_fit_parts != NULL)
	{
		array_fit_parts = get_array_from_float_sequence(py_fit_parts, "fit_parts");
		
		fit_parts = (int*) calloc(6, sizeof(int));
		for(i = 0; i < 6; i++)
			fit_parts[i] = *(int *)(array_fit_parts->data + i*array_fit_parts->strides[0]);
		Py_DECREF(array_fit_parts);
	}
	
	return PyLong_FromLong(set_yolo_config(networks[network_id], nb_box, nb_class, nb_param, max_nb_obj_per_image, 
		IoU_type_char, prior_dist_type_char, C_prior_size, C_prior_noobj_prob, fit_dim, strict_box_size_association, 
		regret_based, rand_startup, rand_prob_best_box_assoc, rand_prob, min_prior_forced_scaling, error_scales, slopes_and_maxes, 
		param_ind_scales, IoU_limits, fit_parts, class_softmax, label_smoothing, diff_flag, error_type, no_override, raw_output, input_branch_id));
}

static PyObject* perf_eval(PyObject* self, PyObject* args)
{
	setlocale(LC_ALL, "C");
	int network_id = -1;
	if(!PyArg_ParseTuple(args, "|i", &network_id))
		return NULL;

	if(network_id == -1)
		network_id = global_current_network_id;

	perf_eval_display(networks[network_id]);
	
	Py_RETURN_NONE;
}


static PyObject* py_load_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	const char *file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int iter, network_id = -1, nb_layers = 0, nb_skip_layers = 0, f_bin = 0, silent = 0;
	static char *kwlist[] = {"file", "iteration", "network", "nb_layers", "nb_skip_layers", "bin", "silent", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "si|iiiii", kwlist, &file, &iter, &network_id, &nb_layers, &nb_skip_layers, &f_bin, &silent))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	load_network(networks[network_id], file, iter, nb_layers, nb_skip_layers, f_bin, silent);
	
	return PyLong_FromLong(networks[network_id]->nb_layers - 1);
}

static PyObject* py_save_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	const char *file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int network_id = -1, save_optim = 0, f_bin = 0;
	static char *kwlist[] = {"file", "network", "save_optim", "bin", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "s|iii", kwlist, &file, &network_id, &save_optim, &f_bin))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	save_network(networks[network_id], file, save_optim, f_bin);
	
	Py_RETURN_NONE;
}


// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int nb_iter, control_interv = 1, wema_replace_every = 0, decoupled_wdecay = 0, network_id = -1;
	int confmat = 0, save_every = 0, shuffle_gpu = 1, shuffle_every = 1, silent = 0, save_bin = 0, save_optim_every = 0;
	double learning_rate = 0.0, momentum = 0.0, decay = 0.0, end_learning_rate = 0.0;
	double wema_rate = -1.0, TC_scale_factor = 1.0, weight_decay = 0.0;
	static char *kwlist[] = {"nb_iter", "learning_rate", "end_learning_rate", "control_interv", "momentum", "lr_decay", 
		"weight_decay", "decoupled_wdecay", "wema_rate", "wema_replace_every", "confmat", "save_every", "save_bin",
		"save_optim_every", "network", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|didddidiiiiiiiidi", kwlist, &nb_iter, &learning_rate, &end_learning_rate, 
		&control_interv, &momentum, &decay, &weight_decay, &decoupled_wdecay, &wema_rate, &wema_replace_every, &confmat, 
		&save_every, &save_bin, &save_optim_every, &network_id, &shuffle_gpu, &shuffle_every, &TC_scale_factor, &silent))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(momentum > 0.0)
		printf(" WARNING: Setting momentum in train is deprecated. This is now part of optimizer setting in init, and can be updated with the appropriate set optimizer parameter function.\n");
	
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS;
	
	train_network(networks[network_id], nb_iter, control_interv, learning_rate, end_learning_rate, 
		decay, weight_decay, decoupled_wdecay, wema_rate, wema_replace_every, confmat, save_every, 
		save_optim_every, save_bin, shuffle_gpu, shuffle_every, TC_scale_factor, silent);
		
	Py_END_ALLOW_THREADS;

	Py_RETURN_NONE;
}


static PyObject* py_forward_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	size_t i, j;
	size_t nb_data, out_size;
	int repeat = 1, network_id = -1, C_drop_mode = AVG_MODEL, no_error = 1, saving = 1, silent = 0, return_output = 0;
	float *c_output = NULL, *arr_data;
	const char *drop_mode = "AVG_MODEL";
	static char *kwlist[] = {"saving", "drop_mode", "no_error", "repeat", "network", "silent", "return_output", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|isiiiii", kwlist, &saving, &drop_mode, &no_error, &repeat, &network_id, &silent, &return_output))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	if(strcmp(drop_mode, "AVG_MODEL") == 0)
		C_drop_mode = AVG_MODEL;
	else if(strcmp(drop_mode, "MC_MODEL") == 0)
		C_drop_mode = MC_MODEL;
	
	networks[network_id]->no_error = no_error;
	
	Py_BEGIN_ALLOW_THREADS;
	
	c_output = pred_testset(networks[network_id], saving, repeat, C_drop_mode, silent, return_output);
	
	Py_END_ALLOW_THREADS;
	
	if(return_output)
	{
		nb_data = networks[network_id]->test.size;
		out_size = networks[network_id]->out_size; 
		if(networks[network_id]->net_layers[networks[network_id]->nb_layers-1]->output_type == FLAT)//no extra +1 dim
			out_size -= 1;
			
		npy_intp dims[2] = {nb_data, out_size};
		PyObject *py_output = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
		
		arr_data = (float*)PyArray_DATA((PyArrayObject*) py_output);
		for(i = 0; i < nb_data*(size_t)repeat; i++)
			for(j = 0; j < out_size; j++)
				arr_data[i*out_size + j] = c_output[i*out_size + j];
		
		free(c_output);
		return py_output;
	}
	else
		Py_RETURN_NONE;
}

static PyObject* py_print_architecture_tex(PyObject* self, PyObject *args, PyObject *kwargs)
{
	setlocale(LC_ALL, "C");
	int network_id = -1, l_size = 1, l_in_size = 1, l_f_size = 1, l_out_size = 1, 
		l_stride = 1, l_padding = 1, l_in_padding = 0, l_activation = 0, l_bias = 0, l_dropout = 0, l_param_count = 0;
	const char *path = "relative_path_to_the_save_file_location_which_must_be_long_enough", *file_name = "A_long_filename_template_for_user_setup";
	static char *kwlist[] = {"path", "file_name", "size", "in_size", "f_size", "out_size", 
		"stride", "padding", "in_padding", "activation", "bias", "dropout", "param_count", "network", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|iiiiiiiiiiii", kwlist, &path, &file_name, &l_size, &l_in_size, 
		&l_f_size, &l_out_size, &l_stride, &l_padding, &l_in_padding, &l_activation, &l_bias, &l_dropout, &l_param_count, &network_id))
		return NULL;
	
	if(network_id == -1)
		network_id = global_current_network_id;
	
	print_architecture_tex(networks[network_id], path, file_name,
		l_size, l_in_size, l_f_size, l_out_size, l_stride, l_padding, 
		l_in_padding, l_activation, l_bias, l_dropout, l_param_count);
	
	Py_RETURN_NONE;
}


// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
	{ "init", (PyCFunction)py_init_network, METH_VARARGS | METH_KEYWORDS, "Initialize network basic shapes and properties" },
	{ "free_network", (PyCFunction)py_free_network, METH_VARARGS | METH_KEYWORDS, "Free all data structures associated with a given network" },
	{ "create_dataset", (PyCFunction)py_create_dataset, METH_VARARGS | METH_KEYWORDS, "Allocate dataset structure" },
	{ "delete_dataset", (PyCFunction)py_delete_dataset, METH_VARARGS | METH_KEYWORDS, "Free dataset structure" },
	{ "swap_data_buffers", py_swap_data_buffers, METH_VARARGS, "Put the selected buffered dataset as current dataset for training"},
	{ "set_optimizer_param", (PyCFunction)py_set_optimizer_param, METH_VARARGS | METH_KEYWORDS, "Update the optimizer parameter based on a formated string"},
	{ "sgd", (PyCFunction)py_sgd, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to the SGD optimizer"},
	{ "adam", (PyCFunction)py_adam, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to the ADAM optimizer"},
	{ "rms_prop", (PyCFunction)py_rms_prop, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to the RMSprop optimizer"},
	{ "linear", (PyCFunction)py_linear, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Linear"},
	{ "relu", (PyCFunction)py_relu, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to ReLU"},
	{ "logistic", (PyCFunction)py_logistic, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Logistic"},
	{ "softmax", (PyCFunction)py_softmax, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Softmax"},
	{ "yolo", (PyCFunction)py_yolo, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to YOLO"},
	{ "input", (PyCFunction)py_input, METH_VARARGS | METH_KEYWORDS, "Add an input layer to the network" },
	{ "dense", (PyCFunction)py_dense, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
	{ "conv",(PyCFunction)py_conv, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
	{ "pool",(PyCFunction)py_pool, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
	{ "norm", (PyCFunction)py_norm, METH_VARARGS | METH_KEYWORDS, "Add a normalization layer to the network"},
	{ "lrn", (PyCFunction)py_lrn, METH_VARARGS | METH_KEYWORDS, "Add a Local Response Normalization layer to the network"},
	{ "grn", (PyCFunction)py_grn, METH_VARARGS | METH_KEYWORDS, "Add a Global Response Normalization layer to the network"},
	{ "merge", (PyCFunction)py_merge, METH_VARARGS | METH_KEYWORDS, "Add a marge layer to the network"},
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
	
	PyObject *m;
	m = PyModule_Create(&CIANNA);
	if (m == NULL)
		return NULL;
	
	return m;
}




















