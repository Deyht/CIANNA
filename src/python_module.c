
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
	int dims[4] = {1,1,1,1}, nb_channels = 1, out_dim, b_size, network_id = nb_networks;
	int dynamic_load = 0, no_logo = 0;
	const char *py_mixed_precision = "off";
	const char *comp_meth = "C_CUDA";
	static char *kwlist[] = {"in_dim", "in_nb_ch", "out_dim", "bias", "b_size", "comp_meth", "network_id", "dynamic_load", "mixed_precision", "no_logo", NULL};
	
	b_size = 10;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii|disiisi", kwlist, &py_dims, &nb_channels, &out_dim, &bias, &b_size, &comp_meth, &network_id, &dynamic_load, &py_mixed_precision, &no_logo))
	    return Py_None;
	
	for(i = 0; i < py_dims->dimensions[0] && i < 3; i++)
	{
		dims[i] = *(int *)(py_dims->data + i*py_dims->strides[0]);
	}
	dims[3] = nb_channels;
	
    init_network(network_id, dims, out_dim, bias, b_size, comp_meth, dynamic_load, py_mixed_precision, no_logo);
	
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
	static char *kwlist[] = {"dataset", "size", "input", "target", "network_id", "silent", NULL};

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
		printf("input dim :%d,", networks[network_id]->input_dim);
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
	static char *kwlist[] = {"dataset", "network_id", "silent", NULL};

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
	int nb_neurons, prev_layer = -1, network_id = nb_networks-1, strict_size = 0;
	const char *activation = "RELU";
	double drop_rate = 0.0, py_bias = 0.0/0.0;
	float *c_bias = NULL;
	static char *kwlist[] = {"nb_neurons", "activation", "bias", "prev_layer", "drop_rate", "strict_size", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|sdidii", kwlist, &nb_neurons, &activation, &py_bias, &prev_layer, &drop_rate, &strict_size, &network_id))
	    return Py_None;

	if(prev_layer == -1)
		prev_layer = networks[network_id]->nb_layers - 1;
	
	
	//if(py_bias == py_bias) //portable but not compatible with fast math
	if(!isnan(py_bias)) //work with math.h only for C99 and later
	{
		c_bias = (float*) calloc(1,sizeof(float));
		*c_bias = (float) py_bias;
	}
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	dense_create(networks[network_id], prev, nb_neurons, activation, c_bias, drop_rate, strict_size, NULL, 0);
	
	return Py_None;
}

static PyObject* py_conv(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int i;
	int nb_filters, prev_layer = -1, network_id = nb_networks-1;
	PyArrayObject *py_f_size = NULL, *py_stride = NULL, *py_padding = NULL, *py_int_padding = NULL, *py_input_shape = NULL;
	int C_f_size[3] = {1,1,1}, C_stride[3] = {1,1,1}, C_padding[3] = {0,0,0}, C_int_padding[3] = {0,0,0}, C_input_shape[4];
	const char *activation = "RELU";
	double drop_rate = 0.0, py_bias = 0.0/0.0; //use nan as "non specified value"
	float *c_bias = NULL;
	
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "int_padding", "activation", "bias", "prev_layer", "input_shape", "drop_rate", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|OOOsdiOdi", kwlist, &py_f_size, &nb_filters, &py_stride, &py_padding, 
		&py_int_padding, &activation, &py_bias, &prev_layer, &py_input_shape, &drop_rate, &network_id))
	    return Py_None;   
	
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
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	conv_create(networks[network_id], prev, C_f_size, nb_filters, C_stride, C_padding, C_int_padding, C_input_shape, activation, c_bias, drop_rate, NULL, 0);
	
	return Py_None;
}


static PyObject* py_pool(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int i;
	int prev_layer = -1, network_id = nb_networks-1, C_pool_type = MAX_pool;
	PyArrayObject *py_pool_size;
	const char *s_pool_type = "MAX";
	int C_pool_size[3] = {1,1,1};
	double drop_rate = 0.0;
	static char *kwlist[] = {"p_size", "prev_layer", "drop_rate", "p_type", "network", NULL};
	layer* prev;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oidsi", kwlist, &py_pool_size, &prev_layer, &drop_rate, &s_pool_type, &network_id))
	    return Py_None;
	
	if(prev_layer == -1)
	    prev_layer = networks[network_id]->nb_layers - 1;

	// Set default to 2 for all dim > 1
	for(i = 0; i < 3; i++)
		if(networks[network_id]->in_dims[i] > 1)
			C_pool_size[i] = 2;
	
	for(i = 0; i < py_pool_size->dimensions[0] && i < 3; i++)
		C_pool_size[i]  = *(int *)(py_pool_size->data  + i*py_pool_size->strides[0]);

	if(strcmp(s_pool_type, "MAX") == 0)
		C_pool_type = MAX_pool;
	else if(strcmp(s_pool_type, "AVG") == 0)
		C_pool_type = AVG_pool;
	else
		C_pool_type = MAX_pool;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	pool_create(networks[network_id], prev, C_pool_size, C_pool_type, drop_rate);
	
	return Py_None;
}

static PyObject* py_set_yolo_params(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i,j;
	int nb_box = 0, nb_class = 0, nb_param = 0, max_nb_obj_per_image = 0, IoU_type;
	int strict_box_size_association = 0, fit_dim = 0, rand_startup = -1, network_id = 0;
	double rand_prob_best_box_assoc = -1.0f, min_prior_forced_scaling = -1.0f;
	PyArrayObject *py_prior_w = NULL, *py_prior_h = NULL, *py_prior_d = NULL, *py_prior_noobj_prob = NULL;
	float *C_prior_w = NULL, *C_prior_h = NULL, *C_prior_d = NULL, *C_prior_noobj_prob = NULL;
	PyArrayObject *py_error_scales = NULL, *py_slopes_and_maxes = NULL, *py_param_ind_scales = NULL, *py_IoU_limits = NULL, *py_fit_parts = NULL;
	const char* IoU_type_char = "empty";
	float *error_scales = NULL, **slopes_and_maxes = NULL, *param_ind_scales = NULL, *IoU_limits = NULL;
	int *fit_parts = NULL;
	float* temp;
	static char *kwlist[] = {"nb_box", "nb_class", "nb_param", "max_nb_obj_per_image", "prior_w", "prior_h", "prior_d", "prior_noobj_prob", "error_scales", "slopes_and_maxes", "param_ind_scales", "IoU_limits", "fit_parts", "IoU_type", "strict_box_size", "fit_dim", "rand_startup", "rand_prob_best_box_assoc", "min_prior_forced_scaling", "network", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiO|OOOOOOOOsiiiddi", kwlist, 
			&nb_box, &nb_class, &nb_param, &max_nb_obj_per_image, &py_prior_w, &py_prior_h, &py_prior_d, &py_prior_noobj_prob, 
			&py_error_scales, &py_slopes_and_maxes, &py_param_ind_scales, &py_IoU_limits, &py_fit_parts, 
			&IoU_type_char, &strict_box_size_association, &fit_dim, &rand_startup, &rand_prob_best_box_assoc,
			&min_prior_forced_scaling, &network_id))
	    return PyLong_FromLong(0);

	// All default values are defined in activ_functions.c
	
	if(strcmp(IoU_type_char, "IoU") == 0)
		IoU_type = IOU;
	else if(strcmp(IoU_type_char, "GIoU") == 0)
		IoU_type = GIOU;
	else if(strcmp(IoU_type_char, "DIoU") == 0)
		IoU_type = DIOU;
	else if(strcmp(IoU_type_char, "DIoU2") == 0)
		IoU_type = DIOU2;
	else
	{
		printf("Warning: Unrecognized IoU type: %s, fallback to default GIoU\n", IoU_type_char);
		IoU_type = GIOU;
	}
	
	if(py_prior_w != NULL)
	{
		C_prior_w = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
		{ 
			C_prior_w[i] = *((float*)(py_prior_w->data + i * py_prior_w->strides[0]));
		}
	}
	
	if(py_prior_h != NULL)
	{
		C_prior_h = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			C_prior_h[i] = *((float*)(py_prior_h->data + i * py_prior_h->strides[0]));
	}
	
	if(py_prior_d != NULL)
	{
		C_prior_d = (float*) calloc(nb_box,sizeof(float));
		for(i = 0; i < nb_box; i++)
			C_prior_d[i] = *((float*)(py_prior_d->data + i * py_prior_d->strides[0]));
	}
	
	if(py_prior_noobj_prob != NULL)
	{
		C_prior_noobj_prob = (float*) calloc(nb_box,sizeof(float));
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
				slopes_and_maxes[i][j] = *(float *)(py_slopes_and_maxes->data + i*py_slopes_and_maxes->strides[0] + j*py_slopes_and_maxes->strides[1]);
	}
	
	if(py_param_ind_scales != NULL)
	{
		param_ind_scales = (float*) calloc(nb_param, sizeof(float));
		for(i = 0; i < nb_param; i++)
			param_ind_scales[i] = *(float *)(py_param_ind_scales->data + i*py_param_ind_scales->strides[0]);
	}
	
	if(py_IoU_limits != NULL)
	{
		IoU_limits = (float*) calloc(6, sizeof(float));
		for(i = 0; i < 6; i++)
			IoU_limits[i] = *(float *)(py_IoU_limits->data + i*py_IoU_limits->strides[0]);
	}
	
	if(py_fit_parts != NULL)
	{
		fit_parts = (int*) calloc(5, sizeof(int));
		for(i = 0; i < 5; i++)
			fit_parts[i] = *(int *)(py_fit_parts->data + i*py_fit_parts->strides[0]);
	}
	
	return PyLong_FromLong(set_yolo_params(networks[network_id], nb_box, nb_class, nb_param, max_nb_obj_per_image, 
		IoU_type, C_prior_w, C_prior_h, C_prior_d, C_prior_noobj_prob, fit_dim, strict_box_size_association, 
		rand_startup, rand_prob_best_box_assoc, min_prior_forced_scaling, error_scales, 
		slopes_and_maxes, param_ind_scales, IoU_limits, fit_parts));
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
	char *file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int epoch, network_id = nb_networks-1, f_bin = 0;
	static char *kwlist[] = {"file", "epoch", "network", "bin", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "si|ii", kwlist, &file, &epoch, &network_id, &f_bin))
	    return Py_None;
	    
	load_network(networks[network_id], file, epoch, f_bin);
	
	return Py_None;
}


// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	
	int py_nb_epoch, py_control_interv = 1, py_confmat = 0, save_every = 0, network_id = nb_networks-1;
	int shuffle_gpu = 1, shuffle_every = 1, silent = 0, save_bin = 0;
	double py_learning_rate=0.02, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0, py_TC_scale_factor = 1.0;
	static char *kwlist[] = {"nb_epoch", "learning_rate", "end_learning_rate", "control_interv", "momentum", "decay", "confmat", "save_every", "save_bin", "network", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "silent", NULL};
	
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|diddiiiiiidi", kwlist, &py_nb_epoch, &py_learning_rate, &py_end_learning_rate, &py_control_interv, &py_momentum, &py_decay, &py_confmat, &save_every, &save_bin, &network_id, &shuffle_gpu, &shuffle_every, &py_TC_scale_factor, &silent))
	    return Py_None;
	
	if(silent == 0)
		printf("\nNetwork %d training with :\n  nb_epoch: %d, control_interv: %d, save_every: %d, save_bin: %d \n  learning_rate: %g, end_learning_rate: %g , momentum: %0.2f, decay: %g \n  confmat: %d, shuffle_gpu: %d , shuffle_every: %d, TC_scale_factor: %g\n", 
			network_id, py_nb_epoch, py_control_interv, save_every, save_bin, py_learning_rate, py_end_learning_rate, py_momentum, 
			py_decay, py_confmat, shuffle_gpu, shuffle_every, py_TC_scale_factor);
		
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS
	
	train_network(networks[network_id], py_nb_epoch, py_control_interv, 
		py_learning_rate, py_end_learning_rate, py_momentum, py_decay, py_confmat, save_every, save_bin, shuffle_gpu, shuffle_every, py_TC_scale_factor);
		
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
	{
		C_drop_mode = AVG_MODEL;
	}
    else if(strcmp(drop_mode, "MC_MODEL") == 0)
	{
		C_drop_mode = MC_MODEL;
	}

	networks[network_id]->no_error = no_error;
	
	Py_BEGIN_ALLOW_THREADS
	
	forward_testset(networks[network_id], step, saving, repeat, C_drop_mode);
	
	Py_END_ALLOW_THREADS
	
	return Py_None;
}

#ifdef CUDA
// Experimental function for now, only for development purpose
static PyObject* py_gan_train(PyObject* self, PyObject *args, PyObject *kwargs)
{
	
	int py_nb_epoch, py_control_interv = 1, py_confmat = 0, save_every = 0, gen_id = nb_networks-2, disc_id = nb_networks-1;
	int shuffle_gpu = 1, shuffle_every = 1, silent = 0, disc_only = 0;
	double py_learning_rate=0.02, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0, py_TC_scale_factor = 1.0, py_gen_disc_lr_ratio = 2.0;
	static char *kwlist[] = {"gen_id", "disc_id", "nb_epoch", "learning_rate", "end_learning_rate", "gen_disc_lr_ratio","control_interv", "momentum", "decay", "confmat", "save_every", "shuffle_gpu", "shuffle_every", "TC_scale_factor", "disc_only", "silent", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "iiid|ddiddiiiidiu", kwlist, &gen_id, &disc_id, &py_nb_epoch, &py_learning_rate, &py_end_learning_rate, &py_gen_disc_lr_ratio, 
		&py_control_interv, &py_momentum, &py_decay, &py_confmat, &save_every, &shuffle_gpu, &shuffle_every, &py_TC_scale_factor, &disc_only, &silent))
	    return Py_None;
	
	if(silent == 0)
		printf("py_nb_epoch %d, py_control_interv %d, py_learning_rate %g, py_end_learning_rate %g , py_momentum %0.2f, py_decay %g, py_confmat %d, save_every %d, shuffle_gpu %d , shuffle_every %d, TC_scale_factor %g\n", 
			py_nb_epoch, py_control_interv, py_learning_rate, py_end_learning_rate, py_momentum, 
			py_decay, py_confmat, save_every, shuffle_gpu, shuffle_every, py_TC_scale_factor);
	
	// GIL MACRO : Allow to serialize C thread with python threads
	Py_BEGIN_ALLOW_THREADS
	
	train_gan(networks[gen_id], networks[disc_id], py_nb_epoch, py_control_interv, py_learning_rate, py_end_learning_rate, 
		py_momentum, py_decay, py_gen_disc_lr_ratio, save_every, 0, shuffle_gpu, shuffle_every, disc_only, py_TC_scale_factor);
	
	Py_END_ALLOW_THREADS

	return Py_None;
}
#endif


// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
    { "init", (PyCFunction)py_init_network, METH_VARARGS | METH_KEYWORDS, "Initialize network basic shapes and properties" },
    { "create_dataset", (PyCFunction)py_create_dataset, METH_VARARGS | METH_KEYWORDS, "Allocate dataset structure" },
    { "delete_dataset", (PyCFunction)py_delete_dataset, METH_VARARGS | METH_KEYWORDS, "Free dataset structure" },
    { "swap_data_buffers", py_swap_data_buffers, METH_VARARGS, "Put the selected buffered dataset as current dataset for training"},
    /*{ "write_formated_dataset", (PyCFunction)py_write_formated_dataset, METH_VARARGS | METH_KEYWORDS, "Write a proper numpy table onto a formated binary dataset file"},*/
    /*{ "load_formated_dataset", (PyCFunction)py_load_formated_dataset, METH_VARARGS | METH_KEYWORDS, "Read a formated binary dataset file and directly store it into a network dataset"},*/
    /*{ "set_normalize_factors", (PyCFunction)py_set_normalize_factors, METH_VARARGS | METH_KEYWORDS, "Set normalization factor for transformation on all the datasets subsequently loaded (with formated loading) in the C framework"},*/
    { "linear", (PyCFunction)py_linear, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Linear"},
    { "relu", (PyCFunction)py_relu, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to ReLU"},
    { "logistic", (PyCFunction)py_logistic, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Logistic"},
    { "softmax", (PyCFunction)py_softmax, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to Softmax"},
    { "yolo", (PyCFunction)py_yolo, METH_VARARGS | METH_KEYWORDS, "Create the string layout corresponding to YOLO"},
    { "dense", (PyCFunction)py_dense, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
    { "conv",(PyCFunction)py_conv, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
    { "set_yolo_params",(PyCFunction)py_set_yolo_params, METH_VARARGS | METH_KEYWORDS, "Set parameters for YOLO output layout" },
    { "pool",(PyCFunction)py_pool, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
    { "perf_eval", perf_eval, METH_VARARGS, "Display each layer time in ms and in percent of the total networj time" },
    { "load", (PyCFunction)py_load_network, METH_VARARGS | METH_KEYWORDS, "Load a previous network structure pre-trained from a file" },
    { "train", (PyCFunction)py_train_network, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
    { "forward", (PyCFunction)py_forward_network, METH_VARARGS | METH_KEYWORDS, "Apply the trained network to the test set and save results" },
    #ifdef CUDA
    { "gan_train", (PyCFunction)py_gan_train, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
    #endif
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
Importing CIANNA Python module V-p.0.6.3 , by D.Cornu\n\
###################################################################\n\n");

	PyObject *m;
    m = PyModule_Create(&CIANNA);
    if (m == NULL)
        return NULL;
	
    return m;
}












// Outdated (non functional anymore) formated dataset approach. Kept for possible futur re-integration.
// Should be replaced by the new dynamic loading capabilities of create/swap/delete dataset functions 
/*
static PyObject* py_write_formated_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int i, j, k, l, m;
	const char *filename;
	const char *input_data_type, *output_data_type;
	int input_data_type_C = c_FP32, output_data_type_C = c_FP32;
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
*/

/*
static PyObject* py_load_formated_dataset(PyObject* self, PyObject *args, PyObject *kwargs)
{
	Dataset data;
	const char *dataset_type;
	const char *filename;
	const char *input_data_type, *output_data_type;
	int input_data_type_C = c_FP32, output_data_type_C = c_FP32;
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
	
	struct timeval time;
	
	init_timing(&time);
	data = load_formated_dataset(networks[network_id], filename, input_data_type_C, output_data_type_C);
	if(silent == 0)
		printf("Time load %f\n",ellapsed_time(time));
	init_timing(&time);
	normalize_dataset(networks[network_id], data);
	if(silent == 0)
		printf("Time normalise %f\n",ellapsed_time(time));
	
	if(networks[network_id]->compute_method == C_CUDA)
	{
		#ifdef CUDA
		if(networks[network_id]->cu_inst.dynamic_load == 0)
			cuda_convert_dataset(networks[network_id], &data);
		#endif
	}
	//else if(networks[network_id]->compute_method == C_CUDA && networks[network_id]->dynamic_load == 1 
	//				&& networks[network_id]->use_cuda_TC)
	//	cuda_convert_host_dataset_FP32(networks[network_id], &data);
	
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
	else if(strcmp(dataset_type,"TRAIN_buf") == 0)
	{
		//printf("Testing test loaded\n");
		networks[network_id]->train_buf = data;
	}
	else if(strcmp(dataset_type,"VALID_buf") == 0)
	{
		//printf("Testing test loaded\n");
		networks[network_id]->valid_buf = data;
	}
	else if(strcmp(dataset_type,"TEST_buf") == 0)
	{
		//printf("Testing test loaded\n");
		networks[network_id]->test_buf = data;
	}
	
	if(silent == 0)
		printf(" Done!\n");
		
	return Py_None;
}
*/


/*
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
*/





















