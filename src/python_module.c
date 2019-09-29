#include <Python.h>
#include <numpy/arrayobject.h>

#include <string.h>
#include "prototypes.h"


Dataset train_set, valid_set, test_set;

// Structures or object related
//############################################################


// Network paramaremeter and data management functions
//############################################################

static PyObject* py_init_network(PyObject* self, PyObject* args)
{
	PyArrayObject *py_dims;
	int i;
	int dims[3] = {1,1,1}, out_dim, b_size, comp_int = C_CUDA;
	char string_comp[10];
	const char *comp_meth = "C_CUDA";
	
	b_size = 10;
	
	if(!PyArg_ParseTuple(args, "Oi|is", &py_dims, &out_dim, &b_size, &comp_meth))
	    return Py_None;
	
	for(i = 0; i < py_dims->dimensions[0]; i++)
	{
		dims[i] = *(int *)(py_dims->data + i*py_dims->strides[0]);
	}
	printf("%s\n", comp_meth);
	
	if(strcmp(comp_meth,"C_CUDA") == 0)
	{
		printf("test 1\n");
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
	
    init_network(dims, out_dim, b_size, comp_int);
    
	printf("Network have been initialized with : \nInput dimensions: %dx%dx%d \nOutput dimension: %d \nBatch size: %d \nUsing %s compute methode\n\n", dims[0], dims[1], dims[2], output_dim, batch_size, string_comp);
	
    return Py_None;
}


static PyObject* py_create_dataset(PyObject* self, PyObject* args)
{
	int i, j, k, l, m;
	Dataset *data = NULL;
	const char *dataset_type;
	double bias = 0.1;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size, flat = 0;

	if(!PyArg_ParseTuple(args, "siOOd|i", &dataset_type, &size, &py_data, &py_target, &bias, &flat))
	    return Py_None;
	    
	printf("test\n");
	
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		printf("setting training set\n");
		data = &train_set;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		printf("setting valid set\n");
		data = &valid_set;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		printf("setting testing test\n");
		data = &test_set;
	}
	
	
	printf("input dim :%d\n", input_dim);
	*data = create_dataset(size, bias);
	
	printf("%d \n", data->nb_batch);
	
	
	printf("Creating dataset with size %d ...\n", data->size);
	
	printf("%ld %ld\n", py_data->dimensions[0], py_data->dimensions[1]); 
	
	
	if(py_data != NULL && py_target != NULL)
	{
		printf("loading input\n");
		for(i = 0; i < data->nb_batch; i++)
			for(j = 0; j < batch_size; j++)
			{
				if(i*batch_size + j >= data->size)
					continue;
				for(k = 0; k < input_depth; k++)
				{
					for(l = 0; l < input_height; l++)
						for(m = 0; m < input_width; m++)
						{
							if(!flat)
	data->input[i][j*(input_dim+1) + k*input_height*input_width + l*input_width + m] = *(double*)(py_data->data + (i*batch_size*input_depth*input_height + j*input_depth*input_height + k*input_height + l)*py_data->strides[0] + m*py_data->strides[1]);
							else
	data->input[i][j*(input_dim+1) + k*input_height*input_width + l*input_width + m] = *(double*)(py_data->data + (i*batch_size + j)*py_data->strides[0] + (k*input_height*input_width + l*input_width + m)*py_data->strides[1]);	
						}
				}
			}
		printf("loading target\n");
		for(i = 0; i < data->nb_batch; i++)
			for(j = 0; j < batch_size; j++)
			{
				if(i*batch_size + j >= data->size)
					continue;
				for(k = 0; k < output_dim; k++)
					data->target[i][j*output_dim + k] = 
						*(double*)(py_target->data + i * (batch_size*py_target->strides[0]) + j * py_target->strides[0] + k*py_target->strides[1]);
			}
	}
	
	#ifdef CUDA
	if(compute_method == C_CUDA)
	{
		printf("Converting dataset to GPU device (CUDA)\n");
		cuda_convert_dataset(data);
	}
	#endif
	
	return Py_None;
}



// Layers functions
//############################################################
static PyObject* py_dense_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int nb_neurons, prev_layer, i_activ = RELU;
	const char *activation = "RELU";
	double drop_rate = 0.0;
	static char *kwlist[] = {"nb_neurons", "activation", "prev_layer", "drop_rate", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|sid", kwlist, &nb_neurons, &activation, &prev_layer, &drop_rate))
	    return Py_None;

	if(strcmp(activation, "RELU") == 0)
		i_activ = RELU;
	else if(strcmp(activation, "LINEAR") == 0)
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC") == 0)
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX") == 0)
		i_activ = SOFTMAX;
	
	printf("prev : %d\n", prev_layer);
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	dense_create(prev, nb_neurons, i_activ, drop_rate);
	
	printf("Added dense layer, L:%d\n", nb_layers-1);
	
	return Py_None;
}

static PyObject* py_conv_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int f_size, nb_filters, stride, padding, prev_layer, i_activ = RELU;
	const char *activation = "RELU";
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "activation", "prev_layer", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	stride = 1; padding = 0;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iisi", kwlist, &f_size, &nb_filters, &stride, &padding, &activation, &prev_layer))
	    return Py_None;

	if(strcmp(activation, "RELU") == 0)
		i_activ = RELU;
	else if(strcmp(activation, "LINEAR") == 0)
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC") == 0)
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX") == 0)
		i_activ = SOFTMAX;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	conv_create(prev, f_size, nb_filters, stride, padding, i_activ);
	
	printf("Added convolutional layer, L:%d\n", nb_layers-1);
	
	return Py_None;
}


static PyObject* py_pool_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int pool_size = 2, prev_layer;
	static char *kwlist[] = {"pool_size", "prev_layer", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii", kwlist, &pool_size, &prev_layer))
	    return Py_None;

	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	pool_create(prev, pool_size);
	
	printf("Added pool layer, L:%d\n", nb_layers-1);
	
	return Py_None;
}




// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int py_nb_epoch, py_control_interv = 1, py_confmat = 0;
	double py_learning_rate=0.0002, py_momentum = 0.0, py_decay = 0.0;
	static char *kwlist[] = {"nb_epoch", "learning_rate", "control_interv", "momentum", "decay", "confmat", NULL};
	
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|iddi", kwlist, &py_nb_epoch, &py_learning_rate, &py_control_interv, &py_momentum, &py_decay, &py_confmat))
	    return Py_None;

	if(py_confmat == 1)
	{
		printf("Enable confmat\n");
		enable_confmat();
	}
	
	train_network(train_set, valid_set, py_nb_epoch, py_control_interv, 
		py_learning_rate, py_momentum, py_decay);

	return Py_None;
}


static PyObject* py_forward_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	const char* pers_file_name = NULL;
	int step = -1;
	int repeat = 1;
	static char *kwlist[] = {"step", "file_name", "repeat", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|isi", kwlist, &step, &pers_file_name, &repeat))
	    return Py_None;
	
	
	forward_network(test_set, step, pers_file_name, repeat);
	
	return Py_None;
}



// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
    { "init_network", py_init_network, METH_VARARGS, "Initialize network basic sahpes and properties" },
    { "create_dataset", py_create_dataset, METH_VARARGS, "Allocate dataset structure and return a corresponding object" },
    { "dense_create", (PyCFunction)py_dense_create, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
    { "conv_create",(PyCFunction)py_conv_create, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
    { "pool_create",(PyCFunction)py_pool_create, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
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
	import_array();
	
	printf("############################################################\n\
Importing CIANNA python module V-0.1, by D.Cornu\n\
############################################################\n\n");

	PyObject *m;
    m = PyModule_Create(&CIANNA);
    if (m == NULL)
        return NULL;
	
    return m;
}








