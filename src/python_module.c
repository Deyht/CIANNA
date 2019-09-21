#include <Python.h>
#include <numpy/arrayobject.h>

#include <string.h>
#include "prototypes.h"


// Structures or object related
//############################################################

typedef struct {
    PyObject_HEAD
    Dataset data;
} py_dataset;

static PyTypeObject py_dataset_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "dataset",
    .tp_doc = "Custom dataset handler for notwork",
    .tp_basicsize = sizeof(py_dataset),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};


// Network paramaremeter and data management functions
//############################################################

static PyObject* py_init_network(PyObject* self, PyObject* args)
{
	PyArrayObject *py_dims;
	int i;
	int dims[3] = {1,1,1}, out_dim, b_size, comp_int = C_CUDA;
	char comp_meth[10], string_comp[10];
	
	b_size = 10;
	sprintf(comp_meth,"C_CUDA");
	
	if(!PyArg_ParseTuple(args, "Oi|is", &py_dims, &out_dim, &b_size, &comp_meth))
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
	
    init_network(dims, out_dim, b_size, comp_int);
    
	printf("Network have been initialized with : \nInput dimensions: %dx%dx%d \nOutput dimension: %d \nBatch size: %d \nUsing %s compute methode\n\n", dims[0], dims[1], dims[2], output_dim, batch_size, string_comp);
	
    return Py_None;
}


static PyObject* py_create_dataset(PyObject* self, PyObject* args)
{
	int i, j, k;
	py_dataset data;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size;

	if(!PyArg_ParseTuple(args, "iOO", &size, &py_data, &py_target))
	    return Py_None;
	
	printf("Creating dataset with size %d ...\n", size);
	data.data = create_dataset(size);
	
	if(py_data != NULL && py_target != NULL)
	{
		for(i = 0; i < data.data.nb_batch; i++)
			for(j = 0; j < batch_size; j++)
			{
				for(k = 0; k < input_dim; k++)
					data.data.input[i][j*(input_dim+1) + k] = 
						*(real*)(py_data->data + i * (batch_size*py_data->strides[1]) 
						+ j * py_data->strides[1] + k*py_data->strides[0]);
				data.data.input[i][j*(input_dim+1) + input_dim] = 0.1;
			}
		
		for(i = 0; i < data.data.nb_batch; i++)
			for(j = 0; j < batch_size; j++)
				for(k = 0; k < output_dim; k++)
					data.data.target[i][j*output_dim + k] = 
						*(real*)(py_target->data + i * (batch_size*py_target->strides[1]) + j * py_target->strides[1] + k*py_target->strides[0]);
	}
	#ifdef CUDA
	if(compute_method == C_CUDA)
	{
		printf("Converting dataset to GPU device (CUDA)\n");
		cuda_convert_dataset(data.data);
	}
	#endif
	
	printf("\n");
	return Py_BuildValue("O", &data);
}



// Layers functions
//############################################################
static PyObject* py_dense_create(PyObject* self, PyObject *args,
                                    PyObject *kwargs)
{	
	int nb_neurons, prev_layer, i_activ = RELU;
	char activation[20] = "RELU";
	static char *kwlist[] = {"nb_neurons", "activation", "prev_layer", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "i|si", kwlist, &nb_neurons, &activation, &prev_layer))
	    return Py_None;

	if(strcmp(activation, "RELU"))
		i_activ = RELU;
	else if(strcmp(activation, "LINEAR"))
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC"))
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX"))
		i_activ = SOFTMAX;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	dense_create(prev, nb_neurons, i_activ);
	
	printf("Added dense layer, number: %d\n", nb_layers-1);
	
	return Py_None;
}

static PyObject* py_conv_create(PyObject* self, PyObject *args,
                                    PyObject *kwargs)
{	
	int f_size, nb_filters, stride, padding, prev_layer, i_activ = RELU;
	char activation[20] = "RELU";
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "activation", "prev_layer", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	stride = 1; padding = 0;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iisi", kwlist, &f_size, &nb_filters, &stride, &padding, &activation, &prev_layer))
	    return Py_None;

	if(strcmp(activation, "RELU"))
		i_activ = RELU;
	else if(strcmp(activation, "LINEAR"))
		i_activ = LINEAR;
	else if(strcmp(activation, "LOGISTIC"))
		i_activ = LOGISTIC;
	else if(strcmp(activation, "SOFTMAX"))
		i_activ = SOFTMAX;
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	conv_create(prev, f_size, nb_filters, stride, padding, i_activ);
	
	printf("Added convolutional layer, number: %d\n", nb_layers-1);
	
	return Py_None;
}


static PyObject* py_pool_create(PyObject* self, PyObject *args,
                                    PyObject *kwargs)
{	
	int pool_size = 2, prev_layer;
	static char *kwlist[] = {"pool_size", "prev_layer", NULL};
	layer* prev;
	
	prev_layer = nb_layers - 1;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|is", kwlist, &pool_size, &prev_layer))
	    return Py_None;

	if(prev_layer < 0)
		prev = NULL;
	else
		prev = net_layers[prev_layer];
		
	pool_create(prev, pool_size);
	
	printf("Added pool layer, number: %d\n", nb_layers-1);
	
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
	if (PyType_Ready(&py_dataset_type) < 0)
    	return NULL;
    	
    m = PyModule_Create(&CIANNA);
    if (m == NULL)
        return NULL;
        
	Py_INCREF(&py_dataset_type);
	if (PyModule_AddObject(m, "dataset", (PyObject *) &py_dataset_type) < 0) 
	{
		Py_DECREF(&py_dataset_type);
		Py_DECREF(m);
		return NULL;
	}
	
    return m;
}








