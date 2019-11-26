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
	int dims[3] = {1,1,1}, out_dim, b_size, comp_int = C_CUDA, network_id = nb_networks, dynamic_load = 0;
	char string_comp[10];
	const char *comp_meth = "C_CUDA";
	static char *kwlist[] = {"dims", "out_dim", "b_size", "comp_meth", "network_id", "dynamic_load", NULL};

	
	b_size = 10;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|isii", kwlist, &py_dims, &out_dim, &b_size, &comp_meth, &network_id, &dynamic_load))
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
	
    init_network(network_id, dims, out_dim, b_size, comp_int, dynamic_load);
    
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
	double bias = 0.1;
	PyArrayObject *py_data = NULL, *py_target = NULL;
	int size, flat = 0, on_gpu = 1;
	int network_id = nb_networks-1;
	static char *kwlist[] = {"dataset", "size", "input", "target", "bias", "flat", "network_id", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOd|iii", kwlist, &dataset_type, &size, &py_data, &py_target, &bias, &flat, &network_id))
	    return Py_None;
	    
	if(networks[network_id]->dynamic_load)
		on_gpu = 0;
	
	
	if(strcmp(dataset_type,"TRAIN") == 0)
	{
		printf("Setting training set\n");
		data = &networks[network_id]->train;
	}
	else if(strcmp(dataset_type,"VALID") == 0)
	{
		printf("Setting valid set\n");
		data = &networks[network_id]->valid;
	}
	else if(strcmp(dataset_type,"TEST") == 0)
	{
		printf("Setting testing test\n");
		data = &networks[network_id]->test;
	}
	
	
	printf("input dim :%d,", networks[network_id]->input_dim);
	*data = create_dataset(networks[network_id], size, bias);
	
	printf("Creating dataset with size %d (nb_batch = %d) ... ", data->size, data->nb_batch);
	
	
	if(py_data != NULL && py_target != NULL)
	{
		for(i = 0; i < data->nb_batch; i++)
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
	data->input[i][j*(networks[network_id]->input_dim+1) + k * 
		networks[network_id]->input_height*networks[network_id]->input_width + l 
		* networks[network_id]->input_width + m] 
		= *(double*)(py_data->data + (i*networks[network_id]->batch_size 
		* networks[network_id]->input_depth*networks[network_id]->input_height 
		+ j*networks[network_id]->input_depth*networks[network_id]->input_height 
		+ k*networks[network_id]->input_height + l)*py_data->strides[0] + m*py_data->strides[1]);
							else
	data->input[i][j*(networks[network_id]->input_dim+1) + k * networks[network_id]->input_height
		* networks[network_id]->input_width + l * networks[network_id]->input_width + m] 
		= *(double*)(py_data->data + (i * networks[network_id]->batch_size + j) 
		* py_data->strides[0] + (k*networks[network_id]->input_height 
		* networks[network_id]->input_width + l * networks[network_id]->input_width + m) 
		* py_data->strides[1]);	
						}
				}
			}
		for(i = 0; i < data->nb_batch; i++)
			for(j = 0; j < networks[network_id]->batch_size; j++)
			{
				if(i*networks[network_id]->batch_size + j >= data->size)
					continue;
				for(k = 0; k < networks[network_id]->output_dim; k++)
					data->target[i][j*networks[network_id]->output_dim + k] 
					= *(double*)(py_target->data + i * (networks[network_id]->batch_size 
					* py_target->strides[0]) + j * py_target->strides[0] + k 
					* py_target->strides[1]);
			}
	}
	printf("Done !\n");
	
	#ifdef CUDA
	if(networks[network_id]->compute_method == C_CUDA && on_gpu)
	{
		printf("Converting dataset to GPU device (CUDA)\n");
		cuda_convert_dataset(networks[network_id], data);
	}
	#endif
	printf("\n");
	
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
	
	printf("prev : %d\n", prev_layer);
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	dense_create(networks[network_id], prev, nb_neurons, i_activ, drop_rate, NULL);
	
	printf("Added dense layer, L:%d\n", networks[network_id]->nb_layers-1);
	
	return Py_None;
}

static PyObject* py_conv_create(PyObject* self, PyObject *args, PyObject *kwargs)
{	
	int f_size, nb_filters, stride, padding, prev_layer = -1, i_activ = RELU, network_id = nb_networks-1;
	const char *activation = "RELU";
	static char *kwlist[] = {"f_size", "nb_filters", "stride", "padding", "activation", "prev_layer", "network", NULL};
	layer* prev;
	
	stride = 1; padding = 0;
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iisii", kwlist, &f_size, &nb_filters, &stride, &padding, &activation, &prev_layer, &network_id))
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
	
	if(prev_layer < 0)
		prev = NULL;
	else
		prev = networks[network_id]->net_layers[prev_layer];
		
	conv_create(networks[network_id], prev, f_size, nb_filters, stride, padding, i_activ, NULL);
	
	printf("Added convolutional layer, L:%d\n", networks[network_id]->nb_layers-1);
	
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
	
	printf("Added pool layer, L:%d\n", networks[network_id]->nb_layers-1);
	
	return Py_None;
}

static PyObject* py_load_network(PyObject* self, PyObject* args)
{
	char* file = "relative_path_to_the_save_file_location_which_must_be_long_enough";
	int epoch, network_id = nb_networks-1;

	if(!PyArg_ParseTuple(args, "si|i", &file, &epoch, &nb_networks-1))
	    return Py_None;
	    
	load_network(networks[network_id], file, epoch);
	
	return Py_None;
}


// Network global functions
//############################################################

static PyObject* py_train_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int py_nb_epoch, py_control_interv = 1, py_confmat = 0, save_net = 0, network_id = nb_networks-1, shuffle_gpu = 1;
	double py_learning_rate=0.02, py_momentum = 0.0, py_decay = 0.0, py_end_learning_rate = 0.0;
	static char *kwlist[] = {"nb_epoch", "learning_rate", "end_learning_rate", "control_interv", "momentum", "decay", "confmat", "save_each", "network", "shuffle_gpu", NULL};
	
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "id|diddiiii", kwlist, &py_nb_epoch, &py_learning_rate, &py_end_learning_rate, &py_control_interv, &py_momentum, &py_decay, &py_confmat, &save_net, &network_id, &shuffle_gpu))
	    return Py_None;
	
	train_network(networks[network_id], py_nb_epoch, py_control_interv, 
		py_learning_rate, py_end_learning_rate, py_momentum, py_decay, py_confmat, save_net, shuffle_gpu);

	return Py_None;
}


static PyObject* py_forward_network(PyObject* self, PyObject *args, PyObject *kwargs)
{
	int step = -1, repeat = 1, network_id = nb_networks-1;
	static char *kwlist[] = {"step", "repeat","network_id", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii", kwlist, &step, &repeat, &network_id))
	    return Py_None;
	
	if(step == -1)
		step = networks[network_id]->epoch;
	
	forward_testset(networks[network_id], step,  repeat);
	
	return Py_None;
}



// Module creation functions
//############################################################

static PyMethodDef CIANNAMethods[] = {
    { "init_network", (PyCFunction)py_init_network, METH_VARARGS | METH_KEYWORDS, "Initialize network basic sahpes and properties" },
    { "create_dataset", (PyCFunction)py_create_dataset, METH_VARARGS | METH_KEYWORDS, "Allocate dataset structure and return a corresponding object" },
    { "dense_create", (PyCFunction)py_dense_create, METH_VARARGS | METH_KEYWORDS, "Add a dense layer to the network" },
    { "conv_create",(PyCFunction)py_conv_create, METH_VARARGS | METH_KEYWORDS, "Add a convolutional layer to the network" },
    { "pool_create",(PyCFunction)py_pool_create, METH_VARARGS | METH_KEYWORDS, "Add a pooling layer to the network" },
    { "train_network", (PyCFunction)py_train_network, METH_VARARGS | METH_KEYWORDS, "Launch a training phase with the specified arguments" },
    { "forward_network", (PyCFunction)py_forward_network, METH_VARARGS | METH_KEYWORDS, "Apply the trained network to the test set and save results" },
    { "load_network", py_load_network, METH_VARARGS, "Load a previous network structure pre-trained from a file" },
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
Importing CIANNA python module V-p.0.3, by D.Cornu\n\
############################################################\n\n");

	PyObject *m;
    m = PyModule_Create(&CIANNA);
    if (m == NULL)
        return NULL;
	
    return m;
}








