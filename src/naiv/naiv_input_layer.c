
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


#include "../prototypes.h"

// Local variables

// Public are in "prototypes.h"

// Private prototypes
void naiv_forward_input_layer(layer *current);
void naiv_backward_input_layer(layer *current);


void naiv_forward_input_layer(layer *current)
{
	//Set output to current network dynamic input of the corresponding branch
	current->output = current->c_network->dyn_input[current->activation_type];
}


void naiv_backward_input_layer(layer *current)
{
	//Do nothing on purpose
	//This place might be used to implement saliency map from gradient backprop to input	
}


void naiv_input_define(layer *current)
{
	current->forward = naiv_forward_input_layer;
	current->backprop = naiv_backward_input_layer;
}




