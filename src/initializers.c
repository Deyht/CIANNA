
/*
	Copyright (C) 2020 David Cornu
	for the Convolutional Interactive Artificial 
	Neural Network by/for Astrophysicists (CIANNA) Code
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


//return a random Real value between 0 <= x < 1
real random_uniform(void)
{
	return  rand()/(real)RAND_MAX;
}

//return a real value following normal distribution with 0 mean and 1 standart deviation
real random_normal(void)
{
	// non optimized box muller normal distribution generator
	double U1, U2;
	
	U1 = rand()*(1.0/RAND_MAX);
	U2 = rand()*(1.0/RAND_MAX);
	
	return sqrt(-2.0*log(U1))*cos(two_pi*U2);
}




void xavier_normal(real *tab, int dim_in, int dim_out, int bias_padding)
{
	int i;
	int size;
	int limit;
	
	size = dim_in*dim_out;
	if(bias_padding)
	{
		size += dim_out;
		limit = size - 1;
	}
	else
		limit = size;
	for(i = 0; i < limit; i++)
	{
		if(bias_padding && (i+1) % (dim_in+bias_padding) == 0)
			tab[i] = 0.0;
		else
		{
			tab[i] = random_normal()*sqrt(2.0/(dim_in+dim_out));
		}
	}
	if(bias_padding)
		tab[size-1] = 1.0;

}



