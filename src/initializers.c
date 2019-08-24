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
	
	/*FILE* f;
	
	f = fopen("weights.txt", "a");
	*/
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
			//tab[i] = 1.0;
			tab[i] = random_normal()*sqrt(2.0/(dim_in+dim_out));
			/*if((i%(dim_in+1+bias_padding)) == 0 && i < (dim_in+bias_padding)*(dim_in+bias_padding))
				tab[i] = 1.0;
			else
				tab[i] = 0.0;*/
		}
	}
	if(bias_padding)
		tab[size-1] = 1.0;
	/*
	fprintf(f, "\n");
	for(i = 0; i < dim_out; i++)
	{
		for(int j = 0; j < dim_in+bias_padding; j++)
			fprintf(f, "%g ", tab[j + i*(dim_in+bias_padding)]);
		fprintf(f,"\n");
	}
	fclose(f);
	*/
}



