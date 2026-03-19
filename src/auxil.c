
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


#include "prototypes.h"

// Local variables
struct timeval t_perf_eval;
struct timeval t_batch_eval, t_epoch_eval;

// Public are in "prototypes.h"


void init_timing(struct timeval* tstart)
{
    gettimeofday(tstart, NULL);
}


float ellapsed_time(struct timeval tstart)
{
    struct timeval tmp;
    long long diff;
    gettimeofday(&tmp, NULL);
    diff = (tmp.tv_usec - tstart.tv_usec);
    diff += (tmp.tv_sec - tstart.tv_sec)*1000000;
    return ((float)diff); //return in micro sec
}


void sig_handler(int signo)
{
	if(signo == SIGINT)
		printf("\n WARNING: program interrupted\n");
	//Could handle exit more gracefully by freeing everything (postponed ATM)
	exit(EXIT_SUCCESS);
}


void print_table(float* tab, int column_size, int nb_column)
{
	int i, j;
	
	for(i = 0; i < nb_column; i++)
	{
		for(j = 0; j < column_size; j++)
		{
			printf("%f ", tab[i*column_size+j]);
		}
		printf("\n");
	}
	printf("\n");
}


void print_iter_advance(network *net, int c_batch, int nb_batch, float loss, float c_perf, int is_training)
{
	int i;
	int size = net->adv_size, l_size = 0;
	
	//Must check for case where the total number of tick change
	printf("\e[?25l");
	if(is_training)
		printf("\rIter: %5d [", net->iter);
	else
		printf("\rFwd : %5d [", net->iter);
	l_size = size*((float)c_batch/nb_batch);
	for(i = 0; i < l_size; i++)
		printf("#");
	for(i = l_size; i < size; i++)
		printf("-");
	printf("] %4d / %4d | B.Loss: %.5f | B.perf.: %.0f it/s ", c_batch, nb_batch, loss, c_perf);
	printf("\e[?25h");
}


int argmax(float *tab, int size)
{
	int i;
	float max;
	int imax;

	max = *tab;
	imax = 0;
	
	for(i = 1; i < size; i++)
	{
		if(tab[i] >= max)
		{
			max = tab[i];
			imax = i;
		}
	}
	return imax;
}


int conv_argmax(float *tab, int offset, int size)
{
	int i;
	float max;
	float *off_tab;
	int imax;
	
	max = *tab;
	imax = 0;
	
	for(i = 1; i < size; i++)
	{	
		off_tab = tab + i*offset;
		if(*off_tab >= max)
		{
			max = *off_tab;
			imax = i;
		}
	}
	return imax;
}


float clip(float n, float lower, float upper) 
{
	return fmax(lower, fmin(n, upper));
}


//return a random Real value between 0 <= x < 1
double random_uniform(void)
{
	return  rand()/(double)RAND_MAX;
}


//return a real value following normal distribution with 0 mean and 1 standard deviation
double random_normal(void)
{
	// non optimized box muller normal distribution generator
	double U1, U2;
	
	U1 = rand()*(1.0/RAND_MAX);
	U2 = rand()*(1.0/RAND_MAX);
	
	return sqrt(-2.0*log(U1))*cos(two_pi*U2);
}


//Warning : the following *eval* functions are used during network training and must not be used anywhere else in the code (would lead to incorrect training metrics)
void eval_init(network *net)
{
	#ifdef CUDA
	if(net->compute_method == C_CUDA)
	{
		cuda_batch_eval_init();
		cuda_epoch_eval_init();
	}
	#endif

	if(net->perf_eval == 1)
	{
		net->fwd_perf =  (float*) calloc(net->nb_layers,sizeof(float));
		net->back_perf = (float*) calloc(net->nb_layers,sizeof(float));
		net->fwd_perf_n =  (int*) calloc(net->nb_layers,sizeof(int));
		net->back_perf_n = (int*) calloc(net->nb_layers,sizeof(int));
	
		#ifdef CUDA
		if(net->compute_method == C_CUDA)
			cuda_perf_eval_init();
		#endif
		net->perf_eval = 2;
	}
	else
		return;
}


void perf_eval_in(network *net)
{
	if(net->perf_eval == 0)
		return;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_perf_eval_in();
		#endif
	}
	else
	{
		init_timing(&t_perf_eval);
	}
}


void batch_eval_in(network *net)
{
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_batch_eval_in();
		#endif
	}
	else
	{
		init_timing(&t_batch_eval);
	}
}


void epoch_eval_in(network *net)
{
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		cuda_epoch_eval_in();
		#endif
	}
	else
	{
		init_timing(&t_epoch_eval);
	}
}


void perf_eval_out(network *net, int layer_id, float *vect, int *n_vect)
{
	float time = 0.0f;
	if(net->perf_eval == 0)
		return;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		time = cuda_perf_eval_out(); //in micro sec
		#endif
	}
	else
	{
		time = ellapsed_time(t_perf_eval);
	}
	
	if(n_vect[layer_id] < 999)
	{
		vect[layer_id] += time/net->batch_size;
		n_vect[layer_id] += 1;
	}
}


float batch_eval_out(network *net)
{
	float time = 0.0f;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		time = cuda_batch_eval_out()/1000000.0f;
		#endif
	}
	else
	{
		time = ellapsed_time(t_batch_eval)/1000000.0f;
	}
	return time;
}


float epoch_eval_out(network *net)
{
	float time = 0.0f;
	if(net->compute_method == C_CUDA)
	{
		#ifdef CUDA
		time = cuda_epoch_eval_out()/1000000.0f;
		#endif
	}
	else
	{
		time = ellapsed_time(t_epoch_eval)/1000000.0f;
	}
	return time;
}


void perf_eval_display(network *net)
{
	int i;
	float *fwd_time, *back_time, *cumul_time;
	float total_fwd = 0.0f, total_back = 0.0f, total_cumul = 0.0f;
	char layer_type_char = '-';
	
	fwd_time   = (float*) calloc(net->nb_layers, sizeof(float));
	back_time  = (float*) calloc(net->nb_layers, sizeof(float));
	cumul_time = (float*) calloc(net->nb_layers, sizeof(float));
	
	if (net->perf_eval == 0)
		return;
	
	printf("\nTotal Net. nb weights: %lld \nTotal Network RAM/VRAM usage : %d MB\n", 
		net->total_nb_param, (int)(net->memory_footprint/1000000));
	printf("(without datasets, and prop. to batch_size)\n");
	
	for(i = 0; i < net->nb_layers; i++)
	{
		fwd_time[i] = net->fwd_perf[i]/net->fwd_perf_n[i];
		total_fwd += fwd_time[i];
		back_time[i] = net->back_perf[i]/net->back_perf_n[i];
		total_back += back_time[i];
		cumul_time[i] = fwd_time[i] + back_time[i];
		total_cumul += cumul_time[i];
		if(net->fwd_perf_n[i] == 0)
			printf("\n WARNING: some layers were not benchmarked\n");
	}
	
	printf("\n     Layer  Type       Forward             Backprop             Cumulated\n");
	printf("       N     T      [µs]  /  [%%]         [µs]  /  [%%]         [µs]  /  [%%]\n");	
	printf("  -------------------------------------------------------------------\n");
	for(i = 0; i < net->nb_layers; i++)
	{
		switch(net->net_layers[i]->type)
		{
			case CONV:
				layer_type_char = 'C';
				break;
			
			case POOL:
				layer_type_char = 'P';
				break;
		
			case NORM:
				layer_type_char = 'N';
				break;
			
			case LRN:
				layer_type_char = 'L';
				break;
			
			case MERGE:
				layer_type_char = 'M';
				break;
		
			case DENSE:
				layer_type_char = 'D';
				break;
		}
		printf("   %5d     %c   %8.1f / %4.1f      %8.1f / %4.1f      %8.1f / %4.1f\n", i+1, layer_type_char,
			fwd_time[i], fwd_time[i]/total_fwd*100.0, back_time[i], back_time[i]/total_back*100.0,
			cumul_time[i], cumul_time[i]/total_cumul*100.0);
	}
	printf("  -------------------------------------------------------------------\n");
	printf("   Total         %8.1f µs          %8.1f µs          %8.1f µs       \n\n", total_fwd, total_back, total_cumul);
	
	free(fwd_time);
	free(back_time);
	free(cumul_time);
}


void print_architecture_tex(network *net, const char *path, const char *file_name,
	int l_size, int l_in_size, int l_f_size, int l_out_size, int l_stride, int l_padding, 
	int l_in_padding, int l_activation, int l_bias, int l_dropout, int l_param_count)
{
	int i;
	int type_count[6] = {0,0,0,0,0,0};
	FILE* f_tex = NULL;
	char full_path_name[200];
	char command[600];
	char activ_str[20];
	layer* c_l = NULL;
	struct stat st = {0};
	conv_param* c_param = NULL;
	pool_param* p_param = NULL;
	norm_param* n_param = NULL;
	lrn_param* ln_param = NULL;
	merge_param* m_param = NULL;
	
	if(stat(path, &st) == -1)
    	mkdir(path, 0700);
	sprintf(full_path_name, "%s%s.tex", path, file_name);
	
	f_tex = fopen(full_path_name, "w+");
	
	fprintf(f_tex, "\
\\documentclass[border=2pt]{standalone}\n\
\\usepackage[utf8]{inputenc}\n\
\\usepackage{array}\n\
\\renewcommand{\\arraystretch}{1.1}\n\
\\begin{document}\n\
\\centering\n\
\\begin{tabular}{");

	fprintf(f_tex, "p{0.6cm}");
	fprintf(f_tex, "p{1.4cm}");
	if(l_in_size) fprintf(f_tex, "p{2.0cm}<{\\centering}");
	if(l_size) fprintf(f_tex, "p{1.6cm}<{\\centering}");
	if(l_f_size) fprintf(f_tex, "p{2.0cm}<{\\centering}");
	if(l_stride) fprintf(f_tex, "p{1.2cm}<{\\centering}");
	if(l_padding) fprintf(f_tex, "p{1.2cm}<{\\centering}");
	if(l_in_padding) fprintf(f_tex, "p{1.2cm}<{\\centering}");
	if(l_out_size) fprintf(f_tex, "p{2.0cm}<{\\centering}");
	if(l_activation) fprintf(f_tex, "p{1.2cm}");
	if(l_bias) fprintf(f_tex, "p{0.8cm}<{\\centering}");
	if(l_dropout) fprintf(f_tex, "p{1.2cm}<{\\centering}");
	if(l_param_count) fprintf(f_tex, "p{1.4cm}<{\\centering}");
		
	fprintf(f_tex,"}\n\
\\hline\\noalign{\\smallskip}\n");

	fprintf(f_tex, "Id. & Type ");
	if(l_in_size) fprintf(f_tex, "& In. size ");
	if(l_size) fprintf(f_tex, "& N. filters ");
	if(l_f_size) fprintf(f_tex, "& F. size ");
	if(l_stride) fprintf(f_tex, "& Stride ");
	if(l_padding) fprintf(f_tex, "& Padding ");
	if(l_in_padding) fprintf(f_tex, "& Intern. Pad. ");
	if(l_out_size) fprintf(f_tex, "& Out. size ");
	if(l_activation) fprintf(f_tex, "& Activ. ");
	if(l_bias) fprintf(f_tex, "& Bias ");
	if(l_dropout) fprintf(f_tex, "& Dropout ");
	if(l_param_count) fprintf(f_tex, "& N. param. ");
	
	fprintf(f_tex, "\\\\\n\
\\hline\\noalign{\\smallskip}\n");

	for(i = 0; i < net->nb_layers; i++)
	{
		fprintf(f_tex,"%d ", i+1);
		c_l = net->net_layers[i];
		switch(c_l->type)
		{
			case CONV:
				c_param = (conv_param*)c_l->param;
				type_count[0] += 1;
				fprintf(f_tex, "& Conv\\_%d ", type_count[0]);
				if(l_in_size) fprintf(f_tex, "& %dx%dx%d ", c_l->prev_dim[0], c_l->prev_dim[1], c_l->prev_dim[2]);
				if(l_size) fprintf(f_tex, "& %d ", c_l->output_dim[3]);
				if(l_f_size) fprintf(f_tex, "& %dx%dx%d ", c_param->f_size[0], c_param->f_size[1], c_param->f_size[2]);
				if(l_stride) fprintf(f_tex, "& %d:%d:%d ", c_param->stride[0], c_param->stride[1], c_param->stride[2]);
				if(l_padding) fprintf(f_tex, "& %d:%d:%d ", c_param->padding[0], c_param->padding[1], c_param->padding[2]);
				if(l_in_padding) fprintf(f_tex, "& %d:%d:%d ", c_param->int_padding[0], c_param->int_padding[1], c_param->int_padding[2]);
				if(l_out_size) fprintf(f_tex, "& %dx%dx%d ", c_l->output_dim[0], c_l->output_dim[1], c_l->output_dim[2]);
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& %0.2f ", c_l->bias_value);
				if(l_dropout) fprintf(f_tex, "& %d\\%% ", (int)(c_l->dropout_rate*100.0f));
				if(l_param_count) fprintf(f_tex, "& %ld ", c_l->nb_params);
				break;
			case POOL:
				p_param = (pool_param*)c_l->param;
				type_count[1] += 1;
				fprintf(f_tex, "& Pool\\_%d ", type_count[1]);
				if(l_in_size) fprintf(f_tex, "& %dx%dx%d ", c_l->prev_dim[0], c_l->prev_dim[1], c_l->prev_dim[2]);
				if(l_size) fprintf(f_tex, "& ");
				if(l_f_size) fprintf(f_tex, "& %dx%dx%d ", p_param->p_size[0], p_param->p_size[1], p_param->p_size[2]);
				if(l_stride) fprintf(f_tex, "& %d:%d:%d ", p_param->stride[0], p_param->stride[1], p_param->stride[2]);
				if(l_padding) fprintf(f_tex, "& %d:%d:%d ", p_param->padding[0], p_param->padding[1], p_param->padding[2]);
				if(l_in_padding) fprintf(f_tex, "& ");
				if(l_out_size) fprintf(f_tex, "& %dx%dx%d ", c_l->output_dim[0], c_l->output_dim[1], c_l->output_dim[2]);
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& ");
				if(l_dropout) fprintf(f_tex, "& %d\\%% ", (int)(c_l->dropout_rate*100.0f));
				if(l_param_count) fprintf(f_tex, "& ");
				break;
			case NORM:
				n_param = (norm_param*)c_l->param;
				type_count[2] += 1;
				fprintf(f_tex, "& Norm\\_%d ", type_count[2]);
				if(l_in_size)
				{
					if(c_l->output_type == SPATIAL)
						fprintf(f_tex, "& %dx%dx%d ", c_l->prev_dim[0], c_l->prev_dim[1], c_l->prev_dim[2]);
					else
					{
						//empty for now
					}
				}
				if(l_size) fprintf(f_tex, "& N.Gr. %d ", n_param->nb_group);
				if(l_f_size) fprintf(f_tex, "& Gr.Size %d ", n_param->group_size);
				if(l_stride) fprintf(f_tex, "& ");
				if(l_padding) fprintf(f_tex, "& Off %d ", n_param->set_off);
				if(l_in_padding) fprintf(f_tex, "& ");
				if(l_out_size)
				{
					if(c_l->output_type == SPATIAL)
						fprintf(f_tex, "& %dx%dx%d ", c_l->output_dim[0], c_l->output_dim[1], c_l->output_dim[2]);
					else
					{
						//empty for now
					}
				}
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& ");
				if(l_dropout) fprintf(f_tex, "& ");
				if(l_param_count) fprintf(f_tex, "& ");
				break;
				
			case LRN:
				ln_param = (lrn_param*)c_l->param;
				type_count[3] += 1;
				fprintf(f_tex, "& LRN\\_%d ", type_count[3]);
				if(l_in_size)
				{
					if(c_l->output_type == SPATIAL)
						fprintf(f_tex, "& %dx%dx%d ", c_l->prev_dim[0], c_l->prev_dim[1], c_l->prev_dim[2]);
					else
					{
						//empty for now
					}
				}
				if(l_size) fprintf(f_tex, "& ch\\_range: %d", ln_param->range);
				if(l_f_size) fprintf(f_tex, "& ");
				if(l_stride) fprintf(f_tex, "& ");
				if(l_padding) fprintf(f_tex, "& ");
				if(l_in_padding) fprintf(f_tex, "& ");
				if(l_out_size)
				{
					if(c_l->output_type == SPATIAL)
						fprintf(f_tex, "& %dx%dx%d ", c_l->output_dim[0], c_l->output_dim[1], c_l->output_dim[2]);
					else
					{
						//empty for now
					}
				}
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& ");
				if(l_dropout) fprintf(f_tex, "& ");
				if(l_param_count) fprintf(f_tex, "& ");
				
				break;
				
			case DENSE:
				type_count[4] += 1;
				fprintf(f_tex, "& Dense\\_%d ", type_count[4]);
				if(l_in_size) fprintf(f_tex, "& %d", 
					c_l->prev_dim[0]*c_l->prev_dim[1]*c_l->prev_dim[2]);
				if(l_size) fprintf(f_tex, "& %d ", c_l->output_dim[3]);
				if(l_f_size) fprintf(f_tex, "& ");
				if(l_stride) fprintf(f_tex, "& ");
				if(l_padding) fprintf(f_tex, "& ");
				if(l_in_padding) fprintf(f_tex, "& ");
				if(l_out_size) fprintf(f_tex, "& %d ", c_l->output_dim[3]);
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& %0.2f ", c_l->bias_value);
				if(l_dropout) fprintf(f_tex, "& %d\\%% ", (int)(c_l->dropout_rate*100.0f));
				if(l_param_count) fprintf(f_tex, "& %ld ", c_l->nb_params);
				break;
				
			case MERGE:
				m_param = (merge_param*)c_l->param;
				type_count[5] += 1;
				fprintf(f_tex, "& MERGE\\_%d ", type_count[5]);
				if(l_in_size) fprintf(f_tex, "L:%d / L:%d ", m_param->previous_id_a, m_param->previous_id_b);
				if(l_size) fprintf(f_tex, "& %d ", c_l->output_dim[3]);
				if(l_f_size) fprintf(f_tex, "& ");
				if(l_stride) fprintf(f_tex, "& ");
				if(l_padding) fprintf(f_tex, "& ");
				if(l_in_padding) fprintf(f_tex, "& ");
				if(l_out_size) fprintf(f_tex, "& %dx%dx%d ", c_l->output_dim[0], c_l->output_dim[1], c_l->output_dim[2]);
				if(l_activation) {fill_string_activ_param(c_l, activ_str,1); fprintf(f_tex, "& %s ", activ_str);}
				if(l_bias) fprintf(f_tex, "& ");
				if(l_dropout) fprintf(f_tex, "& ");
				if(l_param_count) fprintf(f_tex, "& ");
				break;
				
			default:
				printf("\n ERROR: Unrecognized layer type in architechture tex\n");
				exit(EXIT_FAILURE);
				break;
		}
		fprintf(f_tex, "\\\\\n");
	}

	fprintf(f_tex, "\n\
\\hline\\noalign{\\smallskip}\n\
\\end{tabular}\n\
\\end{document}\n");
	
	fclose(f_tex);
	
	sprintf(command, "pdflatex --interaction=batchmode -output-directory=%s %s", path, full_path_name);
	system(command);
	
}


