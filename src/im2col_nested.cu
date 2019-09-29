
#include "prototypes.h"

__global__ void imagesize_loop_im2col(int size, real* output, real* input, int stride, int w_size, int nb_w, int flat_f_size, int f_size, int padding)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	int x, y, w, h;
	w = i % w_size + padding;
	h = i / w_size + padding;
	for(x = 0; x < f_size; x += stride)
		for(y = 0; y < f_size; y+= stride)
			if((w-x) >= 0 && (h-y) >= 0 && (w-x) < nb_w && (h-y) < nb_w)
				output[(w-x) * flat_f_size + (h-y) * nb_w * flat_f_size + x + y*f_size] = input[i];
}

__global__ void im2col_kernel_nested(real* output, real* input, int image_size, int flat_image_size, int stride, int padding, int depth, int batch_size, int f_size, int flat_f_size, int w_size, int nb_area_w, int bias)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int d = blockIdx.y*blockDim.y + threadIdx.y;
//	int z = blockIdx.z*blockDim.z + threadIdx.z;
	
	if( i < batch_size)
	{
		input += i*(image_size * depth + bias);
		output += i*(flat_image_size);
		
		if(d < depth)
		{
			input += d * image_size;
			output += d * f_size*f_size;
			
			imagesize_loop_im2col<<<1,64>>>(image_size, output, input, stride, w_size, nb_area_w, flat_f_size, f_size, padding);
		}
	}
}
