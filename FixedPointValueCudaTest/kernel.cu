

#include "kernel.h"

#include <math_extended.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>


__global__ void kernel(compl const dp_val1[], compl const dp_val2[], compl dp_res[], size_t const size)
{
	assert(dp_val1 != nullptr);
	assert(dp_val2 != nullptr);
	assert(dp_res != nullptr);

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < size)
	{
		dp_res[x] = dp_val1[x] * dp_val2[x];
	}
}


void run_kernel(compl const dp_val1[], compl const dp_val2[], compl dp_res[], size_t const size)
{
	assert(dp_val1 != nullptr);
	assert(dp_val2 != nullptr);
	assert(dp_res != nullptr);

	size_t const block_size = 128;

	unsigned int bigX = static_cast<unsigned int>(ceil_div(size, block_size));

	unsigned int tibX = static_cast<unsigned int>(block_size);


	dim3 const big(bigX);	// blocks in grid
	dim3 const tib(tibX); // threads in block

	kernel << < big, tib >> > (dp_val1, dp_val2, dp_res, size);
}

