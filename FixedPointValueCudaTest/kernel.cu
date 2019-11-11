

#include "kernel.h"

#include <math_extended.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>



template<typename T1, typename T2>
__device__ bool EXPECT_EQ(T1 const& left, T2 const& right)
{
	return left == right;
}



__device__ bool test_Add_simple()
{
	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("0.6000000'60000000'50000000");
	Fixed const result = val1 + val2;

	return EXPECT_EQ(result, "0.9000000'80000000'60000000");
}

__device__ bool test_Add_negative1() {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 + val1;

	return EXPECT_EQ(res, val1);
}

__device__ bool test_Add_negative2() {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val1 + val2;

	return EXPECT_EQ(res, val1);
}

__device__ bool test_Add_zero() {

	Fixed const val1 = 1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 + val1;

	return EXPECT_EQ(res, val1);
}

// Sub
__device__ bool test_Sub_simple() {

	Fixed const val1("0.6000000'30000000'20000000");
	Fixed const val2("0.3000000'30000000'30000000");
	Fixed const result = val1 - val2;

	return EXPECT_EQ(result, "0.2ffffff'ffffffff'f0000000");
}

__device__ bool test_Sub_negative1() {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 - val1;

	return EXPECT_EQ(res, 1.0);
}

__device__ bool test_Sub_negative2() {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val1 - val2;

	return EXPECT_EQ(res, -1.0);
}

__device__ bool test_Sub_zero() {

	Fixed const val1 = 1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 - val1;

	return EXPECT_EQ(res, -1.0);
}

// mul
__device__ bool test_Mul_simple() {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("2.0000000'00000000'00000000");
	Fixed const result = val1 * val2;

	return EXPECT_EQ(result, "0.6000000'40000000'20000000");
}

__device__ bool test_Mul_negative1() {

	Fixed const val1(-0.3);
	Fixed const val2(2.0);
	Fixed const result = val1 * val2;

	return EXPECT_EQ(result, -0.6);
}

__device__ bool test_Mul_negative2() {

	Fixed const val1(0.3);
	Fixed const val2(-2.0);
	Fixed const result = val1 * val2;

	return EXPECT_EQ(result, -0.6);
}

__device__ bool test_Mul_negative3() {

	Fixed const val1(-0.3);
	Fixed const val2(-2.0);
	Fixed const result = val1 * val2;

	return EXPECT_EQ(result, 0.6);
}


// shift
__device__ bool test_LeftShift_simple() {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const result = val1 << 4;

	return EXPECT_EQ(result, "3.0000002'00000001'00000000");
}

__device__ bool test_RightShift_simple() {

	Fixed const val1("3.0000002'00000001'00000000");
	Fixed const result = val1 >> 4;

	return EXPECT_EQ(result, "0.3000000'20000000'10000000");
}

// less
__device__ bool test_Less_simple()
{

	Fixed const val1("1.2");
	Fixed const val2("1.3");
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	return EXPECT_EQ(result1, true);
	return EXPECT_EQ(result2, false);
}

__device__ bool test_Less_negative1()
{

	Fixed const val1(1.2);
	Fixed const val2(-1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	return EXPECT_EQ(result1, false);
	return EXPECT_EQ(result2, true);
}

__device__ bool test_Less_negative2()
{

	Fixed const val1(-1.2);
	Fixed const val2(1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	return EXPECT_EQ(result1, true);
	return EXPECT_EQ(result2, false);
}


__device__ bool test_Less_negative3()
{

	Fixed const val1(-1.2);
	Fixed const val2(-1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	return EXPECT_EQ(result1, false);
	return EXPECT_EQ(result2, true);
}

//equal
__device__ bool test_Equal_simple()
{

	Fixed const val1("0.6000000'40000000'20000000");
	Fixed const val2("0.6000000'40000000'20000000");
	bool const result1 = val1 == val2;

	return EXPECT_EQ(result1, true);
}


__device__ bool testSmallValue_simple()
{
	Fixed const val1 = 0.000000000000005;
	Fixed const val2 = 0.000000000000000;

	Fixed const res = val2 + val1;

	return EXPECT_EQ(res, 0.000000000000005);
}



__global__ void kernel_run_tests(bool dp_res[])
{
	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0)
	{
		dp_res[0] = test_Add_simple();
		dp_res[1] = test_Add_negative1();
		dp_res[2] = test_Add_negative2();
		dp_res[3] = test_Add_zero();
		dp_res[4] = test_Sub_simple();
		dp_res[5] = test_Sub_negative1();
		dp_res[6] = test_Sub_negative2();
		dp_res[7] = test_Sub_zero();
		dp_res[8] = test_Mul_simple();
		dp_res[9] = test_Mul_negative1();
		dp_res[10] = test_Mul_negative2();
		dp_res[11] = test_Mul_negative3();
		dp_res[12] = test_LeftShift_simple();
		dp_res[13] = test_RightShift_simple();
		dp_res[14] = test_Less_simple();
		dp_res[15] = test_Less_negative1();
		dp_res[16] = test_Less_negative2();
		dp_res[17] = test_Less_negative3();
		dp_res[18] = test_Equal_simple();
		dp_res[19] = testSmallValue_simple();
	}
}




void run_tests(bool dp_res[])
{
	size_t const block_size = 128;

	unsigned int bigX = static_cast<unsigned int>(1);

	unsigned int tibX = static_cast<unsigned int>(block_size);


	dim3 const big(bigX);	// blocks in grid
	dim3 const tib(tibX); // threads in block

	kernel_run_tests << < big, tib >> > (dp_res);
}