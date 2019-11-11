#include "gtest/gtest.h"

#include <FixedPointValue.h>

#include <iostream> 

using namespace std;

using Fixed = FixedPointValue<3>;

// Add
TEST(Add, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("0.6000000'60000000'50000000");
	Fixed const result = val1 + val2;

	EXPECT_EQ(result,"0.9000000'80000000'60000000");
}

TEST(Add, negative1) {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 + val1;

	EXPECT_EQ(res, val1);
}

TEST(Add, negative2) {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val1 + val2;

	EXPECT_EQ(res, val1);
}

TEST(Add, zero) {

	Fixed const val1 = 1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 + val1;

	EXPECT_EQ(res, val1);
}

// Sub
TEST(Sub, simple) {

	Fixed const val1("0.6000000'30000000'20000000");
	Fixed const val2("0.3000000'30000000'30000000");
	Fixed const result = val1 - val2;

	EXPECT_EQ(result,"0.2ffffff'ffffffff'f0000000");
}

TEST(Sub, negative1) {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 - val1;

	EXPECT_EQ(res, 1.0);
}

TEST(Sub, negative2) {

	Fixed const val1 = -1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val1 - val2;

	EXPECT_EQ(res, -1.0);
}

TEST(Sub, zero) {

	Fixed const val1 = 1.0;
	Fixed const val2 = 0.0;

	Fixed const res = val2 - val1;

	EXPECT_EQ(res, -1.0);
}

// mul
TEST(Mul, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("2.0000000'00000000'00000000");
	Fixed const result = val1 * val2;

	EXPECT_EQ(result,"0.6000000'40000000'20000000");
}

TEST(Mul, negative1) {

	Fixed const val1(-0.3);
	Fixed const val2(2.0);
	Fixed const result = val1 * val2;

	EXPECT_EQ(result, -0.6);
}

TEST(Mul, negative2) {

	Fixed const val1(0.3);
	Fixed const val2(-2.0);
	Fixed const result = val1 * val2;

	EXPECT_EQ(result, -0.6);
}

TEST(Mul, negative3) {

	Fixed const val1(-0.3);
	Fixed const val2(-2.0);
	Fixed const result = val1 * val2;

	EXPECT_EQ(result, 0.6);
}


// shift
TEST(LeftShift, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const result = val1 << 4;

	EXPECT_EQ(result,"3.0000002'00000001'00000000");
}

TEST(RightShift, simple) {

	Fixed const val1("3.0000002'00000001'00000000");
	Fixed const result = val1 >> 4;

	EXPECT_EQ(result, "0.3000000'20000000'10000000");
}

// less
TEST(Less, simple) 
{

	Fixed const val1("1.2");
	Fixed const val2("1.3");
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	EXPECT_EQ(result1, true);
	EXPECT_EQ(result2, false);
}

TEST(Less, negative1)
{

	Fixed const val1(1.2);
	Fixed const val2(-1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	EXPECT_EQ(result1, false);
	EXPECT_EQ(result2, true);
}

TEST(Less, negative2)
{

	Fixed const val1(-1.2);
	Fixed const val2(1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	EXPECT_EQ(result1, true);
	EXPECT_EQ(result2, false);
}


TEST(Less, negative3)
{

	Fixed const val1(-1.2);
	Fixed const val2(-1.3);
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	EXPECT_EQ(result1, false);
	EXPECT_EQ(result2, true);
}

//equal
TEST(Equal, simple)
{

	Fixed const val1("0.6000000'40000000'20000000");
	Fixed const val2("0.6000000'40000000'20000000");
	bool const result1 = val1 == val2;

	EXPECT_EQ(result1, true);
}



// other
TEST(SmallValue, simple)
{
	Fixed const val1 = 0.000000000000005;
	Fixed const val2 = 0.000000000000000;

	Fixed const res = val2 + val1;

	EXPECT_EQ(res, 0.000000000000005);
}


