#include "gtest/gtest.h"

#include <FixedPointValue.h>

#include <iostream> 

using namespace std;

using Fixed = FixedPointValue<3>;

TEST(Add, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("0.6000000'60000000'50000000");
	Fixed const result = val1 + val2;

	EXPECT_EQ(result,"0.9000000'80000000'60000000");

	cout << result << endl;
}

TEST(Sub, simple) {

	Fixed const val1("0.6000000'30000000'20000000");
	Fixed const val2("0.3000000'30000000'30000000");
	Fixed const result = val1 - val2;

	EXPECT_EQ(result,"0.2ffffff'ffffffff'f0000000");

	cout << result << endl;
}


TEST(Mul, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const val2("2.0000000'00000000'00000000");
	Fixed const result = val1 * val2;

	EXPECT_EQ(result,"0.6000000'40000000'20000000");

	cout << result << endl;
}


TEST(LeftShift, simple) {

	Fixed const val1("0.3000000'20000000'10000000");
	Fixed const result = val1 << 4;

	EXPECT_EQ(result,"3.0000002'00000001'00000000");

	cout << result << endl;
}

TEST(RightShift, simple) {

	Fixed const val1("3.0000002'00000001'00000000");
	Fixed const result = val1 >> 4;

	EXPECT_EQ(result, "0.3000000'20000000'10000000");

	cout << result << endl;
}


TEST(Less, simple) 
{

	Fixed const val1("1.2");
	Fixed const val2("1.3");
	bool const result1 = val1 < val2;
	bool const result2 = val2 < val1;

	EXPECT_EQ(result1, true);
	EXPECT_EQ(result2, false);

	cout << result1 << endl;
	cout << result2 << endl;
}


TEST(Equal, simple)
{

	Fixed const val1("0.6000000'40000000'20000000");
	Fixed const val2("0.6000000'40000000'20000000");
	bool const result1 = val1 == val2;

	EXPECT_EQ(result1, true);

	cout << result1 << endl;
}


TEST(Double, simple)
{

	Fixed const val1 = 0.00008;
	Fixed const val2 = 2.0;
	Fixed const result = val1 * val2;

	EXPECT_EQ(result, Fixed(0.00016));

	cout << result << endl;
}