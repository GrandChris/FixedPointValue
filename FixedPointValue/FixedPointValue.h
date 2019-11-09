///////////////////////////////////////////////////////////////////////////////
// File:		  FixedPointValue.h
// Revision:	  1
// Date Creation: 07.11.2019
// Last Change:	  07.11.2019
// Author:		  Christian Steinbrecher
// Descrition:	  Same as float, but with fixed exponent 
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined __CUDACC__
#define GPU_ENABLED __device__ __forceinline__
#else
#define GPU_ENABLED inline
#endif

#include <ostream>

template<size_t nrUnsignedInts = 3, size_t bitsBeforePoint = 4>
class FixedPointValue
{
public:
	// C-Tor
	constexpr FixedPointValue() = default;

	GPU_ENABLED constexpr FixedPointValue(char const str[]);
	GPU_ENABLED constexpr FixedPointValue(double const val);

	// Arithmetic functions
	GPU_ENABLED constexpr FixedPointValue operator+(FixedPointValue const& right) const;
	GPU_ENABLED constexpr FixedPointValue operator-(FixedPointValue const& right) const;
	GPU_ENABLED constexpr FixedPointValue operator*(FixedPointValue const& right) const;

	GPU_ENABLED constexpr FixedPointValue operator<<(size_t const bits) const;
	GPU_ENABLED constexpr FixedPointValue operator>>(size_t const bits) const;

	GPU_ENABLED constexpr FixedPointValue const & operator+=(FixedPointValue const& right);
	GPU_ENABLED constexpr FixedPointValue const & operator-=(FixedPointValue const& right);
	GPU_ENABLED constexpr FixedPointValue const & operator*=(FixedPointValue const& right);

	// Logic functions
	GPU_ENABLED constexpr bool operator<(FixedPointValue const& right) const;
	GPU_ENABLED constexpr bool operator==(FixedPointValue const& right) const;


	void print(std::ostream & ost) const;
	
private:
	unsigned int mValue[nrUnsignedInts] = { 0 };


	// Option to double the speed of a multiplication
	static constexpr size_t const mMulStartindex = 0;	// high precision
	//static constexpr size_t const mMulStartindex = nrUnsignedInts - 1;	// lower precision, but faster
};


template<size_t nrUnsignedInts, size_t bitsBeforePoint>
std::ostream& operator<<(std::ostream& ost, FixedPointValue<nrUnsignedInts, bitsBeforePoint> const& val);








// #######+++++++ Implementation +++++++#######


#include <iomanip>
#include <cassert>
#include <cmath>

// #######+++++++ Helper Functions +++++++#######

GPU_ENABLED constexpr void add(unsigned int const a, unsigned int const b1, unsigned int const b2, unsigned int& resLow, unsigned int& resHigh)
{
	unsigned long long const res = static_cast<unsigned long long>(a) + b1 + b2;
	resLow = static_cast<unsigned int>(res);
	resHigh = static_cast<unsigned int>(res >> 32);
}

GPU_ENABLED constexpr void mult(unsigned int const a, unsigned int const b, unsigned int& resLow, unsigned int& resHigh)
{
	unsigned long long const res = static_cast<unsigned long long>(a) * b;
	resLow = static_cast<unsigned int>(res);
	resHigh = static_cast<unsigned int>(res >> 32);
}

GPU_ENABLED constexpr void lshift(unsigned int const a, unsigned int& resLow, unsigned int& resHigh, size_t const size)
{
	resLow =  a << size;
	resHigh = a >> (sizeof(unsigned int) * 8 - size);
}

GPU_ENABLED constexpr void rshift(unsigned int const a, unsigned int& resLow, unsigned int& resHigh, size_t const size)
{
	resHigh = a >> size;
	resLow = a << (sizeof(unsigned int) * 8 - size);
}


GPU_ENABLED constexpr unsigned char charToHex(unsigned char const c)
{
	switch (c)
	{
	case '0':
		return 0x0;
	case '1':
		return 0x1;
	case '2':
		return 0x2;
	case '3':
		return 0x3;
	case '4':
		return 0x4;
	case '5':
		return 0x5;
	case '6':
		return 0x6;
	case '7':
		return 0x7;
	case '8':
		return 0x8;
	case '9':
		return 0x9;
	case 'a':
	case 'A':
		return 0xa;
	case 'b':
	case 'B':
		return 0xb;
	case 'c':
	case 'C':
		return 0xc;
	case 'd':
	case 'D':
		return 0xd;
	case 'e':
	case 'E':
		return 0xe;
	case 'f':
	case 'F':
		return 0xf;
	default:
		assert(false);	// value is not hexadecimal
		return 0x0;
	};
}

// <------------

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>::FixedPointValue(char const str[])
{
	size_t const nrHexDigits = sizeof(unsigned int) * 2;

	int j = nrUnsignedInts * nrHexDigits - 1;
	for (int i = 0; str[i] != '\0'; ++i)
	{
		if (str[i] != '\'' && str[i] != '.')
		{		
			if (j < 0)
			{
				assert(false);	// hexadecimal string is to long
				return;
			}

			unsigned int const hexVal = charToHex(str[i]);
			mValue[j / nrHexDigits] |= hexVal << ((j % nrHexDigits) * 4);

			--j;
		}
	}
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>::FixedPointValue(double const val)
{
	size_t const mantisseSize = 52;

	int exponent = 0;
	double const significand = frexp(val, &exponent) + 1;

	unsigned long long const* const pSignificand = (unsigned long long const*) & significand;

	
	//double const siginficand_noComma = val * exponentMult;

	unsigned long long significandLong = (*pSignificand) << (64 - mantisseSize);
	unsigned int const high = static_cast<unsigned int>(significandLong >> 32);
	unsigned int const low = static_cast<unsigned int>(significandLong);

	mValue[nrUnsignedInts - 1] = high;
	if constexpr (nrUnsignedInts >= 2)
	{
		mValue[nrUnsignedInts - 2] = low;
	}

	
	int const leftShifValue = - static_cast<int>(bitsBeforePoint) + exponent ;
	if (leftShifValue < 0)
	{
		*this = *this >> (-leftShifValue);
	}
	else
	{
		*this = *this << leftShifValue;
	}	
}


template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>
FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator+(FixedPointValue const& right) const
{
	FixedPointValue result;

	unsigned int lastHigh = 0;
	for (size_t i = 0; i < nrUnsignedInts; ++i)
	{
		add(mValue[i], right.mValue[i], lastHigh, result.mValue[i], lastHigh);
	}

	if (lastHigh != 0)
	{	// overflow
		
		//std::cout << std::endl << "overflow" << std::endl;
	}

	return result;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>
	FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator-(FixedPointValue const& right) const
{
	FixedPointValue result;

	unsigned int lastHigh = 1;
	for (size_t i = 0; i < nrUnsignedInts; ++i)
	{
		add(mValue[i], ~right.mValue[i], lastHigh, result.mValue[i], lastHigh);
	}

	if (lastHigh != 0)
	{	// overflow

		//std::cout << std::endl << "overflow" << std::endl;
	}

	return result;
}



template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>
FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator*(FixedPointValue const& right) const
{
	size_t const resultSize = nrUnsignedInts * 2 + 1;
	unsigned int result[resultSize] = { 0 };
	
	for (size_t index = mMulStartindex; index < nrUnsignedInts * 2 - 1; ++index)
	{
		for (size_t j = 0; j < nrUnsignedInts; ++j)
		{
			for (size_t i = 0; i < nrUnsignedInts; ++i)
			{
				if (i + j == index)
				{
					unsigned int resLow = 0;
					unsigned int resHigh = 0;
					mult(mValue[i], right.mValue[j], resLow, resHigh);

					unsigned int lastHigh = 0;
					add(result[index], resLow, 0, result[index], lastHigh);
					add(result[index + 1], resHigh, lastHigh, result[index + 1], lastHigh);
					result[index + 2] += lastHigh;
				}
			}
		}
	}

	unsigned int lastHigh = 0;
	for (size_t i = 0; i < resultSize; ++i)
	{
		unsigned int high = 0;
		lshift(result[i], result[i], high, bitsBeforePoint);
		result[i] |= lastHigh;
		lastHigh = high;
	}

	if (result[nrUnsignedInts * 2] != 0)
	{ // overflow

		assert(false);	
	}

	FixedPointValue res;
	for (size_t i = 0; i < nrUnsignedInts; ++i)
	{
		res.mValue[i] = result[i + nrUnsignedInts];
	}

	return res;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint> 
	FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator<<(size_t const bits) const
{
	FixedPointValue<nrUnsignedInts, bitsBeforePoint> res = *this;

	unsigned int lastHigh = 0;
	for (size_t i = 0; i < nrUnsignedInts; ++i)
	{
		unsigned int high = 0;
		lshift(res.mValue[i], res.mValue[i], high, bits);
		res.mValue[i] |= lastHigh;
		lastHigh = high;
	}

	return res;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint>
FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator>>(size_t const bits) const
{
	FixedPointValue<nrUnsignedInts, bitsBeforePoint> res = *this;

	unsigned int lastLow = 0;
	for (int i = nrUnsignedInts-1; i >= 0; --i)
	{
		unsigned int low = 0;
		rshift(res.mValue[i], low, res.mValue[i], bits);
		res.mValue[i] |= lastLow;
		lastLow = low;
	}

	return res;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint> const&
	FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator+=(FixedPointValue const& right)
{
	*this = *this + right;
	return *this;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr FixedPointValue<nrUnsignedInts, bitsBeforePoint> const&
	FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator-=(FixedPointValue const& right)
{
	*this = *this - right;
	return *this;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr  FixedPointValue<nrUnsignedInts, bitsBeforePoint> const&
	FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator*=(FixedPointValue const& right)
{
	*this = *this * right;
	return *this;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr bool FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator<(FixedPointValue const& right) const
{
	for (int i = nrUnsignedInts-1; i >= 0; --i)
	{
		if (mValue[i] != right.mValue[i])
		{
			return mValue[i] < right.mValue[i];
		}
	}

	return false;
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
GPU_ENABLED constexpr bool FixedPointValue<nrUnsignedInts, bitsBeforePoint>::operator==(FixedPointValue const& right) const
{
	bool equal = true;

	for (int i = nrUnsignedInts - 1; i >= 0; --i)
	{
		equal = equal && mValue[i] == right.mValue[i];
	}

	return equal;
}


template<size_t nrUnsignedInts, size_t bitsBeforePoint>
inline void FixedPointValue<nrUnsignedInts, bitsBeforePoint>::print(std::ostream& ost) const
{
	int i = nrUnsignedInts - 1;
	if (i >= 0)
	{
		ost << std::hex <<   (mValue[i] >> 28) << ".";
		ost << std::hex << std::setfill('0') << std::setw(7) << (mValue[i] & 0xfffffff)  << " ";
	}
	--i;

	for (; i >= 0; --i)
	{
		ost << std::hex << std::setfill('0') << std::setw(8) << mValue[i] << " ";
	}
}

template<size_t nrUnsignedInts, size_t bitsBeforePoint>
inline std::ostream& operator<<(std::ostream& ost, FixedPointValue<nrUnsignedInts, bitsBeforePoint> const& val)
{
	val.print(ost);
	return ost;
}