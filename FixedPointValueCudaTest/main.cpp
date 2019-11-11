

#include "kernel.h"

#include <memory>
#include <dp_memory.h>

using namespace std;

ostream& operator<<(ostream& ost, compl const & val)
{
	ost << "[" << val.real << ", " << val.imag << "]" << endl;
	return ost;
}

int main()
{
	// run tests
	auto dp_testsRes = dp_make_unique<bool[]>(testsCount);
	auto hp_testsRes = make_unique<bool[]>(testsCount);
	run_tests(dp_testsRes.get());
	CUDA_CHECK(cudaMemcpy(hp_testsRes.get(), dp_testsRes.get(), testsCount * sizeof(bool), cudaMemcpyDeviceToHost));

	bool success = true;
	for (size_t i = 0; i < testsCount; ++i)
	{
		bool const res = hp_testsRes.get()[i];

		success = success && res;
		cout << "Test " << i << " success: " << boolalpha << res << endl;
	}

	
	cout << endl;
	cout << "All tests success = " << boolalpha << success << endl;


	return 0;
}