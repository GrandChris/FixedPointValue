

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
	size_t const size = 1;
	auto dp_val1 = dp_make_unique<compl[]>(size);
	auto dp_val2 = dp_make_unique<compl[]>(size);
	auto dp_res = dp_make_unique<compl[]>(size);


	auto hp_val1 = make_unique<compl[]>(size);
	auto hp_val2 = make_unique<compl[]>(size);
	auto hp_res = make_unique<compl[]>(size);

	hp_val1.get()[0] = compl("1.2345678'9abcdeff'343");
	hp_val2.get()[0] = compl(2.0);

	CUDA_CHECK(cudaMemcpy(dp_val1.get(), hp_val1.get(), size * sizeof(compl), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dp_val2.get(), hp_val2.get(), size * sizeof(compl), cudaMemcpyHostToDevice));


	run_kernel(dp_val1.get(), dp_val2.get(), dp_res.get(), size);

	CUDA_CHECK(cudaMemcpy(hp_res.get(), dp_res.get(), size * sizeof(compl), cudaMemcpyDeviceToHost));

	cout << "devie: " << hp_res.get()[0] << endl;
	cout << "host:  " << hp_val1.get()[0] * hp_val2.get()[0] << endl;

	return 0;
}