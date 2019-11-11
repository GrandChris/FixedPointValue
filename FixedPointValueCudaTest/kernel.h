#pragma once

#include <FixedPointValue.h>
#include <pfc/pfc_complex.h>

using Fixed = FixedPointValue<3>;

using compl = pfc::complex<Fixed>;

size_t const testsCount = 20;
void run_tests(bool dp_res[]);