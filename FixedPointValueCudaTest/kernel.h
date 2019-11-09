#pragma once

#include <FixedPointValue.h>
#include <pfc/pfc_complex.h>

using Fixed = FixedPointValue<3>;

using compl = pfc::complex<Fixed>;

void run_kernel(compl const dp_val1[], compl const dp_val2[], compl dp_res[], size_t const size);