// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file pythonSum.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The sum of a list of floats as the built-in sum() of CPython computes it, for the cvr feature of SCALAR
// Port of the float loop of builtin_sum_impl, Python/bltinmodule.c, CPython v3.14.10 (PSF-2.0)

#ifndef SCALAR_PYTHON_SUM_HPP
#define SCALAR_PYTHON_SUM_HPP

#include <vector>

namespace scalar {

// sum(values) for a non-empty list of floats. CPython 3.12 and later add floats with the
//   compensated summation of Neumaier, which can differ from a plain loop in the last bit.
double pythonSum(const std::vector<double>& values);

} // namespace scalar

#endif
