// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file pythonSum.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The sum of a list of floats as the built-in sum() of CPython computes it, for the cvr feature of SCALAR
// Port of the float loop of builtin_sum_impl, Python/bltinmodule.c, CPython v3.14.10 (PSF-2.0)

#include "pythonSum.hpp"

#include <cmath>
#include <cstddef>

namespace scalar {

double pythonSum(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    // The start value is the integer 0, and 0 + x turns -0.0 into 0.0 before the float loop
    double total        = 0.0 + values.front();
    double compensation = 0.0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        const double x   = values[i];
        const double sum = total + x;
        if (std::fabs(total) >= std::fabs(x)) compensation += (total - sum) + x;
        else compensation += (x - sum) + total;
        total = sum;
    }
    // CPython skips a compensation that is zero or not finite
    if (compensation != 0.0 && std::isfinite(compensation)) total += compensation;
    return total;
}

} // namespace scalar
