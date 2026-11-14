// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file sequenceMatcher.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The similarity ratio of difflib.SequenceMatcher, which the sys_sim feature of SCALAR uses
// Port of SequenceMatcher of Lib/difflib.py, CPython v3.14.10 (PSF-2.0)

#ifndef SCALAR_SEQUENCE_MATCHER_HPP
#define SCALAR_SEQUENCE_MATCHER_HPP

#include <string_view>

namespace scalar {

// SequenceMatcher(None, a, b).ratio(), autojunk included, with one element per byte;
//   the bytes equal the characters of the Python strings for ASCII text
double sequenceMatcherRatio(std::string_view a, std::string_view b);

} // namespace scalar

#endif
