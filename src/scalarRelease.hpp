// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarRelease.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// The lists that the Python SCALAR hard-codes and the port mirrors, in one place for the move to a newer
// release. cmake/scalarRelease.cmake names the release, and PORT.md lists the steps of a move.

#ifndef SCALAR_RELEASE_HPP
#define SCALAR_RELEASE_HPP

#include <array>
#include <string_view>
#include <utility>

namespace scalar::release {

// FORBIDDEN_TRANSITIONS of scalar_tagger/lm_based_tagger/distilbert_crf.py, tag bigrams that never occur
//   in the training data and that the CRF scores with -1e4
inline constexpr std::array<std::pair<std::string_view, std::string_view>, 15> forbiddenTransitions{{
    {"N", "N"}, {"N", "NPL"}, {"NPL", "NPL"},
    {"N", "PRE"}, {"NM", "PRE"}, {"NPL", "PRE"}, {"V", "PRE"}, {"VM", "PRE"}, {"P", "PRE"}, {"D", "PRE"},
    {"DT", "PRE"}, {"CJ", "PRE"},
    {"PRE", "VM"}, {"DT", "D"}, {"CJ", "CJ"},
}};

// The feature lists of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::array<std::string_view, 12> availableFeatures{
    "context", "hungarian", "cvr", "digit", "digit_connector", "plural_suffix",
    "type", "type_overlap", "language", "sys_sim", "lead_shape", "lead_word"};
inline constexpr std::array<std::string_view, 1>  optInFeatures{"lead_word"};       // OPT_IN_FEATURES
inline constexpr std::array<std::string_view, 1>  internalFeatures{"hungarian_legacy"};  // INTERNAL_FEATURES
inline constexpr std::array<std::string_view, 4>  legacyFeatures{"context", "hungarian_legacy", "cvr", "digit"};

// FEATURE_ORDER, the order of the feature tokens in the model input
inline constexpr std::array<std::string_view, 13> featureOrder{
    "context", "hungarian", "hungarian_legacy", "cvr", "digit", "digit_connector", "plural_suffix",
    "type", "type_overlap", "language", "sys_sim", "lead_shape", "lead_word"};

} // namespace scalar::release

#endif
