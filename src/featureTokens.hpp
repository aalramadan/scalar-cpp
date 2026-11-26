// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file featureTokens.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The feature tokens and the position tokens that precede the words of an identifier in the SCALAR model input
// Port of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py of SCANL/scanl_tagger (GPL-3.0)

#ifndef SCALAR_FEATURE_TOKENS_HPP
#define SCALAR_FEATURE_TOKENS_HPP

#include "nameInContext.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace scalar {

// The tokens of the model input before tokenization: the feature tokens, then each word
//   after its position token
struct ModelInputTokens {
    std::vector<std::string> tokens;
    std::size_t              featureCount = 0;  // the number of leading feature tokens
};

// The features that a checkpoint may select, in the order of AVAILABLE_FEATURES
const std::vector<std::string>& availableFeatures();

// The features of checkpoints saved without a feature list, such as sourceslicer/scalar_lm_best of
//   earlier releases
const std::vector<std::string>& legacyFeatures();

// normalize_selected_features: strips the names, drops empty and repeated ones, and returns
//   them in model-input order. No list selects DEFAULT_FEATURES, every available feature but the
//   opt-in ones. Throws
//   std::invalid_argument with the message of the Python on an unknown name.
std::vector<std::string> normalizeSelectedFeatures(const std::optional<std::vector<std::string>>& selected);

// The features of a checkpoint, as DistilBertTagger.__init__ selects them: the saved list,
//   or the legacy list when the config saves none. Throws std::runtime_error on a feature that
//   the port does not compute, such as lead_word.
std::vector<std::string> modelFeatures(const std::optional<std::vector<std::string>>& savedFeatures);

// The selected_features list of a config.json; none when the key is missing or null.
//   Throws std::runtime_error on another kind of value.
std::optional<std::vector<std::string>> readSavedFeatures(const std::string& configPath);

// get_feature_tokens: the feature tokens of an identifier for a feature list, which is
//   normalized first. The features read the words and the other inputs of the row, never its name;
//   an empty input stands for None of the Python. Throws std::invalid_argument on a byte above 0x7F in the row or the words.
std::vector<std::string> featureTokens(const NameInContext& row, const std::vector<std::string>& words,
                                       const std::vector<std::string>& features);

// build_model_input_tokens: the feature tokens, then @pos_0, @pos_1, or @pos_2 before each word
ModelInputTokens buildModelInputTokens(const NameInContext& row, const std::vector<std::string>& words,
                                       const std::vector<std::string>& features);

} // namespace scalar

#endif
