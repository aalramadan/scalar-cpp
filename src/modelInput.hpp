// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file modelInput.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The model input of one identifier: its input ids and the position of the first subtoken of each word
// Port of steps 1 to 3 of DistilBertTagger.tag_identifier, scalar_tagger/lm_based_tagger/distilbert_tagger.py of SCANL/scanl_tagger (GPL-3.0)

#ifndef SCALAR_MODEL_INPUT_HPP
#define SCALAR_MODEL_INPUT_HPP

#include "featureTokens.hpp"
#include "wordpieceTokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace scalar {

// The ValueError of tag_identifier: some word has no subtoken in the model input, after the
//   truncation to 512 ids or for a word that tokenizes to nothing, such as an empty word
class IdentifierTooLongError : public std::runtime_error {
public:
    IdentifierTooLongError() : std::runtime_error("Identifier is too long for the model input.") {}
};

// The input of the model for one identifier. The attention mask is all ones, since one
//   sequence gets no padding.
struct ModelInput {
    std::vector<std::int32_t> inputIds;       // [CLS], the ids of the tokens, [SEP]
    std::vector<std::size_t>  wordPositions;  // for each word, the index in inputIds of its first subtoken
};

// Steps 1 to 3 of tag_identifier: the feature and position tokens, their tokenization, and the
//   first subtoken of each word. Throws IdentifierTooLongError when a word has no position and
//   std::invalid_argument on a byte above 0x7F or an unknown feature name.
ModelInput encodeIdentifier(const WordPieceTokenizer& tokenizer, const IdentifierRow& row, const std::vector<std::string>& words,
                            const std::vector<std::string>& features);

} // namespace scalar

#endif
