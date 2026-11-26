// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file modelInput.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The model input of one identifier: its input ids and the position of the first subtoken of each word
// Port of steps 1 to 3 of DistilBertTagger.tag_identifier, scalar_tagger/lm_based_tagger/distilbert_tagger.py of SCANL/scanl_tagger (GPL-3.0)

#include "modelInput.hpp"

#include <optional>
#include <utility>

namespace scalar {

ModelInput encodeIdentifier(const WordPieceTokenizer& tokenizer, const IdentifierRow& row, const std::vector<std::string>& words,
                            const std::vector<std::string>& features) {
    const ModelInputTokens tokens   = buildModelInputTokens(row, words, features);
    Encoding               encoding = tokenizer.encode(tokens.tokens);

    // The words sit at the odd offsets after the feature tokens, each after its position token
    ModelInput                 input;
    std::optional<std::size_t> previousWord;
    for (std::size_t index = 0; index < encoding.wordIds.size(); ++index) {
        const std::optional<std::size_t>& word = encoding.wordIds[index];
        if (!word || *word < tokens.featureCount) continue;
        if ((*word - tokens.featureCount) % 2 == 0) continue;
        if (word == previousWord) continue;
        input.wordPositions.push_back(index);
        previousWord = word;
    }
    if (input.wordPositions.size() != words.size()) throw IdentifierTooLongError();
    input.inputIds = std::move(encoding.ids);
    return input;
}

} // namespace scalar
