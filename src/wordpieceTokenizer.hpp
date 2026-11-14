// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file wordpieceTokenizer.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The DistilBERT tokenizer of SCALAR for pre-split words, on ASCII text
// Port of the BertNormalizer, BertPreTokenizer, WordPiece, added-token, truncation, and TemplateProcessing steps of tokenizers v0.23.3 (Apache-2.0), as BertTokenizer of transformers v5.19.0 (Apache-2.0) sets them up

#ifndef SCALAR_WORDPIECE_TOKENIZER_HPP
#define SCALAR_WORDPIECE_TOKENIZER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace scalar {

// The input ids of one sequence, with the index of the input word behind each id
struct Encoding {
    std::vector<std::int32_t>               ids;
    std::vector<std::optional<std::size_t>> wordIds;  // none for [CLS] and [SEP], as word_ids() gives None
};

// The tokenizer that DistilBertTokenizerFast.from_pretrained builds from a model folder. Its
//   methods keep no state between calls, therefore threads may share one instance.
class WordPieceTokenizer {
public:
    // Loads vocab.txt of the folder after it checks that tokenizer.json, tokenizer_config.json,
    //   and special_tokens_map.json hold the settings this port follows. Throws
    //   std::runtime_error on a missing file or on any other setting.
    explicit WordPieceTokenizer(const std::string& modelDirectory);

    // The tokenizer call of tag_identifier with is_split_into_words=True, truncation=True, and
    //   padding=True on one sequence: [CLS], the ids of the words cut to the length limit, and
    //   [SEP]. One sequence needs no padding. Throws std::invalid_argument on a byte above 0x7F.
    Encoding encode(const std::vector<std::string>& words) const;

    // The number of entries of vocab.txt
    std::size_t vocabularySize() const { return vocabulary.size(); }

    // The id of a token of vocab.txt, or none
    std::optional<std::int32_t> tokenId(const std::string& token) const;

private:
    void appendWordIds(std::string_view word, std::vector<std::int32_t>& ids) const;
    void appendTextIds(std::string_view text, std::vector<std::int32_t>& ids) const;
    void appendPieceIds(std::string_view piece, std::vector<std::int32_t>& ids) const;

    std::unordered_map<std::string, std::int32_t>     vocabulary;
    std::vector<std::pair<std::string, std::int32_t>> specialTokens;  // found in the raw word before normalization
    std::array<bool, 256>                             startsSpecialToken{};  // the first bytes of the special tokens
    std::array<std::int32_t, 128>                     punctuationIds{};
    std::int32_t                                      unknownId        = 0;
    std::int32_t                                      classificationId = 0;
    std::int32_t                                      separatorId      = 0;
    std::size_t                                       maxLength        = 0;  // model_max_length, [CLS] and [SEP] included
    std::size_t                                       longestEntry     = 0;  // the bytes of the longest entry of vocab.txt
};

} // namespace scalar

#endif
