// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file caseFormat.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The case format that tests/scalar_reference.py writes, and the comparison of a case with the C++ encoding

#ifndef SCALAR_CASE_FORMAT_HPP
#define SCALAR_CASE_FORMAT_HPP

#include "modelInput.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace scalarTests {

// One case: the inputs of tag_identifier and the outcome of the Python
struct ReferenceCase {
    std::string               group;
    std::string               featureSet;  // legacy or default
    scalar::NameInContext     row;
    std::vector<std::string>  words;
    bool                      tooLong = false;
    std::vector<std::int32_t> ids;
    std::vector<std::size_t>  positions;
};

// The outcome of the C++ encoder for one identifier
struct Outcome {
    bool               tooLong = false;
    scalar::ModelInput input;
};

// The fields of a line, split at each tab
std::vector<std::string_view> splitFields(std::string_view line);

// Reverses escape() of scalar_reference.py: \\, \t, \n, and \xHH
std::string unescapeField(std::string_view field);

// Escapes a string the way scalar_reference.py does, for messages
std::string escapeField(std::string_view text);

// Parses one line of the case format. Throws std::runtime_error on a malformed line.
ReferenceCase parseCase(std::string_view line);

// The normalized feature list of a feature set name: legacy or default
const std::vector<std::string>& featureList(const std::string& featureSet);

// The full context name of the first letter that corpus_words.tsv stores
std::string contextOfLetter(char letter);

// Runs encodeIdentifier and turns IdentifierTooLongError into an outcome
Outcome encodeCase(const scalar::WordPieceTokenizer& tokenizer, const scalar::NameInContext& row,
                   const std::vector<std::string>& words, const std::string& featureSet);

// True when the C++ outcome equals the outcome of the Python
bool sameOutcome(const ReferenceCase& expected, const Outcome& actual);

// The FNV-1a hash (32 bits) of digest() in scalar_reference.py
std::uint32_t digestOf(const Outcome& outcome);

// The inputs and the two outcomes of a case, for a mismatch report
std::string describeMismatch(const ReferenceCase& expected, const Outcome& actual);

// The inputs and the C++ outcome of an identifier, for a report of a digest mismatch
std::string describeOutcome(const std::string& featureSet, const scalar::NameInContext& row, const std::vector<std::string>& words,
                            const Outcome& outcome);

} // namespace scalarTests

#endif
