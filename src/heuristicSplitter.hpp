// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file heuristicSplitter.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// First stage of Ronin, which splits at delimiters, terms with numbers, camel case, and digits.
// Port of spiral/simple_splitters.py (heuristic_split) of Spiral (cnewman fork) commit dff5373, GPL-3.0.

#ifndef SCALAR_HEURISTIC_SPLITTER_HPP
#define SCALAR_HEURISTIC_SPLITTER_HPP

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scalar {

// Splitter of simple_splitters.heuristic_split(identifier, keep_numbers=True). Immutable after
// construction, therefore safe to use from several threads at once.
class HeuristicSplitter {
public:
    // Takes the entries of constants.common_terms_with_numbers, each a regular expression of literal
    // letters and digits, classes such as [0-9], and {m}, {m,}, or {m,n} quantifiers, and the entries
    // of constants.common_suffix_numbers. Throws std::invalid_argument for any other syntax.
    HeuristicSplitter(const std::vector<std::string>& termsWithNumbers, const std::vector<std::string>& suffixNumbers);

    // Tokens of heuristic_split(identifier). Where terms with numbers compete at one position, the
    // longest wins, the one order that keeps every term reachable; Python follows PYTHONHASHSEED.
    std::vector<std::string> split(const std::string& identifier) const;

    // Length of the longest term with numbers that matches 'text' at 'position', ignoring ASCII case, or 0
    std::size_t longestTermAt(const std::string& text, std::size_t position) const;

private:
    // One character class of a term with its repetition limits
    struct Atom {
        std::bitset<256> bytes;
        std::size_t      minCount = 0;
        std::size_t      maxCount = 0;
    };
    using Pattern = std::vector<Atom>;

    // Node of the trie of the terms that hold only letters and digits, indexed by a-z then 0-9
    struct TrieNode {
        std::array<std::int32_t, 36> next{};
        bool                         isTerm = false;
    };

    static Pattern parsePattern(const std::string& term);
    static std::size_t longestMatch(const Pattern& pattern, const std::string& text, std::size_t position);
    void addLiteral(const std::string& term);

    std::vector<TrieNode>    literalTrie;
    std::vector<Pattern>     patterns;
    std::vector<std::string> suffixNumbers;
};

}

#endif
