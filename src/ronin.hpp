// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file ronin.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Ronin identifier splitter of Spiral as SCALAR calls it, ronin.split(identifier) after the default init().
// Port of spiral/ronin.py of Spiral (cnewman fork) commit dff5373, GPL-3.0.

#ifndef SCALAR_RONIN_HPP
#define SCALAR_RONIN_HPP

#include "heuristicSplitter.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace scalar {

// Defaults of Ronin.init() in spiral/ronin.py, which SCALAR keeps
struct RoninParameters {
    static constexpr std::int64_t lowFreqCutoff    = 340;
    static constexpr std::size_t  lengthCutoff     = 2;
    static constexpr std::int64_t shortMinFreq     = 286540;
    static constexpr double       normalExponent   = 0.1527;
    static constexpr double       dictWordExponent = 0.1228;
    static constexpr double       camelBias        = 8.6319;
    static constexpr double       recognitionBias  = 0.0000003635;
    static constexpr double       altExponent      = 1.2165;
};

// Splits identifiers into tokens. Loaded once from a data folder and immutable afterwards,
// therefore split() is safe to call from several threads at once.
class Ronin {
public:
    // Loads the tables that tools/export_ronin_data.py writes to 'dataDirectory'.
    // Throws std::runtime_error when a file is missing or malformed, and std::invalid_argument for a
    // term with numbers outside the syntax that HeuristicSplitter accepts.
    explicit Ronin(const std::string& dataDirectory);

    // Tokens of ronin.split(identifier) with keep_numbers=True. Equal to Python for 7-bit ASCII,
    // with the longest term with numbers where Python's choice depends on PYTHONHASHSEED.
    std::vector<std::string> split(const std::string& identifier) const;

private:
    // Count of a token in the frequency table with its two rescaled scores, math.pow of the count
    // with dict_word_exponent and with normal_exponent, computed once at load time
    struct Frequency {
        std::int64_t count;
        double       dictionaryScore;
        double       normalScore;
    };

    // Score of a split and its tokens, the (score, list) tuples of _same_case_split
    struct ScoredSplit {
        double                   score = 0.0;
        std::vector<std::string> tokens;
    };

    // Results of _same_case_split for the suffixes of one token, indexed by suffix length
    using SuffixResults = std::vector<std::optional<ScoredSplit>>;

    static Frequency rescaled(std::int64_t count);

    const Frequency* frequencyOf(const std::string& token) const;
    double adjustedScore(const std::string& token) const;
    bool recognized(const std::string& token) const;
    bool specialCase(const std::string& token) const;
    bool inDictionary(const std::string& word) const;
    std::string stem(const std::string& token) const;
    bool isPrefix(const std::string& text) const;
    bool isSuffix(const std::string& text) const;
    ScoredSplit sameCaseSplit(const std::string& text, double scoreNs, SuffixResults& suffixResults) const;

    HeuristicSplitter                          heuristic;
    std::unordered_set<std::string>            termsWithNumbers;
    std::unordered_set<std::string>            specialComputingTerms;
    std::unordered_set<std::string>            prefixes;
    std::unordered_set<std::string>            suffixes;
    std::unordered_set<std::string>            dictionary;
    std::unordered_map<std::string, Frequency> frequencies;
    Frequency                                  highestFrequency;
    double                                     biasedThreshold;
};

}

#endif
