// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file ronin.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Ronin identifier splitter of Spiral as SCALAR calls it, ronin.split(identifier) after the default init().
// Port of spiral/ronin.py of Spiral (cnewman fork) commit dff5373, GPL-3.0.

#include "ronin.hpp"
#include "asciiText.hpp"
#include "dataFile.hpp"
#include "englishStemmer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace scalar {

namespace {

// Longest token that sameCaseSplit splits by plain recursion, which needs less than 256 KiB of stack
constexpr std::size_t shallowTokenLength = 1000;

// Entries of a table file of the data folder, one per line; an entry is never empty
std::vector<std::string> readTable(const std::string& dataDirectory, const char* name) {
    const std::string path = (std::filesystem::path(dataDirectory) / name).string();
    const std::optional<DataFile> file = openDataFile(path);
    if (!file) {
        throw std::runtime_error("cannot read the Ronin table " + path);
    }
    std::istringstream input{std::string(file->bytes())};
    std::vector<std::string> entries;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("empty line in the Ronin table " + path);
        }
        entries.push_back(std::move(line));
    }
    if (input.bad() || entries.empty()) {
        throw std::runtime_error("cannot read the entries of the Ronin table " + path);
    }
    return entries;
}

std::unordered_set<std::string> readSet(const std::string& dataDirectory, const char* name) {
    std::vector<std::string> entries = readTable(dataDirectory, name);
    return std::unordered_set<std::string>(std::make_move_iterator(entries.begin()), std::make_move_iterator(entries.end()));
}

// _stem: drops a final 's', since NLTK stems technical terms that end in 's' oddly, and stems the rest
std::string roninStem(const std::string& token) {
    if (token.size() > 1 && token.back() == 's') {
        return token.substr(0, token.size() - 1);
    }
    return stemEnglish(token);
}

}

Ronin::Ronin(const std::string& dataDirectory)
    : heuristic(readTable(dataDirectory, "common_terms_with_numbers.txt"), readTable(dataDirectory, "common_suffix_numbers.txt")),
      termsWithNumbers(readSet(dataDirectory, "common_terms_with_numbers.txt")),
      specialComputingTerms(readSet(dataDirectory, "special_computing_terms.txt")),
      prefixes(readSet(dataDirectory, "prefixes.txt")),
      suffixes(readSet(dataDirectory, "suffixes.txt")),
      dictionary(readSet(dataDirectory, "dictionary.txt")),
      highestFrequency{0, 0.0, 0.0},
      biasedThreshold(0.0) {
    const std::vector<std::string> lines = readTable(dataDirectory, "frequencies.txt");
    frequencies.reserve(lines.size());
    std::int64_t highest = 0;
    for (const std::string& line : lines) {
        const std::size_t tab = line.find('\t');
        std::int64_t count = 0;
        const char* const end = line.data() + line.size();
        const auto parsed = tab == std::string::npos ? std::from_chars_result{end, std::errc::invalid_argument}
                                                     : std::from_chars(line.data() + tab + 1, end, count);
        if (tab == 0 || parsed.ec != std::errc() || parsed.ptr != end || count <= 0) {
            throw std::runtime_error("malformed line in the Ronin table frequencies.txt: " + line);
        }
        frequencies.emplace(line.substr(0, tab), rescaled(count));
        highest = std::max(highest, count);
    }
    highestFrequency = rescaled(highest);
    biasedThreshold = RoninParameters::recognitionBias * static_cast<double>(highest);
}

Ronin::Frequency Ronin::rescaled(std::int64_t count) {
    // _adjusted_score rescales only counts above low_freq_cutoff
    if (count <= RoninParameters::lowFreqCutoff) {
        return {count, 0.0, 0.0};
    }
    const double value = static_cast<double>(count);
    return {count, std::pow(value, RoninParameters::dictWordExponent), std::pow(value, RoninParameters::normalExponent)};
}

// The count that score_function of init() gives 'token': the highest count for a term with numbers,
// else the count of the lowercase token; null stands for a count of 0
const Ronin::Frequency* Ronin::frequencyOf(const std::string& token) const {
    const std::string lowered = toLowerAscii(token);
    if (termsWithNumbers.count(lowered) > 0) {
        return &highestFrequency;
    }
    const auto entry = frequencies.find(lowered);
    return entry == frequencies.end() ? nullptr : &entry->second;
}

// _adjusted_score: 0 for a rare token and for a short token that is neither frequent nor a special
// case, else the count rescaled with the exponent for recognized or for other tokens
double Ronin::adjustedScore(const std::string& token) const {
    if (token.empty()) {
        return 0.0;
    }
    const Frequency* const frequency = frequencyOf(token);
    const std::int64_t count = frequency == nullptr ? 0 : frequency->count;
    if (token.size() <= RoninParameters::lengthCutoff && count <= RoninParameters::shortMinFreq && !specialCase(token)) {
        return 0.0;
    }
    if (count <= RoninParameters::lowFreqCutoff) {
        return 0.0;
    }
    return recognized(token) ? frequency->dictionaryScore : frequency->normalScore;
}

bool Ronin::recognized(const std::string& token) const {
    const std::string lowered = toLowerAscii(token);
    return specialCase(lowered) || inDictionary(lowered);
}

bool Ronin::specialCase(const std::string& token) const {
    return termsWithNumbers.count(token) > 0 || specialComputingTerms.count(token) > 0 ||
           specialComputingTerms.count(roninStem(token)) > 0;
}

bool Ronin::inDictionary(const std::string& word) const {
    // Single letters count as no words
    if (word.size() <= 1) {
        return false;
    }
    return dictionary.count(word) > 0 || dictionary.count(roninStem(word)) > 0;
}

bool Ronin::isPrefix(const std::string& text) const {
    return prefixes.count(toLowerAscii(text)) > 0;
}

bool Ronin::isSuffix(const std::string& text) const {
    return suffixes.count(toLowerAscii(text)) > 0;
}

std::vector<std::string> Ronin::split(const std::string& identifier) const {
    std::vector<std::string> splits;
    for (const std::string& part : heuristic.split(identifier)) {
        if (recognized(part)) {
            splits.push_back(part);
            continue;
        }
        // re.search(r'[A-Z][a-z]', part) finds the first upper-to-lower case transition
        std::size_t i = 0;
        while (i + 1 < part.size() && !(isAsciiUpper(part[i]) && isAsciiLower(part[i + 1]))) {
            ++i;
        }
        if (i + 1 >= part.size()) {
            splits.push_back(part);
            continue;
        }
        // Compares the raw count with an adjusted score, as Ronin does on purpose
        const std::string camel = part.substr(i);
        const Frequency* const camelFrequency = frequencyOf(camel);
        const double camelScore = camelFrequency == nullptr ? 0.0 : static_cast<double>(camelFrequency->count);
        const double altScore = adjustedScore(part.substr(i + 1)) * RoninParameters::camelBias;
        if (camelScore >= altScore) {
            if (i > 0) {
                splits.push_back(part.substr(0, i));
            }
            splits.push_back(camel);
        } else {
            splits.push_back(part.substr(0, i + 1));
            splits.push_back(part.substr(i + 1));
        }
    }

    std::vector<std::string> results;
    for (const std::string& token : splits) {
        SuffixResults suffixResults(token.size() + 1);
        const double scoreNs = adjustedScore(token);
        // sameCaseSplit recurses once per suffix and overflows a small thread stack on a long token.
        //  Splitting the suffixes from the shortest up keeps each call one level deep, with the same results.
        if (token.size() > shallowTokenLength) {
            for (std::size_t length = 1; length < token.size(); ++length) {
                suffixResults[length] = sameCaseSplit(token.substr(token.size() - length), scoreNs, suffixResults);
            }
        }
        ScoredSplit scored = sameCaseSplit(token, scoreNs, suffixResults);
        results.insert(results.end(), std::make_move_iterator(scored.tokens.begin()), std::make_move_iterator(scored.tokens.end()));
    }
    return results;
}

// _same_case_split, whose last candidate after bisect.insort is the maximum under the same (score, list)
// order. A suffix is always split with the same score_ns, therefore its result is kept per suffix length.
Ronin::ScoredSplit Ronin::sameCaseSplit(const std::string& text, double scoreNs, SuffixResults& suffixResults) const {
    if (recognized(text)) {
        return {scoreNs, {text}};
    }
    double maxScore = -1.0;
    const double threshold = std::max(adjustedScore(text), scoreNs);
    std::optional<ScoredSplit> best;
    const auto offer = [&best](double score, std::vector<std::string> tokens) {
        if (!best || best->score < score || (best->score == score && best->tokens < tokens)) {
            best = ScoredSplit{score, std::move(tokens)};
        }
    };

    for (std::size_t i = 1; i < text.size(); ++i) {
        const std::string left = text.substr(0, i);
        const std::string right = text.substr(i);
        if (isPrefix(left) || (text.size() > 5 && isSuffix(right))) {
            continue;
        }
        const double scoreLeft = adjustedScore(left);
        const double scoreRight = adjustedScore(right);
        const bool breakRight = scoreRight > threshold;
        const bool breakLeft = scoreLeft > threshold;
        const double eitherScore = std::max(scoreLeft, scoreRight);

        if (breakLeft && breakRight) {
            if (scoreLeft + scoreRight > maxScore) {
                maxScore = scoreLeft + scoreRight;
                offer(maxScore, {left, right});
            }
        } else if (breakLeft) {
            if (eitherScore > biasedThreshold && recognized(right)) {
                offer(eitherScore, {left, right});
                continue;
            }
            std::optional<ScoredSplit>& rightResult = suffixResults[right.size()];
            if (!rightResult) {
                rightResult = sameCaseSplit(right, scoreNs, suffixResults);
            }
            if (rightResult->tokens.front() != right) {
                std::vector<std::string> tokens{left};
                tokens.insert(tokens.end(), rightResult->tokens.begin(), rightResult->tokens.end());
                const double score = rightResult->score / static_cast<double>(tokens.size());
                offer(score, std::move(tokens));
            } else if (specialCase(right)) {
                maxScore = eitherScore;
                offer(eitherScore, {left, right});
            }
        } else if (breakRight && (recognized(left) || left.size() <= RoninParameters::lengthCutoff || specialCase(right))) {
            if (eitherScore > biasedThreshold) {
                offer(scoreRight / static_cast<double>(left.size()), {left, right});
            } else if (recognized(right)) {
                offer(scoreRight, {left, right});
            }
        }
    }

    if (!best) {
        return {maxScore, {text}};
    }
    const double scaledScore = best->score / std::pow(static_cast<double>(best->tokens.size()), RoninParameters::altExponent);
    if (scaledScore >= scoreNs) {
        return std::move(*best);
    }
    return {scoreNs, {text}};
}

}
