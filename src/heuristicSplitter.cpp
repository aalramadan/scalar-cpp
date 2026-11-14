// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file heuristicSplitter.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// First stage of Ronin, which splits at delimiters, terms with numbers, camel case, and digits.
// Port of spiral/simple_splitters.py (heuristic_split) of Spiral (cnewman fork) commit dff5373, GPL-3.0.

#include "heuristicSplitter.hpp"
#include "asciiText.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace scalar {

namespace {

// _delimiter_chars of simple_splitters, which str.translate turns into spaces
constexpr std::string_view delimiterChars = "$~_.:/@";

bool isHexDigit(char c) {
    return isAsciiDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

bool isOneToSeven(char c) {
    return c >= '1' && c <= '7';
}

bool isLiteralCharacter(char c) {
    return isAsciiLower(c) || isAsciiUpper(c) || isAsciiDigit(c);
}

// Index of a byte in a trie node, with the case of letters folded, or -1
int trieSymbol(char c) {
    if (isAsciiLower(c)) {
        return c - 'a';
    }
    if (isAsciiUpper(c)) {
        return c - 'A';
    }
    if (isAsciiDigit(c)) {
        return 26 + (c - '0');
    }
    return -1;
}

void addIgnoringCase(std::bitset<256>& bytes, char c) {
    bytes.set(static_cast<unsigned char>(c));
    if (isAsciiLower(c)) {
        bytes.set(static_cast<unsigned char>(c - 'a' + 'A'));
    } else if (isAsciiUpper(c)) {
        bytes.set(static_cast<unsigned char>(c - 'A' + 'a'));
    }
}

// Length of the match of _numbers_re, (0x[0-9a-f]+l?|0[1-7]{2,}|\d+) with re.I, at a digit of 'token'.
// The alternatives are tried in this order, as re does.
std::size_t numberLengthAt(const std::string& token, std::size_t position) {
    const std::size_t size = token.size();
    if (token[position] == '0' && position + 2 < size && (token[position + 1] == 'x' || token[position + 1] == 'X') &&
        isHexDigit(token[position + 2])) {
        std::size_t end = position + 2;
        while (end < size && isHexDigit(token[end])) {
            ++end;
        }
        if (end < size && (token[end] == 'l' || token[end] == 'L')) {
            ++end;
        }
        return end - position;
    }
    if (token[position] == '0') {
        std::size_t end = position + 1;
        while (end < size && isOneToSeven(token[end])) {
            ++end;
        }
        if (end - position >= 3) {
            return end - position;
        }
    }
    std::size_t end = position + 1;
    while (end < size && isAsciiDigit(token[end])) {
        ++end;
    }
    return end - position;
}

// Appends the nonempty pieces of _numbers_re.split(token), whose capture group keeps the numbers
void appendNumberSplit(const std::string& token, std::vector<std::string>& tokens) {
    std::size_t start = 0;
    std::size_t position = 0;
    while (position < token.size()) {
        if (!isAsciiDigit(token[position])) {
            ++position;
            continue;
        }
        if (position > start) {
            tokens.push_back(token.substr(start, position - start));
        }
        const std::size_t length = numberLengthAt(token, position);
        tokens.push_back(token.substr(position, length));
        position += length;
        start = position;
    }
    if (start < token.size()) {
        tokens.push_back(token.substr(start));
    }
}

bool endsWith(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}

HeuristicSplitter::HeuristicSplitter(const std::vector<std::string>& termsWithNumbers,
                                     const std::vector<std::string>& suffixNumbersOfTerms)
    : literalTrie(1, TrieNode{}), suffixNumbers(suffixNumbersOfTerms) {
    literalTrie.front().next.fill(-1);
    for (const std::string& term : termsWithNumbers) {
        if (!term.empty() && std::all_of(term.begin(), term.end(), isLiteralCharacter)) {
            addLiteral(term);
        } else {
            patterns.push_back(parsePattern(term));
        }
    }
}

void HeuristicSplitter::addLiteral(const std::string& term) {
    std::size_t node = 0;
    for (const char c : term) {
        const int symbol = trieSymbol(c);
        if (literalTrie[node].next[symbol] < 0) {
            literalTrie[node].next[symbol] = static_cast<std::int32_t>(literalTrie.size());
            literalTrie.push_back(TrieNode{});
            literalTrie.back().next.fill(-1);
        }
        node = static_cast<std::size_t>(literalTrie[node].next[symbol]);
    }
    literalTrie[node].isTerm = true;
}

HeuristicSplitter::Pattern HeuristicSplitter::parsePattern(const std::string& term) {
    const auto reject = [&term](const char* reason) {
        return std::invalid_argument("term with numbers '" + term + "': " + reason);
    };
    Pattern pattern;
    std::size_t i = 0;
    while (i < term.size()) {
        Atom atom{{}, 1, 1};
        if (isLiteralCharacter(term[i])) {
            addIgnoringCase(atom.bytes, term[i]);
            ++i;
        } else if (term[i] == '[') {
            ++i;
            while (i < term.size() && term[i] != ']') {
                const char first = term[i];
                const bool isRange = i + 2 < term.size() && term[i + 1] == '-' && term[i + 2] != ']';
                const char last = isRange ? term[i + 2] : first;
                if (!isLiteralCharacter(first) || !isLiteralCharacter(last) || last < first) {
                    throw reject("unsupported character class");
                }
                for (int c = first; c <= last; ++c) {
                    addIgnoringCase(atom.bytes, static_cast<char>(c));
                }
                i += isRange ? 3 : 1;
            }
            if (i == term.size() || atom.bytes.none()) {
                throw reject("unterminated or empty character class");
            }
            ++i;
        } else {
            throw reject("unsupported regular expression syntax");
        }
        if (i < term.size() && term[i] == '{') {
            const std::size_t close = term.find('}', i);
            const std::string quantifier = close == std::string::npos ? std::string() : term.substr(i + 1, close - i - 1);
            const std::size_t comma = quantifier.find(',');
            const std::string low = quantifier.substr(0, comma);
            const std::string high = comma == std::string::npos ? low : quantifier.substr(comma + 1);
            const auto isNumber = [](const std::string& text) {
                return !text.empty() && std::all_of(text.begin(), text.end(), isAsciiDigit);
            };
            // A count with more than nine digits would overflow std::stoul on some platforms
            if (!isNumber(low) || !(high.empty() || isNumber(high)) || low.size() > 9 || high.size() > 9) {
                throw reject("unsupported quantifier");
            }
            atom.minCount = std::stoul(low);
            atom.maxCount = high.empty() ? std::numeric_limits<std::size_t>::max() : std::stoul(high);
            if (atom.maxCount < atom.minCount) {
                throw reject("quantifier with a maximum below its minimum");
            }
            i = close + 1;
        }
        pattern.push_back(atom);
    }
    if (std::all_of(pattern.begin(), pattern.end(), [](const Atom& atom) { return atom.minCount == 0; })) {
        throw reject("matches the empty string");
    }
    return pattern;
}

// Longest match of 'pattern' at 'position', or 0: the leftmost-longest rule of POSIX, which equals the
// greedy match of re for the patterns of constants.py
std::size_t HeuristicSplitter::longestMatch(const Pattern& pattern, const std::string& text, std::size_t position) {
    const Atom& first = pattern.front();
    if (first.minCount > 0 && (position == text.size() || !first.bytes[static_cast<unsigned char>(text[position])])) {
        return 0;
    }
    std::vector<std::size_t> ends{position};
    for (const Atom& atom : pattern) {
        std::vector<std::size_t> next;
        for (const std::size_t start : ends) {
            std::size_t end = start;
            for (std::size_t count = 0;; ++count, ++end) {
                if (count >= atom.minCount) {
                    next.push_back(end);
                }
                if (count == atom.maxCount || end == text.size() || !atom.bytes[static_cast<unsigned char>(text[end])]) {
                    break;
                }
            }
        }
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        if (next.empty()) {
            return 0;
        }
        ends.swap(next);
    }
    return ends.back() - position;
}

std::size_t HeuristicSplitter::longestTermAt(const std::string& text, std::size_t position) const {
    std::size_t longest = 0;
    std::size_t node = 0;
    for (std::size_t i = position; i < text.size(); ++i) {
        const int symbol = trieSymbol(text[i]);
        if (symbol < 0 || literalTrie[node].next[symbol] < 0) {
            break;
        }
        node = static_cast<std::size_t>(literalTrie[node].next[symbol]);
        if (literalTrie[node].isTerm) {
            longest = i + 1 - position;
        }
    }
    for (const Pattern& pattern : patterns) {
        longest = std::max(longest, longestMatch(pattern, text, position));
    }
    return longest;
}

std::vector<std::string> HeuristicSplitter::split(const std::string& identifier) const {
    std::string translated = identifier;
    for (char& c : translated) {
        if (delimiterChars.find(c) != std::string_view::npos) {
            c = ' ';
        }
    }

    // re.sub(_exceptions_re, r' \1 ', ...) sets each term with numbers apart by spaces
    std::string separated;
    separated.reserve(translated.size() + 8);
    for (std::size_t position = 0; position < translated.size();) {
        const std::size_t length = longestTermAt(translated, position);
        if (length == 0) {
            separated += translated[position];
            ++position;
            continue;
        }
        separated += ' ';
        separated.append(translated, position, length);
        separated += ' ';
        position += length;
    }

    // re.sub(_camel_case, r' \1', ...) puts a space before each uppercase letter that follows a lowercase one
    std::string camel;
    camel.reserve(separated.size() + 8);
    for (std::size_t i = 0; i < separated.size(); ++i) {
        if (i > 0 && isAsciiLower(separated[i - 1]) && isAsciiUpper(separated[i])) {
            camel += ' ';
        }
        camel += separated[i];
    }

    std::vector<std::string> tokens;
    std::size_t position = 0;
    while (position < camel.size()) {
        if (isPythonSpace(camel[position])) {
            ++position;
            continue;
        }
        std::size_t end = position;
        while (end < camel.size() && !isPythonSpace(camel[end])) {
            ++end;
        }
        const std::string token = camel.substr(position, end - position);
        position = end;
        const bool keepsSuffixNumber = !isAsciiDigit(token.front()) &&
            std::any_of(suffixNumbers.begin(), suffixNumbers.end(), [&token](const std::string& suffix) { return endsWith(token, suffix); });
        if (longestTermAt(token, 0) > 0 || keepsSuffixNumber) {
            tokens.push_back(token);
        } else {
            appendNumberSplit(token, tokens);
        }
    }
    return tokens;
}

}
