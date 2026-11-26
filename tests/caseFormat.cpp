// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file caseFormat.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The case format that tests/scalar_reference.py writes, and the comparison of a case with the C++ encoding

#include "caseFormat.hpp"

#include "featureTokens.hpp"

#include <charconv>
#include <stdexcept>

namespace scalarTests {

namespace {

template <typename Number>
std::vector<Number> parseNumbers(std::string_view field) {
    std::vector<Number> numbers;
    std::size_t start = 0;
    while (start < field.size()) {
        std::size_t end = field.find(' ', start);
        if (end == std::string_view::npos) end = field.size();
        Number value{};
        const auto [stop, error] = std::from_chars(field.data() + start, field.data() + end, value);
        if (error != std::errc() || stop != field.data() + end) throw std::runtime_error("malformed number list");
        numbers.push_back(value);
        start = end + 1;
    }
    return numbers;
}

template <typename Number>
std::string joinNumbers(const std::vector<Number>& numbers) {
    std::string joined;
    for (const Number number : numbers) {
        if (!joined.empty()) joined += ' ';
        joined += std::to_string(number);
    }
    return joined;
}

std::string describeInputs(const std::string& featureSet, const scalar::IdentifierRow& row, const std::vector<std::string>& words) {
    std::string text = "features=" + featureSet + " context='" + escapeField(row.context) + "' type='" + escapeField(row.type) +
                       "' language='" + escapeField(row.language) + "' system='" + escapeField(row.systemName) + "' words=[";
    for (std::size_t i = 0; i < words.size(); ++i) text += (i ? ", '" : "'") + escapeField(words[i]) + "'";
    return text + "]";
}

std::string describe(const Outcome& outcome) {
    if (outcome.tooLong) return "too long";
    return "ids=[" + joinNumbers(outcome.input.inputIds) + "] positions=[" + joinNumbers(outcome.input.wordPositions) + "]";
}

} // namespace

std::vector<std::string_view> splitFields(std::string_view line) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t tab = line.find('\t', start);
        fields.push_back(line.substr(start, tab == std::string_view::npos ? std::string_view::npos : tab - start));
        if (tab == std::string_view::npos) return fields;
        start = tab + 1;
    }
}

std::string unescapeField(std::string_view field) {
    std::string text;
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (field[i] != '\\') {
            text += field[i];
            continue;
        }
        if (i + 1 >= field.size()) throw std::runtime_error("dangling escape");
        const char escape = field[++i];
        if (escape == '\\') text += '\\';
        else if (escape == 't') text += '\t';
        else if (escape == 'n') text += '\n';
        else if (escape == 'x' && i + 2 < field.size()) {
            unsigned value = 0;
            const auto [stop, error] = std::from_chars(field.data() + i + 1, field.data() + i + 3, value, 16);
            if (error != std::errc() || stop != field.data() + i + 3) throw std::runtime_error("malformed \\x escape");
            text += static_cast<char>(value);
            i += 2;
        }
        else throw std::runtime_error("unknown escape");
    }
    return text;
}

std::string escapeField(std::string_view text) {
    static const char hexDigits[] = "0123456789abcdef";
    std::string escaped;
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '\\') escaped += "\\\\";
        else if (c == '\t') escaped += "\\t";
        else if (c == '\n') escaped += "\\n";
        else if (byte < 0x20 || byte >= 0x7F) {
            escaped += "\\x";
            escaped += hexDigits[byte >> 4];
            escaped += hexDigits[byte & 0xF];
        }
        else escaped += c;
    }
    return escaped;
}

ReferenceCase parseCase(std::string_view line) {
    const std::vector<std::string_view> fields = splitFields(line);
    if (fields.size() < 8) throw std::runtime_error("a case line has fewer than eight fields");
    ReferenceCase reference;
    reference.group          = unescapeField(fields[0]);
    reference.featureSet     = std::string(fields[1]);
    reference.row.context    = unescapeField(fields[2]);
    reference.row.type       = unescapeField(fields[3]);
    reference.row.language   = unescapeField(fields[4]);
    reference.row.systemName = unescapeField(fields[5]);
    reference.tooLong        = fields[6] == "!";
    if (!reference.tooLong) {
        reference.ids       = parseNumbers<std::int32_t>(fields[6]);
        reference.positions = parseNumbers<std::size_t>(fields[7]);
    }
    for (std::size_t i = 8; i < fields.size(); ++i) reference.words.push_back(unescapeField(fields[i]));
    return reference;
}

const std::vector<std::string>& featureList(const std::string& featureSet) {
    static const std::vector<std::string> legacy = scalar::normalizeSelectedFeatures(scalar::legacyFeatures());
    static const std::vector<std::string> defaults = scalar::normalizeSelectedFeatures(std::nullopt);
    if (featureSet == "legacy") return legacy;
    if (featureSet == "default") return defaults;
    throw std::runtime_error("unknown feature set " + featureSet);
}

std::string contextOfLetter(char letter) {
    for (const char* context : {"FUNCTION", "CLASS", "ATTRIBUTE", "DECLARATION", "PARAMETER"}) {
        if (context[0] == letter) return context;
    }
    throw std::runtime_error(std::string("unknown context letter ") + letter);
}

Outcome encodeCase(const scalar::WordPieceTokenizer& tokenizer, const scalar::IdentifierRow& row,
                   const std::vector<std::string>& words, const std::string& featureSet) {
    Outcome outcome;
    try {
        outcome.input = scalar::encodeIdentifier(tokenizer, row, words, featureList(featureSet));
    }
    catch (const scalar::IdentifierTooLongError&) {
        outcome.tooLong = true;
    }
    return outcome;
}

bool sameOutcome(const ReferenceCase& expected, const Outcome& actual) {
    if (expected.tooLong || actual.tooLong) return expected.tooLong == actual.tooLong;
    return expected.ids == actual.input.inputIds && expected.positions == actual.input.wordPositions;
}

std::uint32_t digestOf(const Outcome& outcome) {
    std::vector<std::uint32_t> values;
    if (outcome.tooLong) {
        values.push_back(0xFFFFFFFEu);
    }
    else {
        values.assign(outcome.input.inputIds.begin(), outcome.input.inputIds.end());
        values.push_back(0xFFFFFFFFu);
        for (const std::size_t position : outcome.input.wordPositions) values.push_back(static_cast<std::uint32_t>(position));
    }
    std::uint32_t hash = 0x811C9DC5u;
    for (const std::uint32_t value : values) {
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 0xFFu;
            hash *= 0x01000193u;
        }
    }
    return hash;
}

std::string describeMismatch(const ReferenceCase& expected, const Outcome& actual) {
    Outcome python;
    python.tooLong              = expected.tooLong;
    python.input.inputIds       = expected.ids;
    python.input.wordPositions  = expected.positions;
    return "MISMATCH " + expected.group + " " + describeInputs(expected.featureSet, expected.row, expected.words) +
           "\n  python: " + describe(python) + "\n  c++:    " + describe(actual);
}

std::string describeOutcome(const std::string& featureSet, const scalar::IdentifierRow& row, const std::vector<std::string>& words,
                            const Outcome& outcome) {
    return describeInputs(featureSet, row, words) + "\n  c++: " + describe(outcome);
}

} // namespace scalarTests
