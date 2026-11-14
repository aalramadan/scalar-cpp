// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file asciiText.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Python str methods on ASCII text, and the check that keeps the input of the encoder ASCII

#include "asciiText.hpp"

#include <algorithm>
#include <stdexcept>

namespace scalar {

void requireAscii(std::string_view text, std::string_view role) {
    const bool ascii = std::all_of(text.begin(), text.end(), [](char c) { return static_cast<unsigned char>(c) <= 0x7F; });
    if (!ascii) {
        throw std::invalid_argument("SCALAR encoder: the " + std::string(role) + " holds a byte above 0x7F");
    }
}

std::string_view pythonStrip(std::string_view text) {
    std::size_t first = 0;
    std::size_t last = text.size();
    while (first < last && isPythonSpace(text[first])) ++first;
    while (last > first && isPythonSpace(text[last - 1])) --last;
    return text.substr(first, last - first);
}

std::string toLowerAscii(std::string_view text) {
    std::string lower(text);
    for (char& c : lower) {
        if (isAsciiUpper(c)) c = static_cast<char>(c - 'A' + 'a');
    }
    return lower;
}

std::string toUpperAscii(std::string_view text) {
    std::string upper(text);
    for (char& c : upper) {
        if (isAsciiLower(c)) c = static_cast<char>(c - 'a' + 'A');
    }
    return upper;
}

bool isAlphaText(std::string_view text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), isAsciiLetter);
}

} // namespace scalar
