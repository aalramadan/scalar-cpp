// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file asciiText.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Character classes, case mapping, and stripping that Python's str methods and re apply to 7-bit
// ASCII text, shared by the splitter and the encoder. A byte of 0x80 or above is no letter, digit,
// or space.

#ifndef SCALAR_ASCII_TEXT_HPP
#define SCALAR_ASCII_TEXT_HPP

#include <string>
#include <string_view>

namespace scalar {

// [a-z] of re and the letters that str.upper() changes
inline bool isAsciiLower(char c) {
    return c >= 'a' && c <= 'z';
}

// [A-Z] of re and the letters that str.lower() changes
inline bool isAsciiUpper(char c) {
    return c >= 'A' && c <= 'Z';
}

inline bool isAsciiLetter(char c) {
    return isAsciiLower(c) || isAsciiUpper(c);
}

// str.isdigit() and \d of re on ASCII text
inline bool isAsciiDigit(char c) {
    return c >= '0' && c <= '9';
}

// str.isspace() of one character, the bytes at which str.split() breaks: \t \n \v \f \r,
//   the separators 0x1C to 0x1F, and space
inline bool isPythonSpace(char c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f');
}

// Throws std::invalid_argument when the text holds a byte above 0x7F, since the encoder
//   follows the Python original only on ASCII; role names the text in the message
void requireAscii(std::string_view text, std::string_view role);

// str.strip() without arguments
std::string_view pythonStrip(std::string_view text);

// str.lower() and str.upper(), which change only ASCII letters here
std::string toLowerAscii(std::string_view text);
std::string toUpperAscii(std::string_view text);

// str.isalpha(), which is false for empty text
bool isAlphaText(std::string_view text);

} // namespace scalar

#endif
