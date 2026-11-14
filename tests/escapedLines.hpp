// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file escapedLines.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Line format of the golden files and of roninTool: fields split by tabs, with \\ and \xHH escapes.

#ifndef SCALAR_ESCAPED_LINES_HPP
#define SCALAR_ESCAPED_LINES_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace scalar {

// Escapes a backslash as \\ and each byte below 0x20, 0x7F, and each byte of 0x80 or more as \xHH
inline std::string escapeField(const std::string& text) {
    static const char hexDigits[] = "0123456789abcdef";
    std::string escaped;
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '\\') {
            escaped += "\\\\";
        } else if (byte < 0x20 || byte >= 0x7f) {
            escaped += "\\x";
            escaped += hexDigits[byte >> 4];
            escaped += hexDigits[byte & 0x0f];
        } else {
            escaped += c;
        }
    }
    return escaped;
}

// Fields of an escaped line, split at tabs and decoded; throws std::invalid_argument for a bad escape
inline std::vector<std::string> unescapeFields(const std::string& line) {
    const auto hexValue = [](char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    };
    std::vector<std::string> fields(1);
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\t') {
            fields.emplace_back();
        } else if (c != '\\') {
            fields.back() += c;
        } else if (i + 1 < line.size() && line[i + 1] == '\\') {
            fields.back() += '\\';
            i += 1;
        } else if (i + 3 < line.size() && line[i + 1] == 'x' && hexValue(line[i + 2]) >= 0 && hexValue(line[i + 3]) >= 0) {
            fields.back() += static_cast<char>(hexValue(line[i + 2]) * 16 + hexValue(line[i + 3]));
            i += 3;
        } else {
            throw std::invalid_argument("bad escape in the line " + line);
        }
    }
    return fields;
}

}

#endif
