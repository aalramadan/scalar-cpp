// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file jsonReader.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the JSON files of a SCALAR model folder, such as tokenizer.json and config.json

#include "jsonReader.hpp"

#include "dataFile.hpp"

#include <cstdint>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace scalar {

namespace {

// Deep enough for every model file; a limit keeps a malformed file from exhausting the stack
constexpr int maxDepth = 256;

class JsonParser {
public:
    explicit JsonParser(std::string_view document) : text(document) {}

    JsonValue parseDocument() {
        JsonValue value = parseValue(0);
        skipSpace();
        if (position != text.size()) fail("unexpected text after the document");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& reason) const {
        throw std::runtime_error("JSON error at byte " + std::to_string(position) + ": " + reason);
    }

    void skipSpace() {
        while (position < text.size() &&
               (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r')) {
            ++position;
        }
    }

    void expectLiteral(std::string_view literal) {
        if (text.compare(position, literal.size(), literal) != 0) fail("invalid literal");
        position += literal.size();
    }

    JsonValue parseValue(int depth) {
        if (depth > maxDepth) fail("nesting too deep");
        skipSpace();
        if (position >= text.size()) fail("unexpected end of the document");
        JsonValue value;
        const char c = text[position];
        if (c == '{') {
            value.kind = JsonValue::Kind::Object;
            parseObject(value, depth);
        }
        else if (c == '[') {
            value.kind = JsonValue::Kind::Array;
            parseArray(value, depth);
        }
        else if (c == '"') {
            value.kind = JsonValue::Kind::String;
            value.text = parseString();
        }
        else if (c == 't') {
            expectLiteral("true");
            value.kind    = JsonValue::Kind::Boolean;
            value.boolean = true;
        }
        else if (c == 'f') {
            expectLiteral("false");
            value.kind = JsonValue::Kind::Boolean;
        }
        else if (c == 'n') {
            expectLiteral("null");
        }
        else {
            value.kind   = JsonValue::Kind::Number;
            value.number = parseNumber();
        }
        return value;
    }

    void parseObject(JsonValue& object, int depth) {
        ++position;
        skipSpace();
        if (position < text.size() && text[position] == '}') {
            ++position;
            return;
        }
        while (true) {
            skipSpace();
            if (position >= text.size() || text[position] != '"') fail("expected a member name");
            object.keys.push_back(parseString());
            skipSpace();
            if (position >= text.size() || text[position] != ':') fail("expected ':' after a member name");
            ++position;
            object.items.push_back(parseValue(depth + 1));
            skipSpace();
            if (position < text.size() && text[position] == ',') {
                ++position;
                continue;
            }
            if (position < text.size() && text[position] == '}') {
                ++position;
                return;
            }
            fail("expected ',' or '}' in an object");
        }
    }

    void parseArray(JsonValue& array, int depth) {
        ++position;
        skipSpace();
        if (position < text.size() && text[position] == ']') {
            ++position;
            return;
        }
        while (true) {
            array.items.push_back(parseValue(depth + 1));
            skipSpace();
            if (position < text.size() && text[position] == ',') {
                ++position;
                continue;
            }
            if (position < text.size() && text[position] == ']') {
                ++position;
                return;
            }
            fail("expected ',' or ']' in an array");
        }
    }

    std::uint32_t parseHexQuad() {
        if (position + 4 > text.size()) fail("truncated \\u escape");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text[position++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return value;
    }

    static void appendUtf8(std::string& out, std::uint32_t codePoint) {
        if (codePoint < 0x80) {
            out += static_cast<char>(codePoint);
        }
        else if (codePoint < 0x800) {
            out += static_cast<char>(0xC0 | (codePoint >> 6));
            out += static_cast<char>(0x80 | (codePoint & 0x3F));
        }
        else if (codePoint < 0x10000) {
            out += static_cast<char>(0xE0 | (codePoint >> 12));
            out += static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (codePoint & 0x3F));
        }
        else {
            out += static_cast<char>(0xF0 | (codePoint >> 18));
            out += static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (codePoint & 0x3F));
        }
    }

    std::string parseString() {
        ++position;
        std::string out;
        while (true) {
            if (position >= text.size()) fail("unterminated string");
            const char c = text[position++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in a string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (position >= text.size()) fail("unterminated escape");
            const char escape = text[position++];
            switch (escape) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    std::uint32_t codePoint = parseHexQuad();
                    if (codePoint >= 0xD800 && codePoint <= 0xDBFF) {
                        if (text.compare(position, 2, "\\u") != 0) fail("unpaired surrogate");
                        position += 2;
                        const std::uint32_t low = parseHexQuad();
                        if (low < 0xDC00 || low > 0xDFFF) fail("unpaired surrogate");
                        codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                    }
                    else if (codePoint >= 0xDC00 && codePoint <= 0xDFFF) {
                        fail("unpaired surrogate");
                    }
                    appendUtf8(out, codePoint);
                    break;
                }
                default:
                    fail("invalid escape");
            }
        }
    }

    double parseNumber() {
        const std::size_t start = position;
        if (position < text.size() && text[position] == '-') ++position;
        const auto skipDigits = [this] {
            const std::size_t first = position;
            while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
            if (position == first) fail("expected a digit");
        };
        skipDigits();
        if (position < text.size() && text[position] == '.') {
            ++position;
            skipDigits();
        }
        if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
            ++position;
            if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position;
            skipDigits();
        }
        // The classic locale reads the decimal point of JSON whatever the locale of the program, and a
        //  stream works on every C++ library, while std::from_chars of double is missing from some
        double value = 0.0;
        std::istringstream number(std::string(text.substr(start, position - start)));
        number.imbue(std::locale::classic());
        number >> value;
        if (number.fail() || number.peek() != std::char_traits<char>::eof()) fail("invalid number");
        return value;
    }

    std::string_view text;
    std::size_t      position = 0;
};

} // namespace

const JsonValue* JsonValue::find(std::string_view key) const {
    if (kind != Kind::Object) return nullptr;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key) return &items[i];
    }
    return nullptr;
}

JsonValue parseJson(std::string_view document) {
    return JsonParser(document).parseDocument();
}

JsonValue readJsonFile(const std::string& path) {
    const std::optional<DataFile> file = openDataFile(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    try {
        return parseJson(file->bytes());
    }
    catch (const std::runtime_error& error) {
        throw std::runtime_error(path + ": " + error.what());
    }
}

} // namespace scalar
