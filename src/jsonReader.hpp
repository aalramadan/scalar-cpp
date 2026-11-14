// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file jsonReader.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the JSON files of a SCALAR model folder, such as tokenizer.json and config.json

#ifndef SCALAR_JSON_READER_HPP
#define SCALAR_JSON_READER_HPP

#include <string>
#include <string_view>
#include <vector>

namespace scalar {

// One value of a JSON document. An object keeps its members in file order.
struct JsonValue {
    enum class Kind { Null, Boolean, Number, String, Array, Object };

    Kind                     kind    = Kind::Null;
    bool                     boolean = false;
    double                   number  = 0.0;
    std::string              text;   // the value of a string
    std::vector<JsonValue>   items;  // the elements of an array, or the member values of an object
    std::vector<std::string> keys;   // the member names of an object, parallel to items

    // The member of an object with the given name, or nullptr for another kind or a missing name
    const JsonValue* find(std::string_view key) const;
};

// Parses a JSON document. Throws std::runtime_error on malformed text.
JsonValue parseJson(std::string_view document);

// Reads and parses a JSON file. Throws std::runtime_error when the file cannot be read or parsed.
JsonValue readJsonFile(const std::string& path);

} // namespace scalar

#endif
