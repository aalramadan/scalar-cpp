// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file goldenData.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Parses the tab-separated reference rows, the CRF cases, and the kernel cases of the golden files.

#include "goldenData.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace scalar {

namespace {

[[noreturn]] void failParse(const std::string& path, std::size_t line, const std::string& reason) {
    throw std::runtime_error(path + ":" + std::to_string(line) + ": " + reason);
}

std::vector<std::string> splitTabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t tab = line.find('\t', start);
        fields.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) {
            return fields;
        }
        start = tab + 1;
    }
}

template <typename Value>
std::vector<Value> parseList(const std::string& text) {
    std::vector<Value> values;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        if constexpr (std::is_same_v<Value, float>) {
            values.push_back(std::strtof(token.c_str(), nullptr));
        } else {
            values.push_back(static_cast<Value>(std::stoll(token)));
        }
    }
    return values;
}

std::vector<float> parseBits(std::istringstream& stream, std::size_t count) {
    std::vector<float> values(count);
    for (float& value : values) {
        std::string token;
        stream >> token;
        const auto bits = static_cast<std::uint32_t>(std::stoul(token, nullptr, 16));
        std::memcpy(&value, &bits, sizeof value);
    }
    return values;
}

}

std::vector<ReferenceRow> readReferenceRows(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot read the reference file '" + path + "'");
    }
    std::vector<ReferenceRow> rows;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = splitTabs(line);
        if (fields.size() != 9) {
            failParse(path, lineNumber, "a row needs 9 tab-separated fields");
        }
        ReferenceRow row;
        row.row     = static_cast<std::size_t>(std::stoull(fields[0]));
        row.name    = fields[1];
        row.context = fields[2];
        row.tooLong = fields[3] == "too-long";
        if (!row.tooLong) {
            if (fields[3] != "ok") {
                failParse(path, lineNumber, "unknown status '" + fields[3] + "'");
            }
            row.input.inputIds      = parseList<std::int32_t>(fields[4]);
            row.input.wordPositions = parseList<std::size_t>(fields[5]);
            row.labels              = parseList<int>(fields[6]);
            row.emissions           = parseList<float>(fields[7]);
            row.margin = fields[8] == "nan" ? std::numeric_limits<double>::quiet_NaN() : std::stod(fields[8]);
            if (row.labels.size() != row.input.wordPositions.size()) {
                failParse(path, lineNumber, "the labels do not match the word positions");
            }
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::vector<CrfCaseSet> readCrfCases(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot read the CRF cases '" + path + "'");
    }
    std::vector<CrfCaseSet> sets;
    std::size_t tags = 0;
    std::vector<float> start, end;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream stream(line);
        std::string kind;
        if (!(stream >> kind) || kind[0] == '#') {
            continue;
        }
        std::string rest;
        std::getline(stream, rest);
        if (kind == "crf") {
            tags = static_cast<std::size_t>(std::stoul(rest));
        } else if (kind == "start") {
            start = parseList<float>(rest);
        } else if (kind == "end") {
            end = parseList<float>(rest);
        } else if (kind == "transitions") {
            sets.push_back({CrfDecoder(tags, start, end, parseList<float>(rest)), {}});
        } else if (kind == "case") {
            if (sets.empty()) {
                failParse(path, lineNumber, "a case precedes its CRF");
            }
            const std::size_t split = rest.find(" path ");
            std::vector<float> values = parseList<float>(rest.substr(0, split));
            CrfCase entry;
            entry.length = static_cast<std::size_t>(values.front());
            entry.emissions.assign(values.begin() + 1, values.end());
            entry.path = parseList<int>(rest.substr(split + 6));
            if (entry.emissions.size() != entry.length * tags || entry.path.size() != entry.length) {
                failParse(path, lineNumber, "a case does not fit its length");
            }
            sets.back().cases.push_back(std::move(entry));
        } else {
            failParse(path, lineNumber, "unknown line '" + kind + "'");
        }
    }
    return sets;
}

std::vector<KernelCase> readKernelCases(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot read the kernel cases '" + path + "'");
    }
    std::vector<KernelCase> cases;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream stream(line);
        std::string kind;
        if (!(stream >> kind) || kind[0] == '#') {
            continue;
        }
        if (kind == "kernel") {
            KernelCase entry;
            stream >> entry.name;
            std::size_t size = 0;
            while (stream >> size) {
                entry.sizes.push_back(size);
            }
            cases.push_back(std::move(entry));
        } else if (kind == "array") {
            if (cases.empty()) {
                failParse(path, lineNumber, "an array precedes its kernel");
            }
            std::size_t count = 0;
            stream >> count;
            cases.back().arrays.push_back(parseBits(stream, count));
        } else {
            failParse(path, lineNumber, "unknown line '" + kind + "'");
        }
    }
    return cases;
}

}
