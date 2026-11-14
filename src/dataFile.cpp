// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file dataFile.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the data files of the tagger from the binary of a program or from disk.

#include "dataFile.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <system_error>

namespace scalar {

namespace {

std::map<std::string, std::string_view, std::less<>>& builtInTable() {
    static std::map<std::string, std::string_view, std::less<>> table;
    return table;
}

// The name of a built-in file, or no name for a path on disk. std::filesystem::path joins a folder
//  and a file with a backslash on Windows, which therefore counts as a slash.
std::optional<std::string> builtInName(const std::string& path) {
    if (!isBuiltInPath(path)) return std::nullopt;
    std::string name = path.substr(builtInPrefix.size());
    std::replace(name.begin(), name.end(), '\\', '/');
    return name;
}

} // namespace

bool isBuiltInPath(const std::string& path) {
    return path.compare(0, builtInPrefix.size(), builtInPrefix) == 0;
}

void addBuiltInFiles(const BuiltInFile* files, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) builtInTable()[files[i].name] = files[i].bytes;
}

std::string_view DataFile::bytes() const {
    return builtIn ? inPlace : std::string_view(copy);
}

std::optional<DataFile> openDataFile(const std::string& path) {
    const std::optional<std::string> name = builtInName(path);
    if (!name) return readFileOnDisk(path);
    const auto found = builtInTable().find(*name);
    if (found == builtInTable().end()) return std::nullopt;
    DataFile file;
    file.builtIn = true;
    file.inPlace = found->second;
    return file;
}

std::optional<DataFile> readFileOnDisk(const std::string& path) {
    // An ifstream opens a folder on Linux without an error and then reads no bytes
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) return std::nullopt;
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    DataFile file;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (!error) {
        file.copy.resize(static_cast<std::size_t>(size));
        input.read(file.copy.data(), static_cast<std::streamsize>(size));
        file.copy.resize(static_cast<std::size_t>(input.gcount()));
        return file;
    }
    // A pipe, such as the <(...) of bash, has no size
    std::ostringstream content;
    content << input.rdbuf();
    file.copy = content.str();
    return file;
}

bool dataFileExists(const std::string& path) {
    if (const std::optional<std::string> name = builtInName(path)) return builtInTable().count(*name) > 0;
    std::error_code error;
    return std::filesystem::exists(path, error);
}

} // namespace scalar
