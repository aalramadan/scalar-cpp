// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file dataFile.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the data files of the tagger. A path that starts with "builtin:" names a file that the build
// put into the binary of the program, while any other path names a file on disk.

#ifndef SCALAR_DATA_FILE_HPP
#define SCALAR_DATA_FILE_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace scalar {

// The start of the path of a built-in file, such as "builtin:model/config.json"
inline constexpr std::string_view builtInPrefix = "builtin:";

// A file that the build put into the binary: its name after the prefix and its bytes
struct BuiltInFile {
    const char*      name;
    std::string_view bytes;
};

// Makes the files readable under their names. The bytes stay where they are, therefore they must
//  outlive every read, as the data of the binary does.
void addBuiltInFiles(const BuiltInFile* files, std::size_t count);

// The bytes of a data file: a built-in file in place, a file on disk as a copy of the whole file
class DataFile {
public:
    std::string_view bytes() const;

private:
    friend std::optional<DataFile> openDataFile(const std::string& path);
    friend std::optional<DataFile> readFileOnDisk(const std::string& path);

    bool             builtIn = false;
    std::string_view inPlace;
    std::string      copy;
};

// True for a path that names a built-in file, such as "builtin:model/config.json"
bool isBuiltInPath(const std::string& path);

// The file at the path, or no file when the path names no built-in file and no readable file on disk
std::optional<DataFile> openDataFile(const std::string& path);

// The file on disk at the path, also for a path that starts with "builtin:", or no file for a folder
//  or a file that cannot be read
std::optional<DataFile> readFileOnDisk(const std::string& path);

// True when the path names a built-in file or an existing file on disk
bool dataFileExists(const std::string& path);

} // namespace scalar

#endif
